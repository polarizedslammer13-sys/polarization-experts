#!/usr/bin/env python3
"""
Train-set expansion sweep: does feeding more (drift-affected) faces help?

For each N in {1600, 3200, 6400, 12000}:
    train_idx = [0, N) face   on R-channel, 4-pol early-stack
    val_idx   = [N, N+200)    used for early stopping
    eval on 5 fixed tail slabs (all >= face 12200, never in any train set)

Same architecture / loss / optimizer / 45-epoch cap / patience-4 as
run_early_stack.py.  Records both per-slab PCC and the mean across the
5 tail slabs for each N.

Outputs:
    /root/autodl-tmp/facedataset_0825/expand_sweep_<ts>/
        N{n}_best_model.pth
        N{n}_results.json
        sweep_summary.json     final 4xN_slab table + mean curve
        sweep_curve.png
"""
import os, sys, json, time, math, random, argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse early-stack pieces directly
sys.path.insert(0, "/root/polarization-experts")
from run_early_stack import (
    UNetEarlyStack, EarlyStackDataset, AdvancedLoss, SSIMLoss, WarmupCosine,
)


# =====================================================================
# Config
# =====================================================================
SPF = "/root/autodl-tmp/facedataset_0825/original/speckles.npy"
PAT = "/root/autodl-tmp/facedataset_0825/original/pattern.npy"

N_VALUES = [1600, 3200, 6400, 12000]
COLOR_CHANNEL = 2   # R
EPOCHS = 45
BATCH = 8
LR = 2e-4
WD = 1e-5
WARMUP = 10
PATIENCE = 4
SEED = 42

VAL_SIZE = 200          # face count just after train end
EVAL_SLAB_SIZE = 200    # face count per eval slab
EVAL_SLAB_STARTS = [12200, 13500, 14800, 16100, 16700]   # all face >= 12200


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval()
    ssim_fn = SSIMLoss()
    tp, ts, tm, n = 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for x, gt in loader:
            x = x.to(device); gt = gt.to(device)
            if gt.dim() == 3: gt = gt.unsqueeze(1)
            pred = model(x)
            p64 = F.adaptive_avg_pool2d(pred, (64, 64))
            g64 = F.adaptive_avg_pool2d(gt, (64, 64))
            tm += F.mse_loss(p64, g64).item() * x.size(0)
            ts += (1 - ssim_fn(p64, g64)).item() * x.size(0)
            for i in range(p64.shape[0]):
                r, _ = pearsonr(p64[i, 0].cpu().numpy().flatten(),
                                g64[i, 0].cpu().numpy().flatten())
                if not np.isnan(r):
                    tp += r; n += 1
    N = len(loader.dataset)
    return {"pcc": tp / max(n, 1), "ssim": ts / N, "mse": tm / N, "n": n}


def train_one_N(N, out_dir, device):
    print(f"\n{'='*70}\n  N = {N} faces  (R channel, 4-pol early stack)\n{'='*70}")
    set_seed(SEED)
    train_idx = list(range(0, N))
    val_idx = list(range(N, N + VAL_SIZE))

    trl = DataLoader(EarlyStackDataset(SPF, PAT, train_idx, COLOR_CHANNEL),
                     batch_size=BATCH, shuffle=True, num_workers=6, pin_memory=True)
    vll = DataLoader(EarlyStackDataset(SPF, PAT, val_idx, COLOR_CHANNEL),
                     batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)

    model = UNetEarlyStack(in_channels=4, base=48).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD, betas=(0.9, 0.999))
    sched = WarmupCosine(opt, warmup=WARMUP, total=EPOCHS)
    loss_fn = AdvancedLoss(device)

    best_pcc = -1.0; best_weights = None; best_epoch = 0; patience = 0
    val_curve = []
    t0 = time.time()

    for ep in range(EPOCHS):
        lr = sched.step(ep)
        model.train()
        for x, gt in tqdm(trl, desc=f"N{N} Ep{ep+1:2d}", leave=False):
            x = x.to(device); gt = gt.to(device)
            if gt.dim() == 3: gt = gt.unsqueeze(1)
            opt.zero_grad()
            loss, _ = loss_fn(model(x), gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # eval every 2 epochs (matches run_early_stack)
        ep_num = ep + 1
        if ep_num % 2 == 0 or ep_num == EPOCHS:
            vm = evaluate(model, vll, device)
            val_curve.append({"epoch": ep_num, **vm})
            improved = vm["pcc"] > best_pcc
            if improved:
                best_pcc = vm["pcc"]
                best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = ep_num
                patience = 0
            else:
                patience += 1
            flag = " <" if improved else ""
            print(f"  Ep {ep_num:3d}/{EPOCHS} | Val PCC={vm['pcc']:.4f} "
                  f"(best={best_pcc:.4f}) | SSIM={vm['ssim']:.4f} | LR={lr:.6f}{flag}")
            if patience >= PATIENCE:
                print(f"  early stop at ep {ep_num}")
                break

    if best_weights:
        model.load_state_dict(best_weights)
    train_time = time.time() - t0

    # save model
    ckpt_path = os.path.join(out_dir, f"N{N}_best_model.pth")
    torch.save({
        "model_state_dict": best_weights,
        "N": N, "best_epoch": best_epoch, "best_val_pcc": best_pcc,
        "color_channel": COLOR_CHANNEL, "fusion_strategy": "early_stack",
    }, ckpt_path)
    print(f"  saved -> {ckpt_path}")

    # eval all 5 tail slabs
    print(f"  eval on tail slabs (face start: {EVAL_SLAB_STARTS}):")
    tail_results = {}
    for s in EVAL_SLAB_STARTS:
        idx = list(range(s, s + EVAL_SLAB_SIZE))
        ds = EarlyStackDataset(SPF, PAT, idx, COLOR_CHANNEL)
        loader = DataLoader(ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
        m = evaluate(model, loader, device)
        tail_results[s] = m
        print(f"    face [{s:5d},{s+EVAL_SLAB_SIZE:5d})  PCC={m['pcc']:.4f}  SSIM={m['ssim']:.4f}")

    mean_tail_pcc = float(np.mean([r["pcc"] for r in tail_results.values()]))
    result = {
        "N": N,
        "train_time_sec": train_time,
        "best_epoch": best_epoch,
        "best_val_pcc": float(best_pcc),
        "val_curve": val_curve,
        "tail_slabs": {str(k): v for k, v in tail_results.items()},
        "mean_tail_pcc": mean_tail_pcc,
    }
    with open(os.path.join(out_dir, f"N{N}_results.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  N={N} done. mean tail PCC = {mean_tail_pcc:.4f}  ({train_time/60:.1f} min)")
    return result


def make_plot(results, out_dir):
    Ns = [r["N"] for r in results]
    means = [r["mean_tail_pcc"] for r in results]
    per_slab = {s: [r["tail_slabs"][str(s)]["pcc"] for r in results]
                for s in EVAL_SLAB_STARTS}

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for s, ys in per_slab.items():
        ax.plot(Ns, ys, "o--", lw=0.8, ms=4, alpha=0.6, label=f"face [{s}..{s+EVAL_SLAB_SIZE})")
    ax.plot(Ns, means, "o-", lw=2.5, ms=9, color="#C62828", label="mean of 5 tail slabs")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns])
    ax.set_xlabel("training set size (face count)")
    ax.set_ylabel("test PCC (tail slabs)")
    ax.set_title("Train-set expansion sweep on full 50994-frame dataset\n"
                 "early-stack-R, 45 epochs cap, eval slabs all >= face 12200")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "sweep_curve.png"), dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="run only this N (otherwise all of {1600,3200,6400,12000})")
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/root/autodl-tmp/facedataset_0825/expand_sweep_{ts}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"out_dir = {out_dir}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device  = {device}")
    if torch.cuda.is_available():
        print(f"GPU     = {torch.cuda.get_device_name(0)}")

    Ns_to_run = [args.n] if args.n else N_VALUES
    results = []
    for N in Ns_to_run:
        r = train_one_N(N, out_dir, device)
        results.append(r)
        # checkpoint partial results after each N
        with open(os.path.join(out_dir, "sweep_summary.json"), "w") as f:
            json.dump({"timestamp": ts, "N_values": Ns_to_run,
                       "eval_slab_starts": EVAL_SLAB_STARTS,
                       "eval_slab_size": EVAL_SLAB_SIZE,
                       "results": results}, f, indent=2)

    if len(results) > 1:
        make_plot(results, out_dir)

    print(f"\n=== sweep done. summary -> {out_dir}/sweep_summary.json ===")
    print("\nSummary table:")
    print(f"  {'N':>6}  {'best_ep':>7}  {'val_pcc':>8}  {'mean_tail_pcc':>13}  {'train_min':>9}")
    for r in results:
        print(f"  {r['N']:6d}  {r['best_epoch']:7d}  {r['best_val_pcc']:8.4f}  "
              f"{r['mean_tail_pcc']:13.4f}  {r['train_time_sec']/60:9.1f}")


if __name__ == "__main__":
    main()

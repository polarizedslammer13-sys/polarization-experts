#!/usr/bin/env python3
"""
PTA-A: Polarisation Transformer Architecture, Direction A.
Stokes projection space reconstruction.

One (mode, seed) per invocation. Appends to results.json on completion.

Modes:
  A0       4-pol normalised,  Stokes 4x4 learnable  (init = theoretical Stokes)
  A1       4-pol normalised,  Stokes 4x4 frozen     (= theoretical Stokes)
  A2       4-pol normalised,  no projection         (early-stack control)
  A2-raw   4-pol raw,         no projection         (isolates normalisation)
  A3       1-pol raw (pol2),  no projection         (single-pol baseline)

Self-checks at startup:
  1. norm_stats.json train_indices_hash matches expected
  2. Stokes projection numerical init matches theoretical Stokes matrix
  3. Param count delta vs A3 baseline matches expectation
  4. GPU required (no silent CPU fallback)

Outputs:
  results.json    appended entry with metrics, curves, W_P trajectory,
                  per-sample test PCC
  viz/<tag>.npz   Stokes feature snapshots (A0, A1 only; seed=1 only)
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Dataset

# This script lives in pta_a/ but reuses UNetPro256 etc. from the parent
# baseline. Inject the parent dir on sys.path so the import resolves regardless
# of cwd.
_THIS_DIR   = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from run_exp2_baseline import UNetPro256, SSIMLoss, AdvancedLoss  # noqa: E402


# ---------------------------------------------------------------------
# Constants (must match run_exp2_baseline.py)
# ---------------------------------------------------------------------
BASE_DIR   = "/root/autodl-tmp/facedataset_0825"
ORIG_DIR   = os.path.join(BASE_DIR, "original")
SPECKLE_F  = os.path.join(ORIG_DIR, "speckles6000_og.npy")
PATTERN_F  = os.path.join(ORIG_DIR, "pattern.npy")

TOTAL_SAMPLES = 2000
TRAIN_IDX     = list(range(0, 1600))
VAL_IDX       = list(range(1600, 1800))
TEST_IDX      = list(range(1800, 2000))

COLOR_CHANNEL = 2          # R, BGR order
POL_CHANNELS  = [1, 2, 3, 4]   # 0°, 45°, 90°, 135°

EPOCHS  = 60
BATCH   = 4
LR      = 2e-4
PROJ_LR = 2e-3    # 10x for projection layer (16 params)
WD      = 1e-5
WARMUP  = 10

HERE         = _THIS_DIR
RESULTS_PATH = os.path.join(HERE, "results.json")
NORM_STATS   = os.path.join(HERE, "norm_stats.json")
VIZ_DIR      = os.path.join(HERE, "viz")

VIZ_INDICES  = [1846, 1851, 1877, 1904, 1920]    # 5 fixed viz samples (in TEST_IDX)
TRAIN_EVAL_SUBSET = 200                          # for train-PCC curve

VALID_MODES = ("A0", "A1", "A2", "A2-raw", "A3")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def hash_indices(indices, extra):
    h = hashlib.sha256()
    for i in indices:
        h.update(int(i).to_bytes(4, "little", signed=False))
    h.update(json.dumps(extra, sort_keys=True).encode())
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------
# Stokes projection
# ---------------------------------------------------------------------
def theoretical_stokes_matrix():
    # Rows = (S0, S1, S2, S_res)
    # Cols = (I0°, I45°, I90°, I135°)
    return np.array([
        [0.5, 0.5, 0.5, 0.5],   # S0 = avg of 4 intensities
        [1.0, 0.0, -1.0, 0.0],  # S1 = I0  - I90
        [0.0, 1.0, 0.0, -1.0],  # S2 = I45 - I135
        [0.0, 0.0, 0.0, 0.0],   # S_res (calibration residual, init 0)
    ], dtype=np.float32)


class StokesProjection(nn.Module):
    def __init__(self, learnable=True):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, kernel_size=1, bias=False)
        W = torch.from_numpy(theoretical_stokes_matrix())
        with torch.no_grad():
            self.conv.weight.copy_(W.unsqueeze(-1).unsqueeze(-1))
        if not learnable:
            for p in self.conv.parameters():
                p.requires_grad = False
        self.learnable = learnable

    def matrix(self):
        return self.conv.weight.detach().cpu().squeeze(-1).squeeze(-1).numpy().copy()

    def forward(self, x):
        return self.conv(x)


# ---------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------
class PTAAModel(nn.Module):
    def __init__(self, mode, base=48):
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"unknown mode: {mode}")
        self.mode = mode
        if mode == "A3":
            self.stokes_proj = None
            self.unet = UNetPro256(in_channels=1, base=base)
        elif mode in ("A2", "A2-raw"):
            self.stokes_proj = None
            self.unet = UNetPro256(in_channels=4, base=base)
        else:  # A0 / A1
            self.stokes_proj = StokesProjection(learnable=(mode == "A0"))
            self.unet = UNetPro256(in_channels=4, base=base)

    def forward(self, x):
        if self.stokes_proj is not None:
            x = self.stokes_proj(x)
        return self.unet(x)


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class StokesDataset(Dataset):
    def __init__(self, speckles_path, patterns_path, indices, mode, calib=None,
                 color_channel=COLOR_CHANNEL):
        self.sp = np.load(speckles_path, mmap_mode="r")
        self.pat = np.load(patterns_path, mmap_mode="r")
        self.idx = list(indices)
        self.mode = mode
        self.cc = color_channel
        if mode in ("A0", "A1", "A2"):
            assert calib is not None, f"mode {mode} requires calib"
            self.calib = np.array(calib, dtype=np.float32).reshape(4, 1, 1)
        else:
            self.calib = None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        oi = self.idx[i]
        si = oi * 3 + self.cc
        if self.mode == "A3":
            sp = self.sp[si, 2].astype(np.float32) / 255.0   # pol2 (45°)
            x = torch.from_numpy(sp).unsqueeze(0).float()
        else:
            sp = self.sp[si, 1:5].astype(np.float32) / 255.0  # pol1..pol4
            if self.calib is not None:
                sp = sp * self.calib
            x = torch.from_numpy(np.ascontiguousarray(sp)).float()
        pat = self.pat[si].astype(np.float32) / 255.0
        pat = cv2.resize(pat, (256, 256), interpolation=cv2.INTER_LINEAR)
        gt = torch.from_numpy(pat).float()
        return x, gt


# ---------------------------------------------------------------------
# Multi-group warmup + cosine scheduler
# ---------------------------------------------------------------------
class WarmupCosineSchedulerMulti:
    def __init__(self, optimizer, warmup, total, eta_min=1e-6):
        self.opt = optimizer
        self.warmup = warmup
        self.total = total
        self.eta_min = eta_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch):
        if epoch < self.warmup:
            factor = (epoch + 1) / self.warmup
            lrs = [b * factor for b in self.base_lrs]
        else:
            progress = (epoch - self.warmup) / max(1, (self.total - self.warmup))
            cosf = 0.5 * (1 + math.cos(math.pi * progress))
            lrs = [self.eta_min + (b - self.eta_min) * cosf for b in self.base_lrs]
        for pg, lr in zip(self.opt.param_groups, lrs):
            pg["lr"] = lr
        return lrs


def build_optimizer(model):
    proj_params = []
    other_params = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (proj_params if n.startswith("stokes_proj.") else other_params).append(p)
    groups = []
    if proj_params:
        groups.append({"params": proj_params, "lr": PROJ_LR, "weight_decay": WD})
    groups.append({"params": other_params, "lr": LR, "weight_decay": WD})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------
def evaluate(model, loader, device, want_per_sample=False):
    model.eval()
    total_pcc = 0.0
    total_ssim = 0.0
    total_mse = 0.0
    num_samples = 0
    per_sample = []
    ssim_fn = SSIMLoss()
    with torch.no_grad():
        for x, gt in loader:
            x = x.to(device)
            gt = gt.to(device)
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)
            pred = model(x)
            pred_64 = F.adaptive_avg_pool2d(pred, (64, 64))
            gt_64 = F.adaptive_avg_pool2d(gt, (64, 64))
            total_mse += F.mse_loss(pred_64, gt_64).item() * x.size(0)
            total_ssim += (1 - ssim_fn(pred_64, gt_64)).item() * x.size(0)
            for i in range(pred.shape[0]):
                p = pred_64[i, 0].cpu().numpy().flatten()
                g = gt_64[i, 0].cpu().numpy().flatten()
                try:
                    val, _ = pearsonr(p, g)
                    if not np.isnan(val):
                        total_pcc += val
                        num_samples += 1
                        if want_per_sample:
                            per_sample.append(float(val))
                except Exception:
                    pass
    out = {
        "pcc": total_pcc / max(num_samples, 1),
        "ssim": total_ssim / len(loader.dataset),
        "mse": total_mse / len(loader.dataset),
    }
    if want_per_sample:
        out["per_sample_pcc"] = per_sample
    return out


# ---------------------------------------------------------------------
# Self-checks
# ---------------------------------------------------------------------
def selfcheck_norm_stats(stats):
    extra = {"color_channel": COLOR_CHANNEL, "pol_channels": POL_CHANNELS}
    expected = hash_indices(TRAIN_IDX, extra)
    actual = stats.get("train_indices_hash", None)
    if actual != expected:
        raise RuntimeError(
            f"norm_stats hash mismatch:\n  expected={expected}\n  actual  ={actual}\n"
            "Re-run `python3 pta_a/norm_stats.py` before proceeding."
        )


def selfcheck_stokes_init(model):
    if model.stokes_proj is None:
        return None
    dev = next(model.parameters()).device
    x = torch.zeros(1, 4, 8, 8, device=dev)
    x[:, 0] = 1.0   # I0°
    x[:, 1] = 2.0   # I45°
    x[:, 2] = 3.0   # I90°
    x[:, 3] = 4.0   # I135°
    with torch.no_grad():
        y = model.stokes_proj(x)
    obs = [float(y[0, c].mean().item()) for c in range(4)]
    exp = [5.0, -2.0, -2.0, 0.0]
    for c, (o, e) in enumerate(zip(obs, exp)):
        assert abs(o - e) < 1e-5, f"Stokes init ch{c}: got {o:.6f}, expected {e}"
    return {"observed": obs, "expected": exp}


def count_params(model, trainable_only=False):
    total = 0
    for p in model.parameters():
        if trainable_only and not p.requires_grad:
            continue
        total += p.numel()
    return total


def selfcheck_param_delta(model, mode):
    expected = 0
    if mode != "A3":
        expected += 1728 - 432           # enc1 first conv: 4ch vs 1ch
    if mode in ("A0", "A1"):
        expected += 16                   # Stokes projection 4x4
    a3 = PTAAModel("A3", base=48)
    n_a3 = count_params(a3, trainable_only=False)
    n_self = count_params(model, trainable_only=False)
    delta = n_self - n_a3
    assert delta == expected, (
        f"param delta vs A3 mismatch: got {delta}, expected {expected}"
    )
    return {"params_a3_ref": n_a3, "params_self": n_self, "delta": delta}


# ---------------------------------------------------------------------
# Stokes feature snapshot (A0/A1 only)
# ---------------------------------------------------------------------
@torch.no_grad()
def snapshot_stokes_features(model, mode, calib, device, indices=VIZ_INDICES):
    if model.stokes_proj is None:
        return None
    model.eval()
    ds = StokesDataset(SPECKLE_F, PATTERN_F, indices, mode, calib=calib)
    inputs, outputs = [], []
    for i in range(len(ds)):
        x, _ = ds[i]
        x = x.unsqueeze(0).to(device)
        y = model.stokes_proj(x)
        inputs.append(x.squeeze(0).cpu().numpy())
        outputs.append(y.squeeze(0).cpu().numpy())
    return np.stack(inputs, 0), np.stack(outputs, 0)


# ---------------------------------------------------------------------
# Results IO
# ---------------------------------------------------------------------
def already_done(mode, seed):
    if not os.path.isfile(RESULTS_PATH):
        return False
    try:
        with open(RESULTS_PATH) as f:
            r = json.load(f)
    except Exception:
        return False
    for e in r:
        c = e.get("config", {})
        if c.get("mode") == mode and c.get("seed") == seed and e.get("completed"):
            return True
    return False


def append_result(entry):
    existing = []
    if os.path.isfile(RESULTS_PATH):
        try:
            with open(RESULTS_PATH) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(entry)
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2)
    os.replace(tmp, RESULTS_PATH)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=list(VALID_MODES))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true",
                        help="rerun even if (mode, seed) already in results")
    args = parser.parse_args()

    if not args.force and already_done(args.mode, args.seed):
        print(f"[SKIP] mode={args.mode} seed={args.seed} already in results")
        return

    # ---------------- Norm stats ----------------
    norm = None
    if args.mode in ("A0", "A1", "A2"):
        if not os.path.isfile(NORM_STATS):
            raise FileNotFoundError(
                f"{NORM_STATS} missing — run `python3 pta_a/norm_stats.py` first."
            )
        with open(NORM_STATS) as f:
            norm = json.load(f)
        selfcheck_norm_stats(norm)
    calib = norm["calib"] if norm else None

    # ---------------- Seed + device ----------------
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "GPU not available — PTA-A requires GPU. "
            "Refusing silent CPU run to avoid contaminating the comparison."
        )

    print("=" * 72)
    print(f"PTA-A | mode={args.mode} | seed={args.seed}")
    print("=" * 72)
    print(f"device      : {device} ({torch.cuda.get_device_name(0)})")
    print(f"calib       : {calib}")
    print(f"pol channels: {POL_CHANNELS}  (=> 0°, 45°, 90°, 135°)")
    print(f"color       : {COLOR_CHANNEL} (R)")

    # ---------------- Data loaders ----------------
    train_ds = StokesDataset(SPECKLE_F, PATTERN_F, TRAIN_IDX, args.mode, calib=calib)
    val_ds   = StokesDataset(SPECKLE_F, PATTERN_F, VAL_IDX,   args.mode, calib=calib)
    test_ds  = StokesDataset(SPECKLE_F, PATTERN_F, TEST_IDX,  args.mode, calib=calib)
    train_eval_ds = StokesDataset(
        SPECKLE_F, PATTERN_F, TRAIN_IDX[:TRAIN_EVAL_SUBSET], args.mode, calib=calib
    )
    train_loader      = DataLoader(train_ds,      batch_size=BATCH, shuffle=True,
                                   num_workers=6, pin_memory=True)
    val_loader        = DataLoader(val_ds,        batch_size=8,     shuffle=False,
                                   num_workers=6, pin_memory=True)
    test_loader       = DataLoader(test_ds,       batch_size=8,     shuffle=False,
                                   num_workers=6, pin_memory=True)
    train_eval_loader = DataLoader(train_eval_ds, batch_size=8,     shuffle=False,
                                   num_workers=4, pin_memory=True)

    # ---------------- Model + self-checks ----------------
    model = PTAAModel(args.mode, base=48).to(device)
    init_check = selfcheck_stokes_init(model)
    param_check = selfcheck_param_delta(model, args.mode)
    n_total = count_params(model, trainable_only=False)
    n_train = count_params(model, trainable_only=True)
    if init_check is not None:
        print(f"  [selfcheck] Stokes init observed={[f'{v:.4f}' for v in init_check['observed']]} "
              f"expected={init_check['expected']}  OK")
    print(f"  [selfcheck] param delta vs A3 = {param_check['delta']} (matches expected)")
    print(f"  params total    : {n_total/1e6:.3f}M")
    print(f"  params trainable: {n_train/1e6:.3f}M")

    # ---------------- Optimiser + scheduler ----------------
    optimizer = build_optimizer(model)
    scheduler = WarmupCosineSchedulerMulti(optimizer, WARMUP, EPOCHS)
    print(f"  base lrs        : {scheduler.base_lrs}")

    loss_fn = AdvancedLoss(device)
    if not getattr(loss_fn, "use_perceptual", False):
        raise RuntimeError(
            "VGG19 perceptual loss failed to load — refusing to run with "
            "degraded 0.4/0.4/0.2 fallback (would contaminate cross-mode comparison)."
        )

    # ---------------- W_P and viz snapshot at init ----------------
    wp_trajectory = []
    if model.stokes_proj is not None:
        wp_trajectory.append(model.stokes_proj.matrix().tolist())
    save_viz = (args.mode in ("A0", "A1")) and (args.seed == 1)
    snap_init = None
    if save_viz:
        os.makedirs(VIZ_DIR, exist_ok=True)
        snap_init = snapshot_stokes_features(model, args.mode, calib, device)

    # ---------------- Training ----------------
    val_pcc_curve, val_ssim_curve = [], []
    train_loss_curve, train_pcc_curve = [], []
    best_pcc, best_epoch = -1.0, -1
    t_train_total = 0.0
    t_start = time.time()

    for epoch in range(EPOCHS):
        t_ep = time.time()
        lrs = scheduler.step(epoch)
        model.train()
        total_loss, batches = 0.0, 0
        for x, gt in train_loader:
            x = x.to(device)
            gt = gt.to(device)
            if gt.dim() == 3:
                gt = gt.unsqueeze(1)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        train_loss = total_loss / max(batches, 1)
        val_metrics   = evaluate(model, val_loader,        device)
        train_metrics = evaluate(model, train_eval_loader, device)
        if val_metrics["pcc"] > best_pcc:
            best_pcc, best_epoch = val_metrics["pcc"], epoch + 1
        ep_time = time.time() - t_ep
        t_train_total += ep_time

        val_pcc_curve.append(val_metrics["pcc"])
        val_ssim_curve.append(val_metrics["ssim"])
        train_loss_curve.append(train_loss)
        train_pcc_curve.append(train_metrics["pcc"])
        if model.stokes_proj is not None:
            wp_trajectory.append(model.stokes_proj.matrix().tolist())

        if (epoch % 5 == 0) or (epoch == EPOCHS - 1):
            lr_str = "/".join(f"{lr:.2e}" for lr in lrs)
            print(
                f"  E{epoch+1:3d}/{EPOCHS} lr={lr_str} "
                f"loss={train_loss:.4f} tPCC={train_metrics['pcc']:.4f} "
                f"vPCC={val_metrics['pcc']:.4f}(best={best_pcc:.4f}@{best_epoch}) "
                f"vSSIM={val_metrics['ssim']:.4f} t={ep_time:.1f}s"
            )

    # ---------------- Test + per-sample PCC ----------------
    t_test_start = time.time()
    test_metrics = evaluate(model, test_loader, device, want_per_sample=True)
    t_test = time.time() - t_test_start

    # ---------------- Viz at final ----------------
    if save_viz:
        snap_final = snapshot_stokes_features(model, args.mode, calib, device)
        viz_path = os.path.join(VIZ_DIR, f"{args.mode}_seed{args.seed}.npz")
        np.savez_compressed(
            viz_path,
            indices=np.array(VIZ_INDICES),
            input=snap_init[0],          # input is deterministic; one copy is enough
            output_init=snap_init[1],
            output_final=snap_final[1],
        )
        print(f"  [viz] Stokes feature snapshot: {viz_path}")

    # ---------------- Assemble + append result ----------------
    cfg_extra = {"color_channel": COLOR_CHANNEL, "pol_channels": POL_CHANNELS}
    entry = {
        "config": {
            "mode": args.mode,
            "seed": args.seed,
            "epochs": EPOCHS,
            "batch": BATCH,
            "base_lr": LR,
            "proj_lr": PROJ_LR,
            "weight_decay": WD,
            "warmup": WARMUP,
            "color_channel": COLOR_CHANNEL,
            "pol_channels": POL_CHANNELS,
            "train_indices_hash": hash_indices(TRAIN_IDX, cfg_extra),
            "params_total": n_total,
            "params_trainable": n_train,
        },
        "best_val_pcc": float(best_pcc),
        "best_val_epoch": int(best_epoch),
        "test_pcc":  float(test_metrics["pcc"]),
        "test_ssim": float(test_metrics["ssim"]),
        "test_mse":  float(test_metrics["mse"]),
        "per_sample_test_pcc": test_metrics["per_sample_pcc"],
        "val_pcc_curve":   val_pcc_curve,
        "val_ssim_curve":  val_ssim_curve,
        "train_loss_curve": train_loss_curve,
        "train_pcc_curve":  train_pcc_curve,
        "wp_trajectory":    wp_trajectory if wp_trajectory else None,
        "wp_theoretical":   (theoretical_stokes_matrix().tolist()
                             if model.stokes_proj is not None else None),
        "train_time_sec": t_train_total,
        "test_time_sec":  t_test,
        "total_time_sec": time.time() - t_start,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "completed": True,
    }
    append_result(entry)

    print("-" * 72)
    print(f"Best Val PCC : {best_pcc:.4f} @ epoch {best_epoch}")
    print(f"Test  PCC    : {test_metrics['pcc']:.4f}")
    print(f"Test  SSIM   : {test_metrics['ssim']:.4f}")
    print(f"Test  MSE    : {test_metrics['mse']:.6f}")
    print(f"Train  time  : {t_train_total/60:.2f} min ({t_train_total:.1f} s)")
    print(f"Wrote        : {RESULTS_PATH}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted")
        raise
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

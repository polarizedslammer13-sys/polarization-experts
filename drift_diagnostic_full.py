#!/usr/bin/env python3
"""
6000 -> 50994 frame drift diagnostic on the full dataset.

Frame layout: raw[i] for face_idx = i // 3, color = i % 3 (0=B, 1=G, 2=R).
This script analyses R-channel only (color offset = 2), pol channel 2 (45 deg).

Diag 1  Speckle self-similarity drift heatmap
        - 16998 R-channel face-samples grouped into B=100 bins
        - within-bin mean speckle, then pairwise corr -> 100x100 heatmap
        - sanity: also do the same on all-color (50994 raw), B=100 bins

Diag 2  Cross-time generalisation with early-stack R checkpoint
        - checkpoint trained on face_idx [0..1599] R-channel
        - 10 disjoint 200-face test slabs spread across face_idx [0..16997]
        - report PCC per slab

Outputs:
    /root/autodl-tmp/facedataset_0825/drift_diag_full/
        drift_heatmap_R_pol2.png       100x100 R-only drift
        drift_heatmap_allcolor.png     100x100 all-color drift
        cross_time_pcc.png             PCC vs face-slab
        summary.json                   numerical results
"""
import os, sys, json, time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse early-stack model + dataset directly
sys.path.insert(0, "/root/polarization-experts")
from run_early_stack import UNetEarlyStack, EarlyStackDataset, SSIMLoss


SPF = "/root/autodl-tmp/facedataset_0825/original/speckles.npy"
PAT = "/root/autodl-tmp/facedataset_0825/original/pattern.npy"
CKPT = "/root/autodl-tmp/facedataset_0825/early_stack_R_20260315_173549/best_model.pth"

OUT_DIR = "/root/autodl-tmp/facedataset_0825/drift_diag_full"
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================================
# Diagnostic 1: drift heatmap
# =====================================================================
def drift_heatmap_R_only(n_bins=100, pol=2):
    """16998 R-only frames -> bin-mean speckle -> corr matrix."""
    print(f"\n[Diag 1a] R-only drift heatmap, pol={pol}, bins={n_bins}")
    t0 = time.time()
    sp = np.load(SPF, mmap_mode="r")    # (50994, 5, 256, 256) uint8
    n_face = sp.shape[0] // 3            # 16998
    bin_size = n_face // n_bins          # ~170 faces per bin
    print(f"  n_face={n_face}, bin_size={bin_size}")

    bin_mean = np.zeros((n_bins, 256, 256), dtype=np.float32)
    for b in range(n_bins):
        a = b * bin_size
        z = (b + 1) * bin_size if b < n_bins - 1 else n_face
        # convert face indices -> raw frame indices (R-channel)
        raw_ids = np.arange(a, z) * 3 + 2
        block = sp[raw_ids, pol].astype(np.float32)  # (bin_size, 256, 256)
        bin_mean[b] = block.mean(axis=0)
        if b % 10 == 0:
            print(f"    bin {b:3d}/{n_bins}  faces [{a:5d},{z:5d})  elapsed {time.time()-t0:.1f}s")

    # corr matrix: each bin mean -> flatten -> z-score -> dot
    flat = bin_mean.reshape(n_bins, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    flat = flat / (flat.std(axis=1, keepdims=True) + 1e-8)
    corr = (flat @ flat.T) / flat.shape[1]

    np.save(os.path.join(OUT_DIR, "corr_R_pol2.npy"), corr)
    np.save(os.path.join(OUT_DIR, "bin_mean_R_pol2.npy"), bin_mean)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(corr, cmap="viridis", vmin=corr.min(), vmax=1.0,
                         origin="upper")
    axes[0].set_title(f"R-channel drift heatmap (pol={pol}, {n_bins} bins of "
                      f"{bin_size} faces)\n"
                      f"x/y = bin index ~ face index / {bin_size}")
    axes[0].set_xlabel("bin index")
    axes[0].set_ylabel("bin index")
    plt.colorbar(im, ax=axes[0], label="pearson corr (bin-mean speckle)")
    # axes [0] also draw bin 9 (faces ~1530..1700, near train end) anchor line
    axes[0].axhline(9, color="red", lw=0.5, alpha=0.5)
    axes[0].axvline(9, color="red", lw=0.5, alpha=0.5)

    # drift curve: correlation with bin 0
    axes[1].plot(corr[0], lw=1.5, label="vs bin 0  (faces 0..170)")
    axes[1].plot(corr[9], lw=1.5, label="vs bin 9  (faces 1530..1700, train end)")
    axes[1].plot(corr[10], lw=1.0, alpha=0.7, label="vs bin 10 (faces 1700..1870)")
    axes[1].set_xlabel("bin index")
    axes[1].set_ylabel("pearson corr")
    axes[1].set_title("drift curves vs anchors")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(min(0.0, corr.min() - 0.05), 1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "drift_heatmap_R_pol2.png"), dpi=120)
    plt.close()
    print(f"  done in {time.time()-t0:.1f}s")
    return {"n_bins": n_bins, "bin_size": bin_size,
            "corr_min": float(corr.min()),
            "corr_bin0_vs_bin99": float(corr[0, n_bins - 1]),
            "corr_bin0_vs_bin9":  float(corr[0, 9]),
            "corr_bin9_vs_bin99": float(corr[9, n_bins - 1])}


def drift_heatmap_all_color(n_bins=100, pol=2):
    """50994 all-color frames -> bin-mean -> corr. Reveals BGR structure."""
    print(f"\n[Diag 1b] all-color drift heatmap, pol={pol}, bins={n_bins}")
    t0 = time.time()
    sp = np.load(SPF, mmap_mode="r")
    n_tot = sp.shape[0]                 # 50994
    bin_size = n_tot // n_bins          # ~510 raw frames per bin

    bin_mean = np.zeros((n_bins, 256, 256), dtype=np.float32)
    for b in range(n_bins):
        a = b * bin_size
        z = (b + 1) * bin_size if b < n_bins - 1 else n_tot
        block = sp[a:z, pol].astype(np.float32)
        bin_mean[b] = block.mean(axis=0)
        if b % 10 == 0:
            print(f"    bin {b:3d}/{n_bins}  raw [{a:5d},{z:5d})  elapsed {time.time()-t0:.1f}s")

    flat = bin_mean.reshape(n_bins, -1)
    flat = flat - flat.mean(axis=1, keepdims=True)
    flat = flat / (flat.std(axis=1, keepdims=True) + 1e-8)
    corr = (flat @ flat.T) / flat.shape[1]
    np.save(os.path.join(OUT_DIR, "corr_allcolor_pol2.npy"), corr)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="viridis", vmin=corr.min(), vmax=1.0,
                   origin="upper")
    ax.set_title(f"All-color drift heatmap (raw frames, pol={pol}, {n_bins} bins)\n"
                 f"each bin = {bin_size} raw frames (mixed BGR)")
    ax.set_xlabel("bin index")
    ax.set_ylabel("bin index")
    plt.colorbar(im, ax=ax, label="pearson corr")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "drift_heatmap_allcolor.png"), dpi=120)
    plt.close()
    print(f"  done in {time.time()-t0:.1f}s")
    return {"n_bins": n_bins, "bin_size": bin_size,
            "corr_min": float(corr.min()),
            "corr_bin0_vs_bin99": float(corr[0, n_bins - 1])}


# =====================================================================
# Diagnostic 2: cross-time inference
# =====================================================================
def _evaluate(model, loader, device):
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


def cross_time_inference(slab_size=200, n_slabs=10):
    """Run early-stack R checkpoint on slabs spread across face 0..16997."""
    print(f"\n[Diag 2] cross-time inference, {n_slabs} slabs of {slab_size} faces")
    t0 = time.time()
    n_face = 16998
    # First slab fixed at [1800..1999] for parity with the original test set.
    # Other slabs distributed evenly across [2000, n_face).
    rest = np.linspace(2000, n_face - slab_size, n_slabs - 1, dtype=int)
    starts = np.concatenate([[1800], rest])
    slabs = [(int(s), int(s) + slab_size) for s in starts]
    print(f"  slabs (face idx, 200 each): {slabs}")

    # Load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")
    model = UNetEarlyStack(in_channels=4, base=48).to(device)
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    print(f"  loaded checkpoint: {CKPT}")

    results = []
    for (a, z) in slabs:
        idx = list(range(a, z))
        ds = EarlyStackDataset(SPF, PAT, idx, color_channel=2)
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
        m = _evaluate(model, loader, device)
        # midpoint as a proxy time coordinate
        results.append({
            "face_start": a, "face_end": z,
            "raw_start": a * 3 + 2, "raw_end": (z - 1) * 3 + 2,
            **m,
        })
        print(f"  faces [{a:5d},{z:5d})  PCC={m['pcc']:.4f}  SSIM={m['ssim']:.4f}  "
              f"MSE={m['mse']:.5f}  elapsed {time.time()-t0:.1f}s")

    # plot
    xs = [r["face_start"] for r in results]
    ys = [r["pcc"] for r in results]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys, "o-", color="#1565C0", lw=1.5, ms=7)
    ax.axhline(ys[0], color="red", lw=0.6, ls="--", alpha=0.5,
               label=f"slab 0 (face 1800-1999) = {ys[0]:.4f}")
    ax.axvline(1600, color="gray", lw=0.6, ls=":", alpha=0.5)
    ax.text(1610, min(ys) + 0.005, "train end", fontsize=8, color="gray")
    ax.set_xlabel("face index (slab start)")
    ax.set_ylabel("test PCC")
    ax.set_title("Cross-time PCC of early-stack-R checkpoint\n"
                 "(trained on face [0..1599] R-channel)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cross_time_pcc.png"), dpi=120)
    plt.close()
    print(f"  done in {time.time()-t0:.1f}s")
    return results


# =====================================================================
def main():
    print(f"=== drift diagnostic, start {datetime.now().isoformat()} ===")
    print(f"out -> {OUT_DIR}")

    out = {"timestamp": datetime.now().isoformat()}
    out["diag1_R_only"]     = drift_heatmap_R_only(n_bins=100, pol=2)
    out["diag1_all_color"]  = drift_heatmap_all_color(n_bins=100, pol=2)
    out["diag2_cross_time"] = cross_time_inference(slab_size=200, n_slabs=10)

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n=== done.  summary -> {OUT_DIR}/summary.json ===")


if __name__ == "__main__":
    main()

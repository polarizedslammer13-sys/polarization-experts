#!/usr/bin/env python3
"""Side-by-side comparison: same 6 tail-slab faces, N=1600 vs N=12000 predictions."""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import pearsonr

sys.path.insert(0, "/root/polarization-experts")
from run_early_stack import UNetEarlyStack, EarlyStackDataset, SSIMLoss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE  = "/root/autodl-tmp/facedataset_0825/expand_sweep_20260517_030248"
CKPTS = [("N=1600",  f"{BASE}/N1600_best_model.pth"),
         ("N=12000", f"{BASE}/N12000_best_model.pth")]
SPF   = "/root/autodl-tmp/facedataset_0825/original/speckles.npy"
PAT   = "/root/autodl-tmp/facedataset_0825/original/pattern.npy"
COLOR = 2
OUT   = f"{BASE}/sample_viz_compare_N1600_vs_N12000.png"

FACES = [12300, 13550, 13650, 14900, 16200, 16800]
SLAB  = {12300:"slab1", 13550:"slab2", 13650:"slab2",
         14900:"slab3", 16200:"slab4", 16800:"slab5"}

device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ssim_fn = SSIMLoss().to(device)

ds     = EarlyStackDataset(SPF, PAT, FACES, COLOR)
loader = DataLoader(ds, batch_size=1, shuffle=False)

gts = []
inputs = []
with torch.no_grad():
    for x, gt in loader:
        x  = x.to(device); gt = gt.to(device)
        if gt.dim() == 3: gt = gt.unsqueeze(1)
        inputs.append(x); gts.append(gt)

# Run each checkpoint on all faces
all_results = {}    # name -> list of dicts
for name, ckpt_path in CKPTS:
    model = UNetEarlyStack(in_channels=4, base=48).to(device)
    ck    = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"loaded {name}  best_epoch={ck['best_epoch']}  best_val_pcc={ck['best_val_pcc']:.4f}")
    rows = []
    with torch.no_grad():
        for i, x in enumerate(inputs):
            pred = model(x)
            gt   = gts[i]
            p64  = F.adaptive_avg_pool2d(pred, (64, 64))
            g64  = F.adaptive_avg_pool2d(gt,   (64, 64))
            pcc  = pearsonr(p64[0,0].cpu().numpy().flatten(),
                            g64[0,0].cpu().numpy().flatten())[0]
            ssim = 1.0 - ssim_fn(p64, g64).item()
            mse  = F.mse_loss(p64, g64).item()
            rows.append({
                "face": FACES[i], "slab": SLAB[FACES[i]],
                "pred": pred[0,0].cpu().numpy(),
                "pcc": float(pcc), "ssim": float(ssim), "mse": float(mse),
            })
    all_results[name] = rows

# print side-by-side summary
print()
print(f"{'face':>6}  {'slab':>5}  |  "
      f"{'N=1600 PCC':>10}  {'SSIM':>6}  |  "
      f"{'N=12000 PCC':>11}  {'SSIM':>6}  |  "
      f"{'dPCC':>6}  {'dSSIM':>6}")
for i in range(len(FACES)):
    r0 = all_results["N=1600"][i]
    r1 = all_results["N=12000"][i]
    print(f"{r0['face']:>6}  {r0['slab']:>5}  |  "
          f"{r0['pcc']:10.4f}  {r0['ssim']:6.4f}  |  "
          f"{r1['pcc']:11.4f}  {r1['ssim']:6.4f}  |  "
          f"{r1['pcc']-r0['pcc']:+.4f}  {r1['ssim']-r0['ssim']:+.4f}")

# figure: 3 rows × 6 cols  GT | N=1600 | N=12000
n = len(FACES)
fig, axes = plt.subplots(3, n, figsize=(2.6*n, 8.2), facecolor="white")

for j in range(n):
    gt_np = gts[j][0,0].cpu().numpy()
    axes[0, j].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[0, j].set_title(f"GT  face {FACES[j]}\n({SLAB[FACES[j]]})", fontsize=9)
    axes[0, j].axis("off")

    r0 = all_results["N=1600"][j]
    axes[1, j].imshow(r0["pred"], cmap="gray", vmin=0, vmax=1)
    axes[1, j].set_title(f"N=1600  PCC={r0['pcc']:.3f}\nSSIM={r0['ssim']:.3f}",
                         fontsize=9, color="#8B0000")
    axes[1, j].axis("off")

    r1 = all_results["N=12000"][j]
    axes[2, j].imshow(r1["pred"], cmap="gray", vmin=0, vmax=1)
    axes[2, j].set_title(f"N=12000  PCC={r1['pcc']:.3f}\nSSIM={r1['ssim']:.3f}",
                         fontsize=9, color="#0B5A0B")
    axes[2, j].axis("off")

# annotate row labels on the far left
for ax, lbl, col in zip(axes[:,0],
                        ["GT", "N=1600 pred", "N=12000 pred"],
                        ["black", "#8B0000", "#0B5A0B"]):
    ax.text(-0.12, 0.5, lbl, transform=ax.transAxes,
            rotation=90, va="center", ha="center",
            fontsize=12, fontweight="bold", color=col)

plt.suptitle("Train-set expansion: same 6 tail-slab faces, N=1600 vs N=12000  "
             "(metrics at 64x64 pooled)", fontsize=12, y=1.0)
plt.tight_layout(rect=[0.015, 0, 1, 0.96])
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"\nsaved -> {OUT}")

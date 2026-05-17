#!/usr/bin/env python3
"""Visualize N=12000 sweep checkpoint on 6 tail-slab test faces (one per slab + 1 extra
from the lowest-PCC slab). Prints + saves per-sample PCC and SSIM (eval at 64x64 to
match training metrics)."""
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

CKPT  = "/root/autodl-tmp/facedataset_0825/expand_sweep_20260517_030248/N12000_best_model.pth"
SPF   = "/root/autodl-tmp/facedataset_0825/original/speckles.npy"
PAT   = "/root/autodl-tmp/facedataset_0825/original/pattern.npy"
COLOR = 2
OUT   = "/root/autodl-tmp/facedataset_0825/expand_sweep_20260517_030248/sample_viz_N12000.png"

# 6 faces — one center per tail slab + an extra from slab 2 (lowest PCC slab).
FACES = [12300, 13550, 13650, 14900, 16200, 16800]
SLAB  = {12300:"slab1 [12200,12400)",
         13550:"slab2 [13500,13700)",
         13650:"slab2 [13500,13700)",
         14900:"slab3 [14800,15000)",
         16200:"slab4 [16100,16300)",
         16800:"slab5 [16700,16900)"}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model  = UNetEarlyStack(in_channels=4, base=48).to(device)
ck     = torch.load(CKPT, map_location=device, weights_only=False)
model.load_state_dict(ck["model_state_dict"])
model.eval()
print(f"loaded N=12000 ckpt  (best_epoch={ck['best_epoch']}  best_val_pcc={ck['best_val_pcc']:.4f})")

ssim_fn = SSIMLoss().to(device)
ds = EarlyStackDataset(SPF, PAT, FACES, COLOR)
loader = DataLoader(ds, batch_size=1, shuffle=False)

rows = []
with torch.no_grad():
    for i, (x, gt) in enumerate(loader):
        x = x.to(device)
        gt = gt.to(device)
        if gt.dim() == 3: gt = gt.unsqueeze(1)
        pred = model(x)
        p64  = F.adaptive_avg_pool2d(pred, (64, 64))
        g64  = F.adaptive_avg_pool2d(gt,   (64, 64))
        pcc  = pearsonr(p64[0,0].cpu().numpy().flatten(),
                        g64[0,0].cpu().numpy().flatten())[0]
        ssim = 1.0 - ssim_fn(p64, g64).item()
        mse  = F.mse_loss(p64, g64).item()
        rows.append({
            "face": FACES[i], "slab": SLAB[FACES[i]],
            "pred": pred[0,0].cpu().numpy(),
            "gt"  : gt[0,0].cpu().numpy(),
            "pcc": float(pcc), "ssim": float(ssim), "mse": float(mse),
        })

# print summary
print()
print(f"{'face':>6}  {'slab':<22}  {'PCC':>7}  {'SSIM':>7}  {'MSE':>8}")
for r in rows:
    print(f"{r['face']:>6}  {r['slab']:<22}  {r['pcc']:7.4f}  {r['ssim']:7.4f}  {r['mse']:8.5f}")

# figure: 3 rows × 6 cols  GT | pred | |diff|
n = len(rows)
fig, axes = plt.subplots(3, n, figsize=(2.6*n, 8.0), facecolor="white")
for j, r in enumerate(rows):
    axes[0, j].imshow(r["gt"],   cmap="gray", vmin=0, vmax=1)
    axes[0, j].set_title(f"GT  face {r['face']}\n{r['slab']}", fontsize=8.5)
    axes[0, j].axis("off")

    axes[1, j].imshow(r["pred"], cmap="gray", vmin=0, vmax=1)
    axes[1, j].set_title(f"N=12000 pred\nPCC={r['pcc']:.4f}  SSIM={r['ssim']:.4f}", fontsize=8.5)
    axes[1, j].axis("off")

    diff = np.abs(r["pred"] - r["gt"])
    im = axes[2, j].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
    axes[2, j].set_title(f"|pred - GT|  max={diff.max():.3f}  mean={diff.mean():.3f}",
                         fontsize=8)
    axes[2, j].axis("off")

plt.suptitle("N=12000 sweep checkpoint — reconstructions on 6 tail-slab test faces\n"
             "(metrics PCC/SSIM/MSE computed at 64x64 pooled to match training eval)",
             fontsize=11, y=1.0)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT, dpi=130, bbox_inches="tight")
print(f"\nsaved -> {OUT}")

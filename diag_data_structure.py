#!/usr/bin/env python3
"""
Diagnose the structure of speckles6000_og.npy.

The question: is `oi * 3 + cc` encoding (sample_index * 3 + color_channel),
i.e. 6000 = 2000 samples * 3 colors at near-identical times?
Or are the 6000 frames sequential in time with cc just being a fixed offset?

Test:
  Adjacent frames (si=0, 1, 2) should be:
    - near-identical speckles + 3 different patterns
      => "3 colors at same time"  (oi-scheme correct, only 2000 temporal samples)
    - dissimilar speckles + dissimilar patterns
      => "sequential in time"     (6000 temporal samples)

Also test similarity between far-apart frames to give an upper bound.
"""
import os
import numpy as np

ORIG = "/root/autodl-tmp/facedataset_0825/original"
SP = np.load(os.path.join(ORIG, "speckles6000_og.npy"), mmap_mode="r")
PT = np.load(os.path.join(ORIG, "pattern.npy"), mmap_mode="r")

print(f"speckle shape: {SP.shape}  (frames, pol_channels, H, W)")
print(f"pattern shape: {PT.shape}")
print()


def corr(a, b):
    a = a.astype(np.float32).flatten()
    b = b.astype(np.float32).flatten()
    a = a - a.mean()
    b = b - b.mean()
    n = np.sqrt((a * a).sum() * (b * b).sum())
    if n < 1e-9:
        return 0.0
    return float((a * b).sum() / n)


print("== Test 1: adjacent frame triplet (si=0,1,2) ==")
print("If oi*3+cc encodes (sample, color):")
print("  - speckle(0) ~ speckle(1) ~ speckle(2)  (same fiber state)")
print("  - pattern(0), pattern(1), pattern(2)    (3 different color frames)")
print()

# Use pol=2 (45deg) across triplets
for triplet_start in [0, 99, 999, 5997]:
    s = [SP[triplet_start + k, 2] for k in range(3)]
    p = [PT[triplet_start + k] for k in range(3)]
    print(f"  triplet si={triplet_start}..{triplet_start+2}")
    print(f"    speckle PCC: (0,1)={corr(s[0],s[1]):.4f}  (1,2)={corr(s[1],s[2]):.4f}  (0,2)={corr(s[0],s[2]):.4f}")
    print(f"    pattern PCC: (0,1)={corr(p[0],p[1]):.4f}  (1,2)={corr(p[1],p[2]):.4f}  (0,2)={corr(p[0],p[2]):.4f}")
    means = [float(s[k].mean()) for k in range(3)]
    print(f"    speckle mean: {means}  (if uniform across colors -> mostly same illumination)")
print()


print("== Test 2: cross-triplet far comparison (si=2 vs si=5999) ==")
print("Sets baseline 'unrelated' PCC level")
a = SP[2, 2]; b = SP[5999, 2]
print(f"  speckle PCC(2, 5999) = {corr(a, b):.4f}")
print(f"  pattern PCC(2, 5999) = {corr(PT[2], PT[5999]):.4f}")
print()


print("== Test 3: drift heatmap across all 6000 frames (binned by 200) ==")
print("Mean speckle in each 200-frame bin (after subtracting global mean),")
print("then PCC matrix between bins. Reveals drift / segmentation structure.")
print()

# Mean speckle (pol=2, single color channel) per 200-frame block, raw frame order
# Then per-color-channel separately (R only: cc=2) to control for color interleaving
N = 6000
BIN = 200
n_bins = N // BIN

bin_means_all = np.zeros((n_bins, 256, 256), dtype=np.float32)
for b in range(n_bins):
    block = SP[b*BIN:(b+1)*BIN, 2].astype(np.float32)
    bin_means_all[b] = block.mean(axis=0)

# Subtract grand mean
gm = bin_means_all.mean(axis=0)
bin_dev = bin_means_all - gm

M = np.zeros((n_bins, n_bins), dtype=np.float32)
for i in range(n_bins):
    for j in range(n_bins):
        M[i, j] = corr(bin_dev[i], bin_dev[j])

np.save("/root/polarization-experts/diag1_drift_corr_allframes.npy", M)
print(f"  saved drift heatmap to diag1_drift_corr_allframes.npy  shape={M.shape}")
print(f"  M[0,0]={M[0,0]:.3f}  M[0,-1]={M[0,-1]:.3f}  M[0,15]={M[0,15]:.3f}")
print(f"  diag-1 mean: {np.diag(M, 1).mean():.3f}  (adjacent bin similarity)")
print(f"  off-diag (|i-j|>=15) mean: {M[np.abs(np.subtract.outer(np.arange(n_bins), np.arange(n_bins))) >= 15].mean():.3f}")
print()


print("== Test 4: same heatmap but using R-channel-only frames (cc=2 = si%3==2) ==")
print("These are the 2000 frames the user actually trains on.")
print("If color interleaving was confounding Test 3, this should look different.")
print()

R_idx = np.arange(2, 6000, 3)   # 2000 frames
assert len(R_idx) == 2000
RBIN = 100   # 100 per bin -> 20 bins
n_rbins = len(R_idx) // RBIN
bin_means_R = np.zeros((n_rbins, 256, 256), dtype=np.float32)
for b in range(n_rbins):
    chunk = R_idx[b*RBIN:(b+1)*RBIN]
    bin_means_R[b] = SP[chunk, 2].astype(np.float32).mean(axis=0)

gmR = bin_means_R.mean(axis=0)
bin_dev_R = bin_means_R - gmR

MR = np.zeros((n_rbins, n_rbins), dtype=np.float32)
for i in range(n_rbins):
    for j in range(n_rbins):
        MR[i, j] = corr(bin_dev_R[i], bin_dev_R[j])

np.save("/root/polarization-experts/diag1_drift_corr_Rchannel.npy", MR)
print(f"  saved R-only heatmap to diag1_drift_corr_Rchannel.npy  shape={MR.shape}")
print(f"  diag-1 mean (adjacent bin sim): {np.diag(MR, 1).mean():.3f}")
print(f"  off-diag (|i-j|>=5) mean: {MR[np.abs(np.subtract.outer(np.arange(n_rbins), np.arange(n_rbins))) >= 5].mean():.3f}")

#!/usr/bin/env python3
"""
PTA-A: compute per-polarisation mean intensity over the training split,
and the multiplicative calibration factors used by modes A0/A1/A2.

Output: norm_stats.json (in the same directory).

Run once before stokes.py. The training script validates the
train_indices_hash on launch and refuses to run if it doesn't match.
"""

import hashlib
import json
import os
from datetime import datetime

import numpy as np


BASE_DIR  = "/root/autodl-tmp/facedataset_0825"
ORIG_DIR  = os.path.join(BASE_DIR, "original")
SPECKLE_F = os.path.join(ORIG_DIR, "speckles6000_og.npy")

TOTAL_SAMPLES = 2000
TRAIN_IDX     = list(range(0, 1600))
COLOR_CHANNEL = 2          # R (BGR order, matches baseline)
POL_CHANNELS  = [1, 2, 3, 4]   # 0°, 45°, 90°, 135°   (pol0 = OG, excluded)

HERE     = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "norm_stats.json")
CHUNK    = 100


def hash_indices(indices, extra):
    h = hashlib.sha256()
    for i in indices:
        h.update(int(i).to_bytes(4, "little", signed=False))
    h.update(json.dumps(extra, sort_keys=True).encode())
    return "sha256:" + h.hexdigest()


def main():
    sp = np.load(SPECKLE_F, mmap_mode="r")
    assert sp.shape == (6000, 5, 256, 256), f"unexpected shape: {sp.shape}"
    assert sp.dtype == np.uint8, f"unexpected dtype: {sp.dtype}"

    sums = np.zeros(len(POL_CHANNELS), dtype=np.float64)
    n_pix = 0

    for start in range(0, len(TRAIN_IDX), CHUNK):
        inds = TRAIN_IDX[start:start + CHUNK]
        rows = [k * 3 + COLOR_CHANNEL for k in inds]
        block = sp[rows].astype(np.float32) / 255.0   # (chunk, 5, 256, 256)
        for j, p in enumerate(POL_CHANNELS):
            sums[j] += float(block[:, p].sum())
        n_pix += block.shape[0] * block.shape[2] * block.shape[3]

    mean_pc = (sums / n_pix).tolist()
    target_mean = float(np.mean(mean_pc))
    calib = [target_mean / m for m in mean_pc]

    extra = {"color_channel": COLOR_CHANNEL, "pol_channels": POL_CHANNELS}
    out = {
        "mean_pc": mean_pc,
        "target_mean": target_mean,
        "calib": calib,
        "train_indices_hash": hash_indices(TRAIN_IDX, extra),
        "n_train_samples": len(TRAIN_IDX),
        "color_channel": COLOR_CHANNEL,
        "pol_channels": POL_CHANNELS,
        "speckle_file": SPECKLE_F,
        "n_pixels_per_channel": n_pix,
        "computed_on": datetime.now().isoformat(timespec="seconds"),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 72)
    print("PTA-A norm stats")
    print("=" * 72)
    print(f"pol channels  : {POL_CHANNELS}  (=> 0°, 45°, 90°, 135°)")
    print(f"color channel : {COLOR_CHANNEL}  (R)")
    print(f"n train samples: {len(TRAIN_IDX)}")
    print(f"mean_pc       : {[f'{m:.5f}' for m in mean_pc]}")
    print(f"target_mean   : {target_mean:.5f}")
    print(f"calib         : {[f'{c:.5f}' for c in calib]}")
    print(f"train hash    : {out['train_indices_hash']}")
    print(f"Wrote         : {OUT_PATH}")


if __name__ == "__main__":
    main()

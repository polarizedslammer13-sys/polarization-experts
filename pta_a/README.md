# PTA-A — Stokes Projection Reconstruction

Self-contained experiment subdirectory for **Direction A** of the
polarization-imaging patent / paper differentiation work.

> **TL;DR** — Replace the "split-and-fuse" multi-channel-MoE paradigm with a
> *physical basis transform*: project 4 linear-polarised speckles through a
> learnable 4×4 matrix (initialised at the theoretical Stokes matrix) before
> sending them into the existing UNetPro256 backbone. Compare against a
> normalised early-stack, a raw early-stack, and the single-polarisation
> baseline to attribute every PCC delta to a concrete cause.

---

## 1. State at a glance

| Step                                       | State        |
| ------------------------------------------ | ------------ |
| `norm_stats.py` written                    | done         |
| `norm_stats.json` computed                 | done         |
| `stokes.py` (5-mode training script)       | done         |
| `run.sh` (tmux orchestrator, 25 runs)      | done         |
| `analyze.py` (summary + 4 plots)           | done         |
| Self-review against CHECKLIST.md           | **pending**  |
| tmux launch of 25 runs                     | **pending**  |
| `analyze.py` on completed `results.json`   | pending      |

The current working directory of every script is `pta_a/` (see `run.sh`).
The parent `polarization-experts/` directory is injected on `sys.path` so
`stokes.py` can re-use `UNetPro256`, `SSIMLoss`, `AdvancedLoss` from the
sibling `run_exp2_baseline.py` without duplicating code.

---

## 2. Run matrix (5 modes × 5 seeds = 25 runs)

| Tag      | Input                  | Projection                            | UNet `in_ch` | Purpose                            |
| -------- | ---------------------- | ------------------------------------- | ------------ | ---------------------------------- |
| **A0**   | 4-pol, calibrated      | 1×1 conv 4→4, **learnable**, init=θ   | 4            | Main proposed method               |
| **A1**   | 4-pol, calibrated      | 1×1 conv 4→4, **frozen**, =θ          | 4            | Physical prior contribution        |
| **A2**   | 4-pol, calibrated      | none                                  | 4            | Normalisation-only effect          |
| **A2-raw** | 4-pol, raw (/255)    | none                                  | 4            | Multi-polarisation itself          |
| **A3**   | 1-pol (pol2 / 45°), raw | none                                 | 1            | Single-polarisation baseline       |

`θ` is the **theoretical Stokes matrix** (rows: S₀, S₁, S₂, S_res):

```
     I0°   I45°   I90°  I135°
S0 [ 0.5    0.5    0.5    0.5  ]
S1 [ 1.0    0      −1.0   0    ]
S2 [ 0      1.0    0     −1.0  ]
S_res [ 0    0      0      0   ]   ← calibration residual, init 0
```

S_res is a 4-th "residual / calibration-error" channel; if it stays near 0
after training, the 3-D Stokes basis is complete; if it diverges, the network
is compensating for system imperfections (non-ideal polarisers, gain
mismatch, …). Either outcome strengthens the patent narrative.

---

## 3. Why this design — key choices and the reasoning

### 3.1 Why a 1×1 conv (not a fully-connected layer)
A `1×1` convolution applies the same 4×4 matrix to every pixel independently.
This matches the *per-pixel definition* of Stokes parameters and produces the
patent-friendly phrasing: **"逐像素方式作用于多偏振散斑，使每个空间位置独立完
成偏振基变换"**.

### 3.2 Why 4→4 (not 4→3)
Keeping the 4-th row (S_res) lets the network either:
- collapse it to zero (validating 3-D Stokes completeness), **or**
- learn a residual that absorbs sensor non-idealities.

Either result is a defensible patent-narrative move. The 4-th row is
described in the patent as "system calibration residual channel".

### 3.3 Why per-channel **multiplicative** calibration (not z-score)
Z-score destroys the linear relation `I₀ + I₉₀ = I₄₅ + I₁₃₅ = S₀` that the
Stokes basis depends on. A single per-channel gain preserves the relation and
only equalises the mean intensity of each polarisation channel on the
training split, fixing the strong asymmetry observed in the data
(pol3/pol4 are ~3× darker than pol1/pol2). See `norm_stats.json` for the
exact `calib` factors used.

### 3.4 Why no per-component stem
We considered routing each Stokes output through a small conv stem and
concatenating, but it would "re-mix" the channels and dilute the physical
orthogonality that the projection just established. The clean ablation is:
A0/A1/A2 share an identical UNet backbone (in_ch=4), and the *only*
difference between A0/A1 and A2 is the presence of the 1×1 projection layer.

### 3.5 Why a separate learning rate for the projection
The projection has 16 parameters vs. ~tens of millions in the UNet. With a
single LR, the projection learns essentially nothing. We use a parameter
group with **10× the main LR** (`PROJ_LR = 2e-3` vs `LR = 2e-4`) plus the
same warmup+cosine schedule (see `stokes.py::build_optimizer`).

### 3.6 Why A2-raw is in the matrix (the late addition)
Without A2-raw, every observed gain `A0 > A3` is a joint effect of *Stokes
projection + normalisation + multi-polarisation*. Adding A2-raw splits the
chain into 5 separately-attributable increments:

```
A3      → 1-pol baseline
A2-raw  → multi-polarisation itself     ( A2-raw − A3 )
A2      → calibration normalisation     ( A2 − A2-raw )
A1      → physical Stokes prior         ( A1 − A2 )
A0      → learnable refinement          ( A0 − A1 )
```

Each delta supports a specific claim in the patent / paper.

### 3.7 Why 5 seeds (not 3)
3-seed paired t-tests have df=2 — almost no chance of `p<0.10` even for real
effects. 5 seeds gives df=4 and a workable threshold.

### 3.8 Decision rule
Direction A is declared successful iff:

```
mean(A0_test_pcc) − mean(A2_test_pcc)  ≥  2 × pooled_std
                                AND
            paired_t_test(A0, A2)  p < 0.10
```

`pooled_std = sqrt( (var(A0_seeds) + var(A2_seeds)) / 2 )`.

If neither passes, we fall back to **Direction B (cross-polarisation residual
correction)**.

---

## 4. Naming conventions inside `stokes.py`

| Constant    | Value                            |
| ----------- | -------------------------------- |
| `POL_CHANNELS` | `[1, 2, 3, 4]` (= 0°, 45°, 90°, 135°; pol0=OG, excluded) |
| `COLOR_CHANNEL` | `2` (R channel in BGR)         |
| `TRAIN_IDX` | `range(0, 1600)`                  |
| `VAL_IDX`   | `range(1600, 1800)`               |
| `TEST_IDX`  | `range(1800, 2000)`               |
| `EPOCHS`    | 60                                |
| `BATCH`     | 4                                 |
| `LR`        | 2e-4 (UNet)                       |
| `PROJ_LR`   | 2e-3 (Stokes projection, 10× LR)  |
| `WARMUP`    | 10 epochs linear → cosine to 1e-6 |
| `VIZ_INDICES` | `[1846, 1851, 1877, 1904, 1920]` (in TEST_IDX) |

The dataset paths and split boundaries are **identical** to
`run_exp2_baseline.py` so A3 should reproduce the 1-pol baseline within
seed-level noise.

---

## 5. How to run

```bash
# 1. (Already done) Compute calibration statistics.
cd /root/polarization-experts/pta_a
python3 norm_stats.py

# 2. Launch the 25-run sweep inside tmux.
tmux new -s pta-a -d 'bash /root/polarization-experts/pta_a/run.sh'
tmux new-window -t pta-a 'watch -n 5 nvidia-smi'
tmux new-window -t pta-a 'tail -F /root/polarization-experts/pta_a/logs/_orchestrator.log'
tmux attach -t pta-a   # detach with Ctrl-b, d

# 3. After the sweep finishes (≈13 h on a single 4090):
cd /root/polarization-experts/pta_a
python3 analyze.py
# → plots/summary.txt and plots/*.png
```

### Resume after interruption
`stokes.py` consults `results.json` and skips any `(mode, seed)` that already
has a `completed=true` entry. Re-launching `run.sh` after a tmux kill or SSH
drop simply continues from where it stopped. To force a re-run of one cell,
delete its entry from `results.json` or pass `--force`.

### Single-run debugging
```bash
python3 stokes.py --mode A0 --seed 1
```
This is the form `run.sh` uses internally; it produces one `[SKIP]` line if
the run is already complete, otherwise it trains, evaluates, and appends to
`results.json`.

---

## 6. Outputs produced

| Path                         | Producer       | Content                                     |
| ---------------------------- | -------------- | ------------------------------------------- |
| `norm_stats.json`            | `norm_stats.py`| `mean_pc`, `calib`, `train_indices_hash`    |
| `results.json`               | `stokes.py`    | One entry per `(mode, seed)`; metrics, curves, W_P trajectory, per-sample test PCC |
| `logs/<mode>_seed<n>.log`    | `run.sh`       | Per-run stdout                              |
| `logs/_orchestrator.log`     | `run.sh`       | High-level run-status timeline              |
| `viz/A0_seed1.npz`           | `stokes.py`    | Input + projection-init + projection-final feature snapshots, 5 fixed test samples |
| `viz/A1_seed1.npz`           | `stokes.py`    | Same, for the frozen-Stokes mode            |
| `plots/summary.txt`          | `analyze.py`   | Per-mode mean±std table + attribution chain + paired t-tests |
| `plots/01_pcc_convergence_seed1.png` | `analyze.py` | Val / train PCC vs epoch for all 5 modes (seed=1) |
| `plots/02_wp_final_heatmap.png` | `analyze.py` | Theoretical / learned / delta matrices (A0 seed=1) |
| `plots/03_test_pcc_distribution.png` | `analyze.py` | ECDF + violin of per-sample test PCC |
| `plots/04_attribution_chain.png` | `analyze.py` | Bar chart: PCC mean±std along A3→A2-raw→A2→A1→A0 |
| `plots/05_stokes_features_sample0.png` | `analyze.py` | One-sample input vs init vs final Stokes feature maps |

---

## 7. Self-checks performed by `stokes.py` at startup

1. **norm_stats hash** — `train_indices_hash` in `norm_stats.json` must equal
   the recomputed hash from `(TRAIN_IDX, COLOR_CHANNEL, POL_CHANNELS)`.
   Refuses to launch on mismatch.
2. **Stokes init numerical** — synthesises `x = (1, 2, 3, 4)` and asserts the
   projection output equals `(5, −2, −2, 0)` to 1e-5.
3. **Param-count delta** — builds an A3 reference model, asserts the
   per-mode parameter delta matches the closed-form expectation
   (`+1296` for in_ch 1→4, `+16` for the Stokes projection).
4. **GPU required** — refuses to fall back to CPU silently; would
   contaminate cross-mode comparisons.
5. **VGG19 perceptual loss must load** — refuses to fall back to the
   degraded 0.4/0.4/0.2 loss weights; would contaminate comparisons too.

Failures here are *intentionally fatal* — we do not want a silent
configuration drift to ruin 13 hours of training.

---

## 8. Patent-narrative anchors

These are the specific phrasings the experiments are designed to support
(see `../thesis.py` and the patent draft separately):

1. *"基于偏振投影空间重构的多模光纤图像重建方法"* — A0 is the canonical
   instance, A1 is the same with a fixed physical-prior projection.
2. *"针对实际系统中各偏振通道因分光比与器件响应差异导致的强度不均衡，本方法
   在偏振投影前引入逐通道归一化预处理"* — supported by A2 vs A2-raw delta.
3. *"所述偏振投影变换以逐像素方式作用于多偏振散斑"* — built-in via the 1×1
   conv; explicit in the `StokesProjection` module.
4. *"系统校准残差通道用于补偿实际系统与理想偏振模型之间的偏差"* — supported
   by inspecting row 4 of the trained `W_P` in `plots/02_wp_final_heatmap.png`.

---

## 9. What is **not** done in this directory (out of scope)

- Round-2 ablations: per-Stokes-component contribution (drop S₀ / S₁ / S₂
  individually).
- Direction B (cross-polarisation residual correction).
- Direction C (compressed-sensing dictionary).
- Patent / paper drafting itself — those live elsewhere.

# PTA-A Pre-Launch Self-Review

This is the "review before opening tmux" checklist. Every item must pass
before launching the 25-run sweep.

> Status is recorded inline as `[ ]` / `[x]`. Once the entire list is checked,
> the launch command at the bottom is fair game.

---

## A. Data plumbing

- [ ] `speckles6000_og.npy` shape is `(6000, 5, 256, 256)` uint8 (already confirmed).
- [ ] `pol_channels = [1, 2, 3, 4]` map to 0° / 45° / 90° / 135° (pol0 = OG, excluded).
- [ ] `color_channel = 2` (R, matches `run_exp2_baseline.py`).
- [ ] Splits identical to baseline: train `0..1599`, val `1600..1799`, test `1800..1999`.
- [ ] Mode A3 path returns 1×256×256 tensors (pol2, raw / 255) for backwards compat.
- [ ] Modes A0/A1/A2 return 4×256×256 calibrated tensors.
- [ ] Mode A2-raw returns 4×256×256 raw tensors (no calib).

## B. Normalisation correctness

- [ ] `norm_stats.json` exists and has the four keys:
      `mean_pc`, `calib`, `train_indices_hash`, `target_mean`.
- [ ] `target_mean ≈ mean(mean_pc) ≈ 0.1499` (cross-check arithmetic).
- [ ] `calib[i] * mean_pc[i]` is the same for all i, equal to `target_mean`.
- [ ] `train_indices_hash` matches the hash recomputed inside `stokes.py`
      at startup (this is the in-script self-check that fails fatally on mismatch).

## C. Stokes projection layer

- [ ] `theoretical_stokes_matrix()` returns the documented 4×4:
      rows = (S₀, S₁, S₂, S_res), cols = (I0°, I45°, I90°, I135°).
- [ ] On A0 init, projecting `x = (1, 2, 3, 4)` produces `(5, −2, −2, 0)`
      (this is the in-script numerical self-check).
- [ ] In A0, `stokes_proj.conv.weight.requires_grad == True`.
- [ ] In A1, every parameter inside `stokes_proj` has `requires_grad == False`.
- [ ] In A2 / A2-raw / A3, `model.stokes_proj is None`.

## D. Optimiser & schedule

- [ ] Two parameter groups when `stokes_proj is not None`:
      `proj_params` at `PROJ_LR=2e-3`, all others at `LR=2e-4`.
- [ ] One parameter group otherwise.
- [ ] AdamW, `weight_decay=1e-5`, `betas=(0.9, 0.999)`.
- [ ] 10-epoch linear warmup → cosine to `eta_min=1e-6` over the remaining 50.
- [ ] Both groups warm up and decay in lockstep (factor multiplied
      against `base_lrs` per group).
- [ ] Grad clip max-norm 1.0 on all parameters.

## E. Loss

- [ ] `AdvancedLoss` weights `0.25 / 0.25 / 0.35 / 0.15` (MSE / L1 / SSIM / VGG).
- [ ] `loss_fn.use_perceptual is True` — script aborts otherwise.
      This guarantees A0..A3 are compared under identical loss.

## F. Evaluation

- [ ] PCC / SSIM / MSE all computed at 64×64 (adaptive-avg-pool from 256).
- [ ] Per-sample PCC stored only for the test set (not val / train_eval).
- [ ] Test evaluation invokes `want_per_sample=True`.

## G. Self-checks at script startup (fail fast)

- [ ] norm_stats hash match.
- [ ] Stokes init numerical match (A0/A1 only).
- [ ] Parameter-count delta vs A3 reference model:
      - A2 / A2-raw: +1296 (enc1 1ch → 4ch first conv: `48*3*3*(4-1)` = 1296)
      - A1 / A0:     +1296 + 16 = +1312
- [ ] GPU available — else abort, never silent CPU fallback.

## H. Outputs & resume semantics

- [ ] `results.json` is the single source of truth for "already done".
- [ ] `already_done(mode, seed)` returns True only if `completed == True`.
- [ ] `append_result` is atomic (write-then-rename via `.tmp`).
- [ ] On `--force`, the old entry is *not* removed but a new one is appended
      (caller's responsibility to dedupe in analysis).
- [ ] `viz/<mode>_seed1.npz` is written only for A0/A1 + seed=1.

## I. Monitoring hooks

- [ ] `wp_trajectory` appended every epoch (A0/A1).
- [ ] `wp_theoretical` stored once per entry (A0/A1).
- [ ] `val_pcc_curve` / `val_ssim_curve` / `train_loss_curve` /
      `train_pcc_curve` populated every epoch.
- [ ] Stokes feature snapshot captured at init **and** at final, A0/A1 seed=1.

## J. Orchestrator

- [ ] `run.sh` `cd`s to its own directory (uses `BASH_SOURCE`).
- [ ] Pre-flight checks `norm_stats.json` and `nvidia-smi`.
- [ ] Per-run stdout/stderr captured to `logs/<tag>.log`.
- [ ] On non-zero exit from `stokes.py`, orchestrator aborts (fail-fast).
- [ ] On `[SKIP]` line, orchestrator counts it as skipped (not failed).

## K. Repro-hash sanity

- [ ] `set_seed(seed)` calls `torch.cuda.manual_seed_all` and sets
      `cudnn.deterministic=True, cudnn.benchmark=False`.
- [ ] DataLoader uses `shuffle=True` only on the train loader.

## L. Open risks (acknowledged, not blockers)

- pol3 / pol4 mean intensity is ~3× lower than pol1 / pol2. We compensate
  in A0/A1/A2 via multiplicative calibration, but not in A2-raw.
  A2-raw is *expected* to be worse than A2 for this reason — that's exactly
  the increment we want to measure.
- 5 seeds at df=4 is the minimum viable for paired t-test at p<0.10.
  If `p` lands between 0.05 and 0.10, the verdict is "weak positive" and
  we should add 5 more seeds before publication-quality claims.

---

## Launch command (only after every box above is ticked)

```bash
tmux new -s pta-a -d 'bash /root/polarization-experts/pta_a/run.sh'
tmux new-window -t pta-a 'watch -n 5 nvidia-smi'
tmux new-window -t pta-a 'tail -F /root/polarization-experts/pta_a/logs/_orchestrator.log'
tmux attach -t pta-a
```

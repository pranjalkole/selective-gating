# Results — selective gating on LOL-v1

Region-stratified evaluation protocol (see `src/selgate/eval/stratified.py`):
test pixels are split **per image** into dark / mid / bright **terciles of the
ground-truth luminance**; PSNR and ΔE2000 are reported per stratum. The
headline claim under test: *selective gating improves bright-region fidelity
at no dark-region cost*.

## Setup (identical for every row)

| Knob | Value |
|---|---|
| Dataset | LOL-v1 (`soumikrakshit/lol-dataset` on Kaggle) |
| Split | committed `configs/split_lol_v1.json`: 461 train / 24 val / 15 test |
| Tuning | **val only**; test-15 evaluated once per run, never used for tuning |
| Budget | 50 000 iters, cosine schedule (all runs equal) |
| Hardware | 1× Kaggle T4 (≈2 h per run) |
| Seed | 0 (`seed_everything`) |
| Backbone | Retinexformer official LOL-v1 config (1.6 M params) |
| Gate | 19 969 params (pixel granularity unless noted) |

## Headline — LOL-v1 test-15 (standard metrics)

| Run | PSNR ↑ | SSIM ↑ | LOE ↓ |
|---|---|---|---|
| **gate_pixel** (ours) | **22.43** | **0.840** | 0.068 |
| baseline (no gate) | *pending* | *pending* | *pending* |
| gate_global | *pending* | *pending* | *pending* |
| gate_patch | *pending* | *pending* | *pending* |
| frozen_backbone | *pending* | *pending* | *pending* |
| no_illum_prior | *pending* | *pending* | *pending* |
| sweep_illum_003 | *pending* | *pending* | *pending* |
| sweep_tv_005 | *pending* | *pending* | *pending* |

## Region-stratified (the protocol) — LOL-v1 test-15

PSNR (dB) / ΔE2000, mean over the 15 images:

| Run | dark | mid | bright | overall ΔE |
|---|---|---|---|---|
| **gate_pixel** (ours) | 22.97 / 8.63 | 22.48 / 9.05 | 22.83 / 8.17 | 8.62 |
| baseline (no gate) | *pending* | *pending* | *pending* | *pending* |
| gate_global | *pending* | *pending* | *pending* | *pending* |
| gate_patch | *pending* | *pending* | *pending* | *pending* |
| frozen_backbone | *pending* | *pending* | *pending* | *pending* |
| no_illum_prior | *pending* | *pending* | *pending* | *pending* |
| sweep_illum_003 | *pending* | *pending* | *pending* | *pending* |
| sweep_tv_005 | *pending* | *pending* | *pending* | *pending* |

## gate_pixel training dynamics (50k iters, T4, 123 min)

Val PSNR (24 val images, evaluated every 5k):

| iter | 5k | 10k | 15k | 20k | 25k | 30k | **35k** | 40k | 45k | 50k |
|---|---|---|---|---|---|---|---|---|---|---|
| PSNR | 21.85 | 21.07 | 22.54 | 21.98 | 22.90 | 22.91 | **23.37** | 22.51 | 22.57 | 22.58 |

`best.pt` is the iter-35k checkpoint (all test numbers above use it).

Gate behaviour over training (last logged iter): `gate_mean 0.948`,
`gate_std 0.043`, `frac_open 0.853`, `frac_closed 0.000` — the gate stays
mostly open but carries real per-pixel modulation; no collapse to a constant.
(The iter-1 value was 0.269 — the gate learns to open early in training.)

## Interpretability (pending `selgate-03-gate-analysis`)

Will report per-stratum gate statistics inside the same dark/mid/bright GT
terciles (does the gate open in brighter regions?), Pearson r between stratum
luminance and gate value, and gate heatmap overlays.

## Files

Raw metric dumps live in the **kernel outputs** (not committed — `results/` is
gitignored); fetch any run's with:

```bash
kaggle kernels output pkbbanzik/selgate-02-gate-pixel -p ./downloaded
# -> selective-gating/results/gate_pixel/test/{standard_metrics.csv,
#    stratified_summary.csv, stratified_per_image.csv, metrics.json, predictions/}
# -> runs/gate_pixel/{best.pt,last.pt,history.json}   (train kernel output)
```

## Notes / caveats

- 50k budget vs the official 150k schedule: absolute PSNR is below the
  Retinexformer paper's ~24 dB, but **all runs share the budget**, so the
  comparisons are internally consistent.
- LPIPS unavailable on the Kaggle image (optional dep, skipped with a warning);
  SSIM/LOE/ΔE2000 are computed by `selgate` itself.
- Every number here was produced by a re-pushable Kaggle kernel
  (`kaggle_push/selgate-*`); kernel logs are the audit trail.

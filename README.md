# Selective Enhancement Gating — *"Don't enhance what's already lit"*

Illumination-Aware Selective Gating for Low-Light Image Enhancement: a minimal,
interpretable add-on to **Retinexformer** (ICCV'23), plus a **region-stratified
evaluation protocol** that measures *where* enhancement changes occur — not
just the average over 15 test images.

```
x_out = g ⊙ F_θ(x) + (1 − g) ⊙ x

L(x)   = cheap illumination estimate (per-pixel RGB max / luma — no extra net)
g      = σ(f_gate([x, L(x)]))     # 3–4 convs, ~20 K params, pixel/patch/global
F_θ    = Retinexformer backbone   # frozen or joint
```

`g ≈ 1` in dark regions (full enhancement), `g ≈ 0` in already-lit regions
(near-identity pass-through). Motivated by LIAR (NeurIPS'25): indiscriminate
enhancement *hurts*; region-aware treatment is the missing piece.

## Repository layout

```
selective-gating/
├── configs/                  # baseline, gate_{pixel,patch,global}, smoke, ablations/
├── src/selgate/
│   ├── paths.py              # local ⇄ Kaggle path resolution (env-aware)
│   ├── config.py             # YAML config + dot-key CLI overrides
│   ├── data/lol.py           # LOL-v1 loader + leak-proof train/val split
│   ├── models/
│   │   ├── backbone.py       # vendored Retinexformer (pinned commit)
│   │   ├── gate.py           # L(x) + f_gate, granularity variants
│   │   └── gated_model.py    # x_out = g·F(x) + (1−g)·x, collapse stats
│   ├── losses.py             # L1 + 0.2(1−SSIM) + λ_s·TV(g) + λ_i·L_illum
│   ├── train.py              # training loop, AMP, gate monitoring, ckpting
│   └── eval/
│       ├── standard.py       # PSNR / SSIM / LPIPS
│       ├── stratified.py     # ★ dark/mid/bright terciles → PSNR + ΔE2000
│       ├── loe.py            # lightness order error (over-enhancement guard)
│       └── cross_dataset.py  # LIME/NPE/DICM → NIQE/BRISQUE/PI (pyiqa)
├── scripts/                  # train / evaluate / cross_dataset / make_split / run_ablations
├── notebooks/                # 00_kaggle_setup · 01_train · 02_evaluate (see KAGGLE.md)
├── tests/                    # 32 tests: split integrity, gate, losses, protocol, smoke train
├── third_party/Retinexformer # vendored base code (pinned, MIT — see third_party/README.md)
└── KAGGLE.md                 # ★ how to run all of this in a Kaggle notebook
```

## Quickstart (local)

```bash
pip install -e ".[dev]"            # or: pip install -r requirements.txt
pytest tests/ -q                   # sanity: 32 tests, CPU, <1 min

# data: put LOL-v1 under data/LOL-v1 (see data/README.md)
python scripts/make_split.py                       # → configs/split_lol_v1.json
python scripts/train.py --config configs/smoke.yaml          # 1-min pipeline check
python scripts/train.py --config configs/gate_pixel.yaml     # real run (~2–4 h GPU)

python scripts/evaluate.py --config configs/gate_pixel.yaml \
    --ckpt runs/gate_pixel/best.pt --split test --out results/gate_pixel
```

## Quickstart (Kaggle)

See **[KAGGLE.md](KAGGLE.md)** — three paths (web upload / CLI / offline
dataset). TL;DR: upload `notebooks/00_kaggle_setup.ipynb`, set `REPO_URL`,
enable GPU + internet, attach LOL-v1, Run All; then `01_train`, then
`02_evaluate`.

## The evaluation protocol (the thesis contribution)

| Level | What | Metrics |
|---|---|---|
| 1. Standard | LOL-v1 test-15 | PSNR / SSIM / LPIPS |
| 2. **Region-stratified** | test pixels split into **dark / mid / bright terciles by GT luminance** | PSNR + ΔE2000 **per stratum** |
| 3. Cross-dataset | LIME, NPE, DICM | NIQE / BRISQUE / PI |
| 4. Naturalness | LOL-v1 | LOE |

Headline claim (or honest negative result):
*"gating improves bright-region fidelity at no dark-region cost."*
Outputs: `stratified_summary.csv` + `stratified_per_image.csv` — reusable for
other methods (drop predictions into a folder and call
`selgate.eval.run_stratified_eval(gt_dir, pred_dir)`).

**Split rule:** validation is carved from the 485 train images only
(`configs/split_lol_v1.json`); the 15 test images are never used for tuning.
`tests/test_split.py` enforces this.

## Ablation matrix

| # | Config | Question |
|---|---|---|
| 1 | `configs/baseline.yaml` | gate off — the control |
| — | `configs/gate_pixel.yaml` | default method |
| 2 | `configs/ablations/no_illum_prior.yaml` | does the interpretability constraint matter? (`λ_i=0`) |
| 3 | `gate_{global,patch,pixel}.yaml` | granularity |
| 4 | `configs/ablations/frozen_backbone.yaml` | gate-only vs joint |
| 5 | `ablations/sweep_*.yaml` | `λ_s`, `λ_i` sweeps (validation only) |

`./scripts/run_ablations.sh` runs them sequentially (GPU-days scale: ~1–2).

## Results

See **[RESULTS.md](RESULTS.md)** — LOL-v1 test-15, region-stratified protocol,
one re-pushable Kaggle kernel per run. Headline so far (`gate_pixel`, 50k
iters, T4): **22.43 dB / SSIM 0.840**, ΔE2000 8.6 overall; remaining rows in
progress.

## References

- Cai et al., *Retinexformer: One-stage Retinex-based Transformer for Low-light
  Image Enhancement*, ICCV 2023 — base method (vendored, MIT).
- LIAR, NeurIPS'25 (nighttime occupancy): indiscriminate enhancement hurts.
- DALE (dark-region-aware) / InterLight (luminance-gated fusion): closest
  prior work — re-check before proposal submission (see idea doc §5).

See `../research-idea-1-selective-gating.md` for the full research plan.

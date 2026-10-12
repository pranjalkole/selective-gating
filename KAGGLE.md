# Running everything in a Kaggle notebook

Three ways to get this repo into Kaggle, pick one:

| Path | When to use | Needs internet? |
|---|---|---|
| **A. Web UI upload** | You just want to run it | Yes (for clone + pip) |
| **B. Kaggle CLI push** | You want reproducible, re-pushable notebooks | Yes |
| **C. Offline dataset** | Internet-disabled kernels (competition rules) | No |

All three end up in the same state: repo at `/kaggle/working/selective-gating`,
`selgate` importable, LOL-v1 attached as data.

---

## 0. One-time prerequisites

1. **Push this repo to GitHub** (public repo = simplest; for a private repo see
   the "private repo" note in Path A step 3):
   ```bash
   cd selective-gating
   git add -A && git commit -m "selgate: initial implementation"
   git remote add origin https://github.com/<you>/selective-gating.git
   git push -u origin main
   ```
2. **Kaggle API token:** Kaggle → profile → Settings → API → *Create New Token*
   → puts `kaggle.json` in `~/.kaggle/` (only needed for Paths B and C):
   ```bash
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
3. **LOL-v1 data on Kaggle:** attach any LOL-v1 dataset to the notebook
   (see `data/README.md` for accepted layouts). If none exists yet, upload your
   own copy once:
   ```bash
   kaggle datasets init -p data_bundle            # edits dataset-metadata.json
   # set "slug": "lol-v1", drag in the LOL-v1 folder, then:
   kaggle datasets create -p data_bundle
   ```

---

## Path A — Web UI (recommended for a first run)

1. Kaggle → **Code → New Notebook** (GPU is *not* enabled by default on
   existing notebooks — set it in step 3).
2. In the editor, create three files and upload the repo's notebooks
   (**Add file → Upload file**): `notebooks/00_kaggle_setup.ipynb`,
   `01_train.ipynb`, `02_evaluate.ipynb`.
   *(Or copy-paste each notebook's cells into three separate notebooks —
   each notebook is self-contained and recreates its own environment.)*
3. In the **right-hand Settings pane** set:
   - **Accelerator:** GPU (T4 / P100 / T4×2) — required for training
   - **Internet:** **On** — required for `git clone` + `pip install`
   - **Data:** **Add data →** attach your LOL-v1 dataset
4. Open `00_kaggle_setup.ipynb`, edit the first cell:
   ```python
   REPO_URL = "https://github.com/<you>/selective-gating.git"
   ```
5. **Run All** on `00`. It must print `SETUP OK` before you continue.
6. Open `01_train.ipynb`, keep `CONFIG = "configs/smoke.yaml"` for the first
   pass (~1–2 min), **Run All**, confirm `checkpoint: .../runs/smoke/last.pt`.
7. Switch `CONFIG` to `configs/gate_pixel.yaml` (the real run, `iters: 150000`,
   ≈2–4 h on a T4) and Run All again. Watch the stdout: `iter .../150000`,
   `val @ ...`, and the `gate_mean / frac_open / frac_closed` collapse monitor.
8. Run `02_evaluate.ipynb` → prints the standard table and the
   **region-stratified table** (dark/mid/bright PSNR + ΔE2000).

> **Private repo:** store a GitHub PAT as a Kaggle **Secret** named
> `GITHUB_TOKEN` (Settings → Secrets), then change the clone cell to:
> ```python
> url = REPO_URL.replace("https://", f"https://oauth2:{os.environ['GITHUB_TOKEN']}@")
> subprocess.run(["git", "clone", "--depth", "1", url, str(repo)], check=True)
> ```

---

## Path B — Kaggle CLI (re-pushable)

Each notebook gets a `kernel-metadata.json`. From the repo root:

```bash
# one metadata file per notebook, e.g. notebooks/00_kaggle_setup/kernel-metadata.json
cat > notebooks/00_kaggle_setup/kernel-metadata.json <<'EOF'
{
  "id": "<you>/selgate-00-setup",
  "title": "selgate-00-setup",
  "code_file": "00_kaggle_setup.ipynb",
  "kernel_type": "notebook",
  "language": "python",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": ["<you>/lol-v1"],
  "competition_sources": [],
  "kernel_sources": []
}
EOF

kaggle kernels init -p notebooks/00_kaggle_setup   # if you prefer the scaffold
kaggle kernels push -p notebooks/00_kaggle_setup   # upload + start running
kaggle kernels status <you>/selgate-00-setup       # poll until "complete"
kaggle kernels output  <you>/selgate-00-setup -p ./kaggle_out   # fetch results
```

Repeat for `01_train` and `02_evaluate` (same metadata shape, change
`id`/`title`/`code_file`). Remember to edit `REPO_URL` inside each notebook
before pushing.

---

## Path C — Offline (internet-disabled kernel)

Use when internet must stay off (competition submission rules):

1. Sync the repo as a Kaggle dataset (run locally; internet only needed here):
   ```bash
   # bundle everything except git/venv/runs
   git archive -o selgate-code.tar.gz HEAD
   mkdir -p /tmp/selgate-code && tar -xf selgate-code.tar.gz -C /tmp/selgate-code
   kaggle datasets init -p /tmp/selgate-code          # set "slug": "selgate-code"
   kaggle datasets create -p /tmp/selgate-code        # or -u to update
   ```
2. In the notebook: **Add data →** attach `selgate-code` *and* LOL-v1.
3. Replace the clone cell of `00`/`01`/`02` with:
   ```python
   repo = Path("/kaggle/working/selective-gating")
   subprocess.run(["cp", "-r", "/kaggle/input/selgate-code", str(repo)], check=True)
   ```
   (copy because `/kaggle/input` is **read-only**; never pip-install into it)
4. Skip `pip install` for core deps — torch/numpy/pillow/pyyaml/einops ship
   with the Kaggle image or are already satisfied; with internet off you can
   only rely on what's preinstalled, which is enough for training + PSNR/SSIM/
   stratified/LOE. `lpips`/`pyiqa` need internet — those two metric families
   degrade gracefully (they're skipped with a warning).

---

## What runs where

| Notebook | Equivalent CLI | Outputs (all under `/kaggle/working`) |
|---|---|---|
| `00_kaggle_setup` | — | prints `SETUP OK` |
| `01_train` | `python scripts/train.py --config <cfg>` | `runs/<name>/{best,last}.pt`, `history.json` |
| `02_evaluate` | `python scripts/evaluate.py ...` | `results/<name>/test/{standard_metrics.csv, stratified_summary.csv, stratified_per_image.csv, metrics.json, predictions/}` |
| (cross-dataset) | `python scripts/cross_dataset.py ...` | `results/<name>/cross/no_ref_scores.csv` |

Everything written to `/kaggle/working` persists as **notebook output**:
download via the Output pane in the UI, or `kaggle kernels output <slug>`.
`selgate` never writes to `/kaggle/input`.

## Time budget

| Run | Config | Roughly |
|---|---|---|
| smoke | `configs/smoke.yaml` (30 iters) | ~1 min |
| full train | `configs/gate_pixel.yaml` (150k iters) | ~2–4 h on T4 |
| ablation ×8 | `scripts/run_ablations.sh` | **won't fit one session** — push one notebook per config (CLI path), or reduce `--set train.iters=...` |
| eval + stratified | `scripts/evaluate.py` on test-15 | ~1 min |
| cross-dataset | `scripts/cross_dataset.py` | minutes |

- **Resume after a timeout:** rerun `01` with
  `--set ...` plus `python scripts/train.py --config ... --resume runs/<name>/last.pt`.
- Tight on session time → train with `--set train.iters=50000`; the
  cosine schedule automatically reshapes to the new length.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Repository not found` / `Could not resolve host` | Internet off | Settings → Internet: **On** |
| `LOL-v1 not found` | Dataset not attached / nested root | Attach data; set `data.root` or `SELGATE_LOL_ROOT` to the dir containing `our485/` or `Train/` |
| `ModuleNotFoundError: selgate` | PYTHONPATH cell not run | Run `00` cells 1–2 in this session (each notebook re-runs them) |
| `PermissionError` under `/kaggle/input` | read-only mount | Work from `/kaggle/working` (the provided code already does) |
| `CUDA out of memory` | T4 smaller than dev GPU | `--set data.batch_size=4 --set data.crop_size=96` |
| Session killed at timeout | 9 h GPU cap | `--resume runs/<name>/last.pt`, or lower `train.iters` |
| `lpips/pyiqa unavailable` | optional deps | Expected w/ internet off — metrics skipped, everything else runs |
| Results from old code | stale clone | `00` cell 1 does `git pull`; restart kernel if weirdness persists |

## Tests before you burn GPU time

```bash
pip install pytest && pytest tests/ -q     # 32 tests, CPU, <1 min
```

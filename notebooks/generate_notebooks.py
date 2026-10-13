#!/usr/bin/env python3
"""Generate the three Kaggle notebooks (kept in sync with scripts/).

Also emits the Path-B push layout:  kaggle_push/<slug>/{*.ipynb, kernel-metadata.json}
"""

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent
REPO_ROOT = NB_DIR.parent

# ---- Path B configuration -------------------------------------------------
GH_USER = "pranjalkole"
REPO_URL = f"https://github.com/{GH_USER}/selective-gating.git"
KAGGLE_USER = "pkbbanzik"
DATASET_SOURCES = [
    "soumikrakshit/lol-dataset",        # LOL-v1: lol_dataset/{our485,eval15}/{low,high}
    "zukniz/retinexformer-weights",     # optional: official-style LOL_v1.pth backbone weights
]
# notebook filename -> (kernel slug, kernel title) for the 3 original kernels
KERNELS = {
    "00_kaggle_setup.ipynb": ("selgate-00-setup", "selgate-00-setup"),
    "01_train.ipynb": ("selgate-01-train", "selgate-01-train"),
    "02_evaluate.ipynb": ("selgate-02-evaluate", "selgate-02-evaluate"),
    "03_gate_analysis.ipynb": ("selgate-03-gate-analysis", "selgate-03-gate-analysis"),
}

# ---- experiment matrix ----------------------------------------------------
# Per-run kernels: one train + one eval kernel per config, eval imports its
# OWN train kernel's output via kernel_sources. Every run uses the same iter
# budget so comparisons are apples-to-apples (cosine schedule reshapes).
TRAIN_ITERS = 50000
RUNS = [
    # wave 1 — headline pair (push these first, both fill the 2 GPU slots)
    ("gate-pixel", "configs/gate_pixel.yaml"),
    ("baseline", "configs/baseline.yaml"),
    # wave 2 — gate granularity ablation
    ("gran-global", "configs/gate_global.yaml"),
    ("gran-patch", "configs/gate_patch.yaml"),
    # wave 3 — model/loss ablations
    ("frozen", "configs/ablations/frozen_backbone.yaml"),
    ("noillum", "configs/ablations/no_illum_prior.yaml"),
    # wave 4 — loss-weight sweeps
    ("illum003", "configs/ablations/sweep_illum_003.yaml"),
    ("tv005", "configs/ablations/sweep_tv_005.yaml"),
]

# Shared first-cell logic: plain anonymous clone (repo is public).
CLONE_CELL = """
REPO_URL = "https://github.com/pranjalkole/selective-gating.git"   # public repo
BRANCH   = "master"

import os, subprocess, sys, shutil
from pathlib import Path

repo = Path("/kaggle/working/selective-gating")
if repo.exists():
    if subprocess.run(["git", "-C", str(repo), "pull", REPO_URL, BRANCH]).returncode != 0:
        print("pull failed -> fresh clone")
        shutil.rmtree(repo)
if not repo.exists():
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, str(repo)], check=True)

print("repo at:", repo)
"""


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.strip("\n").splitlines(keepends=True),
    }


def render(cells):
    """Cells -> (nb dict, notebook JSON text) with the real clone URL baked in."""
    # bake the real clone URL into the notebooks (Path B: no manual editing)
    for c in cells:
        if c["cell_type"] == "code":
            c["source"] = [
                ln.replace(
                    'https://github.com/YOUR_GITHUB_USERNAME/selective-gating.git"  # <-- EDIT ME',
                    REPO_URL + '"',
                )
                for ln in c["source"]
            ]
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    text = json.dumps(nb, indent=1)
    assert "YOUR_GITHUB_USERNAME" not in text, "REPO_URL placeholder left in notebook"
    return nb, text


def save_local(name, text):
    p = NB_DIR / name
    p.write_text(text)
    print("wrote", p)


def save_kernel(slug, title, name, text, kernel_sources=()):
    """Path B: per-kernel push directory with metadata."""
    kdir = REPO_ROOT / "kaggle_push" / slug
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / name).write_text(text)
    meta = {
        "id": f"{KAGGLE_USER}/{slug}",
        "title": title,
        "code_file": name,
        "kernel_type": "notebook",
        "language": "python",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": DATASET_SOURCES,
        "competition_sources": [],
        "kernel_sources": list(kernel_sources),
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", kdir / name, "+ kernel-metadata.json")


def write(name, cells):
    """Original 3 kernels: local notebook + push dir."""
    _, text = render(cells)
    save_local(name, text)
    slug, title = KERNELS[name]
    # 02 imports 01's output (checkpoints) so it can run in a fresh session
    if name == "02_evaluate.ipynb":
        srcs = [f"{KAGGLE_USER}/selgate-01-train"]
    elif name == "03_gate_analysis.ipynb":
        srcs = [f"{KAGGLE_USER}/selgate-01-gate-pixel"]
    else:
        srcs = []
    save_kernel(slug, title, name, text, srcs)


# --------------------------------------------------------------------------
write(
    "00_kaggle_setup.ipynb",
    [
        md(
            """
# 00 — Setup: get `selgate` running in this Kaggle notebook

This notebook (1) clones the `selective-gating` repo into `/kaggle/working`,
(2) makes the `selgate` package importable, (3) checks GPU + dataset wiring,
and (4) runs a 2-second model smoke test.

**Session settings needed** (right-hand Settings pane):
- **Accelerator:** GPU (T4 / P100 / T4x2)
- **Internet:** ON (needed for `git clone` and `pip install`)
- **Data:** attach your LOL-v1 dataset (Add data → search "LOL")

Run the cells **top to bottom**. If any cell fails, stop and fix before
running `01_train.ipynb`.
"""
        ),
        md(
            """## 1. Clone the repo (public — no secrets needed)

The repo (`github.com/pranjalkole/selective-gating`) is public, so a plain
anonymous clone works. If it is ever made private again, attach a
`GITHUB_TOKEN` secret (Add-ons → Secrets) and switch `clone_url` to the
`oauth2:<token>@` form — see `KAGGLE.md`.
"""
        ),
        code(CLONE_CELL),
        md("## 2. Install deps + make `selgate` importable (`/kaggle/input` is read-only, so everything runs from `/kaggle/working`)"),
        code(
            """
src = repo / "src"
sys.path.insert(0, str(src))   # PYTHONPATH only affects child processes, not this kernel
os.environ["PYTHONPATH"] = f"{src}:{os.environ.get('PYTHONPATH', '')}"

# einops/pyyaml ship with the repo requirements; lpips+pyiqa are optional metric extras
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], check=True)

import selgate
print("selgate", selgate.__version__, "->", selgate.__file__)
assert str(repo / "src") in selgate.__file__, "selgate is not loaded from the cloned repo!"
"""
        ),
        md("## 3. Environment report"),
        code(
            """
import torch
from selgate.paths import is_kaggle, repo_root, find_lol_root, out_dir

print("torch:", torch.__version__, "| cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print("on kaggle:", is_kaggle())
print("repo_root:", repo_root())
_ki = Path("/kaggle/input")
def _dirs(p, _d=0):
    if _d > 3 or not p.is_dir():
        return []
    out = []
    for c in sorted(p.iterdir()):
        if c.is_dir():
            out.append(str(c.relative_to(_ki)))
            out += _dirs(c, _d + 1)
    return out
print("/kaggle/input dirs:", _dirs(_ki))

lol = find_lol_root()
print("LOL-v1 root:", lol)
if lol is None:
    raise FileNotFoundError(
        "LOL-v1 not found! Attach the dataset to this notebook (Add data), or set "
        "SELGATE_LOL_ROOT in the next cell to the folder that CONTAINS our485/ or Train/."
    )
from selgate.data import detect_layout
print("layout:", detect_layout(lol))
"""
        ),
        md("## 4. Model smoke test — builds Retinexformer + gate, one random forward pass"),
        code(
            """
import torch
from selgate.config import load_config
from selgate.models import build_backbone, count_params, GateNet, gate_param_count

cfg = load_config(repo / "configs/gate_pixel.yaml")
print("config:", cfg.get_path("name"), "| gate:", cfg.get_path("model.gate.granularity"))

model = build_backbone(**{k: cfg.get_path(f"model.backbone.{k}") for k in ("n_feat", "stage", "num_blocks")})
gate  = GateNet(granularity=cfg.get_path("model.gate.granularity"),
                 width=cfg.get_path("model.gate.width"),
                 depth=cfg.get_path("model.gate.depth"))

x = torch.rand(1, 3, 128, 128)
with torch.no_grad():
    g, _ = gate(x)
    y = model(x)

print(f"backbone params: {count_params(model):,}")
print(f"gate params:     {gate_param_count(gate):,}")
print("output:", tuple(y.shape), "| gate:", tuple(g.shape), f"| gate mean {g.mean():.3f}")
print("\\nSETUP OK — next: run 01_train.ipynb")
"""
        ),
    ],
)

# --------------------------------------------------------------------------
def train_cells(config, iters=None):
    overrides = f'["--set", "train.iters={iters}"]' if iters else "[]"
    return [
        md(
            f"""
# 01 — Train Retinexformer + selective gate

**Run `00_kaggle_setup.ipynb` first** (this notebook re-creates the environment
in its first cell so it can also run standalone).

This kernel trains **`{config}`** (budget: {iters if iters else "config default"} iters).

1. Start with `configs/smoke.yaml` (~1 min) to verify the whole pipeline.
2. Then switch to `configs/gate_pixel.yaml` (the real run: `train.iters: 150000`,
   ≈2–4 h on a Kaggle T4 — check the session timer; use
   `--set train.iters=50000` if you are tight on time).

Checkpoints and logs land in `/kaggle/working/runs/<name>/` — they are saved as
**notebook output** and can be downloaded from the Output tab (or
`kaggle kernels output <slug>`).
"""
        ),
        md("## 1. Environment (clone + install — same as 00, no secrets needed)"),
        code(
            CLONE_CELL + """
src = repo / "src"
sys.path.insert(0, str(src))   # PYTHONPATH only affects child processes, not this kernel
os.environ["PYTHONPATH"] = f"{src}:{os.environ.get('PYTHONPATH', '')}"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], check=True)

import selgate
print("selgate", selgate.__version__)
"""
        ),
        md(
            f"""## 2. Config + run training (`{config}`)

Override `CONFIG` for other runs, or use the template `01_train.ipynb`.

Optional extras:
- warm-start / frozen-backbone ablation (weights dataset is auto-attached):
  `OVERRIDES += ["--set", "model.backbone.pretrained=/kaggle/input/retinexformer-weights/LOL_v1.pth"]`
"""
        ),
        code(
            f"""
CONFIG   = {config!r}
OVERRIDES = {overrides}

cmd = [sys.executable, str(repo / "scripts" / "train.py"), "--config", str(repo / CONFIG), *OVERRIDES]
# auto-resume: if a previous session was killed at the timeout, last.pt is in
# /kaggle/working (persisted notebook output) and training continues from it
from selgate.config import load_config
RUN_NAME = load_config(repo / CONFIG).get_path("name")   # yaml name, not the file stem
_last = Path("/kaggle/working/runs") / RUN_NAME / "last.pt"
if _last.exists():
    cmd += ["--resume", str(_last)]
    print("resuming from", _last)
print(" ".join(cmd))
ret = subprocess.run(cmd, cwd=repo).returncode
assert ret == 0, f"training failed (exit {{ret}})"
"""
        ),
        md("## 3. Training curves + gate monitor"),
        code(
            """
import json
from pathlib import Path
from selgate.config import load_config

name = load_config(repo / CONFIG).get_path("name")   # yaml name, not the file stem
run_dir = Path("/kaggle/working/runs") / name
hist_path = run_dir / "history.json"
if not hist_path.exists():
    # train.out_dir may be overridden by SELGATE_OUT_DIR
    candidates = list(Path("/kaggle/working").rglob("history.json"))
    hist_path = candidates[-1] if candidates else None
assert hist_path and hist_path.exists(), "history.json not found under /kaggle/working"

hist = json.loads(hist_path.read_text())
iters = [h["iter"] for h in hist]

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].plot(iters, [h["total"] for h in hist], label="total")
axes[0].plot(iters, [h["l1"] for h in hist], label="l1")
axes[0].plot(iters, [h.get("illum", 0) for h in hist], label="illum")
axes[0].set_title("losses"); axes[0].legend(); axes[0].set_xlabel("iter")
if "gate_mean" in hist[0]:
    axes[1].plot(iters, [h["gate_mean"] for h in hist], label="gate mean")
    axes[1].fill_between(iters,
                         [h["gate_mean"] - h["gate_std"] for h in hist],
                         [h["gate_mean"] + h["gate_std"] for h in hist], alpha=0.3)
    axes[1].set_title("gate mean ± std"); axes[1].set_xlabel("iter")
    axes[2].plot(iters, [h["frac_open"] for h in hist], label="frac open (g>0.9)")
    axes[2].plot(iters, [h["frac_closed"] for h in hist], label="frac closed (g<0.1)")
    axes[2].set_title("collapse monitor"); axes[2].legend(); axes[2].set_xlabel("iter")
plt.tight_layout(); plt.show()

print("checkpoints:", sorted(p.name for p in run_dir.glob('*.pt')))
print("val PSNR history: see stdout above (val @ ... lines)")
"""
        ),
        md(
            """
## 4. Grab the checkpoint

Download `best.pt` from the **Output** pane on the right
(file tree: `runs/<name>/best.pt`), or from a local terminal:

```bash
kaggle kernels output <your-slug>/<notebook-name> -p ./downloaded
```

Then run `02_evaluate.ipynb` (either in the same Kaggle session or a new one
with the checkpoint attached as a dataset).
"""
        ),
    ]


# template notebook (Path A users pick a config interactively)
write("01_train.ipynb", train_cells("configs/smoke.yaml"))

# per-run train kernels (Path B experiment matrix)
for _suffix, _cfg in RUNS:
    _slug = f"selgate-01-{_suffix}"
    save_kernel(_slug, _slug, "01_train.ipynb",
                render(train_cells(_cfg, TRAIN_ITERS))[1])

# --------------------------------------------------------------------------
def eval_cells(config):
    return [
        md(
            f"""
# 02 — Evaluation: standard + **region-stratified** protocol

Runs on the LOL-v1 **test-15** (never used for tuning), config `{config}`:

| Level | What | Metrics |
|---|---|---|
| 1. Standard | test-15 | PSNR / SSIM / LPIPS |
| 2. **Region-stratified** | dark/mid/bright GT-luminance terciles | PSNR + ΔE2000 **per stratum** |
| 3. Naturalness | LOL-v1 | LOE |
| 4. Cross-dataset (optional) | LIME / NPE / DICM | NIQE / BRISQUE / PI |

**Prerequisites:** the matching `01` kernel has run — its output (checkpoints) is
imported automatically via `kernel_sources`, so this works in a fresh session.
"""
        ),
        md(
            """## 1. Environment (clone + install — same as 00, no secrets needed)

`01`'s output (checkpoints) is auto-imported as an input via `kernel_sources`
in `kernel-metadata.json`, so this kernel can run in a fresh session.
"""
        ),
        code(
            CLONE_CELL + """
sys.path.insert(0, str(repo / "src"))   # PYTHONPATH only affects child processes, not this kernel
os.environ["PYTHONPATH"] = f"{repo / 'src'}:{os.environ.get('PYTHONPATH', '')}"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], check=True)

import torch
print("cuda:", torch.cuda.is_available())
"""
        ),
        md("## 2. Locate config + checkpoint (searches /kaggle/working and imported kernel outputs)"),
        code(
            f"""
CONFIG = {config!r}
RUN_NAME = Path(CONFIG).stem

candidates = sorted(set(Path("/kaggle/working").rglob("best.pt"))
                    | set(Path("/kaggle/input").rglob("best.pt")))
matches = [p for p in candidates if RUN_NAME in str(p)]
if matches:
    CKPT = matches[0]
else:
    # run-name stem can differ slightly from the yaml's internal name (e.g.
    # sweep_illum_003 vs ablation_sweep_illum003) — only fall back to a
    # lone candidate, never silently pick among several.
    assert len(candidates) == 1, "ambiguous checkpoint selection: " + repr(candidates)
    CKPT = candidates[0]
assert CKPT is not None, "no best.pt found — run the matching 01 kernel first"
print("using checkpoint:", CKPT)

OUT = repo / "results" / RUN_NAME / "test"
"""
        ),
        md("## 3. Run standard + stratified evaluation on the test-15"),
        code(
            """
cmd = [sys.executable, str(repo / "scripts" / "evaluate.py"),
       "--config", str(repo / CONFIG), "--ckpt", str(CKPT),
       "--split", "test", "--out", str(OUT)]
# dataset root is auto-detected from the attached data; override with:
# cmd += ["--set", "data.root=/kaggle/input/<your-lol-dataset>/<subdir>"]

print(" ".join(cmd))
ret = subprocess.run(cmd, cwd=repo).returncode
assert ret == 0, f"evaluation failed (exit {ret})"
"""
        ),
        md("## 4. Headline table: per-stratum fidelity"),
        code(
            """
import csv, json
from pathlib import Path

summary = list(csv.DictReader(open(OUT / "stratified_summary.csv")))
print(f"{'stratum':>8} | {'PSNR (dB)':>10} | {'ΔE2000':>8} | {'images':>7}")
print("-" * 46)
for r in summary:
    print(f"{r['stratum']:>8} | {float(r['psnr']):>10.3f} | {float(r['delta_e2000']):>8.3f} | {r['n_images']:>7}")

metrics = json.loads((OUT / "metrics.json").read_text())
print("\\nstandard:", {k: round(v, 4) for k, v in metrics["standard"].items()})
"""
        ),
        md("## 5. Visual check: input / output / gate map side by side"),
        code(
            """
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

preds = sorted((OUT / "predictions").glob("*.png"))
print(len(preds), "enhanced images")

def find_pair(name, root_lol=None):
    from selgate.paths import find_lol_root
    from selgate.data import detect_layout
    root = find_lol_root(); layout = detect_layout(root)
    gt = root / layout[3] / name          # e.g. eval15/high/001.png
    lq = root / layout[2] / name          # e.g. eval15/low/001.png
    if not gt.exists():                   # extension fallback
        gt = next((root / layout[3] / (Path(name).stem + e) for e in (".png", ".jpg", ".jpeg")), None)
        lq = next((root / layout[2] / (Path(name).stem + e) for e in (".png", ".jpg", ".jpeg")), None)
    return lq, gt

n = min(3, len(preds))
fig, axes = plt.subplots(3, n, figsize=(5 * n, 12))
for i in range(n):
    name = preds[i].name
    lq, gt = find_pair(name)
    for row, (lbl, path) in enumerate([("input", lq), ("output", preds[i]), ("GT", gt)]):
        axes[row][i].imshow(np.asarray(Image.open(path).convert("RGB")))
        axes[row][i].set_title(f"{lbl}: {name}" if row == 0 else lbl)
        axes[row][i].axis("off")
plt.tight_layout(); plt.show()
"""
        ),
        md(
            """
## 6. (Optional) Cross-dataset generalisation — needs LIME/NPE/DICM attached

```python
cmd = [sys.executable, str(repo / "scripts" / "cross_dataset.py"),
       "--config", str(repo / CONFIG), "--ckpt", str(CKPT),
       "--input", "/kaggle/input/lime-images", "--name", "LIME",
       "--input", "/kaggle/input/npe-images",  "--name", "NPE",
       "--input", "/kaggle/input/dicom-images", "--name", "DICM",
       "--out", str(repo / "results" / RUN_NAME / "cross")]
subprocess.run(cmd, cwd=repo)
```

Requires `pip install pyiqa` (internet ON) for NIQE/BRISQUE/PI; the enhancement
step itself works without it.
"""
        ),
    ]


# template notebook (Path A users pick a config interactively)
write("02_evaluate.ipynb", eval_cells("configs/gate_pixel.yaml"))

# per-run eval kernels (Path B experiment matrix) — each imports ITS OWN
# train kernel's output so checkpoint selection is unambiguous
for _suffix, _cfg in RUNS:
    _slug = f"selgate-02-{_suffix}"
    save_kernel(_slug, _slug, "02_evaluate.ipynb",
                render(eval_cells(_cfg))[1],
                kernel_sources=[f"{KAGGLE_USER}/selgate-01-{_suffix}"])


# --------------------------------------------------------------------------
def gate_cells(config):
    """03 — gate interpretability: gate values inside the eval strata."""
    return [
        md(
            f"""
# 03 — Gate interpretability analysis (`{config}`)

**Hypothesis:** if the gate is an interpretable illumination selector, its
value should be *higher* (more enhanced output passes through) in
brighter / over-exposed regions and *lower* in dark regions, within each
test image's dark/mid/bright GT-luminance terciles (the same masks as the
stratified protocol).

Outputs (under `results/<run>/gate_analysis/`):
- `gate_stats_by_stratum.csv` — per image × stratum gate statistics
- scatter figure: stratum GT-luminance vs gate value (+ Pearson r)
- gate heatmap overlays for the first 5 test images
"""
        ),
        md("## 1. Environment (clone + install — same as 00, no secrets needed)"),
        code(
            CLONE_CELL + """
sys.path.insert(0, str(repo / "src"))   # PYTHONPATH only affects child processes, not this kernel
os.environ["PYTHONPATH"] = f"{repo / 'src'}:{os.environ.get('PYTHONPATH', '')}"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(repo / "requirements.txt")], check=True)

import torch
print("cuda:", torch.cuda.is_available())
"""
        ),
        md("## 2. Load model + locate test-15"),
        code(
            f"""
CONFIG = {config!r}
RUN_NAME = Path(CONFIG).stem

candidates = sorted(set(Path("/kaggle/working").rglob("best.pt"))
                    | set(Path("/kaggle/input").rglob("best.pt")))
matches = [p for p in candidates if RUN_NAME in str(p)]
if matches:
    CKPT = matches[0]
else:
    assert len(candidates) == 1, "ambiguous checkpoint selection: " + repr(candidates)
    CKPT = candidates[0]
print("using checkpoint:", CKPT)

import csv
import numpy as np
from PIL import Image
from selgate.config import load_config
from selgate.train import load_model_from_checkpoint
from selgate.paths import find_lol_root
from selgate.data import detect_layout
from selgate.eval.stratified import luminance, tercile_masks, STRATA

cfg = load_config(repo / CONFIG)
model = load_model_from_checkpoint(CKPT, cfg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()
print("model on", device, "| gate:", type(model.gate).__name__ if model.gate else None)

root = find_lol_root(); layout = detect_layout(root)
low_dir, high_dir = root / layout[2], root / layout[3]
print("low:", low_dir)
print("high:", high_dir)
"""
        ),
        md("## 3. Per-image gate statistics inside the eval strata"),
        code(
            """
import torch.nn.functional as F

def load01(p):
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float64) / 255.0

def find_low(name, d):
    p = d / name
    if p.exists():
        return p
    return next((d / (Path(name).stem + e) for e in (".png", ".jpg", ".jpeg")
                 if (d / (Path(name).stem + e)).exists()), None)

rows, gate_maps, samples = [], {}, {}
for hp in sorted(high_dir.iterdir()):
    if hp.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
        continue
    lp = find_low(hp.name, low_dir)
    if lp is None:
        print("WARNING: no low-light pair for", hp.name); continue

    gt = load01(hp); lq = load01(lp)
    x = torch.from_numpy(lq).float().permute(2, 0, 1)[None].to(device)
    with torch.no_grad():
        g = model(x)["gate"][0, 0]                    # (H', W')
    if tuple(g.shape) != tuple(x.shape[-2:]):         # patch/global granularity
        g = F.interpolate(g[None, None], size=x.shape[-2:], mode="bilinear")[0, 0]
    g = g.cpu().numpy()
    gate_maps[hp.name] = g
    if len(samples) < 5:
        samples[hp.name] = (lq, g)

    lum = luminance(gt)
    for s in (*STRATA, "overall"):
        m = np.ones_like(lum, dtype=bool) if s == "overall" else tercile_masks(lum)[s]
        rows.append(dict(image=hp.name, stratum=s, n_pixels=int(m.sum()),
                         gt_lum_mean=float(lum[m].mean()),
                         gate_mean=float(g[m].mean()), gate_std=float(g[m].std()),
                         frac_open=float((g[m] > 0.9).mean()),
                         frac_closed=float((g[m] < 0.1).mean())))
    print(hp.name, {s: round(rows[-4 + i]["gate_mean"], 3) for i, s in enumerate(STRATA)})

OUT3 = repo / "results" / RUN_NAME / "gate_analysis"
OUT3.mkdir(parents=True, exist_ok=True)
with open(OUT3 / "gate_stats_by_stratum.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("wrote", OUT3 / "gate_stats_by_stratum.csv", f"({len(rows)} rows)")
"""
        ),
        md("## 4. Aggregate: does the gate track region brightness?"),
        code(
            """
import matplotlib.pyplot as plt

print(f"{'stratum':>8} | {'gate_mean':>9} | {'gate_std':>8} | {'frac_open':>9} | {'frac_closed':>11} | {'gt_lum':>7}")
print("-" * 70)
for s in (*STRATA, "overall"):
    rs = [r for r in rows if r["stratum"] == s]
    print(f"{s:>8} | {np.mean([r['gate_mean'] for r in rs]):>9.3f} | "
          f"{np.mean([r['gate_std'] for r in rs]):>8.3f} | "
          f"{np.mean([r['frac_open'] for r in rs]):>9.3f} | "
          f"{np.mean([r['frac_closed'] for r in rs]):>11.3f} | "
          f"{np.mean([r['gt_lum_mean'] for r in rs]):>7.3f}")

sel = [r for r in rows if r["stratum"] != "overall"]
pts = np.array([[r["gt_lum_mean"], r["gate_mean"]] for r in sel])
r = float(np.corrcoef(pts[:, 0], pts[:, 1])[0, 1])
print(f"Pearson r (stratum GT luminance vs gate value) = {r:.3f}   [n={len(sel)} image-stratum points]")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for s, mk in zip(STRATA, ("o", "s", "^")):
    sub = [x for x in sel if x["stratum"] == s]
    ax.scatter([x["gt_lum_mean"] for x in sub], [x["gate_mean"] for x in sub],
               marker=mk, label=s, alpha=0.75)
ax.set_xlabel("stratum GT luminance (mean)"); ax.set_ylabel("gate value (mean)")
ax.set_title(f"gate vs region brightness — r={r:.3f}"); ax.legend()
plt.tight_layout(); plt.show()
"""
        ),
        md("## 5. Gate heatmaps: input | gate | output"),
        code(
            """
pred_dir = repo / "results" / RUN_NAME / "test" / "predictions"
names = sorted(samples)
fig, axes = plt.subplots(3, len(names), figsize=(4.2 * len(names), 12))
for i, nm in enumerate(names):
    lq, g = samples[nm]
    axes[0][i].imshow(lq); axes[0][i].set_title(f"input: {nm}" if i == 0 else nm)
    im = axes[1][i].imshow(g, cmap="viridis", vmin=0, vmax=1)
    axes[1][i].set_title("gate g" if i == 0 else "")
    plt.colorbar(im, ax=axes[1][i], fraction=0.046)
    out_p = pred_dir / nm
    if out_p.exists():
        axes[2][i].imshow(np.asarray(Image.open(out_p).convert("RGB")))
    axes[2][i].set_title("output" if i == 0 else "")
    for a in (axes[0][i], axes[1][i], axes[2][i]):
        a.axis("off")
plt.tight_layout(); plt.show()
"""
        ),
    ]


# template notebook + the analysis kernel (targets the headline gate model)
write("03_gate_analysis.ipynb", gate_cells("configs/gate_pixel.yaml"))

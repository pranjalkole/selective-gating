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
# notebook filename -> (kernel slug, kernel title)
KERNELS = {
    "00_kaggle_setup.ipynb": ("selgate-00-setup", "selgate-00-setup"),
    "01_train.ipynb": ("selgate-01-train", "selgate-01-train"),
    "02_evaluate.ipynb": ("selgate-02-evaluate", "selgate-02-evaluate"),
}

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


def write(name, cells):
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
    p = NB_DIR / name
    text = json.dumps(nb, indent=1)
    assert "YOUR_GITHUB_USERNAME" not in text, f"REPO_URL placeholder left in {name}"
    p.write_text(text)
    print("wrote", p)

    # Path B: per-kernel push directory with metadata
    slug, title = KERNELS[name]
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
        # 02 imports 01's output (checkpoints) so it can run in a fresh session
        "kernel_sources": (
            [f"{KAGGLE_USER}/selgate-01-train"] if name == "02_evaluate.ipynb" else []
        ),
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("wrote", kdir / name, "+ kernel-metadata.json")


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
write(
    "01_train.ipynb",
    [
        md(
            """
# 01 — Train Retinexformer + selective gate

**Run `00_kaggle_setup.ipynb` first** (this notebook re-creates the environment
in its first cell so it can also run standalone).

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
            """## 2. Choose config + run training

Start with `configs/smoke.yaml` (30 iters, ~1 min), then `configs/gate_pixel.yaml`.

Optional extras:
- fewer iters for a session budget: `OVERRIDES = ["--set", "train.iters=50000"]`
- warm-start / frozen-backbone ablation (weights dataset is auto-attached):
  `OVERRIDES = ["--set", "model.backbone.pretrained=/kaggle/input/retinexformer-weights/LOL_v1.pth"]`
"""
        ),
        code(
            """
CONFIG   = "configs/smoke.yaml"      # <-- start here (fast). Then: "configs/gate_pixel.yaml"
OVERRIDES = []                        # e.g. ["--set", "train.iters=50000"]

cmd = [sys.executable, str(repo / "scripts" / "train.py"), "--config", str(repo / CONFIG), *OVERRIDES]
print(" ".join(cmd))
ret = subprocess.run(cmd, cwd=repo).returncode
assert ret == 0, f"training failed (exit {ret})"
"""
        ),
        md("## 3. Training curves + gate monitor"),
        code(
            """
import json
from pathlib import Path

name = Path(CONFIG).stem
if name == "smoke":
    name = "smoke"
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
    ],
)

# --------------------------------------------------------------------------
write(
    "02_evaluate.ipynb",
    [
        md(
            """
# 02 — Evaluation: standard + **region-stratified** protocol

Runs on the LOL-v1 **test-15** (never used for tuning):

| Level | What | Metrics |
|---|---|---|
| 1. Standard | test-15 | PSNR / SSIM / LPIPS |
| 2. **Region-stratified** | dark/mid/bright GT-luminance terciles | PSNR + ΔE2000 **per stratum** |
| 3. Naturalness | LOL-v1 | LOE |
| 4. Cross-dataset (optional) | LIME / NPE / DICM | NIQE / BRISQUE / PI |

**Prerequisites:** `01` has run at least once — its output (checkpoints) is
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
            """
CONFIG = "configs/gate_pixel.yaml"   # <-- must match the run you trained in 01
RUN_NAME = Path(CONFIG).stem

candidates = sorted(set(Path("/kaggle/working").rglob("best.pt"))
                    | set(Path("/kaggle/input").rglob("best.pt")))
CKPT = next((p for p in candidates if RUN_NAME in str(p)), candidates[-1] if candidates else None)
assert CKPT is not None, "no best.pt found — run 01_train first (its output is imported here)"
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
    ],
)

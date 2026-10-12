#!/usr/bin/env python3
"""Full evaluation on LOL-v1 (test-15 or the carved val split).

    python scripts/evaluate.py --config configs/gate_pixel.yaml \
        --ckpt runs/gate_pixel/best.pt --split test --out results/gate_pixel

Produces:
    <out>/predictions/*.png          enhanced images (unless --skip-inference)
    <out>/standard_metrics.csv       PSNR / SSIM / LPIPS (+ LOE vs. input)
    <out>/stratified_per_image.csv   per-image × per-stratum PSNR + ΔE2000
    <out>/stratified_summary.csv     dark/mid/bright/overall summary (headline table)
    <out>/metrics.json               everything, machine-readable
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selgate.config import load_config  # noqa: E402
from selgate.data import LOLPairedDataset, make_split  # noqa: E402
from selgate.eval.loe import loe  # noqa: E402
from selgate.eval.standard import evaluate_pair  # noqa: E402
from selgate.eval.stratified import run_stratified_eval  # noqa: E402
from selgate.paths import find_lol_root, split_file  # noqa: E402
from selgate.train import load_model_from_checkpoint  # noqa: E402


@torch.no_grad()
def run_inference(model, ds: LOLPairedDataset, pred_dir: Path, device: torch.device) -> List[Dict]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    rows: List[Dict] = []
    for i in range(len(ds)):
        lq, gt, name = ds[i]
        out = model(lq.unsqueeze(0).to(device))["out"].squeeze(0).clamp(0, 1).cpu()
        arr = (out.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
        from PIL import Image

        Image.fromarray(arr).save(pred_dir / name)
        m = evaluate_pair(out, gt, with_lpips=True)
        m_loe = loe(lq.numpy().transpose(1, 2, 0), out.numpy().transpose(1, 2, 0))
        rows.append({"image": name, **m, "loe": m_loe})
        print(f"[{i + 1}/{len(ds)}] {name}: " +
              " ".join(f"{k}={v:.4f}" for k, v in {**m, "loe": m_loe}.items()))
    return rows


def _write_csv(rows: List[Dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Standard + region-stratified evaluation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--out", default=None, help="output dir (default: runs/<name>/results_<split>)")
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--device", default=None)
    ap.add_argument("--skip-inference", action="store_true",
                    help="only (re)score existing <out>/predictions")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    from selgate.paths import out_dir as _out_dir

    out_dir = Path(args.out) if args.out else _out_dir(cfg.get_path("name", "run")) / f"results_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"

    root = find_lol_root(cfg.get_path("data.root"))
    if root is None:
        sys.exit("LOL-v1 not found (data.root / SELGATE_LOL_ROOT / Kaggle attach)")

    if args.split == "test":
        from selgate.data import detect_layout

        layout = detect_layout(root)
        gt_dir, lq_dir = root / layout[3], root / layout[2]
        names = sorted(p.name for p in gt_dir.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
        ds = LOLPairedDataset(root, names, split="test")
    else:
        # reuse the committed split if present, else regenerate deterministically
        split = make_split(root,
                           val_fraction=float(cfg.get_path("data.val_fraction", 0.05)),
                           seed=int(cfg.get_path("data.split_seed", 0)),
                           out_file=Path(cfg.get_path("data.split_file") or split_file()))
        ds = LOLPairedDataset(root, split["val"], split="train")
        # gt dir for stratified eval: materialise references
        gt_dir = out_dir / "references"
        gt_dir.mkdir(exist_ok=True)
        from PIL import Image

        for i in range(len(ds)):
            _, gt, name = ds[i]
            Image.fromarray((gt.permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)).save(gt_dir / name)

    if not args.skip_inference:
        model = load_model_from_checkpoint(args.ckpt, cfg)
        rows = run_inference(model, ds, pred_dir, device)
        _write_csv(rows, out_dir / "standard_metrics.csv")
    else:
        with open(out_dir / "standard_metrics.csv") as f:
            rows = list(csv.DictReader(f))

    # ---- aggregate standard metrics
    summary_std = {}
    for key in ("psnr", "ssim", "lpips", "loe"):
        vals = [float(r[key]) for r in rows if key in r and r[key] not in ("", "None")]
        if vals:
            summary_std[key] = float(np.mean(vals))
    print("[selgate] standard:", {k: round(v, 4) for k, v in summary_std.items()})

    # ---- region-stratified protocol (the contribution)
    strat = run_stratified_eval(gt_dir, pred_dir, out_dir, prefix="stratified")
    print("[selgate] stratified (mean over images):")
    for s, v in strat.items():
        print(f"    {s:>7}: PSNR {v['psnr']:.3f} dB | ΔE2000 {v['delta_e2000']:.3f}")

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"split": args.split, "ckpt": str(args.ckpt), "standard": summary_std,
                   "stratified": strat}, f, indent=2)
    print(f"[selgate] results written to {out_dir}")


if __name__ == "__main__":
    main()

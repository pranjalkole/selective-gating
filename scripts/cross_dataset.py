#!/usr/bin/env python3
"""Cross-dataset generalisation: enhance unpaired sets and score them.

    python scripts/cross_dataset.py --config configs/gate_pixel.yaml \
        --ckpt runs/gate_pixel/best.pt \
        --input data/LIME --name LIME \
        --input data/NPE --name NPE \
        --input data/DICM --name DICM \
        --out results/cross_dataset

Writes <out>/enhanced/<name>/*.png and <out>/no_ref_scores.csv (NIQE/BRISQUE/PI
if pyiqa is installed; the enhancement step always runs).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selgate.config import load_config  # noqa: E402
from selgate.eval.cross_dataset import enhance_folder, no_ref_scores, write_scores_csv  # noqa: E402
from selgate.train import load_model_from_checkpoint  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-dataset no-reference evaluation")
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--input", action="append", required=True,
                    help="input image folder (repeatable)")
    ap.add_argument("--name", action="append", default=None,
                    help="dataset name per --input (same order), default: dirname")
    ap.add_argument("--out", default="results/cross_dataset")
    ap.add_argument("--metrics", default="niqe,brisque,pi")
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config, args.overrides)
    model = load_model_from_checkpoint(args.ckpt, cfg)

    names = args.name or [Path(p).name for p in args.input]
    if len(names) != len(args.input):
        sys.exit("--name count must match --input count")

    out = Path(args.out)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    scores = {}
    for name, inp in zip(names, args.input):
        print(f"[selgate] enhancing {name} from {inp}")
        enhanced = enhance_folder(model, inp, out / "enhanced" / name, device=device)
        s = no_ref_scores(enhanced, metrics=metrics, device=device)
        scores[name] = s
        print(f"[selgate] {name}: {s}")

    csv_path = write_scores_csv(scores, out / "no_ref_scores.csv")
    print(f"[selgate] wrote {csv_path}")


if __name__ == "__main__":
    main()

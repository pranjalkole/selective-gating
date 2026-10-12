#!/usr/bin/env python3
"""Train: python scripts/train.py --config configs/gate_pixel.yaml [--set k=v ...]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selgate.config import load_config  # noqa: E402
from selgate.train import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Retinexformer + selective gate")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="KEY=VALUE", help="dot-key override, repeatable "
                    "(e.g. --set train.iters=1000 --set model.gate.granularity=patch)")
    ap.add_argument("--resume", default=None, help="checkpoint to resume from")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    ckpt = train(cfg, resume=args.resume)
    print(f"[selgate] checkpoint: {ckpt}")


if __name__ == "__main__":
    main()

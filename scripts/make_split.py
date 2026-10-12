#!/usr/bin/env python3
"""Create (or verify) the train/val split carved from the 485 train images.

    python scripts/make_split.py [--root data/LOL-v1] [--val-fraction 0.05] [--seed 0]

Writes configs/split_lol_v1.json and asserts the test-15 never leaks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from selgate.data import make_split  # noqa: E402
from selgate.paths import find_lol_root, split_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None, help="LOL-v1 root (auto-detect if omitted)")
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(split_file()))
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    args = ap.parse_args()

    root = find_lol_root(args.root)
    if root is None:
        sys.exit("LOL-v1 not found. Pass --root or set SELGATE_LOL_ROOT.")

    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} already exists; verifying it instead (use --force to regenerate)")
        with open(out) as f:
            split = json.load(f)
        test = set(split.get("test", []))
        leaked = (set(split["train"]) | set(split["val"])) & test
        if leaked:
            sys.exit(f"LEAKAGE: {sorted(leaked)}")
        print(f"OK: {len(split['train'])} train / {len(split['val'])} val / {len(test)} test, "
              "no leakage")
        return

    split = make_split(root, val_fraction=args.val_fraction, seed=args.seed,
                       out_file=out, force=True)
    print(f"wrote {out}: {len(split['train'])} train / {len(split['val'])} val / "
          f"{len(split['test'])} test")
    print(f"val images: {split['val'][:5]} ...")


if __name__ == "__main__":
    main()

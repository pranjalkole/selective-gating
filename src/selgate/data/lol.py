"""LOL-v1 dataset loading, layout detection, and the train/val split.

Hard rule (benchmark-overfitting guard): the 15 test images are NEVER part of
the train or validation split. Validation is carved from the 485 train images
only, deterministically, and the split is persisted to a JSON file.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# (train_low, train_high, test_low, test_high) relative to the dataset root
_LAYOUTS = [
    ("our485/low", "our485/high", "eval15/low", "eval15/high"),      # original LOL-v1
    ("our485/input", "our485/target", "eval15/input", "eval15/target"),
    ("Train/input", "Train/target", "Test/input", "Test/target"),     # Retinexformer layout
    ("train/low", "train/high", "test/low", "test/high"),
    ("train/input", "train/target", "test/input", "test/target"),
    ("low", "high", "test/low", "test/high"),                         # single-dir + test/
]

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def detect_layout(root: Path) -> Tuple[str, str, str, str]:
    """Return (train_low, train_high, test_low, test_high) dirs relative to root."""
    root = Path(root)
    for layout in _LAYOUTS:
        if (root / layout[0]).is_dir() and (root / layout[1]).is_dir():
            # test dirs may be absent (e.g. an unpaired folder); caller decides
            return layout
    raise FileNotFoundError(
        f"Unrecognised LOL-v1 layout under {root}. Expected one of: "
        "our485/{low,high}+eval15/{low,high}, Train/{input,target}+Test/..., "
        "train/{low,high}+test/{low,high}"
    )


def has_test_split(root: Path) -> bool:
    layout = detect_layout(root)
    return (Path(root) / layout[2]).is_dir() and (Path(root) / layout[3]).is_dir()


def _list_images(d: Path) -> List[str]:
    return sorted(p.name for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def make_split(
    root: Path,
    val_fraction: float = 0.05,
    seed: int = 0,
    out_file: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, List[str]]:
    """Carve a validation split from the *train* images only.

    Deterministic given (sorted filenames, val_fraction, seed). Asserts that
    no test-15 filename leaks into train or val.
    """
    root = Path(root)
    layout = detect_layout(root)
    train_low, train_high, test_low, test_high = layout

    train_names = _list_images(root / train_low)
    test_names = _list_images(root / test_high) if (root / test_high).is_dir() else []

    if out_file and out_file.exists() and not force:
        with open(out_file) as f:
            split = json.load(f)
        _assert_no_leakage(split, test_names, out_file)
        return split

    rng = random.Random(seed)
    names = list(train_names)
    rng.shuffle(names)  # rng-derived order, stable for a fixed seed & file list
    n_val = max(1, int(round(len(names) * val_fraction)))
    val_names = sorted(names[:n_val])
    train_names_out = sorted(names[n_val:])

    split = {"seed": seed, "val_fraction": val_fraction,
             "train": train_names_out, "val": val_names, "test": sorted(test_names)}
    _assert_no_leakage(split, test_names, "in-memory split")

    if out_file:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w") as f:
            json.dump(split, f, indent=2)
    return split


def _assert_no_leakage(split: Dict[str, List[str]], test_names: Sequence[str], where) -> None:
    test_set = set(test_names) | set(split.get("test", []))
    leaked = (set(split["train"]) | set(split["val"])) & test_set
    if leaked:
        raise AssertionError(f"TEST-15 LEAKAGE via {where}: {sorted(leaked)}")
    overlap = set(split["train"]) & set(split["val"])
    if overlap:
        raise AssertionError(f"train/val overlap: {sorted(overlap)}")


def load_split(path: Path) -> Dict[str, List[str]]:
    with open(path) as f:
        return json.load(f)


def read_image(path: Path) -> torch.Tensor:
    """PIL -> float32 tensor (C,H,W) in [0,1]."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class LOLPairedDataset(Dataset):
    """Paired low/normal-light images with optional random crop + flips."""

    def __init__(
        self,
        root: Path,
        names: Sequence[str],
        split: str = "train",
        crop_size: Optional[int] = None,
        augment: bool = False,
    ):
        root = Path(root)
        layout = detect_layout(root)
        if split == "train":
            low_dir, high_dir = root / layout[0], root / layout[1]
        else:
            low_dir, high_dir = root / layout[2], root / layout[3]
        if not low_dir.is_dir():
            raise FileNotFoundError(f"{low_dir} not found")

        self.pairs: List[Tuple[Path, Path]] = []
        missing = []
        for n in names:
            lp, hp = low_dir / n, high_dir / n
            # allow differing extensions between low/high dirs
            if not hp.exists():
                cand = [high_dir / (Path(n).stem + e) for e in IMG_EXTS]
                hp = next((c for c in cand if c.exists()), hp)
            if not lp.exists() or not hp.exists():
                missing.append(n)
                continue
            self.pairs.append((lp, hp))
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing under {low_dir}/{high_dir}: {missing[:5]}")

        self.crop_size = crop_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        lq, gt = read_image(self.pairs[idx][0]), read_image(self.pairs[idx][1])

        if self.crop_size:
            _, h, w = gt.shape
            cs = self.crop_size
            if h < cs or w < cs:  # pad reflect if the image is smaller than the crop
                pad_h, pad_w = max(0, cs - h), max(0, cs - w)
                lq = torch.nn.functional.pad(lq, (0, pad_w, 0, pad_h), mode="reflect")
                gt = torch.nn.functional.pad(gt, (0, pad_w, 0, pad_h), mode="reflect")
                _, h, w = gt.shape
            i = random.randint(0, h - cs)
            j = random.randint(0, w - cs)
            lq, gt = lq[:, i:i + cs, j:j + cs], gt[:, i:i + cs, j:j + cs]
            if self.augment:
                if random.random() < 0.5:
                    lq, gt = torch.flip(lq, [2]), torch.flip(gt, [2])
                if random.random() < 0.5:
                    lq, gt = torch.flip(lq, [1]), torch.flip(gt, [1])

        return lq, gt, self.pairs[idx][0].name


class UnpairedFolderDataset(Dataset):
    """A flat folder of low-light images (LIME / NPE / DICM style)."""

    def __init__(self, root: Path):
        root = Path(root)
        self.files = sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
        if not self.files:
            raise FileNotFoundError(f"no images found under {root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        return read_image(self.files[idx]), self.files[idx].name

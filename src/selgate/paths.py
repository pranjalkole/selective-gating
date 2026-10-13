"""Environment-aware path resolution.

The same code must run in three places:

1. a local dev checkout        -> ``<repo>/data/...``, ``<repo>/runs/...``
2. a Kaggle notebook           -> ``/kaggle/input/<dataset>/...``, ``/kaggle/working/...``
3. anywhere via env overrides   -> ``SELGATE_LOL_ROOT``, ``SELGATE_OUT_DIR``

Nothing else in the package hard-codes a filesystem path.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")

# directory names that mark a LOL-v1 layout inside a data root
_LOL_MARKERS = (
    ("our485", "low"),
    ("our485", "input"),
    ("Train", "input"),
    ("train", "low"),
    ("eval15", "low"),
    ("Test", "input"),
    ("test", "low"),
)


def is_kaggle() -> bool:
    """True when running inside a Kaggle notebook kernel."""
    if KAGGLE_INPUT.exists():
        return True
    return os.environ.get("KAGGLE_KERNEL_RUNNING", "").lower() in ("true", "1", "yes")


def repo_root() -> Path:
    """Root of the selective-gating checkout (contains pyproject.toml)."""
    here = Path(__file__).resolve()
    candidate = here.parents[2]  # src/selgate/paths.py -> repo root
    if (candidate / "pyproject.toml").exists():
        return candidate
    # installed as a site-package: walk up from cwd as a fallback
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "pyproject.toml").exists() and (p / "src" / "selgate").exists():
            return p
    return candidate


def _looks_like_lol_root(root: Path) -> bool:
    return any((root / m1 / m2).is_dir() for m1, m2 in _LOL_MARKERS)


def _iter_dataset_dirs(base: Path, max_depth: int = 6) -> Iterable[Path]:
    """Yield dataset directories under /kaggle/input up to max_depth.

    Depth 6 because Kaggle's mount layout nests: /kaggle/input (or
    /kaggle/input/datasets/<owner>/<slug>) around the actual data root.
    """
    if not base.is_dir():
        return
    yield base
    frontier = [base]
    for _ in range(max_depth):
        nxt = []
        for d in frontier:
            try:
                for child in d.iterdir():
                    if child.is_dir():
                        yield child
                        nxt.append(child)
            except PermissionError:
                continue
        frontier = nxt


def find_lol_root(explicit: Optional[str] = None) -> Optional[Path]:
    """Locate the LOL-v1 dataset root.

    Resolution order:
      1. explicit argument
      2. ``SELGATE_LOL_ROOT`` env var
      3. ``<repo>/data/LOL-v1`` (local)
      4. any dataset attached under ``/kaggle/input`` with a LOL layout
    """
    if explicit:
        p = Path(explicit).expanduser()
        return p if _looks_like_lol_root(p) else p  # trust explicit even if unusual

    env = os.environ.get("SELGATE_LOL_ROOT")
    if env:
        return Path(env).expanduser()

    local = repo_root() / "data" / "LOL-v1"
    if _looks_like_lol_root(local):
        return local

    if is_kaggle():
        for d in _iter_dataset_dirs(KAGGLE_INPUT):
            if _looks_like_lol_root(d):
                return d
    return None


def split_file() -> Path:
    """Committed train/val split (carved from the 485 train images only)."""
    return repo_root() / "configs" / "split_lol_v1.json"


def out_dir(run_name: str, override: Optional[str] = None) -> Path:
    """Where checkpoints/metrics go.

    On Kaggle everything must land in ``/kaggle/working`` so it survives as
    notebook output and can be downloaded / imported by downstream notebooks.
    """
    if override:
        p = Path(override) / run_name
    elif os.environ.get("SELGATE_OUT_DIR"):
        p = Path(os.environ["SELGATE_OUT_DIR"]) / run_name
    elif is_kaggle():
        p = KAGGLE_WORKING / "runs" / run_name
    else:
        p = repo_root() / "runs" / run_name
    p.mkdir(parents=True, exist_ok=True)
    return p

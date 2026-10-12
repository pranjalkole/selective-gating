"""Cross-dataset generalisation: enhance unpaired sets (LIME / NPE / DICM) and
score them with no-reference metrics (NIQE / BRISQUE / PI).

No-reference metrics come from the optional ``pyiqa`` package; if it is not
installed we skip them with a clear message (everything else still works).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from ..data import UnpairedFolderDataset

NR_METRICS = ("niqe", "brisque", "pi")


@torch.no_grad()
def enhance_folder(model, folder: str | Path, out_dir: str | Path, device: str = "cpu",
                   max_size: int = 1024) -> Path:
    """Run the model over every image in ``folder``; save PNGs to ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = UnpairedFolderDataset(folder)
    model = model.to(device).eval()

    for i in range(len(ds)):
        img, name = ds[i]
        # keep memory sane on huge inputs: downscale if needed
        _, h, w = img.shape
        scale = min(1.0, max_size / max(h, w))
        if scale < 1.0:
            nh, nw = int(h * scale) // 2 * 2, int(w * scale) // 2 * 2
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False
            ).squeeze(0)
        out = model(img.unsqueeze(0).to(device))["out"].squeeze(0).clamp(0, 1).cpu()
        arr = (out.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / (Path(name).stem + ".png"))
        if (i + 1) % 20 == 0:
            print(f"[selgate] enhanced {i + 1}/{len(ds)}")
    return out_dir


def no_ref_scores(
    image_dir: str | Path,
    metrics: Sequence[str] = NR_METRICS,
    device: str = "cpu",
) -> Dict[str, float]:
    """Mean NIQE / BRISQUE / PI over a folder of images (via pyiqa)."""
    image_dir = Path(image_dir)
    files = sorted(p for p in image_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp"))
    if not files:
        raise RuntimeError(f"no images in {image_dir}")

    try:
        import pyiqa
    except Exception as e:  # noqa: BLE001
        print(f"[selgate] pyiqa unavailable ({e}); skipping no-reference metrics. "
              "Install with: pip install pyiqa")
        return {}

    table: Dict[str, List[float]] = {}
    for m in metrics:
        if m not in NR_METRICS:
            print(f"[selgate] unknown metric {m!r}, skipping")
            continue
        metric_fn = pyiqa.create_metric(m, device=device)
        vals = []
        for f in files:
            try:
                vals.append(float(metric_fn(str(f))))
            except Exception as e:  # noqa: BLE001
                print(f"[selgate] {m} failed on {f.name}: {e}")
        if vals:
            table[m] = vals

    summary = {m: float(np.mean(v)) for m, v in table.items()}
    summary["n_images"] = len(files)
    return summary


def write_scores_csv(scores_by_set: Dict[str, Dict[str, float]], out_csv: str | Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metric_cols = sorted({m for s in scores_by_set.values() for m in s if m != "n_images"})
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", *metric_cols, "n_images"])
        for name, s in scores_by_set.items():
            w.writerow([name, *[f"{s.get(m, float('nan')):.4f}" for m in metric_cols],
                        s.get("n_images", "")])
    return out_csv

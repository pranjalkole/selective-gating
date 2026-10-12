"""Thin wrapper that builds the vendored Retinexformer backbone.

The official implementation lives in ``third_party/Retinexformer`` (MIT). We
load the architecture file directly by path so we don't depend on the whole
``basicsr`` package (which pulls in cv2 and friends).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn as nn

from ..paths import repo_root

_ARCH_REL = Path("third_party/Retinexformer/basicsr/models/archs/RetinexFormer_arch.py")

_cached_cls = None


def _arch_cls():
    """Import RetinexFormer (and RetinexFormer_Single_Stage) from the vendored file."""
    global _cached_cls
    if _cached_cls is not None:
        return _cached_cls

    import importlib.util

    arch_path = repo_root() / _ARCH_REL
    if not arch_path.exists():
        raise FileNotFoundError(
            f"Vendored Retinexformer not found at {arch_path}. "
            "Run: git submodule / clone caiyuanhao1998/Retinexformer into third_party/"
        )
    spec = importlib.util.spec_from_file_location("selgate_retinexformer_arch", arch_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _cached_cls = mod
    return mod


def build_backbone(
    n_feat: int = 40,
    stage: int = 1,
    num_blocks: Sequence[int] = (1, 2, 2),
    pretrained: Optional[str] = None,
    strict: bool = False,
) -> nn.Module:
    """Build RetinexFormer with the official LOL-v1 hyperparameters by default."""
    mod = _arch_cls()
    model = mod.RetinexFormer(
        in_channels=3, out_channels=3, n_feat=n_feat, stage=stage, num_blocks=list(num_blocks)
    )
    if pretrained:
        load_checkpoint(model, pretrained, strict=strict)
    return model


def _strip_prefix(sd: dict) -> dict:
    """Tolerate common checkpoint key prefixes from basicsr / DDP / lightning."""
    out = {}
    for k, v in sd.items():
        for pfx in ("module.", "params_g.", "model.", "net_g.", "generator."):
            if k.startswith(pfx):
                k = k[len(pfx):]
        out[k] = v
    return out


def load_checkpoint(model: nn.Module, path: str | Path, strict: bool = False) -> nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("params_g", "params", "state_dict", "model", "net_g"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
            break
    sd = _strip_prefix(ckpt if isinstance(ckpt, dict) else {})
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        print(f"[selgate] checkpoint {path}: missing={len(missing)} unexpected={len(unexpected)}")
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

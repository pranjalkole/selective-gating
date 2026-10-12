"""Training loop: joint (or frozen-backbone) training of backbone + gate."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config
from .data import LOLPairedDataset, load_split, make_split
from .eval.standard import psnr
from .losses import SelGateCriterion
from .models import GateNet, SelGateModel, build_backbone, gate_stats
from .paths import find_lol_root, out_dir, split_file


# --------------------------------------------------------------------- building
def build_model(cfg: Config, pretrained: Optional[str] = None) -> SelGateModel:
    b = cfg.get_path("model.backbone", {}) or {}
    backbone = build_backbone(
        n_feat=b.get("n_feat", 40),
        stage=b.get("stage", 1),
        num_blocks=b.get("num_blocks", (1, 2, 2)),
        pretrained=pretrained or b.get("pretrained"),
    )
    g = cfg.get_path("model.gate", {}) or {}
    gate = None
    if g.get("enabled", True):
        gate = GateNet(
            granularity=g.get("granularity", "pixel"),
            width=g.get("width", 32),
            depth=g.get("depth", 4),
            illum_mode=g.get("illum", "max"),
            stride=g.get("stride", 8),
        )
    return SelGateModel(backbone, gate, freeze_backbone=bool(cfg.get_path("model.freeze_backbone", False)))


def load_model_from_checkpoint(ckpt_path: str | Path, cfg: Optional[Config] = None) -> SelGateModel:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = cfg or Config.from_dict(ckpt.get("cfg", {}))
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"], strict=False)
    return model


# ---------------------------------------------------------------------- helpers
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(cfg: Config) -> torch.device:
    pref = cfg.get_path("train.device", None)
    if pref:
        return torch.device(pref)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def validate(model, loader, device, max_images: Optional[int] = None) -> Dict[str, float]:
    model.eval()
    vals = []
    n = 0
    for lq, gt, _ in loader:
        out = model(lq.to(device))["out"].clamp(0, 1).cpu()
        for k in range(out.shape[0]):
            vals.append(psnr(out[k], gt[k]))
            n += 1
            if max_images and n >= max_images:
                break
        if max_images and n >= max_images:
            break
    model.train()
    return {"val_psnr": float(np.mean(vals)) if vals else float("nan"), "n": len(vals)}


# ------------------------------------------------------------------------- train
def train(cfg: Config, resume: Optional[str] = None) -> Path:
    """Train per config; returns the path of the best checkpoint."""
    seed_everything(int(cfg.get_path("seed", 0)))
    device = _device(cfg)
    run_dir = out_dir(cfg.get_path("name", "run"), cfg.get_path("train.out_dir"))
    print(f"[selgate] run dir: {run_dir} | device: {device}")

    # ---------------------------------------------------------------- data
    root = find_lol_root(cfg.get_path("data.root"))
    if root is None:
        raise FileNotFoundError(
            "LOL-v1 not found. Set data.root (config) or SELGATE_LOL_ROOT, "
            "or attach the dataset on Kaggle."
        )
    split_path = Path(cfg.get_path("data.split_file") or split_file())
    split = make_split(
        root,
        val_fraction=float(cfg.get_path("data.val_fraction", 0.05)),
        seed=int(cfg.get_path("data.split_seed", 0)),
        out_file=split_path,
    )
    print(f"[selgate] split: {len(split['train'])} train / {len(split['val'])} val / "
          f"{len(split.get('test', []))} test (test never used for tuning)")

    crop = cfg.get_path("data.crop_size", None)
    train_ds = LOLPairedDataset(root, split["train"], split="train",
                                crop_size=crop, augment=bool(cfg.get_path("data.augment", True)))
    val_ds = LOLPairedDataset(root, split["val"], split="train", crop_size=None, augment=False)

    bs = int(cfg.get_path("data.batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=int(cfg.get_path("data.num_workers", 2)),
                              drop_last=True, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            num_workers=int(cfg.get_path("data.num_workers", 2)))

    # --------------------------------------------------------------- model
    model = build_model(cfg).to(device)
    from .models import count_params

    n_backbone = count_params(model.backbone)
    n_gate = count_params(model.gate) if model.gate else 0
    print(f"[selgate] backbone params: {n_backbone:,} | gate params: {n_gate:,} | "
          f"freeze_backbone={model.freeze_backbone}")

    crit_cfg = cfg.get_path("loss", {}) or {}
    criterion = SelGateCriterion(
        ssim_weight=float(crit_cfg.get("ssim_weight", 0.2)),
        tv_lambda=float(crit_cfg.get("tv_lambda", 0.01)),
        illum_lambda=float(crit_cfg.get("illum_lambda", 0.1)),
        illum_mode=crit_cfg.get("illum_mode", "bce"),
        gate_illum_mode=crit_cfg.get("gate_illum_mode", cfg.get_path("model.gate.illum", "max")),
    ).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(cfg.get_path("train.lr", 2e-4))
    optim = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    total_iter = int(cfg.get_path("train.iters", 5000))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_iter,
                                                       eta_min=float(cfg.get_path("train.lr_min", 1e-6)))
    use_amp = bool(cfg.get_path("train.amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    grad_clip = cfg.get_path("train.grad_clip", 1.0)

    log_every = int(cfg.get_path("train.log_every", 50))
    val_every = int(cfg.get_path("train.val_every", 500))
    max_val_images = cfg.get_path("train.max_val_images", None)

    start_iter, best_psnr, history = 0, -1.0, []
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"], strict=False)
        if "optimizer" in ck:
            optim.load_state_dict(ck["optimizer"])
        if "scheduler" in ck:
            sched.load_state_dict(ck["scheduler"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_iter, best_psnr = ck.get("iter", 0), ck.get("best_psnr", -1.0)
        print(f"[selgate] resumed from {resume} @ iter {start_iter}")

    # ------------------------------------------------------------- loop
    model.train()
    data_iter = iter(train_loader)
    t0 = time.time()
    last_ckpt = run_dir / "last.pt"

    for it in range(start_iter + 1, total_iter + 1):
        try:
            lq, gt, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            lq, gt, _ = next(data_iter)
        lq, gt = lq.to(device, non_blocking=True), gt.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            res = model(lq)
            losses = criterion(res["out"], gt, lq, res.get("gate"), res.get("gate_logits"))
        scaler.scale(losses["total"]).backward()
        if grad_clip:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
        scaler.step(optim)
        scaler.update()
        sched.step()

        if it % log_every == 0 or it == 1:
            gstats = gate_stats(res["gate"]) if model.gate else {}
            lr_now = optim.param_groups[0]["lr"]
            msg = (f"iter {it}/{total_iter} | "
                   + " ".join(f"{k}={float(v.detach()):.4f}" for k, v in losses.items())
                   + f" | lr={lr_now:.2e}"
                   + (" | " + " ".join(f"{k}={v:.3f}" for k, v in gstats.items()) if gstats else ""))
            print(msg)
            history.append({"iter": it, **{k: float(v.detach()) for k, v in losses.items()},
                            "lr": lr_now, **gstats})

        if it % val_every == 0 or it == total_iter:
            metrics = validate(model, val_loader, device, max_val_images)
            print(f"[selgate] val @ {it}: {metrics}")
            ckpt = {"model": model.state_dict(), "cfg": cfg.to_dict(), "iter": it,
                    "val_psnr": metrics["val_psnr"], "best_psnr": max(best_psnr, metrics["val_psnr"]),
                    "optimizer": optim.state_dict(), "scheduler": sched.state_dict(),
                    "scaler": scaler.state_dict()}
            torch.save(ckpt, last_ckpt)
            if metrics["val_psnr"] > best_psnr:
                best_psnr = metrics["val_psnr"]
                torch.save(ckpt, run_dir / "best.pt")
                print(f"[selgate] new best val PSNR {best_psnr:.3f} dB -> best.pt")
            with open(run_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)

    print(f"[selgate] done in {(time.time() - t0) / 60:.1f} min | best val PSNR {best_psnr:.3f} dB")
    return last_ckpt

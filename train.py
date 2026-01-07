"""Minimal trainer for TxUNet radio map prediction."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from data.dataset import IndoorRadioMapDataset, gather_task2_samples, parse_meta
from losses.l1_rmse import L1LossMasked, rmse
from models.radio_unet_tx import TxUNet

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def load_yaml(path: str | Path) -> Dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def split_by_building(pairs: List, train_buildings: List[int], val_buildings: List[int]) -> Tuple[List, List]:
    """Split samples by building ID."""
    train_set = set(train_buildings)
    val_set = set(val_buildings)
    train, val = [], []
    for sp in pairs:
        meta = parse_meta(Path(sp.input_path).name)
        b = meta.get("building")
        if b in train_set:
            train.append(sp)
        elif b in val_set:
            val.append(sp)
    return train, val


def build_model(cfg: Dict) -> nn.Module:
    """Build TxUNet from config."""
    return TxUNet(
        in_ch=cfg.get("in_ch", 3),
        out_ch=cfg.get("out_ch", 1),
        base_ch=cfg.get("base_ch", 48),
        depths=tuple(cfg.get("depths", (4, 6, 6, 8))),
        heads=tuple(cfg.get("heads", (4, 4, 8, 8))),
        expand=float(cfg.get("expand", 2.66)),
        use_checkpoint=cfg.get("use_checkpoint", True),
        ln_eps=float(cfg.get("ln_eps", 1e-5)),
        window0=cfg.get("window0", None),
        window0_stride=cfg.get("window0_stride", None),
    )


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    writer=None,
    log_interval: int = 50,
    global_step: int = 0,
    clip_norm: float | None = None,
) -> tuple[float, int]:
    """Train for one epoch. Returns (average loss, updated global_step)."""
    model.train()
    total_loss = 0.0
    
    for i, (x, y, mask, _) in enumerate(loader):
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        
        # Pad input channels if model expects more
        model_in_ch = model.stem.in_channels
        if x.shape[1] < model_in_ch:
            pad = torch.zeros(x.shape[0], model_in_ch - x.shape[1], x.shape[2], x.shape[3], device=device)
            x = torch.cat([x, pad], dim=1)
        
        optimizer.zero_grad()
        
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=amp_enabled and device.type == "cuda"):
            pred = model(x)
            loss = criterion(pred, y, mask)
        
        scaler.scale(loss).backward()
        if clip_norm is not None and clip_norm > 0.0:
            # Unscale before clipping to avoid fp16/bf16 overflow
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if not torch.isfinite(grad_norm):
                print(f"Step {global_step}: non-finite grad norm ({grad_norm}), skipping step")
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        global_step += 1

        if log_interval > 0 and global_step % log_interval == 0:
            print(f"Step {global_step}: train L1={loss.item():.4f}")
            if writer:
                writer.add_scalar("train/step_l1", loss.item(), global_step)
                writer.flush()
    
    return total_loss / max(len(loader), 1), global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    metric_scale: float,
    amp_dtype: torch.dtype = torch.bfloat16,
    amp_enabled: bool = True,
) -> Tuple[float, float]:
    """Validate model. Returns (rmse_db, l1_loss)."""
    model.eval()
    total_rmse, total_l1 = 0.0, 0.0
    n = 0
    
    for x, y, mask, _ in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        
        # Pad input channels if needed
        model_in_ch = model.stem.in_channels
        if x.shape[1] < model_in_ch:
            pad = torch.zeros(x.shape[0], model_in_ch - x.shape[1], x.shape[2], x.shape[3], device=device)
            x = torch.cat([x, pad], dim=1)
        
        # Use autocast for Flash Attention speedup
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=amp_enabled and device.type == 'cuda'):
            pred = model(x)
        
        pred = torch.clamp(pred, 0.0, 1.0)
        
        total_rmse += rmse(pred * metric_scale, y * metric_scale, mask).item()
        total_l1 += torch.mean(torch.abs(pred - y) * mask).item()
        n += 1
    
    return total_rmse / max(n, 1), total_l1 / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TxUNet")
    parser.add_argument("--model", type=str, required=True, help="Model config YAML")
    parser.add_argument("--data", type=str, required=True, help="Data config YAML")
    parser.add_argument("--config", type=str, required=True, help="Training config YAML")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint to resume from")
    args = parser.parse_args()

    # Load configs
    model_cfg = load_yaml(args.model)
    data_cfg = load_yaml(args.data)
    train_cfg = load_yaml(args.config)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable TF32 for stability/perf on Ampere+
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    print(f"Device: {device}")

    # Data
    resize_hw = tuple(data_cfg.get("resize", [256, 256]))
    y_db_max = float(data_cfg.get("y_db_max", 160.0))
    train_buildings = [int(b) for b in data_cfg["train_buildings"]]
    val_buildings = [int(b) for b in data_cfg["val_buildings"]]

    all_pairs = gather_task2_samples(Path(data_cfg["data_root"]), split="train")
    train_pairs, val_pairs = split_by_building(all_pairs, train_buildings, val_buildings)
    print(f"Train: {len(train_pairs)} | Val: {len(val_pairs)}")

    ds_train = IndoorRadioMapDataset(data_cfg["data_root"], "train", resize_hw, train_pairs, y_db_max)
    ds_val = IndoorRadioMapDataset(data_cfg["data_root"], "val", resize_hw, val_pairs, y_db_max)

    batch_size = int(train_cfg.get("batch_size", 4))
    num_workers = int(train_cfg.get("num_workers", 4))
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Model
    model = build_model(model_cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Resume
    if args.resume and Path(args.resume).is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        print(f"Resumed from: {args.resume}")

    # Optimizer & Loss
    lr = float(train_cfg.get("lr", 3e-4))
    clip_norm = float(train_cfg.get("clip_norm", 0.0))
    amp_enabled = bool(train_cfg.get("amp_enabled", True))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = L1LossMasked()

    # AMP
    amp_dtype = torch.bfloat16 if train_cfg.get("amp_dtype", "fp16") == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Run directory
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path(train_cfg.get("checkpoint_dir", "runs")) / f"{ts}_txunet"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump({"model": model_cfg, "data": data_cfg, "train": train_cfg}, f, indent=2)

    # TensorBoard
    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))
        print(f"TensorBoard: {run_dir / 'tb'}")

    # Training loop
    epochs = int(train_cfg.get("epochs", 100))
    best_rmse = float("inf")

    global_step = 0
    for epoch in range(1, epochs + 1):
        # Train
        train_loss, global_step = train_epoch(
            model,
            loader_train,
            criterion,
            optimizer,
            device,
            scaler,
            amp_dtype,
            amp_enabled=amp_enabled,
            writer=writer,
            log_interval=50,
            global_step=global_step,
            clip_norm=clip_norm if clip_norm > 0.0 else None,
        )
        
        # Validate
        val_rmse, val_l1 = validate(model, loader_val, device, y_db_max, amp_dtype=amp_dtype, amp_enabled=amp_enabled)
        
        # Log
        print(f"Epoch {epoch:03d} | Train L1: {train_loss:.4f} | Val RMSE: {val_rmse:.2f} dB | Val L1: {val_l1:.4f}")
        if writer:
            writer.add_scalar("train/l1", train_loss, epoch)
            writer.add_scalar("val/rmse_db", val_rmse, epoch)
            writer.add_scalar("val/l1", val_l1, epoch)
            writer.flush()

        # Checkpoint
        ckpt = {"state_dict": model.state_dict(), "epoch": epoch, "model_cfg": model_cfg}
        
        # Save last
        torch.save(ckpt, run_dir / "last.ckpt")
        
        # Save best
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(ckpt, run_dir / "best.ckpt")
            print(f"  -> New best: {best_rmse:.2f} dB")

    print(f"Training complete. Best RMSE: {best_rmse:.2f} dB")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()


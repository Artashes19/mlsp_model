"""Unified trainer entrypoint for radio map models (TxUNet / Restormer-Port)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.amp
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore
import ast
import contextlib

from data.dataset import IndoorRadioMapDataset, gather_task2_samples, parse_meta
from losses.l1_rmse import L1LossMasked, rmse
try:
    from losses.charbonnier import CharbonnierLossMasked  # type: ignore
except Exception:
    CharbonnierLossMasked = None  # type: ignore
from models.radio_unet_tx.unet import TxUNet
from models.restormer_port.wrapper import RestormerRadio
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:
    SummaryWriter = None  # type: ignore


def _snapshot_params_cpu(model: nn.Module) -> List[torch.Tensor]:
    """Clone all trainable parameters to CPU as float32 for norm/delta calculations."""
    params: List[torch.Tensor] = []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        params.append(p.detach().float().cpu().clone())
    return params


def _l2_update_to_weight_ratio(params_before: List[torch.Tensor], model: nn.Module) -> float:
    """Compute ||ΔW||_2 / ||W||_2 between current model params and a saved snapshot."""
    delta_sq = 0.0
    weight_sq = 0.0
    idx = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        after = p.detach().float().cpu()
        before = params_before[idx]
        idx += 1
        diff = after - before
        # accumulate squares
        delta_sq += float(diff.pow(2).sum().item())
        weight_sq += float(after.pow(2).sum().item())
    if weight_sq <= 0.0:
        return float("nan")
    delta_norm = delta_sq ** 0.5
    weight_norm = weight_sq ** 0.5
    return float(delta_norm / max(weight_norm, 1e-12))


def load_yaml(path: str | Path) -> Dict:
    if yaml is not None:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    # Fallback: very simple YAML (key: value per line) using literal_eval when possible
    data: Dict[str, object] = {}
    with open(path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if val == "":
                continue
            low = val.lower()
            if low == "true":
                data[key] = True
                continue
            if low == "false":
                data[key] = False
                continue
            try:
                data[key] = ast.literal_eval(val)
            except Exception:
                data[key] = val
    return data


def split_pairs_by_hash(pairs: List, val_ratio: float, seed: int) -> Tuple[List, List]:
    def hkey(stem: str) -> float:
        h = hashlib.md5((stem + str(seed)).encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    train, val = [], []
    for sp in pairs:
        key = Path(sp.input_path).stem
        if hkey(key) < val_ratio:
            val.append(sp)
        else:
            train.append(sp)
    return train, val


def split_pairs_by_building(pairs: List, train_buildings: List[int], val_buildings: List[int]) -> Tuple[List, List]:
    train_set = set(int(b) for b in train_buildings)
    val_set = set(int(b) for b in val_buildings)
    if train_set & val_set:
        raise ValueError(f"train_buildings and val_buildings overlap: {sorted(train_set & val_set)}")

    train: List = []
    val: List = []
    unknown = 0
    for sp in pairs:
        meta = parse_meta(Path(sp.input_path).name)
        b = meta.get("building")
        if b in train_set:
            train.append(sp)
        elif b in val_set:
            val.append(sp)
        else:
            unknown += 1
    if unknown > 0:
        print(f"[split_pairs_by_building] Skipped {unknown} samples with buildings not in provided sets")
    return train, val


def build_model(model_cfg: Dict) -> nn.Module:
    name = model_cfg.get("name")
    if name == "radio_unet_tx":
        return TxUNet(
            in_ch=model_cfg.get("in_ch", 3),
            out_ch=model_cfg.get("out_ch", 1),
            base_ch=model_cfg.get("base_ch", 48),
            depths=tuple(model_cfg.get("depths", (4, 6, 6, 8))),
            heads=tuple(model_cfg.get("heads", (4, 4, 8, 8))),
            expand=float(model_cfg.get("expand", 2.66)),
            use_checkpoint=bool(model_cfg.get("use_checkpoint", True)),
            ln_eps=float(model_cfg.get("ln_eps", 1e-5)),
            apply_dw_on_v=bool(model_cfg.get("apply_dw_on_v", False)),
            residual_scale=float(model_cfg.get("residual_scale", 1.0)),
        )
    elif name == "restormer":
        return RestormerRadio(
            in_ch=model_cfg.get("in_ch", 3),
            out_ch=model_cfg.get("out_ch", 1),
            base_ch=model_cfg.get("base_ch", 48),
            heads=tuple(model_cfg.get("heads", (1, 2, 4, 8))),
            refinement_blocks=int(model_cfg.get("refinement_blocks", 4)),
            expand=float(model_cfg.get("expand", 2.66)),
        )
    else:
        raise ValueError(f"Unknown model name: {name}")


def get_optimizer(params, train_cfg: Dict):
    name = str(train_cfg.get("optimizer", "Adam"))
    lr = float(train_cfg.get("lr", 3e-4))
    betas = train_cfg.get("betas", [0.9, 0.999])
    if isinstance(betas, (list, tuple)) and len(betas) == 2:
        betas_t = (float(betas[0]), float(betas[1]))
    else:
        betas_t = (0.9, 0.999)
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    if name.lower() == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas_t, weight_decay=weight_decay)
    return torch.optim.Adam(params, lr=lr, betas=betas_t, weight_decay=weight_decay)


def get_scheduler(optimizer, train_cfg: Dict, total_steps: int | None = None, warmup_steps: int = 0):
    name = str(train_cfg.get("scheduler", "none")).lower()
    if name == "none":
        return None
    if name in ("cosine", "warmup_cosine"):
        # Per-step cosine with optional linear warmup
        if total_steps is None:
            raise ValueError("total_steps is required for cosine scheduler")
        base_lr = float(optimizer.param_groups[0]["lr"]) if len(optimizer.param_groups) > 0 else float(train_cfg.get("lr", 3e-4))
        min_lr = float(train_cfg.get("min_lr", train_cfg.get("eta_min", 0.0)))
        min_ratio = max(0.0, min_lr / max(1e-12, base_lr))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps and warmup_steps > 0:
                return float(step + 1) / float(max(1, warmup_steps))
            # cosine over remaining steps
            rem = max(1, total_steps - max(0, warmup_steps))
            prog = float(step - warmup_steps) / float(rem)
            prog = min(max(prog, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * prog))
            return float(min_ratio + (1.0 - min_ratio) * cosine)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    raise ValueError(f"Unknown scheduler: {name}")


def train_one_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer,
    grad_clip: float | None,
    writer,
    global_step: int,
    amp: bool,
    scaler: object | None,
    grad_accum_steps: int,
    scheduler,
    on_step: Optional[Callable[[int], None]] = None,
    metric_scale: Optional[float] = None,
    amp_dtype: torch.dtype | None = None,
    channels_last: bool = False,
    sanitize_for_loss: bool = False,
    skip_non_finite: bool = True,
):
    model.train()
    total_loss = 0.0
    accum_count = 0
    optimizer.zero_grad(set_to_none=True)
    # Aggregate training RMSE in dB units across batches for per-epoch logging
    train_rmse_sum = 0.0
    train_rmse_count = 0
    for x, y, m, _ in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
        if channels_last and device.type == "cuda":
            x = x.to(memory_format=torch.channels_last)
        # Match model stem input channels by padding/slicing
        stem_in = getattr(getattr(model, "stem", None), "in_channels", x.shape[1])
        if x.shape[1] < stem_in:
            pad = torch.zeros((x.shape[0], stem_in - x.shape[1], x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > stem_in:
            x = x[:, :stem_in]
        # Compute raw loss (not scaled) for correct logging; scale only for backward
        try:
            if amp and scaler is not None:
                with torch.amp.autocast("cuda", dtype=amp_dtype if amp_dtype is not None else torch.float16):
                    y_hat = model(x)
                    # Optional: sanitize predictions/targets and restrict loss to valid elements
                    if sanitize_for_loss:
                        y_hat_loss = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
                        y_loss = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
                        y_hat_loss = torch.clamp(y_hat_loss, 0.0, 1.0)
                        mask_valid = (torch.isfinite(y_hat) & torch.isfinite(y)).to(y.dtype) * m
                        if torch.sum(mask_valid) <= 0:
                            # Skip batch with no valid targets
                            optimizer.zero_grad(set_to_none=True)
                            if writer is not None:
                                writer.add_scalar("train/skip_empty_mask", 1.0, global_step)
                            continue
                        raw_loss = criterion(y_hat_loss, y_loss, mask_valid)
                    else:
                        raw_loss = criterion(y_hat, y, m)
                    if skip_non_finite and not torch.isfinite(raw_loss).all():
                        optimizer.zero_grad(set_to_none=True)
                        if writer is not None:
                            writer.add_scalar("train/skipped_nonfinite_loss", 1.0, global_step)
                        continue
                    loss = raw_loss / max(1, grad_accum_steps)
                scaler.scale(loss).backward()
            else:
                y_hat = model(x)
                if sanitize_for_loss:
                    y_hat_loss = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
                    y_loss = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
                    y_hat_loss = torch.clamp(y_hat_loss, 0.0, 1.0)
                    mask_valid = (torch.isfinite(y_hat) & torch.isfinite(y)).to(y.dtype) * m
                    if torch.sum(mask_valid) <= 0:
                        optimizer.zero_grad(set_to_none=True)
                        if writer is not None:
                            writer.add_scalar("train/skip_empty_mask", 1.0, global_step)
                        continue
                    raw_loss = criterion(y_hat_loss, y_loss, mask_valid)
                else:
                    raw_loss = criterion(y_hat, y, m)
                if skip_non_finite and not torch.isfinite(raw_loss).all():
                    optimizer.zero_grad(set_to_none=True)
                    if writer is not None:
                        writer.add_scalar("train/skipped_nonfinite_loss", 1.0, global_step)
                    continue
                loss = raw_loss / max(1, grad_accum_steps)
                loss.backward()
        except RuntimeError as e:
            if torch.cuda.is_available() and "out of memory" in str(e).lower():
                try:
                    print(torch.cuda.memory_summary())
                except Exception:
                    pass
            raise

        # Optional: accumulate training RMSE in dB per batch
        if metric_scale is not None:
            with torch.no_grad():
                y_hat_eval = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
                y_eval = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
                y_hat_eval = torch.clamp(y_hat_eval, 0.0, 1.0)
                mask_valid = (torch.isfinite(y_hat_eval) & torch.isfinite(y_eval)).to(y.dtype) * m
                # rmse returns mean over batch; accumulate as average-of-batches to mirror val
                batch_rmse = rmse(y_hat_eval * metric_scale, y_eval * metric_scale, mask_valid)
                train_rmse_sum += float(batch_rmse.item())
                train_rmse_count += 1
        accum_count += 1
        grad_norm_val: Optional[float] = None
        step_now = False
        if accum_count >= max(1, grad_accum_steps):
            # For AMP, unscale grads BEFORE clipping/norm and stepping
            if amp and scaler is not None:
                scaler.unscale_(optimizer)
            # Compute (and optionally clip) grad norm only when stepping
            if grad_clip and grad_clip > 0:
                grad_norm_val = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip).item())
            else:
                # compute grad norm without clipping
                total_sq = 0.0
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    total_sq += float(g.pow(2).sum().item())
                grad_norm_val = total_sq ** 0.5
            # Skip optimizer step if gradients are non-finite
            if skip_non_finite and not math.isfinite(float(grad_norm_val)):
                optimizer.zero_grad(set_to_none=True)
                accum_count = 0
                step_now = False
                if writer is not None:
                    writer.add_scalar("train/grad_nonfinite_step", 1.0, global_step)
                # Do not advance scheduler/global_step on skipped steps
                continue
            if amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            # Step scheduler per optimizer step (per-step schedule)
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accum_count = 0
            step_now = True
        loss_value = float(raw_loss.item())
        total_loss += loss_value
        if writer is not None and step_now:
            writer.add_scalar("train/l1_step", loss_value, global_step)
            if grad_norm_val is not None:
                writer.add_scalar("train/grad_norm", grad_norm_val, global_step)
            # Log AMP loss scale for debugging grad scaling behavior
            if amp and scaler is not None:
                writer.add_scalar("train/loss_scale", float(scaler.get_scale()), global_step)
            # Log LR per optimizer step (first param group)
            if len(optimizer.param_groups) > 0 and "lr" in optimizer.param_groups[0]:
                writer.add_scalar("train/lr_step", float(optimizer.param_groups[0]["lr"]), global_step)
            global_step += 1
            if on_step is not None:
                on_step(global_step)
    train_rmse_epoch = (train_rmse_sum / max(1, train_rmse_count)) if metric_scale is not None else float('nan')
    return total_loss / max(1, len(loader)), train_rmse_epoch, global_step


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_rmse = 0.0
    for x, y, m, _ in loader:
        x = x.to(device)
        y = y.to(device)
        m = m.to(device)
        y_hat = model(x)
        total_rmse += float(rmse(y_hat, y, m).item())
    return total_rmse / max(1, len(loader))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train radio map models (TxUNet / Restormer-Port)")
    parser.add_argument("--model", type=str, required=True, help="Path to model YAML config")
    parser.add_argument("--data", type=str, required=True, help="Path to data YAML config")
    parser.add_argument("--config", type=str, required=True, help="Path to training YAML config")
    parser.add_argument("--limit_train", type=int, default=0, help="Limit number of training samples (0=all)")
    parser.add_argument("--limit_val", type=int, default=0, help="Limit number of validation samples (0=all)")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint (.ckpt) to initialize model weights")
    parser.add_argument("--resume_scheduler_from_epoch", type=int, default=-1, help="Offset LR scheduler and global_step to match this past epoch (>=0).")
    parser.add_argument("--resume_scheduler_from_step", type=int, default=-1, help="Offset LR scheduler and global_step to this past step (>=0).")
    args = parser.parse_args()

    model_cfg = load_yaml(args.model)
    data_cfg = load_yaml(args.data)
    train_cfg = load_yaml(args.config)

    # Device and performance settings
    device_name = str(train_cfg.get("device", "cuda"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    if str(train_cfg.get("allow_tf32", True)).lower() in ("1","true","yes"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        except Exception:
            pass
    if bool(train_cfg.get("cudnn_benchmark", True)):
        try:
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
        except Exception:
            pass

    # Datasets and loaders
    resize_hw = tuple(data_cfg.get("resize", [256, 256]))  # type: ignore[assignment]
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    seed = int(data_cfg.get("seed", 42))
    split_mode = str(data_cfg.get("split_mode", "hash")).lower()
    # Use dB max for scaling metrics back to physical units
    y_db_max = float(data_cfg.get("y_db_max", 160.0))
    metric_scale = y_db_max

    all_pairs = gather_task2_samples(Path(data_cfg["data_root"]), split="train")
    if split_mode == "building":
        train_buildings = [int(b) for b in data_cfg.get("train_buildings", [])]
        val_buildings = [int(b) for b in data_cfg.get("val_buildings", [])]
        if not train_buildings or not val_buildings:
            raise ValueError("split_mode=building requires train_buildings and val_buildings in data config")
        train_pairs, val_pairs = split_pairs_by_building(all_pairs, train_buildings, val_buildings)
        print(f"Split mode: building | train_buildings={sorted(set(train_buildings))} | val_buildings={sorted(set(val_buildings))}")
    else:
        train_pairs, val_pairs = split_pairs_by_hash(all_pairs, val_ratio=val_ratio, seed=seed)
        print(f"Split mode: hash | val_ratio={val_ratio} | seed={seed}")
    if args.limit_train and args.limit_train > 0:
        train_pairs = train_pairs[: args.limit_train]
    if args.limit_val and args.limit_val > 0:
        val_pairs = val_pairs[: args.limit_val]

    ds_train = IndoorRadioMapDataset(root=data_cfg["data_root"], split="train", resize_hw=resize_hw, file_pairs=train_pairs, y_db_max=y_db_max)
    ds_val = IndoorRadioMapDataset(root=data_cfg["data_root"], split="val", resize_hw=resize_hw, file_pairs=val_pairs, y_db_max=y_db_max)

    dl_bs = int(train_cfg.get("batch_size", 1))
    dl_workers = int(train_cfg.get("num_workers", 4))
    dl_pin = bool(train_cfg.get("pin_memory", True))
    dl_persistent = bool(train_cfg.get("persistent_workers", False))
    dl_prefetch = int(train_cfg.get("prefetch_factor", 2))
    dl_drop_last = bool(train_cfg.get("drop_last", False))
    loader_train = DataLoader(
        ds_train,
        batch_size=dl_bs,
        shuffle=True,
        num_workers=dl_workers,
        pin_memory=dl_pin,
        persistent_workers=dl_persistent if dl_workers > 0 else False,
        prefetch_factor=dl_prefetch if dl_workers > 0 else None,
        drop_last=dl_drop_last,
    )
    loader_val = DataLoader(
        ds_val,
        batch_size=1,
        shuffle=False,
        num_workers=dl_workers,
        pin_memory=dl_pin,
        persistent_workers=dl_persistent if dl_workers > 0 else False,
        prefetch_factor=dl_prefetch if dl_workers > 0 else None,
    )

    # Model & optim
    model = build_model(model_cfg).to(device)
    if bool(train_cfg.get("channels_last", False)) and device.type == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.is_file():
            print(f"Loading weights from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)
            state = ckpt.get("state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"[resume] missing keys: {len(missing)}")
            if unexpected:
                print(f"[resume] unexpected keys: {len(unexpected)}")
        else:
            print(f"Warning: resume checkpoint not found: {ckpt_path}")
    optimizer = get_optimizer(model.parameters(), train_cfg)
    loss_name = str(train_cfg.get("loss", "l1")).lower()
    if loss_name.startswith("charb") and CharbonnierLossMasked is not None:
        criterion = CharbonnierLossMasked(epsilon=float(train_cfg.get("epsilon", 1e-3)))
    else:
        criterion = L1LossMasked()

    # Run dir
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = model_cfg.get("name", "model")
    run_name = str(train_cfg.get("run_name", model_name))
    base_dir = Path(str(train_cfg.get("checkpoint_dir", "runs")))
    run_dir = base_dir / f"{ts}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save merged config
    with open(run_dir / "config.json", "w") as f:
        json.dump({"model": model_cfg, "data": data_cfg, "train": train_cfg}, f, indent=2)

    # TensorBoard writer (allow overriding TB root to avoid full home FS)
    writer = None
    if SummaryWriter is not None:
        try:
            tb_root_cfg = str(train_cfg.get("tb_root", "")).strip()
            if tb_root_cfg:
                tb_dir = Path(tb_root_cfg) / run_dir.name / "tb"
            else:
                tb_dir = run_dir / "tb"
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"TensorBoard: {tb_dir}")
        except OSError as e:
            print(f"[warn] TensorBoard writer disabled due to OSError: {e}")
            writer = None
    else:
        print("TensorBoard not available; proceeding without TB logging")

    best_val = float("inf")
    epochs = int(train_cfg.get("epochs", 100))
    grad_clip = train_cfg.get("grad_clip", 1.0)
    amp = bool(train_cfg.get("amp", True))
    amp_dtype_name = str(train_cfg.get("amp_autocast_dtype", "fp16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in ("bf16", "bfloat16") else torch.float16
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    validate_every_steps = int(train_cfg.get("validate_every_steps", 5))
    use_grad_scaler = bool(train_cfg.get("use_grad_scaler", amp and (amp_dtype is torch.float16)))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and use_grad_scaler))
    global_step = 0
    sanitize_for_loss = bool(train_cfg.get("sanitize_train_loss", False))
    skip_non_finite = bool(train_cfg.get("skip_non_finite_batches", True))
    # Build per-step scheduler after knowing steps per epoch
    steps_per_epoch = max(1, (len(loader_train) + max(1, grad_accum_steps) - 1) // max(1, grad_accum_steps))
    total_steps = steps_per_epoch * epochs
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.0))
    warmup_steps = int(round(total_steps * warmup_ratio))
    scheduler = get_scheduler(optimizer, train_cfg, total_steps=total_steps, warmup_steps=warmup_steps)
    # If requested, advance the scheduler and global_step to align with a prior epoch/step
    resume_steps = 0
    if args.resume_scheduler_from_step is not None and args.resume_scheduler_from_step >= 0:
        resume_steps = int(args.resume_scheduler_from_step)
    elif args.resume_scheduler_from_epoch is not None and args.resume_scheduler_from_epoch >= 0:
        resume_steps = int(args.resume_scheduler_from_epoch) * steps_per_epoch
    if scheduler is not None and resume_steps > 0:
        for _ in range(resume_steps):
            scheduler.step()
        global_step = resume_steps
    # Validation function
    @torch.no_grad()
    def validate_metrics():
        model.eval()
        total_rmse = 0.0
        total_l1 = 0.0
        n = 0
        for x, y, m, _ in loader_val:
            x = x.to(device)
            y = y.to(device)
            m = m.to(device)
            # channels_last and pad channels if needed
            if bool(train_cfg.get("channels_last", False)) and device.type == "cuda":
                x = x.to(memory_format=torch.channels_last)
            stem_in = getattr(getattr(model, "stem", None), "in_channels", x.shape[1])
            if x.shape[1] < stem_in:
                pad = torch.zeros((x.shape[0], stem_in - x.shape[1], x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
                x = torch.cat([x, pad], dim=1)
            elif x.shape[1] > stem_in:
                x = x[:, :stem_in]
            ctx = torch.amp.autocast("cuda", dtype=amp_dtype) if amp else contextlib.nullcontext()
            with ctx:
                y_hat = model(x)
            y_hat = torch.nan_to_num(y_hat, nan=0.0, posinf=1.0, neginf=0.0)
            y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=0.0)
            y_hat = torch.clamp(y_hat, 0.0, 1.0)
            mask_valid = (torch.isfinite(y_hat) & torch.isfinite(y)).to(y.dtype) * m
            if torch.sum(mask_valid) <= 0:
                continue
            total_rmse += float(rmse(y_hat * metric_scale, y * metric_scale, mask_valid).item())
            total_l1 += float(criterion(y_hat, y, mask_valid).item())
            n += 1
        if n == 0:
            return float("nan"), float("nan")
        return total_rmse / n, total_l1 / n

    # Callback to run validation every N steps
    def on_step_cb(step_idx: int) -> None:
        if validate_every_steps > 0 and step_idx % validate_every_steps == 0:
            val_rmse_db, val_l1 = validate_metrics()
            print(f"Step {step_idx:06d}: val L1={val_l1:.4f} | val RMSE_dB={val_rmse_db:.4f} dB")
            if writer is not None:
                # Use step-specific tags to avoid mixing step and epoch on the same series
                writer.add_scalar("val/rmse_db_step", val_rmse_db, step_idx)
                writer.add_scalar("val/l1_step", val_l1, step_idx)
    for epoch in range(1, epochs + 1):
        # Snapshot parameters at the start of the epoch to measure update-to-weight ratio
        params_before_epoch = _snapshot_params_cpu(model)
        train_loss, train_rmse_db, global_step = train_one_epoch(
            model,
            loader_train,
            criterion,
            device,
            optimizer,
            grad_clip,
            writer,
            global_step,
            amp,
            scaler,
            grad_accum_steps,
            scheduler,
            on_step=on_step_cb,
            metric_scale=metric_scale,
            amp_dtype=amp_dtype,
            channels_last=bool(train_cfg.get("channels_last", False)),
            sanitize_for_loss=sanitize_for_loss,
            skip_non_finite=skip_non_finite,
        )
        # End-of-epoch validation snapshot
        val_rmse_db, val_l1 = validate_metrics()
        # Compute L2 update-to-weight ratio at epoch end
        try:
            uw_ratio = _l2_update_to_weight_ratio(params_before_epoch, model)
        except Exception:
            uw_ratio = float("nan")
        print(f"Epoch {epoch:03d}: train L1={train_loss:.4f} | train RMSE_dB={train_rmse_db:.4f} | val L1={val_l1:.4f} | val RMSE_dB={val_rmse_db:.4f} dB")
        # TB scalars per epoch
        if writer is not None:
            writer.add_scalar("train/l1_epoch", train_loss, epoch)
            writer.add_scalar("train/rmse_db_epoch", train_rmse_db, epoch)
            writer.add_scalar("val/rmse_db_epoch", val_rmse_db, epoch)
            writer.add_scalar("val/l1_epoch", val_l1, epoch)
            writer.add_scalar("train/update_to_weight_ratio_epoch", uw_ratio, epoch)
            # LR logging (first param group)
            if len(optimizer.param_groups) > 0 and "lr" in optimizer.param_groups[0]:
                writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), epoch)
            # Ensure data is flushed to disk each epoch to improve TB reliability
            writer.flush()
        # checkpoint best
        if val_rmse_db < best_val:
            best_val = val_rmse_db
            ckpt = {
                "state_dict": model.state_dict(),
                "model_cfg": model_cfg,
                "y_db_max": y_db_max,
            }
            torch.save(ckpt, run_dir / "model_best.ckpt")
    print(f"Best val RMSE_dB: {best_val:.4f} dB")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()



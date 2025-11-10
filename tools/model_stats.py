from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
from typing import Dict, Tuple

import torch
from torch import nn

# Ensure project root is on sys.path to import train.py when running as a script from tools/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse helpers and model builder from train.py
from train import load_yaml, build_model  # type: ignore


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def try_compute_flops(model: nn.Module, input_shape: Tuple[int, int, int, int]) -> Tuple[float | None, str]:
    """
    Returns (flops_per_sample, method) where flops_per_sample is in FLOPs (not MACs).
    If unavailable, returns (None, reason).
    """
    b, c, h, w = input_shape
    dummy = torch.randn((1, c, h, w), device=next(model.parameters()).device)
    # Try thop first
    try:
        from thop import profile  # type: ignore

        model.eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        # thop returns MACs; 1 MAC ~ 2 FLOPs
        flops = float(macs) * 2.0
        return flops, "thop"
    except Exception as e:
        pass

    # Try torch.profiler if available (PyTorch >=1.9, with_flops in newer versions)
    try:
        from torch.profiler import profile, ProfilerActivity

        model.eval()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if torch.cuda.is_available() else [ProfilerActivity.CPU]) as prof:
            with torch.no_grad():
                _ = model(dummy)
        # Newer PyTorch can expose flops via events; fallback to None if not found
        # This is heuristic and may not always be present
        total_flops = 0.0
        for evt in prof.key_averages():
            if hasattr(evt, "flops") and evt.flops is not None:
                total_flops += float(evt.flops)
        if total_flops > 0:
            return total_flops, "torch.profiler"
    except Exception:
        pass

    return None, "unavailable"


def bytes_to_mib(x: float) -> float:
    return x / (1024.0 * 1024.0)


def estimate_optimizer_and_grad_memory_bytes(model: nn.Module, dtype: torch.dtype, optimizer_name: str = "Adam") -> int:
    bytes_per_elem = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else torch.iinfo(dtype).bits // 8
    param_bytes = sum(p.numel() for p in model.parameters()) * bytes_per_elem
    grad_bytes = sum(p.numel() for p in model.parameters() if p.requires_grad) * bytes_per_elem
    # Adam has m and v states per parameter (2x params)
    opt_mult = 2 if optimizer_name.lower() in ("adam", "adamw") else 1
    opt_bytes = opt_mult * param_bytes
    return int(param_bytes + grad_bytes + opt_bytes)


def measure_peak_memory_bytes(model: nn.Module, input_shape: Tuple[int, int, int, int], amp: bool) -> Tuple[int, int]:
    """
    Returns (peak_allocated_bytes, peak_reserved_bytes) for one fwd+bwd step with mean loss.
    """
    device = next(model.parameters()).device
    b, c, h, w = input_shape
    x = torch.randn((b, c, h, w), device=device)
    model.train()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    loss: torch.Tensor
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if amp:
        with torch.amp.autocast("cuda"):
            y = model(x)
            loss = y.mean()
        scaler.scale(loss).backward()
    else:
        y = model(x)
        loss = y.mean()
        loss.backward()
    # Report peaks
    peak_alloc = torch.cuda.max_memory_allocated(device) if torch.cuda.is_available() else 0
    peak_rsrv = torch.cuda.max_memory_reserved(device) if torch.cuda.is_available() else 0
    # Cleanup graph gradients
    model.zero_grad(set_to_none=True)
    return int(peak_alloc), int(peak_rsrv)


def try_measure_activation_memory_with_fallback(
    model: nn.Module,
    target_shape: Tuple[int, int, int, int],
    amp: bool,
) -> Tuple[int | None, int | None, str | None]:
    """
    Measure activation memory at target shape; on CUDA OOM, fall back to smaller inputs
    and scale by pixel area.

    Returns (peak_alloc_bytes or None, peak_reserved_bytes or None, note or None)
    """
    b, c, h, w = target_shape
    try:
        alloc, rsrv = measure_peak_memory_bytes(model, (b, c, h, w), amp)
        return alloc, rsrv, None
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" not in msg:
            raise
        # Fallback scales
        for scale in (0.5, 0.25, 0.125):
            hs = max(8, int(h * scale))
            ws = max(8, int(w * scale))
            try:
                alloc_s, rsrv_s = measure_peak_memory_bytes(model, (b, c, hs, ws), amp)
                area_scale = (h * w) / max(1, (hs * ws))
                est_alloc = int(math.ceil(alloc_s * area_scale))
                est_rsrv = int(math.ceil(rsrv_s * area_scale))
                note = f"estimated from {hs}x{ws} (area scale {area_scale:.2f}x)"
                return est_alloc, est_rsrv, note
            except RuntimeError as e2:
                if "out of memory" in str(e2).lower():
                    continue
                else:
                    raise
        return None, None, "activation measurement OOM even at reduced sizes"


def main() -> None:
    parser = argparse.ArgumentParser(description="Model stats: params, memory, FLOPs for TxUNet/Restormer")
    parser.add_argument("--model", required=True, type=str, help="Path to model YAML config")
    parser.add_argument("--data", required=True, type=str, help="Path to data YAML config (for HxW)")
    parser.add_argument("--train_cfg", type=str, default=None, help="Optional training YAML (for AMP/batch size)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for memory measurement (overrides train_cfg)")
    parser.add_argument("--epochs", type=int, default=35, help="Epochs for FLOPs estimate")
    parser.add_argument("--train_size", type=int, default=3000, help="Number of training samples for FLOPs estimate")
    parser.add_argument("--use_gpu0", action="store_true", help="Force CUDA:0 if available")
    args = parser.parse_args()

    model_cfg: Dict = load_yaml(args.model)
    data_cfg: Dict = load_yaml(args.data)
    train_cfg: Dict = load_yaml(args.train_cfg) if args.train_cfg else {}

    # Device selection
    if args.use_gpu0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = build_model(model_cfg).to(device)

    # Input shape
    resize_hw = tuple(data_cfg.get("resize", [256, 256]))
    h, w = int(resize_hw[0]), int(resize_hw[1])
    in_ch = int(model_cfg.get("in_ch", 3))
    out_ch = int(model_cfg.get("out_ch", 1))  # noqa: F841 (unused, kept for clarity)

    # AMP and batch size
    amp = bool(train_cfg.get("amp", False))
    batch_size = int(args.batch_size) if args.batch_size is not None else int(train_cfg.get("batch_size", 1))

    # Parameter counts
    total_params, trainable_params = count_parameters(model)

    # Static memory estimates
    dtype = torch.float16 if amp and torch.cuda.is_available() else torch.float32
    static_mem_bytes = estimate_optimizer_and_grad_memory_bytes(model, dtype, optimizer_name=str(train_cfg.get("optimizer", "Adam")))

    # Peak activation memory measurement (one fwd+bwd step) with OOM fallback and scaling
    peak_alloc, peak_rsrv, peak_note = try_measure_activation_memory_with_fallback(
        model, (batch_size, in_ch, h, w), amp=amp
    )

    # FLOPs (per sample forward) and training FLOPs estimate
    flops_per_sample_forward, flops_method = try_compute_flops(model, (1, in_ch, h, w))
    train_size = int(args.train_size)
    epochs = int(args.epochs)

    # Typical rule of thumb: backward ~ 2x forward FLOPs; weight update adds a bit
    # We use 3x forward FLOPs as an approximate training cost per sample
    if flops_per_sample_forward is not None:
        flops_per_sample_train = 3.0 * flops_per_sample_forward
        total_training_flops = flops_per_sample_train * float(train_size) * float(epochs)
    else:
        flops_per_sample_train = None
        total_training_flops = None

    # Report
    print("==== Model Stats ====")
    print(f"Model: {model_cfg.get('name', 'model')}")
    print(f"Input: (C={in_ch}, H={h}, W={w}) | Batch for memory: {batch_size}")
    print(f"AMP: {amp} | Device: {device}")
    print("")
    print(f"Parameters (total): {total_params:,}")
    print(f"Parameters (trainable): {trainable_params:,}")
    print("")
    print("---- GPU Memory (approx) ----")
    print(f"Static (params + grads + optimizer) ~ {bytes_to_mib(static_mem_bytes):.2f} MiB")
    if torch.cuda.is_available():
        if peak_alloc is not None and peak_rsrv is not None:
            print(f"Peak activation (1 step fwd+bwd) allocated: {bytes_to_mib(peak_alloc):.2f} MiB" + (f" | {peak_note}" if peak_note else ""))
            print(f"Peak activation (1 step) reserved: {bytes_to_mib(peak_rsrv):.2f} MiB" + (f" | {peak_note}" if peak_note else ""))
        else:
            print("Peak activation: OOM at target size; consider smaller input/batch or attention chunking.")
    else:
        print("CUDA not available; activation memory not measured.")
    print("")
    print("---- FLOPs ----")
    if flops_per_sample_forward is not None:
        def human_flops(x: float) -> str:
            units = ["", "K", "M", "G", "T", "P"]
            i = 0
            while x >= 1000.0 and i < len(units) - 1:
                x /= 1000.0
                i += 1
            return f"{x:.3f} {units[i]}FLOPs"

        print(f"Per-sample forward: {human_flops(flops_per_sample_forward)} (method={flops_method})")
        print(f"Per-sample training (~3x): {human_flops(flops_per_sample_train)}")
        if total_training_flops is not None:
            print(f"Total training for {train_size} samples x {epochs} epochs: {human_flops(total_training_flops)}")
    else:
        print("FLOPs unavailable (thop/torch.profiler not available or unsupported ops)")


if __name__ == "__main__":
    main()



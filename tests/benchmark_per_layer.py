"""
Per-layer benchmark: Baseline vs SRA-4 vs NSA-2D.

Profiles every encoder/decoder block individually for the production model.
Reports wall-clock time and peak memory delta for fwd and fwd+bwd per layer.

Run:
  cd mlsp_model/dev-clean
  /auto/home/artashes/miniconda3/envs/dev/bin/python tests/benchmark_per_layer.py
"""
import gc
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEVICE = torch.device("cuda:0")

# Production config
DEPTHS = (4, 6, 6, 8)
HEADS = (4, 4, 8, 8)
BASE_CH = 48


def _make_model(mode, use_checkpoint=False):
    from src.networks.txunet import TxUNetModel
    kwargs = dict(
        in_ch=11, out_ch=1, base_ch=BASE_CH,
        depths=DEPTHS, heads=HEADS,
        expand=2.66, use_checkpoint=use_checkpoint, ln_eps=1e-5,
        rope_enabled=True, rope_base=10000.0,
    )
    if mode == "nsa":
        kwargs.update(
            nsa_enabled=True, sra0_enabled=False,
            nsa_patch_sizes=[8, 8, 4, 4],
            nsa_top_n=[16, 8, 8, 8],
            nsa_window_sizes=[8, 8, 4, 4],
        )
    elif mode == "sra":
        kwargs.update(nsa_enabled=False, sra0_enabled=True, sra0_stride=4)
    elif mode == "nsa-gqa":
        kwargs.update(
            nsa_enabled=True, sra0_enabled=False,
            nsa_patch_sizes=[8, 8, 4, 4],
            nsa_top_n=[16, 8, 8, 8],
            nsa_window_sizes=[8, 8, 4, 4],
            nsa_gqa_group_size=4,
        )
    else:
        kwargs.update(nsa_enabled=False, sra0_enabled=False)
    return TxUNetModel(**kwargs).to(DEVICE)


def _cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _gpu_mem_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


# ── Per-layer profiling via hooks ────────────────────────────────────────────

class LayerProfiler:
    """Attach forward hooks to named submodules to measure time per layer."""

    def __init__(self, model, layer_names):
        """
        layer_names: ordered list of (display_name, module_attr_path)
        """
        self.layer_names = layer_names
        self.model = model
        self.timings = OrderedDict()
        self.hooks = []
        self._cur_start = None

    def _get_module(self, path):
        obj = self.model
        for attr in path.split("."):
            obj = getattr(obj, attr)
        return obj

    def attach(self):
        """Attach pre/post forward hooks."""
        for name, path in self.layer_names:
            mod = self._get_module(path)
            self.timings[name] = []

            def _make_hooks(n):
                def pre_hook(module, inp):
                    torch.cuda.synchronize()
                    self._starts[n] = time.perf_counter()

                def post_hook(module, inp, out):
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - self._starts[n]) * 1000
                    self.timings[n].append(elapsed)

                return pre_hook, post_hook

            pre, post = _make_hooks(name)
            self.hooks.append(mod.register_forward_pre_hook(pre))
            self.hooks.append(mod.register_forward_hook(post))
        self._starts = {}

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def profile_forward(model, x, layer_names, warmup=3, repeats=10):
    """Profile fwd-only per-layer timing."""
    model.eval()
    profiler = LayerProfiler(model, layer_names)
    profiler.attach()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(x)

    # Clear and measure
    for n, _ in layer_names:
        profiler.timings[n] = []

    with torch.no_grad():
        for _ in range(repeats):
            model(x)

    profiler.remove()

    # Compute averages
    results = OrderedDict()
    for n, _ in layer_names:
        vals = profiler.timings[n]
        results[n] = sum(vals) / len(vals) if vals else 0.0
    return results


def profile_forward_memory(model, x, layer_names):
    """Profile fwd-only per-layer peak memory delta."""
    model.eval()
    results = OrderedDict()

    # We can't get per-layer memory delta from hooks easily.
    # Instead, run each layer's forward in isolation by slicing the model.
    # But that's complex. So we measure cumulative peak after each stage.

    # Simpler approach: measure total, then measure by removing layers.
    # Even simpler: use the forward hooks to record memory snapshots.

    class MemSnapHook:
        def __init__(self):
            self.snapshots = OrderedDict()
            self.hooks = []

        def attach(self, model, layer_names):
            for name, path in layer_names:
                mod = model
                for attr in path.split("."):
                    mod = getattr(mod, attr)
                self.snapshots[name] = {"pre": 0, "post": 0}

                def _make(n):
                    def pre_hook(module, inp):
                        torch.cuda.synchronize()
                        self.snapshots[n]["pre"] = torch.cuda.memory_allocated() / 1024 / 1024

                    def post_hook(module, inp, out):
                        torch.cuda.synchronize()
                        self.snapshots[n]["post"] = torch.cuda.memory_allocated() / 1024 / 1024

                    return pre_hook, post_hook

                pre, post = _make(name)
                self.hooks.append(mod.register_forward_pre_hook(pre))
                self.hooks.append(mod.register_forward_hook(post))

        def remove(self):
            for h in self.hooks:
                h.remove()

    _cleanup()
    snap = MemSnapHook()
    snap.attach(model, layer_names)

    with torch.no_grad():
        model(x)

    snap.remove()

    for n, _ in layer_names:
        results[n] = snap.snapshots[n]["post"] - snap.snapshots[n]["pre"]
    return results


def profile_train_step(model, x, target, layer_names, warmup=2, repeats=5):
    """
    Profile a full AMP train step. Returns per-layer fwd time.
    Also returns total step time and peak memory.
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")

    def _step():
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = F.mse_loss(out, target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    # Warmup
    for _ in range(warmup):
        _step()
    torch.cuda.synchronize()

    # Measure total time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        _step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_ms = (t1 - t0) / repeats * 1000

    # Measure peak memory
    _cleanup()
    _step()
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    # Measure per-layer fwd timing within training
    profiler = LayerProfiler(model, layer_names)
    profiler.attach()
    for n, _ in layer_names:
        profiler.timings[n] = []
    for _ in range(repeats):
        _step()
    profiler.remove()

    fwd_per_layer = OrderedDict()
    for n, _ in layer_names:
        vals = profiler.timings[n]
        fwd_per_layer[n] = sum(vals) / len(vals) if vals else 0.0

    del optimizer, scaler
    return fwd_per_layer, total_ms, peak_mb


# ── Layer definitions ────────────────────────────────────────────────────────

C = BASE_CH

LAYER_DEFS = [
    # (display_name, module_path, level, role, spatial_divisor, channels, depth)
    ("stem",       "stem",       -1, "stem",  1,  C,    0),
    ("enc0",       "enc0",        0, "enc",   1,  C,    DEPTHS[0]),
    ("down1",      "down1",       0, "down",  1,  C,    0),
    ("enc1",       "enc1",        1, "enc",   2,  2*C,  DEPTHS[1]),
    ("down2",      "down2",       1, "down",  2,  2*C,  0),
    ("enc2",       "enc2",        2, "enc",   4,  4*C,  DEPTHS[2]),
    ("down3",      "down3",       2, "down",  4,  4*C,  0),
    ("enc3 (btl)", "enc3",        3, "enc",   8,  8*C,  DEPTHS[3]),
    ("up3",        "up3",         3, "up",    8,  8*C,  0),
    ("dec2",       "dec2",        2, "dec",   4,  4*C,  DEPTHS[2]),
    ("up2",        "up2",         2, "up",    4,  4*C,  0),
    ("dec1",       "dec1",        1, "dec",   2,  2*C,  DEPTHS[1]),
    ("up1",        "up1",         1, "up",    2,  2*C,  0),
    ("dec0",       "dec0",        0, "dec",   1,  2*C,  DEPTHS[0]),
    ("dec0_extra", "dec0_extra",  0, "dec",   1,  2*C,  1),
    ("head",       "head_conv1",  -1, "head", 1,  2*C,  0),
]

LAYER_NAMES = [(d[0], d[1]) for d in LAYER_DEFS]

MODES = ["baseline", "sra", "nsa", "nsa-gqa"]
MODE_LABELS = {"baseline": "Baseline", "sra": "SRA-4", "nsa": "NSA-MHA", "nsa-gqa": "NSA-GQA4"}


# ── Main ─────────────────────────────────────────────────────────────────────

def run():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print("=" * 130)
    print(f"Per-Layer Benchmark: Baseline vs SRA-4 vs NSA-2D  ({gpu_name}, {gpu_mem:.0f}GB)")
    print(f"Production config: depths={DEPTHS}, heads={HEADS}, base_ch={BASE_CH}")
    print("=" * 130)

    for res_label, B, H, W in [
        ("B=1 128x128", 1, 128, 128),
        ("B=1 256x256", 1, 256, 256),
    ]:
        print(f"\n{'━' * 130}")
        print(f"  Resolution: {res_label}  (spatial at each level: L0={H}x{W}, L1={H//2}x{W//2}, L2={H//4}x{W//4}, L3={H//8}x{W//8})")
        print(f"{'━' * 130}")

        # ── INFERENCE ──
        print(f"\n  ┌─ Inference (eval, no_grad) {'─' * 98}┐")
        hdr = f"  │ {'Layer':<16} {'ch':>4} {'dep':>3} {'spatial':>9} │"
        for m in MODES:
            hdr += f" {MODE_LABELS[m]+' ms':>11}"
        hdr += " │"
        for m in MODES:
            hdr += f" {MODE_LABELS[m]+' ΔMB':>12}"
        hdr += " │"
        print(hdr)
        print(f"  │ {'─'*16} {'─'*4} {'─'*3} {'─'*9} ┼{'─'*35}─┼{'─'*38}─│")

        fwd_times = {}
        fwd_mems = {}
        fwd_totals = {}
        for mode in MODES:
            try:
                _cleanup()
                model = _make_model(mode)
                x = torch.randn(B, 11, H, W, device=DEVICE)

                ft = profile_forward(model, x, LAYER_NAMES, warmup=3, repeats=10)
                fm = profile_forward_memory(model, x, LAYER_NAMES)
                fwd_times[mode] = ft
                fwd_mems[mode] = fm

                # Total inference
                model.eval()
                _cleanup()
                with torch.no_grad():
                    for _ in range(3):
                        model(x)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for _ in range(10):
                        model(x)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                fwd_totals[mode] = (t1 - t0) / 10 * 1000
                _cleanup()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    model(x)
                torch.cuda.synchronize()
                fwd_totals[mode + "_mem"] = torch.cuda.max_memory_allocated() / 1024 / 1024

                del model, x
                _cleanup()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    _cleanup()
                else:
                    raise

        for dname, mpath, level, role, sdiv, ch, dep in LAYER_DEFS:
            sp = f"{H//sdiv}x{W//sdiv}"
            row = f"  │ {dname:<16} {ch:>4} {dep if dep else '-':>3} {sp:>9} │"
            for mode in MODES:
                if mode in fwd_times and dname in fwd_times[mode]:
                    row += f" {fwd_times[mode][dname]:>9.1f}ms"
                else:
                    row += f" {'--':>11}"
            row += " │"
            for mode in MODES:
                if mode in fwd_mems and dname in fwd_mems[mode]:
                    row += f" {fwd_mems[mode][dname]:>10.1f}MB"
                else:
                    row += f" {'--':>12}"
            row += " │"
            print(row)

        # Totals
        print(f"  │ {'─'*16} {'─'*4} {'─'*3} {'─'*9} ┼{'─'*35}─┼{'─'*38}─│")
        row = f"  │ {'TOTAL':<16} {'':>4} {'':>3} {'':>9} │"
        for mode in MODES:
            if mode in fwd_totals:
                row += f" {fwd_totals[mode]:>9.1f}ms"
            else:
                row += f" {'--':>11}"
        row += " │"
        for mode in MODES:
            k = mode + "_mem"
            if k in fwd_totals:
                row += f" {fwd_totals[k]:>10.0f}MB"
            else:
                row += f" {'--':>12}"
        row += " │"
        print(row)
        print(f"  └{'─' * 128}┘")

        # ── TRAINING ──
        print(f"\n  ┌─ Training (AMP fwd+bwd+optim) {'─' * 94}┐")
        hdr = f"  │ {'Layer':<16} {'ch':>4} {'dep':>3} {'spatial':>9} │"
        for m in MODES:
            hdr += f" {MODE_LABELS[m]+' fwd':>11}"
        hdr += " │"
        hdr += f" {'total step ms':>42}"
        hdr += " │"
        print(hdr)
        print(f"  │ {'─'*16} {'─'*4} {'─'*3} {'─'*9} ┼{'─'*35}─┼{'─'*42}─│")

        train_fwd = {}
        train_total = {}
        train_peak = {}
        for mode in MODES:
            try:
                _cleanup()
                model = _make_model(mode)
                x = torch.randn(B, 11, H, W, device=DEVICE)
                tgt = torch.randn(B, 1, H, W, device=DEVICE)

                fpl, tot, peak = profile_train_step(model, x, tgt, LAYER_NAMES, warmup=2, repeats=5)
                train_fwd[mode] = fpl
                train_total[mode] = tot
                train_peak[mode] = peak

                del model, x, tgt
                _cleanup()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    _cleanup()
                else:
                    raise

        for dname, mpath, level, role, sdiv, ch, dep in LAYER_DEFS:
            sp = f"{H//sdiv}x{W//sdiv}"
            row = f"  │ {dname:<16} {ch:>4} {dep if dep else '-':>3} {sp:>9} │"
            for mode in MODES:
                if mode in train_fwd and dname in train_fwd[mode]:
                    row += f" {train_fwd[mode][dname]:>9.1f}ms"
                else:
                    row += f" {'--':>11}"
            row += " │"
            # Only print totals in the first data row
            if dname == LAYER_DEFS[0][0]:
                for mode in MODES:
                    if mode in train_total:
                        row += f"  {MODE_LABELS[mode]}: {train_total[mode]:.0f}ms"
                    else:
                        row += f"  {MODE_LABELS[mode]}: OOM"
            row += " │" if dname == LAYER_DEFS[0][0] else ""
            print(row)

        # Totals
        print(f"  │ {'─'*16} {'─'*4} {'─'*3} {'─'*9} ┼{'─'*35}─┤")
        row_t = f"  │ {'TOTAL STEP':>16} {'':>4} {'':>3} {'':>9} │"
        for mode in MODES:
            if mode in train_total:
                row_t += f" {train_total[mode]:>9.0f}ms"
            else:
                row_t += f" {'OOM':>11}"
        row_t += " │"
        print(row_t)
        row_m = f"  │ {'PEAK MEMORY':>16} {'':>4} {'':>3} {'':>9} │"
        for mode in MODES:
            if mode in train_peak:
                row_m += f" {train_peak[mode]:>8.0f}  MB"
            else:
                row_m += f" {'OOM':>11}"
        row_m += " │"
        print(row_m)
        print(f"  └{'─' * 128}┘")

    print("\n" + "=" * 130)
    print("Done.")


if __name__ == "__main__":
    run()

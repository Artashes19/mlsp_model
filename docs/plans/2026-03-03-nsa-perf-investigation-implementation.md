# NSA Performance Investigation & Architecture Upgrades — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Profile the NSA FLOP-to-runtime paradox on A6000, then run 4 architecture experiments (250k samples each) comparing FFN residual variants and increased dims/head.

**Architecture:** Two fronts. Front 1: torch.profiler CUDA trace of NSA2DAttention to identify kernel-level bottlenecks and compile graph breaks. Front 2: Make FFN residual configurable + create bighead (base_ch=64, heads=[2,2,4,4]) variant; run 4x 250k sample experiments on h100 8-GPU.

**Tech Stack:** PyTorch, torch.profiler, torch.compile, Hydra configs, SLURM sbatch

---

### Task 1: Write NSA torch.profiler script

**Files:**
- Create: `scripts/profile_nsa_torch_profiler.py`
- Reference: `scripts/profile_nsa2d.py` (existing manual profiler)

**Step 1: Create the profiling script**

This script does what the existing `profile_nsa2d.py` does NOT do: uses `torch.profiler` with CUDA activity tracing to get real kernel-level timings, and also tests torch.compile graph break detection.

```python
#!/usr/bin/env python
"""
NSA2DAttention profiling with torch.profiler (CUDA trace) and compile graph break analysis.

Outputs:
  1. Chrome trace JSON for visual inspection in chrome://tracing
  2. Table of top 20 CUDA kernels by total time
  3. torch.compile graph break report

Usage:
    CUDA_VISIBLE_DEVICES=0 /auto/home/artashes/miniconda3/envs/dev/bin/python scripts/profile_nsa_torch_profiler.py
"""
import sys, os, gc, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from src.networks.txunet import NSA2DAttention, TransformerBlock


def make_nsa_layer(dim, heads, patch, top_n, window, gqa=1, rope=True, device="cuda", dtype=torch.bfloat16):
    """Create an NSA2DAttention layer with given params."""
    layer = NSA2DAttention(
        dim=dim, num_heads=heads,
        patch_size=patch, top_n=top_n, window_size=window,
        gqa_group_size=gqa, rope_enabled=rope, rope_base=10000.0,
    ).to(device=device, dtype=dtype)
    return layer


def profile_layer(layer, x, label, n_warmup=5, n_active=3, output_dir="profile_traces"):
    """Profile a layer's forward+backward with torch.profiler."""
    os.makedirs(output_dir, exist_ok=True)

    # Warmup
    for _ in range(n_warmup):
        x_w = x.clone().detach().requires_grad_(True)
        out = layer(x_w)
        out.sum().backward()
    torch.cuda.synchronize()

    # Profile
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for _ in range(n_active):
            x_p = x.clone().detach().requires_grad_(True)
            out = layer(x_p)
            out.sum().backward()
            torch.cuda.synchronize()

    # Export chrome trace
    trace_path = os.path.join(output_dir, f"{label}_trace.json")
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved: {trace_path}")

    # Print top CUDA kernels
    print(f"\n{'='*80}")
    print(f"TOP 30 CUDA KERNELS — {label}")
    print(f"{'='*80}")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    # Print top CPU ops
    print(f"\n{'='*80}")
    print(f"TOP 20 CPU OPS — {label}")
    print(f"{'='*80}")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    return prof


def check_compile_graph_breaks(layer, x, label):
    """Attempt torch.compile and report graph breaks."""
    print(f"\n{'='*80}")
    print(f"COMPILE GRAPH BREAK ANALYSIS — {label}")
    print(f"{'='*80}")

    import logging
    torch._logging.set_logs(graph_breaks=True)

    try:
        compiled = torch.compile(layer, fullgraph=False, backend="inductor", mode="max-autotune")
        x_c = x.clone().detach().requires_grad_(True)
        out = compiled(x_c)
        out.sum().backward()
        torch.cuda.synchronize()
        print("  torch.compile succeeded (may have graph breaks logged above)")
    except Exception as e:
        print(f"  torch.compile FAILED: {e}")
    finally:
        torch._logging.set_logs(graph_breaks=False)


def time_eager_vs_compile(layer, x, label, n_warmup=10, n_repeat=20):
    """Compare eager vs compiled wall-clock time."""
    print(f"\n{'='*80}")
    print(f"EAGER vs COMPILE TIMING — {label}")
    print(f"{'='*80}")

    # Eager timing
    for _ in range(n_warmup):
        x_w = x.clone().detach().requires_grad_(True)
        layer(x_w).sum().backward()
    torch.cuda.synchronize()

    times_eager = []
    for _ in range(n_repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x_e = x.clone().detach().requires_grad_(True)
        layer(x_e).sum().backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times_eager.append((t1 - t0) * 1000)

    mean_eager = sum(times_eager) / len(times_eager)
    print(f"  Eager:    {mean_eager:.2f} ms (fwd+bwd)")

    # Compile timing
    try:
        compiled = torch.compile(layer, fullgraph=False, backend="inductor", mode="max-autotune")
        for _ in range(n_warmup + 3):  # extra warmup for compile
            x_w = x.clone().detach().requires_grad_(True)
            compiled(x_w).sum().backward()
        torch.cuda.synchronize()

        times_compiled = []
        for _ in range(n_repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            x_c = x.clone().detach().requires_grad_(True)
            compiled(x_c).sum().backward()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times_compiled.append((t1 - t0) * 1000)

        mean_compiled = sum(times_compiled) / len(times_compiled)
        speedup = mean_eager / mean_compiled
        print(f"  Compiled: {mean_compiled:.2f} ms (fwd+bwd)  [{speedup:.2f}x]")
    except Exception as e:
        print(f"  Compiled: FAILED — {e}")


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available"); sys.exit(1)

    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")

    # Warmup CUDA
    for _ in range(3):
        a = torch.randn(256, 256, device=device); _ = a @ a.T
    del a; gc.collect(); torch.cuda.empty_cache()

    # ─── Config: L0 dimensions (the expensive level) ───
    # B=4 to fit on single A6000, same dims as real training
    B, C, H, W = 4, 48, 256, 256
    heads, patch, top_n, window = 4, 8, 32, 16

    print(f"\nConfig: B={B}, C={C}, H={H}, W={W}, heads={heads}, patch={patch}, top_n={top_n}, window={window}")
    print(f"Tokens per sample: {H*W} = {H*W:,}")

    layer = make_nsa_layer(C, heads, patch, top_n, window, gqa=1, rope=True, device=device, dtype=dtype)
    layer.train()
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)

    # 1) torch.profiler trace
    profile_layer(layer, x, "nsa_L0_bf16")

    # 2) Compile graph break analysis
    layer2 = make_nsa_layer(C, heads, patch, top_n, window, gqa=1, rope=True, device=device, dtype=dtype)
    layer2.train()
    check_compile_graph_breaks(layer2, x, "nsa_L0_bf16")

    # 3) Eager vs compile timing
    layer3 = make_nsa_layer(C, heads, patch, top_n, window, gqa=1, rope=True, device=device, dtype=dtype)
    layer3.train()
    time_eager_vs_compile(layer3, x, "nsa_L0_bf16")

    # ─── Also profile a full TransformerBlock with NSA for comparison ───
    print("\n\n" + "#" * 80)
    print("# FULL TransformerBlock WITH NSA (includes FFN)")
    print("#" * 80)

    nsa_attn = make_nsa_layer(C, heads, patch, top_n, window, gqa=1, rope=True, device=device, dtype=dtype)
    block = TransformerBlock(dim=C, heads=heads, expand=2.66, ln_eps=1e-5, attn_module=nsa_attn).to(device=device, dtype=dtype)
    block.train()
    profile_layer(block, x, "transformer_block_nsa_L0_bf16")
    time_eager_vs_compile(block, x, "transformer_block_nsa_L0_bf16")

    # ─── Compare: TransformerBlock with standard attention (SRA4 baseline) ───
    print("\n\n" + "#" * 80)
    print("# FULL TransformerBlock WITH SRA4 (baseline)")
    print("#" * 80)

    block_sra = TransformerBlock(dim=C, heads=heads, expand=2.66, ln_eps=1e-5, kv_stride=4, rope_enabled=True, rope_base=10000.0).to(device=device, dtype=dtype)
    block_sra.train()
    profile_layer(block_sra, x, "transformer_block_sra4_L0_bf16")
    time_eager_vs_compile(block_sra, x, "transformer_block_sra4_L0_bf16")

    print("\n\nDone. Check profile_traces/ for Chrome trace JSONs.")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add scripts/profile_nsa_torch_profiler.py
git commit -m "Add torch.profiler NSA script with compile break analysis"
```

---

### Task 2: Run profiling on ap

**Step 1: Sync code to ap**

```bash
# From np (local), push to origin and pull on ap
git push origin dev-attn
ssh artashes@ap.yc2.io "cd mlsp_model/dev-clean && git fetch origin && git checkout dev-attn && git pull"
```

**Step 2: Run the profiler on ap (A6000)**

```bash
ssh artashes@ap.yc2.io "cd mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=0 /auto/home/artashes/miniconda3/envs/dev/bin/python scripts/profile_nsa_torch_profiler.py 2>&1 | tee profile_nsa_output.log"
```

**Step 3: Analyze output**

Read `profile_nsa_output.log` and the Chrome traces. Identify:
- Which CUDA kernels take the most time
- Whether torch.compile helps or hurts
- Exact graph break locations
- Memory copy overhead from `.contiguous()` calls

Document findings in a brief section appended to the design doc.

---

### Task 3: Make FFN residual configurable (Change 2A)

**Files:**
- Modify: `src/networks/txunet.py:714-748` (GatedDepthwiseFFN)
- Modify: `src/networks/txunet.py:828-834` (TransformerBlock)
- Modify: `src/networks/txunet.py:868-874` (WindowedTransformerBlock)
- Modify: `src/networks/txunet.py:1007-1030` (TxUNetModel.__init__)
- Modify: `configs/network/txunet.yaml`
- Test: `tests/test_ffn_residual.py` (new)

**Step 1: Write the failing test**

Create `tests/test_ffn_residual.py`:

```python
"""Test FFN residual path variants."""
import pytest
import torch
from src.networks.txunet import GatedDepthwiseFFN, TransformerBlock


def test_ffn_internal_residual_default():
    """Default (internal_residual=True): FFN(x) = proj(g) + x."""
    ffn = GatedDepthwiseFFN(dim=48, expand=2.66, internal_residual=True)
    x = torch.randn(1, 48, 8, 8)
    out = ffn(x)
    assert out.shape == x.shape
    # With internal residual, output should be close to input when weights are small
    # (not a strict test, just shape check)


def test_ffn_no_internal_residual():
    """Standard pre-LN: FFN(x) = proj(g), no internal residual."""
    ffn = GatedDepthwiseFFN(dim=48, expand=2.66, internal_residual=False)
    x = torch.randn(1, 48, 8, 8)
    out = ffn(x)
    assert out.shape == x.shape


def test_transformer_block_standard_residual():
    """Standard pre-LN: x = x + ffn(norm2(x)) when ffn_internal_residual=False."""
    block = TransformerBlock(dim=48, heads=4, expand=2.66, ffn_internal_residual=False)
    x = torch.randn(1, 48, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_transformer_block_internal_residual():
    """Legacy: x = ffn(norm2(x)) when ffn_internal_residual=True."""
    block = TransformerBlock(dim=48, heads=4, expand=2.66, ffn_internal_residual=True)
    x = torch.randn(1, 48, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_backward_both_variants():
    """Both variants should produce valid gradients."""
    for internal_res in [True, False]:
        block = TransformerBlock(dim=48, heads=4, expand=2.66, ffn_internal_residual=internal_res)
        x = torch.randn(1, 48, 16, 16, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
```

**Step 2: Run test to verify it fails**

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_ffn_residual.py -v
```

Expected: FAIL — `GatedDepthwiseFFN.__init__() got an unexpected keyword argument 'internal_residual'`

**Step 3: Implement the changes**

**`src/networks/txunet.py` — GatedDepthwiseFFN (lines 714-748):**

Add `internal_residual: bool = True` parameter to `__init__`. In `forward`, conditionally add the residual:

```python
class GatedDepthwiseFFN(nn.Module):
    """
    Gated FFN with two SEPARATE DConvBlock paths.

    When internal_residual=True (legacy):  output = proj(g) + x
    When internal_residual=False (standard): output = proj(g)
    """

    def __init__(self, dim: int, expand: float = 2.66, internal_residual: bool = True) -> None:
        super().__init__()
        hidden = int(round(dim * expand))
        self.internal_residual = internal_residual

        self.branch1 = DConvBlock(dim, hidden)
        self.branch2 = DConvBlock(dim, hidden)

        self.act = nn.GELU()
        self.proj = nn.Conv2d(hidden, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.branch1(x)
        v = self.act(self.branch2(x))
        u = torch.clamp(u, -256.0, 256.0)
        g = u * v
        out = self.proj(g)
        if self.internal_residual:
            out = out + x
        return out
```

**`src/networks/txunet.py` — TransformerBlock (lines 804-834):**

Add `ffn_internal_residual: bool = True` parameter. When `False`, add external residual:

```python
def __init__(
    self,
    dim: int,
    heads: int,
    expand: float = 2.66,
    ln_eps: float = 1e-5,
    kv_stride: int = 1,
    rope_enabled: bool = False,
    rope_base: float = 10000.0,
    attn_module: nn.Module | None = None,
    ffn_internal_residual: bool = True,
) -> None:
    super().__init__()
    self._ffn_internal_residual = ffn_internal_residual
    self.norm1 = LayerNorm2d(dim, eps=ln_eps)
    # ... attn init unchanged ...
    self.norm2 = LayerNorm2d(dim, eps=ln_eps)
    self.ffn = GatedDepthwiseFFN(dim, expand, internal_residual=ffn_internal_residual)

def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = x + self.attn(self.norm1(x))
    if self._ffn_internal_residual:
        x = self.ffn(self.norm2(x))          # legacy: residual inside FFN
    else:
        x = x + self.ffn(self.norm2(x))      # standard: residual outside FFN
    return x
```

Apply the same change to **WindowedTransformerBlock** (lines 842-874).

**`src/networks/txunet.py` — TxUNetModel.__init__ (line 1007):**

Add `ffn_internal_residual: bool = True` parameter. Thread it through to `make_blocks` and `_make_nsa_block_seq`:

```python
def __init__(
    self,
    ...
    nsa_levels: Sequence[int] | None = None,
    ffn_internal_residual: bool = True,
    **kwargs,
) -> None:
    ...
    ffn_kwargs = {"ffn_internal_residual": ffn_internal_residual}
```

Pass `ffn_internal_residual` into every `TransformerBlock` and `make_blocks` call via `block_kwargs`.

**`configs/network/txunet.yaml`:**

Add: `ffn_internal_residual: true  # true=legacy (residual inside FFN), false=standard pre-LN`

**Step 4: Run tests to verify they pass**

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_ffn_residual.py -v
```

Expected: All 5 tests PASS.

**Step 5: Run existing NSA tests to check nothing broke**

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_nsa_2d.py tests/test_nsa_2d_gpu.py -v
```

Expected: All existing tests PASS.

**Step 6: Commit**

```bash
git add src/networks/txunet.py configs/network/txunet.yaml tests/test_ffn_residual.py
git commit -m "feat: make FFN internal residual configurable (2A)"
```

---

### Task 4: Create bighead config (Change 2B)

**Files:**
- Create: `configs/network/txunet_bighead.yaml`

**Step 1: Create the config**

```yaml
_target_: src.networks.TxUNetModel

name: txunet_bighead
in_ch: 11
out_ch: 1
base_ch: 64
depths: [ 4, 6, 6, 8 ]
heads: [ 2, 2, 4, 4 ]
expand: 2.66
use_checkpoint: False
ln_eps: 1e-5
window0: null
window0_stride: null
sra0_enabled: False
sra0_stride: 4
rope_enabled: True
rope_base: 10000.0
nsa_enabled: True
nsa_patch_sizes: [8, 8, 4, 4]
nsa_top_n: [32, 16, 16, 16]
nsa_window_sizes: [16, 16, 8, 8]
nsa_gqa_group_size: 1
nsa_levels: [0, 1]
ffn_internal_residual: true
```

**Step 2: Smoke test the config instantiates**

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -c "
from src.networks.txunet import TxUNetModel
import torch
m = TxUNetModel(in_ch=11, out_ch=1, base_ch=64, depths=(4,6,6,8), heads=(2,2,4,4),
    expand=2.66, nsa_enabled=True, nsa_levels=[0,1], sra0_enabled=False,
    rope_enabled=True, rope_base=10000.0)
x = torch.randn(1, 11, 64, 64)
print('params:', sum(p.numel() for p in m.parameters()) / 1e6, 'M')
print('out:', m(x).shape)
"
```

Expected: Prints parameter count (~2-3x more than baseline) and output shape `[1, 1, 64, 64]`.

**Step 3: Commit**

```bash
git add configs/network/txunet_bighead.yaml
git commit -m "feat: add txunet_bighead config (base_ch=64, heads=[2,2,4,4])"
```

---

### Task 5: Create 4 sbatch files for 250k experiments

**Files:**
- Create: `sbatch_250k_baseline.sbatch`
- Create: `sbatch_250k_stdresid.sbatch`
- Create: `sbatch_250k_bighead.sbatch`
- Create: `sbatch_250k_bighead_stdresid.sbatch`

All based on `sbatch_e2.sbatch` template. 250k samples = 40 epochs × 6240 samples/epoch.

The WSD scheduler with `warmup_samples=6250` gives warmup through ~1 epoch, then the first stable phase runs to 50k samples (~8 epochs), first decay to 56250 (~9 epochs). For a 250k run, we'll see warmup + first 3 stable/decay cycles. Override `max_epochs=40` to cap at ~250k samples.

**Step 1: Create baseline sbatch**

`sbatch_250k_baseline.sbatch` — Partial NSA, current FFN (internal residual), base_ch=48:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=250k_baseline
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=160
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

WORK_DIR="/home/indoor/mlsp_wair_d"
mkdir -p "${WORK_DIR}/logs"
cd "$WORK_DIR"

export WANDB_API_KEY="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"
export HYDRA_FULL_ERROR=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
nvidia-smi -pm 1 2>/dev/null || true

echo "[$(date)] Starting 250k baseline (partial NSA, internal FFN residual, base_ch=48)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

torchrun --standalone --nproc_per_node=8 run.py --config-name=train_synthetic \
  strategy=ddp \
  'trainer.devices=8' \
  'trainer.max_epochs=40' \
  'network.nsa_enabled=True' \
  'network.sra0_enabled=False' \
  'network.nsa_gqa_group_size=1' \
  'network.nsa_levels=[0,1]' \
  'network.rope_enabled=True' \
  'network.rope_base=10000.0' \
  'network.ffn_internal_residual=True' \
  'loggers.wandb.name=250k_baseline_pnsa01'
```

**Step 2: Create stdresid sbatch**

`sbatch_250k_stdresid.sbatch` — same as baseline but `ffn_internal_residual=False`:

Same as above but:
```
  'network.ffn_internal_residual=False' \
  'loggers.wandb.name=250k_stdresid_pnsa01'
```
Job name: `250k_stdresid`

**Step 3: Create bighead sbatch**

`sbatch_250k_bighead.sbatch` — base_ch=64, heads=[2,2,4,4], internal residual:

```
  'network.base_ch=64' \
  'network.heads=[2,2,4,4]' \
  'network.nsa_enabled=True' \
  'network.sra0_enabled=False' \
  'network.nsa_gqa_group_size=1' \
  'network.nsa_levels=[0,1]' \
  'network.rope_enabled=True' \
  'network.rope_base=10000.0' \
  'network.ffn_internal_residual=True' \
  'loggers.wandb.name=250k_bighead_pnsa01'
```
Job name: `250k_bighead`

**Step 4: Create bighead+stdresid sbatch**

`sbatch_250k_bighead_stdresid.sbatch` — both changes combined:

```
  'network.base_ch=64' \
  'network.heads=[2,2,4,4]' \
  'network.ffn_internal_residual=False' \
  'loggers.wandb.name=250k_bighead_stdresid_pnsa01'
```
Job name: `250k_bighead_stdresid`

**Step 5: Commit**

```bash
git add sbatch_250k_*.sbatch
git commit -m "feat: add 4 sbatch files for 250k architecture comparison experiments"
```

---

### Task 6: Smoke test on ap

**Step 1: Push and sync to ap**

```bash
git push origin dev-attn
ssh artashes@ap.yc2.io "cd mlsp_model/dev-clean && git fetch origin && git checkout dev-attn && git pull"
```

**Step 2: Smoke test baseline config**

```bash
ssh artashes@ap.yc2.io "cd mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=0 /auto/home/artashes/miniconda3/envs/dev/bin/python run.py --config-name=train_synthetic fast_dev=true \
  'network.nsa_enabled=True' 'network.sra0_enabled=False' \
  'network.nsa_gqa_group_size=1' 'network.nsa_levels=[0,1]' \
  'network.rope_enabled=True' 'network.rope_base=10000.0' \
  'network.ffn_internal_residual=True' \
  'trainer.devices=[0]'"
```

Expected: Completes a few train + val batches without error.

**Step 3: Smoke test stdresid config**

Same as above but `'network.ffn_internal_residual=False'`.

**Step 4: Smoke test bighead config**

```bash
ssh artashes@ap.yc2.io "cd mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=0 /auto/home/artashes/miniconda3/envs/dev/bin/python run.py --config-name=train_synthetic fast_dev=true \
  'network.base_ch=64' 'network.heads=[2,2,4,4]' \
  'network.nsa_enabled=True' 'network.sra0_enabled=False' \
  'network.nsa_gqa_group_size=1' 'network.nsa_levels=[0,1]' \
  'network.rope_enabled=True' 'network.rope_base=10000.0' \
  'network.ffn_internal_residual=True' \
  'trainer.devices=[0]'"
```

**Step 5: Smoke test bighead+stdresid config**

Same as above but `'network.ffn_internal_residual=False'`.

**Step 6: Run existing tests**

```bash
ssh artashes@ap.yc2.io "cd mlsp_model/dev-clean && /auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_nsa_2d.py tests/test_nsa_2d_gpu.py tests/test_ffn_residual.py -v"
```

Expected: All pass.

---

### Task 7: Push dev-attn and finalize

**Step 1: Final push**

```bash
git push origin dev-attn
```

**Step 2: Verify remote is up to date**

```bash
git log --oneline -5
git log --oneline origin/dev-attn -5
```

Both should show the same commits.

---

## Execution Order Summary

| Task | Description | Depends On | Estimated Time |
|---|---|---|---|
| 1 | Write profiler script | — | 5 min |
| 2 | Run profiling on ap | 1 | 10 min |
| 3 | Make FFN residual configurable (2A) | — | 15 min |
| 4 | Create bighead config (2B) | 3 | 5 min |
| 5 | Create 4 sbatch files | 3, 4 | 10 min |
| 6 | Smoke test on ap | 3, 4, 5 | 15 min |
| 7 | Push dev-attn | 6 | 2 min |

Tasks 1 and 3 can run in parallel. Task 2 needs Task 1. Tasks 4-5 need Task 3. Task 6 needs all prior tasks. Task 7 is last.

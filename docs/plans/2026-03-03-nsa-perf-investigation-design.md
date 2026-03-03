# NSA Performance Investigation & Architecture Upgrades

**Date**: 2026-03-03
**Branch**: dev-attn
**Status**: Design

## Context

Phase C results from 1M sample synthetic pretraining on h100 8-GPU node:

| Variant | FLOPs (fwd+bwd) | Wall-clock | Loss (RMSE dB) | Stable? |
|---|---|---|---|---|
| SRA4 (no compile) | 162.2T | 14h00m | baseline | Yes |
| Full NSA MHA (compile) | 53.9T | 19h18m | ~baseline | NaN at ~80% |
| Partial NSA L0+L1 (compile) | 60.2T | 17h52m | ~baseline | Yes |
| GQA4 (compile) | — | 12h00m | ~baseline | NaN at ~80% |

Key findings:
- All variants converge to within 0.1 dB RMSE — attention mechanism is NOT the loss bottleneck
- NSA has 3x fewer FLOPs but is 38% slower than SRA4 (FLOP-runtime paradox)
- Full NSA and GQA are unstable (NaN) at L2+L3; partial NSA (L0+L1 only) is stable
- Partial NSA is the preferred configuration going forward

## Two Fronts

### Front 1: NSA Runtime Profiling

**Goal**: Understand and fix the FLOP-to-runtime paradox.

**Approach**: Profile on ap (A6000) → analyze → fix → re-benchmark.

#### Step 1: Profiling Script

Write `scripts/profile_nsa_full.py` that:
- Instantiates one NSA2DAttention layer at L0 dimensions (B=24, C=48, H=W=256, heads=4, patch=8, top_n=32, window=16)
- Runs forward + backward with `torch.profiler` (CUDA activities trace)
- Exports Chrome trace (.json) for manual inspection
- Summarizes: per-kernel timing, total SDPA time, total reshape time, total gather/topk time

Run on ap (A6000) to match the GPU used for development.

#### Step 2: Compile Break Analysis

Run with `TORCH_LOGS=graph_breaks` to enumerate every point where torch.compile gives up fusing in NSA.

#### Known Possible Issues (to validate with profiler)

1. **`torch.no_grad()` in importance scoring** — compile graph break
2. **`torch.cuda.mem_get_info()`** — CUDA sync point + compile graph break
3. **6+ `.contiguous()` calls** per forward pass — memory copies that don't appear in FLOPs
4. **3 separate SDPA kernel launches** vs SRA4's single SDPA
5. **`topk` on small dimensions** — memory-bound, poorly optimized
6. **Triton `SelectionAttn2D.apply`** — custom autograd.Function is a graph break
7. **Compiler overhead** on unfusable code may make compile *slower* than eager

#### Step 3: Targeted Fixes (after profiling confirms)

- Replace `torch.no_grad()` → `.detach()` on importance inputs
- Replace `mem_get_info()` → pre-computed fixed chunk size
- Investigate reshape chain optimization
- Consider compile mode options (`reduce-overhead` vs `max-autotune`)

#### Step 4: Re-benchmark

Partial NSA with fixes vs SRA4, both with and without compile, 250k samples on h100.

**Success criteria**: Partial NSA within 20% of SRA4 wall-clock.

---

### Front 2: Architecture Upgrades

**Goal**: Break the loss plateau where all attention variants converge to same RMSE.

#### Change 2A: Fix FFN Residual Path

**Current (non-standard):**
```python
# TransformerBlock.forward:
x = x + self.attn(self.norm1(x))    # attention + residual over raw x
x = self.ffn(self.norm2(x))          # FFN gets norm2(x), adds norm2(x) back internally

# GatedDepthwiseFFN.forward:
return self.proj(g) + x              # x here is norm2(input), not raw input
```

The residual bypasses FFN_body but starts from the **normalized** signal, not the raw signal.
The un-normalized features from attention don't have a direct skip path through the FFN stage.

Note: This was a deliberate stabilization choice — previous runs had NaN without it.

**Proposed (standard pre-LN):**
```python
# TransformerBlock.forward:
x = x + self.attn(self.norm1(x))
x = x + self.ffn(self.norm2(x))     # residual over raw x

# GatedDepthwiseFFN.forward:
return self.proj(g)                   # no internal residual
```

**Test plan**: Run both variants on 250k samples, compare loss curves and stability.

#### Change 2B: Increase dims/head

**Current**: base_ch=48, heads=[4,4,8,8] → 12 dims/head at L0/L1, 24 at L2/L3.

12 dims/head is very small (GPT uses 64-128, ViT uses 64). Attention with only 12 dims
per head has limited representational power — different attention mechanisms may produce
nearly equivalent outputs because there isn't enough capacity per head to differentiate.

**Proposed**: base_ch=64, heads=[2,2,4,4] → 32 dims/head everywhere.

Impact:
- ~1.78x more parameters (base_ch 48→64)
- 32 dims/head gives each head enough capacity to learn distinct attention patterns
- Fewer heads (2 at L0/L1) but each head is much more expressive
- L2: 256ch / 4 heads = 64 dims/head (matches ViT standard)
- Bottleneck: 512ch / 4 heads = 128 dims/head

**Test plan**: Run on 250k samples, compare against baseline with same attention (partial NSA).

---

## Experiment Plan

All experiments use 250k synthetic samples on h100 8-GPU, partial NSA (L0+L1), compile enabled.

| Run | Config | What it tests |
|---|---|---|
| **baseline** | Current: base_ch=48, heads=[4,4,8,8], internal FFN residual | Control |
| **2A-stdresid** | Standard pre-LN residual (no internal FFN residual) | Residual path fix |
| **2B-bighead** | base_ch=64, heads=[2,2,4,4] | dims/head increase |
| **2A+2B** | Both changes combined | Interaction effect |

If any 250k run shows >0.3 dB improvement or significant loss curve shape change, extend to full 1M.

## Files to Modify

### Front 1 (profiling)
- `scripts/profile_nsa_full.py` (new) — profiling script

### Front 2 (architecture)
- `src/networks/txunet.py` — make FFN residual configurable (flag)
- `configs/network/txunet.yaml` — add `ffn_internal_residual: true` (default preserves current behavior)
- `configs/network/txunet_bighead.yaml` (new) — base_ch=64, heads=[2,2,4,4] variant
- 4 sbatch files for the experiments

## Order of Execution

1. Write profiling script, run on ap → analyze results
2. Implement 2A (configurable FFN residual)
3. Implement 2B (txunet_bighead config)
4. Create sbatch files for 250k experiments
5. Smoke test on ap
6. Submit to h100

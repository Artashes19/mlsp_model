# NSA Forward and dQ v3 Redesign

**Date**: 2026-03-06
**Branch**: dev-attn
**Status**: Design

## Problem Statement

The per-query NSA path has already fixed the catastrophic `dK/dV` backward bottleneck with the Tilda-style compact reduction, but the next profiler runs show the remaining runtime is dominated by:

- `_sel_perq_bwd_dq_kernel`
- `_sel_perq_fwd_kernel`

This is true on both A6000 and DGX A100, and it is especially severe on plain MHA (`G=1`).

## Current Evidence

### A100, GQA (`Hq=4, Hkv=1, G=4`, `T=65536`, bf16)

From `artifacts/nsa_diagnostics/torch_profiler_compact_v2_gqa_hq4_hkv1_g4_dgx_a100_gpu1_20260305_000505.txt`:

- `dQ`: `~15.34 ms`
- `forward`: `~11.66 ms`
- `dK/dV`: `~4.55 ms`

### A100, MHA (`Hq=4, Hkv=4, G=1`, `T=65536`, bf16)

From `artifacts/nsa_diagnostics/torch_profiler_compact_v2_mha_hq4_hkv4_g1_dgx_a100_gpu1_20260305_000505.txt`:

- `dQ`: `~60.88 ms`
- `forward`: `~45.59 ms`
- `dK/dV`: `~4.55 ms`

The MHA jump is the key signal. `dK/dV` stays nearly constant while `forward` and `dQ` scale badly.

## Root Cause Hypothesis

The current unified `forward` and `dQ` kernels still carry grouped-head structure that is inefficient for small `G`, especially `G=1`.

Current code forces:

- `BLOCK_G = max(16, next_power_of_2(G))`

So for MHA:

- real grouped-head width is `1`
- kernel still allocates and computes for `16`

This wastes lanes, register space, and SRAM bandwidth in the two kernels that dominate runtime.

## Locked Design Decisions

1. Keep a single unified kernel family for MHA and GQA.
2. Keep direct consumption of `block_idx [B, h_kv, T, top_n]`.
3. Do not add new preprocessing or compaction for `forward` or `dQ`.
4. Remain strictly non-causal.
5. Numerical behavior may drift modestly if naive parity tests still pass with agreed tolerances.
6. Benchmark and tune on DGX A100 now.
7. Retune for H100 later, on `artashes@h100.yc2.io`, before final training runs.

## Proposed Design

### 1. Unified query-block kernel family

Move both `forward` and `dQ` from a query-single dominant execution style to query-block kernels:

- Grid: `(B * h_kv, ceil_div(T, BLOCK_Q))`
- Each program handles a tile of queries for one `(batch, kv_head)` pair

This keeps the existing query-centric `block_idx` contract and removes the need for a new staging pipeline.

### 2. Replace padded `BLOCK_G` with real head tile `BLOCK_H`

Introduce a separate meta-parameter for grouped-head width:

- `BLOCK_H` tracks actual grouped-head width
- do not force `BLOCK_H >= 16`

Expected choices:

- `G=1 -> BLOCK_H=1`
- `G=2 -> BLOCK_H=2`
- `G=4 -> BLOCK_H=4`
- `G=8 -> BLOCK_H=8`

This is the main fix for MHA inefficiency.

### 3. Forward v3

Forward kernel changes:

- process `BLOCK_Q` queries per program
- process `BLOCK_H` grouped heads per program
- keep direct per-query patch reads from `block_idx`
- keep online softmax accumulation
- permit limited simplification if profiler shows meaningful gain and parity still passes

The main goal is to stop paying a 16-lane grouped-head tax when `G=1`.

### 4. dQ v3

Backward `dQ` changes:

- match the same `(B * h_kv, query-block)` mapping as forward
- compute `dq` for a query tile instead of one query at a time
- use `BLOCK_H` instead of padded `BLOCK_G`
- preserve current `dK/dV` compact kernel and integration

This keeps the kernel family structurally aligned and reduces duplicated tuning logic.

### 5. H100-aware but A100-first tuning

Immediate tuning target:

- DGX A100 only

Later retune target:

- H100 host `artashes@h100.yc2.io`

Reason:

- H100 is currently fully occupied
- A100 is free and sufficient to validate the structural redesign
- final launch parameters should still be retuned on H100 before training

## Why This Design

- It attacks the measured bottleneck directly.
- It preserves the correct per-query semantics already shipped.
- It avoids introducing another staging/reordering path that would add memory traffic and debugging surface.
- It keeps one kernel family, as requested.
- It is the lowest-risk design that still has real upside on both A100 and future H100 runs.

## Expected Outcomes

### GQA

Moderate gain expected:

- `forward` should improve somewhat
- `dQ` should improve somewhat
- less upside than MHA because grouped-head reuse is already useful

### MHA

Largest gain expected:

- `forward` should improve materially
- `dQ` should improve materially

Reason:

- MHA currently pays the biggest penalty from padded grouped-head execution

## Validation Plan

### Correctness

Keep existing naive parity tests:

- MHA forward
- MHA `dQ`
- MHA `dK/dV`
- GQA forward
- GQA `dQ`
- GQA `dK/dV`

`dK/dV` tests must remain green even though this redesign does not touch `dK/dV`, because integration breakage is still possible.

### Performance

Required benchmarks:

1. A100 wall-clock benchmark for:
   - current compact-v2 baseline
   - forward/dQ v3
2. A100 `torch.profiler` for:
   - GQA (`Hq=4, Hkv=1, G=4`)
   - MHA (`Hq=4, Hkv=4, G=1`)
3. direct kernel-level timings for:
   - forward only
   - backward q-only
   - backward qkv

### H100 follow-up

When H100 becomes available:

1. rerun wall-clock benchmark
2. rerun profiler
3. tune `BLOCK_Q`, `BLOCK_H`, `BLOCK_KV`, and `num_warps`
4. lock H100 launch heuristics before full training

## Risks and Mitigations

1. Risk: query-block forward increases register pressure.
   - Mitigation: keep `BLOCK_Q` small at first and tune upward only with profiler evidence.

2. Risk: removing padded grouped-head width changes numerical behavior.
   - Mitigation: rely on existing naive parity tests and explicit tolerance checks.

3. Risk: A100-optimal tile sizes do not transfer to H100.
   - Mitigation: treat A100 results as structural validation only; retune on H100 later.

4. Risk: unified kernel becomes too complex.
   - Mitigation: keep one family, but allow separate launch heuristics and meta-parameter tables for different `G` regimes.

## Deliverables

1. Unified `forward v3` kernel using real grouped-head width.
2. Unified `dQ v3` kernel using real grouped-head width.
3. A100 profiler and benchmark artifacts showing before/after changes.
4. H100 retuning checklist for the final training environment.

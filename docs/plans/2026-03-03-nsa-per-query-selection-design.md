# NSA Per-Query Selection Redesign

**Date**: 2026-03-03
**Branch**: dev-attn
**Status**: Design

## Problem Statement

Current NSA selection behavior is not aligned with the intended NSA semantics.

- In MHA mode (`gqa_group_size=1`), selected patches are currently shared globally per sample (`top_idx` effectively `[B, top_k]`), not per query token.
- In GQA mode (`gqa_group_size>1`), selection is per KV head (`[B, h_kv, top_k]`) but still shared across all query tokens.

This removes the core NSA property where different query tokens route to different sparse blocks.

## Locked Design Decisions

1. Remove shared selection mode entirely.
2. Use per-query block indices everywhere: `block_idx [B, h_kv, T, top_k]`.
3. GQA semantics: per-KV-head block indices (`h_kv` axis), not per-Q-head.
4. MHA semantics: per-head block indices naturally follow (`gqa_group_size=1 => h_kv == h_q`).
5. Aggregate grouped head evidence with summation (`sum`) when scoring blocks.
6. No forced block selection (no "always include first/last" rule).
7. Strictly non-causal behavior for this 2D task (no causal path).

## Why This Design

- Restores true NSA behavior: each query token selects its own sparse context.
- Keeps GQA efficiency: one block list per KV group token, broadcast to grouped Q heads.
- Preserves architectural consistency between MHA and GQA via one tensor contract.

## Proposed Tensor Contracts

### Selection Index Contract (new)

- `block_idx`: `torch.int32`, contiguous
- Shape: `[B, h_kv, T, top_k]`

### Existing tensors

- `q`: `[B, h_q, T, d]`
- `k`, `v`: `[B, h_kv, T, d]`
- `G = gqa_group_size = h_q // h_kv`

## Migration Strategy

## Phase 1 (Safety-first semantic fix)

Goal: ship correct per-query behavior with a reference implementation before kernel rewrite.

### A) Per-query top-k scoring

Compute per-query per-KV-head block scores in chunks over `T`.

- Reshape grouped Q: `q_grouped = q.view(B, h_kv, G, T, d)`
- For each chunk:
  - `logits = einsum("bhgtd,bhnd->bhgtn", q_chunk * scale, k_cmp)`
  - `probs = softmax(logits, dim=-1)`
  - `scores = probs.sum(dim=2)`  # sum across grouped Q heads
  - `topk(scores, dim=-1)` => `block_idx_chunk [B, h_kv, chunk, top_k]`
- Concatenate chunks => `block_idx [B, h_kv, T, top_k]`

### B) Reference selection compute path

Use a chunked PyTorch path (non-causal) that consumes `block_idx`:

- Gather selected K/V patch tokens per query from patch tensors.
- Compute attention and output per query token.
- Broadcast per-KV-head outputs to grouped Q heads (for GQA).

This path prioritizes correctness and debuggability.

### C) Remove shared mode logic

Delete/replace any code path that builds shared `top_idx [B, top_k]` or `[B, 1, top_k]`.

## Phase 2 (Triton acceleration)

Goal: recover and exceed runtime while preserving Phase-1 semantics.

### A) Triton per-query selection forward + dQ

Add kernels that directly consume `block_idx [B, h_kv, T, top_k]`.

### B) Triton dK/dV path

1. Implement functional one-pass dK/dV for per-query indices.
2. Upgrade to two-pass reduction design to reduce global atomic contention.

This follows top-performing kernel strategy direction (Tilde/FLA-style two-pass reduction ideas), adapted to our 2D patch layout and non-causal requirements.

## Validation Plan

### Correctness gates

1. Forward parity vs reference path (MHA and GQA).
2. Gradient parity (`dq/dk/dv`) vs reference on small-medium shapes.
3. Assert output/index shapes and dtypes.

### Behavioral gates

1. Verify `block_idx` shape is always `[B, h_kv, T, top_k]`.
2. Verify no shared-topk code path remains.
3. Verify strict non-causal computation.

### Performance gates

1. Per-layer benchmarks at `T = 8192, 16384, 32768`.
2. `torch.profiler` kernel table before/after Triton phase.
3. Track dK/dV kernel share and memory-copy overhead.

### Stability gates

1. A6000 training smoke (forward+backward loops).
2. Multi-GPU training smoke in target environment.
3. NaN/Inf checks under AMP/bf16.

## Risks and Mitigations

1. Risk: Phase-1 reference path may be slower.
   - Mitigation: keep Phase-1 as correctness gate only, then move to Triton phase quickly.

2. Risk: Per-query dK/dV introduces high atomic pressure.
   - Mitigation: prioritize two-pass reduction upgrade in Phase 2.

3. Risk: Memory pressure from per-query gathered tensors.
   - Mitigation: chunk over `T`, reuse buffers, and cap chunk size with OOM fallback.

## Deliverables

1. Updated selection branch semantics with per-query indices.
2. Shared mode fully removed.
3. Reference correctness tests and profiler-backed benchmarks.
4. Triton per-query acceleration path with two-pass dK/dV optimization.

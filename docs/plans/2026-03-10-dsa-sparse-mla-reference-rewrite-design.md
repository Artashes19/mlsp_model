# DSA Sparse MLA Reference Rewrite Design

## Goal

Rewrite the pure-PyTorch sparse MLA reference path so it stays paper-shaped and memory-sane: keep KV in KV-head space, avoid `repeat_interleave(k/v)` in sparse runtime, and avoid materializing huge selected `K/V` tensors.

## Problem

The current sparse MLA reference path is correct enough for small tests but structurally wrong for throughput and memory:

1. `forward_sparse_from_indices()` expands KV across query heads with `repeat_interleave`, which defeats the MQA/GQA sharing that DSA/MLA relies on for efficiency.
2. `gather_sparse_mla_tokens()` materializes explicit selected `K/V` tensors shaped like `[B, h_q, T, K, D]`.
3. At `256x256`, these selected tensors are multi-gigabyte allocations, which is the main remaining runtime/memory mismatch after the streaming indexer fix.

This means the current DSA path is still a correctness-first eager materialization path, not a paper-shaped reference.

## Scope

This slice rewrites only the sparse MLA reference path.

In scope:

1. `src/ops/dsa_sparse_mla.py`
2. `src/networks/dsa_2d.py`
3. `tests/test_dsa_2d_sparse_attention.py`
4. `tests/test_dsa_2d_regression.py`
5. `tests/helpers/dsa_reference.py`
6. `docs/dsa_memory.md`

Out of scope:

1. changing selector semantics
2. changing indexer semantics
3. integrating FlashMLA / DeepGEMM kernels
4. optimizing the selector merge loop in this slice

## Target Behavior

The sparse MLA reference path should:

1. keep `k` and `v` in `[B, h_kv, T, D]`
2. keep `q` in `[B, h_q, T, D]`
3. map query heads to KV heads by `kv_head = q_head // G`
4. consume selected token indices without building `k_selected` / `v_selected`
5. compute sparse attention output directly from original `k/v`
6. preserve current numerical behavior on small parity cases

## Design

### 1. Remove selected `K/V` materialization

Delete the current gather-based strategy from the sparse runtime path.

Instead of:

1. building `k_selected = gather_sparse_mla_tokens(k, idx)`
2. building `v_selected = gather_sparse_mla_tokens(v, idx)`
3. doing dense math over `[B, h_q, T, K, D]`

The new path will iterate over query blocks and selected-token blocks and read from original `k/v` on demand.

### 2. Keep KV in KV-head space

Do not call `repeat_interleave(k, gqa_group_size)` or `repeat_interleave(v, gqa_group_size)` inside sparse runtime.

For each query head `h_q`, compute:

- `h_kv = h_q // gqa_group_size`

Then use that KV head directly when reading selected keys and values.

This restores the structural sharing expected by the paper.

### 3. Streaming sparse MLA reference

Implement a pure-PyTorch streaming helper in `src/ops/dsa_sparse_mla.py`.

Inputs:

- `q[B, h_q, T, D]`
- `k[B, h_kv, T, D]`
- `v[B, h_kv, T, D_v]`
- `idx[B, T, K]`
- `gqa_group_size`
- `softmax_scale`

Behavior:

1. process query tokens in blocks (`BLOCK_Q`)
2. process selected tokens in blocks (`BLOCK_K`)
3. for each query head, map to the proper KV head
4. gather only the current selected token block from original `k/v`
5. accumulate attention using online softmax state (`max`, `lse`, weighted sum)
6. write output directly into `[B, h_q, T, D_v]`

This is still eager PyTorch, but it avoids the huge persistent gathered buffers.

### 4. Reference-first, not performance-first

This rewrite is still a reference path.

Success criterion is:

1. remove the worst materialization mismatch
2. preserve sparse-vs-dense correctness gates
3. materially reduce memory footprint and runtime enough to expose the next real bottleneck

We do not expect this slice alone to match paper throughput.

## Testing Strategy

Strict TDD.

### Easy tests

1. sparse runtime no longer calls `gather_sparse_mla_tokens`
2. sparse runtime no longer uses `repeat_interleave` in the sparse path
3. new helper validates head mapping and bounds

### Hard tests

1. sparse forward matches old reference on small shapes
2. sparse backward still matches dense at `topk=T`
3. repeated and unsorted indices still behave correctly
4. GQA head mapping is correct for `G>1`

### Extra-hard tests

1. regression test that selected `K/V` tensors are not materialized via the old helper
2. parity under duplicated indices
3. parity under adversarial index orders

## Expected Outcome

After this slice:

1. DSA sparse runtime should remain correct on all current DSA tests
2. `256x256` streaming mode should still run
3. the profiler should no longer show giant explicit selected `K/V` gather buffers as the main sparse-MLA problem
4. if runtime is still poor, the next target becomes either:
   - selector merge overhead
   - or the remaining per-block sparse gather path

## Why This Keeps Us Closer To The Paper

This rewrite reduces divergence from DSA/MLA, because:

1. DSA relies on MQA/GQA sharing for efficiency
2. FlashMLA-style sparse attention consumes original KV plus indices, not prebuilt `focused_kv` buffers
3. the current eager `repeat_interleave + gather_selected_tensors` path is exactly the kind of implementation artifact the paper’s fast path avoids

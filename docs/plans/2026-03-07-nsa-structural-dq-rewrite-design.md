# NSA Structural dQ Rewrite Design

**Date**: 2026-03-07
**Branch**: `nsa-triton-longseq-investigation`
**Status**: Design

## Goal

Replace the current query-centric `dQ` selection backward kernel with a structural reverse-mapped kernel that:

1. reuses patch K/V across all queries selecting that patch
2. keeps Triton on the fast `tl.dot` path
3. preserves non-causal per-query NSA semantics
4. remains numerically aligned with current NSA and naive references

## Why This Is The Next Real Target

Current profiler stack agrees that `dQ` is the largest remaining Triton kernel in the practical long-sequence regime.

Reference artifacts:

- `artifacts/nsa_diagnostics/nsa_selection_triton_hotspots_a100_20260307_004533.json`
- `artifacts/nsa_diagnostics/selection_trace_hta_summary_20260307_012823.json`
- `artifacts/nsa_diagnostics/selection_nsys_gpu3_C384_6_3_S256_top8_20260307_013109_stats.txt`
- `artifacts/nsa_diagnostics/selection_proton_gpu3_C384_h6_g3_256_top8_20260307_013503.hatchet.hatchet`

Representative A100 results at `256x256`:

- `C=384, top_n=8`
  - scoring `~12.95 ms`
  - forward `~9.28 ms` after retune
  - `dQ ~15.71 ms`
  - `dK/dV ~5.07 ms`
- `C=384, top_n=16`
  - `dQ ~30.47 ms`
- Proton scoped view:
  - `selection_dq ~31.26 ms`
  - `selection_scoring ~26.13 ms`
  - `selection_forward ~18.40 ms`
  - `selection_dkv ~9.14 ms`

So the next Triton kernel target is not speculative anymore. It is `dQ`.

## Root Cause Diagnosis

### Current kernel shape

Current `dQ` is `_sel_perq_bwd_dq_kernel` in:

- `src/ops/selection_attention_2d_per_query.py`

It is query-centric:

1. one program handles one `(batch, kv_head, query_block)`
2. inside the program, each query loops over `top_n` selected patches
3. for every selected patch, the kernel recomputes and reloads that patch K/V

This means the same patch K/V is reloaded once for every query that selected it.

### Measured patch reuse is huge

Reference artifact:

- `artifacts/nsa_diagnostics/selection_patch_reuse_a100_20260307_014742.json`

At `256x256, p=8` there are `1024` possible patches per KV head.

Measured reuse:

- `C=384, top_n=8`
  - every patch is selected
  - mean queries per used patch: `512`
  - metadata build cost: `~1.75 ms`
- `C=384, top_n=16`
  - every patch is selected
  - mean queries per used patch: `1024`
  - metadata build cost: `~2.26 ms`
- `C=512, top_n=8`
  - every patch is selected
  - mean queries per used patch: `512`

This is the key structural point:

- current query-centric `dQ` reloads each patch K/V roughly hundreds of times
- the reverse mapping metadata is cheap compared to the kernel itself

### Why packed `dQ` was not enough

Packed `dQ` changed locality, but it did not change reuse. The kernel still walked query-by-query and patch-by-patch.

That is why it failed the auto-enable gate:

1. it improved layout only
2. it did not remove the repeated patch reload pattern

## Recommended Rewrite

### Reverse-mapped compact `dQ`

Reuse the existing active-query grouping idea already proven for compact `dK/dV`.

Available helper:

- `_build_active_query_index_per_patch(...)`

This groups query indices by:

- `(batch, kv_head, patch_id)`

Then launch one program per:

- `(batch, kv_head, patch_id, share_head)`

with grid:

- `(B * h_kv * n_patches, G)`

Inside the program:

1. load patch K/V once
2. iterate only the queries that selected this patch, in `BLOCK_Q` tiles
3. compute that patch's contribution to `dQ`
4. atomically accumulate into a float32 `dq_accum`

### Why this avoids the earlier Triton trap

The failed forward real-`G` rewrite lost the fast `tl.dot` path because matrix dimension `M` became `G=1/3/4`.

This compact `dQ` rewrite does not do that.

For compact `dQ`:

- `M = BLOCK_Q` (for example `16` or `32`)
- `N = PP` (typically `64`)
- `K = D` (typically `64`)

So the matmul dimensions stay on the Triton-friendly path even though we process one share-head per program.

That is a major reason this rewrite is promising rather than another version of the failed forward experiment.

## Kernel Outline

### Inputs

Reuse current backward inputs:

- `q`
- `k`
- `v`
- `do`
- `lse`
- `delta`

Add compact metadata:

- `query_idx_sorted`
- `cu_counts`
- `patch_starts`

### Output

Use:

- `dq_accum: [B, h_q, T, d]` float32

Reason:

1. every query/head receives contributions from `top_n` selected patches
2. patch-centric execution means multiple programs contribute to the same `dQ`
3. float32 atomics keep parity safer than atomics into bf16

After the kernel:

- cast `dq_accum` to `q.dtype`

### Program body

Per `(group, share_head)`:

1. decode `(batch, kv_head, patch_id)`
2. load K/V patch tokens once into SRAM
3. read active query range from `cu_counts`
4. loop over active queries in `BLOCK_Q` chunks
5. load:
   - `q`
   - `do`
   - `lse`
   - `delta`
6. recompute:
   - `qk`
   - `p`
   - `dp`
   - `ds`
7. compute partial:
   - `dq_partial = ds @ k`
8. `atomic_add` into `dq_accum`

## Expected Benefits

### Primary

1. stop reloading the same patch K/V for every query
2. move from many small query-single matmuls to query-tiled matmuls
3. preserve compact active-query traversal already validated for `dK/dV`

### Secondary

1. one compact metadata build can serve both:
   - compact `dQ`
   - compact `dK/dV`
2. this can reduce duplicated backward preprocessing

## Risks

### 1. Atomic accumulation into `dQ`

This rewrite trades repeated K/V loads for atomic adds into `dQ`.

Why the risk is acceptable:

1. each query contributes to only `top_n` selected patches
2. collision fan-in on `dQ` is therefore around `8` or `16`, not hundreds
3. this is much better than the old `dK/dV` atomic situation

### 2. Extra float32 memory

`dq_accum` in float32 costs more memory.

For the practical long-sequence regimes we profiled, that is acceptable and predictable.

### 3. `128x128` may not benefit much

Patch reuse exists there too, but the kernel is smaller overall.

So this rewrite should be judged primarily on:

- `256x256`

`128x128` is a sentinel, not the main gate.

## Runtime Policy

Add a third explicit `dQ` mode:

- `unpacked`
- `packed`
- `compact`
- `auto`

Initial policy:

- `auto -> unpacked`

Only move `auto` to `compact` if:

1. parity passes
2. `256x256` shows a real win
3. the gain survives full selection-path benchmarking, not just kernel microbenchmarks

## Validation

### Parity

Required comparisons:

1. compact `dQ` vs unpacked `dQ`
2. compact `dQ` vs naive autograd `dQ`
3. broader MHA and GQA per-query suites stay green

### Benchmarks

A100 first:

1. `C=64, heads=4, G=4`
2. `C=384, heads=6, G=3`
3. `C=512, heads=8, G=4`
4. sizes:
   - `128x128`
   - `256x256`
5. `top_n=8, 16`

Primary success signal:

- meaningful `dQ` improvement at `256x256`

## Decision

This is the first `dQ` direction that is both:

1. supported by profiler data
2. structurally different from the current kernel
3. compatible with Triton fast matmul constraints

So the next implementation should be:

- compact reverse-mapped patch-centric `dQ`

not another query-centric launch retune and not another packing-only variant.

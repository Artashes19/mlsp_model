# NSA Per-Query Triton Kernel Design Notes

Date: 2026-03-04
Branch: `dev-clean`

## Scope

These notes document the per-query (non-causal) Triton backward-kernel changes for NSA 2D selection attention, with focus on MHA path and GQA-compatible behavior.

Primary file:
- `src/ops/selection_attention_2d_per_query.py`

## Problem

For large shapes (e.g. `B=4, C=48, H=W=256, heads=4, patch=8, top_n=32`), profiler runs showed:
- launch failure from CUDA grid-dimension overflow in dK/dV kernel (`invalid argument`)
- then, after launch fix, severe runtime bottleneck in backward:
  - `_sel_perq_bwd_dkv_kernel` dominating step time
  - `_sel_perq_bwd_dq_kernel` also large

## Design Choices Implemented

1. **dK/dV kernel launch-safe mapping**
- Avoid large `gridY` launch dimensions.
- Keep per-query semantics unchanged.

2. **dK/dV query blocking**
- Converted dK/dV kernel to process a block of queries per program:
  - Grid: `(TOP_N, B * h_kv, ceil_div(T, BLOCK_Q))`
  - Loop over `BLOCK_Q` queries with masked loads/stores
  - Preserve atomic accumulation semantics for repeated token selections

3. **dK/dV launch tuning (A6000)**
- Empirical sweep on target large shape favored:
  - lower `num_warps` (2) over 8 for atomics-heavy path
  - larger `BLOCK_Q` (32) for large `T`
- Current heuristics:
  - `BLOCK_Q`: `32` (`T>=4096`), `16` (`T>=512`), else `4`
  - `num_warps`: `2` for `BLOCK_D<=32`, `4` for `<=64`, else `8`

4. **dQ query blocking**
- Converted dQ kernel from query-centric to query-block-centric:
  - Grid: `(B * h_kv, ceil_div(T, BLOCK_Q))`
  - Loop over `BLOCK_Q` queries per program
  - Exact math preserved (no approximation)

5. **No causality path**
- Kept fully non-causal behavior for 2D tasks as requested.

## Update (2026-03-05): Tilda-Style Compact dK/dV (v2)

We aligned dK/dV more closely with Tilda's strategy:

1. **Active-query compaction per KV patch group**
- Precompute compact query lists for each `(batch, kv_head, patch_id)`:
  - `query_idx_sorted`, `cu_counts`
- This removes query-slot explosion from the hot dK/dV path.

2. **Per-share-head compact dK/dV kernel**
- New compact kernel grid:
  - `(B * h_kv * n_patches, G)`
- Each program handles one `(group, share-head)` pair and iterates only active queries for that patch.
- This avoids padded `BLOCK_G` compute in dK/dV and improves occupancy.

3. **No atomics in compact hot loop**
- Kernel writes into share-head-separated buffers:
  - `dk_sh: [G, B, h_kv, T, d]`
  - `dv_sh: [G, B, h_kv, T, d]`
- Final reduction:
  - `dk = dk_sh.sum(dim=0)`
  - `dv = dv_sh.sum(dim=0)`

4. **Legacy fallback preserved**
- `NSA_PERQ_DKV_LEGACY_ATOMIC=1` keeps old atomic kernel for A/B and safety.

5. **No causality**
- Still fully non-causal for 2D tasks.

## Correctness Verification

CUDA tests (per-query MHA + GQA forward/backward parity vs naive references):
- `tests/test_selection_triton.py::TestPerQuerySelectionForward`
- `tests/test_selection_triton.py::TestPerQuerySelectionBackward`
- `tests/test_selection_triton_gqa.py::TestGQAPerQuerySelectionForward`
- `tests/test_selection_triton_gqa.py::TestGQAPerQuerySelectionBackward`

Result after changes: `7 passed`.

## Profiling Evidence (A6000, focused one-step profiler)

Configuration:
- `B=4, C=48, H=W=256, heads=4, patch=8, top_n=32, dtype=bf16`

Measured CUDA totals from focused `torch.profiler` run:

### Before query-block tuning
- `_sel_perq_bwd_dkv_kernel`: `3131 ms`
- `_sel_perq_bwd_dq_kernel`: `836 ms`
- Self CUDA total: `4574 ms`

### After dK/dV + dQ query-block tuning
- `_sel_perq_bwd_dkv_kernel`: `1724 ms`
- `_sel_perq_bwd_dq_kernel`: `299 ms`
- Self CUDA total: `2630 ms`

### Delta
- dK/dV kernel: ~`-45%`
- dQ kernel: ~`-64%`
- Total profiled CUDA time: ~`-42.5%`

## Profiling Evidence (A6000, long-seq operator microbench)

Configuration (matched):
- `B=1, H=W=256, T=65536, Hq=4, Hkv=1, G=4, D=64, P=8, top_n=16, bf16`
- Warmup `5`, iters `5`

Artifacts:
- pre-v2 compact vs legacy vs tilda:
  - `artifacts/nsa_diagnostics/ours_compact_vs_legacy_vs_tilda_ap_a6000_20260304_231821.json`
- v2 compact vs legacy vs tilda:
  - `artifacts/nsa_diagnostics/ours_compact_v2_vs_legacy_vs_tilda_ap_a6000_20260304_232154.json`
- profiler pre-v2 compact vs legacy:
  - `artifacts/nsa_diagnostics/torch_profiler_compact_vs_legacy_dkv_ap_a6000_20260304_231844.txt`
- profiler v2 compact:
  - `artifacts/nsa_diagnostics/torch_profiler_compact_v2_dkv_ap_a6000_20260304_232210.txt`

Key runtime deltas:
- Legacy atomic `bwd_qkv`: `~701 ms`
- Compact v1 `bwd_qkv`: `~156 ms`
- Compact v2 `bwd_qkv`: `~115 ms`
- Tilda `bwd_qkv`: `~90 ms`

Compact v2 improvement over compact v1:
- `bwd_qkv`: ~`-25.8%` (`155.6 -> 115.4 ms`)

Compact v2 improvement over legacy:
- `bwd_qkv`: ~`6.1x` faster (`701.3 -> 115.4 ms`)

Profiler kernel-level deltas (compact v1 -> compact v2):
- `_sel_perq_bwd_dkv_compact_kernel`: `~73.9 ms -> ~27.1 ms`
- `_sel_perq_bwd_dq_kernel`: remains dominant (`~48 ms`)
- `_sel_perq_fwd_kernel`: still large (`~36-37 ms`)

## MHA Path Check (A6000, G=1)

Artifact:
- `artifacts/nsa_diagnostics/ours_compact_v2_mha_vs_legacy_vs_tilda_ap_a6000_20260304_232519.json`
- `artifacts/nsa_diagnostics/torch_profiler_compact_v2_mha_ap_a6000_20260304_232535.txt`

Config:
- `B=1, H=W=256, T=65536, Hq=4, Hkv=4, G=1, D=64, P=8, top_n=16, bf16`

Result summary:
- Ours compact v2 vs legacy:
  - `bwd_qkv`: `369.8 ms` vs `2826.7 ms` (~`7.6x` faster)
- Ours compact v2 vs Tilda:
  - `forward`: `143.8 ms` vs `148.7 ms` (ours slightly faster)
  - `bwd_qkv`: `369.8 ms` vs `344.0 ms` (ours ~`7.5%` slower)

Profiler (ours compact v2, MHA) shows main hotspots are now:
- `_sel_perq_bwd_dq_kernel`: `~185.7 ms`
- `_sel_perq_fwd_kernel`: `~144.3 ms`
- `_sel_perq_bwd_dkv_compact_kernel`: `~27.0 ms`

So on MHA path, dK/dV is no longer the primary bottleneck.

## Open Follow-Ups

1. Optimize dQ (`_sel_perq_bwd_dq_kernel`), now the largest backward kernel on compact v2.
2. Optimize forward (`_sel_perq_fwd_kernel`) with the same compaction/layout principles where feasible.
3. Reduce preprocessing overhead in `_build_active_query_index_per_patch` (`argsort`/`bincount`) for long sequence regimes.
4. Re-run wide diagnostics sweep for apples-to-apples CSV comparison with existing artifacts.

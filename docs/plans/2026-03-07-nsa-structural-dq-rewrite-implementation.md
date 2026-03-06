# NSA Structural dQ Rewrite Implementation Plan

**Date**: 2026-03-07
**Branch**: `nsa-triton-longseq-investigation`
**Status**: Planned

## Goal

Implement and validate a compact reverse-mapped `dQ` kernel for per-query 2D selection attention.

## Scope

In scope:

1. compact `dQ` Triton kernel
2. runtime mode `"compact"`
3. parity tests
4. A100 benchmarks
5. memory update with the decision gate

Out of scope:

1. changing forward
2. changing scoring
3. backend migration
4. changing FFN or shell

## Task 1: Lock parity coverage before kernel work

Files:

- modify `tests/test_selection_triton.py`
- modify `tests/test_selection_triton_gqa.py`

Steps:

1. Add focused tests for `selection_dq_mode="compact"`:
   - MHA
   - GQA
2. Compare:
   - unpacked vs compact outputs and grads
   - compact vs naive reference where the suite already has it
3. Verify the new tests fail before implementation or mode wiring

Success:

- compact-mode tests exist and fail for the right reason before code changes

## Task 2: Implement compact dQ kernel and wrapper

Files:

- modify `src/ops/selection_attention_2d_per_query.py`

Steps:

1. Add `_sel_perq_bwd_dq_compact_kernel`
2. Add `selection_per_query_bwd_dq_compact(...)`
3. Use existing `_build_active_query_index_per_patch(...)`
4. Write into float32 `dq_accum`
5. Cast to `q.dtype` on return
6. Keep unpacked and packed paths untouched

Success:

- compact kernel compiles
- focused compact parity tests pass

## Task 3: Wire runtime mode without changing default behavior

Files:

- modify `src/ops/selection_attention_2d_per_query.py`
- modify `src/networks/txunet.py`

Steps:

1. Allow `selection_dq_mode="compact"`
2. Keep:
   - `auto -> unpacked`
3. Route only explicit `"compact"` to the new path

Success:

- explicit compact mode works
- `auto` behavior is unchanged

## Task 4: Reuse compact query metadata where practical

Files:

- modify `src/ops/selection_attention_2d_per_query.py`

Steps:

1. Avoid rebuilding active-query metadata twice if both compact `dQ` and compact `dK/dV` need it
2. Keep the refactor minimal:
   - internal shared helper or backward-local shared build
3. Do not change forward or packed metadata logic

Success:

- metadata reuse is correct
- no parity regression

## Task 5: Benchmark the compact dQ path on A100

Files:

- modify or reuse `artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py`
- optionally add a focused benchmark helper if needed

Steps:

1. Benchmark:
   - unpacked `dQ`
   - compact `dQ`
2. Cases:
   - `C=64, heads=4, G=4`
   - `C=384, heads=6, G=3`
   - `C=512, heads=8, G=4`
   - `128x128`, `256x256`
   - `top_n=8, 16`
3. Save artifacts

Success:

- we have a clean compact-vs-unpacked comparison

## Task 6: Decision gate

Files:

- modify `docs/nsa_memory.md`

Steps:

1. Record:
   - parity status
   - benchmark results
   - whether compact `dQ` is the new preferred path
2. Only if the gain is real:
   - update `auto` policy in a follow-up task

Success:

- memory reflects the actual result
- no silent auto-enable

## Required Verification

Before claiming success:

1. `pytest tests/test_selection_triton.py tests/test_selection_triton_gqa.py -v`
2. focused A100 benchmark run for compact vs unpacked `dQ`
3. confirm saved artifact paths

## Expected Outcome

Best-case:

1. `dQ` drops materially at `256x256`
2. total selection path improves enough to matter beyond the earlier forward retune

Acceptable outcome:

1. compact `dQ` helps only at `256x256`
2. `auto` stays conservative until we decide the exact regime gate

Failure outcome:

1. parity is hard to maintain
2. atomics erase the reuse gain
3. then we stop and do not force-enable the path

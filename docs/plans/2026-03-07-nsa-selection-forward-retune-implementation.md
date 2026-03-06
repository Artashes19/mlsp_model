# NSA Selection Forward Retune Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retune the current Triton selection forward launch heuristic so the existing forward kernel stops over-allocating warps in the wide-channel `128x128` and `256x256` regimes.

**Architecture:** Keep the current unpacked selection forward kernel and algorithm unchanged. Only adjust the forward launch-meta selection path so the runtime chooses lower `num_warps` for the measured wide-channel cases, while leaving `dQ`, `dK/dV`, and selection semantics untouched.

**Tech Stack:** Python, PyTorch, Triton, pytest, DGX A100 benchmarking over SSH

---

### Task 1: Lock the desired forward-warp behavior with tests

**Files:**
- Modify: `src/ops/selection_attention_2d_per_query.py`
- Modify: `tests/test_selection_triton.py`

**Step 1: Write the failing test**

Add a focused heuristic test in `tests/test_selection_triton.py` that checks the forward selector for the measured wide-channel regimes:

```python
def test_select_num_warps_per_query_prefers_lower_warps_for_wide_forward_cases():
    # Current wide-channel forward cases from A100 sweep.
    assert _select_num_warps_per_query(16, 64, 64, top_n=8) == 4
    assert _select_num_warps_per_query(16, 64, 64, top_n=16) == 2
```

Also keep one legacy-style coverage point so the selector does not collapse everything to low warps:

```python
def test_select_num_warps_per_query_keeps_small_work_case_low():
    assert _select_num_warps_per_query(16, 16, 16, top_n=4) == 2
```

**Step 2: Run test to verify it fails**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_selection_triton.py -k "select_num_warps_per_query" -v
```

Expected: FAIL because `_select_num_warps_per_query(...)` does not yet accept `top_n` and still returns `8` for the wide cases.

**Step 3: Write minimal implementation**

Update `_select_num_warps_per_query(...)` in `src/ops/selection_attention_2d_per_query.py` to accept `top_n` and bias toward lower warps in the measured wide forward regimes. Keep the logic simple and explicit; do not introduce autotune or staging yet.

**Step 4: Run test to verify it passes**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_selection_triton.py -k "select_num_warps_per_query" -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/ops/selection_attention_2d_per_query.py tests/test_selection_triton.py
git commit -m "perf(nsa): retune selection forward warp heuristic"
```

### Task 2: Wire the new selector through the forward launch path

**Files:**
- Modify: `src/ops/selection_attention_2d_per_query.py`
- Test: `tests/test_selection_triton.py`
- Test: `tests/test_selection_triton_gqa.py`

**Step 1: Write the failing test**

Add or extend a focused test that exercises the public forward wrapper and asserts numerical parity still holds after the selector change on both MHA and GQA paths.

Use the existing per-query forward parity cases rather than inventing a new reference path.

**Step 2: Run test to verify it fails**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_selection_triton.py tests/test_selection_triton_gqa.py -k "per_query and forward" -v
```

Expected: if the selector signature is not yet threaded through all call sites, this should fail with a Python argument mismatch or an internal launch-path error.

**Step 3: Write minimal implementation**

Thread `top_n` into both forward launch sites that currently call `_select_num_warps_per_query(...)`.

Do not change:

- kernel math
- packed forward mode
- `dQ`
- `dK/dV`

**Step 4: Run test to verify it passes**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_selection_triton.py tests/test_selection_triton_gqa.py -k "per_query and forward" -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/ops/selection_attention_2d_per_query.py tests/test_selection_triton.py tests/test_selection_triton_gqa.py
git commit -m "test(nsa): preserve forward parity after warp retune"
```

### Task 3: Verify broader parity did not move

**Files:**
- Test: `tests/test_selection_triton.py`
- Test: `tests/test_selection_triton_gqa.py`

**Step 1: Run the broader parity suite**

Run:

```bash
CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest -k per_query tests/test_selection_triton.py tests/test_selection_triton_gqa.py -v
```

Expected: PASS with the existing per-query forward/backward parity coverage.

**Step 2: If anything fails, stop and debug**

Do not stack fixes. Use `superpowers:systematic-debugging` and identify the exact failing path before changing code again.

**Step 3: Commit**

If no code change was needed, skip the commit for this task.

### Task 4: Measure before/after on A100 with the existing investigation harness

**Files:**
- Use: `artifacts/nsa_diagnostics/bench_selection_triton_kernel_sweep.py`
- Use: `artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py`

**Step 1: Run the narrow forward sweep after the retune**

Run on DGX A100:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation && CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/nsa_diagnostics/bench_selection_triton_kernel_sweep.py --configs 384:6:3 512:8:4 --sizes 128 256 --top-ns 8 16 --warmup 3 --iters 5'
```

Expected:

- the new current forward meta should move from `8` warps to `4` or `2` on the measured wide cases
- best-vs-current forward gap should shrink materially

**Step 2: Run hotspot profiling again**

Run:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation && CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py --configs 384:6:3 512:8:4 --sizes 128 256 --top-ns 8 16'
```

Expected:

- forward time drops on the priority `256x256` cases
- `dQ` remains roughly unchanged

**Step 3: Record the comparison**

Summarize:

- old vs new forward ms
- any total selection-path reduction
- whether `dQ` is now clearly the next remaining Triton bottleneck

**Step 4: Commit**

If the harness or docs changed:

```bash
git add artifacts/nsa_diagnostics docs/nsa_memory.md
git commit -m "docs(nsa): record forward retune benchmark results"
```

### Task 5: Update persistent memory and decide the next target

**Files:**
- Modify: `docs/nsa_memory.md`
- Create: `docs/plans/2026-03-07-nsa-selection-dq-next-design.md` only if forward retune lands cleanly and `dQ` remains dominant

**Step 1: Update `docs/nsa_memory.md`**

Record:

- the new selector heuristic rule
- exact A100 before/after numbers
- whether forward meta is now “good enough”
- whether the next Triton-only target is a structural `dQ` change

**Step 2: Decide the next target**

Use this rule:

- if forward retune produces a meaningful long-sequence win and the current forward meta becomes close to best, the next target is `dQ`
- if forward still shows a large best-vs-current gap, do not move on yet

**Step 3: Commit**

```bash
git add docs/nsa_memory.md docs/plans
git commit -m "docs(nsa): plan next Triton target after forward retune"
```

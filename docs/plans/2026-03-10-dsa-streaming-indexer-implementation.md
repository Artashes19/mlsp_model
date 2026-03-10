# DSA Streaming Indexer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the dense DSA indexer materialization with an exact streaming top-k indexer that preserves current selector semantics while removing the first `T x T` OOM source.

**Architecture:** Keep current DeepSeek-aligned indexer preprocessing unchanged and add a new exact streaming scorer/top-k path. Roll it out behind an explicit mode switch in `DSA2DMLAAttention`, validate it against the dense reference, and only then use it for long-sequence benchmarking.

**Tech Stack:** PyTorch, bf16/fp32 test references, pytest, existing DSA diagnostics harnesses on A100.

---

### Task 1: Add failing streaming-indexer unit tests

**Files:**
- Modify: `tests/test_dsa_2d_indexer.py`
- Test: `tests/test_dsa_2d_indexer.py`

**Step 1: Write the failing test**

Add tests for:
- tiny dense-vs-streaming parity
- tie handling
- block-size invariance

Example skeleton:

```python
def test_streaming_indexer_matches_dense_small_case():
    q = ...
    k = ...
    w = ...
    dense_scores = weighted_relu_index_score(q, k, w)
    dense_idx = stable_topk(dense_scores, k=2)
    stream_scores, stream_idx = streaming_weighted_relu_topk(q, k, w, topk=2, block_s=3)
    torch.testing.assert_close(stream_idx, dense_idx)
```

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "streaming_indexer" -v
```

Expected:
- FAIL because `streaming_weighted_relu_topk` does not exist yet

**Step 3: Do not implement yet**

Stop after the red state is verified.

**Step 4: Commit the failing tests**

```bash
git add tests/test_dsa_2d_indexer.py
git commit -m "test(dsa): add streaming indexer parity regressions"
```

### Task 2: Implement exact streaming top-k helper

**Files:**
- Modify: `src/ops/dsa_indexer.py`
- Test: `tests/test_dsa_2d_indexer.py`

**Step 1: Write minimal implementation**

Add:
- `streaming_weighted_relu_topk(...)`
- helper for deterministic merge with explicit tie behavior

Implementation requirements:
- process keys in blocks
- avoid full `[B, j, T, T]`
- avoid full `[B, T, T]`
- use exact merge of running top-k and current block candidates

**Step 2: Run focused tests**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "streaming_indexer" -v
```

Expected:
- PASS

**Step 3: Refactor for clarity only if tests stay green**

Keep YAGNI. No Triton work here.

**Step 4: Commit**

```bash
git add src/ops/dsa_indexer.py tests/test_dsa_2d_indexer.py
git commit -m "feat(dsa): add exact streaming indexer topk"
```

### Task 3: Add reference helper coverage

**Files:**
- Modify: `tests/helpers/dsa_reference.py`
- Test: `tests/test_dsa_2d_indexer.py`

**Step 1: Write the failing test**

Add tests that compare:
- dense reference path
- streaming helper output
- multiple `block_s` values

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "block_size_invariance or dense_reference" -v
```

Expected:
- FAIL until helper/reference glue is complete

**Step 3: Implement minimal reference glue**

Add small helpers in `tests/helpers/dsa_reference.py` only as needed.

**Step 4: Run tests to verify pass**

Run the same command again and confirm PASS.

**Step 5: Commit**

```bash
git add tests/helpers/dsa_reference.py tests/test_dsa_2d_indexer.py
git commit -m "test(dsa): lock streaming indexer reference invariants"
```

### Task 4: Integrate mode switch into DSA module

**Files:**
- Modify: `src/networks/dsa_2d.py`
- Test: `tests/test_dsa_2d_indexer.py`
- Test: `tests/test_dsa_2d_regression.py`

**Step 1: Write the failing tests**

Add tests for:
- `indexer_mode="dense"` uses dense path
- `indexer_mode="streaming"` uses streaming path
- dense and streaming return identical indices on a locked small case

**Step 2: Run tests to verify they fail**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py tests/test_dsa_2d_regression.py -k "indexer_mode" -v
```

Expected:
- FAIL because the mode switch does not exist yet

**Step 3: Implement minimal integration**

In `DSA2DMLAAttention`:
- add explicit `indexer_mode`
- keep default conservative
- dispatch to dense or streaming helper in `build_indexer_logits(...)`

**Step 4: Run targeted tests**

Run the same command and confirm PASS.

**Step 5: Commit**

```bash
git add src/networks/dsa_2d.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_regression.py
git commit -m "feat(dsa): add streaming indexer mode switch"
```

### Task 5: Add long-sequence OOM regression and benchmark wiring

**Files:**
- Modify: `tests/test_dsa_2d_regression.py`
- Modify: `artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py`

**Step 1: Write the failing regression**

Add a regression that checks benchmark results report which indexer mode ran, and that the harness can request streaming mode explicitly.

If a practical OOM regression can be expressed cheaply, add a guarded regression that dense path is expected to fail while streaming mode is expected to complete on a controlled long-sequence case.

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_regression.py -k "indexer_mode or streaming_benchmark" -v
```

Expected:
- FAIL until harness wiring is added

**Step 3: Implement minimal harness changes**

Add:
- indexer mode field in artifacts
- explicit DSA streaming-mode benchmark option

**Step 4: Run tests to verify pass**

Run the same command and confirm PASS.

**Step 5: Commit**

```bash
git add tests/test_dsa_2d_regression.py artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py
git commit -m "perf(dsa): wire streaming indexer into diagnostics"
```

### Task 6: Run full DSA suite

**Files:**
- Verify only

**Step 1: Run the full DSA suite**

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v
```

Expected:
- all tests pass

**Step 2: If any test fails, stop and fix before proceeding**

Do not benchmark until the full suite is green.

**Step 3: Commit if needed**

```bash
git status --short
```

Commit only if code changed during stabilization.

### Task 7: Rerun corrected A100 benchmark with streaming mode

**Files:**
- Modify: `docs/dsa_memory.md`
- Verify: `artifacts/dsa_diagnostics/dsa_benchmark_cuda_bfloat16.json`

**Step 1: Run the benchmark on DGX A100**

Use the corrected harness and explicit streaming mode.

Run:
```bash
ssh artashes@dgx.yc2.io "cd /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation && CUDA_VISIBLE_DEVICES=7 /auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py --device cuda --dtype bfloat16 --warmup 1 --iters 1"
```

If the harness needs a new CLI flag for indexer mode, add it before this step.

**Step 2: Inspect artifact**

Confirm:
- streaming mode is recorded
- dense and/or streaming status is visible
- `256x256` no longer fails in the indexer phase

**Step 3: Update memory**

Record:
- exact command
- artifact path
- whether streaming removed the first OOM source
- whether sparse MLA gather is now the next bottleneck

**Step 4: Commit docs/harness if needed**

```bash
git add docs/dsa_memory.md artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py
git commit -m "docs(dsa): record streaming indexer benchmark outcomes"
```

### Task 8: Decision gate

**Files:**
- Modify: `docs/dsa_memory.md`

**Step 1: Decide rollout status**

Possible outcomes:
- keep `dense` default, `streaming` explicit only
- enable narrow `auto` for long-sequence cases
- keep `streaming` experimental if parity or runtime is not good enough

**Step 2: Record the decision**

Update memory with:
- current default
- exact reasons
- next bottleneck after this step

**Step 3: Final verification**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v
```

Expected:
- PASS

**Step 4: Commit**

```bash
git add docs/dsa_memory.md
git commit -m "docs(dsa): record streaming indexer decision gate"
```

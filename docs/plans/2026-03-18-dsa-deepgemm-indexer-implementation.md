# DSA DeepGEMM Indexer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an H100-only, forward-only `DeepGEMM` selector-logit backend for DSA and keep current selector semantics unchanged.

**Architecture:** Add a new `indexer_backend` dispatch path that swaps only the dense selector logits computation on supported H100 MQA forward cases. Keep `topk` logic, training semantics, and fallback behavior unchanged until parity is proven.

**Tech Stack:** PyTorch, H100 CUDA, `DeepGEMM`, existing DSA reference selector path, pytest.

---

### Task 1: Add failing config and dispatch tests

**Files:**
- Modify: `tests/test_dsa_2d_regression.py`
- Modify: `tests/test_dsa_2d_indexer.py`

**Step 1: Write the failing tests**

Add tests for:
- default `indexer_backend == "auto"`
- unknown `indexer_backend` rejected
- `auto` runtime selector uses `deepgemm` when supported and grad is disabled
- `auto` runtime selector falls back to reference when grad is enabled

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_regression.py tests/test_dsa_2d_indexer.py -k "indexer_backend or auto" -v
```
Expected: FAIL because `indexer_backend` does not exist yet.

**Step 3: Commit the failing-test checkpoint only after the implementation passes later**

No commit yet.

### Task 2: Add the `DeepGEMM` adapter module

**Files:**
- Create: `src/ops/dsa_deepgemm.py`
- Test: `tests/test_dsa_2d_indexer.py`

**Step 1: Write the failing tests**

Add tests for:
- safe lazy import helper
- support check rejects CPU
- support check rejects non-MQA
- support check rejects non-`sm90`

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "deepgemm_import or deepgemm_support" -v
```
Expected: FAIL because the module does not exist.

**Step 3: Write minimal implementation**

In `src/ops/dsa_deepgemm.py`, add:
- `deepgemm_import_or_none()`
- `deepgemm_is_supported(...)`
- `deepgemm_weighted_relu_logits(...)` stub that initially raises when no backend is available

**Step 4: Run test to verify it passes**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "deepgemm_import or deepgemm_support" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/dsa_deepgemm.py tests/test_dsa_2d_indexer.py
git commit -m "feat(dsa): add deepgemm selector adapter scaffold"
```

### Task 3: Add `indexer_backend` to config and runtime dispatch

**Files:**
- Modify: `src/networks/dsa_2d.py`
- Test: `tests/test_dsa_2d_regression.py`
- Test: `tests/test_dsa_2d_indexer.py`

**Step 1: Implement config validation and default**

Add:
- `indexer_backend: str = "auto"`
- validation for `auto|reference|deepgemm`

**Step 2: Implement runtime dispatch in `build_indexer_selection(...)`**

Behavior:
- if `indexer_backend in {"auto", "deepgemm"}` and grad is disabled and support check passes:
  - compute logits via the `DeepGEMM` wrapper
  - apply existing top-k logic
- else:
  - use current reference path

**Step 3: Run focused tests**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_regression.py tests/test_dsa_2d_indexer.py -k "indexer_backend or auto" -v
```
Expected: PASS.

**Step 4: Commit**

```bash
git add src/networks/dsa_2d.py tests/test_dsa_2d_regression.py tests/test_dsa_2d_indexer.py
git commit -m "perf(dsa): add auto deepgemm selector dispatch"
```

### Task 4: Add logits parity tests against the reference path

**Files:**
- Modify: `tests/test_dsa_2d_indexer.py`

**Step 1: Write the failing tests**

Add tests for:
- small-case `DeepGEMM` logits wrapper matches `weighted_relu_index_score`
- selected top-k indices from kernel logits match the reference selection

Use monkeypatched fake kernel responses first if needed to isolate packing/dispatch.

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "deepgemm_logits or deepgemm_selection" -v
```
Expected: FAIL until the wrapper and dispatch are wired fully.

**Step 3: Implement minimal parity-safe wrapper behavior**

Update `src/ops/dsa_deepgemm.py` and any needed call sites so the wrapper consumes prepared `q, k, w` and returns logits in the same dense shape as the reference path.

**Step 4: Run test to verify it passes**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "deepgemm_logits or deepgemm_selection" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/dsa_deepgemm.py src/networks/dsa_2d.py tests/test_dsa_2d_indexer.py
git commit -m "test(dsa): verify deepgemm selector parity"
```

### Task 5: Run the full local DSA suite

**Files:**
- Verify only

**Step 1: Run the full suite**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v
```
Expected: PASS.

**Step 2: Commit if any local test-only adjustments were needed**

```bash
git add <only files changed in this task>
git commit -m "test(dsa): keep suite green after deepgemm selector dispatch"
```

### Task 6: Validate and benchmark on H100

**Files:**
- Modify if needed: `docs/dsa_memory.md`
- Create or reuse cluster temp harnesses as needed

**Step 1: Run H100 parity on supported MQA selector cases**

Compare:
- `indexer_backend="reference"`
- `indexer_backend="deepgemm"`

Measure:
- logits parity
- top-k parity

**Step 2: Run H100 selector-stage benchmark**

Measure:
- logits only
- selector only (`logits + topk`)
- full DSA forward with:
  - `indexer_backend=auto`
  - `sparse_backend=auto`

**Step 3: Record the stable results**

Update `docs/dsa_memory.md` with:
- parity result
- supported slice
- benchmark speedup
- next bottleneck after selector integration

**Step 4: Commit**

```bash
git add docs/dsa_memory.md
git commit -m "docs(dsa): record deepgemm selector results"
```

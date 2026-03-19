# DSA Frozen-Selector Training-Win Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a measurable `256x256` H100 training-step speedup for DSA over dense FlashAttention by freezing the selector and kernelizing only the sparse MLA forward/backward path.

**Architecture:** Keep selector forward-only and frozen after warm-up. Reuse the existing absorbed MLA runtime, `DeepGEMM` selector forward, and `FlashMLA` sparse forward. Add a packed sparse reference operator plus a custom autograd sparse operator that initially matches the packed reference and then becomes the optimized training path.

**Tech Stack:** PyTorch, custom `autograd.Function`, H100 CUDA, `FlashMLA`, `DeepGEMM`, existing DSA absorbed runtime, pytest, Slurm.

---

### Task 1: Add failing tests for frozen-selector training mode

**Files:**
- Modify: `tests/test_dsa_2d_training.py`
- Modify: `tests/test_dsa_2d_regression.py`
- Modify: `src/networks/dsa_2d.py`

**Step 1: Write the failing test**

Add tests that assert:
- selector parameters can be frozen through an explicit training mode or helper
- frozen selector receives no gradients during a sparse training step
- non-selector attention parameters still receive gradients

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -k "frozen_selector" -v
```
Expected: FAIL because frozen-selector training mode does not exist yet.

**Step 3: Write minimal implementation**

In `src/networks/dsa_2d.py`, add a narrow helper or mode that:
- marks selector submodules as frozen
- avoids building unnecessary selector grad state in the sparse training path

**Step 4: Run test to verify it passes**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -k "frozen_selector" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/networks/dsa_2d.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py
git commit -m "feat(dsa): add frozen-selector sparse training mode"
```

### Task 2: Add a packed sparse reference operator and autograd baseline

**Files:**
- Modify: `src/ops/dsa_sparse_mla.py`
- Modify: `tests/helpers/dsa_reference.py`
- Modify: `tests/test_dsa_2d_sparse_attention.py`

**Step 1: Write the failing test**

Add tests for a packed sparse reference that consumes:
- `q_runtime`
- `kv_runtime`
- `indices`

Test:
- forward output parity with the current packed sparse math
- autograd availability for `dq_runtime`
- autograd availability for `dkv_runtime`

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py -k "packed_runtime" -v
```
Expected: FAIL because the packed sparse reference helper does not exist yet.

**Step 3: Write minimal implementation**

In `src/ops/dsa_sparse_mla.py` and `tests/helpers/dsa_reference.py`, add:
- a packed sparse reference forward helper
- a thin wrapper that exposes the same contract for autograd-based backward checking

**Step 4: Run test to verify it passes**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py -k "packed_runtime" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/dsa_sparse_mla.py tests/helpers/dsa_reference.py tests/test_dsa_2d_sparse_attention.py
git commit -m "feat(dsa): add packed sparse reference baseline"
```

### Task 3: Add a custom autograd sparse operator with reference backward

**Files:**
- Create: `src/ops/dsa_sparse_mla_autograd.py`
- Modify: `src/networks/dsa_2d.py`
- Modify: `tests/test_dsa_2d_sparse_attention.py`

**Step 1: Write the failing test**

Add tests that assert:
- the custom sparse autograd op can be called from the DSA sparse path
- forward matches the packed sparse reference
- backward gradients match the packed sparse autograd baseline on small supported cases

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py -k "autograd_op" -v
```
Expected: FAIL because the custom autograd op does not exist yet.

**Step 3: Write minimal implementation**

In `src/ops/dsa_sparse_mla_autograd.py`, add a custom `torch.autograd.Function` that:
- uses `FlashMLA` in forward when supported
- saves packed runtime tensors and softmax stats
- uses the packed PyTorch sparse reference path inside backward as the initial correctness-first implementation

Wire it into `src/networks/dsa_2d.py` only for the frozen-selector sparse training path.

**Step 4: Run test to verify it passes**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py -k "autograd_op" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/dsa_sparse_mla_autograd.py src/networks/dsa_2d.py tests/test_dsa_2d_sparse_attention.py
git commit -m "feat(dsa): add sparse autograd operator scaffold"
```

### Task 4: Replace reference backward with explicit sparse runtime backward

**Files:**
- Modify: `src/ops/dsa_sparse_mla_autograd.py`
- Modify: `tests/test_dsa_2d_sparse_attention.py`
- Modify: `tests/test_dsa_2d_training.py`

**Step 1: Write the failing test**

Add tighter tests for:
- `dq_runtime` parity against the packed autograd reference
- `dkv_runtime` parity against the packed autograd reference
- frozen-selector train-step gradient flow on supported native shapes

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py -k "dq_runtime or dkv_runtime or frozen_selector_train_step" -v
```
Expected: FAIL until explicit backward replaces the reference fallback.

**Step 3: Write minimal implementation**

Update `src/ops/dsa_sparse_mla_autograd.py` so `backward(...)` computes explicit sparse runtime gradients for:
- `q_runtime`
- `kv_runtime`

Keep the first implementation narrow:
- MQA only
- native absorbed runtime shapes only

**Step 4: Run test to verify it passes**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py -k "dq_runtime or dkv_runtime or frozen_selector_train_step" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/dsa_sparse_mla_autograd.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py
git commit -m "perf(dsa): add explicit sparse MLA backward"
```

### Task 5: Keep the full local DSA suite green

**Files:**
- Verify only

**Step 1: Run the full suite**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v
```
Expected: PASS.

**Step 2: Commit if any test-only adjustments were needed**

```bash
git add <only files changed in this task>
git commit -m "test(dsa): keep suite green for sparse training path"
```

### Task 6: Add H100 sparse-operator training benchmark

**Files:**
- Create: `artifacts/dsa_diagnostics/bench_dsa_sparse_training_step.py`
- Modify: `docs/dsa_memory.md`

**Step 1: Write the failing test**

Add a regression test that the benchmark harness:
- imports the DSA module reliably
- exposes a stable result schema for:
  - reference sparse operator
  - fast sparse operator

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_regression.py -k "sparse_training_bench_schema" -v
```
Expected: FAIL because the harness does not exist yet.

**Step 3: Write minimal implementation**

Create a harness that benchmarks forward+backward for the sparse attention operator only on supported native H100 MQA shapes.

**Step 4: Run test to verify it passes**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_regression.py -k "sparse_training_bench_schema" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add artifacts/dsa_diagnostics/bench_dsa_sparse_training_step.py tests/test_dsa_2d_regression.py
git commit -m "perf(dsa): add sparse training operator benchmark harness"
```

### Task 7: Benchmark the full `256x256` training step against dense FlashAttention

**Files:**
- Create or modify: `artifacts/dsa_diagnostics/bench_txunet_dsa_vs_flash_trainstep.py`
- Modify: `docs/dsa_memory.md`

**Step 1: Build the benchmark harness**

The harness must compare:
- dense FlashAttention baseline
- DSA frozen-selector sparse training path

Measure:
- step time
- peak HBM

**Step 2: Run on H100 under Slurm**

Use the `research` partition and record:
- exact config
- exact model path
- exact speedup or lack of speedup

**Step 3: Record the result**

Update `docs/dsa_memory.md` with:
- whether the `256x256` training-step win was achieved
- exact numbers
- next bottleneck if it was not achieved

**Step 4: Commit**

```bash
git add artifacts/dsa_diagnostics/bench_txunet_dsa_vs_flash_trainstep.py docs/dsa_memory.md
git commit -m "docs(dsa): record 256x256 training-step benchmark"
```

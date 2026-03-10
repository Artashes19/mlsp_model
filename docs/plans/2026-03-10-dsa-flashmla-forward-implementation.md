# DSA FlashMLA Forward Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Integrate a first H100 forward-only `FlashMLA` backend into `DSA2DMLAAttention` while keeping the current reference sparse MLA path as a fallback.

**Architecture:** Add a dedicated `FlashMLA` adapter module and a narrow `sparse_backend` dispatch path in `DSA2DMLAAttention`. Keep selector/indexer, MLA projections, RoPE, and training paths unchanged. Restrict the kernel path to `SM90+`, forward-only, `MQA` (`n_kv_heads == 1`), and supported dtypes/layouts.

**Tech Stack:** PyTorch, pytest, optional `FlashMLA`, H100 via Slurm, existing DSA reference tests and diagnostics.

---

### Task 1: Lock backend-dispatch red tests

**Files:**
- Modify: `tests/test_dsa_2d_sparse_attention.py`
- Read: `src/networks/dsa_2d.py`

**Step 1: Write the failing tests**

Add tests that describe the new backend contract:

```python
def test_flashmla_backend_falls_back_to_reference_on_cpu(monkeypatch):
    mod = make_small_cfg_module(sparse_backend="flashmla", n_kv_heads=1)
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1]]], dtype=torch.int64).expand(1, 4, 2).clone()

    called = {"reference": False}

    def _reference(*args, **kwargs):
        called["reference"] = True
        return torch.randn(1, mod.n_heads, 4, mod.v_head_dim)

    monkeypatch.setattr("src.networks.dsa_2d.streaming_sparse_mla_reference", _reference)
    mod.forward_sparse_from_indices(x, idx)
    assert called["reference"]


def test_flashmla_backend_rejects_non_mqa_kernel_path(monkeypatch):
    mod = make_small_cfg_module(sparse_backend="flashmla", n_kv_heads=2)
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1]]], dtype=torch.int64).expand(1, 4, 2).clone()

    called = {"flash": False, "reference": False}

    def _flash(*args, **kwargs):
        called["flash"] = True
        return torch.randn(1, mod.n_heads, 4, mod.v_head_dim)

    def _reference(*args, **kwargs):
        called["reference"] = True
        return torch.randn(1, mod.n_heads, 4, mod.v_head_dim)

    monkeypatch.setattr("src.networks.dsa_2d.flashmla_sparse_mla_forward", _flash)
    monkeypatch.setattr("src.networks.dsa_2d.streaming_sparse_mla_reference", _reference)
    mod.forward_sparse_from_indices(x, idx)
    assert not called["flash"]
    assert called["reference"]
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -k "flashmla_backend_falls_back_to_reference_on_cpu or flashmla_backend_rejects_non_mqa_kernel_path" -v`
Expected: FAIL because `sparse_backend` dispatch does not exist yet.

**Step 3: Do not implement yet**

This task is only the red contract for the new backend.

### Task 2: Add the adapter scaffold

**Files:**
- Create: `src/ops/dsa_flashmla.py`
- Modify: `tests/test_dsa_2d_sparse_attention.py`

**Step 1: Write the failing adapter tests**

Add small adapter-level tests:

```python
def test_flashmla_support_check_rejects_cpu():
    assert not flashmla_is_supported(device=torch.device("cpu"), n_kv_heads=1)


def test_flashmla_support_check_rejects_non_mqa():
    assert not flashmla_is_supported(device=fake_sm90_cuda_device(), n_kv_heads=2)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -k "flashmla_support_check" -v`
Expected: FAIL because the adapter module does not exist.

**Step 3: Write minimal implementation**

In `src/ops/dsa_flashmla.py`, add:

1. `flashmla_is_supported(...)`
2. `flashmla_import_or_none()`
3. `flashmla_sparse_mla_forward(...)` stub that raises `NotImplementedError`

Keep it import-safe when `FlashMLA` is not installed.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -k "flashmla_support_check" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/dsa_flashmla.py tests/test_dsa_2d_sparse_attention.py
git commit -m "feat(dsa): add FlashMLA adapter scaffold"
```

### Task 3: Add `sparse_backend` to the config and runtime dispatch

**Files:**
- Modify: `src/networks/dsa_2d.py`
- Modify: `tests/test_dsa_2d_sparse_attention.py`
- Modify: `tests/test_dsa_2d_regression.py`

**Step 1: Write the failing tests**

Add:

1. config validation test for `sparse_backend`
2. dispatch test proving:
   - `reference` path still uses `streaming_sparse_mla_reference`
   - `flashmla` path tries the adapter only when supported

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_regression.py -k "sparse_backend or flashmla_backend" -v`
Expected: FAIL because config and dispatch do not exist yet.

**Step 3: Write minimal implementation**

In `src/networks/dsa_2d.py`:

1. add `sparse_backend: str = "reference"` to `DSA2DMLAConfig`
2. validate allowed values
3. import `flashmla_sparse_mla_forward` and `flashmla_is_supported`
4. dispatch inside `forward_sparse_from_indices()`

Rules:

1. `reference` always uses `streaming_sparse_mla_reference`
2. `flashmla` uses the adapter only when supported
3. otherwise fall back to reference

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_regression.py -k "sparse_backend or flashmla_backend" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/networks/dsa_2d.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_regression.py
git commit -m "feat(dsa): add sparse backend dispatch for FlashMLA"
```

### Task 4: Implement the first tensor adapter

**Files:**
- Modify: `src/ops/dsa_flashmla.py`
- Modify: `tests/helpers/dsa_reference.py`
- Modify: `tests/test_dsa_2d_sparse_attention.py`

**Step 1: Write the failing tests**

Add small forward parity tests for the adapter on supported shapes:

```python
def test_flashmla_adapter_preserves_output_shape_small_case():
    q, k, v, idx = make_small_mqa_case()
    out = flashmla_sparse_mla_forward(..., force_reference_kernel=True)
    assert out.shape == (1, q.shape[1], q.shape[2], v.shape[-1])


def test_flashmla_adapter_matches_reference_small_case():
    q, k, v, idx = make_small_mqa_case()
    ref = streaming_sparse_mla_reference(...)
    out = flashmla_sparse_mla_forward(..., force_reference_kernel=True)
    torch.testing.assert_close(out, ref)
```

The first version may use a reference-mode adapter path before the real `FlashMLA` call is active, as long as the tensor packing contract is tested.

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -k "flashmla_adapter_preserves_output_shape_small_case or flashmla_adapter_matches_reference_small_case" -v`
Expected: FAIL because the adapter does not implement packing/unpacking yet.

**Step 3: Write minimal implementation**

In `src/ops/dsa_flashmla.py`:

1. pack `q/k/v/idx` into the kernel-facing layout
2. add a temporary internal reference-mode path for environments without `FlashMLA`
3. unpack output back to `[B, Hq, T, D_v]`

Do not build explicit `k_selected/v_selected`.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -k "flashmla_adapter_preserves_output_shape_small_case or flashmla_adapter_matches_reference_small_case" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/dsa_flashmla.py tests/helpers/dsa_reference.py tests/test_dsa_2d_sparse_attention.py
git commit -m "feat(dsa): add FlashMLA tensor adapter"
```

### Task 5: Wire the real `FlashMLA` call on H100

**Files:**
- Modify: `src/ops/dsa_flashmla.py`
- Modify: `artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py`
- Modify: `tests/test_dsa_2d_regression.py`

**Step 1: Write the failing regression tests**

Add tests that confirm:

1. adapter import remains safe when `FlashMLA` is not installed
2. `flashmla` backend still falls back cleanly in unsupported environments
3. benchmark harness records the backend mode

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_regression.py -k "flashmla" -v`
Expected: FAIL until the harness and adapter are updated.

**Step 3: Write minimal implementation**

In `src/ops/dsa_flashmla.py`:

1. call the real `FlashMLA` sparse forward when import and support checks pass
2. keep the safe fallback path otherwise

In `artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py`:

1. allow `sparse_backend`
2. add a focused `MQA` H100 benchmark path

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_regression.py -k "flashmla" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/dsa_flashmla.py artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py tests/test_dsa_2d_regression.py
git commit -m "feat(dsa): wire FlashMLA forward backend"
```

### Task 6: Full verification and H100 Slurm benchmark

**Files:**
- Modify: `docs/dsa_memory.md`
- Read: `artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py`

**Step 1: Run the full DSA test suite**

Run:

```bash
pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v
```

Expected: all green

**Step 2: Run focused H100 benchmark under Slurm**

Inside `/home/indoor/mlsp_wair_d/.worktrees/nsa-triton-longseq-investigation`:

```bash
salloc --gres=gpu:1 --partition=<h100-partition> --time=01:00:00
srun /usr/bin/bash -lc 'source ~/.bashrc >/dev/null 2>&1 || true; conda activate indoorolo && python artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py --device cuda --dtype bfloat16 --indexer-mode streaming --sparse-backend flashmla --warmup 5 --iters 10'
```

Use a focused `MQA` config first.

**Step 3: Record results**

Update `docs/dsa_memory.md` with:

1. support constraints
2. parity result
3. benchmark result
4. whether `flashmla` is the new recommended forward backend on H100 for the supported slice

**Step 4: Commit**

```bash
git add docs/dsa_memory.md
git commit -m "docs(dsa): record FlashMLA forward benchmark results"
```

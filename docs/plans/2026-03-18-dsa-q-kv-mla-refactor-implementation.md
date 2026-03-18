# DSA Q Plus KV MLA Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor `DSA2DMLAAttention` internals to use a DeepSeek-compatible `q + kv` MLA runtime contract while keeping the public module interface stable.

**Architecture:** Keep `DSA2DMLAAttention.forward(x)` unchanged, but replace the current separate `q/k/v` sparse runtime path with a packed MLA representation shared by both the pure-PyTorch reference backend and the future `FlashMLA` backend. Use strict TDD so every internal contract change is covered before code lands.

**Tech Stack:** PyTorch, local DSA test suite, H100 `FlashMLA` integration scaffold

---

### Task 1: Add failing tests for the packed MLA runtime contract

**Files:**
- Modify: `tests/test_dsa_2d_sparse_attention.py`
- Modify: `tests/helpers/dsa_reference.py`

**Step 1: Write the failing tests**

Add tests that require:
1. a new internal builder returning packed MLA runtime tensors
2. stable public forward shape despite the internal refactor
3. packed reference sparse execution parity on a tiny case

Suggested tests:
```python
def test_mla_runtime_builder_returns_packed_q_and_kv():
    mod = DSA2DMLAAttention(make_small_cfg(index_topk=2, n_kv_heads=1)).float()
    x = torch.randn(1, mod.dim, 2, 2)
    runtime = mod._dense_mla_runtime(x)
    assert set(runtime.keys()) >= {"q", "kv", "height", "width", "d_qk", "d_v"}


def test_public_forward_shape_is_unchanged_after_packed_runtime_refactor():
    mod = DSA2DMLAAttention(make_small_cfg(index_topk=2, n_kv_heads=1)).float()
    x = torch.randn(1, mod.dim, 2, 2)
    out = mod(x)
    assert out.shape == x.shape
```

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py -k "packed_q_and_kv or public_forward_shape_is_unchanged_after_packed_runtime_refactor" -v
```

Expected: FAIL because `_dense_mla_runtime` does not exist yet.

**Step 3: Commit the failing tests**

```bash
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation add tests/test_dsa_2d_sparse_attention.py tests/helpers/dsa_reference.py
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation commit -m "test(dsa): add packed MLA runtime contract tests"
```

### Task 2: Implement the minimal packed MLA runtime builder

**Files:**
- Modify: `src/networks/dsa_2d.py`
- Test: `tests/test_dsa_2d_sparse_attention.py`

**Step 1: Write minimal implementation**

Add a new internal helper, e.g. `_dense_mla_runtime(x)`, that returns:
- `q`
- `kv`
- `height`
- `width`
- `d_qk`
- `d_v`
- enough metadata for the backends to interpret the packed layout

Do not remove `_dense_mla_qkv(...)` yet if it helps incremental migration, but the new helper should become the new internal source of truth.

**Step 2: Run targeted tests**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py -k "packed_q_and_kv or public_forward_shape_is_unchanged_after_packed_runtime_refactor" -v
```

Expected: PASS.

**Step 3: Commit**

```bash
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation add src/networks/dsa_2d.py tests/test_dsa_2d_sparse_attention.py
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation commit -m "refactor(dsa): add packed MLA runtime builder"
```

### Task 3: Add failing tests for packed reference sparse MLA

**Files:**
- Modify: `tests/test_dsa_2d_sparse_attention.py`
- Modify: `tests/helpers/dsa_reference.py`
- Modify: `src/ops/dsa_sparse_mla.py`

**Step 1: Write the failing tests**

Add tests that require the reference sparse MLA backend to consume the packed runtime contract instead of separate `q/k/v`.

Suggested test shape:
```python
def test_packed_sparse_mla_reference_matches_old_small_reference():
    mod = DSA2DMLAAttention(make_small_cfg(index_topk=2, n_kv_heads=1)).float()
    x = torch.randn(1, mod.dim, 2, 2)
    runtime = mod._dense_mla_runtime(x)
    idx = torch.tensor([[[0, 1], [1, 0], [0, 1], [1, 0]]], dtype=torch.int64)
    out = packed_sparse_mla_reference(runtime, idx, gqa_group_size=mod.gqa_group_size, softmax_scale=mod.softmax_scale)
    ref = dsa_reference.sparse_mla_reference_from_indices(mod, x, idx)
    torch.testing.assert_close(out, ref)
```

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py -k "packed_sparse_mla_reference_matches_old_small_reference" -v
```

Expected: FAIL because the packed sparse reference path does not exist yet.

**Step 3: Commit the failing test**

```bash
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation add tests/test_dsa_2d_sparse_attention.py tests/helpers/dsa_reference.py
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation commit -m "test(dsa): add packed sparse MLA reference tests"
```

### Task 4: Implement packed reference sparse MLA

**Files:**
- Modify: `src/ops/dsa_sparse_mla.py`
- Modify: `src/networks/dsa_2d.py`
- Test: `tests/test_dsa_2d_sparse_attention.py`

**Step 1: Implement the minimal packed reference path**

Add a reference entrypoint that consumes the new packed MLA runtime contract and returns the same output tensor shape as before.

Update `forward_sparse_from_indices()` to use the packed runtime contract with the reference backend.

**Step 2: Run targeted tests**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py -k "packed_sparse_mla_reference_matches_old_small_reference or topk_equals_t" -v
```

Expected: PASS.

**Step 3: Commit**

```bash
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation add src/ops/dsa_sparse_mla.py src/networks/dsa_2d.py tests/test_dsa_2d_sparse_attention.py
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation commit -m "refactor(dsa): route sparse MLA through packed runtime contract"
```

### Task 5: Add failing tests for FlashMLA packed contract adapter

**Files:**
- Modify: `tests/test_dsa_2d_sparse_attention.py`
- Modify: `src/ops/dsa_flashmla.py`

**Step 1: Write the failing tests**

Add tests that require the `FlashMLA` adapter to consume the new packed MLA runtime contract, not separate `q/k/v` tensors.

Suggested test shape:
```python
def test_flashmla_adapter_accepts_packed_runtime_contract(monkeypatch):
    ...
```

Also keep the current gating tests green.

**Step 2: Run test to verify it fails**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py -k "flashmla_adapter_accepts_packed_runtime_contract or flashmla_backend" -v
```

Expected: FAIL because the adapter signature is still on the old contract.

**Step 3: Commit the failing test**

```bash
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation add tests/test_dsa_2d_sparse_attention.py
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation commit -m "test(dsa): require packed FlashMLA adapter contract"
```

### Task 6: Rewrite the FlashMLA adapter to the packed contract

**Files:**
- Modify: `src/ops/dsa_flashmla.py`
- Modify: `src/networks/dsa_2d.py`
- Test: `tests/test_dsa_2d_sparse_attention.py`

**Step 1: Implement minimal packed-contract adapter**

Update the adapter so it:
- accepts the packed MLA runtime tensors
- performs the minimal reshape required by `FlashMLA`
- keeps CPU/fallback and non-MQA gating behavior unchanged

Keep the actual cluster/H100 call behind support gating.

**Step 2: Run targeted tests**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py -k "flashmla" -v
```

Expected: PASS.

**Step 3: Commit**

```bash
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation add src/ops/dsa_flashmla.py src/networks/dsa_2d.py tests/test_dsa_2d_sparse_attention.py
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation commit -m "refactor(dsa): move FlashMLA adapter to packed MLA contract"
```

### Task 7: Run the full local DSA suite

**Files:**
- No code changes required

**Step 1: Run verification**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v
```

Expected: all DSA tests PASS.

**Step 2: Commit any final local test-driven cleanups**

```bash
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation add src/networks/dsa_2d.py src/ops/dsa_sparse_mla.py src/ops/dsa_flashmla.py tests/test_dsa_2d_sparse_attention.py tests/helpers/dsa_reference.py
 git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation commit -m "test(dsa): keep packed MLA contract green"
```

### Task 8: Update memory and re-run H100 forward validation

**Files:**
- Modify: `docs/dsa_memory.md`
- H100 cluster files are external to the repo

**Step 1: Update memory**

Record:
- new packed MLA runtime invariants
- any shape/dtype constraints for `FlashMLA`
- parity status

**Step 2: Commit memory update**

```bash
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation add docs/dsa_memory.md
git -C /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation commit -m "docs(dsa): record packed MLA runtime contract"
```

**Step 3: Re-run H100 smoke/forward parity**

Use the cluster `research` partition and the working `FlashMLA` install to validate the new adapter against the packed contract.

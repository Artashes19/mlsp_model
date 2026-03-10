# DSA Sparse MLA Reference Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the pure-PyTorch sparse MLA reference path so it preserves MQA/GQA KV sharing and avoids explicit selected `K/V` materialization while keeping current DSA correctness gates green.

**Architecture:** Replace the current `repeat_interleave + gather_selected_tensors` sparse path with a streaming sparse MLA reference helper that reads selected KV blocks on demand from `[B, h_kv, T, D]` tensors and accumulates output directly for `[B, h_q, T, D_v]`. Keep selector semantics unchanged in this slice. Use strict TDD and preserve dense-equivalence gates.

**Tech Stack:** PyTorch, pytest, torch.profiler, existing DSA reference helpers, A100 benchmarking.

---

### Task 1: Lock failing guardrail tests for the old sparse path

**Files:**
- Modify: `tests/test_dsa_2d_sparse_attention.py`
- Modify: `tests/test_dsa_2d_regression.py`
- Read: `src/networks/dsa_2d.py`
- Read: `src/ops/dsa_sparse_mla.py`

**Step 1: Write the failing tests**

Add tests that prove the sparse runtime path must stop using the current materialization pattern.

```python
def test_sparse_runtime_does_not_call_gather_sparse_mla_tokens(monkeypatch):
    mod = make_small_dsa_module(index_topk=3)
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1, 2]]]).expand(1, 4, 3).clone()

    def _fail(*args, **kwargs):
        raise AssertionError("old gather helper must not be used")

    monkeypatch.setattr("src.networks.dsa_2d.gather_sparse_mla_tokens", _fail)
    mod.forward_sparse_from_indices(x, idx)


def test_sparse_runtime_does_not_repeat_interleave_kv(monkeypatch):
    mod = make_small_dsa_module(index_topk=3)
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1, 2]]]).expand(1, 4, 3).clone()

    calls = []
    orig = torch.Tensor.repeat_interleave

    def _wrapped(self, *args, **kwargs):
        calls.append(self.shape)
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "repeat_interleave", _wrapped)
    mod.forward_sparse_from_indices(x, idx)
    assert not calls
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -k "does_not_call_gather_sparse_mla_tokens or does_not_repeat_interleave_kv" -v`
Expected: FAIL because the current sparse path still uses both patterns.

**Step 3: Commit test-only red state**

Do not commit yet. This task is the guardrail for the following implementation.

### Task 2: Add a naive streaming sparse MLA reference helper

**Files:**
- Modify: `src/ops/dsa_sparse_mla.py`
- Modify: `tests/helpers/dsa_reference.py`
- Modify: `tests/test_dsa_2d_sparse_attention.py`

**Step 1: Write the failing helper parity tests**

```python
def test_streaming_sparse_mla_matches_gather_reference_small_case():
    q, k, v, idx, g = make_small_sparse_case()
    ref = gather_sparse_mla_reference(q, k, v, idx, gqa_group_size=g, softmax_scale=q.shape[-1] ** -0.5)
    out = streaming_sparse_mla_reference(q, k, v, idx, gqa_group_size=g, softmax_scale=q.shape[-1] ** -0.5)
    torch.testing.assert_close(out, ref)


def test_streaming_sparse_mla_respects_gqa_head_mapping():
    q, k, v, idx, g = make_small_sparse_case(gqa_group_size=2)
    out = streaming_sparse_mla_reference(q, k, v, idx, gqa_group_size=g, softmax_scale=q.shape[-1] ** -0.5)
    ref = slow_per_head_kv_mapping_reference(q, k, v, idx, gqa_group_size=g, softmax_scale=q.shape[-1] ** -0.5)
    torch.testing.assert_close(out, ref)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -k "streaming_sparse_mla_matches_gather_reference_small_case or streaming_sparse_mla_respects_gqa_head_mapping" -v`
Expected: FAIL because the helper does not exist.

**Step 3: Write minimal implementation**

In `src/ops/dsa_sparse_mla.py`, add:
- `streaming_sparse_mla_reference(...)`
- helper loops over query blocks and selected-token blocks
- maps `query_head -> kv_head` by integer division with `gqa_group_size`
- reads selected KV blocks on demand from original `k/v`
- accumulates logits and output directly

In `tests/helpers/dsa_reference.py`, add small pure-reference helpers for:
- old gather-based sparse MLA reference
- explicit per-head KV mapping check

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -k "streaming_sparse_mla_matches_gather_reference_small_case or streaming_sparse_mla_respects_gqa_head_mapping" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/dsa_sparse_mla.py tests/helpers/dsa_reference.py tests/test_dsa_2d_sparse_attention.py
git commit -m "feat(dsa): add streaming sparse MLA reference helper"
```

### Task 3: Route `forward_sparse_from_indices()` through the new helper

**Files:**
- Modify: `src/networks/dsa_2d.py`
- Modify: `tests/test_dsa_2d_sparse_attention.py`
- Modify: `tests/test_dsa_2d_regression.py`

**Step 1: Write the failing routing/parity tests**

```python
def test_forward_sparse_from_indices_uses_streaming_sparse_helper(monkeypatch):
    mod = make_small_dsa_module(index_topk=3)
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1, 2]]]).expand(1, 4, 3).clone()
    sentinel = torch.randn(1, mod.n_heads, 4, mod.v_head_dim)

    def _streaming(*args, **kwargs):
        return sentinel

    monkeypatch.setattr("src.networks.dsa_2d.streaming_sparse_mla_reference", _streaming)
    out = mod.forward_sparse_from_indices(x, idx)
    assert out.shape == x.shape


def test_forward_sparse_from_indices_matches_old_small_reference():
    mod = make_small_dsa_module(index_topk=3)
    x = torch.randn(1, mod.dim, 2, 2)
    idx = torch.tensor([[[0, 1, 2]]]).expand(1, 4, 3).clone()
    out = mod.forward_sparse_from_indices(x, idx)
    ref = old_small_sparse_forward_reference(mod, x, idx)
    torch.testing.assert_close(out, ref)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -k "uses_streaming_sparse_helper or matches_old_small_reference" -v`
Expected: FAIL because `forward_sparse_from_indices()` still uses the old gather path.

**Step 3: Write minimal implementation**

In `src/networks/dsa_2d.py`:
- import `streaming_sparse_mla_reference`
- in `forward_sparse_from_indices()`:
  - keep `q, k, v, height, width = self._dense_mla_qkv(x)`
  - do **not** `repeat_interleave(k/v)`
  - call `streaming_sparse_mla_reference(...)`
  - reshape/project output as before

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -k "uses_streaming_sparse_helper or matches_old_small_reference or does_not_call_gather_sparse_mla_tokens or does_not_repeat_interleave_kv" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/networks/dsa_2d.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_regression.py
git commit -m "refactor(dsa): route sparse MLA through streaming reference path"
```

### Task 4: Re-lock dense-equivalence and backward gates

**Files:**
- Modify: `tests/test_dsa_2d_sparse_attention.py`
- Modify: `tests/test_dsa_2d_regression.py`
- Read: `src/networks/dsa_2d.py`

**Step 1: Write the failing hard/extra-hard tests**

```python
def test_sparse_mla_matches_dense_when_topk_equals_t_after_streaming_rewrite():
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4, requires_grad=True)
    dense = mod.forward_dense_reference(x)
    sparse = mod.forward_sparse_with_forced_topk(x, topk_equals_t=True)
    torch.testing.assert_close(sparse, dense)


def test_sparse_mla_backward_matches_dense_when_topk_equals_t_after_streaming_rewrite():
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg).float()
    x_dense = torch.randn(1, cfg.dim, 4, 4, requires_grad=True)
    x_sparse = x_dense.detach().clone().requires_grad_(True)
    compare_dense_vs_sparse_backward(mod, x_dense, x_sparse)


def test_sparse_mla_handles_duplicate_and_unsorted_indices_after_streaming_rewrite():
    cfg = make_small_cfg(index_topk=4)
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4)
    idx = torch.tensor([[[3, 1, 3, 0]]], dtype=torch.int64).expand(1, 16, 4).clone()
    out = mod.forward_sparse_from_indices(x, idx)
    assert torch.isfinite(out).all()
```

**Step 2: Run tests to verify they fail if behavior regressed**

Run: `pytest tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_regression.py -k "topk_equals_t or duplicate_and_unsorted" -v`
Expected: either PASS immediately or expose regressions from the rewrite. If they fail, fix only the root cause.

**Step 3: Fix minimal parity issues if needed**

Adjust only:
- online softmax accumulation logic
- query-head to KV-head mapping
- block-accumulation order

Do not change selector behavior.

**Step 4: Run targeted tests to verify they pass**

Run: `pytest tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_regression.py -k "topk_equals_t or duplicate_and_unsorted" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_regression.py src/networks/dsa_2d.py src/ops/dsa_sparse_mla.py
git commit -m "test(dsa): relock sparse MLA parity after streaming rewrite"
```

### Task 5: Full verification and profiling checkpoint

**Files:**
- Modify: `docs/dsa_memory.md`
- Read: `artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py`
- Read: `artifacts/dsa_diagnostics/profile_dsa_2d_module.py`

**Step 1: Run the full DSA suite**

Run: `pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
Expected: all green.

**Step 2: Run the DSA smoke benchmark**

Run: `python artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py --smoke --indexer-mode streaming`
Expected: smoke artifact generated and schema still valid.

**Step 3: Run the real A100 checkpoint benchmark**

Run on DGX A100:

```bash
ssh artashes@dgx.yc2.io "cd /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation && CUDA_VISIBLE_DEVICES=7 /auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py --device cuda --dtype bfloat16 --indexer-mode streaming --warmup 1 --iters 1"
```

Expected:
- `256x256` still runs
- memory/runtime should improve materially versus the previous streaming artifact
- if still bad, profiler should show the next bottleneck more clearly

**Step 4: Update memory**

Record in `docs/dsa_memory.md`:
- whether sparse MLA no longer materializes giant selected `K/V`
- updated `128x128` and `256x256` numbers
- what the next bottleneck becomes

**Step 5: Commit**

```bash
git add docs/dsa_memory.md artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py
# add any new tracked diagnostics scripts if changed
git commit -m "perf(dsa): checkpoint sparse MLA reference rewrite"
```

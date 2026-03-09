# DSA 2D MLA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a standalone non-causal `DSA2DMLAAttention` module for 2D inputs that stays as close as practical to DeepSeek DSA, including MLA, partial RoPE, a FWHT-plus-FP8 lightning indexer, sparse token selection, and the paper's indexer-training recipe.

**Architecture:** Implement the module in separate, testable layers: 2D partial RoPE helpers, MLA decomposition, indexer reference path, sparse MLA attention, and training losses. Use strict TDD with easy, hard, and extra-hard gates so every step has a failing test first and no subsystem advances until the dense-equivalence and gradient checks pass.

**Tech Stack:** PyTorch, Triton later if needed, DeepSeek-V3.2-Exp reference architecture, `fast-hadamard-transform`, pytest, torch.profiler, A100 benchmarking.

---

### Task 1: Seed DSA memory and test scaffolding

**Files:**
- Create: `docs/dsa_memory.md`
- Create: `tests/helpers/dsa_reference.py`
- Create: `tests/helpers/fp8_reference.py`
- Create: `tests/test_dsa_2d_rope.py`
- Create: `tests/test_dsa_2d_mla.py`
- Create: `tests/test_dsa_2d_indexer.py`
- Create: `tests/test_dsa_2d_sparse_attention.py`
- Create: `tests/test_dsa_2d_training.py`
- Create: `tests/test_dsa_2d_regression.py`

**Step 1: Write the failing smoke tests**

```python
def test_dsa_test_scaffold_exists():
    from tests.helpers import dsa_reference
    assert hasattr(dsa_reference, "__file__")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_dsa_2d_rope.py::test_dsa_test_scaffold_exists -v`
Expected: FAIL because the helper module or test file does not exist yet.

**Step 3: Write minimal scaffolding**

Create empty helper modules and placeholder test files with one importing smoke test.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dsa_2d_rope.py::test_dsa_test_scaffold_exists -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/dsa_memory.md tests/helpers/dsa_reference.py tests/helpers/fp8_reference.py tests/test_dsa_2d_*.py
git commit -m "test(dsa): seed DSA test scaffolding"
```

### Task 2: Lock 2D partial RoPE easy tests before code

**Files:**
- Modify: `tests/test_dsa_2d_rope.py`
- Create: `src/ops/dsa_rope.py`

**Step 1: Write the failing easy tests**

```python
def test_partial_rope_leaves_nope_slice_unchanged():
    q = torch.randn(1, 2, 4, 128)
    q_out = apply_partial_rope_2d_non_interleaved(q, H=2, W=2, rope_dim=64)
    assert torch.equal(q[..., :64], q_out[..., :64])


def test_partial_rope_does_not_touch_v():
    v = torch.randn(1, 2, 4, 128)
    v_out = maybe_apply_rope_to_v(v)
    assert torch.equal(v, v_out)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_rope.py -k "nope_slice or touch_v" -v`
Expected: FAIL because `src/ops/dsa_rope.py` does not exist or helpers are undefined.

**Step 3: Write minimal implementation**

Create `src/ops/dsa_rope.py` with:
- `apply_partial_rope_2d_non_interleaved(...)`
- `apply_partial_rope_2d_interleaved(...)`
- no-op helper for `V`

Implement only enough to preserve shapes and keep no-rope slices untouched.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_rope.py -k "nope_slice or touch_v" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_dsa_2d_rope.py src/ops/dsa_rope.py
git commit -m "test(dsa): add easy 2D partial RoPE coverage"
```

### Task 3: Add hard and extra-hard RoPE tests, then finish the helper

**Files:**
- Modify: `tests/test_dsa_2d_rope.py`
- Modify: `tests/helpers/dsa_reference.py`
- Modify: `src/ops/dsa_rope.py`

**Step 1: Write the failing hard and extra-hard tests**

```python
def test_non_interleaved_rope_matches_naive_reference():
    q = torch.randn(2, 3, 16, 128)
    ref = naive_partial_rope_2d_non_interleaved(q, H=4, W=4, rope_dim=64)
    out = apply_partial_rope_2d_non_interleaved(q, H=4, W=4, rope_dim=64)
    torch.testing.assert_close(out, ref)


def test_row_major_position_mapping_matches_explicit_coordinates():
    coords = positions_from_hw(H=3, W=5)
    assert coords[7] == (1, 2)
```

**Step 2: Run tests to verify they fail for the correct reason**

Run: `pytest tests/test_dsa_2d_rope.py -v`
Expected: FAIL on reference parity or coordinate mapping mismatches.

**Step 3: Write minimal implementation to pass**

Add:
- exact 2D coordinate mapping helpers
- interleaved and non-interleaved rotary pairing logic
- naive reference implementations in `tests/helpers/dsa_reference.py`

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_rope.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_dsa_2d_rope.py tests/helpers/dsa_reference.py src/ops/dsa_rope.py
git commit -m "feat(dsa): implement 2D partial RoPE helpers"
```

### Task 4: Lock MLA decomposition easy tests before code

**Files:**
- Create: `src/networks/dsa_2d.py`
- Modify: `tests/test_dsa_2d_mla.py`

**Step 1: Write the failing easy tests**

```python
def test_mla_config_rejects_non_gqa_shape():
    with pytest.raises(ValueError):
        DSA2DMLAConfig(dim=512, n_heads=6, n_kv_heads=4)


def test_mla_projection_splits_have_expected_sizes():
    cfg = DSA2DMLAConfig(dim=1536, n_heads=16, n_kv_heads=4,
                         q_lora_rank=512, kv_lora_rank=256,
                         qk_nope_head_dim=128, qk_rope_head_dim=64,
                         v_head_dim=128, index_n_heads=4, index_head_dim=128, index_topk=64)
    mod = DSA2DMLAAttention(cfg)
    assert mod.qk_head_dim == 192
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_mla.py -k "rejects_non_gqa_shape or splits_have_expected_sizes" -v`
Expected: FAIL because the config/module does not exist.

**Step 3: Write minimal implementation**

Create `src/networks/dsa_2d.py` with:
- `DSA2DMLAConfig`
- skeleton `DSA2DMLAAttention`
- shape validation and dimension bookkeeping only

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_mla.py -k "rejects_non_gqa_shape or splits_have_expected_sizes" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_dsa_2d_mla.py src/networks/dsa_2d.py
git commit -m "test(dsa): lock MLA config and split contracts"
```

### Task 5: Add dense MLA reference and hard backward tests

**Files:**
- Modify: `tests/helpers/dsa_reference.py`
- Modify: `tests/test_dsa_2d_mla.py`
- Modify: `src/networks/dsa_2d.py`

**Step 1: Write the failing hard tests**

```python
def test_dense_mla_forward_matches_reference_small_case():
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg)
    x = torch.randn(2, cfg.dim, 4, 4, dtype=torch.float32)
    ref = dense_mla_reference(mod, x)
    out = mod.forward_dense_reference(x)
    torch.testing.assert_close(out, ref)


def test_dense_mla_backward_matches_reference_small_case():
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg).float()
    x = torch.randn(1, cfg.dim, 4, 4, requires_grad=True)
    compare_dense_mla_backward(mod, x)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_mla.py -k "dense_mla" -v`
Expected: FAIL because `forward_dense_reference` and backward comparison helpers do not exist.

**Step 3: Write minimal implementation**

Add:
- dense MLA reference helper logic
- a dense MLA path inside `DSA2DMLAAttention` used only for correctness and teacher generation
- no sparse selector yet

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_mla.py -k "dense_mla" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/helpers/dsa_reference.py tests/test_dsa_2d_mla.py src/networks/dsa_2d.py
git commit -m "feat(dsa): add dense MLA reference path"
```

### Task 6: Lock indexer easy tests before code

**Files:**
- Create: `src/ops/dsa_indexer.py`
- Modify: `tests/test_dsa_2d_indexer.py`
- Modify: `tests/helpers/fp8_reference.py`

**Step 1: Write the failing easy tests**

```python
def test_fwht_matches_naive_reference():
    x = torch.randn(2, 8)
    ref = naive_fwht(x)
    out = fwht_last_dim(x)
    torch.testing.assert_close(out, ref)


def test_weighted_relu_index_score_matches_naive_reference():
    q = torch.randn(1, 2, 4, 128)
    k = torch.randn(1, 2, 4, 128)
    w = torch.randn(1, 4, 2)
    ref = naive_weighted_relu_index(q, k, w)
    out = weighted_relu_index_score(q, k, w)
    torch.testing.assert_close(out, ref)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_indexer.py -k "fwht or weighted_relu" -v`
Expected: FAIL because indexer helpers do not exist.

**Step 3: Write minimal implementation**

Create `src/ops/dsa_indexer.py` with:
- `fwht_last_dim(...)`
- `weighted_relu_index_score(...)`
- correctness-first implementations only

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_indexer.py -k "fwht or weighted_relu" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_dsa_2d_indexer.py tests/helpers/fp8_reference.py src/ops/dsa_indexer.py
git commit -m "test(dsa): lock basic indexer math"
```

### Task 7: Add FP8 helper tests and implement correctness-first FP8 indexer path

**Files:**
- Modify: `tests/test_dsa_2d_indexer.py`
- Modify: `tests/helpers/fp8_reference.py`
- Modify: `src/ops/dsa_indexer.py`
- Modify: `src/networks/dsa_2d.py`

**Step 1: Write the failing hard and extra-hard tests**

```python
def test_fp8_quant_dequant_matches_reference_scales():
    x = torch.randn(4, 128)
    ref_q, ref_s = reference_fp8_quant(x)
    q, s = act_quant_reference_safe(x)
    torch.testing.assert_close(s, ref_s)
    assert q.shape == ref_q.shape


def test_indexer_topk_handles_ties_and_all_negative_cases():
    scores = torch.tensor([[[-1.0, -1.0, -2.0, -3.0]]])
    idx = stable_topk(scores, k=2)
    assert idx.shape[-1] == 2
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_indexer.py -k "fp8 or ties or negative" -v`
Expected: FAIL because FP8 helpers or stable-topk helpers are missing.

**Step 3: Write minimal implementation**

Add:
- explicit scale computation
- correctness-first quant/dequant helpers
- deterministic top-k wrapper for tests
- `DSA2DIndexer` skeleton that returns logits and indices

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_indexer.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_dsa_2d_indexer.py tests/helpers/fp8_reference.py src/ops/dsa_indexer.py src/networks/dsa_2d.py
git commit -m "feat(dsa): add correctness-first FP8 indexer path"
```

### Task 8: Lock sparse MLA gather and topk-equals-T equivalence tests before code

**Files:**
- Create: `src/ops/dsa_sparse_mla.py`
- Modify: `tests/test_dsa_2d_sparse_attention.py`
- Modify: `tests/helpers/dsa_reference.py`
- Modify: `src/networks/dsa_2d.py`

**Step 1: Write the failing tests**

```python
def test_sparse_mla_gather_matches_reference_order():
    idx = torch.tensor([[[0, 3, 1]]], dtype=torch.int64)
    k = torch.randn(1, 2, 4, 192)
    ref = gather_tokens_reference(k, idx)
    out = gather_sparse_mla_tokens(k, idx)
    torch.testing.assert_close(out, ref)


def test_sparse_mla_matches_dense_when_topk_equals_T():
    cfg = make_small_cfg(index_topk=16)
    mod = DSA2DMLAAttention(cfg)
    x = torch.randn(1, cfg.dim, 4, 4, dtype=torch.float32, requires_grad=True)
    dense = mod.forward_dense_reference(x)
    sparse = mod.forward_sparse_with_forced_topk(x, topk_equals_T=True)
    torch.testing.assert_close(sparse, dense)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -v`
Expected: FAIL because sparse gather and sparse MLA paths do not exist.

**Step 3: Write minimal implementation**

Add:
- sparse gather helper
- sparse MLA reference path
- a forced-`topk=T` path for equivalence testing

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_dsa_2d_sparse_attention.py tests/helpers/dsa_reference.py src/ops/dsa_sparse_mla.py src/networks/dsa_2d.py
git commit -m "feat(dsa): add sparse MLA reference path"
```

### Task 9: Add sparse backward parity and regression stress tests

**Files:**
- Modify: `tests/test_dsa_2d_sparse_attention.py`
- Modify: `tests/test_dsa_2d_regression.py`
- Modify: `src/ops/dsa_sparse_mla.py`
- Modify: `src/networks/dsa_2d.py`

**Step 1: Write the failing hard and extra-hard tests**

```python
def test_sparse_mla_backward_matches_dense_when_topk_equals_T():
    cfg = make_small_cfg(index_topk=16)
    compare_sparse_and_dense_backward(cfg, H=4, W=4)


def test_sparse_mla_handles_repeated_and_unsorted_indices():
    idx = torch.tensor([[[3, 1, 3, 0]]], dtype=torch.int64)
    run_sparse_index_regression_case(idx)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_regression.py -k "topk_equals_T or repeated" -v`
Expected: FAIL because backward parity and regression handling are incomplete.

**Step 3: Write minimal implementation**

Fix the sparse MLA path until:
- forward parity holds
- backward parity holds
- repeated and unsorted index cases are handled or explicitly rejected with tested behavior

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_regression.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_regression.py src/ops/dsa_sparse_mla.py src/networks/dsa_2d.py
git commit -m "test(dsa): add sparse MLA backward and regression coverage"
```

### Task 10: Lock training-loss tests before code

**Files:**
- Modify: `tests/test_dsa_2d_training.py`
- Modify: `src/networks/dsa_2d.py`

**Step 1: Write the failing tests**

```python
def test_dense_teacher_distribution_is_normalized():
    probs = build_dense_teacher_distribution(torch.randn(1, 2, 4, 4))
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)))


def test_indexer_warmup_detaches_main_model_inputs():
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg)
    assert_warmup_detach_contract(mod)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_training.py -k "normalized or detach" -v`
Expected: FAIL because the teacher builder and warm-up detach logic do not exist.

**Step 3: Write minimal implementation**

Add:
- dense teacher distribution builder
- KL loss helper
- warm-up detach plumbing

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_training.py -k "normalized or detach" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_dsa_2d_training.py src/networks/dsa_2d.py
git commit -m "feat(dsa): add warm-up teacher and KL loss helpers"
```

### Task 11: Add hard and extra-hard training behavior tests

**Files:**
- Modify: `tests/test_dsa_2d_training.py`
- Modify: `tests/helpers/dsa_reference.py`
- Modify: `src/networks/dsa_2d.py`

**Step 1: Write the failing tests**

```python
def test_dense_warmup_reduces_kl_on_tiny_problem():
    history = run_tiny_indexer_warmup_steps(num_steps=5)
    assert history[-1] < history[0]


def test_warmup_updates_indexer_but_not_frozen_main_model():
    grads = run_warmup_and_collect_grad_flags()
    assert grads["indexer"] is True
    assert grads["main_model"] is False
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_training.py -k "reduces_kl or frozen_main_model" -v`
Expected: FAIL because the tiny warm-up loop does not exist.

**Step 3: Write minimal implementation**

Add the minimal warm-up and sparse-stage training helpers needed to pass the tests.

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_training.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_dsa_2d_training.py tests/helpers/dsa_reference.py src/networks/dsa_2d.py
git commit -m "test(dsa): validate DSA training-stage behavior"
```

### Task 12: Integrate the full module and add block-level smoke coverage

**Files:**
- Modify: `src/networks/dsa_2d.py`
- Modify: `tests/test_dsa_2d_mla.py`
- Optionally modify: `src/networks/txunet.py` only if a registry/export hook is needed

**Step 1: Write the failing integration tests**

```python
def test_dsa_2d_mla_attention_round_trips_image_shape():
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg)
    x = torch.randn(2, cfg.dim, 8, 8)
    out = mod(x)
    assert out.shape == x.shape


def test_dsa_sparse_and_dense_paths_share_projection_contract():
    cfg = make_small_cfg()
    mod = DSA2DMLAAttention(cfg)
    assert_projection_contract(mod)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_mla.py -k "round_trips_image_shape or projection_contract" -v`
Expected: FAIL because the full integrated forward path is incomplete.

**Step 3: Write minimal implementation**

Finish `DSA2DMLAAttention.forward(...)` by wiring together:
- MLA path
- indexer path
- sparse token selection
- sparse MLA attention
- output projection

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dsa_2d_mla.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/networks/dsa_2d.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py
git commit -m "feat(dsa): integrate standalone 2D MLA DSA module"
```

### Task 13: Add diagnostics and benchmark harnesses only after correctness gates are green

**Files:**
- Create: `artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py`
- Create: `artifacts/dsa_diagnostics/profile_dsa_2d_module.py`
- Modify: `docs/dsa_memory.md`

**Step 1: Write the failing artifact smoke tests**

```python
def test_dsa_benchmark_harness_smoke(tmp_path):
    result = run_benchmark_smoke(output_dir=tmp_path)
    assert "cases" in result
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dsa_2d_regression.py -k "benchmark_harness_smoke" -v`
Expected: FAIL because the benchmark harness does not exist.

**Step 3: Write minimal implementation**

Add harnesses that report:
- dense MLA vs sparse DSA-MLA with `topk = T`
- `128x128` runtime
- `256x256` runtime
- comparisons against current NSA at matched head counts and selected-token budgets

Update `docs/dsa_memory.md` with artifact paths and decisions.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_dsa_2d_regression.py -k "benchmark_harness_smoke" -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py artifacts/dsa_diagnostics/profile_dsa_2d_module.py docs/dsa_memory.md tests/test_dsa_2d_regression.py
git commit -m "perf(dsa): add DSA diagnostics harnesses"
```

### Task 14: Run the full required verification suite

**Files:**
- No code changes expected

**Step 1: Run the full DSA correctness suite**

Run: `pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
Expected: PASS.

**Step 2: Run targeted dense-equivalence checks**

Run: `pytest tests/test_dsa_2d_sparse_attention.py -k "topk_equals_T" -v`
Expected: PASS.

**Step 3: Run benchmark smokes**

Run: `/auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py --smoke`
Expected: JSON artifact is produced under `artifacts/dsa_diagnostics/`.

**Step 4: Review memory and artifact paths**

Confirm `docs/dsa_memory.md` includes:
- source links
- design invariants
- correctness gates
- benchmark artifact paths

**Step 5: Commit final verification state**

```bash
git status --short
```
Expected: clean worktree or only intended untracked diagnostic artifacts.

## Gate Policy

Do not advance from a task unless:

1. the new failing tests were written first
2. the failing tests were run and failed for the correct reason
3. the minimal implementation made them pass
4. the broader impacted suite still passes
5. the step has been committed

## Reuse References To Keep Open While Implementing

Architecture and kernels:

1. <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py>
2. <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py>
3. <https://github.com/deepseek-ai/FlashMLA>
4. <https://github.com/deepseek-ai/DeepGEMM>

Training and math references:

1. <https://arxiv.org/pdf/2512.02556>
2. <https://github.com/lemyx/tilelang-dsa>
3. <https://github.com/Dao-AILab/fast-hadamard-transform>

Chinese reference links:

1. <https://blog.csdn.net/jiemo99/article/details/153729664>
2. <https://blog.csdn.net/2401_85325557/article/details/152787564>
3. <https://blog.csdn.net/weixin_52610848/article/details/152378498>
4. <https://blog.csdn.net/2401_84204207/article/details/155490937>
5. <https://blog.csdn.net/gitblog_00400/article/details/154101623>

# NSA Per-Query Selection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace shared/top-level NSA selection with strict per-query selection (`[B, h_kv, T, top_k]`), keep non-causal behavior, and then recover performance with Triton per-query kernels.

**Architecture:** Two-phase rollout. Phase 1 introduces a correctness-first per-query selection path in `NSA2DAttention` and removes shared mode completely. Phase 2 adds Triton per-query kernels (forward/dQ first, then dK/dV two-pass reduction) and revalidates runtime/profiler metrics.

**Tech Stack:** PyTorch, Triton, torch.profiler, pytest

---

### Task 1: Add failing tests that lock new semantics (per-query only)

**Files:**
- Modify: `tests/test_nsa_gqa.py`
- Modify: `tests/test_selection_triton.py`
- Modify: `tests/test_selection_triton_gqa.py`

**Step 1: Add shape-contract tests for per-query indices**

Add tests in `tests/test_nsa_gqa.py` that validate selection-index contract from the selection scoring utility:

- For GQA (`gqa_group_size=4`), expect `block_idx.shape == (B, h_kv, T, top_k)`.
- For MHA (`gqa_group_size=1`), expect `block_idx.shape == (B, h_q, T, top_k)`.
- Assert `dtype == torch.int32` and `is_contiguous()`.

**Step 2: Add regression test forbidding shared mode**

Add test in `tests/test_nsa_gqa.py` that fails if model still emits shared indices (`[B, top_k]` or `[B,1,top_k]`).

**Step 3: Add non-causal behavior tests**

Add test in `tests/test_nsa_gqa.py` and/or `tests/test_selection_triton*.py` verifying there is no causal masking in selection outputs for crafted inputs.

**Step 4: Run new tests and confirm failures**

Run:
```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_nsa_gqa.py -k "per_query or shared_mode or non_causal" -v
```
Expected: FAIL (before implementation).

**Step 5: Commit test scaffolding**

```bash
git add tests/test_nsa_gqa.py tests/test_selection_triton.py tests/test_selection_triton_gqa.py
git commit -m "test(nsa): lock per-query selection semantics and non-causal behavior"
```

---

### Task 2: Implement per-query block index generation in `NSA2DAttention`

**Files:**
- Modify: `src/networks/txunet.py`
- Test: `tests/test_nsa_gqa.py`

**Step 1: Introduce a dedicated helper for per-query top-k scoring**

In `NSA2DAttention`, add helper (e.g. `_compute_per_query_block_idx`) that:

- Inputs: `q [B,h_q,T,d]`, `k_cmp [B,h_kv,n_patches,d]`, `top_k`, `chunk_size` policy
- Computes grouped scores with `q.view(B,h_kv,G,T,d)`
- Uses `sum` over grouped heads (`dim=2`) to produce `[B,h_kv,chunk,n_patches]`
- Returns `block_idx [B,h_kv,T,top_k]` (`int32`, contiguous)

**Step 2: Remove shared selection logic**

Delete existing MHA shared-topk path:

- remove `importance_batch = importance.sum(dim=1, keepdim=True)` branch
- remove `top_idx.squeeze(1)` usage in Triton call site

**Step 3: Keep strict non-causal behavior**

Ensure new helper and selection branch do not apply causal masks or causal-only special cases.

**Step 4: Run targeted tests**

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_nsa_gqa.py -k "per_query or shared_mode or non_causal" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/networks/txunet.py tests/test_nsa_gqa.py
git commit -m "feat(nsa): compute per-query per-kv-head selection indices"
```

---

### Task 3: Implement Phase-1 reference per-query selection compute path

**Files:**
- Modify: `src/networks/txunet.py`
- Modify: `tests/test_nsa_gqa.py`

**Step 1: Add reference selection compute utility**

Add internal utility (chunked over `T`) that consumes:

- `q [B,h_q,T,d]`, `k/v [B,h_kv,T,d]`, `block_idx [B,h_kv,T,top_k]`

Behavior:

- Gather per-query patch tokens into local selected K/V slices.
- Compute non-causal attention per query token.
- For GQA, broadcast KV-head outputs to grouped Q heads.

**Step 2: Wire `_selection_branch` to use per-query reference path**

For Phase 1, route both MHA and GQA through this reference path after per-query index generation.

**Step 3: Add parity tests vs naive/reference tensors**

In `tests/test_nsa_gqa.py`, add tests that compare selection output and gradients against explicit naive construction for small shapes.

**Step 4: Run tests**

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_nsa_gqa.py -k "selection" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/networks/txunet.py tests/test_nsa_gqa.py
git commit -m "feat(nsa): add per-query non-causal reference selection path"
```

---

### Task 4: Add dedicated per-query Triton op module (forward + dQ first)

**Files:**
- Create: `src/ops/selection_attention_2d_per_query.py`
- Modify: `tests/test_selection_triton.py`
- Modify: `tests/test_selection_triton_gqa.py`

**Step 1: Write failing Triton API tests**

Add tests for new per-query op signatures:

- MHA signature expects `block_idx [B,h_q,T,top_k]`
- GQA signature expects `block_idx [B,h_kv,T,top_k]` + `G`
- Forward parity vs naive reference
- Backward parity for `dq` first

Run and confirm failures.

**Step 2: Implement forward kernel(s) for per-query indices**

In new module:

- Implement query-token-centric kernel reading per-query block ids.
- Keep strict non-causal behavior.

**Step 3: Implement dQ backward kernel**

- Match reference gradients for `dq`.
- Keep `dk/dv` temporarily routed to reference or provisional kernel until Task 5.

**Step 4: Run tests**

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_selection_triton.py tests/test_selection_triton_gqa.py -k "per_query and (forward or dq)" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/selection_attention_2d_per_query.py tests/test_selection_triton.py tests/test_selection_triton_gqa.py
git commit -m "feat(nsa): add per-query triton selection forward and dQ"
```

---

### Task 5: Implement per-query dK/dV with two-pass reduction

**Files:**
- Modify: `src/ops/selection_attention_2d_per_query.py`
- Modify: `tests/test_selection_triton.py`
- Modify: `tests/test_selection_triton_gqa.py`

**Step 1: Add failing tests for `dk/dv` parity and stability**

- Gradient parity vs naive reference for `dk/dv`.
- Stress test with repeated block ids to validate deterministic accumulation behavior.

**Step 2: Implement one-pass baseline dK/dV (if needed)**

- Functional correctness first.

**Step 3: Upgrade to two-pass dK/dV reduction**

- Pass A: partial gradient tiles.
- Pass B: reduction to final `dk/dv`.
- Goal: reduce global atomic contention (Tilde/FLA-inspired direction).

**Step 4: Run tests**

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_selection_triton.py tests/test_selection_triton_gqa.py -k "per_query and backward" -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/selection_attention_2d_per_query.py tests/test_selection_triton.py tests/test_selection_triton_gqa.py
git commit -m "perf(nsa): add per-query two-pass dKV triton reduction"
```

---

### Task 6: Integrate per-query Triton path into `NSA2DAttention`

**Files:**
- Modify: `src/networks/txunet.py`
- Modify: `tests/test_nsa_gqa.py`

**Step 1: Replace Phase-1 reference path with Triton per-query path by default**

- Use new per-query Triton op for both MHA and GQA.
- Keep non-causal only.

**Step 2: Remove any legacy shared mode remnants and temporary code**

- Remove dead branches and temporary toggles.

**Step 3: Run full NSA test set**

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_nsa_gqa.py tests/test_nsa_2d_gpu.py tests/test_selection_triton.py tests/test_selection_triton_gqa.py -v
```
Expected: PASS.

**Step 4: Commit**

```bash
git add src/networks/txunet.py tests/test_nsa_gqa.py
git commit -m "feat(nsa): switch to per-query triton selection and remove shared mode"
```

---

### Task 7: Profiling and benchmark diagnostics

**Files:**
- Modify: `scripts/profile_nsa_torch_profiler.py`
- Create: `scripts/benchmark_nsa_per_query_selection.py`
- Output artifacts: `profile_traces/*`, benchmark logs

**Step 1: Extend profiler script to capture per-query selection path hotspots**

- Add labels for per-query index generation, selection forward, dQ, dK/dV.

**Step 2: Add benchmark script for kernel-level and block-level comparisons**

Benchmark matrix:

- `T`: 8192, 16384, 32768
- `d`: 12, 16, 24, 32, 64
- MHA and GQA configurations
- fwd+bwd timings + kernel breakdown

**Step 3: Run profiling and save outputs**

```bash
CUDA_VISIBLE_DEVICES=0 /auto/home/artashes/miniconda3/envs/dev/bin/python scripts/profile_nsa_torch_profiler.py
CUDA_VISIBLE_DEVICES=0 /auto/home/artashes/miniconda3/envs/dev/bin/python scripts/benchmark_nsa_per_query_selection.py
```

**Step 4: Commit scripts/logging improvements**

```bash
git add scripts/profile_nsa_torch_profiler.py scripts/benchmark_nsa_per_query_selection.py
git commit -m "chore(nsa): add per-query selection profiling and benchmark diagnostics"
```

---

### Task 8: End-to-end smoke and regression gate

**Files:**
- Modify: `tests/benchmark_per_layer.py` (if needed)
- Optional docs update: `docs/plans/2026-03-03-nsa-perf-investigation-design.md`

**Step 1: Run training smoke on A6000**

- Execute short forward+backward loop with NSA enabled and per-query path.
- Check for NaN/Inf and OOM.

**Step 2: Run per-layer benchmark comparison vs previous checkpointed numbers**

- Report deltas and kernel attribution.

**Step 3: Summarize findings in short report snippet**

Include:

- semantic fix confirmation
- runtime delta
- remaining bottlenecks

**Step 4: Commit**

```bash
git add tests/benchmark_per_layer.py docs/plans/2026-03-03-nsa-perf-investigation-design.md
# only if modified
# git commit -m "docs(nsa): record per-query selection rollout results"
```

---

## Definition of Done

1. No shared selection mode remains in runtime path.
2. Selection indices are per-query (`[B, h_kv, T, top_k]`) for both GQA and MHA semantics.
3. Strictly non-causal selection behavior.
4. Tests pass for forward/backward parity and integration.
5. profiler/benchmark artifacts produced for per-query diagnostics.
6. Triton path integrated with per-query dK/dV two-pass reduction.

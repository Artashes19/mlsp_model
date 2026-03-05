# NSA Forward and dQ v3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign the per-query NSA `forward` and `dQ` Triton kernels so they use a unified query-block mapping with real grouped-head width, reducing the dominant runtime bottlenecks on A100 and preparing for later H100 retuning.

**Architecture:** Keep the existing per-query tensor contract and direct `block_idx` consumption. Replace the current padded grouped-head execution in `forward` and `dQ` with unified query-block kernels that use `BLOCK_H` tied to real `G`, while leaving the compact `dK/dV` path unchanged. Validate on DGX A100 first; defer launch retuning for H100 until that machine is available.

**Tech Stack:** PyTorch, Triton, pytest, torch.profiler, CUDA events

---

### Task 1: Lock the current correctness contract before kernel changes

**Files:**
- Modify: `tests/test_selection_triton.py`
- Modify: `tests/test_selection_triton_gqa.py`

**Step 1: Add explicit regression labels for current parity coverage**

In both test files, add or rename tests so the following contracts are easy to run directly:

- MHA per-query forward matches naive
- MHA per-query `dQ` matches naive
- MHA per-query `dK/dV` matches naive
- GQA per-query forward matches naive
- GQA per-query `dQ` matches naive
- GQA per-query `dK/dV` matches naive

Do not change tolerances yet.

**Step 2: Run the focused parity suite**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest -v \
  tests/test_selection_triton.py::TestPerQuerySelectionForward::test_per_query_forward_matches_naive_mha \
  tests/test_selection_triton.py::TestPerQuerySelectionBackward::test_per_query_backward_dq_matches_naive_mha \
  tests/test_selection_triton.py::TestPerQuerySelectionBackward::test_per_query_backward_dk_dv_matches_naive_mha \
  tests/test_selection_triton_gqa.py::TestGQAPerQuerySelectionForward::test_per_query_forward_matches_naive_gqa \
  tests/test_selection_triton_gqa.py::TestGQAPerQuerySelectionBackward::test_per_query_backward_dq_matches_naive_gqa \
  tests/test_selection_triton_gqa.py::TestGQAPerQuerySelectionBackward::test_per_query_backward_dk_dv_matches_naive_gqa
```

Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_selection_triton.py tests/test_selection_triton_gqa.py
git commit -m "test(nsa): lock forward and dq parity coverage before v3 rewrite"
```

---

### Task 2: Add A100 benchmark harness for forward/dQ-only regression tracking

**Files:**
- Create: `artifacts/nsa_diagnostics/bench_forward_dq_v3_a100.py`

**Step 1: Add a focused benchmark script**

Create a script that times three paths for the current kernel implementation:

- `forward`
- `backward_q_only`
- `backward_qkv`

It must support both:

- GQA case: `Hq=4, Hkv=1, G=4`
- MHA case: `Hq=4, Hkv=4, G=1`

Use:

- `B=1`
- `H=W=256`
- `D=64`
- `P=8`
- `top_n=16`
- `dtype=bf16`
- warmup `5`
- iters `10`

Persist JSON output to `artifacts/nsa_diagnostics/`.

**Step 2: Run the script on DGX A100**

Run on an idle A100, e.g.:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=2 /auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/nsa_diagnostics/bench_forward_dq_v3_a100.py'
```

Expected: JSON artifact with baseline timings.

**Step 3: Commit**

```bash
git add artifacts/nsa_diagnostics/bench_forward_dq_v3_a100.py
git commit -m "perf(nsa): add focused A100 forward and dq benchmark harness"
```

---

### Task 3: Add failing structural tests for real grouped-head width

**Files:**
- Modify: `tests/test_selection_triton.py`
- Modify: `tests/test_selection_triton_gqa.py`

**Step 1: Add shape-and-regime tests**

Add tests that exercise:

- `G=1`
- `G=2`
- `G=4`

The tests should ensure the kernels still produce correct outputs and gradients without assuming padded grouped-head execution.

Use small shapes so they run quickly.

**Step 2: Run the tests and confirm they pass on the old code**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_selection_triton.py tests/test_selection_triton_gqa.py -k "per_query and (forward or dq)" -v
```

Expected: PASS.

**Step 3: Commit**

```bash
git add tests/test_selection_triton.py tests/test_selection_triton_gqa.py
git commit -m "test(nsa): add grouped-head regime coverage for forward and dq"
```

---

### Task 4: Rewrite `forward` as unified query-block v3 kernel

**Files:**
- Modify: `src/ops/selection_attention_2d_per_query.py`
- Test: `tests/test_selection_triton.py`
- Test: `tests/test_selection_triton_gqa.py`

**Step 1: Add the failing benchmark checkpoint**

Run the focused A100 benchmark from Task 2 and save the baseline artifact path in your notes before editing.

**Step 2: Replace padded grouped-head forward mapping**

In `src/ops/selection_attention_2d_per_query.py`:

- Replace the current query-single forward kernel with a query-block kernel
- Introduce `BLOCK_H` as the real grouped-head tile width
- Do not force `BLOCK_H >= 16`
- Keep direct reads from `block_idx`
- Keep non-causal behavior

Start conservatively:

- small `BLOCK_Q`
- no extra preprocessing
- preserve current external API

**Step 3: Run forward parity tests**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest -v \
  tests/test_selection_triton.py::TestPerQuerySelectionForward::test_per_query_forward_matches_naive_mha \
  tests/test_selection_triton_gqa.py::TestGQAPerQuerySelectionForward::test_per_query_forward_matches_naive_gqa
```

Expected: PASS.

**Step 4: Run focused benchmark**

Run the A100 benchmark harness and compare `forward` against baseline.

Expected: MHA forward improves materially; GQA forward does not regress badly.

**Step 5: Commit**

```bash
git add src/ops/selection_attention_2d_per_query.py tests/test_selection_triton.py tests/test_selection_triton_gqa.py
git commit -m "perf(nsa): rewrite per-query forward kernel with real grouped-head width"
```

---

### Task 5: Rewrite `dQ` as unified query-block v3 kernel

**Files:**
- Modify: `src/ops/selection_attention_2d_per_query.py`
- Test: `tests/test_selection_triton.py`
- Test: `tests/test_selection_triton_gqa.py`

**Step 1: Keep the new forward path fixed and isolate `dQ` changes**

Do not touch `dK/dV` in this task.

**Step 2: Replace padded grouped-head `dQ` mapping**

In `src/ops/selection_attention_2d_per_query.py`:

- rewrite `dQ` to use the same query-block family shape as forward
- use `BLOCK_H` based on real `G`
- keep direct `block_idx` reads
- allow modest numeric drift only if parity tests still pass

**Step 3: Run `dQ` parity tests**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest -v \
  tests/test_selection_triton.py::TestPerQuerySelectionBackward::test_per_query_backward_dq_matches_naive_mha \
  tests/test_selection_triton_gqa.py::TestGQAPerQuerySelectionBackward::test_per_query_backward_dq_matches_naive_gqa
```

Expected: PASS.

**Step 4: Run full per-query parity suite**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest -v \
  tests/test_selection_triton.py::TestPerQuerySelectionForward::test_per_query_forward_matches_naive_mha \
  tests/test_selection_triton.py::TestPerQuerySelectionBackward::test_per_query_backward_dq_matches_naive_mha \
  tests/test_selection_triton.py::TestPerQuerySelectionBackward::test_per_query_backward_dk_dv_matches_naive_mha \
  tests/test_selection_triton_gqa.py::TestGQAPerQuerySelectionForward::test_per_query_forward_matches_naive_gqa \
  tests/test_selection_triton_gqa.py::TestGQAPerQuerySelectionBackward::test_per_query_backward_dq_matches_naive_gqa \
  tests/test_selection_triton_gqa.py::TestGQAPerQuerySelectionBackward::test_per_query_backward_dk_dv_matches_naive_gqa
```

Expected: PASS.

**Step 5: Run focused A100 benchmark**

Run the benchmark harness again.

Expected:

- MHA `backward_q_only` improves materially
- GQA `backward_q_only` improves somewhat or remains near baseline
- `backward_qkv` improves by the same amount as `backward_q_only`, because `dK/dV` is already small

**Step 6: Commit**

```bash
git add src/ops/selection_attention_2d_per_query.py tests/test_selection_triton.py tests/test_selection_triton_gqa.py
git commit -m "perf(nsa): rewrite per-query dq kernel with real grouped-head width"
```

---

### Task 6: Run A100 profiler before/after comparison and summarize

**Files:**
- Create: `artifacts/nsa_diagnostics/forward_dq_v3_a100_summary_YYYYMMDD.md`

**Step 1: Run `torch.profiler` on DGX A100**

Use the same two shapes:

- GQA: `Hq=4, Hkv=1, G=4`
- MHA: `Hq=4, Hkv=4, G=1`

Capture the top CUDA kernels for compact-v2-after-v3.

**Step 2: Compare against previous profiler artifacts**

Reference:

- `artifacts/nsa_diagnostics/torch_profiler_compact_v2_gqa_hq4_hkv1_g4_dgx_a100_gpu1_20260305_000505.txt`
- `artifacts/nsa_diagnostics/torch_profiler_compact_v2_mha_hq4_hkv4_g1_dgx_a100_gpu1_20260305_000505.txt`

Summarize:

- old `forward`
- new `forward`
- old `dQ`
- new `dQ`
- note whether `dK/dV` remains stable

**Step 3: Write a short markdown summary**

Create a summary artifact in `artifacts/nsa_diagnostics/` with the before/after numbers and one-paragraph interpretation.

**Step 4: Commit**

```bash
git add artifacts/nsa_diagnostics
git commit -m "docs(nsa): summarize A100 forward and dq v3 profiler results"
```

---

### Task 7: Add H100 retuning checklist without blocking A100 delivery

**Files:**
- Modify: `docs/plans/2026-03-06-nsa-forward-dq-v3-design.md`

**Step 1: Add the post-A100 H100 tuning checklist**

Document:

- target host: `artashes@h100.yc2.io`
- rerun the focused benchmark harness
- rerun the profiler
- sweep `BLOCK_Q`, `BLOCK_H`, `BLOCK_KV`, `num_warps`
- lock final launch heuristics for H100

**Step 2: Commit**

```bash
git add docs/plans/2026-03-06-nsa-forward-dq-v3-design.md
git commit -m "docs(nsa): add H100 retuning checklist for forward and dq v3"
```

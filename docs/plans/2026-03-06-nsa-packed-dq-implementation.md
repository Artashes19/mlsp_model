# NSA Packed dQ Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a packed backward `dQ` path for per-query NSA selection, keep it numerically aligned with existing NSA variants and naive PyTorch references, and gate auto-enable on measured A100 speedups.

**Architecture:** Extend the current selection packing work into backward `dQ` only. Reuse packed metadata and contiguous packed K/V patch tables, preserve current grouped-head math, add explicit `selection_dq_mode` runtime control, and require parity against both unpacked and naive references. After the `dQ` decision, collect detailed full-layer profiling baselines for FFN and shell before proposing any optimization there.

**Tech Stack:** PyTorch, Triton, pytest, CUDA events, torch.profiler

---

### Task 1: Lock packed dQ parity targets in tests

**Files:**
- Modify: `tests/test_selection_triton.py`
- Modify: `tests/test_selection_triton_gqa.py`

**Step 1: Write the failing tests**

Add targeted CUDA tests for:

- packed `dQ` vs unpacked `dQ` in MHA
- packed `dQ` vs unpacked `dQ` in GQA
- packed `dQ` vs naive autograd `dQ` in at least one MHA case
- packed `dQ` vs naive autograd `dQ` in at least one GQA case

Use `NSA2DAttention(..., selection_dq_mode="unpacked")` and `selection_dq_mode="packed"` on the same `q/k/v/block_idx`.

**Step 2: Run the targeted tests to verify they fail**

Run:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest -k "packed and per_query and dq" tests/test_selection_triton.py tests/test_selection_triton_gqa.py -v'
```

Expected: FAIL because `selection_dq_mode` and packed `dQ` do not exist yet.

**Step 3: Commit**

```bash
git add tests/test_selection_triton.py tests/test_selection_triton_gqa.py
git commit -m "test(nsa): add packed dQ parity coverage"
```

---

### Task 2: Add runtime control for packed dQ

**Files:**
- Modify: `src/networks/txunet.py`

**Step 1: Write minimal runtime-control implementation**

Add `selection_dq_mode` with allowed values:

- `"unpacked"`
- `"packed"`
- `"auto"`

Keep initial `auto -> unpacked`.

This control should affect backward `dQ` only and must not change forward behavior.

**Step 2: Run a focused import/shape smoke check**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m py_compile src/networks/txunet.py
```

Expected: PASS.

**Step 3: Commit**

```bash
git add src/networks/txunet.py
git commit -m "feat(nsa): add selection dQ runtime mode control"
```

---

### Task 3: Add packed dQ kernel path

**Files:**
- Modify: `src/ops/selection_attention_2d_per_query.py`

**Step 1: Keep the existing unpacked dQ path untouched**

Do not rewrite the current `_sel_perq_bwd_dq_kernel`. Add a parallel packed path instead.

**Step 2: Implement the packed dQ path**

Add:

- packed backward wrapper
- packed Triton `dQ` kernel
- metadata/pointer wiring using:
  - `packed_idx`
  - `cu_unique_counts`
  - packed K/V patch tables

Preserve:

- non-causal semantics
- grouped-head math structure
- current softmax reconstruction logic

Do not touch `dK/dV`.

**Step 3: Wire backward dispatch**

Update custom autograd so `dQ` dispatch respects `selection_dq_mode`:

- `"unpacked"` -> current path
- `"packed"` -> new packed path
- `"auto"` -> current unpacked path for now

**Step 4: Run targeted tests**

Run:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest -k "packed and per_query and dq" tests/test_selection_triton.py tests/test_selection_triton_gqa.py -v'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/ops/selection_attention_2d_per_query.py src/networks/txunet.py tests/test_selection_triton.py tests/test_selection_triton_gqa.py
git commit -m "feat(nsa): add packed dQ backward path"
```

---

### Task 4: Re-run broader numerical parity suites

**Files:**
- No code changes required unless regressions are found

**Step 1: Run the full per-query parity suites**

Run:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest -k per_query tests/test_selection_triton.py tests/test_selection_triton_gqa.py -v'
```

Expected: PASS with the new packed `dQ` path present.

**Step 2: If any failure appears, fix only the root cause**

Do not change unrelated behavior. Re-run the same command until green.

**Step 3: Commit**

```bash
git add src/ops/selection_attention_2d_per_query.py src/networks/txunet.py tests/test_selection_triton.py tests/test_selection_triton_gqa.py
git commit -m "test(nsa): verify packed dQ against broader per-query parity suite"
```

---

### Task 5: Benchmark packed dQ against unpacked dQ

**Files:**
- Modify: `artifacts/nsa_diagnostics/bench_selection_packing_vs_unpacked.py`

**Step 1: Extend the benchmark**

Add measurements for:

- unpacked backward q-only
- packed backward q-only
- unpacked backward qkv
- packed backward qkv

Use the same cases already used for packing forward:

- `64x64`, `128x128`, `256x256`
- `C=64`, `Hq=4`, `Hkv=1`, `G=4`
- `p=8`, `w=16`
- `top_n=8`, `16`
- `bf16`

**Step 2: Run the benchmark on DGX A100**

Run:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/nsa_diagnostics/bench_selection_packing_vs_unpacked.py'
```

Expected: JSON artifact with packed vs unpacked backward timings.

**Step 3: Update memory**

Add to `docs/nsa_memory.md`:

- exact packed `dQ` winners at `128x128` and `256x256`
- whether packed `dQ` is strong enough to justify `auto`
- whether packing work should stop after `dQ`

**Step 4: Commit**

```bash
git add artifacts/nsa_diagnostics/bench_selection_packing_vs_unpacked.py docs/nsa_memory.md
git commit -m "perf(nsa): benchmark packed dQ against unpacked dQ"
```

---

### Task 6: Decide and wire `selection_dq_mode="auto"`

**Files:**
- Modify: `src/networks/txunet.py`
- Modify: `docs/nsa_memory.md`

**Step 1: Re-read the benchmark artifact**

Decision must be one of:

- keep `auto -> unpacked`
- enable packed only for a narrow long-sequence regime

Do not enable packed globally unless the data clearly supports it.

**Step 2: Implement the decision**

Update the `auto` policy in `txunet.py` based on the measured winning regimes only.

**Step 3: Verify parity again**

Run:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest -k per_query tests/test_selection_triton.py tests/test_selection_triton_gqa.py -v'
```

Expected: PASS.

**Step 4: Update memory**

Record:

- the `auto` policy
- why it is narrow or why it stayed unpacked

**Step 5: Commit**

```bash
git add src/networks/txunet.py docs/nsa_memory.md
git commit -m "feat(nsa): gate packed dQ auto mode on measured wins"
```

---

### Task 7: Build granular profiling baseline for FFN and attention shell

**Files:**
- Create: `artifacts/nsa_diagnostics/profile_nsa_layer_granular.py`
- Modify: `docs/nsa_memory.md`

**Step 1: Add a profiling harness**

The harness must separate and report at least:

- attention shell
  - `q_block`
  - `k_block`
  - `v_block`
  - gate
  - `proj`
- attention core
  - compression
  - selection
  - window
- FFN

Use `torch.profiler` and wall-clock timing where helpful.

Target at least:

- `128x128`
- `256x256`

**Step 2: Run the profiler on DGX A100**

Run:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/nsa_diagnostics/profile_nsa_layer_granular.py'
```

Expected: artifact(s) showing the shell/core/FFN split.

**Step 3: Update memory**

Record:

- exact baseline shares
- which shell or FFN components dominate
- no optimization proposal yet, only measured facts

**Step 4: Commit**

```bash
git add artifacts/nsa_diagnostics/profile_nsa_layer_granular.py docs/nsa_memory.md
git commit -m "perf(nsa): add granular baseline profiling for layer shell and ffn"
```

---

### Task 8: Write the post-profile decision note

**Files:**
- Modify: `docs/nsa_memory.md`

**Step 1: Add one explicit note**

State, based on the profiling artifacts, which comes next:

- FFN redesign
- attention-shell optimization
- or stop and revisit sparse attention if the profile still says selection dominates overwhelmingly

This note must cite the profiling artifact.

**Step 2: Verify the note exists**

Run:

```bash
rg -n "post-profile|next after profiling|granular baseline" docs/nsa_memory.md
```

Expected: one clear decision note appears.

**Step 3: Commit**

```bash
git add docs/nsa_memory.md
git commit -m "docs(nsa): record post-profile optimization priority"
```

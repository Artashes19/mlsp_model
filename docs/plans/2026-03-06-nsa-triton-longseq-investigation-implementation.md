# NSA Triton Long-Sequence Investigation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Profile the current Triton-based long-sequence NSA path deeply enough to identify the next single Triton-only optimization target.

**Architecture:** Keep the current backend and current selection semantics unchanged. Add dedicated profiling harnesses for the selection path and existing-kernel sweeps, run them on A100 with the practical long-sequence regimes, and end with a hard decision gate recorded in persistent NSA memory.

**Tech Stack:** PyTorch, Triton, torch.profiler, CUDA events, jq, JSON artifacts

---

### Task 1: Add a dedicated selection-path hotspot harness

**Files:**
- Create: `artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py`

**Step 1: Write the harness**

The harness must measure the current implementation without changing behavior. It should report, per case:

- selection scoring / `_compute_selection_block_idx`
- selection forward attention
- selection backward `dQ`
- selection backward `dK/dV`
- full selection forward+backward

Use the same case family as current long-sequence work:

- `B=1`
- `p=8`
- `w=16`
- `top_n=8, 16`
- `bf16`
- sizes: `128x128`, `256x256`
- configs:
  - `C=64, heads=4, G=4`
  - `C=384, heads=6, G=3`
  - `C=512, heads=8, G=4`

**Step 2: Smoke-check the harness**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m py_compile artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py
```

Expected: PASS

**Step 3: Commit**

```bash
git add artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py
git commit -m "perf(nsa): add Triton selection hotspot profiler"
```

---

### Task 2: Run the hotspot harness on A100 and save artifacts

**Files:**
- No code changes unless the harness fails

**Step 1: Run the profiler**

Run:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py'
```

Expected: JSON and text artifacts under `artifacts/nsa_diagnostics/`

**Step 2: If the harness fails, fix only the real issue**

Do not change NSA behavior. Re-run until artifacts are produced.

**Step 3: Commit only if code changed during fixes**

```bash
git add artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py
git commit -m "fix(nsa): repair Triton selection hotspot profiler"
```

---

### Task 3: Add an existing-kernel sweep harness for forward and dQ

**Files:**
- Create: `artifacts/nsa_diagnostics/bench_selection_triton_kernel_sweep.py`

**Step 1: Write the sweep harness**

The harness should benchmark the current kernel family over a controlled meta-parameter grid without changing kernel algorithms.

Targets:

- `_sel_perq_fwd_kernel`
- `_sel_perq_bwd_dq_kernel`

The sweep should vary practical existing choices such as:

- `BLOCK_Q`
- `num_warps`
- `num_stages`

Keep:

- current padded grouped-head strategy
- current math
- current non-causal behavior

Focus sweep cases:

- primary: `256x256`
- sentinel: `128x128`
- priority configs:
  - `C=384, heads=6, G=3`
  - `C=512, heads=8, G=4`

**Step 2: Smoke-check the harness**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m py_compile artifacts/nsa_diagnostics/bench_selection_triton_kernel_sweep.py
```

Expected: PASS

**Step 3: Commit**

```bash
git add artifacts/nsa_diagnostics/bench_selection_triton_kernel_sweep.py
git commit -m "perf(nsa): add Triton selection kernel sweep harness"
```

---

### Task 4: Run the kernel sweep on A100 and identify easy tuning headroom

**Files:**
- No code changes unless the harness fails

**Step 1: Run the sweep**

Run:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean && CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/nsa_diagnostics/bench_selection_triton_kernel_sweep.py'
```

Expected: JSON and text artifacts with per-config winners

**Step 2: If the harness fails, fix only the harness**

Re-run until artifacts are produced.

**Step 3: Commit only if code changed during fixes**

```bash
git add artifacts/nsa_diagnostics/bench_selection_triton_kernel_sweep.py
git commit -m "fix(nsa): repair Triton selection kernel sweep harness"
```

---

### Task 5: Build the ranked Triton-only hotspot table

**Files:**
- Modify: `docs/nsa_memory.md`

**Step 1: Re-read the new artifacts and summarize the ranking**

The ranking must include at least:

- hotspot name
- ms at `256x256`
- share
- current implementation type: PyTorch or Triton
- best Triton-only improvement hypothesis
- risk

**Step 2: Record the decision gate**

Update `docs/nsa_memory.md` with:

- whether block-index scoring or Triton kernels are the next target
- why
- where `128x128` differs, if it differs

**Step 3: Commit**

```bash
git add docs/nsa_memory.md
git commit -m "docs(nsa): rank Triton-only long-sequence hotspots"
```

---

### Task 6: Write the next optimization plan from the measured winner

**Files:**
- Create: `docs/plans/2026-03-06-nsa-triton-next-target-implementation.md`

**Step 1: Create a follow-on implementation plan for only the winning target**

The plan must choose one of:

- block-index scoring path
- selection forward kernel
- selection backward `dQ` kernel

Do not mix multiple primary targets into one implementation phase.

**Step 2: Save the follow-on plan**

Expected: one focused implementation plan for the next Triton-only squeeze target

**Step 3: Do not start implementation in this phase**

Stop after the plan is written. The purpose of this investigation phase is to remove ambiguity before the next optimization round.

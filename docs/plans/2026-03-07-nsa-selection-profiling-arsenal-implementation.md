# NSA Selection Profiling Arsenal Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add and validate a layered profiling stack for the NSA selection path using `torch.profiler`, HTA, Nsight Systems, Nsight Compute, and optional Triton Proton.

**Architecture:** Keep one selection-only workload harness as the source of truth, then attach progressively lower-level profiling tools to the same cases. Record what each tool adds and update persistent memory with the stable workflow.

**Tech Stack:** Python, PyTorch, HTA, Nsight Systems, Nsight Compute, Triton Proton, DGX A100 over SSH

---

### Task 1: Install and verify the profiling dependencies

**Files:**
- Modify: `docs/nsa_memory.md`

**Step 1: Verify the current environment**

Run:

```bash
which nsys
which ncu
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pip show HolisticTraceAnalysis || true
```

Expected:

- `nsys` and `ncu` present
- `HolisticTraceAnalysis` absent before install

**Step 2: Install HTA in the `dev` environment**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m pip install HolisticTraceAnalysis
```

Expected: successful install

**Step 3: Verify imports**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python - <<'PY'
import importlib.util
for mod in ["hta", "triton.profiler", "triton.profiler.proton"]:
    print(mod, bool(importlib.util.find_spec(mod)))
PY
```

Expected: all relevant modules importable

**Step 4: Update memory**

Record the environment/tool availability in `docs/nsa_memory.md`.

**Step 5: Commit**

```bash
git add docs/nsa_memory.md
git commit -m "chore(nsa): verify profiling tool environment"
```

### Task 2: Enhance the selection `torch.profiler` harness

**Files:**
- Modify: `artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py`

**Step 1: Write the failing smoke check**

Add CLI options for:

- `--with-execution-trace`
- `--with-memory-timeline`
- `--trace-dir`

Then run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m py_compile artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py
```

Expected before implementation: if flags are referenced but not wired, the smoke run should fail.

**Step 2: Implement minimal support**

Extend the harness to:

- export standard Kineto traces
- optionally attach `execution_trace_observer`
- optionally export memory timeline artifacts
- keep current timing behavior unchanged by default

**Step 3: Run a smoke trace**

Run on DGX A100:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation && CUDA_VISIBLE_DEVICES=3 /auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py --configs 384:6:3 --sizes 256 --top-ns 8 --with-execution-trace --with-memory-timeline'
```

Expected: profiler artifacts written successfully

**Step 4: Commit**

```bash
git add artifacts/nsa_diagnostics/profile_nsa_selection_triton_hotspots.py
git commit -m "perf(nsa): add richer torch profiler outputs for selection path"
```

### Task 3: Add HTA analysis over saved traces

**Files:**
- Create: `artifacts/nsa_diagnostics/analyze_selection_trace_with_hta.py`

**Step 1: Write the failing smoke check**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python -m py_compile artifacts/nsa_diagnostics/analyze_selection_trace_with_hta.py
```

Expected: FAIL before the file exists

**Step 2: Implement the analysis script**

The script should:

- accept a Kineto trace path
- load it with HTA
- emit a compact summary covering:
  - temporal breakdown
  - kernel breakdown
  - frequent kernels
  - trace diff hooks if two traces are provided

**Step 3: Run HTA on one selection trace**

Run:

```bash
/auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/nsa_diagnostics/analyze_selection_trace_with_hta.py --trace <path-to-kineto-trace>
```

Expected: summary output and saved artifact

**Step 4: Commit**

```bash
git add artifacts/nsa_diagnostics/analyze_selection_trace_with_hta.py
git commit -m "perf(nsa): add HTA analysis for selection traces"
```

### Task 4: Add an Nsight Systems capture recipe for the selection path

**Files:**
- Create: `artifacts/nsa_diagnostics/run_selection_nsys.sh`

**Step 1: Write the failing smoke check**

Run:

```bash
bash artifacts/nsa_diagnostics/run_selection_nsys.sh
```

Expected: FAIL before the file exists

**Step 2: Implement the wrapper**

The wrapper should:

- run the selection hotspot harness under `nsys`
- use Python CUDA backtraces if supported
- save `.nsys-rep` and exportable summary files into `artifacts/nsa_diagnostics/`

**Step 3: Run one A100 capture**

Run:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation && CUDA_VISIBLE_DEVICES=3 bash artifacts/nsa_diagnostics/run_selection_nsys.sh'
```

Expected: one valid Nsight Systems capture

**Step 4: Commit**

```bash
git add artifacts/nsa_diagnostics/run_selection_nsys.sh
git commit -m "perf(nsa): add Nsight Systems selection capture recipe"
```

### Task 5: Add an Nsight Compute capture recipe for the hottest kernels

**Files:**
- Create: `artifacts/nsa_diagnostics/run_selection_ncu.sh`

**Step 1: Write the failing smoke check**

Run:

```bash
bash artifacts/nsa_diagnostics/run_selection_ncu.sh
```

Expected: FAIL before the file exists

**Step 2: Implement the wrapper**

The wrapper should:

- target `_sel_perq_bwd_dq_kernel` first
- optionally target `_sel_perq_fwd_kernel`
- capture:
  - roofline-related sections
  - source hot spots
  - rules/recommendations
  - profile series if practical

**Step 3: Run one A100 capture**

Run:

```bash
ssh artashes@dgx.yc2.io 'cd /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation && CUDA_VISIBLE_DEVICES=3 bash artifacts/nsa_diagnostics/run_selection_ncu.sh'
```

Expected: one valid Nsight Compute report/export

**Step 4: Commit**

```bash
git add artifacts/nsa_diagnostics/run_selection_ncu.sh
git commit -m "perf(nsa): add Nsight Compute selection capture recipe"
```

### Task 6: Evaluate Proton if it adds value

**Files:**
- Create: `artifacts/nsa_diagnostics/profile_selection_with_proton.py` only if needed

**Step 1: Check whether Proton gives anything beyond current timing harnesses**

If it does not clearly add useful insight, skip implementation and record that decision in memory.

**Step 2: If useful, implement a minimal Proton-based microbenchmark**

Keep scope narrow:

- one or two selection kernels
- same case definitions as the other tools

**Step 3: Commit**

Commit only if a real script is added:

```bash
git add artifacts/nsa_diagnostics/profile_selection_with_proton.py docs/nsa_memory.md
git commit -m "perf(nsa): add Proton microbenchmark for selection kernels"
```

### Task 7: Run the comparative profiling pass and update memory

**Files:**
- Modify: `docs/nsa_memory.md`

**Step 1: Run the tool stack on the same selection cases**

Priority:

- `256x256`
- `C=384,h=6,G=3`
- `C=512,h=8,G=4`
- `top_n=8,16`

Sentinel:

- `128x128` on the same configs

**Step 2: Summarize what each tool added**

Record:

- what new information `torch.profiler` provided
- what HTA added
- what Nsight Systems added
- what Nsight Compute added
- whether Proton was useful or not

**Step 3: Decide the next kernel target**

Use the combined evidence to state whether the next action should target:

- `dQ`
- scoring
- scheduling/launch behavior

**Step 4: Update memory**

Write the stable findings into `docs/nsa_memory.md`.

**Step 5: Commit**

```bash
git add docs/nsa_memory.md artifacts/nsa_diagnostics docs/plans
git commit -m "docs(nsa): record selection profiling arsenal findings"
```

# NSA Selection Profiling Arsenal Design

**Goal**

Expand the profiling/tooling stack for the NSA selection path so we can extract more actionable hotspot information than the current timing-only harnesses provide.

**Scope**

- selection scoring / block-index computation
- selection forward
- selection backward `dQ`
- selection backward `dK/dV`

Do not profile the full transformer block in this phase.

## Current Context

We already have:

- a selection hotspot timing harness
- a direct Triton kernel sweep harness
- A100 baseline artifacts for the current selection path

What we do not yet have:

- richer `torch.profiler` traces with memory/export support
- HTA analysis on saved Kineto traces
- Nsight Systems capture recipes for selection-only runs
- Nsight Compute capture recipes for the hottest kernels
- a consistent “what extra information did each tool add?” report

## Environment Facts

Verified in the current environment:

- `nsys` exists at `/usr/local/bin/nsys`
- `ncu` exists at `/usr/local/cuda/bin/ncu`
- `triton.profiler` and `triton.profiler.proton` are importable in `dev`
- `HolisticTraceAnalysis` is not installed yet in `dev`

## Recommended Approach

Use one layered profiling stack, ordered from cheapest/high-level to most detailed/expensive:

1. Enhanced `torch.profiler`
2. HTA on exported Kineto traces
3. Nsight Systems on the same selection harness
4. Nsight Compute on only the hottest kernel(s)
5. Triton Proton only if it gives useful microbenchmark signal beyond the above

This avoids tool thrash and keeps all tools pointed at the same selection-path workloads.

## Tool Roles

### 1. torch.profiler

Use it to capture:

- operator-level attribution
- execution traces
- memory timelines
- isolated collection windows around selection regions

Expected value:

- separate scoring vs forward vs backward regions more clearly
- confirm whether memory behavior is contributing to selection bottlenecks

### 2. HTA

Use it on saved Kineto traces for:

- temporal breakdown
- kernel breakdown
- frequent-kernel analysis
- before/after trace diff

Expected value:

- easier comparison of profiling runs across commits
- less manual inspection than raw trace browsing

### 3. Nsight Systems

Use it to inspect:

- CPU launch gaps
- Python launch attribution
- CUDA scheduling/stream behavior

Expected value:

- determine whether any meaningful host-side overhead remains in the selection path

### 4. Nsight Compute

Use it only on the hottest kernels first:

1. `_sel_perq_bwd_dq_kernel`
2. `_sel_perq_fwd_kernel`

Capture:

- roofline
- source hot spots
- rules/recommendations
- profile series around kernel meta-parameters

Expected value:

- actual stall reason and memory/compute balance for the next kernel rewrite

### 5. Triton Proton

Treat this as optional in the same phase.

Expected value:

- Triton-native profiling/microbenchmark support

But it is lower priority than Nsight Compute because we need source-level stall diagnosis more urgently than another timing layer.

## Data Flow

Use one common selection workload definition:

- `128x128` sentinel
- `256x256` priority
- `C=384, h=6, G=3`
- `C=512, h=8, G=4`
- `top_n=8,16`

Outputs should land in `artifacts/nsa_diagnostics/` with a stable naming scheme so the results can be diffed and summarized together.

## Deliverables

1. enhanced selection profiler harness
2. HTA analysis script
3. Nsight Systems capture wrapper/script
4. Nsight Compute capture wrapper/script
5. comparative report of what each tool added
6. `docs/nsa_memory.md` update with any new stable findings

## Success Criteria

This phase is successful if:

1. each tool produces usable output on our selection workloads
2. each tool contributes at least one new piece of information beyond the current timing harness
3. we can state clearly whether the next selection work should target:
   - `dQ`
   - scoring
   - launch/scheduling behavior
4. we record the useful tools/workflow in memory so future profiling is not ad hoc

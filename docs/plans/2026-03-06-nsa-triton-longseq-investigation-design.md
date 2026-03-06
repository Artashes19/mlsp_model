# NSA Triton Long-Sequence Investigation Design

## Goal

Maximize the current Triton-based NSA path on long sequences before any backend pivot. The immediate objective is to identify which part of the current selection path is still the best Triton-only optimization target.

## Current State

The current evidence is already strong enough to narrow the search:

1. Long-sequence runtime is still dominated by the selection path.
2. `_compute_selection_block_idx` is still a PyTorch path built from `einsum + softmax + topk`.
3. The selection attention kernels are Triton:
   - `_sel_perq_fwd_kernel`
   - `_sel_perq_bwd_dq_kernel`
   - `_sel_perq_bwd_dkv_compact_kernel`
4. Packed forward is only a narrow locality improvement.
5. Packed `dQ` did not justify `auto`.
6. dK/dV is no longer the main bottleneck after the compact redesign.

The remaining open question is not whether selection dominates. It is which part of selection we should attack first while staying on the current backend:

- selection scoring / block-index computation
- selection forward attention
- selection backward `dQ`

## Scope

Primary focus:

- `256x256`

Sentinel check:

- `128x128`

Configs to keep in scope:

- `C=64, heads=4, G=4`
- `C=384, heads=6, G=3`
- `C=512, heads=8, G=4`

Fixed settings:

- `B=1`
- `p=8`
- `w=16`
- `top_n=8, 16`
- `bf16`
- A100
- non-causal only

## Non-Goals

1. No backend switch in this phase.
2. No FlexAttention / FA-style sparse backend yet.
3. No FFN redesign in this phase.
4. No shell redesign in this phase.
5. No causal support.

## Approaches Considered

### 1. Profiler-first selection investigation

Measure the current path carefully before changing anything:

- full block
- selection scoring only
- selection attention only
- backward split (`dQ`, `dK/dV`)

Pros:

- lowest risk
- directly answers where the next Triton-only gain is
- avoids another wrong optimization branch

Cons:

- does not immediately produce a speedup

### 2. Direct kernel tuning first

Start retuning `_sel_perq_fwd_kernel` and `_sel_perq_bwd_dq_kernel` immediately.

Pros:

- fastest path to touching performance-critical Triton code

Cons:

- high risk of optimizing the wrong part if scoring still dominates

### 3. Scoring-path-first

Assume `_compute_selection_block_idx` is the main blocker and attack it first.

Pros:

- potentially high upside if confirmed

Cons:

- risky if wider-channel regimes shift more time back into selection attention kernels

## Recommendation

Use approach 1.

The next step should be a Triton-only investigation pass that ends with a hard decision gate:

1. If selection scoring dominates, attack block-index computation first.
2. If selection attention forward / `dQ` dominates, attack Triton kernels first.

## Investigation Design

### 1. Full-block validation view

Keep using the full `TransformerBlock` path so the investigation remains tied to actual training-relevant runtime, not only microbenchmarks.

This view should confirm:

- total block forward and forward+backward
- how much attention still dominates at long sequence
- whether `128x128` and `256x256` rank hotspots differently

### 2. Selection-path decomposition

Add a dedicated harness for the selection path only, using the current implementation, to measure:

- `selection_block_idx`
- selection forward attention
- selection backward `dQ`
- selection backward `dK/dV`

This must stay numerically aligned with the current path and must not change selection semantics.

### 3. Existing-kernel sweep

Do not change the kernel algorithms yet. First sweep the existing Triton kernels over practical meta-parameter choices so we know whether easy tuning headroom still exists.

Targets:

- `_sel_perq_fwd_kernel`
- `_sel_perq_bwd_dq_kernel`

Examples of sweep axes:

- `BLOCK_Q`
- `num_warps`
- `num_stages`

The sweep should stay within the existing padded grouped-head strategy. The failed real-`G` rewrite is already a known dead end for now.

### 4. Decision gate

The investigation should end with a ranked hotspot table for long sequence containing:

- hotspot name
- current milliseconds
- share at `256x256`
- current implementation type: PyTorch or Triton
- best Triton-only improvement hypothesis
- expected upside
- risk

## Success Criteria

The phase is successful if it ends with one defensible answer to this question:

What is the next single Triton-only target with the best expected return for long-sequence NSA?

The acceptable answers are:

1. selection scoring / block-index path
2. selection forward kernel
3. selection backward `dQ` kernel

Anything less specific is not enough.

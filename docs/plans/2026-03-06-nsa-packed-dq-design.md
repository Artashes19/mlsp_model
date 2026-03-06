# NSA Packed dQ Design

**Date**: 2026-03-06
**Branch**: `dev-attn`
**Status**: Design

## Goal

Extend the new selection packing work from forward into backward `dQ`, while preserving NSA semantics and enforcing numerical parity against existing NSA variants and naive PyTorch references.

This design does not include packed `dK/dV`. It also does not propose FFN or shell optimizations yet; those remain blocked on granular profiling.

## Why This Is Next

Current evidence says:

1. `dK/dV` is no longer the main bottleneck after the compact Tilda-style redesign.
2. `dQ` remains a dominant backward hotspot.
3. Packing already showed a real forward win at `256x256`, especially for larger selected-token budgets.

From [nsa_memory.md](/auto/home/artashes/mlsp_model/dev-clean/docs/nsa_memory.md):

- `256x256, k=16`
  - selection forward: `13.053 -> 11.226 ms`
  - total attention forward: `21.389 -> 19.852 ms`
- decision gate:
  - proceed to packed `dQ`
  - do not start packed `dK/dV`

So the next justified packing target is `dQ`, not another `dK/dV` rewrite.

## Locked Constraints

1. This NSA path is fully 2D and fully non-causal.
2. Per-query selection semantics must remain unchanged.
3. Selection remains per KV head; for plain MHA it is also per head.
4. Shared selection mode stays removed.
5. Packed `dQ` must stay numerically aligned with:
   - current unpacked NSA path
   - naive PyTorch reference tests
6. Packed `dQ` must not auto-enable unless parity passes and measured speedup is real.
7. FFN and attention-shell optimization proposals are blocked until we collect detailed granular profiling baselines first.

## Scope

This design covers:

1. packed `dQ` kernel path
2. runtime control for `dQ`
3. parity validation
4. A100 benchmarks
5. required profiling baseline for FFN and shell after the `dQ` decision

This design does not cover:

1. packed `dK/dV`
2. FFN implementation changes
3. attention-shell implementation changes
4. H100 backend migration

## Recommended Approach

### 1. Backward-only packing for `dQ`

The first packed `dQ` rollout should build and use packed metadata only inside backward `dQ`.

Do not force forward and backward to share a single packed metadata lifecycle yet.

Why:

1. It isolates whether packed `dQ` itself is worth keeping.
2. It reduces the chance of cross-path regressions.
3. It keeps the parity story simpler.
4. It avoids inventing a larger packed execution model before the measured payoff is clear.

### 2. Reuse the current grouped-head math shape

Do not retry the failed real-`G` redesign.

The packed `dQ` path should preserve the current grouped-head structure and only change the K/V read layout:

- current:
  - `block_idx -> scattered 2D patch-token reads`
- packed `dQ`:
  - `block_idx -> packed_idx -> contiguous packed K/V patch reads`

This keeps the algorithmic behavior stable and limits the change to locality.

### 3. Add `selection_dq_mode`

Introduce a dedicated control for `dQ`:

- `"unpacked"`
- `"packed"`
- `"auto"`

Behavior:

- initial implementation keeps `auto -> unpacked`
- after parity and benchmark gates pass, `auto` may enable packed `dQ` only for winning regimes

This avoids globally enabling a path that only helps at `256x256`.

## Architecture

### Current `dQ` path

Current backward `dQ` uses:

- `_sel_perq_bwd_dq_kernel`
- inputs:
  - `q`
  - `k`
  - `v`
  - `do`
  - `lse`
  - `delta`
  - `block_idx`
  - `patch_starts`

Its main cost comes from repeatedly reconstructing selected K/V tokens through scattered 2D patch arithmetic.

### Packed `dQ` path

Packed `dQ` should add:

1. metadata build
   - `_build_packed_patch_metadata(block_idx)`
2. packed patch gather
   - gather contiguous packed K/V patch tables once per `(batch, kv_head)`
3. packed Triton `dQ` kernel
   - reads `packed_idx`
   - reads `cu_unique_counts`
   - reads packed K/V tables
   - preserves current grouped-head math and softmax reconstruction logic

### Expected benefit

Packed `dQ` does not reduce mathematical work. It targets:

1. better locality
2. fewer scattered token reads
3. better memory behavior at `256x256`

So expected gain is moderate, not dramatic. That is acceptable as long as it is real and restricted to the regimes where it wins.

## Runtime Policy

### Initial rollout

- `selection_forward_mode` remains as currently implemented
- add `selection_dq_mode`
- `selection_dq_mode="auto"` initially maps to unpacked

### Auto-enable gate

Packed `dQ` may become the `auto` path only if all are true:

1. parity passes against unpacked and naive references
2. no regressions in the broader per-query parity suites
3. A100 shows a real long-sequence `dQ` speedup
4. the speedup is strongest where we care most:
   - `256x256`

If those conditions are not met, `auto` stays unpacked.

## Testing and Numerical Alignment

Parity is the main guardrail.

### Required comparisons

1. packed `dQ` vs unpacked `dQ`
2. packed `dQ` vs naive autograd `dQ`
3. packed forward path vs unpacked forward path stays green
4. unpacked `dK/dV` remains unchanged and green

### Required test regimes

1. MHA
2. GQA
3. grouped-head regimes already covered by current per-query test suite

### Broader regression guard

After packed `dQ` lands, rerun the broader per-query suites:

- [test_selection_triton.py](/auto/home/artashes/mlsp_model/dev-clean/tests/test_selection_triton.py)
- [test_selection_triton_gqa.py](/auto/home/artashes/mlsp_model/dev-clean/tests/test_selection_triton_gqa.py)

The standard for success is not just "the new packed test passes". The standard is that packed and unpacked variants remain aligned with the naive references.

## Benchmark Plan

### Packed `dQ` benchmark

Use A100 and the same family of shapes already used for packing forward:

1. `64x64`, `128x128`, `256x256`
2. `C=64`, `Hq=4`, `Hkv=1`, `G=4`
3. `p=8`, `w=16`
4. `top_n=8`, `16`
5. `bf16`

Measure:

1. unpacked backward q-only
2. packed backward q-only
3. unpacked full backward qkv
4. packed full backward qkv

### Decision focus

The real decision cases are:

1. `128x128`
2. `256x256`

If packed `dQ` only helps at `256x256`, that is still acceptable. It just means `auto` must stay narrow.

## FFN and Shell Requirement

Do not propose FFN or shell changes yet.

Before any FFN or attention-shell optimization proposal, collect a detailed granular baseline that separates:

1. attention shell
   - `q_block`
   - `k_block`
   - `v_block`
   - gate
   - `proj`
2. attention core
   - compression
   - selection
   - window
3. FFN

That profiling baseline is required because the next full-block optimization decision must be evidence-based, not intuition-based.

## Risks

1. Packed `dQ` may show only a tiny gain.
   - Mitigation: keep runtime control explicit and gate `auto`.

2. Packed `dQ` may pass local tests but drift against naive references.
   - Mitigation: require both unpacked-parity and naive-parity tests.

3. Backward packing overhead could erase locality gains at smaller sizes.
   - Mitigation: benchmark `64`, `128`, `256` and do not generalize from one regime.

4. The new path could silently perturb full backward behavior.
   - Mitigation: rerun broader per-query suites, not just the new targeted tests.

## Success Criteria

1. Packed `dQ` is numerically aligned with unpacked and naive references.
2. No regression in existing per-query parity suites.
3. A100 shows a real long-sequence backward win, especially at `256x256`.
4. `selection_dq_mode="auto"` is only enabled for regimes supported by measured data.
5. FFN/shell work remains profiling-first until granular baselines exist.

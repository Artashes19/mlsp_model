# DSA Q Plus KV MLA Refactor Design

**Date**: 2026-03-18
**Branch**: `nsa-triton-longseq-investigation`
**Status**: Design

## Goal

Refactor `DSA2DMLAAttention` so its internal MLA runtime representation matches DeepSeek-style `q + kv` tensors instead of the current separate `q/k/v` sparse path, while keeping the public module interface stable.

This is the prerequisite for an honest H100 `FlashMLA` forward integration.

## Why This Refactor Exists

Current DSA MLA internals still diverge from the kernel contract that matters:

1. `_dense_mla_qkv(...)` returns separate `q`, `k`, and `v`
2. sparse reference execution consumes separate `q/k/v`
3. `FlashMLA` sparse prefill is built around a DeepSeek-style MLA representation where the sparse kernel consumes:
   - `q: [s_q, h_q, d_qk]`
   - `kv: [s_kv, h_kv, d_qk]`
   - sparse `indices`
4. forcing a direct swap from the current `q/k/v` path would add repacking and semantic mismatch, which defeats the point of the kernel integration

So the next high-value step is not more adapter logic. It is aligning the DSA MLA representation to the kernel’s native contract.

## Approaches Considered

### 1. Recommended: Keep public interface stable, change internals only

- Preserve `DSA2DMLAAttention.forward(x) -> out`
- Refactor internal MLA build path to produce:
  - `q`
  - `kv_payload`
  - metadata needed to interpret NoPE / RoPE / value slices
- Update reference sparse MLA path and future `FlashMLA` path to consume that contract

Why this is best:

1. this is the path that actually enables speedup
2. it minimizes API churn and test churn
3. it lets us compare `reference` vs `flashmla` under the same internal representation

### 2. Change the public interface too

- Expose lower-level packed tensors directly
- Useful for kernel microbenchmarks
- Not useful for the actual DSA layer as the main entrypoint

Why not now:

1. does not materially improve end-to-end speed by itself
2. adds API churn without solving the core integration mismatch

### 3. Keep current internals and add a heavier adapter

- Leave `q/k/v` path alone
- Build a translation shim into `FlashMLA`

Why not:

1. this preserves the wrong runtime shape
2. adapter overhead and semantic mismatch would contaminate the performance result
3. if it “works,” it still would not tell us whether the real kernel path is good

## Chosen Design

Use approach 1.

Keep the public DSA module stable, and change only the internal MLA runtime contract.

## Target Internal Contract

Instead of returning separate `q`, `k`, `v` from the main MLA builder, introduce a native MLA representation:

1. `q`: query tensor for all query heads
2. `kv`: packed key/value payload in a DeepSeek-compatible MLA layout
3. runtime metadata describing:
   - `d_qk`
   - `d_v`
   - how the NoPE / RoPE / value slices are packed
   - head counts and GQA/MQA mapping

This refactor does **not** mean the DSA module output changes. It only changes how the sparse MLA execution path receives its inputs.

## Internal Components To Change

### 1. MLA builder in `src/networks/dsa_2d.py`

Current:

- `_dense_mla_qkv(...) -> (q, k, v, H, W)`

Target:

- new builder that returns packed MLA runtime tensors, e.g.:
  - `q`
  - `kv`
  - `height`
  - `width`
  - maybe a small metadata object or dataclass if needed

### 2. Reference sparse MLA path in `src/ops/dsa_sparse_mla.py`

Current:

- accepts separate `q/k/v`

Target:

- accepts the same packed MLA runtime contract that `FlashMLA` will use
- remains pure PyTorch and correctness-first
- is the reference backend for parity against the kernel path

### 3. FlashMLA adapter in `src/ops/dsa_flashmla.py`

Current:

- scaffold only
- still pretends the path is `q/k/v/idx`

Target:

- consumes the same packed MLA runtime contract as the reference path
- performs only the minimal reshape/flatten needed for `FlashMLA`
- no semantic repacking beyond what the kernel contract requires

## Data Flow After Refactor

New DSA forward path should become:

1. input `x[B,C,H,W]`
2. build selector inputs as today
3. build sparse token indices as today
4. build packed MLA runtime tensors:
   - `q`
   - `kv`
5. sparse backend dispatch:
   - `reference`: packed reference sparse MLA
   - `flashmla`: H100 `FlashMLA` path when supported
6. output projection
7. reshape back to `[B,C,H,W]`

This makes the kernel integration meaningful because both backends will share the same internal representation.

## Testing Strategy

This refactor must be done with TDD.

### Easy tests

1. MLA builder returns tensors with the expected packed contract shape
2. metadata fields are consistent with config
3. public `forward(x)` output shape remains unchanged

### Hard tests

1. packed reference sparse MLA matches the old small reference on tiny cases
2. `topk=T` sparse path still matches dense forward on supported small shapes
3. backward parity still holds for the reference backend

### Extra-hard tests

1. backend dispatch sees the same internal packed contract for `reference` and `flashmla`
2. `flashmla` fallback gating still works when backend is unsupported
3. small adapter parity remains intact after the representation change

## Success Criteria

This refactor is successful if:

1. the public DSA module API does not change
2. the internal sparse MLA path no longer depends on separate `q/k/v` runtime tensors
3. local DSA parity tests still pass
4. the `FlashMLA` adapter can be rewritten against the new contract cleanly

## Immediate Next Step

1. write failing tests for the new MLA packed runtime contract
2. implement the minimal internal refactor to satisfy those tests
3. then rewrite the reference sparse MLA path to the new contract
4. only after that wire the H100 `flashmla` backend to the same contract

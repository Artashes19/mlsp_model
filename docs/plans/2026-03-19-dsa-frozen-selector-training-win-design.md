# DSA Frozen-Selector Training-Win Design

## Goal

Deliver a concrete training-step speedup for DSA over dense FlashAttention at `256x256`, batch size `1`, on H100.

This design intentionally narrows the problem to the shortest path that can win the benchmark. It does not try to preserve the full paper-style sparse-training recipe during the first performance push.

## Scope

The target path is:

1. a selector that has already been warmed up and is frozen during the sparse training phase
2. `DeepGEMM` selector forward on supported H100 native MQA cases
3. `FlashMLA` sparse forward on the absorbed MLA runtime
4. a custom sparse backward on that same absorbed runtime
5. full training-step benchmarking against dense FlashAttention on the real model path

Out of scope for this slice:

1. selector backward
2. selector updates during sparse training
3. GQA backward
4. non-native runtime shapes
5. generalized deployment policy outside the supported H100 slice

## Success Metric

Only one benchmark counts:

1. same model shell
2. same `256x256` input
3. same batch size
4. same AMP and optimizer policy
5. same outer model code except the attention implementation
6. compare end-to-end training step:
   - forward
   - backward
   - optimizer step

Baseline:

1. dense FlashAttention path

Candidate:

1. DSA with frozen selector
2. `DeepGEMM` selector forward where supported
3. `FlashMLA` sparse forward
4. custom sparse backward

Microbenchmarks and operator-only wins are supporting evidence only.

## Operator Boundary

The training hot path should be reduced to one narrow custom autograd boundary around sparse MLA.

The operator input contract is the absorbed MLA runtime that is already present in `DSA2DMLAAttention`:

1. `q_runtime [B, h_q, T, d_qk]`
2. `kv_runtime [B, h_kv, T, d_qk]`
3. `indices [B, T, topk]`
4. `softmax_scale`

For the first slice:

1. `h_kv = 1`
2. H100 only
3. forward and backward only for the supported native absorbed runtime

The operator returns latent sparse output `[B, h_q, T, d_v]`, which then continues through:

1. `W_UV`
2. existing output projection

The selector is outside this autograd boundary and is treated as frozen for the training-speed benchmark.

## Forward Contract

Forward should continue to use the proven fast path:

1. selector forward computes indices
2. sparse attention forward calls `FlashMLA`

The sparse forward contract is the actual public `FlashMLA` sparse prefill contract:

1. `q [s_q, h_q, d_qk]`
2. `kv [s_kv, h_kv, d_qk]`
3. `indices [s_q, h_kv, topk]`
4. output plus softmax statistics

The local absorbed runtime already matches the important semantics:

1. `q` carries absorbed NoPE plus RoPE
2. `kv` is packed latent-value plus RoPE
3. the first `d_v` channels of `kv` are the value payload
4. the full `d_qk` channels participate in score computation

So the backward should be designed around this exact packed runtime, not around the old separate `q/k/v` reference.

## Backward Contract

The first custom backward only needs to solve the trainable sparse attention core.

It must produce:

1. `dQ_runtime`
2. `dKV_runtime`

`dKV_runtime` has two logical contributions:

1. value-path gradient into `kv[..., :d_v]`
2. score-path gradient through the full packed `kv` channels

The backward should not attempt to propagate:

1. selector gradients
2. gradients through the discrete indices

The rest of the model gradients then flow by normal PyTorch autograd through:

1. `W_UV`
2. `W_UK`
3. `wq_a/q_norm/wq_b`
4. `wkv_a/kv_norm/wkv_b`

## Reference Baseline

The reference truth for backward correctness is a pure PyTorch absorbed sparse operator with autograd.

That reference must:

1. consume the same packed runtime `q_runtime`, `kv_runtime`, and `indices`
2. implement the same sparse attention math as the `FlashMLA` reference
3. let PyTorch autograd produce:
   - `dQ_runtime`
   - `dKV_runtime`

This is the gradient-truth baseline.

It is not the performance baseline.

Dense FlashAttention remains the performance baseline for the final training-step comparison.

## Training Recipe For This Slice

This slice assumes:

1. dense burn-in already exists or will be done separately
2. selector warm-up already exists or will be done separately
3. selector is frozen during the measured sparse training phase

This is an intentional shortcut to remove selector backward from the critical path to the first training-speed win.

If selector staleness becomes a quality problem later, it can be addressed separately by:

1. periodic selector refresh
2. longer warm-up
3. later selector-backward work

Those are not part of this performance-first slice.

## Testing Strategy

Strict TDD remains mandatory.

### Correctness gates

1. packed-runtime sparse reference forward tests
2. packed-runtime sparse autograd backward tests
3. custom backward parity against the packed autograd reference
4. frozen-selector gradient-isolation tests
5. module-level train-step smoke on supported H100 native shapes

### Benchmark gates

1. sparse operator forward+backward benchmark:
   - reference packed sparse op
   - `FlashMLA` forward + custom backward
2. block/module-level training benchmark
3. final full-model benchmark at `256x256`

No success claim is valid until the final full-model benchmark is run.

## Implementation Order

1. add packed-runtime sparse reference helpers and tests
2. add frozen-selector training mode and tests
3. add custom sparse autograd wrapper with reference backward placeholder
4. add correctness tests for runtime gradients
5. replace backward placeholder with the optimized sparse backward path
6. benchmark the sparse operator
7. benchmark the full training step against dense FlashAttention at `256x256`

## Risks

1. selector forward can still erase sparse attention gains if not kept cheap enough
2. custom sparse backward may initially be correct but too slow
3. exact dense-equivalence is not expected because DSA is a different operator; correctness means parity to the packed sparse reference, not to dense FlashAttention
4. if the frozen selector is too stale, speed may improve while model quality regresses

## Decision

This is the shortest defensible route to the stated objective:

1. do not spend time on selector backward now
2. do not broaden support now
3. focus only on the frozen-selector sparse training step at `256x256`

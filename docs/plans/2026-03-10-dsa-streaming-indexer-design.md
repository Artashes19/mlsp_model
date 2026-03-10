# DSA Streaming Indexer Design

## Goal

Remove the first DSA memory wall by replacing the dense `T x T` indexer materialization with an exact streaming top-k indexer, while keeping current DSA selector semantics unchanged.

## Problem

The current correctness-first DSA implementation OOMs at `256x256` before it can benefit from sparsity.

Current failure points:

1. The indexer materializes dense per-head logits in [src/ops/dsa_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/ops/dsa_indexer.py) via `weighted_relu_index_score(...)`, shape `[B, j, T, T]`.
2. It then reduces to dense scores `[B, T, T]`.
3. It then applies full stable sort over the dense score matrix.
4. Only after that does sparse MLA start, so the memory blow-up happens too early.

At `256x256`, `T = 65536`, so dense score tensors already cost tens of GiB.

## Non-Goals

This step does not try to solve all DSA runtime issues.

Out of scope for this rewrite:

1. sparse MLA gather/materialization rewrite
2. Triton kernel rewrite for the indexer
3. approximate selector semantics
4. FlashMLA / DeepGEMM backend integration

## Chosen Approach

Introduce an exact streaming indexer that processes keys in blocks and maintains a running per-query top-k.

Key properties:

1. exact semantics relative to the current dense weighted-ReLU score
2. no dense `[B, j, T, T]` materialization
3. no dense `[B, T, T]` materialization
4. no full `argsort` over all keys
5. preprocessing remains unchanged:
   - DeepSeek-style indexer projections
   - non-interleaved 2D partial RoPE
   - FWHT
   - FP8 quantization path

## API Changes

### New low-level helper

Add a new helper in [src/ops/dsa_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/ops/dsa_indexer.py):

- `streaming_weighted_relu_topk(q, k, w, topk, block_s)`

Inputs:

- `q[B, j, T, D]`
- `k[B, j, S, D]`
- `w[B, T, j]`
- `topk: int`
- `block_s: int`

Outputs:

- `topk_scores[B, T, K]`
- `topk_idx[B, T, K]`

### DSA module integration

Add an explicit mode in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py):

- `indexer_mode = "dense" | "streaming" | "auto"`

Rollout policy:

1. default starts conservative
2. `auto` resolves to `dense` until parity and OOM tests pass
3. after validation, `auto` can select `streaming` for long-sequence regimes

## Exact Streaming Algorithm

For each batch item:

1. Initialize running top-k buffers:
   - `scores[B, T, K] = -inf`
   - `idx[B, T, K] = -1`

2. Iterate over key blocks `s0:s1` of width `BLOCK_S`.

3. For each block:
   - compute weighted-ReLU scores only for that block
   - resulting shape: `[B, T, block_s]`
   - compute absolute token indices for this block

4. Merge step:
   - concatenate running candidates and block candidates
   - candidate score shape becomes `[B, T, K + block_s]`
   - candidate index shape becomes `[B, T, K + block_s]`
   - run local top-k over this merged candidate set only
   - keep best `K`

5. Continue until all key blocks are processed.

6. Return final exact global top-k scores and indices.

This remains exact because every key token is considered once and only pruned after comparison with the current best set.

## Tie Handling

The current dense reference uses stable descending sort.

The streaming path must define deterministic tie behavior explicitly. The planned rule is:

1. higher score wins
2. if scores tie, lower absolute token index wins

Tests must lock this rule before any rollout.

## Memory Impact

Expected memory reduction for the indexer:

Current dense path allocates:

1. `[B, j, T, T]`
2. `[B, T, T]`
3. full sort workspace / indices

Streaming path allocates only:

1. current running top-k `[B, T, K]`
2. per-block temporary scores `[B, T, BLOCK_S]`
3. merge buffers `[B, T, K + BLOCK_S]`

That changes indexer memory from `O(T^2)` to `O(T * (K + BLOCK_S))`.

## Expected Limitation After This Step

Even with the streaming indexer, DSA can still OOM later in sparse MLA because the current sparse path explicitly gathers `K/V` into `[B, h, T, K, D]` tensors.

So the success condition for this step is narrower:

1. remove indexer-originated OOM
2. preserve exact selector semantics
3. provide a base for the next rewrite of sparse MLA gather/materialization

## Testing Strategy

Strict TDD remains mandatory.

### Easy tests

1. streaming helper matches dense score + top-k on tiny CPU examples
2. tie handling is deterministic and follows the chosen rule
3. block size changes do not change final top-k result
4. `topk=1` and `topk=T` edge cases match dense reference

### Hard tests

1. CUDA parity between dense and streaming indexer on small square image grids
2. bf16 coverage
3. adversarial repeated-score cases
4. multi-head indexer cases with `index_n_heads > 1`

### Extra-hard tests

1. regression test that `256x256` indexer build no longer OOMs where dense path does
2. end-to-end DSA path preserves selector outputs when switching `dense -> streaming`
3. benchmark schema captures which indexer mode was used

## Validation Gates

Do not move past this step until all of these are true:

1. dense and streaming indexers match exactly on the locked test matrix
2. full DSA test suite remains green
3. A100 long-sequence benchmark proves the streaming indexer removes the first OOM source
4. `docs/dsa_memory.md` is updated with the new mode, tests, and measured outcomes

## Files In Scope

Primary implementation files:

1. [src/ops/dsa_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/ops/dsa_indexer.py)
2. [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
3. [tests/helpers/dsa_reference.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/helpers/dsa_reference.py)
4. [tests/test_dsa_2d_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_indexer.py)
5. [tests/test_dsa_2d_regression.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_regression.py)
6. [artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py)
7. [docs/dsa_memory.md](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/docs/dsa_memory.md)

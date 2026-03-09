# DSA 2D MLA Design

**Date**: 2026-03-10
**Branch**: `nsa-triton-longseq-investigation`
**Status**: Design

## Goal

Add a standalone `DSA2DMLAAttention` module that matches DeepSeek DSA as closely as practical while adapting only the task geometry:

1. input is `x[B, C, H, W]`
2. positions are `2D (row, col)` instead of `1D`
3. attention is fully non-causal

Everything else should stay DeepSeek-style:

1. MLA attention core
2. MQA/GQA head sharing
3. separate lightning indexer
4. partial RoPE
5. `FWHT -> FP8 -> weighted ReLU logits -> topk`
6. sparse attention over selected tokens
7. dense warm-up plus sparse training with a dedicated indexer loss

## Source Of Truth

Primary sources:

1. DeepSeek DSA paper: <https://arxiv.org/pdf/2512.02556>
2. DeepSeek architecture and inference reference: <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp>
3. DeepSeek inference model file: <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py>
4. DeepSeek inference kernels: <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py>
5. FlashMLA: <https://github.com/deepseek-ai/FlashMLA>
6. DeepGEMM: <https://github.com/deepseek-ai/DeepGEMM>
7. TileLang DSA training operator: <https://github.com/lemyx/tilelang-dsa>
8. fast-hadamard-transform: <https://github.com/Dao-AILab/fast-hadamard-transform>

Secondary Chinese references for cross-checking terminology and training summaries:

1. <https://blog.csdn.net/jiemo99/article/details/153729664>
2. <https://blog.csdn.net/2401_85325557/article/details/152787564>
3. <https://blog.csdn.net/weixin_52610848/article/details/152378498>
4. <https://blog.csdn.net/2401_84204207/article/details/155490937>
5. <https://blog.csdn.net/gitblog_00400/article/details/154101623>

Use policy:

1. the paper and official DeepSeek code are authoritative
2. Chinese links are reference only and must never override the official implementation

## Non-Negotiable Constraints

1. `DSA2DMLAAttention` is a separate module from `NSA2DAttention`
2. the first implementation is MLA-backed from day one; there is no interim plain `Q/K/V` version
3. the module is fully non-causal
4. the selector is global token-level over `T = H * W`
5. the first version supports `MQA/GQA`; full `MHA` is out of scope
6. the indexer is a separate trainable path with its own loss
7. validation must follow strict TDD with easy, hard, and extra-hard gates before advancing to the next subsystem

## High-Level Architecture

Add a new standalone module family under `src/networks/dsa_2d.py`:

1. `DSA2DMLAConfig`
2. `DSA2DIndexer`
3. `DSA2DMLAAttention`

Recommended operator split under `src/ops/`:

1. `dsa_rope.py` for 2D partial RoPE helpers
2. `dsa_indexer.py` for FWHT, FP8 quant helpers, and weighted-ReLU index scoring
3. `dsa_sparse_mla.py` for token gather and sparse MLA attention routines

The forward path should be:

`x -> MLA projections + indexer projections -> partial 2D RoPE -> indexer FWHT -> FP8 indexer activations -> weighted ReLU token scores -> topk token indices -> sparse MLA attention over selected tokens -> proj`

## DeepSeek-Aligned MLA Structure

The main attention path should mirror DeepSeek MLA naming and decomposition as closely as practical.

### Query side

1. `wq_a`
2. `q_norm`
3. `wq_b`
4. split query head into:
   - `q_nope`
   - `q_pe`

### KV side

1. `wkv_a`
2. split into:
   - latent `kv`
   - `k_pe`
3. `kv_norm`
4. `wkv_b`
5. split expanded output into:
   - `k_nope`
   - `v`

### Final MLA head representation

1. `q = [q_nope | q_pe]`
2. `k = [k_nope | k_pe]`
3. `v` as produced by the MLA KV expansion

The head-sharing contract is:

1. `n_heads = number of query heads`
2. `n_kv_heads = number of KV heads`
3. `G = n_heads / n_kv_heads`
4. attention uses `MQA/GQA`, not independent per-head KV projections

## DeepSeek-Aligned Indexer Structure

The indexer must not reuse the current NSA scorer shape. It should follow DSA's separate lightning-indexer idea.

### Indexer path

1. separate query-side low-rank path
2. separate key-side path
3. separate query-dependent head-weight path
4. partial RoPE on indexer `Q/K`
5. `FWHT`
6. `FP8` activation quantization with explicit scales
7. weighted `ReLU(q dot k)` token score
8. `topk` token indices per query

### Indexer score

For query token `t` and key token `s`:

`I(t, s) = sum_j w_j(t) * ReLU(q_idx_j(t) dot k_idx_j(s))`

Important consequences:

1. the indexer does not use softmax to produce selection scores
2. the selector is token-level, not patch-level
3. the indexer is much cheaper than full MLA attention and is designed to be quantization-friendly

## Partial RoPE Rules

DeepSeek uses two different partial RoPE layouts. We should keep that distinction and only adapt positions from `1D` to `2D`.

### MLA RoPE

DeepSeek MLA uses:

1. `qk_nope_head_dim = 128`
2. `qk_rope_head_dim = 64`
3. total `qk_head_dim = 192`

Only the `64`-dim rope slice gets RoPE.

Layout:

1. interleaved RoPE
2. for 2D, split the rope slice into:
   - `32` row dims
   - `32` col dims

### Indexer RoPE

DeepSeek indexer uses:

1. `index_head_dim = 128`
2. rope-active slice `= 64`
3. non-rope slice `= 64`

Layout:

1. non-interleaved RoPE
2. for 2D, split the rope slice into:
   - `32` row dims
   - `32` col dims

### 2D adaptation rule

Use flattened row-major storage but 2D positions for RoPE:

1. `t = row * W + col`
2. `row_t = t // W`
3. `col_t = t % W`

Then apply:

1. row rotation on the row rope slice using `row_t`
2. col rotation on the col rope slice using `col_t`
3. leave the no-rope slice untouched
4. rotate `Q` and `K` only
5. never rotate `V`
6. never rotate the raw image tensor `x`

## Sparse Attention Semantics

The selector returns token indices, not patch indices.

Sparse MLA attention then:

1. gathers selected `K/V` tokens from the MLA path
2. computes exact attention only on those selected tokens
3. remains fully non-causal
4. uses row-major token flattening only as sequence storage, not as a replacement for 2D position encoding

This means the first correctness oracle is:

1. if `topk = T`, sparse MLA must reduce to dense MLA within tolerance

That gate is mandatory before any runtime benchmarking matters.

## Training Plan

The training plan should stay faithful to the paper, adapted to non-causal images.

### Stage 1: Dense warm-up

1. run dense non-causal MLA teacher attention over all `T = H * W` tokens
2. freeze the main model parameters
3. train only the indexer
4. for each query token `t`, sum dense MLA attention scores across main heads and normalize to a teacher distribution `p_t`
5. train the indexer with KL divergence between the teacher distribution and the indexer distribution

Important rule from the paper:

1. indexer inputs are detached from the main graph during indexer training
2. the indexer gets its own loss and should not implicitly rely on the downstream task loss to learn selection

### Stage 2: Sparse training

1. enable token-level top-k selection
2. main model trains with the normal image task loss
3. indexer continues training with the KL alignment loss
4. dense MLA teacher remains available during training-time supervision only

The practical implication is:

1. the indexer learns which pixels matter for each query pixel by imitating dense MLA attention, not by guessing directly from labels

## Reuse Plan

### Reuse for architecture truth

Use these files as the architectural source of truth:

1. DeepSeek `model.py`: MLA decomposition, indexer split, and RoPE layout rules
2. DSA paper: algorithm, training stages, and loss design

### Reuse for correctness-first implementation

1. `fast-hadamard-transform` for working FWHT behavior before any custom optimization
2. DeepSeek `kernel.py` as the reference for:
   - activation quantization
   - `rotate_activation`
   - weighted-ReLU index scoring
3. TileLang DSA as a reference for warm-up KL alignment and training-side operator decomposition

### Reuse later for performance backends

1. FlashMLA as the future sparse MLA execution backend, especially on Hopper-class hardware
2. DeepGEMM as the future high-performance indexer backend for lightning-indexer style kernels

### Reuse boundaries

Do not blindly copy these into the first training bring-up:

1. official DeepSeek sparse prefill runtime as the final performance path
2. any Chinese secondary source as implementation authority

## Validation And Testing Strategy

Implementation must be strictly test-driven. No production code for a subsystem is allowed before the failing tests for that subsystem exist and are observed to fail correctly.

### Test tiers

Every subsystem gets three tiers:

1. easy tests: small deterministic unit tests
2. hard tests: CUDA parity and gradient tests on realistic small shapes
3. extra-hard tests: adversarial numeric, indexing, and end-to-end equivalence gates

### Subsystem 1: 2D partial RoPE

Easy:

1. only `Q/K` rotate
2. `V` is unchanged
3. no-rope slice is unchanged
4. row and col slices rotate independently
5. indexer uses non-interleaved layout
6. MLA uses interleaved layout

Hard:

1. parity to a naive Python/Torch reference on multiple shapes
2. parity across dtypes `fp32` and `bf16`
3. parity for both square and small asymmetric spatial layouts if later enabled

Extra hard:

1. hand-constructed spot checks at exact `(row, col)` positions
2. permutation/indexing checks proving row-major flattening maps back to the right coordinates

### Subsystem 2: MLA decomposition

Easy:

1. exact split sizes for `q_nope`, `q_pe`, `k_nope`, `k_pe`, `v`
2. exact head mapping for `MQA/GQA`
3. flatten/unflatten round-trips `B, C, H, W <-> B, h, T, d`

Hard:

1. dense MLA forward parity against a pure reference implementation
2. backward parity on input and projection grads
3. mixed precision checks

Extra hard:

1. dense vs sparse equivalence with `topk = T`
2. adversarial head/group configurations near edge constraints

### Subsystem 3: Indexer

Easy:

1. FWHT parity to a naive reference
2. FP8 quant/dequant helper sanity with explicit scales
3. weighted `ReLU(q dot k)` score parity to a naive reference
4. top-k correctness on handcrafted toy cases

Hard:

1. end-to-end indexer score parity on realistic small tensors
2. parity between dequant reference path and optimized path
3. gradient-flow checks on the trainable indexer path

Extra hard:

1. all-negative pre-ReLU products
2. repeated-value ties in top-k
3. extreme magnitudes for quantization scales
4. `topk = 1` and `topk = T`

### Subsystem 4: Sparse MLA attention

Easy:

1. gather order correctness from token indices
2. selected-token attention parity to a naive sparse gather-attend reference

Hard:

1. dense MLA vs sparse MLA forward parity when `topk = T`
2. dense MLA vs sparse MLA backward parity when `topk = T`
3. parity at representative shapes and `G` values

Extra hard:

1. exact indexing tests with synthetic single-dominant tokens
2. regression tests around repeated indices, sorted and unsorted top-k, and degenerate token sets

### Subsystem 5: Training losses

Easy:

1. teacher attention distribution sums to `1`
2. KL target normalization is correct
3. indexer detach behavior is enforced

Hard:

1. dense warm-up lowers KL on a tiny synthetic problem
2. sparse-stage KL stays finite while task loss backpropagates through the main model

Extra hard:

1. indexer-only warm-up must not update frozen main-model parameters
2. sparse-stage regression on tiny image task where selection remains stable over several steps

### Step gates

The implementation stops at each of these mandatory gates:

1. RoPE gate
2. MLA gate
3. indexer gate
4. sparse-MLA gate
5. training gate
6. runtime gate

Runtime benchmarking is forbidden until the first five correctness gates are green.

## Benchmarks

Only after correctness gates pass, add DSA diagnostics under `artifacts/dsa_diagnostics/`.

Required early benchmarks:

1. dense MLA vs sparse DSA-MLA with `topk = T`
2. sparse runtime at `128x128`
3. sparse runtime at `256x256`
4. comparisons against the current NSA module on matched `C`, heads, and selected-token budgets

## Risks

1. indexer correctness can drift silently if RoPE layout, FWHT order, or scale handling deviates from DeepSeek
2. the first correctness-first FP8 path may be slower than needed; that is acceptable initially
3. sparse attention kernel work should not begin until the dense-equivalence gate is met
4. image-task quality may require tuning indexer head count and top-k, but that comes after paper-faithful correctness

## Non-Goals For The First Bring-Up

1. replacing current NSA
2. outperforming FlashMLA/DeepGEMM immediately on A100
3. supporting causal mode
4. supporting full independent MHA KV heads
5. introducing image-specific hierarchical shortcuts before paper-faithful DSA works

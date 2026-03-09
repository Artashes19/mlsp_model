# DSA Memory

Purpose:

- persistent memory for DeepSeek-aligned DSA runtime and training work
- record exact architectural decisions, invariants, sources, testing gates, and benchmark outcomes
- update this file after every DSA experiment or implementation that materially changes our understanding

## Locked Invariants

1. DSA is a separate module family from NSA.
2. The target module is `DSA2DMLAAttention`, not an interim plain `Q/K/V` sparse attention module.
3. The module is fully non-causal.
4. Selection is global token-level over `T = H * W` tokens.
5. The first supported head mode is `MQA/GQA`; full independent `MHA` is out of scope.
6. The indexer is a separate trainable path with its own loss.
7. The first implementation should stay as close as practical to the DSA paper and official DeepSeek inference code.

## Intended Deviations From DeepSeek

These are the only planned semantic deviations:

1. input is `x[B, C, H, W]` instead of 1D text embeddings
2. positions are `2D (row, col)` instead of `1D`
3. attention is non-causal instead of causal

Everything else should stay DeepSeek-like unless measurement or correctness forces a change.

## Source Of Truth

Primary sources:

1. DSA paper: <https://arxiv.org/pdf/2512.02556>
2. DeepSeek architecture repo: <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp>
3. DeepSeek `inference/model.py`: <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py>
4. DeepSeek `inference/kernel.py`: <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/kernel.py>
5. FlashMLA: <https://github.com/deepseek-ai/FlashMLA>
6. DeepGEMM: <https://github.com/deepseek-ai/DeepGEMM>
7. TileLang DSA: <https://github.com/lemyx/tilelang-dsa>
8. fast-hadamard-transform: <https://github.com/Dao-AILab/fast-hadamard-transform>

Chinese reference links:

1. <https://blog.csdn.net/jiemo99/article/details/153729664>
2. <https://blog.csdn.net/2401_85325557/article/details/152787564>
3. <https://blog.csdn.net/weixin_52610848/article/details/152378498>
4. <https://blog.csdn.net/2401_84204207/article/details/155490937>
5. <https://blog.csdn.net/gitblog_00400/article/details/154101623>

Authority rule:

1. the paper and official DeepSeek code override all secondary summaries
2. Chinese links are for reference only

## Key DeepSeek Alignment Decisions

### MLA

Use MLA from the start.

1. keep DeepSeek-style query decomposition (`wq_a -> q_norm -> wq_b`)
2. keep DeepSeek-style KV decomposition (`wkv_a -> kv_norm -> wkv_b`)
3. keep partial RoPE split on the main MLA path
4. keep `MQA/GQA`

### Indexer

Keep the lightning indexer separate from the MLA attention payload.

1. separate indexer query path
2. separate indexer key path
3. query-dependent head weights
4. weighted `ReLU(q dot k)` scoring
5. token-level `topk`
6. no softmax-based scorer in the selector itself

### Partial RoPE

DeepSeek uses two different RoPE layouts.

1. MLA RoPE is partial and interleaved.
2. Indexer RoPE is partial and non-interleaved.
3. Only `Q/K` rotate; `V` does not.
4. The raw image tensor `x` is never rotary-encoded directly.

### 2D position rule

1. tokens are flattened row-major for storage
2. RoPE uses `row` and `col`, not the flat token id directly
3. if `t = row * W + col`, then:
   - `row_t = t // W`
   - `col_t = t % W`
4. split the rope-active slice into row and col halves and rotate them independently

## Training Plan

### Stage 1: Dense warm-up

1. run dense non-causal MLA teacher attention
2. freeze the main model
3. train only the indexer
4. target is dense MLA attention distribution summed across heads and normalized
5. optimize indexer with KL divergence
6. detach indexer input from the main graph

### Stage 2: Sparse training

1. enable token-level top-k selection
2. main model trains on the image task loss
3. indexer keeps training with KL alignment to the dense MLA teacher
4. dense teacher is training-only and must not become a hidden runtime dependency

## Testing Policy

Strict TDD is mandatory.

1. no production code without a failing test first
2. every subsystem gets easy, hard, and extra-hard tests
3. do not advance to the next subsystem until the current test gate is green
4. do not claim runtime success before dense-equivalence and backward parity gates pass

## Current Test Gates

1. RoPE gate
- exact 2D partial-RoPE parity to naive reference
- interleaved vs non-interleaved distinction proven

2. MLA gate
- dense MLA forward/backward parity to reference implementation

3. Indexer gate
- FWHT parity
- FP8 helper sanity
- weighted-ReLU score parity
- top-k correctness under ties and adversarial values

4. Sparse MLA gate
- sparse gather correctness
- dense vs sparse equivalence at `topk = T`
- backward parity at `topk = T`

5. Training gate
- teacher distribution normalization
- KL loss sanity
- warm-up detach contract
- tiny warm-up decreases KL

6. Runtime gate
- benchmark only after the previous five gates are green

## Planned File Layout

1. `src/networks/dsa_2d.py`
2. `src/ops/dsa_rope.py`
3. `src/ops/dsa_indexer.py`
4. `src/ops/dsa_sparse_mla.py`
5. `tests/helpers/dsa_reference.py`
6. `tests/helpers/fp8_reference.py`
7. `tests/test_dsa_2d_rope.py`
8. `tests/test_dsa_2d_mla.py`
9. `tests/test_dsa_2d_indexer.py`
10. `tests/test_dsa_2d_sparse_attention.py`
11. `tests/test_dsa_2d_training.py`
12. `tests/test_dsa_2d_regression.py`
13. `artifacts/dsa_diagnostics/`

## Immediate Next Steps

1. create the DSA helper scaffolding and test files
2. implement 2D partial RoPE first under strict TDD
3. implement dense MLA reference path before any sparse kernel path
4. implement the correctness-first indexer path
5. unlock sparse MLA only after dense-equivalence tests exist

## Implemented Gates

### 2026-03-10: Dense MLA reference gate

Status:

1. complete

What landed:

1. `DSA2DMLAAttention.forward_dense_reference(x)` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
2. dense MLA reference helpers in [tests/helpers/dsa_reference.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/helpers/dsa_reference.py)
3. forward and backward parity tests in [tests/test_dsa_2d_mla.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_mla.py)

Locked behavior:

1. dense MLA path uses current DeepSeek-style split contracts:
   - `wq_a -> q_norm -> wq_b`
   - `wkv_a -> [latent_kv | k_pe]`
   - `kv_norm -> wkv_b -> [k_nope | v]`
2. MLA RoPE is applied only to the `q_pe` / `k_pe` slices with the existing interleaved 2D helper
3. `k_pe` is shared across KV heads in the current MLA reference path and then broadcast to `n_kv_heads`
4. dense attention is fully non-causal and expands KV heads to query heads through `G = n_heads / n_kv_heads`

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_mla.py -k "dense_mla" -v`
   - result: `2 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `16 passed`

Notes:

1. `forward()` remains intentionally unimplemented; only `forward_dense_reference()` is live at this gate
2. PyTorch emitted a CUDA initialization warning during the CPU-side backward test run, but all assertions passed

### 2026-03-10: Basic indexer math gate

Status:

1. complete

What landed:

1. correctness-first `fwht_last_dim()` in [src/ops/dsa_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/ops/dsa_indexer.py)
2. correctness-first `weighted_relu_index_score()` in [src/ops/dsa_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/ops/dsa_indexer.py)
3. naive FWHT and weighted-ReLU references in [tests/helpers/fp8_reference.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/helpers/fp8_reference.py)
4. basic indexer parity tests in [tests/test_dsa_2d_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_indexer.py)

Locked behavior:

1. FWHT currently requires a power-of-two last dimension
2. weighted-ReLU index score matches the DSA formula shape:
   - `q[B, heads, T, D]`
   - `k[B, heads, S, D]`
   - `w[B, T, heads]`
   - output `scores[B, T, S]`
3. both helpers are correctness-first and operate via float32 accumulation internally

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "fwht or weighted_relu" -v`
   - result: `2 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `18 passed`

### 2026-03-10: Correctness-first FP8 indexer gate

Status:

1. complete

What landed:

1. `act_quant_reference_safe()` in [src/ops/dsa_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/ops/dsa_indexer.py)
2. `stable_topk()` in [src/ops/dsa_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/ops/dsa_indexer.py)
3. `DSA2DIndexer` skeleton in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
4. FP8 quant reference helper in [tests/helpers/fp8_reference.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/helpers/fp8_reference.py)
5. FP8/top-k/indexer shape tests in [tests/test_dsa_2d_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_indexer.py)

Locked behavior:

1. correctness-first FP8 quant path uses real `torch.float8_e4m3fn`
2. scales are computed per last-dimension vector and kept as explicit float32 tensors
3. `stable_topk()` is deterministic for ties via stable descending sort and currently prefers lower indices first
4. `DSA2DIndexer.forward(q, k, w)` currently:
   - quantizes `q` and `k`
   - dequantizes them via explicit scales
   - computes weighted-ReLU logits
   - returns `(logits, topk_indices)`

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -v`
   - result: `7 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `21 passed`

### 2026-03-10: Sparse gather and topk=T equivalence gate

Status:

1. complete

What landed:

1. `gather_sparse_mla_tokens()` in [src/ops/dsa_sparse_mla.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/ops/dsa_sparse_mla.py)
2. `gather_tokens_reference()` in [tests/helpers/dsa_reference.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/helpers/dsa_reference.py)
3. `DSA2DMLAAttention.forward_sparse_with_forced_topk(..., topk_equals_t=True)` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
4. sparse gather and `topk=T` equivalence tests in [tests/test_dsa_2d_sparse_attention.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_sparse_attention.py)

Locked behavior:

1. sparse token gather currently expects:
   - `tokens[B, heads, T, D]`
   - `idx[B, Q, K]`
   - output `[B, heads, Q, K, D]`
2. the only sparse-forward path implemented at this stage is the correctness gate where every query selects all `T` tokens in row-major order
3. in that forced regime, sparse MLA output must equal `forward_dense_reference()` within test tolerance

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py -v`
   - result: `3 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `23 passed`

## Update Policy

After every meaningful DSA change, update this file with:

1. new invariants or changed assumptions
2. artifact paths
3. parity outcomes
4. benchmark outcomes
5. any divergence from official DeepSeek behavior and the reason for it

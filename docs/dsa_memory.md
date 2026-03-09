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

## Update Policy

After every meaningful DSA change, update this file with:

1. new invariants or changed assumptions
2. artifact paths
3. parity outcomes
4. benchmark outcomes
5. any divergence from official DeepSeek behavior and the reason for it

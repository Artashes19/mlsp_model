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

### 2026-03-10: Indexer RoPE and FWHT preprocessing gate

Status:

1. complete

What landed:

1. `DSA2DMLAConfig` now rejects non-power-of-two `index_head_dim` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
2. `build_indexer_logits()` now applies indexer-style non-interleaved 2D partial RoPE before FWHT in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
3. reference indexer preprocessing helper landed in [tests/helpers/dsa_reference.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/helpers/dsa_reference.py)
4. new tests landed in:
   - [tests/test_dsa_2d_mla.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_mla.py)
   - [tests/test_dsa_2d_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_indexer.py)

Locked behavior:

1. the current correctness-first indexer path is now:
   - project `q/k/w`
   - apply non-interleaved 2D partial RoPE to `q/k`
   - apply FWHT to the full indexer head
   - FP8 quantize
   - weighted-ReLU score
   - stable top-k
2. `index_head_dim` must be a power of two because the correctness-first FWHT implementation requires it
3. indexer preprocessing parity is checked against a naive reference before integration into sparse forward

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_mla.py -k "non_power_of_two_index_head_dim" -v`
   - result: `1 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "reference_preprocessing_path" -v`
   - result: `1 passed`
3. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `35 passed, 1 warning`

### 2026-03-10: DeepSeek-style indexer projection contract gate

Status:

1. complete

What landed:

1. the indexer query path now reuses MLA query latent activations via `wq_a -> q_norm -> index_wq_b` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
2. the indexer key path now uses a shared-key projection plus normalization via `index_wk -> index_k_norm` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
3. the indexer weight path now uses `index_weights_proj` with static scaling by `index_n_heads**-0.5 * index_head_dim**-0.5`
4. the reference path in [tests/helpers/dsa_reference.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/helpers/dsa_reference.py) now matches this DeepSeek-style contract
5. new shape and reference-path tests landed in:
   - [tests/test_dsa_2d_mla.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_mla.py)
   - [tests/test_dsa_2d_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_indexer.py)

Locked behavior:

1. indexer `q` is no longer built from a direct token projection; it now comes from the MLA query latent path
2. indexer `k` is shared across indexer heads and expanded only after normalized projection, RoPE, and FWHT
3. current correctness-first score path remains:
   - DeepSeek-style query/key/weight projections
   - non-interleaved 2D partial RoPE
   - FWHT
   - FP8 quantization
   - weighted-ReLU score
   - stable top-k

### 2026-03-10: Warm-up isolation and benchmark trustworthiness gate

Status:

1. complete

What landed:

1. `build_indexer_logits(..., detach_inputs=True)` now detaches the shared MLA query latent before `index_wq_b` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
2. `indexer_alignment_kl_loss()` now averages per-query KL instead of scaling with sequence length in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
3. DSA benchmark timing now runs under `torch.inference_mode()` and puts compared modules in eval mode in [artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py)
4. reference helper parity for detached warm-up path was updated in [tests/helpers/dsa_reference.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/helpers/dsa_reference.py)
5. regression tests landed in:
   - [tests/test_dsa_2d_training.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_training.py)
   - [tests/test_dsa_2d_regression.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_regression.py)

Locked behavior:

1. dense warm-up no longer backpropagates through `wq_a` or `q_norm`; only indexer parameters are supposed to receive gradients from the KL path
2. the alignment KL is invariant to duplicating the same per-query distribution across more query tokens
3. forward benchmark numbers are now inference-only timings rather than forward-with-autograd timings

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_training.py -k "shared_query_path_params or averaged_per_query" -v`
   - result: `2 passed, 1 warning`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_regression.py -k "grad_disabled" -v`
   - result: `1 passed`
3. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `42 passed, 1 warning`

### 2026-03-10: Corrected A100 DSA four-way benchmark rerun

Status:

1. complete

Why it was rerun:

1. the earlier DSA benchmark harness was timing forward with autograd enabled
2. after the fix, benchmark timing now runs under `torch.inference_mode()` with modules in eval mode
3. earlier forward numbers should therefore be treated as stale

Command:

1. `ssh artashes@dgx.yc2.io "cd /auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation && CUDA_VISIBLE_DEVICES=7 /auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py --device cuda --dtype bfloat16 --warmup 1 --iters 1"`

Artifact:

1. [artifacts/dsa_diagnostics/dsa_benchmark_cuda_bfloat16.json](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/artifacts/dsa_diagnostics/dsa_benchmark_cuda_bfloat16.json)

Updated A100 bf16 results:

1. `128x128, C=384, heads=6, n_kv_heads=2, topk=256`
   - dense MLA: `14.33 ms`
   - DSA sparse: `80.22 ms`
   - NSA: `10.62 ms`
   - Flash MHA: `3.37 ms`
2. `256x256, C=384, heads=6, n_kv_heads=2, topk=256`
   - dense MLA: `oom`
   - DSA sparse: `oom`
   - NSA: `71.33 ms`
   - Flash MHA: `38.57 ms`
3. `128x128, C=512, heads=8, n_kv_heads=2, topk=256`
   - dense MLA: `15.58 ms`
   - DSA sparse: `84.43 ms`
   - NSA: `11.12 ms`
   - Flash MHA: `4.25 ms`
4. `256x256, C=512, heads=8, n_kv_heads=2, topk=256`
   - dense MLA: `oom`
   - DSA sparse: `oom`
   - NSA: `81.74 ms`
   - Flash MHA: `54.69 ms`

Locked takeaway:

1. correcting the harness did not change the qualitative conclusion
2. current correctness-first DSA is still far slower than NSA and dense baselines at `128x128`
3. current DSA and dense MLA still OOM at `256x256`
4. the next DSA runtime work still has to remove dense `T x T` score materialization and explicit gathered sparse `K/V` materialization

### 2026-03-10: Streaming indexer runtime contract split

Status:

1. complete

What landed:

1. `build_indexer_logits(...)` remains the dense training path in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
2. runtime dispatch moved to `build_indexer_selection(...)` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
3. `forward(...)` now consumes `build_indexer_selection(...)`, not `build_indexer_logits(...)`
4. `indexer_mode` was added with conservative default `dense`
5. in both runtime modes, `build_indexer_selection(...)` now returns top-k scores and top-k indices with aligned semantics

Locked behavior:

1. dense warm-up and KL supervision continue to see full dense logits
2. runtime selection is allowed to use exact streaming top-k without changing the training API
3. runtime callers should treat `build_indexer_selection(...)` as a top-k interface, not a dense-logit interface
4. default behavior stays conservative until streaming is benchmarked on A100

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "runtime_selection_dense_uses_dense_path or runtime_selection_dense_and_streaming_match_indices or build_indexer_logits_uses_dense_path_even_in_streaming_mode or runtime_selection_streaming_uses_streaming_helper or forward_uses_runtime_selection_path" -v`
   - result: `5 passed, 15 deselected`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `54 passed, 1 warning`

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_mla.py -k "indexer_projection_shapes_follow_deepseek_style_contract" -v`
   - result: `1 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "reference_preprocessing_path" -v`
   - result: `1 passed`
3. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `36 passed, 1 warning`

### 2026-03-10: RMSNorm alignment gate

Status:

1. complete

What landed:

1. `q_norm`, `kv_norm`, and `index_k_norm` now use `torch.nn.RMSNorm` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
2. norm-type coverage landed in [tests/test_dsa_2d_mla.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_mla.py)

Locked behavior:

1. MLA query latent normalization is now RMSNorm, matching DeepSeek more closely than the previous LayerNorm placeholder
2. MLA KV latent normalization is now RMSNorm
3. indexer key normalization is now RMSNorm

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_mla.py -k "norms_use_rmsnorm" -v`
   - result: `1 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `37 passed, 1 warning`

### 2026-03-10: DSA benchmark harness expansion and A100 baseline

Status:

1. complete

What landed:

1. [bench_dsa_2d_vs_dense_mla.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py) now compares four paths on the same `x[B,C,H,W]` contract:
   - `DSA2DMLAAttention`
   - dense MLA reference
   - current NSA module
   - dense MHA via [EfficientGlobalAttention](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/txunet.py)
2. the harness now records structured per-baseline statuses (`ok` or `oom`) instead of crashing on long-sequence dense baselines
3. smoke/schema coverage expanded in [tests/test_dsa_2d_regression.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_regression.py)

Mixed-precision bug fixed during this work:

1. `build_indexer_logits()` no longer feeds float32 activations into bf16 indexer weights
2. the indexer weight projection now runs in module dtype and is cast to float32 only after projection
3. bf16 regression coverage added in [tests/test_dsa_2d_indexer.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_indexer.py)

Artifacts:

1. smoke benchmark:
   - [dsa_benchmark_smoke.json](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/artifacts/dsa_diagnostics/dsa_benchmark_smoke.json)
2. A100 bf16 suite:
   - [dsa_benchmark_cuda_bfloat16.json](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/artifacts/dsa_diagnostics/dsa_benchmark_cuda_bfloat16.json)
3. A100 DSA hotspot profile:
   - [dsa_profile_a100_128x128_c384_h6_g3_topk256.txt](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/artifacts/dsa_diagnostics/dsa_profile_a100_128x128_c384_h6_g3_topk256.txt)

Observed A100 bf16 results (`GPU 0`, `warmup=1`, `iters=1`):

1. `128x128, C=384, heads=6, G=3, topk=256`
   - dense MLA: `14.82 ms`
   - DSA sparse: `88.38 ms`
   - NSA: `10.68 ms`
   - Flash MHA: `3.29 ms`
2. `256x256, C=384, heads=6, G=3, topk=256`
   - dense MLA: `oom`
   - DSA sparse: `oom`
   - NSA: `70.82 ms`
   - Flash MHA: `38.74 ms`
3. `128x128, C=512, heads=8, G=4, topk=256`
   - dense MLA: `16.19 ms`
   - DSA sparse: `85.43 ms`
   - NSA: `11.23 ms`
   - Flash MHA: `4.19 ms`
4. `256x256, C=512, heads=8, G=4, topk=256`
   - dense MLA: `oom`
   - DSA sparse: `oom`
   - NSA: `81.50 ms`
   - Flash MHA: `52.14 ms`

Profiler signal for current DSA at `128x128, C=384, heads=6, G=3, topk=256`:

1. dominant work is not MLA math itself
2. top CUDA consumers are:
   - segmented `sort/topk`
   - `gather`
   - `bmm`
3. current DSA implementation is therefore bottlenecked by selector/index movement and materialization, not by a tuned sparse MLA kernel

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_regression.py -k "benchmark_harness_smoke or marks_oom_result" -v`
   - result: `2 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_indexer.py -k "supports_bfloat16_module_dtype" -v`
   - result: `1 passed`
3. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `39 passed, 1 warning`

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

### 2026-03-10: Sparse backward and regression gate

Status:

1. complete

What landed:

1. `DSA2DMLAAttention.forward_sparse_from_indices()` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
2. sparse backward parity helper in [tests/helpers/dsa_reference.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/helpers/dsa_reference.py)
3. repeated/unsorted index regression helper in [tests/helpers/dsa_reference.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/helpers/dsa_reference.py)
4. sparse backward and regression tests in:
   - [tests/test_dsa_2d_sparse_attention.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_sparse_attention.py)
   - [tests/test_dsa_2d_regression.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_regression.py)

Locked behavior:

1. `forward_sparse_from_indices(x, idx)` is the first general sparse-by-indices MLA reference path
2. sparse backward matches dense backward when `idx` enumerates all `T` tokens for every query
3. repeated and unsorted indices are currently supported in the sparse gather/reference path
4. sparse gather now rejects out-of-range indices explicitly

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_regression.py -v`
   - result: `6 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `25 passed`

### 2026-03-10: Dense-teacher and KL helper gate

Status:

1. complete

What landed:

1. `build_dense_teacher_distribution()` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
2. `prepare_warmup_indexer_inputs()` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
3. `assert_warmup_detach_contract()` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
4. `indexer_alignment_kl_loss()` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
5. teacher/detach/KL tests in [tests/test_dsa_2d_training.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_training.py)

Locked behavior:

1. dense teacher distribution is built from dense MLA attention scores, summed across heads and renormalized over keys
2. current warm-up detach plumbing explicitly detaches flattened token inputs before they are handed to the future indexer path
3. indexer alignment loss is currently KL divergence between `log_softmax(indexer_logits)` and normalized teacher probabilities

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_training.py -v`
   - result: `4 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `28 passed`

### 2026-03-10: Warm-up behavior gate

Status:

1. complete

What landed:

1. trainable indexer projections in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py):
   - `index_q_proj`
   - `index_k_proj`
   - `index_w_proj`
2. `build_indexer_logits(..., detach_inputs=...)` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
3. tiny warm-up helpers in [tests/helpers/dsa_reference.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/helpers/dsa_reference.py)
4. warm-up behavior tests in [tests/test_dsa_2d_training.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_training.py)

Locked behavior:

1. the first warm-up loop now optimizes trainable indexer projection weights, not the main MLA path
2. warm-up builds dense teacher probabilities from the dense MLA reference path
3. indexer logits are built from detached flattened tokens, so gradients reach indexer projection weights but not the input image tensor or frozen main-model parameters
4. `DSA2DIndexer.forward()` now caps runtime `topk` to the available sequence length for tiny regression/training cases

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_training.py -v`
   - result: `6 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `30 passed`

### 2026-03-10: Full module integration gate

Status:

1. complete

What landed:

1. integrated `DSA2DMLAAttention.forward(x)` in [src/networks/dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py)
2. forward integration tests in [tests/test_dsa_2d_mla.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_mla.py)

Locked behavior:

1. the standalone module now runs end-to-end:
   - build indexer logits/indices
   - use current sparse token selection
   - execute sparse MLA attention
   - return `[B, C, H, W]`
2. the integrated forward currently uses detached indexer inputs, consistent with the DSA warm-up/sparse-stage separation we locked earlier
3. dense and sparse paths share the same output projection contract

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_mla.py -k "round_trips_image_shape or projection_contract" -v`
   - result: `2 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_rope.py tests/test_dsa_2d_mla.py tests/test_dsa_2d_indexer.py tests/test_dsa_2d_sparse_attention.py tests/test_dsa_2d_training.py tests/test_dsa_2d_regression.py -v`
   - result: `32 passed`

### 2026-03-10: Diagnostics harness gate

Status:

1. complete

What landed:

1. benchmark smoke harness in [bench_dsa_2d_vs_dense_mla.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py)
2. profile smoke harness in [profile_dsa_2d_module.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/artifacts/dsa_diagnostics/profile_dsa_2d_module.py)
3. benchmark smoke regression test in [tests/test_dsa_2d_regression.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/tests/test_dsa_2d_regression.py)

Artifacts:

1. benchmark smoke JSON:
   - [dsa_benchmark_smoke.json](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/artifacts/dsa_diagnostics/dsa_benchmark_smoke.json)
2. profile smoke summary:
   - [dsa_profile_smoke.txt](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/artifacts/dsa_diagnostics/dsa_profile_smoke.txt)

Smoke numbers:

1. `smoke_4x4_topk_equals_t`
   - dense `3.77 ms`
   - sparse `3.71 ms`
   - integrated `3.16 ms`

Verification:

1. `/auto/home/artashes/miniconda3/envs/dev/bin/python -m pytest tests/test_dsa_2d_regression.py -k "benchmark_harness_smoke" -v`
   - result: `1 passed`
2. `/auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/dsa_diagnostics/bench_dsa_2d_vs_dense_mla.py --smoke --output-dir artifacts/dsa_diagnostics`
   - result: JSON artifact written
3. `/auto/home/artashes/miniconda3/envs/dev/bin/python artifacts/dsa_diagnostics/profile_dsa_2d_module.py --output-dir artifacts/dsa_diagnostics`
   - result: profile summary artifact written

## Update Policy

After every meaningful DSA change, update this file with:

1. new invariants or changed assumptions
2. artifact paths
3. parity outcomes
4. benchmark outcomes
5. any divergence from official DeepSeek behavior and the reason for it

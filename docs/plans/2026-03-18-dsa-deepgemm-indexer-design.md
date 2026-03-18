# DSA DeepGEMM Indexer Design

## Goal

Add an H100-only, forward-only selector-logit backend for `DSA2DMLAAttention` using `DeepGEMM`, while keeping selector semantics and top-k behavior unchanged.

## Why This Slice

`FlashMLA` already removed the sparse-forward bottleneck for supported native MQA forward cases on H100. The next remaining speed path is the selector/indexer. The cleanest public reuse boundary is the indexer logit kernel, not a fused end-to-end selector.

## Public Reuse Boundary

Primary sources:
- `DeepGEMM`: <https://github.com/deepseek-ai/DeepGEMM>
- `DeepSeek-V3.2-Exp`: <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp>
- `DeepSeek-V3.2-Exp/inference/model.py`: <https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3.2-Exp/main/inference/model.py>

The public DeepSeek code indicates:
1. lightning-indexer logits are delegated to `DeepGEMM`
2. sparse attention is delegated to `FlashMLA`
3. the released indexer path computes logits first, then applies `topk`

That means the correct first selector-kernel slice is:
- replace only dense logit construction
- keep existing `topk` logic unchanged
- keep training semantics unchanged until parity is established

## Architecture

### 1. New backend switch

Extend `DSA2DMLAConfig` with:
- `indexer_backend: str = "auto"`

Allowed values:
- `auto`
- `reference`
- `deepgemm`

Interpretation:
- `reference`: current PyTorch indexer path
- `deepgemm`: try the H100 kernel path when supported, otherwise fall back cleanly
- `auto`: prefer `deepgemm` only for supported forward-only H100 MQA cases

### 2. New H100 adapter module

Add:
- `src/ops/dsa_deepgemm.py`

Responsibilities:
1. lazy import the relevant `DeepGEMM` selector/logit callable
2. expose a support check
3. expose a dense-logit wrapper that accepts prepared indexer tensors and returns dense logits

The wrapper should be narrow:
- input: preprocessed indexer `q, k, w`
- output: dense logits `[B, T, S]`
- no `topk` inside the wrapper
- no selector semantics change

### 3. Integration points

Keep existing selector structure in `src/networks/dsa_2d.py`:
1. `_prepare_indexer_qkw(...)`
2. `build_indexer_logits(...)`
3. `build_indexer_selection(...)`

Planned behavior:
- `build_indexer_logits(...)`
  - stays dense-output by contract
  - may later use `deepgemm` when safe
- `build_indexer_selection(...)`
  - on supported H100 forward-only MQA cases:
    - logits via `DeepGEMM`
    - top-k via current stable PyTorch logic
  - otherwise use the current reference path

This keeps training/warm-up APIs intact while accelerating runtime selection first.

## Support Rules

The first `DeepGEMM` selector slice should be gated at least as tightly as `FlashMLA`:
1. CUDA only
2. `sm90+`
3. forward-only / grad-disabled
4. `n_kv_heads == 1`
5. native indexer shape contract only

If any condition fails:
- fall back to `reference`

## Testing Strategy

### Easy

1. config validation accepts `auto|reference|deepgemm`
2. unknown `indexer_backend` is rejected
3. support check rejects CPU and non-MQA
4. `auto` dispatch uses `deepgemm` only when supported and grad is off
5. `auto` dispatch falls back to reference when grad is on

### Hard

1. small-case logits parity: `DeepGEMM` wrapper vs reference logits
2. selection parity: top-k indices from kernel logits vs reference logits
3. batch and sequence shape preservation

### Extra hard

1. H100 parity on a supported native selector case
2. H100 speed benchmark of selector logits and full selector stage
3. then rerun full DSA forward on H100 with:
   - `indexer_backend=auto`
   - `sparse_backend=auto`

## Success Criteria

1. local DSA suite remains green
2. H100 selector-logit parity is numerically close
3. H100 selector stage shows real speedup
4. full DSA forward on H100 improves beyond the current `FlashMLA`-only gain

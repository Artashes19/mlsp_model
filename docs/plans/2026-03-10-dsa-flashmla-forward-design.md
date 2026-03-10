# DSA FlashMLA Forward Design

**Date**: 2026-03-10
**Branch**: `nsa-triton-longseq-investigation`
**Status**: Design

## Goal

Add a first fast-kernel path for `DSA2DMLAAttention` by integrating `FlashMLA` sparse MLA forward on H100, while keeping the current DSA selector, MLA projections, 2D RoPE, and reference sparse MLA path intact.

This slice is intentionally narrow:

1. forward only
2. H100 / `SM90+` only
3. `MQA` only (`n_kv_heads == 1`)
4. reference fallback remains the default-safe path

## Source Of Truth

Primary sources:

1. DSA paper: <https://arxiv.org/pdf/2512.02556>
2. DeepSeek-V3.2-Exp: <https://github.com/deepseek-ai/DeepSeek-V3.2-Exp>
3. FlashMLA: <https://github.com/deepseek-ai/FlashMLA>
4. DeepGEMM: <https://github.com/deepseek-ai/DeepGEMM>

Important interpretation:

1. `FlashMLA` is the public sparse MLA kernel surface for Hopper / Blackwell
2. `DeepGEMM` is the public selector-logit kernel surface, but does not expose the full fused runtime top-k selector we need
3. therefore the first kernel-backed DSA slice should target sparse MLA forward, not the selector

## Why FlashMLA First

Our current DSA runtime handoff already matches the sparse MLA kernel problem more closely than the selector side:

1. selector side still needs dense logits for warm-up KL training
2. public DeepGEMM gives weighted-ReLU logits, not the whole runtime selector merge
3. sparse MLA side already has a clean runtime contract:
   - `q`
   - `k`
   - `v`
   - `idx`
   - `softmax_scale`
   - output `out`

So the best first reuse is:

1. keep the current selector/indexer
2. replace only sparse MLA forward

## Scope

In scope:

1. add a `FlashMLA` forward adapter for `DSA2DMLAAttention`
2. keep the current reference sparse MLA path as fallback
3. add backend dispatch and validation
4. add forward-only parity tests
5. add H100 Slurm benchmark harness or update the existing DSA harness

Out of scope:

1. backward kernel integration
2. selector / indexer kernel integration
3. dense warm-up training changes
4. GQA sparse-kernel support beyond `MQA`

## Current Swap Point

The clean sparse MLA runtime swap point is:

1. [forward_sparse_from_indices in dsa_2d.py](/auto/home/artashes/mlsp_model/dev-clean/.worktrees/nsa-triton-longseq-investigation/src/networks/dsa_2d.py#L195)

Current upstream contract:

1. `_dense_mla_qkv(...)` produces:
   - `q: [B, Hq, T, D_qk]`
   - `k: [B, Hkv, T, D_qk]`
   - `v: [B, Hkv, T, D_v]`
2. selector produces:
   - `idx: [B, T, K]`
3. sparse MLA returns:
   - `out: [B, Hq, T, D_v]`

The `FlashMLA` integration should replace only the lower-level sparse MLA compute under this interface.

## Target Backend Contract

Add an explicit sparse backend choice to `DSA2DMLAAttention`:

1. `sparse_backend = "reference" | "flashmla"`

Behavior:

1. `reference`
   - use current `streaming_sparse_mla_reference(...)`
2. `flashmla`
   - use a `FlashMLA` adapter when constraints pass
   - otherwise fall back to reference, or hard-error in strict mode if we add one later

The first version should be conservative and explicit, not magical.

## FlashMLA Adapter

Add a dedicated adapter module:

1. `src/ops/dsa_flashmla.py`

Responsibilities:

1. lazy-import `FlashMLA`
2. validate backend support:
   - CUDA only
   - `SM90+`
   - forward only
   - `n_kv_heads == 1`
3. reshape DSA tensors to the public `FlashMLA` sparse prefill API
4. call the kernel
5. reshape output back to the existing DSA contract

## First-Slice Constraints

The first kernel-backed path should be restricted to:

1. `MQA` only
   - `n_kv_heads == 1`
2. `SM90+` only
3. forward only
4. supported dtype/layout only

Why:

1. `FlashMLA` public support is Hopper / Blackwell
2. public sparse prefill support is oriented around `MQA`
3. this reduces the first integration surface and avoids mixing multiple unsupported assumptions

## Tensor Adaptation

Our DSA tensors:

1. `q: [B, Hq, T, D_qk]`
2. `k: [B, 1, T, D_qk]`
3. `v: [B, 1, T, D_v]`
4. `idx: [B, T, K]`

The adapter should flatten batch into sequence-major kernel inputs as needed by `FlashMLA`.

Practical first adapter rule:

1. flatten `(B, T)` into `s_q`
2. flatten `(B, T)` into `s_kv`
3. reshape indices to the per-query sparse-pre\-fill kernel format
4. preserve output semantics:
   - `out -> [B, Hq, T, D_v]`

The adapter must not build explicit `k_selected/v_selected` buffers.

## Fallback Rules

`flashmla` must not silently change semantics.

If constraints do not pass:

1. if `sparse_backend == "reference"`:
   - always use the reference path
2. if `sparse_backend == "flashmla"` and support fails:
   - explicitly fall back to reference for now
   - log or expose a clear reason in tests / diagnostics

This keeps the integration operational without hiding unsupported cases.

## Validation

Forward-only validation for the first slice:

1. backend dispatch tests
2. fallback tests for unsupported cases
3. small-case forward parity vs reference path
4. output shape / dtype preservation
5. H100 benchmark under Slurm

Important gate:

1. first compare `flashmla` forward vs current reference sparse MLA forward on the same `q/k/v/idx`
2. only after that compare end-to-end DSA module timings

## H100 Execution

All GPU-heavy runs on the H100 host must go through Slurm.

Environment and path:

1. host: `root@172.26.30.200`
2. env: `conda activate indoorolo`
3. worktree: `/home/indoor/mlsp_wair_d/.worktrees/nsa-triton-longseq-investigation`

Initial runtime target:

1. `128x128` first
2. then `256x256`
3. `MQA`
4. forward only

## Success Criteria

This slice is successful if:

1. `FlashMLA` forward runs on H100 in the DSA worktree
2. `flashmla` output is numerically close to the reference sparse MLA output
3. the `flashmla` path is materially faster than the current reference sparse MLA path
4. unsupported cases fall back cleanly to reference

## Expected Next Step After This Slice

If this succeeds, the next kernel integration target should be:

1. selector/indexer runtime path via `DeepGEMM` or another fused selector path

If this fails, the most likely causes are:

1. tensor contract mismatch with `FlashMLA`
2. public kernel support assumptions around `MQA` / dimension packing
3. layout or dtype expectations not yet matched by our adapter

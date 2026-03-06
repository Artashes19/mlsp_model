# NSA Memory

Purpose:

- persistent memory for NSA runtime work
- record important results, failures, invariants, and decisions
- update this file after every experiment or implementation that materially changes our understanding

## Locked Invariants

1. This NSA path is 2D and fully non-causal.
2. Per-query selection is mandatory.
3. Selection is per KV head; for plain MHA it is also per head.
4. Shared selection mode is removed and should not return.
5. `window_size = w` means `w x w` local tokens per query.
6. `selected tokens per query = top_n * p^2`.

## Current Focus

1. Improve the full NSA attention layer, not just isolated kernels.
2. Keep FFN and attention shell in scope, because attention-only wins do not guarantee block-level wins.
3. Use A100 for current development and H100 as the final target backend.

## Major Findings

### Old dK/dV bottleneck was real and is fixed

- Tilda-style compact dK/dV redesign removed the catastrophic atomic-heavy backward path.
- A6000 GQA long-seq example:
  - legacy `bwd_qkv ~701 ms`
  - compact v2 `bwd_qkv ~115 ms`
  - about `6.1x` faster
- A6000 MHA long-seq example:
  - legacy `bwd_qkv ~2827 ms`
  - compact v2 `bwd_qkv ~370 ms`
  - about `7.6x` faster

Primary references:

- `docs/plans/2026-03-04-nsa-per-query-triton-kernel-design-notes.md`
- `artifacts/nsa_diagnostics/ours_compact_v2_vs_legacy_vs_tilda_ap_a6000_20260304_232154.json`
- `artifacts/nsa_diagnostics/ours_compact_v2_mha_vs_legacy_vs_tilda_ap_a6000_20260304_232519.json`

### dK/dV is no longer the main bottleneck

Profiler evidence after compact v2 showed the main hotspots are:

- `_sel_perq_bwd_dq_kernel`
- `_sel_perq_fwd_kernel`

Reference:

- `docs/plans/2026-03-04-nsa-per-query-triton-kernel-design-notes.md`

### Real grouped-head forward/dQ rewrite failed

- Attempted to remove padded grouped-head width for forward/dQ.
- On NVIDIA Triton this lost the fast `tl.dot` path for small grouped-head width.
- Result was slower, not faster.
- Important lesson: padded grouped-head execution is not just waste; it also keeps Triton on an optimized matmul path.

Reference:

- `docs/plans/2026-03-06-nsa-forward-dq-v3-design.md`

### p16 is a kernel resource problem, not a data bug

- For practical top-n at `d >= 64`, current p16 kernels exceed shared-memory limits on A6000.
- Example: `C=256, d=64, p16, top_n>=2` fails due to shared memory around `152 KB`, above A6000 block limit.

References:

- `artifacts/nsa_diagnostics/nsa_worst_case_root_cause_20260305.md`
- `artifacts/nsa_diagnostics/attn_baselines_and_p16_investigation_20260305.md`

### Selection is the dominant remaining attention cost

Attention-only benchmark on fixed `q/k/v`, A100, `C=64`, `G=4`, `p=8`, `w=16`:

- `256x256`, `512 selected/query`
  - compression `~0.74 ms`
  - selection `~14.42 ms`
  - window `~0.24 ms`

Attention-only benchmark on fixed `q/k/v`, A100, `C=384`, `Hq=6`, `Hkv=2`, `G=3`, `d=64`, `p=8`, `w=16`:

- `256x256`, `512 selected/query`
  - compression `~0.96 ms`
  - selection `~30.74 ms`
  - window `~0.55 ms`

References:

- `artifacts/nsa_diagnostics/nsa_attention_only_qkv_vs_flash_a100_20260306_120347.json`
- `artifacts/nsa_diagnostics/nsa_attention_only_qkv_vs_flash_a100_c384_h6_hkv2_20260306_122937.json`

### 128x128 is still hard; 256x256 is where NSA starts paying off

At `C=64` attention-only:

- `128x128`: Flash still wins clearly
- `256x256`, `512 selected/query`: NSA forward is roughly tied/slightly ahead, backward is better

At `C=384` attention-only:

- `128x128`: Flash still wins
- `256x256`, `512 selected/query`: NSA forward `~1.29x` faster, forward+backward `~2.32x` faster

This means the asymptotic advantage is real, but current implementation overhead delays the crossover.

## Important Comparisons

### Tilda comparison

- Tilda top-k sparse attention operator did not show our old dK/dV explosion.
- Tilda-style active-query compaction was the right model for fixing our old backward issue.
- Tilda remains useful as a reference for layout and backend discipline.

Reference:

- `artifacts/nsa_diagnostics/ours_vs_tilda_bwd_gap_20260305.md`

### FlashAttention comparison

- Operator-only wins can be misleading.
- Full-module and attention-only comparisons must be kept separate.
- Current trustworthy comparison for attention core is:
  - fixed `q/k/v`
  - NSA = compression + selection + window
  - Flash = dense full attention on same `q/k/v`

References:

- `artifacts/nsa_diagnostics/nsa_attention_only_qkv_vs_flash_a100_20260306_120347.json`
- `artifacts/nsa_diagnostics/nsa_attention_only_qkv_vs_flash_a100_c384_h6_hkv2_20260306_122937.json`

## Things That Failed or Were Reverted

1. Real-`G` forward/dQ Triton rewrite
2. Logsumexp-style selection-score change
3. Treating p16 failures as a selector bug

These should not be retried casually without a new structural idea.

## Current Strategic View

1. Sparse attention still has room to improve, but the remaining problem is structural.
2. Shell and FFN matter for full block speedup and must be optimized too.
3. Best short-term move is selection packing / layout cleanup.
4. Best medium-term move is a Flash/FlexAttention-style sparse backend for H100.

## Current Priority Order

1. Packing-first selection path
2. Full-block breakdown harness
3. FFN runtime redesign
4. Attention shell optimization
5. H100 sparse backend prototype

## Update Protocol

Every meaningful NSA experiment or implementation should update this file with:

1. What changed
2. Exact artifact path or test command
3. What the result means
4. Whether it changes the priority order

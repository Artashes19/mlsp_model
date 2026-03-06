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

## Packing Contract

Current packing metadata contract:

- `unique_patch_ids: [total_unique] int32`
- `cu_unique_counts: [B*h_kv + 1] int32`
- `packed_idx: [B, h_kv, T, top_n] int32`

Interpretation:

- each `(batch, kv_head)` row owns one local packed patch table
- `unique_patch_ids[cu_unique_counts[row]:cu_unique_counts[row+1]]` is that row's sorted patch table
- `packed_idx` remaps every original selected patch id into the row-local packed table

Why packing is per `(batch, kv_head)` and not per query:

1. Per-query packing would duplicate K/V patches heavily across nearby queries.
2. Per-head packing lets many queries reuse the same packed K/V tables.
3. This is the right bridge toward both:
   - a better current Triton path
   - a future H100 block-sparse / FlexAttention-style backend

## Packing Benchmark Snapshot

Reference artifact:

- `artifacts/nsa_diagnostics/selection_packing_vs_unpacked_a100_20260306_132736.json`

Result:

- Packing cost is small enough to continue for `128x128` and `256x256`.
- At `256x256`, packing metadata build plus packed K/V gather is still under `1 ms`:
  - `k=8`: `0.325 + 0.211 = 0.536 ms`
  - `k=16`: `0.557 + 0.219 = 0.776 ms`
- Current unpacked selection forward at the same shape is much larger:
  - `k=8`: `6.809 ms`
  - `k=16`: `13.036 ms`
- So pure packing overhead is about:
  - `7.9%` of current selection forward for `256x256, k=8`
  - `6.0%` of current selection forward for `256x256, k=16`

Important dedup observation:

- Selected patches are heavily reused across queries, which supports packing once per `(batch, kv_head)`:
  - `64x64`: only `63-64` unique patches per head
  - `128x128`: about `251` unique patches at `k=8`, `256` at `k=16`
  - `256x256`: about `820` unique patches at `k=8`, `955` at `k=16`
- In terms of selected slots over unique packed patches:
  - `64x64`: about `520x` at `k=8`, `1024x` at `k=16`
  - `128x128`: about `522x` at `k=8`, `1024x` at `k=16`
  - `256x256`: about `639x` at `k=8`, `1098x` at `k=16`

Interpretation:

1. The packing build and gather work is cheap relative to the current long-sequence selection kernel.
2. Reuse of selected patches across queries is very high, so per-head packing has a strong structural basis.
3. This is promising for `256x256`, somewhat promising for `128x128`, and less compelling for `64x64`.

## Packed Forward Benchmark Snapshot

Reference artifact:

- `artifacts/nsa_diagnostics/selection_packing_vs_unpacked_a100_20260306_133937.json`

What changed:

- Added a real packed forward path behind `selection_forward_mode="packed"`.
- Benchmarked packed vs unpacked forward on the same A100 shapes as Task 3.

Results:

- `128x128`
  - `k=8`: packed selection is essentially tied, but packed total attention is slower
    - selection: `2.390 ms` packed vs `2.386 ms` unpacked
    - total: `4.016 ms` packed vs `3.651 ms` unpacked
  - `k=16`: packed wins modestly
    - selection: `3.203 ms` packed vs `3.419 ms` unpacked
    - total: `4.505 ms` packed vs `4.753 ms` unpacked

- `256x256`
  - `k=8`: packed wins modestly
    - selection: `6.243 ms` packed vs `6.833 ms` unpacked
    - total: `14.859 ms` packed vs `15.191 ms` unpacked
  - `k=16`: packed wins clearly but not dramatically
    - selection: `11.226 ms` packed vs `13.053 ms` unpacked
    - total: `19.852 ms` packed vs `21.389 ms` unpacked

Interpretation:

1. Packing gives a real forward win at `256x256`, stronger when the selected-token budget is larger.
2. At `128x128`, the win is not robust; it only appears for the heavier `k=16` case.
3. The total-attention gain is much smaller than the selection-only gain because compression and window were already cheap.

## Packing Decision Gate

Decision:

- proceed to packed `dQ`
- do not start packed `dK/dV`

Why:

1. The forward data shows packing is a real locality improvement at `256x256`, so the idea is not a dead end.
2. The gain is still modest at total-attention level:
   - `256x256, k=8`: about `1.022x`
   - `256x256, k=16`: about `1.077x`
3. `dK/dV` is already not the main bottleneck after the earlier compact redesign.
4. `dQ` remains one of the main hotspots, so it is the only backward path where packing still has a clear chance to matter.
5. `128x128` evidence is too mixed to justify a full packed-backward expansion right now.

Meaning:

- Next packing work, if any, should target `dQ` only.
- If packed `dQ` does not produce a clear long-sequence gain, stop packing work and pivot to FFN / shell.

## Current Priority Order

1. Packed `dQ` for the selection path
2. FFN runtime redesign
3. Attention shell optimization
4. H100 sparse backend prototype

## Update Protocol

Every meaningful NSA experiment or implementation should update this file with:

1. What changed
2. Exact artifact path or test command
3. What the result means
4. Whether it changes the priority order

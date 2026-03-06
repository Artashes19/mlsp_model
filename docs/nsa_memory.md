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

### Triton-only long-sequence investigation: dQ is large, but forward has the easy win

Reference artifacts:

- `artifacts/nsa_diagnostics/nsa_selection_triton_hotspots_a100_20260306_230704.json`
- `artifacts/nsa_diagnostics/selection_triton_kernel_sweep_a100_20260306_234808.json`
- `artifacts/nsa_diagnostics/selection_triton_kernel_sweep_a100_20260307_001149.json`

Hotspot profile on A100 for the current unpacked selection path showed:

- `256x256`, `C=384`, `top_n=8`
  - scoring `12.95 ms`
  - forward `11.93 ms`
  - dQ `15.71 ms`
  - dK/dV `5.07 ms`
- `256x256`, `C=384`, `top_n=16`
  - scoring `12.99 ms`
  - forward `22.99 ms`
  - dQ `30.58 ms`
  - dK/dV `8.68 ms`
- `256x256`, `C=512`, `top_n=8`
  - scoring `15.13 ms`
  - forward `11.94 ms`
  - dQ `15.81 ms`
  - dK/dV `6.51 ms`
- `256x256`, `C=512`, `top_n=16`
  - scoring `15.21 ms`
  - forward `23.01 ms`
  - dQ `30.69 ms`
  - dK/dV `10.97 ms`

Interpretation:

1. At wide channels and long sequence, `dQ` is the largest single Triton kernel cost.
2. Forward is the next Triton hotspot.
3. Scoring is still significant, but it is no longer the only thing that matters in the wide long-sequence regime.

Current-kernel sweep result:

- Forward meta is under-tuned in both the priority `256x256` regime and the `128x128` sentinel regime.
- dQ current meta is already locally optimal at `256x256`.
- dQ only shows small `BLOCK_Q=16` wins at `128x128`, not at `256x256`.
- The direct Triton sweep harness is compile-heavy and should be treated as a one-shot ranking tool, not a fast inner tuning loop.

Measured forward sweep results on A100:

- `256x256`, `C=384`, `top_n=8`
  - current `num_warps=8`: `15.33 ms`
  - best `num_warps=4`: `9.29 ms`
  - forward-only gain: `1.65x`
- `256x256`, `C=384`, `top_n=16`
  - current `num_warps=8`: `22.95 ms`
  - best `num_warps=2`: `17.67 ms`
  - forward-only gain: `1.30x`
- `256x256`, `C=512`, `top_n=8`
  - current `num_warps=8`: `11.92 ms`
  - best `num_warps=4`: `9.30 ms`
  - forward-only gain: `1.28x`
- `256x256`, `C=512`, `top_n=16`
  - current `num_warps=8`: `22.91 ms`
  - best `num_warps=2`: `17.66 ms`
  - forward-only gain: `1.30x`

Measured dQ sweep results on A100:

- `256x256`, `C=384`, `top_n=8/16`: current `BLOCK_Q=32, num_warps=4` is best in the tested neighborhood
- `256x256`, `C=512`, `top_n=8/16`: current `BLOCK_Q=32, num_warps=4` is best in the tested neighborhood
- `128x128` sentinel:
  - `C=384`, `top_n=8/16`: `BLOCK_Q=16, num_warps=4` is modestly better than current
  - `C=512`, `top_n=8`: current `BLOCK_Q=32, num_warps=4` is best
  - `C=512`, `top_n=16`: `BLOCK_Q=16, num_warps=4` is only about `1.05x` better

Decision from this investigation:

1. The next Triton-only target should start with forward launch-meta retuning.
2. dQ still matters more in absolute time, but there is no easy launch-meta win there at `256x256`.
3. Any next dQ improvement will likely require a structural kernel change, not another small heuristic tweak.
4. `_select_num_warps_per_query(...)` is a known weak heuristic for the wide-channel selection forward regimes we care about.

### Forward warp retune landed cleanly

Reference artifact:

- `artifacts/nsa_diagnostics/nsa_selection_triton_hotspots_a100_20260307_004533.json`

What changed:

- `_select_num_warps_per_query(...)` now uses `top_n` and `seq_len` so it can distinguish:
  - `128x128, top_n=8` wide forward cases: `2` warps
  - `256x256, top_n=8` wide forward cases: `4` warps
  - `top_n=16` wide forward cases: `2` warps

Selector sanity points:

- `(BLOCK_G=16, BLOCK_KV=64, BLOCK_D=64, top_n=8, seq_len=16384) -> 2`
- `(BLOCK_G=16, BLOCK_KV=64, BLOCK_D=64, top_n=16, seq_len=16384) -> 2`
- `(BLOCK_G=16, BLOCK_KV=64, BLOCK_D=64, top_n=8, seq_len=65536) -> 4`
- `(BLOCK_G=16, BLOCK_KV=64, BLOCK_D=64, top_n=16, seq_len=65536) -> 2`

Parity status:

- focused forward parity: passed
- broader per-query parity: `22 passed, 33 deselected`

Measured effect on A100 hotspot profile:

- `256x256`, `C=384`, `top_n=8`
  - forward: `11.93 -> 9.28 ms` (`1.29x`)
  - dQ: `15.71 -> 15.71 ms` (flat)
  - total path: `46.36 -> 44.62 ms` (`1.04x`)
- `256x256`, `C=384`, `top_n=16`
  - forward: `22.99 -> 17.66 ms` (`1.30x`)
  - dQ: `30.58 -> 30.47 ms` (flat)
  - total path: `75.87 -> 71.03 ms` (`1.07x`)
- `256x256`, `C=512`, `top_n=8`
  - forward: `11.94 -> 9.29 ms` (`1.29x`)
  - dQ: `15.81 -> 15.79 ms` (flat)
  - total path: `50.22 -> 48.41 ms` (`1.04x`)
- `256x256`, `C=512`, `top_n=16`
  - forward: `23.01 -> 17.75 ms` (`1.30x`)
  - dQ: `30.69 -> 30.60 ms` (flat)
  - total path: `80.68 -> 75.58 ms` (`1.07x`)

Sentinel `128x128` effect:

- total path improved modestly, about `1.05x`
- forward improved slightly to moderately
- dQ moved little and sometimes within noise

Decision after the retune:

1. The easy Triton-only forward meta miss is now fixed for the measured wide regimes.
2. dQ is now the next remaining Triton kernel target.
3. The next dQ step should be structural, not another small launch-meta tweak.

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
5. FFN and shell changes must be preceded by granular full-layer profiling on A100.
6. Before any backend pivot, there is still a clear Triton-only forward retuning win to capture in the current selection forward kernel.

## Profiling Arsenal

Selection-path profiling stack we want to maintain:

1. `torch.profiler` for region-level attribution, execution traces, and memory timelines
2. HTA for trace diff, temporal breakdown, kernel breakdown, and frequent-kernel analysis
3. Nsight Systems for host/device gap analysis and Python launch attribution
4. Nsight Compute for kernel-level stall analysis, roofline, and profile-series sweeps
5. Triton Proton as an optional Triton-native microbenchmark/profiling layer

Environment facts verified on this machine:

- `nsys` is available at `/usr/local/bin/nsys`
- `ncu` is available at `/usr/local/cuda/bin/ncu`
- `triton.profiler` and `triton.profiler.proton` import successfully in the `dev` env
- `HolisticTraceAnalysis` is not installed yet in the `dev` env and should be installed when we execute the profiling-arsenal plan

Tooling decision:

1. Keep scope selection-only for this phase
2. Build one selection harness that can feed all profiling tools
3. Use cheap/high-level tools first and low-level kernel tools second

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

## Packed dQ Benchmark Snapshot

Reference artifact:

- `artifacts/nsa_diagnostics/selection_packing_vs_unpacked_a100_20260306_140943.json`

Results:

- `64x64`
  - `k=8`
    - `bwd_q_only`: unpacked `0.891 ms`, packed `8.398 ms`
    - `bwd_qkv`: unpacked `8.57 ms`, packed `14.65 ms`
  - `k=16`
    - `bwd_q_only`: unpacked `1.633 ms`, packed `2.843 ms`
    - `bwd_qkv`: unpacked `3.118 ms`, packed `3.58 ms`

- `128x128`
  - `k=8`
    - `bwd_q_only`: unpacked `3.611 ms`, packed `3.97 ms`
    - `bwd_qkv`: unpacked `4.412 ms`, packed `4.926 ms`
  - `k=16`
    - `bwd_q_only`: unpacked `6.21 ms`, packed `6.786 ms`
    - `bwd_qkv`: unpacked `6.787 ms`, packed `6.435 ms`

- `256x256`
  - `k=8`
    - `bwd_q_only`: unpacked `11.758 ms`, packed `12.161 ms`
    - `bwd_qkv`: unpacked `11.23 ms`, packed `11.599 ms`
  - `k=16`
    - `bwd_q_only`: unpacked `19.098 ms`, packed `20.016 ms`
    - `bwd_qkv`: unpacked `19.734 ms`, packed `20.132 ms`

Decision:

- `selection_dq_mode="auto"` stays unpacked
- packed `dQ` is not strong enough to justify auto-enable
- stop packing work after `dQ`

Why:

1. There is no convincing long-sequence win on A100.
2. The only backward case that edges ahead is `128x128, k=16, bwd_qkv`, and the gain is too small and too narrow.
3. `256x256` is the regime that matters most for NSA scaling, and packed `dQ` loses there.
4. The correct next step is granular full-layer profiling, not more packing variants.

## Current Priority Order

1. Keep the next optimization target on the selection path or its backend.
2. Keep packed forward as a narrow locality improvement; do not expand packing work blindly.
3. Keep `selection_dq_mode="auto"` on unpacked unless new data proves otherwise.
4. Treat attention-shell work as secondary, because it is bounded and much smaller than selection at `256x256`.
5. Keep FFN redesign blocked for the current `C=64` regime until either channel count or profiler share says otherwise.
6. Revisit H100 sparse backend work after the current A100 selection bottleneck is pushed further.

Meaning:

- Next packing work, if any, should target `dQ` only.
- If packed `dQ` does not produce a clear long-sequence gain, stop packing work and return to selection-path or backend work.

## Update Protocol

Every meaningful NSA experiment or implementation should update this file with:

1. What changed
2. Exact artifact path or test command
3. What the result means
4. Whether it changes the priority order

## Granular Layer Profiling Baseline

Reference artifacts:

- `artifacts/nsa_diagnostics/nsa_layer_granular_profile_a100_20260306_182020.json`
- `artifacts/nsa_diagnostics/nsa_layer_granular_profile_a100_20260306_182020.txt`
- Harness: `artifacts/nsa_diagnostics/profile_nsa_layer_granular.py`

Setup:

- A100
- `B=1, C=64, heads=4, G=4, p=8, w=16, bf16`
- sizes: `128x128`, `256x256`
- `top_n=8`, `16`
- `5` warmup iterations before benchmark

Validation:

- manual attention reconstruction vs real `NSA2DAttention.forward`: exact match
- manual FFN reconstruction vs real `GatedDepthwiseFFN.forward`: exact match
- profiler wrapper vs real `TransformerBlock.forward`: exact match

Measured block split:

- `128x128, top_n=8`
  - block forward: `5.95 ms`
  - block forward+backward: `11.47 ms`
  - attention forward inside block: `5.56 ms`
  - FFN forward inside block: `0.35 ms`

- `128x128, top_n=16`
  - block forward: `7.26 ms`
  - block forward+backward: `14.67 ms`
  - attention forward inside block: `7.02 ms`
  - FFN forward inside block: `0.44 ms`

- `256x256, top_n=8`
  - block forward: `18.66 ms`
  - block forward+backward: `30.97 ms`
  - attention forward inside block: `17.59 ms`
  - FFN forward inside block: `0.65 ms`

- `256x256, top_n=16`
  - block forward: `24.71 ms`
  - block forward+backward: `39.75 ms`
  - attention forward inside block: `24.07 ms`
  - FFN forward inside block: `0.66 ms`

Measured attention split:

- `128x128, top_n=8`
  - selection total: `3.51 ms`
  - compression: `1.00 ms`
  - window: `0.18 ms`
  - shell total: `1.93 ms`

- `128x128, top_n=16`
  - selection total: `4.41 ms`
  - compression: `1.11 ms`
  - window: `0.21 ms`
  - shell total: `2.25 ms`

- `256x256, top_n=8`
  - selection total: `14.71 ms`
  - compression: `1.41 ms`
  - window: `0.28 ms`
  - shell total: `2.23 ms`

- `256x256, top_n=16`
  - selection total: `20.96 ms`
  - compression: `1.41 ms`
  - window: `0.28 ms`
  - shell total: `2.24 ms`

Shell details:

- q/k/v blocks together stay around `0.32-0.38 ms` at `128x128` and `256x256`
- RoPE stays around `1.10-1.28 ms`
- gate pool + gate MLP stay around `0.18-0.21 ms`
- output proj stays around `0.09-0.10 ms`

FFN details:

- FFN is small at `C=64`
- the two FFN `DConvBlock` branches dominate FFN time:
  - `128x128`: about `0.31-0.33 ms`
  - `256x256`: about `0.51 ms`
- FFN projection is only about `0.07-0.11 ms`

Interpretation:

1. For the current `C=64, G=4` block, FFN is not the next high-impact optimization target.
2. The attention shell is not free, but it is bounded and much smaller than selection at `256x256`.
3. Selection remains the dominant runtime problem:
   - about `63%` of attention forward at `128x128, top_n=8`
   - about `84-87%` of attention forward at `256x256`
4. Compression and window are already small enough that they should not be prioritized.
5. For the current practical regime, the next predictable win is still in the selection path or its backend, not FFN.

Priority update:

1. Keep FFN redesign blocked for this `C=64` regime.
2. Keep shell optimization as secondary work, not the next main target.
3. Return to selection-path work:
   - better selection forward/dQ implementation
   - or backend/layout work that reduces selection cost structurally
4. Revisit FFN after either:
   - higher-channel regimes show larger FFN share
   - or selection stops dominating

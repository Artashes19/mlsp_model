"""
Triton-accelerated GQA 2D selection attention for NSA.

Group-Query Attention variant: each program handles one KV-head and loads
ALL G query heads in its group, sharing K/V loads across the group.

Shapes:
    Q:       [B, h_q, T, d]      (h_q = h_kv * G)
    K, V:    [B, h_kv, T, d]
    top_idx: [B, h_kv, top_n]    (per-KV-head patch selection)

Forward:  _sel_gqa_fwd_kernel
Backward: _sel_gqa_bwd_preprocess_kernel, _sel_gqa_bwd_dq_kernel, _sel_gqa_bwd_dkv_kernel
Autograd: SelectionAttn2DGQA (torch.autograd.Function)
"""
import math

import torch
import triton
import triton.language as tl


# ============================================================
# Helpers
# ============================================================

def _next_power_of_2(n: int) -> int:
    n = max(n, 1)
    return 1 << (n - 1).bit_length()


def _select_block_q_backward(d: int, top_n: int, pp: int) -> int:
    selected_tokens = top_n * pp
    return 32 if (d <= 64 or selected_tokens >= 256) else 64


def _select_num_warps(block_q: int, G: int) -> int:
    total_rows = block_q * G
    if total_rows <= 32:
        return 4
    elif total_rows <= 64:
        return 4
    else:
        return 8


def make_patch_starts(H: int, W: int, patch_size: int, device: torch.device) -> torch.Tensor:
    """
    Compute the flattened offset of each patch's top-left token.

    Patch (ph, pw) has top-left at spatial (ph*p, pw*p),
    which maps to flat index (ph*p)*W + pw*p in row-major [H*W].

    Returns: [n_patches] int32 tensor.
    """
    p = patch_size
    nH, nW = H // p, W // p
    ph = torch.arange(nH, device=device)
    pw = torch.arange(nW, device=device)
    starts = (ph[:, None] * p * W + pw[None, :] * p).reshape(-1)
    return starts.to(torch.int32)


# ============================================================
# Forward Kernel
# ============================================================

@triton.jit
def _sel_gqa_fwd_kernel(
    Q, K, V, O, LSE,
    top_idx, patch_starts,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_lb, stride_lh, stride_lt,
    stride_idxb, stride_idxh, stride_idxi,
    sm_scale,
    T: tl.constexpr,
    D: tl.constexpr,
    W_spatial: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    TOP_N: tl.constexpr,
    H_KV: tl.constexpr,
    G: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_QG: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """
    GQA group-centric forward kernel for selection attention.

    Grid: (cdiv(T, BLOCK_Q), B * H_kv)

    Each program handles one query block of BLOCK_Q positions for ALL G
    query heads belonging to one KV head. K/V are loaded once per patch
    and shared across all G heads.

    Internal block layout: [BLOCK_Q * G, BLOCK_D]
      rows [g*BLOCK_Q : (g+1)*BLOCK_Q] correspond to Q-head kv_idx*G + g
    """
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Decompose program index into batch and kv-head
    kv_idx = pid_bh % H_KV
    b_idx = pid_bh // H_KV

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)  # [BLOCK_Q]
    mask_q = offs_q < T
    offs_d = tl.arange(0, BLOCK_D)  # [BLOCK_D]
    mask_d = offs_d < D

    # Load Q for all G heads: [BLOCK_QG, BLOCK_D]
    # Row index in the combined block: g * BLOCK_Q + local_q
    offs_g = tl.arange(0, BLOCK_QG)  # [BLOCK_QG] = [BLOCK_Q * G]
    g_head = offs_g // BLOCK_Q       # which group member (0..G-1)
    local_q = offs_g % BLOCK_Q       # which position within block

    q_head_idx = kv_idx * G + g_head  # [BLOCK_QG] actual Q-head indices
    q_pos = pid_q * BLOCK_Q + local_q  # [BLOCK_QG] spatial positions

    mask_qg = q_pos < T  # [BLOCK_QG]

    # Pointer: Q[b, q_head_idx, q_pos, d]
    q_ptrs = (Q
              + b_idx * stride_qb
              + q_head_idx[:, None] * stride_qh
              + q_pos[:, None] * stride_qt
              + offs_d[None, :] * stride_qd)
    b_q = tl.load(q_ptrs, mask=mask_qg[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    b_q = b_q * (LOG2E * sm_scale)

    # Online softmax accumulators: [BLOCK_QG]
    b_m = tl.full([BLOCK_QG], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BLOCK_QG], dtype=tl.float32)
    b_o = tl.zeros([BLOCK_QG, BLOCK_D], dtype=tl.float32)

    # Pre-compute local offsets within a patch: [BLOCK_KV]
    offs_local = tl.arange(0, BLOCK_KV)
    local_row = offs_local // P
    local_col = offs_local % P
    mask_pp = offs_local < PP

    for i in range(TOP_N):
        patch_idx = tl.load(top_idx + b_idx * stride_idxb + kv_idx * stride_idxh + i * stride_idxi)
        kv_base = tl.load(patch_starts + patch_idx)

        # Flat token indices for all PP tokens in this patch
        flat_indices = kv_base + local_row * W_spatial + local_col  # [BLOCK_KV]
        valid = mask_pp & (flat_indices < T)

        # Load K block: [BLOCK_KV, BLOCK_D] — shared across G heads
        k_ptrs = (K
                  + b_idx * stride_kb
                  + kv_idx * stride_kh
                  + flat_indices[:, None] * stride_kt
                  + offs_d[None, :] * stride_kd)
        b_k = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # Load V block: [BLOCK_KV, BLOCK_D] — shared across G heads
        v_ptrs = (V
                  + b_idx * stride_vb
                  + kv_idx * stride_vh
                  + flat_indices[:, None] * stride_vt
                  + offs_d[None, :] * stride_vd)
        b_v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # Attention scores: [BLOCK_QG, BLOCK_KV] = Q_all_G @ K^T
        b_s = tl.dot(b_q, tl.trans(b_k))
        b_s = tl.where(valid[None, :], b_s, float("-inf"))

        # Online softmax update (per-row, independent across G heads)
        b_m_new = tl.maximum(b_m, tl.max(b_s, axis=1))
        b_r = tl.exp2(b_m - b_m_new)
        b_p = tl.exp2(b_s - b_m_new[:, None])
        b_p = tl.where(valid[None, :], b_p, 0.0)

        b_acc = b_acc * b_r + tl.sum(b_p, axis=1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(tl.float32), b_v)
        b_m = b_m_new

    # Normalize
    b_o = b_o / tl.maximum(b_acc[:, None], 1e-6)

    # LSE: convert from base-2 to natural log
    b_lse = b_m / LOG2E + tl.log(tl.maximum(b_acc, 1e-6))

    # Store O for all G heads: O[b, q_head_idx, q_pos, d]
    o_ptrs = (O
              + b_idx * stride_ob
              + q_head_idx[:, None] * stride_oh
              + q_pos[:, None] * stride_ot
              + offs_d[None, :] * stride_od)
    tl.store(o_ptrs, b_o.to(O.dtype.element_ty), mask=mask_qg[:, None] & mask_d[None, :])

    # Store LSE for all G heads: LSE[b, q_head_idx, q_pos]
    lse_ptrs = (LSE
                + b_idx * stride_lb
                + q_head_idx * stride_lh
                + q_pos * stride_lt)
    tl.store(lse_ptrs, b_lse, mask=mask_qg)


# ============================================================
# Backward Preprocess Kernel
# ============================================================

@triton.jit
def _sel_gqa_bwd_preprocess_kernel(
    O, DO, DELTA,
    stride_ob, stride_oh, stride_ot, stride_od,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute delta[b,h_q,t] = sum_d(O[b,h_q,t,d] * dO[b,h_q,t,d]).

    Grid: (T, B * h_q)
    """
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    if pid_t < T:
        o_ptrs = O + pid_bh * stride_oh + pid_t * stride_ot + offs_d * stride_od
        do_ptrs = DO + pid_bh * stride_oh + pid_t * stride_ot + offs_d * stride_od
        b_o = tl.load(o_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        b_do = tl.load(do_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        delta = tl.sum(b_o * b_do)
        tl.store(DELTA + pid_bh * T + pid_t, delta)


# ============================================================
# Backward dQ Kernel
# ============================================================

@triton.jit
def _sel_gqa_bwd_dq_kernel(
    Q, K, V, DO, LSE, DELTA, DQ,
    top_idx, patch_starts,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_idxb, stride_idxh, stride_idxi,
    sm_scale,
    T: tl.constexpr,
    D: tl.constexpr,
    W_spatial: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    TOP_N: tl.constexpr,
    H_KV: tl.constexpr,
    G: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_QG: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """
    GQA group-centric dQ kernel (query-stationary).

    Grid: (cdiv(T, BLOCK_Q), B * H_kv)

    Each program computes dQ for BLOCK_Q positions across ALL G heads
    in one KV-head group. No atomics needed (program owns its Q block).
    """
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)

    kv_idx = pid_bh % H_KV
    b_idx = pid_bh // H_KV

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Combined Q/G indexing: [BLOCK_QG]
    offs_g = tl.arange(0, BLOCK_QG)
    g_head = offs_g // BLOCK_Q
    local_q = offs_g % BLOCK_Q

    q_head_idx = kv_idx * G + g_head  # actual Q-head indices
    q_pos = pid_q * BLOCK_Q + local_q  # spatial positions
    mask_qg = q_pos < T

    # Load Q: [BLOCK_QG, BLOCK_D]
    q_ptrs = (Q
              + b_idx * stride_qb
              + q_head_idx[:, None] * stride_qh
              + q_pos[:, None] * stride_qt
              + offs_d[None, :] * stride_qd)
    b_q = tl.load(q_ptrs, mask=mask_qg[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    b_q_scaled = b_q * (LOG2E * sm_scale)

    # Load dO: [BLOCK_QG, BLOCK_D]
    # dO has same layout as O: [B, h_q, T, d]
    do_ptrs = (DO
               + b_idx * stride_qb
               + q_head_idx[:, None] * stride_qh
               + q_pos[:, None] * stride_qt
               + offs_d[None, :] * stride_qd)
    b_do = tl.load(do_ptrs, mask=mask_qg[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # Load LSE and delta for all G heads: [BLOCK_QG]
    # LSE is stored as flat [B*h_q, T]
    h_q_total = H_KV * G
    lse_off = (b_idx * h_q_total + q_head_idx) * T + q_pos
    b_lse = tl.load(LSE + lse_off, mask=mask_qg, other=0.0)
    b_delta = tl.load(DELTA + lse_off, mask=mask_qg, other=0.0)

    b_dq = tl.zeros([BLOCK_QG, BLOCK_D], dtype=tl.float32)

    # Pre-compute local offsets within a patch
    offs_local = tl.arange(0, BLOCK_KV)
    local_row = offs_local // P
    local_col = offs_local % P
    mask_pp = offs_local < PP

    for i in range(TOP_N):
        patch_idx = tl.load(top_idx + b_idx * stride_idxb + kv_idx * stride_idxh + i * stride_idxi)
        kv_base = tl.load(patch_starts + patch_idx)

        flat_indices = kv_base + local_row * W_spatial + local_col
        valid = mask_pp & (flat_indices < T)

        # Load K, V: [BLOCK_KV, BLOCK_D] — shared across G heads
        k_ptrs = (K
                  + b_idx * stride_kb
                  + kv_idx * stride_kh
                  + flat_indices[:, None] * stride_kt
                  + offs_d[None, :] * stride_kd)
        b_k = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = (V
                  + b_idx * stride_vb
                  + kv_idx * stride_vh
                  + flat_indices[:, None] * stride_vt
                  + offs_d[None, :] * stride_vd)
        b_v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # Recompute attention: [BLOCK_QG, BLOCK_KV]
        b_s = tl.dot(b_q_scaled, tl.trans(b_k))
        b_p = tl.exp2(b_s - b_lse[:, None] * LOG2E)
        b_p = tl.where(valid[None, :], b_p, 0.0)

        # dp = dO @ V^T: [BLOCK_QG, BLOCK_KV]
        b_dp = tl.dot(b_do, tl.trans(b_v))

        # ds = p * (dp - delta): [BLOCK_QG, BLOCK_KV]
        b_ds = b_p * (b_dp - b_delta[:, None])

        # dQ += ds @ K * scale: [BLOCK_QG, BLOCK_D]
        b_dq += tl.dot(b_ds.to(tl.float32), b_k) * sm_scale

    # Store dQ for all G heads
    dq_ptrs = (DQ
               + b_idx * stride_qb
               + q_head_idx[:, None] * stride_qh
               + q_pos[:, None] * stride_qt
               + offs_d[None, :] * stride_qd)
    tl.store(dq_ptrs, b_dq.to(DQ.dtype.element_ty), mask=mask_qg[:, None] & mask_d[None, :])


# ============================================================
# Backward dKV Kernel
# ============================================================

@triton.jit
def _sel_gqa_bwd_dkv_kernel(
    Q, K, V, DO, LSE, DELTA, DK, DV,
    top_idx, patch_starts,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_dkb, stride_dkh, stride_dkt, stride_dkd,
    stride_idxb, stride_idxh, stride_idxi,
    sm_scale,
    T: tl.constexpr,
    D: tl.constexpr,
    W_spatial: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    TOP_N: tl.constexpr,
    H_KV: tl.constexpr,
    G: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_QG: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """
    GQA KV-stationary backward kernel with atomics.

    Grid: (1, B * H_kv * TOP_N)

    Each program computes dK, dV for one selected patch by iterating
    over ALL query blocks (loading all G heads per block).
    Uses tl.atomic_add to accumulate since the same token may appear
    in multiple selected patches.
    """
    pid_bh_n = tl.program_id(1)

    n_idx = pid_bh_n % TOP_N
    pid_bh = pid_bh_n // TOP_N
    kv_idx = pid_bh % H_KV
    b_idx = pid_bh // H_KV

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Compute flat indices for all tokens in this patch
    offs_local = tl.arange(0, BLOCK_KV)
    local_row = offs_local // P
    local_col = offs_local % P
    mask_pp = offs_local < PP

    patch_idx = tl.load(top_idx + b_idx * stride_idxb + kv_idx * stride_idxh + n_idx * stride_idxi)
    kv_base = tl.load(patch_starts + patch_idx)
    flat_indices = kv_base + local_row * W_spatial + local_col  # [BLOCK_KV]
    valid = mask_pp & (flat_indices < T)

    # Load K, V: [BLOCK_KV, BLOCK_D]
    k_ptrs = (K
              + b_idx * stride_kb
              + kv_idx * stride_kh
              + flat_indices[:, None] * stride_kt
              + offs_d[None, :] * stride_kd)
    b_k = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    v_ptrs = (V
              + b_idx * stride_vb
              + kv_idx * stride_vh
              + flat_indices[:, None] * stride_vt
              + offs_d[None, :] * stride_vd)
    b_v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # Accumulators: [BLOCK_KV, BLOCK_D]
    b_dk = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)
    b_dv = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)

    # Combined Q/G indexing templates
    offs_g = tl.arange(0, BLOCK_QG)
    g_head = offs_g // BLOCK_Q       # which group member
    local_q_off = offs_g % BLOCK_Q   # position offset within block

    q_head_idx = kv_idx * G + g_head  # [BLOCK_QG]
    h_q_total = H_KV * G

    # Iterate over all query positions in blocks
    for q_start in range(0, T, BLOCK_Q):
        q_pos = q_start + local_q_off  # [BLOCK_QG]
        mask_qg = q_pos < T

        # Load Q: [BLOCK_QG, BLOCK_D]
        q_ptrs = (Q
                  + b_idx * stride_qb
                  + q_head_idx[:, None] * stride_qh
                  + q_pos[:, None] * stride_qt
                  + offs_d[None, :] * stride_qd)
        b_q = tl.load(q_ptrs, mask=mask_qg[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        b_q_scaled = b_q * (LOG2E * sm_scale)

        # Load dO: [BLOCK_QG, BLOCK_D]
        do_ptrs = (DO
                   + b_idx * stride_qb
                   + q_head_idx[:, None] * stride_qh
                   + q_pos[:, None] * stride_qt
                   + offs_d[None, :] * stride_qd)
        b_do = tl.load(do_ptrs, mask=mask_qg[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # Load LSE, delta: [BLOCK_QG]
        lse_off = (b_idx * h_q_total + q_head_idx) * T + q_pos
        b_lse = tl.load(LSE + lse_off, mask=mask_qg, other=0.0)
        b_delta = tl.load(DELTA + lse_off, mask=mask_qg, other=0.0)

        # Recompute attention: [BLOCK_QG, BLOCK_KV]
        b_s = tl.dot(b_q_scaled, tl.trans(b_k))
        b_p = tl.exp2(b_s - b_lse[:, None] * LOG2E)
        b_p = tl.where(mask_qg[:, None] & valid[None, :], b_p, 0.0)

        # dV += P^T @ dO: [BLOCK_KV, BLOCK_QG] @ [BLOCK_QG, BLOCK_D] -> [BLOCK_KV, BLOCK_D]
        # Sums across all G heads naturally via matmul
        b_dv += tl.dot(tl.trans(b_p).to(tl.float32), b_do)

        # dp = dO @ V^T: [BLOCK_QG, BLOCK_KV]
        b_dp = tl.dot(b_do, tl.trans(b_v))

        # ds = p * (dp - delta): [BLOCK_QG, BLOCK_KV]
        b_ds = b_p * (b_dp - b_delta[:, None])

        # dK += dS^T @ Q * scale: [BLOCK_KV, BLOCK_QG] @ [BLOCK_QG, BLOCK_D] -> [BLOCK_KV, BLOCK_D]
        b_dk += tl.dot(tl.trans(b_ds).to(tl.float32), b_q) * sm_scale

    # Atomic add — same token may be selected by multiple patches
    dk_ptrs = (DK
               + b_idx * stride_dkb
               + kv_idx * stride_dkh
               + flat_indices[:, None] * stride_dkt
               + offs_d[None, :] * stride_dkd)
    dv_ptrs = (DV
               + b_idx * stride_dkb
               + kv_idx * stride_dkh
               + flat_indices[:, None] * stride_dkt
               + offs_d[None, :] * stride_dkd)
    tl.atomic_add(dk_ptrs, b_dk.to(DK.dtype.element_ty), mask=valid[:, None] & mask_d[None, :])
    tl.atomic_add(dv_ptrs, b_dv.to(DV.dtype.element_ty), mask=valid[:, None] & mask_d[None, :])


# ============================================================
# Python Wrappers
# ============================================================

def selection_attn_2d_gqa_forward(
    q: torch.Tensor,         # [B, h_q, T, d]
    k: torch.Tensor,         # [B, h_kv, T, d]
    v: torch.Tensor,         # [B, h_kv, T, d]
    top_idx: torch.Tensor,   # [B, h_kv, top_n] int32
    patch_starts: torch.Tensor,  # [n_patches] int32
    pp: int,                 # tokens per patch (P*P)
    H: int,                  # spatial height
    W: int,                  # spatial width
    P: int,                  # patch_size
    scale: float,
    G: int,                  # group size (h_q // h_kv)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    GQA selection attention forward.

    Returns:
        o:   [B, h_q, T, d]
        lse: [B, h_q, T]
    """
    B, h_q, T, d = q.shape
    h_kv = h_q // G
    top_n = top_idx.shape[2]

    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert top_idx.is_contiguous()
    assert h_q == h_kv * G, f"h_q={h_q} must equal h_kv*G={h_kv}*{G}"

    o = torch.empty_like(q)
    lse = torch.empty(B, h_q, T, dtype=torch.float32, device=q.device)

    BLOCK_Q = 16
    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    BLOCK_QG = BLOCK_Q * G
    LOG2E = 1.4426950408889634

    num_warps = _select_num_warps(BLOCK_Q, G)
    grid = (triton.cdiv(T, BLOCK_Q), B * h_kv)

    _sel_gqa_fwd_kernel[grid](
        q, k, v, o, lse,
        top_idx, patch_starts,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        top_idx.stride(0), top_idx.stride(1), top_idx.stride(2),
        scale,
        T=T, D=d, W_spatial=W, P=P, PP=pp,
        TOP_N=top_n, H_KV=h_kv, G=G,
        BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D, BLOCK_KV=BLOCK_KV,
        BLOCK_QG=BLOCK_QG, LOG2E=LOG2E,
        num_warps=num_warps,
    )

    return o, lse


def selection_gqa_bwd_preprocess(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """Compute delta[b,h_q,t] = sum_d(O * dO). Grid over h_q heads."""
    B, h_q, T, d = o.shape
    delta = torch.empty(B * h_q, T, dtype=torch.float32, device=o.device)
    BLOCK_D = max(16, _next_power_of_2(d))

    grid = (T, B * h_q)
    _sel_gqa_bwd_preprocess_kernel[grid](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        T=T, D=d, BLOCK_D=BLOCK_D,
    )
    return delta  # [B*h_q, T]


def selection_gqa_bwd_dq(
    q, k, v, do, lse, delta,
    top_idx, patch_starts,
    pp, H, W, P, scale, G,
):
    """Compute dQ using group-centric query-stationary kernel."""
    B, h_q, T, d = q.shape
    h_kv = h_q // G
    top_n = top_idx.shape[2]
    dq = torch.zeros_like(q)

    BLOCK_Q = 16
    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    BLOCK_QG = BLOCK_Q * G
    LOG2E = 1.4426950408889634
    num_warps = _select_num_warps(BLOCK_Q, G)

    grid = (triton.cdiv(T, BLOCK_Q), B * h_kv)
    _sel_gqa_bwd_dq_kernel[grid](
        q, k, v, do, lse.view(B * h_q, T), delta,
        dq, top_idx, patch_starts,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        top_idx.stride(0), top_idx.stride(1), top_idx.stride(2),
        scale,
        T=T, D=d, W_spatial=W, P=P, PP=pp,
        TOP_N=top_n, H_KV=h_kv, G=G,
        BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D, BLOCK_KV=BLOCK_KV,
        BLOCK_QG=BLOCK_QG, LOG2E=LOG2E,
        num_warps=num_warps,
    )
    return dq


def selection_gqa_bwd_dkv(
    q, k, v, do, lse, delta,
    top_idx, patch_starts,
    pp, H, W, P, scale, G,
):
    """Compute dK, dV using GQA KV-stationary kernel with atomics."""
    B, h_q, T, d = q.shape
    h_kv = h_q // G
    top_n = top_idx.shape[2]

    # bf16 atomics not universally supported; use fp32 for bf16 inputs, fp16 for fp16
    accum_dtype = torch.float16 if k.dtype == torch.float16 else torch.float32
    dk = torch.zeros(B, h_kv, T, d, dtype=accum_dtype, device=k.device)
    dv = torch.zeros(B, h_kv, T, d, dtype=accum_dtype, device=v.device)

    BLOCK_Q = 16
    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    BLOCK_QG = BLOCK_Q * G
    LOG2E = 1.4426950408889634
    num_warps = _select_num_warps(BLOCK_Q, G)

    grid = (1, B * h_kv * top_n)
    _sel_gqa_bwd_dkv_kernel[grid](
        q, k, v, do, lse.view(B * h_q, T), delta,
        dk, dv, top_idx, patch_starts,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        top_idx.stride(0), top_idx.stride(1), top_idx.stride(2),
        scale,
        T=T, D=d, W_spatial=W, P=P, PP=pp,
        TOP_N=top_n, H_KV=h_kv, G=G,
        BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D, BLOCK_KV=BLOCK_KV,
        BLOCK_QG=BLOCK_QG, LOG2E=LOG2E,
        num_warps=num_warps,
    )
    if dk.dtype != k.dtype:
        dk = dk.to(k.dtype)
    if dv.dtype != v.dtype:
        dv = dv.to(v.dtype)
    return dk, dv


# ============================================================
# Autograd Function
# ============================================================

class SelectionAttn2DGQA(torch.autograd.Function):
    """Custom autograd for GQA Triton-accelerated 2D selection attention."""

    @staticmethod
    def forward(ctx, q, k, v, top_idx, patch_starts, pp, H, W, P, scale, G):
        o, lse = selection_attn_2d_gqa_forward(
            q, k, v, top_idx, patch_starts, pp, H, W, P, scale, G,
        )
        ctx.save_for_backward(q, k, v, o, lse, top_idx, patch_starts)
        ctx.pp = pp
        ctx.H = H
        ctx.W = W
        ctx.P = P
        ctx.scale = scale
        ctx.G = G
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, top_idx, patch_starts = ctx.saved_tensors
        do = do.contiguous()

        delta = selection_gqa_bwd_preprocess(o, do)
        dq = selection_gqa_bwd_dq(
            q, k, v, do, lse, delta,
            top_idx, patch_starts,
            ctx.pp, ctx.H, ctx.W, ctx.P, ctx.scale, ctx.G,
        )
        dk, dv = selection_gqa_bwd_dkv(
            q, k, v, do, lse, delta,
            top_idx, patch_starts,
            ctx.pp, ctx.H, ctx.W, ctx.P, ctx.scale, ctx.G,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None

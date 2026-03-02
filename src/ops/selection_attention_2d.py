"""
Triton-accelerated 2D selection attention for NSA.

Implements the selection branch: given top-n selected patch indices,
each query token attends to ALL tokens within those selected patches
using dynamic block indexing and online softmax — without materializing
gathered K/V tensors.

Forward:  triton_sel_fwd_kernel
Backward: triton_sel_bwd_preprocess_kernel, triton_sel_bwd_dq_kernel, triton_sel_bwd_dkv_kernel
Autograd: SelectionAttn2D (torch.autograd.Function)
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
    # On A6000, bwd showed better performance with BLOCK_Q=32 for production settings.
    selected_tokens = top_n * pp
    return 32 if (d <= 64 or selected_tokens >= 256) else 64


def _select_num_warps(block_q: int) -> int:
    return 4 if block_q <= 32 else 8


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
def _sel_fwd_kernel(
    Q, K, V, O, LSE,
    top_idx, patch_starts,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_lb, stride_lh, stride_lt,
    sm_scale,
    T: tl.constexpr,
    D: tl.constexpr,
    W_spatial: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    TOP_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """
    Block-vectorized selection attention forward kernel.

    Grid: (cdiv(T, BLOCK_Q), B * h)

    Loads all PP tokens per patch at once and uses tl.dot for
    Q@K^T and P@V matrix multiplies (tensor core compatible).
    Inner loop: TOP_N iterations instead of TOP_N * PP.
    """
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_bh = pid_bh

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_q = offs_q < T
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Load Q: [BLOCK_Q, BLOCK_D]
    q_ptrs = Q + off_bh * stride_qh + offs_q[:, None] * stride_qt + offs_d[None, :] * stride_qd
    b_q = tl.load(q_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    b_q = b_q * (LOG2E * sm_scale)

    # Online softmax accumulators
    b_m = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BLOCK_Q], dtype=tl.float32)
    b_o = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    num_heads = stride_qb // stride_qh
    b_idx = off_bh // num_heads

    # Pre-compute local offsets within a patch: [BLOCK_KV]
    offs_local = tl.arange(0, BLOCK_KV)
    local_row = offs_local // P
    local_col = offs_local % P
    mask_pp = offs_local < PP

    for i in range(TOP_N):
        patch_idx = tl.load(top_idx + b_idx * TOP_N + i)
        kv_base = tl.load(patch_starts + patch_idx)

        # Flat token indices for all PP tokens in this patch
        flat_indices = kv_base + local_row * W_spatial + local_col  # [BLOCK_KV]
        valid = mask_pp & (flat_indices < T)

        # Load K block: [BLOCK_KV, BLOCK_D]
        k_ptrs = K + off_bh * stride_kh + flat_indices[:, None] * stride_kt + offs_d[None, :] * stride_kd
        b_k = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # Load V block: [BLOCK_KV, BLOCK_D]
        v_ptrs = V + off_bh * stride_vh + flat_indices[:, None] * stride_vt + offs_d[None, :] * stride_vd
        b_v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # Attention scores: [BLOCK_Q, BLOCK_KV] = Q @ K^T
        b_s = tl.dot(b_q, tl.trans(b_k))
        b_s = tl.where(valid[None, :], b_s, float("-inf"))

        # Online softmax update (block version)
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

    # Store O
    o_ptrs = O + off_bh * stride_oh + offs_q[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, b_o.to(O.dtype.element_ty), mask=mask_q[:, None] & mask_d[None, :])

    # Store LSE
    lse_ptrs = LSE + off_bh * stride_lh + offs_q * stride_lt
    tl.store(lse_ptrs, b_lse, mask=mask_q)


def selection_attn_2d_forward(
    q: torch.Tensor,     # [B, h, T, d]
    k: torch.Tensor,     # [B, h, T, d]
    v: torch.Tensor,     # [B, h, T, d]
    top_idx: torch.Tensor,      # [B, top_n] int32
    patch_starts: torch.Tensor,  # [n_patches] int32
    pp: int,             # tokens per patch (P*P)
    H: int,              # spatial height
    W: int,              # spatial width
    P: int,              # patch_size
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Selection attention forward.

    Returns:
        o: [B, h, T, d]
        lse: [B, h, T]
    """
    B, h, T, d = q.shape
    top_n = top_idx.shape[1]

    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()

    o = torch.empty_like(q)
    lse = torch.empty(B, h, T, dtype=torch.float32, device=q.device)

    BLOCK_Q = 64
    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    LOG2E = 1.4426950408889634  # 1 / ln(2)

    grid = (triton.cdiv(T, BLOCK_Q), B * h)

    _sel_fwd_kernel[grid](
        q, k, v, o, lse,
        top_idx, patch_starts,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        lse.stride(0), lse.stride(1), lse.stride(2),
        scale,
        T=T,
        D=d,
        W_spatial=W,
        P=P,
        PP=pp,
        TOP_N=top_n,
        BLOCK_Q=BLOCK_Q,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV,
        LOG2E=LOG2E,
    )

    return o, lse


# ============================================================
# Backward Kernels
# ============================================================

@triton.jit
def _sel_bwd_preprocess_kernel(
    O, DO, DELTA,
    stride_ob, stride_oh, stride_ot, stride_od,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute delta[b,h,t] = sum_d(O[b,h,t,d] * dO[b,h,t,d])."""
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


@triton.jit
def _sel_bwd_dq_kernel(
    Q, K, V, DO, LSE, DELTA, DQ,
    top_idx, patch_starts,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    sm_scale,
    T: tl.constexpr,
    D: tl.constexpr,
    W_spatial: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    TOP_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """Block-vectorized query-stationary dQ kernel."""
    pid_q = tl.program_id(0)
    pid_bh = tl.program_id(1)
    off_bh = pid_bh

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    mask_q = offs_q < T
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    num_heads = stride_qb // stride_qh
    b_idx = off_bh // num_heads

    # Load Q
    q_ptrs = Q + off_bh * stride_qh + offs_q[:, None] * stride_qt + offs_d[None, :] * stride_qd
    b_q = tl.load(q_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    b_q_scaled = b_q * (LOG2E * sm_scale)

    # Load dO
    do_ptrs = DO + off_bh * stride_qh + offs_q[:, None] * stride_qt + offs_d[None, :] * stride_qd
    b_do = tl.load(do_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # Load LSE and delta
    lse_ptrs = LSE + off_bh * T + offs_q
    b_lse = tl.load(lse_ptrs, mask=mask_q, other=0.0)
    delta_ptrs = DELTA + off_bh * T + offs_q
    b_delta = tl.load(delta_ptrs, mask=mask_q, other=0.0)

    b_dq = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    # Pre-compute local offsets within a patch
    offs_local = tl.arange(0, BLOCK_KV)
    local_row = offs_local // P
    local_col = offs_local % P
    mask_pp = offs_local < PP

    for i in range(TOP_N):
        patch_idx = tl.load(top_idx + b_idx * TOP_N + i)
        kv_base = tl.load(patch_starts + patch_idx)

        flat_indices = kv_base + local_row * W_spatial + local_col
        valid = mask_pp & (flat_indices < T)

        # Load K, V blocks: [BLOCK_KV, BLOCK_D]
        k_ptrs = K + off_bh * stride_kh + flat_indices[:, None] * stride_kt + offs_d[None, :] * stride_kd
        b_k = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = V + off_bh * stride_vh + flat_indices[:, None] * stride_vt + offs_d[None, :] * stride_vd
        b_v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # Recompute attention: [BLOCK_Q, BLOCK_KV]
        b_s = tl.dot(b_q_scaled, tl.trans(b_k))
        b_p = tl.exp2(b_s - b_lse[:, None] * LOG2E)
        b_p = tl.where(valid[None, :], b_p, 0.0)

        # dp = dO @ V^T: [BLOCK_Q, BLOCK_KV]
        b_dp = tl.dot(b_do, tl.trans(b_v))

        # ds = p * (dp - delta): [BLOCK_Q, BLOCK_KV]
        b_ds = b_p * (b_dp - b_delta[:, None])

        # dQ += ds @ K * scale: [BLOCK_Q, BLOCK_D]
        b_dq += tl.dot(b_ds.to(tl.float32), b_k) * sm_scale

    # Store dQ
    dq_ptrs = DQ + off_bh * stride_qh + offs_q[:, None] * stride_qt + offs_d[None, :] * stride_qd
    tl.store(dq_ptrs, b_dq.to(DQ.dtype.element_ty), mask=mask_q[:, None] & mask_d[None, :])


@triton.jit
def _sel_bwd_dkv_kernel(
    Q, K, V, DO, LSE, DELTA, DK, DV,
    top_idx, patch_starts,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_dkb, stride_dkh, stride_dkt, stride_dkd,
    sm_scale,
    T: tl.constexpr,
    D: tl.constexpr,
    W_spatial: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    TOP_N: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """
    Block-vectorized KV-stationary backward kernel.

    Grid: (1, B * h * TOP_N)

    Each program computes dK, dV for ALL PP tokens in one selected patch
    by iterating over query blocks. Uses tl.dot for block matmuls.
    """
    pid_bh_n = tl.program_id(1)

    num_heads = stride_qb // stride_qh
    n_idx = pid_bh_n % TOP_N
    off_bh = pid_bh_n // TOP_N
    b_idx = off_bh // num_heads

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    # Compute flat indices for all tokens in this patch
    offs_local = tl.arange(0, BLOCK_KV)
    local_row = offs_local // P
    local_col = offs_local % P
    mask_pp = offs_local < PP

    patch_idx = tl.load(top_idx + b_idx * TOP_N + n_idx)
    kv_base = tl.load(patch_starts + patch_idx)
    flat_indices = kv_base + local_row * W_spatial + local_col  # [BLOCK_KV]
    valid = mask_pp & (flat_indices < T)

    # Load K, V for all tokens in this patch: [BLOCK_KV, BLOCK_D]
    k_ptrs = K + off_bh * stride_kh + flat_indices[:, None] * stride_kt + offs_d[None, :] * stride_kd
    b_k = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    v_ptrs = V + off_bh * stride_vh + flat_indices[:, None] * stride_vt + offs_d[None, :] * stride_vd
    b_v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # Accumulators: [BLOCK_KV, BLOCK_D]
    b_dk = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)
    b_dv = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)

    # Iterate over all query positions in blocks
    for q_start in range(0, T, BLOCK_Q):
        offs_q = q_start + tl.arange(0, BLOCK_Q)
        mask_q = offs_q < T

        # Load Q block: [BLOCK_Q, BLOCK_D]
        q_ptrs = Q + off_bh * stride_qh + offs_q[:, None] * stride_qt + offs_d[None, :] * stride_qd
        b_q = tl.load(q_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        b_q_scaled = b_q * (LOG2E * sm_scale)

        # Load dO: [BLOCK_Q, BLOCK_D]
        do_ptrs = DO + off_bh * stride_qh + offs_q[:, None] * stride_qt + offs_d[None, :] * stride_qd
        b_do = tl.load(do_ptrs, mask=mask_q[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # Load LSE, delta
        lse_ptrs = LSE + off_bh * T + offs_q
        b_lse = tl.load(lse_ptrs, mask=mask_q, other=0.0)
        delta_ptrs = DELTA + off_bh * T + offs_q
        b_delta = tl.load(delta_ptrs, mask=mask_q, other=0.0)

        # Recompute attention: [BLOCK_Q, BLOCK_KV] = Q_scaled @ K^T
        b_s = tl.dot(b_q_scaled, tl.trans(b_k))
        b_p = tl.exp2(b_s - b_lse[:, None] * LOG2E)
        b_p = tl.where(mask_q[:, None] & valid[None, :], b_p, 0.0)

        # dV += p^T @ dO: [BLOCK_KV, BLOCK_Q] @ [BLOCK_Q, BLOCK_D] -> [BLOCK_KV, BLOCK_D]
        b_dv += tl.dot(tl.trans(b_p).to(tl.float32), b_do)

        # dp = dO @ V^T: [BLOCK_Q, BLOCK_KV]
        b_dp = tl.dot(b_do, tl.trans(b_v))

        # ds = p * (dp - delta): [BLOCK_Q, BLOCK_KV]
        b_ds = b_p * (b_dp - b_delta[:, None])

        # dK += ds^T @ Q * scale: [BLOCK_KV, BLOCK_Q] @ [BLOCK_Q, BLOCK_D] -> [BLOCK_KV, BLOCK_D]
        b_dk += tl.dot(tl.trans(b_ds).to(tl.float32), b_q) * sm_scale

    # Atomic add all PP tokens — 2D pointers with 2D mask
    dk_ptrs = DK + off_bh * stride_dkh + flat_indices[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    dv_ptrs = DV + off_bh * stride_dkh + flat_indices[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
    tl.atomic_add(dk_ptrs, b_dk.to(DK.dtype.element_ty), mask=valid[:, None] & mask_d[None, :])
    tl.atomic_add(dv_ptrs, b_dv.to(DV.dtype.element_ty), mask=valid[:, None] & mask_d[None, :])


# ============================================================
# Python wrappers
# ============================================================

def selection_bwd_preprocess(o: torch.Tensor, do: torch.Tensor) -> torch.Tensor:
    """Compute delta[b,h,t] = sum_d(O * dO)."""
    B, h, T, d = o.shape
    delta = torch.empty(B * h, T, dtype=torch.float32, device=o.device)
    BLOCK_D = max(16, _next_power_of_2(d))

    grid = (T, B * h)
    _sel_bwd_preprocess_kernel[grid](
        o, do, delta,
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        T=T, D=d, BLOCK_D=BLOCK_D,
    )
    return delta  # [B*h, T]


def selection_bwd_dq(
    q, k, v, do, lse, delta,
    top_idx, patch_starts,
    pp, H, W, P, scale,
):
    """Compute dQ using query-stationary kernel."""
    B, h, T, d = q.shape
    top_n = top_idx.shape[1]
    dq = torch.zeros_like(q)

    BLOCK_Q = _select_block_q_backward(d, top_n, pp)
    BLOCK_D = max(16, _next_power_of_2(d))
    LOG2E = 1.4426950408889634
    BLOCK_KV = max(16, _next_power_of_2(pp))
    num_warps = _select_num_warps(BLOCK_Q)

    grid = (triton.cdiv(T, BLOCK_Q), B * h)
    _sel_bwd_dq_kernel[grid](
        q, k, v, do, lse.view(B * h, T), delta,
        dq, top_idx, patch_starts,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        scale,
        T=T, D=d, W_spatial=W, P=P, PP=pp,
        TOP_N=top_n, BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV, LOG2E=LOG2E,
        num_warps=num_warps,
    )
    return dq


def selection_bwd_dkv(
    q, k, v, do, lse, delta,
    top_idx, patch_starts,
    pp, H, W, P, scale,
):
    """Compute dK, dV using KV-stationary kernel."""
    B, h, T, d = q.shape
    top_n = top_idx.shape[1]

    # Use fp16 accumulators for fp16 inputs to reduce memory pressure; keep fp32 otherwise.
    # bf16 atomics are not universally supported, so bf16 keeps fp32 accumulation.
    accum_dtype = torch.float16 if k.dtype == torch.float16 else torch.float32
    dk = torch.zeros(B, h, T, d, dtype=accum_dtype, device=k.device)
    dv = torch.zeros(B, h, T, d, dtype=accum_dtype, device=v.device)

    BLOCK_Q = _select_block_q_backward(d, top_n, pp)
    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    LOG2E = 1.4426950408889634
    num_warps = _select_num_warps(BLOCK_Q)

    grid = (1, B * h * top_n)
    _sel_bwd_dkv_kernel[grid](
        q, k, v, do, lse.view(B * h, T), delta,
        dk, dv, top_idx, patch_starts,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        scale,
        T=T, D=d, W_spatial=W, P=P, PP=pp,
        TOP_N=top_n, BLOCK_Q=BLOCK_Q, BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV, LOG2E=LOG2E,
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

class SelectionAttn2D(torch.autograd.Function):
    """Custom autograd for Triton-accelerated 2D selection attention."""

    @staticmethod
    def forward(ctx, q, k, v, top_idx, patch_starts, pp, H, W, P, scale):
        o, lse = selection_attn_2d_forward(q, k, v, top_idx, patch_starts, pp, H, W, P, scale)
        ctx.save_for_backward(q, k, v, o, lse, top_idx, patch_starts)
        ctx.pp = pp
        ctx.H = H
        ctx.W = W
        ctx.P = P
        ctx.scale = scale
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, top_idx, patch_starts = ctx.saved_tensors
        do = do.contiguous()

        delta = selection_bwd_preprocess(o, do)
        dq = selection_bwd_dq(
            q, k, v, do, lse, delta,
            top_idx, patch_starts,
            ctx.pp, ctx.H, ctx.W, ctx.P, ctx.scale,
        )
        dk, dv = selection_bwd_dkv(
            q, k, v, do, lse, delta,
            top_idx, patch_starts,
            ctx.pp, ctx.H, ctx.W, ctx.P, ctx.scale,
        )
        return dq, dk, dv, None, None, None, None, None, None, None

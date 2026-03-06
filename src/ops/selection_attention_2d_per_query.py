"""
Per-query Triton-accelerated 2D selection attention for NSA.

This op enforces per-query selected patch ids:
  block_idx: [B, h_kv, T, top_n] (int32)

Forward and backward (dQ/dK/dV) are Triton kernels.
"""

from __future__ import annotations

import os
import math

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def _next_power_of_2(n: int) -> int:
    n = max(n, 1)
    return 1 << (n - 1).bit_length()


def _select_num_warps_per_query(block_g: int, block_kv: int, block_d: int) -> int:
    work = block_g * block_kv * max(block_d, 8)
    if work <= 1024:
        return 2
    if work <= 4096:
        return 4
    return 8


def _select_num_warps_per_query_dkv(block_g: int, block_kv: int, block_d: int, block_q: int) -> int:
    # dK/dV is atomics-heavy; lower warp counts are typically faster than
    # occupancy-maximizing choices on A6000 for NSA per-query workloads.
    if block_d <= 32:
        return 2
    if block_d <= 64:
        return 4
    return 8


def _build_active_query_index_per_patch(
    block_idx: torch.Tensor,  # [B, h_kv, T, top_n] int32
    n_patches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build compact active-query lists grouped by (batch, kv_head, patch_id).

    Returns:
      query_idx_sorted: [B*h_kv*T*top_n] int32, grouped by global patch id
      cu_counts: [B*h_kv*n_patches + 1] int32 prefix sums for group ranges

    Group id layout:
      group = ((b * h_kv) + kv) * n_patches + patch_id
    """
    B, h_kv, T, top_n = block_idx.shape
    bh = B * h_kv
    device = block_idx.device
    slots_per_head = T * top_n

    # Flatten patch ids for all (b, kv) heads and build unique global ids.
    patch_flat = block_idx.reshape(-1).to(torch.int64)
    bh_offsets = torch.arange(bh, device=device, dtype=torch.int64).repeat_interleave(slots_per_head)
    global_patch = patch_flat + bh_offsets * int(n_patches)

    # Flat query indices corresponding to block_idx flatten order.
    # For a single head row, flatten order is [t0,n0..nK, t1,n0..nK, ...],
    # so query index is floor(flat_slot / top_n).
    query_template = (torch.arange(slots_per_head, device=device, dtype=torch.int64) // top_n)
    query_flat = query_template.repeat(bh)

    # Group queries by global patch id.
    order = torch.argsort(global_patch)
    query_idx_sorted = query_flat[order].to(torch.int32).contiguous()

    # Prefix sums (counts per group) for random-access group ranges in Triton.
    counts = torch.bincount(global_patch, minlength=bh * int(n_patches))
    cu_counts = torch.empty(bh * int(n_patches) + 1, device=device, dtype=torch.int32)
    cu_counts[0] = 0
    cu_counts[1:] = torch.cumsum(counts, dim=0, dtype=torch.int64).to(torch.int32)
    return query_idx_sorted, cu_counts.contiguous()


def _build_packed_patch_metadata(
    block_idx: torch.Tensor,  # [B, h_kv, T, top_n] int32
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-(batch, kv-head) packed patch tables for selection packing.

    Returns:
      unique_patch_ids: [total_unique] int32, concatenated per-head sorted patch ids
      cu_unique_counts: [B*h_kv + 1] int32, prefix sums into unique_patch_ids
      packed_idx: [B, h_kv, T, top_n] int32, local remap into each head's patch table
    """
    B, h_kv, T, top_n = block_idx.shape
    rows = block_idx.view(B * h_kv, T * top_n)

    unique_chunks: list[torch.Tensor] = []
    inverse_chunks: list[torch.Tensor] = []
    counts = [0]

    for row in rows:
        unique_ids, inverse = torch.unique(row, sorted=True, return_inverse=True)
        unique_chunks.append(unique_ids.to(torch.int32))
        inverse_chunks.append(inverse.to(torch.int32))
        counts.append(counts[-1] + int(unique_ids.numel()))

    unique_patch_ids = torch.cat(unique_chunks, dim=0).contiguous()
    cu_unique_counts = torch.tensor(counts, device=block_idx.device, dtype=torch.int32)
    packed_idx = torch.stack(inverse_chunks, dim=0).view(B, h_kv, T, top_n).contiguous()
    return unique_patch_ids, cu_unique_counts, packed_idx


def _bhtd_to_patch_table(
    t: torch.Tensor,  # [B, h_kv, T, d]
    H: int,
    W: int,
    P: int,
) -> torch.Tensor:
    """Convert [B, h_kv, T, d] tokens into [B*h_kv, n_patches, pp, d] patch tables."""
    B, h_kv, _, d = t.shape
    nH, nW = H // P, W // P
    pp = P * P
    n_patches = nH * nW
    t_2d = t.view(B, h_kv, H, W, d)
    t_patches = t_2d.view(B, h_kv, nH, P, nW, P, d).permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    return t_patches.view(B * h_kv, n_patches, pp, d)


def _gather_packed_patch_tables(
    k: torch.Tensor,  # [B, h_kv, T, d]
    v: torch.Tensor,  # [B, h_kv, T, d]
    unique_patch_ids: torch.Tensor,  # [total_unique] int32
    cu_unique_counts: torch.Tensor,  # [B*h_kv + 1] int32
    H: int,
    W: int,
    P: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather unique selected patches into contiguous packed K/V tables.

    Returns:
      packed_k: [total_unique, pp, d]
      packed_v: [total_unique, pp, d]
    """
    k_patches = _bhtd_to_patch_table(k, H, W, P)
    v_patches = _bhtd_to_patch_table(v, H, W, P)

    B, h_kv, _, _ = k.shape
    packed_k_chunks: list[torch.Tensor] = []
    packed_v_chunks: list[torch.Tensor] = []
    for bh in range(B * h_kv):
        start = int(cu_unique_counts[bh].item())
        end = int(cu_unique_counts[bh + 1].item())
        patch_ids = unique_patch_ids[start:end].to(torch.long)
        packed_k_chunks.append(k_patches[bh].index_select(0, patch_ids))
        packed_v_chunks.append(v_patches[bh].index_select(0, patch_ids))

    packed_k = torch.cat(packed_k_chunks, dim=0).contiguous()
    packed_v = torch.cat(packed_v_chunks, dim=0).contiguous()
    return packed_k, packed_v


def make_patch_starts(H: int, W: int, patch_size: int, device: torch.device) -> torch.Tensor:
    """Return flat start offsets (row-major) for each patch top-left token."""
    p = patch_size
    nH, nW = H // p, W // p
    ph = torch.arange(nH, device=device)
    pw = torch.arange(nW, device=device)
    starts = (ph[:, None] * p * W + pw[None, :] * p).reshape(-1)
    return starts.to(torch.int32)


@triton.jit
def _sel_perq_fwd_kernel(
    Q,
    K,
    V,
    O,
    LSE,
    BLOCK_IDX,
    PATCH_STARTS,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lt,
    stride_ib,
    stride_ih,
    stride_it,
    stride_in,
    sm_scale,
    T: tl.constexpr,
    D: tl.constexpr,
    W_spatial: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    TOP_N: tl.constexpr,
    H_KV: tl.constexpr,
    G: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """
    Query-centric per-query forward kernel.

    Grid: (T, B * h_kv)
      pid_t = query token index
      pid_bh = batch+kv-head index
    """
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)

    kv_idx = pid_bh % H_KV
    b_idx = pid_bh // H_KV

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    q_head_idx = kv_idx * G + offs_g

    # Load Q for all group heads at this query token: [G, D]
    q_ptrs = (
        Q
        + b_idx * stride_qb
        + q_head_idx[:, None] * stride_qh
        + pid_t * stride_qt
        + offs_d[None, :] * stride_qd
    )
    b_q = tl.load(q_ptrs, mask=mask_g[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    b_q = b_q * (LOG2E * sm_scale)

    b_m = tl.full([BLOCK_G], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BLOCK_G], dtype=tl.float32)
    b_o = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)

    offs_local = tl.arange(0, BLOCK_KV)
    local_row = offs_local // P
    local_col = offs_local % P
    mask_pp = offs_local < PP

    for i in range(TOP_N):
        patch_idx = tl.load(
            BLOCK_IDX
            + b_idx * stride_ib
            + kv_idx * stride_ih
            + pid_t * stride_it
            + i * stride_in
        )
        kv_base = tl.load(PATCH_STARTS + patch_idx)

        flat_indices = kv_base + local_row * W_spatial + local_col
        valid = mask_pp & (flat_indices < T)

        k_ptrs = (
            K
            + b_idx * stride_kb
            + kv_idx * stride_kh
            + flat_indices[:, None] * stride_kt
            + offs_d[None, :] * stride_kd
        )
        b_k = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = (
            V
            + b_idx * stride_vb
            + kv_idx * stride_vh
            + flat_indices[:, None] * stride_vt
            + offs_d[None, :] * stride_vd
        )
        b_v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        b_s = tl.dot(b_q, tl.trans(b_k))
        b_s = tl.where(valid[None, :], b_s, float("-inf"))

        b_m_new = tl.maximum(b_m, tl.max(b_s, axis=1))
        b_r = tl.exp2(b_m - b_m_new)
        b_p = tl.exp2(b_s - b_m_new[:, None])
        b_p = tl.where(valid[None, :], b_p, 0.0)

        b_acc = b_acc * b_r + tl.sum(b_p, axis=1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(tl.float32), b_v)
        b_m = b_m_new

    b_o = b_o / tl.maximum(b_acc[:, None], 1e-6)
    b_lse = b_m / LOG2E + tl.log(tl.maximum(b_acc, 1e-6))

    o_ptrs = (
        O
        + b_idx * stride_ob
        + q_head_idx[:, None] * stride_oh
        + pid_t * stride_ot
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, b_o.to(O.dtype.element_ty), mask=mask_g[:, None] & mask_d[None, :])

    lse_ptrs = LSE + b_idx * stride_lb + q_head_idx * stride_lh + pid_t * stride_lt
    tl.store(lse_ptrs, b_lse, mask=mask_g)


@triton.jit
def _sel_perq_fwd_packed_kernel(
    Q,
    K_PACKED,
    V_PACKED,
    O,
    LSE,
    PACKED_IDX,
    CU_PACKED_COUNTS,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_pkp,
    stride_pkt,
    stride_pkd,
    stride_pvp,
    stride_pvt,
    stride_pvd,
    stride_ob,
    stride_oh,
    stride_ot,
    stride_od,
    stride_lb,
    stride_lh,
    stride_lt,
    stride_ib,
    stride_ih,
    stride_it,
    stride_in,
    sm_scale,
    T: tl.constexpr,
    D: tl.constexpr,
    PP: tl.constexpr,
    TOP_N: tl.constexpr,
    H_KV: tl.constexpr,
    G: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """Query-centric packed forward kernel reading contiguous packed K/V patch tables."""
    pid_t = tl.program_id(0)
    pid_bh = tl.program_id(1)

    kv_idx = pid_bh % H_KV
    b_idx = pid_bh // H_KV
    packed_base = tl.load(CU_PACKED_COUNTS + pid_bh)

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    q_head_idx = kv_idx * G + offs_g

    q_ptrs = (
        Q
        + b_idx * stride_qb
        + q_head_idx[:, None] * stride_qh
        + pid_t * stride_qt
        + offs_d[None, :] * stride_qd
    )
    b_q = tl.load(q_ptrs, mask=mask_g[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    b_q = b_q * (LOG2E * sm_scale)

    b_m = tl.full([BLOCK_G], float("-inf"), dtype=tl.float32)
    b_acc = tl.zeros([BLOCK_G], dtype=tl.float32)
    b_o = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)

    offs_pp = tl.arange(0, BLOCK_KV)
    mask_pp = offs_pp < PP

    for i in range(TOP_N):
        patch_local = tl.load(
            PACKED_IDX
            + b_idx * stride_ib
            + kv_idx * stride_ih
            + pid_t * stride_it
            + i * stride_in
        )
        patch_global = packed_base + patch_local

        k_ptrs = (
            K_PACKED
            + patch_global * stride_pkp
            + offs_pp[:, None] * stride_pkt
            + offs_d[None, :] * stride_pkd
        )
        b_k = tl.load(k_ptrs, mask=mask_pp[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = (
            V_PACKED
            + patch_global * stride_pvp
            + offs_pp[:, None] * stride_pvt
            + offs_d[None, :] * stride_pvd
        )
        b_v = tl.load(v_ptrs, mask=mask_pp[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        b_s = tl.dot(b_q, tl.trans(b_k))
        b_s = tl.where(mask_pp[None, :], b_s, float("-inf"))
        b_m_new = tl.maximum(b_m, tl.max(b_s, axis=1))
        b_r = tl.exp2(b_m - b_m_new)
        b_p = tl.exp2(b_s - b_m_new[:, None])
        b_p = tl.where(mask_pp[None, :], b_p, 0.0)

        b_acc = b_acc * b_r + tl.sum(b_p, axis=1)
        b_o = b_o * b_r[:, None] + tl.dot(b_p.to(tl.float32), b_v)
        b_m = b_m_new

    b_o = b_o / tl.maximum(b_acc[:, None], 1e-6)
    b_lse = b_m / LOG2E + tl.log(tl.maximum(b_acc, 1e-6))

    o_ptrs = (
        O
        + b_idx * stride_ob
        + q_head_idx[:, None] * stride_oh
        + pid_t * stride_ot
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, b_o.to(O.dtype.element_ty), mask=mask_g[:, None] & mask_d[None, :])

    lse_ptrs = LSE + b_idx * stride_lb + q_head_idx * stride_lh + pid_t * stride_lt
    tl.store(lse_ptrs, b_lse, mask=mask_g)


@triton.jit
def _sel_perq_bwd_dq_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DQ,
    BLOCK_IDX,
    PATCH_STARTS,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_db,
    stride_dh,
    stride_dt,
    stride_dd,
    stride_lb,
    stride_lh,
    stride_lt,
    stride_ib,
    stride_ih,
    stride_it,
    stride_in,
    sm_scale,
    T: tl.constexpr,
    D: tl.constexpr,
    W_spatial: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    TOP_N: tl.constexpr,
    H_KV: tl.constexpr,
    G: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """Query-block-centric per-query dQ kernel."""
    pid_bh = tl.program_id(0)
    pid_qblk = tl.program_id(1)

    kv_idx = pid_bh % H_KV
    b_idx = pid_bh // H_KV

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    q_head_idx = kv_idx * G + offs_g

    offs_local = tl.arange(0, BLOCK_KV)
    local_row = offs_local // P
    local_col = offs_local % P
    mask_pp = offs_local < PP

    q_start = pid_qblk * BLOCK_Q
    for q_off in tl.static_range(0, BLOCK_Q):
        pid_t = q_start + q_off
        mask_t = pid_t < T

        q_ptrs = (
            Q
            + b_idx * stride_qb
            + q_head_idx[:, None] * stride_qh
            + pid_t * stride_qt
            + offs_d[None, :] * stride_qd
        )
        b_q = tl.load(q_ptrs, mask=mask_t & mask_g[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        b_q_scaled = b_q * (LOG2E * sm_scale)

        do_ptrs = (
            DO
            + b_idx * stride_qb
            + q_head_idx[:, None] * stride_qh
            + pid_t * stride_qt
            + offs_d[None, :] * stride_qd
        )
        b_do = tl.load(do_ptrs, mask=mask_t & mask_g[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        lse_ptrs = LSE + b_idx * stride_lb + q_head_idx * stride_lh + pid_t * stride_lt
        b_lse = tl.load(lse_ptrs, mask=mask_t & mask_g, other=0.0)
        delta_ptrs = DELTA + b_idx * stride_lb + q_head_idx * stride_lh + pid_t * stride_lt
        b_delta = tl.load(delta_ptrs, mask=mask_t & mask_g, other=0.0)

        b_dq = tl.zeros([BLOCK_G, BLOCK_D], dtype=tl.float32)

        for i in range(TOP_N):
            patch_ptr = (
                BLOCK_IDX
                + b_idx * stride_ib
                + kv_idx * stride_ih
                + pid_t * stride_it
                + i * stride_in
            )
            patch_idx = tl.load(patch_ptr, mask=mask_t, other=0)
            kv_base = tl.load(PATCH_STARTS + patch_idx)
            flat_indices = kv_base + local_row * W_spatial + local_col
            valid = mask_t & mask_pp & (flat_indices < T)

            k_ptrs = (
                K
                + b_idx * stride_kb
                + kv_idx * stride_kh
                + flat_indices[:, None] * stride_kt
                + offs_d[None, :] * stride_kd
            )
            b_k = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            v_ptrs = (
                V
                + b_idx * stride_vb
                + kv_idx * stride_vh
                + flat_indices[:, None] * stride_vt
                + offs_d[None, :] * stride_vd
            )
            b_v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            b_s = tl.dot(b_q_scaled, tl.trans(b_k))
            b_p = tl.exp2(b_s - b_lse[:, None] * LOG2E)
            b_p = tl.where(mask_t & valid[None, :], b_p, 0.0)

            b_dp = tl.dot(b_do, tl.trans(b_v))
            b_ds = b_p * (b_dp - b_delta[:, None])
            b_dq += tl.dot(b_ds.to(tl.float32), b_k) * sm_scale

        dq_ptrs = (
            DQ
            + b_idx * stride_db
            + q_head_idx[:, None] * stride_dh
            + pid_t * stride_dt
            + offs_d[None, :] * stride_dd
        )
        tl.store(dq_ptrs, b_dq.to(DQ.dtype.element_ty), mask=mask_t & mask_g[:, None] & mask_d[None, :])


@triton.jit
def _sel_perq_bwd_dkv_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    BLOCK_IDX,
    PATCH_STARTS,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_dkb,
    stride_dkh,
    stride_dkt,
    stride_dkd,
    stride_dvb,
    stride_dvh,
    stride_dvt,
    stride_dvd,
    stride_lb,
    stride_lh,
    stride_lt,
    stride_ib,
    stride_ih,
    stride_it,
    stride_in,
    sm_scale,
    T: tl.constexpr,
    D: tl.constexpr,
    W_spatial: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    TOP_N: tl.constexpr,
    H_KV: tl.constexpr,
    G: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """
    Query-block + patch-centric per-query dK/dV kernel with atomics.

    Grid: (TOP_N, B * h_kv, ceil_div(T, BLOCK_Q))
      pid_n    = selected patch slot
      pid_bh   = flattened (batch, kv-head)
      pid_qblk = query block index
    """
    pid_n = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_qblk = tl.program_id(2)

    kv_idx = pid_bh % H_KV
    b_idx = pid_bh // H_KV

    offs_g = tl.arange(0, BLOCK_G)
    mask_g = offs_g < G
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    q_head_idx = kv_idx * G + offs_g

    offs_local = tl.arange(0, BLOCK_KV)
    local_row = offs_local // P
    local_col = offs_local % P
    mask_pp = offs_local < PP

    q_start = pid_qblk * BLOCK_Q

    for q_off in tl.static_range(0, BLOCK_Q):
        pid_t = q_start + q_off
        mask_t = pid_t < T

        patch_ptr = (
            BLOCK_IDX
            + b_idx * stride_ib
            + kv_idx * stride_ih
            + pid_t * stride_it
            + pid_n * stride_in
        )
        patch_idx = tl.load(patch_ptr, mask=mask_t, other=0)
        kv_base = tl.load(PATCH_STARTS + patch_idx)
        flat_indices = kv_base + local_row * W_spatial + local_col
        valid = mask_t & mask_pp & (flat_indices < T)

        # Load K/V tokens for this selected patch.
        k_ptrs = (
            K
            + b_idx * stride_kb
            + kv_idx * stride_kh
            + flat_indices[:, None] * stride_kt
            + offs_d[None, :] * stride_kd
        )
        b_k = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        v_ptrs = (
            V
            + b_idx * stride_vb
            + kv_idx * stride_vh
            + flat_indices[:, None] * stride_vt
            + offs_d[None, :] * stride_vd
        )
        b_v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        # Load q/do for all grouped query heads at this query token.
        q_ptrs = (
            Q
            + b_idx * stride_qb
            + q_head_idx[:, None] * stride_qh
            + pid_t * stride_qt
            + offs_d[None, :] * stride_qd
        )
        b_q = tl.load(q_ptrs, mask=mask_t & mask_g[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        b_q_scaled = b_q * (LOG2E * sm_scale)

        do_ptrs = (
            DO
            + b_idx * stride_qb
            + q_head_idx[:, None] * stride_qh
            + pid_t * stride_qt
            + offs_d[None, :] * stride_qd
        )
        b_do = tl.load(do_ptrs, mask=mask_t & mask_g[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        lse_ptrs = LSE + b_idx * stride_lb + q_head_idx * stride_lh + pid_t * stride_lt
        b_lse = tl.load(lse_ptrs, mask=mask_t & mask_g, other=0.0)
        delta_ptrs = DELTA + b_idx * stride_lb + q_head_idx * stride_lh + pid_t * stride_lt
        b_delta = tl.load(delta_ptrs, mask=mask_t & mask_g, other=0.0)

        # Recompute local probabilities for this patch.
        b_s = tl.dot(b_q_scaled, tl.trans(b_k))
        b_p = tl.exp2(b_s - b_lse[:, None] * LOG2E)
        b_p = tl.where(mask_t & mask_g[:, None] & valid[None, :], b_p, 0.0)

        # dV = P^T @ dO
        b_dv = tl.dot(tl.trans(b_p).to(tl.float32), b_do)

        # dK = dS^T @ Q * scale, where dS = P * (dP - delta)
        b_dp = tl.dot(b_do, tl.trans(b_v))
        b_ds = b_p * (b_dp - b_delta[:, None])
        b_dk = tl.dot(tl.trans(b_ds).to(tl.float32), b_q) * sm_scale

        # Atomic accumulations over repeated token selections.
        dk_ptrs = (
            DK
            + b_idx * stride_dkb
            + kv_idx * stride_dkh
            + flat_indices[:, None] * stride_dkt
            + offs_d[None, :] * stride_dkd
        )
        dv_ptrs = (
            DV
            + b_idx * stride_dvb
            + kv_idx * stride_dvh
            + flat_indices[:, None] * stride_dvt
            + offs_d[None, :] * stride_dvd
        )
        tl.atomic_add(dk_ptrs, b_dk.to(DK.dtype.element_ty), mask=valid[:, None] & mask_d[None, :])
        tl.atomic_add(dv_ptrs, b_dv.to(DV.dtype.element_ty), mask=valid[:, None] & mask_d[None, :])


@triton.jit
def _sel_perq_bwd_dkv_compact_kernel(
    Q,
    K,
    V,
    DO,
    LSE,
    DELTA,
    DK,
    DV,
    QUERY_IDX,
    CU_COUNTS,
    PATCH_STARTS,
    stride_qb,
    stride_qh,
    stride_qt,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kt,
    stride_kd,
    stride_vb,
    stride_vh,
    stride_vt,
    stride_vd,
    stride_dob,
    stride_doh,
    stride_dot,
    stride_dod,
    stride_dks,
    stride_dkb,
    stride_dkh,
    stride_dkt,
    stride_dkd,
    stride_dvs,
    stride_dvb,
    stride_dvh,
    stride_dvt,
    stride_dvd,
    stride_lb,
    stride_lh,
    stride_lt,
    sm_scale,
    T: tl.constexpr,
    D: tl.constexpr,
    W_spatial: tl.constexpr,
    P: tl.constexpr,
    PP: tl.constexpr,
    N_PATCHES: tl.constexpr,
    H_KV: tl.constexpr,
    G: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    LOG2E: tl.constexpr,
):
    """
    Compact block-centric dK/dV backward kernel (Tilda-style grouping).

    Grid: (B * h_kv * n_patches, G)
      pid_group: one (batch, kv_head, patch_id) group
      pid_g: one grouped query head (share-head lane)

    Each program computes dK/dV for a single share-head and writes to
    DK/DV buffers with explicit share-head dimension. Caller reduces over G.
    """
    pid_group = tl.program_id(0)
    pid_g = tl.program_id(1)

    off_bh = pid_group // N_PATCHES
    patch_id = pid_group % N_PATCHES

    kv_idx = off_bh % H_KV
    b_idx = off_bh // H_KV
    if pid_g >= G:
        return

    act_q_start = tl.load(CU_COUNTS + pid_group)
    act_q_end = tl.load(CU_COUNTS + pid_group + 1)
    act_q_len = act_q_end - act_q_start
    if act_q_len <= 0:
        return

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_q = tl.arange(0, BLOCK_Q)
    q_head_idx = kv_idx * G + pid_g

    offs_local = tl.arange(0, BLOCK_KV)
    local_row = offs_local // P
    local_col = offs_local % P
    mask_pp = offs_local < PP

    # Patch token indices are fixed for this group.
    kv_base = tl.load(PATCH_STARTS + patch_id)
    flat_indices = kv_base + local_row * W_spatial + local_col
    valid = mask_pp & (flat_indices < T)

    # Load K/V once and keep in SRAM while iterating active queries.
    k_ptrs = (
        K
        + b_idx * stride_kb
        + kv_idx * stride_kh
        + flat_indices[:, None] * stride_kt
        + offs_d[None, :] * stride_kd
    )
    b_k = tl.load(k_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    v_ptrs = (
        V
        + b_idx * stride_vb
        + kv_idx * stride_vh
        + flat_indices[:, None] * stride_vt
        + offs_d[None, :] * stride_vd
    )
    b_v = tl.load(v_ptrs, mask=valid[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    b_dk = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)
    b_dv = tl.zeros([BLOCK_KV, BLOCK_D], dtype=tl.float32)

    # Iterate only queries selecting this patch (tiled for better vectorization).
    for i in range(0, act_q_len, BLOCK_Q):
        q_idx = tl.load(QUERY_IDX + act_q_start + i + offs_q, mask=offs_q < act_q_len - i, other=0).to(tl.int32)
        q_mask = offs_q < act_q_len - i

        q_ptrs = (
            Q
            + b_idx * stride_qb
            + q_head_idx * stride_qh
            + q_idx[:, None] * stride_qt
            + offs_d[None, :] * stride_qd
        )
        b_q = tl.load(q_ptrs, mask=q_mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        b_q_scaled = b_q * (LOG2E * sm_scale)

        do_ptrs = (
            DO
            + b_idx * stride_dob
            + q_head_idx * stride_doh
            + q_idx[:, None] * stride_dot
            + offs_d[None, :] * stride_dod
        )
        b_do = tl.load(do_ptrs, mask=q_mask[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        lse_ptrs = LSE + b_idx * stride_lb + q_head_idx * stride_lh + q_idx * stride_lt
        b_lse = tl.load(lse_ptrs, mask=q_mask, other=0.0)
        delta_ptrs = DELTA + b_idx * stride_lb + q_head_idx * stride_lh + q_idx * stride_lt
        b_delta = tl.load(delta_ptrs, mask=q_mask, other=0.0)

        b_s = tl.dot(b_q_scaled, tl.trans(b_k))
        b_p = tl.exp2(b_s - b_lse[:, None] * LOG2E)
        b_p = tl.where(q_mask[:, None] & valid[None, :], b_p, 0.0)

        b_dv += tl.dot(tl.trans(b_p).to(tl.float32), b_do)

        b_dp = tl.dot(b_do, tl.trans(b_v))
        b_ds = b_p * (b_dp - b_delta[:, None])
        b_dk += tl.dot(tl.trans(b_ds).to(tl.float32), b_q) * sm_scale

    dk_ptrs = (
        DK
        + pid_g * stride_dks
        + b_idx * stride_dkb
        + kv_idx * stride_dkh
        + flat_indices[:, None] * stride_dkt
        + offs_d[None, :] * stride_dkd
    )
    dv_ptrs = (
        DV
        + pid_g * stride_dvs
        + b_idx * stride_dvb
        + kv_idx * stride_dvh
        + flat_indices[:, None] * stride_dvt
        + offs_d[None, :] * stride_dvd
    )
    tl.store(dk_ptrs, b_dk.to(DK.dtype.element_ty), mask=valid[:, None] & mask_d[None, :])
    tl.store(dv_ptrs, b_dv.to(DV.dtype.element_ty), mask=valid[:, None] & mask_d[None, :])


def _selection_reference_per_query(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: torch.Tensor,
    H: int,
    W: int,
    P: int,
    G: int,
    scale: float,
) -> torch.Tensor:
    """Differentiable reference path for temporary dK/dV in Task 4."""
    B, h_q, T, d = q.shape
    h_kv = k.shape[1]
    top_n = block_idx.shape[-1]
    pp = P * P
    selected_tokens = top_n * pp

    nH, nW = H // P, W // P
    n_patches = nH * nW

    k_2d = k.view(B, h_kv, H, W, d)
    k_patches = k_2d.view(B, h_kv, nH, P, nW, P, d).permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    k_patches = k_patches.view(B * h_kv, n_patches, pp, d)

    v_2d = v.view(B, h_kv, H, W, d)
    v_patches = v_2d.view(B, h_kv, nH, P, nW, P, d).permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    v_patches = v_patches.view(B * h_kv, n_patches, pp, d)

    idx_flat = block_idx.view(B * h_kv, T * top_n).to(torch.int64)
    gather_idx = idx_flat[:, :, None, None].expand(B * h_kv, T * top_n, pp, d)
    k_slc = k_patches.gather(1, gather_idx).view(B * h_kv, T, selected_tokens, d)
    v_slc = v_patches.gather(1, gather_idx).view(B * h_kv, T, selected_tokens, d)

    q_grouped = q.view(B, h_kv, G, T, d).reshape(B * h_kv, G, T, d)
    logits = torch.einsum("bgtd,btsd->bgts", q_grouped * scale, k_slc).float()
    attn = F.softmax(logits, dim=-1).to(v_slc.dtype)
    out = torch.einsum("bgts,btsd->bgtd", attn, v_slc)
    return out.reshape(B, h_q, T, d)


def _selection_reference_dkv_chunked(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    block_idx: torch.Tensor,
    H: int,
    W: int,
    P: int,
    G: int,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Chunked reference dK/dV computation to avoid OOM on large T.

    Uses autograd over chunked per-query gather+attention slices and accumulates
    gradients into full-shape dK/dV tensors.
    """
    B, h_q, T, d = q.shape
    h_kv = k.shape[1]
    top_n = block_idx.shape[-1]
    pp = P * P
    selected_tokens = top_n * pp

    nH, nW = H // P, W // P
    n_patches = nH * nW

    k_2d = k.view(B, h_kv, H, W, d)
    k_patches = k_2d.view(B, h_kv, nH, P, nW, P, d).permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    k_patches = k_patches.view(B * h_kv, n_patches, pp, d)

    v_2d = v.view(B, h_kv, H, W, d)
    v_patches = v_2d.view(B, h_kv, nH, P, nW, P, d).permute(0, 1, 2, 4, 3, 5, 6).contiguous()
    v_patches = v_patches.view(B * h_kv, n_patches, pp, d)

    q_grouped = q.view(B, h_kv, G, T, d).reshape(B * h_kv, G, T, d)
    do_grouped = do.view(B, h_kv, G, T, d).reshape(B * h_kv, G, T, d)
    idx_all = block_idx.view(B * h_kv, T, top_n).to(torch.int64)

    if q.is_cuda:
        free_mem = torch.cuda.mem_get_info(q.device)[0]
        bytes_per_scalar = torch.finfo(k.dtype).bits // 8
        bytes_per_t = B * h_kv * selected_tokens * d * bytes_per_scalar * 4
        chunk_size = int((free_mem * 0.20) // max(bytes_per_t, 1))
        chunk_size = max(4, min(T, chunk_size))
        chunk_size = (chunk_size // 4) * 4 or 4
    else:
        chunk_size = min(T, 64)

    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        t_chunk = end - start

        q_chunk = q_grouped[:, :, start:end, :]
        do_chunk = do_grouped[:, :, start:end, :]
        idx_chunk = idx_all[:, start:end, :]

        gather_idx = idx_chunk.reshape(B * h_kv, t_chunk * top_n, 1, 1).expand(B * h_kv, t_chunk * top_n, pp, d)
        k_slc = k_patches.gather(1, gather_idx).reshape(B * h_kv, t_chunk, selected_tokens, d)
        v_slc = v_patches.gather(1, gather_idx).reshape(B * h_kv, t_chunk, selected_tokens, d)

        logits = torch.einsum("bgtd,btsd->bgts", q_chunk * scale, k_slc).float()
        attn = F.softmax(logits, dim=-1).to(v_slc.dtype)
        out_chunk = torch.einsum("bgts,btsd->bgtd", attn, v_slc)

        gk, gv = torch.autograd.grad(
            out_chunk,
            (k, v),
            grad_outputs=do_chunk,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )
        dk = dk + gk
        dv = dv + gv

    return dk, dv


def selection_attn_2d_per_query_forward(
    q: torch.Tensor,  # [B, h_q, T, d]
    k: torch.Tensor,  # [B, h_kv, T, d]
    v: torch.Tensor,  # [B, h_kv, T, d]
    block_idx: torch.Tensor,  # [B, h_kv, T, top_n] int32
    patch_starts: torch.Tensor,  # [n_patches] int32
    pp: int,
    H: int,
    W: int,
    P: int,
    scale: float,
    G: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-query forward wrapper for both MHA (G=1) and GQA (G>1)."""
    B, h_q, T, d = q.shape
    h_kv = k.shape[1]
    top_n = block_idx.shape[-1]

    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert block_idx.is_contiguous()
    assert block_idx.dtype == torch.int32
    assert h_q == h_kv * G, f"h_q={h_q} must equal h_kv*G={h_kv}*{G}"
    assert block_idx.shape[:3] == (B, h_kv, T), f"expected block_idx [B,h_kv,T,n], got {tuple(block_idx.shape)}"

    o = torch.empty_like(q)
    lse = torch.empty(B, h_q, T, dtype=torch.float32, device=q.device)

    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_G = max(16, _next_power_of_2(G))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    LOG2E = 1.4426950408889634
    num_warps = _select_num_warps_per_query(BLOCK_G, BLOCK_KV, BLOCK_D)

    grid = (T, B * h_kv)
    _sel_perq_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        block_idx,
        patch_starts,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        block_idx.stride(0),
        block_idx.stride(1),
        block_idx.stride(2),
        block_idx.stride(3),
        scale,
        T=T,
        D=d,
        W_spatial=W,
        P=P,
        PP=pp,
        TOP_N=top_n,
        H_KV=h_kv,
        G=G,
        BLOCK_G=BLOCK_G,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV,
        LOG2E=LOG2E,
        num_warps=num_warps,
    )
    return o, lse


def selection_attn_2d_per_query_forward_packed(
    q: torch.Tensor,  # [B, h_q, T, d]
    k: torch.Tensor,  # [B, h_kv, T, d]
    v: torch.Tensor,  # [B, h_kv, T, d]
    block_idx: torch.Tensor,  # [B, h_kv, T, top_n] int32
    H: int,
    W: int,
    P: int,
    scale: float,
    G: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward-only packed selection path for CUDA benchmarking and A/B parity checks."""
    B, h_q, T, d = q.shape
    h_kv = k.shape[1]
    top_n = block_idx.shape[-1]
    pp = P * P

    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert block_idx.is_contiguous()
    assert block_idx.dtype == torch.int32
    assert h_q == h_kv * G, f"h_q={h_q} must equal h_kv*G={h_kv}*{G}"

    unique_patch_ids, cu_unique_counts, packed_idx = _build_packed_patch_metadata(block_idx)
    packed_k, packed_v = _gather_packed_patch_tables(k, v, unique_patch_ids, cu_unique_counts, H, W, P)

    o = torch.empty_like(q)
    lse = torch.empty(B, h_q, T, dtype=torch.float32, device=q.device)

    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_G = max(16, _next_power_of_2(G))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    LOG2E = 1.4426950408889634
    num_warps = _select_num_warps_per_query(BLOCK_G, BLOCK_KV, BLOCK_D)

    grid = (T, B * h_kv)
    _sel_perq_fwd_packed_kernel[grid](
        q,
        packed_k,
        packed_v,
        o,
        lse,
        packed_idx,
        cu_unique_counts,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        packed_k.stride(0),
        packed_k.stride(1),
        packed_k.stride(2),
        packed_v.stride(0),
        packed_v.stride(1),
        packed_v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        packed_idx.stride(0),
        packed_idx.stride(1),
        packed_idx.stride(2),
        packed_idx.stride(3),
        scale,
        T=T,
        D=d,
        PP=pp,
        TOP_N=top_n,
        H_KV=h_kv,
        G=G,
        BLOCK_G=BLOCK_G,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV,
        LOG2E=LOG2E,
        num_warps=num_warps,
    )
    return o, lse


def selection_per_query_bwd_dq(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    block_idx: torch.Tensor,
    patch_starts: torch.Tensor,
    pp: int,
    H: int,
    W: int,
    P: int,
    scale: float,
    G: int,
) -> torch.Tensor:
    """Compute dQ with Triton per-query kernel."""
    B, h_q, T, d = q.shape
    h_kv = k.shape[1]
    top_n = block_idx.shape[-1]

    dq = torch.zeros_like(q)
    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_G = max(16, _next_power_of_2(G))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    LOG2E = 1.4426950408889634
    if T >= 4096:
        BLOCK_Q = 32
    elif T >= 512:
        BLOCK_Q = 16
    else:
        BLOCK_Q = 4
    num_warps = _select_num_warps_per_query_dkv(BLOCK_G, BLOCK_KV, BLOCK_D, BLOCK_Q)

    grid = (B * h_kv, triton.cdiv(T, BLOCK_Q))
    _sel_perq_bwd_dq_kernel[grid](
        q,
        k,
        v,
        do,
        lse,
        delta,
        dq,
        block_idx,
        patch_starts,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        dq.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        block_idx.stride(0),
        block_idx.stride(1),
        block_idx.stride(2),
        block_idx.stride(3),
        scale,
        T=T,
        D=d,
        W_spatial=W,
        P=P,
        PP=pp,
        TOP_N=top_n,
        H_KV=h_kv,
        G=G,
        BLOCK_G=BLOCK_G,
        BLOCK_D=BLOCK_D,
        BLOCK_KV=BLOCK_KV,
        BLOCK_Q=BLOCK_Q,
        LOG2E=LOG2E,
        num_warps=num_warps,
    )
    return dq


def selection_per_query_bwd_dkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    block_idx: torch.Tensor,
    patch_starts: torch.Tensor,
    pp: int,
    H: int,
    W: int,
    P: int,
    scale: float,
    G: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dK/dV with Triton per-query kernel."""
    B, h_q, T, d = q.shape
    h_kv = k.shape[1]
    top_n = block_idx.shape[-1]
    n_patches = (H // P) * (W // P)

    accum_dtype = torch.float16 if k.dtype == torch.float16 else torch.float32

    BLOCK_D = max(16, _next_power_of_2(d))
    BLOCK_G = max(16, _next_power_of_2(G))
    BLOCK_KV = max(16, _next_power_of_2(pp))
    LOG2E = 1.4426950408889634
    if T >= 4096:
        BLOCK_Q_LEGACY = 32
        BLOCK_Q_COMPACT = 64
    elif T >= 512:
        BLOCK_Q_LEGACY = 16
        BLOCK_Q_COMPACT = 32
    else:
        BLOCK_Q_LEGACY = 4
        BLOCK_Q_COMPACT = 16
    num_warps_legacy = _select_num_warps_per_query_dkv(BLOCK_G, BLOCK_KV, BLOCK_D, BLOCK_Q_LEGACY)
    num_warps_compact = _select_num_warps_per_query_dkv(1, BLOCK_KV, BLOCK_D, BLOCK_Q_COMPACT)

    use_legacy_atomic = os.getenv("NSA_PERQ_DKV_LEGACY_ATOMIC", "0") == "1"
    if use_legacy_atomic:
        dk = torch.zeros(B, h_kv, T, d, dtype=accum_dtype, device=k.device)
        dv = torch.zeros(B, h_kv, T, d, dtype=accum_dtype, device=v.device)
        grid = (top_n, B * h_kv, triton.cdiv(T, BLOCK_Q_LEGACY))
        _sel_perq_bwd_dkv_kernel[grid](
            q,
            k,
            v,
            do,
            lse,
            delta,
            dk,
            dv,
            block_idx,
            patch_starts,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            dk.stride(0),
            dk.stride(1),
            dk.stride(2),
            dk.stride(3),
            dv.stride(0),
            dv.stride(1),
            dv.stride(2),
            dv.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            block_idx.stride(0),
            block_idx.stride(1),
            block_idx.stride(2),
            block_idx.stride(3),
            scale,
            T=T,
            D=d,
            W_spatial=W,
            P=P,
            PP=pp,
            TOP_N=top_n,
            H_KV=h_kv,
            G=G,
            BLOCK_G=BLOCK_G,
            BLOCK_D=BLOCK_D,
            BLOCK_KV=BLOCK_KV,
            BLOCK_Q=BLOCK_Q_LEGACY,
            LOG2E=LOG2E,
            num_warps=num_warps_legacy,
        )
    else:
        # Compact active-query lists per (batch, kv_head, patch_id), then run
        # one block-centric program per (group, share-head) and reduce over G.
        query_idx_sorted, cu_counts = _build_active_query_index_per_patch(block_idx, n_patches)
        dk_sh = torch.zeros(G, B, h_kv, T, d, dtype=accum_dtype, device=k.device)
        dv_sh = torch.zeros(G, B, h_kv, T, d, dtype=accum_dtype, device=v.device)
        grid = (B * h_kv * n_patches, G)
        _sel_perq_bwd_dkv_compact_kernel[grid](
            q,
            k,
            v,
            do,
            lse,
            delta,
            dk_sh,
            dv_sh,
            query_idx_sorted,
            cu_counts,
            patch_starts,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            do.stride(0),
            do.stride(1),
            do.stride(2),
            do.stride(3),
            dk_sh.stride(0),
            dk_sh.stride(1),
            dk_sh.stride(2),
            dk_sh.stride(3),
            dk_sh.stride(4),
            dv_sh.stride(0),
            dv_sh.stride(1),
            dv_sh.stride(2),
            dv_sh.stride(3),
            dv_sh.stride(4),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            scale,
            T=T,
            D=d,
            W_spatial=W,
            P=P,
            PP=pp,
            N_PATCHES=n_patches,
            H_KV=h_kv,
            G=G,
            BLOCK_D=BLOCK_D,
            BLOCK_KV=BLOCK_KV,
            BLOCK_Q=BLOCK_Q_COMPACT,
            LOG2E=LOG2E,
            num_warps=num_warps_compact,
        )
        dk = dk_sh.sum(dim=0)
        dv = dv_sh.sum(dim=0)

    if dk.dtype != k.dtype:
        dk = dk.to(k.dtype)
    if dv.dtype != v.dtype:
        dv = dv.to(v.dtype)
    return dk, dv


class SelectionAttn2DPerQuery(torch.autograd.Function):
    """
    Per-query custom autograd.

    Behavior:
      - forward: Triton
      - dQ: Triton
      - dK/dV: Triton
    """

    @staticmethod
    def forward(ctx, q, k, v, block_idx, patch_starts, pp, H, W, P, scale, G):
        o, lse = selection_attn_2d_per_query_forward(q, k, v, block_idx, patch_starts, pp, H, W, P, scale, G)
        ctx.save_for_backward(q, k, v, o, lse, block_idx, patch_starts)
        ctx.pp = pp
        ctx.H = H
        ctx.W = W
        ctx.P = P
        ctx.scale = scale
        ctx.G = G
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, block_idx, patch_starts = ctx.saved_tensors
        do = do.contiguous()

        needs_q, needs_k, needs_v = ctx.needs_input_grad[:3]
        needs_any = needs_q or needs_k or needs_v

        delta = None
        if needs_any:
            delta = (o.float() * do.float()).sum(dim=-1)

        dq = None
        if needs_q:
            dq = selection_per_query_bwd_dq(
                q,
                k,
                v,
                do,
                lse,
                delta,
                block_idx,
                patch_starts,
                ctx.pp,
                ctx.H,
                ctx.W,
                ctx.P,
                ctx.scale,
                ctx.G,
            )

        dk = None
        dv = None
        if needs_k or needs_v:
            dk_ref, dv_ref = selection_per_query_bwd_dkv(
                q,
                k,
                v,
                do,
                lse,
                delta,
                block_idx,
                patch_starts,
                ctx.pp,
                ctx.H,
                ctx.W,
                ctx.P,
                ctx.scale,
                ctx.G,
            )
            if needs_k:
                dk = dk_ref
            if needs_v:
                dv = dv_ref

        return dq, dk, dv, None, None, None, None, None, None, None, None

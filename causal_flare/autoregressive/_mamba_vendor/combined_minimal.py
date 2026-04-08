# Copyright (c) 2024, Tri Dao, Albert Gu.

"""Minimal vendored subset of upstream ``ssd_combined.py``.

This module keeps only the combined chunk-scan + chunk-state backward helper
needed by the separated-FLARE adapter.
"""

import math
from packaging import version

import torch
import triton
import triton.language as tl

from .determinism import alloc_tile_workspace, autotune_configs, finalize_tile_workspace, use_deterministic_mode

TRITON_22 = version.parse(triton.__version__) >= version.parse("2.2.0")


def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


@triton.autotune(
    configs=autotune_configs([
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"])),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"])),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"])),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"])),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"])),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"])),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"])),
        triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"])),
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "dD_ptr"])),
    ]),
    key=["chunk_size", "hdim", "dstate"],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
    x_ptr,
    cb_ptr,
    dout_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    D_ptr,
    b_ptr,
    dstates_ptr,
    dx_ptr,
    ddt_ptr,
    dD_ptr,
    chunk_size,
    hdim,
    dstate,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_cb_batch,
    stride_cb_chunk,
    stride_cb_head,
    stride_cb_csize_m,
    stride_cb_csize_k,
    stride_dout_batch,
    stride_dout_seqlen,
    stride_dout_head,
    stride_dout_hdim,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    stride_D_head,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_b_dstate,
    stride_dstates_batch,
    stride_dstates_chunk,
    stride_dstates_head,
    stride_dstates_hdim,
    stride_dstates_dstate,
    stride_dx_batch,
    stride_dx_seqlen,
    stride_dx_head,
    stride_dx_hdim,
    stride_ddt_batch,
    stride_ddt_chunk,
    stride_ddt_head,
    stride_ddt_csize,
    stride_ddt_tile,
    stride_dD_batch,
    stride_dD_chunk,
    stride_dD_head,
    stride_dD_csize,
    stride_dD_hdim,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    DETERMINISTIC_REDUCTION: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head + pid_n * stride_ddt_tile
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    if not HAS_SEQ_IDX:
        scale = tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0))
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(tl.minimum((dA_cs_last - dA_cs_m), 0.0)), 0.0)

    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_dstates_hdim + offs_dstate[:, None] * stride_dstates_dstate)
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates, input_precision="ieee") * scale[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates, input_precision="ieee")
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= scale[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    k_max = chunk_size_limit
    k_min = pid_m * BLOCK_SIZE_M
    cb_ptrs += k_min * stride_cb_csize_k
    dout_ptrs += k_min * stride_dout_seqlen
    dA_cumsum_ptrs += k_min * stride_dA_cs_csize
    for k in range(k_min, k_max, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < k_max - k), other=0.0)
        dout = tl.load(dout_ptrs, mask=(offs_k[:, None] < k_max - k) & (offs_n[None, :] < hdim), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < k_max - k, other=0.0).to(tl.float32)
        cb *= tl.exp(tl.minimum((dA_cs_k[None, :] - dA_cs_m[:, None]), 0.0))
        mask = (k + offs_k[None, :] >= offs_m[:, None]) & (k + offs_k[None, :] < k_max)
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout, input_precision="ieee")
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dx = acc * dt_m[:, None]
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    if HAS_D:
        dout_res_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
        dout_res = tl.load(dout_res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            d_val = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            d_val = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * d_val
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    if HAS_D:
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            if DETERMINISTIC_REDUCTION:
                tl.store(dD_ptr + pid_n * stride_dD_hdim, dD)
            else:
                tl.atomic_add(dD_ptr, dD)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    if DETERMINISTIC_REDUCTION:
        tl.store(ddt_ptrs, ddt, mask=offs_m < chunk_size)
    else:
        tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)


_CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N = min(
    cfg.kwargs["BLOCK_SIZE_N"] for cfg in _chunk_scan_chunk_state_bwd_dx_kernel.configs
)


def _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=None, seq_idx=None, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    if dx is None:
        dx = torch.empty_like(x)
    deterministic = use_deterministic_mode()
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        block_size_min = 32
        pid_m_tiles = triton.cdiv(chunk_size, block_size_min)
        pid_n_tiles = math.ceil(headdim / _CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N)
        dD_hdim = headdim if D.dim() == 2 else (pid_n_tiles if deterministic else 1)
        dD = torch.zeros(pid_m_tiles, batch, nchunks, nheads, dD_hdim, device=D.device, dtype=torch.float32)
        dD_strides = (dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
    else:
        dD = None
        dD_strides = (0, 0, 0, 0, 0)
    tile_count = math.ceil(headdim / _CHUNK_SCAN_CHUNK_STATE_BWD_DX_MIN_BLOCK_N)
    ddt, stride_ddt_tile = alloc_tile_workspace(
        (batch, nheads, nchunks, chunk_size), tile_count, torch.float32, dout.device, deterministic, zero_init=True
    )
    grid_dx = lambda META: (
        triton.cdiv(chunk_size, META["BLOCK_SIZE_M"]) * triton.cdiv(headdim, META["BLOCK_SIZE_N"]),
        batch * nchunks,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x,
            CB,
            dout,
            dt,
            dA_cumsum,
            seq_idx,
            D,
            B,
            dstates,
            dx,
            ddt,
            dD,
            chunk_size,
            headdim,
            dstate,
            batch,
            seqlen,
            nheads // ngroups,
            x.stride(0),
            x.stride(1),
            x.stride(2),
            x.stride(3),
            CB.stride(0),
            CB.stride(1),
            CB.stride(2),
            CB.stride(-1),
            CB.stride(-2),
            dout.stride(0),
            dout.stride(1),
            dout.stride(2),
            dout.stride(3),
            dt.stride(0),
            dt.stride(2),
            dt.stride(1),
            dt.stride(3),
            dA_cumsum.stride(0),
            dA_cumsum.stride(2),
            dA_cumsum.stride(1),
            dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            D.stride(0) if D is not None else 0,
            B.stride(0),
            B.stride(1),
            B.stride(2),
            B.stride(3),
            dstates.stride(0),
            dstates.stride(1),
            dstates.stride(2),
            dstates.stride(3),
            dstates.stride(4),
            dx.stride(0),
            dx.stride(1),
            dx.stride(2),
            dx.stride(3),
            ddt.stride(0),
            ddt.stride(2),
            ddt.stride(1),
            ddt.stride(3),
            stride_ddt_tile,
            dD_strides[1],
            dD_strides[2],
            dD_strides[3],
            dD_strides[0],
            dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            IS_TRITON_22=TRITON_22,
            DETERMINISTIC_REDUCTION=deterministic,
        )
    if D is not None:
        block_size_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + block_size_actual - 1) // block_size_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2))
        if D.dim() == 1:
            dD = dD.sum(dim=-1)
        dD = dD.to(dtype=D.dtype)
    ddt = finalize_tile_workspace(ddt, deterministic)
    return dx, ddt, dD

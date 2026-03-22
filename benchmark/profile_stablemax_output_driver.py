#!/usr/bin/env python
from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl

from causal_flare._common import _normalize_input_precision, _resolve_attn_scale
from causal_flare.autoregressive.stablemax_triton import (
    _stablemax_forward_config,
    _stablemax_power_mode,
    _stablemax_prepare_score_dot_streamed,
    _stablemax_score_dot_full_panel,
    _stablemax_transform,
    stablemax_output_kernel,
    stablemax_prefix_scan_kernel,
    stablemax_prepare_kernel,
)


def parse_dtype(token: str) -> torch.dtype:
    normalized = token.strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype={token!r}. Expected one of {sorted(mapping)}.")
    return mapping[normalized]


@triton.jit
def _stablemax_output_apply_tile_no_zenc(
    dec_scores_log2,
    enc_scores,
    m_offsets,
    mask_m,
    PrefixDen_ptr,
    PrefixNum_ptr,
    stride_pden_bh,
    stride_pden_nc,
    stride_pden_m,
    stride_pnum_bh,
    stride_pnum_nc,
    stride_pnum_m,
    stride_pnum_d,
    pid_bh,
    pid_nc,
    global_t,
    mask_t,
    d_offsets,
    mask_d,
    causal_mask,
    v_tile,
    row_max,
    row_sum,
    y_num,
    power,
    INPUT_PRECISION: tl.constexpr,
    POWER_MODE: tl.constexpr,
):
    stable_full = _stablemax_transform(enc_scores, power, POWER_MODE=POWER_MODE)
    stable_full = tl.where(mask_t[:, None] & mask_m[None, :], stable_full, 0.0)

    pden_ptrs = PrefixDen_ptr + pid_bh * stride_pden_bh + pid_nc * stride_pden_nc + m_offsets * stride_pden_m
    prefix_den = tl.load(pden_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    local_den = tl.cumsum(stable_full, axis=0)
    total_den = prefix_den[None, :] + local_den
    inv_total_den = tl.where(total_den > 0, 1.0 / total_den, 0.0)

    block_max = tl.max(dec_scores_log2, axis=1)
    new_max = tl.maximum(row_max, block_max)
    new_max_safe = tl.where(new_max == -float("inf"), 0.0, new_max)
    rescale_prev = tl.where(row_max == -float("inf"), 0.0, tl.math.exp2(row_max - new_max_safe))
    both_inf = new_max == -float("inf")
    rescale_prev = tl.where(both_inf & (row_max == -float("inf")), 1.0, rescale_prev)
    p_tile = tl.math.exp2(dec_scores_log2 - new_max_safe[:, None])
    p_tile = tl.where(mask_t[:, None] & mask_m[None, :], p_tile, 0.0)

    pnum_ptrs = (
        PrefixNum_ptr
        + pid_bh * stride_pnum_bh
        + pid_nc * stride_pnum_nc
        + m_offsets[:, None] * stride_pnum_m
        + d_offsets[None, :] * stride_pnum_d
    )
    prefix_num = tl.load(pnum_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    p_tile_eff = p_tile * inv_total_den
    prefix_partial = tl.dot(
        p_tile_eff.to(prefix_num.dtype),
        prefix_num,
        out_dtype=tl.float32,
        input_precision=INPUT_PRECISION,
    )

    w = tl.dot(
        p_tile_eff.to(v_tile.dtype),
        tl.trans(stable_full.to(v_tile.dtype)),
        out_dtype=tl.float32,
        input_precision=INPUT_PRECISION,
    )
    w = tl.where(mask_t[:, None] & mask_t[None, :] & causal_mask, w, 0.0)
    local_partial = tl.dot(w.to(v_tile.dtype), v_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    y_num = y_num * rescale_prev[:, None] + prefix_partial + local_partial
    row_sum = row_sum * rescale_prev + tl.sum(p_tile, axis=1)
    row_max = new_max
    return row_max, row_sum, y_num


@triton.jit
def _stablemax_output_apply_tile_preloaded(
    dec_scores_log2,
    stable_full,
    total_den,
    m_offsets,
    mask_m,
    PrefixNum_ptr,
    stride_pnum_bh,
    stride_pnum_nc,
    stride_pnum_m,
    stride_pnum_d,
    pid_bh,
    pid_nc,
    global_t,
    mask_t,
    d_offsets,
    mask_d,
    causal_mask,
    v_tile,
    row_max,
    row_sum,
    y_num,
    INPUT_PRECISION: tl.constexpr,
):
    inv_total_den = tl.where(total_den > 0, 1.0 / total_den, 0.0)
    block_max = tl.max(dec_scores_log2, axis=1)
    new_max = tl.maximum(row_max, block_max)
    new_max_safe = tl.where(new_max == -float("inf"), 0.0, new_max)
    rescale_prev = tl.where(row_max == -float("inf"), 0.0, tl.math.exp2(row_max - new_max_safe))
    both_inf = new_max == -float("inf")
    rescale_prev = tl.where(both_inf & (row_max == -float("inf")), 1.0, rescale_prev)
    p_tile = tl.math.exp2(dec_scores_log2 - new_max_safe[:, None])
    p_tile = tl.where(mask_t[:, None] & mask_m[None, :], p_tile, 0.0)

    pnum_ptrs = (
        PrefixNum_ptr
        + pid_bh * stride_pnum_bh
        + pid_nc * stride_pnum_nc
        + m_offsets[:, None] * stride_pnum_m
        + d_offsets[None, :] * stride_pnum_d
    )
    prefix_num = tl.load(pnum_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)
    p_tile_eff = p_tile * inv_total_den
    prefix_partial = tl.dot(
        p_tile_eff.to(prefix_num.dtype),
        prefix_num,
        out_dtype=tl.float32,
        input_precision=INPUT_PRECISION,
    )
    w = tl.dot(
        p_tile_eff.to(v_tile.dtype),
        tl.trans(stable_full.to(v_tile.dtype)),
        out_dtype=tl.float32,
        input_precision=INPUT_PRECISION,
    )
    w = tl.where(mask_t[:, None] & mask_t[None, :] & causal_mask, w, 0.0)
    local_partial = tl.dot(w.to(v_tile.dtype), v_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    y_num = y_num * rescale_prev[:, None] + prefix_partial + local_partial
    row_sum = row_sum * rescale_prev + tl.sum(p_tile, axis=1)
    row_max = new_max
    return row_max, row_sum, y_num


@triton.jit
def stablemax_output_kernel_no_zenc(
    Q_ptr,
    K_ptr,
    QDec_ptr,
    KDec_ptr,
    Z_DEC_ptr,
    PrefixDen_ptr,
    PrefixNum_ptr,
    V_ptr,
    O_ptr,
    stride_q_h,
    stride_q_m,
    stride_q_d,
    stride_k_b,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_qd_b,
    stride_qd_n,
    stride_qd_h,
    stride_qd_d,
    stride_kd_h,
    stride_kd_m,
    stride_kd_d,
    stride_zdec_b,
    stride_zdec_h,
    stride_zdec_n,
    stride_pden_bh,
    stride_pden_nc,
    stride_pden_m,
    stride_pnum_bh,
    stride_pnum_nc,
    stride_pnum_m,
    stride_pnum_d,
    stride_v_b,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_o_b,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    N,
    M,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    scale,
    power,
    POWER_MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996
    RCP_LN2: tl.constexpr = 1.4426950408889634
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    t_offsets = tl.arange(0, CHUNK_SIZE)
    global_t = pid_nc * CHUNK_SIZE + t_offsets
    mask_t = global_t < N
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D_VALUE

    qd_bh_base = QDec_ptr + pid_b * stride_qd_b + pid_h * stride_qd_h
    kd_h_base = KDec_ptr + pid_h * stride_kd_h
    q_h_base = Q_ptr + pid_h * stride_q_h
    k_bh_base = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    score_scale_log2 = scale * RCP_LN2

    row_max = tl.full((CHUNK_SIZE,), -float("inf"), tl.float32)
    row_sum = tl.zeros((CHUNK_SIZE,), tl.float32)
    y_num = tl.zeros((CHUNK_SIZE, BLOCK_D), tl.float32)
    causal_mask = global_t[:, None] >= global_t[None, :]

    v_ptrs = (
        V_ptr
        + pid_b * stride_v_b
        + global_t[:, None] * stride_v_n
        + pid_h * stride_v_h
        + d_offsets[None, :] * stride_v_d
    )
    v_tile = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)

    if D_SCORE <= BLOCK_K:
        qd_ptrs = qd_bh_base + global_t[:, None] * stride_qd_n + tl.arange(0, BLOCK_K)[None, :] * stride_qd_d
        qd_tile = tl.load(qd_ptrs, mask=mask_t[:, None], other=0.0)
        k_ptrs = k_bh_base + global_t[:, None] * stride_k_n + tl.arange(0, BLOCK_K)[None, :] * stride_k_d
        k_tile_enc = tl.load(k_ptrs, mask=mask_t[:, None], other=0.0)

    if D_SCORE <= BLOCK_K:
        for m0 in tl.range(0, M, BLOCK_M):
            m_offsets = m0 + tl.arange(0, BLOCK_M)
            mask_m = m_offsets < M
            kd_ptrs = kd_h_base + m_offsets[:, None] * stride_kd_m + tl.arange(0, BLOCK_K)[None, :] * stride_kd_d
            kd_tile = tl.load(kd_ptrs, mask=mask_m[:, None], other=0.0)
            dec_scores_log2 = _stablemax_score_dot_full_panel(qd_tile, kd_tile, INPUT_PRECISION=INPUT_PRECISION) * score_scale_log2
            q_ptrs = q_h_base + m_offsets[:, None] * stride_q_m + tl.arange(0, BLOCK_K)[None, :] * stride_q_d
            q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
            enc_scores = _stablemax_score_dot_full_panel(k_tile_enc, q_tile, INPUT_PRECISION=INPUT_PRECISION) * scale
            row_max, row_sum, y_num = _stablemax_output_apply_tile_no_zenc(
                dec_scores_log2,
                enc_scores,
                m_offsets,
                mask_m,
                PrefixDen_ptr,
                PrefixNum_ptr,
                stride_pden_bh,
                stride_pden_nc,
                stride_pden_m,
                stride_pnum_bh,
                stride_pnum_nc,
                stride_pnum_m,
                stride_pnum_d,
                pid_bh,
                pid_nc,
                global_t,
                mask_t,
                d_offsets,
                mask_d,
                causal_mask,
                v_tile,
                row_max,
                row_sum,
                y_num,
                power,
                INPUT_PRECISION=INPUT_PRECISION,
                POWER_MODE=POWER_MODE,
            )
    else:
        for m0 in tl.range(0, M, BLOCK_M):
            m_offsets = m0 + tl.arange(0, BLOCK_M)
            mask_m = m_offsets < M
            dec_scores_log2 = _stablemax_prepare_score_dot_streamed(
                qd_bh_base,
                kd_h_base,
                global_t,
                m_offsets,
                mask_t,
                mask_m,
                stride_qd_n,
                stride_kd_m,
                stride_qd_d,
                stride_kd_d,
                D_SCORE=D_SCORE,
                BLOCK_K=BLOCK_K,
                INPUT_PRECISION=INPUT_PRECISION,
                scale=score_scale_log2,
            )
            enc_scores = _stablemax_prepare_score_dot_streamed(
                k_bh_base,
                q_h_base,
                global_t,
                m_offsets,
                mask_t,
                mask_m,
                stride_k_n,
                stride_q_m,
                stride_k_d,
                stride_q_d,
                D_SCORE=D_SCORE,
                BLOCK_K=BLOCK_K,
                INPUT_PRECISION=INPUT_PRECISION,
                scale=scale,
            )
            row_max, row_sum, y_num = _stablemax_output_apply_tile_no_zenc(
                dec_scores_log2,
                enc_scores,
                m_offsets,
                mask_m,
                PrefixDen_ptr,
                PrefixNum_ptr,
                stride_pden_bh,
                stride_pden_nc,
                stride_pden_m,
                stride_pnum_bh,
                stride_pnum_nc,
                stride_pnum_m,
                stride_pnum_d,
                pid_bh,
                pid_nc,
                global_t,
                mask_t,
                d_offsets,
                mask_d,
                causal_mask,
                v_tile,
                row_max,
                row_sum,
                y_num,
                power,
                INPUT_PRECISION=INPUT_PRECISION,
                POWER_MODE=POWER_MODE,
            )

    if pid_d == 0:
        z_dec = (row_max + tl.math.log2(tl.maximum(row_sum, 1e-20))) * LN2
        tl.store(
            Z_DEC_ptr + pid_b * stride_zdec_b + pid_h * stride_zdec_h + global_t * stride_zdec_n,
            z_dec,
            mask=mask_t,
        )

    y_tile = y_num * (1.0 / tl.where(row_sum > 0, row_sum, 1.0))[:, None]
    o_ptrs = (
        O_ptr
        + pid_b * stride_o_b
        + global_t[:, None] * stride_o_n
        + pid_h * stride_o_h
        + d_offsets[None, :] * stride_o_d
    )
    tl.store(o_ptrs, y_tile, mask=mask_t[:, None] & mask_d[None, :])


@triton.jit
def stablemax_output_kernel_preload_enc(
    QDec_ptr,
    KDec_ptr,
    Stable_ptr,
    Z_ENC_ptr,
    Z_DEC_ptr,
    PrefixNum_ptr,
    V_ptr,
    O_ptr,
    stride_qd_b,
    stride_qd_n,
    stride_qd_h,
    stride_qd_d,
    stride_kd_h,
    stride_kd_m,
    stride_kd_d,
    stride_stable_b,
    stride_stable_h,
    stride_stable_n,
    stride_stable_m,
    stride_zenc_b,
    stride_zenc_h,
    stride_zenc_n,
    stride_zenc_m,
    stride_zdec_b,
    stride_zdec_h,
    stride_zdec_n,
    stride_pnum_bh,
    stride_pnum_nc,
    stride_pnum_m,
    stride_pnum_d,
    stride_v_b,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_o_b,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    N,
    M,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996
    RCP_LN2: tl.constexpr = 1.4426950408889634
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    pid_d = tl.program_id(2)
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    t_offsets = tl.arange(0, CHUNK_SIZE)
    global_t = pid_nc * CHUNK_SIZE + t_offsets
    mask_t = global_t < N
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D_VALUE

    qd_bh_base = QDec_ptr + pid_b * stride_qd_b + pid_h * stride_qd_h
    kd_h_base = KDec_ptr + pid_h * stride_kd_h
    score_scale_log2 = scale * RCP_LN2

    row_max = tl.full((CHUNK_SIZE,), -float("inf"), tl.float32)
    row_sum = tl.zeros((CHUNK_SIZE,), tl.float32)
    y_num = tl.zeros((CHUNK_SIZE, BLOCK_D), tl.float32)
    causal_mask = global_t[:, None] >= global_t[None, :]

    v_ptrs = (
        V_ptr
        + pid_b * stride_v_b
        + global_t[:, None] * stride_v_n
        + pid_h * stride_v_h
        + d_offsets[None, :] * stride_v_d
    )
    v_tile = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)

    qd_ptrs = qd_bh_base + global_t[:, None] * stride_qd_n + tl.arange(0, BLOCK_K)[None, :] * stride_qd_d
    qd_tile = tl.load(qd_ptrs, mask=mask_t[:, None], other=0.0)

    for m0 in tl.range(0, M, BLOCK_M):
        m_offsets = m0 + tl.arange(0, BLOCK_M)
        mask_m = m_offsets < M
        kd_ptrs = kd_h_base + m_offsets[:, None] * stride_kd_m + tl.arange(0, BLOCK_K)[None, :] * stride_kd_d
        kd_tile = tl.load(kd_ptrs, mask=mask_m[:, None], other=0.0)
        dec_scores_log2 = _stablemax_score_dot_full_panel(qd_tile, kd_tile, INPUT_PRECISION=INPUT_PRECISION) * score_scale_log2
        dec_scores_log2 = tl.where(mask_t[:, None] & mask_m[None, :], dec_scores_log2, -float("inf"))

        stable_ptrs = (
            Stable_ptr
            + pid_b * stride_stable_b
            + pid_h * stride_stable_h
            + global_t[:, None] * stride_stable_n
            + m_offsets[None, :] * stride_stable_m
        )
        stable_full = tl.load(stable_ptrs, mask=mask_t[:, None] & mask_m[None, :], other=0.0)

        zenc_ptrs = (
            Z_ENC_ptr
            + pid_b * stride_zenc_b
            + pid_h * stride_zenc_h
            + global_t[:, None] * stride_zenc_n
            + m_offsets[None, :] * stride_zenc_m
        )
        total_den = tl.load(zenc_ptrs, mask=mask_t[:, None] & mask_m[None, :], other=0.0).to(tl.float32)

        row_max, row_sum, y_num = _stablemax_output_apply_tile_preloaded(
            dec_scores_log2,
            stable_full,
            total_den,
            m_offsets,
            mask_m,
            PrefixNum_ptr,
            stride_pnum_bh,
            stride_pnum_nc,
            stride_pnum_m,
            stride_pnum_d,
            pid_bh,
            pid_nc,
            global_t,
            mask_t,
            d_offsets,
            mask_d,
            causal_mask,
            v_tile,
            row_max,
            row_sum,
            y_num,
            INPUT_PRECISION=INPUT_PRECISION,
        )

    if pid_d == 0:
        z_dec = (row_max + tl.math.log2(tl.maximum(row_sum, 1e-20))) * LN2
        tl.store(
            Z_DEC_ptr + pid_b * stride_zdec_b + pid_h * stride_zdec_h + global_t * stride_zdec_n,
            z_dec,
            mask=mask_t,
        )

    y_tile = y_num * (1.0 / tl.where(row_sum > 0, row_sum, 1.0))[:, None]
    o_ptrs = (
        O_ptr
        + pid_b * stride_o_b
        + global_t[:, None] * stride_o_n
        + pid_h * stride_o_h
        + d_offsets[None, :] * stride_o_d
    )
    tl.store(o_ptrs, y_tile, mask=mask_t[:, None] & mask_d[None, :])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile stablemax output kernel variants.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-heads", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--latent-queries", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--variant", choices=("baseline", "no_zenc", "preload_enc"), default="baseline")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--time", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_heads % args.batch_size != 0:
        raise ValueError("batch_heads must be divisible by batch_size")
    B = args.batch_size
    H = args.batch_heads // args.batch_size
    N = args.seq_len
    M = args.latent_queries
    D = args.head_dim
    C = args.chunk_size
    dtype = parse_dtype(args.dtype)
    device = "cuda"
    power = 2.0
    power_mode = _stablemax_power_mode(power)
    save_stable_is_bf16 = dtype == torch.bfloat16

    torch.manual_seed(0)
    q = torch.randn((H, M, D), device=device, dtype=dtype)
    k = torch.randn((B, N, H, D), device=device, dtype=dtype)
    v = torch.randn((B, N, H, D), device=device, dtype=dtype)
    q_dec = torch.randn((B, N, H, D), device=device, dtype=dtype)
    k_dec = torch.randn((H, M, D), device=device, dtype=dtype)

    cfg = _stablemax_forward_config(M=M, D_score=D, D_value=D, chunk_size=C, input_precision=None)
    scale = _resolve_attn_scale(None, D)
    input_precision = _normalize_input_precision(None, None)
    num_chunks = math.ceil(N / C)
    bh = B * H

    chunk_den = torch.empty((bh, num_chunks, M), device=device, dtype=torch.float32)
    chunk_num = torch.empty((bh, num_chunks, M, D), device=device, dtype=dtype)
    prefix_den = torch.empty_like(chunk_den)
    prefix_num = torch.empty_like(chunk_num)
    stable_enc = torch.empty((1,), device=device, dtype=dtype)
    z_enc = torch.empty((B, H, N, M), device=device, dtype=torch.float32)
    z_dec = torch.empty((B, H, N), device=device, dtype=torch.float32)
    out = torch.empty((B, N, H, D), device=device, dtype=torch.float32)

    prepare_grid = (bh, num_chunks, cfg["NUM_M_TILES"] * cfg["NUM_D_TILES"])
    prefix_grid = (bh, cfg["NUM_M_TILES"] * cfg["NUM_D_TILES"])
    output_grid = (bh, num_chunks, cfg["NUM_D_TILES"])

    stablemax_prepare_kernel[prepare_grid](
        q,
        k,
        v,
        chunk_den,
        chunk_num,
        stable_enc,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *chunk_den.stride(),
        *chunk_num.stride(),
        N,
        M,
        D_SCORE=D,
        D_VALUE=D,
        CHUNK_SIZE=C,
        scale=scale,
        power=power,
        POWER_MODE=power_mode,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_D=cfg["BLOCK_D"],
        BLOCK_K=cfg["BLOCK_K"],
        INPUT_PRECISION=cfg["input_precision"],
        SAVE_STABLEMAX_ENCODE_SCORES=False,
        SAVE_STABLE_IS_BF16=save_stable_is_bf16,
        H=H,
        num_warps=cfg["prepare_num_warps"],
        num_stages=cfg["prepare_num_stages"],
    )
    stablemax_prefix_scan_kernel[prefix_grid](
        chunk_den,
        chunk_num,
        prefix_den,
        prefix_num,
        *chunk_den.stride(),
        *chunk_num.stride(),
        *prefix_den.stride(),
        *prefix_num.stride(),
        M,
        D_VALUE=D,
        NUM_CHUNKS=num_chunks,
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_D=cfg["BLOCK_D"],
        num_warps=cfg["prefix_num_warps"],
        num_stages=cfg["prefix_num_stages"],
    )
    torch.cuda.synchronize()

    stable_all = None
    if args.variant == "preload_enc":
        with torch.no_grad():
            score = scale * torch.einsum("bnhd,hmd->bhnm", k, q)
            one = torch.ones((), device=score.device, dtype=score.dtype)
            pos_base = torch.where(score >= 0, score + one, one)
            neg_base = torch.where(score < 0, one - score, one)
            stable_all = torch.where(score >= 0, pos_base * pos_base, 1.0 / (neg_base * neg_base)).contiguous()
            stable_chunk = stable_all.reshape(B, num_chunks, C, H, M).permute(0, 3, 1, 2, 4).contiguous()
            prefix_den_bhnm = prefix_den.reshape(B, H, num_chunks, M)
            z_enc_chunk = prefix_den_bhnm[:, :, :, None, :] + stable_chunk.cumsum(dim=3)
            z_enc.copy_(z_enc_chunk.reshape(B, H, N, M))

    if args.variant == "baseline":
        launch = lambda: stablemax_output_kernel[output_grid](
            q,
            k,
            q_dec,
            k_dec,
            stable_enc,
            z_enc,
            z_dec,
            prefix_den,
            prefix_num,
            v,
            out,
            *q.stride(),
            *k.stride(),
            *q_dec.stride(),
            *k_dec.stride(),
            *z_enc.stride(),
            *z_dec.stride(),
            *prefix_den.stride(),
            *prefix_num.stride(),
            *v.stride(),
            *out.stride(),
            N,
            M,
            D_SCORE=D,
            D_VALUE=D,
            CHUNK_SIZE=C,
            scale=scale,
            power=power,
            POWER_MODE=power_mode,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_D=cfg["BLOCK_D"],
            BLOCK_K=cfg["BLOCK_K"],
            INPUT_PRECISION=input_precision,
            SAVE_STABLEMAX_ENCODE_SCORES=False,
            SAVE_STABLE_IS_BF16=save_stable_is_bf16,
            H=H,
            num_warps=cfg["output_num_warps"],
            num_stages=cfg["output_num_stages"],
        )
    elif args.variant == "no_zenc":
        launch = lambda: stablemax_output_kernel_no_zenc[output_grid](
            q,
            k,
            q_dec,
            k_dec,
            z_dec,
            prefix_den,
            prefix_num,
            v,
            out,
            *q.stride(),
            *k.stride(),
            *q_dec.stride(),
            *k_dec.stride(),
            *z_dec.stride(),
            *prefix_den.stride(),
            *prefix_num.stride(),
            *v.stride(),
            *out.stride(),
            N,
            M,
            D_SCORE=D,
            D_VALUE=D,
            CHUNK_SIZE=C,
            scale=scale,
            power=power,
            POWER_MODE=power_mode,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_D=cfg["BLOCK_D"],
            BLOCK_K=cfg["BLOCK_K"],
            INPUT_PRECISION=input_precision,
            H=H,
            num_warps=cfg["output_num_warps"],
            num_stages=cfg["output_num_stages"],
        )
    else:
        assert stable_all is not None
        launch = lambda: stablemax_output_kernel_preload_enc[output_grid](
            q_dec,
            k_dec,
            stable_all,
            z_enc,
            z_dec,
            prefix_num,
            v,
            out,
            *q_dec.stride(),
            *k_dec.stride(),
            *stable_all.stride(),
            *z_enc.stride(),
            *z_dec.stride(),
            *prefix_num.stride(),
            *v.stride(),
            *out.stride(),
            N,
            M,
            D_SCORE=D,
            D_VALUE=D,
            CHUNK_SIZE=C,
            scale=scale,
            BLOCK_M=cfg["BLOCK_M"],
            BLOCK_D=cfg["BLOCK_D"],
            BLOCK_K=cfg["BLOCK_K"],
            INPUT_PRECISION=input_precision,
            H=H,
            num_warps=cfg["output_num_warps"],
            num_stages=cfg["output_num_stages"],
        )

    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()

    if args.time:
        times = []
        for _ in range(args.reps):
            s = torch.cuda.Event(True)
            e = torch.cuda.Event(True)
            s.record()
            launch()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
        print({"variant": args.variant, "mean_ms": sum(times) / len(times), "reps": args.reps})
    else:
        for _ in range(args.reps):
            launch()
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()

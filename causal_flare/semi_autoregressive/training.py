from causal_flare._common import *
from causal_flare._reference_utils import (
    resolve_flare_causal_decode_inputs as _resolve_semi_ar_decode_inputs,
    validate_flare_qkv_layouts as _validate_flare_qkv_layouts,
)
from causal_flare.autoregressive.training import _profiled_call, _refresh_profile_totals, _resolve_forward_launch
from causal_flare.semi_autoregressive.reference import _validate_block_causal_config


def _get_semi_ar_forward_bucket_defaults(
    *,
    M: int,
    D_score: int,
    D_value: int,
    block_size: int,
    chunk_size: int,
) -> dict[str, object]:
    max_d = max(D_score, D_value)
    mixed_d = D_score != D_value

    if chunk_size <= 16:
        block_t = chunk_size
    elif chunk_size <= 32:
        block_t = 16
    elif mixed_d:
        block_t = 16
    elif max_d > 64 and M >= 192:
        block_t = 32
    else:
        block_t = min(64, chunk_size)

    if block_size <= 32:
        return {
            "block_t": block_t,
            "prepare_launch": (2, 2),
            "lse_output_launch": (2, 2),
        }

    if max_d <= 64:
        if M > 256:
            return {
                "block_t": min(64, chunk_size),
                "prepare_launch": (8, 2),
                "lse_output_launch": (4, 3),
            }
        if block_size <= 64:
            return {
                "block_t": block_t,
                "prepare_launch": (4, 1),
                "lse_output_launch": (4, 1),
            }
        if block_size > 256:
            return {
                "block_t": min(64, chunk_size),
                "prepare_launch": (8, 2),
                "lse_output_launch": (4, 1),
            }
        return {
            "prepare_launch": (4, 4),
            "lse_output_launch": (4, 2),
            "block_t": block_t,
        }

    if M > 256:
        return {
            "block_t": block_t,
            "prepare_launch": (8, 2),
            "lse_output_launch": (4, 3),
        }

    if mixed_d:
        return {
            "block_t": block_t,
            "prepare_launch": (8, 2) if D_score > D_value else (4, 1),
            "lse_output_launch": (2, 2) if D_score > D_value else (4, 2),
        }

    if M >= 256:
        return {
            "block_t": block_t,
            "prepare_launch": (8, 2),
            "lse_output_launch": (4, 4),
        }

    if block_size > 128:
        return {
            "block_t": block_t,
            "prepare_launch": (4, 1),
            "lse_output_launch": (4, 4),
        }

    return {
        "block_t": block_t,
        "prepare_launch": (4, 2),
        "lse_output_launch": (4, 1),
    }


def _get_semi_ar_forward_config(
    *,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    block_size: int,
    chunk_size: int,
    input_precision=None,
) -> dict[str, object]:
    block_m = min(64, max(16, triton.next_power_of_2(M)))
    block_d = min(64, max(16, triton.next_power_of_2(D_value)))
    block_k = min(64, max(16, triton.next_power_of_2(D_score)))
    bucket_defaults = _get_semi_ar_forward_bucket_defaults(
        M=M,
        D_score=D_score,
        D_value=D_value,
        block_size=block_size,
        chunk_size=chunk_size,
    )
    block_t_env = os.environ.get("FLARE_SEMI_AR_BLOCK_T", "").strip()
    if block_t_env:
        block_t = int(block_t_env)
    else:
        block_t = int(bucket_defaults["block_t"])
    if block_t <= 0 or block_t > chunk_size or chunk_size % block_t != 0:
        raise ValueError(
            "SemiAutoRegressiveFLARE requires BLOCK_T to be a positive divisor of chunk_size. "
            f"Got BLOCK_T={block_t}, chunk_size={chunk_size}."
        )
    if block_t % 16 != 0:
        raise ValueError(f"SemiAutoRegressiveFLARE requires BLOCK_T to be a multiple of 16. Got BLOCK_T={block_t}.")

    block_prepare_warps, block_prepare_stages = _resolve_forward_launch(
        "semi_ar_block_prepare",
        default_num_warps=int(bucket_defaults["prepare_launch"][0]),
        default_num_stages=int(bucket_defaults["prepare_launch"][1]),
    )
    scan_warps, scan_stages = _resolve_forward_launch(
        "semi_ar_scan",
        default_num_warps=8,
        default_num_stages=2,
    )
    block_z_warps, block_z_stages = _resolve_forward_launch(
        "semi_ar_block_z",
        default_num_warps=4,
        default_num_stages=1,
    )
    decoder_warps, decoder_stages = _resolve_forward_launch(
        "semi_ar_decoder_lse",
        default_num_warps=2,
        default_num_stages=1,
    )
    lse_output_warps, lse_output_stages = _resolve_forward_launch(
        "semi_ar_lse_output",
        default_num_warps=int(bucket_defaults["lse_output_launch"][0]),
        default_num_stages=int(bucket_defaults["lse_output_launch"][1]),
    )

    return {
        "NUM_BLOCKS": N // block_size,
        "NUM_GLOBAL_CHUNKS": N // chunk_size,
        "CHUNKS_PER_BLOCK": block_size // chunk_size,
        "BLOCK_SIZE": block_size,
        "CHUNK_SIZE": chunk_size,
        "BLOCK_M": block_m,
        "BLOCK_D": block_d,
        "BLOCK_K": block_k,
        "BLOCK_T": block_t,
        "NUM_M_TILES": triton.cdiv(M, block_m),
        "NUM_D_VALUE_BLOCKS": triton.cdiv(D_value, block_d),
        "input_precision": _normalize_input_precision(input_precision, None),
        "block_prepare_num_warps": block_prepare_warps,
        "block_prepare_num_stages": block_prepare_stages,
        "scan_num_warps": scan_warps,
        "scan_num_stages": scan_stages,
        "block_z_num_warps": block_z_warps,
        "block_z_num_stages": block_z_stages,
        "decoder_num_warps": decoder_warps,
        "decoder_num_stages": decoder_stages,
        "lse_output_num_warps": lse_output_warps,
        "lse_output_num_stages": lse_output_stages,
    }


@triton.jit
def semi_ar_chunk_prepare_kernel(
    K_ptr,
    Q_ptr,
    V_ptr,
    ChunkMax_ptr,
    ChunkDen_ptr,
    ChunkNum_ptr,
    stride_k_b,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_q_h,
    stride_q_m,
    stride_q_d,
    stride_v_b,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_cmax_bh,
    stride_cmax_chunk,
    stride_cmax_m,
    stride_cden_bh,
    stride_cden_chunk,
    stride_cden_m,
    stride_cnum_bh,
    stride_cnum_chunk,
    stride_cnum_m,
    stride_cnum_d,
    BH,
    M,
    N,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    global_chunk = tl.program_id(1)
    pid_md = tl.program_id(2)
    if pid_bh >= BH:
        return

    num_d_tiles = tl.cdiv(D_VALUE, BLOCK_D)
    pid_m = pid_md // num_d_tiles
    pid_d = pid_md - pid_m * num_d_tiles

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    chunk_start = global_chunk * CHUNK_SIZE
    token_offsets = tl.arange(0, CHUNK_SIZE)
    token_idx = chunk_start + token_offsets
    token_mask = token_idx < N

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    mask_md = mask_m[:, None] & mask_d[None, :]
    store_shared = pid_d == 0

    q_base = Q_ptr + pid_h * stride_q_h
    k_base = K_ptr + pid_b * stride_k_b + chunk_start * stride_k_n + pid_h * stride_k_h
    v_base = V_ptr + pid_b * stride_v_b + chunk_start * stride_v_n + pid_h * stride_v_h

    scores = tl.zeros((CHUNK_SIZE, BLOCK_M), dtype=tl.float32)
    for k0 in tl.static_range(0, D_SCORE, BLOCK_K):
        d_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = d_k < D_SCORE
        q_tile = tl.load(
            q_base + m_offsets[:, None] * stride_q_m + d_k[None, :] * stride_q_d,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        k_tile = tl.load(
            k_base + token_offsets[:, None] * stride_k_n + d_k[None, :] * stride_k_d,
            mask=token_mask[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float32)
        scores += tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    scores = scores * scale
    scores = tl.where(token_mask[:, None] & mask_m[None, :], scores, -float("inf"))

    chunk_max = tl.max(scores, axis=0)
    exp_scores = tl.exp(scores - chunk_max[None, :])
    exp_scores = tl.where(token_mask[:, None] & mask_m[None, :], exp_scores, 0.0)
    chunk_den = tl.sum(exp_scores, axis=0)

    v_tile = tl.load(
        v_base + token_offsets[:, None] * stride_v_n + d_offsets[None, :] * stride_v_d,
        mask=token_mask[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    chunk_num = tl.dot(tl.trans(exp_scores), v_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    cmax_ptr = ChunkMax_ptr + pid_bh * stride_cmax_bh + global_chunk * stride_cmax_chunk
    cden_ptr = ChunkDen_ptr + pid_bh * stride_cden_bh + global_chunk * stride_cden_chunk
    cnum_ptr = ChunkNum_ptr + pid_bh * stride_cnum_bh + global_chunk * stride_cnum_chunk
    tl.store(cmax_ptr + m_offsets * stride_cmax_m, chunk_max, mask=store_shared & mask_m)
    tl.store(cden_ptr + m_offsets * stride_cden_m, chunk_den, mask=store_shared & mask_m)
    tl.store(
        cnum_ptr + m_offsets[:, None] * stride_cnum_m + d_offsets[None, :] * stride_cnum_d,
        chunk_num,
        mask=mask_md,
    )


@triton.jit
def semi_ar_block_reduce_from_chunks_kernel(
    ChunkMax_ptr,
    ChunkDen_ptr,
    ChunkNum_ptr,
    BlockMax_ptr,
    BlockDen_ptr,
    BlockNum_ptr,
    stride_cmax_bh,
    stride_cmax_chunk,
    stride_cmax_m,
    stride_cden_bh,
    stride_cden_chunk,
    stride_cden_m,
    stride_cnum_bh,
    stride_cnum_chunk,
    stride_cnum_m,
    stride_cnum_d,
    stride_bmax_bh,
    stride_bmax_blk,
    stride_bmax_m,
    stride_bden_bh,
    stride_bden_blk,
    stride_bden_m,
    stride_bnum_bh,
    stride_bnum_blk,
    stride_bnum_m,
    stride_bnum_d,
    BH,
    M,
    D_VALUE: tl.constexpr,
    NUM_BLOCKS,
    CHUNKS_PER_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    block_idx = tl.program_id(1)
    pid_md = tl.program_id(2)
    if pid_bh >= BH or block_idx >= NUM_BLOCKS:
        return

    num_d_tiles = tl.cdiv(D_VALUE, BLOCK_D)
    pid_m = pid_md // num_d_tiles
    pid_d = pid_md - pid_m * num_d_tiles

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    mask_md = mask_m[:, None] & mask_d[None, :]
    store_shared = pid_d == 0

    block_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    block_den = tl.zeros((BLOCK_M,), tl.float32)
    block_num = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    local_chunk = 0
    while local_chunk < CHUNKS_PER_BLOCK:
        global_chunk = block_idx * CHUNKS_PER_BLOCK + local_chunk
        cmax_ptr = ChunkMax_ptr + pid_bh * stride_cmax_bh + global_chunk * stride_cmax_chunk
        cden_ptr = ChunkDen_ptr + pid_bh * stride_cden_bh + global_chunk * stride_cden_chunk
        cnum_ptr = ChunkNum_ptr + pid_bh * stride_cnum_bh + global_chunk * stride_cnum_chunk

        chunk_max = tl.load(cmax_ptr + m_offsets * stride_cmax_m, mask=mask_m, other=-float("inf")).to(tl.float32)
        chunk_den = tl.load(cden_ptr + m_offsets * stride_cden_m, mask=mask_m, other=0.0).to(tl.float32)
        chunk_num = tl.load(
            cnum_ptr + m_offsets[:, None] * stride_cnum_m + d_offsets[None, :] * stride_cnum_d,
            mask=mask_md,
            other=0.0,
        ).to(tl.float32)

        max_new = tl.maximum(block_max, chunk_max)
        max_new_safe = tl.where(max_new == -float("inf"), 0.0, max_new)
        scale_prev = tl.where(block_max == -float("inf"), 0.0, tl.exp(block_max - max_new_safe))
        scale_chunk = tl.where(chunk_max == -float("inf"), 0.0, tl.exp(chunk_max - max_new_safe))
        both_inf = max_new == -float("inf")
        scale_prev = tl.where(both_inf & (block_max == -float("inf")), 1.0, scale_prev)
        scale_chunk = tl.where(both_inf & (chunk_max == -float("inf")), 1.0, scale_chunk)

        block_den = block_den * scale_prev + chunk_den * scale_chunk
        block_num = block_num * scale_prev[:, None] + chunk_num * scale_chunk[:, None]
        block_max = max_new
        local_chunk += 1

    bmax_ptr = BlockMax_ptr + pid_bh * stride_bmax_bh + block_idx * stride_bmax_blk
    bden_ptr = BlockDen_ptr + pid_bh * stride_bden_bh + block_idx * stride_bden_blk
    bnum_ptr = BlockNum_ptr + pid_bh * stride_bnum_bh + block_idx * stride_bnum_blk
    tl.store(bmax_ptr + m_offsets * stride_bmax_m, block_max, mask=store_shared & mask_m)
    tl.store(bden_ptr + m_offsets * stride_bden_m, block_den, mask=store_shared & mask_m)
    tl.store(
        bnum_ptr + m_offsets[:, None] * stride_bnum_m + d_offsets[None, :] * stride_bnum_d,
        block_num,
        mask=mask_md,
    )


@triton.jit
def semi_ar_block_prepare_kernel(
    K_ptr,
    Q_ptr,
    V_ptr,
    BlockMax_ptr,
    BlockDen_ptr,
    BlockNum_ptr,
    stride_k_b,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_q_h,
    stride_q_m,
    stride_q_d,
    stride_v_b,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_bmax_bh,
    stride_bmax_blk,
    stride_bmax_m,
    stride_bden_bh,
    stride_bden_blk,
    stride_bden_m,
    stride_bnum_bh,
    stride_bnum_blk,
    stride_bnum_m,
    stride_bnum_d,
    BH,
    M,
    N,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    scale,
    BLOCK_SIZE,
    CHUNK_SIZE: tl.constexpr,
    CHUNKS_PER_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    block_idx = tl.program_id(1)
    pid_md = tl.program_id(2)
    if pid_bh >= BH:
        return

    num_d_tiles = tl.cdiv(D_VALUE, BLOCK_D)
    pid_m = pid_md // num_d_tiles
    pid_d = pid_md - pid_m * num_d_tiles

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    mask_md = mask_m[:, None] & mask_d[None, :]
    store_shared = pid_d == 0

    q_base = Q_ptr + pid_h * stride_q_h
    block_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    block_den = tl.zeros((BLOCK_M,), tl.float32)
    block_num = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    local_chunk = 0
    while local_chunk < CHUNKS_PER_BLOCK:
        chunk_start = block_idx * BLOCK_SIZE + local_chunk * CHUNK_SIZE
        token_offsets = tl.arange(0, CHUNK_SIZE)
        token_idx = chunk_start + token_offsets
        token_mask = token_idx < N

        k_base = K_ptr + pid_b * stride_k_b + chunk_start * stride_k_n + pid_h * stride_k_h
        v_base = V_ptr + pid_b * stride_v_b + chunk_start * stride_v_n + pid_h * stride_v_h

        scores = tl.zeros((CHUNK_SIZE, BLOCK_M), dtype=tl.float32)
        for k0 in tl.static_range(0, D_SCORE, BLOCK_K):
            d_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = d_k < D_SCORE
            q_tile = tl.load(
                q_base + m_offsets[:, None] * stride_q_m + d_k[None, :] * stride_q_d,
                mask=mask_m[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            k_tile = tl.load(
                k_base + token_offsets[:, None] * stride_k_n + d_k[None, :] * stride_k_d,
                mask=token_mask[:, None] & mask_k[None, :],
                other=0.0,
            ).to(tl.float32)
            scores += tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        scores = scores * scale
        scores = tl.where(token_mask[:, None] & mask_m[None, :], scores, -float("inf"))

        chunk_max = tl.max(scores, axis=0)
        exp_scores = tl.exp(scores - chunk_max[None, :])
        exp_scores = tl.where(token_mask[:, None] & mask_m[None, :], exp_scores, 0.0)
        chunk_den = tl.sum(exp_scores, axis=0)

        v_tile = tl.load(
            v_base + token_offsets[:, None] * stride_v_n + d_offsets[None, :] * stride_v_d,
            mask=token_mask[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        chunk_num = tl.dot(tl.trans(exp_scores), v_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        max_new = tl.maximum(block_max, chunk_max)
        max_new_safe = tl.where(max_new == -float("inf"), 0.0, max_new)
        scale_prev = tl.where(block_max == -float("inf"), 0.0, tl.exp(block_max - max_new_safe))
        scale_chunk = tl.where(chunk_max == -float("inf"), 0.0, tl.exp(chunk_max - max_new_safe))
        both_inf = max_new == -float("inf")
        scale_prev = tl.where(both_inf & (block_max == -float("inf")), 1.0, scale_prev)
        scale_chunk = tl.where(both_inf & (chunk_max == -float("inf")), 1.0, scale_chunk)
        block_den = block_den * scale_prev + chunk_den * scale_chunk
        block_num = block_num * scale_prev[:, None] + chunk_num * scale_chunk[:, None]
        block_max = max_new
        local_chunk += 1

    bmax_ptr = BlockMax_ptr + pid_bh * stride_bmax_bh + block_idx * stride_bmax_blk
    bden_ptr = BlockDen_ptr + pid_bh * stride_bden_bh + block_idx * stride_bden_blk
    bnum_ptr = BlockNum_ptr + pid_bh * stride_bnum_bh + block_idx * stride_bnum_blk
    tl.store(bmax_ptr + m_offsets * stride_bmax_m, block_max, mask=store_shared & mask_m)
    tl.store(bden_ptr + m_offsets * stride_bden_m, block_den, mask=store_shared & mask_m)
    tl.store(
        bnum_ptr + m_offsets[:, None] * stride_bnum_m + d_offsets[None, :] * stride_bnum_d,
        block_num,
        mask=mask_md,
    )


@triton.jit
def semi_ar_block_scan_kernel(
    BlockMax_ptr,
    BlockDen_ptr,
    BlockNum_ptr,
    PrefixMax_ptr,
    PrefixDen_ptr,
    PrefixNum_ptr,
    LSEEnc_ptr,
    stride_bmax_bh,
    stride_bmax_blk,
    stride_bmax_m,
    stride_bden_bh,
    stride_bden_blk,
    stride_bden_m,
    stride_bnum_bh,
    stride_bnum_blk,
    stride_bnum_m,
    stride_bnum_d,
    stride_pmax_bh,
    stride_pmax_blk,
    stride_pmax_m,
    stride_pden_bh,
    stride_pden_blk,
    stride_pden_m,
    stride_pnum_bh,
    stride_pnum_blk,
    stride_pnum_m,
    stride_pnum_d,
    stride_lsee_bh,
    stride_lsee_blk,
    stride_lsee_m,
    BH,
    M,
    D_VALUE: tl.constexpr,
    NUM_BLOCKS,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_d = tl.program_id(2)
    if pid_bh >= BH:
        return

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    mask_md = mask_m[:, None] & mask_d[None, :]
    store_shared = pid_d == 0

    prefix_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    prefix_den = tl.zeros((BLOCK_M,), tl.float32)
    prefix_num = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    block_idx = 0
    while block_idx < NUM_BLOCKS:
        pmax_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + block_idx * stride_pmax_blk
        pden_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + block_idx * stride_pden_blk
        pnum_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + block_idx * stride_pnum_blk
        tl.store(pmax_ptr + m_offsets * stride_pmax_m, prefix_max, mask=store_shared & mask_m)
        tl.store(pden_ptr + m_offsets * stride_pden_m, prefix_den, mask=store_shared & mask_m)
        tl.store(
            pnum_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d,
            prefix_num,
            mask=mask_md,
        )

        bmax_ptr = BlockMax_ptr + pid_bh * stride_bmax_bh + block_idx * stride_bmax_blk
        bden_ptr = BlockDen_ptr + pid_bh * stride_bden_bh + block_idx * stride_bden_blk
        bnum_ptr = BlockNum_ptr + pid_bh * stride_bnum_bh + block_idx * stride_bnum_blk

        block_max = tl.load(bmax_ptr + m_offsets * stride_bmax_m, mask=mask_m, other=-float("inf")).to(tl.float32)
        block_den = tl.load(bden_ptr + m_offsets * stride_bden_m, mask=mask_m, other=0.0).to(tl.float32)
        block_num = tl.load(
            bnum_ptr + m_offsets[:, None] * stride_bnum_m + d_offsets[None, :] * stride_bnum_d,
            mask=mask_md,
            other=0.0,
        ).to(tl.float32)

        max_new = tl.maximum(prefix_max, block_max)
        max_new_safe = tl.where(max_new == -float("inf"), 0.0, max_new)
        scale_prev = tl.where(prefix_max == -float("inf"), 0.0, tl.exp(prefix_max - max_new_safe))
        scale_block = tl.where(block_max == -float("inf"), 0.0, tl.exp(block_max - max_new_safe))
        both_inf = max_new == -float("inf")
        scale_prev = tl.where(both_inf & (prefix_max == -float("inf")), 1.0, scale_prev)
        scale_block = tl.where(both_inf & (block_max == -float("inf")), 1.0, scale_block)

        prefix_den = prefix_den * scale_prev + block_den * scale_block
        prefix_num = prefix_num * scale_prev[:, None] + block_num * scale_block[:, None]
        prefix_max = max_new

        lse_ptr = LSEEnc_ptr + pid_bh * stride_lsee_bh + block_idx * stride_lsee_blk
        lse = prefix_max + tl.log(tl.maximum(prefix_den, 1e-20))
        tl.store(lse_ptr + m_offsets * stride_lsee_m, lse, mask=store_shared & mask_m)
        block_idx += 1


@triton.jit
def semi_ar_decode_lse_kernel(
    K_ptr,
    Q_ptr,
    QDec_ptr,
    KDec_ptr,
    LSEDec_ptr,
    stride_k_b,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_q_h,
    stride_q_m,
    stride_q_d,
    stride_qdb,
    stride_qdn,
    stride_qdh,
    stride_qdd,
    stride_kdh,
    stride_kdm,
    stride_kdd,
    stride_lsed_bh,
    stride_lsed_n,
    BH,
    M,
    N,
    D_SCORE: tl.constexpr,
    scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    WEIGHT_SHARING_ENC_DEC: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    global_chunk = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    chunk_start = global_chunk * CHUNK_SIZE

    m_local = tl.arange(0, BLOCK_M)
    t_local = tl.arange(0, BLOCK_T)

    t0 = 0
    while t0 < CHUNK_SIZE:
        t_offsets = t0 + t_local
        token_idx = chunk_start + t_offsets
        token_mask = token_idx < N

        lse_max = tl.full((BLOCK_T,), -float("inf"), tl.float32)
        lse_sum = tl.zeros((BLOCK_T,), tl.float32)

        m0 = 0
        while m0 < M:
            m_offsets = m0 + m_local
            mask_m = m_offsets < M
            scores = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)
            for k0 in tl.static_range(0, D_SCORE, BLOCK_K):
                d_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = d_k < D_SCORE
                if WEIGHT_SHARING_ENC_DEC:
                    q_tok = tl.load(
                        K_ptr + pid_b * stride_k_b + token_idx[:, None] * stride_k_n + pid_h * stride_k_h + d_k[None, :] * stride_k_d,
                        mask=token_mask[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    k_bank = tl.load(
                        Q_ptr + pid_h * stride_q_h + m_offsets[:, None] * stride_q_m + d_k[None, :] * stride_q_d,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                else:
                    q_tok = tl.load(
                        QDec_ptr + pid_b * stride_qdb + token_idx[:, None] * stride_qdn + pid_h * stride_qdh + d_k[None, :] * stride_qdd,
                        mask=token_mask[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    k_bank = tl.load(
                        KDec_ptr + pid_h * stride_kdh + m_offsets[:, None] * stride_kdm + d_k[None, :] * stride_kdd,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                scores += tl.dot(k_bank, tl.trans(q_tok), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            scores = scores * scale
            scores = tl.where(mask_m[:, None] & token_mask[None, :], scores, -float("inf"))

            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(lse_max, block_max)
            new_max_safe = tl.where(new_max == -float("inf"), 0.0, new_max)
            rescale_prev = tl.where(lse_max == -float("inf"), 0.0, tl.exp(lse_max - new_max_safe))
            both_inf = new_max == -float("inf")
            rescale_prev = tl.where(both_inf & (lse_max == -float("inf")), 1.0, rescale_prev)
            block_exp = tl.exp(scores - new_max_safe[None, :])
            block_exp = tl.where(mask_m[:, None] & token_mask[None, :], block_exp, 0.0)
            lse_sum = lse_sum * rescale_prev + tl.sum(block_exp, axis=0)
            lse_max = new_max
            m0 += BLOCK_M

        lse = lse_max + tl.log(lse_sum + 1e-20)
        tl.store(LSEDec_ptr + pid_bh * stride_lsed_bh + token_idx * stride_lsed_n, lse, mask=token_mask)
        t0 += BLOCK_T


@triton.jit
def semi_ar_block_z_kernel(
    BlockMax_ptr,
    BlockNum_ptr,
    PrefixMax_ptr,
    PrefixNum_ptr,
    LSEEnc_ptr,
    ZBlock_ptr,
    stride_bmax_bh,
    stride_bmax_blk,
    stride_bmax_m,
    stride_bnum_bh,
    stride_bnum_blk,
    stride_bnum_m,
    stride_bnum_d,
    stride_pmax_bh,
    stride_pmax_blk,
    stride_pmax_m,
    stride_pnum_bh,
    stride_pnum_blk,
    stride_pnum_m,
    stride_pnum_d,
    stride_lsee_bh,
    stride_lsee_blk,
    stride_lsee_m,
    stride_z_bh,
    stride_z_blk,
    stride_z_m,
    stride_z_d,
    BH,
    M,
    D_VALUE: tl.constexpr,
    NUM_BLOCKS,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    block_idx = tl.program_id(1)
    pid_md = tl.program_id(2)
    if pid_bh >= BH:
        return

    num_d_tiles = tl.cdiv(D_VALUE, BLOCK_D)
    pid_m = pid_md // num_d_tiles
    pid_d = pid_md - pid_m * num_d_tiles

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    mask_md = mask_m[:, None] & mask_d[None, :]

    block_max = tl.load(
        BlockMax_ptr + pid_bh * stride_bmax_bh + block_idx * stride_bmax_blk + m_offsets * stride_bmax_m,
        mask=mask_m,
        other=-float("inf"),
    ).to(tl.float32)
    block_num = tl.load(
        BlockNum_ptr
        + pid_bh * stride_bnum_bh
        + block_idx * stride_bnum_blk
        + m_offsets[:, None] * stride_bnum_m
        + d_offsets[None, :] * stride_bnum_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)
    prefix_max = tl.load(
        PrefixMax_ptr + pid_bh * stride_pmax_bh + block_idx * stride_pmax_blk + m_offsets * stride_pmax_m,
        mask=mask_m,
        other=-float("inf"),
    ).to(tl.float32)
    prefix_num = tl.load(
        PrefixNum_ptr
        + pid_bh * stride_pnum_bh
        + block_idx * stride_pnum_blk
        + m_offsets[:, None] * stride_pnum_m
        + d_offsets[None, :] * stride_pnum_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)
    lse_enc = tl.load(
        LSEEnc_ptr + pid_bh * stride_lsee_bh + block_idx * stride_lsee_blk + m_offsets * stride_lsee_m,
        mask=mask_m,
        other=-float("inf"),
    ).to(tl.float32)

    prefix_scale = tl.exp(prefix_max - lse_enc)
    prefix_scale = tl.where(prefix_max == -float("inf"), 0.0, prefix_scale)
    block_scale = tl.exp(block_max - lse_enc)
    block_scale = tl.where(block_max == -float("inf"), 0.0, block_scale)
    z_md = prefix_num * prefix_scale[:, None] + block_num * block_scale[:, None]

    z_ptr = ZBlock_ptr + pid_bh * stride_z_bh + block_idx * stride_z_blk
    tl.store(z_ptr + m_offsets[:, None] * stride_z_m + d_offsets[None, :] * stride_z_d, z_md, mask=mask_md)


@triton.jit
def semi_ar_lse_output_kernel(
    K_ptr,
    Q_ptr,
    QDec_ptr,
    KDec_ptr,
    ZBlock_ptr,
    O_ptr,
    stride_k_b,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_q_h,
    stride_q_m,
    stride_q_d,
    stride_qdb,
    stride_qdn,
    stride_qdh,
    stride_qdd,
    stride_kdh,
    stride_kdm,
    stride_kdd,
    stride_z_bh,
    stride_z_blk,
    stride_z_m,
    stride_z_d,
    stride_o_b,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    BH,
    M,
    N,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    scale,
    CHUNKS_PER_BLOCK,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    WEIGHT_SHARING_ENC_DEC: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    global_q_chunk = tl.program_id(1)
    pid_d = tl.program_id(2)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    block_idx = global_q_chunk // CHUNKS_PER_BLOCK
    q_chunk_start = global_q_chunk * CHUNK_SIZE

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D_VALUE

    t0 = 0
    while t0 < CHUNK_SIZE:
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        token_idx = q_chunk_start + t_offsets
        token_mask = token_idx < N

        lse_max = tl.full((BLOCK_T,), -float("inf"), tl.float32)
        lse_sum = tl.zeros((BLOCK_T,), tl.float32)
        y_num = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)

        m0 = 0
        while m0 < M:
            m_offsets = m0 + tl.arange(0, BLOCK_M)
            mask_m = m_offsets < M
            mask_md = mask_m[:, None] & mask_d[None, :]

            scores_dec = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)
            for k0 in tl.static_range(0, D_SCORE, BLOCK_K):
                d_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = d_k < D_SCORE
                if WEIGHT_SHARING_ENC_DEC:
                    q_tok = tl.load(
                        K_ptr + pid_b * stride_k_b + token_idx[:, None] * stride_k_n + pid_h * stride_k_h + d_k[None, :] * stride_k_d,
                        mask=token_mask[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    k_bank = tl.load(
                        Q_ptr + pid_h * stride_q_h + m_offsets[:, None] * stride_q_m + d_k[None, :] * stride_q_d,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                else:
                    q_tok = tl.load(
                        QDec_ptr + pid_b * stride_qdb + token_idx[:, None] * stride_qdn + pid_h * stride_qdh + d_k[None, :] * stride_qdd,
                        mask=token_mask[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    k_bank = tl.load(
                        KDec_ptr + pid_h * stride_kdh + m_offsets[:, None] * stride_kdm + d_k[None, :] * stride_kdd,
                        mask=mask_m[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                scores_dec += tl.dot(k_bank, tl.trans(q_tok), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            scores_dec = scores_dec * scale
            scores_dec = tl.where(mask_m[:, None] & token_mask[None, :], scores_dec, -float("inf"))

            z_md = tl.load(
                ZBlock_ptr
                + pid_bh * stride_z_bh
                + block_idx * stride_z_blk
                + m_offsets[:, None] * stride_z_m
                + d_offsets[None, :] * stride_z_d,
                mask=mask_md,
                other=0.0,
            ).to(tl.float32)

            block_max = tl.max(scores_dec, axis=0)
            new_max = tl.maximum(lse_max, block_max)
            new_max_safe = tl.where(new_max == -float("inf"), 0.0, new_max)
            rescale_prev = tl.where(lse_max == -float("inf"), 0.0, tl.exp(lse_max - new_max_safe))
            both_inf = new_max == -float("inf")
            rescale_prev = tl.where(both_inf & (lse_max == -float("inf")), 1.0, rescale_prev)
            exp_scores = tl.exp(scores_dec - new_max_safe[None, :])
            exp_scores = tl.where(mask_m[:, None] & token_mask[None, :], exp_scores, 0.0)

            y_num = y_num * rescale_prev[:, None] + tl.dot(
                tl.trans(exp_scores), z_md, out_dtype=tl.float32, input_precision=INPUT_PRECISION
            )
            lse_sum = lse_sum * rescale_prev + tl.sum(exp_scores, axis=0)
            lse_max = new_max
            m0 += BLOCK_M

        inv_den = 1.0 / tl.where(lse_sum > 0, lse_sum, 1.0)
        y_tile = y_num * inv_den[:, None]
        o_ptr = (
            O_ptr
            + pid_b * stride_o_b
            + token_idx[:, None] * stride_o_n
            + pid_h * stride_o_h
            + d_offsets[None, :] * stride_o_d
        )
        tl.store(o_ptr, y_tile, mask=token_mask[:, None] & mask_d[None, :])
        t0 += BLOCK_T


def _run_semi_ar_lse_output_phase(
    Q: torch.Tensor,
    K: torch.Tensor,
    *,
    Q_dec: torch.Tensor,
    K_dec: torch.Tensor,
    z_block: torch.Tensor,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict[str, object],
    weight_sharing_enc_dec: bool,
    out_dtype: torch.dtype,
    kernel_timings: dict[str, float] | None = None,
):
    B = K.size(0)
    BH = B * H
    device = K.device

    def alloc_output():
        return torch.empty((B, N, H, D_value), device=device, dtype=torch.float32)

    O = _profiled_call(device, kernel_timings, "alloc_output", alloc_output)

    def grid(meta):
        return (BH, config["NUM_GLOBAL_CHUNKS"], config["NUM_D_VALUE_BLOCKS"])

    def launch():
        semi_ar_lse_output_kernel[grid](
            K,
            Q,
            Q_dec,
            K_dec,
            z_block,
            O,
            *K.stride(),
            *Q.stride(),
            *Q_dec.stride(),
            *K_dec.stride(),
            *z_block.stride(),
            *O.stride(),
            BH,
            M,
            N,
            D_score,
            D_value,
            scale,
            config["CHUNKS_PER_BLOCK"],
            CHUNK_SIZE=config["CHUNK_SIZE"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            BLOCK_T=config["BLOCK_T"],
            INPUT_PRECISION=config["input_precision"],
            WEIGHT_SHARING_ENC_DEC=weight_sharing_enc_dec,
            H=H,
            num_warps=config["lse_output_num_warps"],
            num_stages=config["lse_output_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_lse_output", launch)
    Y = _profiled_call(device, kernel_timings, "output_cast", lambda: O.to(out_dtype))
    return Y


def _run_semi_ar_prepare_phase(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict[str, object],
    kernel_timings: dict[str, float] | None = None,
):
    BH = K.size(0) * H
    device = K.device

    def alloc_chunk_stats():
        return (
            torch.empty((BH, config["NUM_GLOBAL_CHUNKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_GLOBAL_CHUNKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_GLOBAL_CHUNKS"], M, D_value), device=device, dtype=torch.float32),
        )

    chunk_max, chunk_den, chunk_num = _profiled_call(device, kernel_timings, "alloc_chunk_stats", alloc_chunk_stats)

    def grid(meta):
        return (BH, config["NUM_GLOBAL_CHUNKS"], config["NUM_M_TILES"] * config["NUM_D_VALUE_BLOCKS"])

    def launch():
        semi_ar_chunk_prepare_kernel[grid](
            K,
            Q,
            V,
            chunk_max,
            chunk_den,
            chunk_num,
            *K.stride(),
            *Q.stride(),
            *V.stride(),
            *chunk_max.stride(),
            *chunk_den.stride(),
            *chunk_num.stride(),
            BH,
            M,
            N,
            D_score,
            D_value,
            scale,
            CHUNK_SIZE=config["CHUNK_SIZE"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            INPUT_PRECISION=config["input_precision"],
            H=H,
            num_warps=config["prepare_num_warps"],
            num_stages=config["prepare_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_chunk_prepare", launch)
    return chunk_max, chunk_den, chunk_num


def _run_semi_ar_block_reduce_phase(
    chunk_max: torch.Tensor,
    chunk_den: torch.Tensor,
    chunk_num: torch.Tensor,
    *,
    M: int,
    D_value: int,
    config: dict[str, object],
    kernel_timings: dict[str, float] | None = None,
):
    BH = chunk_max.size(0)
    device = chunk_max.device

    def alloc_block_stats():
        return (
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M, D_value), device=device, dtype=torch.float32),
        )

    block_max, block_den, block_num = _profiled_call(device, kernel_timings, "alloc_block_stats", alloc_block_stats)

    def grid(meta):
        return (BH, config["NUM_BLOCKS"], config["NUM_M_TILES"] * config["NUM_D_VALUE_BLOCKS"])

    def launch():
        semi_ar_block_reduce_from_chunks_kernel[grid](
            chunk_max,
            chunk_den,
            chunk_num,
            block_max,
            block_den,
            block_num,
            *chunk_max.stride(),
            *chunk_den.stride(),
            *chunk_num.stride(),
            *block_max.stride(),
            *block_den.stride(),
            *block_num.stride(),
            BH,
            M,
            D_value,
            config["NUM_BLOCKS"],
            config["CHUNKS_PER_BLOCK"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            num_warps=config["block_prepare_num_warps"],
            num_stages=config["block_prepare_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_block_reduce_from_chunks", launch)
    return block_max, block_den, block_num


def _run_semi_ar_block_prepare_phase(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict[str, object],
    kernel_timings: dict[str, float] | None = None,
):
    BH = K.size(0) * H
    device = K.device

    def alloc_block_stats():
        return (
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M, D_value), device=device, dtype=torch.float32),
        )

    block_max, block_den, block_num = _profiled_call(device, kernel_timings, "alloc_block_stats", alloc_block_stats)

    def grid(meta):
        return (BH, config["NUM_BLOCKS"], config["NUM_M_TILES"] * config["NUM_D_VALUE_BLOCKS"])

    def launch():
        semi_ar_block_prepare_kernel[grid](
            K,
            Q,
            V,
            block_max,
            block_den,
            block_num,
            *K.stride(),
            *Q.stride(),
            *V.stride(),
            *block_max.stride(),
            *block_den.stride(),
            *block_num.stride(),
            BH,
            M,
            N,
            D_score,
            D_value,
            scale,
            config["BLOCK_SIZE"],
            CHUNK_SIZE=config["CHUNK_SIZE"],
            CHUNKS_PER_BLOCK=config["CHUNKS_PER_BLOCK"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            INPUT_PRECISION=config["input_precision"],
            H=H,
            num_warps=config["block_prepare_num_warps"],
            num_stages=config["block_prepare_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_block_prepare", launch)
    return block_max, block_den, block_num


def _run_semi_ar_block_scan_phase(
    block_max: torch.Tensor,
    block_den: torch.Tensor,
    block_num: torch.Tensor,
    *,
    M: int,
    D_value: int,
    config: dict[str, object],
    kernel_timings: dict[str, float] | None = None,
):
    BH = block_max.size(0)
    device = block_max.device

    def alloc_prefix_stats():
        return (
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M, D_value), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
        )

    prefix_max, prefix_den, prefix_num, lse_enc = _profiled_call(
        device, kernel_timings, "alloc_scan_stats", alloc_prefix_stats
    )

    def grid(_meta):
        return (BH, config["NUM_M_TILES"], config["NUM_D_VALUE_BLOCKS"])

    def launch():
        semi_ar_block_scan_kernel[grid](
            block_max,
            block_den,
            block_num,
            prefix_max,
            prefix_den,
            prefix_num,
            lse_enc,
            *block_max.stride(),
            *block_den.stride(),
            *block_num.stride(),
            *prefix_max.stride(),
            *prefix_den.stride(),
            *prefix_num.stride(),
            *lse_enc.stride(),
            BH,
            M,
            D_value,
            config["NUM_BLOCKS"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            num_warps=config["scan_num_warps"],
            num_stages=config["scan_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_block_scan", launch)
    return prefix_max, prefix_den, prefix_num, lse_enc


def _run_semi_ar_block_z_phase(
    block_max: torch.Tensor,
    block_num: torch.Tensor,
    prefix_max: torch.Tensor,
    prefix_num: torch.Tensor,
    lse_enc: torch.Tensor,
    *,
    M: int,
    D_value: int,
    config: dict[str, object],
    kernel_timings: dict[str, float] | None = None,
):
    BH = block_max.size(0)
    device = block_max.device

    def alloc_block_z():
        return torch.empty((BH, config["NUM_BLOCKS"], M, D_value), device=device, dtype=torch.float32)

    z_block = _profiled_call(device, kernel_timings, "alloc_block_z", alloc_block_z)

    def grid(_meta):
        return (BH, config["NUM_BLOCKS"], config["NUM_M_TILES"] * config["NUM_D_VALUE_BLOCKS"])

    def launch():
        semi_ar_block_z_kernel[grid](
            block_max,
            block_num,
            prefix_max,
            prefix_num,
            lse_enc,
            z_block,
            *block_max.stride(),
            *block_num.stride(),
            *prefix_max.stride(),
            *prefix_num.stride(),
            *lse_enc.stride(),
            *z_block.stride(),
            BH,
            M,
            D_value,
            config["NUM_BLOCKS"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            num_warps=config["block_z_num_warps"],
            num_stages=config["block_z_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_block_z", launch)
    return z_block


def _run_semi_ar_decode_lse_phase(
    Q: torch.Tensor,
    K: torch.Tensor,
    *,
    Q_dec: torch.Tensor,
    K_dec: torch.Tensor,
    H: int,
    M: int,
    N: int,
    D_score: int,
    scale: float,
    config: dict[str, object],
    weight_sharing_enc_dec: bool,
    kernel_timings: dict[str, float] | None = None,
):
    BH = K.size(0) * H
    device = K.device

    def alloc_lse():
        return torch.empty((BH, N), device=device, dtype=torch.float32)

    lse_dec = _profiled_call(device, kernel_timings, "alloc_lse_dec", alloc_lse)

    def launch():
        semi_ar_decode_lse_kernel[(BH, config["NUM_GLOBAL_CHUNKS"])](
            K,
            Q,
            Q_dec,
            K_dec,
            lse_dec,
            *K.stride(),
            *Q.stride(),
            *Q_dec.stride(),
            *K_dec.stride(),
            *lse_dec.stride(),
            BH,
            M,
            N,
            D_score,
            scale,
            CHUNK_SIZE=config["CHUNK_SIZE"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_K=config["BLOCK_K"],
            BLOCK_T=config["BLOCK_T"],
            INPUT_PRECISION=config["input_precision"],
            WEIGHT_SHARING_ENC_DEC=weight_sharing_enc_dec,
            H=H,
            num_warps=config["decoder_num_warps"],
            num_stages=config["decoder_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_decode_lse", launch)
    return lse_dec


def _semi_autoregressive_forward_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    block_size: int,
    chunk_size: int,
    scale: float | None = None,
    input_precision=None,
    profile: bool = False,
    Q_dec: torch.Tensor | None = None,
    K_dec: torch.Tensor | None = None,
    return_aux: bool = False,
):
    B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="SemiAutoRegressiveFLARE")
    block_size, chunk_size, _, _, _ = _validate_block_causal_config(
        N=N,
        block_size=block_size,
        chunk_size=chunk_size,
        name="SemiAutoRegressiveFLARE",
    )
    if Q.device.type != "cuda" or K.device.type != "cuda" or V.device.type != "cuda":
        raise ValueError("SemiAutoRegressiveFLARE Triton forward requires CUDA tensors.")

    Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_semi_ar_decode_inputs(
        Q, K, Q_dec, K_dec
    )
    scale = _resolve_attn_scale(scale, D_score)
    cfg = _get_semi_ar_forward_config(
        M=M,
        N=N,
        D_score=D_score,
        D_value=D_value,
        block_size=block_size,
        chunk_size=chunk_size,
        input_precision=input_precision,
    )

    profile_data = {"forward": {}, "backward": {}} if profile else None
    timing_bucket = profile_data["forward"] if profile_data is not None else None

    block_max, block_den, block_num = _run_semi_ar_block_prepare_phase(
        Q,
        K,
        V,
        H=H,
        M=M,
        N=N,
        D_score=D_score,
        D_value=D_value,
        scale=scale,
        config=cfg,
        kernel_timings=timing_bucket,
    )

    prefix_max, prefix_den, prefix_num, lse_enc = _run_semi_ar_block_scan_phase(
        block_max,
        block_den,
        block_num,
        M=M,
        D_value=D_value,
        config=cfg,
        kernel_timings=timing_bucket,
    )
    z_block = _run_semi_ar_block_z_phase(
        block_max,
        block_num,
        prefix_max,
        prefix_num,
        lse_enc,
        M=M,
        D_value=D_value,
        config=cfg,
        kernel_timings=timing_bucket,
    )

    q_dec_tensor = K if weight_sharing_enc_dec else Q_dec
    k_dec_tensor = Q if weight_sharing_enc_dec else K_dec
    lse_dec = None
    if return_aux:
        lse_dec = _run_semi_ar_decode_lse_phase(
            Q,
            K,
            Q_dec=q_dec_tensor,
            K_dec=k_dec_tensor,
            H=H,
            M=M,
            N=N,
            D_score=D_score,
            scale=scale,
            config=cfg,
            weight_sharing_enc_dec=weight_sharing_enc_dec,
            kernel_timings=timing_bucket,
        )
    Y = _run_semi_ar_lse_output_phase(
        Q,
        K,
        Q_dec=q_dec_tensor,
        K_dec=k_dec_tensor,
        z_block=z_block,
        H=H,
        M=M,
        N=N,
        D_score=D_score,
        D_value=D_value,
        scale=scale,
        config=cfg,
        weight_sharing_enc_dec=weight_sharing_enc_dec,
        out_dtype=V.dtype,
        kernel_timings=timing_bucket,
    )

    aux = None
    if return_aux:
        aux = {
            "LSE_dec": None if lse_dec is None else lse_dec.view(B, H, N),
            "LSE_enc": lse_enc.view(B, H, cfg["NUM_BLOCKS"], M),
            "prefix_max": prefix_max.view(B, H, cfg["NUM_BLOCKS"], M),
            "prefix_den": prefix_den.view(B, H, cfg["NUM_BLOCKS"], M),
        }

    if profile_data is not None:
        _refresh_profile_totals(profile_data)
        if return_aux:
            return Y, aux, profile_data
        return Y, profile_data
    if return_aux:
        return Y, aux
    return Y


class SemiAutoRegressiveFLARE(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        scale=None,
        block_size=None,
        chunk_size=None,
        input_precision=None,
        profile=False,
        Q_dec=None,
        K_dec=None,
    ):
        del ctx
        return _semi_autoregressive_forward_triton(
            Q,
            K,
            V,
            block_size=block_size,
            chunk_size=chunk_size,
            scale=scale,
            input_precision=input_precision,
            profile=profile,
            Q_dec=Q_dec,
            K_dec=K_dec,
        )

    @staticmethod
    def backward(ctx, dY, *unused):
        del ctx, dY, unused
        raise NotImplementedError("SemiAutoRegressiveFLARE backward is not implemented yet.")


def flare_semi_autoregressive_trition(
    Q,
    K,
    V,
    *,
    block_size,
    chunk_size,
    scale=None,
    input_precision=None,
    profile: bool = False,
    Q_dec=None,
    K_dec=None,
    return_aux: bool = False,
):
    if return_aux:
        return _semi_autoregressive_forward_triton(
            Q,
            K,
            V,
            block_size=block_size,
            chunk_size=chunk_size,
            scale=scale,
            input_precision=input_precision,
            profile=profile,
            Q_dec=Q_dec,
            K_dec=K_dec,
            return_aux=True,
        )
    return SemiAutoRegressiveFLARE.apply(
        Q,
        K,
        V,
        scale,
        block_size,
        chunk_size,
        input_precision,
        profile,
        Q_dec,
        K_dec,
    )

from causal_flare._common import *
from causal_flare._reference_utils import (
    resolve_flare_causal_decode_inputs as _resolve_semi_ar_decode_inputs,
    validate_flare_qkv_layouts as _validate_flare_qkv_layouts,
)
from causal_flare.autoregressive.training import _profiled_call, _refresh_profile_totals, _resolve_forward_launch
from causal_flare.semi_autoregressive.reference import _validate_block_causal_config

_LN2 = 0.6931471824645996


def _num_storage_kernel_flags(dtype: torch.dtype) -> dict[str, bool]:
    return {
        "USE_BF16_NUM": dtype == torch.bfloat16,
        "USE_FP16_NUM": dtype == torch.float16,
    }


def _get_semi_ar_forward_bucket_defaults(
    *,
    M: int,
    D_score: int,
    D_value: int,
    block_size: int,
    chunk_size: int,
    weight_sharing_enc_dec: bool,
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
        if block_size >= 256 and M <= 128 and not mixed_d and chunk_size >= 32:
            if chunk_size >= 128:
                return {
                    "block_t": 128,
                    "prepare_launch": (4, 3),
                    "lse_output_launch": (4, 1) if weight_sharing_enc_dec else (4, 2),
                }
            return {
                "block_t": 32,
                "prepare_launch": (4, 3),
                "lse_output_launch": (2, 2),
            }
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
            "prepare_launch": (4, 3),
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
    weight_sharing_enc_dec: bool,
    input_precision=None,
) -> dict[str, object]:
    block_m = min(64, max(16, triton.next_power_of_2(M)))
    block_d = min(64, max(16, triton.next_power_of_2(D_value)))
    block_k = min(64, max(16, triton.next_power_of_2(D_score)))
    block_t_prepare = min(32, chunk_size)
    bucket_defaults = _get_semi_ar_forward_bucket_defaults(
        M=M,
        D_score=D_score,
        D_value=D_value,
        block_size=block_size,
        chunk_size=chunk_size,
        weight_sharing_enc_dec=weight_sharing_enc_dec,
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
    block_t_prepare_env = os.environ.get("FLARE_SEMI_AR_BLOCK_T_PREPARE", "").strip()
    if block_t_prepare_env:
        block_t_prepare = int(block_t_prepare_env)
    if block_t_prepare <= 0 or block_t_prepare > chunk_size or chunk_size % block_t_prepare != 0:
        raise ValueError(
            "SemiAutoRegressiveFLARE requires BLOCK_T_PREPARE to be a positive divisor of chunk_size. "
            f"Got BLOCK_T_PREPARE={block_t_prepare}, chunk_size={chunk_size}."
        )
    if block_t_prepare % 16 != 0:
        raise ValueError(
            "SemiAutoRegressiveFLARE requires BLOCK_T_PREPARE to be a multiple of 16. "
            f"Got BLOCK_T_PREPARE={block_t_prepare}."
        )

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
        "BLOCK_T_PREPARE": block_t_prepare,
        "NUM_M_TILES": triton.cdiv(M, block_m),
        "NUM_D_VALUE_BLOCKS": triton.cdiv(D_value, block_d),
        "input_precision": _normalize_input_precision(input_precision, None),
        "block_prepare_num_warps": block_prepare_warps,
        "block_prepare_num_stages": block_prepare_stages,
        "scan_num_warps": scan_warps,
        "scan_num_stages": scan_stages,
        "block_z_num_warps": block_z_warps,
        "block_z_num_stages": block_z_stages,
        "lse_output_num_warps": lse_output_warps,
        "lse_output_num_stages": lse_output_stages,
    }


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
    BLOCK_T_PREPARE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    USE_BF16_NUM: tl.constexpr,
    USE_FP16_NUM: tl.constexpr,
    H: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    num_dtype: tl.constexpr = tl.bfloat16 if USE_BF16_NUM else (tl.float16 if USE_FP16_NUM else tl.float32)
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
    block_start = block_idx * BLOCK_SIZE
    k_block_base = K_ptr + pid_b * stride_k_b + block_start * stride_k_n + pid_h * stride_k_h
    v_block_base = V_ptr + pid_b * stride_v_b + block_start * stride_v_n + pid_h * stride_v_h
    score_scale = scale * RCP_LN2
    block_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    block_den = tl.zeros((BLOCK_M,), tl.float32)
    block_num = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for local_chunk in range(CHUNKS_PER_BLOCK):
        chunk_start = block_start + local_chunk * CHUNK_SIZE
        k_chunk_base = k_block_base + local_chunk * CHUNK_SIZE * stride_k_n
        v_chunk_base = v_block_base + local_chunk * CHUNK_SIZE * stride_v_n
        t0 = 0
        while t0 < CHUNK_SIZE:
            token_offsets = t0 + tl.arange(0, BLOCK_T_PREPARE)
            token_idx = chunk_start + token_offsets
            token_mask = token_idx < N
            k_tile_base = tl.make_block_ptr(
                base=k_chunk_base,
                shape=(CHUNK_SIZE, D_SCORE),
                strides=(stride_k_n, stride_k_d),
                offsets=(t0, 0),
                block_shape=(BLOCK_T_PREPARE, BLOCK_K),
                order=(1, 0),
            )
            v_tile_ptr = tl.make_block_ptr(
                base=v_chunk_base,
                shape=(CHUNK_SIZE, D_VALUE),
                strides=(stride_v_n, stride_v_d),
                offsets=(t0, pid_d * BLOCK_D),
                block_shape=(BLOCK_T_PREPARE, BLOCK_D),
                order=(1, 0),
            )

            scores = tl.zeros((BLOCK_T_PREPARE, BLOCK_M), dtype=tl.float32)
            for k0 in tl.static_range(0, D_SCORE, BLOCK_K):
                d_k = k0 + tl.arange(0, BLOCK_K)
                mask_k = d_k < D_SCORE
                q_tile = tl.load(
                    q_base + m_offsets[:, None] * stride_q_m + d_k[None, :] * stride_q_d,
                    mask=mask_m[:, None] & mask_k[None, :],
                    other=0.0,
                )
                k_tile = tl.load(
                    tl.advance(k_tile_base, (0, k0)),
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
                scores += tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            scores = scores * score_scale
            scores = tl.where(token_mask[:, None] & mask_m[None, :], scores, -float("inf"))

            tile_max = tl.max(scores, axis=0)
            exp_scores = tl.math.exp2(scores - tile_max[None, :])
            exp_scores = tl.where(token_mask[:, None] & mask_m[None, :], exp_scores, 0.0)
            tile_den = tl.sum(exp_scores, axis=0)

            v_tile = tl.load(v_tile_ptr, boundary_check=(0, 1), padding_option="zero")
            tile_num = tl.dot(tl.trans(exp_scores.to(v_tile.dtype)), v_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)

            max_new = tl.maximum(block_max, tile_max)
            max_new_safe = tl.where(max_new == -float("inf"), 0.0, max_new)
            scale_prev = tl.where(block_max == -float("inf"), 0.0, tl.math.exp2(block_max - max_new_safe))
            scale_tile = tl.where(tile_max == -float("inf"), 0.0, tl.math.exp2(tile_max - max_new_safe))
            both_inf = max_new == -float("inf")
            scale_prev = tl.where(both_inf & (block_max == -float("inf")), 1.0, scale_prev)
            scale_tile = tl.where(both_inf & (tile_max == -float("inf")), 1.0, scale_tile)
            block_den = block_den * scale_prev + tile_den * scale_tile
            block_num = block_num * scale_prev[:, None] + tile_num * scale_tile[:, None]
            block_max = max_new
            t0 += BLOCK_T_PREPARE

    bmax_ptr = BlockMax_ptr + pid_bh * stride_bmax_bh + block_idx * stride_bmax_blk
    bden_ptr = BlockDen_ptr + pid_bh * stride_bden_bh + block_idx * stride_bden_blk
    bnum_ptr = BlockNum_ptr + pid_bh * stride_bnum_bh + block_idx * stride_bnum_blk
    tl.store(bmax_ptr + m_offsets * stride_bmax_m, block_max, mask=store_shared & mask_m)
    tl.store(bden_ptr + m_offsets * stride_bden_m, block_den, mask=store_shared & mask_m)
    tl.store(
        bnum_ptr + m_offsets[:, None] * stride_bnum_m + d_offsets[None, :] * stride_bnum_d,
        block_num.to(num_dtype),
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
    USE_BF16_NUM: tl.constexpr,
    USE_FP16_NUM: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996
    num_dtype: tl.constexpr = tl.bfloat16 if USE_BF16_NUM else (tl.float16 if USE_FP16_NUM else tl.float32)
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
            prefix_num.to(num_dtype),
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
        scale_prev = tl.where(prefix_max == -float("inf"), 0.0, tl.math.exp2(prefix_max - max_new_safe))
        scale_block = tl.where(block_max == -float("inf"), 0.0, tl.math.exp2(block_max - max_new_safe))
        both_inf = max_new == -float("inf")
        scale_prev = tl.where(both_inf & (prefix_max == -float("inf")), 1.0, scale_prev)
        scale_block = tl.where(both_inf & (block_max == -float("inf")), 1.0, scale_block)

        prefix_den = prefix_den * scale_prev + block_den * scale_block
        prefix_num = prefix_num * scale_prev[:, None] + block_num * scale_block[:, None]
        prefix_max = max_new

        lse_ptr = LSEEnc_ptr + pid_bh * stride_lsee_bh + block_idx * stride_lsee_blk
        lse = (prefix_max + tl.math.log2(tl.maximum(prefix_den, 1e-20))) * LN2
        tl.store(lse_ptr + m_offsets * stride_lsee_m, lse, mask=store_shared & mask_m)
        block_idx += 1


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
    RCP_LN2: tl.constexpr = 1.4426950408889634
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

    lse_enc_log2 = lse_enc * RCP_LN2
    prefix_scale = tl.math.exp2(prefix_max - lse_enc_log2)
    prefix_scale = tl.where(prefix_max == -float("inf"), 0.0, prefix_scale)
    block_scale = tl.math.exp2(block_max - lse_enc_log2)
    block_scale = tl.where(block_max == -float("inf"), 0.0, block_scale)
    z_md = prefix_num * prefix_scale[:, None] + block_num * block_scale[:, None]

    z_ptr = ZBlock_ptr + pid_bh * stride_z_bh + block_idx * stride_z_blk
    tl.store(z_ptr + m_offsets[:, None] * stride_z_m + d_offsets[None, :] * stride_z_d, z_md, mask=mask_md)


@triton.jit
def semi_ar_lse_output_shared_kernel(
    K_ptr,
    Q_ptr,
    ZBlock_ptr,
    LSEDec_ptr,
    O_ptr,
    stride_k_b,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_q_h,
    stride_q_m,
    stride_q_d,
    stride_z_bh,
    stride_z_blk,
    stride_z_m,
    stride_z_d,
    stride_lsed_bh,
    stride_lsed_n,
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
    H: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996
    RCP_LN2: tl.constexpr = 1.4426950408889634
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
    score_scale = scale * RCP_LN2
    q_base = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    k_base = Q_ptr + pid_h * stride_q_h

    t0 = 0
    while t0 < CHUNK_SIZE:
        token_idx = q_chunk_start + t0 + tl.arange(0, BLOCK_T)
        token_mask = token_idx < N
        q_tile_ptr = tl.make_block_ptr(
            base=q_base,
            shape=(N, D_SCORE),
            strides=(stride_k_n, stride_k_d),
            offsets=(q_chunk_start + t0, 0),
            block_shape=(BLOCK_T, D_SCORE),
            order=(1, 0),
        )
        q_tile = tl.load(q_tile_ptr, boundary_check=(0, 1), padding_option="zero")
        lse_max = tl.full((BLOCK_T,), -float("inf"), tl.float32)
        lse_sum = tl.zeros((BLOCK_T,), tl.float32)
        y_num = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)

        m0 = 0
        while m0 < M:
            m_offsets = m0 + tl.arange(0, BLOCK_M)
            mask_m = m_offsets < M
            mask_md = mask_m[:, None] & mask_d[None, :]
            k_tile_ptr = tl.make_block_ptr(
                base=k_base,
                shape=(M, D_SCORE),
                strides=(stride_q_m, stride_q_d),
                offsets=(m0, 0),
                block_shape=(BLOCK_M, D_SCORE),
                order=(1, 0),
            )
            z_tile_ptr = tl.make_block_ptr(
                base=ZBlock_ptr + pid_bh * stride_z_bh + block_idx * stride_z_blk,
                shape=(M, D_VALUE),
                strides=(stride_z_m, stride_z_d),
                offsets=(m0, pid_d * BLOCK_D),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            k_bank = tl.load(k_tile_ptr, boundary_check=(0, 1), padding_option="zero")
            z_md = tl.load(z_tile_ptr, boundary_check=(0, 1), padding_option="zero")

            scores_dec = tl.dot(q_tile, tl.trans(k_bank), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            scores_dec = scores_dec * score_scale
            scores_dec = tl.where(token_mask[:, None] & mask_m[None, :], scores_dec, -float("inf"))

            block_max = tl.max(scores_dec, axis=1)
            new_max = tl.maximum(lse_max, block_max)
            new_max_safe = tl.where(new_max == -float("inf"), 0.0, new_max)
            rescale_prev = tl.where(lse_max == -float("inf"), 0.0, tl.math.exp2(lse_max - new_max_safe))
            both_inf = new_max == -float("inf")
            rescale_prev = tl.where(both_inf & (lse_max == -float("inf")), 1.0, rescale_prev)
            exp_scores = tl.math.exp2(scores_dec - new_max_safe[:, None])
            exp_scores = tl.where(token_mask[:, None] & mask_m[None, :], exp_scores, 0.0)

            y_num = y_num * rescale_prev[:, None] + tl.dot(
                exp_scores.to(z_md.dtype), z_md, out_dtype=tl.float32, input_precision=INPUT_PRECISION
            )
            lse_sum = lse_sum * rescale_prev + tl.sum(exp_scores, axis=1)
            lse_max = new_max
            m0 += BLOCK_M

        inv_den = 1.0 / tl.where(lse_sum > 0, lse_sum, 1.0)
        y_tile = y_num * inv_den[:, None]
        lse = (lse_max + tl.math.log2(tl.maximum(lse_sum, 1e-20))) * LN2
        tl.store(LSEDec_ptr + pid_bh * stride_lsed_bh + token_idx * stride_lsed_n, lse, mask=token_mask)
        o_ptr = (
            O_ptr
            + pid_b * stride_o_b
            + token_idx[:, None] * stride_o_n
            + pid_h * stride_o_h
            + d_offsets[None, :] * stride_o_d
        )
        tl.store(o_ptr, y_tile, mask=token_mask[:, None] & mask_d[None, :])
        t0 += BLOCK_T


@triton.jit
def semi_ar_lse_output_separate_kernel(
    QDec_ptr,
    KDec_ptr,
    ZBlock_ptr,
    LSEDec_ptr,
    O_ptr,
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
    stride_lsed_bh,
    stride_lsed_n,
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
    H: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996
    RCP_LN2: tl.constexpr = 1.4426950408889634
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
    score_scale = scale * RCP_LN2
    q_base = QDec_ptr + pid_b * stride_qdb + pid_h * stride_qdh
    k_base = KDec_ptr + pid_h * stride_kdh

    t0 = 0
    while t0 < CHUNK_SIZE:
        token_idx = q_chunk_start + t0 + tl.arange(0, BLOCK_T)
        token_mask = token_idx < N
        q_tile_ptr = tl.make_block_ptr(
            base=q_base,
            shape=(N, D_SCORE),
            strides=(stride_qdn, stride_qdd),
            offsets=(q_chunk_start + t0, 0),
            block_shape=(BLOCK_T, D_SCORE),
            order=(1, 0),
        )
        q_tile = tl.load(q_tile_ptr, boundary_check=(0, 1), padding_option="zero")
        lse_max = tl.full((BLOCK_T,), -float("inf"), tl.float32)
        lse_sum = tl.zeros((BLOCK_T,), tl.float32)
        y_num = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)

        m0 = 0
        while m0 < M:
            m_offsets = m0 + tl.arange(0, BLOCK_M)
            mask_m = m_offsets < M
            mask_md = mask_m[:, None] & mask_d[None, :]
            k_tile_ptr = tl.make_block_ptr(
                base=k_base,
                shape=(M, D_SCORE),
                strides=(stride_kdm, stride_kdd),
                offsets=(m0, 0),
                block_shape=(BLOCK_M, D_SCORE),
                order=(1, 0),
            )
            z_tile_ptr = tl.make_block_ptr(
                base=ZBlock_ptr + pid_bh * stride_z_bh + block_idx * stride_z_blk,
                shape=(M, D_VALUE),
                strides=(stride_z_m, stride_z_d),
                offsets=(m0, pid_d * BLOCK_D),
                block_shape=(BLOCK_M, BLOCK_D),
                order=(1, 0),
            )
            k_bank = tl.load(k_tile_ptr, boundary_check=(0, 1), padding_option="zero")
            z_md = tl.load(z_tile_ptr, boundary_check=(0, 1), padding_option="zero")

            scores_dec = tl.dot(q_tile, tl.trans(k_bank), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            scores_dec = scores_dec * score_scale
            scores_dec = tl.where(token_mask[:, None] & mask_m[None, :], scores_dec, -float("inf"))

            block_max = tl.max(scores_dec, axis=1)
            new_max = tl.maximum(lse_max, block_max)
            new_max_safe = tl.where(new_max == -float("inf"), 0.0, new_max)
            rescale_prev = tl.where(lse_max == -float("inf"), 0.0, tl.math.exp2(lse_max - new_max_safe))
            both_inf = new_max == -float("inf")
            rescale_prev = tl.where(both_inf & (lse_max == -float("inf")), 1.0, rescale_prev)
            exp_scores = tl.math.exp2(scores_dec - new_max_safe[:, None])
            exp_scores = tl.where(token_mask[:, None] & mask_m[None, :], exp_scores, 0.0)

            y_num = y_num * rescale_prev[:, None] + tl.dot(
                exp_scores.to(z_md.dtype), z_md, out_dtype=tl.float32, input_precision=INPUT_PRECISION
            )
            lse_sum = lse_sum * rescale_prev + tl.sum(exp_scores, axis=1)
            lse_max = new_max
            m0 += BLOCK_M

        inv_den = 1.0 / tl.where(lse_sum > 0, lse_sum, 1.0)
        y_tile = y_num * inv_den[:, None]
        lse = (lse_max + tl.math.log2(tl.maximum(lse_sum, 1e-20))) * LN2
        tl.store(LSEDec_ptr + pid_bh * stride_lsed_bh + token_idx * stride_lsed_n, lse, mask=token_mask)
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

    def alloc_lse():
        return torch.empty((BH, N), device=device, dtype=torch.float32)

    def alloc_output():
        return torch.empty((B, N, H, D_value), device=device, dtype=out_dtype)

    lse_dec = _profiled_call(device, kernel_timings, "alloc_lse_dec", alloc_lse)
    O = _profiled_call(device, kernel_timings, "alloc_output", alloc_output)

    def grid(meta):
        return (BH, config["NUM_GLOBAL_CHUNKS"], config["NUM_D_VALUE_BLOCKS"])

    if weight_sharing_enc_dec:
        def launch():
            semi_ar_lse_output_shared_kernel[grid](
                K,
                Q,
                z_block,
                lse_dec,
                O,
                *K.stride(),
                *Q.stride(),
                *z_block.stride(),
                *lse_dec.stride(),
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
                H=H,
                num_warps=config["lse_output_num_warps"],
                num_stages=config["lse_output_num_stages"],
            )
    else:
        def launch():
            semi_ar_lse_output_separate_kernel[grid](
                Q_dec,
                K_dec,
                z_block,
                lse_dec,
                O,
                *Q_dec.stride(),
                *K_dec.stride(),
                *z_block.stride(),
                *lse_dec.stride(),
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
                H=H,
                num_warps=config["lse_output_num_warps"],
                num_stages=config["lse_output_num_stages"],
            )

    _profiled_call(device, kernel_timings, "semi_ar_lse_output", launch)
    return O, lse_dec


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
    num_storage_dtype: torch.dtype,
    kernel_timings: dict[str, float] | None = None,
):
    BH = K.size(0) * H
    device = K.device

    def alloc_block_stats():
        return (
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M, D_value), device=device, dtype=num_storage_dtype),
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
            BLOCK_T_PREPARE=config["BLOCK_T_PREPARE"],
            INPUT_PRECISION=config["input_precision"],
            **_num_storage_kernel_flags(num_storage_dtype),
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
    num_storage_dtype = block_num.dtype

    def alloc_prefix_stats():
        return (
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M, D_value), device=device, dtype=num_storage_dtype),
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
            **_num_storage_kernel_flags(num_storage_dtype),
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
    z_dtype = block_num.dtype

    def alloc_block_z():
        return torch.empty((BH, config["NUM_BLOCKS"], M, D_value), device=device, dtype=z_dtype)

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
        weight_sharing_enc_dec=weight_sharing_enc_dec,
        input_precision=input_precision,
    )
    num_storage_dtype = V.dtype

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
        num_storage_dtype=num_storage_dtype,
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
    Y, lse_dec = _run_semi_ar_lse_output_phase(
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

    aux = {
        "LSE_dec": lse_dec.view(B, H, N),
        "LSE_enc": lse_enc.view(B, H, cfg["NUM_BLOCKS"], M),
        "prefix_max": (prefix_max * _LN2).view(B, H, cfg["NUM_BLOCKS"], M),
        "prefix_den": prefix_den.view(B, H, cfg["NUM_BLOCKS"], M),
    }

    if profile_data is not None:
        _refresh_profile_totals(profile_data)
        return Y, aux, profile_data
    return Y, aux


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
):
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

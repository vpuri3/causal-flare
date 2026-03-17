from causal_flare._common import *
from causal_flare._reference_utils import (
    resolve_flare_causal_decode_inputs as _resolve_semi_ar_decode_inputs,
    validate_flare_qkv_layouts as _validate_flare_qkv_layouts,
)
from causal_flare.autoregressive.training import _profiled_bwd_call, _profiled_call, _refresh_profile_totals, _resolve_backward_launch, _resolve_forward_launch
from causal_flare.semi_autoregressive.reference import _block_causal_forward_pytorch, _validate_block_causal_config

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except Exception:
    TensorDescriptor = None

_SEMI_AR_TRITON_ALLOCATOR_SET = False


def _semi_ar_supports_host_descriptor() -> bool:
    if TensorDescriptor is None:
        return False
    try:
        return triton.runtime.driver.active.get_current_target().backend == "cuda"
    except Exception:
        return False


def _semi_ar_use_host_descriptors(env_var: str) -> bool:
    flag = os.environ.get(env_var, "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return _semi_ar_supports_host_descriptor()
    return _semi_ar_supports_host_descriptor()


def _semi_ar_use_prepare_host_descriptors() -> bool:
    return _semi_ar_use_host_descriptors("FLARE_SEMI_AR_PREPARE_HOST_DESC")


def _semi_ar_use_output_host_descriptors() -> bool:
    return _semi_ar_use_host_descriptors("FLARE_SEMI_AR_OUTPUT_HOST_DESC")


def _semi_ar_use_fused_prefix_kernel() -> bool:
    flag = os.environ.get("FLARE_SEMI_AR_FUSED_PREFIX", "").strip().lower()
    return flag in {"1", "true", "yes", "on"}


def _ensure_triton_allocator() -> None:
    global _SEMI_AR_TRITON_ALLOCATOR_SET
    if _SEMI_AR_TRITON_ALLOCATOR_SET:
        return

    def alloc_fn(size: int, _align: int, _stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(alloc_fn)
    _SEMI_AR_TRITON_ALLOCATOR_SET = True


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
    block_m_output = min(64, max(16, triton.next_power_of_2(M)))

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
            "block_m_output": block_m_output,
            "prepare_launch": (2, 2),
            "lse_output_launch": (2, 2),
        }

    if max_d <= 64:
        if block_size >= 256 and M <= 128 and not mixed_d and chunk_size >= 32:
            if chunk_size >= 128:
                return {
                    "block_t": 64,
                    "block_m_output": 32,
                    "prepare_launch": (4, 3),
                    "lse_output_launch": (4, 1),
                }
            return {
                "block_t": 32,
                "block_m_output": 32,
                "prepare_launch": (4, 3),
                "lse_output_launch": (2, 2),
            }
        if M > 256:
            return {
                "block_t": min(64, chunk_size),
                "block_m_output": block_m_output,
                "prepare_launch": (8, 2),
                "lse_output_launch": (4, 3),
            }
        if block_size <= 64:
            return {
                "block_t": block_t,
                "block_m_output": block_m_output,
                "prepare_launch": (4, 1),
                "lse_output_launch": (4, 1),
            }
        if block_size > 256:
            return {
                "block_t": min(64, chunk_size),
                "block_m_output": block_m_output,
                "prepare_launch": (8, 2),
                "lse_output_launch": (4, 1),
            }
        return {
            "block_m_output": block_m_output,
            "prepare_launch": (4, 3),
            "lse_output_launch": (4, 2),
            "block_t": block_t,
        }

    if M > 256:
        return {
            "block_t": block_t,
            "block_m_output": block_m_output,
            "prepare_launch": (8, 2),
            "lse_output_launch": (4, 3),
        }

    if mixed_d:
        return {
            "block_t": block_t,
            "block_m_output": block_m_output,
            "prepare_launch": (8, 2) if D_score > D_value else (4, 1),
            "lse_output_launch": (2, 2) if D_score > D_value else (4, 2),
        }

    if M >= 256:
        return {
            "block_t": block_t,
            "block_m_output": block_m_output,
            "prepare_launch": (8, 2),
            "lse_output_launch": (4, 4),
        }

    if block_size > 128:
        return {
            "block_t": block_t,
            "block_m_output": block_m_output,
            "prepare_launch": (4, 1),
            "lse_output_launch": (4, 4),
        }

    return {
        "block_t": block_t,
        "block_m_output": block_m_output,
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
    block_m_env = os.environ.get("FLARE_SEMI_AR_BLOCK_M", "").strip()
    if block_m_env:
        block_m = int(block_m_env)
    block_d_env = os.environ.get("FLARE_SEMI_AR_BLOCK_D", "").strip()
    if block_d_env:
        block_d = int(block_d_env)
    block_k_env = os.environ.get("FLARE_SEMI_AR_BLOCK_K", "").strip()
    if block_k_env:
        block_k = int(block_k_env)
    if block_m <= 0 or block_m % 16 != 0:
        raise ValueError(f"SemiAutoRegressiveFLARE requires BLOCK_M to be a positive multiple of 16. Got {block_m}.")
    if block_d <= 0 or block_d % 16 != 0:
        raise ValueError(f"SemiAutoRegressiveFLARE requires BLOCK_D to be a positive multiple of 16. Got {block_d}.")
    if block_k <= 0 or block_k % 16 != 0:
        raise ValueError(f"SemiAutoRegressiveFLARE requires BLOCK_K to be a positive multiple of 16. Got {block_k}.")
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
    block_m_output_env = os.environ.get("FLARE_SEMI_AR_BLOCK_M_OUTPUT", "").strip()
    if block_m_output_env:
        block_m_output = int(block_m_output_env)
    else:
        block_m_output = int(bucket_defaults["block_m_output"])
    if block_m_output <= 0 or block_m_output % 16 != 0:
        raise ValueError(
            f"SemiAutoRegressiveFLARE requires BLOCK_M_OUTPUT to be a positive multiple of 16. Got {block_m_output}."
        )
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
    scan_z_warps, scan_z_stages = _resolve_forward_launch(
        "semi_ar_scan_z",
        default_num_warps=8,
        default_num_stages=2,
    )
    fused_prefix_warps, fused_prefix_stages = _resolve_forward_launch(
        "semi_ar_fused_prefix",
        default_num_warps=int(bucket_defaults["prepare_launch"][0]),
        default_num_stages=int(bucket_defaults["prepare_launch"][1]),
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
        "BLOCK_M_OUTPUT": block_m_output,
        "BLOCK_D": block_d,
        "BLOCK_K": block_k,
        "BLOCK_T": block_t,
        "BLOCK_T_PREPARE": block_t_prepare,
        "NUM_M_TILES": triton.cdiv(M, block_m),
        "NUM_D_VALUE_BLOCKS": triton.cdiv(D_value, block_d),
        "input_precision": _normalize_input_precision(input_precision, None),
        "block_prepare_num_warps": block_prepare_warps,
        "block_prepare_num_stages": block_prepare_stages,
        "scan_z_num_warps": scan_z_warps,
        "scan_z_num_stages": scan_z_stages,
        "fused_prefix_num_warps": fused_prefix_warps,
        "fused_prefix_num_stages": fused_prefix_stages,
        "lse_output_num_warps": lse_output_warps,
        "lse_output_num_stages": lse_output_stages,
    }


@triton.jit
def _semi_ar_score_dot_full_panel(lhs, rhs, INPUT_PRECISION: tl.constexpr):
    return tl.dot(lhs, tl.trans(rhs), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

@triton.jit
def _semi_ar_make_2d_desc(
    base,
    rows,
    cols,
    stride_row,
    stride_col,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    return tl.make_block_ptr(
        base=base,
        shape=(rows, cols),
        strides=(stride_row, stride_col),
        offsets=(0, 0),
        block_shape=(BLOCK_ROWS, BLOCK_COLS),
        order=(1, 0),
    )


@triton.jit
def _semi_ar_load_2d_tile(desc, row_offset, col_offset):
    return tl.load(tl.advance(desc, (row_offset, col_offset)), boundary_check=(0, 1), padding_option="zero")


@triton.jit
def _semi_ar_prepare_score_dot_streamed(
    q_desc,
    k_desc,
    k_row_offset,
    m_row_offset,
    D_SCORE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_T_PREPARE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    scores = tl.zeros((BLOCK_T_PREPARE, BLOCK_M), dtype=tl.float32)
    for k0 in tl.range(0, D_SCORE, BLOCK_K):
        q_tile = _semi_ar_load_2d_tile(q_desc, m_row_offset, k0).reshape([BLOCK_M, BLOCK_K])
        k_tile = _semi_ar_load_2d_tile(k_desc, k_row_offset, k0).reshape([BLOCK_T_PREPARE, BLOCK_K])
        scores += _semi_ar_score_dot_full_panel(k_tile, q_tile, INPUT_PRECISION=INPUT_PRECISION)
    return scores


@triton.jit
def _semi_ar_prepare_score_dot_streamed_desc(
    q_desc,
    k_desc,
    pid_b,
    pid_h,
    k_row_offset,
    m_row_offset,
    D_SCORE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_T_PREPARE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    scores = tl.zeros((BLOCK_T_PREPARE, BLOCK_M), dtype=tl.float32)
    for k0 in tl.range(0, D_SCORE, BLOCK_K):
        q_tile = q_desc.load([pid_h, m_row_offset, k0]).reshape([BLOCK_M, BLOCK_K])
        k_tile = k_desc.load([pid_b, pid_h, k_row_offset, k0]).reshape([BLOCK_T_PREPARE, BLOCK_K])
        scores += _semi_ar_score_dot_full_panel(k_tile, q_tile, INPUT_PRECISION=INPUT_PRECISION)
    return scores


@triton.jit
def _semi_ar_prepare_score_dot_two_panel_desc(
    q_panel_0,
    q_panel_1,
    k_desc,
    pid_b,
    pid_h,
    k_row_offset,
    BLOCK_K: tl.constexpr,
    BLOCK_T_PREPARE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    k_panel_0 = k_desc.load([pid_b, pid_h, k_row_offset, 0]).reshape([BLOCK_T_PREPARE, BLOCK_K])
    k_panel_1 = k_desc.load([pid_b, pid_h, k_row_offset, BLOCK_K]).reshape([BLOCK_T_PREPARE, BLOCK_K])
    return _semi_ar_score_dot_full_panel(k_panel_0, q_panel_0, INPUT_PRECISION=INPUT_PRECISION) + _semi_ar_score_dot_full_panel(
        k_panel_1, q_panel_1, INPUT_PRECISION=INPUT_PRECISION
    )


@triton.jit
def _semi_ar_output_score_dot_streamed(
    q_desc,
    k_desc,
    q_row_offset,
    m_row_offset,
    D_SCORE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    scores = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
    for k0 in tl.range(0, D_SCORE, BLOCK_K):
        q_tile = tl.load(tl.advance(q_desc, (q_row_offset, k0)), boundary_check=(0, 1), padding_option="zero")
        k_tile = tl.load(tl.advance(k_desc, (m_row_offset, k0)), boundary_check=(0, 1), padding_option="zero")
        scores += _semi_ar_score_dot_full_panel(q_tile, k_tile, INPUT_PRECISION=INPUT_PRECISION)
    return scores


@triton.jit
def _semi_ar_lse_output_inner(
    q_desc,
    k_desc,
    z_desc,
    q_row_offset,
    z_col_offset,
    token_mask,
    mask_d,
    M,
    score_scale,
    D_SCORE: tl.constexpr,
    BLOCK_M_OUTPUT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    if D_SCORE <= BLOCK_K:
        q_tile = tl.load(tl.advance(q_desc, (q_row_offset, 0)), boundary_check=(0, 1), padding_option="zero")
    lse_max = tl.full((BLOCK_T,), -float("inf"), tl.float32)
    lse_sum = tl.zeros((BLOCK_T,), tl.float32)
    y_num = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)

    for m0 in tl.range(0, M, BLOCK_M_OUTPUT):
        m_offsets = m0 + tl.arange(0, BLOCK_M_OUTPUT)
        mask_m = m_offsets < M
        if D_SCORE <= BLOCK_K:
            k_bank = tl.load(tl.advance(k_desc, (m0, 0)), boundary_check=(0, 1), padding_option="zero")
            scores_dec = _semi_ar_score_dot_full_panel(q_tile, k_bank, INPUT_PRECISION=INPUT_PRECISION)
        else:
            scores_dec = _semi_ar_output_score_dot_streamed(
                q_desc,
                k_desc,
                q_row_offset,
                m0,
                D_SCORE=D_SCORE,
                BLOCK_T=BLOCK_T,
                BLOCK_M=BLOCK_M_OUTPUT,
                BLOCK_K=BLOCK_K,
                INPUT_PRECISION=INPUT_PRECISION,
            )
        z_md = tl.load(tl.advance(z_desc, (m0, z_col_offset)), boundary_check=(0, 1), padding_option="zero")

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

    return y_num, lse_sum, lse_max


@triton.jit
def _semi_ar_lse_output_inner_desc(
    q_desc,
    k_desc,
    z_desc,
    pid_b,
    pid_h,
    pid_bh,
    block_idx,
    q_row_offset,
    z_col_offset,
    token_mask,
    mask_d,
    M,
    score_scale,
    D_SCORE: tl.constexpr,
    BLOCK_M_OUTPUT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    if D_SCORE <= BLOCK_K:
        q_tile = q_desc.load([pid_b, pid_h, q_row_offset, 0]).reshape([BLOCK_T, BLOCK_K])
    lse_max = tl.full((BLOCK_T,), -float("inf"), tl.float32)
    lse_sum = tl.zeros((BLOCK_T,), tl.float32)
    y_num = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)

    for m0 in tl.range(0, M, BLOCK_M_OUTPUT):
        m_offsets = m0 + tl.arange(0, BLOCK_M_OUTPUT)
        mask_m = m_offsets < M
        if D_SCORE <= BLOCK_K:
            k_bank = k_desc.load([pid_h, m0, 0]).reshape([BLOCK_M_OUTPUT, BLOCK_K])
            scores_dec = _semi_ar_score_dot_full_panel(q_tile, k_bank, INPUT_PRECISION=INPUT_PRECISION)
        else:
            scores_dec = tl.zeros((BLOCK_T, BLOCK_M_OUTPUT), dtype=tl.float32)
            for k0 in tl.range(0, D_SCORE, BLOCK_K):
                q_panel = q_desc.load([pid_b, pid_h, q_row_offset, k0]).reshape([BLOCK_T, BLOCK_K])
                k_panel = k_desc.load([pid_h, m0, k0]).reshape([BLOCK_M_OUTPUT, BLOCK_K])
                scores_dec += _semi_ar_score_dot_full_panel(q_panel, k_panel, INPUT_PRECISION=INPUT_PRECISION)
        z_md = z_desc.load([pid_bh, block_idx, m0, z_col_offset]).reshape([BLOCK_M_OUTPUT, BLOCK_D])

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

    return y_num, lse_sum, lse_max


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
    q_desc = _semi_ar_make_2d_desc(q_base, M, D_SCORE, stride_q_m, stride_q_d, BLOCK_M, BLOCK_K)
    k_block_desc = _semi_ar_make_2d_desc(
        k_block_base,
        BLOCK_SIZE,
        D_SCORE,
        stride_k_n,
        stride_k_d,
        BLOCK_T_PREPARE,
        BLOCK_K,
    )
    v_block_desc = _semi_ar_make_2d_desc(
        v_block_base,
        BLOCK_SIZE,
        D_VALUE,
        stride_v_n,
        stride_v_d,
        BLOCK_T_PREPARE,
        BLOCK_D,
    )
    score_scale = scale * RCP_LN2
    block_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    block_den = tl.zeros((BLOCK_M,), tl.float32)
    block_num = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_row_offset = pid_m * BLOCK_M

    if D_SCORE <= BLOCK_K:
        q_full = _semi_ar_load_2d_tile(q_desc, m_row_offset, 0).reshape([BLOCK_M, BLOCK_K])

    for local_chunk in tl.range(0, CHUNKS_PER_BLOCK):
        chunk_start = block_start + local_chunk * CHUNK_SIZE
        for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T_PREPARE):
            token_offsets = t0 + tl.arange(0, BLOCK_T_PREPARE)
            token_idx = chunk_start + token_offsets
            token_mask = token_idx < N
            block_row_offset = local_chunk * CHUNK_SIZE + t0

            if D_SCORE <= BLOCK_K:
                k_tile = _semi_ar_load_2d_tile(k_block_desc, block_row_offset, 0).reshape([BLOCK_T_PREPARE, BLOCK_K])
                scores = _semi_ar_score_dot_full_panel(k_tile, q_full, INPUT_PRECISION=INPUT_PRECISION)
            else:
                scores = _semi_ar_prepare_score_dot_streamed(
                    q_desc,
                    k_block_desc,
                    block_row_offset,
                    m_row_offset,
                    D_SCORE=D_SCORE,
                    BLOCK_K=BLOCK_K,
                    BLOCK_M=BLOCK_M,
                    BLOCK_T_PREPARE=BLOCK_T_PREPARE,
                    INPUT_PRECISION=INPUT_PRECISION,
                )
            scores = scores * score_scale
            scores = tl.where(token_mask[:, None] & mask_m[None, :], scores, -float("inf"))

            tile_max = tl.max(scores, axis=0)
            exp_scores = tl.math.exp2(scores - tile_max[None, :])
            exp_scores = tl.where(token_mask[:, None] & mask_m[None, :], exp_scores, 0.0)
            tile_den = tl.sum(exp_scores, axis=0)

            v_tile = _semi_ar_load_2d_tile(v_block_desc, block_row_offset, pid_d * BLOCK_D).reshape([BLOCK_T_PREPARE, BLOCK_D])
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
def semi_ar_block_prepare_desc_kernel(
    KDesc,
    QDesc,
    VDesc,
    BlockMax_ptr,
    BlockDen_ptr,
    BlockNum_ptr,
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

    block_start = block_idx * BLOCK_SIZE
    score_scale = scale * RCP_LN2
    block_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    block_den = tl.zeros((BLOCK_M,), tl.float32)
    block_num = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    m_row_offset = pid_m * BLOCK_M

    if D_SCORE <= BLOCK_K:
        q_full = QDesc.load([pid_h, m_row_offset, 0]).reshape([BLOCK_M, BLOCK_K])
    elif D_SCORE <= 2 * BLOCK_K:
        q_panel_0 = QDesc.load([pid_h, m_row_offset, 0]).reshape([BLOCK_M, BLOCK_K])
        q_panel_1 = QDesc.load([pid_h, m_row_offset, BLOCK_K]).reshape([BLOCK_M, BLOCK_K])

    for local_chunk in tl.range(0, CHUNKS_PER_BLOCK):
        chunk_start = block_start + local_chunk * CHUNK_SIZE
        for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T_PREPARE):
            token_offsets = t0 + tl.arange(0, BLOCK_T_PREPARE)
            token_idx = chunk_start + token_offsets
            token_mask = token_idx < N

            if D_SCORE <= BLOCK_K:
                k_tile = KDesc.load([pid_b, pid_h, chunk_start + t0, 0]).reshape([BLOCK_T_PREPARE, BLOCK_K])
                scores = _semi_ar_score_dot_full_panel(k_tile, q_full, INPUT_PRECISION=INPUT_PRECISION)
            elif D_SCORE <= 2 * BLOCK_K:
                scores = _semi_ar_prepare_score_dot_two_panel_desc(
                    q_panel_0,
                    q_panel_1,
                    KDesc,
                    pid_b,
                    pid_h,
                    chunk_start + t0,
                    BLOCK_K=BLOCK_K,
                    BLOCK_T_PREPARE=BLOCK_T_PREPARE,
                    INPUT_PRECISION=INPUT_PRECISION,
                )
            else:
                scores = _semi_ar_prepare_score_dot_streamed_desc(
                    QDesc,
                    KDesc,
                    pid_b,
                    pid_h,
                    chunk_start + t0,
                    m_row_offset,
                    D_SCORE=D_SCORE,
                    BLOCK_K=BLOCK_K,
                    BLOCK_M=BLOCK_M,
                    BLOCK_T_PREPARE=BLOCK_T_PREPARE,
                    INPUT_PRECISION=INPUT_PRECISION,
                )
            scores = scores * score_scale
            scores = tl.where(token_mask[:, None] & mask_m[None, :], scores, -float("inf"))

            tile_max = tl.max(scores, axis=0)
            exp_scores = tl.math.exp2(scores - tile_max[None, :])
            exp_scores = tl.where(token_mask[:, None] & mask_m[None, :], exp_scores, 0.0)
            tile_den = tl.sum(exp_scores, axis=0)

            v_tile = VDesc.load([pid_b, pid_h, chunk_start + t0, pid_d * BLOCK_D]).reshape([BLOCK_T_PREPARE, BLOCK_D])

            max_new = tl.maximum(block_max, tile_max)
            max_new_safe = tl.where(max_new == -float("inf"), 0.0, max_new)
            scale_prev = tl.where(block_max == -float("inf"), 0.0, tl.math.exp2(block_max - max_new_safe))
            scale_tile = tl.where(tile_max == -float("inf"), 0.0, tl.math.exp2(tile_max - max_new_safe))
            both_inf = max_new == -float("inf")
            scale_prev = tl.where(both_inf & (block_max == -float("inf")), 1.0, scale_prev)
            scale_tile = tl.where(both_inf & (tile_max == -float("inf")), 1.0, scale_tile)
            block_den = block_den * scale_prev + tile_den * scale_tile
            tile_num = tl.dot(
                tl.trans((exp_scores * scale_tile[None, :]).to(v_tile.dtype)),
                v_tile,
                out_dtype=tl.float32,
                input_precision=INPUT_PRECISION,
            )
            block_num = block_num * scale_prev[:, None] + tile_num
            block_max = max_new

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
def semi_ar_block_scan_z_kernel(
    BlockMax_ptr,
    BlockDen_ptr,
    BlockNum_ptr,
    PrefixMax_ptr,
    PrefixDen_ptr,
    LSEEnc_ptr,
    ZBlock_ptr,
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
        tl.store(pmax_ptr + m_offsets * stride_pmax_m, prefix_max, mask=store_shared & mask_m)
        tl.store(pden_ptr + m_offsets * stride_pden_m, prefix_den, mask=store_shared & mask_m)

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

        full_den = prefix_den * scale_prev + block_den * scale_block
        full_num = prefix_num * scale_prev[:, None] + block_num * scale_block[:, None]
        full_lse_log2 = max_new + tl.math.log2(tl.maximum(full_den, 1e-20))
        full_lse_log2_safe = tl.where(full_lse_log2 == -float("inf"), 0.0, full_lse_log2)
        prefix_scale = tl.where(prefix_max == -float("inf"), 0.0, tl.math.exp2(prefix_max - full_lse_log2_safe))
        block_scale = tl.where(block_max == -float("inf"), 0.0, tl.math.exp2(block_max - full_lse_log2_safe))
        z_md = prefix_num * prefix_scale[:, None] + block_num * block_scale[:, None]

        lse_ptr = LSEEnc_ptr + pid_bh * stride_lsee_bh + block_idx * stride_lsee_blk
        tl.store(lse_ptr + m_offsets * stride_lsee_m, full_lse_log2 * LN2, mask=store_shared & mask_m)

        z_ptr = ZBlock_ptr + pid_bh * stride_z_bh + block_idx * stride_z_blk
        tl.store(
            z_ptr + m_offsets[:, None] * stride_z_m + d_offsets[None, :] * stride_z_d,
            z_md.to(num_dtype),
            mask=mask_md,
        )

        prefix_max = max_new
        prefix_den = full_den
        prefix_num = full_num
        block_idx += 1


@triton.jit
def semi_ar_fused_prefix_z_kernel(
    K_ptr,
    Q_ptr,
    V_ptr,
    PrefixMax_ptr,
    PrefixDen_ptr,
    LSEEnc_ptr,
    ZBlock_ptr,
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
    stride_pmax_bh,
    stride_pmax_blk,
    stride_pmax_m,
    stride_pden_bh,
    stride_pden_blk,
    stride_pden_m,
    stride_lsee_bh,
    stride_lsee_blk,
    stride_lsee_m,
    stride_z_bh,
    stride_z_blk,
    stride_z_m,
    stride_z_d,
    BH,
    M,
    N,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    scale,
    BLOCK_SIZE,
    CHUNK_SIZE: tl.constexpr,
    CHUNKS_PER_BLOCK,
    NUM_BLOCKS,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T_PREPARE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    USE_BF16_NUM: tl.constexpr,
    USE_FP16_NUM: tl.constexpr,
    H: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996
    RCP_LN2: tl.constexpr = 1.4426950408889634
    num_dtype: tl.constexpr = tl.bfloat16 if USE_BF16_NUM else (tl.float16 if USE_FP16_NUM else tl.float32)
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_d = tl.program_id(2)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    mask_md = mask_m[:, None] & mask_d[None, :]
    store_shared = pid_d == 0

    q_base = Q_ptr + pid_h * stride_q_h
    k_base = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    v_base = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h
    q_desc = _semi_ar_make_2d_desc(q_base, M, D_SCORE, stride_q_m, stride_q_d, BLOCK_M, BLOCK_K)
    k_desc = _semi_ar_make_2d_desc(k_base, N, D_SCORE, stride_k_n, stride_k_d, BLOCK_T_PREPARE, BLOCK_K)
    v_desc = _semi_ar_make_2d_desc(v_base, N, D_VALUE, stride_v_n, stride_v_d, BLOCK_T_PREPARE, BLOCK_D)
    score_scale = scale * RCP_LN2
    m_row_offset = pid_m * BLOCK_M

    prefix_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    prefix_den = tl.zeros((BLOCK_M,), tl.float32)
    prefix_num = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    if D_SCORE <= BLOCK_K:
        q_full = _semi_ar_load_2d_tile(q_desc, m_row_offset, 0).reshape([BLOCK_M, BLOCK_K])

    for block_idx in tl.range(0, NUM_BLOCKS):
        pmax_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + block_idx * stride_pmax_blk
        pden_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + block_idx * stride_pden_blk
        tl.store(pmax_ptr + m_offsets * stride_pmax_m, prefix_max, mask=store_shared & mask_m)
        tl.store(pden_ptr + m_offsets * stride_pden_m, prefix_den, mask=store_shared & mask_m)

        block_start = block_idx * BLOCK_SIZE
        for local_chunk in tl.range(0, CHUNKS_PER_BLOCK):
            chunk_start = block_start + local_chunk * CHUNK_SIZE
            for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T_PREPARE):
                token_offsets = t0 + tl.arange(0, BLOCK_T_PREPARE)
                token_idx = chunk_start + token_offsets
                token_mask = token_idx < N

                if D_SCORE <= BLOCK_K:
                    k_tile = _semi_ar_load_2d_tile(k_desc, chunk_start + t0, 0).reshape([BLOCK_T_PREPARE, BLOCK_K])
                    scores = _semi_ar_score_dot_full_panel(k_tile, q_full, INPUT_PRECISION=INPUT_PRECISION)
                else:
                    scores = _semi_ar_prepare_score_dot_streamed(
                        q_desc,
                        k_desc,
                        chunk_start + t0,
                        m_row_offset,
                        D_SCORE=D_SCORE,
                        BLOCK_K=BLOCK_K,
                        BLOCK_M=BLOCK_M,
                        BLOCK_T_PREPARE=BLOCK_T_PREPARE,
                        INPUT_PRECISION=INPUT_PRECISION,
                    )
                scores = scores * score_scale
                scores = tl.where(token_mask[:, None] & mask_m[None, :], scores, -float("inf"))

                tile_max = tl.max(scores, axis=0)
                exp_scores = tl.math.exp2(scores - tile_max[None, :])
                exp_scores = tl.where(token_mask[:, None] & mask_m[None, :], exp_scores, 0.0)
                tile_den = tl.sum(exp_scores, axis=0)

                v_tile = _semi_ar_load_2d_tile(v_desc, chunk_start + t0, pid_d * BLOCK_D).reshape([BLOCK_T_PREPARE, BLOCK_D])
                tile_num = tl.dot(
                    tl.trans(exp_scores.to(v_tile.dtype)),
                    v_tile,
                    out_dtype=tl.float32,
                    input_precision=INPUT_PRECISION,
                )

                max_new = tl.maximum(prefix_max, tile_max)
                max_new_safe = tl.where(max_new == -float("inf"), 0.0, max_new)
                scale_prev = tl.where(prefix_max == -float("inf"), 0.0, tl.math.exp2(prefix_max - max_new_safe))
                scale_tile = tl.where(tile_max == -float("inf"), 0.0, tl.math.exp2(tile_max - max_new_safe))
                both_inf = max_new == -float("inf")
                scale_prev = tl.where(both_inf & (prefix_max == -float("inf")), 1.0, scale_prev)
                scale_tile = tl.where(both_inf & (tile_max == -float("inf")), 1.0, scale_tile)
                prefix_den = prefix_den * scale_prev + tile_den * scale_tile
                prefix_num = prefix_num * scale_prev[:, None] + tile_num * scale_tile[:, None]
                prefix_max = max_new

        full_lse_log2 = prefix_max + tl.math.log2(tl.maximum(prefix_den, 1e-20))
        full_lse_log2_safe = tl.where(full_lse_log2 == -float("inf"), 0.0, full_lse_log2)
        z_scale = tl.where(prefix_max == -float("inf"), 0.0, tl.math.exp2(prefix_max - full_lse_log2_safe))
        z_md = prefix_num * z_scale[:, None]

        lse_ptr = LSEEnc_ptr + pid_bh * stride_lsee_bh + block_idx * stride_lsee_blk
        tl.store(lse_ptr + m_offsets * stride_lsee_m, full_lse_log2 * LN2, mask=store_shared & mask_m)

        z_ptr = ZBlock_ptr + pid_bh * stride_z_bh + block_idx * stride_z_blk
        tl.store(
            z_ptr + m_offsets[:, None] * stride_z_m + d_offsets[None, :] * stride_z_d,
            z_md.to(num_dtype),
            mask=mask_md,
        )


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
    BLOCK_M_OUTPUT: tl.constexpr,
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
    q_desc = _semi_ar_make_2d_desc(q_base, N, D_SCORE, stride_k_n, stride_k_d, BLOCK_T, BLOCK_K)
    k_desc = _semi_ar_make_2d_desc(k_base, M, D_SCORE, stride_q_m, stride_q_d, BLOCK_M_OUTPUT, BLOCK_K)
    z_desc = _semi_ar_make_2d_desc(
        ZBlock_ptr + pid_bh * stride_z_bh + block_idx * stride_z_blk,
        M,
        D_VALUE,
        stride_z_m,
        stride_z_d,
        BLOCK_M_OUTPUT,
        BLOCK_D,
    )

    for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T):
        token_idx = q_chunk_start + t0 + tl.arange(0, BLOCK_T)
        token_mask = token_idx < N
        y_num, lse_sum, lse_max = _semi_ar_lse_output_inner(
            q_desc,
            k_desc,
            z_desc,
            q_chunk_start + t0,
            pid_d * BLOCK_D,
            token_mask,
            mask_d,
            M,
            score_scale,
            D_SCORE=D_SCORE,
            BLOCK_M_OUTPUT=BLOCK_M_OUTPUT,
            BLOCK_D=BLOCK_D,
            BLOCK_K=BLOCK_K,
            BLOCK_T=BLOCK_T,
            INPUT_PRECISION=INPUT_PRECISION,
        )

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
    BLOCK_M_OUTPUT: tl.constexpr,
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
    q_desc = _semi_ar_make_2d_desc(q_base, N, D_SCORE, stride_qdn, stride_qdd, BLOCK_T, BLOCK_K)
    k_desc = _semi_ar_make_2d_desc(k_base, M, D_SCORE, stride_kdm, stride_kdd, BLOCK_M_OUTPUT, BLOCK_K)
    z_desc = _semi_ar_make_2d_desc(
        ZBlock_ptr + pid_bh * stride_z_bh + block_idx * stride_z_blk,
        M,
        D_VALUE,
        stride_z_m,
        stride_z_d,
        BLOCK_M_OUTPUT,
        BLOCK_D,
    )

    for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T):
        token_idx = q_chunk_start + t0 + tl.arange(0, BLOCK_T)
        token_mask = token_idx < N
        y_num, lse_sum, lse_max = _semi_ar_lse_output_inner(
            q_desc,
            k_desc,
            z_desc,
            q_chunk_start + t0,
            pid_d * BLOCK_D,
            token_mask,
            mask_d,
            M,
            score_scale,
            D_SCORE=D_SCORE,
            BLOCK_M_OUTPUT=BLOCK_M_OUTPUT,
            BLOCK_D=BLOCK_D,
            BLOCK_K=BLOCK_K,
            BLOCK_T=BLOCK_T,
            INPUT_PRECISION=INPUT_PRECISION,
        )

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


@triton.jit
def semi_ar_lse_output_desc_kernel(
    QDesc,
    KDesc,
    ZDesc,
    LSEDec_ptr,
    ODesc,
    stride_lsed_bh,
    stride_lsed_n,
    BH,
    M,
    N,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    scale,
    CHUNKS_PER_BLOCK,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_M_OUTPUT: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    USE_BF16_OUTPUT: tl.constexpr,
    USE_FP16_OUTPUT: tl.constexpr,
    H: tl.constexpr,
):
    LN2: tl.constexpr = 0.6931471824645996
    RCP_LN2: tl.constexpr = 1.4426950408889634
    out_dtype: tl.constexpr = tl.bfloat16 if USE_BF16_OUTPUT else (tl.float16 if USE_FP16_OUTPUT else tl.float32)
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

    for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T):
        token_idx = q_chunk_start + t0 + tl.arange(0, BLOCK_T)
        token_mask = token_idx < N
        y_num, lse_sum, lse_max = _semi_ar_lse_output_inner_desc(
            QDesc,
            KDesc,
            ZDesc,
            pid_b,
            pid_h,
            pid_bh,
            block_idx,
            q_chunk_start + t0,
            pid_d * BLOCK_D,
            token_mask,
            mask_d,
            M,
            score_scale,
            D_SCORE=D_SCORE,
            BLOCK_M_OUTPUT=BLOCK_M_OUTPUT,
            BLOCK_D=BLOCK_D,
            BLOCK_K=BLOCK_K,
            BLOCK_T=BLOCK_T,
            INPUT_PRECISION=INPUT_PRECISION,
        )

        inv_den = 1.0 / tl.where(lse_sum > 0, lse_sum, 1.0)
        y_tile = y_num * inv_den[:, None]
        lse = (lse_max + tl.math.log2(tl.maximum(lse_sum, 1e-20))) * LN2
        tl.store(LSEDec_ptr + pid_bh * stride_lsed_bh + token_idx * stride_lsed_n, lse, mask=token_mask)
        ODesc.store(
            [pid_b, pid_h, q_chunk_start + t0, pid_d * BLOCK_D],
            y_tile.to(out_dtype).reshape([1, 1, BLOCK_T, BLOCK_D]),
        )


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

    use_desc = (
        _semi_ar_use_output_host_descriptors()
        and M % int(config["BLOCK_M_OUTPUT"]) == 0
        and D_score % int(config["BLOCK_K"]) == 0
        and D_value % int(config["BLOCK_D"]) == 0
    )

    if use_desc:
        _ensure_triton_allocator()
        q_view = (K if weight_sharing_enc_dec else Q_dec).permute(0, 2, 1, 3)
        k_view = Q if weight_sharing_enc_dec else K_dec
        o_view = O.permute(0, 2, 1, 3)
        q_desc = TensorDescriptor(
            q_view,
            shape=list(q_view.shape),
            strides=list(q_view.stride()),
            block_shape=[1, 1, int(config["BLOCK_T"]), int(config["BLOCK_K"])],
        )
        k_desc = TensorDescriptor(
            k_view,
            shape=list(k_view.shape),
            strides=list(k_view.stride()),
            block_shape=[1, int(config["BLOCK_M_OUTPUT"]), int(config["BLOCK_K"])],
        )
        z_desc = TensorDescriptor(
            z_block,
            shape=list(z_block.shape),
            strides=list(z_block.stride()),
            block_shape=[1, 1, int(config["BLOCK_M_OUTPUT"]), int(config["BLOCK_D"])],
        )
        o_desc = TensorDescriptor(
            o_view,
            shape=list(o_view.shape),
            strides=list(o_view.stride()),
            block_shape=[1, 1, int(config["BLOCK_T"]), int(config["BLOCK_D"])],
        )

        def launch():
            semi_ar_lse_output_desc_kernel[grid](
                q_desc,
                k_desc,
                z_desc,
                lse_dec,
                o_desc,
                *lse_dec.stride(),
                BH,
                M,
                N,
                D_score,
                D_value,
                scale,
                config["CHUNKS_PER_BLOCK"],
                CHUNK_SIZE=config["CHUNK_SIZE"],
                BLOCK_D=config["BLOCK_D"],
                BLOCK_K=config["BLOCK_K"],
                BLOCK_T=config["BLOCK_T"],
                BLOCK_M_OUTPUT=config["BLOCK_M_OUTPUT"],
                INPUT_PRECISION=config["input_precision"],
                USE_BF16_OUTPUT=out_dtype == torch.bfloat16,
                USE_FP16_OUTPUT=out_dtype == torch.float16,
                H=H,
                num_warps=config["lse_output_num_warps"],
                num_stages=config["lse_output_num_stages"],
            )
    elif weight_sharing_enc_dec:
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
                BLOCK_M_OUTPUT=config["BLOCK_M_OUTPUT"],
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
                BLOCK_M_OUTPUT=config["BLOCK_M_OUTPUT"],
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

    use_desc = (
        _semi_ar_use_prepare_host_descriptors()
        and M % int(config["BLOCK_M"]) == 0
        and D_score % int(config["BLOCK_K"]) == 0
        and D_value % int(config["BLOCK_D"]) == 0
    )

    if use_desc:
        _ensure_triton_allocator()
        k_view = K.permute(0, 2, 1, 3)
        v_view = V.permute(0, 2, 1, 3)
        q_desc = TensorDescriptor(
            Q,
            shape=list(Q.shape),
            strides=list(Q.stride()),
            block_shape=[1, int(config["BLOCK_M"]), int(config["BLOCK_K"])],
        )
        k_desc = TensorDescriptor(
            k_view,
            shape=list(k_view.shape),
            strides=list(k_view.stride()),
            block_shape=[1, 1, int(config["BLOCK_T_PREPARE"]), int(config["BLOCK_K"])],
        )
        v_desc = TensorDescriptor(
            v_view,
            shape=list(v_view.shape),
            strides=list(v_view.stride()),
            block_shape=[1, 1, int(config["BLOCK_T_PREPARE"]), int(config["BLOCK_D"])],
        )

        def launch():
            semi_ar_block_prepare_desc_kernel[grid](
                k_desc,
                q_desc,
                v_desc,
                block_max,
                block_den,
                block_num,
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
    else:
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


def _run_semi_ar_block_scan_z_phase(
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
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
        )

    prefix_max, prefix_den, lse_enc = _profiled_call(
        device, kernel_timings, "alloc_scan_stats", alloc_prefix_stats
    )

    def alloc_block_z():
        return torch.empty((BH, config["NUM_BLOCKS"], M, D_value), device=device, dtype=num_storage_dtype)

    z_block = _profiled_call(device, kernel_timings, "alloc_block_z", alloc_block_z)

    def grid(_meta):
        return (BH, config["NUM_M_TILES"], config["NUM_D_VALUE_BLOCKS"])

    def launch():
        semi_ar_block_scan_z_kernel[grid](
            block_max,
            block_den,
            block_num,
            prefix_max,
            prefix_den,
            lse_enc,
            z_block,
            *block_max.stride(),
            *block_den.stride(),
            *block_num.stride(),
            *prefix_max.stride(),
            *prefix_den.stride(),
            *lse_enc.stride(),
            *z_block.stride(),
            BH,
            M,
            D_value,
            config["NUM_BLOCKS"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            **_num_storage_kernel_flags(num_storage_dtype),
            num_warps=config["scan_z_num_warps"],
            num_stages=config["scan_z_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_block_scan_z", launch)
    return prefix_max, prefix_den, lse_enc, z_block


def _run_semi_ar_fused_prefix_z_phase(
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

    def alloc_prefix_stats():
        return (
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
        )

    prefix_max, prefix_den, lse_enc = _profiled_call(device, kernel_timings, "alloc_scan_stats", alloc_prefix_stats)

    def alloc_block_z():
        return torch.empty((BH, config["NUM_BLOCKS"], M, D_value), device=device, dtype=num_storage_dtype)

    z_block = _profiled_call(device, kernel_timings, "alloc_block_z", alloc_block_z)

    def grid(_meta):
        return (BH, config["NUM_M_TILES"], config["NUM_D_VALUE_BLOCKS"])

    def launch():
        semi_ar_fused_prefix_z_kernel[grid](
            K,
            Q,
            V,
            prefix_max,
            prefix_den,
            lse_enc,
            z_block,
            *K.stride(),
            *Q.stride(),
            *V.stride(),
            *prefix_max.stride(),
            *prefix_den.stride(),
            *lse_enc.stride(),
            *z_block.stride(),
            BH,
            M,
            N,
            D_score,
            D_value,
            scale,
            config["BLOCK_SIZE"],
            CHUNK_SIZE=config["CHUNK_SIZE"],
            CHUNKS_PER_BLOCK=config["CHUNKS_PER_BLOCK"],
            NUM_BLOCKS=config["NUM_BLOCKS"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            BLOCK_T_PREPARE=config["BLOCK_T_PREPARE"],
            INPUT_PRECISION=config["input_precision"],
            **_num_storage_kernel_flags(num_storage_dtype),
            H=H,
            num_warps=config["fused_prefix_num_warps"],
            num_stages=config["fused_prefix_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_fused_prefix", launch)
    return prefix_max, prefix_den, lse_enc, z_block


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
    return_backward_state: bool = False,
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

    if _semi_ar_use_fused_prefix_kernel():
        prefix_max, prefix_den, lse_enc, z_block = _run_semi_ar_fused_prefix_z_phase(
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
    else:
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

        prefix_max, prefix_den, lse_enc, z_block = _run_semi_ar_block_scan_z_phase(
            block_max,
            block_den,
            block_num,
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
    }
    backward_state = None
    if return_backward_state:
        backward_state = {
            "prefix_max": prefix_max,
            "prefix_den": prefix_den,
            "lse_enc": lse_enc,
            "lse_dec": lse_dec,
            "z_block": z_block,
        }

    if profile_data is not None:
        _refresh_profile_totals(profile_data)
        if return_backward_state:
            return Y, aux, profile_data, backward_state
        return Y, aux, profile_data
    if return_backward_state:
        return Y, aux, backward_state
    return Y, aux


# ---------------------------------------------------------------------------
# Backward configuration
# ---------------------------------------------------------------------------


def _get_semi_ar_backward_config(
    *,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    block_size: int,
    chunk_size: int,
) -> dict[str, object]:
    """Resolve tile sizes and launch params for the semi-AR Triton backward."""
    block_m = min(64, max(16, triton.next_power_of_2(M)))
    block_d = min(64, max(16, triton.next_power_of_2(D_value)))
    block_k = min(64, max(16, triton.next_power_of_2(D_score)))
    block_t = min(32, chunk_size)

    # Allow env-var overrides consistent with forward naming convention.
    for var, name in [
        ("FLARE_SEMI_AR_BWD_BLOCK_M", "block_m"),
        ("FLARE_SEMI_AR_BWD_BLOCK_D", "block_d"),
        ("FLARE_SEMI_AR_BWD_BLOCK_K", "block_k"),
        ("FLARE_SEMI_AR_BWD_BLOCK_T", "block_t"),
    ]:
        v = os.environ.get(var, "").strip()
        if v:
            locals()[name]  # noqa – just for the linter
            if name == "block_m":
                block_m = int(v)
            elif name == "block_d":
                block_d = int(v)
            elif name == "block_k":
                block_k = int(v)
            elif name == "block_t":
                block_t = int(v)

    # Output-phase tile (decoder side dQ_dec / dK_dec sweep).
    block_m_output = min(64, max(16, triton.next_power_of_2(M)))
    v = os.environ.get("FLARE_SEMI_AR_BWD_BLOCK_M_OUTPUT", "").strip()
    if v:
        block_m_output = int(v)

    num_blocks = N // block_size
    chunks_per_block = block_size // chunk_size
    num_global_chunks = N // chunk_size

    # Launch config: default warps/stages per backward phase.
    enc_dkdv_warps, enc_dkdv_stages = _resolve_backward_launch(
        "semi_ar_enc_dkdv", default_num_warps=4, default_num_stages=2,
    )
    enc_dq_warps, enc_dq_stages = _resolve_backward_launch(
        "semi_ar_enc_dq", default_num_warps=4, default_num_stages=2,
    )
    dec_dqdk_warps, dec_dqdk_stages = _resolve_backward_launch(
        "semi_ar_dec_dqdk", default_num_warps=4, default_num_stages=2,
    )
    preprocess_warps, preprocess_stages = _resolve_backward_launch(
        "semi_ar_preprocess", default_num_warps=4, default_num_stages=1,
    )

    return {
        "NUM_BLOCKS": num_blocks,
        "NUM_GLOBAL_CHUNKS": num_global_chunks,
        "CHUNKS_PER_BLOCK": chunks_per_block,
        "BLOCK_SIZE": block_size,
        "CHUNK_SIZE": chunk_size,
        "BLOCK_M": block_m,
        "BLOCK_D": block_d,
        "BLOCK_K": block_k,
        "BLOCK_T": block_t,
        "BLOCK_M_OUTPUT": block_m_output,
        "NUM_M_TILES": triton.cdiv(M, block_m),
        "NUM_D_VALUE_BLOCKS": triton.cdiv(D_value, block_d),
        "enc_dkdv_num_warps": enc_dkdv_warps,
        "enc_dkdv_num_stages": enc_dkdv_stages,
        "enc_dq_num_warps": enc_dq_warps,
        "enc_dq_num_stages": enc_dq_stages,
        "dec_dqdk_num_warps": dec_dqdk_warps,
        "dec_dqdk_num_stages": dec_dqdk_stages,
        "preprocess_num_warps": preprocess_warps,
        "preprocess_num_stages": preprocess_stages,
    }


# ---------------------------------------------------------------------------
# Triton backward kernels
# ---------------------------------------------------------------------------


@triton.jit
def _semi_ar_bwd_preprocess_kernel(
    O_ptr,
    DO_ptr,
    Delta_ptr,
    stride_o_b, stride_o_n, stride_o_h, stride_o_d,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_delta_bh, stride_delta_n,
    BH, N,
    D_VALUE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    H: tl.constexpr,
):
    """Compute delta[bh, n] = sum_d O[b,n,h,d] * dO[b,n,h,d]."""
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    if pid_bh >= BH:
        return
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < N

    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)
    for d0 in tl.range(0, D_VALUE, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)
        mask_d = d_offsets < D_VALUE
        mask_td = mask_t[:, None] & mask_d[None, :]
        o_ptrs = (O_ptr + pid_b * stride_o_b + t_offsets[:, None] * stride_o_n
                  + pid_h * stride_o_h + d_offsets[None, :] * stride_o_d)
        do_ptrs = (DO_ptr + pid_b * stride_do_b + t_offsets[:, None] * stride_do_n
                   + pid_h * stride_do_h + d_offsets[None, :] * stride_do_d)
        o_tile = tl.load(o_ptrs, mask=mask_td, other=0.0).to(tl.float32)
        do_tile = tl.load(do_ptrs, mask=mask_td, other=0.0).to(tl.float32)
        acc += tl.sum(o_tile * do_tile, axis=1)

    tl.store(Delta_ptr + pid_bh * stride_delta_bh + t_offsets * stride_delta_n, acc, mask=mask_t)


@triton.jit
def _semi_ar_bwd_accum_dz_enc_kernel(
    QDec_ptr,       # [B, N, H, D_score] or alias of K
    KDec_ptr,       # [H, M, D_score] or alias of Q
    Q_ptr,          # [H, M, D_score]
    DO_ptr,         # [B, N, H, D_value]
    LSEDec_ptr,     # [BH, N]
    DZEnc_ptr,      # [BH, NUM_BLOCKS, M, D_value]  — output
    stride_qd_b, stride_qd_n, stride_qd_h, stride_qd_d,
    stride_kd_h, stride_kd_m, stride_kd_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_lsed_bh, stride_lsed_n,
    stride_dz_bh, stride_dz_blk, stride_dz_m, stride_dz_d,
    BH, M, N,
    scale,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_SIZE,
    CHUNK_SIZE: tl.constexpr,
    CHUNKS_PER_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
    WEIGHT_SHARING: tl.constexpr,
):
    """Accumulate dZ_enc[bh, block, m, d] = sum_{t in block} alpha[t,m] * dO[b,t,h,d].

    This is the first backward pass: we need the decoder mixing weights alpha
    to know how gradient flows from each output token into the encoder summary.
    Grid: (BH, NUM_BLOCKS, NUM_D_VALUE_BLOCKS).
    """
    RCP_LN2: tl.constexpr = 1.4426950408889634
    pid_bh = tl.program_id(0)
    block_idx = tl.program_id(1)
    pid_d = tl.program_id(2)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    score_scale = scale * RCP_LN2

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D_VALUE

    # Load K_dec (or Q if weight sharing) for all M latents
    kd_desc = _semi_ar_make_2d_desc(
        (Q_ptr if WEIGHT_SHARING else KDec_ptr) + pid_h * (stride_q_h if WEIGHT_SHARING else stride_kd_h),
        M, D_SCORE,
        stride_q_m if WEIGHT_SHARING else stride_kd_m,
        stride_q_d if WEIGHT_SHARING else stride_kd_d,
        BLOCK_M, BLOCK_K,
    )

    # Accumulate dZ_enc[m, d] over all tokens in this block
    dz_enc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    block_start = block_idx * BLOCK_SIZE
    for local_chunk in tl.range(0, CHUNKS_PER_BLOCK):
        chunk_start = block_start + local_chunk * CHUNK_SIZE
        for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T):
            t_offsets = chunk_start + t0 + tl.arange(0, BLOCK_T)
            mask_t = t_offsets < N

            # Recompute decoder scores: dec_score[t, m]
            dec_scores = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
            for k0 in tl.range(0, D_SCORE, BLOCK_K):
                k_offsets = k0 + tl.arange(0, BLOCK_K)
                mask_k = k_offsets < D_SCORE
                mask_tk = mask_t[:, None] & mask_k[None, :]
                if WEIGHT_SHARING:
                    qd_ptrs = (QDec_ptr + pid_b * stride_qd_b + t_offsets[:, None] * stride_qd_n
                               + pid_h * stride_qd_h + k_offsets[None, :] * stride_qd_d)
                else:
                    qd_ptrs = (QDec_ptr + pid_b * stride_qd_b + t_offsets[:, None] * stride_qd_n
                               + pid_h * stride_qd_h + k_offsets[None, :] * stride_qd_d)
                qd_tile = tl.load(qd_ptrs, mask=mask_tk, other=0.0)
                kd_tile = _semi_ar_load_2d_tile(kd_desc, 0, k0).reshape([BLOCK_M, BLOCK_K])
                dec_scores += tl.dot(qd_tile, tl.trans(kd_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

            dec_scores_scaled = dec_scores * score_scale

            # alpha[t, m] = exp(dec_score[t,m] - LSE_dec[t])
            lse_dec_t = tl.load(LSEDec_ptr + pid_bh * stride_lsed_bh + t_offsets * stride_lsed_n,
                                mask=mask_t, other=0.0)
            alpha = tl.math.exp2(dec_scores_scaled - lse_dec_t[:, None] * RCP_LN2)
            alpha = tl.where(mask_t[:, None] & mask_m[None, :], alpha, 0.0)

            # Load dO[b, t, h, d_tile]
            do_ptrs = (DO_ptr + pid_b * stride_do_b + t_offsets[:, None] * stride_do_n
                       + pid_h * stride_do_h + d_offsets[None, :] * stride_do_d)
            do_tile = tl.load(do_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            # dZ_enc[m, d] += alpha[t, m]^T @ dO[t, d]
            dz_enc += tl.dot(tl.trans(alpha.to(do_tile.dtype)), do_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    # Store
    dz_ptr = DZEnc_ptr + pid_bh * stride_dz_bh + block_idx * stride_dz_blk
    mask_md = mask_m[:, None] & mask_d[None, :]
    tl.store(dz_ptr + m_offsets[:, None] * stride_dz_m + d_offsets[None, :] * stride_dz_d,
             dz_enc, mask=mask_md)


@triton.jit
def _semi_ar_bwd_enc_dkdv_from_dz_kernel(
    Q_ptr,          # [H, M, D_score]
    K_ptr,          # [B, N, H, D_score]
    V_ptr,          # [B, N, H, D_value]
    LSEEnc_ptr,     # [BH, NUM_BLOCKS, M]
    DZEnc_ptr,      # [BH, NUM_BLOCKS, M, D_value]
    DK_ptr,         # [B, N, H, D_score]
    DV_ptr,         # [B, N, H, D_value]
    stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_lsee_bh, stride_lsee_blk, stride_lsee_m,
    stride_dz_bh, stride_dz_blk, stride_dz_m, stride_dz_d,
    stride_dk_b, stride_dk_n, stride_dk_h, stride_dk_d,
    stride_dv_b, stride_dv_n, stride_dv_h, stride_dv_d,
    BH, M, N,
    scale,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_SIZE,
    CHUNK_SIZE: tl.constexpr,
    CHUNKS_PER_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    """Given dZ_enc[bh, block, m, d], compute dK and dV for each token.

    For token t in block g:
      P_enc[m, t] = exp(scale * K[t] @ Q[m] - LSE_enc[g, m])
      dV[t, d] = sum_m P_enc[m, t] * dZ_enc[g, m, d]
      dS_enc[m, t] = P_enc[m, t] * (sum_d V[t,d] * dZ_enc[g,m,d] - delta_enc[g,m])
      dK[t, k] = sum_m dS_enc[m, t] * scale * Q[m, k]

    where delta_enc[g,m] = sum_{t' in block g} P_enc[m,t'] * sum_d V[t',d] * dZ_enc[g,m,d].

    To avoid needing delta_enc pre-computed, we use the FA2 identity:
      delta_enc[g,m] = sum_{t'} P_enc[m,t'] * vdz[t']
    where vdz[t'] = sum_d V[t',d] * dZ_enc[g,m,d].

    We compute delta_enc on-the-fly by doing a reduction over the block first.

    Grid: (BH, NUM_GLOBAL_CHUNKS).
    """
    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471824645996
    pid_bh = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    block_idx = pid_chunk // CHUNKS_PER_BLOCK
    chunk_start = pid_chunk * CHUNK_SIZE
    score_scale = scale * RCP_LN2

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    # Load LSE_enc[bh, block, m]
    lsee_base = LSEEnc_ptr + pid_bh * stride_lsee_bh + block_idx * stride_lsee_blk
    lse_enc_m = tl.load(lsee_base + m_offsets * stride_lsee_m, mask=mask_m, other=0.0)
    lse_enc_log2 = lse_enc_m * RCP_LN2

    # Build Q_enc desc
    q_enc_desc = _semi_ar_make_2d_desc(
        Q_ptr + pid_h * stride_q_h, M, D_SCORE, stride_q_m, stride_q_d, BLOCK_M, BLOCK_K,
    )

    # Load dZ_enc[bh, block, m, :] in tiles
    dz_base = DZEnc_ptr + pid_bh * stride_dz_bh + block_idx * stride_dz_blk

    for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T):
        t_offsets = chunk_start + t0 + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < N

        # Recompute encoder scores: enc_scores[t, m]
        enc_scores = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
        for k0 in tl.range(0, D_SCORE, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < D_SCORE
            mask_tk = mask_t[:, None] & mask_k[None, :]
            k_ptrs = (K_ptr + pid_b * stride_k_b + t_offsets[:, None] * stride_k_n
                      + pid_h * stride_k_h + k_offsets[None, :] * stride_k_d)
            k_tile = tl.load(k_ptrs, mask=mask_tk, other=0.0)
            q_tile = _semi_ar_load_2d_tile(q_enc_desc, 0, k0).reshape([BLOCK_M, BLOCK_K])
            enc_scores += tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        p_enc = tl.math.exp2(enc_scores * score_scale - lse_enc_log2[None, :])
        p_enc = tl.where(mask_t[:, None] & mask_m[None, :], p_enc, 0.0)

        # Compute dV[t, d] = sum_m P_enc[t, m] * dZ_enc[m, d]
        # and    vdz[t, m] = sum_d V[t, d] * dZ_enc[m, d]  (for dS computation)
        dv_acc = tl.zeros((BLOCK_T, BLOCK_D), dtype=tl.float32)
        vdz = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)

        for d0 in tl.range(0, D_VALUE, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D_VALUE

            # dZ_enc[m, d_tile]
            dz_ptrs = dz_base + m_offsets[:, None] * stride_dz_m + d_offsets[None, :] * stride_dz_d
            dz_tile = tl.load(dz_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            # dV[t, d] += P_enc[t, m] @ dZ_enc[m, d]
            if d0 == 0:
                dv_acc = tl.dot(p_enc.to(dz_tile.dtype), dz_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            else:
                # For tiles beyond the first BLOCK_D, we need to accumulate into
                # a wider buffer. Since we store dV per BLOCK_D tile anyway, store now.
                dv_tile = tl.dot(p_enc.to(dz_tile.dtype), dz_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                dv_ptrs = (DV_ptr + pid_b * stride_dv_b + t_offsets[:, None] * stride_dv_n
                           + pid_h * stride_dv_h + d_offsets[None, :] * stride_dv_d)
                # Atomic add since other chunks in same block also contribute
                tl.atomic_add(dv_ptrs, dv_tile, mask=mask_t[:, None] & mask_d[None, :])

            # V[t, d_tile]
            v_ptrs = (V_ptr + pid_b * stride_v_b + t_offsets[:, None] * stride_v_n
                      + pid_h * stride_v_h + d_offsets[None, :] * stride_v_d)
            v_tile = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            # vdz[t, m] += V[t, d] @ dZ_enc[m, d].T
            vdz += tl.dot(v_tile, tl.trans(dz_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        # Store the first BLOCK_D tile of dV (use atomic_add for safety)
        d0_offsets = tl.arange(0, BLOCK_D)
        mask_d0 = d0_offsets < D_VALUE
        dv_ptrs = (DV_ptr + pid_b * stride_dv_b + t_offsets[:, None] * stride_dv_n
                   + pid_h * stride_dv_h + d0_offsets[None, :] * stride_dv_d)
        tl.atomic_add(dv_ptrs, dv_acc, mask=mask_t[:, None] & mask_d0[None, :])

        # dS_enc[t, m] = P_enc[t, m] * (vdz[t, m] - delta_enc_t[t])
        # where delta_enc_t[t] = sum_m P_enc[t, m] * vdz[t, m]
        delta_enc_t = tl.sum(p_enc * vdz, axis=1)  # [BLOCK_T]
        ds_enc = p_enc * (vdz - delta_enc_t[:, None])  # [BLOCK_T, BLOCK_M]

        # dK[t, k] = sum_m dS_enc[t, m] * scale * Q[m, k]
        # ds_enc is P*(vdz-delta) which is dL/d(logit_log2) / ln(2).
        # Full chain: dL/d(K) = ds_enc * ln(2) * score_scale * Q = ds_enc * scale * Q
        # since ln(2) * RCP_LN2 = 1.
        for k0 in tl.range(0, D_SCORE, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < D_SCORE
            q_tile = _semi_ar_load_2d_tile(q_enc_desc, 0, k0).reshape([BLOCK_M, BLOCK_K])
            dk_tile = tl.dot(ds_enc.to(q_tile.dtype), q_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale
            dk_ptrs = (DK_ptr + pid_b * stride_dk_b + t_offsets[:, None] * stride_dk_n
                       + pid_h * stride_dk_h + k_offsets[None, :] * stride_dk_d)
            tl.atomic_add(dk_ptrs, dk_tile, mask=mask_t[:, None] & mask_k[None, :])


@triton.jit
def _semi_ar_bwd_enc_dq_kernel(
    K_ptr,          # [B, N, H, D_score]
    V_ptr,          # [B, N, H, D_value]
    Q_ptr,          # [H, M, D_score]
    LSEEnc_ptr,     # [BH, NUM_BLOCKS, M]
    DZEnc_ptr,      # [BH, NUM_BLOCKS, M, D_value]
    DQ_ptr,         # [H, M, D_score]  — output, atomically accumulated
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_lsee_bh, stride_lsee_blk, stride_lsee_m,
    stride_dz_bh, stride_dz_blk, stride_dz_m, stride_dz_d,
    stride_dq_h, stride_dq_m, stride_dq_d,
    BH, M, N,
    scale,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_SIZE,
    CHUNK_SIZE: tl.constexpr,
    CHUNKS_PER_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    """Compute dQ_enc[h, m, k] from the encoder path.

    dS_enc[g, m, t] = P_enc[g,m,t] * (vdz[g,m,t] - delta_enc[g,m])
    dQ_enc[h, m, k] = sum_{b,g,t} dS_enc[b,g,m,t] * scale * K[b,t,h,k]

    Grid: (BH, NUM_BLOCKS).
    Each program accumulates over all tokens in one block for all M latents.
    """
    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471824645996
    pid_bh = tl.program_id(0)
    block_idx = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    score_scale = scale * RCP_LN2

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    # Load LSE_enc[bh, block, m]
    lsee_base = LSEEnc_ptr + pid_bh * stride_lsee_bh + block_idx * stride_lsee_blk
    lse_enc_m = tl.load(lsee_base + m_offsets * stride_lsee_m, mask=mask_m, other=0.0)
    lse_enc_log2 = lse_enc_m * RCP_LN2

    q_enc_desc = _semi_ar_make_2d_desc(
        Q_ptr + pid_h * stride_q_h, M, D_SCORE, stride_q_m, stride_q_d, BLOCK_M, BLOCK_K,
    )
    dz_base = DZEnc_ptr + pid_bh * stride_dz_bh + block_idx * stride_dz_blk

    # Accumulate dQ contribution: [BLOCK_M, BLOCK_K] per k-tile
    block_start = block_idx * BLOCK_SIZE
    for k0 in tl.range(0, D_SCORE, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D_SCORE
        dq_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for local_chunk in tl.range(0, CHUNKS_PER_BLOCK):
            chunk_start = block_start + local_chunk * CHUNK_SIZE
            for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T):
                t_offsets = chunk_start + t0 + tl.arange(0, BLOCK_T)
                mask_t = t_offsets < N

                # Recompute enc_scores[t, m]
                enc_scores = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
                for ki in tl.range(0, D_SCORE, BLOCK_K):
                    ki_offsets = ki + tl.arange(0, BLOCK_K)
                    mask_ki = ki_offsets < D_SCORE
                    k_ptrs = (K_ptr + pid_b * stride_k_b + t_offsets[:, None] * stride_k_n
                              + pid_h * stride_k_h + ki_offsets[None, :] * stride_k_d)
                    k_tile = tl.load(k_ptrs, mask=mask_t[:, None] & mask_ki[None, :], other=0.0)
                    q_tile = _semi_ar_load_2d_tile(q_enc_desc, 0, ki).reshape([BLOCK_M, BLOCK_K])
                    enc_scores += tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

                p_enc = tl.math.exp2(enc_scores * score_scale - lse_enc_log2[None, :])
                p_enc = tl.where(mask_t[:, None] & mask_m[None, :], p_enc, 0.0)

                # Compute vdz[t, m] = sum_d V[t, d] * dZ_enc[m, d]
                vdz = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
                for d0 in tl.range(0, D_VALUE, BLOCK_D):
                    d_offsets = d0 + tl.arange(0, BLOCK_D)
                    mask_d = d_offsets < D_VALUE
                    v_ptrs = (V_ptr + pid_b * stride_v_b + t_offsets[:, None] * stride_v_n
                              + pid_h * stride_v_h + d_offsets[None, :] * stride_v_d)
                    v_tile = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                    dz_ptrs = dz_base + m_offsets[:, None] * stride_dz_m + d_offsets[None, :] * stride_dz_d
                    dz_tile = tl.load(dz_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                    vdz += tl.dot(v_tile, tl.trans(dz_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

                # dS_enc[t, m] = P_enc[t,m] * (vdz[t,m] - delta_enc_t[t])
                delta_enc_t = tl.sum(p_enc * vdz, axis=1)
                ds_enc = p_enc * (vdz - delta_enc_t[:, None])

                # dQ[m, k] += dS_enc[t, m].T @ K[t, k] * scale
                k_ptrs = (K_ptr + pid_b * stride_k_b + t_offsets[:, None] * stride_k_n
                          + pid_h * stride_k_h + k_offsets[None, :] * stride_k_d)
                k_tile_k = tl.load(k_ptrs, mask=mask_t[:, None] & mask_k[None, :], other=0.0)
                dq_acc += tl.dot(tl.trans(ds_enc.to(k_tile_k.dtype)), k_tile_k, out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        # Atomically accumulate dQ (shared across B and blocks)
        # Same scale derivation as dK: ds_enc * ln(2) * score_scale * K = ds_enc * scale * K
        dq_acc = dq_acc * scale
        dq_ptrs = (DQ_ptr + pid_h * stride_dq_h
                   + m_offsets[:, None] * stride_dq_m + k_offsets[None, :] * stride_dq_d)
        tl.atomic_add(dq_ptrs, dq_acc, mask=mask_m[:, None] & mask_k[None, :])


@triton.jit
def _semi_ar_bwd_dec_kernel(
    QDec_ptr,       # [B, N, H, D_score]  (or K if weight sharing)
    KDec_ptr,       # [H, M, D_score]     (or Q if weight sharing)
    Q_ptr,          # [H, M, D_score]
    K_ptr,          # [B, N, H, D_score]
    V_ptr,          # [B, N, H, D_value]
    Y_ptr,          # [B, N, H, D_value]
    DO_ptr,         # [B, N, H, D_value]
    LSEDec_ptr,     # [BH, N]
    Delta_ptr,      # [BH, N]
    ZBlock_ptr,     # [BH, NUM_BLOCKS, M, D_value]
    DQDec_ptr,      # [B, N, H, D_score]  — output (or contribute to DK)
    DKDec_ptr,      # [H, M, D_score]     — output (or contribute to DQ)
    DK_ptr,         # [B, N, H, D_score]  — for weight-sharing accumulation
    DQ_ptr,         # [H, M, D_score]     — for weight-sharing accumulation
    stride_qd_b, stride_qd_n, stride_qd_h, stride_qd_d,
    stride_kd_h, stride_kd_m, stride_kd_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_y_b, stride_y_n, stride_y_h, stride_y_d,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_lsed_bh, stride_lsed_n,
    stride_delta_bh, stride_delta_n,
    stride_z_bh, stride_z_blk, stride_z_m, stride_z_d,
    stride_dqd_b, stride_dqd_n, stride_dqd_h, stride_dqd_d,
    stride_dkd_h, stride_dkd_m, stride_dkd_d,
    stride_dk_b, stride_dk_n, stride_dk_h, stride_dk_d,
    stride_dq_h, stride_dq_m, stride_dq_d,
    BH, M, N,
    scale,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_SIZE,
    CHUNK_SIZE: tl.constexpr,
    CHUNKS_PER_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
    WEIGHT_SHARING: tl.constexpr,
    SEPARATE_Q_DEC: tl.constexpr,
    SEPARATE_K_DEC: tl.constexpr,
):
    """Compute decoder-side gradients: dQ_dec and dK_dec.

    The decoder path computes:
      alpha[t, m] = softmax_m(scale * Q_dec[t] @ K_dec[m])
      Y[t, d] = sum_m alpha[t, m] * z_block[block(t), m, d]

    So: dAlpha[t, m] = sum_d dO[t, d] * z_block[block(t), m, d]
    And dS_dec[t, m] = alpha[t, m] * (dAlpha[t, m] - delta_dec[t])
    where delta_dec[t] = sum_m alpha[t, m] * dAlpha[t, m]
                       = sum_d Y[t, d] * dO[t, d] = Delta[t]  (precomputed)

    Then:
      dQ_dec[t, k] = sum_m dS_dec[t, m] * scale * K_dec[m, k]
      dK_dec[m, k] = sum_{b, t} dS_dec[t, m] * scale * Q_dec[t, k]

    Grid: (BH, NUM_GLOBAL_CHUNKS).
    """
    RCP_LN2: tl.constexpr = 1.4426950408889634
    LN2: tl.constexpr = 0.6931471824645996
    pid_bh = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    block_idx = pid_chunk // CHUNKS_PER_BLOCK
    chunk_start = pid_chunk * CHUNK_SIZE
    score_scale = scale * RCP_LN2

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    # K_dec descriptor (or Q if weight sharing)
    kd_base = (Q_ptr + pid_h * stride_q_h) if WEIGHT_SHARING else (KDec_ptr + pid_h * stride_kd_h)
    kd_stride_m = stride_q_m if WEIGHT_SHARING else stride_kd_m
    kd_stride_d = stride_q_d if WEIGHT_SHARING else stride_kd_d
    kd_desc = _semi_ar_make_2d_desc(kd_base, M, D_SCORE, kd_stride_m, kd_stride_d, BLOCK_M, BLOCK_K)

    # z_block base for this block
    z_base = ZBlock_ptr + pid_bh * stride_z_bh + block_idx * stride_z_blk

    for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T):
        t_offsets = chunk_start + t0 + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < N

        # Recompute decoder scores: dec_scores[t, m]
        dec_scores = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
        for k0 in tl.range(0, D_SCORE, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < D_SCORE
            mask_tk = mask_t[:, None] & mask_k[None, :]
            if WEIGHT_SHARING:
                qd_ptrs = (K_ptr + pid_b * stride_k_b + t_offsets[:, None] * stride_k_n
                           + pid_h * stride_k_h + k_offsets[None, :] * stride_k_d)
            else:
                qd_ptrs = (QDec_ptr + pid_b * stride_qd_b + t_offsets[:, None] * stride_qd_n
                           + pid_h * stride_qd_h + k_offsets[None, :] * stride_qd_d)
            qd_tile = tl.load(qd_ptrs, mask=mask_tk, other=0.0)
            kd_tile = _semi_ar_load_2d_tile(kd_desc, 0, k0).reshape([BLOCK_M, BLOCK_K])
            dec_scores += tl.dot(qd_tile, tl.trans(kd_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        dec_scores_scaled = dec_scores * score_scale

        # alpha[t, m] = exp(dec_score - LSE_dec[t])
        lse_dec_t = tl.load(LSEDec_ptr + pid_bh * stride_lsed_bh + t_offsets * stride_lsed_n,
                            mask=mask_t, other=0.0)
        alpha = tl.math.exp2(dec_scores_scaled - lse_dec_t[:, None] * RCP_LN2)
        alpha = tl.where(mask_t[:, None] & mask_m[None, :], alpha, 0.0)

        # dAlpha[t, m] = sum_d dO[t,d] * z_block[block, m, d]
        dalpha = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
        for d0 in tl.range(0, D_VALUE, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D_VALUE
            do_ptrs = (DO_ptr + pid_b * stride_do_b + t_offsets[:, None] * stride_do_n
                       + pid_h * stride_do_h + d_offsets[None, :] * stride_do_d)
            do_tile = tl.load(do_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            z_ptrs = z_base + m_offsets[:, None] * stride_z_m + d_offsets[None, :] * stride_z_d
            z_tile = tl.load(z_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            dalpha += tl.dot(do_tile, tl.trans(z_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        # delta_dec[t] = sum_d Y[t,d] * dO[t,d] = Delta[t]  (already precomputed)
        delta_t = tl.load(Delta_ptr + pid_bh * stride_delta_bh + t_offsets * stride_delta_n,
                          mask=mask_t, other=0.0)

        # dS_dec[t, m] = alpha[t,m] * (dAlpha[t,m] - delta_t[t])
        ds_dec = alpha * (dalpha - delta_t[:, None])

        # dQ_dec[t, k] = sum_m dS_dec[t, m] * scale * K_dec[m, k]
        # dK_dec[m, k] += sum_t dS_dec[t, m].T * scale * Q_dec[t, k]
        for k0 in tl.range(0, D_SCORE, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < D_SCORE
            kd_tile = _semi_ar_load_2d_tile(kd_desc, 0, k0).reshape([BLOCK_M, BLOCK_K])

            # dQ_dec[t, k] = dS_dec[t, m] @ K_dec[m, k] * scale
            dqd_tile = tl.dot(ds_dec.to(kd_tile.dtype), kd_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale

            if WEIGHT_SHARING:
                # dQ_dec contributes to dK (since Q_dec = K in weight sharing)
                dk_ptrs = (DK_ptr + pid_b * stride_dk_b + t_offsets[:, None] * stride_dk_n
                           + pid_h * stride_dk_h + k_offsets[None, :] * stride_dk_d)
                tl.atomic_add(dk_ptrs, dqd_tile, mask=mask_t[:, None] & mask_k[None, :])
            elif SEPARATE_Q_DEC:
                dqd_ptrs = (DQDec_ptr + pid_b * stride_dqd_b + t_offsets[:, None] * stride_dqd_n
                            + pid_h * stride_dqd_h + k_offsets[None, :] * stride_dqd_d)
                tl.atomic_add(dqd_ptrs, dqd_tile, mask=mask_t[:, None] & mask_k[None, :])

            # dK_dec[m, k] += dS_dec[t, m].T @ Q_dec[t, k]
            mask_tk = mask_t[:, None] & mask_k[None, :]
            if WEIGHT_SHARING:
                qd_ptrs = (K_ptr + pid_b * stride_k_b + t_offsets[:, None] * stride_k_n
                           + pid_h * stride_k_h + k_offsets[None, :] * stride_k_d)
            else:
                qd_ptrs = (QDec_ptr + pid_b * stride_qd_b + t_offsets[:, None] * stride_qd_n
                           + pid_h * stride_qd_h + k_offsets[None, :] * stride_qd_d)
            qd_tile = tl.load(qd_ptrs, mask=mask_tk, other=0.0)
            dkd_tile = tl.dot(tl.trans(ds_dec.to(qd_tile.dtype)), qd_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale

            if WEIGHT_SHARING:
                # dK_dec contributes to dQ (since K_dec = Q in weight sharing)
                dq_ptrs = (DQ_ptr + pid_h * stride_dq_h
                           + m_offsets[:, None] * stride_dq_m + k_offsets[None, :] * stride_dq_d)
                tl.atomic_add(dq_ptrs, dkd_tile, mask=mask_m[:, None] & mask_k[None, :])
            elif SEPARATE_K_DEC:
                dkd_ptrs = (DKDec_ptr + pid_h * stride_dkd_h
                            + m_offsets[:, None] * stride_dkd_m + k_offsets[None, :] * stride_dkd_d)
                tl.atomic_add(dkd_ptrs, dkd_tile, mask=mask_m[:, None] & mask_k[None, :])


@triton.jit
def _semi_ar_bwd_summary_scalar_scan_kernel(
    LSEEnc_ptr,
    ZBlock_ptr,
    DZBlock_ptr,
    ABlock_ptr,
    DABlock_ptr,
    stride_lsee_bh, stride_lsee_blk, stride_lsee_m,
    stride_z_bh, stride_z_blk, stride_z_m, stride_z_d,
    stride_dz_bh, stride_dz_blk, stride_dz_m, stride_dz_d,
    stride_ab_bh, stride_ab_blk, stride_ab_m,
    stride_da_bh, stride_da_blk, stride_da_m,
    BH, M,
    D_VALUE: tl.constexpr,
    NUM_BLOCKS,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    if pid_bh >= BH:
        return

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    suffix_da = tl.zeros((BLOCK_M,), dtype=tl.float32)

    block_idx = NUM_BLOCKS
    while block_idx > 0:
        block_idx -= 1
        lse_ptr = LSEEnc_ptr + pid_bh * stride_lsee_bh + block_idx * stride_lsee_blk
        lse_curr = tl.load(lse_ptr + m_offsets * stride_lsee_m, mask=mask_m, other=0.0).to(tl.float32)
        a_prefix = tl.math.exp2(lse_curr * RCP_LN2)

        if block_idx > 0:
            lse_prev_ptr = LSEEnc_ptr + pid_bh * stride_lsee_bh + (block_idx - 1) * stride_lsee_blk
            lse_prev = tl.load(lse_prev_ptr + m_offsets * stride_lsee_m, mask=mask_m, other=0.0).to(tl.float32)
            a_prev = tl.math.exp2(lse_prev * RCP_LN2)
        else:
            a_prev = tl.zeros((BLOCK_M,), dtype=tl.float32)

        a_block = tl.maximum(a_prefix - a_prev, 0.0)
        dot_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for d0 in tl.range(0, D_VALUE, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D_VALUE
            z_ptrs = ZBlock_ptr + pid_bh * stride_z_bh + block_idx * stride_z_blk + m_offsets[:, None] * stride_z_m + d_offsets[None, :] * stride_z_d
            dz_ptrs = DZBlock_ptr + pid_bh * stride_dz_bh + block_idx * stride_dz_blk + m_offsets[:, None] * stride_dz_m + d_offsets[None, :] * stride_dz_d
            z_tile = tl.load(z_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            dz_tile = tl.load(dz_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            dot_acc += tl.sum(z_tile * dz_tile, axis=1)

        inv_a = tl.where(a_prefix > 0, 1.0 / a_prefix, 0.0)
        dA_prefix = -dot_acc * inv_a
        suffix_da += dA_prefix

        tl.store(ABlock_ptr + pid_bh * stride_ab_bh + block_idx * stride_ab_blk + m_offsets * stride_ab_m, a_block, mask=mask_m)
        tl.store(DABlock_ptr + pid_bh * stride_da_bh + block_idx * stride_da_blk + m_offsets * stride_da_m, suffix_da, mask=mask_m)


@triton.jit
def _semi_ar_bwd_summary_vector_scan_kernel(
    LSEEnc_ptr,
    DZBlock_ptr,
    DBBlock_ptr,
    stride_lsee_bh, stride_lsee_blk, stride_lsee_m,
    stride_dz_bh, stride_dz_blk, stride_dz_m, stride_dz_d,
    stride_db_bh, stride_db_blk, stride_db_m, stride_db_d,
    BH, M,
    D_VALUE: tl.constexpr,
    NUM_BLOCKS,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    pid_bh = tl.program_id(0)
    pid_md = tl.program_id(1)
    if pid_bh >= BH:
        return

    num_d_tiles = tl.cdiv(D_VALUE, BLOCK_D)
    pid_m = pid_md // num_d_tiles
    pid_d = pid_md - pid_m * num_d_tiles
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    suffix_db = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    block_idx = NUM_BLOCKS
    while block_idx > 0:
        block_idx -= 1
        lse_ptr = LSEEnc_ptr + pid_bh * stride_lsee_bh + block_idx * stride_lsee_blk
        lse_curr = tl.load(lse_ptr + m_offsets * stride_lsee_m, mask=mask_m, other=0.0).to(tl.float32)
        a_prefix = tl.math.exp2(lse_curr * RCP_LN2)
        inv_a = tl.where(a_prefix > 0, 1.0 / a_prefix, 0.0)

        dz_ptrs = DZBlock_ptr + pid_bh * stride_dz_bh + block_idx * stride_dz_blk + m_offsets[:, None] * stride_dz_m + d_offsets[None, :] * stride_dz_d
        dz_tile = tl.load(dz_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        suffix_db += dz_tile * inv_a[:, None]

        db_ptrs = DBBlock_ptr + pid_bh * stride_db_bh + block_idx * stride_db_blk + m_offsets[:, None] * stride_db_m + d_offsets[None, :] * stride_db_d
        tl.store(db_ptrs, suffix_db, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _semi_ar_bwd_enc_dv_kernel(
    Q_ptr,
    K_ptr,
    ABlock_ptr,
    DBBlock_ptr,
    DV_ptr,
    stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_ab_bh, stride_ab_blk, stride_ab_m,
    stride_db_bh, stride_db_blk, stride_db_m, stride_db_d,
    stride_dv_b, stride_dv_n, stride_dv_h, stride_dv_d,
    BH, M, N,
    scale,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_SIZE,
    CHUNK_SIZE: tl.constexpr,
    CHUNKS_PER_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    pid_bh = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    pid_d = tl.program_id(2)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    block_idx = pid_chunk // CHUNKS_PER_BLOCK
    chunk_start = pid_chunk * CHUNK_SIZE
    score_scale = scale * RCP_LN2

    m_offsets = tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE

    a_block = tl.load(ABlock_ptr + pid_bh * stride_ab_bh + block_idx * stride_ab_blk + m_offsets * stride_ab_m, mask=mask_m, other=0.0).to(tl.float32)
    a_safe = tl.maximum(a_block, 1e-20)
    lse_block_log2 = tl.math.log2(a_safe)
    q_desc = _semi_ar_make_2d_desc(Q_ptr + pid_h * stride_q_h, M, D_SCORE, stride_q_m, stride_q_d, BLOCK_M, BLOCK_K)
    db_base = DBBlock_ptr + pid_bh * stride_db_bh + block_idx * stride_db_blk

    for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T):
        t_offsets = chunk_start + t0 + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < N

        enc_scores = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
        for k0 in tl.range(0, D_SCORE, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < D_SCORE
            k_ptrs = K_ptr + pid_b * stride_k_b + t_offsets[:, None] * stride_k_n + pid_h * stride_k_h + k_offsets[None, :] * stride_k_d
            k_tile = tl.load(k_ptrs, mask=mask_t[:, None] & mask_k[None, :], other=0.0)
            q_tile = _semi_ar_load_2d_tile(q_desc, 0, k0).reshape([BLOCK_M, BLOCK_K])
            enc_scores += tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        p_block = tl.math.exp2(enc_scores * score_scale - lse_block_log2[None, :])
        raw = tl.where(mask_t[:, None] & mask_m[None, :], p_block * a_block[None, :], 0.0)
        db_ptrs = db_base + m_offsets[:, None] * stride_db_m + d_offsets[None, :] * stride_db_d
        db_tile = tl.load(db_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        dv_tile = tl.dot(raw.to(db_tile.dtype), db_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        dv_ptrs = DV_ptr + pid_b * stride_dv_b + t_offsets[:, None] * stride_dv_n + pid_h * stride_dv_h + d_offsets[None, :] * stride_dv_d
        tl.store(dv_ptrs, dv_tile, mask=mask_t[:, None] & mask_d[None, :])


@triton.jit
def _semi_ar_bwd_enc_dk_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    ABlock_ptr,
    DABlock_ptr,
    DBBlock_ptr,
    DK_ptr,
    stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_ab_bh, stride_ab_blk, stride_ab_m,
    stride_da_bh, stride_da_blk, stride_da_m,
    stride_db_bh, stride_db_blk, stride_db_m, stride_db_d,
    stride_dk_b, stride_dk_n, stride_dk_h, stride_dk_d,
    BH, M, N,
    scale,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_SIZE,
    CHUNK_SIZE: tl.constexpr,
    CHUNKS_PER_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    pid_bh = tl.program_id(0)
    pid_chunk = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    block_idx = pid_chunk // CHUNKS_PER_BLOCK
    chunk_start = pid_chunk * CHUNK_SIZE
    score_scale = scale * RCP_LN2

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    a_block = tl.load(ABlock_ptr + pid_bh * stride_ab_bh + block_idx * stride_ab_blk + m_offsets * stride_ab_m, mask=mask_m, other=0.0).to(tl.float32)
    a_safe = tl.maximum(a_block, 1e-20)
    lse_block_log2 = tl.math.log2(a_safe)
    dA_block = tl.load(DABlock_ptr + pid_bh * stride_da_bh + block_idx * stride_da_blk + m_offsets * stride_da_m, mask=mask_m, other=0.0).to(tl.float32)
    q_desc = _semi_ar_make_2d_desc(Q_ptr + pid_h * stride_q_h, M, D_SCORE, stride_q_m, stride_q_d, BLOCK_M, BLOCK_K)
    db_base = DBBlock_ptr + pid_bh * stride_db_bh + block_idx * stride_db_blk

    for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T):
        t_offsets = chunk_start + t0 + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < N

        enc_scores = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
        for k0 in tl.range(0, D_SCORE, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < D_SCORE
            k_ptrs = K_ptr + pid_b * stride_k_b + t_offsets[:, None] * stride_k_n + pid_h * stride_k_h + k_offsets[None, :] * stride_k_d
            k_tile = tl.load(k_ptrs, mask=mask_t[:, None] & mask_k[None, :], other=0.0)
            q_tile = _semi_ar_load_2d_tile(q_desc, 0, k0).reshape([BLOCK_M, BLOCK_K])
            enc_scores += tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        p_block = tl.math.exp2(enc_scores * score_scale - lse_block_log2[None, :])
        raw = tl.where(mask_t[:, None] & mask_m[None, :], p_block * a_block[None, :], 0.0)

        v_proj = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
        for d0 in tl.range(0, D_VALUE, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D_VALUE
            v_ptrs = V_ptr + pid_b * stride_v_b + t_offsets[:, None] * stride_v_n + pid_h * stride_v_h + d_offsets[None, :] * stride_v_d
            v_tile = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            db_ptrs = db_base + m_offsets[:, None] * stride_db_m + d_offsets[None, :] * stride_db_d
            db_tile = tl.load(db_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
            v_proj += tl.dot(v_tile, tl.trans(db_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        ds = raw * (dA_block[None, :] + v_proj)
        for k0 in tl.range(0, D_SCORE, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_offsets < D_SCORE
            q_tile = _semi_ar_load_2d_tile(q_desc, 0, k0).reshape([BLOCK_M, BLOCK_K])
            dk_tile = tl.dot(ds.to(q_tile.dtype), q_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale
            dk_ptrs = DK_ptr + pid_b * stride_dk_b + t_offsets[:, None] * stride_dk_n + pid_h * stride_dk_h + k_offsets[None, :] * stride_dk_d
            tl.store(dk_ptrs, dk_tile, mask=mask_t[:, None] & mask_k[None, :])


@triton.jit
def _semi_ar_bwd_enc_dq_from_summary_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    ABlock_ptr,
    DABlock_ptr,
    DBBlock_ptr,
    DQ_ptr,
    stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_ab_bh, stride_ab_blk, stride_ab_m,
    stride_da_bh, stride_da_blk, stride_da_m,
    stride_db_bh, stride_db_blk, stride_db_m, stride_db_d,
    stride_dq_h, stride_dq_m, stride_dq_d,
    BH, M, N,
    scale,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_SIZE,
    CHUNK_SIZE: tl.constexpr,
    CHUNKS_PER_BLOCK,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    pid_bh = tl.program_id(0)
    block_idx = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    score_scale = scale * RCP_LN2
    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    a_block = tl.load(ABlock_ptr + pid_bh * stride_ab_bh + block_idx * stride_ab_blk + m_offsets * stride_ab_m, mask=mask_m, other=0.0).to(tl.float32)
    a_safe = tl.maximum(a_block, 1e-20)
    lse_block_log2 = tl.math.log2(a_safe)
    dA_block = tl.load(DABlock_ptr + pid_bh * stride_da_bh + block_idx * stride_da_blk + m_offsets * stride_da_m, mask=mask_m, other=0.0).to(tl.float32)
    q_desc = _semi_ar_make_2d_desc(Q_ptr + pid_h * stride_q_h, M, D_SCORE, stride_q_m, stride_q_d, BLOCK_M, BLOCK_K)
    db_base = DBBlock_ptr + pid_bh * stride_db_bh + block_idx * stride_db_blk
    block_start = block_idx * BLOCK_SIZE

    for k0 in tl.range(0, D_SCORE, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D_SCORE
        dq_acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

        for local_chunk in tl.range(0, CHUNKS_PER_BLOCK):
            chunk_start = block_start + local_chunk * CHUNK_SIZE
            for t0 in tl.range(0, CHUNK_SIZE, BLOCK_T):
                t_offsets = chunk_start + t0 + tl.arange(0, BLOCK_T)
                mask_t = t_offsets < N

                enc_scores = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
                for ki in tl.range(0, D_SCORE, BLOCK_K):
                    ki_offsets = ki + tl.arange(0, BLOCK_K)
                    mask_ki = ki_offsets < D_SCORE
                    k_ptrs = K_ptr + pid_b * stride_k_b + t_offsets[:, None] * stride_k_n + pid_h * stride_k_h + ki_offsets[None, :] * stride_k_d
                    k_tile = tl.load(k_ptrs, mask=mask_t[:, None] & mask_ki[None, :], other=0.0)
                    q_tile = _semi_ar_load_2d_tile(q_desc, 0, ki).reshape([BLOCK_M, BLOCK_K])
                    enc_scores += tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

                p_block = tl.math.exp2(enc_scores * score_scale - lse_block_log2[None, :])
                raw = tl.where(mask_t[:, None] & mask_m[None, :], p_block * a_block[None, :], 0.0)

                v_proj = tl.zeros((BLOCK_T, BLOCK_M), dtype=tl.float32)
                for d0 in tl.range(0, D_VALUE, BLOCK_D):
                    d_offsets = d0 + tl.arange(0, BLOCK_D)
                    mask_d = d_offsets < D_VALUE
                    v_ptrs = V_ptr + pid_b * stride_v_b + t_offsets[:, None] * stride_v_n + pid_h * stride_v_h + d_offsets[None, :] * stride_v_d
                    v_tile = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                    db_ptrs = db_base + m_offsets[:, None] * stride_db_m + d_offsets[None, :] * stride_db_d
                    db_tile = tl.load(db_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
                    v_proj += tl.dot(v_tile, tl.trans(db_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

                ds = raw * (dA_block[None, :] + v_proj)
                k_ptrs = K_ptr + pid_b * stride_k_b + t_offsets[:, None] * stride_k_n + pid_h * stride_k_h + k_offsets[None, :] * stride_k_d
                k_tile = tl.load(k_ptrs, mask=mask_t[:, None] & mask_k[None, :], other=0.0)
                dq_acc += tl.dot(tl.trans(ds.to(k_tile.dtype)), k_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        dq_ptrs = DQ_ptr + pid_h * stride_dq_h + m_offsets[:, None] * stride_dq_m + k_offsets[None, :] * stride_dq_d
        tl.atomic_add(dq_ptrs, dq_acc * scale, mask=mask_m[:, None] & mask_k[None, :])


# ---------------------------------------------------------------------------
# Backward launcher helpers
# ---------------------------------------------------------------------------


def _run_semi_ar_bwd_summary_scan(
    lse_enc: torch.Tensor,
    dz_block: torch.Tensor,
    z_block: torch.Tensor,
    *,
    M: int,
    D_value: int,
    config: dict,
    kernel_timings=None,
):
    BH = lse_enc.size(0)
    device = lse_enc.device

    def alloc_summary():
        return (
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M), device=device, dtype=torch.float32),
            torch.empty((BH, config["NUM_BLOCKS"], M, D_value), device=device, dtype=torch.float32),
        )

    a_block, dA_block, dB_block = _profiled_call(device, kernel_timings, "alloc_bwd_summary_scan", alloc_summary)

    def scalar_grid(_meta):
        return (BH, 1)

    def vector_grid(_meta):
        return (BH, config["NUM_D_VALUE_BLOCKS"])

    def launch_scalar():
        _semi_ar_bwd_summary_scalar_scan_kernel[scalar_grid](
            lse_enc,
            z_block,
            dz_block,
            a_block,
            dA_block,
            *lse_enc.stride(),
            *z_block.stride(),
            *dz_block.stride(),
            *a_block.stride(),
            *dA_block.stride(),
            BH,
            M,
            D_VALUE=D_value,
            NUM_BLOCKS=config["NUM_BLOCKS"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            num_warps=config["enc_dq_num_warps"],
            num_stages=config["enc_dq_num_stages"],
        )

    def launch_vector():
        _semi_ar_bwd_summary_vector_scan_kernel[vector_grid](
            lse_enc,
            dz_block,
            dB_block,
            *lse_enc.stride(),
            *dz_block.stride(),
            *dB_block.stride(),
            BH,
            M,
            D_VALUE=D_value,
            NUM_BLOCKS=config["NUM_BLOCKS"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            num_warps=config["enc_dkdv_num_warps"],
            num_stages=config["enc_dkdv_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_bwd_summary_scalar_scan", launch_scalar)
    _profiled_call(device, kernel_timings, "semi_ar_bwd_summary_vector_scan", launch_vector)
    return a_block, dA_block, dB_block


def _run_semi_ar_bwd_enc_dv_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    a_block: torch.Tensor,
    dB_block: torch.Tensor,
    dV: torch.Tensor,
    *,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict,
    kernel_timings=None,
):
    B = K.size(0)
    BH = B * H
    device = K.device

    def grid(_meta):
        return (BH, config["NUM_GLOBAL_CHUNKS"], config["NUM_D_VALUE_BLOCKS"])

    def launch():
        _semi_ar_bwd_enc_dv_kernel[grid](
            Q,
            K,
            a_block,
            dB_block,
            dV,
            *Q.stride(),
            *K.stride(),
            *a_block.stride(),
            *dB_block.stride(),
            *dV.stride(),
            BH, M, N,
            scale,
            D_score, D_value,
            config["BLOCK_SIZE"],
            CHUNK_SIZE=config["CHUNK_SIZE"],
            CHUNKS_PER_BLOCK=config["CHUNKS_PER_BLOCK"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            BLOCK_T=config["BLOCK_T"],
            INPUT_PRECISION="ieee",
            H=H,
            num_warps=config["enc_dkdv_num_warps"],
            num_stages=config["enc_dkdv_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_bwd_enc_dv", launch)


def _run_semi_ar_bwd_enc_dk_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    a_block: torch.Tensor,
    dA_block: torch.Tensor,
    dB_block: torch.Tensor,
    dK: torch.Tensor,
    *,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict,
    kernel_timings=None,
):
    B = K.size(0)
    BH = B * H
    device = K.device

    def grid(_meta):
        return (BH, config["NUM_GLOBAL_CHUNKS"])

    def launch():
        _semi_ar_bwd_enc_dk_kernel[grid](
            Q,
            K,
            V,
            a_block,
            dA_block,
            dB_block,
            dK,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *a_block.stride(),
            *dA_block.stride(),
            *dB_block.stride(),
            *dK.stride(),
            BH, M, N,
            scale,
            D_score, D_value,
            config["BLOCK_SIZE"],
            CHUNK_SIZE=config["CHUNK_SIZE"],
            CHUNKS_PER_BLOCK=config["CHUNKS_PER_BLOCK"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            BLOCK_T=config["BLOCK_T"],
            INPUT_PRECISION="ieee",
            H=H,
            num_warps=config["enc_dkdv_num_warps"],
            num_stages=config["enc_dkdv_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_bwd_enc_dk", launch)


def _run_semi_ar_bwd_enc_dq_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    a_block: torch.Tensor,
    dA_block: torch.Tensor,
    dB_block: torch.Tensor,
    dQ: torch.Tensor,
    *,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict,
    kernel_timings=None,
):
    B = K.size(0)
    BH = B * H
    device = K.device

    def grid(_meta):
        return (BH, config["NUM_BLOCKS"])

    def launch():
        _semi_ar_bwd_enc_dq_from_summary_kernel[grid](
            Q,
            K,
            V,
            a_block,
            dA_block,
            dB_block,
            dQ,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *a_block.stride(),
            *dA_block.stride(),
            *dB_block.stride(),
            *dQ.stride(),
            BH, M, N,
            scale,
            D_score, D_value,
            config["BLOCK_SIZE"],
            CHUNK_SIZE=config["CHUNK_SIZE"],
            CHUNKS_PER_BLOCK=config["CHUNKS_PER_BLOCK"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            BLOCK_T=config["BLOCK_T"],
            INPUT_PRECISION="ieee",
            H=H,
            num_warps=config["enc_dq_num_warps"],
            num_stages=config["enc_dq_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_bwd_enc_dq_triton", launch)


def _run_semi_ar_bwd_preprocess(
    Y: torch.Tensor,
    dO: torch.Tensor,
    *,
    H: int,
    N: int,
    D_value: int,
    config: dict,
    kernel_timings=None,
):
    B = Y.size(0)
    BH = B * H
    device = Y.device

    delta = torch.empty((BH, N), device=device, dtype=torch.float32)
    block_t = min(int(config["BLOCK_T"]), 128)
    block_d = int(config["BLOCK_D"])

    def grid(_meta):
        return (BH, triton.cdiv(N, block_t))

    def launch():
        _semi_ar_bwd_preprocess_kernel[grid](
            Y, dO, delta,
            *Y.stride(), *dO.stride(), *delta.stride(),
            BH, N, D_value,
            BLOCK_T=block_t,
            BLOCK_D=block_d,
            H=H,
            num_warps=config["preprocess_num_warps"],
            num_stages=config["preprocess_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_bwd_preprocess", launch)
    return delta


def _run_semi_ar_bwd_accum_dz_enc(
    Q: torch.Tensor,
    K: torch.Tensor,
    dO: torch.Tensor,
    lse_dec: torch.Tensor,
    *,
    Q_dec: torch.Tensor,
    K_dec: torch.Tensor,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict,
    weight_sharing: bool,
    num_storage_dtype: torch.dtype,
    kernel_timings=None,
):
    B = K.size(0)
    BH = B * H
    device = K.device

    dz_enc = torch.zeros(
        (BH, config["NUM_BLOCKS"], M, D_value), device=device, dtype=num_storage_dtype,
    )

    qd_tensor = K if weight_sharing else Q_dec
    kd_tensor = Q if weight_sharing else K_dec

    def grid(_meta):
        return (BH, config["NUM_BLOCKS"], config["NUM_D_VALUE_BLOCKS"])

    def launch():
        _semi_ar_bwd_accum_dz_enc_kernel[grid](
            qd_tensor,
            kd_tensor,
            Q,
            dO,
            lse_dec,
            dz_enc,
            *qd_tensor.stride(),
            *(kd_tensor.stride() if not weight_sharing else Q.stride()),
            *Q.stride(),
            *dO.stride(),
            *lse_dec.stride(),
            *dz_enc.stride(),
            BH, M, N,
            scale,
            D_score, D_value,
            config["BLOCK_SIZE"],
            CHUNK_SIZE=config["CHUNK_SIZE"],
            CHUNKS_PER_BLOCK=config["CHUNKS_PER_BLOCK"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            BLOCK_T=config["BLOCK_T"],
            INPUT_PRECISION="ieee",
            H=H,
            WEIGHT_SHARING=weight_sharing,
            num_warps=config["enc_dkdv_num_warps"],
            num_stages=config["enc_dkdv_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_bwd_accum_dz_enc", launch)
    return dz_enc


def _run_semi_ar_bwd_enc_dkdv(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    lse_enc: torch.Tensor,
    dz_enc: torch.Tensor,
    dK: torch.Tensor,
    dV: torch.Tensor,
    *,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict,
    kernel_timings=None,
):
    B = K.size(0)
    BH = B * H
    device = K.device

    def grid(_meta):
        return (BH, config["NUM_GLOBAL_CHUNKS"])

    def launch():
        _semi_ar_bwd_enc_dkdv_from_dz_kernel[grid](
            Q, K, V,
            lse_enc, dz_enc,
            dK, dV,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *lse_enc.stride(),
            *dz_enc.stride(),
            *dK.stride(),
            *dV.stride(),
            BH, M, N,
            scale,
            D_score, D_value,
            config["BLOCK_SIZE"],
            CHUNK_SIZE=config["CHUNK_SIZE"],
            CHUNKS_PER_BLOCK=config["CHUNKS_PER_BLOCK"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            BLOCK_T=config["BLOCK_T"],
            INPUT_PRECISION="ieee",
            H=H,
            num_warps=config["enc_dkdv_num_warps"],
            num_stages=config["enc_dkdv_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_bwd_enc_dkdv", launch)


def _run_semi_ar_bwd_enc_dq(
    K: torch.Tensor,
    V: torch.Tensor,
    Q: torch.Tensor,
    lse_enc: torch.Tensor,
    dz_enc: torch.Tensor,
    dQ: torch.Tensor,
    *,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict,
    kernel_timings=None,
):
    B = K.size(0)
    BH = B * H
    device = K.device

    def grid(_meta):
        return (BH, config["NUM_BLOCKS"])

    def launch():
        _semi_ar_bwd_enc_dq_kernel[grid](
            K, V, Q,
            lse_enc, dz_enc,
            dQ,
            *K.stride(),
            *V.stride(),
            *Q.stride(),
            *lse_enc.stride(),
            *dz_enc.stride(),
            *dQ.stride(),
            BH, M, N,
            scale,
            D_score, D_value,
            config["BLOCK_SIZE"],
            CHUNK_SIZE=config["CHUNK_SIZE"],
            CHUNKS_PER_BLOCK=config["CHUNKS_PER_BLOCK"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            BLOCK_T=config["BLOCK_T"],
            INPUT_PRECISION="ieee",
            H=H,
            num_warps=config["enc_dq_num_warps"],
            num_stages=config["enc_dq_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_bwd_enc_dq", launch)


def _run_semi_ar_bwd_dec(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Y: torch.Tensor,
    dO: torch.Tensor,
    lse_dec: torch.Tensor,
    delta: torch.Tensor,
    z_block: torch.Tensor,
    dK: torch.Tensor,
    dQ: torch.Tensor,
    *,
    Q_dec: torch.Tensor,
    K_dec: torch.Tensor,
    dQ_dec: torch.Tensor | None,
    dK_dec: torch.Tensor | None,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict,
    weight_sharing: bool,
    separate_Q_dec: bool,
    separate_K_dec: bool,
    kernel_timings=None,
):
    B = K.size(0)
    BH = B * H
    device = K.device

    qd_tensor = K if weight_sharing else Q_dec
    kd_tensor = Q if weight_sharing else K_dec
    dqd_tensor = dQ_dec if (separate_Q_dec and dQ_dec is not None) else dK  # dummy
    dkd_tensor = dK_dec if (separate_K_dec and dK_dec is not None) else dQ  # dummy

    def grid(_meta):
        return (BH, config["NUM_GLOBAL_CHUNKS"])

    def launch():
        _semi_ar_bwd_dec_kernel[grid](
            qd_tensor,
            kd_tensor,
            Q, K, V, Y, dO,
            lse_dec, delta, z_block,
            dqd_tensor, dkd_tensor,
            dK, dQ,
            *qd_tensor.stride(),
            *(kd_tensor.stride() if not weight_sharing else Q.stride()),
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *Y.stride(),
            *dO.stride(),
            *lse_dec.stride(),
            *delta.stride(),
            *z_block.stride(),
            *(dqd_tensor.stride() if dQ_dec is not None else dK.stride()),
            *(dkd_tensor.stride() if dK_dec is not None else dQ.stride()),
            *dK.stride(),
            *dQ.stride(),
            BH, M, N,
            scale,
            D_score, D_value,
            config["BLOCK_SIZE"],
            CHUNK_SIZE=config["CHUNK_SIZE"],
            CHUNKS_PER_BLOCK=config["CHUNKS_PER_BLOCK"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            BLOCK_T=config["BLOCK_T"],
            INPUT_PRECISION="ieee",
            H=H,
            WEIGHT_SHARING=weight_sharing,
            SEPARATE_Q_DEC=separate_Q_dec,
            SEPARATE_K_DEC=separate_K_dec,
            num_warps=config["dec_dqdk_num_warps"],
            num_stages=config["dec_dqdk_num_stages"],
        )

    _profiled_call(device, kernel_timings, "semi_ar_bwd_dec", launch)


def _semi_ar_reverse_block_cumsum(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(torch.cumsum(torch.flip(x, dims=(1,)), dim=1), dims=(1,))


def _semi_ar_backward_block_batch_size(
    *,
    B: int,
    H: int,
    num_blocks: int,
    block_size: int,
    M: int,
) -> int:
    env = os.environ.get("FLARE_SEMI_AR_BWD_BLOCK_BATCH", "").strip()
    if env:
        return max(1, min(num_blocks, int(env)))

    # Keep the largest [B, H, G_batch, BLOCK_SIZE, M] score tensor around
    # ~128 MiB in FP32 by default.
    target_score_elems = 32 * 1024 * 1024
    elems_per_block = max(1, B * H * block_size * M)
    return max(1, min(num_blocks, target_score_elems // elems_per_block))


def _compute_semi_ar_summary_backward_tensors(
    dz_block: torch.Tensor,
    lse_enc: torch.Tensor,
    z_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert prefix-summary gradients into per-source-block summary gradients.

    Forward stores inclusive prefix summaries:
      A_g = exp(LSE_enc[g])
      B_g = A_g * z_block[g]
      z_block[g] = B_g / A_g

    Decoder backward first produces dZ_g = dL / d z_block[g].
    We then map this to gradients on the inclusive prefix totals (A_g, B_g),
    and finally suffix-scan them back to gradients on the local block
    contributions (A_block[s], B_block[s]).
    """

    a_prefix = torch.exp(lse_enc.float())
    z_block_f = z_block.float()
    dz_block_f = dz_block.float()
    inv_a = torch.where(a_prefix > 0, a_prefix.reciprocal(), torch.zeros_like(a_prefix))

    dB_prefix = dz_block_f * inv_a.unsqueeze(-1)
    dA_prefix = -(dz_block_f * z_block_f).sum(dim=-1) * inv_a

    dB_block = _semi_ar_reverse_block_cumsum(dB_prefix)
    dA_block = _semi_ar_reverse_block_cumsum(dA_prefix)

    a_prev = torch.zeros_like(a_prefix)
    a_prev[:, 1:, :] = a_prefix[:, :-1, :]
    a_block = (a_prefix - a_prev).clamp_min(0.0)
    return a_block, dA_block, dB_block


def _semi_ar_decoder_backward_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    Y: torch.Tensor,
    dY: torch.Tensor,
    lse_dec: torch.Tensor,
    z_block: torch.Tensor,
    *,
    scale: float,
    block_size: int,
    chunk_size: int,
    Q_dec: torch.Tensor | None,
    K_dec: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Decoder replay: accumulate dZ_block and decoder/shared parameter gradients."""

    B, N, H, _ = K.shape
    num_blocks = N // block_size
    M = Q.size(1)
    D_value = Y.size(-1)

    q_bank = Q.float()
    k_tokens = K.float().permute(0, 2, 1, 3).contiguous()
    y_tokens = Y.float().permute(0, 2, 1, 3).contiguous()
    dy_tokens = dY.float().permute(0, 2, 1, 3).contiguous()
    q_dec_tokens = k_tokens if Q_dec is None else Q_dec.float().permute(0, 2, 1, 3).contiguous()
    k_dec_bank = q_bank if K_dec is None else K_dec.float()
    lse_dec_view = lse_dec.float().reshape(B, H, N)
    z_block_view = z_block.float().reshape(B, H, num_blocks, M, D_value)
    delta = (y_tokens * dy_tokens).sum(dim=-1)

    q_dec_blocks = q_dec_tokens.view(B, H, num_blocks, block_size, q_dec_tokens.size(-1))
    dy_blocks = dy_tokens.view(B, H, num_blocks, block_size, D_value)
    lse_dec_blocks = lse_dec_view.view(B, H, num_blocks, block_size)
    delta_blocks = delta.view(B, H, num_blocks, block_size)

    dz_block = torch.empty((B, H, num_blocks, M, D_value), device=Q.device, dtype=torch.float32)
    dQ_shared = torch.zeros_like(q_bank)
    dK_shared = torch.zeros_like(k_tokens)
    dK_shared_blocks = dK_shared.view(B, H, num_blocks, block_size, k_tokens.size(-1))
    dQ_dec = torch.zeros_like(q_dec_tokens) if Q_dec is not None else None
    dQ_dec_blocks = dQ_dec.view(B, H, num_blocks, block_size, q_dec_tokens.size(-1)) if dQ_dec is not None else None
    dK_dec = torch.zeros_like(k_dec_bank) if K_dec is not None else None

    del chunk_size  # backward replay is block-batched; chunking only matters for forward scheduling
    block_batch = _semi_ar_backward_block_batch_size(B=B, H=H, num_blocks=num_blocks, block_size=block_size, M=M)

    for block_start in range(0, num_blocks, block_batch):
        block_end = min(block_start + block_batch, num_blocks)
        q_batch = q_dec_blocks[:, :, block_start:block_end, :, :]
        dy_batch = dy_blocks[:, :, block_start:block_end, :, :]
        z_batch = z_block_view[:, :, block_start:block_end, :, :]
        lse_batch = lse_dec_blocks[:, :, block_start:block_end, :]
        delta_batch = delta_blocks[:, :, block_start:block_end, :]

        scores = scale * torch.einsum("bhgtd,hmd->bhgtm", q_batch, k_dec_bank)
        alpha = torch.exp(scores - lse_batch.unsqueeze(-1))
        dz_block[:, :, block_start:block_end, :, :] = torch.einsum("bhgtm,bhgtd->bhgmd", alpha, dy_batch)

        dalpha = torch.einsum("bhgtd,bhgmd->bhgtm", dy_batch, z_batch)
        ds = alpha * (dalpha - delta_batch.unsqueeze(-1))

        grad_query = scale * torch.einsum("bhgtm,hmd->bhgtd", ds, k_dec_bank)
        grad_key = scale * torch.einsum("bhgtm,bhgtd->hmd", ds, q_batch)

        if dQ_dec_blocks is None:
            dK_shared_blocks[:, :, block_start:block_end, :, :] += grad_query
        else:
            dQ_dec_blocks[:, :, block_start:block_end, :, :] += grad_query

        if dK_dec is None:
            dQ_shared += grad_key
        else:
            dK_dec += grad_key

    dK_shared = dK_shared.permute(0, 2, 1, 3).contiguous()
    dQ_dec_out = dQ_dec.permute(0, 2, 1, 3).contiguous() if dQ_dec is not None else None
    dz_block = dz_block.reshape(B * H, num_blocks, M, D_value)
    return dz_block, dQ_shared, dK_shared, dQ_dec_out, dK_dec


def _semi_ar_encoder_backward_torch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    a_block: torch.Tensor,
    dA_block: torch.Tensor,
    dB_block: torch.Tensor,
    *,
    scale: float,
    block_size: int,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replay local encoder chunks against block-summary gradients."""

    B, N, H, _ = K.shape
    num_blocks = N // block_size
    M = Q.size(1)

    q_bank = Q.float()
    k_tokens = K.float().permute(0, 2, 1, 3).contiguous()
    v_tokens = V.float().permute(0, 2, 1, 3).contiguous()

    a_block_view = a_block.reshape(B, H, num_blocks, M)
    dA_block_view = dA_block.reshape(B, H, num_blocks, M)
    dB_block_view = dB_block.reshape(B, H, num_blocks, M, V.size(-1))
    block_lse = torch.log(a_block_view.clamp_min(1e-20))
    k_blocks = k_tokens.view(B, H, num_blocks, block_size, k_tokens.size(-1))
    v_blocks = v_tokens.view(B, H, num_blocks, block_size, v_tokens.size(-1))

    dQ = torch.zeros_like(q_bank)
    dK = torch.zeros_like(k_tokens)
    dV = torch.zeros_like(v_tokens)
    dK_blocks = dK.view(B, H, num_blocks, block_size, k_tokens.size(-1))
    dV_blocks = dV.view(B, H, num_blocks, block_size, v_tokens.size(-1))

    del chunk_size
    block_batch = _semi_ar_backward_block_batch_size(B=B, H=H, num_blocks=num_blocks, block_size=block_size, M=M)

    for block_start in range(0, num_blocks, block_batch):
        block_end = min(block_start + block_batch, num_blocks)
        k_batch = k_blocks[:, :, block_start:block_end, :, :]
        v_batch = v_blocks[:, :, block_start:block_end, :, :]
        a_batch = a_block_view[:, :, block_start:block_end, :]
        dA_batch = dA_block_view[:, :, block_start:block_end, :]
        dB_batch = dB_block_view[:, :, block_start:block_end, :, :]
        lse_batch = block_lse[:, :, block_start:block_end, :]

        scores = scale * torch.einsum("bhgtd,hmd->bhgtm", k_batch, q_bank)
        p_block = torch.exp(scores - lse_batch.unsqueeze(-2))
        raw_weights = p_block * a_batch.unsqueeze(-2)

        dV_blocks[:, :, block_start:block_end, :, :] = torch.einsum("bhgtm,bhgmd->bhgtd", raw_weights, dB_batch)

        v_proj = torch.einsum("bhgtd,bhgmd->bhgtm", v_batch, dB_batch)
        ds = raw_weights * (dA_batch.unsqueeze(-2) + v_proj)

        dK_blocks[:, :, block_start:block_end, :, :] = scale * torch.einsum("bhgtm,hmd->bhgtd", ds, q_bank)
        dQ += scale * torch.einsum("bhgtm,bhgtd->hmd", ds, k_batch)

    dK = dK.permute(0, 2, 1, 3).contiguous()
    dV = dV.permute(0, 2, 1, 3).contiguous()
    return dQ, dK, dV


def _semi_autoregressive_backward_impl(ctx, dY, *unused):
    del unused
    if dY is None:
        return None, None, None, None, None, None, None, None, None, None

    separate_Q_dec = getattr(ctx, "separate_Q_dec", False)
    separate_K_dec = getattr(ctx, "separate_K_dec", False)
    weight_sharing = getattr(ctx, "weight_sharing_enc_dec", True)
    saved = ctx.saved_tensors
    # 9 base tensors (Q, K, V, Y, prefix_max, prefix_den, lse_enc, lse_dec, z_block)
    # + optional Q_dec, K_dec
    expected_len = 9 + int(separate_Q_dec) + int(separate_K_dec)
    if len(saved) != expected_len:
        raise RuntimeError(
            f"SemiAutoRegressiveFLARE backward expected {expected_len} saved tensors "
            f"(9 base + separate_Q_dec={separate_Q_dec} + separate_K_dec={separate_K_dec}), got {len(saved)}."
        )

    Q, K, V, Y, _prefix_max, _prefix_den, lse_enc, lse_dec, z_block = saved[:9]
    idx = 9
    if separate_Q_dec:
        Q_dec_saved = saved[idx]
        idx += 1
    else:
        Q_dec_saved = None
    if separate_K_dec:
        K_dec_saved = saved[idx]
    else:
        K_dec_saved = None

    profile_data = getattr(ctx, "profile_timings", None)
    bwd_timings = profile_data["backward"] if isinstance(profile_data, dict) else None
    bwd_resources = profile_data.setdefault("backward_resources", {}) if isinstance(profile_data, dict) else None
    device = Q.device

    def profiled_bwd_call(key: str, op):
        return _profiled_bwd_call(device, bwd_timings, key, op, resource_bucket=bwd_resources)

    dY_contig = profiled_bwd_call("grad_input_cast", lambda: dY.contiguous())
    B, N, H, D_value = V.shape
    M, D_score = Q.shape[1], Q.shape[2]
    block_size = ctx.block_size
    chunk_size = ctx.chunk_size
    scale = ctx.scale
    bwd_cfg = _get_semi_ar_backward_config(
        M=M,
        N=N,
        D_score=D_score,
        D_value=D_value,
        block_size=block_size,
        chunk_size=chunk_size,
    )

    delta = _run_semi_ar_bwd_preprocess(
        Y,
        dY_contig,
        H=H,
        N=N,
        D_value=D_value,
        config=bwd_cfg,
        kernel_timings=bwd_timings,
    )

    Q_dec_tensor = K if weight_sharing else Q_dec_saved
    K_dec_tensor = Q if weight_sharing else K_dec_saved

    dz_block = _run_semi_ar_bwd_accum_dz_enc(
        Q,
        K,
        dY_contig,
        lse_dec,
        Q_dec=Q_dec_tensor,
        K_dec=K_dec_tensor,
        H=H,
        M=M,
        N=N,
        D_score=D_score,
        D_value=D_value,
        scale=scale,
        config=bwd_cfg,
        weight_sharing=weight_sharing,
        num_storage_dtype=torch.float32,
        kernel_timings=bwd_timings,
    )

    dQ_dec_shared = profiled_bwd_call(
        "alloc_dQ_dec_shared",
        lambda: torch.zeros(Q.shape, device=Q.device, dtype=torch.float32),
    )
    dK_dec_shared = profiled_bwd_call(
        "alloc_dK_dec_shared",
        lambda: torch.zeros(K.shape, device=K.device, dtype=torch.float32),
    )
    dQ_dec = (
        profiled_bwd_call(
            "alloc_dQ_dec",
            lambda: torch.zeros(Q_dec_saved.shape, device=Q_dec_saved.device, dtype=torch.float32),
        )
        if separate_Q_dec
        else None
    )
    dK_dec = (
        profiled_bwd_call(
            "alloc_dK_dec",
            lambda: torch.zeros(K_dec_saved.shape, device=K_dec_saved.device, dtype=torch.float32),
        )
        if separate_K_dec
        else None
    )

    _run_semi_ar_bwd_dec(
        Q,
        K,
        V,
        Y,
        dY_contig,
        lse_dec,
        delta,
        z_block,
        dK_dec_shared,
        dQ_dec_shared,
        Q_dec=Q_dec_tensor,
        K_dec=K_dec_tensor,
        dQ_dec=dQ_dec,
        dK_dec=dK_dec,
        H=H,
        M=M,
        N=N,
        D_score=D_score,
        D_value=D_value,
        scale=scale,
        config=bwd_cfg,
        weight_sharing=weight_sharing,
        separate_Q_dec=separate_Q_dec,
        separate_K_dec=separate_K_dec,
        kernel_timings=bwd_timings,
    )

    a_block, dA_block, dB_block = _run_semi_ar_bwd_summary_scan(
        lse_enc,
        dz_block,
        z_block,
        M=M,
        D_value=D_value,
        config=bwd_cfg,
        kernel_timings=bwd_timings,
    )

    dQ_enc = profiled_bwd_call(
        "alloc_dQ_enc",
        lambda: torch.zeros(Q.shape, device=Q.device, dtype=torch.float32),
    )
    dK_enc = profiled_bwd_call(
        "alloc_dK_enc",
        lambda: torch.empty(K.shape, device=K.device, dtype=torch.float32),
    )
    dV = profiled_bwd_call(
        "alloc_dV_enc",
        lambda: torch.empty(V.shape, device=V.device, dtype=torch.float32),
    )

    _run_semi_ar_bwd_enc_dv_triton(
        Q,
        K,
        a_block,
        dB_block,
        dV,
        H=H,
        M=M,
        N=N,
        D_score=D_score,
        D_value=D_value,
        scale=scale,
        config=bwd_cfg,
        kernel_timings=bwd_timings,
    )
    _run_semi_ar_bwd_enc_dk_triton(
        Q,
        K,
        V,
        a_block,
        dA_block,
        dB_block,
        dK_enc,
        H=H,
        M=M,
        N=N,
        D_score=D_score,
        D_value=D_value,
        scale=scale,
        config=bwd_cfg,
        kernel_timings=bwd_timings,
    )
    _run_semi_ar_bwd_enc_dq_triton(
        Q,
        K,
        V,
        a_block,
        dA_block,
        dB_block,
        dQ_enc,
        H=H,
        M=M,
        N=N,
        D_score=D_score,
        D_value=D_value,
        scale=scale,
        config=bwd_cfg,
        kernel_timings=bwd_timings,
    )

    dQ = (dQ_enc + dQ_dec_shared).to(Q.dtype)
    dK = (dK_enc + dK_dec_shared).to(K.dtype)
    dV = dV.to(V.dtype)
    dQ_dec = dQ_dec.to(Q_dec_saved.dtype) if dQ_dec is not None else None
    dK_dec = dK_dec.to(K_dec_saved.dtype) if dK_dec is not None else None

    if profile_data is not None:
        _refresh_profile_totals(profile_data)
    return dQ, dK, dV, None, None, None, None, None, dQ_dec, dK_dec


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
        resolved_scale = _resolve_attn_scale(scale, Q.size(-1))
        _, _, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_semi_ar_decode_inputs(Q, K, Q_dec, K_dec)
        if profile:
            Y, aux, profile_data, backward_state = _semi_autoregressive_forward_triton(
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
                return_backward_state=True,
            )
            ctx.save_for_backward(
                Q,
                K,
                V,
                Y,
                backward_state["prefix_max"],
                backward_state["prefix_den"],
                backward_state["lse_enc"],
                backward_state["lse_dec"],
                backward_state["z_block"],
                *((Q_dec,) if separate_Q_dec else ()),
                *((K_dec,) if separate_K_dec else ()),
            )
            ctx.scale = resolved_scale
            ctx.block_size = int(block_size)
            ctx.chunk_size = int(chunk_size)
            ctx.weight_sharing_enc_dec = weight_sharing_enc_dec
            ctx.separate_Q_dec = separate_Q_dec
            ctx.separate_K_dec = separate_K_dec
            ctx.profile_timings = profile_data
            return Y, aux, profile_data
        Y, aux, backward_state = _semi_autoregressive_forward_triton(
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
            return_backward_state=True,
        )
        ctx.save_for_backward(
            Q,
            K,
            V,
            Y,
            backward_state["prefix_max"],
            backward_state["prefix_den"],
            backward_state["lse_enc"],
            backward_state["lse_dec"],
            backward_state["z_block"],
            *((Q_dec,) if separate_Q_dec else ()),
            *((K_dec,) if separate_K_dec else ()),
        )
        ctx.scale = resolved_scale
        ctx.block_size = int(block_size)
        ctx.chunk_size = int(chunk_size)
        ctx.weight_sharing_enc_dec = weight_sharing_enc_dec
        ctx.separate_Q_dec = separate_Q_dec
        ctx.separate_K_dec = separate_K_dec
        ctx.profile_timings = None
        return Y, aux

    @staticmethod
    def backward(ctx, dY, *unused):
        return _semi_autoregressive_backward_impl(ctx, dY, *unused)


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

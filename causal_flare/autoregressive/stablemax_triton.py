from causal_flare._common import *
from causal_flare._reference_utils import (
    resolve_flare_causal_decode_inputs as _resolve_flare_causal_decode_inputs,
    validate_flare_qkv_layouts as _validate_flare_qkv_layouts,
)
from causal_flare.autoregressive.stablemax import _resolve_stablemax_chunk_size


def _stablemax_block_k_divisor(D: int, preferred: int) -> int:
    block_k = math.gcd(D, preferred)
    if block_k <= 0 or D % block_k != 0:
        raise ValueError(f"Unable to pick a valid BLOCK_K for D={D}, preferred={preferred}.")
    return block_k


def _stablemax_pick_allowed_cover(size: int) -> int:
    for candidate in (32, 64, 128, 256):
        if size <= candidate:
            return candidate
    return 256


def _stablemax_pick_allowed_block_k(D: int) -> int:
    if D in (32, 64, 128, 256):
        return D
    for candidate in (256, 128, 64, 32):
        if D % candidate == 0:
            return candidate
    raise ValueError(
        "FLAREAutoregressiveStablemaxTriton requires D_score to be divisible by one of "
        "{32, 64, 128, 256}. "
        f"Got D_score={D}."
    )


def _stablemax_power_mode(power: float) -> int:
    if power == 1.0:
        return 1
    if power == 2.0:
        return 2
    return 0


def _stablemax_forward_config(*, M: int, D_score: int, D_value: int, chunk_size: int, input_precision=None) -> dict[str, object]:
    # Kernel-shape policy
    # -------------------
    # This Triton path keeps the full chunk token axis visible inside the fused
    # output kernel. That is the main structural difference from the reference
    # PyTorch implementation, where eager ops can materialize large temporary
    # tensors without the same per-program register pressure constraints.
    #
    # Because one output program simultaneously touches:
    # - the full chunk axis     [CHUNK_SIZE]
    # - a latent tile           [BLOCK_M]
    # - an explicit W matrix    [CHUNK_SIZE, CHUNK_SIZE]
    # - a value-dim tile        [BLOCK_D]
    #
    # Allowed launch-parameter families for this kernel:
    # - BLOCK_M in {32, 64, 128, 256}
    # - BLOCK_D in {32, 64, 128, 256}
    # - BLOCK_K in {32, 64, 128, 256}
    # - CHUNK_SIZE in {32, 64, 128, 256}
    #
    # `BLOCK_M`/`BLOCK_D` are the smallest allowed sizes that cover the logical
    # dimension, while `BLOCK_K` must be an allowed value that exactly divides
    # `D_score`.
    block_m = 32
    block_d = _stablemax_pick_allowed_cover(D_value)
    block_k = _stablemax_pick_allowed_block_k(D_score)
    if chunk_size not in (32, 64, 128, 256):
        raise ValueError(
            "FLAREAutoregressiveStablemaxTriton requires CHUNK_SIZE in {32, 64, 128, 256}. "
            f"Got chunk_size={chunk_size}."
        )
    return {
        "BLOCK_M": block_m,
        "BLOCK_D": block_d,
        "BLOCK_K": block_k,
        "NUM_M_TILES": triton.cdiv(M, block_m),
        "NUM_D_TILES": triton.cdiv(D_value, block_d),
        "input_precision": _normalize_input_precision(input_precision, None),
        "prepare_num_warps": 4 if D_score <= 64 else 8,
        "prepare_num_stages": 2,
        "prefix_num_warps": 4,
        "prefix_num_stages": 2,
        "output_num_warps": 4,
        "output_num_stages": 3 if D_score <= 64 else 2,
    }


@triton.jit
def _stablemax_score_dot_full_panel(lhs, rhs, INPUT_PRECISION: tl.constexpr):
    # Small wrapper so the score matmuls read like the math in the comments:
    #
    #   lhs: [rows, D]
    #   rhs: [cols, D]
    #   out: [rows, cols] = lhs @ rhs^T
    #
    # All score contractions in this file reduce to this form.
    return tl.dot(lhs, tl.trans(rhs), out_dtype=tl.float32, input_precision=INPUT_PRECISION)


@triton.jit
def _stablemax_prepare_score_dot_streamed(
    row_base_ptr,
    col_base_ptr,
    row_offsets,
    col_offsets,
    mask_row,
    mask_col,
    stride_row,
    stride_col,
    stride_d_row,
    stride_d_col,
    D_SCORE: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    scale,
):
    # Encoder-side streamed score builder used by the prepare phase:
    #
    #   scores[row, col] = scale * <row_vec, col_vec>
    scores = tl.zeros((row_offsets.shape[0], col_offsets.shape[0]), dtype=tl.float32)
    for k0 in tl.static_range(0, D_SCORE, BLOCK_K):
        row_ptrs = row_base_ptr + row_offsets[:, None] * stride_row + (k0 + tl.arange(0, BLOCK_K))[None, :] * stride_d_row
        col_ptrs = col_base_ptr + col_offsets[:, None] * stride_col + (k0 + tl.arange(0, BLOCK_K))[None, :] * stride_d_col
        row_tile = tl.load(row_ptrs, mask=mask_row[:, None], other=0.0)
        col_tile = tl.load(col_ptrs, mask=mask_col[:, None], other=0.0)
        scores += _stablemax_score_dot_full_panel(row_tile, col_tile, INPUT_PRECISION=INPUT_PRECISION)
    return scores * scale

@triton.jit
def _stablemax_transform(scores, power, POWER_MODE: tl.constexpr):
    # Same piecewise transform as the PyTorch implementation:
    # - x >= 0: (x + 1)^power
    # - x < 0:  (1 - x)^(-power)
    #
    # Fast paths for the two common public settings:
    # - power == 1: linear / reciprocal
    # - power == 2: square / reciprocal-square
    #
    # The generic path stays in log2/exp2 form so Triton uses the same base-2
    # math family as the online-softmax paths elsewhere in this file.
    pos_base = tl.where(scores >= 0, scores + 1.0, 1.0)
    neg_base = tl.where(scores < 0, 1.0 - scores, 1.0)
    if POWER_MODE == 1:
        pos = pos_base
        neg = 1.0 / neg_base
    elif POWER_MODE == 2:
        pos = pos_base * pos_base
        neg = 1.0 / (neg_base * neg_base)
    else:
        pos = tl.math.exp2(power * tl.math.log2(pos_base))
        neg = tl.math.exp2(-power * tl.math.log2(neg_base))
    return tl.where(scores >= 0, pos, neg)


@triton.jit
def _stablemax_transform_grad(scores, power, POWER_MODE: tl.constexpr):
    # Derivative of the piecewise stablemax score transform.
    #
    # For x >= 0:
    #   s(x)  = (x + 1)^p
    #   s'(x) = p * (x + 1)^(p - 1)
    #
    # For x < 0:
    #   s(x)  = (1 - x)^(-p)
    #   s'(x) = p * (1 - x)^(-(p + 1))
    #
    # The negative branch is positive because d/dx (1 - x)^(-p)
    # introduces a minus sign from the base derivative and another
    # minus sign from the exponent derivative.
    pos_base = tl.where(scores >= 0, scores + 1.0, 1.0)
    neg_base = tl.where(scores < 0, 1.0 - scores, 1.0)
    if POWER_MODE == 1:
        pos = tl.full(scores.shape, 1.0, tl.float32)
        neg = 1.0 / (neg_base * neg_base)
    elif POWER_MODE == 2:
        pos = 2.0 * pos_base
        neg = 2.0 / (neg_base * neg_base * neg_base)
    else:
        pos = power * tl.math.exp2((power - 1.0) * tl.math.log2(pos_base))
        neg = power * tl.math.exp2(-(power + 1.0) * tl.math.log2(neg_base))
    return tl.where(scores >= 0, pos, neg)


@triton.jit
def _stablemax_transform_grad_from_stable(stable, POWER_MODE: tl.constexpr):
    # Recover the stablemax transform derivative directly from the transformed
    # value for the two fast-path power modes we actually optimize:
    #
    # power == 1:
    #   x >= 0 -> stable = x + 1        -> grad = 1
    #   x <  0 -> stable = 1 / (1 - x)  -> grad = stable^2
    #
    # power == 2:
    #   x >= 0 -> stable = (x + 1)^2     -> grad = 2 * sqrt(stable)
    #   x <  0 -> stable = (1 - x)^(-2)  -> grad = 2 * stable^(3/2)
    #
    # We distinguish the branches from `stable >= 1`, which is equivalent to
    # `x >= 0` for these piecewise transforms.
    if POWER_MODE == 1:
        return tl.where(stable >= 1.0, 1.0, stable * stable)
    if POWER_MODE == 2:
        root = tl.sqrt(tl.maximum(stable, 0.0))
        return tl.where(stable >= 1.0, 2.0 * root, 2.0 * stable * root)
    return stable


@triton.jit
def stablemax_bwd_preprocess_delta_kernel(
    O_ptr,
    dO_ptr,
    Delta_ptr,
    stride_o_b,
    stride_o_n,
    stride_o_h,
    stride_o_d,
    stride_do_b,
    stride_do_n,
    stride_do_h,
    stride_do_d,
    stride_delta_bh,
    stride_delta_n,
    N,
    D_VALUE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    H: tl.constexpr,
):
    # FA2-style preprocess:
    #
    #   delta[t] = sum_d O[t, d] * dO[t, d]
    #
    # This lets the decoder-softmax backward use the stable identity
    #
    #   dS_dec[t, m] = P_dec[t, m] * (dAlpha[t, m] - delta[t])
    #
    # instead of materializing a separate row-wise softmax reduction in the
    # hot output-adjoint kernel.
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    t_offsets = tl.arange(0, CHUNK_SIZE)
    global_t = pid_nc * CHUNK_SIZE + t_offsets
    mask_t = global_t < N

    acc = tl.zeros((CHUNK_SIZE,), dtype=tl.float32)
    for d0 in tl.static_range(0, D_VALUE, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)
        mask_d = d_offsets < D_VALUE
        o_ptrs = (
            O_ptr
            + pid_b * stride_o_b
            + global_t[:, None] * stride_o_n
            + pid_h * stride_o_h
            + d_offsets[None, :] * stride_o_d
        )
        do_ptrs = (
            dO_ptr
            + pid_b * stride_do_b
            + global_t[:, None] * stride_do_n
            + pid_h * stride_do_h
            + d_offsets[None, :] * stride_do_d
        )
        o_tile = tl.load(o_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        do_tile = tl.load(do_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(o_tile * do_tile, axis=1)

    delta_ptrs = Delta_ptr + pid_bh * stride_delta_bh + global_t * stride_delta_n
    tl.store(delta_ptrs, acc, mask=mask_t)


@triton.jit
def stablemax_bwd_output_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    QDec_ptr,
    KDec_ptr,
    PrefixDen_ptr,
    PrefixNum_ptr,
    Z_ENC_ptr,
    Z_DEC_ptr,
    dO_ptr,
    Delta_ptr,
    dQDec_ptr,
    dKDec_ptr,
    dV_ptr,
    StableBuf_ptr,
    DecodeOverDenBuf_ptr,
    GradPrevDen_ptr,
    GradPrevNum_ptr,
    stride_q_h,
    stride_q_m,
    stride_q_d,
    stride_k_b,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_qd_b,
    stride_qd_n,
    stride_qd_h,
    stride_qd_d,
    stride_kd_h,
    stride_kd_m,
    stride_kd_d,
    stride_pden_bh,
    stride_pden_nc,
    stride_pden_m,
    stride_pnum_bh,
    stride_pnum_nc,
    stride_pnum_m,
    stride_pnum_d,
    stride_zenc_b,
    stride_zenc_h,
    stride_zenc_n,
    stride_zenc_m,
    stride_zdec_b,
    stride_zdec_h,
    stride_zdec_n,
    stride_do_b,
    stride_do_n,
    stride_do_h,
    stride_do_d,
    stride_delta_bh,
    stride_delta_n,
    stride_dqd_b,
    stride_dqd_n,
    stride_dqd_h,
    stride_dqd_d,
    stride_dkd_h,
    stride_dkd_m,
    stride_dkd_d,
    stride_dv_b,
    stride_dv_n,
    stride_dv_h,
    stride_dv_d,
    stride_sbuf_b,
    stride_sbuf_h,
    stride_sbuf_n,
    stride_sbuf_m,
    stride_rbuf_b,
    stride_rbuf_h,
    stride_rbuf_n,
    stride_rbuf_m,
    stride_gpd_bh,
    stride_gpd_nc,
    stride_gpd_m,
    stride_gpn_bh,
    stride_gpn_nc,
    stride_gpn_m,
    stride_gpn_d,
    N,
    M,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    scale,
    power,
    POWER_MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    # Output-phase reverse mode for one `(batch-head, chunk)`.
    #
    # Forward phase 3 computes, for each token row `t` in the chunk,
    #
    #   y_t = sum_m p_dec[t, m] * z_t[m]
    #
    # where
    #
    #   z_t[m] = (prefix_num[m] + sum_{tau <= t} stable[tau, m] * v_tau) / Z_ENC[t, m]
    #
    # This kernel handles every gradient that depends on decoder probabilities or
    # on the chunk-local replay:
    #
    # - `dQ_dec`, `dK_dec` through the decoder softmax
    # - the phase-3 contribution to `dV`
    # - `d prefix_den` / `d prefix_num`, emitted as chunk-summary adjoints
    # - cached `stable` and `decode_over_den`, which the encoder kernel reuses
    #
    # The key design choice is that the program owns the full chunk token axis.
    # That lets us form the chunk-local `[C, C]` replay adjoint with dense tensor
    # ops once and reuse it across all latent tiles instead of rebuilding it
    # tile-by-tile.
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    t_offsets = tl.arange(0, CHUNK_SIZE)
    global_t = pid_nc * CHUNK_SIZE + t_offsets
    mask_t = global_t < N
    k_offsets = tl.arange(0, BLOCK_K)
    causal_mask = global_t[:, None] >= global_t[None, :]

    qd_bh_base = QDec_ptr + pid_b * stride_qd_b + pid_h * stride_qd_h
    kd_h_base = KDec_ptr + pid_h * stride_kd_h
    q_h_base = Q_ptr + pid_h * stride_q_h
    k_bh_base = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h

    do_ptrs = (
        dO_ptr
        + pid_b * stride_do_b
        + global_t[:, None] * stride_do_n
        + pid_h * stride_do_h
        + tl.arange(0, D_VALUE)[None, :] * stride_do_d
    )
    v_ptrs = (
        V_ptr
        + pid_b * stride_v_b
        + global_t[:, None] * stride_v_n
        + pid_h * stride_v_h
        + tl.arange(0, D_VALUE)[None, :] * stride_v_d
    )
    do_tile = tl.load(do_ptrs, mask=mask_t[:, None], other=0.0)
    v_tile = tl.load(v_ptrs, mask=mask_t[:, None], other=0.0)
    do_lowp = do_tile.to(v_tile.dtype)
    qd_ptrs = qd_bh_base + global_t[:, None] * stride_qd_n + k_offsets[None, :] * stride_qd_d
    k_ptrs = k_bh_base + global_t[:, None] * stride_k_n + k_offsets[None, :] * stride_k_d
    qd_tile = tl.load(qd_ptrs, mask=mask_t[:, None], other=0.0)
    k_tile = tl.load(k_ptrs, mask=mask_t[:, None], other=0.0)
    zdec_ptrs = Z_DEC_ptr + pid_b * stride_zdec_b + pid_h * stride_zdec_h + global_t * stride_zdec_n
    z_dec = tl.load(zdec_ptrs, mask=mask_t, other=0.0).to(tl.float32)

    # `grad_local_mix = dO @ V^T` is the reverse-mode analogue of the final
    # local replay contraction `(W * causal_mask) @ V`.
    #
    # It only depends on chunk-local token rows and value vectors, so we build
    # it once up front and reuse it across every latent tile in the loop below.
    grad_local_mix = tl.dot(do_lowp, tl.trans(v_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    grad_local_mix = tl.where(mask_t[:, None] & mask_t[None, :] & causal_mask, grad_local_mix, 0.0)
    delta_ptrs = Delta_ptr + pid_bh * stride_delta_bh + global_t * stride_delta_n
    delta = tl.load(delta_ptrs, mask=mask_t, other=0.0).to(tl.float32)
    dq_dec_total = tl.zeros((CHUNK_SIZE, BLOCK_K), dtype=tl.float32)
    dv_phase3_total = tl.zeros((CHUNK_SIZE, D_VALUE), dtype=tl.float32)

    for m0 in tl.range(0, M, BLOCK_M):
        # One latent tile gives us:
        # - decoder scores / probabilities for this `m` slice
        # - encoder stable weights for the same `m` slice
        # - chunk-summary adjoints that will be suffix-scanned later
        #
        # The loop is over latent tiles only; there is no token streaming here.
        # All token interactions stay in `[C, BLOCK_M]` or `[C, C]` tensor math.
        m_offsets = m0 + tl.arange(0, BLOCK_M)
        mask_m = m_offsets < M

        kd_ptrs = kd_h_base + m_offsets[:, None] * stride_kd_m + k_offsets[None, :] * stride_kd_d
        q_ptrs = q_h_base + m_offsets[:, None] * stride_q_m + k_offsets[None, :] * stride_q_d
        kd_tile = tl.load(kd_ptrs, mask=mask_m[:, None], other=0.0)
        q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        dec_scores = _stablemax_score_dot_full_panel(qd_tile, kd_tile, INPUT_PRECISION=INPUT_PRECISION) * scale
        enc_scores = _stablemax_score_dot_full_panel(k_tile, q_tile, INPUT_PRECISION=INPUT_PRECISION) * scale
        stable = _stablemax_transform(enc_scores, power, POWER_MODE=POWER_MODE)
        stable = tl.where(mask_t[:, None] & mask_m[None, :], stable, 0.0)

        zenc_ptrs = (
            Z_ENC_ptr
            + pid_b * stride_zenc_b
            + pid_h * stride_zenc_h
            + global_t[:, None] * stride_zenc_n
            + m_offsets[None, :] * stride_zenc_m
        )
        total_den = tl.load(zenc_ptrs, mask=mask_t[:, None] & mask_m[None, :], other=0.0).to(tl.float32)
        inv_total_den = tl.where(total_den > 0, 1.0 / total_den, 0.0)
        p_dec = tl.math.exp2(dec_scores * RCP_LN2 - z_dec[:, None] * RCP_LN2)
        p_dec = tl.where(mask_t[:, None] & mask_m[None, :], p_dec, 0.0)
        decode_over_den = p_dec * inv_total_den

        # Cache the encoder-side quantities that the later encoder kernel needs.
        # This avoids paying for a second encoder transform / normalization
        # rebuild in the backward half that propagates through `stable`.
        sbuf_ptrs = (
            StableBuf_ptr
            + pid_b * stride_sbuf_b
            + pid_h * stride_sbuf_h
            + global_t[:, None] * stride_sbuf_n
            + m_offsets[None, :] * stride_sbuf_m
        )
        tl.store(sbuf_ptrs, stable, mask=mask_t[:, None] & mask_m[None, :])
        rbuf_ptrs = (
            DecodeOverDenBuf_ptr
            + pid_b * stride_rbuf_b
            + pid_h * stride_rbuf_h
            + global_t[:, None] * stride_rbuf_n
            + m_offsets[None, :] * stride_rbuf_m
        )
        tl.store(rbuf_ptrs, decode_over_den, mask=mask_t[:, None] & mask_m[None, :])

        pnum_ptrs = (
            PrefixNum_ptr
            + pid_bh * stride_pnum_bh
            + pid_nc * stride_pnum_nc
            + m_offsets[:, None] * stride_pnum_m
            + tl.arange(0, D_VALUE)[None, :] * stride_pnum_d
        )
        prefix_num = tl.load(pnum_ptrs, mask=mask_m[:, None], other=0.0)

        # Reverse-mode of
        #
        #   y = p_dec @ (prefix_num / Z_ENC) + (p_dec / Z_ENC @ stable^T * causal) @ V
        #
        # `grad_decode_over_den` is the adjoint of the normalized replay weight
        #
        #   decode_over_den = p_dec / Z_ENC
        #
        # for this latent tile. It has two sources:
        # - the prefix replay term `p_dec @ prefix_num`
        # - the local replay term that goes through `stable` and `V`
        grad_decode_over_den = tl.dot(
            do_lowp,
            tl.trans(prefix_num),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
        grad_decode_over_den += tl.dot(
            grad_local_mix.to(v_tile.dtype),
            stable.to(v_tile.dtype),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )

        # `grad_prev_num` is the chunk-summary adjoint of `prefix_num`. The
        # reverse scan kernel will turn these strict-prefix adjoints into the
        # per-chunk summary gradients that feed earlier chunks.
        grad_prev_num = tl.dot(
            tl.trans(decode_over_den.to(v_tile.dtype)),
            do_lowp,
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )

        # Rebuild the same local replay mixing matrix used in forward:
        #
        #   local_mix[t, tau] = sum_m decode_over_den[t, m] * stable[tau, m]
        #
        # then mask it to the causal lower triangle and send the resulting value
        # adjoint back into `V`.
        local_mix = tl.dot(
            decode_over_den.to(v_tile.dtype),
            tl.trans(stable.to(v_tile.dtype)),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
        local_mix = tl.where(mask_t[:, None] & mask_t[None, :] & causal_mask, local_mix, 0.0)
        dv_phase3_total += tl.dot(
            tl.trans(local_mix.to(v_tile.dtype)),
            do_lowp,
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )

        # Decoder softmax backward:
        # - `grad_decode_probs` is the adjoint of the normalized decoder
        #   probability itself
        # - `grad_total_den` is the part that flows through `1 / Z_ENC`
        # - `delta` encodes the row-wise dot-product softmax correction
        grad_decode_probs = grad_decode_over_den * inv_total_den
        grad_total_den = -(grad_decode_over_den * p_dec) * inv_total_den * inv_total_den
        grad_prev_den = tl.sum(grad_total_den, axis=0)

        ds_dec = p_dec * (grad_decode_probs - delta[:, None])
        dq_dec_total += tl.dot(ds_dec.to(kd_tile.dtype), kd_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale
        dkd = tl.dot(tl.trans(ds_dec.to(qd_tile.dtype)), qd_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale

        dkd_ptrs = (
            dKDec_ptr
            + pid_h * stride_dkd_h
            + m_offsets[:, None] * stride_dkd_m
            + k_offsets[None, :] * stride_dkd_d
        )
        tl.atomic_add(dkd_ptrs, dkd, mask=mask_m[:, None])

        gpd_ptrs = GradPrevDen_ptr + pid_bh * stride_gpd_bh + pid_nc * stride_gpd_nc + m_offsets * stride_gpd_m
        tl.store(gpd_ptrs, grad_prev_den, mask=mask_m)
        gpn_ptrs = (
            GradPrevNum_ptr
            + pid_bh * stride_gpn_bh
            + pid_nc * stride_gpn_nc
            + m_offsets[:, None] * stride_gpn_m
            + tl.arange(0, D_VALUE)[None, :] * stride_gpn_d
        )
        tl.store(gpn_ptrs, grad_prev_num, mask=mask_m[:, None])

    dqd_ptrs = (
        dQDec_ptr
        + pid_b * stride_dqd_b
        + global_t[:, None] * stride_dqd_n
        + pid_h * stride_dqd_h
        + k_offsets[None, :] * stride_dqd_d
    )
    tl.store(dqd_ptrs, dq_dec_total, mask=mask_t[:, None])

    dv_ptrs = (
        dV_ptr
        + pid_b * stride_dv_b
        + global_t[:, None] * stride_dv_n
        + pid_h * stride_dv_h
        + tl.arange(0, D_VALUE)[None, :] * stride_dv_d
    )
    tl.store(dv_ptrs, dv_phase3_total, mask=mask_t[:, None])


@triton.jit
def stablemax_bwd_reverse_scan_den_kernel(
    GradPrevDen_ptr,
    GradChunkDen_ptr,
    stride_gpd_bh,
    stride_gpd_nc,
    stride_gpd_m,
    stride_gcd_bh,
    stride_gcd_nc,
    stride_gcd_m,
    M,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Reverse-mode of the strict prefix scan:
    #
    #   prev[c] = sum_{c' < c} chunk[c']
    #
    # so
    #
    #   dchunk[c] = sum_{c' > c} dprev[c']
    #
    # Each program owns one `(bh, m_tile)` slice and carries a suffix sum
    # right-to-left over the chunk axis.
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    suffix = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for nc in range(NUM_CHUNKS - 1, -1, -1):
        gcd_ptrs = GradChunkDen_ptr + pid_bh * stride_gcd_bh + nc * stride_gcd_nc + m_offsets * stride_gcd_m
        tl.store(gcd_ptrs, suffix, mask=mask_m)
        gpd_ptrs = GradPrevDen_ptr + pid_bh * stride_gpd_bh + nc * stride_gpd_nc + m_offsets * stride_gpd_m
        suffix += tl.load(gpd_ptrs, mask=mask_m, other=0.0).to(tl.float32)


@triton.jit
def stablemax_bwd_reverse_scan_num_kernel(
    GradPrevNum_ptr,
    GradChunkNum_ptr,
    stride_gpn_bh,
    stride_gpn_nc,
    stride_gpn_m,
    stride_gpn_d,
    stride_gcn_bh,
    stride_gcn_nc,
    stride_gcn_m,
    stride_gcn_d,
    M,
    D_VALUE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Same reverse strict-prefix logic as the denominator scan, but now for the
    # value-expanded chunk summaries:
    #
    #   prefix_num[c] = sum_{c' < c} chunk_num[c']
    #
    # so every `d prefix_num[c]` contributes to all earlier chunks `c' < c`.
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    suffix = tl.zeros((BLOCK_M, D_VALUE), dtype=tl.float32)
    for nc in range(NUM_CHUNKS - 1, -1, -1):
        gcn_ptrs = (
            GradChunkNum_ptr
            + pid_bh * stride_gcn_bh
            + nc * stride_gcn_nc
            + m_offsets[:, None] * stride_gcn_m
            + tl.arange(0, D_VALUE)[None, :] * stride_gcn_d
        )
        tl.store(gcn_ptrs, suffix, mask=mask_m[:, None])
        gpn_ptrs = (
            GradPrevNum_ptr
            + pid_bh * stride_gpn_bh
            + nc * stride_gpn_nc
            + m_offsets[:, None] * stride_gpn_m
            + tl.arange(0, D_VALUE)[None, :] * stride_gpn_d
        )
        suffix += tl.load(gpn_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)


@triton.jit
def stablemax_bwd_encoder_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    QDec_ptr,
    KDec_ptr,
    PrefixDen_ptr,
    PrefixNum_ptr,
    Z_ENC_ptr,
    Z_DEC_ptr,
    dO_ptr,
    StableBuf_ptr,
    DecodeOverDenBuf_ptr,
    GradChunkDen_ptr,
    GradChunkNum_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    stride_q_h,
    stride_q_m,
    stride_q_d,
    stride_k_b,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_qd_b,
    stride_qd_n,
    stride_qd_h,
    stride_qd_d,
    stride_kd_h,
    stride_kd_m,
    stride_kd_d,
    stride_pden_bh,
    stride_pden_nc,
    stride_pden_m,
    stride_pnum_bh,
    stride_pnum_nc,
    stride_pnum_m,
    stride_pnum_d,
    stride_zenc_b,
    stride_zenc_h,
    stride_zenc_n,
    stride_zenc_m,
    stride_zdec_b,
    stride_zdec_h,
    stride_zdec_n,
    stride_do_b,
    stride_do_n,
    stride_do_h,
    stride_do_d,
    stride_sbuf_b,
    stride_sbuf_h,
    stride_sbuf_n,
    stride_sbuf_m,
    stride_rbuf_b,
    stride_rbuf_h,
    stride_rbuf_n,
    stride_rbuf_m,
    stride_gcd_bh,
    stride_gcd_nc,
    stride_gcd_m,
    stride_gcn_bh,
    stride_gcn_nc,
    stride_gcn_m,
    stride_gcn_d,
    stride_dq_h,
    stride_dq_m,
    stride_dq_d,
    stride_dk_b,
    stride_dk_n,
    stride_dk_h,
    stride_dk_d,
    stride_dv_b,
    stride_dv_n,
    stride_dv_h,
    stride_dv_d,
    N,
    M,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    scale,
    power,
    POWER_MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    # Encoder-side reverse mode for one `(batch-head, chunk)`.
    #
    # By the time we reach this kernel, the output kernel and reverse scans have
    # already produced:
    # - cached `stable[t, m]`
    # - cached `decode_over_den[t, m] = p_dec[t, m] / Z_ENC[t, m]`
    # - suffix-scanned chunk-summary adjoints `grad_chunk_den` / `grad_chunk_num`
    #
    # This kernel combines those three sources into the adjoint of encoder
    # stable weights, then projects that gradient back through:
    #
    #   stable = stablemax_transform(enc_scores)
    #   enc_scores = scale * K_chunk @ Q_tile^T
    #
    # It also accumulates the summary-path contribution to `dV`.
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    t_offsets = tl.arange(0, CHUNK_SIZE)
    global_t = pid_nc * CHUNK_SIZE + t_offsets
    mask_t = global_t < N
    k_offsets = tl.arange(0, BLOCK_K)
    causal_mask = global_t[:, None] >= global_t[None, :]

    qd_bh_base = QDec_ptr + pid_b * stride_qd_b + pid_h * stride_qd_h
    kd_h_base = KDec_ptr + pid_h * stride_kd_h
    q_h_base = Q_ptr + pid_h * stride_q_h
    k_bh_base = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h

    do_ptrs = (
        dO_ptr
        + pid_b * stride_do_b
        + global_t[:, None] * stride_do_n
        + pid_h * stride_do_h
        + tl.arange(0, D_VALUE)[None, :] * stride_do_d
    )
    v_ptrs = (
        V_ptr
        + pid_b * stride_v_b
        + global_t[:, None] * stride_v_n
        + pid_h * stride_v_h
        + tl.arange(0, D_VALUE)[None, :] * stride_v_d
    )
    do_tile = tl.load(do_ptrs, mask=mask_t[:, None], other=0.0)
    v_tile = tl.load(v_ptrs, mask=mask_t[:, None], other=0.0)
    do_lowp = do_tile.to(v_tile.dtype)

    # As in the output kernel, `grad_local_mix = dO @ V^T` is purely chunk-local
    # and shared by every latent tile. Hoist it once so the latent loop only
    # pays for latent-dependent work.
    grad_local_mix = tl.dot(do_lowp, tl.trans(v_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    grad_local_mix = tl.where(mask_t[:, None] & mask_t[None, :] & causal_mask, grad_local_mix, 0.0)
    k_ptrs = k_bh_base + global_t[:, None] * stride_k_n + k_offsets[None, :] * stride_k_d
    k_tile = tl.load(k_ptrs, mask=mask_t[:, None], other=0.0)

    dk_total = tl.zeros((CHUNK_SIZE, BLOCK_K), dtype=tl.float32)
    dv_summary_total = tl.zeros((CHUNK_SIZE, D_VALUE), dtype=tl.float32)
    rev_causal = global_t[:, None] <= global_t[None, :]

    for m0 in tl.range(0, M, BLOCK_M):
        # This loop reconstructs the adjoint of encoder stable weights for one
        # latent tile by combining:
        # - phase-3 local replay gradients
        # - summary-prefix gradients emitted by later chunks
        # - the stablemax transform derivative
        m_offsets = m0 + tl.arange(0, BLOCK_M)
        mask_m = m_offsets < M

        q_ptrs = q_h_base + m_offsets[:, None] * stride_q_m + k_offsets[None, :] * stride_q_d
        q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        sbuf_ptrs = (
            StableBuf_ptr
            + pid_b * stride_sbuf_b
            + pid_h * stride_sbuf_h
            + global_t[:, None] * stride_sbuf_n
            + m_offsets[None, :] * stride_sbuf_m
        )
        stable = tl.load(sbuf_ptrs, mask=mask_t[:, None] & mask_m[None, :], other=0.0).to(tl.float32)
        if POWER_MODE == 0:
            enc_scores = _stablemax_score_dot_full_panel(k_tile, q_tile, INPUT_PRECISION=INPUT_PRECISION) * scale
            stable_grad = _stablemax_transform_grad(enc_scores, power, POWER_MODE=POWER_MODE)
            stable_grad = tl.where(mask_t[:, None] & mask_m[None, :], stable_grad, 0.0)
        else:
            stable_grad = _stablemax_transform_grad_from_stable(stable, POWER_MODE=POWER_MODE)
            stable_grad = tl.where(mask_t[:, None] & mask_m[None, :], stable_grad, 0.0)

        zenc_ptrs = (
            Z_ENC_ptr
            + pid_b * stride_zenc_b
            + pid_h * stride_zenc_h
            + global_t[:, None] * stride_zenc_n
            + m_offsets[None, :] * stride_zenc_m
        )
        total_den = tl.load(zenc_ptrs, mask=mask_t[:, None] & mask_m[None, :], other=0.0).to(tl.float32)
        inv_total_den = tl.where(total_den > 0, 1.0 / total_den, 0.0)
        rbuf_ptrs = (
            DecodeOverDenBuf_ptr
            + pid_b * stride_rbuf_b
            + pid_h * stride_rbuf_h
            + global_t[:, None] * stride_rbuf_n
            + m_offsets[None, :] * stride_rbuf_m
        )
        decode_over_den = tl.load(rbuf_ptrs, mask=mask_t[:, None] & mask_m[None, :], other=0.0).to(tl.float32)

        pnum_ptrs = (
            PrefixNum_ptr
            + pid_bh * stride_pnum_bh
            + pid_nc * stride_pnum_nc
            + m_offsets[:, None] * stride_pnum_m
            + tl.arange(0, D_VALUE)[None, :] * stride_pnum_d
        )
        prefix_num = tl.load(pnum_ptrs, mask=mask_m[:, None], other=0.0)
        grad_decode_over_den = tl.dot(
            do_lowp,
            tl.trans(prefix_num),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
        grad_decode_over_den += tl.dot(
            grad_local_mix.to(v_tile.dtype),
            stable.to(v_tile.dtype),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )

        # Reverse-mode of `decode_over_den = p_dec / Z_ENC` with respect to the
        # encoder normalization `Z_ENC`. This contributes to the adjoint of the
        # cumulative stable weights before we fold in the summary-path terms.
        grad_total_den = -(grad_decode_over_den * decode_over_den) * inv_total_den

        # `grad_w_phase3` is the adjoint of the cumulative encoder replay
        #
        #   W[t, m] = sum_{tau <= t} stable[tau, m]
        #
        # coming only from phase 3. The first term comes from the local replay
        # matrix, and the second term is the cumulative effect of `Z_ENC`.
        grad_w_phase3 = tl.dot(
            tl.trans(grad_local_mix.to(v_tile.dtype)),
            decode_over_den.to(v_tile.dtype),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
        grad_w_phase3 += tl.dot(
            rev_causal.to(v_tile.dtype),
            grad_total_den.to(v_tile.dtype),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )

        gcd_ptrs = GradChunkDen_ptr + pid_bh * stride_gcd_bh + pid_nc * stride_gcd_nc + m_offsets * stride_gcd_m
        grad_chunk_den = tl.load(gcd_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        gcn_ptrs = (
            GradChunkNum_ptr
            + pid_bh * stride_gcn_bh
            + pid_nc * stride_gcn_nc
            + m_offsets[:, None] * stride_gcn_m
            + tl.arange(0, D_VALUE)[None, :] * stride_gcn_d
        )
        grad_chunk_num = tl.load(gcn_ptrs, mask=mask_m[:, None], other=0.0).to(tl.float32)

        # Add the summary-path adjoints:
        # - `grad_chunk_den` contributes a per-latent bias to every token row
        # - `grad_chunk_num` contributes through the summary contraction
        #   `chunk_num = stable^T @ V`
        grad_w_total = grad_w_phase3 + grad_chunk_den[None, :]
        grad_w_total += tl.dot(v_tile, tl.trans(grad_chunk_num.to(v_tile.dtype)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        dv_summary_total += tl.dot(stable.to(v_tile.dtype), grad_chunk_num.to(v_tile.dtype), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        # Finally, push through the stablemax transform and the encoder score
        # projection `enc_scores = scale * K_chunk @ Q_tile^T`.
        grad_score_enc = grad_w_total * stable_grad
        dk_total += tl.dot(grad_score_enc.to(q_tile.dtype), q_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale
        dq = tl.dot(tl.trans(grad_score_enc.to(k_tile.dtype)), k_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale

        dq_ptrs = (
            dQ_ptr
            + pid_h * stride_dq_h
            + m_offsets[:, None] * stride_dq_m
            + k_offsets[None, :] * stride_dq_d
        )
        tl.atomic_add(dq_ptrs, dq, mask=mask_m[:, None])

    dk_ptrs = (
        dK_ptr
        + pid_b * stride_dk_b
        + global_t[:, None] * stride_dk_n
        + pid_h * stride_dk_h
        + k_offsets[None, :] * stride_dk_d
    )
    tl.store(dk_ptrs, dk_total, mask=mask_t[:, None])
    dv_ptrs = (
        dV_ptr
        + pid_b * stride_dv_b
        + global_t[:, None] * stride_dv_n
        + pid_h * stride_dv_h
        + tl.arange(0, D_VALUE)[None, :] * stride_dv_d
    )
    dv_prev = tl.load(dv_ptrs, mask=mask_t[:, None], other=0.0)
    tl.store(dv_ptrs, dv_prev + dv_summary_total, mask=mask_t[:, None])


@triton.jit
def _stablemax_output_apply_tile(
    dec_scores_log2,
    enc_scores,
    m_offsets,
    mask_m,
    Z_ENC_ptr,
    PrefixDen_ptr,
    PrefixNum_ptr,
    stride_zenc_b,
    stride_zenc_h,
    stride_zenc_n,
    stride_zenc_m,
    stride_pden_bh,
    stride_pden_nc,
    stride_pden_m,
    stride_pnum_bh,
    stride_pnum_nc,
    stride_pnum_m,
    stride_pnum_d,
    pid_b,
    pid_h,
    pid_bh,
    pid_nc,
    pid_d,
    global_t,
    mask_t,
    d_offsets,
    mask_d,
    causal_mask,
    v_tile,
    row_max,
    row_sum,
    y_num,
    scale,
    power,
    INPUT_PRECISION: tl.constexpr,
    POWER_MODE: tl.constexpr,
):
    # Consume one completed `(decoder scores, encoder scores)` latent tile in
    # the fused output phase. This keeps the score-building policy separate from
    # the replay/output math so the streamed path can batch score construction
    # across multiple latent tiles while reusing the same row-side loads.
    dec_scores_log2 = tl.where(mask_t[:, None] & mask_m[None, :], dec_scores_log2, -float("inf"))
    stable_full = _stablemax_transform(enc_scores, power, POWER_MODE=POWER_MODE)
    stable_full = tl.where(mask_t[:, None] & mask_m[None, :], stable_full, 0.0)

    pden_ptrs = PrefixDen_ptr + pid_bh * stride_pden_bh + pid_nc * stride_pden_nc + m_offsets * stride_pden_m
    prefix_den = tl.load(pden_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    local_den = tl.cumsum(stable_full, axis=0)
    total_den = prefix_den[None, :] + local_den
    inv_total_den = tl.where(total_den > 0, 1.0 / total_den, 0.0)

    if pid_d == 0:
        zenc_ptrs = (
            Z_ENC_ptr
            + pid_b * stride_zenc_b
            + pid_h * stride_zenc_h
            + global_t[:, None] * stride_zenc_n
            + m_offsets[None, :] * stride_zenc_m
        )
        tl.store(zenc_ptrs, total_den, mask=mask_t[:, None] & mask_m[None, :])

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

    # Follow the FA2 pattern for the hot value-side contractions: keep the
    # online-softmax and denominator state in fp32, but cast the actual dot
    # operands down to the public value dtype so Triton can use the bf16/fp16
    # tensor-core path while still accumulating in fp32.
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
def stablemax_prepare_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    ChunkDen_ptr,
    ChunkNum_ptr,
    stride_q_h,
    stride_q_m,
    stride_q_d,
    stride_k_b,
    stride_k_n,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_den_bh,
    stride_den_nc,
    stride_den_m,
    stride_num_bh,
    stride_num_nc,
    stride_num_m,
    stride_num_d,
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
    # Phase 1: compute the encoder-side stablemax chunk summaries.
    #
    # Forward decomposition
    # ---------------------
    # The PyTorch reference starts by computing, for each chunk:
    #
    #   score_chunk[t, m]        = scale * <k_t, q_m>
    #   stable_score_chunk[t, m] = stablemax(score_chunk[t, m])
    #   score_chunk_den[m]       = sum_t stable_score_chunk[t, m]
    #   score_chunk_num[m, d]    = sum_t stable_score_chunk[t, m] * v_t[d]
    #
    # This kernel is exactly that phase.
    #
    # Program ownership
    # -----------------
    # Each program owns one:
    # - flattened batch-head lane `(b, h)`
    # - encoder chunk index `nc`
    # - latent tile `m_offsets`
    # - value-dimension tile `d_offsets`
    #
    # Layouts seen by this kernel
    # ---------------------------
    # Public inputs:
    # - Q: [H, M, D_score]
    # - K: [B, N, H, D_score]
    # - V: [B, N, H, D_value]
    #
    # Internal chunk-local view for the current program:
    # - token axis: global_tokens = nc * CHUNK_SIZE + token_offsets
    # - scores:     [CHUNK_SIZE, BLOCK_M]
    # - stable:     [CHUNK_SIZE, BLOCK_M]
    # - num:        [BLOCK_M, BLOCK_D]
    #
    # Stored outputs:
    # - ChunkDen: [B*H, NC, M] in fp32
    # - ChunkNum: [B*H, NC, M, D_value] in the public value dtype
    #
    # Phase 3 now recomputes the per-token encoder stable weights directly, so
    # phase 1 only needs to keep the chunk-level denominator and numerator
    # summaries. Even though `ChunkNum` is stored in `V.dtype`, the contraction
    # itself is still accumulated in fp32 before the final store.
    pid_bh = tl.program_id(0)
    pid_nc = tl.program_id(1)
    pid_md = tl.program_id(2)
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    num_d_tiles = tl.cdiv(D_VALUE, BLOCK_D)
    pid_m = pid_md // num_d_tiles
    pid_d = pid_md - pid_m * num_d_tiles

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    token_offsets = tl.arange(0, CHUNK_SIZE)
    global_tokens = pid_nc * CHUNK_SIZE + token_offsets
    mask_t = global_tokens < N
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE

    # Build the encoder score tile
    # ----------------------------
    # `scores` corresponds to the reference `score_chunk` restricted to one
    # chunk and one latent tile.
    #
    # For the common `D_SCORE <= BLOCK_K` case, the full score-head dimension
    # fits in one panel, so we load the entire `[C, D]` token tile and the
    # entire `[BLOCK_M, D]` latent tile once and use a single dense `tl.dot`.
    #
    # Only larger score heads need the streamed helper.
    k_bh_base = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h
    q_h_base = Q_ptr + pid_h * stride_q_h
    if D_SCORE <= BLOCK_K:
        k_ptrs = k_bh_base + global_tokens[:, None] * stride_k_n + tl.arange(0, BLOCK_K)[None, :] * stride_k_d
        q_ptrs = q_h_base + m_offsets[:, None] * stride_q_m + tl.arange(0, BLOCK_K)[None, :] * stride_q_d
        k_tile = tl.load(k_ptrs, mask=mask_t[:, None], other=0.0)
        q_tile = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
        scores = _stablemax_score_dot_full_panel(k_tile, q_tile, INPUT_PRECISION=INPUT_PRECISION) * scale
    else:
        scores = _stablemax_prepare_score_dot_streamed(
            k_bh_base,
            q_h_base,
            global_tokens,
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

    # Apply the stablemax score transform elementwise:
    #
    #   stable[t, m] = s(score[t, m])
    #
    # Invalid token / latent lanes are zeroed so later reductions over the chunk
    # can treat the padded lanes as neutral elements.
    stable = _stablemax_transform(scores, power, POWER_MODE=POWER_MODE)
    stable = tl.where(mask_t[:, None] & mask_m[None, :], stable, 0.0)

    if pid_d == 0:
        # Denominator summary for this latent tile:
        #
        #   chunk_den[m] = sum_t stable[t, m]
        #
        # Only the `pid_d == 0` programs write this because the denominator is
        # shared across all value-dimension tiles.
        den = tl.sum(stable, axis=0)
        den_ptrs = ChunkDen_ptr + pid_bh * stride_den_bh + pid_nc * stride_den_nc + m_offsets * stride_den_m
        tl.store(den_ptrs, den, mask=mask_m)

    # Numerator summary for this `(latent tile, value tile)` pair:
    #
    #   chunk_num[m, d] = stable[:, m]^T @ V_chunk[:, d]
    #
    # This is the tensor form of the reference `torch.bmm` contraction.
    v_ptrs = (
        V_ptr
        + pid_b * stride_v_b
        + global_tokens[:, None] * stride_v_n
        + pid_h * stride_v_h
        + d_offsets[None, :] * stride_v_d
    )
    # Keep the loaded value tile in the public V dtype. The contraction still
    # accumulates in fp32 via `out_dtype=tl.float32`.
    v_tile = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)
    num = tl.dot(tl.trans(stable.to(v_tile.dtype)), v_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    num_ptrs = (
        ChunkNum_ptr
        + pid_bh * stride_num_bh
        + pid_nc * stride_num_nc
        + m_offsets[:, None] * stride_num_m
        + d_offsets[None, :] * stride_num_d
    )
    tl.store(num_ptrs, num, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def stablemax_prefix_scan_kernel(
    ChunkDen_ptr,
    ChunkNum_ptr,
    PrefixDen_ptr,
    PrefixNum_ptr,
    stride_den_bh,
    stride_den_nc,
    stride_den_m,
    stride_num_bh,
    stride_num_nc,
    stride_num_m,
    stride_num_d,
    stride_pden_bh,
    stride_pden_nc,
    stride_pden_m,
    stride_pnum_bh,
    stride_pnum_nc,
    stride_pnum_m,
    stride_pnum_d,
    M,
    D_VALUE: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Phase 2: strict prefix scan over encoder chunks.
    #
    # Reference math
    # --------------
    # The PyTorch implementation constructs strict chunk prefixes:
    #
    #   prefix_den[c] = sum_{c' < c} chunk_den[c']
    #   prefix_num[c] = sum_{c' < c} chunk_num[c']
    #
    # so that chunk `c` can combine:
    # - all earlier-chunk encoder state through the prefix term
    # - the current chunk's within-chunk causal contribution through replay
    #
    # Program ownership
    # -----------------
    # Each program owns one `(batch-head, latent tile, value-dim tile)` slice of
    # the prefix scan state and walks the chunk axis left-to-right.
    #
    # Layouts:
    # - ChunkDen:  [B*H, NC, M] in fp32
    # - ChunkNum:  [B*H, NC, M, D_value] in the public value dtype
    # - PrefixDen: [B*H, NC, M] in fp32
    # - PrefixNum: [B*H, NC, M, D_value] in the public value dtype
    #
    # As in phase 1, the running prefix state stays in fp32 inside the kernel
    # and is only cast on store.
    pid_bh = tl.program_id(0)
    pid_md = tl.program_id(1)

    num_d_tiles = tl.cdiv(D_VALUE, BLOCK_D)
    pid_m = pid_md // num_d_tiles
    pid_d = pid_md - pid_m * num_d_tiles

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE

    # Running strict prefix state. By storing before the update below, chunk `nc`
    # receives the sum of chunks `0 .. nc-1` and not its own chunk summary.
    prefix_den = tl.zeros((BLOCK_M,), dtype=tl.float32)
    prefix_num = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for nc in tl.range(0, NUM_CHUNKS):
        if pid_d == 0:
            pden_ptrs = PrefixDen_ptr + pid_bh * stride_pden_bh + nc * stride_pden_nc + m_offsets * stride_pden_m
            tl.store(pden_ptrs, prefix_den, mask=mask_m)

        pnum_ptrs = (
            PrefixNum_ptr
            + pid_bh * stride_pnum_bh
            + nc * stride_pnum_nc
            + m_offsets[:, None] * stride_pnum_m
            + d_offsets[None, :] * stride_pnum_d
        )
        tl.store(pnum_ptrs, prefix_num, mask=mask_m[:, None] & mask_d[None, :])

        if pid_d == 0:
            # Update the denominator prefix after storing the current strict
            # prefix value.
            den_ptrs = ChunkDen_ptr + pid_bh * stride_den_bh + nc * stride_den_nc + m_offsets * stride_den_m
            prefix_den += tl.load(den_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        # Same strict-prefix update for the numerator summary.
        num_ptrs = (
            ChunkNum_ptr
            + pid_bh * stride_num_bh
            + nc * stride_num_nc
            + m_offsets[:, None] * stride_num_m
            + d_offsets[None, :] * stride_num_d
        )
        prefix_num += tl.load(num_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)


@triton.jit
def stablemax_output_kernel(
    Q_ptr,
    K_ptr,
    QDec_ptr,
    KDec_ptr,
    Z_ENC_ptr,
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
    stride_zenc_b,
    stride_zenc_h,
    stride_zenc_n,
    stride_zenc_m,
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

    # This kernel owns the full latent reduction locally:
    #
    # - no atomics
    # - no 3D `z[t, m, d]`
    #
    # It keeps the decoder softmax state online across latent tiles and, for
    # each `m` tile, uses only 2D tensors:
    #
    #   p_tile_eff    [C, BLOCK_M]
    #   stable_full   [C, BLOCK_M]
    #   prefix_partial[C, BLOCK_D]
    #   W             [C, C]
    #   local_partial [C, BLOCK_D]
    #
    # The current-tile contribution is
    #
    #   contribution = p_tile_eff @ prefix_num
    #                + ((p_tile_eff @ stable^T) * causal_mask) @ V_chunk
    #
    # and online-softmax rescales the previously accumulated numerator when the
    # running row max changes.
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
    # As in phase 1, keep the source value tile in the public V dtype and rely
    # on the dot products below to accumulate in fp32.
    v_tile = tl.load(v_ptrs, mask=mask_t[:, None] & mask_d[None, :], other=0.0)

    if D_SCORE <= BLOCK_K:
        qd_ptrs = qd_bh_base + global_t[:, None] * stride_qd_n + tl.arange(0, BLOCK_K)[None, :] * stride_qd_d
        qd_tile = tl.load(qd_ptrs, mask=mask_t[:, None], other=0.0)
        # The encoder token tile is also fixed across the latent loop, so in
        # the full-panel score case we can hoist it once here as well.
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

            row_max, row_sum, y_num = _stablemax_output_apply_tile(
                dec_scores_log2,
                enc_scores,
                m_offsets,
                mask_m,
                Z_ENC_ptr,
                PrefixDen_ptr,
                PrefixNum_ptr,
                stride_zenc_b,
                stride_zenc_h,
                stride_zenc_n,
                stride_zenc_m,
                stride_pden_bh,
                stride_pden_nc,
                stride_pden_m,
                stride_pnum_bh,
                stride_pnum_nc,
                stride_pnum_m,
                stride_pnum_d,
                pid_b,
                pid_h,
                pid_bh,
                pid_nc,
                pid_d,
                global_t,
                mask_t,
                d_offsets,
                mask_d,
                causal_mask,
                v_tile,
                row_max,
                row_sum,
                y_num,
                scale,
                power,
                INPUT_PRECISION=INPUT_PRECISION,
                POWER_MODE=POWER_MODE,
            )
    else:
        # Streamed fallback:
        # When the score head dimension is larger than `BLOCK_K`, keep the score
        # build simple and stream over the score dimension one `BLOCK_K` slice at
        # a time. The default config now chooses `BLOCK_K = D_SCORE` for the
        # supported score sizes {32, 64, 128, 256}, so this path is mainly a
        # fallback for any future non-full-panel regimes rather than the primary
        # hot path.
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

            row_max, row_sum, y_num = _stablemax_output_apply_tile(
                dec_scores_log2,
                enc_scores,
                m_offsets,
                mask_m,
                Z_ENC_ptr,
                PrefixDen_ptr,
                PrefixNum_ptr,
                stride_zenc_b,
                stride_zenc_h,
                stride_zenc_n,
                stride_zenc_m,
                stride_pden_bh,
                stride_pden_nc,
                stride_pden_m,
                stride_pnum_bh,
                stride_pnum_nc,
                stride_pnum_m,
                stride_pnum_d,
                pid_b,
                pid_h,
                pid_bh,
                pid_nc,
                pid_d,
                global_t,
                mask_t,
                d_offsets,
                mask_d,
                causal_mask,
                v_tile,
                row_max,
                row_sum,
                y_num,
                scale,
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


class FLAREAutoregressiveStablemaxTriton(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Q_dec, K_dec, scale=None, chunk_size=None, power: float = 2.0, input_precision=None):
        """Full Triton forward for chunked stablemax FLARE.

        This follows the same high-level structure as the semi-autoregressive
        forward path and keeps the launch sites directly in `forward` so the
        debugging path is short.

        Public tensor layouts
        ---------------------
        - `Q`:     [H, M, D_score]
        - `K`:     [B, N, H, D_score]
        - `V`:     [B, N, H, D_value]
        - `Q_dec`: [B, N, H, D_score]
        - `K_dec`: [H, M, D_score]

        High-level formula
        ------------------
        This computes the same object as the PyTorch stablemax reference:

            y_t = sum_m p_dec[t, m] *
                  (sum_{tau <= t} stablemax(a_enc[tau, m]) * v_tau) /
                  (sum_{tau <= t} stablemax(a_enc[tau, m]))

        where

            a_enc[tau, m] = scale * <k_tau, q_m>
            p_dec[t, :]   = softmax(scale * <q_dec_t, k_dec_*>)

        Forward phases
        --------------
        The implementation is split into three Triton phases:

        1. `stablemax_prepare_kernel`
           Build per-chunk encoder denominator and numerator summaries.
        2. `stablemax_prefix_scan_kernel`
           Build strict chunk prefixes for denominator and numerator state.
        3. `stablemax_output_kernel`
           Recompute the encoder replay state chunk-at-a-time and finish the
           decoder online-softmax/output accumulation in one fused kernel.
        """
        B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="Stablemax Triton FLARE")
        scale_f = _resolve_attn_scale(scale, D_score)
        power_f = float(power)
        power_mode = _stablemax_power_mode(power_f)
        Q_dec_resolved, K_dec_resolved, _, _, _ = _resolve_flare_causal_decode_inputs(Q, K, Q_dec, K_dec)
        chunk_size_resolved = _resolve_stablemax_chunk_size(N, M, D_score, chunk_size)

        # The current Triton path works chunk-at-a-time and keeps the full chunk
        # token axis live inside phase 3, so we require the resolved chunk size
        # to stay aligned with Triton's tensor-core-friendly tile sizes.
        if chunk_size_resolved % 16 != 0:
            raise ValueError(
                "FLAREAutoregressiveStablemaxTriton requires a chunk size divisible by 16. "
                f"Got chunk_size={chunk_size_resolved}."
            )

        PADDED_LEN = math.ceil(N / chunk_size_resolved) * chunk_size_resolved if N > 0 else 0
        out_dtype = V.dtype

        if N == 0:
            O_empty = torch.empty((B, 0, H, D_value), device=V.device, dtype=out_dtype)
            Z_ENC_empty = torch.empty((B, H, 0, M), device=V.device, dtype=torch.float32)
            Z_DEC_empty = torch.empty((B, H, 0), device=V.device, dtype=torch.float32)
            ctx.empty = True
            ctx.b = B
            ctx.n = N
            ctx.h = H
            ctx.m = M
            ctx.d_score = D_score
            ctx.d_value = D_value
            ctx.q_dtype = Q.dtype
            ctx.k_dtype = K.dtype
            ctx.v_dtype = V.dtype
            ctx.q_dec_dtype = Q_dec_resolved.dtype
            ctx.k_dec_dtype = K_dec_resolved.dtype
            ctx.mark_non_differentiable(Z_ENC_empty, Z_DEC_empty)
            return O_empty, Z_ENC_empty, Z_DEC_empty

        # Chunk geometry.
        #
        # The public tensors remain in their original layouts. Only the Triton
        # intermediates are stored in chunk-indexed form.
        num_chunks = PADDED_LEN // chunk_size_resolved

        config = _stablemax_forward_config(
            M=M,
            D_score=D_score,
            D_value=D_value,
            chunk_size=chunk_size_resolved,
            input_precision=input_precision,
        )
        ctx.empty = False
        ctx.b = B
        ctx.n = N
        ctx.h = H
        ctx.m = M
        ctx.d_score = D_score
        ctx.d_value = D_value
        ctx.chunk_size = chunk_size_resolved
        ctx.num_chunks = num_chunks
        ctx.scale = float(scale_f)
        ctx.power = power_f
        ctx.power_mode = power_mode
        ctx.input_precision = config["input_precision"]
        ctx.q_dtype = Q.dtype
        ctx.k_dtype = K.dtype
        ctx.v_dtype = V.dtype
        ctx.q_dec_dtype = Q_dec_resolved.dtype
        ctx.k_dec_dtype = K_dec_resolved.dtype
        ctx.config_block_m = config["BLOCK_M"]
        ctx.config_block_d = config["BLOCK_D"]
        ctx.config_block_k = config["BLOCK_K"]
        ctx.prepare_num_warps = config["prepare_num_warps"]
        ctx.prepare_num_stages = config["prepare_num_stages"]
        ctx.prefix_num_warps = config["prefix_num_warps"]
        ctx.prefix_num_stages = config["prefix_num_stages"]
        ctx.output_num_warps = config["output_num_warps"]
        ctx.output_num_stages = config["output_num_stages"]

        # Triton-owned encoder-side intermediates:
        # - `chunk_den`: per-chunk denominator summaries, kept in fp32
        # - `chunk_num`: per-chunk numerator summaries, stored in `V.dtype`
        # - `Z_ENC`:     token-wise encoder score sums, `[B, H, N, M]`, fp32
        # - `Z_DEC`:     decoder log-sum-exp, written by the fused output phase
        #
        # The per-token encoder stable weights are no longer materialized here;
        # phase 3 recomputes them directly from `Q` and `K`.
        chunk_den = torch.empty((B * H, num_chunks, M), device=Q.device, dtype=torch.float32)
        chunk_num = torch.empty((B * H, num_chunks, M, D_value), device=Q.device, dtype=V.dtype)
        Z_ENC = torch.empty((B, H, N, M), device=Q.device, dtype=torch.float32)
        Z_DEC = torch.empty((B, H, N), device=Q.device, dtype=torch.float32)

        # Phase 1 launch:
        #   one program per (batch-head, chunk, latent tile, value tile)
        #
        # This directly mirrors the PyTorch reference's per-chunk encoder
        # summary build. The per-token stable weights are reduced immediately
        # into `score_chunk_den` and `score_chunk_num` instead of being stored
        # for later reuse.
        def prepare_grid(_meta):
            return (B * H, num_chunks, config["NUM_M_TILES"] * config["NUM_D_TILES"])

        stablemax_prepare_kernel[prepare_grid](
            Q,
            K,
            V,
            chunk_den,
            chunk_num,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *chunk_den.stride(),
            *chunk_num.stride(),
            N,
            M,
            D_SCORE=D_score,
            D_VALUE=D_value,
            CHUNK_SIZE=chunk_size_resolved,
            scale=scale_f,
            power=power_f,
            POWER_MODE=power_mode,
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            INPUT_PRECISION=config["input_precision"],
            H=H,
            num_warps=config["prepare_num_warps"],
            num_stages=config["prepare_num_stages"],
        )

        # Strict chunk prefixes for the encoder summaries. These correspond to
        # `score_prev_den` and `score_prev_num` in the PyTorch implementation.
        #
        # The denominator prefix stays in fp32. The numerator prefix is stored
        # in `V.dtype`, but each prefix-scan program accumulates it in fp32.
        prefix_den = torch.empty((B * H, num_chunks, M), device=Q.device, dtype=torch.float32)
        prefix_num = torch.empty((B * H, num_chunks, M, D_value), device=Q.device, dtype=V.dtype)

        # Phase 2 launch:
        #   one program per (batch-head, latent tile, value tile)
        #
        # Each program scans left-to-right over the chunk axis and stores the
        # strict prefix state seen by each chunk.
        def prefix_grid(_meta):
            return (B * H, config["NUM_M_TILES"] * config["NUM_D_TILES"])

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
            D_VALUE=D_value,
            NUM_CHUNKS=num_chunks,
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            num_warps=config["prefix_num_warps"],
            num_stages=config["prefix_num_stages"],
        )

        # Public output buffer in [B, N, H, D_value] layout.
        #
        # The fused output kernel owns the full latent reduction locally, so it
        # writes each output tile directly with no intermediate zero-fill or
        # atomic accumulation.
        O = torch.empty((B, N, H, D_value), device=Q.device, dtype=torch.float32)

        # Phase 3 launch:
        #   one program per (batch-head, chunk, value tile)
        #
        # This fused kernel keeps the decoder softmax state online across the
        # latent tiles, writes `Z_DEC`, reconstructs `Z_ENC`, and stores the
        # final output directly.
        def output_grid(_meta):
            return (B * H, num_chunks, config["NUM_D_TILES"])

        stablemax_output_kernel[output_grid](
            Q,
            K,
            Q_dec_resolved,
            K_dec_resolved,
            Z_ENC,
            Z_DEC,
            prefix_den,
            prefix_num,
            V,
            O,
            *Q.stride(),
            *K.stride(),
            *Q_dec_resolved.stride(),
            *K_dec_resolved.stride(),
            *Z_ENC.stride(),
            *Z_DEC.stride(),
            *prefix_den.stride(),
            *prefix_num.stride(),
            *V.stride(),
            *O.stride(),
            N,
            M,
            D_SCORE=D_score,
            D_VALUE=D_value,
            CHUNK_SIZE=chunk_size_resolved,
            scale=scale_f,
            power=power_f,
            POWER_MODE=power_mode,
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            INPUT_PRECISION=config["input_precision"],
            H=H,
            num_warps=config["output_num_warps"],
            num_stages=config["output_num_stages"],
        )
        O_out = O.to(out_dtype)
        ctx.save_for_backward(
            Q,
            K,
            V,
            Q_dec_resolved,
            K_dec_resolved,
            prefix_den,
            prefix_num,
            O,
            Z_ENC,
            Z_DEC,
        )
        ctx.mark_non_differentiable(Z_ENC, Z_DEC)

        # All internal reductions accumulate in fp32. Only the public return
        # value is cast back to the caller-visible output dtype.
        return O_out, Z_ENC, Z_DEC

    @staticmethod
    def backward(ctx, dY, *unused):
        if getattr(ctx, "empty", False):
            B = ctx.b
            N = ctx.n
            H = ctx.h
            M = ctx.m
            D_score = ctx.d_score
            D_value = ctx.d_value
            device = dY.device
            return (
                torch.zeros((H, M, D_score), device=device, dtype=ctx.q_dtype),
                torch.zeros((B, N, H, D_score), device=device, dtype=ctx.k_dtype),
                torch.zeros((B, N, H, D_value), device=device, dtype=ctx.v_dtype),
                torch.zeros((B, N, H, D_score), device=device, dtype=ctx.q_dec_dtype),
                torch.zeros((H, M, D_score), device=device, dtype=ctx.k_dec_dtype),
                None,
                None,
                None,
                None,
            )

        (
            Q,
            K,
            V,
            Q_dec_resolved,
            K_dec_resolved,
            prefix_den,
            prefix_num,
            O,
            Z_ENC,
            Z_DEC,
        ) = ctx.saved_tensors

        B = ctx.b
        N = ctx.n
        H = ctx.h
        M = ctx.m
        D_score = ctx.d_score
        D_value = ctx.d_value
        chunk_size = ctx.chunk_size
        num_chunks = ctx.num_chunks

        # Backward mirrors the PyTorch reference in four stages:
        #
        # 1. preprocess `delta[t] = <O[t], dO[t]>` for decoder-softmax backward
        # 2. phase-3/output adjoints:
        #    - propagate through decoder probabilities and replay to get
        #      `dQ_dec`, `dK_dec`, the phase-3 contribution to `dV`, and strict
        #      prefix-summary adjoints for `prefix_den/prefix_num`
        # 3. reverse-scan those strict-prefix adjoints over chunks
        # 4. encoder adjoints:
        #    - combine phase-3 and summary-path gradients
        #    - propagate through stablemax and encoder score projections to get
        #      `dQ`, `dK`, and the summary contribution to `dV`
        #
        # `Z_ENC` / `Z_DEC` were already saved in forward, so backward can stay
        # faithful to the PyTorch algorithm without re-deriving those
        # normalizers on the host.

        # Backward has different pressure points than forward. The hot chunk-owned
        # kernels benefit from owning a much larger latent tile so they amortize
        # the `[C, C]` replay math across fewer `m` iterations.
        if M >= 64:
            bwd_block_m = 64
        elif M >= 32:
            bwd_block_m = 32
        elif M >= 16:
            bwd_block_m = 16
        else:
            bwd_block_m = ctx.config_block_m
        config = {
            "BLOCK_M": bwd_block_m,
            "BLOCK_D": ctx.config_block_d,
            "BLOCK_K": ctx.config_block_k,
            "NUM_M_TILES": triton.cdiv(M, bwd_block_m),
            "NUM_D_TILES": triton.cdiv(D_value, ctx.config_block_d),
            "input_precision": ctx.input_precision,
            "output_num_warps": ctx.output_num_warps,
            "output_num_stages": ctx.output_num_stages,
        }
        bwd_num_warps = 8
        bwd_num_stages = 2

        if config["NUM_D_TILES"] != 1:
            raise NotImplementedError(
                "FLAREAutoregressiveStablemaxTriton backward currently requires D_value <= BLOCK_D "
                "so the fused value path owns the full value tile."
            )
        if D_score > config["BLOCK_K"]:
            raise NotImplementedError(
                "FLAREAutoregressiveStablemaxTriton backward currently requires the full-panel score regime "
                "(D_score <= BLOCK_K)."
            )

        dO = dY.contiguous().to(torch.float32)
        delta = torch.empty((B * H, N), device=dY.device, dtype=torch.float32)
        dQ = torch.zeros((H, M, D_score), device=dY.device, dtype=torch.float32)
        dK = torch.zeros((B, N, H, D_score), device=dY.device, dtype=torch.float32)
        dV = torch.zeros((B, N, H, D_value), device=dY.device, dtype=torch.float32)
        dQ_dec = torch.zeros((B, N, H, D_score), device=dY.device, dtype=torch.float32)
        dK_dec = torch.zeros((H, M, D_score), device=dY.device, dtype=torch.float32)

        # Saved chunk-local encoder quantities reused between the two hot
        # backward kernels. Caching them here is cheaper than recomputing the
        # same encoder transform and normalization in both kernels.
        stable_buf = torch.empty((B, H, N, M), device=dY.device, dtype=V.dtype)
        decode_over_den_buf = torch.empty((B, H, N, M), device=dY.device, dtype=V.dtype)
        grad_prev_den = torch.empty((B * H, num_chunks, M), device=dY.device, dtype=torch.float32)
        # These chunk-summary adjoints are large HBM workspaces. Keeping them in
        # `V.dtype` preserves the fast low-precision tensor-core path in the
        # producer/consumer kernels while all reductions still accumulate in
        # fp32 inside the kernels themselves.
        grad_prev_num = torch.empty((B * H, num_chunks, M, D_value), device=dY.device, dtype=V.dtype)
        grad_chunk_den = torch.empty_like(grad_prev_den)
        grad_chunk_num = torch.empty_like(grad_prev_num)

        def delta_grid(_meta):
            return (B * H, num_chunks)

        # Stage 1: row-wise decoder-softmax preprocess.
        stablemax_bwd_preprocess_delta_kernel[delta_grid](
            O,
            dO,
            delta,
            *O.stride(),
            *dO.stride(),
            *delta.stride(),
            N,
            D_VALUE=D_value,
            CHUNK_SIZE=chunk_size,
            BLOCK_D=config["BLOCK_D"],
            H=H,
            num_warps=4,
            num_stages=2,
        )

        def output_bwd_grid(_meta):
            return (B * H, num_chunks)

        # Stage 2: output/decoder adjoints plus cached encoder-side state.
        stablemax_bwd_output_kernel[output_bwd_grid](
            Q,
            K,
            V,
            Q_dec_resolved,
            K_dec_resolved,
            prefix_den,
            prefix_num,
            Z_ENC,
            Z_DEC,
            dO,
            delta,
            dQ_dec,
            dK_dec,
            dV,
            stable_buf,
            decode_over_den_buf,
            grad_prev_den,
            grad_prev_num,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *Q_dec_resolved.stride(),
            *K_dec_resolved.stride(),
            *prefix_den.stride(),
            *prefix_num.stride(),
            *Z_ENC.stride(),
            *Z_DEC.stride(),
            *dO.stride(),
            *delta.stride(),
            *dQ_dec.stride(),
            *dK_dec.stride(),
            *dV.stride(),
            *stable_buf.stride(),
            *decode_over_den_buf.stride(),
            *grad_prev_den.stride(),
            *grad_prev_num.stride(),
            N,
            M,
            D_SCORE=D_score,
            D_VALUE=D_value,
            CHUNK_SIZE=chunk_size,
            scale=ctx.scale,
            power=ctx.power,
            POWER_MODE=ctx.power_mode,
            BLOCK_M=config["BLOCK_M"],
            BLOCK_K=config["BLOCK_K"],
            INPUT_PRECISION=config["input_precision"],
            H=H,
            num_warps=bwd_num_warps,
            num_stages=bwd_num_stages,
        )

        def reverse_den_grid(_meta):
            return (B * H, config["NUM_M_TILES"])

        # Stage 3: reverse strict-prefix scans over chunk summaries.
        stablemax_bwd_reverse_scan_den_kernel[reverse_den_grid](
            grad_prev_den,
            grad_chunk_den,
            *grad_prev_den.stride(),
            *grad_chunk_den.stride(),
            M,
            NUM_CHUNKS=num_chunks,
            BLOCK_M=config["BLOCK_M"],
            num_warps=4,
            num_stages=2,
        )

        stablemax_bwd_reverse_scan_num_kernel[reverse_den_grid](
            grad_prev_num,
            grad_chunk_num,
            *grad_prev_num.stride(),
            *grad_chunk_num.stride(),
            M,
            D_VALUE=D_value,
            NUM_CHUNKS=num_chunks,
            BLOCK_M=config["BLOCK_M"],
            num_warps=4,
            num_stages=2,
        )

        # Stage 4: encoder adjoints plus summary-path contribution to `dV`.
        stablemax_bwd_encoder_kernel[output_bwd_grid](
            Q,
            K,
            V,
            Q_dec_resolved,
            K_dec_resolved,
            prefix_den,
            prefix_num,
            Z_ENC,
            Z_DEC,
            dO,
            stable_buf,
            decode_over_den_buf,
            grad_chunk_den,
            grad_chunk_num,
            dQ,
            dK,
            dV,
            *Q.stride(),
            *K.stride(),
            *V.stride(),
            *Q_dec_resolved.stride(),
            *K_dec_resolved.stride(),
            *prefix_den.stride(),
            *prefix_num.stride(),
            *Z_ENC.stride(),
            *Z_DEC.stride(),
            *dO.stride(),
            *stable_buf.stride(),
            *decode_over_den_buf.stride(),
            *grad_chunk_den.stride(),
            *grad_chunk_num.stride(),
            *dQ.stride(),
            *dK.stride(),
            *dV.stride(),
            N,
            M,
            D_SCORE=D_score,
            D_VALUE=D_value,
            CHUNK_SIZE=chunk_size,
            scale=ctx.scale,
            power=ctx.power,
            POWER_MODE=ctx.power_mode,
            BLOCK_M=config["BLOCK_M"],
            BLOCK_K=config["BLOCK_K"],
            INPUT_PRECISION=config["input_precision"],
            H=H,
            num_warps=bwd_num_warps,
            num_stages=bwd_num_stages,
        )

        return (
            dQ.to(ctx.q_dtype),
            dK.to(ctx.k_dtype),
            dV.to(ctx.v_dtype),
            dQ_dec.to(ctx.q_dec_dtype),
            dK_dec.to(ctx.k_dec_dtype),
            None,
            None,
            None,
            None,
        )


def flare_autoregressive_stablemax_triton(
    Q,
    K,
    V,
    scale=None,
    eps=None,
    profile: bool = False,
    chunk_size=None,
    Q_dec=None,
    K_dec=None,
    power: float = 2.0,
    write_gate: bool = False,
    write_gate_fixed_value: float | None = None,
    write_gate_tensor=None,
    state: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    return_state: bool = False,
    input_precision=None,
):
    if eps is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_triton does not use eps.")
    if profile:
        raise NotImplementedError("flare_autoregressive_stablemax_triton does not support profile=True")
    if state is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_triton does not support recurrent state")
    if attention_mask is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_triton does not support attention_mask")
    if return_state:
        raise NotImplementedError("flare_autoregressive_stablemax_triton does not support return_state=True")
    if write_gate or write_gate_fixed_value is not None or write_gate_tensor is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_triton currently implements the non-write-gated path only.")
    Q_dec_resolved, K_dec_resolved, _, _, _ = _resolve_flare_causal_decode_inputs(Q, K, Q_dec, K_dec)
    return FLAREAutoregressiveStablemaxTriton.apply(
        Q,
        K,
        V,
        Q_dec_resolved,
        K_dec_resolved,
        scale,
        chunk_size,
        power,
        input_precision,
    )

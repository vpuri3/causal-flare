from ._common import *
from .torch import _resolve_flare_causal_decode_inputs as _resolve_recurrent_decode_inputs

# NOTE:
# - This file does not implement sequence-length (N-axis) chunking.
# - It is used to prototype inner methods that may later be used inside
#   chunked algorithms.
# - Keep this module self-contained: do not import kernels from chunked.py.


def _get_recurrent_block_d_k(D: int, block_d=None) -> tuple[int, int]:
    block_d_env = os.environ.get("FLARE_RECURRENT_BLOCK_D", "").strip()
    if block_d is not None:
        block_d_val = int(block_d)
    elif block_d_env:
        block_d_val = int(block_d_env)
    else:
        block_d_val = D if D <= 128 else 64
    block_d_val = min(block_d_val, D)
    if block_d_val % 16 != 0:
        raise ValueError(f"Recurrent BLOCK_D must be a multiple of 16. Got BLOCK_D={block_d_val}")

    block_k_env = os.environ.get("FLARE_RECURRENT_BLOCK_K", "").strip()
    if block_k_env:
        block_k_val = int(block_k_env)
    else:
        # Recurrent forward benchmarks favored a 64-wide reduction tile once
        # D-blocking is active: it kept most of the footprint win while cutting
        # the repeated score-reduction loop count materially.
        block_k_val = D if block_d_val == D else 64
    block_k_val = min(block_k_val, D)
    if block_k_val % 16 != 0:
        raise ValueError(f"Recurrent BLOCK_K must be a multiple of 16. Got BLOCK_K={block_k_val}")
    return block_d_val, block_k_val


def _get_recurrent_block_t(block_t=None) -> int:
    block_t_env = os.environ.get("FLARE_RECURRENT_BLOCK_T", "").strip()
    if block_t is not None:
        block_t_val = int(block_t)
    elif block_t_env:
        block_t_val = int(block_t_env)
    else:
        block_t_val = 16

    if block_t_val == 1:
        return 1
    if block_t_val % 16 != 0:
        raise ValueError(f"Recurrent BLOCK_T must be 1 or a multiple of 16. Got BLOCK_T={block_t_val}")
    if block_t_val not in (16, 32):
        raise ValueError(f"Recurrent BLOCK_T currently supports [1, 16, 32]. Got BLOCK_T={block_t_val}")
    return block_t_val


def _get_recurrent_block_m(M: int, block_m=None) -> int:
    block_m_env = os.environ.get("FLARE_RECURRENT_BLOCK_M", "").strip()
    if block_m is not None:
        block_m_val = int(block_m)
    elif block_m_env:
        block_m_val = int(block_m_env)
    else:
        block_m_val = M
    block_m_val = min(block_m_val, M)
    if block_m_val % 16 != 0:
        raise ValueError(f"Recurrent BLOCK_M must be a multiple of 16. Got BLOCK_M={block_m_val}")
    return block_m_val


@triton.jit
def flare_recurrent_decode_lse_kernel(
    Q_dec_ptr,
    K_dec_ptr,
    LSE_dec_ptr,
    stride_qdb, stride_qdh, stride_qdt, stride_qdd,
    stride_kdb, stride_kdh, stride_kdm, stride_kdd,
    stride_lsed_b, stride_lsed_h, stride_lsed_t,
    B, H, M, T,
    scale,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_tb = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H
    t_offsets = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = t_offsets < T

    a_max = tl.full([BLOCK_T], -float("inf"), tl.float32)
    a_sum = tl.zeros([BLOCK_T], tl.float32)
    m0 = 0
    while m0 < M:
        m_offsets = m0 + tl.arange(0, BLOCK_M)
        mask_m = m_offsets < M

        a_sub = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)
        for k0 in tl.static_range(0, D, BLOCK_K):
            d_offsets = k0 + tl.arange(0, BLOCK_K)
            mask_d = d_offsets < D

            q_dec_ptr = Q_dec_ptr + b * stride_qdb + h * stride_qdh + t_offsets[:, None] * stride_qdt + d_offsets[None, :] * stride_qdd
            q_dec = tl.load(q_dec_ptr, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            k_dec_ptr = K_dec_ptr + b * stride_kdb + h * stride_kdh + m_offsets[:, None] * stride_kdm + d_offsets[None, :] * stride_kdd
            k_dec = tl.load(k_dec_ptr, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            a_sub += tl.dot(k_dec, tl.trans(q_dec), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        a_sub = a_sub * scale
        a_sub = tl.where(mask_m[:, None] & mask_t[None, :], a_sub, -float("inf"))

        block_max = tl.max(a_sub, axis=0)
        new_max = tl.maximum(a_max, block_max)
        block_exp = tl.exp(a_sub - new_max[None, :])
        block_exp = tl.where(mask_m[:, None] & mask_t[None, :], block_exp, 0.0)
        a_sum = a_sum * tl.exp(a_max - new_max) + tl.sum(block_exp, axis=0)
        a_max = new_max
        m0 += BLOCK_M

    lse_dec = a_max + tl.log(a_sum + 1e-20)
    lse_dec_ptr = LSE_dec_ptr + b * stride_lsed_b + h * stride_lsed_h + t_offsets * stride_lsed_t
    tl.store(lse_dec_ptr, lse_dec, mask=mask_t)

@triton.jit
def flare_recurrent_orig_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Q_dec_ptr,
    K_dec_ptr,
    LSE_dec_ptr,
    Y_ptr,
    LSE_enc_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_qdb, stride_qdh, stride_qdt, stride_qdd,
    stride_kdb, stride_kdh, stride_kdm, stride_kdd,
    stride_lsed_b, stride_lsed_h, stride_lsed_t,
    stride_yb, stride_yh, stride_yt, stride_yd,
    stride_lsee_b, stride_lsee_h, stride_lsee_t, stride_lsee_m,
    B, H, M,
    T,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WEIGHT_SHARING_ENC_DEC: tl.constexpr,
    SINGLE_M_TILE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    m_state = tl.full([BLOCK_M], -float("inf"), tl.float32)
    d_state = tl.zeros([BLOCK_M], tl.float32)
    u_state = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    tl.static_assert(BLOCK_D > 0)
    tl.static_assert(BLOCK_M > 0)

    t = 0
    while t < T:
        s_t = tl.zeros([BLOCK_M], tl.float32)
        for k0 in tl.static_range(0, D, BLOCK_K):
            d_offsets_k = k0 + tl.arange(0, BLOCK_K)
            mask_d_k = d_offsets_k < D
            q_ptr = q_base + m_offsets[:, None] * stride_qm + d_offsets_k[None, :] * stride_qd
            q_tile = tl.load(q_ptr, mask=mask_m[:, None] & mask_d_k[None, :], other=0.0).to(tl.float32)
            k_ptr = k_base + t * stride_kt
            k_tile = tl.load(k_ptr + d_offsets_k * stride_kd, mask=mask_d_k, other=0.0).to(tl.float32)
            s_t += tl.sum(q_tile * k_tile[None, :], axis=1)

        s_t *= scale
        s_t = tl.where(mask_m, s_t, -float("inf"))

        m_new = tl.maximum(m_state, s_t)
        gamma = tl.exp(m_state - m_new)
        eta = tl.exp(s_t - m_new)
        d_state = d_state * gamma + eta

        v_ptr = v_base + t * stride_vt
        v_block = tl.load(v_ptr + d_offsets * stride_vd, mask=mask_d, other=0.0).to(tl.float32)
        u_state = u_state * gamma[:, None] + eta[:, None] * v_block[None, :]

        inv_d = 1.0 / tl.where(d_state > 0, d_state, 1.0)
        z_block = u_state * inv_d[:, None]

        if WEIGHT_SHARING_ENC_DEC:
            a_t = s_t
        else:
            a_t = tl.zeros([BLOCK_M], tl.float32)
            for k0 in tl.static_range(0, D, BLOCK_K):
                d_offsets_k = k0 + tl.arange(0, BLOCK_K)
                mask_d_k = d_offsets_k < D
                q_dec_ptr = Q_dec_ptr + b * stride_qdb + h * stride_qdh + t * stride_qdt + d_offsets_k * stride_qdd
                q_dec_tile = tl.load(q_dec_ptr, mask=mask_d_k, other=0.0).to(tl.float32)
                k_dec_ptr = K_dec_ptr + b * stride_kdb + h * stride_kdh + m_offsets[:, None] * stride_kdm + d_offsets_k[None, :] * stride_kdd
                k_dec_tile = tl.load(k_dec_ptr, mask=mask_m[:, None] & mask_d_k[None, :], other=0.0).to(tl.float32)
                a_t += tl.sum(k_dec_tile * q_dec_tile[None, :], axis=1)
            a_t = tl.where(mask_m, a_t * scale, -float("inf"))

        lse_dec_ptr = LSE_dec_ptr + b * stride_lsed_b + h * stride_lsed_h + t * stride_lsed_t
        lse_dec_t = tl.load(lse_dec_ptr).to(tl.float32)

        exp_a = tl.exp(a_t - lse_dec_t)
        exp_a = tl.where(mask_m, exp_a, 0.0)
        y_block = tl.sum(exp_a[:, None] * z_block, axis=0)
        y_ptr = y_base + t * stride_yt
        if SINGLE_M_TILE:
            tl.store(y_ptr + d_offsets * stride_yd, y_block.to(tl.float32), mask=mask_d)
        else:
            tl.atomic_add(y_ptr + d_offsets * stride_yd, y_block.to(tl.float32), mask=mask_d)
        lse_enc_ptr = LSE_enc_ptr + b * stride_lsee_b + h * stride_lsee_h + t * stride_lsee_t
        tl.store(lse_enc_ptr + m_offsets * stride_lsee_m, m_new + tl.log(tl.maximum(d_state, 1e-20)), mask=(pid_d == 0) & mask_m)

        m_state = m_new
        t += 1

class RecurrentFLARE(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale=None, block_m=None, block_d=None, Q_dec=None, K_dec=None, block_t=None):
        if Q.dim() != 3 or K.dim() != 4 or V.dim() != 4:
            raise ValueError(
                "Recurrent FLARE expects Q [H, M, D] and K/V [B, N, H, D]. "
                f"Got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
            )
        if K.size() != V.size():
            raise ValueError(f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}")
        if Q.size(0) != K.size(2) or Q.size(2) != K.size(3):
            raise ValueError(
                "Expected Q [H, M, D] and K/V [B, N, H, D] with matching H,D. "
                f"Got Q.shape={Q.shape} and K.shape={K.shape}"
            )

        H, M, D = Q.size()
        B, T, _, _ = K.size()
        scale = _resolve_attn_scale(scale, D)
        Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_recurrent_decode_inputs(
            Q, K, Q_dec, K_dec
        )
        if (M % 16) != 0 or (D % 16) != 0:
            raise ValueError(f"RecurrentFLARE requires M and D be multiples of 16. Got M={M}, D={D}")
        K_bhtd = K.permute(0, 2, 1, 3).contiguous()
        V_bhtd = V.permute(0, 2, 1, 3).contiguous()
        Q_dec_bhtd = Q_dec.permute(0, 2, 1, 3).contiguous() if separate_Q_dec else K_bhtd
        K_dec_bhmd = K_dec.unsqueeze(0).expand(B, -1, -1, -1) if separate_K_dec else Q.unsqueeze(0).expand(B, -1, -1, -1)

        block_m = _get_recurrent_block_m(M, block_m)
        block_d, block_k = _get_recurrent_block_d_k(D, block_d)
        block_t = _get_recurrent_block_t(block_t)
        num_m_tiles = triton.cdiv(M, block_m)

        # Two logsumexp trajectories:
        #   LSE_enc [B, H, T, M]: encoder recurrent normalization per latent query m.
        #   LSE_dec [B, H, T]: decoder normalization over M at each token t.
        if num_m_tiles == 1:
            Y = torch.empty((B, H, T, D), device=Q.device, dtype=Q.dtype)
        else:
            Y = torch.zeros((B, H, T, D), device=Q.device, dtype=Q.dtype)
        LSE_enc = torch.empty((B, H, T, M), device=Q.device, dtype=torch.float32)
        LSE_dec = torch.empty((B, H, T), device=Q.device, dtype=torch.float32)
        grid = (B * H, triton.cdiv(D, block_d), num_m_tiles)

        input_precision = _get_input_precision()
        num_warps = 4 if block_m <= 64 else 8
        flare_recurrent_decode_lse_kernel[(B * H, triton.cdiv(T, 16))](
            Q_dec_bhtd, K_dec_bhmd, LSE_dec,
            Q_dec_bhtd.stride(0), Q_dec_bhtd.stride(1), Q_dec_bhtd.stride(2), Q_dec_bhtd.stride(3),
            K_dec_bhmd.stride(0), K_dec_bhmd.stride(1), K_dec_bhmd.stride(2), K_dec_bhmd.stride(3),
            LSE_dec.stride(0), LSE_dec.stride(1), LSE_dec.stride(2),
            B, H, M, T,
            scale,
            D=D,
            BLOCK_T=16,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            INPUT_PRECISION=input_precision,
            num_warps=num_warps,
            num_stages=2,
        )
        if block_t == 1:
            # Naive token loop path.
            flare_recurrent_orig_fwd_kernel[grid](
                Q, K_bhtd, V_bhtd, Q_dec_bhtd, K_dec_bhmd, LSE_dec, Y, LSE_enc,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K_bhtd.stride(0), K_bhtd.stride(1), K_bhtd.stride(2), K_bhtd.stride(3),
                V_bhtd.stride(0), V_bhtd.stride(1), V_bhtd.stride(2), V_bhtd.stride(3),
                Q_dec_bhtd.stride(0), Q_dec_bhtd.stride(1), Q_dec_bhtd.stride(2), Q_dec_bhtd.stride(3),
                K_dec_bhmd.stride(0), K_dec_bhmd.stride(1), K_dec_bhmd.stride(2), K_dec_bhmd.stride(3),
                LSE_dec.stride(0), LSE_dec.stride(1), LSE_dec.stride(2),
                Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
                LSE_enc.stride(0), LSE_enc.stride(1), LSE_enc.stride(2), LSE_enc.stride(3),
                B, H, M,
                T,
                scale,
                D=D,
                BLOCK_M=block_m,
                BLOCK_D=block_d,
                BLOCK_K=block_k,
                WEIGHT_SHARING_ENC_DEC=weight_sharing_enc_dec,
                SINGLE_M_TILE=num_m_tiles == 1,
                num_warps=num_warps,
                num_stages=2,
            )
        else:
            flare_recurrent_fwd_kernel[grid](
                Q, K_bhtd, V_bhtd, Q_dec_bhtd, K_dec_bhmd, LSE_dec, Y, LSE_enc,
                Q.stride(0), Q.stride(1), Q.stride(2),
                K_bhtd.stride(0), K_bhtd.stride(1), K_bhtd.stride(2), K_bhtd.stride(3),
                V_bhtd.stride(0), V_bhtd.stride(1), V_bhtd.stride(2), V_bhtd.stride(3),
                Q_dec_bhtd.stride(0), Q_dec_bhtd.stride(1), Q_dec_bhtd.stride(2), Q_dec_bhtd.stride(3),
                K_dec_bhmd.stride(0), K_dec_bhmd.stride(1), K_dec_bhmd.stride(2), K_dec_bhmd.stride(3),
                LSE_dec.stride(0), LSE_dec.stride(1), LSE_dec.stride(2),
                Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
                LSE_enc.stride(0), LSE_enc.stride(1), LSE_enc.stride(2), LSE_enc.stride(3),
                B, H, M,
                T,
                scale,
                D=D,
                BLOCK_M=block_m,
                BLOCK_D=block_d,
                BLOCK_K=block_k,
                BLOCK_T=block_t,
                INPUT_PRECISION=input_precision,
                WEIGHT_SHARING_ENC_DEC=weight_sharing_enc_dec,
                SINGLE_M_TILE=num_m_tiles == 1,
                num_warps=num_warps,
                num_stages=2,
            )
        saved_tensors = [Q, K, V, LSE_enc, LSE_dec]
        if separate_Q_dec:
            saved_tensors.append(Q_dec)
        if separate_K_dec:
            saved_tensors.append(K_dec)
        ctx.save_for_backward(*saved_tensors)
        ctx.scale = scale
        ctx.weight_sharing_enc_dec = weight_sharing_enc_dec
        ctx.separate_Q_dec = separate_Q_dec
        ctx.separate_K_dec = separate_K_dec
        ctx.block_t = block_t
        return Y

    @staticmethod
    def backward(ctx, *grad_outputs):
        return _recurrent_flare_backward_impl(ctx, grad_outputs[0])


@triton.jit
def _recurrent_score_vec(
    Q_ptr,
    K_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    pid_b, pid_h, token_idx,
    m_offsets, mask_m,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    s_t = tl.zeros([BLOCK_M], tl.float32)
    for k0 in tl.static_range(0, D, BLOCK_K):
        d_offsets_k = k0 + tl.arange(0, BLOCK_K)
        mask_d_k = d_offsets_k < D
        q_ptr = Q_ptr + pid_h * stride_qh + m_offsets[:, None] * stride_qm + d_offsets_k[None, :] * stride_qd
        q_tile = tl.load(q_ptr, mask=mask_m[:, None] & mask_d_k[None, :], other=0.0).to(tl.float32)
        k_ptr = K_ptr + pid_b * stride_kb + token_idx * stride_kt + pid_h * stride_kh + d_offsets_k * stride_kd
        k_tile = tl.load(k_ptr, mask=mask_d_k, other=0.0).to(tl.float32)
        s_t += tl.sum(q_tile * k_tile[None, :], axis=1)
    return tl.where(mask_m, s_t * scale, -float("inf"))


@triton.jit
def _recurrent_decode_score_vec(
    Q_dec_ptr,
    K_dec_ptr,
    stride_qdb, stride_qdt, stride_qdh, stride_qdd,
    stride_kdh, stride_kdm, stride_kdd,
    pid_b, pid_h, token_idx,
    m_offsets, mask_m,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    a_t = tl.zeros([BLOCK_M], tl.float32)
    for k0 in tl.static_range(0, D, BLOCK_K):
        d_offsets_k = k0 + tl.arange(0, BLOCK_K)
        mask_d_k = d_offsets_k < D
        q_ptr = Q_dec_ptr + pid_b * stride_qdb + token_idx * stride_qdt + pid_h * stride_qdh + d_offsets_k * stride_qdd
        q_tile = tl.load(q_ptr, mask=mask_d_k, other=0.0).to(tl.float32)
        k_ptr = K_dec_ptr + pid_h * stride_kdh + m_offsets[:, None] * stride_kdm + d_offsets_k[None, :] * stride_kdd
        k_tile = tl.load(k_ptr, mask=mask_m[:, None] & mask_d_k[None, :], other=0.0).to(tl.float32)
        a_t += tl.sum(k_tile * q_tile[None, :], axis=1)
    return tl.where(mask_m, a_t * scale, -float("inf"))


@triton.jit
def _recurrent_dy_v_dot(
    V_ptr,
    dO_ptr,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_do_b, stride_do_h, stride_do_t, stride_do_d,
    pid_b, pid_h, tau_idx, t_idx,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    acc = 0.0
    for d0 in tl.static_range(0, D, BLOCK_D):
        d_offsets = d0 + tl.arange(0, BLOCK_D)
        mask_d = d_offsets < D
        v_ptr = V_ptr + pid_b * stride_vb + tau_idx * stride_vt + pid_h * stride_vh + d_offsets * stride_vd
        v_tile = tl.load(v_ptr, mask=mask_d, other=0.0).to(tl.float32)
        dO_ptr_t = dO_ptr + pid_b * stride_do_b + pid_h * stride_do_h + t_idx * stride_do_t + d_offsets * stride_do_d
        dO_tile = tl.load(dO_ptr_t, mask=mask_d, other=0.0).to(tl.float32)
        acc += tl.sum(v_tile * dO_tile, axis=0)
    return acc


@triton.jit
def flare_recurrent_bwd_dg_part_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dO_ptr,
    DG_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_do_b, stride_do_h, stride_do_t, stride_do_d,
    stride_dg_bh, stride_dg_t, stride_dg_m,
    H, M, T,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    m_run = tl.full([BLOCK_M], -float("inf"), tl.float32)
    d_run = tl.zeros([BLOCK_M], tl.float32)
    z_run = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    t0 = 0
    while t0 < T:
        for j in tl.static_range(0, BLOCK_T):
            t = t0 + j
            if t < T:
                s_t = _recurrent_score_vec(
                    Q_ptr,
                    K_ptr,
                    stride_qh, stride_qm, stride_qd,
                    stride_kb, stride_kt, stride_kh, stride_kd,
                    pid_b, pid_h, t,
                    m_offsets, mask_m,
                    scale,
                    D=D,
                    BLOCK_M=BLOCK_M,
                    BLOCK_K=BLOCK_K,
                )

                m_new = tl.maximum(m_run, s_t)
                alpha = tl.exp(m_run - m_new)
                beta = tl.exp(s_t - m_new)
                d_new = alpha * d_run + beta
                coeff_old = tl.where(d_new > 0, (alpha * d_run) / d_new, 0.0)
                coeff_new = tl.where(d_new > 0, beta / d_new, 0.0)

                v_ptr = V_ptr + pid_b * stride_vb + t * stride_vt + pid_h * stride_vh + d_offsets * stride_vd
                v_t = tl.load(v_ptr, mask=mask_d, other=0.0).to(tl.float32)
                z_run = coeff_old[:, None] * z_run + coeff_new[:, None] * v_t[None, :]

                dO_ptr_t = dO_ptr + pid_b * stride_do_b + pid_h * stride_do_h + t * stride_do_t + d_offsets * stride_do_d
                dO_t = tl.load(dO_ptr_t, mask=mask_d, other=0.0).to(tl.float32)
                dg_part = tl.sum(z_run * dO_t[None, :], axis=1)

                dg_ptr = DG_ptr + pid_bh * stride_dg_bh + t * stride_dg_t + m_offsets * stride_dg_m
                tl.atomic_add(dg_ptr, dg_part, mask=mask_m)

                m_run = m_new
                d_run = d_new
        t0 += BLOCK_T


@triton.jit
def flare_recurrent_bwd_dscore_decode_kernel(
    Q_dec_ptr,
    K_dec_ptr,
    LSE_dec_ptr,
    DG_ptr,
    DS_dec_ptr,
    stride_qdb, stride_qdt, stride_qdh, stride_qdd,
    stride_kdh, stride_kdm, stride_kdd,
    stride_lsed_b, stride_lsed_h, stride_lsed_t,
    stride_dg_bh, stride_dg_t, stride_dg_m,
    stride_ds_bh, stride_ds_t, stride_ds_m,
    H, M, T,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_tb = tl.program_id(1)

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    t0 = pid_tb * BLOCK_T
    for j in tl.static_range(0, BLOCK_T):
        t = t0 + j
        if t < T:
            a_t = _recurrent_decode_score_vec(
                Q_dec_ptr,
                K_dec_ptr,
                stride_qdb, stride_qdt, stride_qdh, stride_qdd,
                stride_kdh, stride_kdm, stride_kdd,
                pid_b, pid_h, t,
                m_offsets, mask_m,
                scale,
                D=D,
                BLOCK_M=BLOCK_M,
                BLOCK_K=BLOCK_K,
            )

            lse_dec_ptr = LSE_dec_ptr + pid_b * stride_lsed_b + pid_h * stride_lsed_h + t * stride_lsed_t
            lse_dec_t = tl.load(lse_dec_ptr).to(tl.float32)
            alpha_t = tl.exp(a_t - lse_dec_t)
            alpha_t = tl.where(mask_m, alpha_t, 0.0)

            dg_ptr = DG_ptr + pid_bh * stride_dg_bh + t * stride_dg_t + m_offsets * stride_dg_m
            dg_t = tl.load(dg_ptr, mask=mask_m, other=0.0).to(tl.float32)

            alpha_dot = tl.sum(alpha_t * dg_t, axis=0)
            ds_dec = alpha_t * (dg_t - alpha_dot)
            ds_dec_ptr = DS_dec_ptr + pid_bh * stride_ds_bh + t * stride_ds_t + m_offsets * stride_ds_m
            tl.store(ds_dec_ptr, ds_dec, mask=mask_m)


@triton.jit
def flare_recurrent_bwd_dsz_separate_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    LSE_enc_ptr,
    Q_dec_ptr,
    K_dec_ptr,
    LSE_dec_ptr,
    dO_ptr,
    dV_ptr,
    DS_enc_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kt, stride_kh, stride_kd,
    stride_vb, stride_vt, stride_vh, stride_vd,
    stride_lsee_b, stride_lsee_h, stride_lsee_t, stride_lsee_m,
    stride_qdb, stride_qdt, stride_qdh, stride_qdd,
    stride_kdh, stride_kdm, stride_kdd,
    stride_lsed_b, stride_lsed_h, stride_lsed_t,
    stride_do_b, stride_do_h, stride_do_t, stride_do_d,
    stride_dv_bh, stride_dv_t, stride_dv_d,
    stride_ds_bh, stride_ds_t, stride_ds_m,
    H, M, T,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # scalar
    t = tl.program_id(1)  # scalar

    pid_b = pid_bh // H  # scalar
    pid_h = pid_bh - pid_b * H  # scalar

    m_offsets = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_m = m_offsets < M  # [BLOCK_M]

    a_t = _recurrent_decode_score_vec(  # [BLOCK_M]
        Q_dec_ptr,
        K_dec_ptr,
        stride_qdb, stride_qdt, stride_qdh, stride_qdd,
        stride_kdh, stride_kdm, stride_kdd,
        pid_b, pid_h, t,
        m_offsets, mask_m,
        scale,
        D=D,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )
    lse_dec_ptr = LSE_dec_ptr + pid_b * stride_lsed_b + pid_h * stride_lsed_h + t * stride_lsed_t  # scalar ptr
    lse_dec_t = tl.load(lse_dec_ptr).to(tl.float32)  # scalar
    alpha_t = tl.exp(a_t - lse_dec_t)  # [BLOCK_M]
    alpha_t = tl.where(mask_m, alpha_t, 0.0)  # [BLOCK_M]

    lse_ptr = LSE_enc_ptr + pid_b * stride_lsee_b + pid_h * stride_lsee_h + t * stride_lsee_t + m_offsets * stride_lsee_m  # [BLOCK_M] ptrs
    lse_t = tl.load(lse_ptr, mask=mask_m, other=0.0).to(tl.float32)  # [BLOCK_M]

    r_t = tl.zeros([BLOCK_M], tl.float32)  # [BLOCK_M]
    tau = 0  # scalar
    while tau <= t:
        s_tau = _recurrent_score_vec(  # [BLOCK_M]
            Q_ptr,
            K_ptr,
            stride_qh, stride_qm, stride_qd,
            stride_kb, stride_kt, stride_kh, stride_kd,
            pid_b, pid_h, tau,
            m_offsets, mask_m,
            scale,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )
        c_tau = _recurrent_dy_v_dot(  # scalar
            V_ptr,
            dO_ptr,
            stride_vb, stride_vt, stride_vh, stride_vd,
            stride_do_b, stride_do_h, stride_do_t, stride_do_d,
            pid_b, pid_h, tau, t,
            D=D,
            BLOCK_D=BLOCK_D,
        )
        a_tau = tl.exp(s_tau - lse_t)  # [BLOCK_M]
        r_t += a_tau * c_tau  # [BLOCK_M]
        tau += 1

    tau = 0  # scalar
    while tau <= t:
        s_tau = _recurrent_score_vec(  # [BLOCK_M]
            Q_ptr,
            K_ptr,
            stride_qh, stride_qm, stride_qd,
            stride_kb, stride_kt, stride_kh, stride_kd,
            pid_b, pid_h, tau,
            m_offsets, mask_m,
            scale,
            D=D,
            BLOCK_M=BLOCK_M,
            BLOCK_K=BLOCK_K,
        )
        c_tau = _recurrent_dy_v_dot(  # scalar
            V_ptr,
            dO_ptr,
            stride_vb, stride_vt, stride_vh, stride_vd,
            stride_do_b, stride_do_h, stride_do_t, stride_do_d,
            pid_b, pid_h, tau, t,
            D=D,
            BLOCK_D=BLOCK_D,
        )
        a_tau = tl.exp(s_tau - lse_t)  # [BLOCK_M]
        ds_tau = alpha_t * a_tau * (c_tau - r_t)  # [BLOCK_M]
        w_tau_t = tl.sum(alpha_t * a_tau, axis=0)  # scalar
        d0 = 0  # scalar
        while d0 < D:
            d_offsets = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            mask_d = d_offsets < D  # [BLOCK_D]
            dO_ptr_t = dO_ptr + pid_b * stride_do_b + pid_h * stride_do_h + t * stride_do_t + d_offsets * stride_do_d  # [BLOCK_D] ptrs
            dO_t = tl.load(dO_ptr_t, mask=mask_d, other=0.0).to(tl.float32)  # [BLOCK_D]
            dV_ptr_tau = dV_ptr + pid_bh * stride_dv_bh + tau * stride_dv_t + d_offsets * stride_dv_d  # [BLOCK_D] ptrs
            tl.atomic_add(dV_ptr_tau, w_tau_t * dO_t, mask=mask_d)  # [BLOCK_D]
            d0 += BLOCK_D
        ds_ptr = DS_enc_ptr + pid_bh * stride_ds_bh + tau * stride_ds_t + m_offsets * stride_ds_m  # [BLOCK_M] ptrs
        tl.atomic_add(ds_ptr, ds_tau, mask=mask_m)  # [BLOCK_M]
        tau += 1


def _recurrent_flare_backward_impl(ctx, dY):
    if dY is None:
        return None, None, None, None, None, None, None, None, None

    separate_Q_dec = getattr(ctx, "separate_Q_dec", False)
    separate_K_dec = getattr(ctx, "separate_K_dec", False)
    saved = ctx.saved_tensors
    expected_len = 5 + int(separate_Q_dec) + int(separate_K_dec)
    if len(saved) != expected_len:
        raise RuntimeError(
            f"Recurrent FLARE backward expected {expected_len} saved tensors "
            f"(5 base + separate_Q_dec={separate_Q_dec} + separate_K_dec={separate_K_dec}), got {len(saved)}."
        )

    Q, K, V, LSE_enc, LSE_dec = saved[:5]
    idx = 5
    if separate_Q_dec:
        Q_dec_saved = saved[idx]
        idx += 1
    else:
        Q_dec_saved = K
    if separate_K_dec:
        K_dec_saved = saved[idx]
    else:
        K_dec_saved = Q

    scale = ctx.scale

    H, M, D = Q.size()
    B, T, _, _ = K.size()
    if (M % 16) != 0 or (D % 16) != 0:
        raise ValueError(f"RecurrentFLARE dense backward requires M and D be multiples of 16. Got M={M}, D={D}")

    BH = B * H
    block_m = M
    block_d, block_k = _get_recurrent_block_d_k(D)
    block_t = 16
    input_precision = _get_input_precision()
    num_warps = 4 if block_m <= 64 else 8

    dO = dY.contiguous().to(torch.float32)
    dg_buf = torch.zeros((BH, T, M), device=Q.device, dtype=torch.float32)
    dS_enc = torch.zeros((BH, T, M), device=Q.device, dtype=torch.float32)
    use_shared_decode_aliases = (not separate_Q_dec) and (not separate_K_dec)
    dS_dec = dS_enc if use_shared_decode_aliases else torch.zeros((BH, T, M), device=Q.device, dtype=torch.float32)
    dV_bhtd = torch.zeros((BH, T, D), device=Q.device, dtype=torch.float32)
    dQ = torch.zeros((H, M, D), device=Q.device, dtype=torch.float32)
    dK = torch.zeros((B, T, H, D), device=Q.device, dtype=torch.float32)

    Q_dec_bt_hd = Q_dec_saved if separate_Q_dec else K
    K_dec_hmd = K_dec_saved if separate_K_dec else Q

    flare_recurrent_bwd_dg_part_kernel[(BH, triton.cdiv(D, block_d))](
        Q, K, V, dO, dg_buf,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dg_buf.stride(0), dg_buf.stride(1), dg_buf.stride(2),
        H, M, T,
        scale,
        D=D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        BLOCK_T=block_t,
        num_warps=num_warps,
        num_stages=2,
    )

    flare_recurrent_bwd_dscore_decode_kernel[(BH, triton.cdiv(T, block_t))](
        Q_dec_bt_hd,
        K_dec_hmd,
        LSE_dec,
        dg_buf,
        dS_dec,
        Q_dec_bt_hd.stride(0), Q_dec_bt_hd.stride(1), Q_dec_bt_hd.stride(2), Q_dec_bt_hd.stride(3),
        K_dec_hmd.stride(0), K_dec_hmd.stride(1), K_dec_hmd.stride(2),
        LSE_dec.stride(0), LSE_dec.stride(1), LSE_dec.stride(2),
        dg_buf.stride(0), dg_buf.stride(1), dg_buf.stride(2),
        dS_dec.stride(0), dS_dec.stride(1), dS_dec.stride(2),
        H, M, T,
        scale,
        D=D,
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        BLOCK_T=block_t,
        num_warps=num_warps,
        num_stages=2,
    )

    # this is launching one program per token.
    # There are 2x token loops [0,...t]. so programs have uneven work. this needs to be corrected.
    flare_recurrent_bwd_dsz_separate_kernel[(BH, T)](
        Q,
        K,
        V,
        LSE_enc,
        Q_dec_bt_hd,
        K_dec_hmd,
        LSE_dec,
        dO,
        dV_bhtd,
        dS_enc,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        LSE_enc.stride(0), LSE_enc.stride(1), LSE_enc.stride(2), LSE_enc.stride(3),
        Q_dec_bt_hd.stride(0), Q_dec_bt_hd.stride(1), Q_dec_bt_hd.stride(2), Q_dec_bt_hd.stride(3),
        K_dec_hmd.stride(0), K_dec_hmd.stride(1), K_dec_hmd.stride(2),
        LSE_dec.stride(0), LSE_dec.stride(1), LSE_dec.stride(2),
        dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
        dV_bhtd.stride(0), dV_bhtd.stride(1), dV_bhtd.stride(2),
        dS_enc.stride(0), dS_enc.stride(1), dS_enc.stride(2),
        H, M, T,
        scale,
        D=D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=2,
    )

    dS_enc_qk = dS_enc.unsqueeze(1)
    flare_chunk_bwd_recurrent_qk[(BH, 1)](
        K,
        Q,
        dS_enc_qk,
        dQ,
        dK,
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        Q.stride(0), Q.stride(1), Q.stride(2),
        dS_enc_qk.stride(0), dS_enc_qk.stride(1), dS_enc_qk.stride(2), dS_enc_qk.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        BH, M, T, D, scale,
        CHUNK_SIZE=T,
        BLOCK_T=block_t,
        INPUT_PRECISION=input_precision,
        ACCUM_DK=False,
        H=H,
        num_warps=num_warps,
        num_stages=2,
    )

    dQ_dec = None
    dK_dec = None
    if not use_shared_decode_aliases:
        dK_dec_target = torch.zeros_like(K_dec_hmd, dtype=torch.float32) if separate_K_dec else dQ
        dQ_dec_target = torch.zeros_like(Q_dec_bt_hd, dtype=torch.float32) if separate_Q_dec else dK
        dS_dec_qk = dS_dec.unsqueeze(1)
        flare_chunk_bwd_recurrent_qk[(BH, 1)](
            Q_dec_bt_hd,
            K_dec_hmd,
            dS_dec_qk,
            dK_dec_target,
            dQ_dec_target,
            Q_dec_bt_hd.stride(0), Q_dec_bt_hd.stride(1), Q_dec_bt_hd.stride(2), Q_dec_bt_hd.stride(3),
            K_dec_hmd.stride(0), K_dec_hmd.stride(1), K_dec_hmd.stride(2),
            dS_dec_qk.stride(0), dS_dec_qk.stride(1), dS_dec_qk.stride(2), dS_dec_qk.stride(3),
            dK_dec_target.stride(0), dK_dec_target.stride(1), dK_dec_target.stride(2),
            dQ_dec_target.stride(0), dQ_dec_target.stride(1), dQ_dec_target.stride(2), dQ_dec_target.stride(3),
            BH, M, T, D, scale,
            CHUNK_SIZE=T,
            BLOCK_T=block_t,
            INPUT_PRECISION=input_precision,
            ACCUM_DK=not separate_Q_dec,
            H=H,
            num_warps=num_warps,
            num_stages=2,
        )
        dQ_dec = dQ_dec_target if separate_Q_dec else None
        dK_dec = dK_dec_target if separate_K_dec else None

    dV = dV_bhtd.view(B, H, T, D).permute(0, 2, 1, 3).contiguous()
    return (
        dQ.to(Q.dtype),
        dK.to(K.dtype),
        dV.to(V.dtype),
        None,
        None,
        None,
        None if dQ_dec is None else dQ_dec.to(Q_dec_saved.dtype),
        None if dK_dec is None else dK_dec.to(K_dec_saved.dtype),
        None,
    )


@triton.jit
def flare_recurrent_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Q_dec_ptr,
    K_dec_ptr,
    LSE_dec_ptr,
    Y_ptr,
    LSE_enc_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_qdb, stride_qdh, stride_qdt, stride_qdd,
    stride_kdb, stride_kdh, stride_kdm, stride_kdd,
    stride_lsed_b, stride_lsed_h, stride_lsed_t,
    stride_yb, stride_yh, stride_yt, stride_yd,
    stride_lsee_b, stride_lsee_h, stride_lsee_t, stride_lsee_m,
    B, H, M,
    T,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    WEIGHT_SHARING_ENC_DEC: tl.constexpr,
    SINGLE_M_TILE: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    pid_m = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    m_state = tl.full([BLOCK_M], -float("inf"), tl.float32)
    d_state = tl.zeros([BLOCK_M], tl.float32)
    u_state = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    tl.multiple_of(d_offsets, 8)
    tl.max_contiguous(d_offsets, 16)

    tl.static_assert(BLOCK_D > 0)
    tl.static_assert(BLOCK_M > 0)

    t0 = 0
    while t0 < T:
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T

        # Stage the local [BLOCK_T, BLOCK_D] V tile once per chunk. The score path
        # still needs the full q^T k reduction, so we replay that over BLOCK_K
        # slices of D for each output-d tile.
        v_blk_ptr = tl.make_block_ptr(
            base=v_base,
            shape=(T, D),
            strides=(stride_vt, stride_vd),
            offsets=(t0, pid_d * BLOCK_D),
            block_shape=(BLOCK_T, BLOCK_D),
            order=(1, 0),
        )
        v_sub = tl.load(v_blk_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
        s_sub = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)
        for k0 in tl.static_range(0, D, BLOCK_K):
            d_offsets_k = k0 + tl.arange(0, BLOCK_K)
            mask_d_k = d_offsets_k < D
            q_ptr = q_base + m_offsets[:, None] * stride_qm + d_offsets_k[None, :] * stride_qd
            q_tile = tl.load(q_ptr, mask=mask_m[:, None] & mask_d_k[None, :], other=0.0).to(tl.float32)
            k_blk_ptr = tl.make_block_ptr(
                base=k_base,
                shape=(T, D),
                strides=(stride_kt, stride_kd),
                offsets=(t0, k0),
                block_shape=(BLOCK_T, BLOCK_K),
                order=(1, 0),
            )
            k_tile = tl.load(k_blk_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
            s_sub += tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        s_sub = s_sub * scale
        s_sub = tl.where(mask_m[:, None] & mask_t[None, :], s_sub, -float("inf"))

        lse_dec_ptr = LSE_dec_ptr + b * stride_lsed_b + h * stride_lsed_h + t_offsets * stride_lsed_t
        lse_dec_sub = tl.load(lse_dec_ptr, mask=mask_t, other=0.0).to(tl.float32)

        if WEIGHT_SHARING_ENC_DEC:
            a_sub = s_sub
        else:
            a_sub = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)
            for k0 in tl.static_range(0, D, BLOCK_K):
                d_offsets_k = k0 + tl.arange(0, BLOCK_K)
                mask_d_k = d_offsets_k < D
                q_dec_blk_ptr = tl.make_block_ptr(
                    base=Q_dec_ptr + b * stride_qdb + h * stride_qdh,
                    shape=(T, D),
                    strides=(stride_qdt, stride_qdd),
                    offsets=(t0, k0),
                    block_shape=(BLOCK_T, BLOCK_K),
                    order=(1, 0),
                )
                q_dec_tile = tl.load(q_dec_blk_ptr, boundary_check=(0, 1), padding_option="zero").to(tl.float32)
                k_dec_ptr = K_dec_ptr + b * stride_kdb + h * stride_kdh + m_offsets[:, None] * stride_kdm + d_offsets_k[None, :] * stride_kdd
                k_dec_tile = tl.load(k_dec_ptr, mask=mask_m[:, None] & mask_d_k[None, :], other=0.0).to(tl.float32)
                a_sub += tl.dot(k_dec_tile, tl.trans(q_dec_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            a_sub = a_sub * scale
            a_sub = tl.where(mask_m[:, None] & mask_t[None, :], a_sub, -float("inf"))

        exp_decode_sub = tl.exp(a_sub - lse_dec_sub[None, :])
        exp_decode_sub = tl.where(mask_m[:, None] & mask_t[None, :], exp_decode_sub, 0.0)

        t_idx = tl.arange(0, BLOCK_T)
        for j in tl.static_range(0, BLOCK_T):
            valid_t = (t0 + j) < T
            col_mask = t_idx == j
            s_t = tl.sum(tl.where(col_mask[None, :], s_sub, 0.0), axis=1)
            s_t = tl.where(mask_m & valid_t, s_t, -float("inf"))

            m_new = tl.maximum(m_state, s_t)
            gamma = tl.exp(m_state - m_new)
            eta = tl.exp(s_t - m_new)
            d_state = d_state * gamma + eta

            v_t = tl.sum(tl.where(col_mask[:, None], v_sub, 0.0), axis=0)
            u_state = u_state * gamma[:, None] + eta[:, None] * v_t[None, :]

            inv_d = 1.0 / tl.where(d_state > 0, d_state, 1.0)
            exp_a = tl.sum(tl.where(col_mask[None, :], exp_decode_sub, 0.0), axis=1)

            w = exp_a * inv_d          # [M]
            y_num = tl.sum(w[:, None] * u_state, axis=0)
            y_block = y_num

            y_row_ptr = tl.make_block_ptr(
                base=y_base,
                shape=(T, D),
                strides=(stride_yt, stride_yd),
                offsets=(t0 + j, pid_d * BLOCK_D),
                block_shape=(1, BLOCK_D),
                order=(1, 0),
            )
            y_dtype = y_row_ptr.type.element_ty.element_ty
            if SINGLE_M_TILE:
                tl.store(y_row_ptr, y_block[None, :].to(y_dtype), boundary_check=(0, 1))
            else:
                y_ptr = y_base + (t0 + j) * stride_yt
                tl.atomic_add(y_ptr + d_offsets * stride_yd, y_block.to(tl.float32), mask=valid_t & mask_d)
            lse_row_ptr = LSE_enc_ptr + b * stride_lsee_b + h * stride_lsee_h + (t0 + j) * stride_lsee_t
            tl.store(lse_row_ptr + m_offsets * stride_lsee_m, m_new + tl.log(tl.maximum(d_state, 1e-20)), mask=(pid_d == 0) & mask_m)

            m_state = m_new

        t0 += BLOCK_T


@triton.jit
def flare_chunk_bwd_recurrent_qk(
    K_ptr, Q_ptr,
    dS_ptr,
    dQ_ptr, dK_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_ds_bh, stride_ds_chunk, stride_ds_t, stride_ds_m,
    stride_dq_h, stride_dq_m, stride_dq_d,
    stride_dk_b, stride_dk_n, stride_dk_h, stride_dk_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    ACCUM_DK: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    chunk_start = chunk_idx * CHUNK_SIZE

    m_offsets = tl.arange(0, M)
    d_offsets = tl.arange(0, D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_md = mask_m[:, None] & mask_d[None, :]

    q_vals = tl.load(
        Q_ptr + pid_h * stride_q_h + m_offsets[:, None] * stride_q_m + d_offsets[None, :] * stride_q_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)

    t0 = 0
    while t0 < CHUNK_SIZE:
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        token_valid = (chunk_start + t_offsets) < N
        mask_tm = token_valid[:, None] & mask_m[None, :]
        mask_td = token_valid[:, None] & mask_d[None, :]

        dS_block = tl.load(
            dS_ptr
            + pid_bh * stride_ds_bh
            + chunk_idx * stride_ds_chunk
            + t_offsets[:, None] * stride_ds_t
            + m_offsets[None, :] * stride_ds_m,
            mask=mask_tm,
            other=0.0,
        ).to(tl.float32)

        k_block = tl.load(
            K_ptr
            + pid_b * stride_k_b
            + (chunk_start + t_offsets)[:, None] * stride_k_n
            + pid_h * stride_k_h
            + d_offsets[None, :] * stride_k_d,
            mask=mask_td,
            other=0.0,
        ).to(tl.float32)

        dK_block = tl.dot(dS_block, q_vals, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale
        dQ_block = tl.dot(tl.trans(dS_block), k_block, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale

        tl.atomic_add(
            dQ_ptr + pid_h * stride_dq_h + m_offsets[:, None] * stride_dq_m + d_offsets[None, :] * stride_dq_d,
            dQ_block,
            mask=mask_md,
        )
        dk_ptr = (
            dK_ptr
            + pid_b * stride_dk_b
            + (chunk_start + t_offsets)[:, None] * stride_dk_n
            + pid_h * stride_dk_h
            + d_offsets[None, :] * stride_dk_d
        )
        if ACCUM_DK:
            tl.atomic_add(dk_ptr, dK_block, mask=mask_td)
        else:
            tl.store(dk_ptr, dK_block, mask=mask_td)
        t0 += BLOCK_T

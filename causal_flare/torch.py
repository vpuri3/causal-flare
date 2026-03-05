from ._common import *


def _resolve_flare_causal_decode_inputs(Q_enc, K_enc, Q_dec=None, K_dec=None):
    separate_Q_dec = Q_dec is not None
    separate_K_dec = K_dec is not None
    use_default_q_dec = (not separate_Q_dec) or (Q_dec is K_enc)
    use_default_k_dec = (not separate_K_dec) or (K_dec is Q_enc)
    weight_sharing_enc_dec = use_default_q_dec and use_default_k_dec

    if separate_Q_dec:
        if Q_dec.dim() != 4:
            raise ValueError(
                "Expected Q_dec [B, N, H, D]. "
                f"Got Q_dec.dim()={Q_dec.dim()} and Q_dec.shape={tuple(Q_dec.shape)}"
            )
        if Q_dec.size() != K_enc.size():
            raise ValueError(
                f"Q_dec and K_enc must have the same shape. Got Q_dec.shape={Q_dec.shape} and K_enc.shape={K_enc.shape}"
            )
    else:
        Q_dec = K_enc

    if separate_K_dec:
        if K_dec.dim() != 3:
            raise ValueError(
                "Expected K_dec [H, M, D]. "
                f"Got K_dec.dim()={K_dec.dim()} and K_dec.shape={tuple(K_dec.shape)}"
            )
        if K_dec.size() != Q_enc.size():
            raise ValueError(
                f"K_dec and Q_enc must have the same shape. Got K_dec.shape={K_dec.shape} and Q_enc.shape={Q_enc.shape}"
            )
    else:
        K_dec = Q_enc

    return Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec


def flare_recurrent_pytorch(Q, K, V, scale=None, Q_dec=None, K_dec=None):
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

    B, T, _, _ = K.size()
    H, M, D = Q.size()
    scale = _resolve_attn_scale(scale, D)
    Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_flare_causal_decode_inputs(
        Q, K, Q_dec, K_dec
    )
    device = Q.device
    out_dtype = Q.dtype

    Q_f = Q.float().unsqueeze(0).expand(B, -1, -1, -1)  # [B,H,M,D]
    K_f = K.float().permute(0, 2, 1, 3).contiguous()    # [B,H,T,D]
    V_f = V.float().permute(0, 2, 1, 3).contiguous()    # [B,H,T,D]
    Q_dec_f = Q_dec.float().permute(0, 2, 1, 3).contiguous() if separate_Q_dec else K_f
    K_dec_f = K_dec.float().unsqueeze(0).expand(B, -1, -1, -1) if separate_K_dec else Q_f

    U = torch.zeros((B, H, M, D), device=device, dtype=torch.float32)
    d = torch.zeros((B, H, M), device=device, dtype=torch.float32)
    m = torch.full((B, H, M), -float("inf"), device=device, dtype=torch.float32)
    Y = torch.empty((B, H, T, D), device=device, dtype=out_dtype)

    for t in range(T):
        k_t = K_f[:, :, t, :]
        v_t = V_f[:, :, t, :]
        s_t = torch.einsum("bhmd,bhd->bhm", Q_f, k_t) * scale
        m_new = torch.maximum(m, s_t)
        gamma = torch.exp(m - m_new)
        eta = torch.exp(s_t - m_new)
        d = d * gamma + eta
        U = U * gamma[..., None] + eta[..., None] * v_t[:, :, None, :]
        Z = U / d[..., None]
        if weight_sharing_enc_dec:
            a_t = s_t
        else:
            q_t_dec = Q_dec_f[:, :, t, :]
            a_t = torch.einsum("bhd,bhmd->bhm", q_t_dec, K_dec_f) * scale
        alpha = torch.softmax(a_t, dim=-1)
        y_t = torch.einsum("bhm,bhmd->bhd", alpha, Z)
        Y[:, :, t, :] = y_t.to(out_dtype)
        m = m_new

    return Y


def _flare_recurrent_dense_lse_forward_state(Q, K, V, scale=None, Q_dec=None, K_dec=None):
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

    B, T, _, _ = K.size()
    H, M, D = Q.size()
    scale = _resolve_attn_scale(scale, D)
    Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_flare_causal_decode_inputs(
        Q, K, Q_dec, K_dec
    )

    Q_f = Q.float().unsqueeze(0).expand(B, -1, -1, -1)  # [B,H,M,D]
    K_f = K.float().permute(0, 2, 1, 3).contiguous()    # [B,H,T,D]
    V_f = V.float().permute(0, 2, 1, 3).contiguous()    # [B,H,T,D]
    Q_dec_f = Q_dec.float().permute(0, 2, 1, 3).contiguous() if separate_Q_dec else K_f
    K_dec_f = K_dec.float().unsqueeze(0).expand(B, -1, -1, -1) if separate_K_dec else Q_f

    S_enc = torch.einsum("bhmd,bhtd->bhtm", Q_f, K_f) * scale  # [B,H,T,M]
    LSE_enc = torch.logcumsumexp(S_enc, dim=2)                  # [B,H,T,M]

    if weight_sharing_enc_dec:
        S_dec = S_enc
    else:
        S_dec = torch.einsum("bhtd,bhmd->bhtm", Q_dec_f, K_dec_f) * scale
    LSE_dec = torch.logsumexp(S_dec, dim=-1)                    # [B,H,T]
    alpha = torch.exp(S_dec - LSE_dec.unsqueeze(-1))            # [B,H,T,M]

    m_state = torch.full((B, H, M), -float("inf"), device=Q.device, dtype=torch.float32)
    d_state = torch.zeros((B, H, M), device=Q.device, dtype=torch.float32)
    u_state = torch.zeros((B, H, M, D), device=Q.device, dtype=torch.float32)
    Z = torch.empty((B, H, T, M, D), device=Q.device, dtype=torch.float32)
    for t in range(T):
        s_t = S_enc[:, :, t, :]
        m_new = torch.maximum(m_state, s_t)
        gamma = torch.exp(m_state - m_new)
        eta = torch.exp(s_t - m_new)
        d_state = d_state * gamma + eta
        u_state = u_state * gamma[..., None] + eta[..., None] * V_f[:, :, t, None, :]
        Z[:, :, t, :, :] = u_state / d_state[..., None]
        m_state = m_new

    Y = torch.einsum("bhtm,bhtmd->bhtd", alpha, Z)
    return {
        "Q_f": Q_f,
        "K_f": K_f,
        "V_f": V_f,
        "Q_dec_f": Q_dec_f,
        "K_dec_f": K_dec_f,
        "S_enc": S_enc,
        "LSE_enc": LSE_enc,
        "LSE_dec": LSE_dec,
        "alpha": alpha,
        "Z": Z,
        "Y": Y,
        "scale": scale,
        "separate_Q_dec": separate_Q_dec,
        "separate_K_dec": separate_K_dec,
        "weight_sharing_enc_dec": weight_sharing_enc_dec,
    }


def _flare_recurrent_dense_decode_backward(alpha, Z, dY):
    # y_t = sum_m alpha_tm * z_tm
    dZ = alpha[..., None] * dY[:, :, :, None, :]  # [B, H, T, M, D]
    score_proj = torch.sum(dY[:, :, :, None, :] * Z, dim=-1)  # [B, H, T, M]
    centered = score_proj - torch.sum(alpha * score_proj, dim=-1, keepdim=True)  # [B, H, T, M]
    dS_dec = alpha * centered  # [B, H, T, M]
    return dZ, dS_dec


def _flare_recurrent_dense_encode_backward(S_enc, LSE_enc, V_f, dZ):
    B, H, T, M = S_enc.shape  # scalars
    D = V_f.size(-1)  # scalar
    dS_enc = torch.zeros_like(S_enc)  # [B, H, T, M]
    dV_f = torch.zeros((B, H, T, D), device=S_enc.device, dtype=torch.float32)  # [B, H, T, D]
    for t in range(T):
        s_pref = S_enc[:, :, : t + 1, :]  # [B, H, t+1, M]
        lse_t = LSE_enc[:, :, t : t + 1, :]  # [B, H, 1, M]
        p_t = torch.exp(s_pref - lse_t)  # [B, H, t+1, M]
        dz_t = dZ[:, :, t, :, :]  # [B, H, M, D]
        c_t = torch.einsum("bhud,bhmd->bhum", V_f[:, :, : t + 1, :], dz_t)  # [B, H, t+1, M]
        r_t = torch.sum(p_t * c_t, dim=2, keepdim=True)  # [B, H, 1, M]
        dS_enc[:, :, : t + 1, :] += p_t * (c_t - r_t)  # [B, H, t+1, M]
        dV_f[:, :, : t + 1, :] += torch.einsum("bhum,bhmd->bhud", p_t, dz_t)  # [B, H, t+1, D]
    return dS_enc, dV_f


def flare_recurrent_dense_backward_pytorch(Q, K, V, dY, scale=None, Q_dec=None, K_dec=None):
    """
    Dense backward decomposition using LSE_enc/LSE_dec:
      1) decode-softmax branch (dS_dec, dZ)
      2) encoder recurrent-softmax branch (dS_enc, dV)
      3) score projections to dQ/dK (+ optional dQ_dec/dK_dec)
    """
    state = _flare_recurrent_dense_lse_forward_state(Q, K, V, scale=scale, Q_dec=Q_dec, K_dec=K_dec)
    Q_f = state["Q_f"]  # [B, H, M, D]
    K_f = state["K_f"]  # [B, H, T, D]
    V_f = state["V_f"]  # [B, H, T, D]
    Q_dec_f = state["Q_dec_f"]  # [B, H, T, D]
    K_dec_f = state["K_dec_f"]  # [B, H, M, D]
    S_enc = state["S_enc"]  # [B, H, T, M]
    LSE_enc = state["LSE_enc"]  # [B, H, T, M]
    alpha = state["alpha"]  # [B, H, T, M]
    Z = state["Z"]  # [B, H, T, M, D]
    Y = state["Y"]  # [B, H, T, D]
    scale = state["scale"]  # scalar
    separate_Q_dec = state["separate_Q_dec"]  # bool
    separate_K_dec = state["separate_K_dec"]  # bool
    weight_sharing_enc_dec = state["weight_sharing_enc_dec"]  # bool

    if dY.dim() != 4 or dY.size(0) != K.size(0) or dY.size(1) != K.size(2) or dY.size(2) != K.size(1) or dY.size(3) != K.size(3):
        raise ValueError(
            "Expected dY [B, H, T, D] matching recurrent output layout. "
            f"Got dY.shape={tuple(dY.shape)} for K.shape={tuple(K.shape)}"
        )
    dY_f = dY.float()  # [B, H, T, D]

    dZ, dS_dec = _flare_recurrent_dense_decode_backward(alpha, Z, dY_f)  # [B,H,T,M,D], [B,H,T,M]
    dS_enc, dV_f = _flare_recurrent_dense_encode_backward(S_enc, LSE_enc, V_f, dZ)  # [B,H,T,M], [B,H,T,D]

    if weight_sharing_enc_dec:
        for t in range(S_enc.size(2)):
            dS_enc[:, :, t, :] += dS_dec[:, :, t, :]

    dQ = scale * torch.einsum("bhtm,bhtd->hmd", dS_enc, K_f)  # [H, M, D]
    dK_f = scale * torch.einsum("bhtm,bhmd->bhtd", dS_enc, Q_f)  # [B, H, T, D]

    dQ_dec = None
    dK_dec = None
    if not weight_sharing_enc_dec:
        if separate_Q_dec:
            dQ_dec_f = scale * torch.einsum("bhtm,bhmd->bhtd", dS_dec, K_dec_f)  # [B, H, T, D]
            dQ_dec = dQ_dec_f.permute(0, 2, 1, 3).contiguous()  # [B, T, H, D]
        else:
            # Q_dec aliases K_enc: accumulate directly into dK.
            dK_f += scale * torch.einsum("bhtm,bhmd->bhtd", dS_dec, K_dec_f)  # [B, H, T, D]

        if separate_K_dec:
            dK_dec_f = scale * torch.einsum("bhtm,bhtd->bhmd", dS_dec, Q_dec_f)  # [B, H, M, D]
            dK_dec = dK_dec_f.sum(dim=0).contiguous()  # [H, M, D]
        else:
            # K_dec aliases Q_enc: accumulate directly into dQ.
            dQ += scale * torch.einsum("bhtm,bhtd->bhmd", dS_dec, Q_dec_f).sum(dim=0)  # [H, M, D]

    dK = dK_f.permute(0, 2, 1, 3).contiguous()  # [B, T, H, D]
    dV = dV_f.permute(0, 2, 1, 3).contiguous()  # [B, T, H, D]
    return (
        Y.to(Q.dtype),
        dQ.to(Q.dtype),
        dK.to(K.dtype),
        dV.to(V.dtype),
        None if dQ_dec is None else dQ_dec.to(Q.dtype),
        None if dK_dec is None else dK_dec.to(Q.dtype),
    )

#======================================================================#
#======================================================================#
#======================================================================#
# Prefix, Streaming, and Cached Implementations
#======================================================================#
#======================================================================#
#======================================================================#

def flare_noncausal(Q, K, V, scale=None):
    assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
        "Q, K, V must be 3D and 4D tensors respectively "
        f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
    )
    assert K.size() == V.size(), f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}"
    assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
        "Expected Q [H, M, D] and K/V [B, N, H, D]. "
        f"Got Q.shape={Q.shape} and K.shape={K.shape}"
    )
    H, M, D = Q.size()
    B, N, H, D = K.size()
    scale = _resolve_attn_scale(scale, D)

    Q = Q.unsqueeze(0).expand(B, -1, -1, -1)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)

    Y = F.scaled_dot_product_attention(Q, K, V, is_causal=False, scale=scale)
    Z = F.scaled_dot_product_attention(K, Q, Y, is_causal=False, scale=scale)

    return Y

def causal_SDPA(Q, K, V):
    if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
        raise ValueError(
            "Causal SDPA expects Q, K, V all as 4D tensors for benchmarking. "
            f"Got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
        )
    if Q.size() != K.size() or K.size() != V.size():
        raise ValueError(
            f"Q, K, V must have the same shape for causal SDPA. Got Q.shape={Q.shape}, K.shape={K.shape}, V.shape={V.shape}"
        )
    Q = Q.permute(0, 2, 1, 3)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    return Y

def flare_causal_reference(Q_enc, K_enc, V_enc, Q_dec=None, K_dec=None, scale=None):
    assert Q_enc.dim() == 3 and K_enc.dim() == 4 and V_enc.dim() == 4, (
        "Q_enc, K_enc, V_enc must be 3D and 4D tensors respectively "
        f"got Q_enc.dim()={Q_enc.dim()}, K_enc.dim()={K_enc.dim()}, V_enc.dim()={V_enc.dim()}"
    )
    assert K_enc.size() == V_enc.size(), f"K_enc and V_enc must have the same shape. Got K_enc.shape={K_enc.shape} and V_enc.shape={V_enc.shape}"
    assert Q_enc.size(0) == K_enc.size(2) and Q_enc.size(2) == K_enc.size(3), (
        "Expected Q_enc [H, M, D] and K_enc/V_enc [B, N, H, D]. "
        f"Got Q_enc.shape={Q_enc.shape} and K_enc.shape={K_enc.shape}"
    )
    H, M, D = Q_enc.size()
    B, N, H, D = K_enc.size()
    scale = _resolve_attn_scale(scale, D)

    Q_dec, K_dec, separate_Q_dec, separate_K_dec, _ = _resolve_flare_causal_decode_inputs(Q_enc, K_enc, Q_dec, K_dec)

    if os.environ.get("FLARE_REFERENCE_FP32", "1") == "1":
        Q_enc = Q_enc.float()
        K_enc = K_enc.float()
        V_enc = V_enc.float()
        Q_dec = Q_dec.float() if separate_Q_dec else K_enc
        K_dec = K_dec.float() if separate_K_dec else Q_enc
    else:
        Q_dec = Q_dec.to(K_enc.dtype) if separate_Q_dec else K_enc
        K_dec = K_dec.to(Q_enc.dtype) if separate_K_dec else Q_enc

    Q_enc = Q_enc.unsqueeze(0).expand(B, -1, -1, -1)
    K_enc = K_enc.permute(0, 2, 1, 3)
    V_enc = V_enc.permute(0, 2, 1, 3)
    Q_dec = Q_dec.permute(0, 2, 1, 3) if separate_Q_dec else K_enc
    K_dec = K_dec.unsqueeze(0).expand(B, -1, -1, -1) if separate_K_dec else Q_enc
    Y = torch.zeros_like(K_enc)
    for t in range(N):
        Kt_enc = K_enc[:, :, :t+1, :]
        Vt_enc = V_enc[:, :, :t+1, :]
        Qt_dec = Q_dec[:, :, t:t+1, :]
        Vt_dec = F.scaled_dot_product_attention(Q_enc, Kt_enc, Vt_enc, is_causal=False, scale=scale)
        Yt = F.scaled_dot_product_attention(Qt_dec, K_dec, Vt_dec, is_causal=False, scale=scale)
        Y[:, :, t] = Yt.squeeze(2)

    Y = Y.permute(0, 2, 1, 3)

    return Y


def flare_causal_perceiver_ar(Q, K, V, scale=None):
    """
    Causal FLARE reference computed with a Perceiver-AR style batched formulation.

    This is mathematically equivalent to ``flare_causal_reference``:
      z_t = Attn(Q, K[:t+1], V[:t+1])
      y_t = Attn(K_t, Q, z_t)
    but computes all timesteps in one batched SDPA call per stage.
    """
    assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
        "Q, K, V must be 3D and 4D tensors respectively "
        f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
    )
    assert K.size() == V.size(), f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}"
    assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
        "Expected Q [H, M, D] and K/V [B, N, H, D]. "
        f"Got Q.shape={Q.shape} and K.shape={K.shape}"
    )
    H, M, D = Q.size()
    B, N, Hk, Dk = K.size()
    scale = _resolve_attn_scale(scale, D)
    assert H == Hk and D == Dk, "Incompatible Q/K/V dimensions"

    if os.environ.get("FLARE_REFERENCE_FP32", "1") == "1":
        Q = Q.float()
        K = K.float()
        V = V.float()

    # Layout for SDPA: [B, H, T, D].
    Q_bhmd = Q.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, M, D]
    K_bhnd = K.permute(0, 2, 1, 3)                  # [B, H, N, D]
    V_bhnd = V.permute(0, 2, 1, 3)                  # [B, H, N, D]

    BN = B * N

    # Encode all timesteps in parallel:
    # q_enc[b, t] = Q, k_enc[b, t] = K[b], masked to prefix <= t.
    q_enc = Q_bhmd.unsqueeze(1).expand(B, N, H, M, D).reshape(BN, H, M, D)
    k_enc = K_bhnd.unsqueeze(1).expand(B, N, H, N, D).reshape(BN, H, N, D)
    v_enc = V_bhnd.unsqueeze(1).expand(B, N, H, N, D).reshape(BN, H, N, D)

    i = torch.arange(N, device=K.device)[:, None]
    j = torch.arange(N, device=K.device)[None, :]
    # SDPA bool mask uses True for "allowed", False for "masked".
    enc_mask = (j <= i)[None, :, :].expand(B, N, N).reshape(BN, 1, 1, N)

    Z = F.scaled_dot_product_attention(
        q_enc,
        k_enc,
        v_enc,
        attn_mask=enc_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )  # [BN, H, M, D]

    # Decode all timesteps in parallel:
    # y_t = Attn(k_t, Q, z_t)
    q_dec = K_bhnd.permute(0, 2, 1, 3).reshape(BN, H, 1, D)  # [BN, H, 1, D]
    k_dec = Q_bhmd.unsqueeze(1).expand(B, N, H, M, D).reshape(BN, H, M, D)
    v_dec = Z

    Y = F.scaled_dot_product_attention(
        q_dec,
        k_dec,
        v_dec,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )  # [BN, H, 1, D]

    Y = Y.reshape(B, N, H, D)
    return Y


def flare_causal_perciever_ar(Q, K, V, scale=None):
    """Backward-compatible alias for common misspelling."""
    return flare_causal_perceiver_ar(Q, K, V, scale=scale)

#------------------------------------------------------------------------------#
# PyTorch Implementation of FLARE
#------------------------------------------------------------------------------#

def flare_causal_chunked(Q, K, V, scale=None, eps=None, profile: bool = False, chunk_size=None, Q_dec=None, K_dec=None):
    assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
        "Q, K, V must be 3D and 4D tensors respectively "
        f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
    )
    assert K.size() == V.size(), f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}"
    assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
        "Expected Q [H, M, D] and K/V [B, N, H, D]. "
        f"Got Q.shape={Q.shape} and K.shape={K.shape}"
    )
    H, M, D = Q.size()
    B, N, Hk, Dk = K.size()
    scale = _resolve_attn_scale(scale, D)
    assert H == Hk and D == Dk, "Incompatible K/V dimensions"
    Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_flare_causal_decode_inputs(
        Q, K, Q_dec, K_dec
    )

    device = Q.device
    out_dtype = Q.dtype
    compute_dtype = torch.float32
    if os.environ.get("FLARE_PYTORCH_MATCH_REFERENCE", "") == "1":
        compute_dtype = Q.dtype

    Q_f = Q.to(compute_dtype)
    K_f = K.to(compute_dtype)
    V_f = V.to(compute_dtype)
    Q_dec_f = None
    K_dec_f = None
    if not weight_sharing_enc_dec:
        Q_dec_f = Q_dec.to(compute_dtype).permute(0, 2, 1, 3).contiguous() if separate_Q_dec else K_f.permute(0, 2, 1, 3)
        K_dec_f = K_dec.to(compute_dtype).unsqueeze(0).expand(B, -1, -1, -1) if separate_K_dec else Q_f.unsqueeze(0).expand(B, -1, -1, -1)

    ###
    # CHUNKING & PADDING
    ###

    if chunk_size is None:
        env_chunk = os.environ.get("FLARE_PYTORCH_CHUNK_SIZE", "")
        chunk_size = int(env_chunk) if env_chunk else None
    C = int(chunk_size) if chunk_size is not None else max(64, min(2048, N // 2))
    # Non-stable branch only.
    CHUNK_SIZE = C
    NC = NUM_CHUNKS = math.ceil(N / CHUNK_SIZE)

    PADDED_LEN = NUM_CHUNKS * CHUNK_SIZE
    PAD = PADDED_LEN - N

    if PAD > 0:
        pad_val = torch.zeros((B, PAD, H, D), device=device, dtype=compute_dtype)
        K_f = torch.cat([K_f, pad_val], dim=1)
        V_f = torch.cat([V_f, pad_val], dim=1)

    ###
    # CHUNKING
    ###

    Kc = K_f.reshape(B, NC, C, H, D).permute(0, 3, 1, 2, 4).contiguous()
    Vc = V_f.reshape(B, NC, C, H, D).permute(0, 3, 1, 2, 4).contiguous()

    #---------------------------------------------------------------#
    # Phase 0: Compute scores
    # NEEDS: Q, Kc, Vc
    # RETURNS: score_chunk
    #---------------------------------------------------------------#

    score_chunk = scale * torch.einsum("bhncd,hmd->bhncm", Kc, Q_f)  # [B, H, NC, C, M]

    if PAD > 0:
        score_chunk.view(B, H, PADDED_LEN, M)[:, :, -PAD:, :] = -torch.inf

    phase1_start = phase1_end = None
    phase2_start = phase2_end = None
    phase3_start = phase3_end = None
    if profile and torch.cuda.is_available():
        phase1_start = torch.cuda.Event(enable_timing=True)
        phase1_end = torch.cuda.Event(enable_timing=True)
        phase2_start = torch.cuda.Event(enable_timing=True)
        phase2_end = torch.cuda.Event(enable_timing=True)
        phase3_start = torch.cuda.Event(enable_timing=True)
        phase3_end = torch.cuda.Event(enable_timing=True)
        phase1_start.record()

    #---------------------------------------------------------------#
    # Phase 1: Compute chunk statistics independently for each chunk
    # NEEDS: score_chunk, Vc
    # RETURNS: score_chunk_max, score_chunk_den, score_chunk_num
    #---------------------------------------------------------------#

    score_chunk_max = score_chunk.max(dim=3).values                               # [B, H, NC, M]
    score_chunk_exp = torch.exp(score_chunk - score_chunk_max.unsqueeze(3))       # [B, H, NC, C, M]
    score_chunk_den = score_chunk_exp.sum(dim=3)                                  # [B, H, NC, M]
    BHNC = B * H * NC
    exp_b = score_chunk_exp.reshape(BHNC, C, M)
    V_b = Vc.reshape(BHNC, C, D)
    score_chunk_num = torch.bmm(exp_b.transpose(1, 2), V_b).reshape(B, H, NC, M, D)

    if profile and torch.cuda.is_available():
        phase1_end.record()
        phase2_start.record()

    #---------------------------------------------------------------#
    # Phase 2: Compute prefix statistics from independent chunk statistics
    # NEEDS: score_chunk_max, score_chunk_den, score_chunk_num
    # RETURNS: score_prev_max, score_prev_den, score_prev_num
    #---------------------------------------------------------------#

    # Score suffix (prev) statistics needed for phase 3
    score_prev_max = torch.empty(B, H, NC, M, device=device, dtype=compute_dtype)
    score_prev_den = torch.zeros(B, H, NC, M, device=device, dtype=compute_dtype)
    score_prev_num = torch.zeros(B, H, NC, M, D, device=device, dtype=compute_dtype)

    # temporary variables for prefix statistics
    max_curr = torch.full((B, H, M), -float("inf"), device=device, dtype=compute_dtype)
    den_curr = torch.zeros((B, H, M), device=device, dtype=compute_dtype)
    num_curr = torch.zeros((B, H, M, D), device=device, dtype=compute_dtype)

    for chunk_idx in range(NC):
        score_prev_max[:, :, chunk_idx, :] = max_curr
        score_prev_den[:, :, chunk_idx, :] = den_curr
        score_prev_num[:, :, chunk_idx, :, :] = num_curr

        sc_max = score_chunk_max[:, :, chunk_idx, :]
        sc_den = score_chunk_den[:, :, chunk_idx, :]
        sc_num = score_chunk_num[:, :, chunk_idx, :]

        ###
        ### online softmax update
        ###

        # get new max (including current chunk)
        max_new = torch.maximum(max_curr, sc_max)

        # get rescale factors
        rescale_factor_prev = torch.exp(max_curr - max_new) # rescale factor for previous chunks
        rescale_factor_curr = torch.exp(sc_max - max_new)   # rescale factor for current chunk

        # update denominator, numerator, max
        den_curr = den_curr * rescale_factor_prev + sc_den * rescale_factor_curr
        num_curr = num_curr * rescale_factor_prev.unsqueeze(-1) + sc_num * rescale_factor_curr.unsqueeze(-1)
        max_curr = max_new

    if profile and torch.cuda.is_available():
        phase2_end.record()
        phase3_start.record()

    #---------------------------------------------------------------#
    # Phase 3: Compute output by replaying the within-chunk recurrence from chunk-start prefix stats.
    # NEEDS: score_chunk, score_prev_*, Vc
    # RETURNS: Yc
    #---------------------------------------------------------------#
    S = score_chunk  # [B, H, NC, C, M]
    Yc = torch.empty((B, H, NC, C, D), device=device, dtype=compute_dtype)
    for chunk_idx in range(NC):
        max_curr = score_prev_max[:, :, chunk_idx, :]
        den_curr = score_prev_den[:, :, chunk_idx, :]
        num_curr = score_prev_num[:, :, chunk_idx, :, :]
        for t in range(C):
            token_idx = chunk_idx * C + t
            s_t = S[:, :, chunk_idx, t, :]
            v_t = Vc[:, :, chunk_idx, t, :]

            max_new = torch.maximum(max_curr, s_t)
            rescale_prev = torch.exp(max_curr - max_new)
            rescale_curr = torch.exp(s_t - max_new)
            den_curr = den_curr * rescale_prev + rescale_curr
            num_curr = num_curr * rescale_prev.unsqueeze(-1) + rescale_curr.unsqueeze(-1) * v_t[:, :, None, :]
            z_t = num_curr / den_curr.unsqueeze(-1)

            if token_idx >= N:
                Yc[:, :, chunk_idx, t, :] = 0.0
                max_curr = max_new
                continue

            if weight_sharing_enc_dec:
                a_t = s_t
            else:
                q_t_dec = Q_dec_f[:, :, token_idx, :]
                a_t = torch.einsum("bhd,bhmd->bhm", q_t_dec, K_dec_f) * scale

            alpha = torch.softmax(a_t, dim=-1)
            Yc[:, :, chunk_idx, t, :] = torch.einsum("bhm,bhmd->bhd", alpha, z_t)
            max_curr = max_new

    #---------------------------------------------------------------#
    # Return output
    #---------------------------------------------------------------#

    Y = Yc.reshape(B, H, PADDED_LEN, D)[:, :, :N, :].permute(0, 2, 1, 3).to(out_dtype)
    _check_finite("flare_causal_chunked.Y", Y)
    if profile and torch.cuda.is_available():
        phase3_end.record()
        torch.cuda.synchronize()
        mode = _BWD_PROFILE_MODE or "triton3"
        _BWD_PROFILE_TIMINGS.setdefault(mode, {})
        _BWD_PROFILE_TIMINGS[mode]["phase1_chunk_stats"] = phase1_start.elapsed_time(phase1_end)
        _BWD_PROFILE_TIMINGS[mode]["phase2_prefix"] = phase2_start.elapsed_time(phase2_end)
        _BWD_PROFILE_TIMINGS[mode]["phase3_output"] = phase3_start.elapsed_time(phase3_end)
    return Y


#======================================================================#
# Testing scripts
#======================================================================#

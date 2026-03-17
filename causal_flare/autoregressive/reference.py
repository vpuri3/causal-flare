from causal_flare._common import *
from causal_flare._reference_utils import (
    resolve_flare_causal_decode_inputs as _resolve_flare_causal_decode_inputs,
    validate_flare_qkv_layouts as _validate_flare_qkv_layouts,
)

#======================================================================#
#======================================================================#
#======================================================================#
# Reference implementations
#======================================================================#
#======================================================================#
#======================================================================#

def flare_causal_reference(Q_enc, K_enc, V_enc, Q_dec=None, K_dec=None, scale=None):
    B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q_enc, K_enc, V_enc, name="FLARE reference")
    scale = _resolve_attn_scale(scale, D_score)

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
    Y = torch.zeros((B, H, N, D_value), device=K_enc.device, dtype=V_enc.dtype)
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
    B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="Perceiver-AR FLARE reference")
    scale = _resolve_attn_scale(scale, D_score)

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
    q_enc = Q_bhmd.unsqueeze(1).expand(B, N, H, M, D_score).reshape(BN, H, M, D_score)
    k_enc = K_bhnd.unsqueeze(1).expand(B, N, H, N, D_score).reshape(BN, H, N, D_score)
    v_enc = V_bhnd.unsqueeze(1).expand(B, N, H, N, D_value).reshape(BN, H, N, D_value)

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
    q_dec = K_bhnd.permute(0, 2, 1, 3).reshape(BN, H, 1, D_score)  # [BN, H, 1, D_k]
    k_dec = Q_bhmd.unsqueeze(1).expand(B, N, H, M, D_score).reshape(BN, H, M, D_score)
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

    Y = Y.reshape(B, N, H, D_value)
    return Y


def flare_causal_perciever_ar(Q, K, V, scale=None):
    """Backward-compatible alias for common misspelling."""
    return flare_causal_perceiver_ar(Q, K, V, scale=scale)

#======================================================================#
#======================================================================#
#======================================================================#
# PyTorch Implementations
#======================================================================#
#======================================================================#
#======================================================================#

def _canonicalize_reference_flare_state(
    state: dict[str, torch.Tensor] | None,
    batch_size: int,
    num_heads: int,
    num_latents: int,
    value_head_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if state is None:
        m = torch.full((batch_size, num_heads, num_latents), -float("inf"), device=device, dtype=torch.float32)
        d = torch.zeros((batch_size, num_heads, num_latents), device=device, dtype=torch.float32)
        u = torch.zeros((batch_size, num_heads, num_latents, value_head_dim), device=device, dtype=torch.float32)
        return {"m": m, "d": d, "u": u}

    if not isinstance(state, dict):
        raise ValueError(f"FLARE recurrent state must be a dict with keys {'m', 'd', 'u'}. Got {type(state)}")
    if not all(k in state for k in ("m", "d", "u")):
        raise ValueError(f"Invalid FLARE recurrent state keys: {list(state.keys())}")

    m = state["m"].to(device=device, dtype=torch.float32)
    d = state["d"].to(device=device, dtype=torch.float32)
    u = state["u"].to(device=device, dtype=torch.float32)
    expected_scalar_shape = (batch_size, num_heads, num_latents)
    expected_u_shape = (batch_size, num_heads, num_latents, value_head_dim)
    if tuple(m.shape) != expected_scalar_shape or tuple(d.shape) != expected_scalar_shape or tuple(u.shape) != expected_u_shape:
        raise ValueError(
            "Invalid FLARE recurrent state shapes. "
            f"Expected m/d={expected_scalar_shape}, u={expected_u_shape}; "
            f"got m={tuple(m.shape)}, d={tuple(d.shape)}, u={tuple(u.shape)}"
        )
    return {"m": m, "d": d, "u": u}


def _merge_reference_flare_stats(
    m_a: torch.Tensor,
    d_a: torch.Tensor,
    u_a: torch.Tensor,
    m_b: torch.Tensor,
    d_b: torch.Tensor,
    u_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m_new = torch.maximum(m_a, m_b)
    is_a_inf = torch.isinf(m_a) & (m_a < 0)
    is_b_inf = torch.isinf(m_b) & (m_b < 0)
    is_new_inf = torch.isinf(m_new) & (m_new < 0)
    m_safe = torch.where(is_new_inf, torch.zeros_like(m_new), m_new)

    scale_a = torch.where(
        is_a_inf & is_new_inf,
        torch.ones_like(m_new),
        torch.where(is_a_inf, torch.zeros_like(m_new), torch.exp(m_a - m_safe)),
    )
    scale_b = torch.where(
        is_b_inf & is_new_inf,
        torch.ones_like(m_new),
        torch.where(is_b_inf, torch.zeros_like(m_new), torch.exp(m_b - m_safe)),
    )

    d_new = d_a * scale_a + d_b * scale_b
    u_new = u_a * scale_a[..., None] + u_b * scale_b[..., None]
    return m_new, d_new, u_new


def _resolve_reference_chunk_size(N: int, M: int, D_score: int, chunk_size) -> int:
    if chunk_size is None:
        env_chunk = os.environ.get("FLARE_PYTORCH_CHUNK_SIZE", "")
        chunk_size = int(env_chunk) if env_chunk else None
    if chunk_size is not None:
        return int(chunk_size)
    if D_score <= 32 and M <= 64 and N >= 1024:
        return 64
    if D_score <= 64 and M <= 128 and N >= 1024:
        return 128
    return max(64, min(2048, max(1, N // 2)))


def flare_autoregressive_pytorch(
    Q,
    K,
    V,
    scale=None,
    eps=None,
    profile: bool = False,
    chunk_size=None,
    Q_dec=None,
    K_dec=None,
    state: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    return_state: bool = False,
):
    B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="Chunked FLARE")
    scale = _resolve_attn_scale(scale, D_score)
    Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_flare_causal_decode_inputs(
        Q, K, Q_dec, K_dec
    )
    use_fast_unmasked = attention_mask is None and state is None
    if attention_mask is not None and attention_mask.shape != (B, N):
        raise ValueError(f"attention_mask must be [B, N]. Got {tuple(attention_mask.shape)}")

    device = Q.device
    out_dtype = V.dtype
    compute_dtype = torch.float32
    if os.environ.get("FLARE_PYTORCH_MATCH_REFERENCE", "") == "1":
        compute_dtype = Q.dtype

    if N == 0:
        next_state = _canonicalize_reference_flare_state(
            state=state,
            batch_size=B,
            num_heads=H,
            num_latents=M,
            value_head_dim=D_value,
            device=device,
        )
        Y = torch.empty((B, 0, H, D_value), device=device, dtype=out_dtype)
        return (Y, next_state) if return_state else Y

    Q_f = Q.to(compute_dtype)
    K_f = K.to(compute_dtype)
    V_f = V.to(compute_dtype)
    Q_dec_f = None
    K_dec_f = None
    if not weight_sharing_enc_dec:
        Q_dec_f = Q_dec.to(compute_dtype).permute(0, 2, 1, 3).contiguous() if separate_Q_dec else K_f.permute(0, 2, 1, 3)
        K_dec_f = (
            K_dec.to(compute_dtype).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
            if separate_K_dec else
            Q_f.unsqueeze(0).expand(B, -1, -1, -1)
        )

    C = _resolve_reference_chunk_size(N, M, D_score, chunk_size)
    NC = math.ceil(N / C)
    PADDED_LEN = NC * C
    PAD = PADDED_LEN - N

    valid_tokens = None
    if attention_mask is not None:
        valid_tokens = attention_mask.to(device=device, dtype=torch.bool)
    if PAD > 0:
        K_f = torch.cat([K_f, torch.zeros((B, PAD, H, D_score), device=device, dtype=compute_dtype)], dim=1)
        V_f = torch.cat([V_f, torch.zeros((B, PAD, H, D_value), device=device, dtype=compute_dtype)], dim=1)
        if valid_tokens is None:
            valid_tokens = torch.ones((B, N), device=device, dtype=torch.bool)
        valid_tokens = torch.cat([valid_tokens, torch.zeros((B, PAD), device=device, dtype=torch.bool)], dim=1)
        if not weight_sharing_enc_dec and separate_Q_dec:
            Q_dec_f = torch.cat([Q_dec_f, torch.zeros((B, H, PAD, D_score), device=device, dtype=compute_dtype)], dim=2)
    elif valid_tokens is not None:
        valid_tokens = valid_tokens.contiguous()
    if not weight_sharing_enc_dec and not separate_Q_dec:
        Q_dec_f = K_f.permute(0, 2, 1, 3).contiguous()

    Kc = K_f.reshape(B, NC, C, H, D_score).permute(0, 3, 1, 2, 4).contiguous()
    Vc = V_f.reshape(B, NC, C, H, D_value).permute(0, 3, 1, 2, 4).contiguous()
    valid_chunk = None
    if valid_tokens is not None:
        valid_chunk = valid_tokens.reshape(B, NC, C).unsqueeze(1).unsqueeze(-1)

    st = _canonicalize_reference_flare_state(
        state=state,
        batch_size=B,
        num_heads=H,
        num_latents=M,
        value_head_dim=D_value,
        device=device,
    )

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

    scale_f = float(scale)
    score_chunk = scale_f * torch.einsum("bhncd,hmd->bhncm", Kc, Q_f)
    if valid_chunk is not None:
        score_chunk = torch.where(valid_chunk, score_chunk, torch.full_like(score_chunk, -float("inf")))
    elif PAD > 0:
        score_chunk.view(B, H, PADDED_LEN, M)[:, :, -PAD:, :] = -torch.inf

    score_chunk_max = score_chunk.max(dim=3).values
    score_chunk_max_safe = torch.where(
        torch.isinf(score_chunk_max) & (score_chunk_max < 0),
        torch.zeros_like(score_chunk_max),
        score_chunk_max,
    )
    score_chunk_exp = torch.exp(score_chunk - score_chunk_max_safe.unsqueeze(3))
    if valid_chunk is not None:
        score_chunk_exp = score_chunk_exp * valid_chunk.to(score_chunk_exp.dtype)
    score_chunk_den = score_chunk_exp.sum(dim=3)
    BHNC = B * H * NC
    score_chunk_num = torch.bmm(
        score_chunk_exp.reshape(BHNC, C, M).transpose(1, 2),
        Vc.reshape(BHNC, C, D_value),
    ).reshape(B, H, NC, M, D_value)

    if profile and torch.cuda.is_available():
        phase1_end.record()
        phase2_start.record()

    #---------------------------------------------------------------#
    # Phase 2: Compute prefix statistics from independent chunk statistics
    # NEEDS: score_chunk_max, score_chunk_den, score_chunk_num
    # RETURNS: score_prev_max, score_prev_den, score_prev_num
    #---------------------------------------------------------------#

    score_prev_max = torch.empty(B, H, NC, M, device=device, dtype=torch.float32)
    score_prev_den = torch.empty(B, H, NC, M, device=device, dtype=torch.float32)
    score_prev_num = torch.empty(B, H, NC, M, D_value, device=device, dtype=torch.float32)
    max_curr = st["m"]
    den_curr = st["d"]
    num_curr = st["u"]

    for chunk_idx in range(NC):
        score_prev_max[:, :, chunk_idx, :] = max_curr
        score_prev_den[:, :, chunk_idx, :] = den_curr
        score_prev_num[:, :, chunk_idx, :, :] = num_curr
        sc_max = score_chunk_max[:, :, chunk_idx, :]
        sc_den = score_chunk_den[:, :, chunk_idx, :]
        sc_num = score_chunk_num[:, :, chunk_idx, :, :]
        if use_fast_unmasked:
            max_new = torch.maximum(max_curr, sc_max)
            rescale_prev = torch.exp(max_curr - max_new)
            rescale_curr = torch.exp(sc_max - max_new)
            den_curr = den_curr * rescale_prev + sc_den * rescale_curr
            num_curr = num_curr * rescale_prev.unsqueeze(-1) + sc_num * rescale_curr.unsqueeze(-1)
            max_curr = max_new
        else:
            max_curr, den_curr, num_curr = _merge_reference_flare_stats(
                max_curr,
                den_curr,
                num_curr,
                sc_max,
                sc_den,
                sc_num,
            )
    next_state = {
        "m": max_curr.contiguous(),
        "d": den_curr.contiguous(),
        "u": num_curr.contiguous(),
    }

    if profile and torch.cuda.is_available():
        phase2_end.record()
        phase3_start.record()

    #---------------------------------------------------------------#
    # Phase 3: Compute output by replaying the within-chunk recurrence from chunk-start prefix stats.
    # NEEDS: score_chunk, score_prev_*, Vc
    # RETURNS: Yc
    #---------------------------------------------------------------#
    Yc = torch.empty((B, H, NC, C, D_value), device=device, dtype=torch.float32)
    for chunk_idx in range(NC):
        max_curr = score_prev_max[:, :, chunk_idx, :]
        den_curr = score_prev_den[:, :, chunk_idx, :]
        num_curr = score_prev_num[:, :, chunk_idx, :, :]
        for t in range(C):
            token_idx = chunk_idx * C + t
            s_t = score_chunk[:, :, chunk_idx, t, :]
            v_t = Vc[:, :, chunk_idx, t, :]
            if use_fast_unmasked:
                max_new = torch.maximum(max_curr, s_t)
                rescale_prev = torch.exp(max_curr - max_new)
                rescale_curr = torch.exp(s_t - max_new)
                den_curr = den_curr * rescale_prev + rescale_curr
                num_curr = num_curr * rescale_prev.unsqueeze(-1) + rescale_curr.unsqueeze(-1) * v_t[:, :, None, :]
            else:
                max_new = torch.maximum(max_curr, s_t)
                is_m_inf = torch.isinf(max_curr) & (max_curr < 0)
                is_m_new_inf = torch.isinf(max_new) & (max_new < 0)
                max_new_safe = torch.where(is_m_new_inf, torch.zeros_like(max_new), max_new)
                gamma = torch.where(
                    is_m_inf & is_m_new_inf,
                    torch.ones_like(max_new),
                    torch.where(is_m_inf, torch.zeros_like(max_new), torch.exp(max_curr - max_new_safe)),
                )
                eta = torch.where(is_m_new_inf, torch.zeros_like(s_t), torch.exp(s_t - max_new_safe))
                den_curr = den_curr * gamma + eta
                num_curr = num_curr * gamma.unsqueeze(-1) + eta.unsqueeze(-1) * v_t[:, :, None, :]
            max_curr = max_new

            if token_idx >= N:
                Yc[:, :, chunk_idx, t, :] = 0.0
                continue

            if weight_sharing_enc_dec:
                a_t = s_t
            else:
                q_t_dec = Q_dec_f[:, :, token_idx, :]
                a_t = torch.einsum("bhd,bhmd->bhm", q_t_dec, K_dec_f) * scale_f
            if use_fast_unmasked:
                z_t = num_curr / den_curr.unsqueeze(-1)
                alpha_t = torch.softmax(a_t, dim=-1)
            else:
                d_safe = torch.where(den_curr > 0, den_curr, torch.ones_like(den_curr))
                z_t = num_curr / d_safe.unsqueeze(-1)
                if valid_chunk is not None:
                    valid_t = valid_chunk[:, :, chunk_idx, t, :]
                    a_t = torch.where(valid_t, a_t, torch.zeros_like(a_t))
                    alpha_t = torch.softmax(a_t, dim=-1) * valid_t.to(a_t.dtype)
                else:
                    alpha_t = torch.softmax(a_t, dim=-1)
            Yc[:, :, chunk_idx, t, :] = torch.einsum("bhm,bhmd->bhd", alpha_t, z_t)

    #---------------------------------------------------------------#
    # Return output
    #---------------------------------------------------------------#

    Y_out = Yc.reshape(B, H, PADDED_LEN, D_value)[:, :, :N, :].permute(0, 2, 1, 3).to(out_dtype)
    _check_finite("flare_autoregressive_pytorch.Y", Y_out)
    if profile and torch.cuda.is_available():
        phase3_end.record()
        torch.cuda.synchronize()
        mode = _BWD_PROFILE_MODE or "triton3"
        _BWD_PROFILE_TIMINGS.setdefault(mode, {})
        _BWD_PROFILE_TIMINGS[mode]["phase1_chunk_stats"] = phase1_start.elapsed_time(phase1_end)
        _BWD_PROFILE_TIMINGS[mode]["phase2_prefix"] = phase2_start.elapsed_time(phase2_end)
        _BWD_PROFILE_TIMINGS[mode]["phase3_output"] = phase3_start.elapsed_time(phase3_end)
    return (Y_out, next_state) if return_state else Y_out


def flare_causal_chunked(Q, K, V, scale=None, eps=None, profile: bool = False, chunk_size=None, Q_dec=None, K_dec=None):
    return flare_autoregressive_pytorch(
        Q,
        K,
        V,
        scale=scale,
        eps=eps,
        profile=profile,
        chunk_size=chunk_size,
        Q_dec=Q_dec,
        K_dec=K_dec,
        state=None,
        attention_mask=None,
        return_state=False,
    )

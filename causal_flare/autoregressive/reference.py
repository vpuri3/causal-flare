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


def _stablemax_score_transform(x: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    one = torch.ones((), device=x.device, dtype=x.dtype)
    power_tensor = torch.as_tensor(power, device=x.device, dtype=x.dtype)
    pos_base = torch.where(x >= 0, x + one, one)
    neg_base = torch.where(x < 0, one - x, one)
    pos = torch.pow(pos_base, power_tensor)
    neg = torch.pow(neg_base, -power_tensor)
    return torch.where(x >= 0, pos, neg)


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
    # Phase 1: Per-chunk encoder summaries
    #
    # For each token tau and latent m we first form the encoder scores
    #
    #   s_{tau,m} = <k_tau, q_m> * scale
    #
    # inside each chunk. We then reduce those tokenwise scores into one
    # chunk summary per latent:
    #
    #   chunk_max[c, m] = max_{tau in chunk c} s_{tau,m}
    #   chunk_den[c, m] = sum_{tau in chunk c} exp(s_{tau,m} - chunk_max[c,m])
    #   chunk_num[c, m] = sum_{tau in chunk c} exp(s_{tau,m} - chunk_max[c,m]) v_tau
    #
    # These are exactly the quantities needed to merge chunks with the
    # standard online-softmax recurrence in phase 2.
    #
    # `score_chunk` keeps the raw within-chunk encoder scores because the
    # replay/output phase still needs the tokenwise values.
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
    # Phase 2: Prefix scan over chunk summaries
    #
    # The goal here is to convert independent chunk summaries into the
    # encoder state that is valid at the *start* of each chunk.
    #
    # For chunk `c`, `score_prev_*[:, :, c, ...]` stores the encoder
    # accumulator after consuming all earlier chunks `[0, c)`, but before
    # consuming any token from chunk `c`.
    #
    # In other words, phase 2 lifts the tokenwise online recurrence to the
    # chunk level:
    #
    #   previous prefix stats + current chunk summary -> new prefix stats
    #
    # The unmasked/stateless hot path uses the simpler original online
    # softmax update for speed. Once masks or external recurrent state are
    # involved, we switch to the inf-safe merge helper so padded or masked
    # rows remain numerically well-defined.
    #
    # The final merged stats become `next_state`, which allows this
    # function to act as a prefill-style recurrent update as well as a pure
    # full-sequence forward pass.
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
    # Phase 3: Replay the token recurrence and emit outputs
    #
    # Phase 2 only tells us the encoder state at the *start* of each
    # chunk. To produce y_t for every token, we still need the exact
    # within-chunk encoder state after processing the prefix ending at t.
    #
    # We therefore replay the original tokenwise recurrence inside each
    # chunk:
    #
    #   prefix-at-chunk-start
    #     + token 0 -> z_0
    #     + token 1 -> z_1
    #     ...
    #
    # where `z_t[m]` is the latent value for latent `m` after folding in
    # all encoder tokens up to timestep `t`.
    #
    # Once `z_t` is available, decode stays standard FLARE:
    #
    #   a_t[m]     = decoder score for latent m at timestep t
    #   alpha_t    = softmax(a_t)
    #   y_t        = sum_m alpha_t[m] z_t[m]
    #
    # This phase dominates runtime for the eager PyTorch path because it
    # still contains the token-by-token replay loop.
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


def flare_autoregressive_stablemax_pytorch(
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
    state: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    return_state: bool = False,
):
    B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="StableMax Chunked FLARE")
    scale = _resolve_attn_scale(scale, D_score)
    Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_flare_causal_decode_inputs(
        Q, K, Q_dec, K_dec
    )
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
        next_state["m"] = torch.zeros_like(next_state["m"])
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
    # Phase 1: Per-chunk stablemax encoder summaries
    #
    # This variant keeps the decoder exactly as standard softmax FLARE, but
    # replaces the encoder softmax weights with the stablemax transform
    #
    #   w_{tau,m} = stablemax(s_{tau,m})
    #
    # Unlike softmax, stablemax does not require a running max or any
    # exponential rescaling. The encoder state is therefore just a plain
    # cumulative weighted sum:
    #
    #   d_t[m]   = sum_{tau <= t} w_{tau,m}
    #   u_t[m]   = sum_{tau <= t} w_{tau,m} v_tau
    #
    # So phase 1 only needs to summarize each chunk by:
    #
    #   chunk_den[c, m] = sum_{tau in chunk c} w_{tau,m}
    #   chunk_num[c, m] = sum_{tau in chunk c} w_{tau,m} v_tau
    #
    # There is no `chunk_max` because stablemax does not use max-based
    # normalization.
    #---------------------------------------------------------------#
    scale_f = float(scale)
    score_chunk = scale_f * torch.einsum("bhncd,hmd->bhncm", Kc, Q_f)
    stable_score_chunk = _stablemax_score_transform(score_chunk, power=power)
    if valid_chunk is not None:
        stable_score_chunk = stable_score_chunk * valid_chunk.to(stable_score_chunk.dtype)

    score_chunk_den = stable_score_chunk.sum(dim=3)
    BHNC = B * H * NC
    score_chunk_num = torch.bmm(
        stable_score_chunk.reshape(BHNC, C, M).transpose(1, 2),
        Vc.reshape(BHNC, C, D_value),
    ).reshape(B, H, NC, M, D_value)

    if profile and torch.cuda.is_available():
        phase1_end.record()
        phase2_start.record()

    #---------------------------------------------------------------#
    # Phase 2: Prefix sums of stablemax encoder summaries
    #
    # Since the stablemax encoder recurrence is purely additive, phase 2 is
    # simpler than the softmax version. For each chunk we record the prefix
    # encoder statistics that are valid at the start of that chunk:
    #
    #   prev_den[c, m] = sum of stablemax weights over earlier chunks
    #   prev_num[c, m] = sum of stablemax-weighted values over earlier chunks
    #
    # Then we advance the running state with simple addition.
    #
    # The returned recurrent state again stores only the final encoder
    # accumulators. `m` is unused for stablemax, so we return zeros there
    # to preserve the same state shape as the softmax path.
    #---------------------------------------------------------------#
    score_prev_den = torch.empty(B, H, NC, M, device=device, dtype=torch.float32)
    score_prev_num = torch.empty(B, H, NC, M, D_value, device=device, dtype=torch.float32)
    den_curr = st["d"]
    num_curr = st["u"]
    for chunk_idx in range(NC):
        score_prev_den[:, :, chunk_idx, :] = den_curr
        score_prev_num[:, :, chunk_idx, :, :] = num_curr
        den_curr = den_curr + score_chunk_den[:, :, chunk_idx, :]
        num_curr = num_curr + score_chunk_num[:, :, chunk_idx, :, :]
    next_state = {
        "m": torch.zeros_like(st["m"]),
        "d": den_curr.contiguous(),
        "u": num_curr.contiguous(),
    }

    if profile and torch.cuda.is_available():
        phase2_end.record()
        phase3_start.record()

    #---------------------------------------------------------------#
    # Phase 3A: Decoder probabilities
    #
    # The decoder stays as ordinary softmax attention over latents. We
    # first compute the decoder logits a_{t,m} for every timestep inside
    # each chunk and normalize them with the usual log-sum-exp:
    #
    #   P_{t,m} = softmax_m(a_{t,m})
    #
    # This is independent of the stablemax encoder recurrence, so it can be
    # done in one batched chunkwise pass.
    #---------------------------------------------------------------#
    if weight_sharing_enc_dec:
        decode_logits = score_chunk
    else:
        Q_dec_c = Q_dec_f.reshape(B, H, NC, C, D_score)
        decode_logits = scale_f * torch.einsum("bhncd,bhmd->bhncm", Q_dec_c, K_dec_f)
    if valid_chunk is not None:
        decode_logits = torch.where(valid_chunk, decode_logits, torch.zeros_like(decode_logits))
    decode_lse = torch.logsumexp(decode_logits, dim=-1)
    decode_probs = torch.exp(decode_logits - decode_lse.unsqueeze(-1))
    if valid_chunk is not None:
        decode_probs = decode_probs * valid_chunk.to(decode_probs.dtype)

    #---------------------------------------------------------------#
    # Phase 3B: Within-chunk encoder prefixes + output matmul
    #
    # Rather than materialize
    #
    #   z_t[m] = total_num[t, m] / Z_{t,m}
    #
    # as a [B, H, NC, C, M, D] tensor, we push the reduction over `m`
    # earlier. Writing
    #
    #   R_{t,m} = P_{t,m} / Z_{t,m}
    #
    # gives
    #
    #   y_t = sum_m R_{t,m} * prefix_num[m]
    #       + sum_{tau <= t} sum_m R_{t,m} * w_{tau,m} * v_tau
    #
    # The first term is a direct projection of the chunk-prefix numerators.
    # The second term is a causal [C, C] mixing matrix:
    #
    #   A_{t,tau} = sum_m R_{t,m} * w_{tau,m}
    #
    # followed by a matrix multiply with the value block. This trades the
    # large [C, M, D] intermediate for a smaller [C, C] one.
    #---------------------------------------------------------------#
    chunk_den = stable_score_chunk.cumsum(dim=3)
    total_den = score_prev_den.unsqueeze(3) + chunk_den
    total_den_safe = torch.where(total_den > 0, total_den, torch.ones_like(total_den))
    causal_mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool)).view(1, 1, C, C)
    Yc = torch.empty((B, H, NC, C, D_value), device=device, dtype=torch.float32)
    for chunk_idx in range(NC):
        decode_over_den_chunk = decode_probs[:, :, chunk_idx, :, :] / total_den_safe[:, :, chunk_idx, :, :]
        prefix_out_chunk = torch.einsum("bhcm,bhmd->bhcd", decode_over_den_chunk, score_prev_num[:, :, chunk_idx, :, :])
        local_mix_chunk = torch.matmul(
            decode_over_den_chunk,
            stable_score_chunk[:, :, chunk_idx, :, :].transpose(-1, -2),
        )
        local_mix_chunk = torch.where(causal_mask, local_mix_chunk, torch.zeros_like(local_mix_chunk))
        local_out_chunk = torch.matmul(local_mix_chunk, Vc[:, :, chunk_idx, :, :])
        Yc[:, :, chunk_idx, :, :] = prefix_out_chunk + local_out_chunk

    Y_out = Yc.reshape(B, H, PADDED_LEN, D_value)[:, :, :N, :].permute(0, 2, 1, 3).to(out_dtype)
    _check_finite("flare_autoregressive_stablemax_pytorch.Y", Y_out)
    if profile and torch.cuda.is_available():
        phase3_end.record()
        torch.cuda.synchronize()
        mode = _BWD_PROFILE_MODE or "triton3"
        _BWD_PROFILE_TIMINGS.setdefault(mode, {})
        _BWD_PROFILE_TIMINGS[mode]["phase1_chunk_stats"] = phase1_start.elapsed_time(phase1_end)
        _BWD_PROFILE_TIMINGS[mode]["phase2_prefix"] = phase2_start.elapsed_time(phase2_end)
        _BWD_PROFILE_TIMINGS[mode]["phase3_output"] = phase3_start.elapsed_time(phase3_end)
    return (Y_out, next_state) if return_state else Y_out


def _stablemax_score_transform_grad(x: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    one = torch.ones((), device=x.device, dtype=x.dtype)
    power_tensor = torch.as_tensor(power, device=x.device, dtype=x.dtype)
    pos_base = torch.where(x >= 0, x + one, one)
    neg_base = torch.where(x < 0, one - x, one)
    pos = power_tensor * torch.pow(pos_base, power_tensor - one)
    neg = power_tensor * torch.pow(neg_base, -(power_tensor + one))
    return torch.where(x >= 0, pos, neg)


def _prepare_stablemax_custom_inputs(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    Q_dec: torch.Tensor | None,
    K_dec: torch.Tensor | None,
    scale,
    chunk_size,
    power: float,
    compute_dtype: torch.dtype,
) -> dict[str, object]:
    B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="StableMax Chunked FLARE")
    scale = _resolve_attn_scale(scale, D_score)
    Q_dec_resolved, K_dec_resolved, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_flare_causal_decode_inputs(
        Q, K, Q_dec, K_dec
    )
    device = Q.device

    Q_f = Q.to(compute_dtype)
    K_f = K.to(compute_dtype)
    V_f = V.to(compute_dtype)
    if separate_Q_dec:
        Q_dec_f = Q_dec_resolved.to(compute_dtype).permute(0, 2, 1, 3).contiguous()
    else:
        Q_dec_f = K_f.permute(0, 2, 1, 3).contiguous()
    if separate_K_dec:
        K_dec_f = K_dec_resolved.to(compute_dtype).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
    else:
        K_dec_f = Q_f.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    C = _resolve_reference_chunk_size(N, M, D_score, chunk_size)
    NC = math.ceil(N / C) if N > 0 else 0
    PADDED_LEN = NC * C
    PAD = PADDED_LEN - N
    if PAD > 0:
        K_f = torch.cat([K_f, torch.zeros((B, PAD, H, D_score), device=device, dtype=compute_dtype)], dim=1)
        V_f = torch.cat([V_f, torch.zeros((B, PAD, H, D_value), device=device, dtype=compute_dtype)], dim=1)
        Q_dec_f = torch.cat([Q_dec_f, torch.zeros((B, H, PAD, D_score), device=device, dtype=compute_dtype)], dim=2)

    Kc = K_f.reshape(B, NC, C, H, D_score).permute(0, 3, 1, 2, 4).contiguous() if N > 0 else None
    Vc = V_f.reshape(B, NC, C, H, D_value).permute(0, 3, 1, 2, 4).contiguous() if N > 0 else None
    Q_dec_c = Q_dec_f.reshape(B, H, NC, C, D_score).contiguous() if N > 0 else None
    return {
        "B": B,
        "N": N,
        "H": H,
        "M": M,
        "D_score": D_score,
        "D_value": D_value,
        "device": device,
        "out_dtype": V.dtype,
        "compute_dtype": compute_dtype,
        "scale": float(scale),
        "power": float(power),
        "Q_f": Q_f,
        "Kc": Kc,
        "Vc": Vc,
        "Q_dec_c": Q_dec_c,
        "K_dec_f": K_dec_f,
        "Q_dec_resolved": Q_dec_resolved,
        "K_dec_resolved": K_dec_resolved,
        "C": C,
        "NC": NC,
        "PADDED_LEN": PADDED_LEN,
        "PAD": PAD,
        "BHNC": B * H * NC,
    }


def _forward_stablemax_custom_prepared(prep: dict[str, object], return_cache: bool = False):
    """Run the separate-weight stablemax prefill path on already-packed chunk tensors.

    High-level math
    ----------------
    This computes the FLARE prefill output

        y_t = sum_m p_dec[m, t] * (sum_{tau <= t} s(a_enc[m, tau]) v_tau) / (sum_{tau <= t} s(a_enc[m, tau]))

    where:
    - `a_enc[m, tau] = scale * <k_tau, q_m>` are encoder logits
    - `s(.)` is the stablemax score transform
    - `p_dec[:, t] = softmax(scale * <q_dec_t, k_dec_m>)` are decoder probabilities

    Unlike the softmax reference path, the encoder side does not need max-tracking or
    log-sum-exp rescaling. It only needs prefix sums of:
    - encoder denominators: `sum s(a_enc)`
    - encoder numerators: `sum s(a_enc) * v`

    Tensor layout
    -------------
    The packed tensors use chunked shapes so the hot path can stay in batched matmuls:
    - `Q_f`: `[H, M, D_score]`
      Encoder query bank shared across the batch.
    - `Kc`: `[B, H, NC, C, D_score]`
      Keys, chunked into `NC` chunks of size `C`.
    - `Vc`: `[B, H, NC, C, D_value]`
      Values in the same chunk layout.
    - `Q_dec_c`: `[B, H, NC, C, D_score]`
      Decoder queries, chunked by target token position.
    - `K_dec_f`: `[B, H, M, D_score]`
      Decoder keys, one vector per latent slot.

    Important derived tensors:
    - `score_chunk`: `[B, H, NC, C, M]`
      Encoder logits for every token position inside a chunk against every latent slot.
    - `stable_score_chunk`: `[B, H, NC, C, M]`
      Stablemax-transformed encoder weights.
    - `score_chunk_den`: `[B, H, NC, M]`
      Per-chunk encoder denominator contribution.
    - `score_chunk_num`: `[B, H, NC, M, D_value]`
      Per-chunk encoder numerator contribution.
    - `score_prev_den`: `[B, H, NC, M]`
      Strict-prefix denominator before each chunk.
    - `score_prev_num`: `[B, H, NC, M, D_value]`
      Strict-prefix numerator before each chunk.
    """
    B = prep["B"]
    N = prep["N"]
    H = prep["H"]
    M = prep["M"]
    D_value = prep["D_value"]
    device = prep["device"]
    out_dtype = prep["out_dtype"]
    Q_f = prep["Q_f"]
    Kc = prep["Kc"]
    Vc = prep["Vc"]
    Q_dec_c = prep["Q_dec_c"]
    K_dec_f = prep["K_dec_f"]
    C = prep["C"]
    NC = prep["NC"]
    PADDED_LEN = prep["PADDED_LEN"]
    BHNC = prep["BHNC"]
    scale_f = prep["scale"]
    power = prep["power"]

    if N == 0:
        return torch.empty((B, 0, H, D_value), device=device, dtype=out_dtype)

    # Phase 1: encoder stablemax statistics per chunk.
    #
    # score_chunk[b, h, n, c, m] = scale * <K[b, n, c, h, :], Q[h, m, :]>
    # Shapes:
    # - Kc:          [B, H, NC, C, D_score]
    # - Q_f:         [H, M, D_score]
    # - score_chunk: [B, H, NC, C, M]
    score_chunk = scale_f * torch.einsum("bhncd,hmd->bhncm", Kc, Q_f)

    # stable_score_chunk applies the positive stablemax transform pointwise:
    # s(x) = (1 + x)^power      if x >= 0
    #      = (1 - x)^(-power)   if x < 0
    #
    # This is the encoder-side analogue of exp(.) in softmax attention, but without
    # any log-domain max correction. The encoder recurrence is therefore a plain sum.
    stable_score_chunk = _stablemax_score_transform(score_chunk, power=power)

    # Per-chunk encoder denominator:
    #   den_chunk[n, m] = sum_{c in chunk n} s(score_chunk[n, c, m])
    # Shape: [B, H, NC, M]
    score_chunk_den = stable_score_chunk.sum(dim=3)

    # Per-chunk encoder numerator:
    #   num_chunk[n, m, d] = sum_{c in chunk n} s(score_chunk[n, c, m]) * V[n, c, d]
    # Implemented as a batched matrix multiply over flattened [B, H, NC].
    # Shape: [B, H, NC, M, D_value]
    score_chunk_num = torch.bmm(
        stable_score_chunk.reshape(BHNC, C, M).transpose(1, 2),
        Vc.reshape(BHNC, C, D_value),
    ).reshape(B, H, NC, M, D_value)

    # Strict chunk prefix sums.
    # These are the encoder stats from all chunks before the current chunk:
    #   prev_den[n] = sum_{n' < n} den_chunk[n']
    #   prev_num[n] = sum_{n' < n} num_chunk[n']
    #
    # Shapes:
    # - score_prev_den: [B, H, NC, M]
    # - score_prev_num: [B, H, NC, M, D_value]
    score_prev_den = torch.cumsum(score_chunk_den, dim=2) - score_chunk_den
    score_prev_num = torch.cumsum(score_chunk_num, dim=2) - score_chunk_num

    # Phase 2: decoder softmax probabilities over the latent slots.
    #
    # decode_logits[b, h, n, c, m] = scale * <Q_dec[b, h, n, c, :], K_dec[b, h, m, :]>
    # Shapes:
    # - Q_dec_c:       [B, H, NC, C, D_score]
    # - K_dec_f:       [B, H, M, D_score]
    # - decode_logits: [B, H, NC, C, M]
    decode_logits = scale_f * torch.einsum("bhncd,bhmd->bhncm", Q_dec_c, K_dec_f)
    decode_probs = torch.softmax(decode_logits, dim=-1)

    # Phase 3A: build the encoder normalization seen by each target token.
    #
    # chunk_den is the within-chunk encoder prefix:
    #   chunk_den[n, c, m] = sum_{c' <= c} s(score_chunk[n, c', m])
    #
    # total_den is the full encoder denominator available at each target token:
    #   total_den[n, c, m] = prev_den[n, m] + chunk_den[n, c, m]
    #
    # Shapes:
    # - chunk_den:      [B, H, NC, C, M]
    # - total_den:      [B, H, NC, C, M]
    # - decode_over_den [B, H, NC, C, M]
    chunk_den = stable_score_chunk.cumsum(dim=3)
    total_den = score_prev_den.unsqueeze(3) + chunk_den
    total_den_safe = torch.where(total_den > 0, total_den, torch.ones_like(total_den))
    inv_total_den = total_den_safe.reciprocal()
    decode_over_den = decode_probs * inv_total_den

    # Phase 3B: combine decoder weights with encoder numerators.
    #
    # Prefix contribution from previous chunks:
    #   prefix_out[n, c, d] = sum_m decode_probs[n, c, m] / total_den[n, c, m] * prev_num[n, m, d]
    #
    # Shape: [B, H, NC, C, D_value]
    prefix_out = torch.einsum("bhncm,bhnmd->bhncd", decode_over_den, score_prev_num)

    # Local causal mixing inside the active chunk.
    #
    # For a fixed chunk and target position c, we want:
    #   sum_{tau <= c} sum_m decode_probs[c, m] * s(score_chunk[tau, m]) / total_den[c, m] * V[tau]
    #
    # Write this as:
    #   local_mix[c, tau] = sum_m decode_probs[c, m] / total_den[c, m] * s(score_chunk[tau, m])
    # then enforce tau <= c with a causal mask, and finally:
    #   local_out[c] = sum_tau local_mix[c, tau] * V[tau]
    #
    # Shapes:
    # - local_mix: [B, H, NC, C, C]
    # - local_out: [B, H, NC, C, D_value]
    causal_mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool)).view(1, 1, 1, C, C)
    local_mix = torch.matmul(decode_over_den, stable_score_chunk.transpose(-1, -2))
    local_mix = torch.where(causal_mask, local_mix, torch.zeros_like(local_mix))
    local_out = torch.matmul(local_mix.reshape(BHNC, C, C), Vc.reshape(BHNC, C, D_value)).reshape(B, H, NC, C, D_value)
    Yc = prefix_out + local_out

    # Undo chunk packing and drop right-padding introduced during preparation.
    # Y_out shape: [B, N, H, D_value]
    Y_out = Yc.reshape(B, H, PADDED_LEN, D_value)[:, :, :N, :].permute(0, 2, 1, 3).to(out_dtype)
    _check_finite("FLAREAutoregressiveStablemaxPyTorch.Y", Y_out)
    if return_cache:
        return Y_out, {
            "Q_f": Q_f,
            "Kc": Kc,
            "Vc": Vc,
            "Q_dec_c": Q_dec_c,
            "K_dec_f": K_dec_f,
            "score_chunk": score_chunk,
            "stable_score_chunk": stable_score_chunk,
            "decode_probs": decode_probs,
            "score_prev_num": score_prev_num,
            "inv_total_den": inv_total_den,
        }
    return Y_out


class FLAREAutoregressiveStablemaxPyTorch(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Q_dec, K_dec, scale=None, chunk_size=None, power: float = 2.0):
        compute_dtype = torch.float32
        if os.environ.get("FLARE_PYTORCH_MATCH_REFERENCE", "") == "1":
            compute_dtype = Q.dtype
        prep = _prepare_stablemax_custom_inputs(
            Q,
            K,
            V,
            Q_dec=Q_dec,
            K_dec=K_dec,
            scale=scale,
            chunk_size=chunk_size,
            power=power,
            compute_dtype=compute_dtype,
        )
        if prep["N"] == 0:
            ctx.empty = True
            ctx.b = prep["B"]
            ctx.q_dtype = Q.dtype
            ctx.k_dtype = K.dtype
            ctx.v_dtype = V.dtype
            ctx.q_dec_dtype = prep["Q_dec_resolved"].dtype
            ctx.k_dec_dtype = prep["K_dec_resolved"].dtype
            ctx.n = 0
            ctx.h = prep["H"]
            ctx.m = prep["M"]
            ctx.d_score = prep["D_score"]
            ctx.d_value = prep["D_value"]
            ctx.padded_len = 0
            ctx.c = prep["C"]
            ctx.nc = 0
            ctx.bhnc = 0
            ctx.scale = prep["scale"]
            ctx.power = prep["power"]
            ctx.save_for_backward()
            return torch.empty((prep["B"], 0, prep["H"], prep["D_value"]), device=prep["device"], dtype=prep["out_dtype"])
        ctx.empty = False
        ctx.scale = prep["scale"]
        ctx.power = prep["power"]
        ctx.chunk_size = prep["C"]
        ctx.compute_dtype = compute_dtype
        ctx.q_dtype = Q.dtype
        ctx.k_dtype = K.dtype
        ctx.v_dtype = V.dtype
        ctx.q_dec_dtype = prep["Q_dec_resolved"].dtype
        ctx.k_dec_dtype = prep["K_dec_resolved"].dtype
        ctx.n = prep["N"]
        ctx.h = prep["H"]
        ctx.d_score = prep["D_score"]
        ctx.d_value = prep["D_value"]
        ctx.padded_len = prep["PADDED_LEN"]
        ctx.c = prep["C"]
        ctx.nc = prep["NC"]
        ctx.bhnc = prep["BHNC"]
        Y_out, cache = _forward_stablemax_custom_prepared(prep, return_cache=True)
        ctx.save_for_backward(
            cache["Q_f"],
            cache["Kc"],
            cache["Vc"],
            cache["Q_dec_c"],
            cache["K_dec_f"],
            cache["score_chunk"],
            cache["stable_score_chunk"],
            cache["decode_probs"],
            cache["score_prev_num"],
            cache["inv_total_den"],
        )
        return Y_out

    @staticmethod
    def backward(ctx, dY):
        """Backward for the chunked stablemax prefill path.

        High-level differentiation structure
        ------------------------------------
        The forward can be written as

            Y = prefix_out + local_out

        with

            prefix_out[c] = sum_m R[c, m] * prev_num[m]
            local_out[c]  = sum_tau local_mix[c, tau] * V[tau]
            local_mix     = tril(R @ W^T)
            R             = decode_probs / total_den

        where:
        - `W = stable_score_chunk`
        - `prev_num` and `prev_den` are strict chunk prefixes of encoder statistics
        - `total_den = prev_den + cumsum(W, dim=token)`

        The backward therefore naturally factors into:
        1. Differentiate the output wrt `R`, `W`, `prev_num`, and `V`.
        2. Differentiate `R = decode_probs / total_den`.
        3. Backprop through the chunk-prefix scans producing `prev_num` / `prev_den`.
        4. Backprop through the stablemax transform and the encoder/decoder projections.

        Saved tensor shapes
        -------------------
        - `Q_f`:                [H, M, D_score]
        - `Kc`:                 [B, H, NC, C, D_score]
        - `Vc`:                 [B, H, NC, C, D_value]
        - `Q_dec_c`:            [B, H, NC, C, D_score]
        - `K_dec_f`:            [B, H, M, D_score]
        - `score_chunk`:        [B, H, NC, C, M]
        - `stable_score_chunk`: [B, H, NC, C, M]
        - `decode_probs`:       [B, H, NC, C, M]
        - `score_prev_num`:     [B, H, NC, M, D_value]
        - `inv_total_den`:      [B, H, NC, C, M]

        Incoming gradient:
        - `dY`: [B, N, H, D_value]
        """
        if getattr(ctx, "empty", False):
            B = ctx.b
            H = ctx.h
            M = ctx.m
            N = ctx.n
            D_score = ctx.d_score
            D_value = ctx.d_value
            empty_q = torch.zeros((H, M, D_score), device=dY.device, dtype=ctx.q_dtype)
            empty_kv = torch.zeros((B, N, H, D_score), device=dY.device, dtype=ctx.k_dtype)
            empty_v = torch.zeros((B, N, H, D_value), device=dY.device, dtype=ctx.v_dtype)
            empty_q_dec = torch.zeros((B, N, H, D_score), device=dY.device, dtype=ctx.q_dec_dtype)
            empty_k_dec = torch.zeros((H, M, D_score), device=dY.device, dtype=ctx.k_dec_dtype)
            return empty_q, empty_kv, empty_v, empty_q_dec, empty_k_dec, None, None, None
        (
            Q_f,
            Kc,
            Vc,
            Q_dec_c,
            K_dec_f,
            score_chunk,
            stable_score_chunk,
            decode_probs,
            score_prev_num,
            inv_total_den,
        ) = ctx.saved_tensors
        B = Kc.size(0)
        H = ctx.h
        N = ctx.n
        C = ctx.c
        NC = ctx.nc
        M = Q_f.size(1)
        D_score = ctx.d_score
        D_value = ctx.d_value
        PADDED_LEN = ctx.padded_len
        BHNC = ctx.bhnc
        scale_f = ctx.scale
        power = ctx.power
        device = Kc.device

        if N == 0:
            empty_q = torch.zeros((H, M, D_score), device=device, dtype=ctx.q_dtype)
            empty_kv = torch.zeros((B, N, H, D_score), device=device, dtype=ctx.k_dtype)
            empty_v = torch.zeros((B, N, H, D_value), device=device, dtype=ctx.v_dtype)
            empty_q_dec = torch.zeros((B, N, H, D_score), device=device, dtype=ctx.q_dec_dtype)
            empty_k_dec = torch.zeros((H, M, D_score), device=device, dtype=ctx.k_dec_dtype)
            return empty_q, empty_kv, empty_v, empty_q_dec, empty_k_dec, None, None, None

        stable_score_grad = _stablemax_score_transform_grad(score_chunk, power=power)

        # Reconstruct the forward-side encoder statistics needed by backward.
        #
        # Shapes:
        # - score_chunk_den: [B, H, NC, M]
        # - score_chunk_num: [B, H, NC, M, D_value]
        # - score_prev_den:  [B, H, NC, M]
        score_chunk_den = stable_score_chunk.sum(dim=3)
        score_chunk_num = torch.bmm(
            stable_score_chunk.reshape(BHNC, C, M).transpose(1, 2),
            Vc.reshape(BHNC, C, D_value),
        ).reshape(B, H, NC, M, D_value)
        score_prev_den = torch.cumsum(score_chunk_den, dim=2) - score_chunk_den
        decode_over_den = decode_probs * inv_total_den

        # local_mix[c, tau] = sum_m decode_over_den[c, m] * stable_score_chunk[tau, m],
        # masked so only tau <= c contributes.
        # Shape: [B, H, NC, C, C]
        causal_mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool)).view(1, 1, 1, C, C)
        local_mix = torch.matmul(decode_over_den, stable_score_chunk.transpose(-1, -2))
        local_mix = torch.where(causal_mask, local_mix, torch.zeros_like(local_mix))

        # Re-pack dY into chunk layout to match the forward intermediates.
        # dYc shape: [B, H, NC, C, D_value]
        dY_f = dY.to(torch.float32)
        if PADDED_LEN != N:
            dY_f = torch.cat(
                [dY_f, torch.zeros((B, PADDED_LEN - N, H, D_value), device=device, dtype=torch.float32)],
                dim=1,
        )
        dYc = dY_f.permute(0, 2, 1, 3).reshape(B, H, NC, C, D_value).contiguous()

        # Step 1: differentiate Y = prefix_out + local_out.
        #
        # prefix_out = einsum(decode_over_den, score_prev_num)
        # local_out  = local_mix @ V
        #
        # This yields gradients for:
        # - decode_over_den: [B, H, NC, C, M]
        # - prev_num:        [B, H, NC, M, D_value]
        # - local_mix:       [B, H, NC, C, C]
        # - V:               [B, H, NC, C, D_value]
        grad_decode_over_den = torch.einsum("bhncd,bhnmd->bhncm", dYc, score_prev_num)
        grad_prev_num = torch.einsum("bhncm,bhncd->bhnmd", decode_over_den, dYc)
        grad_local_mix = torch.matmul(dYc, Vc.transpose(-1, -2))
        grad_local_mix = torch.where(causal_mask, grad_local_mix, torch.zeros_like(grad_local_mix))
        grad_V_phase3 = torch.matmul(local_mix.transpose(-1, -2).reshape(BHNC, C, C), dYc.reshape(BHNC, C, D_value)).reshape(B, H, NC, C, D_value)

        # local_mix = tril(decode_over_den @ stable_score_chunk^T)
        #
        # Differentiate the masked [C, C] matmul:
        # - extra grad into decode_over_den from mixing with stable_score_chunk
        # - phase-3 grad into stable_score_chunk from mixing with decode_over_den
        grad_decode_over_den = grad_decode_over_den + torch.matmul(grad_local_mix, stable_score_chunk)
        grad_w_phase3 = torch.matmul(grad_local_mix.transpose(-1, -2), decode_over_den)

        # Step 2: differentiate R = decode_probs / total_den.
        #
        # Elementwise:
        #   R = P * inv_total_den
        # so:
        #   dP          += dR * inv_total_den
        #   dtotal_den  += -(dR * P) / total_den^2
        #
        # Shapes:
        # - grad_decode_probs: [B, H, NC, C, M]
        # - grad_total_den:    [B, H, NC, C, M]
        grad_decode_probs = grad_decode_over_den * inv_total_den
        grad_total_den = -(grad_decode_over_den * decode_probs) * inv_total_den.square()

        # total_den = prev_den.unsqueeze(token) + cumsum(stable_score_chunk, dim=token)
        #
        # Gradient wrt prev_den is the sum over token positions in the chunk.
        # Gradient wrt the within-chunk stable weights is a reverse cumsum over token
        # positions, because each encoder token contributes to all later targets.
        grad_prev_den = grad_total_den.sum(dim=3)
        grad_w_phase3 = grad_w_phase3 + torch.flip(torch.cumsum(torch.flip(grad_total_den, dims=(3,)), dim=3), dims=(3,))

        # Step 3: differentiate decoder softmax.
        #
        # For P = softmax(L), the Jacobian-vector product is:
        #   dL = P * (dP - <dP, P>)
        #
        # Shape: [B, H, NC, C, M]
        softmax_dot = (grad_decode_probs * decode_probs).sum(dim=-1, keepdim=True)
        grad_score_dec = decode_probs * (grad_decode_probs - softmax_dot)

        # Step 4: backprop through chunk-prefix scans.
        #
        # Forward used strict prefixes:
        #   prev_den[n] = sum_{n' < n} score_chunk_den[n']
        #   prev_num[n] = sum_{n' < n} score_chunk_num[n']
        #
        # Their reverse-mode adjoints are strict suffix sums:
        #   dscore_chunk_*[n] = sum_{n' > n} dprev_*[n']
        #
        # Shapes:
        # - grad_score_chunk_den: [B, H, NC, M]
        # - grad_score_chunk_num: [B, H, NC, M, D_value]
        grad_score_chunk_den = torch.flip(torch.cumsum(torch.flip(grad_prev_den, dims=(2,)), dim=2), dims=(2,)) - grad_prev_den
        grad_score_chunk_num = torch.flip(torch.cumsum(torch.flip(grad_prev_num, dims=(2,)), dim=2), dims=(2,)) - grad_prev_num

        # Step 5: combine all encoder-weight gradients.
        #
        # stable_score_chunk participates in three places:
        # 1. directly in the local chunk mixing (`grad_w_phase3`)
        # 2. in score_chunk_den = sum_c stable_score_chunk
        # 3. in score_chunk_num = stable_score_chunk^T @ V
        #
        # Resulting shapes:
        # - grad_w_total: [B, H, NC, C, M]
        # - grad_V_total: [B, H, NC, C, D_value]
        grad_w_total = grad_w_phase3 + grad_score_chunk_den.unsqueeze(3)
        grad_w_total = grad_w_total + torch.einsum("bhnmd,bhncd->bhncm", grad_score_chunk_num, Vc)
        grad_V_total = grad_V_phase3 + torch.einsum("bhncm,bhnmd->bhncd", stable_score_chunk, grad_score_chunk_num)

        # Step 6: differentiate the stablemax transform and the linear score projections.
        #
        # grad_score_enc has shape [B, H, NC, C, M].
        # It contracts with:
        # - encoder queries Q_f: [H, M, D_score]
        # - encoder keys   Kc:   [B, H, NC, C, D_score]
        #
        # grad_score_dec has the same [B, H, NC, C, M] shape and contracts with:
        # - decoder queries Q_dec_c: [B, H, NC, C, D_score]
        # - decoder keys   K_dec_f:  [B, H, M, D_score]
        grad_score_enc = grad_w_total * stable_score_grad
        grad_Q_dec_c = scale_f * torch.einsum("bhncm,bhmd->bhncd", grad_score_dec, K_dec_f)
        grad_K_dec = (scale_f * torch.einsum("bhncm,bhncd->bhmd", grad_score_dec, Q_dec_c)).sum(dim=0)

        # Unpack chunked gradients back to the public tensor layouts:
        # - grad_Q:     [H, M, D_score]
        # - grad_K:     [B, N, H, D_score]
        # - grad_V:     [B, N, H, D_value]
        # - grad_Q_dec: [B, N, H, D_score]
        # - grad_K_dec: [H, M, D_score]
        grad_Q = scale_f * torch.einsum("bhncm,bhncd->hmd", grad_score_enc, Kc)
        grad_Kc = scale_f * torch.einsum("bhncm,hmd->bhncd", grad_score_enc, Q_f)
        grad_K = grad_Kc.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_score)[:, :N, :, :]
        grad_V = grad_V_total.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_value)[:, :N, :, :]
        grad_Q_dec = grad_Q_dec_c.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_score)[:, :N, :, :]
        return (
            grad_Q.to(ctx.q_dtype),
            grad_K.to(ctx.k_dtype),
            grad_V.to(ctx.v_dtype),
            grad_Q_dec.to(ctx.q_dec_dtype),
            grad_K_dec.to(ctx.k_dec_dtype),
            None,
            None,
            None,
        )


def flare_autoregressive_stablemax_pytorch_custom(
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
    state: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    return_state: bool = False,
):
    Q_dec_resolved, K_dec_resolved, _, _, _ = _resolve_flare_causal_decode_inputs(
        Q, K, Q_dec, K_dec
    )
    if profile:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch_custom does not support profile=True")
    if state is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch_custom does not support recurrent state")
    if attention_mask is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch_custom does not support attention_mask")
    if return_state:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch_custom does not support return_state=True")
    return FLAREAutoregressiveStablemaxPyTorch.apply(Q, K, V, Q_dec_resolved, K_dec_resolved, scale, chunk_size, power)

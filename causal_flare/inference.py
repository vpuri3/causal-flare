from ._common import *
from .chunked import *
from .chunked import (
    _autotune_arg,
    _get_chunked_forward_config,
    _maybe_limit_autotune_configs_for_tests,
    _profiled_call,
    _refresh_profile_totals,
    _run_chunked_output_phase,
    _run_chunked_prefix_phase,
    _run_chunked_prepare_phase,
)
from .torch import _resolve_flare_causal_decode_inputs as _resolve_inference_decode_inputs


def _prune_decode_step_autotune_configs(configs, named_args, **kwargs):
    m = int(_autotune_arg(named_args, kwargs, "M"))
    d_value = int(_autotune_arg(named_args, kwargs, "D_VALUE"))
    d_score = int(_autotune_arg(named_args, kwargs, "D_SCORE"))
    keep = []
    for cfg in configs:
        block_m = int(cfg.kwargs["BLOCK_M"])
        block_d = int(cfg.kwargs["BLOCK_D"])
        block_k = int(cfg.kwargs["BLOCK_K"])
        if block_m < m or block_d > d_value or block_k > d_score:
            continue
        keep.append(cfg)
    return _maybe_limit_autotune_configs_for_tests(
        keep,
        score_fn=lambda cfg: (
            int(cfg.kwargs["BLOCK_M"]),
            -int(cfg.kwargs["BLOCK_D"]),
            -int(cfg.kwargs["BLOCK_K"]),
            int(cfg.num_warps),
            int(cfg.num_stages),
        ),
        reverse=False,
    )


_INFERENCE_DECODE_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_D": 16, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 16, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 32, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 32, "BLOCK_D": 32, "BLOCK_K": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 32, "BLOCK_K": 16}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 32, "BLOCK_K": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_D": 64, "BLOCK_K": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_D": 32, "BLOCK_K": 32}, num_warps=2, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_D": 64, "BLOCK_K": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_D": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 256, "BLOCK_D": 64, "BLOCK_K": 32}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 256, "BLOCK_D": 64, "BLOCK_K": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 256, "BLOCK_D": 128, "BLOCK_K": 64}, num_warps=4, num_stages=1),
]

def _init_flare_recurrent_state(
    batch_size: int,
    num_heads: int,
    num_latents: int,
    value_head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:

    SHAPE_D = (batch_size, num_heads, num_latents)
    SHAPE_U = (batch_size, num_heads, num_latents, value_head_dim)

    return {
        "m": torch.full( SHAPE_D, -torch.inf, device=device, dtype=dtype),
        "d": torch.zeros(SHAPE_D, device=device, dtype=dtype),
        "u": torch.zeros(SHAPE_U, device=device, dtype=dtype),
    }

def _canonicalize_flare_state(
    state: dict[str, torch.Tensor] | None,
    batch_size: int,
    num_heads: int,
    num_latents: int,
    value_head_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if state is None:
        return _init_flare_recurrent_state(
            batch_size=batch_size,
            num_heads=num_heads,
            num_latents=num_latents,
            value_head_dim=value_head_dim,
            device=device,
            dtype=torch.float32,
        )

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


def _canonicalize_kv_for_prefill(
    K: torch.Tensor,
    V: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if K.dim() != 4 or V.dim() != 4:
        raise ValueError(f"Prefill expects K/V to be [B, T, H, D]. Got K={tuple(K.shape)}, V={tuple(V.shape)}")
    if K.shape[:3] != V.shape[:3]:
        raise ValueError(
            "Prefill expects K and V to agree on [B, T, H]. "
            f"Got K={tuple(K.shape)}, V={tuple(V.shape)}"
        )
    return K, V


def _canonicalize_kv_for_decode(
    K: torch.Tensor,
    V: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if K.dim() == 4 and K.shape[1] == 1:
        K = K[:, 0]
    if V.dim() == 4 and V.shape[1] == 1:
        V = V[:, 0]
    if K.dim() != 3 or V.dim() != 3:
        raise ValueError(f"Decode expects K/V as [B, H, D] or [B, 1, H, D]. Got K={tuple(K.shape)}, V={tuple(V.shape)}")
    if K.shape[:2] != V.shape[:2]:
        raise ValueError(
            "Decode expects K and V to agree on [B, H]. "
            f"Got K={tuple(K.shape)}, V={tuple(V.shape)}"
        )
    return K, V


def _canonicalize_q_dec_for_decode(
    Q_dec: torch.Tensor | None,
    K_step: torch.Tensor,
    K_input: torch.Tensor | None = None,
    K_prefill: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if Q_dec is None:
        return None

    if Q_dec is K_step or (K_input is not None and Q_dec is K_input):
        return K_prefill if K_prefill is not None else K_step[:, None, :, :]

    if Q_dec.dim() == 3:
        if Q_dec.shape != K_step.shape:
            raise ValueError(
                "Decode Q_dec must match K decode-step shape. "
                f"Got Q_dec.shape={tuple(Q_dec.shape)} and K.shape={tuple(K_step.shape)}"
            )
        return Q_dec[:, None, :, :]

    if Q_dec.dim() == 4:
        if Q_dec.shape[0] != K_step.shape[0] or Q_dec.shape[2] != K_step.shape[1] or Q_dec.shape[3] != K_step.shape[2]:
            raise ValueError(
                "Decode Q_dec [B, 1, H, D] must match K [B, H, D]. "
                f"Got Q_dec.shape={tuple(Q_dec.shape)} and K.shape={tuple(K_step.shape)}"
            )
        if Q_dec.shape[1] != 1:
            raise ValueError(
                "Decode Q_dec with rank-4 must have sequence length 1. "
                f"Got Q_dec.shape={tuple(Q_dec.shape)}"
            )
        return Q_dec

    raise ValueError(
        "Decode Q_dec must be [B, H, D] or [B, 1, H, D]. "
        f"Got Q_dec.shape={tuple(Q_dec.shape)}"
    )


def _merge_flare_stats(
    m_a: torch.Tensor,
    d_a: torch.Tensor,
    u_a: torch.Tensor,
    m_b: torch.Tensor,
    d_b: torch.Tensor,
    u_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m_a = m_a.to(torch.float32)
    d_a = d_a.to(torch.float32)
    u_a = u_a.to(torch.float32)
    m_b = m_b.to(torch.float32)
    d_b = d_b.to(torch.float32)
    u_b = u_b.to(torch.float32)

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


def flare_prefill_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor] | None = None,
    scale: float = 1.0,
    attention_mask: torch.Tensor | None = None,
    Q_dec: torch.Tensor | None = None,
    K_dec: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    K, V = _canonicalize_kv_for_prefill(K, V)
    if Q.dim() != 3:
        raise ValueError(f"Q must be [H, M, D]. Got Q={tuple(Q.shape)}")
    B, T, H, D_score = K.shape
    D_value = V.shape[-1]
    Hq, M, Dq = Q.shape
    if Hq != H or Dq != D_score:
        raise ValueError(f"Incompatible Q/K shapes. Q={tuple(Q.shape)}, K={tuple(K.shape)}")
    if attention_mask is not None and attention_mask.shape != (B, T):
        raise ValueError(f"attention_mask must be [B, T]. Got {tuple(attention_mask.shape)}")
    Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_inference_decode_inputs(
        Q, K, Q_dec, K_dec
    )

    out_dtype = V.dtype
    Q_f = Q.float().unsqueeze(0).expand(B, -1, -1, -1)
    K_f = K.float().permute(0, 2, 1, 3).contiguous()
    V_f = V.float().permute(0, 2, 1, 3).contiguous()
    Q_dec_f = Q_dec.float().permute(0, 2, 1, 3).contiguous() if separate_Q_dec else K_f
    K_dec_f = K_dec.float().unsqueeze(0).expand(B, -1, -1, -1) if separate_K_dec else Q_f

    st = _canonicalize_flare_state(
        state=state,
        batch_size=B,
        num_heads=H,
        num_latents=M,
        value_head_dim=D_value,
        device=K.device,
    )
    m = st["m"]
    d = st["d"]
    u = st["u"]

    Y = torch.empty((B, H, T, D_value), device=K.device, dtype=torch.float32)
    for t in range(T):
        k_t = K_f[:, :, t, :]
        v_t = V_f[:, :, t, :]
        s_t = torch.einsum("bhmd,bhd->bhm", Q_f, k_t) * float(scale)
        if attention_mask is not None:
            valid = attention_mask[:, t].to(torch.bool).view(B, 1, 1)
            s_t = torch.where(valid, s_t, torch.full_like(s_t, -float("inf")))
        else:
            valid = torch.ones((B, 1, 1), device=K.device, dtype=torch.bool)

        m_new = torch.maximum(m, s_t)
        is_m_inf = torch.isinf(m) & (m < 0)
        is_m_new_inf = torch.isinf(m_new) & (m_new < 0)
        m_new_safe = torch.where(is_m_new_inf, torch.zeros_like(m_new), m_new)
        gamma = torch.where(
            is_m_inf & is_m_new_inf,
            torch.ones_like(m_new),
            torch.where(is_m_inf, torch.zeros_like(m_new), torch.exp(m - m_new_safe)),
        )
        eta = torch.where(is_m_new_inf, torch.zeros_like(s_t), torch.exp(s_t - m_new_safe))

        d = d * gamma + eta
        u = u * gamma[..., None] + eta[..., None] * v_t[:, :, None, :]
        m = m_new

        d_safe = torch.where(d > 0, d, torch.ones_like(d))
        z_t = u / d_safe[..., None]

        if weight_sharing_enc_dec:
            a_t = s_t
        else:
            q_t_dec = Q_dec_f[:, :, t, :]
            a_t = torch.einsum("bhd,bhmd->bhm", q_t_dec, K_dec_f) * float(scale)

        s_decode = torch.where(valid, a_t, torch.zeros_like(a_t))
        alpha = torch.softmax(s_decode, dim=-1) * valid.float()
        y_t = torch.einsum("bhm,bhmd->bhd", alpha, z_t)
        Y[:, :, t, :] = y_t

    next_state = {"m": m, "d": d, "u": u}
    return Y.permute(0, 2, 1, 3).to(out_dtype), next_state


def flare_decode_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor],
    scale: float = 1.0,
    attention_mask: torch.Tensor | None = None,
    Q_dec: torch.Tensor | None = None,
    K_dec: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    K_input = K
    K, V = _canonicalize_kv_for_decode(K, V)
    B, _, _ = K.shape
    if attention_mask is not None:
        if attention_mask.dim() == 2 and attention_mask.shape[1] == 1:
            attention_mask = attention_mask[:, 0]
        if attention_mask.dim() != 1 or attention_mask.shape[0] != B:
            raise ValueError(f"Decode attention_mask must be [B] or [B, 1]. Got {tuple(attention_mask.shape)}")
        attention_mask = attention_mask[:, None]

    K_prefill = K[:, None, :, :]
    Q_dec_prefill = _canonicalize_q_dec_for_decode(Q_dec, K, K_input=K_input, K_prefill=K_prefill)

    y, next_state = flare_prefill_pytorch(
        Q=Q,
        K=K_prefill,
        V=V[:, None, :, :],
        Q_dec=Q_dec_prefill,
        K_dec=K_dec,
        state=state,
        scale=scale,
        attention_mask=attention_mask,
    )
    return y, next_state


#======================================================================#
# Triton Cached Implementations
#======================================================================#

@triton.autotune(
    configs=_INFERENCE_DECODE_AUTOTUNE_CONFIGS,
    key=["M", "D_SCORE", "D_VALUE", "HAS_MASK"],
    prune_configs_by={"early_config_prune": _prune_decode_step_autotune_configs},
    reset_to_zero=["Y_ptr"],
    restore_value=["M_ptr", "D_ptr", "U_ptr"],
)
@triton.jit
def flare_recurrent_step_kernel(
    Q_ptr, K_ptr, V_ptr, Q_dec_ptr, K_dec_ptr,
    M_ptr, D_ptr, U_ptr,
    Y_ptr, Mask_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_qdb, stride_qdh, stride_qdd,
    stride_kdh_dec, stride_kdm_dec, stride_kdd_dec,
    stride_mb, stride_mh, stride_mm,
    stride_db, stride_dh, stride_dm,
    stride_ub, stride_uh, stride_um, stride_ud,
    stride_yb, stride_yh, stride_yd,
    stride_mask_b,
    B, H, M, D_SCORE: tl.constexpr, D_VALUE: tl.constexpr, scale,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    WEIGHT_SHARING_ENC_DEC: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    bh = B * H
    if pid_bh >= bh:
        return
    b = pid_bh // H
    h = pid_bh - b * H

    m_offsets = tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    mask_md = mask_m[:, None] & mask_d[None, :]

    valid_t = tl.full((), True, tl.int1)
    if HAS_MASK:
        mask_val = tl.load(Mask_ptr + b * stride_mask_b)
        valid_t = mask_val != 0

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    q_dec_base = Q_dec_ptr + b * stride_qdb + h * stride_qdh
    k_dec_base = K_dec_ptr + h * stride_kdh_dec
    m_base = M_ptr + b * stride_mb + h * stride_mh
    d_base = D_ptr + b * stride_db + h * stride_dh
    u_base = U_ptr + b * stride_ub + h * stride_uh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    v = tl.load(v_base + d_offsets * stride_vd, mask=mask_d, other=0.0).to(tl.float32)

    m_state = tl.load(m_base + m_offsets * stride_mm, mask=mask_m, other=-float("inf")).to(tl.float32)
    d_state = tl.load(d_base + m_offsets * stride_dm, mask=mask_m, other=0.0).to(tl.float32)
    u_state = tl.load(
        u_base + m_offsets[:, None] * stride_um + d_offsets[None, :] * stride_ud,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)

    # Each output-D tile owns only [M, BLOCK_D] of u/y, but the score path still
    # needs the full q_m^T k reduction over all D. We therefore replay that
    # reduction locally across BLOCK_K slices and only let d_block=0 write the
    # shared m/d scalars to avoid races.
    s_raw = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in tl.static_range(0, D_SCORE, BLOCK_K):
        d_offsets_k = k0 + tl.arange(0, BLOCK_K)
        mask_d_k = d_offsets_k < D_SCORE
        q_k = tl.load(
            q_base + m_offsets[:, None] * stride_qm + d_offsets_k[None, :] * stride_qd,
            mask=mask_m[:, None] & mask_d_k[None, :],
            other=0.0,
        ).to(tl.float32)
        k_k = tl.load(k_base + d_offsets_k * stride_kd, mask=mask_d_k, other=0.0).to(tl.float32)
        s_raw += tl.sum(q_k * k_k[None, :], axis=1)
    s_raw = s_raw * scale
    s_t = tl.where(valid_t & mask_m, s_raw, -float("inf"))

    m_new = tl.maximum(m_state, s_t)
    is_m_state_inf = m_state == -float("inf")
    is_m_new_inf = m_new == -float("inf")
    m_new_safe = tl.where(is_m_new_inf, 0.0, m_new)
    gamma = tl.where(
        is_m_state_inf & is_m_new_inf,
        1.0,
        tl.where(is_m_state_inf, 0.0, tl.exp(m_state - m_new_safe)),
    )
    eta = tl.where(is_m_new_inf, 0.0, tl.exp(s_t - m_new_safe))

    d_state = d_state * gamma + eta
    u_state = u_state * gamma[:, None] + eta[:, None] * v[None, :]
    m_state = m_new

    d_safe = tl.where(d_state > 0, d_state, 1.0)
    z = u_state / d_safe[:, None]

    if WEIGHT_SHARING_ENC_DEC:
        a_raw = s_raw
    else:
        a_raw = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in tl.static_range(0, D_SCORE, BLOCK_K):
            d_offsets_k = k0 + tl.arange(0, BLOCK_K)
            mask_d_k = d_offsets_k < D_SCORE
            q_dec_k = tl.load(q_dec_base + d_offsets_k * stride_qdd, mask=mask_d_k, other=0.0).to(tl.float32)
            k_dec_k = tl.load(
                k_dec_base + m_offsets[:, None] * stride_kdm_dec + d_offsets_k[None, :] * stride_kdd_dec,
                mask=mask_m[:, None] & mask_d_k[None, :],
                other=0.0,
            ).to(tl.float32)
            a_raw += tl.sum(k_dec_k * q_dec_k[None, :], axis=1)
        a_raw = a_raw * scale

    s_decode = tl.where(valid_t & mask_m, a_raw, 0.0)
    s_max = tl.max(s_decode, axis=0)
    s_exp = tl.exp(s_decode - s_max)
    s_exp = tl.where(valid_t & mask_m, s_exp, 0.0)
    s_sum = tl.sum(s_exp, axis=0)
    s_sum = tl.where(s_sum > 0, s_sum, 1.0)
    alpha = s_exp / s_sum

    y = tl.sum(alpha[:, None] * z, axis=0)
    y = tl.where(valid_t, y, 0.0)

    tl.store(m_base + m_offsets * stride_mm, m_state, mask=(pid_d == 0) & mask_m)
    tl.store(d_base + m_offsets * stride_dm, d_state, mask=(pid_d == 0) & mask_m)
    tl.store(
        u_base + m_offsets[:, None] * stride_um + d_offsets[None, :] * stride_ud,
        u_state,
        mask=mask_md,
    )
    tl.store(y_base + d_offsets * stride_yd, y, mask=mask_d)


def flare_decode_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor] | None = None,
    scale: float = 1.0,
    input_precision: str | None = None,
    attention_mask: torch.Tensor | None = None,
    Q_dec: torch.Tensor | None = None,
    K_dec: torch.Tensor | None = None,
    profile: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[tuple[torch.Tensor, dict[str, torch.Tensor]], dict[str, object]]:
    if not Q.is_cuda:
        raise RuntimeError(
            "flare_decode requires CUDA tensors. "
        )

    K_input = K
    K, V = _canonicalize_kv_for_decode(K, V)
    _ = _normalize_input_precision(input_precision, None)
    B, H, D_score = K.shape
    D_value = V.shape[-1]
    Hq, M, Dq = Q.shape
    if Hq != H or Dq != D_score:
        raise ValueError(f"Incompatible Q/K shapes. Q={tuple(Q.shape)}, K={tuple(K.shape)}")
    K_resolve = K[:, None, :, :]
    Q_dec_resolve = _canonicalize_q_dec_for_decode(Q_dec, K, K_input=K_input, K_prefill=K_resolve)
    Q_dec_resolve, K_dec_resolve, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_inference_decode_inputs(
        Q, K_resolve, Q_dec_resolve, K_dec
    )
    if attention_mask is not None:
        if attention_mask.dim() == 2 and attention_mask.shape[1] == 1:
            attention_mask = attention_mask[:, 0]
        if attention_mask.dim() != 1 or attention_mask.shape[0] != B:
            raise ValueError(f"Decode attention_mask must be [B] or [B, 1]. Got {tuple(attention_mask.shape)}")
        attention_mask = attention_mask.to(device=K.device, dtype=torch.int32).contiguous()

    st = _canonicalize_flare_state(
        state=state,
        batch_size=B,
        num_heads=H,
        num_latents=M,
        value_head_dim=D_value,
        device=K.device,
    )
    m_state = st["m"].contiguous()
    d_state = st["d"].contiguous()
    u_state = st["u"].contiguous()
    profile_data = {"forward": {}} if profile else None
    forward_timings = profile_data["forward"] if profile_data is not None else None

    def prepare_decode_io():
        q = Q.contiguous().float()
        k = K.contiguous().float()
        v = V.contiguous().float()
        if weight_sharing_enc_dec:
            q_dec_step = k
            k_dec_latent = q
        else:
            q_dec_step = Q_dec_resolve[:, 0, :, :].contiguous().float() if separate_Q_dec else k
            k_dec_latent = K_dec_resolve.contiguous().float() if separate_K_dec else q
        y = torch.empty((B, H, D_value), device=K.device, dtype=torch.float32)
        return q, k, v, q_dec_step, k_dec_latent, y

    q, k, v, q_dec_step, k_dec_latent, y = _profiled_call(
        K.device,
        forward_timings,
        "decode_prepare_io",
        prepare_decode_io,
    )

    grid = lambda meta: (B * H, triton.cdiv(D_value, meta["BLOCK_D"]))
    def launch_decode():
        if attention_mask is None:
            return flare_recurrent_step_kernel[grid](
                q, k, v, q_dec_step, k_dec_latent,
                m_state, d_state, u_state,
                y, y,  # dummy mask ptr for HAS_MASK=False
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                q_dec_step.stride(0), q_dec_step.stride(1), q_dec_step.stride(2),
                k_dec_latent.stride(0), k_dec_latent.stride(1), k_dec_latent.stride(2),
                m_state.stride(0), m_state.stride(1), m_state.stride(2),
                d_state.stride(0), d_state.stride(1), d_state.stride(2),
                u_state.stride(0), u_state.stride(1), u_state.stride(2), u_state.stride(3),
                y.stride(0), y.stride(1), y.stride(2),
                0,
                B, H, M, D_score, D_value, float(scale),
                HAS_MASK=False,
                WEIGHT_SHARING_ENC_DEC=weight_sharing_enc_dec,
            )
        return flare_recurrent_step_kernel[grid](
            q, k, v, q_dec_step, k_dec_latent,
            m_state, d_state, u_state,
            y, attention_mask,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            q_dec_step.stride(0), q_dec_step.stride(1), q_dec_step.stride(2),
            k_dec_latent.stride(0), k_dec_latent.stride(1), k_dec_latent.stride(2),
            m_state.stride(0), m_state.stride(1), m_state.stride(2),
            d_state.stride(0), d_state.stride(1), d_state.stride(2),
            u_state.stride(0), u_state.stride(1), u_state.stride(2), u_state.stride(3),
            y.stride(0), y.stride(1), y.stride(2),
            attention_mask.stride(0),
            B, H, M, D_score, D_value, float(scale),
            HAS_MASK=True,
            WEIGHT_SHARING_ENC_DEC=weight_sharing_enc_dec,
        )

    _profiled_call(K.device, forward_timings, "flare_decode_step", launch_decode)

    next_state = {"m": m_state, "d": d_state, "u": u_state}
    y_out = _profiled_call(K.device, forward_timings, "decode_output_cast", lambda: y[:, None, :, :].to(V.dtype))
    if profile:
        _refresh_profile_totals(profile_data)
        return (y_out, next_state), profile_data
    return y_out, next_state


def flare_prefill_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor] | None = None,
    scale: float = 1.0,
    input_precision: str | None = None,
    attention_mask: torch.Tensor | None = None,
    Q_dec: torch.Tensor | None = None,
    K_dec: torch.Tensor | None = None,
    profile: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[tuple[torch.Tensor, dict[str, torch.Tensor]], dict[str, object]]:
    if not Q.is_cuda:
        raise RuntimeError("flare_prefill requires CUDA tensors.")

    K, V = _canonicalize_kv_for_prefill(K, V)
    B, T, H, D_score = K.shape
    D_value = V.shape[-1]
    if Q.dim() != 3:
        raise ValueError(f"Q must be [H, M, D]. Got Q={tuple(Q.shape)}")
    Hq, M, Dq = Q.shape
    if Hq != H or Dq != D_score:
        raise ValueError(f"Incompatible Q/K shapes. Q={tuple(Q.shape)}, K={tuple(K.shape)}")
    if attention_mask is not None and attention_mask.shape != (B, T):
        raise ValueError(f"attention_mask must be [B, T]. Got {tuple(attention_mask.shape)}")
    Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_inference_decode_inputs(
        Q, K, Q_dec, K_dec
    )
    if T == 0:
        st = _canonicalize_flare_state(
            state=state,
            batch_size=B,
            num_heads=H,
            num_latents=M,
            value_head_dim=D_value,
            device=K.device,
        )
        return torch.empty((B, 0, H, D_value), device=K.device, dtype=V.dtype), st

    if attention_mask is not None:
        # Keep masked prefill semantics identical to recurrent decode while avoiding any PyTorch fallback.
        all_valid = bool((attention_mask != 0).all().item())
        if not all_valid:
            st = _canonicalize_flare_state(
                state=state,
                batch_size=B,
                num_heads=H,
                num_latents=M,
                value_head_dim=D_value,
                device=K.device,
            )
            y_steps = []
            for t in range(T):
                if weight_sharing_enc_dec:
                    q_dec_t = None
                    k_dec_t = None
                else:
                    q_dec_t = Q_dec[:, t, :, :] if separate_Q_dec else None
                    k_dec_t = K_dec if separate_K_dec else None
                y_t, st = flare_decode_triton(
                    Q=Q,
                    K=K[:, t, :, :],
                    V=V[:, t, :, :],
                    state=st,
                    Q_dec=q_dec_t,
                    K_dec=k_dec_t,
                    scale=scale,
                    input_precision=input_precision,
                    attention_mask=attention_mask[:, t],
                )
                y_steps.append(y_t)
            return torch.cat(y_steps, dim=1), st
        attention_mask = None

    st = _canonicalize_flare_state(
        state=state,
        batch_size=B,
        num_heads=H,
        num_latents=M,
        value_head_dim=D_value,
        device=K.device,
    )
    profile_data = {"forward": {}} if profile else None
    forward_timings = profile_data["forward"] if profile_data is not None else None
    m0 = st["m"].reshape(B * H, M)
    d0 = st["d"].reshape(B * H, M)
    u0 = st["u"].reshape(B * H, M, D_value)

    dtype = K.dtype
    cfg = _get_chunked_forward_config(
        M=M,
        N=T,
        score_head_dim=D_score,
        value_head_dim=D_value,
        dtype=dtype,
        input_precision=input_precision,
    )

    q = Q.contiguous()
    k = K.contiguous()
    v = V.contiguous()
    if weight_sharing_enc_dec:
        q_dec_comp = k
        k_dec_comp = q
    else:
        q_dec_comp = Q_dec.contiguous() if separate_Q_dec else k
        k_dec_comp = K_dec.contiguous() if separate_K_dec else q
    chunk_max, chunk_den, chunk_num = _run_chunked_prepare_phase(
        q,
        k,
        v,
        H=H,
        M=M,
        N=T,
        D_score=D_score,
        D_value=D_value,
        scale=float(scale),
        config=cfg,
        kernel_timings=forward_timings,
    )

    prefix_max, prefix_den, prefix_num = _run_chunked_prefix_phase(
        chunk_max, chunk_den, chunk_num,
        M=M,
        D_value=D_value,
        config=cfg,
        kernel_timings=forward_timings,
    )

    m0_expand = m0[:, None, :].expand(-1, cfg["NUM_CHUNKS"], -1)
    d0_expand = d0[:, None, :].expand(-1, cfg["NUM_CHUNKS"], -1)
    u0_expand = u0[:, None, :, :].expand(-1, cfg["NUM_CHUNKS"], -1, -1)
    prefix_max, prefix_den, prefix_num = _profiled_call(
        K.device,
        forward_timings,
        "prefill_merge_prefix_state",
        lambda: _merge_flare_stats(
            m0_expand, d0_expand, u0_expand,
            prefix_max, prefix_den, prefix_num,
        ),
    )
    prefix_max, prefix_den, prefix_num = _profiled_call(
        K.device,
        forward_timings,
        "prefill_prefix_contiguous",
        lambda: (
            prefix_max.contiguous(),
            prefix_den.contiguous(),
            prefix_num.contiguous(),
        ),
    )

    O, _, _ = _run_chunked_output_phase(
        q,
        k,
        v,
        q_dec_comp,
        k_dec_comp,
        prefix_max, prefix_den, prefix_num,
        H=H,
        M=M,
        N=T,
        D_score=D_score,
        D_value=D_value,
        scale=float(scale),
        config=cfg,
        weight_sharing_enc_dec=weight_sharing_enc_dec,
        output_dtype=torch.float32,
        kernel_timings=forward_timings,
    )

    last_prefix_max = prefix_max[:, -1, :]
    last_prefix_den = prefix_den[:, -1, :]
    last_prefix_num = prefix_num[:, -1, :, :]
    last_chunk_max = chunk_max[:, -1, :]
    last_chunk_den = chunk_den[:, -1, :]
    last_chunk_num = chunk_num[:, -1, :, :]
    m_fin, d_fin, u_fin = _profiled_call(
        K.device,
        forward_timings,
        "prefill_merge_final_state",
        lambda: _merge_flare_stats(
            last_prefix_max, last_prefix_den, last_prefix_num,
            last_chunk_max, last_chunk_den, last_chunk_num,
        ),
    )
    next_state = {
        "m": m_fin.reshape(B, H, M).contiguous(),
        "d": d_fin.reshape(B, H, M).contiguous(),
        "u": u_fin.reshape(B, H, M, D_value).contiguous(),
    }

    O_out = _profiled_call(K.device, forward_timings, "prefill_output_cast", lambda: O.to(V.dtype))
    if profile:
        _refresh_profile_totals(profile_data)
        return (O_out, next_state), profile_data
    return O_out, next_state


def flare_prefill(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor] | None = None,
    scale: float = 1.0,
    input_precision: str | None = None,
    attention_mask: torch.Tensor | None = None,
    impl: str = "triton",
    Q_dec: torch.Tensor | None = None,
    K_dec: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if impl == "triton":
        return flare_prefill_triton(
            Q=Q,
            K=K,
            V=V,
            Q_dec=Q_dec,
            K_dec=K_dec,
            state=state,
            scale=scale,
            input_precision=input_precision,
            attention_mask=attention_mask,
        )
    if impl == "pytorch":
        return flare_prefill_pytorch(
            Q=Q,
            K=K,
            V=V,
            Q_dec=Q_dec,
            K_dec=K_dec,
            state=state,
            scale=scale,
            attention_mask=attention_mask,
        )
    raise ValueError(f"Unsupported impl={impl}. Expected 'triton' or 'pytorch'.")


def flare_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor],
    scale: float = 1.0,
    input_precision: str | None = None,
    attention_mask: torch.Tensor | None = None,
    impl: str = "triton",
    Q_dec: torch.Tensor | None = None,
    K_dec: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if impl == "triton":
        return flare_decode_triton(
            Q=Q,
            K=K,
            V=V,
            state=state,
            Q_dec=Q_dec,
            K_dec=K_dec,
            scale=scale,
            input_precision=input_precision,
            attention_mask=attention_mask,
        )
    if impl == "pytorch":
        return flare_decode_pytorch(
            Q=Q,
            K=K,
            V=V,
            state=state,
            Q_dec=Q_dec,
            K_dec=K_dec,
            scale=scale,
            attention_mask=attention_mask,
        )
    raise ValueError(f"Unsupported impl={impl}. Expected 'triton' or 'pytorch'.")


#======================================================================#
# PyTorch Implementations
#======================================================================#

"""Experimental dense autoregressive FLARE variants."""

from causal_flare._common import *
from causal_flare._reference_utils import (
    resolve_flare_causal_decode_inputs as _resolve_flare_causal_decode_inputs,
    validate_flare_qkv_layouts as _validate_flare_qkv_layouts,
)

# NOTE:
# - This file does not implement sequence-length (N-axis) chunking.
# - It is used to prototype inner methods that may later be used inside
#   chunked algorithms.
# - Keep this module self-contained: do not import kernels from training.py.
# - This path is experimental and is not exported from the default package API.


def _flare_recurrent_dense_lse_forward_state(Q, K, V, scale=None, Q_dec=None, K_dec=None):
    B, T, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="Recurrent FLARE")
    scale = _resolve_attn_scale(scale, D_score)
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
    u_state = torch.zeros((B, H, M, D_value), device=Q.device, dtype=torch.float32)
    Z = torch.empty((B, H, T, M, D_value), device=Q.device, dtype=torch.float32)
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
    dZ = alpha[..., None] * dY[:, :, :, None, :]
    score_proj = torch.sum(dY[:, :, :, None, :] * Z, dim=-1)
    centered = score_proj - torch.sum(alpha * score_proj, dim=-1, keepdim=True)
    dS_dec = alpha * centered
    return dZ, dS_dec


def _flare_recurrent_dense_encode_backward(S_enc, LSE_enc, V_f, dZ):
    B, H, T, M = S_enc.shape
    D = V_f.size(-1)
    dS_enc = torch.zeros_like(S_enc)
    dV_f = torch.zeros((B, H, T, D), device=S_enc.device, dtype=torch.float32)
    for t in range(T):
        s_pref = S_enc[:, :, : t + 1, :]
        lse_t = LSE_enc[:, :, t : t + 1, :]
        p_t = torch.exp(s_pref - lse_t)
        dz_t = dZ[:, :, t, :, :]
        c_t = torch.einsum("bhud,bhmd->bhum", V_f[:, :, : t + 1, :], dz_t)
        r_t = torch.sum(p_t * c_t, dim=2, keepdim=True)
        dS_enc[:, :, : t + 1, :] += p_t * (c_t - r_t)
        dV_f[:, :, : t + 1, :] += torch.einsum("bhum,bhmd->bhud", p_t, dz_t)
    return dS_enc, dV_f


def flare_recurrent_dense_backward_pytorch(Q, K, V, dY, scale=None, Q_dec=None, K_dec=None):
    state = _flare_recurrent_dense_lse_forward_state(Q, K, V, scale=scale, Q_dec=Q_dec, K_dec=K_dec)
    Q_f = state["Q_f"]
    K_f = state["K_f"]
    V_f = state["V_f"]
    Q_dec_f = state["Q_dec_f"]
    K_dec_f = state["K_dec_f"]
    S_enc = state["S_enc"]
    LSE_enc = state["LSE_enc"]
    alpha = state["alpha"]
    Z = state["Z"]
    Y = state["Y"]
    scale = state["scale"]
    separate_Q_dec = state["separate_Q_dec"]
    separate_K_dec = state["separate_K_dec"]
    weight_sharing_enc_dec = state["weight_sharing_enc_dec"]

    if dY.dim() != 4 or dY.size(0) != K.size(0) or dY.size(1) != K.size(2) or dY.size(2) != K.size(1) or dY.size(3) != V.size(3):
        raise ValueError(
            "Expected dY [B, H, T, D] matching recurrent output layout. "
            f"Got dY.shape={tuple(dY.shape)} for K.shape={tuple(K.shape)} and V.shape={tuple(V.shape)}"
        )
    dY_f = dY.float()

    dZ, dS_dec = _flare_recurrent_dense_decode_backward(alpha, Z, dY_f)
    dS_enc, dV_f = _flare_recurrent_dense_encode_backward(S_enc, LSE_enc, V_f, dZ)

    if weight_sharing_enc_dec:
        for t in range(S_enc.size(2)):
            dS_enc[:, :, t, :] += dS_dec[:, :, t, :]

    dQ = scale * torch.einsum("bhtm,bhtd->hmd", dS_enc, K_f)
    dK_f = scale * torch.einsum("bhtm,bhmd->bhtd", dS_enc, Q_f)

    dQ_dec = None
    dK_dec = None
    if not weight_sharing_enc_dec:
        if separate_Q_dec:
            dQ_dec_f = scale * torch.einsum("bhtm,bhmd->bhtd", dS_dec, K_dec_f)
            dQ_dec = dQ_dec_f.permute(0, 2, 1, 3).contiguous()
        else:
            dK_f += scale * torch.einsum("bhtm,bhmd->bhtd", dS_dec, K_dec_f)

        if separate_K_dec:
            dK_dec_f = scale * torch.einsum("bhtm,bhtd->bhmd", dS_dec, Q_dec_f)
            dK_dec = dK_dec_f.sum(dim=0).contiguous()
        else:
            dQ += scale * torch.einsum("bhtm,bhtd->bhmd", dS_dec, Q_dec_f).sum(dim=0)

    dK = dK_f.permute(0, 2, 1, 3).contiguous()
    dV = dV_f.permute(0, 2, 1, 3).contiguous()
    return (
        Y.to(Q.dtype),
        dQ.to(Q.dtype),
        dK.to(K.dtype),
        dV.to(V.dtype),
        None if dQ_dec is None else dQ_dec.to(Q.dtype),
        None if dK_dec is None else dK_dec.to(Q.dtype),
    )


def flare_causal_pytorch_dense1(Q, K, V, scale=None, eps=None, profile: bool = False):
    assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
        "Q, K, V must be 3D and 4D tensors respectively "
        f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
    )
    assert K.size() == V.size(), f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}"
    assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
        "Expected Q [H, M, D] and K/V [B, N, H, D]. "
        f"Got Q.shape={Q.shape} and K.shape={K.shape}"
    )
    H_q, M_q, D_q = Q.size()
    B_k, N_k, H_k, D_k = K.size()
    scale = _resolve_attn_scale(scale, D_q)
    assert H_q == H_k and D_q == D_k, "Incompatible K/V dimensions"

    device = Q.device
    out_dtype = Q.dtype
    compute_dtype = torch.float32
    if eps is None:
        eps = _get_eps_for_dtype(Q.dtype)

    Q_f = Q.to(compute_dtype)
    K_f = K.to(compute_dtype)
    V_f = V.to(compute_dtype)

    K_bhnd = K_f.permute(0, 2, 1, 3).contiguous()
    V_bhnd = V_f.permute(0, 2, 1, 3).contiguous()

    S = scale * (Q_f @ K_bhnd.mT)
    P = torch.softmax(S, dim=-2)

    R = torch.full_like(S, -torch.inf)
    L = torch.zeros_like(S)
    R_prev = R[..., 0].clone()
    L_prev = L[..., 0].clone()

    for t in range(N_k):
        s_t = S[..., t].to(torch.float32)
        r_t = torch.maximum(R_prev, s_t)
        L_t = L_prev * torch.exp(R_prev - r_t) + torch.exp(s_t - r_t)
        R[..., t] = r_t
        L[..., t] = L_t
        R_prev, L_prev = r_t, L_t

    K = R.unsqueeze(-2) - R.unsqueeze(-1)
    W = ((P / (L + eps)).unsqueeze(-1) * torch.exp(S - R).unsqueeze(-2) * torch.exp(K)).sum(dim=-3)

    causal = _get_causal_mask(N_k, device)
    W = W.masked_fill(~causal[None, :, :], 0.0)
    Yc = W @ V_bhnd

    Y = Yc.reshape(B_k, H_k, N_k, D_k).permute(0, 2, 1, 3).to(out_dtype)
    return Y


def flare_causal_pytorch_dense(Q, K, V, scale=None, eps=None, profile: bool = False):
    assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
        "Q, K, V must be 3D and 4D tensors respectively "
        f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
    )
    assert K.size() == V.size(), f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}"
    assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
        "Expected Q [H, M, D] and K/V [B, N, H, D]. "
        f"Got Q.shape={Q.shape} and K.shape={K.shape}"
    )
    H_q, M_q, D_q = Q.size()
    B_k, N_k, H_k, D_k = K.size()
    scale = _resolve_attn_scale(scale, D_q)
    assert H_q == H_k and D_q == D_k, "Incompatible K/V dimensions"

    device = Q.device
    out_dtype = Q.dtype
    compute_dtype = torch.float32
    if eps is None:
        eps = _get_eps_for_dtype(Q.dtype)

    Q_f = Q.to(compute_dtype)
    K_f = K.to(compute_dtype)
    V_f = V.to(compute_dtype)

    K_bhnd = K_f.permute(0, 2, 1, 3).contiguous()
    V_bhnd = V_f.permute(0, 2, 1, 3).contiguous()

    S = scale * (Q_f @ K_bhnd.mT)
    A = torch.exp(S)
    D = torch.cumsum(A, dim=-1)
    P = torch.softmax(S, dim=-2)
    W = (P / (D + eps)).mT @ A

    causal = _get_causal_mask(N_k, device)
    W = W.masked_fill(~causal[None, :, :], 0.0)
    Yc = W @ V_bhnd

    Y = Yc.reshape(B_k, H_k, N_k, D_k).permute(0, 2, 1, 3).to(out_dtype)
    return Y

def _get_dense_impl(env_key: str, default: str = "legacy") -> str:
    impl = os.environ.get(env_key, default).strip().lower()
    if impl in ("legacy", "blocked"):
        return impl
    raise ValueError(f"Unsupported {env_key}={impl!r}. Expected one of {'legacy','blocked'}.")


def _get_dense_reduce_block_m(M: int, env_key: str) -> int:
    raw = os.environ.get(env_key, "").strip()
    if raw:
        block = int(raw)
    elif M <= 32:
        block = 16
    elif M <= 64:
        block = 32
    else:
        block = 64
    block = min(block, M)
    if block % 16 != 0:
        raise ValueError(f"{env_key} must be a multiple of 16. Got {block}")
    return block


def _dense_phase3_mblocked(
    X: torch.Tensor,
    Btilde: torch.Tensor,
    V_scaled: torch.Tensor,
    input_precision: str,
    reduce_block_m: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    B, H, M, N = X.shape
    D = V_scaled.size(-1)
    if Btilde.shape != X.shape:
        raise ValueError(f"Btilde shape must match X shape. Got X={tuple(X.shape)} Btilde={tuple(Btilde.shape)}")
    if V_scaled.shape != (B, H, N, D):
        raise ValueError(f"V_scaled shape mismatch. Expected {(B, H, N, D)}, got {tuple(V_scaled.shape)}")

    if (N % 16) != 0:
        raise ValueError(f"Dense blocked phase3 requires N be a multiple of 16. Got N={N}")

    block_d = 16
    block_t = 16
    block_u = 16
    grid = (B * H, triton.cdiv(N, block_t), triton.cdiv(D, block_d))
    use_bf16_matmul = out_dtype == torch.bfloat16
    use_fp16_matmul = out_dtype == torch.float16
    Y = torch.empty((B, H, N, D), device=X.device, dtype=torch.float32)
    flare_dense_phase3_mblocked_kernel[grid](
        X, Btilde, V_scaled, Y,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        Btilde.stride(0), Btilde.stride(1), Btilde.stride(2), Btilde.stride(3),
        V_scaled.stride(0), V_scaled.stride(1), V_scaled.stride(2), V_scaled.stride(3),
        Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
        H, M, N,
        D=D,
        BLOCK_T=block_t,
        BLOCK_U=block_u,
        BLOCK_D=block_d,
        REDUCE_BLOCK_M=reduce_block_m,
        INPUT_PRECISION=input_precision,
        USE_BF16_MATMUL=use_bf16_matmul,
        USE_FP16_MATMUL=use_fp16_matmul,
        num_warps=4,
        num_stages=2,
    )
    return Y.to(out_dtype)


def _dense_phase1_sp(
    Q: torch.Tensor,
    K_bhnd: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    H, M, D = Q.size()
    B, _, N, _ = K_bhnd.size()
    block_m = M
    block_d = D
    block_n = N
    grid = (B * H,)
    num_warps = 4 if block_m <= 64 else 8

    S = torch.empty((B, H, M, N), device=Q.device, dtype=torch.float32)
    P = torch.empty((B, H, M, N), device=Q.device, dtype=torch.float32)
    flare_dense1_phase1_kernel[grid](
        Q, K_bhnd, S, P,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
        S.stride(0), S.stride(1), S.stride(2), S.stride(3),
        P.stride(0), P.stride(1), P.stride(2), P.stride(3),
        B, H, M,
        N,
        scale,
        D=D,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    return S, P


def _denseflare1_blocked_forward(Q: torch.Tensor, K_bhnd: torch.Tensor, V_bhnd: torch.Tensor, scale: float, eps: float) -> torch.Tensor:
    H, M, D = Q.size()
    B, _, N, _ = K_bhnd.size()
    block_m = M
    block_n = N
    num_warps = 4 if block_m <= 64 else 8

    S, _ = _dense_phase1_sp(Q, K_bhnd, scale)
    X = torch.empty((B, H, M, N), device=Q.device, dtype=torch.float32)
    flare_dense1_phase2_kernel[(B * H,)](
        Q, K_bhnd, X,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        B, H, M,
        N,
        scale,
        eps,
        D=D,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_warps=num_warps,
        num_stages=2,
    )

    mu = S.max(dim=2).values  # [B, H, N]
    Btilde = torch.exp(S - mu[:, :, None, :])
    V_scaled = V_bhnd.to(torch.float32) * torch.exp(mu)[:, :, :, None]

    input_precision = _get_input_precision()
    reduce_block_m = _get_dense_reduce_block_m(M, "FLARE_DENSE1_REDUCE_BLOCK_M")
    return _dense_phase3_mblocked(X, Btilde, V_scaled, input_precision, reduce_block_m, Q.dtype)


def _denseflare_blocked_forward(
    Q: torch.Tensor,
    K_bhnd: torch.Tensor,
    V_bhnd: torch.Tensor,
    scale: float,
    eps: float,
    clamp_max: float,
) -> torch.Tensor:
    S, P = _dense_phase1_sp(Q, K_bhnd, scale)
    s_max = S.max(dim=2).values
    s_max = torch.clamp_max(s_max, clamp_max)
    s_max_global = s_max.max(dim=-1).values
    e = torch.exp(s_max - s_max_global[:, :, None])  # [B, H, N]
    A_prime = torch.exp(S - s_max[:, :, None, :])
    D_mat = torch.cumsum(A_prime * e[:, :, None, :], dim=-1)
    eps_scaled = eps * torch.exp(-s_max_global)[:, :, None, None]
    E = P / (D_mat + eps_scaled)
    Btilde = A_prime * e[:, :, None, :]

    input_precision = _get_input_precision()
    reduce_block_m = _get_dense_reduce_block_m(Q.size(1), "FLARE_DENSE_REDUCE_BLOCK_M")
    return _dense_phase3_mblocked(E, Btilde, V_bhnd.to(torch.float32), input_precision, reduce_block_m, Q.dtype)

class DenseFLARE(autograd.Function):
    # Experimental dense path only. Current recommendation is to use
    # recurrent `AutoRegressiveFLARE`; latest runs show dense underperforming it.
    @staticmethod
    def forward(ctx, Q, K, V, scale=None):
        if Q.dim() != 3 or K.dim() != 4 or V.dim() != 4:
            raise ValueError(
                "Dense FLARE expects Q [H, M, D] and K/V [B, N, H, D]. "
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
        B, N, _, _ = K.size()
        scale = _resolve_attn_scale(scale, D)
        if (M % 16) != 0 or (D % 16) != 0 or (N % 16) != 0:
            raise ValueError(f"DenseFLARE requires M, N, D be multiples of 16. Got M={M}, N={N}, D={D}")

        K_bhnd = K.permute(0, 2, 1, 3).contiguous()
        V_bhnd = V.permute(0, 2, 1, 3).contiguous()

        dense_impl = _get_dense_impl("FLARE_DENSE_IMPL", default="legacy")
        if dense_impl == "blocked":
            eps = _get_eps_for_dtype(Q.dtype)
            clamp_max = _get_exp_clamp_for_dtype(Q.dtype)
            Y = _denseflare_blocked_forward(Q, K_bhnd, V_bhnd, scale, eps, clamp_max)
            ctx.save_for_backward(Q, K, V)
            ctx.scale = scale
            ctx._dense_impl = dense_impl
            return Y.permute(0, 2, 1, 3).contiguous()

        block_m = M
        block_d = D
        block_n = N

        Y = torch.empty((B, H, N, D), device=Q.device, dtype=Q.dtype)
        grid = (B * H,)

        num_warps = 4 if block_m <= 64 else 8
        eps = _get_eps_for_dtype(Q.dtype)
        clamp_max = _get_exp_clamp_for_dtype(Q.dtype)
        flare_dense_fwd_kernel[grid](
            Q, K_bhnd, V_bhnd, Y,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
            V_bhnd.stride(0), V_bhnd.stride(1), V_bhnd.stride(2), V_bhnd.stride(3),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            B, H, M,
            N,
            scale,
            eps,
            clamp_max,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )
        ctx.save_for_backward(Q, K, V)
        ctx.scale = scale
        ctx._dense_impl = dense_impl
        return Y.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def backward(ctx, *grad_outputs):
        (Q, K, V) = ctx.saved_tensors
        scale = ctx.scale
        dY = grad_outputs[0]
        if dY is None:
            return None, None, None, None
        dense_impl = getattr(ctx, "_dense_impl", "legacy")
        if dense_impl == "blocked":
            with torch.enable_grad():
                Qr = Q.detach().requires_grad_(True)
                Kr = K.detach().requires_grad_(True)
                Vr = V.detach().requires_grad_(True)
                Yr = flare_causal_pytorch_dense(Qr, Kr, Vr, scale=scale)
                dQ, dK, dV = torch.autograd.grad(
                    outputs=Yr,
                    inputs=(Qr, Kr, Vr),
                    grad_outputs=dY,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )
            out_dtype = Q.dtype
            return dQ.to(out_dtype), dK.to(out_dtype), dV.to(out_dtype), None

        H, M, D = Q.size()
        B, N, _, _ = K.size()
        if (M % 16) != 0 or (D % 16) != 0 or (N % 16) != 0:
            raise ValueError(f"DenseFLARE requires M, N, D be multiples of 16. Got M={M}, N={N}, D={D}")

        K_bhnd = K.permute(0, 2, 1, 3).contiguous()
        V_bhnd = V.permute(0, 2, 1, 3).contiguous()
        dY_bhnd = dY.permute(0, 2, 1, 3).contiguous()

        dQ = torch.zeros_like(Q)
        dK = torch.empty_like(K_bhnd)
        dV = torch.empty_like(V_bhnd)

        block_m = min(M, 32)
        block_d = D
        block_n = N
        grid = (B * H,)

        num_warps = 4 if block_m <= 64 else 8
        eps = _get_eps_for_dtype(Q.dtype)
        flare_dense_bwd_kernel[grid](
            Q, K_bhnd, V_bhnd, dY_bhnd,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
            V_bhnd.stride(0), V_bhnd.stride(1), V_bhnd.stride(2), V_bhnd.stride(3),
            dY_bhnd.stride(0), dY_bhnd.stride(1), dY_bhnd.stride(2), dY_bhnd.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            B, H, M,
            N,
            scale,
            eps,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )

        out_dtype = Q.dtype
        return dQ.to(out_dtype), dK.permute(0, 2, 1, 3).contiguous().to(out_dtype), dV.permute(0, 2, 1, 3).contiguous().to(out_dtype), None

class DenseFLARE1(autograd.Function):
    # Experimental dense path only. Current recommendation is to use
    # recurrent `AutoRegressiveFLARE`; latest runs show dense underperforming it.
    @staticmethod
    def forward(ctx, Q, K, V, scale=None):
        if Q.dim() != 3 or K.dim() != 4 or V.dim() != 4:
            raise ValueError(
                "DenseFLARE1 expects Q [H, M, D] and K/V [B, N, H, D]. "
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
        B, N, _, _ = K.size()
        scale = _resolve_attn_scale(scale, D)
        if (M % 16) != 0 or (D % 16) != 0 or (N % 16) != 0:
            raise ValueError(f"DenseFLARE1 requires M, N, D be multiples of 16. Got M={M}, N={N}, D={D}")

        K_bhnd = K.permute(0, 2, 1, 3).contiguous()
        V_bhnd = V.permute(0, 2, 1, 3).contiguous()

        dense1_impl = _get_dense_impl("FLARE_DENSE1_IMPL", default="legacy")
        if dense1_impl == "blocked":
            eps = _get_eps_for_dtype(Q.dtype)
            Y = _denseflare1_blocked_forward(Q, K_bhnd, V_bhnd, scale, eps)
            return Y.permute(0, 2, 1, 3).contiguous()

        block_m = M
        block_d = D
        block_n = N

        Y = torch.empty((B, H, N, D), device=Q.device, dtype=Q.dtype)
        grid = (B * H,)

        num_warps = 4 if block_m <= 64 else 8
        eps = _get_eps_for_dtype(Q.dtype)
        flare_dense1_fwd_kernel[grid](
            Q, K_bhnd, V_bhnd, Y,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
            V_bhnd.stride(0), V_bhnd.stride(1), V_bhnd.stride(2), V_bhnd.stride(3),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            B, H, M,
            N,
            scale,
            eps,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )
        return Y.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("DenseFLARE1 backward not implemented.")

@triton.jit
def flare_dense_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Y_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_yb, stride_yh, stride_yt, stride_yd,
    B, H, M,
    T,
    scale,
    eps,
    clamp_max,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    # Load full Q and K blocks.
    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block0 = tl.load(q_ptr0, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    k_ptr0 = k_base + n_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
    k_block0 = tl.load(k_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # S = Q @ K^T  -> [M, N]
    S = tl.dot(q_block0, tl.trans(k_block0), out_dtype=tl.float32, allow_tf32=False) * scale
    S = tl.where(mask_m[:, None] & mask_n[None, :], S, -float("inf"))

    # Max-shift per column for stability.
    s_max = tl.max(S, axis=0)
    s_max = tl.minimum(s_max, clamp_max)
    exp_s = tl.exp(S - s_max[None, :])
    exp_s = tl.where(mask_m[:, None] & mask_n[None, :], exp_s, 0.0)

    # l_s = tl.sum(exp_s, axis=0)
    # l_s = tl.where(l_s > 0, l_s, 1.0)
    # P = exp_s / l_s[None, :]
    P = tl.softmax(S, dim=0)

    # # A' = exp(S - max), e = exp(max - max_global) for stability.
    A_prime = exp_s
    s_max_global = tl.max(s_max, axis=0)
    e = tl.exp(s_max - s_max_global)
    eps_scaled = eps * tl.exp(-s_max_global)

    # # Lower-triangular mask L (u <= t) for prefix sums.
    u_idx = n_offsets[:, None]
    t_idx = n_offsets[None, :]
    L = tl.where(u_idx <= t_idx, 1.0, 0.0).to(tl.float32)  # [N, N]

    # # D = A' @ (diag(e) @ L) -> scale columns by e on rows of L.
    L_e = L * e[:, None]
    D_mat = tl.dot(A_prime, L_e, out_dtype=tl.float32, allow_tf32=False)

    # # W = (P / (D + eps))^T @ (A' * e)
    E = P / (D_mat + eps_scaled)
    A_scaled = A_prime * e[None, :]
    W = tl.dot(tl.trans(E), A_scaled, out_dtype=tl.float32, allow_tf32=False)

    # # Apply causal mask (t >= u) and compute Y = W @ V.
    t_idx_rows = n_offsets[:, None]
    u_idx_cols = n_offsets[None, :]
    W = tl.where(t_idx_rows >= u_idx_cols, W, 0.0)

    v_ptr0 = v_base + n_offsets[:, None] * stride_vt + d_offsets[None, :] * stride_vd
    v_block0 = tl.load(v_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    Y_block = tl.dot(W, v_block0, out_dtype=tl.float32, allow_tf32=False)  # [N, D]

    y_ptr0 = y_base + n_offsets[:, None] * stride_yt + d_offsets[None, :] * stride_yd
    tl.store(y_ptr0, Y_block, mask=mask_n[:, None] & mask_d[None, :])

@triton.jit
def flare_dense1_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Y_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_yb, stride_yh, stride_yt, stride_yd,
    B, H, M,
    T,
    scale,
    eps,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D
    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block = tl.load(q_ptr0, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    k_ptr0 = k_base + n_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
    k_block = tl.load(k_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    v_ptr0 = v_base + n_offsets[:, None] * stride_vt + d_offsets[None, :] * stride_vd
    v_block = tl.load(v_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # S = Q @ K^T -> [M, N]
    S = tl.dot(q_block, tl.trans(k_block), out_dtype=tl.float32, allow_tf32=False) * scale
    S = tl.where(mask_m[:, None] & mask_n[None, :], S, -float("inf"))

    # Build X[:, t] online (no R/L materialization).
    X = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    r_prev = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_prev = tl.zeros([BLOCK_M], tl.float32)
    n_idx = n_offsets[None, :]

    for t in tl.static_range(0, BLOCK_N):
        valid_t = t < T
        k_t_ptr = k_base + t * stride_kt + d_offsets * stride_kd
        k_t = tl.load(k_t_ptr, mask=mask_d & valid_t, other=0.0).to(tl.float32)
        s_t = tl.sum(q_block * k_t[None, :], axis=1) * scale
        s_t = tl.where(mask_m & valid_t, s_t, -float("inf"))
        p_t = tl.softmax(s_t, dim=0)
        r_t = tl.maximum(r_prev, s_t)
        l_t = l_prev * tl.exp(r_prev - r_t) + tl.exp(s_t - r_t)
        x_t = p_t / (l_t + eps) * tl.exp(-r_t)
        col_mask = n_idx == t
        X = tl.where(col_mask & valid_t, x_t[:, None], X)
        r_prev = tl.where(valid_t, r_t, r_prev)
        l_prev = tl.where(valid_t, l_t, l_prev)

    mu = tl.max(S, axis=0)
    mu = tl.where(mask_n, mu, 0.0)
    Btilde = tl.exp(tl.where(mask_m[:, None] & mask_n[None, :], S - mu[None, :], -float("inf")))

    W = tl.dot(tl.trans(X), Btilde, out_dtype=tl.float32, allow_tf32=False)

    t_idx = n_offsets[:, None]
    u_idx = n_offsets[None, :]
    W = tl.where(t_idx >= u_idx, W, 0.0)

    exp_mu = tl.exp(mu)
    v_scaled = v_block * exp_mu[:, None]
    Y_block = tl.dot(W, v_scaled, out_dtype=tl.float32, allow_tf32=False)
    y_ptr0 = y_base + n_offsets[:, None] * stride_yt + d_offsets[None, :] * stride_yd
    tl.store(y_ptr0, Y_block, mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def flare_dense1_phase1_kernel(
    Q_ptr,
    K_ptr,
    S_ptr,
    P_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_sb, stride_sh, stride_sm, stride_st,
    stride_pb, stride_ph, stride_pm, stride_pt,
    B, H, M,
    T,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D
    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh

    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block = tl.load(q_ptr0, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    k_ptr0 = k_base + n_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
    k_block = tl.load(k_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    S = tl.dot(q_block, tl.trans(k_block), out_dtype=tl.float32, allow_tf32=False) * scale
    S = tl.where(mask_m[:, None] & mask_n[None, :], S, -float("inf"))
    P = tl.softmax(S, dim=0)

    s_base = S_ptr + b * stride_sb + h * stride_sh
    p_base = P_ptr + b * stride_pb + h * stride_ph
    s_ptr0 = s_base + m_offsets[:, None] * stride_sm + n_offsets[None, :] * stride_st
    p_ptr0 = p_base + m_offsets[:, None] * stride_pm + n_offsets[None, :] * stride_pt
    tl.store(s_ptr0, S, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(p_ptr0, P, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def flare_dense1_phase2_kernel(
    Q_ptr,
    K_ptr,
    X_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_xb, stride_xh, stride_xm, stride_xt,
    B, H, M,
    T,
    scale,
    eps,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh - b * H

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    x_base = X_ptr + b * stride_xb + h * stride_xh

    d_offsets = tl.arange(0, D)
    mask_d = d_offsets < D
    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block = tl.load(q_ptr0, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    r_prev = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_prev = tl.zeros([BLOCK_M], tl.float32)

    for t in tl.static_range(0, BLOCK_N):
        valid_t = t < T
        k_t_ptr = k_base + t * stride_kt + d_offsets * stride_kd
        k_t = tl.load(k_t_ptr, mask=mask_d & valid_t, other=0.0).to(tl.float32)
        s_t = tl.sum(q_block * k_t[None, :], axis=1) * scale
        s_t = tl.where(mask_m & valid_t, s_t, -float("inf"))

        r_t = tl.maximum(r_prev, s_t)
        l_t = l_prev * tl.exp(r_prev - r_t) + tl.exp(s_t - r_t)

        p_t = tl.softmax(s_t, dim=0)
        x_t = p_t / (l_t + eps) * tl.exp(-r_t)

        r_prev = tl.where(valid_t, r_t, r_prev)
        l_prev = tl.where(valid_t, l_t, l_prev)

        x_ptr_t = x_base + m_offsets * stride_xm + t * stride_xt
        tl.store(x_ptr_t, x_t, mask=mask_m & valid_t)


@triton.jit
def flare_dense1_phase3_kernel(
    S_ptr,
    X_ptr,
    V_ptr,
    Y_ptr,
    stride_sb, stride_sh, stride_sm, stride_st,
    stride_xb, stride_xh, stride_xm, stride_xt,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_yb, stride_yh, stride_yt, stride_yd,
    B, H, M,
    T,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D
    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    s_base = S_ptr + b * stride_sb + h * stride_sh
    x_base = X_ptr + b * stride_xb + h * stride_xh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    s_ptr0 = s_base + m_offsets[:, None] * stride_sm + n_offsets[None, :] * stride_st
    x_ptr0 = x_base + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xt

    S = tl.load(s_ptr0, mask=mask_m[:, None] & mask_n[None, :], other=-float("inf"))
    X = tl.load(x_ptr0, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    v_ptr0 = v_base + n_offsets[:, None] * stride_vt + d_offsets[None, :] * stride_vd
    v_block = tl.load(v_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    mu = tl.max(S, axis=0)
    mu = tl.where(mask_n, mu, 0.0)
    Btilde = tl.exp(tl.where(mask_m[:, None] & mask_n[None, :], S - mu[None, :], -float("inf")))

    W = tl.dot(tl.trans(X), Btilde, out_dtype=tl.float32, allow_tf32=False)

    t_idx = n_offsets[:, None]
    u_idx = n_offsets[None, :]
    W = tl.where(t_idx >= u_idx, W, 0.0)

    exp_mu = tl.exp(mu)
    v_scaled = v_block * exp_mu[:, None]
    Y_block = tl.dot(W, v_scaled, out_dtype=tl.float32, allow_tf32=False)
    y_ptr0 = y_base + n_offsets[:, None] * stride_yt + d_offsets[None, :] * stride_yd
    tl.store(y_ptr0, Y_block, mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def flare_dense_phase3_mblocked_kernel(
    X_ptr,
    Btilde_ptr,
    V_ptr,
    Y_ptr,
    stride_xb, stride_xh, stride_xm, stride_xt,
    stride_bb, stride_bh, stride_bm, stride_bt,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_yb, stride_yh, stride_yt, stride_yd,
    H, M, T,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_U: tl.constexpr,
    BLOCK_D: tl.constexpr,
    REDUCE_BLOCK_M: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    USE_BF16_MATMUL: tl.constexpr,
    USE_FP16_MATMUL: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_d = tl.program_id(2)

    b = pid_bh // H
    h = pid_bh - b * H

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_t = t_offsets < T
    mask_d = d_offsets < D

    x_base = X_ptr + b * stride_xb + h * stride_xh
    b_base = Btilde_ptr + b * stride_bb + h * stride_bh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    y_acc = tl.zeros([BLOCK_T, BLOCK_D], tl.float32)

    u0 = 0
    while u0 < T:
        u_offsets = u0 + tl.arange(0, BLOCK_U)
        mask_u = u_offsets < T
        p_acc = tl.zeros([BLOCK_T, BLOCK_U], tl.float32)

        m0 = 0
        while m0 < M:
            m_offsets = m0 + tl.arange(0, REDUCE_BLOCK_M)
            mask_m = m_offsets < M

            x_ptr = x_base + m_offsets[:, None] * stride_xm + t_offsets[None, :] * stride_xt
            b_ptr = b_base + m_offsets[:, None] * stride_bm + u_offsets[None, :] * stride_bt

            x_tile = tl.load(x_ptr, mask=mask_m[:, None] & mask_t[None, :], other=0.0)
            b_tile = tl.load(b_ptr, mask=mask_m[:, None] & mask_u[None, :], other=0.0)

            if USE_BF16_MATMUL:
                p_acc += tl.dot(
                    tl.trans(x_tile.to(tl.bfloat16)),
                    b_tile.to(tl.bfloat16),
                    out_dtype=tl.float32,
                    input_precision=INPUT_PRECISION,
                )
            elif USE_FP16_MATMUL:
                p_acc += tl.dot(
                    tl.trans(x_tile.to(tl.float16)),
                    b_tile.to(tl.float16),
                    out_dtype=tl.float32,
                    input_precision=INPUT_PRECISION,
                )
            else:
                p_acc += tl.dot(
                    tl.trans(x_tile),
                    b_tile,
                    out_dtype=tl.float32,
                    input_precision=INPUT_PRECISION,
                )
            m0 += REDUCE_BLOCK_M

        causal = u_offsets[None, :] <= t_offsets[:, None]
        p_acc = tl.where(mask_t[:, None] & mask_u[None, :] & causal, p_acc, 0.0)

        v_ptr = v_base + u_offsets[:, None] * stride_vt + d_offsets[None, :] * stride_vd
        v_tile = tl.load(v_ptr, mask=mask_u[:, None] & mask_d[None, :], other=0.0)

        if USE_BF16_MATMUL:
            y_acc += tl.dot(
                p_acc.to(tl.bfloat16),
                v_tile.to(tl.bfloat16),
                out_dtype=tl.float32,
                input_precision=INPUT_PRECISION,
            )
        elif USE_FP16_MATMUL:
            y_acc += tl.dot(
                p_acc.to(tl.float16),
                v_tile.to(tl.float16),
                out_dtype=tl.float32,
                input_precision=INPUT_PRECISION,
            )
        else:
            y_acc += tl.dot(
                p_acc,
                v_tile,
                out_dtype=tl.float32,
                input_precision=INPUT_PRECISION,
            )
        u0 += BLOCK_U

    y_ptr = y_base + t_offsets[:, None] * stride_yt + d_offsets[None, :] * stride_yd
    tl.store(y_ptr, y_acc, mask=mask_t[:, None] & mask_d[None, :])


@triton.jit
def flare_dense_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dY_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_dyb, stride_dyh, stride_dyt, stride_dyd,
    stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkt, stride_dkd,
    stride_dvb, stride_dvh, stride_dvt, stride_dvd,
    B, H, M,
    T,
    scale,
    eps,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    dy_base = dY_ptr + b * stride_dyb + h * stride_dyh

    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block = tl.load(q_ptr0, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    k_ptr0 = k_base + n_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
    k_block = tl.load(k_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    v_ptr0 = v_base + n_offsets[:, None] * stride_vt + d_offsets[None, :] * stride_vd
    v_block = tl.load(v_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    dy_ptr0 = dy_base + n_offsets[:, None] * stride_dyt + d_offsets[None, :] * stride_dyd
    dy_block = tl.load(dy_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # S = Q @ K^T  -> [M, N]
    S = tl.dot(q_block, tl.trans(k_block), out_dtype=tl.float32, allow_tf32=False) * scale
    S = tl.where(mask_m[:, None] & mask_n[None, :], S, -float("inf"))

    # Global max shift for A_g
    s_max_col = tl.max(S, axis=0)
    s_max_global = tl.max(s_max_col, axis=0)
    A_g = tl.exp(S - s_max_global)
    A_g = tl.where(mask_m[:, None] & mask_n[None, :], A_g, 0.0)

    # Softmax over M for P
    s_max = s_max_col
    exp_s = tl.exp(S - s_max[None, :])
    exp_s = tl.where(mask_m[:, None] & mask_n[None, :], exp_s, 0.0)

    # l_s = tl.sum(exp_s, axis=0)
    # l_s = tl.where(l_s > 0, l_s, 1.0)
    # P = exp_s / l_s[None, :]
    P = tl.softmax(S, dim=0)

    # Lower triangular mask
    u_idx = n_offsets[:, None]
    t_idx = n_offsets[None, :]
    L = tl.where(u_idx <= t_idx, 1.0, 0.0).to(tl.float32)
    L_t = tl.where(t_idx <= u_idx, 1.0, 0.0).to(tl.float32)

    # D = A_g @ L
    D_mat = tl.dot(A_g, L, out_dtype=tl.float32, allow_tf32=False)
    eps_scaled = eps * tl.exp(-s_max_global)
    invD = 1.0 / (D_mat + eps_scaled)
    E = P * invD

    # W = E^T @ A_g
    W = tl.dot(tl.trans(E), A_g, out_dtype=tl.float32, allow_tf32=False)
    t_idx_rows = n_offsets[:, None]
    u_idx_cols = n_offsets[None, :]
    causal = t_idx_rows >= u_idx_cols
    W = tl.where(causal, W, 0.0)

    # dV = W^T @ dY
    dV_block = tl.dot(tl.trans(W), dy_block, out_dtype=tl.float32, allow_tf32=False)

    # dW = dY @ V^T
    dW = tl.dot(dy_block, tl.trans(v_block), out_dtype=tl.float32, allow_tf32=False)
    dW = tl.where(causal, dW, 0.0)

    # dE = A_g @ dW^T
    dE = tl.dot(A_g, tl.trans(dW), out_dtype=tl.float32, allow_tf32=False)
    # dA_g from W
    dA_g = tl.dot(E, dW, out_dtype=tl.float32, allow_tf32=False)

    # dD from E
    dP = dE * invD
    dInvD = dE * P
    dD = -dInvD * invD * invD

    # dA_g from D: dD @ L^T
    dA_g = dA_g + tl.dot(dD, L_t, out_dtype=tl.float32, allow_tf32=False)

    # dS from A_g (treat shift as constant)
    dS = dA_g * A_g

    # dS from softmax P
    sum_dP_P = tl.sum(dP * P, axis=0)
    dS = dS + P * (dP - sum_dP_P[None, :])

    # dQ = scale * dS @ K
    dQ_block = tl.dot(dS, k_block, out_dtype=tl.float32, allow_tf32=False) * scale
    # dK = scale * dS^T @ Q
    dK_block = tl.dot(tl.trans(dS), q_block, out_dtype=tl.float32, allow_tf32=False) * scale

    # Store dK, dV
    dK_base = dK_ptr + b * stride_dkb + h * stride_dkh
    dV_base = dV_ptr + b * stride_dvb + h * stride_dvh
    dK_ptr0 = dK_base + n_offsets[:, None] * stride_dkt + d_offsets[None, :] * stride_dkd
    dV_ptr0 = dV_base + n_offsets[:, None] * stride_dvt + d_offsets[None, :] * stride_dvd
    tl.store(dK_ptr0, dK_block, mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dV_ptr0, dV_block, mask=mask_n[:, None] & mask_d[None, :])

    # Atomic add dQ (shared across batch)
    dQ_base = dQ_ptr + h * stride_dqh
    dQ_ptr0 = dQ_base + m_offsets[:, None] * stride_dqm + d_offsets[None, :] * stride_dqd
    tl.atomic_add(dQ_ptr0, dQ_block, mask=mask_m[:, None] & mask_d[None, :])

#======================================================================#
# Recurrent Implementations
#======================================================================#

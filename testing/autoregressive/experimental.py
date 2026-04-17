import torch

from causal_flare.autoregressive.experimental import (
    flare_autoregressive_experimental_pytorch,
    flare_autoregressive_experimental_pytorch_mode_1,
    flare_autoregressive_experimental_pytorch_mode_2,
    flare_autoregressive_experimental_pytorch_mode_4,
)

"""Reference checks for Experimental FLARE autoregressive scan modes.

This file intentionally uses explicit token-loop references for correctness checks
against chunkwise kernels. The references define the exact recurrence equations.

Modes covered here:

- mode 0 (baseline):
  S_t = r_t * S_{t-1} + (W_t ⊗ V_t)
  y_t = c_t^T S_t

- mode 1 (aux additive state):
  Z_t = r^z_t * Z_{t-1} + (W^z_t ⊗ V^z_t)
  S_t = r^s_t * S_{t-1} + (W^s_t ⊗ V^s_t) + g^s_t * Z_t
  y_t = c_t^T S_t

- mode 2 (mode 1 + feedback correction into Z):
  q_t = c_t^T S_{t-1}^{base}  (defined using the mode-1 baseline path)
  Z_t = r^z_t * Z_{t-1} + (W^z_t ⊗ V^z_t) - lambda_t * (W^f_t ⊗ q_t)
  S_t = r^s_t * S_{t-1} + (W^s_t ⊗ V^s_t) + g^s_t * Z_t
  y_t = c_t^T S_t

Modes 3/4 are not validated in this file yet.
"""


def _make_inputs(seed: int = 0):
    torch.manual_seed(seed)
    B, N, H, M, D = 2, 48, 4, 12, 16
    dtype = torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def r(*shape):
        return torch.randn(*shape, dtype=dtype, device=device)

    def p(*shape):
        # Retain/gate/lambda in (0, 1) for stable reference comparisons.
        return torch.sigmoid(r(*shape))

    return {
        "B": B,
        "N": N,
        "H": H,
        "M": M,
        "D": D,
        "W_write_s": r(B, N, H, M),
        "V_s": r(B, N, H, D),
        "W_read": r(B, N, H, M),
        "W_retain_s": p(B, N, H),
        "W_write_z": r(B, N, H, M),
        "W_feedback": r(B, N, H, M),
        "V_z": r(B, N, H, D),
        "W_retain_z": p(B, N, H),
        "W_gate_s": p(B, N, H),
        "W_feedback_lambda": 0.15 * p(B, N, H),
    }


def token_loop_reference_mode_0(W_write, V, W_read, W_retain):
    """Reference token loop for aux_mode=0.

    Shapes:
    - W_write: [B, N, H, M]
    - V: [B, N, H, D]
    - W_read: [B, N, H, M]
    - W_retain: [B, N, H] (scalar retain per head per token)
    """
    B, N, H, M = W_write.shape
    D = V.shape[-1]
    S = torch.zeros(B, H, M, D, dtype=W_write.dtype, device=W_write.device)
    ys = []
    for t in range(N):
        rt = W_retain[:, t]  # [B,H]
        wt = W_write[:, t]  # [B,H,M]
        vt = V[:, t]  # [B,H,D]
        ct = W_read[:, t]  # [B,H,M]
        S = rt[..., None, None] * S + torch.einsum("bhm,bhd->bhmd", wt, vt)
        ys.append(torch.einsum("bhm,bhmd->bhd", ct, S))
    return torch.stack(ys, dim=1)


def token_loop_reference_mode_1(
    W_write_s,
    V_s,
    W_read,
    W_retain_s,
    W_write_z,
    V_z,
    W_retain_z,
    W_gate_s,
):
    """Reference token loop for aux_mode=1.

    Recurrence:
    - Z_t = r^z_t * Z_{t-1} + (W^z_t ⊗ V^z_t)
    - S_t = r^s_t * S_{t-1} + (W^s_t ⊗ V^s_t) + g^s_t * Z_t
    - y_t = c_t^T S_t
    """
    B, N, H, M = W_write_s.shape
    D = V_s.shape[-1]
    S = torch.zeros(B, H, M, D, dtype=W_write_s.dtype, device=W_write_s.device)
    Z = torch.zeros_like(S)
    ys = []
    for t in range(N):
        rs = W_retain_s[:, t]
        rz = W_retain_z[:, t]
        gs = W_gate_s[:, t]
        ws = W_write_s[:, t]
        wz = W_write_z[:, t]
        vs = V_s[:, t]
        vz = V_z[:, t]
        ct = W_read[:, t]
        Z = rz[..., None, None] * Z + torch.einsum("bhm,bhd->bhmd", wz, vz)
        S = rs[..., None, None] * S + torch.einsum("bhm,bhd->bhmd", ws, vs) + gs[..., None, None] * Z
        ys.append(torch.einsum("bhm,bhmd->bhd", ct, S))
    return torch.stack(ys, dim=1)


def token_loop_reference_mode_2(
    W_write_s,
    V_s,
    W_read,
    W_retain_s,
    W_write_z,
    W_feedback,
    V_z,
    W_retain_z,
    W_gate_s,
    W_feedback_lambda,
):
    """Reference token loop for aux_mode=2.

    This reference mirrors the current two-pass mode-2 implementation:

    Pass 1 (baseline path):
    - Run mode-1 recurrence to build S^{base}.
    - Compute q_t = c_t^T S_{t-1}^{base} for each token.

    Normalization:
    - q_t is L2-normalized across channels D.
    - W_feedback is L2-normalized across slots M.

    Pass 2 (final path):
    - Z_t = r^z_t * Z_{t-1} + (W^z_t ⊗ V^z_t) - lambda_t * (W^f_t ⊗ q_t)
    - S_t = r^s_t * S_{t-1} + (W^s_t ⊗ V^s_t) + g^s_t * Z_t
    - y_t = c_t^T S_t
    """
    B, N, H, M = W_write_s.shape
    D = V_s.shape[-1]
    wf = W_feedback / torch.linalg.vector_norm(W_feedback, ord=2, dim=-1, keepdim=True).clamp_min(1e-6)

    # Pass 1: baseline mode-1 path used to define q_t.
    S_base = torch.zeros(B, H, M, D, dtype=W_write_s.dtype, device=W_write_s.device)
    Z_base = torch.zeros_like(S_base)
    q_tokens = []
    for t in range(N):
        ct = W_read[:, t]
        q_prev = torch.einsum("bhm,bhmd->bhd", ct, S_base)
        q_tokens.append(q_prev)
        rz = W_retain_z[:, t]
        rs = W_retain_s[:, t]
        gs = W_gate_s[:, t]
        wz = W_write_z[:, t]
        ws = W_write_s[:, t]
        vz = V_z[:, t]
        vs = V_s[:, t]
        Z_base = rz[..., None, None] * Z_base + torch.einsum("bhm,bhd->bhmd", wz, vz)
        S_base = rs[..., None, None] * S_base + torch.einsum("bhm,bhd->bhmd", ws, vs) + gs[..., None, None] * Z_base
    q_tokens = torch.stack(q_tokens, dim=1)
    q_tokens = q_tokens / torch.linalg.vector_norm(q_tokens, ord=2, dim=-1, keepdim=True).clamp_min(1e-6)

    # Pass 2: final recurrence with normalized feedback term.
    S = torch.zeros(B, H, M, D, dtype=W_write_s.dtype, device=W_write_s.device)
    Z = torch.zeros_like(S)
    ys = []
    for t in range(N):
        rs = W_retain_s[:, t]
        rz = W_retain_z[:, t]
        gs = W_gate_s[:, t]
        lam = W_feedback_lambda[:, t]
        ws = W_write_s[:, t]
        wz = W_write_z[:, t]
        wf_t = wf[:, t]
        vs = V_s[:, t]
        vz = V_z[:, t]
        ct = W_read[:, t]
        q_prev = q_tokens[:, t]
        Z = (
            rz[..., None, None] * Z
            + torch.einsum("bhm,bhd->bhmd", wz, vz)
            - torch.einsum("bhm,bhd->bhmd", wf_t, lam[..., None] * q_prev)
        )
        S = rs[..., None, None] * S + torch.einsum("bhm,bhd->bhmd", ws, vs) + gs[..., None, None] * Z
        ys.append(torch.einsum("bhm,bhmd->bhd", ct, S))
    return torch.stack(ys, dim=1)


def token_loop_reference_mode_4(
    W_write_s,
    V_s,
    W_read,
    W_retain_s,
    V_z,
    W_retain_z,
    W_gate_aux,
):
    """Reference token loop for aux_mode=4.

    Recurrence (matches kernel default EXP_FLARE_A4_NORM="both"):
    - Z_t = r^z_t * Z_{t-1} + v^z_t, where v^z_t is split into [z^w_t, z^v_t]
    - z^w_t <- z^w_t / ||z^w_t||_2 (over M), z^v_t <- z^v_t / ||z^v_t||_2 (over D)
    - S_t = r^s_t * S_{t-1} + (W^s_t ⊗ V^s_t) + g^aux_t * (z^w_t ⊗ z^v_t)
    - y_t = c_t^T S_t
    """
    B, N, H, M = W_write_s.shape
    D = V_s.shape[-1]
    S = torch.zeros(B, H, M, D, dtype=W_write_s.dtype, device=W_write_s.device)
    Z = torch.zeros(B, H, M + D, dtype=W_write_s.dtype, device=W_write_s.device)
    ys = []
    for t in range(N):
        rs = W_retain_s[:, t]
        rz = W_retain_z[:, t]
        g_aux = W_gate_aux[:, t]
        ws = W_write_s[:, t]
        vs = V_s[:, t]
        vz = V_z[:, t]
        ct = W_read[:, t]

        Z = rz[..., None] * Z + vz
        z_slot = Z[..., :M]
        z_val = Z[..., M:]
        z_slot = z_slot / torch.linalg.vector_norm(z_slot, ord=2, dim=-1, keepdim=True).clamp_min(1e-6)
        z_val = z_val / torch.linalg.vector_norm(z_val, ord=2, dim=-1, keepdim=True).clamp_min(1e-6)
        S = (
            rs[..., None, None] * S
            + torch.einsum("bhm,bhd->bhmd", ws, vs)
            + torch.einsum("bh,bhm,bhd->bhmd", g_aux, z_slot, z_val)
        )
        ys.append(torch.einsum("bhm,bhmd->bhd", ct, S))
    return torch.stack(ys, dim=1)


def test_mode0_chunkwise_matches_token_loop():
    # Baseline correctness: chunkwise mode-0 scan must match explicit token loop.
    x = _make_inputs(0)
    y_ref = token_loop_reference_mode_0(
        x["W_write_s"],
        x["V_s"],
        x["W_read"],
        x["W_retain_s"],
    )
    y = flare_autoregressive_experimental_pytorch(
        W_write=x["W_write_s"],
        V=x["V_s"],
        W_read=x["W_read"],
        W_retain=x["W_retain_s"],
        chunk_size=16,
    )
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)


def test_mode1_chunkwise_matches_token_loop():
    # Aux mode-1 correctness: chunkwise scan must match explicit two-state token loop.
    x = _make_inputs(1)
    y_ref = token_loop_reference_mode_1(
        x["W_write_s"],
        x["V_s"],
        x["W_read"],
        x["W_retain_s"],
        x["W_write_z"],
        x["V_z"],
        x["W_retain_z"],
        x["W_gate_s"],
    )
    y = flare_autoregressive_experimental_pytorch_mode_1(
        W_write_s=x["W_write_s"],
        V_s=x["V_s"],
        W_read=x["W_read"],
        W_retain_s=x["W_retain_s"],
        W_write_z=x["W_write_z"],
        V_z=x["V_z"],
        W_retain_z=x["W_retain_z"],
        W_gate_s=x["W_gate_s"],
        chunk_size=16,
    )
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)


def test_mode2_lambda_zero_matches_mode1_and_token_loop():
    # Consistency guard:
    # - mode-2 with lambda=0 should reduce to mode-1,
    # - and still match mode-2 reference loop under lambda=0.
    x = _make_inputs(2)
    zero_lambda = torch.zeros_like(x["W_feedback_lambda"])

    y_ref = token_loop_reference_mode_2(
        x["W_write_s"],
        x["V_s"],
        x["W_read"],
        x["W_retain_s"],
        x["W_write_z"],
        x["W_feedback"],
        x["V_z"],
        x["W_retain_z"],
        x["W_gate_s"],
        zero_lambda,
    )
    y_mode1 = flare_autoregressive_experimental_pytorch_mode_1(
        W_write_s=x["W_write_s"],
        V_s=x["V_s"],
        W_read=x["W_read"],
        W_retain_s=x["W_retain_s"],
        W_write_z=x["W_write_z"],
        V_z=x["V_z"],
        W_retain_z=x["W_retain_z"],
        W_gate_s=x["W_gate_s"],
        chunk_size=16,
    )
    y_mode2 = flare_autoregressive_experimental_pytorch_mode_2(
        W_write_s=x["W_write_s"],
        V_s=x["V_s"],
        W_read=x["W_read"],
        W_retain_s=x["W_retain_s"],
        W_write_z=x["W_write_z"],
        W_feedback=x["W_feedback"],
        V_z=x["V_z"],
        W_retain_z=x["W_retain_z"],
        W_gate_s=x["W_gate_s"],
        W_feedback_lambda=zero_lambda,
        chunk_size=16,
    )
    torch.testing.assert_close(y_mode2, y_mode1, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(y_mode2, y_ref, rtol=1e-4, atol=1e-4)


def test_mode2_chunkwise_matches_token_loop():
    # Full mode-2 correctness (non-zero lambda): chunkwise path equals reference token loop.
    x = _make_inputs(3)
    y_ref = token_loop_reference_mode_2(
        x["W_write_s"],
        x["V_s"],
        x["W_read"],
        x["W_retain_s"],
        x["W_write_z"],
        x["W_feedback"],
        x["V_z"],
        x["W_retain_z"],
        x["W_gate_s"],
        x["W_feedback_lambda"],
    )
    y = flare_autoregressive_experimental_pytorch_mode_2(
        W_write_s=x["W_write_s"],
        V_s=x["V_s"],
        W_read=x["W_read"],
        W_retain_s=x["W_retain_s"],
        W_write_z=x["W_write_z"],
        W_feedback=x["W_feedback"],
        V_z=x["V_z"],
        W_retain_z=x["W_retain_z"],
        W_gate_s=x["W_gate_s"],
        W_feedback_lambda=x["W_feedback_lambda"],
        chunk_size=16,
    )
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)


def test_mode4_chunkwise_matches_token_loop():
    x = _make_inputs(4)
    B, N, H, M, D = x["B"], x["N"], x["H"], x["M"], x["D"]
    dtype = x["W_write_s"].dtype
    device = x["W_write_s"].device

    V_z = torch.randn(B, N, H, M + D, dtype=dtype, device=device)
    W_gate_aux = torch.sigmoid(torch.randn(B, N, H, dtype=dtype, device=device))

    y_ref = token_loop_reference_mode_4(
        x["W_write_s"],
        x["V_s"],
        x["W_read"],
        x["W_retain_s"],
        V_z,
        x["W_retain_z"],
        W_gate_aux,
    )
    y = flare_autoregressive_experimental_pytorch_mode_4(
        W_write_s=x["W_write_s"],
        V_s=x["V_s"],
        W_read=x["W_read"],
        W_retain_s=x["W_retain_s"],
        V_z=V_z,
        W_retain_z=x["W_retain_z"],
        W_gate_aux=W_gate_aux,
        chunk_size=16,
    )
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-4)

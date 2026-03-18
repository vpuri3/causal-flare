import pytest
import torch

from causal_flare.autoregressive.stablemax import (
    _stablemax_score_transform,
    flare_autoregressive_stablemax_pytorch,
)


def _sequential_stablemax_write_gated_reference(
    q,
    k,
    v,
    *,
    scale,
    power,
    q_dec,
    k_dec,
    write_gate_fixed_value=None,
    write_gate_scale=None,
    write_gate_bias=None,
):
    B, N, H, D = k.shape
    _, M, _ = q.shape
    Dv = v.size(-1)
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    q_dec_f = q_dec.float()
    k_dec_f = k_dec.float()

    y = torch.empty((B, N, H, Dv), dtype=torch.float32)
    d_state = torch.zeros((B, H, M), dtype=torch.float32)
    z_state = torch.zeros((B, H, M, Dv), dtype=torch.float32)

    if write_gate_fixed_value is not None:
        gate_fixed = float(write_gate_fixed_value)
    else:
        gate_fixed = None

    def _broadcast_gate_param(param):
        param_t = torch.as_tensor(param, dtype=torch.float32)
        if param_t.ndim == 0:
            return param_t.view(1, 1, 1)
        if param_t.ndim == 1 and param_t.shape[0] == H:
            return param_t.view(1, H, 1)
        if param_t.ndim == 2 and param_t.shape == (H, M):
            return param_t.view(1, H, M)
        raise ValueError(f"Unsupported gate parameter shape: {tuple(param_t.shape)}")

    for t in range(N):
        score_t = float(scale) * torch.einsum("bhd,hmd->bhm", k_f[:, t], q_f)
        stable_t = _stablemax_score_transform(score_t, power=power)
        d_state = d_state + stable_t
        if gate_fixed is not None:
            gate_t = torch.full_like(stable_t, gate_fixed)
        else:
            scale_t = _broadcast_gate_param(0.0 if write_gate_scale is None else write_gate_scale)
            bias_t = _broadcast_gate_param(6.0 if write_gate_bias is None else write_gate_bias)
            gate_t = torch.sigmoid(score_t * scale_t + bias_t)
        alpha_t = (stable_t / d_state.clamp_min(torch.finfo(torch.float32).eps)) * gate_t
        z_state = z_state + alpha_t.unsqueeze(-1) * (v_f[:, t, :, None, :] - z_state)
        decode_logits_t = float(scale) * torch.einsum("bhd,hmd->bhm", q_dec_f[:, t], k_dec_f)
        decode_probs_t = torch.softmax(decode_logits_t, dim=-1)
        y[:, t] = torch.einsum("bhm,bhmd->bhd", decode_probs_t, z_state)

    return y


def _make_gate_case(case: str, H: int, M: int):
    if case == "fixed_0.5":
        return {"write_gate_fixed_value": 0.5}
    if case == "head":
        return {
            "write_gate_scale": torch.linspace(-0.15, 0.2, H, dtype=torch.float32),
            "write_gate_bias": torch.linspace(0.5, 1.1, H, dtype=torch.float32),
        }
    if case == "head_latent":
        return {
            "write_gate_scale": torch.linspace(-0.2, 0.25, H * M, dtype=torch.float32).reshape(H, M),
            "write_gate_bias": torch.linspace(0.3, 1.2, H * M, dtype=torch.float32).reshape(H, M),
        }
    raise ValueError(f"Unknown gate case: {case}")


def _sequential_stablemax_write_gated_reference_autograd(
    q,
    k,
    v,
    *,
    scale,
    power,
    q_dec,
    k_dec,
    write_gate_fixed_value=None,
    write_gate_scale=None,
    write_gate_bias=None,
):
    B, N, H, D = k.shape
    _, M, _ = q.shape
    Dv = v.size(-1)

    y = []
    d_state = torch.zeros((B, H, M), device=k.device, dtype=torch.float32)
    z_state = torch.zeros((B, H, M, Dv), device=k.device, dtype=torch.float32)

    def _broadcast_gate_param(param):
        if torch.is_tensor(param):
            param_t = param
        else:
            param_t = torch.as_tensor(param, device=k.device, dtype=torch.float32)
        if param_t.ndim == 0:
            return param_t.view(1, 1, 1)
        if param_t.ndim == 1 and param_t.shape[0] == H:
            return param_t.view(1, H, 1)
        if param_t.ndim == 2 and param_t.shape == (H, M):
            return param_t.view(1, H, M)
        raise ValueError(f"Unsupported gate parameter shape: {tuple(param_t.shape)}")

    for t in range(N):
        score_t = float(scale) * torch.einsum("bhd,hmd->bhm", k[:, t].float(), q.float())
        stable_t = _stablemax_score_transform(score_t, power=power)
        d_state = d_state + stable_t
        if write_gate_fixed_value is not None:
            gate_t = torch.full_like(stable_t, float(write_gate_fixed_value))
        else:
            scale_t = _broadcast_gate_param(0.0 if write_gate_scale is None else write_gate_scale).to(score_t.dtype)
            bias_t = _broadcast_gate_param(6.0 if write_gate_bias is None else write_gate_bias).to(score_t.dtype)
            gate_t = torch.sigmoid(score_t * scale_t + bias_t)
        alpha_t = (stable_t / d_state.clamp_min(torch.finfo(torch.float32).eps)) * gate_t
        z_state = z_state + alpha_t.unsqueeze(-1) * (v[:, t].float()[:, :, None, :] - z_state)
        decode_logits_t = float(scale) * torch.einsum("bhd,hmd->bhm", q_dec[:, t].float(), k_dec.float())
        decode_probs_t = torch.softmax(decode_logits_t, dim=-1)
        y.append(torch.einsum("bhm,bhmd->bhd", decode_probs_t, z_state))

    return torch.stack(y, dim=1)


def test_stablemax_write_gate_identity_matches_baseline():
    torch.manual_seed(0)

    B = 2
    N = 8
    H = 3
    M = 5
    D = 16
    scale = D ** -0.5

    q = torch.randn((H, M, D), dtype=torch.float32)
    k = torch.randn((B, N, H, D), dtype=torch.float32)
    v = torch.randn((B, N, H, D), dtype=torch.float32)
    q_dec = torch.randn((B, N, H, D), dtype=torch.float32)
    k_dec = torch.randn((H, M, D), dtype=torch.float32)

    y_base = flare_autoregressive_stablemax_pytorch(
        q,
        k,
        v,
        scale=scale,
        chunk_size=4,
        Q_dec=q_dec,
        K_dec=k_dec,
    )
    y_gated = flare_autoregressive_stablemax_pytorch(
        q,
        k,
        v,
        scale=scale,
        chunk_size=4,
        Q_dec=q_dec,
        K_dec=k_dec,
        write_gate=True,
        write_gate_fixed_value=1.0,
    )

    torch.testing.assert_close(y_gated, y_base, rtol=1e-4, atol=1e-5)


@torch.no_grad()
@pytest.mark.parametrize("gate_case", ["fixed_0.5", "head", "head_latent"])
def test_stablemax_write_gate_matches_tiny_sequential_reference(gate_case: str):
    torch.manual_seed(1)

    B = 1
    N = 8
    H = 2
    M = 4
    D = 3
    scale = D ** -0.5

    q = torch.randn((H, M, D), dtype=torch.float32)
    k = torch.randn((B, N, H, D), dtype=torch.float32)
    v = torch.randn((B, N, H, D), dtype=torch.float32)
    q_dec = torch.randn((B, N, H, D), dtype=torch.float32)
    k_dec = torch.randn((H, M, D), dtype=torch.float32)
    gate_kwargs = _make_gate_case(gate_case, H, M)

    y_ref = _sequential_stablemax_write_gated_reference(
        q,
        k,
        v,
        scale=scale,
        power=2.0,
        q_dec=q_dec,
        k_dec=k_dec,
        **gate_kwargs,
    )

    for chunk_size in (1, 2, 4, 8):
        y = flare_autoregressive_stablemax_pytorch(
            q,
            k,
            v,
            scale=scale,
            chunk_size=chunk_size,
            Q_dec=q_dec,
            K_dec=k_dec,
            write_gate=True,
            **gate_kwargs,
        )
        torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)


def test_stablemax_write_gate_affine_composition_matches_direct_replay():
    torch.manual_seed(2)

    B = 2
    H = 3
    C = 5
    M = 4
    Dv = 6

    alpha = torch.rand((B, H, C, M), dtype=torch.float32) * 0.8
    v = torch.randn((B, H, C, Dv), dtype=torch.float32)
    z0 = torch.randn((B, H, M, Dv), dtype=torch.float32)

    z_direct = z0.clone()
    for t in range(C):
        z_direct = z_direct + alpha[:, :, t, :].unsqueeze(-1) * (v[:, :, t, None, :] - z_direct)

    A = torch.ones((B, H, M), dtype=torch.float32)
    B_term = torch.zeros((B, H, M, Dv), dtype=torch.float32)
    for t in range(C):
        beta = 1.0 - alpha[:, :, t, :]
        A = beta * A
        B_term = beta.unsqueeze(-1) * B_term + alpha[:, :, t, :].unsqueeze(-1) * v[:, :, t, None, :]

    z_affine = A.unsqueeze(-1) * z0 + B_term
    torch.testing.assert_close(z_affine, z_direct, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("gate_case", ["fixed_0.5", "head"])
def test_stablemax_write_gate_backward_matches_sequential_autograd(gate_case: str):
    torch.manual_seed(3)

    B = 1
    N = 6
    H = 2
    M = 4
    D = 3
    scale = D ** -0.5

    q_seed = torch.randn((H, M, D), dtype=torch.float32)
    k_seed = torch.randn((B, N, H, D), dtype=torch.float32)
    v_seed = torch.randn((B, N, H, D), dtype=torch.float32)
    q_dec_seed = torch.randn((B, N, H, D), dtype=torch.float32)
    k_dec_seed = torch.randn((H, M, D), dtype=torch.float32)
    grad_out = torch.randn((B, N, H, D), dtype=torch.float32)

    gate_kwargs_seed = _make_gate_case(gate_case, H, M)

    q_ref = q_seed.clone().requires_grad_(True)
    k_ref = k_seed.clone().requires_grad_(True)
    v_ref = v_seed.clone().requires_grad_(True)
    q_dec_ref = q_dec_seed.clone().requires_grad_(True)
    k_dec_ref = k_dec_seed.clone().requires_grad_(True)
    gate_kwargs_ref = {
        key: (value.clone().requires_grad_(True) if torch.is_tensor(value) else value)
        for key, value in gate_kwargs_seed.items()
    }
    y_ref = _sequential_stablemax_write_gated_reference_autograd(
        q_ref,
        k_ref,
        v_ref,
        scale=scale,
        power=2.0,
        q_dec=q_dec_ref,
        k_dec=k_dec_ref,
        **gate_kwargs_ref,
    )
    (y_ref * grad_out).sum().backward()

    q = q_seed.clone().requires_grad_(True)
    k = k_seed.clone().requires_grad_(True)
    v = v_seed.clone().requires_grad_(True)
    q_dec = q_dec_seed.clone().requires_grad_(True)
    k_dec = k_dec_seed.clone().requires_grad_(True)
    gate_kwargs = {
        key: (value.clone().requires_grad_(True) if torch.is_tensor(value) else value)
        for key, value in gate_kwargs_seed.items()
    }
    y = flare_autoregressive_stablemax_pytorch(
        q,
        k,
        v,
        scale=scale,
        chunk_size=4,
        Q_dec=q_dec,
        K_dec=k_dec,
        write_gate=True,
        power=2.0,
        **gate_kwargs,
    )
    (y * grad_out).sum().backward()

    torch.testing.assert_close(q.grad, q_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(k.grad, k_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(v.grad, v_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(q_dec.grad, q_dec_ref.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(k_dec.grad, k_dec_ref.grad, rtol=1e-4, atol=1e-5)

    if gate_case == "head":
        torch.testing.assert_close(
            gate_kwargs["write_gate_scale"].grad,
            gate_kwargs_ref["write_gate_scale"].grad,
            rtol=1e-4,
            atol=1e-5,
        )
        torch.testing.assert_close(
            gate_kwargs["write_gate_bias"].grad,
            gate_kwargs_ref["write_gate_bias"].grad,
            rtol=1e-4,
            atol=1e-5,
        )


def test_stablemax_write_gate_backward_handles_partial_requires_grad():
    torch.manual_seed(4)

    B = 1
    N = 6
    H = 2
    M = 4
    D = 3
    scale = D ** -0.5

    q = torch.randn((H, M, D), dtype=torch.float32, requires_grad=True)
    k = torch.randn((B, N, H, D), dtype=torch.float32)
    v = torch.randn((B, N, H, D), dtype=torch.float32, requires_grad=True)
    q_dec = torch.randn((B, N, H, D), dtype=torch.float32)
    k_dec = torch.randn((H, M, D), dtype=torch.float32, requires_grad=True)
    write_gate_scale = torch.linspace(-0.15, 0.2, H, dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn((B, N, H, D), dtype=torch.float32)

    y = flare_autoregressive_stablemax_pytorch(
        q,
        k,
        v,
        scale=scale,
        chunk_size=4,
        Q_dec=q_dec,
        K_dec=k_dec,
        write_gate=True,
        power=2.0,
        write_gate_scale=write_gate_scale,
    )
    (y * grad_out).sum().backward()

    assert q.grad is not None
    assert v.grad is not None
    assert k_dec.grad is not None
    assert write_gate_scale.grad is not None

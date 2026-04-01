import pytest
import torch

from causal_flare.autoregressive.stablemax import (
    _resolve_stablemax_chunk_size_triton,
    _stablemax_score_transform,
    flare_autoregressive_stablemax_mat_decode_pytorch,
    flare_autoregressive_stablemax_pytorch,
)
from causal_flare.autoregressive.stablemax_triton import flare_autoregressive_stablemax_triton


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
    write_gate_tensor=None,
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

    if write_gate_tensor is not None:
        gate_tensor_f = write_gate_tensor.float()
    else:
        gate_tensor_f = None

    if write_gate_fixed_value is not None:
        gate_fixed = float(write_gate_fixed_value)
    else:
        gate_fixed = None

    for t in range(N):
        score_t = float(scale) * torch.einsum("bhd,hmd->bhm", k_f[:, t], q_f)
        stable_t = _stablemax_score_transform(score_t, power=power)
        d_state = d_state + stable_t
        if gate_tensor_f is not None:
            gate_t = gate_tensor_f[:, t]
        elif gate_fixed is not None:
            gate_t = torch.full_like(stable_t, gate_fixed)
        else:
            gate_t = torch.ones_like(stable_t)
        alpha_t = (stable_t / d_state.clamp_min(torch.finfo(torch.float32).eps)) * gate_t
        z_state = z_state + alpha_t.unsqueeze(-1) * (v_f[:, t, :, None, :] - z_state)
        decode_logits_t = float(scale) * torch.einsum("bhd,hmd->bhm", q_dec_f[:, t], k_dec_f)
        decode_probs_t = torch.softmax(decode_logits_t, dim=-1)
        y[:, t] = torch.einsum("bhm,bhmd->bhd", decode_probs_t, z_state)

    return y

def _make_gate_case(case: str):
    if case == "fixed_0.5":
        return {"write_gate_fixed_value": 0.5}
    raise ValueError(f"Unknown gate case: {case}")


def _sequential_stablemax_mat_decode_reference(
    q,
    k,
    v,
    *,
    c_dec,
    scale,
    power,
):
    B, N, H, _ = k.shape
    _, M, _ = q.shape
    Dv = v.size(-1)
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    c_dec_f = c_dec.float()

    y = torch.empty((B, N, H, Dv), dtype=torch.float32)
    den = torch.zeros((B, H, M), dtype=torch.float32)
    num = torch.zeros((B, H, M, Dv), dtype=torch.float32)

    for t in range(N):
        score_t = float(scale) * torch.einsum("bhd,hmd->bhm", k_f[:, t], q_f)
        stable_t = _stablemax_score_transform(score_t, power=power)
        den = den + stable_t
        num = num + stable_t.unsqueeze(-1) * v_f[:, t, :, None, :]
        y[:, t] = torch.einsum("bhm,bhmd->bhd", c_dec_f[:, t] / den.clamp_min(torch.finfo(torch.float32).eps), num)

    return y


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
    write_gate_tensor=None,
):
    B, N, H, D = k.shape
    _, M, _ = q.shape
    Dv = v.size(-1)

    if write_gate_tensor is not None:
        gate_tensor_f = write_gate_tensor.float()
    else:
        gate_tensor_f = None

    y = []
    d_state = torch.zeros((B, H, M), device=k.device, dtype=torch.float32)
    z_state = torch.zeros((B, H, M, Dv), device=k.device, dtype=torch.float32)

    for t in range(N):
        score_t = float(scale) * torch.einsum("bhd,hmd->bhm", k[:, t].float(), q.float())
        stable_t = _stablemax_score_transform(score_t, power=power)
        d_state = d_state + stable_t
        if gate_tensor_f is not None:
            gate_t = gate_tensor_f[:, t].to(score_t.dtype)
        elif write_gate_fixed_value is not None:
            gate_t = torch.full_like(stable_t, float(write_gate_fixed_value))
        else:
            gate_t = torch.ones_like(stable_t)
        alpha_t = (stable_t / d_state.clamp_min(torch.finfo(torch.float32).eps)) * gate_t
        z_state = z_state + alpha_t.unsqueeze(-1) * (v[:, t].float()[:, :, None, :] - z_state)
        decode_logits_t = float(scale) * torch.einsum("bhd,hmd->bhm", q_dec[:, t].float(), k_dec.float())
        decode_probs_t = torch.softmax(decode_logits_t, dim=-1)
        y.append(torch.einsum("bhm,bhmd->bhd", decode_probs_t, z_state))

    return torch.stack(y, dim=1)


def _finite_difference(loss_fn, tensor: torch.Tensor, index: tuple[int, ...], eps: float = 1e-6) -> torch.Tensor:
    with torch.no_grad():
        pos = tensor.detach().clone()
        neg = tensor.detach().clone()
        pos[index] += eps
        neg[index] -= eps
    return (loss_fn(pos) - loss_fn(neg)) / (2.0 * eps)


def _make_fd_case(*, decode_mode: str = "separate", dtype: torch.dtype = torch.float64):
    torch.manual_seed(11)
    B = 1
    N = 5
    H = 2
    M = 3
    D = 2
    scale = D ** -0.5

    q = torch.randn((H, M, D), dtype=dtype)
    k = torch.randn((B, N, H, D), dtype=dtype)
    v = torch.randn((B, N, H, D), dtype=dtype)
    grad_out = torch.randn((B, N, H, D), dtype=dtype)
    if decode_mode == "separate":
        q_dec = torch.randn((B, N, H, D), dtype=dtype)
        k_dec = torch.randn((H, M, D), dtype=dtype)
    elif decode_mode == "shared":
        q_dec = None
        k_dec = None
    else:
        raise ValueError(f"Unsupported decode_mode={decode_mode!r}")

    return {
        "q": q,
        "k": k,
        "v": v,
        "q_dec": q_dec,
        "k_dec": k_dec,
        "grad_out": grad_out,
        "scale": scale,
        "chunk_size": 2,
        "power": 2.0,
    }


def test_resolve_stablemax_chunk_size_triton_shrinks_for_large_value_heads():
    chunk_size = _resolve_stablemax_chunk_size_triton(
        N=2048,
        M=256,
        D_score=128,
        D_value=256,
        chunk_size=None,
    )
    assert chunk_size == 64


def test_resolve_stablemax_chunk_size_triton_keeps_existing_small_value_head_default():
    chunk_size = _resolve_stablemax_chunk_size_triton(
        N=2048,
        M=256,
        D_score=128,
        D_value=128,
        chunk_size=None,
    )
    assert chunk_size == 128


def _stablemax_gate_loss(
    *,
    q,
    k,
    v,
    q_dec,
    k_dec,
    grad_out,
    scale,
    chunk_size,
    power,
    **gate_kwargs,
):
    y = flare_autoregressive_stablemax_pytorch(
        q,
        k,
        v,
        scale=scale,
        chunk_size=chunk_size,
        Q_dec=q_dec,
        K_dec=k_dec,
        write_gate=True,
        power=power,
        **gate_kwargs,
    )
    return (y * grad_out).sum()


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
def test_stablemax_mat_decode_matches_tiny_sequential_reference():
    torch.manual_seed(21)

    B = 1
    N = 8
    H = 2
    M = 4
    D = 3
    scale = D ** -0.5

    q = torch.randn((H, M, D), dtype=torch.float32)
    k = torch.randn((B, N, H, D), dtype=torch.float32)
    v = torch.randn((B, N, H, D), dtype=torch.float32)
    c_dec = torch.randn((B, N, H, M), dtype=torch.float32)

    y_ref = _sequential_stablemax_mat_decode_reference(
        q,
        k,
        v,
        c_dec=c_dec,
        scale=scale,
        power=2.0,
    )

    for chunk_size in (1, 2, 4, 8):
        y = flare_autoregressive_stablemax_mat_decode_pytorch(
            q,
            k,
            v,
            c_dec,
            scale=scale,
            chunk_size=chunk_size,
            power=2.0,
        )
        torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6)


def test_stablemax_mat_decode_backward_matches_finite_difference(monkeypatch):
    monkeypatch.setenv("FLARE_PYTORCH_MATCH_REFERENCE", "1")
    torch.manual_seed(22)

    B = 1
    N = 5
    H = 2
    M = 4
    D = 2
    scale = D ** -0.5

    q = torch.randn((H, M, D), dtype=torch.float64, requires_grad=True)
    k = torch.randn((B, N, H, D), dtype=torch.float64, requires_grad=True)
    v = torch.randn((B, N, H, D), dtype=torch.float64, requires_grad=True)
    c_dec = torch.randn((B, N, H, M), dtype=torch.float64, requires_grad=True)
    grad_out = torch.randn((B, N, H, D), dtype=torch.float64)

    def loss_fn(q_var, k_var, v_var, c_var):
        y = flare_autoregressive_stablemax_mat_decode_pytorch(
            q_var,
            k_var,
            v_var,
            c_var,
            scale=scale,
            chunk_size=2,
            power=2.0,
        )
        return (y * grad_out).sum()

    loss = loss_fn(q, k, v, c_dec)
    loss.backward()

    torch.testing.assert_close(q.grad[0, 1, 0], _finite_difference(lambda x: loss_fn(x, k.detach(), v.detach(), c_dec.detach()), q.detach(), (0, 1, 0)), rtol=5e-3, atol=5e-4)
    torch.testing.assert_close(k.grad[0, 2, 1, 0], _finite_difference(lambda x: loss_fn(q.detach(), x, v.detach(), c_dec.detach()), k.detach(), (0, 2, 1, 0)), rtol=5e-3, atol=5e-4)
    torch.testing.assert_close(v.grad[0, 3, 0, 1], _finite_difference(lambda x: loss_fn(q.detach(), k.detach(), x, c_dec.detach()), v.detach(), (0, 3, 0, 1)), rtol=5e-3, atol=5e-4)
    torch.testing.assert_close(c_dec.grad[0, 4, 1, 2], _finite_difference(lambda x: loss_fn(q.detach(), k.detach(), v.detach(), x), c_dec.detach(), (0, 4, 1, 2)), rtol=5e-3, atol=5e-4)


def test_stablemax_mat_decode_validates_shape():
    B = 1
    N = 4
    H = 2
    M = 3
    D = 2
    scale = D ** -0.5

    q = torch.randn((H, M, D), dtype=torch.float32)
    k = torch.randn((B, N, H, D), dtype=torch.float32)
    v = torch.randn((B, N, H, D), dtype=torch.float32)

    with pytest.raises(ValueError, match="C_dec must be \\[B, N, H, M\\]"):
        flare_autoregressive_stablemax_mat_decode_pytorch(
            q,
            k,
            v,
            torch.zeros((B, N, M), dtype=torch.float32),
            scale=scale,
            chunk_size=2,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton path")
def test_stablemax_triton_forward_matches_pytorch_large_bf16():
    torch.manual_seed(0)

    # Requested validation shape:
    # - N = 2048 tokens
    # - B * H = 128 batch-head lanes
    # - M = 128 latent slots (matching the train-like FLARE shapes in this repo)
    # - D = 32 for both score and value heads
    B = 8
    H = 16
    M = 128
    N = 2048
    D = 32
    scale = D ** -0.5

    q = torch.randn((H, M, D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)
    q_dec = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)
    k_dec = torch.randn((H, M, D), device="cuda", dtype=torch.bfloat16)

    y_ref = flare_autoregressive_stablemax_pytorch(
        q,
        k,
        v,
        scale=scale,
        chunk_size=None,
        Q_dec=q_dec,
        K_dec=k_dec,
        power=2.0,
    )
    y_tri = flare_autoregressive_stablemax_triton(
        q,
        k,
        v,
        scale=scale,
        chunk_size=None,
        Q_dec=q_dec,
        K_dec=k_dec,
        power=2.0,
    )

    assert y_tri.dtype == torch.bfloat16
    assert y_tri.shape == y_ref.shape
    torch.testing.assert_close(y_tri.float(), y_ref.float(), atol=7e-3, rtol=7e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton path")
def test_stablemax_triton_backward_matches_pytorch_small_fp32():
    torch.manual_seed(4)

    B = 2
    H = 2
    M = 32
    N = 128
    D = 32
    scale = D ** -0.5

    q_ref = torch.randn((H, M, D), device="cuda", dtype=torch.float32, requires_grad=True)
    k_ref = torch.randn((B, N, H, D), device="cuda", dtype=torch.float32, requires_grad=True)
    v_ref = torch.randn((B, N, H, D), device="cuda", dtype=torch.float32, requires_grad=True)
    q_dec_ref = torch.randn((B, N, H, D), device="cuda", dtype=torch.float32, requires_grad=True)
    k_dec_ref = torch.randn((H, M, D), device="cuda", dtype=torch.float32, requires_grad=True)

    grad_out = torch.randn((B, N, H, D), device="cuda", dtype=torch.float32)

    y_ref = flare_autoregressive_stablemax_pytorch(
        q_ref,
        k_ref,
        v_ref,
        scale=scale,
        chunk_size=64,
        Q_dec=q_dec_ref,
        K_dec=k_dec_ref,
        power=2.0,
    )
    (y_ref * grad_out).sum().backward()

    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)
    q_dec_tri = q_dec_ref.detach().clone().requires_grad_(True)
    k_dec_tri = k_dec_ref.detach().clone().requires_grad_(True)

    y_tri = flare_autoregressive_stablemax_triton(
        q_tri,
        k_tri,
        v_tri,
        scale=scale,
        chunk_size=64,
        Q_dec=q_dec_tri,
        K_dec=k_dec_tri,
        power=2.0,
    )
    (y_tri.float() * grad_out).sum().backward()

    torch.testing.assert_close(q_tri.grad, q_ref.grad, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(k_tri.grad, k_ref.grad, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(v_tri.grad, v_ref.grad, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(q_dec_tri.grad, q_dec_ref.grad, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(k_dec_tri.grad, k_dec_ref.grad, atol=2e-3, rtol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton path")
@pytest.mark.parametrize("power", [3.0, 4.0])
def test_stablemax_triton_higher_power_fast_paths_match_pytorch_small_fp32(power: float):
    torch.manual_seed(7)

    B = 2
    H = 2
    M = 32
    N = 128
    D = 32
    scale = D ** -0.5
    q_ref = torch.randn((H, M, D), device="cuda", dtype=torch.float32, requires_grad=True)
    k_ref = torch.randn((B, N, H, D), device="cuda", dtype=torch.float32, requires_grad=True)
    v_ref = torch.randn((B, N, H, D), device="cuda", dtype=torch.float32, requires_grad=True)
    q_dec_ref = torch.randn((B, N, H, D), device="cuda", dtype=torch.float32, requires_grad=True)
    k_dec_ref = torch.randn((H, M, D), device="cuda", dtype=torch.float32, requires_grad=True)

    grad_out = torch.randn((B, N, H, D), device="cuda", dtype=torch.float32)

    y_ref = flare_autoregressive_stablemax_pytorch(
        q_ref,
        k_ref,
        v_ref,
        scale=scale,
        chunk_size=64,
        Q_dec=q_dec_ref,
        K_dec=k_dec_ref,
        power=power,
    )
    (y_ref * grad_out).sum().backward()

    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)
    q_dec_tri = q_dec_ref.detach().clone().requires_grad_(True)
    k_dec_tri = k_dec_ref.detach().clone().requires_grad_(True)

    y_tri = flare_autoregressive_stablemax_triton(
        q_tri,
        k_tri,
        v_tri,
        scale=scale,
        chunk_size=64,
        Q_dec=q_dec_tri,
        K_dec=k_dec_tri,
        power=power,
    )
    torch.testing.assert_close(y_tri.float(), y_ref.float(), atol=3e-3, rtol=3e-3)

    (y_tri.float() * grad_out).sum().backward()

    torch.testing.assert_close(q_tri.grad, q_ref.grad, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(k_tri.grad, k_ref.grad, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(v_tri.grad, v_ref.grad, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(q_dec_tri.grad, q_dec_ref.grad, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(k_dec_tri.grad, k_dec_ref.grad, atol=3e-3, rtol=3e-3)

@torch.no_grad()
@pytest.mark.parametrize("gate_case", ["fixed_0.5"])
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
    gate_kwargs = _make_gate_case(gate_case)

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


@torch.no_grad()
def test_stablemax_write_gate_tensor_matches_tiny_sequential_reference():
    torch.manual_seed(12)

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
    write_gate_tensor = torch.sigmoid(torch.randn((B, N, H, M), dtype=torch.float32))

    y_ref = _sequential_stablemax_write_gated_reference(
        q,
        k,
        v,
        scale=scale,
        power=2.0,
        q_dec=q_dec,
        k_dec=k_dec,
        write_gate_tensor=write_gate_tensor,
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
            write_gate_tensor=write_gate_tensor,
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


@pytest.mark.parametrize("gate_case", ["fixed_0.5"])
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

    gate_kwargs_seed = _make_gate_case(gate_case)

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
    write_gate_tensor = torch.sigmoid(torch.randn((B, N, H, M), dtype=torch.float32)).requires_grad_(True)
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
        write_gate_tensor=write_gate_tensor,
    )
    (y * grad_out).sum().backward()

    assert q.grad is not None
    assert v.grad is not None
    assert k_dec.grad is not None
    assert write_gate_tensor.grad is not None


@pytest.mark.parametrize("decode_mode", ["separate", "shared"])
def test_stablemax_write_gate_tensor_backward_matches_finite_difference(monkeypatch, decode_mode: str):
    monkeypatch.setenv("FLARE_PYTORCH_MATCH_REFERENCE", "1")
    case = _make_fd_case(decode_mode=decode_mode)

    q = case["q"].clone().requires_grad_(True)
    k = case["k"].clone().requires_grad_(True)
    v = case["v"].clone().requires_grad_(True)
    q_dec = None if case["q_dec"] is None else case["q_dec"].clone().requires_grad_(True)
    k_dec = None if case["k_dec"] is None else case["k_dec"].clone().requires_grad_(True)
    write_gate_tensor = torch.sigmoid(torch.randn((1, 5, 2, 3), dtype=torch.float64)).requires_grad_(True)

    loss = _stablemax_gate_loss(
        q=q,
        k=k,
        v=v,
        q_dec=q_dec,
        k_dec=k_dec,
        grad_out=case["grad_out"],
        scale=case["scale"],
        chunk_size=case["chunk_size"],
        power=case["power"],
        write_gate_tensor=write_gate_tensor,
    )
    loss.backward()

    def loss_with_q(q_var):
        return _stablemax_gate_loss(q=q_var, k=k.detach(), v=v.detach(), q_dec=None if q_dec is None else q_dec.detach(), k_dec=None if k_dec is None else k_dec.detach(), grad_out=case["grad_out"], scale=case["scale"], chunk_size=case["chunk_size"], power=case["power"], write_gate_tensor=write_gate_tensor.detach())

    def loss_with_k(k_var):
        return _stablemax_gate_loss(q=q.detach(), k=k_var, v=v.detach(), q_dec=None if q_dec is None else q_dec.detach(), k_dec=None if k_dec is None else k_dec.detach(), grad_out=case["grad_out"], scale=case["scale"], chunk_size=case["chunk_size"], power=case["power"], write_gate_tensor=write_gate_tensor.detach())

    def loss_with_v(v_var):
        return _stablemax_gate_loss(q=q.detach(), k=k.detach(), v=v_var, q_dec=None if q_dec is None else q_dec.detach(), k_dec=None if k_dec is None else k_dec.detach(), grad_out=case["grad_out"], scale=case["scale"], chunk_size=case["chunk_size"], power=case["power"], write_gate_tensor=write_gate_tensor.detach())

    def loss_with_gate(gate_var):
        return _stablemax_gate_loss(q=q.detach(), k=k.detach(), v=v.detach(), q_dec=None if q_dec is None else q_dec.detach(), k_dec=None if k_dec is None else k_dec.detach(), grad_out=case["grad_out"], scale=case["scale"], chunk_size=case["chunk_size"], power=case["power"], write_gate_tensor=gate_var)

    torch.testing.assert_close(q.grad[0, 1, 0], _finite_difference(loss_with_q, q.detach(), (0, 1, 0)), rtol=5e-3, atol=5e-4)
    torch.testing.assert_close(k.grad[0, 2, 1, 0], _finite_difference(loss_with_k, k.detach(), (0, 2, 1, 0)), rtol=5e-3, atol=5e-4)
    torch.testing.assert_close(v.grad[0, 3, 0, 1], _finite_difference(loss_with_v, v.detach(), (0, 3, 0, 1)), rtol=5e-3, atol=5e-4)
    torch.testing.assert_close(write_gate_tensor.grad[0, 4, 1, 2], _finite_difference(loss_with_gate, write_gate_tensor.detach(), (0, 4, 1, 2)), rtol=5e-3, atol=5e-4)
    if q_dec is not None:
        def loss_with_q_dec(q_dec_var):
            return _stablemax_gate_loss(q=q.detach(), k=k.detach(), v=v.detach(), q_dec=q_dec_var, k_dec=k_dec.detach(), grad_out=case["grad_out"], scale=case["scale"], chunk_size=case["chunk_size"], power=case["power"], write_gate_tensor=write_gate_tensor.detach())

        def loss_with_k_dec(k_dec_var):
            return _stablemax_gate_loss(q=q.detach(), k=k.detach(), v=v.detach(), q_dec=q_dec.detach(), k_dec=k_dec_var, grad_out=case["grad_out"], scale=case["scale"], chunk_size=case["chunk_size"], power=case["power"], write_gate_tensor=write_gate_tensor.detach())

        torch.testing.assert_close(q_dec.grad[0, 1, 1, 0], _finite_difference(loss_with_q_dec, q_dec.detach(), (0, 1, 1, 0)), rtol=5e-3, atol=5e-4)
        torch.testing.assert_close(k_dec.grad[1, 2, 0], _finite_difference(loss_with_k_dec, k_dec.detach(), (1, 2, 0)), rtol=5e-3, atol=5e-4)


@pytest.mark.parametrize("fixed_value", [0.0, 1.0])
def test_stablemax_write_gate_fixed_edge_backward_matches_finite_difference(monkeypatch, fixed_value: float):
    monkeypatch.setenv("FLARE_PYTORCH_MATCH_REFERENCE", "1")
    case = _make_fd_case(decode_mode="shared")

    q = case["q"].clone().requires_grad_(True)
    v = case["v"].clone().requires_grad_(True)
    loss = _stablemax_gate_loss(
        q=q,
        k=case["k"],
        v=v,
        q_dec=None,
        k_dec=None,
        grad_out=case["grad_out"],
        scale=case["scale"],
        chunk_size=case["chunk_size"],
        power=case["power"],
        write_gate_fixed_value=fixed_value,
    )
    loss.backward()

    def loss_with_q(q_var):
        return _stablemax_gate_loss(q=q_var, k=case["k"], v=v.detach(), q_dec=None, k_dec=None, grad_out=case["grad_out"], scale=case["scale"], chunk_size=case["chunk_size"], power=case["power"], write_gate_fixed_value=fixed_value)

    def loss_with_v(v_var):
        return _stablemax_gate_loss(q=q.detach(), k=case["k"], v=v_var, q_dec=None, k_dec=None, grad_out=case["grad_out"], scale=case["scale"], chunk_size=case["chunk_size"], power=case["power"], write_gate_fixed_value=fixed_value)

    torch.testing.assert_close(q.grad[0, 0, 1], _finite_difference(loss_with_q, q.detach(), (0, 0, 1)), rtol=5e-3, atol=5e-4)
    torch.testing.assert_close(v.grad[0, 2, 1, 0], _finite_difference(loss_with_v, v.detach(), (0, 2, 1, 0)), rtol=5e-3, atol=5e-4)


def test_stablemax_write_gate_tensor_validates_shape_and_range():
    B = 1
    N = 4
    H = 2
    M = 3
    D = 2
    scale = D ** -0.5

    q = torch.randn((H, M, D), dtype=torch.float32)
    k = torch.randn((B, N, H, D), dtype=torch.float32)
    v = torch.randn((B, N, H, D), dtype=torch.float32)

    with pytest.raises(ValueError, match="write_gate_tensor must be \\[B, N, H, M\\] or \\[B, H, NC, C, M\\]"):
        flare_autoregressive_stablemax_pytorch(
            q,
            k,
            v,
            scale=scale,
            chunk_size=2,
            write_gate=True,
            write_gate_tensor=torch.zeros((B, N, M), dtype=torch.float32),
        )

    with pytest.raises(ValueError, match="write_gate_tensor entries must lie in \\[0, 1\\]"):
        flare_autoregressive_stablemax_pytorch(
            q,
            k,
            v,
            scale=scale,
            chunk_size=2,
            write_gate=True,
            write_gate_tensor=torch.full((B, N, H, M), 1.5, dtype=torch.float32),
        )

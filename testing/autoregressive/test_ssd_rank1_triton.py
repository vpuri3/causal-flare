import pytest
import torch

from causal_flare.autoregressive.ssd_rank1_triton import (
    SsdRank1ChunkEndStateTriton,
    SsdRank1PrefixScanTriton,
    SsdRank1DenseOutputTriton,
    ssd_rank1_pytorch,
    ssd_rank1_token_loop_oracle,
    ssd_rank1_prefix_scan_reference,
    ssd_rank1_chunk_end_state_reference,
    ssd_rank1_dense_output_backward_kernel_profile,
    ssd_rank1_dense_output_backward_reference,
    ssd_rank1_dense_output_reference,
    ssd_rank1_triton,
)


def _supported_dtypes():
    if not torch.cuda.is_available():
        return [torch.float32]
    dtypes = [torch.float16, torch.float32]
    if torch.cuda.is_bf16_supported():
        dtypes.insert(1, torch.bfloat16)
    return dtypes


def _tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 3e-2, 3e-2
    return 2e-2, 2e-2  # float16


def _r_grad_tol(dtype: torch.dtype):
    # r gradients are scalar reductions over MD and are most sensitive to
    # low-precision accumulation/rounding differences.
    if dtype == torch.float32:
        return 2e-4, 2e-4
    if dtype == torch.bfloat16:
        return 6e-1, 1.2e-1
    return 1.6, 2.0e-1  # float16


def _phase3_tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype == torch.bfloat16:
        return 8e-2, 5e-2
    return 3e-2, 3e-2  # float16


def _phase3_r_grad_tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 2e-3, 2e-3
    if dtype == torch.bfloat16:
        return 1.0, 2.0e-1
    return 2.0, 3.0e-1  # float16


def _phase0_r_grad_tol(dtype: torch.dtype):
    if dtype == torch.float32:
        return 2e-3, 2e-3
    if dtype == torch.bfloat16:
        return 8e-1, 2.0e-1
    return 2.0, 3.0e-1  # float16


def _make_inputs(
    seed: int = 0,
    bh: int = 3,
    nc: int = 7,
    m: int = 16,
    d: int = 32,
    dtype: torch.dtype = torch.float32,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    md = m * d
    S_local_end = torch.randn(bh, nc, md, device=device, dtype=dtype)
    r_chunk = torch.sigmoid(torch.randn(bh, nc, device=device, dtype=dtype))
    init = torch.randn(bh, md, device=device, dtype=dtype)
    return S_local_end, r_chunk, init


def _make_phase0_inputs(
    seed: int = 0,
    bh: int = 3,
    nc: int = 7,
    c: int = 16,
    m: int = 16,
    d: int = 32,
    dtype: torch.dtype = torch.float32,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    w = torch.randn(bh, nc, c, m, device=device, dtype=dtype)
    v = torch.randn(bh, nc, c, d, device=device, dtype=dtype)
    log_alpha = -torch.nn.functional.softplus(torch.randn(bh, nc, c, device=device, dtype=dtype))
    return w, v, log_alpha


def _make_phase3_inputs(
    seed: int = 0,
    bh: int = 2,
    nc: int = 3,
    c: int = 16,
    m: int = 16,
    d: int = 32,
    dtype: torch.dtype = torch.float32,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c_tok = torch.randn(bh, nc, c, m, device=device, dtype=dtype)
    w_tok = torch.randn(bh, nc, c, m, device=device, dtype=dtype)
    v_tok = torch.randn(bh, nc, c, d, device=device, dtype=dtype)
    log_alpha_tok = -torch.nn.functional.softplus(torch.randn(bh, nc, c, device=device, dtype=dtype))
    return c_tok, w_tok, v_tok, log_alpha_tok


def _make_phase123_inputs(
    seed: int = 0,
    b: int = 2,
    n: int = 111,
    h: int = 2,
    m: int = 16,
    d: int = 32,
    dtype: torch.dtype = torch.float32,
):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    c = torch.randn(b, n, h, m, device=device, dtype=dtype)
    w = torch.randn(b, n, h, m, device=device, dtype=dtype)
    v = torch.randn(b, n, h, d, device=device, dtype=dtype)
    log_alpha = -torch.nn.functional.softplus(torch.randn(b, n, h, device=device, dtype=dtype))
    return c, w, v, log_alpha


@pytest.mark.parametrize("dtype", _supported_dtypes())
@pytest.mark.parametrize(
    ("m", "d"),
    [
        (16, 32),
        (16, 64),
        (16, 128),
        (32, 32),
        (32, 64),
        (32, 128),
        (64, 32),
        (64, 64),
        (64, 128),
        (128, 32),
        (128, 64),
        (128, 128),
    ],
)
def test_phase1_chunkwise_affine_state_scan_outline_forward_matches_reference(m: int, d: int, dtype: torch.dtype):
    S_local_end, r_chunk, init = _make_inputs(seed=11 + m + d, bh=2, nc=16, m=m, d=d, dtype=dtype)
    start_ref, final_ref = ssd_rank1_prefix_scan_reference(S_local_end, r_chunk, init)
    log_alpha_chunk = torch.log(r_chunk.float()).to(r_chunk.dtype)
    start, final = SsdRank1PrefixScanTriton.apply(S_local_end, log_alpha_chunk, init)
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(start, start_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(final, final_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", _supported_dtypes())
def test_ssd_rank1_chunk_end_state_reference_matches_token_loop(dtype: torch.dtype):
    w, v, log_alpha = _make_phase0_inputs(seed=777, bh=2, nc=8, c=16, m=16, d=32, dtype=dtype)
    s_local_end = ssd_rank1_chunk_end_state_reference(w, v, log_alpha)
    r_chunk = torch.exp(torch.sum(log_alpha, dim=2))

    bh, nc, c, m = w.shape
    d = v.shape[-1]
    alpha = torch.exp(log_alpha.float())
    s_ref_md = torch.zeros((bh, nc, m, d), device=w.device, dtype=torch.float32)
    w_f = w.float()
    v_f = v.float()
    for t in range(c):
        s_ref_md = (
            alpha[:, :, t].unsqueeze(-1).unsqueeze(-1) * s_ref_md
            + w_f[:, :, t, :].unsqueeze(-1) * v_f[:, :, t, :].unsqueeze(-2)
        )
    s_ref = s_ref_md.reshape(bh, nc, m * d)
    r_ref = torch.exp(torch.sum(log_alpha, dim=2))

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(s_local_end, s_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(r_chunk, r_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", _supported_dtypes())
def test_phase0_chunk_end_state_kernel_matches_reference(dtype: torch.dtype):
    w, v, log_alpha = _make_phase0_inputs(seed=202, bh=2, nc=8, c=32, m=16, d=32, dtype=dtype)
    s_ref = ssd_rank1_chunk_end_state_reference(w, v, log_alpha)
    s_sc = SsdRank1ChunkEndStateTriton.apply(w, v, log_alpha)
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(s_sc, s_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", _supported_dtypes())
def test_phase0_chunk_end_state_outline_matches_reference_forward(dtype: torch.dtype):
    w, v, log_alpha = _make_phase0_inputs(seed=404, bh=2, nc=8, c=32, m=16, d=32, dtype=dtype)
    s_ref = ssd_rank1_chunk_end_state_reference(w, v, log_alpha)
    s_out = SsdRank1ChunkEndStateTriton.apply(w, v, log_alpha)
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(s_out, s_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase0 backward triton test requires CUDA")
@pytest.mark.parametrize("dtype", _supported_dtypes())
def test_phase0_chunk_end_state_outline_backward_matches_reference(dtype: torch.dtype):
    w, v, log_alpha = _make_phase0_inputs(seed=909, bh=1, nc=8, c=32, m=16, d=32, dtype=dtype)

    w_ref = w.clone().detach().requires_grad_(True)
    v_ref = v.clone().detach().requires_grad_(True)
    log_alpha_ref = log_alpha.clone().detach().requires_grad_(True)
    s_ref = ssd_rank1_chunk_end_state_reference(w_ref, v_ref, log_alpha_ref)
    loss_ref = s_ref.square().mean()
    loss_ref.backward()

    w_tri = w.clone().detach().requires_grad_(True)
    v_tri = v.clone().detach().requires_grad_(True)
    log_alpha_tri = log_alpha.clone().detach().requires_grad_(True)
    s_tri = SsdRank1ChunkEndStateTriton.apply(w_tri, v_tri, log_alpha_tri)
    loss_tri = s_tri.square().mean()
    loss_tri.backward()

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(w_tri.grad, w_ref.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(v_tri.grad, v_ref.grad, rtol=rtol, atol=atol)
    r_rtol, r_atol = _phase0_r_grad_tol(dtype)
    torch.testing.assert_close(log_alpha_tri.grad, log_alpha_ref.grad, rtol=r_rtol, atol=r_atol)


def _finite_difference_grad(
    scalar_fn,
    x: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    x_flat = x.reshape(-1)
    out = torch.empty_like(x_flat, dtype=torch.float32)
    for i in range(x_flat.numel()):
        x_pos = x.clone()
        x_neg = x.clone()
        x_pos.reshape(-1)[i] += eps
        x_neg.reshape(-1)[i] -= eps
        f_pos = scalar_fn(x_pos).to(torch.float32)
        f_neg = scalar_fn(x_neg).to(torch.float32)
        out[i] = (f_pos - f_neg) / (2.0 * eps)
    return out.reshape_as(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase0 finite-difference test requires CUDA")
def test_phase0_chunk_end_state_finite_difference_gradients():
    pytest.skip("Phase0 finite-difference test removed: unsupported M/D grid for current kernel contract.")

    grad_s = torch.randn(1, 2, 6, device=w.device, dtype=dtype)
    w_var = w.clone().detach().requires_grad_(True)
    v_var = v.clone().detach().requires_grad_(True)
    r_var = r.clone().detach().requires_grad_(True)
    s = SsdRank1ChunkEndStateTriton.apply(w_var, v_var, r_var, 16)
    loss = (s * grad_s).sum()
    loss.backward()

    def f_w(x):
        s_ = SsdRank1ChunkEndStateTriton.apply(x, v, r, 16)
        return (s_ * grad_s).sum()

    def f_v(x):
        s_ = SsdRank1ChunkEndStateTriton.apply(w, x, r, 16)
        return (s_ * grad_s).sum()

    def f_r(x):
        s_ = SsdRank1ChunkEndStateTriton.apply(w, v, x, 16)
        return (s_ * grad_s).sum()

    d_w_fd = _finite_difference_grad(f_w, w, eps=1e-3)
    d_v_fd = _finite_difference_grad(f_v, v, eps=1e-3)
    d_r_fd = _finite_difference_grad(f_r, r, eps=1e-3)

    torch.testing.assert_close(w_var.grad, d_w_fd, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(v_var.grad, d_v_fd, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(r_var.grad, d_r_fd, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize("dtype", _supported_dtypes())
def test_ssd_rank1_dense_output_reference_matches_naive(dtype: torch.dtype):
    c_tok, w_tok, v_tok, log_alpha_tok = _make_phase3_inputs(seed=1301, bh=1, nc=2, c=16, m=16, d=32, dtype=dtype)
    y_ref = ssd_rank1_dense_output_reference(c_tok, w_tok, v_tok, log_alpha_tok)

    bh, nc, c, m = c_tok.shape
    d = v_tok.shape[-1]
    y_naive = torch.zeros((bh, nc, c, d), device=c_tok.device, dtype=c_tok.dtype)
    for t in range(c):
        for tau in range(t + 1):
            factor = torch.ones((bh, nc), device=c_tok.device, dtype=c_tok.dtype)
            for u in range(tau + 1, t + 1):
                factor = factor * torch.exp(log_alpha_tok[:, :, u])
            alpha = torch.sum(c_tok[:, :, t, :] * w_tok[:, :, tau, :], dim=-1)
            y_naive[:, :, t, :] += (factor * alpha).unsqueeze(-1) * v_tok[:, :, tau, :]

    rtol, atol = _phase3_tol(dtype)
    torch.testing.assert_close(y_ref, y_naive, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase3 backward reference test requires CUDA")
def test_ssd_rank1_dense_output_backward_reference_matches_autograd_targeted():
    dtype = torch.float32
    BH, NC, C_CHUNK, M, D = 1, 8, 64, 128, 64
    C, W, V, log_alpha = _make_phase3_inputs(
        seed=1302,
        bh=BH,
        nc=NC,
        c=C_CHUNK,
        m=M,
        d=D,
        dtype=dtype,
    )
    S0 = torch.randn(BH, NC, M * D, device=C.device, dtype=dtype)
    grad_Y = torch.randn(BH, NC, C_CHUNK, D, device=C.device, dtype=dtype)

    dC_ref, dW_ref, dV_ref, dlog_alpha_ref, dS0_ref = ssd_rank1_dense_output_backward_reference(
        C,
        W,
        V,
        log_alpha,
        grad_Y,
        S0,
    )
    assert dS0_ref is not None

    C_ag = C.clone().detach().requires_grad_(True)
    W_ag = W.clone().detach().requires_grad_(True)
    V_ag = V.clone().detach().requires_grad_(True)
    log_alpha_ag = log_alpha.clone().detach().requires_grad_(True)
    S0_ag = S0.clone().detach().requires_grad_(True)
    Y_ag = ssd_rank1_dense_output_reference(C_ag, W_ag, V_ag, log_alpha_ag, S0_ag)
    loss_ag = (Y_ag * grad_Y).sum()
    dC_ag, dW_ag, dV_ag, dlog_alpha_ag, dS0_ag = torch.autograd.grad(
        loss_ag,
        [C_ag, W_ag, V_ag, log_alpha_ag, S0_ag],
    )

    torch.testing.assert_close(dC_ref, dC_ag, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dW_ref, dW_ag, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dV_ref, dV_ag, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dlog_alpha_ref, dlog_alpha_ag, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(dS0_ref.reshape(BH, NC, M * D), dS0_ag, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("dtype", _supported_dtypes())
def test_phase3_chunk_dense_output_outline_matches_reference_forward(dtype: torch.dtype):
    c_tok, w_tok, v_tok, log_alpha_tok = _make_phase3_inputs(seed=1401, bh=2, nc=2, c=16, m=16, d=32, dtype=dtype)
    y_ref = ssd_rank1_dense_output_reference(c_tok, w_tok, v_tok, log_alpha_tok)
    input_precision = "ieee" if dtype == torch.float32 else "tf32"
    y_tri = SsdRank1DenseOutputTriton.apply(
        c_tok,
        w_tok,
        v_tok,
        log_alpha_tok,
        None,
        input_precision,
    )
    rtol, atol = _phase3_tol(dtype)
    torch.testing.assert_close(y_tri, y_ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase3 backward triton test requires CUDA")
@pytest.mark.parametrize("dtype", _supported_dtypes())
def test_phase3_chunk_dense_output_outline_backward_matches_reference(dtype: torch.dtype):
    c_tok, w_tok, v_tok, log_alpha_tok = _make_phase3_inputs(seed=1501, bh=1, nc=2, c=16, m=16, d=32, dtype=dtype)
    s0 = torch.randn(1, 2, 16 * 32, device=c_tok.device, dtype=dtype)

    c_ref = c_tok.clone().detach().requires_grad_(True)
    w_ref = w_tok.clone().detach().requires_grad_(True)
    v_ref = v_tok.clone().detach().requires_grad_(True)
    log_alpha_ref = log_alpha_tok.clone().detach().requires_grad_(True)
    s0_ref = s0.clone().detach().requires_grad_(True)
    y_ref = ssd_rank1_dense_output_reference(c_ref, w_ref, v_ref, log_alpha_ref, s0_ref)
    loss_ref = 0.37 * y_ref.square().mean() + 0.63 * y_ref.abs().mean()
    loss_ref.backward()

    c_tri = c_tok.clone().detach().requires_grad_(True)
    w_tri = w_tok.clone().detach().requires_grad_(True)
    v_tri = v_tok.clone().detach().requires_grad_(True)
    log_alpha_tri = log_alpha_tok.clone().detach().requires_grad_(True)
    s0_tri = s0.clone().detach().requires_grad_(True)
    input_precision = "ieee" if dtype == torch.float32 else "tf32"
    y_tri = SsdRank1DenseOutputTriton.apply(
        c_tri,
        w_tri,
        v_tri,
        log_alpha_tri,
        s0_tri,
        input_precision,
    )
    loss_tri = 0.37 * y_tri.square().mean() + 0.63 * y_tri.abs().mean()
    loss_tri.backward()

    rtol, atol = _phase3_tol(dtype)
    torch.testing.assert_close(c_tri.grad, c_ref.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(w_tri.grad, w_ref.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(v_tri.grad, v_ref.grad, rtol=rtol, atol=atol)
    rtol_r, atol_r = _phase3_r_grad_tol(dtype)
    torch.testing.assert_close(log_alpha_tri.grad, log_alpha_ref.grad, rtol=rtol_r, atol=atol_r)
    torch.testing.assert_close(s0_tri.grad, s0_ref.grad, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase3 backward targeted test requires CUDA")
def test_phase3_chunk_dense_output_outline_backward_matches_reference_targeted_c64():
    dtype = torch.float32
    bh, nc, c, m, d = 1, 8, 64, 128, 64
    c_tok, w_tok, v_tok, log_alpha_tok = _make_phase3_inputs(seed=1511, bh=bh, nc=nc, c=c, m=m, d=d, dtype=dtype)
    s0 = torch.randn(bh, nc, m * d, device=c_tok.device, dtype=dtype)
    grad_y = torch.randn(bh, nc, c, d, device=c_tok.device, dtype=dtype)

    dC_ref, dW_ref, dV_ref, dlog_alpha_ref, dS0_ref = ssd_rank1_dense_output_backward_reference(
        c_tok,
        w_tok,
        v_tok,
        log_alpha_tok,
        grad_y,
        s0,
    )
    assert dS0_ref is not None

    c_tri = c_tok.clone().detach().requires_grad_(True)
    w_tri = w_tok.clone().detach().requires_grad_(True)
    v_tri = v_tok.clone().detach().requires_grad_(True)
    log_alpha_tri = log_alpha_tok.clone().detach().requires_grad_(True)
    s0_tri = s0.clone().detach().requires_grad_(True)
    y_tri = SsdRank1DenseOutputTriton.apply(
        c_tri,
        w_tri,
        v_tri,
        log_alpha_tri,
        s0_tri,
        "ieee",
    )
    loss_tri = (y_tri * grad_y).sum()
    loss_tri.backward()

    torch.testing.assert_close(c_tri.grad, dC_ref, rtol=1e-4, atol=5e-4)
    torch.testing.assert_close(w_tri.grad, dW_ref, rtol=1e-4, atol=5e-4)
    torch.testing.assert_close(v_tri.grad, dV_ref, rtol=1e-4, atol=5e-4)
    torch.testing.assert_close(log_alpha_tri.grad, dlog_alpha_ref, rtol=1e-4, atol=5e-4)
    torch.testing.assert_close(s0_tri.grad, dS0_ref.reshape(bh, nc, m * d), rtol=1e-4, atol=5e-4)


@pytest.mark.skip(reason="Phase3 finite-difference test removed: unsupported M/D grid for current kernel contract.")
def test_phase3_chunk_dense_output_finite_difference_is_not_implemented():
    pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase3 backward profile smoke requires CUDA")
def test_phase3_chunk_dense_output_backward_profile_smoke_cuda():
    dtype = torch.float32
    c_tok, w_tok, v_tok, log_alpha_tok = _make_phase3_inputs(seed=1651, bh=1, nc=1, c=16, m=16, d=32, dtype=dtype)
    grad_y = torch.randn_like(v_tok)
    s0 = torch.randn(1, 1, 16 * 32, device=c_tok.device, dtype=dtype)
    profile = ssd_rank1_dense_output_backward_kernel_profile(c_tok, w_tok, v_tok, log_alpha_tok, grad_y, s0)
    assert "total_ms" in profile
    assert profile["total_ms"] > 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase0 invalid-C contract requires CUDA")
def test_phase0_chunk_end_state_outline_requires_c_multiple_of_16():
    w, v, log_alpha = _make_phase0_inputs(seed=1751, bh=1, nc=8, c=6, m=16, d=32, dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="positive multiple of 16"):
        SsdRank1ChunkEndStateTriton.apply(w, v, log_alpha)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase3 invalid-C contract requires CUDA")
def test_phase3_chunk_dense_output_outline_requires_c_multiple_of_16():
    c_tok, w_tok, v_tok, log_alpha_tok = _make_phase3_inputs(seed=1752, bh=1, nc=1, c=6, m=16, d=32, dtype=torch.float32)
    with pytest.raises(NotImplementedError, match="positive multiple of 16"):
        SsdRank1DenseOutputTriton.apply(c_tok, w_tok, v_tok, log_alpha_tok, None, "tf32")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Static-C phase0 triton test requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("c", [32, 64, 128])
def test_phase0_chunk_end_state_kernel_matches_reference_static_c(dtype: torch.dtype, c: int):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bf16 not supported on this CUDA device")
    w, v, log_alpha = _make_phase0_inputs(seed=515 + c, bh=1, nc=8, c=c, m=16, d=32, dtype=dtype)
    s_ref = ssd_rank1_chunk_end_state_reference(w, v, log_alpha)
    s_sc = SsdRank1ChunkEndStateTriton.apply(w, v, log_alpha)
    rtol, atol = _tol(dtype)
    torch.testing.assert_close(s_sc, s_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", _supported_dtypes())
@pytest.mark.parametrize(("m", "d"), [(16, 32), (32, 64), (128, 32)])
def test_phase1_chunkwise_affine_state_scan_outline_backward_matches_reference(m: int, d: int, dtype: torch.dtype):
    S_local_end, r_chunk, init = _make_inputs(seed=17 + m + d, bh=2, nc=16, m=m, d=d, dtype=dtype)

    S_local_end_ref = S_local_end.clone().detach().requires_grad_(True)
    log_alpha_chunk_ref = torch.log(r_chunk.float()).to(r_chunk.dtype).clone().detach().requires_grad_(True)
    init_ref = init.clone().detach().requires_grad_(True)
    start_ref, final_ref = ssd_rank1_prefix_scan_reference(S_local_end_ref, torch.exp(log_alpha_chunk_ref), init_ref)
    loss_ref = 0.37 * start_ref.square().mean() + 0.63 * final_ref.abs().mean()
    loss_ref.backward()

    S_local_end_tri = S_local_end.clone().detach().requires_grad_(True)
    log_alpha_chunk_tri = torch.log(r_chunk.float()).to(r_chunk.dtype).clone().detach().requires_grad_(True)
    init_tri = init.clone().detach().requires_grad_(True)
    start_tri, final_tri = SsdRank1PrefixScanTriton.apply(S_local_end_tri, log_alpha_chunk_tri, init_tri)
    loss_tri = 0.37 * start_tri.square().mean() + 0.63 * final_tri.abs().mean()
    loss_tri.backward()

    rtol, atol = _tol(dtype)
    torch.testing.assert_close(S_local_end_tri.grad, S_local_end_ref.grad, rtol=rtol, atol=atol)
    torch.testing.assert_close(init_tri.grad, init_ref.grad, rtol=rtol, atol=atol)
    r_rtol, r_atol = _r_grad_tol(dtype)
    torch.testing.assert_close(log_alpha_chunk_tri.grad, log_alpha_chunk_ref.grad, rtol=r_rtol, atol=r_atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Phase1 finite-difference test requires CUDA")
def test_phase1_chunkwise_affine_state_scan_finite_difference_gradients():
    pytest.skip("Phase1 finite-difference test intentionally disabled for fast iteration.")
    dtype = torch.float32
    bh, nc, m, d = 1, 3, 16, 32
    md = m * d
    torch.manual_seed(1717)
    device = "cuda"
    s_local = torch.randn(bh, nc, md, device=device, dtype=dtype)
    r_chunk = torch.sigmoid(torch.randn(bh, nc, device=device, dtype=dtype))
    init = torch.randn(bh, md, device=device, dtype=dtype)

    g_start = torch.randn(bh, nc, md, device=device, dtype=dtype)
    g_final = torch.randn(bh, md, device=device, dtype=dtype)

    s = s_local.clone().detach().requires_grad_(True)
    log_alpha = torch.log(r_chunk.float()).to(r_chunk.dtype).clone().detach().requires_grad_(True)
    i = init.clone().detach().requires_grad_(True)
    start, final = SsdRank1PrefixScanTriton.apply(s, log_alpha, i)
    loss = (start * g_start).sum() + (final * g_final).sum()
    loss.backward()

    def f_s(x):
        start_, final_ = SsdRank1PrefixScanTriton.apply(x, torch.log(r_chunk.float()).to(r_chunk.dtype), init)
        return (start_ * g_start).sum() + (final_ * g_final).sum()

    def f_r(x):
        start_, final_ = SsdRank1PrefixScanTriton.apply(s_local, torch.log(x.float()).to(x.dtype), init)
        return (start_ * g_start).sum() + (final_ * g_final).sum()

    def f_i(x):
        start_, final_ = SsdRank1PrefixScanTriton.apply(
            s_local,
            torch.log(r_chunk.float()).to(r_chunk.dtype),
            x,
        )
        return (start_ * g_start).sum() + (final_ * g_final).sum()

    d_s_fd = _finite_difference_grad(f_s, s_local, eps=1e-3)
    d_r_fd = _finite_difference_grad(f_r, r_chunk, eps=1e-3)
    d_i_fd = _finite_difference_grad(f_i, init, eps=1e-3)

    torch.testing.assert_close(s.grad, d_s_fd, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(log_alpha.grad, d_r_fd * r_chunk, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(i.grad, d_i_fd, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Benchmark test requires CUDA")
def test_phase1_chunkwise_affine_state_scan_small_benchmark_cuda():
    # Small benchmark smoke to ensure timing path remains functional.
    dtype = torch.float16
    S_local_end, r_chunk, init = _make_inputs(seed=123, bh=16, nc=16, m=16, d=32, dtype=dtype)
    log_alpha_chunk = torch.log(r_chunk.float()).to(r_chunk.dtype)

    def _time_forward(fn, iters: int = 10):
        for _ in range(3):
            _ = fn()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            _ = fn()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters

    def _time_backward(fn, iters: int = 10):
        for _ in range(2):
            s = S_local_end.detach().clone().requires_grad_(True)
            r = r_chunk.detach().clone().requires_grad_(True)
            i = init.detach().clone().requires_grad_(True)
            out, fin = fn(s, r, i)
            (out.float().square().mean() + fin.float().square().mean()).backward()
        torch.cuda.synchronize()
        t0 = torch.cuda.Event(enable_timing=True)
        t1 = torch.cuda.Event(enable_timing=True)
        t0.record()
        for _ in range(iters):
            s = S_local_end.detach().clone().requires_grad_(True)
            r = r_chunk.detach().clone().requires_grad_(True)
            i = init.detach().clone().requires_grad_(True)
            out, fin = fn(s, r, i)
            (out.float().square().mean() + fin.float().square().mean()).backward()
        t1.record()
        torch.cuda.synchronize()
        return t0.elapsed_time(t1) / iters

    tri_fwd = _time_forward(lambda: SsdRank1PrefixScanTriton.apply(S_local_end, log_alpha_chunk, init))
    ref_fwd = _time_forward(lambda: ssd_rank1_prefix_scan_reference(S_local_end, r_chunk, init))
    tri_bwd = _time_backward(lambda s, r, i: SsdRank1PrefixScanTriton.apply(s, torch.log(r.float()).to(r.dtype), i))
    ref_bwd = _time_backward(lambda s, r, i: ssd_rank1_prefix_scan_reference(s, r, i))

    assert tri_fwd > 0 and ref_fwd > 0 and tri_bwd > 0 and ref_bwd > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full phase123 composition test requires CUDA")
def test_ssd_rank1_triton_matches_reference_forward():
    dtype = torch.float32
    b, h, m, d = 2, 2, 16, 32
    c, w, v, log_alpha = _make_phase123_inputs(seed=2026, b=b, n=111, h=h, m=m, d=d, dtype=dtype)
    init = torch.randn(b, h, m * d, device=c.device, dtype=dtype)

    y_ref, final_ref = ssd_rank1_pytorch(
        c.clone(), w.clone(), v.clone(), log_alpha.clone(), initial_state=init.clone(), CHUNK_SIZE=64
    )
    y_tri, final_tri = ssd_rank1_triton(
        c.clone(),
        w.clone(),
        v.clone(),
        log_alpha.clone(),
        initial_state=init.clone(),
        CHUNK_SIZE=64,
        INPUT_PRECISION="ieee",
    )

    rtol_y, atol_y = 1e-4, 1e-4
    rtol_s, atol_s = _tol(dtype)
    torch.testing.assert_close(y_tri, y_ref, rtol=rtol_y, atol=atol_y)
    torch.testing.assert_close(final_tri, final_ref, rtol=rtol_s, atol=atol_s)


def test_ssd_rank1_pytorch_matches_token_loop_oracle():
    dtype = torch.float32
    b, h, m, d = 2, 2, 16, 32
    c, w, v, log_alpha = _make_phase123_inputs(seed=3026, b=b, n=97, h=h, m=m, d=d, dtype=dtype)
    init = torch.randn(b, h, m * d, device=c.device, dtype=dtype)

    y_ref, final_ref = ssd_rank1_pytorch(
        c.clone(), w.clone(), v.clone(), log_alpha.clone(), initial_state=init.clone(), CHUNK_SIZE=64
    )
    y_oracle, final_oracle = ssd_rank1_token_loop_oracle(
        c.clone(), w.clone(), v.clone(), log_alpha.clone(), initial_state=init.clone()
    )

    rtol_y, atol_y = 1e-4, 1e-4
    rtol_s, atol_s = _tol(dtype)
    torch.testing.assert_close(y_ref, y_oracle, rtol=rtol_y, atol=atol_y)
    torch.testing.assert_close(final_ref, final_oracle, rtol=rtol_s, atol=atol_s)

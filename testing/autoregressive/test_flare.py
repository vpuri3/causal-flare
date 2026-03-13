import os
import math

import pytest
import torch

from causal_flare import flare_autoregressive_triton, flare_recurrent_triton
from causal_flare.autoregressive.inference import (
    flare_decode_pytorch,
    flare_decode_triton,
    flare_prefill_pytorch,
    flare_prefill_triton,
)
from causal_flare.autoregressive.reference import (
    flare_causal_chunked,
    flare_recurrent_dense_backward_pytorch,
    flare_recurrent_pytorch,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
device = torch.device("cuda")


def _run_recurrent_impl(impl_name, q, k, v, *, scale, q_dec=None, k_dec=None, block_t=16):
    if impl_name == "reference":
        return flare_recurrent_pytorch(q, k, v, scale=scale, Q_dec=q_dec, K_dec=k_dec)
    if impl_name == "recurrent":
        return flare_recurrent_triton(q, k, v, scale=scale, Q_dec=q_dec, K_dec=k_dec, block_t=block_t)
    if impl_name == "recurrent_assoc":
        old_impl = os.environ.get("FLARE_RECURRENT_IMPL")
        old_scan = os.environ.get("FLARE_RECURRENT_ASSOC_SCAN")
        try:
            os.environ.pop("FLARE_RECURRENT_IMPL", None)
            os.environ["FLARE_RECURRENT_ASSOC_SCAN"] = "1"
            return flare_recurrent_triton(q, k, v, scale=scale, Q_dec=q_dec, K_dec=k_dec, block_t=block_t)
        finally:
            if old_impl is None:
                os.environ.pop("FLARE_RECURRENT_IMPL", None)
            else:
                os.environ["FLARE_RECURRENT_IMPL"] = old_impl
            if old_scan is None:
                os.environ.pop("FLARE_RECURRENT_ASSOC_SCAN", None)
            else:
                os.environ["FLARE_RECURRENT_ASSOC_SCAN"] = old_scan
    if impl_name == "recurrent_multi":
        old_impl = os.environ.get("FLARE_RECURRENT_IMPL")
        old_scan = os.environ.get("FLARE_RECURRENT_ASSOC_SCAN")
        try:
            os.environ["FLARE_RECURRENT_IMPL"] = "multi"
            os.environ.pop("FLARE_RECURRENT_ASSOC_SCAN", None)
            return flare_recurrent_triton(q, k, v, scale=scale, Q_dec=q_dec, K_dec=k_dec, block_t=block_t)
        finally:
            if old_impl is None:
                os.environ.pop("FLARE_RECURRENT_IMPL", None)
            else:
                os.environ["FLARE_RECURRENT_IMPL"] = old_impl
            if old_scan is None:
                os.environ.pop("FLARE_RECURRENT_ASSOC_SCAN", None)
            else:
                os.environ["FLARE_RECURRENT_ASSOC_SCAN"] = old_scan
    raise ValueError(f"Unknown implementation: {impl_name}")


def _run_chunked_impl(impl_name, q, k, v, *, scale, q_dec=None, k_dec=None, chunk_size=None):
    if impl_name == "reference":
        return flare_causal_chunked(q, k, v, scale=scale, chunk_size=chunk_size, Q_dec=q_dec, K_dec=k_dec)
    if impl_name == "chunked":
        return flare_autoregressive_triton(Q=q, K=k, V=v, scale=scale, chunk_size=chunk_size, Q_dec=q_dec, K_dec=k_dec)
    raise ValueError(f"Unknown implementation: {impl_name}")


@pytest.mark.parametrize("impl_name", ["recurrent"])
@pytest.mark.parametrize("block_t", [1, 16])
def test_recurrent_flare_decode_variants_match_torch_reference(impl_name: str, block_t: int):
    torch.manual_seed(123)

    B = 1
    N = 7
    H = 2
    M = 16
    D = 16
    scale = 1.0 / math.sqrt(D)
    dtype = torch.float32

    q = torch.randn((H, M, D), device=device, dtype=dtype)
    k = torch.randn((B, N, H, D), device=device, dtype=dtype)
    v = torch.randn((B, N, H, D), device=device, dtype=dtype)
    q_dec_rand = torch.randn((B, N, H, D), device=device, dtype=dtype)
    k_dec_rand = torch.randn((H, M, D), device=device, dtype=dtype)

    q_dec_options = (None, k, q_dec_rand)
    k_dec_options = (None, q, k_dec_rand)

    for q_dec in q_dec_options:
        for k_dec in k_dec_options:
            y_ref = _run_recurrent_impl("reference", q, k, v, scale=scale, q_dec=q_dec, k_dec=k_dec, block_t=block_t)
            y_impl = _run_recurrent_impl(impl_name, q, k, v, scale=scale, q_dec=q_dec, k_dec=k_dec, block_t=block_t)
            torch.testing.assert_close(y_impl, y_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("impl_name", ["recurrent_assoc", "recurrent_multi"])
def test_recurrent_experimental_paths_match_torch_reference(impl_name: str):
    torch.manual_seed(7123)

    B = 1
    N = 16
    H = 2
    M = 16
    D = 16
    scale = 1.0 / math.sqrt(D)
    dtype = torch.float32

    q = torch.randn((H, M, D), device=device, dtype=dtype)
    k = torch.randn((B, N, H, D), device=device, dtype=dtype)
    v = torch.randn((B, N, H, D), device=device, dtype=dtype)
    q_dec_rand = torch.randn((B, N, H, D), device=device, dtype=dtype)
    k_dec_rand = torch.randn((H, M, D), device=device, dtype=dtype)

    q_dec_options = (None, k, q_dec_rand)
    k_dec_options = (None, q, k_dec_rand)

    for q_dec in q_dec_options:
        for k_dec in k_dec_options:
            y_ref = _run_recurrent_impl("reference", q, k, v, scale=scale, q_dec=q_dec, k_dec=k_dec, block_t=16)
            y_impl = _run_recurrent_impl(impl_name, q, k, v, scale=scale, q_dec=q_dec, k_dec=k_dec, block_t=16)
            torch.testing.assert_close(y_impl, y_ref, rtol=1e-4, atol=1e-5)


def _run_recurrent_impl_with_grads(
    impl_name,
    q_seed,
    k_seed,
    v_seed,
    *,
    scale,
    grad_out,
    q_dec_mode,
    k_dec_mode,
    q_dec_rand_seed,
    k_dec_rand_seed,
    block_t,
):
    q = q_seed.clone().requires_grad_(True)
    k = k_seed.clone().requires_grad_(True)
    v = v_seed.clone().requires_grad_(True)

    q_dec_rand = None
    if q_dec_mode == "rand":
        q_dec_rand = q_dec_rand_seed.clone().requires_grad_(True)
        q_dec = q_dec_rand
    elif q_dec_mode == "k_enc":
        q_dec = k
    elif q_dec_mode == "none":
        q_dec = None
    else:
        raise ValueError(f"Unknown q_dec_mode: {q_dec_mode}")

    k_dec_rand = None
    if k_dec_mode == "rand":
        k_dec_rand = k_dec_rand_seed.clone().requires_grad_(True)
        k_dec = k_dec_rand
    elif k_dec_mode == "q_enc":
        k_dec = q
    elif k_dec_mode == "none":
        k_dec = None
    else:
        raise ValueError(f"Unknown k_dec_mode: {k_dec_mode}")

    y = _run_recurrent_impl(impl_name, q, k, v, scale=scale, q_dec=q_dec, k_dec=k_dec, block_t=block_t)
    (y * grad_out).sum().backward()

    grads = {
        "q": q.grad.detach(),
        "k": k.grad.detach(),
        "v": v.grad.detach(),
    }
    if q_dec_rand is not None:
        grads["q_dec_rand"] = q_dec_rand.grad.detach()
    if k_dec_rand is not None:
        grads["k_dec_rand"] = k_dec_rand.grad.detach()

    return y.detach(), grads


def _run_chunked_impl_with_grads(
    impl_name,
    q_seed,
    k_seed,
    v_seed,
    *,
    scale,
    grad_out,
    separate_q_dec,
    separate_k_dec,
    q_dec_rand_seed,
    k_dec_rand_seed,
    chunk_size,
):
    q = q_seed.clone().requires_grad_(True)
    k = k_seed.clone().requires_grad_(True)
    v = v_seed.clone().requires_grad_(True)

    q_dec_rand = q_dec_rand_seed.clone().requires_grad_(True) if separate_q_dec else None
    k_dec_rand = k_dec_rand_seed.clone().requires_grad_(True) if separate_k_dec else None
    q_dec = q_dec_rand if separate_q_dec else None
    k_dec = k_dec_rand if separate_k_dec else None

    y = _run_chunked_impl(impl_name, q, k, v, scale=scale, q_dec=q_dec, k_dec=k_dec, chunk_size=chunk_size)
    (y * grad_out).sum().backward()

    grads = {
        "q": q.grad.detach(),
        "k": k.grad.detach(),
        "v": v.grad.detach(),
    }
    if q_dec_rand is not None:
        grads["q_dec_rand"] = q_dec_rand.grad.detach()
    if k_dec_rand is not None:
        grads["k_dec_rand"] = k_dec_rand.grad.detach()
    return y.detach(), grads


@pytest.mark.parametrize("impl_name", ["recurrent"])
@pytest.mark.parametrize("block_t", [1, 16])
def test_recurrent_flare_decode_variants_backward_match_torch_reference(impl_name: str, block_t: int):
    torch.manual_seed(321)

    B = 1
    N = 16
    H = 2
    M = 16
    D = 16
    scale = 1.0 / math.sqrt(D)
    dtype = torch.float32

    q_seed = torch.randn((H, M, D), device=device, dtype=dtype)
    k_seed = torch.randn((B, N, H, D), device=device, dtype=dtype)
    v_seed = torch.randn((B, N, H, D), device=device, dtype=dtype)
    q_dec_rand_seed = torch.randn((B, N, H, D), device=device, dtype=dtype)
    k_dec_rand_seed = torch.randn((H, M, D), device=device, dtype=dtype)
    grad_out = torch.randn((B, H, N, D), device=device, dtype=dtype)

    q_dec_modes = ("none", "k_enc", "rand")
    k_dec_modes = ("none", "q_enc", "rand")

    for q_dec_mode in q_dec_modes:
        for k_dec_mode in k_dec_modes:
            y_ref, g_ref = _run_recurrent_impl_with_grads(
                "reference",
                q_seed,
                k_seed,
                v_seed,
                scale=scale,
                grad_out=grad_out,
                q_dec_mode=q_dec_mode,
                k_dec_mode=k_dec_mode,
                q_dec_rand_seed=q_dec_rand_seed,
                k_dec_rand_seed=k_dec_rand_seed,
                block_t=block_t,
            )
            y_impl, g_impl = _run_recurrent_impl_with_grads(
                impl_name,
                q_seed,
                k_seed,
                v_seed,
                scale=scale,
                grad_out=grad_out,
                q_dec_mode=q_dec_mode,
                k_dec_mode=k_dec_mode,
                q_dec_rand_seed=q_dec_rand_seed,
                k_dec_rand_seed=k_dec_rand_seed,
                block_t=block_t,
            )

            torch.testing.assert_close(y_impl, y_ref, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(g_impl["q"], g_ref["q"], rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(g_impl["k"], g_ref["k"], rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(g_impl["v"], g_ref["v"], rtol=1e-2, atol=1e-3)
            if q_dec_mode == "rand":
                torch.testing.assert_close(g_impl["q_dec_rand"], g_ref["q_dec_rand"], rtol=1e-2, atol=1e-3)
            if k_dec_mode == "rand":
                torch.testing.assert_close(g_impl["k_dec_rand"], g_ref["k_dec_rand"], rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("impl_name", ["recurrent_multi"])
def test_recurrent_multi_kernel_backward_matches_torch_reference(impl_name: str):
    torch.manual_seed(7124)

    B = 1
    N = 16
    H = 2
    M = 16
    D = 16
    scale = 1.0 / math.sqrt(D)
    dtype = torch.float32

    q_seed = torch.randn((H, M, D), device=device, dtype=dtype)
    k_seed = torch.randn((B, N, H, D), device=device, dtype=dtype)
    v_seed = torch.randn((B, N, H, D), device=device, dtype=dtype)
    q_dec_rand_seed = torch.randn((B, N, H, D), device=device, dtype=dtype)
    k_dec_rand_seed = torch.randn((H, M, D), device=device, dtype=dtype)
    grad_out = torch.randn((B, H, N, D), device=device, dtype=dtype)

    for q_dec_mode in ("none", "k_enc", "rand"):
        for k_dec_mode in ("none", "q_enc", "rand"):
            y_ref, g_ref = _run_recurrent_impl_with_grads(
                "reference",
                q_seed,
                k_seed,
                v_seed,
                scale=scale,
                grad_out=grad_out,
                q_dec_mode=q_dec_mode,
                k_dec_mode=k_dec_mode,
                q_dec_rand_seed=q_dec_rand_seed,
                k_dec_rand_seed=k_dec_rand_seed,
                block_t=16,
            )
            y_impl, g_impl = _run_recurrent_impl_with_grads(
                impl_name,
                q_seed,
                k_seed,
                v_seed,
                scale=scale,
                grad_out=grad_out,
                q_dec_mode=q_dec_mode,
                k_dec_mode=k_dec_mode,
                q_dec_rand_seed=q_dec_rand_seed,
                k_dec_rand_seed=k_dec_rand_seed,
                block_t=16,
            )

            torch.testing.assert_close(y_impl, y_ref, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(g_impl["q"], g_ref["q"], rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(g_impl["k"], g_ref["k"], rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(g_impl["v"], g_ref["v"], rtol=1e-2, atol=1e-3)
            if q_dec_mode == "rand":
                torch.testing.assert_close(g_impl["q_dec_rand"], g_ref["q_dec_rand"], rtol=1e-2, atol=1e-3)
            if k_dec_mode == "rand":
                torch.testing.assert_close(g_impl["k_dec_rand"], g_ref["k_dec_rand"], rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("impl_name", ["chunked"])
@pytest.mark.parametrize("d_score,d_value", [(16, 16), (64, 128)])
def test_chunked_flare_decode_separate_flags_forward_match_torch_reference(impl_name: str, d_score: int, d_value: int):
    torch.manual_seed(2028)

    B = 1
    N = 16
    H = 2
    M = 16
    scale = 1.0 / math.sqrt(d_score)
    dtype = torch.float32
    chunk_size = 16

    q = torch.randn((H, M, d_score), device=device, dtype=dtype)
    k = torch.randn((B, N, H, d_score), device=device, dtype=dtype)
    v = torch.randn((B, N, H, d_value), device=device, dtype=dtype)
    q_dec_rand = torch.randn((B, N, H, d_score), device=device, dtype=dtype)
    k_dec_rand = torch.randn((H, M, d_score), device=device, dtype=dtype)

    for separate_q_dec in (False, True):
        for separate_k_dec in (False, True):
            q_dec = q_dec_rand if separate_q_dec else None
            k_dec = k_dec_rand if separate_k_dec else None
            y_ref = _run_chunked_impl("reference", q, k, v, scale=scale, q_dec=q_dec, k_dec=k_dec, chunk_size=chunk_size)
            y_impl = _run_chunked_impl(impl_name, q, k, v, scale=scale, q_dec=q_dec, k_dec=k_dec, chunk_size=chunk_size)
            torch.testing.assert_close(y_impl, y_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("impl_name", ["chunked"])
@pytest.mark.parametrize("d_score,d_value", [(16, 16), (64, 128)])
def test_chunked_flare_decode_separate_flags_backward_match_torch_reference(impl_name: str, d_score: int, d_value: int):
    torch.manual_seed(2029)

    B = 1
    N = 16
    H = 2
    M = 16
    scale = 1.0 / math.sqrt(d_score)
    dtype = torch.float32
    chunk_size = 16

    q_seed = torch.randn((H, M, d_score), device=device, dtype=dtype)
    k_seed = torch.randn((B, N, H, d_score), device=device, dtype=dtype)
    v_seed = torch.randn((B, N, H, d_value), device=device, dtype=dtype)
    q_dec_rand_seed = torch.randn((B, N, H, d_score), device=device, dtype=dtype)
    k_dec_rand_seed = torch.randn((H, M, d_score), device=device, dtype=dtype)
    grad_out = torch.randn((B, N, H, d_value), device=device, dtype=dtype)

    for separate_q_dec in (False, True):
        for separate_k_dec in (False, True):
            y_ref, g_ref = _run_chunked_impl_with_grads(
                "reference",
                q_seed,
                k_seed,
                v_seed,
                scale=scale,
                grad_out=grad_out,
                separate_q_dec=separate_q_dec,
                separate_k_dec=separate_k_dec,
                q_dec_rand_seed=q_dec_rand_seed,
                k_dec_rand_seed=k_dec_rand_seed,
                chunk_size=chunk_size,
            )
            y_impl, g_impl = _run_chunked_impl_with_grads(
                impl_name,
                q_seed,
                k_seed,
                v_seed,
                scale=scale,
                grad_out=grad_out,
                separate_q_dec=separate_q_dec,
                separate_k_dec=separate_k_dec,
                q_dec_rand_seed=q_dec_rand_seed,
                k_dec_rand_seed=k_dec_rand_seed,
                chunk_size=chunk_size,
            )

            torch.testing.assert_close(y_impl, y_ref, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(g_impl["q"], g_ref["q"], rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(g_impl["k"], g_ref["k"], rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(g_impl["v"], g_ref["v"], rtol=1e-2, atol=1e-3)

            if separate_q_dec:
                torch.testing.assert_close(g_impl["q_dec_rand"], g_ref["q_dec_rand"], rtol=1e-2, atol=1e-3)
            if separate_k_dec:
                torch.testing.assert_close(g_impl["k_dec_rand"], g_ref["k_dec_rand"], rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("q_dec_mode", ["none", "k_enc", "rand"])
@pytest.mark.parametrize("k_dec_mode", ["none", "q_enc", "rand"])
@pytest.mark.parametrize("d_score,d_value", [(16, 16), (64, 128)])
def test_inference_prefill_decode_variants_match_pytorch(q_dec_mode: str, k_dec_mode: str, d_score: int, d_value: int):
    torch.manual_seed(1337)

    B = 1
    N = 16
    H = 2
    M = 16
    scale = 1.0 / math.sqrt(d_score)
    dtype = torch.float32

    q = torch.randn((H, M, d_score), device=device, dtype=dtype)
    k = torch.randn((B, N, H, d_score), device=device, dtype=dtype)
    v = torch.randn((B, N, H, d_value), device=device, dtype=dtype)
    q_dec_rand = torch.randn((B, N, H, d_score), device=device, dtype=dtype)
    k_dec_rand = torch.randn((H, M, d_score), device=device, dtype=dtype)

    if q_dec_mode == "none":
        q_dec = None
    elif q_dec_mode == "k_enc":
        q_dec = k
    elif q_dec_mode == "rand":
        q_dec = q_dec_rand
    else:
        raise ValueError(f"Unknown q_dec_mode={q_dec_mode}")

    if k_dec_mode == "none":
        k_dec = None
    elif k_dec_mode == "q_enc":
        k_dec = q
    elif k_dec_mode == "rand":
        k_dec = k_dec_rand
    else:
        raise ValueError(f"Unknown k_dec_mode={k_dec_mode}")

    y_pre_py, s_py = flare_prefill_pytorch(Q=q, K=k, V=v, Q_dec=q_dec, K_dec=k_dec, scale=scale)
    y_pre_tr, s_tr = flare_prefill_triton(Q=q, K=k, V=v, Q_dec=q_dec, K_dec=k_dec, scale=scale)
    torch.testing.assert_close(y_pre_tr, y_pre_py, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(s_tr["m"], s_py["m"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(s_tr["d"], s_py["d"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(s_tr["u"], s_py["u"], rtol=1e-4, atol=1e-5)

    k_next = torch.randn((B, 1, H, d_score), device=device, dtype=dtype)
    v_next = torch.randn((B, 1, H, d_value), device=device, dtype=dtype)
    q_dec_next_rand = torch.randn((B, 1, H, d_score), device=device, dtype=dtype)
    if q_dec_mode == "none":
        q_dec_next = None
    elif q_dec_mode == "k_enc":
        q_dec_next = k_next
    else:
        q_dec_next = q_dec_next_rand

    y_dec_py, _ = flare_decode_pytorch(
        Q=q, K=k_next, V=v_next, state=s_py, Q_dec=q_dec_next, K_dec=k_dec, scale=scale
    )
    y_dec_tr, _ = flare_decode_triton(
        Q=q, K=k_next, V=v_next, state=s_tr, Q_dec=q_dec_next, K_dec=k_dec, scale=scale
    )
    torch.testing.assert_close(y_dec_tr, y_dec_py, rtol=1e-4, atol=1e-5)


def test_inference_decode_shape_validation_rejects_wrong_score_dims():
    torch.manual_seed(20260311)

    B = 1
    N = 4
    H = 2
    M = 16
    d_score = 64
    d_value = 128
    dtype = torch.float32

    q = torch.randn((H, M, d_score), device=device, dtype=dtype)
    k = torch.randn((B, N, H, d_score), device=device, dtype=dtype)
    v = torch.randn((B, N, H, d_value), device=device, dtype=dtype)
    _, state = flare_prefill_pytorch(Q=q, K=k, V=v, scale=1.0 / math.sqrt(d_score))

    bad_q_dec = torch.randn((B, 1, H, d_score + 16), device=device, dtype=dtype)
    bad_k_dec = torch.randn((H, M, d_score + 16), device=device, dtype=dtype)
    k_next = torch.randn((B, 1, H, d_score), device=device, dtype=dtype)
    v_next = torch.randn((B, 1, H, d_value), device=device, dtype=dtype)

    with pytest.raises(ValueError, match="Decode Q_dec"):
        flare_decode_pytorch(Q=q, K=k_next, V=v_next, state=state, Q_dec=bad_q_dec)
    with pytest.raises(ValueError, match="K_dec and Q_enc must have the same shape"):
        flare_prefill_pytorch(Q=q, K=k, V=v, K_dec=bad_k_dec)


def test_recurrent_flare_block_t_1_matches_block_t_16_forward():
    torch.manual_seed(777)

    B = 1
    N = 9
    H = 2
    M = 16
    D = 16
    scale = 1.0 / math.sqrt(D)
    dtype = torch.float32

    q = torch.randn((H, M, D), device=device, dtype=dtype)
    k = torch.randn((B, N, H, D), device=device, dtype=dtype)
    v = torch.randn((B, N, H, D), device=device, dtype=dtype)

    y_naive = flare_recurrent_triton(q, k, v, scale=scale, block_t=1)
    y_blocked = flare_recurrent_triton(q, k, v, scale=scale, block_t=16)
    torch.testing.assert_close(y_naive, y_blocked, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("block_t", [1, 16])
def test_recurrent_flare_dense_backward_strategy_matches_recurrent_impl_default_decode_aliases(block_t: int):
    torch.manual_seed(2026)

    B = 1
    N = 16
    H = 2
    M = 16
    D = 16
    scale = 1.0 / math.sqrt(D)
    dtype = torch.float32

    q_seed = torch.randn((H, M, D), device=device, dtype=dtype)
    k_seed = torch.randn((B, N, H, D), device=device, dtype=dtype)
    v_seed = torch.randn((B, N, H, D), device=device, dtype=dtype)
    grad_out = torch.randn((B, H, N, D), device=device, dtype=dtype)

    y_dense, dQ_dense, dK_dense, dV_dense, dQ_dec_dense, dK_dec_dense = flare_recurrent_dense_backward_pytorch(
        q_seed,
        k_seed,
        v_seed,
        grad_out,
        scale=scale,
        Q_dec=None,
        K_dec=None,
    )
    assert dQ_dec_dense is None
    assert dK_dec_dense is None

    y_impl, g_impl = _run_recurrent_impl_with_grads(
        "recurrent",
        q_seed,
        k_seed,
        v_seed,
        scale=scale,
        grad_out=grad_out,
        q_dec_mode="none",
        k_dec_mode="none",
        q_dec_rand_seed=torch.empty((B, N, H, D), device=device, dtype=dtype),
        k_dec_rand_seed=torch.empty((H, M, D), device=device, dtype=dtype),
        block_t=block_t,
    )

    torch.testing.assert_close(y_dense, y_impl, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(dQ_dense, g_impl["q"], rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(dK_dense, g_impl["k"], rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(dV_dense, g_impl["v"], rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("block_t", [1, 16])
def test_recurrent_flare_dense_backward_strategy_matches_recurrent_impl_nonshared_decode(block_t: int):
    torch.manual_seed(2027)

    B = 1
    N = 16
    H = 2
    M = 16
    D = 16
    scale = 1.0 / math.sqrt(D)
    dtype = torch.float32

    q_seed = torch.randn((H, M, D), device=device, dtype=dtype)
    k_seed = torch.randn((B, N, H, D), device=device, dtype=dtype)
    v_seed = torch.randn((B, N, H, D), device=device, dtype=dtype)
    q_dec_rand_seed = torch.randn((B, N, H, D), device=device, dtype=dtype)
    k_dec_rand_seed = torch.randn((H, M, D), device=device, dtype=dtype)
    grad_out = torch.randn((B, H, N, D), device=device, dtype=dtype)

    q_dec_modes = ("none", "k_enc", "rand")
    k_dec_modes = ("none", "q_enc", "rand")
    for q_dec_mode in q_dec_modes:
        for k_dec_mode in k_dec_modes:
            is_weight_sharing = q_dec_mode in ("none", "k_enc") and k_dec_mode in ("none", "q_enc")
            if is_weight_sharing:
                continue

            if q_dec_mode == "rand":
                q_dec = q_dec_rand_seed
            elif q_dec_mode == "k_enc":
                q_dec = k_seed
            else:
                q_dec = None

            if k_dec_mode == "rand":
                k_dec = k_dec_rand_seed
            elif k_dec_mode == "q_enc":
                k_dec = q_seed
            else:
                k_dec = None

            y_dense, dQ_dense, dK_dense, dV_dense, dQ_dec_dense, dK_dec_dense = flare_recurrent_dense_backward_pytorch(
                q_seed,
                k_seed,
                v_seed,
                grad_out,
                scale=scale,
                Q_dec=q_dec,
                K_dec=k_dec,
            )

            # Alias semantics: autograd accumulates decode grads into shared encode tensors.
            dQ_expected = dQ_dense.clone()
            dK_expected = dK_dense.clone()
            if q_dec_mode == "k_enc" and dQ_dec_dense is not None:
                dK_expected = dK_expected + dQ_dec_dense
            if k_dec_mode == "q_enc" and dK_dec_dense is not None:
                dQ_expected = dQ_expected + dK_dec_dense

            y_impl, g_impl = _run_recurrent_impl_with_grads(
                "recurrent",
                q_seed,
                k_seed,
                v_seed,
                scale=scale,
                grad_out=grad_out,
                q_dec_mode=q_dec_mode,
                k_dec_mode=k_dec_mode,
                q_dec_rand_seed=q_dec_rand_seed,
                k_dec_rand_seed=k_dec_rand_seed,
                block_t=block_t,
            )

            torch.testing.assert_close(y_dense, y_impl, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(dQ_expected, g_impl["q"], rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(dK_expected, g_impl["k"], rtol=1e-2, atol=1e-3)
            torch.testing.assert_close(dV_dense, g_impl["v"], rtol=1e-2, atol=1e-3)

            if q_dec_mode == "rand":
                torch.testing.assert_close(dQ_dec_dense, g_impl["q_dec_rand"], rtol=1e-2, atol=1e-3)
            if k_dec_mode == "rand":
                torch.testing.assert_close(dK_dec_dense, g_impl["k_dec_rand"], rtol=1e-2, atol=1e-3)

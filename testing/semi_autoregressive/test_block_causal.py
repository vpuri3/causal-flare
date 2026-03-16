import os

import pytest
import torch

from causal_flare.semi_autoregressive import (
    flare_semi_autoregressive_decode_trition,
    flare_semi_autoregressive_prefill_trition,
    flare_semi_autoregressive_trition,
)
from causal_flare.semi_autoregressive.reference import (
    _block_causal_forward_pytorch,
    semi_autoregressive_flare_reference,
)


@pytest.mark.parametrize(
    ("block_size", "chunk_size", "message"),
    [
        (24, 16, "multiple of 16"),
        (32, 48, "currently supports chunk_size"),
        (48, 32, "sequence length N to be an exact multiple of block_size"),
    ],
)
def test_block_causal_validation_rejects_invalid_block_chunk_pairs(block_size: int, chunk_size: int, message: str):
    q = torch.randn((2, 4, 16), dtype=torch.float32)
    k = torch.randn((1, 8, 2, 16), dtype=torch.float32)
    v = torch.randn((1, 8, 2, 16), dtype=torch.float32)

    with pytest.raises(ValueError, match=message):
        flare_semi_autoregressive_trition(q, k, v, block_size=block_size, chunk_size=chunk_size)


def test_block_causal_validation_rejects_non_block_aligned_sequence_length():
    q = torch.randn((2, 4, 16), dtype=torch.float32)
    k = torch.randn((1, 33, 2, 16), dtype=torch.float32)
    v = torch.randn((1, 33, 2, 16), dtype=torch.float32)

    with pytest.raises(ValueError, match="exact multiple of block_size"):
        flare_semi_autoregressive_trition(q, k, v, block_size=16, chunk_size=16)


@pytest.mark.parametrize(
    ("block_size", "chunk_size", "seq_len"),
    [
        (16, 16, 48),
        (32, 16, 64),
    ],
)
def test_block_causal_forward_matches_sdpa_reference(
    block_size: int,
    chunk_size: int,
    seq_len: int,
):
    torch.manual_seed(0)

    B = 2
    H = 3
    M = 5
    D = 16
    scale = D ** -0.5

    q = torch.randn((H, M, D), dtype=torch.float32)
    k = torch.randn((B, seq_len, H, D), dtype=torch.float32)
    v = torch.randn((B, seq_len, H, D), dtype=torch.float32)

    y_ref = semi_autoregressive_flare_reference(q, k, v, block_size=block_size, scale=scale)
    y_impl, aux = _block_causal_forward_pytorch(
        q, k, v, block_size=block_size, chunk_size=chunk_size, scale=scale, return_aux=True
    )

    assert aux["LSE_dec"].shape == (B, H, seq_len)
    assert aux["LSE_enc"].shape == (B, H, (seq_len + block_size - 1) // block_size, M)
    torch.testing.assert_close(y_impl, y_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton path")
def test_block_causal_training_wrapper_matches_reference_on_cuda():
    torch.manual_seed(2)

    B = 1
    N = 128
    H = 2
    M = 16
    D = 16
    block_size = 64
    chunk_size = 16
    scale = D ** -0.5

    q = torch.randn((H, M, D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)

    y, aux = flare_semi_autoregressive_trition(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
    )
    y_ref, _ = _block_causal_forward_pytorch(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        return_aux=True,
    )

    torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=2e-2)
    assert aux["LSE_dec"].shape == (B, H, N)
    assert y.dtype == v.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton path")
def test_block_causal_training_wrapper_fused_prefix_matches_reference_on_cuda(monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(9)

    B = 1
    N = 128
    H = 2
    M = 16
    D = 16
    block_size = 64
    chunk_size = 16
    scale = D ** -0.5

    q = torch.randn((H, M, D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)

    monkeypatch.setenv("FLARE_SEMI_AR_FUSED_PREFIX", "1")
    y, aux = flare_semi_autoregressive_trition(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
    )
    y_ref, aux_ref = _block_causal_forward_pytorch(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        return_aux=True,
    )

    torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(aux["LSE_enc"], aux_ref["LSE_enc"], rtol=2e-2, atol=2e-2)
    assert aux["LSE_dec"].shape == (B, H, N)
    assert y.dtype == v.dtype


def test_block_causal_prefill_wrapper_is_not_implemented():
    q = torch.randn((2, 4, 16), dtype=torch.float32)
    k = torch.randn((1, 32, 2, 16), dtype=torch.float32)
    v = torch.randn((1, 32, 2, 16), dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        flare_semi_autoregressive_prefill_trition(q, k, v)


def test_block_causal_decode_wrapper_is_not_implemented():
    q = torch.randn((2, 4, 16), dtype=torch.float32)
    k = torch.randn((1, 2, 16), dtype=torch.float32)
    v = torch.randn((1, 2, 16), dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        flare_semi_autoregressive_decode_trition(q, k, v)


def test_block_causal_reference_forward_matches_reference():
    torch.manual_seed(1)

    B = 2
    N = 32
    H = 2
    M = 4
    D = 16
    block_size = 32
    chunk_size = 16
    scale = D ** -0.5

    q_ref = torch.randn((H, M, D), dtype=torch.float32)
    k_ref = torch.randn((B, N, H, D), dtype=torch.float32)
    v_ref = torch.randn((B, N, H, D), dtype=torch.float32)

    y_ref = semi_autoregressive_flare_reference(q_ref, k_ref, v_ref, block_size=block_size, scale=scale)
    y_impl, _ = _block_causal_forward_pytorch(
        q_ref, k_ref, v_ref, block_size=block_size, chunk_size=chunk_size, scale=scale, return_aux=True
    )

    torch.testing.assert_close(y_impl, y_ref, rtol=1e-4, atol=1e-5)


def test_block_causal_forward_matches_reference_with_separate_decode_weights():
    torch.manual_seed(4)

    B = 2
    N = 32
    H = 2
    M = 4
    D = 16
    block_size = 32
    chunk_size = 16
    scale = D ** -0.5

    q = torch.randn((H, M, D), dtype=torch.float32)
    k = torch.randn((B, N, H, D), dtype=torch.float32)
    v = torch.randn((B, N, H, D), dtype=torch.float32)
    q_dec = torch.randn((B, N, H, D), dtype=torch.float32)
    k_dec = torch.randn((H, M, D), dtype=torch.float32)

    y_ref = semi_autoregressive_flare_reference(
        q,
        k,
        v,
        block_size=block_size,
        scale=scale,
        Q_dec=q_dec,
        K_dec=k_dec,
    )
    y_impl, aux = _block_causal_forward_pytorch(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        Q_dec=q_dec,
        K_dec=k_dec,
        return_aux=True,
    )

    assert aux["LSE_dec"].shape == (B, H, N)
    assert aux["LSE_enc"].shape == (B, H, 1, M)
    torch.testing.assert_close(y_impl, y_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton path")
def test_block_causal_training_wrapper_matches_reference_on_cuda_with_separate_decode_weights():
    torch.manual_seed(5)

    B = 1
    N = 128
    H = 2
    M = 16
    D = 16
    block_size = 64
    chunk_size = 16
    scale = D ** -0.5

    q = torch.randn((H, M, D), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)
    q_dec = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16)
    k_dec = torch.randn((H, M, D), device="cuda", dtype=torch.bfloat16)

    y, aux = flare_semi_autoregressive_trition(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        Q_dec=q_dec,
        K_dec=k_dec,
    )
    y_ref, _ = _block_causal_forward_pytorch(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        Q_dec=q_dec,
        K_dec=k_dec,
        return_aux=True,
    )

    torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=2e-2)
    assert aux["LSE_dec"].shape == (B, H, N)
    assert y.dtype == v.dtype


@pytest.mark.parametrize("head_dim", [128, 192])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton path")
def test_block_causal_training_wrapper_matches_reference_on_cuda_for_large_score_dims(head_dim: int):
    torch.manual_seed(8 + head_dim)

    B = 1
    N = 64
    H = 2
    M = 32
    block_size = 32
    chunk_size = 16
    scale = head_dim ** -0.5

    q = torch.randn((H, M, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((B, N, H, head_dim), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((B, N, H, head_dim), device="cuda", dtype=torch.bfloat16)

    y, aux = flare_semi_autoregressive_trition(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
    )
    y_ref, _ = _block_causal_forward_pytorch(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        return_aux=True,
    )

    torch.testing.assert_close(y, y_ref, rtol=3e-2, atol=3e-2)
    assert aux["LSE_dec"].shape == (B, H, N)
    assert y.dtype == v.dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton path")
def test_block_causal_training_wrapper_backward_matches_reference_on_cuda():
    torch.manual_seed(6)

    B = 1
    N = 64
    H = 2
    M = 16
    D = 16
    block_size = 32
    chunk_size = 16
    scale = D ** -0.5

    q = torch.randn((H, M, D), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16, requires_grad=True)

    y, _ = flare_semi_autoregressive_trition(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
    )
    dy = torch.randn_like(y)
    grads = torch.autograd.grad(y, (q, k, v), grad_outputs=dy)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    y_ref, _ = _block_causal_forward_pytorch(
        q_ref,
        k_ref,
        v_ref,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        return_aux=True,
    )
    grads_ref = torch.autograd.grad(y_ref, (q_ref, k_ref, v_ref), grad_outputs=dy)

    for grad, grad_ref in zip(grads, grads_ref):
        torch.testing.assert_close(grad, grad_ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA Triton path")
def test_block_causal_training_wrapper_backward_matches_reference_on_cuda_with_separate_decode_weights():
    torch.manual_seed(7)

    B = 1
    N = 64
    H = 2
    M = 16
    D = 16
    block_size = 32
    chunk_size = 16
    scale = D ** -0.5

    q = torch.randn((H, M, D), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    q_dec = torch.randn((B, N, H, D), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k_dec = torch.randn((H, M, D), device="cuda", dtype=torch.bfloat16, requires_grad=True)

    y, _ = flare_semi_autoregressive_trition(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        Q_dec=q_dec,
        K_dec=k_dec,
    )
    dy = torch.randn_like(y)
    grads = torch.autograd.grad(y, (q, k, v, q_dec, k_dec), grad_outputs=dy)

    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    q_dec_ref = q_dec.detach().clone().requires_grad_(True)
    k_dec_ref = k_dec.detach().clone().requires_grad_(True)
    y_ref, _ = _block_causal_forward_pytorch(
        q_ref,
        k_ref,
        v_ref,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        Q_dec=q_dec_ref,
        K_dec=k_dec_ref,
        return_aux=True,
    )
    grads_ref = torch.autograd.grad(y_ref, (q_ref, k_ref, v_ref, q_dec_ref, k_dec_ref), grad_outputs=dy)

    for grad, grad_ref in zip(grads, grads_ref):
        torch.testing.assert_close(grad, grad_ref, rtol=2e-2, atol=2e-2)

@pytest.mark.skipif(
    os.environ.get("FLARE_RUN_BLOCK_CAUSAL_PERF", "0") != "1" or not torch.cuda.is_available(),
    reason="set FLARE_RUN_BLOCK_CAUSAL_PERF=1 on CUDA to run the opt-in performance check",
)
def test_block_causal_perf_check_block256_chunk128():
    torch.manual_seed(3)

    batch_size = 8
    num_tokens = 65536
    num_heads = 16
    num_latents = 64
    head_dim = 64
    block_size = 256
    chunk_size = 128
    dtype = torch.bfloat16
    device = torch.device("cuda")

    q = torch.randn((num_heads, num_latents, head_dim), device=device, dtype=dtype)
    k = torch.randn((batch_size, num_tokens, num_heads, head_dim), device=device, dtype=dtype)
    v = torch.randn((batch_size, num_tokens, num_heads, head_dim), device=device, dtype=dtype)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    y, aux = _block_causal_forward_pytorch(q, k, v, block_size=block_size, chunk_size=chunk_size, return_aux=True)
    end.record()
    torch.cuda.synchronize()

    assert y.shape == (batch_size, num_tokens, num_heads, head_dim)
    assert torch.isfinite(y).all()
    assert aux["LSE_dec"].shape == (batch_size, num_heads, num_tokens)
    print(f"[block-causal perf] block={block_size} chunk={chunk_size} elapsed_ms={start.elapsed_time(end):.3f}")

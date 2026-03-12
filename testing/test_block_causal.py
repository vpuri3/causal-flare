import os

import pytest
import torch

from causal_flare.block_causal import (
    _block_causal_forward_torch,
    flare_block_causal_reference,
    flare_block_causal_torch,
    block_causal_sdpa_flex,
    block_causal_sdpa_reference,
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
        flare_block_causal_torch(q, k, v, block_size=block_size, chunk_size=chunk_size)


def test_block_causal_validation_rejects_non_block_aligned_sequence_length():
    q = torch.randn((2, 4, 16), dtype=torch.float32)
    k = torch.randn((1, 33, 2, 16), dtype=torch.float32)
    v = torch.randn((1, 33, 2, 16), dtype=torch.float32)

    with pytest.raises(ValueError, match="sequence length N to be an exact multiple of block_size"):
        flare_block_causal_torch(q, k, v, block_size=16, chunk_size=16)


@pytest.mark.parametrize(
    ("block_size", "chunk_size", "seq_len", "expected_phase1_mode"),
    [
        (16, 16, 48, "block_stats"),
        (32, 16, 64, "chunk_stats"),
    ],
)
def test_block_causal_forward_matches_sdpa_reference(
    block_size: int,
    chunk_size: int,
    seq_len: int,
    expected_phase1_mode: str,
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

    y_ref = flare_block_causal_reference(q, k, v, block_size=block_size, scale=scale)
    y_impl, aux = _block_causal_forward_torch(
        q,
        k,
        v,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        return_aux=True,
    )

    assert aux["phase1_mode"] == expected_phase1_mode
    assert aux["LSE_dec"].shape == (B, H, seq_len)
    assert aux["LSE_enc"].shape == (B, H, (seq_len + block_size - 1) // block_size, M)
    torch.testing.assert_close(y_impl, y_ref, rtol=1e-4, atol=1e-5)


def test_block_causal_autograd_matches_reference():
    torch.manual_seed(1)

    B = 2
    N = 32
    H = 2
    M = 4
    D = 16
    block_size = 32
    chunk_size = 16
    scale = D ** -0.5

    q_ref = torch.randn((H, M, D), dtype=torch.float32, requires_grad=True)
    k_ref = torch.randn((B, N, H, D), dtype=torch.float32, requires_grad=True)
    v_ref = torch.randn((B, N, H, D), dtype=torch.float32, requires_grad=True)
    q_impl = q_ref.detach().clone().requires_grad_(True)
    k_impl = k_ref.detach().clone().requires_grad_(True)
    v_impl = v_ref.detach().clone().requires_grad_(True)
    grad_out = torch.randn((B, N, H, D), dtype=torch.float32)

    y_ref = flare_block_causal_reference(q_ref, k_ref, v_ref, block_size=block_size, scale=scale)
    (y_ref * grad_out).sum().backward()

    y_impl = flare_block_causal_torch(q_impl, k_impl, v_impl, block_size=block_size, chunk_size=chunk_size, scale=scale)
    (y_impl * grad_out).sum().backward()

    torch.testing.assert_close(y_impl, y_ref.detach(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(q_impl.grad, q_ref.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(k_impl.grad, k_ref.grad, rtol=2e-4, atol=2e-5)
    torch.testing.assert_close(v_impl.grad, v_ref.grad, rtol=2e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention coverage requires CUDA")
def test_block_causal_sdpa_flex_matches_masked_sdpa():
    torch.manual_seed(2)

    B = 1
    N = 32
    H = 2
    D = 16
    block_size = 16
    scale = D ** -0.5
    device = torch.device("cuda")

    q = torch.randn((B, N, H, D), device=device, dtype=torch.float32)
    k = torch.randn((B, N, H, D), device=device, dtype=torch.float32)
    v = torch.randn((B, N, H, D), device=device, dtype=torch.float32)

    y_ref = block_causal_sdpa_reference(q, k, v, block_size=block_size, scale=scale)
    y_flex = block_causal_sdpa_flex(q, k, v, block_size=block_size, scale=scale)
    torch.testing.assert_close(y_flex, y_ref, rtol=1e-4, atol=1e-5)


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
    y = flare_block_causal_torch(q, k, v, block_size=block_size, chunk_size=chunk_size)
    end.record()
    torch.cuda.synchronize()

    assert y.shape == (batch_size, num_tokens, num_heads, head_dim)
    assert torch.isfinite(y).all()
    print(f"[block-causal perf] block={block_size} chunk={chunk_size} elapsed_ms={start.elapsed_time(end):.3f}")

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
    block_causal_sdpa_flex,
    block_causal_sdpa_reference,
    flare_block_causal_reference,
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

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        flare_semi_autoregressive_trition(q, k, v, block_size=block_size, chunk_size=chunk_size)


def test_block_causal_validation_rejects_non_block_aligned_sequence_length():
    q = torch.randn((2, 4, 16), dtype=torch.float32)
    k = torch.randn((1, 33, 2, 16), dtype=torch.float32)
    v = torch.randn((1, 33, 2, 16), dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        flare_semi_autoregressive_trition(q, k, v, block_size=16, chunk_size=16)


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
    y_impl, aux = _block_causal_forward_pytorch(
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


def test_block_causal_training_wrapper_is_not_implemented():
    q = torch.randn((2, 4, 16), dtype=torch.float32, requires_grad=True)
    k = torch.randn((1, 32, 2, 16), dtype=torch.float32, requires_grad=True)
    v = torch.randn((1, 32, 2, 16), dtype=torch.float32, requires_grad=True)

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        flare_semi_autoregressive_trition(q, k, v, block_size=32, chunk_size=16)


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

    y_ref = flare_block_causal_reference(q_ref, k_ref, v_ref, block_size=block_size, scale=scale)
    y_impl = _block_causal_forward_pytorch(q_ref, k_ref, v_ref, block_size=block_size, chunk_size=chunk_size, scale=scale)

    torch.testing.assert_close(y_impl, y_ref, rtol=1e-4, atol=1e-5)


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
    y = _block_causal_forward_pytorch(q, k, v, block_size=block_size, chunk_size=chunk_size)
    end.record()
    torch.cuda.synchronize()

    assert y.shape == (batch_size, num_tokens, num_heads, head_dim)
    assert torch.isfinite(y).all()
    print(f"[block-causal perf] block={block_size} chunk={chunk_size} elapsed_ms={start.elapsed_time(end):.3f}")

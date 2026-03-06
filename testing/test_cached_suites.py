import pytest
import torch

from causal_flare import flare_recurrent_triton
from causal_flare.inference import (
    flare_decode_pytorch,
    flare_decode_triton,
    flare_prefill_pytorch,
    flare_prefill_triton,
)
from causal_flare.torch import flare_causal_reference


pytestmark = [
    pytest.mark.regression,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def _check_finite(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), f"{name} contains NaN/Inf"


def test_cached_prefill_decode_regression_suite() -> None:
    torch.manual_seed(0)

    device = torch.device("cuda")
    dtype = torch.float32
    B, H, M, T, D = 2, 4, 32, 17, 32
    T_NEXT = 1
    mask_prob = 0.2
    scale = D ** -0.5

    atol = 1e-4
    rtol = 1e-4

    q = torch.randn(H, M, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    mask = (torch.rand(B, T, device=device) > mask_prob).to(torch.int32)

    # Prefill parity: PyTorch vs Triton
    y_py, s_py = flare_prefill_pytorch(q, k, v, scale=scale, attention_mask=mask)
    y_tr, s_tr = flare_prefill_triton(q, k, v, scale=scale, attention_mask=mask)
    torch.testing.assert_close(y_tr, y_py, atol=atol, rtol=rtol)

    # Decode parity: PyTorch vs Triton
    k_next = torch.randn(B, T_NEXT, H, D, device=device, dtype=dtype)
    v_next = torch.randn(B, T_NEXT, H, D, device=device, dtype=dtype)
    mask_next = (torch.rand(B, T_NEXT, device=device) > mask_prob).to(torch.int32)
    y_dec_py, _ = flare_decode_pytorch(q, k_next, v_next, s_py, scale=scale, attention_mask=mask_next)
    y_dec_tr, _ = flare_decode_triton(q, k_next, v_next, s_tr, scale=scale, attention_mask=mask_next)
    torch.testing.assert_close(y_dec_tr, y_dec_py, atol=atol, rtol=rtol)

    # Continuation consistency
    y_full_py, _ = flare_prefill_pytorch(
        q,
        torch.cat([k, k_next], dim=1),
        torch.cat([v, v_next], dim=1),
        scale=scale,
        attention_mask=torch.cat([mask, mask_next], dim=1),
    )
    torch.testing.assert_close(y_dec_py, y_full_py[:, -T_NEXT:], atol=atol, rtol=rtol)

    # Unmasked sanity: prefill/decode parity and agreement with reference/recurrent
    k_nomask = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v_nomask = torch.randn(B, T, H, D, device=device, dtype=dtype)
    yn_py, sn_py = flare_prefill_pytorch(q, k_nomask, v_nomask, scale=scale, attention_mask=None)
    yn_tr, sn_tr = flare_prefill_triton(q, k_nomask, v_nomask, scale=scale, attention_mask=None)
    torch.testing.assert_close(yn_tr, yn_py, atol=atol, rtol=rtol)

    yn_ref = flare_causal_reference(q, k_nomask, v_nomask, scale=scale)
    torch.testing.assert_close(yn_py, yn_ref, atol=atol, rtol=rtol)

    yn_rec = flare_recurrent_triton(q, k_nomask, v_nomask, scale=scale).permute(0, 2, 1, 3).contiguous()
    torch.testing.assert_close(yn_py, yn_rec, atol=atol, rtol=rtol)

    k_next_nomask = torch.randn(B, T_NEXT, H, D, device=device, dtype=dtype)
    v_next_nomask = torch.randn(B, T_NEXT, H, D, device=device, dtype=dtype)
    ydn_py, _ = flare_decode_pytorch(q, k_next_nomask, v_next_nomask, sn_py, scale=scale, attention_mask=None)
    ydn_tr, _ = flare_decode_triton(q, k_next_nomask, v_next_nomask, sn_tr, scale=scale, attention_mask=None)
    torch.testing.assert_close(ydn_tr, ydn_py, atol=atol, rtol=rtol)

    yn_full_ref = flare_causal_reference(
        q,
        torch.cat([k_nomask, k_next_nomask], dim=1),
        torch.cat([v_nomask, v_next_nomask], dim=1),
        scale=scale,
    )
    torch.testing.assert_close(ydn_py, yn_full_ref[:, -T_NEXT:], atol=atol, rtol=rtol)

    for name, tensor in (
        ("y_py", y_py),
        ("y_tr", y_tr),
        ("y_dec_py", y_dec_py),
        ("y_dec_tr", y_dec_tr),
        ("yn_py", yn_py),
        ("yn_tr", yn_tr),
        ("yn_ref", yn_ref),
        ("yn_rec", yn_rec),
        ("ydn_py", ydn_py),
        ("ydn_tr", ydn_tr),
    ):
        _check_finite(name, tensor)


def test_cached_flaredecoder_wiring_smoke() -> None:
    flare_decoder_mod = pytest.importorskip("fla.models.flare.flare_decoder")
    cache_mod = pytest.importorskip("fla.models.utils")
    FLAREDecoder = flare_decoder_mod.FLAREDecoder
    Cache = cache_mod.Cache

    torch.manual_seed(7)

    device = torch.device("cuda")
    dtype = torch.float32
    B, H, M, T, D = 1, 2, 16, 8, 16
    T_NEXT, T_CONT = 1, 2
    hidden_size = H * D

    mask = (torch.rand(B, T, device=device) > 0.2).to(torch.int32)
    mask_next = (torch.rand(B, T_NEXT, device=device) > 0.2).to(torch.int32)
    mask_cont = (torch.rand(B, T_CONT, device=device) > 0.2).to(torch.int32)

    decoder = FLAREDecoder(
        hidden_size=hidden_size,
        num_heads=H,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        layer_idx=0,
        num_latents=M,
        q_norm=False,
        k_norm=False,
        num_layers_k_proj=-1,
        num_layers_v_proj=-1,
    ).to(device=device, dtype=dtype)
    decoder.eval()

    x = torch.randn(B, T, hidden_size, device=device, dtype=dtype)
    x_next = torch.randn(B, T_NEXT, hidden_size, device=device, dtype=dtype)
    x_cont = torch.randn(B, T_CONT, hidden_size, device=device, dtype=dtype)

    with torch.no_grad():
        o0, _, _ = decoder(x, attention_mask=None, past_key_values=None, use_cache=False)

        cache = Cache()
        o1, _, cache = decoder(x, attention_mask=mask, past_key_values=cache, use_cache=True)
        o2, _, cache = decoder(x_next, attention_mask=mask_next, past_key_values=cache, use_cache=True)
        o3, _, cache = decoder(x_cont, attention_mask=mask_cont, past_key_values=cache, use_cache=True)

    assert tuple(o0.shape) == (B, T, hidden_size)
    assert tuple(o1.shape) == (B, T, hidden_size)
    assert tuple(o2.shape) == (B, T_NEXT, hidden_size)
    assert tuple(o3.shape) == (B, T_CONT, hidden_size)

    for name, tensor in (("o0", o0), ("o1", o1), ("o2", o2), ("o3", o3)):
        _check_finite(name, tensor)

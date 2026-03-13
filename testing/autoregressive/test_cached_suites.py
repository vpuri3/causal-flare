import pytest
import torch

from causal_flare import flare_recurrent_triton
from causal_flare.autoregressive.inference import (
    flare_decode_pytorch,
    flare_decode_triton,
    flare_prefill_pytorch,
    flare_prefill_triton,
)
from causal_flare.autoregressive.reference import flare_causal_reference


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

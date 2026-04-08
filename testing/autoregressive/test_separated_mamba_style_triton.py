import pytest
import torch

from causal_flare.autoregressive.separated import flare_autoregressive_separated_pytorch
from causal_flare.autoregressive.separated_mamba_style_triton import flare_autoregressive_separated_mamba_style_triton


@pytest.mark.parametrize(
    ("shape", "chunk_size", "rtol", "atol"),
    [
        ((2, 37, 3, 17, 48), 16, 5e-4, 5e-4),
        ((2, 257, 4, 64, 64), 256, 1e-3, 1e-3),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_separated_mamba_style_triton_forward_matches_pytorch_reference(shape, chunk_size, rtol, atol) -> None:
    pytest.importorskip("triton")
    torch.manual_seed(13)
    bsz, seqlen, nheads, nslots, value_dim = shape
    U = torch.randn((bsz, seqlen, nheads, value_dim), device="cuda", dtype=torch.float32)
    retain = (1e-3 + 0.999 * torch.rand((bsz, seqlen, nheads), device="cuda", dtype=torch.float32)).clamp_min_(1e-3)
    write = torch.randn((bsz, seqlen, nheads, nslots), device="cuda", dtype=torch.float32)
    decode = torch.randn((bsz, seqlen, nheads, nslots), device="cuda", dtype=torch.float32)

    with torch.no_grad():
        out = flare_autoregressive_separated_mamba_style_triton(
            U=U,
            retain=retain,
            write=write,
            decode_weights=decode,
            chunk_size=chunk_size,
        )
        ref = flare_autoregressive_separated_pytorch(
            U=U,
            retain=retain,
            write=write,
            decode_weights=decode,
            chunk_size=chunk_size,
        )
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_separated_mamba_style_triton_backward_matches_pytorch_reference_small() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(7)
    U = torch.randn((1, 8, 2, 16), device="cuda", dtype=torch.float32, requires_grad=True)
    retain = (1e-3 + 0.999 * torch.rand((1, 8, 2), device="cuda", dtype=torch.float32)).requires_grad_(True)
    write = torch.randn((1, 8, 2, 16), device="cuda", dtype=torch.float32, requires_grad=True)
    decode = torch.randn((1, 8, 2, 16), device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn((1, 8, 2, 16), device="cuda", dtype=torch.float32)

    out = flare_autoregressive_separated_mamba_style_triton(
        U=U,
        retain=retain,
        write=write,
        decode_weights=decode,
        chunk_size=16,
    )
    grads = torch.autograd.grad(out.mul(grad_out).sum(), (U, retain, write, decode))

    U_ref = U.detach().clone().requires_grad_(True)
    retain_ref = retain.detach().clone().requires_grad_(True)
    write_ref = write.detach().clone().requires_grad_(True)
    decode_ref = decode.detach().clone().requires_grad_(True)
    out_ref = flare_autoregressive_separated_pytorch(
        U=U_ref,
        retain=retain_ref,
        write=write_ref,
        decode_weights=decode_ref,
        chunk_size=16,
    )
    grads_ref = torch.autograd.grad(out_ref.mul(grad_out).sum(), (U_ref, retain_ref, write_ref, decode_ref))

    torch.testing.assert_close(out, out_ref, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(grads[0], grads_ref[0], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grads[1], grads_ref[1], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grads[2], grads_ref[2], rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grads[3], grads_ref[3], rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_separated_mamba_style_triton_backward_matches_pytorch_reference_chunk256() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(13)
    U = torch.randn((2, 257, 4, 64), device="cuda", dtype=torch.float32, requires_grad=True)
    retain = (1e-3 + 0.999 * torch.rand((2, 257, 4), device="cuda", dtype=torch.float32)).requires_grad_(True)
    write = torch.randn((2, 257, 4, 64), device="cuda", dtype=torch.float32, requires_grad=True)
    decode = torch.randn((2, 257, 4, 64), device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn((2, 257, 4, 64), device="cuda", dtype=torch.float32)

    out = flare_autoregressive_separated_mamba_style_triton(
        U=U,
        retain=retain,
        write=write,
        decode_weights=decode,
        chunk_size=256,
    )
    grads = torch.autograd.grad(out.mul(grad_out).sum(), (U, retain, write, decode))

    U_ref = U.detach().clone().requires_grad_(True)
    retain_ref = retain.detach().clone().requires_grad_(True)
    write_ref = write.detach().clone().requires_grad_(True)
    decode_ref = decode.detach().clone().requires_grad_(True)
    out_ref = flare_autoregressive_separated_pytorch(
        U=U_ref,
        retain=retain_ref,
        write=write_ref,
        decode_weights=decode_ref,
        chunk_size=256,
    )
    grads_ref = torch.autograd.grad(out_ref.mul(grad_out).sum(), (U_ref, retain_ref, write_ref, decode_ref))

    torch.testing.assert_close(out, out_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grads[0], grads_ref[0], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(grads[1], grads_ref[1], rtol=1e-2, atol=3e-1)
    torch.testing.assert_close(grads[2], grads_ref[2], rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(grads[3], grads_ref[3], rtol=1e-2, atol=1e-2)

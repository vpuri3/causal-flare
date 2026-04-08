import torch

from causal_flare.autoregressive.separated import flare_autoregressive_separated_pytorch
from causal_flare.autoregressive.separated_mamba_style import flare_autoregressive_separated_mamba_style_pytorch


@torch.no_grad()
def _make_inputs(
    *,
    shape: tuple[int, int, int, int, int],
    dtype: torch.dtype,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, seqlen, nheads, nslots, value_dim = shape
    U = torch.randn((bsz, seqlen, nheads, value_dim), device=device, dtype=dtype)
    retain = torch.rand((bsz, seqlen, nheads), device=device, dtype=dtype)
    write = torch.randn((bsz, seqlen, nheads, nslots), device=device, dtype=dtype)
    decode = torch.randn((bsz, seqlen, nheads, nslots), device=device, dtype=dtype)
    return U, retain, write, decode


def test_separated_mamba_style_matches_pytorch_reference_with_decode() -> None:
    torch.manual_seed(7)
    shape = (2, 37, 3, 17, 24)
    chunk_size = 16
    U_base, retain_base, write_base, decode_base = _make_inputs(shape=shape, dtype=torch.float64, device="cpu")
    grad_out = torch.randn((shape[0], shape[1], shape[2], shape[4]), dtype=torch.float64)

    U = U_base.detach().clone().requires_grad_(True)
    retain = retain_base.detach().clone().requires_grad_(True)
    write = write_base.detach().clone().requires_grad_(True)
    decode = decode_base.detach().clone().requires_grad_(True)
    out = flare_autoregressive_separated_mamba_style_pytorch(
        U=U,
        retain=retain,
        write=write,
        decode_weights=decode,
        chunk_size=chunk_size,
    )
    grads = torch.autograd.grad((out * grad_out).sum(), (U, retain, write, decode))

    U_ref = U_base.detach().clone().requires_grad_(True)
    retain_ref = retain_base.detach().clone().requires_grad_(True)
    write_ref = write_base.detach().clone().requires_grad_(True)
    decode_ref = decode_base.detach().clone().requires_grad_(True)
    ref = flare_autoregressive_separated_pytorch(
        U=U_ref,
        retain=retain_ref,
        write=write_ref,
        decode_weights=decode_ref,
        chunk_size=chunk_size,
    )
    ref_grads = torch.autograd.grad((ref * grad_out).sum(), (U_ref, retain_ref, write_ref, decode_ref))

    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=1e-9, atol=1e-9)


def test_separated_mamba_style_matches_pytorch_reference_without_decode() -> None:
    torch.manual_seed(9)
    shape = (2, 19, 2, 11, 8)
    chunk_size = 8
    U, retain, write, _ = _make_inputs(shape=shape, dtype=torch.float64, device="cpu")

    out = flare_autoregressive_separated_mamba_style_pytorch(
        U=U,
        retain=retain,
        write=write,
        decode_weights=None,
        chunk_size=chunk_size,
    )
    ref = flare_autoregressive_separated_pytorch(
        U=U,
        retain=retain,
        write=write,
        decode_weights=None,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)

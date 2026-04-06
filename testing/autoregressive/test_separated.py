import pytest
import torch

from causal_flare.autoregressive.separated import flare_autoregressive_separated_pytorch


def _sequential_separated_reference(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
) -> torch.Tensor:
    bsz, seqlen, nheads, value_dim = U.shape
    nslots = write.shape[3]
    state = torch.zeros((bsz, nheads, nslots, value_dim), device=U.device, dtype=U.dtype)
    outputs = []
    for offset in range(seqlen):
        write_value = write[:, offset].unsqueeze(-1) * U[:, offset].unsqueeze(-2)
        state = retain[:, offset].unsqueeze(-1).unsqueeze(-1) * state + write_value
        outputs.append((decode_weights[:, offset].unsqueeze(-1) * state).sum(dim=2))
    return torch.stack(outputs, dim=1)


def test_separated_chunked_recurrence_matches_sequential_reference() -> None:
    torch.manual_seed(7)
    bsz, seqlen, nheads, nslots, value_dim = (2, 5, 3, 4, 6)
    grad_out = torch.randn((bsz, seqlen, nheads, value_dim), dtype=torch.float64)
    U_base = torch.randn((bsz, seqlen, nheads, value_dim), dtype=torch.float64)
    retain_base = torch.rand((bsz, seqlen, nheads), dtype=torch.float64)
    write_base = torch.randn((bsz, seqlen, nheads, nslots), dtype=torch.float64)
    decode_base = torch.randn((bsz, seqlen, nheads, nslots), dtype=torch.float64)

    U = U_base.detach().clone().requires_grad_(True)
    retain = retain_base.detach().clone().requires_grad_(True)
    write = write_base.detach().clone().requires_grad_(True)
    decode_weights = decode_base.detach().clone().requires_grad_(True)
    out = flare_autoregressive_separated_pytorch(
        U=U,
        retain=retain,
        write=write,
        decode_weights=decode_weights,
        chunk_size=2,
        rmsnorm_read_contrib=False,
    )
    loss = (out * grad_out).sum()
    grads = torch.autograd.grad(loss, (U, retain, write, decode_weights))

    U_ref = U_base.detach().clone().requires_grad_(True)
    retain_ref = retain_base.detach().clone().requires_grad_(True)
    write_ref = write_base.detach().clone().requires_grad_(True)
    decode_ref = decode_base.detach().clone().requires_grad_(True)
    ref = _sequential_separated_reference(
        U=U_ref,
        retain=retain_ref,
        write=write_ref,
        decode_weights=decode_ref,
    )
    ref_loss = (ref * grad_out).sum()
    ref_grads = torch.autograd.grad(ref_loss, (U_ref, retain_ref, write_ref, decode_ref))

    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=1e-9, atol=1e-9)


def test_separated_chunked_recurrence_rmsnorm_read_contrib_not_implemented() -> None:
    U = torch.randn((2, 5, 3, 6), dtype=torch.float64)
    retain = torch.rand((2, 5, 3), dtype=torch.float64)
    write = torch.randn((2, 5, 3, 4), dtype=torch.float64)
    decode_weights = torch.randn((2, 5, 3, 4), dtype=torch.float64)

    with pytest.raises(NotImplementedError, match="rmsnorm_read_contrib=True"):
        flare_autoregressive_separated_pytorch(
            U=U,
            retain=retain,
            write=write,
            decode_weights=decode_weights,
            chunk_size=2,
            rmsnorm_read_contrib=True,
        )

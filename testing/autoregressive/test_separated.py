import pytest
import torch

from causal_flare.autoregressive.separated import (
    _rms_normalize_last_dim,
    chunkwise_affine_state_scan_slots,
    flare_autoregressive_separated_pytorch,
    parallel_history_slot_scan,
)


def _sequential_parallel_history_reference(
    *,
    U: torch.Tensor,
    slot_embed: torch.Tensor,
    decay: float | torch.Tensor | None = None,
    retain: torch.Tensor | None = None,
    write: torch.Tensor | None = None,
) -> torch.Tensor:
    bsz, seqlen, nheads, _ = U.shape
    nslots = slot_embed.shape[1]
    aux_dim = None if slot_embed.ndim == 3 else slot_embed.shape[2]
    if aux_dim is None:
        state = torch.zeros((bsz, nheads, nslots), device=U.device, dtype=U.dtype)
    else:
        state = torch.zeros((bsz, nheads, nslots, aux_dim), device=U.device, dtype=U.dtype)
    outputs = []

    if (retain is None) != (write is None):
        raise ValueError("retain and write must be provided together.")
    if retain is None:
        if decay is None:
            raise ValueError("Provide either decay or retain/write.")
        if torch.is_tensor(decay):
            decay_values = decay.to(device=U.device, dtype=U.dtype).view(1, nheads, 1)
        else:
            decay_values = torch.full((1, nheads, 1), float(decay), device=U.device, dtype=U.dtype)

    for offset in range(seqlen):
        outputs.append(state)
        if aux_dim is None:
            score = torch.einsum('b h d, h m d -> b h m', U[:, offset], slot_embed)
        else:
            score = torch.einsum('b h d, h m a d -> b h m a', U[:, offset], slot_embed)
        if retain is None:
            if aux_dim is None:
                state = decay_values * state + (1.0 - decay_values) * score
            else:
                state = decay_values.unsqueeze(-1) * state + (1.0 - decay_values).unsqueeze(-1) * score
        else:
            if aux_dim is None:
                state = retain[:, offset] * state + write[:, offset] * score
            else:
                state = retain[:, offset].unsqueeze(-1) * state + write[:, offset].unsqueeze(-1) * score

    return torch.stack(outputs, dim=1)


def _sequential_slot_scan_reference(
    *,
    A: torch.Tensor,
    B: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    state = initial_state
    starts = []
    for offset in range(A.shape[1]):
        starts.append(state)
        state = A[:, offset].unsqueeze(-1) * state + B[:, offset]
    return torch.stack(starts, dim=1), state


def _sequential_separated_reference(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
    rmsnorm_read_contrib: bool,
) -> torch.Tensor:
    bsz, seqlen, nheads, value_dim = U.shape
    nslots = retain.shape[3]
    state = torch.zeros((bsz, nheads, nslots, value_dim), device=U.device, dtype=U.dtype)
    outputs = []
    eps = torch.finfo(U.dtype).eps

    for offset in range(seqlen):
        write_value = write[:, offset].unsqueeze(-1) * U[:, offset].unsqueeze(-2)
        state = retain[:, offset].unsqueeze(-1) * state + write_value
        contrib = decode_weights[:, offset].unsqueeze(-1) * state
        if rmsnorm_read_contrib:
            contrib = _rms_normalize_last_dim(contrib, eps=eps, scale_by_sqrt_dim=False)
        outputs.append(contrib.sum(dim=2))

    return torch.stack(outputs, dim=1)


def test_chunkwise_affine_state_scan_slots_matches_sequential_reference() -> None:
    torch.manual_seed(3)
    A_base = torch.rand((4, 5, 6), dtype=torch.float64)
    B_base = torch.randn((4, 5, 6, 3), dtype=torch.float64)
    initial_base = torch.randn((4, 6, 3), dtype=torch.float64)
    grad_chunk = torch.randn((4, 5, 6, 3), dtype=torch.float64)
    grad_final = torch.randn((4, 6, 3), dtype=torch.float64)

    A = A_base.detach().clone().requires_grad_(True)
    B = B_base.detach().clone().requires_grad_(True)
    initial = initial_base.detach().clone().requires_grad_(True)
    chunk_start, final_state = chunkwise_affine_state_scan_slots(A, B, initial)
    loss = (chunk_start * grad_chunk).sum() + (final_state * grad_final).sum()
    grads = torch.autograd.grad(loss, (A, B, initial))

    A_ref = A_base.detach().clone().requires_grad_(True)
    B_ref = B_base.detach().clone().requires_grad_(True)
    initial_ref = initial_base.detach().clone().requires_grad_(True)
    chunk_start_ref, final_state_ref = _sequential_slot_scan_reference(A=A_ref, B=B_ref, initial_state=initial_ref)
    ref_loss = (chunk_start_ref * grad_chunk).sum() + (final_state_ref * grad_final).sum()
    ref_grads = torch.autograd.grad(ref_loss, (A_ref, B_ref, initial_ref))

    torch.testing.assert_close(chunk_start, chunk_start_ref, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(final_state, final_state_ref, rtol=1e-9, atol=1e-9)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=1e-9, atol=1e-9)


def test_parallel_history_slot_scan_matches_sequential_reference() -> None:
    torch.manual_seed(5)
    U = torch.randn((2, 6, 3, 4), dtype=torch.float64, requires_grad=True)
    slot_embed = torch.randn((3, 5, 4), dtype=torch.float64, requires_grad=True)
    grad_out = torch.randn((2, 6, 3, 5), dtype=torch.float64)
    decay = 0.9

    out = parallel_history_slot_scan(U, slot_embed, decay=decay)
    loss = (out * grad_out).sum()
    grads = torch.autograd.grad(loss, (U, slot_embed))

    U_ref = U.detach().clone().requires_grad_(True)
    slot_embed_ref = slot_embed.detach().clone().requires_grad_(True)
    ref = _sequential_parallel_history_reference(U=U_ref, slot_embed=slot_embed_ref, decay=decay)
    ref_loss = (ref * grad_out).sum()
    ref_grads = torch.autograd.grad(ref_loss, (U_ref, slot_embed_ref))

    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=1e-9, atol=1e-9)


def test_parallel_history_slot_scan_accepts_per_head_decay() -> None:
    torch.manual_seed(6)
    U = torch.randn((2, 4, 3, 5), dtype=torch.float64, requires_grad=True)
    slot_embed = torch.randn((3, 7, 5), dtype=torch.float64, requires_grad=True)
    decay = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64, requires_grad=True)
    grad_out = torch.randn((2, 4, 3, 7), dtype=torch.float64)

    out = parallel_history_slot_scan(U, slot_embed, decay=decay)
    loss = (out * grad_out).sum()
    grads = torch.autograd.grad(loss, (U, slot_embed, decay))

    U_ref = U.detach().clone().requires_grad_(True)
    slot_embed_ref = slot_embed.detach().clone().requires_grad_(True)
    decay_ref = decay.detach().clone().requires_grad_(True)
    ref = _sequential_parallel_history_reference(U=U_ref, slot_embed=slot_embed_ref, decay=decay_ref)
    ref_loss = (ref * grad_out).sum()
    ref_grads = torch.autograd.grad(ref_loss, (U_ref, slot_embed_ref, decay_ref))

    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=1e-9, atol=1e-9)


def test_parallel_history_slot_scan_accepts_data_dependent_retain_write() -> None:
    torch.manual_seed(9)
    U = torch.randn((2, 5, 3, 4), dtype=torch.float64, requires_grad=True)
    slot_embed = torch.randn((3, 6, 4), dtype=torch.float64, requires_grad=True)
    retain = torch.sigmoid(torch.randn((2, 5, 3, 6), dtype=torch.float64, requires_grad=True))
    write = torch.randn((2, 5, 3, 6), dtype=torch.float64, requires_grad=True)
    grad_out = torch.randn((2, 5, 3, 6), dtype=torch.float64)

    out = parallel_history_slot_scan(U, slot_embed, retain=retain, write=write)
    loss = (out * grad_out).sum()
    grads = torch.autograd.grad(loss, (U, slot_embed, retain, write))

    U_ref = U.detach().clone().requires_grad_(True)
    slot_embed_ref = slot_embed.detach().clone().requires_grad_(True)
    retain_ref = retain.detach().clone().requires_grad_(True)
    write_ref = write.detach().clone().requires_grad_(True)
    ref = _sequential_parallel_history_reference(
        U=U_ref,
        slot_embed=slot_embed_ref,
        retain=retain_ref,
        write=write_ref,
    )
    ref_loss = (ref * grad_out).sum()
    ref_grads = torch.autograd.grad(ref_loss, (U_ref, slot_embed_ref, retain_ref, write_ref))

    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=1e-9, atol=1e-9)


def test_parallel_history_slot_scan_respects_explicit_chunk_size() -> None:
    torch.manual_seed(10)
    U = torch.randn((2, 7, 3, 4), dtype=torch.float64, requires_grad=True)
    slot_embed = torch.randn((3, 6, 4), dtype=torch.float64, requires_grad=True)
    retain = torch.sigmoid(torch.randn((2, 7, 3, 6), dtype=torch.float64, requires_grad=True))
    write = torch.randn((2, 7, 3, 6), dtype=torch.float64, requires_grad=True)
    grad_out = torch.randn((2, 7, 3, 6), dtype=torch.float64)

    out = parallel_history_slot_scan(U, slot_embed, retain=retain, write=write, chunk_size=3)
    loss = (out * grad_out).sum()
    grads = torch.autograd.grad(loss, (U, slot_embed, retain, write))

    U_ref = U.detach().clone().requires_grad_(True)
    slot_embed_ref = slot_embed.detach().clone().requires_grad_(True)
    retain_ref = retain.detach().clone().requires_grad_(True)
    write_ref = write.detach().clone().requires_grad_(True)
    ref = _sequential_parallel_history_reference(
        U=U_ref,
        slot_embed=slot_embed_ref,
        retain=retain_ref,
        write=write_ref,
    )
    ref_loss = (ref * grad_out).sum()
    ref_grads = torch.autograd.grad(ref_loss, (U_ref, slot_embed_ref, retain_ref, write_ref))

    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=1e-9, atol=1e-9)


def test_parallel_history_slot_scan_accepts_feature_valued_state() -> None:
    torch.manual_seed(11)
    U = torch.randn((2, 4, 3, 5), dtype=torch.float64, requires_grad=True)
    slot_embed = torch.randn((3, 6, 4, 5), dtype=torch.float64, requires_grad=True)
    retain = torch.sigmoid(torch.randn((2, 4, 3, 6), dtype=torch.float64, requires_grad=True))
    write = torch.randn((2, 4, 3, 6), dtype=torch.float64, requires_grad=True)
    grad_out = torch.randn((2, 4, 3, 6, 4), dtype=torch.float64)

    out = parallel_history_slot_scan(U, slot_embed, retain=retain, write=write)
    loss = (out * grad_out).sum()
    grads = torch.autograd.grad(loss, (U, slot_embed, retain, write))

    U_ref = U.detach().clone().requires_grad_(True)
    slot_embed_ref = slot_embed.detach().clone().requires_grad_(True)
    retain_ref = retain.detach().clone().requires_grad_(True)
    write_ref = write.detach().clone().requires_grad_(True)
    ref = _sequential_parallel_history_reference(
        U=U_ref,
        slot_embed=slot_embed_ref,
        retain=retain_ref,
        write=write_ref,
    )
    ref_loss = (ref * grad_out).sum()
    ref_grads = torch.autograd.grad(ref_loss, (U_ref, slot_embed_ref, retain_ref, write_ref))

    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("rmsnorm_read_contrib", [False, True])
def test_separated_chunked_recurrence_matches_sequential_reference(rmsnorm_read_contrib: bool) -> None:
    torch.manual_seed(7)
    shape = (2, 5, 3, 4, 6)
    bsz, seqlen, nheads, nslots, value_dim = shape
    grad_out = torch.randn((bsz, seqlen, nheads, value_dim), dtype=torch.float64)
    U_base = torch.randn((bsz, seqlen, nheads, value_dim), dtype=torch.float64)
    retain_base = torch.rand((bsz, seqlen, nheads, nslots), dtype=torch.float64)
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
        rmsnorm_read_contrib=rmsnorm_read_contrib,
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
        rmsnorm_read_contrib=rmsnorm_read_contrib,
    )
    ref_loss = (ref * grad_out).sum()
    ref_grads = torch.autograd.grad(ref_loss, (U_ref, retain_ref, write_ref, decode_ref))

    torch.testing.assert_close(out, ref, rtol=1e-9, atol=1e-9)
    for grad, ref_grad in zip(grads, ref_grads):
        torch.testing.assert_close(grad, ref_grad, rtol=1e-9, atol=1e-9)

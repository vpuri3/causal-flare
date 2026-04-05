import pytest
import torch

from causal_flare.autoregressive.separated import (
    _main_block_forward_impl,
    flare_autoregressive_separated_pytorch,
    parallel_history_slot_scan,
)
from causal_flare.autoregressive.separated_trition import (
    flare_autoregressive_separated_trition,
    parallel_history_slot_scan_trition,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_separated_trition_forward_matches_pytorch(chunk_size: int) -> None:
    torch.manual_seed(0)
    B, N, H, M, D = 2, chunk_size * 4, 4, 32, 64
    U = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    retain = torch.sigmoid(torch.randn(B, N, H, M, device="cuda", dtype=torch.bfloat16))
    write = torch.randn(B, N, H, M, device="cuda", dtype=torch.bfloat16) * 0.1
    decode = torch.randn(B, N, H, M, device="cuda", dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        ref = flare_autoregressive_separated_pytorch(
            U=U,
            retain=retain,
            write=write,
            decode_weights=decode,
            chunk_size=chunk_size,
            rmsnorm_read_contrib=False,
        )
        out, final_state = flare_autoregressive_separated_trition(
            U=U,
            retain=retain,
            write=write,
            decode_weights=decode,
            chunk_size=chunk_size,
            return_final_state=True,
        )

    rel_l2 = ((out - ref).float().norm() / (ref.float().norm() + 1e-12)).item()
    max_abs = (out - ref).abs().max().item()
    assert out.shape == ref.shape
    assert final_state.shape == (B, H, M, D)
    assert rel_l2 < 0.03, rel_l2
    assert max_abs < 0.03, max_abs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_separated_trition_backward_matches_pytorch(chunk_size: int) -> None:
    torch.manual_seed(0)
    B, N, H, M, D = 1, chunk_size * 2, 2, 32, 64

    def make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        U = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        retain = torch.sigmoid(torch.randn(B, N, H, M, device="cuda", dtype=torch.bfloat16))
        write = torch.randn(B, N, H, M, device="cuda", dtype=torch.bfloat16) * 0.1
        decode = torch.randn(B, N, H, M, device="cuda", dtype=torch.bfloat16) * 0.1
        return U, retain, write, decode

    U_ref, retain_ref, write_ref, decode_ref = [x.detach().clone().requires_grad_(True) for x in make_inputs()]
    U_tri = U_ref.detach().clone().requires_grad_(True)
    retain_tri = retain_ref.detach().clone().requires_grad_(True)
    write_tri = write_ref.detach().clone().requires_grad_(True)
    decode_tri = decode_ref.detach().clone().requires_grad_(True)

    ref_out, ref_final_state, _ = _main_block_forward_impl(
        state0=torch.zeros((B * H, M, D), device="cuda", dtype=U_ref.dtype),
        retain_block=retain_ref.view(B, 2, chunk_size, H, M).permute(0, 3, 1, 2, 4).reshape(B * H, 2, chunk_size, M),
        write_block=write_ref.view(B, 2, chunk_size, H, M).permute(0, 3, 1, 2, 4).reshape(B * H, 2, chunk_size, M),
        U_block=U_ref.view(B, 2, chunk_size, H, D).permute(0, 3, 1, 2, 4).reshape(B * H, 2, chunk_size, D),
        decode_block=decode_ref.view(B, 2, chunk_size, H, M).permute(0, 3, 1, 2, 4).reshape(B * H, 2, chunk_size, M),
        rmsnorm_read_contrib=False,
    )
    ref = ref_out.reshape(B, H, 2, chunk_size, D).permute(0, 2, 3, 1, 4).reshape(B, N, H, D)
    ref_final_state = ref_final_state.reshape(B, H, M, D)
    out, final_state = flare_autoregressive_separated_trition(
        U=U_tri,
        retain=retain_tri,
        write=write_tri,
        decode_weights=decode_tri,
        chunk_size=chunk_size,
        return_final_state=True,
    )
    tri_loss = out.float().square().mean() + final_state.float().square().mean() * 1e-3
    grad_out, grad_final = torch.autograd.grad(tri_loss, (out, final_state), retain_graph=True)
    ref_grads = torch.autograd.grad(
        outputs=(ref, ref_final_state),
        inputs=(U_ref, retain_ref, write_ref, decode_ref),
        grad_outputs=(grad_out, grad_final),
    )
    tri_loss.backward()

    def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
        return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()

    grad_u_ref, grad_retain_ref, grad_write_ref, grad_decode_ref = ref_grads

    assert rel_l2(U_tri.grad, grad_u_ref) < 0.03
    assert rel_l2(retain_tri.grad, grad_retain_ref) < 0.06
    assert rel_l2(write_tri.grad, grad_write_ref) < 0.03
    assert rel_l2(decode_tri.grad, grad_decode_ref) < 0.03


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_parallel_history_slot_scan_trition_forward_matches_pytorch(chunk_size: int) -> None:
    torch.manual_seed(0)
    B, N, H, M, A, D = 2, chunk_size * 4, 4, 32, 4, 64
    U = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
    slot_embed = torch.randn(H, M, A, D, device="cuda", dtype=torch.bfloat16) * 0.1
    retain = torch.sigmoid(torch.randn(B, N, H, M, device="cuda", dtype=torch.bfloat16))
    write = torch.randn(B, N, H, M, device="cuda", dtype=torch.bfloat16) * 0.1
    readout = torch.randn(H, M, A, device="cuda", dtype=torch.bfloat16) * 0.1

    with torch.no_grad():
        ref = parallel_history_slot_scan(
            U=U,
            slot_embed=slot_embed,
            retain=retain,
            write=write,
            readout=readout,
            chunk_size=chunk_size,
        )
        out = parallel_history_slot_scan_trition(
            U=U,
            slot_embed=slot_embed,
            retain=retain,
            write=write,
            readout=readout,
            chunk_size=chunk_size,
        )

    rel_l2 = ((out - ref).float().norm() / (ref.float().norm() + 1e-12)).item()
    max_abs = (out - ref).abs().max().item()
    assert out.shape == ref.shape
    assert rel_l2 < 0.03, rel_l2
    assert max_abs < 0.03, max_abs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_parallel_history_slot_scan_trition_backward_matches_pytorch(chunk_size: int) -> None:
    torch.manual_seed(0)
    B, N, H, M, A, D = 1, chunk_size * 2, 2, 32, 4, 64

    def make_inputs():
        U = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
        slot_embed = torch.randn(H, M, A, D, device="cuda", dtype=torch.bfloat16) * 0.1
        retain = torch.sigmoid(torch.randn(B, N, H, M, device="cuda", dtype=torch.bfloat16))
        write = torch.randn(B, N, H, M, device="cuda", dtype=torch.bfloat16) * 0.1
        readout = torch.randn(H, M, A, device="cuda", dtype=torch.bfloat16) * 0.1
        return U, slot_embed, retain, write, readout

    U_ref, slot_ref, retain_ref, write_ref, readout_ref = [x.detach().clone().requires_grad_(True) for x in make_inputs()]
    U_tri = U_ref.detach().clone().requires_grad_(True)
    slot_tri = slot_ref.detach().clone().requires_grad_(True)
    retain_tri = retain_ref.detach().clone().requires_grad_(True)
    write_tri = write_ref.detach().clone().requires_grad_(True)
    readout_tri = readout_ref.detach().clone().requires_grad_(True)

    ref = parallel_history_slot_scan(
        U=U_ref,
        slot_embed=slot_ref,
        retain=retain_ref,
        write=write_ref,
        readout=readout_ref,
        chunk_size=chunk_size,
    )
    ref_loss = ref.float().square().mean()
    ref_loss.backward()

    tri = parallel_history_slot_scan_trition(
        U=U_tri,
        slot_embed=slot_tri,
        retain=retain_tri,
        write=write_tri,
        readout=readout_tri,
        chunk_size=chunk_size,
    )
    tri_loss = tri.float().square().mean()
    tri_loss.backward()

    def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
        return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-12)).item()

    assert rel_l2(U_tri.grad, U_ref.grad) < 0.05
    assert rel_l2(slot_tri.grad, slot_ref.grad) < 0.05
    assert rel_l2(retain_tri.grad, retain_ref.grad) < 0.08
    assert rel_l2(write_tri.grad, write_ref.grad) < 0.05
    assert rel_l2(readout_tri.grad, readout_ref.grad) < 0.05

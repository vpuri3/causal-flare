import pytest
import torch

from causal_flare.autoregressive._mamba_vendor.separated_adapters import separated_chunk_scan, separated_chunk_state
from causal_flare.autoregressive.separated_mamba_style import separated_chunk_scan_ref, separated_chunk_state_ref
from causal_flare.autoregressive.separated_mamba_style_triton import _prepare_inputs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_separated_chunk_state_wrapper_matches_stage_reference() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(7)
    bsz, seqlen, nheads, nslots, value_dim = 1, 16, 2, 8, 8
    U = torch.randn((bsz, seqlen, nheads, value_dim), device="cuda", dtype=torch.float32)
    retain = 1e-3 + 0.999 * torch.rand((bsz, seqlen, nheads), device="cuda", dtype=torch.float32)
    write = torch.randn((bsz, seqlen, nheads, nslots), device="cuda", dtype=torch.float32)
    decode = torch.randn((bsz, seqlen, nheads, nslots), device="cuda", dtype=torch.float32)

    prepared = _prepare_inputs(U=U, retain=retain, write=write, decode_weights=decode, chunk_size=16)
    U_seq = prepared["U_seq"]
    write_seq = prepared["write_seq"]
    dt = prepared["dt"]
    dA_cumsum = prepared["dA_cumsum"]

    B = write_seq.detach().clone().requires_grad_(True)
    x = U_seq.detach().clone().requires_grad_(True)
    dt_leaf = dt.detach().clone().requires_grad_(True)
    dA_leaf = dA_cumsum.detach().clone().requires_grad_(True)

    out = separated_chunk_state(B, x, dt_leaf, dA_leaf, states_in_fp32=True)
    grad_out = torch.randn_like(out)
    grads = torch.autograd.grad(out, (B, x, dt_leaf, dA_leaf), grad_out)

    U_chunk = U.view(bsz, 1, seqlen, nheads, value_dim).permute(0, 1, 3, 2, 4).contiguous().requires_grad_(True)
    write_chunk = (
        write.view(bsz, 1, seqlen, nheads, nslots).permute(0, 1, 3, 2, 4).contiguous().requires_grad_(True)
    )
    log_cumsum = (
        torch.cumsum(
            retain.view(bsz, 1, seqlen, nheads).permute(0, 1, 3, 2).contiguous().clamp_min(1e-3).log(),
            dim=-1,
        )
        .detach()
        .clone()
        .requires_grad_(True)
    )
    out_ref = separated_chunk_state_ref(
        write_chunk=write_chunk,
        U_chunk=U_chunk,
        log_cumsum=log_cumsum,
    ).permute(0, 1, 2, 4, 3).reshape_as(out)
    grads_ref = torch.autograd.grad(out_ref, (write_chunk, U_chunk, log_cumsum), grad_out)
    dB_ref = grads_ref[0].permute(0, 1, 3, 2, 4).reshape_as(B)
    dx_ref = grads_ref[1].permute(0, 1, 3, 2, 4).reshape_as(x)
    ddA_ref = grads_ref[2].transpose(1, 2)

    torch.testing.assert_close(out, out_ref, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(grads[0], dB_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grads[1], dx_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grads[3], ddA_ref, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_separated_chunk_scan_wrapper_matches_stage_reference() -> None:
    pytest.importorskip("triton")
    torch.manual_seed(11)
    bsz, seqlen, nheads, nslots, value_dim = 1, 16, 2, 8, 8
    U = torch.randn((bsz, seqlen, nheads, value_dim), device="cuda", dtype=torch.float32)
    retain = 1e-3 + 0.999 * torch.rand((bsz, seqlen, nheads), device="cuda", dtype=torch.float32)
    write = torch.randn((bsz, seqlen, nheads, nslots), device="cuda", dtype=torch.float32)
    decode = torch.randn((bsz, seqlen, nheads, nslots), device="cuda", dtype=torch.float32)

    prepared = _prepare_inputs(U=U, retain=retain, write=write, decode_weights=decode, chunk_size=16)
    B = prepared["write_seq"].detach().clone().requires_grad_(True)
    C = prepared["decode_seq"].detach().clone().requires_grad_(True)
    x = prepared["U_seq"].detach().clone().requires_grad_(True)
    dt_leaf = prepared["dt"].detach().clone().requires_grad_(True)
    dA_leaf = prepared["dA_cumsum"].detach().clone().requires_grad_(True)
    prev_states = torch.randn(
        (bsz, prepared["nchunks"], nheads, value_dim, nslots),
        device="cuda",
        dtype=torch.float32,
    ).requires_grad_(True)

    out = separated_chunk_scan(B, C, x, dt_leaf, dA_leaf, prev_states)
    grad_out = torch.randn_like(out)
    grad_out[:, seqlen:] = 0
    grads = torch.autograd.grad(out.mul(grad_out).sum(), (B, C, x, dA_leaf, prev_states))

    U_chunk = U.view(bsz, 1, seqlen, nheads, value_dim).permute(0, 1, 3, 2, 4).contiguous().requires_grad_(True)
    write_chunk = (
        write.view(bsz, 1, seqlen, nheads, nslots).permute(0, 1, 3, 2, 4).contiguous().requires_grad_(True)
    )
    decode_chunk = (
        decode.view(bsz, 1, seqlen, nheads, nslots).permute(0, 1, 3, 2, 4).contiguous().requires_grad_(True)
    )
    log_cumsum = (
        torch.cumsum(
            retain.view(bsz, 1, seqlen, nheads).permute(0, 1, 3, 2).contiguous().clamp_min(1e-3).log(),
            dim=-1,
        )
        .detach()
        .clone()
        .requires_grad_(True)
    )
    prev_states_ref = prev_states.detach().clone().transpose(-1, -2).requires_grad_(True)
    out_ref = separated_chunk_scan_ref(
        write_chunk=write_chunk,
        decode_chunk=decode_chunk,
        U_chunk=U_chunk,
        log_cumsum=log_cumsum,
        prev_states=prev_states_ref,
    ).permute(0, 1, 3, 2, 4).reshape_as(out)
    grads_ref = torch.autograd.grad(out_ref.mul(grad_out).sum(), (write_chunk, decode_chunk, U_chunk, log_cumsum, prev_states_ref))

    B_ref = grads_ref[0].permute(0, 1, 3, 2, 4).reshape_as(B)
    C_ref = grads_ref[1].permute(0, 1, 3, 2, 4).reshape_as(C)
    x_ref = grads_ref[2].permute(0, 1, 3, 2, 4).reshape_as(x)
    dA_ref = grads_ref[3].transpose(1, 2)
    prev_ref = grads_ref[4].transpose(-1, -2)

    torch.testing.assert_close(out, out_ref, rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(grads[0], B_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grads[1], C_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grads[2], x_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grads[3], dA_ref, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(grads[4], prev_ref, rtol=2e-3, atol=2e-3)

from __future__ import annotations

"""Separated-FLARE specific wrappers over vendored upstream Mamba SSD helpers.

These wrappers keep the upstream kernel structure, but adjust backward
contracts where the separated-FLARE ``dA_cumsum`` semantics differ from the
standalone vendored wrapper semantics.
"""

import torch

from .chunk_scan import (
    _chunk_scan_bwd_dC,
    _chunk_scan_bwd_dcb,
    _chunk_scan_bwd_dstates,
    _chunk_scan_bwd_ddAcs_unstable,
    _chunk_scan_bwd_dx,
    _chunk_scan_fwd,
)
from .chunk_state import _chunk_state_bwd_db, _chunk_state_bwd_dx, _chunk_state_fwd
from .ssd_bmm import _bmm_chunk_bwd, _bmm_chunk_fwd


def _chunk_prev_term(C: torch.Tensor, dA_cumsum: torch.Tensor, prev_states: torch.Tensor) -> torch.Tensor:
    """Compute the chunk-start-state contribution to chunk-scan output.

    Shapes:
    - C: [B, S, H, M]
    - dA_cumsum: [B, H, NC, C]
    - prev_states: [B, NC, H, D, M]

    Returns:
    - [B, S, H, D]
    """
    batch, seqlen, nheads, dstate = C.shape
    nchunks = dA_cumsum.shape[2]
    chunk_size = dA_cumsum.shape[3]
    headdim = prev_states.shape[3]
    C_chunk = C.reshape(batch, nchunks, chunk_size, nheads, dstate).permute(0, 1, 3, 2, 4).contiguous()
    scale = dA_cumsum.exp().permute(0, 2, 1, 3).contiguous()
    prev_term = torch.einsum("bchsm,bchdm->bchsd", C_chunk, prev_states)
    prev_term = prev_term * scale.unsqueeze(-1)
    return prev_term.permute(0, 1, 3, 2, 4).reshape(batch, seqlen, nheads, headdim).contiguous()


class SeparatedChunkStateFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, x, dt, dA_cumsum, states_in_fp32=True):
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen <= nchunks * chunk_size
        _, _, ngroups, dstate = B.shape
        assert B.shape == (batch, seqlen, ngroups, dstate)
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        if B.stride(-1) != 1:
            B = B.contiguous()
        if x.stride(-1) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        states = _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=states_in_fp32)
        ctx.save_for_backward(B, x, dt, dA_cumsum)
        return states

    @staticmethod
    def backward(ctx, dstates):
        B, x, dt, dA_cumsum = ctx.saved_tensors
        _, _, ngroups, _ = B.shape
        if dstates.stride(-1) != 1:
            dstates = dstates.contiguous()
        need_ddt = ctx.needs_input_grad[2]
        dx, ddt, ddA_from_dx = _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates, return_ddt=need_ddt)
        dB, ddA_from_db = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, B=B, ngroups=ngroups)
        if dB.dtype != B.dtype:
            dB = dB.to(B.dtype)
        # For separated FLARE, dA_cumsum is an independent cumulative log-retain
        # input. The x-side kernel returns the missing local contribution in a
        # pre-cumsum form; the standalone vendored wrapper drops it because the
        # upstream combined backward accounts for it elsewhere.
        ddA_cumsum = ddA_from_db + torch.cumsum(ddA_from_dx, dim=-1)
        if ddA_cumsum.dtype != dA_cumsum.dtype:
            ddA_cumsum = ddA_cumsum.to(dA_cumsum.dtype)
        return dB, dx, ddt, ddA_cumsum, None


def separated_chunk_state(B, x, dt, dA_cumsum, states_in_fp32=True):
    return SeparatedChunkStateFn.apply(B, x, dt, dA_cumsum, states_in_fp32)


class SeparatedChunkScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, C, x, dt, dA_cumsum, prev_states):
        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, dstate = B.shape
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen == nchunks * chunk_size
        assert C.shape == B.shape
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if x.stride(-1) != 1 and x.stride(1) != 1:
            x = x.contiguous()
        CB = _bmm_chunk_fwd(C, B, chunk_size)
        out, _, prev_term = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, prev_states, return_prev_term=True)
        # Reuse the auxiliary buffer so backward can consume the local-only term
        # directly instead of rebuilding the previous-state contribution eagerly.
        out_local = prev_term.mul(-1).add_(out)
        ctx.save_for_backward(B, C, CB, x, dt, dA_cumsum, prev_states, out_local)
        return out

    @staticmethod
    def backward(ctx, dout):
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        B, C, CB, x, dt, dA_cumsum, prev_states, out_local = ctx.saved_tensors
        _, _, ngroups, _ = B.shape
        dprev_states = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, dtype=prev_states.dtype)
        dC, ddA_prev = _chunk_scan_bwd_dC(prev_states, dA_cumsum, dout, C=C, ngroups=ngroups)
        if dC.dtype != C.dtype:
            dC = dC.to(C.dtype)
        dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, ngroups=ngroups)
        if dCB.dtype != CB.dtype:
            dCB = dCB.to(CB.dtype)
        dB = _bmm_chunk_bwd(C, dCB)
        dC = _bmm_chunk_bwd(B, dCB.transpose(-1, -2), residual=dC)
        dx, ddt = _chunk_scan_bwd_dx(CB, x, dt, dA_cumsum, dout, D=None)

        ddA_local, _ = _chunk_scan_bwd_ddAcs_unstable(
            x,
            dt,
            out_local,
            dout,
            ddt,
            D=None,
            subtract_ddtdt=True,
        )
        if ddA_local.dtype != dA_cumsum.dtype:
            ddA_local = ddA_local.to(dA_cumsum.dtype)
        if ddA_prev.dtype != dA_cumsum.dtype:
            ddA_prev = ddA_prev.to(dA_cumsum.dtype)
        ddA_cumsum = ddA_local + ddA_prev
        return dB, dC, dx, ddt, ddA_cumsum, dprev_states


def separated_chunk_scan(B, C, x, dt, dA_cumsum, prev_states):
    return SeparatedChunkScanFn.apply(B, C, x, dt, dA_cumsum, prev_states)


__all__ = [
    "separated_chunk_state",
    "SeparatedChunkStateFn",
    "separated_chunk_scan",
    "SeparatedChunkScanFn",
]

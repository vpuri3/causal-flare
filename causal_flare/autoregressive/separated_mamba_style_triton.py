from __future__ import annotations

"""Separated-FLARE adapter over vendored upstream Mamba SSD Triton wrappers.

This path intentionally adopts the upstream SSD parameterization in the
positive-retain regime:

- ``dt`` is fixed to ``1``
- ``A_t`` is represented as ``log(clamp(retain_t, min=1e-3))``
- ``B`` maps to separated ``write``
- ``C`` maps to separated ``decode``
- ``x`` maps to separated ``U``

With that mapping, the public API can call the vendored upstream Triton
forward wrappers directly. The only eager PyTorch glue in this file is the
boundary adaptation:

- chunk padding / reshaping
- clamping ``retain`` before taking ``log``
- ``decode_weights=None`` is still unsupported here

Backward uses separated-specific vendored adapter wrappers that preserve the
upstream kernel structure but adjust the ``dA_cumsum`` contracts to match the
separated-FLARE recurrence.
"""

import torch

from causal_flare.autoregressive._mamba_vendor.separated_adapters import separated_chunk_scan, separated_chunk_state
from causal_flare.autoregressive._mamba_vendor.state_passing import state_passing
from causal_flare.autoregressive.separated_mamba_style import _reshape_separated_inputs_to_chunks

_MIN_RETAIN = 1e-3


def _chunk_to_seq_5d(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 1, 3, 2, 4).reshape(x.shape[0], x.shape[1] * x.shape[3], x.shape[2], x.shape[4]).contiguous()


def _prepare_inputs(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor | None,
    chunk_size: int | None,
) -> dict[str, torch.Tensor | int]:
    (
        U_chunk,
        retain_chunk,
        write_chunk,
        decode_chunk,
        bsz,
        seqlen,
        nheads,
        value_dim,
        nslots,
        _chunk_size_resolved,
        _padded_len,
    ) = _reshape_separated_inputs_to_chunks(
        U=U,
        retain=retain,
        write=write,
        decode_weights=decode_weights,
        chunk_size=chunk_size,
    )
    if decode_chunk is None:
        raise NotImplementedError(
            "flare_autoregressive_separated_mamba_style_triton does not support decode_weights=None yet."
        )
    retain_pos_chunk = retain_chunk.clamp_min(_MIN_RETAIN)
    dA_cumsum = torch.cumsum(retain_pos_chunk.log(), dim=-1).transpose(1, 2).contiguous()
    # This path fixes dt=1 everywhere. Keep it as a broadcasted scalar view so
    # we don't allocate and write a full dense tensor just to feed constant ones
    # into the vendored kernels.
    dt = dA_cumsum.new_ones((1, 1, 1, 1)).expand_as(dA_cumsum)
    U_seq = _chunk_to_seq_5d(U_chunk)
    write_seq = _chunk_to_seq_5d(write_chunk)
    decode_seq = _chunk_to_seq_5d(decode_chunk)
    return {
        "U_seq": U_seq,
        "write_seq": write_seq,
        "decode_seq": decode_seq,
        "dA_cumsum": dA_cumsum,
        "dt": dt,
        "bsz": bsz,
        "seqlen": seqlen,
        "nheads": nheads,
        "value_dim": value_dim,
        "nslots": nslots,
        "nchunks": U_chunk.shape[1],
    }


def _flare_autoregressive_separated_mamba_style_triton_impl(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
    chunk_size: int | None,
) -> torch.Tensor:
    prepared = _prepare_inputs(
        U=U,
        retain=retain,
        write=write,
        decode_weights=decode_weights,
        chunk_size=chunk_size,
    )
    seqlen = prepared["seqlen"]
    if seqlen == 0:
        return U.new_empty((U.shape[0], 0, U.shape[2], U.shape[3]))

    U_seq = prepared["U_seq"]
    write_seq = prepared["write_seq"]
    decode_seq = prepared["decode_seq"]
    dt = prepared["dt"]
    dA_cumsum = prepared["dA_cumsum"]
    bsz = prepared["bsz"]
    nheads = prepared["nheads"]
    value_dim = prepared["value_dim"]
    nslots = prepared["nslots"]
    nchunks = prepared["nchunks"]

    chunk_states = separated_chunk_state(write_seq, U_seq, dt, dA_cumsum, states_in_fp32=True)
    prev_states_flat, _ = state_passing(
        chunk_states.reshape(bsz, nchunks, nheads, value_dim * nslots).contiguous(),
        dA_cumsum[..., -1],
        initial_states=U_seq.new_zeros((bsz, nheads, value_dim * nslots)),
    )
    prev_states = prev_states_flat.reshape(bsz, nchunks, nheads, value_dim, nslots).contiguous()
    out_padded = separated_chunk_scan(write_seq, decode_seq, U_seq, dt, dA_cumsum, prev_states)
    out = out_padded[:, :seqlen]
    if out.dtype != U.dtype:
        out = out.to(U.dtype)
    return out if out.is_contiguous() else out.contiguous()


def flare_autoregressive_separated_mamba_style_hybrid(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    return _flare_autoregressive_separated_mamba_style_triton_impl(
        U=U,
        retain=retain,
        write=write,
        decode_weights=decode_weights,
        chunk_size=chunk_size,
    )


def flare_autoregressive_separated_mamba_style_triton(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    return _flare_autoregressive_separated_mamba_style_triton_impl(
        U=U,
        retain=retain,
        write=write,
        decode_weights=decode_weights,
        chunk_size=chunk_size,
    )


__all__ = [
    "flare_autoregressive_separated_mamba_style_hybrid",
    "flare_autoregressive_separated_mamba_style_triton",
]

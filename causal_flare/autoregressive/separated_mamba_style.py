from __future__ import annotations

import math

import torch

from causal_flare.autoregressive.separated import _pad_sequence_dim, _resolve_separated_chunk_size


def _reshape_separated_inputs_to_chunks(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor | None,
    chunk_size: int | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    if U.ndim != 4:
        raise ValueError(f"U must be [B,N,H,D]. Got {tuple(U.shape)}.")
    if retain.ndim != 3:
        raise ValueError(f"retain must be [B,N,H]. Got {tuple(retain.shape)}.")
    if write.ndim != 4:
        raise ValueError(f"write must be [B,N,H,M]. Got {tuple(write.shape)}.")

    bsz, seqlen, nheads, value_dim = U.shape
    if retain.shape != (bsz, seqlen, nheads):
        raise ValueError("retain must share [B,N,H] with U.")
    if write.shape[:3] != (bsz, seqlen, nheads):
        raise ValueError("write must share [B,N,H] with U.")
    nslots = write.shape[-1]
    if decode_weights is not None and decode_weights.shape != (bsz, seqlen, nheads, nslots):
        raise ValueError("decode_weights must match [B,N,H,M].")

    if seqlen == 0:
        return (
            U.new_empty((bsz, 0, nheads, 0, value_dim)),
            retain.new_empty((bsz, 0, nheads, 0)),
            write.new_empty((bsz, 0, nheads, 0, nslots)),
            None if decode_weights is None else decode_weights.new_empty((bsz, 0, nheads, 0, nslots)),
            bsz,
            seqlen,
            nheads,
            value_dim,
            nslots,
            0,
            0,
        )

    accum_dtype = torch.float32 if U.dtype in {torch.float16, torch.bfloat16} else U.dtype
    U_acc = U.to(accum_dtype)
    retain_acc = retain.to(accum_dtype)
    write_acc = write.to(accum_dtype)
    decode_acc = None if decode_weights is None else decode_weights.to(accum_dtype)

    chunk_size = _resolve_separated_chunk_size(seqlen, nslots, value_dim, chunk_size)
    nchunks = math.ceil(seqlen / chunk_size)
    padded_len = nchunks * chunk_size
    pad = padded_len - seqlen
    if pad > 0:
        U_acc = _pad_sequence_dim(U_acc, pad, fill_value=0.0)
        retain_acc = _pad_sequence_dim(retain_acc, pad, fill_value=1.0)
        write_acc = _pad_sequence_dim(write_acc, pad, fill_value=0.0)
        if decode_acc is not None:
            decode_acc = _pad_sequence_dim(decode_acc, pad, fill_value=0.0)

    U_chunk = U_acc.view(bsz, nchunks, chunk_size, nheads, value_dim).permute(0, 1, 3, 2, 4).contiguous()
    retain_chunk = retain_acc.view(bsz, nchunks, chunk_size, nheads).permute(0, 1, 3, 2).contiguous()
    write_chunk = write_acc.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 1, 3, 2, 4).contiguous()
    decode_chunk = None
    if decode_acc is not None:
        decode_chunk = decode_acc.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 1, 3, 2, 4).contiguous()

    return U_chunk, retain_chunk, write_chunk, decode_chunk, bsz, seqlen, nheads, value_dim, nslots, chunk_size, padded_len


def separated_chunk_state_ref(
    *,
    write_chunk: torch.Tensor,
    U_chunk: torch.Tensor,
    log_cumsum: torch.Tensor,
) -> torch.Tensor:
    """Separated-FLARE analogue of Mamba's ``chunk_state_ref``.

    Mamba names:
    - ``B`` -> ``write``
    - ``x`` -> ``U``
    - ``dA_cumsum`` -> ``log_cumsum``

    Returns the zero-initial-state chunk summary, i.e. the local state at the
    end of each chunk.
    """

    suffix_exclusive = torch.exp(log_cumsum[..., -1:] - log_cumsum)
    return torch.einsum("bnhcm,bnhc,bnhcd->bnhmd", write_chunk, suffix_exclusive, U_chunk)


def separated_state_passing_ref(
    *,
    chunk_states: torch.Tensor,
    chunk_log_decay: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separated-FLARE analogue of Mamba's ``state_passing_ref``."""

    if initial_state is None:
        initial_state = torch.zeros_like(chunk_states[:, 0])
    states = torch.cat([initial_state.unsqueeze(1), chunk_states], dim=1)
    chunk_log_decay = torch.nn.functional.pad(chunk_log_decay.permute(0, 2, 1), (1, 0))
    chunk_log_decay = torch.cumsum(chunk_log_decay, dim=-1)
    nchunks_plus_one = chunk_log_decay.shape[-1]
    segment_sum = chunk_log_decay[:, :, :, None] - chunk_log_decay[:, :, None, :]
    decay = torch.exp(segment_sum)
    causal_mask = torch.tril(torch.ones(nchunks_plus_one, nchunks_plus_one, device=states.device, dtype=torch.bool))
    decay = decay.masked_fill(~causal_mask, 0.0)
    out = torch.einsum("bhzc,bchmd->bzhmd", decay.to(dtype=states.dtype), states)
    return out[:, :-1], out[:, -1]


def separated_chunk_scan_ref(
    *,
    write_chunk: torch.Tensor,
    decode_chunk: torch.Tensor,
    U_chunk: torch.Tensor,
    log_cumsum: torch.Tensor,
    prev_states: torch.Tensor,
) -> torch.Tensor:
    """Separated-FLARE analogue of Mamba's ``chunk_scan_ref``."""

    chunk_size = U_chunk.shape[-2]
    cb = torch.einsum("bnhlm,bnhsm->bnhls", decode_chunk, write_chunk)
    segment_sum = log_cumsum[..., :, None] - log_cumsum[..., None, :]
    decay = torch.exp(segment_sum)
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=U_chunk.device, dtype=torch.bool))
    scores_decay = (cb * decay).masked_fill(~causal_mask, 0.0)
    out_local = torch.einsum("bnhls,bnhsd->bnhld", scores_decay.to(dtype=U_chunk.dtype), U_chunk)
    out_prev = torch.einsum("bnhlm,bnhmd->bnhld", decode_chunk, prev_states) * torch.exp(log_cumsum).unsqueeze(-1)
    return out_local + out_prev


def flare_autoregressive_separated_mamba_style_pytorch(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor | None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Mamba-structured PyTorch decomposition of separated FLARE.

    This keeps the separated recurrence but follows the same reference staging
    Mamba uses in PyTorch:
    1. chunk-local state summaries
    2. inter-chunk state passing
    3. in-chunk scan/readout from local interactions plus previous state
    """

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
        chunk_size_resolved,
        padded_len,
    ) = _reshape_separated_inputs_to_chunks(
        U=U,
        retain=retain,
        write=write,
        decode_weights=decode_weights,
        chunk_size=chunk_size,
    )
    if seqlen == 0:
        if decode_weights is None:
            return U.new_empty((bsz, 0, nheads, nslots, value_dim))
        return U.new_empty((bsz, 0, nheads, value_dim))

    tiny = torch.finfo(retain_chunk.dtype).tiny
    log_cumsum = torch.cumsum(retain_chunk.clamp_min(tiny).log(), dim=-1)

    chunk_states = separated_chunk_state_ref(
        write_chunk=write_chunk,
        U_chunk=U_chunk,
        log_cumsum=log_cumsum,
    )
    prev_states, final_state = separated_state_passing_ref(
        chunk_states=chunk_states,
        chunk_log_decay=log_cumsum[..., -1],
        initial_state=U_chunk.new_zeros((bsz, nheads, nslots, value_dim)),
    )

    if decode_chunk is None:
        causal_mask = torch.tril(
            torch.ones((chunk_size_resolved, chunk_size_resolved), device=U.device, dtype=torch.bool)
        )
        decay = torch.exp(log_cumsum[..., :, None] - log_cumsum[..., None, :]).masked_fill(~causal_mask, 0.0)
        token_states = (
            torch.einsum(
                "bnhsm,bnhts,bnhsd->bnhtmd",
                write_chunk,
                decay,
                U_chunk,
            )
            + prev_states.unsqueeze(-3) * torch.exp(log_cumsum).unsqueeze(-1).unsqueeze(-1)
        )
        out = token_states.permute(0, 1, 3, 2, 4, 5).reshape(bsz, padded_len, nheads, nslots, value_dim)
        return out[:, :seqlen].to(U.dtype).contiguous()

    out_chunk = separated_chunk_scan_ref(
        write_chunk=write_chunk,
        decode_chunk=decode_chunk,
        U_chunk=U_chunk,
        log_cumsum=log_cumsum,
        prev_states=prev_states,
    )
    out = out_chunk.permute(0, 1, 3, 2, 4).reshape(bsz, padded_len, nheads, value_dim)
    return out[:, :seqlen].to(U.dtype).contiguous()


__all__ = [
    "separated_chunk_state_ref",
    "separated_state_passing_ref",
    "separated_chunk_scan_ref",
    "flare_autoregressive_separated_mamba_style_pytorch",
]

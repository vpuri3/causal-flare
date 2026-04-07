import math
import os

import torch


def _resolve_separated_chunk_size(N: int, M: int, D: int, chunk_size: int | None) -> int:
    if chunk_size is None:
        env_chunk = os.environ.get("FLARE_SEPARATED_CHUNK_SIZE", "")
        chunk_size = int(env_chunk) if env_chunk else None
    if chunk_size is not None:
        return int(chunk_size)
    if D <= 32 and M <= 64 and N >= 1024:
        return 128
    if D <= 64 and M <= 128 and N >= 1024:
        return 128
    return max(64, min(256, max(1, N // 2)))


def _rms_normalize_last_dim(x: torch.Tensor, *, eps: float, scale_by_sqrt_dim: bool = False) -> torch.Tensor:
    rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    y = x / rms
    if scale_by_sqrt_dim:
        y = y / math.sqrt(x.shape[-1])
    return y


def _affine_prefix_scan_flat(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scan_A = A
    scan_B = B
    length = A.size(1)
    step = 1
    while step < length:
        shifted_A = torch.cat([torch.ones_like(scan_A[:, :step]), scan_A[:, :-step]], dim=1)
        shifted_B = torch.cat([torch.zeros_like(scan_B[:, :step]), scan_B[:, :-step]], dim=1)
        scan_B = scan_B + scan_A * shifted_B
        scan_A = scan_A * shifted_A
        step <<= 1
    return scan_A, scan_B


def _chunkwise_affine_state_scan(
    A: torch.Tensor,
    B: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if A.ndim != 3 or B.ndim != 3 or initial_state.ndim != 2:
        raise ValueError(
            "_chunkwise_affine_state_scan expects A=[P,NC,S], B=[P,NC,S], initial_state=[P,S]. "
            f"Got A={tuple(A.shape)}, B={tuple(B.shape)}, initial_state={tuple(initial_state.shape)}."
        )
    if A.shape != B.shape:
        raise ValueError(f"A and B must have identical shapes. Got A={tuple(A.shape)}, B={tuple(B.shape)}.")
    if A.shape[0] != initial_state.shape[0] or A.shape[2] != initial_state.shape[1]:
        raise ValueError(
            "initial_state must match the flattened batch/state dimensions of A/B. "
            f"Got A={tuple(A.shape)}, initial_state={tuple(initial_state.shape)}."
        )

    if A.shape[1] == 0:
        return A.new_empty((A.shape[0], 0, A.shape[2])), initial_state

    inc_A, inc_B = _affine_prefix_scan_flat(A, B)
    excl_A = torch.cat([torch.ones_like(inc_A[:, :1]), inc_A[:, :-1]], dim=1)
    excl_B = torch.cat([torch.zeros_like(inc_B[:, :1]), inc_B[:, :-1]], dim=1)
    chunk_start = excl_A * initial_state.unsqueeze(1) + excl_B
    final_state = inc_A[:, -1] * initial_state + inc_B[:, -1]
    return chunk_start, final_state


def _segment_log_sums(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 1:
        raise ValueError(f"x must have a time dimension. Got {tuple(x.shape)}.")
    t = x.size(-1)
    expanded = x.unsqueeze(-1).expand(*x.shape, t)
    lower_exclusive = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=-1)
    expanded = expanded.masked_fill(~lower_exclusive, 0)
    expanded = torch.cumsum(expanded, dim=-2)
    lower_inclusive = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=0)
    return expanded.masked_fill(~lower_inclusive, -torch.inf)


def _pad_sequence_dim(x: torch.Tensor, pad: int, *, fill_value: float) -> torch.Tensor:
    if pad <= 0:
        return x
    fill = torch.full((x.shape[0], pad, *x.shape[2:]), fill_value, device=x.device, dtype=x.dtype)
    return torch.cat([x, fill], dim=1)


def _main_ssd_forward_chunked(
    *,
    initial_state: torch.Tensor,
    U_chunk: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_chunk: torch.Tensor,
    decode_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if initial_state.ndim != 3:
        raise ValueError(f"initial_state must be [P,M,D]. Got {tuple(initial_state.shape)}.")
    if U_chunk.ndim != 4:
        raise ValueError(f"U_chunk must be [P,NC,C,D]. Got {tuple(U_chunk.shape)}.")
    if retain_chunk.ndim != 3:
        raise ValueError(f"retain_chunk must be [P,NC,C]. Got {tuple(retain_chunk.shape)}.")
    if write_chunk.ndim != 4:
        raise ValueError(f"write_chunk must be [P,NC,C,M]. Got {tuple(write_chunk.shape)}.")
    if decode_chunk.ndim != 4:
        raise ValueError(f"decode_chunk must be [P,NC,C,M]. Got {tuple(decode_chunk.shape)}.")

    flat_batch, nchunks, chunk_size, value_dim = U_chunk.shape
    nslots = write_chunk.shape[-1]
    if initial_state.shape != (flat_batch, nslots, value_dim):
        raise ValueError(
            "initial_state must match [P,M,D] implied by write_chunk/U_chunk. "
            f"Got initial_state={tuple(initial_state.shape)}, expected={(flat_batch, nslots, value_dim)}."
        )

    tiny = torch.finfo(retain_chunk.dtype).tiny
    log_retain = retain_chunk.clamp_min(tiny).log()
    a_cumsum = torch.cumsum(log_retain, dim=-1)
    decay_intra = torch.exp(_segment_log_sums(log_retain))
    chunk_A = torch.exp(a_cumsum[..., -1])
    suffix_exclusive = torch.exp(a_cumsum[..., -1:] - a_cumsum)

    chunk_B = torch.einsum(
        "pcsm,pcs,pcsd->pcmd",
        write_chunk,
        suffix_exclusive,
        U_chunk,
    )
    start_states, final_state = _chunkwise_affine_state_scan(
        chunk_A.unsqueeze(-1).expand(-1, -1, nslots * value_dim),
        chunk_B.reshape(flat_batch, nchunks, nslots * value_dim),
        initial_state.reshape(flat_batch, nslots * value_dim),
    )
    start_states = start_states.reshape(flat_batch, nchunks, nslots, value_dim)

    # Stage 1: diagonal in-chunk contributions from writes inside the current chunk.
    y_diag = torch.einsum(
        "pctm,pcsm,pcts,pcsd->pctd",
        decode_chunk,
        write_chunk,
        decay_intra,
        U_chunk,
    )
    # Stage 2: off-diagonal chunk-prefix contribution from the state entering each chunk.
    y_off = torch.einsum(
        "pctm,pcmd,pct->pctd",
        decode_chunk,
        start_states,
        torch.exp(a_cumsum),
    )
    return y_diag + y_off, final_state


def flare_autoregressive_separated_pytorch(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    if U.ndim != 4:
        raise ValueError(f"U must be [B,N,H,D]. Got {tuple(U.shape)}.")
    if retain.ndim != 3:
        raise ValueError(f"retain must be [B,N,H]. Got {tuple(retain.shape)}.")
    if write.ndim != 4:
        raise ValueError(f"write must be [B,N,H,M]. Got {tuple(write.shape)}.")
    if decode_weights.ndim != 4:
        raise ValueError(f"decode_weights must be [B,N,H,M]. Got {tuple(decode_weights.shape)}.")

    bsz, seqlen, nheads, value_dim = U.shape
    if retain.shape != (bsz, seqlen, nheads):
        raise ValueError("retain must share [B,N,H] with U.")
    if write.shape[:3] != (bsz, seqlen, nheads):
        raise ValueError("write must share [B,N,H] with U.")
    nslots = write.shape[3]
    if decode_weights.shape != (bsz, seqlen, nheads, nslots):
        raise ValueError("decode_weights must match [B,N,H,M].")
    if seqlen == 0:
        return U.new_empty((bsz, 0, nheads, value_dim))

    accum_dtype = torch.float32 if U.dtype in {torch.float16, torch.bfloat16} else U.dtype
    U_acc = U.to(accum_dtype)
    retain_acc = retain.to(accum_dtype)
    write_acc = write.to(accum_dtype)
    decode_acc = decode_weights.to(accum_dtype)

    chunk_size = _resolve_separated_chunk_size(seqlen, nslots, value_dim, chunk_size)
    nchunks = math.ceil(seqlen / chunk_size)
    padded_len = nchunks * chunk_size
    pad = padded_len - seqlen
    if pad > 0:
        U_acc = _pad_sequence_dim(U_acc, pad, fill_value=0.0)
        retain_acc = _pad_sequence_dim(retain_acc, pad, fill_value=1.0)
        write_acc = _pad_sequence_dim(write_acc, pad, fill_value=0.0)
        decode_acc = _pad_sequence_dim(decode_acc, pad, fill_value=0.0)

    flat_batch = bsz * nheads
    U_chunk = U_acc.view(bsz, nchunks, chunk_size, nheads, value_dim).permute(0, 3, 1, 2, 4).reshape(
        flat_batch, nchunks, chunk_size, value_dim
    )
    retain_chunk = retain_acc.view(bsz, nchunks, chunk_size, nheads).permute(0, 3, 1, 2).reshape(
        flat_batch, nchunks, chunk_size
    )
    write_chunk = write_acc.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 3, 1, 2, 4).reshape(
        flat_batch, nchunks, chunk_size, nslots
    )
    decode_chunk = decode_acc.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 3, 1, 2, 4).reshape(
        flat_batch, nchunks, chunk_size, nslots
    )

    out_chunk, _ = _main_ssd_forward_chunked(
        initial_state=U_chunk.new_zeros((flat_batch, nslots, value_dim)),
        U_chunk=U_chunk,
        retain_chunk=retain_chunk,
        write_chunk=write_chunk,
        decode_chunk=decode_chunk,
    )
    out = out_chunk.reshape(bsz, nheads, nchunks, chunk_size, value_dim).permute(0, 2, 3, 1, 4).reshape(
        bsz,
        padded_len,
        nheads,
        value_dim,
    )
    return out[:, :seqlen].to(U.dtype).contiguous()


__all__ = [
    "_rms_normalize_last_dim",
    "flare_autoregressive_separated_pytorch",
]

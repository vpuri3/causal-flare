import math
import os

import torch
from torch import autograd


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


def _resolve_chunk_block_size(nchunks: int, *, env_name: str, default: int = 8) -> int:
    env = os.environ.get(env_name, "")
    if env:
        return max(1, min(nchunks, int(env)))
    return max(1, min(nchunks, default))


def _rms_normalize_last_dim(x: torch.Tensor, *, eps: float, scale_by_sqrt_dim: bool = False) -> torch.Tensor:
    rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    y = x / rms
    if scale_by_sqrt_dim:
        y = y / math.sqrt(x.shape[-1])
    return y


def _affine_prefix_scan_flat(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scan_A = A.clone()
    scan_B = B.clone()
    length = A.size(1)
    step = 1
    while step < length:
        shifted_A = torch.cat([torch.ones_like(scan_A[:, :step]), scan_A[:, :-step]], dim=1)
        shifted_B = torch.cat([torch.zeros_like(scan_B[:, :step]), scan_B[:, :-step]], dim=1)
        scan_B = scan_B + scan_A * shifted_B
        scan_A = scan_A * shifted_A
        step <<= 1
    return scan_A, scan_B


def _affine_scan_forward_flat(
    A: torch.Tensor,
    B: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if A.ndim != 3 or B.ndim != 3 or initial_state.ndim != 2:
        raise ValueError(
            "_affine_scan_forward_flat expects A=[P,T,S], B=[P,T,S], initial_state=[P,S]. "
            f"Got A={tuple(A.shape)}, B={tuple(B.shape)}, initial_state={tuple(initial_state.shape)}."
        )
    if A.shape != B.shape:
        raise ValueError(f"A and B must have identical shapes. Got A={tuple(A.shape)}, B={tuple(B.shape)}.")
    if A.shape[0] != initial_state.shape[0] or A.shape[2] != initial_state.shape[1]:
        raise ValueError(
            "initial_state must match the flattened batch/state dimensions of A/B. "
            f"Got A={tuple(A.shape)}, initial_state={tuple(initial_state.shape)}."
        )

    _, nchunks, _ = A.shape
    if nchunks == 0:
        chunk_start = A.new_empty((A.shape[0], 0, A.shape[2]))
        return chunk_start, chunk_start, chunk_start, initial_state

    inc_A, inc_B = _affine_prefix_scan_flat(A, B)
    excl_A = torch.cat([torch.ones_like(inc_A[:, :1]), inc_A[:, :-1]], dim=1)
    excl_B = torch.cat([torch.zeros_like(inc_B[:, :1]), inc_B[:, :-1]], dim=1)
    chunk_start = excl_A * initial_state.unsqueeze(1) + excl_B
    final_state = inc_A[:, -1] * initial_state + inc_B[:, -1]
    return inc_A, inc_B, chunk_start, final_state


def _affine_scan_backward_flat(
    A: torch.Tensor,
    chunk_start: torch.Tensor,
    grad_chunk_start: torch.Tensor,
    grad_final_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if grad_chunk_start is None:
        grad_chunk_start = torch.zeros_like(chunk_start)
    if grad_final_state is None:
        grad_final_state = torch.zeros((A.shape[0], A.shape[2]), device=A.device, dtype=A.dtype)

    _, _, rev_starts, grad_initial_state = _affine_scan_forward_flat(
        A.flip(1),
        grad_chunk_start.flip(1),
        grad_final_state,
    )
    grad_B = rev_starts.flip(1)
    grad_A = grad_B * chunk_start
    return grad_A, grad_B, grad_initial_state


class ChunkwiseAffineStateScan(autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor, initial_state: torch.Tensor):
        _, _, chunk_start, final_state = _affine_scan_forward_flat(A, B, initial_state)
        ctx.save_for_backward(A, chunk_start)
        return chunk_start, final_state

    @staticmethod
    def backward(ctx, grad_chunk_start: torch.Tensor, grad_final_state: torch.Tensor):
        A, chunk_start = ctx.saved_tensors
        grad_A, grad_B, grad_initial_state = _affine_scan_backward_flat(A, chunk_start, grad_chunk_start, grad_final_state)
        return grad_A, grad_B, grad_initial_state


def chunkwise_affine_state_scan(
    A: torch.Tensor,
    B: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ChunkwiseAffineStateScan.apply(A, B, initial_state)


def chunkwise_affine_state_scan_slots(
    A: torch.Tensor,
    B: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if A.ndim != 3:
        raise ValueError(f"A must be [P,NC,M]. Got {tuple(A.shape)}.")
    if B.ndim == 3:
        return chunkwise_affine_state_scan(A, B, initial_state)
    if B.ndim != 4:
        raise ValueError(f"B must be [P,NC,M] or [P,NC,M,D]. Got {tuple(B.shape)}.")
    if initial_state.ndim != 3:
        raise ValueError(f"initial_state must be [P,M,D]. Got {tuple(initial_state.shape)}.")
    if B.shape[:3] != A.shape:
        raise ValueError(f"B must match A in [P,NC,M]. Got A={tuple(A.shape)}, B={tuple(B.shape)}.")
    if initial_state.shape != (A.shape[0], A.shape[2], B.shape[3]):
        raise ValueError(
            "initial_state must match [P,M,D] of B. "
            f"Got B={tuple(B.shape)}, initial_state={tuple(initial_state.shape)}."
        )
    value_dim = B.shape[3]
    A_flat = A.unsqueeze(-1).expand(-1, -1, -1, value_dim).reshape(A.shape[0], A.shape[1], -1)
    B_flat = B.reshape(B.shape[0], B.shape[1], -1)
    initial_flat = initial_state.reshape(initial_state.shape[0], -1)
    chunk_start_flat, final_state_flat = chunkwise_affine_state_scan(A_flat, B_flat, initial_flat)
    chunk_start = chunk_start_flat.reshape(A.shape[0], A.shape[1], A.shape[2], value_dim)
    final_state = final_state_flat.reshape(A.shape[0], A.shape[2], value_dim)
    return chunk_start, final_state


def _segment_log_sums(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 1:
        raise ValueError(f"x must have a time dimension. Got {tuple(x.shape)}.")
    t = x.size(-1)
    x = x.unsqueeze(-1).expand(*x.shape, t)
    lower = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~lower, 0)
    x = torch.cumsum(x, dim=-2)
    valid = torch.tril(torch.ones(t, t, dtype=torch.bool, device=x.device), diagonal=0)
    x = x.masked_fill(~valid, -torch.inf)
    return x


_LOWER_TRI_FLOAT_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _lower_tri_float(size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (size, str(device), dtype)
    mask = _LOWER_TRI_FLOAT_CACHE.get(key)
    if mask is None:
        mask = torch.tril(torch.ones(size, size, device=device, dtype=dtype))
        _LOWER_TRI_FLOAT_CACHE[key] = mask
    return mask


def _pad_sequence_dim(x: torch.Tensor, pad: int, *, fill_value: float) -> torch.Tensor:
    if pad <= 0:
        return x
    fill = torch.full((x.shape[0], pad, *x.shape[2:]), fill_value, device=x.device, dtype=x.dtype)
    return torch.cat([x, fill], dim=1)


def _chunk_summary_terms_flat(
    retain_chunk: torch.Tensor,
    write_value_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk_A, suffix_exclusive, _, _ = _chunk_retention_terms(retain_chunk)
    if retain_chunk.ndim == 2:
        chunk_B = torch.einsum('pc,pcmd->pmd', suffix_exclusive, write_value_chunk)
    else:
        chunk_B = torch.einsum('pmc,pcmd->pmd', suffix_exclusive, write_value_chunk)
    return chunk_A, chunk_B


def _chunk_retention_terms(
    retain_chunk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if retain_chunk.ndim not in {2, 3}:
        raise ValueError(f"retain_chunk must be [P,C] or [P,C,M]. Got {tuple(retain_chunk.shape)}.")
    if retain_chunk.ndim == 2:
        tiny = torch.finfo(retain_chunk.dtype).tiny
        log_retain = retain_chunk.clamp_min(tiny).log()
        prefix = torch.cumsum(log_retain, dim=-1)
        decay_out = torch.exp(prefix)
        diff = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
        lower_tri = _lower_tri_float(
            retain_chunk.shape[1],
            device=retain_chunk.device,
            dtype=retain_chunk.dtype,
        ).unsqueeze(0)
        diff = diff.masked_fill(lower_tri == 0, -torch.inf)
        decay_intra = torch.exp(diff)
        chunk_A = torch.exp(prefix[..., -1])
        suffix_exclusive = torch.exp(prefix[..., -1:] - prefix)
        return chunk_A, suffix_exclusive, decay_intra, decay_out
    tiny = torch.finfo(retain_chunk.dtype).tiny
    log_retain = retain_chunk.clamp_min(tiny).log().transpose(1, 2)
    prefix = torch.cumsum(log_retain, dim=-1)
    decay_out = torch.exp(prefix).transpose(1, 2)
    diff = prefix.unsqueeze(-1) - prefix.unsqueeze(-2)
    lower_tri = _lower_tri_float(
        retain_chunk.shape[1],
        device=retain_chunk.device,
        dtype=retain_chunk.dtype,
    ).unsqueeze(0).unsqueeze(0)
    diff = diff.masked_fill(lower_tri == 0, -torch.inf)
    decay_intra = torch.exp(diff)
    chunk_A = torch.exp(prefix[..., -1])
    suffix_exclusive = torch.exp(prefix[..., -1:] - prefix)
    return chunk_A, suffix_exclusive, decay_intra, decay_out


def _chunk_inclusive_states_flat(
    state0: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_value_chunk: torch.Tensor,
) -> torch.Tensor:
    _, _, decay_intra, decay_out = _chunk_retention_terms(retain_chunk)
    if retain_chunk.ndim == 2:
        off_state = decay_out.unsqueeze(-1).unsqueeze(-1) * state0.unsqueeze(1)
        diag_state = torch.einsum('pts,psmd->ptmd', decay_intra, write_value_chunk)
    else:
        off_state = decay_out.unsqueeze(-1) * state0.unsqueeze(1)
        diag_state = torch.einsum('pmts,psmd->ptmd', decay_intra, write_value_chunk)
    return off_state + diag_state


def _main_chunk_output(
    state0: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_chunk: torch.Tensor,
    U_chunk: torch.Tensor,
    decode_chunk: torch.Tensor,
    *,
    rmsnorm_read_contrib: bool,
) -> torch.Tensor:
    if state0.ndim != 3:
        raise ValueError(f"state0 must be [P,M,D]. Got {tuple(state0.shape)}.")
    if retain_chunk.ndim != 2:
        raise ValueError(f"retain_chunk must be [P,C]. Got {tuple(retain_chunk.shape)}.")
    if write_chunk.ndim != 3:
        raise ValueError(f"write_chunk must be [P,C,M]. Got {tuple(write_chunk.shape)}.")
    if U_chunk.ndim != 3:
        raise ValueError(f"U_chunk must be [P,C,D]. Got {tuple(U_chunk.shape)}.")
    if decode_chunk.ndim != 3:
        raise ValueError(f"decode_chunk must be [P,C,M]. Got {tuple(decode_chunk.shape)}.")

    if rmsnorm_read_contrib:
        # TODO(vedantpu): reintroduce a fast chunk-local path for
        # `rmsnorm_read_contrib=True` after the non-RMSNorm recurrence is fully
        # optimized and stable.
        raise NotImplementedError("rmsnorm_read_contrib=True is temporarily unsupported in separated.py.")

    _, _, decay_intra, decay_out = _chunk_retention_terms(retain_chunk)
    y_off = torch.einsum('ptm,pt,pmd->ptd', decode_chunk, decay_out, state0)
    local_kernel = torch.einsum('ptm,pts,psm->pts', decode_chunk, decay_intra, write_chunk)
    y_diag = torch.einsum('pts,psd->ptd', local_kernel, U_chunk)
    return y_off + y_diag


def _exclusive_chunk_states(state0: torch.Tensor, inclusive_states: torch.Tensor) -> torch.Tensor:
    if inclusive_states.shape[1] == 0:
        return inclusive_states
    return torch.cat([state0.unsqueeze(1), inclusive_states[:, :-1]], dim=1)


def _main_chunk_forward_impl(
    state0: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_chunk: torch.Tensor,
    U_chunk: torch.Tensor,
    decode_chunk: torch.Tensor,
    *,
    rmsnorm_read_contrib: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rmsnorm_read_contrib:
        # TODO(vedantpu): restore an optimized RMSNorm-read-contrib path once the
        # non-RMSNorm implementation is in its final form.
        raise NotImplementedError("rmsnorm_read_contrib=True is temporarily unsupported in separated.py.")

    chunk_A, suffix_exclusive, decay_intra, decay_out = _chunk_retention_terms(retain_chunk)
    y_off = torch.einsum('ptm,pt,pmd->ptd', decode_chunk, decay_out, state0)
    local_kernel = torch.einsum('ptm,pts,psm->pts', decode_chunk, decay_intra, write_chunk)
    out = y_off + torch.einsum('pts,psd->ptd', local_kernel, U_chunk)
    write_value_chunk = write_chunk.unsqueeze(-1) * U_chunk.unsqueeze(-2)
    chunk_B = torch.einsum('pc,pcmd->pmd', suffix_exclusive, write_value_chunk)
    state1 = chunk_A.unsqueeze(-1).unsqueeze(-1) * state0 + chunk_B
    return out, state1


def _main_block_forward_impl(
    state0: torch.Tensor,
    retain_block: torch.Tensor,
    write_block: torch.Tensor,
    U_block: torch.Tensor,
    decode_block: torch.Tensor,
    *,
    rmsnorm_read_contrib: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_batch, block_chunks, chunk_size, value_dim = U_block.shape
    nslots = write_block.shape[-1]
    write_value_block = write_block.unsqueeze(-1) * U_block.unsqueeze(-2)
    chunk_A_flat, chunk_B_flat = _chunk_summary_terms_flat(
        retain_block.reshape(flat_batch * block_chunks, chunk_size),
        write_value_block.reshape(flat_batch * block_chunks, chunk_size, nslots, value_dim),
    )
    chunk_A = chunk_A_flat.reshape(flat_batch, block_chunks)
    chunk_B = chunk_B_flat.reshape(flat_batch, block_chunks, nslots, value_dim)
    start_states, final_state = chunkwise_affine_state_scan(
        chunk_A.unsqueeze(-1).expand(-1, -1, nslots * value_dim),
        chunk_B.reshape(flat_batch, block_chunks, -1),
        state0.reshape(flat_batch, -1),
    )
    start_states = start_states.reshape(flat_batch, block_chunks, nslots, value_dim)
    final_state = final_state.reshape(flat_batch, nslots, value_dim)
    out = _main_chunk_output(
        start_states.reshape(flat_batch * block_chunks, nslots, value_dim),
        retain_block.reshape(flat_batch * block_chunks, chunk_size),
        write_block.reshape(flat_batch * block_chunks, chunk_size, nslots),
        U_block.reshape(flat_batch * block_chunks, chunk_size, value_dim),
        decode_block.reshape(flat_batch * block_chunks, chunk_size, nslots),
        rmsnorm_read_contrib=rmsnorm_read_contrib,
    ).reshape(flat_batch, block_chunks, chunk_size, value_dim)
    return out, final_state, start_states


def _aux_chunk_forward_impl(
    state0: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_chunk: torch.Tensor,
    U_chunk: torch.Tensor,
    slot_embed: torch.Tensor,
    readout: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, nheads, _, value_dim = U_chunk.shape
    nslots = retain_chunk.shape[-1]
    aux_dim = state0.shape[-1]
    if slot_embed.ndim == 3:
        score_chunk = torch.einsum('bhcd,hmd->bhcm', U_chunk, slot_embed).unsqueeze(-1)
    else:
        score_chunk = torch.einsum('bhcd,hmad->bhcma', U_chunk, slot_embed)

    flat_state = state0.reshape(bsz * nheads, nslots, aux_dim)
    flat_retain = retain_chunk.reshape(bsz * nheads, U_chunk.shape[2], nslots)
    flat_write = write_chunk.reshape(bsz * nheads, U_chunk.shape[2], nslots)
    flat_score = score_chunk.reshape(bsz * nheads, U_chunk.shape[2], nslots, aux_dim)
    write_value_chunk = flat_write.unsqueeze(-1) * flat_score
    chunk_A, suffix_exclusive, decay_intra, decay_out = _chunk_retention_terms(flat_retain)
    inclusive_states = decay_out.unsqueeze(-1) * flat_state.unsqueeze(1) + torch.einsum('pmts,psmd->ptmd', decay_intra, write_value_chunk)
    exclusive_states = _exclusive_chunk_states(flat_state, inclusive_states).reshape(bsz, nheads, U_chunk.shape[2], nslots, aux_dim)
    chunk_B = torch.einsum('pmc,pcmd->pmd', suffix_exclusive, write_value_chunk)
    state1 = (chunk_A.unsqueeze(-1) * flat_state + chunk_B).reshape(bsz, nheads, nslots, aux_dim)
    if readout is not None:
        out = torch.einsum('bhcma,hma->bhcm', exclusive_states, readout)
    elif aux_dim == 1:
        out = exclusive_states.squeeze(-1)
    else:
        out = exclusive_states
    return out, state1


def _aux_block_forward_impl(
    state0: torch.Tensor,
    retain_block: torch.Tensor,
    write_block: torch.Tensor,
    U_block: torch.Tensor,
    slot_embed: torch.Tensor,
    readout: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, nheads, block_chunks, chunk_size, value_dim = U_block.shape
    nslots = retain_block.shape[-1]
    aux_dim = state0.shape[-1]

    if slot_embed.ndim == 3:
        score_block = torch.einsum('bhkcd,hmd->bhkcm', U_block, slot_embed).unsqueeze(-1)
    else:
        score_block = torch.einsum('bhkcd,hmad->bhkcma', U_block, slot_embed)

    flat_batch = bsz * nheads
    flat_state0 = state0.reshape(flat_batch, nslots, aux_dim)
    flat_retain = retain_block.permute(0, 1, 2, 3, 4).reshape(flat_batch, block_chunks, chunk_size, nslots)
    flat_write = write_block.permute(0, 1, 2, 3, 4).reshape(flat_batch, block_chunks, chunk_size, nslots)
    flat_score = score_block.reshape(flat_batch, block_chunks, chunk_size, nslots, aux_dim)
    write_value_block = flat_write.unsqueeze(-1) * flat_score
    chunk_A_flat, chunk_B_flat = _chunk_summary_terms_flat(
        flat_retain.reshape(flat_batch * block_chunks, chunk_size, nslots),
        write_value_block.reshape(flat_batch * block_chunks, chunk_size, nslots, aux_dim),
    )
    chunk_A = chunk_A_flat.reshape(flat_batch, block_chunks, nslots)
    chunk_B = chunk_B_flat.reshape(flat_batch, block_chunks, nslots, aux_dim)
    start_states, final_state = chunkwise_affine_state_scan_slots(chunk_A, chunk_B, flat_state0)

    state0_batch = start_states.reshape(bsz, nheads, block_chunks, nslots, aux_dim).permute(0, 2, 1, 3, 4).reshape(
        bsz * block_chunks, nheads, nslots, aux_dim
    )
    retain_batch = retain_block.permute(0, 2, 1, 3, 4).reshape(bsz * block_chunks, nheads, chunk_size, nslots)
    write_batch = write_block.permute(0, 2, 1, 3, 4).reshape(bsz * block_chunks, nheads, chunk_size, nslots)
    U_batch = U_block.permute(0, 2, 1, 3, 4).reshape(bsz * block_chunks, nheads, chunk_size, value_dim)
    out, _ = _aux_chunk_forward_impl(state0_batch, retain_batch, write_batch, U_batch, slot_embed, readout)
    if readout is not None or aux_dim == 1:
        out = out.reshape(bsz, block_chunks, nheads, chunk_size, nslots).permute(0, 2, 1, 3, 4)
    else:
        out = out.reshape(bsz, block_chunks, nheads, chunk_size, nslots, aux_dim).permute(0, 2, 1, 3, 4, 5)
    final_state = final_state.reshape(bsz, nheads, nslots, aux_dim)
    start_states = start_states.reshape(bsz, nheads, block_chunks, nslots, aux_dim)
    return out, final_state, start_states


def _main_chunk_backward_prepare(
    *,
    state0: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_chunk: torch.Tensor,
    U_chunk: torch.Tensor,
    decode_chunk: torch.Tensor,
    grad_out: torch.Tensor,
    rmsnorm_read_contrib: bool,
) -> tuple[torch.dtype, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rmsnorm_read_contrib:
        raise NotImplementedError("rmsnorm_read_contrib=True is temporarily unsupported in separated.py.")

    accum_dtype = torch.float32 if state0.dtype in {torch.float16, torch.bfloat16} else state0.dtype
    state0_acc = state0.to(accum_dtype)
    retain_acc = retain_chunk.to(accum_dtype)
    write_acc = write_chunk.to(accum_dtype)
    U_acc = U_chunk.to(accum_dtype)
    decode_acc = decode_chunk.to(accum_dtype)
    grad_out_acc = grad_out.to(accum_dtype)

    write_value_chunk = write_acc.unsqueeze(-1) * U_acc.unsqueeze(-2)
    chunk_A, _, _, _ = _chunk_retention_terms(retain_acc)
    states = _chunk_inclusive_states_flat(state0_acc, retain_acc, write_value_chunk)
    prev_states = torch.cat([state0_acc.unsqueeze(1), states[:, :-1]], dim=1)

    grad_decode = (grad_out_acc.unsqueeze(2) * states).sum(dim=-1)
    grad_from_output = decode_acc.unsqueeze(-1) * grad_out_acc.unsqueeze(2)

    return accum_dtype, chunk_A, retain_acc, write_acc, U_acc, prev_states, grad_from_output, grad_decode


def _chunk_reverse_scan(
    retain_acc: torch.Tensor,
    grad_from_output: torch.Tensor,
    grad_state1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    rev_retain = torch.cat([torch.zeros_like(retain_acc[:, :1]), retain_acc[:, 1:].flip(1)], dim=1)
    rev_input = grad_from_output.flip(1)
    rev_input[:, 0] = rev_input[:, 0] + grad_state1
    zero_state = torch.zeros_like(grad_state1)
    grad_state_post = _chunk_inclusive_states_flat(zero_state, rev_retain, rev_input).flip(1)
    if retain_acc.ndim == 2:
        grad_state0 = retain_acc[:, 0].unsqueeze(-1).unsqueeze(-1) * grad_state_post[:, 0]
    else:
        grad_state0 = retain_acc[:, 0].unsqueeze(-1) * grad_state_post[:, 0]
    return grad_state_post, grad_state0


def _main_chunk_backward_finalize(
    *,
    retain_acc: torch.Tensor,
    write_acc: torch.Tensor,
    U_acc: torch.Tensor,
    prev_states: torch.Tensor,
    grad_state_post: torch.Tensor,
    grad_decode: torch.Tensor,
    retain_dtype: torch.dtype,
    write_dtype: torch.dtype,
    U_dtype: torch.dtype,
    decode_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_retain = (grad_state_post * prev_states).sum(dim=(-1, -2))
    grad_write = (grad_state_post * U_acc.unsqueeze(2)).sum(dim=-1)
    grad_U = (grad_state_post * write_acc.unsqueeze(-1)).sum(dim=2)
    return (
        grad_retain.to(retain_dtype),
        grad_write.to(write_dtype),
        grad_U.to(U_dtype),
        grad_decode.to(decode_dtype),
    )


def _main_chunk_backward_manual(
    *,
    state0: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_chunk: torch.Tensor,
    U_chunk: torch.Tensor,
    decode_chunk: torch.Tensor,
    grad_out: torch.Tensor,
    grad_state1: torch.Tensor,
    rmsnorm_read_contrib: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rmsnorm_read_contrib:
        raise NotImplementedError("rmsnorm_read_contrib=True is temporarily unsupported in separated.py.")

    accum_dtype = torch.float32 if state0.dtype in {torch.float16, torch.bfloat16} else state0.dtype
    state0_acc = state0.to(accum_dtype)
    retain_acc = retain_chunk.to(accum_dtype)
    write_acc = write_chunk.to(accum_dtype)
    U_acc = U_chunk.to(accum_dtype)
    decode_acc = decode_chunk.to(accum_dtype)
    grad_out_acc = grad_out.to(accum_dtype)
    grad_state = grad_state1.to(accum_dtype)

    write_value_chunk = write_acc.unsqueeze(-1) * U_acc.unsqueeze(-2)
    states = _chunk_inclusive_states_flat(state0_acc, retain_acc, write_value_chunk)
    prev_states = torch.cat([state0_acc.unsqueeze(1), states[:, :-1]], dim=1)

    grad_decode = (grad_out_acc.unsqueeze(2) * states).sum(dim=-1)
    grad_from_output = decode_acc.unsqueeze(-1) * grad_out_acc.unsqueeze(2)

    rev_retain = torch.cat([torch.zeros_like(retain_acc[:, :1]), retain_acc[:, 1:].flip(1)], dim=1)
    rev_input = grad_from_output.flip(1)
    rev_input[:, 0] = rev_input[:, 0] + grad_state
    grad_state_post = _chunk_inclusive_states_flat(torch.zeros_like(state0_acc), rev_retain, rev_input).flip(1)
    grad_retain = (grad_state_post * prev_states).sum(dim=(-1, -2))
    grad_write = (grad_state_post * U_acc.unsqueeze(2)).sum(dim=-1)
    grad_U = (grad_state_post * write_acc.unsqueeze(-1)).sum(dim=2)
    grad_state = retain_acc[:, 0].unsqueeze(-1).unsqueeze(-1) * grad_state_post[:, 0]

    return (
        grad_state,
        grad_retain.to(retain_chunk.dtype),
        grad_write.to(write_chunk.dtype),
        grad_U.to(U_chunk.dtype),
        grad_decode.to(decode_chunk.dtype),
    )


def _main_chunk_backward_batch_manual(
    *,
    state0: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_chunk: torch.Tensor,
    U_chunk: torch.Tensor,
    decode_chunk: torch.Tensor,
    grad_out: torch.Tensor,
    grad_state1: torch.Tensor,
    rmsnorm_read_contrib: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_chunks = state0.shape[1]
    nslots = state0.shape[2]
    value_dim = state0.shape[3]
    chunk_size = U_chunk.shape[2]
    flat_state0 = state0.reshape(-1, nslots, value_dim)
    flat_retain = retain_chunk.reshape(-1, chunk_size)
    flat_write = write_chunk.reshape(-1, chunk_size, nslots)
    flat_U = U_chunk.reshape(-1, chunk_size, value_dim)
    flat_decode = decode_chunk.reshape(-1, chunk_size, nslots)
    flat_grad_out = grad_out.reshape(-1, chunk_size, value_dim)

    accum_dtype, chunk_A, retain_acc, write_acc, U_acc, prev_states, grad_from_output, grad_decode = _main_chunk_backward_prepare(
        state0=flat_state0,
        retain_chunk=flat_retain,
        write_chunk=flat_write,
        U_chunk=flat_U,
        decode_chunk=flat_decode,
        grad_out=flat_grad_out,
        rmsnorm_read_contrib=rmsnorm_read_contrib,
    )

    zero_state = torch.zeros_like(flat_state0)
    _, grad_state_from_output = _chunk_reverse_scan(retain_acc, grad_from_output, zero_state)
    grad_state_from_output = grad_state_from_output.reshape(state0.shape[0], block_chunks, nslots, value_dim)
    chunk_A = chunk_A.reshape(state0.shape[0], block_chunks)
    rev_chunk_starts, grad_state_block = chunkwise_affine_state_scan(
        chunk_A.flip(1).unsqueeze(-1).expand(-1, -1, nslots * value_dim),
        grad_state_from_output.flip(1).reshape(state0.shape[0], block_chunks, -1),
        grad_state1.to(accum_dtype).reshape(state0.shape[0], -1),
    )
    grad_state1_chunk = rev_chunk_starts.flip(1).reshape(-1, nslots, value_dim)
    grad_state_block = grad_state_block.reshape(state0.shape[0], nslots, value_dim)
    grad_state_post, _ = _chunk_reverse_scan(retain_acc, grad_from_output, grad_state1_chunk)
    grad_retain, grad_write, grad_U, grad_decode = _main_chunk_backward_finalize(
        retain_acc=retain_acc,
        write_acc=write_acc,
        U_acc=U_acc,
        prev_states=prev_states,
        grad_state_post=grad_state_post,
        grad_decode=grad_decode,
        retain_dtype=retain_chunk.dtype,
        write_dtype=write_chunk.dtype,
        U_dtype=U_chunk.dtype,
        decode_dtype=decode_chunk.dtype,
    )

    return (
        grad_state_block.to(state0.dtype),
        grad_retain.reshape(retain_chunk.shape),
        grad_write.reshape(write_chunk.shape),
        grad_U.reshape(U_chunk.shape),
        grad_decode.reshape(decode_chunk.shape),
        chunk_A,
    )


def _aux_chunk_backward_manual(
    *,
    state0: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_chunk: torch.Tensor,
    U_chunk: torch.Tensor,
    slot_embed: torch.Tensor,
    readout: torch.Tensor | None,
    grad_out: torch.Tensor,
    grad_state1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    bsz, nheads, chunk_size, value_dim = U_chunk.shape
    nslots = retain_chunk.shape[-1]
    aux_dim = state0.shape[-1]
    accum_dtype = torch.float32 if state0.dtype in {torch.float16, torch.bfloat16} else state0.dtype

    state0_acc = state0.to(accum_dtype)
    retain_acc = retain_chunk.to(accum_dtype)
    write_acc = write_chunk.to(accum_dtype)
    U_acc = U_chunk.to(accum_dtype)
    grad_out_acc = grad_out.to(accum_dtype)
    grad_state = grad_state1.to(accum_dtype)

    if slot_embed.ndim == 3:
        slot_embed_acc = slot_embed.to(accum_dtype)
        score_chunk = torch.einsum('bhcd,hmd->bhcm', U_acc, slot_embed_acc).unsqueeze(-1)
    else:
        slot_embed_acc = slot_embed.to(accum_dtype)
        score_chunk = torch.einsum('bhcd,hmad->bhcma', U_acc, slot_embed_acc)

    flat_state0 = state0_acc.reshape(bsz * nheads, nslots, aux_dim)
    flat_retain = retain_acc.reshape(bsz * nheads, chunk_size, nslots)
    flat_write = write_acc.reshape(bsz * nheads, chunk_size, nslots)
    flat_score = score_chunk.reshape(bsz * nheads, chunk_size, nslots, aux_dim)
    write_value_chunk = flat_write.unsqueeze(-1) * flat_score
    inclusive_states = _chunk_inclusive_states_flat(flat_state0, flat_retain, write_value_chunk).reshape(
        bsz, nheads, chunk_size, nslots, aux_dim
    )
    exclusive_states = torch.cat([state0_acc.unsqueeze(2), inclusive_states[:, :, :-1]], dim=2)

    grad_readout = torch.zeros_like(readout, dtype=accum_dtype) if readout is not None else None

    if readout is not None:
        readout_acc = readout.to(accum_dtype)
        grad_exclusive = grad_out_acc.unsqueeze(-1) * readout_acc.unsqueeze(0).unsqueeze(2)
        grad_readout = torch.einsum('bhcm,bhcma->hma', grad_out_acc, exclusive_states)
    elif aux_dim == 1:
        grad_exclusive = grad_out_acc.unsqueeze(-1)
    else:
        grad_exclusive = grad_out_acc

    flat_exclusive = grad_exclusive.reshape(bsz * nheads, chunk_size, nslots, aux_dim)
    flat_retain = retain_acc.reshape(bsz * nheads, chunk_size, nslots)
    rev_retain = torch.cat([torch.zeros_like(flat_retain[:, :1]), flat_retain[:, 1:].flip(1)], dim=1)
    shifted_exclusive = torch.cat([flat_exclusive[:, 1:], torch.zeros_like(flat_exclusive[:, :1])], dim=1)
    rev_input = shifted_exclusive.flip(1)
    rev_input[:, 0] = rev_input[:, 0] + grad_state.reshape(bsz * nheads, nslots, aux_dim)
    grad_state_post = _chunk_inclusive_states_flat(
        torch.zeros((bsz * nheads, nslots, aux_dim), device=state0.device, dtype=accum_dtype),
        rev_retain,
        rev_input,
    ).flip(1).reshape(bsz, nheads, chunk_size, nslots, aux_dim)
    grad_retain = (grad_state_post * exclusive_states).sum(dim=-1)
    grad_write = (grad_state_post * score_chunk).sum(dim=-1)
    grad_score = grad_state_post * write_acc.unsqueeze(-1)
    if slot_embed.ndim == 3:
        grad_score_scalar = grad_score.squeeze(-1)
        grad_U = torch.einsum('bhcm,hmd->bhcd', grad_score_scalar, slot_embed_acc)
        grad_slot_embed = torch.einsum('bhcd,bhcm->hmd', U_acc, grad_score_scalar)
    else:
        grad_U = torch.einsum('bhcma,hmad->bhcd', grad_score, slot_embed_acc)
        grad_slot_embed = torch.einsum('bhcd,bhcma->hmad', U_acc, grad_score)
    grad_state = grad_exclusive[:, :, 0] + retain_acc[:, :, 0].unsqueeze(-1) * grad_state_post[:, :, 0]

    return (
        grad_state.to(state0.dtype),
        grad_retain.to(retain_chunk.dtype),
        grad_write.to(write_chunk.dtype),
        grad_U.to(U_chunk.dtype),
        grad_slot_embed.to(slot_embed.dtype),
        None if grad_readout is None else grad_readout.to(readout.dtype),
    )


def _aux_chunk_backward_prepare(
    *,
    state0: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_chunk: torch.Tensor,
    U_chunk: torch.Tensor,
    slot_embed: torch.Tensor,
    readout: torch.Tensor | None,
    grad_out: torch.Tensor,
) -> tuple[torch.dtype, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    bsz, nheads, chunk_size, value_dim = U_chunk.shape
    nslots = retain_chunk.shape[-1]
    aux_dim = state0.shape[-1]
    accum_dtype = torch.float32 if state0.dtype in {torch.float16, torch.bfloat16} else state0.dtype

    state0_acc = state0.to(accum_dtype)
    retain_acc = retain_chunk.to(accum_dtype)
    write_acc = write_chunk.to(accum_dtype)
    U_acc = U_chunk.to(accum_dtype)
    grad_out_acc = grad_out.to(accum_dtype)
    slot_embed_acc = slot_embed.to(accum_dtype)
    if readout is not None:
        readout_acc = readout.to(accum_dtype)
    else:
        readout_acc = None

    if slot_embed.ndim == 3:
        score_chunk = torch.einsum('bhcd,hmd->bhcm', U_acc, slot_embed_acc).unsqueeze(-1)
    else:
        score_chunk = torch.einsum('bhcd,hmad->bhcma', U_acc, slot_embed_acc)

    flat_state0 = state0_acc.reshape(bsz * nheads, nslots, aux_dim)
    flat_retain = retain_acc.reshape(bsz * nheads, chunk_size, nslots)
    flat_write = write_acc.reshape(bsz * nheads, chunk_size, nslots)
    flat_score = score_chunk.reshape(bsz * nheads, chunk_size, nslots, aux_dim)
    write_value_chunk = flat_write.unsqueeze(-1) * flat_score
    chunk_A, _, _, _ = _chunk_retention_terms(flat_retain)
    inclusive_states = _chunk_inclusive_states_flat(flat_state0, flat_retain, write_value_chunk).reshape(
        bsz, nheads, chunk_size, nslots, aux_dim
    )
    exclusive_states = torch.cat([state0_acc.unsqueeze(2), inclusive_states[:, :, :-1]], dim=2)

    grad_readout = torch.zeros_like(readout, dtype=accum_dtype) if readout is not None else None
    if readout_acc is not None:
        grad_exclusive = grad_out_acc.unsqueeze(-1) * readout_acc.unsqueeze(0).unsqueeze(2)
        grad_readout = torch.einsum('bhcm,bhcma->hma', grad_out_acc, exclusive_states)
    elif aux_dim == 1:
        grad_exclusive = grad_out_acc.unsqueeze(-1)
    else:
        grad_exclusive = grad_out_acc

    return (
        accum_dtype,
        chunk_A,
        retain_acc,
        write_acc,
        U_acc,
        score_chunk,
        exclusive_states,
        grad_exclusive,
        slot_embed_acc,
        grad_readout,
    )


def _aux_chunk_backward_finalize(
    *,
    retain_acc: torch.Tensor,
    write_acc: torch.Tensor,
    U_acc: torch.Tensor,
    score_chunk: torch.Tensor,
    exclusive_states: torch.Tensor,
    grad_state_post: torch.Tensor,
    slot_embed_acc: torch.Tensor,
    slot_embed_dtype: torch.dtype,
    readout_grad: torch.Tensor | None,
    retain_dtype: torch.dtype,
    write_dtype: torch.dtype,
    U_dtype: torch.dtype,
    readout_dtype: torch.dtype | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    grad_retain = (grad_state_post * exclusive_states).sum(dim=-1)
    grad_write = (grad_state_post * score_chunk).sum(dim=-1)
    grad_score = grad_state_post * write_acc.unsqueeze(-1)
    if slot_embed_acc.ndim == 3:
        grad_score_scalar = grad_score.squeeze(-1)
        grad_U = torch.einsum('bhcm,hmd->bhcd', grad_score_scalar, slot_embed_acc)
        grad_slot_embed = torch.einsum('bhcd,bhcm->hmd', U_acc, grad_score_scalar)
    else:
        grad_U = torch.einsum('bhcma,hmad->bhcd', grad_score, slot_embed_acc)
        grad_slot_embed = torch.einsum('bhcd,bhcma->hmad', U_acc, grad_score)
    return (
        grad_retain.to(retain_dtype),
        grad_write.to(write_dtype),
        grad_U.to(U_dtype),
        grad_slot_embed.to(slot_embed_dtype),
        None if readout_grad is None or readout_dtype is None else readout_grad.to(readout_dtype),
    )


def _aux_chunk_backward_batch_manual(
    *,
    state0: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_chunk: torch.Tensor,
    U_chunk: torch.Tensor,
    slot_embed: torch.Tensor,
    readout: torch.Tensor | None,
    grad_out: torch.Tensor,
    grad_state1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    bsz, nheads, block_chunks, nslots, aux_dim = state0.shape
    chunk_size = U_chunk.shape[3]
    state0_batch = state0.permute(0, 2, 1, 3, 4).reshape(bsz * block_chunks, nheads, nslots, aux_dim)
    retain_batch = retain_chunk.permute(0, 2, 1, 3, 4).reshape(bsz * block_chunks, nheads, chunk_size, nslots)
    write_batch = write_chunk.permute(0, 2, 1, 3, 4).reshape(bsz * block_chunks, nheads, chunk_size, nslots)
    U_batch = U_chunk.permute(0, 2, 1, 3, 4).reshape(bsz * block_chunks, nheads, chunk_size, U_chunk.shape[-1])
    if grad_out.ndim == 6:
        grad_out_batch = grad_out.permute(0, 2, 1, 3, 4, 5).reshape(bsz * block_chunks, nheads, chunk_size, nslots, aux_dim)
    else:
        grad_out_batch = grad_out.permute(0, 2, 1, 3, 4).reshape(bsz * block_chunks, nheads, chunk_size, nslots)

    (
        accum_dtype,
        chunk_A_flat,
        retain_acc,
        write_acc,
        U_acc,
        score_chunk,
        exclusive_states,
        grad_exclusive,
        slot_embed_acc,
        grad_readout_direct,
    ) = _aux_chunk_backward_prepare(
        state0=state0_batch,
        retain_chunk=retain_batch,
        write_chunk=write_batch,
        U_chunk=U_batch,
        slot_embed=slot_embed,
        readout=readout,
        grad_out=grad_out_batch,
    )

    flat_exclusive = grad_exclusive.reshape(bsz * block_chunks * nheads, chunk_size, nslots, aux_dim)
    flat_retain = retain_acc.reshape(bsz * block_chunks * nheads, chunk_size, nslots)
    shifted_exclusive = torch.cat([flat_exclusive[:, 1:], torch.zeros_like(flat_exclusive[:, :1])], dim=1)
    zero_state = torch.zeros((bsz * block_chunks * nheads, nslots, aux_dim), device=state0.device, dtype=accum_dtype)
    _, grad_state_from_output_flat = _chunk_reverse_scan(flat_retain, shifted_exclusive, zero_state)
    grad_state_from_output_flat = flat_exclusive[:, 0] + grad_state_from_output_flat
    grad_state_from_output = grad_state_from_output_flat.reshape(bsz, block_chunks, nheads, nslots, aux_dim).permute(0, 2, 1, 3, 4)
    chunk_A = chunk_A_flat.reshape(bsz, block_chunks, nheads, nslots).permute(0, 2, 1, 3)
    rev_chunk_starts, grad_state_block = chunkwise_affine_state_scan_slots(
        chunk_A.flip(2).reshape(bsz * nheads, block_chunks, nslots),
        grad_state_from_output.flip(2).reshape(bsz * nheads, block_chunks, nslots, aux_dim),
        grad_state1.to(accum_dtype).reshape(bsz * nheads, nslots, aux_dim),
    )
    grad_state1_chunk = rev_chunk_starts.flip(1).reshape(bsz, nheads, block_chunks, nslots, aux_dim)
    grad_state1_batch = grad_state1_chunk.permute(0, 2, 1, 3, 4).reshape(bsz * block_chunks * nheads, nslots, aux_dim)
    grad_state_post_flat, _ = _chunk_reverse_scan(flat_retain, shifted_exclusive, grad_state1_batch)
    grad_state_block = grad_state_block.reshape(bsz, nheads, nslots, aux_dim)
    grad_state_post = grad_state_post_flat.reshape(bsz * block_chunks, nheads, chunk_size, nslots, aux_dim)

    grad_retain, grad_write, grad_U, grad_slot_embed, grad_readout = _aux_chunk_backward_finalize(
        retain_acc=retain_acc,
        write_acc=write_acc,
        U_acc=U_acc,
        score_chunk=score_chunk,
        exclusive_states=exclusive_states,
        grad_state_post=grad_state_post,
        slot_embed_acc=slot_embed_acc,
        slot_embed_dtype=slot_embed.dtype,
        readout_grad=grad_readout_direct,
        retain_dtype=retain_chunk.dtype,
        write_dtype=write_chunk.dtype,
        U_dtype=U_chunk.dtype,
        readout_dtype=None if readout is None else readout.dtype,
    )

    return (
        grad_state_block.to(state0.dtype),
        grad_retain.reshape(bsz, block_chunks, nheads, chunk_size, nslots).permute(0, 2, 1, 3, 4),
        grad_write.reshape(bsz, block_chunks, nheads, chunk_size, nslots).permute(0, 2, 1, 3, 4),
        grad_U.reshape(bsz, block_chunks, nheads, chunk_size, U_chunk.shape[-1]).permute(0, 2, 1, 3, 4),
        grad_slot_embed,
        grad_readout,
        chunk_A,
    )


class FLAREAutoregressiveSeparatedFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        U_chunk: torch.Tensor,
        retain_chunk: torch.Tensor,
        write_chunk: torch.Tensor,
        decode_chunk: torch.Tensor,
    ) -> torch.Tensor:
        flat_batch, nchunks, chunk_size, value_dim = U_chunk.shape
        nslots = write_chunk.shape[-1]
        write_value_chunk = write_chunk.unsqueeze(-1) * U_chunk.unsqueeze(-2)
        chunk_A_flat, chunk_B_flat = _chunk_summary_terms_flat(
            retain_chunk.reshape(flat_batch * nchunks, chunk_size),
            write_value_chunk.reshape(flat_batch * nchunks, chunk_size, nslots, value_dim),
        )
        chunk_A = chunk_A_flat.reshape(flat_batch, nchunks)
        chunk_B = chunk_B_flat.reshape(flat_batch, nchunks, nslots, value_dim)
        start_states, _ = chunkwise_affine_state_scan(
            chunk_A.unsqueeze(-1).expand(-1, -1, nslots * value_dim),
            chunk_B.reshape(flat_batch, nchunks, -1),
            U_chunk.new_zeros((flat_batch, nslots * value_dim)),
        )
        start_states = start_states.reshape(flat_batch, nchunks, nslots, value_dim)
        out_chunk = _main_chunk_output(
            start_states.reshape(flat_batch * nchunks, nslots, value_dim),
            retain_chunk.reshape(flat_batch * nchunks, chunk_size),
            write_chunk.reshape(flat_batch * nchunks, chunk_size, nslots),
            U_chunk.reshape(flat_batch * nchunks, chunk_size, value_dim),
            decode_chunk.reshape(flat_batch * nchunks, chunk_size, nslots),
            rmsnorm_read_contrib=False,
        ).reshape(flat_batch, nchunks, chunk_size, value_dim)

        ctx.save_for_backward(U_chunk, retain_chunk, write_chunk, decode_chunk, start_states)
        return out_chunk

    @staticmethod
    def backward(ctx, grad_out_chunk: torch.Tensor):
        U_chunk, retain_chunk, write_chunk, decode_chunk, start_states = ctx.saved_tensors
        flat_batch, nchunks, _, value_dim = U_chunk.shape
        nslots = write_chunk.shape[-1]

        grad_U = torch.zeros_like(U_chunk) if U_chunk.requires_grad else None
        grad_retain = torch.zeros_like(retain_chunk) if retain_chunk.requires_grad else None
        grad_write = torch.zeros_like(write_chunk) if write_chunk.requires_grad else None
        grad_decode = torch.zeros_like(decode_chunk) if decode_chunk.requires_grad else None
        grad_state, grad_retain_cur, grad_write_cur, grad_U_cur, grad_decode_cur, _ = _main_chunk_backward_batch_manual(
            state0=start_states,
            retain_chunk=retain_chunk,
            write_chunk=write_chunk,
            U_chunk=U_chunk,
            decode_chunk=decode_chunk,
            grad_out=grad_out_chunk,
            grad_state1=U_chunk.new_zeros((flat_batch, nslots, value_dim)),
            rmsnorm_read_contrib=False,
        )
        if grad_retain is not None:
            grad_retain.copy_(grad_retain_cur)
        if grad_write is not None:
            grad_write.copy_(grad_write_cur)
        if grad_U is not None:
            grad_U.copy_(grad_U_cur)
        if grad_decode is not None:
            grad_decode.copy_(grad_decode_cur)

        return grad_U, grad_retain, grad_write, grad_decode


class ParallelHistorySlotScanFunction(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        U_chunk: torch.Tensor,
        retain_chunk: torch.Tensor,
        write_chunk: torch.Tensor,
        slot_embed: torch.Tensor,
        readout: torch.Tensor | None,
        aux_dim: int,
        block_chunks: int,
    ) -> torch.Tensor:
        bsz, nheads, nchunks, chunk_size, value_dim = U_chunk.shape
        nslots = retain_chunk.shape[-1]
        state = U_chunk.new_zeros((bsz, nheads, nslots, aux_dim))
        nblocks = math.ceil(nchunks / block_chunks)
        block_start_states = U_chunk.new_empty((bsz, nheads, nblocks, nslots, aux_dim))
        if readout is not None or aux_dim == 1:
            out_chunk = U_chunk.new_empty((bsz, nheads, nchunks, chunk_size, nslots))
        else:
            out_chunk = U_chunk.new_empty((bsz, nheads, nchunks, chunk_size, nslots, aux_dim))

        for block_idx, block_start in enumerate(range(0, nchunks, block_chunks)):
            block_end = min(block_start + block_chunks, nchunks)
            block_start_states[:, :, block_idx].copy_(state)
            out_cur, state, _ = _aux_block_forward_impl(
                state,
                retain_chunk[:, :, block_start:block_end],
                write_chunk[:, :, block_start:block_end],
                U_chunk[:, :, block_start:block_end],
                slot_embed,
                readout,
            )
            out_chunk[:, :, block_start:block_end].copy_(out_cur)

        tensors = [U_chunk, retain_chunk, write_chunk, slot_embed, block_start_states]
        ctx.has_readout = readout is not None
        if readout is not None:
            tensors.append(readout)
        ctx.save_for_backward(*tensors)
        ctx.aux_dim = int(aux_dim)
        ctx.block_chunks = int(block_chunks)
        return out_chunk

    @staticmethod
    def backward(ctx, grad_out_chunk: torch.Tensor):
        saved = ctx.saved_tensors
        if ctx.has_readout:
            U_chunk, retain_chunk, write_chunk, slot_embed, block_start_states, readout = saved
        else:
            U_chunk, retain_chunk, write_chunk, slot_embed, block_start_states = saved
            readout = None
        bsz, nheads, nchunks, _, value_dim = U_chunk.shape
        nslots = retain_chunk.shape[-1]
        aux_dim = ctx.aux_dim
        block_chunks = ctx.block_chunks
        nblocks = block_start_states.shape[2]

        grad_U = torch.zeros_like(U_chunk) if U_chunk.requires_grad else None
        grad_retain = torch.zeros_like(retain_chunk) if retain_chunk.requires_grad else None
        grad_write = torch.zeros_like(write_chunk) if write_chunk.requires_grad else None
        grad_slot_embed = torch.zeros_like(slot_embed) if slot_embed.requires_grad else None
        grad_readout = torch.zeros_like(readout) if readout is not None and readout.requires_grad else None
        grad_state = U_chunk.new_zeros((bsz, nheads, nslots, aux_dim))

        for block_idx in range(nblocks - 1, -1, -1):
            block_start = block_idx * block_chunks
            block_end = min(block_start + block_chunks, nchunks)
            _, _, start_states = _aux_block_forward_impl(
                block_start_states[:, :, block_idx],
                retain_chunk[:, :, block_start:block_end],
                write_chunk[:, :, block_start:block_end],
                U_chunk[:, :, block_start:block_end],
                slot_embed,
                readout,
            )
            grad_state, grad_retain_cur, grad_write_cur, grad_U_cur, grad_slot_embed_cur, grad_readout_cur, _ = _aux_chunk_backward_batch_manual(
                state0=start_states,
                retain_chunk=retain_chunk[:, :, block_start:block_end],
                write_chunk=write_chunk[:, :, block_start:block_end],
                U_chunk=U_chunk[:, :, block_start:block_end],
                slot_embed=slot_embed,
                readout=readout,
                grad_out=grad_out_chunk[:, :, block_start:block_end],
                grad_state1=grad_state,
            )
            if grad_retain is not None:
                grad_retain[:, :, block_start:block_end].copy_(grad_retain_cur)
            if grad_write is not None:
                grad_write[:, :, block_start:block_end].copy_(grad_write_cur)
            if grad_U is not None:
                grad_U[:, :, block_start:block_end].copy_(grad_U_cur)
            if grad_slot_embed is not None:
                grad_slot_embed = grad_slot_embed + grad_slot_embed_cur
            if grad_readout is not None and grad_readout_cur is not None:
                grad_readout = grad_readout + grad_readout_cur

        if readout is None:
            return grad_U, grad_retain, grad_write, grad_slot_embed, None, None, None
        return grad_U, grad_retain, grad_write, grad_slot_embed, grad_readout, None, None


def flare_autoregressive_separated_pytorch(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
    chunk_size: int | None = None,
    rmsnorm_read_contrib: bool = False,
) -> torch.Tensor:
    if rmsnorm_read_contrib:
        # TODO(vedantpu): reintroduce the optimized
        # `rmsnorm_read_contrib=True` path after the non-RMSNorm recurrence is
        # fully optimized and stable.
        raise NotImplementedError("rmsnorm_read_contrib=True is temporarily unsupported in separated.py.")

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
    if write.shape != (bsz, seqlen, nheads, nslots):
        raise ValueError("write must match retain shape.")
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

    out_chunk = FLAREAutoregressiveSeparatedFunction.apply(U_chunk, retain_chunk, write_chunk, decode_chunk)

    out = out_chunk.reshape(bsz, nheads, nchunks, chunk_size, value_dim).permute(0, 2, 3, 1, 4).reshape(
        bsz,
        padded_len,
        nheads,
        value_dim,
    )
    return out[:, :seqlen].to(U.dtype).contiguous()


def parallel_history_slot_scan(
    U: torch.Tensor,
    slot_embed: torch.Tensor,
    *,
    decay: float | torch.Tensor | None = None,
    retain: torch.Tensor | None = None,
    write: torch.Tensor | None = None,
    readout: torch.Tensor | None = None,
    chunk_size: int | None = None,
) -> torch.Tensor:
    if U.ndim != 4:
        raise ValueError(f"U must be [B,N,H,D]. Got {tuple(U.shape)}.")
    if slot_embed.ndim not in {3, 4}:
        raise ValueError(f"slot_embed must be [H,M,D] or [H,M,A,D]. Got {tuple(slot_embed.shape)}.")
    if (retain is None) != (write is None):
        raise ValueError("retain and write must be provided together.")

    bsz, seqlen, nheads, value_dim = U.shape
    if slot_embed.shape[0] != nheads or slot_embed.shape[-1] != value_dim:
        raise ValueError(
            "slot_embed must match [H,D] of U. "
            f"Got U={tuple(U.shape)}, slot_embed={tuple(slot_embed.shape)}."
        )

    if slot_embed.ndim == 3:
        nslots = slot_embed.shape[1]
        aux_dim = 1
    else:
        nslots = slot_embed.shape[1]
        aux_dim = slot_embed.shape[2]

    if retain is None:
        if decay is None:
            raise ValueError("Provide either decay or retain/write.")
        if torch.is_tensor(decay):
            if decay.ndim == 0:
                decay_view = decay.to(device=U.device, dtype=U.dtype).view(1, 1, 1, 1)
            elif decay.shape == (nheads,):
                decay_view = decay.to(device=U.device, dtype=U.dtype).view(1, 1, nheads, 1)
            else:
                raise ValueError(f"decay tensor must be scalar or [H]. Got {tuple(decay.shape)}.")
        else:
            decay_view = torch.full((1, 1, 1, 1), float(decay), device=U.device, dtype=U.dtype)
        retain = decay_view.expand(bsz, seqlen, nheads, nslots)
        write = (1.0 - decay_view).expand(bsz, seqlen, nheads, nslots)
    else:
        if retain.shape != (bsz, seqlen, nheads, nslots):
            raise ValueError(f"retain must be [B,N,H,M]. Got {tuple(retain.shape)}.")
        if write.shape != retain.shape:
            raise ValueError(f"write must match retain shape. Got retain={tuple(retain.shape)}, write={tuple(write.shape)}.")

    if readout is not None:
        if readout.ndim == 2:
            readout = readout.unsqueeze(-1)
        if readout.shape != (nheads, nslots, aux_dim):
            raise ValueError(
                f"readout must be [H,M,A]. Got readout={tuple(readout.shape)}, expected {(nheads, nslots, aux_dim)}."
            )

    if seqlen == 0:
        if readout is not None or aux_dim == 1:
            return U.new_empty((bsz, 0, nheads, nslots))
        return U.new_empty((bsz, 0, nheads, nslots, aux_dim))

    chunk_size = _resolve_separated_chunk_size(seqlen, nslots, aux_dim, chunk_size)
    nchunks = math.ceil(seqlen / chunk_size)
    padded_len = nchunks * chunk_size
    pad = padded_len - seqlen
    if pad > 0:
        U = _pad_sequence_dim(U, pad, fill_value=0.0)
        retain = _pad_sequence_dim(retain, pad, fill_value=1.0)
        write = _pad_sequence_dim(write, pad, fill_value=0.0)

    U_chunk = U.view(bsz, nchunks, chunk_size, nheads, value_dim).permute(0, 3, 1, 2, 4).contiguous()
    retain_chunk = retain.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 3, 1, 2, 4).contiguous()
    write_chunk = write.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 3, 1, 2, 4).contiguous()

    block_chunks = _resolve_chunk_block_size(nchunks, env_name="FLARE_SEPARATED_AUX_CHUNK_BLOCKS", default=2)
    out_chunk = ParallelHistorySlotScanFunction.apply(
        U_chunk,
        retain_chunk,
        write_chunk,
        slot_embed,
        readout,
        aux_dim,
        block_chunks,
    )

    if readout is not None:
        out = out_chunk.permute(0, 2, 3, 1, 4).reshape(bsz, padded_len, nheads, nslots)
        return out[:, :seqlen].contiguous()
    if aux_dim == 1:
        out = out_chunk.permute(0, 2, 3, 1, 4).reshape(bsz, padded_len, nheads, nslots)
        return out[:, :seqlen].contiguous()
    out = out_chunk.permute(0, 2, 3, 1, 4, 5).reshape(bsz, padded_len, nheads, nslots, aux_dim)
    return out[:, :seqlen].contiguous()

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


def _rms_normalize_last_dim(x: torch.Tensor, *, eps: float, scale_by_sqrt_dim: bool = False) -> torch.Tensor:
    rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    y = x / rms
    if scale_by_sqrt_dim:
        y = y / math.sqrt(x.shape[-1])
    return y


def _rms_normalize_tail_dims(x: torch.Tensor, *, eps: float) -> torch.Tensor:
    tail = x.shape[-2] * x.shape[-1]
    x_flat = x.reshape(*x.shape[:-2], tail)
    y_flat = _rms_normalize_last_dim(x_flat, eps=eps, scale_by_sqrt_dim=False)
    return y_flat.reshape_as(x)


def _rms_normalize_tail_dims_backward(x: torch.Tensor, grad_y: torch.Tensor, *, eps: float) -> torch.Tensor:
    tail = x.shape[-2] * x.shape[-1]
    x_flat = x.reshape(*x.shape[:-2], tail)
    grad_y_flat = grad_y.reshape_as(x_flat)
    mean_sq = x_flat.square().mean(dim=-1, keepdim=True)
    rms = torch.sqrt(mean_sq + eps)
    dot = (grad_y_flat * x_flat).sum(dim=-1, keepdim=True)
    grad_x_flat = grad_y_flat / rms - x_flat * (dot / (tail * rms.pow(3)))
    return grad_x_flat.reshape_as(x)


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


class ChunkwiseAffineStateScan(autograd.Function):
    @staticmethod
    def forward(ctx, A: torch.Tensor, B: torch.Tensor, initial_state: torch.Tensor):
        if A.ndim != 3 or B.ndim != 3 or initial_state.ndim != 2:
            raise ValueError(
                "ChunkwiseAffineStateScan expects A=[P,NC,S], B=[P,NC,S], initial_state=[P,S]. "
                f"Got A={tuple(A.shape)}, B={tuple(B.shape)}, initial_state={tuple(initial_state.shape)}."
            )
        if A.shape != B.shape:
            raise ValueError(f"A and B must have identical shapes. Got A={tuple(A.shape)}, B={tuple(B.shape)}.")
        if A.shape[0] != initial_state.shape[0] or A.shape[2] != initial_state.shape[1]:
            raise ValueError(
                "initial_state must match the flattened batch/state dimensions of A/B. "
                f"Got A={tuple(A.shape)}, initial_state={tuple(initial_state.shape)}."
            )

        p, nc, s = A.shape
        if nc == 0:
            chunk_start = A.new_empty((p, 0, s))
            ctx.save_for_backward(A, chunk_start)
            return chunk_start, initial_state

        inc_A, inc_B = _affine_prefix_scan_flat(A, B)
        excl_A = torch.cat([torch.ones_like(inc_A[:, :1]), inc_A[:, :-1]], dim=1)
        excl_B = torch.cat([torch.zeros_like(inc_B[:, :1]), inc_B[:, :-1]], dim=1)
        chunk_start = excl_A * initial_state.unsqueeze(1) + excl_B
        final_state = inc_A[:, -1] * initial_state + inc_B[:, -1]
        ctx.save_for_backward(A, chunk_start)
        return chunk_start, final_state

    @staticmethod
    def backward(ctx, grad_chunk_start: torch.Tensor, grad_final_state: torch.Tensor):
        A, chunk_start = ctx.saved_tensors
        if grad_chunk_start is None:
            grad_chunk_start = torch.zeros_like(chunk_start)
        if grad_final_state is None:
            grad_final_state = torch.zeros((A.shape[0], A.shape[2]), device=A.device, dtype=A.dtype)

        p, nc, s = A.shape
        grad_A = torch.zeros_like(A)
        grad_B = torch.zeros_like(A)
        grad_state_next = grad_final_state
        for idx in range(nc - 1, -1, -1):
            grad_A[:, idx] = grad_state_next * chunk_start[:, idx]
            grad_B[:, idx] = grad_state_next
            grad_state = grad_chunk_start[:, idx] + grad_state_next * A[:, idx]
            grad_state_next = grad_state
        grad_initial = grad_state_next
        return grad_A, grad_B, grad_initial


def chunkwise_affine_state_scan(
    A: torch.Tensor,
    B: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return ChunkwiseAffineStateScan.apply(A, B, initial_state)


def chunkwise_affine_state_scan_autograd(
    A: torch.Tensor,
    B: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if A.shape != B.shape:
        raise ValueError(f"A and B must have identical shapes. Got A={tuple(A.shape)}, B={tuple(B.shape)}.")
    if A.shape[0] != initial_state.shape[0] or A.shape[2] != initial_state.shape[1]:
        raise ValueError(
            "initial_state must match the flattened batch/state dimensions of A/B. "
            f"Got A={tuple(A.shape)}, initial_state={tuple(initial_state.shape)}."
        )
    p, nc, s = A.shape
    if nc == 0:
        chunk_start = A.new_empty((p, 0, s))
        return chunk_start, initial_state
    inc_A, inc_B = _affine_prefix_scan_flat(A, B)
    excl_A = torch.cat([torch.ones_like(inc_A[:, :1]), inc_A[:, :-1]], dim=1)
    excl_B = torch.cat([torch.zeros_like(inc_B[:, :1]), inc_B[:, :-1]], dim=1)
    chunk_start = excl_A * initial_state.unsqueeze(1) + excl_B
    final_state = inc_A[:, -1] * initial_state + inc_B[:, -1]
    return chunk_start, final_state


class LocalAffineStateMix(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        z_init: torch.Tensor,
        retain: torch.Tensor,
        write_value: torch.Tensor,
        decode_weights: torch.Tensor,
        rmsnorm_read_contrib: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if z_init.ndim != 6:
            raise ValueError(f"z_init must be [B,H,NC,M,G,Dg]. Got {tuple(z_init.shape)}")
        if retain.ndim != 6:
            raise ValueError(f"retain must be [B,H,NC,C,M,G]. Got {tuple(retain.shape)}")
        if write_value.ndim != 7:
            raise ValueError(f"write_value must be [B,H,NC,C,M,G,Dg]. Got {tuple(write_value.shape)}")
        if decode_weights.ndim != 5:
            raise ValueError(f"decode_weights must be [B,H,NC,C,M]. Got {tuple(decode_weights.shape)}")
        bsz, nheads, nchunks, nslots, ngroups, group_dim = z_init.shape
        c = retain.shape[3]
        if retain.shape != (bsz, nheads, nchunks, c, nslots, ngroups):
            raise ValueError("retain shape does not match z_init.")
        if write_value.shape != (bsz, nheads, nchunks, c, nslots, ngroups, group_dim):
            raise ValueError("write_value shape does not match z_init/retain.")
        if decode_weights.shape != (bsz, nheads, nchunks, c, nslots):
            raise ValueError("decode_weights shape does not match z_init/retain.")

        z_state = z_init
        outputs = []
        eps = torch.finfo(z_init.dtype).eps
        for offset in range(c):
            retain_step = retain[:, :, :, offset]
            write_step = write_value[:, :, :, offset]
            z_state = retain_step.unsqueeze(-1) * z_state + write_step
            contrib = decode_weights[:, :, :, offset].to(z_state.dtype).unsqueeze(-1).unsqueeze(-1) * z_state
            if rmsnorm_read_contrib:
                contrib = _rms_normalize_tail_dims(contrib, eps=eps)
            y_step = contrib.sum(dim=3)
            outputs.append(y_step)
        chunk_out = torch.stack(outputs, dim=3)
        ctx.save_for_backward(z_init, retain, write_value, decode_weights)
        ctx.rmsnorm_read_contrib = bool(rmsnorm_read_contrib)
        return z_state, chunk_out

    @staticmethod
    def backward(ctx, grad_z_final: torch.Tensor, grad_chunk_out: torch.Tensor):
        z_init, retain, write_value, decode_weights = ctx.saved_tensors
        rmsnorm_read_contrib = ctx.rmsnorm_read_contrib
        if grad_z_final is None:
            grad_z_final = torch.zeros_like(z_init)
        if grad_chunk_out is None:
            grad_chunk_out = torch.zeros(
                (*decode_weights.shape[:4], z_init.shape[4], z_init.shape[5]),
                device=z_init.device,
                dtype=z_init.dtype,
            )
        bsz, nheads, nchunks, nslots, ngroups, group_dim = z_init.shape
        c = retain.shape[3]

        z_hist = [z_init]
        z_state = z_init
        eps = torch.finfo(z_init.dtype).eps
        for offset in range(c):
            z_state = retain[:, :, :, offset].unsqueeze(-1) * z_state + write_value[:, :, :, offset]
            z_hist.append(z_state)

        grad_retain = torch.empty_like(retain)
        grad_write_value = torch.empty_like(write_value)
        grad_decode = torch.empty_like(decode_weights)

        grad_z_next = grad_z_final
        for offset in range(c - 1, -1, -1):
            z_prev = z_hist[offset]
            z_curr = z_hist[offset + 1]
            decode_step = decode_weights[:, :, :, offset].to(z_curr.dtype)
            dy_step = grad_chunk_out[:, :, :, offset]
            contrib = decode_step.unsqueeze(-1).unsqueeze(-1) * z_curr
            grad_contrib = dy_step.unsqueeze(3).expand_as(contrib)
            if rmsnorm_read_contrib:
                grad_contrib = _rms_normalize_tail_dims_backward(contrib, grad_contrib, eps=eps)
            grad_decode[:, :, :, offset] = (grad_contrib * z_curr).sum(dim=(-1, -2)).to(grad_decode.dtype)
            grad_z_total = grad_z_next + decode_step.unsqueeze(-1).unsqueeze(-1) * grad_contrib
            grad_retain[:, :, :, offset] = (grad_z_total * z_prev).sum(dim=-1).to(grad_retain.dtype)
            grad_write_value[:, :, :, offset] = grad_z_total
            grad_z_next = retain[:, :, :, offset].unsqueeze(-1) * grad_z_total

        return grad_z_next, grad_retain, grad_write_value, grad_decode, None


def local_affine_state_mix(
    z_init: torch.Tensor,
    retain: torch.Tensor,
    write_value: torch.Tensor,
    decode_weights: torch.Tensor,
    rmsnorm_read_contrib: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return LocalAffineStateMix.apply(z_init, retain, write_value, decode_weights, rmsnorm_read_contrib)


def _local_affine_state_mix_autograd(
    z_init: torch.Tensor,
    retain: torch.Tensor,
    write_value: torch.Tensor,
    decode_weights: torch.Tensor,
    *,
    rmsnorm_read_contrib: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    z_state = z_init
    outputs = []
    eps = torch.finfo(z_init.dtype).eps
    c = retain.shape[3]
    for offset in range(c):
        z_state = retain[:, :, :, offset].unsqueeze(-1) * z_state + write_value[:, :, :, offset]
        contrib = decode_weights[:, :, :, offset].to(z_state.dtype).unsqueeze(-1).unsqueeze(-1) * z_state
        if rmsnorm_read_contrib:
            contrib = _rms_normalize_tail_dims(contrib, eps=eps)
        outputs.append(contrib.sum(dim=3))
    chunk_out = torch.stack(outputs, dim=3)
    return z_state, chunk_out


def flare_autoregressive_separated_pytorch(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
    chunk_size: int | None = None,
    rmsnorm_read_contrib: bool = False,
) -> torch.Tensor:
    if U.ndim != 5:
        raise ValueError(f"U must be [B,N,H,G,Dg]. Got {tuple(U.shape)}")
    if retain.ndim != 5:
        raise ValueError(f"retain must be [B,N,H,M,G]. Got {tuple(retain.shape)}")
    if write.ndim != 5:
        raise ValueError(f"write must be [B,N,H,M,G]. Got {tuple(write.shape)}")
    if decode_weights.ndim != 4:
        raise ValueError(f"decode_weights must be [B,N,H,M]. Got {tuple(decode_weights.shape)}")

    bsz, seqlen, nheads, ngroups, group_dim = U.shape
    if retain.shape[:3] != (bsz, seqlen, nheads) or write.shape[:3] != (bsz, seqlen, nheads):
        raise ValueError("retain/write must share [B,N,H] with U.")
    nslots = retain.shape[3]
    if write.shape != (bsz, seqlen, nheads, nslots, ngroups):
        raise ValueError("write shape must match retain shape.")
    if decode_weights.shape != (bsz, seqlen, nheads, nslots):
        raise ValueError("decode_weights shape must match [B,N,H,M].")

    chunk_size = _resolve_separated_chunk_size(seqlen, nslots, ngroups * group_dim, chunk_size)
    nchunks = math.ceil(seqlen / chunk_size) if seqlen > 0 else 0
    padded_len = nchunks * chunk_size
    pad = padded_len - seqlen

    if seqlen == 0:
        return U.new_empty((bsz, 0, nheads, ngroups * group_dim))

    if pad > 0:
        U = torch.cat([U, torch.zeros((bsz, pad, nheads, ngroups, group_dim), device=U.device, dtype=U.dtype)], dim=1)
        retain = torch.cat(
            [retain, torch.ones((bsz, pad, nheads, nslots, ngroups), device=retain.device, dtype=retain.dtype)],
            dim=1,
        )
        write = torch.cat([write, torch.zeros((bsz, pad, nheads, nslots, ngroups), device=write.device, dtype=write.dtype)], dim=1)
        decode_weights = torch.cat(
            [decode_weights, torch.zeros((bsz, pad, nheads, nslots), device=decode_weights.device, dtype=decode_weights.dtype)],
            dim=1,
        )

    U_chunk = U.view(bsz, nchunks, chunk_size, nheads, ngroups, group_dim).permute(0, 3, 1, 2, 4, 5).contiguous()
    retain_chunk = retain.view(bsz, nchunks, chunk_size, nheads, nslots, ngroups).permute(0, 3, 1, 2, 4, 5).contiguous()
    write_chunk = write.view(bsz, nchunks, chunk_size, nheads, nslots, ngroups).permute(0, 3, 1, 2, 4, 5).contiguous()
    decode_chunk = decode_weights.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 3, 1, 2, 4).contiguous()

    write_value = write_chunk.unsqueeze(-1) * U_chunk.unsqueeze(4)
    suffix_inclusive = torch.flip(torch.cumprod(torch.flip(retain_chunk, dims=(3,)), dim=3), dims=(3,))
    suffix_exclusive = torch.cat(
        [suffix_inclusive[:, :, :, 1:, :, :], torch.ones_like(suffix_inclusive[:, :, :, :1, :, :])],
        dim=3,
    )
    chunk_A = retain_chunk.prod(dim=3)
    chunk_B = torch.einsum("bhncmg,bhncmgd->bhnmgd", suffix_exclusive, write_value)

    state_size = nslots * ngroups * group_dim
    A_flat = chunk_A.unsqueeze(-1).expand(-1, -1, -1, -1, -1, group_dim).reshape(bsz * nheads, nchunks, state_size)
    B_flat = chunk_B.reshape(bsz * nheads, nchunks, state_size)
    initial_state = torch.zeros((bsz * nheads, state_size), device=U.device, dtype=U.dtype)
    chunk_start_flat, _ = chunkwise_affine_state_scan(A_flat, B_flat, initial_state)
    chunk_start = chunk_start_flat.reshape(bsz, nheads, nchunks, nslots, ngroups, group_dim)

    _, chunk_out = local_affine_state_mix(
        chunk_start,
        retain_chunk,
        write_value,
        decode_chunk,
        rmsnorm_read_contrib,
    )
    out = chunk_out.permute(0, 2, 3, 1, 4, 5).reshape(bsz, padded_len, nheads, ngroups * group_dim)
    return out[:, :seqlen].contiguous()

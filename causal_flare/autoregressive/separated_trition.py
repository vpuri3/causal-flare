import math

import torch
import triton
import triton.language as tl


def _pad_sequence_dim(x: torch.Tensor, padded_len: int, fill_value: float) -> torch.Tensor:
    seqlen = x.shape[1]
    if seqlen == padded_len:
        return x
    out = torch.empty((x.shape[0], padded_len, *x.shape[2:]), device=x.device, dtype=x.dtype)
    out[:, :seqlen].copy_(x)
    if padded_len > seqlen:
        out[:, seqlen:].fill_(fill_value)
    return out


def _compute_aux_score_chunked(
    *,
    U: torch.Tensor,
    slot_embed: torch.Tensor,
    nchunks: int,
    chunk_size: int,
) -> torch.Tensor:
    bsz, padded_len, nheads, value_dim = U.shape
    nslots = slot_embed.shape[1]
    aux_dim = slot_embed.shape[2]
    flat_batch = bsz * nheads
    bn = flat_batch * nchunks
    block_m = _pick_block_m(nslots)
    block_a = _pick_block_a(aux_dim)
    block_d = _pick_block_d(value_dim)

    u_chunked = U.view(bsz, nchunks, chunk_size, nheads, value_dim).permute(0, 3, 1, 2, 4).reshape(
        bn, chunk_size, value_dim
    ).contiguous()
    score_chunked = torch.empty((bn, chunk_size, nslots, aux_dim), device=U.device, dtype=torch.float32)
    score_grid = (triton.cdiv(aux_dim, block_a), triton.cdiv(nslots, block_m), bn)
    _aux_score_kernel[score_grid](
        u_chunked,
        slot_embed,
        score_chunked,
        u_chunked.stride(0),
        u_chunked.stride(1),
        u_chunked.stride(2),
        slot_embed.stride(0),
        slot_embed.stride(1),
        slot_embed.stride(2),
        slot_embed.stride(3),
        score_chunked.stride(0),
        score_chunked.stride(1),
        score_chunked.stride(2),
        score_chunked.stride(3),
        NCHUNKS=nchunks,
        NHEADS=nheads,
        CHUNK_SIZE=chunk_size,
        M=nslots,
        A=aux_dim,
        D=value_dim,
        BLOCK_M=block_m,
        BLOCK_A=block_a,
        BLOCK_D=block_d,
        num_warps=4 if block_d <= 64 else 8,
        num_stages=2,
    )
    return score_chunked


def _exclusive_chunk_states_triton(state0: torch.Tensor, inclusive_states: torch.Tensor) -> torch.Tensor:
    if inclusive_states.shape[1] == 0:
        return inclusive_states
    exclusive = torch.empty_like(inclusive_states)
    exclusive[:, 0] = state0
    if inclusive_states.shape[1] > 1:
        exclusive[:, 1:] = inclusive_states[:, :-1]
    return exclusive


def _chunk_inclusive_states_triton(
    state0: torch.Tensor,
    retain_chunk: torch.Tensor,
    write_value_chunk: torch.Tensor,
) -> torch.Tensor:
    state = state0
    inclusive = torch.empty_like(write_value_chunk)
    for t in range(retain_chunk.shape[1]):
        state = retain_chunk[:, t].unsqueeze(-1) * state + write_value_chunk[:, t]
        inclusive[:, t] = state
    return inclusive


def _chunk_reverse_scan_triton(
    retain_acc: torch.Tensor,
    grad_from_output: torch.Tensor,
    grad_state1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_state_post = torch.empty_like(grad_from_output)
    carry = grad_state1
    for t in range(retain_acc.shape[1] - 1, -1, -1):
        carry = carry + grad_from_output[:, t]
        grad_state_post[:, t] = carry
        if t > 0:
            carry = carry * retain_acc[:, t].unsqueeze(-1)
    grad_state0 = retain_acc[:, 0].unsqueeze(-1) * grad_state_post[:, 0]
    return grad_state_post, grad_state0


def _resolve_separated_trition_chunk_size(N: int, M: int, D: int, chunk_size: int | None) -> int:
    if chunk_size is not None:
        return int(chunk_size)
    if D <= 64 and M <= 64 and N >= 1024:
        return 32
    return 32


def _require_supported_chunk_size(chunk_size: int) -> int:
    if chunk_size not in (16, 32):
        raise ValueError(
            "separated_trition forward currently supports chunk_size in {16, 32}. "
            f"Got chunk_size={chunk_size}."
        )
    return int(chunk_size)


def _pick_block_m(M: int) -> int:
    if M <= 16:
        return 16
    if M <= 32:
        return 32
    if M <= 64:
        return 64
    raise ValueError(f"separated_trition forward currently supports M <= 64. Got M={M}.")


def _pick_block_d(D: int) -> int:
    if D <= 32:
        return 32
    if D <= 64:
        return 64
    if D <= 128:
        return 128
    if D <= 256:
        return 256
    raise ValueError(f"separated_trition forward currently supports D <= 256. Got D={D}.")


def _pick_block_t(chunk_size: int) -> int:
    if chunk_size <= 16:
        return 16
    if chunk_size <= 32:
        return 32
    if chunk_size <= 64:
        return 32
    if chunk_size <= 128:
        return 32
    if chunk_size <= 256:
        return 32
    raise ValueError(f"separated_trition forward currently supports chunk_size <= 256. Got chunk_size={chunk_size}.")


def _pick_block_a(aux_dim: int) -> int:
    if aux_dim <= 1:
        return 1
    if aux_dim <= 2:
        return 2
    if aux_dim <= 4:
        return 4
    if aux_dim <= 8:
        return 8
    if aux_dim <= 16:
        return 16
    raise ValueError(f"aux Triton path currently supports aux_dim <= 16. Got aux_dim={aux_dim}.")


@triton.jit
def _aux_score_kernel(
    u_ptr,
    slot_ptr,
    score_ptr,
    stride_u_bn,
    stride_u_t,
    stride_u_d,
    stride_slot_h,
    stride_slot_m,
    stride_slot_a,
    stride_slot_d,
    stride_sc_bn,
    stride_sc_t,
    stride_sc_m,
    stride_sc_a,
    NCHUNKS: tl.constexpr,
    NHEADS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    A: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_A: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bn = tl.program_id(2)

    bh = pid_bn // NCHUNKS
    head = bh % NHEADS

    offs_a = pid_a * BLOCK_A + tl.arange(0, BLOCK_A)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_a = offs_a < A
    mask_m = offs_m < M

    for t in tl.static_range(0, CHUNK_SIZE):
        acc = tl.zeros([BLOCK_M, BLOCK_A], dtype=tl.float32)
        for d0 in tl.static_range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D
            u_ptrs = u_ptr + pid_bn * stride_u_bn + t * stride_u_t + offs_d * stride_u_d
            slot_ptrs = (
                slot_ptr
                + head * stride_slot_h
                + offs_m[:, None, None] * stride_slot_m
                + offs_a[None, :, None] * stride_slot_a
                + offs_d[None, None, :] * stride_slot_d
            )
            u_row = tl.load(u_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            slot_tile = tl.load(slot_ptrs, mask=mask_m[:, None, None] & mask_a[None, :, None] & mask_d[None, None, :], other=0.0).to(tl.float32)
            acc += tl.sum(slot_tile * u_row[None, None, :], axis=2)

        score_ptrs = (
            score_ptr
            + pid_bn * stride_sc_bn
            + t * stride_sc_t
            + offs_m[:, None] * stride_sc_m
            + offs_a[None, :] * stride_sc_a
        )
        tl.store(score_ptrs, acc, mask=mask_m[:, None] & mask_a[None, :])


@triton.jit
def _separated_prepare_kernel(
    retain_ptr,
    write_ptr,
    decode_ptr,
    a_ptr,
    b_ptr,
    chunk_a_ptr,
    stride_r_bn,
    stride_r_t,
    stride_r_m,
    stride_w_bn,
    stride_w_t,
    stride_w_m,
    stride_c_bn,
    stride_c_t,
    stride_c_m,
    stride_a_bn,
    stride_a_t,
    stride_a_m,
    stride_b_bn,
    stride_b_t,
    stride_b_m,
    stride_ca_bn,
    stride_ca_m,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    tiny = 1.1754943508222875e-38

    log_prefix = tl.zeros([BLOCK_M], dtype=tl.float32)
    last_prefix = tl.zeros([BLOCK_M], dtype=tl.float32)

    for t in tl.static_range(0, CHUNK_SIZE):
        retain_ptrs = retain_ptr + pid_bn * stride_r_bn + t * stride_r_t + offs_m * stride_r_m
        write_ptrs = write_ptr + pid_bn * stride_w_bn + t * stride_w_t + offs_m * stride_w_m
        decode_ptrs = decode_ptr + pid_bn * stride_c_bn + t * stride_c_t + offs_m * stride_c_m

        retain_row = tl.load(retain_ptrs, mask=mask_m, other=1.0).to(tl.float32)
        write_row = tl.load(write_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        decode_row = tl.load(decode_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        log_prefix += tl.log(tl.maximum(retain_row, tiny))
        prefix_row = tl.exp(log_prefix)
        prefix_safe = tl.maximum(prefix_row, tiny)

        a_row = decode_row * prefix_row
        b_row = write_row / prefix_safe

        a_ptrs = a_ptr + pid_bn * stride_a_bn + t * stride_a_t + offs_m * stride_a_m
        b_ptrs = b_ptr + pid_bn * stride_b_bn + t * stride_b_t + offs_m * stride_b_m
        tl.store(a_ptrs, a_row, mask=mask_m)
        tl.store(b_ptrs, b_row, mask=mask_m)
        last_prefix = prefix_row

    chunk_a_ptrs = chunk_a_ptr + pid_bn * stride_ca_bn + offs_m * stride_ca_m
    tl.store(chunk_a_ptrs, last_prefix, mask=mask_m)


@triton.jit
def _aux_chunk_a_kernel(
    retain_ptr,
    chunk_a_ptr,
    stride_r_bn,
    stride_r_t,
    stride_r_m,
    stride_ca_bn,
    stride_ca_m,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M

    prod = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    for t in tl.static_range(0, CHUNK_SIZE):
        retain_ptrs = retain_ptr + pid_bn * stride_r_bn + t * stride_r_t + offs_m * stride_r_m
        retain_row = tl.load(retain_ptrs, mask=mask_m, other=1.0).to(tl.float32)
        prod *= retain_row

    chunk_a_ptrs = chunk_a_ptr + pid_bn * stride_ca_bn + offs_m * stride_ca_m
    tl.store(chunk_a_ptrs, prod, mask=mask_m)


@triton.jit
def _aux_chunk_summary_kernel(
    score_ptr,
    retain_ptr,
    write_ptr,
    chunk_b_ptr,
    stride_sc_bn,
    stride_sc_t,
    stride_sc_m,
    stride_sc_a,
    stride_r_bn,
    stride_r_t,
    stride_r_m,
    stride_w_bn,
    stride_w_t,
    stride_w_m,
    stride_cb_bh,
    stride_cb_bn,
    stride_cb_m,
    stride_cb_a,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    A: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_A: tl.constexpr,
):
    pid_a = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bn = tl.program_id(2)

    offs_a = pid_a * BLOCK_A + tl.arange(0, BLOCK_A)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_a = offs_a < A
    mask_m = offs_m < M

    state = tl.zeros([BLOCK_M, BLOCK_A], dtype=tl.float32)

    for t in tl.static_range(0, CHUNK_SIZE):
        retain_ptrs = retain_ptr + pid_bn * stride_r_bn + t * stride_r_t + offs_m * stride_r_m
        write_ptrs = write_ptr + pid_bn * stride_w_bn + t * stride_w_t + offs_m * stride_w_m
        score_ptrs = (
            score_ptr
            + pid_bn * stride_sc_bn
            + t * stride_sc_t
            + offs_m[:, None] * stride_sc_m
            + offs_a[None, :] * stride_sc_a
        )

        retain_row = tl.load(retain_ptrs, mask=mask_m, other=1.0).to(tl.float32)
        write_row = tl.load(write_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        score_row = tl.load(score_ptrs, mask=mask_m[:, None] & mask_a[None, :], other=0.0).to(tl.float32)
        state = retain_row[:, None] * state + write_row[:, None] * score_row

    chunk_b_ptrs = (
        chunk_b_ptr
        + pid_bn * stride_cb_bn
        + offs_m[:, None] * stride_cb_m
        + offs_a[None, :] * stride_cb_a
    )
    tl.store(chunk_b_ptrs, state, mask=mask_m[:, None] & mask_a[None, :])


@triton.jit
def _aux_output_kernel(
    score_ptr,
    retain_ptr,
    write_ptr,
    state0_ptr,
    readout_ptr,
    out_ptr,
    stride_sc_bn,
    stride_sc_t,
    stride_sc_m,
    stride_sc_a,
    stride_r_bn,
    stride_r_t,
    stride_r_m,
    stride_w_bn,
    stride_w_t,
    stride_w_m,
    stride_s0_bn,
    stride_s0_m,
    stride_s0_a,
    stride_ro_bh,
    stride_ro_m,
    stride_ro_a,
    stride_out_bn,
    stride_out_t,
    stride_out_m,
    NCHUNKS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    A: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_A: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)

    bh = pid_bn // NCHUNKS
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_a = tl.arange(0, BLOCK_A)
    mask_m = offs_m < M
    mask_a = offs_a < A

    state_ptrs = state0_ptr + pid_bn * stride_s0_bn + offs_m[:, None] * stride_s0_m + offs_a[None, :] * stride_s0_a
    readout_ptrs = readout_ptr + bh * stride_ro_bh + offs_m[:, None] * stride_ro_m + offs_a[None, :] * stride_ro_a
    state = tl.load(state_ptrs, mask=mask_m[:, None] & mask_a[None, :], other=0.0).to(tl.float32)
    readout = tl.load(readout_ptrs, mask=mask_m[:, None] & mask_a[None, :], other=0.0).to(tl.float32)

    for t in tl.static_range(0, CHUNK_SIZE):
        out_ptrs = out_ptr + pid_bn * stride_out_bn + t * stride_out_t + offs_m * stride_out_m
        out_row = tl.sum(state * readout, axis=1)
        tl.store(out_ptrs, out_row.to(tl.bfloat16), mask=mask_m)

        retain_ptrs = retain_ptr + pid_bn * stride_r_bn + t * stride_r_t + offs_m * stride_r_m
        write_ptrs = write_ptr + pid_bn * stride_w_bn + t * stride_w_t + offs_m * stride_w_m
        score_ptrs = (
            score_ptr
            + pid_bn * stride_sc_bn
            + t * stride_sc_t
            + offs_m[:, None] * stride_sc_m
            + offs_a[None, :] * stride_sc_a
        )
        retain_row = tl.load(retain_ptrs, mask=mask_m, other=1.0).to(tl.float32)
        write_row = tl.load(write_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        score_row = tl.load(score_ptrs, mask=mask_m[:, None] & mask_a[None, :], other=0.0).to(tl.float32)
        state = retain_row[:, None] * state + write_row[:, None] * score_row


@triton.jit
def _separated_output_kernel_tiled(
    u_ptr,
    a_ptr,
    b_ptr,
    state0_ptr,
    y_ptr,
    stride_u_bn,
    stride_u_t,
    stride_u_d,
    stride_a_bn,
    stride_a_t,
    stride_a_m,
    stride_b_bn,
    stride_b_t,
    stride_b_m,
    stride_s0_bn,
    stride_s0_m,
    stride_s0_d,
    stride_y_bn,
    stride_y_t,
    stride_y_d,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_bn = tl.program_id(2)

    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < CHUNK_SIZE
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    u_ptrs = u_ptr + pid_bn * stride_u_bn + offs_t[:, None] * stride_u_t + offs_d[None, :] * stride_u_d
    a_mask = mask_t[:, None]

    y_tile = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)

    for m0 in tl.static_range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        a_ptrs = a_ptr + pid_bn * stride_a_bn + offs_t[:, None] * stride_a_t + offs_m[None, :] * stride_a_m
        b_ptrs = b_ptr + pid_bn * stride_b_bn + offs_t[:, None] * stride_b_t + offs_m[None, :] * stride_b_m
        state0_ptrs = state0_ptr + pid_bn * stride_s0_bn + offs_m[:, None] * stride_s0_m + offs_d[None, :] * stride_s0_d

        a_tile = tl.load(a_ptrs, mask=a_mask & mask_m[None, :], other=0.0)
        state0_tile = tl.load(state0_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0)

        y_tile += tl.dot(a_tile, state0_tile, out_dtype=tl.float32)

        for s0 in tl.static_range(0, CHUNK_SIZE, BLOCK_T):
            offs_s = s0 + tl.arange(0, BLOCK_T)
            mask_s = offs_s < CHUNK_SIZE
            b_src_ptrs = b_ptr + pid_bn * stride_b_bn + offs_s[:, None] * stride_b_t + offs_m[None, :] * stride_b_m
            u_src_ptrs = u_ptr + pid_bn * stride_u_bn + offs_s[:, None] * stride_u_t + offs_d[None, :] * stride_u_d

            b_src = tl.load(b_src_ptrs, mask=mask_s[:, None] & mask_m[None, :], other=0.0)
            u_src = tl.load(u_src_ptrs, mask=mask_s[:, None] & mask_d[None, :], other=0.0)

            local = tl.dot(a_tile, tl.trans(b_src), out_dtype=tl.float32)
            causal = offs_t[:, None] >= offs_s[None, :]
            local = tl.where(causal & mask_t[:, None] & mask_s[None, :], local, 0.0)
            y_tile += tl.dot(local, u_src.to(tl.float32), out_dtype=tl.float32)

    y_ptrs = y_ptr + pid_bn * stride_y_bn + offs_t[:, None] * stride_y_t + offs_d[None, :] * stride_y_d
    tl.store(y_ptrs, y_tile.to(tl.bfloat16), mask=mask_t[:, None] & mask_d[None, :])


@triton.jit
def _main_chunk_summary_direct_kernel(
    u_ptr,
    retain_ptr,
    write_ptr,
    chunk_b_ptr,
    stride_u_bn,
    stride_u_t,
    stride_u_d,
    stride_r_bn,
    stride_r_t,
    stride_r_m,
    stride_w_bn,
    stride_w_t,
    stride_w_m,
    stride_cb_bh,
    stride_cb_bn,
    stride_cb_m,
    stride_cb_d,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bn = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_d = offs_d < D
    mask_m = offs_m < M

    state = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for t in tl.static_range(0, CHUNK_SIZE):
        u_ptrs = u_ptr + pid_bn * stride_u_bn + t * stride_u_t + offs_d * stride_u_d
        retain_ptrs = retain_ptr + pid_bn * stride_r_bn + t * stride_r_t + offs_m * stride_r_m
        write_ptrs = write_ptr + pid_bn * stride_w_bn + t * stride_w_t + offs_m * stride_w_m

        u_row = tl.load(u_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        retain_row = tl.load(retain_ptrs, mask=mask_m, other=1.0).to(tl.float32)
        write_row = tl.load(write_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        state = retain_row[:, None] * state + write_row[:, None] * u_row[None, :]

    chunk_b_ptrs = (
        chunk_b_ptr
        + pid_bn * stride_cb_bn
        + offs_m[:, None] * stride_cb_m
        + offs_d[None, :] * stride_cb_d
    )
    tl.store(chunk_b_ptrs, state, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _main_output_direct_kernel(
    u_ptr,
    retain_ptr,
    write_ptr,
    decode_ptr,
    state0_ptr,
    y_ptr,
    stride_u_bn,
    stride_u_t,
    stride_u_d,
    stride_r_bn,
    stride_r_t,
    stride_r_m,
    stride_w_bn,
    stride_w_t,
    stride_w_m,
    stride_c_bn,
    stride_c_t,
    stride_c_m,
    stride_s0_bn,
    stride_s0_m,
    stride_s0_d,
    stride_y_bn,
    stride_y_t,
    stride_y_d,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_bn = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    for m0 in tl.static_range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        state0_ptrs = state0_ptr + pid_bn * stride_s0_bn + offs_m[:, None] * stride_s0_m + offs_d[None, :] * stride_s0_d
        state = tl.load(state0_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        for t in tl.static_range(0, CHUNK_SIZE):
            u_ptrs = u_ptr + pid_bn * stride_u_bn + t * stride_u_t + offs_d * stride_u_d
            retain_ptrs = retain_ptr + pid_bn * stride_r_bn + t * stride_r_t + offs_m * stride_r_m
            write_ptrs = write_ptr + pid_bn * stride_w_bn + t * stride_w_t + offs_m * stride_w_m
            decode_ptrs = decode_ptr + pid_bn * stride_c_bn + t * stride_c_t + offs_m * stride_c_m

            u_row = tl.load(u_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            retain_row = tl.load(retain_ptrs, mask=mask_m, other=1.0).to(tl.float32)
            write_row = tl.load(write_ptrs, mask=mask_m, other=0.0).to(tl.float32)
            decode_row = tl.load(decode_ptrs, mask=mask_m, other=0.0).to(tl.float32)

            state = retain_row[:, None] * state + write_row[:, None] * u_row[None, :]
            y_ptrs = y_ptr + pid_bn * stride_y_bn + t * stride_y_t + offs_d * stride_y_d
            prev = tl.load(y_ptrs, mask=mask_d, other=0.0).to(tl.float32)
            contrib = tl.sum(decode_row[:, None] * state, axis=0)
            tl.store(y_ptrs, (prev + contrib).to(tl.bfloat16), mask=mask_d)


@triton.jit
def _separated_prefix_state_kernel(
    chunk_a_ptr,
    chunk_b_ptr,
    state0_ptr,
    prefix_state_ptr,
    final_state_ptr,
    stride_ca_bh,
    stride_ca_nc,
    stride_ca_m,
    stride_cb_bh,
    stride_cb_nc,
    stride_cb_m,
    stride_cb_d,
    stride_s0_bh,
    stride_s0_m,
    stride_s0_d,
    stride_ps_bh,
    stride_ps_nc,
    stride_ps_m,
    stride_ps_d,
    stride_fs_bh,
    stride_fs_m,
    stride_fs_d,
    NUM_CHUNKS: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bh = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_d = offs_d < D
    mask_m = offs_m < M

    state_ptrs = state0_ptr + pid_bh * stride_s0_bh + offs_m[:, None] * stride_s0_m + offs_d[None, :] * stride_s0_d
    state = tl.load(state_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    for nc in tl.static_range(0, NUM_CHUNKS):
        prefix_ptrs = (
            prefix_state_ptr
            + pid_bh * stride_ps_bh
            + nc * stride_ps_nc
            + offs_m[:, None] * stride_ps_m
            + offs_d[None, :] * stride_ps_d
        )
        tl.store(prefix_ptrs, state, mask=mask_m[:, None] & mask_d[None, :])

        chunk_a_ptrs = chunk_a_ptr + pid_bh * stride_ca_bh + nc * stride_ca_nc + offs_m * stride_ca_m
        chunk_b_ptrs = (
            chunk_b_ptr
            + pid_bh * stride_cb_bh
            + nc * stride_cb_nc
            + offs_m[:, None] * stride_cb_m
            + offs_d[None, :] * stride_cb_d
        )
        chunk_a = tl.load(chunk_a_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        chunk_b = tl.load(chunk_b_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        state = chunk_a[:, None] * state + chunk_b

    final_ptrs = final_state_ptr + pid_bh * stride_fs_bh + offs_m[:, None] * stride_fs_m + offs_d[None, :] * stride_fs_d
    tl.store(final_ptrs, state, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _separated_prefix_state_backward_kernel(
    chunk_a_ptr,
    prefix_state_ptr,
    grad_prefix_out_ptr,
    grad_final_ptr,
    grad_chunk_a_ptr,
    grad_chunk_b_ptr,
    grad_initial_ptr,
    stride_ca_bh,
    stride_ca_nc,
    stride_ca_m,
    stride_ps_bh,
    stride_ps_nc,
    stride_ps_m,
    stride_ps_d,
    stride_gpo_bh,
    stride_gpo_nc,
    stride_gpo_m,
    stride_gpo_d,
    stride_gf_bh,
    stride_gf_m,
    stride_gf_d,
    stride_gca_bh,
    stride_gca_nc,
    stride_gca_m,
    stride_gcb_bh,
    stride_gcb_nc,
    stride_gcb_m,
    stride_gcb_d,
    stride_gi_bh,
    stride_gi_m,
    stride_gi_d,
    NUM_CHUNKS: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bh = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_d = offs_d < D
    mask_m = offs_m < M

    grad_state_ptrs = grad_final_ptr + pid_bh * stride_gf_bh + offs_m[:, None] * stride_gf_m + offs_d[None, :] * stride_gf_d
    grad_state = tl.load(grad_state_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    for rev_idx in tl.static_range(0, NUM_CHUNKS):
        nc = NUM_CHUNKS - 1 - rev_idx

        prefix_ptrs = (
            prefix_state_ptr
            + pid_bh * stride_ps_bh
            + nc * stride_ps_nc
            + offs_m[:, None] * stride_ps_m
            + offs_d[None, :] * stride_ps_d
        )
        prefix_state = tl.load(prefix_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

        chunk_a_ptrs = chunk_a_ptr + pid_bh * stride_ca_bh + nc * stride_ca_nc + offs_m * stride_ca_m
        chunk_a = tl.load(chunk_a_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        grad_chunk_b_ptrs = (
            grad_chunk_b_ptr
            + pid_bh * stride_gcb_bh
            + nc * stride_gcb_nc
            + offs_m[:, None] * stride_gcb_m
            + offs_d[None, :] * stride_gcb_d
        )
        tl.store(grad_chunk_b_ptrs, grad_state, mask=mask_m[:, None] & mask_d[None, :])

        grad_chunk_a_partial = tl.sum(grad_state * prefix_state, axis=1)
        grad_chunk_a_ptrs = grad_chunk_a_ptr + pid_bh * stride_gca_bh + nc * stride_gca_nc + offs_m * stride_gca_m
        tl.atomic_add(grad_chunk_a_ptrs, grad_chunk_a_partial, mask=mask_m)

        grad_prefix_out_ptrs = (
            grad_prefix_out_ptr
            + pid_bh * stride_gpo_bh
            + nc * stride_gpo_nc
            + offs_m[:, None] * stride_gpo_m
            + offs_d[None, :] * stride_gpo_d
        )
        grad_prefix_out = tl.load(grad_prefix_out_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        grad_state = grad_prefix_out + chunk_a[:, None] * grad_state

    grad_initial_ptrs = grad_initial_ptr + pid_bh * stride_gi_bh + offs_m[:, None] * stride_gi_m + offs_d[None, :] * stride_gi_d
    tl.store(grad_initial_ptrs, grad_state, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _main_grad_state0_kernel(
    a_ptr,
    grad_out_ptr,
    grad_state0_ptr,
    stride_a_bn,
    stride_a_t,
    stride_a_m,
    stride_go_bn,
    stride_go_t,
    stride_go_d,
    stride_gs_bn,
    stride_gs_m,
    stride_gs_d,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bn = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_d = offs_d < D
    mask_m = offs_m < M

    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for t in tl.static_range(0, CHUNK_SIZE):
        a_ptrs = a_ptr + pid_bn * stride_a_bn + t * stride_a_t + offs_m * stride_a_m
        go_ptrs = grad_out_ptr + pid_bn * stride_go_bn + t * stride_go_t + offs_d * stride_go_d
        a_row = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        go_row = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        acc += a_row[:, None] * go_row[None, :]

    out_ptrs = grad_state0_ptr + pid_bn * stride_gs_bn + offs_m[:, None] * stride_gs_m + offs_d[None, :] * stride_gs_d
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _main_grad_a_direct_kernel(
    grad_out_ptr,
    state0_ptr,
    grad_a_ptr,
    stride_go_bn,
    stride_go_t,
    stride_go_d,
    stride_s0_bn,
    stride_s0_m,
    stride_s0_d,
    stride_ga_bn,
    stride_ga_t,
    stride_ga_m,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bn = tl.program_id(2)

    offs_d = tl.arange(0, BLOCK_D)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_d = offs_d < D
    mask_m = offs_m < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for d0 in tl.static_range(0, D, BLOCK_D):
        cur_d = d0 + offs_d
        cur_mask_d = cur_d < D
        go_ptrs = grad_out_ptr + pid_bn * stride_go_bn + pid_t * stride_go_t + cur_d * stride_go_d
        s0_ptrs = state0_ptr + pid_bn * stride_s0_bn + offs_m[:, None] * stride_s0_m + cur_d[None, :] * stride_s0_d
        go_row = tl.load(go_ptrs, mask=cur_mask_d, other=0.0).to(tl.float32)
        s0_tile = tl.load(s0_ptrs, mask=mask_m[:, None] & cur_mask_d[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(s0_tile * go_row[None, :], axis=1)

    out_ptrs = grad_a_ptr + pid_bn * stride_ga_bn + pid_t * stride_ga_t + offs_m * stride_ga_m
    tl.store(out_ptrs, acc, mask=mask_m)


@triton.jit
def _main_k_gradk_kernel(
    a_ptr,
    b_ptr,
    grad_out_ptr,
    u_ptr,
    k_ptr,
    grad_k_ptr,
    stride_a_bn,
    stride_a_t,
    stride_a_m,
    stride_b_bn,
    stride_b_t,
    stride_b_m,
    stride_go_bn,
    stride_go_t,
    stride_go_d,
    stride_u_bn,
    stride_u_t,
    stride_u_d,
    stride_k_bn,
    stride_k_t,
    stride_k_s,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_bn = tl.program_id(2)

    if pid_s > pid_t:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    acc_k = tl.zeros([], dtype=tl.float32)
    acc_gk = tl.zeros([], dtype=tl.float32)

    for m0 in tl.static_range(0, M, BLOCK_M):
        cur_m = m0 + offs_m
        mask_m = cur_m < M
        a_ptrs = a_ptr + pid_bn * stride_a_bn + pid_t * stride_a_t + cur_m * stride_a_m
        b_ptrs = b_ptr + pid_bn * stride_b_bn + pid_s * stride_b_t + cur_m * stride_b_m
        a_row = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        b_row = tl.load(b_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        acc_k += tl.sum(a_row * b_row)

    for d0 in tl.static_range(0, D, BLOCK_D):
        cur_d = d0 + offs_d
        mask_d = cur_d < D
        go_ptrs = grad_out_ptr + pid_bn * stride_go_bn + pid_t * stride_go_t + cur_d * stride_go_d
        u_ptrs = u_ptr + pid_bn * stride_u_bn + pid_s * stride_u_t + cur_d * stride_u_d
        go_row = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        u_row = tl.load(u_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        acc_gk += tl.sum(go_row * u_row)

    k_out_ptr = k_ptr + pid_bn * stride_k_bn + pid_t * stride_k_t + pid_s * stride_k_s
    grad_k_out_ptr = grad_k_ptr + pid_bn * stride_k_bn + pid_t * stride_k_t + pid_s * stride_k_s
    tl.store(k_out_ptr, acc_k)
    tl.store(grad_k_out_ptr, acc_gk)


@triton.jit
def _main_grad_u_from_k_kernel(
    k_ptr,
    grad_out_ptr,
    grad_u_ptr,
    stride_k_bn,
    stride_k_t,
    stride_k_s,
    stride_go_bn,
    stride_go_t,
    stride_go_d,
    stride_gu_bn,
    stride_gu_t,
    stride_gu_d,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_bn = tl.program_id(2)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for s in tl.static_range(0, CHUNK_SIZE):
        k_ptrs = k_ptr + pid_bn * stride_k_bn + s * stride_k_t + pid_t * stride_k_s
        go_ptrs = grad_out_ptr + pid_bn * stride_go_bn + s * stride_go_t + offs_d * stride_go_d
        k_val = tl.load(k_ptrs).to(tl.float32)
        go_row = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        acc += k_val * go_row
    out_ptrs = grad_u_ptr + pid_bn * stride_gu_bn + pid_t * stride_gu_t + offs_d * stride_gu_d
    tl.store(out_ptrs, acc, mask=mask_d)


@triton.jit
def _main_grad_a_corr_kernel(
    grad_k_ptr,
    b_ptr,
    grad_a_ptr,
    stride_gk_bn,
    stride_gk_t,
    stride_gk_s,
    stride_b_bn,
    stride_b_t,
    stride_b_m,
    stride_ga_bn,
    stride_ga_t,
    stride_ga_m,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bn = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for s in tl.static_range(0, CHUNK_SIZE):
        gk_ptrs = grad_k_ptr + pid_bn * stride_gk_bn + pid_t * stride_gk_t + s * stride_gk_s
        b_ptrs = b_ptr + pid_bn * stride_b_bn + s * stride_b_t + offs_m * stride_b_m
        gk = tl.load(gk_ptrs).to(tl.float32)
        b_row = tl.load(b_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        acc += gk * b_row
    out_ptrs = grad_a_ptr + pid_bn * stride_ga_bn + pid_t * stride_ga_t + offs_m * stride_ga_m
    prev = tl.load(out_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    tl.store(out_ptrs, prev + acc, mask=mask_m)


@triton.jit
def _main_grad_b_kernel(
    grad_k_ptr,
    a_ptr,
    grad_b_ptr,
    stride_gk_bn,
    stride_gk_t,
    stride_gk_s,
    stride_a_bn,
    stride_a_t,
    stride_a_m,
    stride_gb_bn,
    stride_gb_t,
    stride_gb_m,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bn = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for t in tl.static_range(0, CHUNK_SIZE):
        gk_ptrs = grad_k_ptr + pid_bn * stride_gk_bn + t * stride_gk_t + pid_s * stride_gk_s
        a_ptrs = a_ptr + pid_bn * stride_a_bn + t * stride_a_t + offs_m * stride_a_m
        gk = tl.load(gk_ptrs).to(tl.float32)
        a_row = tl.load(a_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        acc += gk * a_row
    out_ptrs = grad_b_ptr + pid_bn * stride_gb_bn + pid_s * stride_gb_t + offs_m * stride_gb_m
    tl.store(out_ptrs, acc, mask=mask_m)


@triton.jit
def _main_t_kernel(
    b_ptr,
    u_ptr,
    t_ptr,
    stride_b_bn,
    stride_b_t,
    stride_b_m,
    stride_u_bn,
    stride_u_t,
    stride_u_d,
    stride_t_bn,
    stride_t_m,
    stride_t_d,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bn = tl.program_id(2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_d = offs_d < D
    mask_m = offs_m < M
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    for t in tl.static_range(0, CHUNK_SIZE):
        b_ptrs = b_ptr + pid_bn * stride_b_bn + t * stride_b_t + offs_m * stride_b_m
        u_ptrs = u_ptr + pid_bn * stride_u_bn + t * stride_u_t + offs_d * stride_u_d
        b_row = tl.load(b_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        u_row = tl.load(u_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        acc += b_row[:, None] * u_row[None, :]
    out_ptrs = t_ptr + pid_bn * stride_t_bn + offs_m[:, None] * stride_t_m + offs_d[None, :] * stride_t_d
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_d[None, :])


@triton.jit
def _main_grad_b_extra_kernel(
    u_ptr,
    grad_t_ptr,
    grad_b_ptr,
    stride_u_bn,
    stride_u_t,
    stride_u_d,
    stride_gt_bn,
    stride_gt_m,
    stride_gt_d,
    stride_gb_bn,
    stride_gb_t,
    stride_gb_m,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_bn = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    offs_d = tl.arange(0, BLOCK_D)
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for d0 in tl.static_range(0, D, BLOCK_D):
        cur_d = d0 + offs_d
        mask_d = cur_d < D
        u_ptrs = u_ptr + pid_bn * stride_u_bn + pid_t * stride_u_t + cur_d * stride_u_d
        gt_ptrs = grad_t_ptr + pid_bn * stride_gt_bn + offs_m[:, None] * stride_gt_m + cur_d[None, :] * stride_gt_d
        u_row = tl.load(u_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        gt_tile = tl.load(gt_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(gt_tile * u_row[None, :], axis=1)
    out_ptrs = grad_b_ptr + pid_bn * stride_gb_bn + pid_t * stride_gb_t + offs_m * stride_gb_m
    prev = tl.load(out_ptrs, mask=mask_m, other=0.0).to(tl.float32)
    tl.store(out_ptrs, prev + acc, mask=mask_m)


@triton.jit
def _main_grad_u_extra_kernel(
    b_ptr,
    grad_t_ptr,
    grad_u_ptr,
    stride_b_bn,
    stride_b_t,
    stride_b_m,
    stride_gt_bn,
    stride_gt_m,
    stride_gt_d,
    stride_gu_bn,
    stride_gu_t,
    stride_gu_d,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_d = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_bn = tl.program_id(2)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    offs_m = tl.arange(0, BLOCK_M)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for m0 in tl.static_range(0, M, BLOCK_M):
        cur_m = m0 + offs_m
        mask_m = cur_m < M
        b_ptrs = b_ptr + pid_bn * stride_b_bn + pid_t * stride_b_t + cur_m * stride_b_m
        gt_ptrs = grad_t_ptr + pid_bn * stride_gt_bn + cur_m[:, None] * stride_gt_m + offs_d[None, :] * stride_gt_d
        b_row = tl.load(b_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        gt_tile = tl.load(gt_ptrs, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        acc += tl.sum(gt_tile * b_row[:, None], axis=0)
    out_ptrs = grad_u_ptr + pid_bn * stride_gu_bn + pid_t * stride_gu_t + offs_d * stride_gu_d
    prev = tl.load(out_ptrs, mask=mask_d, other=0.0).to(tl.float32)
    tl.store(out_ptrs, prev + acc, mask=mask_d)


@triton.jit
def _separated_retain_backward_kernel(
    retain_ptr,
    write_ptr,
    decode_ptr,
    grad_a_ptr,
    grad_b_ptr,
    grad_chunk_a_ptr,
    grad_retain_ptr,
    grad_write_ptr,
    grad_decode_ptr,
    stride_r_bn,
    stride_r_t,
    stride_r_m,
    stride_w_bn,
    stride_w_t,
    stride_w_m,
    stride_c_bn,
    stride_c_t,
    stride_c_m,
    stride_ga_bn,
    stride_ga_t,
    stride_ga_m,
    stride_gb_bn,
    stride_gb_t,
    stride_gb_m,
    stride_gca_bn,
    stride_gca_m,
    stride_gr_bn,
    stride_gr_t,
    stride_gr_m,
    stride_gw_bn,
    stride_gw_t,
    stride_gw_m,
    stride_gd_bn,
    stride_gd_t,
    stride_gd_m,
    CHUNK_SIZE: tl.constexpr,
    M: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bn = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    tiny = 1.1754943508222875e-38

    prefix_row = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    for t in tl.static_range(0, CHUNK_SIZE):
        retain_ptrs = retain_ptr + pid_bn * stride_r_bn + t * stride_r_t + offs_m * stride_r_m
        retain_row = tl.load(retain_ptrs, mask=mask_m, other=1.0).to(tl.float32)
        prefix_row *= tl.maximum(retain_row, tiny)

        ga_ptrs = grad_a_ptr + pid_bn * stride_ga_bn + t * stride_ga_t + offs_m * stride_ga_m
        gb_ptrs = grad_b_ptr + pid_bn * stride_gb_bn + t * stride_gb_t + offs_m * stride_gb_m
        write_ptrs = write_ptr + pid_bn * stride_w_bn + t * stride_w_t + offs_m * stride_w_m
        decode_ptrs = decode_ptr + pid_bn * stride_c_bn + t * stride_c_t + offs_m * stride_c_m
        ga_row = tl.load(ga_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        gb_row = tl.load(gb_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        write_row = tl.load(write_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        decode_row = tl.load(decode_ptrs, mask=mask_m, other=0.0).to(tl.float32)

        gd_ptrs = grad_decode_ptr + pid_bn * stride_gd_bn + t * stride_gd_t + offs_m * stride_gd_m
        gw_ptrs = grad_write_ptr + pid_bn * stride_gw_bn + t * stride_gw_t + offs_m * stride_gw_m
        tl.store(gd_ptrs, ga_row * prefix_row, mask=mask_m)
        tl.store(gw_ptrs, gb_row / tl.maximum(prefix_row, tiny), mask=mask_m)

    gca_ptrs = grad_chunk_a_ptr + pid_bn * stride_gca_bn + offs_m * stride_gca_m
    gca = tl.load(gca_ptrs, mask=mask_m, other=0.0).to(tl.float32)

    carry = tl.zeros([BLOCK_M], dtype=tl.float32)
    for rev in tl.static_range(0, CHUNK_SIZE):
        t = CHUNK_SIZE - 1 - rev
        pref_row = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
        for s in tl.static_range(0, CHUNK_SIZE):
            retain_ptrs = retain_ptr + pid_bn * stride_r_bn + s * stride_r_t + offs_m * stride_r_m
            retain_row = tl.load(retain_ptrs, mask=mask_m, other=1.0).to(tl.float32)
            if s <= t:
                pref_row *= tl.maximum(retain_row, tiny)

        ga_ptrs = grad_a_ptr + pid_bn * stride_ga_bn + t * stride_ga_t + offs_m * stride_ga_m
        gb_ptrs = grad_b_ptr + pid_bn * stride_gb_bn + t * stride_gb_t + offs_m * stride_gb_m
        write_ptrs = write_ptr + pid_bn * stride_w_bn + t * stride_w_t + offs_m * stride_w_m
        decode_ptrs = decode_ptr + pid_bn * stride_c_bn + t * stride_c_t + offs_m * stride_c_m
        gp_row = tl.load(ga_ptrs, mask=mask_m, other=0.0).to(tl.float32) * tl.load(
            decode_ptrs, mask=mask_m, other=0.0
        ).to(tl.float32)
        gb_row = tl.load(gb_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        write_row = tl.load(write_ptrs, mask=mask_m, other=0.0).to(tl.float32)
        gp_row -= gb_row * write_row / tl.maximum(pref_row * pref_row, tiny)
        if t == CHUNK_SIZE - 1:
            gp_row += gca
        carry += gp_row * pref_row
        retain_ptrs = retain_ptr + pid_bn * stride_r_bn + t * stride_r_t + offs_m * stride_r_m
        retain_row = tl.load(retain_ptrs, mask=mask_m, other=1.0).to(tl.float32)
        gr_ptrs = grad_retain_ptr + pid_bn * stride_gr_bn + t * stride_gr_t + offs_m * stride_gr_m
        tl.store(gr_ptrs, carry / tl.maximum(retain_row, tiny), mask=mask_m)


def _separated_forward_buffers(
    *,
    u_flat: torch.Tensor,
    retain_flat: torch.Tensor,
    write_flat: torch.Tensor,
    decode_flat: torch.Tensor,
    state0: torch.Tensor,
    flat_batch: int,
    nchunks: int,
    chunk_size: int,
    nslots: int,
    value_dim: int,
    block_m: int,
    block_d: int,
    block_t: int,
    need_output: bool,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    bn = flat_batch * nchunks
    chunk_a_buf = torch.empty((bn, nslots), device=u_flat.device, dtype=torch.float32)
    chunk_b_buf = torch.empty((flat_batch, nchunks, nslots, value_dim), device=u_flat.device, dtype=torch.float32)
    prefix_state = torch.empty((flat_batch, nchunks, nslots, value_dim), device=u_flat.device, dtype=torch.float32)
    final_state = torch.empty((flat_batch, nslots, value_dim), device=u_flat.device, dtype=torch.float32)
    out_flat = torch.zeros((bn, chunk_size, value_dim), device=u_flat.device, dtype=u_flat.dtype) if need_output else None

    chunk_a_grid = (triton.cdiv(nslots, block_m), bn)
    summary_grid = (triton.cdiv(value_dim, block_d), triton.cdiv(nslots, block_m), bn)
    prefix_grid = (triton.cdiv(value_dim, block_d), triton.cdiv(nslots, block_m), flat_batch)
    output_grid = (triton.cdiv(value_dim, block_d), bn)

    _aux_chunk_a_kernel[chunk_a_grid](
        retain_flat,
        chunk_a_buf,
        retain_flat.stride(0),
        retain_flat.stride(1),
        retain_flat.stride(2),
        chunk_a_buf.stride(0),
        chunk_a_buf.stride(1),
        CHUNK_SIZE=chunk_size,
        M=nslots,
        BLOCK_M=block_m,
        num_warps=4 if nslots <= 32 else 8,
        num_stages=2,
    )

    _main_chunk_summary_direct_kernel[summary_grid](
        u_flat,
        retain_flat,
        write_flat,
        chunk_b_buf,
        u_flat.stride(0),
        u_flat.stride(1),
        u_flat.stride(2),
        retain_flat.stride(0),
        retain_flat.stride(1),
        retain_flat.stride(2),
        write_flat.stride(0),
        write_flat.stride(1),
        write_flat.stride(2),
        chunk_b_buf.stride(0),
        chunk_b_buf.stride(1),
        chunk_b_buf.stride(2),
        chunk_b_buf.stride(3),
        CHUNK_SIZE=chunk_size,
        M=nslots,
        D=value_dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4 if block_d <= 64 else 8,
        num_stages=2,
    )

    chunk_a_scan = chunk_a_buf.view(flat_batch, nchunks, nslots)
    _separated_prefix_state_kernel[prefix_grid](
        chunk_a_scan,
        chunk_b_buf,
        state0,
        prefix_state,
        final_state,
        chunk_a_scan.stride(0),
        chunk_a_scan.stride(1),
        chunk_a_scan.stride(2),
        chunk_b_buf.stride(0),
        chunk_b_buf.stride(1),
        chunk_b_buf.stride(2),
        chunk_b_buf.stride(3),
        state0.stride(0),
        state0.stride(1),
        state0.stride(2),
        prefix_state.stride(0),
        prefix_state.stride(1),
        prefix_state.stride(2),
        prefix_state.stride(3),
        final_state.stride(0),
        final_state.stride(1),
        final_state.stride(2),
        NUM_CHUNKS=nchunks,
        M=nslots,
        D=value_dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=4 if block_d <= 64 else 8,
        num_stages=2,
    )

    if out_flat is not None:
        _main_output_direct_kernel[output_grid](
            u_flat,
            retain_flat,
            write_flat,
            decode_flat,
            prefix_state.reshape(bn, nslots, value_dim),
            out_flat,
            u_flat.stride(0),
            u_flat.stride(1),
            u_flat.stride(2),
            retain_flat.stride(0),
            retain_flat.stride(1),
            retain_flat.stride(2),
            write_flat.stride(0),
            write_flat.stride(1),
            write_flat.stride(2),
            decode_flat.stride(0),
            decode_flat.stride(1),
            decode_flat.stride(2),
            prefix_state.reshape(bn, nslots, value_dim).stride(0),
            prefix_state.reshape(bn, nslots, value_dim).stride(1),
            prefix_state.reshape(bn, nslots, value_dim).stride(2),
            out_flat.stride(0),
            out_flat.stride(1),
            out_flat.stride(2),
            CHUNK_SIZE=chunk_size,
            M=nslots,
            D=value_dim,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            num_warps=4 if block_d <= 64 else 8,
            num_stages=2,
        )

    return out_flat, chunk_a_scan, chunk_b_buf, final_state


class _SeparatedTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        U: torch.Tensor,
        retain: torch.Tensor,
        write: torch.Tensor,
        decode_weights: torch.Tensor,
        initial_state: torch.Tensor | None,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if U.ndim != 4:
            raise ValueError(f"U must be [B,N,H,D]. Got {tuple(U.shape)}.")
        if retain.ndim != 4 or write.ndim != 4 or decode_weights.ndim != 4:
            raise ValueError("retain, write, decode_weights must all be [B,N,H,M].")
        if not (U.is_cuda and retain.is_cuda and write.is_cuda and decode_weights.is_cuda):
            raise ValueError("separated_trition forward requires CUDA tensors.")

        bsz, seqlen, nheads, value_dim = U.shape
        nslots = retain.shape[-1]
        if retain.shape != (bsz, seqlen, nheads, nslots):
            raise ValueError("retain must be [B,N,H,M].")
        if write.shape != retain.shape or decode_weights.shape != retain.shape:
            raise ValueError("write/decode_weights must match retain shape.")
        if initial_state is not None and initial_state.shape != (bsz, nheads, nslots, value_dim):
            raise ValueError(
                "initial_state must be [B,H,M,D]. "
                f"Got initial_state={tuple(initial_state.shape)}, expected={(bsz, nheads, nslots, value_dim)}."
            )

        chunk_size = _require_supported_chunk_size(chunk_size)
        block_m = _pick_block_m(nslots)
        block_d = _pick_block_d(value_dim)
        block_t = _pick_block_t(chunk_size)

        nchunks = math.ceil(seqlen / chunk_size)
        padded_len = nchunks * chunk_size
        U = _pad_sequence_dim(U, padded_len, 0.0)
        retain = _pad_sequence_dim(retain, padded_len, 1.0)
        write = _pad_sequence_dim(write, padded_len, 0.0)
        decode_weights = _pad_sequence_dim(decode_weights, padded_len, 0.0)

        flat_batch = bsz * nheads
        u_chunked = U.view(bsz, nchunks, chunk_size, nheads, value_dim).permute(0, 3, 1, 2, 4).reshape(
            flat_batch, nchunks, chunk_size, value_dim
        ).contiguous()
        retain_chunked = retain.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 3, 1, 2, 4).reshape(
            flat_batch, nchunks, chunk_size, nslots
        ).contiguous()
        write_chunked = write.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 3, 1, 2, 4).reshape(
            flat_batch, nchunks, chunk_size, nslots
        ).contiguous()
        decode_chunked = decode_weights.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 3, 1, 2, 4).reshape(
            flat_batch, nchunks, chunk_size, nslots
        ).contiguous()
        u_flat = u_chunked.reshape(flat_batch * nchunks, chunk_size, value_dim)
        retain_flat = retain_chunked.reshape(flat_batch * nchunks, chunk_size, nslots)
        write_flat = write_chunked.reshape(flat_batch * nchunks, chunk_size, nslots)
        decode_flat = decode_chunked.reshape(flat_batch * nchunks, chunk_size, nslots)

        if initial_state is None:
            state0 = U.new_zeros((flat_batch, nslots, value_dim))
            initial_state_is_none = True
        else:
            state0 = initial_state.reshape(flat_batch, nslots, value_dim).contiguous()
            initial_state_is_none = False

        out_flat, _, _, final_state = _separated_forward_buffers(
            u_flat=u_flat,
            retain_flat=retain_flat,
            write_flat=write_flat,
            decode_flat=decode_flat,
            state0=state0,
            flat_batch=flat_batch,
            nchunks=nchunks,
            chunk_size=chunk_size,
            nslots=nslots,
            value_dim=value_dim,
            block_m=block_m,
            block_d=block_d,
            block_t=block_t,
            need_output=True,
        )

        out = out_flat.reshape(bsz, nheads, nchunks, chunk_size, value_dim).permute(0, 2, 3, 1, 4).reshape(
            bsz,
            padded_len,
            nheads,
            value_dim,
        )
        final_state = final_state.reshape(bsz, nheads, nslots, value_dim)
        ctx.save_for_backward(u_flat, retain_flat, write_flat, decode_flat, state0)
        ctx.bsz = bsz
        ctx.nheads = nheads
        ctx.nchunks = nchunks
        ctx.chunk_size = chunk_size
        ctx.nslots = nslots
        ctx.value_dim = value_dim
        ctx.block_m = block_m
        ctx.block_d = block_d
        ctx.block_t = block_t
        ctx.initial_state_is_none = initial_state_is_none
        return out[:, :seqlen].contiguous(), final_state

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor, grad_final_state: torch.Tensor | None):
        from causal_flare.autoregressive.separated import _main_block_forward_impl

        u_flat, retain_flat, write_flat, decode_flat, state0 = ctx.saved_tensors
        bsz = ctx.bsz
        nheads = ctx.nheads
        nchunks = ctx.nchunks
        chunk_size = ctx.chunk_size
        nslots = ctx.nslots
        value_dim = ctx.value_dim
        padded_len = nchunks * chunk_size

        if grad_out is None:
            grad_out = torch.zeros((bsz, padded_len, nheads, value_dim), device=u_flat.device, dtype=u_flat.dtype)
        grad_out_padded = _pad_sequence_dim(grad_out, padded_len, 0.0).contiguous()
        grad_final_state = (
            torch.zeros((bsz, nheads, nslots, value_dim), device=u_flat.device, dtype=u_flat.dtype)
            if grad_final_state is None
            else grad_final_state.contiguous()
        )

        U_ref = u_flat.reshape(bsz * nheads, nchunks, chunk_size, value_dim)
        retain_ref = retain_flat.reshape(bsz * nheads, nchunks, chunk_size, nslots)
        write_ref = write_flat.reshape(bsz * nheads, nchunks, chunk_size, nslots)
        decode_ref = decode_flat.reshape(bsz * nheads, nchunks, chunk_size, nslots)
        initial_ref = state0.reshape(bsz * nheads, nslots, value_dim)

        with torch.enable_grad():
            U_ref = U_ref.detach().clone().requires_grad_(True)
            retain_ref = retain_ref.detach().clone().requires_grad_(True)
            write_ref = write_ref.detach().clone().requires_grad_(True)
            decode_ref = decode_ref.detach().clone().requires_grad_(True)
            initial_ref = initial_ref.detach().clone().requires_grad_(True)

            out_ref, final_ref, _ = _main_block_forward_impl(
                state0=initial_ref,
                retain_block=retain_ref,
                write_block=write_ref,
                U_block=U_ref,
                decode_block=decode_ref,
                rmsnorm_read_contrib=False,
            )
            out_ref = out_ref.reshape(bsz, nheads, nchunks, chunk_size, value_dim).permute(0, 2, 3, 1, 4).reshape(
                bsz, padded_len, nheads, value_dim
            )
            final_ref = final_ref.reshape(bsz, nheads, nslots, value_dim)

            grads = torch.autograd.grad(
                outputs=(out_ref, final_ref),
                inputs=(U_ref, retain_ref, write_ref, decode_ref, initial_ref),
                grad_outputs=(grad_out_padded, grad_final_state),
                allow_unused=False,
            )

        grad_u, grad_retain, grad_write, grad_decode, grad_initial = grads
        grad_u = grad_u.reshape(bsz, nheads, nchunks, chunk_size, value_dim).permute(0, 2, 3, 1, 4).reshape(
            bsz, padded_len, nheads, value_dim
        )
        grad_retain = grad_retain.reshape(bsz, nheads, nchunks, chunk_size, nslots).permute(0, 2, 3, 1, 4).reshape(
            bsz, padded_len, nheads, nslots
        )
        grad_write = grad_write.reshape(bsz, nheads, nchunks, chunk_size, nslots).permute(0, 2, 3, 1, 4).reshape(
            bsz, padded_len, nheads, nslots
        )
        grad_decode = grad_decode.reshape(bsz, nheads, nchunks, chunk_size, nslots).permute(0, 2, 3, 1, 4).reshape(
            bsz, padded_len, nheads, nslots
        )
        grad_u = grad_u[:, : grad_out.shape[1]].contiguous()
        grad_retain = grad_retain[:, : grad_out.shape[1]].contiguous()
        grad_write = grad_write[:, : grad_out.shape[1]].contiguous()
        grad_decode = grad_decode[:, : grad_out.shape[1]].contiguous()
        grad_initial = grad_initial.reshape(bsz, nheads, nslots, value_dim)
        if ctx.initial_state_is_none:
            grad_initial = None
        return grad_u, grad_retain, grad_write, grad_decode, grad_initial, None


def flare_autoregressive_separated_trition(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    chunk_size: int | None = None,
    return_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    Triton implementation of the separated FLARE recurrence.

    Recurrence:
      state_t[b,h,m,d] = retain_t[b,h,m] * state_{t-1}[b,h,m,d] + write_t[b,h,m] * U_t[b,h,d]
      y_t[b,h,d]       = sum_m decode_t[b,h,m] * state_t[b,h,m,d]

    This implementation runs the chunkwise prepare / summary / prefix-state /
    dense intra-chunk output decomposition fully on device. Backward recomputes
    compact forward intermediates instead of saving chunk-local activations.
    """
    if chunk_size is None:
        chunk_size = _resolve_separated_trition_chunk_size(
            U.shape[1],
            retain.shape[-1],
            U.shape[-1],
            chunk_size,
        )
    out, final_state = _SeparatedTritonFunction.apply(
        U,
        retain,
        write,
        decode_weights,
        initial_state,
        int(chunk_size),
    )
    if return_final_state:
        return out, final_state
    return out


# Keep a correctly spelled alias so later integrations do not need to preserve
# the filename typo in user-facing APIs.
flare_autoregressive_separated_triton = flare_autoregressive_separated_trition

def _parallel_history_slot_scan_forward_triton(
    *,
    U: torch.Tensor,
    slot_embed: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    readout: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    bsz, seqlen, nheads, value_dim = U.shape
    nslots = retain.shape[-1]
    aux_dim = slot_embed.shape[2]
    chunk_size = _require_supported_chunk_size(int(chunk_size))
    block_m = _pick_block_m(nslots)
    block_a = _pick_block_a(aux_dim)
    nchunks = math.ceil(seqlen / chunk_size)
    padded_len = nchunks * chunk_size
    U = _pad_sequence_dim(U, padded_len, 0.0)
    retain = _pad_sequence_dim(retain, padded_len, 1.0)
    write = _pad_sequence_dim(write, padded_len, 0.0)
    flat_batch = bsz * nheads
    bn = flat_batch * nchunks
    score_chunked = _compute_aux_score_chunked(U=U, slot_embed=slot_embed, nchunks=nchunks, chunk_size=chunk_size)
    retain_chunked = retain.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 3, 1, 2, 4).reshape(
        bn, chunk_size, nslots
    ).contiguous()
    write_chunked = write.view(bsz, nchunks, chunk_size, nheads, nslots).permute(0, 3, 1, 2, 4).reshape(
        bn, chunk_size, nslots
    ).contiguous()

    chunk_a_buf = torch.empty((bn, nslots), device=U.device, dtype=torch.float32)
    chunk_b_buf = torch.empty((flat_batch, nchunks, nslots, aux_dim), device=U.device, dtype=torch.float32)
    prefix_state = torch.empty((flat_batch, nchunks, nslots, aux_dim), device=U.device, dtype=torch.float32)
    final_state = torch.empty((flat_batch, nslots, aux_dim), device=U.device, dtype=torch.float32)
    out_flat = torch.empty((bn, chunk_size, nslots), device=U.device, dtype=U.dtype)

    chunk_a_grid = (triton.cdiv(nslots, block_m), bn)
    chunk_b_grid = (triton.cdiv(aux_dim, block_a), triton.cdiv(nslots, block_m), bn)
    prefix_grid = (triton.cdiv(aux_dim, block_a), triton.cdiv(nslots, block_m), flat_batch)
    out_grid = (triton.cdiv(nslots, block_m), bn)

    _aux_chunk_a_kernel[chunk_a_grid](
        retain_chunked,
        chunk_a_buf,
        retain_chunked.stride(0),
        retain_chunked.stride(1),
        retain_chunked.stride(2),
        chunk_a_buf.stride(0),
        chunk_a_buf.stride(1),
        CHUNK_SIZE=chunk_size,
        M=nslots,
        BLOCK_M=block_m,
        num_warps=4 if nslots <= 32 else 8,
        num_stages=2,
    )

    _aux_chunk_summary_kernel[chunk_b_grid](
        score_chunked,
        retain_chunked,
        write_chunked,
        chunk_b_buf,
        score_chunked.stride(0),
        score_chunked.stride(1),
        score_chunked.stride(2),
        score_chunked.stride(3),
        retain_chunked.stride(0),
        retain_chunked.stride(1),
        retain_chunked.stride(2),
        write_chunked.stride(0),
        write_chunked.stride(1),
        write_chunked.stride(2),
        chunk_b_buf.stride(0),
        chunk_b_buf.stride(1),
        chunk_b_buf.stride(2),
        chunk_b_buf.stride(3),
        CHUNK_SIZE=chunk_size,
        M=nslots,
        A=aux_dim,
        BLOCK_M=block_m,
        BLOCK_A=block_a,
        num_warps=4 if aux_dim <= 8 else 8,
        num_stages=2,
    )

    chunk_a_scan = chunk_a_buf.view(flat_batch, nchunks, nslots)
    state0 = U.new_zeros((flat_batch, nslots, aux_dim), dtype=torch.float32)
    _separated_prefix_state_kernel[prefix_grid](
        chunk_a_scan,
        chunk_b_buf,
        state0,
        prefix_state,
        final_state,
        chunk_a_scan.stride(0),
        chunk_a_scan.stride(1),
        chunk_a_scan.stride(2),
        chunk_b_buf.stride(0),
        chunk_b_buf.stride(1),
        chunk_b_buf.stride(2),
        chunk_b_buf.stride(3),
        state0.stride(0),
        state0.stride(1),
        state0.stride(2),
        prefix_state.stride(0),
        prefix_state.stride(1),
        prefix_state.stride(2),
        prefix_state.stride(3),
        final_state.stride(0),
        final_state.stride(1),
        final_state.stride(2),
        NUM_CHUNKS=nchunks,
        M=nslots,
        D=aux_dim,
        BLOCK_M=block_m,
        BLOCK_D=block_a,
        num_warps=4 if aux_dim <= 8 else 8,
        num_stages=2,
    )

    readout_bh = readout.unsqueeze(0).expand(bsz, -1, -1, -1).reshape(flat_batch, nslots, aux_dim).contiguous().to(torch.float32)
    _aux_output_kernel[out_grid](
        score_chunked,
        retain_chunked,
        write_chunked,
        prefix_state.reshape(bn, nslots, aux_dim),
        readout_bh,
        out_flat,
        score_chunked.stride(0),
        score_chunked.stride(1),
        score_chunked.stride(2),
        score_chunked.stride(3),
        retain_chunked.stride(0),
        retain_chunked.stride(1),
        retain_chunked.stride(2),
        write_chunked.stride(0),
        write_chunked.stride(1),
        write_chunked.stride(2),
        prefix_state.reshape(bn, nslots, aux_dim).stride(0),
        prefix_state.reshape(bn, nslots, aux_dim).stride(1),
        prefix_state.reshape(bn, nslots, aux_dim).stride(2),
        readout_bh.stride(0),
        readout_bh.stride(1),
        readout_bh.stride(2),
        out_flat.stride(0),
        out_flat.stride(1),
        out_flat.stride(2),
        NCHUNKS=nchunks,
        CHUNK_SIZE=chunk_size,
        M=nslots,
        A=aux_dim,
        BLOCK_M=block_m,
        BLOCK_A=block_a,
        num_warps=4 if aux_dim <= 8 else 8,
        num_stages=2,
    )

    out = out_flat.reshape(bsz, nheads, nchunks, chunk_size, nslots).permute(0, 2, 3, 1, 4).reshape(
        bsz, padded_len, nheads, nslots
    )
    return out[:, :seqlen].contiguous()


class _ParallelHistorySlotScanTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, slot_embed, retain, write, readout, chunk_size):
        out = _parallel_history_slot_scan_forward_triton(
            U=U,
            slot_embed=slot_embed,
            retain=retain,
            write=write,
            readout=readout,
            chunk_size=int(chunk_size),
        )
        ctx.save_for_backward(U, slot_embed, retain, write, readout)
        ctx.chunk_size = int(chunk_size)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        U, slot_embed, retain, write, readout = ctx.saved_tensors
        from causal_flare.autoregressive.separated import parallel_history_slot_scan as _parallel_history_slot_scan_ref

        with torch.enable_grad():
            U_ref = U.detach().clone().requires_grad_(True)
            slot_embed_ref = slot_embed.detach().clone().requires_grad_(True)
            retain_ref = retain.detach().clone().requires_grad_(True)
            write_ref = write.detach().clone().requires_grad_(True)
            readout_ref = readout.detach().clone().requires_grad_(True)
            out_ref = _parallel_history_slot_scan_ref(
                U=U_ref,
                slot_embed=slot_embed_ref,
                retain=retain_ref,
                write=write_ref,
                readout=readout_ref,
                chunk_size=ctx.chunk_size,
            )
            grad_U, grad_slot_embed, grad_retain, grad_write, grad_readout = torch.autograd.grad(
                out_ref,
                (U_ref, slot_embed_ref, retain_ref, write_ref, readout_ref),
                grad_outputs=grad_out,
            )

        return grad_U, grad_slot_embed, grad_retain, grad_write, grad_readout, None


def parallel_history_slot_scan_trition(
    *,
    U: torch.Tensor,
    slot_embed: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    readout: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    if chunk_size is None:
        chunk_size = _resolve_separated_trition_chunk_size(
            U.shape[1],
            retain.shape[-1],
            U.shape[-1],
            chunk_size,
        )
    return _ParallelHistorySlotScanTritonFunction.apply(
        U,
        slot_embed,
        retain,
        write,
        readout,
        int(chunk_size),
    )

#!/usr/bin/env python3
"""Reduce the FA2 Triton warp-specialize compilation failure on Hopper."""

from __future__ import annotations

import argparse
import math

import torch
import triton
import triton.language as tl


def _device_cc() -> tuple[int, int]:
    return torch.cuda.get_device_capability()


@triton.jit
def pointer_qk_only(
    q_ptr,
    k_ptr,
    o_ptr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=(offs_m[:, None] < N_CTX), other=0.0)
    last_qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptr + (start_n + offs_n)[None, :] * HEAD_DIM + offs_d[:, None],
            mask=(start_n + offs_n)[None, :] < N_CTX,
            other=0.0,
        )
        last_qk = tl.dot(q, k)

    tl.store(
        o_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :],
        last_qk,
        mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < BLOCK_N),
    )


@triton.jit
def pointer_qk_only_staged(
    q_ptr,
    k_ptr,
    o_ptr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=(offs_m[:, None] < N_CTX), other=0.0)
    last_qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N, num_stages=2, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptr + (start_n + offs_n)[None, :] * HEAD_DIM + offs_d[:, None],
            mask=(start_n + offs_n)[None, :] < N_CTX,
            other=0.0,
        )
        last_qk = tl.dot(q, k)

    tl.store(
        o_ptr + offs_m[:, None] * BLOCK_N + offs_n[None, :],
        last_qk,
        mask=(offs_m[:, None] < N_CTX) & (offs_n[None, :] < BLOCK_N),
    )


@triton.jit
def pointer_load_only(
    v_ptr,
    o_ptr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    start_m = tl.program_id(0)
    offs_rows = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    acc = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for start_n in tl.range(0, N_CTX, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v = tl.load(
            v_ptr + (start_n + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(start_n + offs_n)[:, None] < N_CTX,
            other=0.0,
        )
        acc += tl.sum(v, axis=0)

    tl.store(
        o_ptr + offs_rows[:, None] * HEAD_DIM + offs_d[None, :],
        acc[None, :],
        mask=offs_rows[:, None] < N_CTX,
    )


@triton.jit
def pointer_pv_only(
    v_ptr,
    o_ptr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    start_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    p = tl.full((BLOCK_M, BLOCK_N), 1.0, tl.float16)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        v = tl.load(
            v_ptr + (start_n + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(start_n + offs_n)[:, None] < N_CTX,
            other=0.0,
        )
        acc = tl.dot(p, v, acc)

    tl.store(
        o_ptr + (start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] * HEAD_DIM + offs_d[None, :],
        acc,
        mask=(start_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None] < N_CTX,
    )


@triton.jit
def pointer_attention_like(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + offs_m[:, None] * HEAD_DIM + offs_d[None, :], mask=(offs_m[:, None] < N_CTX), other=0.0)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(
            k_ptr + (start_n + offs_n)[None, :] * HEAD_DIM + offs_d[:, None],
            mask=(start_n + offs_n)[None, :] < N_CTX,
            other=0.0,
        )
        v = tl.load(
            v_ptr + (start_n + offs_n)[:, None] * HEAD_DIM + offs_d[None, :],
            mask=(start_n + offs_n)[:, None] < N_CTX,
            other=0.0,
        )
        scores = tl.dot(q, k).to(v.dtype)
        acc = tl.dot(scores, v, acc)

    tl.store(o_ptr + offs_m[:, None] * HEAD_DIM + offs_d[None, :], acc, mask=(offs_m[:, None] < N_CTX))


@triton.jit
def desc_attention_like(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)

    desc_q = tl.make_tensor_descriptor(q_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(k_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    desc_v = tl.make_tensor_descriptor(v_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    desc_o = tl.make_tensor_descriptor(o_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])

    q = desc_q.load([start_m * BLOCK_M, 0])
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([start_n, 0]).T
        v = desc_v.load([start_n, 0])
        scores = tl.dot(q, k).to(v.dtype)
        acc = tl.dot(scores, v, acc)

    desc_o.store([start_m * BLOCK_M, 0], acc.to(q.dtype))


@triton.jit
def desc_softmax_attention_like(
    sm_scale,
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    m_ptr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    desc_q = tl.make_tensor_descriptor(q_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(k_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    desc_v = tl.make_tensor_descriptor(v_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    desc_o = tl.make_tensor_descriptor(o_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])

    q = desc_q.load([start_m * BLOCK_M, 0])
    qk_scale = sm_scale * 1.44269504
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for start_n in tl.range(0, N_CTX, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([start_n, 0]).T
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = desc_v.load([start_n, 0])
        acc = tl.dot(p.to(v.dtype), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(m_ptr + offs_m, m_i, mask=offs_m < N_CTX)
    desc_o.store([start_m * BLOCK_M, 0], acc.to(q.dtype))


@triton.jit
def desc_softmax_attention_causal_two_loop(
    sm_scale,
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    m_ptr,
    N_CTX,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    warp_specialize: tl.constexpr,
):
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    desc_q = tl.make_tensor_descriptor(q_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])
    desc_k = tl.make_tensor_descriptor(k_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    desc_v = tl.make_tensor_descriptor(v_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_N, HEAD_DIM])
    desc_o = tl.make_tensor_descriptor(o_ptr, [N_CTX, HEAD_DIM], [HEAD_DIM, 1], [BLOCK_M, HEAD_DIM])

    q = desc_q.load([start_m * BLOCK_M, 0])
    qk_scale = sm_scale * 1.44269504
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for start_n in tl.range(0, start_m * BLOCK_M, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([start_n, 0]).T
        qk = tl.dot(q, k)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = desc_v.load([start_n, 0])
        acc = tl.dot(p.to(v.dtype), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    for start_n in tl.range(start_m * BLOCK_M, (start_m + 1) * BLOCK_M, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([start_n, 0]).T
        qk = tl.dot(q, k)
        mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, axis=1)
        acc = acc * alpha[:, None]
        v = desc_v.load([start_n, 0])
        acc = tl.dot(p.to(v.dtype), v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij

    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    tl.store(m_ptr + offs_m, m_i, mask=offs_m < N_CTX)
    desc_o.store([start_m * BLOCK_M, 0], acc.to(q.dtype))


def _alloc_inputs(n_ctx: int, head_dim: int, dtype: torch.dtype):
    q = torch.randn((n_ctx, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((n_ctx, head_dim), device="cuda", dtype=dtype)
    v = torch.randn((n_ctx, head_dim), device="cuda", dtype=dtype)
    o = torch.empty_like(q)
    m = torch.empty((n_ctx,), device="cuda", dtype=torch.float32)
    return q, k, v, o, m


def _run_case(name: str, kernel, *, n_ctx: int, head_dim: int, block_m: int, block_n: int, dtype: torch.dtype, warp_specialize: bool):
    q, k, v, o, m = _alloc_inputs(n_ctx, head_dim, dtype)
    grid = (triton.cdiv(n_ctx, block_m),)
    try:
        if kernel is pointer_load_only:
            kernel[grid](
                v,
                o,
                n_ctx,
                HEAD_DIM=head_dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                warp_specialize=warp_specialize,
                num_warps=4,
                num_stages=2,
            )
        elif kernel is pointer_qk_only or kernel is pointer_qk_only_staged:
            o_qk = torch.empty((n_ctx, block_n), device="cuda", dtype=torch.float32)
            kernel[grid](
                q,
                k,
                o_qk,
                n_ctx,
                HEAD_DIM=head_dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                warp_specialize=warp_specialize,
                num_warps=4,
                num_stages=2,
            )
        elif kernel is pointer_pv_only:
            kernel[grid](
                v,
                o,
                n_ctx,
                HEAD_DIM=head_dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                warp_specialize=warp_specialize,
                num_warps=4,
                num_stages=2,
            )
        elif kernel is pointer_attention_like:
            kernel[grid](
                q,
                k,
                v,
                o,
                n_ctx,
                HEAD_DIM=head_dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                warp_specialize=warp_specialize,
                num_warps=4,
                num_stages=2,
            )
        elif kernel is desc_attention_like:
            kernel[grid](
                q,
                k,
                v,
                o,
                n_ctx,
                HEAD_DIM=head_dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                warp_specialize=warp_specialize,
                num_warps=4,
                num_stages=2,
            )
        else:
            kernel[grid](
                1.0 / math.sqrt(head_dim),
                q,
                k,
                v,
                o,
                m,
                n_ctx,
                HEAD_DIM=head_dim,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                warp_specialize=warp_specialize,
                num_warps=4,
                num_stages=2,
            )
        torch.cuda.synchronize()
        print(f"{name:34s} warp_specialize={warp_specialize:<5} PASS")
        return True
    except Exception as exc:
        print(f"{name:34s} warp_specialize={warp_specialize:<5} FAIL {type(exc).__name__}: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")

    print(f"device={torch.cuda.get_device_name(0)} cc={_device_cc()} triton={triton.__version__}")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    triton.set_allocator(lambda size, align, stream: torch.empty(size, dtype=torch.int8, device="cuda"))

    cases = [
        ("pointer_load_only", pointer_load_only),
        ("pointer_qk_only", pointer_qk_only),
        ("pointer_qk_only_staged", pointer_qk_only_staged),
        ("pointer_pv_only", pointer_pv_only),
        ("pointer_attention_like", pointer_attention_like),
        ("desc_attention_like", desc_attention_like),
        ("desc_softmax_attention_like", desc_softmax_attention_like),
        ("desc_softmax_causal_two_loop", desc_softmax_attention_causal_two_loop),
    ]
    for name, kernel in cases:
        for warp_specialize in (False, True):
            _run_case(
                name,
                kernel,
                n_ctx=args.N,
                head_dim=args.D,
                block_m=64,
                block_n=32,
                dtype=dtype,
                warp_specialize=warp_specialize,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

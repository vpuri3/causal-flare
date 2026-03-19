#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass

import torch
import triton

from benchmark.kernel_sweep_utils import (
    PhaseSweepSpec,
    bench_result_dict,
    benchmark_named_callables,
    best_sweep_result,
    sweep_result_dict,
    tune_phase_specs,
)
from causal_flare.autoregressive.stablemax import flare_autoregressive_stablemax_pytorch
from causal_flare.autoregressive.stablemax_triton import (
    _stablemax_power_mode,
    _stablemax_forward_config,
    flare_autoregressive_stablemax_triton,
    stablemax_output_kernel,
    stablemax_prefix_scan_kernel,
    stablemax_prepare_kernel,
)
from causal_flare.autoregressive.training import flare_autoregressive_triton


@dataclass(frozen=True)
class StablemaxCase:
    batch_size: int
    num_heads: int
    seq_len: int
    latent_queries: int
    head_dim: int
    dtype: str
    power: float


def parse_dtype(token: str) -> torch.dtype:
    normalized = token.strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype={token!r}. Expected one of {sorted(mapping)}.")
    return mapping[normalized]


def make_case(args: argparse.Namespace) -> StablemaxCase:
    if args.batch_heads % args.batch_size != 0:
        raise ValueError(f"batch_heads={args.batch_heads} must be divisible by batch_size={args.batch_size}.")
    return StablemaxCase(
        batch_size=args.batch_size,
        num_heads=args.batch_heads // args.batch_size,
        seq_len=args.seq_len,
        latent_queries=args.latent_queries,
        head_dim=args.head_dim,
        dtype=args.dtype,
        power=float(args.power),
    )


def allocate_inputs(case: StablemaxCase, *, device: str) -> tuple[torch.Tensor, ...]:
    dtype = parse_dtype(case.dtype)
    B, H, N, M, D = case.batch_size, case.num_heads, case.seq_len, case.latent_queries, case.head_dim
    q = torch.randn((H, M, D), device=device, dtype=dtype)
    k = torch.randn((B, N, H, D), device=device, dtype=dtype)
    v = torch.randn((B, N, H, D), device=device, dtype=dtype)
    q_dec = torch.randn((B, N, H, D), device=device, dtype=dtype)
    k_dec = torch.randn((H, M, D), device=device, dtype=dtype)
    return q, k, v, q_dec, k_dec


def default_phase_candidates() -> dict[str, list[tuple[str, dict[str, int]]]]:
    # Keep the default sweep intentionally compact and centered around buckets
    # that were already competitive on smaller stablemax reference runs. The
    # goal here is a reusable "quick tuning" utility for large-context kernels,
    # not an exhaustive autotuner that spends most of its time compiling.
    prepare_like = [
        (
            f"bm{bm}_w{nw}_s{ns}",
            {"BLOCK_M": bm, "num_warps": nw, "num_stages": ns},
        )
        for bm, nw, ns in itertools.product((64, 128), (4,), (2, 3))
    ]
    return {
        "prepare": list(prepare_like),
        "prefix": [
            (
                f"bm{bm}_w{nw}_s1",
                {"BLOCK_M": bm, "num_warps": nw, "num_stages": 1},
            )
            for bm, nw in itertools.product((32, 64), (4, 8))
        ],
        "output_fused": [
            (
                f"bm{bm}_w{nw}_s{ns}",
                {"BLOCK_M": bm, "num_warps": nw, "num_stages": ns},
            )
            for bm, nw, ns in itertools.product((32, 64), (2, 4), (1, 2))
        ],
    }


def _phase_best_params(
    tuned: dict[str, dict[str, object] | None],
    phase_name: str,
) -> dict[str, int]:
    phase_payload = tuned.get(phase_name)
    if phase_payload is None:
        raise RuntimeError(f"No valid tuned configuration was found for phase={phase_name!r}.")
    params = phase_payload.get("params")
    if not isinstance(params, dict):
        raise RuntimeError(f"Malformed tuned configuration payload for phase={phase_name!r}: {phase_payload!r}")
    return {str(key): int(value) for key, value in params.items()}


def precompute_reference_state(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_dec: torch.Tensor,
    k_dec: torch.Tensor,
    *,
    chunk_size: int,
    scale: float,
    power: float,
    input_precision: str,
) -> dict[str, torch.Tensor | int]:
    B, N, H, D_value = v.shape[0], v.shape[1], v.shape[2], v.shape[3]
    M = q.shape[1]
    power_mode = _stablemax_power_mode(float(power))
    padded_len = math.ceil(N / chunk_size) * chunk_size
    num_chunks = padded_len // chunk_size
    chunk_den = torch.empty((B * H, num_chunks, M), device=q.device, dtype=torch.float32)
    chunk_num = torch.empty((B * H, num_chunks, M, D_value), device=q.device, dtype=v.dtype)
    prefix_den = torch.empty((B * H, num_chunks, M), device=q.device, dtype=torch.float32)
    prefix_num = torch.empty((B * H, num_chunks, M, D_value), device=q.device, dtype=v.dtype)
    num_d_tiles = triton.cdiv(D_value, 32)

    stablemax_prepare_kernel[(B * H, num_chunks, triton.cdiv(M, 32) * num_d_tiles)](
        q,
        k,
        v,
        chunk_den,
        chunk_num,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *chunk_den.stride(),
        *chunk_num.stride(),
        N,
        M,
        D_SCORE=q.shape[-1],
        D_VALUE=D_value,
        CHUNK_SIZE=chunk_size,
        scale=scale,
        power=power,
        POWER_MODE=power_mode,
        BLOCK_M=32,
        BLOCK_D=32,
        BLOCK_K=32,
        INPUT_PRECISION=input_precision,
        H=H,
        num_warps=4,
        num_stages=2,
    )
    stablemax_prefix_scan_kernel[(B * H, triton.cdiv(M, 32) * num_d_tiles)](
        chunk_den,
        chunk_num,
        prefix_den,
        prefix_num,
        *chunk_den.stride(),
        *chunk_num.stride(),
        *prefix_den.stride(),
        *prefix_num.stride(),
        M,
        D_VALUE=D_value,
        NUM_CHUNKS=num_chunks,
        BLOCK_M=32,
        BLOCK_D=32,
        num_warps=4,
        num_stages=2,
    )
    torch.cuda.synchronize()
    return {
        "num_chunks": num_chunks,
        "chunk_den": chunk_den,
        "chunk_num": chunk_num,
        "prefix_den": prefix_den,
        "prefix_num": prefix_num,
    }


def tune_phases_for_chunk(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_dec: torch.Tensor,
    k_dec: torch.Tensor,
    *,
    chunk_size: int,
    scale: float,
    power: float,
    warmup: int,
    reps: int,
    input_precision: str,
) -> dict[str, dict[str, object] | None]:
    B, N, H, D_value = v.shape[0], v.shape[1], v.shape[2], v.shape[3]
    M = q.shape[1]
    power_mode = _stablemax_power_mode(float(power))
    num_d_tiles = triton.cdiv(D_value, 32)
    ref = precompute_reference_state(
        q,
        k,
        v,
        q_dec,
        k_dec,
        chunk_size=chunk_size,
        scale=scale,
        power=power,
        input_precision=input_precision,
    )
    num_chunks = int(ref["num_chunks"])
    candidates = default_phase_candidates()

    def build_prepare(params: dict[str, int]):
        chunk_den = torch.empty_like(ref["chunk_den"])
        chunk_num = torch.empty_like(ref["chunk_num"])

        def launch():
            stablemax_prepare_kernel[(B * H, num_chunks, triton.cdiv(M, params["BLOCK_M"]) * num_d_tiles)](
                q,
                k,
                v,
                chunk_den,
                chunk_num,
                *q.stride(),
                *k.stride(),
                *v.stride(),
                *chunk_den.stride(),
                *chunk_num.stride(),
                N,
                M,
                D_SCORE=q.shape[-1],
                D_VALUE=D_value,
                CHUNK_SIZE=chunk_size,
                scale=scale,
                power=power,
                POWER_MODE=power_mode,
                BLOCK_M=params["BLOCK_M"],
                BLOCK_D=32,
                BLOCK_K=32,
                INPUT_PRECISION=input_precision,
                H=H,
                num_warps=params["num_warps"],
                num_stages=params["num_stages"],
            )

        return launch

    def build_prefix(params: dict[str, int]):
        prefix_den = torch.empty_like(ref["prefix_den"])
        prefix_num = torch.empty_like(ref["prefix_num"])

        def launch():
            stablemax_prefix_scan_kernel[(B * H, triton.cdiv(M, params["BLOCK_M"]) * num_d_tiles)](
                ref["chunk_den"],
                ref["chunk_num"],
                prefix_den,
                prefix_num,
                *ref["chunk_den"].stride(),
                *ref["chunk_num"].stride(),
                *prefix_den.stride(),
                *prefix_num.stride(),
                M,
                D_VALUE=D_value,
                NUM_CHUNKS=num_chunks,
                BLOCK_M=params["BLOCK_M"],
                BLOCK_D=32,
                num_warps=params["num_warps"],
                num_stages=params["num_stages"],
            )

        return launch

    def build_output_fused(params: dict[str, int]):
        z_enc = torch.empty((B, H, N, M), device=q.device, dtype=torch.float32)
        z_dec = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
        out = torch.empty((B, N, H, D_value), device=q.device, dtype=torch.float32)

        def launch():
            stablemax_output_kernel[(B * H, num_chunks, num_d_tiles)](
                q,
                k,
                q_dec,
                k_dec,
                z_enc,
                z_dec,
                ref["prefix_den"],
                ref["prefix_num"],
                v,
                out,
                *q.stride(),
                *k.stride(),
                *q_dec.stride(),
                *k_dec.stride(),
                *z_enc.stride(),
                *z_dec.stride(),
                *ref["prefix_den"].stride(),
                *ref["prefix_num"].stride(),
                *v.stride(),
                *out.stride(),
                N,
                M,
                D_SCORE=q.shape[-1],
                D_VALUE=D_value,
                CHUNK_SIZE=chunk_size,
                scale=scale,
                power=power,
                POWER_MODE=power_mode,
                BLOCK_M=params["BLOCK_M"],
                BLOCK_D=32,
                BLOCK_K=32,
                INPUT_PRECISION=input_precision,
                H=H,
                num_warps=params["num_warps"],
                num_stages=params["num_stages"],
            )

        return launch

    phase_specs = (
        PhaseSweepSpec(name="prepare", candidates=tuple(candidates["prepare"]), build_callable=build_prepare),
        PhaseSweepSpec(name="prefix", candidates=tuple(candidates["prefix"]), build_callable=build_prefix),
        PhaseSweepSpec(name="output_fused", candidates=tuple(candidates["output_fused"]), build_callable=build_output_fused),
    )
    phase_results = tune_phase_specs(phase_specs, warmup=warmup, reps=reps)
    tuned: dict[str, dict[str, object] | None] = {}
    for phase_name, results in phase_results.items():
        best = best_sweep_result(results)
        tuned[phase_name] = None if best is None else sweep_result_dict(best)
    return tuned


def run_tuned_stablemax_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_dec: torch.Tensor,
    k_dec: torch.Tensor,
    *,
    power: float,
    chunk_size: int,
    input_precision: str,
    tuned: dict[str, dict[str, object] | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, N, H, D_value = v.shape[0], v.shape[1], v.shape[2], v.shape[3]
    M = q.shape[1]
    scale = q.shape[-1] ** -0.5
    power_mode = _stablemax_power_mode(float(power))
    padded_len = math.ceil(N / chunk_size) * chunk_size
    num_chunks = padded_len // chunk_size

    prepare_params = _phase_best_params(tuned, "prepare")
    prefix_params = _phase_best_params(tuned, "prefix")
    output_params = _phase_best_params(tuned, "output_fused")

    prepare_num_d_tiles = triton.cdiv(D_value, 32)
    prefix_num_d_tiles = triton.cdiv(D_value, 32)
    output_num_d_tiles = triton.cdiv(D_value, 32)

    chunk_den = torch.empty((B * H, num_chunks, M), device=q.device, dtype=torch.float32)
    chunk_num = torch.empty((B * H, num_chunks, M, D_value), device=q.device, dtype=v.dtype)
    prefix_den = torch.empty((B * H, num_chunks, M), device=q.device, dtype=torch.float32)
    prefix_num = torch.empty((B * H, num_chunks, M, D_value), device=q.device, dtype=v.dtype)
    z_enc = torch.empty((B, H, N, M), device=q.device, dtype=torch.float32)
    z_dec = torch.empty((B, H, N), device=q.device, dtype=torch.float32)
    out = torch.empty((B, N, H, D_value), device=q.device, dtype=torch.float32)

    stablemax_prepare_kernel[(B * H, num_chunks, triton.cdiv(M, prepare_params["BLOCK_M"]) * prepare_num_d_tiles)](
        q,
        k,
        v,
        chunk_den,
        chunk_num,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *chunk_den.stride(),
        *chunk_num.stride(),
        N,
        M,
        D_SCORE=q.shape[-1],
        D_VALUE=D_value,
        CHUNK_SIZE=chunk_size,
        scale=scale,
        power=power,
        POWER_MODE=power_mode,
        BLOCK_M=prepare_params["BLOCK_M"],
        BLOCK_D=32,
        BLOCK_K=32,
        INPUT_PRECISION=input_precision,
        H=H,
        num_warps=prepare_params["num_warps"],
        num_stages=prepare_params["num_stages"],
    )

    stablemax_prefix_scan_kernel[(B * H, triton.cdiv(M, prefix_params["BLOCK_M"]) * prefix_num_d_tiles)](
        chunk_den,
        chunk_num,
        prefix_den,
        prefix_num,
        *chunk_den.stride(),
        *chunk_num.stride(),
        *prefix_den.stride(),
        *prefix_num.stride(),
        M,
        D_VALUE=D_value,
        NUM_CHUNKS=num_chunks,
        BLOCK_M=prefix_params["BLOCK_M"],
        BLOCK_D=32,
        num_warps=prefix_params["num_warps"],
        num_stages=prefix_params["num_stages"],
    )

    stablemax_output_kernel[(B * H, num_chunks, output_num_d_tiles)](
        q,
        k,
        q_dec,
        k_dec,
        z_enc,
        z_dec,
        prefix_den,
        prefix_num,
        v,
        out,
        *q.stride(),
        *k.stride(),
        *q_dec.stride(),
        *k_dec.stride(),
        *z_enc.stride(),
        *z_dec.stride(),
        *prefix_den.stride(),
        *prefix_num.stride(),
        *v.stride(),
        *out.stride(),
        N,
        M,
        D_SCORE=q.shape[-1],
        D_VALUE=D_value,
        CHUNK_SIZE=chunk_size,
        scale=scale,
        power=power,
        POWER_MODE=power_mode,
        BLOCK_M=output_params["BLOCK_M"],
        BLOCK_D=32,
        BLOCK_K=32,
        INPUT_PRECISION=input_precision,
        H=H,
        num_warps=output_params["num_warps"],
        num_stages=output_params["num_stages"],
    )
    return out.to(v.dtype), z_enc, z_dec


def benchmark_end_to_end_variants(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_dec: torch.Tensor,
    k_dec: torch.Tensor,
    *,
    power: float,
    chunk_size: int,
    warmup: int,
    reps: int,
    input_precision: str,
    tuned: dict[str, dict[str, object] | None] | None,
) -> dict[str, dict[str, object]]:
    scale = q.shape[-1] ** -0.5
    variants = {
        "stablemax_pytorch": lambda: flare_autoregressive_stablemax_pytorch(
            q,
            k,
            v,
            scale=scale,
            chunk_size=chunk_size,
            Q_dec=q_dec,
            K_dec=k_dec,
            power=power,
        ),
        "stablemax_triton": lambda: flare_autoregressive_stablemax_triton(
            q,
            k,
            v,
            scale=scale,
            chunk_size=chunk_size,
            Q_dec=q_dec,
            K_dec=k_dec,
            power=power,
        ),
        "flare_triton": lambda: flare_autoregressive_triton(
            q,
            k,
            v,
            scale=scale,
            chunk_size=chunk_size,
            Q_dec=q_dec,
            K_dec=k_dec,
        ),
    }
    if tuned is not None:
        variants["stablemax_triton_tuned"] = lambda: run_tuned_stablemax_forward(
            q,
            k,
            v,
            q_dec,
            k_dec,
            power=power,
            chunk_size=chunk_size,
            input_precision=input_precision,
            tuned=tuned,
        )
    bench_results = benchmark_named_callables(variants, warmup=warmup, reps=reps)
    return {name: bench_result_dict(result) for name, result in bench_results.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune stablemax phase kernels and benchmark stablemax/FLARE variants.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-heads", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--latent-queries", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--chunk-sizes", type=str, default="32,64,128,256")
    parser.add_argument("--tune-warmup", type=int, default=3)
    parser.add_argument("--tune-reps", type=int, default=8)
    parser.add_argument("--bench-warmup", type=int, default=5)
    parser.add_argument("--bench-reps", type=int, default=20)
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()

    case = make_case(args)
    q, k, v, q_dec, k_dec = allocate_inputs(case, device="cuda")
    scale = q.shape[-1] ** -0.5
    input_precision = _stablemax_forward_config(
        M=case.latent_queries,
        D_score=case.head_dim,
        D_value=case.head_dim,
        chunk_size=32,
        input_precision=None,
    )["input_precision"]

    chunk_sizes = tuple(int(token.strip()) for token in args.chunk_sizes.split(",") if token.strip())
    report: dict[str, object] = {
        "case": asdict(case),
        "separate_decode_inputs": True,
        "chunk_sizes": chunk_sizes,
        "phase_tuning": {},
        "forward_comparison_ms": {},
    }

    for chunk_size in chunk_sizes:
        tuned_chunk: dict[str, dict[str, object] | None] | None = None
        if not args.skip_tuning:
            tuned_chunk = tune_phases_for_chunk(
                q,
                k,
                v,
                q_dec,
                k_dec,
                chunk_size=chunk_size,
                scale=scale,
                power=case.power,
                warmup=args.tune_warmup,
                reps=args.tune_reps,
                input_precision=input_precision,
            )
            report["phase_tuning"][chunk_size] = tuned_chunk
        report["forward_comparison_ms"][chunk_size] = benchmark_end_to_end_variants(
            q,
            k,
            v,
            q_dec,
            k_dec,
            power=case.power,
            chunk_size=chunk_size,
            warmup=args.bench_warmup,
            reps=args.bench_reps,
            input_precision=input_precision,
            tuned=tuned_chunk,
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

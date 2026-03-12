#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import torch
import triton
import triton.testing
from triton.runtime.driver import driver

from causal_flare import flare_chunk_triton
from causal_flare._common import _resolve_attn_scale
from causal_flare.chunked import (
    _get_chunked_forward_config,
    _resolve_chunked_decode_inputs,
    flare_chunk_decoder_lse,
    flare_chunk_fwd,
    flare_chunk_prefix,
    flare_chunk_prepare,
)


@dataclass(frozen=True)
class Case:
    name: str
    batch_size: int
    num_heads: int
    seq_len: int
    latent_queries: int
    head_dim: int


@dataclass(frozen=True)
class KernelProfile:
    name: str
    num_warps: int
    num_stages: int
    regs_per_thread: int
    spills: int
    shared_bytes: int
    threads_per_cta: int
    occupancy_pct_est: float
    active_ctas_per_sm_est: int
    limiting_factors: list[str]


DEFAULT_CASES = [
    Case("h16_d32", batch_size=8, num_heads=16, seq_len=4096, latent_queries=64, head_dim=32),
    Case("h4_d128", batch_size=8, num_heads=4, seq_len=4096, latent_queries=64, head_dim=128),
    Case("h4_d256", batch_size=8, num_heads=4, seq_len=4096, latent_queries=64, head_dim=256),
]

DEFAULT_CONFIGS = {
    "default": {},
}


def parse_case(spec: str) -> Case:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 6:
        raise ValueError(
            f"Invalid --case={spec!r}. Expected name,batch,heads,seq,latent_queries,head_dim."
        )
    name, batch_size, num_heads, seq_len, latent_queries, head_dim = parts
    return Case(
        name=name,
        batch_size=int(batch_size),
        num_heads=int(num_heads),
        seq_len=int(seq_len),
        latent_queries=int(latent_queries),
        head_dim=int(head_dim),
    )


def parse_config(spec: str) -> tuple[str, dict[str, str]]:
    if ":" not in spec:
        raise ValueError(f"Invalid --config={spec!r}. Expected name:key=value,key=value.")
    name, raw_items = spec.split(":", 1)
    env = {}
    if raw_items.strip():
        for item in raw_items.split(","):
            key, value = item.split("=", 1)
            env[key.strip()] = value.strip()
    return name.strip(), env


def parse_dtype(name: str) -> torch.dtype:
    token = name.strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if token not in mapping:
        raise ValueError(f"Unsupported dtype={name!r}. Expected one of {sorted(mapping)}.")
    return mapping[token]


@contextlib.contextmanager
def temp_env(overrides: dict[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def estimate_occupancy(compiled_kernel, device_props: dict[str, int]) -> tuple[float, int, list[str]]:
    metadata = compiled_kernel.metadata
    threads_per_cta = metadata.num_warps * device_props["warp_size"]
    regs_per_cta = compiled_kernel.n_regs * threads_per_cta
    shared_per_cta = metadata.shared
    max_warps_per_sm = device_props["max_threads_per_sm"] // device_props["warp_size"]

    limits = {
        "arch": 32,
        "threads": device_props["max_threads_per_sm"] // threads_per_cta,
        "warps": max_warps_per_sm // metadata.num_warps,
        "regs": device_props["max_num_regs"] // regs_per_cta if regs_per_cta > 0 else 32,
        "shared": device_props["max_shared_mem"] // shared_per_cta if shared_per_cta > 0 else 32,
    }
    active_ctas = max(1, min(limits.values()))
    occupancy = active_ctas * metadata.num_warps / max_warps_per_sm
    limiting = [name for name, value in limits.items() if value == active_ctas]
    return occupancy * 100.0, active_ctas, limiting


def estimate_occupancy_from_resource_dict(resource: dict[str, int], device_props: dict[str, int]) -> tuple[float, int, list[str]]:
    threads_per_cta = int(resource["num_warps"]) * device_props["warp_size"]
    regs_per_cta = int(resource["regs_per_thread"]) * threads_per_cta
    shared_per_cta = int(resource["shared_bytes"])
    max_warps_per_sm = device_props["max_threads_per_sm"] // device_props["warp_size"]
    limits = {
        "arch": 32,
        "threads": device_props["max_threads_per_sm"] // threads_per_cta,
        "warps": max_warps_per_sm // int(resource["num_warps"]),
        "regs": device_props["max_num_regs"] // regs_per_cta if regs_per_cta > 0 else 32,
        "shared": device_props["max_shared_mem"] // shared_per_cta if shared_per_cta > 0 else 32,
    }
    active_ctas = max(1, min(limits.values()))
    occupancy = active_ctas * int(resource["num_warps"]) / max_warps_per_sm
    limiting = [name for name, value in limits.items() if value == active_ctas]
    return occupancy * 100.0, active_ctas, limiting


def compile_forward_kernels(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    input_precision: str | None,
) -> dict[str, object]:
    h, m, d = q.shape
    b, n, _, _ = k.shape
    scale = _resolve_attn_scale(None, d)
    cfg = _get_chunked_forward_config(
        M=m,
        N=n,
        score_head_dim=d,
        value_head_dim=d,
        dtype=k.dtype,
        chunk_size=None,
        input_precision=input_precision,
    )
    q_dec, k_dec, separate_q_dec, separate_k_dec, weight_sharing = _resolve_chunked_decode_inputs(q, k, None, None)
    q_dec_comp = q_dec if separate_q_dec else k
    k_dec_comp = k_dec if separate_k_dec else q

    bh = b * h
    device = q.device
    stats_dtype = cfg["stats_dtype"]
    chunk_max = torch.full((bh, cfg["NUM_CHUNKS"], m), -float("inf"), device=device, dtype=stats_dtype)
    chunk_den = torch.zeros((bh, cfg["NUM_CHUNKS"], m), device=device, dtype=stats_dtype)
    chunk_num = torch.zeros((bh, cfg["NUM_CHUNKS"], m, d), device=device, dtype=stats_dtype)

    q_stride = q.stride()
    k_stride = k.stride()
    v_stride = v.stride()
    prepare_kernel = flare_chunk_prepare[
        (bh, cfg["NUM_CHUNKS"], cfg["NUM_M_BLOCKS"] * cfg["NUM_PREPARE_D_BLOCKS"])
    ](
        k,
        q,
        v,
        chunk_max,
        chunk_den,
        chunk_num,
        k_stride[0],
        k_stride[1],
        k_stride[2],
        k_stride[3],
        q_stride[0],
        q_stride[1],
        q_stride[2],
        v_stride[0],
        v_stride[1],
        v_stride[2],
        v_stride[3],
        *chunk_max.stride(),
        *chunk_den.stride(),
        *chunk_num.stride(),
        bh,
        m,
        n,
        d,
        scale,
        CHUNK_SIZE=cfg["CHUNK_SIZE"],
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_D=cfg["PREPARE_BLOCK_D"],
        BLOCK_K=cfg["PREPARE_BLOCK_K"],
        NUM_D_BLOCKS=cfg["NUM_PREPARE_D_BLOCKS"],
        USE_FP16=cfg["use_fp16"],
        USE_BF16=cfg["use_bf16"],
        USE_FP32_STATS=cfg["stats_fp32"],
        INPUT_PRECISION=cfg["input_precision"],
        H=h,
        num_warps=cfg["prepare_num_warps"],
        num_stages=cfg["prepare_num_stages"],
    )

    prefix_max = torch.empty((bh, cfg["NUM_CHUNKS"], m), device=device, dtype=stats_dtype)
    prefix_den = torch.zeros((bh, cfg["NUM_CHUNKS"], m), device=device, dtype=stats_dtype)
    prefix_num = torch.zeros((bh, cfg["NUM_CHUNKS"], m, d), device=device, dtype=stats_dtype)
    prefix_kernel = flare_chunk_prefix[(bh, cfg["NUM_PREFIX_M_BLOCKS"], cfg["NUM_PREFIX_D_BLOCKS"])](
        chunk_max,
        chunk_den,
        chunk_num,
        prefix_max,
        prefix_den,
        prefix_num,
        *chunk_max.stride(),
        *chunk_den.stride(),
        *chunk_num.stride(),
        *prefix_max.stride(),
        *prefix_den.stride(),
        *prefix_num.stride(),
        bh,
        m,
        d,
        cfg["NUM_CHUNKS"],
        BLOCK_M=cfg["PREFIX_BLOCK_M"],
        BLOCK_D=cfg["PREFIX_BLOCK_D"],
        USE_FP16=cfg["use_fp16"],
        USE_BF16=cfg["use_bf16"],
        USE_FP32_STATS=cfg["stats_fp32"],
        num_warps=cfg["prefix_num_warps"],
        num_stages=cfg["prefix_num_stages"],
    )

    num_fwd_m_tiles = triton.cdiv(m, cfg["BLOCK_M"])
    single_writer_output = num_fwd_m_tiles == 1 and cfg["NUM_FWD_D_BLOCKS"] == 1
    if single_writer_output:
        output = torch.empty((b, n, h, d), device=device, dtype=torch.float32)
    else:
        output = torch.zeros((b, n, h, d), device=device, dtype=torch.float32)
    lse_enc = torch.full((b, h, n, m), -float("inf"), device=device, dtype=torch.float32)
    lse_dec = torch.empty((b * h, n), device=device, dtype=torch.float32)
    q_dec_stride = q_dec_comp.stride()
    k_dec_stride = k_dec_comp.stride()

    decoder_lse_kernel = flare_chunk_decoder_lse[(b * h, cfg["NUM_CHUNKS"])](
        q_dec_comp,
        k_dec_comp,
        lse_dec,
        q_dec_stride[0],
        q_dec_stride[1],
        q_dec_stride[2],
        q_dec_stride[3],
        k_dec_stride[0],
        k_dec_stride[1],
        k_dec_stride[2],
        *lse_dec.stride(),
        b * h,
        m,
        n,
        d,
        scale,
        CHUNK_SIZE=cfg["CHUNK_SIZE"],
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_K=cfg["FWD_BLOCK_K"],
        BLOCK_T=cfg["BLOCK_T"],
        INPUT_PRECISION=cfg["input_precision"],
        H=h,
        num_warps=cfg["decoder_num_warps"],
        num_stages=cfg["decoder_num_stages"],
    )

    fwd_kernel = flare_chunk_fwd[(b * h, cfg["NUM_CHUNKS"], num_fwd_m_tiles * cfg["NUM_FWD_D_BLOCKS"])](
        k,
        q,
        v,
        q_dec_comp,
        k_dec_comp,
        prefix_max,
        prefix_den,
        prefix_num,
        output,
        lse_enc,
        lse_dec,
        k_stride[0],
        k_stride[1],
        k_stride[2],
        k_stride[3],
        q_stride[0],
        q_stride[1],
        q_stride[2],
        v_stride[0],
        v_stride[1],
        v_stride[2],
        v_stride[3],
        q_dec_stride[0],
        q_dec_stride[1],
        q_dec_stride[2],
        q_dec_stride[3],
        k_dec_stride[0],
        k_dec_stride[1],
        k_dec_stride[2],
        *prefix_max.stride(),
        *prefix_den.stride(),
        *prefix_num.stride(),
        *output.stride(),
        *lse_enc.stride(),
        *lse_dec.stride(),
        b * h,
        m,
        n,
        d,
        scale,
        NUM_D_BLOCKS=cfg["NUM_FWD_D_BLOCKS"],
        CHUNK_SIZE=cfg["CHUNK_SIZE"],
        BLOCK_M=cfg["BLOCK_M"],
        BLOCK_D=cfg["FWD_BLOCK_D"],
        BLOCK_K=cfg["FWD_BLOCK_K"],
        BLOCK_T=cfg["BLOCK_T"],
        USE_FP16=cfg["use_fp16"],
        USE_BF16=cfg["use_bf16"],
        USE_FP32_STATS=cfg["stats_fp32"],
        INPUT_PRECISION=cfg["input_precision"],
        SINGLE_WRITER_OUTPUT=single_writer_output,
        WEIGHT_SHARING_ENC_DEC=weight_sharing,
        H=h,
        num_warps=cfg["fwd_num_warps"],
        num_stages=cfg["fwd_num_stages"],
    )
    torch.cuda.synchronize()
    return {
        "flare_chunk_prepare": prepare_kernel,
        "flare_chunk_prefix": prefix_kernel,
        "flare_chunk_decoder_lse": decoder_lse_kernel,
        "flare_chunk_fwd": fwd_kernel,
    }


def collect_kernel_profiles(compiled_kernels: dict[str, object], device_idx: int) -> dict[str, KernelProfile]:
    triton_props = driver.active.utils.get_device_properties(device_idx)
    device_props = {
        "max_shared_mem": int(triton_props["max_shared_mem"]),
        "max_num_regs": int(triton_props["max_num_regs"]),
        "warp_size": int(triton_props["warpSize"]),
        "max_threads_per_sm": int(torch.cuda.get_device_properties(device_idx).max_threads_per_multi_processor),
    }
    profiles = {}
    for name, compiled in compiled_kernels.items():
        occupancy_pct, active_ctas, limiting = estimate_occupancy(compiled, device_props)
        profiles[name] = KernelProfile(
            name=name,
            num_warps=int(compiled.metadata.num_warps),
            num_stages=int(compiled.metadata.num_stages),
            regs_per_thread=int(compiled.n_regs),
            spills=int(compiled.n_spills),
            shared_bytes=int(compiled.metadata.shared),
            threads_per_cta=int(compiled.metadata.num_warps * device_props["warp_size"]),
            occupancy_pct_est=occupancy_pct,
            active_ctas_per_sm_est=active_ctas,
            limiting_factors=limiting,
        )
    return profiles


def collect_recorded_kernel_profiles(recorded_resources: dict[str, dict[str, int]], device_idx: int) -> dict[str, KernelProfile]:
    triton_props = driver.active.utils.get_device_properties(device_idx)
    device_props = {
        "max_shared_mem": int(triton_props["max_shared_mem"]),
        "max_num_regs": int(triton_props["max_num_regs"]),
        "warp_size": int(triton_props["warpSize"]),
        "max_threads_per_sm": int(torch.cuda.get_device_properties(device_idx).max_threads_per_multi_processor),
    }
    profiles = {}
    for name, resource in recorded_resources.items():
        occupancy_pct, active_ctas, limiting = estimate_occupancy_from_resource_dict(resource, device_props)
        profiles[name] = KernelProfile(
            name=name,
            num_warps=int(resource["num_warps"]),
            num_stages=int(resource["num_stages"]),
            regs_per_thread=int(resource["regs_per_thread"]),
            spills=int(resource["spills"]),
            shared_bytes=int(resource["shared_bytes"]),
            threads_per_cta=int(int(resource["num_warps"]) * device_props["warp_size"]),
            occupancy_pct_est=occupancy_pct,
            active_ctas_per_sm_est=active_ctas,
            limiting_factors=limiting,
        )
    return profiles


def average_profile_timings(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    input_precision: str | None,
    include_backward: bool,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, int]]]:
    def zero_grads() -> None:
        if q.grad is not None:
            q.grad.zero_()
        if k.grad is not None:
            k.grad.zero_()
        if v.grad is not None:
            v.grad.zero_()

    def run_once(profile: bool) -> dict[str, object] | None:
        zero_grads()
        result = flare_chunk_triton(q, k, v, input_precision=input_precision, profile=profile)
        if profile:
            out, profile_data = result
        else:
            out = result
            profile_data = None
        if include_backward:
            out.float().sum().backward()
        return profile_data

    for _ in range(warmup):
        run_once(profile=False)
    torch.cuda.synchronize()

    forward_totals: dict[str, float] = defaultdict(float)
    backward_totals: dict[str, float] = defaultdict(float)
    backward_resources: dict[str, dict[str, int]] = {}
    for _ in range(iterations):
        profile = run_once(profile=True)
        assert profile is not None
        for key, value in profile["forward"].items():
            forward_totals[key] += float(value)
        for key, value in profile["backward"].items():
            backward_totals[key] += float(value)
        if include_backward:
            backward_resources = {
                name: dict(resource)
                for name, resource in profile.get("backward_resources", {}).items()
            }
    return (
        {key: value / iterations for key, value in forward_totals.items()},
        {key: value / iterations for key, value in backward_totals.items()},
        backward_resources,
    )


def bench_end_to_end_ms(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    input_precision: str | None,
    include_backward: bool,
) -> float:
    def op():
        if q.grad is not None:
            q.grad.zero_()
        if k.grad is not None:
            k.grad.zero_()
        if v.grad is not None:
            v.grad.zero_()
        out = flare_chunk_triton(q, k, v, input_precision=input_precision)
        if include_backward:
            out.float().sum().backward()

    return float(triton.testing.do_bench(op))


def bench_case(
    case: Case,
    *,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    input_precision: str | None,
    mode: str,
    env: dict[str, str],
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    device_idx = device.index or 0
    include_backward = mode in {"backward", "both"}

    q = torch.randn(
        case.num_heads,
        case.latent_queries,
        case.head_dim,
        device=device,
        dtype=dtype,
        requires_grad=include_backward,
    )
    k = torch.randn(
        case.batch_size,
        case.seq_len,
        case.num_heads,
        case.head_dim,
        device=device,
        dtype=dtype,
        requires_grad=include_backward,
    )
    v = torch.randn(
        case.batch_size,
        case.seq_len,
        case.num_heads,
        case.head_dim,
        device=device,
        dtype=dtype,
        requires_grad=include_backward,
    )

    with temp_env(env):
        try:
            compiled_kernels = compile_forward_kernels(q, k, v, input_precision=input_precision)
            kernel_profiles = collect_kernel_profiles(compiled_kernels, device_idx)
        except KeyError:
            # The standalone forward compile profiler still expects some
            # pre-autotune config fields that the runtime no longer exposes.
            # Keep timing sweeps running and omit only the compiled resource
            # metadata until the profiler path is brought back in sync.
            compiled_kernels = {}
            kernel_profiles = {}
        avg_forward_timings, avg_backward_timings, backward_resources = average_profile_timings(
            q,
            k,
            v,
            warmup=warmup,
            iterations=iterations,
            input_precision=input_precision,
            include_backward=include_backward,
        )
        end_to_end_ms = bench_end_to_end_ms(
            q,
            k,
            v,
            input_precision=input_precision,
            include_backward=include_backward,
        )

    forward_total_ms = sum(avg_forward_timings.values())
    backward_total_ms = sum(avg_backward_timings.values())
    backward_kernel_profiles = collect_recorded_kernel_profiles(backward_resources, device_idx) if backward_resources else {}
    forward_rows = []
    for name, timing_ms in sorted(avg_forward_timings.items(), key=lambda item: item[1], reverse=True):
        row = {
            "name": name,
            "avg_ms": timing_ms,
            "pct_of_forward_profile": 100.0 * timing_ms / forward_total_ms if forward_total_ms else 0.0,
        }
        if name in kernel_profiles:
            row.update(asdict(kernel_profiles[name]))
        forward_rows.append(row)

    backward_rows = []
    for name, timing_ms in sorted(avg_backward_timings.items(), key=lambda item: item[1], reverse=True):
        row = {
            "name": name,
            "avg_ms": timing_ms,
            "pct_of_backward_profile": 100.0 * timing_ms / backward_total_ms if backward_total_ms else 0.0,
        }
        if name in backward_kernel_profiles:
            row.update(asdict(backward_kernel_profiles[name]))
        backward_rows.append(row)

    return {
        "case": asdict(case),
        "dtype": str(dtype).removeprefix("torch."),
        "input_precision": input_precision or "default",
        "mode": mode,
        "env": dict(env),
        "end_to_end_ms": end_to_end_ms,
        "forward_profile_total_ms": forward_total_ms,
        "backward_profile_total_ms": backward_total_ms,
        "forward_kernels": forward_rows,
        "backward_kernels": backward_rows,
    }


def select_configs(args) -> dict[str, dict[str, str]]:
    if args.config:
        return dict(parse_config(spec) for spec in args.config)
    return dict(DEFAULT_CONFIGS)


def select_cases(args) -> list[Case]:
    if args.case:
        return [parse_case(spec) for spec in args.case]
    return list(DEFAULT_CASES)


def print_summary(results: list[dict]) -> None:
    for result in results:
        case = result["case"]
        label = (
            f"{case['name']} B={case['batch_size']} H={case['num_heads']} "
            f"N={case['seq_len']} M={case['latent_queries']} D={case['head_dim']}"
        )
        env = result["env"] or {}
        env_label = ", ".join(f"{key}={value}" for key, value in sorted(env.items())) or "default"
        print(f"\n== {label} | config={env_label}")
        print(
            f"end_to_end_ms={result['end_to_end_ms']:.3f} "
            f"forward_profile_total_ms={result['forward_profile_total_ms']:.3f} "
            f"backward_profile_total_ms={result['backward_profile_total_ms']:.3f}"
        )
        print("forward kernels")
        print(
            "kernel".ljust(28),
            "avg_ms".rjust(10),
            "%fwd".rjust(8),
            "regs".rjust(8),
            "smem_kb".rjust(10),
            "occ%".rjust(8),
            "cta/sm".rjust(8),
            "limit".rjust(16),
        )
        for kernel in result["forward_kernels"]:
            limit = ",".join(kernel.get("limiting_factors", []))
            print(
                kernel["name"].ljust(28),
                f"{kernel['avg_ms']:.3f}".rjust(10),
                f"{kernel['pct_of_forward_profile']:.1f}".rjust(8),
                f"{kernel.get('regs_per_thread', 0)}".rjust(8),
                f"{kernel.get('shared_bytes', 0) / 1024:.1f}".rjust(10),
                f"{kernel.get('occupancy_pct_est', 0.0):.1f}".rjust(8),
                f"{kernel.get('active_ctas_per_sm_est', 0)}".rjust(8),
                limit.rjust(16),
            )
        if result["backward_kernels"]:
            print("backward kernels")
            print(
                "kernel".ljust(36),
                "avg_ms".rjust(10),
                "%bwd".rjust(8),
                "regs".rjust(8),
                "smem_kb".rjust(10),
                "occ%".rjust(8),
                "cta/sm".rjust(8),
                "limit".rjust(16),
            )
            for kernel in result["backward_kernels"]:
                limit = ",".join(kernel.get("limiting_factors", []))
                print(
                    kernel["name"].ljust(36),
                    f"{kernel['avg_ms']:.3f}".rjust(10),
                    f"{kernel['pct_of_backward_profile']:.1f}".rjust(8),
                    f"{kernel.get('regs_per_thread', 0)}".rjust(8),
                    f"{kernel.get('shared_bytes', 0) / 1024:.1f}".rjust(10),
                    f"{kernel.get('occupancy_pct_est', 0.0):.1f}".rjust(8),
                    f"{kernel.get('active_ctas_per_sm_est', 0)}".rjust(8),
                    limit.rjust(16),
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile chunked FLARE per-kernel timings and compiled resource usage.")
    parser.add_argument(
        "--case",
        action="append",
        help="Repeatable. Format: name,batch,heads,seq_len,latent_queries,head_dim",
    )
    parser.add_argument(
        "--config",
        action="append",
        help="Repeatable. Format: name:key=value,key=value. Use name: for the default env.",
    )
    parser.add_argument("--dtype", default="bf16", help="bf16, fp16, or fp32.")
    parser.add_argument("--input-precision", default=None, help="ieee, tf32, or tf32x3. Default uses repo default.")
    parser.add_argument(
        "--mode",
        choices=("forward", "backward", "both"),
        default="both",
        help="Profile forward only, or include backward timings in the run.",
    )
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="Optional output path for JSON results.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for chunked FLARE profiling.")

    dtype = parse_dtype(args.dtype)
    cases = select_cases(args)
    configs = select_configs(args)
    results = []
    for case in cases:
        for _, env in configs.items():
            result = bench_case(
                case,
                dtype=dtype,
                warmup=args.warmup,
                iterations=args.iterations,
                input_precision=args.input_precision,
                mode=args.mode,
                env=env,
                seed=args.seed,
            )
            result["config_name"] = next(name for name, cfg in configs.items() if cfg == env)
            results.append(result)

    print_summary(results)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

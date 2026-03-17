#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
import triton
import triton.testing

try:
    from .profile_chunked_flare import (
        Case,
        collect_kernel_profiles,
        compile_forward_kernels,
        parse_case,
        parse_dtype,
        temp_env,
    )
except ImportError:
    from profile_chunked_flare import (
        Case,
        collect_kernel_profiles,
        compile_forward_kernels,
        parse_case,
        parse_dtype,
        temp_env,
    )

from causal_flare import flare_autoregressive_decode_triton, flare_autoregressive_prefill_triton
from causal_flare.autoregressive.inference import _get_decode_step_config, flare_recurrent_step_kernel


DEFAULT_CASES = [
    Case("h16_d32", batch_size=8, num_heads=16, seq_len=4096, latent_queries=64, head_dim=32),
    Case("h4_d128", batch_size=8, num_heads=4, seq_len=4096, latent_queries=64, head_dim=128),
    Case("h4_d256", batch_size=8, num_heads=4, seq_len=4096, latent_queries=64, head_dim=256),
]


def clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in state.items()}


def compile_decode_kernel(
    q: torch.Tensor,
    k_step: torch.Tensor,
    v_step: torch.Tensor,
    *,
    base_state: dict[str, torch.Tensor],
) -> dict[str, object]:
    b, h, d = k_step.shape
    m = q.shape[1]
    cfg = _get_decode_step_config(m, d, d)
    state = clone_state(base_state)
    q_comp = q.contiguous().float()
    k_comp = k_step.contiguous().float()
    v_comp = v_step.contiguous().float()
    q_dec_step = k_comp
    k_dec_latent = q_comp
    y = torch.empty((b, h, d), device=k_step.device, dtype=torch.float32)
    compiled = flare_recurrent_step_kernel[(b * h, triton.cdiv(d, int(cfg["BLOCK_D"])))](
        q_comp,
        k_comp,
        v_comp,
        q_dec_step,
        k_dec_latent,
        state["m"],
        state["d"],
        state["u"],
        y,
        y,
        q_comp.stride(0),
        q_comp.stride(1),
        q_comp.stride(2),
        k_comp.stride(0),
        k_comp.stride(1),
        k_comp.stride(2),
        v_comp.stride(0),
        v_comp.stride(1),
        v_comp.stride(2),
        q_dec_step.stride(0),
        q_dec_step.stride(1),
        q_dec_step.stride(2),
        k_dec_latent.stride(0),
        k_dec_latent.stride(1),
        k_dec_latent.stride(2),
        state["m"].stride(0),
        state["m"].stride(1),
        state["m"].stride(2),
        state["d"].stride(0),
        state["d"].stride(1),
        state["d"].stride(2),
        state["u"].stride(0),
        state["u"].stride(1),
        state["u"].stride(2),
        state["u"].stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        0,
        b,
        h,
        m,
        d,
        d,
        float(d ** -0.5 if d > 8 else 1.0),
        HAS_MASK=False,
        BLOCK_M=int(cfg["BLOCK_M"]),
        BLOCK_D=int(cfg["BLOCK_D"]),
        BLOCK_K=int(cfg["BLOCK_K"]),
        WEIGHT_SHARING_ENC_DEC=True,
        num_warps=int(cfg["num_warps"]),
        num_stages=int(cfg["num_stages"]),
    )
    torch.cuda.synchronize()
    return {"flare_decode_step": compiled}


def average_prefill_profile_timings(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
    input_precision: str | None,
) -> dict[str, float]:
    def run_once(profile: bool) -> dict[str, object] | None:
        result = flare_autoregressive_prefill_triton(q, k, v, input_precision=input_precision, profile=profile)
        if profile:
            (_, _), profile_data = result
            return profile_data
        return None

    for _ in range(warmup):
        run_once(profile=False)
    torch.cuda.synchronize()

    forward_totals: dict[str, float] = defaultdict(float)
    for _ in range(iterations):
        profile = run_once(profile=True)
        assert profile is not None
        for key, value in profile["forward"].items():
            forward_totals[key] += float(value)
    return {key: value / iterations for key, value in forward_totals.items()}


def average_decode_profile_timings(
    q: torch.Tensor,
    k_steps: torch.Tensor,
    v_steps: torch.Tensor,
    *,
    base_state: dict[str, torch.Tensor],
    warmup: int,
    iterations: int,
    input_precision: str | None,
) -> dict[str, float]:
    decode_steps = k_steps.shape[0]

    def run_once(profile: bool) -> dict[str, float] | None:
        state = clone_state(base_state)
        totals: dict[str, float] = defaultdict(float)
        for step_idx in range(decode_steps):
            result = flare_autoregressive_decode_triton(
                q,
                k_steps[step_idx],
                v_steps[step_idx],
                state=state,
                input_precision=input_precision,
                profile=profile,
            )
            if profile:
                (_, state), profile_data = result
                for key, value in profile_data["forward"].items():
                    totals[key] += float(value)
            else:
                _, state = result
        if not profile:
            return None
        return {key: value / decode_steps for key, value in totals.items()}

    for _ in range(warmup):
        run_once(profile=False)
    torch.cuda.synchronize()

    accumulated: dict[str, float] = defaultdict(float)
    for _ in range(iterations):
        profile = run_once(profile=True)
        assert profile is not None
        for key, value in profile.items():
            accumulated[key] += float(value)
    return {key: value / iterations for key, value in accumulated.items()}


def bench_prefill_end_to_end_ms(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    input_precision: str | None,
) -> float:
    return float(
        triton.testing.do_bench(
            lambda: flare_autoregressive_prefill_triton(q, k, v, input_precision=input_precision)
        )
    )


def bench_decode_end_to_end_ms(
    q: torch.Tensor,
    k_steps: torch.Tensor,
    v_steps: torch.Tensor,
    *,
    base_state: dict[str, torch.Tensor],
    input_precision: str | None,
) -> float:
    def op():
        state = clone_state(base_state)
        for step_idx in range(k_steps.shape[0]):
            _, state = flare_autoregressive_decode_triton(
                q,
                k_steps[step_idx],
                v_steps[step_idx],
                state=state,
                input_precision=input_precision,
            )

    return float(triton.testing.do_bench(op))


def select_cases(args) -> list[Case]:
    if args.case:
        return [parse_case(spec) for spec in args.case]
    return list(DEFAULT_CASES)


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
    decode_steps: int,
) -> list[dict]:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    device_idx = device.index or 0
    q = torch.randn(case.num_heads, case.latent_queries, case.head_dim, device=device, dtype=dtype)
    k = torch.randn(case.batch_size, case.seq_len, case.num_heads, case.head_dim, device=device, dtype=dtype)
    v = torch.randn(case.batch_size, case.seq_len, case.num_heads, case.head_dim, device=device, dtype=dtype)
    requested_modes = ("prefill", "decode") if mode == "both" else (mode,)
    results: list[dict] = []

    with temp_env(env):
        if "prefill" in requested_modes:
            compiled_kernels = compile_forward_kernels(q, k, v, input_precision=input_precision)
            kernel_profiles = collect_kernel_profiles(compiled_kernels, device_idx)
            avg_forward_timings = average_prefill_profile_timings(
                q,
                k,
                v,
                warmup=warmup,
                iterations=iterations,
                input_precision=input_precision,
            )
            end_to_end_ms = bench_prefill_end_to_end_ms(q, k, v, input_precision=input_precision)
            total_ms = sum(avg_forward_timings.values())
            rows = []
            for name, timing_ms in sorted(avg_forward_timings.items(), key=lambda item: item[1], reverse=True):
                row = {
                    "name": name,
                    "avg_ms": timing_ms,
                    "pct_of_profile": 100.0 * timing_ms / total_ms if total_ms else 0.0,
                }
                if name in kernel_profiles:
                    row.update(asdict(kernel_profiles[name]))
                rows.append(row)
            results.append(
                {
                    "case": asdict(case),
                    "benchmark_mode": "prefill",
                    "dtype": str(dtype).removeprefix("torch."),
                    "input_precision": input_precision or "default",
                    "env": dict(env),
                    "end_to_end_ms": end_to_end_ms,
                    "ms_per_token": end_to_end_ms / case.seq_len,
                    "profile_total_ms": total_ms,
                    "kernels": rows,
                }
            )

        if "decode" in requested_modes:
            k_prompt = torch.randn(case.batch_size, case.seq_len, case.num_heads, case.head_dim, device=device, dtype=dtype)
            v_prompt = torch.randn(case.batch_size, case.seq_len, case.num_heads, case.head_dim, device=device, dtype=dtype)
            k_steps = torch.randn(decode_steps, case.batch_size, case.num_heads, case.head_dim, device=device, dtype=dtype)
            v_steps = torch.randn(decode_steps, case.batch_size, case.num_heads, case.head_dim, device=device, dtype=dtype)
            _, base_state = flare_autoregressive_prefill_triton(q, k_prompt, v_prompt, input_precision=input_precision)
            compiled_decode = compile_decode_kernel(
                q,
                k_steps[0],
                v_steps[0],
                base_state=base_state,
            )
            decode_profiles = collect_kernel_profiles(compiled_decode, device_idx)
            avg_forward_timings = average_decode_profile_timings(
                q,
                k_steps,
                v_steps,
                base_state=base_state,
                warmup=warmup,
                iterations=iterations,
                input_precision=input_precision,
            )
            end_to_end_ms = bench_decode_end_to_end_ms(
                q,
                k_steps,
                v_steps,
                base_state=base_state,
                input_precision=input_precision,
            )
            total_ms = sum(avg_forward_timings.values())
            rows = []
            for name, timing_ms in sorted(avg_forward_timings.items(), key=lambda item: item[1], reverse=True):
                row = {
                    "name": name,
                    "avg_ms": timing_ms,
                    "pct_of_profile": 100.0 * timing_ms / total_ms if total_ms else 0.0,
                }
                if name in decode_profiles:
                    row.update(asdict(decode_profiles[name]))
                rows.append(row)
            results.append(
                {
                    "case": asdict(case),
                    "benchmark_mode": "decode",
                    "dtype": str(dtype).removeprefix("torch."),
                    "input_precision": input_precision or "default",
                    "env": dict(env),
                    "decode_steps": decode_steps,
                    "end_to_end_ms": end_to_end_ms,
                    "ms_per_token": end_to_end_ms / decode_steps,
                    "profile_total_ms": total_ms,
                    "kernels": rows,
                }
            )
    return results


def print_summary(results: list[dict]) -> None:
    for result in results:
        case = result["case"]
        label = (
            f"{result['benchmark_mode']} {case['name']} B={case['batch_size']} H={case['num_heads']} "
            f"N={case['seq_len']} M={case['latent_queries']} D={case['head_dim']}"
        )
        env = result["env"] or {}
        env_label = ", ".join(f"{key}={value}" for key, value in sorted(env.items())) or "default"
        print(f"\n== {label} | config={env_label}")
        print(
            f"end_to_end_ms={result['end_to_end_ms']:.3f} "
            f"ms_per_token={result['ms_per_token']:.6f} "
            f"profile_total_ms={result['profile_total_ms']:.3f}"
        )
        print(
            "kernel".ljust(30),
            "avg_ms".rjust(10),
            "%prof".rjust(8),
            "regs".rjust(8),
            "smem_kb".rjust(10),
            "occ%".rjust(8),
            "cta/sm".rjust(8),
            "limit".rjust(16),
        )
        for kernel in result["kernels"]:
            limit = ",".join(kernel.get("limiting_factors", []))
            print(
                kernel["name"].ljust(30),
                f"{kernel['avg_ms']:.3f}".rjust(10),
                f"{kernel['pct_of_profile']:.1f}".rjust(8),
                f"{kernel.get('regs_per_thread', 0)}".rjust(8),
                f"{kernel.get('shared_bytes', 0) / 1024:.1f}".rjust(10),
                f"{kernel.get('occupancy_pct_est', 0.0):.1f}".rjust(8),
                f"{kernel.get('active_ctas_per_sm_est', 0)}".rjust(8),
                limit.rjust(16),
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile FLARE prefill/decode per-kernel timings and resource usage.")
    parser.add_argument(
        "--case",
        action="append",
        help="Repeatable. Format: name,batch,heads,seq_len,latent_queries,head_dim",
    )
    parser.add_argument("--dtype", default="bf16", help="bf16, fp16, or fp32.")
    parser.add_argument("--input-precision", default=None, help="ieee, tf32, or tf32x3.")
    parser.add_argument("--mode", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="Optional output path for JSON results.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FLARE inference profiling.")
    dtype = parse_dtype(args.dtype)
    cases = select_cases(args)
    results: list[dict] = []
    for case in cases:
        results.extend(
            bench_case(
                case,
                dtype=dtype,
                warmup=args.warmup,
                iterations=args.iterations,
                input_precision=args.input_precision,
                mode=args.mode,
                env={},
                seed=args.seed,
                decode_steps=args.decode_steps,
            )
        )

    print_summary(results)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

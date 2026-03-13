#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import torch
import triton
import triton.testing
from triton.runtime.driver import driver

from causal_flare import flare_recurrent_triton
from causal_flare._common import _resolve_attn_scale
from causal_flare.autoregressive.recurrent import (
    _alloc_recurrent_outputs,
    _launch_recurrent_decode_lse,
    _recurrent_multi_kernel_output,
    _recurrent_prepare_inputs,
    _resolve_recurrent_launch,
    flare_recurrent_assoc_lse_enc_kernel,
    flare_recurrent_decode_lse_kernel,
    flare_recurrent_fwd_kernel,
    flare_recurrent_multi_output_kernel,
    flare_recurrent_orig_fwd_kernel,
)

try:
    from .tuning_catalog import (
        RECURRENT_TUNING_FAMILY_GROUPS as FAMILY_GROUPS,
        CandidateConfig,
        build_recurrent_family_candidates,
    )
except ImportError:
    from tuning_catalog import (
        RECURRENT_TUNING_FAMILY_GROUPS as FAMILY_GROUPS,
        CandidateConfig,
        build_recurrent_family_candidates,
    )


DEFAULT_D_VALUES = (16, 32, 64, 96, 128, 192, 256)
DEFAULT_M_VALUES = (16, 32, 64, 96, 128, 192, 256, 384, 512)
DEFAULT_N_VALUES = (64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_BH_VALUES = (8, 16, 32, 64, 128)
DEFAULT_BLOCK_DS = (16, 32, 64, 128, 256)
DEFAULT_BLOCK_KS = (16, 32, 64, 128, 256)
DEFAULT_ORIG_BLOCK_TS = (1, 16, 32)
DEFAULT_MULTI_BLOCK_TS = (16, 32, 64, 128)
DEFAULT_BACKWARD_BLOCK_TS = (16, 32, 64, 128)


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
    grid: tuple[int, ...]
    num_warps: int
    num_stages: int
    regs_per_thread: int
    spills: int
    shared_bytes: int
    threads_per_cta: int
    occupancy_pct_est: float
    active_ctas_per_sm_est: int
    limiting_factors: list[str]


def parse_int_list(spec: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if spec is None:
        return default
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer in {spec!r}.")
    return tuple(values)


def parse_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s.lower()]


def canonical_case_name(batch_size: int, num_heads: int, seq_len: int, latent_queries: int, head_dim: int) -> str:
    return f"b{batch_size}_h{num_heads}_n{seq_len}_m{latent_queries}_d{head_dim}"


def build_cases(
    *,
    d_values: tuple[int, ...],
    m_values: tuple[int, ...],
    n_values: tuple[int, ...],
    bh_values: tuple[int, ...],
    batch_size: int,
) -> list[Case]:
    cases = []
    for bh in bh_values:
        if bh <= 0 or (bh % batch_size) != 0:
            raise ValueError(f"Each BH must be positive and divisible by batch_size={batch_size}. Got BH={bh}.")
        num_heads = bh // batch_size
        for seq_len in n_values:
            for latent_queries in m_values:
                for head_dim in d_values:
                    cases.append(
                        Case(
                            name=canonical_case_name(batch_size, num_heads, seq_len, latent_queries, head_dim),
                            batch_size=batch_size,
                            num_heads=num_heads,
                            seq_len=seq_len,
                            latent_queries=latent_queries,
                            head_dim=head_dim,
                        )
                    )
    return cases


def parse_family_list(spec: str | None) -> tuple[str, ...]:
    if not spec:
        return FAMILY_GROUPS["impl"] + FAMILY_GROUPS["forward_core"] + FAMILY_GROUPS["forward_multi"]
    requested: list[str] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if token in FAMILY_GROUPS:
            for family in FAMILY_GROUPS[token]:
                if family not in requested:
                    requested.append(family)
            continue
        if token not in FAMILY_GROUPS["all"] and token != "combined":
            raise ValueError(
                f"Unsupported family/group {token!r}. Expected one of {sorted(set(FAMILY_GROUPS['all']) | set(FAMILY_GROUPS) | {'combined'})}."
            )
        if token not in requested:
            requested.append(token)
    return tuple(requested)


def parse_extra_config(spec: str) -> CandidateConfig:
    if ":" not in spec:
        raise ValueError(f"Invalid --extra-config={spec!r}. Expected name:key=value,key=value.")
    name, raw_items = spec.split(":", 1)
    env = {}
    if raw_items.strip():
        for item in raw_items.split(","):
            key, value = item.split("=", 1)
            env[key.strip()] = value.strip()
    return CandidateConfig(family="extra", name=name.strip(), env=env)


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


def get_device_props(device_idx: int) -> dict[str, int]:
    tp = driver.active.utils.get_device_properties(device_idx)
    return {
        "max_shared_mem": int(tp["max_shared_mem"]),
        "max_num_regs": int(tp["max_num_regs"]),
        "warp_size": int(tp["warpSize"]),
        "max_threads_per_sm": int(torch.cuda.get_device_properties(device_idx).max_threads_per_multi_processor),
    }


def estimate_occupancy(compiled, props: dict[str, int]) -> tuple[float, int, list[str]]:
    meta = compiled.metadata
    threads = meta.num_warps * props["warp_size"]
    regs_cta = compiled.n_regs * threads
    smem_cta = meta.shared
    max_warps = props["max_threads_per_sm"] // props["warp_size"]
    limits = {
        "arch": 32,
        "threads": props["max_threads_per_sm"] // threads,
        "warps": max_warps // meta.num_warps,
        "regs": props["max_num_regs"] // regs_cta if regs_cta else 32,
        "shared": props["max_shared_mem"] // smem_cta if smem_cta else 32,
    }
    active = max(1, min(limits.values()))
    occ = active * meta.num_warps / max_warps * 100.0
    limiting = [k for k, value in limits.items() if value == active]
    return occ, active, limiting


def make_profile(name: str, grid: tuple[int, ...], compiled, props: dict[str, int]) -> KernelProfile:
    occ, ctas, limiting = estimate_occupancy(compiled, props)
    meta = compiled.metadata
    return KernelProfile(
        name=name,
        grid=grid,
        num_warps=int(meta.num_warps),
        num_stages=int(meta.num_stages),
        regs_per_thread=int(compiled.n_regs),
        spills=int(compiled.n_spills),
        shared_bytes=int(meta.shared),
        threads_per_cta=int(meta.num_warps * props["warp_size"]),
        occupancy_pct_est=round(occ, 2),
        active_ctas_per_sm_est=ctas,
        limiting_factors=limiting,
    )


def current_impl_name() -> str:
    impl = os.environ.get("FLARE_RECURRENT_IMPL", "").strip().lower()
    if impl == "multi":
        return "multi"
    if os.environ.get("FLARE_RECURRENT_ASSOC_SCAN", "0").strip() != "0":
        return "assoc_scan"
    return "orig"


def compile_forward_kernels(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float) -> tuple[dict[str, object], dict]:
    prepared = _recurrent_prepare_inputs(q, k, v, scale, None, None, None, None, None)
    _, lse_enc, lse_dec = _alloc_recurrent_outputs(prepared)
    props: dict[str, object] = {}

    decode_num_warps, decode_num_stages = _resolve_recurrent_launch(
        "decode",
        default_num_warps=prepared["base_num_warps"],
        default_num_stages=2,
    )
    grid_decode = (prepared["B"] * prepared["H"], triton.cdiv(prepared["T"], prepared["decode_block_t"]))
    decode_kernel = flare_recurrent_decode_lse_kernel.warmup(
        prepared["Q_dec_bhtd"], prepared["K_dec_bhmd"], lse_dec,
        prepared["Q_dec_bhtd"].stride(0), prepared["Q_dec_bhtd"].stride(1), prepared["Q_dec_bhtd"].stride(2), prepared["Q_dec_bhtd"].stride(3),
        prepared["K_dec_bhmd"].stride(0), prepared["K_dec_bhmd"].stride(1), prepared["K_dec_bhmd"].stride(2), prepared["K_dec_bhmd"].stride(3),
        lse_dec.stride(0), lse_dec.stride(1), lse_dec.stride(2),
        prepared["B"], prepared["H"], prepared["M"], prepared["T"], prepared["scale"],
        D=prepared["D"],
        BLOCK_T=prepared["decode_block_t"],
        BLOCK_M=prepared["block_m"],
        BLOCK_K=prepared["block_k"],
        INPUT_PRECISION=prepared["input_precision"],
        num_warps=decode_num_warps,
        num_stages=decode_num_stages,
        grid=grid_decode,
    )
    props["LSE_DEC"] = (decode_kernel, grid_decode)

    impl = current_impl_name()
    if impl == "multi":
        assoc_num_warps, assoc_num_stages = _resolve_recurrent_launch(
            "assoc_lse",
            default_num_warps=prepared["base_num_warps"],
            default_num_stages=2,
        )
        grid_assoc = (prepared["B"] * prepared["H"], prepared["num_m_tiles"])
        assoc_kernel = flare_recurrent_assoc_lse_enc_kernel.warmup(
            prepared["Q"], prepared["K_bhtd"], lse_enc,
            prepared["Q"].stride(0), prepared["Q"].stride(1), prepared["Q"].stride(2),
            prepared["K_bhtd"].stride(0), prepared["K_bhtd"].stride(1), prepared["K_bhtd"].stride(2), prepared["K_bhtd"].stride(3),
            lse_enc.stride(0), lse_enc.stride(1), lse_enc.stride(2), lse_enc.stride(3),
            prepared["B"], prepared["H"], prepared["M"], prepared["T"], prepared["scale"],
            D=prepared["D"],
            BLOCK_M=prepared["block_m"],
            BLOCK_K=prepared["block_k"],
            BLOCK_T=prepared["multi_assoc_block_t"],
            INPUT_PRECISION=prepared["input_precision"],
            num_warps=assoc_num_warps,
            num_stages=assoc_num_stages,
            grid=grid_assoc,
        )
        props["LSE_ENC"] = (assoc_kernel, grid_assoc)

        y = torch.empty((prepared["B"], prepared["H"], prepared["T"], prepared["D"]), device=q.device, dtype=q.dtype)
        replay_num_warps, replay_num_stages = _resolve_recurrent_launch(
            "replay",
            default_num_warps=prepared["base_num_warps"],
            default_num_stages=2,
        )
        grid_output = (prepared["B"] * prepared["H"], triton.cdiv(prepared["D"], prepared["block_d"]), prepared["num_m_tiles"])
        output_kernel = flare_recurrent_multi_output_kernel.warmup(
            prepared["Q"], prepared["K_bhtd"], prepared["V_bhtd"], lse_enc,
            prepared["Q_dec_bhtd"], prepared["K_dec_bhmd"], lse_dec, y,
            prepared["Q"].stride(0), prepared["Q"].stride(1), prepared["Q"].stride(2),
            prepared["K_bhtd"].stride(0), prepared["K_bhtd"].stride(1), prepared["K_bhtd"].stride(2), prepared["K_bhtd"].stride(3),
            prepared["V_bhtd"].stride(0), prepared["V_bhtd"].stride(1), prepared["V_bhtd"].stride(2), prepared["V_bhtd"].stride(3),
            lse_enc.stride(0), lse_enc.stride(1), lse_enc.stride(2), lse_enc.stride(3),
            prepared["Q_dec_bhtd"].stride(0), prepared["Q_dec_bhtd"].stride(1), prepared["Q_dec_bhtd"].stride(2), prepared["Q_dec_bhtd"].stride(3),
            prepared["K_dec_bhmd"].stride(0), prepared["K_dec_bhmd"].stride(1), prepared["K_dec_bhmd"].stride(2), prepared["K_dec_bhmd"].stride(3),
            lse_dec.stride(0), lse_dec.stride(1), lse_dec.stride(2),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            prepared["B"], prepared["H"], prepared["M"], prepared["T"], prepared["scale"],
            D=prepared["D"],
            BLOCK_M=prepared["block_m"],
            BLOCK_D=prepared["block_d"],
            BLOCK_K=prepared["block_k"],
            BLOCK_T=prepared["multi_replay_block_t"],
            INPUT_PRECISION=prepared["input_precision"],
            WEIGHT_SHARING_ENC_DEC=prepared["weight_sharing_enc_dec"],
            SINGLE_M_TILE=prepared["num_m_tiles"] == 1,
            num_warps=replay_num_warps,
            num_stages=replay_num_stages,
            grid=grid_output,
        )
        props["OUTPUT"] = (output_kernel, grid_output)
    else:
        fwd_num_warps, fwd_num_stages = _resolve_recurrent_launch(
            "fwd",
            default_num_warps=prepared["base_num_warps"],
            default_num_stages=2,
        )
        y, lse_enc_orig, lse_dec_orig = _alloc_recurrent_outputs(prepared)
        grid_fwd = (prepared["B"] * prepared["H"], triton.cdiv(prepared["D"], prepared["block_d"]), prepared["num_m_tiles"])
        kernel = flare_recurrent_orig_fwd_kernel if prepared["block_t"] == 1 else flare_recurrent_fwd_kernel
        fwd_kernel = kernel.warmup(
            prepared["Q"], prepared["K_bhtd"], prepared["V_bhtd"], prepared["Q_dec_bhtd"], prepared["K_dec_bhmd"], lse_dec_orig, y, lse_enc_orig,
            prepared["Q"].stride(0), prepared["Q"].stride(1), prepared["Q"].stride(2),
            prepared["K_bhtd"].stride(0), prepared["K_bhtd"].stride(1), prepared["K_bhtd"].stride(2), prepared["K_bhtd"].stride(3),
            prepared["V_bhtd"].stride(0), prepared["V_bhtd"].stride(1), prepared["V_bhtd"].stride(2), prepared["V_bhtd"].stride(3),
            prepared["Q_dec_bhtd"].stride(0), prepared["Q_dec_bhtd"].stride(1), prepared["Q_dec_bhtd"].stride(2), prepared["Q_dec_bhtd"].stride(3),
            prepared["K_dec_bhmd"].stride(0), prepared["K_dec_bhmd"].stride(1), prepared["K_dec_bhmd"].stride(2), prepared["K_dec_bhmd"].stride(3),
            lse_dec_orig.stride(0), lse_dec_orig.stride(1), lse_dec_orig.stride(2),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            lse_enc_orig.stride(0), lse_enc_orig.stride(1), lse_enc_orig.stride(2), lse_enc_orig.stride(3),
            prepared["B"], prepared["H"], prepared["M"], prepared["T"], prepared["scale"],
            D=prepared["D"],
            BLOCK_M=prepared["block_m"],
            BLOCK_D=prepared["block_d"],
            BLOCK_K=prepared["block_k"],
            BLOCK_T=prepared["block_t"],
            INPUT_PRECISION=prepared["input_precision"],
            WEIGHT_SHARING_ENC_DEC=prepared["weight_sharing_enc_dec"],
            SINGLE_M_TILE=prepared["num_m_tiles"] == 1,
            USE_ASSOC_SCAN=impl == "assoc_scan",
            num_warps=fwd_num_warps,
            num_stages=fwd_num_stages,
            grid=grid_fwd,
        )
        props["FWD_MAIN"] = (fwd_kernel, grid_fwd)

    torch.cuda.synchronize()
    return props, prepared


def collect_kernel_profiles(compiled_kernels: dict[str, tuple[object, tuple[int, ...]]], device_idx: int) -> dict[str, KernelProfile]:
    props = get_device_props(device_idx)
    return {
        name: make_profile(name, grid, compiled, props)
        for name, (compiled, grid) in compiled_kernels.items()
    }


def profile_forward_timings(prepared: dict, *, warmup: int, iterations: int) -> dict[str, float]:
    impl = current_impl_name()
    _, lse_enc, lse_dec = _alloc_recurrent_outputs(prepared)

    def bench(fn):
        return float(triton.testing.do_bench(fn, warmup=warmup, rep=iterations))

    timings = {
        "LSE_DEC": bench(lambda: _launch_recurrent_decode_lse(prepared, lse_dec)),
    }
    if impl == "multi":
        assoc_num_warps, assoc_num_stages = _resolve_recurrent_launch(
            "assoc_lse",
            default_num_warps=prepared["base_num_warps"],
            default_num_stages=2,
        )
        grid_assoc = (prepared["B"] * prepared["H"], prepared["num_m_tiles"])
        timings["LSE_ENC"] = bench(
            lambda: flare_recurrent_assoc_lse_enc_kernel[grid_assoc](
                prepared["Q"], prepared["K_bhtd"], lse_enc,
                prepared["Q"].stride(0), prepared["Q"].stride(1), prepared["Q"].stride(2),
                prepared["K_bhtd"].stride(0), prepared["K_bhtd"].stride(1), prepared["K_bhtd"].stride(2), prepared["K_bhtd"].stride(3),
                lse_enc.stride(0), lse_enc.stride(1), lse_enc.stride(2), lse_enc.stride(3),
                prepared["B"], prepared["H"], prepared["M"], prepared["T"], prepared["scale"],
                D=prepared["D"],
                BLOCK_M=prepared["block_m"],
                BLOCK_K=prepared["block_k"],
                BLOCK_T=prepared["multi_assoc_block_t"],
                INPUT_PRECISION=prepared["input_precision"],
                num_warps=assoc_num_warps,
                num_stages=assoc_num_stages,
            )
        )
        _launch_recurrent_decode_lse(prepared, lse_dec)
        flare_recurrent_assoc_lse_enc_kernel[grid_assoc](
            prepared["Q"], prepared["K_bhtd"], lse_enc,
            prepared["Q"].stride(0), prepared["Q"].stride(1), prepared["Q"].stride(2),
            prepared["K_bhtd"].stride(0), prepared["K_bhtd"].stride(1), prepared["K_bhtd"].stride(2), prepared["K_bhtd"].stride(3),
            lse_enc.stride(0), lse_enc.stride(1), lse_enc.stride(2), lse_enc.stride(3),
            prepared["B"], prepared["H"], prepared["M"], prepared["T"], prepared["scale"],
            D=prepared["D"],
            BLOCK_M=prepared["block_m"],
            BLOCK_K=prepared["block_k"],
            BLOCK_T=prepared["multi_assoc_block_t"],
            INPUT_PRECISION=prepared["input_precision"],
            num_warps=assoc_num_warps,
            num_stages=assoc_num_stages,
        )
        timings["OUTPUT"] = bench(lambda: _recurrent_multi_kernel_output(prepared, lse_enc, lse_dec))
    else:
        y, lse_enc_orig, lse_dec_orig = _alloc_recurrent_outputs(prepared)
        fwd_num_warps, fwd_num_stages = _resolve_recurrent_launch(
            "fwd",
            default_num_warps=prepared["base_num_warps"],
            default_num_stages=2,
        )
        grid_fwd = (prepared["B"] * prepared["H"], triton.cdiv(prepared["D"], prepared["block_d"]), prepared["num_m_tiles"])
        kernel = flare_recurrent_orig_fwd_kernel if prepared["block_t"] == 1 else flare_recurrent_fwd_kernel
        _launch_recurrent_decode_lse(prepared, lse_dec_orig)
        timings["FWD_MAIN"] = bench(
            lambda: kernel[grid_fwd](
                prepared["Q"], prepared["K_bhtd"], prepared["V_bhtd"], prepared["Q_dec_bhtd"], prepared["K_dec_bhmd"], lse_dec_orig, y, lse_enc_orig,
                prepared["Q"].stride(0), prepared["Q"].stride(1), prepared["Q"].stride(2),
                prepared["K_bhtd"].stride(0), prepared["K_bhtd"].stride(1), prepared["K_bhtd"].stride(2), prepared["K_bhtd"].stride(3),
                prepared["V_bhtd"].stride(0), prepared["V_bhtd"].stride(1), prepared["V_bhtd"].stride(2), prepared["V_bhtd"].stride(3),
                prepared["Q_dec_bhtd"].stride(0), prepared["Q_dec_bhtd"].stride(1), prepared["Q_dec_bhtd"].stride(2), prepared["Q_dec_bhtd"].stride(3),
                prepared["K_dec_bhmd"].stride(0), prepared["K_dec_bhmd"].stride(1), prepared["K_dec_bhmd"].stride(2), prepared["K_dec_bhmd"].stride(3),
                lse_dec_orig.stride(0), lse_dec_orig.stride(1), lse_dec_orig.stride(2),
                y.stride(0), y.stride(1), y.stride(2), y.stride(3),
                lse_enc_orig.stride(0), lse_enc_orig.stride(1), lse_enc_orig.stride(2), lse_enc_orig.stride(3),
                prepared["B"], prepared["H"], prepared["M"], prepared["T"], prepared["scale"],
                D=prepared["D"],
                BLOCK_M=prepared["block_m"],
                BLOCK_D=prepared["block_d"],
                BLOCK_K=prepared["block_k"],
                BLOCK_T=prepared["block_t"],
                INPUT_PRECISION=prepared["input_precision"],
                WEIGHT_SHARING_ENC_DEC=prepared["weight_sharing_enc_dec"],
                SINGLE_M_TILE=prepared["num_m_tiles"] == 1,
                USE_ASSOC_SCAN=impl == "assoc_scan",
                num_warps=fwd_num_warps,
                num_stages=fwd_num_stages,
            )
        )
    return timings


def bench_end_to_end_ms(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, scale: float, include_backward: bool) -> float:
    def op():
        if q.grad is not None:
            q.grad.zero_()
        if k.grad is not None:
            k.grad.zero_()
        if v.grad is not None:
            v.grad.zero_()
        out = flare_recurrent_triton(q, k, v, scale=scale, block_t=16)
        if include_backward:
            out.float().sum().backward()

    return float(triton.testing.do_bench(op))


def bench_case(
    case: Case,
    *,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    env: dict[str, str],
    seed: int,
    mode: str,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    device_idx = device.index or 0
    include_backward = mode in {"backward", "both"}
    q = torch.randn(case.num_heads, case.latent_queries, case.head_dim, device=device, dtype=dtype, requires_grad=include_backward)
    k = torch.randn(case.batch_size, case.seq_len, case.num_heads, case.head_dim, device=device, dtype=dtype, requires_grad=include_backward)
    v = torch.randn(case.batch_size, case.seq_len, case.num_heads, case.head_dim, device=device, dtype=dtype, requires_grad=include_backward)
    scale = _resolve_attn_scale(None, case.head_dim)

    with temp_env(env):
        compiled_kernels, prepared = compile_forward_kernels(q, k, v, scale=scale)
        kernel_profiles = collect_kernel_profiles(compiled_kernels, device_idx)
        forward_timings = profile_forward_timings(prepared, warmup=warmup, iterations=iterations)
        end_to_end_ms = bench_end_to_end_ms(q, k, v, scale=scale, include_backward=include_backward)
        impl = current_impl_name()

    forward_total_ms = sum(forward_timings.values())
    forward_rows = []
    for name, timing_ms in sorted(forward_timings.items(), key=lambda item: item[1], reverse=True):
        row = {
            "name": name,
            "avg_ms": timing_ms,
            "pct_of_forward_profile": 100.0 * timing_ms / forward_total_ms if forward_total_ms else 0.0,
        }
        if name in kernel_profiles:
            row.update(asdict(kernel_profiles[name]))
        forward_rows.append(row)

    return {
        "case": asdict(case),
        "dtype": str(dtype).removeprefix("torch."),
        "mode": mode,
        "impl": impl,
        "env": dict(env),
        "end_to_end_ms": end_to_end_ms,
        "forward_profile_total_ms": forward_total_ms,
        "forward_kernels": forward_rows,
    }


def result_key(case: Case, candidate: CandidateConfig, *, dtype: str, mode: str) -> tuple:
    return (
        case.batch_size,
        case.num_heads,
        case.seq_len,
        case.latent_queries,
        case.head_dim,
        dtype,
        mode,
        candidate.family,
        candidate.name,
        tuple(sorted(candidate.env.items())),
    )


def build_error_record(*, case: Case, candidate: CandidateConfig, dtype: str, mode: str, error: Exception) -> dict:
    return {
        "case": asdict(case),
        "dtype": dtype,
        "mode": mode,
        "impl": "error",
        "env": dict(candidate.env),
        "end_to_end_ms": float("inf"),
        "forward_profile_total_ms": float("inf"),
        "forward_kernels": [],
        "config_name": candidate.name,
        "family": candidate.family,
        "status": "error",
        "error": str(error)[:4000],
    }


def merge_family_winners(rows: list[dict], *, objective: str) -> CandidateConfig | None:
    merged_env: dict[str, str] = {}
    for family in sorted({row["family"] for row in rows if row["family"] not in {"default", "combined"}}):
        family_rows = [row for row in rows if row["family"] == family]
        if not family_rows:
            continue
        best = min(family_rows, key=lambda row: float(row[objective]))
        merged_env.update(best["env"])
    if not merged_env:
        return None
    return CandidateConfig("combined", "combined_best_by_family", merged_env)


def summarize_results(results: list[dict], *, objective: str) -> dict[str, object]:
    by_case = defaultdict(list)
    for row in results:
        by_case[row["case"]["name"]].append(row)

    per_case = []
    grouped = {"by_d": defaultdict(list), "by_m": defaultdict(list), "by_n": defaultdict(list), "by_bh": defaultdict(list)}
    for _case_name, rows in sorted(by_case.items()):
        rows_sorted = sorted(rows, key=lambda row: float(row[objective]))
        best = rows_sorted[0]
        default = next((row for row in rows if row["config_name"] == "default"), best)
        speedup = float(default[objective]) / float(best[objective]) if float(best[objective]) > 0 else float("inf")
        summary_row = {
            "case": best["case"],
            "best_config": best["config_name"],
            "best_family": best["family"],
            "best_impl": best["impl"],
            "best_value_ms": float(best[objective]),
            "default_value_ms": float(default[objective]),
            "speedup_vs_default": speedup,
            "best_env": best["env"],
            "best_hot_kernel": best["forward_kernels"][0] if best["forward_kernels"] else None,
        }
        per_case.append(summary_row)
        grouped["by_d"][best["case"]["head_dim"]].append(summary_row)
        grouped["by_m"][best["case"]["latent_queries"]].append(summary_row)
        grouped["by_n"][best["case"]["seq_len"]].append(summary_row)
        grouped["by_bh"][best["case"]["batch_size"] * best["case"]["num_heads"]].append(summary_row)

    grouped_summary = {}
    for group_name, buckets in grouped.items():
        grouped_summary[group_name] = {}
        for bucket, rows in sorted(buckets.items()):
            winner_counts = defaultdict(int)
            for row in rows:
                winner_counts[row["best_config"]] += 1
            grouped_summary[group_name][bucket] = {
                "num_cases": len(rows),
                "winner_counts": dict(sorted(winner_counts.items())),
            }
    return {"objective": objective, "num_results": len(results), "num_cases": len(per_case), "per_case": per_case, "grouped": grouped_summary}


def format_summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Recurrent FLARE Matrix Tuning Summary",
        "",
        f"- Objective: `{summary['objective']}`",
        f"- Cases: `{summary['num_cases']}`",
        "",
        "| Case | Impl | Best config | Best ms | Default ms | Speedup |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["per_case"]:
        case = row["case"]
        label = f"B={case['batch_size']} H={case['num_heads']} N={case['seq_len']} M={case['latent_queries']} D={case['head_dim']}"
        lines.append(
            f"| `{label}` | `{row['best_impl']}` | `{row['best_config']}` | {row['best_value_ms']:.6f} | "
            f"{row['default_value_ms']:.6f} | {row['speedup_vs_default']:.2f}x |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row))
        handle.write("\n")


def record_key(record: dict) -> tuple:
    case = record["case"]
    return (
        case["batch_size"],
        case["num_heads"],
        case["seq_len"],
        case["latent_queries"],
        case["head_dim"],
        record["dtype"],
        record["mode"],
        record["family"],
        record["config_name"],
        tuple(sorted(record["env"].items())),
    )


def load_existing_results(path: Path | None) -> dict[tuple, dict]:
    if path is None or not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records[record_key(row)] = row
    return records


def resolve_output_paths(
    *,
    run_name: str | None,
    jsonl_path: Path | None,
    summary_json_path: Path | None,
    summary_md_path: Path | None,
    shard_index: int,
    num_shards: int,
) -> tuple[str, Path, Path, Path]:
    if run_name is None:
        run_name = datetime.now().strftime("recurrent-%Y%m%d-%H%M%S")
    root = Path("results") / "recurrent_flare_matrix" / run_name
    shard_stem = f"shard{shard_index:03d}-of-{num_shards:03d}"
    return (
        run_name,
        jsonl_path or (root / f"runs-{shard_stem}.jsonl"),
        summary_json_path or (root / f"summary-{shard_stem}.json"),
        summary_md_path or (root / f"summary-{shard_stem}.md"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeatable RecurrentFLARE launch-config sweeps over a D/M/N/BH matrix.")
    parser.add_argument("--d-values", default=None)
    parser.add_argument("--m-values", default=None)
    parser.add_argument("--n-values", default=None)
    parser.add_argument("--bh-values", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-ds", default="16,32,64,128,256")
    parser.add_argument("--block-ks", default="16,32,64,128,256")
    parser.add_argument("--orig-block-ts", default="1,16,32")
    parser.add_argument("--multi-block-ts", default="16,32,64,128")
    parser.add_argument("--backward-block-ts", default="16,32,64,128")
    parser.add_argument("--families", default=None)
    parser.add_argument("--extra-config", action="append", default=[])
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--mode", choices=("forward", "both", "backward"), default="forward")
    parser.add_argument("--objective", choices=("end_to_end_ms", "forward_profile_total_ms"), default="end_to_end_ms")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dtype = parse_dtype(args.dtype)
    d_values = parse_int_list(args.d_values, DEFAULT_D_VALUES)
    m_values = parse_int_list(args.m_values, DEFAULT_M_VALUES)
    n_values = parse_int_list(args.n_values, DEFAULT_N_VALUES)
    bh_values = parse_int_list(args.bh_values, DEFAULT_BH_VALUES)
    block_ds = parse_int_list(args.block_ds, DEFAULT_BLOCK_DS)
    block_ks = parse_int_list(args.block_ks, DEFAULT_BLOCK_KS)
    orig_block_ts = parse_int_list(args.orig_block_ts, DEFAULT_ORIG_BLOCK_TS)
    multi_block_ts = parse_int_list(args.multi_block_ts, DEFAULT_MULTI_BLOCK_TS)
    backward_block_ts = parse_int_list(args.backward_block_ts, DEFAULT_BACKWARD_BLOCK_TS)
    families = parse_family_list(args.families)
    extra_configs = [parse_extra_config(spec) for spec in args.extra_config]

    cases = build_cases(
        d_values=d_values,
        m_values=m_values,
        n_values=n_values,
        bh_values=bh_values,
        batch_size=args.batch_size,
    )
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"Expected 0 <= shard_index < num_shards. Got {args.shard_index=} {args.num_shards=}.")
    cases = [case for index, case in enumerate(cases) if (index % args.num_shards) == args.shard_index]

    run_name, jsonl_path, summary_json_path, summary_md_path = resolve_output_paths(
        run_name=args.run_name,
        jsonl_path=args.jsonl,
        summary_json_path=args.summary_json,
        summary_md_path=args.summary_md,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    existing_by_key = load_existing_results(jsonl_path if args.resume else None)
    results_by_key = {record_key(record): record for record in existing_by_key.values()}
    dtype_name = str(dtype).removeprefix("torch.")

    for case in cases:
        family_candidates = build_recurrent_family_candidates(
            head_dim=case.head_dim,
            latent_queries=case.latent_queries,
            seq_len=case.seq_len,
            block_ds=block_ds,
            block_ks=block_ks,
            orig_block_ts=orig_block_ts,
            multi_block_ts=multi_block_ts,
            backward_block_ts=backward_block_ts,
        )
        for extra in extra_configs:
            family_candidates.setdefault(extra.family, []).append(extra)

        case_candidates: list[CandidateConfig] = []
        seen = set()
        for family in families:
            if family == "combined":
                continue
            for candidate in family_candidates.get(family, []):
                key = (candidate.family, tuple(sorted(candidate.env.items())))
                if key in seen:
                    continue
                seen.add(key)
                case_candidates.append(candidate)

        case_results: list[dict] = []
        for candidate in case_candidates:
            key = result_key(case, candidate, dtype=dtype_name, mode=args.mode)
            if args.resume and key in existing_by_key:
                record = existing_by_key[key]
            else:
                try:
                    record = bench_case(
                        case,
                        dtype=dtype,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        env=candidate.env,
                        seed=args.seed,
                        mode=args.mode,
                    )
                    record["config_name"] = candidate.name
                    record["family"] = candidate.family
                    record["status"] = "ok"
                    record["error"] = ""
                except Exception as exc:  # noqa: BLE001
                    record = build_error_record(case=case, candidate=candidate, dtype=dtype_name, mode=args.mode, error=exc)
                append_jsonl(jsonl_path, record)
                existing_by_key[key] = record
            case_results.append(record)
            results_by_key[record_key(record)] = record

        if "combined" in families:
            combined = merge_family_winners(case_results, objective=args.objective)
            if combined is not None:
                key = result_key(case, combined, dtype=dtype_name, mode=args.mode)
                if args.resume and key in existing_by_key:
                    record = existing_by_key[key]
                else:
                    try:
                        record = bench_case(
                            case,
                            dtype=dtype,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            env=combined.env,
                            seed=args.seed,
                            mode=args.mode,
                        )
                        record["config_name"] = combined.name
                        record["family"] = combined.family
                        record["status"] = "ok"
                        record["error"] = ""
                    except Exception as exc:  # noqa: BLE001
                        record = build_error_record(case=case, candidate=combined, dtype=dtype_name, mode=args.mode, error=exc)
                    append_jsonl(jsonl_path, record)
                    existing_by_key[key] = record
                results_by_key[record_key(record)] = record

    summary = summarize_results(list(results_by_key.values()), objective=args.objective)
    summary["run_name"] = run_name
    summary["mode"] = args.mode
    summary["dtype"] = dtype_name
    summary["families"] = list(families)
    summary["matrix"] = {
        "d_values": list(d_values),
        "m_values": list(m_values),
        "n_values": list(n_values),
        "bh_values": list(bh_values),
        "batch_size": args.batch_size,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }
    summary["artifacts"] = {
        "raw_jsonl": str(jsonl_path),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
    }
    print(json.dumps(summary, indent=2))
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md_path.parent.mkdir(parents=True, exist_ok=True)
    summary_md_path.write_text(format_summary_markdown(summary), encoding="utf-8")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, Callable, Iterable

import torch


@dataclass(frozen=True)
class BenchmarkStats:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float


@dataclass(frozen=True)
class SweepResult:
    name: str
    params: dict[str, Any]
    stats: BenchmarkStats | None
    error: str | None = None


@dataclass(frozen=True)
class BenchResult:
    name: str
    stats: BenchmarkStats | None
    error: str | None = None


@dataclass(frozen=True)
class PhaseSweepSpec:
    name: str
    candidates: tuple[tuple[str, dict[str, Any]], ...]
    build_callable: Callable[[dict[str, Any]], Callable[[], Any]]


def bench_cuda_callable(fn: Callable[[], Any], *, warmup: int, reps: int) -> BenchmarkStats:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for i in range(reps):
        starts[i].record()
        fn()
        ends[i].record()
    torch.cuda.synchronize()

    times = [float(starts[i].elapsed_time(ends[i])) for i in range(reps)]
    return BenchmarkStats(
        mean_ms=sum(times) / len(times),
        median_ms=float(median(times)),
        min_ms=min(times),
        max_ms=max(times),
    )


def sweep_cuda_configs(
    candidates: Iterable[tuple[str, dict[str, Any]]],
    build_callable: Callable[[dict[str, Any]], Callable[[], Any]],
    *,
    warmup: int,
    reps: int,
) -> list[SweepResult]:
    results: list[SweepResult] = []
    for name, params in candidates:
        try:
            fn = build_callable(params)
            stats = bench_cuda_callable(fn, warmup=warmup, reps=reps)
            results.append(SweepResult(name=name, params=dict(params), stats=stats))
        except Exception as exc:  # pragma: no cover - intentionally broad for tuning sweeps
            results.append(SweepResult(name=name, params=dict(params), stats=None, error=str(exc)))
    return results


def best_sweep_result(results: Iterable[SweepResult]) -> SweepResult | None:
    valid = [result for result in results if result.stats is not None]
    if not valid:
        return None
    return min(valid, key=lambda result: result.stats.mean_ms)


def tune_phase_specs(
    specs: Iterable[PhaseSweepSpec],
    *,
    warmup: int,
    reps: int,
) -> dict[str, list[SweepResult]]:
    return {
        spec.name: sweep_cuda_configs(spec.candidates, spec.build_callable, warmup=warmup, reps=reps)
        for spec in specs
    }


def benchmark_named_callables(
    variants: dict[str, Callable[[], Any]],
    *,
    warmup: int,
    reps: int,
) -> dict[str, BenchResult]:
    results: dict[str, BenchResult] = {}
    for name, fn in variants.items():
        try:
            results[name] = BenchResult(name=name, stats=bench_cuda_callable(fn, warmup=warmup, reps=reps))
        except Exception as exc:  # pragma: no cover - intentionally broad for benchmark sweeps
            results[name] = BenchResult(name=name, stats=None, error=str(exc))
    return results


def stats_dict(stats: BenchmarkStats) -> dict[str, float]:
    return asdict(stats)


def sweep_result_dict(result: SweepResult) -> dict[str, Any]:
    payload = {
        "name": result.name,
        "params": dict(result.params),
        "error": result.error,
    }
    if result.stats is not None:
        payload["stats"] = stats_dict(result.stats)
    else:
        payload["stats"] = None
    return payload


def bench_result_dict(result: BenchResult) -> dict[str, Any]:
    payload = {
        "error": result.error,
    }
    if result.stats is not None:
        payload["stats"] = stats_dict(result.stats)
    else:
        payload["stats"] = None
    return payload

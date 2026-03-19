#!/usr/bin/env python
"""
Fast benchmark for one forward+backward training step:
- FLARE chunk Triton (`flare_autoregressive_triton`) for multiple latent counts M
- FLARE stablemax Triton (`flare_autoregressive_stablemax_triton`) for multiple latent counts M
- FlashAttention2 Triton (`flash_attention2_triton_bnhd`)

Benchmarks are saved incrementally to raw JSONL so long runs are resumable.
Summary CSV/JSON are written separately and can be re-plotted without rerunning.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import torch
import triton.testing

from causal_flare import flare_autoregressive_triton
from causal_flare.autoregressive.stablemax_triton import flare_autoregressive_stablemax_triton


DEFAULT_SEQ_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_FLARE_M = [32, 64, 128, 256]
DEFAULT_PROVIDERS = ["fa2_triton", "flare_autoregressive_triton", "flare_autoregressive_stablemax_triton"]

PROVIDER_ORDER = {
    "fa2_triton": 0,
    "flare_autoregressive_triton": 1,
    "flare_autoregressive_stablemax_triton": 2,
}

PROVIDER_LABELS = {
    "fa2_triton": "FlashAttention2 Triton",
    "flare_autoregressive_triton": "FLARE Chunk Triton",
    "flare_autoregressive_stablemax_triton": "FLARE Stablemax Triton",
}

PROVIDER_STYLES = {
    ("fa2_triton", None): {"color": "#000000", "marker": "o", "linestyle": "-"},
    ("flare_autoregressive_triton", 32): {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
    ("flare_autoregressive_triton", 64): {"color": "#2ca02c", "marker": "s", "linestyle": "--"},
    ("flare_autoregressive_triton", 128): {"color": "#d62728", "marker": "s", "linestyle": "--"},
    ("flare_autoregressive_triton", 256): {"color": "#9467bd", "marker": "s", "linestyle": "--"},
    ("flare_autoregressive_stablemax_triton", 32): {"color": "#ff7f0e", "marker": "*", "linestyle": ":"},
    ("flare_autoregressive_stablemax_triton", 64): {"color": "#2ca02c", "marker": "*", "linestyle": ":"},
    ("flare_autoregressive_stablemax_triton", 128): {"color": "#d62728", "marker": "*", "linestyle": ":"},
    ("flare_autoregressive_stablemax_triton", 256): {"color": "#9467bd", "marker": "*", "linestyle": ":"},
}


@dataclass(frozen=True)
class BenchCase:
    provider: str
    n: int
    m: int | None


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists() and (parent / "causal_flare").exists():
            return parent
    raise RuntimeError(f"Unable to resolve repository root from {__file__}")


def parse_len_token(token: str) -> int:
    t = token.strip().lower().replace("_", "")
    if t.endswith("k"):
        return int(float(t[:-1]) * 1024)
    return int(t)


def parse_int_list(value: str) -> list[int]:
    tokens = [t for t in value.split(",") if t.strip()]
    if not tokens:
        return []
    return [parse_len_token(t) for t in tokens]


def parse_tokens_list(values: list[str] | None) -> list[int]:
    if values is None:
        return []
    out: list[int] = []
    for item in values:
        out.extend(parse_int_list(item))
    return out


def build_cases(seq_lengths: list[int], flare_m_list: list[int], providers: list[str]) -> list[BenchCase]:
    cases: list[BenchCase] = []
    for provider in providers:
        if provider == "fa2_triton":
            for n in seq_lengths:
                cases.append(BenchCase(provider=provider, n=n, m=None))
            continue
        for m in flare_m_list:
            for n in seq_lengths:
                cases.append(BenchCase(provider=provider, n=n, m=m))
    return cases


def _case_key(
    *,
    provider: str,
    n: int,
    m: int | None,
    batch_size: int,
    num_heads: int,
    score_head_dim: int,
    value_head_dim: int,
    dtype: str,
) -> tuple[Any, ...]:
    return (
        provider,
        int(n),
        None if m is None else int(m),
        int(batch_size),
        int(num_heads),
        int(score_head_dim),
        int(value_head_dim),
        dtype,
    )


def _case_key_from_row(row: dict[str, Any]) -> tuple[Any, ...]:
    score_head_dim = int(row.get("score_head_dim", row.get("head_dim")))
    value_head_dim = int(row.get("value_head_dim", score_head_dim))
    return _case_key(
        provider=str(row["provider"]),
        n=int(row["N"]),
        m=None if row.get("M") is None else int(row["M"]),
        batch_size=int(row["batch_size"]),
        num_heads=int(row["num_heads"]),
        score_head_dim=score_head_dim,
        value_head_dim=value_head_dim,
        dtype=str(row["dtype"]),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def summarize_latest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        by_key[_case_key_from_row(row)] = row
    data = list(by_key.values())
    data.sort(
        key=lambda r: (
            r["provider"],
            float("inf") if r["M"] is None else int(r["M"]),
            int(r["N"]),
            int(r.get("score_head_dim", r.get("head_dim"))),
            int(r.get("value_head_dim", r.get("score_head_dim", r.get("head_dim")))),
        )
    )
    return data


def write_summary_files(rows: list[dict[str, Any]], csv_path: Path, json_path: Path) -> pd.DataFrame:
    summary_rows = summarize_latest(rows)
    df = pd.DataFrame(summary_rows)
    if not df.empty:
        provider_order = df["provider"].map(lambda provider: PROVIDER_ORDER.get(provider, len(PROVIDER_ORDER)))
        df = (
            df.assign(_provider_order=provider_order)
            .sort_values(["_provider_order", "M", "N"], na_position="first")
            .drop(columns="_provider_order")
            .reset_index(drop=True)
        )
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    return df


def _clear_grads(tensors: list[torch.Tensor]) -> None:
    for t in tensors:
        if t.grad is not None:
            t.grad = None


def _sanitize_error_text(exc: Exception) -> str:
    text = str(exc).strip().replace("\n", "\\n").replace("\t", " ")
    return text[:4000]


def _build_flare_step_fn(
    *,
    b: int,
    h: int,
    n: int,
    m: int,
    d_score: int,
    d_value: int,
    dtype: torch.dtype,
    device: torch.device,
    scale: float,
    flare_chunk_size: int | None,
    flare_input_precision: str | None,
):
    q = torch.randn(h, m, d_score, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, n, h, d_score, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, n, h, d_value, device=device, dtype=dtype, requires_grad=True)
    tensors = [q, k, v]

    def run_step() -> None:
        _clear_grads(tensors)
        y = flare_autoregressive_triton(
            q,
            k,
            v,
            scale=scale,
            chunk_size=flare_chunk_size,
            input_precision=flare_input_precision,
        )
        loss = y.float().square().mean()
        loss.backward()

    return run_step


def _build_flare_stablemax_step_fn(
    *,
    b: int,
    h: int,
    n: int,
    m: int,
    d_score: int,
    d_value: int,
    dtype: torch.dtype,
    device: torch.device,
    scale: float,
    flare_chunk_size: int | None,
    flare_input_precision: str | None,
):
    q = torch.randn(h, m, d_score, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, n, h, d_score, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, n, h, d_value, device=device, dtype=dtype, requires_grad=True)
    tensors = [q, k, v]
    stablemax_chunk_size = flare_chunk_size
    if stablemax_chunk_size is None:
        if m >= 256:
            stablemax_chunk_size = 64 if d_score <= 32 else 128
        elif d_score <= 32 and m <= 64 and n >= 1024:
            stablemax_chunk_size = 64
        elif d_score <= 64 and m <= 128 and n >= 1024:
            stablemax_chunk_size = 128
        else:
            stablemax_chunk_size = 256 if n >= 256 else 32

    def run_step() -> None:
        _clear_grads(tensors)
        y = flare_autoregressive_stablemax_triton(
            q,
            k,
            v,
            scale=scale,
            chunk_size=stablemax_chunk_size,
            input_precision=flare_input_precision,
        )
        if isinstance(y, tuple):
            y = y[0]
        loss = y.float().square().mean()
        loss.backward()

    return run_step


def _build_fa2_step_fn(
    *,
    b: int,
    h: int,
    n: int,
    d: int,
    dtype: torch.dtype,
    device: torch.device,
    scale: float,
    warp_specialize: bool,
):
    from implementations.flash_attention2_triton import flash_attention2_triton_bnhd

    q = torch.randn(b, n, h, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(b, n, h, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(b, n, h, d, device=device, dtype=dtype, requires_grad=True)
    tensors = [q, k, v]

    def run_step() -> None:
        _clear_grads(tensors)
        y = flash_attention2_triton_bnhd(
            q,
            k,
            v,
            causal=True,
            sm_scale=scale,
            warp_specialize=warp_specialize,
        )
        loss = y.float().square().mean()
        loss.backward()

    return run_step


def benchmark_one_case(
    *,
    case: BenchCase,
    batch_size: int,
    num_heads: int,
    score_head_dim: int,
    value_head_dim: int,
    dtype: torch.dtype,
    dtype_name: str,
    device: torch.device,
    warmup: int,
    rep: int,
    precompile: bool,
    flare_chunk_size: int | None,
    flare_input_precision: str | None,
    fa2_warp_specialize: bool,
    run_tag: str,
    device_name: str,
) -> dict[str, Any]:
    scale = score_head_dim ** -0.5 if score_head_dim > 8 else 1.0
    t0 = time.time()
    base_row: dict[str, Any] = {
        "run_tag": run_tag,
        "timestamp": int(t0),
        "provider": case.provider,
        "N": case.n,
        "M": case.m,
        "batch_size": batch_size,
        "num_heads": num_heads,
        "score_head_dim": score_head_dim,
        "value_head_dim": value_head_dim,
        "dtype": dtype_name,
        "device": str(device),
        "device_name": device_name,
        "warmup": warmup,
        "rep": rep,
        "scale": scale,
        "status": "unknown",
        "median_ms": math.nan,
        "p20_ms": math.nan,
        "p80_ms": math.nan,
        "error": "",
    }
    if case.provider in {"flare_autoregressive_triton", "flare_autoregressive_stablemax_triton"}:
        base_row["flare_chunk_size"] = flare_chunk_size
        base_row["flare_input_precision"] = flare_input_precision
    if case.provider == "fa2_triton":
        base_row["fa2_warp_specialize"] = fa2_warp_specialize

    try:
        if case.provider == "flare_autoregressive_triton":
            if case.m is None:
                raise ValueError("FLARE case missing M.")
            run_step = _build_flare_step_fn(
                b=batch_size,
                h=num_heads,
                n=case.n,
                m=case.m,
                d_score=score_head_dim,
                d_value=value_head_dim,
                dtype=dtype,
                device=device,
                scale=scale,
                flare_chunk_size=flare_chunk_size,
                flare_input_precision=flare_input_precision,
            )
        elif case.provider == "flare_autoregressive_stablemax_triton":
            if case.m is None:
                raise ValueError("Stablemax FLARE case missing M.")
            run_step = _build_flare_stablemax_step_fn(
                b=batch_size,
                h=num_heads,
                n=case.n,
                m=case.m,
                d_score=score_head_dim,
                d_value=value_head_dim,
                dtype=dtype,
                device=device,
                scale=scale,
                flare_chunk_size=flare_chunk_size,
                flare_input_precision=flare_input_precision,
            )
        elif case.provider == "fa2_triton":
            run_step = _build_fa2_step_fn(
                b=batch_size,
                h=num_heads,
                n=case.n,
                d=score_head_dim,
                dtype=dtype,
                device=device,
                scale=scale,
                warp_specialize=fa2_warp_specialize,
            )
        else:
            raise ValueError(f"Unsupported provider {case.provider}")

        if precompile:
            run_step()
            torch.cuda.synchronize(device)

        p50_ms, p20_ms, p80_ms = triton.testing.do_bench(
            run_step,
            warmup=warmup,
            rep=rep,
            quantiles=[0.5, 0.2, 0.8],
        )
        base_row["status"] = "ok"
        base_row["median_ms"] = float(p50_ms)
        base_row["p20_ms"] = float(p20_ms)
        base_row["p80_ms"] = float(p80_ms)
    except RuntimeError as exc:
        err_text = _sanitize_error_text(exc)
        if "out of memory" in err_text.lower():
            base_row["status"] = "oom"
        else:
            base_row["status"] = "error"
        base_row["error"] = err_text
    except Exception as exc:  # noqa: BLE001
        base_row["status"] = "error"
        base_row["error"] = _sanitize_error_text(exc)
    finally:
        torch.cuda.empty_cache()

    base_row["elapsed_s"] = float(time.time() - t0)
    return base_row


def run_benchmark(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError(f"Expected CUDA device; got {device}.")

    raw_jsonl_path = output_dir / args.raw_jsonl
    csv_path = output_dir / args.summary_csv
    json_path = output_dir / args.summary_json

    all_rows = load_jsonl(raw_jsonl_path)
    existing_keys = {_case_key_from_row(row) for row in all_rows}

    cases = build_cases(
        seq_lengths=args.seq_lengths,
        flare_m_list=args.flare_m,
        providers=args.providers,
    )

    run_tag = f"run_{int(time.time())}"
    device_name = torch.cuda.get_device_name(device)
    print(f"Benchmark run_tag={run_tag}")
    print(f"Device: {device_name}")
    print(f"Cases requested: {len(cases)}")

    completed = 0
    skipped = 0
    for idx, case in enumerate(cases, start=1):
        key = _case_key(
            provider=case.provider,
            n=case.n,
            m=case.m,
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            score_head_dim=args.score_head_dim,
            value_head_dim=args.value_head_dim,
            dtype=args.dtype,
        )
        if (not args.rerun_existing) and key in existing_keys:
            skipped += 1
            print(
                f"[{idx:02d}/{len(cases)}] skip existing: provider={case.provider} N={case.n} M={case.m} "
                f"Dk={args.score_head_dim} Dv={args.value_head_dim}"
            )
            continue

        print(
            f"[{idx:02d}/{len(cases)}] run: provider={case.provider} N={case.n} M={case.m} "
            f"Dk={args.score_head_dim} Dv={args.value_head_dim}"
        )
        row = benchmark_one_case(
            case=case,
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            score_head_dim=args.score_head_dim,
            value_head_dim=args.value_head_dim,
            dtype=dtype,
            dtype_name=args.dtype,
            device=device,
            warmup=args.warmup,
            rep=args.rep,
            precompile=(not args.no_precompile),
            flare_chunk_size=args.flare_chunk_size,
            flare_input_precision=args.flare_input_precision,
            fa2_warp_specialize=args.fa2_warp_specialize,
            run_tag=run_tag,
            device_name=device_name,
        )
        append_jsonl(raw_jsonl_path, row)
        all_rows.append(row)
        existing_keys.add(key)
        completed += 1
        print(
            f"  -> status={row['status']} median={row['median_ms']:.3f}ms "
            f"p20={row['p20_ms']:.3f}ms p80={row['p80_ms']:.3f}ms"
        )
        if row["status"] != "ok" and row["error"]:
            print(f"     error: {row['error']}")

    df = write_summary_files(all_rows, csv_path=csv_path, json_path=json_path)
    print(f"Benchmark done. completed={completed}, skipped={skipped}")
    print(f"Raw timings:    {raw_jsonl_path}")
    print(f"Summary CSV:    {csv_path}")
    print(f"Summary JSON:   {json_path}")
    return df


def _curve_label(provider: str, m: int | None) -> str:
    label = PROVIDER_LABELS.get(provider, provider)
    if m is None:
        return label
    return f"{label} (M={m})"


def plot_summary(df: pd.DataFrame, png_path: Path, pdf_path: Path, title: str) -> None:
    if df.empty:
        raise ValueError("Summary dataframe is empty.")

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No successful benchmark rows to plot.")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    groups = []
    for provider in DEFAULT_PROVIDERS:
        sub = ok[ok["provider"] == provider]
        if sub.empty:
            continue
        if provider == "fa2_triton":
            groups.append((provider, None, sub))
        else:
            for m in sorted(sub["M"].dropna().astype(int).unique().tolist()):
                groups.append((provider, int(m), sub[sub["M"] == m]))

    for provider, m, sub in groups:
        sub = sub.sort_values("N")
        style = PROVIDER_STYLES.get((provider, m), {"color": "#7f7f7f", "marker": "o", "linestyle": "-"})
        label = _curve_label(provider, m)
        x = sub["N"].to_numpy()
        y = sub["median_ms"].to_numpy()
        y_lo = sub["p20_ms"].to_numpy()
        y_hi = sub["p80_ms"].to_numpy()

        ax.plot(
            x,
            y,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=style["color"],
            linewidth=2.0,
            markersize=8 if style["marker"] == "*" else 6,
            label=label,
        )
        ax.fill_between(x, y_lo, y_hi, color=style["color"], alpha=0.15)

    ax.set_title(title)
    ax.set_xlabel("Sequence length N")
    ax.set_ylabel("Time per train step (forward+backward), ms")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.grid(alpha=0.25, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=220)
    fig.savefig(pdf_path, dpi=220)
    plt.close(fig)


def run_plot(args: argparse.Namespace, output_dir: Path) -> None:
    csv_path = output_dir / args.summary_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    png_path = output_dir / args.plot_png
    pdf_path = output_dir / args.plot_pdf
    plot_summary(df, png_path=png_path, pdf_path=pdf_path, title=args.figure_title)
    print(f"Wrote plot PNG: {png_path}")
    print(f"Wrote plot PDF: {pdf_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark FLARE chunk Triton vs FA2 Triton train-step times across sequence lengths. "
            "FLARE supports score/value head-dimension splits; FA2 requires equal score/value dims."
        )
    )
    parser.add_argument("--mode", choices=["benchmark", "plot", "both"], default="benchmark")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--score-head-dim", type=int, default=32, help="Score/logit head dimension D_k used by Q and K.")
    parser.add_argument(
        "--value-head-dim",
        type=int,
        default=None,
        help="Value/output head dimension D_v used by V and Y. Default matches --score-head-dim.",
    )
    parser.add_argument("--head-dim", dest="score_head_dim", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--seq-lengths",
        nargs="+",
        default=None,
        help="Lengths (comma-separated and/or space-separated). Supports k suffix, e.g. 2048 4096 65k 128k",
    )
    parser.add_argument(
        "--flare-m",
        nargs="+",
        default=None,
        help="FLARE latent counts M (comma-separated and/or space-separated), e.g. 32 64 128 256",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        choices=DEFAULT_PROVIDERS,
        default=None,
        help=(
            "Benchmark only the selected providers. "
            "Defaults to all providers: fa2_triton flare_autoregressive_triton flare_autoregressive_stablemax_triton"
        ),
    )
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations for triton.testing.do_bench")
    parser.add_argument("--rep", type=int, default=5, help="Timed repetitions for triton.testing.do_bench")
    parser.add_argument("--no-precompile", action="store_true", help="Disable one compile warmup step before timing")
    parser.add_argument("--rerun-existing", action="store_true", help="Rerun cases even if already present in raw JSONL")
    parser.add_argument("--skip-fa2", action="store_true", help="Skip FlashAttention2 runs. Required for mixed D_k/D_v benchmarks.")
    parser.add_argument("--skip-flare", action="store_true")
    parser.add_argument("--skip-stablemax", action="store_true")
    parser.add_argument("--flare-chunk-size", type=int, default=None)
    parser.add_argument(
        "--flare-input-precision",
        choices=["ieee", "tf32", "tf32x3"],
        default=None,
        help="Optional FLARE input precision override.",
    )
    parser.add_argument("--fa2-warp-specialize", action="store_true", help="Enable FA2 warp specialization path")

    parser.add_argument("--output-dir", type=str, default="results/flare_fa2_train_step")
    parser.add_argument("--raw-jsonl", type=str, default="train_step_timings_raw.jsonl")
    parser.add_argument("--summary-csv", type=str, default="train_step_timings.csv")
    parser.add_argument("--summary-json", type=str, default="train_step_timings.json")
    parser.add_argument("--plot-png", type=str, default="train_step_timings.png")
    parser.add_argument("--plot-pdf", type=str, default="train_step_timings.pdf")
    parser.add_argument(
        "--figure-title",
        type=str,
        default="FLARE vs FLARE Stablemax vs FA2 Triton: one training step latency",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.seq_lengths = parse_tokens_list(args.seq_lengths) if args.seq_lengths is not None else list(DEFAULT_SEQ_LENGTHS)
    args.flare_m = parse_tokens_list(args.flare_m) if args.flare_m is not None else list(DEFAULT_FLARE_M)
    if args.providers is None:
        args.providers = list(DEFAULT_PROVIDERS)
    else:
        args.providers = list(dict.fromkeys(args.providers))
    if args.skip_fa2:
        args.providers = [provider for provider in args.providers if provider != "fa2_triton"]
    if args.skip_flare:
        args.providers = [provider for provider in args.providers if provider != "flare_autoregressive_triton"]
    if args.skip_stablemax:
        args.providers = [provider for provider in args.providers if provider != "flare_autoregressive_stablemax_triton"]
    if args.value_head_dim is None:
        args.value_head_dim = args.score_head_dim

    if args.num_heads <= 0 or args.score_head_dim <= 0 or args.value_head_dim <= 0:
        raise ValueError("num-heads, score-head-dim, and value-head-dim must be > 0.")
    if not args.providers:
        raise ValueError("No providers selected. Use --providers or remove skip flags.")
    if any(n <= 0 for n in args.seq_lengths):
        raise ValueError(f"All sequence lengths must be > 0, got {args.seq_lengths}")
    if any(m <= 0 for m in args.flare_m):
        raise ValueError(f"All FLARE M values must be > 0, got {args.flare_m}")

    # FA2 backward kernel in this module requires sequence length multiple of 128.
    if any((n % 128) != 0 for n in args.seq_lengths):
        raise ValueError(f"All sequence lengths must be multiples of 128 for FA2 Triton backward: {args.seq_lengths}")
    if any((m % 16) != 0 for m in args.flare_m):
        raise ValueError(f"All FLARE M values must be multiples of 16: {args.flare_m}")
    if "fa2_triton" in args.providers:
        if args.score_head_dim != args.value_head_dim:
            raise ValueError(
                "FA2 benchmarking fundamentally requires score_head_dim == value_head_dim. "
                "Use --skip-fa2 for mixed-dimension FLARE-only train-step benchmarks."
            )
        if args.score_head_dim not in {16, 32, 64, 128, 256}:
            raise ValueError(
                f"FA2 Triton expects score_head_dim in {{16, 32, 64, 128, 256}}; got {args.score_head_dim}"
            )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    root = _repo_root()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("benchmark", "both"):
        run_benchmark(args=args, output_dir=output_dir)
    if args.mode in ("plot", "both"):
        run_plot(args=args, output_dir=output_dir)


if __name__ == "__main__":
    main()

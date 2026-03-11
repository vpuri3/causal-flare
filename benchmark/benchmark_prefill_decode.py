#!/usr/bin/env python
"""
Fast benchmark for FLARE cached-inference methods:
- prefill: `flare_prefill_triton`
- decode: `flare_decode_triton`

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

from causal_flare import flare_decode_triton, flare_prefill_triton


DEFAULT_SEQ_LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536, 131072]
DEFAULT_FLARE_M = [32, 64, 128, 256]
DEFAULT_BENCH_MODES = ("prefill", "decode")


@dataclass(frozen=True)
class BenchCase:
    mode: str
    provider: str
    n: int
    m: int


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


def parse_modes(values: list[str] | None) -> list[str]:
    if values is None:
        return list(DEFAULT_BENCH_MODES)
    modes: list[str] = []
    for value in values:
        for token in value.split(","):
            mode = token.strip().lower()
            if mode:
                modes.append(mode)
    valid_modes = {"prefill", "decode"}
    invalid = [m for m in modes if m not in valid_modes]
    if invalid:
        raise ValueError(f"Invalid --bench-modes entries: {invalid}. Expected subset of {sorted(valid_modes)}.")
    deduped: list[str] = []
    for mode in modes:
        if mode not in deduped:
            deduped.append(mode)
    return deduped


def build_cases(
    seq_lengths: list[int],
    flare_m_list: list[int],
    bench_modes: list[str],
    include_triton: bool,
) -> list[BenchCase]:
    providers: list[str] = []
    if include_triton:
        providers.append("triton")

    cases: list[BenchCase] = []
    for mode in bench_modes:
        for provider in providers:
            for m in flare_m_list:
                for n in seq_lengths:
                    cases.append(BenchCase(mode=mode, provider=provider, n=n, m=m))
    return cases


def _case_key(
    *,
    mode: str,
    provider: str,
    n: int,
    m: int,
    batch_size: int,
    num_heads: int,
    score_head_dim: int,
    value_head_dim: int,
    dtype: str,
    decode_steps: int,
) -> tuple[Any, ...]:
    return (
        mode,
        provider,
        int(n),
        int(m),
        int(batch_size),
        int(num_heads),
        int(score_head_dim),
        int(value_head_dim),
        dtype,
        int(decode_steps),
    )


def _case_key_from_row(row: dict[str, Any]) -> tuple[Any, ...]:
    score_head_dim = int(row.get("score_head_dim", row.get("head_dim")))
    value_head_dim = int(row.get("value_head_dim", score_head_dim))
    return _case_key(
        mode=str(row["mode"]),
        provider=str(row["provider"]),
        n=int(row["N"]),
        m=int(row["M"]),
        batch_size=int(row["batch_size"]),
        num_heads=int(row["num_heads"]),
        score_head_dim=score_head_dim,
        value_head_dim=value_head_dim,
        dtype=str(row["dtype"]),
        decode_steps=int(row["decode_steps"]),
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
            str(r["mode"]),
            str(r["provider"]),
            int(r["M"]),
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
        df = df.sort_values(["mode", "provider", "M", "N"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    return df


def _sanitize_error_text(exc: Exception) -> str:
    text = str(exc).strip().replace("\n", "\\n").replace("\t", " ")
    return text[:4000]


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in state.items()}


def _build_prefill_fn(
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
    input_precision: str | None,
):
    q = torch.randn(h, m, d_score, device=device, dtype=dtype)
    k = torch.randn(b, n, h, d_score, device=device, dtype=dtype)
    v = torch.randn(b, n, h, d_value, device=device, dtype=dtype)

    def run() -> None:
        flare_prefill_triton(Q=q, K=k, V=v, scale=scale, input_precision=input_precision)

    return run


def _build_decode_fn(
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
    input_precision: str | None,
    decode_steps: int,
):
    q = torch.randn(h, m, d_score, device=device, dtype=dtype)
    k_prompt = torch.randn(b, n, h, d_score, device=device, dtype=dtype)
    v_prompt = torch.randn(b, n, h, d_value, device=device, dtype=dtype)
    k_steps = torch.randn(decode_steps, b, h, d_score, device=device, dtype=dtype)
    v_steps = torch.randn(decode_steps, b, h, d_value, device=device, dtype=dtype)

    _, base_state = flare_prefill_triton(
        Q=q,
        K=k_prompt,
        V=v_prompt,
        scale=scale,
        input_precision=input_precision,
    )

    state = _clone_state(base_state)

    def run() -> None:
        for key in state:
            state[key].copy_(base_state[key])
        st = state
        for step_idx in range(decode_steps):
            _, st = flare_decode_triton(
                Q=q,
                K=k_steps[step_idx],
                V=v_steps[step_idx],
                state=st,
                scale=scale,
                input_precision=input_precision,
            )

    return run


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
    decode_steps: int,
    prefill_input_precision: str | None,
    run_tag: str,
    device_name: str,
) -> dict[str, Any]:
    scale = score_head_dim ** -0.5 if score_head_dim > 8 else 1.0
    t0 = time.time()
    base_row: dict[str, Any] = {
        "run_tag": run_tag,
        "timestamp": int(t0),
        "mode": case.mode,
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
        "decode_steps": decode_steps,
        "scale": scale,
        "status": "unknown",
        "median_ms": math.nan,
        "p20_ms": math.nan,
        "p80_ms": math.nan,
        "tokens_per_s": math.nan,
        "ms_per_token": math.nan,
        "p20_ms_per_token": math.nan,
        "p80_ms_per_token": math.nan,
        "error": "",
    }
    if case.provider == "triton":
        base_row["prefill_input_precision"] = prefill_input_precision

    try:
        if case.mode == "prefill":
            run_case = _build_prefill_fn(
                b=batch_size,
                h=num_heads,
                n=case.n,
                m=case.m,
                d_score=score_head_dim,
                d_value=value_head_dim,
                dtype=dtype,
                device=device,
                scale=scale,
                input_precision=prefill_input_precision,
            )
            token_count = batch_size * case.n
        elif case.mode == "decode":
            run_case = _build_decode_fn(
                b=batch_size,
                h=num_heads,
                n=case.n,
                m=case.m,
                d_score=score_head_dim,
                d_value=value_head_dim,
                dtype=dtype,
                device=device,
                scale=scale,
                input_precision=prefill_input_precision,
                decode_steps=decode_steps,
            )
            token_count = batch_size * decode_steps
        else:
            raise ValueError(f"Unsupported mode={case.mode}")

        if precompile:
            run_case()
            torch.cuda.synchronize(device)

        p50_ms, p20_ms, p80_ms = triton.testing.do_bench(
            run_case,
            warmup=warmup,
            rep=rep,
            quantiles=[0.5, 0.2, 0.8],
        )
        base_row["status"] = "ok"
        base_row["median_ms"] = float(p50_ms)
        base_row["p20_ms"] = float(p20_ms)
        base_row["p80_ms"] = float(p80_ms)
        elapsed_s = max(float(p50_ms) * 1e-3, 1e-9)
        base_row["tokens_per_s"] = float(token_count / elapsed_s)
        per_token_denom = case.n if case.mode == "prefill" else decode_steps
        base_row["ms_per_token"] = float(p50_ms) / float(per_token_denom)
        base_row["p20_ms_per_token"] = float(p20_ms) / float(per_token_denom)
        base_row["p80_ms_per_token"] = float(p80_ms) / float(per_token_denom)
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

    include_triton = not args.skip_triton
    cases = build_cases(
        seq_lengths=args.seq_lengths,
        flare_m_list=args.flare_m,
        bench_modes=args.bench_modes,
        include_triton=include_triton,
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
            mode=case.mode,
            provider=case.provider,
            n=case.n,
            m=case.m,
            batch_size=args.batch_size,
            num_heads=args.num_heads,
            score_head_dim=args.score_head_dim,
            value_head_dim=args.value_head_dim,
            dtype=args.dtype,
            decode_steps=args.decode_steps,
        )
        if (not args.rerun_existing) and key in existing_keys:
            skipped += 1
            print(
                f"[{idx:03d}/{len(cases)}] skip existing: "
                f"mode={case.mode} provider={case.provider} N={case.n} M={case.m} "
                f"Dk={args.score_head_dim} Dv={args.value_head_dim}"
            )
            continue

        print(
            f"[{idx:03d}/{len(cases)}] run: "
            f"mode={case.mode} provider={case.provider} N={case.n} M={case.m} "
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
            decode_steps=args.decode_steps,
            prefill_input_precision=args.prefill_input_precision,
            run_tag=run_tag,
            device_name=device_name,
        )
        append_jsonl(raw_jsonl_path, row)
        all_rows.append(row)
        existing_keys.add(key)
        completed += 1
        print(
            f"  -> status={row['status']} median={row['median_ms']:.3f}ms "
            f"p20={row['p20_ms']:.3f}ms p80={row['p80_ms']:.3f}ms "
            f"tok/s={row['tokens_per_s']:.2f}"
        )
        if row["status"] != "ok" and row["error"]:
            print(f"     error: {row['error']}")

    df = write_summary_files(all_rows, csv_path=csv_path, json_path=json_path)
    print(f"Benchmark done. completed={completed}, skipped={skipped}")
    print(f"Raw timings:    {raw_jsonl_path}")
    print(f"Summary CSV:    {csv_path}")
    print(f"Summary JSON:   {json_path}")
    return df


def _curve_label(provider: str, m: int) -> str:
    return f"Triton (M={m})"


def plot_summary(df: pd.DataFrame, png_path: Path, pdf_path: Path, title: str) -> None:
    if df.empty:
        raise ValueError("Summary dataframe is empty.")

    ok = df[df["status"] == "ok"].copy()
    if ok.empty:
        raise ValueError("No successful benchmark rows to plot.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    provider_colors = {"triton": "#1f77b4"}
    m_styles = {32: "-", 64: "--", 128: "-.", 256: ":"}

    mode_specs = [
        ("prefill", "median_ms", "p20_ms", "p80_ms", "Prefill Time per Forward (ms)"),
        ("decode", "ms_per_token", "p20_ms_per_token", "p80_ms_per_token", "Decode Time per Token (ms)"),
    ]

    for ax, (mode, y_col, y_lo_col, y_hi_col, y_label) in zip(axes, mode_specs):
        mode_df = ok[ok["mode"] == mode]
        if mode_df.empty:
            ax.text(0.5, 0.5, f"No successful {mode} rows", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue

        groups = []
        provider = "triton"
        sub_provider = mode_df[mode_df["provider"] == provider]
        if not sub_provider.empty:
            for m in sorted(sub_provider["M"].astype(int).unique().tolist()):
                groups.append((provider, m, sub_provider[sub_provider["M"] == m]))

        for provider, m, sub in groups:
            sub = sub.sort_values("N")
            color = provider_colors.get(provider, "#7f7f7f")
            line_style = m_styles.get(int(m), "-")
            label = _curve_label(provider, int(m))
            x = sub["N"].to_numpy()
            y = sub[y_col].to_numpy()
            y_lo = sub[y_lo_col].to_numpy()
            y_hi = sub[y_hi_col].to_numpy()

            ax.plot(x, y, marker="o", linestyle=line_style, color=color, linewidth=2.0, label=label)
            ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.12)

        ax.set_title(mode.capitalize())
        ax.set_xlabel("Sequence length N")
        ax.set_ylabel(y_label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(title)
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
            "Benchmark FLARE prefill/decode methods across sequence lengths and latent counts, "
            "with explicit score/value head dimensions."
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
        "--bench-modes",
        nargs="+",
        default=None,
        help="Benchmark modes (prefill/decode), comma-separated and/or space-separated. Default: prefill decode",
    )
    parser.add_argument("--decode-steps", type=int, default=256, help="Decode tokens per timed call for decode mode")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations for triton.testing.do_bench")
    parser.add_argument("--rep", type=int, default=5, help="Timed repetitions for triton.testing.do_bench")
    parser.add_argument("--no-precompile", action="store_true", help="Disable one compile warmup step before timing")
    parser.add_argument("--rerun-existing", action="store_true", help="Rerun cases even if already present in raw JSONL")
    parser.add_argument("--skip-triton", action="store_true")
    parser.add_argument(
        "--prefill-input-precision",
        choices=["ieee", "tf32", "tf32x3"],
        default=None,
        help="Optional Triton input precision override for prefill/decode kernels.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/flare_prefill_decode_methods",
    )
    parser.add_argument("--raw-jsonl", type=str, default="prefill_decode_timings_raw.jsonl")
    parser.add_argument("--summary-csv", type=str, default="prefill_decode_timings.csv")
    parser.add_argument("--summary-json", type=str, default="prefill_decode_timings.json")
    parser.add_argument("--plot-png", type=str, default="prefill_decode_timings.png")
    parser.add_argument("--plot-pdf", type=str, default="prefill_decode_timings.pdf")
    parser.add_argument(
        "--figure-title",
        type=str,
        default="FLARE prefill/decode methods latency",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.seq_lengths = parse_tokens_list(args.seq_lengths) if args.seq_lengths is not None else list(DEFAULT_SEQ_LENGTHS)
    args.flare_m = parse_tokens_list(args.flare_m) if args.flare_m is not None else list(DEFAULT_FLARE_M)
    args.bench_modes = parse_modes(args.bench_modes)
    if args.value_head_dim is None:
        args.value_head_dim = args.score_head_dim

    if args.batch_size <= 0 or args.num_heads <= 0 or args.score_head_dim <= 0 or args.value_head_dim <= 0:
        raise ValueError("batch-size, num-heads, score-head-dim, and value-head-dim must be > 0.")
    if args.decode_steps <= 0:
        raise ValueError(f"decode-steps must be > 0. Got {args.decode_steps}")
    if not args.bench_modes:
        raise ValueError("No benchmark modes requested.")
    if args.skip_triton:
        raise ValueError("--skip-triton is set; nothing to run.")
    if any(n <= 0 for n in args.seq_lengths):
        raise ValueError(f"All sequence lengths must be > 0, got {args.seq_lengths}")
    if any(m <= 0 for m in args.flare_m):
        raise ValueError(f"All FLARE M values must be > 0, got {args.flare_m}")

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

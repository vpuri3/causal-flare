#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import triton

from causal_flare.semi_autoregressive import flare_semi_autoregressive_trition


SUPPORTED_CHUNK_SIZES = (16, 32, 64, 128)
FAST_PREPARE_LAUNCHES = ((2, 2), (4, 1), (4, 2), (4, 3), (4, 4), (8, 2))
FULL_PREPARE_LAUNCHES = ((2, 1), (2, 2), (4, 1), (4, 2), (4, 3), (4, 4), (8, 1), (8, 2))
FAST_OUTPUT_LAUNCHES = ((2, 2), (4, 1), (4, 2), (4, 3), (4, 4), (8, 2))
FULL_OUTPUT_LAUNCHES = ((2, 1), (2, 2), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (8, 1), (8, 2))
FAST_SAMPLE_CASES = (
    "32:32:32:32",
    "64:64:64:64",
    "128:32:32:128",
    "128:128:128:128",
    "128:32:128:256",
    "128:128:32:256",
    "192:96:96:192",
    "256:32:32:256",
    "256:128:128:256",
    "256:64:64:384",
    "512:32:32:256",
    "512:128:128:512",
)


@dataclass(frozen=True)
class Case:
    M: int
    D_score: int
    D_value: int
    block_size: int
    chunk_size: int

    @property
    def name(self) -> str:
        return f"m{self.M}_ds{self.D_score}_dv{self.D_value}_blk{self.block_size}_chk{self.chunk_size}"


def parse_case(spec: str) -> Case:
    m_str, ds_str, dv_str, blk_str = (part.strip() for part in spec.split(":"))
    M = int(m_str)
    D_score = int(ds_str)
    D_value = int(dv_str)
    block_size = int(blk_str)
    return Case(
        M=M,
        D_score=D_score,
        D_value=D_value,
        block_size=block_size,
        chunk_size=largest_supported_chunk(block_size),
    )


def largest_supported_chunk(block_size: int) -> int:
    for chunk_size in reversed(SUPPORTED_CHUNK_SIZES):
        if chunk_size <= block_size and block_size % chunk_size == 0:
            return chunk_size
    raise ValueError(
        f"Unable to choose a supported chunk size for block_size={block_size}. "
        f"Expected a multiple of one of {SUPPORTED_CHUNK_SIZES}."
    )


def parse_pair_list(spec: str) -> tuple[tuple[int, int], ...]:
    pairs = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        warps_str, stages_str = item.split("x", 1)
        pairs.append((int(warps_str), int(stages_str)))
    if not pairs:
        raise ValueError(f"Expected at least one launch pair in {spec!r}.")
    return tuple(pairs)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def bucket_m(M: int) -> str:
    if M <= 64:
        return "M<=64"
    if M <= 256:
        return "64<M<=256"
    return "M>256"


def bucket_d(D_score: int, D_value: int) -> str:
    width = max(D_score, D_value)
    mixed = D_score != D_value
    if width <= 64:
        prefix = "D<=64"
    elif width <= 128:
        prefix = "64<D<=128"
    else:
        prefix = "D>128"
    return f"{prefix} mixed" if mixed else prefix


def bucket_block(block_size: int) -> str:
    if block_size <= 64:
        return "block<=64"
    if block_size <= 256:
        return "64<block<=256"
    if block_size <= 512:
        return "256<block<=512"
    return "block>512"


def default_block_t_choices(chunk_size: int) -> tuple[int, ...]:
    candidates = [value for value in (16, 32, 64, 128) if value <= chunk_size and chunk_size % value == 0]
    return tuple(candidates)


def align_num_tokens(num_tokens: int, block_size: int) -> int:
    return triton.cdiv(num_tokens, block_size) * block_size


def apply_env(overrides: dict[str, str]) -> dict[str, str | None]:
    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        os.environ[key] = value
    return previous


def restore_env(previous: dict[str, str | None]) -> None:
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def make_inputs(case: Case, *, batch_size: int, num_heads: int, num_tokens: int, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(0)
    q = torch.randn((num_heads, case.M, case.D_score), device=device, dtype=dtype)
    k = torch.randn((batch_size, num_tokens, num_heads, case.D_score), device=device, dtype=dtype)
    v = torch.randn((batch_size, num_tokens, num_heads, case.D_value), device=device, dtype=dtype)
    return q, k, v


def bench_case(
    case: Case,
    *,
    batch_size: int,
    num_heads: int,
    num_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
    prepare_launch: tuple[int, int],
    lse_output_launch: tuple[int, int],
    block_t: int,
) -> dict[str, Any]:
    aligned_num_tokens = align_num_tokens(num_tokens, case.block_size)
    q, k, v = make_inputs(
        case,
        batch_size=batch_size,
        num_heads=num_heads,
        num_tokens=aligned_num_tokens,
        dtype=dtype,
        device=device,
    )
    scale = case.D_score ** -0.5
    overrides = {
        "FLARE_SEMI_AR_BLOCK_PREPARE_NUM_WARPS": str(prepare_launch[0]),
        "FLARE_SEMI_AR_BLOCK_PREPARE_NUM_STAGES": str(prepare_launch[1]),
        "FLARE_SEMI_AR_LSE_OUTPUT_NUM_WARPS": str(lse_output_launch[0]),
        "FLARE_SEMI_AR_LSE_OUTPUT_NUM_STAGES": str(lse_output_launch[1]),
        "FLARE_SEMI_AR_BLOCK_T": str(block_t),
    }
    previous = apply_env(overrides)
    try:
        fn = lambda: flare_semi_autoregressive_trition(
            q,
            k,
            v,
            block_size=case.block_size,
            chunk_size=case.chunk_size,
            scale=scale,
        )[0]
        fn()
        torch.cuda.synchronize()
        total_ms = triton.testing.do_bench(fn, warmup=2, rep=3)
        _, _, profile = flare_semi_autoregressive_trition(
            q,
            k,
            v,
            block_size=case.block_size,
            chunk_size=case.chunk_size,
            scale=scale,
            profile=True,
        )
    finally:
        restore_env(previous)
    forward = profile["forward"]
    return {
        "case": asdict(case),
        "num_tokens": aligned_num_tokens,
        "total_ms": float(total_ms),
        "block_prepare_ms": float(forward.get("semi_ar_block_prepare", float("nan"))),
        "block_scan_z_ms": float(forward.get("semi_ar_block_scan_z", float("nan"))),
        "lse_output_ms": float(forward.get("semi_ar_lse_output", float("nan"))),
        "output_cast_ms": float(forward.get("output_cast", float("nan"))),
        "prepare_launch": {"num_warps": prepare_launch[0], "num_stages": prepare_launch[1]},
        "lse_output_launch": {"num_warps": lse_output_launch[0], "num_stages": lse_output_launch[1]},
        "block_t": block_t,
        "m_bucket": bucket_m(case.M),
        "d_bucket": bucket_d(case.D_score, case.D_value),
        "block_bucket": bucket_block(case.block_size),
    }


def summarize_best(rows: list[dict[str, Any]], *, metric: str) -> list[dict[str, Any]]:
    best_by_case: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_name = Case(**row["case"]).name
        current = best_by_case.get(case_name)
        if current is None or row[metric] < current[metric]:
            best_by_case[case_name] = row
    return [best_by_case[name] for name in sorted(best_by_case)]


def aggregate_counts(rows: list[dict[str, Any]], *, selector: str) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket_key = f"{row['m_bucket']} | {row['d_bucket']} | {row['block_bucket']}"
        grouped.setdefault(bucket_key, {})
        key = selector_value(row, selector)
        grouped[bucket_key][key] = grouped[bucket_key].get(key, 0) + 1
    return grouped


def selector_value(row: dict[str, Any], selector: str) -> str:
    if selector == "prepare_launch":
        launch = row["prepare_launch"]
        return f"{launch['num_warps']}w/{launch['num_stages']}s"
    if selector == "lse_output_launch":
        launch = row["lse_output_launch"]
        return f"{launch['num_warps']}w/{launch['num_stages']}s"
    if selector == "block_t":
        return str(row["block_t"])
    raise ValueError(f"Unsupported selector {selector!r}.")


def format_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Semi-AR Matrix Tuning Summary",
        "",
        f"Run: `{summary['run_name']}`",
        f"Timestamp: `{summary['timestamp']}`",
        f"Raw results: `{summary['raw_jsonl']}`",
        "",
        "## Cases",
        "",
    ]
    for case in summary["cases"]:
        lines.append(
            f"- `{case['name']}`: M={case['M']}, D_score={case['D_score']}, D_value={case['D_value']}, "
            f"block_size={case['block_size']}, chunk_size={case['chunk_size']}"
        )
    lines.extend(["", "## Best Prepare Launch Per Case", ""])
    for row in summary["best_prepare"]:
        lines.append(
            f"- `{Case(**row['case']).name}`: {selector_value(row, 'prepare_launch')} "
            f"(prepare={row['block_prepare_ms']:.3f} ms, total={row['total_ms']:.3f} ms)"
        )
    lines.extend(["", "## Best LSE Output Launch Per Case", ""])
    for row in summary["best_output"]:
        lines.append(
            f"- `{Case(**row['case']).name}`: BLOCK_T={row['block_t']}, {selector_value(row, 'lse_output_launch')} "
            f"(lse_output={row['lse_output_ms']:.3f} ms, total={row['total_ms']:.3f} ms)"
        )
    lines.extend(["", "## Bucket Winner Counts", ""])
    for title, counts in (
        ("Prepare Launch", summary["prepare_bucket_counts"]),
        ("LSE Output Launch", summary["output_bucket_counts"]),
        ("BLOCK_T", summary["block_t_bucket_counts"]),
    ):
        lines.extend([f"### {title}", ""])
        for bucket, bucket_counts in sorted(counts.items()):
            formatted = ", ".join(f"`{name}` x{count}" for name, count in sorted(bucket_counts.items()))
            lines.append(f"- `{bucket}`: {formatted}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def resolve_output_paths(*, run_name: str | None, jsonl_path: Path | None, summary_json_path: Path | None, summary_md_path: Path | None):
    if run_name is None:
        run_name = datetime.now().strftime("semi_ar_matrix_%Y%m%d_%H%M%S")
    root = Path("results") / "semi_ar_matrix" / run_name
    return (
        run_name,
        jsonl_path or (root / "runs.jsonl"),
        summary_json_path or (root / "summary.json"),
        summary_md_path or (root / "summary.md"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a sampled semi-autoregressive FLARE forward tuning sweep.")
    parser.add_argument("--cases", default=",".join(FAST_SAMPLE_CASES), help="Comma-separated M:D_score:D_value:block_size cases.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-tokens", type=int, default=49152)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--candidate-set", choices=("fast", "full"), default="fast")
    parser.add_argument("--prepare-launches", default=None)
    parser.add_argument("--output-launches", default=None)
    parser.add_argument("--jsonl", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--summary-md", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Semi-AR matrix tuning requires CUDA.")
    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    cases = [parse_case(spec) for spec in args.cases.split(",") if spec.strip()]
    invalid_cases = [case.name for case in cases if (args.num_tokens % case.block_size) != 0]
    if invalid_cases:
        raise ValueError(
            f"All sampled cases require num_tokens to be an exact multiple of block_size. "
            f"Got num_tokens={args.num_tokens}, invalid_cases={invalid_cases}."
        )
    if args.prepare_launches is None:
        prepare_launches = FAST_PREPARE_LAUNCHES if args.candidate_set == "fast" else FULL_PREPARE_LAUNCHES
    else:
        prepare_launches = parse_pair_list(args.prepare_launches)
    if args.output_launches is None:
        output_launches = FAST_OUTPUT_LAUNCHES if args.candidate_set == "fast" else FULL_OUTPUT_LAUNCHES
    else:
        output_launches = parse_pair_list(args.output_launches)
    run_name, jsonl_path, summary_json_path, summary_md_path = resolve_output_paths(
        run_name=args.run_name,
        jsonl_path=args.jsonl,
        summary_json_path=args.summary_json,
        summary_md_path=args.summary_md,
    )

    raw_rows: list[dict[str, Any]] = []
    best_prepare_rows: list[dict[str, Any]] = []
    best_output_rows: list[dict[str, Any]] = []

    for case in cases:
        block_t_rows = []
        for block_t in default_block_t_choices(case.chunk_size):
            row = bench_case(
                case,
                batch_size=args.batch_size,
                num_heads=args.num_heads,
                num_tokens=args.num_tokens,
                dtype=dtype,
                device=device,
                prepare_launch=(4, 2),
                lse_output_launch=(4, 2),
                block_t=block_t,
            )
            row["phase_family"] = "block_t"
            raw_rows.append(row)
            append_jsonl(jsonl_path, row)
            block_t_rows.append(row)
        best_block_t = min(block_t_rows, key=lambda row: row["total_ms"])["block_t"]

        prepare_rows = []
        for launch in prepare_launches:
            row = bench_case(
                case,
                batch_size=args.batch_size,
                num_heads=args.num_heads,
                num_tokens=args.num_tokens,
                dtype=dtype,
                device=device,
                prepare_launch=launch,
                lse_output_launch=(4, 2),
                block_t=best_block_t,
            )
            row["phase_family"] = "prepare_launch"
            raw_rows.append(row)
            append_jsonl(jsonl_path, row)
            prepare_rows.append(row)
        best_prepare = min(prepare_rows, key=lambda row: row["total_ms"])
        best_prepare_rows.append(best_prepare)

        output_rows = []
        for launch in output_launches:
            row = bench_case(
                case,
                batch_size=args.batch_size,
                num_heads=args.num_heads,
                num_tokens=args.num_tokens,
                dtype=dtype,
                device=device,
                prepare_launch=(
                    best_prepare["prepare_launch"]["num_warps"],
                    best_prepare["prepare_launch"]["num_stages"],
                ),
                lse_output_launch=launch,
                block_t=best_block_t,
            )
            row["phase_family"] = "lse_output_launch"
            raw_rows.append(row)
            append_jsonl(jsonl_path, row)
            output_rows.append(row)
        best_output_rows.append(min(output_rows, key=lambda row: row["total_ms"]))

    summary = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "raw_jsonl": str(jsonl_path),
        "summary_json": str(summary_json_path),
        "summary_md": str(summary_md_path),
        "cases": [{**asdict(case), "name": case.name} for case in cases],
        "best_prepare": best_prepare_rows,
        "best_output": best_output_rows,
        "prepare_bucket_counts": aggregate_counts(best_prepare_rows, selector="prepare_launch"),
        "output_bucket_counts": aggregate_counts(best_output_rows, selector="lse_output_launch"),
        "block_t_bucket_counts": aggregate_counts(
            [min([row for row in raw_rows if row["phase_family"] == "block_t" and Case(**row["case"]).name == case.name], key=lambda row: row["total_ms"]) for case in cases],
            selector="block_t",
        ),
    }
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary_md_path.write_text(format_summary_markdown(summary), encoding="utf-8")
    print(json.dumps({"run_name": run_name, "summary_json": str(summary_json_path), "summary_md": str(summary_md_path)}, indent=2))


if __name__ == "__main__":
    main()

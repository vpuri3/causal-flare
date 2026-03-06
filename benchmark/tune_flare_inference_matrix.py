#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from .profile_chunked_flare import Case, parse_dtype
    from .profile_flare_inference import bench_case
except ImportError:
    from profile_chunked_flare import Case, parse_dtype
    from profile_flare_inference import bench_case


DEFAULT_D_VALUES = (16, 32, 64, 96, 128, 192, 256, 384, 512)
DEFAULT_M_VALUES = (16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048)
DEFAULT_N_VALUES = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
DEFAULT_BH_VALUES = (8,)
DEFAULT_CHUNK_SIZES = (64, 128, 256, 512)
DEFAULT_BLOCK_VALUES = (16, 32, 64, 128, 256, 512)

FAMILY_GROUPS = {
    "prefill": (
        "default",
        "chunk_size",
        "prefill_block_m",
        "prefill_block_d",
        "prefill_block_k",
        "prefill_launch",
        "combined",
    ),
    "decode": (
        "default",
        "decode_block_m",
        "decode_block_d",
        "decode_block_k",
        "decode_launch",
        "combined",
    ),
}


@dataclass(frozen=True)
class CandidateConfig:
    family: str
    name: str
    env: dict[str, str]


def parse_int_list(spec: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if spec is None:
        return default
    values = [int(token.strip()) for token in spec.split(",") if token.strip()]
    if not values:
        raise ValueError(f"Expected at least one integer in {spec!r}.")
    return tuple(values)


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


def power_of_two_values(limit: int) -> list[int]:
    out = []
    value = 16
    while value <= limit:
        out.append(value)
        value *= 2
    return out


def divisor_values(limit: int, candidates: tuple[int, ...]) -> list[int]:
    return [value for value in candidates if value <= limit and (limit % value) == 0]


def build_candidates(case: Case, *, mode: str, chunk_sizes: tuple[int, ...], block_values: tuple[int, ...]) -> list[CandidateConfig]:
    d = case.head_dim
    m = case.latent_queries
    n = case.seq_len
    candidates = [CandidateConfig("default", "default", {})]
    if mode == "prefill":
        candidates.extend(
            CandidateConfig("chunk_size", f"chunk_size_{chunk_size}", {"FLARE_CHUNK_SIZE": str(chunk_size)})
            for chunk_size in chunk_sizes
            if chunk_size <= n
        )
        candidates.extend(
            CandidateConfig(
                "prefill_block_m",
                f"prefill_block_m_{block_m}",
                {"FLARE_BLOCK_M": str(block_m), "FLARE_PREFIX_BLOCK_M": str(block_m)},
            )
            for block_m in power_of_two_values(m)
        )
        candidates.extend(
            CandidateConfig(
                "prefill_block_d",
                f"prefill_block_d_{block_d}",
                {
                    "FLARE_PREPARE_BLOCK_D": str(block_d),
                    "FLARE_PREFIX_BLOCK_D": str(block_d),
                    "FLARE_FWD_BLOCK_D": str(block_d),
                },
            )
            for block_d in block_values
            if block_d <= d
        )
        for prepare_block_k in divisor_values(d, block_values):
            for fwd_block_k in divisor_values(d, block_values):
                if prepare_block_k < fwd_block_k:
                    continue
                candidates.append(
                    CandidateConfig(
                        "prefill_block_k",
                        f"prefill_block_k_p{prepare_block_k}_f{fwd_block_k}",
                        {
                            "FLARE_PREPARE_BLOCK_K": str(prepare_block_k),
                            "FLARE_FWD_BLOCK_K": str(fwd_block_k),
                        },
                    )
                )
        candidates.extend(
            (
                CandidateConfig(
                    "prefill_launch",
                    "prefill_launch_legacy_8w3s",
                    {
                        "FLARE_PREPARE_NUM_WARPS": "8",
                        "FLARE_PREPARE_NUM_STAGES": "3",
                        "FLARE_PREFIX_NUM_WARPS": "8",
                        "FLARE_PREFIX_NUM_STAGES": "3",
                        "FLARE_DECODER_NUM_WARPS": "8",
                        "FLARE_DECODER_NUM_STAGES": "3",
                        "FLARE_FWD_NUM_WARPS": "8",
                        "FLARE_FWD_NUM_STAGES": "3",
                    },
                ),
                CandidateConfig(
                    "prefill_launch",
                    "prefill_launch_balanced_4w2s_4w1s",
                    {
                        "FLARE_PREPARE_NUM_WARPS": "4",
                        "FLARE_PREPARE_NUM_STAGES": "2",
                        "FLARE_PREFIX_NUM_WARPS": "4",
                        "FLARE_PREFIX_NUM_STAGES": "2",
                        "FLARE_DECODER_NUM_WARPS": "4",
                        "FLARE_DECODER_NUM_STAGES": "1",
                        "FLARE_FWD_NUM_WARPS": "4",
                        "FLARE_FWD_NUM_STAGES": "1",
                    },
                ),
                CandidateConfig(
                    "prefill_launch",
                    "prefill_launch_light_fwd_2w1s",
                    {
                        "FLARE_PREPARE_NUM_WARPS": "4",
                        "FLARE_PREPARE_NUM_STAGES": "2",
                        "FLARE_PREFIX_NUM_WARPS": "4",
                        "FLARE_PREFIX_NUM_STAGES": "2",
                        "FLARE_DECODER_NUM_WARPS": "4",
                        "FLARE_DECODER_NUM_STAGES": "1",
                        "FLARE_FWD_NUM_WARPS": "2",
                        "FLARE_FWD_NUM_STAGES": "1",
                    },
                ),
            )
        )
    else:
        candidates.extend(
            CandidateConfig(
                "decode_block_m",
                f"decode_block_m_{block_m}",
                {"FLARE_DECODE_BLOCK_M": str(block_m)},
            )
            for block_m in power_of_two_values(max(m, 16))
            if block_m >= m
        )
        candidates.extend(
            CandidateConfig(
                "decode_block_d",
                f"decode_block_d_{block_d}",
                {"FLARE_DECODE_BLOCK_D": str(block_d)},
            )
            for block_d in block_values
            if block_d <= d
        )
        candidates.extend(
            CandidateConfig(
                "decode_block_k",
                f"decode_block_k_{block_k}",
                {"FLARE_DECODE_BLOCK_K": str(block_k)},
            )
            for block_k in divisor_values(d, block_values)
        )
        candidates.extend(
            (
                CandidateConfig("decode_launch", "decode_launch_default", {}),
                CandidateConfig(
                    "decode_launch",
                    "decode_launch_2w1s",
                    {"FLARE_DECODE_NUM_WARPS": "2", "FLARE_DECODE_NUM_STAGES": "1"},
                ),
                CandidateConfig(
                    "decode_launch",
                    "decode_launch_4w1s",
                    {"FLARE_DECODE_NUM_WARPS": "4", "FLARE_DECODE_NUM_STAGES": "1"},
                ),
                CandidateConfig(
                    "decode_launch",
                    "decode_launch_4w2s",
                    {"FLARE_DECODE_NUM_WARPS": "4", "FLARE_DECODE_NUM_STAGES": "2"},
                ),
                CandidateConfig(
                    "decode_launch",
                    "decode_launch_8w2s",
                    {"FLARE_DECODE_NUM_WARPS": "8", "FLARE_DECODE_NUM_STAGES": "2"},
                ),
            )
        )

    deduped = []
    seen = set()
    for candidate in candidates:
        key = (candidate.family, tuple(sorted(candidate.env.items())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def select_families(mode: str, spec: str | None) -> tuple[str, ...]:
    available = FAMILY_GROUPS[mode]
    if not spec:
        return available
    requested: list[str] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if token == "all":
            for family in available:
                if family not in requested:
                    requested.append(family)
            continue
        if token not in available:
            raise ValueError(f"Unsupported family {token!r} for mode={mode}. Expected subset of {available}.")
        if token not in requested:
            requested.append(token)
    return tuple(requested)


def result_key(record: dict) -> tuple:
    case = record["case"]
    return (
        record["benchmark_mode"],
        case["batch_size"],
        case["num_heads"],
        case["seq_len"],
        case["latent_queries"],
        case["head_dim"],
        record["dtype"],
        record["input_precision"],
        tuple(sorted(record["env"].items())),
        record.get("decode_steps", 0),
        record["config_name"],
    )


def planned_result_key(
    *,
    case: Case,
    mode: str,
    dtype_name: str,
    input_precision: str | None,
    env: dict[str, str],
    decode_steps: int,
    config_name: str,
) -> tuple:
    return (
        mode,
        case.batch_size,
        case.num_heads,
        case.seq_len,
        case.latent_queries,
        case.head_dim,
        dtype_name,
        input_precision or "default",
        tuple(sorted(env.items())),
        decode_steps if mode == "decode" else 0,
        config_name,
    )


def load_existing_results(path: Path | None) -> dict[tuple, dict]:
    if path is None or not path.exists():
        return {}
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        records[result_key(row)] = row
    return records


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row))
        handle.write("\n")


def resolve_output_paths(
    *,
    run_name: str | None,
    jsonl_path: Path | None,
    summary_json_path: Path | None,
    summary_md_path: Path | None,
    shard_index: int,
    num_shards: int,
    mode: str,
) -> tuple[str, Path, Path, Path]:
    if run_name is None:
        run_name = datetime.now().strftime(f"inference-{mode}-%Y%m%d-%H%M%S")
    root = Path("results") / "flare_inference_matrix" / run_name
    shard_stem = f"shard{shard_index:03d}-of-{num_shards:03d}"
    return (
        run_name,
        jsonl_path or (root / f"runs-{shard_stem}.jsonl"),
        summary_json_path or (root / f"summary-{shard_stem}.json"),
        summary_md_path or (root / f"summary-{shard_stem}.md"),
    )


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
    for case_name, rows in sorted(by_case.items()):
        rows_sorted = sorted(rows, key=lambda row: float(row[objective]))
        best = rows_sorted[0]
        default = next((row for row in rows if row["config_name"] == "default"), best)
        speedup = float(default[objective]) / float(best[objective]) if float(best[objective]) > 0 else float("inf")
        summary_row = {
            "case": best["case"],
            "best_config": best["config_name"],
            "best_family": best["family"],
            "best_value_ms": float(best[objective]),
            "default_value_ms": float(default[objective]),
            "speedup_vs_default": speedup,
            "best_env": best["env"],
            "best_hot_kernel": best["kernels"][0] if best["kernels"] else None,
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


def build_error_row(
    *,
    case: Case,
    mode: str,
    dtype_name: str,
    input_precision: str | None,
    env: dict[str, str],
    decode_steps: int,
    config_name: str,
    family: str,
    error: Exception,
) -> dict:
    row = {
        "case": {
            "name": case.name,
            "batch_size": case.batch_size,
            "num_heads": case.num_heads,
            "seq_len": case.seq_len,
            "latent_queries": case.latent_queries,
            "head_dim": case.head_dim,
        },
        "benchmark_mode": mode,
        "dtype": dtype_name,
        "input_precision": input_precision or "default",
        "env": dict(env),
        "decode_steps": decode_steps if mode == "decode" else 0,
        "end_to_end_ms": float("inf"),
        "ms_per_token": float("inf"),
        "profile_total_ms": float("inf"),
        "kernels": [],
        "status": "error",
        "error": str(error)[:4000],
        "config_name": config_name,
        "family": family,
    }
    return row


def format_summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# FLARE Inference Matrix Tuning Summary",
        "",
        f"- Objective: `{summary['objective']}`",
        f"- Cases: `{summary['num_cases']}`",
        "",
        "| Case | Best config | Best ms | Default ms | Speedup |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["per_case"]:
        case = row["case"]
        label = f"B={case['batch_size']} H={case['num_heads']} N={case['seq_len']} M={case['latent_queries']} D={case['head_dim']}"
        lines.append(
            f"| `{label}` | `{row['best_config']}` | {row['best_value_ms']:.6f} | "
            f"{row['default_value_ms']:.6f} | {row['speedup_vs_default']:.2f}x |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeatable FLARE prefill/decode launch-config sweeps over a D/M/N/BH matrix.")
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--d-values", default=None)
    parser.add_argument("--m-values", default=None)
    parser.add_argument("--n-values", default=None)
    parser.add_argument("--bh-values", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--chunk-sizes", default="64,128,256,512")
    parser.add_argument("--block-values", default="16,32,64,128,256,512")
    parser.add_argument("--families", default=None, help="Comma-separated family list for the selected mode. Default uses the mode preset.")
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--input-precision", default=None)
    parser.add_argument("--decode-steps", type=int, default=256)
    parser.add_argument("--objective", choices=("end_to_end_ms", "ms_per_token", "profile_total_ms"), default="end_to_end_ms")
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
    chunk_sizes = parse_int_list(args.chunk_sizes, DEFAULT_CHUNK_SIZES)
    block_values = parse_int_list(args.block_values, DEFAULT_BLOCK_VALUES)
    families = select_families(args.mode, args.families)
    cases = build_cases(
        d_values=d_values,
        m_values=m_values,
        n_values=n_values,
        bh_values=bh_values,
        batch_size=args.batch_size,
    )
    cases = [case for idx, case in enumerate(cases) if (idx % args.num_shards) == args.shard_index]
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    if args.resume and args.jsonl is None and args.run_name is None:
        raise ValueError("Resume mode requires either --jsonl or --run-name.")

    run_name, jsonl_path, summary_json_path, summary_md_path = resolve_output_paths(
        run_name=args.run_name,
        jsonl_path=args.jsonl,
        summary_json_path=args.summary_json,
        summary_md_path=args.summary_md,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        mode=args.mode,
    )
    existing = load_existing_results(jsonl_path if args.resume else None)
    results_by_key = dict(existing)
    dtype_name = str(dtype).removeprefix("torch.")

    for case in cases:
        candidates = [
            candidate
            for candidate in build_candidates(case, mode=args.mode, chunk_sizes=chunk_sizes, block_values=block_values)
            if candidate.family in families and candidate.family != "combined"
        ]
        case_rows: list[dict] = []
        for candidate in candidates:
            key = planned_result_key(
                case=case,
                mode=args.mode,
                dtype_name=dtype_name,
                input_precision=args.input_precision,
                env=candidate.env,
                decode_steps=args.decode_steps,
                config_name=candidate.name,
            )
            if args.resume and key in existing:
                row = existing[key]
            else:
                try:
                    row = bench_case(
                        case,
                        dtype=dtype,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        input_precision=args.input_precision,
                        mode=args.mode,
                        env=candidate.env,
                        seed=args.seed,
                        decode_steps=args.decode_steps,
                    )[0]
                    row["status"] = "ok"
                    row["error"] = ""
                    row["config_name"] = candidate.name
                    row["family"] = candidate.family
                except Exception as exc:  # noqa: BLE001
                    row = build_error_row(
                        case=case,
                        mode=args.mode,
                        dtype_name=dtype_name,
                        input_precision=args.input_precision,
                        env=candidate.env,
                        decode_steps=args.decode_steps,
                        config_name=candidate.name,
                        family=candidate.family,
                        error=exc,
                    )
                append_jsonl(jsonl_path, row)
            results_by_key[key] = row
            case_rows.append(row)

        combined = merge_family_winners(case_rows, objective=args.objective)
        if combined is not None and "combined" in families:
            key = planned_result_key(
                case=case,
                mode=args.mode,
                dtype_name=dtype_name,
                input_precision=args.input_precision,
                env=combined.env,
                decode_steps=args.decode_steps,
                config_name=combined.name,
            )
            if args.resume and key in existing:
                row = existing[key]
            else:
                try:
                    row = bench_case(
                        case,
                        dtype=dtype,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        input_precision=args.input_precision,
                        mode=args.mode,
                        env=combined.env,
                        seed=args.seed,
                        decode_steps=args.decode_steps,
                    )[0]
                    row["status"] = "ok"
                    row["error"] = ""
                    row["config_name"] = combined.name
                    row["family"] = combined.family
                except Exception as exc:  # noqa: BLE001
                    row = build_error_row(
                        case=case,
                        mode=args.mode,
                        dtype_name=dtype_name,
                        input_precision=args.input_precision,
                        env=combined.env,
                        decode_steps=args.decode_steps,
                        config_name=combined.name,
                        family=combined.family,
                        error=exc,
                    )
                append_jsonl(jsonl_path, row)
            results_by_key[key] = row

    summary = summarize_results(list(results_by_key.values()), objective=args.objective)
    summary["run_name"] = run_name
    summary["mode"] = args.mode
    summary["families"] = list(families)
    summary["dtype"] = str(dtype).removeprefix("torch.")
    summary["input_precision"] = args.input_precision or "default"
    summary["matrix"] = {
        "d_values": list(d_values),
        "m_values": list(m_values),
        "n_values": list(n_values),
        "bh_values": list(bh_values),
        "batch_size": args.batch_size,
        "decode_steps": args.decode_steps,
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

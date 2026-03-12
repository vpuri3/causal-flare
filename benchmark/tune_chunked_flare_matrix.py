#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

try:
    from .profile_chunked_flare import Case, bench_case, parse_dtype
    from .tuning_catalog import (
        CHUNKED_TUNING_FAMILY_GROUPS as FAMILY_GROUPS,
        CandidateConfig,
        build_chunked_family_candidates,
    )
except ImportError:
    from profile_chunked_flare import Case, bench_case, parse_dtype
    from tuning_catalog import (
        CHUNKED_TUNING_FAMILY_GROUPS as FAMILY_GROUPS,
        CandidateConfig,
        build_chunked_family_candidates,
    )


FAST_D_VALUES = (64, 128, 256)
FAST_M_VALUES = (64, 512)
FAST_N_VALUES = (2048, 32768)
FAST_BH_VALUES = (8,)
FULL_D_VALUES = (16, 32, 64, 96, 128, 192, 256, 384, 512)
FULL_M_VALUES = (16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048)
FULL_N_VALUES = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
FULL_BH_VALUES = (8,)
DEFAULT_CHUNK_SIZES = (32, 64, 128, 256)
DEFAULT_FORWARD_BLOCK_KS = (16, 32, 64, 128)
DEFAULT_BACKWARD_BLOCK_KS = (16, 32, 64, 128)
DEFAULT_FORWARD_BLOCK_DS = (16, 32, 64, 128, 256)
DEFAULT_BACKWARD_BLOCK_DS = (16, 32, 64, 128, 256)
DEFAULT_BLOCK_TS = (16, 32, 64)
DEFAULT_SCALAR_APPLY_PANELS = (4, 8, 16)

def parse_int_list(spec: str | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if spec is None:
        return default
    values = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError(f"Expected at least one integer in {spec!r}.")
    return tuple(values)


def parse_family_list(spec: str) -> tuple[str, ...]:
    requested: list[str] = []
    for token in (part.strip() for part in spec.split(",")):
        if not token:
            continue
        if token in FAMILY_GROUPS:
            for family in FAMILY_GROUPS[token]:
                if family not in requested:
                    requested.append(family)
            continue
        if token not in FAMILY_GROUPS["all"]:
            raise ValueError(
                f"Unsupported family/group {token!r}. Expected one of {sorted(set(FAMILY_GROUPS['all']) | set(FAMILY_GROUPS))}."
            )
        if token not in requested:
            requested.append(token)
    return tuple(requested)


def default_family_group(*, mode: str, full_matrix: bool) -> str:
    if full_matrix:
        if mode == "forward":
            return "full_forward"
        if mode == "backward":
            return "full_backward"
        return "core"
    if mode == "forward":
        return "fast_forward"
    if mode == "backward":
        return "fast_backward"
    return "core_fast"


def default_matrix_values(*, full_matrix: bool) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if full_matrix:
        return FULL_D_VALUES, FULL_M_VALUES, FULL_N_VALUES, FULL_BH_VALUES
    return FAST_D_VALUES, FAST_M_VALUES, FAST_N_VALUES, FAST_BH_VALUES


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


def expand_candidates_for_case(
    case: Case,
    *,
    families: tuple[str, ...],
    chunk_sizes: tuple[int, ...],
    forward_block_ks: tuple[int, ...],
    backward_block_ks: tuple[int, ...],
    forward_block_ds: tuple[int, ...],
    backward_block_ds: tuple[int, ...],
    block_ts: tuple[int, ...],
    scalar_apply_panels: tuple[int, ...],
    extra_configs: list[CandidateConfig],
) -> tuple[list[CandidateConfig], dict[str, list[CandidateConfig]]]:
    family_candidates = build_chunked_family_candidates(
        head_dim=case.head_dim,
        latent_queries=case.latent_queries,
        seq_len=case.seq_len,
        chunk_sizes=chunk_sizes,
        forward_block_ks=forward_block_ks,
        backward_block_ks=backward_block_ks,
        forward_block_ds=forward_block_ds,
        backward_block_ds=backward_block_ds,
        block_ts=block_ts,
        scalar_apply_panels=scalar_apply_panels,
    )
    extras_by_family: dict[str, list[CandidateConfig]] = {}
    for extra in extra_configs:
        family_candidates.setdefault(extra.family, []).append(extra)
        extras_by_family.setdefault(extra.family, []).append(extra)
    expanded: list[CandidateConfig] = []
    for family in families:
        expanded.extend(family_candidates.get(family, []))
    for family, extras in extras_by_family.items():
        if family not in families:
            expanded.extend(extras)
    deduped: list[CandidateConfig] = []
    seen = set()
    for candidate in expanded:
        key = (candidate.family, tuple(sorted(candidate.env.items())))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped, family_candidates


def objective_value(result: dict, objective: str) -> float:
    return float(result[objective])


def result_key(case: Case, candidate: CandidateConfig, *, dtype: str, input_precision: str | None, mode: str) -> tuple:
    return (
        case.batch_size,
        case.num_heads,
        case.seq_len,
        case.latent_queries,
        case.head_dim,
        dtype,
        input_precision or "default",
        mode,
        candidate.family,
        candidate.name,
        tuple(sorted(candidate.env.items())),
    )


def build_error_record(
    *,
    case: Case,
    candidate: CandidateConfig,
    dtype: str,
    input_precision: str | None,
    mode: str,
    error: Exception,
) -> dict:
    return {
        "case": asdict(case),
        "dtype": dtype,
        "input_precision": input_precision or "default",
        "mode": mode,
        "env": dict(candidate.env),
        "end_to_end_ms": float("inf"),
        "forward_profile_total_ms": float("inf"),
        "backward_profile_total_ms": float("inf"),
        "forward_kernels": [],
        "backward_kernels": [],
        "config_name": candidate.name,
        "family": candidate.family,
        "status": "error",
        "error": str(error)[:4000],
    }


def record_key(record: dict) -> tuple:
    case = record["case"]
    return (
        case["batch_size"],
        case["num_heads"],
        case["seq_len"],
        case["latent_queries"],
        case["head_dim"],
        record["dtype"],
        record["input_precision"],
        record["mode"],
        record["family"],
        record["config_name"],
        tuple(sorted(record["env"].items())),
    )


def load_existing_results(path: Path | None) -> tuple[dict[tuple, dict], list[dict]]:
    if path is None or not path.exists():
        return {}, []
    by_key: dict[tuple, dict] = {}
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        key = record_key(record)
        by_key[key] = record
        records.append(record)
    return by_key, records


def append_jsonl(path: Path | None, record: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record))
        handle.write("\n")


def merge_family_winners(case_results: list[dict], families: tuple[str, ...], objective: str) -> CandidateConfig | None:
    merged_env: dict[str, str] = {}
    used_family = False
    for family in families:
        if family in {"default", "combined"}:
            continue
        family_rows = [row for row in case_results if row["family"] == family]
        if not family_rows:
            continue
        best_row = min(family_rows, key=lambda row: objective_value(row, objective))
        merged_env.update(best_row["env"])
        used_family = used_family or bool(best_row["env"])
    if not used_family:
        return None
    return CandidateConfig(family="combined", name="combined_best_by_family", env=merged_env)


def select_case_shard(cases: list[Case], *, shard_index: int, num_shards: int) -> list[Case]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive. Got {num_shards}.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}). Got {shard_index}.")
    return [case for idx, case in enumerate(cases) if (idx % num_shards) == shard_index]


def summarize_results(results: list[dict], *, objective: str) -> dict[str, object]:
    per_case: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        per_case[row["case"]["name"]].append(row)

    case_summaries = []
    grouped = {
        "by_d": defaultdict(list),
        "by_m": defaultdict(list),
        "by_n": defaultdict(list),
        "by_bh": defaultdict(list),
    }

    for case_name, rows in sorted(per_case.items()):
        rows_sorted = sorted(rows, key=lambda row: objective_value(row, objective))
        best = rows_sorted[0]
        default = next((row for row in rows if row["config_name"] == "default"), best)
        default_value = objective_value(default, objective)
        best_value = objective_value(best, objective)
        speedup = default_value / best_value if best_value > 0 else float("inf")
        summary_row = {
            "case": best["case"],
            "default_config": default["config_name"],
            "default_value_ms": default_value,
            "best_config": best["config_name"],
            "best_family": best["family"],
            "best_value_ms": best_value,
            "speedup_vs_default": speedup,
            "best_env": best["env"],
            "best_forward_hot_kernel": best["forward_kernels"][0] if best["forward_kernels"] else None,
            "best_backward_hot_kernel": best["backward_kernels"][0] if best["backward_kernels"] else None,
            "top3": [
                {
                    "config_name": row["config_name"],
                    "family": row["family"],
                    "value_ms": objective_value(row, objective),
                    "env": row["env"],
                }
                for row in rows_sorted[:3]
            ],
        }
        case_summaries.append(summary_row)

        grouped["by_d"][best["case"]["head_dim"]].append(summary_row)
        grouped["by_m"][best["case"]["latent_queries"]].append(summary_row)
        grouped["by_n"][best["case"]["seq_len"]].append(summary_row)
        grouped["by_bh"][best["case"]["batch_size"] * best["case"]["num_heads"]].append(summary_row)

    grouped_summary = {}
    for group_name, buckets in grouped.items():
        grouped_summary[group_name] = {}
        for bucket, rows in sorted(buckets.items()):
            winner_counts = defaultdict(int)
            speedups = defaultdict(list)
            for row in rows:
                winner_counts[row["best_config"]] += 1
                speedups[row["best_config"]].append(row["speedup_vs_default"])
            grouped_summary[group_name][bucket] = {
                "num_cases": len(rows),
                "winner_counts": dict(sorted(winner_counts.items())),
                "avg_speedup_by_winner": {
                    config_name: sum(values) / len(values)
                    for config_name, values in sorted(speedups.items())
                },
            }

    return {
        "objective": objective,
        "num_results": len(results),
        "num_cases": len(case_summaries),
        "per_case": case_summaries,
        "grouped": grouped_summary,
    }


def format_summary_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Chunked FLARE Matrix Tuning Summary",
        "",
        f"- Objective: `{summary['objective']}`",
        f"- Cases: `{summary['num_cases']}`",
        f"- Runs: `{summary['num_results']}`",
        "",
        "## Per-case winners",
        "",
        "| Case | Best config | Best ms | Default ms | Speedup |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in summary["per_case"]:
        case = row["case"]
        case_label = (
            f"B={case['batch_size']} H={case['num_heads']} "
            f"N={case['seq_len']} M={case['latent_queries']} D={case['head_dim']}"
        )
        lines.append(
            f"| `{case_label}` | `{row['best_config']}` | {row['best_value_ms']:.3f} | "
            f"{row['default_value_ms']:.3f} | {row['speedup_vs_default']:.2f}x |"
        )
    lines.extend(["", "## Grouped winner counts", ""])
    for group_name, buckets in summary["grouped"].items():
        lines.append(f"### {group_name}")
        lines.append("")
        lines.append("| Bucket | Cases | Winner counts |")
        lines.append("| --- | ---: | --- |")
        for bucket, data in buckets.items():
            winners = ", ".join(f"`{name}` x{count}" for name, count in data["winner_counts"].items()) or "`none`"
            lines.append(f"| `{bucket}` | {data['num_cases']} | {winners} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run repeatable chunked FLARE launch-config sweeps over a D/M/N/BH matrix."
    )
    parser.add_argument("--d-values", default=None, help="Comma-separated D values. Default uses the fast anchor matrix.")
    parser.add_argument("--m-values", default=None, help="Comma-separated M values. Default uses the fast anchor matrix.")
    parser.add_argument("--n-values", default=None, help="Comma-separated N values. Default uses the fast anchor matrix.")
    parser.add_argument(
        "--bh-values",
        default=None,
        help="Comma-separated BH values. By default this stays at one representative BH because the kernels parallelize over BH.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size used to realize each BH value. Heads are derived as H=BH/B.",
    )
    parser.add_argument(
        "--families",
        default=None,
        help=(
            "Comma-separated family/group list. Default depends on --mode and --full-matrix. "
            "Groups include fast_forward, fast_backward, core_fast, full_forward, full_backward, core, extended, all."
        ),
    )
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="Use the exhaustive historical shape matrix and broader default family sets instead of the fast anchor defaults.",
    )
    parser.add_argument(
        "--extra-config",
        action="append",
        help="Repeatable extra named env config: name:key=value,key=value",
    )
    parser.add_argument("--chunk-sizes", default="32,64,128,256", help="Comma-separated CHUNK_SIZE candidates.")
    parser.add_argument("--forward-block-ks", default="16,32,64,128", help="Comma-separated forward BLOCK_K candidates.")
    parser.add_argument("--backward-block-ks", default="16,32,64,128", help="Comma-separated backward BLOCK_K candidates.")
    parser.add_argument("--forward-block-ds", default="16,32,64,128,256", help="Comma-separated forward BLOCK_D candidates.")
    parser.add_argument("--backward-block-ds", default="16,32,64,128,256", help="Comma-separated backward BLOCK_*D candidates.")
    parser.add_argument("--block-ts", default="16,32,64", help="Comma-separated BLOCK_T candidates for backward token-panel sweeps.")
    parser.add_argument("--scalar-apply-panels", default="4,8,16", help="Comma-separated scalar apply panel candidates.")
    parser.add_argument("--dtype", default="bf16", help="bf16, fp16, or fp32.")
    parser.add_argument("--input-precision", default=None, help="ieee, tf32, or tf32x3. Default uses repo default.")
    parser.add_argument(
        "--mode",
        choices=("forward", "backward", "both"),
        default="both",
        help="Whether to benchmark forward only or include backward work in the objective.",
    )
    parser.add_argument(
        "--objective",
        choices=("end_to_end_ms", "forward_profile_total_ms", "backward_profile_total_ms"),
        default="end_to_end_ms",
        help="Metric used to rank candidates inside each case.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cases", type=int, default=None, help="Optional cap after sharding, useful for smoke runs.")
    parser.add_argument("--num-shards", type=int, default=1, help="Split the case list into N shards for parallel tuning jobs.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index to run from the sharded case list.")
    parser.add_argument("--jsonl", type=Path, default=None, help="Optional JSONL checkpoint file. Appends one record per run.")
    parser.add_argument("--summary-json", type=Path, default=None, help="Optional JSON summary output.")
    parser.add_argument("--summary-md", type=Path, default=None, help="Optional markdown summary output.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional artifact directory name under results/chunked_flare_matrix. Set this when you want stable resume paths.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing --jsonl records and skip already completed case/config pairs.",
    )
    return parser


def resolve_output_paths(
    *,
    run_name: str | None,
    jsonl_path: Path | None,
    summary_json_path: Path | None,
    summary_md_path: Path | None,
    shard_index: int,
    num_shards: int,
    mode: str,
    objective: str,
) -> tuple[str, Path, Path, Path]:
    if run_name is None:
        run_name = datetime.now().strftime(f"{mode}-{objective}-%Y%m%d-%H%M%S")
    root = Path("results") / "chunked_flare_matrix" / run_name
    shard_stem = f"shard{shard_index:03d}-of-{num_shards:03d}"
    resolved_jsonl = jsonl_path or (root / f"runs-{shard_stem}.jsonl")
    resolved_summary_json = summary_json_path or (root / f"summary-{shard_stem}.json")
    resolved_summary_md = summary_md_path or (root / f"summary-{shard_stem}.md")
    return run_name, resolved_jsonl, resolved_summary_json, resolved_summary_md


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    dtype = parse_dtype(args.dtype)
    default_d_values, default_m_values, default_n_values, default_bh_values = default_matrix_values(full_matrix=args.full_matrix)
    d_values = parse_int_list(args.d_values, default=default_d_values)
    m_values = parse_int_list(args.m_values, default=default_m_values)
    n_values = parse_int_list(args.n_values, default=default_n_values)
    bh_values = parse_int_list(args.bh_values, default=default_bh_values)
    families = parse_family_list(args.families or default_family_group(mode=args.mode, full_matrix=args.full_matrix))
    chunk_sizes = parse_int_list(args.chunk_sizes, default=DEFAULT_CHUNK_SIZES)
    forward_block_ks = parse_int_list(args.forward_block_ks, default=DEFAULT_FORWARD_BLOCK_KS)
    backward_block_ks = parse_int_list(args.backward_block_ks, default=DEFAULT_BACKWARD_BLOCK_KS)
    forward_block_ds = parse_int_list(args.forward_block_ds, default=DEFAULT_FORWARD_BLOCK_DS)
    backward_block_ds = parse_int_list(args.backward_block_ds, default=DEFAULT_BACKWARD_BLOCK_DS)
    block_ts = parse_int_list(args.block_ts, default=DEFAULT_BLOCK_TS)
    scalar_apply_panels = parse_int_list(args.scalar_apply_panels, default=DEFAULT_SCALAR_APPLY_PANELS)
    extra_configs = [parse_extra_config(spec) for spec in (args.extra_config or [])]
    if args.resume and args.jsonl is None and args.run_name is None:
        raise ValueError("Resume mode requires either --jsonl or --run-name so the script can find the previous raw results.")

    cases = build_cases(
        d_values=d_values,
        m_values=m_values,
        n_values=n_values,
        bh_values=bh_values,
        batch_size=args.batch_size,
    )
    cases = select_case_shard(cases, shard_index=args.shard_index, num_shards=args.num_shards)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]

    run_name, jsonl_path, summary_json_path, summary_md_path = resolve_output_paths(
        run_name=args.run_name,
        jsonl_path=args.jsonl,
        summary_json_path=args.summary_json,
        summary_md_path=args.summary_md,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        mode=args.mode,
        objective=args.objective,
    )

    existing_by_key, existing_records = load_existing_results(jsonl_path if args.resume else None)
    results_by_key = {record_key(record): record for record in existing_records}
    dtype_name = str(dtype).removeprefix("torch.")

    for case in cases:
        case_candidates, _ = expand_candidates_for_case(
            case,
            families=tuple(family for family in families if family != "combined"),
            chunk_sizes=chunk_sizes,
            forward_block_ks=forward_block_ks,
            backward_block_ks=backward_block_ks,
            forward_block_ds=forward_block_ds,
            backward_block_ds=backward_block_ds,
            block_ts=block_ts,
            scalar_apply_panels=scalar_apply_panels,
            extra_configs=extra_configs,
        )
        case_results: list[dict] = []
        for candidate in case_candidates:
            key = result_key(
                case,
                candidate,
                dtype=str(dtype).removeprefix("torch."),
                input_precision=args.input_precision,
                mode=args.mode,
            )
            if args.resume and key in existing_by_key:
                record = existing_by_key[key]
            else:
                try:
                    record = bench_case(
                        case,
                        dtype=dtype,
                        warmup=args.warmup,
                        iterations=args.iterations,
                        input_precision=args.input_precision,
                        mode=args.mode,
                        env=candidate.env,
                        seed=args.seed,
                    )
                    record["config_name"] = candidate.name
                    record["family"] = candidate.family
                    record["status"] = "ok"
                    record["error"] = ""
                except Exception as exc:  # noqa: BLE001
                    record = build_error_record(
                        case=case,
                        candidate=candidate,
                        dtype=dtype_name,
                        input_precision=args.input_precision,
                        mode=args.mode,
                        error=exc,
                    )
                append_jsonl(jsonl_path, record)
                existing_by_key[key] = record
            case_results.append(record)
            results_by_key[key] = record

        if "combined" in families:
            combined = merge_family_winners(case_results, families=families, objective=args.objective)
            if combined is not None:
                key = result_key(
                    case,
                    combined,
                    dtype=str(dtype).removeprefix("torch."),
                    input_precision=args.input_precision,
                    mode=args.mode,
                )
                if args.resume and key in existing_by_key:
                    record = existing_by_key[key]
                else:
                    try:
                        record = bench_case(
                            case,
                            dtype=dtype,
                            warmup=args.warmup,
                            iterations=args.iterations,
                            input_precision=args.input_precision,
                            mode=args.mode,
                            env=combined.env,
                            seed=args.seed,
                        )
                        record["config_name"] = combined.name
                        record["family"] = combined.family
                        record["status"] = "ok"
                        record["error"] = ""
                    except Exception as exc:  # noqa: BLE001
                        record = build_error_record(
                            case=case,
                            candidate=combined,
                            dtype=dtype_name,
                            input_precision=args.input_precision,
                            mode=args.mode,
                            error=exc,
                        )
                    append_jsonl(jsonl_path, record)
                    existing_by_key[key] = record
                results_by_key[key] = record

    summary = summarize_results(list(results_by_key.values()), objective=args.objective)
    summary["run_name"] = run_name
    summary["mode"] = args.mode
    summary["dtype"] = str(dtype).removeprefix("torch.")
    summary["input_precision"] = args.input_precision or "default"
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

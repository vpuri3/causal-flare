#!/usr/bin/env python3
"""Run Nsight Compute against an arbitrary command with a reusable section preset."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SECTIONS = (
    "LaunchStats",
    "Occupancy",
    "SchedulerStats",
    "WarpStateStats",
    "SpeedOfLight",
    "ComputeWorkloadAnalysis",
    "MemoryWorkloadAnalysis",
    "InstructionStats",
)


def _cuda_version_key(path: Path) -> tuple[int, ...]:
    match = re.search(r"cuda-(\d+)\.(\d+)", str(path))
    if match:
        return int(match.group(1)), int(match.group(2))
    if "cuda" in path.parts:
        return (999, 999)
    return (0, 0)


def find_ncu_binary(explicit: str | None = None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_ncu = os.environ.get("NCU_BIN", "").strip()
    if env_ncu:
        candidates.append(Path(env_ncu))
    which_ncu = shutil.which("ncu")
    if which_ncu:
        candidates.append(Path(which_ncu))
    candidates.extend(sorted(Path("/usr/local").glob("cuda-*/bin/ncu"), key=_cuda_version_key, reverse=True))
    candidates.append(Path("/usr/local/cuda/bin/ncu"))

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        "Could not find an `ncu` binary. Checked $NCU_BIN, PATH, /usr/local/cuda-*/bin/ncu, and /usr/local/cuda/bin/ncu."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ncu-bin", help="Explicit path to the ncu binary.")
    parser.add_argument("--show-ncu", action="store_true", help="Print the resolved ncu path and exit.")
    parser.add_argument(
        "--list-default-sections",
        action="store_true",
        help="Print the default section preset and exit.",
    )
    parser.add_argument(
        "--section",
        action="append",
        default=[],
        help="Repeatable Nsight Compute section name. Defaults to a useful kernel-diagnosis preset.",
    )
    parser.add_argument("--kernel-name", help="Kernel name or regex passed to --kernel-name.")
    parser.add_argument(
        "--kernel-name-base",
        default="function",
        help="Value for --kernel-name-base. Defaults to function.",
    )
    parser.add_argument("--launch-skip", type=int, default=1, help="Value for --launch-skip. Defaults to 1.")
    parser.add_argument("--launch-count", type=int, default=1, help="Value for --launch-count. Defaults to 1.")
    parser.add_argument(
        "--target-processes",
        default="all",
        help="Value for --target-processes. Defaults to all.",
    )
    parser.add_argument("--page", help="Optional value for --page.")
    parser.add_argument("--csv", action="store_true", help="Pass --csv to ncu.")
    parser.add_argument("-o", "--output", help="Optional ncu report output path.")
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Repeatable environment variable override forwarded to the profiled command.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Print the resolved ncu command and exit without running it.",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to profile. Separate it from this wrapper with `--`.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ncu_bin = find_ncu_binary(args.ncu_bin)

    if args.show_ncu:
        print(ncu_bin)
        return 0
    if args.list_default_sections:
        print("\n".join(DEFAULT_SECTIONS))
        return 0

    if not args.kernel_name:
        raise SystemExit("--kernel-name is required unless --show-ncu or --list-default-sections is used.")

    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("A profiled command is required after `--`.")

    sections = list(args.section) or list(DEFAULT_SECTIONS)
    ncu_cmd = [
        str(ncu_bin),
        "--target-processes",
        args.target_processes,
        "--kernel-name-base",
        args.kernel_name_base,
        "--kernel-name",
        args.kernel_name,
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
    ]
    for section in sections:
        ncu_cmd.extend(["--section", section])
    if args.page:
        ncu_cmd.extend(["--page", args.page])
    if args.csv:
        ncu_cmd.append("--csv")
    if args.output:
        ncu_cmd.extend(["-o", args.output])
    ncu_cmd.extend(command)

    env = os.environ.copy()
    for item in args.env:
        if "=" not in item:
            raise SystemExit(f"Invalid --env entry `{item}`. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        env[key] = value

    print(f"[profile_kernel_ncu] using ncu: {ncu_bin}", file=sys.stderr)
    print(f"[profile_kernel_ncu] command: {' '.join(ncu_cmd)}", file=sys.stderr)
    if args.print_only:
        return 0

    result = subprocess.run(ncu_cmd, env=env)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())

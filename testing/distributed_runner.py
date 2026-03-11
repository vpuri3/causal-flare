from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from testing.suite_runners.parity import _parity_tests
from testing.suite_runners.regression_bundle import _apply_regression_defaults
from testing.suite_runners.trainlike_multistep_parity import _trainlike_multistep_parity
from testing.suite_runners.trainlike_sanity import _trainlike_sanity


@dataclass(frozen=True)
class Task:
    kind: str
    name: str
    weight: int

    @property
    def label(self) -> str:
        return f"{self.kind}:{self.name}"


@dataclass(frozen=True)
class TaskResult:
    task: Task
    gpu_id: str
    worker_index: int
    returncode: int
    log_path: Path


@dataclass(frozen=True)
class WorkerSlot:
    gpu_id: str
    worker_index: int

    @property
    def label(self) -> str:
        return f"gpu={self.gpu_id}/worker={self.worker_index}"


_BUNDLE_NODEID = "testing/test_regression_suites.py::test_regression_bundle"
_DEFAULT_PYTEST_ARGS = ["testing", "-q"]

_SUITE_RUNNERS: dict[str, Callable[[], None]] = {
    "parity": _parity_tests,
    "trainlike_sanity": _trainlike_sanity,
    "trainlike_multistep_parity": _trainlike_multistep_parity,
}

_WEIGHT_OVERRIDES = {
    "testing/test_cached_suites.py::test_cached_prefill_decode_regression_suite": 6,
}

_PYTEST_OPTIONS_WITH_VALUE = {
    "-c",
    "-k",
    "-m",
    "-o",
    "--basetemp",
    "--capture",
    "--confcutdir",
    "--deselect",
    "--durations",
    "--durations-min",
    "--ignore",
    "--ignore-glob",
    "--junitxml",
    "--log-cli-level",
    "--log-file",
    "--log-file-level",
    "--maxfail",
    "--rootdir",
    "--tb",
    "--verbosity",
}


def _normalize_pytest_args(raw_args: list[str]) -> list[str]:
    args = list(raw_args)
    if args and args[0] == "--":
        args = args[1:]
    return args or list(_DEFAULT_PYTEST_ARGS)


def _ensure_test_env_defaults(env: dict[str, str] | None = None, *, reduce_autotune: bool) -> dict[str, str] | None:
    target = os.environ if env is None else env
    if reduce_autotune:
        target.setdefault("FLARE_TEST_REDUCE_AUTOTUNE", "1")
    else:
        target.pop("FLARE_TEST_REDUCE_AUTOTUNE", None)
    return env


def _visible_gpu_ids(override: str | None = None) -> list[str]:
    raw = override if override is not None else os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if raw:
        return [token.strip() for token in raw.split(",") if token.strip()]

    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []
    return [str(idx) for idx in range(torch.cuda.device_count())]


def _parse_collected_nodeids(stdout: str) -> list[str]:
    nodeids = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith("testing/"):
            nodeids.append(stripped)
    return nodeids


def _collection_pytest_args(pytest_args: list[str]) -> list[str]:
    args: list[str] = []
    for arg in pytest_args:
        if arg == "--quiet":
            continue
        if arg.startswith("-") and set(arg[1:]) == {"q"}:
            continue
        args.append(arg)
    return [*args, "--collect-only", "-q"]


def _execution_pytest_args(pytest_args: list[str]) -> list[str]:
    args: list[str] = []
    expect_value = False
    for arg in pytest_args:
        if expect_value:
            args.append(arg)
            expect_value = False
            continue
        if arg == "--":
            break
        if arg.startswith("--") and "=" in arg:
            args.append(arg)
            continue
        if arg in _PYTEST_OPTIONS_WITH_VALUE:
            args.append(arg)
            expect_value = True
            continue
        if arg.startswith("-"):
            args.append(arg)
            continue
        # Drop positional test selectors like "testing" when rerunning a single nodeid.
    return args


def _regression_bundle_replacement_tasks(*, full_matrix: bool) -> list[Task]:
    tasks = [
        Task(kind="suite", name="parity", weight=6),
        Task(kind="suite", name="trainlike_sanity", weight=7),
    ]
    if full_matrix:
        tasks.append(Task(kind="suite", name="trainlike_multistep_parity", weight=10))
    return tasks


def _expand_tasks(nodeids: list[str], *, full_matrix: bool) -> list[Task]:
    tasks: list[Task] = []
    for nodeid in nodeids:
        if nodeid == _BUNDLE_NODEID:
            tasks.extend(_regression_bundle_replacement_tasks(full_matrix=full_matrix))
            continue
        tasks.append(Task(kind="pytest", name=nodeid, weight=_weight_for_nodeid(nodeid)))
    return tasks


def _weight_for_nodeid(nodeid: str) -> int:
    if nodeid in _WEIGHT_OVERRIDES:
        return _WEIGHT_OVERRIDES[nodeid]
    if nodeid.startswith("testing/test_regression_suites.py::test_correctness_suite["):
        return 3
    if nodeid.startswith("testing/test_regression_suites.py::test_grad_checks_suite["):
        return 3
    if nodeid.startswith("testing/test_regression_suites.py::test_autotune_launch_coverage_suite["):
        return 2
    if nodeid.startswith("testing/test_regression_suites.py::test_long_context_accuracy_suite["):
        return 3
    if nodeid.startswith("testing/test_regression_suites.py::test_chunk_size_sensitivity_suite["):
        return 2
    if nodeid.startswith("testing/test_regression_suites.py::test_sharp_softmax_bwd_regression_suite["):
        return 2
    if "test_finite_difference.py::" in nodeid:
        return 3
    if "test_inference_prefill_decode_variants_match_pytorch" in nodeid:
        return 2
    return 1


def _schedule_tasks(tasks: list[Task], gpu_ids: list[str]) -> list[list[Task]]:
    if not gpu_ids:
        raise ValueError("Cannot schedule tasks without at least one GPU id.")

    assignments: list[list[Task]] = [[] for _ in gpu_ids]
    loads = [0 for _ in gpu_ids]
    for task in sorted(tasks, key=lambda item: (-item.weight, item.label)):
        shard_idx = min(range(len(gpu_ids)), key=lambda idx: (loads[idx], idx))
        assignments[shard_idx].append(task)
        loads[shard_idx] += task.weight
    return assignments


def _build_worker_slots(gpu_ids: list[str], workers_per_gpu: int) -> list[WorkerSlot]:
    if workers_per_gpu < 1:
        raise ValueError("workers_per_gpu must be at least 1.")
    return [
        WorkerSlot(gpu_id=gpu_id, worker_index=worker_index)
        for gpu_id in gpu_ids
        for worker_index in range(workers_per_gpu)
    ]


def _collect_nodeids(pytest_args: list[str], repo_root: Path) -> list[str]:
    cmd = [sys.executable, "-m", "pytest", *_collection_pytest_args(pytest_args)]
    result = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "pytest collection failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    nodeids = _parse_collected_nodeids(result.stdout)
    if not nodeids:
        raise RuntimeError("pytest collection produced no test nodeids.")
    return nodeids


def _run_suite_task(task_name: str, *, reduce_autotune: bool) -> None:
    runner = _SUITE_RUNNERS.get(task_name)
    if runner is None:
        known = ", ".join(sorted(_SUITE_RUNNERS))
        raise ValueError(f"Unknown suite task '{task_name}'. Known: {known}")
    _ensure_test_env_defaults(reduce_autotune=reduce_autotune)
    _apply_regression_defaults()
    runner()


def _task_command(task: Task, pytest_args: list[str]) -> list[str]:
    if task.kind == "pytest":
        return [sys.executable, "-m", "pytest", *_execution_pytest_args(pytest_args), task.name]
    if task.kind == "suite":
        return [sys.executable, "-m", "testing.distributed_runner", "--run-suite", task.name]
    raise ValueError(f"Unknown task kind '{task.kind}'")


def _task_rerun_hint(task: Task, gpu_id: str, pytest_args: list[str]) -> str:
    cmd = _task_command(task, pytest_args)
    return f"CUDA_VISIBLE_DEVICES={gpu_id} {' '.join(cmd)}"


def _write_shard_header(log_file, shard_index: int, gpu_id: str, tasks: list[Task]) -> None:
    log_file.write(f"[distributed-runner] shard={shard_index} gpu={gpu_id} tasks={len(tasks)}\n")
    for task in tasks:
        log_file.write(f"[distributed-runner] queued {task.label} weight={task.weight}\n")
    log_file.flush()


def _run_shard(
    shard_index: int,
    slot: WorkerSlot,
    tasks: list[Task],
    pytest_args: list[str],
    repo_root: Path,
    log_dir: Path,
    reduce_autotune: bool,
) -> list[TaskResult]:
    results: list[TaskResult] = []
    log_path = log_dir / f"shard-{shard_index:02d}-gpu-{slot.gpu_id}-worker-{slot.worker_index}.log"
    env = os.environ.copy()
    _ensure_test_env_defaults(env, reduce_autotune=reduce_autotune)
    env["CUDA_VISIBLE_DEVICES"] = slot.gpu_id
    env.setdefault("PYTHONUNBUFFERED", "1")

    with log_path.open("w", encoding="utf-8") as log_file:
        _write_shard_header(log_file, shard_index, slot.label, tasks)
        for task in tasks:
            cmd = _task_command(task, pytest_args)
            log_file.write(f"\n[distributed-runner] starting {task.label}\n")
            log_file.write(f"[distributed-runner] command: {' '.join(cmd)}\n")
            log_file.flush()
            completed = subprocess.run(
                cmd,
                cwd=repo_root,
                env=env,
                text=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            )
            results.append(
                TaskResult(
                    task=task,
                    gpu_id=slot.gpu_id,
                    worker_index=slot.worker_index,
                    returncode=completed.returncode,
                    log_path=log_path,
                )
            )
            log_file.write(f"[distributed-runner] finished {task.label} rc={completed.returncode}\n")
            log_file.flush()
            if completed.returncode != 0:
                break

    return results


def _print_plan(assignments: list[list[Task]], slots: list[WorkerSlot]) -> None:
    print("[distributed-runner] task plan")
    for shard_index, tasks in enumerate(assignments):
        weight = sum(task.weight for task in tasks)
        labels = ", ".join(task.label for task in tasks) if tasks else "<idle>"
        slot = slots[shard_index]
        print(f"  shard={shard_index} {slot.label} weight={weight} tasks={len(tasks)} :: {labels}")


def _run_distributed(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    _ensure_test_env_defaults(reduce_autotune=True)
    pytest_args = _normalize_pytest_args(args.pytest_args)
    gpu_ids = _visible_gpu_ids(args.visible_gpus)
    if not gpu_ids:
        raise RuntimeError("No visible GPUs detected. Set CUDA_VISIBLE_DEVICES or use a CUDA-enabled environment.")

    nodeids = _collect_nodeids(pytest_args, repo_root)
    tasks = _expand_tasks(nodeids, full_matrix=args.full_matrix)
    if not tasks:
        raise RuntimeError("No runnable distributed tasks were produced from pytest collection.")

    slots = _build_worker_slots(gpu_ids, args.workers_per_gpu)
    assignments = _schedule_tasks(tasks, [slot.label for slot in slots])
    if args.dry_run:
        _print_plan(assignments, slots)
        return 0

    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    _print_plan(assignments, slots)

    failures: list[TaskResult] = []
    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        future_map = {
            pool.submit(
                _run_shard,
                shard_index,
                slots[shard_index],
                assignments[shard_index],
                pytest_args,
                repo_root,
                log_dir,
                not args.full_matrix,
            ): shard_index
            for shard_index in range(len(slots))
            if assignments[shard_index]
        }
        for future in as_completed(future_map):
            shard_results = future.result()
            failures.extend(result for result in shard_results if result.returncode != 0)

    if failures:
        print("[distributed-runner] failures detected")
        for result in failures:
            print(
                f"  {result.task.label} on gpu={result.gpu_id}/worker={result.worker_index} "
                f"failed rc={result.returncode} log={result.log_path}"
            )
            print(f"    rerun: {_task_rerun_hint(result.task, result.gpu_id, pytest_args)}")
        return 1

    print("[distributed-runner] all shards completed successfully")
    return 0


def _run_single_suite(args: argparse.Namespace) -> int:
    _run_suite_task(args.run_suite, reduce_autotune=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Shard the collected pytest suite across visible GPUs.")
    parser.add_argument("--visible-gpus", default=None, help="Optional explicit GPU id list, e.g. '0,1,2,3'.")
    parser.add_argument("--workers-per-gpu", type=int, default=1, help="Independent workers to launch per visible GPU.")
    parser.add_argument("--log-dir", default="/tmp/flare-pytest-shards", help="Directory for per-shard logs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the shard plan without running anything.")
    parser.add_argument("--run-suite", default=None, choices=sorted(_SUITE_RUNNERS), help=argparse.SUPPRESS)
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Arguments forwarded to pytest collection/execution.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.full_matrix = "--full-matrix" in _normalize_pytest_args(args.pytest_args)
    if args.run_suite is not None:
        raise SystemExit(_run_single_suite(args))
    raise SystemExit(_run_distributed(args))


if __name__ == "__main__":
    main()

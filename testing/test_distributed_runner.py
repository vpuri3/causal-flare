import pytest

import testing.distributed_runner as distributed_runner


def test_visible_gpu_ids_returns_requested_prefix_of_available(monkeypatch) -> None:
    monkeypatch.setattr(distributed_runner, "_filter_usable_gpu_ids", lambda gpu_ids: ["0", "1", "2"])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert distributed_runner._visible_gpu_ids(2) == ["0", "1"]


def test_filter_usable_gpu_ids_drops_unusable_requested_gpus() -> None:
    checker = lambda gpu_id: gpu_id in {"0", "2"}
    assert distributed_runner._filter_usable_gpu_ids(["0", "1", "2", "3"], checker=checker) == ["0", "2"]


def test_visible_gpu_ids_errors_when_requested_count_exceeds_usable_count(monkeypatch) -> None:
    monkeypatch.setattr(distributed_runner, "_filter_usable_gpu_ids", lambda requested_gpu_ids: ["0"])
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError) as exc_info:
        distributed_runner._visible_gpu_ids(2)
    message = str(exc_info.value)
    assert "Requested GPU count exceeds available usable GPU count" in message
    assert "requested_count=2" in message
    assert "usable_gpu_ids=['0']" in message


def test_visible_gpu_ids_defaults_to_all_available(monkeypatch) -> None:
    monkeypatch.setattr(distributed_runner, "_filter_usable_gpu_ids", lambda gpu_ids: ["0", "2"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2")
    assert distributed_runner._visible_gpu_ids() == ["0", "2"]


def test_parse_collected_nodeids_filters_pytest_summary_lines() -> None:
    stdout = "\n".join(
        [
            "testing/autoregressive/test_flare.py::test_one",
            "testing/autoregressive/test_regression_suites.py::test_regression_bundle",
            "",
            "2 tests collected in 0.10s",
        ]
    )
    assert distributed_runner._parse_collected_nodeids(stdout) == [
        "testing/autoregressive/test_flare.py::test_one",
        "testing/autoregressive/test_regression_suites.py::test_regression_bundle",
    ]


def test_collection_pytest_args_collapses_existing_quiet_flags() -> None:
    assert distributed_runner._collection_pytest_args(["testing", "-q", "--run-regression"]) == [
        "testing",
        "--run-regression",
        "--collect-only",
        "-q",
    ]


def test_run_suite_task_applies_regression_defaults(monkeypatch) -> None:
    seen = {"called": False}

    def fake_apply_defaults() -> None:
        seen["defaults"] = True

    def fake_runner() -> None:
        seen["called"] = True

    monkeypatch.setattr(distributed_runner, "_apply_regression_defaults", fake_apply_defaults)
    monkeypatch.setitem(distributed_runner._SUITE_RUNNERS, "parity", fake_runner)
    distributed_runner._run_suite_task("parity")
    assert seen == {"called": True, "defaults": True}


def test_execution_pytest_args_drops_root_selector_but_keeps_flags() -> None:
    assert distributed_runner._execution_pytest_args(
        ["testing", "--run-regression", "--full-matrix", "-q", "-k", "sharp"]
    ) == [
        "--run-regression",
        "--full-matrix",
        "-q",
        "-k",
        "sharp",
    ]


def test_expand_tasks_replaces_regression_bundle_with_unique_components() -> None:
    tasks = distributed_runner._expand_tasks(
        [
            "testing/autoregressive/test_flare.py::test_one",
            "testing/autoregressive/test_regression_suites.py::test_regression_bundle",
        ],
        full_matrix=False,
    )
    assert [task.label for task in tasks] == [
        "pytest:testing/autoregressive/test_flare.py::test_one",
        "suite:parity",
        "suite:trainlike_sanity",
    ]


def test_expand_tasks_adds_multistep_suite_for_full_matrix() -> None:
    tasks = distributed_runner._expand_tasks(
        ["testing/autoregressive/test_regression_suites.py::test_regression_bundle"],
        full_matrix=True,
    )
    assert [task.label for task in tasks] == [
        "suite:parity",
        "suite:trainlike_sanity",
        "suite:trainlike_multistep_parity",
    ]


def test_build_worker_slots_expands_each_gpu() -> None:
    slots = distributed_runner._build_worker_slots(["0", "1"], 2)
    assert [slot.label for slot in slots] == [
        "gpu=0/worker=0",
        "gpu=0/worker=1",
        "gpu=1/worker=0",
        "gpu=1/worker=1",
    ]


def test_schedule_tasks_balances_weighted_workloads() -> None:
    tasks = [
        distributed_runner.Task(kind="pytest", name="heavy", weight=10),
        distributed_runner.Task(kind="pytest", name="medium", weight=6),
        distributed_runner.Task(kind="pytest", name="light_a", weight=1),
        distributed_runner.Task(kind="pytest", name="light_b", weight=1),
    ]
    slots = distributed_runner._build_worker_slots(["0", "1"], 1)
    assignments = distributed_runner._schedule_tasks(tasks, slots)
    labels = [[task.name for task in shard] for shard in assignments]
    assert labels == [["heavy"], ["medium", "light_a", "light_b"]]


def test_schedule_tasks_keeps_autotune_coverage_on_worker_zero_slots() -> None:
    tasks = [
        distributed_runner.Task(
            kind="pytest",
            name="testing/autoregressive/test_regression_suites.py::test_autotune_launch_coverage_suite[autotune1]",
            weight=2,
        ),
        distributed_runner.Task(
            kind="pytest",
            name="testing/autoregressive/test_regression_suites.py::test_autotune_launch_coverage_suite[autotune2]",
            weight=2,
        ),
        distributed_runner.Task(kind="pytest", name="ordinary", weight=1),
    ]
    slots = distributed_runner._build_worker_slots(["0", "1"], 2)
    assignments = distributed_runner._schedule_tasks(tasks, slots)
    coverage_slots = [
        slots[idx].worker_index
        for idx, shard in enumerate(assignments)
        for task in shard
        if distributed_runner._is_autotune_coverage_task(task)
    ]
    assert coverage_slots == [0, 0]


def test_task_command_targets_single_nodeid_without_collecting_testing_root() -> None:
    task = distributed_runner.Task(kind="pytest", name="testing/autoregressive/test_flare.py::test_one", weight=1)
    command = distributed_runner._task_command(task, ["testing", "--run-regression", "-q"])
    assert command == [
        distributed_runner.sys.executable,
        "-m",
        "pytest",
        "--run-regression",
        "-q",
        "testing/autoregressive/test_flare.py::test_one",
    ]

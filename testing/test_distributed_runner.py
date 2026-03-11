import testing.distributed_runner as distributed_runner


def test_visible_gpu_ids_prefers_explicit_override() -> None:
    assert distributed_runner._visible_gpu_ids("3,5") == ["3", "5"]


def test_parse_collected_nodeids_filters_pytest_summary_lines() -> None:
    stdout = "\n".join(
        [
            "testing/test_flare.py::test_one",
            "testing/test_regression_suites.py::test_regression_bundle",
            "",
            "2 tests collected in 0.10s",
        ]
    )
    assert distributed_runner._parse_collected_nodeids(stdout) == [
        "testing/test_flare.py::test_one",
        "testing/test_regression_suites.py::test_regression_bundle",
    ]


def test_collection_pytest_args_collapses_existing_quiet_flags() -> None:
    assert distributed_runner._collection_pytest_args(["testing", "-q", "--run-regression"]) == [
        "testing",
        "--run-regression",
        "--collect-only",
        "-q",
    ]


def test_run_suite_task_sets_reduced_autotune_env(monkeypatch) -> None:
    seen = {}

    def fake_runner() -> None:
        seen["env"] = distributed_runner.os.environ.get("FLARE_TEST_REDUCE_AUTOTUNE")

    monkeypatch.delenv("FLARE_TEST_REDUCE_AUTOTUNE", raising=False)
    monkeypatch.setitem(distributed_runner._SUITE_RUNNERS, "parity", fake_runner)
    distributed_runner._run_suite_task("parity", reduce_autotune=True)
    assert seen["env"] == "1"


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
            "testing/test_flare.py::test_one",
            "testing/test_regression_suites.py::test_regression_bundle",
        ],
        full_matrix=False,
    )
    assert [task.label for task in tasks] == [
        "pytest:testing/test_flare.py::test_one",
        "suite:parity",
        "suite:trainlike_sanity",
    ]


def test_expand_tasks_adds_multistep_suite_for_full_matrix() -> None:
    tasks = distributed_runner._expand_tasks(
        ["testing/test_regression_suites.py::test_regression_bundle"],
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
    assignments = distributed_runner._schedule_tasks(tasks, ["0", "1"])
    labels = [[task.name for task in shard] for shard in assignments]
    assert labels == [["heavy"], ["medium", "light_a", "light_b"]]


def test_task_command_targets_single_nodeid_without_collecting_testing_root() -> None:
    task = distributed_runner.Task(kind="pytest", name="testing/test_flare.py::test_one", weight=1)
    command = distributed_runner._task_command(task, ["testing", "--run-regression", "-q"])
    assert command == [
        distributed_runner.sys.executable,
        "-m",
        "pytest",
        "--run-regression",
        "-q",
        "testing/test_flare.py::test_one",
    ]

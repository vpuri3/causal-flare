import os

import pytest


os.environ.setdefault("FLARE_TEST_REDUCE_AUTOTUNE", "1")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-regression",
        action="store_true",
        default=False,
        help="Run regression-marked suites (slower than default unit tests).",
    )
    parser.addoption(
        "--run-stress",
        action="store_true",
        default=False,
        help="Run stress-marked suites (long-running, typically nightly).",
    )
    parser.addoption(
        "--full-matrix",
        action="store_true",
        default=False,
        help="Run full regression/stress matrices without reduced bounded defaults.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_regression = config.getoption("--run-regression")
    run_stress = config.getoption("--run-stress")

    skip_regression = pytest.mark.skip(reason="needs --run-regression option to run")
    skip_stress = pytest.mark.skip(reason="needs --run-stress option to run")

    for item in items:
        if "stress" in item.keywords and not run_stress:
            item.add_marker(skip_stress)
        if "regression" in item.keywords and not run_regression:
            item.add_marker(skip_regression)

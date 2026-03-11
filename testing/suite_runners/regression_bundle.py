"""Extracted regression/stress suite implementation."""

from testing.suite_runners.common import *

from testing.suite_runners.autotune_launch_coverage import _autotune_launch_coverage_suite_shard
from testing.suite_runners.chunk_size_sensitivity import _chunk_size_sensitivity_suite_shard
from testing.suite_runners.correctness import _run_correctness_suite_shard
from testing.suite_runners.grad_checks import _run_grad_checks_suite_shard
from testing.suite_runners.long_context_accuracy import _long_context_accuracy_suite_shard
from testing.suite_runners.parity import _parity_tests
from testing.suite_runners.sharp_softmax_bwd_regression import _sharp_softmax_bwd_regression_suite_shard
from testing.suite_runners.trainlike_multistep_parity import _trainlike_multistep_parity
from testing.suite_runners.trainlike_sanity import _trainlike_sanity

_CORRECTNESS_NUM_SHARDS = 8
_GRAD_CHECKS_NUM_SHARDS = 8
_AUTOTUNE_COVERAGE_NUM_SHARDS = 8
_SHARP_BWD_NUM_SHARDS = 12
_LONGCTX_NUM_SHARDS = 8
_CHUNK_SENS_NUM_SHARDS = 8


def _apply_regression_defaults() -> None:
    # Deterministic, bounded runtime defaults for CI/local gating.
    os.environ["FLARE_CORRECTNESS_STRICT"] = "1"
    os.environ["FLARE_CORRECTNESS_DTYPES"] = os.environ.get("FLARE_CORRECTNESS_DTYPES", "bfloat16")
    os.environ["FLARE_CORRECTNESS_SHAPES"] = os.environ.get("FLARE_CORRECTNESS_SHAPES", "1,2,512,128,32")
    os.environ["FLARE_CORRECTNESS_QK_STDS"] = os.environ.get("FLARE_CORRECTNESS_QK_STDS", "1.0")
    os.environ["FLARE_CORRECTNESS_DECODE_SEPARATION_MODES"] = os.environ.get(
        "FLARE_CORRECTNESS_DECODE_SEPARATION_MODES", "00,10,01,11"
    )
    os.environ["FLARE_CORRECTNESS_GRAD"] = os.environ.get("FLARE_CORRECTNESS_GRAD", "1")
    os.environ["FLARE_CORRECTNESS_GRAD_LIMIT"] = os.environ.get("FLARE_CORRECTNESS_GRAD_LIMIT", "4")

    os.environ["FLARE_PARITY_STRICT"] = "1"
    os.environ["FLARE_PARITY_B"] = os.environ.get("FLARE_PARITY_B", "1")
    os.environ["FLARE_PARITY_H"] = os.environ.get("FLARE_PARITY_H", "8")
    os.environ["FLARE_PARITY_M"] = os.environ.get("FLARE_PARITY_M", "128")
    os.environ["FLARE_PARITY_N"] = os.environ.get("FLARE_PARITY_N", "512")
    os.environ["FLARE_PARITY_D"] = os.environ.get("FLARE_PARITY_D", "32")
    os.environ["FLARE_PARITY_DTYPE"] = os.environ.get("FLARE_PARITY_DTYPE", "bfloat16")
    os.environ["FLARE_PARITY_TRAIN_STEPS"] = os.environ.get("FLARE_PARITY_TRAIN_STEPS", "0")

    os.environ["FLARE_TRAINLIKE_STRICT"] = "1"
    os.environ["FLARE_TRAINLIKE_COMPARE"] = os.environ.get("FLARE_TRAINLIKE_COMPARE", "1")
    os.environ["FLARE_TRAINLIKE_STEPS"] = os.environ.get("FLARE_TRAINLIKE_STEPS", "2")
    os.environ["FLARE_TRAINLIKE_CONFIGS"] = os.environ.get("FLARE_TRAINLIKE_CONFIGS", "1,8,128,1024,32")
    os.environ["FLARE_TRAINLIKE_DTYPE"] = os.environ.get("FLARE_TRAINLIKE_DTYPE", "bfloat16")
    os.environ["FLARE_SHARP_BWD_DECODE_SEPARATION_MODES"] = os.environ.get(
        "FLARE_SHARP_BWD_DECODE_SEPARATION_MODES", "00,10,01,11"
    )


_REGRESSION_COMPONENTS = {
    "parity": _parity_tests,
    "trainlike_sanity": _trainlike_sanity,
    "trainlike_multistep_parity": _trainlike_multistep_parity,
}
_REGRESSION_COMPONENTS.update(
    {
        f"autotune_launch_coverage_{shard + 1}": (
            lambda shard=shard: _autotune_launch_coverage_suite_shard(shard, _AUTOTUNE_COVERAGE_NUM_SHARDS)
        )
        for shard in range(_AUTOTUNE_COVERAGE_NUM_SHARDS)
    }
)
_REGRESSION_COMPONENTS.update(
    {
        f"correctness_{shard + 1}": (lambda shard=shard: _run_correctness_suite_shard(shard, _CORRECTNESS_NUM_SHARDS))
        for shard in range(_CORRECTNESS_NUM_SHARDS)
    }
)
_REGRESSION_COMPONENTS.update(
    {
        f"grad_checks_{shard + 1}": (lambda shard=shard: _run_grad_checks_suite_shard(shard, _GRAD_CHECKS_NUM_SHARDS))
        for shard in range(_GRAD_CHECKS_NUM_SHARDS)
    }
)
_REGRESSION_COMPONENTS.update(
    {
        f"long_context_accuracy_{shard + 1}": (
            lambda shard=shard: _long_context_accuracy_suite_shard(shard, _LONGCTX_NUM_SHARDS)
        )
        for shard in range(_LONGCTX_NUM_SHARDS)
    }
)
_REGRESSION_COMPONENTS.update(
    {
        f"chunk_size_sensitivity_{shard + 1}": (
            lambda shard=shard: _chunk_size_sensitivity_suite_shard(shard, _CHUNK_SENS_NUM_SHARDS)
        )
        for shard in range(_CHUNK_SENS_NUM_SHARDS)
    }
)
_REGRESSION_COMPONENTS.update(
    {
        f"sharp_softmax_bwd_regression_{shard + 1}": (
            lambda shard=shard: _sharp_softmax_bwd_regression_suite_shard(shard, _SHARP_BWD_NUM_SHARDS)
        )
        for shard in range(_SHARP_BWD_NUM_SHARDS)
    }
)


def _default_regression_components(*, extended: bool) -> list[str]:
    components = [
        *[f"correctness_{shard + 1}" for shard in range(_CORRECTNESS_NUM_SHARDS)],
        *[f"grad_checks_{shard + 1}" for shard in range(_GRAD_CHECKS_NUM_SHARDS)],
        *[f"autotune_launch_coverage_{shard + 1}" for shard in range(_AUTOTUNE_COVERAGE_NUM_SHARDS)],
        "parity",
        "trainlike_sanity",
    ]
    if extended:
        components.extend(
            [
                *[f"long_context_accuracy_{shard + 1}" for shard in range(_LONGCTX_NUM_SHARDS)],
                "trainlike_multistep_parity",
                *[f"chunk_size_sensitivity_{shard + 1}" for shard in range(_CHUNK_SENS_NUM_SHARDS)],
                *[f"sharp_softmax_bwd_regression_{shard + 1}" for shard in range(_SHARP_BWD_NUM_SHARDS)],
            ]
        )
    return components


def _run_regression_components(components: list[str]) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("FLARE regression test requires CUDA.")

    _apply_regression_defaults()
    for component in components:
        runner = _REGRESSION_COMPONENTS.get(component)
        if runner is None:
            known = ", ".join(sorted(_REGRESSION_COMPONENTS))
            raise ValueError(f"Unknown regression component '{component}'. Known: {known}")

        print("=" * 100)
        print(f"[FLARE REGRESSION] {component}")
        print("=" * 100)
        runner()


def _regression_test() -> None:
    extended = _strict_mode_enabled("FLARE_REGRESSION_EXTENDED", default=False)
    _run_regression_components(_default_regression_components(extended=extended))
    print("[FLARE REGRESSION] all checks passed.")

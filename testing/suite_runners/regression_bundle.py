"""Extracted regression/stress suite implementation."""

from testing.suite_runners.common import *

from testing.suite_runners.chunk_size_sensitivity import _chunk_size_sensitivity_suite
from testing.suite_runners.correctness import _run_correctness_suite
from testing.suite_runners.grad_checks import _run_grad_checks_suite
from testing.suite_runners.long_context_accuracy import _long_context_accuracy_suite
from testing.suite_runners.parity import _parity_tests
from testing.suite_runners.sharp_softmax_bwd_regression import _sharp_softmax_bwd_regression_suite
from testing.suite_runners.trainlike_multistep_parity import _trainlike_multistep_parity
from testing.suite_runners.trainlike_sanity import _trainlike_sanity


def _regression_test():
    if not torch.cuda.is_available():
        raise RuntimeError("FLARE regression test requires CUDA.")

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

    print("=" * 100)
    print("[FLARE REGRESSION] correctness suite")
    print("=" * 100)
    _run_correctness_suite()

    print("=" * 100)
    print("[FLARE REGRESSION] gradient checks suite")
    print("=" * 100)
    _run_grad_checks_suite()

    print("=" * 100)
    print("[FLARE REGRESSION] parity suite")
    print("=" * 100)
    _parity_tests()

    print("=" * 100)
    print("[FLARE REGRESSION] trainlike suite")
    print("=" * 100)
    _trainlike_sanity()

    if _strict_mode_enabled("FLARE_REGRESSION_EXTENDED", default=False):
        print("=" * 100)
        print("[FLARE REGRESSION] long-context suite")
        print("=" * 100)
        _long_context_accuracy_suite()

        print("=" * 100)
        print("[FLARE REGRESSION] trainlike multi-step parity")
        print("=" * 100)
        _trainlike_multistep_parity()

        print("=" * 100)
        print("[FLARE REGRESSION] chunk-size sensitivity")
        print("=" * 100)
        _chunk_size_sensitivity_suite()

        print("=" * 100)
        print("[FLARE REGRESSION] sharp backward regression")
        print("=" * 100)
        _sharp_softmax_bwd_regression_suite()

    print("[FLARE REGRESSION] all checks passed.")

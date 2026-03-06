import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _set_regression_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Keep defaults bounded so opt-in pytest runs stay practical.
    monkeypatch.setenv("FLARE_CORRECTNESS_STRICT", "1")
    monkeypatch.setenv("FLARE_CORRECTNESS_DTYPES", "bfloat16")
    monkeypatch.setenv("FLARE_CORRECTNESS_SHAPES", "1,2,256,64,32")
    monkeypatch.setenv("FLARE_CORRECTNESS_QK_STDS", "1.0")
    monkeypatch.setenv("FLARE_CORRECTNESS_DECODE_SEPARATION_MODES", "00,11")
    monkeypatch.setenv("FLARE_CORRECTNESS_GRAD", "1")
    monkeypatch.setenv("FLARE_CORRECTNESS_GRAD_LIMIT", "1")
    monkeypatch.setenv("FLARE_CORRECTNESS_SUITE_GRAD", "0")
    monkeypatch.setenv("FLARE_PARITY_STRICT", "1")
    monkeypatch.setenv("FLARE_PARITY_B", "1")
    monkeypatch.setenv("FLARE_PARITY_H", "4")
    monkeypatch.setenv("FLARE_PARITY_M", "64")
    monkeypatch.setenv("FLARE_PARITY_N", "256")
    monkeypatch.setenv("FLARE_PARITY_D", "32")
    monkeypatch.setenv("FLARE_PARITY_DTYPE", "bfloat16")
    monkeypatch.setenv("FLARE_PARITY_TRAIN_STEPS", "0")
    monkeypatch.setenv("FLARE_TRAINLIKE_STRICT", "1")
    monkeypatch.setenv("FLARE_TRAINLIKE_COMPARE", "1")
    monkeypatch.setenv("FLARE_TRAINLIKE_STEPS", "1")
    monkeypatch.setenv("FLARE_TRAINLIKE_CONFIGS", "1,4,64,256,32")
    monkeypatch.setenv("FLARE_TRAINLIKE_DTYPE", "bfloat16")


@pytest.mark.regression
def test_correctness_suite(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> None:
    from testing.suites.correctness import _run_correctness_suite

    if not pytestconfig.getoption("--full-matrix"):
        _set_regression_defaults(monkeypatch)
    _run_correctness_suite()


@pytest.mark.regression
def test_grad_checks_suite(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> None:
    from testing.suites.grad_checks import _run_grad_checks_suite

    # Always run grad checks in this dedicated suite.
    monkeypatch.setenv("FLARE_CORRECTNESS_GRAD", "1")
    if not pytestconfig.getoption("--full-matrix"):
        _set_regression_defaults(monkeypatch)
        monkeypatch.setenv("FLARE_CORRECTNESS_SUITE_GRAD", "0")
    _run_grad_checks_suite()


@pytest.mark.regression
def test_regression_bundle(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> None:
    from testing.suites.regression_bundle import _regression_test

    if not pytestconfig.getoption("--full-matrix"):
        _set_regression_defaults(monkeypatch)
        monkeypatch.setenv("FLARE_REGRESSION_EXTENDED", "0")
    else:
        monkeypatch.setenv("FLARE_REGRESSION_EXTENDED", "1")
    _regression_test()


@pytest.mark.regression
@pytest.mark.stress
def test_sharp_softmax_bwd_regression_suite(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> None:
    from testing.suites.sharp_softmax_bwd_regression import _sharp_softmax_bwd_regression_suite

    if not pytestconfig.getoption("--full-matrix"):
        monkeypatch.setenv("FLARE_SHARP_BWD_DTYPE", "bfloat16")
        monkeypatch.setenv("FLARE_SHARP_BWD_STRICT", "1")
        monkeypatch.setenv("FLARE_SHARP_BWD_PRECISIONS", "ieee")
        monkeypatch.setenv("FLARE_SHARP_BWD_CONFIGS", "1,4,16,256,32")
        monkeypatch.setenv("FLARE_SHARP_BWD_STRESS_CONFIGS", "1,4,16,512,32")
        monkeypatch.setenv("FLARE_SHARP_BWD_QK_STDS", "1.0,2.0")
        monkeypatch.setenv("FLARE_SHARP_BWD_STRESS_QK_STDS", "4.0")
        monkeypatch.setenv("FLARE_SHARP_BWD_DECODE_SEPARATION_MODES", "00,11")
    _sharp_softmax_bwd_regression_suite()


@pytest.mark.regression
@pytest.mark.stress
def test_long_context_accuracy_suite(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> None:
    from testing.suites.long_context_accuracy import _long_context_accuracy_suite

    if not pytestconfig.getoption("--full-matrix"):
        monkeypatch.setenv("FLARE_LONGCTX_DTYPE", "bfloat16")
        monkeypatch.setenv("FLARE_LONGCTX_STRICT", "1")
        monkeypatch.setenv("FLARE_LONGCTX_PRECISIONS", "ieee")
        monkeypatch.setenv("FLARE_LONGCTX_CONFIGS", "1,4,16,512,32")
    _long_context_accuracy_suite()


@pytest.mark.regression
@pytest.mark.stress
def test_chunk_size_sensitivity_suite(monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config) -> None:
    from testing.suites.chunk_size_sensitivity import _chunk_size_sensitivity_suite

    if not pytestconfig.getoption("--full-matrix"):
        monkeypatch.setenv("FLARE_CHUNK_SENS_DTYPE", "bfloat16")
        monkeypatch.setenv("FLARE_CHUNK_SENS_STRICT", "1")
        monkeypatch.setenv("FLARE_CHUNK_SENS_PRECISIONS", "ieee")
        monkeypatch.setenv("FLARE_CHUNK_SENS_CONFIGS", "1,4,16,512,32")
        monkeypatch.setenv("FLARE_CHUNK_SENS_BWD_CHUNKS", "16,32")
    _chunk_size_sensitivity_suite()

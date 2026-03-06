# Testing Overview

This directory has two testing layers:

- Pytest-collected test modules in `testing/test_*.py` (unit/parity/reference/finite-difference plus regression wrappers).
- Extracted suite implementations in `testing/suite_runners/*.py` used by regression/stress wrappers.

## Pytest Entry Point and Collection

Primary entry point:

```bash
pytest testing -q
```

- Pytest options and markers are configured in `pyproject.toml` (`[tool.pytest.ini_options]`) and `testing/conftest.py`.
- `testing/suite_runners/*.py` are helper/suite modules; pytest does not collect them directly because they do not define `test_*` functions.
- Regression/stress tests are still collected from `testing/test_*.py`, but skipped unless opt-in flags are supplied.

## Run Commands

Run default tests:

```bash
pytest testing -q
```

Run regression block only:

```bash
pytest testing --run-regression -q
```

Run regression + stress blocks:

```bash
pytest testing --run-regression --run-stress -q
```

Run full long-running regression/stress matrices:

```bash
pytest testing --run-regression --run-stress --full-matrix -q
```

## Extracted Regression/Stress Suites

The suite implementations below were extracted into `testing/suite_runners/` (one module per suite):

- `testing/suite_runners/correctness.py` -> `_run_correctness_suite`
- `testing/suite_runners/parity.py` -> `_parity_tests`
- `testing/suite_runners/trainlike_sanity.py` -> `_trainlike_sanity`
- `testing/suite_runners/trainlike_projected.py` -> `_trainlike_projected`
- `testing/suite_runners/long_context_accuracy.py` -> `_long_context_accuracy_suite`
- `testing/suite_runners/trainlike_multistep_parity.py` -> `_trainlike_multistep_parity`
- `testing/suite_runners/chunk_size_sensitivity.py` -> `_chunk_size_sensitivity_suite`
- `testing/suite_runners/sharp_softmax_bwd_regression.py` -> `_sharp_softmax_bwd_regression_suite`
- `testing/suite_runners/grad_checks.py` -> `_run_grad_checks_suite`
- `testing/suite_runners/regression_bundle.py` -> `_regression_test`

Pytest wrappers that currently call these suites:

- `testing/test_regression_suites.py`:
  - direct wrappers: `_run_correctness_suite`, `_run_grad_checks_suite`, `_regression_test`
  - stress wrappers: `_sharp_softmax_bwd_regression_suite`, `_long_context_accuracy_suite`, `_chunk_size_sensitivity_suite`
  - note: `_parity_tests`, `_trainlike_sanity`, and `_trainlike_multistep_parity` are run via `_regression_test` (bundle), not direct top-level wrappers.
- `testing/test_cached_suites.py`:
  - additional regression-marked pytest module (not in `testing/suite_runners/`)

Suite implementation currently not wired into collected pytest wrappers:

- `testing/suite_runners/trainlike_projected.py` -> `_trainlike_projected`

Legacy env-flag entrypoints in `testing/test.py` (`FLARE_DEBUG_*`, `FLARE_RECURRENT_TEST`, `FLARE_CACHED_TEST`) were removed.

## What These Suites Cover

- `_run_correctness_suite`
  - Broad forward/backward correctness over decode-sharing modes, dtype/shape grids, and precision modes.
  - Compares Triton chunked path vs reference and PyTorch chunked noise model.
- `_parity_tests`
  - End-to-end parity checks between Triton chunked and reference baselines (forward + grads).
- `_trainlike_sanity`
  - Training-like optimization-loop sanity checks, gradient finiteness, and bounded drift behavior.
- `_trainlike_projected`
  - Projected trainlike update behavior with explicit projection parameterization.
  - Not currently invoked by `testing/test_regression_suites.py` or `testing/test_cached_suites.py`.
- `_long_context_accuracy_suite`
  - Accuracy/gradient behavior at long sequence lengths and precision modes.
- `_trainlike_multistep_parity`
  - Multi-step parity under repeated updates (loss/output/gradient agreement).
- `_chunk_size_sensitivity_suite`
  - Sensitivity to backward chunk-size selection and gradient consistency across chunk settings.
- `_sharp_softmax_bwd_regression_suite`
  - Sharp-softmax backward stress coverage for non-finite detection and gradient drift bounds.
- `_run_grad_checks_suite`
  - Dedicated gradient-only validation across decode-separation modes and input-scale sweeps.
  - Includes the historical `_gradcheck_suite` matrix when `FLARE_CORRECTNESS_SUITE_GRAD=1`.
- `_regression_test`
  - Bundle runner that executes core suites with bounded deterministic defaults.

## Test Matrices (Per Suite)

Below are the key matrix dimensions each suite exercises. Most are configurable via env vars used inside `testing/suite_runners/*`.

- `_run_correctness_suite`
  - Dtypes: `FLARE_CORRECTNESS_DTYPES` (default typically `bfloat16,float16`).
  - Shapes: `FLARE_CORRECTNESS_SHAPES` as `(B,H,N,M,D)` tuples.
  - Input score scales (`qk_std`): `FLARE_CORRECTNESS_QK_STDS` (commonly includes `0.5,1,2,4`).
  - Decode separation matrix: `FLARE_CORRECTNESS_DECODE_SEPARATION_MODES`:
    - `00` => `separate_q_dec=False`, `separate_k_dec=False`
    - `10` => `True, False`
    - `01` => `False, True`
    - `11` => `True, True`
  - Precision mode sweep: derived from chunk config + `FLARE_INPUT_PRECISION` behavior.
  - Optional gradient checks: enabled by `FLARE_CORRECTNESS_GRAD` and bounded by `FLARE_CORRECTNESS_GRAD_LIMIT`.

- `_parity_tests`
  - Single configurable shape via `FLARE_PARITY_B/H/M/N/D`.
  - Configurable dtype: `FLARE_PARITY_DTYPE`.
  - Optional short train-step loop: `FLARE_PARITY_TRAIN_STEPS`.
  - Tightness thresholds via `FLARE_PARITY_*` error/cosine bounds.

- `_trainlike_sanity`
  - Config list: `FLARE_TRAINLIKE_CONFIGS` (semicolon-separated `(B,H,M,N,D)` tuples).
  - Dtype: `FLARE_TRAINLIKE_DTYPE`.
  - Steps/lr/qkv scale: `FLARE_TRAINLIKE_STEPS`, `FLARE_TRAINLIKE_LR`, `FLARE_TRAINLIKE_QKV_STD`.
  - Optional compare/probe toggles: `FLARE_TRAINLIKE_COMPARE`, `FLARE_TRAINLIKE_HEAD_PROBE`.

- `_trainlike_projected`
  - Config list: `FLARE_TRAINLIKE_CONFIGS`.
  - Dtype/steps/lr/std: `FLARE_TRAINLIKE_DTYPE`, `FLARE_TRAINLIKE_STEPS`, `FLARE_TRAINLIKE_LR`, `FLARE_TRAINLIKE_QKV_STD`.
  - Implementation toggle: `FLARE_TRAINLIKE_IMPL` (e.g., triton path).

- `_long_context_accuracy_suite`
  - Precision sweep: `FLARE_LONGCTX_PRECISIONS` (e.g., `ieee,tf32,tf32x3`).
  - Long-context config grid: `FLARE_LONGCTX_CONFIGS` over `(B,H,M,N,D)` with large `N`.
  - Dtype/seed/strictness: `FLARE_LONGCTX_DTYPE`, `FLARE_LONGCTX_SEED`, `FLARE_LONGCTX_STRICT`.

- `_trainlike_multistep_parity`
  - Precision sweep: `FLARE_TRAINLIKE_PARITY_PRECISIONS`.
  - Config grid: `FLARE_TRAINLIKE_PARITY_CONFIGS`.
  - Steps/lr/std: `FLARE_TRAINLIKE_PARITY_STEPS`, `FLARE_TRAINLIKE_PARITY_LR`, `FLARE_TRAINLIKE_PARITY_QKV_STD`.
  - Optional gradient checks: `FLARE_TRAINLIKE_PARITY_CHECK_GRADS`.

- `_chunk_size_sensitivity_suite`
  - Precision sweep: `FLARE_CHUNK_SENS_PRECISIONS`.
  - Config grid: `FLARE_CHUNK_SENS_CONFIGS`.
  - Backward chunk-size sweep: `FLARE_CHUNK_SENS_BWD_CHUNKS` (e.g., `16,32,64`).
  - Dtype/strictness/runtime strictness: `FLARE_CHUNK_SENS_DTYPE`, `FLARE_CHUNK_SENS_STRICT`, `FLARE_CHUNK_SENS_STRICT_RUNTIME`.

- `_sharp_softmax_bwd_regression_suite`
  - Precision sweep: `FLARE_SHARP_BWD_PRECISIONS`.
  - Main config grid: `FLARE_SHARP_BWD_CONFIGS`.
  - Stress config grid: `FLARE_SHARP_BWD_STRESS_CONFIGS`.
  - Score-scale sweeps:
    - standard: `FLARE_SHARP_BWD_QK_STDS` (often `1,2,4,8`)
    - stress: `FLARE_SHARP_BWD_STRESS_QK_STDS` (often high-scale tails)
  - Decode separation matrix: `FLARE_SHARP_BWD_DECODE_SEPARATION_MODES` (`00/10/01/11`).

- `_run_grad_checks_suite`
  - Dtypes/shapes/std/modes reuse the correctness env grid:
    - `FLARE_CORRECTNESS_DTYPES`
    - `FLARE_CORRECTNESS_SHAPES`
    - `FLARE_CORRECTNESS_QK_STDS`
    - `FLARE_CORRECTNESS_DECODE_SEPARATION_MODES`
  - Main gradient gate toggles/bounds:
    - `FLARE_CORRECTNESS_GRAD=1`
    - `FLARE_CORRECTNESS_GRAD_LIMIT`
    - `FLARE_CORRECTNESS_GRAD_MEAN_ABS_MAX`
    - `FLARE_CORRECTNESS_GRAD_MAX_ABS_MAX`
    - `FLARE_CORRECTNESS_GRAD_COS_MIN`
  - Optional historical gradcheck matrix:
    - `FLARE_CORRECTNESS_SUITE_GRAD=1`
    - `FLARE_CORRECTNESS_SUITE_GRAD_SCALES`

- `_regression_test` (bundle)
  - Sets bounded defaults for correctness/parity/trainlike and runs them as a deterministic gate.
  - Optional extended matrix via `FLARE_REGRESSION_EXTENDED=1` to include longctx/trainlike-multistep/chunk-sensitivity/sharp-bwd blocks.

## Wrapper Defaults (Short vs Full)

`testing/test_regression_suites.py` is a wrapper layer:

- Without `--full-matrix`:
  - Applies reduced defaults for faster iteration.
  - Key bounded values:
    - `FLARE_CORRECTNESS_SHAPES=1,2,256,64,32`
    - `FLARE_CORRECTNESS_QK_STDS=1.0`
    - `FLARE_CORRECTNESS_DECODE_SEPARATION_MODES=00,11`
    - `FLARE_CORRECTNESS_GRAD_LIMIT=1`
    - `FLARE_PARITY_{B,H,M,N,D}=1,4,64,256,32`
    - `FLARE_TRAINLIKE_CONFIGS=1,4,64,256,32`
    - Stress wrappers use reduced configs (for example longctx `N=512`).
  - Regression bundle forces `FLARE_REGRESSION_EXTENDED=0`.

- With `--full-matrix`:
  - Wrapper does not apply bounded matrix overrides.
  - Regression bundle sets `FLARE_REGRESSION_EXTENDED=1`.
  - Suite-native defaults execute, including large-N long-context and sharp-bwd sweeps.

## Notes

- Markers/options are defined in `testing/conftest.py`:
  - `--run-regression`
  - `--run-stress`
  - `--full-matrix` (disables reduced bounded defaults in wrappers)
- The wrappers that expose these suites as pytest tests are in `testing/test_regression_suites.py`.

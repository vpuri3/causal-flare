# Repository Guidelines

## Session Startup
- At the start of every session, read `TODO.md` and align ongoing work with current priorities.

## Project Structure & Module Organization
- `causal_flare/`: core FLARE implementations (`chunked`, `dense`, `recurrent`, `torch`, inference wiring).
- `testing/`: pytest suites, reference validation, finite-difference checks, and regression harnesses.
- `benchmark/`: prefill/decode and train-step benchmark scripts plus implementation adapters.
- `scripts/`: local setup and job submission helpers.

## Build, Test, and Development Commands
- `pip install -e .`: editable install for local development.
- `pip install -e .[dev,benchmark]`: install benchmark/dev extras.
- `pytest`: run the full local test suite.
- `pytest testing/test_flare.py`: run a focused FLARE correctness suite.
- `python benchmark/benchmark_prefill_decode.py --help`: prefill/decode benchmark options.
- `python benchmark/benchmark_train_step.py --help`: train-step benchmark options.

## Coding Style & Naming Conventions
- Python 3.10+ with 4-space indentation.
- Line length target: 127 (see `pyproject.toml`).
- Linting: `ruff` with import-order checks.
- Tests use `test_*.py` filenames in `testing/` and `test_*` function naming.
- For Triton-backed implementations, do not add torch fallbacks in production paths; temporary debug fallbacks are allowed only while diagnosing issues and should be removed afterward.

## Testing Guidelines
- Framework: `pytest` (configured in `pyproject.toml`).
- Keep tests close to feature area under `testing/`.
- Prefer focused kernel/ops tests plus targeted integration tests.
- Do not loosen tolerances or reduce/skip coverage to force passing tests; fix root causes.

## Local Testing & Logging Workflow
- Activate the repo venv (`source .venv/bin/activate` or `/vact`) before tests/benchmarks.
- For long runs, pipe output to a log (`tee /tmp/...`) for later inspection.
- After runs, scan logs for NaN/Inf, unstable gradients, or major perf regressions and summarize findings.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects.
- In PRs, summarize behavior changes, list tests run (or why not), and include perf notes for kernel changes.

## Security & Configuration Tips
- CUDA + Triton are runtime dependencies; keep versions aligned with `pyproject.toml`.
- Keep experiment artifacts and large outputs out of source-controlled core paths.

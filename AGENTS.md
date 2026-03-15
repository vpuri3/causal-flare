# Repository Guidelines

## Session Startup
- At the start of every session, read `TODO.md` and align ongoing work with current priorities.
- Before starting opportunistic refactors or side quests, check whether the request aligns with current `TODO.md` priorities.

## Workflow Expectations
- This is an experimental repo. Do not write extra code purely to preserve backward compatibility unless the user explicitly asks for it.
- Do not treat package-internal helper/state conventions as API boundaries by default. The API boundary that matters unless the user says otherwise is between this package's exported functions and the rest of world (ROW).
- When the user asks for refactors that move code, prefer moving entire function definitions intact rather than splitting one function across files unless the user explicitly asks for that.
- For complex tasks, state key assumptions early when behavior, masking semantics, shapes, or numerical expectations are ambiguous.
- For complex user queries, consider writing a short task list up front so progress and completion status are clear.
- Prefer the smallest safe change that solves the requested problem; avoid incidental refactors unless they are required for correctness, maintainability, or testability.
- If work is blocked, report exactly what is blocked, what was tried, and what remains unverified.
- In final summaries for substantial tasks, clearly separate what was completed, what was not completed, and what was not verified.
- For changes to Triton kernels or other performance-critical paths, call out any shape constraints, workspace/saved-buffer changes, backward-pass implications, and whether performance was measured.

## Interactive Execution Behavior
- If the user interrupts a running thinking/log stream with `ESC`, do not treat that alone as a reason to terminate any background terminal task.
- Treat the interruption as new user input to consider while allowing the existing background task or thinking stream to continue when feasible.
- It is acceptable to stop the background task if there is a concrete reason to do so, but do not close it by default merely because the user said something mid-stream.
- Prefer handling both in parallel when possible: continue monitoring the ongoing task while responding to the new user input.

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
- Run the most targeted relevant tests first for the touched area before broader suite runs.
- For numerics-sensitive changes, verify both correctness and tolerance behavior against the strongest available reference instead of masking regressions with threshold changes.
- For performance-sensitive changes, include before/after benchmark numbers when feasible; if benchmarks were not run, say so explicitly.

## Local Testing & Logging Workflow
- Activate the repo venv (`source .venv/bin/activate` or `/vact`) before tests/benchmarks.
- For long runs, pipe output to a log (`tee /tmp/...`) for later inspection.
- After runs, scan logs for NaN/Inf, unstable gradients, or major perf regressions and summarize findings.

## Commit & Pull Request Guidelines
- Use detailed, imperative commit messages.
- In PRs, summarize behavior changes, list tests run (or why not), and include perf notes for kernel changes.

## Security & Configuration Tips
- CUDA + Triton are runtime dependencies; keep versions aligned with `pyproject.toml`.
- Keep experiment artifacts and large outputs out of source-controlled core paths.

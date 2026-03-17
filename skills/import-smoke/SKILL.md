---
name: import-smoke
description: Use when the task is to smoke test that specific causal_flare imports resolve cleanly without running the pytest suite. This skill provides a reusable script for checking direct module and symbol imports and reporting where each symbol came from.
---

# Import Smoke

Use this skill for quick import validation after refactors that move functions between modules.

## Workflow

1. Run the bundled script from the `causal-flare` repo root.
2. Prefer explicit `module:symbol1,symbol2` checks for the symbols touched by the change.
3. Do not run pytest when the user only asked for an import smoke test.

Default command:

```bash
python skills/import-smoke/scripts/check_imports.py
```

Targeted command for the recurrent/dense move:

```bash
python skills/import-smoke/scripts/check_imports.py \
  --import causal_flare.autoregressive.recurrent:flare_recurrent_pytorch \
  --import causal_flare.autoregressive.dense:flare_recurrent_dense_backward_pytorch
```

## Output Expectations

- Exit `0` if every requested import resolves.
- Print one line per checked symbol with the module path and resolved symbol name.
- Exit nonzero on the first failed import with a concise error message.

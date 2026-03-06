# causal-flare
[![Tests](https://github.com/vedantpu/causal-flare/actions/workflows/tests.yml/badge.svg)](https://github.com/vedantpu/causal-flare/actions/workflows/tests.yml)

Standalone experimental packaging of `fla/models/flare/causal_flare` from the flash-linear-attention project.

## Related reading

- Blog post: [From Encoder to Decoder: Extending FLARE to Memory-Efficient Causal Attention](https://vpuri3.github.io/blog/from-encoder-to-decoder-extending-flare-to-memory-efficient-causal-attention/)

## Status

- Version: `0.0.1`
- Stability: experimental / API may change without notice

## Install

```bash
pip install causal-flare==0.0.1
```

For local development:

```bash
pip install -e .[dev,test,benchmark]
```

## Quick start

```python
import torch
from causal_flare import flare_chunk_triton

B, N, H, M, D = 1, 256, 8, 64, 32
q = torch.randn(H, M, D, device="cuda", dtype=torch.bfloat16)
k = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
v = torch.randn(B, N, H, D, device="cuda", dtype=torch.bfloat16)
y = flare_chunk_triton(q, k, v)
```

## Benchmarks

```bash
python benchmark/benchmark_prefill_decode.py --help
python benchmark/benchmark_train_step.py --help
```

## Testing

Fast/default tests:

```bash
pytest testing -q
```

Regression pytest block (opt-in, reduced matrix for faster iteration):

```bash
pytest testing --run-regression -q
```

Stress pytest block (opt-in, long-running):

```bash
pytest testing --run-regression --run-stress -q
```

Full matrix (historical long-running coverage):

```bash
pytest testing --run-regression --run-stress --full-matrix -q
```

CI note: `.github/workflows/tests.yml` runs on a self-hosted GPU runner (`gpu`, `cuda` labels), fails fast if CUDA is unavailable, and uses `--full-matrix`.

Short vs full:

- Short (`--run-regression`, optional `--run-stress`): wrapper applies bounded defaults (smaller shapes/configs) for quick gating.
- Full (`--full-matrix`): wrapper does not apply bounded overrides and enables extended regression coverage; this is the multi-minute heavy run.

Extracted regression/stress suite implementations live in `testing/suite_runners/` (one file per suite, including `grad_checks.py`), plus additional regression-only pytest suites in `testing/test_cached_suites.py`.
See `testing/README.md` for the full suite list and coverage summary.

## Notes

- This project currently keeps optional integration hooks for running inside the original FLA monorepo.
- CUDA + Triton are runtime dependencies.
- `flash-attn` is a dev extra (`.[dev]`) and is intended for benchmark-only workflows.

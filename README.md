# causal-flare

Standalone experimental packaging of `fla/models/flare/causal_flare` from the flash-linear-attention project.

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

## Notes

- This project currently keeps optional integration hooks for running inside the original FLA monorepo.
- CUDA + Triton are runtime dependencies.
- `flash-attn` is a dev extra (`.[dev]`) and is intended for benchmark-only workflows.

# Semi-AR Matrix Tuning Workflow

This is the semi-autoregressive counterpart to the chunked/recurrent matrix workflows.

The default runtime path should stay heuristic-only. Compile-heavy exploration belongs here, not in `causal_flare/semi_autoregressive/training.py`.

Current runner:

- [`tune_semi_ar_matrix.py`](./tune_semi_ar_matrix.py)

Current runtime consumer:

- [`causal_flare/semi_autoregressive/training.py`](../causal_flare/semi_autoregressive/training.py)

## Scope

This workflow currently targets the hot forward kernels:

- `semi_ar_block_prepare`
- `semi_ar_lse_output`
- `BLOCK_T` for the merged output path

It intentionally does not try to cover every `(M, D_score, D_value, block_size)` combination. Use sampled anchor cases first, then promote narrow bucket heuristics.

Assumptions in the current runner:

- `BH` is held fixed at a representative CUDA-friendly value.
- `N` is held large (`49152` by default) because the relevant kernels are not expected to be strongly `N`-sensitive once the sequence is long enough, and that anchor stays aligned with more of the sampled block sizes than `65536`.
- `chunk_size` is chosen as the largest supported divisor of `block_size` in `{16, 32, 64, 128}`.
- Large-block-size special handling is intentionally out of scope for now.

## Recommended Workflow

1. Run the sampled anchor sweep:

```bash
source .venv/bin/activate
python benchmark/tune_semi_ar_matrix.py --run-name semi-ar-anchor
```

2. Inspect:

- `results/semi_ar_matrix/<run-name>/runs.jsonl`
- `results/semi_ar_matrix/<run-name>/summary.json`
- `results/semi_ar_matrix/<run-name>/summary.md`

3. Promote only stable patterns back into `training.py`:

- launch defaults bucketed by `M`
- launch defaults bucketed by `max(D_score, D_value)` plus a mixed-`D` flag when needed
- launch defaults bucketed by `block_size`
- keep env overrides intact for explicit experiments

4. Rerun a smaller confirmation sweep after changing runtime heuristics.

## What To Look For

- `BLOCK_T` stability:
  - if `BLOCK_T=64` wins almost everywhere once `chunk_size >= 64`, keep the simple rule
  - if small-block cases favor `16` or `32`, preserve that via chunk-size/block-size buckets

- `semi_ar_block_prepare` winners:
  - watch `block_size` and `max(D_score, D_value)` first
  - `M` may matter, but the prepare kernel usually looks more width/block driven than `N` driven

- `semi_ar_lse_output` winners:
  - watch `M`, `D_value`, and mixed `D_score != D_value` cases
  - if `4 warps / 2 stages` keeps winning, keep the runtime rule narrow rather than inventing extra families

## Current Goal

The goal is not an exhaustive matrix. The goal is to produce a small, evidence-based runtime bucket policy that avoids paying runtime autotune costs on the user path.

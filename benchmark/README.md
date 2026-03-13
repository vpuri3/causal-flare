# Benchmark Tuning Guide

This directory contains the benchmark and profiling workflows for the current FLARE launch-tuning policy.

Use this file as the high-level source of truth for which knobs are owned by offline matrix tuning versus fixed runtime heuristics.

## Why Runtime Uses Heuristics

We briefly tried moving chunked and inference runtime config selection over to Triton autotune on the default user path.

That experiment did not hold up operationally:

- cold-cache compile and first-use latency became extremely large
- the expensive cost was paid during normal user execution instead of during an explicit offline tuning workflow
- the resulting user experience was materially worse even when steady-state performance was acceptable

The current policy therefore follows the `72394e2d2a109be2d8464147c17f10b28f423fcc` workflow again:

- benchmark scripts own the search
- runtime code uses promoted heuristic bucket rules
- Triton autotune is not part of the default user path

## Current Policy

### Autoregressive

Runtime entry point:

- [`causal_flare/autoregressive/training.py`](../causal_flare/autoregressive/training.py)

Benchmark tools:

- [`benchmark_prefill_decode.py`](./benchmark_prefill_decode.py)
- [`profile_chunked_flare.py`](./profile_chunked_flare.py)
- [`measure_cold_start.py`](./measure_cold_start.py)
- [`tune_chunked_flare_matrix.py`](./tune_chunked_flare_matrix.py)
- [`chunked_flare_matrix_workflow.md`](./chunked_flare_matrix_workflow.md)

The autoregressive chunked path is tuned offline, then promoted back into runtime heuristics.

- The matrix runner owns the knob search.
- Runtime consumes the promoted bucket policy plus optional env overrides.
- Default user execution does not run Triton autotune searches or multi-candidate runtime compile loops.

Offline-tuned chunked knobs:

- `CHUNK_SIZE`
- `FLARE_BLOCK_M`
- `FLARE_PREFIX_BLOCK_M`
- `FLARE_PREPARE_BLOCK_D`
- `FLARE_PREPARE_BLOCK_K`
- `FLARE_PREFIX_BLOCK_D`
- `FLARE_FWD_BLOCK_D`
- `FLARE_FWD_BLOCK_K`
- `FLARE_BLOCK_T`
- `FLARE_PREPARE_NUM_WARPS` / `FLARE_PREPARE_NUM_STAGES`
- `FLARE_PREFIX_NUM_WARPS` / `FLARE_PREFIX_NUM_STAGES`
- `FLARE_DECODER_NUM_WARPS` / `FLARE_DECODER_NUM_STAGES`
- `FLARE_FWD_NUM_WARPS` / `FLARE_FWD_NUM_STAGES`
- `FLARE_LSE_BWD_BLOCK_M`
- `FLARE_LSE_BWD_SCORE_BLOCK_M`
- `FLARE_LSE_BWD_BLOCK_DV`
- `FLARE_LSE_BWD_BLOCK_K`
- `FLARE_LSE_BWD_BLOCK_D_PART`
- `FLARE_LSE_BWD_QK_BLOCK_D`
- `FLARE_LSE_BWD_BLOCK_T_QK`
- `FLARE_LSE_BWD_BLOCK_T_REPLAY`
- `FLARE_LSE_BWD_BLOCK_T_STATE`
- `FLARE_LSE_BWD_BLOCK_T_APPLY`
- `FLARE_LSE_BWD_SCALAR_APPLY_PANEL`
- `FLARE_LSE_BWD_REPLAY_NUM_WARPS` / `FLARE_LSE_BWD_REPLAY_NUM_STAGES`
- `FLARE_LSE_BWD_STATE_NUM_WARPS` / `FLARE_LSE_BWD_STATE_NUM_STAGES`
- `FLARE_LSE_BWD_QK_NUM_WARPS` / `FLARE_LSE_BWD_QK_NUM_STAGES`
- `FLARE_LSE_BWD_SCAN_APPLY_NUM_WARPS` / `FLARE_LSE_BWD_SCAN_APPLY_NUM_STAGES`

Practical implication:

- Benchmarks should use a staged matrix sweep and then promote a narrow policy back into runtime bucket rules.
- The default matrix should use representative anchors first, with `--full-matrix` reserved for confirmation.

### Inference

Runtime entry point:

- [`causal_flare/autoregressive/inference.py`](../causal_flare/autoregressive/inference.py)

Benchmark tools:

- [`benchmark_prefill_decode.py`](./benchmark_prefill_decode.py)
- [`benchmark_train_step.py`](./benchmark_train_step.py)
- [`profile_flare_inference.py`](./profile_flare_inference.py)
- [`measure_cold_start.py`](./measure_cold_start.py)
- [`tune_flare_inference_matrix.py`](./tune_flare_inference_matrix.py)
- [`flare_inference_matrix_workflow.md`](./flare_inference_matrix_workflow.md)

Inference is split into prefill and decode:

- Prefill reuses the chunked forward heuristics and benchmark-owned env knobs.
- Decode uses one heuristic-selected recurrent-step config plus optional `FLARE_DECODE_*` env overrides.
- Default user execution does not run Triton autotune search loops or runtime launch sweeps.

Practical implication:

- Prefill and decode benchmarking should sweep the explicit env knobs offline, then promote the winners back into runtime heuristics.

### Head-Dimension Terminology

- `score_head_dim` / `D_k`: the inner dimension used by `Q` and `K` for score/logit reductions and attention scaling.
- `value_head_dim` / `D_v`: the inner dimension used by `V`, outputs `Y`, and recurrent numerator/state storage.
- FLARE inference and chunked training benchmarks can exercise mixed-dimension cases such as `D_k=64, D_v=128`.
- FlashAttention2 comparison paths still fundamentally require `D_k == D_v`.

## What To Sweep

### Chunked benchmark sweeps

Recommended:

- `FLARE_CHUNK_SIZE`
- `FLARE_BLOCK_M`
- `FLARE_PREFIX_BLOCK_M`
- `FLARE_PREPARE_BLOCK_D`
- `FLARE_PREPARE_BLOCK_K`
- `FLARE_PREFIX_BLOCK_D`
- `FLARE_FWD_BLOCK_D`
- `FLARE_FWD_BLOCK_K`
- forward launch presets
- `FLARE_LSE_BWD_BLOCK_M`
- `FLARE_LSE_BWD_SCORE_BLOCK_M`
- `FLARE_LSE_BWD_BLOCK_DV`
- `FLARE_LSE_BWD_BLOCK_K`
- `FLARE_LSE_BWD_BLOCK_D_PART`
- `FLARE_LSE_BWD_QK_BLOCK_D`
- `FLARE_LSE_BWD_BLOCK_T_QK`
- backward launch presets

Default anchor matrix and staging live in [chunked_flare_matrix_workflow.md](./chunked_flare_matrix_workflow.md).

### Inference benchmark sweeps

Prefill:

- `FLARE_CHUNK_SIZE`
- `FLARE_BLOCK_M`
- `FLARE_PREPARE_BLOCK_D`
- `FLARE_PREPARE_BLOCK_K`
- `FLARE_PREFIX_BLOCK_D`
- `FLARE_FWD_BLOCK_D`
- `FLARE_FWD_BLOCK_K`
- prefill launch presets

Decode:

- `FLARE_DECODE_BLOCK_D`
- `FLARE_DECODE_BLOCK_K`
- `FLARE_DECODE_NUM_WARPS`
- `FLARE_DECODE_NUM_STAGES`

Example mixed-dimension benchmark:

```bash
python benchmark/benchmark_prefill_decode.py \
  --score-head-dim 64 \
  --value-head-dim 128 \
  --bench-modes prefill decode
```

Example mixed-dimension train-step benchmark:

```bash
python benchmark/benchmark_train_step.py \
  --score-head-dim 64 \
  --value-head-dim 128 \
  --skip-fa2
```

## Notes

- Runtime no longer owns any default-path Triton autotune search in chunked or inference decode.
- The benchmark workflow is intentionally where compile-heavy exploration happens; normal inference and training should compile only the chosen heuristic path.
- If a knob is removed from runtime heuristics, remove it from the matrix workflow at the same time.

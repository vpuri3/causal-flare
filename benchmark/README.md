# Benchmark Tuning Guide

This directory contains the benchmark and profiling workflows for the current FLARE launch-tuning policy.

Use this file as the high-level source of truth for what is still tuned manually and what is now owned by Triton autotune.

## Current Policy

### Chunked

Runtime entry point:

- [`causal_flare/chunked.py`](../causal_flare/chunked.py)

Benchmark tools:

- [`profile_chunked_flare.py`](./profile_chunked_flare.py)
- [`tune_chunked_flare_matrix.py`](./tune_chunked_flare_matrix.py)
- [`chunked_flare_matrix_workflow.md`](./chunked_flare_matrix_workflow.md)

Chunked is tuned hierarchically:

- Structural knobs are still chosen outside Triton autotune.
  - `CHUNK_SIZE`
  - forward `BLOCK_M`
  - backward `BLOCK_M`
  - backward `BLOCK_DV`
- Kernel-local tile and launch choices are handled by Triton autotune.
  - forward `prepare`, `prefix`, `decoder_lse`, `fwd`
  - backward replay/summary/apply/reduce kernels

Practical implication:

- Benchmarks should primarily sweep structural families.
- Local kernel tiles such as `BLOCK_D`, `BLOCK_K`, `BLOCK_T`, `num_warps`, and `num_stages` are no longer first-class runtime tuning env knobs.

### Inference

Runtime entry point:

- [`causal_flare/inference.py`](../causal_flare/inference.py)

Benchmark tools:

- [`profile_flare_inference.py`](./profile_flare_inference.py)
- [`tune_flare_inference_matrix.py`](./tune_flare_inference_matrix.py)
- [`flare_inference_matrix_workflow.md`](./flare_inference_matrix_workflow.md)

Inference is split into prefill and decode:

- Prefill reuses the Chunked forward kernels.
  - Structural sweeps still make sense for `CHUNK_SIZE` and `BLOCK_M`.
  - Local forward kernel launch choices are autotuned.
- Decode uses one Triton recurrent-step kernel with a curated autotune shortlist.
  - Shared and nonshared decode now reuse the same autotune choice.
  - There are no runtime `FLARE_DECODE_*` launch env overrides anymore.

Practical implication:

- Decode benchmarking should compare end-to-end latency and per-kernel profile totals, not try to brute-force runtime launch envs that no longer exist.

## What To Sweep

### Chunked benchmark sweeps

Recommended:

- `FLARE_CHUNK_SIZE`
- `FLARE_BLOCK_M`
- `FLARE_LSE_BWD_BLOCK_M`
- `FLARE_LSE_BWD_BLOCK_DV`

Avoid rebuilding a manual search over:

- per-kernel `BLOCK_D`
- per-kernel `BLOCK_K`
- per-kernel `BLOCK_T`
- `num_warps`
- `num_stages`

### Inference benchmark sweeps

Prefill:

- `FLARE_CHUNK_SIZE`
- `FLARE_BLOCK_M`

Decode:

- no runtime launch env sweep
- measure the runtime autotuned path directly

## Notes

- The older workflow markdown files may still contain historical staging detail from the manual-launch era. Use them for matrix structure and artifact conventions, but treat this README and the runtime code as authoritative for which knobs are still real.
- If a new kernel-local knob becomes worth exposing again, update this README at the same time as the runtime and benchmark scripts.

# FLARE Inference Matrix Tuning Workflow

Current source of truth for which knobs are owned by offline tuning and then promoted into runtime heuristics:

- [`README.md`](./README.md)

This workflow is intentionally benchmark-owned rather than runtime-autotuned. We tried using Triton autotune on the default runtime path, but compile and first-use latency became extremely large, so inference returned to heuristic config selection plus offline matrix sweeps.

Use this workflow to retune FLARE inference after major kernel/code changes:

- prefill: [`profile_flare_inference.py`](./profile_flare_inference.py) and [`tune_flare_inference_matrix.py`](./tune_flare_inference_matrix.py) with `--mode prefill`
- decode: [`profile_flare_inference.py`](./profile_flare_inference.py) and [`tune_flare_inference_matrix.py`](./tune_flare_inference_matrix.py) with `--mode decode`

## Matrix

- `D_k = 16, 32, 64, 96, 128, 192, 256, 384, 512`
- `D_v = D_k` for baseline sweeps, plus targeted mixed-dimension checkpoints such as `D_k=64, D_v=128`
- `M = 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048`
- `N = 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072`
- `BH = multiples of 4`

As with training, keep `BH` fixed at one representative multiple of 4 unless you have evidence the `B/H` factorization matters.

## Saved Artifacts

Each run saves under `results/flare_inference_matrix/<run-name>/`:

- `runs-shardXXX-of-YYY.jsonl`
  - one record per `(case, config)`
  - preserves end-to-end latency, per-kernel timings, and compiled-kernel resource metadata where available
- `summary-shardXXX-of-YYY.json`
- `summary-shardXXX-of-YYY.md`

## Prefill Seed

Use one representative seed point such as `BH=8`, `M=64`, `N=4096`:

```bash
source .venv/bin/activate
python benchmark/tune_flare_inference_matrix.py \
  --mode prefill \
  --run-name inference-prefill-seed-bh8-m64-n4096 \
  --d-values 16,32,64,96,128,192,256,384,512 \
  --m-values 64 \
  --n-values 4096 \
  --bh-values 8 \
  --families default,chunk_size,prefill_block_m,combined
```

Focus on:

- `CHUNK_SIZE`
- prefill `BLOCK_M`
- prefill `BLOCK_D` / `BLOCK_K`
- prefill launch presets

## Decode Seed

```bash
python benchmark/tune_flare_inference_matrix.py \
  --mode decode \
  --run-name inference-decode-seed-bh8-m64-n4096 \
  --d-values 16,32,64,96,128,192,256,384,512 \
  --m-values 64 \
  --n-values 4096 \
  --bh-values 8 \
  --decode-steps 256 \
  --families default,decode_block_d,decode_block_k,decode_launch,combined
```

Focus on:

- decode `BLOCK_D`
- decode `BLOCK_K`
- decode launch geometry (`FLARE_DECODE_NUM_WARPS`, `FLARE_DECODE_NUM_STAGES`)

## Notes

- Prefill and decode both run with heuristic-selected configs in the default user path.
- The matrix runner owns any comparison across `FLARE_DECODE_*` and prefill launch/tile env knobs.
- This separation is deliberate: benchmark jobs absorb the expensive search/compile work so normal inference does not.
- Failed candidate configs are recorded as error rows instead of aborting the entire shard.
- For the lightweight end-to-end benchmark/profiler CLIs, prefer explicit `--score-head-dim` / `--value-head-dim` naming when exercising mixed-dimension cases.

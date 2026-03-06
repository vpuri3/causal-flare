# FLARE Inference Matrix Tuning Workflow

Use this workflow to retune FLARE inference after major kernel/code changes:

- prefill: [`profile_flare_inference.py`](./profile_flare_inference.py) and [`tune_flare_inference_matrix.py`](./tune_flare_inference_matrix.py) with `--mode prefill`
- decode: [`profile_flare_inference.py`](./profile_flare_inference.py) and [`tune_flare_inference_matrix.py`](./tune_flare_inference_matrix.py) with `--mode decode`

## Matrix

- `D = 16, 32, 64, 96, 128, 192, 256, 384, 512`
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
  --families default,chunk_size,prefill_block_d,prefill_block_k,prefill_launch,combined
```

Focus on:

- `CHUNK_SIZE`
- prefill `BLOCK_D`
- prefill `BLOCK_K`
- phase-specific forward launches

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

- Decode now validates `BLOCK_D` as a power-of-two tile and requires `BLOCK_K` to divide `D`.
- Failed candidate configs are recorded as error rows instead of aborting the entire shard.

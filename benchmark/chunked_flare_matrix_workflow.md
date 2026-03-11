# Chunked FLARE Matrix Tuning Workflow

Current source of truth for which knobs are still structural versus autotuned:

- [`README.md`](./README.md)

This workflow turns launch-config tuning into a repeatable sweep over the expected deployment matrix:

- `D = 16, 32, 64, 96, 128, 192, 256, 384, 512`
- `M = 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048`
- `N = 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072`
- `BH = multiples of 4`

The runner is [`tune_chunked_flare_matrix.py`](./tune_chunked_flare_matrix.py). It reuses the per-kernel profiler in [`profile_chunked_flare.py`](./profile_chunked_flare.py) and automatically saves raw profiling artifacts plus a grouped summary.

## Why This Is Staged

Do not brute-force the full Cartesian product of every launch knob across the entire matrix in one pass. That is too expensive and makes it hard to separate width-driven effects from `M` or `N` effects.

Use three passes:

1. Seed `D` families at one representative `(M, N, BH)` point.
2. Refine `BLOCK_M`, `CHUNK_SIZE`, and related knobs over the wider `M x N` matrix using the seeded width families.
3. Confirm the promoted configs across the full matrix, then update the heuristic buckets in `causal_flare/chunked.py`.

`BH` is usually held fixed at one representative multiple of 4 because the kernels parallelize over `B * H`. If a regression appears BH-sensitive, rerun the same stage with a different `--batch-size` / `--bh-values` factorization.

## Saved Artifacts

Every run saves under `results/chunked_flare_matrix/<run-name>/`:

- `runs-shardXXX-of-YYY.jsonl`
  - canonical raw profiling data
  - one JSON record per `(case, config)` run
  - preserves:
    - end-to-end latency
    - forward per-kernel timings
    - backward per-kernel timings
    - forward compiled-kernel occupancy estimate
    - forward register pressure
    - forward shared-memory usage
- `summary-shardXXX-of-YYY.json`
  - best config per case
  - grouped winner counts by `D`, `M`, `N`, and `BH`
- `summary-shardXXX-of-YYY.md`
  - quick human-readable summary

Use `--run-name <name> --resume` to continue a long sweep.

## Stage 1: Seed Width Families

Start with one representative training point such as `BH=8`, `M=64`, `N=4096`, then sweep all `D` buckets with the core families:

```bash
source .venv/bin/activate
python benchmark/tune_chunked_flare_matrix.py \
  --run-name d-seed-bh8-m64-n4096 \
  --d-values 16,32,64,96,128,192,256,384,512 \
  --m-values 64 \
  --n-values 4096 \
  --bh-values 8 \
  --families core,combined \
  --mode both
```

Goal:

- identify width-driven winners for:
  - forward `BLOCK_D`, `BLOCK_K`, and per-phase launches
  - backward replay/QK launches and wide-D replay tile choices

Promote only patterns that win consistently across neighboring `D` buckets.

## Stage 2: Refine `M` And `N`

Once the `D` families are seeded, sweep the larger `M x N` space while keeping the focus on `BLOCK_M`, `CHUNK_SIZE`, and token-panel knobs:

```bash
python benchmark/tune_chunked_flare_matrix.py \
  --run-name mn-refine-bh8 \
  --d-values 16,32,64,96,128,192,256,384,512 \
  --m-values 16,32,64,96,128,192,256,384,512,768,1024,2048 \
  --n-values 1024,2048,4096,8192,16384,32768,65536,131072 \
  --bh-values 8 \
  --families chunk_size,forward_block_m,backward_block_m,backward_block_t_qk,backward_block_t_state,combined \
  --mode both \
  --num-shards 8 \
  --shard-index 0
```

Run the same command for each shard index.

Goal:

- decide where `BLOCK_M=32` vs `64` vs larger tiles win
- decide whether `CHUNK_SIZE` should change with `N`
- verify whether backward token-panel choices need separate large-`N` handling

## Stage 3: Confirm Promoted Families

After reading the stage-1 and stage-2 summaries, rerun only the promoted families or explicit hand-picked configs:

```bash
python benchmark/tune_chunked_flare_matrix.py \
  --run-name full-confirm-bh8 \
  --d-values 16,32,64,96,128,192,256,384,512 \
  --m-values 16,32,64,96,128,192,256,384,512,768,1024,2048 \
  --n-values 1024,2048,4096,8192,16384,32768,65536,131072 \
  --bh-values 8 \
  --families default \
  --extra-config promoted_wide_d:FLARE_FWD_BLOCK_D=128,FLARE_FWD_BLOCK_K=32,FLARE_LSE_BWD_BLOCK_M=64,FLARE_LSE_BWD_BLOCK_K=32 \
  --extra-config promoted_large_n:FLARE_CHUNK_SIZE=256,FLARE_BLOCK_M=64,FLARE_PREFIX_BLOCK_M=64 \
  --mode both \
  --num-shards 8 \
  --shard-index 0
```

Goal:

- validate the small set of configs that are candidates for promotion into `chunked.py`
- keep the promoted defaults narrow and evidence-based

## Families

`core`:

- `default`
- `chunk_size`
- `forward_block_m`
- `forward_block_d`
- `forward_block_k`
- `forward_launch`
- `backward_block_m`
- `backward_block_k`
- `backward_qk_block_d`
- `backward_block_t_qk`
- `backward_launch`

`extended`:

- `backward_block_dv`
- `backward_block_d_part`
- `backward_block_t_replay`
- `backward_block_t_state`
- `backward_block_t_apply`
- `backward_scalar_apply_panel`
- `backward_fused_chunk_scan`

`combined`:

- merges the best winner from each selected family for the same case
- useful as a cheap second pass before promoting a config family

## Reading The Results

Start with `summary.json`, then inspect the raw JSONL for the winning cases.

Things to look for:

- large wins concentrated in one `D` bucket:
  - promote a `D`-bucket heuristic, not a global one
- wins concentrated in one `N` bucket:
  - consider `CHUNK_SIZE` or prefix/replay launch updates
- wins concentrated in one `M` bucket:
  - consider `BLOCK_M` or replay subtile changes
- forward hot kernel still `flare_chunk_fwd` with low occupancy or high regs:
  - likely replay/QK work duplication or register pressure
- backward hot kernel still `flare_chunk_bwd_lse_p_part`:
  - keep tuning replay family before touching other phases

Current limitation:

- forward resource metadata is saved today
- backward raw records currently preserve timings only, not compiled occupancy/register metadata

If backward kernel resource metadata becomes important, extend `profile_chunked_flare.py` in the same style as the forward compiled-kernel capture.

# Chunked FLARE Matrix Tuning Workflow

Current source of truth for which knobs are owned by offline tuning and then promoted into runtime heuristics:

- [`README.md`](./README.md)

This workflow exists because runtime Triton autotune on the chunked user path proved too expensive in practice. We tried it, but cold-cache compile and first-use latency became extremely large, so the project moved back to offline matrix sweeps plus promoted heuristic buckets in `causal_flare/chunked.py`.

This workflow turns launch-config tuning into a repeatable sweep over the expected deployment matrix:

- fast default anchors: `D_k = 64, 128, 256`
- `D_v = D_k` for baseline sweeps, plus targeted mixed-dimension confirmations such as `D_k=64, D_v=128`
- fast default anchors: `M = 64, 512`
- fast default anchors: `N = 2048, 32768`
- `BH = 8` by default

Use `--full-matrix` only for explicit exhaustive confirmation sweeps. The default runner now tunes representative shape buckets first rather than treating the full deployment matrix as the day-to-day workflow or pushing that exploration into runtime.

The runner is [`tune_chunked_flare_matrix.py`](./tune_chunked_flare_matrix.py). It reuses the per-kernel profiler in [`profile_chunked_flare.py`](./profile_chunked_flare.py) and automatically saves raw profiling artifacts plus a grouped summary.

## Why This Is Staged

Do not brute-force the full Cartesian product of every launch knob across the entire matrix in one pass. That is too expensive and makes it hard to separate width-driven effects from `M` or `N` effects.

The same logic applies to runtime policy: users should not pay this exploration cost on their first forward or backward call. The runtime only consumes the promoted winners.

Use three passes:

1. Seed `D` families at one representative `(M, N, BH)` point.
2. Refine `BLOCK_M`, width-dependent tile choices, and launch presets over the wider `M x N` anchor matrix using the seeded width families.
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

Start with one representative training point such as `BH=8`, `M=64`, `N=2048`, then sweep the anchor `D` buckets with the width families:

```bash
source .venv/bin/activate
python benchmark/tune_chunked_flare_matrix.py \
  --run-name d-seed-bh8-m64-n2048 \
  --d-values 64,128,256 \
  --m-values 64 \
  --n-values 2048 \
  --bh-values 8 \
  --families full_forward \
  --mode forward
```

Goal:

- identify width-driven winners for:
  - `CHUNK_SIZE`
  - forward `BLOCK_M`
  - `FLARE_PREPARE_BLOCK_D` / `FLARE_PREPARE_BLOCK_K`
  - `FLARE_PREFIX_BLOCK_D`
  - `FLARE_FWD_BLOCK_D` / `FLARE_FWD_BLOCK_K`
  - forward launch presets

Promote only patterns that win consistently across neighboring `D` buckets.

## Stage 2: Refine `M` And `N`

Once the `D` buckets are seeded, sweep the larger `M x N` anchor space while keeping the focus on the promoted runtime knobs:

```bash
python benchmark/tune_chunked_flare_matrix.py \
  --run-name mn-refine-bh8 \
  --d-values 64,128,256 \
  --m-values 64,512 \
  --n-values 2048,32768 \
  --bh-values 8 \
  --families core_fast,combined \
  --mode both \
  --num-shards 4 \
  --shard-index 0
```

Run the same command for each shard index.

Goal:

- decide where `BLOCK_M=32` vs `64` vs larger tiles win
- decide whether `CHUNK_SIZE=32,64,128,256` should change with `N`
- decide whether backward `BLOCK_M` / `BLOCK_DV` need separate large-`N` handling
- decide whether replay/QK launch presets still need different wide-`D` or large-`N` handling

## Stage 3: Confirm Promoted Families

After reading the stage-1 and stage-2 summaries, rerun only the promoted families or explicit hand-picked configs:

```bash
python benchmark/tune_chunked_flare_matrix.py \
  --run-name full-confirm-bh8 \
  --full-matrix \
  --bh-values 8 \
  --families default \
  --extra-config promoted_structural:FLARE_CHUNK_SIZE=64,FLARE_BLOCK_M=64,FLARE_LSE_BWD_BLOCK_M=64,FLARE_LSE_BWD_BLOCK_DV=128 \
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

`core_fast`:

- `default`
- `chunk_size`
- `forward_block_m`
- `backward_block_m`
- `backward_block_dv`

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

Important:

- The runtime path is heuristic-only by default.
- That is intentional. We previously tried runtime Triton autotune, but the compile/autotune cost was too high for acceptable user experience.
- These families are the place where tile and launch alternatives are explored before promoting winners back into `causal_flare/chunked.py`.

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

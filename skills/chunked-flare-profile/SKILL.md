---
name: chunked-flare-profile
description: Use when profiling or tuning chunked FLARE launch heuristics after kernel/code changes. This skill covers forward and backward per-kernel timings, Triton forward-kernel resource inspection, and launch-config sweeps over block sizes, chunk size, and per-phase warp/stage knobs using the repo profiler script.
---

# Chunked FLARE Profile

Use this skill when the task is to profile chunked FLARE on CUDA, identify forward or backward bottlenecks, or retune launch heuristics for a specific `(B, H, N, M, D, dtype)` case after meaningful kernel changes.

For full-matrix retuning across the supported `D x M x N` space:

- training / chunked forward+backward: [`../../benchmark/chunked_flare_matrix_workflow.md`](../../benchmark/chunked_flare_matrix_workflow.md) and [`../../benchmark/tune_chunked_flare_matrix.py`](../../benchmark/tune_chunked_flare_matrix.py)
- inference prefill/decode: [`../../benchmark/flare_inference_matrix_workflow.md`](../../benchmark/flare_inference_matrix_workflow.md) and [`../../benchmark/tune_flare_inference_matrix.py`](../../benchmark/tune_flare_inference_matrix.py)

Both runners save raw profiling artifacts automatically so later kernel work can diff per-kernel timing, occupancy, and register pressure.

## Quick Start

1. Read `TODO.md` and keep benchmark-quality reproducibility in scope.
2. Activate the repo venv.
3. Run [`../../benchmark/profile_chunked_flare.py`](../../benchmark/profile_chunked_flare.py) in `--mode both` for the target case.
4. For forward, focus first on `flare_chunk_fwd`, then `flare_chunk_prepare`, then `flare_chunk_decoder_lse`.
5. For backward, focus first on `flare_chunk_bwd_lse_p_part`, then `flare_chunk_bwd_recurrent_qk`, then the state/apply kernels.

Typical command for the investigated large-D cases:

```bash
source .venv/bin/activate
python benchmark/profile_chunked_flare.py \
  --mode both \
  --case h16_d32,8,16,4096,64,32 \
  --case h4_d128,8,4,4096,64,128 \
  --case h4_d256,8,4,4096,64,256 \
  --config default:
```

The profiler reports:

- End-to-end latency from `triton.testing.do_bench`
- Per-kernel forward and backward timings from `profile=True`
- Triton compiled-kernel `n_regs`, spills, shared memory for forward kernels
- Estimated CTA residency / occupancy from compiled forward metadata

Important limitations:

- If `ncu` is unavailable, treat occupancy as an estimate derived from Triton metadata plus device limits.
- The script currently reports compiled resource metadata for forward kernels only; use backward timings to identify the hot phase, then sweep backward launch knobs directly.

For matrix-wide tuning:

- hold `BH` fixed at one representative multiple of 4 unless you have evidence that a regression depends on `B/H` factorization
- keep raw artifacts under `results/chunked_flare_matrix/<run-name>/`
- use `--run-name <name> --resume` so long shard sweeps can be continued safely
- do not brute-force every knob over the entire matrix in one pass; use the staged process below

## What To Tune

Start with launch geometry before changing algorithm structure.

Forward knobs:

- `FLARE_BLOCK_M`
- `FLARE_CHUNK_SIZE`
- `FLARE_PREPARE_NUM_WARPS`, `FLARE_PREPARE_NUM_STAGES`
- `FLARE_PREFIX_NUM_WARPS`, `FLARE_PREFIX_NUM_STAGES`
- `FLARE_DECODER_NUM_WARPS`, `FLARE_DECODER_NUM_STAGES`
- `FLARE_FWD_NUM_WARPS`, `FLARE_FWD_NUM_STAGES`
- `FLARE_PREPARE_BLOCK_D`, `FLARE_PREPARE_BLOCK_K`
- `FLARE_PREFIX_BLOCK_M`, `FLARE_PREFIX_BLOCK_D`
- `FLARE_FWD_BLOCK_D`, `FLARE_FWD_BLOCK_K`

Backward knobs:

- `FLARE_LSE_BWD_BLOCK_M`
- `FLARE_LSE_BWD_SCORE_BLOCK_M`
- `FLARE_LSE_BWD_BLOCK_DV`
- `FLARE_LSE_BWD_BLOCK_K`
- `FLARE_LSE_BWD_BLOCK_D_PART`
- `FLARE_LSE_BWD_BLOCK_T_QK`
- `FLARE_LSE_BWD_QK_BLOCK_D`
- `FLARE_LSE_BWD_REPLAY_NUM_WARPS`, `FLARE_LSE_BWD_REPLAY_NUM_STAGES`
- `FLARE_LSE_BWD_STATE_NUM_WARPS`, `FLARE_LSE_BWD_STATE_NUM_STAGES`
- `FLARE_LSE_BWD_QK_NUM_WARPS`, `FLARE_LSE_BWD_QK_NUM_STAGES`
- `FLARE_LSE_BWD_SCAN_APPLY_NUM_WARPS`, `FLARE_LSE_BWD_SCAN_APPLY_NUM_STAGES`

Avoid these as a first move for `D=128` forward:

- Forcing `BLOCK_D=64` in prepare/fwd. In this repo it increased duplicated score-reduction work and made `D=128` materially slower.

## Interpretation

Use this ordering when reading results:

1. `end_to_end_ms`: confirms whether a tuning change matters in the actual pass.
2. Forward: `flare_chunk_fwd` domination usually means repeated QK work or register pressure in the replay kernel.
3. Backward: `flare_chunk_bwd_lse_p_part` domination usually means the replay family (`BLOCK_M`, `BLOCK_K`, replay warps) is wrong.
4. `regs_per_thread` + `cta/sm` + `limit`: use this on forward kernels to distinguish register pressure from shared-memory pressure.
5. Check whether a tuning change moved time into a different phase instead of actually removing work.

Heuristics from the current H100 bf16 study at `B=8, N=4096, M=64`:

- Forward buckets:
  - `D=16/32/64` keep the low-D family.
  - `D=96` prefers `64D` tiles and `fwd=2 warps/1 stage`.
  - `D=128` prefers `128D` tiles, `PREPARE_BLOCK_K=64`, `FWD_BLOCK_K=32`, `decoder=4w/1s`, `fwd=4w/1s`.
  - `D=192/256/384` prefer `128D` forward tiles with `FWD_BLOCK_K=32`.
  - `D=512` prefers `128D` forward tiles with `FWD_BLOCK_K=64`.
- Backward buckets:
  - `D=16/32/64` keep the smaller replay family.
  - `D=96/128/192/256/384/512` all preferred `BLOCK_M=64`, `BLOCK_K=32`, `BLOCK_D_PART=64`, `QK_BLOCK_D=64`, `replay=4 warps/2 stages`.

## Comparison Workflow

When the user asks why one shape is slower than another, compare a width-matched baseline and the target shape in the same run. For this repo the most useful comparison is:

```bash
python benchmark/profile_chunked_flare.py \
  --mode both \
  --case h16_d32,8,16,4096,64,32 \
  --case h4_d128,8,4,4096,64,128 \
  --case h4_d256,8,4,4096,64,256 \
  --config default:
```

To sweep a candidate backward replay family against the default:

```bash
python benchmark/profile_chunked_flare.py \
  --mode both \
  --case h4_d256,8,4,4096,64,256 \
  --config default: \
  --config bwd_replay_m64_k32:FLARE_LSE_BWD_BLOCK_M=64,FLARE_LSE_BWD_SCORE_BLOCK_M=64,FLARE_LSE_BWD_BLOCK_K=32,FLARE_LSE_BWD_REPLAY_NUM_WARPS=4
```

Then summarize:

- Which kernel owns the extra milliseconds
- Whether the regression is from register pressure, shared memory, or duplicated reduction work
- Whether the tuned default already fixes it or whether more invasive kernel work is needed

## Matrix Workflow

Use [`../../benchmark/tune_chunked_flare_matrix.py`](../../benchmark/tune_chunked_flare_matrix.py) when the job is broader than one shape. The workflow is:

1. Seed width-driven families at one representative `(M, N, BH)` point.
2. Sweep `M` and `N` to tune `BLOCK_M`, `CHUNK_SIZE`, and token-panel knobs.
3. Confirm only the promoted config families across the full matrix, then patch the heuristic buckets in `causal_flare/chunked.py`.

Representative stage-1 command:

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

Representative stage-2 command:

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

Important artifacts:

- `runs-shardXXX-of-YYY.jsonl`: raw per-case, per-config profiling records
- `summary-shardXXX-of-YYY.json`: best-config summary and grouped winners
- `summary-shardXXX-of-YYY.md`: quick human-readable report

The raw JSONL is the canonical source for later kernel work. It preserves forward per-kernel timing plus forward occupancy/register/shared-memory signals, and backward per-kernel timings.

## Inference Workflow

Use [`../../benchmark/profile_flare_inference.py`](../../benchmark/profile_flare_inference.py) for one-off prefill/decode profiling and [`../../benchmark/tune_flare_inference_matrix.py`](../../benchmark/tune_flare_inference_matrix.py) for matrix sweeps.

Representative prefill seed:

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

Representative decode seed:

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

Inference artifacts are saved under `results/flare_inference_matrix/<run-name>/`.

## Reporting Guidance

Include concrete numbers, not just conclusions.

Minimum useful summary:

- Shape and dtype
- End-to-end latency
- Top forward and backward kernels with percentages
- `n_regs`, shared memory, estimated CTA residency for the hottest forward kernel
- Recommended next knob or structural bottleneck

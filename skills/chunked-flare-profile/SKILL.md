---
name: chunked-flare-profile
description: Use when profiling or tuning chunked FLARE launch heuristics after kernel/code changes. This skill covers forward and backward per-kernel timings, Triton forward-kernel resource inspection, and launch-config sweeps over block sizes, chunk size, and per-phase warp/stage knobs using the repo profiler script.
---

# Chunked FLARE Profile

Use this skill when the task is to profile chunked FLARE on CUDA, identify forward or backward bottlenecks, or retune launch heuristics for a specific `(B, H, N, M, D, dtype)` case after meaningful kernel changes.

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

## Reporting Guidance

Include concrete numbers, not just conclusions.

Minimum useful summary:

- Shape and dtype
- End-to-end latency
- Top forward and backward kernels with percentages
- `n_regs`, shared memory, estimated CTA residency for the hottest forward kernel
- Recommended next knob or structural bottleneck

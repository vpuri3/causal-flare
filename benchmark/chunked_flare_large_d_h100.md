# Chunked FLARE Launch-Tuning Notes on H100

Hardware and setup:

- GPU: NVIDIA H100 80GB HBM3 (`sm90`)
- dtype: `bf16`
- shapes compared at fixed `B=8`, `N=4096`, `M=64`
- width buckets studied: `D=16, 32, 64, 96, 128, 192, 256, 384, 512`
- baseline comparison used during the original large-D investigation: `H=16, D=32`

Profiler entry point:

- [`profile_chunked_flare.py`](./profile_chunked_flare.py)

## Main Findings

The tuned defaults now use explicit D buckets in both forward and backward.

- Forward:
  - `D=128` was primarily a launch-geometry problem.
  - `D=256+` is partly a launch problem and partly structural because `flare_chunk_fwd` repeats QK score reduction across output-D tiles.
- Backward:
  - From `D=96` upward, the dominant kernel is `flare_chunk_bwd_lse_p_part`.
  - The old replay family (`BLOCK_M=32`, `BLOCK_K=64`, `8 warps`) was the wrong tradeoff on H100 for `M=64`.
  - A single `BLOCK_M=64` replay tile with `BLOCK_K=32` and `4 replay warps` roughly halves wide-D backward time.

## Forward Timings

Measured end-to-end forward latency:

| Case | Old default heuristic | New default heuristic |
| --- | ---: | ---: |
| `H=16, D=32` | `1.915 ms` | `1.920 ms` |
| `H=4, D=128` | `2.825 ms` | `1.910 ms` |
| `H=4, D=256` | `11.175 ms` | `4.629 ms` |

Representative explicit forward bucket defaults:

| D bucket | Key forward defaults |
| --- | --- |
| `16` | keep low-D path |
| `32` | `decoder=4w/1s`, `fwd=4w/1s` |
| `64` | `decoder=4w/1s`, `fwd=4w/1s` |
| `96` | `64D` tiles, `FWD_BLOCK_K=32`, `fwd=2w/1s` |
| `128` | `128D` tiles, `PREPARE_BLOCK_K=64`, `FWD_BLOCK_K=32`, `decoder=4w/1s`, `fwd=4w/1s` |
| `192` | `128D` tiles, `FWD_BLOCK_K=32`, `fwd=2w/1s` |
| `256` | `128D` tiles, `FWD_BLOCK_K=32`, `fwd=4w/1s` |
| `384` | `128D` tiles, `FWD_BLOCK_K=32`, `fwd=4w/1s` |
| `512` | `128D` tiles, `FWD_BLOCK_K=64`, `fwd=4w/1s` |

## Backward Timings

Measured end-to-end forward+backward latency for the active default path before and after the wide-D backward retune:

| D bucket | Old default | New default | Change |
| --- | ---: | ---: | ---: |
| `96` | `12.621 ms` | `6.151 ms` | `2.05x` faster |
| `128` | `12.354 ms` | `6.350 ms` | `1.95x` faster |
| `192` | `19.927 ms` | `10.098 ms` | `1.97x` faster |
| `256` | `26.776 ms` | `13.706 ms` | `1.95x` faster |
| `384` | `44.152 ms` | `22.552 ms` | `1.96x` faster |
| `512` | `64.068 ms` | `32.975 ms` | `1.94x` faster |

The hot backward kernel was consistently `flare_chunk_bwd_lse_p_part`. Representative kernel reductions:

| D bucket | Old `p_part` | New `p_part` |
| --- | ---: | ---: |
| `96` | `7.718 ms` | `1.750 ms` |
| `128` | `7.756 ms` | `2.129 ms` |
| `192` | `12.212 ms` | `3.401 ms` |
| `256` | `17.054 ms` | `4.971 ms` |
| `384` | `28.754 ms` | `9.077 ms` |
| `512` | `43.768 ms` | `14.393 ms` |

Backward bucket policy now encoded in code:

| D bucket | Key backward defaults |
| --- | --- |
| `16` | `BLOCK_M=32/64`, `BLOCK_K=16`, `BLOCK_D_PART=16`, `QK_BLOCK_D=16`, `replay=4w/2s` |
| `32` | `BLOCK_M=64`, `BLOCK_K=32`, `BLOCK_D_PART=32`, `QK_BLOCK_D=32`, `replay=4w/2s` |
| `64` | `BLOCK_M=64`, `BLOCK_K=64`, `BLOCK_D_PART=64`, `QK_BLOCK_D=64`, `replay=4w/2s` |
| `96` | `BLOCK_M=64`, `BLOCK_K=32`, `BLOCK_D_PART=64`, `QK_BLOCK_D=64`, `replay=4w/2s` |
| `128` | `BLOCK_M=64`, `BLOCK_K=32`, `BLOCK_D_PART=64`, `QK_BLOCK_D=64`, `replay=4w/2s` |
| `192` | `BLOCK_M=64`, `BLOCK_K=32`, `BLOCK_D_PART=64`, `QK_BLOCK_D=64`, `replay=4w/2s` |
| `256` | `BLOCK_M=64`, `BLOCK_K=32`, `BLOCK_D_PART=64`, `QK_BLOCK_D=64`, `replay=4w/2s` |
| `384` | `BLOCK_M=64`, `BLOCK_K=32`, `BLOCK_D_PART=64`, `QK_BLOCK_D=64`, `replay=4w/2s` |
| `512` | `BLOCK_M=64`, `BLOCK_K=32`, `BLOCK_D_PART=64`, `QK_BLOCK_D=64`, `replay=4w/2s` |

## Resource Signals

Representative compiled-kernel metrics from tuned forward defaults:

| Case | Kernel | Regs/thread | Shared memory | Estimated CTA/SM | Main limit |
| --- | --- | ---: | ---: | ---: | --- |
| `H=4, D=128` | `flare_chunk_fwd` | `255` | `33.0 KiB` | `2` | registers |
| `H=4, D=128` | `flare_chunk_prepare` | `231` | `192.0 KiB` | `1` | registers + shared memory |
| `H=4, D=256` | `flare_chunk_fwd` | `228` | `33.0 KiB` | `2` | registers |
| `H=4, D=256` | `flare_chunk_prepare` | `255` | `192.0 KiB` | `1` | registers + shared memory |

Interpretation:

- `flare_chunk_fwd` remains the main forward bottleneck for large D and is register-limited.
- `flare_chunk_prepare` is both register- and shared-memory-constrained, but it is not the dominant forward runtime.
- Wide-D backward is no longer blocked by the old replay family; after the retune the next meaningful wide-D bottleneck is again the forward replay kernel.

## What Did Not Help

- Forcing `BLOCK_D=64` at `D=128` forward made the kernel much slower.
- Keeping the old wide-D backward replay family (`BLOCK_M=32`, `BLOCK_K=64`, `8 warps`) left `flare_chunk_bwd_lse_p_part` massively over-provisioned and slow.
- Using `BLOCK_D_PART=32` for `D=128` backward was a regression.
- Increasing backward QK tile width to `QK_BLOCK_D=128` at `D=256` did not offset the larger contraction footprint.

## Remaining Bottleneck

After the tuned defaults:

- `D=128` forward is in a good place for this workload.
- `D=256+` forward is still dominated by `flare_chunk_fwd`, with repeated full-D score reduction across output-D tiles.
- Backward is materially healthier across wide D, and the next big win would come from structural forward replay changes rather than more launch tweaking.

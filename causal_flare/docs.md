# ChunkedFLARE Notes

This document describes the current `ChunkedFLARE` implementation in
[`chunked.py`](./chunked.py). It replaces older torch-reference notes and is
meant to explain the actual Triton code path that runs today.

Within this folder, "causal FLARE" now effectively means the chunked
implementation: a causal decoder that preserves the recurrent softmax algebra,
but parallelizes most work by splitting the sequence into chunks.

## Tensor Layout

`ChunkedFLARE` uses the external layout directly:

- `Q`: `[H, M, D]`
- `K`: `[B, N, H, D]`
- `V`: `[B, N, H, D]`
- output `Y`: `[B, N, H, D]`

Where:

- `B` = batch size
- `H` = number of heads
- `N` = sequence length
- `M` = number of latent queries
- `D` = head dimension

The implementation does not permute inputs into a different public layout.

## High-Level Idea

For each token `t`, ChunkedFLARE maintains a causal encoder state over the
latent rows `m = 0..M-1`:

- a running max `mu_t(m)`
- a running denominator `d_t(m)`
- a running numerator `u_t(m, :)`

These are the usual online-softmax statistics for the scores

`score_t(m) = scale * <q_m, k_t>`.

The normalized latent state is:

`z_t(m, :) = u_t(m, :) / d_t(m)`.

The output token is then a decode over the latent rows:

`y_t = sum_m softmax_m(score_t)(m) * z_t(m, :)`.

The key optimization is to break the `N` tokens into chunks of size `C`, do
chunk-local recurrent work independently, scan chunk summaries, and then replay
token-level work inside each chunk.

## Forward Pass

The forward pass is implemented by `ChunkedFLARE.forward` in
[`chunked.py`](./chunked.py) and consists of three stages.

### 1. Build Chunk Configuration

`_get_chunked_forward_config` chooses:

- `CHUNK_SIZE`
- `BLOCK_M`
- `BLOCK_D` / `BLOCK_K` tiling for each phase
- `BLOCK_T` token micro-tile
- Triton launch settings

There is one narrow default override: if no explicit chunk size is provided and
the shape is the current measured H100 target (`bf16`, `M=64`, `D=32`,
`N=2048`), the default chunk size is `64`; otherwise it stays `128`.

### 2. Prepare Per-Chunk Encoder Statistics

`_run_chunked_prepare_phase` launches `flare_chunk_prepare`.

This kernel processes each chunk independently and computes chunk-local online
softmax summaries for the encoder recurrence:

- `chunk_max[chunk, m]`: final running max after the chunk
- `chunk_den[chunk, m]`: final running denominator after the chunk
- `chunk_num[chunk, m, d]`: final running numerator after the chunk

Conceptually, this is "run the encoder recurrence inside each chunk, starting
from an empty state, and keep only the chunk's summary state."

These summaries are enough to compose chunks later using the same online-softmax
merge rule.

### 3. Prefix-Scan the Chunk Summaries

`_run_chunked_prefix_phase` launches `flare_chunk_prefix`.

This phase performs a chunk-level scan over the sequence and converts the
chunk-local summaries into prefix states:

- `prefix_max[chunk, m]`
- `prefix_den[chunk, m]`
- `prefix_num[chunk, m, d]`

Each prefix entry is the encoder state *before* that chunk begins. In other
words, it is the causal carry coming from all earlier chunks.

This lets the final forward replay start each chunk from the correct causal
state without serializing over all `N` tokens.

### 4. Replay Tokens and Emit Outputs

`_run_chunked_output_phase` allocates:

- `O[B, N, H, D]`: outputs
- `L[B, H, N, M]`: per-token/per-row encoder log-normalizers
- `LSE_M[B*H, N]`: per-token decode log-sum-exp over latent rows

It then runs:

1. `flare_chunk_decoder_lse` (unless inline decode-LSE is enabled)
2. `flare_chunk_fwd`

`flare_chunk_decoder_lse` computes:

- `LSE_M[t] = logsumexp_m(score_t(m))`

This is the row-softmax normalizer for the decode stage.

`flare_chunk_fwd` then replays each chunk token-by-token, seeded by
`prefix_max/prefix_den/prefix_num`. For every token:

1. Recompute `score_t(m) = scale * <q_m, k_t>`
2. Update the encoder recurrence (`mu_t`, `d_t`, `u_t`)
3. Form the normalized latent state `z_t = u_t / d_t`
4. Form decode weights `P_t(m) = exp(score_t(m) - LSE_M[t])`
5. Emit `y_t = sum_m P_t(m) * z_t(m, :)`
6. Store `L[t, m] = log(d_t(m)) + mu_t(m)`

`L` is the encoder-side stabilized log-normalizer for every `(t, m)`. It is not
needed for the forward output itself, but it is saved because backward needs the
exact same normalization frame when reconstructing gradients.

### 5. Forward Fast Paths

Two active forward fast paths matter:

- If there is exactly one writer per output element (the common `M=BLOCK_M`,
  `D=BLOCK_D` case), `O` is allocated with `empty` and `flare_chunk_fwd` uses
  `tl.store` instead of `tl.atomic_add`.
- In that same single-writer regime, an experimental env-gated path can compute
  `LSE_M` inside `flare_chunk_fwd` and skip `flare_chunk_decoder_lse`, but this
  is opt-in (`FLARE_FWD_INLINE_LSE_M=1`) rather than the default.

### 6. Saved Tensors

The forward pass saves:

- `Q`, `K`, `V`
- `L`
- `LSE_M`
- `prefix_max`, `prefix_den`, `prefix_num`

Backward reconstructs the necessary local Jacobians from these tensors rather
than storing a full per-token recurrent state.

## Backward Pass

The backward pass is implemented by `_chunked_flare_lse_backward_impl` in
[`chunked.py`](./chunked.py). It deliberately mirrors the forward structure:
first reconstruct the per-token stabilized terms, then run a reverse-time suffix
scan, then contract back to `dQ/dK`.

All major temporary buffers are allocated in `float32`, even when the model runs
in `bf16` or `fp16`. The backward pass repeatedly recomputes exponentials and
composes long prefix/suffix reductions, so the extra precision is used to keep
the replay stable.

### 1. Cast `dO` and Allocate Workspaces

`dO` is first made contiguous and cast to `float32`.

The main workspaces are:

- `p_buf[BH, N, M]`
- `gp_buf[BH, N]`
- `g_buf[BH, N, M]`
- `dS[BH, num_chunks, chunk_size, M]`
- `dV_part[BH, num_chunks, num_m_tiles, chunk_size, D]`

Depending on the shape, the backward path may also allocate:

- `b_local`, `a_local`, `scale_buf` for chunk-local reverse summaries
- `b_in`, `a_in` for the reverse chunk scan
- `vb_part` for multi-`DV`-tile partial reductions
- `dV_bhtd` if `dV` must be reduced across multiple `M` tiles

For the common single-`M`-tile case, `dV_part` can be viewed directly as final
`dV`, so the explicit `dV` reduction buffer is skipped.

### 2. Phase 1: Replay the Forward Encoder State

`flare_chunk_bwd_lse_p_part` is the first major backward kernel.

For each `(batch, head, chunk, M-tile, D-tile)` it replays the forward encoder
recurrence left-to-right inside the chunk, starting from the prefix state saved
by forward.

This reconstructs the normalized latent state `z_t(m, :)` at every token, but
instead of storing `z_t` explicitly, the kernel emits the scalar quantities
needed by the rest of the backward pass:

- `p_t(m) = <z_t(m, :), dO_t>`
- `g_t(m) = exp(score_t(m) - LSE_M[t])`
- `gp_t = sum_m g_t(m) * p_t(m)`

These are the only per-token intermediates the later reverse-time pass needs.

Two details matter:

- If the replay is split over multiple `D` tiles, `p_buf` and `gp_buf` are
  accumulated with atomics; otherwise each program writes directly.
- `g_t` is stored, but the actual encoder attention weight
  `a_t(m) = exp(score_t(m) - L[t, m])` is **not** stored anymore. Later kernels
  reconstruct `a_t` from `g_t`, `LSE_M`, and `L`.

### 3. Phase 2: Build Reverse Chunk Summaries

In the generic path, `flare_chunk_bwd_lse_chunk_summary` walks each chunk
right-to-left and computes the effect of all later tokens in that chunk on
earlier-token gradients.

It produces two suffix carries:

- `b_local`: vector carry over `g_t(m) * dO_t`
- `a_local`: scalar carry over `g_t(m) * p_t(m)`

and one chunk-boundary scale:

- `scale_buf`: how to transport a carry from chunk `k+1` into chunk `k`'s
  normalization frame

These are the chunk-local summaries of the reverse recurrence.

### 4. Phase 2b: Reverse-Scan the Chunk Carries

`flare_chunk_bwd_lse_carry_scan` performs an exclusive reverse scan over chunks.

It turns the local chunk summaries into incoming future carries:

- `b_in[chunk]`: vector carry entering that chunk from all strictly later chunks
- `a_in[chunk]`: scalar carry entering that chunk from all strictly later chunks

This is the reverse-time analogue of the forward prefix scan.

### 5. Phase 3: Replay Backward Within Each Chunk

This phase computes:

- `dV`
- `dS` (the gradient with respect to raw scores)

There are two active variants.

#### Single-`DV` Fast Path

If there is only one `DV` tile, `flare_chunk_bwd_lse_chunk_apply_fused_single_dv`
handles the entire per-chunk reverse replay in one kernel.

For each token, walking right-to-left, it:

1. Loads the incoming suffix carries `b_in` and `a_in`
2. Rescales them across token boundaries using `L`
3. Reconstructs `a_t(m)` from `g_t`, `LSE_M[t]`, and `L[t, m]`
4. Updates the carries with the current token
5. Writes:
   - `dV_part[t, :] = sum_m a_t(m) * b_carry_t(m, :)`
   - `dS[t, m]`

The `dS` decomposition is:

- `ds_g`: gradient through the decode softmax over latent rows
- `ds_z`: gradient through the normalized encoder state update

and the stored result is `dS = ds_g + ds_z`.

This is the common fast path for the target `D=32` regime.

#### Multi-`DV` Generic Path

If the backward state is tiled over `D`, the work is split:

- `flare_chunk_bwd_lse_chunk_apply_part`:
  - replays the vector carry per `DV` tile
  - writes partial `dV`
  - writes `vb_part = <b_carry_t(m), v_t>` per `DV` tile
- `flare_chunk_bwd_lse_chunk_apply_finalize`:
  - recomputes the scalar carry
  - sums `vb_part` across `DV` tiles
  - reconstructs `a_t`
  - writes final `dS`

This split exists because `dS` needs the sum over all `DV` tiles, while each
program in the partial kernel only owns one `DV` slice.

### 6. Experimental Fully Fused Reverse Scan

There is one additional opt-in path:

- `flare_chunk_bwd_lse_scan_apply_fused_single_dv`

If `FLARE_LSE_BWD_FUSE_CHUNK_SCAN=1` and the shape is single-`DV`, it fuses:

- chunk summary
- reverse chunk scan
- per-chunk apply

into one kernel that walks all chunks in reverse while keeping the future carry
in registers.

This is wired into the code, but it is not the default because it can lose badly
on occupancy-sensitive shapes.

### 7. Reduce or Bypass `dV`

After Phase 3:

- if there are multiple `M` tiles, `flare_chunk_bwd_lse_dv_reduce` sums
  `dV_part` across the `M`-tile axis
- if there is only one `M` tile, `dV_part` is reshaped directly and the reduce
  kernel is skipped

This is a pure reduction step; all FLARE-specific algebra is already done.

### 8. Phase 4: Contract `dS` Back to `dQ` and `dK`

Once `dS` is available, the problem is just backprop through:

`score = scale * (Q @ K^T)`.

`flare_chunk_bwd_recurrent_qk` therefore performs the standard contractions:

- `dK_chunk = dS_chunk @ Q * scale`
- `dQ += dS_chunk^T @ K_chunk * scale`

`dQ` uses atomic adds because all chunks contribute to the same `[H, M, D]`
tensor. `dK` is written directly because each chunk owns a disjoint token slice.

Finally, `dV` is reshaped back to `[B, N, H, D]`, and `dQ`, `dK`, `dV` are cast
back to the input dtypes.

## What the Buffers Mean

The most important saved/replayed quantities are:

- `L[t, m]`:
  encoder-side stabilized log-normalizer after token `t`
- `LSE_M[t]`:
  decode-side log-normalizer across latent rows for token `t`
- `g_t(m) = exp(score_t(m) - LSE_M[t])`:
  decode softmax weight
- `a_t(m) = exp(score_t(m) - L[t, m])`:
  encoder attention weight for token `t` under latent row `m`
- `p_t(m) = <z_t(m), dO_t>`:
  backward projection of the output gradient through the normalized latent state
- `gp_t = sum_m g_t(m) * p_t(m)`:
  reduction needed for the decode-softmax Jacobian
- `dS[t, m]`:
  gradient with respect to the raw score `score_t(m)`

If you keep these definitions straight, the forward and backward kernels read as
two replays of the same stabilized recurrence: one in forward to produce outputs,
and one in backward to reconstruct local Jacobians without storing the full
per-token latent state.

## Profiling Support

If `ChunkedFLARE.apply(..., profile=True)` is used, the implementation returns a
timings dictionary with per-kernel forward timings immediately and fills the
backward timings in-place when autograd runs.

The forward and backward names in that timings dict match the actual Triton
kernel names used in the dispatcher, so the profiling output is a direct map of
the pipeline described above.

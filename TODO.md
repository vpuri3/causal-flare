# TODO

- [ ] Restore heuristic/offline tuning workflow from `72394e2d2a109be2d8464147c17f10b28f423fcc` so normal user execution no longer pays runtime Triton autotune costs, with:
  - [x] inventory current tuning ownership versus `72394e2` across `causal_flare/chunked.py`, `causal_flare/inference.py`, and the benchmark workflow docs; write down which knobs must move back out of runtime autotune.
  - [x] restore heuristic config selection as the default runtime path for chunked training and inference prefill/decode, using narrow promoted bucket rules instead of runtime search.
    - [x] inference decode: replace runtime decode-step autotune with one heuristic-selected config plus env overrides.
    - [x] chunked training forward/backward: restore promoted bucket tables and explicit launch selection for prepare/prefix/decoder/fwd and backward replay/reduction kernels.
  - [x] demote Triton autotune to explicit developer/benchmark mode rather than the default user path.
  - [x] update `benchmark/chunked_flare_matrix_workflow.md`, `benchmark/flare_inference_matrix_workflow.md`, and the corresponding tuning runners/catalogs so offline matrix sweeps again own the promoted knobs.
  - [x] add a reproducible cold-cache startup benchmark for representative anchor shapes and use it as a regression guardrail.
  - [ ] rerun targeted correctness/regression coverage plus cold/warm startup measurements and compare against the rollback success criteria.

- [ ] High priority: write a Block-Causal FLARE implementation using the chunkwise training algorithm, with:
  - contiguous token blocks of size `B`, where tokens in block `k` can attend to all tokens in blocks `<= k`, i.e. if token `t` is in block `k`, its receptive field is tokens `1:(B*k)`.
  - an effective attention mask that keeps all within-block interactions and all interactions with preceding blocks, while zeroing only the strictly upper block-triangular region.
  - an implementation strategy that exploits the existing chunkwise training algorithm rather than introducing a separate dense-style training path.
  - an explicit requirement that block size `B` be a multiple of chunk size `C`, so the block-causal mask lines up cleanly with chunk boundaries and training-time summaries.
  - an initial supported chunk-size set of `C in {16, 32, 64, 128}`, with any additional chunk sizes treated as follow-up work rather than silently supported.
  - kernel/API validation that rejects invalid `(B, C)` combinations early with precise errors instead of allowing misaligned masking semantics.
  - a torch prototype in `causal_flare/block_causal.py` that mirrors the `ChunkedFLARE` forward structure before any CUDA/Triton kernel work.
  - targeted correctness tests against a reference block-causal attention mask, including cases that exercise multiple blocks per chunk and multiple chunks per sequence.
  - a performance check to confirm that the chunkwise formulation actually gives the intended efficiency advantage over a naive masked implementation for representative `(B, C)` settings.

- [ ] High priority: investigate using Triton's `associative_scan` for forward computations in the recurrent and chunked implementations, with:
  - an explicit audit of which forward recurrences/prefix-style reductions in `RecurrentFLARE` and `ChunkedFLARE` can be expressed cleanly as an associative combine without contorting the kernel structure or weakening numerics.
  - a prototype path using `triton.language.associative_scan` for the forward computation where it fits naturally, plus a direct comparison against the current implementation for correctness, numerical stability, and performance.
  - attention to whether this can simplify launch structure, reduce manual scan logic, or improve maintainability, rather than assuming it is automatically faster.
  - a follow-up backward-pass investigation: determine whether the same associative-scan formulation can be applied in `bwd`, and if not, document the exact dependency or reduction pattern that blocks it.

- [ ] Try a new multi-kernel `RecurrentFLARE` implementation by writing new `fwd_impl` / `bwd_impl` methods and wiring them into `RecurrentFLARE` behind env-var selection, with:
  - steps 1/2: precompute `LSE_ENC` and `LSE_DEC`. `LSE_ENC` may also be usable as a score-saving buffer. Computing `LSE_ENC` will likely require a token loop.
  - step 3: use dense matrix math to execute the forward pass per block, tiling over `N`, `M`, and `D` for efficiency.
  - willingness to store a `[BLOCK_M, BLOCK_T, BLOCK_T]` matrix if needed, since this path may be able to use tensor cores efficiently enough to justify it.
  - a matching backward pass for the new multi-kernel path.
  - an explicit competitiveness check against the current recurrent approach; if this path is competitive, consider whether the same method should later be applied to chunked.

- [ ] Refactor the FLARE prefill path to remove dedicated prefill-only entry points and use one canonical forward path, with:
  - masking support
  - optional return of final recurrent cache state `(m, d, u)` in addition to token outputs.

- [ ] Support different numbers of query heads and KV heads (`H_q != H_kv`) for grouped-query / multi-query style layouts, with:
  - explicit public shape conventions instead of one shared `H`: `Q:[H_q, M, D]`, `K/V:[B, N, H_kv, D]`, output `Y:[B, N, H_q, D]`, and recurrent state/prefix summaries indexed by query head count because the encoder statistics remain query-head-specific even when KV heads are shared.
  - well-defined head-routing semantics, including the fundamental constraint that each query head must map to a KV head. If the intended first implementation only supports `H_q % H_kv == 0`, make that divisibility requirement explicit and centralize the `q_head -> kv_head` mapping logic instead of duplicating ad hoc integer division in kernels and Python wrappers.
  - canonicalization and validation updates across prefill/decode paths so we stop assuming `Q`, `K`, and `V` all expose the same head axis length, while still rejecting ambiguous or unsupported head-group layouts early with precise error messages.
  - a full audit of chunked, recurrent, dense, inference, and torch-reference paths so score computation loads the correct shared KV head, but running maxima, denominators, numerators, latent states, and outputs continue to live on the query-head axis.
  - Triton/kernel indexing work to remove hidden `H_q == H_kv` assumptions from launch grids, stride calculations, temporary buffers, prefix workspaces, and backward reductions. The important detail is to preserve correct accumulation when multiple query heads read the same KV head without accidentally aliasing state or gradients.
  - backward-pass design for grouped KV sharing, especially around `dK`/`dV` accumulation. Multiple query heads contributing to one KV head means the reduction strategy needs to be explicit and tested rather than relying on current one-head-to-one-head assumptions.
  - targeted correctness tests for representative MQA/GQA shapes (for example `H_q=16, H_kv=4` and `H_q=16, H_kv=1`), plus benchmark coverage so we can quantify whether shared-KV support changes memory traffic, occupancy, or numerical behavior relative to the equal-head baseline.
  - docs/CLI/benchmark plumbing cleanup so argument names and reporting distinguish `num_query_heads` from `num_kv_heads`, and so remaining unsupported combinations are documented as implementation limits rather than implicit shape accidents.

- [ ] Remove the `ChunkedFLARE` backward `a_buf` workspace again by recomputing `a_t` on the fly from stable quantities
  (not from underflowed `g_t`), with minimal extra compute and without reintroducing sharp-softmax regressions.

- [x] Investigate whether chunked FLARE still needs explicit `prefix_max` / `prefix_den` now that forward saves `LSE_ENC`, with:
  - a derivation of exactly which downstream uses of `prefix_max` and `prefix_den` are already covered by `LSE_ENC = log(sum(exp(...)))` and which, if any, still require the separate max/den split for numerical or kernel-structure reasons.
  - a clear decision on `prefix_num`: whether the numerator summary still carries irreducible information needed for replay/output/backward, or whether some/all of its current uses can also be rewritten in terms of `LSE_ENC` plus other already-saved state.
  - an implementation audit across chunked forward, backward, inference, and docs so we can delete redundant prefix summaries rather than carrying both representations out of inertia.
  - attention to memory-bandwidth and workspace tradeoffs, since removing redundant prefix state may matter as much as the algebraic simplification.

- [ ] Low-priority investigation: can backward avoid materializing score-gradient buffers (`dS_enc`, `dS_dec`) and contract directly into `dQ` / `dK` (and decode-side `dQ_dec` / `dK_dec`) instead, with:
  - a clear baseline inventory of where score-space gradients are currently written and reread in the recurrent/chunked backward paths, including the separate “produce `dS` first, then contract into Q/K” structure in the current chunked implementation.
  - an explicit fused-backward design sketch for both encoder and decode branches: once the local softmax derivative terms are known, accumulate their contribution into `dQ`/`dK` immediately instead of storing full `[... , T, M]` score-gradient tiles to HBM.
  - careful accounting for the real tradeoff, because the obvious upside is lower VRAM and less score-buffer traffic, but the downside may be more recompute of Q/K tiles, worse tensor-core utilization, extra atomics/reductions into shared `dQ`/`dK`, or loss of phase separation that currently keeps the kernels simpler and numerically easier to reason about.
  - a concrete decision on whether the fused path should target only encode-side score grads first, or also decode-side `dS_dec`; the decode branch may have a different cost/benefit profile when weights are shared versus unshared.
  - special attention to gradient accumulation hazards in the chunked case, where multiple token tiles and/or multiple D tiles may contribute to the same `dQ` or `dK` slice. If direct contraction requires atomics, staged partial buffers, or a different launch decomposition, that should be part of the experiment rather than hidden as an implementation detail.
  - measurement criteria that treat this as an optimization experiment, not an automatic improvement: compare peak memory, achieved bandwidth, kernel time, end-to-end backward time, and numerical agreement against the current buffered implementation before deciding whether the extra complexity is justified.
  - a preference to keep this exploratory for now unless the measurements are compelling; this looks worth testing because it could reduce VRAM materially, but it should stay behind higher-priority correctness and feature work until there is data.

- [ ] Low-priority: support internal padding / ragged final chunks in `ChunkedFLARE` so callers no longer need `N` to be an exact multiple of `CHUNK_SIZE`, with:
  - a clear public contract that the chunked path may pad internally up to `ceil_div(N, CHUNK_SIZE) * CHUNK_SIZE`, but still returns outputs and gradients only for the original `N` tokens.
  - a forward-path audit covering prepare, prefix, decoder-LSE, and replay kernels so padded tokens are truly inert: no score contribution, no recurrent-state updates, no decode-normalizer contamination, and no accidental writes to visible outputs.
  - a backward-path audit covering all chunk-local workspaces (`dS_*`, `dV_part`, carry buffers, replay state, etc.) so padded tail positions likewise contribute exactly zero gradient and do not perturb prefix scans, reductions, or final reshapes back to length `N`.
  - a decision on whether to implement this as explicit pre-padding at the Python boundary or as stricter masked handling inside the existing kernels. The main goal is to remove the external divisibility requirement without exploding code complexity or paying unnecessary extra memory traffic on already-aligned shapes.
  - attention to the numerics of masked/padded positions, especially around `-inf` score handling, log-sum-exp paths, and recurrent max/denominator updates. The padded region must behave like nonexistent tokens, not just like zero-valued tokens.
  - targeted tests for non-divisible sequence lengths in both forward and backward paths (for example `N=1000, CHUNK_SIZE=128` and very short tails such as `N=129, CHUNK_SIZE=128`), including parity against the reference implementation and checks that padded tails do not change results for the valid prefix.
  - a small performance check before enabling this broadly, since internal padding can increase work by up to almost one extra chunk for short tails. This looks worth supporting for ergonomics and integration simplicity, but it should remain low priority until more pressing correctness and feature work lands.

- [ ] Add a PyTorch chunked backward implementation so tests can use a matching chunked backward noise model
  (instead of autograd as a stopgap) when scaling tolerances relative to a PyTorch baseline.

- [ ] Continue updating FLARE tests to mirror FlashAttention-style validation:
  - use a strict FP32 causal FLARE reference as the oracle
  - use a matching PyTorch implementation as the noise model with explicit scoped matmul precision controls
  - scale forward/backward tolerances relative to the PyTorch noise model instead of relying on fixed hardcoded thresholds.

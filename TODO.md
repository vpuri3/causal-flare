# TODO

- [ ] Refactor the FLARE prefill path to remove dedicated prefill-only entry points and use one canonical forward path, with:
  - masking support
  - optional return of final recurrent cache state `(m, d, u)` in addition to token outputs.

- [ ] Build publication-quality FLARE vs Transformer systems benchmarks and plots for:
  - decode speed and memory
  - prefill speed and memory
  - training speed and memory
  using standardized experiment settings and reproducible scripts.

- [ ] Remove the `ChunkedFLARE` backward `a_buf` workspace again by recomputing `a_t` on the fly from stable quantities
  (not from underflowed `g_t`), with minimal extra compute and without reintroducing sharp-softmax regressions.

- [ ] Add a PyTorch chunked backward implementation so tests can use a matching chunked backward noise model
  (instead of autograd as a stopgap) when scaling tolerances relative to a PyTorch baseline.

- [ ] Continue updating FLARE tests to mirror FlashAttention-style validation:
  - use a strict FP32 causal FLARE reference as the oracle
  - use a matching PyTorch implementation as the noise model with explicit scoped matmul precision controls
  - scale forward/backward tolerances relative to the PyTorch noise model instead of relying on fixed hardcoded thresholds.

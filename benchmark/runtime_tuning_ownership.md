# Runtime Tuning Ownership

This note captures the tuning-ownership rollback after the runtime-autotune detour.

Reference point:
- oracle workflow and runtime policy from commit `72394e2d2a109be2d8464147c17f10b28f423fcc`

## Desired Ownership Model

- Benchmark matrix runners own launch/tile exploration.
- Runtime code consumes promoted heuristic buckets.
- Default runtime does not use Triton autotune search.

Why:

- we tried Triton autotune on the default runtime path
- cold-cache compile and first-use latency became extremely large
- the user experience was materially worse than the older heuristic-plus-offline-tuning workflow

## Current Status vs `72394e2`

### Chunked training runtime

`72394e2` default behavior:
- heuristic bucket selection in `causal_flare/autoregressive/training.py`
- explicit width buckets via `_get_chunked_forward_bucket_defaults(...)`
- explicit backward buckets via `_get_chunked_backward_bucket_defaults(...)`
- per-phase launch defaults resolved by `_resolve_forward_launch(...)` / `_resolve_backward_launch(...)`
- runtime env overrides remained possible, but no `@triton.autotune` search loop on the user path

Current default behavior:
- heuristic structural selection in `_get_chunked_forward_config(...)` and `_get_chunked_backward_bucket_defaults(...)`
- one explicit promoted config path per specialization on the default user path
- no `@triton.autotune` search loop on the user path

Current chunked runtime-owned knobs that should remain heuristic/promoted:
- `CHUNK_SIZE`
- `BLOCK_M`
- forward width-bucket defaults for `BLOCK_D` / `BLOCK_K`
- backward width-bucket defaults for replay/QK/state tiles
- forward/backward launch choices by bucket

Chunked knobs that remain benchmark-owned instead of runtime-searched:
- prepare kernel local tile alternatives
- prefix kernel local tile alternatives
- decoder-LSE local tile alternatives
- forward replay kernel local tile alternatives
- backward partial-replay local tile alternatives
- backward gp-reduce local tile alternatives

### Inference runtime

`72394e2` default behavior:
- prefill reused chunked heuristic config selection
- decode used one explicit heuristic-selected launch family, not a runtime autotune sweep

Current default behavior:
- prefill reuses the chunked heuristic runtime path
- decode uses one explicit heuristic-selected launch family plus env overrides

Current inference knobs that should remain heuristic/promoted:
- prefill `CHUNK_SIZE`
- prefill `BLOCK_M`
- decode `BLOCK_M`, `BLOCK_D`, `BLOCK_K`, and launch geometry selected from promoted buckets

Inference knobs that remain benchmark-owned instead of runtime-searched:
- decode-step local config alternatives in `flare_recurrent_step_kernel`

## Implications

To stay aligned with the `72394e2` workflow:
- keep heuristic-only default runtime dispatch for chunked and inference decode
- keep offline tuning scripts as the place where candidate families are explored
- promote measured winners back into the runtime bucket tables
- do not reintroduce runtime Triton autotune search into the default user path

## Measurable Acceptance Criteria

- Default cold-cache first call compiles one chosen config path per specialization, not multiple autotune candidates.
- No default user-path kernel enters Triton autotune benchmarking loops.
- Cold-cache startup on anchor shapes stays bounded by compile time for the chosen heuristic path rather than exploding due to runtime search.

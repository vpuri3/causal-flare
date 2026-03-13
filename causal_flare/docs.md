# FLARE Branches

This repository is organized around two prediction regimes.

## `autoregressive`

Standard next-token-prediction FLARE for causal language modeling.

Primary entry points:

- [`causal_flare.autoregressive`](./autoregressive/__init__.py)
- [`causal_flare.autoregressive.training`](./autoregressive/training.py)
- [`causal_flare.autoregressive.inference`](./autoregressive/inference.py)

Public wrappers:

- `flare_autoregressive_triton`
- `flare_decode_triton`
- `flare_prefill_triton`

Experimental implementations live under:

- [`causal_flare.autoregressive.experimental`](./autoregressive/experimental.py)

See [`causal_flare/autoregressive/docs.md`](./autoregressive/docs.md).

## `semi_autoregressive`

Next token-block prediction FLARE for block-causal / semi-autoregressive settings
such as diffusion language modeling, video streaming or synthesis, and PDE time
series modeling.

Primary entry points:

- [`causal_flare.semi_autoregressive`](./semi_autoregressive/__init__.py)
- [`causal_flare.semi_autoregressive.training`](./semi_autoregressive/training.py)
- [`causal_flare.semi_autoregressive.inference`](./semi_autoregressive/inference.py)

Public wrapper:

- `flare_semi_autoregressive_trition`
- `flare_semi_autoregressive_prefill_trition`
- `flare_semi_autoregressive_decode_trition`

See [`causal_flare/semi_autoregressive/docs.md`](./semi_autoregressive/docs.md).

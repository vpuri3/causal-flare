# AutoRegressiveFLARE Notes

This document describes the current autoregressive branch of the repository.

Main implementation modules:

- [`training.py`](./training.py): production next-token-prediction training path
- [`inference.py`](./inference.py): prefill and decode wrappers
- [`experimental.py`](./experimental.py): opt-in exports for dense and recurrent variants

## Scope

`AutoRegressiveFLARE` is the default production FLARE path for standard causal
language modeling. It uses latent-query recurrent softmax algebra with chunked
training and cached inference.

Public tensor layout:

- `Q`: `[H, M, D_k]`
- `K`: `[B, N, H, D_k]`
- `V`: `[B, N, H, D_v]`
- output `Y`: `[B, N, H, D_v]`

Public wrappers:

- `flare_autoregressive_triton`
- `flare_decode_triton`
- `flare_prefill_triton`

## Production vs Experimental

Production:

- chunked training / full-sequence autoregressive forward
- cached prefill / decode inference

Experimental:

- dense FLARE
- recurrent FLARE

The dense and recurrent implementations are intentionally not exported from the
default package root. They remain available through
[`experimental.py`](./experimental.py).

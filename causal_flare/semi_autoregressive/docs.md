# SemiAutoRegressiveFLARE Notes

This document describes the semi-autoregressive branch of the repository.

Main implementation module:

- [`training.py`](./training.py)
- [`inference.py`](./inference.py)
- [`reference.py`](./reference.py)

## Scope

`SemiAutoRegressiveFLARE` is the block-causal next token-block prediction path.
It targets semi-autoregressive or autoregressive settings where a query token is
allowed to attend to all tokens in its own block and all preceding blocks.

Representative applications:

- diffusion language modeling
- video streaming / synthesis
- PDE time series modeling
- other token-block prediction workloads

Public wrapper:

- `flare_semi_autoregressive_trition`
- `flare_semi_autoregressive_prefill_trition`
- `flare_semi_autoregressive_decode_trition`

## Current State

The public semi-autoregressive training, prefill, and decode wrappers are
currently explicit stubs that raise `NotImplementedError`.

The current executable torch/reference implementation is kept in:

- [`reference.py`](./reference.py)

The key validation rules are:

- `block_size` must be explicit
- `chunk_size` must be explicit
- `chunk_size` must be one of `16, 32, 64, 128`
- `block_size` must be a multiple of `chunk_size`
- sequence length must be block-aligned

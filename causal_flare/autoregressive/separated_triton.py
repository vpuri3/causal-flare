from __future__ import annotations

import torch


def _raise_separated_triton_not_implemented() -> None:
    raise NotImplementedError(
        "flare_autoregressive_separated_triton is intentionally disabled. "
        "The previous implementation was removed because it was not a clean upstream-style Triton SSD port. "
        "Use flare_autoregressive_separated_mamba_style_pytorch for correctness, or continue the fresh port in "
        "separated_mamba_style_triton.py."
    )


class _SeparatedAutoregressiveTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, retain, write, decode_weights, chunk_size):
        _raise_separated_triton_not_implemented()

    @staticmethod
    def backward(ctx, grad_out):
        _raise_separated_triton_not_implemented()


def flare_autoregressive_separated_hybrid(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    _raise_separated_triton_not_implemented()


def flare_autoregressive_separated_trition(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    _raise_separated_triton_not_implemented()


def flare_autoregressive_separated_triton(
    *,
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode_weights: torch.Tensor,
    chunk_size: int | None = None,
) -> torch.Tensor:
    _raise_separated_triton_not_implemented()


__all__ = [
    "flare_autoregressive_separated_hybrid",
    "flare_autoregressive_separated_trition",
    "flare_autoregressive_separated_triton",
]

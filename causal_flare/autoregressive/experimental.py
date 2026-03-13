"""Experimental autoregressive FLARE variants.

Dense and recurrent paths stay available for development work, but they are not
part of the default exported API.
"""

from .dense import DenseFLARE, DenseFLARE1, flare_causal_pytorch_dense, flare_causal_pytorch_dense1
from .recurrent import RecurrentFLARE, flare_recurrent_triton

__all__ = [
    "DenseFLARE",
    "DenseFLARE1",
    "RecurrentFLARE",
    "flare_causal_pytorch_dense",
    "flare_causal_pytorch_dense1",
    "flare_recurrent_triton",
]

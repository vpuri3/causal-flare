from . import autoregressive, semi_autoregressive
from .autoregressive.training import flare_autoregressive_triton
from .autoregressive.inference import flare_decode_triton, flare_prefill_triton
from .autoregressive.recurrent import flare_recurrent_triton
from .semi_autoregressive.training import flare_semi_autoregressive_trition
from .semi_autoregressive.inference import (
    flare_semi_autoregressive_decode_trition,
    flare_semi_autoregressive_prefill_trition,
)

__version__ = "0.0.1"

__all__ = [
    "autoregressive",
    "semi_autoregressive",
    "flare_autoregressive_triton",
    "flare_semi_autoregressive_decode_trition",
    "flare_semi_autoregressive_prefill_trition",
    "flare_semi_autoregressive_trition",
    "__version__",
]

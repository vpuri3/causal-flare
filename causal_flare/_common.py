#
# Causal FLARE Implementation with Triton Fused Kernel
# Algorithmic discussion and design notes live in:
#   causal_flare/docs.md
#

import math
import os
import time
import warnings
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torch.autograd as autograd

import triton
import triton.language as tl
import triton.testing

from flash_attn import flash_attn_func

# Backward profiling (set by main for a single run)
_BWD_PROFILE_MODE = None  # "triton3"
_BWD_PROFILE_TIMINGS = {}
_DEBUG_PREFIX_STATS = {}
_TF32_ENV_DEPRECATION_WARNED = False
_RECURRENT_CUDA_EXT = None
_RECURRENT_CUDA_EXT_ERR = None


def _set_bwd_profile_mode(mode):
    global _BWD_PROFILE_MODE
    _BWD_PROFILE_MODE = mode
    if mode is not None:
        _BWD_PROFILE_TIMINGS[mode] = {}


def _bwd_profile_enabled() -> bool:
    return _BWD_PROFILE_MODE is not None


def _record_bwd_timing(key, ms):
    if _BWD_PROFILE_MODE is None:
        return
    _BWD_PROFILE_TIMINGS.setdefault(_BWD_PROFILE_MODE, {})[key] = ms

def _accumulate_timing(bucket: dict[str, float] | None, key: str, ms: float | None) -> None:
    if bucket is None or ms is None:
        return
    bucket[key] = bucket.get(key, 0.0) + float(ms)

def _measure_op_ms(device: torch.device, enabled: bool, op):
    if not enabled:
        return op(), None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = op()
        end.record()
        torch.cuda.synchronize(device)
        return result, float(start.elapsed_time(end))
    start = time.perf_counter()
    result = op()
    return result, float((time.perf_counter() - start) * 1e3)

def _time_cuda(op):
    if _BWD_PROFILE_MODE is None or not torch.cuda.is_available():
        op()
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    op()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)

@contextmanager
def _temp_env_var(key: str, value: str):
    prev = os.environ.get(key, None)
    os.environ[key] = value
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prev

_causal_mask_cache: dict[tuple[int, torch.device], torch.Tensor] = {}

def _get_causal_mask(C: int, device: torch.device) -> torch.Tensor:
    key = (C, device)
    mask = _causal_mask_cache.get(key)
    if mask is None:
        mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool))
        _causal_mask_cache[key] = mask
    return mask

def _get_eps_for_dtype(dtype: torch.dtype) -> float:
    env_eps = os.environ.get("FLARE_EPS", "")
    if env_eps:
        try:
            return float(env_eps)
        except ValueError:
            pass
    if dtype == torch.float32:
        return 1e-6
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-4
    return 1e-6


def _get_bwd_s_storage_dtype(input_dtype: torch.dtype) -> torch.dtype:
    """Storage dtype for S buffer in backward recompute/store path."""
    mode = os.environ.get("FLARE_BWD_S_DTYPE", "fp32").strip().lower()
    if mode in ("fp32", "float32"):
        return torch.float32
    if mode in ("bf16", "bfloat16"):
        return torch.bfloat16
    if mode in ("fp16", "float16"):
        return torch.float16
    # auto: legacy compact-storage policy.
    if input_dtype == torch.bfloat16:
        return torch.bfloat16
    if input_dtype == torch.float16:
        return torch.float16
    return torch.float32


def _get_allclose_tols(dtype: torch.dtype) -> tuple[float, float]:
    if dtype == torch.float32:
        return 1e-5, 1e-5
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2, 1e-2
    return 1e-5, 1e-5

def _rel_l2_err(a: torch.Tensor, b: torch.Tensor) -> float:
    num = torch.linalg.norm((a - b).float())
    den = torch.linalg.norm(b.float())
    return (num / (den + 1e-12)).item()

def _max_rel_err(a: torch.Tensor, b: torch.Tensor, atol: float) -> float:
    denom = b.abs().float() + atol
    return ((a - b).abs().float() / denom).amax().item()

def _max_abs_idx(delta: torch.Tensor) -> tuple[int, ...]:
    flat_idx = torch.argmax(delta.reshape(-1))
    return tuple(int(x) for x in torch.unravel_index(flat_idx, delta.shape))

def _tensors_all_finite(*tensors: torch.Tensor) -> bool:
    for t in tensors:
        if t is None:
            continue
        if not torch.isfinite(t).all():
            return False
    return True

def _per_axis_max(delta: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "per_b": delta.amax(dim=(1, 2, 3)),
        "per_t": delta.amax(dim=(0, 2, 3)),
        "per_h": delta.amax(dim=(0, 1, 3)),
        "per_d": delta.amax(dim=(0, 1, 2)),
    }

def _check_finite(name: str, tensor: torch.Tensor) -> None:
    if torch.isfinite(tensor).all():
        return
    with torch.no_grad():
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        t = tensor.float()
        t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
        t_min = t.min().item() if t.numel() > 0 else 0.0
        t_max = t.max().item() if t.numel() > 0 else 0.0
    print(
        f"[FLARE DEBUG] {name} has NaN/Inf "
        f"(nan={nan_count}, inf={inf_count}), "
        f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
        f"min={t_min:.6g}, max={t_max:.6g}",
        flush=True,
    )
    raise RuntimeError(f"FLARE numerical issue in {name}")

def _check_finite_allow_neg_inf(name: str, tensor: torch.Tensor) -> None:
    if torch.isnan(tensor).any():
        _check_finite(name, tensor)
    if torch.isposinf(tensor).any():
        _check_finite(name, tensor)

def _get_exp_clamp_for_dtype(dtype: torch.dtype) -> float:
    if dtype == torch.float32:
        return 80.0
    if dtype in (torch.float16, torch.bfloat16):
        return 40.0
    return 80.0


def _resolve_attn_scale(scale, head_dim: int) -> float:
    if scale is not None:
        return float(scale)
    return 1.0 if head_dim <= 8 else head_dim ** -0.5

def _get_allow_tf32() -> bool:
    """Deprecated compatibility helper for FLARE_ALLOW_TF32."""
    global _TF32_ENV_DEPRECATION_WARNED
    env = os.environ.get("FLARE_ALLOW_TF32", "")
    if env:
        if not _TF32_ENV_DEPRECATION_WARNED:
            warnings.warn(
                "FLARE_ALLOW_TF32 is deprecated; use FLARE_INPUT_PRECISION "
                "with one of {'ieee','tf32','tf32x3'}.",
                stacklevel=2,
            )
            _TF32_ENV_DEPRECATION_WARNED = True
        return env != "0"
    return False


def _get_input_precision() -> str:
    """Preferred precision selector for ChunkedFLARE Triton dots."""
    env = os.environ.get("FLARE_INPUT_PRECISION", "").strip().lower()
    if env in ("tf32", "tf32x3", "ieee"):
        return env
    # Backward-compatible knob.
    legacy = os.environ.get("FLARE_ALLOW_TF32", "")
    if legacy:
        return "tf32" if _get_allow_tf32() else "ieee"
    # Default mode: tf32x3 gives IEEE-like accuracy with higher throughput in current suite.
    return "tf32x3"


def _recurrent_cuda_forward(Q: torch.Tensor, K_bhtd: torch.Tensor, V_bhtd: torch.Tensor, scale: float) -> torch.Tensor:
    global _RECURRENT_CUDA_EXT, _RECURRENT_CUDA_EXT_ERR
    if _RECURRENT_CUDA_EXT is None and _RECURRENT_CUDA_EXT_ERR is None:
        try:
            from .cuda_ext import recurrent_flare_cuda as _rec_cuda_ext
            _RECURRENT_CUDA_EXT = _rec_cuda_ext
        except Exception as exc:  # pragma: no cover - optional fast path.
            try:
                # Compatibility fallback when this package is vendored under FLA.
                from fla.models.flare.cuda_ext import recurrent_flare_cuda as _rec_cuda_ext  # type: ignore
                _RECURRENT_CUDA_EXT = _rec_cuda_ext
            except Exception:
                _RECURRENT_CUDA_EXT_ERR = exc
    if _RECURRENT_CUDA_EXT is None:
        raise RuntimeError(f"Recurrent CUDA extension unavailable: {_RECURRENT_CUDA_EXT_ERR}")
    return _RECURRENT_CUDA_EXT.forward(Q, K_bhtd, V_bhtd, scale)


def _normalize_input_precision(input_precision, allow_tf32):
    """Resolve precision mode, preferring input_precision over deprecated allow_tf32."""
    if input_precision is not None:
        if isinstance(input_precision, bool):
            mode = "tf32" if input_precision else "ieee"
        else:
            mode = str(input_precision).strip().lower()
    elif allow_tf32 is not None:
        mode = "tf32" if bool(allow_tf32) else "ieee"
    else:
        mode = _get_input_precision()
    if mode not in ("tf32", "tf32x3", "ieee"):
        raise ValueError(
            f"Unsupported input_precision={input_precision!r}. "
            "Expected one of {'tf32','tf32x3','ieee'}."
        )
    return mode

def _get_bwd_launch_config(
    M: int,
    N: int,
    D: int,
    chunk_size: int,
    phase: str = "phase1",
) -> tuple[int, int, int]:
    """Return (num_warps, num_stages, block_t) for backward kernels.

    Phase-specific env overrides:
      - FLARE_BWD_{PHASE}_NUM_WARPS / NUM_STAGES / BLOCK_T
    Global env overrides (fallback):
      - FLARE_BWD_NUM_WARPS / NUM_STAGES / BLOCK_T
    """
    phase_key = phase.upper()

    env_warps = (
        os.environ.get(f"FLARE_BWD_{phase_key}_NUM_WARPS", "")
        or os.environ.get("FLARE_BWD_NUM_WARPS", "")
    )
    env_stages = (
        os.environ.get(f"FLARE_BWD_{phase_key}_NUM_STAGES", "")
        or os.environ.get("FLARE_BWD_NUM_STAGES", "")
    )
    env_block_t = (
        os.environ.get(f"FLARE_BWD_{phase_key}_BLOCK_T", "")
        or os.environ.get("FLARE_BWD_BLOCK_T", "")
    )
    if env_warps and env_stages and env_block_t:
        return int(env_warps), int(env_stages), int(env_block_t)

    hopper = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9

    # Explicit buckets for production ranges:
    # D: 16/32/64, M: 16..512, N: 2k..65k
    if D <= 16:
        d_bucket = 16
    elif D <= 32:
        d_bucket = 32
    elif D <= 64:
        d_bucket = 64
    else:
        d_bucket = 128

    if M <= 16:
        m_bucket = 16
    elif M <= 32:
        m_bucket = 32
    elif M <= 64:
        m_bucket = 64
    elif M <= 128:
        m_bucket = 128
    elif M <= 256:
        m_bucket = 256
    elif M <= 512:
        m_bucket = 512
    else:
        m_bucket = 1024

    if N <= 2048:
        n_bucket = 2048
    elif N <= 8192:
        n_bucket = 8192
    elif N <= 32768:
        n_bucket = 32768
    elif N <= 65536:
        n_bucket = 65536
    else:
        n_bucket = 131072

    if phase == "phase1":
        # Hot kernel: shape-aware policy for production ranges.
        if n_bucket >= 32768:
            num_warps = 4 if d_bucket <= 32 else 8
            num_stages = 1
        elif d_bucket <= 32 and m_bucket >= 128 and n_bucket >= 8192:
            # Long-context + wide latent axis is register-heavy in phase-1.
            # Fewer warps improves effective occupancy on Hopper.
            num_warps = 4
            num_stages = 1
        elif d_bucket <= 32 and m_bucket <= 64:
            num_warps = 8 if hopper else 4
            num_stages = 2 if chunk_size <= 64 else 1
        elif d_bucket <= 64 and m_bucket <= 128:
            num_warps = 8
            num_stages = 1
        else:
            num_warps = 8
            num_stages = 1
    elif phase == "phase23":
        # Prefix/stats backward: lighter math, prefer conservative launch.
        if n_bucket >= 32768:
            num_warps = 4 if d_bucket <= 32 else 8
            num_stages = 1
        else:
            num_warps = 8 if (d_bucket >= 64 or m_bucket >= 256) else 4
            num_stages = 2 if (n_bucket <= 2048 and m_bucket <= 64) else 1
    elif phase == "phase1_state":
        # Split phase-1 state kernel: for small/medium (M,D), fewer warps
        # reduce scheduling overhead and improve effective occupancy.
        num_warps = 4 if (d_bucket <= 64 and m_bucket <= 128) else 8
        num_stages = 1
    elif phase == "phase1_qk":
        # Split phase-1 qk kernel: lighter math, usually benefits from conservative launch.
        num_warps = 4 if (d_bucket <= 64 and m_bucket <= 128) else 8
        num_stages = 1
    elif phase == "prepare_prefix":
        # Recompute path for chunk/prefix state.
        num_warps = 8 if (m_bucket >= 256 and hopper) else 4
        num_stages = 2 if (n_bucket <= 8192 and chunk_size <= 64) else 1
    else:  # recurrent and fallback
        num_warps = 8 if (d_bucket >= 64 or (hopper and d_bucket >= 32)) else 4
        num_stages = 1

    if env_warps:
        num_warps = int(env_warps)
    if env_stages:
        num_stages = int(env_stages)

    # Default BLOCK_T policy: 16 everywhere unless chunk is smaller.
    block_t_store = int(env_block_t) if env_block_t else (16 if chunk_size >= 16 else chunk_size)
    if block_t_store % 16 != 0:
        raise ValueError(
            f"Invalid BLOCK_T={block_t_store}. FLARE backward BLOCK_T must be a multiple of 16."
        )
    if block_t_store > chunk_size:
        block_t_store = chunk_size
    if block_t_store <= 0:
        block_t_store = 16
    return num_warps, num_stages, block_t_store


def _get_bwd_chunk_size(forward_chunk_size: int, M: int, N: int, D: int, input_precision: str) -> int:
    """Pick a backward chunk size that avoids Triton shared-memory overflows."""
    env = os.environ.get("FLARE_BWD_CHUNK_SIZE", "")
    if env:
        c = int(env)
    else:
        # Phase-1 backward materializes several [C, M] and [C, C] fp32 buffers.
        # Empirical safe caps on H100:
        # - M >= 128 needs C <= 64 to avoid shared-memory OOR.
        # - M <= 64 can usually run C=128 (unless larger D/TF32 pressure).
        tf32_like = input_precision != "ieee"
        tf32x3 = input_precision == "tf32x3"
        if M >= 128:
            # For common long-context training shapes with D<=32, C=64 is both
            # valid and significantly faster than C=32 (fewer chunk launches).
            # Keep the conservative C=32 fallback for wider heads.
            if D <= 32:
                cap = 64
            else:
                # Phase-1 backward allocates multiple [C, M] / [C, C] fp32 tiles.
                # Under TF32-like modes with larger D, C=64 can exceed Hopper smem.
                cap = 32 if (D >= 64 or tf32_like) else 64
        elif M >= 64:
            cap = 64 if (D >= 64 or (tf32_like and D >= 32)) else 128
        elif D >= 96:
            # Large head-dim path is shared-memory heavy even for small-M.
            cap = 32
        elif D >= 64:
            # For M < 64 with D=64, C=128 can exceed Hopper shared-memory limits.
            cap = 64
        elif N >= 32768:
            # Long-context path: fewer launches helps, keep a conservative cap under TF32.
            cap = 128 if tf32_like else 256
        elif N >= 8192:
            cap = 128 if D >= 64 else 256
        else:
            cap = 128
        if tf32x3 and M >= 32:
            cap = min(cap, 64)
        c = min(forward_chunk_size, cap)
    c = max(16, c)
    return c

class PhaseProfiler:
    """
    Lightweight phase profiler.

    Usage (inside your function):
        prof = PhaseProfiler(device, enabled=profile)
        with prof.phase("Phase 0 (scores)"):
            ...
        with prof.phase("My custom phase name"):
            ...
        timings = prof.timings()  # Ordered by first occurrence, includes "Total"
    """

    def __init__(self, device: torch.device, enabled: bool = False):
        self.device = device
        self.enabled = enabled
        self._timings_ms: dict[str, float] = {}
        self._order: list[str] = []

    def _now(self):
        # Returns a start handle. Use _elapsed_ms(handle) for elapsed time.
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            return (start, end)
        return time.perf_counter()

    def _elapsed_ms(self, start_handle) -> float:
        if self.device.type == "cuda":
            start, end = start_handle
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end)
        return (time.perf_counter() - start_handle) * 1e3

    def add(self, name: str, elapsed_ms: float) -> None:
        if name not in self._timings_ms:
            self._order.append(name)
            self._timings_ms[name] = 0.0
        self._timings_ms[name] += float(elapsed_ms)

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = self._now()
        try:
            yield
        finally:
            self.add(name, self._elapsed_ms(t0))

    def total_ms(self) -> float:
        return float(sum(self._timings_ms.values()))

    def timings(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in self._order:
            out[k] = self._timings_ms[k]
        out["Total"] = self.total_ms()
        return out

#======================================================================#
# NOTE:
# Recurrent mode (`ChunkedFLARE`) is the recommended path for production use.
# The dense formulation has shown fundamental numerical instability in our
# sharp/long-context regimes.
#
# We continue to run separate dense experiments in `DenseFLARE` and
# `DenseFLARE1`, but latest results indicate both are currently worse than
# recurrent `ChunkedFLARE` (stability and quality).


__all__ = [name for name in globals() if name != "__builtins__"]

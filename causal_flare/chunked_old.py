#
# Causal FLARE Implementation with Triton Fused Kernel
#

import math
import os
import time
from contextlib import contextmanager

import torch
import torch.nn.functional as F
import torch.autograd as autograd

import triton
import triton.language as tl
import triton.testing

# Backward profiling (set by main for a single run)
_BWD_PROFILE_MODE = None  # "triton3"
_BWD_PROFILE_TIMINGS = {}
_DEBUG_PREFIX_STATS = {}


def _set_bwd_profile_mode(mode):
    global _BWD_PROFILE_MODE
    _BWD_PROFILE_MODE = mode
    if mode is not None:
        _BWD_PROFILE_TIMINGS[mode] = {}


def _record_bwd_timing(key, ms):
    if _BWD_PROFILE_MODE is None:
        return
    _BWD_PROFILE_TIMINGS.setdefault(_BWD_PROFILE_MODE, {})[key] = ms

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
class ChunkedFLAREOld(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        scale=1.0,
        profile=False,
    ):
        """
        Forward pass of causal FLARE.

        Args:
            Q: [H, M, D] - learnable latent queries
            K: [B, N, H, D] - keys from input sequence
            V: [B, N, H, D] - values from input sequence
            scale: scaling factor for attention
            profile: if True, return timings dict as second return value
        Returns:
            Y: [B, N, H, D] - output sequence
            If profile=True, returns (Y, timings_dict)
        """
        assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
            "Q, K, V must be 3D and 4D tensors respectively "
            f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
        )
        assert K.size() == V.size(), f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}"
        assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
            "Expected Q [H, M, D] and K/V [B, N, H, D]. "
            f"Got Q.shape={Q.shape} and K.shape={K.shape}"
        )

        ctx.scale = float(scale)

        H, M, D = Q.size()
        B, N, H, D = K.size()
        BH = B * H

        dtype = K.dtype
        device = Q.device
        Q_comp = Q
        K_comp = K
        V_comp = V
        use_fp16 = dtype == torch.float16
        use_bf16 = dtype == torch.bfloat16
        compute_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

        env_chunk = os.environ.get("FLARE_CHUNK_SIZE", "")
        CHUNK_SIZE = int(env_chunk) if env_chunk else 32

        O = torch.empty((B, N, H, D), device=device, dtype=compute_dtype)
        O.zero_()

        # Ensure block sizes are positive and divisible by 16
        assert M % 16 == 0, f"M must be divisible by 16. Got M={M}."
        assert D % 16 == 0, f"D must be divisible by 16. Got D={D}."
        assert CHUNK_SIZE % 16 == 0, f"CHUNK_SIZE must be divisible by 16. Got CHUNK_SIZE={CHUNK_SIZE}."

        block_m_env = os.environ.get("FLARE_BLOCK_M", "")
        if block_m_env:
            BLOCK_M = int(block_m_env)
        else:
            BLOCK_M = 64 if M >= 64 else 32
        if BLOCK_M > M:
            BLOCK_M = M
        assert BLOCK_M % 16 == 0, f"BLOCK_M must be divisible by 16. Got BLOCK_M={BLOCK_M}."
        if os.environ.get("FLARE_DEBUG_PREFIX_STATS", "0") == "1":
            _DEBUG_PREFIX_STATS["block_m"] = BLOCK_M
            _DEBUG_PREFIX_STATS["chunk_size"] = CHUNK_SIZE

        block_t_env = os.environ.get("FLARE_BLOCK_T", "")
        if block_t_env:
            BLOCK_T = int(block_t_env)
        else:
            BLOCK_T = 16 if CHUNK_SIZE >= 16 else CHUNK_SIZE
        if BLOCK_T > CHUNK_SIZE:
            BLOCK_T = CHUNK_SIZE
        assert BLOCK_T > 0, f"BLOCK_T must be > 0. Got BLOCK_T={BLOCK_T}."

        eps = _get_eps_for_dtype(dtype)
        clamp_max = _get_exp_clamp_for_dtype(dtype)
        scale = float(scale)
        num_warps = 4 if D <= 64 else 8
        num_stages = 2 if D <= 64 else 3
        stats_fp32 = True
        stats_dtype = torch.float32

        # Profiler setup
        prof = PhaseProfiler(device, enabled=profile)

        Q_stride = Q_comp.stride()
        K_stride = K_comp.stride()
        V_stride = V_comp.stride()
        O_stride = O.stride()

        NUM_CHUNKS = math.ceil(N / CHUNK_SIZE)
        NUM_M_BLOCKS = math.ceil(M / BLOCK_M)

        #---------------------------------------------------------------#
        # Phase 1: Compute chunk statistics independently for each chunk
        #---------------------------------------------------------------#
        with prof.phase("Phase 1 (chunk stats)"):
            grid_stats = (BH, NUM_CHUNKS, NUM_M_BLOCKS)

            chunk_max = torch.full((BH, NUM_CHUNKS, M), -float('inf'), device=device, dtype=stats_dtype)
            chunk_den = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=stats_dtype)
            chunk_num = torch.zeros((BH, NUM_CHUNKS, M, D), device=device, dtype=stats_dtype)

            cmax_stride = chunk_max.stride()
            cden_stride = chunk_den.stride()
            cnum_stride = chunk_num.stride()

            grid_stats = (BH, NUM_CHUNKS, NUM_M_BLOCKS)
            flare_chunk_prepare[grid_stats](
                K_comp, Q_comp, V_comp,
                chunk_max, chunk_den, chunk_num,
                K_stride[0], K_stride[1], K_stride[2], K_stride[3],
                Q_stride[0], Q_stride[1], Q_stride[2],
                V_stride[0], V_stride[1], V_stride[2], V_stride[3],
                cmax_stride[0], cmax_stride[1], cmax_stride[2],
                cden_stride[0], cden_stride[1], cden_stride[2],
                cnum_stride[0], cnum_stride[1], cnum_stride[2], cnum_stride[3],
                BH, M, N, D, scale,
                CHUNK_SIZE=CHUNK_SIZE,
                BLOCK_M=BLOCK_M,
                USE_FP16=use_fp16,
                USE_BF16=use_bf16,
                USE_FP32_STATS=stats_fp32,
                ALLOW_TF32=True,
                H=H,
                num_warps=num_warps,
                num_stages=num_stages,
            )

            # Only synchronize for profiling to avoid stalling the pipeline in the hot path.
            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()

            chunk_max = chunk_max.contiguous()
            chunk_den = chunk_den.contiguous()
            chunk_num = chunk_num.contiguous()
            cmax_stride = chunk_max.stride()
            cden_stride = chunk_den.stride()
            cnum_stride = chunk_num.stride()
            if os.environ.get("FLARE_DEBUG_PREFIX_STATS", "0") == "1":
                _DEBUG_PREFIX_STATS["chunk_max"] = chunk_max.detach().clone()
                _DEBUG_PREFIX_STATS["chunk_den"] = chunk_den.detach().clone()
                _DEBUG_PREFIX_STATS["chunk_num"] = chunk_num.detach().clone()

        #---------------------------------------------------------------#
        # Phase 2: Compute prefix statistics
        # IDEA: can implement parallel scan over chunks (chunking over chunks)
        #---------------------------------------------------------------#
        with prof.phase("Phase 2 (prefix stats)"):
            prefix_max = torch.empty((BH, NUM_CHUNKS, M), device=device, dtype=stats_dtype)
            prefix_den = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=stats_dtype)
            prefix_num = torch.zeros((BH, NUM_CHUNKS, M, D), device=device, dtype=stats_dtype)

            pmax_stride = prefix_max.stride()
            pden_stride = prefix_den.stride()
            pnum_stride = prefix_num.stride()

            grid_prefix = (BH,)  # One kernel per batch/head, processes all chunks sequentially
            flare_chunk_prefix[grid_prefix](
                chunk_max, chunk_den, chunk_num,
                prefix_max, prefix_den, prefix_num,
                cmax_stride[0], cmax_stride[1], cmax_stride[2],
                cden_stride[0], cden_stride[1], cden_stride[2],
                cnum_stride[0], cnum_stride[1], cnum_stride[2], cnum_stride[3],
                pmax_stride[0], pmax_stride[1], pmax_stride[2],
                pden_stride[0], pden_stride[1], pden_stride[2],
                pnum_stride[0], pnum_stride[1], pnum_stride[2], pnum_stride[3],
                BH, M, D, NUM_CHUNKS,
                USE_FP16=use_fp16,
                USE_BF16=use_bf16,
                USE_FP32_STATS=stats_fp32,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        prefix_max = prefix_max.contiguous()
        prefix_den = prefix_den.contiguous()
        prefix_num = prefix_num.contiguous()

        pmax_stride = prefix_max.stride()
        pden_stride = prefix_den.stride()
        pnum_stride = prefix_num.stride()

        if os.environ.get("FLARE_DEBUG_PREFIX_STATS", "0") == "1":
            _DEBUG_PREFIX_STATS["prefix_max"] = prefix_max.detach().clone()
            _DEBUG_PREFIX_STATS["prefix_den"] = prefix_den.detach().clone()
            _DEBUG_PREFIX_STATS["prefix_num"] = prefix_num.detach().clone()

        #---------------------------------------------------------------#
        # Phase 3: Dense output computation (Triton kernel)
        #---------------------------------------------------------------#
        with prof.phase("Phase 3 (output computation)"):
            grid_fwd = (BH, NUM_CHUNKS)
            flare_chunk_fwd[grid_fwd](
                K_comp, Q_comp, V_comp,
                prefix_max, prefix_den, prefix_num,
                O,
                K_stride[0], K_stride[1], K_stride[2], K_stride[3],
                Q_stride[0], Q_stride[1], Q_stride[2],
                V_stride[0], V_stride[1], V_stride[2], V_stride[3],
                pmax_stride[0], pmax_stride[1], pmax_stride[2],
                pden_stride[0], pden_stride[1], pden_stride[2],
                pnum_stride[0], pnum_stride[1], pnum_stride[2], pnum_stride[3],
                O_stride[0], O_stride[1], O_stride[2], O_stride[3],
                BH, M, N, D, scale, eps, clamp_max,
                CHUNK_SIZE=CHUNK_SIZE,
                BLOCK_T=BLOCK_T,
                USE_FP16=use_fp16,
                USE_BF16=use_bf16,
                USE_FP32_STATS=stats_fp32,
                ALLOW_TF32=True,
                STABLE_SCAN=False,
                H=H,
                num_warps=num_warps,
                num_stages=num_stages,
            )

        #---------------------------------------------------------------#
        ctx.save_for_backward(Q_comp, K_comp, V_comp, chunk_max, chunk_den, chunk_num, prefix_max, prefix_den, prefix_num)
        ctx.chunk_size = CHUNK_SIZE
        ctx.eps = eps
        ctx.use_fp16 = use_fp16
        ctx.use_bf16 = use_bf16
        ctx.out_dtype = dtype
        ctx.use_fp32_stats = stats_fp32
        ctx.block_m = BLOCK_M
        ctx.H = H
        ctx.M = M
        ctx.N = N
        ctx.D = D
        #---------------------------------------------------------------#

        if profile:
            return O.to(dtype), prof.timings()
        return O.to(dtype)

    @staticmethod
    def backward(ctx, dO, dTimings=None):
        if dO is None:
            return None, None, None, None, None
        Q, K, V, chunk_max, chunk_den, chunk_num, prefix_max, prefix_den, prefix_num = ctx.saved_tensors
        scale = ctx.scale
        eps = ctx.eps
        CHUNK_SIZE = ctx.chunk_size
        H = ctx.H
        M = ctx.M
        N = ctx.N
        D = ctx.D
        clamp_max = _get_exp_clamp_for_dtype(ctx.out_dtype)
        out_dtype = ctx.out_dtype

        B = K.size(0)
        BH = B * H
        NUM_CHUNKS = math.ceil(N / CHUNK_SIZE)

        device = Q.device
        compute_dtype = torch.float32

        Q_bwd = Q.float()
        K_bwd = K.float()
        V_bwd = V.float()
        dO_bwd = dO.float()

        BLOCK_M = ctx.block_m
        cmax_stride = chunk_max.stride()
        cden_stride = chunk_den.stride()
        cnum_stride = chunk_num.stride()
        pmax_stride = prefix_max.stride()
        pden_stride = prefix_den.stride()
        pnum_stride = prefix_num.stride()

        K_stride = K_bwd.stride()
        Q_stride = Q_bwd.stride()
        V_stride = V_bwd.stride()

        dQ_direct = torch.zeros((H, M, D), device=device, dtype=compute_dtype)
        dK_direct = torch.zeros((B, N, H, D), device=device, dtype=compute_dtype)
        dV_direct = torch.zeros((B, N, H, D), device=device, dtype=compute_dtype)

        dPmax = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=compute_dtype)
        dPden = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=compute_dtype)
        dPnum = torch.zeros((BH, NUM_CHUNKS, M, D), device=device, dtype=compute_dtype)
        dCmax = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=compute_dtype)
        dCden = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=compute_dtype)
        dCnum = torch.zeros((BH, NUM_CHUNKS, M, D), device=device, dtype=compute_dtype)

        num_warps = 4 if D <= 64 else 8
        num_stages = 1
        # per-chunk forward recompute -> stored buffers
        s_buf = torch.empty((BH, NUM_CHUNKS, CHUNK_SIZE, M), device=device, dtype=compute_dtype)
        m_buf = torch.empty((BH, NUM_CHUNKS, CHUNK_SIZE, M), device=device, dtype=compute_dtype)
        l_buf = torch.empty((BH, NUM_CHUNKS, CHUNK_SIZE, M), device=device, dtype=compute_dtype)
        sumexpv_buf = torch.empty((BH, NUM_CHUNKS, CHUNK_SIZE, M, D), device=device, dtype=compute_dtype)

        s_stride = s_buf.stride()
        m_stride = m_buf.stride()
        l_stride = l_buf.stride()
        sev_stride = sumexpv_buf.stride()

        flare_chunk_fwd_store[(BH, NUM_CHUNKS)](
            K_bwd, Q_bwd, V_bwd,
            prefix_max, prefix_den, prefix_num,
            s_buf, m_buf, l_buf, sumexpv_buf,
            K_stride[0], K_stride[1], K_stride[2], K_stride[3],
            Q_stride[0], Q_stride[1], Q_stride[2],
            V_stride[0], V_stride[1], V_stride[2], V_stride[3],
            pmax_stride[0], pmax_stride[1], pmax_stride[2],
            pden_stride[0], pden_stride[1], pden_stride[2],
            pnum_stride[0], pnum_stride[1], pnum_stride[2], pnum_stride[3],
            s_stride[0], s_stride[1], s_stride[2], s_stride[3],
            m_stride[0], m_stride[1], m_stride[2], m_stride[3],
            l_stride[0], l_stride[1], l_stride[2], l_stride[3],
            sev_stride[0], sev_stride[1], sev_stride[2], sev_stride[3], sev_stride[4],
            BH, M, N, D, scale, eps, clamp_max,
            CHUNK_SIZE=CHUNK_SIZE,
            BLOCK_T=CHUNK_SIZE,
            USE_FP16=False,
            USE_BF16=False,
            USE_FP32_STATS=True,
            ALLOW_TF32=True,
            H=H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        flare_chunk_bwd_recurrent[(BH, NUM_CHUNKS)](
            K_bwd, Q_bwd, V_bwd,
            prefix_max, prefix_den, prefix_num,
            s_buf, m_buf, l_buf, sumexpv_buf,
            dO_bwd,
            dQ_direct, dK_direct, dV_direct,
            dPmax, dPden, dPnum,
            K_stride[0], K_stride[1], K_stride[2], K_stride[3],
            Q_stride[0], Q_stride[1], Q_stride[2],
            V_stride[0], V_stride[1], V_stride[2], V_stride[3],
            pmax_stride[0], pmax_stride[1], pmax_stride[2],
            pden_stride[0], pden_stride[1], pden_stride[2],
            pnum_stride[0], pnum_stride[1], pnum_stride[2], pnum_stride[3],
            s_stride[0], s_stride[1], s_stride[2], s_stride[3],
            m_stride[0], m_stride[1], m_stride[2], m_stride[3],
            l_stride[0], l_stride[1], l_stride[2], l_stride[3],
            sev_stride[0], sev_stride[1], sev_stride[2], sev_stride[3], sev_stride[4],
            dO_bwd.stride()[0], dO_bwd.stride()[1], dO_bwd.stride()[2], dO_bwd.stride()[3],
            dQ_direct.stride()[0], dQ_direct.stride()[1], dQ_direct.stride()[2],
            dK_direct.stride()[0], dK_direct.stride()[1], dK_direct.stride()[2], dK_direct.stride()[3],
            dV_direct.stride()[0], dV_direct.stride()[1], dV_direct.stride()[2], dV_direct.stride()[3],
            dPmax.stride()[0], dPmax.stride()[1], dPmax.stride()[2],
            dPden.stride()[0], dPden.stride()[1], dPden.stride()[2],
            dPnum.stride()[0], dPnum.stride()[1], dPnum.stride()[2], dPnum.stride()[3],
            BH, M, N, D, scale, eps, clamp_max,
            CHUNK_SIZE=CHUNK_SIZE,
            USE_FP16=False,
            USE_BF16=False,
            USE_FP32_STATS=True,
            ALLOW_TF32=True,
            H=H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        # Backprop through prefix scan into chunk stats
        flare_chunk_prefix_bwd[(BH,)](
            chunk_max, chunk_den, chunk_num,
            prefix_max, prefix_den, prefix_num,
            dPmax, dPden, dPnum,
            dCmax, dCden, dCnum,
            cmax_stride[0], cmax_stride[1], cmax_stride[2],
            cden_stride[0], cden_stride[1], cden_stride[2],
            cnum_stride[0], cnum_stride[1], cnum_stride[2], cnum_stride[3],
            pmax_stride[0], pmax_stride[1], pmax_stride[2],
            pden_stride[0], pden_stride[1], pden_stride[2],
            pnum_stride[0], pnum_stride[1], pnum_stride[2], pnum_stride[3],
            dPmax.stride()[0], dPmax.stride()[1], dPmax.stride()[2],
            dPden.stride()[0], dPden.stride()[1], dPden.stride()[2],
            dPnum.stride()[0], dPnum.stride()[1], dPnum.stride()[2], dPnum.stride()[3],
            dCmax.stride()[0], dCmax.stride()[1], dCmax.stride()[2],
            dCden.stride()[0], dCden.stride()[1], dCden.stride()[2],
            dCnum.stride()[0], dCnum.stride()[1], dCnum.stride()[2], dCnum.stride()[3],
            BH, M, D, NUM_CHUNKS,
            USE_FP16=False,
            USE_BF16=False,
        )

        dQ_prefix = torch.zeros((H, M, D), device=device, dtype=compute_dtype)
        dK_prefix = torch.zeros((B, N, H, D), device=device, dtype=compute_dtype)
        dV_prefix = torch.zeros((B, N, H, D), device=device, dtype=compute_dtype)

        flare_chunk_stats_bwd3[(BH, NUM_CHUNKS)](
            K_bwd, Q_bwd, V_bwd,
            dCmax, dCden, dCnum,
            dQ_prefix, dK_prefix, dV_prefix,
            K_stride[0], K_stride[1], K_stride[2], K_stride[3],
            Q_stride[0], Q_stride[1], Q_stride[2],
            V_stride[0], V_stride[1], V_stride[2], V_stride[3],
            dCmax.stride()[0], dCmax.stride()[1], dCmax.stride()[2],
            dCden.stride()[0], dCden.stride()[1], dCden.stride()[2],
            dCnum.stride()[0], dCnum.stride()[1], dCnum.stride()[2], dCnum.stride()[3],
            dQ_prefix.stride()[0], dQ_prefix.stride()[1], dQ_prefix.stride()[2],
            dK_prefix.stride()[0], dK_prefix.stride()[1], dK_prefix.stride()[2], dK_prefix.stride()[3],
            dV_prefix.stride()[0], dV_prefix.stride()[1], dV_prefix.stride()[2], dV_prefix.stride()[3],
            BH, M, N, D, scale,
            CHUNK_SIZE=CHUNK_SIZE,
            USE_FP16=False,
            USE_BF16=False,
            USE_FP32_STATS=True,
            H=H,
            num_warps=num_warps,
            num_stages=num_stages,
        )

        dQ = dQ_direct + dQ_prefix
        dK = dK_direct + dK_prefix
        dV = dV_direct + dV_prefix

        return dQ.to(out_dtype), dK.to(out_dtype), dV.to(out_dtype), None, None

@triton.jit
def flare_chunk_prepare(
    K_ptr, Q_ptr, V_ptr,
    ChunkMax_ptr, ChunkDen_ptr, ChunkNum_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_cmax_bh, stride_cmax_chunk, stride_cmax_m,
    stride_cden_bh, stride_cden_chunk, stride_cden_m,
    stride_cnum_bh, stride_cnum_chunk, stride_cnum_m, stride_cnum_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    pid_m = tl.program_id(2)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, D)
    token_offsets = tl.arange(0, CHUNK_SIZE)
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_entries = mask_m[:, None] & mask_d[None, :]

    chunk_start = chunk_idx * CHUNK_SIZE
    
    # Load all queries [M, D]
    Q_base_ptr = Q_ptr + pid_h * stride_q_h
    state_dtype = tl.float32 if USE_FP32_STATS else (tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    Q_vals = tl.load(
        Q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets[None, :] * stride_q_d,
        mask=mask_entries,
        other=0.0,
    ).to(state_dtype)  # [M, D]

    K_base_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + chunk_start * stride_k_n
    V_base_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + chunk_start * stride_v_n
    token_idx = chunk_start + token_offsets
    token_mask = token_idx < N  # [C]

    # Load K/V chunk once and compute S with a blocked matmul
    K_chunk = tl.load(
        K_base_ptr + token_offsets[:, None] * stride_k_n + d_offsets[None, :] * stride_k_d,
        mask=token_mask[:, None] & mask_d[None, :],
        other=0.0,
    ).to(state_dtype)  # [C, D]
    V_chunk = tl.load(
        V_base_ptr + token_offsets[:, None] * stride_v_n + d_offsets[None, :] * stride_v_d,
        mask=token_mask[:, None] & mask_d[None, :],
        other=0.0,
    ).to(state_dtype)  # [C, D]

    S = tl.dot(K_chunk, tl.trans(Q_vals), out_dtype=tl.float32, allow_tf32=ALLOW_TF32) * scale  # [C, BM]
    S = tl.where(token_mask[:, None] & mask_m[None, :], S, -float("inf"))

    # Compute chunk statistics in one pass over the chunk.
    chunk_max = tl.max(S, axis=0)  # [BM]
    chunk_max = tl.where(mask_m, chunk_max, 0.0)
    expS = tl.exp(S - chunk_max[None, :])
    expS = tl.where(token_mask[:, None] & mask_m[None, :], expS, 0.0)
    expS_dot = expS if USE_FP32_STATS else (expS.to(tl.bfloat16) if USE_BF16 else (expS.to(tl.float16) if USE_FP16 else expS))
    chunk_den = tl.sum(expS, axis=0)  # [BM]
    chunk_num = tl.dot(tl.trans(expS_dot), V_chunk, out_dtype=tl.float32, allow_tf32=ALLOW_TF32)  # [BM, D]

    # Store chunk statistics
    chunk_max_ptr = ChunkMax_ptr + pid_bh * stride_cmax_bh + chunk_idx * stride_cmax_chunk
    chunk_den_ptr = ChunkDen_ptr + pid_bh * stride_cden_bh + chunk_idx * stride_cden_chunk
    chunk_num_ptr = ChunkNum_ptr + pid_bh * stride_cnum_bh + chunk_idx * stride_cnum_chunk

    tl.store(chunk_max_ptr + m_offsets * stride_cmax_m, chunk_max, mask=mask_m,)
    tl.store(chunk_den_ptr + m_offsets * stride_cden_m, chunk_den, mask=mask_m,)
    tl.store(chunk_num_ptr + m_offsets[:, None] * stride_cnum_m + d_offsets[None, :] * stride_cnum_d, chunk_num, mask=mask_entries,)


@triton.jit
def flare_chunk_prefix(
    ChunkMax_ptr, ChunkDen_ptr, ChunkNum_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    stride_cmax_bh, stride_cmax_chunk, stride_cmax_m,
    stride_cden_bh, stride_cden_chunk, stride_cden_m,
    stride_cnum_bh, stride_cnum_chunk, stride_cnum_m, stride_cnum_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    BH, M: tl.constexpr, D: tl.constexpr, NUM_CHUNKS,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    if pid_bh >= BH:
        return

    m_offsets = tl.arange(0, M)
    d_offsets = tl.arange(0, D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_entries = mask_m[:, None] & mask_d[None, :]

    state_dtype = tl.float32 if USE_FP32_STATS else (tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    # Initialize prefix state (will be updated as we process chunks sequentially)
    prefix_max_state = tl.full((M,), -float("inf"), dtype=state_dtype)
    prefix_den_state = tl.zeros((M,), dtype=state_dtype)
    prefix_num_state = tl.zeros((M, D), dtype=state_dtype)

    # Process all chunks sequentially for this batch/head
    # This eliminates redundant computation - we compute prefix stats once per chunk
    for chunk_idx in tl.range(0, NUM_CHUNKS):
        # Store prefix statistics BEFORE merging with current chunk
        # prefix_stats[chunk_idx] = cumulative stats from chunks 0 to chunk_idx-1
        prefix_max_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + chunk_idx * stride_pmax_chunk
        prefix_den_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + chunk_idx * stride_pden_chunk
        prefix_num_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + chunk_idx * stride_pnum_chunk

        tl.store(prefix_max_ptr + m_offsets * stride_pmax_m, prefix_max_state, mask=mask_m,)
        tl.store(prefix_den_ptr + m_offsets * stride_pden_m, prefix_den_state, mask=mask_m,)
        tl.store(prefix_num_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d, prefix_num_state, mask=mask_entries,)

        # Load chunk statistics for current chunk
        chunk_max_ptr = ChunkMax_ptr + pid_bh * stride_cmax_bh + chunk_idx * stride_cmax_chunk
        chunk_den_ptr = ChunkDen_ptr + pid_bh * stride_cden_bh + chunk_idx * stride_cden_chunk
        chunk_num_ptr = ChunkNum_ptr + pid_bh * stride_cnum_bh + chunk_idx * stride_cnum_chunk

        chunk_cm = tl.load(
            chunk_max_ptr + m_offsets * stride_cmax_m,
            mask=mask_m,
            other=-float("inf"),
        ).to(state_dtype)
        chunk_cd = tl.load(
            chunk_den_ptr + m_offsets * stride_cden_m,
            mask=mask_m,
            other=0.0,
        ).to(state_dtype)
        chunk_cs = tl.load(
            chunk_num_ptr + m_offsets[:, None] * stride_cnum_m + d_offsets[None, :] * stride_cnum_d,
            mask=mask_entries,
            other=0.0,
        ).to(state_dtype)

        # Merge current chunk statistics with prefix state using numerically stable softmax combination
        # This updates prefix_state to include chunks 0 to chunk_idx (for next iteration)
        max_new = tl.maximum(prefix_max_state, chunk_cm)
        scale_prev = tl.exp((prefix_max_state - max_new).to(tl.float32)).to(state_dtype)
        scale_chunk = tl.exp((chunk_cm - max_new).to(tl.float32)).to(state_dtype)

        prefix_den_state = (prefix_den_state * scale_prev + chunk_cd * scale_chunk).to(state_dtype)
        prefix_num_state = (prefix_num_state * scale_prev[:, None] + chunk_cs * scale_chunk[:, None]).to(state_dtype)
        prefix_max_state = max_new.to(state_dtype)


@triton.jit
def flare_chunk_prefix_bwd(
    ChunkMax_ptr, ChunkDen_ptr, ChunkNum_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    dPrefixMax_ptr, dPrefixDen_ptr, dPrefixNum_ptr,
    dChunkMax_ptr, dChunkDen_ptr, dChunkNum_ptr,
    stride_cmax_bh, stride_cmax_chunk, stride_cmax_m,
    stride_cden_bh, stride_cden_chunk, stride_cden_m,
    stride_cnum_bh, stride_cnum_chunk, stride_cnum_m, stride_cnum_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    stride_dpmax_bh, stride_dpmax_chunk, stride_dpmax_m,
    stride_dpden_bh, stride_dpden_chunk, stride_dpden_m,
    stride_dpnum_bh, stride_dpnum_chunk, stride_dpnum_m, stride_dpnum_d,
    stride_dcmax_bh, stride_dcmax_chunk, stride_dcmax_m,
    stride_dcden_bh, stride_dcden_chunk, stride_dcden_m,
    stride_dcnum_bh, stride_dcnum_chunk, stride_dcnum_m, stride_dcnum_d,
    BH, M: tl.constexpr, D: tl.constexpr, NUM_CHUNKS,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    if pid_bh >= BH:
        return

    m_offsets = tl.arange(0, M)
    d_offsets = tl.arange(0, D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_entries = mask_m[:, None] & mask_d[None, :]

    state_dtype = tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32)

    g_next_max = tl.zeros((M,), dtype=tl.float32)
    g_next_den = tl.zeros((M,), dtype=tl.float32)
    g_next_num = tl.zeros((M, D), dtype=tl.float32)

    for idx in tl.range(0, NUM_CHUNKS):
        chunk_idx = NUM_CHUNKS - 1 - idx

        pmax_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + chunk_idx * stride_pmax_chunk
        pden_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + chunk_idx * stride_pden_chunk
        pnum_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + chunk_idx * stride_pnum_chunk

        cmax_ptr = ChunkMax_ptr + pid_bh * stride_cmax_bh + chunk_idx * stride_cmax_chunk
        cden_ptr = ChunkDen_ptr + pid_bh * stride_cden_bh + chunk_idx * stride_cden_chunk
        cnum_ptr = ChunkNum_ptr + pid_bh * stride_cnum_bh + chunk_idx * stride_cnum_chunk

        dpmax_ptr = dPrefixMax_ptr + pid_bh * stride_dpmax_bh + chunk_idx * stride_dpmax_chunk
        dpden_ptr = dPrefixDen_ptr + pid_bh * stride_dpden_bh + chunk_idx * stride_dpden_chunk
        dpnum_ptr = dPrefixNum_ptr + pid_bh * stride_dpnum_bh + chunk_idx * stride_dpnum_chunk

        dcmax_ptr = dChunkMax_ptr + pid_bh * stride_dcmax_bh + chunk_idx * stride_dcmax_chunk
        dcden_ptr = dChunkDen_ptr + pid_bh * stride_dcden_bh + chunk_idx * stride_dcden_chunk
        dcnum_ptr = dChunkNum_ptr + pid_bh * stride_dcnum_bh + chunk_idx * stride_dcnum_chunk

        prev_max = tl.load(pmax_ptr + m_offsets * stride_pmax_m, mask=mask_m, other=-float("inf")).to(tl.float32)
        prev_den = tl.load(pden_ptr + m_offsets * stride_pden_m, mask=mask_m, other=0.0).to(tl.float32)
        prev_num = tl.load(
            pnum_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d,
            mask=mask_entries,
            other=0.0,
        ).to(tl.float32)

        chunk_max = tl.load(cmax_ptr + m_offsets * stride_cmax_m, mask=mask_m, other=-float("inf")).to(tl.float32)
        chunk_den = tl.load(cden_ptr + m_offsets * stride_cden_m, mask=mask_m, other=0.0).to(tl.float32)
        chunk_num = tl.load(
            cnum_ptr + m_offsets[:, None] * stride_cnum_m + d_offsets[None, :] * stride_cnum_d,
            mask=mask_entries,
            other=0.0,
        ).to(tl.float32)

        dp_max = tl.load(dpmax_ptr + m_offsets * stride_dpmax_m, mask=mask_m, other=0.0).to(tl.float32)
        dp_den = tl.load(dpden_ptr + m_offsets * stride_dpden_m, mask=mask_m, other=0.0).to(tl.float32)
        dp_num = tl.load(
            dpnum_ptr + m_offsets[:, None] * stride_dpnum_m + d_offsets[None, :] * stride_dpnum_d,
            mask=mask_entries,
            other=0.0,
        ).to(tl.float32)

        max_new = tl.maximum(prev_max, chunk_max)
        scale_prev = tl.exp(prev_max - max_new)
        scale_chunk = tl.exp(chunk_max - max_new)

        # Gradients through den/num merges
        g_scale_prev = g_next_den * prev_den + tl.sum(g_next_num * prev_num, axis=1)
        g_scale_chunk = g_next_den * chunk_den + tl.sum(g_next_num * chunk_num, axis=1)

        g_prev_den = g_next_den * scale_prev
        g_chunk_den = g_next_den * scale_chunk
        g_prev_num = g_next_num * scale_prev[:, None]
        g_chunk_num = g_next_num * scale_chunk[:, None]

        g_next_max_total = g_next_max - g_scale_prev * scale_prev - g_scale_chunk * scale_chunk
        mask_prev = prev_max >= chunk_max
        mask_chunk = ~mask_prev

        g_prev_max = g_scale_prev * scale_prev + g_next_max_total * mask_prev
        g_chunk_max = g_scale_chunk * scale_chunk + g_next_max_total * mask_chunk

        # Add direct gradient from prefix usage (before merge)
        g_prev_max += dp_max
        g_prev_den += dp_den
        g_prev_num += dp_num

        g_next_max = g_prev_max
        g_next_den = g_prev_den
        g_next_num = g_prev_num

        tl.store(dcmax_ptr + m_offsets * stride_dcmax_m, g_chunk_max, mask=mask_m)
        tl.store(dcden_ptr + m_offsets * stride_dcden_m, g_chunk_den, mask=mask_m)
        tl.store(
            dcnum_ptr + m_offsets[:, None] * stride_dcnum_m + d_offsets[None, :] * stride_dcnum_d,
            g_chunk_num,
            mask=mask_entries,
        )
@triton.jit
def flare_chunk_fwd(
    K_ptr, Q_ptr, V_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    O_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    stride_o_b, stride_o_n, stride_o_h, stride_o_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale, eps, clamp_max,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    STABLE_SCAN: tl.constexpr,
    H: tl.constexpr,
):
    """
    Phase 3: Dense output computation kernel.
    """
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    chunk_start = chunk_idx * CHUNK_SIZE

    c_offsets = tl.arange(0, CHUNK_SIZE)
    m_offsets = tl.arange(0, M)
    d_offsets = tl.arange(0, D)

    token_idx = chunk_start + c_offsets
    token_mask = token_idx < N  # [C]
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_md = mask_m[:, None] & mask_d[None, :]
    mask_kv = token_mask[:, None] & mask_d[None, :]

    # ---- Load Q [M, D] ----
    Q_base_ptr = Q_ptr + pid_h * stride_q_h
    Q_vals = tl.load(
        Q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets[None, :] * stride_q_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))  # [M, D]
    Q_vals_f = Q_vals.to(tl.float32)

    # ---- Load prefix stats for this chunk (stats from all previous chunks) ----
    pmax_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + chunk_idx * stride_pmax_chunk
    pden_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + chunk_idx * stride_pden_chunk
    pnum_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + chunk_idx * stride_pnum_chunk

    stats_dtype = tl.float32 if USE_FP32_STATS else (tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    prefix_max = tl.load(
        pmax_ptr + m_offsets * stride_pmax_m,
        mask=mask_m,
        other=-float("inf"),
    ).to(stats_dtype)  # [M]
    prefix_den = tl.load(
        pden_ptr + m_offsets * stride_pden_m,
        mask=mask_m,
        other=0.0,
    ).to(stats_dtype)  # [M]
    prefix_num = tl.load(
        pnum_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d,
        mask=mask_md,
        other=0.0,
    ).to(stats_dtype)  # [M, D]

    prefix_max_f = prefix_max.to(tl.float32)
    prefix_den_f = prefix_den.to(tl.float32)
    prefix_num_f = prefix_num.to(tl.float32)
    prefix_max_clamped = tl.minimum(prefix_max_f, clamp_max)
    exp_prev_max = tl.exp(prefix_max_clamped)
    sum_prev_exp = prefix_den_f * exp_prev_max
    sum_prev_exp_v = prefix_num_f * exp_prev_max[:, None]

    # ---- Load K,V chunks [C, D] ----
    K_base_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + chunk_start * stride_k_n
    V_base_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + chunk_start * stride_v_n
    K_chunk = tl.load(
        K_base_ptr + c_offsets[:, None] * stride_k_n + d_offsets[None, :] * stride_k_d,
        mask=mask_kv,
        other=0.0,
    ).to(tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))  # [C, D]
    V_chunk = tl.load(
        V_base_ptr + c_offsets[:, None] * stride_v_n + d_offsets[None, :] * stride_v_d,
        mask=mask_kv,
        other=0.0,
    ).to(tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))  # [C, D]
    K_chunk_f = K_chunk.to(tl.float32)
    V_chunk_f = V_chunk.to(tl.float32)

    O_base_ptr = O_ptr + pid_b * stride_o_b + pid_h * stride_o_h + chunk_start * stride_o_n

    m_state = prefix_max_f
    l_state = prefix_den_f
    n_state = prefix_num_f

    t0 = 0
    while t0 < CHUNK_SIZE:
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < CHUNK_SIZE
        token_mask_t = (chunk_start + t_offsets) < N

        k_ptr = K_base_ptr + t_offsets[:, None] * stride_k_n + d_offsets[None, :] * stride_k_d
        k_sub = tl.load(
            k_ptr,
            mask=token_mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        s_sub = tl.dot(Q_vals_f, tl.trans(k_sub), out_dtype=tl.float32, allow_tf32=ALLOW_TF32) * scale
        s_sub = tl.where(mask_m[:, None] & token_mask_t[None, :], s_sub, -float("inf"))

        t_idx = tl.arange(0, BLOCK_T)
        for j in tl.static_range(0, BLOCK_T):
            valid_t = (t0 + j) < CHUNK_SIZE
            token_valid = (chunk_start + t0 + j) < N
            col_mask = t_idx == j
            s_t = tl.sum(tl.where(col_mask[None, :], s_sub, 0.0), axis=1)
            s_t = tl.where(token_valid & mask_m, s_t, -float("inf"))
            s_t = tl.where(s_t == s_t, s_t, -float("inf"))

            m_new = tl.maximum(m_state, s_t)
            is_m_state_inf = m_state == -float("inf")
            is_m_new_inf = m_new == -float("inf")
            m_new_safe = tl.where(is_m_new_inf, 0.0, m_new)

            exp_prev = tl.where(
                is_m_state_inf & is_m_new_inf,
                1.0,
                tl.where(is_m_state_inf, 0.0, tl.exp(m_state - m_new_safe)),
            )
            exp_s = tl.where(is_m_new_inf, 0.0, tl.exp(s_t - m_new_safe))

            v_t = tl.load(
                V_base_ptr + (t0 + j) * stride_v_n + d_offsets * stride_v_d,
                mask=token_valid & mask_d,
                other=0.0,
            ).to(tl.float32)

            l_state = l_state * exp_prev + exp_s
            n_state = n_state * exp_prev[:, None] + exp_s[:, None] * v_t[None, :]
            m_state = m_new

            exp_m = tl.exp(tl.minimum(m_state, clamp_max))
            sum_exp = l_state * exp_m
            sum_exp_v = n_state * exp_m[:, None]
            den_total_t = sum_exp

            s_max = tl.max(s_t, axis=0)
            s_exp = tl.exp(s_t - s_max)
            s_exp = tl.where(token_valid & mask_m, s_exp, 0.0)
            s_sum = tl.sum(s_exp, axis=0)
            P_t = tl.where(token_valid, s_exp / (s_sum + 1e-20), 0.0)

            expA_t = P_t / (den_total_t + eps)
            o_t = tl.sum(expA_t[:, None] * sum_exp_v, axis=0)
            tl.store(
                O_base_ptr + (t0 + j) * stride_o_n + d_offsets * stride_o_d,
                o_t,
                mask=token_valid & mask_d,
            )

        t0 += BLOCK_T

@triton.jit
def flare_chunk_fwd_store(
    K_ptr, Q_ptr, V_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    S_ptr, M_ptr, L_ptr, SumExpV_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    stride_s_bh, stride_s_chunk, stride_s_t, stride_s_m,
    stride_m_bh, stride_m_chunk, stride_m_t, stride_m_m,
    stride_l_bh, stride_l_chunk, stride_l_t, stride_l_m,
    stride_sev_bh, stride_sev_chunk, stride_sev_t, stride_sev_m, stride_sev_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale, eps, clamp_max,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    chunk_start = chunk_idx * CHUNK_SIZE

    c_offsets = tl.arange(0, CHUNK_SIZE)
    m_offsets = tl.arange(0, M)
    d_offsets = tl.arange(0, D)

    token_idx = chunk_start + c_offsets
    token_mask = token_idx < N
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_md = mask_m[:, None] & mask_d[None, :]

    Q_base_ptr = Q_ptr + pid_h * stride_q_h
    Q_vals = tl.load(
        Q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets[None, :] * stride_q_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    Q_vals_f = Q_vals.to(tl.float32)

    pmax_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + chunk_idx * stride_pmax_chunk
    pden_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + chunk_idx * stride_pden_chunk
    pnum_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + chunk_idx * stride_pnum_chunk

    stats_dtype = tl.float32 if USE_FP32_STATS else (tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    prefix_max = tl.load(pmax_ptr + m_offsets * stride_pmax_m, mask=mask_m, other=-float("inf")).to(stats_dtype)
    prefix_den = tl.load(pden_ptr + m_offsets * stride_pden_m, mask=mask_m, other=0.0).to(stats_dtype)
    prefix_num = tl.load(
        pnum_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d,
        mask=mask_md,
        other=0.0,
    ).to(stats_dtype)

    K_base_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + chunk_start * stride_k_n
    V_base_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + chunk_start * stride_v_n

    m_state = prefix_max.to(tl.float32)
    l_state = prefix_den.to(tl.float32)
    n_state = prefix_num.to(tl.float32)

    t0 = 0
    while t0 < CHUNK_SIZE:
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        token_mask_t = (chunk_start + t_offsets) < N

        k_ptr = K_base_ptr + t_offsets[:, None] * stride_k_n + d_offsets[None, :] * stride_k_d
        k_sub = tl.load(
            k_ptr,
            mask=token_mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        s_sub = tl.dot(Q_vals_f, tl.trans(k_sub), out_dtype=tl.float32, allow_tf32=ALLOW_TF32) * scale
        s_sub = tl.where(mask_m[:, None] & token_mask_t[None, :], s_sub, -float("inf"))

        t_idx = tl.arange(0, BLOCK_T)
        for j in tl.static_range(0, BLOCK_T):
            token_valid = (chunk_start + t0 + j) < N
            col_mask = t_idx == j
            s_t = tl.sum(tl.where(col_mask[None, :], s_sub, 0.0), axis=1)
            s_t = tl.where(token_valid & mask_m, s_t, -float("inf"))
            s_t = tl.where(s_t == s_t, s_t, -float("inf"))

            m_new = tl.maximum(m_state, s_t)
            is_m_state_inf = m_state == -float("inf")
            is_m_new_inf = m_new == -float("inf")
            m_new_safe = tl.where(is_m_new_inf, 0.0, m_new)

            exp_prev = tl.where(
                is_m_state_inf & is_m_new_inf,
                1.0,
                tl.where(is_m_state_inf, 0.0, tl.exp(m_state - m_new_safe)),
            )
            exp_s = tl.where(is_m_new_inf, 0.0, tl.exp(s_t - m_new_safe))

            v_t = tl.load(
                V_base_ptr + (t0 + j) * stride_v_n + d_offsets * stride_v_d,
                mask=token_valid & mask_d,
                other=0.0,
            ).to(tl.float32)

            l_state = l_state * exp_prev + exp_s
            n_state = n_state * exp_prev[:, None] + exp_s[:, None] * v_t[None, :]
            m_state = m_new

            exp_m = tl.exp(tl.minimum(m_state, clamp_max))
            sum_exp_v = n_state * exp_m[:, None]

            tl.store(
                S_ptr + pid_bh * stride_s_bh + chunk_idx * stride_s_chunk + (t0 + j) * stride_s_t + m_offsets * stride_s_m,
                s_t,
                mask=mask_m,
            )
            tl.store(
                M_ptr + pid_bh * stride_m_bh + chunk_idx * stride_m_chunk + (t0 + j) * stride_m_t + m_offsets * stride_m_m,
                m_state,
                mask=mask_m,
            )
            tl.store(
                L_ptr + pid_bh * stride_l_bh + chunk_idx * stride_l_chunk + (t0 + j) * stride_l_t + m_offsets * stride_l_m,
                l_state,
                mask=mask_m,
            )
            tl.store(
                SumExpV_ptr
                + pid_bh * stride_sev_bh
                + chunk_idx * stride_sev_chunk
                + (t0 + j) * stride_sev_t
                + m_offsets[:, None] * stride_sev_m
                + d_offsets[None, :] * stride_sev_d,
                sum_exp_v,
                mask=mask_md,
            )
        t0 += BLOCK_T

@triton.jit
def flare_chunk_bwd_recurrent(
    K_ptr, Q_ptr, V_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    S_ptr, M_ptr, L_ptr, SumExpV_ptr,
    dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    dPmax_ptr, dPden_ptr, dPnum_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    stride_s_bh, stride_s_chunk, stride_s_t, stride_s_m,
    stride_m_bh, stride_m_chunk, stride_m_t, stride_m_m,
    stride_l_bh, stride_l_chunk, stride_l_t, stride_l_m,
    stride_sev_bh, stride_sev_chunk, stride_sev_t, stride_sev_m, stride_sev_d,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_dq_h, stride_dq_m, stride_dq_d,
    stride_dk_b, stride_dk_n, stride_dk_h, stride_dk_d,
    stride_dv_b, stride_dv_n, stride_dv_h, stride_dv_d,
    stride_dpmax_bh, stride_dpmax_chunk, stride_dpmax_m,
    stride_dpden_bh, stride_dpden_chunk, stride_dpden_m,
    stride_dpnum_bh, stride_dpnum_chunk, stride_dpnum_m, stride_dpnum_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale, eps, clamp_max,
    CHUNK_SIZE: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH:
        return
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    chunk_start = chunk_idx * CHUNK_SIZE

    m_offsets = tl.arange(0, M)
    d_offsets = tl.arange(0, D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_md = mask_m[:, None] & mask_d[None, :]

    Q_base_ptr = Q_ptr + pid_h * stride_q_h
    Q_vals = tl.load(
        Q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets[None, :] * stride_q_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)

    pmax_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + chunk_idx * stride_pmax_chunk
    pden_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + chunk_idx * stride_pden_chunk
    pnum_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + chunk_idx * stride_pnum_chunk

    prefix_max = tl.load(pmax_ptr + m_offsets * stride_pmax_m, mask=mask_m, other=-float("inf")).to(tl.float32)
    prefix_den = tl.load(pden_ptr + m_offsets * stride_pden_m, mask=mask_m, other=0.0).to(tl.float32)
    prefix_num = tl.load(
        pnum_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)

    d_m = tl.zeros((M,), tl.float32)
    d_l = tl.zeros((M,), tl.float32)
    d_n = tl.zeros((M, D), tl.float32)

    t_rev = 0
    while t_rev < CHUNK_SIZE:
        t = CHUNK_SIZE - 1 - t_rev
        token_valid = (chunk_start + t) < N
        valid_f = token_valid.to(tl.float32)

        s_t = tl.load(
            S_ptr + pid_bh * stride_s_bh + chunk_idx * stride_s_chunk + t * stride_s_t + m_offsets * stride_s_m,
            mask=mask_m,
            other=-float("inf"),
        ).to(tl.float32)
        m_t = tl.load(
            M_ptr + pid_bh * stride_m_bh + chunk_idx * stride_m_chunk + t * stride_m_t + m_offsets * stride_m_m,
            mask=mask_m,
            other=-float("inf"),
        ).to(tl.float32)
        l_t = tl.load(
            L_ptr + pid_bh * stride_l_bh + chunk_idx * stride_l_chunk + t * stride_l_t + m_offsets * stride_l_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        sumexpv_t = tl.load(
            SumExpV_ptr
            + pid_bh * stride_sev_bh
            + chunk_idx * stride_sev_chunk
            + t * stride_sev_t
            + m_offsets[:, None] * stride_sev_m
            + d_offsets[None, :] * stride_sev_d,
            mask=mask_md,
            other=0.0,
        ).to(tl.float32)
        s_t = tl.where(token_valid, s_t, 0.0)

        exp_m = tl.exp(tl.minimum(m_t, clamp_max))
        sum_exp = l_t * exp_m
        sum_exp_v = sumexpv_t
        exp_m_safe = tl.where(exp_m > 0, exp_m, 1.0)
        n_t = sum_exp_v / exp_m_safe[:, None]

        s_max = tl.max(s_t, axis=0)
        s_exp = tl.exp(s_t - s_max)
        s_exp = tl.where(mask_m, s_exp, 0.0)
        s_sum = tl.sum(s_exp, axis=0)
        P_t = s_exp / (s_sum + 1e-20)
        P_t = P_t * valid_f

        expA_t = P_t / (sum_exp + eps)
        dO_t = tl.load(
            dO_ptr + pid_b * stride_do_b + (chunk_start + t) * stride_do_n + pid_h * stride_do_h + d_offsets * stride_do_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        dO_t = dO_t * valid_f

        d_sumexp_v = expA_t[:, None] * dO_t[None, :]
        d_expA = tl.sum(sum_exp_v * dO_t[None, :], axis=1)
        inv_den = 1.0 / (sum_exp + eps)
        d_sumexp = -d_expA * P_t * (inv_den * inv_den)

        d_l_out = d_sumexp * exp_m
        d_exp_m = d_sumexp * l_t + tl.sum(d_sumexp_v * n_t, axis=1)
        clamp_mask = m_t <= clamp_max
        d_m_out = d_exp_m * exp_m * clamp_mask
        d_n_out = d_sumexp_v * exp_m[:, None]

        dP = d_expA * inv_den
        dP_dot = tl.sum(dP * P_t, axis=0)
        d_s_soft = P_t * (dP - dP_dot)

        d_m_total = d_m + d_m_out
        d_l_total = d_l + d_l_out
        d_n_total = d_n + d_n_out

        if t > 0:
            m_prev = tl.load(
                M_ptr + pid_bh * stride_m_bh + chunk_idx * stride_m_chunk + (t - 1) * stride_m_t + m_offsets * stride_m_m,
                mask=mask_m,
                other=-float("inf"),
            ).to(tl.float32)
            l_prev = tl.load(
                L_ptr + pid_bh * stride_l_bh + chunk_idx * stride_l_chunk + (t - 1) * stride_l_t + m_offsets * stride_l_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            sumexpv_prev = tl.load(
                SumExpV_ptr
                + pid_bh * stride_sev_bh
                + chunk_idx * stride_sev_chunk
                + (t - 1) * stride_sev_t
                + m_offsets[:, None] * stride_sev_m
                + d_offsets[None, :] * stride_sev_d,
                mask=mask_md,
                other=0.0,
            ).to(tl.float32)
            exp_m_prev = tl.exp(tl.minimum(m_prev, clamp_max))
            exp_m_prev_safe = tl.where(exp_m_prev > 0, exp_m_prev, 1.0)
            n_prev = sumexpv_prev / exp_m_prev_safe[:, None]
        else:
            m_prev = prefix_max
            l_prev = prefix_den
            n_prev = prefix_num

        exp_prev = tl.where(token_valid, tl.exp(m_prev - m_t), 1.0)
        exp_s = tl.where(token_valid, tl.exp(s_t - m_t), 0.0)

        d_exp_prev = d_l_total * l_prev + tl.sum(d_n_total * n_prev, axis=1)
        v_t = tl.load(
            V_ptr + pid_b * stride_v_b + (chunk_start + t) * stride_v_n + pid_h * stride_v_h + d_offsets * stride_v_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        d_exp_s = d_l_total + tl.sum(d_n_total * v_t[None, :], axis=1)

        d_l_prev = d_l_total * exp_prev
        d_n_prev = d_n_total * exp_prev[:, None]
        d_m_prev = d_exp_prev * exp_prev
        d_m_t_from_exp = -(d_exp_prev * exp_prev + d_exp_s * exp_s)
        d_s_from_exp = d_exp_s * exp_s

        d_m_total = d_m_total + d_m_t_from_exp
        mask_prev = m_prev >= s_t
        d_m_prev = d_m_prev + d_m_total * mask_prev
        d_s_from_max = d_m_total * (~mask_prev)

        d_s = d_s_soft + d_s_from_exp + d_s_from_max

        dV_t = tl.sum(d_n_total * exp_s[:, None], axis=0)
        k_t = tl.load(
            K_ptr + pid_b * stride_k_b + (chunk_start + t) * stride_k_n + pid_h * stride_k_h + d_offsets * stride_k_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        dK_t = tl.sum(d_s[:, None] * Q_vals, axis=0) * scale
        dQ_acc = d_s[:, None] * k_t[None, :] * scale

        dQ_base_ptr = dQ_ptr + pid_h * stride_dq_h
        tl.atomic_add(
            dQ_base_ptr + m_offsets[:, None] * stride_dq_m + d_offsets[None, :] * stride_dq_d,
            dQ_acc * valid_f,
            mask=mask_md,
        )

        dK_base_ptr = dK_ptr + pid_b * stride_dk_b + (chunk_start + t) * stride_dk_n + pid_h * stride_dk_h
        tl.store(
            dK_base_ptr + d_offsets * stride_dk_d,
            dK_t * valid_f,
            mask=mask_d,
        )
        dV_base_ptr = dV_ptr + pid_b * stride_dv_b + (chunk_start + t) * stride_dv_n + pid_h * stride_dv_h
        tl.store(
            dV_base_ptr + d_offsets * stride_dv_d,
            dV_t * valid_f,
            mask=mask_d,
        )

        d_m = tl.where(token_valid, d_m_prev, d_m)
        d_l = tl.where(token_valid, d_l_prev, d_l)
        d_n = tl.where(token_valid, d_n_prev, d_n)
        t_rev += 1

    tl.store(dPmax_ptr + pid_bh * stride_dpmax_bh + chunk_idx * stride_dpmax_chunk + m_offsets * stride_dpmax_m, d_m, mask=mask_m)
    tl.store(dPden_ptr + pid_bh * stride_dpden_bh + chunk_idx * stride_dpden_chunk + m_offsets * stride_dpden_m, d_l, mask=mask_m)
    tl.store(
        dPnum_ptr + pid_bh * stride_dpnum_bh + chunk_idx * stride_dpnum_chunk + m_offsets[:, None] * stride_dpnum_m + d_offsets[None, :] * stride_dpnum_d,
        d_n,
        mask=mask_md,
    )


@triton.jit
def flare_chunk_fwd3_stable_debug(
    K_ptr, Q_ptr, V_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    DebugS_ptr, DebugM_ptr, DebugL_ptr, DebugDen_ptr, DebugA_ptr, DebugO_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    stride_dbg_t, stride_dbg_m,
    stride_dbg_o_t, stride_dbg_o_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale, eps, clamp_max,
    CHUNK_SIZE: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH:
        return
    if pid_bh != 0 or chunk_idx != 1:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    chunk_start = chunk_idx * CHUNK_SIZE

    c_offsets = tl.arange(0, CHUNK_SIZE)
    m_offsets = tl.arange(0, M)
    d_offsets = tl.arange(0, D)

    token_idx = chunk_start + c_offsets
    token_mask = token_idx < N  # [C]
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_md = mask_m[:, None] & mask_d[None, :]
    mask_kv = token_mask[:, None] & mask_d[None, :]

    # Load Q [M, D]
    Q_base_ptr = Q_ptr + pid_h * stride_q_h
    Q_vals = tl.load(
        Q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets[None, :] * stride_q_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)

    # Load prefix stats
    pmax_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + chunk_idx * stride_pmax_chunk
    pden_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + chunk_idx * stride_pden_chunk
    pnum_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + chunk_idx * stride_pnum_chunk

    prefix_max = tl.load(pmax_ptr + m_offsets * stride_pmax_m, mask=mask_m, other=-float("inf")).to(tl.float32)
    prefix_den = tl.load(pden_ptr + m_offsets * stride_pden_m, mask=mask_m, other=0.0).to(tl.float32)
    prefix_num = tl.load(
        pnum_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)

    m_state = prefix_max
    l_state = prefix_den
    n_state = prefix_num

    # Load K/V base
    K_base_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + chunk_start * stride_k_n
    V_base_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + chunk_start * stride_v_n

    t = 0
    while t < CHUNK_SIZE:
        token_valid = (chunk_start + t) < N
        k_t = tl.load(
            K_base_ptr + t * stride_k_n + d_offsets * stride_k_d,
            mask=token_valid & mask_d,
            other=0.0,
        ).to(tl.float32)
        v_t = tl.load(
            V_base_ptr + t * stride_v_n + d_offsets * stride_v_d,
            mask=token_valid & mask_d,
            other=0.0,
        ).to(tl.float32)

        s_t = tl.sum(Q_vals * k_t[None, :], axis=1) * scale
        s_t = tl.where(token_valid & mask_m, s_t, -float("inf"))

        m_new = tl.maximum(m_state, s_t)
        is_m_state_inf = m_state == -float("inf")
        is_m_new_inf = m_new == -float("inf")
        m_new_safe = tl.where(is_m_new_inf, 0.0, m_new)

        exp_prev = tl.where(
            is_m_state_inf & is_m_new_inf,
            1.0,
            tl.where(is_m_state_inf, 0.0, tl.exp(m_state - m_new_safe)),
        )
        exp_s = tl.where(is_m_new_inf, 0.0, tl.exp(s_t - m_new_safe))

        l_state = l_state * exp_prev + exp_s
        n_state = n_state * exp_prev[:, None] + exp_s[:, None] * v_t[None, :]
        m_state = m_new

        exp_m = tl.exp(tl.minimum(m_state, clamp_max))
        sum_exp = l_state * exp_m
        sum_exp_v = n_state * exp_m[:, None]
        den_total_t = sum_exp

        s_max = tl.max(s_t, axis=0)
        s_exp = tl.exp(s_t - s_max)
        s_exp = tl.where(token_valid & mask_m, s_exp, 0.0)
        s_sum = tl.sum(s_exp, axis=0)
        P_t = tl.where(token_valid, s_exp / (s_sum + 1e-20), 0.0)
        expA_t = P_t / (den_total_t + eps)
        o_t = tl.sum(expA_t[:, None] * sum_exp_v, axis=0)

        tl.store(DebugS_ptr + t * stride_dbg_t + m_offsets * stride_dbg_m, s_t, mask=mask_m)
        tl.store(DebugM_ptr + t * stride_dbg_t + m_offsets * stride_dbg_m, m_state, mask=mask_m)
        tl.store(DebugL_ptr + t * stride_dbg_t + m_offsets * stride_dbg_m, l_state, mask=mask_m)
        tl.store(DebugDen_ptr + t * stride_dbg_t + m_offsets * stride_dbg_m, den_total_t, mask=mask_m)
        tl.store(DebugA_ptr + t * stride_dbg_t + m_offsets * stride_dbg_m, expA_t, mask=mask_m)
        t += 1
        tl.store(
            DebugO_ptr + t * stride_dbg_o_t + d_offsets * stride_dbg_o_d,
            o_t,
            mask=mask_d,
        )


@triton.jit
def flare_chunk_bwd(
    K_ptr, Q_ptr, V_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    dPmax_ptr, dPden_ptr, dPnum_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_dq_h, stride_dq_m, stride_dq_d,
    stride_dk_b, stride_dk_n, stride_dk_h, stride_dk_d,
    stride_dv_b, stride_dv_n, stride_dv_h, stride_dv_d,
    stride_dpmax_bh, stride_dpmax_chunk, stride_dpmax_m,
    stride_dpden_bh, stride_dpden_chunk, stride_dpden_m,
    stride_dpnum_bh, stride_dpnum_chunk, stride_dpnum_m, stride_dpnum_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale, eps, clamp_max,
    CHUNK_SIZE: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    USE_DENSE_DDEN: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    chunk_start = chunk_idx * CHUNK_SIZE

    c_offsets = tl.arange(0, CHUNK_SIZE)
    m_offsets = tl.arange(0, M)
    d_offsets = tl.arange(0, D)

    token_idx = chunk_start + c_offsets
    token_mask = token_idx < N
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_md = mask_m[:, None] & mask_d[None, :]
    mask_kv = token_mask[:, None] & mask_d[None, :]

    state_dtype = tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32)
    stats_dtype = tl.float32 if USE_FP32_STATS else state_dtype

    Q_base_ptr = Q_ptr + pid_h * stride_q_h
    Q_vals = tl.load(
        Q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets[None, :] * stride_q_d,
        mask=mask_md,
        other=0.0,
    ).to(state_dtype)
    Q_vals_f = Q_vals.to(tl.float32)

    pmax_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + chunk_idx * stride_pmax_chunk
    pden_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + chunk_idx * stride_pden_chunk
    pnum_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + chunk_idx * stride_pnum_chunk

    prefix_max_raw = tl.load(pmax_ptr + m_offsets * stride_pmax_m, mask=mask_m, other=-float("inf")).to(stats_dtype)
    prefix_max_raw_f = prefix_max_raw.to(tl.float32)
    prefix_max_f = tl.minimum(prefix_max_raw_f, clamp_max)
    prefix_max = prefix_max_f.to(stats_dtype)
    prefix_den = tl.load(pden_ptr + m_offsets * stride_pden_m, mask=mask_m, other=0.0).to(stats_dtype)
    prefix_num = tl.load(
        pnum_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d,
        mask=mask_md,
        other=0.0,
    ).to(stats_dtype)

    exp_prev_max = tl.exp(prefix_max_f)
    sum_prev_exp = prefix_den.to(tl.float32) * exp_prev_max
    sum_prev_exp_v = prefix_num.to(tl.float32) * exp_prev_max[:, None]

    K_base_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + chunk_start * stride_k_n
    V_base_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + chunk_start * stride_v_n
    K_chunk = tl.load(
        K_base_ptr + c_offsets[:, None] * stride_k_n + d_offsets[None, :] * stride_k_d,
        mask=mask_kv,
        other=0.0,
    ).to(state_dtype)
    K_chunk_f = K_chunk.to(tl.float32)
    V_chunk = tl.load(
        V_base_ptr + c_offsets[:, None] * stride_v_n + d_offsets[None, :] * stride_v_d,
        mask=mask_kv,
        other=0.0,
    ).to(state_dtype)

    dO_base_ptr = dO_ptr + pid_b * stride_do_b + pid_h * stride_do_h + chunk_start * stride_do_n
    dO_chunk = tl.load(
        dO_base_ptr + c_offsets[:, None] * stride_do_n + d_offsets[None, :] * stride_do_d,
        mask=mask_kv,
        other=0.0,
    ).to(state_dtype)
    dO_chunk = tl.where(token_mask[:, None], dO_chunk, 0.0)

    S = tl.dot(K_chunk_f, tl.trans(Q_vals_f), out_dtype=tl.float32, allow_tf32=ALLOW_TF32) * scale
    S = tl.where(token_mask[:, None] & mask_m[None, :], S, -float("inf"))

    # Stable dense formulation (match forward non-stable branch)
    s_max = tl.max(S, axis=1)
    s_max = tl.where(token_mask, s_max, 0.0)
    s_exp = tl.exp(S - s_max[:, None])
    s_exp = tl.where(token_mask[:, None] & mask_m[None, :], s_exp, 0.0)

    s_max_clamped = tl.minimum(s_max, clamp_max)
    e = tl.exp(s_max_clamped)
    e = tl.where(token_mask, e, 0.0)

    A_scaled = s_exp * e[:, None]  # [C, M]
    den_chunk = tl.cumsum(A_scaled, axis=0)
    den_total = den_chunk + sum_prev_exp[None, :]

    s_sum = tl.sum(s_exp, axis=1)
    P = tl.where(token_mask[:, None], s_exp / (s_sum[:, None] + 1e-20), 0.0)

    inv_den = 1.0 / (den_total + eps)
    expA = P * inv_den
    expA = tl.where(token_mask[:, None] & mask_m[None, :], expA, 0.0)

    expA_dot = expA.to(tl.float32)
    expS_dot = A_scaled.to(tl.float32)
    dO_dot = dO_chunk.to(tl.float32)

    W = tl.dot(expA_dot, tl.trans(expS_dot), out_dtype=tl.float32, allow_tf32=ALLOW_TF32)
    causal = c_offsets[None, :] <= c_offsets[:, None]
    causal = causal & token_mask[:, None] & token_mask[None, :]
    W = tl.where(causal, W, 0.0)

    W_fp32 = W.to(tl.float32)
    dV_chunk = tl.dot(tl.trans(W_fp32), dO_dot, out_dtype=tl.float32, allow_tf32=ALLOW_TF32)

    dW = tl.dot(dO_dot, tl.trans(V_chunk.to(tl.float32)), out_dtype=tl.float32, allow_tf32=ALLOW_TF32).to(tl.float32)
    dW = tl.where(causal, dW, 0.0)
    dExpA = tl.dot(dW, expS_dot, out_dtype=tl.float32, allow_tf32=ALLOW_TF32) + tl.dot(
        dO_dot, tl.trans(sum_prev_exp_v.to(tl.float32)), out_dtype=tl.float32, allow_tf32=ALLOW_TF32
    )
    dExpS_from_W = tl.dot(tl.trans(dW), expA_dot, out_dtype=tl.float32, allow_tf32=ALLOW_TF32)

    dP = dExpA * inv_den
    dDen_total = -(dExpA * P) * (inv_den * inv_den)
    dP = tl.where(token_mask[:, None] & mask_m[None, :], dP, 0.0)
    dDen_total = tl.where(token_mask[:, None] & mask_m[None, :], dDen_total, 0.0)

    d_sum_prev_exp = tl.sum(dDen_total, axis=0)
    d_sum_prev_exp_v = tl.dot(tl.trans(expA_dot), dO_dot, out_dtype=tl.float32, allow_tf32=ALLOW_TF32).to(tl.float32)

    dPmax_ptr = dPmax_ptr + pid_bh * stride_dpmax_bh + chunk_idx * stride_dpmax_chunk
    dPden_ptr = dPden_ptr + pid_bh * stride_dpden_bh + chunk_idx * stride_dpden_chunk
    dPnum_ptr = dPnum_ptr + pid_bh * stride_dpnum_bh + chunk_idx * stride_dpnum_chunk

    d_prefix_den = d_sum_prev_exp * exp_prev_max
    d_prefix_num = d_sum_prev_exp_v * exp_prev_max[:, None]
    clamp_mask_pmax = prefix_max_raw_f <= clamp_max
    d_prefix_max = exp_prev_max * (
        d_sum_prev_exp * prefix_den.to(tl.float32) + tl.sum(d_sum_prev_exp_v * prefix_num.to(tl.float32), axis=1)
    )
    d_prefix_max = d_prefix_max * clamp_mask_pmax

    tl.store(dPmax_ptr + m_offsets * stride_dpmax_m, d_prefix_max, mask=mask_m)
    tl.store(dPden_ptr + m_offsets * stride_dpden_m, d_prefix_den, mask=mask_m)
    tl.store(
        dPnum_ptr + m_offsets[:, None] * stride_dpnum_m + d_offsets[None, :] * stride_dpnum_d,
        d_prefix_num,
        mask=mask_md,
    )

    if USE_DENSE_DDEN:
        u_idx = c_offsets[:, None]
        t_idx = c_offsets[None, :]
        L_t = tl.where(t_idx <= u_idx, 1.0, 0.0).to(tl.float32)  # [C, C] = L^T
        dExpS = dExpS_from_W + tl.dot(dDen_total, L_t, out_dtype=tl.float32, allow_tf32=ALLOW_TF32)
    else:
        prefix = tl.cumsum(dDen_total, axis=0)  # [C, M]
        sum_all = tl.sum(dDen_total, axis=0)  # [M]
        suffix = sum_all[None, :] - prefix + dDen_total  # [C, M]
        dExpS = dExpS_from_W + suffix  # [C, M] == dA_scaled

    dP_dot = tl.sum(dP * P, axis=1)  # [C]
    dS_softmax = P * (dP - dP_dot[:, None])  # [C, M]

    # dS from A_scaled with max-shift clamp handling.
    dS_from_A = dExpS * expS_dot
    clamped = s_max > clamp_max
    sum_dA_A = tl.sum(dS_from_A, axis=1)  # [C]
    mask_argmax = S == s_max[:, None]
    mask_argmax = mask_argmax & token_mask[:, None] & mask_m[None, :]
    clamped_f = clamped.to(tl.float32)
    dS_from_A = dS_from_A - mask_argmax * (sum_dA_A[:, None] * clamped_f[:, None])

    dS = dS_from_A + dS_softmax  # [C, M]
    dS = tl.where(token_mask[:, None] & mask_m[None, :], dS, 0.0)

    dK_chunk = tl.dot(dS, Q_vals_f, out_dtype=tl.float32, allow_tf32=ALLOW_TF32) * scale  # [C, D]
    dQ_acc = tl.dot(tl.trans(dS), K_chunk_f, out_dtype=tl.float32, allow_tf32=ALLOW_TF32) * scale  # [M, D]

    dQ_base_ptr = dQ_ptr + pid_h * stride_dq_h
    tl.atomic_add(
        dQ_base_ptr + m_offsets[:, None] * stride_dq_m + d_offsets[None, :] * stride_dq_d,
        dQ_acc,
        mask=mask_md,
    )

    dK_base_ptr = dK_ptr + pid_b * stride_dk_b + pid_h * stride_dk_h + chunk_start * stride_dk_n
    tl.store(
        dK_base_ptr + c_offsets[:, None] * stride_dk_n + d_offsets[None, :] * stride_dk_d,
        dK_chunk,
        mask=mask_kv,
    )

    dV_base_ptr = dV_ptr + pid_b * stride_dv_b + pid_h * stride_dv_h + chunk_start * stride_dv_n
    tl.store(
        dV_base_ptr + c_offsets[:, None] * stride_dv_n + d_offsets[None, :] * stride_dv_d,
        dV_chunk,
        mask=mask_kv,
    )

@triton.jit
def flare_chunk_stats_bwd3(
    K_ptr, Q_ptr, V_ptr,
    dCmax_ptr, dCden_ptr, dCnum_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_dcmax_bh, stride_dcmax_chunk, stride_dcmax_m,
    stride_dcden_bh, stride_dcden_chunk, stride_dcden_m,
    stride_dcnum_bh, stride_dcnum_chunk, stride_dcnum_m, stride_dcnum_d,
    stride_dq_h, stride_dq_m, stride_dq_d,
    stride_dk_b, stride_dk_n, stride_dk_h, stride_dk_d,
    stride_dv_b, stride_dv_n, stride_dv_h, stride_dv_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale,
    CHUNK_SIZE: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    chunk_start = chunk_idx * CHUNK_SIZE

    c_offsets = tl.arange(0, CHUNK_SIZE)
    m_offsets = tl.arange(0, M)
    d_offsets = tl.arange(0, D)

    token_idx = chunk_start + c_offsets
    token_mask = token_idx < N
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_md = mask_m[:, None] & mask_d[None, :]
    mask_kv = token_mask[:, None] & mask_d[None, :]
    mask_cm = token_mask[:, None] & mask_m[None, :]

    state_dtype = tl.float32 if USE_FP32_STATS else (tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))

    Q_base_ptr = Q_ptr + pid_h * stride_q_h
    Q_vals = tl.load(
        Q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets[None, :] * stride_q_d,
        mask=mask_md,
        other=0.0,
    ).to(state_dtype)
    Q_vals_f = Q_vals.to(tl.float32)

    K_base_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + chunk_start * stride_k_n
    V_base_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + chunk_start * stride_v_n
    K_chunk = tl.load(
        K_base_ptr + c_offsets[:, None] * stride_k_n + d_offsets[None, :] * stride_k_d,
        mask=mask_kv,
        other=0.0,
    ).to(state_dtype)
    V_chunk = tl.load(
        V_base_ptr + c_offsets[:, None] * stride_v_n + d_offsets[None, :] * stride_v_d,
        mask=mask_kv,
        other=0.0,
    ).to(state_dtype)
    K_chunk_f = K_chunk.to(tl.float32)
    V_chunk_f = V_chunk.to(tl.float32)

    dCmax_ptr = dCmax_ptr + pid_bh * stride_dcmax_bh + chunk_idx * stride_dcmax_chunk
    dCden_ptr = dCden_ptr + pid_bh * stride_dcden_bh + chunk_idx * stride_dcden_chunk
    dCnum_ptr = dCnum_ptr + pid_bh * stride_dcnum_bh + chunk_idx * stride_dcnum_chunk

    dCmax = tl.load(dCmax_ptr + m_offsets * stride_dcmax_m, mask=mask_m, other=0.0).to(tl.float32)
    dCden = tl.load(dCden_ptr + m_offsets * stride_dcden_m, mask=mask_m, other=0.0).to(tl.float32)
    dCnum = tl.load(
        dCnum_ptr + m_offsets[:, None] * stride_dcnum_m + d_offsets[None, :] * stride_dcnum_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)

    S = tl.dot(K_chunk_f, tl.trans(Q_vals_f), out_dtype=tl.float32, allow_tf32=False) * scale
    S = tl.where(mask_cm, S, -float("inf"))
    chunk_max = tl.max(S, axis=0)

    expS = tl.exp(S - chunk_max[None, :])
    expS = tl.where(mask_cm, expS, 0.0)

    dE = dCden[None, :] + tl.dot(V_chunk_f, tl.trans(dCnum), out_dtype=tl.float32, allow_tf32=False)
    dS_e = dE * expS
    sum_dS_e = tl.sum(dS_e, axis=0)
    dM = dCmax - sum_dS_e
    mask_max = (S == chunk_max[None, :]) & mask_cm
    dS = dS_e + dM[None, :] * mask_max

    dV_chunk = tl.dot(expS, dCnum, out_dtype=tl.float32, allow_tf32=False)
    dK_chunk = tl.dot(dS, Q_vals_f, out_dtype=tl.float32, allow_tf32=False) * scale
    dQ_acc = tl.dot(tl.trans(dS), K_chunk_f, out_dtype=tl.float32, allow_tf32=False) * scale

    dQ_base_ptr = dQ_ptr + pid_h * stride_dq_h
    tl.atomic_add(
        dQ_base_ptr + m_offsets[:, None] * stride_dq_m + d_offsets[None, :] * stride_dq_d,
        dQ_acc,
        mask=mask_md,
    )

    dK_base_ptr = dK_ptr + pid_b * stride_dk_b + pid_h * stride_dk_h + chunk_start * stride_dk_n
    tl.store(
        dK_base_ptr + c_offsets[:, None] * stride_dk_n + d_offsets[None, :] * stride_dk_d,
        dK_chunk,
        mask=mask_kv,
    )

    dV_base_ptr = dV_ptr + pid_b * stride_dv_b + pid_h * stride_dv_h + chunk_start * stride_dv_n
    tl.store(
        dV_base_ptr + c_offsets[:, None] * stride_dv_n + d_offsets[None, :] * stride_dv_d,
        dV_chunk,
        mask=mask_kv,
    )

#======================================================================#
# Dense implementation
#======================================================================#
def flare_causal_pytorch_dense1(Q, K, V, scale=1.0, eps=None, profile: bool = False):
    assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
        "Q, K, V must be 3D and 4D tensors respectively "
        f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
    )
    assert K.size() == V.size(), f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}"
    assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
        "Expected Q [H, M, D] and K/V [B, N, H, D]. "
        f"Got Q.shape={Q.shape} and K.shape={K.shape}"
    )
    H_q, M_q, D_q = Q.size()
    B_k, N_k, H_k, D_k = K.size()
    assert H_q == H_k and D_q == D_k, "Incompatible K/V dimensions"

    device = Q.device
    out_dtype = Q.dtype
    compute_dtype = torch.float32
    if eps is None:
        eps = _get_eps_for_dtype(Q.dtype)

    Q_f = Q.to(compute_dtype)
    K_f = K.to(compute_dtype)
    V_f = V.to(compute_dtype)

    K_bhnd = K_f.permute(0, 2, 1, 3).contiguous()  # [B, H, N, D]
    V_bhnd = V_f.permute(0, 2, 1, 3).contiguous()  # [B, H, N, D]

    #---------------------------------------------------------------#
    # extremely stable
    #---------------------------------------------------------------#
    S = scale * (Q_f @ K_bhnd.mT) # [B H M N]
    P = torch.softmax(S, dim=-2)

    # R_{mt} = \max_{u<=t} S_{mu}
    # L_{mt} = \sum_{u<=t} \exp(S_{mu} - R_{mt})
    # D_{mt} = \exp(R_{mt}) * L_{mt}
    R = torch.full_like(S, -torch.inf)
    L = torch.zeros_like(S)
    
    R_prev = R[..., 0].clone() # cloning only for autograd
    L_prev = L[..., 0].clone()

    for t in range(N_k):
        s_t = S[..., t].to(torch.float32)                  # [B,H,M]
        r_t = torch.maximum(R_prev, s_t)                   # [B,H,M]
        L_t = L_prev * torch.exp(R_prev - r_t) + torch.exp(s_t - r_t)
        R[..., t] = r_t
        L[..., t] = L_t
        R_prev, L_prev = r_t, L_t

    K = R.unsqueeze(-2) - R.unsqueeze(-1) # [B,H,M,N,N]
    W = ((P / (L + eps)).unsqueeze(-1) * torch.exp(S - R).unsqueeze(-2) * torch.exp(K)).sum(dim=-3)

    causal = _get_causal_mask(N_k, device)
    W = W.masked_fill(~causal[None, :, :], 0.0)
    Yc = W @ V_bhnd

    #---------------------------------------------------------------#
    Y = Yc.reshape(B_k, H_k, N_k, D_k).permute(0, 2, 1, 3).to(out_dtype)

    return Y

def flare_causal_pytorch_dense(Q, K, V, scale=1.0, eps=None, profile: bool = False):
    assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
        "Q, K, V must be 3D and 4D tensors respectively "
        f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
    )
    assert K.size() == V.size(), f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}"
    assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
        "Expected Q [H, M, D] and K/V [B, N, H, D]. "
        f"Got Q.shape={Q.shape} and K.shape={K.shape}"
    )
    H_q, M_q, D_q = Q.size()
    B_k, N_k, H_k, D_k = K.size()
    assert H_q == H_k and D_q == D_k, "Incompatible K/V dimensions"

    device = Q.device
    out_dtype = Q.dtype
    compute_dtype = torch.float32
    if eps is None:
        eps = _get_eps_for_dtype(Q.dtype)

    Q_f = Q.to(compute_dtype)
    K_f = K.to(compute_dtype)
    V_f = V.to(compute_dtype)

    K_bhnd = K_f.permute(0, 2, 1, 3).contiguous()  # [B, H, N, D]
    V_bhnd = V_f.permute(0, 2, 1, 3).contiguous()  # [B, H, N, D]

    #---------------------------------------------------------------#
    # Original (can be numerically unstable)
    #---------------------------------------------------------------#
    S = scale * (Q_f @ K_bhnd.mT) # [B H M N]
    A = torch.exp(S)
    D = torch.cumsum(A, dim=-1)
    P = torch.softmax(S, dim=-2)
    W = (P / (D + eps)).mT @ A

    causal = _get_causal_mask(N_k, device)
    W = W.masked_fill(~causal[None, :, :], 0.0)
    Yc = W @ V_bhnd

    #---------------------------------------------------------------#
    Y = Yc.reshape(B_k, H_k, N_k, D_k).permute(0, 2, 1, 3).to(out_dtype)

    return Y

class DenseFLARE(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale=1.0):
        if Q.dim() != 3 or K.dim() != 4 or V.dim() != 4:
            raise ValueError(
                "Dense FLARE expects Q [H, M, D] and K/V [B, N, H, D]. "
                f"Got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
            )
        if K.size() != V.size():
            raise ValueError(f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}")
        if Q.size(0) != K.size(2) or Q.size(2) != K.size(3):
            raise ValueError(
                "Expected Q [H, M, D] and K/V [B, N, H, D] with matching H,D. "
                f"Got Q.shape={Q.shape} and K.shape={K.shape}"
            )

        H, M, D = Q.size()
        B, N, _, _ = K.size()
        if (M % 16) != 0 or (D % 16) != 0 or (N % 16) != 0:
            raise ValueError(f"DenseFLARE requires M, N, D be multiples of 16. Got M={M}, N={N}, D={D}")

        K_bhnd = K.permute(0, 2, 1, 3).contiguous()
        V_bhnd = V.permute(0, 2, 1, 3).contiguous()

        block_m = M
        block_d = D
        block_n = N

        Y = torch.empty((B, H, N, D), device=Q.device, dtype=Q.dtype)
        grid = (B * H,)

        num_warps = 4 if block_m <= 64 else 8
        eps = _get_eps_for_dtype(Q.dtype)
        clamp_max = _get_exp_clamp_for_dtype(Q.dtype)
        flare_dense_fwd_kernel[grid](
            Q, K_bhnd, V_bhnd, Y,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
            V_bhnd.stride(0), V_bhnd.stride(1), V_bhnd.stride(2), V_bhnd.stride(3),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            B, H, M,
            N,
            scale,
            eps,
            clamp_max,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )
        ctx.save_for_backward(Q, K, V)
        ctx.scale = scale
        return Y.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def backward(ctx, *grad_outputs):
        (Q, K, V) = ctx.saved_tensors
        scale = ctx.scale
        dY = grad_outputs[0]
        if dY is None:
            return None, None, None, None

        H, M, D = Q.size()
        B, N, _, _ = K.size()
        if (M % 16) != 0 or (D % 16) != 0 or (N % 16) != 0:
            raise ValueError(f"DenseFLARE requires M, N, D be multiples of 16. Got M={M}, N={N}, D={D}")

        K_bhnd = K.permute(0, 2, 1, 3).contiguous()
        V_bhnd = V.permute(0, 2, 1, 3).contiguous()
        dY_bhnd = dY.permute(0, 2, 1, 3).contiguous()

        dQ = torch.zeros_like(Q)
        dK = torch.empty_like(K_bhnd)
        dV = torch.empty_like(V_bhnd)

        block_m = M
        block_d = D
        block_n = N
        grid = (B * H,)

        num_warps = 4 if block_m <= 64 else 8
        eps = _get_eps_for_dtype(Q.dtype)
        flare_dense_bwd_kernel[grid](
            Q, K_bhnd, V_bhnd, dY_bhnd,
            dQ, dK, dV,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
            V_bhnd.stride(0), V_bhnd.stride(1), V_bhnd.stride(2), V_bhnd.stride(3),
            dY_bhnd.stride(0), dY_bhnd.stride(1), dY_bhnd.stride(2), dY_bhnd.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
            B, H, M,
            N,
            scale,
            eps,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )

        out_dtype = Q.dtype
        return dQ.to(out_dtype), dK.permute(0, 2, 1, 3).contiguous().to(out_dtype), dV.permute(0, 2, 1, 3).contiguous().to(out_dtype), None

class DenseFLARE1(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale=1.0):
        if Q.dim() != 3 or K.dim() != 4 or V.dim() != 4:
            raise ValueError(
                "DenseFLARE1 expects Q [H, M, D] and K/V [B, N, H, D]. "
                f"Got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
            )
        if K.size() != V.size():
            raise ValueError(f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}")
        if Q.size(0) != K.size(2) or Q.size(2) != K.size(3):
            raise ValueError(
                "Expected Q [H, M, D] and K/V [B, N, H, D] with matching H,D. "
                f"Got Q.shape={Q.shape} and K.shape={K.shape}"
            )

        H, M, D = Q.size()
        B, N, _, _ = K.size()
        if (M % 16) != 0 or (D % 16) != 0 or (N % 16) != 0:
            raise ValueError(f"DenseFLARE1 requires M, N, D be multiples of 16. Got M={M}, N={N}, D={D}")

        K_bhnd = K.permute(0, 2, 1, 3).contiguous()
        V_bhnd = V.permute(0, 2, 1, 3).contiguous()

        block_m = M
        block_d = D
        block_n = N

        Y = torch.empty((B, H, N, D), device=Q.device, dtype=Q.dtype)
        grid = (B * H,)

        num_warps = 4 if block_m <= 64 else 8
        eps = _get_eps_for_dtype(Q.dtype)
        flare_dense1_fwd_kernel[grid](
            Q, K_bhnd, V_bhnd, Y,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
            V_bhnd.stride(0), V_bhnd.stride(1), V_bhnd.stride(2), V_bhnd.stride(3),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            B, H, M,
            N,
            scale,
            eps,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        )
        return Y.permute(0, 2, 1, 3).contiguous()

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("DenseFLARE1 backward not implemented.")


def _denseflare1_phase_bench(Q, K, V, scale=1.0):
    H, M, D = Q.size()
    B, N, _, _ = K.size()
    if (M % 16) != 0 or (D % 16) != 0 or (N % 16) != 0:
        raise ValueError(f"DenseFLARE1 requires M, N, D be multiples of 16. Got M={M}, N={N}, D={D}")

    K_bhnd = K.permute(0, 2, 1, 3).contiguous()
    V_bhnd = V.permute(0, 2, 1, 3).contiguous()

    block_m = M
    block_d = D
    block_n = N
    grid = (B * H,)
    num_warps = 4 if block_m <= 64 else 8
    eps = _get_eps_for_dtype(Q.dtype)

    S = torch.empty((B, H, M, N), device=Q.device, dtype=torch.float32)
    P = torch.empty((B, H, M, N), device=Q.device, dtype=torch.float32)
    X = torch.empty((B, H, M, N), device=Q.device, dtype=torch.float32)
    Y = torch.empty((B, H, N, D), device=Q.device, dtype=Q.dtype)

    t_phase1 = triton.testing.do_bench(
        lambda: flare_dense1_phase1_kernel[grid](
            Q, K_bhnd, S, P,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
            S.stride(0), S.stride(1), S.stride(2), S.stride(3),
            P.stride(0), P.stride(1), P.stride(2), P.stride(3),
            B, H, M,
            N,
            scale,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        ),
        warmup=2,
        rep=2,
    )

    t_phase2 = triton.testing.do_bench(
        lambda: flare_dense1_phase2_kernel[grid](
            Q, K_bhnd, X,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
            X.stride(0), X.stride(1), X.stride(2), X.stride(3),
            B, H, M,
            N,
            scale,
            eps,
            D=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        ),
        warmup=2,
        rep=2,
    )

    t_phase3 = triton.testing.do_bench(
        lambda: flare_dense1_phase3_kernel[grid](
            S, X, V_bhnd, Y,
            S.stride(0), S.stride(1), S.stride(2), S.stride(3),
            X.stride(0), X.stride(1), X.stride(2), X.stride(3),
            V_bhnd.stride(0), V_bhnd.stride(1), V_bhnd.stride(2), V_bhnd.stride(3),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            B, H, M,
            N,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        ),
        warmup=2,
        rep=2,
    )

    return t_phase1, t_phase2, t_phase3

@triton.jit
def flare_dense_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Y_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_yb, stride_yh, stride_yt, stride_yd,
    B, H, M,
    T,
    scale,
    eps,
    clamp_max,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    # Load full Q and K blocks.
    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block0 = tl.load(q_ptr0, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    k_ptr0 = k_base + n_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
    k_block0 = tl.load(k_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # S = Q @ K^T  -> [M, N]
    S = tl.dot(q_block0, tl.trans(k_block0), out_dtype=tl.float32, allow_tf32=False) * scale
    S = tl.where(mask_m[:, None] & mask_n[None, :], S, -float("inf"))

    # Max-shift per column for stability.
    s_max = tl.max(S, axis=0)
    s_max = tl.minimum(s_max, clamp_max)
    exp_s = tl.exp(S - s_max[None, :])
    exp_s = tl.where(mask_m[:, None] & mask_n[None, :], exp_s, 0.0)

    # l_s = tl.sum(exp_s, axis=0)
    # l_s = tl.where(l_s > 0, l_s, 1.0)
    # P = exp_s / l_s[None, :]
    P = tl.softmax(S, dim=0)

    # # A' = exp(S - max), e = exp(max - max_global) for stability.
    A_prime = exp_s
    s_max_global = tl.max(s_max, axis=0)
    e = tl.exp(s_max - s_max_global)
    eps_scaled = eps * tl.exp(-s_max_global)

    # # Lower-triangular mask L (u <= t) for prefix sums.
    u_idx = n_offsets[:, None]
    t_idx = n_offsets[None, :]
    L = tl.where(u_idx <= t_idx, 1.0, 0.0).to(tl.float32)  # [N, N]

    # # D = A' @ (diag(e) @ L) -> scale columns by e on rows of L.
    L_e = L * e[:, None]
    D_mat = tl.dot(A_prime, L_e, out_dtype=tl.float32, allow_tf32=False)

    # # W = (P / (D + eps))^T @ (A' * e)
    E = P / (D_mat + eps_scaled)
    A_scaled = A_prime * e[None, :]
    W = tl.dot(tl.trans(E), A_scaled, out_dtype=tl.float32, allow_tf32=False)

    # # Apply causal mask (t >= u) and compute Y = W @ V.
    t_idx_rows = n_offsets[:, None]
    u_idx_cols = n_offsets[None, :]
    W = tl.where(t_idx_rows >= u_idx_cols, W, 0.0)

    v_ptr0 = v_base + n_offsets[:, None] * stride_vt + d_offsets[None, :] * stride_vd
    v_block0 = tl.load(v_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    Y_block = tl.dot(W, v_block0, out_dtype=tl.float32, allow_tf32=False)  # [N, D]

    y_ptr0 = y_base + n_offsets[:, None] * stride_yt + d_offsets[None, :] * stride_yd
    tl.store(y_ptr0, Y_block, mask=mask_n[:, None] & mask_d[None, :])

@triton.jit
def flare_dense1_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Y_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_yb, stride_yh, stride_yt, stride_yd,
    B, H, M,
    T,
    scale,
    eps,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D
    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block = tl.load(q_ptr0, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    k_ptr0 = k_base + n_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
    k_block = tl.load(k_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    v_ptr0 = v_base + n_offsets[:, None] * stride_vt + d_offsets[None, :] * stride_vd
    v_block = tl.load(v_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # S = Q @ K^T -> [M, N]
    S = tl.dot(q_block, tl.trans(k_block), out_dtype=tl.float32, allow_tf32=False) * scale
    S = tl.where(mask_m[:, None] & mask_n[None, :], S, -float("inf"))

    # Build X[:, t] online (no R/L materialization).
    X = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
    r_prev = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_prev = tl.zeros([BLOCK_M], tl.float32)
    n_idx = n_offsets[None, :]

    for t in tl.static_range(0, BLOCK_N):
        valid_t = t < T
        k_t_ptr = k_base + t * stride_kt + d_offsets * stride_kd
        k_t = tl.load(k_t_ptr, mask=mask_d & valid_t, other=0.0).to(tl.float32)
        s_t = tl.sum(q_block * k_t[None, :], axis=1) * scale
        s_t = tl.where(mask_m & valid_t, s_t, -float("inf"))
        p_t = tl.softmax(s_t, dim=0)
        r_t = tl.maximum(r_prev, s_t)
        l_t = l_prev * tl.exp(r_prev - r_t) + tl.exp(s_t - r_t)
        x_t = p_t / (l_t + eps) * tl.exp(-r_t)
        col_mask = n_idx == t
        X = tl.where(col_mask & valid_t, x_t[:, None], X)
        r_prev = tl.where(valid_t, r_t, r_prev)
        l_prev = tl.where(valid_t, l_t, l_prev)

    mu = tl.max(S, axis=0)
    mu = tl.where(mask_n, mu, 0.0)
    Btilde = tl.exp(tl.where(mask_m[:, None] & mask_n[None, :], S - mu[None, :], -float("inf")))

    W = tl.dot(tl.trans(X), Btilde, out_dtype=tl.float32, allow_tf32=False)

    t_idx = n_offsets[:, None]
    u_idx = n_offsets[None, :]
    W = tl.where(t_idx >= u_idx, W, 0.0)

    exp_mu = tl.exp(mu)
    v_scaled = v_block * exp_mu[:, None]
    Y_block = tl.dot(W, v_scaled, out_dtype=tl.float32, allow_tf32=False)
    y_ptr0 = y_base + n_offsets[:, None] * stride_yt + d_offsets[None, :] * stride_yd
    tl.store(y_ptr0, Y_block, mask=mask_n[:, None] & mask_d[None, :])


@triton.jit
def flare_dense1_phase1_kernel(
    Q_ptr,
    K_ptr,
    S_ptr,
    P_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_sb, stride_sh, stride_sm, stride_st,
    stride_pb, stride_ph, stride_pm, stride_pt,
    B, H, M,
    T,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D
    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh

    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block = tl.load(q_ptr0, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    k_ptr0 = k_base + n_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
    k_block = tl.load(k_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    S = tl.dot(q_block, tl.trans(k_block), out_dtype=tl.float32, allow_tf32=False) * scale
    S = tl.where(mask_m[:, None] & mask_n[None, :], S, -float("inf"))
    P = tl.softmax(S, dim=0)

    s_base = S_ptr + b * stride_sb + h * stride_sh
    p_base = P_ptr + b * stride_pb + h * stride_ph
    s_ptr0 = s_base + m_offsets[:, None] * stride_sm + n_offsets[None, :] * stride_st
    p_ptr0 = p_base + m_offsets[:, None] * stride_pm + n_offsets[None, :] * stride_pt
    tl.store(s_ptr0, S, mask=mask_m[:, None] & mask_n[None, :])
    tl.store(p_ptr0, P, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def flare_dense1_phase2_kernel(
    Q_ptr,
    K_ptr,
    X_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_xb, stride_xh, stride_xm, stride_xt,
    B, H, M,
    T,
    scale,
    eps,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh - b * H

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    x_base = X_ptr + b * stride_xb + h * stride_xh

    d_offsets = tl.arange(0, D)
    mask_d = d_offsets < D
    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block = tl.load(q_ptr0, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    r_prev = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_prev = tl.zeros([BLOCK_M], tl.float32)

    for t in tl.static_range(0, BLOCK_N):
        valid_t = t < T
        k_t_ptr = k_base + t * stride_kt + d_offsets * stride_kd
        k_t = tl.load(k_t_ptr, mask=mask_d & valid_t, other=0.0).to(tl.float32)
        s_t = tl.sum(q_block * k_t[None, :], axis=1) * scale
        s_t = tl.where(mask_m & valid_t, s_t, -float("inf"))

        r_t = tl.maximum(r_prev, s_t)
        l_t = l_prev * tl.exp(r_prev - r_t) + tl.exp(s_t - r_t)

        p_t = tl.softmax(s_t, dim=0)
        x_t = p_t / (l_t + eps) * tl.exp(-r_t)

        r_prev = tl.where(valid_t, r_t, r_prev)
        l_prev = tl.where(valid_t, l_t, l_prev)

        x_ptr_t = x_base + m_offsets * stride_xm + t * stride_xt
        tl.store(x_ptr_t, x_t, mask=mask_m & valid_t)


@triton.jit
def flare_dense1_phase3_kernel(
    S_ptr,
    X_ptr,
    V_ptr,
    Y_ptr,
    stride_sb, stride_sh, stride_sm, stride_st,
    stride_xb, stride_xh, stride_xm, stride_xt,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_yb, stride_yh, stride_yt, stride_yd,
    B, H, M,
    T,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D
    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M
    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    s_base = S_ptr + b * stride_sb + h * stride_sh
    x_base = X_ptr + b * stride_xb + h * stride_xh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    s_ptr0 = s_base + m_offsets[:, None] * stride_sm + n_offsets[None, :] * stride_st
    x_ptr0 = x_base + m_offsets[:, None] * stride_xm + n_offsets[None, :] * stride_xt

    S = tl.load(s_ptr0, mask=mask_m[:, None] & mask_n[None, :], other=-float("inf"))
    X = tl.load(x_ptr0, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

    v_ptr0 = v_base + n_offsets[:, None] * stride_vt + d_offsets[None, :] * stride_vd
    v_block = tl.load(v_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    mu = tl.max(S, axis=0)
    mu = tl.where(mask_n, mu, 0.0)
    Btilde = tl.exp(tl.where(mask_m[:, None] & mask_n[None, :], S - mu[None, :], -float("inf")))

    W = tl.dot(tl.trans(X), Btilde, out_dtype=tl.float32, allow_tf32=False)

    t_idx = n_offsets[:, None]
    u_idx = n_offsets[None, :]
    W = tl.where(t_idx >= u_idx, W, 0.0)

    exp_mu = tl.exp(mu)
    v_scaled = v_block * exp_mu[:, None]
    Y_block = tl.dot(W, v_scaled, out_dtype=tl.float32, allow_tf32=False)
    y_ptr0 = y_base + n_offsets[:, None] * stride_yt + d_offsets[None, :] * stride_yd
    tl.store(y_ptr0, Y_block, mask=mask_n[:, None] & mask_d[None, :])

@triton.jit
def flare_dense_bwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    dY_ptr,
    dQ_ptr,
    dK_ptr,
    dV_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_dyb, stride_dyh, stride_dyt, stride_dyd,
    stride_dqh, stride_dqm, stride_dqd,
    stride_dkb, stride_dkh, stride_dkt, stride_dkd,
    stride_dvb, stride_dvh, stride_dvt, stride_dvd,
    B, H, M,
    T,
    scale,
    eps,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    n_offsets = tl.arange(0, BLOCK_N)
    mask_n = n_offsets < T

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    dy_base = dY_ptr + b * stride_dyb + h * stride_dyh

    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block = tl.load(q_ptr0, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    k_ptr0 = k_base + n_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
    k_block = tl.load(k_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    v_ptr0 = v_base + n_offsets[:, None] * stride_vt + d_offsets[None, :] * stride_vd
    v_block = tl.load(v_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    dy_ptr0 = dy_base + n_offsets[:, None] * stride_dyt + d_offsets[None, :] * stride_dyd
    dy_block = tl.load(dy_ptr0, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

    # S = Q @ K^T  -> [M, N]
    S = tl.dot(q_block, tl.trans(k_block), out_dtype=tl.float32, allow_tf32=False) * scale
    S = tl.where(mask_m[:, None] & mask_n[None, :], S, -float("inf"))

    # Global max shift for A_g
    s_max_col = tl.max(S, axis=0)
    s_max_global = tl.max(s_max_col, axis=0)
    A_g = tl.exp(S - s_max_global)
    A_g = tl.where(mask_m[:, None] & mask_n[None, :], A_g, 0.0)

    # Softmax over M for P
    s_max = s_max_col
    exp_s = tl.exp(S - s_max[None, :])
    exp_s = tl.where(mask_m[:, None] & mask_n[None, :], exp_s, 0.0)

    # l_s = tl.sum(exp_s, axis=0)
    # l_s = tl.where(l_s > 0, l_s, 1.0)
    # P = exp_s / l_s[None, :]
    P = tl.softmax(S, dim=0)

    # Lower triangular mask
    u_idx = n_offsets[:, None]
    t_idx = n_offsets[None, :]
    L = tl.where(u_idx <= t_idx, 1.0, 0.0).to(tl.float32)
    L_t = tl.where(t_idx <= u_idx, 1.0, 0.0).to(tl.float32)

    # D = A_g @ L
    D_mat = tl.dot(A_g, L, out_dtype=tl.float32, allow_tf32=False)
    eps_scaled = eps * tl.exp(-s_max_global)
    invD = 1.0 / (D_mat + eps_scaled)
    E = P * invD

    # W = E^T @ A_g
    W = tl.dot(tl.trans(E), A_g, out_dtype=tl.float32, allow_tf32=False)
    t_idx_rows = n_offsets[:, None]
    u_idx_cols = n_offsets[None, :]
    causal = t_idx_rows >= u_idx_cols
    W = tl.where(causal, W, 0.0)

    # dV = W^T @ dY
    dV_block = tl.dot(tl.trans(W), dy_block, out_dtype=tl.float32, allow_tf32=False)

    # dW = dY @ V^T
    dW = tl.dot(dy_block, tl.trans(v_block), out_dtype=tl.float32, allow_tf32=False)
    dW = tl.where(causal, dW, 0.0)

    # dE = A_g @ dW^T
    dE = tl.dot(A_g, tl.trans(dW), out_dtype=tl.float32, allow_tf32=False)
    # dA_g from W
    dA_g = tl.dot(E, dW, out_dtype=tl.float32, allow_tf32=False)

    # dD from E
    dP = dE * invD
    dInvD = dE * P
    dD = -dInvD * invD * invD

    # dA_g from D: dD @ L^T
    dA_g = dA_g + tl.dot(dD, L_t, out_dtype=tl.float32, allow_tf32=False)

    # dS from A_g (treat shift as constant)
    dS = dA_g * A_g

    # dS from softmax P
    sum_dP_P = tl.sum(dP * P, axis=0)
    dS = dS + P * (dP - sum_dP_P[None, :])

    # dQ = scale * dS @ K
    dQ_block = tl.dot(dS, k_block, out_dtype=tl.float32, allow_tf32=False) * scale
    # dK = scale * dS^T @ Q
    dK_block = tl.dot(tl.trans(dS), q_block, out_dtype=tl.float32, allow_tf32=False) * scale

    # Store dK, dV
    dK_base = dK_ptr + b * stride_dkb + h * stride_dkh
    dV_base = dV_ptr + b * stride_dvb + h * stride_dvh
    dK_ptr0 = dK_base + n_offsets[:, None] * stride_dkt + d_offsets[None, :] * stride_dkd
    dV_ptr0 = dV_base + n_offsets[:, None] * stride_dvt + d_offsets[None, :] * stride_dvd
    tl.store(dK_ptr0, dK_block, mask=mask_n[:, None] & mask_d[None, :])
    tl.store(dV_ptr0, dV_block, mask=mask_n[:, None] & mask_d[None, :])

    # Atomic add dQ (shared across batch)
    dQ_base = dQ_ptr + h * stride_dqh
    dQ_ptr0 = dQ_base + m_offsets[:, None] * stride_dqm + d_offsets[None, :] * stride_dqd
    tl.atomic_add(dQ_ptr0, dQ_block, mask=mask_m[:, None] & mask_d[None, :])

#======================================================================#
# Recurrent Implementations
#======================================================================#
class RecurrentFLAREOrig(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale=1.0, block_m=None, block_d=None):
        if Q.dim() != 3 or K.dim() != 4 or V.dim() != 4:
            raise ValueError(
                "Recurrent FLARE expects Q [H, M, D] and K/V [B, N, H, D]. "
                f"Got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
            )
        if K.size() != V.size():
            raise ValueError(f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}")
        if Q.size(0) != K.size(2) or Q.size(2) != K.size(3):
            raise ValueError(
                "Expected Q [H, M, D] and K/V [B, N, H, D] with matching H,D. "
                f"Got Q.shape={Q.shape} and K.shape={K.shape}"
            )

        H, M, D = Q.size()
        B, T, _, _ = K.size()
        if (M % 16) != 0 or (D % 16) != 0:
            raise ValueError(f"RecurrentFLARE requires M and D be multiples of 16. Got M={M}, D={D}")
        K_bhtd = K.permute(0, 2, 1, 3).contiguous()
        V_bhtd = V.permute(0, 2, 1, 3).contiguous()

        block_m = M
        block_d = D

        Y = torch.empty((B, H, T, D), device=Q.device, dtype=Q.dtype)
        grid = (B * H, triton.cdiv(D, block_d))

        num_warps = 4 if block_m <= 64 else 8
        flare_recurrent_orig_fwd_kernel[grid](
            Q, K_bhtd, V_bhtd, Y,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhtd.stride(0), K_bhtd.stride(1), K_bhtd.stride(2), K_bhtd.stride(3),
            V_bhtd.stride(0), V_bhtd.stride(1), V_bhtd.stride(2), V_bhtd.stride(3),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            B, H, M,
            T,
            scale,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            num_warps=num_warps,
            num_stages=2,
        )
        return Y

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("RecurrentFLARE backward not implemented.")

@triton.jit
def flare_recurrent_orig_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Y_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_yb, stride_yh, stride_yt, stride_yd,
    B, H, M,
    T,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    m_state = tl.full([BLOCK_M], -float("inf"), tl.float32)
    d_state = tl.zeros([BLOCK_M], tl.float32)
    u_state = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    # Load full Q once; reuse across all timesteps.
    d_block_offsets0 = tl.arange(0, BLOCK_D)
    mask_db0 = d_block_offsets0 < D
    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_block_offsets0[None, :] * stride_qd
    q_block0 = tl.load(q_ptr0, mask=mask_m[:, None] & mask_db0[None, :], other=0.0).to(tl.float32)

    tl.static_assert(BLOCK_D > 0)
    tl.static_assert(BLOCK_M > 0)

    t = 0
    while t < T:
        s_t = tl.zeros([BLOCK_M], tl.float32)
        k_ptr = k_base + t * stride_kt
        k_block0 = tl.load(k_ptr + d_block_offsets0 * stride_kd, mask=mask_db0, other=0.0).to(tl.float32)
        s_t += tl.sum(q_block0 * k_block0[None, :], axis=1)

        s_t *= scale
        s_t = tl.where(mask_m, s_t, -float("inf"))

        m_new = tl.maximum(m_state, s_t)
        gamma = tl.exp(m_state - m_new)
        eta = tl.exp(s_t - m_new)
        d_state = d_state * gamma + eta

        v_ptr = v_base + t * stride_vt
        v_block = tl.load(v_ptr + d_offsets * stride_vd, mask=mask_d, other=0.0).to(tl.float32)
        u_state = u_state * gamma[:, None] + eta[:, None] * v_block[None, :]

        inv_d = 1.0 / tl.where(d_state > 0, d_state, 1.0)
        z_block = u_state * inv_d[:, None]

        s_max = tl.max(s_t, axis=0)
        exp_s = tl.exp(s_t - s_max)
        exp_s = tl.where(mask_m, exp_s, 0.0)
        l_s = tl.sum(exp_s, axis=0)
        l_s = tl.where(l_s > 0, l_s, 1.0)

        y_block = tl.sum(exp_s[:, None] * z_block, axis=0) / l_s
        y_ptr = y_base + t * stride_yt
        tl.store(y_ptr + d_offsets * stride_yd, y_block.to(tl.float32), mask=mask_d)

        m_state = m_new
        t += 1

class RecurrentFLARE(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale=1.0, block_m=None, block_d=None):
        if Q.dim() != 3 or K.dim() != 4 or V.dim() != 4:
            raise ValueError(
                "Recurrent FLARE expects Q [H, M, D] and K/V [B, N, H, D]. "
                f"Got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
            )
        if K.size() != V.size():
            raise ValueError(f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}")
        if Q.size(0) != K.size(2) or Q.size(2) != K.size(3):
            raise ValueError(
                "Expected Q [H, M, D] and K/V [B, N, H, D] with matching H,D. "
                f"Got Q.shape={Q.shape} and K.shape={K.shape}"
            )

        H, M, D = Q.size()
        B, T, _, _ = K.size()
        if (M % 16) != 0 or (D % 16) != 0:
            raise ValueError(f"RecurrentFLARE requires M and D be multiples of 16. Got M={M}, D={D}")
        K_bhtd = K.permute(0, 2, 1, 3).contiguous()
        V_bhtd = V.permute(0, 2, 1, 3).contiguous()

        block_m = M
        block_d = D
        # Triton tl.dot requires M,N,K >= 16. Use a minimum BLOCK_T of 16 and mask.
        block_t = 16

        Y = torch.empty((B, H, T, D), device=Q.device, dtype=Q.dtype)
        grid = (B * H,)

        num_warps = 4 if block_m <= 64 else 8
        flare_recurrent_fwd_kernel[grid](
            Q, K_bhtd, V_bhtd, Y,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhtd.stride(0), K_bhtd.stride(1), K_bhtd.stride(2), K_bhtd.stride(3),
            V_bhtd.stride(0), V_bhtd.stride(1), V_bhtd.stride(2), V_bhtd.stride(3),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            B, H, M,
            T,
            scale,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            BLOCK_T=block_t,
            num_warps=num_warps,
            num_stages=2,
        )
        return Y

    @staticmethod
    def backward(ctx, *grad_outputs):
        raise RuntimeError("RecurrentFLARE backward not implemented.")

@triton.jit
def flare_recurrent_fwd_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    Y_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_yb, stride_yh, stride_yt, stride_yd,
    B, H, M,
    T,
    scale,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_bh = tl.program_id(0)

    b = pid_bh // H
    h = pid_bh - b * H

    d_offsets = tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    m_offsets = tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    m_state = tl.full([BLOCK_M], -float("inf"), tl.float32)
    d_state = tl.zeros([BLOCK_M], tl.float32)
    u_state = tl.zeros([BLOCK_M, BLOCK_D], tl.float32)

    # Load full Q once; reuse across all timesteps.
    q_ptr0 = q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q_block0 = tl.load(q_ptr0, mask=mask_m[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
    tl.multiple_of(d_offsets, 8)
    tl.max_contiguous(d_offsets, 16)

    tl.static_assert(BLOCK_D > 0)
    tl.static_assert(BLOCK_M > 0)

    t0 = 0
    while t0 < T:
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        mask_t = t_offsets < T

        k_ptr = k_base + t_offsets[:, None] * stride_kt + d_offsets[None, :] * stride_kd
        k_sub = tl.load(k_ptr, mask=mask_t[:, None] & mask_d[None, :], other=0.0).to(tl.float32)
        s_sub = tl.dot(q_block0, tl.trans(k_sub), out_dtype=tl.float32) * scale
        s_sub = tl.where(mask_m[:, None] & mask_t[None, :], s_sub, -float("inf"))

        t_idx = tl.arange(0, BLOCK_T)
        for j in tl.static_range(0, BLOCK_T):
            valid_t = (t0 + j) < T
            col_mask = t_idx == j
            s_t = tl.sum(tl.where(col_mask[None, :], s_sub, 0.0), axis=1)
            s_t = tl.where(mask_m & valid_t, s_t, -float("inf"))

            m_new = tl.maximum(m_state, s_t)
            gamma = tl.exp(m_state - m_new)
            eta = tl.exp(s_t - m_new)
            d_state = d_state * gamma + eta

            v_ptr = v_base + (t0 + j) * stride_vt + d_offsets * stride_vd
            v_block = tl.load(v_ptr, mask=mask_d & valid_t, other=0.0).to(tl.float32)
            u_state = u_state * gamma[:, None] + eta[:, None] * v_block[None, :]

            inv_d = 1.0 / tl.where(d_state > 0, d_state, 1.0)
            z_block = u_state * inv_d[:, None]

            s_max = tl.max(s_t, axis=0)
            exp_s = tl.exp(s_t - s_max)
            exp_s = tl.where(mask_m, exp_s, 0.0)
            l_s = tl.sum(exp_s, axis=0)
            l_s = tl.where(l_s > 0, l_s, 1.0)

            y_block = tl.sum(exp_s[:, None] * z_block, axis=0) / l_s
            y_ptr = y_base + (t0 + j) * stride_yt
            tl.store(y_ptr + d_offsets * stride_yd, y_block, mask=mask_d & valid_t)

            m_state = m_new

        t0 += BLOCK_T

def flare_recurrent_pytorch(Q, K, V, scale=1.0):
    if Q.dim() != 3 or K.dim() != 4 or V.dim() != 4:
        raise ValueError(
            "Recurrent FLARE expects Q [H, M, D] and K/V [B, N, H, D]. "
            f"Got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
        )
    if K.size() != V.size():
        raise ValueError(f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}")
    if Q.size(0) != K.size(2) or Q.size(2) != K.size(3):
        raise ValueError(
            "Expected Q [H, M, D] and K/V [B, N, H, D] with matching H,D. "
            f"Got Q.shape={Q.shape} and K.shape={K.shape}"
        )

    B, T, _, _ = K.size()
    H, M, D = Q.size()
    device = Q.device
    out_dtype = Q.dtype

    Q_f = Q.float().unsqueeze(0).expand(B, -1, -1, -1)  # [B,H,M,D]
    K_f = K.float().permute(0, 2, 1, 3).contiguous()    # [B,H,T,D]
    V_f = V.float().permute(0, 2, 1, 3).contiguous()    # [B,H,T,D]

    U = torch.zeros((B, H, M, D), device=device, dtype=torch.float32)
    d = torch.zeros((B, H, M), device=device, dtype=torch.float32)
    m = torch.full((B, H, M), -float("inf"), device=device, dtype=torch.float32)
    Y = torch.empty((B, H, T, D), device=device, dtype=out_dtype)

    for t in range(T):
        k_t = K_f[:, :, t, :]
        v_t = V_f[:, :, t, :]
        s_t = torch.einsum("bhmd,bhd->bhm", Q_f, k_t) * scale
        m_new = torch.maximum(m, s_t)
        gamma = torch.exp(m - m_new)
        eta = torch.exp(s_t - m_new)
        d = d * gamma + eta
        U = U * gamma[..., None] + eta[..., None] * v_t[:, :, None, :]
        Z = U / d[..., None]
        alpha = torch.softmax(s_t, dim=-1)
        y_t = torch.einsum("bhm,bhmd->bhd", alpha, Z)
        Y[:, :, t, :] = y_t.to(out_dtype)
        m = m_new

    return Y

#======================================================================#
#======================================================================#
#======================================================================#
# Prefix, Streaming, and Cached Implementations
#======================================================================#
#======================================================================#
#======================================================================#

def _init_flare_recurrent_state(
    batch_size: int,
    num_heads: int,
    num_latents: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:

    SHAPE_D = (batch_size, num_heads, num_latents)
    SHAPE_U = (batch_size, num_heads, num_latents, head_dim)

    return {
        "m": torch.full( SHAPE_D, -torch.inf, device=device, dtype=dtype),
        "d": torch.zeros(SHAPE_D, device=device, dtype=dtype),
        "u": torch.zeros(SHAPE_U, device=device, dtype=dtype),
    }

def _canonicalize_flare_state(
    state: dict[str, torch.Tensor] | None,
    batch_size: int,
    num_heads: int,
    num_latents: int,
    head_dim: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if state is None:
        return _init_flare_recurrent_state(
            batch_size=batch_size,
            num_heads=num_heads,
            num_latents=num_latents,
            head_dim=head_dim,
            device=device,
            dtype=torch.float32,
        )

    if not isinstance(state, dict):
        raise ValueError(f"FLARE recurrent state must be a dict with keys {'m', 'd', 'u'}. Got {type(state)}")

    if not all(k in state for k in ("m", "d", "u")):
        raise ValueError(f"Invalid FLARE recurrent state keys: {list(state.keys())}")

    m = state["m"].to(device=device, dtype=torch.float32)
    d = state["d"].to(device=device, dtype=torch.float32)
    u = state["u"].to(device=device, dtype=torch.float32)

    return {"m": m, "d": d, "u": u}


def _canonicalize_kv_for_prefill(
    K: torch.Tensor,
    V: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if K.dim() != 4 or V.dim() != 4:
        raise ValueError(f"Prefill expects K/V to be [B, T, H, D]. Got K={tuple(K.shape)}, V={tuple(V.shape)}")
    if K.shape != V.shape:
        raise ValueError(f"Prefill expects K and V to have same shape. Got K={tuple(K.shape)}, V={tuple(V.shape)}")
    return K, V


def _canonicalize_kv_for_decode(
    K: torch.Tensor,
    V: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if K.dim() == 4 and K.shape[1] == 1:
        K = K[:, 0]
    if V.dim() == 4 and V.shape[1] == 1:
        V = V[:, 0]
    if K.dim() != 3 or V.dim() != 3:
        raise ValueError(f"Decode expects K/V as [B, H, D] or [B, 1, H, D]. Got K={tuple(K.shape)}, V={tuple(V.shape)}")
    if K.shape != V.shape:
        raise ValueError(f"K and V must have same shape. Got K={tuple(K.shape)}, V={tuple(V.shape)}")
    return K, V


def _merge_flare_stats(
    m_a: torch.Tensor,
    d_a: torch.Tensor,
    u_a: torch.Tensor,
    m_b: torch.Tensor,
    d_b: torch.Tensor,
    u_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m_a = m_a.to(torch.float32)
    d_a = d_a.to(torch.float32)
    u_a = u_a.to(torch.float32)
    m_b = m_b.to(torch.float32)
    d_b = d_b.to(torch.float32)
    u_b = u_b.to(torch.float32)

    m_new = torch.maximum(m_a, m_b)
    is_a_inf = torch.isinf(m_a) & (m_a < 0)
    is_b_inf = torch.isinf(m_b) & (m_b < 0)
    is_new_inf = torch.isinf(m_new) & (m_new < 0)
    m_safe = torch.where(is_new_inf, torch.zeros_like(m_new), m_new)

    scale_a = torch.where(
        is_a_inf & is_new_inf,
        torch.ones_like(m_new),
        torch.where(is_a_inf, torch.zeros_like(m_new), torch.exp(m_a - m_safe)),
    )
    scale_b = torch.where(
        is_b_inf & is_new_inf,
        torch.ones_like(m_new),
        torch.where(is_b_inf, torch.zeros_like(m_new), torch.exp(m_b - m_safe)),
    )

    d_new = d_a * scale_a + d_b * scale_b
    u_new = u_a * scale_a[..., None] + u_b * scale_b[..., None]
    return m_new, d_new, u_new


#======================================================================#
# PyTorch Cached Implementations
#======================================================================#

def flare_prefill_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor] | None = None,
    scale: float = 1.0,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    K, V = _canonicalize_kv_for_prefill(K, V)
    if Q.dim() != 3:
        raise ValueError(f"Q must be [H, M, D]. Got Q={tuple(Q.shape)}")
    B, T, H, D = K.shape
    Hq, M, Dq = Q.shape
    if Hq != H or Dq != D:
        raise ValueError(f"Incompatible Q/K shapes. Q={tuple(Q.shape)}, K={tuple(K.shape)}")
    if attention_mask is not None and attention_mask.shape != (B, T):
        raise ValueError(f"attention_mask must be [B, T]. Got {tuple(attention_mask.shape)}")

    out_dtype = V.dtype
    Q_f = Q.float().unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, M, D]
    K_f = K.float().permute(0, 2, 1, 3).contiguous()    # [B, H, T, D]
    V_f = V.float().permute(0, 2, 1, 3).contiguous()    # [B, H, T, D]

    st = _canonicalize_flare_state(
        state=state,
        batch_size=B,
        num_heads=H,
        num_latents=M,
        head_dim=D,
        device=K.device,
    )
    m = st["m"]
    d = st["d"]
    u = st["u"]

    Y = torch.empty((B, H, T, D), device=K.device, dtype=torch.float32)
    for t in range(T):
        k_t = K_f[:, :, t, :]
        v_t = V_f[:, :, t, :]
        s_t = torch.einsum("bhmd,bhd->bhm", Q_f, k_t) * float(scale)
        if attention_mask is not None:
            valid = attention_mask[:, t].to(torch.bool).view(B, 1, 1)
            s_t = torch.where(valid, s_t, torch.full_like(s_t, -float("inf")))
        else:
            valid = torch.ones((B, 1, 1), device=K.device, dtype=torch.bool)

        m_new = torch.maximum(m, s_t)
        is_m_inf = torch.isinf(m) & (m < 0)
        is_m_new_inf = torch.isinf(m_new) & (m_new < 0)
        m_new_safe = torch.where(is_m_new_inf, torch.zeros_like(m_new), m_new)
        gamma = torch.where(
            is_m_inf & is_m_new_inf,
            torch.ones_like(m_new),
            torch.where(is_m_inf, torch.zeros_like(m_new), torch.exp(m - m_new_safe)),
        )
        eta = torch.where(is_m_new_inf, torch.zeros_like(s_t), torch.exp(s_t - m_new_safe))

        d = d * gamma + eta
        u = u * gamma[..., None] + eta[..., None] * v_t[:, :, None, :]
        m = m_new

        d_safe = torch.where(d > 0, d, torch.ones_like(d))
        z_t = u / d_safe[..., None]
        s_decode = torch.where(valid, s_t, torch.zeros_like(s_t))
        alpha = torch.softmax(s_decode, dim=-1) * valid.float()
        y_t = torch.einsum("bhm,bhmd->bhd", alpha, z_t)
        Y[:, :, t, :] = y_t

    next_state = {"m": m, "d": d, "u": u}
    return Y.permute(0, 2, 1, 3).to(out_dtype), next_state


def flare_decode_pytorch(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor],
    scale: float = 1.0,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    K, V = _canonicalize_kv_for_decode(K, V)
    B, _, _ = K.shape
    if attention_mask is not None:
        if attention_mask.dim() == 2 and attention_mask.shape[1] == 1:
            attention_mask = attention_mask[:, 0]
        if attention_mask.dim() != 1 or attention_mask.shape[0] != B:
            raise ValueError(f"Decode attention_mask must be [B] or [B, 1]. Got {tuple(attention_mask.shape)}")
        attention_mask = attention_mask[:, None]
    y, next_state = flare_prefill_pytorch(
        Q=Q,
        K=K[:, None, :, :],
        V=V[:, None, :, :],
        state=state,
        scale=scale,
        attention_mask=attention_mask,
    )
    return y, next_state


#======================================================================#
# Triton Cached Implementations
#======================================================================#

@triton.jit
def flare_recurrent_step_kernel(
    Q_ptr, K_ptr, V_ptr,
    M_ptr, D_ptr, U_ptr,
    Y_ptr, Mask_ptr,
    stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kd,
    stride_vb, stride_vh, stride_vd,
    stride_mb, stride_mh, stride_mm,
    stride_db, stride_dh, stride_dm,
    stride_ub, stride_uh, stride_um, stride_ud,
    stride_yb, stride_yh, stride_yd,
    stride_mask_b,
    B, H, M, D, scale,
    HAS_MASK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    bh = B * H
    if pid_bh >= bh:
        return
    b = pid_bh // H
    h = pid_bh - b * H

    m_offsets = tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_md = mask_m[:, None] & mask_d[None, :]

    valid_t = 1
    if HAS_MASK:
        mask_val = tl.load(Mask_ptr + b * stride_mask_b)
        valid_t = mask_val != 0

    q_base = Q_ptr + h * stride_qh
    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    m_base = M_ptr + b * stride_mb + h * stride_mh
    d_base = D_ptr + b * stride_db + h * stride_dh
    u_base = U_ptr + b * stride_ub + h * stride_uh
    y_base = Y_ptr + b * stride_yb + h * stride_yh

    q = tl.load(
        q_base + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)
    k = tl.load(k_base + d_offsets * stride_kd, mask=mask_d, other=0.0).to(tl.float32)
    v = tl.load(v_base + d_offsets * stride_vd, mask=mask_d, other=0.0).to(tl.float32)

    m_state = tl.load(m_base + m_offsets * stride_mm, mask=mask_m, other=-float("inf")).to(tl.float32)
    d_state = tl.load(d_base + m_offsets * stride_dm, mask=mask_m, other=0.0).to(tl.float32)
    u_state = tl.load(
        u_base + m_offsets[:, None] * stride_um + d_offsets[None, :] * stride_ud,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)

    s_raw = tl.sum(q * k[None, :], axis=1) * scale
    s_t = tl.where(valid_t & mask_m, s_raw, -float("inf"))

    m_new = tl.maximum(m_state, s_t)
    is_m_state_inf = m_state == -float("inf")
    is_m_new_inf = m_new == -float("inf")
    m_new_safe = tl.where(is_m_new_inf, 0.0, m_new)
    gamma = tl.where(
        is_m_state_inf & is_m_new_inf,
        1.0,
        tl.where(is_m_state_inf, 0.0, tl.exp(m_state - m_new_safe)),
    )
    eta = tl.where(is_m_new_inf, 0.0, tl.exp(s_t - m_new_safe))

    d_state = d_state * gamma + eta
    u_state = u_state * gamma[:, None] + eta[:, None] * v[None, :]
    m_state = m_new

    d_safe = tl.where(d_state > 0, d_state, 1.0)
    z = u_state / d_safe[:, None]

    s_decode = tl.where(valid_t & mask_m, s_raw, 0.0)
    s_max = tl.max(s_decode, axis=0)
    s_exp = tl.exp(s_decode - s_max)
    s_exp = tl.where(valid_t & mask_m, s_exp, 0.0)
    s_sum = tl.sum(s_exp, axis=0)
    s_sum = tl.where(s_sum > 0, s_sum, 1.0)
    alpha = s_exp / s_sum

    y = tl.sum(alpha[:, None] * z, axis=0)
    y = tl.where(valid_t, y, 0.0)

    tl.store(m_base + m_offsets * stride_mm, m_state, mask=mask_m)
    tl.store(d_base + m_offsets * stride_dm, d_state, mask=mask_m)
    tl.store(
        u_base + m_offsets[:, None] * stride_um + d_offsets[None, :] * stride_ud,
        u_state,
        mask=mask_md,
    )
    tl.store(y_base + d_offsets * stride_yd, y, mask=mask_d)


def flare_decode_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor] | None = None,
    scale: float = 1.0,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not Q.is_cuda:
        raise RuntimeError(
            "flare_decode requires CUDA tensors. "
        )

    K, V = _canonicalize_kv_for_decode(K, V)
    B, H, D = K.shape
    Hq, M, Dq = Q.shape
    if Hq != H or Dq != D:
        raise ValueError(f"Incompatible Q/K shapes. Q={tuple(Q.shape)}, K={tuple(K.shape)}")
    if attention_mask is not None:
        if attention_mask.dim() == 2 and attention_mask.shape[1] == 1:
            attention_mask = attention_mask[:, 0]
        if attention_mask.dim() != 1 or attention_mask.shape[0] != B:
            raise ValueError(f"Decode attention_mask must be [B] or [B, 1]. Got {tuple(attention_mask.shape)}")
        attention_mask = attention_mask.to(device=K.device, dtype=torch.int32).contiguous()

    st = _canonicalize_flare_state(
        state=state,
        batch_size=B,
        num_heads=H,
        num_latents=M,
        head_dim=D,
        device=K.device,
    )
    m_state = st["m"].contiguous()
    d_state = st["d"].contiguous()
    u_state = st["u"].contiguous()

    q = Q.contiguous().float()
    k = K.contiguous().float()
    v = V.contiguous().float()
    y = torch.empty((B, H, D), device=K.device, dtype=torch.float32)

    block_m = max(16, triton.next_power_of_2(M))
    block_d = max(16, triton.next_power_of_2(D))
    grid = (B * H,)
    if attention_mask is None:
        flare_recurrent_step_kernel[grid](
            q, k, v,
            m_state, d_state, u_state,
            y, y,  # dummy mask ptr for HAS_MASK=False
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            m_state.stride(0), m_state.stride(1), m_state.stride(2),
            d_state.stride(0), d_state.stride(1), d_state.stride(2),
            u_state.stride(0), u_state.stride(1), u_state.stride(2), u_state.stride(3),
            y.stride(0), y.stride(1), y.stride(2),
            0,
            B, H, M, D, float(scale),
            HAS_MASK=False,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
        )
    else:
        flare_recurrent_step_kernel[grid](
            q, k, v,
            m_state, d_state, u_state,
            y, attention_mask,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            m_state.stride(0), m_state.stride(1), m_state.stride(2),
            d_state.stride(0), d_state.stride(1), d_state.stride(2),
            u_state.stride(0), u_state.stride(1), u_state.stride(2), u_state.stride(3),
            y.stride(0), y.stride(1), y.stride(2),
            attention_mask.stride(0),
            B, H, M, D, float(scale),
            HAS_MASK=True,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
        )

    next_state = {"m": m_state, "d": d_state, "u": u_state}
    return y[:, None, :, :].to(V.dtype), next_state


def flare_prefill_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor] | None = None,
    scale: float = 1.0,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if not Q.is_cuda:
        raise RuntimeError("flare_prefill requires CUDA tensors.")

    K, V = _canonicalize_kv_for_prefill(K, V)
    B, T, H, D = K.shape
    if Q.dim() != 3:
        raise ValueError(f"Q must be [H, M, D]. Got Q={tuple(Q.shape)}")
    Hq, M, Dq = Q.shape
    if Hq != H or Dq != D:
        raise ValueError(f"Incompatible Q/K shapes. Q={tuple(Q.shape)}, K={tuple(K.shape)}")
    if attention_mask is not None and attention_mask.shape != (B, T):
        raise ValueError(f"attention_mask must be [B, T]. Got {tuple(attention_mask.shape)}")
    if T == 0:
        st = _canonicalize_flare_state(
            state=state,
            batch_size=B,
            num_heads=H,
            num_latents=M,
            head_dim=D,
            device=K.device,
        )
        return torch.empty((B, 0, H, D), device=K.device, dtype=V.dtype), st

    if attention_mask is not None:
        # Keep masked prefill semantics identical to recurrent decode while avoiding any PyTorch fallback.
        all_valid = bool((attention_mask != 0).all().item())
        if not all_valid:
            st = _canonicalize_flare_state(
                state=state,
                batch_size=B,
                num_heads=H,
                num_latents=M,
                head_dim=D,
                device=K.device,
            )
            y_steps = []
            for t in range(T):
                y_t, st = flare_decode_triton(
                    Q=Q,
                    K=K[:, t, :, :],
                    V=V[:, t, :, :],
                    state=st,
                    scale=scale,
                    attention_mask=attention_mask[:, t],
                )
                y_steps.append(y_t)
            return torch.cat(y_steps, dim=1), st
        attention_mask = None

    st = _canonicalize_flare_state(
        state=state,
        batch_size=B,
        num_heads=H,
        num_latents=M,
        head_dim=D,
        device=K.device,
    )
    m0 = st["m"].reshape(B * H, M)
    d0 = st["d"].reshape(B * H, M)
    u0 = st["u"].reshape(B * H, M, D)

    dtype = K.dtype
    eps = _get_eps_for_dtype(dtype)
    clamp_max = _get_exp_clamp_for_dtype(dtype)
    env_chunk = os.environ.get("FLARE_CHUNK_SIZE", "")
    CHUNK_SIZE = int(env_chunk) if env_chunk else 128
    CHUNK_SIZE = max(1, CHUNK_SIZE)

    block_m_env = os.environ.get("FLARE_BLOCK_M", "")
    BLOCK_M = int(block_m_env) if block_m_env else triton.next_power_of_2(max(1, min(M, 128)))
    BLOCK_M = max(1, BLOCK_M)

    block_t_env = os.environ.get("FLARE_PREFILL_BLOCK_T", "")
    if block_t_env:
        BLOCK_T = int(block_t_env)
    else:
        BLOCK_T = 16 if CHUNK_SIZE >= 16 else CHUNK_SIZE
    BLOCK_T = max(1, min(BLOCK_T, CHUNK_SIZE))

    BH = B * H
    NUM_CHUNKS = math.ceil(T / CHUNK_SIZE)
    NUM_M_BLOCKS = math.ceil(M / BLOCK_M)

    use_fp16 = dtype == torch.float16
    use_bf16 = dtype == torch.bfloat16
    stats_fp32 = True
    num_warps = 4 if D <= 64 else 8
    num_stages = 2 if D <= 64 else 3

    q = Q.contiguous()
    k = K.contiguous()
    v = V.contiguous()
    q_stride = q.stride()
    k_stride = k.stride()
    v_stride = v.stride()

    chunk_max = torch.full((BH, NUM_CHUNKS, M), -float("inf"), device=K.device, dtype=torch.float32)
    chunk_den = torch.zeros((BH, NUM_CHUNKS, M), device=K.device, dtype=torch.float32)
    chunk_num = torch.zeros((BH, NUM_CHUNKS, M, D), device=K.device, dtype=torch.float32)
    flare_chunk_prepare[(BH, NUM_CHUNKS, NUM_M_BLOCKS)](
        k, q, v,
        chunk_max, chunk_den, chunk_num,
        k_stride[0], k_stride[1], k_stride[2], k_stride[3],
        q_stride[0], q_stride[1], q_stride[2],
        v_stride[0], v_stride[1], v_stride[2], v_stride[3],
        *chunk_max.stride(),
        *chunk_den.stride(),
        *chunk_num.stride(),
        BH, M, T, D, float(scale),
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_M=BLOCK_M,
        USE_FP16=use_fp16,
        USE_BF16=use_bf16,
        USE_FP32_STATS=stats_fp32,
        ALLOW_TF32=True,
        H=H,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    prefix_max = torch.empty((BH, NUM_CHUNKS, M), device=K.device, dtype=torch.float32)
    prefix_den = torch.zeros((BH, NUM_CHUNKS, M), device=K.device, dtype=torch.float32)
    prefix_num = torch.zeros((BH, NUM_CHUNKS, M, D), device=K.device, dtype=torch.float32)
    flare_chunk_prefix[(BH,)](
        chunk_max, chunk_den, chunk_num,
        prefix_max, prefix_den, prefix_num,
        *chunk_max.stride(),
        *chunk_den.stride(),
        *chunk_num.stride(),
        *prefix_max.stride(),
        *prefix_den.stride(),
        *prefix_num.stride(),
        BH, M, D, NUM_CHUNKS,
        USE_FP16=use_fp16,
        USE_BF16=use_bf16,
        USE_FP32_STATS=stats_fp32,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    m0_expand = m0[:, None, :].expand(-1, NUM_CHUNKS, -1)
    d0_expand = d0[:, None, :].expand(-1, NUM_CHUNKS, -1)
    u0_expand = u0[:, None, :, :].expand(-1, NUM_CHUNKS, -1, -1)
    prefix_max, prefix_den, prefix_num = _merge_flare_stats(
        m0_expand, d0_expand, u0_expand,
        prefix_max, prefix_den, prefix_num,
    )
    prefix_max = prefix_max.contiguous()
    prefix_den = prefix_den.contiguous()
    prefix_num = prefix_num.contiguous()

    O = torch.empty((B, T, H, D), device=K.device, dtype=torch.float32)
    o_stride = O.stride()
    flare_chunk_fwd[(BH, NUM_CHUNKS)](
        k, q, v,
        prefix_max, prefix_den, prefix_num,
        O,
        k_stride[0], k_stride[1], k_stride[2], k_stride[3],
        q_stride[0], q_stride[1], q_stride[2],
        v_stride[0], v_stride[1], v_stride[2], v_stride[3],
        *prefix_max.stride(),
        *prefix_den.stride(),
        *prefix_num.stride(),
        o_stride[0], o_stride[1], o_stride[2], o_stride[3],
        BH, M, T, D, float(scale), eps, clamp_max,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_T=BLOCK_T,
        USE_FP16=use_fp16,
        USE_BF16=use_bf16,
        USE_FP32_STATS=stats_fp32,
        ALLOW_TF32=True,
        STABLE_SCAN=False,
        H=H,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    last_prefix_max = prefix_max[:, -1, :]
    last_prefix_den = prefix_den[:, -1, :]
    last_prefix_num = prefix_num[:, -1, :, :]
    last_chunk_max = chunk_max[:, -1, :]
    last_chunk_den = chunk_den[:, -1, :]
    last_chunk_num = chunk_num[:, -1, :, :]
    m_fin, d_fin, u_fin = _merge_flare_stats(
        last_prefix_max, last_prefix_den, last_prefix_num,
        last_chunk_max, last_chunk_den, last_chunk_num,
    )
    next_state = {
        "m": m_fin.reshape(B, H, M).contiguous(),
        "d": d_fin.reshape(B, H, M).contiguous(),
        "u": u_fin.reshape(B, H, M, D).contiguous(),
    }

    return O.to(V.dtype), next_state


def flare_prefill(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor] | None = None,
    scale: float = 1.0,
    attention_mask: torch.Tensor | None = None,
    impl: str = "triton",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if impl == "triton":
        return flare_prefill_triton(
            Q=Q,
            K=K,
            V=V,
            state=state,
            scale=scale,
            attention_mask=attention_mask,
        )
    if impl == "pytorch":
        return flare_prefill_pytorch(
            Q=Q,
            K=K,
            V=V,
            state=state,
            scale=scale,
            attention_mask=attention_mask,
        )
    raise ValueError(f"Unsupported impl={impl}. Expected 'triton' or 'pytorch'.")


def flare_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    state: dict[str, torch.Tensor],
    scale: float = 1.0,
    attention_mask: torch.Tensor | None = None,
    impl: str = "triton",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if impl == "triton":
        return flare_decode_triton(
            Q=Q,
            K=K,
            V=V,
            state=state,
            scale=scale,
            attention_mask=attention_mask,
        )
    if impl == "pytorch":
        return flare_decode_pytorch(
            Q=Q,
            K=K,
            V=V,
            state=state,
            scale=scale,
            attention_mask=attention_mask,
        )
    raise ValueError(f"Unsupported impl={impl}. Expected 'triton' or 'pytorch'.")


#======================================================================#
# PyTorch Implementations
#======================================================================#

def flare_noncausal(Q, K, V, scale=1.0):
    assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
        "Q, K, V must be 3D and 4D tensors respectively "
        f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
    )
    assert K.size() == V.size(), f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}"
    assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
        "Expected Q [H, M, D] and K/V [B, N, H, D]. "
        f"Got Q.shape={Q.shape} and K.shape={K.shape}"
    )
    H, M, D = Q.size()
    B, N, H, D = K.size()

    Q = Q.unsqueeze(0).expand(B, -1, -1, -1)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)

    Y = F.scaled_dot_product_attention(Q, K, V, is_causal=False, scale=scale)
    Z = F.scaled_dot_product_attention(K, Q, Y, is_causal=False, scale=scale)

    return Y

def causal_SDPA(Q, K, V):
    if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
        raise ValueError(
            "Causal SDPA expects Q, K, V all as 4D tensors for benchmarking. "
            f"Got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
        )
    if Q.size() != K.size() or K.size() != V.size():
        raise ValueError(
            f"Q, K, V must have the same shape for causal SDPA. Got Q.shape={Q.shape}, K.shape={K.shape}, V.shape={V.shape}"
        )
    Q = Q.permute(0, 2, 1, 3)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    Y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

    return Y

def flare_causal_reference(Q, K, V, scale=1.0):
    assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
        "Q, K, V must be 3D and 4D tensors respectively "
        f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
    )
    assert K.size() == V.size(), f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}"
    assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
        "Expected Q [H, M, D] and K/V [B, N, H, D]. "
        f"Got Q.shape={Q.shape} and K.shape={K.shape}"
    )
    H, M, D = Q.size()
    B, N, H, D = K.size()

    if os.environ.get("FLARE_REFERENCE_FP32", "1") == "1":
        Q = Q.float()
        K = K.float()
        V = V.float()
    Q_bhmd = Q.unsqueeze(0).expand(B, -1, -1, -1)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    Y = torch.zeros_like(K)
    for t in range(N):
        Kt = K[:, :, :t+1, :]
        Vt = V[:, :, :t+1, :]
        Zt = F.scaled_dot_product_attention(Q_bhmd, Kt, Vt, is_causal=False, scale=scale)
        Yt = F.scaled_dot_product_attention(Kt, Q_bhmd, Zt, is_causal=False, scale=scale)
        Y[:, :, t] = Yt[:, :, t]

    Y = Y.permute(0, 2, 1, 3)

    return Y

#------------------------------------------------------------------------------#
# PyTorch Implementation of FLARE
#------------------------------------------------------------------------------#

def flare_causal_chunked(Q, K, V, scale=1.0, eps=None, profile: bool = False, chunk_size=None):
    assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
        "Q, K, V must be 3D and 4D tensors respectively "
        f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
    )
    assert K.size() == V.size(), f"K and V must have the same shape. Got K.shape={K.shape} and V.shape={V.shape}"
    assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
        "Expected Q [H, M, D] and K/V [B, N, H, D]. "
        f"Got Q.shape={Q.shape} and K.shape={K.shape}"
    )
    H, M, D = Q.size()
    B, N, Hk, Dk = K.size()
    assert H == Hk and D == Dk, "Incompatible K/V dimensions"

    device = Q.device
    out_dtype = Q.dtype
    compute_dtype = torch.float32
    if os.environ.get("FLARE_PYTORCH_MATCH_REFERENCE", "") == "1":
        compute_dtype = Q.dtype
    if eps is None:
        eps = _get_eps_for_dtype(Q.dtype)

    Q_f = Q.to(compute_dtype)
    K_f = K.to(compute_dtype)
    V_f = V.to(compute_dtype)

    ###
    # CHUNKING & PADDING
    ###

    if chunk_size is None:
        env_chunk = os.environ.get("FLARE_PYTORCH_CHUNK_SIZE", "")
        chunk_size = int(env_chunk) if env_chunk else None
    C = int(chunk_size) if chunk_size is not None else max(64, min(2048, N // 2))
    # Non-stable branch only.
    CHUNK_SIZE = C
    NC = NUM_CHUNKS = math.ceil(N / CHUNK_SIZE)

    PADDED_LEN = NUM_CHUNKS * CHUNK_SIZE
    PAD = PADDED_LEN - N

    if PAD > 0:
        pad_val = torch.zeros((B, PAD, H, D), device=device, dtype=compute_dtype)
        K_f = torch.cat([K_f, pad_val], dim=1)
        V_f = torch.cat([V_f, pad_val], dim=1)

    ###
    # CHUNKING
    ###

    Kc = K_f.reshape(B, NC, C, H, D).permute(0, 3, 1, 2, 4).contiguous()
    Vc = V_f.reshape(B, NC, C, H, D).permute(0, 3, 1, 2, 4).contiguous()

    #---------------------------------------------------------------#
    # Phase 0: Compute scores
    # NEEDS: Q, Kc, Vc
    # RETURNS: score_chunk
    #---------------------------------------------------------------#

    score_chunk = scale * torch.einsum("bhncd,hmd->bhncm", Kc, Q_f)  # [B, H, NC, C, M]

    if PAD > 0:
        score_chunk.view(B, H, PADDED_LEN, M)[:, :, -PAD:, :] = -torch.inf

    phase1_start = phase1_end = None
    phase2_start = phase2_end = None
    phase3_start = phase3_end = None
    if profile and torch.cuda.is_available():
        phase1_start = torch.cuda.Event(enable_timing=True)
        phase1_end = torch.cuda.Event(enable_timing=True)
        phase2_start = torch.cuda.Event(enable_timing=True)
        phase2_end = torch.cuda.Event(enable_timing=True)
        phase3_start = torch.cuda.Event(enable_timing=True)
        phase3_end = torch.cuda.Event(enable_timing=True)
        phase1_start.record()

    #---------------------------------------------------------------#
    # Phase 1: Compute chunk statistics independently for each chunk
    # NEEDS: score_chunk, Vc
    # RETURNS: score_chunk_max, score_chunk_den, score_chunk_num
    #---------------------------------------------------------------#

    score_chunk_max = score_chunk.max(dim=3).values                               # [B, H, NC, M]
    score_chunk_exp = torch.exp(score_chunk - score_chunk_max.unsqueeze(3))       # [B, H, NC, C, M]
    score_chunk_den = score_chunk_exp.sum(dim=3)                                  # [B, H, NC, M]
    BHNC = B * H * NC
    exp_b = score_chunk_exp.reshape(BHNC, C, M)
    V_b = Vc.reshape(BHNC, C, D)
    score_chunk_num = torch.bmm(exp_b.transpose(1, 2), V_b).reshape(B, H, NC, M, D)

    if profile and torch.cuda.is_available():
        phase1_end.record()
        phase2_start.record()

    #---------------------------------------------------------------#
    # Phase 2: Compute prefix statistics from independent chunk statistics
    # NEEDS: score_chunk_max, score_chunk_den, score_chunk_num
    # RETURNS: score_prev_max, score_prev_den, score_prev_num
    #---------------------------------------------------------------#

    # Score suffix (prev) statistics needed for phase 3
    score_prev_max = torch.empty(B, H, NC, M, device=device, dtype=compute_dtype)
    score_prev_den = torch.zeros(B, H, NC, M, device=device, dtype=compute_dtype)
    score_prev_num = torch.zeros(B, H, NC, M, D, device=device, dtype=compute_dtype)

    # temporary variables for prefix statistics
    max_curr = torch.full((B, H, M), -float("inf"), device=device, dtype=compute_dtype)
    den_curr = torch.zeros((B, H, M), device=device, dtype=compute_dtype)
    num_curr = torch.zeros((B, H, M, D), device=device, dtype=compute_dtype)

    for chunk_idx in range(NC):
        score_prev_max[:, :, chunk_idx, :] = max_curr
        score_prev_den[:, :, chunk_idx, :] = den_curr
        score_prev_num[:, :, chunk_idx, :, :] = num_curr

        sc_max = score_chunk_max[:, :, chunk_idx, :]
        sc_den = score_chunk_den[:, :, chunk_idx, :]
        sc_num = score_chunk_num[:, :, chunk_idx, :]

        ###
        ### online softmax update
        ###

        # get new max (including current chunk)
        max_new = torch.maximum(max_curr, sc_max)

        # get rescale factors
        rescale_factor_prev = torch.exp(max_curr - max_new) # rescale factor for previous chunks
        rescale_factor_curr = torch.exp(sc_max - max_new)   # rescale factor for current chunk

        # update denominator, numerator, max
        den_curr = den_curr * rescale_factor_prev + sc_den * rescale_factor_curr
        num_curr = num_curr * rescale_factor_prev.unsqueeze(-1) + sc_num * rescale_factor_curr.unsqueeze(-1)
        max_curr = max_new

    if profile and torch.cuda.is_available():
        phase2_end.record()
        phase3_start.record()

    #---------------------------------------------------------------#
    # Phase 3: Compute output for all tokens in a chunk in parallel for all chunks
    # NEEDS: score_chunk, Vc
    # RETURNS: Yc
    #---------------------------------------------------------------#

    clamp_max = _get_exp_clamp_for_dtype(Q.dtype)
    if os.environ.get("FLARE_PYTORCH_DISABLE_CLAMP", "") == "1":
        clamp_max = float("inf")
    S = score_chunk  # [B, H, NC, C, M]
    _check_finite_allow_neg_inf("flare_causal_chunked.score_prev_max", score_prev_max)
    score_prev_max = score_prev_max.clamp(max=clamp_max)
    exp_prev_max = torch.exp(score_prev_max)  # [B, H, NC, M]
    _check_finite("flare_causal_chunked.exp_prev_max", exp_prev_max)
    _check_finite("flare_causal_chunked.score_prev_num", score_prev_num)
    prev_num = score_prev_num * exp_prev_max.unsqueeze(-1)
    _check_finite("flare_causal_chunked.prev_num", prev_num)

    if os.environ.get("FLARE_PYTORCH_EXACT_FORMULA", "") == "1":
        Yc = torch.zeros((B, H, NC, C, D), device=device, dtype=compute_dtype)
        for chunk_idx in range(NC):
            for t in range(C):
                token_idx = chunk_idx * C + t
                if token_idx >= N:
                    continue
                Kt = K_f[:, : token_idx + 1, :, :]  # [B, t+1, H, D]
                Vt = V_f[:, : token_idx + 1, :, :]
                S_all = scale * torch.einsum("bthd,hmd->bthm", Kt, Q_f)  # [B,t+1,H,M]
                m_u = S_all.max(dim=1).values
                exp_u = torch.exp(S_all - m_u.unsqueeze(1))
                den_total = exp_u.sum(dim=1) * torch.exp(m_u)
                sum_exp_v = torch.einsum("bthm,bthd->bhmd", exp_u, Vt) * torch.exp(m_u).unsqueeze(-1)
                S_t = S_all[:, -1, :, :]
                P_t = torch.softmax(S_t, dim=-1)
                expA_t = P_t / (den_total + eps)
                y_t = torch.einsum("bhm,bhmd->bhd", expA_t, sum_exp_v)
                Yc[:, :, chunk_idx, t, :] = y_t
    else:
        S_exp = S.clamp(max=clamp_max)
        expS = torch.exp(S_exp)  # [B, H, NC, C, M]
        _check_finite("flare_causal_chunked.expS", expS)

        # Causal cumulative sum within the chunk for each latent:
        den_chunk = torch.cumsum(expS, dim=3)  # [B, H, NC, C, M]
        den_total = den_chunk + (score_prev_den * exp_prev_max).unsqueeze(3)  # [B, H, NC, C, M]
        _check_finite("flare_causal_chunked.den_total", den_total)

        # Latent softmax per token (summation over M)
        P = torch.softmax(S, dim=-1)  # [B, H, NC, C, M]

        # expA[t,m] = softmax(S[t])[m] / den_total[t,m]
        expA = P / (den_total + eps)  # [B, H, NC, C, M]
        _check_finite("flare_causal_chunked.expA", expA)

        # Previous-prefix contribution:
        # y_prev[t] = sum_m expA[t,m] * num_prev_actual[m]
        BHNC = B * H * NC
        expA_b = expA.reshape(BHNC, C, M)
        expS_b = expS.reshape(BHNC, C, M)
        prev_num_b = prev_num.reshape(BHNC, M, D)
        Vc_b = Vc.reshape(BHNC, C, D)

        y_prev_b = torch.bmm(expA_b, prev_num_b)
        _check_finite("flare_causal_chunked.y_prev_b", y_prev_b)

        W_b = torch.bmm(expA_b, expS_b.transpose(1, 2))
        causal = _get_causal_mask(C, device)
        W_b = W_b.masked_fill(~causal[None, :, :], 0.0)
        y_curr_b = torch.bmm(W_b, Vc_b)
        _check_finite("flare_causal_chunked.y_curr_b", y_curr_b)

        y_prev = y_prev_b.reshape(B, H, NC, C, D)
        y_curr = y_curr_b.reshape(B, H, NC, C, D)
        Yc = y_prev + y_curr
    # Non-stable branch only.

    #---------------------------------------------------------------#
    # Return output
    #---------------------------------------------------------------#

    Y = Yc.reshape(B, H, PADDED_LEN, D)[:, :, :N, :].permute(0, 2, 1, 3).to(out_dtype)
    _check_finite("flare_causal_chunked.Y", Y)
    if profile and torch.cuda.is_available():
        phase3_end.record()
        torch.cuda.synchronize()
        mode = _BWD_PROFILE_MODE or "triton3"
        _BWD_PROFILE_TIMINGS.setdefault(mode, {})
        _BWD_PROFILE_TIMINGS[mode]["phase1_chunk_stats"] = phase1_start.elapsed_time(phase1_end)
        _BWD_PROFILE_TIMINGS[mode]["phase2_prefix"] = phase2_start.elapsed_time(phase2_end)
        _BWD_PROFILE_TIMINGS[mode]["phase3_output"] = phase3_start.elapsed_time(phase3_end)
    return Y


#======================================================================#
# Testing scripts
#======================================================================#
def optimize_for_h100():
    """Apply H100-specific optimizations for maximum performance"""

    # Environment variables for H100 optimization
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8 API
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # H100-specific PyTorch backend optimizations
    # Enable TF32 tensor cores: FP32 operations use TF32 (10-bit mantissa, 8-bit exponent)
    # This provides ~6× better accuracy than BF16 with only ~1% performance cost
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # FP8 support (only available in newer PyTorch versions)
    try:
        torch.backends.cuda.matmul.allow_fp8_e4m3fn = True
        torch.backends.cuda.matmul.allow_fp8_e5m2 = True
    except AttributeError:
        pass

    # Performance vs reproducibility settings (choose one)
    torch.backends.cudnn.benchmark = True  # Enable for better performance
    torch.backends.cudnn.deterministic = False  # Disable for better performance

    # Memory management optimizations
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        torch.cuda.empty_cache()

    print("H100 optimizations applied successfully!")

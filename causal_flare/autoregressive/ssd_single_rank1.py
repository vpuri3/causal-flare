"""Single-state SSD rank-1 reference + Triton implementation.

Recurrence per `(b, h)` lane:
  alpha_t = exp(log_alpha_t), with log_alpha_t <= 0
  S_t[D]  = alpha_t * S_{t-1}[D] + V_t[D]
  y_t[D]  = S_t[D]
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

import triton
import triton.language as tl
from causal_flare.autoregressive.ssd_rank1_triton import (
    _ssd_rank1_prefix_scan_forward_impl as _ssd_single_rank1_phase2_prefix_scan_forward_impl,
    ssd_rank1_prefix_scan_bwd_dense_kernel as ssd_single_rank1_phase2_prefix_scan_bwd_dense_kernel,
)

_SUPPORTED_D_VALUES = {32, 64, 128}
_SUPPORTED_CHUNK_SIZES = {32, 64, 128, 256}
_SUPPORTED_BLOCK_X_VALUES = (128, 64, 32, 16)
_INV_LN2 = 1.4426950408889634
_USE_BF16_DOT_FOR_TENSOR_CORES = True

_EXPERIMENTAL_TRITON_ALLOCATOR_SET = False
_PHASE_BWD_WORKSPACE: dict[tuple, torch.Tensor] = {}


@dataclass(frozen=True)
class _Phase1ForwardLaunchConfig:
    block_d: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class _Phase3ForwardLaunchConfig:
    block_d: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class _Phase3BackwardLaunchConfig:
    block_d: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class _Phase1BackwardLaunchConfig:
    block_d: int
    num_warps: int
    num_stages: int


def _ensure_triton_allocator() -> None:
    global _EXPERIMENTAL_TRITON_ALLOCATOR_SET
    if _EXPERIMENTAL_TRITON_ALLOCATOR_SET:
        return

    def _alloc(size: int, _align: int, _stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(_alloc)
    _EXPERIMENTAL_TRITON_ALLOCATOR_SET = True


def _workspace_tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dev_key = (device.type, device.index if device.type == "cuda" else -1)
    key = (name, dev_key, dtype, shape)
    t = _PHASE_BWD_WORKSPACE.get(key)
    if t is None:
        t = torch.empty(shape, device=device, dtype=dtype)
        _PHASE_BWD_WORKSPACE[key] = t
    return t


def _require_nonpositive_log_alpha(log_alpha: torch.Tensor, *, where: str) -> None:
    if torch.any(log_alpha > 0):
        max_val = float(torch.max(log_alpha).item())
        raise ValueError(f"{where} requires log_alpha <= 0 everywhere; observed max(log_alpha)={max_val}.")


def _require_supported_chunk_size(chunk_size: int, *, where: str) -> None:
    if chunk_size not in _SUPPORTED_CHUNK_SIZES:
        raise NotImplementedError(
            f"{where} requires CHUNK_SIZE in {sorted(_SUPPORTED_CHUNK_SIZES)}; got CHUNK_SIZE={chunk_size}."
        )


def _require_supported_forward_contract(BH: int, N: int, *, CHUNK_SIZE: int, where: str) -> None:
    if BH % 8 != 0:
        raise NotImplementedError(f"{where} requires BH to be a multiple of 8; got BH={BH}.")
    if N % 512 != 0:
        raise NotImplementedError(f"{where} requires N to be a multiple of 512; got N={N}.")
    _require_supported_chunk_size(CHUNK_SIZE, where=where)


def _require_chunk_size_multiple_of_16(c_size: int, *, where: str) -> None:
    if c_size == 0:
        raise NotImplementedError(f"{where} does not support C==0.")
    if c_size % 16 != 0:
        raise NotImplementedError(f"{where} requires C to be a positive multiple of 16; got C={c_size}.")


def _select_largest_block_size(size: int, candidates: tuple[int, ...], *, where: str, label: str) -> int:
    for block in candidates:
        if block <= size and size % block == 0:
            return block
    raise NotImplementedError(
        f"{where} requires {label} divisible by one of {list(candidates)}; got {label}={size}."
    )


def _select_chunk_size_heuristic(*, N: int, D: int, BH: int) -> int:
    """Heuristic CHUNK_SIZE picker for the supported contract."""
    # Tuned hot-shape override:
    #   B=32, H=16, N=2048, D=64/128 -> CHUNK_SIZE=64 gave the best e2e fwd+bwd.
    if N == 2048 and BH == 512 and D in (64, 128):
        return 64
    if D >= 128:
        return 32
    if D <= 32 and N >= 4096 and BH <= 2048:
        return 128
    return 64


def _select_phase2_block_nc(*, NC: int) -> int:
    """Heuristic BLOCK_NC for phase-2 kernels."""
    if NC >= 32 and NC % 32 == 0:
        return 32
    return 16


def _select_phase2_md_launch(*, MD: int) -> tuple[int, int, int]:
    """Heuristic phase-2 BLOCK_MD/num_warps/num_stages for imported rank1 kernels."""
    if MD % 64 != 0:
        raise NotImplementedError(f"_select_phase2_md_launch requires MD divisible by 64; got MD={MD}.")
    if MD >= 8192:
        return (512 if MD % 512 == 0 else 256, 8, 3)
    if MD >= 4096:
        return (256, 4, 3)
    if MD >= 2048:
        return (128, 4, 2)
    if MD >= 128:
        return (128, 4, 2)
    return (64, 2, 2)


def _select_phase1_forward_launch_config(*, BH: int, NC: int, C_CHUNK: int, D: int) -> _Phase1ForwardLaunchConfig:
    # Tuned hot-shape override for N=2048, BH=512, CHUNK=64.
    if C_CHUNK == 64 and BH == 512 and NC == 32 and D == 64:
        return _Phase1ForwardLaunchConfig(block_d=64, num_warps=2, num_stages=2)
    if C_CHUNK == 64 and BH == 512 and NC == 32 and D == 128:
        return _Phase1ForwardLaunchConfig(block_d=128, num_warps=2, num_stages=2)

    block_d = _select_largest_block_size(D, _SUPPORTED_BLOCK_X_VALUES, where="phase1_fwd", label="D")
    work_items = BH * NC
    if C_CHUNK == 64 and block_d == 64 and work_items >= 4096:
        return _Phase1ForwardLaunchConfig(block_d=64, num_warps=2, num_stages=3)
    if block_d >= 64:
        return _Phase1ForwardLaunchConfig(block_d=block_d, num_warps=4, num_stages=2)
    return _Phase1ForwardLaunchConfig(block_d=block_d, num_warps=2, num_stages=2)


def _select_phase3_forward_launch_config(*, BH: int, NC: int, C_CHUNK: int, D: int) -> _Phase3ForwardLaunchConfig:
    # Tuned hot-shape override for N=2048, BH=512, CHUNK=64.
    if C_CHUNK == 64 and BH == 512 and NC == 32 and D == 64:
        return _Phase3ForwardLaunchConfig(block_d=64, num_warps=2, num_stages=3)
    if C_CHUNK == 64 and BH == 512 and NC == 32 and D == 128:
        return _Phase3ForwardLaunchConfig(block_d=128, num_warps=4, num_stages=2)

    block_d = _select_largest_block_size(D, _SUPPORTED_BLOCK_X_VALUES, where="phase3_fwd", label="D")
    work_items = BH * NC
    if C_CHUNK == 64 and block_d == 64 and work_items >= 4096:
        return _Phase3ForwardLaunchConfig(block_d=64, num_warps=2, num_stages=3)
    if block_d >= 64:
        return _Phase3ForwardLaunchConfig(block_d=block_d, num_warps=4, num_stages=2)
    return _Phase3ForwardLaunchConfig(block_d=block_d, num_warps=2, num_stages=2)


def _select_phase3_backward_launch_config(*, BH: int, NC: int, C_CHUNK: int, D: int) -> _Phase3BackwardLaunchConfig:
    # Tuned hot-shape override for N=2048, BH=512, CHUNK=64.
    if C_CHUNK == 64 and BH == 512 and NC == 32 and D == 64:
        return _Phase3BackwardLaunchConfig(block_d=64, num_warps=2, num_stages=3)
    if C_CHUNK == 64 and BH == 512 and NC == 32 and D == 128:
        # Retuned after enabling bf16 tensor-core dot path in phase-3 backward.
        return _Phase3BackwardLaunchConfig(block_d=32, num_warps=2, num_stages=3)

    block_d = 64 if D % 64 == 0 else _select_largest_block_size(D, (32, 16), where="phase3_bwd", label="D")
    if C_CHUNK == 64 and block_d == 64 and BH * NC >= 4096:
        return _Phase3BackwardLaunchConfig(block_d=64, num_warps=4, num_stages=2)
    if block_d >= 32:
        return _Phase3BackwardLaunchConfig(block_d=block_d, num_warps=4, num_stages=3)
    return _Phase3BackwardLaunchConfig(block_d=block_d, num_warps=2, num_stages=3)


def _select_phase1_backward_launch_config(*, D: int) -> _Phase1BackwardLaunchConfig:
    # Tuned hot-shape override for the N=2048 runs above.
    if D == 64:
        return _Phase1BackwardLaunchConfig(block_d=64, num_warps=2, num_stages=2)
    if D == 128:
        return _Phase1BackwardLaunchConfig(block_d=128, num_warps=4, num_stages=2)

    block_d = _select_largest_block_size(D, _SUPPORTED_BLOCK_X_VALUES, where="phase1_bwd", label="D")
    if D >= 128:
        return _Phase1BackwardLaunchConfig(block_d=block_d, num_warps=4, num_stages=2)
    return _Phase1BackwardLaunchConfig(block_d=block_d, num_warps=2, num_stages=2)


def _prepare_triton_inputs(
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    where: str,
    CHUNK_SIZE: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, int]:
    if V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            f"{where} expects V=[B,N,H,D], log_alpha=[B,N,H]. "
            f"Got V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    B, N, H, D = V.shape
    if log_alpha.shape != (B, N, H):
        raise ValueError(f"{where}: log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if V.device != log_alpha.device:
        raise ValueError(f"{where}: V and log_alpha must be on the same device.")
    if V.dtype != log_alpha.dtype:
        raise ValueError(f"{where}: V and log_alpha must share dtype.")
    if D not in _SUPPORTED_D_VALUES:
        raise NotImplementedError(f"{where} requires D in {sorted(_SUPPORTED_D_VALUES)}; got D={D}.")
    _require_supported_forward_contract(B * H, N, CHUNK_SIZE=CHUNK_SIZE, where=where)
    _require_nonpositive_log_alpha(log_alpha, where=where)

    NC_data = triton.cdiv(N, CHUNK_SIZE)
    padded_tokens = NC_data * CHUNK_SIZE
    token_pad = padded_tokens - N
    if token_pad > 0:
        z_v = torch.zeros((B, token_pad, H, D), device=V.device, dtype=V.dtype)
        z_r = torch.zeros((B, token_pad, H), device=log_alpha.device, dtype=log_alpha.dtype)
        V = torch.cat([V, z_v], dim=1)
        log_alpha = torch.cat([log_alpha, z_r], dim=1)

    V_chunk = (
        V.reshape(B, NC_data, CHUNK_SIZE, H, D)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(B * H, NC_data, CHUNK_SIZE, D)
    )
    log_alpha_chunk = (
        log_alpha.reshape(B, NC_data, CHUNK_SIZE, H)
        .permute(0, 3, 1, 2)
        .contiguous()
        .reshape(B * H, NC_data, CHUNK_SIZE)
    )

    NC_exec = max(NC_data, 16)
    NC_exec = triton.cdiv(NC_exec, 16) * 16
    if NC_exec > NC_data:
        pad_chunks = NC_exec - NC_data
        z_v = torch.zeros((B * H, pad_chunks, CHUNK_SIZE, D), device=V.device, dtype=V.dtype)
        z_r = torch.zeros((B * H, pad_chunks, CHUNK_SIZE), device=log_alpha.device, dtype=log_alpha.dtype)
        V_chunk = torch.cat([V_chunk, z_v], dim=1)
        log_alpha_chunk = torch.cat([log_alpha_chunk, z_r], dim=1)

    BH = B * H
    if initial_state is None:
        init_flat = torch.zeros((BH, D), device=V.device, dtype=V.dtype)
    elif initial_state.ndim == 2 and initial_state.shape == (BH, D):
        init_flat = initial_state
    elif initial_state.ndim == 3 and initial_state.shape == (B, H, D):
        init_flat = initial_state.reshape(BH, D)
    else:
        raise ValueError(
            f"{where}: initial_state must be [BH,D] or [B,H,D]. "
            f"Got {tuple(initial_state.shape)} with BH={BH}, D={D}."
        )
    if init_flat.device != V.device:
        raise ValueError(f"{where}: initial_state must be on the same device as V/log_alpha.")
    if init_flat.dtype != V.dtype:
        raise ValueError(f"{where}: initial_state must share dtype with V/log_alpha.")

    return V_chunk, log_alpha_chunk, init_flat, B, N, H, D, NC_exec


def _prepare_unchunked_inputs(
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    where: str,
    CHUNK_SIZE: int,
) -> tuple[torch.Tensor, int, int, int, int, int]:
    if V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            f"{where} expects V=[B,N,H,D], log_alpha=[B,N,H]. "
            f"Got V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    B, N, H, D = V.shape
    if log_alpha.shape != (B, N, H):
        raise ValueError(f"{where}: log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if V.device != log_alpha.device:
        raise ValueError(f"{where}: V and log_alpha must be on the same device.")
    if V.dtype != log_alpha.dtype:
        raise ValueError(f"{where}: V and log_alpha must share dtype.")
    if D not in _SUPPORTED_D_VALUES:
        raise NotImplementedError(f"{where} requires D in {sorted(_SUPPORTED_D_VALUES)}; got D={D}.")
    _require_supported_forward_contract(B * H, N, CHUNK_SIZE=CHUNK_SIZE, where=where)
    _require_nonpositive_log_alpha(log_alpha, where=where)

    BH = B * H
    if initial_state is None:
        init_flat = torch.zeros((BH, D), device=V.device, dtype=V.dtype)
    elif initial_state.ndim == 2 and initial_state.shape == (BH, D):
        init_flat = initial_state
    elif initial_state.ndim == 3 and initial_state.shape == (B, H, D):
        init_flat = initial_state.reshape(BH, D)
    else:
        raise ValueError(
            f"{where}: initial_state must be [BH,D] or [B,H,D]. "
            f"Got {tuple(initial_state.shape)} with BH={BH}, D={D}."
        )
    if init_flat.device != V.device:
        raise ValueError(f"{where}: initial_state must be on the same device as V/log_alpha.")
    if init_flat.dtype != V.dtype:
        raise ValueError(f"{where}: initial_state must share dtype with V/log_alpha.")

    NC_data = triton.cdiv(N, CHUNK_SIZE)
    NC_exec = max(NC_data, 16)
    NC_exec = triton.cdiv(NC_exec, 16) * 16
    return init_flat, B, N, H, D, NC_exec


def _restore_output_layout(
    y_chunk: torch.Tensor,
    final_state_flat: torch.Tensor,
    *,
    B: int,
    N: int,
    H: int,
    C: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    D = y_chunk.shape[-1]
    y = (
        y_chunk.reshape(B, H, y_chunk.shape[1], C, D)
        .permute(0, 2, 3, 1, 4)
        .reshape(B, y_chunk.shape[1] * C, H, D)
    )[:, :N]
    final_state = final_state_flat.reshape(B, H, D)
    return y, final_state


def _reshape_log_alpha_per_chunk_to_bnh(
    log_alpha_per_chunk: torch.Tensor,
    *,
    B: int,
    H: int,
    NC: int,
    where: str,
) -> torch.Tensor:
    BH = B * H
    if log_alpha_per_chunk.ndim == 2 and log_alpha_per_chunk.shape == (BH, NC):
        return log_alpha_per_chunk.reshape(B, H, NC).permute(0, 2, 1).contiguous()
    if log_alpha_per_chunk.ndim == 3 and log_alpha_per_chunk.shape == (B, NC, H):
        return log_alpha_per_chunk if log_alpha_per_chunk.is_contiguous() else log_alpha_per_chunk.contiguous()
    raise ValueError(
        f"{where}: log_alpha_per_chunk must be [BH,NC] or [B,NC,H]. "
        f"Got {tuple(log_alpha_per_chunk.shape)} with B={B}, H={H}, NC={NC}."
    )


# --------------------------------------------------------------------------------------------------
# ORACLE / PYTORCH REFERENCE
# --------------------------------------------------------------------------------------------------
def ssd_single_rank1_token_loop_oracle(
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "ssd_single_rank1_token_loop_oracle expects V=[B,N,H,D], log_alpha=[B,N,H]. "
            f"Got V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    B, N, H, D = V.shape
    if log_alpha.shape != (B, N, H):
        raise ValueError(f"log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if V.device != log_alpha.device:
        raise ValueError("V and log_alpha must be on the same device.")
    if V.dtype != log_alpha.dtype:
        raise ValueError("V and log_alpha must share dtype.")
    if N == 0:
        raise NotImplementedError("ssd_single_rank1_token_loop_oracle does not support N==0.")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_single_rank1_token_loop_oracle")

    BH = B * H
    if initial_state is None:
        S = torch.zeros((B, H, D), device=V.device, dtype=V.dtype)
    elif initial_state.ndim == 2 and initial_state.shape == (BH, D):
        S = initial_state.reshape(B, H, D)
    elif initial_state.ndim == 3 and initial_state.shape == (B, H, D):
        S = initial_state
    else:
        raise ValueError(
            f"initial_state must be [BH,D] or [B,H,D]. "
            f"Got {tuple(initial_state.shape)} with BH={BH}, D={D}."
        )
    if S.device != V.device:
        raise ValueError("initial_state must be on the same device as V/log_alpha.")
    if S.dtype != V.dtype:
        raise ValueError("initial_state must share dtype with V/log_alpha.")

    alpha = torch.exp2(log_alpha * _INV_LN2)
    ys = []
    for t in range(N):
        S = alpha[:, t].unsqueeze(-1) * S + V[:, t]
        ys.append(S)
    y = torch.stack(ys, dim=1)
    return y, S


def ssd_single_rank1_chunk_end_state_reference(
    V: torch.Tensor,
    log_alpha: torch.Tensor,
) -> torch.Tensor:
    if V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "ssd_single_rank1_chunk_end_state_reference expects V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
            f"Got V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    BH, NC, C, D = V.shape
    if log_alpha.shape != (BH, NC, C):
        raise ValueError(f"log_alpha must be [BH,NC,C]={BH, NC, C}; got {tuple(log_alpha.shape)}.")
    if V.device != log_alpha.device:
        raise ValueError("V and log_alpha must be on the same device.")
    if V.dtype != log_alpha.dtype:
        raise ValueError("V and log_alpha must share dtype.")
    if C == 0:
        raise NotImplementedError("ssd_single_rank1_chunk_end_state_reference does not support C==0.")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_single_rank1_chunk_end_state_reference")

    log_alpha_f = log_alpha.float()
    V_f = V.float()
    log_alpha_rev = torch.flip(log_alpha_f, dims=[-1])
    log_suffix_incl_rev = torch.cumsum(log_alpha_rev, dim=-1)
    log_suffix_excl_rev = log_suffix_incl_rev - log_alpha_rev
    r = torch.flip(torch.exp2(log_suffix_excl_rev * _INV_LN2), dims=[-1])
    return torch.sum(r.unsqueeze(-1) * V_f, dim=-2)


def ssd_single_rank1_prefix_scan_reference(
    S_local_end: torch.Tensor,
    alpha_chunk: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if S_local_end.ndim != 3 or alpha_chunk.ndim != 2 or initial_state.ndim != 2:
        raise ValueError(
            "ssd_single_rank1_prefix_scan_reference expects "
            "S_local_end=[BH,NC,D], alpha_chunk=[BH,NC], initial_state=[BH,D]."
        )
    BH, NC, D = S_local_end.shape
    if alpha_chunk.shape != (BH, NC):
        raise ValueError(f"alpha_chunk must be [BH,NC]={BH, NC}; got {tuple(alpha_chunk.shape)}.")
    if initial_state.shape != (BH, D):
        raise ValueError(f"initial_state must be [BH,D]={BH, D}; got {tuple(initial_state.shape)}.")
    if NC == 0:
        raise NotImplementedError("ssd_single_rank1_prefix_scan_reference does not support NC==0.")

    scan_r = alpha_chunk.float()
    scan_s = S_local_end.float()
    initial_state_f = initial_state.float()
    step = 1
    while step < NC:
        shifted_r = torch.cat([torch.ones_like(scan_r[:, :step]), scan_r[:, :-step]], dim=1)
        shifted_s = torch.cat([torch.zeros_like(scan_s[:, :step]), scan_s[:, :-step]], dim=1)
        scan_s = scan_s + scan_r.unsqueeze(-1) * shifted_s
        scan_r = scan_r * shifted_r
        step <<= 1

    excl_r = torch.cat([torch.ones_like(scan_r[:, :1]), scan_r[:, :-1]], dim=1)
    excl_s = torch.cat([torch.zeros_like(scan_s[:, :1]), scan_s[:, :-1]], dim=1)
    chunk_start = excl_r.unsqueeze(-1) * initial_state_f.unsqueeze(1) + excl_s
    final_state = scan_r[:, -1].unsqueeze(-1) * initial_state_f + scan_s[:, -1]
    return chunk_start, final_state


def ssd_single_rank1_dense_output_reference(
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    S0: torch.Tensor | None = None,
) -> torch.Tensor:
    if V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "ssd_single_rank1_dense_output_reference expects V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
            f"Got V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    BH, NC, C, D = V.shape
    if log_alpha.shape != (BH, NC, C):
        raise ValueError(f"log_alpha must be [BH,NC,C]={BH, NC, C}; got {tuple(log_alpha.shape)}.")
    if V.device != log_alpha.device:
        raise ValueError("V and log_alpha must be on the same device.")
    if V.dtype != log_alpha.dtype:
        raise ValueError("V and log_alpha must share dtype.")
    if C == 0:
        raise NotImplementedError("ssd_single_rank1_dense_output_reference does not support C==0.")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_single_rank1_dense_output_reference")

    if S0 is None:
        S0_vec = torch.zeros((BH, NC, D), device=V.device, dtype=torch.float32)
    elif S0.ndim == 2 and S0.shape == (BH, D):
        S0_vec = S0[:, None, :].expand(BH, NC, D).to(torch.float32)
    elif S0.ndim == 3 and S0.shape == (BH, NC, D):
        S0_vec = S0.to(torch.float32)
    else:
        raise ValueError(f"S0 must be [BH,D] or [BH,NC,D]. Got {tuple(S0.shape)}.")

    V_f = V.float()
    log_alpha_f = log_alpha.float()

    c_idx = torch.arange(C, device=V.device)
    tril = (c_idx.view(C, 1) >= c_idx.view(1, C)).view(1, 1, C, C)

    log_p = torch.cumsum(log_alpha_f, dim=-1)
    log_delta = (log_p[..., :, None] - log_p[..., None, :]) * _INV_LN2
    log_delta = torch.where(tril, log_delta, torch.zeros_like(log_delta))
    L = torch.where(tril, torch.exp2(log_delta), torch.zeros_like(log_delta))

    p = torch.exp2(log_p * _INV_LN2).unsqueeze(-1)
    Y_off = p * S0_vec.unsqueeze(-2)
    Y_diag = torch.matmul(L, V_f)
    return (Y_diag + Y_off).to(V.dtype)


def ssd_single_rank1_dense_output_backward_reference(
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    grad_Y: torch.Tensor,
    S0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if V.ndim != 4 or log_alpha.ndim != 3 or grad_Y.ndim != 4:
        raise ValueError(
            "ssd_single_rank1_dense_output_backward_reference expects "
            "V=[BH,NC,C,D], log_alpha=[BH,NC,C], grad_Y=[BH,NC,C,D]."
        )
    BH, NC, C, D = V.shape
    if log_alpha.shape != (BH, NC, C):
        raise ValueError(f"log_alpha must be [BH,NC,C]={BH, NC, C}; got {tuple(log_alpha.shape)}.")
    if grad_Y.shape != (BH, NC, C, D):
        raise ValueError(f"grad_Y must be [BH,NC,C,D]={BH, NC, C, D}; got {tuple(grad_Y.shape)}.")
    if V.device != log_alpha.device or V.device != grad_Y.device:
        raise ValueError("V, log_alpha, grad_Y must be on the same device.")
    if V.dtype != log_alpha.dtype or V.dtype != grad_Y.dtype:
        raise ValueError("V, log_alpha, grad_Y must share dtype.")
    if C == 0:
        raise NotImplementedError("ssd_single_rank1_dense_output_backward_reference does not support C==0.")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_single_rank1_dense_output_backward_reference")

    if S0 is None:
        S0_vec = torch.zeros((BH, NC, D), device=V.device, dtype=torch.float32)
        return_s0_grad = False
    elif S0.ndim == 2 and S0.shape == (BH, D):
        S0_vec = S0[:, None, :].expand(BH, NC, D).to(torch.float32)
        return_s0_grad = True
    elif S0.ndim == 3 and S0.shape == (BH, NC, D):
        S0_vec = S0.to(torch.float32)
        return_s0_grad = True
    else:
        raise ValueError(f"S0 must be [BH,D] or [BH,NC,D]. Got {tuple(S0.shape)}.")

    V_f = V.float()
    log_alpha_f = log_alpha.float()
    grad_Y_f = grad_Y.float()

    c_idx = torch.arange(C, device=V.device)
    tril = (c_idx.view(C, 1) >= c_idx.view(1, C)).view(1, 1, C, C)

    log_p = torch.cumsum(log_alpha_f, dim=-1)
    p = torch.exp2(log_p * _INV_LN2)
    log_delta = (log_p[..., :, None] - log_p[..., None, :]) * _INV_LN2
    log_delta = torch.where(tril, log_delta, torch.zeros_like(log_delta))
    L = torch.where(tril, torch.exp2(log_delta), torch.zeros_like(log_delta))

    dV = L.mT @ grad_Y_f
    dL = grad_Y_f @ V_f.mT

    Q = dL * L
    left_prefix = torch.cumsum(Q, dim=-1)
    left_of_k = torch.zeros_like(Q)
    left_of_k[..., :, 1:] = left_prefix[..., :, :-1]
    suffix_rows = torch.flip(torch.cumsum(torch.flip(left_of_k, dims=[-2]), dim=-2), dims=[-2])
    dlog_diag = torch.diagonal(suffix_rows, dim1=-2, dim2=-1)

    dlog_off_src = (grad_Y_f * S0_vec.unsqueeze(-2)).sum(dim=-1) * p
    dlog_off = torch.flip(torch.cumsum(torch.flip(dlog_off_src, dims=[-1]), dim=-1), dims=[-1])
    dlog = dlog_diag + dlog_off

    dS0 = (grad_Y_f * p.unsqueeze(-1)).sum(dim=-2).to(V.dtype) if return_s0_grad else None
    return dV.to(V.dtype), dlog.to(log_alpha.dtype), dS0


def ssd_single_rank1_pytorch(
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if CHUNK_SIZE is None:
        CHUNK_SIZE = 64
    V_chunk, log_alpha_chunk, init_flat, B, N, H, _D, _NC = _prepare_triton_inputs(
        V, log_alpha, initial_state, where="ssd_single_rank1_pytorch", CHUNK_SIZE=CHUNK_SIZE
    )
    S_local_end = ssd_single_rank1_chunk_end_state_reference(V_chunk, log_alpha_chunk)
    alpha_chunk = torch.exp2(torch.sum(log_alpha_chunk.float(), dim=-1) * _INV_LN2)
    S0_chunk, S1_chunk = ssd_single_rank1_prefix_scan_reference(S_local_end, alpha_chunk, init_flat)
    y_chunk = ssd_single_rank1_dense_output_reference(V_chunk, log_alpha_chunk, S0_chunk)
    return _restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)


# --------------------------------------------------------------------------------------------------
# PHASE 1: LOCAL CHUNK END-STATE (TRITON)
# --------------------------------------------------------------------------------------------------
@triton.jit
def ssd_single_rank1_phase1_fwd_un_kernel(
    V_ptr,
    log_alpha_ptr,
    out_s_end_ptr,
    out_log_chunk_ptr,
    b_size,
    h_size,
    n_size,
    nc_size,
    c_size,
    d_size,
    stride_v_b,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_r_b,
    stride_r_n,
    stride_r_h,
    stride_out_bh,
    stride_out_nc,
    stride_out_d,
    stride_log_bh,
    stride_log_nc,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    D_STATIC: tl.constexpr,
):
    pid_bhnc = tl.program_id(0)
    pid_d_tile = tl.program_id(1)
    bh_size = b_size * h_size
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size
    d_start = pid_d_tile * BLOCK_D

    offs_c = tl.arange(0, BLOCK_C)
    offs_d = d_start + tl.arange(0, BLOCK_D)
    n_idx = NC_IDX * BLOCK_C + offs_c
    mask_c = n_idx < n_size
    mask_d = offs_d < d_size
    mask_cd = mask_c[:, None] & mask_d[None, :]

    log_ptrs = log_alpha_ptr + B_IDX * stride_r_b + n_idx * stride_r_n + H_IDX * stride_r_h
    log_alpha_vals = tl.load(log_ptrs, mask=mask_c, other=0.0).to(tl.float32)

    if pid_d_tile == 0:
        chunk_sum = tl.sum(log_alpha_vals, axis=0)
        tl.store(out_log_chunk_ptr + BH_IDX * stride_log_bh + NC_IDX * stride_log_nc, chunk_sum)

    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    factors = tl.exp2((log_suffix_incl - log_alpha_vals) * 1.4426950408889634)

    v_ptrs = (
        V_ptr
        + B_IDX * stride_v_b
        + n_idx[:, None] * stride_v_n
        + H_IDX * stride_v_h
        + offs_d[None, :] * stride_v_d
    )
    V_f = tl.load(v_ptrs, mask=mask_cd, other=0.0).to(tl.float32)
    s_end = tl.sum(factors[:, None] * V_f, axis=0)

    out_ptrs = out_s_end_ptr + BH_IDX * stride_out_bh + NC_IDX * stride_out_nc + offs_d * stride_out_d
    tl.store(out_ptrs, s_end, mask=mask_d)


def _phase1_forward_replay_from_unchunked(
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    *,
    H: int,
    NC: int,
    CHUNK_SIZE: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError("_phase1_forward_replay_from_unchunked expects V=[B,N,H,D], log_alpha=[B,N,H].")
    B, N, H_in, D = V.shape
    if H_in != H:
        raise ValueError(f"H mismatch: expected {H}, got {H_in}.")
    if log_alpha.shape != (B, N, H):
        raise ValueError(f"log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if not V.is_cuda:
        raise NotImplementedError("_phase1_forward_replay_from_unchunked requires CUDA tensors.")
    _require_chunk_size_multiple_of_16(CHUNK_SIZE, where="_phase1_forward_replay_from_unchunked")
    if D not in _SUPPORTED_D_VALUES:
        raise NotImplementedError(
            f"_phase1_forward_replay_from_unchunked requires D in {sorted(_SUPPORTED_D_VALUES)}; got D={D}."
        )
    _ensure_triton_allocator()

    BH = B * H
    cfg = _select_phase1_forward_launch_config(BH=BH, NC=NC, C_CHUNK=CHUNK_SIZE, D=D)
    BLOCK_D = cfg.block_d

    S_local_end = torch.empty((BH, NC, D), device=V.device, dtype=torch.float32)
    log_chunk = torch.empty((BH, NC), device=V.device, dtype=torch.float32)
    grid = (BH * NC, D // BLOCK_D)
    ssd_single_rank1_phase1_fwd_un_kernel[grid](
        V,
        log_alpha,
        S_local_end,
        log_chunk,
        B,
        H,
        N,
        NC,
        CHUNK_SIZE,
        D,
        *V.stride(),
        *log_alpha.stride(),
        *S_local_end.stride(),
        *log_chunk.stride(),
        BLOCK_D=BLOCK_D,
        BLOCK_C=CHUNK_SIZE,
        D_STATIC=D,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    return S_local_end, log_chunk


# --------------------------------------------------------------------------------------------------
# PHASE 3: CHUNK DENSE REPLAY (TRITON)
# --------------------------------------------------------------------------------------------------
@triton.jit
def ssd_single_rank1_phase3_fwd_un_kernel(
    V_ptr,
    log_alpha_ptr,
    S0_ptr,
    out_y_ptr,
    out_y_off_ptr,
    b_size,
    h_size,
    n_size,
    nc_size,
    c_size,
    d_size,
    stride_v_b,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_r_b,
    stride_r_n,
    stride_r_h,
    stride_s0_bh,
    stride_s0_nc,
    stride_s0_d,
    stride_out_y_bh,
    stride_out_y_nc,
    stride_out_y_c,
    stride_out_y_d,
    stride_out_y_off_bh,
    stride_out_y_off_nc,
    stride_out_y_off_c,
    stride_out_y_off_d,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    D_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    STORE_Y_OFF: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
):
    pid_bhnc = tl.program_id(0)
    bh_size = b_size * h_size
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    offs_c = tl.arange(0, BLOCK_C)
    n_idx = NC_IDX * BLOCK_C + offs_c
    mask_c = n_idx < n_size

    log_ptrs = log_alpha_ptr + B_IDX * stride_r_b + n_idx * stride_r_n + H_IDX * stride_r_h
    log_alpha_vals = tl.load(log_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    valid = offs_c[None, :] <= offs_c[:, None]
    log_p = tl.cumsum(log_alpha_vals, axis=0)
    log_delta = (log_p[:, None] - log_p[None, :]) * 1.4426950408889634
    log_delta = tl.where(valid, log_delta, 0.0)
    L = tl.where(valid, tl.exp2(log_delta), 0.0)
    p = tl.exp2(log_p * 1.4426950408889634)
    if USE_BF16_DOT:
        L_tc = L.to(tl.bfloat16)

    for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
        d_start = d_blk * BLOCK_D
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d_size
        mask_cd = mask_c[:, None] & mask_d[None, :]

        v_ptrs = (
            V_ptr
            + B_IDX * stride_v_b
            + n_idx[:, None] * stride_v_n
            + H_IDX * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        V_in = tl.load(v_ptrs, mask=mask_cd, other=0.0)
        V_f = V_in.to(tl.float32)
        if USE_BF16_DOT:
            Y_diag = tl.dot(L_tc, V_f.to(tl.bfloat16), out_dtype=tl.float32)
        else:
            Y_diag = tl.dot(L, V_f, out_dtype=tl.float32, input_precision=INPUT_PRECISION)

        s0_ptrs = S0_ptr + BH_IDX * stride_s0_bh + NC_IDX * stride_s0_nc + offs_d * stride_s0_d
        S0_tile = tl.load(s0_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        Y_off = p[:, None] * S0_tile[None, :]
        Y = Y_diag + Y_off

        out_ptrs = (
            out_y_ptr
            + BH_IDX * stride_out_y_bh
            + NC_IDX * stride_out_y_nc
            + offs_c[:, None] * stride_out_y_c
            + offs_d[None, :] * stride_out_y_d
        )
        tl.store(out_ptrs, Y.to(V_in.dtype), mask=mask_cd)

        if STORE_Y_OFF:
            out_off_ptrs = (
                out_y_off_ptr
                + BH_IDX * stride_out_y_off_bh
                + NC_IDX * stride_out_y_off_nc
                + offs_c[:, None] * stride_out_y_off_c
                + offs_d[None, :] * stride_out_y_off_d
            )
            tl.store(out_off_ptrs, Y_off.to(V_in.dtype), mask=mask_cd)


def _phase3_forward_from_unchunked(
    V_un: torch.Tensor,
    log_alpha_un: torch.Tensor,
    S0: torch.Tensor,
    *,
    H: int,
    NC: int,
    CHUNK_SIZE: int,
    input_precision: str,
    RETURN_Y_OFF: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if V_un.ndim != 4 or log_alpha_un.ndim != 3:
        raise ValueError("_phase3_forward_from_unchunked expects V=[B,N,H,D], log_alpha=[B,N,H].")
    B, N, H_in, D = V_un.shape
    if H_in != H:
        raise ValueError(f"H mismatch: expected {H}, got {H_in}.")
    if log_alpha_un.shape != (B, N, H):
        raise ValueError(f"log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha_un.shape)}.")
    BH = B * H
    C = CHUNK_SIZE
    if S0.shape != (BH, NC, D):
        raise ValueError(f"S0 must be [BH,NC,D]={BH, NC, D}; got {tuple(S0.shape)}.")
    _require_chunk_size_multiple_of_16(C, where="_phase3_forward_from_unchunked")
    if D not in _SUPPORTED_D_VALUES:
        raise NotImplementedError(
            f"_phase3_forward_from_unchunked requires D in {sorted(_SUPPORTED_D_VALUES)}; got D={D}."
        )
    if not V_un.is_cuda:
        raise NotImplementedError("_phase3_forward_from_unchunked requires CUDA tensors.")
    _ensure_triton_allocator()

    cfg = _select_phase3_forward_launch_config(BH=BH, NC=NC, C_CHUNK=C, D=D)
    BLOCK_D = cfg.block_d
    out = torch.empty((BH, NC, C, D), device=V_un.device, dtype=V_un.dtype)
    if RETURN_Y_OFF:
        y_off: torch.Tensor | None = torch.empty_like(out)
        out_off = y_off
    else:
        y_off = None
        out_off = out
    grid = (BH * NC,)
    ssd_single_rank1_phase3_fwd_un_kernel[grid](
        V_un,
        log_alpha_un,
        S0,
        out,
        out_off,
        B,
        H,
        N,
        NC,
        C,
        D,
        *V_un.stride(),
        *log_alpha_un.stride(),
        *S0.stride(),
        *out.stride(),
        *out_off.stride(),
        BLOCK_D=BLOCK_D,
        BLOCK_C=C,
        D_STATIC=D,
        INPUT_PRECISION=input_precision,
        STORE_Y_OFF=RETURN_Y_OFF,
        USE_BF16_DOT=_USE_BF16_DOT_FOR_TENSOR_CORES,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    if RETURN_Y_OFF:
        assert y_off is not None
        return out, y_off
    return out


@triton.jit
def ssd_single_rank1_phase3_bwd_un_kernel(
    V_ptr,
    log_alpha_ptr,
    grad_y_ptr,
    S0_ptr,
    out_dV_ptr,
    out_dlog_alpha_ptr,
    out_dS0_ptr,
    b_size,
    h_size,
    n_size,
    nc_size,
    c_size,
    d_size,
    stride_v_b,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_r_b,
    stride_r_n,
    stride_r_h,
    stride_gy_bh,
    stride_gy_nc,
    stride_gy_c,
    stride_gy_d,
    stride_s0_bh,
    stride_s0_nc,
    stride_s0_d,
    stride_dv_b,
    stride_dv_n,
    stride_dv_h,
    stride_dv_d,
    stride_dlog_alpha_bh,
    stride_dlog_alpha_nc,
    stride_dlog_alpha_c,
    stride_ds0_bh,
    stride_ds0_nc,
    stride_ds0_d,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    D_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    USE_BF16_DOT: tl.constexpr,
):
    pid_bhnc = tl.program_id(0)
    bh_size = b_size * h_size
    if pid_bhnc >= bh_size * nc_size:
        return
    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    offs_c = tl.arange(0, BLOCK_C)
    n_idx = NC_IDX * BLOCK_C + offs_c
    mask_c = n_idx < n_size

    log_ptrs = log_alpha_ptr + B_IDX * stride_r_b + n_idx * stride_r_n + H_IDX * stride_r_h
    log_alpha_vals = tl.load(log_ptrs, mask=mask_c, other=0.0).to(tl.float32)

    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    valid_tri = col_idx <= row_idx
    log_p = tl.cumsum(log_alpha_vals, axis=0)
    p = tl.exp2(log_p * 1.4426950408889634)
    log_delta = (log_p[:, None] - log_p[None, :]) * 1.4426950408889634
    log_delta = tl.where(valid_tri, log_delta, 0.0)
    L = tl.where(valid_tri, tl.exp2(log_delta), 0.0)
    if USE_BF16_DOT:
        # Reuse the static CxC conversion across all D tiles in this program.
        L_t_tc = tl.trans(L).to(tl.bfloat16)

    dL = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    src = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
        d_start = d_blk * BLOCK_D
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d_size
        mask_cd = mask_c[:, None] & mask_d[None, :]

        v_ptrs = (
            V_ptr
            + B_IDX * stride_v_b
            + n_idx[:, None] * stride_v_n
            + H_IDX * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        V_f = tl.load(v_ptrs, mask=mask_cd, other=0.0).to(tl.float32)

        gy_ptrs = (
            grad_y_ptr
            + BH_IDX * stride_gy_bh
            + NC_IDX * stride_gy_nc
            + offs_c[:, None] * stride_gy_c
            + offs_d[None, :] * stride_gy_d
        )
        G = tl.load(gy_ptrs, mask=mask_cd, other=0.0).to(tl.float32)

        if USE_BF16_DOT:
            G_tc = G.to(tl.bfloat16)
            V_t_tc = tl.trans(V_f).to(tl.bfloat16)
            dV_tile = tl.dot(L_t_tc, G_tc, out_dtype=tl.float32)
            dL += tl.dot(G_tc, V_t_tc, out_dtype=tl.float32)
        else:
            dV_tile = tl.dot(tl.trans(L), G, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dL += tl.dot(G, tl.trans(V_f), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        dv_ptrs = (
            out_dV_ptr
            + B_IDX * stride_dv_b
            + n_idx[:, None] * stride_dv_n
            + H_IDX * stride_dv_h
            + offs_d[None, :] * stride_dv_d
        )
        tl.store(dv_ptrs, dV_tile, mask=mask_cd)

        s0_ptrs = S0_ptr + BH_IDX * stride_s0_bh + NC_IDX * stride_s0_nc + offs_d * stride_s0_d
        S0_tile = tl.load(s0_ptrs, mask=mask_d, other=0.0).to(tl.float32)
        ds0_tile = tl.sum(p[:, None] * G, axis=0)
        ds0_ptrs = out_dS0_ptr + BH_IDX * stride_ds0_bh + NC_IDX * stride_ds0_nc + offs_d * stride_ds0_d
        tl.store(ds0_ptrs, ds0_tile, mask=mask_d)

        src += p * tl.sum(G * S0_tile[None, :], axis=1)

    Q = dL * L
    left_prefix = tl.cumsum(Q, axis=1)
    left_of = left_prefix - Q
    suffix_rows = tl.flip(tl.cumsum(tl.flip(left_of, 0), axis=0), 0)
    is_diag = offs_c[:, None] == offs_c[None, :]
    dlog_diag = tl.sum(tl.where(is_diag, suffix_rows, 0.0), axis=1)
    dlog_off = tl.cumsum(src, axis=0, reverse=True)
    dlog_alpha = dlog_diag + dlog_off
    dlog_alpha_ptrs = (
        out_dlog_alpha_ptr
        + BH_IDX * stride_dlog_alpha_bh
        + NC_IDX * stride_dlog_alpha_nc
        + offs_c * stride_dlog_alpha_c
    )
    tl.store(dlog_alpha_ptrs, dlog_alpha)


def _phase3_backward_from_unchunked(
    V_un: torch.Tensor,
    log_alpha_un: torch.Tensor,
    grad_y_chunk: torch.Tensor,
    S0: torch.Tensor,
    *,
    H: int,
    NC: int,
    CHUNK_SIZE: int,
    input_precision: str,
    return_s0_grad_fp32: bool = False,
    dlog_alpha_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if V_un.ndim != 4 or log_alpha_un.ndim != 3:
        raise ValueError("_phase3_backward_from_unchunked expects V=[B,N,H,D], log_alpha=[B,N,H].")
    B, N, H_in, D = V_un.shape
    if H_in != H:
        raise ValueError(f"H mismatch: expected {H}, got {H_in}.")
    if log_alpha_un.shape != (B, N, H):
        raise ValueError(f"log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha_un.shape)}.")
    BH = B * H
    C = CHUNK_SIZE
    if grad_y_chunk.shape != (BH, NC, C, D):
        raise ValueError(f"grad_y_chunk must be [BH,NC,C,D]={BH, NC, C, D}; got {tuple(grad_y_chunk.shape)}.")
    if S0.shape != (BH, NC, D):
        raise ValueError(f"S0 must be [BH,NC,D]={BH, NC, D}; got {tuple(S0.shape)}.")
    _require_chunk_size_multiple_of_16(C, where="_phase3_backward_from_unchunked")
    _ensure_triton_allocator()

    cfg = _select_phase3_backward_launch_config(BH=BH, NC=NC, C_CHUNK=C, D=D)
    BLOCK_D = cfg.block_d
    grad_y_in = grad_y_chunk if grad_y_chunk.is_contiguous() else grad_y_chunk.contiguous()

    dV_out = torch.empty_like(V_un)
    if dlog_alpha_out is None:
        dlog_alpha = torch.empty((BH, NC, C), device=V_un.device, dtype=log_alpha_un.dtype)
    else:
        if dlog_alpha_out.shape != (BH, NC, C):
            raise ValueError(f"dlog_alpha_out must be [BH,NC,C]={BH, NC, C}; got {tuple(dlog_alpha_out.shape)}.")
        if dlog_alpha_out.device != V_un.device:
            raise ValueError("dlog_alpha_out must be on the same device as inputs.")
        if dlog_alpha_out.dtype not in (log_alpha_un.dtype, torch.float32):
            raise ValueError("dlog_alpha_out must match log_alpha dtype or be float32.")
        dlog_alpha = dlog_alpha_out
    dS0 = torch.empty((BH, NC, D), device=V_un.device, dtype=torch.float32)
    grid = (BH * NC,)
    ssd_single_rank1_phase3_bwd_un_kernel[grid](
        V_un,
        log_alpha_un,
        grad_y_in,
        S0,
        dV_out,
        dlog_alpha,
        dS0,
        B,
        H,
        N,
        NC,
        C,
        D,
        *V_un.stride(),
        *log_alpha_un.stride(),
        *grad_y_in.stride(),
        *S0.stride(),
        *dV_out.stride(),
        *dlog_alpha.stride(),
        *dS0.stride(),
        BLOCK_D=BLOCK_D,
        BLOCK_C=C,
        D_STATIC=D,
        INPUT_PRECISION=input_precision,
        USE_BF16_DOT=_USE_BF16_DOT_FOR_TENSOR_CORES,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
    )
    dS0_out = dS0 if return_s0_grad_fp32 else dS0.to(V_un.dtype)
    return dV_out, dlog_alpha, dS0_out


# --------------------------------------------------------------------------------------------------
# PHASE 1 BACKWARD (TRITON)
# --------------------------------------------------------------------------------------------------
@triton.jit
def ssd_single_rank1_phase1_bwd_un_accum_kernel(
    grad_s_ptr,
    V_ptr,
    log_alpha_ptr,
    out_dV_ptr,
    out_dlog_alpha_ptr,
    b_size,
    h_size,
    n_size,
    nc_size,
    c_size,
    d_size,
    stride_gs_bh,
    stride_gs_nc,
    stride_gs_d,
    stride_v_b,
    stride_v_n,
    stride_v_h,
    stride_v_d,
    stride_r_b,
    stride_r_n,
    stride_r_h,
    stride_dv_b,
    stride_dv_n,
    stride_dv_h,
    stride_dv_d,
    stride_dlog_alpha_bh,
    stride_dlog_alpha_nc,
    stride_dlog_alpha_c,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    D_STATIC: tl.constexpr,
):
    pid_bhnc = tl.program_id(0)
    bh_size = b_size * h_size
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    offs_c = tl.arange(0, BLOCK_C)
    n_idx = NC_IDX * BLOCK_C + offs_c
    mask_c = n_idx < n_size

    log_ptrs = log_alpha_ptr + B_IDX * stride_r_b + n_idx * stride_r_n + H_IDX * stride_r_h
    log_alpha_vals = tl.load(log_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    factors = tl.exp2((log_suffix_incl - log_alpha_vals) * 1.4426950408889634)

    b_vals = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
        d_start = d_blk * BLOCK_D
        offs_d = d_start + tl.arange(0, BLOCK_D)
        mask_d = offs_d < d_size
        mask_cd = mask_c[:, None] & mask_d[None, :]

        gs_ptrs = grad_s_ptr + BH_IDX * stride_gs_bh + NC_IDX * stride_gs_nc + offs_d * stride_gs_d
        g = tl.load(gs_ptrs, mask=mask_d, other=0.0).to(tl.float32)

        dv_ptrs = (
            out_dV_ptr
            + B_IDX * stride_dv_b
            + n_idx[:, None] * stride_dv_n
            + H_IDX * stride_dv_h
            + offs_d[None, :] * stride_dv_d
        )
        prev_dV = tl.load(dv_ptrs, mask=mask_cd, other=0.0).to(tl.float32)
        dV_add = factors[:, None] * g[None, :]
        tl.store(dv_ptrs, prev_dV + dV_add, mask=mask_cd)

        v_ptrs = (
            V_ptr
            + B_IDX * stride_v_b
            + n_idx[:, None] * stride_v_n
            + H_IDX * stride_v_h
            + offs_d[None, :] * stride_v_d
        )
        V_f = tl.load(v_ptrs, mask=mask_cd, other=0.0).to(tl.float32)
        b_vals += factors * tl.sum(V_f * g[None, :], axis=1)

    dlog_alpha = tl.cumsum(b_vals, axis=0) - b_vals
    dlog_alpha_ptrs = (
        out_dlog_alpha_ptr
        + BH_IDX * stride_dlog_alpha_bh
        + NC_IDX * stride_dlog_alpha_nc
        + offs_c * stride_dlog_alpha_c
    )
    prev_dlog_alpha = tl.load(dlog_alpha_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    tl.store(dlog_alpha_ptrs, prev_dlog_alpha + dlog_alpha, mask=mask_c)


def _phase1_backward_from_unchunked_accum(
    grad_s: torch.Tensor,
    V_un: torch.Tensor,
    log_alpha_un: torch.Tensor,
    dV_out: torch.Tensor,
    *,
    H: int,
    NC: int,
    CHUNK_SIZE: int,
    dlog_alpha_out: torch.Tensor | None = None,
) -> torch.Tensor:
    if grad_s.ndim != 3 or V_un.ndim != 4 or log_alpha_un.ndim != 3 or dV_out.ndim != 4:
        raise ValueError(
            "_phase1_backward_from_unchunked_accum expects "
            "grad_s=[BH,NC,D], V=[B,N,H,D], log_alpha=[B,N,H], dV_out=[B,N,H,D]."
        )
    B, N, H_in, D = V_un.shape
    if H_in != H:
        raise ValueError(f"H mismatch: expected {H}, got {H_in}.")
    if log_alpha_un.shape != (B, N, H):
        raise ValueError(f"log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha_un.shape)}.")
    if dV_out.shape != (B, N, H, D):
        raise ValueError(f"dV_out must be [B,N,H,D]={B, N, H, D}; got {tuple(dV_out.shape)}.")
    BH = B * H
    C = CHUNK_SIZE
    if grad_s.shape != (BH, NC, D):
        raise ValueError(f"grad_s must be [BH,NC,D]={BH, NC, D}; got {tuple(grad_s.shape)}.")
    _ensure_triton_allocator()

    cfg = _select_phase1_backward_launch_config(D=D)
    BLOCK_D = cfg.block_d
    if dlog_alpha_out is None:
        dlog_alpha_buffer = torch.zeros((BH, NC, C), device=V_un.device, dtype=log_alpha_un.dtype)
    else:
        if dlog_alpha_out.shape != (BH, NC, C):
            raise ValueError(f"dlog_alpha_out must be [BH,NC,C]={BH, NC, C}; got {tuple(dlog_alpha_out.shape)}.")
        if dlog_alpha_out.device != V_un.device:
            raise ValueError("dlog_alpha_out must be on the same device as inputs.")
        if dlog_alpha_out.dtype not in (log_alpha_un.dtype, torch.float32):
            raise ValueError("dlog_alpha_out must match log_alpha dtype or be float32.")
        dlog_alpha_buffer = dlog_alpha_out
    grid = (BH * NC,)
    ssd_single_rank1_phase1_bwd_un_accum_kernel[grid](
        grad_s,
        V_un,
        log_alpha_un,
        dV_out,
        dlog_alpha_buffer,
        B,
        H,
        N,
        NC,
        C,
        D,
        *grad_s.stride(),
        *V_un.stride(),
        *log_alpha_un.stride(),
        *dV_out.stride(),
        *dlog_alpha_buffer.stride(),
        BLOCK_D=BLOCK_D,
        BLOCK_C=C,
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        D_STATIC=D,
    )
    return dlog_alpha_buffer


# --------------------------------------------------------------------------------------------------
# PHASE 2 BACKWARD WRAPPER (REUSED FROM MATRIX-STATE FILE)
# --------------------------------------------------------------------------------------------------
def _phase2_backward_impl(
    grad_chunk_start: torch.Tensor,
    grad_final_state: torch.Tensor,
    S0_chunk: torch.Tensor,
    log_alpha_per_chunk: torch.Tensor,
    *,
    B: int,
    H: int,
    compute_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    BH, NC, D = grad_chunk_start.shape
    if B * H != BH:
        raise ValueError(f"_phase2_backward_impl expected B*H == BH; got B={B}, H={H}, BH={BH}.")
    grad_chunk_start_f = (
        grad_chunk_start if grad_chunk_start.dtype == torch.float32 and grad_chunk_start.is_contiguous()
        else grad_chunk_start.float().contiguous()
    )
    grad_final_state_f = (
        grad_final_state if grad_final_state.dtype == torch.float32 and grad_final_state.is_contiguous()
        else grad_final_state.float().contiguous()
    )
    S0_chunk_f = S0_chunk if S0_chunk.dtype == torch.float32 and S0_chunk.is_contiguous() else S0_chunk.float().contiguous()
    log_alpha_per_chunk_bnh = _reshape_log_alpha_per_chunk_to_bnh(
        log_alpha_per_chunk, B=B, H=H, NC=NC, where="_phase2_backward_impl"
    )
    log_alpha_per_chunk_f = (
        log_alpha_per_chunk_bnh
        if log_alpha_per_chunk_bnh.dtype == torch.float32 and log_alpha_per_chunk_bnh.is_contiguous()
        else log_alpha_per_chunk_bnh.float().contiguous()
    )

    dS_local_end = _workspace_tensor(
        "single_phase2_dS_local_end",
        (BH, NC, D),
        device=grad_chunk_start.device,
        dtype=torch.float32,
    )
    d_log_alpha_per_chunk = _workspace_tensor(
        "single_phase2_dlog_per_chunk",
        (BH, NC),
        device=grad_chunk_start.device,
        dtype=torch.float32,
    )
    d_init = _workspace_tensor(
        "single_phase2_dinit",
        (BH, D),
        device=grad_chunk_start.device,
        dtype=torch.float32,
    )

    block_nc = _select_phase2_block_nc(NC=NC)
    block_md, phase2_num_warps, phase2_num_stages = _select_phase2_md_launch(MD=D)
    grid = lambda META: (BH,)
    ssd_single_rank1_phase2_prefix_scan_bwd_dense_kernel[grid](
        grad_chunk_start_f,
        grad_final_state_f,
        S0_chunk_f,
        log_alpha_per_chunk_f,
        dS_local_end,
        d_log_alpha_per_chunk,
        d_init,
        B,
        H,
        BH,
        NC,
        D,
        *grad_chunk_start_f.stride(),
        *grad_final_state_f.stride(),
        *S0_chunk_f.stride(),
        *log_alpha_per_chunk_f.stride(),
        *dS_local_end.stride(),
        *d_log_alpha_per_chunk.stride(),
        *d_init.stride(),
        BLOCK_NC=block_nc,
        BLOCK_MD=block_md,
        NC_STATIC=NC,
        USE_FP32_COMPUTE=(compute_dtype == torch.float32),
        USE_BF16_COMPUTE=(compute_dtype == torch.bfloat16),
        HAS_GRAD_FINAL=True,
        WRITE_D_INIT=True,
        num_warps=phase2_num_warps,
        num_stages=phase2_num_stages,
    )
    return dS_local_end, d_log_alpha_per_chunk, d_init


# --------------------------------------------------------------------------------------------------
# UNIFIED AUTOGRAD ENTRYPOINT
# --------------------------------------------------------------------------------------------------
class SsdSingleRank1Triton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        V: torch.Tensor,
        log_alpha: torch.Tensor,
        initial_state: torch.Tensor | None,
        CHUNK_SIZE: int,
        INPUT_PRECISION: str,
    ):
        init_flat, B, N, H, D, NC_exec = _prepare_unchunked_inputs(
            V, log_alpha, initial_state, where="SsdSingleRank1Triton.forward", CHUNK_SIZE=CHUNK_SIZE
        )

        S_local_end, log_alpha_per_chunk = _phase1_forward_replay_from_unchunked(
            V,
            log_alpha,
            H=H,
            NC=NC_exec,
            CHUNK_SIZE=CHUNK_SIZE,
        )

        # Keep phase-2 in fp32 for final-state stability; output tensors are small.
        S0_chunk, S1_chunk = _ssd_single_rank1_phase2_prefix_scan_forward_impl(
            S_local_end, log_alpha_per_chunk, init_flat, torch.float32
        )

        y_chunk = _phase3_forward_from_unchunked(
            V,
            log_alpha,
            S0_chunk,
            H=H,
            NC=NC_exec,
            CHUNK_SIZE=CHUNK_SIZE,
            input_precision=INPUT_PRECISION,
            RETURN_Y_OFF=False,
        )

        # Save unchunked inputs; replay chunking in backward to reduce forward
        # activation residency from chunked copies.
        ctx.save_for_backward(V, log_alpha, init_flat)
        ctx.CHUNK_SIZE = CHUNK_SIZE
        ctx.INPUT_PRECISION = INPUT_PRECISION
        ctx.B = B
        ctx.N = N
        ctx.H = H
        ctx.D = D
        ctx.NC_exec = NC_exec
        ctx.compute_dtype = V.dtype
        ctx.phase2_compute_dtype = torch.float32
        ctx.initial_state_shape = None if initial_state is None else tuple(initial_state.shape)
        return y_chunk, S1_chunk

    @staticmethod
    def backward(ctx, grad_y_chunk, grad_S1_chunk):
        V_in, log_alpha_in, init_flat = ctx.saved_tensors
        CHUNK_SIZE = ctx.CHUNK_SIZE
        INPUT_PRECISION = ctx.INPUT_PRECISION
        B = ctx.B
        N = ctx.N
        H = ctx.H
        D = ctx.D
        NC = ctx.NC_exec
        compute_dtype = ctx.compute_dtype
        phase2_compute_dtype = ctx.phase2_compute_dtype
        initial_state_shape = ctx.initial_state_shape
        BH = B * H

        # Replay phase-1 / phase-2 directly from unchunked inputs to avoid
        # materializing chunked V/log copies in backward.
        S_local_end, log_alpha_per_chunk = _phase1_forward_replay_from_unchunked(
            V_in,
            log_alpha_in,
            H=H,
            NC=NC,
            CHUNK_SIZE=CHUNK_SIZE,
        )
        S0_chunk, _ = _ssd_single_rank1_phase2_prefix_scan_forward_impl(
            S_local_end, log_alpha_per_chunk, init_flat, phase2_compute_dtype
        )

        # Use one shared dlog_alpha buffer across phase-3 and phase-1.
        dlog_alpha = torch.zeros((BH, NC, CHUNK_SIZE), device=V_in.device, dtype=torch.float32)

        # Phase-3 backward.
        dV, dlog_alpha, dS0 = _phase3_backward_from_unchunked(
            V_in,
            log_alpha_in,
            grad_y_chunk,
            S0_chunk,
            H=H,
            NC=NC,
            CHUNK_SIZE=CHUNK_SIZE,
            input_precision=INPUT_PRECISION,
            return_s0_grad_fp32=True,
            dlog_alpha_out=dlog_alpha,
        )

        # Phase-2 backward.
        grad_final = (
            grad_S1_chunk
            if grad_S1_chunk is not None
            else torch.zeros((BH, D), device=V_in.device, dtype=phase2_compute_dtype)
        )
        dS_local_end, dlog_alpha_per_chunk, d_init_f32 = _phase2_backward_impl(
            dS0.reshape(BH, NC, D),
            grad_final,
            S0_chunk,
            log_alpha_per_chunk,
            B=B,
            H=H,
            compute_dtype=phase2_compute_dtype,
        )

        # Phase-1 backward.
        _phase1_backward_from_unchunked_accum(
            dS_local_end,
            V_in,
            log_alpha_in,
            dV,
            H=H,
            NC=NC,
            CHUNK_SIZE=CHUNK_SIZE,
            dlog_alpha_out=dlog_alpha,
        )

        dlog_alpha.add_(dlog_alpha_per_chunk.unsqueeze(-1))
        dlog_alpha_chunk = dlog_alpha

        n_exec = NC * CHUNK_SIZE
        dlog_alpha_view = torch.as_strided(
            dlog_alpha_chunk,
            size=(B, n_exec, H),
            stride=(H * n_exec, 1, n_exec),
        )
        dlog_alpha = dlog_alpha_view[:, :N, :].to(log_alpha_in.dtype)

        if initial_state_shape is None:
            d_init = None
        elif len(initial_state_shape) == 2:
            d_init = d_init_f32.reshape(BH, D).to(compute_dtype)
        else:
            d_init = d_init_f32.reshape(B, H, D).to(compute_dtype)

        return dV, dlog_alpha, d_init, None, None


def ssd_single_rank1_triton(
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
    INPUT_PRECISION: str = "tf32",
) -> tuple[torch.Tensor, torch.Tensor]:
    if CHUNK_SIZE is None:
        B0, N0, H0, D0 = V.shape
        CHUNK_SIZE = _select_chunk_size_heuristic(N=N0, D=D0, BH=B0 * H0)
    y_chunk, S1_chunk = SsdSingleRank1Triton.apply(
        V, log_alpha, initial_state, CHUNK_SIZE, INPUT_PRECISION
    )
    B, N, H, _ = V.shape
    return _restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)


__all__ = ["ssd_single_rank1_pytorch", "ssd_single_rank1_triton"]

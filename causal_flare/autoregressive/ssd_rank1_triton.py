"""Triton path for SSD rank-1 autoregressive chunk-prefix state scans."""

from __future__ import annotations

from dataclasses import dataclass

import torch

import triton
import triton.language as tl

_SUPPORTED_M_VALUES = {16, 32, 64, 128}
_SUPPORTED_D_VALUES = {32, 64, 128}
_SUPPORTED_CHUNK_SIZES = {32, 64, 128, 256}
_SUPPORTED_BLOCK_T_VALUES = (16, 8, 4, 2, 1)
_SUPPORTED_BLOCK_X_VALUES = (128, 64, 32, 16)
_SUPPORTED_PHASE1_BLOCK_T_VALUES = {1, 2, 4, 8, 16}
_EXPERIMENTAL_TRITON_ALLOCATOR_SET = False
_INV_LN2 = 1.4426950408889634

# Reusable buffers for Phase-3 backward intermediates to reduce allocation overhead.
_PHASE3_BWD_WORKSPACE: dict[tuple, torch.Tensor] = {}


@dataclass(frozen=True)
class _Phase3ForwardLaunchConfig:
    block_m: int
    block_d: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class _Phase3BackwardLaunchConfig:
    block_m: int
    block_d: int
    fused_a1a2_num_warps: int
    fused_off_num_warps: int
    num_stages: int


@dataclass(frozen=True)
class _Phase1BackwardLaunchConfig:
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


def _ssd_rank1_bwd_workspace_tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Get or create a reusable Phase-3 backward workspace tensor."""
    dev_key = (device.type, device.index if device.type == "cuda" else -1)
    key = (name, dev_key, dtype, shape)
    t = _PHASE3_BWD_WORKSPACE.get(key)
    if t is None:
        t = torch.empty(shape, device=device, dtype=dtype)
        _PHASE3_BWD_WORKSPACE[key] = t
    return t


def _require_supported_md(m_size: int, d_size: int, *, where: str) -> None:
    if m_size not in _SUPPORTED_M_VALUES:
        raise NotImplementedError(f"{where} requires M in {sorted(_SUPPORTED_M_VALUES)}; got M={m_size}.")
    if d_size not in _SUPPORTED_D_VALUES:
        raise NotImplementedError(f"{where} requires D in {sorted(_SUPPORTED_D_VALUES)}; got D={d_size}.")


def _require_chunk_size_multiple_of_16(c_size: int, *, where: str) -> None:
    """Enforce the current Triton chunk-size contract."""
    if c_size == 0:
        raise NotImplementedError(f"{where} does not support C==0.")
    if c_size % 16 != 0:
        raise NotImplementedError(f"{where} requires C to be a positive multiple of 16; got C={c_size}.")


def _require_nc_descriptor_width(nc_size: int, *, where: str) -> None:
    if nc_size < 8:
        raise NotImplementedError(f"{where} requires NC >= 8 for descriptor vector width; got NC={nc_size}.")


def _require_nc_multiple_of_16(nc_size: int, *, where: str) -> None:
    if nc_size % 16 != 0:
        raise NotImplementedError(f"{where} requires NC to be a multiple of 16; got NC={nc_size}.")


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


def _select_largest_block_size(size: int, candidates: tuple[int, ...], *, where: str, label: str) -> int:
    for block in candidates:
        if block <= size and size % block == 0:
            return block
    raise NotImplementedError(
        f"{where} requires {label} to be divisible by one of {list(candidates)}; got {label}={size}."
    )


def _select_chunk_size_heuristic(*, N: int, M: int, D: int, BH: int) -> int:
    """Heuristic CHUNK_SIZE picker for the supported contract."""
    if max(M, D) >= 128:
        return 32
    if max(M, D) <= 32 and N >= 4096 and BH <= 2048:
        return 128
    return 64


def _select_phase2_block_nc(*, NC: int) -> int:
    """Heuristic BLOCK_NC for phase-2 kernels."""
    if NC >= 32 and NC % 32 == 0:
        return 32
    return 16


def _select_phase3_forward_launch_config(
    *,
    BH: int,
    NC: int,
    C_CHUNK: int,
    M: int,
    D: int,
    where: str,
) -> _Phase3ForwardLaunchConfig:
    block_m = _select_largest_block_size(M, _SUPPORTED_BLOCK_X_VALUES, where=where, label="M")
    block_d = _select_largest_block_size(D, _SUPPORTED_BLOCK_X_VALUES, where=where, label="D")

    work_items = BH * NC
    # On high-parallel BF16 workloads with C=64, lower warps with deeper pipelining
    # consistently improved phase-3 forward in benchmarking.
    if C_CHUNK == 64 and block_m == 64 and block_d == 64 and work_items >= 4096:
        return _Phase3ForwardLaunchConfig(block_m=64, block_d=64, num_warps=2, num_stages=3)
    if block_m >= 64 and block_d >= 64:
        return _Phase3ForwardLaunchConfig(block_m=block_m, block_d=block_d, num_warps=4, num_stages=2)
    return _Phase3ForwardLaunchConfig(block_m=block_m, block_d=block_d, num_warps=2, num_stages=2)


def _select_phase3_backward_launch_config(
    *,
    BH: int,
    NC: int,
    C_CHUNK: int,
    M: int,
    D: int,
) -> _Phase3BackwardLaunchConfig:
    block_m = 64 if M % 64 == 0 else _select_largest_block_size(M, (32, 16), where="phase3_bwd", label="M")
    block_d = 64 if D % 64 == 0 else _select_largest_block_size(D, (32, 16), where="phase3_bwd", label="D")

    # Best-known setting for the common BF16 trainer shape (M=D=C=64).
    if C_CHUNK == 64 and block_m == 64 and block_d == 64 and BH * NC >= 4096:
        return _Phase3BackwardLaunchConfig(
            block_m=64,
            block_d=64,
            fused_a1a2_num_warps=4,
            fused_off_num_warps=4,
            num_stages=2,
        )
    if block_m >= 32 and block_d >= 32:
        return _Phase3BackwardLaunchConfig(
            block_m=block_m,
            block_d=block_d,
            fused_a1a2_num_warps=4,
            fused_off_num_warps=4,
            num_stages=3,
        )
    return _Phase3BackwardLaunchConfig(
        block_m=block_m,
        block_d=block_d,
        fused_a1a2_num_warps=2,
        fused_off_num_warps=2,
        num_stages=3,
    )


def _select_phase1_backward_launch_config(*, M: int, D: int) -> _Phase1BackwardLaunchConfig:
    if M >= 128 or D >= 128:
        return _Phase1BackwardLaunchConfig(num_warps=4, num_stages=2)
    return _Phase1BackwardLaunchConfig(num_warps=2, num_stages=2)


def _require_phase1_block_t(block_t: int, c_size: int, *, where: str) -> None:
    if block_t not in _SUPPORTED_PHASE1_BLOCK_T_VALUES:
        raise NotImplementedError(
            f"{where} requires BLOCK_T in {sorted(_SUPPORTED_PHASE1_BLOCK_T_VALUES)}; got BLOCK_T={block_t}."
        )
    if block_t < 16:
        raise NotImplementedError(f"{where} requires BLOCK_T >= 16 for tensor-core Phase-1 path; got BLOCK_T={block_t}.")
    if block_t > c_size:
        raise NotImplementedError(f"{where} requires BLOCK_T <= C; got BLOCK_T={block_t}, C={c_size}.")
    if c_size % block_t != 0:
        raise NotImplementedError(f"{where} requires C divisible by BLOCK_T; got C={c_size}, BLOCK_T={block_t}.")


def _ssd_rank1_prepare_unchunked_inputs(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    where: str,
    CHUNK_SIZE: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, int, int]:
    """Validate and chunk unchunked mode-0 tensors into `[BH, NC, C, *]` layouts."""
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            f"{where} expects C=[B,N,H,M], W=[B,N,H,M], V=[B,N,H,D], log_alpha=[B,N,H]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    B, N, H, M = C.shape
    if W.shape != (B, N, H, M):
        raise ValueError(f"{where}: W must be [B,N,H,M]={B, N, H, M}; got {tuple(W.shape)}.")
    if V.shape[:3] != (B, N, H):
        raise ValueError(f"{where}: V must match [B,N,H,*]={B, N, H}; got {tuple(V.shape)}.")
    D = V.shape[-1]
    if log_alpha.shape != (B, N, H):
        raise ValueError(f"{where}: log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if C.device != W.device or C.device != V.device or C.device != log_alpha.device:
        raise ValueError(f"{where}: C, W, V, log_alpha must be on the same device.")
    if C.dtype != W.dtype or C.dtype != V.dtype or C.dtype != log_alpha.dtype:
        raise ValueError(f"{where}: C, W, V, log_alpha must share dtype.")
    if N == 0:
        raise NotImplementedError(f"{where} does not support N==0.")
    _require_supported_md(M, D, where=where)
    _require_supported_forward_contract(B * H, N, CHUNK_SIZE=CHUNK_SIZE, where=where)
    _require_nonpositive_log_alpha(log_alpha, where=where)

    NC_data = triton.cdiv(N, CHUNK_SIZE)
    padded_tokens = NC_data * CHUNK_SIZE
    token_pad = padded_tokens - N
    if token_pad > 0:
        z_cm = torch.zeros((B, token_pad, H, M), device=C.device, dtype=C.dtype)
        z_vd = torch.zeros((B, token_pad, H, D), device=V.device, dtype=V.dtype)
        o_r = torch.zeros((B, token_pad, H), device=log_alpha.device, dtype=log_alpha.dtype)
        C = torch.cat([C, z_cm], dim=1)
        W = torch.cat([W, z_cm], dim=1)
        V = torch.cat([V, z_vd], dim=1)
        log_alpha = torch.cat([log_alpha, o_r], dim=1)

    C_chunk = (
        C.reshape(B, NC_data, CHUNK_SIZE, H, M)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(B * H, NC_data, CHUNK_SIZE, M)
    )
    W_chunk = (
        W.reshape(B, NC_data, CHUNK_SIZE, H, M)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .reshape(B * H, NC_data, CHUNK_SIZE, M)
    )
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
        z_c = torch.zeros((B * H, pad_chunks, CHUNK_SIZE, M), device=C.device, dtype=C.dtype)
        z_v = torch.zeros((B * H, pad_chunks, CHUNK_SIZE, D), device=V.device, dtype=V.dtype)
        o_r = torch.zeros((B * H, pad_chunks, CHUNK_SIZE), device=log_alpha.device, dtype=log_alpha.dtype)
        C_chunk = torch.cat([C_chunk, z_c], dim=1)
        W_chunk = torch.cat([W_chunk, z_c], dim=1)
        V_chunk = torch.cat([V_chunk, z_v], dim=1)
        log_alpha_chunk = torch.cat([log_alpha_chunk, o_r], dim=1)

    BH = B * H
    MD = M * D
    if initial_state is None:
        init_flat = torch.zeros((BH, MD), device=C.device, dtype=C.dtype)
    elif initial_state.ndim == 2 and initial_state.shape == (BH, MD):
        init_flat = initial_state
    elif initial_state.ndim == 3 and initial_state.shape == (B, H, MD):
        init_flat = initial_state.reshape(BH, MD)
    elif initial_state.ndim == 4 and initial_state.shape == (B, H, M, D):
        init_flat = initial_state.reshape(BH, MD)
    else:
        raise ValueError(
            f"{where}: initial_state must be [BH,MD], [B,H,MD], or [B,H,M,D]. "
            f"Got {tuple(initial_state.shape)} with expected BH={BH}, MD={MD}."
        )
    if init_flat.device != C.device:
        raise ValueError(f"{where}: initial_state must be on the same device as C/W/V/log_alpha.")
    if init_flat.dtype != C.dtype:
        raise ValueError(f"{where}: initial_state must share dtype with C/W/V/log_alpha.")

    return C_chunk, W_chunk, V_chunk, log_alpha_chunk, init_flat, B, N, H, M, D, NC_exec


def _ssd_rank1_restore_output_layout(
    y_chunk: torch.Tensor,
    final_state_flat: torch.Tensor,
    *,
    B: int,
    N: int,
    H: int,
    C: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map `[BH, NC, C, D]`/`[BH, MD]` back to `[B,N,H,D]`/`[B,H,MD]`."""
    D = y_chunk.shape[-1]
    y = (
        y_chunk.reshape(B, H, y_chunk.shape[1], C, D)
        .permute(0, 2, 3, 1, 4)
        .reshape(B, y_chunk.shape[1] * C, H, D)
    )[:, :N]
    final_state = final_state_flat.reshape(B, H, -1)
    return y, final_state


# ==================================================================================================
# ORACLE / TORCH ENTRYPOINT
# ==================================================================================================
def ssd_rank1_pytorch(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SSD rank-1 PyTorch oracle entrypoint."""
    if CHUNK_SIZE is None:
        B, N, H, M = C.shape
        D = V.shape[-1]
        CHUNK_SIZE = _select_chunk_size_heuristic(N=N, M=M, D=D, BH=B * H)

    C_chunk, W_chunk, V_chunk, log_alpha_chunk, init_flat, B, N, H, M, D, _ = _ssd_rank1_prepare_unchunked_inputs(
        C,
        W,
        V,
        log_alpha,
        initial_state,
        where="ssd_rank1_pytorch",
        CHUNK_SIZE=CHUNK_SIZE,
    )

    # =========================================
    # PHASE 1 local chunk end-state
    # =========================================
    S_local_end = ssd_rank1_chunk_end_state_reference(W_chunk, V_chunk, log_alpha_chunk)
    alpha_chunk = torch.exp2(torch.sum(log_alpha_chunk, dim=-1) * _INV_LN2)

    # =========================================
    # PHASE 2 prefix scan over chunks
    # =========================================
    S0_chunk, S1_chunk = ssd_rank1_prefix_scan_reference(S_local_end, alpha_chunk, init_flat)

    # =========================================
    # PHASE 3 dense chunk-local output
    # =========================================
    y_chunk = ssd_rank1_dense_output_reference(C_chunk, W_chunk, V_chunk, log_alpha_chunk, S0_chunk)

    return _ssd_rank1_restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)


# --------------------------------------------------------------------------------------------------
# ORACLE: TOKEN-LOOP REFERENCE
# --------------------------------------------------------------------------------------------------
def ssd_rank1_token_loop_oracle(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unchunked token-loop oracle for mode-0 recurrence.

    Recurrence:
      alpha_t = exp(log_alpha_t),  with log_alpha_t <= 0
      S_t = alpha_t * S_{t-1} + W_t ⊗ V_t
      y_t = C_t @ S_t

    Inputs:
    - C: `[B, N, H, M]`
    - W: `[B, N, H, M]`
    - V: `[B, N, H, D]`
    - log_alpha: `[B, N, H]`
    - initial_state: optional `[BH,MD]`, `[B,H,MD]`, or `[B,H,M,D]`

    Returns:
    - y: `[B, N, H, D]`
    - final_state: `[B, H, MD]`
    """
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "ssd_rank1_token_loop_oracle expects C=[B,N,H,M], W=[B,N,H,M], V=[B,N,H,D], log_alpha=[B,N,H]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    B, N, H, M = C.shape
    if W.shape != (B, N, H, M):
        raise ValueError(f"W must be [B,N,H,M]={B, N, H, M}; got {tuple(W.shape)}.")
    if V.shape[:3] != (B, N, H):
        raise ValueError(f"V must match [B,N,H,*]={B, N, H}; got {tuple(V.shape)}.")
    D = V.shape[-1]
    if log_alpha.shape != (B, N, H):
        raise ValueError(f"log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if C.device != W.device or C.device != V.device or C.device != log_alpha.device:
        raise ValueError("C, W, V, log_alpha must be on the same device.")
    if C.dtype != W.dtype or C.dtype != V.dtype or C.dtype != log_alpha.dtype:
        raise ValueError("C, W, V, log_alpha must share dtype.")
    if N == 0:
        raise NotImplementedError("ssd_rank1_token_loop_oracle does not support N==0.")
    _require_supported_md(M, D, where="ssd_rank1_token_loop_oracle")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank1_token_loop_oracle")

    BH = B * H
    MD = M * D
    if initial_state is None:
        S = torch.zeros((B, H, M, D), device=C.device, dtype=C.dtype)
    elif initial_state.ndim == 2 and initial_state.shape == (BH, MD):
        S = initial_state.reshape(B, H, M, D)
    elif initial_state.ndim == 3 and initial_state.shape == (B, H, MD):
        S = initial_state.reshape(B, H, M, D)
    elif initial_state.ndim == 4 and initial_state.shape == (B, H, M, D):
        S = initial_state
    else:
        raise ValueError(
            f"initial_state must be [BH,MD], [B,H,MD], or [B,H,M,D]. "
            f"Got {tuple(initial_state.shape)} with BH={BH}, MD={MD}."
        )
    if S.device != C.device:
        raise ValueError("initial_state must be on the same device as C/W/V/log_alpha.")
    if S.dtype != C.dtype:
        raise ValueError("initial_state must share dtype with C/W/V/log_alpha.")

    alpha = torch.exp2(log_alpha * _INV_LN2)
    ys = []
    for t in range(N):
        S = (
            alpha[:, t].unsqueeze(-1).unsqueeze(-1) * S
            + W[:, t].unsqueeze(-1) * V[:, t].unsqueeze(-2)
        )
        ys.append(torch.matmul(C[:, t].unsqueeze(-2), S).squeeze(-2))

    y = torch.stack(ys, dim=1)
    final_state = S.reshape(B, H, MD)
    return y, final_state


# --------------------------------------------------------------------------------------------------
# END ORACLE: TOKEN-LOOP REFERENCE
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# END ORACLE / TORCH ENTRYPOINT
# --------------------------------------------------------------------------------------------------

# ==================================================================================================
# REFERENCE FUNCTIONS
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# PHASE 1: LOCAL CHUNK END-STATE COMPUTATION
# --------------------------------------------------------------------------------------------------
def ssd_rank1_chunk_end_state_reference(
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
) -> torch.Tensor:
    """Phase-1 reference: per-chunk local end-state from factorized writes.

    Inputs:
    - W: `[BH, NC, C, M]`
    - V: `[BH, NC, C, D]`
    - log_alpha: `[BH, NC, C]`
      Scalar log-retain factor per token, constrained to `<= 0`.

    Output:
    - S_local_end: `[BH, NC, MD]`
      End S for each chunk assuming zero chunk start:
        S_{t+1} = alpha_t * S_t + W_t ⊗ V_t, with S_0 = 0
        where `alpha_t = exp(log_alpha_t)`.
      and `S_local_end = S_C`.

    Closed form:
      S_local_end = sum_{j=0..C-1} (prod_{u=j+1..C-1} alpha_u) * (W_j ⊗ V_j)

    Vectorized implementation:
      Let
        r_j = prod_{u=j+1..C-1} alpha_u
      then for each `(BH, NC)` chunk lane:
        S_local_end[M,D] = ( (r[T] * W^T[M,T]) @ V[T,D] )
    """
    if W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "ssd_rank1_chunk_end_state_reference expects W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
            f"Got W={tuple(W.shape)}, V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    BH, NC, C, M = W.shape
    if V.shape[:3] != (BH, NC, C):
        raise ValueError(
            "V must match [BH,NC,C,*] from W. "
            f"Got V={tuple(V.shape)}, W={tuple(W.shape)}."
        )
    D = V.shape[-1]
    if log_alpha.shape != (BH, NC, C):
        raise ValueError(
            "log_alpha must match [BH,NC,C] from W/V. "
            f"Got log_alpha={tuple(log_alpha.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}."
        )
    if W.device != V.device or W.device != log_alpha.device:
        raise ValueError("W, V, log_alpha must be on the same device.")
    if W.dtype != V.dtype or W.dtype != log_alpha.dtype:
        raise ValueError("W, V, log_alpha must share dtype.")
    if C == 0:
        raise NotImplementedError("ssd_rank1_chunk_end_state_reference does not support C==0.")
    _require_supported_md(M, D, where="ssd_rank1_chunk_end_state_reference")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank1_chunk_end_state_reference")
    log_alpha_f = log_alpha.float()
    W_f = W.float()
    V_f = V.float()
    # Build exclusive suffix retain factors:
    #   r[j] = prod_{u=j+1..C-1} alpha[u] = exp(sum_{u=j+1..C-1} log_alpha[u]).
    # We compute this via reverse cumsum in log-space for better numerical behavior.
    log_alpha_rev = torch.flip(log_alpha_f, dims=[-1])
    log_suffix_incl_rev = torch.cumsum(log_alpha_rev, dim=-1)
    log_suffix_excl_rev = log_suffix_incl_rev - log_alpha_rev
    r = torch.flip(torch.exp2(log_suffix_excl_rev * _INV_LN2), dims=[-1])

    # S[M,D] = (r[T] * W^T[M,T]) @ V[T,D], batched across [BH,NC].
    weighted_w_t = W_f.transpose(-1, -2) * r.unsqueeze(-2)
    S_local_end = torch.matmul(weighted_w_t, V_f).reshape(BH, NC, M * D)
    return S_local_end


# --------------------------------------------------------------------------------------------------
# END PHASE 1: LOCAL CHUNK END-STATE COMPUTATION
# --------------------------------------------------------------------------------------------------


def ssd_rank1_prefix_scan_reference(
    S_local_end: torch.Tensor,
    alpha_chunk: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference math for Phase-2 chunk-prefix state propagation.

    Inputs:
    - S_local_end: `[BH, NC, MD]` (or `[BHG, NC, MD]` if grouping is folded outside)
    - alpha_chunk: `[BH, NC]` (or `[BHG, NC]`)
    - initial_state: `[BH, MD]` (or `[BHG, MD]`)

    Chunk-local recurrence:

      S_end[c] = alpha_chunk[c] * S_start[c] + S_local_end[c]
      S_start[c+1] = S_end[c]
      S_start[0] = initial_state

    Closed-form inclusive prefix coefficients:

      inc_r[c] = prod_{k=0..c} alpha_chunk[k]
      inc_s[c] = sum_{j=0..c} ( prod_{k=j+1..c} alpha_chunk[k] ) * S_local_end[j]

    so:

      S_end[c] = inc_r[c] * initial_state + inc_s[c]

    Exclusive chunk starts:

      excl_r[0] = 1, excl_s[0] = 0
      excl_r[c] = inc_r[c-1], excl_s[c] = inc_s[c-1] for c > 0

      chunk_start[c] = excl_r[c] * initial_state + excl_s[c]
      final_state    = inc_r[NC-1] * initial_state + inc_s[NC-1]
    """
    if S_local_end.ndim != 3 or alpha_chunk.ndim != 2 or initial_state.ndim != 2:
        raise ValueError(
            "ssd_rank1_prefix_scan_reference expects "
            "S_local_end=[BH,NC,MD], alpha_chunk=[BH,NC], initial_state=[BH,MD]. "
            f"Got S_local_end={tuple(S_local_end.shape)}, alpha_chunk={tuple(alpha_chunk.shape)}, "
            f"initial_state={tuple(initial_state.shape)}."
        )
    BH, NC, MD = S_local_end.shape
    if alpha_chunk.shape != (BH, NC):
        raise ValueError(
            "alpha_chunk must match [BH,NC] from S_local_end. "
            f"Got alpha_chunk={tuple(alpha_chunk.shape)}, S_local_end={tuple(S_local_end.shape)}."
        )
    if initial_state.shape != (BH, MD):
        raise ValueError(
            "initial_state must match [BH,MD] from S_local_end. "
            f"Got initial_state={tuple(initial_state.shape)}, S_local_end={tuple(S_local_end.shape)}."
        )
    if NC == 0:
        raise NotImplementedError("ssd_rank1_prefix_scan_reference does not support NC==0.")

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


@triton.jit
def ssd_rank1_chunk_end_state_fwd_kernel(
    W_ptr,
    V_ptr,
    log_alpha_ptr,
    out_s_local_end_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    d_size,
    stride_w_bh,
    stride_w_nc,
    stride_w_c,
    stride_w_m,
    stride_v_bh,
    stride_v_nc,
    stride_v_c,
    stride_v_d,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_out_s_local_end_bh,
    stride_out_s_local_end_nc,
    stride_out_s_local_end_m,
    stride_out_s_local_end_d,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Phase-1 Triton forward for factorized writes.

    State definition for one `(BH, NC)`:

      S_{t+1} = alpha_t * S_t + W_t \otimes V_t,    S_0 = 0
      alpha_t = exp(log_alpha_t), with log_alpha_t <= 0.

    where `W_t` is length-`M`, `V_t` is length-`D`, and `S_t` is `[M, D]`.
    The output we need is:

      S_local_end = S_C

    Closed-form weights for token contributions:

      factor[t] = prod_{u=t+1..C-1} alpha_u
      S_C       = sum_{t=0..C-1} factor[t] * (W_t \otimes V_t)

    Kernel strategy:
    1. Load all `log_alpha_t` for this `(BH, NC)`, form `alpha_t = exp(log_alpha_t)`,
       then compute suffix products for `factor[t]`.
    2. For each token block of size `BLOCK_T`, accumulate:
         S += sum_t (factor_t * W_t[:, None] * V_t[None, :])
       using a GEMM-like reduction to hit tensor cores:
         A = sqrt(factor)[:, None] * W_block    # [BLOCK_T, BLOCK_M]
         B = sqrt(factor)[:, None] * V_block    # [BLOCK_T, BLOCK_D]
         S += A^T @ B                           # [BLOCK_M, BLOCK_D]
    3. Store `state_tile` into `S_local_end[BH, NC, M, D]`.

    Notes:
    - `BLOCK_C` is padded to next power-of-two so token-lane vector ops are legal.
    - Masked token loads use `other=1.0` so padded lanes are multiplicative identity.
    - Internal accumulation is FP32 for stability across fp16/bf16/fp32 inputs.
    """
    pid_bhnc = tl.program_id(0)
    pid_m_tile = tl.program_id(1)
    pid_d_tile = tl.program_id(2)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size

    m_start = pid_m_tile * BLOCK_M
    d_start = pid_d_tile * BLOCK_D
    offs_m = pid_m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = pid_d_tile * BLOCK_D + tl.arange(0, BLOCK_D)

    w_desc = tl.make_tensor_descriptor(
        W_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_w_bh, stride_w_nc, stride_w_c, stride_w_m],
        block_shape=[1, 1, BLOCK_T, BLOCK_M],
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_v_bh, stride_v_nc, stride_v_c, stride_v_d],
        block_shape=[1, 1, BLOCK_T, BLOCK_D],
    )
    r_desc = tl.make_tensor_descriptor(
        log_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_r_bh, stride_r_nc, stride_r_c],
        block_shape=[1, 1, BLOCK_C],
    )
    out_s_desc = tl.make_tensor_descriptor(
        out_s_local_end_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_out_s_local_end_bh, stride_out_s_local_end_nc, stride_out_s_local_end_m, stride_out_s_local_end_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    INV_LN2 = 1.4426950408889634
    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    r_factors = tl.exp2((log_suffix_incl - log_alpha_vals) * INV_LN2)

    S = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    offs_t_full = tl.arange(0, BLOCK_C)
    for t_start in tl.static_range(0, C_STATIC, BLOCK_T):
        offs_tb = t_start + tl.arange(0, BLOCK_T)
        factors = tl.sum(
            tl.where(offs_t_full[None, :] == offs_tb[:, None], r_factors[None, :], 0.0),
            axis=1,
        )
        W_blk = tl.reshape(w_desc.load([BH_IDX, NC_IDX, t_start, m_start]), (BLOCK_T, BLOCK_M))
        V_blk = tl.reshape(v_desc.load([BH_IDX, NC_IDX, t_start, d_start]), (BLOCK_T, BLOCK_D))
        sqrt_factors = tl.sqrt(factors)[:, None]
        # A^T @ B reproduces sum_t factor[t] * outer(W_t, V_t).
        A = (sqrt_factors * W_blk.to(tl.float32)).to(W_blk.dtype)
        B = (sqrt_factors * V_blk.to(tl.float32)).to(V_blk.dtype)
        S += tl.dot(tl.trans(A), B, out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    out_s_desc.store([BH_IDX, NC_IDX, m_start, d_start], tl.reshape(S, (1, 1, BLOCK_M, BLOCK_D)))

    # alpha_chunk belongs to phase-1 scope but is computed outside this kernel.


@triton.jit
def ssd_rank1_chunk_end_state_bwd_dw_chunk_kernel(
    grad_s_local_end_ptr,
    V_ptr,
    log_alpha_ptr,
    out_dw_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    d_size,
    stride_grad_s_bh,
    stride_grad_s_nc,
    stride_grad_s_m,
    stride_grad_s_d,
    stride_v_bh,
    stride_v_nc,
    stride_v_c,
    stride_v_d,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_out_dw_bh,
    stride_out_dw_nc,
    stride_out_dw_c,
    stride_out_dw_m,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    D_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """Phase-1 backward kernel for `dW` with chunk-owned programs.

    Program mapping:
    - `pid0`: flattened `(BH, NC)` chunk index
    - `pid1`: M tile

    Math:
      factor[t] = prod_{u=t+1..C-1} alpha_u, alpha_u = exp(log_alpha_u)
      dW[t, m]  = factor[t] * sum_d G[m, d] * V[t, d]

    This kernel computes a full `[C, BLOCK_M]` tile per program (not per-token),
    so launch count scales with `M/BLOCK_M` instead of `C * M/BLOCK_M`.
    """
    pid_bhnc = tl.program_id(0)
    pid_m_tile = tl.program_id(1)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size

    r_desc = tl.make_tensor_descriptor(
        log_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_r_bh, stride_r_nc, stride_r_c],
        block_shape=[1, 1, BLOCK_C],
    )
    grad_s_desc = tl.make_tensor_descriptor(
        grad_s_local_end_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_grad_s_bh, stride_grad_s_nc, stride_grad_s_m, stride_grad_s_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_v_bh, stride_v_nc, stride_v_c, stride_v_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    out_dw_desc = tl.make_tensor_descriptor(
        out_dw_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_out_dw_bh, stride_out_dw_nc, stride_out_dw_c, stride_out_dw_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )

    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    INV_LN2 = 1.4426950408889634
    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    factors = tl.exp2((log_suffix_incl - log_alpha_vals) * INV_LN2)

    m_start = pid_m_tile * BLOCK_M
    acc = tl.zeros((BLOCK_C, BLOCK_M), dtype=tl.float32)
    for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
        d_start = d_blk * BLOCK_D
        g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
        v_tile = tl.reshape(v_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
        acc += tl.dot(v_tile, tl.trans(g_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    dW_tile = factors[:, None] * acc
    w_dtype_probe = tl.reshape(v_desc.load([BH_IDX, NC_IDX, 0, 0]), (BLOCK_C, BLOCK_D))
    if ACCUMULATE:
        old_dw = tl.reshape(out_dw_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        dW_tile = dW_tile + old_dw
    out_dw_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW_tile.to(w_dtype_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))


@triton.jit
def ssd_rank1_chunk_end_state_bwd_dv_b_chunk_kernel(
    grad_s_local_end_ptr,
    W_ptr,
    V_ptr,
    log_alpha_ptr,
    out_dv_ptr,
    out_b_vals_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    d_size,
    n_d_tiles_size,
    stride_grad_s_bh,
    stride_grad_s_nc,
    stride_grad_s_m,
    stride_grad_s_d,
    stride_w_bh,
    stride_w_nc,
    stride_w_c,
    stride_w_m,
    stride_v_bh,
    stride_v_nc,
    stride_v_c,
    stride_v_d,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_out_dv_bh,
    stride_out_dv_nc,
    stride_out_dv_c,
    stride_out_dv_d,
    stride_out_b_bh,
    stride_out_b_nc,
    stride_out_b_c,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    M_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """Phase-1 backward kernel for `dV` + `b_t` with chunk-owned programs.

    Program mapping:
    - `pid0`: flattened `(BH, NC)` chunk index
    - `pid1`: D tile

    Computes:
      dv_base[t, d] = sum_m G[m, d] * W[t, m]
      dV[t, d]      = factor[t] * dv_base[t, d]
      b_t[t]       += sum_{d in tile} dv_base[t, d] * V[t, d]
    """
    pid_bhnc = tl.program_id(0)
    pid_d_tile = tl.program_id(1)
    if pid_bhnc >= bh_size * nc_size or pid_d_tile >= n_d_tiles_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size

    r_desc = tl.make_tensor_descriptor(
        log_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_r_bh, stride_r_nc, stride_r_c],
        block_shape=[1, 1, BLOCK_C],
    )
    grad_s_desc = tl.make_tensor_descriptor(
        grad_s_local_end_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_grad_s_bh, stride_grad_s_nc, stride_grad_s_m, stride_grad_s_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    w_desc = tl.make_tensor_descriptor(
        W_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_w_bh, stride_w_nc, stride_w_c, stride_w_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_v_bh, stride_v_nc, stride_v_c, stride_v_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    out_dv_desc = tl.make_tensor_descriptor(
        out_dv_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_out_dv_bh, stride_out_dv_nc, stride_out_dv_c, stride_out_dv_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    INV_LN2 = 1.4426950408889634
    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    factors = tl.exp2((log_suffix_incl - log_alpha_vals) * INV_LN2)

    d_start = pid_d_tile * BLOCK_D
    acc = tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
        w_tile = tl.reshape(w_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        acc += tl.dot(w_tile, g_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    dV_tile = factors[:, None] * acc
    v_dtype_probe = tl.reshape(v_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D))
    if ACCUMULATE:
        old_dv = tl.reshape(out_dv_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
        dV_tile = dV_tile + old_dv
    out_dv_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV_tile.to(v_dtype_probe.dtype), (1, 1, BLOCK_C, BLOCK_D)))

    v_tile = tl.reshape(v_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
    b_partial = tl.sum(acc * v_tile, axis=1)
    offs_c = tl.arange(0, BLOCK_C)
    b_ptrs = out_b_vals_ptr + BH_IDX * stride_out_b_bh + NC_IDX * stride_out_b_nc + offs_c * stride_out_b_c
    tl.atomic_add(b_ptrs, b_partial)


@triton.jit
def ssd_rank1_chunk_end_state_bwd_dr_kernel(
    b_vals_ptr,
    log_alpha_ptr,
    out_dlog_alpha_ptr,
    bh_size,
    nc_size,
    c_size,
    stride_b_bh,
    stride_b_nc,
    stride_b_c,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_out_dlog_alpha_bh,
    stride_out_dlog_alpha_nc,
    stride_out_dlog_alpha_c,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
):
    """Phase-1 backward kernel for `d log_alpha`.

    Inputs:
    - `b_vals`: scalar `b_t` per token
    - `log_alpha`
    Dense vector form for one `(BH, NC)`:

      factor[t] = prod_{u=t+1..C-1} alpha_u
      g[t]      = b_t * factor[t]
      d(log_alpha_t) = sum_{j=0..t-1} g[j]

    i.e. `dlog_alpha` is an exclusive prefix-sum of `g` along token axis.
    """
    pid_bhnc = tl.program_id(0)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size

    r_desc = tl.make_tensor_descriptor(
        log_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_r_bh, stride_r_nc, stride_r_c],
        block_shape=[1, 1, BLOCK_C],
    )
    b_desc = tl.make_tensor_descriptor(
        b_vals_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_b_bh, stride_b_nc, stride_b_c],
        block_shape=[1, 1, BLOCK_C],
    )
    out_dr_desc = tl.make_tensor_descriptor(
        out_dlog_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_out_dlog_alpha_bh, stride_out_dlog_alpha_nc, stride_out_dlog_alpha_c],
        block_shape=[1, 1, BLOCK_C],
    )

    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,))
    b_vals_in = tl.reshape(b_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,))
    log_alpha_f = log_alpha_vals.to(tl.float32)
    INV_LN2 = 1.4426950408889634
    b_vals = b_vals_in.to(tl.float32)
    log_suffix_incl = tl.cumsum(log_alpha_f, axis=0, reverse=True)
    factors = tl.exp2((log_suffix_incl - log_alpha_f) * INV_LN2)
    g_vals = b_vals * factors
    prefix_incl = tl.cumsum(g_vals, axis=0)
    grad_vec = prefix_incl - g_vals
    out_dr_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(grad_vec.to(log_alpha_vals.dtype), (1, 1, BLOCK_C)))


# ==================================================================================================
# PHASE 1 AUTOGRAD
# ==================================================================================================

def _ssd_rank1_chunk_end_state_forward_impl(
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    BLOCK_T: int = 32,
) -> torch.Tensor:
    if W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "_ssd_rank1_chunk_end_state_forward_impl expects "
            "W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
            f"Got W={tuple(W.shape)}, V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    BH, NC, C, _ = W.shape
    if V.shape[:3] != (BH, NC, C):
        raise ValueError(
            "V must match [BH,NC,C,*] from W. "
            f"Got V={tuple(V.shape)}, W={tuple(W.shape)}."
        )
    if log_alpha.shape != (BH, NC, C):
        raise ValueError(
            "log_alpha must match [BH,NC,C] from W/V. "
            f"Got log_alpha={tuple(log_alpha.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}."
        )
    if W.device != V.device or W.device != log_alpha.device:
        raise ValueError("W, V, log_alpha must be on the same device.")
    if W.dtype != V.dtype or W.dtype != log_alpha.dtype:
        raise ValueError("W, V, log_alpha must share dtype.")

    M = W.shape[-1]
    D = V.shape[-1]
    _require_supported_md(M, D, where="_ssd_rank1_chunk_end_state_forward_impl")
    _require_chunk_size_multiple_of_16(C, where="_ssd_rank1_chunk_end_state_forward_impl")
    BLOCK_T = _select_largest_block_size(
        C,
        _SUPPORTED_BLOCK_T_VALUES,
        where="_ssd_rank1_chunk_end_state_forward_impl",
        label="C",
    )
    _require_nc_descriptor_width(NC, where="_ssd_rank1_chunk_end_state_forward_impl")
    _require_phase1_block_t(BLOCK_T, C, where="_ssd_rank1_chunk_end_state_forward_impl")
    _require_nonpositive_log_alpha(log_alpha, where="_ssd_rank1_chunk_end_state_forward_impl")
    if not W.is_cuda:
            raise NotImplementedError("_ssd_rank1_chunk_end_state_forward_impl requires CUDA tensors.")
    _ensure_triton_allocator()

    s_local_end_md = torch.empty((BH, NC, M, D), device=W.device, dtype=torch.float32)
    BLOCK_M = _select_largest_block_size(
        M,
        _SUPPORTED_BLOCK_X_VALUES,
        where="_ssd_rank1_chunk_end_state_forward_impl",
        label="M",
    )
    BLOCK_D = _select_largest_block_size(
        D,
        _SUPPORTED_BLOCK_X_VALUES,
        where="_ssd_rank1_chunk_end_state_forward_impl",
        label="D",
    )
    BLOCK_T = _select_largest_block_size(
        C,
        _SUPPORTED_BLOCK_T_VALUES,
        where="_ssd_rank1_chunk_end_state_forward_impl",
        label="C",
    )
    grid = (BH * NC, M // BLOCK_M, D // BLOCK_D)
    ssd_rank1_chunk_end_state_fwd_kernel[grid](
        W,
        V,
        log_alpha,
        s_local_end_md,
        BH,
        NC,
        C,
        M,
        D,
        *W.stride(),
        *V.stride(),
        *log_alpha.stride(),
        *s_local_end_md.stride(),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        BLOCK_T=BLOCK_T,
        BLOCK_C=C,
        C_STATIC=C,
        INPUT_PRECISION="ieee" if W.dtype == torch.float32 else "tf32",
    )
    return s_local_end_md.reshape(BH, NC, M * D)


_PHASE2_PREFIX_SCAN_FWD_CONFIGS = [
    triton.Config({"BLOCK_MD": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_MD": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_MD": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_MD": 512}, num_warps=8, num_stages=3),
]


# --------------------------------------------------------------------------------------------------
# PHASE 2: PREFIX SCAN OVER CHUNKS
# --------------------------------------------------------------------------------------------------
@triton.autotune(configs=_PHASE2_PREFIX_SCAN_FWD_CONFIGS, key=["md_size", "nc_size"])
@triton.jit
def ssd_rank1_prefix_scan_fwd_kernel(
    s_local_ptr,
    log_alpha_chunk_ptr,
    init_ptr,
    out_prefix_ptr,
    out_final_ptr,
    bh_size,
    nc_size,
    md_size,
    stride_s_local_bh,
    stride_s_local_nc,
    stride_s_local_md,
    stride_log_alpha_chunk_bh,
    stride_log_alpha_chunk_nc,
    stride_init_bh,
    stride_init_md,
    stride_out_prefix_bh,
    stride_out_prefix_nc,
    stride_out_prefix_md,
    stride_out_final_bh,
    stride_out_final_md,
    BLOCK_NC: tl.constexpr,
    BLOCK_MD: tl.constexpr,
    NC_STATIC: tl.constexpr,
    USE_FP32_COMPUTE: tl.constexpr,
    USE_BF16_COMPUTE: tl.constexpr,
):
    """Single-kernel dense blocked phase-2 forward.

    One program owns one `BH` lane and loops over chunk blocks then MD tiles:
      1) Form local block prefixes in log space.
      2) Build direct strict-lower carry matrix for this block:
           L0[i,j] = 1_{j<i} * exp(log_p_excl[i] - log_p_incl[j]).
      3) Compute block chunk-start states:
           S0_block = L0 @ S_loc + p[:, None] * S_in[None, :].
      3) Store `S0_block`, update carry:
           S_in <- alpha_last * S0_block_last + S_local_block_last
      4) Repeat for next block.

    After loop:
      out_final = S_in
    """
    pid_bh = tl.program_id(0)
    if pid_bh >= bh_size:
        return

    s_local_desc = tl.make_tensor_descriptor(
        s_local_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_s_local_bh, stride_s_local_nc, stride_s_local_md],
        block_shape=[1, BLOCK_NC, BLOCK_MD],
    )
    log_alpha_chunk_desc = tl.make_tensor_descriptor(
        log_alpha_chunk_ptr,
        shape=[bh_size, nc_size],
        strides=[stride_log_alpha_chunk_bh, stride_log_alpha_chunk_nc],
        block_shape=[1, BLOCK_NC],
    )
    init_desc = tl.make_tensor_descriptor(
        init_ptr,
        shape=[bh_size, md_size],
        strides=[stride_init_bh, stride_init_md],
        block_shape=[1, BLOCK_MD],
    )
    out_prefix_desc = tl.make_tensor_descriptor(
        out_prefix_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_out_prefix_bh, stride_out_prefix_nc, stride_out_prefix_md],
        block_shape=[1, BLOCK_NC, BLOCK_MD],
    )
    out_final_desc = tl.make_tensor_descriptor(
        out_final_ptr,
        shape=[bh_size, md_size],
        strides=[stride_out_final_bh, stride_out_final_md],
        block_shape=[1, BLOCK_MD],
    )

    INV_LN2 = 1.4426950408889634
    offs_blk = tl.arange(0, BLOCK_NC)
    row_ids = tl.expand_dims(offs_blk, axis=1)
    col_ids = tl.expand_dims(offs_blk, axis=0)
    l0_mask = row_ids > col_ids
    last_mask = offs_blk == (BLOCK_NC - 1)

    for c0 in tl.static_range(0, NC_STATIC, BLOCK_NC):
        # L-related block math (prefixes and scaling vectors) computed once per NC block.
        log_alpha_blk = tl.reshape(log_alpha_chunk_desc.load([pid_bh, c0]), (BLOCK_NC,)).to(tl.float32)
        log_alpha_blk_l2 = log_alpha_blk * INV_LN2
        log_p_incl_l2 = tl.cumsum(log_alpha_blk_l2, axis=0)
        log_p_excl_l2 = log_p_incl_l2 - log_alpha_blk_l2
        p = tl.exp2(log_p_excl_l2)
        alpha_last = tl.exp2(tl.sum(tl.where(last_mask, log_alpha_blk_l2, 0.0), axis=0))
        l0_log2 = tl.expand_dims(log_p_excl_l2, axis=1) - tl.expand_dims(log_p_incl_l2, axis=0)
        l0_f32 = tl.where(l0_mask, tl.exp2(l0_log2), 0.0)
        if USE_FP32_COMPUTE:
            l0_tc = l0_f32
        else:
            if USE_BF16_COMPUTE:
                l0_tc = l0_f32.to(tl.bfloat16)
            else:
                l0_tc = l0_f32.to(tl.float16)

        md0 = 0
        while md0 < md_size:
            if c0 == 0:
                S_in = tl.reshape(init_desc.load([pid_bh, md0]), (BLOCK_MD,)).to(tl.float32)
            else:
                S_in = tl.reshape(out_final_desc.load([pid_bh, md0]), (BLOCK_MD,)).to(tl.float32)

            s_tile_f32 = tl.reshape(s_local_desc.load([pid_bh, c0, md0]), (BLOCK_NC, BLOCK_MD)).to(tl.float32)
            if USE_FP32_COMPUTE:
                y0 = tl.dot(l0_tc, s_tile_f32, out_dtype=tl.float32, input_precision="ieee")
            else:
                if USE_BF16_COMPUTE:
                    s_tile_tc = s_tile_f32.to(tl.bfloat16)
                else:
                    s_tile_tc = s_tile_f32.to(tl.float16)
                y0 = tl.dot(l0_tc, s_tile_tc, out_dtype=tl.float32, input_precision="tf32")

            acc = y0 + tl.expand_dims(p, axis=1) * tl.expand_dims(S_in, axis=0)

            out_prefix_desc.store([pid_bh, c0, md0], tl.reshape(acc, (1, BLOCK_NC, BLOCK_MD)))

            s0_last = tl.sum(tl.where(tl.expand_dims(last_mask, axis=1), acc, 0.0), axis=0)
            s_local_last = tl.sum(tl.where(tl.expand_dims(last_mask, axis=1), s_tile_f32, 0.0), axis=0)
            S_next = alpha_last * s0_last + s_local_last
            out_final_desc.store([pid_bh, md0], tl.reshape(S_next, (1, BLOCK_MD)))
            md0 += BLOCK_MD


_PHASE2_PREFIX_SCAN_BWD_CONFIGS = [
    triton.Config({"BLOCK_MD": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_MD": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_MD": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_MD": 512}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_PHASE2_PREFIX_SCAN_BWD_CONFIGS, key=["md_size", "nc_size"])
@triton.jit
def ssd_rank1_prefix_scan_bwd_dense_kernel(
    grad_prefix_ptr,
    grad_final_ptr,
    chunk_start_ptr,
    log_alpha_chunk_ptr,
    d_s_local_ptr,
    d_log_alpha_ptr,
    d_init_ptr,
    bh_size,
    nc_size,
    md_size,
    stride_grad_prefix_bh,
    stride_grad_prefix_nc,
    stride_grad_prefix_md,
    stride_grad_final_bh,
    stride_grad_final_md,
    stride_chunk_start_bh,
    stride_chunk_start_nc,
    stride_chunk_start_md,
    stride_log_alpha_chunk_bh,
    stride_log_alpha_chunk_nc,
    stride_d_s_local_bh,
    stride_d_s_local_nc,
    stride_d_s_local_md,
    stride_d_log_alpha_bh,
    stride_d_log_alpha_nc,
    stride_d_init_bh,
    stride_d_init_md,
    BLOCK_MD: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    NC_STATIC: tl.constexpr,
    USE_FP32_COMPUTE: tl.constexpr,
    USE_BF16_COMPUTE: tl.constexpr,
):
    """Dense blocked reverse pass for phase-2 prefix scan.

    Recurrence in chunk index c:
      S_{c+1} = alpha[c] * S_c + S_local[c]
      y_start[c] = S_c
      y_final = S_NC

    Reverse recurrence:
      lambda_NC = g_final
      lambda_c  = g_start[c] + alpha[c] * lambda_{c+1}
      dS_local[c] = lambda_{c+1}
      dalpha[c]   = sum_md(lambda_{c+1} * S_c)
      dlog_alpha[c] = dalpha[c] * alpha[c]

    Kernel structure mirrors forward:
      - one program per BH lane
      - static loop over NC blocks in reverse order
      - inner loop over MD tiles
      - per block uses dense strict-lower carry matrix in reversed chunk order:
          lambda_next_rev = L0_rev @ g_rev + p_rev * lambda_in
    """
    pid_bh = tl.program_id(0)
    if pid_bh >= bh_size:
        return

    grad_prefix_desc = tl.make_tensor_descriptor(
        grad_prefix_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_grad_prefix_bh, stride_grad_prefix_nc, stride_grad_prefix_md],
        block_shape=[1, BLOCK_NC, BLOCK_MD],
    )
    grad_final_desc = tl.make_tensor_descriptor(
        grad_final_ptr,
        shape=[bh_size, md_size],
        strides=[stride_grad_final_bh, stride_grad_final_md],
        block_shape=[1, BLOCK_MD],
    )
    chunk_start_desc = tl.make_tensor_descriptor(
        chunk_start_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_chunk_start_bh, stride_chunk_start_nc, stride_chunk_start_md],
        block_shape=[1, BLOCK_NC, BLOCK_MD],
    )
    log_alpha_chunk_desc = tl.make_tensor_descriptor(
        log_alpha_chunk_ptr,
        shape=[bh_size, nc_size],
        strides=[stride_log_alpha_chunk_bh, stride_log_alpha_chunk_nc],
        block_shape=[1, BLOCK_NC],
    )
    d_s_local_desc = tl.make_tensor_descriptor(
        d_s_local_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_d_s_local_bh, stride_d_s_local_nc, stride_d_s_local_md],
        block_shape=[1, BLOCK_NC, BLOCK_MD],
    )
    d_log_alpha_desc = tl.make_tensor_descriptor(
        d_log_alpha_ptr,
        shape=[bh_size, nc_size],
        strides=[stride_d_log_alpha_bh, stride_d_log_alpha_nc],
        block_shape=[1, BLOCK_NC],
    )
    d_init_desc = tl.make_tensor_descriptor(
        d_init_ptr,
        shape=[bh_size, md_size],
        strides=[stride_d_init_bh, stride_d_init_md],
        block_shape=[1, BLOCK_MD],
    )

    INV_LN2 = 1.4426950408889634
    offs_nc = tl.arange(0, BLOCK_NC)
    row_ids = tl.expand_dims(offs_nc, axis=1)
    col_ids = tl.expand_dims(offs_nc, axis=0)
    l0_mask = row_ids > col_ids
    last_mask = offs_nc == (BLOCK_NC - 1)

    for blk in tl.static_range(0, NC_STATIC // BLOCK_NC):
        c0 = NC_STATIC - (blk + 1) * BLOCK_NC

        log_alpha_blk = tl.reshape(log_alpha_chunk_desc.load([pid_bh, c0]), (BLOCK_NC,)).to(tl.float32)
        log_alpha_blk_l2 = log_alpha_blk * INV_LN2
        log_alpha_rev_l2 = tl.flip(log_alpha_blk_l2, 0)
        log_p_incl_rev_l2 = tl.cumsum(log_alpha_rev_l2, axis=0)
        log_p_excl_rev_l2 = log_p_incl_rev_l2 - log_alpha_rev_l2
        p_rev = tl.exp2(log_p_excl_rev_l2)
        alpha_last_rev = tl.exp2(tl.sum(tl.where(last_mask, log_alpha_rev_l2, 0.0), axis=0))

        l0_log2 = tl.expand_dims(log_p_excl_rev_l2, axis=1) - tl.expand_dims(log_p_incl_rev_l2, axis=0)
        l0_f32 = tl.where(l0_mask, tl.exp2(l0_log2), 0.0)
        if USE_FP32_COMPUTE:
            l0_tc = l0_f32
        else:
            if USE_BF16_COMPUTE:
                l0_tc = l0_f32.to(tl.bfloat16)
            else:
                l0_tc = l0_f32.to(tl.float16)

        dlog_block = tl.zeros((BLOCK_NC,), dtype=tl.float32)
        md0 = 0
        while md0 < md_size:
            if blk == 0:
                lambda_in = tl.reshape(grad_final_desc.load([pid_bh, md0]), (BLOCK_MD,)).to(tl.float32)
            else:
                lambda_in = tl.reshape(d_init_desc.load([pid_bh, md0]), (BLOCK_MD,)).to(tl.float32)

            g_block = tl.reshape(grad_prefix_desc.load([pid_bh, c0, md0]), (BLOCK_NC, BLOCK_MD)).to(tl.float32)
            g_rev = tl.flip(g_block, 0)

            if USE_FP32_COMPUTE:
                y0 = tl.dot(l0_tc, g_rev, out_dtype=tl.float32, input_precision="ieee")
            else:
                if USE_BF16_COMPUTE:
                    g_rev_tc = g_rev.to(tl.bfloat16)
                else:
                    g_rev_tc = g_rev.to(tl.float16)
                y0 = tl.dot(l0_tc, g_rev_tc, out_dtype=tl.float32, input_precision="tf32")

            lambda_next_rev = y0 + tl.expand_dims(p_rev, axis=1) * tl.expand_dims(lambda_in, axis=0)
            lambda_next = tl.flip(lambda_next_rev, 0)
            d_s_local_desc.store([pid_bh, c0, md0], tl.reshape(lambda_next, (1, BLOCK_NC, BLOCK_MD)))

            s_block = tl.reshape(chunk_start_desc.load([pid_bh, c0, md0]), (BLOCK_NC, BLOCK_MD)).to(tl.float32)
            dlog_block += tl.sum(lambda_next * s_block, axis=1)

            lambda_start_last = tl.sum(tl.where(tl.expand_dims(last_mask, axis=1), lambda_next_rev, 0.0), axis=0)
            g_last_rev = tl.sum(tl.where(tl.expand_dims(last_mask, axis=1), g_rev, 0.0), axis=0)
            lambda_out = alpha_last_rev * lambda_start_last + g_last_rev
            d_init_desc.store([pid_bh, md0], tl.reshape(lambda_out, (1, BLOCK_MD)))
            md0 += BLOCK_MD

        alpha_blk = tl.exp2(log_alpha_blk_l2)
        dlog_out = dlog_block * alpha_blk
        d_log_alpha_desc.store([pid_bh, c0], tl.reshape(dlog_out, (1, BLOCK_NC)))


# ==================================================================================================
# PHASE 2 AUTOGRAD
# ==================================================================================================

def _ssd_rank1_prefix_scan_forward_impl(
    S_local_end: torch.Tensor,
    log_alpha_chunk: torch.Tensor,
    initial_state: torch.Tensor,
    compute_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if S_local_end.ndim != 3 or log_alpha_chunk.ndim != 2 or initial_state.ndim != 2:
        raise ValueError(
            "_ssd_rank1_prefix_scan_forward_impl expects "
            "S_local_end=[BH,NC,MD], log_alpha_chunk=[BH,NC], initial_state=[BH,MD]. "
            f"Got S_local_end={tuple(S_local_end.shape)}, log_alpha_chunk={tuple(log_alpha_chunk.shape)}, "
            f"initial_state={tuple(initial_state.shape)}."
        )
    BH, NC, MD = S_local_end.shape
    if log_alpha_chunk.shape != (BH, NC):
        raise ValueError(f"log_alpha_chunk must be [BH,NC]. Got {tuple(log_alpha_chunk.shape)} vs ({BH}, {NC}).")
    if initial_state.shape != (BH, MD):
        raise ValueError(f"initial_state must be [BH,MD]. Got {tuple(initial_state.shape)} vs ({BH}, {MD}).")
    if S_local_end.device != log_alpha_chunk.device or S_local_end.device != initial_state.device:
        raise ValueError("S_local_end, log_alpha_chunk, and initial_state must be on the same device.")
    if NC == 0:
        raise NotImplementedError("_ssd_rank1_prefix_scan_forward_impl does not support NC==0.")
    _require_nc_descriptor_width(NC, where="_ssd_rank1_prefix_scan_forward_impl")
    _require_nc_multiple_of_16(NC, where="_ssd_rank1_prefix_scan_forward_impl")
    if not S_local_end.is_cuda:
        raise NotImplementedError("_ssd_rank1_prefix_scan_forward_impl requires CUDA tensors.")
    _ensure_triton_allocator()

    if compute_dtype is None:
        compute_dtype = torch.float32
    if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "_ssd_rank1_prefix_scan_forward_impl supports compute_dtype in "
            "{torch.float16, torch.bfloat16, torch.float32}. "
            f"Got compute_dtype={compute_dtype}."
        )

    S_local_end_in = S_local_end if S_local_end.is_contiguous() else S_local_end.contiguous()
    if log_alpha_chunk.dtype == torch.float32 and log_alpha_chunk.is_contiguous():
        log_alpha_chunk_f = log_alpha_chunk
    else:
        log_alpha_chunk_f = log_alpha_chunk.float().contiguous()
    if initial_state.dtype == torch.float32 and initial_state.is_contiguous():
        initial_state_f = initial_state
    else:
        initial_state_f = initial_state.float().contiguous()

    chunk_start = torch.empty((BH, NC, MD), device=S_local_end.device, dtype=torch.float32)
    final_state = torch.empty((BH, MD), device=S_local_end.device, dtype=torch.float32)

    block_nc = _select_phase2_block_nc(NC=NC)
    grid_fwd = lambda META: (BH,)
    ssd_rank1_prefix_scan_fwd_kernel[grid_fwd](
        S_local_end_in,
        log_alpha_chunk_f,
        initial_state_f,
        chunk_start,
        final_state,
        BH,
        NC,
        MD,
        *S_local_end_in.stride(),
        *log_alpha_chunk_f.stride(),
        *initial_state_f.stride(),
        *chunk_start.stride(),
        *final_state.stride(),
        NC_STATIC=NC,
        BLOCK_NC=block_nc,
        USE_FP32_COMPUTE=(compute_dtype == torch.float32),
        USE_BF16_COMPUTE=(compute_dtype == torch.bfloat16),
    )
    return chunk_start, final_state


# --------------------------------------------------------------------------------------------------
# END PHASE 2: PREFIX SCAN OVER CHUNKS
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# PHASE 3: DENSE OUTPUT COMPUTATION
# --------------------------------------------------------------------------------------------------
def ssd_rank1_dense_output_reference(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    S0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Phase-3 PyTorch dense CxC reference matching kernel numerics.

    Target equation:

      y_t = sum_{tau<=t} [prod_{u=tau+1..t} alpha_u] * (C_t^T W_tau) * V_tau
            + [prod_{u=0..t} alpha_u] * (C_t^T S0)
      where alpha_u = exp(log_alpha_u).

    Dense construction per `(BH, NC)`:
    1. Build causal carry matrix `L` in FP32.
    2. Build `R = C @ W^T` in FP32.
    3. Form `K = L * R` in FP32.
    4. Contract `Y_diag = K @ V` for intra-chunk contributions.
    5. Add chunk-prefix contribution
         `Y_off[t] = exp(sum_{u=0..t} log_alpha_u) * (C_t^T S0)`.
    """
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "ssd_rank1_dense_output_reference expects C=[BH,NC,C,M], "
            "W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, "
            f"V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    BH, NC, C_CHUNK, M = C.shape
    if W.shape != (BH, NC, C_CHUNK, M):
        raise ValueError(f"W must be [BH,NC,C,M]={BH, NC, C_CHUNK, M}; got {tuple(W.shape)}.")
    if V.shape[:3] != (BH, NC, C_CHUNK):
        raise ValueError(f"V must match [BH,NC,C,*]={BH, NC, C_CHUNK}; got {tuple(V.shape)}.")
    D = V.shape[-1]
    if log_alpha.shape != (BH, NC, C_CHUNK):
        raise ValueError(f"log_alpha must be [BH,NC,C]={BH, NC, C_CHUNK}; got {tuple(log_alpha.shape)}.")
    if C.device != W.device or C.device != V.device or C.device != log_alpha.device:
        raise ValueError("C, W, V, log_alpha must be on the same device.")
    if C.dtype != W.dtype or C.dtype != V.dtype or C.dtype != log_alpha.dtype:
        raise ValueError("C, W, V, log_alpha must share dtype.")
    if C_CHUNK == 0:
        raise NotImplementedError("ssd_rank1_dense_output_reference does not support C==0.")
    _require_supported_md(M, D, where="ssd_rank1_dense_output_reference")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank1_dense_output_reference")
    if S0 is None:
        S0_md = torch.zeros((BH, NC, M, D), device=C.device, dtype=torch.float32)
    else:
        if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
            S0_md = S0.reshape(BH, NC, M, D).to(torch.float32)
        elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
            S0_md = S0.to(torch.float32)
        else:
            raise ValueError(
                "S0 must be [BH,NC,MD] or [BH,NC,M,D] matching C/W/V. "
                f"Got S0={tuple(S0.shape)} with expected BH={BH}, NC={NC}, M={M}, D={D}."
            )
        if S0.device != C.device:
            raise ValueError("S0 must be on the same device as C/W/V/log_alpha.")

    C_f = C.float()
    W_f = W.float()
    V_f = V.float()
    log_alpha_f = log_alpha.float()

    c_idx = torch.arange(C_CHUNK, device=C.device)
    tril = (c_idx.view(C_CHUNK, 1) >= c_idx.view(1, C_CHUNK)).view(1, 1, C_CHUNK, C_CHUNK)

    log_p = torch.cumsum(log_alpha_f, dim=-1)
    # Guard against masked-overflow autograd pathology: upper-tri deltas can be
    # large positive even though they are masked out. Zero masked lanes before
    # exp2 so no inf is created while preserving exact valid-lane math.
    log_delta_l2 = (log_p[..., :, None] - log_p[..., None, :]) * _INV_LN2
    log_delta_l2 = torch.where(
        tril,
        log_delta_l2,
        torch.zeros((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32),
    )
    L = torch.where(
        tril,
        torch.exp2(log_delta_l2),
        torch.zeros((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32),
    )

    p = torch.exp2(log_p * _INV_LN2).to(torch.float32).unsqueeze(-1)
    Y_off = p * torch.matmul(C_f, S0_md)

    K = (L * (C_f @ W_f.mT)).to(torch.float32)
    Y_diag = torch.matmul(K, V_f)

    return (Y_diag + Y_off).to(V.dtype)


def ssd_rank1_dense_output_backward_reference(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    grad_Y: torch.Tensor,
    S0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Phase-3 backward reference for dense chunk-local output.

    Forward (per `(BH, NC)` chunk):
      R = C @ W^T                        # [C, C]
      L[i,j] = 1_{j<=i} * prod_{u=j+1..i} alpha_u, alpha_u = exp(log_alpha_u)
      K = L * R                          # [C, C]
      Y_diag = K @ V                     # [C, D]
      Y_off  = p * (C @ S0), p_i = prod_{u=0..i} alpha_u
      Y = Y_diag + Y_off

    Dense backward:
      dV = K^T @ dY
      dK = dY @ V^T
      dR = dK * L
      dC_diag = dR @ W
      dW = dR^T @ C

    For `log_alpha` from the dense carry matrix:
      Q = dK * R * L
      dlog_alpha[k] += sum_{i>=k} sum_{j<k} Q[i,j]

    For `log_alpha` from `Y_off`:
      dB = dY * p, B = C @ S0
      dlog_alpha[k] += sum_{i>=k} (dY_i dot B_i) * p_i

    Returns:
    - dC: `[BH, NC, C, M]`
    - dW: `[BH, NC, C, M]`
    - dV: `[BH, NC, C, D]`
    - dlog_alpha: `[BH, NC, C]`
    - dS0: `[BH, NC, M, D]` if `S0` is provided, else `None`
    """
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3 or grad_Y.ndim != 4:
        raise ValueError(
            "ssd_rank1_dense_output_backward_reference expects "
            "C=[BH,NC,C,M], W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C], grad_Y=[BH,NC,C,D]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}, "
            f"log_alpha={tuple(log_alpha.shape)}, grad_Y={tuple(grad_Y.shape)}."
        )
    BH, NC, C_CHUNK, M = C.shape
    if W.shape != (BH, NC, C_CHUNK, M):
        raise ValueError(f"W must be [BH,NC,C,M]={BH, NC, C_CHUNK, M}; got {tuple(W.shape)}.")
    if V.shape[:3] != (BH, NC, C_CHUNK):
        raise ValueError(f"V must match [BH,NC,C,*]={BH, NC, C_CHUNK}; got {tuple(V.shape)}.")
    D = V.shape[-1]
    if log_alpha.shape != (BH, NC, C_CHUNK):
        raise ValueError(f"log_alpha must be [BH,NC,C]={BH, NC, C_CHUNK}; got {tuple(log_alpha.shape)}.")
    if grad_Y.shape != (BH, NC, C_CHUNK, D):
        raise ValueError(f"grad_Y must be [BH,NC,C,D]={BH, NC, C_CHUNK, D}; got {tuple(grad_Y.shape)}.")
    if C.device != W.device or C.device != V.device or C.device != log_alpha.device or C.device != grad_Y.device:
        raise ValueError("C, W, V, log_alpha, grad_Y must be on the same device.")
    if C.dtype != W.dtype or C.dtype != V.dtype or C.dtype != log_alpha.dtype or C.dtype != grad_Y.dtype:
        raise ValueError("C, W, V, log_alpha, grad_Y must share dtype.")
    if C_CHUNK == 0:
        raise NotImplementedError("ssd_rank1_dense_output_backward_reference does not support C==0.")
    _require_supported_md(M, D, where="ssd_rank1_dense_output_backward_reference")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank1_dense_output_backward_reference")

    if S0 is None:
        S0_md = torch.zeros((BH, NC, M, D), device=C.device, dtype=torch.float32)
        return_s0_grad = False
    else:
        if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
            S0_md = S0.reshape(BH, NC, M, D).to(torch.float32)
        elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
            S0_md = S0.to(torch.float32)
        else:
            raise ValueError(
                "S0 must be [BH,NC,MD] or [BH,NC,M,D] matching C/W/V. "
                f"Got S0={tuple(S0.shape)} with expected BH={BH}, NC={NC}, M={M}, D={D}."
            )
        if S0.device != C.device:
            raise ValueError("S0 must be on the same device as C/W/V/log_alpha/grad_Y.")
        return_s0_grad = True

    C_f = C.float()
    W_f = W.float()
    V_f = V.float()
    log_alpha_f = log_alpha.float()
    grad_Y_f = grad_Y.float()

    c_idx = torch.arange(C_CHUNK, device=C.device)
    tril = (c_idx.view(C_CHUNK, 1) >= c_idx.view(1, C_CHUNK)).view(1, 1, C_CHUNK, C_CHUNK)

    log_p = torch.cumsum(log_alpha_f, dim=-1)
    p = torch.exp2(log_p * _INV_LN2)
    log_delta_l2 = (log_p[..., :, None] - log_p[..., None, :]) * _INV_LN2
    log_delta_l2 = torch.where(
        tril,
        log_delta_l2,
        torch.zeros((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32),
    )
    L = torch.where(
        tril,
        torch.exp2(log_delta_l2),
        torch.zeros((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32),
    )
    R = C_f @ W_f.mT
    K = L * R

    # Main dense path.
    dV = K.mT @ grad_Y_f
    dK = grad_Y_f @ V_f.mT
    dR = dK * L
    dC_diag = dR @ W_f
    dW = dR.mT @ C_f

    # Off-path via Y_off = p * (C @ S0).
    B = C_f @ S0_md
    dB = grad_Y_f * p.unsqueeze(-1)
    dC_off = dB @ S0_md.mT
    dS0 = C_f.mT @ dB

    dC = dC_diag + dC_off

    # dlog_alpha from dense carry path: Q = dK * R * L and rectangle sums.
    Q = dK * R * L
    left_prefix = torch.cumsum(Q, dim=-1)
    left_of_k = torch.zeros_like(Q)
    left_of_k[..., :, 1:] = left_prefix[..., :, :-1]
    suffix_rows = torch.flip(torch.cumsum(torch.flip(left_of_k, dims=[-2]), dim=-2), dims=[-2])
    dlog_alpha_diag = torch.diagonal(suffix_rows, dim1=-2, dim2=-1)

    # dlog_alpha from off-path p_i factors.
    dlog_alpha_off_src = (grad_Y_f * B).sum(dim=-1) * p
    dlog_alpha_off = torch.flip(torch.cumsum(torch.flip(dlog_alpha_off_src, dims=[-1]), dim=-1), dims=[-1])
    dlog_alpha = dlog_alpha_diag + dlog_alpha_off

    dC_out = dC.to(C.dtype)
    dW_out = dW.to(W.dtype)
    dV_out = dV.to(V.dtype)
    dlog_alpha_out = dlog_alpha.to(log_alpha.dtype)
    dS0_out = dS0.to(C.dtype) if return_s0_grad else None
    return dC_out, dW_out, dV_out, dlog_alpha_out, dS0_out


@triton.jit
def ssd_rank1_dense_output_fwd_kernel(
    C_ptr,
    W_ptr,
    V_ptr,
    log_alpha_ptr,
    S0_ptr,
    out_y_ptr,
    out_y_off_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    d_size,
    stride_c_bh,
    stride_c_nc,
    stride_c_c,
    stride_c_m,
    stride_w_bh,
    stride_w_nc,
    stride_w_c,
    stride_w_m,
    stride_v_bh,
    stride_v_nc,
    stride_v_c,
    stride_v_d,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_s0_bh,
    stride_s0_nc,
    stride_s0_m,
    stride_s0_d,
    stride_out_y_bh,
    stride_out_y_nc,
    stride_out_y_c,
    stride_out_y_d,
    stride_out_y_off_bh,
    stride_out_y_off_nc,
    stride_out_y_off_c,
    stride_out_y_off_d,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    M_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    WRITE_Y_OFF: tl.constexpr,
):
    """Phase-3 Triton forward following the dense CxC reference structure.

    For each `(BH, NC)` chunk, build:
    1. `L[t, tau] = 1_{tau<=t} * exp(sum_{u=tau+1..t} log_alpha_u)`.
    2. `R = C @ W^T`.
    3. `K = L * R`.
    4. `Y_diag = K @ V`.
    5. `Y_off[t] = exp(sum_{u=0..t} log_alpha_u) * (C_t^T S0)`.
    6. `Y = Y_diag + Y_off`.

    Program id mapping:
    - `pid0`: flattened `(BH, NC)`
    - `pid1`: output `D` tile
    """
    pid_bhnc = tl.program_id(0)
    pid_d_tile = tl.program_id(1)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_c_bh, stride_c_nc, stride_c_c, stride_c_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    w_desc = tl.make_tensor_descriptor(
        W_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_w_bh, stride_w_nc, stride_w_c, stride_w_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_v_bh, stride_v_nc, stride_v_c, stride_v_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    r_desc = tl.make_tensor_descriptor(
        log_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_r_bh, stride_r_nc, stride_r_c],
        block_shape=[1, 1, BLOCK_C],
    )
    s0_desc = tl.make_tensor_descriptor(
        S0_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_s0_bh, stride_s0_nc, stride_s0_m, stride_s0_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    out_desc = tl.make_tensor_descriptor(
        out_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_out_y_bh, stride_out_y_nc, stride_out_y_c, stride_out_y_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )

    offs_c = tl.arange(0, BLOCK_C)

    # Step 1: build dense causal factor matrix L[t, tau] in log space.
    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    log_p_incl = tl.cumsum(log_alpha_vals, axis=0)
    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    valid = col_idx <= row_idx
    INV_LN2 = 1.4426950408889634
    log_delta_l2 = (log_p_incl[:, None] - log_p_incl[None, :]) * INV_LN2
    # Keep masked upper-tri lanes finite even if the compiler evaluates both arms.
    log_delta_l2 = tl.where(valid, log_delta_l2, 0.0)
    L = tl.where(valid, tl.exp2(log_delta_l2), 0.0)

    # Step 2: build dense score matrix R = C @ W^T.
    R = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        C = tl.reshape(c_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        W = tl.reshape(w_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        R += tl.dot(C, tl.trans(W), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    # Step 3: form K = L * R.
    K = L * R

    # Step 4: compute Y_diag = K @ V in FP32.
    d_start = pid_d_tile * BLOCK_D
    V_in = tl.reshape(v_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D))
    V_f = V_in.to(tl.float32)
    Y_diag = tl.dot(K, V_f, out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    # Step 5: compute Y_off = prefix_incl * (C @ S0).
    Y_off_base = tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        C_blk = tl.reshape(c_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        S0_blk = tl.reshape(s0_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
        Y_off_base += tl.dot(C_blk, S0_blk, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    p = tl.exp2(log_p_incl * INV_LN2)
    Y_off = p[:, None] * Y_off_base
    Y = Y_diag + Y_off

    out_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(Y.to(V_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))
    if WRITE_Y_OFF:
        out_y_off_desc = tl.make_tensor_descriptor(
            out_y_off_ptr,
            shape=[bh_size, nc_size, c_size, d_size],
            strides=[stride_out_y_off_bh, stride_out_y_off_nc, stride_out_y_off_c, stride_out_y_off_d],
            block_shape=[1, 1, BLOCK_C, BLOCK_D],
        )
        out_y_off_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(Y_off.to(V_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))


@triton.jit
def ssd_rank1_dense_output_bwd_a1_kernel(
    C_ptr,
    W_ptr,
    V_ptr,
    log_alpha_ptr,
    grad_y_ptr,
    out_dV_ptr,
    out_dK_partials_ptr,
    out_R_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    d_size,
    n_d_tiles,
    stride_c_bh,
    stride_c_nc,
    stride_c_c,
    stride_c_m,
    stride_w_bh,
    stride_w_nc,
    stride_w_c,
    stride_w_m,
    stride_v_bh,
    stride_v_nc,
    stride_v_c,
    stride_v_d,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_gy_bh,
    stride_gy_nc,
    stride_gy_c,
    stride_gy_d,
    stride_dv_bh,
    stride_dv_nc,
    stride_dv_c,
    stride_dv_d,
    stride_dk_flat,
    stride_dk_c0,
    stride_dk_c1,
    stride_rout_bh,
    stride_rout_nc,
    stride_rout_c0,
    stride_rout_c1,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    M_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Phase-3 backward A1: per `(BH,NC,D-tile)` compute `dV` tile and `dK` partial.

    For one chunk:
      R = C @ W^T
      L from log_alpha
      K = L * R
      dV[:, d_tile]      = K^T @ dY[:, d_tile]
      dK_partial[d_tile] = dY[:, d_tile] @ V[:, d_tile]^T
    """
    pid_bhnc = tl.program_id(0)
    pid_d_tile = tl.program_id(1)
    if pid_bhnc >= bh_size * nc_size or pid_d_tile >= n_d_tiles:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_c_bh, stride_c_nc, stride_c_c, stride_c_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    w_desc = tl.make_tensor_descriptor(
        W_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_w_bh, stride_w_nc, stride_w_c, stride_w_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_v_bh, stride_v_nc, stride_v_c, stride_v_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    r_desc = tl.make_tensor_descriptor(
        log_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_r_bh, stride_r_nc, stride_r_c],
        block_shape=[1, 1, BLOCK_C],
    )
    gy_desc = tl.make_tensor_descriptor(
        grad_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_gy_bh, stride_gy_nc, stride_gy_c, stride_gy_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    dv_desc = tl.make_tensor_descriptor(
        out_dV_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_dv_bh, stride_dv_nc, stride_dv_c, stride_dv_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    dk_desc = tl.make_tensor_descriptor(
        out_dK_partials_ptr,
        shape=[bh_size * nc_size * n_d_tiles, c_size, c_size],
        strides=[stride_dk_flat, stride_dk_c0, stride_dk_c1],
        block_shape=[1, BLOCK_C, BLOCK_C],
    )
    r_out_desc = tl.make_tensor_descriptor(
        out_R_ptr,
        shape=[bh_size, nc_size, c_size, c_size],
        strides=[stride_rout_bh, stride_rout_nc, stride_rout_c0, stride_rout_c1],
        block_shape=[1, 1, BLOCK_C, BLOCK_C],
    )

    offs_c = tl.arange(0, BLOCK_C)
    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    log_p_incl = tl.cumsum(log_alpha_vals, axis=0)
    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    valid = col_idx <= row_idx
    INV_LN2 = 1.4426950408889634
    log_delta_l2 = (log_p_incl[:, None] - log_p_incl[None, :]) * INV_LN2
    log_delta_l2 = tl.where(valid, log_delta_l2, 0.0)
    L = tl.where(valid, tl.exp2(log_delta_l2), 0.0)

    R = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        C_blk = tl.reshape(c_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        W_blk = tl.reshape(w_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        R += tl.dot(C_blk, tl.trans(W_blk), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    K = L * R

    d_start = pid_d_tile * BLOCK_D
    G_in = tl.reshape(gy_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D))
    V_in = tl.reshape(v_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D))
    G = G_in.to(tl.float32)
    V_f = V_in.to(tl.float32)

    dV_tile = tl.dot(tl.trans(K), G, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    dv_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV_tile.to(V_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))

    dK_partial = tl.dot(G, tl.trans(V_f), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    flat_idx = pid_bhnc * n_d_tiles + pid_d_tile
    dk_desc.store([flat_idx, 0, 0], tl.reshape(dK_partial, (1, BLOCK_C, BLOCK_C)))

    if pid_d_tile == 0:
        r_out_desc.store([BH_IDX, NC_IDX, 0, 0], tl.reshape(R, (1, 1, BLOCK_C, BLOCK_C)))


@triton.jit
def ssd_rank1_dense_output_bwd_a2_kernel(
    C_ptr,
    W_ptr,
    log_alpha_ptr,
    dK_ptr,
    R_ptr,
    out_dC_diag_ptr,
    out_dW_ptr,
    out_Q_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    stride_c_bh,
    stride_c_nc,
    stride_c_c,
    stride_c_m,
    stride_w_bh,
    stride_w_nc,
    stride_w_c,
    stride_w_m,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_dk_bh,
    stride_dk_nc,
    stride_dk_c0,
    stride_dk_c1,
    stride_rbuf_bh,
    stride_rbuf_nc,
    stride_rbuf_c0,
    stride_rbuf_c1,
    stride_dc_bh,
    stride_dc_nc,
    stride_dc_c,
    stride_dc_m,
    stride_dw_bh,
    stride_dw_nc,
    stride_dw_c,
    stride_dw_m,
    stride_q_bh,
    stride_q_nc,
    stride_q_c0,
    stride_q_c1,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Phase-3 backward A2: per `(BH,NC,M-tile)` compute `dC_diag`, `dW`; and `Q` once."""
    pid_bhnc = tl.program_id(0)
    pid_m_tile = tl.program_id(1)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    m_start = pid_m_tile * BLOCK_M

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_c_bh, stride_c_nc, stride_c_c, stride_c_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    w_desc = tl.make_tensor_descriptor(
        W_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_w_bh, stride_w_nc, stride_w_c, stride_w_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    r_desc = tl.make_tensor_descriptor(
        log_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_r_bh, stride_r_nc, stride_r_c],
        block_shape=[1, 1, BLOCK_C],
    )
    dk_desc = tl.make_tensor_descriptor(
        dK_ptr,
        shape=[bh_size, nc_size, c_size, c_size],
        strides=[stride_dk_bh, stride_dk_nc, stride_dk_c0, stride_dk_c1],
        block_shape=[1, 1, BLOCK_C, BLOCK_C],
    )
    rbuf_desc = tl.make_tensor_descriptor(
        R_ptr,
        shape=[bh_size, nc_size, c_size, c_size],
        strides=[stride_rbuf_bh, stride_rbuf_nc, stride_rbuf_c0, stride_rbuf_c1],
        block_shape=[1, 1, BLOCK_C, BLOCK_C],
    )
    dc_desc = tl.make_tensor_descriptor(
        out_dC_diag_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_dc_bh, stride_dc_nc, stride_dc_c, stride_dc_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    dw_desc = tl.make_tensor_descriptor(
        out_dW_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_dw_bh, stride_dw_nc, stride_dw_c, stride_dw_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    q_desc = tl.make_tensor_descriptor(
        out_Q_ptr,
        shape=[bh_size, nc_size, c_size, c_size],
        strides=[stride_q_bh, stride_q_nc, stride_q_c0, stride_q_c1],
        block_shape=[1, 1, BLOCK_C, BLOCK_C],
    )

    offs_c = tl.arange(0, BLOCK_C)
    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    log_p_incl = tl.cumsum(log_alpha_vals, axis=0)
    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    valid = col_idx <= row_idx
    INV_LN2 = 1.4426950408889634
    log_delta_l2 = (log_p_incl[:, None] - log_p_incl[None, :]) * INV_LN2
    log_delta_l2 = tl.where(valid, log_delta_l2, 0.0)
    L = tl.where(valid, tl.exp2(log_delta_l2), 0.0)

    dK = tl.reshape(dk_desc.load([BH_IDX, NC_IDX, 0, 0]), (BLOCK_C, BLOCK_C)).to(tl.float32)
    dR = dK * L

    W_in = tl.reshape(w_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M))
    C_in = tl.reshape(c_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M))
    W_tile = W_in.to(tl.float32)
    C_tile = C_in.to(tl.float32)
    dC_diag_tile = tl.dot(dR, W_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    dW_tile = tl.dot(tl.trans(dR), C_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    dc_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dC_diag_tile.to(C_in.dtype), (1, 1, BLOCK_C, BLOCK_M)))
    dw_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW_tile.to(C_in.dtype), (1, 1, BLOCK_C, BLOCK_M)))

    if pid_m_tile == 0:
        R = tl.reshape(rbuf_desc.load([BH_IDX, NC_IDX, 0, 0]), (BLOCK_C, BLOCK_C)).to(tl.float32)
        Q = dR * R
        q_desc.store([BH_IDX, NC_IDX, 0, 0], tl.reshape(Q, (1, 1, BLOCK_C, BLOCK_C)))



@triton.jit
def ssd_rank1_dense_output_bwd_fused_a1a2_kernel(
    C_ptr,
    W_ptr,
    V_ptr,
    log_alpha_ptr,
    grad_y_ptr,
    out_dV_ptr,
    out_dC_diag_ptr,
    out_dW_ptr,
    out_dlog_diag_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    d_size,
    stride_c_bh,
    stride_c_nc,
    stride_c_c,
    stride_c_m,
    stride_w_bh,
    stride_w_nc,
    stride_w_c,
    stride_w_m,
    stride_v_bh,
    stride_v_nc,
    stride_v_c,
    stride_v_d,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_gy_bh,
    stride_gy_nc,
    stride_gy_c,
    stride_gy_d,
    stride_dv_bh,
    stride_dv_nc,
    stride_dv_c,
    stride_dv_d,
    stride_dc_bh,
    stride_dc_nc,
    stride_dc_c,
    stride_dc_m,
    stride_dw_bh,
    stride_dw_nc,
    stride_dw_c,
    stride_dw_m,
    stride_dlog_bh,
    stride_dlog_nc,
    stride_dlog_c,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    M_STATIC: tl.constexpr,
    D_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Fused Phase-3 backward A1+A2+logalpha_diag: per (BH,NC) chunk compute dV, dC_diag, dW, dlog_diag.

    Eliminates L recomputation, dK global round-trip, and Q/R_buf global buffers.

    For one chunk:
      1. Build L from log_alpha cumsum
      2. R = C @ W^T (accumulated over M-tiles)
      3. K = L * R
      4. dV = K^T @ dY  (per D-tile, stored immediately)
      5. dK = dY @ V^T  (accumulated over D-tiles in registers)
      6. dR = dK * L
      7. dC_diag = dR @ W  (per M-tile)
      8. dW = dR^T @ C     (per M-tile)
      9. Q = dR * R, then dlog_diag in-register (no global Q/R write)
    """
    pid_bhnc = tl.program_id(0)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_c_bh, stride_c_nc, stride_c_c, stride_c_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    w_desc = tl.make_tensor_descriptor(
        W_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_w_bh, stride_w_nc, stride_w_c, stride_w_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_v_bh, stride_v_nc, stride_v_c, stride_v_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    r_desc = tl.make_tensor_descriptor(
        log_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_r_bh, stride_r_nc, stride_r_c],
        block_shape=[1, 1, BLOCK_C],
    )
    gy_desc = tl.make_tensor_descriptor(
        grad_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_gy_bh, stride_gy_nc, stride_gy_c, stride_gy_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    dv_desc = tl.make_tensor_descriptor(
        out_dV_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_dv_bh, stride_dv_nc, stride_dv_c, stride_dv_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    dc_desc = tl.make_tensor_descriptor(
        out_dC_diag_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_dc_bh, stride_dc_nc, stride_dc_c, stride_dc_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    dw_desc = tl.make_tensor_descriptor(
        out_dW_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_dw_bh, stride_dw_nc, stride_dw_c, stride_dw_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    dlog_desc = tl.make_tensor_descriptor(
        out_dlog_diag_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_dlog_bh, stride_dlog_nc, stride_dlog_c],
        block_shape=[1, 1, BLOCK_C],
    )

    # ---- Step 1: Build L ----
    offs_c = tl.arange(0, BLOCK_C)
    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    log_p_incl = tl.cumsum(log_alpha_vals, axis=0)
    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    valid = col_idx <= row_idx
    INV_LN2 = 1.4426950408889634
    log_delta_l2 = (log_p_incl[:, None] - log_p_incl[None, :]) * INV_LN2
    log_delta_l2 = tl.where(valid, log_delta_l2, 0.0)
    L = tl.where(valid, tl.exp2(log_delta_l2), 0.0)

    # ---- Step 2: R = C @ W^T ----
    R = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        C_blk = tl.reshape(c_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        W_blk = tl.reshape(w_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        R += tl.dot(C_blk, tl.trans(W_blk), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    # ---- Step 3: K = L * R ----
    K = L * R

    # ---- Step 4 & 5: dV per D-tile, accumulate dK in registers ----
    dK = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
        d_start = d_blk * BLOCK_D
        G_in = tl.reshape(gy_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D))
        V_in = tl.reshape(v_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D))
        G = G_in.to(tl.float32)
        V_f = V_in.to(tl.float32)

        dV_tile = tl.dot(tl.trans(K), G, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        dv_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV_tile.to(V_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))

        dK += tl.dot(G, tl.trans(V_f), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    # ---- Step 6: dR = dK * L ----
    dR = dK * L

    # ---- Step 7 & 8: dC_diag and dW per M-tile ----
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        W_in = tl.reshape(w_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M))
        C_in = tl.reshape(c_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M))
        W_tile = W_in.to(tl.float32)
        C_tile = C_in.to(tl.float32)

        dC_diag_tile = tl.dot(dR, W_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        dW_tile = tl.dot(tl.trans(dR), C_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        dc_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dC_diag_tile.to(C_in.dtype), (1, 1, BLOCK_C, BLOCK_M)))
        dw_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW_tile.to(C_in.dtype), (1, 1, BLOCK_C, BLOCK_M)))

    # ---- Step 9: compute dlog_diag from Q = dR * R (in-register, no global write) ----
    Q = dR * R
    left_prefix = tl.cumsum(Q, axis=1)
    left_of = left_prefix - Q
    suffix_rows = tl.flip(tl.cumsum(tl.flip(left_of, 0), axis=0), 0)
    is_diag = offs_c[:, None] == offs_c[None, :]
    dlog = tl.sum(tl.where(is_diag, suffix_rows, 0.0), axis=1)
    dlog_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(dlog, (1, 1, BLOCK_C)))



@triton.jit
def ssd_rank1_dense_output_bwd_fused_off_kernel(
    C_ptr,
    grad_y_ptr,
    log_alpha_ptr,
    S0_ptr,
    out_dC_off_ptr,
    out_dS0_ptr,
    out_dlog_off_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    d_size,
    stride_c_bh,
    stride_c_nc,
    stride_c_c,
    stride_c_m,
    stride_gy_bh,
    stride_gy_nc,
    stride_gy_c,
    stride_gy_d,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_s0_bh,
    stride_s0_nc,
    stride_s0_m,
    stride_s0_d,
    stride_dc_bh,
    stride_dc_nc,
    stride_dc_c,
    stride_dc_m,
    stride_ds0_bh,
    stride_ds0_nc,
    stride_ds0_m,
    stride_ds0_d,
    stride_dlog_bh,
    stride_dlog_nc,
    stride_dlog_c,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    M_STATIC: tl.constexpr,
    D_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Fused off-path backward: per (BH,NC) chunk compute dC_off, dS0, dlog_off.

    Builds log_alpha cumsum once, then computes all three off-path gradients.
    Eliminates two redundant log_alpha cumsum recomputations from split kernels.
    """
    pid_bhnc = tl.program_id(0)
    if pid_bhnc >= bh_size * nc_size:
        return
    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_c_bh, stride_c_nc, stride_c_c, stride_c_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    gy_desc = tl.make_tensor_descriptor(
        grad_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_gy_bh, stride_gy_nc, stride_gy_c, stride_gy_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    r_desc = tl.make_tensor_descriptor(
        log_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_r_bh, stride_r_nc, stride_r_c],
        block_shape=[1, 1, BLOCK_C],
    )
    s0_desc = tl.make_tensor_descriptor(
        S0_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_s0_bh, stride_s0_nc, stride_s0_m, stride_s0_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    dc_desc = tl.make_tensor_descriptor(
        out_dC_off_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_dc_bh, stride_dc_nc, stride_dc_c, stride_dc_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    ds0_desc = tl.make_tensor_descriptor(
        out_dS0_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_ds0_bh, stride_ds0_nc, stride_ds0_m, stride_ds0_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    dlog_desc = tl.make_tensor_descriptor(
        out_dlog_off_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_dlog_bh, stride_dlog_nc, stride_dlog_c],
        block_shape=[1, 1, BLOCK_C],
    )

    # ---- Build prefix products p[i] = exp2(cumsum(log_alpha)) once ----
    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    log_p = tl.cumsum(log_alpha_vals, axis=0)
    INV_LN2 = 1.4426950408889634
    p = tl.exp2(log_p * INV_LN2)

    # ---- dC_off, dS0, and src for dlog_off: iterate over (M-tile, D-tile) ----
    src = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        dC_off = tl.zeros((BLOCK_C, BLOCK_M), dtype=tl.float32)
        C_tile = tl.reshape(c_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
            d_start = d_blk * BLOCK_D
            G_tile = tl.reshape(gy_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
            dB_tile = p[:, None] * G_tile
            S0_tile = tl.reshape(s0_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
            y_off_part = p[:, None] * tl.dot(C_tile, S0_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            src += tl.sum(G_tile * y_off_part, axis=1)
            dC_off += tl.dot(dB_tile, tl.trans(S0_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dS0_tile = tl.dot(tl.trans(C_tile), dB_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            ds0_desc.store([BH_IDX, NC_IDX, m_start, d_start], tl.reshape(dS0_tile, (1, 1, BLOCK_M, BLOCK_D)))

        # Accumulate dC_off into dC_diag (read-modify-write)
        c_probe = tl.reshape(c_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M))
        dc_prev = tl.reshape(dc_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        dc_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape((dc_prev + dC_off).to(c_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))

    # ---- dlog_off from accumulated src ----
    dlog_off = tl.cumsum(src, axis=0, reverse=True)
    out_prev = tl.reshape(dlog_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    dlog_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(out_prev + dlog_off, (1, 1, BLOCK_C)))


@triton.jit
def ssd_rank1_dense_output_bwd_off_dc_kernel(
    C_ptr,
    grad_y_ptr,
    log_alpha_ptr,
    S0_ptr,
    out_dC_off_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    d_size,
    stride_c_bh,
    stride_c_nc,
    stride_c_c,
    stride_c_m,
    stride_gy_bh,
    stride_gy_nc,
    stride_gy_c,
    stride_gy_d,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_s0_bh,
    stride_s0_nc,
    stride_s0_m,
    stride_s0_d,
    stride_dc_bh,
    stride_dc_nc,
    stride_dc_c,
    stride_dc_m,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    D_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Off-path kernel accumulating `dC_off = dB @ S0^T` into `out_dC`."""
    pid_bhnc = tl.program_id(0)
    pid_m_tile = tl.program_id(1)
    if pid_bhnc >= bh_size * nc_size:
        return
    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    m_start = pid_m_tile * BLOCK_M

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_c_bh, stride_c_nc, stride_c_c, stride_c_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    gy_desc = tl.make_tensor_descriptor(
        grad_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_gy_bh, stride_gy_nc, stride_gy_c, stride_gy_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    r_desc = tl.make_tensor_descriptor(
        log_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_r_bh, stride_r_nc, stride_r_c],
        block_shape=[1, 1, BLOCK_C],
    )
    s0_desc = tl.make_tensor_descriptor(
        S0_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_s0_bh, stride_s0_nc, stride_s0_m, stride_s0_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    dc_desc = tl.make_tensor_descriptor(
        out_dC_off_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_dc_bh, stride_dc_nc, stride_dc_c, stride_dc_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )

    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    log_p = tl.cumsum(log_alpha_vals, axis=0)
    INV_LN2 = 1.4426950408889634
    p = tl.exp2(log_p * INV_LN2)

    dC_off = tl.zeros((BLOCK_C, BLOCK_M), dtype=tl.float32)
    for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
        d_start = d_blk * BLOCK_D
        G_tile = tl.reshape(gy_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
        dB_tile = p[:, None] * G_tile
        S0_tile = tl.reshape(s0_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
        dC_off += tl.dot(dB_tile, tl.trans(S0_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    c_probe = tl.reshape(c_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M))
    dc_prev = tl.reshape(dc_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
    dc_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape((dc_prev + dC_off).to(c_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))


@triton.jit
def ssd_rank1_dense_output_bwd_off_ds0_kernel(
    C_ptr,
    grad_y_ptr,
    log_alpha_ptr,
    out_dS0_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    d_size,
    stride_c_bh,
    stride_c_nc,
    stride_c_c,
    stride_c_m,
    stride_gy_bh,
    stride_gy_nc,
    stride_gy_c,
    stride_gy_d,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_ds0_bh,
    stride_ds0_nc,
    stride_ds0_m,
    stride_ds0_d,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Off-path kernel for `dS0 = C^T @ dB`, where `dB = dY * p`."""
    pid_bhnc = tl.program_id(0)
    pid_m_tile = tl.program_id(1)
    pid_d_tile = tl.program_id(2)
    if pid_bhnc >= bh_size * nc_size:
        return
    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    m_start = pid_m_tile * BLOCK_M
    d_start = pid_d_tile * BLOCK_D

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_c_bh, stride_c_nc, stride_c_c, stride_c_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    gy_desc = tl.make_tensor_descriptor(
        grad_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_gy_bh, stride_gy_nc, stride_gy_c, stride_gy_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    r_desc = tl.make_tensor_descriptor(
        log_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_r_bh, stride_r_nc, stride_r_c],
        block_shape=[1, 1, BLOCK_C],
    )
    ds0_desc = tl.make_tensor_descriptor(
        out_dS0_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_ds0_bh, stride_ds0_nc, stride_ds0_m, stride_ds0_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )

    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    log_p = tl.cumsum(log_alpha_vals, axis=0)
    INV_LN2 = 1.4426950408889634
    p = tl.exp2(log_p * INV_LN2)

    C_tile = tl.reshape(c_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
    G_tile = tl.reshape(gy_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
    dB_tile = p[:, None] * G_tile
    dS0_tile = tl.dot(tl.trans(C_tile), dB_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    ds0_desc.store([BH_IDX, NC_IDX, m_start, d_start], tl.reshape(dS0_tile, (1, 1, BLOCK_M, BLOCK_D)))


@triton.jit
def ssd_rank1_dense_output_bwd_logalpha_diag_kernel(
    Q_ptr,
    out_dlog_diag_ptr,
    bh_size,
    nc_size,
    c_size,
    stride_q_bh,
    stride_q_nc,
    stride_q_c0,
    stride_q_c1,
    stride_out_bh,
    stride_out_nc,
    stride_out_c,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
):
    """Compute `dlog_alpha_diag[k] = sum_{i>=k} sum_{j<k} Q[i,j]` in O(C^2).

    Vectorized form:
    1) left_prefix[i,k] = sum_{j<=k} Q[i,j]
    2) left_of[i,k]     = left_prefix[i,k] - Q[i,k] = sum_{j<k} Q[i,j]
    3) suffix_rows[i,k] = sum_{u>=i} left_of[u,k]
    4) dlog[k]          = suffix_rows[k,k]
    """
    pid_bhnc = tl.program_id(0)
    if pid_bhnc >= bh_size * nc_size:
        return
    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size

    q_desc = tl.make_tensor_descriptor(
        Q_ptr,
        shape=[bh_size, nc_size, c_size, c_size],
        strides=[stride_q_bh, stride_q_nc, stride_q_c0, stride_q_c1],
        block_shape=[1, 1, BLOCK_C, BLOCK_C],
    )
    out_desc = tl.make_tensor_descriptor(
        out_dlog_diag_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_out_bh, stride_out_nc, stride_out_c],
        block_shape=[1, 1, BLOCK_C],
    )

    Q = tl.reshape(q_desc.load([BH_IDX, NC_IDX, 0, 0]), (BLOCK_C, BLOCK_C)).to(tl.float32)
    left_prefix = tl.cumsum(Q, axis=1)
    left_of = left_prefix - Q
    suffix_rows = tl.flip(tl.cumsum(tl.flip(left_of, 0), axis=0), 0)
    offs = tl.arange(0, BLOCK_C)
    is_diag = offs[:, None] == offs[None, :]
    dlog = tl.sum(tl.where(is_diag, suffix_rows, 0.0), axis=1)

    out_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(dlog, (1, 1, BLOCK_C)))


@triton.jit
def ssd_rank1_dense_output_bwd_logalpha_off_kernel(
    y_off_ptr,
    grad_y_ptr,
    out_dlog_off_ptr,
    bh_size,
    nc_size,
    c_size,
    d_size,
    stride_yoff_bh,
    stride_yoff_nc,
    stride_yoff_c,
    stride_yoff_d,
    stride_gy_bh,
    stride_gy_nc,
    stride_gy_c,
    stride_gy_d,
    stride_out_bh,
    stride_out_nc,
    stride_out_c,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    D_STATIC: tl.constexpr,
):
    """Compute off-path `dlog_alpha` from cached `Y_off`.

    Since `Y_off[i, d] = p_i * B[i, d]`, the source term is:
      src[i] = sum_d grad_y[i, d] * Y_off[i, d]
    and:
      dlog_alpha[k] = sum_{i>=k} src[i]
    """
    pid_bhnc = tl.program_id(0)
    if pid_bhnc >= bh_size * nc_size:
        return
    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size

    y_off_desc = tl.make_tensor_descriptor(
        y_off_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_yoff_bh, stride_yoff_nc, stride_yoff_c, stride_yoff_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    gy_desc = tl.make_tensor_descriptor(
        grad_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_gy_bh, stride_gy_nc, stride_gy_c, stride_gy_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    out_desc = tl.make_tensor_descriptor(
        out_dlog_off_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_out_bh, stride_out_nc, stride_out_c],
        block_shape=[1, 1, BLOCK_C],
    )

    src = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
        d_start = d_blk * BLOCK_D
        Y_off_tile = tl.reshape(y_off_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
        G_tile = tl.reshape(gy_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
        src += tl.sum(G_tile * Y_off_tile, axis=1)
    dlog = tl.cumsum(src, axis=0, reverse=True)
    out_prev = tl.reshape(out_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    out_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(out_prev + dlog, (1, 1, BLOCK_C)))


# ==================================================================================================
# PHASE 3 AUTOGRAD
# ==================================================================================================

def _ssd_rank1_dense_output_forward_impl(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    S0: torch.Tensor | None = None,
    INPUT_PRECISION: str = "tf32",
    RETURN_Y_OFF: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "_ssd_rank1_dense_output_forward_impl expects C=[BH,NC,C,M], "
            "W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, "
            f"V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    BH, NC, C_CHUNK, M = C.shape
    if W.shape != (BH, NC, C_CHUNK, M):
        raise ValueError(f"W must be [BH,NC,C,M]={BH, NC, C_CHUNK, M}; got {tuple(W.shape)}.")
    if V.shape[:3] != (BH, NC, C_CHUNK):
        raise ValueError(f"V must match [BH,NC,C,*]={BH, NC, C_CHUNK}; got {tuple(V.shape)}.")
    D = V.shape[-1]
    if log_alpha.shape != (BH, NC, C_CHUNK):
        raise ValueError(f"log_alpha must be [BH,NC,C]={BH, NC, C_CHUNK}; got {tuple(log_alpha.shape)}.")
    if C.device != W.device or C.device != V.device or C.device != log_alpha.device:
        raise ValueError("C, W, V, log_alpha must be on the same device.")
    if C.dtype != W.dtype or C.dtype != V.dtype or C.dtype != log_alpha.dtype:
        raise ValueError("C, W, V, log_alpha must share dtype.")
    _require_supported_md(M, D, where="_ssd_rank1_dense_output_forward_impl")
    _require_chunk_size_multiple_of_16(C_CHUNK, where="_ssd_rank1_dense_output_forward_impl")
    _require_nonpositive_log_alpha(log_alpha, where="_ssd_rank1_dense_output_forward_impl")
    if not C.is_cuda:
        raise NotImplementedError("_ssd_rank1_dense_output_forward_impl requires CUDA tensors.")
    _ensure_triton_allocator()

    if S0 is None:
        S0_md = torch.zeros((BH, NC, M, D), device=C.device, dtype=C.dtype)
    else:
        if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
            S0_md = S0.reshape(BH, NC, M, D)
            if S0.dtype != C.dtype:
                S0_md = S0_md.to(C.dtype)
        elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
            S0_md = S0 if S0.dtype == C.dtype else S0.to(C.dtype)
        else:
            raise ValueError(
                f"S0 must be [BH,NC,MD]=({BH},{NC},{M * D}) or [BH,NC,M,D]=({BH},{NC},{M},{D}); "
                f"got {tuple(S0.shape)}."
            )
        if S0_md.device != C.device:
            raise ValueError("S0 must be on the same device as C/W/V/log_alpha.")

    out = torch.empty((BH, NC, C_CHUNK, D), device=C.device, dtype=C.dtype)
    y_off_saved = torch.empty((BH, NC, C_CHUNK, D), device=C.device, dtype=C.dtype) if RETURN_Y_OFF else out
    phase3_fwd_cfg = _select_phase3_forward_launch_config(
        BH=BH,
        NC=NC,
        C_CHUNK=C_CHUNK,
        M=M,
        D=D,
        where="_ssd_rank1_dense_output_forward_impl",
    )
    grid = (BH * NC, D // phase3_fwd_cfg.block_d)
    ssd_rank1_dense_output_fwd_kernel[grid](
        C,
        W,
        V,
        log_alpha,
        S0_md,
        out,
        y_off_saved,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *C.stride(),
        *W.stride(),
        *V.stride(),
        *log_alpha.stride(),
        *S0_md.stride(),
        *out.stride(),
        *y_off_saved.stride(),
        BLOCK_M=phase3_fwd_cfg.block_m,
        BLOCK_D=phase3_fwd_cfg.block_d,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        M_STATIC=M,
        INPUT_PRECISION=INPUT_PRECISION,
        WRITE_Y_OFF=RETURN_Y_OFF,
        num_warps=phase3_fwd_cfg.num_warps,
        num_stages=phase3_fwd_cfg.num_stages,
    )
    if RETURN_Y_OFF:
        return out, y_off_saved
    return out


def _ssd_rank1_dense_output_backward_impl(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    grad_out: torch.Tensor | None,
    *,
    S0_saved: torch.Tensor,
    input_precision: str,
    s0_was_flat: bool,
    return_s0_grad_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Phase-3 backward for chunk-local dense replay/output.

    Per `(BH, NC)` forward equations:
      R = C @ W^T
      L[i,j] = 1_{j<=i} * prod_{u=j+1..i} alpha_u, alpha_u = exp(log_alpha_u)
      K = L * R
      Y_diag = K @ V
      Y_off  = p * (C @ S0), p_i = prod_{u=0..i} alpha_u
      Y = Y_diag + Y_off

    Backward decomposition used here:
      1) Fused dense-path kernel computes `dV`, `dC_diag`, `dW`, `dlog_diag`.
      2) Off-path kernels add the `Y_off` contributions:
         - `dC_off` is accumulated in-place into `dC_diag`
         - `dS0`
         - `dlog_off` accumulated in-place into `dlog_diag`

    Returned tensors are phase-3 contributions only:
      - `dC`, `dW`, `dV`, `dlog_alpha`
      - `dS0` (used by phase-2 backward), optionally kept in fp32.
    """
    BH, NC, C_CHUNK, M = C.shape
    D = V.shape[-1]
    _require_supported_md(M, D, where="_ssd_rank1_dense_output_backward_impl")

    if grad_out is None:
        grad_out = torch.zeros((BH, NC, C_CHUNK, D), device=C.device, dtype=C.dtype)
    if grad_out.shape != (BH, NC, C_CHUNK, D):
        raise ValueError(
            f"grad_out must be [BH,NC,C,D]=({BH},{NC},{C_CHUNK},{D}); got {tuple(grad_out.shape)}."
        )
    _require_chunk_size_multiple_of_16(C_CHUNK, where="_ssd_rank1_dense_output_backward_impl")
    if not C.is_cuda:
        raise NotImplementedError("_ssd_rank1_dense_output_backward_impl requires CUDA tensors.")
    _ensure_triton_allocator()

    if S0_saved.ndim == 3 and S0_saved.shape == (BH, NC, M * D):
        S0_md = S0_saved.reshape(BH, NC, M, D)
    elif S0_saved.ndim == 4 and S0_saved.shape == (BH, NC, M, D):
        S0_md = S0_saved
    else:
        raise ValueError(
            f"S0_saved must be [BH,NC,MD]=({BH},{NC},{M * D}) or [BH,NC,M,D]=({BH},{NC},{M},{D}); "
            f"got {tuple(S0_saved.shape)}."
        )

    phase3_bwd_cfg = _select_phase3_backward_launch_config(
        BH=BH,
        NC=NC,
        C_CHUNK=C_CHUNK,
        M=M,
        D=D,
    )
    BLOCK_M = phase3_bwd_cfg.block_m
    BLOCK_D = phase3_bwd_cfg.block_d
    if M % BLOCK_M != 0 or D % BLOCK_D != 0:
        raise NotImplementedError(
            f"_ssd_rank1_dense_output_backward_impl requires M%BLOCK_M==0 and D%BLOCK_D==0; "
            f"got M={M}, D={D}, BLOCK_M={BLOCK_M}, BLOCK_D={BLOCK_D}."
        )
    dV = torch.empty_like(V)
    dC_diag = torch.empty_like(C)
    dW = torch.empty_like(W)
    dlog_diag = _ssd_rank1_bwd_workspace_tensor(
        "dlog_diag",
        (BH, NC, C_CHUNK),
        device=C.device,
        dtype=torch.float32,
    )
    grad_out_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()

    grid_fused = (BH * NC,)
    # Main dense replay/output path:
    #   dV = K^T dY, dR = dK * L, dC_diag = dR W, dW = dR^T C, dlog_diag from Q=dK*R*L
    ssd_rank1_dense_output_bwd_fused_a1a2_kernel[grid_fused](
        C,
        W,
        V,
        log_alpha,
        grad_out_c,
        dV,
        dC_diag,
        dW,
        dlog_diag,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *C.stride(),
        *W.stride(),
        *V.stride(),
        *log_alpha.stride(),
        *grad_out_c.stride(),
        *dV.stride(),
        *dC_diag.stride(),
        *dW.stride(),
        *dlog_diag.stride(),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        M_STATIC=M,
        D_STATIC=D,
        INPUT_PRECISION=input_precision,
        num_warps=phase3_bwd_cfg.fused_a1a2_num_warps,
        num_stages=phase3_bwd_cfg.num_stages,
    )

    has_s0 = S0_md.numel() > 0
    if has_s0:
        dS0 = torch.empty((BH, NC, M, D), device=C.device, dtype=torch.float32)
        grid_off = (BH * NC,)
        # Fused off-path kernel computes:
        #   - dC_off (accumulated into dC_diag)
        #   - dS0
        #   - dlog_off (accumulated into dlog_diag)
        ssd_rank1_dense_output_bwd_fused_off_kernel[grid_off](
            C,
            grad_out_c,
            log_alpha,
            S0_md,
            dC_diag,
            dS0,
            dlog_diag,
            BH,
            NC,
            C_CHUNK,
            M,
            D,
            *C.stride(),
            *grad_out_c.stride(),
            *log_alpha.stride(),
            *S0_md.stride(),
            *dC_diag.stride(),
            *dS0.stride(),
            *dlog_diag.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=C_CHUNK,
            M_STATIC=M,
            D_STATIC=D,
            INPUT_PRECISION=input_precision,
            num_warps=phase3_bwd_cfg.fused_off_num_warps,
            num_stages=phase3_bwd_cfg.num_stages,
        )

        dC = dC_diag
        dlog_alpha = dlog_diag
        dS0_out = dS0.reshape(BH, NC, M * D) if s0_was_flat else dS0
        if not return_s0_grad_fp32:
            dS0_out = dS0_out.to(C.dtype)
    else:
        dC = dC_diag
        dlog_alpha = dlog_diag
        dS0_out = None

    return dC, dW, dV, dlog_alpha.to(log_alpha.dtype), dS0_out


def ssd_rank1_dense_output_backward_kernel_profile(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    grad_out: torch.Tensor,
    S0: torch.Tensor | None = None,
    INPUT_PRECISION: str = "tf32",
    BLOCK_M: int | None = None,
    BLOCK_D: int | None = None,
    FUSED_A1A2_NUM_WARPS: int | None = None,
    OFF_NUM_WARPS: int | None = None,
    NUM_STAGES: int | None = None,
    WARMUP: bool = True,
) -> dict[str, float]:
    """Profile split Phase-3 Triton backward kernels and return per-stage timings (ms).

    This mirrors the phase-3 backward path without autograd plumbing so kernel-level
    bottlenecks can be tuned quickly.
    """
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3 or grad_out.ndim != 4:
        raise ValueError(
            "ssd_rank1_dense_output_backward_kernel_profile expects "
            "C=[BH,NC,C,M], W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C], grad_out=[BH,NC,C,D]."
        )
    BH, NC, C_CHUNK, M = C.shape
    if W.shape != (BH, NC, C_CHUNK, M):
        raise ValueError("W shape mismatch.")
    if V.shape[:3] != (BH, NC, C_CHUNK):
        raise ValueError("V shape mismatch.")
    D = V.shape[-1]
    if log_alpha.shape != (BH, NC, C_CHUNK):
        raise ValueError("log_alpha shape mismatch.")
    if grad_out.shape != (BH, NC, C_CHUNK, D):
        raise ValueError("grad_out shape mismatch.")
    _require_supported_md(M, D, where="ssd_rank1_dense_output_backward_kernel_profile")
    _require_chunk_size_multiple_of_16(C_CHUNK, where="ssd_rank1_dense_output_backward_kernel_profile")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank1_dense_output_backward_kernel_profile")
    if not C.is_cuda:
        raise NotImplementedError("ssd_rank1_dense_output_backward_kernel_profile requires CUDA tensors.")
    _ensure_triton_allocator()

    if S0 is None:
        S0_md = torch.zeros((BH, NC, M, D), device=C.device, dtype=C.dtype)
        has_s0 = False
    else:
        if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
            S0_md = S0.reshape(BH, NC, M, D).to(C.dtype)
        elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
            S0_md = S0.to(C.dtype)
        else:
            raise ValueError("S0 shape mismatch.")
        has_s0 = True

    default_cfg = _select_phase3_backward_launch_config(
        BH=BH,
        NC=NC,
        C_CHUNK=C_CHUNK,
        M=M,
        D=D,
    )
    BLOCK_M = default_cfg.block_m if BLOCK_M is None else BLOCK_M
    BLOCK_D = default_cfg.block_d if BLOCK_D is None else BLOCK_D
    FUSED_A1A2_NUM_WARPS = (
        default_cfg.fused_a1a2_num_warps if FUSED_A1A2_NUM_WARPS is None else FUSED_A1A2_NUM_WARPS
    )
    OFF_NUM_WARPS = default_cfg.fused_off_num_warps if OFF_NUM_WARPS is None else OFF_NUM_WARPS
    NUM_STAGES = default_cfg.num_stages if NUM_STAGES is None else NUM_STAGES
    if M % BLOCK_M != 0 or D % BLOCK_D != 0:
        raise NotImplementedError(
            f"Profile currently requires M%BLOCK_M==0 and D%BLOCK_D==0; got M={M}, D={D}, BLOCK_M={BLOCK_M}, BLOCK_D={BLOCK_D}."
        )

    grad_out_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()

    times: dict[str, float] = {}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def _time(label: str, fn) -> None:
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times[label] = float(start.elapsed_time(end))

    dV = torch.empty_like(V)
    dC_diag = torch.empty_like(C)
    dW = torch.empty_like(W)
    dlog_diag = torch.empty((BH, NC, C_CHUNK), device=C.device, dtype=torch.float32)

    def _run_fused():
        grid_fused = (BH * NC,)
        ssd_rank1_dense_output_bwd_fused_a1a2_kernel[grid_fused](
            C,
            W,
            V,
            log_alpha,
            grad_out_c,
            dV,
            dC_diag,
            dW,
            dlog_diag,
            BH,
            NC,
            C_CHUNK,
            M,
            D,
            *C.stride(),
            *W.stride(),
            *V.stride(),
            *log_alpha.stride(),
            *grad_out_c.stride(),
            *dV.stride(),
            *dC_diag.stride(),
            *dW.stride(),
            *dlog_diag.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=C_CHUNK,
            C_STATIC=C_CHUNK,
            M_STATIC=M,
            D_STATIC=D,
            INPUT_PRECISION=INPUT_PRECISION,
            num_warps=FUSED_A1A2_NUM_WARPS,
            num_stages=NUM_STAGES,
        )

    if WARMUP:
        _run_fused()
        torch.cuda.synchronize()
    _time("fused_a1a2_ms", _run_fused)
    times["a1_ms"] = 0.0
    times["dk_reduce_ms"] = 0.0
    times["a2_ms"] = 0.0
    times["log_diag_ms"] = 0.0

    if has_s0:
        dS0 = torch.empty((BH, NC, M, D), device=C.device, dtype=torch.float32)
        def _run_fused_off():
            grid_off = (BH * NC,)
            ssd_rank1_dense_output_bwd_fused_off_kernel[grid_off](
                C,
                grad_out_c,
                log_alpha,
                S0_md,
                dC_diag,
                dS0,
                dlog_diag,
                BH,
                NC,
                C_CHUNK,
                M,
                D,
                *C.stride(),
                *grad_out_c.stride(),
                *log_alpha.stride(),
                *S0_md.stride(),
                *dC_diag.stride(),
                *dS0.stride(),
                *dlog_diag.stride(),
                BLOCK_M=BLOCK_M,
                BLOCK_D=BLOCK_D,
                BLOCK_C=C_CHUNK,
                M_STATIC=M,
                D_STATIC=D,
                INPUT_PRECISION=INPUT_PRECISION,
                num_warps=OFF_NUM_WARPS,
                num_stages=NUM_STAGES,
            )

        if WARMUP:
            _run_fused_off()
            torch.cuda.synchronize()
        _time("fused_off_ms", _run_fused_off)
        times["off_dc_ms"] = 0.0
        times["off_ds0_ms"] = 0.0
        times["log_off_ms"] = 0.0

        times["combine_ms"] = 0.0
    else:
        times["fused_off_ms"] = 0.0
        times["off_dc_ms"] = 0.0
        times["off_ds0_ms"] = 0.0
        times["log_off_ms"] = 0.0
        times["combine_ms"] = 0.0

    times["total_ms"] = (
        times.get("fused_a1a2_ms", times["a1_ms"] + times["dk_reduce_ms"] + times["a2_ms"])
        + times["log_diag_ms"]
        + times.get("fused_off_ms", 0.0)
        + times["off_dc_ms"]
        + times["off_ds0_ms"]
        + times["log_off_ms"]
        + times["combine_ms"]
    )
    times["BLOCK_M"] = float(BLOCK_M)
    times["BLOCK_D"] = float(BLOCK_D)
    times["FUSED_A1A2_NUM_WARPS"] = float(FUSED_A1A2_NUM_WARPS)
    times["OFF_NUM_WARPS"] = float(OFF_NUM_WARPS)

    return times




# --------------------------------------------------------------------------------------------------
# END PHASE 3: DENSE OUTPUT COMPUTATION
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# PHASE 123: FULL PARALLEL SCAN (UNIFIED AUTOGRAD)
# --------------------------------------------------------------------------------------------------

class SsdRank1Triton(torch.autograd.Function):
    """Unified SSD rank-1 Triton autograd function (Phases 1+2+3).

    High-level dataflow:
      1) Forward chunks [B,N,H,*] -> [BH,NC,C,*], then runs Phase 1/2/3 kernels.
      2) Forward returns:
           - y_chunk: chunk-local outputs [BH,NC,C,D]
           - S1_chunk: final recurrent state per BH lane [BH,MD]
      3) Backward recomputes the minimum needed intermediates (Phase 1 + 2 forward)
         and then executes explicit Phase 3 -> Phase 2 -> Phase 1 backward kernels.

    Why unified:
      - We avoid a stack of nested autograd.Functions and keep one explicit backward.
      - We can choose exactly which tensors are retained versus recomputed.
      - We keep memory pressure manageable by not saving large fp32 intermediates
        that are cheap to regenerate.
    """

    @staticmethod
    def forward(
        ctx,
        C: torch.Tensor,
        W: torch.Tensor,
        V: torch.Tensor,
        log_alpha: torch.Tensor,
        initial_state: torch.Tensor | None,
        CHUNK_SIZE: int,
        INPUT_PRECISION: str,
    ):
        # Step 0: validate contract and convert to kernel-friendly chunked layout.
        # Input  layout: C/W [B,N,H,M], V [B,N,H,D], log_alpha [B,N,H]
        # Kernel layout: C/W [BH,NC,C,M], V [BH,NC,C,D], log_alpha [BH,NC,C]
        # where BH=B*H, NC=#chunks, C=CHUNK_SIZE.
        C_chunk, W_chunk, V_chunk, log_alpha_chunk, init_flat, B, N, H, M, D, NC_exec = _ssd_rank1_prepare_unchunked_inputs(
            C, W, V, log_alpha, initial_state,
            where="SsdRank1Triton.forward",
            CHUNK_SIZE=CHUNK_SIZE,
        )

        # =========================================
        # FORWARD PHASE 1: CHUNK LOCAL END-STATE
        # =========================================
        # Per-chunk local end-state summaries.
        # Produces S_local_end [BH,NC,MD] in fp32-like accumulate domain.
        S_local_end = _ssd_rank1_chunk_end_state_forward_impl(W_chunk, V_chunk, log_alpha_chunk, 32)

        # =========================================
        # FORWARD PHASE 2: PREFIX SCAN OVER CHUNKS
        # =========================================
        # Prefix-scan local chunk summaries to produce
        # chunk-start state S0 per chunk and final state S1 at end of sequence.
        log_alpha_per_chunk = torch.sum(log_alpha_chunk.float(), dim=-1)
        S0_chunk, S1_chunk = _ssd_rank1_prefix_scan_forward_impl(
            S_local_end, log_alpha_per_chunk, init_flat, V_chunk.dtype,
        )

        # =========================================
        # FORWARD PHASE 3: DENSE CHUNK OUTPUT
        # =========================================
        # Dense intra-chunk output.
        y_chunk = _ssd_rank1_dense_output_forward_impl(
            C_chunk, W_chunk, V_chunk, log_alpha_chunk, S0_chunk, INPUT_PRECISION,
            RETURN_Y_OFF=False,
        )

        # Saved tensors policy:
        # - Save user-facing unchunked tensors and replay chunking in backward.
        # - Do not save S_local_end or S0_chunk (recompute them in backward).
        # This keeps memory lower while preserving fast backward paths.
        if initial_state is None:
            ctx.save_for_backward(C, W, V, log_alpha)
            ctx.has_initial_state = False
        else:
            ctx.save_for_backward(C, W, V, log_alpha, initial_state)
            ctx.has_initial_state = True
        ctx.CHUNK_SIZE = CHUNK_SIZE
        ctx.INPUT_PRECISION = INPUT_PRECISION
        ctx.initial_state_shape = None if initial_state is None else tuple(initial_state.shape)

        # Return chunked outputs; outer wrapper restores [B,N,H,*] layout.
        return y_chunk, S1_chunk

    @staticmethod
    def backward(ctx, grad_y_chunk, grad_S1_chunk):
        """Backward pass for unified phase-1/2/3 SSD rank-1.

        Input adjoints:
          - grad_y_chunk  = dL/dY_chunk  [BH,NC,C,D]
          - grad_S1_chunk = dL/dS_final  [BH,MD] (optional)

        Chain-rule schedule:
          1) Replay phase-1 and phase-2 forwards to recover `S_local_end`, `S0_chunk`.
          2) Phase-3 backward:
               (C, W, V, log_alpha, S0) + dY
                 -> dC_p3, dW_p3, dV_p3, dlog_p3, dS0
          3) Phase-2 backward:
               (S_local_end, log_alpha_per_chunk, init) + (dS0, dS_final)
                 -> dS_local_end, dlog_per_chunk, dinit
          4) Phase-1 backward:
               (W, V, log_alpha) + dS_local_end
                 -> dW_p1, dV_p1, dlog_p1
          5) Accumulate and reshape:
               dW = dW_p3 + dW_p1
               dV = dV_p3 + dV_p1
               dlog = dlog_p3 + dlog_p1 + broadcast(dlog_per_chunk over chunk tokens)
        """
        # Upstream gradients:
        #   grad_y_chunk  : dL/dy_chunk  [BH,NC,C,D]
        #   grad_S1_chunk : dL/dS1_chunk [BH,MD] or None
        # We compute gradients for C/W/V/log_alpha/initial_state and None for
        # non-tensor args (CHUNK_SIZE, INPUT_PRECISION).
        if ctx.has_initial_state:
            C, W, V, log_alpha, initial_state = ctx.saved_tensors
        else:
            C, W, V, log_alpha = ctx.saved_tensors
            initial_state = None
        CHUNK_SIZE = ctx.CHUNK_SIZE
        INPUT_PRECISION = ctx.INPUT_PRECISION
        C_chunk, W_chunk, V_chunk, log_alpha_chunk, init_flat, B, N, H, M, D, NC_exec = _ssd_rank1_prepare_unchunked_inputs(
            C, W, V, log_alpha, initial_state,
            where="SsdRank1Triton.backward",
            CHUNK_SIZE=CHUNK_SIZE,
        )
        input_dtype = V_chunk.dtype
        compute_dtype = V_chunk.dtype
        initial_state_shape = ctx.initial_state_shape
        BH = B * H
        NC = NC_exec
        MD = M * D

        # =========================================
        # BACKWARD REPLAY: RECOMPUTE FORWARD INTERMEDIATES
        # =========================================
        # Phase 1/2 forward replay for full-sequence backward.
        S_local_end = _ssd_rank1_chunk_end_state_forward_impl(W_chunk, V_chunk, log_alpha_chunk, 32)
        log_alpha_per_chunk = torch.sum(log_alpha_chunk.float(), dim=-1)
        S0_chunk, _ = _ssd_rank1_prefix_scan_forward_impl(
            S_local_end, log_alpha_per_chunk, init_flat, compute_dtype,
        )
        del S_local_end

        # Phase-3 backward on full [BH,NC,C,*] tensors.
        dC_chunk, dW_chunk, dV_chunk, dlog_chunk, dS0 = _ssd_rank1_dense_output_backward_impl(
            C_chunk,
            W_chunk,
            V_chunk,
            log_alpha_chunk,
            grad_y_chunk,
            S0_saved=S0_chunk,
            input_precision=INPUT_PRECISION,
            s0_was_flat=True,
            return_s0_grad_fp32=True,
        )
        del grad_y_chunk, C_chunk

        # Phase-2 backward on full [BH,NC,MD] tensors.
        grad_final_for_p2 = (
            grad_S1_chunk
            if grad_S1_chunk is not None
            else torch.zeros((BH, MD), device=V_chunk.device, dtype=torch.float32)
        )
        grad_chunk_start_f = dS0 if dS0.dtype == torch.float32 and dS0.is_contiguous() else dS0.float().contiguous()
        grad_final_f = (
            grad_final_for_p2
            if grad_final_for_p2.dtype == torch.float32 and grad_final_for_p2.is_contiguous()
            else grad_final_for_p2.float().contiguous()
        )
        s0_chunk_f = S0_chunk if S0_chunk.dtype == torch.float32 and S0_chunk.is_contiguous() else S0_chunk.float().contiguous()
        log_alpha_per_chunk_f = (
            log_alpha_per_chunk
            if log_alpha_per_chunk.dtype == torch.float32 and log_alpha_per_chunk.is_contiguous()
            else log_alpha_per_chunk.float().contiguous()
        )
        # Reuse dS0 buffer for dS_local_end to avoid an extra full [BH,NC,MD] allocation.
        # The phase-2 kernel loads grad_prefix before writing d_s_local for each tile.
        dS_local_end = grad_chunk_start_f
        d_log_per_chunk = _ssd_rank1_bwd_workspace_tensor(
            "phase2_dlog_per_chunk_full",
            (BH, NC),
            device=V_chunk.device,
            dtype=torch.float32,
        )
        d_init_f32 = _ssd_rank1_bwd_workspace_tensor(
            "phase2_dinit_full",
            (BH, MD),
            device=V_chunk.device,
            dtype=torch.float32,
        )
        block_nc = _select_phase2_block_nc(NC=NC)
        grid_bwd = lambda META: (BH,)
        ssd_rank1_prefix_scan_bwd_dense_kernel[grid_bwd](
            grad_chunk_start_f,
            grad_final_f,
            s0_chunk_f,
            log_alpha_per_chunk_f,
            dS_local_end,
            d_log_per_chunk,
            d_init_f32,
            BH,
            NC,
            MD,
            *grad_chunk_start_f.stride(),
            *grad_final_f.stride(),
            *s0_chunk_f.stride(),
            *log_alpha_per_chunk_f.stride(),
            *dS_local_end.stride(),
            *d_log_per_chunk.stride(),
            *d_init_f32.stride(),
            BLOCK_NC=block_nc,
            NC_STATIC=NC,
            USE_FP32_COMPUTE=(compute_dtype == torch.float32),
            USE_BF16_COMPUTE=(compute_dtype == torch.bfloat16),
        )
        del S0_chunk, dS0, grad_final_f, s0_chunk_f, log_alpha_per_chunk_f, log_alpha_per_chunk

        # Phase-1 backward on full [BH,NC,MD], accumulating into phase-3 grads.
        BLOCK_M = _select_largest_block_size(
            M,
            _SUPPORTED_BLOCK_X_VALUES,
            where="SsdRank1Triton.backward",
            label="M",
        )
        BLOCK_D = _select_largest_block_size(
            D,
            _SUPPORTED_BLOCK_X_VALUES,
            where="SsdRank1Triton.backward",
            label="D",
        )
        n_m_tiles = M // BLOCK_M
        n_d_tiles = D // BLOCK_D
        phase1_bwd_cfg = _select_phase1_backward_launch_config(M=M, D=D)

        grad_s_md = dS_local_end.reshape(BH, NC, M, D)
        dlog_p1 = torch.empty_like(log_alpha_chunk)
        grid_dw = (BH * NC, n_m_tiles)
        ssd_rank1_chunk_end_state_bwd_dw_chunk_kernel[grid_dw](
            grad_s_md,
            V_chunk,
            log_alpha_chunk,
            dW_chunk,
            BH,
            NC,
            CHUNK_SIZE,
            M,
            D,
            *grad_s_md.stride(),
            *V_chunk.stride(),
            *log_alpha_chunk.stride(),
            *dW_chunk.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=CHUNK_SIZE,
            C_STATIC=CHUNK_SIZE,
            D_STATIC=D,
            INPUT_PRECISION="ieee" if input_dtype == torch.float32 else "tf32",
            ACCUMULATE=True,
            num_warps=phase1_bwd_cfg.num_warps,
            num_stages=phase1_bwd_cfg.num_stages,
        )

        b_vals = _ssd_rank1_bwd_workspace_tensor(
            "phase1_b_vals_full",
            (BH, NC, CHUNK_SIZE),
            device=V_chunk.device,
            dtype=torch.float32,
        )
        b_vals.zero_()
        grid_dv = (BH * NC, n_d_tiles)
        ssd_rank1_chunk_end_state_bwd_dv_b_chunk_kernel[grid_dv](
            grad_s_md,
            W_chunk,
            V_chunk,
            log_alpha_chunk,
            dV_chunk,
            b_vals,
            BH,
            NC,
            CHUNK_SIZE,
            M,
            D,
            n_d_tiles,
            *grad_s_md.stride(),
            *W_chunk.stride(),
            *V_chunk.stride(),
            *log_alpha_chunk.stride(),
            *dV_chunk.stride(),
            *b_vals.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=CHUNK_SIZE,
            C_STATIC=CHUNK_SIZE,
            M_STATIC=M,
            INPUT_PRECISION="ieee" if input_dtype == torch.float32 else "tf32",
            ACCUMULATE=True,
            num_warps=phase1_bwd_cfg.num_warps,
            num_stages=phase1_bwd_cfg.num_stages,
        )
        grid_dr = (BH * NC,)
        ssd_rank1_chunk_end_state_bwd_dr_kernel[grid_dr](
            b_vals,
            log_alpha_chunk,
            dlog_p1,
            BH,
            NC,
            CHUNK_SIZE,
            *b_vals.stride(),
            *log_alpha_chunk.stride(),
            *dlog_p1.stride(),
            BLOCK_C=CHUNK_SIZE,
            C_STATIC=CHUNK_SIZE,
            num_warps=phase1_bwd_cfg.num_warps,
            num_stages=phase1_bwd_cfg.num_stages,
        )

        dlog_chunk.add_(dlog_p1)
        dlog_chunk.add_(d_log_per_chunk.unsqueeze(-1).to(log_alpha_chunk.dtype))
        del grad_chunk_start_f, dS_local_end, b_vals, W_chunk, V_chunk, log_alpha_chunk

        # Restore initial_state gradient in the same shape family the caller used.
        if initial_state_shape is not None:
            if len(initial_state_shape) == 4:
                d_init = d_init_f32.reshape(B, H, M, D).to(input_dtype)
            else:
                d_init = d_init_f32.reshape(B, H, M * D).to(input_dtype)
        else:
            d_init = None

        dC = _ssd_rank1_restore_grad_layout(dC_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        dW = _ssd_rank1_restore_grad_layout(dW_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        dV = _ssd_rank1_restore_grad_layout(dV_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        nc_data = triton.cdiv(N, CHUNK_SIZE)
        dlog_alpha = (
            dlog_chunk[:, :nc_data, :]
            .reshape(B, H, nc_data, CHUNK_SIZE)
            .permute(0, 2, 3, 1)
            .reshape(B, nc_data * CHUNK_SIZE, H)
        )[:, :N, :]

        # Return one gradient slot per forward argument:
        # (C, W, V, log_alpha, initial_state, CHUNK_SIZE, INPUT_PRECISION)
        return dC, dW, dV, dlog_alpha, d_init, None, None


def _ssd_rank1_restore_grad_layout(
    grad_chunk: torch.Tensor,
    *,
    B: int,
    N: int,
    H: int,
    C: int,
) -> torch.Tensor:
    """Map [BH, NC, C, *] gradients back to [B, N, H, *]."""
    trailing = grad_chunk.shape[3:]
    g = grad_chunk.reshape(B, H, grad_chunk.shape[1], C, *trailing)
    g = g.permute(0, 2, 3, 1, 4)
    g = g.reshape(B, grad_chunk.shape[1] * C, H, *trailing)
    return g[:, :N]

def ssd_rank1_triton(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
    INPUT_PRECISION: str = "tf32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """SSD rank-1 Triton entrypoint using unified autograd with recomputation."""
    if CHUNK_SIZE is None:
        B0, N0, H0, M0 = C.shape
        D0 = V.shape[-1]
        CHUNK_SIZE = _select_chunk_size_heuristic(N=N0, M=M0, D=D0, BH=B0 * H0)

    y_chunk, S1_chunk = SsdRank1Triton.apply(
        C, W, V, log_alpha, initial_state, CHUNK_SIZE, INPUT_PRECISION,
    )

    B = C.shape[0]
    N = C.shape[1]
    H = C.shape[2]
    return _ssd_rank1_restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)


# --------------------------------------------------------------------------------------------------
# END PHASE 123: FULL PARALLEL SCAN
# --------------------------------------------------------------------------------------------------


__all__ = [
    "ssd_rank1_pytorch",
    "ssd_rank1_triton",
]

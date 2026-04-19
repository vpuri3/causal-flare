"""Triton path for SSD rank-1 autoregressive chunk-prefix state scans.

This file currently implements forward for mode-0 chunk-prefix propagation and
keeps backward as an explicit TODO.
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl

_SUPPORTED_M_VALUES = {16, 32, 64, 128}
_SUPPORTED_D_VALUES = {32, 64, 128}
_SUPPORTED_PHASE1_BLOCK_T_VALUES = {16, 32, 64}
_EXPERIMENTAL_TRITON_ALLOCATOR_SET = False
_INV_LN2 = 1.4426950408889634

# Phase-3 backward tuning knobs (C is compile-time; tune M/D tiles and launch).
_PHASE3_BWD_BLOCK_M = 64
_PHASE3_BWD_BLOCK_D = 64
_PHASE3_BWD_A1_NUM_WARPS = 8
_PHASE3_BWD_A2_NUM_WARPS = 8
_PHASE3_BWD_OFF_NUM_WARPS = 8
_PHASE3_BWD_DIAG_NUM_WARPS = 2
_PHASE3_BWD_NUM_STAGES = 2

# Reusable buffers for Phase-3 backward intermediates to reduce allocation overhead.
_PHASE3_BWD_WORKSPACE: dict[tuple, torch.Tensor] = {}


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
        CHUNK_SIZE = 64

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
    alpha = torch.exp2(log_alpha_f * _INV_LN2)
    W_f = W.float()
    V_f = V.float()

    # Recurrence inside each chunk on matrix state S_t in [M, D]:
    # S_{t+1} = alpha_t * S_t + W_t[:, None] * V_t[None, :]
    # where alpha_t = exp(log_alpha_t).
    S = torch.zeros((BH, NC, M, D), device=W.device, dtype=torch.float32)
    for t in range(C):
        S = (
            alpha[:, :, t].unsqueeze(-1).unsqueeze(-1) * S
            + W_f[:, :, t, :].unsqueeze(-1) * V_f[:, :, t, :].unsqueeze(-2)
        )
    S_local_end = S.reshape(BH, NC, M * D)
    return S_local_end


# --------------------------------------------------------------------------------------------------
# END PHASE 1: LOCAL CHUNK END-STATE COMPUTATION
# --------------------------------------------------------------------------------------------------




# --------------------------------------------------------------------------------------------------
# PHASE 2: PREFIX SCAN OVER CHUNKS
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
    L = torch.where(
        tril,
        torch.exp2((log_p[..., :, None] - log_p[..., None, :]) * _INV_LN2),
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
    L = torch.where(
        tril,
        torch.exp2((log_p[..., :, None] - log_p[..., None, :]) * _INV_LN2),
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


# --------------------------------------------------------------------------------------------------
# END PHASE 3: DENSE OUTPUT COMPUTATION
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# PHASE 123: FULL PARALLEL SCAN
# --------------------------------------------------------------------------------------------------
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
    _require_chunk_size_multiple_of_16(CHUNK_SIZE, where=where)
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


# --------------------------------------------------------------------------------------------------
# END PHASE 123: FULL PARALLEL SCAN
# --------------------------------------------------------------------------------------------------


# ==================================================================================================
# TRITON KERNELS + AUTOGRAD
# ==================================================================================================
# --------------------------------------------------------------------------------------------------
# PHASE 1: LOCAL CHUNK END-STATE COMPUTATION
# --------------------------------------------------------------------------------------------------
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
    out_dw_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW_tile.to(w_dtype_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))


@triton.jit
def ssd_rank1_chunk_end_state_bwd_dv_b_chunk_kernel(
    grad_s_local_end_ptr,
    W_ptr,
    V_ptr,
    log_alpha_ptr,
    out_dv_ptr,
    out_b_partials_ptr,
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
    stride_out_b_dt,
    stride_out_b_c,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    M_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Phase-1 backward kernel for `dV` + `b_t` partials with chunk-owned programs.

    Program mapping:
    - `pid0`: flattened `(BH, NC)` chunk index
    - `pid1`: D tile

    Computes:
      dv_base[t, d] = sum_m G[m, d] * W[t, m]
      dV[t, d]      = factor[t] * dv_base[t, d]
      b_partial[t]  = sum_{d in tile} dv_base[t, d] * V[t, d]
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
    out_b_desc = tl.make_tensor_descriptor(
        out_b_partials_ptr,
        shape=[bh_size, nc_size, n_d_tiles_size, c_size],
        strides=[stride_out_b_bh, stride_out_b_nc, stride_out_b_dt, stride_out_b_c],
        block_shape=[1, 1, 1, BLOCK_C],
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
    out_dv_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV_tile.to(v_dtype_probe.dtype), (1, 1, BLOCK_C, BLOCK_D)))

    v_tile = tl.reshape(v_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
    b_partial = tl.sum(acc * v_tile, axis=1)
    out_b_desc.store([BH_IDX, NC_IDX, pid_d_tile, 0], tl.reshape(b_partial, (1, 1, 1, BLOCK_C)))


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
class SsdRank1ChunkEndStateTriton(torch.autograd.Function):
    """Autograd contour for Phase-1 chunk summaries.

    Inputs:
    - `W`: `[BH, NC, C, M]`
    - `V`: `[BH, NC, C, D]`
    - `log_alpha`: `[BH, NC, C]` with `log_alpha <= 0`

    Output:
    - `S_local_end`: `[BH, NC, MD]`

    Per-token recurrence on matrix state:
      alpha_t = exp(log_alpha_t)
      S_{t+1} = alpha_t * S_t + W_t ⊗ V_t,  S_0 = 0
      S_local_end = S_C
    Forward and backward run Triton kernels on CUDA only.
    """

    @staticmethod
    def forward(
        ctx,
        W: torch.Tensor,
        V: torch.Tensor,
        log_alpha: torch.Tensor,
        BLOCK_T: int = 32,
    ) -> torch.Tensor:
        if W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
            raise ValueError(
                "SsdRank1ChunkEndStateTriton.forward expects "
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
        _require_supported_md(M, D, where="SsdRank1ChunkEndStateTriton.forward")
        _require_chunk_size_multiple_of_16(C, where="SsdRank1ChunkEndStateTriton.forward")
        _require_nc_descriptor_width(NC, where="SsdRank1ChunkEndStateTriton.forward")
        _require_phase1_block_t(BLOCK_T, C, where="SsdRank1ChunkEndStateTriton.forward")
        _require_nonpositive_log_alpha(log_alpha, where="SsdRank1ChunkEndStateTriton.forward")
        if not W.is_cuda:
            raise NotImplementedError("SsdRank1ChunkEndStateTriton.forward requires CUDA tensors.")
        _ensure_triton_allocator()

        s_local_end_md = torch.empty((BH, NC, M, D), device=W.device, dtype=torch.float32)
        BLOCK_M = 16
        BLOCK_D = 32
        grid = (
            BH * NC,
            M // BLOCK_M,
            D // BLOCK_D,
        )
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
        s_local_end = s_local_end_md.reshape(BH, NC, M * D)
        ctx.save_for_backward(W, V, log_alpha)
        return s_local_end

    @staticmethod
    def backward(ctx, grad_s_local_end: torch.Tensor):
        """Backward for Phase-0 factorized chunk summary.

        Let:
        - `G = dL/dS_local_end`, shape `[BH, NC, M, D]`
        - `factor[t] = prod_{u=t+1..C-1} alpha_u`, where `alpha_u=exp(log_alpha_u)`

        Then:

          dW_t[m] = factor[t] * sum_d G[m,d] * V_t[d]
          dV_t[d] = factor[t] * sum_m G[m,d] * W_t[m]
          b_t     = sum_{m,d} G[m,d] * W_t[m] * V_t[d]

        `dR_t` uses scalar token recurrences built from `b_t` and the upstream
        gradient for `alpha_chunk`.

        Implementation split:
        1. `dW` kernel (D-reduction)
        2. `dV` kernel (M-reduction)
        3. `b_t` partial kernel + host tile reduction
        4. `dR` scalar recurrence kernel
        """
        W, V, log_alpha = ctx.saved_tensors
        BH, NC, C, M = W.shape
        D = V.shape[-1]
        MD = M * D
        _require_supported_md(M, D, where="SsdRank1ChunkEndStateTriton.backward")

        if grad_s_local_end is None:
            grad_s_local_end = torch.zeros((BH, NC, MD), device=W.device, dtype=torch.float32)
        if grad_s_local_end.shape != (BH, NC, MD):
            raise ValueError(
                f"grad_s_local_end must be [BH,NC,MD]=({BH},{NC},{MD}); got {tuple(grad_s_local_end.shape)}."
            )

        _require_chunk_size_multiple_of_16(C, where="SsdRank1ChunkEndStateTriton.backward")
        _require_nc_descriptor_width(NC, where="SsdRank1ChunkEndStateTriton.backward")
        if not W.is_cuda:
            raise NotImplementedError("SsdRank1ChunkEndStateTriton.backward requires CUDA tensors.")
        _ensure_triton_allocator()

        grad_s_md = grad_s_local_end.float().contiguous().view(BH, NC, M, D)
        d_W = torch.empty_like(W)
        d_V = torch.empty_like(V)
        d_log_alpha = torch.empty_like(log_alpha)
        BLOCK_M = 16
        BLOCK_D = 32
        n_m_tiles = M // BLOCK_M
        n_d_tiles = D // BLOCK_D

        # dW kernel: chunk-owned programs, tiled over M.
        grid_dw = (BH * NC, n_m_tiles)
        ssd_rank1_chunk_end_state_bwd_dw_chunk_kernel[grid_dw](
            grad_s_md,
            V,
            log_alpha,
            d_W,
            BH,
            NC,
            C,
            M,
            D,
            *grad_s_md.stride(),
            *V.stride(),
            *log_alpha.stride(),
            *d_W.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=C,
            C_STATIC=C,
            D_STATIC=D,
            INPUT_PRECISION="ieee" if W.dtype == torch.float32 else "tf32",
            num_warps=2,
            num_stages=2,
        )

        # dV + b-partials kernel: chunk-owned programs, tiled over D.
        b_partials = torch.empty((BH, NC, n_d_tiles, C), device=W.device, dtype=torch.float32)
        grid_dv = (BH * NC, n_d_tiles)
        ssd_rank1_chunk_end_state_bwd_dv_b_chunk_kernel[grid_dv](
            grad_s_md,
            W,
            V,
            log_alpha,
            d_V,
            b_partials,
            BH,
            NC,
            C,
            M,
            D,
            n_d_tiles,
            *grad_s_md.stride(),
            *W.stride(),
            *V.stride(),
            *log_alpha.stride(),
            *d_V.stride(),
            *b_partials.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=C,
            C_STATIC=C,
            M_STATIC=M,
            INPUT_PRECISION="ieee" if W.dtype == torch.float32 else "tf32",
            num_warps=2,
            num_stages=2,
        )
        b_vals = torch.sum(b_partials, dim=2).contiguous()

        # dlog_alpha from b-values and alpha_chunk upstream grad.
        grid_dr = (BH * NC,)
        ssd_rank1_chunk_end_state_bwd_dr_kernel[grid_dr](
            b_vals,
            log_alpha,
            d_log_alpha,
            BH,
            NC,
            C,
            *b_vals.stride(),
            *log_alpha.stride(),
            *d_log_alpha.stride(),
            BLOCK_C=C,
            C_STATIC=C,
            num_warps=2,
            num_stages=2,
        )
        return d_W, d_V, d_log_alpha, None

# --------------------------------------------------------------------------------------------------
# END PHASE 1: LOCAL CHUNK END-STATE COMPUTATION
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# PHASE 3: DENSE OUTPUT COMPUTATION
# --------------------------------------------------------------------------------------------------
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
    out_y_off_desc = tl.make_tensor_descriptor(
        out_y_off_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_out_y_off_bh, stride_out_y_off_nc, stride_out_y_off_c, stride_out_y_off_d],
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
    L = tl.where(valid, tl.exp2((log_p_incl[:, None] - log_p_incl[None, :]) * INV_LN2), 0.0)

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
    L = tl.where(valid, tl.exp2((log_p_incl[:, None] - log_p_incl[None, :]) * INV_LN2), 0.0)

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
    L = tl.where(valid, tl.exp2((log_p_incl[:, None] - log_p_incl[None, :]) * INV_LN2), 0.0)

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
    A1_NUM_WARPS: int | None = None,
    A2_NUM_WARPS: int | None = None,
    OFF_NUM_WARPS: int | None = None,
    DIAG_NUM_WARPS: int | None = None,
    NUM_STAGES: int | None = None,
    WARMUP: bool = True,
) -> dict[str, float]:
    """Profile split Phase-3 Triton backward kernels and return per-stage timings (ms).

    This mirrors `SsdRank1DenseOutputTriton.backward` without autograd plumbing so
    kernel-level bottlenecks can be tuned quickly.
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
        y_off_saved = None
    else:
        if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
            S0_md = S0.reshape(BH, NC, M, D).to(C.dtype)
        elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
            S0_md = S0.to(C.dtype)
        else:
            raise ValueError("S0 shape mismatch.")
        has_s0 = True
        log_p = torch.cumsum(log_alpha.float(), dim=-1)
        y_off_saved = (torch.exp2(log_p * _INV_LN2).unsqueeze(-1) * torch.matmul(C.float(), S0_md.float())).to(C.dtype).contiguous()

    BLOCK_M = min(_PHASE3_BWD_BLOCK_M, M) if BLOCK_M is None else BLOCK_M
    BLOCK_D = min(_PHASE3_BWD_BLOCK_D, D) if BLOCK_D is None else BLOCK_D
    A1_NUM_WARPS = _PHASE3_BWD_A1_NUM_WARPS if A1_NUM_WARPS is None else A1_NUM_WARPS
    A2_NUM_WARPS = _PHASE3_BWD_A2_NUM_WARPS if A2_NUM_WARPS is None else A2_NUM_WARPS
    OFF_NUM_WARPS = _PHASE3_BWD_OFF_NUM_WARPS if OFF_NUM_WARPS is None else OFF_NUM_WARPS
    DIAG_NUM_WARPS = _PHASE3_BWD_DIAG_NUM_WARPS if DIAG_NUM_WARPS is None else DIAG_NUM_WARPS
    NUM_STAGES = _PHASE3_BWD_NUM_STAGES if NUM_STAGES is None else NUM_STAGES
    if M % BLOCK_M != 0 or D % BLOCK_D != 0:
        raise NotImplementedError(
            f"Profile currently requires M%BLOCK_M==0 and D%BLOCK_D==0; got M={M}, D={D}, BLOCK_M={BLOCK_M}, BLOCK_D={BLOCK_D}."
        )

    n_d_tiles = D // BLOCK_D
    n_m_tiles = M // BLOCK_M
    grad_out_c = grad_out.contiguous()

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
    if n_d_tiles == 1:
        dK = torch.empty((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32)
        dK_partials = None
        dK_partials_flat = dK.view(BH * NC, C_CHUNK, C_CHUNK)
    else:
        dK_partials = torch.empty((BH, NC, n_d_tiles, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32)
        dK_partials_flat = dK_partials.view(BH * NC * n_d_tiles, C_CHUNK, C_CHUNK)
        dK = torch.empty((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32)
    R_buf = torch.empty((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32)

    def _run_a1():
        grid_a1 = (BH * NC, n_d_tiles)
        ssd_rank1_dense_output_bwd_a1_kernel[grid_a1](
            C,
            W,
            V,
            log_alpha,
            grad_out_c,
            dV,
            dK_partials_flat,
            R_buf,
            BH,
            NC,
            C_CHUNK,
            M,
            D,
            n_d_tiles,
            *C.stride(),
            *W.stride(),
            *V.stride(),
            *log_alpha.stride(),
            *grad_out_c.stride(),
            *dV.stride(),
            *dK_partials_flat.stride(),
            *R_buf.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=C_CHUNK,
            C_STATIC=C_CHUNK,
            M_STATIC=M,
            INPUT_PRECISION=INPUT_PRECISION,
            num_warps=A1_NUM_WARPS,
            num_stages=NUM_STAGES,
        )

    if WARMUP:
        _run_a1()
        torch.cuda.synchronize()
    _time("a1_ms", _run_a1)

    if n_d_tiles == 1:
        times["dk_reduce_ms"] = 0.0
    else:
        def _run_reduce():
            torch.sum(dK_partials, dim=2, out=dK)

        if WARMUP:
            _run_reduce()
            torch.cuda.synchronize()
        _time("dk_reduce_ms", _run_reduce)

    dC_diag = torch.empty_like(C)
    dW = torch.empty_like(W)
    Q = torch.empty((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32)

    def _run_a2():
        grid_a2 = (BH * NC, n_m_tiles)
        ssd_rank1_dense_output_bwd_a2_kernel[grid_a2](
            C,
            W,
            log_alpha,
            dK,
            R_buf,
            dC_diag,
            dW,
            Q,
            BH,
            NC,
            C_CHUNK,
            M,
            *C.stride(),
            *W.stride(),
            *log_alpha.stride(),
            *dK.stride(),
            *R_buf.stride(),
            *dC_diag.stride(),
            *dW.stride(),
            *Q.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_C=C_CHUNK,
            C_STATIC=C_CHUNK,
            INPUT_PRECISION=INPUT_PRECISION,
            num_warps=A2_NUM_WARPS,
            num_stages=NUM_STAGES,
        )

    if WARMUP:
        _run_a2()
        torch.cuda.synchronize()
    _time("a2_ms", _run_a2)

    dlog_diag = torch.empty((BH, NC, C_CHUNK), device=C.device, dtype=torch.float32)

    def _run_diag():
        grid_diag = (BH * NC,)
        ssd_rank1_dense_output_bwd_logalpha_diag_kernel[grid_diag](
            Q,
            dlog_diag,
            BH,
            NC,
            C_CHUNK,
            *Q.stride(),
            *dlog_diag.stride(),
            BLOCK_C=C_CHUNK,
            C_STATIC=C_CHUNK,
            num_warps=DIAG_NUM_WARPS,
            num_stages=NUM_STAGES,
        )

    if WARMUP:
        _run_diag()
        torch.cuda.synchronize()
    _time("log_diag_ms", _run_diag)

    if has_s0:
        dS0 = torch.empty((BH, NC, M, D), device=C.device, dtype=torch.float32)
        def _run_off_dc():
            grid_off_dc = (BH * NC, n_m_tiles)
            ssd_rank1_dense_output_bwd_off_dc_kernel[grid_off_dc](
                C,
                grad_out_c,
                log_alpha,
                S0_md,
                dC_diag,
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
                BLOCK_M=BLOCK_M,
                BLOCK_D=BLOCK_D,
                BLOCK_C=C_CHUNK,
                C_STATIC=C_CHUNK,
                D_STATIC=D,
                INPUT_PRECISION=INPUT_PRECISION,
                num_warps=OFF_NUM_WARPS,
                num_stages=NUM_STAGES,
            )

        if WARMUP:
            _run_off_dc()
            torch.cuda.synchronize()
        _time("off_dc_ms", _run_off_dc)

        def _run_off_ds0():
            grid_off_ds0 = (BH * NC, n_m_tiles, n_d_tiles)
            ssd_rank1_dense_output_bwd_off_ds0_kernel[grid_off_ds0](
                C,
                grad_out_c,
                log_alpha,
                dS0,
                BH,
                NC,
                C_CHUNK,
                M,
                D,
                *C.stride(),
                *grad_out_c.stride(),
                *log_alpha.stride(),
                *dS0.stride(),
                BLOCK_M=BLOCK_M,
                BLOCK_D=BLOCK_D,
                BLOCK_C=C_CHUNK,
                INPUT_PRECISION=INPUT_PRECISION,
                num_warps=OFF_NUM_WARPS,
                num_stages=NUM_STAGES,
            )

        if WARMUP:
            _run_off_ds0()
            torch.cuda.synchronize()
        _time("off_ds0_ms", _run_off_ds0)

        def _run_off_log():
            grid_off_log = (BH * NC,)
            ssd_rank1_dense_output_bwd_logalpha_off_kernel[grid_off_log](
                y_off_saved,
                grad_out_c,
                dlog_diag,
                BH,
                NC,
                C_CHUNK,
                D,
                *y_off_saved.stride(),
                *grad_out_c.stride(),
                *dlog_diag.stride(),
                BLOCK_D=BLOCK_D,
                BLOCK_C=C_CHUNK,
                D_STATIC=D,
                num_warps=OFF_NUM_WARPS,
                num_stages=NUM_STAGES,
            )

        if WARMUP:
            _run_off_log()
            torch.cuda.synchronize()
        _time("log_off_ms", _run_off_log)

        def _run_combine():
            _ = dS0

        if WARMUP:
            _run_combine()
            torch.cuda.synchronize()
        _time("combine_ms", _run_combine)
    else:
        times["off_dc_ms"] = 0.0
        times["off_ds0_ms"] = 0.0
        times["log_off_ms"] = 0.0
        times["combine_ms"] = 0.0

    times["total_ms"] = (
        times["a1_ms"]
        + times["dk_reduce_ms"]
        + times["a2_ms"]
        + times["log_diag_ms"]
        + times["off_dc_ms"]
        + times["off_ds0_ms"]
        + times["log_off_ms"]
        + times["combine_ms"]
    )
    times["BLOCK_M"] = float(BLOCK_M)
    times["BLOCK_D"] = float(BLOCK_D)
    times["A1_NUM_WARPS"] = float(A1_NUM_WARPS)
    times["A2_NUM_WARPS"] = float(A2_NUM_WARPS)
    times["OFF_NUM_WARPS"] = float(OFF_NUM_WARPS)
    times["DIAG_NUM_WARPS"] = float(DIAG_NUM_WARPS)
    return times


class SsdRank1DenseOutputTriton(torch.autograd.Function):
    """Autograd contour for dense chunk-local outputs.

    Forward computes:

      y_t = sum_{tau=0..t} [prod_{u=tau+1..t} alpha_u] * (C_t^T W_tau) * V_tau
      where alpha_u = exp(log_alpha_u).

    with:
    - `C`: `[BH, NC, C, M]`
    - `W`: `[BH, NC, C, M]`
    - `V`: `[BH, NC, C, D]`
    - `log_alpha`: `[BH, NC, C]` with `log_alpha <= 0`
    - `S0`: optional `[BH, NC, MD]` or `[BH, NC, M, D]` chunk-start state
    - output `y`: `[BH, NC, C, D]`

    Backward currently uses split Triton kernels:
    - A1: `dV` + `dK` partials + cached `R`
    - A2: `dC_diag`, `dW`, `Q`
    - Off-path kernels: `dC_off`, `dS0`, `dlog_alpha_off`
    - Logalpha diag kernel from `Q`
    """

    @staticmethod
    def forward(
        ctx,
        C: torch.Tensor,
        W: torch.Tensor,
        V: torch.Tensor,
        log_alpha: torch.Tensor,
        S0: torch.Tensor | None = None,
        INPUT_PRECISION: str = "tf32",
    ) -> torch.Tensor:
        if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
            raise ValueError(
                "SsdRank1DenseOutputTriton.forward expects C=[BH,NC,C,M], "
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
        _require_supported_md(M, D, where="SsdRank1DenseOutputTriton.forward")
        _require_chunk_size_multiple_of_16(C_CHUNK, where="SsdRank1DenseOutputTriton.forward")
        _require_nonpositive_log_alpha(log_alpha, where="SsdRank1DenseOutputTriton.forward")
        if not C.is_cuda:
            raise NotImplementedError("SsdRank1DenseOutputTriton.forward requires CUDA tensors.")
        _ensure_triton_allocator()

        if S0 is None:
            S0_md = torch.zeros((BH, NC, M, D), device=C.device, dtype=C.dtype)
            ctx.s0_was_flat = False
        else:
            if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
                S0_md = S0.reshape(BH, NC, M, D).to(C.dtype)
                ctx.s0_was_flat = True
            elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
                S0_md = S0.to(C.dtype)
                ctx.s0_was_flat = False
            else:
                raise ValueError(
                    f"S0 must be [BH,NC,MD]=({BH},{NC},{M * D}) or [BH,NC,M,D]=({BH},{NC},{M},{D}); "
                    f"got {tuple(S0.shape)}."
                )
            if S0_md.device != C.device:
                raise ValueError("S0 must be on the same device as C/W/V/log_alpha.")

        out = torch.empty((BH, NC, C_CHUNK, D), device=C.device, dtype=C.dtype)
        y_off_saved = torch.empty((BH, NC, C_CHUNK, D), device=C.device, dtype=C.dtype)
        BLOCK_M = min(32, M)
        BLOCK_D = min(64, D)
        grid = (BH * NC, D // BLOCK_D)
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
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=C_CHUNK,
            C_STATIC=C_CHUNK,
            M_STATIC=M,
            INPUT_PRECISION=INPUT_PRECISION,
            num_warps=4,
            num_stages=2,
        )
        if S0 is None:
            S0_saved = torch.empty(0, device=C.device, dtype=C.dtype)
            y_off_out = torch.empty(0, device=C.device, dtype=C.dtype)
        else:
            S0_saved = S0_md
            y_off_out = y_off_saved
        ctx.input_precision = INPUT_PRECISION
        ctx.save_for_backward(C, W, V, log_alpha, S0_saved, y_off_out)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        C, W, V, log_alpha, S0_saved, y_off_saved = ctx.saved_tensors
        BH, NC, C_CHUNK, M = C.shape
        D = V.shape[-1]
        _require_supported_md(M, D, where="SsdRank1DenseOutputTriton.backward")

        if grad_out is None:
            grad_out = torch.zeros((BH, NC, C_CHUNK, D), device=C.device, dtype=C.dtype)
        if grad_out.shape != (BH, NC, C_CHUNK, D):
            raise ValueError(
                f"grad_out must be [BH,NC,C,D]=({BH},{NC},{C_CHUNK},{D}); got {tuple(grad_out.shape)}."
            )
        _require_chunk_size_multiple_of_16(C_CHUNK, where="SsdRank1DenseOutputTriton.backward")
        if not C.is_cuda:
            raise NotImplementedError("SsdRank1DenseOutputTriton.backward requires CUDA tensors.")
        _ensure_triton_allocator()

        INPUT_PRECISION = getattr(ctx, "input_precision", "tf32")
        BLOCK_M = min(_PHASE3_BWD_BLOCK_M, M)
        BLOCK_D = min(_PHASE3_BWD_BLOCK_D, D)
        if M % BLOCK_M != 0 or D % BLOCK_D != 0:
            raise NotImplementedError(
                f"SsdRank1DenseOutputTriton.backward requires M%BLOCK_M==0 and D%BLOCK_D==0; "
                f"got M={M}, D={D}, BLOCK_M={BLOCK_M}, BLOCK_D={BLOCK_D}."
            )
        n_d_tiles = D // BLOCK_D
        n_m_tiles = M // BLOCK_M

        # A1: dV and dK partials (+ R cache).
        dV = torch.empty_like(V)
        if n_d_tiles == 1:
            dK = _ssd_rank1_bwd_workspace_tensor(
                "dK",
                (BH, NC, C_CHUNK, C_CHUNK),
                device=C.device,
                dtype=torch.float32,
            )
            dK_partials = None
            dK_partials_flat = dK.view(BH * NC, C_CHUNK, C_CHUNK)
        else:
            dK_partials = _ssd_rank1_bwd_workspace_tensor(
                "dK_partials",
                (BH, NC, n_d_tiles, C_CHUNK, C_CHUNK),
                device=C.device,
                dtype=torch.float32,
            )
            dK_partials_flat = dK_partials.view(BH * NC * n_d_tiles, C_CHUNK, C_CHUNK)
            dK = _ssd_rank1_bwd_workspace_tensor(
                "dK",
                (BH, NC, C_CHUNK, C_CHUNK),
                device=C.device,
                dtype=torch.float32,
            )
        R_buf = _ssd_rank1_bwd_workspace_tensor(
            "R_buf",
            (BH, NC, C_CHUNK, C_CHUNK),
            device=C.device,
            dtype=torch.float32,
        )
        grad_out_c = grad_out.contiguous()

        grid_a1 = (BH * NC, n_d_tiles)
        ssd_rank1_dense_output_bwd_a1_kernel[grid_a1](
            C,
            W,
            V,
            log_alpha,
            grad_out_c,
            dV,
            dK_partials_flat,
            R_buf,
            BH,
            NC,
            C_CHUNK,
            M,
            D,
            n_d_tiles,
            *C.stride(),
            *W.stride(),
            *V.stride(),
            *log_alpha.stride(),
            *grad_out_c.stride(),
            *dV.stride(),
            *dK_partials_flat.stride(),
            *R_buf.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=C_CHUNK,
            C_STATIC=C_CHUNK,
            M_STATIC=M,
            INPUT_PRECISION=INPUT_PRECISION,
            num_warps=_PHASE3_BWD_A1_NUM_WARPS,
            num_stages=_PHASE3_BWD_NUM_STAGES,
        )
        if n_d_tiles != 1:
            torch.sum(dK_partials, dim=2, out=dK)

        # A2: dC_diag, dW, Q.
        dC_diag = torch.empty_like(C)
        dW = torch.empty_like(W)
        Q = _ssd_rank1_bwd_workspace_tensor(
            "Q",
            (BH, NC, C_CHUNK, C_CHUNK),
            device=C.device,
            dtype=torch.float32,
        )
        grid_a2 = (BH * NC, n_m_tiles)
        ssd_rank1_dense_output_bwd_a2_kernel[grid_a2](
            C,
            W,
            log_alpha,
            dK,
            R_buf,
            dC_diag,
            dW,
            Q,
            BH,
            NC,
            C_CHUNK,
            M,
            *C.stride(),
            *W.stride(),
            *log_alpha.stride(),
            *dK.stride(),
            *R_buf.stride(),
            *dC_diag.stride(),
            *dW.stride(),
            *Q.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_C=C_CHUNK,
            C_STATIC=C_CHUNK,
            INPUT_PRECISION=INPUT_PRECISION,
            num_warps=_PHASE3_BWD_A2_NUM_WARPS,
            num_stages=_PHASE3_BWD_NUM_STAGES,
        )

        # dlog_alpha diagonal contribution from Q.
        dlog_diag = _ssd_rank1_bwd_workspace_tensor(
            "dlog_diag",
            (BH, NC, C_CHUNK),
            device=C.device,
            dtype=torch.float32,
        )
        grid_diag = (BH * NC,)
        ssd_rank1_dense_output_bwd_logalpha_diag_kernel[grid_diag](
            Q,
            dlog_diag,
            BH,
            NC,
            C_CHUNK,
            *Q.stride(),
            *dlog_diag.stride(),
            BLOCK_C=C_CHUNK,
            C_STATIC=C_CHUNK,
            num_warps=_PHASE3_BWD_DIAG_NUM_WARPS,
            num_stages=_PHASE3_BWD_NUM_STAGES,
        )

        has_s0 = S0_saved.numel() > 0
        if has_s0:
            dS0 = torch.empty((BH, NC, M, D), device=C.device, dtype=torch.float32)
            grid_off_dc = (BH * NC, n_m_tiles)
            ssd_rank1_dense_output_bwd_off_dc_kernel[grid_off_dc](
                C,
                grad_out_c,
                log_alpha,
                S0_saved,
                dC_diag,
                BH,
                NC,
                C_CHUNK,
                M,
                D,
                *C.stride(),
                *grad_out_c.stride(),
                *log_alpha.stride(),
                *S0_saved.stride(),
                *dC_diag.stride(),
                BLOCK_M=BLOCK_M,
                BLOCK_D=BLOCK_D,
                BLOCK_C=C_CHUNK,
                C_STATIC=C_CHUNK,
                D_STATIC=D,
                INPUT_PRECISION=INPUT_PRECISION,
                num_warps=_PHASE3_BWD_OFF_NUM_WARPS,
                num_stages=_PHASE3_BWD_NUM_STAGES,
            )

            grid_off_ds0 = (BH * NC, n_m_tiles, n_d_tiles)
            ssd_rank1_dense_output_bwd_off_ds0_kernel[grid_off_ds0](
                C,
                grad_out_c,
                log_alpha,
                dS0,
                BH,
                NC,
                C_CHUNK,
                M,
                D,
                *C.stride(),
                *grad_out_c.stride(),
                *log_alpha.stride(),
                *dS0.stride(),
                BLOCK_M=BLOCK_M,
                BLOCK_D=BLOCK_D,
                BLOCK_C=C_CHUNK,
                INPUT_PRECISION=INPUT_PRECISION,
                num_warps=_PHASE3_BWD_OFF_NUM_WARPS,
                num_stages=_PHASE3_BWD_NUM_STAGES,
            )

            grid_off_log = (BH * NC,)
            ssd_rank1_dense_output_bwd_logalpha_off_kernel[grid_off_log](
                y_off_saved,
                grad_out_c,
                dlog_diag,
                BH,
                NC,
                C_CHUNK,
                D,
                *y_off_saved.stride(),
                *grad_out_c.stride(),
                *dlog_diag.stride(),
                BLOCK_D=BLOCK_D,
                BLOCK_C=C_CHUNK,
                D_STATIC=D,
                num_warps=_PHASE3_BWD_OFF_NUM_WARPS,
                num_stages=_PHASE3_BWD_NUM_STAGES,
            )

            dC = dC_diag
            dlog_alpha = dlog_diag
            if getattr(ctx, "s0_was_flat", False):
                dS0_out = dS0.reshape(BH, NC, M * D).to(C.dtype)
            else:
                dS0_out = dS0.to(C.dtype)
        else:
            dC = dC_diag
            dlog_alpha = dlog_diag
            dS0_out = None

        return dC, dW, dV, dlog_alpha.to(log_alpha.dtype), dS0_out, None

# --------------------------------------------------------------------------------------------------
# END PHASE 3: DENSE OUTPUT COMPUTATION
# --------------------------------------------------------------------------------------------------


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
class SsdRank1PrefixScanTriton(torch.autograd.Function):
    """Autograd contour for mode-0 chunk-prefix scan (log-domain chunk retain).

    Forward:
      Inputs:
        S_local_end : [BH, NC, MD] (or [BHG, NC, MD])
        log_alpha_chunk : [BH, NC] (or [BHG, NC])
        init   : [BH, MD] (or [BHG, MD])
      Outputs:
        chunk_start : [BH, NC, MD] (or [BHG, NC, MD])
        final_state : [BH, MD] (or [BHG, MD])

      Dense blocked formulation:
        log_prefix_excl[c] = sum_{u=0..c-1} log_alpha_chunk[u]
        log_prefix_incl[c] = log_prefix_excl[c] + log_alpha_chunk[c]
        L0[i,j]            = 1_{j<i} * exp(log_prefix_excl[i] - log_prefix_incl[j])
        chunk_start        = L0 @ S_local_end + exp(log_prefix_excl)[:, None] * init
        final_state        = exp(log_alpha_chunk[last]) * chunk_start[last] + S_local_end[last]

    Backward (high level):
      Let upstream grads be:
        g_start[c] = dL/d chunk_start[c]
        g_final    = dL/d final_state

      1) Build gradient wrt S-trajectory with reverse recurrence:
           g_end[NC-1] starts from g_final
           g_start[c] accumulates from explicit chunk_start grad and from next chunk:
             g_start[c] += g_chunk_start[c] + alpha_chunk[c] * g_end[c]
         (with boundary terms around c=0 handled explicitly).

      2) Parameter grads from local Jacobians:
           dL/dr_chunk[c] += sum_md g_end[c, md] * chunk_start[c, md]
           dL/dS_local_end[c, md] += g_end[c, md]

      3) Initial-S grad:
           dL/dinit = g_chunk_start[0] + alpha_chunk[0] * g_end[0]
         equivalently obtained from reverse scan terminal.

    Kernel plan:
      - FWD:
        one kernel over `(BH, MD-tile)` with static loop over chunk blocks;
        forms strict-lower `L` on-the-fly, emits `chunk_start`, carries `S_in`,
        and writes `final_state` at loop end.
      - BWD:
        one dense blocked reverse kernel over `BH`:
        a) rebuild reversed-block carry `L0_rev` in log space;
        b) compute `lambda_next_rev = L0_rev @ g_rev + p_rev * lambda_in`;
        c) emit `dS_local_end`, accumulate `dlog_alpha`, and update `dinit`.
    """

    @staticmethod
    def forward(
        ctx,
        S_local_end: torch.Tensor,
        log_alpha_chunk: torch.Tensor,
        initial_state: torch.Tensor,
        compute_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Shape / contract checks.
        if S_local_end.ndim != 3 or log_alpha_chunk.ndim != 2 or initial_state.ndim != 2:
            raise ValueError(
                "SsdRank1PrefixScanTriton.forward expects "
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
            raise NotImplementedError("SsdRank1PrefixScanTriton.forward does not support NC==0.")
        _require_nc_descriptor_width(NC, where="SsdRank1PrefixScanTriton.forward")
        _require_nc_multiple_of_16(NC, where="SsdRank1PrefixScanTriton.forward")
        if not S_local_end.is_cuda:
            raise NotImplementedError("SsdRank1PrefixScanTriton.forward requires CUDA tensors.")
        _ensure_triton_allocator()

        if compute_dtype is None:
            compute_dtype = torch.float32
        if compute_dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise NotImplementedError(
                "SsdRank1PrefixScanTriton.forward supports compute_dtype in "
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

        block_nc = 32 if (NC >= 32 and NC % 32 == 0) else 16
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

        ctx.save_for_backward(log_alpha_chunk_f, chunk_start)
        ctx.s_local_dtype = S_local_end.dtype
        ctx.log_alpha_dtype = log_alpha_chunk.dtype
        ctx.init_dtype = initial_state.dtype
        ctx.phase2_use_fp32_compute = compute_dtype == torch.float32
        ctx.phase2_use_bf16_compute = compute_dtype == torch.bfloat16
        return chunk_start, final_state

    @staticmethod
    def backward(ctx, grad_chunk_start: torch.Tensor, grad_final_state: torch.Tensor):
        log_alpha_chunk, chunk_start = ctx.saved_tensors

        if grad_chunk_start.shape != chunk_start.shape:
            raise ValueError(
                f"grad_chunk_start must match chunk_start shape. Got {tuple(grad_chunk_start.shape)} vs {tuple(chunk_start.shape)}."
            )
        if grad_final_state.shape != (chunk_start.shape[0], chunk_start.shape[2]):
            raise ValueError(
                f"grad_final_state must have shape [BH,MD]={(chunk_start.shape[0], chunk_start.shape[2])}, "
                f"got {tuple(grad_final_state.shape)}."
            )

        BH, NC, MD = chunk_start.shape
        if NC == 0:
            raise NotImplementedError("SsdRank1PrefixScanTriton.backward does not support NC==0.")
        _require_nc_descriptor_width(NC, where="SsdRank1PrefixScanTriton.backward")
        _require_nc_multiple_of_16(NC, where="SsdRank1PrefixScanTriton.backward")
        if not chunk_start.is_cuda:
            raise NotImplementedError("SsdRank1PrefixScanTriton.backward requires CUDA tensors.")
        _ensure_triton_allocator()

        grad_chunk_start_f = grad_chunk_start.float().contiguous()
        grad_final_state_f = grad_final_state.float().contiguous()
        d_s_local_f32 = torch.empty((BH, NC, MD), device=chunk_start.device, dtype=torch.float32)
        d_log_alpha_f32 = torch.empty((BH, NC), device=chunk_start.device, dtype=torch.float32)
        d_init_f32 = torch.empty((BH, MD), device=chunk_start.device, dtype=torch.float32)
        block_nc = 32 if (NC >= 32 and NC % 32 == 0) else 16

        grid_bwd = lambda META: (BH,)
        ssd_rank1_prefix_scan_bwd_dense_kernel[grid_bwd](
            grad_chunk_start_f,
            grad_final_state_f,
            chunk_start,
            log_alpha_chunk,
            d_s_local_f32,
            d_log_alpha_f32,
            d_init_f32,
            BH,
            NC,
            MD,
            *grad_chunk_start_f.stride(),
            *grad_final_state_f.stride(),
            *chunk_start.stride(),
            *log_alpha_chunk.stride(),
            *d_s_local_f32.stride(),
            *d_log_alpha_f32.stride(),
            *d_init_f32.stride(),
            BLOCK_NC=block_nc,
            NC_STATIC=NC,
            USE_FP32_COMPUTE=ctx.phase2_use_fp32_compute,
            USE_BF16_COMPUTE=ctx.phase2_use_bf16_compute,
        )
        return d_s_local_f32.to(ctx.s_local_dtype), d_log_alpha_f32.to(ctx.log_alpha_dtype), d_init_f32.to(ctx.init_dtype), None

# --------------------------------------------------------------------------------------------------
# END PHASE 2: PREFIX SCAN OVER CHUNKS
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# PHASE 123: FULL PARALLEL SCAN
# --------------------------------------------------------------------------------------------------
def ssd_rank1_triton(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
    INPUT_PRECISION: str = "tf32",
) -> tuple[torch.Tensor, torch.Tensor]:
    """SSD rank-1 Triton entrypoint.

    This composes the phase-1/2/3 Triton autograd functions directly.
    """
    if CHUNK_SIZE is None:
        CHUNK_SIZE = 64

    C_chunk, W_chunk, V_chunk, log_alpha_chunk, init_flat, B, N, H, M, D, _ = _ssd_rank1_prepare_unchunked_inputs(
        C,
        W,
        V,
        log_alpha,
        initial_state,
        where="ssd_rank1_triton",
        CHUNK_SIZE=CHUNK_SIZE,
    )

    # =========================================
    # PHASE 1 local chunk end-state
    # =========================================
    S_local_end = SsdRank1ChunkEndStateTriton.apply(
        W_chunk,
        V_chunk,
        log_alpha_chunk,
        32,
    )

    # =========================================
    # PHASE 2 prefix scan over chunks
    # =========================================
    log_alpha_per_chunk = torch.sum(log_alpha_chunk.float(), dim=-1)
    S0_chunk, S1_chunk = SsdRank1PrefixScanTriton.apply(
        S_local_end,
        log_alpha_per_chunk,
        init_flat,
        V_chunk.dtype,
    )

    # =========================================
    # PHASE 3 dense chunk-local output
    # =========================================
    y_chunk = SsdRank1DenseOutputTriton.apply(
        C_chunk,
        W_chunk,
        V_chunk,
        log_alpha_chunk,
        S0_chunk,
        INPUT_PRECISION,
    )

    return _ssd_rank1_restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)


# --------------------------------------------------------------------------------------------------
# END PHASE 123: FULL PARALLEL SCAN
# --------------------------------------------------------------------------------------------------


__all__ = [
    "ssd_rank1_pytorch",
    "ssd_rank1_triton",
]

"""Triton path for experimental autoregressive chunk-prefix state scans.

This file currently implements forward for mode-0 chunk-prefix propagation and
keeps backward as an explicit TODO.
"""

from __future__ import annotations

import torch

import triton
import triton.language as tl

_SUPPORTED_M_VALUES = {16, 32, 64, 128}
_SUPPORTED_D_VALUES = {32, 64, 128}
_SUPPORTED_PHASE1_BLOCK_T_VALUES = {4, 8, 16, 32}
_EXPERIMENTAL_TRITON_ALLOCATOR_SET = False


def _ensure_triton_allocator() -> None:
    global _EXPERIMENTAL_TRITON_ALLOCATOR_SET
    if _EXPERIMENTAL_TRITON_ALLOCATOR_SET:
        return

    def _alloc(size: int, _align: int, _stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(_alloc)
    _EXPERIMENTAL_TRITON_ALLOCATOR_SET = True


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


def _require_nonpositive_log_alpha(log_alpha: torch.Tensor, *, where: str) -> None:
    if torch.any(log_alpha > 0):
        max_val = float(torch.max(log_alpha).item())
        raise ValueError(f"{where} requires log_alpha <= 0 everywhere; observed max(log_alpha)={max_val}.")


def _require_phase1_block_t(block_t: int, c_size: int, *, where: str) -> None:
    if block_t not in _SUPPORTED_PHASE1_BLOCK_T_VALUES:
        raise NotImplementedError(
            f"{where} requires BLOCK_T in {sorted(_SUPPORTED_PHASE1_BLOCK_T_VALUES)}; got BLOCK_T={block_t}."
        )
    if block_t > c_size:
        raise NotImplementedError(f"{where} requires BLOCK_T <= C; got BLOCK_T={block_t}, C={c_size}.")
    if c_size % block_t != 0:
        raise NotImplementedError(f"{where} requires C divisible by BLOCK_T; got C={c_size}, BLOCK_T={block_t}.")

# ==================================================================================================
# REFERENCE FUNCTIONS
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# PHASE 1: LOCAL CHUNK END-STATE COMPUTATION
# --------------------------------------------------------------------------------------------------
def phase1_local_chunk_end_state_reference(
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
            "phase1_local_chunk_end_state_reference expects W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
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
        raise NotImplementedError("phase1_local_chunk_end_state_reference does not support C==0.")
    _require_supported_md(M, D, where="phase1_local_chunk_end_state_reference")
    _require_nonpositive_log_alpha(log_alpha, where="phase1_local_chunk_end_state_reference")
    log_alpha_f = log_alpha.float()
    alpha = torch.exp(log_alpha_f)
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
def phase2_prefix_scan_reference(
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
            "phase2_prefix_scan_reference expects "
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
        raise NotImplementedError("phase2_prefix_scan_reference does not support NC==0.")

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
def phase3_dense_output_reference(
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
            "phase3_dense_output_reference expects C=[BH,NC,C,M], "
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
        raise NotImplementedError("phase3_dense_output_reference does not support C==0.")
    _require_supported_md(M, D, where="phase3_dense_output_reference")
    _require_nonpositive_log_alpha(log_alpha, where="phase3_dense_output_reference")
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
        torch.exp(log_p[..., :, None] - log_p[..., None, :]),
        torch.zeros((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32),
    )

    p = torch.exp(log_p).to(torch.float32).unsqueeze(-1)
    Y_off = p * torch.matmul(C_f, S0_md)

    K = (L * (C_f @ W_f.mT)).to(torch.float32)
    Y_diag = torch.matmul(K, V_f)

    return (Y_diag + Y_off).to(V.dtype)


# --------------------------------------------------------------------------------------------------
# END PHASE 3: DENSE OUTPUT COMPUTATION
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# PHASE 123: FULL PARALLEL SCAN
# --------------------------------------------------------------------------------------------------
def _phase123_prepare_unchunked_inputs(
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

    NC_exec = max(NC_data, 8)
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


def _phase123_restore_output_layout(
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


def phase123_full_parallel_scan_reference(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full mode-0 parallel scan reference from unchunked `[B,N,H,*]` inputs.

    Inputs:
    - C: `[B, N, H, M]` read vectors.
    - W: `[B, N, H, M]` write vectors.
    - V: `[B, N, H, D]` value vectors.
    - log_alpha: `[B, N, H]` scalar retains.
    - initial_state: optional `[BH,MD]`, `[B,H,MD]`, or `[B,H,M,D]`.
    - CHUNK_SIZE: chunk length (`None` -> default `64`).

    Returns:
    - y: `[B, N, H, D]` token outputs.
    - final_state: `[B, H, MD]` final S after all chunks.

    Composition:
    1. Chunk/pad unchunked inputs into `[BH, NC, C, *]`.
    2. Phase-1 computes chunk-local summaries:
         `S_local_end`, `alpha_chunk`.
    3. Phase-2 computes chunk-start states from those summaries:
         `S0_chunk`, `final_state`.
    4. Phase-3 computes chunk-local dense outputs, including prefix-state readout:
         `y_chunk`.
    5. Unchunk and trim back to `N` tokens.
    """
    if CHUNK_SIZE is None:
        CHUNK_SIZE = 64

    C_chunk, W_chunk, V_chunk, log_alpha_chunk, init_flat, B, N, H, M, D, _ = _phase123_prepare_unchunked_inputs(
        C,
        W,
        V,
        log_alpha,
        initial_state,
        where="phase123_full_parallel_scan_reference",
        CHUNK_SIZE=CHUNK_SIZE,
    )

    # =========================================
    # PHASE 1 local chunk end-state
    # =========================================
    S_local_end = phase1_local_chunk_end_state_reference(W_chunk, V_chunk, log_alpha_chunk)
    alpha_chunk = torch.exp(torch.sum(log_alpha_chunk, dim=-1))

    # =========================================
    # PHASE 2 prefix scan over chunks
    # =========================================
    S0_chunk, S1_chunk = phase2_prefix_scan_reference(S_local_end, alpha_chunk, init_flat)

    # =========================================
    # PHASE 3 dense chunk-local output
    # =========================================
    y_chunk = phase3_dense_output_reference(C_chunk, W_chunk, V_chunk, log_alpha_chunk, S0_chunk)

    return _phase123_restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)


def phase123_full_token_loop_oracle(
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
            "phase123_full_token_loop_oracle expects C=[B,N,H,M], W=[B,N,H,M], V=[B,N,H,D], log_alpha=[B,N,H]. "
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
        raise NotImplementedError("phase123_full_token_loop_oracle does not support N==0.")
    _require_supported_md(M, D, where="phase123_full_token_loop_oracle")
    _require_nonpositive_log_alpha(log_alpha, where="phase123_full_token_loop_oracle")

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

    alpha = torch.exp(log_alpha)
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
# END PHASE 123: FULL PARALLEL SCAN
# --------------------------------------------------------------------------------------------------


# ==================================================================================================
# TRITON KERNELS + AUTOGRAD
# ==================================================================================================
# --------------------------------------------------------------------------------------------------
# PHASE 1: LOCAL CHUNK END-STATE COMPUTATION
# --------------------------------------------------------------------------------------------------
@triton.jit
def phase1_local_chunk_end_state_fwd_kernel(
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
       implemented as:
         W_scaled = factor[:, None] * W_block   # [BLOCK_T, BLOCK_M]
         S += W_scaled^T @ V_block              # [BLOCK_M, BLOCK_D]
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
    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    r_factors = tl.exp(log_suffix_incl - log_alpha_vals)

    S = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    offs_t_full = tl.arange(0, BLOCK_C)
    for t_start in tl.static_range(0, C_STATIC, BLOCK_T):
        offs_tb = t_start + tl.arange(0, BLOCK_T)
        factors = tl.sum(
            tl.where(offs_t_full[None, :] == offs_tb[:, None], r_factors[None, :], 0.0),
            axis=1,
        )
        W_blk = tl.reshape(w_desc.load([BH_IDX, NC_IDX, t_start, m_start]), (BLOCK_T, BLOCK_M)).to(tl.float32)
        V_blk = tl.reshape(v_desc.load([BH_IDX, NC_IDX, t_start, d_start]), (BLOCK_T, BLOCK_D)).to(tl.float32)
        S += tl.sum((factors[:, None, None] * W_blk[:, :, None] * V_blk[:, None, :]), axis=0)

    out_s_desc.store([BH_IDX, NC_IDX, m_start, d_start], tl.reshape(S, (1, 1, BLOCK_M, BLOCK_D)))

    # alpha_chunk belongs to phase-1 scope but is computed outside this kernel.


@triton.jit
def phase1_local_chunk_end_state_bwd_dw_kernel(
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
):
    """Phase-0 backward kernel for `dW`.

    Inputs:
    - `grad_s_local_end`: `G = dL/dS_local_end`, shape `[BH, NC, M, D]`
    - `V`, `log_alpha`

    For each token `t`:

      dW_t[m] = factor[t] * sum_d G[m, d] * V_t[d]

    where:

      factor[t] = prod_{u=t+1..C-1} alpha_u where alpha_u = exp(log_alpha_u)

    Program mapping:
    - pid0 over `(BH, NC)`, pid1 over token `t`, pid2 over `M`-tiles.
    - Reduction axis is `D`.
    """
    pid_bhnc = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_m_tile = tl.program_id(2)
    if pid_bhnc >= bh_size * nc_size or pid_c >= c_size:
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
        block_shape=[1, 1, 1, BLOCK_D],
    )
    out_dw_desc = tl.make_tensor_descriptor(
        out_dw_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_out_dw_bh, stride_out_dw_nc, stride_out_dw_c, stride_out_dw_m],
        block_shape=[1, 1, 1, BLOCK_M],
    )

    offs_t = tl.arange(0, BLOCK_C)
    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    r_factors = tl.exp(log_suffix_incl - log_alpha_vals)
    c_mask = offs_t == pid_c
    factor_c = tl.sum(tl.where(c_mask, r_factors, 0.0), axis=0)

    m_start = pid_m_tile * BLOCK_M
    offs_m = pid_m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    v_dtype_probe = tl.reshape(v_desc.load([BH_IDX, NC_IDX, pid_c, 0]), (BLOCK_D,))

    d_start = 0
    while d_start < d_size:
        g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
        v_tile_in = tl.reshape(v_desc.load([BH_IDX, NC_IDX, pid_c, d_start]), (BLOCK_D,))
        v_tile = v_tile_in.to(tl.float32)
        acc += tl.sum(g_tile * v_tile[None, :], axis=1)
        d_start += BLOCK_D

    out_dw_desc.store([BH_IDX, NC_IDX, pid_c, m_start], tl.reshape((factor_c * acc).to(v_dtype_probe.dtype), (1, 1, 1, BLOCK_M)))


@triton.jit
def phase1_local_chunk_end_state_bwd_dv_kernel(
    grad_s_local_end_ptr,
    W_ptr,
    log_alpha_ptr,
    out_dv_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    d_size,
    stride_grad_s_bh,
    stride_grad_s_nc,
    stride_grad_s_m,
    stride_grad_s_d,
    stride_w_bh,
    stride_w_nc,
    stride_w_c,
    stride_w_m,
    stride_r_bh,
    stride_r_nc,
    stride_r_c,
    stride_out_dv_bh,
    stride_out_dv_nc,
    stride_out_dv_c,
    stride_out_dv_d,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
):
    """Phase-0 backward kernel for `dV`.

    Inputs:
    - `grad_s_local_end`: `G = dL/dS_local_end`, shape `[BH, NC, M, D]`
    - `W`, `log_alpha`

    For each token `t`:

      dV_t[d] = factor[t] * sum_m G[m, d] * W_t[m]

    where:

      factor[t] = prod_{u=t+1..C-1} alpha_u where alpha_u = exp(log_alpha_u)

    Program mapping:
    - pid0 over `(BH, NC)`, pid1 over token `t`, pid2 over `D`-tiles.
    - Reduction axis is `M`.
    """
    pid_bhnc = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_d_tile = tl.program_id(2)
    if pid_bhnc >= bh_size * nc_size or pid_c >= c_size:
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
        block_shape=[1, 1, 1, BLOCK_M],
    )
    out_dv_desc = tl.make_tensor_descriptor(
        out_dv_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_out_dv_bh, stride_out_dv_nc, stride_out_dv_c, stride_out_dv_d],
        block_shape=[1, 1, 1, BLOCK_D],
    )

    offs_t = tl.arange(0, BLOCK_C)
    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    r_factors = tl.exp(log_suffix_incl - log_alpha_vals)
    c_mask = offs_t == pid_c
    factor_c = tl.sum(tl.where(c_mask, r_factors, 0.0), axis=0)

    d_start = pid_d_tile * BLOCK_D
    offs_d = pid_d_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    w_dtype_probe = tl.reshape(w_desc.load([BH_IDX, NC_IDX, pid_c, 0]), (BLOCK_M,))

    m_start = 0
    while m_start < m_size:
        g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
        w_tile_in = tl.reshape(w_desc.load([BH_IDX, NC_IDX, pid_c, m_start]), (BLOCK_M,))
        w_tile = w_tile_in.to(tl.float32)
        acc += tl.sum(g_tile * w_tile[:, None], axis=0)
        m_start += BLOCK_M

    out_dv_desc.store([BH_IDX, NC_IDX, pid_c, d_start], tl.reshape((factor_c * acc).to(w_dtype_probe.dtype), (1, 1, 1, BLOCK_D)))


@triton.jit
def phase1_local_chunk_end_state_bwd_b_partials_kernel(
    grad_s_local_end_ptr,
    W_ptr,
    V_ptr,
    out_b_partials_ptr,
    bh_size,
    nc_size,
    c_size,
    m_size,
    d_size,
    n_m_tiles_size,
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
    stride_out_b_bh,
    stride_out_b_nc,
    stride_out_b_c,
    stride_out_b_tile,
    stride_out_b_vec,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Phase-0 backward helper: tiled partials for scalar `b_t`.

    b_t = sum_{m,d} G[m,d] * W_t[m] * V_t[d]

    This kernel computes one M-tile partial per program. Host code sums partials
    over tile index to recover full `b_t`.
    """
    pid_bhnc = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_m_tile = tl.program_id(2)
    if pid_bhnc >= bh_size * nc_size or pid_c >= c_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size

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
        block_shape=[1, 1, 1, BLOCK_D],
    )
    w_desc = tl.make_tensor_descriptor(
        W_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_w_bh, stride_w_nc, stride_w_c, stride_w_m],
        block_shape=[1, 1, 1, BLOCK_M],
    )
    out_b_desc = tl.make_tensor_descriptor(
        out_b_partials_ptr,
        shape=[bh_size, nc_size, c_size, n_m_tiles_size, 4],
        strides=[stride_out_b_bh, stride_out_b_nc, stride_out_b_c, stride_out_b_tile, stride_out_b_vec],
        block_shape=[1, 1, 1, 1, 4],
    )

    m_start = pid_m_tile * BLOCK_M
    acc_m = tl.zeros((BLOCK_M,), dtype=tl.float32)

    d_start = 0
    while d_start < d_size:
        g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
        v_tile = tl.reshape(v_desc.load([BH_IDX, NC_IDX, pid_c, d_start]), (BLOCK_D,)).to(tl.float32)
        acc_m += tl.sum(g_tile * v_tile[None, :], axis=1)
        d_start += BLOCK_D

    w_tile = tl.reshape(w_desc.load([BH_IDX, NC_IDX, pid_c, m_start]), (BLOCK_M,)).to(tl.float32)
    partial = tl.sum(acc_m * w_tile, axis=0)
    partial_vec = tl.zeros((4,), dtype=tl.float32) + partial
    out_b_desc.store([BH_IDX, NC_IDX, pid_c, pid_m_tile, 0], tl.reshape(partial_vec, (1, 1, 1, 1, 4)))


@triton.jit
def phase1_local_chunk_end_state_bwd_dr_kernel(
    b_vals_ptr,
    log_alpha_ptr,
    grad_r_chunk_ptr,
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
    stride_grad_r_chunk_bh,
    stride_grad_r_chunk_nc,
    stride_out_dlog_alpha_bh,
    stride_out_dlog_alpha_nc,
    stride_out_dlog_alpha_c,
    BLOCK_NC: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
):
    """Phase-1 backward kernel for `d log_alpha`.

    Inputs:
    - `b_vals`: scalar `b_t` per token
    - `log_alpha`
    - `grad_r_chunk`: upstream gradient for `alpha_chunk`

    Needed scalar recurrences for one `(BH, NC)`:

      factor[t] = prod_{u=t+1..C-1} alpha_u
      left[t]   = sum_{j=0..t-1} (prod_{u=j+1..t-1} alpha_u) * b_j
      prefix[t] = prod_{u=0..t-1} alpha_u

      d(alpha_t) = factor[t] * left[t] + grad_r_chunk * (prefix[t] * factor[t])
      d(log_alpha_t) = d(alpha_t) * alpha_t

    Interpretation:
    - First term is contribution via `S_local_end`.
    - Second term is contribution via `alpha_chunk = prod_u alpha_u = exp(sum_u log_alpha_u)`.
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
    grad_r_chunk_desc = tl.make_tensor_descriptor(
        grad_r_chunk_ptr,
        shape=[bh_size, nc_size],
        strides=[stride_grad_r_chunk_bh, stride_grad_r_chunk_nc],
        block_shape=[1, BLOCK_NC],
    )
    out_dr_desc = tl.make_tensor_descriptor(
        out_dlog_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_out_dlog_alpha_bh, stride_out_dlog_alpha_nc, stride_out_dlog_alpha_c],
        block_shape=[1, 1, BLOCK_C],
    )

    offs_t = tl.arange(0, BLOCK_C)
    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,))
    b_vals_in = tl.reshape(b_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,))
    log_alpha_f = log_alpha_vals.to(tl.float32)
    alpha_vals = tl.exp(log_alpha_f)
    b_vals = b_vals_in.to(tl.float32)
    log_suffix_incl = tl.cumsum(log_alpha_f, axis=0, reverse=True)
    factors = tl.exp(log_suffix_incl - log_alpha_f)

    nc_start = (NC_IDX // BLOCK_NC) * BLOCK_NC
    offs_nc = nc_start + tl.arange(0, BLOCK_NC)
    grad_r_chunk_vec = tl.reshape(grad_r_chunk_desc.load([BH_IDX, nc_start]), (BLOCK_NC,)).to(tl.float32)
    grad_r_chunk = tl.sum(tl.where(offs_nc == NC_IDX, grad_r_chunk_vec, 0.0), axis=0)

    left = 0.0
    prefix = 1.0
    grad_vec = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for t in tl.static_range(0, C_STATIC):
        t_mask = offs_t == t
        factor_t = tl.sum(tl.where(t_mask, factors, 0.0), axis=0)
        b_t = tl.sum(tl.where(t_mask, b_vals, 0.0), axis=0)
        d_alpha_t = factor_t * left + grad_r_chunk * (prefix * factor_t)
        alpha_t = tl.sum(tl.where(t_mask, alpha_vals, 0.0), axis=0)
        d_log_alpha_t = d_alpha_t * alpha_t
        grad_vec += tl.where(t_mask, d_log_alpha_t, 0.0)
        left = alpha_t * left + b_t
        prefix = prefix * alpha_t
    out_dr_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(grad_vec.to(log_alpha_vals.dtype), (1, 1, BLOCK_C)))


# ==================================================================================================
# PHASE 1 AUTOGRAD
# ==================================================================================================
class Phase1LocalChunkEndStateTrition(torch.autograd.Function):
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
        BLOCK_T: int = 8,
    ) -> torch.Tensor:
        if W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
            raise ValueError(
                "Phase1LocalChunkEndStateTrition.forward expects "
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
        _require_supported_md(M, D, where="Phase1LocalChunkEndStateTrition.forward")
        _require_chunk_size_multiple_of_16(C, where="Phase1LocalChunkEndStateTrition.forward")
        _require_nc_descriptor_width(NC, where="Phase1LocalChunkEndStateTrition.forward")
        _require_phase1_block_t(BLOCK_T, C, where="Phase1LocalChunkEndStateTrition.forward")
        _require_nonpositive_log_alpha(log_alpha, where="Phase1LocalChunkEndStateTrition.forward")
        if not W.is_cuda:
            raise NotImplementedError("Phase1LocalChunkEndStateTrition.forward requires CUDA tensors.")
        _ensure_triton_allocator()

        s_local_end_md = torch.empty((BH, NC, M, D), device=W.device, dtype=torch.float32)
        BLOCK_M = 16
        BLOCK_D = 32
        grid = (
            BH * NC,
            M // BLOCK_M,
            D // BLOCK_D,
        )
        phase1_local_chunk_end_state_fwd_kernel[grid](
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
        _require_supported_md(M, D, where="Phase1LocalChunkEndStateTrition.backward")

        if grad_s_local_end is None:
            grad_s_local_end = torch.zeros((BH, NC, MD), device=W.device, dtype=torch.float32)
        if grad_s_local_end.shape != (BH, NC, MD):
            raise ValueError(
                f"grad_s_local_end must be [BH,NC,MD]=({BH},{NC},{MD}); got {tuple(grad_s_local_end.shape)}."
            )

        _require_chunk_size_multiple_of_16(C, where="Phase1LocalChunkEndStateTrition.backward")
        _require_nc_descriptor_width(NC, where="Phase1LocalChunkEndStateTrition.backward")
        if not W.is_cuda:
            raise NotImplementedError("Phase1LocalChunkEndStateTrition.backward requires CUDA tensors.")
        _ensure_triton_allocator()

        grad_s_md = grad_s_local_end.float().contiguous().view(BH, NC, M, D)
        d_W = torch.empty_like(W)
        d_V = torch.empty_like(V)
        BLOCK_M = 16
        BLOCK_D = 32

        grid_dw = (
            BH * NC,
            C,
            M // BLOCK_M,
        )
        phase1_local_chunk_end_state_bwd_dw_kernel[grid_dw](
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
            num_warps=2,
            num_stages=2,
        )

        grid_dv = (
            BH * NC,
            C,
            D // BLOCK_D,
        )
        phase1_local_chunk_end_state_bwd_dv_kernel[grid_dv](
            grad_s_md,
            W,
            log_alpha,
            d_V,
            BH,
            NC,
            C,
            M,
            D,
            *grad_s_md.stride(),
            *W.stride(),
            *log_alpha.stride(),
            *d_V.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=C,
            C_STATIC=C,
            num_warps=2,
            num_stages=2,
        )

        n_m_tiles = M // BLOCK_M
        b_partials = torch.empty((BH, NC, C, n_m_tiles, 4), device=W.device, dtype=torch.float32)
        grid_b = (BH * NC, C, n_m_tiles)
        phase1_local_chunk_end_state_bwd_b_partials_kernel[grid_b](
            grad_s_md,
            W,
            V,
            b_partials,
            BH,
            NC,
            C,
            M,
            D,
            n_m_tiles,
            *grad_s_md.stride(),
            *W.stride(),
            *V.stride(),
            *b_partials.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            num_warps=2,
            num_stages=2,
        )
        b_vals = b_partials[..., 0].sum(dim=-1).contiguous()

        d_log_alpha = torch.empty_like(log_alpha)
        grid_dr = (BH * NC,)
        block_nc = NC
        grad_alpha_chunk = torch.zeros((BH, NC), device=log_alpha.device, dtype=log_alpha.dtype)
        phase1_local_chunk_end_state_bwd_dr_kernel[grid_dr](
            b_vals,
            log_alpha,
            grad_alpha_chunk,
            d_log_alpha,
            BH,
            NC,
            C,
            *b_vals.stride(),
            *log_alpha.stride(),
            *grad_alpha_chunk.stride(),
            *d_log_alpha.stride(),
            BLOCK_NC=block_nc,
            BLOCK_C=C,
            C_STATIC=C,
            num_warps=2,
            num_stages=2,
        )
        return d_W, d_V, d_log_alpha, None


def phase1_local_chunk_end_state_triton_outline(
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    BLOCK_T: int = 8,
) -> torch.Tensor:
    """Phase-1 autograd entrypoint (Triton kernel forward)."""
    return Phase1LocalChunkEndStateTrition.apply(W, V, log_alpha, BLOCK_T)


# --------------------------------------------------------------------------------------------------
# END PHASE 1: LOCAL CHUNK END-STATE COMPUTATION
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# PHASE 3: DENSE OUTPUT COMPUTATION
# --------------------------------------------------------------------------------------------------
@triton.jit
def phase3_dense_output_fwd_kernel(
    C_ptr,
    W_ptr,
    V_ptr,
    log_alpha_ptr,
    S0_ptr,
    out_y_ptr,
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

    offs_c = tl.arange(0, BLOCK_C)

    # Step 1: build dense causal factor matrix L[t, tau] in log space.
    log_alpha_vals = tl.reshape(r_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    log_p_incl = tl.cumsum(log_alpha_vals, axis=0)
    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    valid = col_idx <= row_idx
    L = tl.where(valid, tl.exp(log_p_incl[:, None] - log_p_incl[None, :]), 0.0)

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
    p = tl.exp(log_p_incl)
    Y_off = p[:, None] * Y_off_base
    Y = Y_diag + Y_off

    out_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(Y.to(V_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))


# ==================================================================================================
# PHASE 3 AUTOGRAD
# ==================================================================================================
def phase3_dense_output_backward_kernel_profile(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    grad_out: torch.Tensor,
) -> dict[str, float]:
    """Backward profiling is disabled until the dense-output backward is rebuilt."""
    del C, W, V, log_alpha, grad_out
    raise NotImplementedError(
        "Dense-output Triton backward profiling is unavailable while the backward "
        "path is being rebuilt from scratch."
    )


class Phase3DenseOutputTrition(torch.autograd.Function):
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

    Backward is intentionally disabled while the dense chunk-local backward is
    rebuilt from scratch.
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
                "Phase3DenseOutputTrition.forward expects C=[BH,NC,C,M], "
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
        _require_supported_md(M, D, where="Phase3DenseOutputTrition.forward")
        _require_chunk_size_multiple_of_16(C_CHUNK, where="Phase3DenseOutputTrition.forward")
        _require_nonpositive_log_alpha(log_alpha, where="Phase3DenseOutputTrition.forward")
        if not C.is_cuda:
            raise NotImplementedError("Phase3DenseOutputTrition.forward requires CUDA tensors.")
        _ensure_triton_allocator()

        if S0 is None:
            S0_md = torch.zeros((BH, NC, M, D), device=C.device, dtype=C.dtype)
        else:
            if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
                S0_md = S0.reshape(BH, NC, M, D).to(C.dtype)
            elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
                S0_md = S0.to(C.dtype)
            else:
                raise ValueError(
                    f"S0 must be [BH,NC,MD]=({BH},{NC},{M * D}) or [BH,NC,M,D]=({BH},{NC},{M},{D}); "
                    f"got {tuple(S0.shape)}."
                )
            if S0_md.device != C.device:
                raise ValueError("S0 must be on the same device as C/W/V/log_alpha.")

        out = torch.empty((BH, NC, C_CHUNK, D), device=C.device, dtype=C.dtype)
        BLOCK_M = min(32, M)
        BLOCK_D = min(64, D)
        grid = (BH * NC, D // BLOCK_D)
        phase3_dense_output_fwd_kernel[grid](
            C,
            W,
            V,
            log_alpha,
            S0_md,
            out,
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
        else:
            S0_saved = S0_md
        ctx.save_for_backward(C, W, V, log_alpha, S0_saved)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        C, W, V, log_alpha, _ = ctx.saved_tensors
        BH, NC, C_CHUNK, M = C.shape
        D = V.shape[-1]
        _require_supported_md(M, D, where="Phase3DenseOutputTrition.backward")

        if grad_out is None:
            grad_out = torch.zeros((BH, NC, C_CHUNK, D), device=C.device, dtype=C.dtype)
        if grad_out.shape != (BH, NC, C_CHUNK, D):
            raise ValueError(
                f"grad_out must be [BH,NC,C,D]=({BH},{NC},{C_CHUNK},{D}); got {tuple(grad_out.shape)}."
            )
        _require_chunk_size_multiple_of_16(C_CHUNK, where="Phase3DenseOutputTrition.backward")
        if not C.is_cuda:
            raise NotImplementedError("Phase3DenseOutputTrition.backward requires CUDA tensors.")

        raise NotImplementedError(
            "Dense-output Triton backward has been reset. "
            "Rebuild the new A1/A2/B scaffold before enabling this path again."
        )


def phase3_dense_output_triton_outline(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    S0: torch.Tensor | None = None,
    INPUT_PRECISION: str = "tf32",
) -> torch.Tensor:
    """Phase-3 autograd entrypoint (Triton forward)."""
    return Phase3DenseOutputTrition.apply(C, W, V, log_alpha, S0, INPUT_PRECISION)


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
def phase2_prefix_scan_fwd_kernel(
    s_local_ptr,
    r_chunk_ptr,
    init_ptr,
    out_prefix_ptr,
    out_final_ptr,
    bh_size,
    nc_size,
    md_size,
    stride_s_local_bh,
    stride_s_local_nc,
    stride_s_local_md,
    stride_r_chunk_bh,
    stride_r_chunk_nc,
    stride_init_bh,
    stride_init_md,
    stride_out_prefix_bh,
    stride_out_prefix_nc,
    stride_out_prefix_md,
    stride_out_final_bh,
    stride_out_final_md,
    BLOCK_MD: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    NC_STATIC: tl.constexpr,
):
    """Mode-0 forward prefix scan in one kernel.

    Per `(BH, MD)` lane:
      S_scan <- init[BH, MD]

      for c in [0, NC):
        out_prefix[BH, c, MD] <- S_scan                    # exclusive prefix
        S_scan <- alpha_chunk[BH, c] * S_scan + s_local[BH, c, MD]

      out_final[BH, MD] <- S_scan

    This avoids materializing `[BH, NC, MD]` broadcasted retain tensors and uses
    scalar `alpha_chunk[BH, c]` directly.
    """
    pid_bh = tl.program_id(0)
    pid_md_tile = tl.program_id(1)
    if pid_bh >= bh_size:
        return

    offs_md = pid_md_tile * BLOCK_MD + tl.arange(0, BLOCK_MD)
    mask_md = offs_md < md_size

    s_local_desc = tl.make_tensor_descriptor(
        s_local_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_s_local_bh, stride_s_local_nc, stride_s_local_md],
        block_shape=[1, 1, BLOCK_MD],
    )
    r_chunk_desc = tl.make_tensor_descriptor(
        r_chunk_ptr,
        shape=[bh_size, nc_size],
        strides=[stride_r_chunk_bh, stride_r_chunk_nc],
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
        block_shape=[1, 1, BLOCK_MD],
    )
    out_final_desc = tl.make_tensor_descriptor(
        out_final_ptr,
        shape=[bh_size, md_size],
        strides=[stride_out_final_bh, stride_out_final_md],
        block_shape=[1, BLOCK_MD],
    )

    init_vec = tl.reshape(init_desc.load([pid_bh, pid_md_tile * BLOCK_MD]), (BLOCK_MD,))
    S_scan = init_vec.to(tl.float32)
    r_vals = tl.reshape(r_chunk_desc.load([pid_bh, 0]), (BLOCK_NC,)).to(tl.float32)
    offs_nc = tl.arange(0, BLOCK_NC)

    for c in tl.static_range(0, NC_STATIC):
        out_prefix_desc.store([pid_bh, c, pid_md_tile * BLOCK_MD], tl.reshape(S_scan, (1, 1, BLOCK_MD)))
        r = tl.sum(tl.where(offs_nc == c, r_vals, 0.0), axis=0)
        s_local = tl.reshape(s_local_desc.load([pid_bh, c, pid_md_tile * BLOCK_MD]), (BLOCK_MD,)).to(tl.float32)
        S_scan = r * S_scan + s_local

    out_final_desc.store([pid_bh, pid_md_tile * BLOCK_MD], tl.reshape(S_scan, (1, BLOCK_MD)))


_PHASE2_PREFIX_SCAN_BWD_CONFIGS = [
    triton.Config({"BLOCK_MD": 64}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_MD": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_MD": 256}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_MD": 512}, num_warps=8, num_stages=3),
]


@triton.autotune(configs=_PHASE2_PREFIX_SCAN_BWD_CONFIGS, key=["md_size", "nc_size"])
@triton.jit
def phase2_prefix_scan_bwd_kernel(
    grad_prefix_ptr,
    grad_final_ptr,
    chunk_start_ptr,
    r_chunk_ptr,
    d_s_local_ptr,
    d_r_partials_ptr,
    d_init_ptr,
    bh_size,
    nc_size,
    md_size,
    n_tiles_size,
    stride_grad_prefix_bh,
    stride_grad_prefix_nc,
    stride_grad_prefix_md,
    stride_grad_final_bh,
    stride_grad_final_md,
    stride_chunk_start_bh,
    stride_chunk_start_nc,
    stride_chunk_start_md,
    stride_r_chunk_bh,
    stride_r_chunk_nc,
    stride_d_s_local_bh,
    stride_d_s_local_nc,
    stride_d_s_local_md,
    stride_d_r_partials_bh,
    stride_d_r_partials_nc,
    stride_d_r_partials_tile,
    stride_d_r_partials_vec,
    stride_d_init_bh,
    stride_d_init_md,
    BLOCK_MD: tl.constexpr,
    BLOCK_NC: tl.constexpr,
):
    """Reverse-mode for mode-0 prefix scan.

    Forward per `(BH, MD)` lane:
      S_{c+1} = alpha_chunk[c] * S_c + S_local_end[c]
      y_c = S_c
      y_final = S_NC

    Backward:
      Let g_c = dL/dy_c and g_final = dL/dy_final.
      Start g_next = g_final, then for c from NC-1 downto 0:

        dS_local_end[c] += g_next
        dr_chunk[c]     += sum_md(g_next * S_c)
        g_prev           = g_c + alpha_chunk[c] * g_next
        g_next           = g_prev

      dinit = g_next at loop end.

    This kernel computes MD-tile partials for `dr_chunk`; host-side code reduces
    partials across MD tiles after kernel launch.
    """
    pid_bh = tl.program_id(0)
    pid_md_tile = tl.program_id(1)
    if pid_bh >= bh_size:
        return

    offs_md = pid_md_tile * BLOCK_MD + tl.arange(0, BLOCK_MD)
    grad_prefix_desc = tl.make_tensor_descriptor(
        grad_prefix_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_grad_prefix_bh, stride_grad_prefix_nc, stride_grad_prefix_md],
        block_shape=[1, 1, BLOCK_MD],
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
        block_shape=[1, 1, BLOCK_MD],
    )
    r_chunk_desc = tl.make_tensor_descriptor(
        r_chunk_ptr,
        shape=[bh_size, nc_size],
        strides=[stride_r_chunk_bh, stride_r_chunk_nc],
        block_shape=[1, BLOCK_NC],
    )
    d_s_local_desc = tl.make_tensor_descriptor(
        d_s_local_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_d_s_local_bh, stride_d_s_local_nc, stride_d_s_local_md],
        block_shape=[1, 1, BLOCK_MD],
    )
    d_r_partials_desc = tl.make_tensor_descriptor(
        d_r_partials_ptr,
        shape=[bh_size, nc_size, n_tiles_size, 4],
        strides=[stride_d_r_partials_bh, stride_d_r_partials_nc, stride_d_r_partials_tile, stride_d_r_partials_vec],
        block_shape=[1, BLOCK_NC, 1, 4],
    )
    d_init_desc = tl.make_tensor_descriptor(
        d_init_ptr,
        shape=[bh_size, md_size],
        strides=[stride_d_init_bh, stride_d_init_md],
        block_shape=[1, BLOCK_MD],
    )

    grad_final_vec = tl.reshape(grad_final_desc.load([pid_bh, pid_md_tile * BLOCK_MD]), (BLOCK_MD,))
    g_next = grad_final_vec.to(tl.float32)
    r_vals = tl.reshape(r_chunk_desc.load([pid_bh, 0]), (BLOCK_NC,)).to(tl.float32)
    offs_nc = tl.arange(0, BLOCK_NC)
    dr_partials = tl.zeros((BLOCK_NC,), dtype=tl.float32)

    c = nc_size
    while c > 0:
        c = c - 1

        # dS_local_end[c, md] = g_next[md]
        d_s_local_desc.store([pid_bh, c, pid_md_tile * BLOCK_MD], tl.reshape(g_next.to(grad_final_vec.dtype), (1, 1, BLOCK_MD)))

        # dr_chunk[c] partial reduction over MD tile.
        S_c = tl.reshape(chunk_start_desc.load([pid_bh, c, pid_md_tile * BLOCK_MD]), (BLOCK_MD,)).to(tl.float32)
        dr_partial = tl.sum(g_next * S_c, axis=0)
        c_mask = offs_nc == c
        dr_partials += tl.where(c_mask, dr_partial, 0.0)

        # g_prev = g_c + r_c * g_next
        g_c = tl.reshape(grad_prefix_desc.load([pid_bh, c, pid_md_tile * BLOCK_MD]), (BLOCK_MD,)).to(tl.float32)
        r_c = tl.sum(tl.where(c_mask, r_vals, 0.0), axis=0)
        g_next = g_c + r_c * g_next

    dr_block = tl.expand_dims(dr_partials, axis=1) + tl.zeros((BLOCK_NC, 4), dtype=tl.float32)
    d_r_partials_desc.store([pid_bh, 0, pid_md_tile, 0], tl.reshape(dr_block, (1, BLOCK_NC, 1, 4)))
    d_init_desc.store([pid_bh, pid_md_tile * BLOCK_MD], tl.reshape(g_next.to(grad_final_vec.dtype), (1, BLOCK_MD)))


# ==================================================================================================
# PHASE 2 AUTOGRAD
# ==================================================================================================
class Phase2PrefixScanTrition(torch.autograd.Function):
    """Autograd contour for mode-0 chunk-prefix scan (scalar chunk retain).

    Forward:
      Inputs:
        S_local_end : [BH, NC, MD] (or [BHG, NC, MD])
        alpha_chunk     : [BH, NC] (or [BHG, NC])
        init   : [BH, MD] (or [BHG, MD])
      Outputs:
        chunk_start : [BH, NC, MD] (or [BHG, NC, MD])
        final_state : [BH, MD] (or [BHG, MD])

      Equations per `(BH, MD)` lane:
        S_end[c] = alpha_chunk[c] * S_start[c] + S_local_end[c]
        S_start[0] = init
        chunk_start[c] = S_start[c]
        final_state = S_end[NC-1]

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
        one kernel over `(BH, MD-tile)` that loops over chunks and emits
        exclusive chunk-prefix states plus final state.
      - BWD:
        a) reverse scan kernel for adjoint S accumulation over chunk axis;
        b) local grad kernel for `dr_chunk`, `dS_local_end`, and `dinit`.
    """

    @staticmethod
    def forward(
        ctx,
        S_local_end: torch.Tensor,
        alpha_chunk: torch.Tensor,
        initial_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Shape / contract checks.
        if S_local_end.ndim != 3 or alpha_chunk.ndim != 2 or initial_state.ndim != 2:
            raise ValueError(
                "Phase2PrefixScanTrition.forward expects "
                "S_local_end=[BH,NC,MD], alpha_chunk=[BH,NC], initial_state=[BH,MD]. "
                f"Got S_local_end={tuple(S_local_end.shape)}, alpha_chunk={tuple(alpha_chunk.shape)}, "
                f"initial_state={tuple(initial_state.shape)}."
            )
        BH, NC, MD = S_local_end.shape
        if alpha_chunk.shape != (BH, NC):
            raise ValueError(f"alpha_chunk must be [BH,NC]. Got {tuple(alpha_chunk.shape)} vs ({BH}, {NC}).")
        if initial_state.shape != (BH, MD):
            raise ValueError(f"initial_state must be [BH,MD]. Got {tuple(initial_state.shape)} vs ({BH}, {MD}).")
        if S_local_end.device != alpha_chunk.device or S_local_end.device != initial_state.device:
            raise ValueError("S_local_end, alpha_chunk, and initial_state must be on the same device.")
        if NC == 0:
            raise NotImplementedError("Phase2PrefixScanTrition.forward does not support NC==0.")
        _require_nc_descriptor_width(NC, where="Phase2PrefixScanTrition.forward")
        if not S_local_end.is_cuda:
            raise NotImplementedError("Phase2PrefixScanTrition.forward requires CUDA tensors.")
        _ensure_triton_allocator()

        S_local_end_f = S_local_end.float().contiguous()
        alpha_chunk_f = alpha_chunk.float().contiguous()
        initial_state_f = initial_state.float().contiguous()

        chunk_start = torch.empty((BH, NC, MD), device=S_local_end.device, dtype=torch.float32)
        final_state = torch.empty((BH, MD), device=S_local_end.device, dtype=torch.float32)
        block_nc = NC

        grid = lambda META: (BH, triton.cdiv(MD, META["BLOCK_MD"]))
        phase2_prefix_scan_fwd_kernel[grid](
            S_local_end_f,
            alpha_chunk_f,
            initial_state_f,
            chunk_start,
            final_state,
            BH,
            NC,
            MD,
            *S_local_end_f.stride(),
            *alpha_chunk_f.stride(),
            *initial_state_f.stride(),
            *chunk_start.stride(),
            *final_state.stride(),
            BLOCK_NC=block_nc,
            NC_STATIC=NC,
        )
        ctx.save_for_backward(S_local_end_f, alpha_chunk_f, initial_state_f, chunk_start, final_state)
        ctx.s_local_dtype = S_local_end.dtype
        ctx.alpha_dtype = alpha_chunk.dtype
        ctx.init_dtype = initial_state.dtype
        return chunk_start, final_state

    @staticmethod
    def backward(ctx, grad_chunk_start: torch.Tensor, grad_final_state: torch.Tensor):
        S_local_end, alpha_chunk, initial_state, chunk_start, _ = ctx.saved_tensors
        del initial_state

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
            raise NotImplementedError("Phase2PrefixScanTrition.backward does not support NC==0.")
        _require_nc_descriptor_width(NC, where="Phase2PrefixScanTrition.backward")
        if not chunk_start.is_cuda:
            raise NotImplementedError("Phase2PrefixScanTrition.backward requires CUDA tensors.")
        _ensure_triton_allocator()

        grad_chunk_start_f = grad_chunk_start.float().contiguous()
        grad_final_state_f = grad_final_state.float().contiguous()

        d_s_local = torch.empty_like(S_local_end)
        n_tiles_max = triton.cdiv(MD, 64)
        d_alpha_partials = torch.zeros((BH, NC, n_tiles_max, 4), device=chunk_start.device, dtype=torch.float32)
        d_init = torch.empty((BH, MD), device=chunk_start.device, dtype=torch.float32)
        block_nc = NC

        grid = lambda META: (BH, triton.cdiv(MD, META["BLOCK_MD"]))
        phase2_prefix_scan_bwd_kernel[grid](
            grad_chunk_start_f,
            grad_final_state_f,
            chunk_start,
            alpha_chunk,
            d_s_local,
            d_alpha_partials,
            d_init,
            BH,
            NC,
            MD,
            n_tiles_max,
            *grad_chunk_start_f.stride(),
            *grad_final_state_f.stride(),
            *chunk_start.stride(),
            *alpha_chunk.stride(),
            *d_s_local.stride(),
            *d_alpha_partials.stride(),
            *d_init.stride(),
            BLOCK_NC=block_nc,
        )

        best_cfg = getattr(phase2_prefix_scan_bwd_kernel, "best_config", None)
        d_alpha_partials_reduced = d_alpha_partials[..., 0]
        if best_cfg is not None and "BLOCK_MD" in best_cfg.kwargs:
            n_tiles_active = triton.cdiv(MD, int(best_cfg.kwargs["BLOCK_MD"]))
            d_alpha_accum = d_alpha_partials_reduced[:, :, :n_tiles_active].sum(dim=-1)
        else:
            # Fallback path should be rare; keep correctness if best_config is unavailable.
            d_alpha_accum = d_alpha_partials_reduced.sum(dim=-1)
        d_alpha = d_alpha_accum.to(ctx.alpha_dtype)
        return d_s_local.to(ctx.s_local_dtype), d_alpha, d_init.to(ctx.init_dtype)


def phase2_prefix_scan_triton_outline(
    S_local_end: torch.Tensor,
    alpha_chunk: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """High-level entry point for the chunk-prefix state scan path."""
    return Phase2PrefixScanTrition.apply(S_local_end, alpha_chunk, initial_state)


# --------------------------------------------------------------------------------------------------
# END PHASE 2: PREFIX SCAN OVER CHUNKS
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# PHASE 123: FULL PARALLEL SCAN
# --------------------------------------------------------------------------------------------------
def phase123_full_parallel_scan_trition(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
    INPUT_PRECISION: str = "tf32",
    PHASE1_BLOCK_T: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full mode-0 parallel scan from unchunked inputs via Triton phase wrappers.

    This is the Triton counterpart to `phase123_full_parallel_scan_reference`.
    Phase-1/2/3 are autograd Functions; this top-level composition stays a plain function.
    """
    if CHUNK_SIZE is None:
        CHUNK_SIZE = 64

    C_chunk, W_chunk, V_chunk, log_alpha_chunk, init_flat, B, N, H, M, D, _ = _phase123_prepare_unchunked_inputs(
        C,
        W,
        V,
        log_alpha,
        initial_state,
        where="phase123_full_parallel_scan_trition",
        CHUNK_SIZE=CHUNK_SIZE,
    )

    # =========================================
    # PHASE 1 local chunk end-state
    # =========================================
    S_local_end = phase1_local_chunk_end_state_triton_outline(
        W_chunk,
        V_chunk,
        log_alpha_chunk,
        BLOCK_T=PHASE1_BLOCK_T,
    )
    alpha_chunk = torch.exp(torch.sum(log_alpha_chunk, dim=-1))

    # =========================================
    # PHASE 2 prefix scan over chunks
    # =========================================
    S0_chunk, S1_chunk = phase2_prefix_scan_triton_outline(S_local_end, alpha_chunk, init_flat)

    # =========================================
    # PHASE 3 dense chunk-local output
    # =========================================
    y_chunk = phase3_dense_output_triton_outline(
        C_chunk,
        W_chunk,
        V_chunk,
        log_alpha_chunk,
        S0_chunk,
        INPUT_PRECISION=INPUT_PRECISION,
    )

    return _phase123_restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)


def phase123_full_parallel_scan_triton_outline(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
    INPUT_PRECISION: str = "tf32",
    PHASE1_BLOCK_T: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility alias; prefer `phase123_full_parallel_scan_trition`."""
    return phase123_full_parallel_scan_trition(
        C,
        W,
        V,
        log_alpha,
        initial_state=initial_state,
        CHUNK_SIZE=CHUNK_SIZE,
        INPUT_PRECISION=INPUT_PRECISION,
        PHASE1_BLOCK_T=PHASE1_BLOCK_T,
    )


# --------------------------------------------------------------------------------------------------
# END PHASE 123: FULL PARALLEL SCAN
# --------------------------------------------------------------------------------------------------


__all__ = [
    "phase1_local_chunk_end_state_reference",
    "phase1_local_chunk_end_state_fwd_kernel",
    "Phase1LocalChunkEndStateTrition",
    "phase1_local_chunk_end_state_triton_outline",
    "phase2_prefix_scan_reference",
    "phase2_prefix_scan_fwd_kernel",
    "phase2_prefix_scan_bwd_kernel",
    "Phase2PrefixScanTrition",
    "phase2_prefix_scan_triton_outline",
    "phase3_dense_output_reference",
    "phase3_dense_output_fwd_kernel",
    "Phase3DenseOutputTrition",
    "phase3_dense_output_triton_outline",
    "phase3_dense_output_backward_kernel_profile",
    "phase123_full_parallel_scan_reference",
    "phase123_full_token_loop_oracle",
    "phase123_full_parallel_scan_trition",
    "phase123_full_parallel_scan_triton_outline",
]

"""Experimental autoregressive FLARE variants.

Dense and recurrent paths stay available for development work, but they are not
part of the default exported API.
"""

import math
import os

import torch

from causal_flare._common import _check_finite

def _resolve_experimental_chunk_size(N: int, M: int, D_score: int, chunk_size) -> int:
    if chunk_size is None:
        env_chunk = os.environ.get("FLARE_PYTORCH_CHUNK_SIZE", "")
        chunk_size = int(env_chunk) if env_chunk else None
    if chunk_size is not None:
        return int(chunk_size)
    if D_score <= 32 and M <= 64 and N >= 1024:
        return 64
    if D_score <= 64 and M <= 128 and N >= 1024:
        return 128
    return max(64, min(2048, max(1, N // 2)))


def _affine_prefix_scan_flat(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scan_A = A
    scan_B = B
    length = A.size(1)
    step = 1
    while step < length:
        shifted_A = torch.cat([torch.ones_like(scan_A[:, :step]), scan_A[:, :-step]], dim=1)
        shifted_B = torch.cat([torch.zeros_like(scan_B[:, :step]), scan_B[:, :-step]], dim=1)
        scan_B = scan_B + scan_A * shifted_B
        scan_A = scan_A * shifted_A
        step <<= 1
    return scan_A, scan_B


def _chunkwise_affine_state_scan(A: torch.Tensor, B: torch.Tensor, initial_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if A.shape != B.shape:
        raise ValueError(f"A and B must have the same shape. Got A={tuple(A.shape)}, B={tuple(B.shape)}.")
    if A.ndim != 3 or initial_state.ndim != 2:
        raise ValueError(
            "_chunkwise_affine_state_scan expects A=[P,NC,S], B=[P,NC,S], initial_state=[P,S]. "
            f"Got A={tuple(A.shape)}, B={tuple(B.shape)}, initial_state={tuple(initial_state.shape)}."
        )
    if A.shape[0] != initial_state.shape[0] or A.shape[2] != initial_state.shape[1]:
        raise ValueError(
            "initial_state must match flattened P/S dimensions. "
            f"Got A={tuple(A.shape)}, initial_state={tuple(initial_state.shape)}."
        )
    if A.shape[1] == 0:
        return A.new_empty((A.shape[0], 0, A.shape[2])), initial_state

    inc_A, inc_B = _affine_prefix_scan_flat(A, B)
    excl_A = torch.cat([torch.ones_like(inc_A[:, :1]), inc_A[:, :-1]], dim=1)
    excl_B = torch.cat([torch.zeros_like(inc_B[:, :1]), inc_B[:, :-1]], dim=1)
    chunk_start = excl_A * initial_state.unsqueeze(1) + excl_B
    final_state = inc_A[:, -1] * initial_state + inc_B[:, -1]
    return chunk_start, final_state


def _flare_autoregressive_experimental_chunkwise_recurrence(
    W_write: torch.Tensor,
    V: torch.Tensor,
    W_read: torch.Tensor,
    W_retain: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    """Chunkwise recurrent scan with separated-style vectorized chunk processing."""
    B, N, H, M = W_write.shape
    D_value = V.size(-1)
    NC = math.ceil(N / chunk_size) if N > 0 else 0
    padded_len = NC * chunk_size
    pad = padded_len - N
    if pad > 0:
        z_w = torch.zeros((B, pad, H, M), device=W_write.device, dtype=W_write.dtype)
        z_r = torch.ones((B, pad, H), device=W_retain.device, dtype=W_retain.dtype)
        z_v = torch.zeros((B, pad, H, D_value), device=V.device, dtype=V.dtype)
        W_write = torch.cat([W_write, z_w], dim=1)
        W_read = torch.cat([W_read, z_w], dim=1)
        W_retain = torch.cat([W_retain, z_r], dim=1)
        V = torch.cat([V, z_v], dim=1)

    r = W_retain.reshape(B, NC, chunk_size, H).permute(0, 3, 1, 2).contiguous()
    w = W_write.reshape(B, NC, chunk_size, H, M).permute(0, 3, 1, 2, 4).contiguous()
    read = W_read.reshape(B, NC, chunk_size, H, M).permute(0, 3, 1, 2, 4).contiguous()
    v = V.reshape(B, NC, chunk_size, H, D_value).permute(0, 3, 1, 2, 4).contiguous()

    c_idx = torch.arange(chunk_size, device=W_write.device)
    tril = (c_idx.view(chunk_size, 1) >= c_idx.view(1, chunk_size)).view(1, 1, chunk_size, chunk_size)
    strictly_lower = (c_idx.view(chunk_size, 1) > c_idx.view(1, chunk_size)).view(1, 1, chunk_size, chunk_size)

    p = torch.cumprod(r, dim=3)
    r_row = r.unsqueeze(4).expand(B, H, NC, chunk_size, chunk_size)
    A = torch.where(strictly_lower, r_row, torch.ones_like(r_row))
    T = torch.cumprod(A, dim=3) * tril

    A_chunk = p[:, :, :, -1]
    T_last = T[:, :, :, -1, :]
    B_chunk = torch.einsum("bhnj,bhnjm,bhnjd->bhnmd", T_last, w, v)

    P = B * H
    S = M * D_value
    start_flat, _ = _chunkwise_affine_state_scan(
        A_chunk.reshape(P, NC, 1).expand(P, NC, S),
        B_chunk.reshape(P, NC, S),
        initial_state=B_chunk.new_zeros((P, S)),
    )
    start_states = start_flat.reshape(B, H, NC, M, D_value)

    rw = torch.einsum("bhncm,bhnjm->bhncj", read, w)
    y_diag = torch.einsum("bhncj,bhncj,bhnjd->bhncd", rw, T, v)
    y_off = p.unsqueeze(-1) * torch.einsum("bhncm,bhnmd->bhncd", read, start_states)
    y = (y_diag + y_off).reshape(B, H, padded_len, D_value).permute(0, 2, 1, 3).contiguous()
    return y[:, :N]


def flare_autoregressive_experimental_pytorch(
    W_write,
    V,
    W_read,
    W_retain,
    eps=None,
    profile: bool = False,
    chunk_size=None,
    state: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    return_state: bool = False,
):
    """Experimental chunkwise recurrent path with explicit retain/write weights.

    Expected tensor shapes:
    - W_write: ``[B, N, H, M]``
    - V: ``[B, N, H, D_value]``
    - W_read: ``[B, N, H, M]``
    - W_retain: ``[B, N, H]`` or ``[B, N, H, G]`` where ``G`` divides ``M``
    """
    del eps
    if profile:
        raise NotImplementedError("flare_autoregressive_experimental_pytorch does not support profile=True")
    if state is not None:
        raise NotImplementedError("flare_autoregressive_experimental_pytorch does not support recurrent state")
    if attention_mask is not None:
        raise NotImplementedError("flare_autoregressive_experimental_pytorch does not support attention_mask")
    if return_state:
        raise NotImplementedError("flare_autoregressive_experimental_pytorch does not support return_state=True")
    if W_write.dim() != 4:
        raise ValueError(f"FLARE Experimental PyTorch expected W_write with shape [B, N, H, M], got {tuple(W_write.shape)}")
    if V.dim() != 4:
        raise ValueError(f"FLARE Experimental PyTorch expected V with shape [B, N, H, D_value], got {tuple(V.shape)}")
    if W_read.dim() != 4:
        raise ValueError(f"FLARE Experimental PyTorch expected W_read with shape [B, N, H, M], got {tuple(W_read.shape)}")
    if W_retain.dim() not in {3, 4}:
        raise ValueError(
            "FLARE Experimental PyTorch expected W_retain with shape [B, N, H] or [B, N, H, G], "
            f"got {tuple(W_retain.shape)}",
        )

    if W_retain.dim() == 4:
        B, N, H, M = W_write.shape
        Bg, Ng, Hg, G = W_retain.shape
        if (Bg, Ng, Hg) != (B, N, H):
            raise ValueError(
                f"Grouped W_retain leading dims must match W_write: got W_retain={tuple(W_retain.shape)}, "
                f"W_write={tuple(W_write.shape)}"
            )
        if G <= 0:
            raise ValueError(f"Grouped W_retain must have a positive group count. Got G={G}.")
        if M % G != 0:
            raise ValueError(f"Grouped W_retain requires G to divide M. Got M={M}, G={G}.")
        slots_per_group = M // G
        W_write_grouped = W_write.view(B, N, H, G, slots_per_group).permute(0, 3, 1, 2, 4).reshape(
            B * G, N, H, slots_per_group
        )
        W_read_grouped = W_read.view(B, N, H, G, slots_per_group).permute(0, 3, 1, 2, 4).reshape(
            B * G, N, H, slots_per_group
        )
        V_grouped = V.unsqueeze(1).expand(B, G, N, H, V.shape[-1]).reshape(B * G, N, H, V.shape[-1])
        W_retain_grouped = W_retain.permute(0, 3, 1, 2).reshape(B * G, N, H)
        Y_grouped = flare_autoregressive_experimental_pytorch(
            W_write=W_write_grouped,
            V=V_grouped,
            W_read=W_read_grouped,
            W_retain=W_retain_grouped,
            eps=None,
            profile=profile,
            chunk_size=chunk_size,
            state=state,
            attention_mask=attention_mask,
            return_state=return_state,
        )
        return Y_grouped.view(B, G, N, H, -1).sum(dim=1)

    B, N, H, M = W_write.shape
    Bv, Nv, Hv, _ = V.shape
    Br, Nr, Hr, Mr = W_read.shape
    Bt, Nt, Ht = W_retain.shape
    if (Bv, Nv, Hv) != (B, N, H):
        raise ValueError(
            f"FLARE Experimental PyTorch V leading dims must match W_write: got V={tuple(V.shape)}, W_write={tuple(W_write.shape)}"
        )
    if (Br, Nr, Hr, Mr) != (B, N, H, M):
        raise ValueError(
            f"FLARE Experimental PyTorch W_read shape must match W_write in [B,N,H,M]: got W_read={tuple(W_read.shape)}, W_write={tuple(W_write.shape)}"
        )
    if (Bt, Nt, Ht) != (B, N, H):
        raise ValueError(
            f"FLARE Experimental PyTorch W_retain shape must match W_write leading dims in [B,N,H]: got W_retain={tuple(W_retain.shape)}, W_write={tuple(W_write.shape)}"
        )

    output_dtype = W_write.dtype
    C = _resolve_experimental_chunk_size(N, M, M, chunk_size)
    compute_dtype = torch.float32
    if os.environ.get("FLARE_PYTORCH_MATCH_REFERENCE", "") == "1":
        compute_dtype = W_write.dtype

    Y_f = _flare_autoregressive_experimental_chunkwise_recurrence(
        W_write=W_write.to(compute_dtype),
        V=V.to(compute_dtype),
        W_read=W_read.to(compute_dtype),
        W_retain=W_retain.to(compute_dtype),
        chunk_size=C,
    )
    Y_out = Y_f.to(output_dtype)
    _check_finite("flare_autoregressive_experimental_pytorch.Y", Y_out)
    return Y_out


def flare_autoregressive_experimental_aux_pytorch(
    W_write_s,
    V_s,
    W_read,
    W_retain_s,
    W_write_z,
    V_z,
    W_retain_z,
    W_gate_s,
    eps=None,
    profile: bool = False,
    chunk_size=None,
    state: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    return_state: bool = False,
):
    """Experimental FLARE scan with an auxiliary recurrent state.

    Recurrence per head (all retains/gates are scalars per token/head):
      Z_t = r^z_t * Z_{t-1} + (W^z_t ⊗ V^z_t)
      S_t = r^s_t * S_{t-1} + (W^s_t ⊗ V^s_t) + g^s_t * Z_t
      y_t = c_t^T S_t

    This implementation avoids token-step loops by:
    1) building chunk-local dense kernels (Ls, Lz, Ksz) with shape [C, C],
    2) evaluating in-chunk contributions with dense contractions,
    3) propagating chunk boundary states with chunk-level affine scans.

    There is still a loop over chunks in the sequence dimension in older versions;
    this current version computes all chunks in parallel and only scans across
    chunk summaries (no loop over tokens, no loop over chunks).
    """
    del eps
    if profile:
        raise NotImplementedError("flare_autoregressive_experimental_aux_pytorch does not support profile=True")
    if state is not None:
        raise NotImplementedError("flare_autoregressive_experimental_aux_pytorch does not support recurrent state")
    if attention_mask is not None:
        raise NotImplementedError("flare_autoregressive_experimental_aux_pytorch does not support attention_mask")
    if return_state:
        raise NotImplementedError("flare_autoregressive_experimental_aux_pytorch does not support return_state=True")
    for name, tensor in {
        "W_write_s": W_write_s,
        "V_s": V_s,
        "W_read": W_read,
        "W_retain_s": W_retain_s,
        "W_write_z": W_write_z,
        "V_z": V_z,
        "W_retain_z": W_retain_z,
        "W_gate_s": W_gate_s,
    }.items():
        if name in {"V_s", "V_z"} and tensor.dim() != 4:
            raise ValueError(f"FLARE Experimental Aux PyTorch expected {name} with shape [B, N, H, D], got {tuple(tensor.shape)}")
        if name not in {"V_s", "V_z"} and tensor.dim() not in {3, 4}:
            raise ValueError(f"FLARE Experimental Aux PyTorch expected {name} with rank 3/4, got {tuple(tensor.shape)}")

    if W_write_s.dim() != 4 or W_write_z.dim() != 4 or W_read.dim() != 4:
        raise ValueError("Aux write/read tensors must be [B, N, H, M].")
    if W_retain_s.dim() == 4:
        if W_retain_s.shape[-1] != 1:
            raise ValueError("Aux retain/gate use per-head scalars. Set num_groups=1 for aux mode.")
        W_retain_s = W_retain_s.squeeze(-1)
    if W_retain_z.dim() == 4:
        if W_retain_z.shape[-1] != 1:
            raise ValueError("Aux retain/gate use per-head scalars. Set num_groups=1 for aux mode.")
        W_retain_z = W_retain_z.squeeze(-1)
    if W_gate_s.dim() == 4:
        if W_gate_s.shape[-1] != 1:
            raise ValueError("Aux retain/gate use per-head scalars. Set num_groups=1 for aux mode.")
        W_gate_s = W_gate_s.squeeze(-1)
    if W_retain_s.dim() != 3 or W_retain_z.dim() != 3 or W_gate_s.dim() != 3:
        raise ValueError("Aux retain/gate tensors must be [B, N, H] (or [B, N, H, 1]).")

    B, N, H, M = W_write_s.shape
    if W_write_z.shape != (B, N, H, M) or W_read.shape != (B, N, H, M):
        raise ValueError("Aux write/read tensors must share shape [B, N, H, M].")
    if V_s.shape[:3] != (B, N, H) or V_z.shape[:3] != (B, N, H):
        raise ValueError("Aux value tensors must share leading [B, N, H] dims with write/read tensors.")
    if W_retain_s.shape != (B, N, H) or W_retain_z.shape != (B, N, H) or W_gate_s.shape != (B, N, H):
        raise ValueError("Aux retain/gate tensors must have shape [B, N, H].")

    # Keep output in the input dtype; optionally run aux math in fp32 for debugging.
    output_dtype = W_write_s.dtype
    compute_dtype = W_write_s.dtype
    if os.environ.get("FLARE_PYTORCH_AUX_FP32", "") == "1":
        compute_dtype = torch.float32

    W_write_s = W_write_s.to(compute_dtype)
    V_s = V_s.to(compute_dtype)
    W_read = W_read.to(compute_dtype)
    W_retain_s = W_retain_s.to(compute_dtype)
    W_write_z = W_write_z.to(compute_dtype)
    V_z = V_z.to(compute_dtype)
    W_retain_z = W_retain_z.to(compute_dtype)
    W_gate_s = W_gate_s.to(compute_dtype)

    # B: batch, N: seq, H: heads, M: slots, D: value dim.
    D = V_s.shape[-1]
    C = _resolve_experimental_chunk_size(N, M, M, chunk_size)
    NC = math.ceil(N / C) if N > 0 else 0
    padded_len = NC * C
    pad = padded_len - N
    # Right-pad to a multiple of chunk size so every chunk has identical shape.
    if pad > 0:
        z_w = torch.zeros((B, pad, H, M), device=W_write_s.device, dtype=W_write_s.dtype)
        z_v = torch.zeros((B, pad, H, D), device=V_s.device, dtype=V_s.dtype)
        z_r = torch.ones((B, pad, H), device=W_retain_s.device, dtype=W_retain_s.dtype)
        z_g = torch.zeros((B, pad, H), device=W_gate_s.device, dtype=W_gate_s.dtype)
        W_write_s = torch.cat([W_write_s, z_w], dim=1)
        W_write_z = torch.cat([W_write_z, z_w], dim=1)
        W_read = torch.cat([W_read, z_w], dim=1)
        V_s = torch.cat([V_s, z_v], dim=1)
        V_z = torch.cat([V_z, z_v], dim=1)
        W_retain_s = torch.cat([W_retain_s, z_r], dim=1)
        W_retain_z = torch.cat([W_retain_z, z_r], dim=1)
        W_gate_s = torch.cat([W_gate_s, z_g], dim=1)

    if NC == 0:
        return W_write_s.new_zeros((B, 0, H, D), dtype=output_dtype)

    # Layout used below:
    # - write/read tensors: [B, H, NC, C, M]
    # - value tensors:      [B, H, NC, C, D]
    # - retain/gate:        [B, H, NC, C]
    write_s = W_write_s.reshape(B, NC, C, H, M).permute(0, 3, 1, 2, 4).contiguous()  # [B,H,NC,C,M]
    write_z = W_write_z.reshape(B, NC, C, H, M).permute(0, 3, 1, 2, 4).contiguous()
    read = W_read.reshape(B, NC, C, H, M).permute(0, 3, 1, 2, 4).contiguous()
    value_s = V_s.reshape(B, NC, C, H, D).permute(0, 3, 1, 2, 4).contiguous()  # [B,H,NC,C,D]
    value_z = V_z.reshape(B, NC, C, H, D).permute(0, 3, 1, 2, 4).contiguous()
    retain_s = W_retain_s.reshape(B, NC, C, H).permute(0, 3, 1, 2).contiguous()  # [B,H,NC,C]
    retain_z = W_retain_z.reshape(B, NC, C, H).permute(0, 3, 1, 2).contiguous()
    gate_s = W_gate_s.reshape(B, NC, C, H).permute(0, 3, 1, 2).contiguous()

    # Causal lower-triangular masks used to build per-chunk transport matrices.
    c_idx = torch.arange(C, device=W_write_s.device)
    tril = (c_idx.view(C, 1) >= c_idx.view(1, C)).view(1, 1, C, C)
    strictly_lower = (c_idx.view(C, 1) > c_idx.view(1, C)).view(1, 1, C, C)

    # Build scalar transport kernels for S and Z inside each chunk:
    # Ls[i,j] = prod_{u=j+1..i} r^s_u, Lz analogously, with causal masking.
    ones = torch.ones((B, H, NC, C, C), device=W_write_s.device, dtype=compute_dtype)
    As = torch.where(strictly_lower, retain_s.unsqueeze(-1).expand(B, H, NC, C, C), ones)
    Az = torch.where(strictly_lower, retain_z.unsqueeze(-1).expand(B, H, NC, C, C), ones)
    Ls = torch.cumprod(As, dim=3) * tril
    Lz = torch.cumprod(Az, dim=3) * tril
    ps = torch.cumprod(retain_s, dim=3)  # [B,H,NC,C]
    pz = torch.cumprod(retain_z, dim=3)
    # Cross-kernel from Z writes into S readout path:
    # Ksz = Ls * diag(g^s) * Lz (implemented with broadcast + matmul).
    Ksz = torch.matmul(Ls * gate_s.unsqueeze(-2), Lz)  # [B,H,NC,C,C]
    # Prefix cross carry from chunk entry Z state to each local token.
    k_pref = torch.matmul(Ls, (gate_s * pz).unsqueeze(-1)).squeeze(-1)  # [B,H,NC,C]

    # Slot overlap terms between read coefficients c_t and write coefficients W_t.
    Rs = torch.einsum("bhnim,bhnjm->bhnij", read, write_s)
    Rz = torch.einsum("bhnim,bhnjm->bhnij", read, write_z)
    # In-chunk contribution using only local writes/values within this chunk.
    Y_diag = torch.einsum("bhnij,bhnij,bhnjd->bhnid", Rs, Ls, value_s) + torch.einsum("bhnij,bhnij,bhnjd->bhnid", Rz, Ksz, value_z)

    # Chunk summaries for boundary-state scan.
    # End-of-chunk summaries define an affine map from chunk-entry states:
    #   Z_out = pz_end * Z_in + Bz_end
    #   S_out = ps_end * S_in + k_end * Z_in + Bs_end
    Ls_end = Ls[..., -1, :]  # [B,H,NC,C]
    Lz_end = Lz[..., -1, :]
    Ksz_end = Ksz[..., -1, :]
    Bs_end = torch.einsum("bhnj,bhnjm,bhnjd->bhnmd", Ls_end, write_s, value_s) + torch.einsum(
        "bhnj,bhnjm,bhnjd->bhnmd", Ksz_end, write_z, value_z
    )
    Bz_end = torch.einsum("bhnj,bhnjm,bhnjd->bhnmd", Lz_end, write_z, value_z)
    ps_end = ps[..., -1]  # [B,H,NC]
    pz_end = pz[..., -1]
    k_end = k_pref[..., -1]

    # Flatten [B,H] so we can reuse the scalar chunkwise affine scan helper.
    P = B * H
    Sdim = M * D
    z_init = torch.zeros((P, Sdim), device=W_write_s.device, dtype=compute_dtype)
    # Z chunk-entry state for every chunk.
    z_start_flat, _ = _chunkwise_affine_state_scan(
        pz_end.reshape(P, NC, 1).expand(P, NC, Sdim),
        Bz_end.reshape(P, NC, Sdim),
        initial_state=z_init,
    )
    z_start = z_start_flat.reshape(B, H, NC, M, D)

    # Given Z chunk-entry states, build S affine bias and scan S chunk-entry states.
    s_bias = (k_end.unsqueeze(-1).unsqueeze(-1) * z_start + Bs_end).reshape(P, NC, Sdim)
    s_start_flat, _ = _chunkwise_affine_state_scan(
        ps_end.reshape(P, NC, 1).expand(P, NC, Sdim),
        s_bias,
        initial_state=z_init,
    )
    s_start = s_start_flat.reshape(B, H, NC, M, D)

    # Off-chunk contribution from chunk-entry S/Z states into each local token.
    Y_off = (
        ps.unsqueeze(-1) * torch.einsum("bhncm,bhnmd->bhncd", read, s_start)
        + k_pref.unsqueeze(-1) * torch.einsum("bhncm,bhnmd->bhncd", read, z_start)
    )
    Y = Y_diag + Y_off
    Y = Y.reshape(B, H, padded_len, D).permute(0, 2, 1, 3).contiguous()
    Y = Y[:, :N].to(output_dtype)
    _check_finite("flare_autoregressive_experimental_aux_pytorch.Y", Y)
    return Y


__all__ = [
    "flare_autoregressive_experimental_aux_pytorch",
    "flare_autoregressive_experimental_pytorch",
]

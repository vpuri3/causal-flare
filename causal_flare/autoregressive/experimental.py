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
    power: float = 2.0,
    state: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    return_state: bool = False,
):
    """Experimental chunkwise recurrent path with explicit retain/write weights.

    Expected tensor shapes:
    - W_write: ``[B, N, H, M]``
    - V: ``[B, N, H, D_value]``
    - W_read: ``[B, N, H, M]``
    - W_retain: ``[B, N, H]``
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
    if W_retain.dim() != 3:
        raise ValueError(f"FLARE Experimental PyTorch expected W_retain with shape [B, N, H], got {tuple(W_retain.shape)}")

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

    del power
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
    Y_out = Y_f.to(V.dtype)
    _check_finite("flare_autoregressive_experimental_pytorch.Y", Y_out)
    return Y_out


__all__ = [
    "flare_autoregressive_experimental_pytorch",
]

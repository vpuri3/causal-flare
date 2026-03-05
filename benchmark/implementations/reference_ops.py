import torch
import torch.nn.functional as F

from causal_flare._common import _resolve_attn_scale


def flare_noncausal(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float | None = None) -> torch.Tensor:
    if Q.dim() != 3 or K.dim() != 4 or V.dim() != 4:
        raise ValueError(
            "flare_noncausal expects Q [H, M, D] and K/V [B, N, H, D]. "
            f"Got Q.dim={Q.dim()}, K.dim={K.dim()}, V.dim={V.dim()}"
        )
    if K.size() != V.size():
        raise ValueError(f"K and V must have the same shape. Got K={tuple(K.shape)}, V={tuple(V.shape)}")
    if Q.size(0) != K.size(2) or Q.size(2) != K.size(3):
        raise ValueError(
            "Expected Q [H, M, D] and K/V [B, N, H, D] with matching H,D. "
            f"Got Q={tuple(Q.shape)}, K={tuple(K.shape)}"
        )

    B, _, _, D = K.size()
    scale = _resolve_attn_scale(scale, D)
    Q = Q.unsqueeze(0).expand(B, -1, -1, -1)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    Y = F.scaled_dot_product_attention(Q, K, V, is_causal=False, scale=scale)
    _ = F.scaled_dot_product_attention(K, Q, Y, is_causal=False, scale=scale)
    return Y


def causal_sdpa(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
        raise ValueError(
            "causal_sdpa expects Q, K, V all as [B, N, H, D] for benchmarking. "
            f"Got Q.dim={Q.dim()}, K.dim={K.dim()}, V.dim={V.dim()}"
        )
    if Q.size() != K.size() or K.size() != V.size():
        raise ValueError(f"Q, K, V must have the same shape. Got Q={tuple(Q.shape)}, K={tuple(K.shape)}, V={tuple(V.shape)}")
    Q = Q.permute(0, 2, 1, 3)
    K = K.permute(0, 2, 1, 3)
    V = V.permute(0, 2, 1, 3)
    return F.scaled_dot_product_attention(Q, K, V, is_causal=True)


# Backward-compatible alias used by existing diagnostics script.
causal_SDPA = causal_sdpa

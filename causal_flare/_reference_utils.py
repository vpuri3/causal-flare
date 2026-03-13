from ._common import *


def validate_flare_qkv_layouts(Q, K, V, *, name: str):
    if Q.dim() != 3 or K.dim() != 4 or V.dim() != 4:
        raise ValueError(
            f"{name} expects Q [H, M, D_k] and K/V [B, N, H, D]. "
            f"Got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
        )
    if K.shape[:3] != V.shape[:3]:
        raise ValueError(
            f"{name} expects K and V to agree on [B, N, H]. Got K.shape={K.shape} and V.shape={V.shape}"
        )
    if Q.size(0) != K.size(2) or Q.size(2) != K.size(3):
        raise ValueError(
            f"{name} expects Q [H, M, D_k] and K [B, N, H, D_k] with matching H,D_k. "
            f"Got Q.shape={Q.shape} and K.shape={K.shape}"
        )
    H, M, D_score = Q.shape
    B, N, _, D_value = V.shape
    return B, N, H, M, D_score, D_value


def resolve_flare_causal_decode_inputs(Q_enc, K_enc, Q_dec=None, K_dec=None):
    separate_Q_dec = Q_dec is not None
    separate_K_dec = K_dec is not None
    use_default_q_dec = (not separate_Q_dec) or (Q_dec is K_enc)
    use_default_k_dec = (not separate_K_dec) or (K_dec is Q_enc)
    weight_sharing_enc_dec = use_default_q_dec and use_default_k_dec

    if separate_Q_dec:
        if Q_dec.dim() != 4:
            raise ValueError(
                "Expected Q_dec [B, N, H, D]. "
                f"Got Q_dec.dim()={Q_dec.dim()} and Q_dec.shape={tuple(Q_dec.shape)}"
            )
        if Q_dec.size() != K_enc.size():
            raise ValueError(
                f"Q_dec and K_enc must have the same shape. Got Q_dec.shape={Q_dec.shape} and K_enc.shape={K_enc.shape}"
            )
    else:
        Q_dec = K_enc

    if separate_K_dec:
        if K_dec.dim() != 3:
            raise ValueError(
                "Expected K_dec [H, M, D]. "
                f"Got K_dec.dim()={K_dec.dim()} and K_dec.shape={tuple(K_dec.shape)}"
            )
        if K_dec.size() != Q_enc.size():
            raise ValueError(
                f"K_dec and Q_enc must have the same shape. Got K_dec.shape={K_dec.shape} and Q_enc.shape={Q_enc.shape}"
            )
    else:
        K_dec = Q_enc

    return Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec

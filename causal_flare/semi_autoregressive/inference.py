import torch
import torch.autograd as autograd


class SemiAutoRegressivePrefillFLARE(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, state=None, scale=None, attention_mask=None, Q_dec=None, K_dec=None):
        del ctx, Q, K, V, state, scale, attention_mask, Q_dec, K_dec
        raise NotImplementedError("SemiAutoRegressivePrefillFLARE is not implemented yet.")

    @staticmethod
    def backward(ctx, dY, dState=None):
        del ctx, dY, dState
        raise NotImplementedError("SemiAutoRegressivePrefillFLARE backward is not implemented yet.")


class SemiAutoRegressiveDecodeFLARE(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, state=None, scale=None, Q_dec=None, K_dec=None):
        del ctx, Q, K, V, state, scale, Q_dec, K_dec
        raise NotImplementedError("SemiAutoRegressiveDecodeFLARE is not implemented yet.")

    @staticmethod
    def backward(ctx, dY, dState=None):
        del ctx, dY, dState
        raise NotImplementedError("SemiAutoRegressiveDecodeFLARE backward is not implemented yet.")


def flare_semi_autoregressive_prefill_trition(Q, K, V, *, state=None, scale=None, attention_mask=None, Q_dec=None, K_dec=None):
    del Q, K, V, state, scale, attention_mask, Q_dec, K_dec
    raise NotImplementedError("flare_semi_autoregressive_prefill_trition is not implemented yet.")


def flare_semi_autoregressive_decode_trition(Q, K, V, *, state=None, scale=None, Q_dec=None, K_dec=None):
    del Q, K, V, state, scale, Q_dec, K_dec
    raise NotImplementedError("flare_semi_autoregressive_decode_trition is not implemented yet.")

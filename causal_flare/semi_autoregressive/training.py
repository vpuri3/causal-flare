import torch
import torch.autograd as autograd


class SemiAutoRegressiveFLARE(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale=None, block_size=None, chunk_size=None, Q_dec=None, K_dec=None):
        del ctx, Q, K, V, scale, block_size, chunk_size, Q_dec, K_dec
        raise NotImplementedError("SemiAutoRegressiveFLARE training is not implemented yet.")

    @staticmethod
    def backward(ctx, dY):
        del ctx, dY
        raise NotImplementedError("SemiAutoRegressiveFLARE training backward is not implemented yet.")


def flare_semi_autoregressive_trition(Q, K, V, *, block_size, chunk_size, scale=None, Q_dec=None, K_dec=None):
    del Q, K, V, block_size, chunk_size, scale, Q_dec, K_dec
    raise NotImplementedError("flare_semi_autoregressive_trition is not implemented yet.")

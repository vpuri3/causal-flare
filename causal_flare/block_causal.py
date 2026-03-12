import math
import time

import torch
import torch.autograd as autograd
import torch.nn.functional as F

from ._common import _check_finite, _resolve_attn_scale
from .torch import _resolve_flare_causal_decode_inputs, _validate_flare_qkv_layouts

try:
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
except ImportError:  # pragma: no cover - older PyTorch
    create_block_mask = None
    flex_attention = None


_SUPPORTED_BLOCK_CAUSAL_CHUNK_SIZES = (16, 32, 64, 128)


def _validate_block_causal_config(*, N: int, block_size, chunk_size, name: str):
    if block_size is None:
        raise ValueError(f"{name} requires block_size to be specified explicitly.")
    if chunk_size is None:
        raise ValueError(f"{name} requires chunk_size to be specified explicitly.")

    block_size = int(block_size)
    chunk_size = int(chunk_size)

    if block_size <= 0:
        raise ValueError(f"{name} requires block_size > 0. Got block_size={block_size}.")
    if block_size % 16 != 0:
        raise ValueError(
            f"{name} requires block_size to be a positive multiple of 16. Got block_size={block_size}."
        )
    if chunk_size not in _SUPPORTED_BLOCK_CAUSAL_CHUNK_SIZES:
        raise ValueError(
            f"{name} currently supports chunk_size in {_SUPPORTED_BLOCK_CAUSAL_CHUNK_SIZES}. "
            f"Got chunk_size={chunk_size}."
        )
    if N % block_size != 0:
        raise ValueError(
            f"{name} requires sequence length N to be an exact multiple of block_size. "
            f"Got N={N}, block_size={block_size}."
        )
    if block_size % chunk_size != 0:
        raise ValueError(
            f"{name} requires block_size to be an integer multiple of chunk_size so blocks align with chunk summaries. "
            f"Got block_size={block_size}, chunk_size={chunk_size}."
        )

    num_blocks = math.ceil(N / block_size) if N > 0 else 0
    num_chunks = math.ceil(N / chunk_size) if N > 0 else 0
    chunks_per_block = block_size // chunk_size
    return block_size, chunk_size, num_blocks, num_chunks, chunks_per_block


def _block_query_limits(length: int, block_size: int, device: torch.device) -> torch.Tensor:
    q_idx = torch.arange(length, device=device)
    return torch.clamp(((q_idx // block_size) + 1) * block_size, max=length)


def _block_causal_mask(length: int, block_size: int, device: torch.device) -> torch.Tensor:
    if length == 0:
        return torch.zeros((0, 0), device=device, dtype=torch.bool)
    kv_idx = torch.arange(length, device=device)
    limits = _block_query_limits(length, block_size, device)
    return kv_idx.unsqueeze(0) < limits.unsqueeze(1)


def _block_causal_phase1(score_chunk: torch.Tensor, Vc: torch.Tensor, *, chunks_per_block: int):
    # Phase 1 keeps chunks independent: for each (block, local_chunk) pair we
    # compute the local online-softmax sufficient statistics over that chunk
    # alone. There is intentionally no interaction across chunks here.
    B, H, NB, CPB, C, M = score_chunk.shape
    D_value = Vc.size(-1)
    bhnc = B * H * NB * CPB

    chunk_max = score_chunk.max(dim=4).values
    safe_chunk_max = torch.where(torch.isfinite(chunk_max), chunk_max, 0.0)
    chunk_exp = torch.where(
        torch.isfinite(score_chunk),
        torch.exp(score_chunk - safe_chunk_max.unsqueeze(4)),
        0.0,
    )
    chunk_den = chunk_exp.sum(dim=4)
    chunk_num = torch.bmm(
        chunk_exp.reshape(bhnc, C, M).transpose(1, 2),
        Vc.reshape(bhnc, C, D_value),
    ).reshape(B, H, NB, CPB, M, D_value)

    return {
        "chunk_max": chunk_max,
        "chunk_den": chunk_den,
        "chunk_num": chunk_num,
    }


def _block_causal_phase2a(chunk_max: torch.Tensor, chunk_den: torch.Tensor, chunk_num: torch.Tensor):
    # Phase 2a merges the chunk-local statistics within each block, producing
    # one block-local softmax summary per latent.
    B, H, NB, CPB, M = chunk_max.shape
    D_value = chunk_num.size(-1)
    block_max = torch.empty((B, H, NB, M), device=chunk_max.device, dtype=chunk_max.dtype)
    block_den = torch.empty((B, H, NB, M), device=chunk_den.device, dtype=chunk_den.dtype)
    block_num = torch.empty((B, H, NB, M, D_value), device=chunk_num.device, dtype=chunk_num.dtype)

    for block_idx in range(NB):
        max_curr = torch.full((B, H, M), -float("inf"), device=chunk_max.device, dtype=chunk_max.dtype)
        den_curr = torch.zeros((B, H, M), device=chunk_den.device, dtype=chunk_den.dtype)
        num_curr = torch.zeros((B, H, M, D_value), device=chunk_num.device, dtype=chunk_num.dtype)
        for local_chunk in range(CPB):
            cm = chunk_max[:, :, block_idx, local_chunk, :]
            cd = chunk_den[:, :, block_idx, local_chunk, :]
            cn = chunk_num[:, :, block_idx, local_chunk, :, :]
            max_new = torch.maximum(max_curr, cm)
            rescale_prev = torch.exp(max_curr - max_new)
            rescale_curr = torch.exp(cm - max_new)
            den_curr = den_curr * rescale_prev + cd * rescale_curr
            num_curr = num_curr * rescale_prev.unsqueeze(-1) + cn * rescale_curr.unsqueeze(-1)
            max_curr = max_new
        block_max[:, :, block_idx, :] = max_curr
        block_den[:, :, block_idx, :] = den_curr
        block_num[:, :, block_idx, :, :] = num_curr
    return block_max, block_den, block_num


def _block_causal_phase2b(block_max: torch.Tensor, block_den: torch.Tensor, block_num: torch.Tensor):
    # Phase 2b performs the prefix scan over blocks. prefix_*[blk] is the
    # encoder state contributed by all blocks strictly before blk.
    B, H, NB, M = block_max.shape
    D_value = block_num.size(-1)
    prefix_max = torch.empty_like(block_max)
    prefix_den = torch.empty_like(block_den)
    prefix_num = torch.empty_like(block_num)

    max_curr = torch.full((B, H, M), -float("inf"), device=block_max.device, dtype=block_max.dtype)
    den_curr = torch.zeros((B, H, M), device=block_max.device, dtype=block_max.dtype)
    num_curr = torch.zeros((B, H, M, D_value), device=block_max.device, dtype=block_max.dtype)

    for block_idx in range(NB):
        prefix_max[:, :, block_idx, :] = max_curr
        prefix_den[:, :, block_idx, :] = den_curr
        prefix_num[:, :, block_idx, :, :] = num_curr

        bm = block_max[:, :, block_idx, :]
        bd = block_den[:, :, block_idx, :]
        bn = block_num[:, :, block_idx, :, :]
        max_new = torch.maximum(max_curr, bm)
        rescale_prev = torch.exp(max_curr - max_new)
        rescale_block = torch.exp(bm - max_new)
        den_curr = den_curr * rescale_prev + bd * rescale_block
        num_curr = num_curr * rescale_prev.unsqueeze(-1) + bn * rescale_block.unsqueeze(-1)
        max_curr = max_new

    return prefix_max, prefix_den, prefix_num


def _block_causal_forward_torch(
    Q,
    K,
    V,
    *,
    block_size,
    chunk_size,
    scale=None,
    Q_dec=None,
    K_dec=None,
    return_aux: bool = False,
):
    B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="Block-Causal FLARE")
    block_size, chunk_size, _, _, chunks_per_block = _validate_block_causal_config(
        N=N,
        block_size=block_size,
        chunk_size=chunk_size,
        name="Block-Causal FLARE",
    )
    scale = _resolve_attn_scale(scale, D_score)
    Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_flare_causal_decode_inputs(
        Q, K, Q_dec, K_dec
    )

    compute_dtype = torch.float32
    out_dtype = V.dtype
    num_blocks = N // block_size
    chunks_per_block = block_size // chunk_size

    Q_f = Q.to(compute_dtype)
    K_f = K.to(compute_dtype)
    V_f = V.to(compute_dtype)

    if weight_sharing_enc_dec:
        Q_dec_f = None
        K_dec_f = None
    else:
        q_dec_comp = Q_dec.to(compute_dtype) if separate_Q_dec else K_f
        k_dec_comp = K_dec.to(compute_dtype) if separate_K_dec else Q_f
        Q_dec_f = q_dec_comp.permute(0, 2, 1, 3).contiguous()
        K_dec_f = k_dec_comp.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    # ------------------------------------------------------------------ #
    # Phase 0: block-major reshape and encoder score preparation.
    #
    # We reshape K/V into
    #   [B, H, NUM_BLOCKS, NUM_CHUNKS_PER_BLOCK, CHUNK_SIZE, D]
    # so all later phases can reason about chunk-local work nested inside a
    # block. The encoder scores then have shape
    #   [B, H, NUM_BLOCKS, NUM_CHUNKS_PER_BLOCK, CHUNK_SIZE, M].
    # ------------------------------------------------------------------ #
    Kc = K_f.reshape(B, num_blocks, chunks_per_block, chunk_size, H, D_score).permute(0, 4, 1, 2, 3, 5).contiguous()
    Vc = V_f.reshape(B, num_blocks, chunks_per_block, chunk_size, H, D_value).permute(0, 4, 1, 2, 3, 5).contiguous()
    score_chunk = scale * torch.einsum("bhgxcd,hmd->bhgxcm", Kc, Q_f)

    # ------------------------------------------------------------------ #
    # Phase 1: chunk-local statistics only.
    #
    # Each chunk produces its own (max, denominator, numerator) summary:
    #   chunk_max : [B, H, NUM_BLOCKS, NUM_CHUNKS_PER_BLOCK, M]
    #   chunk_den : [B, H, NUM_BLOCKS, NUM_CHUNKS_PER_BLOCK, M]
    #   chunk_num : [B, H, NUM_BLOCKS, NUM_CHUNKS_PER_BLOCK, M, D]
    # No chunk-to-chunk merging happens in this phase.
    # ------------------------------------------------------------------ #
    phase1 = _block_causal_phase1(score_chunk, Vc, chunks_per_block=chunks_per_block)

    # ------------------------------------------------------------------ #
    # Phase 2a: reduce chunk-local statistics into one summary per block.
    #
    # This merges the NUM_CHUNKS_PER_BLOCK independent chunk summaries into
    # block-local (max, denominator, numerator) tensors:
    #   block_max : [B, H, NUM_BLOCKS, M]
    #   block_den : [B, H, NUM_BLOCKS, M]
    #   block_num : [B, H, NUM_BLOCKS, M, D]
    # ------------------------------------------------------------------ #
    block_max, block_den, block_num = _block_causal_phase2a(
        phase1["chunk_max"], phase1["chunk_den"], phase1["chunk_num"]
    )

    # ------------------------------------------------------------------ #
    # Phase 2b: prefix scan over block-local statistics.
    #
    # prefix_*[blk] contains the encoder state from all blocks < blk. This is
    # the state used to inject "all previous blocks" into the outputs for the
    # current block.
    # ------------------------------------------------------------------ #
    prefix_max, prefix_den, prefix_num = _block_causal_phase2b(block_max, block_den, block_num)

    # ------------------------------------------------------------------ #
    # Phase 3a: compute LSE_enc and LSE_dec.
    #
    # First compute the full block-visible encoder normalization LSE_enc for
    # each block. Then compute the decoder normalization LSE_dec separately for
    # each output chunk in that block. The output contraction happens later in
    # phase 3b using these saved normalizers.
    # ------------------------------------------------------------------ #
    full_block_max = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=compute_dtype)
    full_block_den = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=compute_dtype)
    LSE_dec = torch.empty((B, H, N), device=Q.device, dtype=compute_dtype)
    for block_idx in range(num_blocks):
        max_curr = prefix_max[:, :, block_idx, :]
        den_curr = prefix_den[:, :, block_idx, :]
        for local_chunk in range(chunks_per_block):
            cm = phase1["chunk_max"][:, :, block_idx, local_chunk, :]
            cd = phase1["chunk_den"][:, :, block_idx, local_chunk, :]
            max_new = torch.maximum(max_curr, cm)
            rescale_prev = torch.exp(max_curr - max_new)
            rescale_curr = torch.exp(cm - max_new)
            den_curr = den_curr * rescale_prev + cd * rescale_curr
            max_curr = max_new
        full_block_max[:, :, block_idx, :] = max_curr
        full_block_den[:, :, block_idx, :] = den_curr

        for local_q_chunk in range(chunks_per_block):
            token_start = (block_idx * chunks_per_block + local_q_chunk) * chunk_size
            if weight_sharing_enc_dec:
                dec_scores = score_chunk[:, :, block_idx, local_q_chunk, :, :]
            else:
                q_t_dec = Q_dec_f[:, :, token_start : token_start + chunk_size, :]
                dec_scores = torch.einsum("bhcd,bhmd->bhcm", q_t_dec, K_dec_f) * scale
            LSE_dec[:, :, token_start : token_start + chunk_size] = torch.logsumexp(dec_scores, dim=-1)

    LSE_enc = torch.log(full_block_den) + full_block_max

    # ------------------------------------------------------------------ #
    # Phase 3b: compute outputs block-by-block.
    #
    # With LSE_enc and LSE_dec fixed, each block forms its prefix contribution
    # from earlier blocks and then accumulates dense contributions from chunks
    # inside the current block.
    # ------------------------------------------------------------------ #
    Yc = torch.empty((B, H, num_blocks, chunks_per_block, chunk_size, D_value), device=Q.device, dtype=compute_dtype)

    for block_idx in range(num_blocks):
        prefix_block_num = prefix_num[:, :, block_idx, :, :]
        lse_enc_block = LSE_enc[:, :, block_idx, :]
        prefix_scale = torch.exp(prefix_max[:, :, block_idx, :] - lse_enc_block).unsqueeze(-1)
        prefix_value = prefix_block_num * prefix_scale

        for local_q_chunk in range(chunks_per_block):
            token_start = (block_idx * chunks_per_block + local_q_chunk) * chunk_size

            if weight_sharing_enc_dec:
                dec_scores = score_chunk[:, :, block_idx, local_q_chunk, :, :]
            else:
                q_t_dec = Q_dec_f[:, :, token_start : token_start + chunk_size, :]
                dec_scores = torch.einsum("bhcd,bhmd->bhcm", q_t_dec, K_dec_f) * scale
            lse_dec_chunk = LSE_dec[:, :, token_start : token_start + chunk_size]
            alpha = torch.exp(dec_scores - lse_dec_chunk.unsqueeze(-1))

            y_chunk = torch.einsum("bhcm,bhmd->bhcd", alpha, prefix_value)
            for local_src_chunk in range(chunks_per_block):
                enc_weights = torch.exp(score_chunk[:, :, block_idx, local_src_chunk, :, :] - lse_enc_block.unsqueeze(2))
                beta = torch.einsum("bhcm,bhum->bhcu", alpha, enc_weights)
                y_chunk = y_chunk + torch.einsum("bhcu,bhud->bhcd", beta, Vc[:, :, block_idx, local_src_chunk, :, :])
            Yc[:, :, block_idx, local_q_chunk, :, :] = y_chunk

    Y = Yc.reshape(B, H, N, D_value).permute(0, 2, 1, 3).to(out_dtype)
    _check_finite("block_causal.Y", Y)

    if not return_aux:
        return Y
    phase1_mode = "block_stats" if chunks_per_block == 1 else "chunk_stats"
    return Y, {"LSE_dec": LSE_dec, "LSE_enc": LSE_enc, "phase1_mode": phase1_mode}


def flare_block_causal_reference(Q, K, V, *, block_size, scale=None, Q_dec=None, K_dec=None):
    B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="Block-Causal FLARE reference")
    _validate_block_causal_config(N=N, block_size=block_size, chunk_size=16, name="Block-Causal FLARE reference")
    scale = _resolve_attn_scale(scale, D_score)
    Q_dec, K_dec, separate_Q_dec, separate_K_dec, _ = _resolve_flare_causal_decode_inputs(Q, K, Q_dec, K_dec)

    Q_enc = Q.float().unsqueeze(0).expand(B, -1, -1, -1)
    K_enc = K.float().permute(0, 2, 1, 3).contiguous()
    V_enc = V.float().permute(0, 2, 1, 3).contiguous()
    Q_dec_f = Q_dec.float().permute(0, 2, 1, 3).contiguous() if separate_Q_dec else K_enc
    K_dec_f = K_dec.float().unsqueeze(0).expand(B, -1, -1, -1).contiguous() if separate_K_dec else Q_enc

    Y = torch.empty((B, H, N, D_value), device=K.device, dtype=torch.float32)
    num_blocks = math.ceil(N / block_size)
    for block_idx in range(num_blocks):
        block_start = block_idx * block_size
        block_end = min((block_idx + 1) * block_size, N)
        z_blk = F.scaled_dot_product_attention(
            Q_enc,
            K_enc[:, :, :block_end, :],
            V_enc[:, :, :block_end, :],
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        y_blk = F.scaled_dot_product_attention(
            Q_dec_f[:, :, block_start:block_end, :],
            K_dec_f,
            z_blk,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
        )
        Y[:, :, block_start:block_end, :] = y_blk
    return Y.permute(0, 2, 1, 3).to(V.dtype)


def _validate_sdpa_qkv(Q, K, V, *, name: str):
    if Q.dim() != 4 or K.dim() != 4 or V.dim() != 4:
        raise ValueError(
            f"{name} expects Q, K, V as [B, N, H, D]. "
            f"Got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
        )
    if Q.shape != K.shape or K.shape != V.shape:
        raise ValueError(
            f"{name} expects Q, K, V to have identical shapes. "
            f"Got Q.shape={Q.shape}, K.shape={K.shape}, V.shape={V.shape}"
        )
    B, N, H, D = Q.shape
    return B, N, H, D


def block_causal_sdpa_reference(Q, K, V, *, block_size, scale=None):
    B, N, H, D = _validate_sdpa_qkv(Q, K, V, name="Block-causal SDPA reference")
    _validate_block_causal_config(N=N, block_size=block_size, chunk_size=16, name="Block-causal SDPA reference")
    scale = _resolve_attn_scale(scale, D)
    mask = _block_causal_mask(N, block_size, Q.device).view(1, 1, N, N)
    return F.scaled_dot_product_attention(
        Q.permute(0, 2, 1, 3).float(),
        K.permute(0, 2, 1, 3).float(),
        V.permute(0, 2, 1, 3).float(),
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    ).permute(0, 2, 1, 3).to(V.dtype)


def block_causal_sdpa_flex(Q, K, V, *, block_size, scale=None):
    if create_block_mask is None or flex_attention is None:
        raise RuntimeError("torch.nn.attention.flex_attention is not available in this PyTorch build.")
    B, N, H, D = _validate_sdpa_qkv(Q, K, V, name="Block-causal SDPA flex")
    _validate_block_causal_config(N=N, block_size=block_size, chunk_size=16, name="Block-causal SDPA flex")
    scale = _resolve_attn_scale(scale, D)
    q_bhnd = Q.permute(0, 2, 1, 3).contiguous()
    k_bhnd = K.permute(0, 2, 1, 3).contiguous()
    v_bhnd = V.permute(0, 2, 1, 3).contiguous()
    flex_block_size = min(128, block_size)

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        del batch_idx, head_idx
        block_limit = ((q_idx // block_size) + 1) * block_size
        block_limit = torch.clamp(block_limit, max=N)
        return kv_idx < block_limit

    block_mask = create_block_mask(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=N,
        KV_LEN=N,
        device=Q.device,
        BLOCK_SIZE=flex_block_size,
        _compile=False,
    )
    return flex_attention(q_bhnd, k_bhnd, v_bhnd, block_mask=block_mask, scale=scale).permute(0, 2, 1, 3).to(V.dtype)


class BlockCausalFLARE(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale=None, block_size=None, chunk_size=None, Q_dec=None, K_dec=None):
        Y = _block_causal_forward_torch(
            Q,
            K,
            V,
            block_size=block_size,
            chunk_size=chunk_size,
            scale=scale,
            Q_dec=Q_dec,
            K_dec=K_dec,
            return_aux=False,
        )
        tensors = [Q, K, V]
        if Q_dec is not None:
            tensors.append(Q_dec)
        if K_dec is not None:
            tensors.append(K_dec)
        ctx.save_for_backward(*tensors)
        ctx.scale = scale
        ctx.block_size = block_size
        ctx.chunk_size = chunk_size
        ctx.has_q_dec = Q_dec is not None
        ctx.has_k_dec = K_dec is not None
        return Y

    @staticmethod
    def backward(ctx, dY):
        saved = list(ctx.saved_tensors)
        Q = saved.pop(0)
        K = saved.pop(0)
        V = saved.pop(0)
        Q_dec = saved.pop(0) if ctx.has_q_dec else None
        K_dec = saved.pop(0) if ctx.has_k_dec else None

        with torch.enable_grad():
            Q_r = Q.detach().requires_grad_(ctx.needs_input_grad[0])
            K_r = K.detach().requires_grad_(ctx.needs_input_grad[1])
            V_r = V.detach().requires_grad_(ctx.needs_input_grad[2])
            Q_dec_r = None
            K_dec_r = None
            if ctx.has_q_dec:
                Q_dec_r = Q_dec.detach().requires_grad_(ctx.needs_input_grad[6])
            if ctx.has_k_dec:
                K_dec_r = K_dec.detach().requires_grad_(ctx.needs_input_grad[7])

            Y = _block_causal_forward_torch(
                Q_r,
                K_r,
                V_r,
                block_size=ctx.block_size,
                chunk_size=ctx.chunk_size,
                scale=ctx.scale,
                Q_dec=Q_dec_r,
                K_dec=K_dec_r,
                return_aux=False,
            )

            inputs = [Q_r, K_r, V_r]
            if ctx.has_q_dec:
                inputs.append(Q_dec_r)
            if ctx.has_k_dec:
                inputs.append(K_dec_r)
            grads = torch.autograd.grad(Y, inputs, dY, allow_unused=True)

        grad_iter = iter(grads)
        dQ = next(grad_iter)
        dK = next(grad_iter)
        dV = next(grad_iter)
        dQ_dec = next(grad_iter) if ctx.has_q_dec else None
        dK_dec = next(grad_iter) if ctx.has_k_dec else None
        return dQ, dK, dV, None, None, None, dQ_dec, dK_dec


def flare_block_causal_torch(Q, K, V, *, block_size, chunk_size, scale=None, Q_dec=None, K_dec=None):
    return BlockCausalFLARE.apply(Q, K, V, scale, block_size, chunk_size, Q_dec, K_dec)


def _benchmark_callable(fn, *, warmup: int, iters: int, device: torch.device):
    for _ in range(warmup):
        _ = fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = fn()
        end.record()
        torch.cuda.synchronize(device)
        return out, float(start.elapsed_time(end) / iters)

    start_s = time.perf_counter()
    for _ in range(iters):
        out = fn()
    elapsed_ms = (time.perf_counter() - start_s) * 1e3 / iters
    return out, float(elapsed_ms)


def benchmark_block_causal_torch(
    Q,
    K,
    V,
    *,
    block_size,
    chunk_size,
    scale=None,
    warmup: int = 2,
    iters: int = 5,
    compare_reference: bool = True,
):
    scale = _resolve_attn_scale(scale, Q.size(-1))
    device = K.device

    def run_impl():
        return flare_block_causal_torch(Q, K, V, block_size=block_size, chunk_size=chunk_size, scale=scale)

    y_impl, impl_ms = _benchmark_callable(run_impl, warmup=warmup, iters=iters, device=device)
    results = {
        "impl_ms": impl_ms,
        "impl_shape": tuple(y_impl.shape),
        "reference_ms": None,
        "reference_max_abs": None,
        "reference_mean_abs": None,
    }

    if compare_reference:
        def run_ref():
            return flare_block_causal_reference(Q, K, V, block_size=block_size, scale=scale)

        y_ref, ref_ms = _benchmark_callable(run_ref, warmup=1, iters=max(1, min(2, iters)), device=device)
        delta = (y_impl.float() - y_ref.float()).abs()
        results["reference_ms"] = ref_ms
        results["reference_max_abs"] = float(delta.max().item())
        results["reference_mean_abs"] = float(delta.mean().item())

    return results


def benchmark_block_causal_sdpa_flex(Q, K, V, *, block_size, scale=None, warmup: int = 2, iters: int = 5, compare_reference: bool = True):
    scale = _resolve_attn_scale(scale, Q.size(-1))
    device = Q.device

    def run_impl():
        return block_causal_sdpa_flex(Q, K, V, block_size=block_size, scale=scale)

    y_impl, impl_ms = _benchmark_callable(run_impl, warmup=warmup, iters=iters, device=device)
    results = {
        "impl_ms": impl_ms,
        "impl_shape": tuple(y_impl.shape),
        "reference_ms": None,
        "reference_max_abs": None,
        "reference_mean_abs": None,
    }
    if compare_reference:
        def run_ref():
            return block_causal_sdpa_reference(Q, K, V, block_size=block_size, scale=scale)

        y_ref, ref_ms = _benchmark_callable(run_ref, warmup=1, iters=max(1, min(2, iters)), device=device)
        delta = (y_impl.float() - y_ref.float()).abs()
        results["reference_ms"] = ref_ms
        results["reference_max_abs"] = float(delta.max().item())
        results["reference_mean_abs"] = float(delta.mean().item())
    return results

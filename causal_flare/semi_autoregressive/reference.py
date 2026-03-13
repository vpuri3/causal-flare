import math
import time

import torch
import torch.nn.functional as F

from causal_flare._common import _resolve_attn_scale
from causal_flare._reference_utils import (
    resolve_flare_causal_decode_inputs as _resolve_flare_causal_decode_inputs,
    validate_flare_qkv_layouts as _validate_flare_qkv_layouts,
)


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


def _block_causal_forward_pytorch(
    Q,
    K,
    V,
    *,
    block_size,
    chunk_size,
    scale=None,
    Q_dec=None,
    K_dec=None,
    save_chunk_stats: bool = True,
    return_aux: bool = False,
):
    """Reference semi-autoregressive / block-causal forward pass.

    Shapes:
    - `Q`: `[H, M, D_score]`
      Shared latent/query bank. `H` heads, `M` latents per head, `D_score` score dimension.
    - `K`: `[B, N, H, D_score]`
      Encoder keys for `B` sequences and `N` tokens.
    - `V`: `[B, N, H, D_value]`
      Encoder values.
    - `Q_dec`:
      Optional decode-side queries. Canonicalized to `[B, H, N, D_score]` when separate.
    - `K_dec`:
      Optional decode-side keys. Canonicalized to `[B, H, M, D_score]` when separate.

    Returns:
    - `Y`: `[B, N, H, D_value]`
    - If `return_aux=True`, also returns:
      - `LSE_dec`: `[B, H, N]`
      - `LSE_enc`: `[B, H, num_blocks, M]`
    """

    # This is the readable reference version of the semi-autoregressive / block-causal
    # algorithm. It is organized as a sequence of explicit phases that mirror the
    # intended chunkwise training structure:
    #
    # Phase 0: validate shapes, canonicalize decode inputs, and reshape K/V into
    #          block-aligned chunk views.
    # Phase 1: compute chunk-local encoder statistics for every source chunk inside
    #          each block. These are the basic chunk summaries the later phases scan.
    # Phase 2: merge chunk-local summaries into per-block summaries, then prefix-scan
    #          those block summaries so each block sees all strictly previous blocks.
    # Phase 3: finish the encoder and decoder normalizers. For the encoder this means
    #          the full within-block cumulative log-sum-exp. For the decoder this means
    #          one log-sum-exp over the latent axis for each query token chunk.
    # Phase 4: assemble outputs by combining two sources:
    #          1. the prefix contribution from all completed previous blocks
    #          2. the within-block contribution from every source chunk in the current block
    # The implementation is intentionally explicit rather than clever so the dataflow is easy
    # to compare against the future Triton path.

    # Phase 0: validate the block-causal layout and canonicalize decode-side inputs.
    # Public layout:
    #   Q: [H, M, D_score]
    #   K: [B, N, H, D_score]
    #   V: [B, N, H, D_value]
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
    stats_dtype = torch.float32
    out_dtype = V.dtype
    num_blocks = N // block_size
    chunks_per_block = block_size // chunk_size

    # Run score computation and value accumulation in FP32. This keeps the numerically
    # sensitive softmax-weighted numerator path stable even when the caller provides BF16.
    Q_f = Q.to(compute_dtype)  # [H, M, D_score]
    K_f = K.to(compute_dtype)  # [B, N, H, D_score]
    V_f = V.to(compute_dtype)  # [B, N, H, D_value]

    # The decode branch can either share weights with the encode branch or use separate
    # Q_dec/K_dec projections. Canonicalize both cases into the same [B, H, ...] layout.
    if weight_sharing_enc_dec:
        Q_dec_f = None
        K_dec_f = None
    else:
        q_dec_comp = Q_dec.to(compute_dtype) if separate_Q_dec else K_f
        k_dec_comp = K_dec.to(compute_dtype) if separate_K_dec else Q_f
        Q_dec_f = q_dec_comp.permute(0, 2, 1, 3).contiguous()  # [B, H, N, D_score]
        K_dec_f = k_dec_comp.unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # [B, H, M, D_score]

    # Reinterpret K/V as [B, H, block, local_chunk, token_in_chunk, D]. This makes the
    # later block-local and chunk-local scans explicit and avoids re-slicing flat [B, N, ...]
    # tensors over and over inside the phase loops.
    Kc = K_f.reshape(B, num_blocks, chunks_per_block, chunk_size, H, D_score).permute(0, 4, 1, 2, 3, 5).contiguous()
    Kc = Kc  # [B, H, num_blocks, chunks_per_block, chunk_size, D_score]
    Vc = V_f.reshape(B, num_blocks, chunks_per_block, chunk_size, H, D_value).permute(0, 4, 1, 2, 3, 5).contiguous()
    Vc = Vc  # [B, H, num_blocks, chunks_per_block, chunk_size, D_value]

    BHB = B * H
    if save_chunk_stats:
        # Phase 1 (large-block regime): materialize chunk-local stats so later phases can
        # reuse them instead of recomputing. This better models kernels that prefer extra
        # HBM traffic over repeated chunk replay when blocks contain many chunks.
        score_chunk = scale * torch.einsum("bhgxcd,hmd->bhgxcm", Kc, Q_f)  # [B, H, num_blocks, chunks_per_block, chunk_size, M]
        chunk_max = score_chunk.max(dim=4).values.to(stats_dtype)  # [B, H, num_blocks, chunks_per_block, M]
        chunk_exp = torch.exp(score_chunk - chunk_max.to(compute_dtype).unsqueeze(4))  # [B, H, num_blocks, chunks_per_block, chunk_size, M]
        chunk_den = chunk_exp.to(stats_dtype).sum(dim=4)  # [B, H, num_blocks, chunks_per_block, M]
        chunk_num = torch.bmm(
            chunk_exp.reshape(B * H * num_blocks * chunks_per_block, chunk_size, M).transpose(1, 2),
            Vc.reshape(B * H * num_blocks * chunks_per_block, chunk_size, D_value),
        ).reshape(B, H, num_blocks, chunks_per_block, M, D_value)  # [B, H, num_blocks, chunks_per_block, M, D_value]

        # Phase 2 (large-block regime): merge chunk summaries into block summaries, then
        # scan those block summaries across blocks.
        block_max = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)  # [B, H, num_blocks, M]
        block_den = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)  # [B, H, num_blocks, M]
        block_num = torch.empty((B, H, num_blocks, M, D_value), device=Q.device, dtype=compute_dtype)
        for block_idx in range(num_blocks):
            max_block = torch.full((B, H, M), -float("inf"), device=Q.device, dtype=stats_dtype)  # [B, H, M]
            den_block = torch.zeros((B, H, M), device=Q.device, dtype=stats_dtype)  # [B, H, M]
            num_block = torch.zeros((B, H, M, D_value), device=Q.device, dtype=compute_dtype)  # [B, H, M, D_value]
            for local_chunk in range(chunks_per_block):
                cm = chunk_max[:, :, block_idx, local_chunk, :]
                cd = chunk_den[:, :, block_idx, local_chunk, :]
                cn = chunk_num[:, :, block_idx, local_chunk, :, :]
                max_new = torch.maximum(max_block, cm)
                rescale_prev = torch.exp(max_block - max_new)
                rescale_chunk = torch.exp(cm - max_new)
                den_block = den_block * rescale_prev + cd * rescale_chunk
                num_block = (
                    num_block * rescale_prev.to(compute_dtype).unsqueeze(-1)
                    + cn * rescale_chunk.to(compute_dtype).unsqueeze(-1)
                )
                max_block = max_new
            block_max[:, :, block_idx, :] = max_block
            block_den[:, :, block_idx, :] = den_block
            block_num[:, :, block_idx, :, :] = num_block
    else:
        # Phase 1 (small-block regime): chunking is only the internal reduction schedule
        # used to build one encoder summary per block. A kernel-faithful implementation can
        # keep chunk-local work on-chip and materialize only block summaries to HBM.
        block_max = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)  # [B, H, num_blocks, M]
        block_den = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)  # [B, H, num_blocks, M]
        block_num = torch.empty((B, H, num_blocks, M, D_value), device=Q.device, dtype=compute_dtype)
        for block_idx in range(num_blocks):
            max_block = torch.full((B, H, M), -float("inf"), device=Q.device, dtype=stats_dtype)  # [B, H, M]
            den_block = torch.zeros((B, H, M), device=Q.device, dtype=stats_dtype)  # [B, H, M]
            num_block = torch.zeros((B, H, M, D_value), device=Q.device, dtype=compute_dtype)  # [B, H, M, D_value]

            for local_chunk in range(chunks_per_block):
                k_chunk = Kc[:, :, block_idx, local_chunk, :, :]  # [B, H, chunk_size, D_score]
                v_chunk = Vc[:, :, block_idx, local_chunk, :, :]  # [B, H, chunk_size, D_value]
                score_chunk = scale * torch.einsum("bhcd,hmd->bhcm", k_chunk, Q_f)  # [B, H, chunk_size, M]
                chunk_max = score_chunk.max(dim=2).values.to(stats_dtype)  # [B, H, M]
                chunk_exp = torch.exp(score_chunk - chunk_max.to(compute_dtype).unsqueeze(2))  # [B, H, chunk_size, M]
                chunk_den = chunk_exp.to(stats_dtype).sum(dim=2)  # [B, H, M]
                chunk_num = torch.bmm(
                    chunk_exp.reshape(BHB, chunk_size, M).transpose(1, 2),
                    v_chunk.reshape(BHB, chunk_size, D_value),
                ).reshape(B, H, M, D_value)  # [B, H, M, D_value]

                max_new = torch.maximum(max_block, chunk_max)
                rescale_prev = torch.exp(max_block - max_new)
                rescale_chunk = torch.exp(chunk_max - max_new)
                den_block = den_block * rescale_prev + chunk_den * rescale_chunk
                num_block = (
                    num_block * rescale_prev.to(compute_dtype).unsqueeze(-1)
                    + chunk_num * rescale_chunk.to(compute_dtype).unsqueeze(-1)
                )
                max_block = max_new

            block_max[:, :, block_idx, :] = max_block
            block_den[:, :, block_idx, :] = den_block
            block_num[:, :, block_idx, :, :] = num_block

    # Phase 2: prefix-scan the per-block summaries. This is the first saved-summary level
    # the eventual Triton path would likely write, since block summaries are sufficient to
    # represent all completed earlier blocks.
    prefix_max = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)  # [B, H, num_blocks, M]
    prefix_den = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)  # [B, H, num_blocks, M]
    prefix_num = torch.empty((B, H, num_blocks, M, D_value), device=Q.device, dtype=compute_dtype)
    full_block_max = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)  # [B, H, num_blocks, M]
    full_block_den = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)  # [B, H, num_blocks, M]
    max_curr = torch.full((B, H, M), -float("inf"), device=Q.device, dtype=stats_dtype)  # [B, H, M]
    den_curr = torch.zeros((B, H, M), device=Q.device, dtype=stats_dtype)  # [B, H, M]
    num_curr = torch.zeros((B, H, M, D_value), device=Q.device, dtype=compute_dtype)  # [B, H, M, D_value]
    for block_idx in range(num_blocks):
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
        num_curr = (
            num_curr * rescale_prev.to(compute_dtype).unsqueeze(-1)
            + bn * rescale_block.to(compute_dtype).unsqueeze(-1)
        )
        max_curr = max_new

        full_block_max[:, :, block_idx, :] = max_curr
        full_block_den[:, :, block_idx, :] = den_curr

    # Phase 3a: compute all decode-side scores in block/chunk layout, then one logsumexp
    # over the latent axis produces the decoder normalizer for every query token.
    if weight_sharing_enc_dec:
        if save_chunk_stats:
            dec_scores = score_chunk  # [B, H, num_blocks, chunks_per_block, chunk_size, M]
        else:
            dec_scores = scale * torch.einsum("bhgxcd,hmd->bhgxcm", Kc, Q_f)  # [B, H, num_blocks, chunks_per_block, chunk_size, M]
    else:
        Q_dec_chunks = Q_dec_f.reshape(B, H, num_blocks, chunks_per_block, chunk_size, D_score)
        Kc_dec = K_dec_f.unsqueeze(2).unsqueeze(3).expand(-1, -1, num_blocks, chunks_per_block, -1, -1)
        dec_scores = scale * (Kc_dec @ Q_dec_chunks.mT).transpose(-1, -2)
    LSE_dec = torch.logsumexp(dec_scores.to(stats_dtype), dim=-1).reshape(B, H, N)  # [B, H, N]

    # Phase 3b: The encoder-side normalization is stored as log(sum(exp(.))) so later phases do not
    # need to carry separate max/denominator tensors unless they are explicitly inspecting
    # intermediate summaries.
    LSE_enc = torch.log(full_block_den) + full_block_max  # [B, H, num_blocks, M]

    # Phase 4: assemble outputs. Each query chunk receives:
    #   - a prefix contribution from all completed previous blocks, already summarized in
    #     prefix_num / prefix_max / LSE_enc
    #   - a current-block contribution formed by mixing decoder weights (alpha) with
    #     encoder weights for each source chunk in the same block
    #
    # The inner local_src_chunk loop is intentionally explicit: it makes it obvious that
    # within-block interactions are retained in full, while earlier blocks only appear via
    # their prefix summary.
    Yc = torch.empty((B, H, num_blocks, chunks_per_block, chunk_size, D_value), device=Q.device, dtype=compute_dtype)
    # Yc: [B, H, num_blocks, chunks_per_block, chunk_size, D_value]
    for block_idx in range(num_blocks):
        prefix_block_num = prefix_num[:, :, block_idx, :, :]  # [B, H, M, D_value]
        lse_enc_block = LSE_enc[:, :, block_idx, :]  # [B, H, M]
        prefix_scale = torch.exp(prefix_max[:, :, block_idx, :] - lse_enc_block).to(compute_dtype).unsqueeze(-1)  # [B, H, M, 1]
        prefix_value = prefix_block_num * prefix_scale  # [B, H, M, D_value]

        for local_q_chunk in range(chunks_per_block):
            token_start = (block_idx * chunks_per_block + local_q_chunk) * chunk_size

            dec_scores_chunk = dec_scores[:, :, block_idx, local_q_chunk, :, :]  # [B, H, chunk_size, M]
            lse_dec_chunk = LSE_dec[:, :, token_start : token_start + chunk_size]  # [B, H, chunk_size]
            alpha = torch.exp(dec_scores_chunk - lse_dec_chunk.to(compute_dtype).unsqueeze(-1))  # [B, H, chunk_size, M]

            y_chunk = torch.einsum("bhcm,bhmd->bhcd", alpha, prefix_value)  # [B, H, chunk_size, D_value]
            for local_src_chunk in range(chunks_per_block):
                if save_chunk_stats:
                    enc_scores_chunk = score_chunk[:, :, block_idx, local_src_chunk, :, :]
                else:
                    enc_scores_chunk = scale * torch.einsum(
                        "bhcd,hmd->bhcm", Kc[:, :, block_idx, local_src_chunk, :, :], Q_f
                    )  # [B, H, chunk_size, M]
                enc_weights = torch.exp(
                    enc_scores_chunk - lse_enc_block.to(compute_dtype).unsqueeze(2)
                )  # [B, H, chunk_size, M]
                beta = torch.einsum("bhcm,bhum->bhcu", alpha, enc_weights)  # [B, H, chunk_size, chunk_size]
                y_chunk = y_chunk + torch.einsum(
                    "bhcu,bhud->bhcd", beta, Vc[:, :, block_idx, local_src_chunk, :, :]
                )  # [B, H, chunk_size, D_value]
            Yc[:, :, block_idx, local_q_chunk, :, :] = y_chunk

    # Collapse the block/chunk view back to the public [B, N, H, D_value] layout.
    Y = Yc.reshape(B, H, N, D_value).permute(0, 2, 1, 3).to(out_dtype)  # [B, N, H, D_value]
    if not return_aux:
        return Y
    return Y, {"LSE_dec": LSE_dec, "LSE_enc": LSE_enc, "save_chunk_stats": save_chunk_stats}


def _block_causal_forward_pytorch_chunk_stats(
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
    return _block_causal_forward_pytorch(
        Q,
        K,
        V,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        Q_dec=Q_dec,
        K_dec=K_dec,
        save_chunk_stats=True,
        return_aux=return_aux,
    )


def _block_causal_forward_pytorch_block_stats(
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
    return _block_causal_forward_pytorch(
        Q,
        K,
        V,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        Q_dec=Q_dec,
        K_dec=K_dec,
        save_chunk_stats=False,
        return_aux=return_aux,
    )


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


def semi_autoregressive_flare_reference(Q, K, V, *, block_size, scale=None, Q_dec=None, K_dec=None):
    B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="Block-Causal FLARE reference")
    _validate_block_causal_config(N=N, block_size=block_size, chunk_size=16, name="Block-Causal FLARE reference")
    scale = _resolve_attn_scale(scale, D_score)
    Q_dec, K_dec, separate_Q_dec, separate_K_dec, _ = _resolve_flare_causal_decode_inputs(Q, K, Q_dec, K_dec)

    compute_dtype = Q.dtype
    Q_enc = Q.to(compute_dtype).unsqueeze(0).expand(B, -1, -1, -1)
    K_enc = K.to(compute_dtype).permute(0, 2, 1, 3).contiguous()
    V_enc = V.to(compute_dtype).permute(0, 2, 1, 3).contiguous()
    Q_dec_f = Q_dec.to(compute_dtype).permute(0, 2, 1, 3).contiguous() if separate_Q_dec else K_enc
    K_dec_f = K_dec.to(compute_dtype).unsqueeze(0).expand(B, -1, -1, -1).contiguous() if separate_K_dec else Q_enc

    Y = torch.empty((B, H, N, D_value), device=K.device, dtype=compute_dtype)
    num_blocks = (N + block_size - 1) // block_size
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


def benchmark_block_causal_torch(
    Q,
    K,
    V,
    *,
    block_size,
    chunk_size,
    save_chunk_stats: bool = True,
    scale=None,
    warmup: int = 2,
    iters: int = 5,
    compare_reference: bool = True,
):
    scale = _resolve_attn_scale(scale, Q.size(-1))
    device = K.device

    def run_impl():
        return _block_causal_forward_pytorch(
            Q, K, V, block_size=block_size, chunk_size=chunk_size, scale=scale, save_chunk_stats=save_chunk_stats
        )

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
            return semi_autoregressive_flare_reference(Q, K, V, block_size=block_size, scale=scale)

        y_ref, ref_ms = _benchmark_callable(run_ref, warmup=1, iters=max(1, min(2, iters)), device=device)
        delta = (y_impl.float() - y_ref.float()).abs()
        results["reference_ms"] = ref_ms
        results["reference_max_abs"] = float(delta.max().item())
        results["reference_mean_abs"] = float(delta.mean().item())

    return results

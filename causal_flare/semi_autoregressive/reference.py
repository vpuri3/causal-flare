import math
import time

import torch
import torch.nn.functional as F

from causal_flare._common import _check_finite, _resolve_attn_scale
from causal_flare._reference_utils import (
    resolve_flare_causal_decode_inputs as _resolve_flare_causal_decode_inputs,
    validate_flare_qkv_layouts as _validate_flare_qkv_layouts,
)

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
    out_dtype = V.dtype
    num_blocks = N // block_size
    chunks_per_block = block_size // chunk_size

    # The reference path runs the numerically sensitive reductions in FP32 regardless
    # of input dtype so the intermediate max/denominator/numerator statistics are stable.
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

    # Phase 1: score every latent against every token inside each source chunk. The result
    # has shape [B, H, block, local_chunk, token_in_chunk, latent].
    score_chunk = scale * torch.einsum("bhgxcd,hmd->bhgxcm", Kc, Q_f)  # [B, H, num_blocks, chunks_per_block, chunk_size, M]

    # Compress each source chunk down to the usual stable-softmax triplet:
    #   max over tokens
    #   denominator over tokens
    #   numerator over tokens against V
    # These chunk summaries are the smallest units that can later be merged into larger
    # block summaries without revisiting token-level encoder scores.
    BHNBCPB = B * H * num_blocks * chunks_per_block
    chunk_max = score_chunk.max(dim=4).values  # [B, H, num_blocks, chunks_per_block, M]
    safe_chunk_max = torch.where(torch.isfinite(chunk_max), chunk_max, 0.0)  # [B, H, num_blocks, chunks_per_block, M]
    chunk_exp = torch.where(
        torch.isfinite(score_chunk),
        torch.exp(score_chunk - safe_chunk_max.unsqueeze(4)),
        0.0,
    )  # [B, H, num_blocks, chunks_per_block, chunk_size, M]
    chunk_den = chunk_exp.sum(dim=4)  # [B, H, num_blocks, chunks_per_block, M]
    chunk_num = torch.bmm(
        chunk_exp.reshape(BHNBCPB, chunk_size, M).transpose(1, 2),
        Vc.reshape(BHNBCPB, chunk_size, D_value),
    ).reshape(B, H, num_blocks, chunks_per_block, M, D_value)  # [B, H, num_blocks, chunks_per_block, M, D_value]

    # Phase 2a: merge chunk summaries within each block. After this loop, each block has
    # one encoder summary representing all tokens in that block, but not any earlier blocks.
    block_max = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=compute_dtype)  # [B, H, num_blocks, M]
    block_den = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=compute_dtype)  # [B, H, num_blocks, M]
    block_num = torch.empty((B, H, num_blocks, M, D_value), device=Q.device, dtype=compute_dtype)  # [B, H, num_blocks, M, D_value]
    for block_idx in range(num_blocks):
        max_curr = torch.full((B, H, M), -float("inf"), device=Q.device, dtype=compute_dtype)  # [B, H, M]
        den_curr = torch.zeros((B, H, M), device=Q.device, dtype=compute_dtype)  # [B, H, M]
        num_curr = torch.zeros((B, H, M, D_value), device=Q.device, dtype=compute_dtype)  # [B, H, M, D_value]
        for local_chunk in range(chunks_per_block):
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

    # Phase 2b: exclusive prefix-scan the per-block summaries. For block k, the prefix
    # tensors store the encoder summary for all strictly previous blocks (< k). This is
    # exactly the semi-autoregressive / block-causal receptive field contribution that is
    # already complete before we process the current block.
    prefix_max = torch.empty_like(block_max)  # [B, H, num_blocks, M]
    prefix_den = torch.empty_like(block_den)  # [B, H, num_blocks, M]
    prefix_num = torch.empty_like(block_num)  # [B, H, num_blocks, M, D_value]
    max_curr = torch.full((B, H, M), -float("inf"), device=Q.device, dtype=compute_dtype)  # [B, H, M]
    den_curr = torch.zeros((B, H, M), device=Q.device, dtype=compute_dtype)  # [B, H, M]
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
        num_curr = num_curr * rescale_prev.unsqueeze(-1) + bn * rescale_block.unsqueeze(-1)
        max_curr = max_new

    # Phase 3a: rebuild the full encoder normalizer for each block by combining the
    # previous-block prefix with the current block's local chunks. This gives one encoder
    # LSE per latent and block. At the same time, compute decoder LSE values per query token
    # chunk, which normalize the latent mixing weights for the decode branch.
    full_block_max = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=compute_dtype)  # [B, H, num_blocks, M]
    full_block_den = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=compute_dtype)  # [B, H, num_blocks, M]
    LSE_dec = torch.empty((B, H, N), device=Q.device, dtype=compute_dtype)  # [B, H, N]
    for block_idx in range(num_blocks):
        max_curr = prefix_max[:, :, block_idx, :]
        den_curr = prefix_den[:, :, block_idx, :]
        for local_chunk in range(chunks_per_block):
            cm = chunk_max[:, :, block_idx, local_chunk, :]
            cd = chunk_den[:, :, block_idx, local_chunk, :]
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
                dec_scores = score_chunk[:, :, block_idx, local_q_chunk, :, :]  # [B, H, chunk_size, M]
            else:
                q_t_dec = Q_dec_f[:, :, token_start : token_start + chunk_size, :]
                dec_scores = torch.einsum("bhcd,bhmd->bhcm", q_t_dec, K_dec_f) * scale  # [B, H, chunk_size, M]
            LSE_dec[:, :, token_start : token_start + chunk_size] = torch.logsumexp(dec_scores, dim=-1)

    # The encoder-side normalization is stored as log(sum(exp(.))) so later phases do not
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
        prefix_scale = torch.exp(prefix_max[:, :, block_idx, :] - lse_enc_block).unsqueeze(-1)  # [B, H, M, 1]
        prefix_value = prefix_block_num * prefix_scale  # [B, H, M, D_value]

        for local_q_chunk in range(chunks_per_block):
            token_start = (block_idx * chunks_per_block + local_q_chunk) * chunk_size

            if weight_sharing_enc_dec:
                dec_scores = score_chunk[:, :, block_idx, local_q_chunk, :, :]  # [B, H, chunk_size, M]
            else:
                q_t_dec = Q_dec_f[:, :, token_start : token_start + chunk_size, :]  # [B, H, chunk_size, D_score]
                dec_scores = torch.einsum("bhcd,bhmd->bhcm", q_t_dec, K_dec_f) * scale  # [B, H, chunk_size, M]
            lse_dec_chunk = LSE_dec[:, :, token_start : token_start + chunk_size]  # [B, H, chunk_size]
            alpha = torch.exp(dec_scores - lse_dec_chunk.unsqueeze(-1))  # [B, H, chunk_size, M]

            y_chunk = torch.einsum("bhcm,bhmd->bhcd", alpha, prefix_value)  # [B, H, chunk_size, D_value]
            for local_src_chunk in range(chunks_per_block):
                enc_weights = torch.exp(
                    score_chunk[:, :, block_idx, local_src_chunk, :, :] - lse_enc_block.unsqueeze(2)
                )  # [B, H, chunk_size, M]
                beta = torch.einsum("bhcm,bhum->bhcu", alpha, enc_weights)  # [B, H, chunk_size, chunk_size]
                y_chunk = y_chunk + torch.einsum(
                    "bhcu,bhud->bhcd", beta, Vc[:, :, block_idx, local_src_chunk, :, :]
                )  # [B, H, chunk_size, D_value]
            Yc[:, :, block_idx, local_q_chunk, :, :] = y_chunk

    # Collapse the block/chunk view back to the public [B, N, H, D_value] layout.
    Y = Yc.reshape(B, H, N, D_value).permute(0, 2, 1, 3).to(out_dtype)  # [B, N, H, D_value]
    _check_finite("block_causal.Y", Y)

    if not return_aux:
        return Y
    phase1_mode = "block_stats" if chunks_per_block == 1 else "chunk_stats"
    return Y, {"LSE_dec": LSE_dec, "LSE_enc": LSE_enc, "phase1_mode": phase1_mode}


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
    _, N, _, D = _validate_sdpa_qkv(Q, K, V, name="Block-causal SDPA flex")
    if N > 8192:
        raise ValueError(f"Block-causal SDPA flex is currently restricted to N <= 8192. Got N={N}.")
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

    def score_mod(score, batch_idx, head_idx, q_idx, kv_idx):
        return torch.where(
            mask_mod(batch_idx, head_idx, q_idx, kv_idx),
            score,
            torch.tensor(-float("inf"), device=score.device, dtype=score.dtype),
        )

    # The block-causal sparsity pattern depends only on token indices, not on batch or head.
    # Let FlexAttention broadcast those axes instead of specializing the mask over [B, H].
    block_mask = create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=N,
        KV_LEN=N,
        device=Q.device,
        BLOCK_SIZE=flex_block_size,
        _compile=True,
    )
    return flex_attention(
        q_bhnd,
        k_bhnd,
        v_bhnd,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        kernel_options={"BLOCKS_ARE_CONTIGUOUS": True},
    ).permute(0, 2, 1, 3).to(V.dtype)


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
        return _block_causal_forward_pytorch(Q, K, V, block_size=block_size, chunk_size=chunk_size, scale=scale)

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

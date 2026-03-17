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


def _build_block_causal_forward_state(
    Q,
    K,
    V,
    *,
    block_size,
    chunk_size,
    scale=None,
    Q_dec=None,
    K_dec=None,
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
        _return_state=True,
    )


def _block_causal_summary_backward_pytorch(
    LSE_enc: torch.Tensor,
    z_block: torch.Tensor,
    dZ_block: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map dZ on inclusive block summaries back to local block-summary gradients.

    Forward phase 2 produces, for each block `g`:
      A_g = exp(LSE_enc[g])
      B_g = A_g * z_block[g]

    where `A_g` is the inclusive encoder normalizer through block `g`, and `B_g` is the
    corresponding inclusive value numerator. Decoder backward first gives us `dZ_block`,
    i.e. gradients with respect to the normalized summaries `z_block[g] = B_g / A_g`.

    This helper converts those gradients back into:
    - `a_block[g]`: the *local* encoder normalizer contributed by block `g`
    - `dA_block[g]`: gradient on each local block normalizer
    - `dB_block[g]`: gradient on each local block numerator

    The reverse cumulative sums appear because every local block summary contributes to
    every later inclusive summary.
    """

    a_prefix = torch.exp(LSE_enc.float())
    z_block_f = z_block.float()
    dZ_block_f = dZ_block.float()
    inv_a = torch.where(a_prefix > 0, a_prefix.reciprocal(), torch.zeros_like(a_prefix))

    dB_prefix = dZ_block_f * inv_a.unsqueeze(-1)
    dA_prefix = -(dZ_block_f * z_block_f).sum(dim=-1) * inv_a

    dB_block = torch.flip(torch.cumsum(torch.flip(dB_prefix, dims=(2,)), dim=2), dims=(2,))
    dA_block = torch.flip(torch.cumsum(torch.flip(dA_prefix, dims=(2,)), dim=2), dims=(2,))

    a_prev = torch.zeros_like(a_prefix)
    a_prev[:, :, 1:, :] = a_prefix[:, :, :-1, :]
    a_block = (a_prefix - a_prev).clamp_min(0.0)
    return a_block, dA_block, dB_block


def _block_causal_backward_pytorch(
    Q,
    K,
    V,
    dY,
    *,
    block_size,
    chunk_size,
    scale=None,
    Q_dec=None,
    K_dec=None,
    return_aux: bool = False,
):
    """Reference backward pass matching `_block_causal_forward_pytorch`.

    The backward is structured in the same phases as the forward:
    1. replay decoder softmaxes to produce `dZ_block` and decoder-side gradients
    2. map `dZ_block` back to local encoder block-summary gradients
    3. replay encoder block softmaxes to recover `dQ`, `dK`, and `dV`
    """

    state = _build_block_causal_forward_state(
        Q,
        K,
        V,
        block_size=block_size,
        chunk_size=chunk_size,
        scale=scale,
        Q_dec=Q_dec,
        K_dec=K_dec,
    )

    B = state["B"]
    N = state["N"]
    H = state["H"]
    M = state["M"]
    D_value = state["D_value"]
    block_size = state["block_size"]
    chunk_size = state["chunk_size"]
    num_blocks = state["num_blocks"]
    chunks_per_block = state["chunks_per_block"]
    scale = state["scale"]

    q_bank = state["Q_f"]
    k_tokens = state["K_tokens"]
    v_tokens = state["V_tokens"]
    q_dec_tokens = state["Q_dec_tokens"]
    k_dec_bank = state["K_dec_bank"]
    y_tokens = state["Y"].float().permute(0, 2, 1, 3).contiguous()
    dy_tokens = dY.float().permute(0, 2, 1, 3).contiguous()
    lse_dec = state["LSE_dec"].float()
    lse_enc = state["LSE_enc"].float()
    z_block = state["z_block"].float()

    dQ_shared = torch.zeros_like(q_bank)
    dK_shared_tokens = torch.zeros_like(k_tokens)
    dQ_dec_tokens = torch.zeros_like(q_dec_tokens) if state["separate_Q_dec"] else None
    dK_dec = torch.zeros_like(k_dec_bank) if state["separate_K_dec"] else None
    dZ_block = torch.zeros((B, H, num_blocks, M, D_value), device=Q.device, dtype=torch.float32)

    # For a softmax output `Y[t]`, `delta[t] = <dY[t], Y[t]>` is the standard contraction
    # used to form `dS = P * (dP - delta)`. We precompute it once because phase 1 replays
    # every decoder token against the latent bank.
    delta = (y_tokens * dy_tokens).sum(dim=-1)

    # -------------------------------------------------------------------------
    # Phase 1: replay decoder softmaxes
    # -------------------------------------------------------------------------
    # The forward output phase computed:
    #   alpha[t, m] = softmax_dec(dec_scores[t, m])
    #   Y[t, d] = sum_m alpha[t, m] * z_block[block(t), m, d]
    #
    # Backward therefore has two jobs in this phase:
    # 1. accumulate dZ_block from all tokens that consume the same block summary
    # 2. propagate decoder-softmax gradients into the decode/shared Q/K parameters
    #
    # Crucially, this phase touches only decoder-side quantities plus the saved `z_block`;
    # raw encoder tokens do not reappear until phase 3.
    for block_idx in range(num_blocks):
        z_block_blk = z_block[:, :, block_idx, :, :]
        for local_q_chunk in range(chunks_per_block):
            token_start = block_idx * block_size + local_q_chunk * chunk_size
            token_end = token_start + chunk_size

            q_chunk = q_dec_tokens[:, :, token_start:token_end, :]
            dy_chunk = dy_tokens[:, :, token_start:token_end, :]
            lse_dec_chunk = lse_dec[:, :, token_start:token_end]
            delta_chunk = delta[:, :, token_start:token_end]

            # Replay the decoder weights for this query chunk against the latent bank.
            dec_scores_chunk = scale * torch.einsum("bhcd,hmd->bhcm", q_chunk, k_dec_bank)
            alpha = torch.exp(dec_scores_chunk - lse_dec_chunk.unsqueeze(-1))

            # Every token in this block consumes the same `z_block[block_idx]`, so their
            # contributions all accumulate into the same dZ slice.
            dZ_block[:, :, block_idx, :, :] += torch.einsum("bhcm,bhcd->bhmd", alpha, dy_chunk)

            # Differentiate Y[t] = alpha[t] @ z_block[block(t)] through the decoder softmax.
            dAlpha = torch.einsum("bhcd,bhmd->bhcm", dy_chunk, z_block_blk)
            dS_dec = alpha * (dAlpha - delta_chunk.unsqueeze(-1))
            grad_query = scale * torch.einsum("bhcm,hmd->bhcd", dS_dec, k_dec_bank)
            grad_key = scale * torch.einsum("bhcm,bhcd->hmd", dS_dec, q_chunk)

            # When decode weights are shared, decoder-query gradients land on encoder-token
            # K and decoder-key gradients land on the shared latent bank Q. Otherwise they
            # go to the explicit Q_dec / K_dec tensors.
            if dQ_dec_tokens is None:
                dK_shared_tokens[:, :, token_start:token_end, :] += grad_query
            else:
                dQ_dec_tokens[:, :, token_start:token_end, :] += grad_query

            if dK_dec is None:
                dQ_shared += grad_key
            else:
                dK_dec += grad_key

    # -------------------------------------------------------------------------
    # Phase 2: map inclusive `dZ_block` back to local block summary gradients
    # -------------------------------------------------------------------------
    # Forward phase 2 turned local block summaries into inclusive `z_block`. Before we can
    # replay encoder tokens, we need gradients with respect to each *local* block's
    # numerator/normalizer contribution.
    a_block, dA_block, dB_block = _block_causal_summary_backward_pytorch(lse_enc, z_block, dZ_block)

    dQ_enc = torch.zeros_like(q_bank)
    dK_tokens = torch.zeros_like(k_tokens)
    dV_tokens = torch.zeros_like(v_tokens)
    block_lse = torch.log(a_block.clamp_min(1e-20))

    # -------------------------------------------------------------------------
    # Phase 3: replay encoder block softmaxes
    # -------------------------------------------------------------------------
    # This mirrors forward phase 1, but now each local encoder chunk replays its block-local
    # softmax so we can push gradients from `(dA_block, dB_block)` back into:
    # - encoder values V through the local numerator path
    # - encoder keys K and shared latent bank Q through the local score path
    for block_idx in range(num_blocks):
        a_block_blk = a_block[:, :, block_idx, :]
        dA_block_blk = dA_block[:, :, block_idx, :]
        dB_block_blk = dB_block[:, :, block_idx, :, :]
        lse_block = block_lse[:, :, block_idx, :]
        for local_src_chunk in range(chunks_per_block):
            token_start = block_idx * block_size + local_src_chunk * chunk_size
            token_end = token_start + chunk_size

            k_chunk = k_tokens[:, :, token_start:token_end, :]
            v_chunk = v_tokens[:, :, token_start:token_end, :]

            # `p_block` is the block-local encoder softmax probability in the local block
            # frame. Multiplying by `a_block_blk` converts it back to the raw local weight
            # on this block's numerator/normalizer contribution.
            enc_scores_chunk = scale * torch.einsum("bhcd,hmd->bhcm", k_chunk, q_bank)
            p_block = torch.exp(enc_scores_chunk - lse_block.unsqueeze(2))
            raw_weights = p_block * a_block_blk.unsqueeze(2)

            # The local numerator is a weighted sum of encoder values, so dV is just a
            # contraction of those raw weights with the numerator gradient.
            dV_tokens[:, :, token_start:token_end, :] = torch.einsum("bhcm,bhmd->bhcd", raw_weights, dB_block_blk)

            # Differentiate the same local numerator w.r.t. the encoder scores, then push
            # those score gradients back into encoder K and shared latent-bank Q.
            v_proj = torch.einsum("bhcd,bhmd->bhcm", v_chunk, dB_block_blk)
            dS_enc = raw_weights * (dA_block_blk.unsqueeze(2) + v_proj)

            dK_tokens[:, :, token_start:token_end, :] = scale * torch.einsum("bhcm,hmd->bhcd", dS_enc, q_bank)
            dQ_enc += scale * torch.einsum("bhcm,bhcd->hmd", dS_enc, k_chunk)

    # Merge the encoder-side latent-bank gradient with any shared decoder contribution,
    # then convert everything back to the public layouts/dtypes expected by callers.
    dQ = (dQ_enc + dQ_shared).to(Q.dtype)
    dK = (dK_tokens + dK_shared_tokens).permute(0, 2, 1, 3).contiguous().to(K.dtype)
    dV = dV_tokens.permute(0, 2, 1, 3).contiguous().to(V.dtype)
    dQ_dec = dQ_dec_tokens.permute(0, 2, 1, 3).contiguous().to(Q_dec.dtype) if dQ_dec_tokens is not None else None
    dK_dec = dK_dec.to(K_dec.dtype) if dK_dec is not None else None

    if not return_aux:
        return dQ, dK, dV, dQ_dec, dK_dec
    return (dQ, dK, dV, dQ_dec, dK_dec), {
        "Y": state["Y"],
        "LSE_dec": lse_dec,
        "LSE_enc": lse_enc,
        "z_block": z_block,
        "dZ_block": dZ_block,
        "a_block": a_block,
        "dA_block": dA_block,
        "dB_block": dB_block,
    }


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
    _return_state: bool = False,
):
    """Reference semi-autoregressive / block-causal forward pass.

    The implementation intentionally follows the same three-phase structure as
    `SemiAutoRegressiveFLARE.forward` in the Triton path:
    1. prepare: summarize each encoder block into local max/den/num statistics
    2. scan_block_z: inclusive-scan those local summaries into one normalized `z_block` per block
    3. output: compute decoder LSEs and outputs using only the per-block `z_block` summaries

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

    # Canonicalize everything into the layouts the three phases operate on:
    # - encoder tokens: [B, H, N, ...]
    # - decoder queries: [B, H, N, D_score]
    # - latent bank: [H, M, D_score]
    # The reference keeps intermediates in FP32 so the local and scanned softmax state is
    # easy to compare against the Triton path without BF16/FP16 accumulator noise.
    Q_f = Q.to(compute_dtype)
    K_f = K.to(compute_dtype)
    V_f = V.to(compute_dtype)
    K_tokens = K_f.permute(0, 2, 1, 3).contiguous()
    V_tokens = V_f.permute(0, 2, 1, 3).contiguous()
    q_dec_comp = Q_dec.to(compute_dtype) if separate_Q_dec else K_f
    k_dec_comp = K_dec.to(compute_dtype) if separate_K_dec else Q_f
    Q_dec_tokens = q_dec_comp.permute(0, 2, 1, 3).contiguous()
    K_dec_bank = k_dec_comp

    # The encoder prepare loop works in block/chunk coordinates because `chunk_size` is
    # still part of the public block-causal reference contract today.
    Kc = K_f.reshape(B, num_blocks, chunks_per_block, chunk_size, H, D_score).permute(0, 4, 1, 2, 3, 5).contiguous()
    Vc = V_f.reshape(B, num_blocks, chunks_per_block, chunk_size, H, D_value).permute(0, 4, 1, 2, 3, 5).contiguous()

    # -------------------------------------------------------------------------
    # Phase 1: prepare
    # -------------------------------------------------------------------------
    # Compute one local encoder summary per block:
    #   block_max[g, m]
    #   block_den[g, m]
    #   block_num[g, m, d]
    #
    # These summarize only tokens inside block `g`. No cross-block information is mixed
    # here yet. This directly matches the role of the Triton prepare kernel.
    bh = B * H
    block_max = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)
    block_den = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)
    block_num = torch.empty((B, H, num_blocks, M, D_value), device=Q.device, dtype=compute_dtype)

    for block_idx in range(num_blocks):
        max_block = torch.full((B, H, M), -float("inf"), device=Q.device, dtype=stats_dtype)
        den_block = torch.zeros((B, H, M), device=Q.device, dtype=stats_dtype)
        num_block = torch.zeros((B, H, M, D_value), device=Q.device, dtype=compute_dtype)

        for local_chunk in range(chunks_per_block):
            k_chunk = Kc[:, :, block_idx, local_chunk, :, :]
            v_chunk = Vc[:, :, block_idx, local_chunk, :, :]

            # Score each token in this encoder chunk against every latent in the shared bank.
            # Shape: [B, H, chunk_size, M]
            score_chunk = scale * torch.einsum("bhcd,hmd->bhcm", k_chunk, Q_f)
            chunk_max = score_chunk.max(dim=2).values.to(stats_dtype)
            chunk_exp = torch.exp(score_chunk - chunk_max.to(compute_dtype).unsqueeze(2))
            chunk_den = chunk_exp.to(stats_dtype).sum(dim=2)
            chunk_num = torch.bmm(
                chunk_exp.reshape(bh, chunk_size, M).transpose(1, 2),
                v_chunk.reshape(bh, chunk_size, D_value),
            ).reshape(B, H, M, D_value)

            # Merge the current chunk into the running block-local softmax state using the
            # standard stable max/denominator rescaling formula.
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

    # -------------------------------------------------------------------------
    # Phase 2: scan_block_z
    # -------------------------------------------------------------------------
    # Inclusive-scan the local block summaries so block `g` sees all encoder tokens from
    # blocks `<= g`. This produces the encoder-side outputs consumed by backward/output:
    #   LSE_enc[g, m]
    #   z_block[g, m, d]
    prefix_max = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)
    prefix_den = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)
    full_block_max = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)
    full_block_den = torch.empty((B, H, num_blocks, M), device=Q.device, dtype=stats_dtype)
    full_block_num = torch.empty((B, H, num_blocks, M, D_value), device=Q.device, dtype=compute_dtype)

    max_curr = torch.full((B, H, M), -float("inf"), device=Q.device, dtype=stats_dtype)
    den_curr = torch.zeros((B, H, M), device=Q.device, dtype=stats_dtype)
    num_curr = torch.zeros((B, H, M, D_value), device=Q.device, dtype=compute_dtype)

    for block_idx in range(num_blocks):
        # Save the exclusive prefix before folding in the current block. These match the
        # corresponding saved forward summaries in the Triton path.
        prefix_max[:, :, block_idx, :] = max_curr
        prefix_den[:, :, block_idx, :] = den_curr

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
        full_block_num[:, :, block_idx, :, :] = num_curr

    LSE_enc = torch.log(full_block_den.clamp_min(1e-20)) + full_block_max
    z_block = full_block_num * torch.exp(full_block_max - LSE_enc).to(compute_dtype).unsqueeze(-1)

    # -------------------------------------------------------------------------
    # Phase 3: output
    # -------------------------------------------------------------------------
    # Decoder-side work now depends only on:
    # - per-token decode queries
    # - the latent key bank
    # - the per-block encoder summary `z_block`
    #
    # This is the main debugging-friendly structural point: once phase 2 is done, output
    # never replays raw encoder tokens.
    q_dec_blocks = Q_dec_tokens.reshape(B, H, num_blocks, block_size, D_score)
    dec_scores = scale * torch.einsum("bhgtd,hmd->bhgtm", q_dec_blocks, K_dec_bank)
    lse_dec_blocks = torch.logsumexp(dec_scores.to(stats_dtype), dim=-1)
    alpha = torch.exp(dec_scores - lse_dec_blocks.to(compute_dtype).unsqueeze(-1))
    Y_blocks = torch.einsum("bhgtm,bhgmd->bhgtd", alpha, z_block)
    Y = Y_blocks.reshape(B, H, N, D_value).permute(0, 2, 1, 3).to(out_dtype)

    if _return_state:
        return {
            "B": B,
            "N": N,
            "H": H,
            "M": M,
            "D_score": D_score,
            "D_value": D_value,
            "block_size": block_size,
            "chunk_size": chunk_size,
            "num_blocks": num_blocks,
            "chunks_per_block": chunks_per_block,
            "scale": scale,
            "Q_f": Q_f,
            "K_f": K_f,
            "V_f": V_f,
            "K_tokens": K_tokens,
            "V_tokens": V_tokens,
            "Q_dec_tokens": Q_dec_tokens,
            "K_dec_bank": K_dec_bank,
            "Kc": Kc,
            "Vc": Vc,
            "Y": Y,
            "LSE_dec": lse_dec_blocks.reshape(B, H, N),
            "LSE_enc": LSE_enc,
            "z_block": z_block,
            "prefix_max": prefix_max,
            "prefix_den": prefix_den,
            "block_max": block_max,
            "block_den": block_den,
            "block_num": block_num,
            "separate_Q_dec": separate_Q_dec,
            "separate_K_dec": separate_K_dec,
            "weight_sharing_enc_dec": weight_sharing_enc_dec,
        }

    if not return_aux:
        return Y
    return Y, {"LSE_dec": lse_dec_blocks.reshape(B, H, N), "LSE_enc": LSE_enc}


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
            return semi_autoregressive_flare_reference(Q, K, V, block_size=block_size, scale=scale)

        y_ref, ref_ms = _benchmark_callable(run_ref, warmup=1, iters=max(1, min(2, iters)), device=device)
        delta = (y_impl.float() - y_ref.float()).abs()
        results["reference_ms"] = ref_ms
        results["reference_max_abs"] = float(delta.max().item())
        results["reference_mean_abs"] = float(delta.mean().item())

    return results

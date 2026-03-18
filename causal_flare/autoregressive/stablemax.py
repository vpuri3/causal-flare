from causal_flare._common import *
from causal_flare._reference_utils import (
    resolve_flare_causal_decode_inputs as _resolve_flare_causal_decode_inputs,
    validate_flare_qkv_layouts as _validate_flare_qkv_layouts,
)


def _resolve_stablemax_chunk_size(N: int, M: int, D_score: int, chunk_size) -> int:
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


def _stablemax_score_transform(x: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    one = torch.ones((), device=x.device, dtype=x.dtype)
    power_tensor = torch.as_tensor(power, device=x.device, dtype=x.dtype)
    pos_base = torch.where(x >= 0, x + one, one)
    neg_base = torch.where(x < 0, one - x, one)
    pos = torch.pow(pos_base, power_tensor)
    neg = torch.pow(neg_base, -power_tensor)
    return torch.where(x >= 0, pos, neg)


def _stablemax_score_transform_grad(x: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    one = torch.ones((), device=x.device, dtype=x.dtype)
    power_tensor = torch.as_tensor(power, device=x.device, dtype=x.dtype)
    pos_base = torch.where(x >= 0, x + one, one)
    neg_base = torch.where(x < 0, one - x, one)
    pos = power_tensor * torch.pow(pos_base, power_tensor - one)
    neg = power_tensor * torch.pow(neg_base, -(power_tensor + one))
    return torch.where(x >= 0, pos, neg)


def _broadcast_stablemax_gate_param(param, *, H: int, M: int, device: torch.device, dtype: torch.dtype, name: str) -> torch.Tensor:
    if torch.is_tensor(param):
        param_t = param.to(device=device, dtype=dtype)
    else:
        param_t = torch.as_tensor(param, device=device, dtype=dtype)
    if param_t.ndim == 0:
        return param_t.view(1, 1, 1, 1, 1)
    if param_t.ndim == 1 and param_t.shape[0] == H:
        return param_t.view(1, H, 1, 1, 1)
    if param_t.ndim == 2 and param_t.shape == (H, M):
        return param_t.view(1, H, 1, 1, M)
    raise ValueError(f"{name} must be scalar, [H], or [H, M]. Got {tuple(param_t.shape)}")


def _reduce_broadcast_stablemax_gate_grad(grad: torch.Tensor, param: torch.Tensor, *, H: int, M: int, name: str) -> torch.Tensor:
    if param.ndim == 0:
        return grad.sum().to(dtype=param.dtype)
    if param.ndim == 1 and param.shape[0] == H:
        return grad.sum(dim=(0, 2, 3, 4)).to(dtype=param.dtype)
    if param.ndim == 2 and param.shape == (H, M):
        return grad.sum(dim=(0, 2, 3)).to(dtype=param.dtype)
    raise ValueError(f"{name} must be scalar, [H], or [H, M]. Got {tuple(param.shape)}")


def _affine_prefix_scan_chunkwise(A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scan_A = A.clone()
    scan_B = B.clone()
    length = A.size(-2)
    step = 1
    while step < length:
        shifted_A = torch.cat([torch.ones_like(scan_A[..., :step, :]), scan_A[..., :-step, :]], dim=-2)
        shifted_B = torch.cat([torch.zeros_like(scan_B[..., :step, :, :]), scan_B[..., :-step, :, :]], dim=-3)
        scan_B = scan_B + scan_A.unsqueeze(-1) * shifted_B
        scan_A = scan_A * shifted_A
        step <<= 1
    return scan_A, scan_B


def _stablemax_write_gate(
    score_chunk: torch.Tensor,
    *,
    H: int,
    M: int,
    valid_chunk: torch.Tensor | None,
    write_gate_mode: str,
    write_gate_init: str,
    write_gate_fixed_value: float | None,
    write_gate_scale,
    write_gate_bias,
) -> torch.Tensor:
    if write_gate_fixed_value is not None:
        gate_value = float(write_gate_fixed_value)
        if gate_value < 0.0 or gate_value > 1.0:
            raise ValueError(f"write_gate_fixed_value must lie in [0, 1]. Got {gate_value}")
        gate = torch.full_like(score_chunk, gate_value)
    else:
        if write_gate_mode != "postnorm_sigmoid":
            raise ValueError(f"Unsupported write_gate_mode={write_gate_mode!r}. Expected 'postnorm_sigmoid'.")
        if write_gate_init not in {"identity", "zero"}:
            raise ValueError(f"Unsupported write_gate_init={write_gate_init!r}. Expected 'identity' or 'zero'.")
        if write_gate_scale is None and write_gate_bias is None and write_gate_init == "identity":
            gate = torch.ones_like(score_chunk)
        else:
            default_scale = 0.0
            default_bias = 6.0 if write_gate_init == "identity" else 0.0
            scale_t = (
                torch.full((), default_scale, device=score_chunk.device, dtype=score_chunk.dtype).view(1, 1, 1, 1, 1)
                if write_gate_scale is None else
                _broadcast_stablemax_gate_param(
                    write_gate_scale,
                    H=H,
                    M=M,
                    device=score_chunk.device,
                    dtype=score_chunk.dtype,
                    name="write_gate_scale",
                )
            )
            bias_t = (
                torch.full((), default_bias, device=score_chunk.device, dtype=score_chunk.dtype).view(1, 1, 1, 1, 1)
                if write_gate_bias is None else
                _broadcast_stablemax_gate_param(
                    write_gate_bias,
                    H=H,
                    M=M,
                    device=score_chunk.device,
                    dtype=score_chunk.dtype,
                    name="write_gate_bias",
                )
            )
            gate = torch.sigmoid(score_chunk * scale_t + bias_t)
    if valid_chunk is not None:
        gate = gate * valid_chunk.to(dtype=gate.dtype)
    return gate


class FLAREAutoregressiveStablemaxPyTorch(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Q_dec, K_dec, scale=None, chunk_size=None, power: float = 2.0):
        compute_dtype = torch.float32
        if os.environ.get("FLARE_PYTORCH_MATCH_REFERENCE", "") == "1":
            compute_dtype = Q.dtype

        B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="StableMax Chunked FLARE")
        scale = _resolve_attn_scale(scale, D_score)
        Q_dec_resolved, K_dec_resolved, separate_Q_dec, separate_K_dec, _ = _resolve_flare_causal_decode_inputs(
            Q, K, Q_dec, K_dec
        )
        device = Q.device
        out_dtype = V.dtype
        scale_f = float(scale)
        power = float(power)

        Q_f = Q.to(compute_dtype)
        K_f = K.to(compute_dtype)
        V_f = V.to(compute_dtype)
        if separate_Q_dec:
            Q_dec_f = Q_dec_resolved.to(compute_dtype).permute(0, 2, 1, 3).contiguous()
        else:
            Q_dec_f = K_f.permute(0, 2, 1, 3).contiguous()
        if separate_K_dec:
            K_dec_f = K_dec_resolved.to(compute_dtype).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        else:
            K_dec_f = Q_f.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

        C = _resolve_stablemax_chunk_size(N, M, D_score, chunk_size)
        NC = math.ceil(N / C) if N > 0 else 0
        PADDED_LEN = NC * C
        PAD = PADDED_LEN - N
        if PAD > 0:
            K_f = torch.cat([K_f, torch.zeros((B, PAD, H, D_score), device=device, dtype=compute_dtype)], dim=1)
            V_f = torch.cat([V_f, torch.zeros((B, PAD, H, D_value), device=device, dtype=compute_dtype)], dim=1)
            Q_dec_f = torch.cat([Q_dec_f, torch.zeros((B, H, PAD, D_score), device=device, dtype=compute_dtype)], dim=2)

        if N == 0:
            ctx.empty = True
            ctx.b = B
            ctx.q_dtype = Q.dtype
            ctx.k_dtype = K.dtype
            ctx.v_dtype = V.dtype
            ctx.q_dec_dtype = Q_dec_resolved.dtype
            ctx.k_dec_dtype = K_dec_resolved.dtype
            ctx.n = 0
            ctx.h = H
            ctx.m = M
            ctx.d_score = D_score
            ctx.d_value = D_value
            ctx.padded_len = 0
            ctx.c = C
            ctx.nc = 0
            ctx.bhnc = 0
            ctx.scale = scale_f
            ctx.power = power
            ctx.save_for_backward()
            return torch.empty((B, 0, H, D_value), device=device, dtype=out_dtype)

        Kc = K_f.reshape(B, NC, C, H, D_score).permute(0, 3, 1, 2, 4).contiguous()
        Vc = V_f.reshape(B, NC, C, H, D_value).permute(0, 3, 1, 2, 4).contiguous()
        Q_dec_c = Q_dec_f.reshape(B, H, NC, C, D_score).contiguous()
        BHNC = B * H * NC

        ctx.empty = False
        ctx.scale = scale_f
        ctx.power = power
        ctx.chunk_size = C
        ctx.compute_dtype = compute_dtype
        ctx.q_dtype = Q.dtype
        ctx.k_dtype = K.dtype
        ctx.v_dtype = V.dtype
        ctx.q_dec_dtype = Q_dec_resolved.dtype
        ctx.k_dec_dtype = K_dec_resolved.dtype
        ctx.n = N
        ctx.h = H
        ctx.d_score = D_score
        ctx.d_value = D_value
        ctx.padded_len = PADDED_LEN
        ctx.c = C
        ctx.nc = NC
        ctx.bhnc = BHNC

        # Run the packed stablemax prefill path directly in the autograd Function so
        # the implementation is self-contained and does not depend on external
        # preparation/forward helpers.
        #
        # High-level math
        # ----------------
        # This computes
        #
        #   y_t = sum_m p_dec[m, t] * (sum_{tau <= t} s(a_enc[m, tau]) v_tau) / (sum_{tau <= t} s(a_enc[m, tau]))
        #
        # where:
        # - a_enc[m, tau] = scale * <k_tau, q_m>
        # - s(.) is the stablemax score transform
        # - p_dec[:, t] = softmax(scale * <q_dec_t, k_dec_m>)
        #
        # Tensor layout:
        # - Q_f:     [H, M, D_score]
        # - Kc:      [B, H, NC, C, D_score]
        # - Vc:      [B, H, NC, C, D_value]
        # - Q_dec_c: [B, H, NC, C, D_score]
        # - K_dec_f: [B, H, M, D_score]

        # Phase 1: per-chunk stablemax encoder summaries.
        # - score_chunk:        [B, H, NC, C, M]
        # - stable_score_chunk: [B, H, NC, C, M]
        # - score_chunk_den:    [B, H, NC, M]
        # - score_chunk_num:    [B, H, NC, M, D_value]
        score_chunk = scale_f * torch.einsum("bhncd,hmd->bhncm", Kc, Q_f)
        stable_score_chunk = _stablemax_score_transform(score_chunk, power=power)
        score_chunk_den = stable_score_chunk.sum(dim=3)
        score_chunk_num = torch.bmm(
            stable_score_chunk.reshape(BHNC, C, M).transpose(1, 2),
            Vc.reshape(BHNC, C, D_value),
        ).reshape(B, H, NC, M, D_value)

        # Chunk prefixes over the encoder statistics.
        # - score_prev_den: [B, H, NC, M]
        # - score_prev_num: [B, H, NC, M, D_value]
        score_prev_den = torch.cumsum(score_chunk_den, dim=2) - score_chunk_den
        score_prev_num = torch.cumsum(score_chunk_num, dim=2) - score_chunk_num

        # Phase 2: decoder softmax over latent slots.
        # - decode_logits: [B, H, NC, C, M]
        # - decode_probs:  [B, H, NC, C, M]
        decode_logits = scale_f * torch.einsum("bhncd,bhmd->bhncm", Q_dec_c, K_dec_f)
        decode_probs = torch.softmax(decode_logits, dim=-1)

        # Phase 3A: token-wise encoder normalization seen by each decoder query.
        # - chunk_den:       [B, H, NC, C, M]
        # - total_den:       [B, H, NC, C, M]
        # - decode_over_den: [B, H, NC, C, M]
        chunk_den = stable_score_chunk.cumsum(dim=3)
        total_den = score_prev_den.unsqueeze(3) + chunk_den
        total_den_safe = torch.where(total_den > 0, total_den, torch.ones_like(total_den))
        inv_total_den = total_den_safe.reciprocal()
        decode_over_den = decode_probs * inv_total_den

        # Phase 3B: reduce over the latent dimension before applying values.
        # - prefix_out: [B, H, NC, C, D_value]
        # - local_mix:  [B, H, NC, C, C]
        # - local_out:  [B, H, NC, C, D_value]
        prefix_out = torch.einsum("bhncm,bhnmd->bhncd", decode_over_den, score_prev_num)
        causal_mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool)).view(1, 1, 1, C, C)
        local_mix = torch.matmul(decode_over_den, stable_score_chunk.transpose(-1, -2))
        local_mix = torch.where(causal_mask, local_mix, torch.zeros_like(local_mix))
        local_out = torch.matmul(local_mix.reshape(BHNC, C, C), Vc.reshape(BHNC, C, D_value)).reshape(B, H, NC, C, D_value)
        Yc = prefix_out + local_out

        Y_out = Yc.reshape(B, H, PADDED_LEN, D_value)[:, :, :N, :].permute(0, 2, 1, 3).to(out_dtype)
        _check_finite("FLAREAutoregressiveStablemaxPyTorch.Y", Y_out)

        ctx.save_for_backward(
            Q_f,
            Kc,
            Vc,
            Q_dec_c,
            K_dec_f,
            score_chunk,
            stable_score_chunk,
            decode_probs,
            score_prev_num,
            inv_total_den,
        )
        return Y_out

    @staticmethod
    def backward(ctx, dY):
        """Backward for the chunked stablemax prefill path.

        High-level differentiation structure
        ------------------------------------
        The forward can be written as

            Y = prefix_out + local_out

        with

            prefix_out[c] = sum_m R[c, m] * prev_num[m]
            local_out[c]  = sum_tau local_mix[c, tau] * V[tau]
            local_mix     = tril(R @ W^T)
            R             = decode_probs / total_den

        where:
        - `W = stable_score_chunk`
        - `prev_num` and `prev_den` are strict chunk prefixes of encoder statistics
        - `total_den = prev_den + cumsum(W, dim=token)`

        The backward therefore naturally factors into:
        1. Differentiate the output wrt `R`, `W`, `prev_num`, and `V`.
        2. Differentiate `R = decode_probs / total_den`.
        3. Backprop through the chunk-prefix scans producing `prev_num` / `prev_den`.
        4. Backprop through the stablemax transform and the encoder/decoder projections.

        Saved tensor shapes
        -------------------
        - `Q_f`:                [H, M, D_score]
        - `Kc`:                 [B, H, NC, C, D_score]
        - `Vc`:                 [B, H, NC, C, D_value]
        - `Q_dec_c`:            [B, H, NC, C, D_score]
        - `K_dec_f`:            [B, H, M, D_score]
        - `score_chunk`:        [B, H, NC, C, M]
        - `stable_score_chunk`: [B, H, NC, C, M]
        - `decode_probs`:       [B, H, NC, C, M]
        - `score_prev_num`:     [B, H, NC, M, D_value]
        - `inv_total_den`:      [B, H, NC, C, M]

        Incoming gradient:
        - `dY`: [B, N, H, D_value]
        """
        if getattr(ctx, "empty", False):
            B = ctx.b
            H = ctx.h
            M = ctx.m
            N = ctx.n
            D_score = ctx.d_score
            D_value = ctx.d_value
            empty_q = torch.zeros((H, M, D_score), device=dY.device, dtype=ctx.q_dtype)
            empty_kv = torch.zeros((B, N, H, D_score), device=dY.device, dtype=ctx.k_dtype)
            empty_v = torch.zeros((B, N, H, D_value), device=dY.device, dtype=ctx.v_dtype)
            empty_q_dec = torch.zeros((B, N, H, D_score), device=dY.device, dtype=ctx.q_dec_dtype)
            empty_k_dec = torch.zeros((H, M, D_score), device=dY.device, dtype=ctx.k_dec_dtype)
            return empty_q, empty_kv, empty_v, empty_q_dec, empty_k_dec, None, None, None
        (
            Q_f,
            Kc,
            Vc,
            Q_dec_c,
            K_dec_f,
            score_chunk,
            stable_score_chunk,
            decode_probs,
            score_prev_num,
            inv_total_den,
        ) = ctx.saved_tensors
        B = Kc.size(0)
        H = ctx.h
        N = ctx.n
        C = ctx.c
        NC = ctx.nc
        M = Q_f.size(1)
        D_score = ctx.d_score
        D_value = ctx.d_value
        PADDED_LEN = ctx.padded_len
        BHNC = ctx.bhnc
        scale_f = ctx.scale
        power = ctx.power
        device = Kc.device

        if N == 0:
            empty_q = torch.zeros((H, M, D_score), device=device, dtype=ctx.q_dtype)
            empty_kv = torch.zeros((B, N, H, D_score), device=device, dtype=ctx.k_dtype)
            empty_v = torch.zeros((B, N, H, D_value), device=device, dtype=ctx.v_dtype)
            empty_q_dec = torch.zeros((B, N, H, D_score), device=device, dtype=ctx.q_dec_dtype)
            empty_k_dec = torch.zeros((H, M, D_score), device=device, dtype=ctx.k_dec_dtype)
            return empty_q, empty_kv, empty_v, empty_q_dec, empty_k_dec, None, None, None

        stable_score_grad = _stablemax_score_transform_grad(score_chunk, power=power)

        # Reconstruct the forward-side encoder statistics needed by backward.
        #
        # Shapes:
        # - score_chunk_den: [B, H, NC, M]
        # - score_chunk_num: [B, H, NC, M, D_value]
        # - score_prev_den:  [B, H, NC, M]
        score_chunk_den = stable_score_chunk.sum(dim=3)
        score_chunk_num = torch.bmm(
            stable_score_chunk.reshape(BHNC, C, M).transpose(1, 2),
            Vc.reshape(BHNC, C, D_value),
        ).reshape(B, H, NC, M, D_value)
        score_prev_den = torch.cumsum(score_chunk_den, dim=2) - score_chunk_den
        decode_over_den = decode_probs * inv_total_den

        # local_mix[c, tau] = sum_m decode_over_den[c, m] * stable_score_chunk[tau, m],
        # masked so only tau <= c contributes.
        # Shape: [B, H, NC, C, C]
        causal_mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool)).view(1, 1, 1, C, C)
        local_mix = torch.matmul(decode_over_den, stable_score_chunk.transpose(-1, -2))
        local_mix = torch.where(causal_mask, local_mix, torch.zeros_like(local_mix))

        # Re-pack dY into chunk layout to match the forward intermediates.
        # dYc shape: [B, H, NC, C, D_value]
        dY_f = dY.to(torch.float32)
        if PADDED_LEN != N:
            dY_f = torch.cat(
                [dY_f, torch.zeros((B, PADDED_LEN - N, H, D_value), device=device, dtype=torch.float32)],
                dim=1,
            )
        dYc = dY_f.permute(0, 2, 1, 3).reshape(B, H, NC, C, D_value).contiguous()

        # Step 1: differentiate Y = prefix_out + local_out.
        #
        # prefix_out = einsum(decode_over_den, score_prev_num)
        # local_out  = local_mix @ V
        #
        # This yields gradients for:
        # - decode_over_den: [B, H, NC, C, M]
        # - prev_num:        [B, H, NC, M, D_value]
        # - local_mix:       [B, H, NC, C, C]
        # - V:               [B, H, NC, C, D_value]
        grad_decode_over_den = torch.einsum("bhncd,bhnmd->bhncm", dYc, score_prev_num)
        grad_prev_num = torch.einsum("bhncm,bhncd->bhnmd", decode_over_den, dYc)
        grad_local_mix = torch.matmul(dYc, Vc.transpose(-1, -2))
        grad_local_mix = torch.where(causal_mask, grad_local_mix, torch.zeros_like(grad_local_mix))
        grad_V_phase3 = torch.matmul(local_mix.transpose(-1, -2).reshape(BHNC, C, C), dYc.reshape(BHNC, C, D_value)).reshape(B, H, NC, C, D_value)

        # local_mix = tril(decode_over_den @ stable_score_chunk^T)
        #
        # Differentiate the masked [C, C] matmul:
        # - extra grad into decode_over_den from mixing with stable_score_chunk
        # - phase-3 grad into stable_score_chunk from mixing with decode_over_den
        grad_decode_over_den = grad_decode_over_den + torch.matmul(grad_local_mix, stable_score_chunk)
        grad_w_phase3 = torch.matmul(grad_local_mix.transpose(-1, -2), decode_over_den)

        # Step 2: differentiate R = decode_probs / total_den.
        #
        # Elementwise:
        #   R = P * inv_total_den
        # so:
        #   dP          += dR * inv_total_den
        #   dtotal_den  += -(dR * P) / total_den^2
        #
        # Shapes:
        # - grad_decode_probs: [B, H, NC, C, M]
        # - grad_total_den:    [B, H, NC, C, M]
        grad_decode_probs = grad_decode_over_den * inv_total_den
        grad_total_den = -(grad_decode_over_den * decode_probs) * inv_total_den.square()

        # total_den = prev_den.unsqueeze(token) + cumsum(stable_score_chunk, dim=token)
        #
        # Gradient wrt prev_den is the sum over token positions in the chunk.
        # Gradient wrt the within-chunk stable weights is a reverse cumsum over token
        # positions, because each encoder token contributes to all later targets.
        grad_prev_den = grad_total_den.sum(dim=3)
        grad_w_phase3 = grad_w_phase3 + torch.flip(torch.cumsum(torch.flip(grad_total_den, dims=(3,)), dim=3), dims=(3,))

        # Step 3: differentiate decoder softmax.
        #
        # For P = softmax(L), the Jacobian-vector product is:
        #   dL = P * (dP - <dP, P>)
        #
        # Shape: [B, H, NC, C, M]
        softmax_dot = (grad_decode_probs * decode_probs).sum(dim=-1, keepdim=True)
        grad_score_dec = decode_probs * (grad_decode_probs - softmax_dot)

        # Step 4: backprop through chunk-prefix scans.
        #
        # Forward used strict prefixes:
        #   prev_den[n] = sum_{n' < n} score_chunk_den[n']
        #   prev_num[n] = sum_{n' < n} score_chunk_num[n']
        #
        # Their reverse-mode adjoints are strict suffix sums:
        #   dscore_chunk_*[n] = sum_{n' > n} dprev_*[n']
        #
        # Shapes:
        # - grad_score_chunk_den: [B, H, NC, M]
        # - grad_score_chunk_num: [B, H, NC, M, D_value]
        grad_score_chunk_den = torch.flip(torch.cumsum(torch.flip(grad_prev_den, dims=(2,)), dim=2), dims=(2,)) - grad_prev_den
        grad_score_chunk_num = torch.flip(torch.cumsum(torch.flip(grad_prev_num, dims=(2,)), dim=2), dims=(2,)) - grad_prev_num

        # Step 5: combine all encoder-weight gradients.
        #
        # stable_score_chunk participates in three places:
        # 1. directly in the local chunk mixing (`grad_w_phase3`)
        # 2. in score_chunk_den = sum_c stable_score_chunk
        # 3. in score_chunk_num = stable_score_chunk^T @ V
        #
        # Resulting shapes:
        # - grad_w_total: [B, H, NC, C, M]
        # - grad_V_total: [B, H, NC, C, D_value]
        grad_w_total = grad_w_phase3 + grad_score_chunk_den.unsqueeze(3)
        grad_w_total = grad_w_total + torch.einsum("bhnmd,bhncd->bhncm", grad_score_chunk_num, Vc)
        grad_V_total = grad_V_phase3 + torch.einsum("bhncm,bhnmd->bhncd", stable_score_chunk, grad_score_chunk_num)

        # Step 6: differentiate the stablemax transform and the linear score projections.
        #
        # grad_score_enc has shape [B, H, NC, C, M].
        # It contracts with:
        # - encoder queries Q_f: [H, M, D_score]
        # - encoder keys   Kc:   [B, H, NC, C, D_score]
        #
        # grad_score_dec has the same [B, H, NC, C, M] shape and contracts with:
        # - decoder queries Q_dec_c: [B, H, NC, C, D_score]
        # - decoder keys   K_dec_f:  [B, H, M, D_score]
        grad_score_enc = grad_w_total * stable_score_grad
        grad_Q_dec_c = scale_f * torch.einsum("bhncm,bhmd->bhncd", grad_score_dec, K_dec_f)
        grad_K_dec = (scale_f * torch.einsum("bhncm,bhncd->bhmd", grad_score_dec, Q_dec_c)).sum(dim=0)

        # Unpack chunked gradients back to the public tensor layouts:
        # - grad_Q:     [H, M, D_score]
        # - grad_K:     [B, N, H, D_score]
        # - grad_V:     [B, N, H, D_value]
        # - grad_Q_dec: [B, N, H, D_score]
        # - grad_K_dec: [H, M, D_score]
        grad_Q = scale_f * torch.einsum("bhncm,bhncd->hmd", grad_score_enc, Kc)
        grad_Kc = scale_f * torch.einsum("bhncm,hmd->bhncd", grad_score_enc, Q_f)
        grad_K = grad_Kc.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_score)[:, :N, :, :]
        grad_V = grad_V_total.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_value)[:, :N, :, :]
        grad_Q_dec = grad_Q_dec_c.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_score)[:, :N, :, :]
        return (
            grad_Q.to(ctx.q_dtype),
            grad_K.to(ctx.k_dtype),
            grad_V.to(ctx.v_dtype),
            grad_Q_dec.to(ctx.q_dec_dtype),
            grad_K_dec.to(ctx.k_dec_dtype),
            None,
            None,
            None,
        )

def flare_autoregressive_stablemax_pytorch(
    Q,
    K,
    V,
    scale=None,
    eps=None,
    profile: bool = False,
    chunk_size=None,
    Q_dec=None,
    K_dec=None,
    power: float = 2.0,
    write_gate: bool = False,
    write_gate_mode: str = "postnorm_sigmoid",
    write_gate_init: str = "identity",
    write_gate_fixed_value: float | None = None,
    write_gate_scale=None,
    write_gate_bias=None,
    state: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    return_state: bool = False,
):
    Q_dec_resolved, K_dec_resolved, _, _, _ = _resolve_flare_causal_decode_inputs(
        Q, K, Q_dec, K_dec
    )
    if profile:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch does not support profile=True")
    if state is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch does not support recurrent state")
    if attention_mask is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch does not support attention_mask")
    if return_state:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch does not support return_state=True")
    if not write_gate:
        if write_gate_fixed_value is not None or write_gate_scale is not None or write_gate_bias is not None:
            raise ValueError("write_gate parameters require write_gate=True")
        return FLAREAutoregressiveStablemaxPyTorch.apply(Q, K, V, Q_dec_resolved, K_dec_resolved, scale, chunk_size, power)
    return FLAREAutoregressiveStablemaxWriteGatedPyTorch.apply(
        Q,
        K,
        V,
        Q_dec_resolved,
        K_dec_resolved,
        scale,
        chunk_size,
        power,
        write_gate_mode,
        write_gate_init,
        write_gate_fixed_value,
        write_gate_scale,
        write_gate_bias,
    )

class FLAREAutoregressiveStablemaxWriteGatedPyTorch(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        Q_dec,
        K_dec,
        scale=None,
        chunk_size=None,
        power: float = 2.0,
        write_gate_mode: str = "postnorm_sigmoid",
        write_gate_init: str = "identity",
        write_gate_fixed_value: float | None = None,
        write_gate_scale=None,
        write_gate_bias=None,
    ):
        compute_dtype = torch.float32
        if os.environ.get("FLARE_PYTORCH_MATCH_REFERENCE", "") == "1":
            compute_dtype = Q.dtype

        B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="StableMax Chunked FLARE Write-Gated")
        scale_resolved = _resolve_attn_scale(scale, D_score)
        Q_dec_resolved, K_dec_resolved, _, _, _ = _resolve_flare_causal_decode_inputs(Q, K, Q_dec, K_dec)

        ctx.empty = N == 0
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        ctx.power = power
        ctx.write_gate_mode = write_gate_mode
        ctx.write_gate_init = write_gate_init
        ctx.write_gate_fixed_value = write_gate_fixed_value
        ctx.compute_dtype = compute_dtype
        ctx.b = B
        ctx.n = N
        ctx.h = H
        ctx.m = M
        ctx.d_score = D_score
        ctx.d_value = D_value
        ctx.scale_resolved = float(scale_resolved)
        C = _resolve_stablemax_chunk_size(N, M, D_score, chunk_size)
        ctx.c = C
        ctx.nc = math.ceil(N / C) if N > 0 else 0
        ctx.padded_len = ctx.nc * C
        ctx.q_dtype = Q.dtype
        ctx.k_dtype = K.dtype
        ctx.v_dtype = V.dtype
        ctx.q_dec_dtype = Q_dec_resolved.dtype
        ctx.k_dec_dtype = K_dec_resolved.dtype
        ctx.has_gate_scale_tensor = torch.is_tensor(write_gate_scale)
        ctx.has_gate_bias_tensor = torch.is_tensor(write_gate_bias)
        ctx.write_gate_scale_const = None if ctx.has_gate_scale_tensor else write_gate_scale
        ctx.write_gate_bias_const = None if ctx.has_gate_bias_tensor else write_gate_bias

        device = Q.device
        out_dtype = V.dtype
        scale_f = float(scale_resolved)
        power = float(power)
        eps = torch.finfo(torch.float32).eps

        Q_f = Q.to(compute_dtype)
        K_f = K.to(compute_dtype)
        V_f = V.to(compute_dtype)
        if Q_dec_resolved is not K:
            Q_dec_f = Q_dec_resolved.to(compute_dtype).permute(0, 2, 1, 3).contiguous()
        else:
            Q_dec_f = K_f.permute(0, 2, 1, 3).contiguous()
        if K_dec_resolved is not Q:
            K_dec_f = K_dec_resolved.to(compute_dtype).unsqueeze(0).expand(B, -1, -1, -1).contiguous()
        else:
            K_dec_f = Q_f.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

        PADDED_LEN = ctx.padded_len
        PAD = PADDED_LEN - N
        valid_chunk = None
        if PAD > 0:
            K_f = torch.cat([K_f, torch.zeros((B, PAD, H, D_score), device=device, dtype=compute_dtype)], dim=1)
            V_f = torch.cat([V_f, torch.zeros((B, PAD, H, D_value), device=device, dtype=compute_dtype)], dim=1)
            Q_dec_f = torch.cat([Q_dec_f, torch.zeros((B, H, PAD, D_score), device=device, dtype=compute_dtype)], dim=2)
            valid_tokens = torch.cat(
                [
                    torch.ones((B, N), device=device, dtype=torch.bool),
                    torch.zeros((B, PAD), device=device, dtype=torch.bool),
                ],
                dim=1,
            )
            valid_chunk = valid_tokens.reshape(B, ctx.nc, C).unsqueeze(1).unsqueeze(-1)

        if N == 0:
            ctx.save_for_backward(
                Q_f,
                torch.empty((B, H, 0, C, D_score), device=device, dtype=compute_dtype),
                torch.empty((B, H, 0, C, D_value), device=device, dtype=compute_dtype),
                torch.empty((B, H, 0, C, D_score), device=device, dtype=compute_dtype),
                K_dec_f,
                torch.empty((B, H, 0, C, M), device=device, dtype=compute_dtype),
                torch.empty((B, H, 0, C, M), device=device, dtype=compute_dtype),
                torch.empty((B, H, 0, C, M), device=device, dtype=compute_dtype),
                torch.empty((B, H, 0, M), device=device, dtype=compute_dtype),
                torch.empty((B, H, 0, M, D_value), device=device, dtype=compute_dtype),
                *( [write_gate_scale] if ctx.has_gate_scale_tensor else [] ),
                *( [write_gate_bias] if ctx.has_gate_bias_tensor else [] ),
            )
            return torch.empty((B, 0, H, D_value), device=device, dtype=out_dtype)

        Kc = K_f.reshape(B, ctx.nc, C, H, D_score).permute(0, 3, 1, 2, 4).contiguous()
        Vc = V_f.reshape(B, ctx.nc, C, H, D_value).permute(0, 3, 1, 2, 4).contiguous()
        Q_dec_c = Q_dec_f.reshape(B, H, ctx.nc, C, D_score).contiguous()

        # Stablemax without write gating represents each latent by additive
        # encoder statistics `num / den`. Post-normalization write gating changes
        # the latent update to the affine recurrence
        #
        #   d_t     = d_{t-1} + s_t
        #   alpha_t = (s_t / d_t) * g_t
        #   z_t     = (1 - alpha_t) z_{t-1} + alpha_t v_t
        #
        # so the chunked execution now needs:
        # 1. denominator prefix scan
        # 2. per-chunk affine summaries `(A_chunk, B_chunk)`
        # 3. affine scan over chunks to recover chunk-start latents
        # 4. exact within-chunk output replay
        score_chunk = scale_f * torch.einsum("bhncd,hmd->bhncm", Kc, Q_f)
        stable_score_chunk = _stablemax_score_transform(score_chunk, power=power)
        if valid_chunk is not None:
            stable_score_chunk = stable_score_chunk * valid_chunk.to(dtype=stable_score_chunk.dtype)
        score_chunk_den = stable_score_chunk.sum(dim=3)
        score_prev_den = torch.cumsum(score_chunk_den, dim=2) - score_chunk_den

        gate_chunk = _stablemax_write_gate(
            score_chunk,
            H=H,
            M=M,
            valid_chunk=valid_chunk,
            write_gate_mode=write_gate_mode,
            write_gate_init=write_gate_init,
            write_gate_fixed_value=write_gate_fixed_value,
            write_gate_scale=write_gate_scale,
            write_gate_bias=write_gate_bias,
        )

        total_den = score_prev_den.unsqueeze(3) + stable_score_chunk.cumsum(dim=3)
        total_den_safe = total_den.clamp_min(eps)
        alpha_chunk = (stable_score_chunk / total_den_safe) * gate_chunk
        beta_chunk = 1.0 - alpha_chunk
        beta_safe = beta_chunk.clamp_min(eps)
        prefix_prod = torch.cumprod(beta_safe, dim=3)
        replay_source_coeff = alpha_chunk / prefix_prod

        suffix_beta_inclusive = torch.flip(torch.cumprod(torch.flip(beta_safe, dims=(3,)), dim=3), dims=(3,))
        suffix_beta_exclusive = torch.cat(
            [
                suffix_beta_inclusive[:, :, :, 1:, :],
                torch.ones_like(suffix_beta_inclusive[:, :, :, :1, :]),
            ],
            dim=3,
        )
        chunk_source_coeff = alpha_chunk * suffix_beta_exclusive
        chunk_A = prefix_prod[:, :, :, -1, :]
        chunk_B = torch.einsum("bhncm,bhncd->bhnmd", chunk_source_coeff, Vc)

        z_prev_chunk = torch.empty((B, H, ctx.nc, M, D_value), device=device, dtype=torch.float32)
        z_curr = torch.zeros((B, H, M, D_value), device=device, dtype=torch.float32)
        for chunk_idx in range(ctx.nc):
            z_prev_chunk[:, :, chunk_idx, :, :] = z_curr
            z_curr = chunk_A[:, :, chunk_idx, :].unsqueeze(-1) * z_curr + chunk_B[:, :, chunk_idx, :, :]

        decode_logits = scale_f * torch.einsum("bhncd,bhmd->bhncm", Q_dec_c, K_dec_f)
        decode_probs = torch.softmax(decode_logits, dim=-1)
        recurrent_factor = decode_probs * prefix_prod
        prefix_out = torch.einsum("bhncm,bhnmd->bhncd", recurrent_factor, z_prev_chunk)
        causal_mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool)).view(1, 1, 1, C, C)
        local_mix = torch.matmul(recurrent_factor, replay_source_coeff.transpose(-1, -2))
        local_mix = torch.where(causal_mask, local_mix, torch.zeros_like(local_mix))
        BHNC = B * H * ctx.nc
        local_out = torch.matmul(
            local_mix.reshape(BHNC, C, C),
            Vc.reshape(BHNC, C, D_value),
        ).reshape(B, H, ctx.nc, C, D_value)
        Yc = prefix_out + local_out
        if PAD > 0:
            Yc[:, :, -1, N - (ctx.nc - 1) * C :, :] = 0.0

        Y = Yc.reshape(B, H, PADDED_LEN, D_value)[:, :, :N, :].permute(0, 2, 1, 3).to(out_dtype)
        _check_finite("FLAREAutoregressiveStablemaxWriteGatedPyTorch.Y", Y)

        saved = [
            Q_f,
            Kc,
            Vc,
            Q_dec_c,
            K_dec_f,
            score_chunk,
            stable_score_chunk,
            decode_probs,
            score_prev_den,
            z_prev_chunk,
        ]
        if ctx.has_gate_scale_tensor:
            saved.append(write_gate_scale)
        if ctx.has_gate_bias_tensor:
            saved.append(write_gate_bias)
        ctx.save_for_backward(*saved)
        return Y

    @staticmethod
    def backward(ctx, dY):
        if getattr(ctx, "empty", False):
            empty_q = torch.zeros((ctx.h, ctx.m, ctx.d_score), device=dY.device, dtype=ctx.q_dtype)
            empty_k = torch.zeros((ctx.b, ctx.n, ctx.h, ctx.d_score), device=dY.device, dtype=ctx.k_dtype)
            empty_v = torch.zeros((ctx.b, ctx.n, ctx.h, ctx.d_value), device=dY.device, dtype=ctx.v_dtype)
            empty_q_dec = torch.zeros((ctx.b, ctx.n, ctx.h, ctx.d_score), device=dY.device, dtype=ctx.q_dec_dtype)
            empty_k_dec = torch.zeros((ctx.h, ctx.m, ctx.d_score), device=dY.device, dtype=ctx.k_dec_dtype)
            return (
                empty_q,
                empty_k,
                empty_v,
                empty_q_dec,
                empty_k_dec,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        saved = list(ctx.saved_tensors)
        Q_f = saved.pop(0)
        Kc = saved.pop(0)
        Vc = saved.pop(0)
        Q_dec_c = saved.pop(0)
        K_dec_f = saved.pop(0)
        score_chunk = saved.pop(0)
        stable_score_chunk = saved.pop(0)
        decode_probs = saved.pop(0)
        score_prev_den = saved.pop(0)
        z_prev_chunk = saved.pop(0)
        gate_scale_saved = saved.pop(0) if ctx.has_gate_scale_tensor else ctx.write_gate_scale_const
        gate_bias_saved = saved.pop(0) if ctx.has_gate_bias_tensor else ctx.write_gate_bias_const

        B = ctx.b
        N = ctx.n
        H = ctx.h
        M = ctx.m
        D_score = ctx.d_score
        D_value = ctx.d_value
        C = ctx.c
        NC = ctx.nc
        PADDED_LEN = ctx.padded_len
        scale_f = ctx.scale_resolved
        power = float(ctx.power)
        device = Kc.device
        compute_dtype = Q_f.dtype
        eps = torch.finfo(compute_dtype).eps

        valid_chunk = None
        PAD = PADDED_LEN - N
        if PAD > 0:
            valid_tokens = torch.cat(
                [
                    torch.ones((B, N), device=device, dtype=torch.bool),
                    torch.zeros((B, PAD), device=device, dtype=torch.bool),
                ],
                dim=1,
            )
            valid_chunk = valid_tokens.reshape(B, NC, C).unsqueeze(1).unsqueeze(-1)

        dY_f = dY.to(compute_dtype)
        if PAD > 0:
            dY_f = torch.cat(
                [dY_f, torch.zeros((B, PAD, H, D_value), device=device, dtype=compute_dtype)],
                dim=1,
            )
        dYc = dY_f.permute(0, 2, 1, 3).reshape(B, H, NC, C, D_value).contiguous()

        grad_decode_probs = torch.zeros_like(decode_probs)
        grad_score_enc = torch.zeros_like(score_chunk)
        grad_V_total = torch.zeros_like(Vc)
        grad_gate_scale = None
        grad_gate_bias = None

        use_parametric_gate = (
            ctx.write_gate_fixed_value is None
            and not (gate_scale_saved is None and gate_bias_saved is None and ctx.write_gate_init == "identity")
        )
        scale_broadcast = None
        bias_broadcast = None
        if use_parametric_gate:
            scale_broadcast = (
                torch.full((), 0.0, device=device, dtype=compute_dtype).view(1, 1, 1, 1, 1)
                if gate_scale_saved is None else
                _broadcast_stablemax_gate_param(
                    gate_scale_saved,
                    H=H,
                    M=M,
                    device=device,
                    dtype=compute_dtype,
                    name="write_gate_scale",
                )
            )
            bias_broadcast = (
                torch.full((), 6.0 if ctx.write_gate_init == "identity" else 0.0, device=device, dtype=compute_dtype).view(1, 1, 1, 1, 1)
                if gate_bias_saved is None else
                _broadcast_stablemax_gate_param(
                    gate_bias_saved,
                    H=H,
                    M=M,
                    device=device,
                    dtype=compute_dtype,
                    name="write_gate_bias",
                )
            )
            if ctx.has_gate_scale_tensor and ctx.needs_input_grad[11]:
                grad_gate_scale = torch.zeros_like(gate_scale_saved, dtype=compute_dtype)
            if ctx.has_gate_bias_tensor and ctx.needs_input_grad[12]:
                grad_gate_bias = torch.zeros_like(gate_bias_saved, dtype=compute_dtype)

        stable_score_grad = _stablemax_score_transform_grad(score_chunk, power=power)
        causal_mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool)).view(1, 1, 1, C, C)

        score_chunk_all = score_chunk
        s_chunk = stable_score_chunk
        p_chunk = decode_probs
        v_chunk = Vc
        d_start = score_prev_den
        z_start = z_prev_chunk

        total_den = d_start.unsqueeze(3) + s_chunk.cumsum(dim=3)
        total_den_safe = total_den.clamp_min(eps)
        if ctx.write_gate_fixed_value is not None:
            gate_chunk = torch.full_like(s_chunk, float(ctx.write_gate_fixed_value))
            gate_sigmoid = None
            gate_mask = None
        elif use_parametric_gate:
            gate_logits = score_chunk_all * scale_broadcast + bias_broadcast
            gate_sigmoid = torch.sigmoid(gate_logits)
            if valid_chunk is not None:
                gate_mask = valid_chunk.to(dtype=compute_dtype)
                gate_chunk = gate_sigmoid * gate_mask
            else:
                gate_mask = None
                gate_chunk = gate_sigmoid
        else:
            gate_chunk = torch.ones_like(s_chunk)
            gate_sigmoid = None
            gate_mask = None

        alpha_chunk = (s_chunk / total_den_safe) * gate_chunk
        beta_chunk = 1.0 - alpha_chunk
        beta_safe = beta_chunk.clamp_min(eps)
        prefix_prod = torch.cumprod(beta_safe, dim=3)
        recurrent_factor = p_chunk * prefix_prod
        replay_source = alpha_chunk / prefix_prod

        suffix_inclusive = torch.flip(torch.cumprod(torch.flip(beta_safe, dims=(3,)), dim=3), dims=(3,))
        suffix_exclusive = torch.cat(
            [suffix_inclusive[:, :, :, 1:, :], torch.ones_like(suffix_inclusive[:, :, :, :1, :])],
            dim=3,
        )
        chunk_source = alpha_chunk * suffix_exclusive
        chunk_A = prefix_prod[:, :, :, -1, :]

        BHNC = B * H * NC
        recurrent_factor_flat = recurrent_factor.reshape(BHNC, C, M)
        replay_source_flat = replay_source.reshape(BHNC, C, M)
        v_flat = v_chunk.reshape(BHNC, C, D_value)
        dy_flat = dYc.reshape(BHNC, C, D_value)
        z_start_flat = z_start.reshape(BHNC, M, D_value)

        grad_recurrent_factor = torch.einsum("bcd,bmd->bcm", dy_flat, z_start_flat)
        grad_z_start_local = torch.einsum("bcm,bcd->bmd", recurrent_factor_flat, dy_flat).reshape(B, H, NC, M, D_value)

        local_mix = torch.matmul(recurrent_factor_flat, replay_source_flat.transpose(1, 2)).reshape(B, H, NC, C, C)
        local_mix = torch.where(causal_mask, local_mix, torch.zeros_like(local_mix))
        grad_local_mix = torch.matmul(dy_flat, v_flat.transpose(1, 2)).reshape(B, H, NC, C, C)
        grad_local_mix = torch.where(causal_mask, grad_local_mix, torch.zeros_like(grad_local_mix))

        grad_recurrent_factor = grad_recurrent_factor + torch.matmul(
            grad_local_mix.reshape(BHNC, C, C),
            replay_source_flat,
        )
        grad_replay_source = torch.matmul(
            grad_local_mix.transpose(-1, -2).reshape(BHNC, C, C),
            recurrent_factor_flat,
        ).reshape(B, H, NC, C, M)
        grad_V_total = grad_V_total + torch.matmul(
            local_mix.transpose(-1, -2).reshape(BHNC, C, C),
            dy_flat,
        ).reshape(B, H, NC, C, D_value)

        grad_recurrent_factor = grad_recurrent_factor.reshape(B, H, NC, C, M)
        grad_decode_probs = grad_recurrent_factor * prefix_prod
        grad_prefix_prod = grad_recurrent_factor * p_chunk

        rev_A = torch.flip(chunk_A, dims=(2,))
        rev_B = torch.flip(grad_z_start_local, dims=(2,))
        _, rev_grad_z_total = _affine_prefix_scan_chunkwise(rev_A, rev_B)
        grad_z_total = torch.flip(rev_grad_z_total, dims=(2,))
        grad_chunk_B = torch.cat(
            [grad_z_total[:, :, 1:, :, :], torch.zeros_like(grad_z_total[:, :, :1, :, :])],
            dim=2,
        )
        grad_chunk_A = (grad_chunk_B * z_start).sum(dim=-1)

        grad_chunk_source = torch.einsum("bhnmd,bhncd->bhncm", grad_chunk_B, v_chunk)
        grad_V_total = grad_V_total + torch.einsum("bhncm,bhnmd->bhncd", chunk_source, grad_chunk_B)

        grad_alpha_chunk = grad_replay_source / prefix_prod + grad_chunk_source * suffix_exclusive
        grad_prefix_prod = grad_prefix_prod - (grad_replay_source * alpha_chunk) / prefix_prod.square()
        grad_prefix_prod[:, :, :, -1, :] = grad_prefix_prod[:, :, :, -1, :] + grad_chunk_A

        grad_beta_from_prefix = torch.flip(
            torch.cumsum(torch.flip(grad_prefix_prod * prefix_prod, dims=(3,)), dim=3),
            dims=(3,),
        ) / beta_safe
        grad_suffix_exclusive = grad_chunk_source * alpha_chunk
        grad_beta_from_suffix = (torch.cumsum(grad_suffix_exclusive * suffix_exclusive, dim=3) - grad_suffix_exclusive * suffix_exclusive) / beta_safe
        grad_alpha_chunk = grad_alpha_chunk - grad_beta_from_prefix - grad_beta_from_suffix

        inv_total_den = total_den_safe.reciprocal()
        grad_gate_chunk = grad_alpha_chunk * (s_chunk * inv_total_den)
        grad_ratio_chunk = grad_alpha_chunk * gate_chunk
        grad_s_chunk = grad_ratio_chunk * inv_total_den
        grad_d_raw = -(grad_ratio_chunk * s_chunk) * inv_total_den.square()
        grad_s_chunk = grad_s_chunk + torch.flip(
            torch.cumsum(torch.flip(grad_d_raw, dims=(3,)), dim=3),
            dims=(3,),
        )
        grad_prev_den_local = grad_d_raw.sum(dim=3)
        grad_chunk_den = torch.flip(torch.cumsum(torch.flip(grad_prev_den_local, dims=(2,)), dim=2), dims=(2,)) - grad_prev_den_local
        grad_s_chunk = grad_s_chunk + grad_chunk_den.unsqueeze(3)

        grad_score_enc = grad_s_chunk * stable_score_grad
        if use_parametric_gate:
            grad_gate_pre = grad_gate_chunk if gate_mask is None else grad_gate_chunk * gate_mask
            grad_gate_logit = grad_gate_pre * gate_sigmoid * (1.0 - gate_sigmoid)
            grad_score_enc = grad_score_enc + grad_gate_logit * scale_broadcast
            if grad_gate_scale is not None:
                scale_contrib = grad_gate_logit * score_chunk_all
                if gate_scale_saved.ndim == 0:
                    grad_gate_scale = grad_gate_scale + scale_contrib.sum()
                elif gate_scale_saved.ndim == 1:
                    grad_gate_scale = grad_gate_scale + scale_contrib.sum(dim=(0, 2, 3, 4))
                else:
                    grad_gate_scale = grad_gate_scale + scale_contrib.sum(dim=(0, 2, 3))
            if grad_gate_bias is not None:
                if gate_bias_saved.ndim == 0:
                    grad_gate_bias = grad_gate_bias + grad_gate_logit.sum()
                elif gate_bias_saved.ndim == 1:
                    grad_gate_bias = grad_gate_bias + grad_gate_logit.sum(dim=(0, 2, 3, 4))
                else:
                    grad_gate_bias = grad_gate_bias + grad_gate_logit.sum(dim=(0, 2, 3))

        softmax_dot = (grad_decode_probs * decode_probs).sum(dim=-1, keepdim=True)
        grad_score_dec = decode_probs * (grad_decode_probs - softmax_dot)

        grad_Q_dec_c = scale_f * torch.einsum("bhncm,bhmd->bhncd", grad_score_dec, K_dec_f)
        grad_K_dec = (scale_f * torch.einsum("bhncm,bhncd->bhmd", grad_score_dec, Q_dec_c)).sum(dim=0)

        grad_Q = scale_f * torch.einsum("bhncm,bhncd->hmd", grad_score_enc, Kc)
        grad_Kc = scale_f * torch.einsum("bhncm,hmd->bhncd", grad_score_enc, Q_f)
        grad_K = grad_Kc.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_score)[:, :N, :, :]
        grad_V = grad_V_total.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_value)[:, :N, :, :]
        grad_Q_dec = grad_Q_dec_c.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_score)[:, :N, :, :]

        grad_Q = grad_Q.to(ctx.q_dtype) if ctx.needs_input_grad[0] else None
        grad_K = grad_K.to(ctx.k_dtype) if ctx.needs_input_grad[1] else None
        grad_V = grad_V.to(ctx.v_dtype) if ctx.needs_input_grad[2] else None
        grad_Q_dec = grad_Q_dec.to(ctx.q_dec_dtype) if ctx.needs_input_grad[3] else None
        grad_K_dec = grad_K_dec.to(ctx.k_dec_dtype) if ctx.needs_input_grad[4] else None
        if grad_gate_scale is not None:
            grad_gate_scale = grad_gate_scale.to(dtype=gate_scale_saved.dtype)
        if grad_gate_bias is not None:
            grad_gate_bias = grad_gate_bias.to(dtype=gate_bias_saved.dtype)

        return (
            grad_Q,
            grad_K,
            grad_V,
            grad_Q_dec,
            grad_K_dec,
            None,
            None,
            None,
            None,
            None,
            None,
            grad_gate_scale,
            grad_gate_bias,
        )

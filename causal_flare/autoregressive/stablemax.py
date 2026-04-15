from causal_flare._common import *
from causal_flare._reference_utils import (
    resolve_flare_causal_decode_inputs as _resolve_flare_causal_decode_inputs,
    validate_flare_qkv_layouts as _validate_flare_qkv_layouts,
)
import warnings

try:
    from torch._higher_order_ops.scan import scan as _torch_scan
except Exception:
    _torch_scan = None


def _resolve_stablemax_chunk_size_pytorch(N: int, M: int, D_score: int, chunk_size) -> int:
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


def _resolve_stablemax_chunk_size_triton(N: int, M: int, D_score: int, D_value: int, chunk_size) -> int:
    if chunk_size is None:
        env_chunk = os.environ.get("FLARE_PYTORCH_CHUNK_SIZE", "")
        chunk_size = int(env_chunk) if env_chunk else None
    if chunk_size is not None:
        return int(chunk_size)
    # The Triton backward output kernel keeps the full value-head dimension live
    # inside a chunk-owned program. Large value heads can therefore overflow
    # shared memory even when the forward-oriented `D_score` / `M` heuristic
    # would have picked `128`.
    if D_value > 128 and (M >= 256 or D_score >= 128):
        return 64
    if D_score <= 32 and M <= 64:
        return 64
    del N
    return 128


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


def _stablemax_local_recurrence_scan(
    z_init: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    value_steps: torch.Tensor,
    decode_weights: torch.Tensor,
    *,
    value_dtype: torch.dtype,
    value_num_groups: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _torch_scan is None:
        raise RuntimeError("torch._higher_order_ops.scan is unavailable")

    if value_num_groups > 1:
        group_dim = value_steps.shape[-1]

        def _group_step(z_prev, xs):
            alpha_step, beta_step, value_step, decode_step = xs
            z_next = (beta_step.unsqueeze(-1) * z_prev + alpha_step.unsqueeze(-1) * value_step).contiguous()
            y_step = (
                decode_step.to(value_dtype).unsqueeze(-1).unsqueeze(-1) * z_next
            ).sum(dim=2).flatten(start_dim=-2).contiguous()
            return z_next, y_step

        scan_inputs = (
            alpha.permute(2, 0, 1, 3, 4).contiguous(),
            beta.permute(2, 0, 1, 3, 4).contiguous(),
            value_steps.permute(2, 0, 1, 3, 4, 5).contiguous(),
            decode_weights.permute(2, 0, 1, 3).contiguous(),
        )
    else:
        def _scalar_step(z_prev, xs):
            alpha_step, beta_step, value_step, decode_step = xs
            z_next = (beta_step.unsqueeze(-1) * z_prev + alpha_step.unsqueeze(-1) * value_step).contiguous()
            y_step = (decode_step.to(value_dtype).unsqueeze(-1) * z_next).sum(dim=2).contiguous()
            return z_next, y_step

        scan_inputs = (
            alpha.permute(2, 0, 1, 3).contiguous(),
            beta.permute(2, 0, 1, 3).contiguous(),
            value_steps.permute(2, 0, 1, 3, 4).contiguous(),
            decode_weights.permute(2, 0, 1, 3).contiguous(),
        )

    z_final, y_scan = _torch_scan(_group_step if value_num_groups > 1 else _scalar_step, z_init, scan_inputs)
    chunk_out = y_scan.permute(1, 2, 0, 3).contiguous()
    return z_final.contiguous(), chunk_out


class _FLARELocalStateMixAutograd(autograd.Function):
    @staticmethod
    def forward(ctx, z_init, alpha, beta, value_steps, decode_weights):
        grouped = z_init.ndim == 5
        value_dtype = z_init.dtype
        B, H, C = alpha.shape[0], alpha.shape[1], alpha.shape[2]
        z_state = z_init
        outputs = []
        if grouped:
            G, Dg = z_init.shape[3], z_init.shape[4]
            for offset in range(C):
                alpha_step = alpha[:, :, offset]
                beta_step = beta[:, :, offset]
                value_step = value_steps[:, :, offset]
                z_state = beta_step.unsqueeze(-1) * z_state + alpha_step.unsqueeze(-1) * value_step
                y_step = (
                    decode_weights[:, :, offset].to(value_dtype).unsqueeze(-1).unsqueeze(-1) * z_state
                ).sum(dim=2).reshape(B, H, G * Dg)
                outputs.append(y_step)
        else:
            for offset in range(C):
                alpha_step = alpha[:, :, offset]
                beta_step = beta[:, :, offset]
                value_step = value_steps[:, :, offset]
                z_state = beta_step.unsqueeze(-1) * z_state + alpha_step.unsqueeze(-1) * value_step
                y_step = (decode_weights[:, :, offset].to(value_dtype).unsqueeze(-1) * z_state).sum(dim=2)
                outputs.append(y_step)
        chunk_out = torch.stack(outputs, dim=2)
        ctx.grouped = grouped
        ctx.save_for_backward(z_init, alpha, beta, value_steps, decode_weights)
        return z_state, chunk_out

    @staticmethod
    def backward(ctx, grad_z_final, grad_chunk_out):
        z_init, alpha, beta, value_steps, decode_weights = ctx.saved_tensors
        grouped = ctx.grouped
        value_dtype = z_init.dtype
        z_hist = [z_init]
        z_state = z_init
        C = alpha.shape[2]
        for offset in range(C):
            alpha_step = alpha[:, :, offset]
            beta_step = beta[:, :, offset]
            value_step = value_steps[:, :, offset]
            z_state = beta_step.unsqueeze(-1) * z_state + alpha_step.unsqueeze(-1) * value_step
            z_hist.append(z_state)

        grad_alpha = torch.empty_like(alpha)
        grad_beta = torch.empty_like(beta)
        grad_value_steps = torch.empty_like(value_steps)
        grad_decode = torch.empty_like(decode_weights)

        grad_z_next = grad_z_final
        if grouped:
            G, Dg = z_init.shape[3], z_init.shape[4]
            for offset in range(C - 1, -1, -1):
                z_prev = z_hist[offset]
                z_curr = z_hist[offset + 1]
                decode_step = decode_weights[:, :, offset].to(value_dtype)
                dy_step = grad_chunk_out[:, :, offset].reshape(grad_chunk_out.shape[0], grad_chunk_out.shape[1], G, Dg)
                grad_decode[:, :, offset] = (dy_step.unsqueeze(2) * z_curr).sum(dim=(-1, -2)).to(grad_decode.dtype)
                grad_z_total = grad_z_next + decode_step.unsqueeze(-1).unsqueeze(-1) * dy_step.unsqueeze(2)
                grad_beta[:, :, offset] = (grad_z_total * z_prev).sum(dim=-1).to(grad_beta.dtype)
                grad_alpha[:, :, offset] = (grad_z_total * value_steps[:, :, offset]).sum(dim=-1).to(grad_alpha.dtype)
                grad_value_steps[:, :, offset] = alpha[:, :, offset].unsqueeze(-1) * grad_z_total
                grad_z_next = beta[:, :, offset].unsqueeze(-1) * grad_z_total
        else:
            for offset in range(C - 1, -1, -1):
                z_prev = z_hist[offset]
                z_curr = z_hist[offset + 1]
                decode_step = decode_weights[:, :, offset].to(value_dtype)
                dy_step = grad_chunk_out[:, :, offset]
                grad_decode[:, :, offset] = (dy_step.unsqueeze(2) * z_curr).sum(dim=-1).to(grad_decode.dtype)
                grad_z_total = grad_z_next + decode_step.unsqueeze(-1) * dy_step.unsqueeze(2)
                grad_beta[:, :, offset] = (grad_z_total * z_prev).sum(dim=-1).to(grad_beta.dtype)
                grad_alpha[:, :, offset] = (grad_z_total * value_steps[:, :, offset]).sum(dim=-1).to(grad_alpha.dtype)
                grad_value_steps[:, :, offset] = alpha[:, :, offset].unsqueeze(-1) * grad_z_total
                grad_z_next = beta[:, :, offset].unsqueeze(-1) * grad_z_total

        return grad_z_next, grad_alpha, grad_beta, grad_value_steps, grad_decode


def _stablemax_local_recurrence_autograd(
    z_init: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    value_steps: torch.Tensor,
    decode_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _FLARELocalStateMixAutograd.apply(z_init, alpha, beta, value_steps, decode_weights)


def _pack_stablemax_gate_tensor(
    gate_tensor,
    *,
    B: int,
    N: int,
    H: int,
    M: int,
    NC: int,
    C: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(gate_tensor):
        raise ValueError(f"{name} must be a torch.Tensor. Got {type(gate_tensor).__name__}")
    gate_t = gate_tensor.to(device=device, dtype=dtype)
    if gate_t.ndim == 4 and gate_t.shape == (B, N, H, M):
        padded_len = NC * C
        if padded_len != N:
            gate_t = torch.cat(
                [gate_t, torch.zeros((B, padded_len - N, H, M), device=device, dtype=dtype)],
                dim=1,
            )
        return gate_t.permute(0, 2, 1, 3).reshape(B, H, NC, C, M).contiguous()
    if gate_t.ndim == 5 and gate_t.shape == (B, H, NC, C, M):
        return gate_t.contiguous()
    raise ValueError(f"{name} must be [B, N, H, M] or [B, H, NC, C, M]. Got {tuple(gate_t.shape)}")


def _pack_stablemax_group_gate_tensor(
    gate_tensor,
    *,
    B: int,
    N: int,
    H: int,
    M: int,
    G: int,
    NC: int,
    C: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(gate_tensor):
        raise ValueError(f"{name} must be a torch.Tensor. Got {type(gate_tensor).__name__}")
    gate_t = gate_tensor.to(device=device, dtype=dtype)
    if gate_t.ndim == 5 and gate_t.shape == (B, N, H, M, G):
        padded_len = NC * C
        if padded_len != N:
            gate_t = torch.cat(
                [gate_t, torch.zeros((B, padded_len - N, H, M, G), device=device, dtype=dtype)],
                dim=1,
            )
        return gate_t.permute(0, 2, 1, 3, 4).reshape(B, H, NC, C, M, G).contiguous()
    if gate_t.ndim == 6 and gate_t.shape == (B, H, NC, C, M, G):
        return gate_t.contiguous()
    raise ValueError(f"{name} must be [B, N, H, M, G] or [B, H, NC, C, M, G]. Got {tuple(gate_t.shape)}")


def _pack_stablemax_decode_tensor(
    decode_tensor,
    *,
    B: int,
    N: int,
    H: int,
    M: int,
    NC: int,
    C: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(decode_tensor):
        raise ValueError(f"{name} must be a torch.Tensor. Got {type(decode_tensor).__name__}")
    decode_t = decode_tensor.to(device=device, dtype=dtype)
    if decode_t.ndim != 4 or decode_t.shape != (B, N, H, M):
        raise ValueError(f"{name} must be [B, N, H, M]. Got {tuple(decode_t.shape)}")
    padded_len = NC * C
    if padded_len != N:
        decode_t = torch.cat(
            [decode_t, torch.zeros((B, padded_len - N, H, M), device=device, dtype=dtype)],
            dim=1,
        )
    return decode_t.permute(0, 2, 1, 3).reshape(B, H, NC, C, M).contiguous()


def _unpack_stablemax_decode_tensor(decode_tensor: torch.Tensor, *, N: int) -> torch.Tensor:
    B, H, NC, C, M = decode_tensor.shape
    return decode_tensor.permute(0, 2, 3, 1, 4).reshape(B, NC * C, H, M)[:, :N, :, :].contiguous()


def _wandb_rank0_active() -> bool:
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def _masked_mean(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return x.mean()
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (x * mask_f).sum() / denom


def _log_stablemax_latent_diagnostics(
    *,
    stable_score_chunk: torch.Tensor,
    decode_probs: torch.Tensor,
    alpha_chunk: torch.Tensor,
    valid_chunk: torch.Tensor | None,
    layer_idx: int | None,
) -> None:
    if not _wandb_rank0_active():
        return
    try:
        import wandb

        if wandb.run is None:
            return

        dtype = stable_score_chunk.dtype
        device = stable_score_chunk.device
        _, H, _, _, M = stable_score_chunk.shape
        eps = torch.finfo(dtype).eps
        log_m = torch.log(torch.tensor(float(M), device=device, dtype=dtype)).clamp_min(eps)

        token_mask = None
        if valid_chunk is not None:
            token_mask = valid_chunk.expand(-1, H, -1, -1, 1).squeeze(-1)

        write_den = stable_score_chunk.sum(dim=-1, keepdim=True).clamp_min(eps)
        write_probs = stable_score_chunk / write_den
        write_entropy = -(write_probs * write_probs.clamp_min(eps).log()).sum(dim=-1) / log_m
        write_top1 = write_probs.amax(dim=-1)
        write_eff_slots = torch.exp(write_entropy * log_m)

        decode_entropy = -(decode_probs * decode_probs.clamp_min(eps).log()).sum(dim=-1) / log_m
        decode_top1 = decode_probs.amax(dim=-1)

        alpha_sum = alpha_chunk.sum(dim=-1)
        alpha_top1 = alpha_chunk.amax(dim=-1)

        slot_mass = stable_score_chunk.sum(dim=(0, 2, 3))
        slot_share = slot_mass / slot_mass.sum(dim=-1, keepdim=True).clamp_min(eps)
        slot_entropy = -(slot_share * slot_share.clamp_min(eps).log()).sum(dim=-1)
        slot_eff_slots = torch.exp(slot_entropy)
        slot_max_share = slot_share.amax(dim=-1)
        slot_underused_frac = (slot_share < (0.1 / float(M))).to(dtype=dtype).mean(dim=-1)

        layer_suffix = f"layer_{layer_idx}" if layer_idx is not None else "layer_unknown"
        wandb.log(
            {
                f"train/flare_write_entropy_norm/{layer_suffix}": _masked_mean(write_entropy, token_mask).item(),
                f"train/flare_write_top1_mean/{layer_suffix}": _masked_mean(write_top1, token_mask).item(),
                f"train/flare_write_effective_slots_mean/{layer_suffix}": _masked_mean(write_eff_slots, token_mask).item(),
                f"train/flare_decode_entropy_norm/{layer_suffix}": _masked_mean(decode_entropy, token_mask).item(),
                f"train/flare_decode_top1_mean/{layer_suffix}": _masked_mean(decode_top1, token_mask).item(),
                f"train/flare_alpha_sum_mean/{layer_suffix}": _masked_mean(alpha_sum, token_mask).item(),
                f"train/flare_alpha_top1_mean/{layer_suffix}": _masked_mean(alpha_top1, token_mask).item(),
                f"train/flare_slot_effective_slots_mean/{layer_suffix}": slot_eff_slots.mean().item(),
                f"train/flare_slot_effective_slots_frac/{layer_suffix}": (slot_eff_slots / float(M)).mean().item(),
                f"train/flare_slot_max_share_mean/{layer_suffix}": slot_max_share.mean().item(),
                f"train/flare_slot_underused_frac/{layer_suffix}": slot_underused_frac.mean().item(),
            },
            commit=False,
        )
    except Exception:
        pass


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


def _rms_normalize_last_dim(x: torch.Tensor, *, eps: float, scale_by_sqrt_dim: bool = False) -> torch.Tensor:
    rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    y = x / rms
    if scale_by_sqrt_dim:
        y = y / math.sqrt(x.shape[-1])
    return y


def _rms_normalize_last_dim_backward(
    grad_y: torch.Tensor,
    x: torch.Tensor,
    *,
    eps: float,
    scale_by_sqrt_dim: bool = False,
) -> torch.Tensor:
    dim = x.shape[-1]
    rms = torch.sqrt(x.square().mean(dim=-1, keepdim=True) + eps)
    grad_x = grad_y / rms
    proj = (grad_y * x).sum(dim=-1, keepdim=True) / (dim * rms.pow(3))
    grad_x = grad_x - x * proj
    if scale_by_sqrt_dim:
        grad_x = grad_x / math.sqrt(dim)
    return grad_x


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

        C = _resolve_stablemax_chunk_size_pytorch(N, M, D_score, chunk_size)
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
        compute_dtype = Q_f.dtype

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
        dY_f = dY.to(compute_dtype)
        if PADDED_LEN != N:
            dY_f = torch.cat(
                [dY_f, torch.zeros((B, PADDED_LEN - N, H, D_value), device=device, dtype=compute_dtype)],
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
    Q_dec,
    K_dec,
    scale=None,
    eps=None,
    profile: bool = False,
    chunk_size=None,
    power: float = 2.0,
    state: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    return_state: bool = False,
):
    """Chunked stablemax FLARE forward/backward with explicit decode projections.

    Expected tensor shapes:
    - Q: ``[H, M, D_score]``
        Static encoder latent query bank per head.
    - K: ``[B, N, H, D_score]``
        Per-token encoder keys.
    - V: ``[B, N, H, D_value]``
        Per-token values.
    - Q_dec: ``[B, N, H, D_score]``
        Per-token decode queries.
    - K_dec: ``[H, M, D_score]``
        Static decode latent keys per head.

    Notes:
    - Returns output with shape ``[B, N, H, D_value]``.
    - ``scale`` defaults to ``1 / sqrt(D_score)`` when omitted.
    - This API is full-sequence only: ``profile``, ``state``, ``attention_mask``,
      and ``return_state=True`` are currently unsupported.
    """
    if profile:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch does not support profile=True")
    if state is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch does not support recurrent state")
    if attention_mask is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch does not support attention_mask")
    if return_state:
        raise NotImplementedError("flare_autoregressive_stablemax_pytorch does not support return_state=True")
    return FLAREAutoregressiveStablemaxPyTorch.apply(Q, K, V, Q_dec, K_dec, scale, chunk_size, power)

class FLAREAutoregressiveStablemaxMatDecodePyTorch(autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, C_dec, scale=None, chunk_size=None, power: float = 2.0):
        compute_dtype = torch.float32
        if os.environ.get("FLARE_PYTORCH_MATCH_REFERENCE", "") == "1":
            compute_dtype = Q.dtype

        B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="StableMax Chunked FLARE Mat Decode")
        scale = _resolve_attn_scale(scale, D_score)
        device = Q.device
        out_dtype = V.dtype
        scale_f = float(scale)
        power = float(power)

        Q_f = Q.to(compute_dtype)
        K_f = K.to(compute_dtype)
        V_f = V.to(compute_dtype)

        C = _resolve_stablemax_chunk_size_pytorch(N, M, D_score, chunk_size)
        NC = math.ceil(N / C) if N > 0 else 0
        PADDED_LEN = NC * C
        PAD = PADDED_LEN - N
        if PAD > 0:
            K_f = torch.cat([K_f, torch.zeros((B, PAD, H, D_score), device=device, dtype=compute_dtype)], dim=1)
            V_f = torch.cat([V_f, torch.zeros((B, PAD, H, D_value), device=device, dtype=compute_dtype)], dim=1)

        Kc = K_f.reshape(B, NC, C, H, D_score).permute(0, 3, 1, 2, 4).contiguous()
        Vc = V_f.reshape(B, NC, C, H, D_value).permute(0, 3, 1, 2, 4).contiguous()
        C_dec_c = _pack_stablemax_decode_tensor(C_dec, B=B, N=N, H=H, M=M, NC=NC, C=C, device=device, dtype=compute_dtype, name="C_dec")
        BHNC = B * H * NC

        ctx.empty = False
        ctx.scale = scale_f
        ctx.power = power
        ctx.chunk_size = C
        ctx.compute_dtype = compute_dtype
        ctx.q_dtype = Q.dtype
        ctx.k_dtype = K.dtype
        ctx.v_dtype = V.dtype
        ctx.c_dec_dtype = C_dec.dtype
        ctx.n = N
        ctx.h = H
        ctx.d_score = D_score
        ctx.d_value = D_value
        ctx.padded_len = PADDED_LEN
        ctx.c = C
        ctx.nc = NC
        ctx.bhnc = BHNC

        score_chunk = scale_f * torch.einsum("bhncd,hmd->bhncm", Kc, Q_f)
        stable_score_chunk = _stablemax_score_transform(score_chunk, power=power)
        score_chunk_den = stable_score_chunk.sum(dim=3)
        score_chunk_num = torch.bmm(stable_score_chunk.reshape(BHNC, C, M).mT, Vc.reshape(BHNC, C, D_value)).reshape(B, H, NC, M, D_value)

        score_prev_den = torch.cumsum(score_chunk_den, dim=2) - score_chunk_den
        score_prev_num = torch.cumsum(score_chunk_num, dim=2) - score_chunk_num

        chunk_den = stable_score_chunk.cumsum(dim=3)
        total_den = score_prev_den.unsqueeze(3) + chunk_den
        total_den_safe = torch.where(total_den > 0, total_den, torch.ones_like(total_den))
        inv_total_den = total_den_safe.reciprocal()
        decode_over_den = C_dec_c * inv_total_den

        prefix_out = torch.einsum("bhncm,bhnmd->bhncd", decode_over_den, score_prev_num)
        causal_mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool)).view(1, 1, 1, C, C)
        local_mix = torch.matmul(decode_over_den, stable_score_chunk.transpose(-1, -2))
        local_mix = torch.where(causal_mask, local_mix, torch.zeros_like(local_mix))
        local_out = torch.matmul(local_mix.reshape(BHNC, C, C), Vc.reshape(BHNC, C, D_value)).reshape(B, H, NC, C, D_value)
        Yc = prefix_out + local_out

        Y_out = Yc.reshape(B, H, PADDED_LEN, D_value)[:, :, :N, :].permute(0, 2, 1, 3).to(out_dtype)
        _check_finite("FLAREAutoregressiveStablemaxMatDecodePyTorch.Y", Y_out)

        ctx.save_for_backward(
            Q_f,
            Kc,
            Vc,
            C_dec_c,
            score_chunk,
            stable_score_chunk,
            score_prev_num,
            inv_total_den,
        )
        return Y_out

    @staticmethod
    def backward(ctx, dY):
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
            empty_c_dec = torch.zeros((B, N, H, M), device=dY.device, dtype=ctx.c_dec_dtype)
            return empty_q, empty_kv, empty_v, empty_c_dec, None, None, None

        (Q_f, Kc, Vc, C_dec_c, score_chunk, stable_score_chunk, score_prev_num, inv_total_den) = ctx.saved_tensors
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
        compute_dtype = Q_f.dtype

        stable_score_grad = _stablemax_score_transform_grad(score_chunk, power=power)

        score_chunk_den = stable_score_chunk.sum(dim=3)
        score_chunk_num = torch.bmm(
            stable_score_chunk.reshape(BHNC, C, M).transpose(1, 2),
            Vc.reshape(BHNC, C, D_value),
        ).reshape(B, H, NC, M, D_value)
        score_prev_den = torch.cumsum(score_chunk_den, dim=2) - score_chunk_den
        decode_over_den = C_dec_c * inv_total_den

        causal_mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool)).view(1, 1, 1, C, C)
        local_mix = torch.matmul(decode_over_den, stable_score_chunk.transpose(-1, -2))
        local_mix = torch.where(causal_mask, local_mix, torch.zeros_like(local_mix))

        dY_f = dY.to(compute_dtype)
        if PADDED_LEN != N:
            dY_f = torch.cat(
                [dY_f, torch.zeros((B, PADDED_LEN - N, H, D_value), device=device, dtype=compute_dtype)],
                dim=1,
            )
        dYc = dY_f.permute(0, 2, 1, 3).reshape(B, H, NC, C, D_value).contiguous()

        grad_decode_over_den = torch.einsum("bhncd,bhnmd->bhncm", dYc, score_prev_num)
        grad_prev_num = torch.einsum("bhncm,bhncd->bhnmd", decode_over_den, dYc)
        grad_local_mix = torch.matmul(dYc, Vc.transpose(-1, -2))
        grad_local_mix = torch.where(causal_mask, grad_local_mix, torch.zeros_like(grad_local_mix))
        grad_V_phase3 = torch.matmul(
            local_mix.transpose(-1, -2).reshape(BHNC, C, C),
            dYc.reshape(BHNC, C, D_value),
        ).reshape(B, H, NC, C, D_value)

        grad_decode_over_den = grad_decode_over_den + torch.matmul(grad_local_mix, stable_score_chunk)
        grad_w_phase3 = torch.matmul(grad_local_mix.transpose(-1, -2), decode_over_den)

        grad_C_dec = grad_decode_over_den * inv_total_den
        grad_total_den = -(grad_decode_over_den * C_dec_c) * inv_total_den.square()

        grad_prev_den = grad_total_den.sum(dim=3)
        grad_w_phase3 = grad_w_phase3 + torch.flip(torch.cumsum(torch.flip(grad_total_den, dims=(3,)), dim=3), dims=(3,))

        grad_score_chunk_den = torch.flip(torch.cumsum(torch.flip(grad_prev_den, dims=(2,)), dim=2), dims=(2,)) - grad_prev_den
        grad_score_chunk_num = torch.flip(torch.cumsum(torch.flip(grad_prev_num, dims=(2,)), dim=2), dims=(2,)) - grad_prev_num

        grad_w_total = grad_w_phase3 + grad_score_chunk_den.unsqueeze(3)
        grad_w_total = grad_w_total + torch.einsum("bhnmd,bhncd->bhncm", grad_score_chunk_num, Vc)
        grad_V_total = grad_V_phase3 + torch.einsum("bhncm,bhnmd->bhncd", stable_score_chunk, grad_score_chunk_num)

        grad_score_enc = grad_w_total * stable_score_grad

        grad_Q = scale_f * torch.einsum("bhncm,bhncd->hmd", grad_score_enc, Kc)
        grad_Kc = scale_f * torch.einsum("bhncm,hmd->bhncd", grad_score_enc, Q_f)
        grad_K = grad_Kc.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_score)[:, :N, :, :]
        grad_V = grad_V_total.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_value)[:, :N, :, :]
        grad_C_dec = grad_C_dec.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, M)[:, :N, :, :]
        return (
            grad_Q.to(ctx.q_dtype),
            grad_K.to(ctx.k_dtype),
            grad_V.to(ctx.v_dtype),
            grad_C_dec.to(ctx.c_dec_dtype),
            None,
            None,
            None,
        )

def flare_autoregressive_stablemax_mat_decode_pytorch(
    Q,
    K,
    V,
    C_dec,
    scale=None,
    eps=None,
    profile: bool = False,
    chunk_size=None,
    power: float = 2.0,
    state: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    return_state: bool = False,
):
    """Chunked stablemax FLARE with externally supplied decode weights.

    Expected tensor shapes:
    - Q: ``[H, M, D_score]``
        Static encoder latent query bank per head.
    - K: ``[B, N, H, D_score]``
        Per-token encoder keys.
    - V: ``[B, N, H, D_value]``
        Per-token values.
    - C_dec: ``[B, N, H, M]``
        Precomputed decode weights over the latent axis. Typically this is
        ``softmax(scale * einsum('bnhd,hmd->bnhm', Q_dec, K_dec), dim=-1)``.

    Notes:
    - Returns output with shape ``[B, N, H, D_value]``.
    - ``scale`` defaults to ``1 / sqrt(D_score)`` when omitted.
    - This API is full-sequence only: ``profile``, ``state``, ``attention_mask``,
      and ``return_state=True`` are currently unsupported.
    """
    del eps
    if profile:
        raise NotImplementedError("flare_autoregressive_stablemax_mat_decode_pytorch does not support profile=True")
    if state is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_mat_decode_pytorch does not support recurrent state")
    if attention_mask is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_mat_decode_pytorch does not support attention_mask")
    if return_state:
        raise NotImplementedError("flare_autoregressive_stablemax_mat_decode_pytorch does not support return_state=True")
    return FLAREAutoregressiveStablemaxMatDecodePyTorch.apply(Q, K, V, C_dec, scale, chunk_size, power)



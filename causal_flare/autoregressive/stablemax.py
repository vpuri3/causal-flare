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


def _stablemax_write_gate(
    score_chunk: torch.Tensor,
    *,
    B: int,
    N: int,
    H: int,
    M: int,
    NC: int,
    C: int,
    valid_chunk: torch.Tensor | None,
    write_gate_fixed_value: float | None,
    write_gate_tensor,
) -> torch.Tensor:
    if write_gate_tensor is not None:
        gate = _pack_stablemax_gate_tensor(
            write_gate_tensor,
            B=B,
            N=N,
            H=H,
            M=M,
            NC=NC,
            C=C,
            device=score_chunk.device,
            dtype=score_chunk.dtype,
            name="write_gate_tensor",
        )
        if torch.any(gate < 0.0) or torch.any(gate > 1.0):
            raise ValueError("write_gate_tensor entries must lie in [0, 1]")
    elif write_gate_fixed_value is not None:
        gate_value = float(write_gate_fixed_value)
        if gate_value < 0.0 or gate_value > 1.0:
            raise ValueError(f"write_gate_fixed_value must lie in [0, 1]. Got {gate_value}")
        gate = torch.full_like(score_chunk, gate_value)
    else:
        gate = torch.ones_like(score_chunk)
    if valid_chunk is not None:
        gate = gate * valid_chunk.to(dtype=gate.dtype)
    return gate


def _apply_stablemax_write_topk(
    score_chunk: torch.Tensor,
    stable_score_chunk: torch.Tensor,
    *,
    write_topk: int,
    valid_chunk: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    _, _, _, _, m = stable_score_chunk.shape
    if write_topk <= 0 or write_topk >= m:
        return stable_score_chunk, None

    score_for_topk = score_chunk
    if valid_chunk is not None:
        neg_inf = torch.full((), float("-inf"), device=score_chunk.device, dtype=score_chunk.dtype)
        score_for_topk = torch.where(valid_chunk, score_chunk, neg_inf)

    topk_indices = torch.topk(score_for_topk, k=min(int(write_topk), m), dim=-1).indices
    topk_mask = torch.zeros_like(stable_score_chunk, dtype=torch.bool)
    topk_mask.scatter_(-1, topk_indices, True)
    if valid_chunk is not None:
        topk_mask = topk_mask & valid_chunk
    return torch.where(topk_mask, stable_score_chunk, torch.zeros_like(stable_score_chunk)), topk_mask


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

        if N == 0:
            ctx.empty = True
            ctx.b = B
            ctx.q_dtype = Q.dtype
            ctx.k_dtype = K.dtype
            ctx.v_dtype = V.dtype
            ctx.c_dec_dtype = C_dec.dtype
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
        C_dec_c = _pack_stablemax_decode_tensor(
            C_dec,
            B=B,
            N=N,
            H=H,
            M=M,
            NC=NC,
            C=C,
            device=device,
            dtype=compute_dtype,
            name="C_dec",
        )
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
        score_chunk_num = torch.bmm(
            stable_score_chunk.reshape(BHNC, C, M).transpose(1, 2),
            Vc.reshape(BHNC, C, D_value),
        ).reshape(B, H, NC, M, D_value)

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

        (
            Q_f,
            Kc,
            Vc,
            C_dec_c,
            score_chunk,
            stable_score_chunk,
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
            empty_c_dec = torch.zeros((B, N, H, M), device=device, dtype=ctx.c_dec_dtype)
            return empty_q, empty_kv, empty_v, empty_c_dec, None, None, None

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
    write_gate_fixed_value: float | None = None,
    write_gate_tensor=None,
    retain_tensor=None,
    write_topk: int = 0,
    write_score_bias_tensor=None,
    write_score_scale_tensor=None,
    decode_score_bias_tensor=None,
    decode_score_scale_tensor=None,
    decode_mode: str = "softmax",
    state_mix_mode: str = "normalized",
    log_diagnostics: bool = False,
    diagnostics_layer_idx: int | None = None,
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
        if (
            write_gate_fixed_value is not None or
            write_gate_tensor is not None or
            retain_tensor is not None or
            write_topk > 0 or
            write_score_bias_tensor is not None or
            write_score_scale_tensor is not None or
            decode_score_bias_tensor is not None or
            decode_score_scale_tensor is not None or
            decode_mode != "softmax" or
            state_mix_mode != "normalized"
        ):
            raise ValueError("write_gate and dynamic latent score parameters require write_gate=True")
        return FLAREAutoregressiveStablemaxPyTorch.apply(Q, K, V, Q_dec_resolved, K_dec_resolved, scale, chunk_size, power)
    if write_gate_tensor is not None and write_gate_fixed_value is not None:
        raise ValueError("write_gate_tensor is mutually exclusive with write_gate_fixed_value")
    if state_mix_mode not in {"normalized", "decoupled_rw", "decoupled_rw_signed"}:
        raise ValueError(f"Unsupported state_mix_mode: {state_mix_mode!r}")
    if state_mix_mode != "normalized" and retain_tensor is None:
        raise ValueError("state_mix_mode != 'normalized' requires retain_tensor")
    return FLAREAutoregressiveStablemaxWriteGatedPyTorch.apply(
        Q,
        K,
        V,
        Q_dec_resolved,
        K_dec_resolved,
        scale,
        chunk_size,
        power,
        write_gate_fixed_value,
        write_gate_tensor,
        retain_tensor,
        write_topk,
        write_score_bias_tensor,
        write_score_scale_tensor,
        decode_score_bias_tensor,
        decode_score_scale_tensor,
        decode_mode,
        state_mix_mode,
        log_diagnostics,
        diagnostics_layer_idx,
    )


def flare_autoregressive_stablemax_experimental_pytorch(
    Q,
    K,
    V,
    *,
    hidden_states: torch.Tensor,
    q_dynamic_mode: str = "full",
    q_dynamic_proj_weight: torch.Tensor | None = None,
    q_dynamic_proj_bias: torch.Tensor | None = None,
    q_outer_latent_proj_weight: torch.Tensor | None = None,
    q_outer_latent_proj_bias: torch.Tensor | None = None,
    q_outer_head_proj_weight: torch.Tensor | None = None,
    q_outer_head_proj_bias: torch.Tensor | None = None,
    q_low_rank_proj_weight: torch.Tensor | None = None,
    q_low_rank_proj_bias: torch.Tensor | None = None,
    q_low_rank_basis: torch.Tensor | None = None,
    q_grouped_outer_latent_proj_weight: torch.Tensor | None = None,
    q_grouped_outer_latent_proj_bias: torch.Tensor | None = None,
    q_grouped_outer_head_proj_weight: torch.Tensor | None = None,
    q_grouped_outer_head_proj_bias: torch.Tensor | None = None,
    q_grouped_outer_num_groups: int | None = None,
    v_slot_proj_weight: torch.Tensor | None = None,
    v_slot_proj_bias: torch.Tensor | None = None,
    scale=None,
    chunk_size=None,
    Q_dec=None,
    K_dec=None,
    power: float = 2.0,
    write_gate_tensor=None,
    retain_gate_tensor=None,
    write_group_gate_tensor=None,
    retain_group_gate_tensor=None,
    write_score_bias_tensor=None,
    write_score_scale_tensor=None,
    decode_score_bias_tensor=None,
    decode_score_scale_tensor=None,
    write_mode: str = "normalized",
    read_mode: str = "softmax",
    state_mix_mode: str = "normalized",
    value_num_groups: int = 1,
):
    if hidden_states.ndim != 3:
        raise ValueError(f"hidden_states must be [B, N, hidden]. Got {tuple(hidden_states.shape)}")
    if write_gate_tensor is not None and (write_gate_tensor.ndim != 4 or write_gate_tensor.shape[:3] != K.shape[:3]):
        raise ValueError(
            f"write_gate_tensor must be [B, N, H, M] with [B, N, H] matching K. Got {tuple(write_gate_tensor.shape)}"
        )

    B, N, H, M, D_score, D_value = _validate_flare_qkv_layouts(Q, K, V, name="StableMax Experimental FLARE")
    if value_num_groups <= 0:
        raise ValueError(f"value_num_groups must be > 0. Got {value_num_groups}.")
    if D_value % value_num_groups != 0:
        raise ValueError(
            f"value_num_groups must divide D_value. Got D_value={D_value}, value_num_groups={value_num_groups}.",
        )
    Q_dec_resolved, K_dec_resolved, _, _, _ = _resolve_flare_causal_decode_inputs(Q, K, Q_dec, K_dec)
    if K_dec_resolved.dim() != 3:
        raise ValueError(f"Experimental stablemax requires static K_dec [H, M, D]. Got {tuple(K_dec_resolved.shape)}")
    if write_group_gate_tensor is not None:
        expected_shape = (B, N, H, M, value_num_groups)
        if write_group_gate_tensor.shape != expected_shape:
            raise ValueError(
                "write_group_gate_tensor must be [B, N, H, M, G]. "
                f"Got {tuple(write_group_gate_tensor.shape)}, expected {expected_shape}.",
            )
    if retain_gate_tensor is not None and (retain_gate_tensor.ndim != 4 or retain_gate_tensor.shape[:3] != K.shape[:3]):
        raise ValueError(
            f"retain_gate_tensor must be [B, N, H, M] with [B, N, H] matching K. Got {tuple(retain_gate_tensor.shape)}"
        )
    if write_score_bias_tensor is not None and (write_score_bias_tensor.ndim != 4 or write_score_bias_tensor.shape[:3] != K.shape[:3]):
        raise ValueError(
            f"write_score_bias_tensor must be [B, N, H, M] with [B, N, H] matching K. Got {tuple(write_score_bias_tensor.shape)}"
        )
    if write_score_scale_tensor is not None and (write_score_scale_tensor.ndim != 4 or write_score_scale_tensor.shape[:3] != K.shape[:3]):
        raise ValueError(
            f"write_score_scale_tensor must be [B, N, H, M] with [B, N, H] matching K. Got {tuple(write_score_scale_tensor.shape)}"
        )
    if decode_score_bias_tensor is not None and (decode_score_bias_tensor.ndim != 4 or decode_score_bias_tensor.shape[:3] != K.shape[:3]):
        raise ValueError(
            f"decode_score_bias_tensor must be [B, N, H, M] with [B, N, H] matching K. Got {tuple(decode_score_bias_tensor.shape)}"
        )
    if decode_score_scale_tensor is not None and (decode_score_scale_tensor.ndim != 4 or decode_score_scale_tensor.shape[:3] != K.shape[:3]):
        raise ValueError(
            f"decode_score_scale_tensor must be [B, N, H, M] with [B, N, H] matching K. Got {tuple(decode_score_scale_tensor.shape)}"
        )
    if retain_group_gate_tensor is not None:
        expected_shape = (B, N, H, M, value_num_groups)
        if retain_group_gate_tensor.shape != expected_shape:
            raise ValueError(
                "retain_group_gate_tensor must be [B, N, H, M, G]. "
                f"Got {tuple(retain_group_gate_tensor.shape)}, expected {expected_shape}.",
            )

    score_dtype = torch.float32
    if os.environ.get("FLARE_PYTORCH_MATCH_REFERENCE", "") == "1":
        score_dtype = Q.dtype
    value_dtype = V.dtype
    scale_resolved = float(_resolve_attn_scale(scale, D_score))
    power = float(power)
    eps = torch.finfo(score_dtype).eps
    state_mix_write_scale = 0.1

    experimental_chunk_cap = int(os.environ.get("FLARE_EXPERIMENTAL_CHUNK_SIZE", "16"))
    if experimental_chunk_cap < 1:
        raise ValueError(f"FLARE_EXPERIMENTAL_CHUNK_SIZE must be >= 1. Got {experimental_chunk_cap}")
    if chunk_size is None:
        chunk_size = _resolve_stablemax_chunk_size_pytorch(N, M, D_score, chunk_size)
    chunk_size = min(int(chunk_size), experimental_chunk_cap)

    Q_f = Q.to(score_dtype)
    K_f = K.to(score_dtype)
    V_f = V.to(value_dtype)
    hidden_proj = hidden_states
    Q_dec_f = Q_dec_resolved.to(score_dtype)
    K_dec_f = K_dec_resolved.to(score_dtype)
    gate_f = None if write_gate_tensor is None else write_gate_tensor.to(score_dtype)
    retain_f = None if retain_gate_tensor is None else retain_gate_tensor.to(score_dtype)
    group_gate_f = None if write_group_gate_tensor is None else write_group_gate_tensor.to(score_dtype)
    retain_group_f = None if retain_group_gate_tensor is None else retain_group_gate_tensor.to(score_dtype)

    group_dim = D_value // value_num_groups
    if value_num_groups > 1:
        z_state = torch.zeros((B, H, M, value_num_groups, group_dim), device=K.device, dtype=value_dtype)
        d_state = torch.zeros((B, H, M, value_num_groups), device=K.device, dtype=score_dtype)
    else:
        z_state = torch.zeros((B, H, M, D_value), device=K.device, dtype=value_dtype)
        d_state = torch.zeros((B, H, M), device=K.device, dtype=score_dtype)
    outputs: list[torch.Tensor] = []
    heads_per_group = None
    if q_dynamic_mode == "grouped_outer":
        if q_grouped_outer_num_groups is None or q_grouped_outer_num_groups <= 0:
            raise ValueError(
                "q_grouped_outer_num_groups must be provided and > 0 when q_dynamic_mode='grouped_outer'."
            )
        if H % q_grouped_outer_num_groups != 0:
            raise ValueError(
                f"q_grouped_outer_num_groups must divide num_heads. Got H={H}, groups={q_grouped_outer_num_groups}."
            )
        heads_per_group = H // q_grouped_outer_num_groups

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        K_chunk = K_f[:, start:end].permute(0, 2, 1, 3).contiguous()
        V_chunk = V_f[:, start:end].permute(0, 2, 1, 3).contiguous()
        H_chunk = hidden_proj[:, start:end]
        Q_dec_chunk = Q_dec_f[:, start:end].permute(0, 2, 1, 3).contiguous()
        C = end - start

        if q_dynamic_mode == "full":
            if q_dynamic_proj_weight is None:
                raise ValueError("q_dynamic_proj_weight is required when q_dynamic_mode='full'.")
            q_delta = F.linear(H_chunk, q_dynamic_proj_weight, q_dynamic_proj_bias).to(score_dtype)
            q_delta = q_delta.view(B, C, H, M, D_score).permute(0, 2, 1, 3, 4).contiguous()
            score_chunk = scale_resolved * torch.einsum(
                "bhcd,bhcmd->bhcm",
                K_chunk,
                q_delta + Q_f.unsqueeze(0).unsqueeze(2),
            )
        elif q_dynamic_mode == "outer":
            if q_outer_latent_proj_weight is None or q_outer_head_proj_weight is None:
                raise ValueError("outer dynamic Q_enc mode requires both latent and head projection weights.")
            q_outer_latent = F.linear(H_chunk, q_outer_latent_proj_weight, q_outer_latent_proj_bias).to(score_dtype)
            q_outer_latent = q_outer_latent.view(B, C, H, M).permute(0, 2, 1, 3).contiguous()
            q_outer_head = F.linear(H_chunk, q_outer_head_proj_weight, q_outer_head_proj_bias).to(score_dtype)
            q_outer_head = q_outer_head.view(B, C, H, D_score).permute(0, 2, 1, 3).contiguous()
            score_chunk = scale_resolved * torch.einsum("bhcd,hmd->bhcm", K_chunk, Q_f)
            token_head_score = scale_resolved * torch.einsum("bhcd,bhcd->bhc", K_chunk, q_outer_head).unsqueeze(-1)
            score_chunk = score_chunk + q_outer_latent * token_head_score
        elif q_dynamic_mode == "low_rank":
            if q_low_rank_proj_weight is None or q_low_rank_basis is None:
                raise ValueError("low_rank dynamic Q_enc mode requires both projection weights and basis.")
            rank = q_low_rank_basis.shape[1]
            q_low_rank_coeff = F.linear(H_chunk, q_low_rank_proj_weight, q_low_rank_proj_bias).to(score_dtype)
            q_low_rank_coeff = q_low_rank_coeff.view(B, C, H, M, rank).permute(0, 2, 1, 3, 4).contiguous()
            k_basis_score = scale_resolved * torch.einsum("bhcd,hrd->bhcr", K_chunk, q_low_rank_basis.to(score_dtype))
            score_chunk = scale_resolved * torch.einsum("bhcd,hmd->bhcm", K_chunk, Q_f)
            score_chunk = score_chunk + torch.einsum("bhcmr,bhcr->bhcm", q_low_rank_coeff, k_basis_score)
        elif q_dynamic_mode == "grouped_outer":
            if q_grouped_outer_latent_proj_weight is None or q_grouped_outer_head_proj_weight is None:
                raise ValueError("grouped_outer dynamic Q_enc mode requires both latent and head projection weights.")
            q_group_latent = F.linear(
                H_chunk,
                q_grouped_outer_latent_proj_weight,
                q_grouped_outer_latent_proj_bias,
            ).to(score_dtype)
            q_group_latent = q_group_latent.view(B, C, q_grouped_outer_num_groups, M).permute(0, 2, 1, 3).contiguous()
            q_group_head = F.linear(
                H_chunk,
                q_grouped_outer_head_proj_weight,
                q_grouped_outer_head_proj_bias,
            ).to(score_dtype)
            q_group_head = q_group_head.view(B, C, q_grouped_outer_num_groups, D_score).permute(0, 2, 1, 3).contiguous()
            q_outer_latent = q_group_latent.repeat_interleave(heads_per_group, dim=1)
            q_outer_head = q_group_head.repeat_interleave(heads_per_group, dim=1)
            score_chunk = scale_resolved * torch.einsum("bhcd,hmd->bhcm", K_chunk, Q_f)
            token_head_score = scale_resolved * torch.einsum("bhcd,bhcd->bhc", K_chunk, q_outer_head).unsqueeze(-1)
            score_chunk = score_chunk + q_outer_latent * token_head_score
        else:
            score_chunk = scale_resolved * torch.einsum("bhcd,hmd->bhcm", K_chunk, Q_f)

        if write_score_scale_tensor is not None:
            write_score_scale_chunk = write_score_scale_tensor[:, start:end].permute(0, 2, 1, 3).contiguous().to(score_dtype)
            score_chunk = score_chunk * write_score_scale_chunk
        if write_score_bias_tensor is not None:
            write_score_bias_chunk = write_score_bias_tensor[:, start:end].permute(0, 2, 1, 3).contiguous().to(score_dtype)
            score_chunk = score_chunk + write_score_bias_chunk

        gate_chunk = torch.ones_like(score_chunk) if gate_f is None else gate_f[:, start:end].permute(0, 2, 1, 3).contiguous()
        force_sequential_replay = state_mix_mode != "normalized"
        if state_mix_mode == "normalized" and write_mode == "normalized":
            stable_score = _stablemax_score_transform(score_chunk, power=power)
            if value_num_groups > 1:
                if group_gate_f is None:
                    group_gate_chunk = gate_chunk.unsqueeze(-1).expand(-1, -1, -1, -1, value_num_groups)
                else:
                    group_gate_chunk = group_gate_f[:, start:end].permute(0, 2, 1, 3, 4).contiguous()
                signal = stable_score.unsqueeze(-1) * group_gate_chunk
                total_den = d_state.unsqueeze(2) + signal.cumsum(dim=2)
                total_den_safe = total_den.clamp_min(eps)
                alpha = signal / total_den_safe
                beta = 1.0 - alpha
            else:
                total_den = d_state.unsqueeze(2) + stable_score.cumsum(dim=2)
                total_den_safe = total_den.clamp_min(eps)
                alpha = (stable_score / total_den_safe) * gate_chunk
                beta = 1.0 - alpha
        elif state_mix_mode == "normalized" and write_mode == "rmsnorm_signed":
            stable_score = _stablemax_score_transform(score_chunk, power=power)
            signed_score = _rms_normalize_last_dim(score_chunk, eps=eps, scale_by_sqrt_dim=True)
            if value_num_groups > 1:
                if group_gate_f is None:
                    group_gate_chunk = gate_chunk.unsqueeze(-1).expand(-1, -1, -1, -1, value_num_groups)
                else:
                    group_gate_chunk = group_gate_f[:, start:end].permute(0, 2, 1, 3, 4).contiguous()
                alpha = signed_score.unsqueeze(-1) * group_gate_chunk
                beta = 1.0 - alpha.abs().clamp(max=0.99)
            else:
                alpha = signed_score * gate_chunk
                beta = 1.0 - alpha.abs().clamp(max=0.99)
            total_den = None
        elif state_mix_mode == "decoupled_rw":
            stable_score = _stablemax_score_transform(score_chunk, power=power)
            address = stable_score / stable_score.sum(dim=-1, keepdim=True).clamp_min(eps)
            total_den = None
            if value_num_groups > 1:
                if group_gate_f is None:
                    group_gate_chunk = gate_chunk.unsqueeze(-1).expand(-1, -1, -1, -1, value_num_groups)
                else:
                    group_gate_chunk = group_gate_f[:, start:end].permute(0, 2, 1, 3, 4).contiguous()
                if retain_group_f is not None:
                    beta = retain_group_f[:, start:end].permute(0, 2, 1, 3, 4).contiguous()
                elif retain_f is not None:
                    beta = retain_f[:, start:end].permute(0, 2, 1, 3).contiguous().unsqueeze(-1).expand(
                        -1, -1, -1, -1, value_num_groups
                    )
                else:
                    raise ValueError("decoupled_rw requires retain gate tensors")
                alpha = state_mix_write_scale * group_gate_chunk * address.unsqueeze(-1) * (1.0 - beta)
            else:
                if retain_f is None:
                    raise ValueError("decoupled_rw requires retain_gate_tensor")
                beta = retain_f[:, start:end].permute(0, 2, 1, 3).contiguous()
                alpha = state_mix_write_scale * gate_chunk * address * (1.0 - beta)
        elif state_mix_mode == "decoupled_rw_signed":
            stable_score = _stablemax_score_transform(score_chunk, power=power)
            signed_score = 0.1 * _rms_normalize_last_dim(score_chunk, eps=eps, scale_by_sqrt_dim=True)
            total_den = None
            if value_num_groups > 1:
                if group_gate_f is None:
                    group_gate_chunk = gate_chunk.unsqueeze(-1).expand(-1, -1, -1, -1, value_num_groups)
                else:
                    group_gate_chunk = group_gate_f[:, start:end].permute(0, 2, 1, 3, 4).contiguous()
                if retain_group_f is not None:
                    beta = retain_group_f[:, start:end].permute(0, 2, 1, 3, 4).contiguous()
                elif retain_f is not None:
                    beta = retain_f[:, start:end].permute(0, 2, 1, 3).contiguous().unsqueeze(-1).expand(
                        -1, -1, -1, -1, value_num_groups
                    )
                else:
                    raise ValueError("decoupled_rw_signed requires retain gate tensors")
                alpha = group_gate_chunk * signed_score.unsqueeze(-1) * (1.0 - beta)
            else:
                if retain_f is None:
                    raise ValueError("decoupled_rw_signed requires retain_gate_tensor")
                beta = retain_f[:, start:end].permute(0, 2, 1, 3).contiguous()
                alpha = gate_chunk * signed_score * (1.0 - beta)
        else:
            raise ValueError(f"Unsupported experimental state/write mode combination: {state_mix_mode!r}, {write_mode!r}")

        if v_slot_proj_weight is not None:
            v_delta = F.linear(H_chunk, v_slot_proj_weight, v_slot_proj_bias).to(value_dtype)
            v_delta = v_delta.view(B, C, H, M, D_value).permute(0, 2, 1, 3, 4).contiguous()
        else:
            v_delta = None

        decode_logits = scale_resolved * torch.einsum("bhcd,hmd->bhcm", Q_dec_chunk, K_dec_f)
        if decode_score_scale_tensor is not None:
            decode_score_scale_chunk = decode_score_scale_tensor[:, start:end].permute(0, 2, 1, 3).contiguous().to(score_dtype)
            decode_logits = decode_logits * decode_score_scale_chunk
        if decode_score_bias_tensor is not None:
            decode_score_bias_chunk = decode_score_bias_tensor[:, start:end].permute(0, 2, 1, 3).contiguous().to(score_dtype)
            decode_logits = decode_logits + decode_score_bias_chunk
        if read_mode == "softmax":
            decode_weights = torch.softmax(decode_logits, dim=-1)
        elif read_mode == "signed_rmsnorm":
            decode_weights = _rms_normalize_last_dim(decode_logits, eps=eps, scale_by_sqrt_dim=True)
        else:
            raise ValueError(f"Unsupported experimental read_mode: {read_mode!r}")

        z_prev = z_state
        causal_mask = torch.tril(torch.ones((C, C), device=K.device, dtype=torch.bool))

        if v_delta is None and not force_sequential_replay:
            if value_num_groups > 1:
                alpha_value = alpha.to(value_dtype)
                beta_safe = beta.clamp_min(eps)
                prefix_prod = torch.cumprod(beta_safe, dim=2)
                chunk_A = prefix_prod[:, :, -1, :, :]
                replay_source_coeff = alpha / prefix_prod
                suffix_beta_inclusive = torch.flip(
                    torch.cumprod(torch.flip(beta_safe, dims=(2,)), dim=2),
                    dims=(2,),
                )
                suffix_beta_exclusive = torch.cat(
                    [
                        suffix_beta_inclusive[:, :, 1:, :, :],
                        torch.ones_like(suffix_beta_inclusive[:, :, :1, :, :]),
                    ],
                    dim=2,
                )
                chunk_source_coeff = alpha_value * suffix_beta_exclusive.to(value_dtype)
                V_chunk_group = V_chunk.view(B, H, C, value_num_groups, group_dim)
                chunk_B = torch.einsum("bhcmg,bhcgd->bhmgd", chunk_source_coeff, V_chunk_group)
                recurrent_factor = decode_weights.unsqueeze(-1) * prefix_prod
                prefix_out_groups: list[torch.Tensor] = []
                local_out_groups: list[torch.Tensor] = []
                causal_mask_2d = causal_mask.view(1, 1, C, C)
                for group_idx in range(value_num_groups):
                    recurrent_factor_g = recurrent_factor[:, :, :, :, group_idx]
                    replay_source_g = replay_source_coeff[:, :, :, :, group_idx]
                    prefix_out_groups.append(
                        torch.einsum(
                            "bhcm,bhmd->bhcd",
                            recurrent_factor_g.to(value_dtype),
                            z_prev[:, :, :, group_idx, :],
                        )
                    )
                    local_mix_g = torch.matmul(recurrent_factor_g, replay_source_g.transpose(-1, -2))
                    local_mix_g = torch.where(causal_mask_2d, local_mix_g, torch.zeros_like(local_mix_g))
                    local_out_groups.append(
                        torch.matmul(local_mix_g.to(value_dtype), V_chunk_group[:, :, :, group_idx, :])
                    )
                prefix_out = torch.stack(prefix_out_groups, dim=3)
                local_out = torch.stack(local_out_groups, dim=3)
                z_state = chunk_A.to(value_dtype).unsqueeze(-1) * z_state + chunk_B
                chunk_out = (prefix_out + local_out).reshape(B, H, C, D_value)
            else:
                alpha_value = alpha.to(value_dtype)
                beta_safe = beta.clamp_min(eps)
                prefix_prod = torch.cumprod(beta_safe, dim=2)
                chunk_A = prefix_prod[:, :, -1, :]
                replay_source_coeff = alpha / prefix_prod
                suffix_beta_inclusive = torch.flip(
                    torch.cumprod(torch.flip(beta_safe, dims=(2,)), dim=2),
                    dims=(2,),
                )
                suffix_beta_exclusive = torch.cat(
                    [
                        suffix_beta_inclusive[:, :, 1:, :],
                        torch.ones_like(suffix_beta_inclusive[:, :, :1, :]),
                    ],
                    dim=2,
                )
                chunk_source_coeff = alpha_value * suffix_beta_exclusive.to(value_dtype)
                chunk_B = torch.einsum("bhcm,bhcd->bhmd", chunk_source_coeff, V_chunk)
                recurrent_factor = decode_weights * prefix_prod
                prefix_out = torch.einsum("bhcm,bhmd->bhcd", recurrent_factor.to(value_dtype), z_prev)
                local_mix = torch.matmul(recurrent_factor, replay_source_coeff.transpose(-1, -2))
                local_mix = torch.where(causal_mask.view(1, 1, C, C), local_mix, torch.zeros_like(local_mix))
                local_out = torch.matmul(local_mix.to(value_dtype), V_chunk)
                z_state = chunk_A.to(value_dtype).unsqueeze(-1) * z_state + chunk_B
                chunk_out = prefix_out + local_out
        else:
            if value_num_groups > 1:
                base_value_steps = V_chunk.view(B, H, C, value_num_groups, group_dim).unsqueeze(3)
                value_steps = base_value_steps.expand(-1, -1, -1, M, -1, -1)
                if v_delta is not None:
                    value_steps = value_steps + v_delta.view(B, H, C, M, value_num_groups, group_dim)
                z_state, chunk_out = _stablemax_local_recurrence_autograd(
                    z_state,
                    alpha.to(value_dtype),
                    beta.to(value_dtype),
                    value_steps.to(value_dtype),
                    decode_weights.to(value_dtype),
                )
            else:
                base_value_steps = V_chunk.unsqueeze(3).expand(-1, -1, -1, M, -1)
                value_steps = base_value_steps if v_delta is None else base_value_steps + v_delta
                z_state, chunk_out = _stablemax_local_recurrence_autograd(
                    z_state,
                    alpha.to(value_dtype),
                    beta.to(value_dtype),
                    value_steps.to(value_dtype),
                    decode_weights.to(value_dtype),
                )

        outputs.append(chunk_out.permute(0, 2, 1, 3).contiguous())
        if total_den is not None:
            d_state = total_den[:, :, -1, :]

    return torch.cat(outputs, dim=1).to(V.dtype)


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
    write_gate: bool = False,
    write_gate_fixed_value: float | None = None,
    write_gate_tensor=None,
    state: dict[str, torch.Tensor] | None = None,
    attention_mask: torch.Tensor | None = None,
    return_state: bool = False,
):
    del eps
    if profile:
        raise NotImplementedError("flare_autoregressive_stablemax_mat_decode_pytorch does not support profile=True")
    if state is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_mat_decode_pytorch does not support recurrent state")
    if attention_mask is not None:
        raise NotImplementedError("flare_autoregressive_stablemax_mat_decode_pytorch does not support attention_mask")
    if return_state:
        raise NotImplementedError("flare_autoregressive_stablemax_mat_decode_pytorch does not support return_state=True")
    if write_gate:
        raise NotImplementedError("flare_autoregressive_stablemax_mat_decode_pytorch does not support write_gate=True")
    if write_gate_fixed_value is not None or write_gate_tensor is not None:
        raise ValueError("write_gate parameters require write_gate=True")
    return FLAREAutoregressiveStablemaxMatDecodePyTorch.apply(Q, K, V, C_dec, scale, chunk_size, power)

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
        write_gate_fixed_value: float | None = None,
        write_gate_tensor=None,
        retain_tensor=None,
        write_topk: int = 0,
        write_score_bias_tensor=None,
        write_score_scale_tensor=None,
        decode_score_bias_tensor=None,
        decode_score_scale_tensor=None,
        decode_mode: str = "softmax",
        state_mix_mode: str = "normalized",
        log_diagnostics: bool = False,
        diagnostics_layer_idx: int | None = None,
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
        ctx.write_gate_fixed_value = write_gate_fixed_value
        ctx.write_topk = int(write_topk)
        ctx.compute_dtype = compute_dtype
        ctx.b = B
        ctx.n = N
        ctx.h = H
        ctx.m = M
        ctx.d_score = D_score
        ctx.d_value = D_value
        ctx.scale_resolved = float(scale_resolved)
        C = _resolve_stablemax_chunk_size_pytorch(N, M, D_score, chunk_size)
        ctx.c = C
        ctx.nc = math.ceil(N / C) if N > 0 else 0
        ctx.padded_len = ctx.nc * C
        ctx.q_dtype = Q.dtype
        ctx.k_dtype = K.dtype
        ctx.v_dtype = V.dtype
        ctx.q_dec_dtype = Q_dec_resolved.dtype
        ctx.k_dec_dtype = K_dec_resolved.dtype
        ctx.has_gate_tensor = torch.is_tensor(write_gate_tensor)
        ctx.write_gate_tensor_const = None if ctx.has_gate_tensor else write_gate_tensor
        ctx.has_retain_tensor = torch.is_tensor(retain_tensor)
        ctx.retain_tensor_const = None if ctx.has_retain_tensor else retain_tensor
        ctx.has_write_topk_mask = False
        ctx.has_write_score_bias_tensor = torch.is_tensor(write_score_bias_tensor)
        ctx.write_score_bias_tensor_const = None if ctx.has_write_score_bias_tensor else write_score_bias_tensor
        ctx.has_write_score_scale_tensor = torch.is_tensor(write_score_scale_tensor)
        ctx.write_score_scale_tensor_const = None if ctx.has_write_score_scale_tensor else write_score_scale_tensor
        ctx.has_decode_score_bias_tensor = torch.is_tensor(decode_score_bias_tensor)
        ctx.decode_score_bias_tensor_const = None if ctx.has_decode_score_bias_tensor else decode_score_bias_tensor
        ctx.has_decode_score_scale_tensor = torch.is_tensor(decode_score_scale_tensor)
        ctx.decode_score_scale_tensor_const = None if ctx.has_decode_score_scale_tensor else decode_score_scale_tensor
        ctx.decode_mode = decode_mode
        ctx.state_mix_mode = state_mix_mode
        ctx.log_diagnostics = bool(log_diagnostics)
        ctx.diagnostics_layer_idx = diagnostics_layer_idx

        device = Q.device
        out_dtype = V.dtype
        scale_f = float(scale_resolved)
        power = float(power)
        eps = torch.finfo(compute_dtype).eps
        state_mix_write_scale = 0.1

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
                *([write_gate_tensor] if ctx.has_gate_tensor else []),
                *([retain_tensor] if ctx.has_retain_tensor else []),
                *([write_score_bias_tensor] if ctx.has_write_score_bias_tensor else []),
                *([write_score_scale_tensor] if ctx.has_write_score_scale_tensor else []),
                *([decode_score_bias_tensor] if ctx.has_decode_score_bias_tensor else []),
                *([decode_score_scale_tensor] if ctx.has_decode_score_scale_tensor else []),
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
        score_chunk_raw = scale_f * torch.einsum("bhncd,hmd->bhncm", Kc, Q_f)
        score_chunk = score_chunk_raw
        write_score_bias_chunk = None
        if write_score_bias_tensor is not None:
            write_score_bias_chunk = _pack_stablemax_decode_tensor(
                write_score_bias_tensor,
                B=B,
                N=N,
                H=H,
                M=M,
                NC=ctx.nc,
                C=C,
                device=device,
                dtype=compute_dtype,
                name="write_score_bias_tensor",
            )
            if valid_chunk is not None:
                write_score_bias_chunk = write_score_bias_chunk * valid_chunk.to(dtype=compute_dtype)
            score_chunk = score_chunk + write_score_bias_chunk
        write_score_scale_chunk = None
        if write_score_scale_tensor is not None:
            write_score_scale_chunk = _pack_stablemax_decode_tensor(
                write_score_scale_tensor,
                B=B,
                N=N,
                H=H,
                M=M,
                NC=ctx.nc,
                C=C,
                device=device,
                dtype=compute_dtype,
                name="write_score_scale_tensor",
            )
            if valid_chunk is not None:
                write_score_scale_chunk = torch.where(
                    valid_chunk,
                    write_score_scale_chunk,
                    torch.ones_like(write_score_scale_chunk),
                )
            score_chunk = score_chunk_raw * write_score_scale_chunk
            if write_score_bias_chunk is not None:
                score_chunk = score_chunk + write_score_bias_chunk
        stable_score_chunk = _stablemax_score_transform(score_chunk, power=power)
        if valid_chunk is not None:
            stable_score_chunk = stable_score_chunk * valid_chunk.to(dtype=stable_score_chunk.dtype)
        stable_score_chunk, write_topk_mask = _apply_stablemax_write_topk(
            score_chunk,
            stable_score_chunk,
            write_topk=ctx.write_topk,
            valid_chunk=valid_chunk,
        )
        ctx.has_write_topk_mask = write_topk_mask is not None
        score_chunk_den = stable_score_chunk.sum(dim=3)
        score_prev_den = torch.cumsum(score_chunk_den, dim=2) - score_chunk_den

        gate_chunk = _stablemax_write_gate(
            score_chunk,
            B=B,
            N=N,
            H=H,
            M=M,
            NC=ctx.nc,
            C=C,
            valid_chunk=valid_chunk,
            write_gate_fixed_value=write_gate_fixed_value,
            write_gate_tensor=write_gate_tensor,
        )
        retain_chunk = None
        if retain_tensor is not None:
            retain_chunk = _pack_stablemax_gate_tensor(
                retain_tensor,
                B=B,
                N=N,
                H=H,
                M=M,
                NC=ctx.nc,
                C=C,
                device=device,
                dtype=compute_dtype,
                name="retain_tensor",
            )
            if valid_chunk is not None:
                retain_chunk = torch.where(
                    valid_chunk,
                    retain_chunk,
                    torch.ones_like(retain_chunk),
                )

        total_den = score_prev_den.unsqueeze(3) + stable_score_chunk.cumsum(dim=3)
        total_den_safe = total_den.clamp_min(eps)
        if state_mix_mode == "normalized":
            alpha_chunk = (stable_score_chunk / total_den_safe) * gate_chunk
            beta_chunk = 1.0 - alpha_chunk
        elif state_mix_mode == "decoupled_rw":
            address_den = stable_score_chunk.sum(dim=-1, keepdim=True).clamp_min(eps)
            address_chunk = stable_score_chunk / address_den
            beta_chunk = retain_chunk
            alpha_chunk = state_mix_write_scale * gate_chunk * address_chunk * (1.0 - beta_chunk)
        elif state_mix_mode == "decoupled_rw_signed":
            address_chunk = _rms_normalize_last_dim(score_chunk, eps=eps, scale_by_sqrt_dim=True)
            if valid_chunk is not None:
                address_chunk = torch.where(valid_chunk, address_chunk, torch.zeros_like(address_chunk))
            beta_chunk = retain_chunk
            alpha_chunk = state_mix_write_scale * gate_chunk * address_chunk * (1.0 - beta_chunk)
        else:
            raise ValueError(f"Unsupported state_mix_mode: {state_mix_mode!r}")
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

        z_prev_chunk = torch.empty((B, H, ctx.nc, M, D_value), device=device, dtype=compute_dtype)
        z_curr = torch.zeros((B, H, M, D_value), device=device, dtype=compute_dtype)
        for chunk_idx in range(ctx.nc):
            z_prev_chunk[:, :, chunk_idx, :, :] = z_curr
            z_curr = chunk_A[:, :, chunk_idx, :].unsqueeze(-1) * z_curr + chunk_B[:, :, chunk_idx, :, :]

        decode_logits_raw = scale_f * torch.einsum("bhncd,bhmd->bhncm", Q_dec_c, K_dec_f)
        decode_logits = decode_logits_raw
        if decode_score_scale_tensor is not None:
            decode_score_scale_chunk = _pack_stablemax_decode_tensor(
                decode_score_scale_tensor,
                B=B,
                N=N,
                H=H,
                M=M,
                NC=ctx.nc,
                C=C,
                device=device,
                dtype=compute_dtype,
                name="decode_score_scale_tensor",
            )
            if valid_chunk is not None:
                decode_score_scale_chunk = torch.where(
                    valid_chunk,
                    decode_score_scale_chunk,
                    torch.ones_like(decode_score_scale_chunk),
                )
            decode_logits = decode_logits_raw * decode_score_scale_chunk
        if decode_score_bias_tensor is not None:
            decode_score_bias_chunk = _pack_stablemax_decode_tensor(
                decode_score_bias_tensor,
                B=B,
                N=N,
                H=H,
                M=M,
                NC=ctx.nc,
                C=C,
                device=device,
                dtype=compute_dtype,
                name="decode_score_bias_tensor",
            )
            if valid_chunk is not None:
                decode_score_bias_chunk = decode_score_bias_chunk * valid_chunk.to(dtype=compute_dtype)
            decode_logits = decode_logits + decode_score_bias_chunk
        if decode_mode == "softmax":
            decode_probs = torch.softmax(decode_logits, dim=-1)
            decode_diag_probs = decode_probs
        elif decode_mode == "signed_rmsnorm":
            decode_probs = _rms_normalize_last_dim(decode_logits, eps=eps, scale_by_sqrt_dim=True)
            decode_diag_probs = decode_probs.abs()
            decode_diag_probs = decode_diag_probs / decode_diag_probs.sum(dim=-1, keepdim=True).clamp_min(eps)
        else:
            raise ValueError(f"Unsupported decode_mode: {decode_mode!r}")
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
        if not torch.isfinite(Y).all():
            debug_stats = {
                "score_chunk_absmax": float(score_chunk.detach().abs().max().item()),
                "stable_score_absmax": float(stable_score_chunk.detach().abs().max().item()),
                "gate_min": float(gate_chunk.detach().min().item()),
                "gate_max": float(gate_chunk.detach().max().item()),
                "beta_min": float(beta_chunk.detach().min().item()),
                "beta_max": float(beta_chunk.detach().max().item()),
                "alpha_absmax": float(alpha_chunk.detach().abs().max().item()),
                "prefix_prod_absmax": float(prefix_prod.detach().abs().max().item()),
                "prefix_prod_min": float(prefix_prod.detach().min().item()),
                "replay_source_absmax": float(replay_source_coeff.detach().abs().max().item()),
                "recurrent_factor_absmax": float(recurrent_factor.detach().abs().max().item()),
                "local_mix_absmax": float(local_mix.detach().abs().max().item()),
                "chunk_B_absmax": float(chunk_B.detach().abs().max().item()),
                "z_prev_chunk_absmax": float(z_prev_chunk.detach().abs().max().item()),
                "prefix_out_absmax": float(prefix_out.detach().abs().max().item()),
                "local_out_absmax": float(local_out.detach().abs().max().item()),
                "Yc_absmax": float(Yc.detach().abs().max().item()),
            }
            raise RuntimeError(f"FLARE fast decoupled state-mix produced nonfinite Y with stats={debug_stats}")
        _check_finite("FLAREAutoregressiveStablemaxWriteGatedPyTorch.Y", Y)
        if ctx.log_diagnostics:
            _log_stablemax_latent_diagnostics(
                stable_score_chunk=stable_score_chunk.detach(),
                decode_probs=decode_diag_probs.detach(),
                alpha_chunk=alpha_chunk.detach(),
                valid_chunk=valid_chunk,
                layer_idx=ctx.diagnostics_layer_idx,
            )

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
        if ctx.has_gate_tensor:
            saved.append(write_gate_tensor)
        if ctx.has_retain_tensor:
            saved.append(retain_tensor)
        if ctx.has_write_topk_mask:
            saved.append(write_topk_mask)
        if ctx.has_write_score_bias_tensor:
            saved.append(write_score_bias_tensor)
        if ctx.has_write_score_scale_tensor:
            saved.append(write_score_scale_tensor)
        if ctx.has_decode_score_bias_tensor:
            saved.append(decode_score_bias_tensor)
        if ctx.has_decode_score_scale_tensor:
            saved.append(decode_score_scale_tensor)
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
        gate_tensor_saved = saved.pop(0) if ctx.has_gate_tensor else ctx.write_gate_tensor_const
        retain_tensor_saved = saved.pop(0) if ctx.has_retain_tensor else ctx.retain_tensor_const
        write_topk_mask_saved = saved.pop(0) if ctx.has_write_topk_mask else None
        write_score_bias_saved = (
            saved.pop(0) if ctx.has_write_score_bias_tensor else ctx.write_score_bias_tensor_const
        )
        write_score_scale_saved = (
            saved.pop(0) if ctx.has_write_score_scale_tensor else ctx.write_score_scale_tensor_const
        )
        decode_score_bias_saved = (
            saved.pop(0) if ctx.has_decode_score_bias_tensor else ctx.decode_score_bias_tensor_const
        )
        decode_score_scale_saved = (
            saved.pop(0) if ctx.has_decode_score_scale_tensor else ctx.decode_score_scale_tensor_const
        )

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
        state_mix_write_scale = 0.1

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
        grad_gate_tensor = None
        grad_retain_tensor = None
        grad_write_score_bias = None
        grad_write_score_scale = None
        grad_decode_score_bias = None
        grad_decode_score_scale = None

        use_direct_gate = gate_tensor_saved is not None

        stable_score_grad = _stablemax_score_transform_grad(score_chunk, power=power)
        causal_mask = torch.tril(torch.ones((C, C), device=device, dtype=torch.bool)).view(1, 1, 1, C, C)

        s_chunk = stable_score_chunk
        p_chunk = decode_probs
        v_chunk = Vc
        d_start = score_prev_den
        z_start = z_prev_chunk

        total_den = d_start.unsqueeze(3) + s_chunk.cumsum(dim=3)
        total_den_safe = total_den.clamp_min(eps)
        if use_direct_gate:
            gate_chunk = _pack_stablemax_gate_tensor(
                gate_tensor_saved,
                B=B,
                N=N,
                H=H,
                M=M,
                NC=NC,
                C=C,
                device=device,
                dtype=compute_dtype,
                name="write_gate_tensor",
            )
            if valid_chunk is not None:
                gate_mask = valid_chunk.to(dtype=compute_dtype)
                gate_chunk = gate_chunk * gate_mask
            else:
                gate_mask = None
            if ctx.has_gate_tensor and ctx.needs_input_grad[9]:
                grad_gate_tensor = torch.zeros_like(gate_tensor_saved, dtype=compute_dtype)
        elif ctx.write_gate_fixed_value is not None:
            gate_chunk = torch.full_like(s_chunk, float(ctx.write_gate_fixed_value))
            gate_mask = None
        else:
            gate_chunk = torch.ones_like(s_chunk)
            gate_mask = None
        retain_chunk = None
        retain_mask = None
        if retain_tensor_saved is not None:
            retain_chunk = _pack_stablemax_gate_tensor(
                retain_tensor_saved,
                B=B,
                N=N,
                H=H,
                M=M,
                NC=NC,
                C=C,
                device=device,
                dtype=compute_dtype,
                name="retain_tensor",
            )
            if valid_chunk is not None:
                retain_mask = valid_chunk.to(dtype=compute_dtype)
                retain_chunk = torch.where(valid_chunk, retain_chunk, torch.ones_like(retain_chunk))
            if ctx.has_retain_tensor and ctx.needs_input_grad[10]:
                grad_retain_tensor = torch.zeros_like(retain_tensor_saved, dtype=compute_dtype)

        write_score_bias_chunk = None
        if write_score_bias_saved is not None:
            write_score_bias_chunk = _pack_stablemax_decode_tensor(
                write_score_bias_saved,
                B=B,
                N=N,
                H=H,
                M=M,
                NC=NC,
                C=C,
                device=device,
                dtype=compute_dtype,
                name="write_score_bias_tensor",
            )
            if valid_chunk is not None:
                write_score_bias_chunk = write_score_bias_chunk * valid_chunk.to(dtype=compute_dtype)
        write_score_scale_chunk = None
        if write_score_scale_saved is not None:
            write_score_scale_chunk = _pack_stablemax_decode_tensor(
                write_score_scale_saved,
                B=B,
                N=N,
                H=H,
                M=M,
                NC=NC,
                C=C,
                device=device,
                dtype=compute_dtype,
                name="write_score_scale_tensor",
            )
            if valid_chunk is not None:
                write_score_scale_chunk = torch.where(
                    valid_chunk,
                    write_score_scale_chunk,
                    torch.ones_like(write_score_scale_chunk),
                )

        decode_score_bias_chunk = None
        if decode_score_bias_saved is not None:
            decode_score_bias_chunk = _pack_stablemax_decode_tensor(
                decode_score_bias_saved,
                B=B,
                N=N,
                H=H,
                M=M,
                NC=NC,
                C=C,
                device=device,
                dtype=compute_dtype,
                name="decode_score_bias_tensor",
            )
            if valid_chunk is not None:
                decode_score_bias_chunk = decode_score_bias_chunk * valid_chunk.to(dtype=compute_dtype)
        decode_score_scale_chunk = None
        if decode_score_scale_saved is not None:
            decode_score_scale_chunk = _pack_stablemax_decode_tensor(
                decode_score_scale_saved,
                B=B,
                N=N,
                H=H,
                M=M,
                NC=NC,
                C=C,
                device=device,
                dtype=compute_dtype,
                name="decode_score_scale_tensor",
            )
            if valid_chunk is not None:
                decode_score_scale_chunk = torch.where(
                    valid_chunk,
                    decode_score_scale_chunk,
                    torch.ones_like(decode_score_scale_chunk),
                )

        if ctx.state_mix_mode == "normalized":
            alpha_chunk = (s_chunk / total_den_safe) * gate_chunk
            beta_chunk = 1.0 - alpha_chunk
            address_chunk = None
        elif ctx.state_mix_mode == "decoupled_rw":
            address_den = s_chunk.sum(dim=-1, keepdim=True).clamp_min(eps)
            address_chunk = s_chunk / address_den
            beta_chunk = retain_chunk
            alpha_chunk = state_mix_write_scale * gate_chunk * address_chunk * (1.0 - beta_chunk)
        elif ctx.state_mix_mode == "decoupled_rw_signed":
            address_chunk = _rms_normalize_last_dim(score_chunk, eps=eps, scale_by_sqrt_dim=True)
            if valid_chunk is not None:
                address_chunk = torch.where(valid_chunk, address_chunk, torch.zeros_like(address_chunk))
            beta_chunk = retain_chunk
            alpha_chunk = state_mix_write_scale * gate_chunk * address_chunk * (1.0 - beta_chunk)
        else:
            raise ValueError(f"Unsupported state_mix_mode in backward: {ctx.state_mix_mode!r}")
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
        grad_beta_chunk = grad_beta_from_prefix + grad_beta_from_suffix

        if ctx.state_mix_mode == "normalized":
            grad_alpha_chunk = grad_alpha_chunk - grad_beta_chunk

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
            grad_chunk_den = (
                torch.flip(torch.cumsum(torch.flip(grad_prev_den_local, dims=(2,)), dim=2), dims=(2,))
                - grad_prev_den_local
            )
            grad_s_chunk = grad_s_chunk + grad_chunk_den.unsqueeze(3)

            grad_score_enc = grad_s_chunk * stable_score_grad
        elif ctx.state_mix_mode == "decoupled_rw":
            one_minus_beta = 1.0 - beta_chunk
            grad_gate_chunk = grad_alpha_chunk * (state_mix_write_scale * address_chunk * one_minus_beta)
            grad_address_chunk = grad_alpha_chunk * (state_mix_write_scale * gate_chunk * one_minus_beta)
            grad_beta_total = grad_beta_chunk - grad_alpha_chunk * (state_mix_write_scale * gate_chunk * address_chunk)
            grad_retain_chunk = grad_beta_total * beta_chunk * (1.0 - beta_chunk)
            grad_address_dot = (grad_address_chunk * address_chunk).sum(dim=-1, keepdim=True)
            grad_s_chunk = (grad_address_chunk - grad_address_dot) / address_den
            grad_score_enc = grad_s_chunk * stable_score_grad
        elif ctx.state_mix_mode == "decoupled_rw_signed":
            one_minus_beta = 1.0 - beta_chunk
            grad_gate_chunk = grad_alpha_chunk * (state_mix_write_scale * address_chunk * one_minus_beta)
            grad_address_chunk = grad_alpha_chunk * (state_mix_write_scale * gate_chunk * one_minus_beta)
            grad_beta_total = grad_beta_chunk - grad_alpha_chunk * (state_mix_write_scale * gate_chunk * address_chunk)
            grad_retain_chunk = grad_beta_total * beta_chunk * (1.0 - beta_chunk)
            grad_score_enc = _rms_normalize_last_dim_backward(
                grad_address_chunk,
                score_chunk,
                eps=eps,
                scale_by_sqrt_dim=True,
            )
        else:
            raise ValueError(f"Unsupported state_mix_mode in backward: {ctx.state_mix_mode!r}")
        if write_topk_mask_saved is not None:
            grad_score_enc = grad_score_enc * write_topk_mask_saved.to(dtype=grad_score_enc.dtype)
        if use_direct_gate:
            grad_gate_chunk = grad_gate_chunk if gate_mask is None else grad_gate_chunk * gate_mask
            grad_gate_tensor = _unpack_stablemax_decode_tensor(grad_gate_chunk, N=N)
        if grad_retain_tensor is not None:
            grad_retain_chunk = grad_retain_chunk if retain_mask is None else grad_retain_chunk * retain_mask
            grad_retain_tensor = _unpack_stablemax_decode_tensor(grad_retain_chunk, N=N)

        score_chunk_raw = scale_f * torch.einsum("bhncd,hmd->bhncm", Kc, Q_f)
        grad_score_enc_raw = grad_score_enc
        if write_score_scale_chunk is not None:
            grad_write_score_scale = grad_score_enc * score_chunk_raw
            grad_score_enc_raw = grad_score_enc_raw * write_score_scale_chunk
        if write_score_bias_chunk is not None:
            grad_write_score_bias = grad_score_enc

        decode_logits_raw = scale_f * torch.einsum("bhncd,bhmd->bhncm", Q_dec_c, K_dec_f)
        decode_logits = decode_logits_raw
        if decode_score_scale_chunk is not None:
            decode_logits = decode_logits_raw * decode_score_scale_chunk
        if decode_score_bias_chunk is not None:
            decode_logits = decode_logits + decode_score_bias_chunk

        if ctx.decode_mode == "softmax":
            softmax_dot = (grad_decode_probs * decode_probs).sum(dim=-1, keepdim=True)
            grad_score_dec = decode_probs * (grad_decode_probs - softmax_dot)
        elif ctx.decode_mode == "signed_rmsnorm":
            grad_score_dec = _rms_normalize_last_dim_backward(
                grad_decode_probs,
                decode_logits,
                eps=eps,
                scale_by_sqrt_dim=True,
            )
        else:
            raise ValueError(f"Unsupported decode_mode in backward: {ctx.decode_mode!r}")
        grad_score_dec_raw = grad_score_dec
        if decode_score_scale_chunk is not None:
            grad_decode_score_scale = grad_score_dec * decode_logits_raw
            grad_score_dec_raw = grad_score_dec_raw * decode_score_scale_chunk
        if decode_score_bias_chunk is not None:
            grad_decode_score_bias = grad_score_dec

        grad_Q_dec_c = scale_f * torch.einsum("bhncm,bhmd->bhncd", grad_score_dec_raw, K_dec_f)
        grad_K_dec = (scale_f * torch.einsum("bhncm,bhncd->bhmd", grad_score_dec_raw, Q_dec_c)).sum(dim=0)

        grad_Q = scale_f * torch.einsum("bhncm,bhncd->hmd", grad_score_enc_raw, Kc)
        grad_Kc = scale_f * torch.einsum("bhncm,hmd->bhncd", grad_score_enc_raw, Q_f)
        grad_K = grad_Kc.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_score)[:, :N, :, :]
        grad_V = grad_V_total.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_value)[:, :N, :, :]
        grad_Q_dec = grad_Q_dec_c.permute(0, 2, 3, 1, 4).reshape(B, PADDED_LEN, H, D_score)[:, :N, :, :]

        grad_Q = grad_Q.to(ctx.q_dtype) if ctx.needs_input_grad[0] else None
        grad_K = grad_K.to(ctx.k_dtype) if ctx.needs_input_grad[1] else None
        grad_V = grad_V.to(ctx.v_dtype) if ctx.needs_input_grad[2] else None
        grad_Q_dec = grad_Q_dec.to(ctx.q_dec_dtype) if ctx.needs_input_grad[3] else None
        grad_K_dec = grad_K_dec.to(ctx.k_dec_dtype) if ctx.needs_input_grad[4] else None
        if grad_gate_tensor is not None:
            grad_gate_tensor = grad_gate_tensor.to(dtype=gate_tensor_saved.dtype)
        if grad_retain_tensor is not None:
            grad_retain_tensor = grad_retain_tensor.to(dtype=retain_tensor_saved.dtype)
        if grad_write_score_bias is not None:
            grad_write_score_bias = _unpack_stablemax_decode_tensor(grad_write_score_bias, N=N).to(
                dtype=write_score_bias_saved.dtype,
            )
        if grad_write_score_scale is not None:
            grad_write_score_scale = _unpack_stablemax_decode_tensor(grad_write_score_scale, N=N).to(
                dtype=write_score_scale_saved.dtype,
            )
        if grad_decode_score_bias is not None:
            grad_decode_score_bias = _unpack_stablemax_decode_tensor(grad_decode_score_bias, N=N).to(
                dtype=decode_score_bias_saved.dtype,
            )
        if grad_decode_score_scale is not None:
            grad_decode_score_scale = _unpack_stablemax_decode_tensor(grad_decode_score_scale, N=N).to(
                dtype=decode_score_scale_saved.dtype,
            )

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
            grad_gate_tensor,
            grad_retain_tensor,
            None,
            grad_write_score_bias,
            grad_write_score_scale,
            grad_decode_score_bias,
            grad_decode_score_scale,
            None,
            None,
            None,
            None,
        )

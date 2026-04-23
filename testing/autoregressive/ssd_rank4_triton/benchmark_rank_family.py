from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import causal_flare.autoregressive.ssd_rank1_triton as s1  # noqa: E402
import causal_flare.autoregressive.ssd_rank3_triton as s3  # noqa: E402
import causal_flare.autoregressive.ssd_rank4_triton as s4  # noqa: E402


# Snapshot in the header comment of ssd_rank4_triton.py.
RANK4_COMMENT_BASELINE = {
    1: {"fwd_p1": 0.740, "fwd_p2": 0.450, "fwd_p3": 0.867, "fwd_total": 2.057, "bwd_p3": 3.482, "bwd_p2": 0.619, "bwd_p1": 1.006, "bwd_total": 5.107, "step_total": 7.164},
    2: {"fwd_p1": 1.149, "fwd_p2": 0.447, "fwd_p3": 1.148, "fwd_total": 2.744, "bwd_p3": 5.294, "bwd_p2": 0.616, "bwd_p1": 2.425, "bwd_total": 8.335, "step_total": 11.078},
    3: {"fwd_p1": 1.700, "fwd_p2": 0.452, "fwd_p3": 1.753, "fwd_total": 3.905, "bwd_p3": 7.151, "bwd_p2": 0.616, "bwd_p1": 3.763, "bwd_total": 11.530, "step_total": 15.435},
    4: {"fwd_p1": 2.145, "fwd_p2": 0.447, "fwd_p3": 2.932, "fwd_total": 5.525, "bwd_p3": 9.006, "bwd_p2": 0.616, "bwd_p1": 4.980, "bwd_total": 14.601, "step_total": 20.126},
}

TIMING_KEYS = ("fwd_p1", "fwd_p2", "fwd_p3", "fwd_total", "bwd_p3", "bwd_p2", "bwd_p1", "bwd_total", "step_total")


def _time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _avg(xs: list[float]) -> float:
    return float(sum(xs) / len(xs))


def _make_inputs(
    *,
    B: int,
    H: int,
    N: int,
    M: int,
    D: int,
    rank: int,
    dtype: torch.dtype,
    device: torch.device,
    log_alpha_mode: str,
) -> tuple[torch.Tensor, ...]:
    C = torch.randn(B, N, H, M, device=device, dtype=dtype)
    W1 = torch.randn(B, N, H, M, device=device, dtype=dtype)
    V1 = torch.randn(B, N, H, D, device=device, dtype=dtype)
    if log_alpha_mode == "softplus":
        log_alpha_f32 = -torch.nn.functional.softplus(torch.randn(B, N, H, device=device, dtype=torch.float32))
    elif log_alpha_mode == "narrow":
        log_alpha_f32 = -0.05 * torch.rand(B, N, H, device=device, dtype=torch.float32)
    elif log_alpha_mode == "mid":
        log_alpha_f32 = -0.5 * torch.rand(B, N, H, device=device, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported log_alpha_mode={log_alpha_mode}.")
    log_alpha = log_alpha_f32.to(dtype)
    W2 = torch.randn(B, N, H, M, device=device, dtype=dtype) if rank >= 2 else None
    V2 = torch.randn(B, N, H, D, device=device, dtype=dtype) if rank >= 2 else None
    W3 = torch.randn(B, N, H, M, device=device, dtype=dtype) if rank >= 3 else None
    V3 = torch.randn(B, N, H, D, device=device, dtype=dtype) if rank >= 3 else None
    W4 = torch.randn(B, N, H, M, device=device, dtype=dtype) if rank >= 4 else None
    V4 = torch.randn(B, N, H, D, device=device, dtype=dtype) if rank >= 4 else None
    return C, W1, V1, log_alpha, W2, V2, W3, V3, W4, V4


def _summarize_phase_lists(
    fwd_p1: list[float],
    fwd_p2: list[float],
    fwd_p3: list[float],
    bwd_p3: list[float],
    bwd_p2: list[float],
    bwd_p1: list[float],
) -> dict[str, float]:
    out = {
        "fwd_p1": _avg(fwd_p1),
        "fwd_p2": _avg(fwd_p2),
        "fwd_p3": _avg(fwd_p3),
        "bwd_p3": _avg(bwd_p3),
        "bwd_p2": _avg(bwd_p2),
        "bwd_p1": _avg(bwd_p1),
    }
    out["fwd_total"] = out["fwd_p1"] + out["fwd_p2"] + out["fwd_p3"]
    out["bwd_total"] = out["bwd_p3"] + out["bwd_p2"] + out["bwd_p1"]
    out["step_total"] = out["fwd_total"] + out["bwd_total"]
    return out


def _phase_timing_rank1_backend(
    *,
    B: int,
    H: int,
    N: int,
    M: int,
    D: int,
    chunk_size: int,
    input_precision: str,
    warmup: int,
    iters: int,
    log_alpha_mode: str,
) -> dict[str, float]:
    s1._ensure_triton_allocator()
    s1.set_ssd_rank1_static_shape(N=N, M=M, D=D)
    device = torch.device("cuda")
    C, W1, V1, log_alpha, _, _, _, _, _, _ = _make_inputs(
        B=B,
        H=H,
        N=N,
        M=M,
        D=D,
        rank=1,
        dtype=torch.bfloat16,
        device=device,
        log_alpha_mode=log_alpha_mode,
    )

    cfg = s1._validate_static_hot_path_contract(C, W1, V1, log_alpha, None, chunk_size, input_precision, True)
    Cc, W1c, V1c, logc, _, _, _, _, _, NC, BH = s1._ssd_rank1_prepare_unchunked_inputs_static(C, W1, V1, log_alpha, cfg=cfg)
    ws = s1._get_static_workspace(device=device, cfg_key=(BH, N, M, D), cfg=cfg)

    fwd_p1, fwd_p2, fwd_p3 = [], [], []
    bwd_p3, bwd_p2, bwd_p1 = [], [], []

    for i in range(warmup + iters):
        t_f1 = _time_ms(lambda: s1._ssd_rank1_chunk_end_state_forward_impl_static(W1c, V1c, logc, cfg=cfg, ws=ws))

        s_local = ws.s_local_end_md.reshape(BH, NC, M * D)
        log_per_chunk = logc.sum(dim=2, dtype=torch.float32).contiguous()
        final_state = torch.empty((BH, M * D), device=device, dtype=torch.float32)
        t_f2 = _time_ms(lambda: s1._ssd_rank1_phase2_forward_static(s_local, log_per_chunk, final_state, cfg=cfg, ws=ws))

        s0_chunk = ws.phase2_chunk_start
        t_f3 = _time_ms(lambda: s1._ssd_rank1_dense_output_forward_impl_static(Cc, W1c, V1c, logc, s0_chunk, cfg=cfg, ws=ws))

        y_chunk = s1._ssd_rank1_dense_output_forward_impl_static(Cc, W1c, V1c, logc, s0_chunk, cfg=cfg, ws=ws)
        grad_y = torch.randn_like(y_chunk)

        t_b3 = _time_ms(lambda: s1._ssd_rank1_dense_output_backward_impl_static(Cc, W1c, V1c, logc, grad_y, s0_chunk, cfg=cfg, ws=ws))
        dC, dW1, dV1, dlog_phase3, dS0 = s1._ssd_rank1_dense_output_backward_impl_static(
            Cc, W1c, V1c, logc, grad_y, s0_chunk, cfg=cfg, ws=ws
        )
        del dC
        dlog_chunk = ws.dlog_chunk_accum
        dlog_chunk.copy_(dlog_phase3)

        grad_final = ws.phase2_grad_final_zero
        grad_final.zero_()
        t_b2 = _time_ms(lambda: s1._ssd_rank1_phase2_backward_static(dS0, grad_final, s0_chunk, log_per_chunk, cfg=cfg, ws=ws))
        dS_local_end, d_log_per_chunk, _ = s1._ssd_rank1_phase2_backward_static(
            dS0, grad_final, s0_chunk, log_per_chunk, cfg=cfg, ws=ws
        )
        dlog_chunk.add_(d_log_per_chunk.unsqueeze(-1).to(dlog_chunk.dtype))

        grad_s_md = dS_local_end.reshape(BH, NC, M, D)
        grid_phase1 = (BH * NC,)
        t_b1 = _time_ms(
            lambda: s1.ssd_rank1_chunk_end_state_bwd_fused_kernel[grid_phase1](
                grad_s_md,
                W1c,
                V1c,
                logc,
                dW1,
                dV1,
                dlog_chunk,
                B,
                H,
                BH,
                NC,
                cfg.chunk_size,
                M,
                D,
                *grad_s_md.stride(),
                *W1c.stride(),
                *V1c.stride(),
                *logc.stride(),
                *dW1.stride(),
                *dV1.stride(),
                *dlog_chunk.stride(),
                BLOCK_M=cfg.phase1_block_m,
                BLOCK_D=cfg.phase1_block_d,
                BLOCK_C=cfg.chunk_size,
                C_STATIC=cfg.chunk_size,
                M_STATIC=M,
                D_STATIC=D,
                INPUT_PRECISION=cfg.input_precision,
                USE_BF16_DOT_INPUTS=True,
                ACCUMULATE=True,
                num_warps=cfg.phase1_backward.num_warps,
                num_stages=cfg.phase1_backward.num_stages,
            )
        )

        if i >= warmup:
            fwd_p1.append(t_f1)
            fwd_p2.append(t_f2)
            fwd_p3.append(t_f3)
            bwd_p3.append(t_b3)
            bwd_p2.append(t_b2)
            bwd_p1.append(t_b1)

    return _summarize_phase_lists(fwd_p1, fwd_p2, fwd_p3, bwd_p3, bwd_p2, bwd_p1)


def _phase_timing_rank3_backend(
    *,
    rank: int,
    B: int,
    H: int,
    N: int,
    M: int,
    D: int,
    chunk_size: int,
    input_precision: str,
    warmup: int,
    iters: int,
    log_alpha_mode: str,
) -> dict[str, float]:
    if rank < 1 or rank > 3:
        raise ValueError(f"ssd_rank3 backend only supports rank in [1,3], got {rank}.")

    s3._ensure_triton_allocator()
    s3.set_ssd_rank1_static_shape(N=N, M=M, D=D)
    device = torch.device("cuda")
    C, W1, V1, log_alpha, W2, V2, W3, V3, _, _ = _make_inputs(
        B=B,
        H=H,
        N=N,
        M=M,
        D=D,
        rank=3,
        dtype=torch.bfloat16,
        device=device,
        log_alpha_mode=log_alpha_mode,
    )
    has_rank2 = rank >= 2
    has_rank3 = rank >= 3
    if not has_rank2:
        W2, V2 = W1.detach(), V1.detach()
    if not has_rank3:
        W3, V3 = W1.detach(), V1.detach()

    cfg = s3._validate_static_hot_path_contract(C, W1, V1, log_alpha, None, chunk_size, input_precision, True)
    Cc, W1c, V1c, W2c, V2c, W3c, V3c, logc, _, _, _, _, _, NC, BH = s3._ssd_rank3_prepare_unchunked_inputs_static(
        C, W1, V1, W2, V2, W3, V3, log_alpha, cfg=cfg
    )
    ws = s3._get_static_workspace(device=device, cfg_key=(BH, N, M, D), cfg=cfg, allocate_phase3_s0=False)

    fwd_p1, fwd_p2, fwd_p3 = [], [], []
    bwd_p3, bwd_p2, bwd_p1 = [], [], []

    for i in range(warmup + iters):
        t_f1 = _time_ms(
            lambda: s3._ssd_rank3_chunk_end_state_forward_impl_static(
                W1c, V1c, W2c, V2c, W3c, V3c, logc, cfg=cfg, ws=ws, has_rank2=has_rank2, has_rank3=has_rank3
            )
        )

        s_local = ws.s_local_end_md.reshape(BH, NC, M * D)
        log_per_chunk = logc.sum(dim=2, dtype=torch.float32).contiguous()
        final_state = torch.empty((BH, M * D), device=device, dtype=torch.float32)
        t_f2 = _time_ms(lambda: s3._ssd_rank1_phase2_forward_static(s_local, log_per_chunk, final_state, cfg=cfg, ws=ws))

        s0_chunk = ws.phase2_chunk_start
        t_f3 = _time_ms(
            lambda: s3._ssd_rank3_dense_output_forward_impl_static(
                Cc, W1c, V1c, W2c, V2c, W3c, V3c, logc, s0_chunk, cfg=cfg, ws=ws, has_rank2=has_rank2, has_rank3=has_rank3
            )
        )

        y_chunk = s3._ssd_rank3_dense_output_forward_impl_static(
            Cc, W1c, V1c, W2c, V2c, W3c, V3c, logc, s0_chunk, cfg=cfg, ws=ws, has_rank2=has_rank2, has_rank3=has_rank3
        )
        grad_y = torch.randn_like(y_chunk)

        t_b3 = _time_ms(
            lambda: s3._ssd_rank3_dense_output_backward_impl_static(
                Cc,
                W1c,
                V1c,
                W2c,
                V2c,
                W3c,
                V3c,
                logc,
                grad_y,
                s0_chunk,
                cfg=cfg,
                ws=ws,
                has_rank2=has_rank2,
                has_rank3=has_rank3,
            )
        )
        dC, dW1, dV1, dW2, dV2, dW3, dV3, dlog_phase3, dS0 = s3._ssd_rank3_dense_output_backward_impl_static(
            Cc, W1c, V1c, W2c, V2c, W3c, V3c, logc, grad_y, s0_chunk, cfg=cfg, ws=ws, has_rank2=has_rank2, has_rank3=has_rank3
        )
        del dC
        dlog_chunk = ws.dlog_chunk_accum
        dlog_chunk.copy_(dlog_phase3)

        grad_final = ws.phase2_grad_final_zero
        grad_final.zero_()
        t_b2 = _time_ms(lambda: s3._ssd_rank1_phase2_backward_static(dS0, grad_final, s0_chunk, log_per_chunk, cfg=cfg, ws=ws))
        dS_local_end, d_log_per_chunk, _ = s3._ssd_rank1_phase2_backward_static(
            dS0, grad_final, s0_chunk, log_per_chunk, cfg=cfg, ws=ws
        )
        dlog_chunk.add_(d_log_per_chunk.unsqueeze(-1).to(dlog_chunk.dtype))

        grad_s_md = dS_local_end.reshape(BH, NC, M, D)
        t_b1 = _time_ms(
            lambda: s3._ssd_rank3_chunk_end_state_backward_impl_static(
                grad_s_md,
                W1c,
                V1c,
                W2c,
                V2c,
                W3c,
                V3c,
                logc,
                dW1,
                dV1,
                dW2,
                dV2,
                dW3,
                dV3,
                dlog_chunk,
                cfg=cfg,
                has_rank2=has_rank2,
                has_rank3=has_rank3,
            )
        )

        if i >= warmup:
            fwd_p1.append(t_f1)
            fwd_p2.append(t_f2)
            fwd_p3.append(t_f3)
            bwd_p3.append(t_b3)
            bwd_p2.append(t_b2)
            bwd_p1.append(t_b1)

    return _summarize_phase_lists(fwd_p1, fwd_p2, fwd_p3, bwd_p3, bwd_p2, bwd_p1)


def _phase_timing_rank4_backend(
    *,
    rank: int,
    B: int,
    H: int,
    N: int,
    M: int,
    D: int,
    chunk_size: int,
    input_precision: str,
    warmup: int,
    iters: int,
    log_alpha_mode: str,
) -> dict[str, float]:
    if rank < 1 or rank > 4:
        raise ValueError(f"ssd_rank4 backend only supports rank in [1,4], got {rank}.")

    s4._ensure_triton_allocator()
    s4.set_ssd_rank1_static_shape(N=N, M=M, D=D)
    device = torch.device("cuda")
    C, W1, V1, log_alpha, W2, V2, W3, V3, W4, V4 = _make_inputs(
        B=B,
        H=H,
        N=N,
        M=M,
        D=D,
        rank=4,
        dtype=torch.bfloat16,
        device=device,
        log_alpha_mode=log_alpha_mode,
    )
    if rank < 2:
        W2, V2 = W1.detach(), V1.detach()
    if rank < 3:
        W3, V3 = W1.detach(), V1.detach()
    if rank < 4:
        W4, V4 = W1.detach(), V1.detach()

    cfg = s4._validate_static_hot_path_contract(C, W1, V1, log_alpha, None, chunk_size, input_precision, True)
    Cc, W1c, V1c, W2c, V2c, W3c, V3c, W4c, V4c, logc, _, _, _, _, _, NC, BH = s4._ssd_rank4_prepare_unchunked_inputs_static(
        C, W1, V1, W2, V2, W3, V3, W4, V4, log_alpha, cfg=cfg
    )
    ws = s4._get_static_workspace(device=device, cfg_key=(BH, N, M, D), cfg=cfg, allocate_phase3_s0=False)

    fwd_p1, fwd_p2, fwd_p3 = [], [], []
    bwd_p3, bwd_p2, bwd_p1 = [], [], []

    for i in range(warmup + iters):
        t_f1 = _time_ms(
            lambda: s4._ssd_rank4_chunk_end_state_forward_impl_static(
                W1c, V1c, W2c, V2c, W3c, V3c, W4c, V4c, logc, cfg=cfg, ws=ws, rank=rank
            )
        )

        s_local = ws.s_local_end_md.reshape(BH, NC, M * D)
        log_per_chunk = logc.sum(dim=2, dtype=torch.float32).contiguous()
        final_state = torch.empty((BH, M * D), device=device, dtype=torch.float32)
        t_f2 = _time_ms(lambda: s4._ssd_rank1_phase2_forward_static(s_local, log_per_chunk, final_state, cfg=cfg, ws=ws))

        s0_chunk = ws.phase2_chunk_start
        t_f3 = _time_ms(
            lambda: s4._ssd_rank4_dense_output_forward_impl_static(
                Cc, W1c, V1c, W2c, V2c, W3c, V3c, W4c, V4c, logc, s0_chunk, cfg=cfg, ws=ws, rank=rank
            )
        )

        y_chunk = s4._ssd_rank4_dense_output_forward_impl_static(
            Cc, W1c, V1c, W2c, V2c, W3c, V3c, W4c, V4c, logc, s0_chunk, cfg=cfg, ws=ws, rank=rank
        )
        grad_y = torch.randn_like(y_chunk)

        t_b3 = _time_ms(
            lambda: s4._ssd_rank4_dense_output_backward_impl_static(
                Cc, W1c, V1c, W2c, V2c, W3c, V3c, W4c, V4c, logc, grad_y, s0_chunk, cfg=cfg, ws=ws, rank=rank
            )
        )
        dC, dW1, dV1, dW2, dV2, dW3, dV3, dW4, dV4, dlog_phase3, dS0 = s4._ssd_rank4_dense_output_backward_impl_static(
            Cc, W1c, V1c, W2c, V2c, W3c, V3c, W4c, V4c, logc, grad_y, s0_chunk, cfg=cfg, ws=ws, rank=rank
        )
        del dC
        dlog_chunk = ws.dlog_chunk_accum
        dlog_chunk.copy_(dlog_phase3)

        grad_final = ws.phase2_grad_final_zero
        grad_final.zero_()
        t_b2 = _time_ms(lambda: s4._ssd_rank1_phase2_backward_static(dS0, grad_final, s0_chunk, log_per_chunk, cfg=cfg, ws=ws))
        dS_local_end, d_log_per_chunk, _ = s4._ssd_rank1_phase2_backward_static(
            dS0, grad_final, s0_chunk, log_per_chunk, cfg=cfg, ws=ws
        )
        dlog_chunk.add_(d_log_per_chunk.unsqueeze(-1).to(dlog_chunk.dtype))

        grad_s_md = dS_local_end.reshape(BH, NC, M, D)
        t_b1 = _time_ms(
            lambda: s4._ssd_rank4_chunk_end_state_backward_impl_static(
                grad_s_md,
                W1c,
                V1c,
                W2c,
                V2c,
                W3c,
                V3c,
                W4c,
                V4c,
                logc,
                dW1,
                dV1,
                dW2,
                dV2,
                dW3,
                dV3,
                dW4,
                dV4,
                dlog_chunk,
                cfg=cfg,
                rank=rank,
            )
        )

        if i >= warmup:
            fwd_p1.append(t_f1)
            fwd_p2.append(t_f2)
            fwd_p3.append(t_f3)
            bwd_p3.append(t_b3)
            bwd_p2.append(t_b2)
            bwd_p1.append(t_b1)

    return _summarize_phase_lists(fwd_p1, fwd_p2, fwd_p3, bwd_p3, bwd_p2, bwd_p1)


def _print_table(title: str, rows: dict[int, dict[str, float]]) -> None:
    print(title)
    header = "| rank | " + " | ".join(TIMING_KEYS) + " |"
    sep = "|---|" + "|".join(["---"] * len(TIMING_KEYS)) + "|"
    print(header)
    print(sep)
    for rank in sorted(rows):
        vals = [f"{rows[rank][k]:.6f}" for k in TIMING_KEYS]
        print("| " + str(rank) + " | " + " | ".join(vals) + " |")
    print()


def _print_rank4_comment_deltas(rows: dict[int, dict[str, float]]) -> None:
    print("rank4 backend deltas vs rank4 comment snapshot (new - baseline)")
    header = "| rank | " + " | ".join(TIMING_KEYS) + " |"
    sep = "|---|" + "|".join(["---"] * len(TIMING_KEYS)) + "|"
    print(header)
    print(sep)
    for rank in sorted(rows):
        if rank not in RANK4_COMMENT_BASELINE:
            continue
        vals = [f"{(rows[rank][k] - RANK4_COMMENT_BASELINE[rank][k]):+.6f}" for k in TIMING_KEYS]
        print("| " + str(rank) + " | " + " | ".join(vals) + " |")
    print()


def _parse_rank_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Reusable SSD phase benchmark suite across rank1/rank3/rank4 Triton backends.")
    parser.add_argument("--backend", type=str, default="all", choices=["all", "rank1", "rank3", "rank4"])
    parser.add_argument("--rank1-ranks", type=str, default="1")
    parser.add_argument("--rank3-ranks", type=str, default="1,2,3")
    parser.add_argument("--rank4-ranks", type=str, default="1,2,3,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--input-precision", type=str, default="tf32")
    parser.add_argument("--timing-warmup", type=int, default=3)
    parser.add_argument("--timing-iters", type=int, default=8)
    parser.add_argument("--log-alpha-mode", type=str, default="softplus", choices=["softplus", "narrow", "mid"])
    parser.add_argument("--shape", type=int, nargs=5, default=[32, 32, 2048, 64, 64], metavar=("B", "H", "N", "M", "D"))
    parser.add_argument("--device", type=int, default=None, help="CUDA device index. Default: current device.")
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")
    if args.device is not None:
        torch.cuda.set_device(args.device)

    B, H, N, M, D = args.shape
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    output: dict[str, dict[int, dict[str, float]]] = {}

    if args.backend in ("all", "rank1"):
        rows: dict[int, dict[str, float]] = {}
        for rank in _parse_rank_list(args.rank1_ranks):
            if rank != 1:
                raise ValueError(f"rank1 backend only supports rank 1, got {rank}.")
            rows[rank] = _phase_timing_rank1_backend(
                B=B,
                H=H,
                N=N,
                M=M,
                D=D,
                chunk_size=args.chunk_size,
                input_precision=args.input_precision,
                warmup=args.timing_warmup,
                iters=args.timing_iters,
                log_alpha_mode=args.log_alpha_mode,
            )
        output["rank1"] = rows
        _print_table("Timing (ms) - ssd_rank1_triton backend", rows)

    if args.backend in ("all", "rank3"):
        rows = {}
        for rank in _parse_rank_list(args.rank3_ranks):
            rows[rank] = _phase_timing_rank3_backend(
                rank=rank,
                B=B,
                H=H,
                N=N,
                M=M,
                D=D,
                chunk_size=args.chunk_size,
                input_precision=args.input_precision,
                warmup=args.timing_warmup,
                iters=args.timing_iters,
                log_alpha_mode=args.log_alpha_mode,
            )
        output["rank3"] = rows
        _print_table("Timing (ms) - ssd_rank3_triton backend", rows)

    if args.backend in ("all", "rank4"):
        rows = {}
        for rank in _parse_rank_list(args.rank4_ranks):
            rows[rank] = _phase_timing_rank4_backend(
                rank=rank,
                B=B,
                H=H,
                N=N,
                M=M,
                D=D,
                chunk_size=args.chunk_size,
                input_precision=args.input_precision,
                warmup=args.timing_warmup,
                iters=args.timing_iters,
                log_alpha_mode=args.log_alpha_mode,
            )
        output["rank4"] = rows
        _print_table("Timing (ms) - ssd_rank4_triton backend", rows)
        _print_rank4_comment_deltas(rows)

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Wrote JSON timings to {path}")
    else:
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

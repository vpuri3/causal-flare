from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import causal_flare.autoregressive.ssd_rank4_triton as s4  # noqa: E402


TIMING_BASELINE = {
    1: {
        "fwd_p1": 0.740,
        "fwd_p2": 0.450,
        "fwd_p3": 0.867,
        "fwd_total": 2.057,
        "bwd_p3": 3.482,
        "bwd_p2": 0.619,
        "bwd_p1": 1.006,
        "bwd_total": 5.107,
        "step_total": 7.164,
    },
    2: {
        "fwd_p1": 1.149,
        "fwd_p2": 0.447,
        "fwd_p3": 1.148,
        "fwd_total": 2.744,
        "bwd_p3": 5.294,
        "bwd_p2": 0.616,
        "bwd_p1": 2.425,
        "bwd_total": 8.335,
        "step_total": 11.078,
    },
    3: {
        "fwd_p1": 1.700,
        "fwd_p2": 0.452,
        "fwd_p3": 1.753,
        "fwd_total": 3.905,
        "bwd_p3": 7.151,
        "bwd_p2": 0.616,
        "bwd_p1": 3.763,
        "bwd_total": 11.530,
        "step_total": 15.435,
    },
    4: {
        "fwd_p1": 2.145,
        "fwd_p2": 0.447,
        "fwd_p3": 2.932,
        "fwd_total": 5.525,
        "bwd_p3": 9.006,
        "bwd_p2": 0.616,
        "bwd_p1": 4.980,
        "bwd_total": 14.601,
        "step_total": 20.126,
    },
}

ACCURACY_BASELINE = {
    1: {
        "y_rel_l2": 0.00422655,
        "state_rel_l2": 0.00339675,
        "grad_rel_l2_global": 0.00445372,
        "dlog_rel_l2": 0.00606593,
    },
    2: {
        "y_rel_l2": 0.00457310,
        "state_rel_l2": 0.00373069,
        "grad_rel_l2_global": 0.00469502,
        "dlog_rel_l2": 0.00641442,
    },
    3: {
        "y_rel_l2": 0.00479475,
        "state_rel_l2": 0.00391656,
        "grad_rel_l2_global": 0.00483656,
        "dlog_rel_l2": 0.00671780,
    },
    4: {
        "y_rel_l2": 0.00497534,
        "state_rel_l2": 0.00417529,
        "grad_rel_l2_global": 0.00495182,
        "dlog_rel_l2": 0.00675443,
    },
}


def _rel_l2(ref: torch.Tensor, test: torch.Tensor, eps: float = 1e-12) -> float:
    num = torch.linalg.vector_norm(ref.float() - test.float())
    den = torch.linalg.vector_norm(ref.float()).clamp_min(eps)
    return float((num / den).item())


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


def _time_ms(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def _phase_timing_rank(
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
    s4.set_ssd_rank1_static_shape(N=N, M=M, D=D)
    device = torch.device("cuda")
    C, W1, V1, log_alpha, W2, V2, W3, V3, W4, V4 = _make_inputs(
        B=B,
        H=H,
        N=N,
        M=M,
        D=D,
        rank=rank,
        dtype=torch.bfloat16,
        device=device,
        log_alpha_mode=log_alpha_mode,
    )

    cfg = s4._validate_static_hot_path_contract(
        C,
        W1,
        V1,
        log_alpha,
        None,
        chunk_size,
        input_precision,
        True,
    )
    has_rank2 = rank >= 2
    has_rank3 = rank >= 3
    has_rank4 = rank >= 4
    if not has_rank2:
        W2, V2 = W1.detach(), V1.detach()
    if not has_rank3:
        W3, V3 = W1.detach(), V1.detach()
    if not has_rank4:
        W4, V4 = W1.detach(), V1.detach()

    Cc, W1c, V1c, W2c, V2c, W3c, V3c, W4c, V4c, logc, _, _, _, _, _, NC, BH = s4._ssd_rank4_prepare_unchunked_inputs_static(
        C,
        W1,
        V1,
        W2,
        V2,
        W3,
        V3,
        W4,
        V4,
        log_alpha,
        cfg=cfg,
    )
    ws = s4._get_static_workspace(device=device, cfg_key=(BH, N, M, D), cfg=cfg, allocate_phase3_s0=False)

    fwd_p1, fwd_p2, fwd_p3 = [], [], []
    bwd_p3, bwd_p2, bwd_p1 = [], [], []

    for i in range(warmup + iters):
        t_f1 = _time_ms(
            lambda: s4._ssd_rank4_chunk_end_state_forward_impl_static(
                W1c,
                V1c,
                W2c,
                V2c,
                W3c,
                V3c,
                W4c,
                V4c,
                logc,
                cfg=cfg,
                ws=ws,
                rank=rank,
            )
        )

        s_local = ws.s_local_end_md.reshape(BH, NC, M * D)
        log_per_chunk = logc.sum(dim=2, dtype=torch.float32).contiguous()
        final_state = torch.empty((BH, M * D), device=device, dtype=torch.float32)
        t_f2 = _time_ms(lambda: s4._ssd_rank1_phase2_forward_static(s_local, log_per_chunk, final_state, cfg=cfg, ws=ws))

        s0_chunk = ws.phase2_chunk_start
        t_f3 = _time_ms(
            lambda: s4._ssd_rank4_dense_output_forward_impl_static(
                Cc,
                W1c,
                V1c,
                W2c,
                V2c,
                W3c,
                V3c,
                W4c,
                V4c,
                logc,
                s0_chunk,
                cfg=cfg,
                ws=ws,
                rank=rank,
            )
        )

        y_chunk = s4._ssd_rank4_dense_output_forward_impl_static(
            Cc,
            W1c,
            V1c,
            W2c,
            V2c,
            W3c,
            V3c,
            W4c,
            V4c,
            logc,
            s0_chunk,
            cfg=cfg,
            ws=ws,
            rank=rank,
        )
        grad_y = torch.randn_like(y_chunk)

        t_b3 = _time_ms(
            lambda: s4._ssd_rank4_dense_output_backward_impl_static(
                Cc,
                W1c,
                V1c,
                W2c,
                V2c,
                W3c,
                V3c,
                W4c,
                V4c,
                logc,
                grad_y,
                s0_chunk,
                cfg=cfg,
                ws=ws,
                rank=rank,
            )
        )
        dC, dW1, dV1, dW2, dV2, dW3, dV3, dW4, dV4, dlog_phase3, dS0 = s4._ssd_rank4_dense_output_backward_impl_static(
            Cc,
            W1c,
            V1c,
            W2c,
            V2c,
            W3c,
            V3c,
            W4c,
            V4c,
            logc,
            grad_y,
            s0_chunk,
            cfg=cfg,
            ws=ws,
            rank=rank,
        )
        dlog_chunk = ws.dlog_chunk_accum
        dlog_chunk.copy_(dlog_phase3)

        grad_final = ws.phase2_grad_final_zero
        grad_final.zero_()
        t_b2 = _time_ms(lambda: s4._ssd_rank1_phase2_backward_static(dS0, grad_final, s0_chunk, log_per_chunk, cfg=cfg, ws=ws))
        dS_local_end, d_log_per_chunk, _ = s4._ssd_rank1_phase2_backward_static(
            dS0,
            grad_final,
            s0_chunk,
            log_per_chunk,
            cfg=cfg,
            ws=ws,
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

    def avg(xs: list[float]) -> float:
        return float(sum(xs) / len(xs))
    out = {
        "fwd_p1": avg(fwd_p1),
        "fwd_p2": avg(fwd_p2),
        "fwd_p3": avg(fwd_p3),
        "bwd_p3": avg(bwd_p3),
        "bwd_p2": avg(bwd_p2),
        "bwd_p1": avg(bwd_p1),
    }
    out["fwd_total"] = out["fwd_p1"] + out["fwd_p2"] + out["fwd_p3"]
    out["bwd_total"] = out["bwd_p3"] + out["bwd_p2"] + out["bwd_p1"]
    out["step_total"] = out["fwd_total"] + out["bwd_total"]
    return out


def _accuracy_rank(
    *,
    rank: int,
    B: int,
    H: int,
    N: int,
    M: int,
    D: int,
    chunk_size: int,
    input_precision: str,
    log_alpha_mode: str,
) -> dict[str, float]:
    s4.set_ssd_rank1_static_shape(N=N, M=M, D=D)
    device = torch.device("cuda")
    C, W1, V1, log_alpha, W2, V2, W3, V3, W4, V4 = _make_inputs(
        B=B,
        H=H,
        N=N,
        M=M,
        D=D,
        rank=rank,
        dtype=torch.bfloat16,
        device=device,
        log_alpha_mode=log_alpha_mode,
    )

    vars_ = [C, W1, V1]
    if rank >= 2:
        vars_.extend([W2, V2])
    if rank >= 3:
        vars_.extend([W3, V3])
    if rank >= 4:
        vars_.extend([W4, V4])
    vars_.append(log_alpha)
    for t in vars_:
        t.requires_grad_(True)

    C = vars_[0]
    W1 = vars_[1]
    V1 = vars_[2]
    idx = 3
    if rank >= 2:
        W2, V2 = vars_[idx], vars_[idx + 1]
        idx += 2
    else:
        W2, V2 = None, None
    if rank >= 3:
        W3, V3 = vars_[idx], vars_[idx + 1]
        idx += 2
    else:
        W3, V3 = None, None
    if rank >= 4:
        W4, V4 = vars_[idx], vars_[idx + 1]
        idx += 2
    else:
        W4, V4 = None, None
    log_alpha = vars_[idx]

    y_ref, s_ref = s4.ssd_rank4_token_loop_oracle(C, W1, V1, log_alpha, W2, V2, W3, V3, W4, V4)
    y_tri, s_tri = s4.ssd_rank4_triton(
        C,
        W1,
        V1,
        log_alpha,
        W2,
        V2,
        W3,
        V3,
        W4,
        V4,
        CHUNK_SIZE=chunk_size,
        INPUT_PRECISION=input_precision,
        RETURN_FINAL_STATE=True,
    )

    grad_y = torch.randn_like(y_ref)
    grad_s = torch.randn_like(s_ref)
    loss_ref = (y_ref * grad_y).sum() + (s_ref * grad_s).sum()
    loss_tri = (y_tri * grad_y).sum() + (s_tri * grad_s).sum()
    grads_ref = torch.autograd.grad(loss_ref, vars_, retain_graph=False)
    grads_tri = torch.autograd.grad(loss_tri, vars_, retain_graph=False)

    num = 0.0
    den = 0.0
    for g_ref, g_tri in zip(grads_ref[:-1], grads_tri[:-1]):
        g_ref_f = g_ref.float()
        g_tri_f = g_tri.float()
        num += float(torch.sum((g_ref_f - g_tri_f) ** 2).item())
        den += float(torch.sum(g_ref_f**2).item())

    return {
        "y_rel_l2": _rel_l2(y_ref, y_tri),
        "state_rel_l2": _rel_l2(s_ref, s_tri),
        "grad_rel_l2_global": math.sqrt(num / max(den, 1e-12)),
        "dlog_rel_l2": _rel_l2(grads_ref[-1], grads_tri[-1]),
    }


def _print_table(
    title: str,
    data: dict[int, dict[str, float]],
    baseline: dict[int, dict[str, float]],
) -> None:
    print(title)
    keys = list(next(iter(data.values())).keys())
    header = "| rank | " + " | ".join(keys) + " |"
    sep = "|---|" + "|".join(["---"] * len(keys)) + "|"
    print(header)
    print(sep)
    for rank in sorted(data):
        vals = [f"{data[rank][k]:.6f}" for k in keys]
        print("| " + str(rank) + " | " + " | ".join(vals) + " |")
    print()

    print(title + " deltas (new - baseline)")
    print(header)
    print(sep)
    for rank in sorted(data):
        vals = [f"{(data[rank][k] - baseline[rank][k]):+.6f}" for k in keys]
        print("| " + str(rank) + " | " + " | ".join(vals) + " |")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reusable rank-4 Triton benchmark/accuracy utility.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--input-precision", type=str, default="tf32")
    parser.add_argument("--timing-warmup", type=int, default=5)
    parser.add_argument("--timing-iters", type=int, default=20)
    parser.add_argument("--skip-timing", action="store_true")
    parser.add_argument("--skip-accuracy", action="store_true")
    parser.add_argument(
        "--timing-log-alpha-mode",
        type=str,
        default="softplus",
        choices=["softplus", "narrow", "mid"],
    )
    parser.add_argument(
        "--accuracy-log-alpha-mode",
        type=str,
        default="softplus",
        choices=["softplus", "narrow", "mid"],
    )
    parser.add_argument("--timing-shape", type=int, nargs=5, default=[32, 32, 2048, 64, 64], metavar=("B", "H", "N", "M", "D"))
    parser.add_argument("--accuracy-shape", type=int, nargs=5, default=[1, 8, 1024, 64, 64], metavar=("B", "H", "N", "M", "D"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    torch.manual_seed(args.seed)
    s4._ensure_triton_allocator()

    if not args.skip_timing:
        B, H, N, M, D = args.timing_shape
        timing = {
            rank: _phase_timing_rank(
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
                log_alpha_mode=args.timing_log_alpha_mode,
            )
            for rank in (1, 2, 3, 4)
        }
        _print_table("Timing (ms)", timing, TIMING_BASELINE)
        print("TIMING_JSON")
        print(json.dumps(timing, indent=2))
        print()

    if not args.skip_accuracy:
        B, H, N, M, D = args.accuracy_shape
        accuracy = {
            rank: _accuracy_rank(
                rank=rank,
                B=B,
                H=H,
                N=N,
                M=M,
                D=D,
                chunk_size=args.chunk_size,
                input_precision=args.input_precision,
                log_alpha_mode=args.accuracy_log_alpha_mode,
            )
            for rank in (1, 2, 3, 4)
        }
        _print_table("Accuracy (global rel L2)", accuracy, ACCURACY_BASELINE)
        print("ACCURACY_JSON")
        print(json.dumps(accuracy, indent=2))
        print()


if __name__ == "__main__":
    main()

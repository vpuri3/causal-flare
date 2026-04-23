from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import causal_flare.autoregressive.ssd_rank4_triton as s4  # noqa: E402


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


def _run_single_backward(
    *,
    rank: int,
    Cc: torch.Tensor,
    W1c: torch.Tensor,
    V1c: torch.Tensor,
    W2c: torch.Tensor,
    V2c: torch.Tensor,
    W3c: torch.Tensor,
    V3c: torch.Tensor,
    W4c: torch.Tensor,
    V4c: torch.Tensor,
    logc: torch.Tensor,
    cfg: s4._StaticSsdRank1ShapeConfig,
    ws: s4._StaticSsdRank1Workspace,
    BH: int,
    NC: int,
    M: int,
    D: int,
) -> None:
    s4._ssd_rank4_chunk_end_state_forward_impl_static(
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
    s_local = ws.s_local_end_md.reshape(BH, NC, M * D)
    log_per_chunk = logc.sum(dim=2, dtype=torch.float32).contiguous()
    final_state = torch.empty((BH, M * D), device=Cc.device, dtype=torch.float32)
    s4._ssd_rank1_phase2_forward_static(s_local, log_per_chunk, final_state, cfg=cfg, ws=ws)
    s0_chunk = ws.phase2_chunk_start
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
    s4._ssd_rank4_dense_output_backward_impl_static(
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Reusable rank-aware bwd_p3 profiling driver for ssd_rank4_triton.")
    parser.add_argument("--rank", type=int, default=4, choices=[1, 2, 3, 4])
    parser.add_argument("--shape", type=int, nargs=5, default=[32, 32, 2048, 64, 64], metavar=("B", "H", "N", "M", "D"))
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--input-precision", type=str, default="tf32")
    parser.add_argument("--log-alpha-mode", type=str, default="softplus", choices=["softplus", "narrow", "mid"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=2, help="Total backward passes; use >=2 to warm once then profile.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if args.iters < 1:
        raise ValueError("--iters must be >= 1.")

    B, H, N, M, D = args.shape
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    s4._ensure_triton_allocator()
    s4.set_ssd_rank1_static_shape(N=N, M=M, D=D)
    C, W1, V1, log_alpha, W2, V2, W3, V3, W4, V4 = _make_inputs(
        B=B,
        H=H,
        N=N,
        M=M,
        D=D,
        rank=args.rank,
        dtype=torch.bfloat16,
        device=device,
        log_alpha_mode=args.log_alpha_mode,
    )

    cfg = s4._validate_static_hot_path_contract(C, W1, V1, log_alpha, None, args.chunk_size, args.input_precision, True)
    has_rank2 = args.rank >= 2
    has_rank3 = args.rank >= 3
    has_rank4 = args.rank >= 4
    if not has_rank2:
        W2, V2 = W1.detach(), V1.detach()
    if not has_rank3:
        W3, V3 = W1.detach(), V1.detach()
    if not has_rank4:
        W4, V4 = W1.detach(), V1.detach()

    Cc, W1c, V1c, W2c, V2c, W3c, V3c, W4c, V4c, logc, _, _, _, _, _, NC, BH = s4._ssd_rank4_prepare_unchunked_inputs_static(
        C, W1, V1, W2, V2, W3, V3, W4, V4, log_alpha, cfg=cfg
    )
    ws = s4._get_static_workspace(device=device, cfg_key=(BH, N, M, D), cfg=cfg, allocate_phase3_s0=False)

    for _ in range(args.iters):
        _run_single_backward(
            rank=args.rank,
            Cc=Cc,
            W1c=W1c,
            V1c=V1c,
            W2c=W2c,
            V2c=V2c,
            W3c=W3c,
            V3c=V3c,
            W4c=W4c,
            V4c=V4c,
            logc=logc,
            cfg=cfg,
            ws=ws,
            BH=BH,
            NC=NC,
            M=M,
            D=D,
        )
    torch.cuda.synchronize()
    print(
        f"profile_bwd_p3_driver completed rank={args.rank} shape=({B},{H},{N},{M},{D}) "
        f"iters={args.iters} split_dlog_main=True"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Launch the semi-AR prepare kernel in a warmup-then-profile pattern."""

from __future__ import annotations

import argparse
import math

import torch

from causal_flare.semi_autoregressive.training import _get_semi_ar_forward_config, semi_ar_block_prepare_kernel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--N", type=int, default=65536)
    parser.add_argument("--D", type=int, default=32, help="Shared score/value head dimension for this driver.")
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-launches", type=int, default=1)
    parser.add_argument("--profile-launches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.device != "cuda":
        raise SystemExit("This profiling driver currently expects --device cuda.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available.")
    if args.profile_launches <= 0:
        raise SystemExit("--profile-launches must be > 0.")

    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(args.seed)

    scale = 1.0 / math.sqrt(args.D)
    device = torch.device(args.device)
    Q = torch.randn((args.H, args.M, args.D), device=device, dtype=dtype)
    K = torch.randn((args.B, args.N, args.H, args.D), device=device, dtype=dtype)
    V = torch.randn((args.B, args.N, args.H, args.D), device=device, dtype=dtype)

    config = _get_semi_ar_forward_config(
        M=args.M,
        N=args.N,
        D_score=args.D,
        D_value=args.D,
        block_size=args.block_size,
        chunk_size=args.chunk_size,
        weight_sharing_enc_dec=True,
        input_precision=None,
    )

    BH = args.B * args.H
    block_max = torch.empty((BH, config["NUM_BLOCKS"], args.M), device=device, dtype=torch.float32)
    block_den = torch.empty((BH, config["NUM_BLOCKS"], args.M), device=device, dtype=torch.float32)
    block_num = torch.empty((BH, config["NUM_BLOCKS"], args.M, args.D), device=device, dtype=dtype)

    def grid(meta):
        return (BH, config["NUM_BLOCKS"], config["NUM_M_TILES"] * config["NUM_D_VALUE_BLOCKS"])

    launcher = semi_ar_block_prepare_kernel[grid]

    def launch() -> None:
        launcher(
            K,
            Q,
            V,
            block_max,
            block_den,
            block_num,
            *K.stride(),
            *Q.stride(),
            *V.stride(),
            *block_max.stride(),
            *block_den.stride(),
            *block_num.stride(),
            BH,
            args.M,
            args.N,
            args.D,
            args.D,
            scale,
            config["BLOCK_SIZE"],
            CHUNK_SIZE=config["CHUNK_SIZE"],
            CHUNKS_PER_BLOCK=config["CHUNKS_PER_BLOCK"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["BLOCK_D"],
            BLOCK_K=config["BLOCK_K"],
            BLOCK_T_PREPARE=config["BLOCK_T_PREPARE"],
            INPUT_PRECISION=config["input_precision"],
            USE_BF16_NUM=dtype == torch.bfloat16,
            USE_FP16_NUM=dtype == torch.float16,
            H=args.H,
            num_warps=config["block_prepare_num_warps"],
            num_stages=config["block_prepare_num_stages"],
        )

    total_launches = args.warmup_launches + args.profile_launches
    for _ in range(total_launches):
        launch()
        torch.cuda.synchronize(device)

    print(
        "profile_semi_ar_prepare_driver complete "
        f"(warmup={args.warmup_launches}, profile={args.profile_launches}, "
        f"B={args.B}, H={args.H}, M={args.M}, N={args.N}, D={args.D}, "
        f"block_size={args.block_size}, chunk_size={args.chunk_size}, dtype={args.dtype})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

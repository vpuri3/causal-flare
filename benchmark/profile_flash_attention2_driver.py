#!/usr/bin/env python3
"""Launch the local Triton FA2 forward kernel in a warmup-then-profile pattern."""

from __future__ import annotations

import argparse
import math

import torch

from implementations.flash_attention2_triton import flash_attention2_triton_bnhd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--D", type=int, default=32)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup-launches", type=int, default=1)
    parser.add_argument("--profile-launches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warp-specialize", action="store_true")
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
    q = torch.randn((args.B, args.N, args.H, args.D), device=device, dtype=dtype)
    k = torch.randn((args.B, args.N, args.H, args.D), device=device, dtype=dtype)
    v = torch.randn((args.B, args.N, args.H, args.D), device=device, dtype=dtype)

    def launch() -> None:
        flash_attention2_triton_bnhd(
            q,
            k,
            v,
            causal=True,
            sm_scale=scale,
            warp_specialize=args.warp_specialize,
        )

    total_launches = args.warmup_launches + args.profile_launches
    for _ in range(total_launches):
        launch()
        torch.cuda.synchronize(device)

    print(
        "profile_flash_attention2_driver complete "
        f"(warmup={args.warmup_launches}, profile={args.profile_launches}, "
        f"B={args.B}, H={args.H}, N={args.N}, D={args.D}, dtype={args.dtype}, "
        f"warp_specialize={args.warp_specialize})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

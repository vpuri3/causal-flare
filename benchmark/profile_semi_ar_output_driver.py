#!/usr/bin/env python3
"""Launch the semi-AR output kernel in a warmup-then-profile pattern."""

from __future__ import annotations

import argparse
import math

import torch

from causal_flare.semi_autoregressive.training import (
    _ensure_triton_allocator,
    _get_semi_ar_forward_config,
    _semi_ar_use_output_host_descriptors,
    semi_ar_lse_output_desc_kernel,
    semi_ar_lse_output_separate_kernel,
    semi_ar_lse_output_shared_kernel,
)
from triton.tools.tensor_descriptor import TensorDescriptor


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
    parser.add_argument(
        "--separate-decoder",
        action="store_true",
        help="Profile the unshared decoder-path variant with explicit Q_dec/K_dec tensors.",
    )
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
    config = _get_semi_ar_forward_config(
        M=args.M,
        N=args.N,
        D_score=args.D,
        D_value=args.D,
        block_size=args.block_size,
        chunk_size=args.chunk_size,
        weight_sharing_enc_dec=not args.separate_decoder,
        input_precision=None,
    )

    BH = args.B * args.H
    q_dec = torch.randn((args.B, args.N, args.H, args.D), device=device, dtype=dtype) if args.separate_decoder else K
    k_dec = torch.randn((args.H, args.M, args.D), device=device, dtype=dtype) if args.separate_decoder else Q
    z_block = torch.randn((BH, config["NUM_BLOCKS"], args.M, args.D), device=device, dtype=dtype)
    lse_dec = torch.empty((BH, args.N), device=device, dtype=torch.float32)
    output = torch.empty((args.B, args.N, args.H, args.D), device=device, dtype=dtype)

    def grid(_meta):
        return (BH, config["NUM_GLOBAL_CHUNKS"], config["NUM_D_VALUE_BLOCKS"])

    def launch() -> None:
        use_desc = (
            _semi_ar_use_output_host_descriptors()
            and args.M % int(config["BLOCK_M_OUTPUT"]) == 0
            and args.D % int(config["BLOCK_K"]) == 0
            and args.D % int(config["BLOCK_D"]) == 0
        )
        if use_desc:
            _ensure_triton_allocator()
            q_view = (q_dec if args.separate_decoder else K).permute(0, 2, 1, 3)
            k_view = k_dec if args.separate_decoder else Q
            o_view = output.permute(0, 2, 1, 3)
            q_desc = TensorDescriptor(
                q_view,
                shape=list(q_view.shape),
                strides=list(q_view.stride()),
                block_shape=[1, 1, config["BLOCK_T"], config["BLOCK_K"]],
            )
            k_desc = TensorDescriptor(
                k_view,
                shape=list(k_view.shape),
                strides=list(k_view.stride()),
                block_shape=[1, config["BLOCK_M_OUTPUT"], config["BLOCK_K"]],
            )
            z_desc = TensorDescriptor(
                z_block,
                shape=list(z_block.shape),
                strides=list(z_block.stride()),
                block_shape=[1, 1, config["BLOCK_M_OUTPUT"], config["BLOCK_D"]],
            )
            o_desc = TensorDescriptor(
                o_view,
                shape=list(o_view.shape),
                strides=list(o_view.stride()),
                block_shape=[1, 1, config["BLOCK_T"], config["BLOCK_D"]],
            )
            semi_ar_lse_output_desc_kernel[grid](
                q_desc,
                k_desc,
                z_desc,
                lse_dec,
                o_desc,
                *lse_dec.stride(),
                BH,
                args.M,
                args.N,
                args.D,
                args.D,
                scale,
                config["CHUNKS_PER_BLOCK"],
                CHUNK_SIZE=config["CHUNK_SIZE"],
                BLOCK_D=config["BLOCK_D"],
                BLOCK_K=config["BLOCK_K"],
                BLOCK_T=config["BLOCK_T"],
                BLOCK_M_OUTPUT=config["BLOCK_M_OUTPUT"],
                INPUT_PRECISION=config["input_precision"],
                USE_BF16_OUTPUT=dtype == torch.bfloat16,
                USE_FP16_OUTPUT=dtype == torch.float16,
                H=args.H,
                num_warps=config["lse_output_num_warps"],
                num_stages=config["lse_output_num_stages"],
            )
        elif args.separate_decoder:
            semi_ar_lse_output_separate_kernel[grid](
                q_dec,
                k_dec,
                z_block,
                lse_dec,
                output,
                *q_dec.stride(),
                *k_dec.stride(),
                *z_block.stride(),
                *lse_dec.stride(),
                *output.stride(),
                BH,
                args.M,
                args.N,
                args.D,
                args.D,
                scale,
                config["CHUNKS_PER_BLOCK"],
                CHUNK_SIZE=config["CHUNK_SIZE"],
                BLOCK_M=config["BLOCK_M"],
                BLOCK_M_OUTPUT=config["BLOCK_M_OUTPUT"],
                BLOCK_D=config["BLOCK_D"],
                BLOCK_K=config["BLOCK_K"],
                BLOCK_T=config["BLOCK_T"],
                INPUT_PRECISION=config["input_precision"],
                H=args.H,
                num_warps=config["lse_output_num_warps"],
                num_stages=config["lse_output_num_stages"],
            )
        else:
            semi_ar_lse_output_shared_kernel[grid](
                K,
                Q,
                z_block,
                lse_dec,
                output,
                *K.stride(),
                *Q.stride(),
                *z_block.stride(),
                *lse_dec.stride(),
                *output.stride(),
                BH,
                args.M,
                args.N,
                args.D,
                args.D,
                scale,
                config["CHUNKS_PER_BLOCK"],
                CHUNK_SIZE=config["CHUNK_SIZE"],
                BLOCK_M=config["BLOCK_M"],
                BLOCK_M_OUTPUT=config["BLOCK_M_OUTPUT"],
                BLOCK_D=config["BLOCK_D"],
                BLOCK_K=config["BLOCK_K"],
                BLOCK_T=config["BLOCK_T"],
                INPUT_PRECISION=config["input_precision"],
                H=args.H,
                num_warps=config["lse_output_num_warps"],
                num_stages=config["lse_output_num_stages"],
            )

    total_launches = args.warmup_launches + args.profile_launches
    for _ in range(total_launches):
        launch()
        torch.cuda.synchronize(device)

    print(
        "profile_semi_ar_output_driver complete "
        f"(warmup={args.warmup_launches}, profile={args.profile_launches}, "
        f"B={args.B}, H={args.H}, M={args.M}, N={args.N}, D={args.D}, "
        f"block_size={args.block_size}, chunk_size={args.chunk_size}, dtype={args.dtype}, "
        f"decoder_variant={'separate' if args.separate_decoder else 'shared'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

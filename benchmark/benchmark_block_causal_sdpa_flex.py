import argparse

import torch

from causal_flare.block_causal import benchmark_block_causal_sdpa_flex


def _parse_args():
    parser = argparse.ArgumentParser(description="Benchmark vanilla block-causal SDPA using FlexAttention masking.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-tokens", type=int, default=65536)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--compare-reference", action="store_true")
    return parser.parse_args()


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def main():
    args = _parse_args()
    device = torch.device(args.device)
    dtype = _resolve_dtype(args.dtype)

    torch.manual_seed(0)
    q = torch.randn((args.batch_size, args.num_tokens, args.num_heads, args.head_dim), device=device, dtype=dtype)
    k = torch.randn((args.batch_size, args.num_tokens, args.num_heads, args.head_dim), device=device, dtype=dtype)
    v = torch.randn((args.batch_size, args.num_tokens, args.num_heads, args.head_dim), device=device, dtype=dtype)

    results = benchmark_block_causal_sdpa_flex(
        q,
        k,
        v,
        block_size=args.block_size,
        warmup=args.warmup,
        iters=args.iters,
        compare_reference=args.compare_reference,
    )

    print("Block-causal SDPA Flex benchmark")
    print(f"shape: B={args.batch_size} N={args.num_tokens} H={args.num_heads} D={args.head_dim}")
    print(f"config: block_size={args.block_size} dtype={args.dtype} device={args.device}")
    print(f"impl_ms: {results['impl_ms']:.3f}")
    if results["reference_ms"] is not None:
        print(f"reference_ms: {results['reference_ms']:.3f}")
        print(f"reference_max_abs: {results['reference_max_abs']:.6g}")
        print(f"reference_mean_abs: {results['reference_mean_abs']:.6g}")


if __name__ == "__main__":
    main()

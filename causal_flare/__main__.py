import argparse


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m causal_flare",
        description=(
            "Run FLARE diagnostics + benchmark comparisons on CUDA.\n"
            "By default this runs the autoregressive diagnostics matrix. With --semi_ar\n"
            "it runs the semi-autoregressive/block-causal comparison flow instead."
        ),
        epilog=(
            "Examples:\n"
            "  python -m causal_flare --B 1 --H 8 --M 16 --N 32 --D 16 --dtype bfloat16\n"
            "  python -m causal_flare --semi_ar --block-size 256 --chunk-size 128\n"
            "  FLARE_COMPILE_TIMINGS=0 python -m causal_flare --N 512 --dtype float16\n"
            "\n"
            "Useful env vars:\n"
            "  FLARE_COMPILE_TIMINGS=0|1   Toggle compile timing probes in output\n"
            "  FLARE_PYTORCH_CHUNK_SIZE    Override PyTorch chunk size baseline\n"
            "  FLARE_REFERENCE_FP32=0|1    Toggle FP32 reference math mode"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--semi_ar", action="store_true", help="Run the semi-autoregressive/block-causal comparison flow.")
    parser.add_argument("--B", type=int, default=None, help="Batch size (number of sequences).")
    parser.add_argument("--H", type=int, default=None, help="Number of attention heads.")
    parser.add_argument("--M", type=int, default=None, help="Latent/query length per head.")
    parser.add_argument("--N", type=int, default=None, help="Context/sequence length.")
    parser.add_argument(
        "--D",
        type=int,
        default=None,
        help="Score head dimension D_k. This diagnostics CLI still assumes value_head_dim == score_head_dim.",
    )
    parser.add_argument("--block-size", type=int, default=None, help="Semi-autoregressive block size.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Semi-autoregressive chunk size.")
    parser.add_argument(
        "--dtype",
        default=None,
        help="Torch dtype name used for tensor initialization (e.g. bfloat16, float16, float32).",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    optimize_for_h100()
    if args.semi_ar:
        print("Selected mode: semi_autoregressive")
        run_semi_autoregressive_main(
            B=4 if args.B is None else args.B,
            H=16 if args.H is None else args.H,
            M=64 if args.M is None else args.M,
            N=65536 if args.N is None else args.N,
            D=32 if args.D is None else args.D,
            dtype="bfloat16" if args.dtype is None else args.dtype,
            block_size=256 if args.block_size is None else args.block_size,
            chunk_size=128 if args.chunk_size is None else args.chunk_size,
        )
        return

    print("Selected mode: autoregressive")
    run_module_main(
        B=8 if args.B is None else args.B,
        H=16 if args.H is None else args.H,
        M=64 if args.M is None else args.M,
        N=2048 if args.N is None else args.N,
        D=32 if args.D is None else args.D,
        dtype="bfloat16" if args.dtype is None else args.dtype,
    )


def optimize_for_h100() -> None:
    """Apply H100-specific optimizations for maximum performance"""

    import os
    import torch

    # Environment variables for H100 optimization
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True,roundup_power2_divisions:16"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "0"  # Async execution
    os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "1"  # Enable cuDNN v8 API
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # H100-specific PyTorch backend optimizations
    # Enable TF32 tensor cores: FP32 operations use TF32 (10-bit mantissa, 8-bit exponent)
    # This provides ~6x better accuracy than BF16 with only ~1% performance cost
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # FP8 support (only available in newer PyTorch versions)
    try:
        torch.backends.cuda.matmul.allow_fp8_e4m3fn = True
        torch.backends.cuda.matmul.allow_fp8_e5m2 = True
    except AttributeError:
        pass

    # Performance vs reproducibility settings (choose one)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Memory management optimizations
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)
        torch.cuda.empty_cache()

    print("H100 optimizations applied successfully!")

def run_semi_autoregressive_main(
    B: int = 4,
    H: int = 16,
    M: int = 256,
    N: int = 65536,
    D: int = 32,
    dtype: str = "bfloat16",
    block_size: int = 256,
    chunk_size: int = 128,
):
    import math
    import os
    import time

    import torch
    import torch.nn.functional as F
    import triton
    from flash_attn import flash_attn_func
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from causal_flare.semi_autoregressive import flare_semi_autoregressive_trition
    from causal_flare.semi_autoregressive.reference import (
        _block_causal_forward_pytorch_block_stats,
        _block_causal_forward_pytorch_chunk_stats,
        semi_autoregressive_flare_reference,
    )
    from testing.suite_runners.common import compute_errors, measure_memory

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_obj = getattr(torch, dtype)
    scale = D ** -0.5

    torch.manual_seed(0)
    Q_flare = torch.randn((H, M, D), device=device, dtype=dtype_obj)
    K_flare = torch.randn((B, N, H, D), device=device, dtype=dtype_obj)
    V_flare = torch.randn((B, N, H, D), device=device, dtype=dtype_obj)
    Q_sdpa = torch.randn((B, N, H, D), device=device, dtype=dtype_obj)
    K_sdpa = torch.randn((B, N, H, D), device=device, dtype=dtype_obj)
    V_sdpa = torch.randn((B, N, H, D), device=device, dtype=dtype_obj)

    compile_timing_enabled = os.environ.get("FLARE_COMPILE_TIMINGS", "1") == "1"

    def _compile_suffix(ms: float) -> str:
        if math.isnan(ms):
            return ""
        return f" (compile={ms:.2f} ms)"

    def _probe_compile(run_fn) -> float:
        if not compile_timing_enabled:
            return float("nan")
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    def _format_err(errors: dict, prefix: str, kind: str) -> str:
        mean_key = f"{prefix}_mean_{kind}_err"
        max_key = f"{prefix}_max_{kind}_err"
        if errors and mean_key in errors and max_key in errors:
            return f"{errors[mean_key]:.2e}/{errors[max_key]:.2e}"
        return "N/A"

    def _run_impl(label: str, fn, *, ref=None, ref_prefix: str | None = None, skip_reason: str | None = None):
        if skip_reason is not None:
            return {
                "label": label,
                "ms": float("nan"),
                "mem": 0.0,
                "errors": {},
                "status": f"Skipped: {skip_reason}",
                "compile_ms": float("nan"),
            }

        try:
            compile_ms = _probe_compile(fn)
            y, mem = measure_memory(fn)
            if device.type == "cuda":
                ms = triton.testing.do_bench(fn, warmup=2, rep=2)
            else:
                start = time.perf_counter()
                for _ in range(2):
                    fn()
                ms = (time.perf_counter() - start) * 1e3 / 2
            errors = compute_errors(y, ref, ref_prefix) if ref is not None and ref_prefix is not None else {}
            return {
                "label": label,
                "ms": ms,
                "mem": mem,
                "errors": errors,
                "status": "OK",
                "compile_ms": compile_ms,
            }
        except Exception as exc:
            return {
                "label": label,
                "ms": float("nan"),
                "mem": 0.0,
                "errors": {},
                "status": f"Failed: {exc}",
                "compile_ms": float("nan"),
            }

    def _run_sdpa(q, k, v, *, is_causal: bool):
        q_in = q.permute(0, 2, 1, 3)
        k_in = k.permute(0, 2, 1, 3)
        v_in = v.permute(0, 2, 1, 3)
        return F.scaled_dot_product_attention(
            q_in,
            k_in,
            v_in,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=scale,
        ).permute(0, 2, 1, 3).to(v.dtype)

    def _run_sdpa_fa2(q, k, v, *, is_causal: bool):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return _run_sdpa(q, k, v, is_causal=is_causal)

    def _run_flash_attn(q, k, v, *, is_causal: bool):
        return flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=scale, causal=is_causal)

    print(
        f"Testing Semi-Autoregressive FLARE Forward Pass "
        f"(B={B}, H={H}, M={M}, N={N}, D={D}, block_size={block_size}, chunk_size={chunk_size}, dtype={dtype_obj}, device={device})"
    )

    print("Measuring semi-AR FLARE reference...", end=" ", flush=True)
    flare_ref_result = _run_impl(
        "FLARE Semi AR Reference",
        lambda: semi_autoregressive_flare_reference(Q_flare, K_flare, V_flare, block_size=block_size, scale=scale),
    )
    print(flare_ref_result["status"] + _compile_suffix(flare_ref_result["compile_ms"]))

    flare_ref_output = None
    if flare_ref_result["status"] == "OK":
        flare_ref_output = semi_autoregressive_flare_reference(Q_flare, K_flare, V_flare, block_size=block_size, scale=scale)

    print("Measuring semi-AR chunk-stats PyTorch reference...", end=" ", flush=True)
    flare_chunk_stats_result = _run_impl(
        "FLARE chunk-stats",
        lambda: _block_causal_forward_pytorch_chunk_stats(
            Q_flare, K_flare, V_flare, block_size=block_size, chunk_size=chunk_size, scale=scale
        ),
        ref=flare_ref_output,
        ref_prefix="semi_ar_chunk_stats",
    )
    print(flare_chunk_stats_result["status"] + _compile_suffix(flare_chunk_stats_result["compile_ms"]))

    print("Measuring semi-AR block-stats PyTorch reference...", end=" ", flush=True)
    flare_block_stats_result = _run_impl(
        "FLARE block-stats",
        lambda: _block_causal_forward_pytorch_block_stats(
            Q_flare, K_flare, V_flare, block_size=block_size, chunk_size=chunk_size, scale=scale
        ),
        ref=flare_ref_output,
        ref_prefix="semi_ar_block_stats",
    )
    print(flare_block_stats_result["status"] + _compile_suffix(flare_block_stats_result["compile_ms"]))

    triton_skip_reason = None if device.type == "cuda" else "requires CUDA"

    print("Measuring SemiAutoRegressiveFLARE Triton...", end=" ", flush=True)
    semi_ar_triton_result = _run_impl(
        "SemiAutoRegressiveFLARE Triton",
        lambda: flare_semi_autoregressive_trition(
            Q_flare,
            K_flare,
            V_flare,
            block_size=block_size,
            chunk_size=chunk_size,
            scale=scale,
        ),
        ref=flare_ref_output,
        ref_prefix="semi_ar_triton",
        skip_reason=triton_skip_reason,
    )
    print(semi_ar_triton_result["status"] + _compile_suffix(semi_ar_triton_result["compile_ms"]))

    print("Measuring causal SDPA via flash_attn...", end=" ", flush=True)
    sdpa_causal_result = _run_impl(
        "Causal SDPA (flash_attn)",
        lambda: _run_flash_attn(Q_sdpa, K_sdpa, V_sdpa, is_causal=True),
        skip_reason=triton_skip_reason,
    )
    print(sdpa_causal_result["status"] + _compile_suffix(sdpa_causal_result["compile_ms"]))

    print("Measuring causal SDPA FA2...", end=" ", flush=True)
    sdpa_causal_fa2_result = _run_impl(
        "Causal SDPA FA2",
        lambda: _run_sdpa_fa2(Q_sdpa, K_sdpa, V_sdpa, is_causal=True),
        skip_reason=triton_skip_reason,
    )
    print(sdpa_causal_fa2_result["status"] + _compile_suffix(sdpa_causal_fa2_result["compile_ms"]))

    triton_profile = None
    if device.type == "cuda" and semi_ar_triton_result["status"] == "OK":
        _, triton_profile = flare_semi_autoregressive_trition(
            Q_flare,
            K_flare,
            V_flare,
            block_size=block_size,
            chunk_size=chunk_size,
            scale=scale,
            profile=True,
        )

    print("\n" + "=" * 136)
    print(f"{'Implementation':<42} {'Time (ms)':<12} {'Memory (GB)':<15} {'Abs Err (mean/max)':<22} {'Rel Err (mean/max)':<22} {'Status'}")
    print("-" * 136)
    rows = [
        (flare_ref_result, None),
        (flare_chunk_stats_result, "semi_ar_chunk_stats"),
        (flare_block_stats_result, "semi_ar_block_stats"),
        (semi_ar_triton_result, "semi_ar_triton"),
        (sdpa_causal_result, None),
        (sdpa_causal_fa2_result, None),
    ]
    for row, prefix in rows:
        ms_str = f"{row['ms']:.2f}" if not math.isnan(row["ms"]) else "N/A"
        abs_err_str = _format_err(row["errors"], prefix, "abs") if prefix is not None else "N/A"
        rel_err_str = _format_err(row["errors"], prefix, "rel") if prefix is not None else "N/A"
        print(
            f"{row['label']:<42} {ms_str:<12} {row['mem']:<15.2e} "
            f"{abs_err_str:<22} {rel_err_str:<22} "
            f"{row['status']}"
        )

    if triton_profile is not None:
        forward = triton_profile.get("forward", {})
        kernel_names = sorted(forward.keys())
        print("\n" + "=" * 96)
        print(f"{'SemiAutoRegressiveFLARE Forward Kernel':<44} {'Time (ms)':<20}")
        print("-" * 96)
        for kernel_name in kernel_names:
            kernel_ms = forward.get(kernel_name)
            kernel_ms_str = f"{kernel_ms:.3f}" if kernel_ms is not None else "N/A"
            print(f"{kernel_name:<44} {kernel_ms_str:<20}")

    return

def run_module_main(B: int = 1, H: int = 8, M: int = 128, N: int = 2048, D: int = 16, dtype: str = "bfloat16"):
    import math
    import os
    import time
    import warnings

    import torch
    import triton
    from testing.suite_runners.common import (
        DenseFLARE,
        DenseFLARE1,
        RecurrentFLARE,
        _BWD_PROFILE_TIMINGS,
        _denseflare1_phase_bench,
        _get_eps_for_dtype,
        _set_bwd_profile_mode,
        _temp_env_var,
        causal_SDPA,
        compute_errors,
        flare_causal_chunked,
        flare_causal_perciever_ar,
        flare_causal_pytorch_dense,
        flare_causal_reference,
        flare_autoregressive_triton,
        flare_recurrent_pytorch,
        measure_memory,
    )

    device = torch.device('cuda')
    dtype = getattr(torch, dtype)
    try:
        from benchmark.implementations.flash_attention2_triton import (
            flash_attention2_triton_bnhd as _flash_attention2_triton_bnhd,
        )
    except Exception as exc:
        _flash_attention2_triton_bnhd = None
        print(f"[FLARE DEBUG] FlashAttention2 Triton baseline unavailable: {exc}")

    scale = (D ** -0.5)

    torch.manual_seed(0)
    Q = torch.randn(H, M, D, device=device, dtype=dtype)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype)

    chunk_env = os.environ.get("FLARE_PYTORCH_CHUNK_SIZE", "")
    if chunk_env:
        chunk_size = int(chunk_env)
    else:
        chunk_size = None

    compile_timing_enabled = os.environ.get("FLARE_COMPILE_TIMINGS", "1") == "1"

    def _compile_suffix(ms: float) -> str:
        if math.isnan(ms):
            return ""
        return f" (compile={ms:.2f} ms)"

    def _probe_compile(run_fn) -> float:
        if not compile_timing_enabled:
            return float("nan")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    #======================================================================#
    # Benchmark reference implementation
    #======================================================================#
    print("Measuring reference implementation...", end=" ", flush=True)

    ref_fp32_enabled = os.environ.get("FLARE_REFERENCE_FP32", "1") == "1"
    ref_bf16_enabled = os.environ.get("FLARE_REFERENCE_BF16", "0") == "1"
    if N > 4096:
        warnings.warn(f"Reference implementation skipped for N={N} > 4096 (too slow).")
        Y_reference = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
        Y_reference_bf16 = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
        ref_mem = 0.0
        ref_ms = 0.0
        ref_bf16_mem = 0.0
        ref_bf16_ms = 0.0
        ref_errors = dict()
        ref_bf16_errors = dict()
        print("Skipped (N > 4096)")
    else:
        ref_compile_ms = float("nan")
        ref_bf16_compile_ms = float("nan")
        # Memory measurement
        if ref_fp32_enabled:
            ref_compile_ms = _probe_compile(lambda: flare_causal_reference(Q, K, V, scale=scale))
            Y_reference, ref_mem = measure_memory(flare_causal_reference, Q, K, V, scale=scale)
            ref_ms = triton.testing.do_bench(lambda: flare_causal_reference(Q, K, V, scale=scale), warmup=2, rep=2)
        else:
            Y_reference = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
            ref_mem = 0.0
            ref_ms = 0.0

        # Timing using triton.testing.do_bench
        if ref_bf16_enabled:
            os.environ["FLARE_REFERENCE_FP32"] = "0"
            ref_bf16_compile_ms = _probe_compile(lambda: flare_causal_reference(Q, K, V, scale=scale))
            Y_reference_bf16, ref_bf16_mem = measure_memory(flare_causal_reference, Q, K, V, scale=scale)
            ref_bf16_ms = triton.testing.do_bench(
                lambda: flare_causal_reference(Q, K, V, scale=scale), warmup=2, rep=2
            )
            os.environ["FLARE_REFERENCE_FP32"] = "1" if ref_fp32_enabled else "0"
        else:
            Y_reference_bf16 = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
            ref_bf16_mem = 0.0
            ref_bf16_ms = 0.0

        ref_errors = dict()
        ref_bf16_errors = dict()

        extra = _compile_suffix(ref_compile_ms)
        if ref_bf16_enabled:
            extra += _compile_suffix(ref_bf16_compile_ms).replace("compile=", ", bf16_compile=")
        print(f"Done{extra}")

    #======================================================================#
    # Benchmark PyTorch implementation 2
    #======================================================================#
    print("Measuring PyTorch implementation 2...", end=" ", flush=True)

    try:
        pytorch2_compile_ms = _probe_compile(
            lambda: flare_causal_chunked(Q, K, V, scale=scale, chunk_size=chunk_size)
        )
        Y_pytorch2, pytorch2_mem = measure_memory(
            flare_causal_chunked, Q, K, V, scale=scale, chunk_size=chunk_size
        )
        pytorch2_ms = triton.testing.do_bench(
            lambda: flare_causal_chunked(Q, K, V, scale=scale, chunk_size=chunk_size),
            warmup=2,
            rep=2,
        )
        pytorch2_errors = compute_errors(Y_pytorch2, Y_reference, 'pytorch2') if N <= 4096 and ref_fp32_enabled else {}
    except Exception as exc:
        print(f"[FLARE DEBUG] PyTorch2 forward failed: {exc}")
        Y_pytorch2 = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
        pytorch2_mem = 0.0
        pytorch2_ms = float("nan")
        pytorch2_errors = {}
        pytorch2_compile_ms = float("nan")

    print(f"Done{_compile_suffix(pytorch2_compile_ms)}")

    #======================================================================#
    # Benchmark PCVR-AR-like impl implementation
    #======================================================================#
    print("Measuring PCVR-AR-like impl implementation...", end=" ", flush=True)
    try:
        perciever_imlp_compile_ms = _probe_compile(
            lambda: flare_causal_perciever_ar(Q, K, V, scale=scale)
        )
        Y_perciever_imlp, perciever_imlp_mem = measure_memory(
            flare_causal_perciever_ar, Q, K, V, scale=scale
        )
        perciever_imlp_ms = triton.testing.do_bench(
            lambda: flare_causal_perciever_ar(Q, K, V, scale=scale),
            warmup=2,
            rep=2,
        )
        perciever_imlp_errors = (
            compute_errors(Y_perciever_imlp, Y_reference, "perciever_imlp")
            if N <= 4096 and ref_fp32_enabled
            else {}
        )
    except Exception as exc:
        print(f"[FLARE DEBUG] PCVR-AR-like impl forward failed: {exc}")
        Y_perciever_imlp = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
        perciever_imlp_mem = 0.0
        perciever_imlp_ms = float("nan")
        perciever_imlp_errors = {}
        perciever_imlp_compile_ms = float("nan")
    print(f"Done{_compile_suffix(perciever_imlp_compile_ms)}")

    #======================================================================#
    # Benchmark PyTorch dense / DenseFLARE (small-N only)
    #======================================================================#
    run_dense = N <= 128
    if run_dense:
        print("Measuring PyTorch dense implementation...", end=" ", flush=True)
        pytorch_dense_compile_ms = _probe_compile(lambda: flare_causal_pytorch_dense(Q, K, V, scale=scale))

        # Memory measurement
        Y_pytorch_dense, pytorch_dense_mem = measure_memory(
            flare_causal_pytorch_dense, Q, K, V, scale=scale
        )

        # Timing using triton.testing.do_bench
        pytorch_dense_ms = triton.testing.do_bench(
            lambda: flare_causal_pytorch_dense(Q, K, V, scale=scale), warmup=2, rep=2
        )

        # Compute errors for PyTorch dense (against reference if available)
        pytorch_dense_errors = compute_errors(Y_pytorch_dense, Y_reference, 'pytorch_dense') if N <= 4096 and ref_fp32_enabled else {}

        print(f"Done{_compile_suffix(pytorch_dense_compile_ms)}")

        #======================================================================#
        # Benchmark DenseFLARE implementation
        #======================================================================#
        print("Measuring DenseFLARE implementation...", end=" ", flush=True)
        dense_triton_compile_ms = _probe_compile(lambda: DenseFLARE.apply(Q, K, V, scale))

        Y_dense_triton, dense_triton_mem = measure_memory(DenseFLARE.apply, Q, K, V, scale)
        dense_triton_ms = triton.testing.do_bench(lambda: DenseFLARE.apply(Q, K, V, scale), warmup=2, rep=2)

        dense_triton_errors = compute_errors(Y_dense_triton, Y_reference, 'dense_triton') if N <= 4096 and ref_fp32_enabled else {}

        print(f"Done{_compile_suffix(dense_triton_compile_ms)}")

        #======================================================================#
        # Benchmark DenseFLARE1 implementation
        #======================================================================#
        print("Measuring DenseFLARE1 implementation...", end=" ", flush=True)
        dense1_triton_compile_ms = _probe_compile(lambda: DenseFLARE1.apply(Q, K, V, scale))

        Y_dense1_triton, dense1_triton_mem = measure_memory(DenseFLARE1.apply, Q, K, V, scale)
        dense1_triton_ms = triton.testing.do_bench(lambda: DenseFLARE1.apply(Q, K, V, scale), warmup=2, rep=2)

        dense1_triton_errors = compute_errors(Y_dense1_triton, Y_reference, 'dense1_triton') if N <= 4096 and ref_fp32_enabled else {}

        print(f"Done{_compile_suffix(dense1_triton_compile_ms)}")

        dense1_phase1_ms, dense1_phase2_ms, dense1_phase3_ms = _denseflare1_phase_bench(Q, K, V, scale)
    else:
        Y_pytorch_dense = torch.full((B, N, H, D), float("nan"), device=device, dtype=dtype)
        Y_dense_triton = torch.full((B, N, H, D), float("nan"), device=device, dtype=dtype)
        pytorch_dense_mem = 0.0
        pytorch_dense_ms = 0.0
        dense_triton_mem = 0.0
        dense_triton_ms = 0.0
        pytorch_dense_errors = {}
        dense_triton_errors = {}
        Y_dense1_triton = torch.full((B, N, H, D), float("nan"), device=device, dtype=dtype)
        dense1_triton_mem = 0.0
        dense1_triton_ms = 0.0
        dense1_triton_errors = {}
        dense1_phase1_ms = 0.0
        dense1_phase2_ms = 0.0
        dense1_phase3_ms = 0.0
        print("Skipped PyTorch dense / DenseFLARE (N > 128)")

    if N <= 4096 and ref_fp32_enabled:
        # Reference vs reference (sanity)
        ref_errors = compute_errors(Y_reference, Y_reference, 'reference')
    if N <= 4096 and ref_bf16_enabled:
        ref_bf16_errors = compute_errors(Y_reference_bf16, Y_reference, 'reference_bf16')

    if os.environ.get("FLARE_STRICT_FP32_CHECK", "") == "1" and N <= 4096:
        with torch.no_grad():
            Q_fp32 = Q.float()
            K_fp32 = K.float()
            V_fp32 = V.float()
            Y_ref32 = flare_causal_reference(Q_fp32, K_fp32, V_fp32, scale=scale)
            Y_p232 = flare_causal_chunked(Q_fp32, K_fp32, V_fp32, scale=scale, chunk_size=chunk_size)
            err32 = compute_errors(Y_p232, Y_ref32, "pytorch2_fp32")
            print(
                f"[FLARE DEBUG] PyTorch2 vs Reference (FP32 inputs) "
                f"abs={err32['pytorch2_fp32_mean_abs_err']:.2e}/{err32['pytorch2_fp32_max_abs_err']:.2e} "
                f"rel={err32['pytorch2_fp32_mean_rel_err']:.2e}/{err32['pytorch2_fp32_max_rel_err']:.2e}"
            )

    if os.environ.get("FLARE_DEBUG_P2_REF", "") == "1" and N <= 4096 and ref_fp32_enabled and not torch.isnan(Y_reference).any():
        with torch.no_grad():
            err = (Y_pytorch2 - Y_reference).abs()  # [B,N,H,D]
            per_token_max = err.amax(dim=(0, 2, 3))  # [N]
            per_token_mean = err.mean(dim=(0, 2, 3))  # [N]
            topk = min(5, N)
            vals, idx = torch.topk(per_token_max, k=topk)
            print("[FLARE DEBUG] PyTorch2 vs Reference per-token max abs err:")
            for i in range(topk):
                t = int(idx[i].item())
                print(f"  t={t} max_abs={vals[i].item():.3e} mean_abs={per_token_mean[t].item():.3e}")
    if os.environ.get("FLARE_DEBUG_P2_FORMULA", "") == "1" and N <= 4096 and ref_fp32_enabled and not torch.isnan(Y_reference).any():
        with torch.no_grad():
            Qf = Q.float()
            Kf = K.float()
            Vf = V.float()
            max_tokens_env = os.environ.get("FLARE_DEBUG_P2_FORMULA_TOKENS", "")
            max_tokens = int(max_tokens_env) if max_tokens_env else min(8, N)
            use_stable = os.environ.get("FLARE_DEBUG_P2_FORMULA_STABLE", "1") == "1"
            eps_dbg = _get_eps_for_dtype(Q.dtype)
            print("[FLARE DEBUG] Formula vs Reference/PyTorch2 (per-token):")
            for t in range(max_tokens):
                Kt = Kf[:, : t + 1, :, :]  # [B,t+1,H,D]
                Vt = Vf[:, : t + 1, :, :]
                S_all = scale * torch.einsum("bthd,hmd->bthm", Kt, Qf)  # [B,t+1,H,M]
                S_t = S_all[:, -1, :, :]  # [B,H,M]
                if use_stable:
                    m_u = S_all.max(dim=1).values  # [B,H,M]
                    exp_u = torch.exp(S_all - m_u.unsqueeze(1))  # [B,t+1,H,M]
                    den_total = exp_u.sum(dim=1) * torch.exp(m_u)  # [B,H,M]
                    sum_exp_v = torch.einsum("bthm,bthd->bhmd", exp_u, Vt) * torch.exp(m_u).unsqueeze(-1)
                else:
                    exp_u = torch.exp(S_all)
                    den_total = exp_u.sum(dim=1)
                    sum_exp_v = torch.einsum("bthm,bthd->bhmd", exp_u, Vt)
                P_t = torch.softmax(S_t, dim=-1)
                expA_t = P_t / (den_total + eps_dbg)
                y_formula = torch.einsum("bhm,bhmd->bhd", expA_t, sum_exp_v)
                y_ref = Y_reference[:, t, :, :]
                y_p2 = Y_pytorch2[:, t, :, :]
                err_ref = (y_formula - y_ref).abs().amax().item()
                err_p2 = (y_formula - y_p2).abs().amax().item()
                print(f"  t={t} max_abs_formula_vs_ref={err_ref:.3e} max_abs_formula_vs_p2={err_p2:.3e}")

    #======================================================================#
    # Benchmark AutoRegressiveFLARE implementation (input_precision modes)
    #======================================================================#
    def _normalize_chunked_forward_timings(raw_timings):
        if not isinstance(raw_timings, dict):
            return {}

        # New profiler format: {"forward": {...}, "backward": {...}, "..._total_ms": ...}
        forward_timings = raw_timings.get("forward")
        if isinstance(forward_timings, dict):
            normalized = {
                k: float(v)
                for k, v in forward_timings.items()
                if isinstance(v, (int, float))
            }
            forward_total = raw_timings.get("forward_total_ms")
            if isinstance(forward_total, (int, float)):
                normalized["Total"] = float(forward_total)
            elif normalized:
                normalized["Total"] = sum(normalized.values())
            return normalized

        # Legacy profiler format: flat phase-name -> ms mapping.
        normalized = {
            k: float(v)
            for k, v in raw_timings.items()
            if k != "Total" and isinstance(v, (int, float))
        }
        total = raw_timings.get("Total")
        if isinstance(total, (int, float)):
            normalized["Total"] = float(total)
        elif normalized:
            normalized["Total"] = sum(normalized.values())
        return normalized

    print("Measuring AutoRegressiveFLARE implementation...", end=" ", flush=True)
    causalflare_variants = [
        ("AutoRegressiveFLARE (ieee)", "ieee", "triton3_ieee"),
        ("AutoRegressiveFLARE (tf32)", "tf32", "triton3_tf32"),
        ("AutoRegressiveFLARE (tf32x3)", "tf32x3", "triton3_tf32x3"),
    ]
    causalflare_results = {}
    triton3_avg_timings_by_variant = {}
    for row_name, precision_mode, err_prefix in causalflare_variants:
        with _temp_env_var("FLARE_INPUT_PRECISION", precision_mode):
            compile_ms = _probe_compile(lambda: flare_autoregressive_triton(Q, K, V, scale))
            Y_t3, t3_mem = measure_memory(flare_autoregressive_triton, Q, K, V, scale)
            t3_ms = triton.testing.do_bench(lambda: flare_autoregressive_triton(Q, K, V, scale))
            t3_errors = compute_errors(Y_t3, Y_reference, err_prefix) if N <= 4096 and ref_fp32_enabled else {}
            causalflare_results[row_name] = {
                "Y": Y_t3,
                "mem": t3_mem,
                "ms": t3_ms,
                "errors": t3_errors,
                "compile_ms": compile_ms,
            }
            timings_list = []
            with torch.no_grad():
                for _ in range(25):
                    flare_autoregressive_triton(Q, K, V, scale)
                for _ in range(100):
                    _, timings = flare_autoregressive_triton(Q, K, V, scale, None, None, True)
                    normalized_timings = _normalize_chunked_forward_timings(timings)
                    if normalized_timings:
                        timings_list.append(normalized_timings)
            if timings_list:
                variant_timings = {}
                phase_names = sorted({k for t in timings_list for k in t.keys() if k != "Total"})
                for name in phase_names:
                    variant_timings[name] = sum(t.get(name, 0.0) for t in timings_list) / len(timings_list)
                variant_timings["Total"] = sum(variant_timings.values())
                if variant_timings["Total"] > 0 and t3_ms > 0:
                    scale_factor = t3_ms / variant_timings["Total"]
                    for name in phase_names:
                        variant_timings[name] *= scale_factor
                    variant_timings["Total"] = t3_ms
                triton3_avg_timings_by_variant[row_name] = variant_timings
    # Preserve old aliases for downstream logic/profiling text.
    triton3_ms = causalflare_results["AutoRegressiveFLARE (ieee)"]["ms"]
    triton3_mem = causalflare_results["AutoRegressiveFLARE (ieee)"]["mem"]
    triton3_errors = causalflare_results["AutoRegressiveFLARE (ieee)"]["errors"]
    print(
        "Done "
        + ", ".join(
            f"{name.split()[-1].strip('()')}_compile={vals['compile_ms']:.2f} ms"
            for name, vals in causalflare_results.items()
        )
    )

    print("Measuring PyTorch2 (full-chunk) baseline...", end=" ", flush=True)
    old_compile_ms = _probe_compile(lambda: flare_causal_chunked(Q, K, V, scale=scale, chunk_size=K.size(1)))
    Y_old, old_mem = measure_memory(flare_causal_chunked, Q, K, V, scale=scale, chunk_size=K.size(1))
    old_ms = triton.testing.do_bench(lambda: flare_causal_chunked(Q, K, V, scale=scale, chunk_size=K.size(1)))
    old_errors = compute_errors(Y_old, Y_reference, "triton3_old") if N <= 4096 and ref_fp32_enabled else {}
    print(f"Done{_compile_suffix(old_compile_ms)}")

    #======================================================================#
    # Benchmark Causal SDPA implementation
    #======================================================================#
    print("Measuring Causal SDPA implementation...", end=" ", flush=True)

    Q_ = torch.rand_like(K)
    K_ = torch.rand_like(K)
    V_ = torch.rand_like(K)
    # Triton tutorial FA2 kernel in this environment is fp16-only.
    Q_fa2_in = Q_.to(torch.float16).contiguous()
    K_fa2_in = K_.to(torch.float16).contiguous()
    V_fa2_in = V_.to(torch.float16).contiguous()

    sdpa_compile_ms = _probe_compile(lambda: causal_SDPA(Q_, K_, V_))

    # Memory measurement
    _, SDPA_mem = measure_memory(causal_SDPA, Q_, K_, V_)

    # Timing using triton.testing.do_bench
    SDPA_ms = triton.testing.do_bench(lambda: causal_SDPA(Q_, K_, V_))

    print(f"Done{_compile_suffix(sdpa_compile_ms)}")

    #======================================================================#
    # Benchmark Triton FlashAttention2 tutorial implementation
    #======================================================================#
    print("Measuring FlashAttention2 (Triton tutorial) implementation...", end=" ", flush=True)
    if _flash_attention2_triton_bnhd is None:
        fa2_triton_compile_ms = float("nan")
        fa2_triton_mem = 0.0
        fa2_triton_ms = float("nan")
        print("Skipped (import unavailable)")
    else:
        try:
            fa2_triton_compile_ms = _probe_compile(
                lambda: _flash_attention2_triton_bnhd(Q_fa2_in, K_fa2_in, V_fa2_in, causal=True, sm_scale=scale)
            )
            _, fa2_triton_mem = measure_memory(
                _flash_attention2_triton_bnhd, Q_fa2_in, K_fa2_in, V_fa2_in, causal=True, sm_scale=scale
            )
            fa2_triton_ms = triton.testing.do_bench(
                lambda: _flash_attention2_triton_bnhd(Q_fa2_in, K_fa2_in, V_fa2_in, causal=True, sm_scale=scale)
            )
            print(f"Done{_compile_suffix(fa2_triton_compile_ms)}")
        except Exception as exc:
            print(f"[FLARE DEBUG] FlashAttention2 Triton forward failed: {exc}")
            fa2_triton_compile_ms = float("nan")
            fa2_triton_mem = 0.0
            fa2_triton_ms = float("nan")
            print("Failed")

    #======================================================================#
    # Benchmark Recurrent FLARE implementation (small-N only)
    #======================================================================#
    run_recurrent_triton = N <= 2048
    run_recurrent_pytorch = N <= 128
    if run_recurrent_triton:
        print("Measuring RecurrentFLARE implementation...", end=" ", flush=True)
        rec_orig_compile_ms = _probe_compile(lambda: RecurrentFLARE.apply(Q, K, V, scale, None, None, None, None, 1))
        rec_tr_compile_ms = _probe_compile(lambda: RecurrentFLARE.apply(Q, K, V, scale))

        rec_warmup = int(os.environ.get("FLARE_RECURRENT_WARMUP", "2"))
        rec_rep = int(os.environ.get("FLARE_RECURRENT_REP", "2"))

        rec_ref_available = N <= 4096 and ref_fp32_enabled and not torch.isnan(Y_reference).any()
        if rec_ref_available:
            Y_rec_ref = Y_reference.permute(0, 2, 1, 3).contiguous()
        else:
            Y_rec_ref = torch.full((B, H, N, D), float('nan'), device=device, dtype=dtype)

        if run_recurrent_pytorch:
            rec_pt_compile_ms = _probe_compile(lambda: flare_recurrent_pytorch(Q, K, V, scale=scale))
            Y_rec_pt, rec_pt_mem = measure_memory(flare_recurrent_pytorch, Q, K, V, scale=scale)
            rec_pt_ms = triton.testing.do_bench(
                lambda: flare_recurrent_pytorch(Q, K, V, scale=scale), warmup=rec_warmup, rep=rec_rep
            )
        else:
            Y_rec_pt = torch.full((B, H, N, D), float("nan"), device=device, dtype=dtype)
            rec_pt_mem = 0.0
            rec_pt_ms = float("nan")
            rec_pt_compile_ms = float("nan")

        Y_rec_tr, rec_tr_mem = measure_memory(RecurrentFLARE.apply, Q, K, V, scale)
        rec_tr_ms = triton.testing.do_bench(
            lambda: RecurrentFLARE.apply(Q, K, V, scale), warmup=rec_warmup, rep=rec_rep
        )

        Y_rec_orig, rec_orig_mem = measure_memory(RecurrentFLARE.apply, Q, K, V, scale, None, None, None, None, 1)
        rec_orig_ms = triton.testing.do_bench(
            lambda: RecurrentFLARE.apply(Q, K, V, scale, None, None, None, None, 1), warmup=rec_warmup, rep=rec_rep
        )

        rec_pt_errors = compute_errors(Y_rec_pt, Y_rec_ref, "rec_pt") if (rec_ref_available and run_recurrent_pytorch) else {}
        rec_tr_errors = compute_errors(Y_rec_tr, Y_rec_ref, "rec_tr") if rec_ref_available else {}
        rec_orig_errors = compute_errors(Y_rec_orig, Y_rec_ref, "rec_orig") if rec_ref_available else {}

        rec_extra = (
            f", pt_compile={rec_pt_compile_ms:.2f} ms"
            if run_recurrent_pytorch and not math.isnan(rec_pt_compile_ms)
            else ""
        )
        print(
            f"Done (Recurrent Triton path: permute, orig_compile={rec_orig_compile_ms:.2f} ms, "
            f"triton_compile={rec_tr_compile_ms:.2f} ms{rec_extra})"
        )
    else:
        Y_rec_pt = torch.full((B, H, N, D), float("nan"), device=device, dtype=dtype)
        Y_rec_tr = torch.full((B, H, N, D), float("nan"), device=device, dtype=dtype)
        Y_rec_orig = torch.full((B, H, N, D), float("nan"), device=device, dtype=dtype)
        rec_pt_mem = 0.0
        rec_pt_ms = float("nan")
        rec_tr_mem = 0.0
        rec_tr_ms = 0.0
        rec_orig_mem = 0.0
        rec_orig_ms = 0.0
        rec_pt_errors = {}
        rec_tr_errors = {}
        rec_orig_errors = {}
        print("Skipped RecurrentFLARE (N > 2048)")

    #======================================================================#
    # Print results table
    #======================================================================#
    # Calculate speedups against FA2 Triton when available.
    speedup_base_name = "FA2 Triton" if fa2_triton_ms > 0 and not math.isnan(fa2_triton_ms) else "Causal SDPA"
    speedup_base_ms = fa2_triton_ms if speedup_base_name == "FA2 Triton" else SDPA_ms
    ref_speedup       = speedup_base_ms / ref_ms       if ref_ms > 0 else 'N/A'
    pytorch2_speedup  = speedup_base_ms / pytorch2_ms  if pytorch2_ms > 0 else 0.0
    perciever_imlp_speedup = speedup_base_ms / perciever_imlp_ms if perciever_imlp_ms > 0 else 0.0
    pytorch_dense_speedup = speedup_base_ms / pytorch_dense_ms if pytorch_dense_ms > 0 else 0.0
    dense_triton_speedup = speedup_base_ms / dense_triton_ms if dense_triton_ms > 0 else 0.0
    dense1_triton_speedup = speedup_base_ms / dense1_triton_ms if dense1_triton_ms > 0 else 0.0
    fa2_triton_speedup = speedup_base_ms / fa2_triton_ms if fa2_triton_ms > 0 else 0.0
    triton3_speedup   = speedup_base_ms / triton3_ms   if triton3_ms > 0 else 0.0
    old_speedup = speedup_base_ms / old_ms if old_ms > 0 else 0.0
    causalflare_speedups = {
        name: (speedup_base_ms / vals["ms"] if vals["ms"] > 0 else 0.0)
        for name, vals in causalflare_results.items()
    }
    SDPA_speedup      = speedup_base_ms / SDPA_ms if SDPA_ms > 0 else 0.0

    has_nan_ref = torch.isnan(Y_reference).any()

    print("\n" + "="*100)
    print(f"Testing Causal/Recurrent FLARE Forward Pass (B={B}, H={H}, M={M}, N={N}, D={D}, dtype={dtype})")
    print(f"Speedup baseline: {speedup_base_name}")
    print("="*100)
    print(f"{'Implementation':<20} {'Time (ms)':<10} {'Speedup':<10} {'Memory (GB)':<15} "
          f"{'Abs Err (mean/max)':<20} {'Rel Err (mean/max)':<25}")
    print("-"*100)

    # Reference (FP32) row
    ref_abs_err_str = (f"{ref_errors['reference_mean_abs_err']:.2e}/{ref_errors['reference_max_abs_err']:.2e}") if ref_errors and 'reference_mean_abs_err' in ref_errors and 'reference_max_abs_err' in ref_errors else "N/A"
    ref_rel_err_str = (f"{ref_errors['reference_mean_rel_err']:.2e}/{ref_errors['reference_max_rel_err']:.2e}") if ref_errors and 'reference_mean_rel_err' in ref_errors and 'reference_max_rel_err' in ref_errors else "N/A"
    ref_speedup_str = (f"{ref_speedup:.2f}x") if isinstance(ref_speedup, (int, float)) and not math.isnan(ref_speedup) and ref_speedup != float('inf') else "N/A"
    print(f"{'Reference (FP32)':<20} {ref_ms:<10.2f} {ref_speedup_str:<10} {ref_mem:<15.2e} "
          f"{ref_abs_err_str:<20} {ref_rel_err_str:<20}")

    # Optional Reference (BF16) row (compared against FP32 reference)
    if ref_bf16_enabled:
        print("Note: Reference (BF16) errors are computed vs Reference (FP32).")
        ref_bf16_abs_err_str = (f"{ref_bf16_errors['reference_bf16_mean_abs_err']:.2e}/{ref_bf16_errors['reference_bf16_max_abs_err']:.2e}") if ref_bf16_errors and 'reference_bf16_mean_abs_err' in ref_bf16_errors and 'reference_bf16_max_abs_err' in ref_bf16_errors else "N/A"
        ref_bf16_rel_err_str = (f"{ref_bf16_errors['reference_bf16_mean_rel_err']:.2e}/{ref_bf16_errors['reference_bf16_max_rel_err']:.2e}") if ref_bf16_errors and 'reference_bf16_mean_rel_err' in ref_bf16_errors and 'reference_bf16_max_rel_err' in ref_bf16_errors else "N/A"
        ref_bf16_speedup = speedup_base_ms / ref_bf16_ms if ref_bf16_ms > 0 else float("nan")
        ref_bf16_speedup_str = (f"{ref_bf16_speedup:.2f}x") if isinstance(ref_bf16_speedup, (int, float)) and not math.isnan(ref_bf16_speedup) and ref_bf16_speedup != float('inf') else "N/A"
        print(f"{'Reference (BF16)':<20} {ref_bf16_ms:<10.2f} {ref_bf16_speedup_str:<10} {ref_bf16_mem:<15.2e} "
              f"{ref_bf16_abs_err_str:<20} {ref_bf16_rel_err_str:<20}")

    # PyTorch 2 row
    if pytorch2_errors:
        pytorch2_abs_err_str = f"{pytorch2_errors['pytorch2_mean_abs_err']:.2e}/{pytorch2_errors['pytorch2_max_abs_err']:.2e}"
        pytorch2_rel_err_str = f"{pytorch2_errors['pytorch2_mean_rel_err']:.2e}/{pytorch2_errors['pytorch2_max_rel_err']:.2e}"
    else:
        pytorch2_abs_err_str = "N/A"
        pytorch2_rel_err_str = "N/A"
    pytorch2_speedup_str = f"{pytorch2_speedup:.2f}x" if pytorch2_speedup != float('inf') and not math.isnan(pytorch2_speedup) else "N/A"
    print(f"{'PyTorch 2':<20} {pytorch2_ms:<10.2f} {pytorch2_speedup_str:<10} {pytorch2_mem:<15.2e} "
          f"{pytorch2_abs_err_str:<20} {pytorch2_rel_err_str:<20}")

    # PCVR-AR-like impl row
    if perciever_imlp_errors:
        perciever_imlp_abs_err_str = (
            f"{perciever_imlp_errors['perciever_imlp_mean_abs_err']:.2e}/"
            f"{perciever_imlp_errors['perciever_imlp_max_abs_err']:.2e}"
        )
        perciever_imlp_rel_err_str = (
            f"{perciever_imlp_errors['perciever_imlp_mean_rel_err']:.2e}/"
            f"{perciever_imlp_errors['perciever_imlp_max_rel_err']:.2e}"
        )
    else:
        perciever_imlp_abs_err_str = "N/A"
        perciever_imlp_rel_err_str = "N/A"
    perciever_imlp_speedup_str = (
        f"{perciever_imlp_speedup:.2f}x"
        if perciever_imlp_speedup != float('inf') and not math.isnan(perciever_imlp_speedup)
        else "N/A"
    )
    print(
        f"{'PCVR-AR-like impl':<20} {perciever_imlp_ms:<10.2f} {perciever_imlp_speedup_str:<10} "
        f"{perciever_imlp_mem:<15.2e} {perciever_imlp_abs_err_str:<20} {perciever_imlp_rel_err_str:<20}"
    )

    # PyTorch dense row
    if run_dense:
        if pytorch_dense_errors:
            pytorch_dense_abs_err_str = f"{pytorch_dense_errors['pytorch_dense_mean_abs_err']:.2e}/{pytorch_dense_errors['pytorch_dense_max_abs_err']:.2e}"
            pytorch_dense_rel_err_str = f"{pytorch_dense_errors['pytorch_dense_mean_rel_err']:.2e}/{pytorch_dense_errors['pytorch_dense_max_rel_err']:.2e}"
        else:
            pytorch_dense_abs_err_str = "N/A"
            pytorch_dense_rel_err_str = "N/A"
        pytorch_dense_speedup_str = f"{pytorch_dense_speedup:.2f}x" if pytorch_dense_speedup != float('inf') and not math.isnan(pytorch_dense_speedup) else "N/A"
        print(f"{'PyTorch Dense':<20} {pytorch_dense_ms:<10.2f} {pytorch_dense_speedup_str:<10} {pytorch_dense_mem:<15.2e} "
              f"{pytorch_dense_abs_err_str:<20} {pytorch_dense_rel_err_str:<20}")

        # DenseFLARE row
        if dense_triton_errors:
            dense_triton_abs_err_str = f"{dense_triton_errors['dense_triton_mean_abs_err']:.2e}/{dense_triton_errors['dense_triton_max_abs_err']:.2e}"
            dense_triton_rel_err_str = f"{dense_triton_errors['dense_triton_mean_rel_err']:.2e}/{dense_triton_errors['dense_triton_max_rel_err']:.2e}"
        else:
            dense_triton_abs_err_str = "N/A"
            dense_triton_rel_err_str = "N/A"
        dense_triton_speedup_str = f"{dense_triton_speedup:.2f}x" if dense_triton_speedup != float('inf') and not math.isnan(dense_triton_speedup) else "N/A"
        print(f"{'DenseFLARE':<20} {dense_triton_ms:<10.2f} {dense_triton_speedup_str:<10} {dense_triton_mem:<15.2e} "
              f"{dense_triton_abs_err_str:<20} {dense_triton_rel_err_str:<20}")

        # DenseFLARE1 row
        if dense1_triton_errors:
            dense1_triton_abs_err_str = f"{dense1_triton_errors['dense1_triton_mean_abs_err']:.2e}/{dense1_triton_errors['dense1_triton_max_abs_err']:.2e}"
            dense1_triton_rel_err_str = f"{dense1_triton_errors['dense1_triton_mean_rel_err']:.2e}/{dense1_triton_errors['dense1_triton_max_rel_err']:.2e}"
        else:
            dense1_triton_abs_err_str = "N/A"
            dense1_triton_rel_err_str = "N/A"
        dense1_triton_speedup_str = f"{dense1_triton_speedup:.2f}x" if dense1_triton_speedup != float('inf') and not math.isnan(dense1_triton_speedup) else "N/A"
        print(f"{'DenseFLARE1':<20} {dense1_triton_ms:<10.2f} {dense1_triton_speedup_str:<10} {dense1_triton_mem:<15.2e} "
              f"{dense1_triton_abs_err_str:<20} {dense1_triton_rel_err_str:<20}")

    # AutoRegressiveFLARE rows (input_precision variants)
    for row_name, _, err_prefix in causalflare_variants:
        row = causalflare_results[row_name]
        row_errors = row["errors"]
        if row_errors:
            abs_err_str = f"{row_errors[f'{err_prefix}_mean_abs_err']:.2e}/{row_errors[f'{err_prefix}_max_abs_err']:.2e}"
            rel_err_str = f"{row_errors[f'{err_prefix}_mean_rel_err']:.2e}/{row_errors[f'{err_prefix}_max_rel_err']:.2e}"
        else:
            abs_err_str = "N/A"
            rel_err_str = "N/A"
        speedup_val = causalflare_speedups[row_name]
        speedup_str = f"{speedup_val:.2f}x" if speedup_val != float('inf') and not math.isnan(speedup_val) else "N/A"
        print(f"{row_name:<20} {row['ms']:<10.2f} {speedup_str:<10} {row['mem']:<15.2e} "
              f"{abs_err_str:<20} {rel_err_str:<20}")

    if old_errors:
        old_abs_err_str = f"{old_errors['triton3_old_mean_abs_err']:.2e}/{old_errors['triton3_old_max_abs_err']:.2e}"
        old_rel_err_str = f"{old_errors['triton3_old_mean_rel_err']:.2e}/{old_errors['triton3_old_max_rel_err']:.2e}"
    else:
        old_abs_err_str = "N/A"
        old_rel_err_str = "N/A"
    old_speedup_str = f"{old_speedup:.2f}x" if old_speedup != float('inf') and not math.isnan(old_speedup) else "N/A"
    print(f"{'PyTorch2 full-chunk':<20} {old_ms:<10.2f} {old_speedup_str:<10} {old_mem:<15.2e} "
          f"{old_abs_err_str:<20} {old_rel_err_str:<20}")

    # Causal SDPA row
    SDPA_speedup_str = f"{SDPA_speedup:.2f}x" # if SDPA_speedup != float('inf') and not math.isnan(SDPA_speedup) and not has_nan_ref else "N/A"
    print(f"{'Causal SDPA':<20} {SDPA_ms:<10.2f} {SDPA_speedup_str:<10} {SDPA_mem:<15.2e} "
          f"{'N/A':<20} {'N/A':<20}")

    # FlashAttention2 Triton tutorial row
    fa2_triton_speedup_str = (
        f"{fa2_triton_speedup:.2f}x"
        if fa2_triton_speedup != float("inf") and not math.isnan(fa2_triton_speedup)
        else "N/A"
    )
    fa2_triton_ms_str = f"{fa2_triton_ms:.2f}" if not math.isnan(fa2_triton_ms) else "N/A"
    print(f"{'FA2 Triton':<20} {fa2_triton_ms_str:<10} {fa2_triton_speedup_str:<10} {fa2_triton_mem:<15.2e} "
          f"{'N/A':<20} {'N/A':<20}")
    if run_recurrent_triton:
        rec_pt_speedup = speedup_base_ms / rec_pt_ms if rec_pt_ms > 0 else float("nan")
        rec_tr_speedup = speedup_base_ms / rec_tr_ms if rec_tr_ms > 0 else float("nan")
        rec_orig_speedup = speedup_base_ms / rec_orig_ms if rec_orig_ms > 0 else float("nan")

        rec_pt_speedup_str = f"{rec_pt_speedup:.2f}x" if not math.isnan(rec_pt_speedup) else "N/A"
        rec_tr_speedup_str = f"{rec_tr_speedup:.2f}x" if not math.isnan(rec_tr_speedup) else "N/A"
        rec_orig_speedup_str = f"{rec_orig_speedup:.2f}x" if not math.isnan(rec_orig_speedup) else "N/A"

        if rec_pt_errors:
            rec_pt_abs_err_str = f"{rec_pt_errors['rec_pt_mean_abs_err']:.2e}/{rec_pt_errors['rec_pt_max_abs_err']:.2e}"
            rec_pt_rel_err_str = f"{rec_pt_errors['rec_pt_mean_rel_err']:.2e}/{rec_pt_errors['rec_pt_max_rel_err']:.2e}"
        else:
            rec_pt_abs_err_str = "N/A"
            rec_pt_rel_err_str = "N/A"

        if rec_tr_errors:
            rec_tr_abs_err_str = f"{rec_tr_errors['rec_tr_mean_abs_err']:.2e}/{rec_tr_errors['rec_tr_max_abs_err']:.2e}"
            rec_tr_rel_err_str = f"{rec_tr_errors['rec_tr_mean_rel_err']:.2e}/{rec_tr_errors['rec_tr_max_rel_err']:.2e}"
        else:
            rec_tr_abs_err_str = "N/A"
            rec_tr_rel_err_str = "N/A"
        if rec_orig_errors:
            rec_orig_abs_err_str = f"{rec_orig_errors['rec_orig_mean_abs_err']:.2e}/{rec_orig_errors['rec_orig_max_abs_err']:.2e}"
            rec_orig_rel_err_str = f"{rec_orig_errors['rec_orig_mean_rel_err']:.2e}/{rec_orig_errors['rec_orig_max_rel_err']:.2e}"
        else:
            rec_orig_abs_err_str = "N/A"
            rec_orig_rel_err_str = "N/A"
        if run_recurrent_pytorch:
            print(f"{'Recurrent PyTorch':<20} {rec_pt_ms:<10.2f} {rec_pt_speedup_str:<10} {rec_pt_mem:<15.2e} "
                  f"{rec_pt_abs_err_str:<20} {rec_pt_rel_err_str:<20}")
        print(f"{'Recurrent Orig':<20} {rec_orig_ms:<10.2f} {rec_orig_speedup_str:<10} {rec_orig_mem:<15.2e} "
              f"{rec_orig_abs_err_str:<20} {rec_orig_rel_err_str:<20}")
        print(f"{'Recurrent Triton':<20} {rec_tr_ms:<10.2f} {rec_tr_speedup_str:<10} {rec_tr_mem:<15.2e} "
              f"{rec_tr_abs_err_str:<20} {rec_tr_rel_err_str:<20}")

    if run_dense:
        print("="*100)
        print("DenseFLARE1 Phase Profiling")
        print("="*100)
        total_dense1 = dense1_phase1_ms + dense1_phase2_ms + dense1_phase3_ms
        if total_dense1 > 0:
            p1 = 100.0 * dense1_phase1_ms / total_dense1
            p2 = 100.0 * dense1_phase2_ms / total_dense1
            p3 = 100.0 * dense1_phase3_ms / total_dense1
        else:
            p1 = p2 = p3 = 0.0
        print(f"{'Phase 1 (S,P)':<28} {dense1_phase1_ms:>7.3f}ms / {p1:>5.1f}%")
        print(f"{'Phase 2 (R,L)':<28} {dense1_phase2_ms:>7.3f}ms / {p2:>5.1f}%")
        print(f"{'Phase 3 (W,Y)':<28} {dense1_phase3_ms:>7.3f}ms / {p3:>5.1f}%")

    #======================================================================#
    # Print AutoRegressiveFLARE Forward Phase Profiling
    #======================================================================#
    if triton3_avg_timings_by_variant:
        print("="*100)
        print("AutoRegressiveFLARE Forward Phase Profiling Comparison")
        print("="*100)

        # Collect all unique phase names across all variants
        all_phases = set()
        for timings in triton3_avg_timings_by_variant.values():
            all_phases.update([k for k in timings.keys() if k != "Total"])

        # Sort phases by phase number (Phase 0, Phase 1, etc.)
        phase_order = sorted(all_phases, key=lambda x: (
            int(x.split()[1]) if len(x.split()) > 1 and x.split()[1].isdigit() else 999,
            x
        ))

        # Print header (tighter column spacing)
        phase_col_w = 34
        variant_col_w = 21
        header = f"{'Phase':<{phase_col_w}}"
        for row_name, _, _ in causalflare_variants:
            header += f" {row_name:<{variant_col_w}}"
        print(header)
        print("-"*100)

        # Print each phase row
        for phase_name in phase_order:
            row = f"{phase_name:<{phase_col_w}}"
            for variant_name, _, _ in causalflare_variants:
                timings = triton3_avg_timings_by_variant.get(variant_name, {})
                total3 = timings.get("Total", 0.0)
                if phase_name in timings:
                    phase_time3 = timings[phase_name]
                    percentage3 = (phase_time3 / total3 * 100) if total3 > 0 else 0.0
                    cell3 = f"{phase_time3:.3f}ms / {percentage3:.1f}%"
                    row += f" {cell3:<{variant_col_w}}"
                else:
                    row += f" {'N/A':<{variant_col_w}}"

            print(row)

        # Print total row
        print("-"*100)
        total_row = f"{'Total':<{phase_col_w}}"
        for variant_name, _, _ in causalflare_variants:
            timings = triton3_avg_timings_by_variant.get(variant_name, {})
            if "Total" in timings:
                total3_val = timings["Total"]
                total_row += f" {total3_val:.3f}ms / 100.0%".ljust(variant_col_w + 1)
            else:
                total_row += f" {'N/A':<{variant_col_w}}"

        print(total_row)
        print("="*100 + "\n")

    #======================================================================#
    # Benchmark backward pass
    #======================================================================#
    def _unwrap_output(output):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _run_backward(func, q, k, v, *args, **kwargs):
        if q.grad is not None:
            q.grad = None
        if k.grad is not None:
            k.grad = None
        if v.grad is not None:
            v.grad = None
        out = _unwrap_output(func(q, k, v, *args, **kwargs))
        out.sum().backward()

    def _bench_backward(func, q, k, v, *args, **kwargs):
        def _run():
            with torch.enable_grad():
                _run_backward(func, q, k, v, *args, **kwargs)

        ms = triton.testing.do_bench(_run)
        _, mem = measure_memory(_run)
        return ms, mem

    def _probe_backward_compile(func, q_src, k_src, v_src, *args, **kwargs):
        def _run():
            q = q_src.detach().requires_grad_(True)
            k = k_src.detach().requires_grad_(True)
            v = v_src.detach().requires_grad_(True)
            with torch.enable_grad():
                _run_backward(func, q, k, v, *args, **kwargs)
        return _probe_compile(_run)

    def _grad_errors(name, qg, kg, vg, qg_ref, kg_ref, vg_ref):
        q_err = compute_errors(qg, qg_ref, f"{name}_qg")
        k_err = compute_errors(kg, kg_ref, f"{name}_kg")
        v_err = compute_errors(vg, vg_ref, f"{name}_vg")
        mean_abs = max(q_err[f"{name}_qg_mean_abs_err"], k_err[f"{name}_kg_mean_abs_err"], v_err[f"{name}_vg_mean_abs_err"])
        max_abs = max(q_err[f"{name}_qg_max_abs_err"], k_err[f"{name}_kg_max_abs_err"], v_err[f"{name}_vg_max_abs_err"])
        mean_rel = max(q_err[f"{name}_qg_mean_rel_err"], k_err[f"{name}_kg_mean_rel_err"], v_err[f"{name}_vg_mean_rel_err"])
        max_rel = max(q_err[f"{name}_qg_max_rel_err"], k_err[f"{name}_kg_max_rel_err"], v_err[f"{name}_vg_max_rel_err"])
        allclose = q_err[f"{name}_qg_allclose"] and k_err[f"{name}_kg_allclose"] and v_err[f"{name}_vg_allclose"]
        return {
            "allclose": allclose,
            "mean_abs": mean_abs,
            "max_abs": max_abs,
            "mean_rel": mean_rel,
            "max_rel": max_rel,
        }

    # Prepare leaf tensors per method
    Q_ref = Q.detach().requires_grad_(True)
    K_ref = K.detach().requires_grad_(True)
    V_ref = V.detach().requires_grad_(True)

    Q_p2_ref = Q.detach().requires_grad_(True)
    K_p2_ref = K.detach().requires_grad_(True)
    V_p2_ref = V.detach().requires_grad_(True)

    Q_p2 = Q.detach().requires_grad_(True)
    K_p2 = K.detach().requires_grad_(True)
    V_p2 = V.detach().requires_grad_(True)

    Q_perciever_imlp = Q.detach().requires_grad_(True)
    K_perciever_imlp = K.detach().requires_grad_(True)
    V_perciever_imlp = V.detach().requires_grad_(True)
    Q_old_bwd = Q.detach().requires_grad_(True)
    K_old_bwd = K.detach().requires_grad_(True)
    V_old_bwd = V.detach().requires_grad_(True)

    Q_sd = Q_.detach().requires_grad_(True)
    K_sd = K_.detach().requires_grad_(True)
    V_sd = V_.detach().requires_grad_(True)
    Q_fa2 = Q_fa2_in.detach().requires_grad_(True)
    K_fa2 = K_fa2_in.detach().requires_grad_(True)
    V_fa2 = V_fa2_in.detach().requires_grad_(True)
    Q_rec_tr_bwd = Q.detach().requires_grad_(True)
    K_rec_tr_bwd = K.detach().requires_grad_(True)
    V_rec_tr_bwd = V.detach().requires_grad_(True)
    Q_rec_orig_bwd = Q.detach().requires_grad_(True)
    K_rec_orig_bwd = K.detach().requires_grad_(True)
    V_rec_orig_bwd = V.detach().requires_grad_(True)
    Q_pd = Q.detach().requires_grad_(True)
    K_pd = K.detach().requires_grad_(True)
    V_pd = V.detach().requires_grad_(True)
    Q_dense = Q.detach().requires_grad_(True)
    K_dense = K.detach().requires_grad_(True)
    V_dense = V.detach().requires_grad_(True)

    bwd_results = {}
    bwd_grad_errors = {}
    _BWD_PROFILE_TIMINGS.clear()

    # Accuracy reference grads from PyTorch 2
    try:
        with torch.enable_grad():
            _run_backward(flare_causal_chunked, Q_p2_ref, K_p2_ref, V_p2_ref, scale=scale)
        qg_ref = Q_p2_ref.grad.detach()
        kg_ref = K_p2_ref.grad.detach()
        vg_ref = V_p2_ref.grad.detach()
    except Exception as exc:
        print(f"[FLARE DEBUG] PyTorch2 backward reference failed: {exc}")
        Q_ref_f = Q.detach().float().requires_grad_(True)
        K_ref_f = K.detach().float().requires_grad_(True)
        V_ref_f = V.detach().float().requires_grad_(True)
        with torch.enable_grad():
            _run_backward(flare_causal_reference, Q_ref_f, K_ref_f, V_ref_f, scale=scale)
        qg_ref = Q_ref_f.grad.detach()
        kg_ref = K_ref_f.grad.detach()
        vg_ref = V_ref_f.grad.detach()

    # Benchmark + grad accuracy for selected methods
    try:
        print("Measuring backward Reference...", end=" ", flush=True)
        ref_bwd_compile_ms = _probe_backward_compile(flare_causal_reference, Q, K, V, scale=scale)
        bwd_results["Reference"] = _bench_backward(flare_causal_reference, Q_ref, K_ref, V_ref, scale=scale)
        _run_backward(flare_causal_reference, Q_ref, K_ref, V_ref, scale=scale)
        bwd_grad_errors["Reference"] = _grad_errors("reference", Q_ref.grad, K_ref.grad, V_ref.grad, qg_ref, kg_ref, vg_ref)
        print(f"Done{_compile_suffix(ref_bwd_compile_ms)}")
    except Exception:
        bwd_results["Reference"] = (float("nan"), 0.0)
        print("Failed")

    try:
        print("Measuring backward PyTorch 2...", end=" ", flush=True)
        p2_bwd_compile_ms = _probe_backward_compile(flare_causal_chunked, Q, K, V, scale=scale)
        bwd_results["PyTorch 2"] = _bench_backward(
            flare_causal_chunked, Q_p2, K_p2, V_p2, scale=scale
        )
        _run_backward(flare_causal_chunked, Q_p2, K_p2, V_p2, scale=scale)
        bwd_grad_errors["PyTorch 2"] = _grad_errors("pytorch2", Q_p2.grad, K_p2.grad, V_p2.grad, qg_ref, kg_ref, vg_ref)
        print(f"Done{_compile_suffix(p2_bwd_compile_ms)}")
    except Exception as exc:
        print(f"[FLARE DEBUG] PyTorch2 backward failed: {exc}")
        bwd_results["PyTorch 2"] = (float("nan"), 0.0)

    try:
        print("Measuring backward FA2 Triton...", end=" ", flush=True)
        if _flash_attention2_triton_bnhd is not None:
            fa2_bwd_compile_ms = _probe_backward_compile(
                _flash_attention2_triton_bnhd, Q_fa2_in, K_fa2_in, V_fa2_in, causal=True, sm_scale=scale
            )
            bwd_results["FA2 Triton"] = _bench_backward(
                _flash_attention2_triton_bnhd, Q_fa2, K_fa2, V_fa2, causal=True, sm_scale=scale
            )
            print(f"Done{_compile_suffix(fa2_bwd_compile_ms)}")
        else:
            bwd_results["FA2 Triton"] = (float("nan"), 0.0)
            print("Skipped (import unavailable)")
    except Exception as exc:
        print(f"[FLARE DEBUG] FA2 Triton backward failed: {exc}")
        bwd_results["FA2 Triton"] = (float("nan"), 0.0)

    try:
        print("Measuring backward PCVR-AR-like impl...", end=" ", flush=True)
        pcvr_bwd_compile_ms = _probe_backward_compile(flare_causal_perciever_ar, Q, K, V, scale=scale)
        bwd_results["PCVR-AR-like impl"] = _bench_backward(
            flare_causal_perciever_ar, Q_perciever_imlp, K_perciever_imlp, V_perciever_imlp, scale=scale
        )
        _run_backward(flare_causal_perciever_ar, Q_perciever_imlp, K_perciever_imlp, V_perciever_imlp, scale=scale)
        bwd_grad_errors["PCVR-AR-like impl"] = _grad_errors(
            "perciever_imlp",
            Q_perciever_imlp.grad,
            K_perciever_imlp.grad,
            V_perciever_imlp.grad,
            qg_ref,
            kg_ref,
            vg_ref,
        )
        print(f"Done{_compile_suffix(pcvr_bwd_compile_ms)}")
    except Exception as exc:
        print(f"[FLARE DEBUG] PCVR-AR-like impl backward failed (commonly OOM; non-blocking): {exc}")
        bwd_results["PCVR-AR-like impl"] = (float("nan"), 0.0)

    try:
        print("Measuring backward PyTorch2 full-chunk...", end=" ", flush=True)
        old_bwd_compile_ms = _probe_backward_compile(
            flare_causal_chunked, Q, K, V, scale=scale, chunk_size=K.size(1)
        )
        bwd_results["PyTorch2 full-chunk"] = _bench_backward(
            flare_causal_chunked, Q_old_bwd, K_old_bwd, V_old_bwd, scale=scale, chunk_size=K.size(1)
        )
        _run_backward(flare_causal_chunked, Q_old_bwd, K_old_bwd, V_old_bwd, scale=scale, chunk_size=K.size(1))
        bwd_grad_errors["PyTorch2 full-chunk"] = _grad_errors(
            "triton3_old",
            Q_old_bwd.grad,
            K_old_bwd.grad,
            V_old_bwd.grad,
            qg_ref,
            kg_ref,
            vg_ref,
        )
        print(f"Done{_compile_suffix(old_bwd_compile_ms)}")
    except Exception as exc:
        print(f"[FLARE DEBUG] PyTorch2 full-chunk backward failed: {exc}")
        bwd_results["PyTorch2 full-chunk"] = (float("nan"), 0.0)

    for cf_name, precision_mode, grad_prefix in [
        ("AutoRegressiveFLARE (ieee)", "ieee", "triton3_ieee"),
        ("AutoRegressiveFLARE (tf32)", "tf32", "triton3_tf32"),
        ("AutoRegressiveFLARE (tf32x3)", "tf32x3", "triton3_tf32x3"),
    ]:
        Q_t3 = Q.detach().requires_grad_(True)
        K_t3 = K.detach().requires_grad_(True)
        V_t3 = V.detach().requires_grad_(True)
        try:
            print(f"Measuring backward {cf_name}...", end=" ", flush=True)
            with _temp_env_var("FLARE_INPUT_PRECISION", precision_mode):
                cf_compile_ms = _probe_backward_compile(flare_autoregressive_triton, Q, K, V, scale)
                print(f"[compile done {cf_compile_ms:.2f} ms]", end=" ", flush=True)
                bwd_results[cf_name] = _bench_backward(flare_autoregressive_triton, Q_t3, K_t3, V_t3, scale)
                print("[bench done]", end=" ", flush=True)
                mode_key = f"triton3_{precision_mode}"
                if torch.cuda.is_available():
                    warmup = int(os.environ.get("FLARE_BWD_PROFILE_WARMUP", "10"))
                    reps = int(os.environ.get("FLARE_BWD_PROFILE_REPS", "50"))
                    print(f"[profiling warmup={warmup}]", end=" ", flush=True)
                    for _ in range(warmup):
                        _run_backward(flare_autoregressive_triton, Q_t3, K_t3, V_t3, scale)
                    timings_list = []
                    print(f"[profiling reps={reps}]", end=" ", flush=True)
                    for _ in range(reps):
                        _set_bwd_profile_mode(mode_key)
                        _run_backward(flare_autoregressive_triton, Q_t3, K_t3, V_t3, scale)
                        timings_list.append(dict(_BWD_PROFILE_TIMINGS.get(mode_key, {})))
                        _set_bwd_profile_mode(None)
                    if timings_list:
                        avg_timings = {}
                        all_keys = set()
                        for t in timings_list:
                            all_keys.update(t.keys())
                        for k in all_keys:
                            avg_timings[k] = sum(t.get(k, 0.0) for t in timings_list) / len(timings_list)
                        _BWD_PROFILE_TIMINGS[mode_key] = avg_timings
                else:
                    _set_bwd_profile_mode(mode_key)
                    _run_backward(flare_autoregressive_triton, Q_t3, K_t3, V_t3, scale)
                    _set_bwd_profile_mode(None)
            print(f"Done{_compile_suffix(cf_compile_ms)}")
            bwd_grad_errors[cf_name] = _grad_errors(grad_prefix, Q_t3.grad, K_t3.grad, V_t3.grad, qg_ref, kg_ref, vg_ref)
        except Exception as exc:
            _set_bwd_profile_mode(None)
            print(f"[FLARE DEBUG] {cf_name} backward failed: {exc}")
            bwd_results[cf_name] = (float("nan"), 0.0)

    if run_recurrent_triton:
        try:
            bwd_results["Recurrent Orig"] = _bench_backward(
                RecurrentFLARE.apply, Q_rec_orig_bwd, K_rec_orig_bwd, V_rec_orig_bwd, scale, None, None, None, None, 1
            )
            _run_backward(RecurrentFLARE.apply, Q_rec_orig_bwd, K_rec_orig_bwd, V_rec_orig_bwd, scale, None, None, None, None, 1)
            bwd_grad_errors["Recurrent Orig"] = _grad_errors(
                "rec_orig_bwd",
                Q_rec_orig_bwd.grad,
                K_rec_orig_bwd.grad,
                V_rec_orig_bwd.grad,
                qg_ref,
                kg_ref,
                vg_ref,
            )
        except Exception as exc:
            print(f"[FLARE DEBUG] Recurrent Orig backward failed: {exc}")
            bwd_results["Recurrent Orig"] = (float("nan"), 0.0)

        try:
            bwd_results["Recurrent Triton"] = _bench_backward(
                RecurrentFLARE.apply, Q_rec_tr_bwd, K_rec_tr_bwd, V_rec_tr_bwd, scale
            )
            _run_backward(RecurrentFLARE.apply, Q_rec_tr_bwd, K_rec_tr_bwd, V_rec_tr_bwd, scale)
            bwd_grad_errors["Recurrent Triton"] = _grad_errors(
                "rec_tr_bwd",
                Q_rec_tr_bwd.grad,
                K_rec_tr_bwd.grad,
                V_rec_tr_bwd.grad,
                qg_ref,
                kg_ref,
                vg_ref,
            )
        except Exception as exc:
            print(f"[FLARE DEBUG] Recurrent Triton backward failed: {exc}")
            bwd_results["Recurrent Triton"] = (float("nan"), 0.0)
    else:
        bwd_results["Recurrent Orig"] = (float("nan"), 0.0)
        bwd_results["Recurrent Triton"] = (float("nan"), 0.0)

    try:
        bwd_results["Causal SDPA"] = _bench_backward(causal_SDPA, Q_sd, K_sd, V_sd)
    except Exception:
        bwd_results["Causal SDPA"] = (float("nan"), 0.0)

    dense_bwd_err = None
    pytorch_dense_bwd_err = None
    if N <= 128:
        try:
            bwd_results["PyTorch Dense"] = _bench_backward(
                flare_causal_pytorch_dense, Q_pd, K_pd, V_pd, scale=scale
            )
            _run_backward(flare_causal_pytorch_dense, Q_pd, K_pd, V_pd, scale=scale)
            bwd_grad_errors["PyTorch Dense"] = _grad_errors(
                "pytorch_dense", Q_pd.grad, K_pd.grad, V_pd.grad, qg_ref, kg_ref, vg_ref
            )
        except Exception as exc:
            pytorch_dense_bwd_err = str(exc)
            bwd_results["PyTorch Dense"] = (float("nan"), 0.0)

        try:
            bwd_results["DenseFLARE"] = _bench_backward(DenseFLARE.apply, Q_dense, K_dense, V_dense, scale)
            _run_backward(DenseFLARE.apply, Q_dense, K_dense, V_dense, scale)
            bwd_grad_errors["DenseFLARE"] = _grad_errors("dense", Q_dense.grad, K_dense.grad, V_dense.grad, qg_ref, kg_ref, vg_ref)
        except Exception as exc:
            dense_bwd_err = str(exc)
            bwd_results["DenseFLARE"] = (float("nan"), 0.0)
    else:
        bwd_results["PyTorch Dense"] = (float("nan"), 0.0)
        bwd_results["DenseFLARE"] = (float("nan"), 0.0)

    print("\n" + "="*100)
    print(f"Testing Causal/Recurrent FLARE Backward Pass (B={B}, H={H}, M={M}, N={N}, D={D}, dtype={dtype})")
    print("="*100)
    print(f"{'Implementation':<20} {'Time (ms)':<10} {'Speedup':<10} {'Memory (GB)':<15} "
          f"{'Abs Err (mean/max)':<20} {'Rel Err (mean/max)':<20}")
    print("-"*100)

    fa2_bwd_ms = bwd_results.get("FA2 Triton", (float("nan"), 0.0))[0]
    bwd_speedup_base_name = "FA2 Triton" if fa2_bwd_ms > 0 and not math.isnan(fa2_bwd_ms) else "Causal SDPA"
    bwd_speedup_base_ms = fa2_bwd_ms if bwd_speedup_base_name == "FA2 Triton" else bwd_results["Causal SDPA"][0]
    for name in [
        "Reference",
        "PyTorch 2",
        "PCVR-AR-like impl",
        "PyTorch2 full-chunk",
        "AutoRegressiveFLARE (ieee)",
        "AutoRegressiveFLARE (tf32)",
        "AutoRegressiveFLARE (tf32x3)",
        "Recurrent Orig",
        "Recurrent Triton",
        "PyTorch Dense",
        "DenseFLARE",
        "FA2 Triton",
        "Causal SDPA",
    ]:
        ms, mem = bwd_results[name]
        speedup = (
            bwd_speedup_base_ms / ms
            if ms > 0 and not math.isnan(ms) and bwd_speedup_base_ms > 0 and not math.isnan(bwd_speedup_base_ms)
            else float("nan")
        )
        speedup_str = f"{speedup:.2f}x" if not math.isnan(speedup) else "N/A"
        ms_str = f"{ms:.2f}" if not math.isnan(ms) else "N/A"

        if name in bwd_grad_errors:
            err = bwd_grad_errors[name]
            abs_err_str = f"{err['mean_abs']:.2e}/{err['max_abs']:.2e}"
            rel_err_str = f"{err['mean_rel']:.2e}/{err['max_rel']:.2e}"
        else:
            abs_err_str = "N/A"
            rel_err_str = "N/A"

        if name == "PyTorch Dense" and pytorch_dense_bwd_err:
            print(f"{name:<20} {ms_str:<10} {speedup_str:<10} {mem:<15.2e} "
                  f"{abs_err_str:<20} {rel_err_str:<20} (ERR: {pytorch_dense_bwd_err})")
        elif name == "DenseFLARE" and dense_bwd_err:
            print(f"{name:<20} {ms_str:<10} {speedup_str:<10} {mem:<15.2e} "
                  f"{abs_err_str:<20} {rel_err_str:<20} (ERR: {dense_bwd_err})")
        else:
            print(f"{name:<20} {ms_str:<10} {speedup_str:<10} {mem:<15.2e} "
                  f"{abs_err_str:<20} {rel_err_str:<20}")

    if _BWD_PROFILE_TIMINGS:
        print("="*100)
        print("AutoRegressiveFLARE Backward Phase Profiling")
        print("="*100)
        phase_col_w = 34
        variant_col_w = 21
        header = f"{'Phase':<{phase_col_w}}"
        bwd_profile_cols = [
            ("AutoRegressiveFLARE (ieee)", "triton3_ieee"),
            ("AutoRegressiveFLARE (tf32)", "triton3_tf32"),
            ("AutoRegressiveFLARE (tf32x3)", "triton3_tf32x3"),
        ]
        for label, mode_key in bwd_profile_cols:
            if mode_key in _BWD_PROFILE_TIMINGS:
                header += f" {label:<{variant_col_w}}"
        print(header)
        print("-"*100)

        totals = {}
        for name, timings in _BWD_PROFILE_TIMINGS.items():
            totals[name] = sum(timings.values())

        # Normalize phase timings to match benchmark total time (do_bench) per precision mode.
        for label, mode_key in [
            ("AutoRegressiveFLARE (ieee)", "triton3_ieee"),
            ("AutoRegressiveFLARE (tf32)", "triton3_tf32"),
            ("AutoRegressiveFLARE (tf32x3)", "triton3_tf32x3"),
        ]:
            if mode_key in _BWD_PROFILE_TIMINGS and label in bwd_results:
                bench_ms = bwd_results[label][0]
                total_ms = totals.get(mode_key, 0.0)
                if total_ms > 0 and bench_ms > 0 and not math.isnan(bench_ms):
                    scale = bench_ms / total_ms
                    for k in _BWD_PROFILE_TIMINGS[mode_key]:
                        _BWD_PROFILE_TIMINGS[mode_key][k] *= scale
                    totals[mode_key] = bench_ms

        all_phases = set()
        for timings in _BWD_PROFILE_TIMINGS.values():
            all_phases.update(timings.keys())
        phase_order = sorted(all_phases)

        for phase in phase_order:
            row = f"{phase:<{phase_col_w}}"
            for _, mode_key in bwd_profile_cols:
                if mode_key in _BWD_PROFILE_TIMINGS:
                    val = _BWD_PROFILE_TIMINGS[mode_key].get(phase, None)
                    total = totals.get(mode_key, 0.0)
                    if val is not None and total > 0:
                        cell = f"{val:.3f}ms / {val / total * 100:.1f}%"
                    else:
                        cell = "N/A"
                    row += f" {cell:<{variant_col_w}}"
            print(row)

        print("-"*100)
        total_row = f"{'Total':<{phase_col_w}}"
        for _, mode_key in bwd_profile_cols:
            if mode_key in totals:
                total_row += f" {totals[mode_key]:.3f}ms / 100.0%".ljust(variant_col_w + 1)
        print(total_row)
        print("="*100)

    return

#======================================================================#
#======================================================================#
#======================================================================#
# END MAIN FUNCTION
#======================================================================#
#======================================================================#
#======================================================================#

#======================================================================#
if __name__ == "__main__":
    main()
#======================================================================#
#

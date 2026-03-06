"""Extracted gradient-check regression/stress suite implementation."""

from testing.suite_runners.common import *


def _grad_report(name: str, g_pred: torch.Tensor, g_ref: torch.Tensor, atol: float) -> None:
    delta = (g_pred - g_ref).abs()
    max_abs = delta.amax().item()
    mean_abs = delta.mean().item()
    rel_l2 = _rel_l2_err(g_pred, g_ref)
    max_rel = _max_rel_err(g_pred, g_ref, atol)
    idx = _max_abs_idx(delta)
    finite_pred = torch.isfinite(g_pred).all().item()
    finite_ref = torch.isfinite(g_ref).all().item()
    print(
        f"[GRAD] {name}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"rel_l2={rel_l2:.3e} max_rel={max_rel:.3e} finite={finite_pred}/{finite_ref} worst_idx={idx}"
    )


def _gradcheck_suite(dtype: torch.dtype, B: int, H: int, N: int, M: int, D: int, scale: float, atol: float) -> None:
    device = torch.device("cuda")
    Q = torch.randn(H, M, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)
    R = torch.randn(B, N, H, D, device=device, dtype=dtype)

    def _loss_ref(q, k, v):
        y = flare_causal_reference(q, k, v, scale=scale)
        return (y * R).sum()

    def _loss_p2(q, k, v):
        y = flare_causal_chunked(q, k, v, scale=scale)
        return (y * R).sum()

    def _loss_t3(q, k, v):
        y = flare_chunk_triton(q, k, v, scale)
        return (y * R).sum()

    loss_ref = _loss_ref(Q, K, V)
    gq_ref, gk_ref, gv_ref = torch.autograd.grad(loss_ref, (Q, K, V), retain_graph=True)
    loss_p2 = _loss_p2(Q, K, V)
    gq_p2, gk_p2, gv_p2 = torch.autograd.grad(loss_p2, (Q, K, V), retain_graph=True)
    loss_t3 = None
    gq_t3 = gk_t3 = gv_t3 = None
    t3_err = None
    try:
        prev_bwd_fp32 = os.environ.get("FLARE_TRITON3_BWD_FP32", "")
        prev_force_stats = os.environ.get("FLARE_TRITON_FORCE_FP32_STATS", "")
        os.environ["FLARE_TRITON3_BWD_FP32"] = "1"
        os.environ["FLARE_TRITON_FORCE_FP32_STATS"] = "1"
        loss_t3 = _loss_t3(Q, K, V)
        gq_t3, gk_t3, gv_t3 = torch.autograd.grad(loss_t3, (Q, K, V), retain_graph=False)
    except Exception as exc:
        t3_err = str(exc)
    finally:
        if prev_bwd_fp32 == "":
            os.environ.pop("FLARE_TRITON3_BWD_FP32", None)
        else:
            os.environ["FLARE_TRITON3_BWD_FP32"] = prev_bwd_fp32
        if prev_force_stats == "":
            os.environ.pop("FLARE_TRITON_FORCE_FP32_STATS", None)
        else:
            os.environ["FLARE_TRITON_FORCE_FP32_STATS"] = prev_force_stats

    loss_t3_str = f"{loss_t3.item():.3e}" if loss_t3 is not None else "N/A"
    print(f"[GRAD] loss_ref={loss_ref.item():.3e} loss_p2={loss_p2.item():.3e} loss_t3={loss_t3_str}")
    _grad_report("dQ Pytorch2 vs Ref", gq_p2, gq_ref, atol)
    _grad_report("dK Pytorch2 vs Ref", gk_p2, gk_ref, atol)
    _grad_report("dV Pytorch2 vs Ref", gv_p2, gv_ref, atol)
    if t3_err is None and gq_t3 is not None:
        _grad_report("dQ Triton vs Ref", gq_t3, gq_ref, atol)
        _grad_report("dK Triton vs Ref", gk_t3, gk_ref, atol)
        _grad_report("dV Triton vs Ref", gv_t3, gv_ref, atol)
    else:
        print(f"[GRAD] Triton backward skipped: {t3_err}")


def _run_correctness_reference_grads_with_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    scale: float,
    Q_dec: torch.Tensor | None,
    K_dec: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    Q_dec_ref = Q_dec.detach().clone().requires_grad_(True) if Q_dec is not None else None
    K_dec_ref = K_dec.detach().clone().requires_grad_(True) if K_dec is not None else None
    with _temp_env_var("FLARE_REFERENCE_FP32", "1"):
        with _scoped_float32_math_mode(allow_tf32=False):
            Y_ref = flare_causal_reference(Q_ref, K_ref, V_ref, Q_dec=Q_dec_ref, K_dec=K_dec_ref, scale=scale)
            Y_ref.sum().backward()
    return {
        "q": Q_ref.grad,
        "k": K_ref.grad,
        "v": V_ref.grad,
        "q_dec": None if Q_dec_ref is None else Q_dec_ref.grad,
        "k_dec": None if K_dec_ref is None else K_dec_ref.grad,
    }


def _run_correctness_pytorch2_grads_with_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    scale: float,
    chunk_size: int,
    allow_tf32: bool,
    Q_dec: torch.Tensor | None,
    K_dec: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    Q_dec_ref = Q_dec.detach().clone().requires_grad_(True) if Q_dec is not None else None
    K_dec_ref = K_dec.detach().clone().requires_grad_(True) if K_dec is not None else None
    with _scoped_float32_math_mode(allow_tf32=allow_tf32):
        Y_ref = flare_causal_chunked(
            Q_ref, K_ref, V_ref, scale=scale, chunk_size=chunk_size, Q_dec=Q_dec_ref, K_dec=K_dec_ref
        )
        Y_ref.sum().backward()
    return {
        "q": Q_ref.grad,
        "k": K_ref.grad,
        "v": V_ref.grad,
        "q_dec": None if Q_dec_ref is None else Q_dec_ref.grad,
        "k_dec": None if K_dec_ref is None else K_dec_ref.grad,
    }


def _run_grad_checks_suite() -> None:
    if not torch.cuda.is_available():
        print("[FLARE DEBUG] CUDA not available, skipping gradient checks suite.")
        return

    device = torch.device("cuda")
    strict = _strict_mode_enabled("FLARE_CORRECTNESS_STRICT", default=True)
    decode_modes = _parse_decode_separation_modes("FLARE_CORRECTNESS_DECODE_SEPARATION_MODES")

    seed_env = os.environ.get("FLARE_CORRECTNESS_SEED", "")
    if seed_env:
        seed = int(seed_env)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    failures: list[str] = []

    dtypes_env = os.environ.get("FLARE_CORRECTNESS_DTYPES", "")
    if dtypes_env:
        dtypes = [getattr(torch, name.strip()) for name in dtypes_env.split(",") if name.strip()]
    else:
        dtypes = [torch.bfloat16, torch.float16]

    qk_stds_env = os.environ.get("FLARE_CORRECTNESS_QK_STDS", "")
    if qk_stds_env:
        qk_stds = [float(x) for x in qk_stds_env.split(",") if x.strip()]
    else:
        qk_stds = [0.5, 1.0, 2.0, 4.0]

    shapes_env = os.environ.get("FLARE_CORRECTNESS_SHAPES", "")
    if shapes_env:
        shapes = []
        for spec in shapes_env.split(";"):
            if not spec.strip():
                continue
            b, h, n, m, d = (int(x) for x in spec.split(","))
            shapes.append((b, h, n, m, d))
    else:
        shapes = [
            (1, 2, 1024, 128, 32),
        ]

    grad_shapes = [
        (1, 1, 16, 16, 16),
        (1, 1, 33, 16, 32),
    ]
    grad_scales_env = os.environ.get("FLARE_CORRECTNESS_SUITE_GRAD_SCALES", "")
    if grad_scales_env:
        grad_scales = [float(x) for x in grad_scales_env.split(",") if x.strip()]
    else:
        grad_scales = [1.0]

    grad_limit_env = os.environ.get("FLARE_CORRECTNESS_GRAD_LIMIT", "")
    if grad_limit_env:
        grad_limit = int(grad_limit_env)
    else:
        grad_limit = max(1, len(decode_modes))

    grad_count = 0

    for dtype in dtypes:
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        grad_mean_abs_max = _env_float("FLARE_CORRECTNESS_GRAD_MEAN_ABS_MAX", 5e-3)
        grad_max_abs_max = _env_float("FLARE_CORRECTNESS_GRAD_MAX_ABS_MAX", 8e-1)
        grad_cos_min = _env_float("FLARE_CORRECTNESS_GRAD_COS_MIN", 0.999)
        fa_grad_slack = _env_float("FLARE_CORRECTNESS_FA_GRAD_SLACK", 1e-5)

        for (B, H, N, M, D) in shapes:
            scale = D ** -0.5
            input_precision, chunk_size = _correctness_chunk_config(M=M, N=N, D=D, dtype=dtype)
            fa_grad_mult = _env_float(
                "FLARE_CORRECTNESS_FA_GRAD_MULT",
                2.0 if input_precision == "tf32x3" else 5.0,
            )
            noise_allow_tf32 = input_precision != "ieee"

            for qk_std in qk_stds:
                dist_mode = os.environ.get("FLARE_CORRECTNESS_QK_DIST", "normal")
                outlier_p = float(os.environ.get("FLARE_CORRECTNESS_QK_OUTLIER_P", "0.01"))
                outlier_scale = float(os.environ.get("FLARE_CORRECTNESS_QK_OUTLIER_SCALE", "25.0"))
                if dist_mode == "outlier":
                    Q = qk_std * torch.randn(H, M, D, device=device, dtype=dtype)
                    K = qk_std * torch.randn(B, N, H, D, device=device, dtype=dtype)
                    if outlier_p > 0.0 and outlier_scale > 0.0:
                        q_mask = torch.rand_like(Q) < outlier_p
                        k_mask = torch.rand_like(K) < outlier_p
                        Q = Q + q_mask * (outlier_scale * qk_std * torch.randn_like(Q))
                        K = K + k_mask * (outlier_scale * qk_std * torch.randn_like(K))
                else:
                    Q = qk_std * torch.randn(H, M, D, device=device, dtype=dtype)
                    K = qk_std * torch.randn(B, N, H, D, device=device, dtype=dtype)

                V = torch.randn(B, N, H, D, device=device, dtype=dtype)
                Q_dec_rand = qk_std * torch.randn(B, N, H, D, device=device, dtype=dtype)
                K_dec_rand = qk_std * torch.randn(H, M, D, device=device, dtype=dtype)

                for separate_q_dec, separate_k_dec in decode_modes:
                    if os.environ.get("FLARE_CORRECTNESS_GRAD", "0") != "1" or grad_count >= grad_limit:
                        continue

                    decode_label = _decode_mode_label(separate_q_dec, separate_k_dec)
                    Q_dec = Q_dec_rand if separate_q_dec else None
                    K_dec = K_dec_rand if separate_k_dec else None

                    print(
                        f"[SUITE GRAD] dtype={dtype} B={B} H={H} N={N} M={M} D={D} "
                        f"scale={scale:.6g} qk_std={qk_std} input_precision={input_precision} {decode_label}"
                    )

                    grads_ref = _run_correctness_reference_grads_with_decode(
                        Q, K, V, scale=scale, Q_dec=Q_dec, K_dec=K_dec
                    )
                    grads_noise = _run_correctness_pytorch2_grads_with_decode(
                        Q,
                        K,
                        V,
                        scale=scale,
                        chunk_size=chunk_size,
                        allow_tf32=noise_allow_tf32,
                        Q_dec=Q_dec,
                        K_dec=K_dec,
                    )

                    Q_t3 = Q.detach().clone().requires_grad_(True)
                    K_t3 = K.detach().clone().requires_grad_(True)
                    V_t3 = V.detach().clone().requires_grad_(True)
                    Q_dec_t3 = Q_dec.detach().clone().requires_grad_(True) if Q_dec is not None else None
                    K_dec_t3 = K_dec.detach().clone().requires_grad_(True) if K_dec is not None else None
                    Y_t3 = flare_chunk_triton(
                        Q_t3, K_t3, V_t3, scale, chunk_size, input_precision, False, Q_dec_t3, K_dec_t3
                    )
                    Y_t3.sum().backward()
                    grads_t3 = {
                        "q": Q_t3.grad,
                        "k": K_t3.grad,
                        "v": V_t3.grad,
                        "q_dec": None if Q_dec_t3 is None else Q_dec_t3.grad,
                        "k_dec": None if K_dec_t3 is None else K_dec_t3.grad,
                    }

                    grad_specs = [
                        ("dQ", "q", "suite_qg"),
                        ("dK", "k", "suite_kg"),
                        ("dV", "v", "suite_vg"),
                    ]
                    if separate_q_dec:
                        grad_specs.append(("dQ_dec", "q_dec", "suite_qdg"))
                    if separate_k_dec:
                        grad_specs.append(("dK_dec", "k_dec", "suite_kdg"))

                    for grad_name, key, prefix in grad_specs:
                        err = compute_errors(grads_t3[key], grads_ref[key], prefix)
                        noise_err = compute_errors(grads_noise[key], grads_ref[key], f"{prefix}_noise")
                        cos = _tensor_cosine(grads_t3[key], grads_ref[key])
                        print(
                            "[SUITE GRAD] "
                            f"{grad_name} mean_abs={err[f'{prefix}_mean_abs_err']:.3e} "
                            f"max_abs={err[f'{prefix}_max_abs_err']:.3e} "
                            f"noise_max_abs={noise_err[f'{prefix}_noise_max_abs_err']:.3e} "
                            f"cos={cos:.5f}"
                        )

                        if strict and input_precision == "ieee":
                            _maybe_record_failure(
                                failures,
                                err[f"{prefix}_mean_abs_err"] > grad_mean_abs_max,
                                (
                                    f"Grad checks suite failed ({grad_name} mean_abs): "
                                    f"{err[f'{prefix}_mean_abs_err']:.3e} > {grad_mean_abs_max:.3e} "
                                    f"for dtype={dtype}, B={B}, H={H}, N={N}, M={M}, D={D}, "
                                    f"qk_std={qk_std}, {decode_label}"
                                ),
                            )
                            _maybe_record_failure(
                                failures,
                                err[f"{prefix}_max_abs_err"] > grad_max_abs_max,
                                (
                                    f"Grad checks suite failed ({grad_name} max_abs): "
                                    f"{err[f'{prefix}_max_abs_err']:.3e} > {grad_max_abs_max:.3e} "
                                    f"for dtype={dtype}, B={B}, H={H}, N={N}, M={M}, D={D}, "
                                    f"qk_std={qk_std}, {decode_label}"
                                ),
                            )
                            _maybe_record_failure(
                                failures,
                                cos < grad_cos_min,
                                (
                                    f"Grad checks suite failed ({grad_name} cosine): "
                                    f"{cos:.6f} < {grad_cos_min:.6f} for dtype={dtype}, B={B}, H={H}, "
                                    f"N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                                ),
                            )
                        elif strict:
                            grad_limit_scaled = _scaled_error_limit(
                                noise_err[f"{prefix}_noise_max_abs_err"], fa_grad_mult, fa_grad_slack
                            )
                            _maybe_record_failure(
                                failures,
                                err[f"{prefix}_max_abs_err"] > grad_limit_scaled,
                                (
                                    f"Grad checks suite failed ({grad_name} max_abs vs reference): "
                                    f"{err[f'{prefix}_max_abs_err']:.3e} > {grad_limit_scaled:.3e} "
                                    f"(noise={noise_err[f'{prefix}_noise_max_abs_err']:.3e}, "
                                    f"mult={fa_grad_mult:.2f}, slack={fa_grad_slack:.1e}) "
                                    f"for input_precision={input_precision}, dtype={dtype}, B={B}, H={H}, "
                                    f"N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                                ),
                            )

                    grad_count += 1

        if os.environ.get("FLARE_CORRECTNESS_SUITE_GRAD", "0") == "1":
            for (B, H, N, M, D) in grad_shapes:
                for scale in grad_scales:
                    print(f"[SUITE GRAD] dtype={dtype} B={B} H={H} N={N} M={M} D={D} scale={scale}")
                    _gradcheck_suite(dtype, B, H, N, M, D, scale=scale, atol=atol)

    if strict and failures:
        summary = "\n".join(f"- {msg}" for msg in failures[:12])
        extra = "" if len(failures) <= 12 else f"\n- ... and {len(failures) - 12} more"
        raise AssertionError(f"FLARE gradient validation failed ({len(failures)} issues):\n{summary}{extra}")

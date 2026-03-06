"""Extracted regression/stress suite implementation."""

from testing.suites.common import *


def _run_correctness_suite():
    device = torch.device("cuda")
    strict = _strict_mode_enabled("FLARE_CORRECTNESS_STRICT", default=True)
    enforce_pytorch2_gate = _strict_mode_enabled("FLARE_CORRECTNESS_ENFORCE_PYTORCH2", default=False)
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
    for dtype in dtypes:
        atol = 1e-2 if dtype == torch.bfloat16 else 1e-3
        fwd_mean_abs_max = _env_float(
            "FLARE_CORRECTNESS_FWD_MEAN_ABS_MAX",
            3e-4 if dtype == torch.bfloat16 else 2e-4,
        )
        fwd_max_abs_max = _env_float(
            "FLARE_CORRECTNESS_FWD_MAX_ABS_MAX",
            3e-2 if dtype == torch.bfloat16 else 2e-2,
        )
        fa_fwd_mult = _env_float("FLARE_CORRECTNESS_FA_FWD_MULT", 2.0)
        fa_fwd_slack = _env_float("FLARE_CORRECTNESS_FA_FWD_SLACK", 1e-5)
        for (B, H, N, M, D) in shapes:
            scale = (D ** -0.5)
            input_precision, chunk_size = _correctness_chunk_config(M=M, N=N, D=D, dtype=dtype)
            noise_allow_tf32 = input_precision != "ieee"
            for qk_std in qk_stds:
                dist_mode = os.environ.get("FLARE_CORRECTNESS_QK_DIST", "normal")
                outlier_p = float(os.environ.get("FLARE_CORRECTNESS_QK_OUTLIER_P", "0.01"))
                outlier_scale = float(os.environ.get("FLARE_CORRECTNESS_QK_OUTLIER_SCALE", "25.0"))
                if dist_mode == "outlier":
                    Q = qk_std * torch.randn(H, M, D, device=device, dtype=dtype)
                    K = qk_std * torch.randn(B, N, H, D, device=device, dtype=dtype)
                    if outlier_p > 0.0 and outlier_scale > 0.0:
                        q_mask = (torch.rand_like(Q) < outlier_p)
                        k_mask = (torch.rand_like(K) < outlier_p)
                        Q = Q + q_mask * (outlier_scale * qk_std * torch.randn_like(Q))
                        K = K + k_mask * (outlier_scale * qk_std * torch.randn_like(K))
                else:
                    Q = qk_std * torch.randn(H, M, D, device=device, dtype=dtype)
                    K = qk_std * torch.randn(B, N, H, D, device=device, dtype=dtype)
                V = torch.randn(B, N, H, D, device=device, dtype=dtype)
                Q_dec_rand = qk_std * torch.randn(B, N, H, D, device=device, dtype=dtype)
                K_dec_rand = qk_std * torch.randn(H, M, D, device=device, dtype=dtype)
                for separate_q_dec, separate_k_dec in decode_modes:
                    decode_label = _decode_mode_label(separate_q_dec, separate_k_dec)
                    Q_dec = Q_dec_rand if separate_q_dec else None
                    K_dec = K_dec_rand if separate_k_dec else None
                    print(
                        f"\n[SUITE] dtype={dtype} B={B} H={H} N={N} M={M} D={D} "
                        f"scale={scale:.6g} qk_std={qk_std} input_precision={input_precision} "
                        f"chunk_size={chunk_size} {decode_label}"
                    )

                    # Validation policy:
                    # - flare_causal_reference is the canonical correctness oracle.
                    # - flare_causal_chunked (run with the same input dtype/chunking
                    #   as ChunkedFLARE and explicit matmul TF32 controls) is the
                    #   FlashAttention-style noise model for non-IEEE modes.
                    Y_ref = _run_correctness_reference(Q, K, V, scale=scale, Q_dec=Q_dec, K_dec=K_dec)
                    Y_pytorch2 = _run_correctness_pytorch2(
                        Q, K, V, scale=scale, chunk_size=chunk_size, allow_tf32=noise_allow_tf32, Q_dec=Q_dec, K_dec=K_dec
                    )
                    Y_t3 = flare_chunk_triton(Q, K, V, scale, chunk_size, input_precision, False, Q_dec, K_dec)
                    y_p2_err = compute_errors(Y_pytorch2, Y_ref, "suite_p2")
                    y_t3_err = compute_errors(Y_t3, Y_ref, "suite_t3")
                    _print_err_report("PyTorch 2 vs Ref", Y_pytorch2, Y_ref, atol)
                    _print_err_report("ChunkedFLARE vs Ref", Y_t3, Y_ref, atol)

                    y_pa_err = None
                    if (not separate_q_dec) and (not separate_k_dec):
                        Y_pa = flare_causal_perceiver_ar(Q, K, V, scale=scale)
                        y_pa_err = compute_errors(Y_pa, Y_ref, "suite_pa")
                        _print_err_report("PCVR-AR-like impl vs Ref", Y_pa, Y_ref, atol)

                    if strict and y_pa_err is not None:
                        _maybe_record_failure(
                            failures,
                            y_pa_err["suite_pa_mean_abs_err"] > fwd_mean_abs_max,
                            (
                                "Correctness suite failed (PCVR-AR-like impl mean_abs): "
                                f"{y_pa_err['suite_pa_mean_abs_err']:.3e} > {fwd_mean_abs_max:.3e} "
                                f"for dtype={dtype}, B={B}, H={H}, N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                            ),
                        )
                        _maybe_record_failure(
                            failures,
                            y_pa_err["suite_pa_max_abs_err"] > fwd_max_abs_max,
                            (
                                "Correctness suite failed (PCVR-AR-like impl max_abs): "
                                f"{y_pa_err['suite_pa_max_abs_err']:.3e} > {fwd_max_abs_max:.3e} "
                                f"for dtype={dtype}, B={B}, H={H}, N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                            ),
                        )
                    if strict and enforce_pytorch2_gate:
                        _maybe_record_failure(
                            failures,
                            y_p2_err["suite_p2_mean_abs_err"] > fwd_mean_abs_max,
                            (
                                "Correctness suite failed (PyTorch 2 mean_abs): "
                                f"{y_p2_err['suite_p2_mean_abs_err']:.3e} > {fwd_mean_abs_max:.3e} "
                                f"for dtype={dtype}, B={B}, H={H}, N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                            ),
                        )
                        _maybe_record_failure(
                            failures,
                            y_p2_err["suite_p2_max_abs_err"] > fwd_max_abs_max,
                            (
                                "Correctness suite failed (PyTorch 2 max_abs): "
                                f"{y_p2_err['suite_p2_max_abs_err']:.3e} > {fwd_max_abs_max:.3e} "
                                f"for dtype={dtype}, B={B}, H={H}, N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                            ),
                        )
                    if strict and input_precision == "ieee":
                        _maybe_record_failure(
                            failures,
                            y_t3_err["suite_t3_mean_abs_err"] > fwd_mean_abs_max,
                            (
                                "Correctness suite failed (ChunkedFLARE mean_abs): "
                                f"{y_t3_err['suite_t3_mean_abs_err']:.3e} > {fwd_mean_abs_max:.3e} "
                                f"for dtype={dtype}, B={B}, H={H}, N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                            ),
                        )
                        _maybe_record_failure(
                            failures,
                            y_t3_err["suite_t3_max_abs_err"] > fwd_max_abs_max,
                            (
                                "Correctness suite failed (ChunkedFLARE max_abs): "
                                f"{y_t3_err['suite_t3_max_abs_err']:.3e} > {fwd_max_abs_max:.3e} "
                                f"for dtype={dtype}, B={B}, H={H}, N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                            ),
                        )
                    elif strict:
                        t3_fwd_max_abs_limit = _scaled_error_limit(
                            y_p2_err["suite_p2_max_abs_err"], fa_fwd_mult, fa_fwd_slack
                        )
                        _maybe_record_failure(
                            failures,
                            y_t3_err["suite_t3_max_abs_err"] > t3_fwd_max_abs_limit,
                            (
                                "Correctness suite failed (ChunkedFLARE max_abs vs reference): "
                                f"{y_t3_err['suite_t3_max_abs_err']:.3e} > {t3_fwd_max_abs_limit:.3e} "
                                f"(pytorch2={y_p2_err['suite_p2_max_abs_err']:.3e}, mult={fa_fwd_mult:.2f}, "
                                f"slack={fa_fwd_slack:.1e}) for input_precision={input_precision}, dtype={dtype}, "
                                f"B={B}, H={H}, N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                            ),
                        )
                    try:
                        _check_finite("PyTorch2.Y", Y_pytorch2)
                    except RuntimeError as exc:
                        print(f"[FLARE DEBUG] {exc}")

                    if (not separate_q_dec) and (not separate_k_dec):
                        if N <= 128 and (N % 16) == 0 and (M % 16) == 0 and (D % 16) == 0:
                            Y_dense = DenseFLARE.apply(Q, K, V, scale)
                            _print_err_report("DenseFLARE vs Ref", Y_dense, Y_ref, atol)
                            try:
                                _check_finite("DenseFLARE.Y_triton", Y_dense)
                            except RuntimeError as exc:
                                print(f"[FLARE DEBUG] {exc}")

                    # Per-axis worst cases (print top-3 tokens/heads)
                    delta = (Y_t3 - Y_ref).abs()
                    per = _per_axis_max(delta)
                    per_t = per["per_t"]
                    per_h = per["per_h"]
                    topk_t = torch.topk(per_t, k=min(3, per_t.numel()))
                    topk_h = torch.topk(per_h, k=min(3, per_h.numel()))
                    print("[CHECK] per_t max_abs top:", [(int(i), float(v)) for v, i in zip(topk_t.values, topk_t.indices)])
                    print("[CHECK] per_h max_abs top:", [(int(i), float(v)) for v, i in zip(topk_h.values, topk_h.indices)])

                    # Gradient checks were extracted to testing/suites/grad_checks.py.

                    # _causality_check("PyTorch Dense", lambda q,k,v,scale: flare_causal_pytorch_dense(q, k, v, scale=scale), Q, K, V, scale, [0, 1, N//2, N-2], atol)
                    # _causality_check("DenseFLARE", lambda q,k,v,scale: DenseFLARE.apply(q, k, v, scale), Q, K, V, scale, [0, 1, N//2, N-2], atol)
                    if (
                        os.environ.get("FLARE_CORRECTNESS_RECURRENT_CAUSALITY", "0") == "1"
                        and (not separate_q_dec)
                        and (not separate_k_dec)
                    ):
                        prefixes = [0, 1, N // 2, N - 2]
                        prefixes = [p for p in prefixes if p >= 16]
                        if prefixes:
                            _causality_check(
                                "RecurrentFLARE",
                                lambda q, k, v, scale: RecurrentFLARE.apply(q, k, v, scale),
                                Q,
                                K,
                                V,
                                scale,
                                prefixes,
                                atol,
                                token_dim=2,
                            )
    if strict and failures:
        summary = "\n".join(f"- {msg}" for msg in failures[:12])
        extra = "" if len(failures) <= 12 else f"\n- ... and {len(failures) - 12} more"
        raise AssertionError(f"FLARE correctness validation failed ({len(failures)} issues):\n{summary}{extra}")

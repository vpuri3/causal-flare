"""Extracted regression/stress suite implementation."""

from testing.suite_runners.common import *


def _sharp_softmax_bwd_regression_suite():
    if not torch.cuda.is_available():
        print("[FLARE SHARP BWD] CUDA not available, skipping.")
        return

    device = torch.device("cuda")
    dtype_name = os.environ.get("FLARE_SHARP_BWD_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name)
    seed = int(os.environ.get("FLARE_SHARP_BWD_SEED", "0"))
    strict = _strict_mode_enabled("FLARE_SHARP_BWD_STRICT", default=True)
    precisions = _precision_modes_from_env("FLARE_SHARP_BWD_PRECISIONS")
    configs = _parse_bhmnd_configs(
        "FLARE_SHARP_BWD_CONFIGS",
        "1,16,16,512,64;1,16,16,1024,64;1,16,16,2048,64",
    )
    stress_configs = _parse_bhmnd_configs(
        "FLARE_SHARP_BWD_STRESS_CONFIGS",
        "1,16,64,2048,32",
    )
    qk_stds = [float(x.strip()) for x in os.environ.get("FLARE_SHARP_BWD_QK_STDS", "1.0,2.0,4.0,8.0").split(",") if x.strip()]
    stress_qk_stds = [float(x.strip()) for x in os.environ.get("FLARE_SHARP_BWD_STRESS_QK_STDS", "4.0,8.0").split(",") if x.strip()]
    kernel_mode = os.environ.get("FLARE_BWD_PHASE1_KERNEL", "recurrent").strip().lower()
    rel_l2_max = _env_float("FLARE_SHARP_BWD_GRAD_REL_L2_MAX", 5e-3)
    cos_min = _env_float("FLARE_SHARP_BWD_GRAD_COS_MIN", 0.999)
    stress_rel_l2_max = _env_float("FLARE_SHARP_BWD_STRESS_GRAD_REL_L2_MAX", 0.15)
    stress_cos_min = _env_float("FLARE_SHARP_BWD_STRESS_GRAD_COS_MIN", 0.995)
    fwd_max_abs_max = _env_float("FLARE_SHARP_BWD_FWD_MAX_ABS_MAX", 2e-2)
    decode_modes = _parse_decode_separation_modes("FLARE_SHARP_BWD_DECODE_SEPARATION_MODES")
    expect_fail = _strict_mode_enabled("FLARE_SHARP_BWD_EXPECT_FAIL", default=False)
    failures: list[str] = []
    total_cases = 0
    failed_cases = 0

    print("=" * 100)
    print("[FLARE SHARP BWD] Sharp-softmax backward regression suite")
    print("=" * 100)
    print(
        f"[FLARE SHARP BWD] kernel_mode={kernel_mode or 'recurrent'} dtype={dtype} "
        f"qk_stds={qk_stds} configs={len(configs)} decode_modes={decode_modes}"
    )
    print("[FLARE SHARP BWD] baseline=flare_causal_reference (fp32 by default; override via FLARE_REFERENCE_FP32)")

    def _run_one_case(
        *,
        precision: str,
        B: int,
        H: int,
        M: int,
        N: int,
        D: int,
        qk_std: float,
        scale: float,
        separate_q_dec: bool,
        separate_k_dec: bool,
        stress: bool,
    ) -> None:
        nonlocal total_cases, failed_cases
        total_cases += 1
        decode_label = _decode_mode_label(separate_q_dec, separate_k_dec)

        seed_offset = 100_000 if stress else 0
        torch.manual_seed(seed + seed_offset + B + H + M + N + D + int(100 * qk_std))
        Q = qk_std * torch.randn(H, M, D, device=device, dtype=dtype)
        K = qk_std * torch.randn(B, N, H, D, device=device, dtype=dtype)
        V = torch.randn(B, N, H, D, device=device, dtype=dtype)
        g = torch.randn(B, N, H, D, device=device, dtype=dtype)
        Q_dec_base = qk_std * torch.randn(B, N, H, D, device=device, dtype=dtype)
        K_dec_base = qk_std * torch.randn(H, M, D, device=device, dtype=dtype)

        Qn = Q.detach().clone().requires_grad_(True)
        Kn = K.detach().clone().requires_grad_(True)
        Vn = V.detach().clone().requires_grad_(True)
        Qr = Q.detach().clone().requires_grad_(True)
        Kr = K.detach().clone().requires_grad_(True)
        Vr = V.detach().clone().requires_grad_(True)
        Qdn = Q_dec_base.detach().clone().requires_grad_(True) if separate_q_dec else None
        Kdn = K_dec_base.detach().clone().requires_grad_(True) if separate_k_dec else None
        Qdr = Q_dec_base.detach().clone().requires_grad_(True) if separate_q_dec else None
        Kdr = K_dec_base.detach().clone().requires_grad_(True) if separate_k_dec else None

        Yn = flare_chunk_triton(Qn, Kn, Vn, scale, None, None, False, Qdn, Kdn)
        Yr = flare_causal_reference(Qr, Kr, Vr, Q_dec=Qdr, K_dec=Kdr, scale=scale)
        (Yn.float() * g.float()).sum().backward()
        (Yr.float() * g.float()).sum().backward()

        finite_new = bool(
            torch.isfinite(Yn).all()
            and torch.isfinite(Qn.grad).all()
            and torch.isfinite(Kn.grad).all()
            and torch.isfinite(Vn.grad).all()
            and (True if Qdn is None else torch.isfinite(Qdn.grad).all())
            and (True if Kdn is None else torch.isfinite(Kdn.grad).all())
        )
        finite_ref = bool(
            torch.isfinite(Yr).all()
            and torch.isfinite(Qr.grad).all()
            and torch.isfinite(Kr.grad).all()
            and torch.isfinite(Vr.grad).all()
            and (True if Qdr is None else torch.isfinite(Qdr.grad).all())
            and (True if Kdr is None else torch.isfinite(Kdr.grad).all())
        )

        fwd_err = compute_errors(Yn, Yr, "sharp_bwd_fwd")
        grad_specs = [
            ("dQ", Qn.grad, Qr.grad, "sharp_bwd_dq"),
            ("dK", Kn.grad, Kr.grad, "sharp_bwd_dk"),
            ("dV", Vn.grad, Vr.grad, "sharp_bwd_dv"),
        ]
        if separate_q_dec:
            grad_specs.append(("dQ_dec", Qdn.grad, Qdr.grad, "sharp_bwd_dq_dec"))
        if separate_k_dec:
            grad_specs.append(("dK_dec", Kdn.grad, Kdr.grad, "sharp_bwd_dk_dec"))

        grad_metrics = []
        for name, g_new, g_ref, prefix in grad_specs:
            err = compute_errors(g_new, g_ref, prefix)
            cos = _tensor_cosine(g_new, g_ref)
            rel_l2 = (g_new.float() - g_ref.float()).norm() / (g_ref.float().norm() + 1e-8)
            grad_metrics.append((name, err, cos, rel_l2.item(), prefix))

        rel_l2_cap = stress_rel_l2_max if stress else rel_l2_max
        cos_cap = stress_cos_min if stress else cos_min
        case_fail = (
            (not finite_new)
            or (not finite_ref)
            or fwd_err["sharp_bwd_fwd_max_abs_err"] > fwd_max_abs_max
            or any(metric[3] > rel_l2_cap for metric in grad_metrics)
            or any(metric[2] < cos_cap for metric in grad_metrics)
        )
        if case_fail:
            failed_cases += 1

        name_blob = "/".join(metric[0] for metric in grad_metrics)
        rel_blob = "/".join(f"{metric[3]:.3e}" for metric in grad_metrics)
        cos_blob = "/".join(f"{metric[2]:.6f}" for metric in grad_metrics)
        max_abs_blob = "/".join(f"{metric[1][f'{metric[4]}_max_abs_err']:.3e}" for metric in grad_metrics)
        prefix = "[FLARE SHARP BWD STRESS]" if stress else "[FLARE SHARP BWD]"
        print(
            f"{prefix} precision={precision} B={B} H={H} M={M} N={N} D={D} qk_std={qk_std:.2f} {decode_label} "
            f"finite(new/ref)={int(finite_new)}/{int(finite_ref)} "
            f"fwd_max_abs={fwd_err['sharp_bwd_fwd_max_abs_err']:.3e} "
            f"grad_names={name_blob} grad_rel_l2={rel_blob} grad_cos={cos_blob} grad_max_abs={max_abs_blob}"
        )

        if strict:
            mode = "stress " if stress else ""
            _maybe_record_failure(
                failures,
                not finite_ref,
                (
                    f"Sharp BWD {mode}non-finite reference tensors "
                    f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, qk_std={qk_std}, {decode_label})"
                ),
            )
            _maybe_record_failure(
                failures,
                not finite_new,
                (
                    f"Sharp BWD {mode}non-finite new tensors "
                    f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, qk_std={qk_std}, {decode_label})"
                ),
            )
            if not stress:
                _maybe_record_failure(
                    failures,
                    fwd_err["sharp_bwd_fwd_max_abs_err"] > fwd_max_abs_max,
                    (
                        f"Sharp BWD fwd drift too high: {fwd_err['sharp_bwd_fwd_max_abs_err']:.3e} > {fwd_max_abs_max:.3e} "
                        f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, qk_std={qk_std}, {decode_label})"
                    ),
                )
            for name, _, cos, rel_l2, _ in grad_metrics:
                _maybe_record_failure(
                    failures,
                    rel_l2 > rel_l2_cap,
                    (
                        f"Sharp BWD {mode}{name} rel_l2 too high: {rel_l2:.3e} > {rel_l2_cap:.3e} "
                        f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, qk_std={qk_std}, {decode_label})"
                    ),
                )
                _maybe_record_failure(
                    failures,
                    cos < cos_cap,
                    (
                        f"Sharp BWD {mode}{name} cosine too low: {cos:.6f} < {cos_cap:.6f} "
                        f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, qk_std={qk_std}, {decode_label})"
                    ),
                )

        del Q, K, V, g, Q_dec_base, K_dec_base
        del Qn, Kn, Vn, Qr, Kr, Vr, Yn, Yr
        if Qdn is not None:
            del Qdn, Qdr
        if Kdn is not None:
            del Kdn, Kdr
        torch.cuda.empty_cache()

    for precision in precisions:
        with _temp_env_var("FLARE_INPUT_PRECISION", precision):
            print(f"[FLARE SHARP BWD] precision={precision}")
            for cfg in configs:
                B = cfg["B"]
                H = cfg["H"]
                M = cfg["M"]
                N = cfg["N"]
                D = cfg["D"]
                scale = D ** -0.5
                for qk_std in qk_stds:
                    for separate_q_dec, separate_k_dec in decode_modes:
                        _run_one_case(
                            precision=precision,
                            B=B,
                            H=H,
                            M=M,
                            N=N,
                            D=D,
                            qk_std=qk_std,
                            scale=scale,
                            separate_q_dec=separate_q_dec,
                            separate_k_dec=separate_k_dec,
                            stress=False,
                        )

    # Stress-only regression cases:
    # focus on non-finite detection + bounded drift under sharp long-context.
    for precision in precisions:
        with _temp_env_var("FLARE_INPUT_PRECISION", precision):
            for cfg in stress_configs:
                B = cfg["B"]
                H = cfg["H"]
                M = cfg["M"]
                N = cfg["N"]
                D = cfg["D"]
                scale = D ** -0.5
                for qk_std in stress_qk_stds:
                    for separate_q_dec, separate_k_dec in decode_modes:
                        _run_one_case(
                            precision=precision,
                            B=B,
                            H=H,
                            M=M,
                            N=N,
                            D=D,
                            qk_std=qk_std,
                            scale=scale,
                            separate_q_dec=separate_q_dec,
                            separate_k_dec=separate_k_dec,
                            stress=True,
                        )

    print(
        f"[FLARE SHARP BWD] summary: failed_cases={failed_cases}/{total_cases} "
        f"(expect_fail={expect_fail})"
    )

    if strict and expect_fail:
        if failed_cases == 0:
            raise AssertionError(
                "FLARE sharp BWD expected failure mode, but no cases failed. "
                "This likely means thresholds/configs are too loose."
            )
        print("[FLARE SHARP BWD] expected-failure mode: failure observed as expected.")
        return

    if strict and failures:
        summary = "\n".join(f"- {msg}" for msg in failures[:12])
        extra = "" if len(failures) <= 12 else f"\n- ... and {len(failures) - 12} more"
        raise AssertionError(f"FLARE sharp backward regression failed ({len(failures)} issues):\n{summary}{extra}")
    print("[FLARE SHARP BWD] all checks passed.")

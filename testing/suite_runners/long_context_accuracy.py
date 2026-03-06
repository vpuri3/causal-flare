"""Extracted regression/stress suite implementation."""

from testing.suite_runners.common import *


def _long_context_accuracy_suite():
    if not torch.cuda.is_available():
        print("[FLARE LONGCTX] CUDA not available, skipping.")
        return

    device = torch.device("cuda")
    dtype_name = os.environ.get("FLARE_LONGCTX_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name)
    seed = int(os.environ.get("FLARE_LONGCTX_SEED", "0"))
    strict = _strict_mode_enabled("FLARE_LONGCTX_STRICT", default=True)
    precisions = _precision_modes_from_env("FLARE_LONGCTX_PRECISIONS")
    configs = _parse_bhmnd_configs(
        "FLARE_LONGCTX_CONFIGS",
        "1,8,128,8192,32;1,8,128,16384,32",
    )

    fwd_ref_margin_max = _env_float("FLARE_LONGCTX_FWD_REF_MARGIN_MAX", 5e-4)
    fwd_new_old_max_abs_max = _env_float("FLARE_LONGCTX_FWD_NEW_OLD_MAX_ABS_MAX", 8e-3)
    grad_new_old_max_abs_max = _env_float("FLARE_LONGCTX_GRAD_NEW_OLD_MAX_ABS_MAX", 2e-2)
    grad_new_old_cos_min = _env_float("FLARE_LONGCTX_GRAD_NEW_OLD_COS_MIN", 0.999)
    failures: list[str] = []

    print("=" * 100)
    print("[FLARE LONGCTX] Long-context accuracy suite")
    print("=" * 100)
    for precision in precisions:
        with _temp_env_var("FLARE_INPUT_PRECISION", precision):
            print(f"[FLARE LONGCTX] precision={precision} dtype={dtype}")
            for cfg in configs:
                B = cfg["B"]
                H = cfg["H"]
                M = cfg["M"]
                N = cfg["N"]
                D = cfg["D"]
                scale = D ** -0.5
                torch.manual_seed(seed + B + H + M + N + D)

                Q = torch.randn(H, M, D, device=device, dtype=dtype)
                K = torch.randn(B, N, H, D, device=device, dtype=dtype)
                V = torch.randn(B, N, H, D, device=device, dtype=dtype)

                Qn = Q.detach().clone().requires_grad_(True)
                Kn = K.detach().clone().requires_grad_(True)
                Vn = V.detach().clone().requires_grad_(True)
                Qo = Q.detach().clone().requires_grad_(True)
                Ko = K.detach().clone().requires_grad_(True)
                Vo = V.detach().clone().requires_grad_(True)
                Qr = Q.detach().clone().float().requires_grad_(True)
                Kr = K.detach().clone().float().requires_grad_(True)
                Vr = V.detach().clone().float().requires_grad_(True)

                try:
                    Yn = flare_chunk_triton(Qn, Kn, Vn, scale)
                    Yo = flare_causal_chunked(Qo, Ko, Vo, scale=scale, chunk_size=Ko.size(1))
                    Yr = flare_causal_pytorch_dense(Qr, Kr, Vr, scale=scale)

                    torch.manual_seed(seed + 77 + B + H + M + N + D)
                    g = torch.randn_like(Yn)
                    (Yn.float() * g.float()).sum().backward()
                    (Yo.float() * g.float()).sum().backward()
                    (Yr.float() * g.float()).sum().backward()
                except RuntimeError as exc:
                    msg = (
                        f"[FLARE LONGCTX] runtime failure precision={precision} "
                        f"B={B} H={H} M={M} N={N} D={D}: {exc}"
                    )
                    print(msg)
                    if strict:
                        failures.append(msg)
                    continue

                fwd_new_ref = compute_errors(Yn, Yr, "longctx_fwd_new_ref")
                fwd_old_ref = compute_errors(Yo, Yr, "longctx_fwd_old_ref")
                fwd_new_old = compute_errors(Yn, Yo, "longctx_fwd_new_old")
                dq_new_old = compute_errors(Qn.grad, Qo.grad, "longctx_dq_new_old")
                dk_new_old = compute_errors(Kn.grad, Ko.grad, "longctx_dk_new_old")
                dv_new_old = compute_errors(Vn.grad, Vo.grad, "longctx_dv_new_old")
                dq_cos = _tensor_cosine(Qn.grad, Qo.grad)
                dk_cos = _tensor_cosine(Kn.grad, Ko.grad)
                dv_cos = _tensor_cosine(Vn.grad, Vo.grad)

                print(
                    "[FLARE LONGCTX] "
                    f"B={B} H={H} M={M} N={N} D={D} "
                    f"fwd(new-ref/old-ref/new-old) max_abs="
                    f"{fwd_new_ref['longctx_fwd_new_ref_max_abs_err']:.3e}/"
                    f"{fwd_old_ref['longctx_fwd_old_ref_max_abs_err']:.3e}/"
                    f"{fwd_new_old['longctx_fwd_new_old_max_abs_err']:.3e} "
                    f"grad(new-old) max_abs dQ/dK/dV="
                    f"{dq_new_old['longctx_dq_new_old_max_abs_err']:.3e}/"
                    f"{dk_new_old['longctx_dk_new_old_max_abs_err']:.3e}/"
                    f"{dv_new_old['longctx_dv_new_old_max_abs_err']:.3e} "
                    f"cos dQ/dK/dV={dq_cos:.6f}/{dk_cos:.6f}/{dv_cos:.6f}"
                )

                if strict:
                    new_ref = fwd_new_ref["longctx_fwd_new_ref_max_abs_err"]
                    old_ref = fwd_old_ref["longctx_fwd_old_ref_max_abs_err"]
                    _maybe_record_failure(
                        failures,
                        new_ref > old_ref + fwd_ref_margin_max,
                        (
                            "Longctx new-ref drift over old-ref margin: "
                            f"{new_ref:.3e} > {old_ref:.3e} + {fwd_ref_margin_max:.3e} "
                            f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D})"
                        ),
                    )
                    _maybe_record_failure(
                        failures,
                        fwd_new_old["longctx_fwd_new_old_max_abs_err"] > fwd_new_old_max_abs_max,
                        (
                            "Longctx new-old fwd drift too high: "
                            f"{fwd_new_old['longctx_fwd_new_old_max_abs_err']:.3e} > {fwd_new_old_max_abs_max:.3e} "
                            f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D})"
                        ),
                    )
                    for name, err, c in (
                        ("dQ", dq_new_old["longctx_dq_new_old_max_abs_err"], dq_cos),
                        ("dK", dk_new_old["longctx_dk_new_old_max_abs_err"], dk_cos),
                        ("dV", dv_new_old["longctx_dv_new_old_max_abs_err"], dv_cos),
                    ):
                        _maybe_record_failure(
                            failures,
                            err > grad_new_old_max_abs_max,
                            (
                                f"Longctx {name} new-old max_abs too high: {err:.3e} > {grad_new_old_max_abs_max:.3e} "
                                f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D})"
                            ),
                        )
                        _maybe_record_failure(
                            failures,
                            c < grad_new_old_cos_min,
                            (
                                f"Longctx {name} new-old cosine too low: {c:.6f} < {grad_new_old_cos_min:.6f} "
                                f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D})"
                            ),
                        )

    if strict and failures:
        summary = "\n".join(f"- {msg}" for msg in failures[:12])
        extra = "" if len(failures) <= 12 else f"\n- ... and {len(failures) - 12} more"
        raise AssertionError(f"FLARE long-context validation failed ({len(failures)} issues):\n{summary}{extra}")
    print("[FLARE LONGCTX] all checks passed.")

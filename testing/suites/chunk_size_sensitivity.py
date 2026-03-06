"""Extracted regression/stress suite implementation."""

from testing.suites.common import *


def _chunk_size_sensitivity_suite():
    if not torch.cuda.is_available():
        print("[FLARE CHUNK SENS] CUDA not available, skipping.")
        return

    device = torch.device("cuda")
    dtype_name = os.environ.get("FLARE_CHUNK_SENS_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name)
    seed = int(os.environ.get("FLARE_CHUNK_SENS_SEED", "0"))
    strict = _strict_mode_enabled("FLARE_CHUNK_SENS_STRICT", default=True)
    precisions = _precision_modes_from_env("FLARE_CHUNK_SENS_PRECISIONS")
    configs = _parse_bhmnd_configs(
        "FLARE_CHUNK_SENS_CONFIGS",
        "1,8,128,1024,32;1,8,128,2048,64",
    )
    chunk_sizes = [int(x.strip()) for x in os.environ.get("FLARE_CHUNK_SENS_BWD_CHUNKS", "32,64").split(",") if x.strip()]
    failures: list[str] = []
    strict_runtime = _strict_mode_enabled("FLARE_CHUNK_SENS_STRICT_RUNTIME", default=False)
    fwd_max_abs_max = _env_float("FLARE_CHUNK_SENS_FWD_MAX_ABS_MAX", 8e-3)
    grad_max_abs_max = _env_float("FLARE_CHUNK_SENS_GRAD_MAX_ABS_MAX", 2e-2)
    grad_cos_min = _env_float("FLARE_CHUNK_SENS_GRAD_COS_MIN", 0.999)

    def _run_once(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
        grad_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        qx = q.detach().clone().requires_grad_(True)
        kx = k.detach().clone().requires_grad_(True)
        vx = v.detach().clone().requires_grad_(True)
        y = flare_chunk_triton(qx, kx, vx, scale)
        (y.float() * grad_out.float()).sum().backward()
        return y.detach(), qx.grad.detach(), kx.grad.detach(), vx.grad.detach()

    print("=" * 100)
    print("[FLARE CHUNK SENS] Backward chunk-size sensitivity")
    print("=" * 100)
    for precision in precisions:
        with _temp_env_var("FLARE_INPUT_PRECISION", precision):
            print(f"[FLARE CHUNK SENS] precision={precision} dtype={dtype} chunks={chunk_sizes}")
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
                torch.manual_seed(seed + 10_000 + B + H + M + N + D)
                grad_out = torch.randn(B, N, H, D, device=device, dtype=dtype)

                with _temp_env_var("FLARE_BWD_CHUNK_SIZE", ""):
                    os.environ.pop("FLARE_BWD_CHUNK_SIZE", None)
                    try:
                        y_base, dq_base, dk_base, dv_base = _run_once(Q, K, V, scale, grad_out)
                    except Exception as exc:
                        msg = (
                            f"[FLARE CHUNK SENS] baseline runtime failure precision={precision} "
                            f"B={B} H={H} M={M} N={N} D={D}: {exc}"
                        )
                        print(msg)
                        if strict:
                            failures.append(msg)
                        continue

                for csz in chunk_sizes:
                    with _temp_env_var("FLARE_BWD_CHUNK_SIZE", str(csz)):
                        try:
                            y_c, dq_c, dk_c, dv_c = _run_once(Q, K, V, scale, grad_out)
                        except Exception as exc:
                            msg = (
                                f"[FLARE CHUNK SENS] runtime failure precision={precision} chunk={csz} "
                                f"B={B} H={H} M={M} N={N} D={D}: {exc}"
                            )
                            print(msg)
                            if strict and strict_runtime:
                                failures.append(msg)
                            continue

                    fwd_err = compute_errors(y_c, y_base, "chunk_sens_fwd")
                    dq_err = compute_errors(dq_c, dq_base, "chunk_sens_dq")
                    dk_err = compute_errors(dk_c, dk_base, "chunk_sens_dk")
                    dv_err = compute_errors(dv_c, dv_base, "chunk_sens_dv")
                    dq_cos = _tensor_cosine(dq_c, dq_base)
                    dk_cos = _tensor_cosine(dk_c, dk_base)
                    dv_cos = _tensor_cosine(dv_c, dv_base)
                    print(
                        "[FLARE CHUNK SENS] "
                        f"B={B} H={H} M={M} N={N} D={D} chunk={csz} "
                        f"fwd_max_abs={fwd_err['chunk_sens_fwd_max_abs_err']:.3e} "
                        f"dQ/dK/dV max_abs={dq_err['chunk_sens_dq_max_abs_err']:.3e}/"
                        f"{dk_err['chunk_sens_dk_max_abs_err']:.3e}/{dv_err['chunk_sens_dv_max_abs_err']:.3e} "
                        f"cos={dq_cos:.6f}/{dk_cos:.6f}/{dv_cos:.6f}"
                    )

                    if strict:
                        _maybe_record_failure(
                            failures,
                            fwd_err["chunk_sens_fwd_max_abs_err"] > fwd_max_abs_max,
                            (
                                "Chunk sensitivity fwd max_abs too high: "
                                f"{fwd_err['chunk_sens_fwd_max_abs_err']:.3e} > {fwd_max_abs_max:.3e} "
                                f"(precision={precision}, chunk={csz}, B={B}, H={H}, M={M}, N={N}, D={D})"
                            ),
                        )
                        for name, err, c in (
                            ("dQ", dq_err["chunk_sens_dq_max_abs_err"], dq_cos),
                            ("dK", dk_err["chunk_sens_dk_max_abs_err"], dk_cos),
                            ("dV", dv_err["chunk_sens_dv_max_abs_err"], dv_cos),
                        ):
                            _maybe_record_failure(
                                failures,
                                err > grad_max_abs_max,
                                (
                                    f"Chunk sensitivity {name} max_abs too high: {err:.3e} > {grad_max_abs_max:.3e} "
                                    f"(precision={precision}, chunk={csz}, B={B}, H={H}, M={M}, N={N}, D={D})"
                                ),
                            )
                            _maybe_record_failure(
                                failures,
                                c < grad_cos_min,
                                (
                                    f"Chunk sensitivity {name} cosine too low: {c:.6f} < {grad_cos_min:.6f} "
                                    f"(precision={precision}, chunk={csz}, B={B}, H={H}, M={M}, N={N}, D={D})"
                                ),
                            )

    if strict and failures:
        summary = "\n".join(f"- {msg}" for msg in failures[:12])
        extra = "" if len(failures) <= 12 else f"\n- ... and {len(failures) - 12} more"
        raise AssertionError(f"FLARE chunk sensitivity failed ({len(failures)} issues):\n{summary}{extra}")
    print("[FLARE CHUNK SENS] all checks passed.")


"""Extracted regression/stress suite implementation."""

from testing.suites.common import *


def _trainlike_sanity():
    if not torch.cuda.is_available():
        print("[FLARE DEBUG] CUDA not available, skipping trainlike sanity.")
        return
    device = torch.device("cuda")
    dtype_name = os.environ.get("FLARE_TRAINLIKE_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name)
    scale_mode = os.environ.get("FLARE_TRAINLIKE_SCALE", "sqrt")
    compare = os.environ.get("FLARE_TRAINLIKE_COMPARE", "0") == "1"
    steps = int(os.environ.get("FLARE_TRAINLIKE_STEPS", "4"))
    seed = int(os.environ.get("FLARE_TRAINLIKE_SEED", "0"))
    qkv_std = float(os.environ.get("FLARE_TRAINLIKE_QKV_STD", "1.0"))
    log_thresh = float(os.environ.get("FLARE_TRAINLIKE_LOG_THRESH", "10"))
    head_probe = os.environ.get("FLARE_TRAINLIKE_HEAD_PROBE", "0") == "1"
    verbose = os.environ.get("FLARE_TRAINLIKE_VERBOSE", "0") == "1"
    strict = _strict_mode_enabled("FLARE_TRAINLIKE_STRICT", default=True)
    failures: list[str] = []
    grad_norm_max = _env_float("FLARE_TRAINLIKE_GRAD_NORM_MAX", math.exp(log_thresh))
    cmp_mean_abs_max = _env_float("FLARE_TRAINLIKE_CMP_MEAN_ABS_MAX", 1e-6)
    cmp_max_abs_max = _env_float("FLARE_TRAINLIKE_CMP_MAX_ABS_MAX", 1e-4)

    configs = [
        dict(B=4, H=16, M=128, N=2048, D=32),
        dict(B=8, H=16, M=128, N=2048, D=32),
    ]
    if os.environ.get("FLARE_TRAINLIKE_CONFIGS"):
        configs = []
        for spec in os.environ["FLARE_TRAINLIKE_CONFIGS"].split(";"):
            b, h, m, n, d = (int(x) for x in spec.split(","))
            configs.append(dict(B=b, H=h, M=m, N=n, D=d))

    torch.manual_seed(seed)
    for cfg in configs:
        B = cfg["B"]
        H = cfg["H"]
        M = cfg["M"]
        N = cfg["N"]
        D = cfg["D"]
        scale = (D ** -0.5) if scale_mode == "sqrt" else 1.0
        print("=" * 100)
        print(f"[FLARE TRAINLIKE] B={B} H={H} M={M} N={N} D={D} dtype={dtype} scale={scale:.6g}")

        try:
            q_norms = []
            k_norms = []
            v_norms = []
            for step in range(steps):
                step_seed = seed + 1000 * step + B + H + M + N + D
                torch.manual_seed(step_seed)
                Q = (qkv_std * torch.randn(H, M, D, device=device, dtype=dtype)).requires_grad_()
                K = (qkv_std * torch.randn(B, N, H, D, device=device, dtype=dtype)).requires_grad_()
                V = (qkv_std * torch.randn(B, N, H, D, device=device, dtype=dtype)).requires_grad_()

                Y = flare_chunk_triton(Q, K, V, scale)
                loss = Y.float().pow(2).mean()
                loss.backward()

                q_norm = Q.grad.float().norm().item()
                k_norm = K.grad.float().norm().item()
                v_norm = V.grad.float().norm().item()
                q_norms.append(q_norm)
                k_norms.append(k_norm)
                v_norms.append(v_norm)
                if strict:
                    _maybe_record_failure(
                        failures,
                        (not math.isfinite(q_norm)) or (not math.isfinite(k_norm)) or (not math.isfinite(v_norm)),
                        (
                            f"Trainlike non-finite grad norm at step={step} "
                            f"(dQ={q_norm}, dK={k_norm}, dV={v_norm}) "
                            f"for B={B}, H={H}, M={M}, N={N}, D={D}"
                        ),
                    )
                    _maybe_record_failure(
                        failures,
                        max(q_norm, k_norm, v_norm) > grad_norm_max,
                        (
                            f"Trainlike grad norm too large at step={step}: "
                            f"max={max(q_norm, k_norm, v_norm):.3e} > {grad_norm_max:.3e} "
                            f"for B={B}, H={H}, M={M}, N={N}, D={D}"
                        ),
                    )

                if verbose or max(q_norm, k_norm, v_norm) > math.exp(log_thresh):
                    print(
                        f"[FLARE TRAINLIKE] step={step} grad norms: "
                        f"dQ={q_norm:.6g} dK={k_norm:.6g} dV={v_norm:.6g}"
                    )

                if head_probe:
                    q_head = Q.grad.float().reshape(H, M, D).norm(dim=(1, 2))
                    k_head = K.grad.float().reshape(B * N, H, D).norm(dim=(0, 2))
                    v_head = V.grad.float().reshape(B * N, H, D).norm(dim=(0, 2))
                    print(
                        "[FLARE TRAINLIKE] head grad norms (min/mean/max): "
                        f"dQ {q_head.min().item():.3e}/{q_head.mean().item():.3e}/{q_head.max().item():.3e} | "
                        f"dK {k_head.min().item():.3e}/{k_head.mean().item():.3e}/{k_head.max().item():.3e} | "
                        f"dV {v_head.min().item():.3e}/{v_head.mean().item():.3e}/{v_head.max().item():.3e}"
                    )

                if compare:
                    Q2 = Q.detach().clone().requires_grad_(True)
                    K2 = K.detach().clone().requires_grad_(True)
                    V2 = V.detach().clone().requires_grad_(True)
                    Y2 = flare_causal_chunked(Q2, K2, V2, scale=scale)
                    loss2 = Y2.float().pow(2).mean()
                    loss2.backward()
                    dq_err = compute_errors(Q.grad, Q2.grad, "trainlike_dq")
                    dk_err = compute_errors(K.grad, K2.grad, "trainlike_dk")
                    dv_err = compute_errors(V.grad, V2.grad, "trainlike_dv")
                    print(
                        "[FLARE TRAINLIKE] grad err (mean/max abs): "
                        f"dQ {dq_err['trainlike_dq_mean_abs_err']:.2e}/{dq_err['trainlike_dq_max_abs_err']:.2e} | "
                        f"dK {dk_err['trainlike_dk_mean_abs_err']:.2e}/{dk_err['trainlike_dk_max_abs_err']:.2e} | "
                        f"dV {dv_err['trainlike_dv_mean_abs_err']:.2e}/{dv_err['trainlike_dv_max_abs_err']:.2e}"
                    )
                    if strict:
                        for name, err in (("dQ", dq_err), ("dK", dk_err), ("dV", dv_err)):
                            _maybe_record_failure(
                                failures,
                                err[f"trainlike_{name.lower()}_mean_abs_err"] > cmp_mean_abs_max,
                                (
                                    f"Trainlike compare {name} mean_abs too high at step={step}: "
                                    f"{err[f'trainlike_{name.lower()}_mean_abs_err']:.3e} > {cmp_mean_abs_max:.3e} "
                                    f"for B={B}, H={H}, M={M}, N={N}, D={D}"
                                ),
                            )
                            _maybe_record_failure(
                                failures,
                                err[f"trainlike_{name.lower()}_max_abs_err"] > cmp_max_abs_max,
                                (
                                    f"Trainlike compare {name} max_abs too high at step={step}: "
                                    f"{err[f'trainlike_{name.lower()}_max_abs_err']:.3e} > {cmp_max_abs_max:.3e} "
                                    f"for B={B}, H={H}, M={M}, N={N}, D={D}"
                                ),
                            )

            def _summarize(vals):
                vals_t = torch.tensor(vals, dtype=torch.float32)
                return vals_t.min().item(), vals_t.mean().item(), vals_t.max().item()

            q_min, q_mean, q_max = _summarize(q_norms)
            k_min, k_mean, k_max = _summarize(k_norms)
            v_min, v_mean, v_max = _summarize(v_norms)
            print(
                "[FLARE TRAINLIKE] grad norm summary (min/mean/max): "
                f"dQ {q_min:.3e}/{q_mean:.3e}/{q_max:.3e} | "
                f"dK {k_min:.3e}/{k_mean:.3e}/{k_max:.3e} | "
                f"dV {v_min:.3e}/{v_mean:.3e}/{v_max:.3e}"
            )
        except RuntimeError as exc:
            print(f"[FLARE TRAINLIKE] failed: {exc}")
            if strict:
                failures.append(f"Trainlike runtime failure for B={B}, H={H}, M={M}, N={N}, D={D}: {exc}")
    if strict and failures:
        summary = "\n".join(f"- {msg}" for msg in failures[:12])
        extra = "" if len(failures) <= 12 else f"\n- ... and {len(failures) - 12} more"
        raise AssertionError(f"FLARE trainlike validation failed ({len(failures)} issues):\n{summary}{extra}")


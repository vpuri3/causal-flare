"""Extracted regression/stress suite implementation."""

from testing.suite_runners.common import *


def _trainlike_multistep_parity():
    if not torch.cuda.is_available():
        print("[FLARE TRAINLIKE PARITY] CUDA not available, skipping.")
        return

    device = torch.device("cuda")
    dtype_name = os.environ.get("FLARE_TRAINLIKE_PARITY_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name)
    precisions = _precision_modes_from_env("FLARE_TRAINLIKE_PARITY_PRECISIONS")
    configs = _parse_bhmnd_configs(
        "FLARE_TRAINLIKE_PARITY_CONFIGS",
        "1,8,128,1024,32;1,8,128,2048,64",
    )
    steps = int(os.environ.get("FLARE_TRAINLIKE_PARITY_STEPS", "20"))
    seed = int(os.environ.get("FLARE_TRAINLIKE_PARITY_SEED", "0"))
    lr = float(os.environ.get("FLARE_TRAINLIKE_PARITY_LR", "1e-3"))
    qkv_std = float(os.environ.get("FLARE_TRAINLIKE_PARITY_QKV_STD", "1.0"))
    strict = _strict_mode_enabled("FLARE_TRAINLIKE_PARITY_STRICT", default=True)
    failures: list[str] = []
    loss_delta_max = _env_float("FLARE_TRAINLIKE_PARITY_LOSS_DELTA_MAX", 5e-3)
    grad_cos_min = _env_float("FLARE_TRAINLIKE_PARITY_GRAD_COS_MIN", -1.0)
    output_mean_abs_max = _env_float("FLARE_TRAINLIKE_PARITY_OUTPUT_MEAN_ABS_MAX", 5e-3)
    output_max_abs_max = _env_float("FLARE_TRAINLIKE_PARITY_OUTPUT_MAX_ABS_MAX", 3.0)
    check_grads = _strict_mode_enabled("FLARE_TRAINLIKE_PARITY_CHECK_GRADS", default=False)
    grad_rel_l2_max = _env_float("FLARE_TRAINLIKE_PARITY_GRAD_REL_L2_MAX", 2.0)

    print("=" * 100)
    print("[FLARE TRAINLIKE PARITY] Multi-step new-vs-pytorch2(full-chunk) parity")
    print("=" * 100)
    for precision in precisions:
        with _temp_env_var("FLARE_INPUT_PRECISION", precision):
            print(f"[FLARE TRAINLIKE PARITY] precision={precision} dtype={dtype} steps={steps} lr={lr:g}")
            for cfg in configs:
                B = cfg["B"]
                H = cfg["H"]
                M = cfg["M"]
                N = cfg["N"]
                D = cfg["D"]
                scale = D ** -0.5
                torch.manual_seed(seed + B + H + M + N + D)
                hidden = qkv_std * torch.randn(B, N, H, D, device=device, dtype=dtype)
                latent_q_init = qkv_std * torch.randn(H, M, D, device=device, dtype=dtype)
                wk_init = qkv_std * torch.randn(H, D, D, device=device, dtype=dtype)
                wv_init = qkv_std * torch.randn(H, D, D, device=device, dtype=dtype)

                latent_q_new = torch.nn.Parameter(latent_q_init.detach().clone())
                wk_new = torch.nn.Parameter(wk_init.detach().clone())
                wv_new = torch.nn.Parameter(wv_init.detach().clone())
                latent_q_old = torch.nn.Parameter(latent_q_init.detach().clone())
                wk_old = torch.nn.Parameter(wk_init.detach().clone())
                wv_old = torch.nn.Parameter(wv_init.detach().clone())

                for step in range(steps):
                    for p in (latent_q_new, wk_new, wv_new, latent_q_old, wk_old, wv_old):
                        if p.grad is not None:
                            p.grad = None

                    k_new = torch.einsum("bnhd,hde->bnhe", hidden, wk_new)
                    v_new = torch.einsum("bnhd,hde->bnhe", hidden, wv_new)
                    k_old = torch.einsum("bnhd,hde->bnhe", hidden, wk_old)
                    v_old = torch.einsum("bnhd,hde->bnhe", hidden, wv_old)
                    y_new = flare_autoregressive_triton(latent_q_new, k_new, v_new, scale)
                    y_old = flare_causal_chunked(latent_q_old, k_old, v_old, scale=scale, chunk_size=k_old.size(1))

                    loss_new = y_new.float().pow(2).mean()
                    loss_old = y_old.float().pow(2).mean()

                    loss_new.backward()
                    loss_old.backward()

                    dq_cos = _tensor_cosine(latent_q_new.grad, latent_q_old.grad)
                    dwk_cos = _tensor_cosine(wk_new.grad, wk_old.grad)
                    dwv_cos = _tensor_cosine(wv_new.grad, wv_old.grad)
                    out_err = compute_errors(y_new, y_old, "trainlike_parity_out")
                    lq_err = compute_errors(latent_q_new.grad, latent_q_old.grad, "trainlike_parity_lqg")
                    wk_err = compute_errors(wk_new.grad, wk_old.grad, "trainlike_parity_wkg")
                    wv_err = compute_errors(wv_new.grad, wv_old.grad, "trainlike_parity_wvg")
                    lq_rel_l2 = (latent_q_new.grad.float() - latent_q_old.grad.float()).norm() / (
                        latent_q_old.grad.float().norm() + 1e-8
                    )
                    wk_rel_l2 = (wk_new.grad.float() - wk_old.grad.float()).norm() / (wk_old.grad.float().norm() + 1e-8)
                    wv_rel_l2 = (wv_new.grad.float() - wv_old.grad.float()).norm() / (wv_old.grad.float().norm() + 1e-8)
                    loss_delta = abs(loss_new.item() - loss_old.item())
                    print(
                        f"[FLARE TRAINLIKE PARITY] B={B} H={H} M={M} N={N} D={D} step={step} "
                        f"loss_delta={loss_delta:.3e} "
                        f"out_mean/max_abs={out_err['trainlike_parity_out_mean_abs_err']:.3e}/"
                        f"{out_err['trainlike_parity_out_max_abs_err']:.3e} "
                        f"grad_mean_abs(lq/wk/wv)="
                        f"{lq_err['trainlike_parity_lqg_mean_abs_err']:.3e}/"
                        f"{wk_err['trainlike_parity_wkg_mean_abs_err']:.3e}/"
                        f"{wv_err['trainlike_parity_wvg_mean_abs_err']:.3e} "
                        f"grad_mean_rel(lq/wk/wv)="
                        f"{lq_err['trainlike_parity_lqg_mean_rel_err']:.3e}/"
                        f"{wk_err['trainlike_parity_wkg_mean_rel_err']:.3e}/"
                        f"{wv_err['trainlike_parity_wvg_mean_rel_err']:.3e} "
                        f"grad_rel_l2(lq/wk/wv)={lq_rel_l2.item():.3e}/{wk_rel_l2.item():.3e}/{wv_rel_l2.item():.3e} "
                        f"grad_cos(lq/wk/wv)={dq_cos:.6f}/{dwk_cos:.6f}/{dwv_cos:.6f}"
                    )

                    if strict:
                        _maybe_record_failure(
                            failures,
                            (not math.isfinite(loss_new.item())) or (not math.isfinite(loss_old.item())),
                            (
                                "Trainlike parity non-finite loss: "
                                f"new={loss_new.item()}, old={loss_old.item()} "
                                f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, step={step})"
                            ),
                        )
                        _maybe_record_failure(
                            failures,
                            loss_delta > loss_delta_max,
                            (
                                f"Trainlike parity loss delta too high: {loss_delta:.3e} > {loss_delta_max:.3e} "
                                f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, step={step})"
                            ),
                        )
                        _maybe_record_failure(
                            failures,
                            out_err["trainlike_parity_out_mean_abs_err"] > output_mean_abs_max,
                            (
                                "Trainlike parity output mean_abs too high: "
                                f"{out_err['trainlike_parity_out_mean_abs_err']:.3e} > {output_mean_abs_max:.3e} "
                                f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, step={step})"
                            ),
                        )
                        _maybe_record_failure(
                            failures,
                            out_err["trainlike_parity_out_max_abs_err"] > output_max_abs_max,
                            (
                                "Trainlike parity output max_abs too high: "
                                f"{out_err['trainlike_parity_out_max_abs_err']:.3e} > {output_max_abs_max:.3e} "
                                f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, step={step})"
                            ),
                        )
                        if check_grads:
                            for name, rel_l2 in (("latent_q", lq_rel_l2), ("Wk", wk_rel_l2), ("Wv", wv_rel_l2)):
                                _maybe_record_failure(
                                    failures,
                                    rel_l2.item() > grad_rel_l2_max,
                                    (
                                        f"Trainlike parity grad rel_l2 too high for {name}: "
                                        f"{rel_l2.item():.3e} > {grad_rel_l2_max:.3e} "
                                        f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, step={step})"
                                    ),
                                )
                        if check_grads and grad_cos_min >= 0.0:
                            for name, c in (("latent_q", dq_cos), ("Wk", dwk_cos), ("Wv", dwv_cos)):
                                _maybe_record_failure(
                                    failures,
                                    c < grad_cos_min,
                                    (
                                        f"Trainlike parity grad cosine too low for {name}: {c:.6f} < {grad_cos_min:.6f} "
                                        f"(precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, step={step})"
                                    ),
                                )

                    with torch.no_grad():
                        latent_q_new -= lr * latent_q_new.grad
                        wk_new -= lr * wk_new.grad
                        wv_new -= lr * wv_new.grad
                        latent_q_old -= lr * latent_q_old.grad
                        wk_old -= lr * wk_old.grad
                        wv_old -= lr * wv_old.grad

    if strict and failures:
        summary = "\n".join(f"- {msg}" for msg in failures[:12])
        extra = "" if len(failures) <= 12 else f"\n- ... and {len(failures) - 12} more"
        raise AssertionError(f"FLARE trainlike parity failed ({len(failures)} issues):\n{summary}{extra}")
    print("[FLARE TRAINLIKE PARITY] all checks passed.")

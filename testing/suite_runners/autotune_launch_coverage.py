"""Small autotune correctness coverage for representative launch configs."""

from testing.suite_runners.common import *
from testing.suite_runners.grad_checks import _run_correctness_pytorch2_grads_with_decode


def _autotune_launch_coverage_suite(*, shard_index: int | None = None, num_shards: int | None = None) -> None:
    if not torch.cuda.is_available():
        print("[FLARE AUTOTUNE] CUDA not available, skipping.")
        return

    device = torch.device("cuda")
    dtype_name = os.environ.get("FLARE_AUTOTUNE_COVERAGE_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name)
    seed = int(os.environ.get("FLARE_AUTOTUNE_COVERAGE_SEED", "0"))
    strict = _strict_mode_enabled("FLARE_AUTOTUNE_COVERAGE_STRICT", default=True)
    precisions = _precision_modes_from_env("FLARE_AUTOTUNE_COVERAGE_PRECISIONS", default="ieee")
    configs = _parse_bhmnd_configs(
        "FLARE_AUTOTUNE_COVERAGE_CONFIGS",
        "1,4,16,256,32;1,8,64,1024,64",
    )
    qk_stds = [float(x.strip()) for x in os.environ.get("FLARE_AUTOTUNE_COVERAGE_QK_STDS", "4.0").split(",") if x.strip()]
    decode_modes = _parse_decode_separation_modes("FLARE_AUTOTUNE_COVERAGE_DECODE_SEPARATION_MODES", "11")
    env_num_shards = int(os.environ.get("FLARE_AUTOTUNE_COVERAGE_NUM_SHARDS", "1"))
    env_shard_index = int(os.environ.get("FLARE_AUTOTUNE_COVERAGE_SHARD_INDEX", "0"))
    if num_shards is None:
        num_shards = env_num_shards
    if shard_index is None:
        shard_index = env_shard_index
    if num_shards <= 0:
        raise ValueError(f"FLARE_AUTOTUNE_COVERAGE_NUM_SHARDS must be positive, got {num_shards}.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"FLARE_AUTOTUNE_COVERAGE_SHARD_INDEX must be in [0, {num_shards}), got {shard_index}.")

    fwd_mean_abs_max = _env_float("FLARE_AUTOTUNE_COVERAGE_FWD_MEAN_ABS_MAX", 3e-4 if dtype == torch.bfloat16 else 2e-4)
    fwd_max_abs_max = _env_float("FLARE_AUTOTUNE_COVERAGE_FWD_MAX_ABS_MAX", 3e-2 if dtype == torch.bfloat16 else 2e-2)
    grad_max_abs_max = _env_float("FLARE_AUTOTUNE_COVERAGE_GRAD_MAX_ABS_MAX", 8e-1)
    grad_cos_min = _env_float("FLARE_AUTOTUNE_COVERAGE_GRAD_COS_MIN", 0.999)
    fa_fwd_mult = _env_float("FLARE_AUTOTUNE_COVERAGE_FA_FWD_MULT", 2.0)
    fa_fwd_slack = _env_float("FLARE_AUTOTUNE_COVERAGE_FA_FWD_SLACK", 1e-5)
    fa_grad_mult = _env_float("FLARE_AUTOTUNE_COVERAGE_FA_GRAD_MULT", 5.0)
    fa_grad_slack = _env_float("FLARE_AUTOTUNE_COVERAGE_FA_GRAD_SLACK", 1e-5)
    failures: list[str] = []

    all_cases = []
    for precision in precisions:
        for cfg in configs:
            for qk_std in qk_stds:
                for separate_q_dec, separate_k_dec in decode_modes:
                    all_cases.append((precision, cfg, qk_std, separate_q_dec, separate_k_dec))
    selected_cases = [case for idx, case in enumerate(all_cases) if idx % num_shards == shard_index]

    print("=" * 100)
    print("[FLARE AUTOTUNE] Autotune launch coverage")
    print("=" * 100)
    print(
        f"[FLARE AUTOTUNE] dtype={dtype} shard={shard_index + 1}/{num_shards} "
        f"selected_cases={len(selected_cases)}/{len(all_cases)}"
    )

    for precision, cfg, qk_std, separate_q_dec, separate_k_dec in selected_cases:
        with _temp_env_var("FLARE_INPUT_PRECISION", precision):
            B = cfg["B"]
            H = cfg["H"]
            M = cfg["M"]
            N = cfg["N"]
            D = cfg["D"]
            scale = D ** -0.5
            decode_label = _decode_mode_label(separate_q_dec, separate_k_dec)
            torch.manual_seed(seed + B + H + M + N + D + int(100 * qk_std) + (10 if separate_q_dec else 0) + (20 if separate_k_dec else 0))

            Q = qk_std * torch.randn(H, M, D, device=device, dtype=dtype)
            K = qk_std * torch.randn(B, N, H, D, device=device, dtype=dtype)
            V = torch.randn(B, N, H, D, device=device, dtype=dtype)
            Q_dec = qk_std * torch.randn(B, N, H, D, device=device, dtype=dtype) if separate_q_dec else None
            K_dec = qk_std * torch.randn(H, M, D, device=device, dtype=dtype) if separate_k_dec else None

            Y_ref = _run_correctness_reference(Q, K, V, scale=scale, Q_dec=Q_dec, K_dec=K_dec)
            Y_noise = _run_correctness_pytorch2(
                Q,
                K,
                V,
                scale=scale,
                chunk_size=N,
                allow_tf32=(precision != "ieee"),
                Q_dec=Q_dec,
                K_dec=K_dec,
            )
            grads_noise = _run_correctness_pytorch2_grads_with_decode(
                Q,
                K,
                V,
                scale=scale,
                chunk_size=N,
                allow_tf32=(precision != "ieee"),
                Q_dec=Q_dec,
                K_dec=K_dec,
            )

            Q_t3 = Q.detach().clone().requires_grad_(True)
            K_t3 = K.detach().clone().requires_grad_(True)
            V_t3 = V.detach().clone().requires_grad_(True)
            Q_dec_t3 = Q_dec.detach().clone().requires_grad_(True) if Q_dec is not None else None
            K_dec_t3 = K_dec.detach().clone().requires_grad_(True) if K_dec is not None else None
            Q_ref = Q.detach().clone().requires_grad_(True)
            K_ref = K.detach().clone().requires_grad_(True)
            V_ref = V.detach().clone().requires_grad_(True)
            Q_dec_ref = Q_dec.detach().clone().requires_grad_(True) if Q_dec is not None else None
            K_dec_ref = K_dec.detach().clone().requires_grad_(True) if K_dec is not None else None

            Y_t3 = flare_chunk_triton(Q_t3, K_t3, V_t3, scale, None, None, False, Q_dec_t3, K_dec_t3)
            with _temp_env_var("FLARE_REFERENCE_FP32", "1"):
                with _scoped_float32_math_mode(allow_tf32=False):
                    Y_ref_grad = flare_causal_reference(
                        Q_ref, K_ref, V_ref, Q_dec=Q_dec_ref, K_dec=K_dec_ref, scale=scale
                    )

            torch.manual_seed(seed + 99 + B + H + M + N + D)
            g = torch.randn_like(Y_t3)
            (Y_t3.float() * g.float()).sum().backward()
            (Y_ref_grad.float() * g.float()).sum().backward()

            y_err = compute_errors(Y_t3, Y_ref, "autotune_fwd")
            y_noise_err = compute_errors(Y_noise, Y_ref, "autotune_noise_fwd")
            grad_specs = [
                ("dQ", Q_t3.grad, Q_ref.grad, "autotune_dq"),
                ("dK", K_t3.grad, K_ref.grad, "autotune_dk"),
                ("dV", V_t3.grad, V_ref.grad, "autotune_dv"),
            ]
            if separate_q_dec:
                grad_specs.append(("dQ_dec", Q_dec_t3.grad, Q_dec_ref.grad, "autotune_dq_dec"))
            if separate_k_dec:
                grad_specs.append(("dK_dec", K_dec_t3.grad, K_dec_ref.grad, "autotune_dk_dec"))

            finite_new = bool(
                torch.isfinite(Y_t3).all()
                and all(torch.isfinite(grad).all() for _, grad, _, _ in grad_specs)
            )
            finite_ref = bool(
                torch.isfinite(Y_ref).all()
                and all(torch.isfinite(ref_grad).all() for _, _, ref_grad, _ in grad_specs)
            )
            print(
                f"[FLARE AUTOTUNE] precision={precision} B={B} H={H} M={M} N={N} D={D} qk_std={qk_std:.2f} {decode_label} "
                f"finite(new/ref)={int(finite_new)}/{int(finite_ref)} "
                f"fwd_max_abs={y_err['autotune_fwd_max_abs_err']:.3e}"
            )

            if strict:
                _maybe_record_failure(
                    failures,
                    not finite_new,
                    f"Autotune coverage non-finite new tensors (precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, qk_std={qk_std}, {decode_label})",
                )
                _maybe_record_failure(
                    failures,
                    not finite_ref,
                    f"Autotune coverage non-finite reference tensors (precision={precision}, B={B}, H={H}, M={M}, N={N}, D={D}, qk_std={qk_std}, {decode_label})",
                )
                if precision == "ieee":
                    # BF16/FP16 inputs can have a noise floor above the historical
                    # absolute gate even when Triton matches the PyTorch chunked
                    # implementation exactly. Keep the hard threshold as a floor,
                    # but do not require accuracy materially better than the noise model.
                    fwd_mean_abs_limit = fwd_mean_abs_max
                    fwd_max_abs_limit = fwd_max_abs_max
                    if dtype in (torch.bfloat16, torch.float16):
                        fwd_mean_abs_limit = max(
                            fwd_mean_abs_limit,
                            _scaled_error_limit(
                                y_noise_err["autotune_noise_fwd_mean_abs_err"],
                                fa_fwd_mult,
                                fa_fwd_slack,
                            ),
                        )
                        fwd_max_abs_limit = max(
                            fwd_max_abs_limit,
                            _scaled_error_limit(
                                y_noise_err["autotune_noise_fwd_max_abs_err"],
                                fa_fwd_mult,
                                fa_fwd_slack,
                            ),
                        )
                    _maybe_record_failure(
                        failures,
                        y_err["autotune_fwd_mean_abs_err"] > fwd_mean_abs_limit,
                        f"Autotune coverage fwd mean_abs too high: {y_err['autotune_fwd_mean_abs_err']:.3e} > {fwd_mean_abs_limit:.3e}",
                    )
                    _maybe_record_failure(
                        failures,
                        y_err["autotune_fwd_max_abs_err"] > fwd_max_abs_limit,
                        f"Autotune coverage fwd max_abs too high: {y_err['autotune_fwd_max_abs_err']:.3e} > {fwd_max_abs_limit:.3e}",
                    )
                else:
                    fwd_limit = _scaled_error_limit(y_noise_err["autotune_noise_fwd_max_abs_err"], fa_fwd_mult, fa_fwd_slack)
                    _maybe_record_failure(
                        failures,
                        y_err["autotune_fwd_max_abs_err"] > fwd_limit,
                        f"Autotune coverage fwd max_abs too high: {y_err['autotune_fwd_max_abs_err']:.3e} > {fwd_limit:.3e}",
                    )

                for grad_name, grad_new, grad_ref, prefix in grad_specs:
                    err = compute_errors(grad_new, grad_ref, prefix)
                    cos = _tensor_cosine(grad_new, grad_ref)
                    if precision == "ieee":
                        _maybe_record_failure(
                            failures,
                            err[f"{prefix}_max_abs_err"] > grad_max_abs_max,
                            f"Autotune coverage {grad_name} max_abs too high: {err[f'{prefix}_max_abs_err']:.3e} > {grad_max_abs_max:.3e}",
                        )
                    else:
                        noise_key = {
                            "autotune_dq": "q",
                            "autotune_dk": "k",
                            "autotune_dv": "v",
                            "autotune_dq_dec": "q_dec",
                            "autotune_dk_dec": "k_dec",
                        }[prefix]
                        noise_err = compute_errors(grads_noise[noise_key], grad_ref, f"{prefix}_noise")
                        grad_limit = _scaled_error_limit(
                            noise_err[f"{prefix}_noise_max_abs_err"],
                            fa_grad_mult,
                            fa_grad_slack,
                        )
                        _maybe_record_failure(
                            failures,
                            err[f"{prefix}_max_abs_err"] > grad_limit,
                            f"Autotune coverage {grad_name} max_abs too high: {err[f'{prefix}_max_abs_err']:.3e} > {grad_limit:.3e}",
                        )
                    _maybe_record_failure(
                        failures,
                        cos < grad_cos_min,
                        f"Autotune coverage {grad_name} cosine too low: {cos:.6f} < {grad_cos_min:.6f}",
                    )

    if strict and failures:
        summary = "\n".join(f"- {msg}" for msg in failures[:12])
        extra = "" if len(failures) <= 12 else f"\n- ... and {len(failures) - 12} more"
        raise AssertionError(f"FLARE autotune launch coverage failed ({len(failures)} issues):\n{summary}{extra}")
    print("[FLARE AUTOTUNE] all checks passed.")


def _autotune_launch_coverage_suite_shard(shard_index: int, num_shards: int) -> None:
    _autotune_launch_coverage_suite(shard_index=shard_index, num_shards=num_shards)

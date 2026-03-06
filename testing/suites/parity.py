"""Extracted regression/stress suite implementation."""

from testing.suites.common import *


def _parity_tests():
    if not torch.cuda.is_available():
        print("[FLARE DEBUG] CUDA not available, skipping parity tests.")
        return

    device = torch.device("cuda")
    dtype_name = os.environ.get("FLARE_PARITY_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name)
    B = int(os.environ.get("FLARE_PARITY_B", "1"))
    H = int(os.environ.get("FLARE_PARITY_H", "8"))
    M = int(os.environ.get("FLARE_PARITY_M", "128"))
    N = int(os.environ.get("FLARE_PARITY_N", "2048"))
    D = int(os.environ.get("FLARE_PARITY_D", "32"))
    seed = int(os.environ.get("FLARE_PARITY_SEED", "0"))
    scale_mode = os.environ.get("FLARE_PARITY_SCALE", "none")
    scale = (D ** -0.5) if scale_mode == "sqrt" else 1.0
    strict = _strict_mode_enabled("FLARE_PARITY_STRICT", default=True)
    failures: list[str] = []
    fwd_mean_abs_max = _env_float("FLARE_PARITY_FWD_MEAN_ABS_MAX", 3e-4)
    fwd_max_abs_max = _env_float("FLARE_PARITY_FWD_MAX_ABS_MAX", 3e-2)
    q_grad_mean_abs_max = _env_float("FLARE_PARITY_Q_GRAD_MEAN_ABS_MAX", 3e-3)
    k_grad_mean_abs_max = _env_float("FLARE_PARITY_K_GRAD_MEAN_ABS_MAX", 2e-3)
    v_grad_mean_abs_max = _env_float("FLARE_PARITY_V_GRAD_MEAN_ABS_MAX", 5e-4)
    grad_cos_min = _env_float("FLARE_PARITY_GRAD_COS_MIN", 0.999)

    torch.manual_seed(seed)
    Q = torch.randn(H, M, D, device=device, dtype=dtype)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype)

    print("=" * 100)
    print("[FLARE PARITY] Forward/Backward parity checks")
    print("=" * 100)
    print(
        f"[FLARE PARITY] B={B} H={H} M={M} N={N} D={D} dtype={dtype} "
        f"scale={scale:.6g}"
    )

    with torch.no_grad():
        Y_ref = flare_causal_chunked(Q, K, V, scale=scale)
        Y_t3 = flare_chunk_triton(Q, K, V, scale)
    fwd_err = compute_errors(Y_t3, Y_ref, "triton_fwd")
    print(
        "[FLARE PARITY] fwd "
        f"mean_abs={fwd_err['triton_fwd_mean_abs_err']:.3e} "
        f"max_abs={fwd_err['triton_fwd_max_abs_err']:.3e} "
        f"mean_rel={fwd_err['triton_fwd_mean_rel_err']:.3e} "
        f"max_rel={fwd_err['triton_fwd_max_rel_err']:.3e} "
        f"allclose={fwd_err['triton_fwd_allclose']}"
    )
    if strict:
        _maybe_record_failure(
            failures,
            fwd_err["triton_fwd_mean_abs_err"] > fwd_mean_abs_max,
            f"Parity fwd mean_abs too high: {fwd_err['triton_fwd_mean_abs_err']:.3e} > {fwd_mean_abs_max:.3e}",
        )
        _maybe_record_failure(
            failures,
            fwd_err["triton_fwd_max_abs_err"] > fwd_max_abs_max,
            f"Parity fwd max_abs too high: {fwd_err['triton_fwd_max_abs_err']:.3e} > {fwd_max_abs_max:.3e}",
        )

    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    Q_t3 = Q.detach().clone().requires_grad_(True)
    K_t3 = K.detach().clone().requires_grad_(True)
    V_t3 = V.detach().clone().requires_grad_(True)

    Y_ref = flare_causal_chunked(Q_ref, K_ref, V_ref, scale=scale)
    Y_t3 = flare_chunk_triton(Q_t3, K_t3, V_t3, scale)
    Y_ref.sum().backward()
    Y_t3.sum().backward()

    q_err = compute_errors(Q_t3.grad, Q_ref.grad, "triton_qg")
    k_err = compute_errors(K_t3.grad, K_ref.grad, "triton_kg")
    v_err = compute_errors(V_t3.grad, V_ref.grad, "triton_vg")
    q_cos = _tensor_cosine(Q_t3.grad, Q_ref.grad)
    k_cos = _tensor_cosine(K_t3.grad, K_ref.grad)
    v_cos = _tensor_cosine(V_t3.grad, V_ref.grad)

    print(
        "[FLARE PARITY] grad Q "
        f"mean_abs={q_err['triton_qg_mean_abs_err']:.3e} "
        f"max_abs={q_err['triton_qg_max_abs_err']:.3e} "
        f"mean_rel={q_err['triton_qg_mean_rel_err']:.3e} "
        f"max_rel={q_err['triton_qg_max_rel_err']:.3e} "
        f"cos={q_cos:.5f}"
    )
    print(
        "[FLARE PARITY] grad K "
        f"mean_abs={k_err['triton_kg_mean_abs_err']:.3e} "
        f"max_abs={k_err['triton_kg_max_abs_err']:.3e} "
        f"mean_rel={k_err['triton_kg_mean_rel_err']:.3e} "
        f"max_rel={k_err['triton_kg_max_rel_err']:.3e} "
        f"cos={k_cos:.5f}"
    )
    print(
        "[FLARE PARITY] grad V "
        f"mean_abs={v_err['triton_vg_mean_abs_err']:.3e} "
        f"max_abs={v_err['triton_vg_max_abs_err']:.3e} "
        f"mean_rel={v_err['triton_vg_mean_rel_err']:.3e} "
        f"max_rel={v_err['triton_vg_max_rel_err']:.3e} "
        f"cos={v_cos:.5f}"
    )
    if strict:
        _maybe_record_failure(
            failures,
            q_err["triton_qg_mean_abs_err"] > q_grad_mean_abs_max,
            f"Parity dQ mean_abs too high: {q_err['triton_qg_mean_abs_err']:.3e} > {q_grad_mean_abs_max:.3e}",
        )
        _maybe_record_failure(
            failures,
            k_err["triton_kg_mean_abs_err"] > k_grad_mean_abs_max,
            f"Parity dK mean_abs too high: {k_err['triton_kg_mean_abs_err']:.3e} > {k_grad_mean_abs_max:.3e}",
        )
        _maybe_record_failure(
            failures,
            v_err["triton_vg_mean_abs_err"] > v_grad_mean_abs_max,
            f"Parity dV mean_abs too high: {v_err['triton_vg_mean_abs_err']:.3e} > {v_grad_mean_abs_max:.3e}",
        )
        _maybe_record_failure(
            failures,
            min(q_cos, k_cos, v_cos) < grad_cos_min,
            f"Parity grad cosine too low: min={min(q_cos, k_cos, v_cos):.6f} < {grad_cos_min:.6f}",
        )

    steps = int(os.environ.get("FLARE_PARITY_TRAIN_STEPS", "5"))
    lr = float(os.environ.get("FLARE_PARITY_LR", "1e-2"))
    qkv_std = float(os.environ.get("FLARE_PARITY_QKV_STD", "1.0"))
    if steps <= 0:
        return

    torch.manual_seed(seed + 123)
    hidden = qkv_std * torch.randn(B, N, H, D, device=device, dtype=dtype)
    latent_q_ref = torch.nn.Parameter(qkv_std * torch.randn(H, M, D, device=device, dtype=dtype))
    Wk_ref = torch.nn.Parameter(qkv_std * torch.randn(H, D, D, device=device, dtype=dtype))
    Wv_ref = torch.nn.Parameter(qkv_std * torch.randn(H, D, D, device=device, dtype=dtype))

    latent_q_t3 = torch.nn.Parameter(latent_q_ref.detach().clone())
    Wk_t3 = torch.nn.Parameter(Wk_ref.detach().clone())
    Wv_t3 = torch.nn.Parameter(Wv_ref.detach().clone())

    print("=" * 100)
    print("[FLARE PARITY] Short training parity")
    print("=" * 100)
    for step in range(steps):
        for p in (latent_q_ref, Wk_ref, Wv_ref):
            if p.grad is not None:
                p.grad = None
        for p in (latent_q_t3, Wk_t3, Wv_t3):
            if p.grad is not None:
                p.grad = None

        K_ref = torch.einsum("bnhd,hde->bnhe", hidden, Wk_ref)
        V_ref = torch.einsum("bnhd,hde->bnhe", hidden, Wv_ref)
        K_t3 = torch.einsum("bnhd,hde->bnhe", hidden, Wk_t3)
        V_t3 = torch.einsum("bnhd,hde->bnhe", hidden, Wv_t3)

        Y_ref = flare_causal_chunked(latent_q_ref, K_ref, V_ref, scale=scale)
        Y_t3 = flare_chunk_triton(latent_q_t3, K_t3, V_t3, scale)

        loss_ref = Y_ref.float().pow(2).mean()
        loss_t3 = Y_t3.float().pow(2).mean()
        loss_ref.backward()
        loss_t3.backward()

        g_lq_cos = _tensor_cosine(latent_q_t3.grad, latent_q_ref.grad)
        g_wk_cos = _tensor_cosine(Wk_t3.grad, Wk_ref.grad)
        g_wv_cos = _tensor_cosine(Wv_t3.grad, Wv_ref.grad)

        print(
            f"[FLARE PARITY][step {step}] "
            f"loss_ref={loss_ref.item():.6e} loss_t3={loss_t3.item():.6e} "
            f"delta={abs(loss_t3.item() - loss_ref.item()):.3e} "
            f"cos(lq)={g_lq_cos:.4f} cos(Wk)={g_wk_cos:.4f} cos(Wv)={g_wv_cos:.4f}"
        )
        if strict and _strict_mode_enabled("FLARE_PARITY_STRICT_TRAIN", default=False):
            loss_delta_max = _env_float("FLARE_PARITY_TRAIN_LOSS_DELTA_MAX", 50.0)
            train_cos_min = _env_float("FLARE_PARITY_TRAIN_GRAD_COS_MIN", 0.20)
            _maybe_record_failure(
                failures,
                abs(loss_t3.item() - loss_ref.item()) > loss_delta_max,
                (
                    f"Parity train loss delta too high at step={step}: "
                    f"{abs(loss_t3.item() - loss_ref.item()):.3e} > {loss_delta_max:.3e}"
                ),
            )
            _maybe_record_failure(
                failures,
                min(g_lq_cos, g_wk_cos, g_wv_cos) < train_cos_min,
                (
                    f"Parity train grad cosine too low at step={step}: "
                    f"min={min(g_lq_cos, g_wk_cos, g_wv_cos):.4f} < {train_cos_min:.4f}"
                ),
            )

        with torch.no_grad():
            latent_q_ref -= lr * latent_q_ref.grad
            Wk_ref -= lr * Wk_ref.grad
            Wv_ref -= lr * Wv_ref.grad
            latent_q_t3 -= lr * latent_q_t3.grad
            Wk_t3 -= lr * Wk_t3.grad
            Wv_t3 -= lr * Wv_t3.grad
    if strict and failures:
        summary = "\n".join(f"- {msg}" for msg in failures[:12])
        extra = "" if len(failures) <= 12 else f"\n- ... and {len(failures) - 12} more"
        raise AssertionError(f"FLARE parity validation failed ({len(failures)} issues):\n{summary}{extra}")


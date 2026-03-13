import torch

from causal_flare.autoregressive.reference import flare_causal_chunked, flare_causal_reference, flare_recurrent_pytorch


def _run_impl(name, q_enc, k_enc, v_enc, *, q_dec=None, k_dec=None, scale=None):
    if name == "reference":
        return flare_causal_reference(q_enc, k_enc, v_enc, Q_dec=q_dec, K_dec=k_dec, scale=scale)
    if name == "recurrent":
        y = flare_recurrent_pytorch(q_enc, k_enc, v_enc, scale=scale, Q_dec=q_dec, K_dec=k_dec)
        return y.permute(0, 2, 1, 3)
    if name == "pytorch2":
        return flare_causal_chunked(
            q_enc,
            k_enc,
            v_enc,
            scale=scale,
            chunk_size=k_enc.size(1),
            Q_dec=q_dec,
            K_dec=k_dec,
        )
    raise ValueError(f"Unknown implementation: {name}")


def _run_impl_with_grads(
    name,
    q_enc,
    k_enc,
    v_enc,
    *,
    q_dec_mode,
    k_dec_mode,
    scale,
    grad_out,
    q_dec_rand_seed,
    k_dec_rand_seed,
):
    q_enc_i = q_enc.clone().requires_grad_(True)
    k_enc_i = k_enc.clone().requires_grad_(True)
    v_enc_i = v_enc.clone().requires_grad_(True)

    q_dec_rand_i = None
    if q_dec_mode == "rand":
        q_dec_rand_i = q_dec_rand_seed.clone().requires_grad_(True)
        q_dec_i = q_dec_rand_i
    elif q_dec_mode == "k_enc":
        q_dec_i = k_enc_i
    elif q_dec_mode == "none":
        q_dec_i = None
    else:
        raise ValueError(f"Unknown q_dec_mode: {q_dec_mode}")

    k_dec_rand_i = None
    if k_dec_mode == "rand":
        k_dec_rand_i = k_dec_rand_seed.clone().requires_grad_(True)
        k_dec_i = k_dec_rand_i
    elif k_dec_mode == "q_enc":
        k_dec_i = q_enc_i
    elif k_dec_mode == "none":
        k_dec_i = None
    else:
        raise ValueError(f"Unknown k_dec_mode: {k_dec_mode}")

    y = _run_impl(name, q_enc_i, k_enc_i, v_enc_i, q_dec=q_dec_i, k_dec=k_dec_i, scale=scale)
    loss = (y * grad_out).sum()
    loss.backward()

    grads = {
        "q_enc": q_enc_i.grad,
        "k_enc": k_enc_i.grad,
        "v_enc": v_enc_i.grad,
    }
    if q_dec_rand_i is not None:
        grads["q_dec_rand"] = q_dec_rand_i.grad
    if k_dec_rand_i is not None:
        grads["k_dec_rand"] = k_dec_rand_i.grad

    return y.detach(), grads


def test_flare_causal_reference_default_decode_aliases_match_legacy_path():
    torch.manual_seed(0)

    B = 2
    N = 7
    H = 3
    M = 5
    D = 16

    q_enc = torch.randn((H, M, D), dtype=torch.float32)
    k_enc = torch.randn((B, N, H, D), dtype=torch.float32)
    v_enc = torch.randn((B, N, H, D), dtype=torch.float32)

    for impl_name in ("reference", "recurrent", "pytorch2"):
        y_default = _run_impl(impl_name, q_enc, k_enc, v_enc)
        y_q_default = _run_impl(impl_name, q_enc, k_enc, v_enc, q_dec=k_enc)
        y_k_default = _run_impl(impl_name, q_enc, k_enc, v_enc, k_dec=q_enc)
        y_both_default = _run_impl(impl_name, q_enc, k_enc, v_enc, q_dec=k_enc, k_dec=q_enc)

        torch.testing.assert_close(y_default, y_q_default, rtol=0.0, atol=0.0)
        torch.testing.assert_close(y_default, y_k_default, rtol=0.0, atol=0.0)
        torch.testing.assert_close(y_default, y_both_default, rtol=0.0, atol=0.0)


def test_flare_causal_implementations_match_for_all_decode_variants():
    torch.manual_seed(1)

    B = 2
    N = 6
    H = 3
    M = 5
    D = 16
    scale = D ** -0.5

    q_enc = torch.randn((H, M, D), dtype=torch.float32)
    k_enc = torch.randn((B, N, H, D), dtype=torch.float32)
    v_enc = torch.randn((B, N, H, D), dtype=torch.float32)
    q_dec_rand = torch.randn((B, N, H, D), dtype=torch.float32)
    k_dec_rand = torch.randn((H, M, D), dtype=torch.float32)

    q_dec_options = (None, k_enc, q_dec_rand)
    k_dec_options = (None, q_enc, k_dec_rand)

    for q_dec in q_dec_options:
        for k_dec in k_dec_options:
            y_ref = _run_impl("reference", q_enc, k_enc, v_enc, q_dec=q_dec, k_dec=k_dec, scale=scale)
            y_recurrent = _run_impl("recurrent", q_enc, k_enc, v_enc, q_dec=q_dec, k_dec=k_dec, scale=scale)
            y_pytorch2 = _run_impl("pytorch2", q_enc, k_enc, v_enc, q_dec=q_dec, k_dec=k_dec, scale=scale)

            torch.testing.assert_close(y_recurrent, y_ref, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(y_pytorch2, y_ref, rtol=1e-4, atol=1e-5)


def test_flare_causal_implementations_backward_match_for_all_decode_variants():
    torch.manual_seed(2)

    B = 2
    N = 6
    H = 3
    M = 5
    D = 16
    scale = D ** -0.5

    q_enc = torch.randn((H, M, D), dtype=torch.float32)
    k_enc = torch.randn((B, N, H, D), dtype=torch.float32)
    v_enc = torch.randn((B, N, H, D), dtype=torch.float32)
    q_dec_rand_seed = torch.randn((B, N, H, D), dtype=torch.float32)
    k_dec_rand_seed = torch.randn((H, M, D), dtype=torch.float32)
    grad_out = torch.randn((B, N, H, D), dtype=torch.float32)

    q_dec_modes = ("none", "k_enc", "rand")
    k_dec_modes = ("none", "q_enc", "rand")

    for q_dec_mode in q_dec_modes:
        for k_dec_mode in k_dec_modes:
            y_ref, g_ref = _run_impl_with_grads(
                "reference",
                q_enc,
                k_enc,
                v_enc,
                q_dec_mode=q_dec_mode,
                k_dec_mode=k_dec_mode,
                scale=scale,
                grad_out=grad_out,
                q_dec_rand_seed=q_dec_rand_seed,
                k_dec_rand_seed=k_dec_rand_seed,
            )
            y_recurrent, g_recurrent = _run_impl_with_grads(
                "recurrent",
                q_enc,
                k_enc,
                v_enc,
                q_dec_mode=q_dec_mode,
                k_dec_mode=k_dec_mode,
                scale=scale,
                grad_out=grad_out,
                q_dec_rand_seed=q_dec_rand_seed,
                k_dec_rand_seed=k_dec_rand_seed,
            )
            y_pytorch2, g_pytorch2 = _run_impl_with_grads(
                "pytorch2",
                q_enc,
                k_enc,
                v_enc,
                q_dec_mode=q_dec_mode,
                k_dec_mode=k_dec_mode,
                scale=scale,
                grad_out=grad_out,
                q_dec_rand_seed=q_dec_rand_seed,
                k_dec_rand_seed=k_dec_rand_seed,
            )

            torch.testing.assert_close(y_recurrent, y_ref, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(y_pytorch2, y_ref, rtol=1e-4, atol=1e-5)

            for key in ("q_enc", "k_enc", "v_enc"):
                torch.testing.assert_close(g_recurrent[key], g_ref[key], rtol=5e-3, atol=5e-4)
                torch.testing.assert_close(g_pytorch2[key], g_ref[key], rtol=5e-3, atol=5e-4)

            if q_dec_mode == "rand":
                torch.testing.assert_close(g_recurrent["q_dec_rand"], g_ref["q_dec_rand"], rtol=5e-3, atol=5e-4)
                torch.testing.assert_close(g_pytorch2["q_dec_rand"], g_ref["q_dec_rand"], rtol=5e-3, atol=5e-4)
            if k_dec_mode == "rand":
                torch.testing.assert_close(g_recurrent["k_dec_rand"], g_ref["k_dec_rand"], rtol=5e-3, atol=5e-4)
                torch.testing.assert_close(g_pytorch2["k_dec_rand"], g_ref["k_dec_rand"], rtol=5e-3, atol=5e-4)

"""Extracted regression/stress suite implementation."""

from testing.suites.common import *


def _trainlike_projected():
    if not torch.cuda.is_available():
        print("[FLARE DEBUG] CUDA not available, skipping trainlike projected.")
        return
    device = torch.device("cuda")
    dtype_name = os.environ.get("FLARE_TRAINLIKE_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name)
    impl = os.environ.get("FLARE_TRAINLIKE_IMPL", "triton3")
    steps = int(os.environ.get("FLARE_TRAINLIKE_STEPS", "8"))
    seed = int(os.environ.get("FLARE_TRAINLIKE_SEED", "0"))
    lr = float(os.environ.get("FLARE_TRAINLIKE_LR", "1e-3"))
    qkv_std = float(os.environ.get("FLARE_TRAINLIKE_QKV_STD", "1.0"))
    log_every = int(os.environ.get("FLARE_TRAINLIKE_LOG_EVERY", "1"))
    scale_mode = os.environ.get("FLARE_TRAINLIKE_SCALE", "sqrt")

    configs = [
        dict(B=2, H=8, M=128, N=512, D=32),
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
        print(
            f"[FLARE TRAINLIKE PROJ] impl={impl} B={B} H={H} M={M} N={N} D={D} "
            f"dtype={dtype} lr={lr:g} scale={scale:.6g}"
        )

        torch.manual_seed(seed + 1234)
        hidden = qkv_std * torch.randn(B, N, H, D, device=device, dtype=dtype)
        latent_q = torch.nn.Parameter(qkv_std * torch.randn(H, M, D, device=device, dtype=dtype))
        Wk = torch.nn.Parameter(qkv_std * torch.randn(H, D, D, device=device, dtype=dtype))
        Wv = torch.nn.Parameter(qkv_std * torch.randn(H, D, D, device=device, dtype=dtype))

        params = [latent_q, Wk, Wv]

        for step in range(steps):
            for p in params:
                if p.grad is not None:
                    p.grad = None

            K = torch.einsum("bnhd,hde->bnhe", hidden, Wk)
            V = torch.einsum("bnhd,hde->bnhe", hidden, Wv)

            if impl == "pytorch2":
                Y = flare_causal_chunked(latent_q, K, V, scale=scale)
            else:
                Y = flare_chunk_triton(latent_q, K, V, scale)

            loss = Y.float().pow(2).mean()
            loss.backward()

            with torch.no_grad():
                for p in params:
                    p -= lr * p.grad

            if step % log_every == 0:
                k_norm = K.float().norm().item()
                v_norm = V.float().norm().item()
                k_max = K.float().abs().max().item()
                v_max = V.float().abs().max().item()
                lq_norm = latent_q.float().norm().item()
                lq_max = latent_q.float().abs().max().item()
                g_lq = latent_q.grad.float().norm().item()
                g_wk = Wk.grad.float().norm().item()
                g_wv = Wv.grad.float().norm().item()
                print(
                    "[FLARE TRAINLIKE PROJ] "
                    f"step={step} "
                    f"latent_q|max={lq_max:.3e},norm={lq_norm:.3e} "
                    f"K|max={k_max:.3e},norm={k_norm:.3e} "
                    f"V|max={v_max:.3e},norm={v_norm:.3e} "
                    f"grads: d_latent_q={g_lq:.3e} dWk={g_wk:.3e} dWv={g_wv:.3e}"
                )


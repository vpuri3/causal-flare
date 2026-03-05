from . import _common as _common_impl
from . import chunked as _chunked_impl
from ._common import *
from .chunked import *
from .chunked_old import ChunkedFLAREOld
from .dense import *
from .inference import *
from .recurrent import *
from .torch import *


def _denseflare1_phase_bench(Q, K, V, scale=1.0):
    H, M, D = Q.size()
    B, N, _, _ = K.size()
    if (M % 16) != 0 or (D % 16) != 0 or (N % 16) != 0:
        raise ValueError(f"DenseFLARE1 requires M, N, D be multiples of 16. Got M={M}, N={N}, D={D}")

    K_bhnd = K.permute(0, 2, 1, 3).contiguous()
    V_bhnd = V.permute(0, 2, 1, 3).contiguous()

    block_m = M
    block_d = D
    block_n = N
    grid = (B * H,)
    num_warps = 4 if block_m <= 64 else 8
    eps = _get_eps_for_dtype(Q.dtype)

    S = torch.empty((B, H, M, N), device=Q.device, dtype=torch.float32)
    P = torch.empty((B, H, M, N), device=Q.device, dtype=torch.float32)
    X = torch.empty((B, H, M, N), device=Q.device, dtype=torch.float32)
    Y = torch.empty((B, H, N, D), device=Q.device, dtype=Q.dtype)

    t_phase1 = triton.testing.do_bench(
        lambda: flare_dense1_phase1_kernel[grid](
            Q, K_bhnd, S, P,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
            S.stride(0), S.stride(1), S.stride(2), S.stride(3),
            P.stride(0), P.stride(1), P.stride(2), P.stride(3),
            B, H, M,
            N,
            scale,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        ),
        warmup=2,
        rep=2,
    )

    t_phase2 = triton.testing.do_bench(
        lambda: flare_dense1_phase2_kernel[grid](
            Q, K_bhnd, X,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhnd.stride(0), K_bhnd.stride(1), K_bhnd.stride(2), K_bhnd.stride(3),
            X.stride(0), X.stride(1), X.stride(2), X.stride(3),
            B, H, M,
            N,
            scale,
            eps,
            D=D,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        ),
        warmup=2,
        rep=2,
    )

    t_phase3 = triton.testing.do_bench(
        lambda: flare_dense1_phase3_kernel[grid](
            S, X, V_bhnd, Y,
            S.stride(0), S.stride(1), S.stride(2), S.stride(3),
            X.stride(0), X.stride(1), X.stride(2), X.stride(3),
            V_bhnd.stride(0), V_bhnd.stride(1), V_bhnd.stride(2), V_bhnd.stride(3),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            B, H, M,
            N,
            D=D,
            BLOCK_M=block_m,
            BLOCK_D=block_d,
            BLOCK_N=block_n,
            num_warps=num_warps,
            num_stages=2,
        ),
        warmup=2,
        rep=2,
    )
    return t_phase1, t_phase2, t_phase3

def _print_err_report(name: str, pred: torch.Tensor, ref: torch.Tensor, atol: float) -> None:
    delta = (pred - ref).abs()
    max_abs = delta.amax().item()
    mean_abs = delta.mean().item()
    rel_l2 = _rel_l2_err(pred, ref)
    max_rel = _max_rel_err(pred, ref, atol)
    b, t, h, d = _max_abs_idx(delta)
    finite_pred = torch.isfinite(pred).all().item()
    finite_ref = torch.isfinite(ref).all().item()
    print(f"[CHECK] {name}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_l2={rel_l2:.3e} max_rel={max_rel:.3e} finite={finite_pred}/{finite_ref} worst_idx=(b={b},t={t},h={h},d={d})")

def _causality_check(name, func, Q, K, V, scale, t_indices, atol, token_dim: int) -> None:
    with torch.no_grad():
        B, N, H, D = K.shape
        print(
            f"[CAUSALITY] {name}: method_full_vs_prefix | prefix_vs_ref | full_vs_ref",
            flush=True,
        )
        def _select_token(Y: torch.Tensor, t_idx: int) -> torch.Tensor:
            if Y.dim() != 4:
                raise ValueError(f"{name} output must be 4D, got shape={tuple(Y.shape)}")
            if token_dim == 1:
                return Y[:, t_idx]  # [B, H, D]
            if token_dim == 2:
                return Y[:, :, t_idx]  # [B, H, D]
            raise ValueError(f"{name} invalid token_dim={token_dim}")
        for t in t_indices:
            if t >= N - 1:
                continue
            K_prefix = K[:, : t + 1].clone()
            V_prefix = V[:, : t + 1].clone()
            Y_full = func(Q, K, V, scale=scale)
            Y_pref = func(Q, K_prefix, V_prefix, scale=scale)
            Y_full_ref = flare_causal_reference(Q, K, V, scale=scale)
            y_full_t = _select_token(Y_full, t)
            y_pref_t = _select_token(Y_pref, t)
            y_ref_t = Y_full_ref[:, t]
            delta_pref_full = (y_pref_t - y_full_t).abs().amax().item()
            delta_pref_ref = (y_pref_t - y_ref_t).abs().amax().item()
            delta_full_ref = (y_full_t - y_ref_t).abs().amax().item()
            status = "OK" if delta_pref_full <= atol else "FAIL"
            status_ref = "OK" if delta_pref_ref <= atol else "FAIL"
            status_full_ref = "OK" if delta_full_ref <= atol else "FAIL"
            print(
                f"[CAUSALITY] {name} t={t} "
                f"method_full_vs_prefix={delta_pref_full:.3e} "
                f"prefix_vs_ref={delta_pref_ref:.3e} "
                f"full_vs_ref={delta_full_ref:.3e} "
                f"({status}/{status_ref}/{status_full_ref})"
            )

def _grad_report(name: str, g_pred: torch.Tensor, g_ref: torch.Tensor, atol: float) -> None:
    delta = (g_pred - g_ref).abs()
    max_abs = delta.amax().item()
    mean_abs = delta.mean().item()
    rel_l2 = _rel_l2_err(g_pred, g_ref)
    max_rel = _max_rel_err(g_pred, g_ref, atol)
    idx = _max_abs_idx(delta)
    finite_pred = torch.isfinite(g_pred).all().item()
    finite_ref = torch.isfinite(g_ref).all().item()
    print(f"[GRAD] {name}: max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} rel_l2={rel_l2:.3e} max_rel={max_rel:.3e} finite={finite_pred}/{finite_ref} worst_idx={idx}")

def _gradcheck_suite(dtype: torch.dtype, B: int, H: int, N: int, M: int, D: int, scale: float, atol: float) -> None:
    device = torch.device("cuda")
    Q = torch.randn(H, M, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)
    R = torch.randn(B, N, H, D, device=device, dtype=dtype)

    def _loss_ref(q, k, v):
        y = flare_causal_reference(q, k, v, scale=scale)
        return (y * R).sum()

    def _loss_p2(q, k, v):
        y = flare_causal_chunked(q, k, v, scale=scale)
        return (y * R).sum()

    def _loss_t3(q, k, v):
        y = flare_chunk_triton(q, k, v, scale)
        return (y * R).sum()

    loss_ref = _loss_ref(Q, K, V)
    gq_ref, gk_ref, gv_ref = torch.autograd.grad(loss_ref, (Q, K, V), retain_graph=True)
    loss_p2 = _loss_p2(Q, K, V)
    gq_p2, gk_p2, gv_p2 = torch.autograd.grad(loss_p2, (Q, K, V), retain_graph=True)
    loss_t3 = None
    gq_t3 = gk_t3 = gv_t3 = None
    t3_err = None
    try:
        prev_bwd_fp32 = os.environ.get("FLARE_TRITON3_BWD_FP32", "")
        prev_force_stats = os.environ.get("FLARE_TRITON_FORCE_FP32_STATS", "")
        os.environ["FLARE_TRITON3_BWD_FP32"] = "1"
        os.environ["FLARE_TRITON_FORCE_FP32_STATS"] = "1"
        loss_t3 = _loss_t3(Q, K, V)
        gq_t3, gk_t3, gv_t3 = torch.autograd.grad(loss_t3, (Q, K, V), retain_graph=False)
    except Exception as exc:
        t3_err = str(exc)
    finally:
        if prev_bwd_fp32 == "":
            os.environ.pop("FLARE_TRITON3_BWD_FP32", None)
        else:
            os.environ["FLARE_TRITON3_BWD_FP32"] = prev_bwd_fp32
        if prev_force_stats == "":
            os.environ.pop("FLARE_TRITON_FORCE_FP32_STATS", None)
        else:
            os.environ["FLARE_TRITON_FORCE_FP32_STATS"] = prev_force_stats

    loss_t3_str = f"{loss_t3.item():.3e}" if loss_t3 is not None else "N/A"
    print(f"[GRAD] loss_ref={loss_ref.item():.3e} loss_p2={loss_p2.item():.3e} loss_t3={loss_t3_str}")
    _grad_report("dQ Pytorch2 vs Ref", gq_p2, gq_ref, atol)
    _grad_report("dK Pytorch2 vs Ref", gk_p2, gk_ref, atol)
    _grad_report("dV Pytorch2 vs Ref", gv_p2, gv_ref, atol)
    if t3_err is None and gq_t3 is not None:
        _grad_report("dQ Triton vs Ref", gq_t3, gq_ref, atol)
        _grad_report("dK Triton vs Ref", gk_t3, gk_ref, atol)
        _grad_report("dV Triton vs Ref", gv_t3, gv_ref, atol)
    else:
        print(f"[GRAD] Triton backward skipped: {t3_err}")


@contextmanager
def _scoped_float32_math_mode(*, allow_tf32: bool):
    prev_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
    get_precision = getattr(torch, "get_float32_matmul_precision", None)
    set_precision = getattr(torch, "set_float32_matmul_precision", None)
    prev_precision = get_precision() if callable(get_precision) else None
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if callable(set_precision):
        set_precision("high" if allow_tf32 else "highest")
    try:
        yield
    finally:
        if callable(set_precision) and prev_precision is not None:
            set_precision(prev_precision)
        torch.backends.cudnn.allow_tf32 = prev_cudnn_allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_allow_tf32


def _correctness_chunk_config(M: int, N: int, D: int, dtype: torch.dtype) -> tuple[str, int]:
    input_precision = _common_impl._normalize_input_precision(None, None)
    cfg = _chunked_impl._get_chunked_forward_config(
        M=M,
        N=N,
        D=D,
        dtype=dtype,
        chunk_size=None,
        input_precision=input_precision,
    )
    return str(cfg["input_precision"]), int(cfg["CHUNK_SIZE"])


def _run_correctness_reference(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    scale: float,
    Q_dec: torch.Tensor | None = None,
    K_dec: torch.Tensor | None = None,
) -> torch.Tensor:
    with _temp_env_var("FLARE_REFERENCE_FP32", "1"):
        with _scoped_float32_math_mode(allow_tf32=False):
            return flare_causal_reference(Q, K, V, Q_dec=Q_dec, K_dec=K_dec, scale=scale)


def _run_correctness_pytorch2(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    scale: float,
    chunk_size: int,
    allow_tf32: bool,
    Q_dec: torch.Tensor | None = None,
    K_dec: torch.Tensor | None = None,
) -> torch.Tensor:
    with _scoped_float32_math_mode(allow_tf32=allow_tf32):
        return flare_causal_chunked(Q, K, V, scale=scale, chunk_size=chunk_size, Q_dec=Q_dec, K_dec=K_dec)


def _run_correctness_reference_grads(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    scale: float,
) -> dict[str, torch.Tensor | None]:
    return _run_correctness_reference_grads_with_decode(Q, K, V, scale=scale, Q_dec=None, K_dec=None)


def _run_correctness_reference_grads_with_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    scale: float,
    Q_dec: torch.Tensor | None,
    K_dec: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    Q_dec_ref = Q_dec.detach().clone().requires_grad_(True) if Q_dec is not None else None
    K_dec_ref = K_dec.detach().clone().requires_grad_(True) if K_dec is not None else None
    with _temp_env_var("FLARE_REFERENCE_FP32", "1"):
        with _scoped_float32_math_mode(allow_tf32=False):
            Y_ref = flare_causal_reference(Q_ref, K_ref, V_ref, Q_dec=Q_dec_ref, K_dec=K_dec_ref, scale=scale)
            Y_ref.sum().backward()
    return {
        "q": Q_ref.grad,
        "k": K_ref.grad,
        "v": V_ref.grad,
        "q_dec": None if Q_dec_ref is None else Q_dec_ref.grad,
        "k_dec": None if K_dec_ref is None else K_dec_ref.grad,
    }


def _run_correctness_pytorch2_grads(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    scale: float,
    chunk_size: int,
    allow_tf32: bool,
) -> dict[str, torch.Tensor | None]:
    return _run_correctness_pytorch2_grads_with_decode(
        Q, K, V, scale=scale, chunk_size=chunk_size, allow_tf32=allow_tf32, Q_dec=None, K_dec=None
    )


def _run_correctness_pytorch2_grads_with_decode(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    *,
    scale: float,
    chunk_size: int,
    allow_tf32: bool,
    Q_dec: torch.Tensor | None,
    K_dec: torch.Tensor | None,
) -> dict[str, torch.Tensor | None]:
    Q_ref = Q.detach().clone().requires_grad_(True)
    K_ref = K.detach().clone().requires_grad_(True)
    V_ref = V.detach().clone().requires_grad_(True)
    Q_dec_ref = Q_dec.detach().clone().requires_grad_(True) if Q_dec is not None else None
    K_dec_ref = K_dec.detach().clone().requires_grad_(True) if K_dec is not None else None
    with _scoped_float32_math_mode(allow_tf32=allow_tf32):
        Y_ref = flare_causal_chunked(
            Q_ref, K_ref, V_ref, scale=scale, chunk_size=chunk_size, Q_dec=Q_dec_ref, K_dec=K_dec_ref
        )
        Y_ref.sum().backward()
    return {
        "q": Q_ref.grad,
        "k": K_ref.grad,
        "v": V_ref.grad,
        "q_dec": None if Q_dec_ref is None else Q_dec_ref.grad,
        "k_dec": None if K_dec_ref is None else K_dec_ref.grad,
    }


def _scaled_error_limit(err: float, multiplier: float, slack: float) -> float:
    return float(multiplier) * float(err) + float(slack)

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
    grad_shapes = [
        (1, 1, 16, 16, 16),
        (1, 1, 33, 16, 32),
    ]
    grad_scales_env = os.environ.get("FLARE_CORRECTNESS_SUITE_GRAD_SCALES", "")
    if grad_scales_env:
        grad_scales = [float(x) for x in grad_scales_env.split(",") if x.strip()]
    else:
        grad_scales = [1.0]
    grad_limit_env = os.environ.get("FLARE_CORRECTNESS_GRAD_LIMIT", "")
    if grad_limit_env:
        grad_limit = int(grad_limit_env)
    else:
        # By default, run at least one backward check per decode-separation mode.
        grad_limit = max(1, len(decode_modes))
    grad_count = 0
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
        grad_mean_abs_max = _env_float("FLARE_CORRECTNESS_GRAD_MEAN_ABS_MAX", 5e-3)
        grad_max_abs_max = _env_float("FLARE_CORRECTNESS_GRAD_MAX_ABS_MAX", 8e-1)
        grad_cos_min = _env_float("FLARE_CORRECTNESS_GRAD_COS_MIN", 0.999)
        fa_fwd_mult = _env_float("FLARE_CORRECTNESS_FA_FWD_MULT", 2.0)
        fa_fwd_slack = _env_float("FLARE_CORRECTNESS_FA_FWD_SLACK", 1e-5)
        fa_grad_slack = _env_float("FLARE_CORRECTNESS_FA_GRAD_SLACK", 1e-5)
        for (B, H, N, M, D) in shapes:
            scale = (D ** -0.5)
            input_precision, chunk_size = _correctness_chunk_config(M=M, N=N, D=D, dtype=dtype)
            # Forward uses the FlashAttention-style 2x bound directly. Backward
            # still relies on autograd through the PyTorch chunked path (not a
            # matching chunked backward implementation yet), so keep modest extra
            # headroom until the dedicated PyTorch chunked backward baseline
            # exists. tf32x3 is closer to the PyTorch noise model than plain
            # tf32, so it can use the tighter default.
            fa_grad_mult = _env_float(
                "FLARE_CORRECTNESS_FA_GRAD_MULT",
                2.0 if input_precision == "tf32x3" else 5.0,
            )
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

                    if os.environ.get("FLARE_CORRECTNESS_GRAD", "0") == "1" and grad_count < grad_limit:
                        grads_ref = _run_correctness_reference_grads_with_decode(
                            Q, K, V, scale=scale, Q_dec=Q_dec, K_dec=K_dec
                        )
                        grads_noise = _run_correctness_pytorch2_grads_with_decode(
                            Q, K, V, scale=scale, chunk_size=chunk_size, allow_tf32=noise_allow_tf32, Q_dec=Q_dec, K_dec=K_dec
                        )
                        Q_t3 = Q.detach().clone().requires_grad_(True)
                        K_t3 = K.detach().clone().requires_grad_(True)
                        V_t3 = V.detach().clone().requires_grad_(True)
                        Q_dec_t3 = Q_dec.detach().clone().requires_grad_(True) if Q_dec is not None else None
                        K_dec_t3 = K_dec.detach().clone().requires_grad_(True) if K_dec is not None else None
                        Y_t3 = flare_chunk_triton(
                            Q_t3, K_t3, V_t3, scale, chunk_size, input_precision, False, Q_dec_t3, K_dec_t3
                        )
                        Y_t3.sum().backward()
                        grads_t3 = {
                            "q": Q_t3.grad,
                            "k": K_t3.grad,
                            "v": V_t3.grad,
                            "q_dec": None if Q_dec_t3 is None else Q_dec_t3.grad,
                            "k_dec": None if K_dec_t3 is None else K_dec_t3.grad,
                        }
                        grad_specs = [
                            ("dQ", "q", "suite_qg"),
                            ("dK", "k", "suite_kg"),
                            ("dV", "v", "suite_vg"),
                        ]
                        if separate_q_dec:
                            grad_specs.append(("dQ_dec", "q_dec", "suite_qdg"))
                        if separate_k_dec:
                            grad_specs.append(("dK_dec", "k_dec", "suite_kdg"))

                        for grad_name, key, prefix in grad_specs:
                            err = compute_errors(grads_t3[key], grads_ref[key], prefix)
                            noise_err = compute_errors(grads_noise[key], grads_ref[key], f"{prefix}_noise")
                            cos = _tensor_cosine(grads_t3[key], grads_ref[key])
                            print(
                                "[SUITE GRAD] "
                                f"{grad_name} mean_abs={err[f'{prefix}_mean_abs_err']:.3e} "
                                f"max_abs={err[f'{prefix}_max_abs_err']:.3e} "
                                f"noise_max_abs={noise_err[f'{prefix}_noise_max_abs_err']:.3e} "
                                f"cos={cos:.5f}"
                            )
                            if strict and input_precision == "ieee":
                                _maybe_record_failure(
                                    failures,
                                    err[f"{prefix}_mean_abs_err"] > grad_mean_abs_max,
                                    (
                                        f"Correctness suite failed ({grad_name} mean_abs): "
                                        f"{err[f'{prefix}_mean_abs_err']:.3e} > {grad_mean_abs_max:.3e} "
                                        f"for dtype={dtype}, B={B}, H={H}, N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                                    ),
                                )
                                _maybe_record_failure(
                                    failures,
                                    err[f"{prefix}_max_abs_err"] > grad_max_abs_max,
                                    (
                                        f"Correctness suite failed ({grad_name} max_abs): "
                                        f"{err[f'{prefix}_max_abs_err']:.3e} > {grad_max_abs_max:.3e} "
                                        f"for dtype={dtype}, B={B}, H={H}, N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                                    ),
                                )
                                _maybe_record_failure(
                                    failures,
                                    cos < grad_cos_min,
                                    (
                                        f"Correctness suite failed ({grad_name} cosine): "
                                        f"{cos:.6f} < {grad_cos_min:.6f} "
                                        f"for dtype={dtype}, B={B}, H={H}, N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                                    ),
                                )
                            elif strict:
                                grad_limit_scaled = _scaled_error_limit(
                                    noise_err[f"{prefix}_noise_max_abs_err"], fa_grad_mult, fa_grad_slack
                                )
                                _maybe_record_failure(
                                    failures,
                                    err[f"{prefix}_max_abs_err"] > grad_limit_scaled,
                                    (
                                        f"Correctness suite failed ({grad_name} max_abs vs reference): "
                                        f"{err[f'{prefix}_max_abs_err']:.3e} > {grad_limit_scaled:.3e} "
                                        f"(noise={noise_err[f'{prefix}_noise_max_abs_err']:.3e}, "
                                        f"mult={fa_grad_mult:.2f}, slack={fa_grad_slack:.1e}) "
                                        f"for input_precision={input_precision}, dtype={dtype}, "
                                        f"B={B}, H={H}, N={N}, M={M}, D={D}, qk_std={qk_std}, {decode_label}"
                                    ),
                                )
                        grad_count += 1

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
        if os.environ.get("FLARE_CORRECTNESS_SUITE_GRAD", "0") == "1":
            for (B, H, N, M, D) in grad_shapes:
                for scale in grad_scales:
                    print(f"\n[SUITE GRAD] dtype={dtype} B={B} H={H} N={N} M={M} D={D} scale={scale}")
                    _gradcheck_suite(dtype, B, H, N, M, D, scale=scale, atol=atol)
    if strict and failures:
        summary = "\n".join(f"- {msg}" for msg in failures[:12])
        extra = "" if len(failures) <= 12 else f"\n- ... and {len(failures) - 12} more"
        raise AssertionError(f"FLARE correctness validation failed ({len(failures)} issues):\n{summary}{extra}")

def optimize_for_h100():
    """Apply H100-specific optimizations for maximum performance"""

    # Environment variables for H100 optimization
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True,roundup_power2_divisions:16'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async execution
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8 API
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    # H100-specific PyTorch backend optimizations
    # Enable TF32 tensor cores: FP32 operations use TF32 (10-bit mantissa, 8-bit exponent)
    # This provides ~6× better accuracy than BF16 with only ~1% performance cost
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True

    # FP8 support (only available in newer PyTorch versions)
    try:
        torch.backends.cuda.matmul.allow_fp8_e4m3fn = True
        torch.backends.cuda.matmul.allow_fp8_e5m2 = True
    except AttributeError:
        pass

    # Performance vs reproducibility settings (choose one)
    torch.backends.cudnn.benchmark = True  # Enable for better performance
    torch.backends.cudnn.deterministic = False  # Disable for better performance

    # Memory management optimizations
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of GPU memory
        torch.cuda.empty_cache()

    print("H100 optimizations applied successfully!")


def _debug_triton_bwd_compare():
    if not torch.cuda.is_available():
        print("[FLARE DEBUG] CUDA not available, skipping Triton3 backward debug.")
        return
    if "flare_chunk_bwd" not in globals() or "flare_chunk_prefix_bwd" not in globals():
        print("[FLARE DEBUG] Legacy Triton backward debug path is unavailable; old chunked backward helpers were removed.")
        return
    device = torch.device("cuda")
    B, H, M, N, D = 1, 1, 16, 128, 16
    scale = 1.0 / math.sqrt(D)
    dtype = torch.float32
    clamp_max = _get_exp_clamp_for_dtype(dtype)
    Q = torch.randn(H, M, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)

    CHUNK_SIZE = 64
    NUM_CHUNKS = math.ceil(N / CHUNK_SIZE)

    Kc = K.reshape(B, NUM_CHUNKS, CHUNK_SIZE, H, D).permute(0, 3, 1, 2, 4).contiguous()
    Vc = V.reshape(B, NUM_CHUNKS, CHUNK_SIZE, H, D).permute(0, 3, 1, 2, 4).contiguous()

    score_chunk = scale * torch.einsum("bhncd,hmd->bhncm", Kc, Q)  # [B,H,NC,C,M]
    chunk_max = score_chunk.max(dim=3).values
    expS = torch.exp(score_chunk - chunk_max.unsqueeze(3))
    chunk_den = expS.sum(dim=3)
    chunk_num = torch.einsum("bhncm,bhncd->bhnmd", expS, Vc)

    prefix_max = torch.empty((B, H, NUM_CHUNKS, M), device=device, dtype=dtype)
    prefix_den = torch.zeros((B, H, NUM_CHUNKS, M), device=device, dtype=dtype)
    prefix_num = torch.zeros((B, H, NUM_CHUNKS, M, D), device=device, dtype=dtype)

    max_curr = torch.full((B, H, M), -float("inf"), device=device, dtype=dtype)
    den_curr = torch.zeros((B, H, M), device=device, dtype=dtype)
    num_curr = torch.zeros((B, H, M, D), device=device, dtype=dtype)
    for chunk_idx in range(NUM_CHUNKS):
        prefix_max[:, :, chunk_idx, :] = max_curr
        prefix_den[:, :, chunk_idx, :] = den_curr
        prefix_num[:, :, chunk_idx, :, :] = num_curr
        sc_max = chunk_max[:, :, chunk_idx, :]
        sc_den = chunk_den[:, :, chunk_idx, :]
        sc_num = chunk_num[:, :, chunk_idx, :, :]
        max_new = torch.maximum(max_curr, sc_max)
        rescale_prev = torch.exp(max_curr - max_new)
        rescale_curr = torch.exp(sc_max - max_new)
        den_curr = den_curr * rescale_prev + sc_den * rescale_curr
        num_curr = num_curr * rescale_prev.unsqueeze(-1) + sc_num * rescale_curr.unsqueeze(-1)
        max_curr = max_new

    dO = torch.randn(B, N, H, D, device=device, dtype=dtype)

    def _stats(name, t):
        t_f = t.float()
        nan = torch.isnan(t_f).sum().item()
        inf = torch.isinf(t_f).sum().item()
        print(f"[FLARE DEBUG] {name}: mean={t_f.mean().item():.3e}, max={t_f.max().item():.3e}, "
              f"min={t_f.min().item():.3e}, nan={nan}, inf={inf}")

    def _print_diff(name, a, b):
        diff = (a - b).float().abs()
        print(f"[FLARE DEBUG] {name} diff: mean={diff.mean().item():.3e}, max={diff.max().item():.3e}")

    # Phase3 torch reference (grads wrt Q,K,V,prefix stats)
    prefix_max_t = prefix_max.clone().requires_grad_(True)
    prefix_den_t = prefix_den.clone().requires_grad_(True)
    prefix_num_t = prefix_num.clone().requires_grad_(True)

    score_chunk_t = score_chunk
    s_max_t = score_chunk_t.max(dim=-1).values
    s_max_t = torch.clamp(s_max_t, max=clamp_max)
    exp_prev_max = torch.exp(prefix_max_t.clamp(max=clamp_max))
    A_prime_t = torch.exp(score_chunk_t - s_max_t.unsqueeze(-1))
    A_scaled_t = A_prime_t * torch.exp(s_max_t).unsqueeze(-1)
    den_chunk = torch.cumsum(A_scaled_t, dim=3)
    den_total = den_chunk + (prefix_den_t * exp_prev_max).unsqueeze(3)
    P = torch.softmax(score_chunk_t, dim=-1)
    expA = P / (den_total + 1e-6)
    prev_num = prefix_num_t * exp_prev_max.unsqueeze(-1)
    y_prev = torch.einsum("bhncm,bhnmd->bhncd", expA, prev_num)
    W = torch.einsum("bhncm,bhnsm->bhncs", expA, A_scaled_t)
    causal = torch.tril(torch.ones((CHUNK_SIZE, CHUNK_SIZE), device=device, dtype=torch.bool))
    W = W.masked_fill(~causal[None, None, None, :, :], 0.0)
    y_curr = torch.einsum("bhncs,bhnsd->bhncd", W, Vc)
    Yc = y_prev + y_curr
    Y = Yc.reshape(B, H, N, D).permute(0, 2, 1, 3)

    dQ_ref, dK_ref, dV_ref, dPmax_ref, dPden_ref, dPnum_ref = torch.autograd.grad(
        Y,
        (Q, K, V, prefix_max_t, prefix_den_t, prefix_num_t),
        grad_outputs=dO,
        retain_graph=True,
        create_graph=False,
    )

    # Phase3 dS reference (treat prefix stats as constants)
    S_leaf = score_chunk_t.detach().clone().requires_grad_(True)
    prefix_max_const = prefix_max.detach()
    prefix_den_const = prefix_den.detach()
    prefix_num_const = prefix_num.detach()

    s_max_leaf = S_leaf.max(dim=-1).values
    s_max_leaf = torch.clamp(s_max_leaf, max=clamp_max)
    exp_prev_max_const = torch.exp(prefix_max_const.clamp(max=clamp_max))
    A_prime_leaf = torch.exp(S_leaf - s_max_leaf.unsqueeze(-1))
    A_scaled_leaf = A_prime_leaf * torch.exp(s_max_leaf).unsqueeze(-1)
    den_chunk_leaf = torch.cumsum(A_scaled_leaf, dim=3)
    den_total_leaf = den_chunk_leaf + (prefix_den_const * exp_prev_max_const).unsqueeze(3)
    P_leaf = torch.softmax(S_leaf, dim=-1)
    expA_leaf = P_leaf / (den_total_leaf + 1e-6)
    prev_num_leaf = prefix_num_const * exp_prev_max_const.unsqueeze(-1)
    y_prev_leaf = torch.einsum("bhncm,bhnmd->bhncd", expA_leaf, prev_num_leaf)
    W_leaf = torch.einsum("bhncm,bhnsm->bhncs", expA_leaf, A_scaled_leaf)
    W_leaf = W_leaf.masked_fill(~causal[None, None, None, :, :], 0.0)
    y_curr_leaf = torch.einsum("bhncs,bhnsd->bhncd", W_leaf, Vc)
    Y_leaf = (y_prev_leaf + y_curr_leaf).reshape(B, H, N, D).permute(0, 2, 1, 3)
    dS_ref = torch.autograd.grad(Y_leaf, S_leaf, grad_outputs=dO, retain_graph=False, create_graph=False)[0]

    # Phase3 dS formula (torch)
    expA_b = expA.reshape(B * H * NUM_CHUNKS, CHUNK_SIZE, M)
    expS_b = A_scaled_t.reshape(B * H * NUM_CHUNKS, CHUNK_SIZE, M)
    dO_c = dO.permute(0, 2, 1, 3).reshape(B, H, NUM_CHUNKS, CHUNK_SIZE, D)
    dO_b = dO_c.reshape(B * H * NUM_CHUNKS, CHUNK_SIZE, D)
    V_b = Vc.reshape(B * H * NUM_CHUNKS, CHUNK_SIZE, D)
    sum_prev_exp_v = (prefix_num_const * exp_prev_max_const.unsqueeze(-1)).reshape(B * H * NUM_CHUNKS, M, D)

    W_b = torch.bmm(expA_b, expS_b.transpose(1, 2))
    W_b = W_b.masked_fill(~causal[None, :, :], 0.0)
    dW_b = torch.bmm(dO_b, V_b.transpose(1, 2))
    dW_b = dW_b.masked_fill(~causal[None, :, :], 0.0)
    dExpA_b = torch.bmm(dW_b, expS_b) + torch.bmm(dO_b, sum_prev_exp_v.transpose(1, 2))
    dExpS_from_W_b = torch.bmm(dW_b.transpose(1, 2), expA_b)

    inv_den = 1.0 / (den_total + 1e-6)
    dP = dExpA_b.reshape_as(expA) * inv_den
    dDen_total = -(dExpA_b.reshape_as(expA) * P) * (inv_den * inv_den)
    d_sum_prev_exp = dDen_total.sum(dim=3)
    d_sum_prev_exp_v = torch.bmm(expA_b.transpose(1, 2), dO_b).reshape(B, H, NUM_CHUNKS, M, D)

    prefix = torch.cumsum(dDen_total, dim=3)
    sum_all = dDen_total.sum(dim=3, keepdim=True)
    suffix = sum_all - prefix + dDen_total

    dExpS = dExpS_from_W_b.reshape_as(A_scaled_t) + suffix
    dP_dot = (dP * P).sum(dim=-1, keepdim=True)
    dS_softmax = P * (dP - dP_dot)
    dS_from_A = dExpS * A_scaled_t
    clamped = s_max_t > clamp_max
    if clamped.any():
        sum_dA_A = dS_from_A.sum(dim=-1, keepdim=True)
        mask_argmax = score_chunk_t == s_max_t.unsqueeze(-1)
        dS_from_A = torch.where(clamped.unsqueeze(-1), dS_from_A - mask_argmax * sum_dA_A, dS_from_A)
    dS_formula = dS_from_A + dS_softmax

    _print_diff("dS_phase3", dS_formula, dS_ref)

    _stats("dPmax_ref", dPmax_ref)
    _stats("dPden_ref", dPden_ref)
    _stats("dPnum_ref", dPnum_ref)

    # Prefix backward reference: grads wrt chunk stats
    chunk_max_t = chunk_max.clone().requires_grad_(True)
    chunk_den_t = chunk_den.clone().requires_grad_(True)
    chunk_num_t = chunk_num.clone().requires_grad_(True)

    prefix_max_list = []
    prefix_den_list = []
    prefix_num_list = []
    max_curr = torch.full((B, H, M), -float("inf"), device=device, dtype=dtype)
    den_curr = torch.zeros((B, H, M), device=device, dtype=dtype)
    num_curr = torch.zeros((B, H, M, D), device=device, dtype=dtype)
    for chunk_idx in range(NUM_CHUNKS):
        prefix_max_list.append(max_curr)
        prefix_den_list.append(den_curr)
        prefix_num_list.append(num_curr)
        sc_max = chunk_max_t[:, :, chunk_idx, :]
        sc_den = chunk_den_t[:, :, chunk_idx, :]
        sc_num = chunk_num_t[:, :, chunk_idx, :, :]
        max_new = torch.maximum(max_curr, sc_max)
        rescale_prev = torch.exp(max_curr - max_new)
        rescale_curr = torch.exp(sc_max - max_new)
        den_curr = den_curr * rescale_prev + sc_den * rescale_curr
        num_curr = num_curr * rescale_prev.unsqueeze(-1) + sc_num * rescale_curr.unsqueeze(-1)
        max_curr = max_new
    prefix_max_ref = torch.stack(prefix_max_list, dim=2)
    prefix_den_ref = torch.stack(prefix_den_list, dim=2)
    prefix_num_ref = torch.stack(prefix_num_list, dim=2)

    dCmax_ref, dCden_ref, dCnum_ref = torch.autograd.grad(
        outputs=(prefix_max_ref, prefix_den_ref, prefix_num_ref),
        inputs=(chunk_max_t, chunk_den_t, chunk_num_t),
        grad_outputs=(dPmax_ref, dPden_ref, dPnum_ref),
        retain_graph=False,
        create_graph=False,
    )
    _stats("dCmax_ref", dCmax_ref)
    _stats("dCden_ref", dCden_ref)
    _stats("dCnum_ref", dCnum_ref)

    # Triton phase3 kernel + prefix bwd
    BH = B * H
    dQ_direct = torch.zeros((H, M, D), device=device, dtype=torch.float32)
    dK_direct = torch.zeros((B, N, H, D), device=device, dtype=torch.float32)
    dV_direct = torch.zeros((B, N, H, D), device=device, dtype=torch.float32)
    dPmax = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=torch.float32)
    dPden = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=torch.float32)
    dPnum = torch.zeros((BH, NUM_CHUNKS, M, D), device=device, dtype=torch.float32)
    dCmax = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=torch.float32)
    dCden = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=torch.float32)
    dCnum = torch.zeros((BH, NUM_CHUNKS, M, D), device=device, dtype=torch.float32)

    clamp_max = _get_exp_clamp_for_dtype(dtype)
    flare_chunk_bwd[(BH, NUM_CHUNKS)](
        K, Q, V,
        prefix_max.view(BH, NUM_CHUNKS, M), prefix_den.view(BH, NUM_CHUNKS, M), prefix_num.view(BH, NUM_CHUNKS, M, D),
        dO,
        dQ_direct, dK_direct, dV_direct,
        dPmax, dPden, dPnum,
        *K.stride(), *Q.stride(), *V.stride(),
        *prefix_max.view(BH, NUM_CHUNKS, M).stride(), *prefix_den.view(BH, NUM_CHUNKS, M).stride(),
        *prefix_num.view(BH, NUM_CHUNKS, M, D).stride(),
        *dO.stride(),
        *dQ_direct.stride(),
        *dK_direct.stride(),
        *dV_direct.stride(),
        *dPmax.stride(), *dPden.stride(), *dPnum.stride(),
        BH, M, N, D, scale, 1e-6, clamp_max,
        CHUNK_SIZE=CHUNK_SIZE,
        USE_FP16=False,
        USE_BF16=False,
        USE_FP32_STATS=True,
        INPUT_PRECISION="ieee",
        USE_DENSE_DDEN=False,
        H=H,
    )

    flare_chunk_prefix_bwd[(BH,)](
        chunk_max.view(BH, NUM_CHUNKS, M), chunk_den.view(BH, NUM_CHUNKS, M), chunk_num.view(BH, NUM_CHUNKS, M, D),
        prefix_max.view(BH, NUM_CHUNKS, M), prefix_den.view(BH, NUM_CHUNKS, M), prefix_num.view(BH, NUM_CHUNKS, M, D),
        dPmax, dPden, dPnum,
        dCmax, dCden, dCnum,
        *chunk_max.view(BH, NUM_CHUNKS, M).stride(),
        *chunk_den.view(BH, NUM_CHUNKS, M).stride(),
        *chunk_num.view(BH, NUM_CHUNKS, M, D).stride(),
        *prefix_max.view(BH, NUM_CHUNKS, M).stride(),
        *prefix_den.view(BH, NUM_CHUNKS, M).stride(),
        *prefix_num.view(BH, NUM_CHUNKS, M, D).stride(),
        *dPmax.stride(), *dPden.stride(), *dPnum.stride(),
        *dCmax.stride(), *dCden.stride(), *dCnum.stride(),
        BH, M, D, NUM_CHUNKS,
        USE_FP16=False,
        USE_BF16=False,
    )

    _print_diff("dPmax", dPmax.view_as(dPmax_ref), dPmax_ref)
    _print_diff("dPden", dPden.view_as(dPden_ref), dPden_ref)
    _print_diff("dPnum", dPnum.view_as(dPnum_ref), dPnum_ref)
    _print_diff("dCmax", dCmax.view_as(dCmax_ref), dCmax_ref)
    _print_diff("dCden", dCden.view_as(dCden_ref), dCden_ref)
    _print_diff("dCnum", dCnum.view_as(dCnum_ref), dCnum_ref)
    _print_diff("dQ_phase3", dQ_direct, dQ_ref)

    # Chunk-stats backward using triton dC*
    dCmax_t = dCmax.view_as(chunk_max_t)
    dCden_t = dCden.view_as(chunk_den_t)
    dCnum_t = dCnum.view_as(chunk_num_t)
    dQ_prefix, dK_prefix, dV_prefix = torch.autograd.grad(
        outputs=(chunk_max_t, chunk_den_t, chunk_num_t),
        inputs=(Q, K, V),
        grad_outputs=(dCmax_t, dCden_t, dCnum_t),
        retain_graph=False,
        create_graph=False,
    )
    dQ_total = dQ_direct + dQ_prefix
    dK_total = dK_direct + dK_prefix
    dV_total = dV_direct + dV_prefix

    _print_diff("dQ_total", dQ_total, dQ_ref)
    _print_diff("dK_total", dK_total, dK_ref)
    _print_diff("dV_total", dV_total, dV_ref)


@triton.jit
def flare_chunk_fwd3_debug(
    K_ptr, Q_ptr, V_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    S_ptr, ExpS_ptr, P_ptr, Den_ptr, ExpA_ptr, W_ptr, O_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    stride_s_bh, stride_s_chunk, stride_s_c, stride_s_m,
    stride_exps_bh, stride_exps_chunk, stride_exps_c, stride_exps_m,
    stride_p_bh, stride_p_chunk, stride_p_c, stride_p_m,
    stride_den_bh, stride_den_chunk, stride_den_c, stride_den_m,
    stride_expa_bh, stride_expa_chunk, stride_expa_c, stride_expa_m,
    stride_w_bh, stride_w_chunk, stride_w_c, stride_w_s,
    stride_o_bh, stride_o_chunk, stride_o_c, stride_o_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale, eps, clamp_max,
    CHUNK_SIZE: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H

    chunk_start = chunk_idx * CHUNK_SIZE

    c_offsets = tl.arange(0, CHUNK_SIZE)
    m_offsets = tl.arange(0, M)
    d_offsets = tl.arange(0, D)

    token_idx = chunk_start + c_offsets
    token_mask = token_idx < N  # [C]
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_md = mask_m[:, None] & mask_d[None, :]
    mask_kv = token_mask[:, None] & mask_d[None, :]

    Q_base_ptr = Q_ptr + pid_h * stride_q_h
    Q_vals = tl.load(
        Q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets[None, :] * stride_q_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    Q_vals_f = Q_vals.to(tl.float32)

    pmax_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + chunk_idx * stride_pmax_chunk
    pden_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + chunk_idx * stride_pden_chunk
    pnum_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + chunk_idx * stride_pnum_chunk

    prefix_max = tl.load(
        pmax_ptr + m_offsets * stride_pmax_m,
        mask=mask_m,
        other=-float("inf"),
    ).to(tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    prefix_den = tl.load(
        pden_ptr + m_offsets * stride_pden_m,
        mask=mask_m,
        other=0.0,
    ).to(tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    prefix_num = tl.load(
        pnum_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))

    prefix_max_f = tl.minimum(prefix_max.to(tl.float32), clamp_max)
    exp_prev_max = tl.exp(prefix_max_f)
    sum_prev_exp = prefix_den.to(tl.float32) * exp_prev_max
    sum_prev_exp_v = prefix_num.to(tl.float32) * exp_prev_max[:, None]

    K_base_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + chunk_start * stride_k_n
    V_base_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + chunk_start * stride_v_n
    K_chunk = tl.load(
        K_base_ptr + c_offsets[:, None] * stride_k_n + d_offsets[None, :] * stride_k_d,
        mask=mask_kv,
        other=0.0,
    ).to(tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    V_chunk = tl.load(
        V_base_ptr + c_offsets[:, None] * stride_v_n + d_offsets[None, :] * stride_v_d,
        mask=mask_kv,
        other=0.0,
    ).to(tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    K_chunk_f = K_chunk.to(tl.float32)
    V_chunk_f = V_chunk.to(tl.float32)

    S = tl.dot(K_chunk_f, tl.trans(Q_vals_f)) * scale
    S = tl.where(token_mask[:, None] & mask_m[None, :], S, -float("inf"))
    S_exp = tl.minimum(S, clamp_max)
    expS = tl.exp(S_exp)
    expS = tl.where(token_mask[:, None] & mask_m[None, :], expS, 0.0)
    den_chunk = tl.cumsum(expS, axis=0)
    den_total = den_chunk + sum_prev_exp[None, :]

    s_max = tl.max(S, axis=1)
    s_max = tl.where(token_mask, s_max, 0.0)
    s_exp = tl.exp(S - s_max[:, None])
    s_exp = tl.where(token_mask[:, None] & mask_m[None, :], s_exp, 0.0)
    s_sum = tl.sum(s_exp, axis=1)
    P = tl.where(token_mask[:, None], s_exp / (s_sum[:, None] + 1e-20), 0.0)

    expA = P / (den_total + eps)
    expA = tl.where(token_mask[:, None] & mask_m[None, :], expA, 0.0)

    O_prev = tl.dot(expA, sum_prev_exp_v)
    W = tl.dot(expA, tl.trans(expS))
    causal = c_offsets[None, :] <= c_offsets[:, None]
    causal = causal & token_mask[:, None] & token_mask[None, :]
    W = tl.where(causal, W, 0.0)
    O_curr = tl.dot(W, V_chunk_f)
    O_out = O_prev + O_curr

    s_ptr = S_ptr + pid_bh * stride_s_bh + chunk_idx * stride_s_chunk
    exps_ptr = ExpS_ptr + pid_bh * stride_exps_bh + chunk_idx * stride_exps_chunk
    p_ptr = P_ptr + pid_bh * stride_p_bh + chunk_idx * stride_p_chunk
    den_ptr = Den_ptr + pid_bh * stride_den_bh + chunk_idx * stride_den_chunk
    expa_ptr = ExpA_ptr + pid_bh * stride_expa_bh + chunk_idx * stride_expa_chunk
    w_ptr = W_ptr + pid_bh * stride_w_bh + chunk_idx * stride_w_chunk
    o_ptr = O_ptr + pid_bh * stride_o_bh + chunk_idx * stride_o_chunk

    tl.store(s_ptr + c_offsets[:, None] * stride_s_c + m_offsets[None, :] * stride_s_m, S, mask=token_mask[:, None] & mask_m[None, :])
    tl.store(exps_ptr + c_offsets[:, None] * stride_exps_c + m_offsets[None, :] * stride_exps_m, expS, mask=token_mask[:, None] & mask_m[None, :])
    tl.store(p_ptr + c_offsets[:, None] * stride_p_c + m_offsets[None, :] * stride_p_m, P, mask=token_mask[:, None] & mask_m[None, :])
    tl.store(den_ptr + c_offsets[:, None] * stride_den_c + m_offsets[None, :] * stride_den_m, den_total, mask=token_mask[:, None] & mask_m[None, :])
    tl.store(expa_ptr + c_offsets[:, None] * stride_expa_c + m_offsets[None, :] * stride_expa_m, expA, mask=token_mask[:, None] & mask_m[None, :])
    tl.store(w_ptr + c_offsets[:, None] * stride_w_c + c_offsets[None, :] * stride_w_s, W, mask=token_mask[:, None] & token_mask[None, :])
    tl.store(o_ptr + c_offsets[:, None] * stride_o_c + d_offsets[None, :] * stride_o_d, O_out, mask=token_mask[:, None] & mask_d[None, :])


def _debug_triton_fwd_compare():
    if not torch.cuda.is_available():
        print("[FLARE DEBUG] CUDA not available, skipping Triton3 forward debug.")
        return
    device = torch.device("cuda")
    B, H, M, N, D = 1, 1, 16, 128, 16
    CHUNK_SIZE = 64
    NUM_CHUNKS = math.ceil(N / CHUNK_SIZE)
    scale = 1.0 / math.sqrt(D)
    eps = 1e-6
    dtype = torch.float32
    clamp_max = _get_exp_clamp_for_dtype(dtype)

    Q = torch.randn(H, M, D, device=device, dtype=dtype)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype)

    Kc = K.reshape(B, NUM_CHUNKS, CHUNK_SIZE, H, D).permute(0, 3, 1, 2, 4).contiguous()
    Vc = V.reshape(B, NUM_CHUNKS, CHUNK_SIZE, H, D).permute(0, 3, 1, 2, 4).contiguous()

    score_chunk = scale * torch.einsum("bhncd,hmd->bhncm", Kc, Q)
    chunk_max = score_chunk.max(dim=3).values
    expS_chunkstats = torch.exp(score_chunk - chunk_max.unsqueeze(3))
    chunk_den = expS_chunkstats.sum(dim=3)
    chunk_num = torch.einsum("bhncm,bhncd->bhnmd", expS_chunkstats, Vc)

    prefix_max = torch.empty((B, H, NUM_CHUNKS, M), device=device, dtype=dtype)
    prefix_den = torch.zeros((B, H, NUM_CHUNKS, M), device=device, dtype=dtype)
    prefix_num = torch.zeros((B, H, NUM_CHUNKS, M, D), device=device, dtype=dtype)

    max_curr = torch.full((B, H, M), -float("inf"), device=device, dtype=dtype)
    den_curr = torch.zeros((B, H, M), device=device, dtype=dtype)
    num_curr = torch.zeros((B, H, M, D), device=device, dtype=dtype)
    for chunk_idx in range(NUM_CHUNKS):
        prefix_max[:, :, chunk_idx, :] = max_curr
        prefix_den[:, :, chunk_idx, :] = den_curr
        prefix_num[:, :, chunk_idx, :, :] = num_curr
        sc_max = chunk_max[:, :, chunk_idx, :]
        sc_den = chunk_den[:, :, chunk_idx, :]
        sc_num = chunk_num[:, :, chunk_idx, :, :]
        max_new = torch.maximum(max_curr, sc_max)
        rescale_prev = torch.exp(max_curr - max_new)
        rescale_curr = torch.exp(sc_max - max_new)
        den_curr = den_curr * rescale_prev + sc_den * rescale_curr
        num_curr = num_curr * rescale_prev.unsqueeze(-1) + sc_num * rescale_curr.unsqueeze(-1)
        max_curr = max_new

    def _stats(name, t):
        t_f = t.float()
        nan = torch.isnan(t_f).sum().item()
        inf = torch.isinf(t_f).sum().item()
        print(f"[FLARE DEBUG] {name}: mean={t_f.mean().item():.3e}, max={t_f.max().item():.3e}, "
              f"min={t_f.min().item():.3e}, nan={nan}, inf={inf}")

    def _print_diff(name, a, b):
        diff = (a - b).float().abs()
        print(f"[FLARE DEBUG] {name} diff: mean={diff.mean().item():.3e}, max={diff.max().item():.3e}")

    # Torch reference for all chunks
    S_ref = []
    expS_ref = []
    P_ref = []
    den_ref = []
    expA_ref = []
    W_ref = []
    O_ref = []
    for chunk_idx in range(NUM_CHUNKS):
        score = score_chunk[:, :, chunk_idx, :, :]  # [B,H,C,M]
        expS_chunk = torch.exp(score.clamp(max=clamp_max))
        den_chunk = torch.cumsum(expS_chunk, dim=2)
        exp_prev_max = torch.exp(prefix_max[:, :, chunk_idx, :].clamp(max=clamp_max))
        sum_prev_exp = prefix_den[:, :, chunk_idx, :] * exp_prev_max
        sum_prev_exp_v = prefix_num[:, :, chunk_idx, :, :] * exp_prev_max.unsqueeze(-1)
        den_total = den_chunk + sum_prev_exp.unsqueeze(2)
        P_chunk = torch.softmax(score, dim=-1)
        expA_chunk = P_chunk / (den_total + eps)
        y_prev = torch.einsum("bhcm,bhmd->bhcd", expA_chunk, sum_prev_exp_v)
        W = torch.einsum("bhcm,bhsm->bhcs", expA_chunk, expS_chunk)
        causal = torch.tril(torch.ones((CHUNK_SIZE, CHUNK_SIZE), device=device, dtype=torch.bool))
        W = W.masked_fill(~causal[None, None, :, :], 0.0)
        y_curr = torch.einsum("bhcs,bhsd->bhcd", W, Vc[:, :, chunk_idx, :, :])
        Y = y_prev + y_curr
        S_ref.append(score)
        expS_ref.append(expS_chunk)
        P_ref.append(P_chunk)
        den_ref.append(den_total)
        expA_ref.append(expA_chunk)
        W_ref.append(W)
        O_ref.append(Y)

    S_ref = torch.stack(S_ref, dim=2)
    expS_ref = torch.stack(expS_ref, dim=2)
    P_ref = torch.stack(P_ref, dim=2)
    den_ref = torch.stack(den_ref, dim=2)
    expA_ref = torch.stack(expA_ref, dim=2)
    W_ref = torch.stack(W_ref, dim=2)
    O_ref = torch.stack(O_ref, dim=2)

    # Triton chunk/prefix stats for comparison
    BH = B * H
    chunk_max_t = torch.empty((BH, NUM_CHUNKS, M), device=device, dtype=torch.float32)
    chunk_den_t = torch.empty((BH, NUM_CHUNKS, M), device=device, dtype=torch.float32)
    chunk_num_t = torch.empty((BH, NUM_CHUNKS, M, D), device=device, dtype=torch.float32)
    prefix_max_t = torch.empty((BH, NUM_CHUNKS, M), device=device, dtype=torch.float32)
    prefix_den_t = torch.empty((BH, NUM_CHUNKS, M), device=device, dtype=torch.float32)
    prefix_num_t = torch.empty((BH, NUM_CHUNKS, M, D), device=device, dtype=torch.float32)

    BLOCK_M = 16
    NUM_M_BLOCKS = math.ceil(M / BLOCK_M)
    flare_chunk_prepare[(BH, NUM_CHUNKS, NUM_M_BLOCKS)](
        K, Q, V,
        chunk_max_t, chunk_den_t, chunk_num_t,
        *K.stride(), *Q.stride(), *V.stride(),
        *chunk_max_t.stride(), *chunk_den_t.stride(), *chunk_num_t.stride(),
        BH, M, N, D, scale,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_M=BLOCK_M,
        USE_FP16=False,
        USE_BF16=False,
        USE_FP32_STATS=True,
        INPUT_PRECISION="ieee",
        H=H,
    )

    flare_chunk_prefix[(BH,)](
        chunk_max_t, chunk_den_t, chunk_num_t,
        prefix_max_t, prefix_den_t, prefix_num_t,
        *chunk_max_t.stride(), *chunk_den_t.stride(), *chunk_num_t.stride(),
        *prefix_max_t.stride(), *prefix_den_t.stride(), *prefix_num_t.stride(),
        BH, M, D, NUM_CHUNKS,
        USE_FP16=False,
        USE_BF16=False,
        USE_FP32_STATS=True,
    )

    BH = B * H
    S_dbg = torch.empty((BH, NUM_CHUNKS, CHUNK_SIZE, M), device=device, dtype=torch.float32)
    expS_dbg = torch.empty_like(S_dbg)
    P_dbg = torch.empty_like(S_dbg)
    den_dbg = torch.empty_like(S_dbg)
    expA_dbg = torch.empty_like(S_dbg)
    W_dbg = torch.empty((BH, NUM_CHUNKS, CHUNK_SIZE, CHUNK_SIZE), device=device, dtype=torch.float32)
    O_dbg = torch.empty((BH, NUM_CHUNKS, CHUNK_SIZE, D), device=device, dtype=torch.float32)

    flare_chunk_fwd3_debug[(BH, NUM_CHUNKS)](
        K, Q, V,
        prefix_max.view(BH, NUM_CHUNKS, M), prefix_den.view(BH, NUM_CHUNKS, M), prefix_num.view(BH, NUM_CHUNKS, M, D),
        S_dbg, expS_dbg, P_dbg, den_dbg, expA_dbg, W_dbg, O_dbg,
        *K.stride(), *Q.stride(), *V.stride(),
        *prefix_max.view(BH, NUM_CHUNKS, M).stride(),
        *prefix_den.view(BH, NUM_CHUNKS, M).stride(),
        *prefix_num.view(BH, NUM_CHUNKS, M, D).stride(),
        *S_dbg.stride(), *expS_dbg.stride(), *P_dbg.stride(), *den_dbg.stride(), *expA_dbg.stride(),
        *W_dbg.stride(),
        *O_dbg.stride(),
        BH, M, N, D, scale, eps, clamp_max,
        CHUNK_SIZE=CHUNK_SIZE,
        USE_FP16=False,
        USE_BF16=False,
        H=H,
    )

    _stats("S_ref", S_ref)
    _stats("expS_ref", expS_ref)
    _stats("P_ref", P_ref)
    _stats("den_ref", den_ref)
    _stats("expA_ref", expA_ref)
    _stats("W_ref", W_ref)
    _stats("O_ref", O_ref)

    _print_diff("S", S_dbg.view_as(S_ref), S_ref)
    _print_diff("expS", expS_dbg.view_as(expS_ref), expS_ref)
    _print_diff("P", P_dbg.view_as(P_ref), P_ref)
    _print_diff("den_total", den_dbg.view_as(den_ref), den_ref)
    _print_diff("expA", expA_dbg.view_as(expA_ref), expA_ref)
    _print_diff("W", W_dbg.view_as(W_ref), W_ref)
    _print_diff("O", O_dbg.view_as(O_ref), O_ref)

    _print_diff("chunk_max", chunk_max_t.view_as(chunk_max), chunk_max)
    _print_diff("chunk_den", chunk_den_t.view_as(chunk_den), chunk_den)
    _print_diff("chunk_num", chunk_num_t.view_as(chunk_num), chunk_num)
    _print_diff("prefix_max", prefix_max_t.view_as(prefix_max), prefix_max)
    _print_diff("prefix_den", prefix_den_t.view_as(prefix_den), prefix_den)
    _print_diff("prefix_num", prefix_num_t.view_as(prefix_num), prefix_num)

#======================================================================#
def _debug_triton_stable_kernel():
    if not torch.cuda.is_available():
        print("[FLARE DEBUG] CUDA not available, skipping stable kernel debug.")
        return
    device = torch.device("cuda")
    B, H, M, N, D = 1, 1, 64, 128, 32
    CHUNK_SIZE = 64
    scale = 1.0
    dtype = torch.float32
    M_DEBUG = M
    clamp_max = _get_exp_clamp_for_dtype(dtype)
    eps = _get_eps_for_dtype(dtype)

    torch.manual_seed(0)
    Q = torch.randn(H, M, D, device=device, dtype=dtype)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype)

    # Prefix stats (Triton)
    BH = B * H
    NUM_CHUNKS = math.ceil(N / CHUNK_SIZE)
    chunk_max_t = torch.full((BH, NUM_CHUNKS, M), -float("inf"), device=device, dtype=torch.float32)
    chunk_den_t = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=torch.float32)
    chunk_num_t = torch.zeros((BH, NUM_CHUNKS, M, D), device=device, dtype=torch.float32)

    BLOCK_M = 32
    NUM_M_BLOCKS = math.ceil(M / BLOCK_M)
    flare_chunk_prepare[(BH, NUM_CHUNKS, NUM_M_BLOCKS)](
        K, Q, V,
        chunk_max_t, chunk_den_t, chunk_num_t,
        *K.stride(), *Q.stride(), *V.stride(),
        *chunk_max_t.stride(), *chunk_den_t.stride(), *chunk_num_t.stride(),
        BH, M, N, D, scale,
        CHUNK_SIZE=CHUNK_SIZE,
        BLOCK_M=BLOCK_M,
        USE_FP16=False,
        USE_BF16=False,
        USE_FP32_STATS=True,
        INPUT_PRECISION="tf32",
        H=H,
    )

    prefix_max_t = torch.empty((BH, NUM_CHUNKS, M), device=device, dtype=torch.float32)
    prefix_den_t = torch.zeros((BH, NUM_CHUNKS, M), device=device, dtype=torch.float32)
    prefix_num_t = torch.zeros((BH, NUM_CHUNKS, M, D), device=device, dtype=torch.float32)
    flare_chunk_prefix[(BH,)](
        chunk_max_t, chunk_den_t, chunk_num_t,
        prefix_max_t, prefix_den_t, prefix_num_t,
        *chunk_max_t.stride(), *chunk_den_t.stride(), *chunk_num_t.stride(),
        *prefix_max_t.stride(), *prefix_den_t.stride(), *prefix_num_t.stride(),
        BH, M, D, NUM_CHUNKS,
        USE_FP16=False,
        USE_BF16=False,
        USE_FP32_STATS=True,
    )

    # Debug buffers [C, M_DEBUG] + output [C, D]
    dbg_shape = (CHUNK_SIZE, M_DEBUG)
    dbg_s = torch.empty(dbg_shape, device=device, dtype=torch.float32)
    dbg_m = torch.empty(dbg_shape, device=device, dtype=torch.float32)
    dbg_l = torch.empty(dbg_shape, device=device, dtype=torch.float32)
    dbg_den = torch.empty(dbg_shape, device=device, dtype=torch.float32)
    dbg_a = torch.empty(dbg_shape, device=device, dtype=torch.float32)
    dbg_o = torch.empty((CHUNK_SIZE, D), device=device, dtype=torch.float32)

    flare_chunk_fwd3_stable_debug[(BH, NUM_CHUNKS)](
        K, Q, V,
        prefix_max_t, prefix_den_t, prefix_num_t,
        dbg_s, dbg_m, dbg_l, dbg_den, dbg_a, dbg_o,
        *K.stride(), *Q.stride(), *V.stride(),
        *prefix_max_t.stride(), *prefix_den_t.stride(), *prefix_num_t.stride(),
        *dbg_s.stride(),
        *dbg_o.stride(),
        BH, M, N, D, scale, eps, clamp_max,
        CHUNK_SIZE=CHUNK_SIZE,
        H=H,
    )

    # Python reference for chunk 1
    chunk_idx = 1
    Kc = K.reshape(B, NUM_CHUNKS, CHUNK_SIZE, H, D).permute(0, 3, 1, 2, 4).contiguous()
    Vc = V.reshape(B, NUM_CHUNKS, CHUNK_SIZE, H, D).permute(0, 3, 1, 2, 4).contiguous()
    S_chunk = scale * torch.einsum("bhcd,hmd->bhcm", Kc[:, :, chunk_idx], Q)
    V_chunk = Vc[:, :, chunk_idx]

    prefix_max = prefix_max_t.view(B, H, NUM_CHUNKS, M)[:, :, chunk_idx]
    prefix_den = prefix_den_t.view(B, H, NUM_CHUNKS, M)[:, :, chunk_idx]
    prefix_num = prefix_num_t.view(B, H, NUM_CHUNKS, M, D)[:, :, chunk_idx]

    m_state = prefix_max.clone()
    l_state = prefix_den.clone()
    n_state = prefix_num.clone()
    o_ref = torch.empty((CHUNK_SIZE, D), device=device, dtype=torch.float32)
    s_ref = torch.empty((CHUNK_SIZE, M_DEBUG), device=device, dtype=torch.float32)
    m_ref = torch.empty((CHUNK_SIZE, M_DEBUG), device=device, dtype=torch.float32)
    l_ref = torch.empty((CHUNK_SIZE, M_DEBUG), device=device, dtype=torch.float32)
    den_ref = torch.empty((CHUNK_SIZE, M_DEBUG), device=device, dtype=torch.float32)
    a_ref = torch.empty((CHUNK_SIZE, M_DEBUG), device=device, dtype=torch.float32)

    for t in range(CHUNK_SIZE):
        s_t = S_chunk[:, :, t].clamp(max=clamp_max)
        v_t = V_chunk[:, :, t]
        m_new = torch.maximum(m_state, s_t)
        same_inf = torch.isneginf(m_state) & torch.isneginf(m_new)
        exp_prev = torch.where(
            same_inf,
            torch.ones_like(m_new),
            torch.where(torch.isneginf(m_state), torch.zeros_like(m_new), torch.exp(m_state - m_new)),
        )
        exp_s = torch.where(torch.isneginf(m_new), torch.zeros_like(m_new), torch.exp(s_t - m_new))
        l_state = l_state * exp_prev + exp_s
        n_state = n_state * exp_prev.unsqueeze(-1) + exp_s.unsqueeze(-1) * v_t.unsqueeze(-2)
        m_state = m_new

        exp_m = torch.exp(torch.clamp(m_state, max=clamp_max))
        sum_exp = l_state * exp_m
        sum_exp_v = n_state * exp_m.unsqueeze(-1)
        den_total = sum_exp

        P_t = torch.softmax(s_t, dim=-1)
        expA_t = P_t / (den_total + eps)
        o_t = torch.einsum("bhm,bhmd->bhd", expA_t, sum_exp_v)

        s_ref[t] = s_t[0, 0, :M_DEBUG]
        m_ref[t] = m_state[0, 0, :M_DEBUG]
        l_ref[t] = l_state[0, 0, :M_DEBUG]
        den_ref[t] = den_total[0, 0, :M_DEBUG]
        a_ref[t] = expA_t[0, 0, :M_DEBUG]
        o_ref[t] = o_t[0, 0]

    def _cmp(name, a, b):
        diff = (a - b).abs()
        print(f"[FLARE DEBUG] {name} diff mean={diff.mean().item():.3e} max={diff.max().item():.3e}")

    _cmp("s_t", dbg_s, s_ref)
    _cmp("m_state", dbg_m, m_ref)
    _cmp("l_state", dbg_l, l_ref)
    _cmp("den_total", dbg_den, den_ref)
    _cmp("expA_t", dbg_a, a_ref)
    _cmp("o_t", dbg_o, o_ref)

#======================================================================#
def measure_memory(func, *args, **kwargs):
    """Measure peak memory usage of a function using Triton's memory profiling."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_mem = torch.cuda.memory_allocated()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        peak_mem = (torch.cuda.max_memory_allocated() - start_mem) / 1e9
        return result, peak_mem
    else:
        return func(*args, **kwargs), 0

def compute_errors(y_pred, y_ref, name):
    """Compute detailed error metrics between predicted and reference outputs."""
    y_pred_f = y_pred.float()
    y_ref_f = y_ref.float()

    abs_err = (y_pred_f - y_ref_f).abs()
    ref_magnitude = y_ref_f.abs()
    rel_err = abs_err / (ref_magnitude + 1e-8)

    atol, rtol = _get_allclose_tols(y_pred.dtype)
    allclose = torch.allclose(y_pred_f, y_ref_f, atol=atol, rtol=rtol)

    return {
        f'{name}_max_abs_err': abs_err.max().item(),
        f'{name}_mean_abs_err': abs_err.mean().item(),
        f'{name}_max_rel_err': rel_err.max().item(),
        f'{name}_mean_rel_err': rel_err.mean().item(),
        f'{name}_allclose': allclose,
    }


def _strict_mode_enabled(env_key: str, default: bool = False) -> bool:
    val = os.environ.get(env_key, "")
    if not val:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(env_key: str, default: float) -> float:
    val = os.environ.get(env_key, "")
    return float(val) if val else float(default)


def _maybe_record_failure(failures: list[str], cond: bool, msg: str) -> None:
    if cond:
        failures.append(msg)


def _precision_modes_from_env(env_key: str, default: str = "ieee,tf32,tf32x3") -> list[str]:
    raw = os.environ.get(env_key, default)
    modes = [m.strip().lower() for m in raw.split(",") if m.strip()]
    valid = {"ieee", "tf32", "tf32x3"}
    bad = [m for m in modes if m not in valid]
    if bad:
        raise ValueError(f"{env_key} has invalid precision(s): {bad}. Expected subset of {sorted(valid)}.")
    if not modes:
        raise ValueError(f"{env_key} produced no precision modes.")
    return modes


def _parse_bhmnd_configs(env_key: str, default: str) -> list[dict[str, int]]:
    raw = os.environ.get(env_key, default)
    cfgs: list[dict[str, int]] = []
    for spec in raw.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        b, h, m, n, d = (int(x.strip()) for x in spec.split(","))
        cfgs.append(dict(B=b, H=h, M=m, N=n, D=d))
    if not cfgs:
        raise ValueError(f"{env_key} produced no configs.")
    return cfgs


def _parse_decode_separation_modes(env_key: str, default: str = "00,10,01,11") -> list[tuple[bool, bool]]:
    raw = os.environ.get(env_key, default)
    modes: list[tuple[bool, bool]] = []
    valid = {"00", "10", "01", "11"}
    for spec in raw.split(","):
        spec = spec.strip()
        if not spec:
            continue
        if spec not in valid:
            raise ValueError(
                f"{env_key} has invalid mode '{spec}'. Expected comma-separated subset of {sorted(valid)}."
            )
        modes.append((spec[0] == "1", spec[1] == "1"))
    if not modes:
        raise ValueError(f"{env_key} produced no decode separation modes.")
    return modes


def _decode_mode_label(separate_q_dec: bool, separate_k_dec: bool) -> str:
    return f"separate_q_dec={int(separate_q_dec)} separate_k_dec={int(separate_k_dec)}"

def _tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a_f = a.float().reshape(-1)
    b_f = b.float().reshape(-1)
    denom = (a_f.norm() * b_f.norm()).item()
    if denom == 0.0:
        return 1.0 if torch.allclose(a_f, b_f) else 0.0
    return (torch.dot(a_f, b_f) / denom).item()

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
    from .chunked_old import ChunkedFLAREOld

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
                    Yo = ChunkedFLAREOld.apply(Qo, Ko, Vo, scale)
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
    from .chunked_old import ChunkedFLAREOld
    failures: list[str] = []
    loss_delta_max = _env_float("FLARE_TRAINLIKE_PARITY_LOSS_DELTA_MAX", 5e-3)
    grad_cos_min = _env_float("FLARE_TRAINLIKE_PARITY_GRAD_COS_MIN", -1.0)
    output_mean_abs_max = _env_float("FLARE_TRAINLIKE_PARITY_OUTPUT_MEAN_ABS_MAX", 5e-3)
    output_max_abs_max = _env_float("FLARE_TRAINLIKE_PARITY_OUTPUT_MAX_ABS_MAX", 3.0)
    check_grads = _strict_mode_enabled("FLARE_TRAINLIKE_PARITY_CHECK_GRADS", default=False)
    grad_rel_l2_max = _env_float("FLARE_TRAINLIKE_PARITY_GRAD_REL_L2_MAX", 2.0)

    print("=" * 100)
    print("[FLARE TRAINLIKE PARITY] Multi-step new-vs-old parity")
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
                    y_new = flare_chunk_triton(latent_q_new, k_new, v_new, scale)
                    y_old = ChunkedFLAREOld.apply(latent_q_old, k_old, v_old, scale)

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


def _regression_test():
    if not torch.cuda.is_available():
        raise RuntimeError("FLARE regression test requires CUDA.")

    # Deterministic, bounded runtime defaults for CI/local gating.
    os.environ["FLARE_CORRECTNESS_STRICT"] = "1"
    os.environ["FLARE_CORRECTNESS_DTYPES"] = os.environ.get("FLARE_CORRECTNESS_DTYPES", "bfloat16")
    os.environ["FLARE_CORRECTNESS_SHAPES"] = os.environ.get("FLARE_CORRECTNESS_SHAPES", "1,2,512,128,32")
    os.environ["FLARE_CORRECTNESS_QK_STDS"] = os.environ.get("FLARE_CORRECTNESS_QK_STDS", "1.0")
    os.environ["FLARE_CORRECTNESS_DECODE_SEPARATION_MODES"] = os.environ.get(
        "FLARE_CORRECTNESS_DECODE_SEPARATION_MODES", "00,10,01,11"
    )
    os.environ["FLARE_CORRECTNESS_GRAD"] = os.environ.get("FLARE_CORRECTNESS_GRAD", "1")
    os.environ["FLARE_CORRECTNESS_GRAD_LIMIT"] = os.environ.get("FLARE_CORRECTNESS_GRAD_LIMIT", "4")

    os.environ["FLARE_PARITY_STRICT"] = "1"
    os.environ["FLARE_PARITY_B"] = os.environ.get("FLARE_PARITY_B", "1")
    os.environ["FLARE_PARITY_H"] = os.environ.get("FLARE_PARITY_H", "8")
    os.environ["FLARE_PARITY_M"] = os.environ.get("FLARE_PARITY_M", "128")
    os.environ["FLARE_PARITY_N"] = os.environ.get("FLARE_PARITY_N", "512")
    os.environ["FLARE_PARITY_D"] = os.environ.get("FLARE_PARITY_D", "32")
    os.environ["FLARE_PARITY_DTYPE"] = os.environ.get("FLARE_PARITY_DTYPE", "bfloat16")
    os.environ["FLARE_PARITY_TRAIN_STEPS"] = os.environ.get("FLARE_PARITY_TRAIN_STEPS", "0")

    os.environ["FLARE_TRAINLIKE_STRICT"] = "1"
    os.environ["FLARE_TRAINLIKE_COMPARE"] = os.environ.get("FLARE_TRAINLIKE_COMPARE", "1")
    os.environ["FLARE_TRAINLIKE_STEPS"] = os.environ.get("FLARE_TRAINLIKE_STEPS", "2")
    os.environ["FLARE_TRAINLIKE_CONFIGS"] = os.environ.get("FLARE_TRAINLIKE_CONFIGS", "1,8,128,1024,32")
    os.environ["FLARE_TRAINLIKE_DTYPE"] = os.environ.get("FLARE_TRAINLIKE_DTYPE", "bfloat16")
    os.environ["FLARE_SHARP_BWD_DECODE_SEPARATION_MODES"] = os.environ.get(
        "FLARE_SHARP_BWD_DECODE_SEPARATION_MODES", "00,10,01,11"
    )

    print("=" * 100)
    print("[FLARE REGRESSION] correctness suite")
    print("=" * 100)
    _run_correctness_suite()

    print("=" * 100)
    print("[FLARE REGRESSION] parity suite")
    print("=" * 100)
    _parity_tests()

    print("=" * 100)
    print("[FLARE REGRESSION] trainlike suite")
    print("=" * 100)
    _trainlike_sanity()

    if _strict_mode_enabled("FLARE_REGRESSION_EXTENDED", default=False):
        print("=" * 100)
        print("[FLARE REGRESSION] long-context suite")
        print("=" * 100)
        _long_context_accuracy_suite()

        print("=" * 100)
        print("[FLARE REGRESSION] trainlike multi-step parity")
        print("=" * 100)
        _trainlike_multistep_parity()

        print("=" * 100)
        print("[FLARE REGRESSION] chunk-size sensitivity")
        print("=" * 100)
        _chunk_size_sensitivity_suite()

        print("=" * 100)
        print("[FLARE REGRESSION] sharp backward regression")
        print("=" * 100)
        _sharp_softmax_bwd_regression_suite()

    print("[FLARE REGRESSION] all checks passed.")


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

def _run_recurrent_flare_test():
    device = torch.device("cuda")
    dtype_name = os.environ.get("FLARE_RECURRENT_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name)
    B = int(os.environ.get("FLARE_RECURRENT_B", "1"))
    H = int(os.environ.get("FLARE_RECURRENT_H", "4"))
    M = int(os.environ.get("FLARE_RECURRENT_M", "64"))
    T = int(os.environ.get("FLARE_RECURRENT_T", "128"))
    D = int(os.environ.get("FLARE_RECURRENT_D", "32"))

    torch.manual_seed(0)
    Q = torch.randn(H, M, D, device=device, dtype=dtype)
    K = torch.randn(B, T, H, D, device=device, dtype=dtype)
    V = torch.randn(B, T, H, D, device=device, dtype=dtype)
    scale = float(os.environ.get("FLARE_RECURRENT_SCALE", str(D ** -0.5)))
    backend = os.environ.get("FLARE_RECURRENT_BACKEND", "triton").strip().lower()

    print("Running Reference (causal) for recurrent compare...", flush=True)
    Y_ref = flare_causal_reference(Q, K, V, scale=scale).permute(0, 2, 1, 3).contiguous()
    print("Running RecurrentFLARE PyTorch...", flush=True)
    Y_pt = flare_recurrent_pytorch(Q, K, V, scale=scale)
    print(f"Running RecurrentFLARE ({backend})...", flush=True)
    Y_tr = RecurrentFLARE.apply(Q, K, V, scale)

    atol = 1e-3 if dtype in (torch.float16, torch.bfloat16) else 1e-5
    _print_err_report("RecurrentFLARE PyTorch vs Ref", Y_pt, Y_ref, atol)
    _print_err_report("RecurrentFLARE Triton vs Ref", Y_tr, Y_ref, atol)
    _check_finite("RecurrentFLARE.Y_triton", Y_tr)


def _run_cached_impl_test():
    if not torch.cuda.is_available():
        print("[FLARE CACHED TEST] CUDA not available, skipping.")
        return

    device = torch.device("cuda")
    dtype_name = os.environ.get("FLARE_CACHED_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name)

    B = int(os.environ.get("FLARE_CACHED_B", "2"))
    H = int(os.environ.get("FLARE_CACHED_H", "4"))
    M = int(os.environ.get("FLARE_CACHED_M", "32"))
    T = int(os.environ.get("FLARE_CACHED_T", "17"))
    D = int(os.environ.get("FLARE_CACHED_D", "32"))
    T_NEXT = int(os.environ.get("FLARE_CACHED_TNEXT", "1"))
    T_CONT = int(os.environ.get("FLARE_CACHED_TCONT", "3"))
    mask_prob = float(os.environ.get("FLARE_CACHED_MASK_PROB", "0.2"))
    scale_mode = os.environ.get("FLARE_CACHED_SCALE", "sqrt")
    scale = (D ** -0.5) if scale_mode == "sqrt" else 1.0
    seed = int(os.environ.get("FLARE_CACHED_SEED", "0"))
    atol_default, rtol_default = _get_allclose_tols(dtype)
    atol = float(os.environ.get("FLARE_CACHED_ATOL", str(atol_default)))
    rtol = float(os.environ.get("FLARE_CACHED_RTOL", str(rtol_default)))
    ref_atol_default, ref_rtol_default = _get_allclose_tols(dtype)
    ref_atol = float(os.environ.get("FLARE_CACHED_REF_ATOL", str(ref_atol_default)))
    ref_rtol = float(os.environ.get("FLARE_CACHED_REF_RTOL", str(ref_rtol_default)))

    try:
        from fla.models.flare.flare_decoder import FLAREDecoder  # type: ignore
        from fla.models.utils import Cache  # type: ignore
    except Exception as exc:
        print(f"[FLARE CACHED TEST] FLA integration modules unavailable, skipping: {exc}")
        return

    torch.manual_seed(seed)
    Q = torch.randn(H, M, D, device=device, dtype=dtype)
    K = torch.randn(B, T, H, D, device=device, dtype=dtype)
    V = torch.randn(B, T, H, D, device=device, dtype=dtype)
    mask = (torch.rand(B, T, device=device) > mask_prob).to(torch.int32)

    print(
        f"[FLARE CACHED TEST] dtype={dtype} B={B} H={H} M={M} T={T} D={D} "
        f"T_NEXT={T_NEXT} T_CONT={T_CONT} scale={scale:.6g}"
    )

    # 1) Prefill parity: PyTorch vs Triton
    Y_py, S_py = flare_prefill_pytorch(Q, K, V, scale=scale, attention_mask=mask)
    Y_tr, S_tr = flare_prefill_triton(Q, K, V, scale=scale, attention_mask=mask)
    _print_err_report("Cached Prefill Triton vs PyTorch", Y_tr, Y_py, atol)
    if not torch.allclose(Y_tr.float(), Y_py.float(), atol=atol, rtol=rtol):
        raise AssertionError("Cached prefill mismatch between Triton and PyTorch.")

    # 2) Decode parity: PyTorch vs Triton
    K_next = torch.randn(B, T_NEXT, H, D, device=device, dtype=dtype)
    V_next = torch.randn(B, T_NEXT, H, D, device=device, dtype=dtype)
    mask_next = (torch.rand(B, T_NEXT, device=device) > mask_prob).to(torch.int32)
    Yd_py, _ = flare_decode_pytorch(Q, K_next, V_next, S_py, scale=scale, attention_mask=mask_next)
    Yd_tr, _ = flare_decode_triton(Q, K_next, V_next, S_tr, scale=scale, attention_mask=mask_next)
    _print_err_report("Cached Decode Triton vs PyTorch", Yd_tr, Yd_py, atol)
    if not torch.allclose(Yd_tr.float(), Yd_py.float(), atol=atol, rtol=rtol):
        raise AssertionError("Cached decode mismatch between Triton and PyTorch.")

    # 3) Continuation consistency
    Y_full_py, _ = flare_prefill_pytorch(
        Q,
        torch.cat([K, K_next], dim=1),
        torch.cat([V, V_next], dim=1),
        scale=scale,
        attention_mask=torch.cat([mask, mask_next], dim=1),
    )
    _print_err_report("Cached Continuation Last Token", Yd_py, Y_full_py[:, -T_NEXT:], atol)
    if not torch.allclose(Yd_py.float(), Y_full_py[:, -T_NEXT:].float(), atol=atol, rtol=rtol):
        raise AssertionError("Cached continuation check failed (prefill+decode vs one-shot prefill).")

    # 4) Unmasked sanity checks vs reference/recurrent implementations
    K_nomask = torch.randn(B, T, H, D, device=device, dtype=dtype)
    V_nomask = torch.randn(B, T, H, D, device=device, dtype=dtype)
    Yn_py, Sn_py = flare_prefill_pytorch(Q, K_nomask, V_nomask, scale=scale, attention_mask=None)
    Yn_tr, Sn_tr = flare_prefill_triton(Q, K_nomask, V_nomask, scale=scale, attention_mask=None)
    _print_err_report("Cached Prefill (NoMask) Triton vs PyTorch", Yn_tr, Yn_py, atol)
    if not torch.allclose(Yn_tr.float(), Yn_py.float(), atol=atol, rtol=rtol):
        raise AssertionError("Cached prefill (no-mask) mismatch between Triton and PyTorch.")

    Yn_ref = flare_causal_reference(Q, K_nomask, V_nomask, scale=scale)
    _print_err_report("Cached Prefill (NoMask) vs Reference", Yn_py, Yn_ref, ref_atol)
    if not torch.allclose(Yn_py.float(), Yn_ref.float(), atol=ref_atol, rtol=ref_rtol):
        raise AssertionError("Cached prefill (no-mask) mismatch vs flare_causal_reference.")

    Yn_rec = RecurrentFLARE.apply(Q, K_nomask, V_nomask, scale).permute(0, 2, 1, 3).contiguous()
    _print_err_report("Cached Prefill (NoMask) vs RecurrentFLARE", Yn_py, Yn_rec, ref_atol)
    if not torch.allclose(Yn_py.float(), Yn_rec.float(), atol=ref_atol, rtol=ref_rtol):
        raise AssertionError("Cached prefill (no-mask) mismatch vs RecurrentFLARE.")

    Yn_rec_orig = RecurrentFLARE.apply(Q, K_nomask, V_nomask, scale, None, None, None, None, 1).permute(0, 2, 1, 3).contiguous()
    _print_err_report("Cached Prefill (NoMask) vs RecurrentFLAREOrig", Yn_py, Yn_rec_orig, ref_atol)
    if not torch.allclose(Yn_py.float(), Yn_rec_orig.float(), atol=ref_atol, rtol=ref_rtol):
        raise AssertionError("Cached prefill (no-mask) mismatch vs RecurrentFLAREOrig.")

    K_next_nomask = torch.randn(B, T_NEXT, H, D, device=device, dtype=dtype)
    V_next_nomask = torch.randn(B, T_NEXT, H, D, device=device, dtype=dtype)
    Ydn_py, _ = flare_decode_pytorch(Q, K_next_nomask, V_next_nomask, Sn_py, scale=scale, attention_mask=None)
    Ydn_tr, _ = flare_decode_triton(Q, K_next_nomask, V_next_nomask, Sn_tr, scale=scale, attention_mask=None)
    _print_err_report("Cached Decode (NoMask) Triton vs PyTorch", Ydn_tr, Ydn_py, atol)
    if not torch.allclose(Ydn_tr.float(), Ydn_py.float(), atol=atol, rtol=rtol):
        raise AssertionError("Cached decode (no-mask) mismatch between Triton and PyTorch.")

    Yn_full_ref = flare_causal_reference(
        Q,
        torch.cat([K_nomask, K_next_nomask], dim=1),
        torch.cat([V_nomask, V_next_nomask], dim=1),
        scale=scale,
    )
    _print_err_report("Cached Decode (NoMask) vs Reference Last Token", Ydn_py, Yn_full_ref[:, -T_NEXT:], ref_atol)
    if not torch.allclose(Ydn_py.float(), Yn_full_ref[:, -T_NEXT:].float(), atol=ref_atol, rtol=ref_rtol):
        raise AssertionError("Cached decode (no-mask) mismatch vs reference continuation.")

    # 5) FLAREDecoder wiring checks
    hidden_size = H * D
    dec = FLAREDecoder(
        hidden_size=hidden_size,
        num_heads=H,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        layer_idx=0,
        num_latents=M,
        q_norm=False,
        k_norm=False,
        num_layers_k_proj=-1,
        num_layers_v_proj=-1,
    ).to(device=device, dtype=dtype)
    dec.eval()

    x = torch.randn(B, T, hidden_size, device=device, dtype=dtype)
    with torch.no_grad():
        # No-cache full-sequence path currently has no explicit attention_mask support in FLAREDecoder.
        # Keep mask validation in cached prefill/decode checks above.
        o0, _, _ = dec(x, attention_mask=None, past_key_values=None, use_cache=False)
    print(f"[FLARE CACHED TEST] decoder no-cache ok shape={tuple(o0.shape)}")

    cache = Cache()
    with torch.no_grad():
        o1, _, cache = dec(x, attention_mask=mask, past_key_values=cache, use_cache=True)
        x2 = torch.randn(B, T_NEXT, hidden_size, device=device, dtype=dtype)
        o2, _, cache = dec(x2, attention_mask=mask_next, past_key_values=cache, use_cache=True)
    print(
        f"[FLARE CACHED TEST] decoder cached prefill/decode ok shapes={tuple(o1.shape)} {tuple(o2.shape)}"
    )

    x3 = torch.randn(B, T_CONT, hidden_size, device=device, dtype=dtype)
    mask3 = (torch.rand(B, T_CONT, device=device) > mask_prob).to(torch.int32)
    with torch.no_grad():
        o3, _, cache = dec(x3, attention_mask=mask3, past_key_values=cache, use_cache=True)
    print(f"[FLARE CACHED TEST] decoder cached continuation ok shape={tuple(o3.shape)}")

    for name, tensor in (
        ("Y_py", Y_py),
        ("Y_tr", Y_tr),
        ("Yd_py", Yd_py),
        ("Yd_tr", Yd_tr),
        ("o0", o0),
        ("o1", o1),
        ("o2", o2),
        ("o3", o3),
    ):
        _check_finite(f"cached_test.{name}", tensor)

    print("[FLARE CACHED TEST] all checks passed.")


def main(B: int = 1, H: int = 8, M: int = 128, N: int = 2048, D: int = 16, dtype: str = 'bfloat16'):
    from .chunked_old import ChunkedFLAREOld

    if os.environ.get("FLARE_REGRESSION_TEST", "0") == "1":
        _regression_test()
        return
    if os.environ.get("FLARE_CORRECTNESS_SUITE", "0") == "1":
        _run_correctness_suite()
        return
    if os.environ.get("FLARE_DEBUG_BWD") == "1":
        _debug_triton_bwd_compare()
        return
    if os.environ.get("FLARE_DEBUG_FWD") == "1":
        _debug_triton_fwd_compare()
        return
    if os.environ.get("FLARE_DEBUG_STABLE_KERNEL") == "1":
        _debug_triton_stable_kernel()
        return
    if os.environ.get("FLARE_PARITY") == "1":
        _parity_tests()
        return
    if os.environ.get("FLARE_TRAINLIKE") == "1":
        _trainlike_sanity()
        return
    if os.environ.get("FLARE_LONGCTX_ACCURACY", "0") == "1":
        _long_context_accuracy_suite()
        return
    if os.environ.get("FLARE_TRAINLIKE_PARITY", "0") == "1":
        _trainlike_multistep_parity()
        return
    if os.environ.get("FLARE_CHUNK_SENSITIVITY", "0") == "1":
        _chunk_size_sensitivity_suite()
        return
    if os.environ.get("FLARE_SHARP_BWD_REGRESSION", "0") == "1":
        _sharp_softmax_bwd_regression_suite()
        return
    if os.environ.get("FLARE_TRAINLIKE_PROJ") == "1":
        _trainlike_projected()
        return
    if os.environ.get("FLARE_RECURRENT_TEST", "0") == "1":
        _run_recurrent_flare_test()
        return
    if os.environ.get("FLARE_CACHED_TEST", "0") == "1":
        _run_cached_impl_test()
        return
    device = torch.device('cuda')
    dtype = getattr(torch, dtype)
    try:
        from .flash_attention2_triton import (
            flash_attention2_triton_bnhd as _flash_attention2_triton_bnhd,
        )
    except Exception as exc:
        _flash_attention2_triton_bnhd = None
        print(f"[FLARE DEBUG] FlashAttention2 Triton baseline unavailable: {exc}")

    scale = (D ** -0.5)

    torch.manual_seed(0)
    Q = torch.randn(H, M, D, device=device, dtype=dtype)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype)

    chunk_env = os.environ.get("FLARE_PYTORCH_CHUNK_SIZE", "")
    if chunk_env:
        chunk_size = int(chunk_env)
    else:
        chunk_size = None

    compile_timing_enabled = os.environ.get("FLARE_COMPILE_TIMINGS", "1") == "1"

    def _compile_suffix(ms: float) -> str:
        if math.isnan(ms):
            return ""
        return f" (compile={ms:.2f} ms)"

    def _probe_compile(run_fn) -> float:
        if not compile_timing_enabled:
            return float("nan")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1e3

    #======================================================================#
    # Benchmark reference implementation
    #======================================================================#
    print("Measuring reference implementation...", end=" ", flush=True)

    ref_fp32_enabled = os.environ.get("FLARE_REFERENCE_FP32", "1") == "1"
    ref_bf16_enabled = os.environ.get("FLARE_REFERENCE_BF16", "0") == "1"
    if N > 4096:
        warnings.warn(f"Reference implementation skipped for N={N} > 4096 (too slow).")
        Y_reference = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
        Y_reference_bf16 = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
        ref_mem = 0.0
        ref_ms = 0.0
        ref_bf16_mem = 0.0
        ref_bf16_ms = 0.0
        ref_errors = dict()
        ref_bf16_errors = dict()
        print("Skipped (N > 4096)")
    else:
        ref_compile_ms = float("nan")
        ref_bf16_compile_ms = float("nan")
        # Memory measurement
        if ref_fp32_enabled:
            ref_compile_ms = _probe_compile(lambda: flare_causal_reference(Q, K, V, scale=scale))
            Y_reference, ref_mem = measure_memory(flare_causal_reference, Q, K, V, scale=scale)
            ref_ms = triton.testing.do_bench(lambda: flare_causal_reference(Q, K, V, scale=scale), warmup=2, rep=2)
        else:
            Y_reference = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
            ref_mem = 0.0
            ref_ms = 0.0

        # Timing using triton.testing.do_bench
        if ref_bf16_enabled:
            os.environ["FLARE_REFERENCE_FP32"] = "0"
            ref_bf16_compile_ms = _probe_compile(lambda: flare_causal_reference(Q, K, V, scale=scale))
            Y_reference_bf16, ref_bf16_mem = measure_memory(flare_causal_reference, Q, K, V, scale=scale)
            ref_bf16_ms = triton.testing.do_bench(
                lambda: flare_causal_reference(Q, K, V, scale=scale), warmup=2, rep=2
            )
            os.environ["FLARE_REFERENCE_FP32"] = "1" if ref_fp32_enabled else "0"
        else:
            Y_reference_bf16 = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
            ref_bf16_mem = 0.0
            ref_bf16_ms = 0.0

        ref_errors = dict()
        ref_bf16_errors = dict()

        extra = _compile_suffix(ref_compile_ms)
        if ref_bf16_enabled:
            extra += _compile_suffix(ref_bf16_compile_ms).replace("compile=", ", bf16_compile=")
        print(f"Done{extra}")

    #======================================================================#
    # Benchmark PyTorch implementation 2
    #======================================================================#
    print("Measuring PyTorch implementation 2...", end=" ", flush=True)

    try:
        pytorch2_compile_ms = _probe_compile(
            lambda: flare_causal_chunked(Q, K, V, scale=scale, chunk_size=chunk_size)
        )
        Y_pytorch2, pytorch2_mem = measure_memory(
            flare_causal_chunked, Q, K, V, scale=scale, chunk_size=chunk_size
        )
        pytorch2_ms = triton.testing.do_bench(
            lambda: flare_causal_chunked(Q, K, V, scale=scale, chunk_size=chunk_size),
            warmup=2,
            rep=2,
        )
        pytorch2_errors = compute_errors(Y_pytorch2, Y_reference, 'pytorch2') if N <= 4096 and ref_fp32_enabled else {}
    except Exception as exc:
        print(f"[FLARE DEBUG] PyTorch2 forward failed: {exc}")
        Y_pytorch2 = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
        pytorch2_mem = 0.0
        pytorch2_ms = float("nan")
        pytorch2_errors = {}
        pytorch2_compile_ms = float("nan")

    print(f"Done{_compile_suffix(pytorch2_compile_ms)}")

    #======================================================================#
    # Benchmark PCVR-AR-like impl implementation
    #======================================================================#
    print("Measuring PCVR-AR-like impl implementation...", end=" ", flush=True)
    try:
        perciever_imlp_compile_ms = _probe_compile(
            lambda: flare_causal_perciever_ar(Q, K, V, scale=scale)
        )
        Y_perciever_imlp, perciever_imlp_mem = measure_memory(
            flare_causal_perciever_ar, Q, K, V, scale=scale
        )
        perciever_imlp_ms = triton.testing.do_bench(
            lambda: flare_causal_perciever_ar(Q, K, V, scale=scale),
            warmup=2,
            rep=2,
        )
        perciever_imlp_errors = (
            compute_errors(Y_perciever_imlp, Y_reference, "perciever_imlp")
            if N <= 4096 and ref_fp32_enabled
            else {}
        )
    except Exception as exc:
        print(f"[FLARE DEBUG] PCVR-AR-like impl forward failed: {exc}")
        Y_perciever_imlp = torch.full((B, N, H, D), float('nan'), device=device, dtype=dtype)
        perciever_imlp_mem = 0.0
        perciever_imlp_ms = float("nan")
        perciever_imlp_errors = {}
        perciever_imlp_compile_ms = float("nan")
    print(f"Done{_compile_suffix(perciever_imlp_compile_ms)}")

    #======================================================================#
    # Benchmark PyTorch dense / DenseFLARE (small-N only)
    #======================================================================#
    run_dense = N <= 128
    if run_dense:
        print("Measuring PyTorch dense implementation...", end=" ", flush=True)
        pytorch_dense_compile_ms = _probe_compile(lambda: flare_causal_pytorch_dense(Q, K, V, scale=scale))

        # Memory measurement
        Y_pytorch_dense, pytorch_dense_mem = measure_memory(
            flare_causal_pytorch_dense, Q, K, V, scale=scale
        )

        # Timing using triton.testing.do_bench
        pytorch_dense_ms = triton.testing.do_bench(
            lambda: flare_causal_pytorch_dense(Q, K, V, scale=scale), warmup=2, rep=2
        )

        # Compute errors for PyTorch dense (against reference if available)
        pytorch_dense_errors = compute_errors(Y_pytorch_dense, Y_reference, 'pytorch_dense') if N <= 4096 and ref_fp32_enabled else {}

        print(f"Done{_compile_suffix(pytorch_dense_compile_ms)}")

        #======================================================================#
        # Benchmark DenseFLARE implementation
        #======================================================================#
        print("Measuring DenseFLARE implementation...", end=" ", flush=True)
        dense_triton_compile_ms = _probe_compile(lambda: DenseFLARE.apply(Q, K, V, scale))

        Y_dense_triton, dense_triton_mem = measure_memory(DenseFLARE.apply, Q, K, V, scale)
        dense_triton_ms = triton.testing.do_bench(lambda: DenseFLARE.apply(Q, K, V, scale), warmup=2, rep=2)

        dense_triton_errors = compute_errors(Y_dense_triton, Y_reference, 'dense_triton') if N <= 4096 and ref_fp32_enabled else {}

        print(f"Done{_compile_suffix(dense_triton_compile_ms)}")

        #======================================================================#
        # Benchmark DenseFLARE1 implementation
        #======================================================================#
        print("Measuring DenseFLARE1 implementation...", end=" ", flush=True)
        dense1_triton_compile_ms = _probe_compile(lambda: DenseFLARE1.apply(Q, K, V, scale))

        Y_dense1_triton, dense1_triton_mem = measure_memory(DenseFLARE1.apply, Q, K, V, scale)
        dense1_triton_ms = triton.testing.do_bench(lambda: DenseFLARE1.apply(Q, K, V, scale), warmup=2, rep=2)

        dense1_triton_errors = compute_errors(Y_dense1_triton, Y_reference, 'dense1_triton') if N <= 4096 and ref_fp32_enabled else {}

        print(f"Done{_compile_suffix(dense1_triton_compile_ms)}")

        dense1_phase1_ms, dense1_phase2_ms, dense1_phase3_ms = _denseflare1_phase_bench(Q, K, V, scale)
    else:
        Y_pytorch_dense = torch.full((B, N, H, D), float("nan"), device=device, dtype=dtype)
        Y_dense_triton = torch.full((B, N, H, D), float("nan"), device=device, dtype=dtype)
        pytorch_dense_mem = 0.0
        pytorch_dense_ms = 0.0
        dense_triton_mem = 0.0
        dense_triton_ms = 0.0
        pytorch_dense_errors = {}
        dense_triton_errors = {}
        Y_dense1_triton = torch.full((B, N, H, D), float("nan"), device=device, dtype=dtype)
        dense1_triton_mem = 0.0
        dense1_triton_ms = 0.0
        dense1_triton_errors = {}
        dense1_phase1_ms = 0.0
        dense1_phase2_ms = 0.0
        dense1_phase3_ms = 0.0
        print("Skipped PyTorch dense / DenseFLARE (N > 128)")

    if N <= 4096 and ref_fp32_enabled:
        # Reference vs reference (sanity)
        ref_errors = compute_errors(Y_reference, Y_reference, 'reference')
    if N <= 4096 and ref_bf16_enabled:
        ref_bf16_errors = compute_errors(Y_reference_bf16, Y_reference, 'reference_bf16')

    if os.environ.get("FLARE_STRICT_FP32_CHECK", "") == "1" and N <= 4096:
        with torch.no_grad():
            Q_fp32 = Q.float()
            K_fp32 = K.float()
            V_fp32 = V.float()
            Y_ref32 = flare_causal_reference(Q_fp32, K_fp32, V_fp32, scale=scale)
            Y_p232 = flare_causal_chunked(Q_fp32, K_fp32, V_fp32, scale=scale, chunk_size=chunk_size)
            err32 = compute_errors(Y_p232, Y_ref32, "pytorch2_fp32")
            print(
                f"[FLARE DEBUG] PyTorch2 vs Reference (FP32 inputs) "
                f"abs={err32['pytorch2_fp32_mean_abs_err']:.2e}/{err32['pytorch2_fp32_max_abs_err']:.2e} "
                f"rel={err32['pytorch2_fp32_mean_rel_err']:.2e}/{err32['pytorch2_fp32_max_rel_err']:.2e}"
            )

    if os.environ.get("FLARE_DEBUG_P2_REF", "") == "1" and N <= 4096 and ref_fp32_enabled and not torch.isnan(Y_reference).any():
        with torch.no_grad():
            err = (Y_pytorch2 - Y_reference).abs()  # [B,N,H,D]
            per_token_max = err.amax(dim=(0, 2, 3))  # [N]
            per_token_mean = err.mean(dim=(0, 2, 3))  # [N]
            topk = min(5, N)
            vals, idx = torch.topk(per_token_max, k=topk)
            print("[FLARE DEBUG] PyTorch2 vs Reference per-token max abs err:")
            for i in range(topk):
                t = int(idx[i].item())
                print(f"  t={t} max_abs={vals[i].item():.3e} mean_abs={per_token_mean[t].item():.3e}")
    if os.environ.get("FLARE_DEBUG_P2_FORMULA", "") == "1" and N <= 4096 and ref_fp32_enabled and not torch.isnan(Y_reference).any():
        with torch.no_grad():
            Qf = Q.float()
            Kf = K.float()
            Vf = V.float()
            max_tokens_env = os.environ.get("FLARE_DEBUG_P2_FORMULA_TOKENS", "")
            max_tokens = int(max_tokens_env) if max_tokens_env else min(8, N)
            use_stable = os.environ.get("FLARE_DEBUG_P2_FORMULA_STABLE", "1") == "1"
            eps_dbg = _get_eps_for_dtype(Q.dtype)
            print("[FLARE DEBUG] Formula vs Reference/PyTorch2 (per-token):")
            for t in range(max_tokens):
                Kt = Kf[:, : t + 1, :, :]  # [B,t+1,H,D]
                Vt = Vf[:, : t + 1, :, :]
                S_all = scale * torch.einsum("bthd,hmd->bthm", Kt, Qf)  # [B,t+1,H,M]
                S_t = S_all[:, -1, :, :]  # [B,H,M]
                if use_stable:
                    m_u = S_all.max(dim=1).values  # [B,H,M]
                    exp_u = torch.exp(S_all - m_u.unsqueeze(1))  # [B,t+1,H,M]
                    den_total = exp_u.sum(dim=1) * torch.exp(m_u)  # [B,H,M]
                    sum_exp_v = torch.einsum("bthm,bthd->bhmd", exp_u, Vt) * torch.exp(m_u).unsqueeze(-1)
                else:
                    exp_u = torch.exp(S_all)
                    den_total = exp_u.sum(dim=1)
                    sum_exp_v = torch.einsum("bthm,bthd->bhmd", exp_u, Vt)
                P_t = torch.softmax(S_t, dim=-1)
                expA_t = P_t / (den_total + eps_dbg)
                y_formula = torch.einsum("bhm,bhmd->bhd", expA_t, sum_exp_v)
                y_ref = Y_reference[:, t, :, :]
                y_p2 = Y_pytorch2[:, t, :, :]
                err_ref = (y_formula - y_ref).abs().amax().item()
                err_p2 = (y_formula - y_p2).abs().amax().item()
                print(f"  t={t} max_abs_formula_vs_ref={err_ref:.3e} max_abs_formula_vs_p2={err_p2:.3e}")

    #======================================================================#
    # Benchmark ChunkedFLARE implementation (input_precision modes)
    #======================================================================#
    def _normalize_chunked_forward_timings(raw_timings):
        if not isinstance(raw_timings, dict):
            return {}

        # New profiler format: {"forward": {...}, "backward": {...}, "..._total_ms": ...}
        forward_timings = raw_timings.get("forward")
        if isinstance(forward_timings, dict):
            normalized = {
                k: float(v)
                for k, v in forward_timings.items()
                if isinstance(v, (int, float))
            }
            forward_total = raw_timings.get("forward_total_ms")
            if isinstance(forward_total, (int, float)):
                normalized["Total"] = float(forward_total)
            elif normalized:
                normalized["Total"] = sum(normalized.values())
            return normalized

        # Legacy profiler format: flat phase-name -> ms mapping.
        normalized = {
            k: float(v)
            for k, v in raw_timings.items()
            if k != "Total" and isinstance(v, (int, float))
        }
        total = raw_timings.get("Total")
        if isinstance(total, (int, float)):
            normalized["Total"] = float(total)
        elif normalized:
            normalized["Total"] = sum(normalized.values())
        return normalized

    print("Measuring ChunkedFLARE implementation...", end=" ", flush=True)
    causalflare_variants = [
        ("ChunkedFLARE (ieee)", "ieee", "triton3_ieee"),
        ("ChunkedFLARE (tf32)", "tf32", "triton3_tf32"),
        ("ChunkedFLARE (tf32x3)", "tf32x3", "triton3_tf32x3"),
    ]
    causalflare_results = {}
    triton3_avg_timings_by_variant = {}
    for row_name, precision_mode, err_prefix in causalflare_variants:
        with _temp_env_var("FLARE_INPUT_PRECISION", precision_mode):
            compile_ms = _probe_compile(lambda: flare_chunk_triton(Q, K, V, scale))
            Y_t3, t3_mem = measure_memory(flare_chunk_triton, Q, K, V, scale)
            t3_ms = triton.testing.do_bench(lambda: flare_chunk_triton(Q, K, V, scale))
            t3_errors = compute_errors(Y_t3, Y_reference, err_prefix) if N <= 4096 and ref_fp32_enabled else {}
            causalflare_results[row_name] = {
                "Y": Y_t3,
                "mem": t3_mem,
                "ms": t3_ms,
                "errors": t3_errors,
                "compile_ms": compile_ms,
            }
            timings_list = []
            with torch.no_grad():
                for _ in range(25):
                    flare_chunk_triton(Q, K, V, scale)
                for _ in range(100):
                    _, timings = flare_chunk_triton(Q, K, V, scale, None, None, True)
                    normalized_timings = _normalize_chunked_forward_timings(timings)
                    if normalized_timings:
                        timings_list.append(normalized_timings)
            if timings_list:
                variant_timings = {}
                phase_names = sorted({k for t in timings_list for k in t.keys() if k != "Total"})
                for name in phase_names:
                    variant_timings[name] = sum(t.get(name, 0.0) for t in timings_list) / len(timings_list)
                variant_timings["Total"] = sum(variant_timings.values())
                if variant_timings["Total"] > 0 and t3_ms > 0:
                    scale_factor = t3_ms / variant_timings["Total"]
                    for name in phase_names:
                        variant_timings[name] *= scale_factor
                    variant_timings["Total"] = t3_ms
                triton3_avg_timings_by_variant[row_name] = variant_timings
    # Preserve old aliases for downstream logic/profiling text.
    triton3_ms = causalflare_results["ChunkedFLARE (ieee)"]["ms"]
    triton3_mem = causalflare_results["ChunkedFLARE (ieee)"]["mem"]
    triton3_errors = causalflare_results["ChunkedFLARE (ieee)"]["errors"]
    print(
        "Done "
        + ", ".join(
            f"{name.split()[-1].strip('()')}_compile={vals['compile_ms']:.2f} ms"
            for name, vals in causalflare_results.items()
        )
    )

    print("Measuring ChunkedFLAREOld implementation...", end=" ", flush=True)
    old_compile_ms = _probe_compile(lambda: ChunkedFLAREOld.apply(Q, K, V, scale))
    Y_old, old_mem = measure_memory(ChunkedFLAREOld.apply, Q, K, V, scale)
    old_ms = triton.testing.do_bench(lambda: ChunkedFLAREOld.apply(Q, K, V, scale))
    old_errors = compute_errors(Y_old, Y_reference, "triton3_old") if N <= 4096 and ref_fp32_enabled else {}
    print(f"Done{_compile_suffix(old_compile_ms)}")

    #======================================================================#
    # Benchmark Causal SDPA implementation
    #======================================================================#
    print("Measuring Causal SDPA implementation...", end=" ", flush=True)

    Q_ = torch.rand_like(K)
    K_ = torch.rand_like(K)
    V_ = torch.rand_like(K)
    # Triton tutorial FA2 kernel in this environment is fp16-only.
    Q_fa2_in = Q_.to(torch.float16).contiguous()
    K_fa2_in = K_.to(torch.float16).contiguous()
    V_fa2_in = V_.to(torch.float16).contiguous()

    sdpa_compile_ms = _probe_compile(lambda: causal_SDPA(Q_, K_, V_))

    # Memory measurement
    _, SDPA_mem = measure_memory(causal_SDPA, Q_, K_, V_)

    # Timing using triton.testing.do_bench
    SDPA_ms = triton.testing.do_bench(lambda: causal_SDPA(Q_, K_, V_))

    print(f"Done{_compile_suffix(sdpa_compile_ms)}")

    #======================================================================#
    # Benchmark Triton FlashAttention2 tutorial implementation
    #======================================================================#
    print("Measuring FlashAttention2 (Triton tutorial) implementation...", end=" ", flush=True)
    if _flash_attention2_triton_bnhd is None:
        fa2_triton_compile_ms = float("nan")
        fa2_triton_mem = 0.0
        fa2_triton_ms = float("nan")
        print("Skipped (import unavailable)")
    else:
        try:
            fa2_triton_compile_ms = _probe_compile(
                lambda: _flash_attention2_triton_bnhd(Q_fa2_in, K_fa2_in, V_fa2_in, causal=True, sm_scale=scale)
            )
            _, fa2_triton_mem = measure_memory(
                _flash_attention2_triton_bnhd, Q_fa2_in, K_fa2_in, V_fa2_in, causal=True, sm_scale=scale
            )
            fa2_triton_ms = triton.testing.do_bench(
                lambda: _flash_attention2_triton_bnhd(Q_fa2_in, K_fa2_in, V_fa2_in, causal=True, sm_scale=scale)
            )
            print(f"Done{_compile_suffix(fa2_triton_compile_ms)}")
        except Exception as exc:
            print(f"[FLARE DEBUG] FlashAttention2 Triton forward failed: {exc}")
            fa2_triton_compile_ms = float("nan")
            fa2_triton_mem = 0.0
            fa2_triton_ms = float("nan")
            print("Failed")

    #======================================================================#
    # Benchmark Recurrent FLARE implementation (small-N only)
    #======================================================================#
    run_recurrent_triton = N <= 2048
    run_recurrent_pytorch = N <= 128
    if run_recurrent_triton:
        print("Measuring RecurrentFLARE implementation...", end=" ", flush=True)
        rec_orig_compile_ms = _probe_compile(lambda: RecurrentFLARE.apply(Q, K, V, scale, None, None, None, None, 1))
        rec_tr_compile_ms = _probe_compile(lambda: RecurrentFLARE.apply(Q, K, V, scale))

        rec_warmup = int(os.environ.get("FLARE_RECURRENT_WARMUP", "2"))
        rec_rep = int(os.environ.get("FLARE_RECURRENT_REP", "2"))

        rec_ref_available = N <= 4096 and ref_fp32_enabled and not torch.isnan(Y_reference).any()
        if rec_ref_available:
            Y_rec_ref = Y_reference.permute(0, 2, 1, 3).contiguous()
        else:
            Y_rec_ref = torch.full((B, H, N, D), float('nan'), device=device, dtype=dtype)

        if run_recurrent_pytorch:
            rec_pt_compile_ms = _probe_compile(lambda: flare_recurrent_pytorch(Q, K, V, scale=scale))
            Y_rec_pt, rec_pt_mem = measure_memory(flare_recurrent_pytorch, Q, K, V, scale=scale)
            rec_pt_ms = triton.testing.do_bench(
                lambda: flare_recurrent_pytorch(Q, K, V, scale=scale), warmup=rec_warmup, rep=rec_rep
            )
        else:
            Y_rec_pt = torch.full((B, H, N, D), float("nan"), device=device, dtype=dtype)
            rec_pt_mem = 0.0
            rec_pt_ms = float("nan")
            rec_pt_compile_ms = float("nan")

        Y_rec_tr, rec_tr_mem = measure_memory(RecurrentFLARE.apply, Q, K, V, scale)
        rec_tr_ms = triton.testing.do_bench(
            lambda: RecurrentFLARE.apply(Q, K, V, scale), warmup=rec_warmup, rep=rec_rep
        )

        Y_rec_orig, rec_orig_mem = measure_memory(RecurrentFLARE.apply, Q, K, V, scale, None, None, None, None, 1)
        rec_orig_ms = triton.testing.do_bench(
            lambda: RecurrentFLARE.apply(Q, K, V, scale, None, None, None, None, 1), warmup=rec_warmup, rep=rec_rep
        )

        rec_pt_errors = compute_errors(Y_rec_pt, Y_rec_ref, "rec_pt") if (rec_ref_available and run_recurrent_pytorch) else {}
        rec_tr_errors = compute_errors(Y_rec_tr, Y_rec_ref, "rec_tr") if rec_ref_available else {}
        rec_orig_errors = compute_errors(Y_rec_orig, Y_rec_ref, "rec_orig") if rec_ref_available else {}

        rec_extra = (
            f", pt_compile={rec_pt_compile_ms:.2f} ms"
            if run_recurrent_pytorch and not math.isnan(rec_pt_compile_ms)
            else ""
        )
        print(
            f"Done (Recurrent Triton path: permute, orig_compile={rec_orig_compile_ms:.2f} ms, "
            f"triton_compile={rec_tr_compile_ms:.2f} ms{rec_extra})"
        )
    else:
        Y_rec_pt = torch.full((B, H, N, D), float("nan"), device=device, dtype=dtype)
        Y_rec_tr = torch.full((B, H, N, D), float("nan"), device=device, dtype=dtype)
        Y_rec_orig = torch.full((B, H, N, D), float("nan"), device=device, dtype=dtype)
        rec_pt_mem = 0.0
        rec_pt_ms = float("nan")
        rec_tr_mem = 0.0
        rec_tr_ms = 0.0
        rec_orig_mem = 0.0
        rec_orig_ms = 0.0
        rec_pt_errors = {}
        rec_tr_errors = {}
        rec_orig_errors = {}
        print("Skipped RecurrentFLARE (N > 2048)")

    #======================================================================#
    # Print results table
    #======================================================================#
    # Calculate speedups against FA2 Triton when available.
    speedup_base_name = "FA2 Triton" if fa2_triton_ms > 0 and not math.isnan(fa2_triton_ms) else "Causal SDPA"
    speedup_base_ms = fa2_triton_ms if speedup_base_name == "FA2 Triton" else SDPA_ms
    ref_speedup       = speedup_base_ms / ref_ms       if ref_ms > 0 else 'N/A'
    pytorch2_speedup  = speedup_base_ms / pytorch2_ms  if pytorch2_ms > 0 else 0.0
    perciever_imlp_speedup = speedup_base_ms / perciever_imlp_ms if perciever_imlp_ms > 0 else 0.0
    pytorch_dense_speedup = speedup_base_ms / pytorch_dense_ms if pytorch_dense_ms > 0 else 0.0
    dense_triton_speedup = speedup_base_ms / dense_triton_ms if dense_triton_ms > 0 else 0.0
    dense1_triton_speedup = speedup_base_ms / dense1_triton_ms if dense1_triton_ms > 0 else 0.0
    fa2_triton_speedup = speedup_base_ms / fa2_triton_ms if fa2_triton_ms > 0 else 0.0
    triton3_speedup   = speedup_base_ms / triton3_ms   if triton3_ms > 0 else 0.0
    old_speedup = speedup_base_ms / old_ms if old_ms > 0 else 0.0
    causalflare_speedups = {
        name: (speedup_base_ms / vals["ms"] if vals["ms"] > 0 else 0.0)
        for name, vals in causalflare_results.items()
    }
    SDPA_speedup      = speedup_base_ms / SDPA_ms if SDPA_ms > 0 else 0.0

    has_nan_ref = torch.isnan(Y_reference).any()

    print("\n" + "="*100)
    print(f"Testing Causal/Recurrent FLARE Forward Pass (B={B}, H={H}, M={M}, N={N}, D={D}, dtype={dtype})")
    print(f"Speedup baseline: {speedup_base_name}")
    print("="*100)
    print(f"{'Implementation':<20} {'Time (ms)':<10} {'Speedup':<10} {'Memory (GB)':<15} "
          f"{'Abs Err (mean/max)':<20} {'Rel Err (mean/max)':<25}")
    print("-"*100)

    # Reference (FP32) row
    ref_abs_err_str = (f"{ref_errors['reference_mean_abs_err']:.2e}/{ref_errors['reference_max_abs_err']:.2e}") if ref_errors and 'reference_mean_abs_err' in ref_errors and 'reference_max_abs_err' in ref_errors else "N/A"
    ref_rel_err_str = (f"{ref_errors['reference_mean_rel_err']:.2e}/{ref_errors['reference_max_rel_err']:.2e}") if ref_errors and 'reference_mean_rel_err' in ref_errors and 'reference_max_rel_err' in ref_errors else "N/A"
    ref_speedup_str = (f"{ref_speedup:.2f}x") if isinstance(ref_speedup, (int, float)) and not math.isnan(ref_speedup) and ref_speedup != float('inf') else "N/A"
    print(f"{'Reference (FP32)':<20} {ref_ms:<10.2f} {ref_speedup_str:<10} {ref_mem:<15.2e} "
          f"{ref_abs_err_str:<20} {ref_rel_err_str:<20}")

    # Optional Reference (BF16) row (compared against FP32 reference)
    if ref_bf16_enabled:
        print("Note: Reference (BF16) errors are computed vs Reference (FP32).")
        ref_bf16_abs_err_str = (f"{ref_bf16_errors['reference_bf16_mean_abs_err']:.2e}/{ref_bf16_errors['reference_bf16_max_abs_err']:.2e}") if ref_bf16_errors and 'reference_bf16_mean_abs_err' in ref_bf16_errors and 'reference_bf16_max_abs_err' in ref_bf16_errors else "N/A"
        ref_bf16_rel_err_str = (f"{ref_bf16_errors['reference_bf16_mean_rel_err']:.2e}/{ref_bf16_errors['reference_bf16_max_rel_err']:.2e}") if ref_bf16_errors and 'reference_bf16_mean_rel_err' in ref_bf16_errors and 'reference_bf16_max_rel_err' in ref_bf16_errors else "N/A"
        ref_bf16_speedup = speedup_base_ms / ref_bf16_ms if ref_bf16_ms > 0 else float("nan")
        ref_bf16_speedup_str = (f"{ref_bf16_speedup:.2f}x") if isinstance(ref_bf16_speedup, (int, float)) and not math.isnan(ref_bf16_speedup) and ref_bf16_speedup != float('inf') else "N/A"
        print(f"{'Reference (BF16)':<20} {ref_bf16_ms:<10.2f} {ref_bf16_speedup_str:<10} {ref_bf16_mem:<15.2e} "
              f"{ref_bf16_abs_err_str:<20} {ref_bf16_rel_err_str:<20}")

    # PyTorch 2 row
    if pytorch2_errors:
        pytorch2_abs_err_str = f"{pytorch2_errors['pytorch2_mean_abs_err']:.2e}/{pytorch2_errors['pytorch2_max_abs_err']:.2e}"
        pytorch2_rel_err_str = f"{pytorch2_errors['pytorch2_mean_rel_err']:.2e}/{pytorch2_errors['pytorch2_max_rel_err']:.2e}"
    else:
        pytorch2_abs_err_str = "N/A"
        pytorch2_rel_err_str = "N/A"
    pytorch2_speedup_str = f"{pytorch2_speedup:.2f}x" if pytorch2_speedup != float('inf') and not math.isnan(pytorch2_speedup) else "N/A"
    print(f"{'PyTorch 2':<20} {pytorch2_ms:<10.2f} {pytorch2_speedup_str:<10} {pytorch2_mem:<15.2e} "
          f"{pytorch2_abs_err_str:<20} {pytorch2_rel_err_str:<20}")

    # PCVR-AR-like impl row
    if perciever_imlp_errors:
        perciever_imlp_abs_err_str = (
            f"{perciever_imlp_errors['perciever_imlp_mean_abs_err']:.2e}/"
            f"{perciever_imlp_errors['perciever_imlp_max_abs_err']:.2e}"
        )
        perciever_imlp_rel_err_str = (
            f"{perciever_imlp_errors['perciever_imlp_mean_rel_err']:.2e}/"
            f"{perciever_imlp_errors['perciever_imlp_max_rel_err']:.2e}"
        )
    else:
        perciever_imlp_abs_err_str = "N/A"
        perciever_imlp_rel_err_str = "N/A"
    perciever_imlp_speedup_str = (
        f"{perciever_imlp_speedup:.2f}x"
        if perciever_imlp_speedup != float('inf') and not math.isnan(perciever_imlp_speedup)
        else "N/A"
    )
    print(
        f"{'PCVR-AR-like impl':<20} {perciever_imlp_ms:<10.2f} {perciever_imlp_speedup_str:<10} "
        f"{perciever_imlp_mem:<15.2e} {perciever_imlp_abs_err_str:<20} {perciever_imlp_rel_err_str:<20}"
    )

    # PyTorch dense row
    if run_dense:
        if pytorch_dense_errors:
            pytorch_dense_abs_err_str = f"{pytorch_dense_errors['pytorch_dense_mean_abs_err']:.2e}/{pytorch_dense_errors['pytorch_dense_max_abs_err']:.2e}"
            pytorch_dense_rel_err_str = f"{pytorch_dense_errors['pytorch_dense_mean_rel_err']:.2e}/{pytorch_dense_errors['pytorch_dense_max_rel_err']:.2e}"
        else:
            pytorch_dense_abs_err_str = "N/A"
            pytorch_dense_rel_err_str = "N/A"
        pytorch_dense_speedup_str = f"{pytorch_dense_speedup:.2f}x" if pytorch_dense_speedup != float('inf') and not math.isnan(pytorch_dense_speedup) else "N/A"
        print(f"{'PyTorch Dense':<20} {pytorch_dense_ms:<10.2f} {pytorch_dense_speedup_str:<10} {pytorch_dense_mem:<15.2e} "
              f"{pytorch_dense_abs_err_str:<20} {pytorch_dense_rel_err_str:<20}")

        # DenseFLARE row
        if dense_triton_errors:
            dense_triton_abs_err_str = f"{dense_triton_errors['dense_triton_mean_abs_err']:.2e}/{dense_triton_errors['dense_triton_max_abs_err']:.2e}"
            dense_triton_rel_err_str = f"{dense_triton_errors['dense_triton_mean_rel_err']:.2e}/{dense_triton_errors['dense_triton_max_rel_err']:.2e}"
        else:
            dense_triton_abs_err_str = "N/A"
            dense_triton_rel_err_str = "N/A"
        dense_triton_speedup_str = f"{dense_triton_speedup:.2f}x" if dense_triton_speedup != float('inf') and not math.isnan(dense_triton_speedup) else "N/A"
        print(f"{'DenseFLARE':<20} {dense_triton_ms:<10.2f} {dense_triton_speedup_str:<10} {dense_triton_mem:<15.2e} "
              f"{dense_triton_abs_err_str:<20} {dense_triton_rel_err_str:<20}")

        # DenseFLARE1 row
        if dense1_triton_errors:
            dense1_triton_abs_err_str = f"{dense1_triton_errors['dense1_triton_mean_abs_err']:.2e}/{dense1_triton_errors['dense1_triton_max_abs_err']:.2e}"
            dense1_triton_rel_err_str = f"{dense1_triton_errors['dense1_triton_mean_rel_err']:.2e}/{dense1_triton_errors['dense1_triton_max_rel_err']:.2e}"
        else:
            dense1_triton_abs_err_str = "N/A"
            dense1_triton_rel_err_str = "N/A"
        dense1_triton_speedup_str = f"{dense1_triton_speedup:.2f}x" if dense1_triton_speedup != float('inf') and not math.isnan(dense1_triton_speedup) else "N/A"
        print(f"{'DenseFLARE1':<20} {dense1_triton_ms:<10.2f} {dense1_triton_speedup_str:<10} {dense1_triton_mem:<15.2e} "
              f"{dense1_triton_abs_err_str:<20} {dense1_triton_rel_err_str:<20}")

    # ChunkedFLARE rows (input_precision variants)
    for row_name, _, err_prefix in causalflare_variants:
        row = causalflare_results[row_name]
        row_errors = row["errors"]
        if row_errors:
            abs_err_str = f"{row_errors[f'{err_prefix}_mean_abs_err']:.2e}/{row_errors[f'{err_prefix}_max_abs_err']:.2e}"
            rel_err_str = f"{row_errors[f'{err_prefix}_mean_rel_err']:.2e}/{row_errors[f'{err_prefix}_max_rel_err']:.2e}"
        else:
            abs_err_str = "N/A"
            rel_err_str = "N/A"
        speedup_val = causalflare_speedups[row_name]
        speedup_str = f"{speedup_val:.2f}x" if speedup_val != float('inf') and not math.isnan(speedup_val) else "N/A"
        print(f"{row_name:<20} {row['ms']:<10.2f} {speedup_str:<10} {row['mem']:<15.2e} "
              f"{abs_err_str:<20} {rel_err_str:<20}")

    if old_errors:
        old_abs_err_str = f"{old_errors['triton3_old_mean_abs_err']:.2e}/{old_errors['triton3_old_max_abs_err']:.2e}"
        old_rel_err_str = f"{old_errors['triton3_old_mean_rel_err']:.2e}/{old_errors['triton3_old_max_rel_err']:.2e}"
    else:
        old_abs_err_str = "N/A"
        old_rel_err_str = "N/A"
    old_speedup_str = f"{old_speedup:.2f}x" if old_speedup != float('inf') and not math.isnan(old_speedup) else "N/A"
    print(f"{'ChunkedFLAREOld':<20} {old_ms:<10.2f} {old_speedup_str:<10} {old_mem:<15.2e} "
          f"{old_abs_err_str:<20} {old_rel_err_str:<20}")

    # Causal SDPA row
    SDPA_speedup_str = f"{SDPA_speedup:.2f}x" # if SDPA_speedup != float('inf') and not math.isnan(SDPA_speedup) and not has_nan_ref else "N/A"
    print(f"{'Causal SDPA':<20} {SDPA_ms:<10.2f} {SDPA_speedup_str:<10} {SDPA_mem:<15.2e} "
          f"{'N/A':<20} {'N/A':<20}")

    # FlashAttention2 Triton tutorial row
    fa2_triton_speedup_str = (
        f"{fa2_triton_speedup:.2f}x"
        if fa2_triton_speedup != float("inf") and not math.isnan(fa2_triton_speedup)
        else "N/A"
    )
    fa2_triton_ms_str = f"{fa2_triton_ms:.2f}" if not math.isnan(fa2_triton_ms) else "N/A"
    print(f"{'FA2 Triton':<20} {fa2_triton_ms_str:<10} {fa2_triton_speedup_str:<10} {fa2_triton_mem:<15.2e} "
          f"{'N/A':<20} {'N/A':<20}")
    if run_recurrent_triton:
        rec_pt_speedup = speedup_base_ms / rec_pt_ms if rec_pt_ms > 0 else float("nan")
        rec_tr_speedup = speedup_base_ms / rec_tr_ms if rec_tr_ms > 0 else float("nan")
        rec_orig_speedup = speedup_base_ms / rec_orig_ms if rec_orig_ms > 0 else float("nan")

        rec_pt_speedup_str = f"{rec_pt_speedup:.2f}x" if not math.isnan(rec_pt_speedup) else "N/A"
        rec_tr_speedup_str = f"{rec_tr_speedup:.2f}x" if not math.isnan(rec_tr_speedup) else "N/A"
        rec_orig_speedup_str = f"{rec_orig_speedup:.2f}x" if not math.isnan(rec_orig_speedup) else "N/A"

        if rec_pt_errors:
            rec_pt_abs_err_str = f"{rec_pt_errors['rec_pt_mean_abs_err']:.2e}/{rec_pt_errors['rec_pt_max_abs_err']:.2e}"
            rec_pt_rel_err_str = f"{rec_pt_errors['rec_pt_mean_rel_err']:.2e}/{rec_pt_errors['rec_pt_max_rel_err']:.2e}"
        else:
            rec_pt_abs_err_str = "N/A"
            rec_pt_rel_err_str = "N/A"

        if rec_tr_errors:
            rec_tr_abs_err_str = f"{rec_tr_errors['rec_tr_mean_abs_err']:.2e}/{rec_tr_errors['rec_tr_max_abs_err']:.2e}"
            rec_tr_rel_err_str = f"{rec_tr_errors['rec_tr_mean_rel_err']:.2e}/{rec_tr_errors['rec_tr_max_rel_err']:.2e}"
        else:
            rec_tr_abs_err_str = "N/A"
            rec_tr_rel_err_str = "N/A"
        if rec_orig_errors:
            rec_orig_abs_err_str = f"{rec_orig_errors['rec_orig_mean_abs_err']:.2e}/{rec_orig_errors['rec_orig_max_abs_err']:.2e}"
            rec_orig_rel_err_str = f"{rec_orig_errors['rec_orig_mean_rel_err']:.2e}/{rec_orig_errors['rec_orig_max_rel_err']:.2e}"
        else:
            rec_orig_abs_err_str = "N/A"
            rec_orig_rel_err_str = "N/A"
        if run_recurrent_pytorch:
            print(f"{'Recurrent PyTorch':<20} {rec_pt_ms:<10.2f} {rec_pt_speedup_str:<10} {rec_pt_mem:<15.2e} "
                  f"{rec_pt_abs_err_str:<20} {rec_pt_rel_err_str:<20}")
        print(f"{'Recurrent Orig':<20} {rec_orig_ms:<10.2f} {rec_orig_speedup_str:<10} {rec_orig_mem:<15.2e} "
              f"{rec_orig_abs_err_str:<20} {rec_orig_rel_err_str:<20}")
        print(f"{'Recurrent Triton':<20} {rec_tr_ms:<10.2f} {rec_tr_speedup_str:<10} {rec_tr_mem:<15.2e} "
              f"{rec_tr_abs_err_str:<20} {rec_tr_rel_err_str:<20}")

    if run_dense:
        print("="*100)
        print("DenseFLARE1 Phase Profiling")
        print("="*100)
        total_dense1 = dense1_phase1_ms + dense1_phase2_ms + dense1_phase3_ms
        if total_dense1 > 0:
            p1 = 100.0 * dense1_phase1_ms / total_dense1
            p2 = 100.0 * dense1_phase2_ms / total_dense1
            p3 = 100.0 * dense1_phase3_ms / total_dense1
        else:
            p1 = p2 = p3 = 0.0
        print(f"{'Phase 1 (S,P)':<28} {dense1_phase1_ms:>7.3f}ms / {p1:>5.1f}%")
        print(f"{'Phase 2 (R,L)':<28} {dense1_phase2_ms:>7.3f}ms / {p2:>5.1f}%")
        print(f"{'Phase 3 (W,Y)':<28} {dense1_phase3_ms:>7.3f}ms / {p3:>5.1f}%")

    #======================================================================#
    # Print ChunkedFLARE Forward Phase Profiling
    #======================================================================#
    if triton3_avg_timings_by_variant:
        print("="*100)
        print("ChunkedFLARE Forward Phase Profiling Comparison")
        print("="*100)

        # Collect all unique phase names across all variants
        all_phases = set()
        for timings in triton3_avg_timings_by_variant.values():
            all_phases.update([k for k in timings.keys() if k != "Total"])

        # Sort phases by phase number (Phase 0, Phase 1, etc.)
        phase_order = sorted(all_phases, key=lambda x: (
            int(x.split()[1]) if len(x.split()) > 1 and x.split()[1].isdigit() else 999,
            x
        ))

        # Print header (tighter column spacing)
        phase_col_w = 34
        variant_col_w = 21
        header = f"{'Phase':<{phase_col_w}}"
        for row_name, _, _ in causalflare_variants:
            header += f" {row_name:<{variant_col_w}}"
        print(header)
        print("-"*100)

        # Print each phase row
        for phase_name in phase_order:
            row = f"{phase_name:<{phase_col_w}}"
            for variant_name, _, _ in causalflare_variants:
                timings = triton3_avg_timings_by_variant.get(variant_name, {})
                total3 = timings.get("Total", 0.0)
                if phase_name in timings:
                    phase_time3 = timings[phase_name]
                    percentage3 = (phase_time3 / total3 * 100) if total3 > 0 else 0.0
                    cell3 = f"{phase_time3:.3f}ms / {percentage3:.1f}%"
                    row += f" {cell3:<{variant_col_w}}"
                else:
                    row += f" {'N/A':<{variant_col_w}}"

            print(row)

        # Print total row
        print("-"*100)
        total_row = f"{'Total':<{phase_col_w}}"
        for variant_name, _, _ in causalflare_variants:
            timings = triton3_avg_timings_by_variant.get(variant_name, {})
            if "Total" in timings:
                total3_val = timings["Total"]
                total_row += f" {total3_val:.3f}ms / 100.0%".ljust(variant_col_w + 1)
            else:
                total_row += f" {'N/A':<{variant_col_w}}"

        print(total_row)
        print("="*100 + "\n")

    #======================================================================#
    # Benchmark backward pass
    #======================================================================#
    def _unwrap_output(output):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def _run_backward(func, q, k, v, *args, **kwargs):
        if q.grad is not None:
            q.grad = None
        if k.grad is not None:
            k.grad = None
        if v.grad is not None:
            v.grad = None
        out = _unwrap_output(func(q, k, v, *args, **kwargs))
        out.sum().backward()

    def _bench_backward(func, q, k, v, *args, **kwargs):
        def _run():
            with torch.enable_grad():
                _run_backward(func, q, k, v, *args, **kwargs)

        ms = triton.testing.do_bench(_run)
        _, mem = measure_memory(_run)
        return ms, mem

    def _probe_backward_compile(func, q_src, k_src, v_src, *args, **kwargs):
        def _run():
            q = q_src.detach().requires_grad_(True)
            k = k_src.detach().requires_grad_(True)
            v = v_src.detach().requires_grad_(True)
            with torch.enable_grad():
                _run_backward(func, q, k, v, *args, **kwargs)
        return _probe_compile(_run)

    def _grad_errors(name, qg, kg, vg, qg_ref, kg_ref, vg_ref):
        q_err = compute_errors(qg, qg_ref, f"{name}_qg")
        k_err = compute_errors(kg, kg_ref, f"{name}_kg")
        v_err = compute_errors(vg, vg_ref, f"{name}_vg")
        mean_abs = max(q_err[f"{name}_qg_mean_abs_err"], k_err[f"{name}_kg_mean_abs_err"], v_err[f"{name}_vg_mean_abs_err"])
        max_abs = max(q_err[f"{name}_qg_max_abs_err"], k_err[f"{name}_kg_max_abs_err"], v_err[f"{name}_vg_max_abs_err"])
        mean_rel = max(q_err[f"{name}_qg_mean_rel_err"], k_err[f"{name}_kg_mean_rel_err"], v_err[f"{name}_vg_mean_rel_err"])
        max_rel = max(q_err[f"{name}_qg_max_rel_err"], k_err[f"{name}_kg_max_rel_err"], v_err[f"{name}_vg_max_rel_err"])
        allclose = q_err[f"{name}_qg_allclose"] and k_err[f"{name}_kg_allclose"] and v_err[f"{name}_vg_allclose"]
        return {
            "allclose": allclose,
            "mean_abs": mean_abs,
            "max_abs": max_abs,
            "mean_rel": mean_rel,
            "max_rel": max_rel,
        }

    # Prepare leaf tensors per method
    Q_ref = Q.detach().requires_grad_(True)
    K_ref = K.detach().requires_grad_(True)
    V_ref = V.detach().requires_grad_(True)

    Q_p2_ref = Q.detach().requires_grad_(True)
    K_p2_ref = K.detach().requires_grad_(True)
    V_p2_ref = V.detach().requires_grad_(True)

    Q_p2 = Q.detach().requires_grad_(True)
    K_p2 = K.detach().requires_grad_(True)
    V_p2 = V.detach().requires_grad_(True)

    Q_perciever_imlp = Q.detach().requires_grad_(True)
    K_perciever_imlp = K.detach().requires_grad_(True)
    V_perciever_imlp = V.detach().requires_grad_(True)
    Q_old_bwd = Q.detach().requires_grad_(True)
    K_old_bwd = K.detach().requires_grad_(True)
    V_old_bwd = V.detach().requires_grad_(True)

    Q_sd = Q_.detach().requires_grad_(True)
    K_sd = K_.detach().requires_grad_(True)
    V_sd = V_.detach().requires_grad_(True)
    Q_fa2 = Q_fa2_in.detach().requires_grad_(True)
    K_fa2 = K_fa2_in.detach().requires_grad_(True)
    V_fa2 = V_fa2_in.detach().requires_grad_(True)
    Q_rec_tr_bwd = Q.detach().requires_grad_(True)
    K_rec_tr_bwd = K.detach().requires_grad_(True)
    V_rec_tr_bwd = V.detach().requires_grad_(True)
    Q_rec_orig_bwd = Q.detach().requires_grad_(True)
    K_rec_orig_bwd = K.detach().requires_grad_(True)
    V_rec_orig_bwd = V.detach().requires_grad_(True)
    Q_pd = Q.detach().requires_grad_(True)
    K_pd = K.detach().requires_grad_(True)
    V_pd = V.detach().requires_grad_(True)
    Q_dense = Q.detach().requires_grad_(True)
    K_dense = K.detach().requires_grad_(True)
    V_dense = V.detach().requires_grad_(True)

    bwd_results = {}
    bwd_grad_errors = {}
    _BWD_PROFILE_TIMINGS.clear()

    # Accuracy reference grads from PyTorch 2
    try:
        with torch.enable_grad():
            _run_backward(flare_causal_chunked, Q_p2_ref, K_p2_ref, V_p2_ref, scale=scale)
        qg_ref = Q_p2_ref.grad.detach()
        kg_ref = K_p2_ref.grad.detach()
        vg_ref = V_p2_ref.grad.detach()
    except Exception as exc:
        print(f"[FLARE DEBUG] PyTorch2 backward reference failed: {exc}")
        Q_ref_f = Q.detach().float().requires_grad_(True)
        K_ref_f = K.detach().float().requires_grad_(True)
        V_ref_f = V.detach().float().requires_grad_(True)
        with torch.enable_grad():
            _run_backward(flare_causal_reference, Q_ref_f, K_ref_f, V_ref_f, scale=scale)
        qg_ref = Q_ref_f.grad.detach()
        kg_ref = K_ref_f.grad.detach()
        vg_ref = V_ref_f.grad.detach()

    # Benchmark + grad accuracy for selected methods
    try:
        print("Measuring backward Reference...", end=" ", flush=True)
        ref_bwd_compile_ms = _probe_backward_compile(flare_causal_reference, Q, K, V, scale=scale)
        bwd_results["Reference"] = _bench_backward(flare_causal_reference, Q_ref, K_ref, V_ref, scale=scale)
        _run_backward(flare_causal_reference, Q_ref, K_ref, V_ref, scale=scale)
        bwd_grad_errors["Reference"] = _grad_errors("reference", Q_ref.grad, K_ref.grad, V_ref.grad, qg_ref, kg_ref, vg_ref)
        print(f"Done{_compile_suffix(ref_bwd_compile_ms)}")
    except Exception:
        bwd_results["Reference"] = (float("nan"), 0.0)
        print("Failed")

    try:
        print("Measuring backward PyTorch 2...", end=" ", flush=True)
        p2_bwd_compile_ms = _probe_backward_compile(flare_causal_chunked, Q, K, V, scale=scale)
        bwd_results["PyTorch 2"] = _bench_backward(
            flare_causal_chunked, Q_p2, K_p2, V_p2, scale=scale
        )
        _run_backward(flare_causal_chunked, Q_p2, K_p2, V_p2, scale=scale)
        bwd_grad_errors["PyTorch 2"] = _grad_errors("pytorch2", Q_p2.grad, K_p2.grad, V_p2.grad, qg_ref, kg_ref, vg_ref)
        print(f"Done{_compile_suffix(p2_bwd_compile_ms)}")
    except Exception as exc:
        print(f"[FLARE DEBUG] PyTorch2 backward failed: {exc}")
        bwd_results["PyTorch 2"] = (float("nan"), 0.0)

    try:
        print("Measuring backward FA2 Triton...", end=" ", flush=True)
        if _flash_attention2_triton_bnhd is not None:
            fa2_bwd_compile_ms = _probe_backward_compile(
                _flash_attention2_triton_bnhd, Q_fa2_in, K_fa2_in, V_fa2_in, causal=True, sm_scale=scale
            )
            bwd_results["FA2 Triton"] = _bench_backward(
                _flash_attention2_triton_bnhd, Q_fa2, K_fa2, V_fa2, causal=True, sm_scale=scale
            )
            print(f"Done{_compile_suffix(fa2_bwd_compile_ms)}")
        else:
            bwd_results["FA2 Triton"] = (float("nan"), 0.0)
            print("Skipped (import unavailable)")
    except Exception as exc:
        print(f"[FLARE DEBUG] FA2 Triton backward failed: {exc}")
        bwd_results["FA2 Triton"] = (float("nan"), 0.0)

    try:
        print("Measuring backward PCVR-AR-like impl...", end=" ", flush=True)
        pcvr_bwd_compile_ms = _probe_backward_compile(flare_causal_perciever_ar, Q, K, V, scale=scale)
        bwd_results["PCVR-AR-like impl"] = _bench_backward(
            flare_causal_perciever_ar, Q_perciever_imlp, K_perciever_imlp, V_perciever_imlp, scale=scale
        )
        _run_backward(flare_causal_perciever_ar, Q_perciever_imlp, K_perciever_imlp, V_perciever_imlp, scale=scale)
        bwd_grad_errors["PCVR-AR-like impl"] = _grad_errors(
            "perciever_imlp",
            Q_perciever_imlp.grad,
            K_perciever_imlp.grad,
            V_perciever_imlp.grad,
            qg_ref,
            kg_ref,
            vg_ref,
        )
        print(f"Done{_compile_suffix(pcvr_bwd_compile_ms)}")
    except Exception as exc:
        print(f"[FLARE DEBUG] PCVR-AR-like impl backward failed (commonly OOM; non-blocking): {exc}")
        bwd_results["PCVR-AR-like impl"] = (float("nan"), 0.0)

    try:
        print("Measuring backward ChunkedFLAREOld...", end=" ", flush=True)
        old_bwd_compile_ms = _probe_backward_compile(ChunkedFLAREOld.apply, Q, K, V, scale)
        bwd_results["ChunkedFLAREOld"] = _bench_backward(ChunkedFLAREOld.apply, Q_old_bwd, K_old_bwd, V_old_bwd, scale)
        _run_backward(ChunkedFLAREOld.apply, Q_old_bwd, K_old_bwd, V_old_bwd, scale)
        bwd_grad_errors["ChunkedFLAREOld"] = _grad_errors(
            "triton3_old",
            Q_old_bwd.grad,
            K_old_bwd.grad,
            V_old_bwd.grad,
            qg_ref,
            kg_ref,
            vg_ref,
        )
        print(f"Done{_compile_suffix(old_bwd_compile_ms)}")
    except Exception as exc:
        print(f"[FLARE DEBUG] ChunkedFLAREOld backward failed: {exc}")
        bwd_results["ChunkedFLAREOld"] = (float("nan"), 0.0)

    for cf_name, precision_mode, grad_prefix in [
        ("ChunkedFLARE (ieee)", "ieee", "triton3_ieee"),
        ("ChunkedFLARE (tf32)", "tf32", "triton3_tf32"),
        ("ChunkedFLARE (tf32x3)", "tf32x3", "triton3_tf32x3"),
    ]:
        Q_t3 = Q.detach().requires_grad_(True)
        K_t3 = K.detach().requires_grad_(True)
        V_t3 = V.detach().requires_grad_(True)
        try:
            print(f"Measuring backward {cf_name}...", end=" ", flush=True)
            with _temp_env_var("FLARE_INPUT_PRECISION", precision_mode):
                cf_compile_ms = _probe_backward_compile(flare_chunk_triton, Q, K, V, scale)
                print(f"[compile done {cf_compile_ms:.2f} ms]", end=" ", flush=True)
                bwd_results[cf_name] = _bench_backward(flare_chunk_triton, Q_t3, K_t3, V_t3, scale)
                print("[bench done]", end=" ", flush=True)
                mode_key = f"triton3_{precision_mode}"
                if torch.cuda.is_available():
                    warmup = int(os.environ.get("FLARE_BWD_PROFILE_WARMUP", "10"))
                    reps = int(os.environ.get("FLARE_BWD_PROFILE_REPS", "50"))
                    print(f"[profiling warmup={warmup}]", end=" ", flush=True)
                    for _ in range(warmup):
                        _run_backward(flare_chunk_triton, Q_t3, K_t3, V_t3, scale)
                    timings_list = []
                    print(f"[profiling reps={reps}]", end=" ", flush=True)
                    for _ in range(reps):
                        _set_bwd_profile_mode(mode_key)
                        _run_backward(flare_chunk_triton, Q_t3, K_t3, V_t3, scale)
                        timings_list.append(dict(_BWD_PROFILE_TIMINGS.get(mode_key, {})))
                        _set_bwd_profile_mode(None)
                    if timings_list:
                        avg_timings = {}
                        all_keys = set()
                        for t in timings_list:
                            all_keys.update(t.keys())
                        for k in all_keys:
                            avg_timings[k] = sum(t.get(k, 0.0) for t in timings_list) / len(timings_list)
                        _BWD_PROFILE_TIMINGS[mode_key] = avg_timings
                else:
                    _set_bwd_profile_mode(mode_key)
                    _run_backward(flare_chunk_triton, Q_t3, K_t3, V_t3, scale)
                    _set_bwd_profile_mode(None)
            print(f"Done{_compile_suffix(cf_compile_ms)}")
            bwd_grad_errors[cf_name] = _grad_errors(grad_prefix, Q_t3.grad, K_t3.grad, V_t3.grad, qg_ref, kg_ref, vg_ref)
        except Exception as exc:
            _set_bwd_profile_mode(None)
            print(f"[FLARE DEBUG] {cf_name} backward failed: {exc}")
            bwd_results[cf_name] = (float("nan"), 0.0)

    if run_recurrent_triton:
        try:
            bwd_results["Recurrent Orig"] = _bench_backward(
                RecurrentFLARE.apply, Q_rec_orig_bwd, K_rec_orig_bwd, V_rec_orig_bwd, scale, None, None, None, None, 1
            )
            _run_backward(RecurrentFLARE.apply, Q_rec_orig_bwd, K_rec_orig_bwd, V_rec_orig_bwd, scale, None, None, None, None, 1)
            bwd_grad_errors["Recurrent Orig"] = _grad_errors(
                "rec_orig_bwd",
                Q_rec_orig_bwd.grad,
                K_rec_orig_bwd.grad,
                V_rec_orig_bwd.grad,
                qg_ref,
                kg_ref,
                vg_ref,
            )
        except Exception as exc:
            print(f"[FLARE DEBUG] Recurrent Orig backward failed: {exc}")
            bwd_results["Recurrent Orig"] = (float("nan"), 0.0)

        try:
            bwd_results["Recurrent Triton"] = _bench_backward(
                RecurrentFLARE.apply, Q_rec_tr_bwd, K_rec_tr_bwd, V_rec_tr_bwd, scale
            )
            _run_backward(RecurrentFLARE.apply, Q_rec_tr_bwd, K_rec_tr_bwd, V_rec_tr_bwd, scale)
            bwd_grad_errors["Recurrent Triton"] = _grad_errors(
                "rec_tr_bwd",
                Q_rec_tr_bwd.grad,
                K_rec_tr_bwd.grad,
                V_rec_tr_bwd.grad,
                qg_ref,
                kg_ref,
                vg_ref,
            )
        except Exception as exc:
            print(f"[FLARE DEBUG] Recurrent Triton backward failed: {exc}")
            bwd_results["Recurrent Triton"] = (float("nan"), 0.0)
    else:
        bwd_results["Recurrent Orig"] = (float("nan"), 0.0)
        bwd_results["Recurrent Triton"] = (float("nan"), 0.0)

    try:
        bwd_results["Causal SDPA"] = _bench_backward(causal_SDPA, Q_sd, K_sd, V_sd)
    except Exception:
        bwd_results["Causal SDPA"] = (float("nan"), 0.0)

    dense_bwd_err = None
    pytorch_dense_bwd_err = None
    if N <= 128:
        try:
            bwd_results["PyTorch Dense"] = _bench_backward(
                flare_causal_pytorch_dense, Q_pd, K_pd, V_pd, scale=scale
            )
            _run_backward(flare_causal_pytorch_dense, Q_pd, K_pd, V_pd, scale=scale)
            bwd_grad_errors["PyTorch Dense"] = _grad_errors(
                "pytorch_dense", Q_pd.grad, K_pd.grad, V_pd.grad, qg_ref, kg_ref, vg_ref
            )
        except Exception as exc:
            pytorch_dense_bwd_err = str(exc)
            bwd_results["PyTorch Dense"] = (float("nan"), 0.0)

        try:
            bwd_results["DenseFLARE"] = _bench_backward(DenseFLARE.apply, Q_dense, K_dense, V_dense, scale)
            _run_backward(DenseFLARE.apply, Q_dense, K_dense, V_dense, scale)
            bwd_grad_errors["DenseFLARE"] = _grad_errors("dense", Q_dense.grad, K_dense.grad, V_dense.grad, qg_ref, kg_ref, vg_ref)
        except Exception as exc:
            dense_bwd_err = str(exc)
            bwd_results["DenseFLARE"] = (float("nan"), 0.0)
    else:
        bwd_results["PyTorch Dense"] = (float("nan"), 0.0)
        bwd_results["DenseFLARE"] = (float("nan"), 0.0)

    print("\n" + "="*100)
    print(f"Testing Causal/Recurrent FLARE Backward Pass (B={B}, H={H}, M={M}, N={N}, D={D}, dtype={dtype})")
    print("="*100)
    print(f"{'Implementation':<20} {'Time (ms)':<10} {'Speedup':<10} {'Memory (GB)':<15} "
          f"{'Abs Err (mean/max)':<20} {'Rel Err (mean/max)':<20}")
    print("-"*100)

    fa2_bwd_ms = bwd_results.get("FA2 Triton", (float("nan"), 0.0))[0]
    bwd_speedup_base_name = "FA2 Triton" if fa2_bwd_ms > 0 and not math.isnan(fa2_bwd_ms) else "Causal SDPA"
    bwd_speedup_base_ms = fa2_bwd_ms if bwd_speedup_base_name == "FA2 Triton" else bwd_results["Causal SDPA"][0]
    for name in [
        "Reference",
        "PyTorch 2",
        "PCVR-AR-like impl",
        "ChunkedFLAREOld",
        "ChunkedFLARE (ieee)",
        "ChunkedFLARE (tf32)",
        "ChunkedFLARE (tf32x3)",
        "Recurrent Orig",
        "Recurrent Triton",
        "PyTorch Dense",
        "DenseFLARE",
        "FA2 Triton",
        "Causal SDPA",
    ]:
        ms, mem = bwd_results[name]
        speedup = (
            bwd_speedup_base_ms / ms
            if ms > 0 and not math.isnan(ms) and bwd_speedup_base_ms > 0 and not math.isnan(bwd_speedup_base_ms)
            else float("nan")
        )
        speedup_str = f"{speedup:.2f}x" if not math.isnan(speedup) else "N/A"
        ms_str = f"{ms:.2f}" if not math.isnan(ms) else "N/A"

        if name in bwd_grad_errors:
            err = bwd_grad_errors[name]
            abs_err_str = f"{err['mean_abs']:.2e}/{err['max_abs']:.2e}"
            rel_err_str = f"{err['mean_rel']:.2e}/{err['max_rel']:.2e}"
        else:
            abs_err_str = "N/A"
            rel_err_str = "N/A"

        if name == "PyTorch Dense" and pytorch_dense_bwd_err:
            print(f"{name:<20} {ms_str:<10} {speedup_str:<10} {mem:<15.2e} "
                  f"{abs_err_str:<20} {rel_err_str:<20} (ERR: {pytorch_dense_bwd_err})")
        elif name == "DenseFLARE" and dense_bwd_err:
            print(f"{name:<20} {ms_str:<10} {speedup_str:<10} {mem:<15.2e} "
                  f"{abs_err_str:<20} {rel_err_str:<20} (ERR: {dense_bwd_err})")
        else:
            print(f"{name:<20} {ms_str:<10} {speedup_str:<10} {mem:<15.2e} "
                  f"{abs_err_str:<20} {rel_err_str:<20}")

    if _BWD_PROFILE_TIMINGS:
        print("="*100)
        print("ChunkedFLARE Backward Phase Profiling")
        print("="*100)
        phase_col_w = 34
        variant_col_w = 21
        header = f"{'Phase':<{phase_col_w}}"
        bwd_profile_cols = [
            ("ChunkedFLARE (ieee)", "triton3_ieee"),
            ("ChunkedFLARE (tf32)", "triton3_tf32"),
            ("ChunkedFLARE (tf32x3)", "triton3_tf32x3"),
        ]
        for label, mode_key in bwd_profile_cols:
            if mode_key in _BWD_PROFILE_TIMINGS:
                header += f" {label:<{variant_col_w}}"
        print(header)
        print("-"*100)

        totals = {}
        for name, timings in _BWD_PROFILE_TIMINGS.items():
            totals[name] = sum(timings.values())

        # Normalize phase timings to match benchmark total time (do_bench) per precision mode.
        for label, mode_key in [
            ("ChunkedFLARE (ieee)", "triton3_ieee"),
            ("ChunkedFLARE (tf32)", "triton3_tf32"),
            ("ChunkedFLARE (tf32x3)", "triton3_tf32x3"),
        ]:
            if mode_key in _BWD_PROFILE_TIMINGS and label in bwd_results:
                bench_ms = bwd_results[label][0]
                total_ms = totals.get(mode_key, 0.0)
                if total_ms > 0 and bench_ms > 0 and not math.isnan(bench_ms):
                    scale = bench_ms / total_ms
                    for k in _BWD_PROFILE_TIMINGS[mode_key]:
                        _BWD_PROFILE_TIMINGS[mode_key][k] *= scale
                    totals[mode_key] = bench_ms

        all_phases = set()
        for timings in _BWD_PROFILE_TIMINGS.values():
            all_phases.update(timings.keys())
        phase_order = sorted(all_phases)

        for phase in phase_order:
            row = f"{phase:<{phase_col_w}}"
            for _, mode_key in bwd_profile_cols:
                if mode_key in _BWD_PROFILE_TIMINGS:
                    val = _BWD_PROFILE_TIMINGS[mode_key].get(phase, None)
                    total = totals.get(mode_key, 0.0)
                    if val is not None and total > 0:
                        cell = f"{val:.3f}ms / {val / total * 100:.1f}%"
                    else:
                        cell = "N/A"
                    row += f" {cell:<{variant_col_w}}"
            print(row)

        print("-"*100)
        total_row = f"{'Total':<{phase_col_w}}"
        for _, mode_key in bwd_profile_cols:
            if mode_key in totals:
                total_row += f" {totals[mode_key]:.3f}ms / 100.0%".ljust(variant_col_w + 1)
        print(total_row)
        print("="*100)

    return

def run_module_main(
    B: int = 8,
    H: int = 8,
    M: int = 64,
    N: int = 2048,
    D: int = 32,
    dtype: str = "bfloat16",
):
    cache_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "cache", "triton")
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("TRITON_CACHE_DIR", cache_dir)
    optimize_for_h100()
    return main(B=B, H=H, M=M, N=N, D=D, dtype=dtype)


#======================================================================#
if __name__ == "__main__":
    run_module_main()
#======================================================================#
#

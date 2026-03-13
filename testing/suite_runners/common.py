import causal_flare._common as _common_impl
import causal_flare.autoregressive.training as _chunked_impl
from causal_flare._common import *
from causal_flare.autoregressive.training import *
from causal_flare.autoregressive.dense import *
from causal_flare.autoregressive.inference import *
from causal_flare.autoregressive.recurrent import *
from causal_flare.autoregressive.reference import *
from benchmark.implementations.reference_ops import causal_SDPA

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
        score_head_dim=D,
        value_head_dim=D,
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










def _scaled_error_limit(err: float, multiplier: float, slack: float) -> float:
    return float(multiplier) * float(err) + float(slack)


#======================================================================#

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


# Export underscore-prefixed helpers for suite modules that use `from ... import *`.
__all__ = [name for name in globals() if not name.startswith("__")]

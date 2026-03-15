from causal_flare._common import *
from causal_flare._reference_utils import resolve_flare_causal_decode_inputs as _resolve_chunked_decode_inputs

def _maybe_record_kernel_resource(bucket: dict[str, dict[str, int]] | None, key: str, result) -> None:
    if bucket is None or result is None:
        return
    metadata = getattr(result, "metadata", None)
    n_regs = getattr(result, "n_regs", None)
    n_spills = getattr(result, "n_spills", None)
    if metadata is None or n_regs is None or n_spills is None:
        return
    bucket[key] = {
        "num_warps": int(metadata.num_warps),
        "num_stages": int(metadata.num_stages),
        "regs_per_thread": int(n_regs),
        "spills": int(n_spills),
        "shared_bytes": int(metadata.shared),
    }

def _profiled_call(device: torch.device, bucket: dict[str, float] | None, key: str, op):
    result, ms = _measure_op_ms(device, bucket is not None, op)
    _accumulate_timing(bucket, key, ms)
    return result

def _profiled_bwd_call(
    device: torch.device,
    bucket: dict[str, float] | None,
    key: str,
    op,
    *,
    resource_bucket: dict[str, dict[str, int]] | None = None,
):
    enabled = bucket is not None or _bwd_profile_enabled()
    result, ms = _measure_op_ms(device, enabled, op)
    _accumulate_timing(bucket, key, ms)
    if ms is not None:
        _record_bwd_timing(key, ms)
    _maybe_record_kernel_resource(resource_bucket, key, result)
    return result

def _refresh_profile_totals(profile_data: dict[str, object] | None) -> None:
    if profile_data is None:
        return
    forward = profile_data.get("forward", {})
    backward = profile_data.get("backward", {})
    forward_total = float(sum(forward.values())) if isinstance(forward, dict) else 0.0
    backward_total = float(sum(backward.values())) if isinstance(backward, dict) else 0.0
    profile_data["forward_total_ms"] = forward_total
    profile_data["backward_total_ms"] = backward_total
    profile_data["total_ms"] = forward_total + backward_total


def _resolve_forward_launch(
    phase: str,
    *,
    default_num_warps: int,
    default_num_stages: int,
) -> tuple[int, int]:
    phase_key = phase.upper()
    env_warps = os.environ.get(f"FLARE_{phase_key}_NUM_WARPS", "") or os.environ.get("FLARE_NUM_WARPS", "")
    env_stages = os.environ.get(f"FLARE_{phase_key}_NUM_STAGES", "") or os.environ.get("FLARE_NUM_STAGES", "")
    num_warps = int(env_warps) if env_warps else default_num_warps
    num_stages = int(env_stages) if env_stages else default_num_stages
    return num_warps, num_stages


def _resolve_backward_launch(
    phase: str,
    *,
    default_num_warps: int,
    default_num_stages: int,
) -> tuple[int, int]:
    phase_key = phase.upper()
    env_warps = os.environ.get(f"FLARE_LSE_BWD_{phase_key}_NUM_WARPS", "") or os.environ.get("FLARE_LSE_BWD_NUM_WARPS", "")
    env_stages = os.environ.get(f"FLARE_LSE_BWD_{phase_key}_NUM_STAGES", "") or os.environ.get("FLARE_LSE_BWD_NUM_STAGES", "")
    num_warps = int(env_warps) if env_warps else default_num_warps
    num_stages = int(env_stages) if env_stages else default_num_stages
    return num_warps, num_stages


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _largest_power_of_two_leq(value: int) -> int:
    if value <= 0:
        raise ValueError(f"Expected positive value, got {value}.")
    return 1 << (value.bit_length() - 1)


def _block_k_divisor(D: int, preferred: int) -> int:
    block_k = math.gcd(D, preferred)
    if block_k <= 0 or D % block_k != 0:
        raise ValueError(f"Unable to pick a valid BLOCK_K for D={D}, preferred={preferred}.")
    return block_k


def _snap_block_d_default(D: int, preferred: int) -> int:
    """Return a valid power-of-two D tile no larger than D."""
    tile = min(D, preferred)
    return tile if _is_power_of_two(tile) else _largest_power_of_two_leq(tile)


def _snap_block_m_default(M: int, preferred: int) -> int:
    """Return a valid power-of-two M tile no larger than M."""
    tile = min(M, preferred)
    return tile if _is_power_of_two(tile) else _largest_power_of_two_leq(tile)


def _require_power_of_two_tile(name: str, value: int) -> None:
    if not _is_power_of_two(value):
        raise ValueError(f"{name} must be a power of two. Got {name}={value}.")


def _get_chunked_forward_bucket(D: int) -> int:
    """Return the profiled forward bucket for the current head dimension.

    The buckets are explicit because the best H100 launch family is not monotonic in D:
    - 16 keeps the old low-overhead launch path.
    - 32/64 prefer lighter decoder/fwd launches than the historical 4w/2s policy.
    - 96 benefits from 64D tiles and 2 fwd warps.
    - 128 is the untiled large-state case.
    - 192/256/384/512 all preferred 128D replay tiles in profiling, but the best
      fwd warp count differs between 192 and the wider buckets.
    """
    for bucket in (16, 32, 64, 96, 128, 192, 256, 384, 512):
        if D <= bucket:
            return bucket
    return 512


def _get_chunked_forward_bucket_defaults(D: int) -> dict[str, object]:
    bucket = _get_chunked_forward_bucket(D)
    if bucket == 16:
        return {
            "prepare_block_d": 16,
            "prepare_block_k": 16,
            "prefix_block_d": 16,
            "fwd_block_d": 16,
            "fwd_block_k": 16,
            "prepare_launch": (4, 2),
            "prefix_launch": (4, 2),
            "decoder_launch": (4, 2),
            "fwd_launch": (4, 2),
        }
    if bucket == 32:
        return {
            "prepare_block_d": 32,
            "prepare_block_k": 32,
            "prefix_block_d": 32,
            "fwd_block_d": 32,
            "fwd_block_k": 32,
            "prepare_launch": (4, 2),
            "prefix_launch": (4, 2),
            "decoder_launch": (4, 1),
            "fwd_launch": (4, 1),
        }
    if bucket == 64:
        return {
            "prepare_block_d": 64,
            "prepare_block_k": 64,
            "prefix_block_d": 64,
            "fwd_block_d": 64,
            "fwd_block_k": 64,
            "prepare_launch": (4, 2),
            "prefix_launch": (4, 2),
            "decoder_launch": (4, 1),
            "fwd_launch": (4, 1),
        }
    if bucket == 96:
        return {
            "prepare_block_d": 64,
            "prepare_block_k": 32,
            "prefix_block_d": 64,
            "fwd_block_d": 64,
            "fwd_block_k": 32,
            "prepare_launch": (8, 3),
            "prefix_launch": (8, 3),
            "decoder_launch": (4, 1),
            "fwd_launch": (2, 1),
        }
    if bucket == 128:
        return {
            "prepare_block_d": 128,
            "prepare_block_k": 64,
            "prefix_block_d": 128,
            "fwd_block_d": 128,
            "fwd_block_k": 32,
            "prepare_launch": (8, 3),
            "prefix_launch": (8, 3),
            "decoder_launch": (4, 1),
            "fwd_launch": (4, 1),
        }
    if bucket == 192:
        return {
            "prepare_block_d": 128,
            "prepare_block_k": 64,
            "prefix_block_d": 128,
            "fwd_block_d": 128,
            "fwd_block_k": 32,
            "prepare_launch": (8, 3),
            "prefix_launch": (8, 3),
            "decoder_launch": (4, 1),
            "fwd_launch": (2, 1),
        }
    if bucket == 256:
        return {
            "prepare_block_d": 128,
            "prepare_block_k": 64,
            "prefix_block_d": 128,
            "fwd_block_d": 128,
            "fwd_block_k": 32,
            "prepare_launch": (8, 3),
            "prefix_launch": (8, 3),
            "decoder_launch": (4, 1),
            "fwd_launch": (4, 1),
        }
    if bucket == 384:
        return {
            "prepare_block_d": 128,
            "prepare_block_k": 64,
            "prefix_block_d": 128,
            "fwd_block_d": 128,
            "fwd_block_k": 32,
            "prepare_launch": (8, 3),
            "prefix_launch": (8, 3),
            "decoder_launch": (4, 1),
            "fwd_launch": (4, 1),
        }
    return {
        "prepare_block_d": 128,
        "prepare_block_k": 64,
        "prefix_block_d": 128,
        "fwd_block_d": 128,
        "fwd_block_k": 64,
        "prepare_launch": (8, 3),
        "prefix_launch": (8, 3),
        "decoder_launch": (4, 1),
        "fwd_launch": (4, 1),
    }


def _get_chunked_backward_bucket(D: int) -> int:
    """Return the backward launch bucket for the current head dimension."""
    return _get_chunked_forward_bucket(D)


def _get_chunked_backward_bucket_defaults(M: int, D_score: int, D_value: int) -> dict[str, object]:
    """Return structural backward defaults for the nearest supported score/value buckets."""
    bucket_score = _get_chunked_backward_bucket(D_score)
    bucket_value = _get_chunked_backward_bucket(D_value)
    wide_m_tile = _snap_block_m_default(M, 64)
    narrow_m_tile = _snap_block_m_default(M, 32 if M < 64 else 64)
    block_dv_state = 16 if bucket_value == 16 else (32 if bucket_value in (32, 64) else 64)
    if bucket_score == 16:
        return {
            "block_m": narrow_m_tile,
            "block_dv_state": block_dv_state,
            "block_k": 16,
            "block_d_part": 16,
            "qk_block_d": 16,
            "replay_launch": (4, 2),
            "state_launch": (4, 2),
            "qk_launch": (4, 2),
        }
    if bucket_score == 32:
        return {
            "block_m": wide_m_tile,
            "block_dv_state": block_dv_state,
            "block_k": 32,
            "block_d_part": 32,
            "qk_block_d": 32,
            "replay_launch": (4, 2),
            "state_launch": (4, 2),
            "qk_launch": (4, 2),
        }
    if bucket_score == 64:
        return {
            "block_m": wide_m_tile,
            "block_dv_state": block_dv_state,
            "block_k": 64,
            "block_d_part": 64,
            "qk_block_d": 64,
            "replay_launch": (4, 2),
            "state_launch": (4, 2),
            "qk_launch": (4, 2),
        }
    return {
        "block_m": wide_m_tile,
        "block_dv_state": block_dv_state,
        "block_k": 32,
        "block_d_part": 64,
        "qk_block_d": 64,
        "replay_launch": (4, 2),
        "state_launch": (4, 2),
        "qk_launch": (4, 2),
    }


def _select_chunked_bwd_qk_launch(M: int, D_score: int, chunk_size: int) -> tuple[int, int, int, int]:
    if D_score <= 32 and M <= 128 and chunk_size >= 64:
        block_t = 64
    else:
        block_t = 16 if chunk_size <= 32 else 32
    block_t = min(block_t, chunk_size)
    if (block_t % 16) != 0:
        raise ValueError(f"ChunkedFLARE LSE backward requires BLOCK_T_QK be a multiple of 16. Got BLOCK_T_QK={block_t}")
    preferred_block_d = 64 if D_score >= 64 else D_score
    block_d = _snap_block_d_default(D_score, preferred_block_d)
    if (block_d % 16) != 0:
        raise ValueError(f"ChunkedFLARE LSE backward requires QK_BLOCK_D be a multiple of 16. Got BLOCK_D={block_d}")
    return block_t, block_d, 4, 2


def _get_chunked_forward_config(
    M: int,
    N: int,
    score_head_dim: int,
    value_head_dim: int,
    dtype: torch.dtype,
    chunk_size=None,
    input_precision=None,
) -> dict[str, object]:
    use_fp16 = dtype == torch.float16
    use_bf16 = dtype == torch.bfloat16
    compute_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    if chunk_size is None:
        env_chunk = os.environ.get("FLARE_CHUNK_SIZE", "")
        if env_chunk:
            chunk_size = int(env_chunk)
        else:
            # Reduced-matrix tuning now consistently favors 64 over the
            # historical 128 baseline across the representative chunked
            # training shapes we care about.
            chunk_size = 64
    else:
        chunk_size = int(chunk_size)

    # Ensure block sizes are positive and divisible by 16
    assert M % 16 == 0, f"M must be divisible by 16. Got M={M}."
    assert score_head_dim % 16 == 0, f"score_head_dim must be divisible by 16. Got score_head_dim={score_head_dim}."
    assert value_head_dim % 16 == 0, f"value_head_dim must be divisible by 16. Got value_head_dim={value_head_dim}."
    assert chunk_size % 16 == 0, f"CHUNK_SIZE must be divisible by 16. Got CHUNK_SIZE={chunk_size}."

    # NOTE: each latent query (M axis) is independent in this forward path.
    # Today we use one BLOCK_M policy per launch, but this can be extended to
    # per-kernel/per-program BLOCK_M choices without changing the algorithm.
    block_m_env = os.environ.get("FLARE_BLOCK_M", "")
    if block_m_env:
        block_m = int(block_m_env)
    else:
        if M <= 16:
            block_m = _snap_block_m_default(M, 16)
        elif M <= 32:
            block_m = _snap_block_m_default(M, 32)
        else:
            block_m = _snap_block_m_default(M, 64)
    if block_m > M:
        raise ValueError(f"BLOCK_M must be <= M. Got M={M}, BLOCK_M={block_m}.")
    assert block_m % 16 == 0, f"BLOCK_M must be divisible by 16. Got BLOCK_M={block_m}."
    _require_power_of_two_tile("BLOCK_M", block_m)
    if os.environ.get("FLARE_DEBUG_PREFIX_STATS", "0") == "1":
        _DEBUG_PREFIX_STATS["block_m"] = block_m
        _DEBUG_PREFIX_STATS["chunk_size"] = chunk_size

    score_defaults = _get_chunked_forward_bucket_defaults(score_head_dim)
    value_defaults = _get_chunked_forward_bucket_defaults(value_head_dim)
    launch_defaults = _get_chunked_forward_bucket_defaults(max(score_head_dim, value_head_dim))

    prepare_block_d_env = os.environ.get("FLARE_PREPARE_BLOCK_D", "")
    prepare_block_d = int(prepare_block_d_env) if prepare_block_d_env else _snap_block_d_default(
        value_head_dim, int(value_defaults["prepare_block_d"])
    )
    if prepare_block_d > value_head_dim:
        raise ValueError(
            f"FLARE_PREPARE_BLOCK_D must be <= value_head_dim. Got value_head_dim={value_head_dim}, FLARE_PREPARE_BLOCK_D={prepare_block_d}."
        )
    assert prepare_block_d % 16 == 0, f"FLARE_PREPARE_BLOCK_D must be divisible by 16. Got {prepare_block_d}."
    _require_power_of_two_tile("FLARE_PREPARE_BLOCK_D", prepare_block_d)

    prepare_block_k_env = os.environ.get("FLARE_PREPARE_BLOCK_K", "")
    prepare_block_k = int(prepare_block_k_env) if prepare_block_k_env else _block_k_divisor(
        score_head_dim, int(score_defaults["prepare_block_k"])
    )
    assert prepare_block_k % 16 == 0, f"FLARE_PREPARE_BLOCK_K must be divisible by 16. Got {prepare_block_k}."
    assert score_head_dim % prepare_block_k == 0, (
        f"FLARE_PREPARE_BLOCK_K must divide score_head_dim. Got score_head_dim={score_head_dim}, FLARE_PREPARE_BLOCK_K={prepare_block_k}."
    )

    prefix_block_m_env = os.environ.get("FLARE_PREFIX_BLOCK_M", "")
    prefix_block_m = int(prefix_block_m_env) if prefix_block_m_env else block_m
    if prefix_block_m > M:
        raise ValueError(f"FLARE_PREFIX_BLOCK_M must be <= M. Got M={M}, FLARE_PREFIX_BLOCK_M={prefix_block_m}.")
    assert prefix_block_m % 16 == 0, f"FLARE_PREFIX_BLOCK_M must be divisible by 16. Got {prefix_block_m}."
    _require_power_of_two_tile("FLARE_PREFIX_BLOCK_M", prefix_block_m)

    prefix_block_d_env = os.environ.get("FLARE_PREFIX_BLOCK_D", "")
    prefix_block_d = int(prefix_block_d_env) if prefix_block_d_env else _snap_block_d_default(
        value_head_dim, int(value_defaults["prefix_block_d"])
    )
    if prefix_block_d > value_head_dim:
        raise ValueError(
            f"FLARE_PREFIX_BLOCK_D must be <= value_head_dim. Got value_head_dim={value_head_dim}, FLARE_PREFIX_BLOCK_D={prefix_block_d}."
        )
    assert prefix_block_d % 16 == 0, f"FLARE_PREFIX_BLOCK_D must be divisible by 16. Got {prefix_block_d}."
    _require_power_of_two_tile("FLARE_PREFIX_BLOCK_D", prefix_block_d)

    fwd_block_d_env = os.environ.get("FLARE_FWD_BLOCK_D", "")
    fwd_block_d = int(fwd_block_d_env) if fwd_block_d_env else _snap_block_d_default(
        value_head_dim, int(value_defaults["fwd_block_d"])
    )
    if fwd_block_d > value_head_dim:
        raise ValueError(f"FLARE_FWD_BLOCK_D must be <= value_head_dim. Got value_head_dim={value_head_dim}, FLARE_FWD_BLOCK_D={fwd_block_d}.")
    assert fwd_block_d % 16 == 0, f"FLARE_FWD_BLOCK_D must be divisible by 16. Got {fwd_block_d}."
    _require_power_of_two_tile("FLARE_FWD_BLOCK_D", fwd_block_d)

    fwd_block_k_env = os.environ.get("FLARE_FWD_BLOCK_K", "")
    fwd_block_k = int(fwd_block_k_env) if fwd_block_k_env else _block_k_divisor(
        score_head_dim, int(score_defaults["fwd_block_k"])
    )
    assert fwd_block_k % 16 == 0, f"FLARE_FWD_BLOCK_K must be divisible by 16. Got {fwd_block_k}."
    assert score_head_dim % fwd_block_k == 0, (
        f"FLARE_FWD_BLOCK_K must divide score_head_dim. Got score_head_dim={score_head_dim}, FLARE_FWD_BLOCK_K={fwd_block_k}."
    )

    block_t_env = os.environ.get("FLARE_BLOCK_T", "")
    block_t = int(block_t_env) if block_t_env else (16 if chunk_size >= 16 else chunk_size)
    block_t = min(block_t, chunk_size)

    prepare_num_warps, prepare_num_stages = _resolve_forward_launch(
        "prepare",
        default_num_warps=int(launch_defaults["prepare_launch"][0]),
        default_num_stages=int(launch_defaults["prepare_launch"][1]),
    )
    prefix_num_warps, prefix_num_stages = _resolve_forward_launch(
        "prefix",
        default_num_warps=int(launch_defaults["prefix_launch"][0]),
        default_num_stages=int(launch_defaults["prefix_launch"][1]),
    )
    decoder_num_warps, decoder_num_stages = _resolve_forward_launch(
        "decoder",
        default_num_warps=int(score_defaults["decoder_launch"][0]),
        default_num_stages=int(score_defaults["decoder_launch"][1]),
    )
    fwd_num_warps, fwd_num_stages = _resolve_forward_launch(
        "fwd",
        default_num_warps=int(launch_defaults["fwd_launch"][0]),
        default_num_stages=int(launch_defaults["fwd_launch"][1]),
    )

    return {
        "CHUNK_SIZE": chunk_size,
        "BLOCK_M": block_m,
        "PREPARE_BLOCK_D": prepare_block_d,
        "PREPARE_BLOCK_K": prepare_block_k,
        "PREFIX_BLOCK_M": prefix_block_m,
        "PREFIX_BLOCK_D": prefix_block_d,
        "FWD_BLOCK_D": fwd_block_d,
        "FWD_BLOCK_K": fwd_block_k,
        "BLOCK_T": block_t,
        "NUM_CHUNKS": math.ceil(N / chunk_size),
        "NUM_M_BLOCKS": math.ceil(M / block_m),
        "NUM_PREPARE_D_BLOCKS": math.ceil(value_head_dim / prepare_block_d),
        "NUM_PREFIX_M_BLOCKS": math.ceil(M / prefix_block_m),
        "NUM_PREFIX_D_BLOCKS": math.ceil(value_head_dim / prefix_block_d),
        "NUM_FWD_D_BLOCKS": math.ceil(value_head_dim / fwd_block_d),
        "use_fp16": use_fp16,
        "use_bf16": use_bf16,
        "compute_dtype": compute_dtype,
        "stats_fp32": True,
        "stats_dtype": torch.float32,
        "eps": _get_eps_for_dtype(dtype),
        "clamp_max": _get_exp_clamp_for_dtype(dtype),
        "input_precision": _normalize_input_precision(input_precision, None),
        "score_head_dim": score_head_dim,
        "value_head_dim": value_head_dim,
        "prepare_num_warps": prepare_num_warps,
        "prepare_num_stages": prepare_num_stages,
        "prefix_num_warps": prefix_num_warps,
        "prefix_num_stages": prefix_num_stages,
        "decoder_num_warps": decoder_num_warps,
        "decoder_num_stages": decoder_num_stages,
        "fwd_num_warps": fwd_num_warps,
        "fwd_num_stages": fwd_num_stages,
    }


def _run_chunked_prepare_phase(
    Q_comp: torch.Tensor,
    K_comp: torch.Tensor,
    V_comp: torch.Tensor,
    *,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict[str, object],
    kernel_timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    BH = K_comp.size(0) * H
    stats_dtype = config["stats_dtype"]
    device = Q_comp.device

    def alloc_chunk_stats():
        return (
            torch.full((BH, config["NUM_CHUNKS"], M), -float("inf"), device=device, dtype=stats_dtype),
            torch.zeros((BH, config["NUM_CHUNKS"], M), device=device, dtype=stats_dtype),
            torch.zeros((BH, config["NUM_CHUNKS"], M, D_value), device=device, dtype=stats_dtype),
        )

    chunk_max, chunk_den, chunk_num = _profiled_call(
        device,
        kernel_timings,
        "alloc_chunk_stats",
        alloc_chunk_stats,
    )

    Q_stride = Q_comp.stride()
    K_stride = K_comp.stride()
    V_stride = V_comp.stride()

    # Phase 1 uses the third grid dimension as a flattened (m_block, d_block)
    # space because Triton only gives us 3 launch dimensions. Each program owns
    # one [BLOCK_M, BLOCK_D] numerator tile and recomputes the shared
    # score/max/den recurrence locally for that m_block.
    def prepare_grid(meta):
        return (BH, config["NUM_CHUNKS"], config["NUM_M_BLOCKS"] * triton.cdiv(D_value, meta["BLOCK_D"]))

    def launch_prepare():
        flare_chunk_prepare[prepare_grid](
            K_comp, Q_comp, V_comp,
            chunk_max, chunk_den, chunk_num,
            K_stride[0], K_stride[1], K_stride[2], K_stride[3],
            Q_stride[0], Q_stride[1], Q_stride[2],
            V_stride[0], V_stride[1], V_stride[2], V_stride[3],
            *chunk_max.stride(),
            *chunk_den.stride(),
            *chunk_num.stride(),
            BH, M, N, D_score, D_value, scale,
            CHUNK_SIZE=config["CHUNK_SIZE"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["PREPARE_BLOCK_D"],
            BLOCK_K=config["PREPARE_BLOCK_K"],
            USE_FP16=config["use_fp16"],
            USE_BF16=config["use_bf16"],
            USE_FP32_STATS=config["stats_fp32"],
            INPUT_PRECISION=config["input_precision"],
            H=H,
            num_warps=config["prepare_num_warps"],
            num_stages=config["prepare_num_stages"],
        )

    _profiled_call(device, kernel_timings, "flare_chunk_prepare", launch_prepare)
    return chunk_max, chunk_den, chunk_num


def _run_chunked_prefix_phase(
    chunk_max: torch.Tensor,
    chunk_den: torch.Tensor,
    chunk_num: torch.Tensor,
    *,
    M: int,
    D_value: int,
    config: dict[str, object],
    kernel_timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    BH = chunk_max.size(0)
    stats_dtype = config["stats_dtype"]
    device = chunk_max.device

    def alloc_prefix_stats():
        return (
            torch.empty((BH, config["NUM_CHUNKS"], M), device=device, dtype=stats_dtype),
            torch.zeros((BH, config["NUM_CHUNKS"], M), device=device, dtype=stats_dtype),
            torch.zeros((BH, config["NUM_CHUNKS"], M, D_value), device=device, dtype=stats_dtype),
        )

    prefix_max, prefix_den, prefix_num = _profiled_call(
        device,
        kernel_timings,
        "alloc_prefix_stats",
        alloc_prefix_stats,
    )

    def prefix_grid(_meta):
        return (BH, config["NUM_PREFIX_M_BLOCKS"], triton.cdiv(D_value, config["PREFIX_BLOCK_D"]))

    def launch_prefix():
        flare_chunk_prefix[prefix_grid](
            chunk_max, chunk_den, chunk_num,
            prefix_max, prefix_den, prefix_num,
            *chunk_max.stride(),
            *chunk_den.stride(),
            *chunk_num.stride(),
            *prefix_max.stride(),
            *prefix_den.stride(),
            *prefix_num.stride(),
            BH, M, D_value, config["NUM_CHUNKS"],
            BLOCK_M=config["PREFIX_BLOCK_M"],
            BLOCK_D=config["PREFIX_BLOCK_D"],
            USE_FP16=config["use_fp16"],
            USE_BF16=config["use_bf16"],
            USE_FP32_STATS=config["stats_fp32"],
            num_warps=config["prefix_num_warps"],
            num_stages=config["prefix_num_stages"],
        )

    _profiled_call(device, kernel_timings, "flare_chunk_prefix", launch_prefix)
    return prefix_max, prefix_den, prefix_num


def _run_chunked_output_phase(
    Q_comp: torch.Tensor,
    K_comp: torch.Tensor,
    V_comp: torch.Tensor,
    Q_dec_comp: torch.Tensor,
    K_dec_comp: torch.Tensor,
    prefix_max: torch.Tensor,
    prefix_den: torch.Tensor,
    prefix_num: torch.Tensor,
    *,
    H: int,
    M: int,
    N: int,
    D_score: int,
    D_value: int,
    scale: float,
    config: dict[str, object],
    weight_sharing_enc_dec: bool,
    output_dtype: torch.dtype,
    zero_fill_output: bool = False,
    kernel_timings: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = K_comp.size(0)
    device = K_comp.device
    num_fwd_m_tiles = triton.cdiv(M, config["BLOCK_M"])

    def alloc_output_buffers():
        # With autotuned BLOCK_D the kernel may choose either the single-writer
        # or atomic-accumulation path, so keep the destination zero-initialized.
        O = torch.zeros((B, N, H, D_value), device=device, dtype=torch.float32)
        LSE_enc = torch.full((B, H, N, M), -float("inf"), device=device, dtype=torch.float32)
        LSE_dec = torch.empty((B * H, N), device=device, dtype=torch.float32)
        return O, LSE_enc, LSE_dec

    O, LSE_enc, LSE_dec = _profiled_call(
        device,
        kernel_timings,
        "alloc_output_buffers",
        alloc_output_buffers,
    )
    Q_stride = Q_comp.stride()
    K_stride = K_comp.stride()
    V_stride = V_comp.stride()
    Q_dec_stride = Q_dec_comp.stride()
    K_dec_stride = K_dec_comp.stride()
    O_stride = O.stride()
    LSE_enc_stride = LSE_enc.stride()
    LSE_dec_stride = LSE_dec.stride()

    def launch_decoder_lse():
        flare_chunk_decoder_lse[(B * H, config["NUM_CHUNKS"])](
            Q_dec_comp, K_dec_comp, LSE_dec,
            Q_dec_stride[0], Q_dec_stride[1], Q_dec_stride[2], Q_dec_stride[3],
            K_dec_stride[0], K_dec_stride[1], K_dec_stride[2],
            LSE_dec_stride[0], LSE_dec_stride[1],
            B * H, M, N, D_score, scale,
            CHUNK_SIZE=config["CHUNK_SIZE"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_K=config["FWD_BLOCK_K"],
            BLOCK_T=config["BLOCK_T"],
            INPUT_PRECISION=config["input_precision"],
            H=H,
            num_warps=config["decoder_num_warps"],
            num_stages=config["decoder_num_stages"],
        )

    _profiled_call(device, kernel_timings, "flare_chunk_decoder_lse", launch_decoder_lse)

    def fwd_grid(_meta):
        return (B * H, config["NUM_CHUNKS"], num_fwd_m_tiles * triton.cdiv(D_value, config["FWD_BLOCK_D"]))

    def launch_fwd():
        # Forward remains M/D tiled, while decoder normalization is supplied by
        # a dedicated LSE kernel so each tile can reuse precomputed decode
        # weights across the inner token replay loop.
        flare_chunk_fwd[fwd_grid](
            K_comp, Q_comp, V_comp, Q_dec_comp, K_dec_comp,
            prefix_max, prefix_den, prefix_num,
            O, LSE_enc, LSE_dec,
            K_stride[0], K_stride[1], K_stride[2], K_stride[3],
            Q_stride[0], Q_stride[1], Q_stride[2],
            V_stride[0], V_stride[1], V_stride[2], V_stride[3],
            Q_dec_stride[0], Q_dec_stride[1], Q_dec_stride[2], Q_dec_stride[3],
            K_dec_stride[0], K_dec_stride[1], K_dec_stride[2],
            *prefix_max.stride(),
            *prefix_den.stride(),
            *prefix_num.stride(),
            O_stride[0], O_stride[1], O_stride[2], O_stride[3],
            LSE_enc_stride[0], LSE_enc_stride[1], LSE_enc_stride[2], LSE_enc_stride[3],
            LSE_dec_stride[0], LSE_dec_stride[1],
            B * H, M, N, D_score, D_value, scale,
            CHUNK_SIZE=config["CHUNK_SIZE"],
            BLOCK_M=config["BLOCK_M"],
            BLOCK_D=config["FWD_BLOCK_D"],
            BLOCK_K=config["FWD_BLOCK_K"],
            BLOCK_T=config["BLOCK_T"],
            USE_FP16=config["use_fp16"],
            USE_BF16=config["use_bf16"],
            USE_FP32_STATS=config["stats_fp32"],
            INPUT_PRECISION=config["input_precision"],
            SINGLE_M_TILE=num_fwd_m_tiles == 1,
            WEIGHT_SHARING_ENC_DEC=weight_sharing_enc_dec,
            H=H,
            num_warps=config["fwd_num_warps"],
            num_stages=config["fwd_num_stages"],
        )

    _profiled_call(device, kernel_timings, "flare_chunk_fwd", launch_fwd)

    def cast_output():
        return O if O.dtype == output_dtype else O.to(output_dtype)

    O_out = _profiled_call(device, kernel_timings, "output_cast", cast_output)
    return O_out, LSE_enc, LSE_dec


class AutoRegressiveFLARE(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        Q,
        K,
        V,
        scale=None,
        chunk_size=None,
        input_precision=None,
        profile=False,
        Q_dec=None,
        K_dec=None,
    ):
        """
        Forward pass of causal FLARE.

        Args:
            Q: [H, M, D] - learnable latent queries
            K: [B, N, H, D] - keys from input sequence
            V: [B, N, H, D] - values from input sequence
            scale: optional scaling factor for attention. Defaults to 1.0 when D <= 8,
                otherwise D ** -0.5.
            input_precision: Triton dot precision mode in {"ieee","tf32","tf32x3"}.
            profile: if True, return a timings dict as a second output. The dict is
                populated with per-kernel forward timings immediately, and its
                backward section is filled in-place if the returned output later
                participates in autograd.
        Returns:
            Y: [B, N, H, D] - output sequence
            If profile=True, returns (Y, timings_dict) where:
                timings_dict["forward"]  -> per-kernel forward timings in ms
                timings_dict["backward"] -> per-kernel backward timings in ms
        """
        assert Q.dim() == 3 and K.dim() == 4 and V.dim() == 4, (
            "Q, K, V must be 3D and 4D tensors respectively "
            f"got Q.dim()={Q.dim()}, K.dim()={K.dim()}, V.dim()={V.dim()}"
        )
        assert K.shape[:3] == V.shape[:3], (
            "Expected K/V to agree on [B, N, H]. "
            f"Got K.shape={K.shape} and V.shape={V.shape}"
        )
        assert Q.size(0) == K.size(2) and Q.size(2) == K.size(3), (
            "Expected Q [H, M, D_k] and K [B, N, H, D_k]. "
            f"Got Q.shape={Q.shape} and K.shape={K.shape}"
        )

        H, M, D_score = Q.size()
        B, N, H, _ = K.size()
        D_value = V.size(3)
        dtype = K.dtype
        device = Q.device
        Q_comp = Q
        K_comp = K
        V_comp = V
        Q_dec, K_dec, separate_Q_dec, separate_K_dec, weight_sharing_enc_dec = _resolve_chunked_decode_inputs(
            Q, K, Q_dec, K_dec
        )
        Q_dec_comp = Q_dec if separate_Q_dec else K_comp
        K_dec_comp = K_dec if separate_K_dec else Q_comp
        scale = _resolve_attn_scale(scale, D_score)
        ctx.scale = scale
        cfg = _get_chunked_forward_config(
            M=M,
            N=N,
            score_head_dim=D_score,
            value_head_dim=D_value,
            dtype=dtype,
            chunk_size=chunk_size,
            input_precision=input_precision,
        )

        profile_data = {"forward": {}, "backward": {}} if profile else None
        forward_timings = profile_data["forward"] if profile_data is not None else None

        #---------------------------------------------------------------#
        # Parallel scan structure over sequence tokens (N axis):
        #   Split N key/value tokens into NUM_CHUNKS chunks of size CHUNK_SIZE.
        #   Phase 1 + Phase 2 perform chunk-level online-softmax composition
        #   (FA2-style state merge) to build per-chunk running softmax state
        #   (running max, denominator, numerator) before token-level rollout.
        # Phase 1: compute chunk-local sufficient statistics independently.
        #---------------------------------------------------------------#
        chunk_max, chunk_den, chunk_num = _run_chunked_prepare_phase(
            Q_comp,
            K_comp,
            V_comp,
            H=H,
            M=M,
            N=N,
            D_score=D_score,
            D_value=D_value,
            scale=scale,
            config=cfg,
            kernel_timings=forward_timings,
        )

        if os.environ.get("FLARE_DEBUG_PREFIX_STATS", "0") == "1":
            _DEBUG_PREFIX_STATS["chunk_max"] = chunk_max.detach().clone()
            _DEBUG_PREFIX_STATS["chunk_den"] = chunk_den.detach().clone()
            _DEBUG_PREFIX_STATS["chunk_num"] = chunk_num.detach().clone()

        #---------------------------------------------------------------#
        # Phase 2: combine chunk-local statistics into per-chunk prefix state.
        # Each chunk receives prefix_{chunk} = merged stats from all earlier chunks
        # (running max/den/num), i.e. the online-softmax state at chunk start.
        #---------------------------------------------------------------#
        prefix_max, prefix_den, prefix_num = _run_chunked_prefix_phase(
            chunk_max, chunk_den, chunk_num,
            M=M,
            D_value=D_value,
            config=cfg,
            kernel_timings=forward_timings,
        )

        if os.environ.get("FLARE_DEBUG_PREFIX_STATS", "0") == "1":
            _DEBUG_PREFIX_STATS["prefix_max"] = prefix_max.detach().clone()
            _DEBUG_PREFIX_STATS["prefix_den"] = prefix_den.detach().clone()
            _DEBUG_PREFIX_STATS["prefix_num"] = prefix_num.detach().clone()

        #---------------------------------------------------------------#
        # Phase 3: in parallel over chunks, continue token-level online-softmax
        # updates inside each chunk, initialized from Phase-2 prefix state.
        #---------------------------------------------------------------#
        O, LSE_enc, LSE_dec = _run_chunked_output_phase(
            Q_comp,
            K_comp,
            V_comp,
            Q_dec_comp,
            K_dec_comp,
            prefix_max, prefix_den, prefix_num,
            H=H,
            M=M,
            N=N,
            D_score=D_score,
            D_value=D_value,
            scale=scale,
            config=cfg,
            weight_sharing_enc_dec=weight_sharing_enc_dec,
            output_dtype=cfg["compute_dtype"],
            zero_fill_output=False,
            kernel_timings=forward_timings,
        )

        #---------------------------------------------------------------#
        # Save phase-3 per-token logsumexp trajectory (L) and chunk-start prefix
        # state for the LSE backward path.
        saved_tensors = [Q_comp, K_comp, V_comp, LSE_enc, LSE_dec, prefix_max, prefix_den, prefix_num]
        if separate_Q_dec:
            saved_tensors.append(Q_dec)
        if separate_K_dec:
            saved_tensors.append(K_dec)
        ctx.save_for_backward(*saved_tensors)
        ctx.chunk_size = cfg["CHUNK_SIZE"]
        ctx.eps = cfg["eps"]
        ctx.use_fp16 = cfg["use_fp16"]
        ctx.use_bf16 = cfg["use_bf16"]
        ctx.out_dtype = dtype
        ctx.use_fp32_stats = cfg["stats_fp32"]
        ctx.input_precision = cfg["input_precision"]
        ctx.H = H
        ctx.M = M
        ctx.N = N
        ctx.D_score = D_score
        ctx.D_value = D_value
        ctx.weight_sharing_enc_dec = weight_sharing_enc_dec
        ctx.separate_Q_dec = separate_Q_dec
        ctx.separate_K_dec = separate_K_dec
        ctx.profile_timings = profile_data
        #---------------------------------------------------------------#

        if profile:
            _refresh_profile_totals(profile_data)
            return O, profile_data
        return O

    @staticmethod
    def backward(ctx, dO, dTimings=None):
        if dO is None:
            return None, None, None, None, None, None, None, None, None
        return _chunked_flare_lse_backward_impl(ctx, dO, dTimings)


@triton.jit
def flare_chunk_prepare(
    K_ptr, Q_ptr, V_ptr,
    ChunkMax_ptr, ChunkDen_ptr, ChunkNum_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_cmax_bh, stride_cmax_chunk, stride_cmax_m,
    stride_cden_bh, stride_cden_chunk, stride_cden_m,
    stride_cnum_bh, stride_cnum_chunk, stride_cnum_m, stride_cnum_d,
    BH, M: tl.constexpr, N, D_SCORE: tl.constexpr, D_VALUE: tl.constexpr, scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    pid_md = tl.program_id(2)
    num_d_blocks = tl.cdiv(D_VALUE, BLOCK_D)
    pid_m = pid_md // num_d_blocks
    pid_d = pid_md % num_d_blocks
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    RCP_LN2: tl.constexpr = 1.4426950408889634

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets_out = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    token_offsets = tl.arange(0, CHUNK_SIZE)
    mask_m = m_offsets < M
    mask_d_out = d_offsets_out < D_VALUE
    mask_entries = mask_m[:, None] & mask_d_out[None, :]
    # chunk_max/chunk_den are shared across all output-D tiles for a fixed
    # (bh, chunk, m_block): they only depend on the score matrix S[t, m], which
    # is defined after reducing over the full head dimension D. Every d_block
    # recomputes that score reduction locally, but only d_block=0 writes the
    # shared scalar outputs to avoid write races. chunk_num is disjoint per D tile.
    store_stats = pid_d == 0

    chunk_start = chunk_idx * CHUNK_SIZE
    Q_base_ptr = Q_ptr + pid_h * stride_q_h
    K_base_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + chunk_start * stride_k_n
    V_base_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + chunk_start * stride_v_n
    token_idx = chunk_start + token_offsets
    token_mask = token_idx < N  # [C]

    state_dtype = tl.float32 if USE_FP32_STATS else (tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    # We tile the *output* numerator over D, but the score matrix S[t, m] still
    # needs the full q_m^T k_t reduction over the entire head dimension. To make
    # that exact, every output-D tile reruns the score reduction in BLOCK_K chunks.
    # This repeats score work across d_block programs, but keeps each program's
    # Q/K/V working set and chunk_num accumulator bounded by BLOCK_M/BLOCK_D.
    S = tl.zeros((CHUNK_SIZE, BLOCK_M), dtype=tl.float32)
    for dk in tl.static_range(0, D_SCORE, BLOCK_K):
        d_offsets_k = dk + tl.arange(0, BLOCK_K)
        mask_d_k = d_offsets_k < D_SCORE
        q_tile = tl.load(
            Q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets_k[None, :] * stride_q_d,
            mask=mask_m[:, None] & mask_d_k[None, :],
            other=0.0,
        ).to(state_dtype)
        k_tile = tl.load(
            K_base_ptr + token_offsets[:, None] * stride_k_n + d_offsets_k[None, :] * stride_k_d,
            mask=token_mask[:, None] & mask_d_k[None, :],
            other=0.0,
        ).to(state_dtype)
        S += tl.dot(k_tile, tl.trans(q_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    score_scale = scale * RCP_LN2
    S = S * score_scale  # [C, BM] in log2 units
    S = tl.where(token_mask[:, None] & mask_m[None, :], S, -float("inf"))

    # Compute the chunk-local scalar softmax stats from the fully reduced scores.
    chunk_max = tl.max(S, axis=0)  # [BM]
    chunk_max = tl.where(mask_m, chunk_max, 0.0)
    expS = tl.math.exp2(S - chunk_max[None, :])
    expS = tl.where(token_mask[:, None] & mask_m[None, :], expS, 0.0)
    expS_dot = expS if USE_FP32_STATS else (expS.to(tl.bfloat16) if USE_BF16 else (expS.to(tl.float16) if USE_FP16 else expS))
    chunk_den = tl.sum(expS, axis=0)  # [BM]
    # The numerator is separable across output-D tiles once expS is known, so
    # each program only forms its own [BLOCK_M, BLOCK_D] slice here.
    V_tile = tl.load(
        V_base_ptr + token_offsets[:, None] * stride_v_n + d_offsets_out[None, :] * stride_v_d,
        mask=token_mask[:, None] & mask_d_out[None, :],
        other=0.0,
    ).to(state_dtype)
    chunk_num = tl.dot(tl.trans(expS_dot), V_tile, out_dtype=tl.float32, input_precision=INPUT_PRECISION)  # [BM, BD]

    # Store chunk statistics
    chunk_max_ptr = ChunkMax_ptr + pid_bh * stride_cmax_bh + chunk_idx * stride_cmax_chunk
    chunk_den_ptr = ChunkDen_ptr + pid_bh * stride_cden_bh + chunk_idx * stride_cden_chunk
    chunk_num_ptr = ChunkNum_ptr + pid_bh * stride_cnum_bh + chunk_idx * stride_cnum_chunk

    tl.store(chunk_max_ptr + m_offsets * stride_cmax_m, chunk_max, mask=store_stats & mask_m)
    tl.store(chunk_den_ptr + m_offsets * stride_cden_m, chunk_den, mask=store_stats & mask_m)
    tl.store(
        chunk_num_ptr + m_offsets[:, None] * stride_cnum_m + d_offsets_out[None, :] * stride_cnum_d,
        chunk_num,
        mask=mask_entries,
    )


@triton.jit
def flare_chunk_prefix(
    ChunkMax_ptr, ChunkDen_ptr, ChunkNum_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    stride_cmax_bh, stride_cmax_chunk, stride_cmax_m,
    stride_cden_bh, stride_cden_chunk, stride_cden_m,
    stride_cnum_bh, stride_cnum_chunk, stride_cnum_m, stride_cnum_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    BH, M: tl.constexpr, D_VALUE: tl.constexpr, NUM_CHUNKS,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_d = tl.program_id(2)
    if pid_bh >= BH:
        return

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    mask_entries = mask_m[:, None] & mask_d[None, :]
    # prefix_max/prefix_den are shared across all D tiles for a fixed (bh, m_block):
    # they depend only on the chunk-level scalar stats, not on the numerator's D slice.
    # Each D tile recomputes the same [BLOCK_M] recurrence locally, but only d_block=0
    # writes those shared outputs to avoid write races. prefix_num is disjoint per D tile.
    store_stats = pid_d == 0

    state_dtype = tl.float32 if USE_FP32_STATS else (tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    # Initialize prefix state (will be updated as we process chunks sequentially)
    prefix_max_state = tl.full((BLOCK_M,), -float("inf"), dtype=state_dtype)
    prefix_den_state = tl.zeros((BLOCK_M,), dtype=state_dtype)
    prefix_num_state = tl.zeros((BLOCK_M, BLOCK_D), dtype=state_dtype)

    # Process all chunks sequentially for this batch/head
    # This eliminates redundant computation - we compute prefix stats once per chunk
    for chunk_idx in tl.range(0, NUM_CHUNKS):
        # Store prefix statistics BEFORE merging with current chunk
        # prefix_stats[chunk_idx] = cumulative stats from chunks 0 to chunk_idx-1
        prefix_max_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + chunk_idx * stride_pmax_chunk
        prefix_den_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + chunk_idx * stride_pden_chunk
        prefix_num_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + chunk_idx * stride_pnum_chunk

        tl.store(prefix_max_ptr + m_offsets * stride_pmax_m, prefix_max_state, mask=store_stats & mask_m)
        tl.store(prefix_den_ptr + m_offsets * stride_pden_m, prefix_den_state, mask=store_stats & mask_m)
        tl.store(prefix_num_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d, prefix_num_state, mask=mask_entries,)

        # Load chunk statistics for current chunk
        chunk_max_ptr = ChunkMax_ptr + pid_bh * stride_cmax_bh + chunk_idx * stride_cmax_chunk
        chunk_den_ptr = ChunkDen_ptr + pid_bh * stride_cden_bh + chunk_idx * stride_cden_chunk
        chunk_num_ptr = ChunkNum_ptr + pid_bh * stride_cnum_bh + chunk_idx * stride_cnum_chunk

        chunk_cm = tl.load(
            chunk_max_ptr + m_offsets * stride_cmax_m,
            mask=mask_m,
            other=-float("inf"),
        ).to(state_dtype)
        chunk_cd = tl.load(
            chunk_den_ptr + m_offsets * stride_cden_m,
            mask=mask_m,
            other=0.0,
        ).to(state_dtype)
        chunk_cs = tl.load(
            chunk_num_ptr + m_offsets[:, None] * stride_cnum_m + d_offsets[None, :] * stride_cnum_d,
            mask=mask_entries,
            other=0.0,
        ).to(state_dtype)

        # Merge current chunk statistics with prefix state using numerically stable softmax combination
        # This updates prefix_state to include chunks 0 to chunk_idx (for next iteration)
        max_new = tl.maximum(prefix_max_state, chunk_cm)
        scale_prev = tl.math.exp2((prefix_max_state - max_new).to(tl.float32)).to(state_dtype)
        scale_chunk = tl.math.exp2((chunk_cm - max_new).to(tl.float32)).to(state_dtype)

        prefix_den_state = (prefix_den_state * scale_prev + chunk_cd * scale_chunk).to(state_dtype)
        prefix_num_state = (prefix_num_state * scale_prev[:, None] + chunk_cs * scale_chunk[:, None]).to(state_dtype)
        prefix_max_state = max_new.to(state_dtype)


@triton.jit
def flare_chunk_decoder_lse(
    Q_dec_ptr, K_dec_ptr, LSE_dec_ptr,
    stride_qdb, stride_qdt, stride_qdh, stride_qdd,
    stride_kdh, stride_kdm, stride_kdd,
    stride_lsed_bh, stride_lsed_n,
    BH, M: tl.constexpr, N, D_SCORE: tl.constexpr, scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    chunk_start = chunk_idx * CHUNK_SIZE
    RCP_LN2: tl.constexpr = 1.4426950408889634
    score_scale = scale * RCP_LN2

    m_local = tl.arange(0, BLOCK_M)
    t_local = tl.arange(0, BLOCK_T)

    # NOTE:
    # This is the chunked analogue of recurrent decode-LSE:
    # tile [BLOCK_M, BLOCK_K] x [BLOCK_T, BLOCK_K] and run online softmax over M.
    # It keeps decode normalization numerically stable and tensor-core friendly.
    K_dec_base_ptr = K_dec_ptr + pid_h * stride_kdh
    Q_dec_base_ptr = Q_dec_ptr + pid_b * stride_qdb + pid_h * stride_qdh + chunk_start * stride_qdt

    t0 = 0
    while t0 < CHUNK_SIZE:
        t_offsets = t0 + t_local
        token_idx = chunk_start + t_offsets
        token_mask_t = token_idx < N
        lse_max = tl.full((BLOCK_T,), -float("inf"), tl.float32)
        lse_sum = tl.zeros((BLOCK_T,), tl.float32)

        m0 = 0
        while m0 < M:
            m_offsets = m0 + m_local
            mask_m = m_offsets < M
            s_sub = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)
            for k0 in tl.static_range(0, D_SCORE, BLOCK_K):
                d_offsets_k = k0 + tl.arange(0, BLOCK_K)
                mask_d_k = d_offsets_k < D_SCORE
                q_dec = tl.load(
                    Q_dec_base_ptr + t_offsets[:, None] * stride_qdt + d_offsets_k[None, :] * stride_qdd,
                    mask=token_mask_t[:, None] & mask_d_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                k_dec = tl.load(
                    K_dec_base_ptr + m_offsets[:, None] * stride_kdm + d_offsets_k[None, :] * stride_kdd,
                    mask=mask_m[:, None] & mask_d_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                s_sub += tl.dot(k_dec, tl.trans(q_dec), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

            s_sub = s_sub * score_scale
            s_sub = tl.where(mask_m[:, None] & token_mask_t[None, :], s_sub, -float("inf"))
            block_max = tl.max(s_sub, axis=0)
            new_max = tl.maximum(lse_max, block_max)
            same_inf = (lse_max == -float("inf")) & (new_max == -float("inf"))
            rescale_prev = tl.where(same_inf, 1.0, tl.math.exp2(lse_max - new_max))
            rescale_prev = tl.where(token_mask_t, rescale_prev, 1.0)
            block_exp = tl.math.exp2(s_sub - new_max[None, :])
            block_exp = tl.where(mask_m[:, None] & token_mask_t[None, :], block_exp, 0.0)
            lse_sum = lse_sum * rescale_prev + tl.sum(block_exp, axis=0)
            lse_max = new_max
            m0 += BLOCK_M

        lse_m = lse_max + tl.math.log2(lse_sum + 1e-20)
        tl.store(
            LSE_dec_ptr + pid_bh * stride_lsed_bh + token_idx * stride_lsed_n,
            lse_m,
            mask=token_mask_t,
        )
        t0 += BLOCK_T


@triton.jit
def flare_chunk_fwd(
    K_ptr, Q_ptr, V_ptr, Q_dec_ptr, K_dec_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    O_ptr, LSE_enc_ptr, LSE_dec_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_qdb, stride_qdt, stride_qdh, stride_qdd,
    stride_kdh, stride_kdm, stride_kdd,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    stride_o_b, stride_o_n, stride_o_h, stride_o_d,
    stride_lsee_b, stride_lsee_h, stride_lsee_n, stride_lsee_m,
    stride_lsed_bh, stride_lsed_n,
    BH, M: tl.constexpr, N, D_SCORE: tl.constexpr, D_VALUE: tl.constexpr, scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    SINGLE_M_TILE: tl.constexpr,
    WEIGHT_SHARING_ENC_DEC: tl.constexpr,
    H: tl.constexpr,
):
    """
    Phase 3: Dense output computation kernel.
    """
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    pid_md = tl.program_id(2)
    num_d_blocks = tl.cdiv(D_VALUE, BLOCK_D)
    if pid_bh >= BH:
        return

    pid_m = pid_md // num_d_blocks
    pid_d = pid_md - pid_m * num_d_blocks
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    RCP_LN2: tl.constexpr = 1.4426950408889634
    score_scale = scale * RCP_LN2

    chunk_start = chunk_idx * CHUNK_SIZE

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    mask_md = mask_m[:, None] & mask_d[None, :]

    # ---- Load Q [M, D] ----
    Q_base_ptr = Q_ptr + pid_h * stride_q_h
    # ---- Load prefix stats for this chunk (stats from all previous chunks) ----
    pmax_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + chunk_idx * stride_pmax_chunk
    pden_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + chunk_idx * stride_pden_chunk
    pnum_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + chunk_idx * stride_pnum_chunk

    stats_dtype = tl.float32 if USE_FP32_STATS else (tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    prefix_max = tl.load(
        pmax_ptr + m_offsets * stride_pmax_m,
        mask=mask_m,
        other=-float("inf"),
    ).to(stats_dtype)  # [M]
    prefix_den = tl.load(
        pden_ptr + m_offsets * stride_pden_m,
        mask=mask_m,
        other=0.0,
    ).to(stats_dtype)  # [M]
    prefix_num = tl.load(
        pnum_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d,
        mask=mask_md,
        other=0.0,
    ).to(stats_dtype)  # [M, BD]

    prefix_max_f = prefix_max.to(tl.float32)
    prefix_den_f = prefix_den.to(tl.float32)
    prefix_num_f = prefix_num.to(tl.float32)

    # ---- Load the local V chunk [C, BD] ----
    K_base_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + chunk_start * stride_k_n
    V_base_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + chunk_start * stride_v_n
    Q_dec_base_ptr = Q_dec_ptr + pid_b * stride_qdb + pid_h * stride_qdh + chunk_start * stride_qdt
    K_dec_base_ptr = K_dec_ptr + pid_h * stride_kdh

    O_base_ptr = O_ptr + pid_b * stride_o_b + pid_h * stride_o_h + chunk_start * stride_o_n

    m_state = prefix_max_f
    l_state = prefix_den_f
    n_state = prefix_num_f

    t0 = 0
    while t0 < CHUNK_SIZE:
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        token_mask_t = (chunk_start + t_offsets) < N

        # Each output-D tile owns only [M, BLOCK_D] of the numerator/output, but
        # the score matrix still needs the full q_m^T k_t reduction. We therefore
        # replay that score reduction across all BLOCK_K slices of D for every
        # d_block. This duplicates scalar score/max/den work, but keeps n_state
        # and output accumulation bounded to [M, BLOCK_D].
        s_sub = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)
        for k0 in tl.static_range(0, D_SCORE, BLOCK_K):
            d_offsets_k = k0 + tl.arange(0, BLOCK_K)
            mask_d_k = d_offsets_k < D_SCORE
            q_k = tl.load(
                Q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets_k[None, :] * stride_q_d,
                mask=mask_m[:, None] & mask_d_k[None, :],
                other=0.0,
            ).to(tl.float32)
            k_ptr = K_base_ptr + t_offsets[:, None] * stride_k_n + d_offsets_k[None, :] * stride_k_d
            k_sub = tl.load(
                k_ptr,
                mask=token_mask_t[:, None] & mask_d_k[None, :],
                other=0.0,
            ).to(tl.float32)
            s_sub += tl.dot(q_k, tl.trans(k_sub), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        s_sub = s_sub * score_scale
        s_sub = tl.where(mask_m[:, None] & token_mask_t[None, :], s_sub, -float("inf"))
        s_sub = tl.where(s_sub == s_sub, s_sub, -float("inf"))

        if WEIGHT_SHARING_ENC_DEC:
            decode_scores = s_sub
        else:
            decode_scores = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)
            for k0 in tl.static_range(0, D_SCORE, BLOCK_K):
                d_offsets_k = k0 + tl.arange(0, BLOCK_K)
                mask_d_k = d_offsets_k < D_SCORE
                q_dec = tl.load(
                    Q_dec_base_ptr + t_offsets[:, None] * stride_qdt + d_offsets_k[None, :] * stride_qdd,
                    mask=token_mask_t[:, None] & mask_d_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                k_dec = tl.load(
                    K_dec_base_ptr + m_offsets[:, None] * stride_kdm + d_offsets_k[None, :] * stride_kdd,
                    mask=mask_m[:, None] & mask_d_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                decode_scores += tl.dot(k_dec, tl.trans(q_dec), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            decode_scores = decode_scores * score_scale
            decode_scores = tl.where(mask_m[:, None] & token_mask_t[None, :], decode_scores, -float("inf"))
            decode_scores = tl.where(decode_scores == decode_scores, decode_scores, -float("inf"))

        lse_block = tl.load(
            LSE_dec_ptr + pid_bh * stride_lsed_bh + (chunk_start + t_offsets) * stride_lsed_n,
            mask=token_mask_t,
            other=0.0,
        ).to(tl.float32)
        exp_decode_sub = tl.math.exp2(decode_scores - lse_block[None, :])
        exp_decode_sub = tl.where(mask_m[:, None] & token_mask_t[None, :], exp_decode_sub, 0.0)

        t_idx = tl.arange(0, BLOCK_T)
        for j in tl.static_range(0, BLOCK_T):
            valid_t = (t0 + j) < CHUNK_SIZE
            token_valid = (chunk_start + t0 + j) < N
            col_mask = t_idx[None, :] == j
            s_t = tl.sum(tl.where(col_mask, s_sub, 0.0), axis=1)
            s_t = tl.where(token_valid & mask_m, s_t, -float("inf"))
            s_t = tl.where(s_t == s_t, s_t, -float("inf"))

            m_new = tl.maximum(m_state, s_t)
            is_m_state_inf = m_state == -float("inf")
            is_m_new_inf = m_new == -float("inf")
            m_new_safe = tl.where(is_m_new_inf, 0.0, m_new)

            exp_prev = tl.where(
                is_m_state_inf & is_m_new_inf,
                1.0,
                tl.where(is_m_state_inf, 0.0, tl.math.exp2(m_state - m_new_safe)),
            )
            exp_s = tl.where(is_m_new_inf, 0.0, tl.math.exp2(s_t - m_new_safe))

            v_t = tl.load(
                V_base_ptr + (t0 + j) * stride_v_n + d_offsets * stride_v_d,
                mask=token_valid & mask_d,
                other=0.0,
            ).to(tl.float32)

            l_state = l_state * exp_prev + exp_s
            n_state = n_state * exp_prev[:, None] + exp_s[:, None] * v_t[None, :]
            m_state = m_new

            inv_l = 1.0 / tl.where(l_state > 0, l_state, 1.0)
            exp_a = tl.sum(tl.where(col_mask, exp_decode_sub, 0.0), axis=1)
            w = exp_a * inv_l
            o_t = tl.sum(w[:, None] * n_state, axis=0)
            o_ptr = O_base_ptr + (t0 + j) * stride_o_n + d_offsets * stride_o_d
            if SINGLE_M_TILE and BLOCK_D == D_VALUE:
                tl.store(o_ptr, o_t, mask=token_valid & mask_d)
            else:
                tl.atomic_add(o_ptr, o_t, mask=token_valid & mask_d)
            lse_t = m_state + tl.math.log2(tl.maximum(l_state, 1e-20))
            tl.store(
                LSE_enc_ptr
                + pid_b * stride_lsee_b
                + pid_h * stride_lsee_h
                + (chunk_start + t0 + j) * stride_lsee_n
                + m_offsets * stride_lsee_m,
                lse_t,
                mask=(pid_d == 0) & token_valid & mask_m,
            )

        t0 += BLOCK_T

@triton.jit
def flare_chunk_fwd_store(
    K_ptr, Q_ptr, V_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    S_ptr, M_ptr, L_ptr, SumExpV_ptr, NEnd_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    stride_s_bh, stride_s_chunk, stride_s_t, stride_s_m,
    stride_m_bh, stride_m_chunk, stride_m_t, stride_m_m,
    stride_l_bh, stride_l_chunk, stride_l_t, stride_l_m,
    stride_sev_bh, stride_sev_chunk, stride_sev_t, stride_sev_m, stride_sev_d,
    stride_nend_bh, stride_nend_chunk, stride_nend_m, stride_nend_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale, eps, clamp_max,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    STORE_S: tl.constexpr,
    STORE_SUMEXPV: tl.constexpr,
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
    token_mask = token_idx < N
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_md = mask_m[:, None] & mask_d[None, :]

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

    stats_dtype = tl.float32 if USE_FP32_STATS else (tl.bfloat16 if USE_BF16 else (tl.float16 if USE_FP16 else tl.float32))
    prefix_max = tl.load(pmax_ptr + m_offsets * stride_pmax_m, mask=mask_m, other=-float("inf")).to(stats_dtype)
    prefix_den = tl.load(pden_ptr + m_offsets * stride_pden_m, mask=mask_m, other=0.0).to(stats_dtype)
    prefix_num = tl.load(
        pnum_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d,
        mask=mask_md,
        other=0.0,
    ).to(stats_dtype)

    K_base_ptr = K_ptr + pid_b * stride_k_b + pid_h * stride_k_h + chunk_start * stride_k_n
    V_base_ptr = V_ptr + pid_b * stride_v_b + pid_h * stride_v_h + chunk_start * stride_v_n

    m_state = prefix_max.to(tl.float32)
    l_state = prefix_den.to(tl.float32)
    n_state = prefix_num.to(tl.float32)

    t0 = 0
    while t0 < CHUNK_SIZE:
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        token_mask_t = (chunk_start + t_offsets) < N

        k_ptr = K_base_ptr + t_offsets[:, None] * stride_k_n + d_offsets[None, :] * stride_k_d
        k_sub = tl.load(
            k_ptr,
            mask=token_mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)

        s_sub = tl.dot(Q_vals_f, tl.trans(k_sub), out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale
        s_sub = tl.where(mask_m[:, None] & token_mask_t[None, :], s_sub, -float("inf"))
        s_sub = tl.where(s_sub == s_sub, s_sub, -float("inf"))

        if STORE_S:
            # S for this [BLOCK_T, M] tile is fully determined by s_sub; store once as a block.
            tl.store(
                S_ptr
                + pid_bh * stride_s_bh
                + chunk_idx * stride_s_chunk
                + (t0 + tl.arange(0, BLOCK_T))[:, None] * stride_s_t
                + m_offsets[None, :] * stride_s_m,
                tl.trans(s_sub),
                mask=token_mask_t[:, None] & mask_m[None, :],
            )

        t_idx = tl.arange(0, BLOCK_T)
        for j in tl.static_range(0, BLOCK_T):
            token_valid = (chunk_start + t0 + j) < N
            col_mask = t_idx == j
            s_t = tl.sum(tl.where(col_mask[None, :], s_sub, 0.0), axis=1)
            s_t = tl.where(token_valid & mask_m, s_t, -float("inf"))

            m_new = tl.maximum(m_state, s_t)
            is_m_state_inf = m_state == -float("inf")
            is_m_new_inf = m_new == -float("inf")
            m_new_safe = tl.where(is_m_new_inf, 0.0, m_new)

            exp_prev = tl.where(
                is_m_state_inf & is_m_new_inf,
                1.0,
                tl.where(is_m_state_inf, 0.0, tl.exp(m_state - m_new_safe)),
            )
            exp_s = tl.where(is_m_new_inf, 0.0, tl.exp(s_t - m_new_safe))

            v_t = tl.load(
                V_base_ptr + (t0 + j) * stride_v_n + d_offsets * stride_v_d,
                mask=token_valid & mask_d,
                other=0.0,
            ).to(tl.float32)

            l_state = l_state * exp_prev + exp_s
            n_state = n_state * exp_prev[:, None] + exp_s[:, None] * v_t[None, :]
            m_state = m_new

            exp_m = tl.exp(tl.minimum(m_state, clamp_max))
            sum_exp_v = n_state * exp_m[:, None]

            tl.store(
                M_ptr + pid_bh * stride_m_bh + chunk_idx * stride_m_chunk + (t0 + j) * stride_m_t + m_offsets * stride_m_m,
                m_state,
                mask=mask_m,
            )
            tl.store(
                L_ptr + pid_bh * stride_l_bh + chunk_idx * stride_l_chunk + (t0 + j) * stride_l_t + m_offsets * stride_l_m,
                l_state,
                mask=mask_m,
            )
            if STORE_SUMEXPV:
                tl.store(
                    SumExpV_ptr
                    + pid_bh * stride_sev_bh
                    + chunk_idx * stride_sev_chunk
                    + (t0 + j) * stride_sev_t
                    + m_offsets[:, None] * stride_sev_m
                    + d_offsets[None, :] * stride_sev_d,
                    sum_exp_v,
                    mask=mask_md,
                )
        t0 += BLOCK_T

    # Store chunk-terminal scaled numerator state (sumexpv), not raw n_state.
    # This stabilizes split backward transport by avoiding exp_prev division in
    # the common (unclamped) reverse update path.
    exp_m_end = tl.exp(tl.minimum(m_state, clamp_max))
    sumexpv_end = n_state * exp_m_end[:, None]
    tl.store(
        NEnd_ptr
        + pid_bh * stride_nend_bh
        + chunk_idx * stride_nend_chunk
        + m_offsets[:, None] * stride_nend_m
        + d_offsets[None, :] * stride_nend_d,
        sumexpv_end,
        mask=mask_md,
    )

@triton.jit
def flare_chunk_bwd_recurrent_state(
    K_ptr, Q_ptr, V_ptr,
    PrefixMax_ptr, PrefixDen_ptr, PrefixNum_ptr,
    S_ptr, M_ptr, L_ptr, NState_ptr, LSE_ptr,
    dO_ptr,
    dS_ptr, dV_ptr, dNstate_ptr, dQ_ptr, dK_ptr,
    dPmax_ptr, dPden_ptr, dPnum_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    stride_s_bh, stride_s_chunk, stride_s_t, stride_s_m,
    stride_m_bh, stride_m_chunk, stride_m_t, stride_m_m,
    stride_l_bh, stride_l_chunk, stride_l_t, stride_l_m,
    stride_nstate_bh, stride_nstate_chunk, stride_nstate_m, stride_nstate_d,
    stride_lse_b, stride_lse_h, stride_lse_n, stride_lse_m,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_ds_bh, stride_ds_chunk, stride_ds_t, stride_ds_m,
    stride_dv_b, stride_dv_n, stride_dv_h, stride_dv_d,
    stride_dns_bh, stride_dns_chunk, stride_dns_m, stride_dns_d,
    stride_dq_h, stride_dq_m, stride_dq_d,
    stride_dk_b, stride_dk_n, stride_dk_h, stride_dk_d,
    stride_dpmax_bh, stride_dpmax_chunk, stride_dpmax_m,
    stride_dpden_bh, stride_dpden_chunk, stride_dpden_m,
    stride_dpnum_bh, stride_dpnum_chunk, stride_dpnum_m, stride_dpnum_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale, eps, clamp_max,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    USE_FP16: tl.constexpr,
    USE_BF16: tl.constexpr,
    USE_FP32_STATS: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    RECOMPUTE_S: tl.constexpr,
    FUSE_QK: tl.constexpr,
    STORE_DP: tl.constexpr,
    USE_CTX_LSE: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    if pid_bh >= BH:
        return
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    chunk_start = chunk_idx * CHUNK_SIZE

    m_offsets = tl.arange(0, M)
    mask_m = m_offsets < M

    pmax_ptr = PrefixMax_ptr + pid_bh * stride_pmax_bh + chunk_idx * stride_pmax_chunk
    pden_ptr = PrefixDen_ptr + pid_bh * stride_pden_bh + chunk_idx * stride_pden_chunk
    pnum_ptr = PrefixNum_ptr + pid_bh * stride_pnum_bh + chunk_idx * stride_pnum_chunk

    prefix_max = tl.load(pmax_ptr + m_offsets * stride_pmax_m, mask=mask_m, other=-float("inf")).to(tl.float32)
    prefix_den = tl.load(pden_ptr + m_offsets * stride_pden_m, mask=mask_m, other=0.0).to(tl.float32)

    d_m = tl.zeros((M,), tl.float32)
    d_l = tl.zeros((M,), tl.float32)

    t_rev = 0
    while t_rev < CHUNK_SIZE:
        t = CHUNK_SIZE - 1 - t_rev
        token_valid = (chunk_start + t) < N
        valid_f = token_valid.to(tl.float32)

        if RECOMPUTE_S:
            s_t = tl.zeros((M,), tl.float32)
            d0s = 0
            while d0s < D:
                d_offsets_s = d0s + tl.arange(0, BLOCK_D)
                mask_d_s = d_offsets_s < D
                mask_md_s = mask_m[:, None] & mask_d_s[None, :]
                q_tile = tl.load(
                    Q_ptr + pid_h * stride_q_h + m_offsets[:, None] * stride_q_m + d_offsets_s[None, :] * stride_q_d,
                    mask=mask_md_s,
                    other=0.0,
                ).to(tl.float32)
                k_t_tile = tl.load(
                    K_ptr + pid_b * stride_k_b + (chunk_start + t) * stride_k_n + pid_h * stride_k_h + d_offsets_s * stride_k_d,
                    mask=mask_d_s,
                    other=0.0,
                ).to(tl.float32)
                s_t += tl.sum(q_tile * k_t_tile[None, :], axis=1)
                d0s += BLOCK_D
            s_t = s_t * scale
        else:
            s_t = tl.load(
                S_ptr + pid_bh * stride_s_bh + chunk_idx * stride_s_chunk + t * stride_s_t + m_offsets * stride_s_m,
                mask=mask_m,
                other=-float("inf"),
            ).to(tl.float32)
        m_t = tl.load(
            M_ptr + pid_bh * stride_m_bh + chunk_idx * stride_m_chunk + t * stride_m_t + m_offsets * stride_m_m,
            mask=mask_m,
            other=-float("inf"),
        ).to(tl.float32)
        if USE_CTX_LSE:
            lse_t = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + (chunk_start + t) * stride_lse_n + m_offsets * stride_lse_m,
                mask=token_valid & mask_m,
                other=-float("inf"),
            ).to(tl.float32)
            l_t = tl.exp(lse_t - m_t)
            l_t = tl.where(token_valid, l_t, 0.0)
        else:
            l_t = tl.load(
                L_ptr + pid_bh * stride_l_bh + chunk_idx * stride_l_chunk + t * stride_l_t + m_offsets * stride_l_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
        s_t = tl.where(token_valid, s_t, 0.0)

        exp_m = tl.exp(tl.minimum(m_t, clamp_max))
        if USE_CTX_LSE:
            sum_exp = tl.exp(lse_t + tl.minimum(m_t, clamp_max) - m_t)
            sum_exp = tl.where(token_valid, sum_exp, 0.0)
        else:
            sum_exp = l_t * exp_m

        s_max = tl.max(s_t, axis=0)
        s_exp = tl.exp(s_t - s_max)
        s_exp = tl.where(mask_m, s_exp, 0.0)
        s_sum = tl.sum(s_exp, axis=0)
        P_t = s_exp / (s_sum + 1e-20)
        P_t = P_t * valid_f

        expA_t = P_t / (sum_exp + eps)

        if t > 0:
            m_prev = tl.load(
                M_ptr + pid_bh * stride_m_bh + chunk_idx * stride_m_chunk + (t - 1) * stride_m_t + m_offsets * stride_m_m,
                mask=mask_m,
                other=-float("inf"),
            ).to(tl.float32)
            if USE_CTX_LSE:
                lse_prev = tl.load(
                    LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + (chunk_start + (t - 1)) * stride_lse_n + m_offsets * stride_lse_m,
                    mask=mask_m,
                    other=-float("inf"),
                ).to(tl.float32)
                l_prev = tl.exp(lse_prev - m_prev)
            else:
                l_prev = tl.load(
                    L_ptr + pid_bh * stride_l_bh + chunk_idx * stride_l_chunk + (t - 1) * stride_l_t + m_offsets * stride_l_m,
                    mask=mask_m,
                    other=0.0,
                ).to(tl.float32)
        else:
            m_prev = prefix_max
            if USE_CTX_LSE and chunk_start > 0:
                lse_prev = tl.load(
                    LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + (chunk_start - 1) * stride_lse_n + m_offsets * stride_lse_m,
                    mask=mask_m,
                    other=-float("inf"),
                ).to(tl.float32)
                l_prev = tl.exp(lse_prev - m_prev)
            else:
                l_prev = prefix_den

        exp_prev = tl.where(token_valid, tl.exp(m_prev - m_t), 1.0)
        exp_s = tl.where(token_valid, tl.exp(s_t - m_t), 0.0)
        inv_den = 1.0 / (sum_exp + eps)

        d_expA = tl.zeros((M,), tl.float32)
        # Track <d_n_total, exp_prev * n_prev> directly to avoid instability from
        # reconstructing n_prev via division by tiny exp_prev in sharp regimes.
        d_n_prev_dot_scaled = tl.zeros((M,), tl.float32)
        d_n_v_dot = tl.zeros((M,), tl.float32)
        d_n_dot_n_t = tl.zeros((M,), tl.float32)

        d0 = 0
        while d0 < D:
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D
            mask_md = mask_m[:, None] & mask_d[None, :]

            dO_tile = tl.load(
                dO_ptr + pid_b * stride_do_b + (chunk_start + t) * stride_do_n + pid_h * stride_do_h + d_offsets * stride_do_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32) * valid_f
            # NState stores sumexpv_t = n_t * exp(min(m_t, clamp_max)).
            u_t_tile = tl.load(
                NState_ptr
                + pid_bh * stride_nstate_bh
                + chunk_idx * stride_nstate_chunk
                + m_offsets[:, None] * stride_nstate_m
                + d_offsets[None, :] * stride_nstate_d,
                mask=mask_md,
                other=0.0,
            ).to(tl.float32)
            d_sumexp_v_tile = expA_t[:, None] * dO_tile[None, :]
            d_expA += tl.sum(u_t_tile * dO_tile[None, :], axis=1)

            d_u_tile = tl.load(
                dNstate_ptr
                + pid_bh * stride_dns_bh
                + chunk_idx * stride_dns_chunk
                + m_offsets[:, None] * stride_dns_m
                + d_offsets[None, :] * stride_dns_d,
                mask=mask_md,
                other=0.0,
            ).to(tl.float32)
            d_u_total_tile = d_u_tile + d_sumexp_v_tile
            inv_exp_m = tl.where(exp_m > 1e-20, 1.0 / exp_m, 0.0)
            n_t_tile = u_t_tile * inv_exp_m[:, None]
            d_n_total_tile = d_u_total_tile * exp_m[:, None]
            d_n_dot_n_t += tl.sum(d_sumexp_v_tile * n_t_tile, axis=1)
            v_t_tile = tl.load(
                V_ptr + pid_b * stride_v_b + (chunk_start + t) * stride_v_n + pid_h * stride_v_h + d_offsets * stride_v_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)

            beta = exp_s * exp_m
            u_prev_scaled_tile = u_t_tile - beta[:, None] * v_t_tile[None, :]
            n_prev_scaled_tile = n_t_tile - exp_s[:, None] * v_t_tile[None, :]
            alpha_prev = exp_prev * tl.exp(
                tl.minimum(m_t, clamp_max) - tl.minimum(m_prev, clamp_max)
            )
            if t > 0:
                alpha_prev_safe = tl.where(alpha_prev > 1e-20, alpha_prev, 1.0)
                u_prev_tile = u_prev_scaled_tile / alpha_prev_safe[:, None]
                u_prev_tile = tl.where(alpha_prev[:, None] > 1e-20, u_prev_tile, u_t_tile)
            else:
                u_prev_tile = tl.load(
                    pnum_ptr + m_offsets[:, None] * stride_pnum_m + d_offsets[None, :] * stride_pnum_d,
                    mask=mask_md,
                    other=0.0,
                ).to(tl.float32) * tl.exp(tl.minimum(m_prev, clamp_max))[:, None]

            d_n_prev_dot_scaled += tl.sum(d_n_total_tile * n_prev_scaled_tile, axis=1)
            d_n_v_dot += tl.sum(d_n_total_tile * v_t_tile[None, :], axis=1)

            dV_t_tile = tl.sum(d_n_total_tile * exp_s[:, None], axis=0) * valid_f
            tl.store(
                dV_ptr + pid_b * stride_dv_b + (chunk_start + t) * stride_dv_n + pid_h * stride_dv_h + d_offsets * stride_dv_d,
                dV_t_tile,
                mask=mask_d,
            )

            d_u_prev_tile = d_u_total_tile * alpha_prev[:, None]
            d_u_next_tile = tl.where(token_valid, d_u_prev_tile, d_u_tile)
            tl.store(
                dNstate_ptr
                + pid_bh * stride_dns_bh
                + chunk_idx * stride_dns_chunk
                + m_offsets[:, None] * stride_dns_m
                + d_offsets[None, :] * stride_dns_d,
                d_u_next_tile,
                mask=mask_md,
            )
            tl.store(
                NState_ptr
                + pid_bh * stride_nstate_bh
                + chunk_idx * stride_nstate_chunk
                + m_offsets[:, None] * stride_nstate_m
                + d_offsets[None, :] * stride_nstate_d,
                tl.where(token_valid, u_prev_tile, u_t_tile),
                mask=mask_md,
            )
            d0 += BLOCK_D

        d_sumexp = -d_expA * P_t * (inv_den * inv_den)
        d_l_out = d_sumexp * exp_m
        d_exp_m = d_sumexp * l_t + d_n_dot_n_t
        clamp_mask = m_t <= clamp_max
        d_m_out = d_exp_m * exp_m * clamp_mask
        d_m_total = d_m + d_m_out
        d_l_total = d_l + d_l_out

        dP = d_expA * inv_den
        dP_dot = tl.sum(dP * P_t, axis=0)
        d_s_soft = P_t * (dP - dP_dot)

        d_exp_prev_scaled = d_l_total * l_prev * exp_prev + d_n_prev_dot_scaled
        d_exp_s = d_l_total + d_n_v_dot
        d_l_prev = d_l_total * exp_prev
        d_m_prev = d_exp_prev_scaled
        d_m_t_from_exp = -(d_exp_prev_scaled + d_exp_s * exp_s)
        d_s_from_exp = d_exp_s * exp_s

        d_m_total = d_m_total + d_m_t_from_exp
        mask_prev = m_prev >= s_t
        d_m_prev = d_m_prev + d_m_total * mask_prev
        d_s_from_max = d_m_total * (~mask_prev)
        d_s = d_s_soft + d_s_from_exp + d_s_from_max

        if FUSE_QK:
            d0_qk = 0
            while d0_qk < D:
                d_offsets_qk = d0_qk + tl.arange(0, BLOCK_D)
                mask_d_qk = d_offsets_qk < D
                mask_md_qk = mask_m[:, None] & mask_d_qk[None, :]
                q_tile = tl.load(
                    Q_ptr + pid_h * stride_q_h + m_offsets[:, None] * stride_q_m + d_offsets_qk[None, :] * stride_q_d,
                    mask=mask_md_qk,
                    other=0.0,
                ).to(tl.float32)
                k_t_tile = tl.load(
                    K_ptr + pid_b * stride_k_b + (chunk_start + t) * stride_k_n + pid_h * stride_k_h + d_offsets_qk * stride_k_d,
                    mask=mask_d_qk,
                    other=0.0,
                ).to(tl.float32)
                dK_tile = tl.sum(d_s[:, None] * q_tile, axis=0) * scale * valid_f
                tl.store(
                    dK_ptr + pid_b * stride_dk_b + (chunk_start + t) * stride_dk_n + pid_h * stride_dk_h + d_offsets_qk * stride_dk_d,
                    dK_tile,
                    mask=mask_d_qk,
                )
                dQ_tile = d_s[:, None] * k_t_tile[None, :] * scale * valid_f
                tl.atomic_add(
                    dQ_ptr + pid_h * stride_dq_h + m_offsets[:, None] * stride_dq_m + d_offsets_qk[None, :] * stride_dq_d,
                    dQ_tile,
                    mask=mask_md_qk,
                )
                d0_qk += BLOCK_D
        else:
            tl.store(
                dS_ptr + pid_bh * stride_ds_bh + chunk_idx * stride_ds_chunk + t * stride_ds_t + m_offsets * stride_ds_m,
                d_s,
                mask=mask_m,
            )

        d_m = tl.where(token_valid, d_m_prev, d_m)
        d_l = tl.where(token_valid, d_l_prev, d_l)
        t_rev += 1

    if STORE_DP:
        tl.store(
            dPmax_ptr + pid_bh * stride_dpmax_bh + chunk_idx * stride_dpmax_chunk + m_offsets * stride_dpmax_m,
            d_m,
            mask=mask_m,
        )
        tl.store(
            dPden_ptr + pid_bh * stride_dpden_bh + chunk_idx * stride_dpden_chunk + m_offsets * stride_dpden_m,
            d_l,
            mask=mask_m,
        )
        # dNstate holds final dPnum after reverse scan.
        d0 = 0
        while d0 < D:
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D
            mask_md = mask_m[:, None] & mask_d[None, :]
            d_u_final = tl.load(
                dNstate_ptr
                + pid_bh * stride_dns_bh
                + chunk_idx * stride_dns_chunk
                + m_offsets[:, None] * stride_dns_m
                + d_offsets[None, :] * stride_dns_d,
                mask=mask_md,
                other=0.0,
            )
            # dPnum expects gradient wrt unscaled prefix_num (n-space),
            # while dNstate accumulates gradient wrt scaled prefix sumexpv.
            exp_prefix = tl.exp(tl.minimum(prefix_max, clamp_max))
            d_n_final = d_u_final * exp_prefix[:, None]
            tl.store(
                dPnum_ptr
                + pid_bh * stride_dpnum_bh
                + chunk_idx * stride_dpnum_chunk
                + m_offsets[:, None] * stride_dpnum_m
                + d_offsets[None, :] * stride_dpnum_d,
                d_n_final,
                mask=mask_md,
            )
            d0 += BLOCK_D


@triton.jit
def flare_chunk_bwd_recurrent_qk(
    K_ptr, Q_ptr,
    dS_ptr,
    dQ_ptr, dK_ptr,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_q_h, stride_q_m, stride_q_d,
    stride_ds_bh, stride_ds_chunk, stride_ds_t, stride_ds_m,
    stride_dq_h, stride_dq_m, stride_dq_d,
    stride_dk_b, stride_dk_n, stride_dk_h, stride_dk_d,
    BH, M: tl.constexpr, N, D: tl.constexpr, scale,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    ACCUM_DK: tl.constexpr,
    H: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    pid_d = tl.program_id(2)
    if pid_bh >= BH:
        return

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    chunk_start = chunk_idx * CHUNK_SIZE

    m_offsets = tl.arange(0, M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D
    mask_md = mask_m[:, None] & mask_d[None, :]

    q_vals = tl.load(
        Q_ptr + pid_h * stride_q_h + m_offsets[:, None] * stride_q_m + d_offsets[None, :] * stride_q_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)

    t0 = 0
    while t0 < CHUNK_SIZE:
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        token_valid = (chunk_start + t_offsets) < N
        mask_tm = token_valid[:, None] & mask_m[None, :]
        mask_td = token_valid[:, None] & mask_d[None, :]

        dS_block = tl.load(
            dS_ptr
            + pid_bh * stride_ds_bh
            + chunk_idx * stride_ds_chunk
            + t_offsets[:, None] * stride_ds_t
            + m_offsets[None, :] * stride_ds_m,
            mask=mask_tm,
            other=0.0,
        ).to(tl.float32)

        k_block = tl.load(
            K_ptr
            + pid_b * stride_k_b
            + (chunk_start + t_offsets)[:, None] * stride_k_n
            + pid_h * stride_k_h
            + d_offsets[None, :] * stride_k_d,
            mask=mask_td,
            other=0.0,
        ).to(tl.float32)

        dK_block = tl.dot(dS_block, q_vals, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale
        dQ_block = tl.dot(tl.trans(dS_block), k_block, out_dtype=tl.float32, input_precision=INPUT_PRECISION) * scale

        tl.atomic_add(dQ_ptr + pid_h * stride_dq_h + m_offsets[:, None] * stride_dq_m + d_offsets[None, :] * stride_dq_d, dQ_block, mask=mask_md)
        dk_ptr = (
            dK_ptr
            + pid_b * stride_dk_b
            + (chunk_start + t_offsets)[:, None] * stride_dk_n
            + pid_h * stride_dk_h
            + d_offsets[None, :] * stride_dk_d
        )
        if ACCUM_DK:
            tl.atomic_add(dk_ptr, dK_block, mask=mask_td)
        else:
            tl.store(dk_ptr, dK_block, mask=mask_td)
        t0 += BLOCK_T


@triton.jit
def _chunk_decode_score_vec(
    Q_dec_ptr,
    K_dec_ptr,
    stride_qdb, stride_qdt, stride_qdh, stride_qdd,
    stride_kdh, stride_kdm, stride_kdd,
    pid_b, pid_h, token_idx,
    m_offsets, mask_m,
    scale,
    D_SCORE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    RCP_LN2: tl.constexpr = 1.4426950408889634
    score_scale = scale * RCP_LN2
    a_t = tl.zeros([BLOCK_M], tl.float32)
    for d0 in tl.static_range(0, D_SCORE, BLOCK_K):
        d_offsets = d0 + tl.arange(0, BLOCK_K)
        mask_d = d_offsets < D_SCORE
        q_dec = tl.load(
            Q_dec_ptr + pid_b * stride_qdb + token_idx * stride_qdt + pid_h * stride_qdh + d_offsets * stride_qdd,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        k_dec = tl.load(
            K_dec_ptr + pid_h * stride_kdh + m_offsets[:, None] * stride_kdm + d_offsets[None, :] * stride_kdd,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        a_t += tl.sum(k_dec * q_dec[None, :], axis=1)
    return tl.where(mask_m, a_t * score_scale, -float("inf"))


@triton.jit
def flare_chunk_bwd_lse_p_part(
    Q_ptr,
    K_ptr,
    V_ptr,
    Q_dec_ptr,
    K_dec_ptr,
    PrefixMax_ptr,
    PrefixDen_ptr,
    PrefixNum_ptr,
    LSE_dec_ptr,
    dO_ptr,
    P_ptr,
    G_ptr,
    A_ptr,
    stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_pmax_bh, stride_pmax_chunk, stride_pmax_m,
    stride_pden_bh, stride_pden_chunk, stride_pden_m,
    stride_pnum_bh, stride_pnum_chunk, stride_pnum_m, stride_pnum_d,
    stride_qdb, stride_qdt, stride_qdh, stride_qdd,
    stride_kdh, stride_kdm, stride_kdd,
    stride_lsed_bh, stride_lsed_t,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_p_bh, stride_p_t, stride_p_m,
    stride_g_bh, stride_g_t, stride_g_m,
    stride_a_bh, stride_a_t, stride_a_m,
    H, M, N,
    scale,
    NUM_M_TILES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D_SCORE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_M_REPLAY: tl.constexpr,
    NUM_REPLAY_TILES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    USE_BLOCKED_SCORE_REPLAY: tl.constexpr,
    USE_SUBTILED_SCORE_REPLAY: tl.constexpr,
    WEIGHT_SHARING_ENC_DEC: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_cm = tl.program_id(1)
    pid_d = tl.program_id(2)
    RCP_LN2: tl.constexpr = 1.4426950408889634
    score_scale = scale * RCP_LN2

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    chunk_idx = pid_cm // NUM_M_TILES
    pid_m = pid_cm - chunk_idx * NUM_M_TILES
    m_start = pid_m * BLOCK_M
    chunk_start = chunk_idx * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, N)
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE

    if not USE_SUBTILED_SCORE_REPLAY:
        mask_prefix = mask_m & (chunk_start > 0)
        prefix_max = tl.load(
            PrefixMax_ptr
            + pid_bh * stride_pmax_bh
            + chunk_idx * stride_pmax_chunk
            + m_offsets * stride_pmax_m,
            mask=mask_prefix,
            other=-float("inf"),
        ).to(tl.float32)
        prefix_den = tl.load(
            PrefixDen_ptr
            + pid_bh * stride_pden_bh
            + chunk_idx * stride_pden_chunk
            + m_offsets * stride_pden_m,
            mask=mask_prefix,
            other=0.0,
        ).to(tl.float32)
        prefix_num = tl.load(
            PrefixNum_ptr
            + pid_bh * stride_pnum_bh
            + chunk_idx * stride_pnum_chunk
            + m_offsets[:, None] * stride_pnum_m
            + d_offsets[None, :] * stride_pnum_d,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        m_run = prefix_max
        d_run = prefix_den
        inv_prefix_den = 1.0 / tl.maximum(prefix_den, 1e-20)
        z_run = tl.where(mask_prefix[:, None], prefix_num * inv_prefix_den[:, None], 0.0)

    q_base_ptr = Q_ptr + pid_h * stride_q_h
    t = chunk_start
    if USE_BLOCKED_SCORE_REPLAY:
        while t < chunk_end:
            t_offsets = t + tl.arange(0, BLOCK_T)
            token_mask_t = t_offsets < chunk_end
            s_block = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)
            for d0 in tl.static_range(0, D_SCORE, BLOCK_K):
                d_offsets_k = d0 + tl.arange(0, BLOCK_K)
                mask_d_k = d_offsets_k < D_SCORE
                q_tile = tl.load(
                    q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets_k[None, :] * stride_q_d,
                    mask=mask_m[:, None] & mask_d_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                k_tile = tl.load(
                    K_ptr
                    + pid_b * stride_k_b
                    + t_offsets[:, None] * stride_k_n
                    + pid_h * stride_k_h
                    + d_offsets_k[None, :] * stride_k_d,
                    mask=token_mask_t[:, None] & mask_d_k[None, :],
                    other=0.0,
                ).to(tl.float32)
                s_block += tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            s_block = s_block * score_scale
            s_block = tl.where(mask_m[:, None] & token_mask_t[None, :], s_block, -float("inf"))

            t_idx = tl.arange(0, BLOCK_T)
            for j in tl.static_range(0, BLOCK_T):
                token_valid = (t + j) < chunk_end
                s_t = tl.sum(tl.where(t_idx[None, :] == j, s_block, 0.0), axis=1)
                s_t = tl.where(token_valid & mask_m, s_t, -float("inf"))

                m_new = tl.maximum(m_run, s_t)
                alpha = tl.math.exp2(m_run - m_new)
                beta = tl.math.exp2(s_t - m_new)
                d_new = alpha * d_run + beta
                inv_d = 1.0 / tl.maximum(d_new, 1e-20)
                coeff_old = alpha * d_run * inv_d
                coeff_new = beta * inv_d

                v_ptr = V_ptr + pid_b * stride_v_b + (t + j) * stride_v_n + pid_h * stride_v_h + d_offsets * stride_v_d
                v_t = tl.load(v_ptr, mask=token_valid & mask_d, other=0.0).to(tl.float32)
                z_run = coeff_old[:, None] * z_run + coeff_new[:, None] * v_t[None, :]

                do_ptr = dO_ptr + pid_b * stride_do_b + (t + j) * stride_do_n + pid_h * stride_do_h + d_offsets * stride_do_d
                u_t = tl.load(do_ptr, mask=token_valid & mask_d, other=0.0).to(tl.float32)
                p_part = tl.sum(z_run * u_t[None, :], axis=1)
                p_ptr = P_ptr + pid_bh * stride_p_bh + (t + j) * stride_p_t + m_offsets * stride_p_m
                if BLOCK_D < D_VALUE:
                    tl.atomic_add(p_ptr, p_part, mask=token_valid & mask_m)
                else:
                    tl.store(p_ptr, p_part, mask=token_valid & mask_m)
                lse_dec_t = tl.load(
                    LSE_dec_ptr + pid_bh * stride_lsed_bh + (t + j) * stride_lsed_t,
                    mask=token_valid,
                    other=0.0,
                ).to(tl.float32)
                if token_valid:
                    if WEIGHT_SHARING_ENC_DEC:
                        dec_score_t = s_t
                    else:
                        dec_score_t = _chunk_decode_score_vec(
                            Q_dec_ptr,
                            K_dec_ptr,
                            stride_qdb, stride_qdt, stride_qdh, stride_qdd,
                            stride_kdh, stride_kdm, stride_kdd,
                            pid_b, pid_h, t + j,
                            m_offsets, mask_m,
                            scale,
                            D_SCORE=D_SCORE,
                            BLOCK_M=BLOCK_M,
                            BLOCK_K=BLOCK_K,
                        )
                else:
                    dec_score_t = tl.full([BLOCK_M], -float("inf"), tl.float32)
                g_t = tl.where(
                    token_valid & mask_m,
                    tl.math.exp2(tl.minimum(dec_score_t - lse_dec_t, 0.0)),
                    0.0,
                )
                tl.store(
                    G_ptr + pid_bh * stride_g_bh + (t + j) * stride_g_t + m_offsets * stride_g_m,
                    g_t,
                    mask=token_valid & mask_m,
                )
                a_t = tl.where(token_valid & mask_m, coeff_new, 0.0)
                tl.store(
                    A_ptr + pid_bh * stride_a_bh + (t + j) * stride_a_t + m_offsets * stride_a_m,
                    a_t,
                    mask=token_valid & mask_m,
                )
                m_run = m_new
                d_run = d_new

            t += BLOCK_T
    else:
        if USE_SUBTILED_SCORE_REPLAY:
            t_local = tl.arange(0, BLOCK_T)
            tile_m_end = tl.minimum(m_start + BLOCK_M, M)

            sub_offsets_0 = m_start + tl.arange(0, BLOCK_M_REPLAY)
            mask_m_sub_0 = sub_offsets_0 < tile_m_end
            mask_prefix_sub_0 = mask_m_sub_0 & (chunk_start > 0)
            prefix_max_sub_0 = tl.load(
                PrefixMax_ptr
                + pid_bh * stride_pmax_bh
                + chunk_idx * stride_pmax_chunk
                + sub_offsets_0 * stride_pmax_m,
                mask=mask_prefix_sub_0,
                other=-float("inf"),
            ).to(tl.float32)
            prefix_den_sub_0 = tl.load(
                PrefixDen_ptr
                + pid_bh * stride_pden_bh
                + chunk_idx * stride_pden_chunk
                + sub_offsets_0 * stride_pden_m,
                mask=mask_prefix_sub_0,
                other=0.0,
            ).to(tl.float32)
            prefix_num_sub_0 = tl.load(
                PrefixNum_ptr
                + pid_bh * stride_pnum_bh
                + chunk_idx * stride_pnum_chunk
                + sub_offsets_0[:, None] * stride_pnum_m
                + d_offsets[None, :] * stride_pnum_d,
                mask=mask_m_sub_0[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            m_run_sub_0 = prefix_max_sub_0
            d_run_sub_0 = prefix_den_sub_0
            inv_prefix_den_sub_0 = 1.0 / tl.maximum(prefix_den_sub_0, 1e-20)
            z_run_sub_0 = tl.where(mask_prefix_sub_0[:, None], prefix_num_sub_0 * inv_prefix_den_sub_0[:, None], 0.0)

            if NUM_REPLAY_TILES > 1:
                sub_offsets_1 = m_start + BLOCK_M_REPLAY + tl.arange(0, BLOCK_M_REPLAY)
                mask_m_sub_1 = sub_offsets_1 < tile_m_end
                mask_prefix_sub_1 = mask_m_sub_1 & (chunk_start > 0)
                prefix_max_sub_1 = tl.load(
                    PrefixMax_ptr
                    + pid_bh * stride_pmax_bh
                    + chunk_idx * stride_pmax_chunk
                    + sub_offsets_1 * stride_pmax_m,
                    mask=mask_prefix_sub_1,
                    other=-float("inf"),
                ).to(tl.float32)
                prefix_den_sub_1 = tl.load(
                    PrefixDen_ptr
                    + pid_bh * stride_pden_bh
                    + chunk_idx * stride_pden_chunk
                    + sub_offsets_1 * stride_pden_m,
                    mask=mask_prefix_sub_1,
                    other=0.0,
                ).to(tl.float32)
                prefix_num_sub_1 = tl.load(
                    PrefixNum_ptr
                    + pid_bh * stride_pnum_bh
                    + chunk_idx * stride_pnum_chunk
                    + sub_offsets_1[:, None] * stride_pnum_m
                    + d_offsets[None, :] * stride_pnum_d,
                    mask=mask_m_sub_1[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                m_run_sub_1 = prefix_max_sub_1
                d_run_sub_1 = prefix_den_sub_1
                inv_prefix_den_sub_1 = 1.0 / tl.maximum(prefix_den_sub_1, 1e-20)
                z_run_sub_1 = tl.where(mask_prefix_sub_1[:, None], prefix_num_sub_1 * inv_prefix_den_sub_1[:, None], 0.0)
            if NUM_REPLAY_TILES > 2:
                sub_offsets_2 = m_start + 2 * BLOCK_M_REPLAY + tl.arange(0, BLOCK_M_REPLAY)
                mask_m_sub_2 = sub_offsets_2 < tile_m_end
                mask_prefix_sub_2 = mask_m_sub_2 & (chunk_start > 0)
                prefix_max_sub_2 = tl.load(
                    PrefixMax_ptr
                    + pid_bh * stride_pmax_bh
                    + chunk_idx * stride_pmax_chunk
                    + sub_offsets_2 * stride_pmax_m,
                    mask=mask_prefix_sub_2,
                    other=-float("inf"),
                ).to(tl.float32)
                prefix_den_sub_2 = tl.load(
                    PrefixDen_ptr
                    + pid_bh * stride_pden_bh
                    + chunk_idx * stride_pden_chunk
                    + sub_offsets_2 * stride_pden_m,
                    mask=mask_prefix_sub_2,
                    other=0.0,
                ).to(tl.float32)
                prefix_num_sub_2 = tl.load(
                    PrefixNum_ptr
                    + pid_bh * stride_pnum_bh
                    + chunk_idx * stride_pnum_chunk
                    + sub_offsets_2[:, None] * stride_pnum_m
                    + d_offsets[None, :] * stride_pnum_d,
                    mask=mask_m_sub_2[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                m_run_sub_2 = prefix_max_sub_2
                d_run_sub_2 = prefix_den_sub_2
                inv_prefix_den_sub_2 = 1.0 / tl.maximum(prefix_den_sub_2, 1e-20)
                z_run_sub_2 = tl.where(mask_prefix_sub_2[:, None], prefix_num_sub_2 * inv_prefix_den_sub_2[:, None], 0.0)
            if NUM_REPLAY_TILES > 3:
                sub_offsets_3 = m_start + 3 * BLOCK_M_REPLAY + tl.arange(0, BLOCK_M_REPLAY)
                mask_m_sub_3 = sub_offsets_3 < tile_m_end
                mask_prefix_sub_3 = mask_m_sub_3 & (chunk_start > 0)
                prefix_max_sub_3 = tl.load(
                    PrefixMax_ptr
                    + pid_bh * stride_pmax_bh
                    + chunk_idx * stride_pmax_chunk
                    + sub_offsets_3 * stride_pmax_m,
                    mask=mask_prefix_sub_3,
                    other=-float("inf"),
                ).to(tl.float32)
                prefix_den_sub_3 = tl.load(
                    PrefixDen_ptr
                    + pid_bh * stride_pden_bh
                    + chunk_idx * stride_pden_chunk
                    + sub_offsets_3 * stride_pden_m,
                    mask=mask_prefix_sub_3,
                    other=0.0,
                ).to(tl.float32)
                prefix_num_sub_3 = tl.load(
                    PrefixNum_ptr
                    + pid_bh * stride_pnum_bh
                    + chunk_idx * stride_pnum_chunk
                    + sub_offsets_3[:, None] * stride_pnum_m
                    + d_offsets[None, :] * stride_pnum_d,
                    mask=mask_m_sub_3[:, None] & mask_d[None, :],
                    other=0.0,
                ).to(tl.float32)
                m_run_sub_3 = prefix_max_sub_3
                d_run_sub_3 = prefix_den_sub_3
                inv_prefix_den_sub_3 = 1.0 / tl.maximum(prefix_den_sub_3, 1e-20)
                z_run_sub_3 = tl.where(mask_prefix_sub_3[:, None], prefix_num_sub_3 * inv_prefix_den_sub_3[:, None], 0.0)

            while t < chunk_end:
                t_offsets = t + t_local
                token_mask_t = t_offsets < chunk_end
                s_block_0 = tl.zeros((BLOCK_M_REPLAY, BLOCK_T), tl.float32)
                if NUM_REPLAY_TILES > 1:
                    s_block_1 = tl.zeros((BLOCK_M_REPLAY, BLOCK_T), tl.float32)
                if NUM_REPLAY_TILES > 2:
                    s_block_2 = tl.zeros((BLOCK_M_REPLAY, BLOCK_T), tl.float32)
                if NUM_REPLAY_TILES > 3:
                    s_block_3 = tl.zeros((BLOCK_M_REPLAY, BLOCK_T), tl.float32)

                for d0 in tl.static_range(0, D_SCORE, BLOCK_K):
                    d_offsets_k = d0 + tl.arange(0, BLOCK_K)
                    mask_d_k = d_offsets_k < D_SCORE
                    k_tile = tl.load(
                        K_ptr
                        + pid_b * stride_k_b
                        + t_offsets[:, None] * stride_k_n
                        + pid_h * stride_k_h
                        + d_offsets_k[None, :] * stride_k_d,
                        mask=token_mask_t[:, None] & mask_d_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    q_tile_0 = tl.load(
                        q_base_ptr + sub_offsets_0[:, None] * stride_q_m + d_offsets_k[None, :] * stride_q_d,
                        mask=mask_m_sub_0[:, None] & mask_d_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    s_block_0 = s_block_0 + tl.dot(q_tile_0, tl.trans(k_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                    if NUM_REPLAY_TILES > 1:
                        q_tile_1 = tl.load(
                            q_base_ptr + sub_offsets_1[:, None] * stride_q_m + d_offsets_k[None, :] * stride_q_d,
                            mask=mask_m_sub_1[:, None] & mask_d_k[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        s_block_1 = s_block_1 + tl.dot(
                            q_tile_1,
                            tl.trans(k_tile),
                            out_dtype=tl.float32,
                            input_precision=INPUT_PRECISION,
                        )
                    if NUM_REPLAY_TILES > 2:
                        q_tile_2 = tl.load(
                            q_base_ptr + sub_offsets_2[:, None] * stride_q_m + d_offsets_k[None, :] * stride_q_d,
                            mask=mask_m_sub_2[:, None] & mask_d_k[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        s_block_2 = s_block_2 + tl.dot(
                            q_tile_2,
                            tl.trans(k_tile),
                            out_dtype=tl.float32,
                            input_precision=INPUT_PRECISION,
                        )
                    if NUM_REPLAY_TILES > 3:
                        q_tile_3 = tl.load(
                            q_base_ptr + sub_offsets_3[:, None] * stride_q_m + d_offsets_k[None, :] * stride_q_d,
                            mask=mask_m_sub_3[:, None] & mask_d_k[None, :],
                            other=0.0,
                        ).to(tl.float32)
                        s_block_3 = s_block_3 + tl.dot(
                            q_tile_3,
                            tl.trans(k_tile),
                            out_dtype=tl.float32,
                            input_precision=INPUT_PRECISION,
                        )

                s_block_0 = s_block_0 * score_scale
                s_block_0 = tl.where(mask_m_sub_0[:, None] & token_mask_t[None, :], s_block_0, -float("inf"))
                if NUM_REPLAY_TILES > 1:
                    s_block_1 = s_block_1 * score_scale
                    s_block_1 = tl.where(mask_m_sub_1[:, None] & token_mask_t[None, :], s_block_1, -float("inf"))
                if NUM_REPLAY_TILES > 2:
                    s_block_2 = s_block_2 * score_scale
                    s_block_2 = tl.where(mask_m_sub_2[:, None] & token_mask_t[None, :], s_block_2, -float("inf"))
                if NUM_REPLAY_TILES > 3:
                    s_block_3 = s_block_3 * score_scale
                    s_block_3 = tl.where(mask_m_sub_3[:, None] & token_mask_t[None, :], s_block_3, -float("inf"))

                for j in tl.static_range(0, BLOCK_T):
                    token_valid = (t + j) < chunk_end
                    select_j = t_local == j
                    token_idx = t + j
                    v_ptr = V_ptr + pid_b * stride_v_b + token_idx * stride_v_n + pid_h * stride_v_h + d_offsets * stride_v_d
                    v_t = tl.load(v_ptr, mask=token_valid & mask_d, other=0.0).to(tl.float32)
                    do_ptr = dO_ptr + pid_b * stride_do_b + token_idx * stride_do_n + pid_h * stride_do_h + d_offsets * stride_do_d
                    u_t = tl.load(do_ptr, mask=token_valid & mask_d, other=0.0).to(tl.float32)
                    lse_dec_t = tl.load(
                        LSE_dec_ptr + pid_bh * stride_lsed_bh + token_idx * stride_lsed_t,
                        mask=token_valid,
                        other=0.0,
                    ).to(tl.float32)

                    s_t_0 = tl.sum(tl.where(select_j[None, :], s_block_0, 0.0), axis=1)
                    s_t_0 = tl.where(token_valid & mask_m_sub_0, s_t_0, -float("inf"))
                    m_new_0 = tl.maximum(m_run_sub_0, s_t_0)
                    alpha_0 = tl.math.exp2(m_run_sub_0 - m_new_0)
                    beta_0 = tl.math.exp2(s_t_0 - m_new_0)
                    d_new_0 = alpha_0 * d_run_sub_0 + beta_0
                    inv_d_0 = 1.0 / tl.maximum(d_new_0, 1e-20)
                    coeff_old_0 = alpha_0 * d_run_sub_0 * inv_d_0
                    coeff_new_0 = beta_0 * inv_d_0
                    z_new_0 = coeff_old_0[:, None] * z_run_sub_0 + coeff_new_0[:, None] * v_t[None, :]
                    p_part_0 = tl.sum(z_new_0 * u_t[None, :], axis=1)
                    p_ptr_0 = P_ptr + pid_bh * stride_p_bh + token_idx * stride_p_t + sub_offsets_0 * stride_p_m
                    if BLOCK_D < D_VALUE:
                        tl.atomic_add(p_ptr_0, p_part_0, mask=token_valid & mask_m_sub_0)
                    else:
                        tl.store(p_ptr_0, p_part_0, mask=token_valid & mask_m_sub_0)
                    if token_valid:
                        if WEIGHT_SHARING_ENC_DEC:
                            dec_score_0 = s_t_0
                        else:
                            dec_score_0 = _chunk_decode_score_vec(
                                Q_dec_ptr,
                                K_dec_ptr,
                                stride_qdb, stride_qdt, stride_qdh, stride_qdd,
                                stride_kdh, stride_kdm, stride_kdd,
                                pid_b, pid_h, token_idx,
                                sub_offsets_0, mask_m_sub_0,
                                scale,
                                D_SCORE=D_SCORE,
                                BLOCK_M=BLOCK_M_REPLAY,
                                BLOCK_K=BLOCK_K,
                            )
                    else:
                        dec_score_0 = tl.full([BLOCK_M_REPLAY], -float("inf"), tl.float32)
                    g_t_0 = tl.where(
                        token_valid & mask_m_sub_0,
                        tl.math.exp2(tl.minimum(dec_score_0 - lse_dec_t, 0.0)),
                        0.0,
                    )
                    tl.store(
                        G_ptr + pid_bh * stride_g_bh + token_idx * stride_g_t + sub_offsets_0 * stride_g_m,
                        g_t_0,
                        mask=token_valid & mask_m_sub_0,
                    )
                    a_t_0 = tl.where(token_valid & mask_m_sub_0, coeff_new_0, 0.0)
                    tl.store(
                        A_ptr + pid_bh * stride_a_bh + token_idx * stride_a_t + sub_offsets_0 * stride_a_m,
                        a_t_0,
                        mask=token_valid & mask_m_sub_0,
                    )
                    m_run_sub_0 = m_new_0
                    d_run_sub_0 = d_new_0
                    z_run_sub_0 = z_new_0

                    if NUM_REPLAY_TILES > 1:
                        s_t_1 = tl.sum(tl.where(select_j[None, :], s_block_1, 0.0), axis=1)
                        s_t_1 = tl.where(token_valid & mask_m_sub_1, s_t_1, -float("inf"))
                        m_new_1 = tl.maximum(m_run_sub_1, s_t_1)
                        alpha_1 = tl.math.exp2(m_run_sub_1 - m_new_1)
                        beta_1 = tl.math.exp2(s_t_1 - m_new_1)
                        d_new_1 = alpha_1 * d_run_sub_1 + beta_1
                        inv_d_1 = 1.0 / tl.maximum(d_new_1, 1e-20)
                        coeff_old_1 = alpha_1 * d_run_sub_1 * inv_d_1
                        coeff_new_1 = beta_1 * inv_d_1
                        z_new_1 = coeff_old_1[:, None] * z_run_sub_1 + coeff_new_1[:, None] * v_t[None, :]
                        p_part_1 = tl.sum(z_new_1 * u_t[None, :], axis=1)
                        p_ptr_1 = P_ptr + pid_bh * stride_p_bh + token_idx * stride_p_t + sub_offsets_1 * stride_p_m
                        if BLOCK_D < D_VALUE:
                            tl.atomic_add(p_ptr_1, p_part_1, mask=token_valid & mask_m_sub_1)
                        else:
                            tl.store(p_ptr_1, p_part_1, mask=token_valid & mask_m_sub_1)
                        if token_valid:
                            if WEIGHT_SHARING_ENC_DEC:
                                dec_score_1 = s_t_1
                            else:
                                dec_score_1 = _chunk_decode_score_vec(
                                    Q_dec_ptr,
                                    K_dec_ptr,
                                    stride_qdb, stride_qdt, stride_qdh, stride_qdd,
                                    stride_kdh, stride_kdm, stride_kdd,
                                    pid_b, pid_h, token_idx,
                                    sub_offsets_1, mask_m_sub_1,
                                    scale,
                                    D_SCORE=D_SCORE,
                                    BLOCK_M=BLOCK_M_REPLAY,
                                    BLOCK_K=BLOCK_K,
                                )
                        else:
                            dec_score_1 = tl.full([BLOCK_M_REPLAY], -float("inf"), tl.float32)
                        g_t_1 = tl.where(
                            token_valid & mask_m_sub_1,
                            tl.math.exp2(tl.minimum(dec_score_1 - lse_dec_t, 0.0)),
                            0.0,
                        )
                        tl.store(
                            G_ptr + pid_bh * stride_g_bh + token_idx * stride_g_t + sub_offsets_1 * stride_g_m,
                            g_t_1,
                            mask=token_valid & mask_m_sub_1,
                        )
                        a_t_1 = tl.where(token_valid & mask_m_sub_1, coeff_new_1, 0.0)
                        tl.store(
                            A_ptr + pid_bh * stride_a_bh + token_idx * stride_a_t + sub_offsets_1 * stride_a_m,
                            a_t_1,
                            mask=token_valid & mask_m_sub_1,
                        )
                        m_run_sub_1 = m_new_1
                        d_run_sub_1 = d_new_1
                        z_run_sub_1 = z_new_1
                    if NUM_REPLAY_TILES > 2:
                        s_t_2 = tl.sum(tl.where(select_j[None, :], s_block_2, 0.0), axis=1)
                        s_t_2 = tl.where(token_valid & mask_m_sub_2, s_t_2, -float("inf"))
                        m_new_2 = tl.maximum(m_run_sub_2, s_t_2)
                        alpha_2 = tl.math.exp2(m_run_sub_2 - m_new_2)
                        beta_2 = tl.math.exp2(s_t_2 - m_new_2)
                        d_new_2 = alpha_2 * d_run_sub_2 + beta_2
                        inv_d_2 = 1.0 / tl.maximum(d_new_2, 1e-20)
                        coeff_old_2 = alpha_2 * d_run_sub_2 * inv_d_2
                        coeff_new_2 = beta_2 * inv_d_2
                        z_new_2 = coeff_old_2[:, None] * z_run_sub_2 + coeff_new_2[:, None] * v_t[None, :]
                        p_part_2 = tl.sum(z_new_2 * u_t[None, :], axis=1)
                        p_ptr_2 = P_ptr + pid_bh * stride_p_bh + token_idx * stride_p_t + sub_offsets_2 * stride_p_m
                        if BLOCK_D < D_VALUE:
                            tl.atomic_add(p_ptr_2, p_part_2, mask=token_valid & mask_m_sub_2)
                        else:
                            tl.store(p_ptr_2, p_part_2, mask=token_valid & mask_m_sub_2)
                        if token_valid:
                            if WEIGHT_SHARING_ENC_DEC:
                                dec_score_2 = s_t_2
                            else:
                                dec_score_2 = _chunk_decode_score_vec(
                                    Q_dec_ptr,
                                    K_dec_ptr,
                                    stride_qdb, stride_qdt, stride_qdh, stride_qdd,
                                    stride_kdh, stride_kdm, stride_kdd,
                                    pid_b, pid_h, token_idx,
                                    sub_offsets_2, mask_m_sub_2,
                                    scale,
                                    D_SCORE=D_SCORE,
                                    BLOCK_M=BLOCK_M_REPLAY,
                                    BLOCK_K=BLOCK_K,
                                )
                        else:
                            dec_score_2 = tl.full([BLOCK_M_REPLAY], -float("inf"), tl.float32)
                        g_t_2 = tl.where(
                            token_valid & mask_m_sub_2,
                            tl.math.exp2(tl.minimum(dec_score_2 - lse_dec_t, 0.0)),
                            0.0,
                        )
                        tl.store(
                            G_ptr + pid_bh * stride_g_bh + token_idx * stride_g_t + sub_offsets_2 * stride_g_m,
                            g_t_2,
                            mask=token_valid & mask_m_sub_2,
                        )
                        a_t_2 = tl.where(token_valid & mask_m_sub_2, coeff_new_2, 0.0)
                        tl.store(
                            A_ptr + pid_bh * stride_a_bh + token_idx * stride_a_t + sub_offsets_2 * stride_a_m,
                            a_t_2,
                            mask=token_valid & mask_m_sub_2,
                        )
                        m_run_sub_2 = m_new_2
                        d_run_sub_2 = d_new_2
                        z_run_sub_2 = z_new_2
                    if NUM_REPLAY_TILES > 3:
                        s_t_3 = tl.sum(tl.where(select_j[None, :], s_block_3, 0.0), axis=1)
                        s_t_3 = tl.where(token_valid & mask_m_sub_3, s_t_3, -float("inf"))
                        m_new_3 = tl.maximum(m_run_sub_3, s_t_3)
                        alpha_3 = tl.math.exp2(m_run_sub_3 - m_new_3)
                        beta_3 = tl.math.exp2(s_t_3 - m_new_3)
                        d_new_3 = alpha_3 * d_run_sub_3 + beta_3
                        inv_d_3 = 1.0 / tl.maximum(d_new_3, 1e-20)
                        coeff_old_3 = alpha_3 * d_run_sub_3 * inv_d_3
                        coeff_new_3 = beta_3 * inv_d_3
                        z_new_3 = coeff_old_3[:, None] * z_run_sub_3 + coeff_new_3[:, None] * v_t[None, :]
                        p_part_3 = tl.sum(z_new_3 * u_t[None, :], axis=1)
                        p_ptr_3 = P_ptr + pid_bh * stride_p_bh + token_idx * stride_p_t + sub_offsets_3 * stride_p_m
                        if BLOCK_D < D_VALUE:
                            tl.atomic_add(p_ptr_3, p_part_3, mask=token_valid & mask_m_sub_3)
                        else:
                            tl.store(p_ptr_3, p_part_3, mask=token_valid & mask_m_sub_3)
                        if token_valid:
                            if WEIGHT_SHARING_ENC_DEC:
                                dec_score_3 = s_t_3
                            else:
                                dec_score_3 = _chunk_decode_score_vec(
                                    Q_dec_ptr,
                                    K_dec_ptr,
                                    stride_qdb, stride_qdt, stride_qdh, stride_qdd,
                                    stride_kdh, stride_kdm, stride_kdd,
                                    pid_b, pid_h, token_idx,
                                    sub_offsets_3, mask_m_sub_3,
                                    scale,
                                    D_SCORE=D_SCORE,
                                    BLOCK_M=BLOCK_M_REPLAY,
                                    BLOCK_K=BLOCK_K,
                                )
                        else:
                            dec_score_3 = tl.full([BLOCK_M_REPLAY], -float("inf"), tl.float32)
                        g_t_3 = tl.where(
                            token_valid & mask_m_sub_3,
                            tl.math.exp2(tl.minimum(dec_score_3 - lse_dec_t, 0.0)),
                            0.0,
                        )
                        tl.store(
                            G_ptr + pid_bh * stride_g_bh + token_idx * stride_g_t + sub_offsets_3 * stride_g_m,
                            g_t_3,
                            mask=token_valid & mask_m_sub_3,
                        )
                        a_t_3 = tl.where(token_valid & mask_m_sub_3, coeff_new_3, 0.0)
                        tl.store(
                            A_ptr + pid_bh * stride_a_bh + token_idx * stride_a_t + sub_offsets_3 * stride_a_m,
                            a_t_3,
                            mask=token_valid & mask_m_sub_3,
                        )
                        m_run_sub_3 = m_new_3
                        d_run_sub_3 = d_new_3
                        z_run_sub_3 = z_new_3

                t += BLOCK_T
        else:
            while t < chunk_end:
                s_t = tl.zeros([BLOCK_M], tl.float32)
                for d0 in tl.static_range(0, D_SCORE, BLOCK_K):
                    d_offsets_k = d0 + tl.arange(0, BLOCK_K)
                    mask_d_k = d_offsets_k < D_SCORE
                    q_tile = tl.load(
                        q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets_k[None, :] * stride_q_d,
                        mask=mask_m[:, None] & mask_d_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    k_tile = tl.load(
                        K_ptr + pid_b * stride_k_b + t * stride_k_n + pid_h * stride_k_h + d_offsets_k * stride_k_d,
                        mask=mask_d_k,
                        other=0.0,
                    ).to(tl.float32)
                    s_t += tl.sum(q_tile * k_tile[None, :], axis=1)
                s_t = tl.where(mask_m, s_t * score_scale, -float("inf"))

                m_new = tl.maximum(m_run, s_t)
                alpha = tl.math.exp2(m_run - m_new)
                beta = tl.math.exp2(s_t - m_new)
                d_new = alpha * d_run + beta
                inv_d = 1.0 / tl.maximum(d_new, 1e-20)
                coeff_old = alpha * d_run * inv_d
                coeff_new = beta * inv_d

                v_ptr = V_ptr + pid_b * stride_v_b + t * stride_v_n + pid_h * stride_v_h + d_offsets * stride_v_d
                v_t = tl.load(v_ptr, mask=mask_d, other=0.0).to(tl.float32)
                z_run = coeff_old[:, None] * z_run + coeff_new[:, None] * v_t[None, :]

                do_ptr = dO_ptr + pid_b * stride_do_b + t * stride_do_n + pid_h * stride_do_h + d_offsets * stride_do_d
                u_t = tl.load(do_ptr, mask=mask_d, other=0.0).to(tl.float32)
                p_part = tl.sum(z_run * u_t[None, :], axis=1)
                p_ptr = P_ptr + pid_bh * stride_p_bh + t * stride_p_t + m_offsets * stride_p_m
                if BLOCK_D < D_VALUE:
                    tl.atomic_add(p_ptr, p_part, mask=mask_m)
                else:
                    tl.store(p_ptr, p_part, mask=mask_m)
                lse_dec_t = tl.load(LSE_dec_ptr + pid_bh * stride_lsed_bh + t * stride_lsed_t).to(tl.float32)
                if WEIGHT_SHARING_ENC_DEC:
                    dec_score_t = s_t
                else:
                    dec_score_t = _chunk_decode_score_vec(
                        Q_dec_ptr,
                        K_dec_ptr,
                        stride_qdb, stride_qdt, stride_qdh, stride_qdd,
                        stride_kdh, stride_kdm, stride_kdd,
                        pid_b, pid_h, t,
                        m_offsets, mask_m,
                        scale,
                        D_SCORE=D_SCORE,
                        BLOCK_M=BLOCK_M,
                        BLOCK_K=BLOCK_K,
                    )
                g_t = tl.where(mask_m, tl.math.exp2(tl.minimum(dec_score_t - lse_dec_t, 0.0)), 0.0)
                tl.store(G_ptr + pid_bh * stride_g_bh + t * stride_g_t + m_offsets * stride_g_m, g_t, mask=mask_m)
                a_t = tl.where(mask_m, coeff_new, 0.0)
                tl.store(A_ptr + pid_bh * stride_a_bh + t * stride_a_t + m_offsets * stride_a_m, a_t, mask=mask_m)
                m_run = m_new
                d_run = d_new
                t += 1


@triton.jit
def flare_chunk_bwd_lse_gp_reduce(
    P_ptr,
    G_ptr,
    GP_ptr,
    stride_p_bh, stride_p_t, stride_p_m,
    stride_g_bh, stride_g_t, stride_g_m,
    stride_gp_bh, stride_gp_t,
    N,
    M,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    chunk_start = chunk_idx * CHUNK_SIZE

    t_local = tl.arange(0, BLOCK_T)
    m_local = tl.arange(0, BLOCK_M)
    t0 = 0
    while t0 < CHUNK_SIZE:
        t_offsets = chunk_start + t0 + t_local
        token_mask = t_offsets < N
        gp_acc = tl.zeros((BLOCK_T,), tl.float32)

        m_start = 0
        while m_start < M:
            m_offsets = m_start + m_local
            mask_m = m_offsets < M
            p_local = tl.load(
                P_ptr + pid_bh * stride_p_bh + t_offsets[:, None] * stride_p_t + m_offsets[None, :] * stride_p_m,
                mask=token_mask[:, None] & mask_m[None, :],
                other=0.0,
            ).to(tl.float32)
            g_local = tl.load(
                G_ptr + pid_bh * stride_g_bh + t_offsets[:, None] * stride_g_t + m_offsets[None, :] * stride_g_m,
                mask=token_mask[:, None] & mask_m[None, :],
                other=0.0,
            ).to(tl.float32)
            gp_acc += tl.sum(g_local * p_local, axis=1)
            m_start += BLOCK_M

        tl.store(
            GP_ptr + pid_bh * stride_gp_bh + t_offsets * stride_gp_t,
            gp_acc,
            mask=token_mask,
        )
        t0 += BLOCK_T


@triton.jit
def flare_chunk_bwd_lse_score_materialize(
    Q_ptr,
    K_ptr,
    Score_ptr,
    stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_n, stride_k_h, stride_k_d,
    stride_score_bh, stride_score_t, stride_score_m,
    H, M, N,
    scale,
    NUM_SCORE_T_TILES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_ct = tl.program_id(1)
    pid_m = tl.program_id(2)

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    chunk_idx = pid_ct // NUM_SCORE_T_TILES
    pid_t = pid_ct - chunk_idx * NUM_SCORE_T_TILES

    m_start = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    chunk_start = chunk_idx * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, N)
    t_start = chunk_start + pid_t * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    token_mask = t_offsets < chunk_end

    q_base_ptr = Q_ptr + pid_h * stride_q_h
    s_block = tl.zeros((BLOCK_M, BLOCK_T), dtype=tl.float32)

    for d0 in tl.static_range(0, D, BLOCK_K):
        d_offsets = d0 + tl.arange(0, BLOCK_K)
        mask_d = d_offsets < D
        q_tile = tl.load(
            q_base_ptr + m_offsets[:, None] * stride_q_m + d_offsets[None, :] * stride_q_d,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        k_tile = tl.load(
            K_ptr
            + pid_b * stride_k_b
            + t_offsets[:, None] * stride_k_n
            + pid_h * stride_k_h
            + d_offsets[None, :] * stride_k_d,
            mask=token_mask[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        s_block += tl.dot(q_tile, tl.trans(k_tile), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    tl.store(
        Score_ptr
        + pid_bh * stride_score_bh
        + t_offsets[:, None] * stride_score_t
        + m_offsets[None, :] * stride_score_m,
        tl.trans(s_block * scale),
        mask=token_mask[:, None] & mask_m[None, :],
    )


@triton.jit
def flare_chunk_bwd_lse_p_part_from_scores(
    Score_ptr,
    V_ptr,
    ZPrefixUnnorm_ptr,
    LSE_ptr,
    LSE_M_ptr,
    dO_ptr,
    P_ptr,
    GP_ptr,
    A_ptr,
    stride_score_bh, stride_score_t, stride_score_m,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_zpun_bh, stride_zpun_chunk, stride_zpun_m, stride_zpun_d,
    stride_lse_b, stride_lse_h, stride_lse_t, stride_lse_m,
    stride_lsem_bh, stride_lsem_t,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_p_bh, stride_p_t, stride_p_m,
    stride_gp_bh, stride_gp_t,
    stride_a_bh, stride_a_t, stride_a_m,
    H, M, N,
    NUM_M_TILES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_cm = tl.program_id(1)
    pid_d = tl.program_id(2)

    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    chunk_idx = pid_cm // NUM_M_TILES
    pid_m = pid_cm - chunk_idx * NUM_M_TILES
    m_start = pid_m * BLOCK_M
    chunk_start = chunk_idx * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, N)
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = m_offsets < M
    mask_d = d_offsets < D

    z_prefix_unnorm = tl.load(
        ZPrefixUnnorm_ptr
        + pid_bh * stride_zpun_bh
        + chunk_idx * stride_zpun_chunk
        + m_offsets[:, None] * stride_zpun_m
        + d_offsets[None, :] * stride_zpun_d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    mask_prefix = mask_m & (chunk_start > 0)
    lse_init = tl.load(
        LSE_ptr
        + pid_b * stride_lse_b
        + pid_h * stride_lse_h
        + (chunk_start - 1) * stride_lse_t
        + m_offsets * stride_lse_m,
        mask=mask_prefix,
        other=-float("inf"),
    ).to(tl.float32)
    lse_run = lse_init
    z_run = tl.where(mask_prefix[:, None], z_prefix_unnorm * tl.exp(-lse_init[:, None]), 0.0)

    t = chunk_start
    while t < chunk_end:
        s_t = tl.load(
            Score_ptr + pid_bh * stride_score_bh + t * stride_score_t + m_offsets * stride_score_m,
            mask=mask_m,
            other=-float("inf"),
        ).to(tl.float32)

        lse_m = tl.maximum(lse_run, s_t)
        lse_new = lse_m + tl.log(tl.exp(lse_run - lse_m) + tl.exp(s_t - lse_m))
        coeff_old = tl.exp(lse_run - lse_new)
        coeff_new = tl.exp(s_t - lse_new)

        v_t = tl.load(
            V_ptr + pid_b * stride_v_b + t * stride_v_n + pid_h * stride_v_h + d_offsets * stride_v_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        z_run = coeff_old[:, None] * z_run + coeff_new[:, None] * v_t[None, :]

        u_t = tl.load(
            dO_ptr + pid_b * stride_do_b + t * stride_do_n + pid_h * stride_do_h + d_offsets * stride_do_d,
            mask=mask_d,
            other=0.0,
        ).to(tl.float32)
        p_part = tl.sum(z_run * u_t[None, :], axis=1)
        p_ptr = P_ptr + pid_bh * stride_p_bh + t * stride_p_t + m_offsets * stride_p_m
        if BLOCK_D < D:
            tl.atomic_add(p_ptr, p_part, mask=mask_m)
        else:
            tl.store(p_ptr, p_part, mask=mask_m)

        lse_m_t = tl.load(LSE_M_ptr + pid_bh * stride_lsem_bh + t * stride_lsem_t).to(tl.float32)
        g_t = tl.where(mask_m, tl.exp(tl.minimum(s_t - lse_m_t, 0.0)), 0.0)
        lse_t = tl.load(
            LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_t + m_offsets * stride_lse_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        a_t = tl.where(mask_m, tl.exp(tl.minimum(s_t - lse_t, 0.0)), 0.0)
        tl.store(Score_ptr + pid_bh * stride_score_bh + t * stride_score_t + m_offsets * stride_score_m, g_t, mask=mask_m)
        tl.store(A_ptr + pid_bh * stride_a_bh + t * stride_a_t + m_offsets * stride_a_m, a_t, mask=mask_m)

        gp_part = tl.sum(g_t * p_part, axis=0)
        gp_ptr = GP_ptr + pid_bh * stride_gp_bh + t * stride_gp_t
        if BLOCK_D < D:
            tl.atomic_add(gp_ptr, gp_part)
        else:
            tl.store(gp_ptr, gp_part)

        lse_run = lse_new
        t += 1


@triton.jit
def flare_chunk_bwd_lse_chunk_summary(
    LSE_ptr,
    dO_ptr,
    P_ptr,
    G_ptr,
    BLocal_ptr,
    ALocal_ptr,
    Scale_ptr,
    stride_lse_b, stride_lse_h, stride_lse_n, stride_lse_m,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_p_bh, stride_p_t, stride_p_m,
    stride_g_bh, stride_g_t, stride_g_m,
    stride_blocal_bh, stride_blocal_chunk, stride_blocal_tile, stride_blocal_m, stride_blocal_d,
    stride_alocal_bh, stride_alocal_chunk, stride_alocal_tile, stride_alocal_m,
    stride_scale_bh, stride_scale_chunk, stride_scale_tile, stride_scale_m,
    H, M, N,
    NUM_DV_TILES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    USE_BLOCKED_SUFFIX_SUMMARY: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    pid_md = tl.program_id(2)
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    pid_m = pid_md // NUM_DV_TILES
    pid_dv = pid_md - pid_m * NUM_DV_TILES
    m_start = pid_m * BLOCK_M
    chunk_start = chunk_idx * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, N)
    m_local = tl.arange(0, BLOCK_M)
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    d_offsets = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    store_shared = pid_dv == 0

    b_carry = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)
    a_carry = tl.zeros([BLOCK_M], tl.float32)

    if USE_BLOCKED_SUFFIX_SUMMARY:
        t_local = tl.arange(0, BLOCK_T)

        t = chunk_end
        while t > chunk_start:
            t_start = t - BLOCK_T
            t_offsets = t_start + t_local
            token_valid = (t_offsets >= chunk_start) & (t_offsets < t)
            next_offsets = t_offsets + 1
            next_valid = next_offsets < chunk_end

            g_block = tl.load(
                G_ptr + pid_bh * stride_g_bh + t_offsets[None, :] * stride_g_t + m_offsets[:, None] * stride_g_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            p_block = tl.load(
                P_ptr + pid_bh * stride_p_bh + t_offsets[None, :] * stride_p_t + m_offsets[:, None] * stride_p_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            u_block = tl.load(
                dO_ptr + pid_b * stride_do_b + t_offsets[:, None] * stride_do_n + pid_h * stride_do_h + d_offsets[None, :] * stride_do_d,
                mask=token_valid[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            lse_curr = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t_offsets[None, :] * stride_lse_n + m_offsets[:, None] * stride_lse_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            select_first = t_local == 0
            lse_start = tl.sum(tl.where(select_first[None, :], lse_curr, 0.0), axis=1)
            # For a reverse microblock [t_start, t), the token weight carried
            # back to the block start is
            #   exp(L[t_start] - L[token]) * g[token]
            # This is exactly the telescoped product of the per-token
            # rescaling factors, but it lets us form the whole block weight
            # vector in one broadcast instead of a token loop.
            weights = tl.where(
                mask_m[:, None] & token_valid[None, :],
                tl.math.exp2(tl.minimum(lse_start[:, None] - lse_curr, 0.0)) * g_block,
                0.0,
            )
            if t < chunk_end:
                lse_boundary = tl.load(
                    LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                    mask=mask_m,
                    other=0.0,
                ).to(tl.float32)
                scale_prefix = tl.math.exp2(tl.minimum(lse_start - lse_boundary, 0.0))
            else:
                # The last microblock in the chunk has no intra-chunk future
                # carry. The incoming local carry is zero on that iteration, so
                # any scale would be equivalent; use zero to reflect that the
                # block is self-contained.
                scale_prefix = tl.zeros([BLOCK_M], tl.float32)

            b_block = tl.dot(weights, u_block, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            a_block = tl.sum(weights * p_block, axis=1)
            b_carry = b_block + scale_prefix[:, None] * b_carry
            a_carry = a_block + scale_prefix * a_carry
            t = t_start
    else:
        t = chunk_end
        while t > chunk_start:
            t -= 1

            lse_ptr = LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m
            lse_curr = tl.load(lse_ptr, mask=mask_m, other=0.0).to(tl.float32)

            if (t + 1) < chunk_end:
                lse_next_ptr = LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + (t + 1) * stride_lse_n + m_offsets * stride_lse_m
                lse_next = tl.load(lse_next_ptr, mask=mask_m, other=0.0).to(tl.float32)
                carry_scale = tl.math.exp2(tl.minimum(lse_curr - lse_next, 0.0))
                b_carry = carry_scale[:, None] * b_carry
                a_carry = carry_scale * a_carry

            p_local = tl.load(
                P_ptr + pid_bh * stride_p_bh + t * stride_p_t + m_offsets * stride_p_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            g_t = tl.load(
                G_ptr + pid_bh * stride_g_bh + t * stride_g_t + m_offsets * stride_g_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)

            do_ptr = dO_ptr + pid_b * stride_do_b + t * stride_do_n + pid_h * stride_do_h + d_offsets * stride_do_d
            u_t = tl.load(do_ptr, mask=mask_d, other=0.0).to(tl.float32)
            b_carry = b_carry + g_t[:, None] * u_t[None, :]
            a_carry = a_carry + g_t * p_local

    tl.store(
        BLocal_ptr
        + pid_bh * stride_blocal_bh
        + chunk_idx * stride_blocal_chunk
        + pid_m * stride_blocal_tile
        + m_local[:, None] * stride_blocal_m
        + d_offsets[None, :] * stride_blocal_d,
        b_carry,
        mask=mask_m[:, None] & mask_d[None, :],
    )
    tl.store(
        ALocal_ptr + pid_bh * stride_alocal_bh + chunk_idx * stride_alocal_chunk + pid_m * stride_alocal_tile + m_local * stride_alocal_m,
        a_carry,
        mask=store_shared & mask_m,
    )
    next_start = chunk_start + CHUNK_SIZE
    if next_start < N:
        lse_start = tl.load(
            LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + chunk_start * stride_lse_n + m_offsets * stride_lse_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        lse_next = tl.load(
            LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + next_start * stride_lse_n + m_offsets * stride_lse_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        scale_out = tl.math.exp2(tl.minimum(lse_start - lse_next, 0.0))
    else:
        scale_out = tl.zeros([BLOCK_M], tl.float32)
    tl.store(
        Scale_ptr + pid_bh * stride_scale_bh + chunk_idx * stride_scale_chunk + pid_m * stride_scale_tile + m_local * stride_scale_m,
        scale_out,
        mask=store_shared & mask_m,
    )


@triton.jit
def flare_chunk_bwd_lse_carry_scan(
    BLocal_ptr,
    ALocal_ptr,
    Scale_ptr,
    BIn_ptr,
    AIn_ptr,
    stride_blocal_bh, stride_blocal_chunk, stride_blocal_tile, stride_blocal_m, stride_blocal_d,
    stride_alocal_bh, stride_alocal_chunk, stride_alocal_tile, stride_alocal_m,
    stride_scale_bh, stride_scale_chunk, stride_scale_tile, stride_scale_m,
    stride_bin_bh, stride_bin_chunk, stride_bin_tile, stride_bin_m, stride_bin_d,
    stride_ain_bh, stride_ain_chunk, stride_ain_tile, stride_ain_m,
    M,
    NUM_CHUNKS,
    NUM_DV_TILES: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_md = tl.program_id(1)

    pid_m = pid_md // NUM_DV_TILES
    pid_dv = pid_md - pid_m * NUM_DV_TILES
    m_start = pid_m * BLOCK_M
    m_local = tl.arange(0, BLOCK_M)
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    d_offsets = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE
    store_shared = pid_dv == 0

    future_b = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)
    future_a = tl.zeros([BLOCK_M], tl.float32)

    chunk_idx = NUM_CHUNKS
    while chunk_idx > 0:
        chunk_idx -= 1

        tl.store(
            BIn_ptr
            + pid_bh * stride_bin_bh
            + chunk_idx * stride_bin_chunk
            + pid_m * stride_bin_tile
            + m_local[:, None] * stride_bin_m
            + d_offsets[None, :] * stride_bin_d,
            future_b,
            mask=mask_m[:, None] & mask_d[None, :],
        )
        tl.store(
            AIn_ptr + pid_bh * stride_ain_bh + chunk_idx * stride_ain_chunk + pid_m * stride_ain_tile + m_local * stride_ain_m,
            future_a,
            mask=store_shared & mask_m,
        )

        scale_chunk = tl.load(
            Scale_ptr + pid_bh * stride_scale_bh + chunk_idx * stride_scale_chunk + pid_m * stride_scale_tile + m_local * stride_scale_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)
        b_local = tl.load(
            BLocal_ptr
            + pid_bh * stride_blocal_bh
            + chunk_idx * stride_blocal_chunk
            + pid_m * stride_blocal_tile
            + m_local[:, None] * stride_blocal_m
            + d_offsets[None, :] * stride_blocal_d,
            mask=mask_m[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        a_local = tl.load(
            ALocal_ptr + pid_bh * stride_alocal_bh + chunk_idx * stride_alocal_chunk + pid_m * stride_alocal_tile + m_local * stride_alocal_m,
            mask=mask_m,
            other=0.0,
        ).to(tl.float32)

        future_b = b_local + scale_chunk[:, None] * future_b
        future_a = a_local + scale_chunk * future_a


@triton.jit
def flare_chunk_bwd_lse_chunk_apply_part(
    V_ptr,
    LSE_ptr,
    dO_ptr,
    G_ptr,
    A_ptr,
    BIn_ptr,
    VBPart_ptr,
    dVPart_ptr,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_lse_b, stride_lse_h, stride_lse_n, stride_lse_m,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_g_bh, stride_g_t, stride_g_m,
    stride_a_bh, stride_a_t, stride_a_m,
    stride_bin_bh, stride_bin_chunk, stride_bin_tile, stride_bin_m, stride_bin_d,
    stride_vbp_bh, stride_vbp_chunk, stride_vbp_tile, stride_vbp_dv, stride_vbp_t, stride_vbp_m,
    stride_dvp_bh, stride_dvp_chunk, stride_dvp_tile, stride_dvp_t, stride_dvp_d,
    H, M, N,
    NUM_DV_TILES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SCALAR_PANEL_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    USE_BLOCKED_APPLY: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    pid_md = tl.program_id(2)
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    pid_m = pid_md // NUM_DV_TILES
    pid_dv = pid_md - pid_m * NUM_DV_TILES
    m_start = pid_m * BLOCK_M
    chunk_start = chunk_idx * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, N)
    m_local = tl.arange(0, BLOCK_M)
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    d_offsets = pid_dv * BLOCK_DV + tl.arange(0, BLOCK_DV)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE

    b_carry = tl.load(
        BIn_ptr
        + pid_bh * stride_bin_bh
        + chunk_idx * stride_bin_chunk
        + pid_m * stride_bin_tile
        + m_local[:, None] * stride_bin_m
        + d_offsets[None, :] * stride_bin_d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)

    if USE_BLOCKED_APPLY:
        t_local = tl.arange(0, BLOCK_T)
        t = chunk_end
        while t > chunk_start:
            t_start = tl.maximum(t - BLOCK_T, chunk_start)
            t_offsets = t_start + t_local
            token_valid = t_offsets < t

            g_block = tl.load(
                G_ptr + pid_bh * stride_g_bh + t_offsets[None, :] * stride_g_t + m_offsets[:, None] * stride_g_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            a_block = tl.load(
                A_ptr + pid_bh * stride_a_bh + t_offsets[None, :] * stride_a_t + m_offsets[:, None] * stride_a_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            lse_block = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t_offsets[None, :] * stride_lse_n + m_offsets[:, None] * stride_lse_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            u_block = tl.load(
                dO_ptr + pid_b * stride_do_b + t_offsets[:, None] * stride_do_n + pid_h * stride_do_h + d_offsets[None, :] * stride_do_d,
                mask=token_valid[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            v_block = tl.load(
                V_ptr + pid_b * stride_v_b + t_offsets[:, None] * stride_v_n + pid_h * stride_v_h + d_offsets[None, :] * stride_v_d,
                mask=token_valid[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)

            b_boundary = b_carry
            lse_boundary = tl.zeros([BLOCK_M], tl.float32)
            if t < N:
                lse_boundary = tl.load(
                    LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                    mask=mask_m,
                    other=0.0,
                ).to(tl.float32)

            for j in tl.range(0, BLOCK_T):
                token_idx = t_start + j
                token_is_valid = token_idx < t
                select_j = t_local == j

                lse_j = tl.sum(tl.where(select_j[None, :], lse_block, 0.0), axis=1)
                a_j = tl.sum(tl.where(select_j[None, :], a_block, 0.0), axis=1)
                v_j = tl.sum(tl.where(select_j[:, None], v_block, 0.0), axis=0)

                if t < N:
                    scale_j = tl.math.exp2(tl.minimum(lse_j - lse_boundary, 0.0))
                else:
                    scale_j = tl.zeros([BLOCK_M], tl.float32)

                weight_j = tl.math.exp2(tl.minimum(lse_j[:, None] - lse_block, 0.0))
                weight_j = tl.where(mask_m[:, None] & token_valid[None, :], weight_j * g_block, 0.0)
                weight_j = tl.where(t_local[None, :] >= j, weight_j, 0.0)

                b_j = tl.dot(weight_j, u_block, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                b_j = b_j + scale_j[:, None] * b_boundary
                dV_j = tl.sum(a_j[:, None] * b_j, axis=0)
                vb_j = tl.sum(b_j * v_j[None, :], axis=1)

                tl.store(
                    VBPart_ptr
                    + pid_bh * stride_vbp_bh
                    + chunk_idx * stride_vbp_chunk
                    + pid_m * stride_vbp_tile
                    + pid_dv * stride_vbp_dv
                    + (token_idx - chunk_start) * stride_vbp_t
                    + m_local * stride_vbp_m,
                    vb_j,
                    mask=token_is_valid & mask_m,
                )
                tl.store(
                    dVPart_ptr
                    + pid_bh * stride_dvp_bh
                    + chunk_idx * stride_dvp_chunk
                    + pid_m * stride_dvp_tile
                    + (token_idx - chunk_start) * stride_dvp_t
                    + d_offsets * stride_dvp_d,
                    dV_j,
                    mask=token_is_valid & mask_d,
                )

            select_first = t_local == 0
            lse_first = tl.sum(tl.where(select_first[None, :], lse_block, 0.0), axis=1)
            if t < N:
                scale_first = tl.math.exp2(tl.minimum(lse_first - lse_boundary, 0.0))
            else:
                scale_first = tl.zeros([BLOCK_M], tl.float32)
            weight_first = tl.math.exp2(tl.minimum(lse_first[:, None] - lse_block, 0.0))
            weight_first = tl.where(mask_m[:, None] & token_valid[None, :], weight_first * g_block, 0.0)
            b_carry = tl.dot(weight_first, u_block, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            b_carry = b_carry + scale_first[:, None] * b_boundary
            t = t_start
    else:
        t = chunk_end
        lse_next = tl.zeros([BLOCK_M], tl.float32)
        if t < N:
            lse_next = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
        while t > chunk_start:
            t -= 1
            lse_curr = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)

            if t + 1 < N:
                carry_scale = tl.math.exp2(tl.minimum(lse_curr - lse_next, 0.0))
                b_carry = carry_scale[:, None] * b_carry

            g_t = tl.load(
                G_ptr + pid_bh * stride_g_bh + t * stride_g_t + m_offsets * stride_g_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            a_self = tl.load(
                A_ptr + pid_bh * stride_a_bh + t * stride_a_t + m_offsets * stride_a_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            u_t = tl.load(
                dO_ptr + pid_b * stride_do_b + t * stride_do_n + pid_h * stride_do_h + d_offsets * stride_do_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            b_carry = b_carry + g_t[:, None] * u_t[None, :]

            v_t = tl.load(
                V_ptr + pid_b * stride_v_b + t * stride_v_n + pid_h * stride_v_h + d_offsets * stride_v_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            dV_acc = tl.sum(a_self[:, None] * b_carry, axis=0)
            vb_part = tl.sum(b_carry * v_t[None, :], axis=1)

            t_local_scalar = t - chunk_start
            tl.store(
                VBPart_ptr
                + pid_bh * stride_vbp_bh
                + chunk_idx * stride_vbp_chunk
                + pid_m * stride_vbp_tile
                + pid_dv * stride_vbp_dv
                + t_local_scalar * stride_vbp_t
                + m_local * stride_vbp_m,
                vb_part,
                mask=mask_m,
            )
            tl.store(
                dVPart_ptr
                + pid_bh * stride_dvp_bh
                + chunk_idx * stride_dvp_chunk
                + pid_m * stride_dvp_tile
                + t_local_scalar * stride_dvp_t
                + d_offsets * stride_dvp_d,
                dV_acc,
                mask=mask_d,
            )
            lse_next = lse_curr


@triton.jit
def flare_chunk_bwd_lse_chunk_apply_finalize(
    LSE_ptr,
    P_ptr,
    G_ptr,
    GP_ptr,
    A_ptr,
    AIn_ptr,
    VBPart_ptr,
    DS_enc_ptr,
    DS_dec_ptr,
    stride_lse_b, stride_lse_h, stride_lse_n, stride_lse_m,
    stride_p_bh, stride_p_t, stride_p_m,
    stride_g_bh, stride_g_t, stride_g_m,
    stride_gp_bh, stride_gp_t,
    stride_a_bh, stride_a_t, stride_a_m,
    stride_ain_bh, stride_ain_chunk, stride_ain_tile, stride_ain_m,
    stride_vbp_bh, stride_vbp_chunk, stride_vbp_tile, stride_vbp_dv, stride_vbp_t, stride_vbp_m,
    stride_dse_bh, stride_dse_chunk, stride_dse_t, stride_dse_m,
    stride_dsd_bh, stride_dsd_chunk, stride_dsd_t, stride_dsd_m,
    H, M, N,
    NUM_DV_TILES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SCALAR_PANEL_T: tl.constexpr,
    USE_BLOCKED_APPLY: tl.constexpr,
    WEIGHT_SHARING_ENC_DEC: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    m_start = pid_m * BLOCK_M
    chunk_start = chunk_idx * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, N)
    m_local = tl.arange(0, BLOCK_M)
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < M

    a_carry = tl.load(
        AIn_ptr + pid_bh * stride_ain_bh + chunk_idx * stride_ain_chunk + pid_m * stride_ain_tile + m_local * stride_ain_m,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)

    if USE_BLOCKED_APPLY:
        t_local = tl.arange(0, BLOCK_T)
        t = chunk_end
        while t > chunk_start:
            t_start = tl.maximum(t - BLOCK_T, chunk_start)
            t_offsets = t_start + t_local
            token_valid = t_offsets < t

            p_block = tl.load(
                P_ptr + pid_bh * stride_p_bh + t_offsets[None, :] * stride_p_t + m_offsets[:, None] * stride_p_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            g_block = tl.load(
                G_ptr + pid_bh * stride_g_bh + t_offsets[None, :] * stride_g_t + m_offsets[:, None] * stride_g_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            a_block = tl.load(
                A_ptr + pid_bh * stride_a_bh + t_offsets[None, :] * stride_a_t + m_offsets[:, None] * stride_a_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            lse_block = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t_offsets[None, :] * stride_lse_n + m_offsets[:, None] * stride_lse_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            gp_block = tl.load(
                GP_ptr + pid_bh * stride_gp_bh + t_offsets * stride_gp_t,
                mask=token_valid,
                other=0.0,
            ).to(tl.float32)

            vb_block = tl.zeros((BLOCK_T, BLOCK_M), tl.float32)
            dv_tile = 0
            while dv_tile < NUM_DV_TILES:
                vb_part = tl.load(
                    VBPart_ptr
                    + pid_bh * stride_vbp_bh
                    + chunk_idx * stride_vbp_chunk
                    + pid_m * stride_vbp_tile
                    + dv_tile * stride_vbp_dv
                    + t_offsets[:, None] * stride_vbp_t
                    + m_local[None, :] * stride_vbp_m,
                    mask=token_valid[:, None] & mask_m[None, :],
                    other=0.0,
                ).to(tl.float32)
                vb_block += vb_part
                dv_tile += 1

            a_boundary = a_carry
            lse_boundary = tl.zeros([BLOCK_M], tl.float32)
            if t < N:
                lse_boundary = tl.load(
                    LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                    mask=mask_m,
                    other=0.0,
                ).to(tl.float32)

            for j in tl.range(0, BLOCK_T):
                token_idx = t_start + j
                token_is_valid = token_idx < t
                select_j = t_local == j

                lse_j = tl.sum(tl.where(select_j[None, :], lse_block, 0.0), axis=1)
                p_j = tl.sum(tl.where(select_j[None, :], p_block, 0.0), axis=1)
                g_j = tl.sum(tl.where(select_j[None, :], g_block, 0.0), axis=1)
                a_j = tl.sum(tl.where(select_j[None, :], a_block, 0.0), axis=1)
                gp_j = tl.sum(tl.where(select_j, gp_block, 0.0), axis=0)
                vb_j = tl.sum(tl.where(select_j[:, None], vb_block, 0.0), axis=0)

                if t < N:
                    scale_j = tl.math.exp2(tl.minimum(lse_j - lse_boundary, 0.0))
                else:
                    scale_j = tl.zeros([BLOCK_M], tl.float32)

                weight_j = tl.math.exp2(tl.minimum(lse_j[:, None] - lse_block, 0.0))
                weight_j = tl.where(mask_m[:, None] & token_valid[None, :], weight_j * g_block, 0.0)
                weight_j = tl.where(t_local[None, :] >= j, weight_j, 0.0)

                a_j_carry = tl.sum(weight_j * p_block, axis=1) + scale_j * a_boundary
                ds_g_j = g_j * (p_j - gp_j)
                ds_z_j = a_j * (vb_j - a_j_carry)
                if WEIGHT_SHARING_ENC_DEC:
                    tl.store(
                        DS_enc_ptr
                        + pid_bh * stride_dse_bh
                        + chunk_idx * stride_dse_chunk
                        + (token_idx - chunk_start) * stride_dse_t
                        + m_offsets * stride_dse_m,
                        ds_g_j + ds_z_j,
                        mask=token_is_valid & mask_m,
                    )
                else:
                    tl.store(
                        DS_enc_ptr
                        + pid_bh * stride_dse_bh
                        + chunk_idx * stride_dse_chunk
                        + (token_idx - chunk_start) * stride_dse_t
                        + m_offsets * stride_dse_m,
                        ds_z_j,
                        mask=token_is_valid & mask_m,
                    )
                    tl.store(
                        DS_dec_ptr
                        + pid_bh * stride_dsd_bh
                        + chunk_idx * stride_dsd_chunk
                        + (token_idx - chunk_start) * stride_dsd_t
                        + m_offsets * stride_dsd_m,
                        ds_g_j,
                        mask=token_is_valid & mask_m,
                    )

            select_first = t_local == 0
            lse_first = tl.sum(tl.where(select_first[None, :], lse_block, 0.0), axis=1)
            if t < N:
                scale_first = tl.math.exp2(tl.minimum(lse_first - lse_boundary, 0.0))
            else:
                scale_first = tl.zeros([BLOCK_M], tl.float32)
            weight_first = tl.math.exp2(tl.minimum(lse_first[:, None] - lse_block, 0.0))
            weight_first = tl.where(mask_m[:, None] & token_valid[None, :], weight_first * g_block, 0.0)
            a_carry = tl.sum(weight_first * p_block, axis=1) + scale_first * a_boundary
            t = t_start
    else:
        t = chunk_end
        lse_next = tl.zeros([BLOCK_M], tl.float32)
        if t < N:
            lse_next = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
        while t > chunk_start:
            t -= 1
            lse_curr = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)

            if t + 1 < N:
                carry_scale = tl.math.exp2(tl.minimum(lse_curr - lse_next, 0.0))
                a_carry = carry_scale * a_carry

            p_local = tl.load(
                P_ptr + pid_bh * stride_p_bh + t * stride_p_t + m_offsets * stride_p_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            g_t = tl.load(
                G_ptr + pid_bh * stride_g_bh + t * stride_g_t + m_offsets * stride_g_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            a_self = tl.load(
                A_ptr + pid_bh * stride_a_bh + t * stride_a_t + m_offsets * stride_a_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            gp_t = tl.load(
                GP_ptr + pid_bh * stride_gp_bh + t * stride_gp_t,
            ).to(tl.float32)
            ds_g = g_t * (p_local - gp_t)

            a_carry = a_carry + g_t * p_local

            vb_dot = tl.zeros([BLOCK_M], tl.float32)
            dv_tile = 0
            while dv_tile < NUM_DV_TILES:
                vb_part = tl.load(
                    VBPart_ptr
                    + pid_bh * stride_vbp_bh
                    + chunk_idx * stride_vbp_chunk
                    + pid_m * stride_vbp_tile
                    + dv_tile * stride_vbp_dv
                    + (t - chunk_start) * stride_vbp_t
                    + m_local * stride_vbp_m,
                    mask=mask_m,
                    other=0.0,
                ).to(tl.float32)
                vb_dot += vb_part
                dv_tile += 1

            ds_z = a_self * (vb_dot - a_carry)
            if WEIGHT_SHARING_ENC_DEC:
                tl.store(
                    DS_enc_ptr
                    + pid_bh * stride_dse_bh
                    + chunk_idx * stride_dse_chunk
                    + (t - chunk_start) * stride_dse_t
                    + m_offsets * stride_dse_m,
                    ds_g + ds_z,
                    mask=mask_m,
                )
            else:
                tl.store(
                    DS_enc_ptr
                    + pid_bh * stride_dse_bh
                    + chunk_idx * stride_dse_chunk
                    + (t - chunk_start) * stride_dse_t
                    + m_offsets * stride_dse_m,
                    ds_z,
                    mask=mask_m,
                )
                tl.store(
                    DS_dec_ptr
                    + pid_bh * stride_dsd_bh
                    + chunk_idx * stride_dsd_chunk
                    + (t - chunk_start) * stride_dsd_t
                    + m_offsets * stride_dsd_m,
                    ds_g,
                    mask=mask_m,
                )
            lse_next = lse_curr


@triton.jit
def flare_chunk_bwd_lse_chunk_apply_fused_single_dv(
    V_ptr,
    LSE_ptr,
    dO_ptr,
    P_ptr,
    G_ptr,
    GP_ptr,
    A_ptr,
    BIn_ptr,
    AIn_ptr,
    DS_enc_ptr,
    DS_dec_ptr,
    dVPart_ptr,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_lse_b, stride_lse_h, stride_lse_n, stride_lse_m,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_p_bh, stride_p_t, stride_p_m,
    stride_g_bh, stride_g_t, stride_g_m,
    stride_gp_bh, stride_gp_t,
    stride_a_bh, stride_a_t, stride_a_m,
    stride_bin_bh, stride_bin_chunk, stride_bin_tile, stride_bin_m, stride_bin_d,
    stride_ain_bh, stride_ain_chunk, stride_ain_tile, stride_ain_m,
    stride_dse_bh, stride_dse_chunk, stride_dse_t, stride_dse_m,
    stride_dsd_bh, stride_dsd_chunk, stride_dsd_t, stride_dsd_m,
    stride_dvp_bh, stride_dvp_chunk, stride_dvp_tile, stride_dvp_t, stride_dvp_d,
    H, M, N,
    CHUNK_SIZE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_T: tl.constexpr,
    SCALAR_PANEL_T: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    USE_BLOCKED_APPLY: tl.constexpr,
    USE_SCALAR_PANEL_APPLY: tl.constexpr,
    WEIGHT_SHARING_ENC_DEC: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    pid_m = tl.program_id(2)
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    m_start = pid_m * BLOCK_M
    chunk_start = chunk_idx * CHUNK_SIZE
    chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, N)
    m_local = tl.arange(0, BLOCK_M)
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_DV)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE

    b_carry = tl.load(
        BIn_ptr
        + pid_bh * stride_bin_bh
        + chunk_idx * stride_bin_chunk
        + pid_m * stride_bin_tile
        + m_local[:, None] * stride_bin_m
        + d_offsets[None, :] * stride_bin_d,
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.float32)
    a_carry = tl.load(
        AIn_ptr + pid_bh * stride_ain_bh + chunk_idx * stride_ain_chunk + pid_m * stride_ain_tile + m_local * stride_ain_m,
        mask=mask_m,
        other=0.0,
    ).to(tl.float32)

    if USE_BLOCKED_APPLY:
        t_local = tl.arange(0, BLOCK_T)
        t = chunk_end
        while t > chunk_start:
            t_start = tl.maximum(t - BLOCK_T, chunk_start)
            t_offsets = t_start + t_local
            token_valid = t_offsets < t

            p_block = tl.load(
                P_ptr + pid_bh * stride_p_bh + t_offsets[None, :] * stride_p_t + m_offsets[:, None] * stride_p_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            g_block = tl.load(
                G_ptr + pid_bh * stride_g_bh + t_offsets[None, :] * stride_g_t + m_offsets[:, None] * stride_g_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            a_block = tl.load(
                A_ptr + pid_bh * stride_a_bh + t_offsets[None, :] * stride_a_t + m_offsets[:, None] * stride_a_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            lse_block = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t_offsets[None, :] * stride_lse_n + m_offsets[:, None] * stride_lse_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            u_block = tl.load(
                dO_ptr + pid_b * stride_do_b + t_offsets[:, None] * stride_do_n + pid_h * stride_do_h + d_offsets[None, :] * stride_do_d,
                mask=token_valid[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            v_block = tl.load(
                V_ptr + pid_b * stride_v_b + t_offsets[:, None] * stride_v_n + pid_h * stride_v_h + d_offsets[None, :] * stride_v_d,
                mask=token_valid[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            gp_block = tl.load(
                GP_ptr + pid_bh * stride_gp_bh + t_offsets * stride_gp_t,
                mask=token_valid,
                other=0.0,
            ).to(tl.float32)

            b_boundary = b_carry
            a_boundary = a_carry

            select_first = t_local == 0
            lse_first = tl.sum(tl.where(select_first[None, :], lse_block, 0.0), axis=1)
            g_first = tl.sum(tl.where(select_first[None, :], g_block, 0.0), axis=1)
            a_first = tl.sum(tl.where(select_first[None, :], a_block, 0.0), axis=1)
            p_first = tl.sum(tl.where(select_first[None, :], p_block, 0.0), axis=1)
            gp_first = tl.sum(tl.where(select_first, gp_block, 0.0), axis=0)
            v_first = tl.sum(tl.where(select_first[:, None], v_block, 0.0), axis=0)

            lse_boundary = tl.zeros([BLOCK_M], tl.float32)
            if t < N:
                lse_boundary = tl.load(
                    LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                    mask=mask_m,
                    other=0.0,
                ).to(tl.float32)
                scale_first = tl.math.exp2(tl.minimum(lse_first - lse_boundary, 0.0))
            else:
                scale_first = tl.zeros([BLOCK_M], tl.float32)

            weight_first = tl.math.exp2(tl.minimum(lse_first[:, None] - lse_block, 0.0))
            weight_first = tl.where(mask_m[:, None] & token_valid[None, :], weight_first * g_block, 0.0)
            b_first = tl.dot(weight_first, u_block, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            b_first = b_first + scale_first[:, None] * b_boundary
            a_first_carry = tl.sum(weight_first * p_block, axis=1) + scale_first * a_boundary

            dV_first = tl.sum(a_first[:, None] * b_first, axis=0)
            vb_first = tl.sum(b_first * v_first[None, :], axis=1)
            ds_g_first = g_first * (p_first - gp_first)
            ds_z_first = a_first * (vb_first - a_first_carry)

            tl.store(
                dVPart_ptr
                + pid_bh * stride_dvp_bh
                + chunk_idx * stride_dvp_chunk
                + pid_m * stride_dvp_tile
                + (t_start - chunk_start) * stride_dvp_t
                + d_offsets * stride_dvp_d,
                dV_first,
                mask=mask_d,
            )
            if WEIGHT_SHARING_ENC_DEC:
                tl.store(
                    DS_enc_ptr
                    + pid_bh * stride_dse_bh
                    + chunk_idx * stride_dse_chunk
                    + (t_start - chunk_start) * stride_dse_t
                    + m_offsets * stride_dse_m,
                    ds_g_first + ds_z_first,
                    mask=mask_m,
                )
            else:
                tl.store(
                    DS_enc_ptr
                    + pid_bh * stride_dse_bh
                    + chunk_idx * stride_dse_chunk
                    + (t_start - chunk_start) * stride_dse_t
                    + m_offsets * stride_dse_m,
                    ds_z_first,
                    mask=mask_m,
                )
                tl.store(
                    DS_dec_ptr
                    + pid_bh * stride_dsd_bh
                    + chunk_idx * stride_dsd_chunk
                    + (t_start - chunk_start) * stride_dsd_t
                    + m_offsets * stride_dsd_m,
                    ds_g_first,
                    mask=mask_m,
                )

            for j in tl.range(1, BLOCK_T):
                token_idx = t_start + j
                token_is_valid = token_idx < t
                select_j = t_local == j

                lse_j = tl.sum(tl.where(select_j[None, :], lse_block, 0.0), axis=1)
                g_j = tl.sum(tl.where(select_j[None, :], g_block, 0.0), axis=1)
                a_j = tl.sum(tl.where(select_j[None, :], a_block, 0.0), axis=1)
                p_j = tl.sum(tl.where(select_j[None, :], p_block, 0.0), axis=1)
                gp_j = tl.sum(tl.where(select_j, gp_block, 0.0), axis=0)
                v_j = tl.sum(tl.where(select_j[:, None], v_block, 0.0), axis=0)

                if t < N:
                    scale_j = tl.math.exp2(tl.minimum(lse_j - lse_boundary, 0.0))
                else:
                    scale_j = tl.zeros([BLOCK_M], tl.float32)

                weight_j = tl.math.exp2(tl.minimum(lse_j[:, None] - lse_block, 0.0))
                weight_j = tl.where(mask_m[:, None] & token_valid[None, :], weight_j * g_block, 0.0)
                weight_j = tl.where(t_local[None, :] >= j, weight_j, 0.0)

                b_j = tl.dot(weight_j, u_block, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                b_j = b_j + scale_j[:, None] * b_boundary
                a_j_carry = tl.sum(weight_j * p_block, axis=1) + scale_j * a_boundary

                dV_j = tl.sum(a_j[:, None] * b_j, axis=0)
                vb_j = tl.sum(b_j * v_j[None, :], axis=1)
                ds_g_j = g_j * (p_j - gp_j)
                ds_z_j = a_j * (vb_j - a_j_carry)

                tl.store(
                    dVPart_ptr
                    + pid_bh * stride_dvp_bh
                    + chunk_idx * stride_dvp_chunk
                    + pid_m * stride_dvp_tile
                    + (token_idx - chunk_start) * stride_dvp_t
                    + d_offsets * stride_dvp_d,
                    dV_j,
                    mask=token_is_valid & mask_d,
                )
                if WEIGHT_SHARING_ENC_DEC:
                    tl.store(
                        DS_enc_ptr
                        + pid_bh * stride_dse_bh
                        + chunk_idx * stride_dse_chunk
                        + (token_idx - chunk_start) * stride_dse_t
                        + m_offsets * stride_dse_m,
                        ds_g_j + ds_z_j,
                        mask=token_is_valid & mask_m,
                    )
                else:
                    tl.store(
                        DS_enc_ptr
                        + pid_bh * stride_dse_bh
                        + chunk_idx * stride_dse_chunk
                        + (token_idx - chunk_start) * stride_dse_t
                        + m_offsets * stride_dse_m,
                        ds_z_j,
                        mask=token_is_valid & mask_m,
                    )
                    tl.store(
                        DS_dec_ptr
                        + pid_bh * stride_dsd_bh
                        + chunk_idx * stride_dsd_chunk
                        + (token_idx - chunk_start) * stride_dsd_t
                        + m_offsets * stride_dsd_m,
                        ds_g_j,
                        mask=token_is_valid & mask_m,
                    )

            b_carry = b_first
            a_carry = a_first_carry
            t = t_start
    elif USE_SCALAR_PANEL_APPLY:
        t_panel_local = tl.arange(0, SCALAR_PANEL_T)
        t = chunk_end
        lse_next = tl.zeros([BLOCK_M], tl.float32)
        if t < N:
            lse_next = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
        while t > chunk_start:
            t_start = tl.maximum(t - SCALAR_PANEL_T, chunk_start)
            t_offsets = t_start + t_panel_local
            token_valid = t_offsets < t

            lse_block = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t_offsets[None, :] * stride_lse_n + m_offsets[:, None] * stride_lse_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            p_block = tl.load(
                P_ptr + pid_bh * stride_p_bh + t_offsets[None, :] * stride_p_t + m_offsets[:, None] * stride_p_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            g_block = tl.load(
                G_ptr + pid_bh * stride_g_bh + t_offsets[None, :] * stride_g_t + m_offsets[:, None] * stride_g_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            a_block = tl.load(
                A_ptr + pid_bh * stride_a_bh + t_offsets[None, :] * stride_a_t + m_offsets[:, None] * stride_a_m,
                mask=mask_m[:, None] & token_valid[None, :],
                other=0.0,
            ).to(tl.float32)
            u_block = tl.load(
                dO_ptr + pid_b * stride_do_b + t_offsets[:, None] * stride_do_n + pid_h * stride_do_h + d_offsets[None, :] * stride_do_d,
                mask=token_valid[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            v_block = tl.load(
                V_ptr + pid_b * stride_v_b + t_offsets[:, None] * stride_v_n + pid_h * stride_v_h + d_offsets[None, :] * stride_v_d,
                mask=token_valid[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)

            panel_t = t
            while panel_t > t_start:
                panel_t -= 1
                panel_idx = panel_t - t_start
                select_j = t_panel_local == panel_idx

                lse_curr = tl.sum(tl.where(select_j[None, :], lse_block, 0.0), axis=1)
                p_local = tl.sum(tl.where(select_j[None, :], p_block, 0.0), axis=1)
                g_t = tl.sum(tl.where(select_j[None, :], g_block, 0.0), axis=1)
                a_self = tl.sum(tl.where(select_j[None, :], a_block, 0.0), axis=1)
                u_t = tl.sum(tl.where(select_j[:, None], u_block, 0.0), axis=0)
                v_t = tl.sum(tl.where(select_j[:, None], v_block, 0.0), axis=0)

                if panel_t + 1 < N:
                    carry_scale = tl.math.exp2(tl.minimum(lse_curr - lse_next, 0.0))
                    b_carry = carry_scale[:, None] * b_carry
                    a_carry = carry_scale * a_carry

                b_carry = b_carry + g_t[:, None] * u_t[None, :]
                dV_acc = tl.sum(a_self[:, None] * b_carry, axis=0)
                vb_dot = tl.sum(b_carry * v_t[None, :], axis=1)
                gp_t = tl.load(
                    GP_ptr + pid_bh * stride_gp_bh + panel_t * stride_gp_t,
                ).to(tl.float32)
                ds_g = g_t * (p_local - gp_t)
                a_carry = a_carry + g_t * p_local
                ds_z = a_self * (vb_dot - a_carry)

                t_local_scalar = panel_t - chunk_start
                tl.store(
                    dVPart_ptr
                    + pid_bh * stride_dvp_bh
                    + chunk_idx * stride_dvp_chunk
                    + pid_m * stride_dvp_tile
                    + t_local_scalar * stride_dvp_t
                    + d_offsets * stride_dvp_d,
                    dV_acc,
                    mask=mask_d,
                )
                if WEIGHT_SHARING_ENC_DEC:
                    tl.store(
                        DS_enc_ptr + pid_bh * stride_dse_bh + chunk_idx * stride_dse_chunk + t_local_scalar * stride_dse_t + m_offsets * stride_dse_m,
                        ds_g + ds_z,
                        mask=mask_m,
                    )
                else:
                    tl.store(
                        DS_enc_ptr + pid_bh * stride_dse_bh + chunk_idx * stride_dse_chunk + t_local_scalar * stride_dse_t + m_offsets * stride_dse_m,
                        ds_z,
                        mask=mask_m,
                    )
                    tl.store(
                        DS_dec_ptr + pid_bh * stride_dsd_bh + chunk_idx * stride_dsd_chunk + t_local_scalar * stride_dsd_t + m_offsets * stride_dsd_m,
                        ds_g,
                        mask=mask_m,
                    )
                lse_next = lse_curr

            t = t_start
    else:
        t = chunk_end
        lse_next = tl.zeros([BLOCK_M], tl.float32)
        if t < N:
            lse_next = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
        while t > chunk_start:
            t -= 1
            lse_curr = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            if t + 1 < N:
                carry_scale = tl.math.exp2(tl.minimum(lse_curr - lse_next, 0.0))
                b_carry = carry_scale[:, None] * b_carry
                a_carry = carry_scale * a_carry

            p_local = tl.load(
                P_ptr + pid_bh * stride_p_bh + t * stride_p_t + m_offsets * stride_p_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            g_t = tl.load(
                G_ptr + pid_bh * stride_g_bh + t * stride_g_t + m_offsets * stride_g_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            a_self = tl.load(
                A_ptr + pid_bh * stride_a_bh + t * stride_a_t + m_offsets * stride_a_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)

            u_t = tl.load(
                dO_ptr + pid_b * stride_do_b + t * stride_do_n + pid_h * stride_do_h + d_offsets * stride_do_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            b_carry = b_carry + g_t[:, None] * u_t[None, :]

            v_t = tl.load(
                V_ptr + pid_b * stride_v_b + t * stride_v_n + pid_h * stride_v_h + d_offsets * stride_v_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            dV_acc = tl.sum(a_self[:, None] * b_carry, axis=0)
            vb_dot = tl.sum(b_carry * v_t[None, :], axis=1)

            gp_t = tl.load(
                GP_ptr + pid_bh * stride_gp_bh + t * stride_gp_t,
            ).to(tl.float32)
            ds_g = g_t * (p_local - gp_t)
            a_carry = a_carry + g_t * p_local
            ds_z = a_self * (vb_dot - a_carry)

            t_local_scalar = t - chunk_start
            tl.store(
                dVPart_ptr
                + pid_bh * stride_dvp_bh
                + chunk_idx * stride_dvp_chunk
                + pid_m * stride_dvp_tile
                + t_local_scalar * stride_dvp_t
                + d_offsets * stride_dvp_d,
                dV_acc,
                mask=mask_d,
            )
            if WEIGHT_SHARING_ENC_DEC:
                tl.store(
                    DS_enc_ptr + pid_bh * stride_dse_bh + chunk_idx * stride_dse_chunk + t_local_scalar * stride_dse_t + m_offsets * stride_dse_m,
                    ds_g + ds_z,
                    mask=mask_m,
                )
            else:
                tl.store(
                    DS_enc_ptr + pid_bh * stride_dse_bh + chunk_idx * stride_dse_chunk + t_local_scalar * stride_dse_t + m_offsets * stride_dse_m,
                    ds_z,
                    mask=mask_m,
                )
                tl.store(
                    DS_dec_ptr + pid_bh * stride_dsd_bh + chunk_idx * stride_dsd_chunk + t_local_scalar * stride_dsd_t + m_offsets * stride_dsd_m,
                    ds_g,
                    mask=mask_m,
                )
            lse_next = lse_curr


@triton.jit
def flare_chunk_bwd_lse_scan_apply_fused_single_dv(
    V_ptr,
    LSE_ptr,
    dO_ptr,
    P_ptr,
    G_ptr,
    GP_ptr,
    A_ptr,
    DS_enc_ptr,
    DS_dec_ptr,
    dVPart_ptr,
    stride_v_b, stride_v_n, stride_v_h, stride_v_d,
    stride_lse_b, stride_lse_h, stride_lse_n, stride_lse_m,
    stride_do_b, stride_do_n, stride_do_h, stride_do_d,
    stride_p_bh, stride_p_t, stride_p_m,
    stride_g_bh, stride_g_t, stride_g_m,
    stride_gp_bh, stride_gp_t,
    stride_a_bh, stride_a_t, stride_a_m,
    stride_dse_bh, stride_dse_chunk, stride_dse_t, stride_dse_m,
    stride_dsd_bh, stride_dsd_chunk, stride_dsd_t, stride_dsd_m,
    stride_dvp_bh, stride_dvp_chunk, stride_dvp_tile, stride_dvp_t, stride_dvp_d,
    NUM_CHUNKS,
    H, M, N,
    CHUNK_SIZE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    WEIGHT_SHARING_ENC_DEC: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = pid_bh // H
    pid_h = pid_bh - pid_b * H
    m_start = pid_m * BLOCK_M
    m_local = tl.arange(0, BLOCK_M)
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    d_offsets = tl.arange(0, BLOCK_DV)
    mask_m = m_offsets < M
    mask_d = d_offsets < D_VALUE

    # Carry entering the logical position just after the current chunk. As we
    # step chunks from the end of the sequence back to the start, the carry
    # stays in registers and naturally becomes the next chunk's "future" input.
    b_carry = tl.zeros([BLOCK_M, BLOCK_DV], tl.float32)
    a_carry = tl.zeros([BLOCK_M], tl.float32)

    chunk_idx = NUM_CHUNKS
    while chunk_idx > 0:
        chunk_idx -= 1
        chunk_start = chunk_idx * CHUNK_SIZE
        chunk_end = tl.minimum(chunk_start + CHUNK_SIZE, N)

        t = chunk_end
        lse_next = tl.zeros([BLOCK_M], tl.float32)
        if t < N:
            lse_next = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
        while t > chunk_start:
            t -= 1
            lse_curr = tl.load(
                LSE_ptr + pid_b * stride_lse_b + pid_h * stride_lse_h + t * stride_lse_n + m_offsets * stride_lse_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            if t + 1 < N:
                carry_scale = tl.math.exp2(tl.minimum(lse_curr - lse_next, 0.0))
                b_carry = carry_scale[:, None] * b_carry
                a_carry = carry_scale * a_carry

            p_local = tl.load(
                P_ptr + pid_bh * stride_p_bh + t * stride_p_t + m_offsets * stride_p_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            g_t = tl.load(
                G_ptr + pid_bh * stride_g_bh + t * stride_g_t + m_offsets * stride_g_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)
            a_self = tl.load(
                A_ptr + pid_bh * stride_a_bh + t * stride_a_t + m_offsets * stride_a_m,
                mask=mask_m,
                other=0.0,
            ).to(tl.float32)

            u_t = tl.load(
                dO_ptr + pid_b * stride_do_b + t * stride_do_n + pid_h * stride_do_h + d_offsets * stride_do_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            b_carry = b_carry + g_t[:, None] * u_t[None, :]

            v_t = tl.load(
                V_ptr + pid_b * stride_v_b + t * stride_v_n + pid_h * stride_v_h + d_offsets * stride_v_d,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
            dV_acc = tl.sum(a_self[:, None] * b_carry, axis=0)
            vb_dot = tl.sum(b_carry * v_t[None, :], axis=1)

            gp_t = tl.load(
                GP_ptr + pid_bh * stride_gp_bh + t * stride_gp_t,
            ).to(tl.float32)
            ds_g = g_t * (p_local - gp_t)
            a_carry = a_carry + g_t * p_local
            ds_z = a_self * (vb_dot - a_carry)

            t_local = t - chunk_start
            tl.store(
                dVPart_ptr
                + pid_bh * stride_dvp_bh
                + chunk_idx * stride_dvp_chunk
                + pid_m * stride_dvp_tile
                + t_local * stride_dvp_t
                + d_offsets * stride_dvp_d,
                dV_acc,
                mask=mask_d,
            )
            if WEIGHT_SHARING_ENC_DEC:
                tl.store(
                    DS_enc_ptr + pid_bh * stride_dse_bh + chunk_idx * stride_dse_chunk + t_local * stride_dse_t + m_offsets * stride_dse_m,
                    ds_g + ds_z,
                    mask=mask_m,
                )
            else:
                tl.store(
                    DS_enc_ptr + pid_bh * stride_dse_bh + chunk_idx * stride_dse_chunk + t_local * stride_dse_t + m_offsets * stride_dse_m,
                    ds_z,
                    mask=mask_m,
                )
                tl.store(
                    DS_dec_ptr + pid_bh * stride_dsd_bh + chunk_idx * stride_dsd_chunk + t_local * stride_dsd_t + m_offsets * stride_dsd_m,
                    ds_g,
                    mask=mask_m,
                )
            lse_next = lse_curr


@triton.jit
def flare_chunk_bwd_lse_dv_reduce(
    dVPart_ptr,
    dV_ptr,
    stride_dvp_bh, stride_dvp_chunk, stride_dvp_tile, stride_dvp_t, stride_dvp_d,
    stride_dv_bh, stride_dv_t, stride_dv_d,
    NUM_M_TILES,
    N,
    CHUNK_SIZE: tl.constexpr,
    D_VALUE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    chunk_idx = tl.program_id(1)
    pid_d = tl.program_id(2)

    t_offsets = tl.arange(0, CHUNK_SIZE)
    d_offsets = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    global_t = chunk_idx * CHUNK_SIZE + t_offsets
    mask_t = global_t < N
    mask_d = d_offsets < D_VALUE
    acc = tl.zeros([CHUNK_SIZE, BLOCK_D], tl.float32)

    m_tile = 0
    while m_tile < NUM_M_TILES:
        dv_part = tl.load(
            dVPart_ptr
            + pid_bh * stride_dvp_bh
            + chunk_idx * stride_dvp_chunk
            + m_tile * stride_dvp_tile
            + t_offsets[:, None] * stride_dvp_t
            + d_offsets[None, :] * stride_dvp_d,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += dv_part
        m_tile += 1

    tl.store(
        dV_ptr + pid_bh * stride_dv_bh + global_t[:, None] * stride_dv_t + d_offsets[None, :] * stride_dv_d,
        acc,
        mask=mask_t[:, None] & mask_d[None, :],
    )


def _chunked_flare_lse_backward_impl(ctx, dO, dTimings=None):
    if dO is None:
        return None, None, None, None, None, None, None, None, None

    separate_Q_dec = getattr(ctx, "separate_Q_dec", False)
    separate_K_dec = getattr(ctx, "separate_K_dec", False)
    weight_sharing_enc_dec = getattr(ctx, "weight_sharing_enc_dec", True)
    saved = ctx.saved_tensors
    expected_len = 8 + int(separate_Q_dec) + int(separate_K_dec)
    if len(saved) != expected_len:
        raise RuntimeError(
            f"ChunkedFLARE backward expected {expected_len} saved tensors "
            f"(8 base + separate_Q_dec={separate_Q_dec} + separate_K_dec={separate_K_dec}), got {len(saved)}."
        )

    Q, K, V, LSE_enc_ctx, LSE_dec, prefix_max, prefix_den, prefix_num = saved[:8]
    idx = 8
    if separate_Q_dec:
        Q_dec_saved = saved[idx]
        idx += 1
    else:
        Q_dec_saved = K
    if separate_K_dec:
        K_dec_saved = saved[idx]
    else:
        K_dec_saved = Q
    scale = ctx.scale

    H, M, D = Q.shape
    B, N, Hk, Dk = K.shape
    Dv = V.shape[3]
    if Hk != H or Dk != D:
        raise RuntimeError(
            f"ChunkedFLARE backward expected K to match Q head/dim. Got Q.shape={Q.shape}, K.shape={K.shape}."
        )
    if K.shape[:3] != V.shape[:3]:
        raise RuntimeError(
            "ChunkedFLARE backward expected K and V to agree on [B, N, H]. "
            f"Got K.shape={K.shape}, V.shape={V.shape}."
        )
    if (M % 16) != 0 or (D % 16) != 0 or (Dv % 16) != 0:
        raise ValueError(
            "ChunkedFLARE backward requires M, D_score, and D_value be multiples of 16. "
            f"Got M={M}, D_score={D}, D_value={Dv}"
        )

    device = Q.device
    profile_data = dTimings if isinstance(dTimings, dict) else getattr(ctx, "profile_timings", None)
    bwd_timings = profile_data["backward"] if isinstance(profile_data, dict) else None
    bwd_resources = profile_data.setdefault("backward_resources", {}) if isinstance(profile_data, dict) else None
    BH = B * H
    chunk_size = int(ctx.chunk_size)
    num_chunks = math.ceil(N / chunk_size)
    bwd_defaults = _get_chunked_backward_bucket_defaults(M, D, Dv)
    block_m_env = os.environ.get("FLARE_LSE_BWD_BLOCK_M", "")
    if block_m_env:
        block_m = int(block_m_env)
    else:
        block_m = int(bwd_defaults["block_m"])
    if block_m > M:
        raise ValueError(f"ChunkedFLARE LSE backward requires BLOCK_M <= M. Got M={M}, BLOCK_M={block_m}")
    if (block_m % 16) != 0:
        raise ValueError(f"ChunkedFLARE LSE backward requires BLOCK_M be a multiple of 16. Got BLOCK_M={block_m}")
    _require_power_of_two_tile("FLARE_LSE_BWD_BLOCK_M", block_m)
    block_dv_env = os.environ.get("FLARE_LSE_BWD_BLOCK_DV", "")
    if block_dv_env:
        block_dv_state = int(block_dv_env)
    else:
        block_dv_state = min(Dv, int(bwd_defaults["block_dv_state"]))
    if block_dv_state > Dv:
        block_dv_state = Dv
    if (block_dv_state % 16) != 0:
        raise ValueError(
            f"ChunkedFLARE LSE backward requires BLOCK_DV be a multiple of 16. Got BLOCK_DV={block_dv_state}"
        )
    block_m_replay_env = os.environ.get("FLARE_LSE_BWD_SCORE_BLOCK_M", "")
    block_m_replay = int(block_m_replay_env) if block_m_replay_env else block_m
    if block_m_replay > block_m:
        raise ValueError(
            f"ChunkedFLARE LSE backward requires BLOCK_M_REPLAY <= BLOCK_M. Got BLOCK_M={block_m}, BLOCK_M_REPLAY={block_m_replay}."
        )
    if (block_m_replay % 16) != 0:
        raise ValueError(f"ChunkedFLARE LSE backward requires BLOCK_M_REPLAY be a multiple of 16. Got {block_m_replay}")
    block_k_env = os.environ.get("FLARE_LSE_BWD_BLOCK_K", "")
    block_k_replay = int(block_k_env) if block_k_env else int(bwd_defaults["block_k"])
    if block_k_replay > D:
        block_k_replay = D
    if (block_k_replay % 16) != 0:
        raise ValueError(f"ChunkedFLARE LSE backward requires BLOCK_K be a multiple of 16. Got BLOCK_K={block_k_replay}")
    if D % block_k_replay != 0:
        raise ValueError(f"ChunkedFLARE LSE backward requires BLOCK_K divide D_score. Got D_score={D}, BLOCK_K={block_k_replay}")
    block_d_part_env = os.environ.get("FLARE_LSE_BWD_BLOCK_D_PART", "")
    block_d_part = int(block_d_part_env) if block_d_part_env else min(Dv, int(bwd_defaults["block_d_part"]))
    if block_d_part > Dv:
        block_d_part = Dv
    if (block_d_part % 16) != 0:
        raise ValueError(f"ChunkedFLARE LSE backward requires BLOCK_D_PART be a multiple of 16. Got BLOCK_D_PART={block_d_part}")
    block_t_replay_env = os.environ.get("FLARE_LSE_BWD_BLOCK_T_REPLAY", "")
    block_t_replay = int(block_t_replay_env) if block_t_replay_env else (16 if chunk_size <= 32 else 32)
    block_t_replay = min(block_t_replay, chunk_size)
    if (block_t_replay % 16) != 0:
        raise ValueError(f"ChunkedFLARE LSE backward requires BLOCK_T_REPLAY be a multiple of 16. Got {block_t_replay}")
    block_t_state_env = os.environ.get("FLARE_LSE_BWD_BLOCK_T_STATE", "")
    block_t_state = int(block_t_state_env) if block_t_state_env else (16 if chunk_size <= 32 else 32)
    block_t_state = min(block_t_state, chunk_size)
    if (block_t_state % 16) != 0:
        raise ValueError(f"ChunkedFLARE LSE backward requires BLOCK_T_STATE be a multiple of 16. Got {block_t_state}")
    replay_num_warps, replay_num_stages = _resolve_backward_launch(
        "replay",
        default_num_warps=int(bwd_defaults["replay_launch"][0]),
        default_num_stages=int(bwd_defaults["replay_launch"][1]),
    )
    state_num_warps, state_num_stages = _resolve_backward_launch(
        "state",
        default_num_warps=int(bwd_defaults["state_launch"][0]),
        default_num_stages=int(bwd_defaults["state_launch"][1]),
    )
    use_blocked_suffix_summary = True
    block_t_qk, block_d_qk, qk_num_warps, qk_num_stages = _select_chunked_bwd_qk_launch(M, D, chunk_size)
    use_blocked_apply = False
    use_scalar_panel_apply = False
    block_t_apply = 16
    scalar_panel_t_apply = 4
    num_m_tiles = triton.cdiv(M, block_m)
    num_dv_state_tiles = triton.cdiv(Dv, block_dv_state)
    num_replay_tiles = triton.cdiv(block_m, block_m_replay)
    use_fused_single_dv_apply = num_dv_state_tiles == 1
    use_fused_chunk_scan = False
    num_qk_d_tiles = triton.cdiv(D, block_d_qk)
    use_blocked_score_replay = block_m <= 32
    use_subtiled_score_replay = (not use_blocked_score_replay) and (block_m_replay < block_m)
    if use_subtiled_score_replay and num_replay_tiles > 4:
        raise ValueError(
            "ChunkedFLARE LSE backward currently supports at most 4 internal score-replay subtiles. "
            f"Got BLOCK_M={block_m}, BLOCK_M_REPLAY={block_m_replay}."
        )

    # The backward pass is split into a few replay/scanning phases:
    # 1) replay the forward normalized-value recurrence inside each chunk to recover per-token scalar terms,
    # 2) either:
    #    - summarize each chunk + reverse-scan chunk carries + replay with the scanned carry, or
    #    - in the single-DV-tile fast path, fuse that whole reverse pass into one kernel,
    # 3) contract dS with Q/K to obtain dQ and dK.
    #
    # All temporary tensors below are float32 even when inputs are lower precision. The kernels repeatedly
    # combine exponentials and long prefix/suffix reductions, so keeping the workspaces in fp32 avoids
    # compounding numerical error while we reconstruct the same stabilized algebra used in the forward pass.
    def profiled_bwd_call(key: str, op):
        return _profiled_bwd_call(device, bwd_timings, key, op, resource_bucket=bwd_resources)

    dO_bnhd = profiled_bwd_call(
        "grad_input_cast",
        lambda: dO.contiguous().to(torch.float32),
    )

    def alloc_bwd_buffers():
        # Per-token/query scalar p_t(m) = <z_t(m), dO_t>, where z_t(m) is the normalized running value state
        # after token t has been incorporated for query row m. The autotuned replay kernel may choose either
        # a single D tile or a multi-tile path, so keep the destination zero-initialized.
        p_buf = torch.zeros((BH, N, M), device=device, dtype=torch.float32)
        # g_t(m) = exp(score_dec_t(m) - LSE_dec_t): token-wise softmax weight over the M decode rows.
        g_buf = torch.empty((BH, N, M), device=device, dtype=torch.float32)
        # gp_buf[t] = sum_m g_t(m) * p_t(m): the global M-axis reduction used by ds_g.
        gp_buf = torch.empty((BH, N), device=device, dtype=torch.float32)
        # a_t(m) = exp(score_t(m) - L_t(m)): encoder-side weight of token t inside latent row m.
        # We keep this explicitly instead of reconstructing it from g_t because sharp-softmax
        # regimes can drive g_t to exact zero while a_t is still non-negligible.
        a_buf = torch.empty((BH, N, M), device=device, dtype=torch.float32)
        # Encode score-space gradient dS_enc for each chunk/token/query row.
        dS_enc = torch.empty((BH, num_chunks, chunk_size, M), device=device, dtype=torch.float32)
        # Decode score-space gradient dS_dec. When encode/decode share weights this aliases dS_enc
        # to avoid an extra workspace allocation.
        dS_dec = dS_enc if weight_sharing_enc_dec else torch.empty((BH, num_chunks, chunk_size, M), device=device, dtype=torch.float32)
        # dV is reduced over M tiles. When there is only one M tile, dV_part already holds the final values
        # and we can view it directly instead of allocating a separate reduction target.
        if num_m_tiles > 1:
            dV_bhtd = torch.zeros((BH, N, Dv), device=device, dtype=torch.float32)
        else:
            dV_bhtd = None
        dQ = torch.zeros((H, M, D), device=device, dtype=torch.float32)
        dK = torch.zeros((B, N, H, D), device=device, dtype=torch.float32)
        # The generic path materializes chunk summaries plus scanned carries.
        # The experimental fused chunk-scan path keeps those carries in
        # registers across chunks and skips these workspaces entirely.
        if not use_fused_chunk_scan:
            # Per-chunk reverse-time summaries:
            # - b_local stores the vector suffix carry over dO (kept per D tile),
            # - a_local stores the scalar suffix carry over g * p,
            # - scale_buf tells us how to re-express the next chunk's carry in the current chunk's log-space.
            b_local = torch.empty((BH, num_chunks, num_m_tiles, block_m, Dv), device=device, dtype=torch.float32)
            a_local = torch.empty((BH, num_chunks, num_m_tiles, block_m), device=device, dtype=torch.float32)
            scale_buf = torch.empty((BH, num_chunks, num_m_tiles, block_m), device=device, dtype=torch.float32)
            # Reverse exclusive scan outputs: the carry entering each chunk from all strictly later chunks.
            b_in = torch.empty_like(b_local)
            a_in = torch.empty_like(a_local)
        else:
            b_local = None
            a_local = None
            scale_buf = None
            b_in = None
            a_in = None

        # Partial outputs from the second replay:
        # - vb_part stores per-D-tile pieces of <b_carry_t(m), v_t> so finalize can sum across D tiles,
        # - dV_part stores each M-tile's contribution to dV before the M reduction.
        if num_dv_state_tiles > 1:
            vb_part = torch.empty(
                (BH, num_chunks, num_m_tiles, num_dv_state_tiles, chunk_size, block_m),
                device=device,
                dtype=torch.float32,
            )
        else:
            vb_part = None
        dV_part = torch.empty((BH, num_chunks, num_m_tiles, chunk_size, Dv), device=device, dtype=torch.float32)
        return p_buf, g_buf, gp_buf, a_buf, dS_enc, dS_dec, dV_bhtd, dQ, dK, b_local, a_local, scale_buf, b_in, a_in, vb_part, dV_part

    (
        p_buf,
        g_buf,
        gp_buf,
        a_buf,
        dS_enc,
        dS_dec,
        dV_bhtd,
        dQ,
        dK,
        b_local,
        a_local,
        scale_buf,
        b_in,
        a_in,
        vb_part,
        dV_part,
    ) = profiled_bwd_call(
        "alloc_bwd_buffers",
        alloc_bwd_buffers,
    )

    # Phase 1: replay the forward recurrence inside each chunk.
    #
    # For each (batch-head, chunk, M-tile, D-tile) this kernel walks tokens left-to-right, rebuilds the
    # normalized running value state z_t(m), and emits:
    # - p_buf[t, m] = <z_t(m), dO_t>,
    # - g_buf[t, m] = exp(score_t(m) - LSE_M_t),
    # - a_buf[t, m] = exp(score_t(m) - L_t(m)).
    #
    def launch_bwd_p_part():
        return flare_chunk_bwd_lse_p_part[
            (BH, num_chunks * num_m_tiles, triton.cdiv(Dv, block_d_part))
        ](
            Q, K, V, Q_dec_saved, K_dec_saved, prefix_max, prefix_den, prefix_num, LSE_dec, dO_bnhd, p_buf, g_buf, a_buf,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            prefix_max.stride(0), prefix_max.stride(1), prefix_max.stride(2),
            prefix_den.stride(0), prefix_den.stride(1), prefix_den.stride(2),
            prefix_num.stride(0), prefix_num.stride(1), prefix_num.stride(2), prefix_num.stride(3),
            Q_dec_saved.stride(0), Q_dec_saved.stride(1), Q_dec_saved.stride(2), Q_dec_saved.stride(3),
            K_dec_saved.stride(0), K_dec_saved.stride(1), K_dec_saved.stride(2),
            LSE_dec.stride(0), LSE_dec.stride(1),
            dO_bnhd.stride(0), dO_bnhd.stride(1), dO_bnhd.stride(2), dO_bnhd.stride(3),
            p_buf.stride(0), p_buf.stride(1), p_buf.stride(2),
            g_buf.stride(0), g_buf.stride(1), g_buf.stride(2),
            a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
            H, M, N,
            scale,
            NUM_M_TILES=num_m_tiles,
            CHUNK_SIZE=chunk_size,
            D_SCORE=D,
            D_VALUE=Dv,
            BLOCK_M=block_m,
            BLOCK_M_REPLAY=block_m_replay,
            NUM_REPLAY_TILES=num_replay_tiles,
            BLOCK_D=block_d_part,
            BLOCK_K=block_k_replay,
            BLOCK_T=block_t_replay,
            INPUT_PRECISION=ctx.input_precision,
            USE_BLOCKED_SCORE_REPLAY=use_blocked_score_replay,
            USE_SUBTILED_SCORE_REPLAY=use_subtiled_score_replay,
            WEIGHT_SHARING_ENC_DEC=weight_sharing_enc_dec,
            num_warps=replay_num_warps,
            num_stages=replay_num_stages,
        )

    profiled_bwd_call("flare_chunk_bwd_lse_p_part", launch_bwd_p_part)

    def launch_bwd_gp_reduce():
        # Reduce gp in chunk-tiled token panels to avoid one-program-per-token
        # launch overhead on long contexts.
        return flare_chunk_bwd_lse_gp_reduce[(BH, num_chunks)](
            p_buf,
            g_buf,
            gp_buf,
            p_buf.stride(0), p_buf.stride(1), p_buf.stride(2),
            g_buf.stride(0), g_buf.stride(1), g_buf.stride(2),
            gp_buf.stride(0), gp_buf.stride(1),
            N,
            M,
            CHUNK_SIZE=chunk_size,
            BLOCK_M=block_m,
            BLOCK_T=block_t_state,
            num_warps=state_num_warps,
            num_stages=state_num_stages,
        )

    profiled_bwd_call("flare_chunk_bwd_lse_gp_reduce", launch_bwd_gp_reduce)

    if use_fused_chunk_scan:
        # Experimental path: walk chunks and tokens in one reverse kernel,
        # carrying the future suffix state entirely in registers. This subsumes
        # the old summary + carry-scan + apply phases, but it is opt-in because
        # the larger per-program working set can lose on occupancy-sensitive shapes.
        def launch_bwd_apply_fused():
            return flare_chunk_bwd_lse_scan_apply_fused_single_dv[(BH, num_m_tiles)](
                V, LSE_enc_ctx, dO_bnhd, p_buf, g_buf, gp_buf, a_buf, dS_enc, dS_dec, dV_part,
                V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                LSE_enc_ctx.stride(0), LSE_enc_ctx.stride(1), LSE_enc_ctx.stride(2), LSE_enc_ctx.stride(3),
                dO_bnhd.stride(0), dO_bnhd.stride(1), dO_bnhd.stride(2), dO_bnhd.stride(3),
                p_buf.stride(0), p_buf.stride(1), p_buf.stride(2),
                g_buf.stride(0), g_buf.stride(1), g_buf.stride(2),
                gp_buf.stride(0), gp_buf.stride(1),
                a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
                dS_enc.stride(0), dS_enc.stride(1), dS_enc.stride(2), dS_enc.stride(3),
                dS_dec.stride(0), dS_dec.stride(1), dS_dec.stride(2), dS_dec.stride(3),
                dV_part.stride(0), dV_part.stride(1), dV_part.stride(2), dV_part.stride(3), dV_part.stride(4),
                num_chunks,
                H, M, N,
                CHUNK_SIZE=chunk_size, D_VALUE=Dv, BLOCK_M=block_m, BLOCK_DV=block_dv_state,
                WEIGHT_SHARING_ENC_DEC=weight_sharing_enc_dec,
            )

        profiled_bwd_call("flare_chunk_bwd_lse_scan_apply_fused", launch_bwd_apply_fused)
    else:
        # Generic path:
        # 2a) summarize each chunk independently in reverse time,
        # 2b) reverse-scan chunk carries,
        # 3a/3b) replay each chunk with the scanned carry to produce dV and dS.
        def launch_bwd_chunk_summary():
            return flare_chunk_bwd_lse_chunk_summary[(BH, num_chunks, num_m_tiles * num_dv_state_tiles)](
                LSE_enc_ctx, dO_bnhd, p_buf, g_buf, b_local, a_local, scale_buf,
                LSE_enc_ctx.stride(0), LSE_enc_ctx.stride(1), LSE_enc_ctx.stride(2), LSE_enc_ctx.stride(3),
                dO_bnhd.stride(0), dO_bnhd.stride(1), dO_bnhd.stride(2), dO_bnhd.stride(3),
                p_buf.stride(0), p_buf.stride(1), p_buf.stride(2),
                g_buf.stride(0), g_buf.stride(1), g_buf.stride(2),
                b_local.stride(0), b_local.stride(1), b_local.stride(2), b_local.stride(3), b_local.stride(4),
                a_local.stride(0), a_local.stride(1), a_local.stride(2), a_local.stride(3),
                scale_buf.stride(0), scale_buf.stride(1), scale_buf.stride(2), scale_buf.stride(3),
                H, M, N,
                NUM_DV_TILES=num_dv_state_tiles,
                CHUNK_SIZE=chunk_size, D_VALUE=Dv, BLOCK_M=block_m, BLOCK_DV=block_dv_state,
                BLOCK_T=block_t_apply,
                INPUT_PRECISION=ctx.input_precision,
                USE_BLOCKED_SUFFIX_SUMMARY=use_blocked_suffix_summary,
                num_warps=2,
                num_stages=1,
            )

        profiled_bwd_call("flare_chunk_bwd_lse_chunk_summary", launch_bwd_chunk_summary)

        def launch_bwd_carry_scan():
            return flare_chunk_bwd_lse_carry_scan[(BH, num_m_tiles * num_dv_state_tiles)](
                b_local, a_local, scale_buf, b_in, a_in,
                b_local.stride(0), b_local.stride(1), b_local.stride(2), b_local.stride(3), b_local.stride(4),
                a_local.stride(0), a_local.stride(1), a_local.stride(2), a_local.stride(3),
                scale_buf.stride(0), scale_buf.stride(1), scale_buf.stride(2), scale_buf.stride(3),
                b_in.stride(0), b_in.stride(1), b_in.stride(2), b_in.stride(3), b_in.stride(4),
                a_in.stride(0), a_in.stride(1), a_in.stride(2), a_in.stride(3),
                M, num_chunks,
                NUM_DV_TILES=num_dv_state_tiles,
                D_VALUE=Dv, BLOCK_M=block_m, BLOCK_DV=block_dv_state,
            )

        profiled_bwd_call("flare_chunk_bwd_lse_carry_scan", launch_bwd_carry_scan)

        if use_fused_single_dv_apply:
            def launch_bwd_apply_fused():
                return flare_chunk_bwd_lse_chunk_apply_fused_single_dv[(BH, num_chunks, num_m_tiles)](
                    V, LSE_enc_ctx, dO_bnhd, p_buf, g_buf, gp_buf, a_buf, b_in, a_in, dS_enc, dS_dec, dV_part,
                    V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                    LSE_enc_ctx.stride(0), LSE_enc_ctx.stride(1), LSE_enc_ctx.stride(2), LSE_enc_ctx.stride(3),
                    dO_bnhd.stride(0), dO_bnhd.stride(1), dO_bnhd.stride(2), dO_bnhd.stride(3),
                    p_buf.stride(0), p_buf.stride(1), p_buf.stride(2),
                    g_buf.stride(0), g_buf.stride(1), g_buf.stride(2),
                    gp_buf.stride(0), gp_buf.stride(1),
                    a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
                    b_in.stride(0), b_in.stride(1), b_in.stride(2), b_in.stride(3), b_in.stride(4),
                    a_in.stride(0), a_in.stride(1), a_in.stride(2), a_in.stride(3),
                    dS_enc.stride(0), dS_enc.stride(1), dS_enc.stride(2), dS_enc.stride(3),
                    dS_dec.stride(0), dS_dec.stride(1), dS_dec.stride(2), dS_dec.stride(3),
                    dV_part.stride(0), dV_part.stride(1), dV_part.stride(2), dV_part.stride(3), dV_part.stride(4),
                    H, M, N,
                    CHUNK_SIZE=chunk_size, D_VALUE=Dv, BLOCK_M=block_m, BLOCK_DV=block_dv_state,
                    BLOCK_T=block_t_apply, SCALAR_PANEL_T=scalar_panel_t_apply,
                    INPUT_PRECISION=ctx.input_precision,
                    USE_BLOCKED_APPLY=use_blocked_apply,
                    USE_SCALAR_PANEL_APPLY=use_scalar_panel_apply,
                    WEIGHT_SHARING_ENC_DEC=weight_sharing_enc_dec,
                )

            profiled_bwd_call("flare_chunk_bwd_lse_chunk_apply_fused", launch_bwd_apply_fused)
        else:
            def launch_bwd_apply_part():
                return flare_chunk_bwd_lse_chunk_apply_part[(BH, num_chunks, num_m_tiles * num_dv_state_tiles)](
                    V, LSE_enc_ctx, dO_bnhd, g_buf, a_buf, b_in, vb_part, dV_part,
                    V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                    LSE_enc_ctx.stride(0), LSE_enc_ctx.stride(1), LSE_enc_ctx.stride(2), LSE_enc_ctx.stride(3),
                    dO_bnhd.stride(0), dO_bnhd.stride(1), dO_bnhd.stride(2), dO_bnhd.stride(3),
                    g_buf.stride(0), g_buf.stride(1), g_buf.stride(2),
                    a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
                    b_in.stride(0), b_in.stride(1), b_in.stride(2), b_in.stride(3), b_in.stride(4),
                    vb_part.stride(0), vb_part.stride(1), vb_part.stride(2), vb_part.stride(3), vb_part.stride(4), vb_part.stride(5),
                    dV_part.stride(0), dV_part.stride(1), dV_part.stride(2), dV_part.stride(3), dV_part.stride(4),
                    H, M, N,
                    NUM_DV_TILES=num_dv_state_tiles,
                    CHUNK_SIZE=chunk_size, D_VALUE=Dv, BLOCK_M=block_m, BLOCK_DV=block_dv_state,
                    BLOCK_T=block_t_apply, SCALAR_PANEL_T=scalar_panel_t_apply,
                    INPUT_PRECISION=ctx.input_precision, USE_BLOCKED_APPLY=use_blocked_apply,
                    num_warps=2,
                    num_stages=1,
                )

            profiled_bwd_call("flare_chunk_bwd_lse_chunk_apply_part", launch_bwd_apply_part)

            def launch_bwd_apply_finalize():
                return flare_chunk_bwd_lse_chunk_apply_finalize[(BH, num_chunks, num_m_tiles)](
                    LSE_enc_ctx, p_buf, g_buf, gp_buf, a_buf, a_in, vb_part, dS_enc, dS_dec,
                    LSE_enc_ctx.stride(0), LSE_enc_ctx.stride(1), LSE_enc_ctx.stride(2), LSE_enc_ctx.stride(3),
                    p_buf.stride(0), p_buf.stride(1), p_buf.stride(2),
                    g_buf.stride(0), g_buf.stride(1), g_buf.stride(2),
                    gp_buf.stride(0), gp_buf.stride(1),
                    a_buf.stride(0), a_buf.stride(1), a_buf.stride(2),
                    a_in.stride(0), a_in.stride(1), a_in.stride(2), a_in.stride(3),
                    vb_part.stride(0), vb_part.stride(1), vb_part.stride(2), vb_part.stride(3), vb_part.stride(4), vb_part.stride(5),
                    dS_enc.stride(0), dS_enc.stride(1), dS_enc.stride(2), dS_enc.stride(3),
                    dS_dec.stride(0), dS_dec.stride(1), dS_dec.stride(2), dS_dec.stride(3),
                    H, M, N,
                    NUM_DV_TILES=num_dv_state_tiles,
                    CHUNK_SIZE=chunk_size, BLOCK_M=block_m, USE_BLOCKED_APPLY=use_blocked_apply,
                    BLOCK_T=block_t_apply, SCALAR_PANEL_T=scalar_panel_t_apply,
                    WEIGHT_SHARING_ENC_DEC=weight_sharing_enc_dec,
                    num_warps=2,
                    num_stages=1,
                )

            profiled_bwd_call("flare_chunk_bwd_lse_chunk_apply_finalize", launch_bwd_apply_finalize)

    # Phase 3c: reduce the per-M-tile dV partials into the final dV buffer.
    # Each token/dimension location is just a sum over query tiles, so when there is only one M tile
    # we can bypass the reduction kernel and reuse dV_part directly.
    if num_m_tiles > 1:
        def launch_bwd_dv_reduce():
            return flare_chunk_bwd_lse_dv_reduce[lambda meta: (BH, num_chunks, triton.cdiv(Dv, meta["BLOCK_D"]))](
                dV_part, dV_bhtd,
                dV_part.stride(0), dV_part.stride(1), dV_part.stride(2), dV_part.stride(3), dV_part.stride(4),
                dV_bhtd.stride(0), dV_bhtd.stride(1), dV_bhtd.stride(2),
                num_m_tiles, N,
                CHUNK_SIZE=chunk_size, D_VALUE=Dv, BLOCK_D=block_dv_state,
                num_warps=2,
                num_stages=1,
            )

        profiled_bwd_call("flare_chunk_bwd_lse_dv_reduce", launch_bwd_dv_reduce)
    else:
        dV_bhtd = profiled_bwd_call(
            "flare_chunk_bwd_lse_dv_bypass",
            lambda: dV_part.squeeze(2).reshape(BH, num_chunks * chunk_size, Dv)[:, :N, :],
        )

    # Phase 4: backprop through the raw score definition score = scale * (Q @ K^T).
    #
    # At this point the FLARE-specific algebra is finished: `dS_enc` is the gradient with respect to the
    # pre-softmax scores for every token/query-row pair. The recurrent QK kernel therefore performs the
    # standard contractions:
    # - dK_chunk = dS_chunk @ Q * scale
    # - dQ      += dS_chunk^T @ K_chunk * scale
    #
    # dQ uses atomic adds because each chunk contributes to the same [H, M, D] tensor.
    def launch_bwd_recurrent_qk():
        return flare_chunk_bwd_recurrent_qk[(BH, num_chunks, num_qk_d_tiles)](
            K, Q,
            dS_enc,
            dQ, dK,
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            Q.stride(0), Q.stride(1), Q.stride(2),
            dS_enc.stride(0), dS_enc.stride(1), dS_enc.stride(2), dS_enc.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            BH, M, N, D, scale,
            CHUNK_SIZE=chunk_size,
            BLOCK_T=block_t_qk,
            BLOCK_D=block_d_qk,
            INPUT_PRECISION=ctx.input_precision,
            ACCUM_DK=False,
            H=H,
            num_warps=qk_num_warps,
            num_stages=qk_num_stages,
        )

    profiled_bwd_call("flare_chunk_bwd_recurrent_qk", launch_bwd_recurrent_qk)

    dQ_dec = None
    dK_dec = None
    if not weight_sharing_enc_dec:
        dK_dec_target = torch.zeros_like(K_dec_saved, dtype=torch.float32) if separate_K_dec else dQ
        dQ_dec_target = torch.zeros_like(Q_dec_saved, dtype=torch.float32) if separate_Q_dec else dK

        def launch_bwd_recurrent_qk_decode():
            return flare_chunk_bwd_recurrent_qk[(BH, num_chunks, num_qk_d_tiles)](
                Q_dec_saved,
                K_dec_saved,
                dS_dec,
                dK_dec_target,
                dQ_dec_target,
                Q_dec_saved.stride(0), Q_dec_saved.stride(1), Q_dec_saved.stride(2), Q_dec_saved.stride(3),
                K_dec_saved.stride(0), K_dec_saved.stride(1), K_dec_saved.stride(2),
                dS_dec.stride(0), dS_dec.stride(1), dS_dec.stride(2), dS_dec.stride(3),
                dK_dec_target.stride(0), dK_dec_target.stride(1), dK_dec_target.stride(2),
                dQ_dec_target.stride(0), dQ_dec_target.stride(1), dQ_dec_target.stride(2), dQ_dec_target.stride(3),
                BH, M, N, D, scale,
                CHUNK_SIZE=chunk_size,
                BLOCK_T=block_t_qk,
                BLOCK_D=block_d_qk,
                INPUT_PRECISION=ctx.input_precision,
                ACCUM_DK=not separate_Q_dec,
                H=H,
                num_warps=qk_num_warps,
                num_stages=qk_num_stages,
            )

        profiled_bwd_call("flare_chunk_bwd_recurrent_qk_decode", launch_bwd_recurrent_qk_decode)
        dQ_dec = dQ_dec_target if separate_Q_dec else None
        dK_dec = dK_dec_target if separate_K_dec else None

    dV = profiled_bwd_call(
        "grad_output_relayout",
        lambda: dV_bhtd.view(B, H, N, Dv).permute(0, 2, 1, 3).contiguous(),
    )
    dQ_out, dK_out, dV_out, dQ_dec_out, dK_dec_out = profiled_bwd_call(
        "grad_return_casts",
        lambda: (
            dQ.to(Q.dtype),
            dK.to(K.dtype),
            dV.to(V.dtype),
            None if dQ_dec is None else dQ_dec.to(Q_dec_saved.dtype),
            None if dK_dec is None else dK_dec.to(K_dec_saved.dtype),
        ),
    )
    if profile_data is not None:
        _refresh_profile_totals(profile_data)
    return dQ_out, dK_out, dV_out, None, None, None, None, dQ_dec_out, dK_dec_out


def flare_autoregressive_triton(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float | None = None,
    chunk_size: int | None = None,
    input_precision: str | None = None,
    profile: bool = False,
    Q_dec: torch.Tensor | None = None,
    K_dec: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, object]]:
    return AutoRegressiveFLARE.apply(Q, K, V, scale, chunk_size, input_precision, profile, Q_dec, K_dec)


# Backward-compatible aliases while downstream callers migrate.
ChunkedFLARE = AutoRegressiveFLARE
flare_chunk_triton = flare_autoregressive_triton

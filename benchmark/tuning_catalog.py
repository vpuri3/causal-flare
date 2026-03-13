from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class CandidateConfig:
    family: str
    name: str
    env: dict[str, str]


@dataclass(frozen=True)
class TunableParameter:
    name: str
    env_vars: tuple[str, ...]
    phases: tuple[str, ...]
    independent_of: tuple[str, ...]
    connected_to: tuple[str, ...]
    constraints: str
    notes: str


@dataclass(frozen=True)
class TuningCatalog:
    implementation: str
    summary: str
    family_groups: dict[str, tuple[str, ...]]
    parameters: tuple[TunableParameter, ...]

    def to_dict(self) -> dict:
        return {
            "implementation": self.implementation,
            "summary": self.summary,
            "family_groups": {key: list(value) for key, value in self.family_groups.items()},
            "parameters": [asdict(param) for param in self.parameters],
        }


CHUNKED_TUNING_FAMILY_GROUPS = {
    "fast_forward": (
        "default",
        "chunk_size",
        "forward_block_m",
    ),
    "fast_backward": (
        "default",
        "chunk_size",
        "backward_block_m",
    ),
    "core_fast": (
        "default",
        "chunk_size",
        "forward_block_m",
        "backward_block_m",
    ),
    "full_forward": (
        "default",
        "chunk_size",
        "forward_block_m",
        "forward_block_d",
        "forward_block_k",
        "forward_launch",
    ),
    "full_backward": (
        "default",
        "chunk_size",
        "backward_block_m",
        "backward_block_dv",
        "backward_block_k",
        "backward_block_d_part",
        "backward_qk_block_d",
        "backward_block_t_qk",
        "backward_block_t_replay",
        "backward_block_t_state",
        "backward_block_t_apply",
        "backward_scalar_apply_panel",
        "backward_fused_chunk_scan",
        "backward_launch",
    ),
    "core": (
        "default",
        "chunk_size",
        "forward_block_m",
        "forward_block_d",
        "forward_block_k",
        "forward_launch",
        "backward_block_m",
        "backward_block_k",
        "backward_qk_block_d",
        "backward_block_t_qk",
        "backward_launch",
    ),
    "extended": (
        "backward_block_dv",
        "backward_block_d_part",
        "backward_block_t_replay",
        "backward_block_t_state",
        "backward_block_t_apply",
        "backward_scalar_apply_panel",
        "backward_fused_chunk_scan",
    ),
}
CHUNKED_TUNING_FAMILY_GROUPS["all"] = (
    CHUNKED_TUNING_FAMILY_GROUPS["core"] + CHUNKED_TUNING_FAMILY_GROUPS["extended"] + ("combined",)
)

RECURRENT_TUNING_FAMILY_GROUPS = {
    "impl": (
        "default",
        "impl",
    ),
    "forward_core": (
        "block_m",
        "block_d",
        "block_k",
        "orig_block_t",
        "orig_launch",
    ),
    "forward_multi": (
        "multi_decode_block_t",
        "multi_assoc_block_t",
        "multi_replay_block_t",
        "multi_decode_launch",
        "multi_assoc_launch",
        "multi_replay_launch",
    ),
    "backward": (
        "bwd_block_t",
        "bwd_dg_launch",
        "bwd_dscore_launch",
        "bwd_dsz_launch",
        "bwd_qk_launch",
    ),
}
RECURRENT_TUNING_FAMILY_GROUPS["all"] = (
    RECURRENT_TUNING_FAMILY_GROUPS["impl"]
    + RECURRENT_TUNING_FAMILY_GROUPS["forward_core"]
    + RECURRENT_TUNING_FAMILY_GROUPS["forward_multi"]
    + RECURRENT_TUNING_FAMILY_GROUPS["backward"]
)


def power_of_two_values(limit: int) -> list[int]:
    values = []
    value = 16
    while value <= limit:
        values.append(value)
        value *= 2
    return values


def filter_divisors(values: Iterable[int], dim: int) -> list[int]:
    return [value for value in values if value <= dim and (dim % value) == 0]


def filter_tiles(values: Iterable[int], dim: int) -> list[int]:
    return [value for value in values if value <= dim]


def _filter_chunked_forward_chunk_sizes(values: Iterable[int], *, latent_queries: int, head_dim: int, seq_len: int) -> list[int]:
    valid = [value for value in values if value <= seq_len and (value % 16) == 0]
    # `CHUNK_SIZE=256` regularly spills over the shared-memory budget once the
    # forward working set reaches even moderate `M x D`. Keep it available only
    # for the smaller regimes where it is plausibly viable.
    if latent_queries > 32 or head_dim > 64:
        valid = [value for value in valid if value != 256]
    return valid


def _filter_forward_block_m_values(values: Iterable[int]) -> list[int]:
    # Wider forward M tiles have repeatedly fallen off the shared-memory cliff
    # in tuning smoke runs while providing no evidence of wins over 64.
    return [value for value in values if value <= 64]


def _filter_backward_block_m_values(values: Iterable[int]) -> list[int]:
    # Very large backward M tiles were consistently dominated in smoke runs and
    # can produce extremely slow scans/replays on long-context shapes.
    return [value for value in values if value <= 128]


def chunked_forward_launch_presets() -> tuple[CandidateConfig, ...]:
    return (
        CandidateConfig(
            family="forward_launch",
            name="forward_launch_legacy_8w3s",
            env={
                "FLARE_PREPARE_NUM_WARPS": "8",
                "FLARE_PREPARE_NUM_STAGES": "3",
                "FLARE_PREFIX_NUM_WARPS": "8",
                "FLARE_PREFIX_NUM_STAGES": "3",
                "FLARE_DECODER_NUM_WARPS": "8",
                "FLARE_DECODER_NUM_STAGES": "3",
                "FLARE_FWD_NUM_WARPS": "8",
                "FLARE_FWD_NUM_STAGES": "3",
            },
        ),
        CandidateConfig(
            family="forward_launch",
            name="forward_launch_balanced_4w2s_4w1s",
            env={
                "FLARE_PREPARE_NUM_WARPS": "4",
                "FLARE_PREPARE_NUM_STAGES": "2",
                "FLARE_PREFIX_NUM_WARPS": "4",
                "FLARE_PREFIX_NUM_STAGES": "2",
                "FLARE_DECODER_NUM_WARPS": "4",
                "FLARE_DECODER_NUM_STAGES": "1",
                "FLARE_FWD_NUM_WARPS": "4",
                "FLARE_FWD_NUM_STAGES": "1",
            },
        ),
        CandidateConfig(
            family="forward_launch",
            name="forward_launch_light_fwd_2w1s",
            env={
                "FLARE_PREPARE_NUM_WARPS": "4",
                "FLARE_PREPARE_NUM_STAGES": "2",
                "FLARE_PREFIX_NUM_WARPS": "4",
                "FLARE_PREFIX_NUM_STAGES": "2",
                "FLARE_DECODER_NUM_WARPS": "4",
                "FLARE_DECODER_NUM_STAGES": "1",
                "FLARE_FWD_NUM_WARPS": "2",
                "FLARE_FWD_NUM_STAGES": "1",
            },
        ),
        CandidateConfig(
            family="forward_launch",
            name="forward_launch_wide_prepare_8w3s",
            env={
                "FLARE_PREPARE_NUM_WARPS": "8",
                "FLARE_PREPARE_NUM_STAGES": "3",
                "FLARE_PREFIX_NUM_WARPS": "8",
                "FLARE_PREFIX_NUM_STAGES": "3",
                "FLARE_DECODER_NUM_WARPS": "4",
                "FLARE_DECODER_NUM_STAGES": "1",
                "FLARE_FWD_NUM_WARPS": "4",
                "FLARE_FWD_NUM_STAGES": "1",
            },
        ),
    )


def chunked_backward_launch_presets() -> tuple[CandidateConfig, ...]:
    return (
        CandidateConfig(
            family="backward_launch",
            name="backward_launch_legacy_8w3s",
            env={
                "FLARE_LSE_BWD_REPLAY_NUM_WARPS": "8",
                "FLARE_LSE_BWD_REPLAY_NUM_STAGES": "3",
                "FLARE_LSE_BWD_STATE_NUM_WARPS": "8",
                "FLARE_LSE_BWD_STATE_NUM_STAGES": "3",
                "FLARE_LSE_BWD_QK_NUM_WARPS": "8",
                "FLARE_LSE_BWD_QK_NUM_STAGES": "3",
                "FLARE_LSE_BWD_SCAN_APPLY_NUM_WARPS": "8",
                "FLARE_LSE_BWD_SCAN_APPLY_NUM_STAGES": "3",
            },
        ),
        CandidateConfig(
            family="backward_launch",
            name="backward_launch_balanced_4w2s",
            env={
                "FLARE_LSE_BWD_REPLAY_NUM_WARPS": "4",
                "FLARE_LSE_BWD_REPLAY_NUM_STAGES": "2",
                "FLARE_LSE_BWD_STATE_NUM_WARPS": "4",
                "FLARE_LSE_BWD_STATE_NUM_STAGES": "2",
                "FLARE_LSE_BWD_QK_NUM_WARPS": "4",
                "FLARE_LSE_BWD_QK_NUM_STAGES": "2",
                "FLARE_LSE_BWD_SCAN_APPLY_NUM_WARPS": "4",
                "FLARE_LSE_BWD_SCAN_APPLY_NUM_STAGES": "2",
            },
        ),
        CandidateConfig(
            family="backward_launch",
            name="backward_launch_light_4w1s",
            env={
                "FLARE_LSE_BWD_REPLAY_NUM_WARPS": "4",
                "FLARE_LSE_BWD_REPLAY_NUM_STAGES": "1",
                "FLARE_LSE_BWD_STATE_NUM_WARPS": "4",
                "FLARE_LSE_BWD_STATE_NUM_STAGES": "1",
                "FLARE_LSE_BWD_QK_NUM_WARPS": "4",
                "FLARE_LSE_BWD_QK_NUM_STAGES": "1",
                "FLARE_LSE_BWD_SCAN_APPLY_NUM_WARPS": "4",
                "FLARE_LSE_BWD_SCAN_APPLY_NUM_STAGES": "1",
            },
        ),
        CandidateConfig(
            family="backward_launch",
            name="backward_launch_compact_2w1s",
            env={
                "FLARE_LSE_BWD_REPLAY_NUM_WARPS": "2",
                "FLARE_LSE_BWD_REPLAY_NUM_STAGES": "1",
                "FLARE_LSE_BWD_STATE_NUM_WARPS": "2",
                "FLARE_LSE_BWD_STATE_NUM_STAGES": "1",
                "FLARE_LSE_BWD_QK_NUM_WARPS": "2",
                "FLARE_LSE_BWD_QK_NUM_STAGES": "1",
                "FLARE_LSE_BWD_SCAN_APPLY_NUM_WARPS": "2",
                "FLARE_LSE_BWD_SCAN_APPLY_NUM_STAGES": "1",
            },
        ),
    )


def build_chunked_family_candidates(
    *,
    head_dim: int,
    latent_queries: int,
    seq_len: int,
    chunk_sizes: tuple[int, ...],
    forward_block_ks: tuple[int, ...],
    backward_block_ks: tuple[int, ...],
    forward_block_ds: tuple[int, ...],
    backward_block_ds: tuple[int, ...],
    block_ts: tuple[int, ...],
    scalar_apply_panels: tuple[int, ...],
) -> dict[str, list[CandidateConfig]]:
    d = head_dim
    m = latent_queries
    n = seq_len
    forward_block_m_values = _filter_forward_block_m_values(power_of_two_values(m))
    backward_block_m_values = _filter_backward_block_m_values(power_of_two_values(m))
    valid_forward_block_d = filter_tiles(forward_block_ds, d)
    valid_backward_block_d = filter_tiles(backward_block_ds, d)
    valid_forward_block_k = filter_divisors(forward_block_ks, d)
    valid_backward_block_k = filter_divisors(backward_block_ks, d)
    valid_chunk_sizes = _filter_chunked_forward_chunk_sizes(chunk_sizes, latent_queries=m, head_dim=d, seq_len=n)
    valid_block_ts = [value for value in block_ts if value <= min(n, 512) and (value % 16) == 0]
    valid_scalar_apply_panels = [value for value in scalar_apply_panels if value <= min(128, n)]

    families: dict[str, list[CandidateConfig]] = {"default": [CandidateConfig("default", "default", {})]}
    families["chunk_size"] = [
        CandidateConfig("chunk_size", f"chunk_size_{chunk_size}", {"FLARE_CHUNK_SIZE": str(chunk_size)})
        for chunk_size in valid_chunk_sizes
    ]
    families["forward_block_m"] = [
        CandidateConfig(
            "forward_block_m",
            f"forward_block_m_{block_m}",
            {
                "FLARE_BLOCK_M": str(block_m),
                "FLARE_PREFIX_BLOCK_M": str(block_m),
            },
        )
        for block_m in forward_block_m_values
    ]
    families["forward_block_d"] = [
        CandidateConfig(
            "forward_block_d",
            f"forward_block_d_{block_d}",
            {
                "FLARE_PREPARE_BLOCK_D": str(block_d),
                "FLARE_PREFIX_BLOCK_D": str(block_d),
                "FLARE_FWD_BLOCK_D": str(block_d),
            },
        )
        for block_d in valid_forward_block_d
    ]
    families["forward_block_k"] = [
        CandidateConfig(
            "forward_block_k",
            f"forward_block_k_p{prepare_block_k}_f{fwd_block_k}",
            {
                "FLARE_PREPARE_BLOCK_K": str(prepare_block_k),
                "FLARE_FWD_BLOCK_K": str(fwd_block_k),
            },
        )
        for prepare_block_k in valid_forward_block_k
        for fwd_block_k in valid_forward_block_k
        if prepare_block_k >= fwd_block_k
    ]
    families["forward_launch"] = list(chunked_forward_launch_presets())

    backward_block_m_candidates: list[CandidateConfig] = []
    for block_m in backward_block_m_values:
        block_m_replay = min(block_m, 64)
        backward_block_m_candidates.append(
            CandidateConfig(
                "backward_block_m",
                f"backward_block_m_{block_m}_score_block_m_{block_m_replay}",
                {
                    "FLARE_LSE_BWD_BLOCK_M": str(block_m),
                    "FLARE_LSE_BWD_SCORE_BLOCK_M": str(block_m_replay),
                },
            )
        )
    families["backward_block_m"] = backward_block_m_candidates
    families["backward_block_dv"] = [
        CandidateConfig("backward_block_dv", f"backward_block_dv_{block_dv}", {"FLARE_LSE_BWD_BLOCK_DV": str(block_dv)})
        for block_dv in valid_backward_block_d
    ]
    families["backward_block_k"] = [
        CandidateConfig("backward_block_k", f"backward_block_k_{block_k}", {"FLARE_LSE_BWD_BLOCK_K": str(block_k)})
        for block_k in valid_backward_block_k
    ]
    families["backward_block_d_part"] = [
        CandidateConfig(
            "backward_block_d_part",
            f"backward_block_d_part_{block_d_part}",
            {"FLARE_LSE_BWD_BLOCK_D_PART": str(block_d_part)},
        )
        for block_d_part in valid_backward_block_d
    ]
    families["backward_qk_block_d"] = [
        CandidateConfig("backward_qk_block_d", f"backward_qk_block_d_{block_d}", {"FLARE_LSE_BWD_QK_BLOCK_D": str(block_d)})
        for block_d in valid_backward_block_d
    ]
    families["backward_block_t_qk"] = [
        CandidateConfig("backward_block_t_qk", f"backward_block_t_qk_{block_t}", {"FLARE_LSE_BWD_BLOCK_T_QK": str(block_t)})
        for block_t in valid_block_ts
    ]
    families["backward_block_t_replay"] = [
        CandidateConfig(
            "backward_block_t_replay",
            f"backward_block_t_replay_{block_t}",
            {"FLARE_LSE_BWD_BLOCK_T_REPLAY": str(block_t)},
        )
        for block_t in valid_block_ts
    ]
    families["backward_block_t_state"] = [
        CandidateConfig(
            "backward_block_t_state",
            f"backward_block_t_state_{block_t}",
            {"FLARE_LSE_BWD_BLOCK_T_STATE": str(block_t)},
        )
        for block_t in valid_block_ts
    ]
    families["backward_block_t_apply"] = [
        CandidateConfig(
            "backward_block_t_apply",
            f"backward_block_t_apply_{block_t}",
            {"FLARE_LSE_BWD_BLOCK_T_APPLY": str(block_t)},
        )
        for block_t in valid_block_ts
    ]
    families["backward_scalar_apply_panel"] = [
        CandidateConfig(
            "backward_scalar_apply_panel",
            f"backward_scalar_apply_panel_{panel}",
            {"FLARE_LSE_BWD_SCALAR_APPLY_PANEL": str(panel)},
        )
        for panel in valid_scalar_apply_panels
    ]
    families["backward_fused_chunk_scan"] = [
        CandidateConfig("backward_fused_chunk_scan", "backward_fused_chunk_scan_on", {"FLARE_LSE_BWD_FUSE_CHUNK_SCAN": "1"}),
        CandidateConfig("backward_fused_chunk_scan", "backward_fused_chunk_scan_off", {"FLARE_LSE_BWD_FUSE_CHUNK_SCAN": "0"}),
    ]
    families["backward_launch"] = list(chunked_backward_launch_presets())
    families["combined"] = []
    return families


def recurrent_launch_presets(
    *,
    family: str,
    phase: str,
    impl_env: dict[str, str] | None = None,
) -> tuple[CandidateConfig, ...]:
    base = {} if impl_env is None else dict(impl_env)
    return (
        CandidateConfig(family=family, name=f"{family}_default", env=base),
        CandidateConfig(
            family=family,
            name=f"{family}_4w1s",
            env=base | {
                f"FLARE_RECURRENT_{phase.upper()}_NUM_WARPS": "4",
                f"FLARE_RECURRENT_{phase.upper()}_NUM_STAGES": "1",
            },
        ),
        CandidateConfig(
            family=family,
            name=f"{family}_4w2s",
            env=base | {
                f"FLARE_RECURRENT_{phase.upper()}_NUM_WARPS": "4",
                f"FLARE_RECURRENT_{phase.upper()}_NUM_STAGES": "2",
            },
        ),
        CandidateConfig(
            family=family,
            name=f"{family}_8w2s",
            env=base | {
                f"FLARE_RECURRENT_{phase.upper()}_NUM_WARPS": "8",
                f"FLARE_RECURRENT_{phase.upper()}_NUM_STAGES": "2",
            },
        ),
    )


def build_recurrent_family_candidates(
    *,
    head_dim: int,
    latent_queries: int,
    seq_len: int,
    block_ds: tuple[int, ...],
    block_ks: tuple[int, ...],
    orig_block_ts: tuple[int, ...],
    multi_block_ts: tuple[int, ...],
    backward_block_ts: tuple[int, ...],
) -> dict[str, list[CandidateConfig]]:
    d = head_dim
    m = latent_queries
    n = seq_len
    valid_block_d = filter_tiles(block_ds, d)
    valid_block_k = filter_divisors(block_ks, d)
    valid_orig_block_t = [value for value in orig_block_ts if value == 1 or (value <= n and value in (16, 32))]
    valid_multi_block_t = [value for value in multi_block_ts if value <= min(n, 128) and value in (16, 32, 64, 128)]
    valid_backward_block_t = [value for value in backward_block_ts if value <= min(n, 128) and value in (16, 32, 64, 128)]

    families: dict[str, list[CandidateConfig]] = {"default": [CandidateConfig("default", "default", {})]}
    families["impl"] = [
        CandidateConfig("impl", "impl_orig", {}),
        CandidateConfig("impl", "impl_assoc_scan", {"FLARE_RECURRENT_ASSOC_SCAN": "1"}),
        CandidateConfig("impl", "impl_multi", {"FLARE_RECURRENT_IMPL": "multi"}),
    ]
    families["block_m"] = [
        CandidateConfig("block_m", f"block_m_{block_m}", {"FLARE_RECURRENT_BLOCK_M": str(block_m)})
        for block_m in power_of_two_values(m)
    ]
    families["block_d"] = [
        CandidateConfig("block_d", f"block_d_{block_d}", {"FLARE_RECURRENT_BLOCK_D": str(block_d)})
        for block_d in valid_block_d
    ]
    families["block_k"] = [
        CandidateConfig("block_k", f"block_k_{block_k}", {"FLARE_RECURRENT_BLOCK_K": str(block_k)})
        for block_k in valid_block_k
    ]
    families["orig_block_t"] = [
        CandidateConfig("orig_block_t", f"orig_block_t_{block_t}", {"FLARE_RECURRENT_BLOCK_T": str(block_t)})
        for block_t in valid_orig_block_t
    ]
    families["orig_launch"] = list(recurrent_launch_presets(family="orig_launch", phase="fwd"))
    multi_base = {"FLARE_RECURRENT_IMPL": "multi"}
    families["multi_decode_block_t"] = [
        CandidateConfig(
            "multi_decode_block_t",
            f"multi_decode_block_t_{block_t}",
            multi_base | {"FLARE_RECURRENT_MULTI_DECODE_BLOCK_T": str(block_t)},
        )
        for block_t in valid_multi_block_t
    ]
    families["multi_assoc_block_t"] = [
        CandidateConfig(
            "multi_assoc_block_t",
            f"multi_assoc_block_t_{block_t}",
            multi_base | {"FLARE_RECURRENT_MULTI_ASSOC_LSE_BLOCK_T": str(block_t)},
        )
        for block_t in valid_multi_block_t
    ]
    families["multi_replay_block_t"] = [
        CandidateConfig(
            "multi_replay_block_t",
            f"multi_replay_block_t_{block_t}",
            multi_base | {"FLARE_RECURRENT_MULTI_REPLAY_BLOCK_T": str(block_t)},
        )
        for block_t in valid_multi_block_t
    ]
    families["multi_decode_launch"] = list(recurrent_launch_presets(family="multi_decode_launch", phase="decode", impl_env=multi_base))
    families["multi_assoc_launch"] = list(recurrent_launch_presets(family="multi_assoc_launch", phase="assoc_lse", impl_env=multi_base))
    families["multi_replay_launch"] = list(recurrent_launch_presets(family="multi_replay_launch", phase="replay", impl_env=multi_base))
    families["bwd_block_t"] = [
        CandidateConfig("bwd_block_t", f"bwd_block_t_{block_t}", {"FLARE_RECURRENT_BWD_BLOCK_T": str(block_t)})
        for block_t in valid_backward_block_t
    ]
    families["bwd_dg_launch"] = list(recurrent_launch_presets(family="bwd_dg_launch", phase="bwd_dg"))
    families["bwd_dscore_launch"] = list(recurrent_launch_presets(family="bwd_dscore_launch", phase="bwd_dscore"))
    families["bwd_dsz_launch"] = list(recurrent_launch_presets(family="bwd_dsz_launch", phase="bwd_dsz"))
    families["bwd_qk_launch"] = list(recurrent_launch_presets(family="bwd_qk_launch", phase="bwd_qk"))
    return families


def get_recurrent_tuning_catalog() -> TuningCatalog:
    return TuningCatalog(
        implementation="RecurrentFLARE",
        summary="Launch knobs for the original fused recurrent path plus the experimental multi-kernel forward/backward phases.",
        family_groups=RECURRENT_TUNING_FAMILY_GROUPS,
        parameters=(
            TunableParameter(
                name="latent tile",
                env_vars=("FLARE_RECURRENT_BLOCK_M",),
                phases=("orig_fwd", "multi_decode", "multi_lse_enc", "multi_output", "backward"),
                independent_of=("all phase-local BLOCK_T knobs",),
                connected_to=("num_m_tiles", "base num_warps heuristic", "atomic vs single-tile output behavior"),
                constraints="Multiple of 16 and <= M.",
                notes="Shared latent tiling across nearly every recurrent kernel. This is the main cross-phase coupling knob.",
            ),
            TunableParameter(
                name="value/output tile",
                env_vars=("FLARE_RECURRENT_BLOCK_D",),
                phases=("orig_fwd", "multi_output", "bwd_dg", "bwd_dsz"),
                independent_of=("decode/assoc BLOCK_T knobs",),
                connected_to=("FLARE_RECURRENT_BLOCK_K", "output grid over D"),
                constraints="Multiple of 16 and <= D.",
                notes="Controls output-vector tiling. It is global today, so changing it affects both orig and multi output kernels.",
            ),
            TunableParameter(
                name="score reduction tile",
                env_vars=("FLARE_RECURRENT_BLOCK_K",),
                phases=("orig_fwd", "multi_decode", "multi_lse_enc", "multi_output", "backward"),
                independent_of=("launch warps/stages",),
                connected_to=("FLARE_RECURRENT_BLOCK_D", "D divisibility", "score recompute cost across all kernels"),
                constraints="Multiple of 16 and <= D.",
                notes="Shared score dot-product reduction tile. This is connected across every phase that recomputes QK scores.",
            ),
            TunableParameter(
                name="orig fused token tile",
                env_vars=("FLARE_RECURRENT_BLOCK_T",),
                phases=("orig_fwd",),
                independent_of=("FLARE_RECURRENT_MULTI_*_BLOCK_T",),
                connected_to=("orig launch shape",),
                constraints="Allowed values: 1, 16, 32.",
                notes="Only affects the original fused recurrent forward. It is intentionally decoupled from the multi-kernel token tiles.",
            ),
            TunableParameter(
                name="multi decode token tile",
                env_vars=("FLARE_RECURRENT_MULTI_DECODE_BLOCK_T",),
                phases=("multi_decode",),
                independent_of=("FLARE_RECURRENT_MULTI_ASSOC_LSE_BLOCK_T", "FLARE_RECURRENT_MULTI_REPLAY_BLOCK_T"),
                connected_to=("decode grid over N",),
                constraints="Allowed values: 16, 32, 64, 128.",
                notes="Independent stage-1 knob for LSE_DEC. This should be tuned separately from the other multi kernels.",
            ),
            TunableParameter(
                name="multi associative LSE token tile",
                env_vars=("FLARE_RECURRENT_MULTI_ASSOC_LSE_BLOCK_T",),
                phases=("multi_lse_enc",),
                independent_of=("FLARE_RECURRENT_MULTI_DECODE_BLOCK_T", "FLARE_RECURRENT_MULTI_REPLAY_BLOCK_T"),
                connected_to=("assoc-scan working-set width",),
                constraints="Allowed values: 16, 32, 64, 128.",
                notes="Independent stage-2 knob for LSE_ENC. Larger values increase scan width and register/shared-memory pressure.",
            ),
            TunableParameter(
                name="multi output token tile",
                env_vars=("FLARE_RECURRENT_MULTI_REPLAY_BLOCK_T",),
                phases=("multi_output",),
                independent_of=("FLARE_RECURRENT_MULTI_DECODE_BLOCK_T", "FLARE_RECURRENT_MULTI_ASSOC_LSE_BLOCK_T"),
                connected_to=("local causal W tile size", "prefix carry update cost"),
                constraints="Allowed values: 16, 32, 64, 128.",
                notes="Independent stage-3 knob for the dense masked output kernel. This is currently the main performance driver in multi forward.",
            ),
            TunableParameter(
                name="phase launch controls",
                env_vars=(
                    "FLARE_RECURRENT_NUM_WARPS",
                    "FLARE_RECURRENT_NUM_STAGES",
                    "FLARE_RECURRENT_DECODE_NUM_WARPS",
                    "FLARE_RECURRENT_DECODE_NUM_STAGES",
                    "FLARE_RECURRENT_FWD_NUM_WARPS",
                    "FLARE_RECURRENT_FWD_NUM_STAGES",
                    "FLARE_RECURRENT_ASSOC_LSE_NUM_WARPS",
                    "FLARE_RECURRENT_ASSOC_LSE_NUM_STAGES",
                    "FLARE_RECURRENT_REPLAY_NUM_WARPS",
                    "FLARE_RECURRENT_REPLAY_NUM_STAGES",
                ),
                phases=("orig_fwd", "multi_decode", "multi_lse_enc", "multi_output"),
                independent_of=("BLOCK_T choices for unrelated phases",),
                connected_to=("phase-specific resource limits", "register pressure from BLOCK_M/BLOCK_D/BLOCK_T"),
                constraints="Positive integers understood by Triton launch metadata.",
                notes="Global defaults exist, but phase-specific overrides are independent and should be tuned per kernel family.",
            ),
            TunableParameter(
                name="backward token tile",
                env_vars=("FLARE_RECURRENT_BWD_BLOCK_T",),
                phases=("bwd_dg", "bwd_dscore", "bwd_qk"),
                independent_of=("multi forward BLOCK_T knobs",),
                connected_to=("backward launch shapes", "score-replay panel width"),
                constraints="Allowed values currently follow multi block-T validation: 16, 32, 64, 128.",
                notes="Shared backward token-panel knob. Unlike multi forward, recurrent backward still uses one shared BLOCK_T across several kernels.",
            ),
            TunableParameter(
                name="backward phase launch controls",
                env_vars=(
                    "FLARE_RECURRENT_BWD_DG_NUM_WARPS",
                    "FLARE_RECURRENT_BWD_DG_NUM_STAGES",
                    "FLARE_RECURRENT_BWD_DSCORE_NUM_WARPS",
                    "FLARE_RECURRENT_BWD_DSCORE_NUM_STAGES",
                    "FLARE_RECURRENT_BWD_DSZ_NUM_WARPS",
                    "FLARE_RECURRENT_BWD_DSZ_NUM_STAGES",
                    "FLARE_RECURRENT_BWD_QK_NUM_WARPS",
                    "FLARE_RECURRENT_BWD_QK_NUM_STAGES",
                ),
                phases=("bwd_dg", "bwd_dscore", "bwd_dsz", "bwd_qk"),
                independent_of=("forward-only launch knobs",),
                connected_to=("shared BLOCK_M/BLOCK_D/BLOCK_K choices",),
                constraints="Positive integers understood by Triton launch metadata.",
                notes="These launches are phase-local but inherit pressure from the shared backward tiling knobs.",
            ),
        ),
    )


def get_chunked_tuning_catalog() -> TuningCatalog:
    return TuningCatalog(
        implementation="AutoRegressiveFLARE",
        summary="Forward and backward launch knobs used by the chunked tuning matrix sweep, grouped by independently sweepable families.",
        family_groups=CHUNKED_TUNING_FAMILY_GROUPS,
        parameters=(
            TunableParameter(
                name="chunk size",
                env_vars=("FLARE_CHUNK_SIZE",),
                phases=("forward", "backward"),
                independent_of=("most launch metadata presets",),
                connected_to=("all chunk-local kernel grids", "numerical work per chunk"),
                constraints="Positive multiple of 16 and <= N in the current tuner.",
                notes="Top-level structural knob. It changes both kernel launch counts and per-kernel arithmetic intensity.",
            ),
            TunableParameter(
                name="forward latent tiles",
                env_vars=("FLARE_BLOCK_M", "FLARE_PREFIX_BLOCK_M"),
                phases=("prepare", "prefix", "decoder", "forward"),
                independent_of=("backward-only block-T families",),
                connected_to=("each other", "num_m_tiles", "forward launch presets"),
                constraints="Multiples of 16, usually swept together as powers of two up to M.",
                notes="These two are intentionally coupled in the matrix tuner today because the prefix and forward latent tiling usually move together.",
            ),
            TunableParameter(
                name="forward D tiles",
                env_vars=("FLARE_PREPARE_BLOCK_D", "FLARE_PREFIX_BLOCK_D", "FLARE_FWD_BLOCK_D"),
                phases=("prepare", "prefix", "forward"),
                independent_of=("backward block-D families",),
                connected_to=("each other", "forward launch presets"),
                constraints="Tile values <= D.",
                notes="The current sweep treats these as one family because those kernels share the same value/output dimension and usually want similar D tiles.",
            ),
            TunableParameter(
                name="forward K tiles",
                env_vars=("FLARE_PREPARE_BLOCK_K", "FLARE_FWD_BLOCK_K"),
                phases=("prepare", "forward"),
                independent_of=("forward block-D family",),
                connected_to=("D divisibility", "prepare_block_k >= fwd_block_k in the current tuner"),
                constraints="Divisors of D.",
                notes="Connected pair for score-reduction tiles. The tuner preserves the empirical ordering constraint `prepare >= fwd`.",
            ),
            TunableParameter(
                name="forward launch presets",
                env_vars=(
                    "FLARE_PREPARE_NUM_WARPS",
                    "FLARE_PREPARE_NUM_STAGES",
                    "FLARE_PREFIX_NUM_WARPS",
                    "FLARE_PREFIX_NUM_STAGES",
                    "FLARE_DECODER_NUM_WARPS",
                    "FLARE_DECODER_NUM_STAGES",
                    "FLARE_FWD_NUM_WARPS",
                    "FLARE_FWD_NUM_STAGES",
                ),
                phases=("prepare", "prefix", "decoder", "forward"),
                independent_of=("backward launch presets",),
                connected_to=("forward block_m/block_d/block_k choices",),
                constraints="Positive integers consumed by Triton launches.",
                notes="The shared tuner keeps these as presets because the phase launches are correlated in practice.",
            ),
            TunableParameter(
                name="backward latent tiles",
                env_vars=("FLARE_LSE_BWD_BLOCK_M", "FLARE_LSE_BWD_SCORE_BLOCK_M"),
                phases=("bwd_replay", "bwd_state", "bwd_qk"),
                independent_of=("forward families",),
                connected_to=("each other", "`score_block_m <= block_m` heuristic"),
                constraints="Multiples of 16.",
                notes="Coupled family in the tuner because score replay uses a narrower latent tile than the main backward state kernels.",
            ),
            TunableParameter(
                name="backward D and K tiles",
                env_vars=(
                    "FLARE_LSE_BWD_BLOCK_DV",
                    "FLARE_LSE_BWD_BLOCK_D_PART",
                    "FLARE_LSE_BWD_QK_BLOCK_D",
                    "FLARE_LSE_BWD_BLOCK_K",
                ),
                phases=("bwd_replay", "bwd_state", "bwd_qk", "bwd_scan_apply"),
                independent_of=("backward block-T families",),
                connected_to=("head-dim divisibility", "backward launch presets"),
                constraints="Current tuner sweeps each family independently with values <= D or dividing D.",
                notes="These are logically separate families in the matrix sweep even though all of them affect backward score/state recomputation cost.",
            ),
            TunableParameter(
                name="backward token-panel tiles",
                env_vars=(
                    "FLARE_LSE_BWD_BLOCK_T_QK",
                    "FLARE_LSE_BWD_BLOCK_T_REPLAY",
                    "FLARE_LSE_BWD_BLOCK_T_STATE",
                    "FLARE_LSE_BWD_BLOCK_T_APPLY",
                ),
                phases=("bwd_qk", "bwd_replay", "bwd_state", "bwd_apply"),
                independent_of=("one another in the tuner",),
                connected_to=("chunk size", "tl.dot minimum panel constraints in replay/apply"),
                constraints="Multiples of 16; some kernels impose additional minimums such as replay/apply >= 16.",
                notes="These are intentionally independent families because each backward phase stresses a different token-panel shape.",
            ),
            TunableParameter(
                name="backward apply mode toggles",
                env_vars=("FLARE_LSE_BWD_SCALAR_APPLY_PANEL", "FLARE_LSE_BWD_FUSE_CHUNK_SCAN"),
                phases=("bwd_apply", "bwd_scan_apply"),
                independent_of=("forward families",),
                connected_to=("backward_block_t_apply", "scan/apply launch presets"),
                constraints="Panel values are positive; fused flag is 0/1.",
                notes="Small structural toggles that materially change the apply kernel decomposition.",
            ),
            TunableParameter(
                name="backward launch presets",
                env_vars=(
                    "FLARE_LSE_BWD_REPLAY_NUM_WARPS",
                    "FLARE_LSE_BWD_REPLAY_NUM_STAGES",
                    "FLARE_LSE_BWD_STATE_NUM_WARPS",
                    "FLARE_LSE_BWD_STATE_NUM_STAGES",
                    "FLARE_LSE_BWD_QK_NUM_WARPS",
                    "FLARE_LSE_BWD_QK_NUM_STAGES",
                    "FLARE_LSE_BWD_SCAN_APPLY_NUM_WARPS",
                    "FLARE_LSE_BWD_SCAN_APPLY_NUM_STAGES",
                ),
                phases=("bwd_replay", "bwd_state", "bwd_qk", "bwd_scan_apply"),
                independent_of=("forward launch presets",),
                connected_to=("backward block_m/block_t/block_d families",),
                constraints="Positive integers consumed by Triton launches.",
                notes="Kept as correlated presets for the matrix sweep, but the individual phase env vars remain available for narrower follow-up tuning.",
            ),
        ),
    )


def get_all_tuning_catalogs() -> tuple[TuningCatalog, ...]:
    return (get_recurrent_tuning_catalog(), get_chunked_tuning_catalog())

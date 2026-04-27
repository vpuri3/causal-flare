"""Triton path for SSD rank-{1,4} autoregressive chunk-prefix state scans."""

# -----------------------------------------------------------------------------------------------
# Single-Step Decode Timing + Accuracy Snapshot (2026-04-23)
# Shape: B=32, H=32, N=1, M=64, D=64
# Dtype: C/W*/V*/log_alpha=bf16, state=fp32, output=bf16
# Oracle: ssd_rank4_token_loop_oracle (bf16 path)
#
# rank | oracle_ms | pytorch_ms | triton_ms | pytorch_y_rel_l2 | triton_y_rel_l2
# 1    | 0.141     | 0.159      | 0.130     | 0.00350121       | 0.00350122
# 2    | 0.156     | 0.196      | 0.143     | 0.00383046       | 0.00383046
# 3    | 0.182     | 0.232      | 0.147     | 0.00401724       | 0.00401721
# 4    | 0.195     | 0.272      | 0.155     | 0.00424435       | 0.00424447
# state_rel_l2 by rank (pytorch ~= triton):
# 1: 0.00267819, 2: 0.00302500, 3: 0.00329586, 4: 0.00352702
# -----------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------------
# Phase Timing Snapshot (ms) for ssd_rank4_triton, averaged over 8 iterations (3 warmup)
# Shape: B=32, H=32, N=2048, M=64, D=64, CHUNK_SIZE=64
# Dtype: bf16 forward, fp32 backward (with bf16 tensor-core dot inputs)
#
# rank | fwd_p1 | fwd_p2 | fwd_p3 | fwd_total | bwd_p3 | bwd_p2 | bwd_p1 | bwd_total | step_total
# 1    | 0.742  | 0.452  | 0.870  | 2.064     | 3.668  | 0.620  | 1.007  | 5.295     | 7.359
# 2    | 1.157  | 0.446  | 1.158  | 2.762     | 5.363  | 0.618  | 2.444  | 8.425     | 11.187
# 3    | 1.700  | 0.448  | 1.761  | 3.910     | 6.680  | 0.617  | 3.777  | 11.074    | 14.983
# 4    | 2.134  | 0.449  | 2.941  | 5.524     | 8.945  | 0.617  | 4.982  | 14.545    | 20.069
# -----------------------------------------------------------------------------------------------
#
# Global Relative Error Snapshot (L2): ||ref - test|| / ||ref||
# Shape: B=1, H=8, N=1024, M=64, D=64, CHUNK_SIZE=64
# Dtype: bf16 forward, fp32 backward (with bf16 tensor-core dot inputs)
# Oracle: ssd_rank4_token_loop_oracle, Test: ssd_rank4_triton
#
# rank | y_rel_l2  | state_rel_l2 | grad_rel_l2_global | dlog_rel_l2
# 1    | 0.00421616| 0.00314452   | 0.00447197         | 0.00599766
# 2    | 0.00455039| 0.00376819   | 0.00456141         | 0.00645272
# 3    | 0.00478571| 0.00384032   | 0.00464509         | 0.00653440
# 4    | 0.00497253| 0.00432699   | 0.00472132         | 0.00669059
# -----------------------------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

import triton
import triton.language as tl

_SUPPORTED_M_VALUES = {16, 32, 64, 96, 128}
_SUPPORTED_D_VALUES = {32, 64, 128}
_SUPPORTED_CHUNK_SIZES = {32, 64, 128, 256}
_SUPPORTED_BLOCK_T_VALUES = (16, 8, 4, 2, 1)
_SUPPORTED_BLOCK_X_VALUES = (128, 64, 32, 16)
_SUPPORTED_PHASE1_BLOCK_T_VALUES = {1, 2, 4, 8, 16}
_EXPERIMENTAL_TRITON_ALLOCATOR_SET = False
_INV_LN2 = 1.4426950408889634

# Reusable buffers for Phase-3 backward intermediates to reduce allocation overhead.
_PHASE3_BWD_WORKSPACE: dict[tuple, torch.Tensor] = {}


@dataclass(frozen=True)
class _Phase3ForwardLaunchConfig:
    block_m: int
    block_d: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class _Phase3BackwardLaunchConfig:
    block_m: int
    block_d: int
    fused_a1a2_num_warps: int
    fused_off_num_warps: int
    num_stages: int
    fused_dv_num_warps: int
    fused_main_num_warps: int
    fused_dv_num_stages: int
    fused_main_num_stages: int
    fused_off_num_stages: int


@dataclass(frozen=True)
class _Phase1BackwardLaunchConfig:
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class _Phase2LaunchConfig:
    block_md: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class _DecodeLaunchConfig:
    block_m: int
    block_d: int
    num_warps: int
    num_stages: int


_SSD_RANK4_DECODE_TUNED: dict[tuple[int, int, int], _DecodeLaunchConfig] = {
    # key: (BH, M, D)
    (256, 64, 64): _DecodeLaunchConfig(block_m=64, block_d=64, num_warps=2, num_stages=2),
    (1024, 64, 64): _DecodeLaunchConfig(block_m=64, block_d=64, num_warps=2, num_stages=2),
    (256, 96, 64): _DecodeLaunchConfig(block_m=32, block_d=64, num_warps=2, num_stages=2),
    (1024, 96, 64): _DecodeLaunchConfig(block_m=32, block_d=64, num_warps=2, num_stages=2),
    (256, 128, 64): _DecodeLaunchConfig(block_m=64, block_d=64, num_warps=2, num_stages=2),
    (1024, 128, 64): _DecodeLaunchConfig(block_m=64, block_d=64, num_warps=2, num_stages=2),
}


@dataclass(frozen=True)
class _StaticSsdRank1ShapeConfig:
    chunk_size: int
    input_dtype: torch.dtype
    input_precision: str
    has_initial_state: bool
    return_final_state: bool
    phase1_block_t: int
    phase1_block_m: int
    phase1_block_d: int
    phase2_block_nc: int
    phase2_launch: _Phase2LaunchConfig
    phase3_forward: _Phase3ForwardLaunchConfig
    phase3_backward: _Phase3BackwardLaunchConfig
    phase1_backward: _Phase1BackwardLaunchConfig


@dataclass
class _StaticSsdRank1Workspace:
    s_local_end_md: torch.Tensor
    phase2_init_dummy: torch.Tensor
    phase2_chunk_start: torch.Tensor
    phase3_s0_md_fwd: torch.Tensor
    phase3_s0_md_bwd: torch.Tensor
    phase3_dlog_fp32: torch.Tensor
    phase3_dS0: torch.Tensor
    phase2_dlog_per_chunk: torch.Tensor
    phase2_dinit: torch.Tensor
    phase2_final_replay: torch.Tensor
    dlog_chunk_accum: torch.Tensor
    phase2_grad_final_zero: torch.Tensor


def _make_static_cfg(
    *,
    chunk_size: int,
    phase1_block_t: int,
    phase1_block_m: int,
    phase1_block_d: int,
    phase2_block_nc: int,
    phase2_block_md: int,
    phase2_num_warps: int,
    phase2_num_stages: int,
    phase3_fwd_block_m: int,
    phase3_fwd_block_d: int,
    phase3_fwd_num_warps: int,
    phase3_fwd_num_stages: int,
    phase3_bwd_block_m: int,
    phase3_bwd_block_d: int,
    phase3_bwd_num_warps: int,
    phase3_bwd_num_stages: int,
    phase1_bwd_num_warps: int,
    phase1_bwd_num_stages: int,
    phase3_bwd_dv_num_warps: int | None = None,
    phase3_bwd_main_num_warps: int | None = None,
    phase3_bwd_off_num_warps: int | None = None,
    phase3_bwd_dv_num_stages: int | None = None,
    phase3_bwd_main_num_stages: int | None = None,
    phase3_bwd_off_num_stages: int | None = None,
) -> _StaticSsdRank1ShapeConfig:
    return _StaticSsdRank1ShapeConfig(
        chunk_size=chunk_size,
        input_dtype=torch.bfloat16,
        input_precision="tf32",
        has_initial_state=False,
        return_final_state=True,
        phase1_block_t=phase1_block_t,
        phase1_block_m=phase1_block_m,
        phase1_block_d=phase1_block_d,
        phase2_block_nc=phase2_block_nc,
        phase2_launch=_Phase2LaunchConfig(
            block_md=phase2_block_md,
            num_warps=phase2_num_warps,
            num_stages=phase2_num_stages,
        ),
        phase3_forward=_Phase3ForwardLaunchConfig(
            block_m=phase3_fwd_block_m,
            block_d=phase3_fwd_block_d,
            num_warps=phase3_fwd_num_warps,
            num_stages=phase3_fwd_num_stages,
        ),
        phase3_backward=_Phase3BackwardLaunchConfig(
            block_m=phase3_bwd_block_m,
            block_d=phase3_bwd_block_d,
            fused_a1a2_num_warps=phase3_bwd_main_num_warps
            if phase3_bwd_main_num_warps is not None
            else phase3_bwd_num_warps,
            fused_off_num_warps=phase3_bwd_off_num_warps
            if phase3_bwd_off_num_warps is not None
            else phase3_bwd_num_warps,
            num_stages=phase3_bwd_num_stages,
            fused_dv_num_warps=phase3_bwd_dv_num_warps
            if phase3_bwd_dv_num_warps is not None
            else phase3_bwd_num_warps,
            fused_main_num_warps=phase3_bwd_main_num_warps
            if phase3_bwd_main_num_warps is not None
            else phase3_bwd_num_warps,
            fused_dv_num_stages=phase3_bwd_dv_num_stages
            if phase3_bwd_dv_num_stages is not None
            else phase3_bwd_num_stages,
            fused_main_num_stages=phase3_bwd_main_num_stages
            if phase3_bwd_main_num_stages is not None
            else phase3_bwd_num_stages,
            fused_off_num_stages=phase3_bwd_off_num_stages
            if phase3_bwd_off_num_stages is not None
            else phase3_bwd_num_stages,
        ),
        phase1_backward=_Phase1BackwardLaunchConfig(
            num_warps=phase1_bwd_num_warps,
            num_stages=phase1_bwd_num_stages,
        ),
    )


_STATIC_SSD_RANK1_SHAPE_CONFIGS: dict[tuple[int, int, int], _StaticSsdRank1ShapeConfig] = {
    # key: (N, M, D). BH is intentionally ignored for static launch-config selection.
    (2048, 16, 32): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=16,
        phase1_block_d=32,
        phase2_block_nc=32,
        phase2_block_md=64,
        phase2_num_warps=2,
        phase2_num_stages=2,
        phase3_fwd_block_m=16,
        phase3_fwd_block_d=32,
        phase3_fwd_num_warps=2,
        phase3_fwd_num_stages=2,
        phase3_bwd_block_m=16,
        phase3_bwd_block_d=32,
        phase3_bwd_num_warps=2,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=2,
        phase1_bwd_num_stages=2,
    ),
    (2048, 16, 64): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=16,
        phase1_block_d=64,
        phase2_block_nc=32,
        phase2_block_md=128,
        phase2_num_warps=4,
        phase2_num_stages=2,
        phase3_fwd_block_m=16,
        phase3_fwd_block_d=64,
        phase3_fwd_num_warps=2,
        phase3_fwd_num_stages=2,
        phase3_bwd_block_m=16,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=2,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=2,
        phase1_bwd_num_stages=2,
    ),
    (2048, 16, 128): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=16,
        phase1_block_d=128,
        phase2_block_nc=32,
        phase2_block_md=256,
        phase2_num_warps=4,
        phase2_num_stages=3,
        phase3_fwd_block_m=16,
        phase3_fwd_block_d=128,
        phase3_fwd_num_warps=2,
        phase3_fwd_num_stages=2,
        phase3_bwd_block_m=16,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=2,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=2,
        phase1_bwd_num_stages=2,
    ),
    (2048, 64, 64): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=64,
        phase1_block_d=64,
        phase2_block_nc=32,
        phase2_block_md=256,
        phase2_num_warps=4,
        phase2_num_stages=2,
        phase3_fwd_block_m=64,
        phase3_fwd_block_d=64,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=4,
        phase3_bwd_block_m=64,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=3,
    ),
    (2048, 64, 128): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=64,
        phase1_block_d=128,
        phase2_block_nc=32,
        phase2_block_md=512,
        phase2_num_warps=8,
        phase2_num_stages=3,
        phase3_fwd_block_m=64,
        phase3_fwd_block_d=128,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=2,
        phase3_bwd_block_m=64,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=2,
    ),
    (1024, 64, 64): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=64,
        phase1_block_d=64,
        phase2_block_nc=16,
        phase2_block_md=256,
        phase2_num_warps=4,
        phase2_num_stages=3,
        phase3_fwd_block_m=64,
        phase3_fwd_block_d=64,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=4,
        phase3_bwd_block_m=64,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=3,
    ),
    (1024, 64, 128): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=64,
        phase1_block_d=128,
        phase2_block_nc=16,
        phase2_block_md=512,
        phase2_num_warps=8,
        phase2_num_stages=3,
        phase3_fwd_block_m=64,
        phase3_fwd_block_d=128,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=2,
        phase3_bwd_block_m=64,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=2,
    ),
    (2048, 32, 32): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=32,
        phase1_block_d=32,
        phase2_block_nc=32,
        phase2_block_md=128,
        phase2_num_warps=4,
        phase2_num_stages=2,
        phase3_fwd_block_m=32,
        phase3_fwd_block_d=32,
        phase3_fwd_num_warps=2,
        phase3_fwd_num_stages=3,
        phase3_bwd_block_m=32,
        phase3_bwd_block_d=32,
        phase3_bwd_num_warps=2,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=2,
        phase1_bwd_num_stages=2,
    ),
    (2048, 32, 64): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=32,
        phase1_block_d=64,
        phase2_block_nc=32,
        phase2_block_md=256,
        phase2_num_warps=4,
        phase2_num_stages=3,
        phase3_fwd_block_m=32,
        phase3_fwd_block_d=64,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=3,
        phase3_bwd_block_m=32,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=2,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=2,
    ),
    (2048, 32, 128): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=32,
        phase1_block_d=128,
        phase2_block_nc=32,
        phase2_block_md=512,
        phase2_num_warps=8,
        phase2_num_stages=3,
        phase3_fwd_block_m=32,
        phase3_fwd_block_d=128,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=2,
        phase3_bwd_block_m=32,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=2,
    ),
    (2048, 64, 32): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=64,
        phase1_block_d=32,
        phase2_block_nc=32,
        phase2_block_md=256,
        phase2_num_warps=4,
        phase2_num_stages=3,
        phase3_fwd_block_m=64,
        phase3_fwd_block_d=32,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=4,
        phase3_bwd_block_m=64,
        phase3_bwd_block_d=32,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=3,
    ),
    (2048, 96, 32): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=32,
        phase1_block_d=32,
        phase2_block_nc=16,
        phase2_block_md=256,
        phase2_num_warps=4,
        phase2_num_stages=3,
        phase3_fwd_block_m=32,
        phase3_fwd_block_d=32,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=3,
        phase3_bwd_block_m=32,
        phase3_bwd_block_d=32,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=3,
    ),
    (2048, 96, 64): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=32,
        phase1_block_d=64,
        phase2_block_nc=16,
        phase2_block_md=256,
        phase2_num_warps=8,
        phase2_num_stages=3,
        phase3_fwd_block_m=32,
        phase3_fwd_block_d=64,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=3,
        phase3_bwd_block_m=32,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=3,
    ),
    (2048, 96, 128): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=32,
        phase1_block_d=128,
        phase2_block_nc=16,
        phase2_block_md=512,
        phase2_num_warps=8,
        phase2_num_stages=3,
        phase3_fwd_block_m=32,
        phase3_fwd_block_d=128,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=2,
        phase3_bwd_block_m=32,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=2,
    ),
    (2048, 128, 32): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=64,
        phase1_block_d=32,
        phase2_block_nc=16,
        phase2_block_md=512,
        phase2_num_warps=8,
        phase2_num_stages=3,
        phase3_fwd_block_m=64,
        phase3_fwd_block_d=32,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=3,
        phase3_bwd_block_m=64,
        phase3_bwd_block_d=32,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=3,
    ),
    (2048, 128, 64): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=64,
        phase1_block_d=64,
        phase2_block_nc=16,
        phase2_block_md=512,
        phase2_num_warps=8,
        phase2_num_stages=3,
        phase3_fwd_block_m=64,
        phase3_fwd_block_d=64,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=3,
        phase3_bwd_block_m=64,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=3,
    ),
    (2048, 128, 128): _make_static_cfg(
        chunk_size=64,
        phase1_block_t=16,
        phase1_block_m=64,
        phase1_block_d=128,
        phase2_block_nc=16,
        phase2_block_md=512,
        phase2_num_warps=8,
        phase2_num_stages=2,
        phase3_fwd_block_m=64,
        phase3_fwd_block_d=128,
        phase3_fwd_num_warps=4,
        phase3_fwd_num_stages=2,
        phase3_bwd_block_m=64,
        phase3_bwd_block_d=64,
        phase3_bwd_num_warps=4,
        phase3_bwd_num_stages=3,
        phase1_bwd_num_warps=4,
        phase1_bwd_num_stages=2,
    ),
}

_STATIC_SSD_SHAPE_RANK_CONFIGS: dict[tuple[int, int, int, int], _StaticSsdRank1ShapeConfig] = {
    (N, M, D, rank): cfg
    for (N, M, D), cfg in _STATIC_SSD_RANK1_SHAPE_CONFIGS.items()
    for rank in (1, 2, 3, 4)
}

# Rank-specific phase-3 backward tuning for the common trainer shape.
for _key, _phase3_override in {
    (2048, 64, 64, 1): dict(
        fused_dv_num_warps=2,
        fused_main_num_warps=4,
        fused_off_num_warps=4,
        fused_dv_num_stages=3,
        fused_main_num_stages=3,
        fused_off_num_stages=4,
    ),
    (2048, 64, 64, 2): dict(
        fused_dv_num_warps=2,
        fused_main_num_warps=8,
        fused_off_num_warps=4,
        fused_dv_num_stages=3,
        fused_main_num_stages=4,
        fused_off_num_stages=4,
    ),
    (2048, 64, 64, 3): dict(
        fused_dv_num_warps=2,
        fused_main_num_warps=2,
        fused_off_num_warps=8,
        fused_dv_num_stages=3,
        fused_main_num_stages=3,
        fused_off_num_stages=3,
    ),
}.items():
    if _key in _STATIC_SSD_SHAPE_RANK_CONFIGS:
        _cfg = _STATIC_SSD_SHAPE_RANK_CONFIGS[_key]
        _p3 = replace(
            _cfg.phase3_backward,
            **_phase3_override,
            # Keep legacy alias fields coherent.
            fused_a1a2_num_warps=_phase3_override["fused_main_num_warps"],
            num_stages=_phase3_override["fused_main_num_stages"],
        )
        _STATIC_SSD_SHAPE_RANK_CONFIGS[_key] = replace(_cfg, phase3_backward=_p3)

_ACTIVE_STATIC_SSD_RANK1_KEY: tuple[int, int, int, int] | None = None
_ACTIVE_STATIC_SSD_RANK1_CONFIG: _StaticSsdRank1ShapeConfig | None = None
_STATIC_SSD_RANK1_WORKSPACES: dict[tuple[tuple[int, int, int, int], tuple[str, int], bool], _StaticSsdRank1Workspace] = {}


def _get_static_workspace(
    *,
    device: torch.device,
    cfg_key: tuple[int, int, int, int],
    cfg: _StaticSsdRank1ShapeConfig,
    allocate_phase3_s0: bool = True,
) -> _StaticSsdRank1Workspace:
    BH, N, M, D = cfg_key
    NC = N // cfg.chunk_size
    MD = M * D
    dev_key = (device.type, device.index if device.type == "cuda" else -1)
    key = (cfg_key, dev_key, allocate_phase3_s0)
    ws = _STATIC_SSD_RANK1_WORKSPACES.get(key)
    if ws is None:
        s_local_end_md = torch.empty((BH, NC, M, D), device=device, dtype=torch.float32)
        if allocate_phase3_s0:
            phase3_s0_md_fwd = torch.empty((BH, NC, M, D), device=device, dtype=cfg.input_dtype)
            phase3_s0_md_bwd = torch.empty((BH, NC, M, D), device=device, dtype=cfg.input_dtype)
        else:
            # Rank-2/3 static path passes S0 views directly into phase-3 kernels.
            phase3_s0_md_fwd = torch.empty((1, 1, 1, 1), device=device, dtype=cfg.input_dtype)
            phase3_s0_md_bwd = torch.empty((1, 1, 1, 1), device=device, dtype=cfg.input_dtype)
        ws = _StaticSsdRank1Workspace(
            s_local_end_md=s_local_end_md,
            phase2_init_dummy=torch.empty((1, 1), device=device, dtype=torch.float32),
            phase2_chunk_start=torch.empty((BH, NC, MD), device=device, dtype=torch.float32),
            phase3_s0_md_fwd=phase3_s0_md_fwd,
            phase3_s0_md_bwd=phase3_s0_md_bwd,
            phase3_dlog_fp32=torch.empty((BH, NC, cfg.chunk_size), device=device, dtype=torch.float32),
            # Reuse phase-1 output storage for phase-3 dS0: lifetimes do not overlap.
            phase3_dS0=s_local_end_md,
            phase2_dlog_per_chunk=torch.empty((BH, NC), device=device, dtype=torch.float32),
            phase2_dinit=torch.empty((BH, MD), device=device, dtype=torch.float32),
            phase2_final_replay=torch.empty((BH, MD), device=device, dtype=torch.float32),
            dlog_chunk_accum=torch.empty((BH, NC, cfg.chunk_size), device=device, dtype=cfg.input_dtype),
            phase2_grad_final_zero=torch.empty((BH, MD), device=device, dtype=torch.float32),
        )
        _STATIC_SSD_RANK1_WORKSPACES[key] = ws
    return ws


def _lookup_static_ssd_rank1_shape_config(*, N: int, M: int, D: int) -> _StaticSsdRank1ShapeConfig:
    key = (N, M, D)
    cfg = _STATIC_SSD_RANK1_SHAPE_CONFIGS.get(key)
    if cfg is None:
        supported = sorted(_STATIC_SSD_RANK1_SHAPE_CONFIGS.keys())
        raise NotImplementedError(
            "No static SSD rank1 launch config for shape "
            f"(N={N}, M={M}, D={D}). Supported keys: {supported}."
        )
    return cfg


def _lookup_static_ssd_shape_rank_config(*, N: int, M: int, D: int, RANK: int) -> _StaticSsdRank1ShapeConfig:
    key = (N, M, D, RANK)
    cfg = _STATIC_SSD_SHAPE_RANK_CONFIGS.get(key)
    if cfg is None:
        supported = sorted(_STATIC_SSD_SHAPE_RANK_CONFIGS.keys())
        raise NotImplementedError(
            "No static SSD launch config for shape+rank "
            f"(N={N}, M={M}, D={D}, RANK={RANK}). Supported keys: {supported}."
        )
    return cfg


def _set_active_static_shape_rank(*, N: int, M: int, D: int, RANK: int) -> None:
    global _ACTIVE_STATIC_SSD_RANK1_KEY, _ACTIVE_STATIC_SSD_RANK1_CONFIG
    cfg = _lookup_static_ssd_shape_rank_config(N=N, M=M, D=D, RANK=RANK)
    _ACTIVE_STATIC_SSD_RANK1_KEY = (N, M, D, RANK)
    _ACTIVE_STATIC_SSD_RANK1_CONFIG = cfg


def set_ssd_rank1_static_shape(*, N: int, M: int, D: int) -> None:
    """Bind a single static shape config for all subsequent hot-path calls."""
    _set_active_static_shape_rank(N=N, M=M, D=D, RANK=1)


def clear_ssd_rank1_static_shape() -> None:
    """Clear globally bound static shape config (debug/testing utility)."""
    global _ACTIVE_STATIC_SSD_RANK1_KEY, _ACTIVE_STATIC_SSD_RANK1_CONFIG
    _ACTIVE_STATIC_SSD_RANK1_KEY = None
    _ACTIVE_STATIC_SSD_RANK1_CONFIG = None


def _validate_static_hot_path_contract(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None,
    CHUNK_SIZE: int | None,
    INPUT_PRECISION: str,
    RETURN_FINAL_STATE: bool,
    RANK: int = 1,
) -> _StaticSsdRank1ShapeConfig:
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "Static SSD rank1 path expects C/W=[B,N,H,M], V=[B,N,H,D], log_alpha=[B,N,H]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    B, N, H, M = C.shape
    if W.shape != (B, N, H, M):
        raise ValueError(f"Static SSD rank1 path requires W shape [B,N,H,M]={B, N, H, M}; got {tuple(W.shape)}.")
    if V.shape[:3] != (B, N, H):
        raise ValueError(f"Static SSD rank1 path requires V shape [B,N,H,*]={B, N, H}; got {tuple(V.shape)}.")
    D = V.shape[-1]
    if log_alpha.shape != (B, N, H):
        raise ValueError(f"Static SSD rank1 path requires log_alpha shape [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if C.device != W.device or C.device != V.device or C.device != log_alpha.device:
        raise ValueError("Static SSD rank1 path requires C/W/V/log_alpha on the same device.")
    if not C.is_cuda:
        raise NotImplementedError("Static SSD rank1 path requires CUDA tensors.")
    if not C.is_contiguous() or not W.is_contiguous() or not V.is_contiguous() or not log_alpha.is_contiguous():
        raise ValueError("Static SSD rank1 path requires contiguous C/W/V/log_alpha.")
    global _ACTIVE_STATIC_SSD_RANK1_KEY, _ACTIVE_STATIC_SSD_RANK1_CONFIG
    shape_key = (N, M, D, RANK)
    if _ACTIVE_STATIC_SSD_RANK1_CONFIG is None:
        _set_active_static_shape_rank(N=N, M=M, D=D, RANK=RANK)
    assert _ACTIVE_STATIC_SSD_RANK1_KEY is not None
    if _ACTIVE_STATIC_SSD_RANK1_KEY[:3] != shape_key[:3]:
        raise ValueError(
            "Static SSD rank1 path is bound to one shape per process. "
            f"Active={_ACTIVE_STATIC_SSD_RANK1_KEY}, requested={shape_key}. "
            "Call clear_ssd_rank1_static_shape() then set_ssd_rank1_static_shape(...) to switch."
        )
    if _ACTIVE_STATIC_SSD_RANK1_KEY != shape_key:
        _set_active_static_shape_rank(N=N, M=M, D=D, RANK=RANK)
    assert _ACTIVE_STATIC_SSD_RANK1_CONFIG is not None
    cfg = _ACTIVE_STATIC_SSD_RANK1_CONFIG
    if cfg.has_initial_state:
        raise NotImplementedError("Static SSD rank1 config with initial_state is not implemented in this path.")
    if initial_state is not None:
        raise ValueError("Static SSD rank1 path requires initial_state=None.")
    if C.dtype != cfg.input_dtype or W.dtype != cfg.input_dtype or V.dtype != cfg.input_dtype or log_alpha.dtype != cfg.input_dtype:
        raise ValueError(
            f"Static SSD rank1 path requires dtype={cfg.input_dtype}; "
            f"got C={C.dtype}, W={W.dtype}, V={V.dtype}, log_alpha={log_alpha.dtype}."
        )
    if INPUT_PRECISION != cfg.input_precision:
        raise ValueError(
            f"Static SSD rank1 path requires INPUT_PRECISION='{cfg.input_precision}'; got '{INPUT_PRECISION}'."
        )
    if RETURN_FINAL_STATE != cfg.return_final_state:
        raise ValueError(
            f"Static SSD rank1 path requires RETURN_FINAL_STATE={cfg.return_final_state}; got {RETURN_FINAL_STATE}."
        )
    if CHUNK_SIZE is None:
        raise ValueError(
            "Static SSD rank1 path requires explicit CHUNK_SIZE so launch/shape constants remain fixed. "
            f"Expected CHUNK_SIZE={cfg.chunk_size}."
        )
    if CHUNK_SIZE != cfg.chunk_size:
        raise ValueError(
            f"Static SSD rank1 path requires CHUNK_SIZE={cfg.chunk_size} for this shape; got CHUNK_SIZE={CHUNK_SIZE}."
        )

    _require_supported_md(M, D, where="_validate_static_hot_path_contract")
    _require_supported_forward_contract(B * H, N, CHUNK_SIZE=cfg.chunk_size, where="_validate_static_hot_path_contract")
    _require_nonpositive_log_alpha(log_alpha, where="_validate_static_hot_path_contract")
    if N % cfg.chunk_size != 0:
        raise NotImplementedError(
            f"Static SSD rank1 path requires N divisible by CHUNK_SIZE={cfg.chunk_size}; got N={N}."
        )
    return cfg


def _ensure_triton_allocator() -> None:
    global _EXPERIMENTAL_TRITON_ALLOCATOR_SET
    if _EXPERIMENTAL_TRITON_ALLOCATOR_SET:
        return

    def _alloc(size: int, _align: int, _stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")

    triton.set_allocator(_alloc)
    _EXPERIMENTAL_TRITON_ALLOCATOR_SET = True


def _ssd_rank1_bwd_workspace_tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Get or create a reusable Phase-3 backward workspace tensor."""
    dev_key = (device.type, device.index if device.type == "cuda" else -1)
    key = (name, dev_key, dtype, shape)
    t = _PHASE3_BWD_WORKSPACE.get(key)
    if t is None:
        t = torch.empty(shape, device=device, dtype=dtype)
        _PHASE3_BWD_WORKSPACE[key] = t
    return t


def _require_supported_md(m_size: int, d_size: int, *, where: str) -> None:
    if m_size not in _SUPPORTED_M_VALUES:
        raise NotImplementedError(f"{where} requires M in {sorted(_SUPPORTED_M_VALUES)}; got M={m_size}.")
    if d_size not in _SUPPORTED_D_VALUES:
        raise NotImplementedError(f"{where} requires D in {sorted(_SUPPORTED_D_VALUES)}; got D={d_size}.")


def _require_chunk_size_multiple_of_16(c_size: int, *, where: str) -> None:
    """Enforce the current Triton chunk-size contract."""
    if c_size == 0:
        raise NotImplementedError(f"{where} does not support C==0.")
    if c_size % 16 != 0:
        raise NotImplementedError(f"{where} requires C to be a positive multiple of 16; got C={c_size}.")


def _require_nc_descriptor_width(nc_size: int, *, where: str) -> None:
    if nc_size < 8:
        raise NotImplementedError(f"{where} requires NC >= 8 for descriptor vector width; got NC={nc_size}.")


def _require_nc_multiple_of_16(nc_size: int, *, where: str) -> None:
    if nc_size % 16 != 0:
        raise NotImplementedError(f"{where} requires NC to be a multiple of 16; got NC={nc_size}.")


def _require_nonpositive_log_alpha(log_alpha: torch.Tensor, *, where: str) -> None:
    if torch.any(log_alpha > 0):
        max_val = float(torch.max(log_alpha).item())
        raise ValueError(f"{where} requires log_alpha <= 0 everywhere; observed max(log_alpha)={max_val}.")


def _require_supported_chunk_size(chunk_size: int, *, where: str) -> None:
    if chunk_size not in _SUPPORTED_CHUNK_SIZES:
        raise NotImplementedError(
            f"{where} requires CHUNK_SIZE in {sorted(_SUPPORTED_CHUNK_SIZES)}; got CHUNK_SIZE={chunk_size}."
        )


def _require_supported_forward_contract(BH: int, N: int, *, CHUNK_SIZE: int, where: str) -> None:
    if BH % 8 != 0:
        raise NotImplementedError(f"{where} requires BH to be a multiple of 8; got BH={BH}.")
    if N % 512 != 0:
        raise NotImplementedError(f"{where} requires N to be a multiple of 512; got N={N}.")
    _require_supported_chunk_size(CHUNK_SIZE, where=where)


def _select_largest_block_size(size: int, candidates: tuple[int, ...], *, where: str, label: str) -> int:
    for block in candidates:
        if block <= size and size % block == 0:
            return block
    raise NotImplementedError(
        f"{where} requires {label} to be divisible by one of {list(candidates)}; got {label}={size}."
    )


def _select_chunk_size_heuristic(*, N: int, M: int, D: int, BH: int) -> int:
    """Heuristic CHUNK_SIZE picker for the supported contract."""
    if max(M, D) >= 128:
        return 32
    if max(M, D) <= 32 and N >= 4096 and BH <= 2048:
        return 128
    return 64


def _select_phase2_block_nc(*, NC: int) -> int:
    """Heuristic BLOCK_NC for phase-2 kernels."""
    if NC >= 32 and NC % 32 == 0:
        return 32
    return 16


def _select_phase2_launch_config(*, MD: int, NC: int, where: str) -> _Phase2LaunchConfig:
    if MD % 64 != 0:
        raise NotImplementedError(f"{where} requires MD to be divisible by 64; got MD={MD}.")
    if NC % 16 != 0:
        raise NotImplementedError(f"{where} requires NC to be divisible by 16; got NC={NC}.")
    if MD >= 8192:
        return _Phase2LaunchConfig(block_md=512 if MD % 512 == 0 else 256, num_warps=8, num_stages=3)
    if MD >= 4096:
        return _Phase2LaunchConfig(block_md=256, num_warps=4, num_stages=3)
    if MD >= 2048:
        return _Phase2LaunchConfig(block_md=128, num_warps=4, num_stages=2)
    return _Phase2LaunchConfig(block_md=64, num_warps=2, num_stages=2)


def _select_phase3_forward_launch_config(
    *,
    BH: int,
    NC: int,
    C_CHUNK: int,
    M: int,
    D: int,
    where: str,
) -> _Phase3ForwardLaunchConfig:
    block_m = _select_largest_block_size(M, _SUPPORTED_BLOCK_X_VALUES, where=where, label="M")
    block_d = _select_largest_block_size(D, _SUPPORTED_BLOCK_X_VALUES, where=where, label="D")

    work_items = BH * NC
    if C_CHUNK == 64 and block_m == 64 and block_d == 64 and work_items >= 4096:
        return _Phase3ForwardLaunchConfig(block_m=64, block_d=64, num_warps=4, num_stages=4)
    if block_m >= 64 and block_d >= 64:
        return _Phase3ForwardLaunchConfig(block_m=block_m, block_d=block_d, num_warps=4, num_stages=2)
    return _Phase3ForwardLaunchConfig(block_m=block_m, block_d=block_d, num_warps=2, num_stages=2)


def _select_phase3_backward_launch_config(
    *,
    BH: int,
    NC: int,
    C_CHUNK: int,
    M: int,
    D: int,
) -> _Phase3BackwardLaunchConfig:
    block_m = 64 if M % 64 == 0 else _select_largest_block_size(M, (32, 16), where="phase3_bwd", label="M")
    block_d = 64 if D % 64 == 0 else _select_largest_block_size(D, (32, 16), where="phase3_bwd", label="D")

    # Best-known setting for the common BF16 trainer shape (M=D=C=64).
    if C_CHUNK == 64 and block_m == 64 and block_d == 64 and BH * NC >= 4096:
        return _Phase3BackwardLaunchConfig(
            block_m=64,
            block_d=64,
            fused_a1a2_num_warps=4,
            fused_off_num_warps=4,
            num_stages=3,
            fused_dv_num_warps=4,
            fused_main_num_warps=4,
            fused_dv_num_stages=3,
            fused_main_num_stages=3,
            fused_off_num_stages=3,
        )
    if block_m >= 32 and block_d >= 32:
        return _Phase3BackwardLaunchConfig(
            block_m=block_m,
            block_d=block_d,
            fused_a1a2_num_warps=4,
            fused_off_num_warps=4,
            num_stages=3,
            fused_dv_num_warps=4,
            fused_main_num_warps=4,
            fused_dv_num_stages=3,
            fused_main_num_stages=3,
            fused_off_num_stages=3,
        )
    return _Phase3BackwardLaunchConfig(
        block_m=block_m,
        block_d=block_d,
        fused_a1a2_num_warps=2,
        fused_off_num_warps=2,
        num_stages=3,
        fused_dv_num_warps=2,
        fused_main_num_warps=2,
        fused_dv_num_stages=3,
        fused_main_num_stages=3,
        fused_off_num_stages=3,
    )


def _select_phase1_backward_launch_config(
    *,
    M: int,
    D: int,
) -> _Phase1BackwardLaunchConfig:
    if M == 64 and D == 64:
        return _Phase1BackwardLaunchConfig(num_warps=4, num_stages=3)
    if M >= 128 or D >= 128:
        return _Phase1BackwardLaunchConfig(num_warps=4, num_stages=2)
    return _Phase1BackwardLaunchConfig(num_warps=2, num_stages=2)


def _require_phase1_block_t(block_t: int, c_size: int, *, where: str) -> None:
    if block_t not in _SUPPORTED_PHASE1_BLOCK_T_VALUES:
        raise NotImplementedError(
            f"{where} requires BLOCK_T in {sorted(_SUPPORTED_PHASE1_BLOCK_T_VALUES)}; got BLOCK_T={block_t}."
        )
    if block_t < 16:
        raise NotImplementedError(f"{where} requires BLOCK_T >= 16 for tensor-core Phase-1 path; got BLOCK_T={block_t}.")
    if block_t > c_size:
        raise NotImplementedError(f"{where} requires BLOCK_T <= C; got BLOCK_T={block_t}, C={c_size}.")
    if c_size % block_t != 0:
        raise NotImplementedError(f"{where} requires C divisible by BLOCK_T; got C={c_size}, BLOCK_T={block_t}.")


def _ssd_rank1_prepare_unchunked_inputs(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None,
    *,
    where: str,
    CHUNK_SIZE: int,
    materialize_zero_init: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, int, int, int, int, int, int]:
    """Validate and chunk unchunked mode-0 tensors into `[B, NC, C, H, *]` views."""
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            f"{where} expects C=[B,N,H,M], W=[B,N,H,M], V=[B,N,H,D], log_alpha=[B,N,H]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    B, N, H, M = C.shape
    if W.shape != (B, N, H, M):
        raise ValueError(f"{where}: W must be [B,N,H,M]={B, N, H, M}; got {tuple(W.shape)}.")
    if V.shape[:3] != (B, N, H):
        raise ValueError(f"{where}: V must match [B,N,H,*]={B, N, H}; got {tuple(V.shape)}.")
    D = V.shape[-1]
    if log_alpha.shape != (B, N, H):
        raise ValueError(f"{where}: log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if C.device != W.device or C.device != V.device or C.device != log_alpha.device:
        raise ValueError(f"{where}: C, W, V, log_alpha must be on the same device.")
    if C.dtype != W.dtype or C.dtype != V.dtype or C.dtype != log_alpha.dtype:
        raise ValueError(f"{where}: C, W, V, log_alpha must share dtype.")
    if N == 0:
        raise NotImplementedError(f"{where} does not support N==0.")
    _require_supported_md(M, D, where=where)
    _require_supported_forward_contract(B * H, N, CHUNK_SIZE=CHUNK_SIZE, where=where)
    _require_nonpositive_log_alpha(log_alpha, where=where)

    if N % CHUNK_SIZE != 0:
        raise NotImplementedError(
            f"{where} requires N divisible by CHUNK_SIZE without padding; got N={N}, CHUNK_SIZE={CHUNK_SIZE}."
        )
    NC_data = N // CHUNK_SIZE
    _require_nc_descriptor_width(NC_data, where=where)

    # No-op chunk views over original [B,N,H,*] storage.
    C_chunk = C.view(B, NC_data, CHUNK_SIZE, H, M)
    W_chunk = W.view(B, NC_data, CHUNK_SIZE, H, M)
    V_chunk = V.view(B, NC_data, CHUNK_SIZE, H, D)
    log_alpha_chunk = log_alpha.view(B, NC_data, CHUNK_SIZE, H)

    NC_exec = NC_data

    BH = B * H
    MD = M * D
    if initial_state is None:
        init_flat = torch.zeros((BH, MD), device=C.device, dtype=C.dtype) if materialize_zero_init else None
    elif initial_state.ndim == 2 and initial_state.shape == (BH, MD):
        init_flat = initial_state
    elif initial_state.ndim == 3 and initial_state.shape == (B, H, MD):
        init_flat = initial_state.reshape(BH, MD)
    elif initial_state.ndim == 4 and initial_state.shape == (B, H, M, D):
        init_flat = initial_state.reshape(BH, MD)
    else:
        raise ValueError(
            f"{where}: initial_state must be [BH,MD], [B,H,MD], or [B,H,M,D]. "
            f"Got {tuple(initial_state.shape)} with expected BH={BH}, MD={MD}."
        )
    if init_flat is not None:
        if init_flat.device != C.device:
            raise ValueError(f"{where}: initial_state must be on the same device as C/W/V/log_alpha.")
        if init_flat.dtype != C.dtype:
            raise ValueError(f"{where}: initial_state must share dtype with C/W/V/log_alpha.")

    return C_chunk, W_chunk, V_chunk, log_alpha_chunk, init_flat, B, N, H, M, D, NC_exec


def _ssd_rank1_restore_output_layout(
    y_chunk: torch.Tensor,
    final_state_flat: torch.Tensor | None,
    *,
    B: int,
    N: int,
    H: int,
    C: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Map `[BH, NC, C, D]`/`[BH, MD]` back to `[B,N,H,D]`/`[B,H,MD]`."""
    D = y_chunk.shape[-1]
    y = (
        y_chunk.reshape(B, H, y_chunk.shape[1], C, D)
        .permute(0, 2, 3, 1, 4)
        .reshape(B, y_chunk.shape[1] * C, H, D)
    )[:, :N]
    final_state = None if final_state_flat is None else final_state_flat.reshape(B, H, -1)
    return y, final_state


def _ssd_rank4_collect_rank_terms_unchunked(
    C: torch.Tensor,
    W1: torch.Tensor | None,
    V1: torch.Tensor | None,
    W2: torch.Tensor | None = None,
    V2: torch.Tensor | None = None,
    W3: torch.Tensor | None = None,
    V3: torch.Tensor | None = None,
    W4: torch.Tensor | None = None,
    V4: torch.Tensor | None = None,
    *,
    where: str,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int, int, int, int, int]:
    if W1 is None or V1 is None:
        raise ValueError(f"{where}: W1 and V1 are required.")
    if C.ndim != 4 or W1.ndim != 4 or V1.ndim != 4:
        raise ValueError(
            f"{where} expects C/W1/V1 with rank 4 and shapes [B,N,H,*]. "
            f"Got C={tuple(C.shape)}, W1={tuple(W1.shape)}, V1={tuple(V1.shape)}."
        )
    B, N, H, M = C.shape
    D = V1.shape[-1]
    if W1.shape != (B, N, H, M):
        raise ValueError(f"{where}: W1 must be [B,N,H,M]={B, N, H, M}; got {tuple(W1.shape)}.")
    if V1.shape != (B, N, H, D):
        raise ValueError(f"{where}: V1 must be [B,N,H,D]={B, N, H, D}; got {tuple(V1.shape)}.")

    terms: list[tuple[torch.Tensor, torch.Tensor]] = [(W1, V1)]
    for rank, Wk, Vk in ((2, W2, V2), (3, W3, V3), (4, W4, V4)):
        if Wk is None and Vk is None:
            continue
        if Wk is None or Vk is None:
            raise ValueError(f"{where}: W{rank} and V{rank} must be both None or both tensors.")
        if Wk.ndim != 4 or Vk.ndim != 4:
            raise ValueError(f"{where}: W{rank}/V{rank} must be rank-4 tensors.")
        if Wk.shape != (B, N, H, M):
            raise ValueError(f"{where}: W{rank} must be [B,N,H,M]={B, N, H, M}; got {tuple(Wk.shape)}.")
        if Vk.shape != (B, N, H, D):
            raise ValueError(f"{where}: V{rank} must be [B,N,H,D]={B, N, H, D}; got {tuple(Vk.shape)}.")
        if Wk.device != C.device or Vk.device != C.device:
            raise ValueError(f"{where}: W{rank}/V{rank} must be on the same device as C.")
        if Wk.dtype != C.dtype or Vk.dtype != C.dtype:
            raise ValueError(f"{where}: W{rank}/V{rank} must share dtype with C.")
        terms.append((Wk, Vk))

    for rank, (Wk, Vk) in enumerate(terms, start=1):
        if Wk.device != C.device or Vk.device != C.device:
            raise ValueError(f"{where}: W{rank}/V{rank} must be on the same device as C.")
        if Wk.dtype != C.dtype or Vk.dtype != C.dtype:
            raise ValueError(f"{where}: W{rank}/V{rank} must share dtype with C.")
    return terms, B, N, H, M, D


def _ssd_rank4_collect_rank_terms_chunked(
    W1: torch.Tensor | None,
    V1: torch.Tensor | None,
    W2: torch.Tensor | None = None,
    V2: torch.Tensor | None = None,
    W3: torch.Tensor | None = None,
    V3: torch.Tensor | None = None,
    W4: torch.Tensor | None = None,
    V4: torch.Tensor | None = None,
    *,
    where: str,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int, int, int, int, int]:
    if W1 is None or V1 is None:
        raise ValueError(f"{where}: W1 and V1 are required.")
    if W1.ndim != 4 or V1.ndim != 4:
        raise ValueError(f"{where}: W1/V1 must be rank-4 chunked tensors.")
    BH, NC, C_CHUNK, M = W1.shape
    D = V1.shape[-1]
    if V1.shape != (BH, NC, C_CHUNK, D):
        raise ValueError(f"{where}: V1 must be [BH,NC,C,D]={BH, NC, C_CHUNK, D}; got {tuple(V1.shape)}.")

    terms: list[tuple[torch.Tensor, torch.Tensor]] = [(W1, V1)]
    for rank, Wk, Vk in ((2, W2, V2), (3, W3, V3), (4, W4, V4)):
        if Wk is None and Vk is None:
            continue
        if Wk is None or Vk is None:
            raise ValueError(f"{where}: W{rank} and V{rank} must be both None or both tensors.")
        if Wk.ndim != 4 or Vk.ndim != 4:
            raise ValueError(f"{where}: W{rank}/V{rank} must be rank-4 chunked tensors.")
        if Wk.shape != (BH, NC, C_CHUNK, M):
            raise ValueError(f"{where}: W{rank} must be [BH,NC,C,M]={BH, NC, C_CHUNK, M}; got {tuple(Wk.shape)}.")
        if Vk.shape != (BH, NC, C_CHUNK, D):
            raise ValueError(f"{where}: V{rank} must be [BH,NC,C,D]={BH, NC, C_CHUNK, D}; got {tuple(Vk.shape)}.")
        if Wk.device != W1.device or Vk.device != W1.device:
            raise ValueError(f"{where}: W{rank}/V{rank} must be on the same device as W1/V1.")
        if Wk.dtype != W1.dtype or Vk.dtype != W1.dtype:
            raise ValueError(f"{where}: W{rank}/V{rank} must share dtype with W1/V1.")
        terms.append((Wk, Vk))
    return terms, BH, NC, C_CHUNK, M, D


def ssd_rank4_chunk_end_state_reference(
    W1: torch.Tensor,
    V1: torch.Tensor,
    log_alpha: torch.Tensor,
    W2: torch.Tensor | None = None,
    V2: torch.Tensor | None = None,
    W3: torch.Tensor | None = None,
    V3: torch.Tensor | None = None,
    W4: torch.Tensor | None = None,
    V4: torch.Tensor | None = None,
) -> torch.Tensor:
    """Phase-1 rank-4 reference: local chunk end-state for up to four rank-1 writes."""
    terms, BH, NC, C_CHUNK, M, D = _ssd_rank4_collect_rank_terms_chunked(
        W1, V1, W2, V2, W3, V3, W4, V4, where="ssd_rank4_chunk_end_state_reference"
    )
    if log_alpha.ndim != 3 or log_alpha.shape != (BH, NC, C_CHUNK):
        raise ValueError(
            "ssd_rank4_chunk_end_state_reference expects log_alpha=[BH,NC,C]. "
            f"Got log_alpha={tuple(log_alpha.shape)} with expected {(BH, NC, C_CHUNK)}."
        )
    if log_alpha.device != W1.device:
        raise ValueError("log_alpha must be on the same device as W/V.")
    if log_alpha.dtype != W1.dtype:
        raise ValueError("log_alpha must share dtype with W/V.")
    if C_CHUNK == 0:
        raise NotImplementedError("ssd_rank4_chunk_end_state_reference does not support C==0.")
    _require_supported_md(M, D, where="ssd_rank4_chunk_end_state_reference")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank4_chunk_end_state_reference")

    log_alpha_f = log_alpha.float()
    log_alpha_rev = torch.flip(log_alpha_f, dims=[-1])
    log_suffix_incl_rev = torch.cumsum(log_alpha_rev, dim=-1)
    log_suffix_excl_rev = log_suffix_incl_rev - log_alpha_rev
    r = torch.flip(torch.exp2(log_suffix_excl_rev * _INV_LN2), dims=[-1])

    S_local_end_md = torch.zeros((BH, NC, M, D), device=W1.device, dtype=torch.float32)
    for Wk, Vk in terms:
        weighted_w_t = Wk.float().transpose(-1, -2) * r.unsqueeze(-2)
        S_local_end_md = S_local_end_md + torch.matmul(weighted_w_t, Vk.float())
    return S_local_end_md.reshape(BH, NC, M * D)


def ssd_rank4_dense_output_reference(
    C: torch.Tensor,
    W1: torch.Tensor,
    V1: torch.Tensor,
    log_alpha: torch.Tensor,
    S0: torch.Tensor | None = None,
    W2: torch.Tensor | None = None,
    V2: torch.Tensor | None = None,
    W3: torch.Tensor | None = None,
    V3: torch.Tensor | None = None,
    W4: torch.Tensor | None = None,
    V4: torch.Tensor | None = None,
) -> torch.Tensor:
    """Phase-3 rank-4 dense reference: sums up to four rank-1 write terms."""
    terms, BH, NC, C_CHUNK, M, D = _ssd_rank4_collect_rank_terms_chunked(
        W1, V1, W2, V2, W3, V3, W4, V4, where="ssd_rank4_dense_output_reference"
    )
    if C.ndim != 4 or C.shape != (BH, NC, C_CHUNK, M):
        raise ValueError(f"C must be [BH,NC,C,M]={BH, NC, C_CHUNK, M}; got {tuple(C.shape)}.")
    if log_alpha.ndim != 3 or log_alpha.shape != (BH, NC, C_CHUNK):
        raise ValueError(f"log_alpha must be [BH,NC,C]={BH, NC, C_CHUNK}; got {tuple(log_alpha.shape)}.")
    if C.device != W1.device or log_alpha.device != W1.device:
        raise ValueError("C and log_alpha must be on the same device as W/V.")
    if C.dtype != W1.dtype or log_alpha.dtype != W1.dtype:
        raise ValueError("C and log_alpha must share dtype with W/V.")
    if C_CHUNK == 0:
        raise NotImplementedError("ssd_rank4_dense_output_reference does not support C==0.")
    _require_supported_md(M, D, where="ssd_rank4_dense_output_reference")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank4_dense_output_reference")

    if S0 is None:
        S0_md = torch.zeros((BH, NC, M, D), device=C.device, dtype=torch.float32)
    else:
        if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
            S0_md = S0.reshape(BH, NC, M, D).to(torch.float32)
        elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
            S0_md = S0.to(torch.float32)
        else:
            raise ValueError(
                "S0 must be [BH,NC,MD] or [BH,NC,M,D] matching C/W/V. "
                f"Got S0={tuple(S0.shape)} with expected BH={BH}, NC={NC}, M={M}, D={D}."
            )
        if S0.device != C.device:
            raise ValueError("S0 must be on the same device as C/W/V/log_alpha.")

    C_f = C.float()
    log_alpha_f = log_alpha.float()
    c_idx = torch.arange(C_CHUNK, device=C.device)
    tril = (c_idx.view(C_CHUNK, 1) >= c_idx.view(1, C_CHUNK)).view(1, 1, C_CHUNK, C_CHUNK)

    log_p = torch.cumsum(log_alpha_f, dim=-1)
    log_delta_l2 = (log_p[..., :, None] - log_p[..., None, :]) * _INV_LN2
    log_delta_l2 = torch.where(
        tril,
        log_delta_l2,
        torch.zeros((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32),
    )
    L = torch.where(
        tril,
        torch.exp2(log_delta_l2),
        torch.zeros((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32),
    )

    p = torch.exp2(log_p * _INV_LN2).unsqueeze(-1)
    Y_off = p * torch.matmul(C_f, S0_md)

    Y_diag = torch.zeros((BH, NC, C_CHUNK, D), device=C.device, dtype=torch.float32)
    for Wk, Vk in terms:
        Rk = C_f @ Wk.float().mT
        Kk = L * Rk
        Y_diag = Y_diag + torch.matmul(Kk, Vk.float())
    return (Y_diag + Y_off).to(V1.dtype)


def ssd_rank4_pytorch(
    C: torch.Tensor,
    W1: torch.Tensor,
    V1: torch.Tensor,
    log_alpha: torch.Tensor,
    W2: torch.Tensor | None = None,
    V2: torch.Tensor | None = None,
    W3: torch.Tensor | None = None,
    V3: torch.Tensor | None = None,
    W4: torch.Tensor | None = None,
    V4: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
    RETURN_FINAL_STATE: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """SSD rank-4 PyTorch reference/oracle entrypoint with up to four write terms."""
    terms, B, N, H, M, D = _ssd_rank4_collect_rank_terms_unchunked(
        C, W1, V1, W2, V2, W3, V3, W4, V4, where="ssd_rank4_pytorch"
    )
    if log_alpha.ndim != 3 or log_alpha.shape != (B, N, H):
        raise ValueError(f"ssd_rank4_pytorch: log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if log_alpha.device != C.device or log_alpha.dtype != C.dtype:
        raise ValueError("ssd_rank4_pytorch: log_alpha must be on the same device and dtype as C.")

    if CHUNK_SIZE is None:
        CHUNK_SIZE = _select_chunk_size_heuristic(N=N, M=M, D=D, BH=B * H)

    W1_t, V1_t = terms[0]
    C_chunk, _, _, log_alpha_chunk, init_flat, B, N, H, M, D, _ = _ssd_rank1_prepare_unchunked_inputs(
        C,
        W1_t,
        V1_t,
        log_alpha,
        initial_state,
        where="ssd_rank4_pytorch",
        CHUNK_SIZE=CHUNK_SIZE,
        materialize_zero_init=True,
    )
    NC = C_chunk.shape[1]

    C_chunk_bh = C_chunk.permute(0, 3, 1, 2, 4).reshape(B * H, NC, CHUNK_SIZE, M)
    log_alpha_chunk_bh = log_alpha_chunk.permute(0, 3, 1, 2).reshape(B * H, NC, CHUNK_SIZE)

    w_chunk_bh: list[torch.Tensor] = []
    v_chunk_bh: list[torch.Tensor] = []
    for Wk, Vk in terms:
        Wk_chunk = Wk.view(B, NC, CHUNK_SIZE, H, M)
        Vk_chunk = Vk.view(B, NC, CHUNK_SIZE, H, D)
        w_chunk_bh.append(Wk_chunk.permute(0, 3, 1, 2, 4).reshape(B * H, NC, CHUNK_SIZE, M))
        v_chunk_bh.append(Vk_chunk.permute(0, 3, 1, 2, 4).reshape(B * H, NC, CHUNK_SIZE, D))

    W2_chunk = w_chunk_bh[1] if len(w_chunk_bh) > 1 else None
    V2_chunk = v_chunk_bh[1] if len(v_chunk_bh) > 1 else None
    W3_chunk = w_chunk_bh[2] if len(w_chunk_bh) > 2 else None
    V3_chunk = v_chunk_bh[2] if len(v_chunk_bh) > 2 else None
    W4_chunk = w_chunk_bh[3] if len(w_chunk_bh) > 3 else None
    V4_chunk = v_chunk_bh[3] if len(v_chunk_bh) > 3 else None

    S_local_end = ssd_rank4_chunk_end_state_reference(
        w_chunk_bh[0],
        v_chunk_bh[0],
        log_alpha_chunk_bh,
        W2=W2_chunk,
        V2=V2_chunk,
        W3=W3_chunk,
        V3=V3_chunk,
        W4=W4_chunk,
        V4=V4_chunk,
    )
    alpha_chunk = torch.exp2(torch.sum(log_alpha_chunk_bh, dim=-1) * _INV_LN2)
    S0_chunk, S1_chunk = ssd_rank1_prefix_scan_reference(S_local_end, alpha_chunk, init_flat)

    y_chunk = ssd_rank4_dense_output_reference(
        C_chunk_bh,
        w_chunk_bh[0],
        v_chunk_bh[0],
        log_alpha_chunk_bh,
        S0_chunk,
        W2=W2_chunk,
        V2=V2_chunk,
        W3=W3_chunk,
        V3=V3_chunk,
        W4=W4_chunk,
        V4=V4_chunk,
    )

    y, final_state = _ssd_rank1_restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
    return y, final_state if RETURN_FINAL_STATE else None


def ssd_rank4_decode_pytorch(
    *,
    C: torch.Tensor,
    W1: torch.Tensor,
    V1: torch.Tensor,
    log_alpha: torch.Tensor,
    W2: torch.Tensor | None = None,
    V2: torch.Tensor | None = None,
    W3: torch.Tensor | None = None,
    V3: torch.Tensor | None = None,
    W4: torch.Tensor | None = None,
    V4: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Token-loop decode/prefill reference for rank-4 recurrence with optional rank terms.

    Contract:
      - single decode step only (`N == 1`)
      - token stream (`C/W*/V*`) can be bf16/fp16/fp32
      - recurrent `state` is accumulated/returned in fp32
      - output `y` is returned in `C.dtype`
    """
    terms, B, N, H, M, D = _ssd_rank4_collect_rank_terms_unchunked(
        C, W1, V1, W2, V2, W3, V3, W4, V4, where="ssd_rank4_decode_pytorch"
    )
    if N != 1:
        raise ValueError(f"ssd_rank4_decode_pytorch is single-step only and requires N==1. Got N={N}.")

    if log_alpha.ndim == 4:
        if log_alpha.shape[-1] != 1:
            raise ValueError(
                "ssd_rank4_decode_pytorch expects log_alpha=[B,N,H] or [B,N,H,1]. "
                f"Got {tuple(log_alpha.shape)}."
            )
        log_alpha = log_alpha.squeeze(-1)
    if log_alpha.ndim != 3 or log_alpha.shape != (B, N, H):
        raise ValueError(f"ssd_rank4_decode_pytorch: log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if log_alpha.device != C.device:
        raise ValueError("ssd_rank4_decode_pytorch: log_alpha must be on the same device as C.")
    if not log_alpha.is_floating_point():
        raise ValueError("ssd_rank4_decode_pytorch: log_alpha must be floating point.")

    log_alpha_f = log_alpha.to(torch.float32).clamp_max(0.0)
    _require_nonpositive_log_alpha(log_alpha_f, where="ssd_rank4_decode_pytorch")
    alpha = torch.exp2(log_alpha_f * _INV_LN2)

    BH = B * H
    MD = M * D
    if state is None:
        S = torch.zeros((B, H, M, D), device=C.device, dtype=torch.float32)
    elif state.ndim == 2 and state.shape == (BH, MD):
        S = state.reshape(B, H, M, D).to(torch.float32)
    elif state.ndim == 3 and state.shape == (B, H, MD):
        S = state.reshape(B, H, M, D).to(torch.float32)
    elif state.ndim == 4 and state.shape == (B, H, M, D):
        S = state.to(torch.float32)
    else:
        raise ValueError(
            "ssd_rank4_decode_pytorch: state must be [BH,MD], [B,H,MD], or [B,H,M,D]. "
            f"Got {tuple(state.shape)} with BH={BH}, MD={MD}."
        )
    if S.device != C.device:
        raise ValueError("ssd_rank4_decode_pytorch: state must share device with C/W/V/log_alpha.")
    if not S.is_floating_point():
        raise ValueError("ssd_rank4_decode_pytorch: state must be floating point.")

    write_sum = torch.zeros((B, H, M, D), device=C.device, dtype=torch.float32)
    for Wk, Vk in terms:
        write_sum = write_sum + Wk[:, 0].to(torch.float32).unsqueeze(-1) * Vk[:, 0].to(torch.float32).unsqueeze(-2)
    S = alpha[:, 0].unsqueeze(-1).unsqueeze(-1) * S + write_sum
    y = torch.matmul(C[:, 0].to(torch.float32).unsqueeze(-2), S).squeeze(-2).to(C.dtype).unsqueeze(1)
    return y, S


@triton.jit
def ssd_rank4_decode_step_kernel(
    C_ptr,
    W1_ptr,
    V1_ptr,
    W2_ptr,
    V2_ptr,
    W3_ptr,
    V3_ptr,
    W4_ptr,
    V4_ptr,
    log_alpha_ptr,
    state_ptr,
    Y_ptr,
    bh_size,
    m_size,
    d_size,
    stride_c_bh,
    stride_c_m,
    stride_w1_bh,
    stride_w1_m,
    stride_v1_bh,
    stride_v1_d,
    stride_w2_bh,
    stride_w2_m,
    stride_v2_bh,
    stride_v2_d,
    stride_w3_bh,
    stride_w3_m,
    stride_v3_bh,
    stride_v3_d,
    stride_w4_bh,
    stride_w4_m,
    stride_v4_bh,
    stride_v4_d,
    stride_log_bh,
    stride_state_bh,
    stride_state_m,
    stride_state_d,
    stride_y_bh,
    stride_y_d,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_d = tl.program_id(1)
    if pid_bh >= bh_size:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_m = offs_m < m_size
    mask_d = offs_d < d_size
    mask_md = mask_m[:, None] & mask_d[None, :]

    c = tl.load(C_ptr + pid_bh * stride_c_bh + offs_m * stride_c_m, mask=mask_m, other=0.0).to(tl.float32)
    w1 = tl.load(W1_ptr + pid_bh * stride_w1_bh + offs_m * stride_w1_m, mask=mask_m, other=0.0).to(tl.float32)
    w2 = tl.load(W2_ptr + pid_bh * stride_w2_bh + offs_m * stride_w2_m, mask=mask_m, other=0.0).to(tl.float32)
    w3 = tl.load(W3_ptr + pid_bh * stride_w3_bh + offs_m * stride_w3_m, mask=mask_m, other=0.0).to(tl.float32)
    w4 = tl.load(W4_ptr + pid_bh * stride_w4_bh + offs_m * stride_w4_m, mask=mask_m, other=0.0).to(tl.float32)

    v1 = tl.load(V1_ptr + pid_bh * stride_v1_bh + offs_d * stride_v1_d, mask=mask_d, other=0.0).to(tl.float32)
    v2 = tl.load(V2_ptr + pid_bh * stride_v2_bh + offs_d * stride_v2_d, mask=mask_d, other=0.0).to(tl.float32)
    v3 = tl.load(V3_ptr + pid_bh * stride_v3_bh + offs_d * stride_v3_d, mask=mask_d, other=0.0).to(tl.float32)
    v4 = tl.load(V4_ptr + pid_bh * stride_v4_bh + offs_d * stride_v4_d, mask=mask_d, other=0.0).to(tl.float32)

    s_prev = tl.load(
        state_ptr + pid_bh * stride_state_bh + offs_m[:, None] * stride_state_m + offs_d[None, :] * stride_state_d,
        mask=mask_md,
        other=0.0,
    ).to(tl.float32)

    log_a = tl.load(log_alpha_ptr + pid_bh * stride_log_bh).to(tl.float32)
    INV_LN2 = 1.4426950408889634
    alpha = tl.exp2(log_a * INV_LN2)
    write = (
        w1[:, None] * v1[None, :]
        + w2[:, None] * v2[None, :]
        + w3[:, None] * v3[None, :]
        + w4[:, None] * v4[None, :]
    )
    s_new = alpha * s_prev + write
    tl.store(
        state_ptr + pid_bh * stride_state_bh + offs_m[:, None] * stride_state_m + offs_d[None, :] * stride_state_d,
        s_new,
        mask=mask_md,
    )
    y = tl.sum(c[:, None] * s_new, axis=0)
    tl.store(Y_ptr + pid_bh * stride_y_bh + offs_d * stride_y_d, y, mask=mask_d)


def _select_rank4_decode_launch_config(*, BH: int, M: int, D: int) -> _DecodeLaunchConfig:
    tuned = _SSD_RANK4_DECODE_TUNED.get((BH, M, D))
    if tuned is not None:
        return tuned

    block_m = _select_largest_block_size(M, _SUPPORTED_BLOCK_X_VALUES, where="rank4_decode_step", label="M")
    block_d = _select_largest_block_size(D, _SUPPORTED_BLOCK_X_VALUES, where="rank4_decode_step", label="D")
    if block_m >= 64 and block_d >= 64:
        if BH >= 1024:
            return _DecodeLaunchConfig(block_m=block_m, block_d=block_d, num_warps=2, num_stages=2)
        if BH >= 256:
            return _DecodeLaunchConfig(block_m=block_m, block_d=block_d, num_warps=2, num_stages=3)
        return _DecodeLaunchConfig(block_m=block_m, block_d=block_d, num_warps=4, num_stages=2)
    if block_m >= 32 and block_d >= 32:
        return _DecodeLaunchConfig(block_m=block_m, block_d=block_d, num_warps=2, num_stages=2)
    return _DecodeLaunchConfig(block_m=block_m, block_d=block_d, num_warps=1, num_stages=2)


def _prepare_rank_term_decode_2d(
    term: torch.Tensor | None,
    *,
    B: int,
    N: int,
    H: int,
    K: int,
) -> torch.Tensor | None:
    if term is None:
        return None
    return term.view(B, N, H, K).permute(0, 2, 1, 3).reshape(B * H, N, K).contiguous()


def ssd_rank4_decode_triton(
    *,
    C: torch.Tensor,
    W1: torch.Tensor,
    V1: torch.Tensor,
    log_alpha: torch.Tensor,
    W2: torch.Tensor | None = None,
    V2: torch.Tensor | None = None,
    W3: torch.Tensor | None = None,
    V3: torch.Tensor | None = None,
    W4: torch.Tensor | None = None,
    V4: torch.Tensor | None = None,
    state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton token-loop decode/prefill for rank-4 recurrence with optional rank terms.

    Contract:
      - single decode step only (`N == 1`)
      - token stream (`C/W*/V*`) can be bf16/fp16/fp32
      - recurrent `state` is accumulated/returned in fp32
      - output `y` is returned in `C.dtype`
    """
    if not C.is_cuda:
        raise NotImplementedError("ssd_rank4_decode_triton requires CUDA tensors.")

    terms, B, N, H, M, D = _ssd_rank4_collect_rank_terms_unchunked(
        C, W1, V1, W2, V2, W3, V3, W4, V4, where="ssd_rank4_decode_triton"
    )
    if N != 1:
        raise ValueError(f"ssd_rank4_decode_triton is single-step only and requires N==1. Got N={N}.")

    if log_alpha.ndim == 4:
        if log_alpha.shape[-1] != 1:
            raise ValueError(
                "ssd_rank4_decode_triton expects log_alpha=[B,N,H] or [B,N,H,1]. "
                f"Got {tuple(log_alpha.shape)}."
            )
        log_alpha = log_alpha.squeeze(-1)
    if log_alpha.ndim != 3 or log_alpha.shape != (B, N, H):
        raise ValueError(f"ssd_rank4_decode_triton: log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if log_alpha.device != C.device:
        raise ValueError("ssd_rank4_decode_triton: log_alpha must be on the same device as C.")
    if not log_alpha.is_floating_point():
        raise ValueError("ssd_rank4_decode_triton: log_alpha must be floating point.")
    log_alpha_f = log_alpha.to(torch.float32).clamp_max(0.0)
    _require_nonpositive_log_alpha(log_alpha_f, where="ssd_rank4_decode_triton")

    BH = B * H
    MD = M * D
    if state is None:
        state_bhmd = torch.zeros((BH, M, D), device=C.device, dtype=torch.float32)
    elif state.ndim == 2 and state.shape == (BH, MD):
        state_bhmd = state.to(device=C.device, dtype=torch.float32).reshape(BH, M, D)
        if not state_bhmd.is_contiguous():
            state_bhmd = state_bhmd.contiguous()
    elif state.ndim == 3 and state.shape == (B, H, MD):
        state_bhmd = state.to(device=C.device, dtype=torch.float32).reshape(BH, M, D)
        if not state_bhmd.is_contiguous():
            state_bhmd = state_bhmd.contiguous()
    elif state.ndim == 4 and state.shape == (B, H, M, D):
        state_bhmd = state.to(device=C.device, dtype=torch.float32).reshape(BH, M, D)
        if not state_bhmd.is_contiguous():
            state_bhmd = state_bhmd.contiguous()
    else:
        raise ValueError(
            "ssd_rank4_decode_triton: state must be [BH,MD], [B,H,MD], or [B,H,M,D]. "
            f"Got {tuple(state.shape)} with BH={BH}, MD={MD}."
        )
    if not state_bhmd.is_floating_point():
        raise ValueError("ssd_rank4_decode_triton: state must be floating point.")

    c_t = C[:, 0].reshape(BH, M)
    if not c_t.is_contiguous():
        c_t = c_t.contiguous()
    log_t = log_alpha_f[:, 0].reshape(BH)
    if not log_t.is_contiguous():
        log_t = log_t.contiguous()
    y_t = torch.empty((BH, D), device=C.device, dtype=C.dtype)

    w_terms = [t[0][:, 0].reshape(BH, M) for t in terms]
    v_terms = [t[1][:, 0].reshape(BH, D) for t in terms]
    while len(w_terms) < 4:
        w_terms.append(None)
        v_terms.append(None)

    zero_w = torch.zeros((BH, M), device=C.device, dtype=C.dtype)
    zero_v = torch.zeros((BH, D), device=C.device, dtype=C.dtype)

    _ensure_triton_allocator()
    decode_cfg = _select_rank4_decode_launch_config(BH=BH, M=M, D=D)
    grid = (BH, triton.cdiv(D, decode_cfg.block_d))
    w1_t = w_terms[0] if w_terms[0] is not None else zero_w
    v1_t = v_terms[0] if v_terms[0] is not None else zero_v
    w2_t = w_terms[1] if w_terms[1] is not None else zero_w
    v2_t = v_terms[1] if v_terms[1] is not None else zero_v
    w3_t = w_terms[2] if w_terms[2] is not None else zero_w
    v3_t = v_terms[2] if v_terms[2] is not None else zero_v
    w4_t = w_terms[3] if w_terms[3] is not None else zero_w
    v4_t = v_terms[3] if v_terms[3] is not None else zero_v
    if not w1_t.is_contiguous():
        w1_t = w1_t.contiguous()
    if not v1_t.is_contiguous():
        v1_t = v1_t.contiguous()
    if not w2_t.is_contiguous():
        w2_t = w2_t.contiguous()
    if not v2_t.is_contiguous():
        v2_t = v2_t.contiguous()
    if not w3_t.is_contiguous():
        w3_t = w3_t.contiguous()
    if not v3_t.is_contiguous():
        v3_t = v3_t.contiguous()
    if not w4_t.is_contiguous():
        w4_t = w4_t.contiguous()
    if not v4_t.is_contiguous():
        v4_t = v4_t.contiguous()

    ssd_rank4_decode_step_kernel[grid](
        c_t,
        w1_t,
        v1_t,
        w2_t,
        v2_t,
        w3_t,
        v3_t,
        w4_t,
        v4_t,
        log_t,
        state_bhmd,
        y_t,
        BH,
        M,
        D,
        c_t.stride(0),
        c_t.stride(1),
        w1_t.stride(0),
        w1_t.stride(1),
        v1_t.stride(0),
        v1_t.stride(1),
        w2_t.stride(0),
        w2_t.stride(1),
        v2_t.stride(0),
        v2_t.stride(1),
        w3_t.stride(0),
        w3_t.stride(1),
        v3_t.stride(0),
        v3_t.stride(1),
        w4_t.stride(0),
        w4_t.stride(1),
        v4_t.stride(0),
        v4_t.stride(1),
        log_t.stride(0),
        state_bhmd.stride(0),
        state_bhmd.stride(1),
        state_bhmd.stride(2),
        y_t.stride(0),
        y_t.stride(1),
        BLOCK_M=decode_cfg.block_m,
        BLOCK_D=decode_cfg.block_d,
        num_warps=decode_cfg.num_warps,
        num_stages=decode_cfg.num_stages,
    )

    y = y_t.view(B, H, D).unsqueeze(1).contiguous()
    final_state = state_bhmd.view(B, H, M, D)
    return y, final_state


def ssd_rank4_token_loop_oracle(
    C: torch.Tensor,
    W1: torch.Tensor,
    V1: torch.Tensor,
    log_alpha: torch.Tensor,
    W2: torch.Tensor | None = None,
    V2: torch.Tensor | None = None,
    W3: torch.Tensor | None = None,
    V3: torch.Tensor | None = None,
    W4: torch.Tensor | None = None,
    V4: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unchunked token-loop oracle for rank-4 recurrence with optional extra write terms."""
    terms, B, N, H, M, D = _ssd_rank4_collect_rank_terms_unchunked(
        C, W1, V1, W2, V2, W3, V3, W4, V4, where="ssd_rank4_token_loop_oracle"
    )
    if log_alpha.ndim != 3 or log_alpha.shape != (B, N, H):
        raise ValueError(
            "ssd_rank4_token_loop_oracle expects log_alpha=[B,N,H]. "
            f"Got log_alpha={tuple(log_alpha.shape)} with expected {(B, N, H)}."
        )
    if log_alpha.device != C.device or log_alpha.dtype != C.dtype:
        raise ValueError("ssd_rank4_token_loop_oracle: log_alpha must share device/dtype with C.")
    if N == 0:
        raise NotImplementedError("ssd_rank4_token_loop_oracle does not support N==0.")
    _require_supported_md(M, D, where="ssd_rank4_token_loop_oracle")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank4_token_loop_oracle")

    BH = B * H
    MD = M * D
    if initial_state is None:
        S = torch.zeros((B, H, M, D), device=C.device, dtype=C.dtype)
    elif initial_state.ndim == 2 and initial_state.shape == (BH, MD):
        S = initial_state.reshape(B, H, M, D)
    elif initial_state.ndim == 3 and initial_state.shape == (B, H, MD):
        S = initial_state.reshape(B, H, M, D)
    elif initial_state.ndim == 4 and initial_state.shape == (B, H, M, D):
        S = initial_state
    else:
        raise ValueError(
            f"initial_state must be [BH,MD], [B,H,MD], or [B,H,M,D]. "
            f"Got {tuple(initial_state.shape)} with BH={BH}, MD={MD}."
        )
    if S.device != C.device:
        raise ValueError("initial_state must be on the same device as C/W/V/log_alpha.")
    if S.dtype != C.dtype:
        raise ValueError("initial_state must share dtype with C/W/V/log_alpha.")

    alpha = torch.exp2(log_alpha * _INV_LN2)
    ys = []
    for t in range(N):
        write_sum = torch.zeros((B, H, M, D), device=C.device, dtype=C.dtype)
        for Wk, Vk in terms:
            write_sum = write_sum + Wk[:, t].unsqueeze(-1) * Vk[:, t].unsqueeze(-2)
        S = alpha[:, t].unsqueeze(-1).unsqueeze(-1) * S + write_sum
        ys.append(torch.matmul(C[:, t].unsqueeze(-2), S).squeeze(-2))
    y = torch.stack(ys, dim=1)
    final_state = S.reshape(B, H, MD)
    return y, final_state


def ssd_rank4_triton(
    C: torch.Tensor,
    W1: torch.Tensor,
    V1: torch.Tensor,
    log_alpha: torch.Tensor,
    W2: torch.Tensor | None = None,
    V2: torch.Tensor | None = None,
    W3: torch.Tensor | None = None,
    V3: torch.Tensor | None = None,
    W4: torch.Tensor | None = None,
    V4: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
    INPUT_PRECISION: str = "tf32",
    RETURN_FINAL_STATE: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """SSD rank-4 Triton entrypoint."""
    terms, B, N, H, M, D = _ssd_rank4_collect_rank_terms_unchunked(
        C,
        W1,
        V1,
        W2,
        V2,
        W3,
        V3,
        W4,
        V4,
        where="ssd_rank4_triton",
    )
    if log_alpha.ndim != 3 or log_alpha.shape != (B, N, H):
        raise ValueError(
            "ssd_rank4_triton expects log_alpha=[B,N,H]. "
            f"Got log_alpha={tuple(log_alpha.shape)} with expected {(B, N, H)}."
        )
    if log_alpha.device != C.device:
        raise ValueError("ssd_rank4_triton: log_alpha must be on the same device as C/W/V.")
    if log_alpha.dtype != C.dtype:
        raise ValueError("ssd_rank4_triton: log_alpha must share dtype with C/W/V.")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank4_triton")
    _require_supported_md(M, D, where="ssd_rank4_triton")

    rank = 1
    if W2 is not None and V2 is not None:
        rank = 2
        if not W2.is_contiguous() or not V2.is_contiguous():
            raise ValueError("ssd_rank4_triton requires contiguous W2/V2 on the static path.")
    if W3 is not None and V3 is not None:
        rank = 3
        if not W3.is_contiguous() or not V3.is_contiguous():
            raise ValueError("ssd_rank4_triton requires contiguous W3/V3 on the static path.")
    if W4 is not None and V4 is not None:
        rank = 4
        if not W4.is_contiguous() or not V4.is_contiguous():
            raise ValueError("ssd_rank4_triton requires contiguous W4/V4 on the static path.")
    cfg = _validate_static_hot_path_contract(
        C,
        W1,
        V1,
        log_alpha,
        initial_state,
        CHUNK_SIZE,
        INPUT_PRECISION,
        RETURN_FINAL_STATE,
        RANK=rank,
    )

    W2_eff = W2 if rank >= 2 else W1.detach()
    V2_eff = V2 if rank >= 2 else V1.detach()
    W3_eff = W3 if rank >= 3 else W1.detach()
    V3_eff = V3 if rank >= 3 else V1.detach()
    W4_eff = W4 if rank >= 4 else W1.detach()
    V4_eff = V4 if rank >= 4 else V1.detach()

    y_chunk, S1_chunk = SsdRank4TritonStatic.apply(
        C,
        W1,
        V1,
        W2_eff,
        V2_eff,
        W3_eff,
        V3_eff,
        W4_eff,
        V4_eff,
        log_alpha,
        initial_state,
        cfg.chunk_size,
        cfg.input_precision,
        cfg.return_final_state,
        rank,
    )
    return _ssd_rank1_restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=cfg.chunk_size)


# ==================================================================================================
# ORACLE / TORCH ENTRYPOINT
# ==================================================================================================
def ssd_rank1_pytorch(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
    RETURN_FINAL_STATE: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """SSD rank-1 PyTorch oracle entrypoint."""
    if CHUNK_SIZE is None:
        B, N, H, M = C.shape
        D = V.shape[-1]
        CHUNK_SIZE = _select_chunk_size_heuristic(N=N, M=M, D=D, BH=B * H)

    C_chunk, W_chunk, V_chunk, log_alpha_chunk, init_flat, B, N, H, M, D, _ = _ssd_rank1_prepare_unchunked_inputs(
        C,
        W,
        V,
        log_alpha,
        initial_state,
        where="ssd_rank1_pytorch",
        CHUNK_SIZE=CHUNK_SIZE,
        materialize_zero_init=True,
    )

    # Reference helpers still consume [BH,NC,C,*]; convert from [B,NC,C,H,*].
    C_chunk_bh = C_chunk.permute(0, 3, 1, 2, 4).reshape(B * H, C_chunk.shape[1], C_chunk.shape[2], M)
    W_chunk_bh = W_chunk.permute(0, 3, 1, 2, 4).reshape(B * H, W_chunk.shape[1], W_chunk.shape[2], M)
    V_chunk_bh = V_chunk.permute(0, 3, 1, 2, 4).reshape(B * H, V_chunk.shape[1], V_chunk.shape[2], D)
    log_alpha_chunk_bh = log_alpha_chunk.permute(0, 3, 1, 2).reshape(B * H, log_alpha_chunk.shape[1], log_alpha_chunk.shape[2])

    # =========================================
    # PHASE 1 local chunk end-state
    # =========================================
    S_local_end = ssd_rank1_chunk_end_state_reference(W_chunk_bh, V_chunk_bh, log_alpha_chunk_bh)
    alpha_chunk = torch.exp2(torch.sum(log_alpha_chunk_bh, dim=-1) * _INV_LN2)

    # =========================================
    # PHASE 2 prefix scan over chunks
    # =========================================
    S0_chunk, S1_chunk = ssd_rank1_prefix_scan_reference(S_local_end, alpha_chunk, init_flat)

    # =========================================
    # PHASE 3 dense chunk-local output
    # =========================================
    y_chunk = ssd_rank1_dense_output_reference(C_chunk_bh, W_chunk_bh, V_chunk_bh, log_alpha_chunk_bh, S0_chunk)

    y, final_state = _ssd_rank1_restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
    return y, final_state if RETURN_FINAL_STATE else None


# --------------------------------------------------------------------------------------------------
# ORACLE: TOKEN-LOOP REFERENCE
# --------------------------------------------------------------------------------------------------
def ssd_rank1_token_loop_oracle(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unchunked token-loop oracle for mode-0 recurrence.

    Recurrence:
      alpha_t = exp(log_alpha_t),  with log_alpha_t <= 0
      S_t = alpha_t * S_{t-1} + W_t ⊗ V_t
      y_t = C_t @ S_t

    Inputs:
    - C: `[B, N, H, M]`
    - W: `[B, N, H, M]`
    - V: `[B, N, H, D]`
    - log_alpha: `[B, N, H]`
    - initial_state: optional `[BH,MD]`, `[B,H,MD]`, or `[B,H,M,D]`

    Returns:
    - y: `[B, N, H, D]`
    - final_state: `[B, H, MD]`
    """
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "ssd_rank1_token_loop_oracle expects C=[B,N,H,M], W=[B,N,H,M], V=[B,N,H,D], log_alpha=[B,N,H]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    B, N, H, M = C.shape
    if W.shape != (B, N, H, M):
        raise ValueError(f"W must be [B,N,H,M]={B, N, H, M}; got {tuple(W.shape)}.")
    if V.shape[:3] != (B, N, H):
        raise ValueError(f"V must match [B,N,H,*]={B, N, H}; got {tuple(V.shape)}.")
    D = V.shape[-1]
    if log_alpha.shape != (B, N, H):
        raise ValueError(f"log_alpha must be [B,N,H]={B, N, H}; got {tuple(log_alpha.shape)}.")
    if C.device != W.device or C.device != V.device or C.device != log_alpha.device:
        raise ValueError("C, W, V, log_alpha must be on the same device.")
    if C.dtype != W.dtype or C.dtype != V.dtype or C.dtype != log_alpha.dtype:
        raise ValueError("C, W, V, log_alpha must share dtype.")
    if N == 0:
        raise NotImplementedError("ssd_rank1_token_loop_oracle does not support N==0.")
    _require_supported_md(M, D, where="ssd_rank1_token_loop_oracle")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank1_token_loop_oracle")

    BH = B * H
    MD = M * D
    if initial_state is None:
        S = torch.zeros((B, H, M, D), device=C.device, dtype=C.dtype)
    elif initial_state.ndim == 2 and initial_state.shape == (BH, MD):
        S = initial_state.reshape(B, H, M, D)
    elif initial_state.ndim == 3 and initial_state.shape == (B, H, MD):
        S = initial_state.reshape(B, H, M, D)
    elif initial_state.ndim == 4 and initial_state.shape == (B, H, M, D):
        S = initial_state
    else:
        raise ValueError(
            f"initial_state must be [BH,MD], [B,H,MD], or [B,H,M,D]. "
            f"Got {tuple(initial_state.shape)} with BH={BH}, MD={MD}."
        )
    if S.device != C.device:
        raise ValueError("initial_state must be on the same device as C/W/V/log_alpha.")
    if S.dtype != C.dtype:
        raise ValueError("initial_state must share dtype with C/W/V/log_alpha.")

    alpha = torch.exp2(log_alpha * _INV_LN2)
    ys = []
    for t in range(N):
        S = (
            alpha[:, t].unsqueeze(-1).unsqueeze(-1) * S
            + W[:, t].unsqueeze(-1) * V[:, t].unsqueeze(-2)
        )
        ys.append(torch.matmul(C[:, t].unsqueeze(-2), S).squeeze(-2))

    y = torch.stack(ys, dim=1)
    final_state = S.reshape(B, H, MD)
    return y, final_state


# --------------------------------------------------------------------------------------------------
# END ORACLE: TOKEN-LOOP REFERENCE
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# END ORACLE / TORCH ENTRYPOINT
# --------------------------------------------------------------------------------------------------

# ==================================================================================================
# REFERENCE FUNCTIONS
# ==================================================================================================

# --------------------------------------------------------------------------------------------------
# PHASE 1: LOCAL CHUNK END-STATE COMPUTATION
# --------------------------------------------------------------------------------------------------
def ssd_rank1_chunk_end_state_reference(
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
) -> torch.Tensor:
    """Phase-1 reference: per-chunk local end-state from factorized writes.

    Inputs:
    - W: `[BH, NC, C, M]`
    - V: `[BH, NC, C, D]`
    - log_alpha: `[BH, NC, C]`
      Scalar log-retain factor per token, constrained to `<= 0`.

    Output:
    - S_local_end: `[BH, NC, MD]`
      End S for each chunk assuming zero chunk start:
        S_{t+1} = alpha_t * S_t + W_t ⊗ V_t, with S_0 = 0
        where `alpha_t = exp(log_alpha_t)`.
      and `S_local_end = S_C`.

    Closed form:
      S_local_end = sum_{j=0..C-1} (prod_{u=j+1..C-1} alpha_u) * (W_j ⊗ V_j)

    Vectorized implementation:
      Let
        r_j = prod_{u=j+1..C-1} alpha_u
      then for each `(BH, NC)` chunk lane:
        S_local_end[M,D] = ( (r[T] * W^T[M,T]) @ V[T,D] )
    """
    if W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "ssd_rank1_chunk_end_state_reference expects W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
            f"Got W={tuple(W.shape)}, V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    BH, NC, C, M = W.shape
    if V.shape[:3] != (BH, NC, C):
        raise ValueError(
            "V must match [BH,NC,C,*] from W. "
            f"Got V={tuple(V.shape)}, W={tuple(W.shape)}."
        )
    D = V.shape[-1]
    if log_alpha.shape != (BH, NC, C):
        raise ValueError(
            "log_alpha must match [BH,NC,C] from W/V. "
            f"Got log_alpha={tuple(log_alpha.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}."
        )
    if W.device != V.device or W.device != log_alpha.device:
        raise ValueError("W, V, log_alpha must be on the same device.")
    if W.dtype != V.dtype or W.dtype != log_alpha.dtype:
        raise ValueError("W, V, log_alpha must share dtype.")
    if C == 0:
        raise NotImplementedError("ssd_rank1_chunk_end_state_reference does not support C==0.")
    _require_supported_md(M, D, where="ssd_rank1_chunk_end_state_reference")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank1_chunk_end_state_reference")
    log_alpha_f = log_alpha.float()
    W_f = W.float()
    V_f = V.float()
    # Build exclusive suffix retain factors:
    #   r[j] = prod_{u=j+1..C-1} alpha[u] = exp(sum_{u=j+1..C-1} log_alpha[u]).
    # We compute this via reverse cumsum in log-space for better numerical behavior.
    log_alpha_rev = torch.flip(log_alpha_f, dims=[-1])
    log_suffix_incl_rev = torch.cumsum(log_alpha_rev, dim=-1)
    log_suffix_excl_rev = log_suffix_incl_rev - log_alpha_rev
    r = torch.flip(torch.exp2(log_suffix_excl_rev * _INV_LN2), dims=[-1])

    # S[M,D] = (r[T] * W^T[M,T]) @ V[T,D], batched across [BH,NC].
    weighted_w_t = W_f.transpose(-1, -2) * r.unsqueeze(-2)
    S_local_end = torch.matmul(weighted_w_t, V_f).reshape(BH, NC, M * D)
    return S_local_end


# --------------------------------------------------------------------------------------------------
# END PHASE 1: LOCAL CHUNK END-STATE COMPUTATION
# --------------------------------------------------------------------------------------------------


def ssd_rank1_prefix_scan_reference(
    S_local_end: torch.Tensor,
    alpha_chunk: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference math for Phase-2 chunk-prefix state propagation.

    Inputs:
    - S_local_end: `[BH, NC, MD]` (or `[BHG, NC, MD]` if grouping is folded outside)
    - alpha_chunk: `[BH, NC]` (or `[BHG, NC]`)
    - initial_state: `[BH, MD]` (or `[BHG, MD]`)

    Chunk-local recurrence:

      S_end[c] = alpha_chunk[c] * S_start[c] + S_local_end[c]
      S_start[c+1] = S_end[c]
      S_start[0] = initial_state

    Closed-form inclusive prefix coefficients:

      inc_r[c] = prod_{k=0..c} alpha_chunk[k]
      inc_s[c] = sum_{j=0..c} ( prod_{k=j+1..c} alpha_chunk[k] ) * S_local_end[j]

    so:

      S_end[c] = inc_r[c] * initial_state + inc_s[c]

    Exclusive chunk starts:

      excl_r[0] = 1, excl_s[0] = 0
      excl_r[c] = inc_r[c-1], excl_s[c] = inc_s[c-1] for c > 0

      chunk_start[c] = excl_r[c] * initial_state + excl_s[c]
      final_state    = inc_r[NC-1] * initial_state + inc_s[NC-1]
    """
    if S_local_end.ndim != 3 or alpha_chunk.ndim != 2 or initial_state.ndim != 2:
        raise ValueError(
            "ssd_rank1_prefix_scan_reference expects "
            "S_local_end=[BH,NC,MD], alpha_chunk=[BH,NC], initial_state=[BH,MD]. "
            f"Got S_local_end={tuple(S_local_end.shape)}, alpha_chunk={tuple(alpha_chunk.shape)}, "
            f"initial_state={tuple(initial_state.shape)}."
        )
    BH, NC, MD = S_local_end.shape
    if alpha_chunk.shape != (BH, NC):
        raise ValueError(
            "alpha_chunk must match [BH,NC] from S_local_end. "
            f"Got alpha_chunk={tuple(alpha_chunk.shape)}, S_local_end={tuple(S_local_end.shape)}."
        )
    if initial_state.shape != (BH, MD):
        raise ValueError(
            "initial_state must match [BH,MD] from S_local_end. "
            f"Got initial_state={tuple(initial_state.shape)}, S_local_end={tuple(S_local_end.shape)}."
        )
    if NC == 0:
        raise NotImplementedError("ssd_rank1_prefix_scan_reference does not support NC==0.")

    scan_r = alpha_chunk.float()
    scan_s = S_local_end.float()
    initial_state_f = initial_state.float()
    step = 1
    while step < NC:
        shifted_r = torch.cat([torch.ones_like(scan_r[:, :step]), scan_r[:, :-step]], dim=1)
        shifted_s = torch.cat([torch.zeros_like(scan_s[:, :step]), scan_s[:, :-step]], dim=1)
        scan_s = scan_s + scan_r.unsqueeze(-1) * shifted_s
        scan_r = scan_r * shifted_r
        step <<= 1

    excl_r = torch.cat([torch.ones_like(scan_r[:, :1]), scan_r[:, :-1]], dim=1)
    excl_s = torch.cat([torch.zeros_like(scan_s[:, :1]), scan_s[:, :-1]], dim=1)
    chunk_start = excl_r.unsqueeze(-1) * initial_state_f.unsqueeze(1) + excl_s
    final_state = scan_r[:, -1].unsqueeze(-1) * initial_state_f + scan_s[:, -1]
    return chunk_start, final_state


@triton.jit
def ssd_rank1_chunk_end_state_fwd_kernel(
    W_ptr,
    V_ptr,
    log_alpha_ptr,
    out_s_local_end_ptr,
    b_size: tl.constexpr,
    h_size: tl.constexpr,
    bh_size: tl.constexpr,
    nc_size: tl.constexpr,
    c_size: tl.constexpr,
    m_size: tl.constexpr,
    d_size: tl.constexpr,
    stride_w_b: tl.constexpr,
    stride_w_nc: tl.constexpr,
    stride_w_c: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_nc: tl.constexpr,
    stride_v_c: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_r_b: tl.constexpr,
    stride_r_nc: tl.constexpr,
    stride_r_c: tl.constexpr,
    stride_r_h: tl.constexpr,
    stride_out_s_local_end_bh: tl.constexpr,
    stride_out_s_local_end_nc: tl.constexpr,
    stride_out_s_local_end_m: tl.constexpr,
    stride_out_s_local_end_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Phase-1 Triton forward for factorized writes.

    State definition for one `(BH, NC)`:

      S_{t+1} = alpha_t * S_t + W_t \otimes V_t,    S_0 = 0
      alpha_t = exp(log_alpha_t), with log_alpha_t <= 0.

    where `W_t` is length-`M`, `V_t` is length-`D`, and `S_t` is `[M, D]`.
    The output we need is:

      S_local_end = S_C

    Closed-form weights for token contributions:

      factor[t] = prod_{u=t+1..C-1} alpha_u
      S_C       = sum_{t=0..C-1} factor[t] * (W_t \otimes V_t)

    Kernel strategy:
    1. Load all `log_alpha_t` for this `(BH, NC)`, form `alpha_t = exp(log_alpha_t)`,
       then compute suffix products for `factor[t]`.
    2. For each token block of size `BLOCK_T`, accumulate:
         S += sum_t (factor_t * W_t[:, None] * V_t[None, :])
       using a GEMM-like reduction to hit tensor cores:
         A = sqrt(factor)[:, None] * W_block    # [BLOCK_T, BLOCK_M]
         B = sqrt(factor)[:, None] * V_block    # [BLOCK_T, BLOCK_D]
         S += A^T @ B                           # [BLOCK_M, BLOCK_D]
    3. Store `state_tile` into `S_local_end[BH, NC, M, D]`.

    Notes:
    - `BLOCK_C` is padded to next power-of-two so token-lane vector ops are legal.
    - Masked token loads use `other=1.0` so padded lanes are multiplicative identity.
    - Internal accumulation is FP32 for stability.
    """
    pid_bhnc = tl.program_id(0)
    pid_m_tile = tl.program_id(1)
    pid_d_tile = tl.program_id(2)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    m_start = pid_m_tile * BLOCK_M
    d_start = pid_d_tile * BLOCK_D
    w_desc = tl.make_tensor_descriptor(
        W_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_T, 1, BLOCK_M],
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_T, 1, BLOCK_D],
    )
    out_s_desc = tl.make_tensor_descriptor(
        out_s_local_end_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_out_s_local_end_bh, stride_out_s_local_end_nc, stride_out_s_local_end_m, stride_out_s_local_end_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    offs_c = tl.arange(0, BLOCK_C)
    r_base = B_IDX * stride_r_b + NC_IDX * stride_r_nc + H_IDX * stride_r_h
    log_alpha_in = tl.load(log_alpha_ptr + r_base + offs_c * stride_r_c)
    log_alpha_vals = log_alpha_in.to(tl.float32)
    INV_LN2 = 1.4426950408889634
    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    r_factors = tl.exp2((log_suffix_incl - log_alpha_vals) * INV_LN2)

    S = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    offs_t_full = tl.arange(0, BLOCK_C)
    for t_start in tl.static_range(0, C_STATIC, BLOCK_T):
        offs_tb = t_start + tl.arange(0, BLOCK_T)
        factors = tl.sum(
            tl.where(offs_t_full[None, :] == offs_tb[:, None], r_factors[None, :], 0.0),
            axis=1,
        )
        W_blk = tl.reshape(w_desc.load([B_IDX, NC_IDX, t_start, H_IDX, m_start]), (BLOCK_T, BLOCK_M))
        V_blk = tl.reshape(v_desc.load([B_IDX, NC_IDX, t_start, H_IDX, d_start]), (BLOCK_T, BLOCK_D))
        sqrt_factors = tl.sqrt(factors)[:, None]
        A_f32 = sqrt_factors * W_blk.to(tl.float32)
        B_f32 = sqrt_factors * V_blk.to(tl.float32)
        # A^T @ B reproduces sum_t factor[t] * outer(W_t, V_t).
        S = S + tl.dot(
            tl.trans(A_f32.to(tl.bfloat16)),
            B_f32.to(tl.bfloat16),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
    out_s_desc.store([BH_IDX, NC_IDX, m_start, d_start], tl.reshape(S, (1, 1, BLOCK_M, BLOCK_D)))

    # alpha_chunk belongs to phase-1 scope but is computed outside this kernel.


@triton.jit
def ssd_rank1_chunk_end_state_bwd_fused_kernel(
    grad_s_local_end_ptr,
    W_ptr,
    V_ptr,
    log_alpha_ptr,
    out_dw_ptr,
    out_dv_ptr,
    out_dlog_alpha_ptr,
    b_size: tl.constexpr,
    h_size: tl.constexpr,
    bh_size: tl.constexpr,
    nc_size: tl.constexpr,
    c_size: tl.constexpr,
    m_size: tl.constexpr,
    d_size: tl.constexpr,
    stride_grad_s_bh: tl.constexpr,
    stride_grad_s_nc: tl.constexpr,
    stride_grad_s_m: tl.constexpr,
    stride_grad_s_d: tl.constexpr,
    stride_w_b: tl.constexpr,
    stride_w_nc: tl.constexpr,
    stride_w_c: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_nc: tl.constexpr,
    stride_v_c: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_r_b: tl.constexpr,
    stride_r_nc: tl.constexpr,
    stride_r_c: tl.constexpr,
    stride_r_h: tl.constexpr,
    stride_out_dw_bh: tl.constexpr,
    stride_out_dw_nc: tl.constexpr,
    stride_out_dw_c: tl.constexpr,
    stride_out_dw_m: tl.constexpr,
    stride_out_dv_bh: tl.constexpr,
    stride_out_dv_nc: tl.constexpr,
    stride_out_dv_c: tl.constexpr,
    stride_out_dv_d: tl.constexpr,
    stride_out_dlog_alpha_bh: tl.constexpr,
    stride_out_dlog_alpha_nc: tl.constexpr,
    stride_out_dlog_alpha_c: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    M_STATIC: tl.constexpr,
    D_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    """Phase-1 backward fused kernel for dW, dV, and dlog_alpha per `(BH,NC)` chunk."""
    pid_bhnc = tl.program_id(0)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    grad_s_desc = tl.make_tensor_descriptor(
        grad_s_local_end_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_grad_s_bh, stride_grad_s_nc, stride_grad_s_m, stride_grad_s_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    w_desc = tl.make_tensor_descriptor(
        W_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    out_dw_desc = tl.make_tensor_descriptor(
        out_dw_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_out_dw_bh, stride_out_dw_nc, stride_out_dw_c, stride_out_dw_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    out_dv_desc = tl.make_tensor_descriptor(
        out_dv_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_out_dv_bh, stride_out_dv_nc, stride_out_dv_c, stride_out_dv_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    out_dr_desc = tl.make_tensor_descriptor(
        out_dlog_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_out_dlog_alpha_bh, stride_out_dlog_alpha_nc, stride_out_dlog_alpha_c],
        block_shape=[1, 1, BLOCK_C],
    )

    offs_c = tl.arange(0, BLOCK_C)
    r_base = B_IDX * stride_r_b + NC_IDX * stride_r_nc + H_IDX * stride_r_h
    log_alpha_in = tl.load(log_alpha_ptr + r_base + offs_c * stride_r_c)
    log_alpha_vals = log_alpha_in.to(tl.float32)
    INV_LN2 = 1.4426950408889634
    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    factors = tl.exp2((log_suffix_incl - log_alpha_vals) * INV_LN2)

    # dW over M-tiles.
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        acc = tl.zeros((BLOCK_C, BLOCK_M), dtype=tl.float32)
        for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
            d_start = d_blk * BLOCK_D
            g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
            v_tile = tl.reshape(v_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
            g_tile_tc = g_tile.to(tl.bfloat16)
            v_tile_tc = v_tile.to(tl.bfloat16)
            acc = acc + tl.dot(
                v_tile_tc,
                tl.trans(g_tile_tc),
                out_dtype=tl.float32,
                input_precision=INPUT_PRECISION,
            )
        dW_tile = factors[:, None] * acc
        w_dtype_probe = tl.reshape(w_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        if ACCUMULATE:
            old_dw = tl.reshape(out_dw_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
            dW_tile += old_dw
        out_dw_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW_tile.to(w_dtype_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))

    # dV and b_vals over D-tiles.
    b_vals = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
        d_start = d_blk * BLOCK_D
        acc = tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32)
        for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
            m_start = m_blk * BLOCK_M
            g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
            w_tile = tl.reshape(w_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
            g_tile_tc = g_tile.to(tl.bfloat16)
            w_tile_tc = w_tile.to(tl.bfloat16)
            acc = acc + tl.dot(
                w_tile_tc,
                g_tile_tc,
                out_dtype=tl.float32,
                input_precision=INPUT_PRECISION,
            )
        dV_tile = factors[:, None] * acc
        v_dtype_probe = tl.reshape(v_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
        if ACCUMULATE:
            old_dv = tl.reshape(out_dv_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
            dV_tile += old_dv
        out_dv_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV_tile.to(v_dtype_probe.dtype), (1, 1, BLOCK_C, BLOCK_D)))

        v_tile = tl.reshape(v_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
        b_vals += tl.sum(acc * v_tile, axis=1)

    g_vals = b_vals * factors
    prefix_incl = tl.cumsum(g_vals, axis=0)
    grad_vec = prefix_incl - g_vals
    if ACCUMULATE:
        grad_vec += tl.reshape(out_dr_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    out_dr_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(grad_vec.to(log_alpha_in.dtype), (1, 1, BLOCK_C)))


@triton.jit
def ssd_rank4_chunk_end_state_fwd_kernel(
    W1_ptr,
    V1_ptr,
    W2_ptr,
    V2_ptr,
    W3_ptr,
    V3_ptr,
    W4_ptr,
    V4_ptr,
    log_alpha_ptr,
    out_s_local_end_ptr,
    b_size: tl.constexpr,
    h_size: tl.constexpr,
    bh_size: tl.constexpr,
    nc_size: tl.constexpr,
    c_size: tl.constexpr,
    m_size: tl.constexpr,
    d_size: tl.constexpr,
    stride_w_b: tl.constexpr,
    stride_w_nc: tl.constexpr,
    stride_w_c: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_nc: tl.constexpr,
    stride_v_c: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_r_b: tl.constexpr,
    stride_r_nc: tl.constexpr,
    stride_r_c: tl.constexpr,
    stride_r_h: tl.constexpr,
    stride_out_s_local_end_bh: tl.constexpr,
    stride_out_s_local_end_nc: tl.constexpr,
    stride_out_s_local_end_m: tl.constexpr,
    stride_out_s_local_end_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    RANK: tl.constexpr,
):
    """Phase-1 rank-4 forward: fused write accumulation with static rank guards."""
    pid_bhnc = tl.program_id(0)
    pid_m_tile = tl.program_id(1)
    pid_d_tile = tl.program_id(2)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    m_start = pid_m_tile * BLOCK_M
    d_start = pid_d_tile * BLOCK_D

    w1_desc = tl.make_tensor_descriptor(
        W1_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_T, 1, BLOCK_M],
    )
    v1_desc = tl.make_tensor_descriptor(
        V1_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_T, 1, BLOCK_D],
    )
    w2_desc = tl.make_tensor_descriptor(
        W2_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_T, 1, BLOCK_M],
    )
    v2_desc = tl.make_tensor_descriptor(
        V2_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_T, 1, BLOCK_D],
    )
    w3_desc = tl.make_tensor_descriptor(
        W3_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_T, 1, BLOCK_M],
    )
    v3_desc = tl.make_tensor_descriptor(
        V3_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_T, 1, BLOCK_D],
    )
    w4_desc = tl.make_tensor_descriptor(
        W4_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_T, 1, BLOCK_M],
    )
    v4_desc = tl.make_tensor_descriptor(
        V4_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_T, 1, BLOCK_D],
    )
    out_s_desc = tl.make_tensor_descriptor(
        out_s_local_end_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_out_s_local_end_bh, stride_out_s_local_end_nc, stride_out_s_local_end_m, stride_out_s_local_end_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )

    offs_c = tl.arange(0, BLOCK_C)
    r_base = B_IDX * stride_r_b + NC_IDX * stride_r_nc + H_IDX * stride_r_h
    log_alpha_in = tl.load(log_alpha_ptr + r_base + offs_c * stride_r_c)
    log_alpha_vals = log_alpha_in.to(tl.float32)
    INV_LN2 = 1.4426950408889634
    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    r_factors = tl.exp2((log_suffix_incl - log_alpha_vals) * INV_LN2)

    S = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    offs_t_full = tl.arange(0, BLOCK_C)
    for t_start in tl.static_range(0, C_STATIC, BLOCK_T):
        offs_tb = t_start + tl.arange(0, BLOCK_T)
        factors = tl.sum(
            tl.where(offs_t_full[None, :] == offs_tb[:, None], r_factors[None, :], 0.0),
            axis=1,
        )
        sqrt_factors = tl.sqrt(factors)[:, None]

        W1_blk = tl.reshape(w1_desc.load([B_IDX, NC_IDX, t_start, H_IDX, m_start]), (BLOCK_T, BLOCK_M))
        V1_blk = tl.reshape(v1_desc.load([B_IDX, NC_IDX, t_start, H_IDX, d_start]), (BLOCK_T, BLOCK_D))
        A1_f32 = sqrt_factors * W1_blk.to(tl.float32)
        B1_f32 = sqrt_factors * V1_blk.to(tl.float32)
        S = S + tl.dot(
            tl.trans(A1_f32.to(tl.bfloat16)),
            B1_f32.to(tl.bfloat16),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
        if RANK >= 2:
            W2_blk = tl.reshape(w2_desc.load([B_IDX, NC_IDX, t_start, H_IDX, m_start]), (BLOCK_T, BLOCK_M))
            V2_blk = tl.reshape(v2_desc.load([B_IDX, NC_IDX, t_start, H_IDX, d_start]), (BLOCK_T, BLOCK_D))
            A2_f32 = sqrt_factors * W2_blk.to(tl.float32)
            B2_f32 = sqrt_factors * V2_blk.to(tl.float32)
            S = S + tl.dot(
                tl.trans(A2_f32.to(tl.bfloat16)),
                B2_f32.to(tl.bfloat16),
                out_dtype=tl.float32,
                input_precision=INPUT_PRECISION,
            )
        if RANK >= 3:
            W3_blk = tl.reshape(w3_desc.load([B_IDX, NC_IDX, t_start, H_IDX, m_start]), (BLOCK_T, BLOCK_M))
            V3_blk = tl.reshape(v3_desc.load([B_IDX, NC_IDX, t_start, H_IDX, d_start]), (BLOCK_T, BLOCK_D))
            A3_f32 = sqrt_factors * W3_blk.to(tl.float32)
            B3_f32 = sqrt_factors * V3_blk.to(tl.float32)
            S = S + tl.dot(
                tl.trans(A3_f32.to(tl.bfloat16)),
                B3_f32.to(tl.bfloat16),
                out_dtype=tl.float32,
                input_precision=INPUT_PRECISION,
            )
        if RANK >= 4:
            W4_blk = tl.reshape(w4_desc.load([B_IDX, NC_IDX, t_start, H_IDX, m_start]), (BLOCK_T, BLOCK_M))
            V4_blk = tl.reshape(v4_desc.load([B_IDX, NC_IDX, t_start, H_IDX, d_start]), (BLOCK_T, BLOCK_D))
            A4_f32 = sqrt_factors * W4_blk.to(tl.float32)
            B4_f32 = sqrt_factors * V4_blk.to(tl.float32)
            S = S + tl.dot(
                tl.trans(A4_f32.to(tl.bfloat16)),
                B4_f32.to(tl.bfloat16),
                out_dtype=tl.float32,
                input_precision=INPUT_PRECISION,
            )
    out_s_desc.store([BH_IDX, NC_IDX, m_start, d_start], tl.reshape(S, (1, 1, BLOCK_M, BLOCK_D)))


@triton.jit
def ssd_rank4_chunk_end_state_bwd_fused_kernel(
    grad_s_local_end_ptr,
    W1_ptr,
    V1_ptr,
    W2_ptr,
    V2_ptr,
    W3_ptr,
    V3_ptr,
    W4_ptr,
    V4_ptr,
    log_alpha_ptr,
    out_dw1_ptr,
    out_dv1_ptr,
    out_dw2_ptr,
    out_dv2_ptr,
    out_dw3_ptr,
    out_dv3_ptr,
    out_dw4_ptr,
    out_dv4_ptr,
    out_dlog_alpha_ptr,
    b_size: tl.constexpr,
    h_size: tl.constexpr,
    bh_size: tl.constexpr,
    nc_size: tl.constexpr,
    c_size: tl.constexpr,
    m_size: tl.constexpr,
    d_size: tl.constexpr,
    stride_grad_s_bh: tl.constexpr,
    stride_grad_s_nc: tl.constexpr,
    stride_grad_s_m: tl.constexpr,
    stride_grad_s_d: tl.constexpr,
    stride_w_b: tl.constexpr,
    stride_w_nc: tl.constexpr,
    stride_w_c: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_nc: tl.constexpr,
    stride_v_c: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_r_b: tl.constexpr,
    stride_r_nc: tl.constexpr,
    stride_r_c: tl.constexpr,
    stride_r_h: tl.constexpr,
    stride_out_dw_bh: tl.constexpr,
    stride_out_dw_nc: tl.constexpr,
    stride_out_dw_c: tl.constexpr,
    stride_out_dw_m: tl.constexpr,
    stride_out_dv_bh: tl.constexpr,
    stride_out_dv_nc: tl.constexpr,
    stride_out_dv_c: tl.constexpr,
    stride_out_dv_d: tl.constexpr,
    stride_out_dlog_alpha_bh: tl.constexpr,
    stride_out_dlog_alpha_nc: tl.constexpr,
    stride_out_dlog_alpha_c: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    M_STATIC: tl.constexpr,
    D_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    RANK: tl.constexpr,
):
    """Phase-1 rank-4 backward fused kernel."""
    pid_bhnc = tl.program_id(0)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    grad_s_desc = tl.make_tensor_descriptor(
        grad_s_local_end_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_grad_s_bh, stride_grad_s_nc, stride_grad_s_m, stride_grad_s_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    w1_desc = tl.make_tensor_descriptor(
        W1_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v1_desc = tl.make_tensor_descriptor(
        V1_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    w2_desc = tl.make_tensor_descriptor(
        W2_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v2_desc = tl.make_tensor_descriptor(
        V2_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    w3_desc = tl.make_tensor_descriptor(
        W3_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v3_desc = tl.make_tensor_descriptor(
        V3_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    w4_desc = tl.make_tensor_descriptor(
        W4_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v4_desc = tl.make_tensor_descriptor(
        V4_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    out_dw1_desc = tl.make_tensor_descriptor(
        out_dw1_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_out_dw_bh, stride_out_dw_nc, stride_out_dw_c, stride_out_dw_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    out_dv1_desc = tl.make_tensor_descriptor(
        out_dv1_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_out_dv_bh, stride_out_dv_nc, stride_out_dv_c, stride_out_dv_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    out_dw2_desc = tl.make_tensor_descriptor(
        out_dw2_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_out_dw_bh, stride_out_dw_nc, stride_out_dw_c, stride_out_dw_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    out_dv2_desc = tl.make_tensor_descriptor(
        out_dv2_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_out_dv_bh, stride_out_dv_nc, stride_out_dv_c, stride_out_dv_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    out_dw3_desc = tl.make_tensor_descriptor(
        out_dw3_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_out_dw_bh, stride_out_dw_nc, stride_out_dw_c, stride_out_dw_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    out_dv3_desc = tl.make_tensor_descriptor(
        out_dv3_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_out_dv_bh, stride_out_dv_nc, stride_out_dv_c, stride_out_dv_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    out_dw4_desc = tl.make_tensor_descriptor(
        out_dw4_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_out_dw_bh, stride_out_dw_nc, stride_out_dw_c, stride_out_dw_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    out_dv4_desc = tl.make_tensor_descriptor(
        out_dv4_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_out_dv_bh, stride_out_dv_nc, stride_out_dv_c, stride_out_dv_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    out_dr_desc = tl.make_tensor_descriptor(
        out_dlog_alpha_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_out_dlog_alpha_bh, stride_out_dlog_alpha_nc, stride_out_dlog_alpha_c],
        block_shape=[1, 1, BLOCK_C],
    )

    offs_c = tl.arange(0, BLOCK_C)
    r_base = B_IDX * stride_r_b + NC_IDX * stride_r_nc + H_IDX * stride_r_h
    log_alpha_in = tl.load(log_alpha_ptr + r_base + offs_c * stride_r_c)
    log_alpha_vals = log_alpha_in.to(tl.float32)
    INV_LN2 = 1.4426950408889634
    log_suffix_incl = tl.cumsum(log_alpha_vals, axis=0, reverse=True)
    factors = tl.exp2((log_suffix_incl - log_alpha_vals) * INV_LN2)

    # dW over M tiles, processed rank-by-rank to reduce live accumulator pressure.
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        acc1 = tl.zeros((BLOCK_C, BLOCK_M), dtype=tl.float32)
        for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
            d_start = d_blk * BLOCK_D
            g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
            g_tile_tc = g_tile.to(tl.bfloat16)
            v1_tile = tl.reshape(v1_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
            acc1 = acc1 + tl.dot(v1_tile.to(tl.bfloat16), tl.trans(g_tile_tc), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        dW1_tile = factors[:, None] * acc1
        w1_probe = tl.reshape(w1_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        if ACCUMULATE:
            dW1_tile += tl.reshape(out_dw1_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        out_dw1_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW1_tile.to(w1_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))

        if RANK >= 2:
            acc2 = tl.zeros((BLOCK_C, BLOCK_M), dtype=tl.float32)
            for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
                d_start = d_blk * BLOCK_D
                g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
                g_tile_tc = g_tile.to(tl.bfloat16)
                v2_tile = tl.reshape(v2_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
                acc2 = acc2 + tl.dot(v2_tile.to(tl.bfloat16), tl.trans(g_tile_tc), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dW2_tile = factors[:, None] * acc2
            w2_probe = tl.reshape(w2_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            if ACCUMULATE:
                dW2_tile += tl.reshape(out_dw2_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
            out_dw2_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW2_tile.to(w2_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))

        if RANK >= 3:
            acc3 = tl.zeros((BLOCK_C, BLOCK_M), dtype=tl.float32)
            for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
                d_start = d_blk * BLOCK_D
                g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
                g_tile_tc = g_tile.to(tl.bfloat16)
                v3_tile = tl.reshape(v3_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
                acc3 = acc3 + tl.dot(v3_tile.to(tl.bfloat16), tl.trans(g_tile_tc), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dW3_tile = factors[:, None] * acc3
            w3_probe = tl.reshape(w3_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            if ACCUMULATE:
                dW3_tile += tl.reshape(out_dw3_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
            out_dw3_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW3_tile.to(w3_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))
        if RANK >= 4:
            acc4 = tl.zeros((BLOCK_C, BLOCK_M), dtype=tl.float32)
            for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
                d_start = d_blk * BLOCK_D
                g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
                g_tile_tc = g_tile.to(tl.bfloat16)
                v4_tile = tl.reshape(v4_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
                acc4 = acc4 + tl.dot(v4_tile.to(tl.bfloat16), tl.trans(g_tile_tc), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dW4_tile = factors[:, None] * acc4
            w4_probe = tl.reshape(w4_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            if ACCUMULATE:
                dW4_tile += tl.reshape(out_dw4_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
            out_dw4_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW4_tile.to(w4_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))

    # dV and dlog ingredients, likewise rank-by-rank.
    b_vals = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
        d_start = d_blk * BLOCK_D
        acc1 = tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32)
        for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
            m_start = m_blk * BLOCK_M
            g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
            g_tile_tc = g_tile.to(tl.bfloat16)
            w1_tile = tl.reshape(w1_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
            acc1 = acc1 + tl.dot(w1_tile.to(tl.bfloat16), g_tile_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        dV1_tile = factors[:, None] * acc1
        v1_probe = tl.reshape(v1_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
        if ACCUMULATE:
            dV1_tile += tl.reshape(out_dv1_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
        out_dv1_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV1_tile.to(v1_probe.dtype), (1, 1, BLOCK_C, BLOCK_D)))
        v1_tile = tl.reshape(v1_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
        b_vals += tl.sum(acc1 * v1_tile, axis=1)

        if RANK >= 2:
            acc2 = tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32)
            for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
                m_start = m_blk * BLOCK_M
                g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
                g_tile_tc = g_tile.to(tl.bfloat16)
                w2_tile = tl.reshape(w2_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
                acc2 = acc2 + tl.dot(w2_tile.to(tl.bfloat16), g_tile_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dV2_tile = factors[:, None] * acc2
            v2_probe = tl.reshape(v2_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
            if ACCUMULATE:
                dV2_tile += tl.reshape(out_dv2_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
            out_dv2_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV2_tile.to(v2_probe.dtype), (1, 1, BLOCK_C, BLOCK_D)))
            v2_tile = tl.reshape(v2_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
            b_vals += tl.sum(acc2 * v2_tile, axis=1)

        if RANK >= 3:
            acc3 = tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32)
            for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
                m_start = m_blk * BLOCK_M
                g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
                g_tile_tc = g_tile.to(tl.bfloat16)
                w3_tile = tl.reshape(w3_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
                acc3 = acc3 + tl.dot(w3_tile.to(tl.bfloat16), g_tile_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dV3_tile = factors[:, None] * acc3
            v3_probe = tl.reshape(v3_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
            if ACCUMULATE:
                dV3_tile += tl.reshape(out_dv3_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
            out_dv3_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV3_tile.to(v3_probe.dtype), (1, 1, BLOCK_C, BLOCK_D)))
            v3_tile = tl.reshape(v3_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
            b_vals += tl.sum(acc3 * v3_tile, axis=1)
        if RANK >= 4:
            acc4 = tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32)
            for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
                m_start = m_blk * BLOCK_M
                g_tile = tl.reshape(grad_s_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D)).to(tl.float32)
                g_tile_tc = g_tile.to(tl.bfloat16)
                w4_tile = tl.reshape(w4_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
                acc4 = acc4 + tl.dot(w4_tile.to(tl.bfloat16), g_tile_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dV4_tile = factors[:, None] * acc4
            v4_probe = tl.reshape(v4_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
            if ACCUMULATE:
                dV4_tile += tl.reshape(out_dv4_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
            out_dv4_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV4_tile.to(v4_probe.dtype), (1, 1, BLOCK_C, BLOCK_D)))
            v4_tile = tl.reshape(v4_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
            b_vals += tl.sum(acc4 * v4_tile, axis=1)

    g_vals = b_vals * factors
    prefix_incl = tl.cumsum(g_vals, axis=0)
    grad_vec = prefix_incl - g_vals
    if ACCUMULATE:
        grad_vec += tl.reshape(out_dr_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    out_dr_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(grad_vec.to(log_alpha_in.dtype), (1, 1, BLOCK_C)))


# ==================================================================================================
# PHASE 1 AUTOGRAD
# ==================================================================================================

def _ssd_rank1_chunk_end_state_forward_impl(
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    BLOCK_T: int = 32,
) -> torch.Tensor:
    if W.device != V.device or W.device != log_alpha.device:
        raise ValueError("W, V, log_alpha must be on the same device.")
    if W.dtype != V.dtype or W.dtype != log_alpha.dtype:
        raise ValueError("W, V, log_alpha must share dtype.")
    if W.ndim == 5 and V.ndim == 5 and log_alpha.ndim == 4:
        B, NC, C, H, M = W.shape
        if V.shape[:4] != (B, NC, C, H):
            raise ValueError(f"V must be [B,NC,C,H,D] with leading dims {(B, NC, C, H)}; got {tuple(V.shape)}.")
        D = V.shape[-1]
        if log_alpha.shape != (B, NC, C, H):
            raise ValueError(
                "log_alpha must match [B,NC,C,H] from W/V. "
                f"Got log_alpha={tuple(log_alpha.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}."
            )
        W_5d, V_5d, log_alpha_4d = W, V, log_alpha
    elif W.ndim == 4 and V.ndim == 4 and log_alpha.ndim == 3:
        BH, NC, C, M = W.shape
        if V.shape[:3] != (BH, NC, C):
            raise ValueError(
                "V must match [BH,NC,C,*] from W. "
                f"Got V={tuple(V.shape)}, W={tuple(W.shape)}."
            )
        D = V.shape[-1]
        if log_alpha.shape != (BH, NC, C):
            raise ValueError(
                "log_alpha must match [BH,NC,C] from W/V. "
                f"Got log_alpha={tuple(log_alpha.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}."
            )
        B, H = BH, 1
        W_5d = W.unsqueeze(3)
        V_5d = V.unsqueeze(3)
        log_alpha_4d = log_alpha.unsqueeze(3)
    else:
        raise ValueError(
            "_ssd_rank1_chunk_end_state_forward_impl expects either "
            "W=[B,NC,C,H,M], V=[B,NC,C,H,D], log_alpha=[B,NC,C,H] "
            "or legacy W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
            f"Got W={tuple(W.shape)}, V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )

    BH = B * H
    _require_supported_md(M, D, where="_ssd_rank1_chunk_end_state_forward_impl")
    _require_chunk_size_multiple_of_16(C, where="_ssd_rank1_chunk_end_state_forward_impl")
    BLOCK_T = _select_largest_block_size(
        C,
        _SUPPORTED_BLOCK_T_VALUES,
        where="_ssd_rank1_chunk_end_state_forward_impl",
        label="C",
    )
    _require_nc_descriptor_width(NC, where="_ssd_rank1_chunk_end_state_forward_impl")
    _require_phase1_block_t(BLOCK_T, C, where="_ssd_rank1_chunk_end_state_forward_impl")
    _require_nonpositive_log_alpha(log_alpha_4d, where="_ssd_rank1_chunk_end_state_forward_impl")
    if not W_5d.is_cuda:
            raise NotImplementedError("_ssd_rank1_chunk_end_state_forward_impl requires CUDA tensors.")
    _ensure_triton_allocator()

    s_local_end_md = torch.empty((BH, NC, M, D), device=W_5d.device, dtype=torch.float32)
    BLOCK_M = _select_largest_block_size(
        M,
        _SUPPORTED_BLOCK_X_VALUES,
        where="_ssd_rank1_chunk_end_state_forward_impl",
        label="M",
    )
    BLOCK_D = _select_largest_block_size(
        D,
        _SUPPORTED_BLOCK_X_VALUES,
        where="_ssd_rank1_chunk_end_state_forward_impl",
        label="D",
    )
    BLOCK_T = _select_largest_block_size(
        C,
        _SUPPORTED_BLOCK_T_VALUES,
        where="_ssd_rank1_chunk_end_state_forward_impl",
        label="C",
    )
    grid = (BH * NC, M // BLOCK_M, D // BLOCK_D)
    ssd_rank1_chunk_end_state_fwd_kernel[grid](
        W_5d,
        V_5d,
        log_alpha_4d,
        s_local_end_md,
        B,
        H,
        BH,
        NC,
        C,
        M,
        D,
        *W_5d.stride(),
        *V_5d.stride(),
        *log_alpha_4d.stride(),
        *s_local_end_md.stride(),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        BLOCK_T=BLOCK_T,
        BLOCK_C=C,
        C_STATIC=C,
        INPUT_PRECISION="ieee" if W_5d.dtype == torch.float32 else "tf32",
    )
    return s_local_end_md.reshape(BH, NC, M * D)


# --------------------------------------------------------------------------------------------------
# PHASE 2: PREFIX SCAN OVER CHUNKS
# --------------------------------------------------------------------------------------------------
@triton.jit
def ssd_rank1_prefix_scan_fwd_kernel(
    s_local_ptr,
    log_alpha_chunk_ptr,
    init_ptr,
    out_prefix_ptr,
    out_final_ptr,
    b_size,
    h_size,
    bh_size,
    nc_size,
    md_size,
    stride_s_local_bh,
    stride_s_local_nc,
    stride_s_local_md,
    stride_log_alpha_chunk_b,
    stride_log_alpha_chunk_nc,
    stride_log_alpha_chunk_h,
    stride_init_bh,
    stride_init_md,
    stride_out_prefix_bh,
    stride_out_prefix_nc,
    stride_out_prefix_md,
    stride_out_final_bh,
    stride_out_final_md,
    BLOCK_NC: tl.constexpr,
    BLOCK_MD: tl.constexpr,
    NC_STATIC: tl.constexpr,
    USE_FP32_COMPUTE: tl.constexpr,
    HAS_INITIAL_STATE: tl.constexpr,
    RETURN_FINAL_STATE: tl.constexpr,
):
    """Single-kernel dense blocked phase-2 forward.

    One program owns one `BH` lane and loops over chunk blocks then MD tiles:
      1) Form local block prefixes in log space.
      2) Build direct strict-lower carry matrix for this block:
           L0[i,j] = 1_{j<i} * exp(log_p_excl[i] - log_p_incl[j]).
      3) Compute block chunk-start states:
           S0_block = L0 @ S_loc + p[:, None] * S_in[None, :].
      3) Store `S0_block`, update carry:
           S_in <- alpha_last * S0_block_last + S_local_block_last
      4) Repeat for next block.

    After loop:
      out_final = S_in
    """
    pid_bh = tl.program_id(0)
    if pid_bh >= bh_size:
        return

    s_local_desc = tl.make_tensor_descriptor(
        s_local_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_s_local_bh, stride_s_local_nc, stride_s_local_md],
        block_shape=[1, BLOCK_NC, BLOCK_MD],
    )
    init_desc = tl.make_tensor_descriptor(
        init_ptr,
        shape=[bh_size, md_size],
        strides=[stride_init_bh, stride_init_md],
        block_shape=[1, BLOCK_MD],
    )
    out_prefix_desc = tl.make_tensor_descriptor(
        out_prefix_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_out_prefix_bh, stride_out_prefix_nc, stride_out_prefix_md],
        block_shape=[1, BLOCK_NC, BLOCK_MD],
    )
    out_final_desc = tl.make_tensor_descriptor(
        out_final_ptr,
        shape=[bh_size, md_size],
        strides=[stride_out_final_bh, stride_out_final_md],
        block_shape=[1, BLOCK_MD],
    )

    INV_LN2 = 1.4426950408889634
    offs_blk = tl.arange(0, BLOCK_NC)
    row_ids = tl.expand_dims(offs_blk, axis=1)
    col_ids = tl.expand_dims(offs_blk, axis=0)
    l0_mask = row_ids > col_ids
    last_mask = offs_blk == (BLOCK_NC - 1)

    for c0 in tl.static_range(0, NC_STATIC, BLOCK_NC):
        # L-related block math (prefixes and scaling vectors) computed once per NC block.
        offs_nc = tl.arange(0, BLOCK_NC)
        b_idx = pid_bh // h_size
        h_idx = pid_bh - b_idx * h_size
        log_alpha_ptrs = (
            log_alpha_chunk_ptr
            + b_idx * stride_log_alpha_chunk_b
            + (c0 + offs_nc) * stride_log_alpha_chunk_nc
            + h_idx * stride_log_alpha_chunk_h
        )
        log_alpha_blk = tl.load(log_alpha_ptrs).to(tl.float32)
        log_alpha_blk_l2 = log_alpha_blk * INV_LN2
        log_p_incl_l2 = tl.cumsum(log_alpha_blk_l2, axis=0)
        log_p_excl_l2 = log_p_incl_l2 - log_alpha_blk_l2
        p = tl.exp2(log_p_excl_l2)
        alpha_last = tl.exp2(tl.sum(tl.where(last_mask, log_alpha_blk_l2, 0.0), axis=0))
        l0_log2 = tl.expand_dims(log_p_excl_l2, axis=1) - tl.expand_dims(log_p_incl_l2, axis=0)
        l0_f32 = tl.where(l0_mask, tl.exp2(l0_log2), 0.0)
        if USE_FP32_COMPUTE:
            l0_tc = l0_f32
        else:
            l0_tc = l0_f32.to(tl.bfloat16)

        md0 = 0
        while md0 < md_size:
            if c0 == 0:
                if HAS_INITIAL_STATE:
                    S_in = tl.reshape(init_desc.load([pid_bh, md0]), (BLOCK_MD,)).to(tl.float32)
                else:
                    S_in = tl.zeros((BLOCK_MD,), dtype=tl.float32)
            else:
                S_in = tl.reshape(out_final_desc.load([pid_bh, md0]), (BLOCK_MD,)).to(tl.float32)

            s_tile_f32 = tl.reshape(s_local_desc.load([pid_bh, c0, md0]), (BLOCK_NC, BLOCK_MD)).to(tl.float32)
            if USE_FP32_COMPUTE:
                y0 = tl.dot(l0_tc, s_tile_f32, out_dtype=tl.float32, input_precision="ieee")
            else:
                s_tile_tc = s_tile_f32.to(tl.bfloat16)
                y0 = tl.dot(l0_tc, s_tile_tc, out_dtype=tl.float32, input_precision="tf32")

            acc = y0 + tl.expand_dims(p, axis=1) * tl.expand_dims(S_in, axis=0)

            out_prefix_desc.store([pid_bh, c0, md0], tl.reshape(acc, (1, BLOCK_NC, BLOCK_MD)))

            s0_last = tl.sum(tl.where(tl.expand_dims(last_mask, axis=1), acc, 0.0), axis=0)
            s_local_last = tl.sum(tl.where(tl.expand_dims(last_mask, axis=1), s_tile_f32, 0.0), axis=0)
            S_next = alpha_last * s0_last + s_local_last
            if RETURN_FINAL_STATE or c0 != (NC_STATIC - BLOCK_NC):
                out_final_desc.store([pid_bh, md0], tl.reshape(S_next, (1, BLOCK_MD)))
            md0 += BLOCK_MD


@triton.jit
def ssd_rank1_prefix_scan_bwd_dense_kernel(
    grad_prefix_ptr,
    grad_final_ptr,
    chunk_start_ptr,
    log_alpha_chunk_ptr,
    d_s_local_ptr,
    d_log_alpha_ptr,
    d_init_ptr,
    b_size,
    h_size,
    bh_size,
    nc_size,
    md_size,
    stride_grad_prefix_bh,
    stride_grad_prefix_nc,
    stride_grad_prefix_md,
    stride_grad_final_bh,
    stride_grad_final_md,
    stride_chunk_start_bh,
    stride_chunk_start_nc,
    stride_chunk_start_md,
    stride_log_alpha_chunk_b,
    stride_log_alpha_chunk_nc,
    stride_log_alpha_chunk_h,
    stride_d_s_local_bh,
    stride_d_s_local_nc,
    stride_d_s_local_md,
    stride_d_log_alpha_bh,
    stride_d_log_alpha_nc,
    stride_d_init_bh,
    stride_d_init_md,
    BLOCK_MD: tl.constexpr,
    BLOCK_NC: tl.constexpr,
    NC_STATIC: tl.constexpr,
    USE_FP32_COMPUTE: tl.constexpr,
    HAS_GRAD_FINAL: tl.constexpr,
    WRITE_D_INIT: tl.constexpr,
):
    """Dense blocked reverse pass for phase-2 prefix scan.

    Recurrence in chunk index c:
      S_{c+1} = alpha[c] * S_c + S_local[c]
      y_start[c] = S_c
      y_final = S_NC

    Reverse recurrence:
      lambda_NC = g_final
      lambda_c  = g_start[c] + alpha[c] * lambda_{c+1}
      dS_local[c] = lambda_{c+1}
      dalpha[c]   = sum_md(lambda_{c+1} * S_c)
      dlog_alpha[c] = dalpha[c] * alpha[c]

    Kernel structure mirrors forward:
      - one program per BH lane
      - static loop over NC blocks in reverse order
      - inner loop over MD tiles
      - per block uses dense strict-lower carry matrix in reversed chunk order:
          lambda_next_rev = L0_rev @ g_rev + p_rev * lambda_in
    """
    pid_bh = tl.program_id(0)
    if pid_bh >= bh_size:
        return

    grad_prefix_desc = tl.make_tensor_descriptor(
        grad_prefix_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_grad_prefix_bh, stride_grad_prefix_nc, stride_grad_prefix_md],
        block_shape=[1, BLOCK_NC, BLOCK_MD],
    )
    grad_final_desc = tl.make_tensor_descriptor(
        grad_final_ptr,
        shape=[bh_size, md_size],
        strides=[stride_grad_final_bh, stride_grad_final_md],
        block_shape=[1, BLOCK_MD],
    )
    chunk_start_desc = tl.make_tensor_descriptor(
        chunk_start_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_chunk_start_bh, stride_chunk_start_nc, stride_chunk_start_md],
        block_shape=[1, BLOCK_NC, BLOCK_MD],
    )
    d_s_local_desc = tl.make_tensor_descriptor(
        d_s_local_ptr,
        shape=[bh_size, nc_size, md_size],
        strides=[stride_d_s_local_bh, stride_d_s_local_nc, stride_d_s_local_md],
        block_shape=[1, BLOCK_NC, BLOCK_MD],
    )
    d_log_alpha_desc = tl.make_tensor_descriptor(
        d_log_alpha_ptr,
        shape=[bh_size, nc_size],
        strides=[stride_d_log_alpha_bh, stride_d_log_alpha_nc],
        block_shape=[1, BLOCK_NC],
    )
    d_init_desc = tl.make_tensor_descriptor(
        d_init_ptr,
        shape=[bh_size, md_size],
        strides=[stride_d_init_bh, stride_d_init_md],
        block_shape=[1, BLOCK_MD],
    )

    INV_LN2 = 1.4426950408889634
    offs_nc = tl.arange(0, BLOCK_NC)
    row_ids = tl.expand_dims(offs_nc, axis=1)
    col_ids = tl.expand_dims(offs_nc, axis=0)
    l0_mask = row_ids > col_ids
    last_mask = offs_nc == (BLOCK_NC - 1)

    for blk in tl.static_range(0, NC_STATIC // BLOCK_NC):
        c0 = NC_STATIC - (blk + 1) * BLOCK_NC

        b_idx = pid_bh // h_size
        h_idx = pid_bh - b_idx * h_size
        log_alpha_ptrs = (
            log_alpha_chunk_ptr
            + b_idx * stride_log_alpha_chunk_b
            + (c0 + offs_nc) * stride_log_alpha_chunk_nc
            + h_idx * stride_log_alpha_chunk_h
        )
        log_alpha_blk = tl.load(log_alpha_ptrs).to(tl.float32)
        log_alpha_blk_l2 = log_alpha_blk * INV_LN2
        log_alpha_rev_l2 = tl.flip(log_alpha_blk_l2, 0)
        log_p_incl_rev_l2 = tl.cumsum(log_alpha_rev_l2, axis=0)
        log_p_excl_rev_l2 = log_p_incl_rev_l2 - log_alpha_rev_l2
        p_rev = tl.exp2(log_p_excl_rev_l2)
        alpha_last_rev = tl.exp2(tl.sum(tl.where(last_mask, log_alpha_rev_l2, 0.0), axis=0))

        l0_log2 = tl.expand_dims(log_p_excl_rev_l2, axis=1) - tl.expand_dims(log_p_incl_rev_l2, axis=0)
        l0_f32 = tl.where(l0_mask, tl.exp2(l0_log2), 0.0)
        if USE_FP32_COMPUTE:
            l0_tc = l0_f32
        else:
            # Tensor-core path uses BF16 inputs with FP32 accumulation.
            l0_tc = l0_f32.to(tl.bfloat16)

        dlog_block = tl.zeros((BLOCK_NC,), dtype=tl.float32)
        md0 = 0
        while md0 < md_size:
            if blk == 0:
                if HAS_GRAD_FINAL:
                    lambda_in = tl.reshape(grad_final_desc.load([pid_bh, md0]), (BLOCK_MD,)).to(tl.float32)
                else:
                    lambda_in = tl.zeros((BLOCK_MD,), dtype=tl.float32)
            else:
                lambda_in = tl.reshape(d_init_desc.load([pid_bh, md0]), (BLOCK_MD,)).to(tl.float32)

            g_block = tl.reshape(grad_prefix_desc.load([pid_bh, c0, md0]), (BLOCK_NC, BLOCK_MD)).to(tl.float32)
            g_rev = tl.flip(g_block, 0)

            if USE_FP32_COMPUTE:
                y0 = tl.dot(l0_tc, g_rev, out_dtype=tl.float32, input_precision="ieee")
            else:
                g_rev_tc = g_rev.to(tl.bfloat16)
                y0 = tl.dot(l0_tc, g_rev_tc, out_dtype=tl.float32, input_precision="tf32")

            lambda_next_rev = y0 + tl.expand_dims(p_rev, axis=1) * tl.expand_dims(lambda_in, axis=0)
            lambda_next = tl.flip(lambda_next_rev, 0)
            d_s_local_desc.store([pid_bh, c0, md0], tl.reshape(lambda_next, (1, BLOCK_NC, BLOCK_MD)))

            s_block = tl.reshape(chunk_start_desc.load([pid_bh, c0, md0]), (BLOCK_NC, BLOCK_MD)).to(tl.float32)
            dlog_block += tl.sum(lambda_next * s_block, axis=1)

            lambda_start_last = tl.sum(tl.where(tl.expand_dims(last_mask, axis=1), lambda_next_rev, 0.0), axis=0)
            g_last_rev = tl.sum(tl.where(tl.expand_dims(last_mask, axis=1), g_rev, 0.0), axis=0)
            lambda_out = alpha_last_rev * lambda_start_last + g_last_rev
            if WRITE_D_INIT or blk != (NC_STATIC // BLOCK_NC - 1):
                d_init_desc.store([pid_bh, md0], tl.reshape(lambda_out, (1, BLOCK_MD)))
            md0 += BLOCK_MD

        alpha_blk = tl.exp2(log_alpha_blk_l2)
        dlog_out = dlog_block * alpha_blk
        d_log_alpha_desc.store([pid_bh, c0], tl.reshape(dlog_out, (1, BLOCK_NC)))


# ==================================================================================================
# PHASE 2 AUTOGRAD
# ==================================================================================================

def _ssd_rank1_prefix_scan_forward_impl(
    S_local_end: torch.Tensor,
    log_alpha_chunk: torch.Tensor,
    initial_state: torch.Tensor | None,
    compute_dtype: torch.dtype | None = None,
    return_final_state: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if S_local_end.ndim != 3 or log_alpha_chunk.ndim not in (2, 3):
        raise ValueError(
            "_ssd_rank1_prefix_scan_forward_impl expects "
            "S_local_end=[BH,NC,MD], log_alpha_chunk=[B,NC,H] or [BH,NC]. "
            f"Got S_local_end={tuple(S_local_end.shape)}, log_alpha_chunk={tuple(log_alpha_chunk.shape)}."
        )
    BH, NC, MD = S_local_end.shape
    if log_alpha_chunk.ndim == 3:
        B_la, NC_la, H_la = log_alpha_chunk.shape
        if NC_la != NC:
            raise ValueError(f"log_alpha_chunk NC must match S_local_end NC. Got {NC_la} vs {NC}.")
        if B_la * H_la != BH:
            raise ValueError(f"log_alpha_chunk B*H must match BH. Got B={B_la}, H={H_la}, BH={BH}.")
        log_alpha_chunk_bnh = log_alpha_chunk
    else:
        if log_alpha_chunk.shape != (BH, NC):
            raise ValueError(f"log_alpha_chunk must be [BH,NC]={BH, NC} for 2D mode; got {tuple(log_alpha_chunk.shape)}.")
        B_la, H_la = BH, 1
        log_alpha_chunk_bnh = log_alpha_chunk.unsqueeze(-1)
    has_initial_state = initial_state is not None
    if has_initial_state and initial_state.shape != (BH, MD):
        raise ValueError(f"initial_state must be [BH,MD]. Got {tuple(initial_state.shape)} vs ({BH}, {MD}).")
    if S_local_end.device != log_alpha_chunk_bnh.device:
        raise ValueError("S_local_end and log_alpha_chunk must be on the same device.")
    if has_initial_state and S_local_end.device != initial_state.device:
        raise ValueError("S_local_end, log_alpha_chunk, and initial_state must be on the same device.")
    if NC == 0:
        raise NotImplementedError("_ssd_rank1_prefix_scan_forward_impl does not support NC==0.")
    _require_nc_descriptor_width(NC, where="_ssd_rank1_prefix_scan_forward_impl")
    _require_nc_multiple_of_16(NC, where="_ssd_rank1_prefix_scan_forward_impl")
    if not S_local_end.is_cuda:
        raise NotImplementedError("_ssd_rank1_prefix_scan_forward_impl requires CUDA tensors.")
    _ensure_triton_allocator()

    if compute_dtype is None:
        compute_dtype = torch.float32
    if compute_dtype not in (torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "_ssd_rank1_prefix_scan_forward_impl supports compute_dtype in "
            "{torch.bfloat16, torch.float32}. "
            f"Got compute_dtype={compute_dtype}."
        )

    S_local_end_in = S_local_end if S_local_end.is_contiguous() else S_local_end.contiguous()
    if log_alpha_chunk_bnh.dtype == torch.float32 and log_alpha_chunk_bnh.is_contiguous():
        log_alpha_chunk_f = log_alpha_chunk_bnh
    else:
        log_alpha_chunk_f = log_alpha_chunk_bnh.float().contiguous()
    if has_initial_state:
        if initial_state.dtype == torch.float32:
            initial_state_f = initial_state
        else:
            initial_state_f = initial_state.float()
    else:
        initial_state_f = torch.empty((1, 1), device=S_local_end.device, dtype=torch.float32)

    chunk_start = torch.empty((BH, NC, MD), device=S_local_end.device, dtype=torch.float32)
    final_state = torch.empty((BH, MD), device=S_local_end.device, dtype=torch.float32)

    block_nc = _select_phase2_block_nc(NC=NC)
    phase2_cfg = _select_phase2_launch_config(MD=MD, NC=NC, where="_ssd_rank1_prefix_scan_forward_impl")
    def grid_fwd(_meta):
        return (BH,)
    ssd_rank1_prefix_scan_fwd_kernel[grid_fwd](
        S_local_end_in,
        log_alpha_chunk_f,
        initial_state_f,
        chunk_start,
        final_state,
        B_la,
        H_la,
        BH,
        NC,
        MD,
        *S_local_end_in.stride(),
        *log_alpha_chunk_f.stride(),
        *initial_state_f.stride(),
        *chunk_start.stride(),
        *final_state.stride(),
        NC_STATIC=NC,
        BLOCK_NC=block_nc,
        BLOCK_MD=phase2_cfg.block_md,
        USE_FP32_COMPUTE=(compute_dtype == torch.float32),
        HAS_INITIAL_STATE=has_initial_state,
        RETURN_FINAL_STATE=return_final_state,
        num_warps=phase2_cfg.num_warps,
        num_stages=phase2_cfg.num_stages,
    )
    if return_final_state:
        return chunk_start, final_state
    return chunk_start, None


# --------------------------------------------------------------------------------------------------
# END PHASE 2: PREFIX SCAN OVER CHUNKS
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# PHASE 3: DENSE OUTPUT COMPUTATION
# --------------------------------------------------------------------------------------------------
def ssd_rank1_dense_output_reference(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    S0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Phase-3 PyTorch dense CxC reference matching kernel numerics.

    Target equation:

      y_t = sum_{tau<=t} [prod_{u=tau+1..t} alpha_u] * (C_t^T W_tau) * V_tau
            + [prod_{u=0..t} alpha_u] * (C_t^T S0)
      where alpha_u = exp(log_alpha_u).

    Dense construction per `(BH, NC)`:
    1. Build causal carry matrix `L` in FP32.
    2. Build `R = C @ W^T` in FP32.
    3. Form `K = L * R` in FP32.
    4. Contract `Y_diag = K @ V` for intra-chunk contributions.
    5. Add chunk-prefix contribution
         `Y_off[t] = exp(sum_{u=0..t} log_alpha_u) * (C_t^T S0)`.
    """
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "ssd_rank1_dense_output_reference expects C=[BH,NC,C,M], "
            "W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, "
            f"V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    BH, NC, C_CHUNK, M = C.shape
    if W.shape != (BH, NC, C_CHUNK, M):
        raise ValueError(f"W must be [BH,NC,C,M]={BH, NC, C_CHUNK, M}; got {tuple(W.shape)}.")
    if V.shape[:3] != (BH, NC, C_CHUNK):
        raise ValueError(f"V must match [BH,NC,C,*]={BH, NC, C_CHUNK}; got {tuple(V.shape)}.")
    D = V.shape[-1]
    if log_alpha.shape != (BH, NC, C_CHUNK):
        raise ValueError(f"log_alpha must be [BH,NC,C]={BH, NC, C_CHUNK}; got {tuple(log_alpha.shape)}.")
    if C.device != W.device or C.device != V.device or C.device != log_alpha.device:
        raise ValueError("C, W, V, log_alpha must be on the same device.")
    if C.dtype != W.dtype or C.dtype != V.dtype or C.dtype != log_alpha.dtype:
        raise ValueError("C, W, V, log_alpha must share dtype.")
    if C_CHUNK == 0:
        raise NotImplementedError("ssd_rank1_dense_output_reference does not support C==0.")
    _require_supported_md(M, D, where="ssd_rank1_dense_output_reference")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank1_dense_output_reference")
    if S0 is None:
        S0_md = torch.zeros((BH, NC, M, D), device=C.device, dtype=torch.float32)
    else:
        if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
            S0_md = S0.reshape(BH, NC, M, D).to(torch.float32)
        elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
            S0_md = S0.to(torch.float32)
        else:
            raise ValueError(
                "S0 must be [BH,NC,MD] or [BH,NC,M,D] matching C/W/V. "
                f"Got S0={tuple(S0.shape)} with expected BH={BH}, NC={NC}, M={M}, D={D}."
            )
        if S0.device != C.device:
            raise ValueError("S0 must be on the same device as C/W/V/log_alpha.")

    C_f = C.float()
    W_f = W.float()
    V_f = V.float()
    log_alpha_f = log_alpha.float()

    c_idx = torch.arange(C_CHUNK, device=C.device)
    tril = (c_idx.view(C_CHUNK, 1) >= c_idx.view(1, C_CHUNK)).view(1, 1, C_CHUNK, C_CHUNK)

    log_p = torch.cumsum(log_alpha_f, dim=-1)
    # Guard against masked-overflow autograd pathology: upper-tri deltas can be
    # large positive even though they are masked out. Zero masked lanes before
    # exp2 so no inf is created while preserving exact valid-lane math.
    log_delta_l2 = (log_p[..., :, None] - log_p[..., None, :]) * _INV_LN2
    log_delta_l2 = torch.where(
        tril,
        log_delta_l2,
        torch.zeros((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32),
    )
    L = torch.where(
        tril,
        torch.exp2(log_delta_l2),
        torch.zeros((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32),
    )

    p = torch.exp2(log_p * _INV_LN2).to(torch.float32).unsqueeze(-1)
    Y_off = p * torch.matmul(C_f, S0_md)

    K = (L * (C_f @ W_f.mT)).to(torch.float32)
    Y_diag = torch.matmul(K, V_f)

    return (Y_diag + Y_off).to(V.dtype)


def ssd_rank1_dense_output_backward_reference(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    grad_Y: torch.Tensor,
    S0: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Phase-3 backward reference for dense chunk-local output.

    Forward (per `(BH, NC)` chunk):
      R = C @ W^T                        # [C, C]
      L[i,j] = 1_{j<=i} * prod_{u=j+1..i} alpha_u, alpha_u = exp(log_alpha_u)
      K = L * R                          # [C, C]
      Y_diag = K @ V                     # [C, D]
      Y_off  = p * (C @ S0), p_i = prod_{u=0..i} alpha_u
      Y = Y_diag + Y_off

    Dense backward:
      dV = K^T @ dY
      dK = dY @ V^T
      dR = dK * L
      dC_diag = dR @ W
      dW = dR^T @ C

    For `log_alpha` from the dense carry matrix:
      Q = dK * R * L
      dlog_alpha[k] += sum_{i>=k} sum_{j<k} Q[i,j]

    For `log_alpha` from `Y_off`:
      dB = dY * p, B = C @ S0
      dlog_alpha[k] += sum_{i>=k} (dY_i dot B_i) * p_i

    Returns:
    - dC: `[BH, NC, C, M]`
    - dW: `[BH, NC, C, M]`
    - dV: `[BH, NC, C, D]`
    - dlog_alpha: `[BH, NC, C]`
    - dS0: `[BH, NC, M, D]` if `S0` is provided, else `None`
    """
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3 or grad_Y.ndim != 4:
        raise ValueError(
            "ssd_rank1_dense_output_backward_reference expects "
            "C=[BH,NC,C,M], W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C], grad_Y=[BH,NC,C,D]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}, "
            f"log_alpha={tuple(log_alpha.shape)}, grad_Y={tuple(grad_Y.shape)}."
        )
    BH, NC, C_CHUNK, M = C.shape
    if W.shape != (BH, NC, C_CHUNK, M):
        raise ValueError(f"W must be [BH,NC,C,M]={BH, NC, C_CHUNK, M}; got {tuple(W.shape)}.")
    if V.shape[:3] != (BH, NC, C_CHUNK):
        raise ValueError(f"V must match [BH,NC,C,*]={BH, NC, C_CHUNK}; got {tuple(V.shape)}.")
    D = V.shape[-1]
    if log_alpha.shape != (BH, NC, C_CHUNK):
        raise ValueError(f"log_alpha must be [BH,NC,C]={BH, NC, C_CHUNK}; got {tuple(log_alpha.shape)}.")
    if grad_Y.shape != (BH, NC, C_CHUNK, D):
        raise ValueError(f"grad_Y must be [BH,NC,C,D]={BH, NC, C_CHUNK, D}; got {tuple(grad_Y.shape)}.")
    if C.device != W.device or C.device != V.device or C.device != log_alpha.device or C.device != grad_Y.device:
        raise ValueError("C, W, V, log_alpha, grad_Y must be on the same device.")
    if C.dtype != W.dtype or C.dtype != V.dtype or C.dtype != log_alpha.dtype or C.dtype != grad_Y.dtype:
        raise ValueError("C, W, V, log_alpha, grad_Y must share dtype.")
    if C_CHUNK == 0:
        raise NotImplementedError("ssd_rank1_dense_output_backward_reference does not support C==0.")
    _require_supported_md(M, D, where="ssd_rank1_dense_output_backward_reference")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank1_dense_output_backward_reference")

    if S0 is None:
        S0_md = torch.zeros((BH, NC, M, D), device=C.device, dtype=torch.float32)
        return_s0_grad = False
    else:
        if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
            S0_md = S0.reshape(BH, NC, M, D).to(torch.float32)
        elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
            S0_md = S0.to(torch.float32)
        else:
            raise ValueError(
                "S0 must be [BH,NC,MD] or [BH,NC,M,D] matching C/W/V. "
                f"Got S0={tuple(S0.shape)} with expected BH={BH}, NC={NC}, M={M}, D={D}."
            )
        if S0.device != C.device:
            raise ValueError("S0 must be on the same device as C/W/V/log_alpha/grad_Y.")
        return_s0_grad = True

    C_f = C.float()
    W_f = W.float()
    V_f = V.float()
    log_alpha_f = log_alpha.float()
    grad_Y_f = grad_Y.float()

    c_idx = torch.arange(C_CHUNK, device=C.device)
    tril = (c_idx.view(C_CHUNK, 1) >= c_idx.view(1, C_CHUNK)).view(1, 1, C_CHUNK, C_CHUNK)

    log_p = torch.cumsum(log_alpha_f, dim=-1)
    p = torch.exp2(log_p * _INV_LN2)
    log_delta_l2 = (log_p[..., :, None] - log_p[..., None, :]) * _INV_LN2
    log_delta_l2 = torch.where(
        tril,
        log_delta_l2,
        torch.zeros((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32),
    )
    L = torch.where(
        tril,
        torch.exp2(log_delta_l2),
        torch.zeros((BH, NC, C_CHUNK, C_CHUNK), device=C.device, dtype=torch.float32),
    )
    R = C_f @ W_f.mT
    K = L * R

    # Main dense path.
    dV = K.mT @ grad_Y_f
    dK = grad_Y_f @ V_f.mT
    dR = dK * L
    dC_diag = dR @ W_f
    dW = dR.mT @ C_f

    # Off-path via Y_off = p * (C @ S0).
    B = C_f @ S0_md
    dB = grad_Y_f * p.unsqueeze(-1)
    dC_off = dB @ S0_md.mT
    dS0 = C_f.mT @ dB

    dC = dC_diag + dC_off

    # dlog_alpha from dense carry path: Q = dK * R * L and rectangle sums.
    Q = dK * R * L
    left_prefix = torch.cumsum(Q, dim=-1)
    left_of_k = torch.zeros_like(Q)
    left_of_k[..., :, 1:] = left_prefix[..., :, :-1]
    suffix_rows = torch.flip(torch.cumsum(torch.flip(left_of_k, dims=[-2]), dim=-2), dims=[-2])
    dlog_alpha_diag = torch.diagonal(suffix_rows, dim1=-2, dim2=-1)

    # dlog_alpha from off-path p_i factors.
    dlog_alpha_off_src = (grad_Y_f * B).sum(dim=-1) * p
    dlog_alpha_off = torch.flip(torch.cumsum(torch.flip(dlog_alpha_off_src, dims=[-1]), dim=-1), dims=[-1])
    dlog_alpha = dlog_alpha_diag + dlog_alpha_off

    dC_out = dC.to(C.dtype)
    dW_out = dW.to(W.dtype)
    dV_out = dV.to(V.dtype)
    dlog_alpha_out = dlog_alpha.to(log_alpha.dtype)
    dS0_out = dS0.to(C.dtype) if return_s0_grad else None
    return dC_out, dW_out, dV_out, dlog_alpha_out, dS0_out


@triton.jit
def ssd_rank1_dense_output_fwd_kernel(
    C_ptr,
    W_ptr,
    V_ptr,
    log_alpha_ptr,
    S0_ptr,
    out_y_ptr,
    out_y_off_ptr,
    b_size: tl.constexpr,
    h_size: tl.constexpr,
    bh_size: tl.constexpr,
    nc_size: tl.constexpr,
    c_size: tl.constexpr,
    m_size: tl.constexpr,
    d_size: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_nc: tl.constexpr,
    stride_c_c: tl.constexpr,
    stride_c_h: tl.constexpr,
    stride_c_m: tl.constexpr,
    stride_w_b: tl.constexpr,
    stride_w_nc: tl.constexpr,
    stride_w_c: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_nc: tl.constexpr,
    stride_v_c: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_r_b: tl.constexpr,
    stride_r_nc: tl.constexpr,
    stride_r_c: tl.constexpr,
    stride_r_h: tl.constexpr,
    stride_s0_bh: tl.constexpr,
    stride_s0_nc: tl.constexpr,
    stride_s0_m: tl.constexpr,
    stride_s0_d: tl.constexpr,
    stride_out_y_bh: tl.constexpr,
    stride_out_y_nc: tl.constexpr,
    stride_out_y_c: tl.constexpr,
    stride_out_y_d: tl.constexpr,
    stride_out_y_off_bh: tl.constexpr,
    stride_out_y_off_nc: tl.constexpr,
    stride_out_y_off_c: tl.constexpr,
    stride_out_y_off_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    M_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    WRITE_Y_OFF: tl.constexpr,
):
    """Phase-3 Triton forward following the dense CxC reference structure.

    For each `(BH, NC)` chunk, build:
    1. `L[t, tau] = 1_{tau<=t} * exp(sum_{u=tau+1..t} log_alpha_u)`.
    2. `R = C @ W^T`.
    3. `K = L * R`.
    4. `Y_diag = K @ V`.
    5. `Y_off[t] = exp(sum_{u=0..t} log_alpha_u) * (C_t^T S0)`.
    6. `Y = Y_diag + Y_off`.

    Program id mapping:
    - `pid0`: flattened `(BH, NC)`
    - `pid1`: output `D` tile
    """
    pid_bhnc = tl.program_id(0)
    pid_d_tile = tl.program_id(1)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_c_b, stride_c_nc, stride_c_c, stride_c_h, stride_c_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    w_desc = tl.make_tensor_descriptor(
        W_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    s0_desc = tl.make_tensor_descriptor(
        S0_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_s0_bh, stride_s0_nc, stride_s0_m, stride_s0_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    out_desc = tl.make_tensor_descriptor(
        out_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_out_y_bh, stride_out_y_nc, stride_out_y_c, stride_out_y_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )

    offs_c = tl.arange(0, BLOCK_C)

    # Step 1: build dense causal factor matrix L[t, tau] in log space.
    r_base = B_IDX * stride_r_b + NC_IDX * stride_r_nc + H_IDX * stride_r_h
    log_alpha_vals = tl.load(log_alpha_ptr + r_base + offs_c * stride_r_c).to(tl.float32)
    log_p_incl = tl.cumsum(log_alpha_vals, axis=0)
    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    valid = col_idx <= row_idx
    INV_LN2 = 1.4426950408889634
    log_delta_l2 = (log_p_incl[:, None] - log_p_incl[None, :]) * INV_LN2
    # Keep masked upper-tri lanes finite even if the compiler evaluates both arms.
    log_delta_l2 = tl.where(valid, log_delta_l2, 0.0)
    L = tl.where(valid, tl.exp2(log_delta_l2), 0.0)

    # Step 2: build dense score matrix R = C @ W^T.
    R = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        C = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        W = tl.reshape(w_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        C_tc = C.to(tl.bfloat16)
        W_tc = W.to(tl.bfloat16)
        R = R + tl.dot(
            C_tc,
            tl.trans(W_tc),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
    # Step 3: form K = L * R.
    K = L * R

    # Step 4: compute Y_diag = K @ V in FP32.
    d_start = pid_d_tile * BLOCK_D
    V_in = tl.reshape(v_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
    K_tc = K.to(tl.bfloat16)
    V_tc = V_in.to(tl.bfloat16)
    Y_diag = tl.dot(
        K_tc,
        V_tc,
        out_dtype=tl.float32,
        input_precision=INPUT_PRECISION,
    )
    # Step 5: compute Y_off = prefix_incl * (C @ S0).
    Y_off_base = tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        C_blk = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        S0_blk = tl.reshape(s0_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D))
        C_blk_tc = C_blk.to(tl.bfloat16)
        S0_blk_tc = S0_blk.to(tl.bfloat16)
        Y_off_base = Y_off_base + tl.dot(
            C_blk_tc,
            S0_blk_tc,
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
    p = tl.exp2(log_p_incl * INV_LN2)
    Y_off = p[:, None] * Y_off_base
    Y = Y_diag + Y_off

    out_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(Y.to(V_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))
    if WRITE_Y_OFF:
        out_y_off_desc = tl.make_tensor_descriptor(
            out_y_off_ptr,
            shape=[bh_size, nc_size, c_size, d_size],
            strides=[stride_out_y_off_bh, stride_out_y_off_nc, stride_out_y_off_c, stride_out_y_off_d],
            block_shape=[1, 1, BLOCK_C, BLOCK_D],
        )
        out_y_off_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(Y_off.to(V_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))


@triton.jit
def ssd_rank1_dense_output_bwd_fused_kernel(
    C_ptr,
    W_ptr,
    V_ptr,
    log_alpha_ptr,
    grad_y_ptr,
    S0_ptr,
    out_dV_ptr,
    out_dC_ptr,
    out_dW_ptr,
    out_dlog_ptr,
    out_dS0_ptr,
    b_size: tl.constexpr,
    h_size: tl.constexpr,
    bh_size: tl.constexpr,
    nc_size: tl.constexpr,
    c_size: tl.constexpr,
    m_size: tl.constexpr,
    d_size: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_nc: tl.constexpr,
    stride_c_c: tl.constexpr,
    stride_c_h: tl.constexpr,
    stride_c_m: tl.constexpr,
    stride_w_b: tl.constexpr,
    stride_w_nc: tl.constexpr,
    stride_w_c: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_nc: tl.constexpr,
    stride_v_c: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_r_b: tl.constexpr,
    stride_r_nc: tl.constexpr,
    stride_r_c: tl.constexpr,
    stride_r_h: tl.constexpr,
    stride_gy_bh: tl.constexpr,
    stride_gy_nc: tl.constexpr,
    stride_gy_c: tl.constexpr,
    stride_gy_d: tl.constexpr,
    stride_s0_bh: tl.constexpr,
    stride_s0_nc: tl.constexpr,
    stride_s0_m: tl.constexpr,
    stride_s0_d: tl.constexpr,
    stride_dv_bh: tl.constexpr,
    stride_dv_nc: tl.constexpr,
    stride_dv_c: tl.constexpr,
    stride_dv_d: tl.constexpr,
    stride_dc_bh: tl.constexpr,
    stride_dc_nc: tl.constexpr,
    stride_dc_c: tl.constexpr,
    stride_dc_m: tl.constexpr,
    stride_dw_bh: tl.constexpr,
    stride_dw_nc: tl.constexpr,
    stride_dw_c: tl.constexpr,
    stride_dw_m: tl.constexpr,
    stride_dlog_bh: tl.constexpr,
    stride_dlog_nc: tl.constexpr,
    stride_dlog_c: tl.constexpr,
    stride_ds0_bh: tl.constexpr,
    stride_ds0_nc: tl.constexpr,
    stride_ds0_m: tl.constexpr,
    stride_ds0_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    M_STATIC: tl.constexpr,
    D_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    HAS_S0: tl.constexpr,
):
    """Fused Phase-3 backward: per (BH,NC) compute dV/dC/dW/dlog_alpha, and optional dS0.

    Unified schedule:
      - Dense path: dV, dC_diag, dW, dlog_diag.
      - Optional off path (HAS_S0): dC_off accumulation, dS0, dlog_off accumulation.

    For one chunk:
      1. Build L from log_alpha cumsum
      2. R = C @ W^T (accumulated over M-tiles)
      3. K = L * R
      4. dV = K^T @ dY  (per D-tile, stored immediately)
      5. dK = dY @ V^T  (accumulated over D-tiles in registers)
      6. dR = dK * L
      7. dC_diag = dR @ W  (per M-tile)
      8. dW = dR^T @ C     (per M-tile)
      9. Q = dR * R, then dlog_diag in-register (no global Q/R write)

    If HAS_S0:
      10. p = exp(cumsum(log_alpha))
      11. dC_off += (p * dY) @ S0^T
      12. dS0 = C^T @ (p * dY)
      13. dlog_off from reverse cumsum of src_i = <dY_i, p_i * (C_i @ S0)>
    """
    pid_bhnc = tl.program_id(0)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_c_b, stride_c_nc, stride_c_c, stride_c_h, stride_c_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    w_desc = tl.make_tensor_descriptor(
        W_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v_desc = tl.make_tensor_descriptor(
        V_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    gy_desc = tl.make_tensor_descriptor(
        grad_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_gy_bh, stride_gy_nc, stride_gy_c, stride_gy_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    s0_desc = tl.make_tensor_descriptor(
        S0_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_s0_bh, stride_s0_nc, stride_s0_m, stride_s0_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    dv_desc = tl.make_tensor_descriptor(
        out_dV_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_dv_bh, stride_dv_nc, stride_dv_c, stride_dv_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    dc_desc = tl.make_tensor_descriptor(
        out_dC_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_dc_bh, stride_dc_nc, stride_dc_c, stride_dc_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    dw_desc = tl.make_tensor_descriptor(
        out_dW_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_dw_bh, stride_dw_nc, stride_dw_c, stride_dw_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    dlog_desc = tl.make_tensor_descriptor(
        out_dlog_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_dlog_bh, stride_dlog_nc, stride_dlog_c],
        block_shape=[1, 1, BLOCK_C],
    )
    ds0_desc = tl.make_tensor_descriptor(
        out_dS0_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_ds0_bh, stride_ds0_nc, stride_ds0_m, stride_ds0_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )

    # ---- Step 1: Build L ----
    offs_c = tl.arange(0, BLOCK_C)
    r_base = B_IDX * stride_r_b + NC_IDX * stride_r_nc + H_IDX * stride_r_h
    log_alpha_vals = tl.load(log_alpha_ptr + r_base + offs_c * stride_r_c).to(tl.float32)
    log_p_incl = tl.cumsum(log_alpha_vals, axis=0)
    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    valid = col_idx <= row_idx
    INV_LN2 = 1.4426950408889634
    log_delta_l2 = (log_p_incl[:, None] - log_p_incl[None, :]) * INV_LN2
    log_delta_l2 = tl.where(valid, log_delta_l2, 0.0)
    L = tl.where(valid, tl.exp2(log_delta_l2), 0.0)

    # ---- Step 2: R = C @ W^T ----
    R = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        C_blk = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        W_blk = tl.reshape(w_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        C_blk_tc = C_blk.to(tl.bfloat16)
        W_blk_tc = W_blk.to(tl.bfloat16)
        R = R + tl.dot(
            C_blk_tc,
            tl.trans(W_blk_tc),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
    # ---- Step 3: K = L * R ----
    K = L * R
    K_tc = K.to(tl.bfloat16)

    # ---- Step 4 & 5: dV per D-tile, accumulate dK in registers ----
    dK = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
        d_start = d_blk * BLOCK_D
        G_in = tl.reshape(gy_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D))
        V_in = tl.reshape(v_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
        G_tc = G_in.to(tl.bfloat16)
        V_tc = V_in.to(tl.bfloat16)
        dV_tile = tl.dot(
            tl.trans(K_tc),
            G_tc,
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
        dv_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV_tile.to(V_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))

        dK = dK + tl.dot(
            G_tc,
            tl.trans(V_tc),
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
    # ---- Step 6: dR = dK * L ----
    dR = dK * L
    dR_tc = dR.to(tl.bfloat16)

    # ---- Step 7 & 8: dC_diag and dW per M-tile ----
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        W_in = tl.reshape(w_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        C_in = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        W_tile_tc = W_in.to(tl.bfloat16)
        C_tile_tc = C_in.to(tl.bfloat16)
        dC_diag_tile = tl.dot(
            dR_tc,
            W_tile_tc,
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
        dW_tile = tl.dot(
            tl.trans(dR_tc),
            C_tile_tc,
            out_dtype=tl.float32,
            input_precision=INPUT_PRECISION,
        )
        dc_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dC_diag_tile.to(C_in.dtype), (1, 1, BLOCK_C, BLOCK_M)))
        dw_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW_tile.to(C_in.dtype), (1, 1, BLOCK_C, BLOCK_M)))

    # ---- Step 9: compute dlog_diag from Q = dR * R (in-register, no global write) ----
    Q = dR * R
    left_prefix = tl.cumsum(Q, axis=1)
    left_of = left_prefix - Q
    suffix_rows = tl.cumsum(left_of, axis=0, reverse=True)
    is_diag = offs_c[:, None] == offs_c[None, :]
    dlog = tl.sum(tl.where(is_diag, suffix_rows, 0.0), axis=1)
    dlog_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(dlog, (1, 1, BLOCK_C)))

    if HAS_S0:
        # ---- Off path: dC_off, dS0, dlog_off ----
        log_p = tl.cumsum(log_alpha_vals, axis=0)
        p = tl.exp2(log_p * INV_LN2)
        src = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
            m_start = m_blk * BLOCK_M
            dC_off = tl.zeros((BLOCK_C, BLOCK_M), dtype=tl.float32)
            C_tile_in = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            C_tile_tc = C_tile_in.to(tl.bfloat16)
            for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
                d_start = d_blk * BLOCK_D
                G_tile = tl.reshape(gy_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
                dB_tile = p[:, None] * G_tile
                S0_tile = tl.reshape(s0_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D))
                dB_tile_tc = dB_tile.to(tl.bfloat16)
                S0_tile_tc = S0_tile.to(tl.bfloat16)
                y_off_part = p[:, None] * tl.dot(
                    C_tile_tc,
                    S0_tile_tc,
                    out_dtype=tl.float32,
                    input_precision=INPUT_PRECISION,
                )
                src += tl.sum(G_tile * y_off_part, axis=1)
                dC_off = dC_off + tl.dot(
                    dB_tile_tc,
                    tl.trans(S0_tile_tc),
                    out_dtype=tl.float32,
                    input_precision=INPUT_PRECISION,
                )
                dS0_tile = tl.dot(
                    tl.trans(C_tile_tc),
                    dB_tile_tc,
                    out_dtype=tl.float32,
                    input_precision=INPUT_PRECISION,
                )
                ds0_desc.store([BH_IDX, NC_IDX, m_start, d_start], tl.reshape(dS0_tile, (1, 1, BLOCK_M, BLOCK_D)))

            c_probe = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            dc_prev = tl.reshape(dc_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
            dc_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape((dc_prev + dC_off).to(c_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))

        dlog_off = tl.cumsum(src, axis=0, reverse=True)
        out_prev = tl.reshape(dlog_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
        dlog_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(out_prev + dlog_off, (1, 1, BLOCK_C)))


@triton.jit
def ssd_rank4_dense_output_fwd_kernel(
    C_ptr,
    W1_ptr,
    V1_ptr,
    W2_ptr,
    V2_ptr,
    W3_ptr,
    V3_ptr,
    W4_ptr,
    V4_ptr,
    log_alpha_ptr,
    S0_ptr,
    out_y_ptr,
    out_y_off_ptr,
    b_size: tl.constexpr,
    h_size: tl.constexpr,
    bh_size: tl.constexpr,
    nc_size: tl.constexpr,
    c_size: tl.constexpr,
    m_size: tl.constexpr,
    d_size: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_nc: tl.constexpr,
    stride_c_c: tl.constexpr,
    stride_c_h: tl.constexpr,
    stride_c_m: tl.constexpr,
    stride_w_b: tl.constexpr,
    stride_w_nc: tl.constexpr,
    stride_w_c: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_nc: tl.constexpr,
    stride_v_c: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_r_b: tl.constexpr,
    stride_r_nc: tl.constexpr,
    stride_r_c: tl.constexpr,
    stride_r_h: tl.constexpr,
    stride_s0_bh: tl.constexpr,
    stride_s0_nc: tl.constexpr,
    stride_s0_m: tl.constexpr,
    stride_s0_d: tl.constexpr,
    stride_out_y_bh: tl.constexpr,
    stride_out_y_nc: tl.constexpr,
    stride_out_y_c: tl.constexpr,
    stride_out_y_d: tl.constexpr,
    stride_out_y_off_bh: tl.constexpr,
    stride_out_y_off_nc: tl.constexpr,
    stride_out_y_off_c: tl.constexpr,
    stride_out_y_off_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    M_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    WRITE_Y_OFF: tl.constexpr,
    RANK: tl.constexpr,
):
    """Phase-3 rank-4 forward: fused rank accumulation with shared L."""
    pid_bhnc = tl.program_id(0)
    pid_d_tile = tl.program_id(1)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_c_b, stride_c_nc, stride_c_c, stride_c_h, stride_c_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    w1_desc = tl.make_tensor_descriptor(
        W1_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v1_desc = tl.make_tensor_descriptor(
        V1_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    w2_desc = tl.make_tensor_descriptor(
        W2_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v2_desc = tl.make_tensor_descriptor(
        V2_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    w3_desc = tl.make_tensor_descriptor(
        W3_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v3_desc = tl.make_tensor_descriptor(
        V3_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    w4_desc = tl.make_tensor_descriptor(
        W4_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v4_desc = tl.make_tensor_descriptor(
        V4_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    s0_desc = tl.make_tensor_descriptor(
        S0_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_s0_bh, stride_s0_nc, stride_s0_m, stride_s0_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    out_desc = tl.make_tensor_descriptor(
        out_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_out_y_bh, stride_out_y_nc, stride_out_y_c, stride_out_y_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )

    offs_c = tl.arange(0, BLOCK_C)
    r_base = B_IDX * stride_r_b + NC_IDX * stride_r_nc + H_IDX * stride_r_h
    log_alpha_vals = tl.load(log_alpha_ptr + r_base + offs_c * stride_r_c).to(tl.float32)
    log_p_incl = tl.cumsum(log_alpha_vals, axis=0)
    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    valid = col_idx <= row_idx
    INV_LN2 = 1.4426950408889634
    log_delta_l2 = (log_p_incl[:, None] - log_p_incl[None, :]) * INV_LN2
    log_delta_l2 = tl.where(valid, log_delta_l2, 0.0)
    L = tl.where(valid, tl.exp2(log_delta_l2), 0.0)

    R1 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    if RANK >= 2:
        R2 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    if RANK >= 3:
        R3 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    if RANK >= 4:
        R4 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        C_blk = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        C_tc = C_blk.to(tl.bfloat16)
        W1_blk = tl.reshape(w1_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        R1 = R1 + tl.dot(C_tc, tl.trans(W1_blk.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        if RANK >= 2:
            W2_blk = tl.reshape(w2_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            R2 = R2 + tl.dot(C_tc, tl.trans(W2_blk.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        if RANK >= 3:
            W3_blk = tl.reshape(w3_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            R3 = R3 + tl.dot(C_tc, tl.trans(W3_blk.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        if RANK >= 4:
            W4_blk = tl.reshape(w4_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            R4 = R4 + tl.dot(C_tc, tl.trans(W4_blk.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    K1 = L * R1
    if RANK >= 2:
        K2 = L * R2
    if RANK >= 3:
        K3 = L * R3
    if RANK >= 4:
        K4 = L * R4
    d_start = pid_d_tile * BLOCK_D
    V1_in = tl.reshape(v1_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
    Y_diag = tl.dot(K1.to(tl.bfloat16), V1_in.to(tl.bfloat16), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    if RANK >= 2:
        V2_in = tl.reshape(v2_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
        Y_diag = Y_diag + tl.dot(K2.to(tl.bfloat16), V2_in.to(tl.bfloat16), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    if RANK >= 3:
        V3_in = tl.reshape(v3_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
        Y_diag = Y_diag + tl.dot(K3.to(tl.bfloat16), V3_in.to(tl.bfloat16), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    if RANK >= 4:
        V4_in = tl.reshape(v4_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
        Y_diag = Y_diag + tl.dot(K4.to(tl.bfloat16), V4_in.to(tl.bfloat16), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    Y_off_base = tl.zeros((BLOCK_C, BLOCK_D), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        C_blk = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        S0_blk = tl.reshape(s0_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D))
        Y_off_base = Y_off_base + tl.dot(C_blk.to(tl.bfloat16), S0_blk.to(tl.bfloat16), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    p = tl.exp2(log_p_incl * INV_LN2)
    Y_off = p[:, None] * Y_off_base
    Y = Y_diag + Y_off

    out_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(Y.to(V1_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))
    if WRITE_Y_OFF:
        out_y_off_desc = tl.make_tensor_descriptor(
            out_y_off_ptr,
            shape=[bh_size, nc_size, c_size, d_size],
            strides=[stride_out_y_off_bh, stride_out_y_off_nc, stride_out_y_off_c, stride_out_y_off_d],
            block_shape=[1, 1, BLOCK_C, BLOCK_D],
        )
        out_y_off_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(Y_off.to(V1_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))


@triton.jit
def ssd_rank4_dense_output_bwd_fused_kernel(
    C_ptr,
    W1_ptr,
    V1_ptr,
    W2_ptr,
    V2_ptr,
    W3_ptr,
    V3_ptr,
    W4_ptr,
    V4_ptr,
    log_alpha_ptr,
    grad_y_ptr,
    S0_ptr,
    out_dV1_ptr,
    out_dC_ptr,
    out_dW1_ptr,
    out_dV2_ptr,
    out_dW2_ptr,
    out_dV3_ptr,
    out_dW3_ptr,
    out_dV4_ptr,
    out_dW4_ptr,
    out_dlog_ptr,
    out_dS0_ptr,
    b_size: tl.constexpr,
    h_size: tl.constexpr,
    bh_size: tl.constexpr,
    nc_size: tl.constexpr,
    c_size: tl.constexpr,
    m_size: tl.constexpr,
    d_size: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_nc: tl.constexpr,
    stride_c_c: tl.constexpr,
    stride_c_h: tl.constexpr,
    stride_c_m: tl.constexpr,
    stride_w_b: tl.constexpr,
    stride_w_nc: tl.constexpr,
    stride_w_c: tl.constexpr,
    stride_w_h: tl.constexpr,
    stride_w_m: tl.constexpr,
    stride_v_b: tl.constexpr,
    stride_v_nc: tl.constexpr,
    stride_v_c: tl.constexpr,
    stride_v_h: tl.constexpr,
    stride_v_d: tl.constexpr,
    stride_r_b: tl.constexpr,
    stride_r_nc: tl.constexpr,
    stride_r_c: tl.constexpr,
    stride_r_h: tl.constexpr,
    stride_gy_bh: tl.constexpr,
    stride_gy_nc: tl.constexpr,
    stride_gy_c: tl.constexpr,
    stride_gy_d: tl.constexpr,
    stride_s0_bh: tl.constexpr,
    stride_s0_nc: tl.constexpr,
    stride_s0_m: tl.constexpr,
    stride_s0_d: tl.constexpr,
    stride_dv_bh: tl.constexpr,
    stride_dv_nc: tl.constexpr,
    stride_dv_c: tl.constexpr,
    stride_dv_d: tl.constexpr,
    stride_dc_bh: tl.constexpr,
    stride_dc_nc: tl.constexpr,
    stride_dc_c: tl.constexpr,
    stride_dc_m: tl.constexpr,
    stride_dw_bh: tl.constexpr,
    stride_dw_nc: tl.constexpr,
    stride_dw_c: tl.constexpr,
    stride_dw_m: tl.constexpr,
    stride_dlog_bh: tl.constexpr,
    stride_dlog_nc: tl.constexpr,
    stride_dlog_c: tl.constexpr,
    stride_ds0_bh: tl.constexpr,
    stride_ds0_nc: tl.constexpr,
    stride_ds0_m: tl.constexpr,
    stride_ds0_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    C_STATIC: tl.constexpr,
    M_STATIC: tl.constexpr,
    D_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    HAS_S0: tl.constexpr,
    RANK: tl.constexpr,
    COMPUTE_DV: tl.constexpr,
    COMPUTE_MAIN: tl.constexpr,
):
    """Phase-3 rank-4 backward kernel with compile-time split modes.

    - COMPUTE_DV=True: compute only dV* terms.
    - COMPUTE_MAIN=True: compute dK -> dC/dW and dlog terms.
    """
    pid_bhnc = tl.program_id(0)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_c_b, stride_c_nc, stride_c_c, stride_c_h, stride_c_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    w1_desc = tl.make_tensor_descriptor(
        W1_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v1_desc = tl.make_tensor_descriptor(
        V1_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    w2_desc = tl.make_tensor_descriptor(
        W2_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v2_desc = tl.make_tensor_descriptor(
        V2_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    w3_desc = tl.make_tensor_descriptor(
        W3_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v3_desc = tl.make_tensor_descriptor(
        V3_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    w4_desc = tl.make_tensor_descriptor(
        W4_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_w_b, stride_w_nc, stride_w_c, stride_w_h, stride_w_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    v4_desc = tl.make_tensor_descriptor(
        V4_ptr,
        shape=[b_size, nc_size, c_size, h_size, d_size],
        strides=[stride_v_b, stride_v_nc, stride_v_c, stride_v_h, stride_v_d],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_D],
    )
    gy_desc = tl.make_tensor_descriptor(
        grad_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_gy_bh, stride_gy_nc, stride_gy_c, stride_gy_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    if COMPUTE_DV:
        dv1_desc = tl.make_tensor_descriptor(
            out_dV1_ptr,
            shape=[bh_size, nc_size, c_size, d_size],
            strides=[stride_dv_bh, stride_dv_nc, stride_dv_c, stride_dv_d],
            block_shape=[1, 1, BLOCK_C, BLOCK_D],
        )
        if RANK >= 2:
            dv2_desc = tl.make_tensor_descriptor(
                out_dV2_ptr,
                shape=[bh_size, nc_size, c_size, d_size],
                strides=[stride_dv_bh, stride_dv_nc, stride_dv_c, stride_dv_d],
                block_shape=[1, 1, BLOCK_C, BLOCK_D],
            )
        if RANK >= 3:
            dv3_desc = tl.make_tensor_descriptor(
                out_dV3_ptr,
                shape=[bh_size, nc_size, c_size, d_size],
                strides=[stride_dv_bh, stride_dv_nc, stride_dv_c, stride_dv_d],
                block_shape=[1, 1, BLOCK_C, BLOCK_D],
            )
        if RANK >= 4:
            dv4_desc = tl.make_tensor_descriptor(
                out_dV4_ptr,
                shape=[bh_size, nc_size, c_size, d_size],
                strides=[stride_dv_bh, stride_dv_nc, stride_dv_c, stride_dv_d],
                block_shape=[1, 1, BLOCK_C, BLOCK_D],
            )
    if COMPUTE_MAIN:
        dc_desc = tl.make_tensor_descriptor(
            out_dC_ptr,
            shape=[bh_size, nc_size, c_size, m_size],
            strides=[stride_dc_bh, stride_dc_nc, stride_dc_c, stride_dc_m],
            block_shape=[1, 1, BLOCK_C, BLOCK_M],
        )
        dw1_desc = tl.make_tensor_descriptor(
            out_dW1_ptr,
            shape=[bh_size, nc_size, c_size, m_size],
            strides=[stride_dw_bh, stride_dw_nc, stride_dw_c, stride_dw_m],
            block_shape=[1, 1, BLOCK_C, BLOCK_M],
        )
        if RANK >= 2:
            dw2_desc = tl.make_tensor_descriptor(
                out_dW2_ptr,
                shape=[bh_size, nc_size, c_size, m_size],
                strides=[stride_dw_bh, stride_dw_nc, stride_dw_c, stride_dw_m],
                block_shape=[1, 1, BLOCK_C, BLOCK_M],
            )
        if RANK >= 3:
            dw3_desc = tl.make_tensor_descriptor(
                out_dW3_ptr,
                shape=[bh_size, nc_size, c_size, m_size],
                strides=[stride_dw_bh, stride_dw_nc, stride_dw_c, stride_dw_m],
                block_shape=[1, 1, BLOCK_C, BLOCK_M],
            )
        if RANK >= 4:
            dw4_desc = tl.make_tensor_descriptor(
                out_dW4_ptr,
                shape=[bh_size, nc_size, c_size, m_size],
                strides=[stride_dw_bh, stride_dw_nc, stride_dw_c, stride_dw_m],
                block_shape=[1, 1, BLOCK_C, BLOCK_M],
            )
        dlog_desc = tl.make_tensor_descriptor(
            out_dlog_ptr,
            shape=[bh_size, nc_size, c_size],
            strides=[stride_dlog_bh, stride_dlog_nc, stride_dlog_c],
            block_shape=[1, 1, BLOCK_C],
        )
    if COMPUTE_MAIN and HAS_S0:
        s0_desc = tl.make_tensor_descriptor(
            S0_ptr,
            shape=[bh_size, nc_size, m_size, d_size],
            strides=[stride_s0_bh, stride_s0_nc, stride_s0_m, stride_s0_d],
            block_shape=[1, 1, BLOCK_M, BLOCK_D],
        )
        ds0_desc = tl.make_tensor_descriptor(
            out_dS0_ptr,
            shape=[bh_size, nc_size, m_size, d_size],
            strides=[stride_ds0_bh, stride_ds0_nc, stride_ds0_m, stride_ds0_d],
            block_shape=[1, 1, BLOCK_M, BLOCK_D],
        )

    offs_c = tl.arange(0, BLOCK_C)
    r_base = B_IDX * stride_r_b + NC_IDX * stride_r_nc + H_IDX * stride_r_h
    log_alpha_vals = tl.load(log_alpha_ptr + r_base + offs_c * stride_r_c).to(tl.float32)
    log_p_incl = tl.cumsum(log_alpha_vals, axis=0)
    row_idx = offs_c[:, None]
    col_idx = offs_c[None, :]
    valid = col_idx <= row_idx
    INV_LN2 = 1.4426950408889634
    log_delta_l2 = (log_p_incl[:, None] - log_p_incl[None, :]) * INV_LN2
    log_delta_l2 = tl.where(valid, log_delta_l2, 0.0)
    L = tl.where(valid, tl.exp2(log_delta_l2), 0.0)

    R1 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    if RANK >= 2:
        R2 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    if RANK >= 3:
        R3 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    if RANK >= 4:
        R4 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        C_blk = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        C_tc = C_blk.to(tl.bfloat16)
        W1_blk = tl.reshape(w1_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        R1 = R1 + tl.dot(C_tc, tl.trans(W1_blk.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        if RANK >= 2:
            W2_blk = tl.reshape(w2_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            R2 = R2 + tl.dot(C_tc, tl.trans(W2_blk.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        if RANK >= 3:
            W3_blk = tl.reshape(w3_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            R3 = R3 + tl.dot(C_tc, tl.trans(W3_blk.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
        if RANK >= 4:
            W4_blk = tl.reshape(w4_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            R4 = R4 + tl.dot(C_tc, tl.trans(W4_blk.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
    if COMPUTE_DV:
        K1_tc = (L * R1).to(tl.bfloat16)
        if RANK >= 2:
            K2_tc = (L * R2).to(tl.bfloat16)
        if RANK >= 3:
            K3_tc = (L * R3).to(tl.bfloat16)
        if RANK >= 4:
            K4_tc = (L * R4).to(tl.bfloat16)
    if COMPUTE_MAIN:
        dK1 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
        if RANK >= 2:
            dK2 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
        if RANK >= 3:
            dK3 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
        if RANK >= 4:
            dK4 = tl.zeros((BLOCK_C, BLOCK_C), dtype=tl.float32)
    for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
        d_start = d_blk * BLOCK_D
        G_in = tl.reshape(gy_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D))
        G_tc = G_in.to(tl.bfloat16)
        if COMPUTE_DV:
            V1_in = tl.reshape(v1_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
            dV1_tile = tl.dot(tl.trans(K1_tc), G_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dv1_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV1_tile.to(V1_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))
            if RANK >= 2:
                V2_in = tl.reshape(v2_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
                dV2_tile = tl.dot(tl.trans(K2_tc), G_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                dv2_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV2_tile.to(V2_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))
            if RANK >= 3:
                V3_in = tl.reshape(v3_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
                dV3_tile = tl.dot(tl.trans(K3_tc), G_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                dv3_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV3_tile.to(V3_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))
            if RANK >= 4:
                V4_in = tl.reshape(v4_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
                dV4_tile = tl.dot(tl.trans(K4_tc), G_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                dv4_desc.store([BH_IDX, NC_IDX, 0, d_start], tl.reshape(dV4_tile.to(V4_in.dtype), (1, 1, BLOCK_C, BLOCK_D)))

        if COMPUTE_MAIN:
            V1_in = tl.reshape(v1_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
            dK1 = dK1 + tl.dot(G_tc, tl.trans(V1_in.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            if RANK >= 2:
                V2_in = tl.reshape(v2_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
                dK2 = dK2 + tl.dot(G_tc, tl.trans(V2_in.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            if RANK >= 3:
                V3_in = tl.reshape(v3_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
                dK3 = dK3 + tl.dot(G_tc, tl.trans(V3_in.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            if RANK >= 4:
                V4_in = tl.reshape(v4_desc.load([B_IDX, NC_IDX, 0, H_IDX, d_start]), (BLOCK_C, BLOCK_D))
                dK4 = dK4 + tl.dot(G_tc, tl.trans(V4_in.to(tl.bfloat16)), out_dtype=tl.float32, input_precision=INPUT_PRECISION)

    if COMPUTE_MAIN:
        for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
            m_start = m_blk * BLOCK_M
            C_in = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            C_tc = C_in.to(tl.bfloat16)
            dR1_tc = (dK1 * L).to(tl.bfloat16)
            W1_in = tl.reshape(w1_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            W1_tc = W1_in.to(tl.bfloat16)
            dC_tile = tl.dot(dR1_tc, W1_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dW1_tile = tl.dot(tl.trans(dR1_tc), C_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            if RANK >= 2:
                dR2_tc = (dK2 * L).to(tl.bfloat16)
                W2_in = tl.reshape(w2_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
                W2_tc = W2_in.to(tl.bfloat16)
                dC_tile = dC_tile + tl.dot(dR2_tc, W2_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                dW2_tile = tl.dot(tl.trans(dR2_tc), C_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            if RANK >= 3:
                dR3_tc = (dK3 * L).to(tl.bfloat16)
                W3_in = tl.reshape(w3_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
                W3_tc = W3_in.to(tl.bfloat16)
                dC_tile = dC_tile + tl.dot(dR3_tc, W3_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                dW3_tile = tl.dot(tl.trans(dR3_tc), C_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            if RANK >= 4:
                dR4_tc = (dK4 * L).to(tl.bfloat16)
                W4_in = tl.reshape(w4_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
                W4_tc = W4_in.to(tl.bfloat16)
                dC_tile = dC_tile + tl.dot(dR4_tc, W4_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                dW4_tile = tl.dot(tl.trans(dR4_tc), C_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dc_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dC_tile.to(C_in.dtype), (1, 1, BLOCK_C, BLOCK_M)))
            dw1_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW1_tile.to(C_in.dtype), (1, 1, BLOCK_C, BLOCK_M)))
            if RANK >= 2:
                dw2_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW2_tile.to(C_in.dtype), (1, 1, BLOCK_C, BLOCK_M)))
            if RANK >= 3:
                dw3_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW3_tile.to(C_in.dtype), (1, 1, BLOCK_C, BLOCK_M)))
            if RANK >= 4:
                dw4_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape(dW4_tile.to(C_in.dtype), (1, 1, BLOCK_C, BLOCK_M)))

        Q = (dK1 * L) * R1
        if RANK >= 2:
            Q += (dK2 * L) * R2
        if RANK >= 3:
            Q += (dK3 * L) * R3
        if RANK >= 4:
            Q += (dK4 * L) * R4
        left_prefix = tl.cumsum(Q, axis=1)
        left_of = left_prefix - Q
        suffix_rows = tl.cumsum(left_of, axis=0, reverse=True)
        is_diag = offs_c[:, None] == offs_c[None, :]
        dlog = tl.sum(tl.where(is_diag, suffix_rows, 0.0), axis=1)
        dlog_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(dlog, (1, 1, BLOCK_C)))

    if HAS_S0:
        log_p = tl.cumsum(log_alpha_vals, axis=0)
        p = tl.exp2(log_p * INV_LN2)
        src = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
            m_start = m_blk * BLOCK_M
            dC_off = tl.zeros((BLOCK_C, BLOCK_M), dtype=tl.float32)
            C_tile_in = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            C_tile_tc = C_tile_in.to(tl.bfloat16)
            for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
                d_start = d_blk * BLOCK_D
                G_tile = tl.reshape(gy_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
                dB_tile = p[:, None] * G_tile
                S0_tile = tl.reshape(s0_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D))
                dB_tc = dB_tile.to(tl.bfloat16)
                S0_tc = S0_tile.to(tl.bfloat16)
                y_off_part = p[:, None] * tl.dot(C_tile_tc, S0_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                dC_off = dC_off + tl.dot(dB_tc, tl.trans(S0_tc), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                dS0_tile = tl.dot(tl.trans(C_tile_tc), dB_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
                src += tl.sum(G_tile * y_off_part, axis=1)
                ds0_desc.store([BH_IDX, NC_IDX, m_start, d_start], tl.reshape(dS0_tile, (1, 1, BLOCK_M, BLOCK_D)))

            c_probe = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
            dc_prev = tl.reshape(dc_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
            dc_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape((dc_prev + dC_off).to(c_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))

        dlog_off = tl.cumsum(src, axis=0, reverse=True)
        dlog_prev = tl.reshape(dlog_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
        dlog_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(dlog_prev + dlog_off, (1, 1, BLOCK_C)))


@triton.jit
def ssd_rank4_dense_output_bwd_off_kernel(
    C_ptr,
    log_alpha_ptr,
    grad_y_ptr,
    S0_ptr,
    out_dC_ptr,
    out_dlog_ptr,
    out_dS0_ptr,
    b_size: tl.constexpr,
    h_size: tl.constexpr,
    bh_size: tl.constexpr,
    nc_size: tl.constexpr,
    c_size: tl.constexpr,
    m_size: tl.constexpr,
    d_size: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_nc: tl.constexpr,
    stride_c_c: tl.constexpr,
    stride_c_h: tl.constexpr,
    stride_c_m: tl.constexpr,
    stride_r_b: tl.constexpr,
    stride_r_nc: tl.constexpr,
    stride_r_c: tl.constexpr,
    stride_r_h: tl.constexpr,
    stride_gy_bh: tl.constexpr,
    stride_gy_nc: tl.constexpr,
    stride_gy_c: tl.constexpr,
    stride_gy_d: tl.constexpr,
    stride_s0_bh: tl.constexpr,
    stride_s0_nc: tl.constexpr,
    stride_s0_m: tl.constexpr,
    stride_s0_d: tl.constexpr,
    stride_dc_bh: tl.constexpr,
    stride_dc_nc: tl.constexpr,
    stride_dc_c: tl.constexpr,
    stride_dc_m: tl.constexpr,
    stride_dlog_bh: tl.constexpr,
    stride_dlog_nc: tl.constexpr,
    stride_dlog_c: tl.constexpr,
    stride_ds0_bh: tl.constexpr,
    stride_ds0_nc: tl.constexpr,
    stride_ds0_m: tl.constexpr,
    stride_ds0_d: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    M_STATIC: tl.constexpr,
    D_STATIC: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
):
    """Rank-4 phase-3 off-path backward: dC_off + dS0 + dlog_off."""
    pid_bhnc = tl.program_id(0)
    if pid_bhnc >= bh_size * nc_size:
        return

    BH_IDX = pid_bhnc // nc_size
    NC_IDX = pid_bhnc % nc_size
    B_IDX = BH_IDX // h_size
    H_IDX = BH_IDX % h_size

    c_desc = tl.make_tensor_descriptor(
        C_ptr,
        shape=[b_size, nc_size, c_size, h_size, m_size],
        strides=[stride_c_b, stride_c_nc, stride_c_c, stride_c_h, stride_c_m],
        block_shape=[1, 1, BLOCK_C, 1, BLOCK_M],
    )
    gy_desc = tl.make_tensor_descriptor(
        grad_y_ptr,
        shape=[bh_size, nc_size, c_size, d_size],
        strides=[stride_gy_bh, stride_gy_nc, stride_gy_c, stride_gy_d],
        block_shape=[1, 1, BLOCK_C, BLOCK_D],
    )
    s0_desc = tl.make_tensor_descriptor(
        S0_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_s0_bh, stride_s0_nc, stride_s0_m, stride_s0_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )
    dc_desc = tl.make_tensor_descriptor(
        out_dC_ptr,
        shape=[bh_size, nc_size, c_size, m_size],
        strides=[stride_dc_bh, stride_dc_nc, stride_dc_c, stride_dc_m],
        block_shape=[1, 1, BLOCK_C, BLOCK_M],
    )
    dlog_desc = tl.make_tensor_descriptor(
        out_dlog_ptr,
        shape=[bh_size, nc_size, c_size],
        strides=[stride_dlog_bh, stride_dlog_nc, stride_dlog_c],
        block_shape=[1, 1, BLOCK_C],
    )
    ds0_desc = tl.make_tensor_descriptor(
        out_dS0_ptr,
        shape=[bh_size, nc_size, m_size, d_size],
        strides=[stride_ds0_bh, stride_ds0_nc, stride_ds0_m, stride_ds0_d],
        block_shape=[1, 1, BLOCK_M, BLOCK_D],
    )

    offs_c = tl.arange(0, BLOCK_C)
    r_base = B_IDX * stride_r_b + NC_IDX * stride_r_nc + H_IDX * stride_r_h
    log_alpha_vals = tl.load(log_alpha_ptr + r_base + offs_c * stride_r_c).to(tl.float32)
    INV_LN2 = 1.4426950408889634
    log_p = tl.cumsum(log_alpha_vals, axis=0)
    p = tl.exp2(log_p * INV_LN2)
    src = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for m_blk in tl.static_range(0, triton.cdiv(M_STATIC, BLOCK_M)):
        m_start = m_blk * BLOCK_M
        dC_off = tl.zeros((BLOCK_C, BLOCK_M), dtype=tl.float32)
        C_tile_in = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        C_tile_tc = C_tile_in.to(tl.bfloat16)
        for d_blk in tl.static_range(0, triton.cdiv(D_STATIC, BLOCK_D)):
            d_start = d_blk * BLOCK_D
            G_tile = tl.reshape(gy_desc.load([BH_IDX, NC_IDX, 0, d_start]), (BLOCK_C, BLOCK_D)).to(tl.float32)
            dB_tile = p[:, None] * G_tile
            S0_tile = tl.reshape(s0_desc.load([BH_IDX, NC_IDX, m_start, d_start]), (BLOCK_M, BLOCK_D))
            dB_tc = dB_tile.to(tl.bfloat16)
            S0_tc = S0_tile.to(tl.bfloat16)
            y_off_part = p[:, None] * tl.dot(C_tile_tc, S0_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dC_off = dC_off + tl.dot(dB_tc, tl.trans(S0_tc), out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            dS0_tile = tl.dot(tl.trans(C_tile_tc), dB_tc, out_dtype=tl.float32, input_precision=INPUT_PRECISION)
            src += tl.sum(G_tile * y_off_part, axis=1)
            ds0_desc.store([BH_IDX, NC_IDX, m_start, d_start], tl.reshape(dS0_tile, (1, 1, BLOCK_M, BLOCK_D)))

        c_probe = tl.reshape(c_desc.load([B_IDX, NC_IDX, 0, H_IDX, m_start]), (BLOCK_C, BLOCK_M))
        dc_prev = tl.reshape(dc_desc.load([BH_IDX, NC_IDX, 0, m_start]), (BLOCK_C, BLOCK_M)).to(tl.float32)
        dc_desc.store([BH_IDX, NC_IDX, 0, m_start], tl.reshape((dc_prev + dC_off).to(c_probe.dtype), (1, 1, BLOCK_C, BLOCK_M)))

    dlog_off = tl.cumsum(src, axis=0, reverse=True)
    dlog_prev = tl.reshape(dlog_desc.load([BH_IDX, NC_IDX, 0]), (BLOCK_C,)).to(tl.float32)
    dlog_desc.store([BH_IDX, NC_IDX, 0], tl.reshape(dlog_prev + dlog_off, (1, 1, BLOCK_C)))



# ==================================================================================================
# PHASE 3 AUTOGRAD
# ==================================================================================================

def _ssd_rank1_dense_output_forward_impl(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    S0: torch.Tensor | None = None,
    INPUT_PRECISION: str = "tf32",
    RETURN_Y_OFF: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if C.device != W.device or C.device != V.device or C.device != log_alpha.device:
        raise ValueError("C, W, V, log_alpha must be on the same device.")
    if C.dtype != W.dtype or C.dtype != V.dtype or C.dtype != log_alpha.dtype:
        raise ValueError("C, W, V, log_alpha must share dtype.")
    if C.ndim == 5 and W.ndim == 5 and V.ndim == 5 and log_alpha.ndim == 4:
        B, NC, C_CHUNK, H, M = C.shape
        if W.shape != (B, NC, C_CHUNK, H, M):
            raise ValueError(f"W must be [B,NC,C,H,M]={B, NC, C_CHUNK, H, M}; got {tuple(W.shape)}.")
        if V.shape[:4] != (B, NC, C_CHUNK, H):
            raise ValueError(f"V must match [B,NC,C,H,*]={B, NC, C_CHUNK, H}; got {tuple(V.shape)}.")
        D = V.shape[-1]
        if log_alpha.shape != (B, NC, C_CHUNK, H):
            raise ValueError(f"log_alpha must be [B,NC,C,H]={B, NC, C_CHUNK, H}; got {tuple(log_alpha.shape)}.")
        C_5d, W_5d, V_5d, log_alpha_4d = C, W, V, log_alpha
    elif C.ndim == 4 and W.ndim == 4 and V.ndim == 4 and log_alpha.ndim == 3:
        BH, NC, C_CHUNK, M = C.shape
        if W.shape != (BH, NC, C_CHUNK, M):
            raise ValueError(f"W must be [BH,NC,C,M]={BH, NC, C_CHUNK, M}; got {tuple(W.shape)}.")
        if V.shape[:3] != (BH, NC, C_CHUNK):
            raise ValueError(f"V must match [BH,NC,C,*]={BH, NC, C_CHUNK}; got {tuple(V.shape)}.")
        D = V.shape[-1]
        if log_alpha.shape != (BH, NC, C_CHUNK):
            raise ValueError(f"log_alpha must be [BH,NC,C]={BH, NC, C_CHUNK}; got {tuple(log_alpha.shape)}.")
        B, H = BH, 1
        C_5d = C.unsqueeze(3)
        W_5d = W.unsqueeze(3)
        V_5d = V.unsqueeze(3)
        log_alpha_4d = log_alpha.unsqueeze(3)
    else:
        raise ValueError(
            "_ssd_rank1_dense_output_forward_impl expects either "
            "C/W=[B,NC,C,H,M], V=[B,NC,C,H,D], log_alpha=[B,NC,C,H] "
            "or legacy C/W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    BH = B * H
    _require_supported_md(M, D, where="_ssd_rank1_dense_output_forward_impl")
    _require_chunk_size_multiple_of_16(C_CHUNK, where="_ssd_rank1_dense_output_forward_impl")
    _require_nonpositive_log_alpha(log_alpha_4d, where="_ssd_rank1_dense_output_forward_impl")
    if not C_5d.is_cuda:
        raise NotImplementedError("_ssd_rank1_dense_output_forward_impl requires CUDA tensors.")
    _ensure_triton_allocator()

    if S0 is None:
        S0_md = torch.zeros((BH, NC, M, D), device=C_5d.device, dtype=C_5d.dtype)
    else:
        if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
            S0_md = S0.reshape(BH, NC, M, D)
            if S0.dtype != C_5d.dtype:
                S0_md = S0_md.to(C_5d.dtype)
        elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
            S0_md = S0 if S0.dtype == C_5d.dtype else S0.to(C_5d.dtype)
        else:
            raise ValueError(
                f"S0 must be [BH,NC,MD]=({BH},{NC},{M * D}) or [BH,NC,M,D]=({BH},{NC},{M},{D}); "
                f"got {tuple(S0.shape)}."
            )
        if S0_md.device != C_5d.device:
            raise ValueError("S0 must be on the same device as C/W/V/log_alpha.")

    out = torch.empty((BH, NC, C_CHUNK, D), device=C_5d.device, dtype=C_5d.dtype)
    y_off_saved = torch.empty((BH, NC, C_CHUNK, D), device=C_5d.device, dtype=C_5d.dtype) if RETURN_Y_OFF else out
    phase3_fwd_cfg = _select_phase3_forward_launch_config(
        BH=BH,
        NC=NC,
        C_CHUNK=C_CHUNK,
        M=M,
        D=D,
        where="_ssd_rank1_dense_output_forward_impl",
    )
    grid = (BH * NC, D // phase3_fwd_cfg.block_d)
    ssd_rank1_dense_output_fwd_kernel[grid](
        C_5d,
        W_5d,
        V_5d,
        log_alpha_4d,
        S0_md,
        out,
        y_off_saved,
        B,
        H,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *C_5d.stride(),
        *W_5d.stride(),
        *V_5d.stride(),
        *log_alpha_4d.stride(),
        *S0_md.stride(),
        *out.stride(),
        *y_off_saved.stride(),
        BLOCK_M=phase3_fwd_cfg.block_m,
        BLOCK_D=phase3_fwd_cfg.block_d,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        M_STATIC=M,
        INPUT_PRECISION=INPUT_PRECISION,
        WRITE_Y_OFF=RETURN_Y_OFF,
        num_warps=phase3_fwd_cfg.num_warps,
        num_stages=phase3_fwd_cfg.num_stages,
    )
    if RETURN_Y_OFF:
        return out, y_off_saved
    return out


def _ssd_rank1_dense_output_backward_impl(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    grad_out: torch.Tensor | None,
    *,
    S0_saved: torch.Tensor,
    input_precision: str,
    s0_was_flat: bool,
    return_s0_grad_fp32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Phase-3 backward for chunk-local dense replay/output.

    Per `(BH, NC)` forward equations:
      R = C @ W^T
      L[i,j] = 1_{j<=i} * prod_{u=j+1..i} alpha_u, alpha_u = exp(log_alpha_u)
      K = L * R
      Y_diag = K @ V
      Y_off  = p * (C @ S0), p_i = prod_{u=0..i} alpha_u
      Y = Y_diag + Y_off

    Backward decomposition used here:
      1) One fused kernel computes dense-path gradients:
         - `dV`, `dC_diag`, `dW`, `dlog_diag`
      2) When `S0` exists, the same kernel also applies off-path contributions:
         - `dC_off` accumulated in-place into `dC_diag`
         - `dS0`
         - `dlog_off` accumulated in-place into `dlog_diag`

    Returned tensors are phase-3 contributions only:
      - `dC`, `dW`, `dV`, `dlog_alpha`
      - `dS0` (used by phase-2 backward), optionally kept in fp32.
    """
    if C.ndim == 5 and W.ndim == 5 and V.ndim == 5 and log_alpha.ndim == 4:
        B, NC, C_CHUNK, H, M = C.shape
        if W.shape != (B, NC, C_CHUNK, H, M):
            raise ValueError(f"W must be [B,NC,C,H,M]={B, NC, C_CHUNK, H, M}; got {tuple(W.shape)}.")
        if V.shape[:4] != (B, NC, C_CHUNK, H):
            raise ValueError(f"V must match [B,NC,C,H,*]={B, NC, C_CHUNK, H}; got {tuple(V.shape)}.")
        D = V.shape[-1]
        if log_alpha.shape != (B, NC, C_CHUNK, H):
            raise ValueError(f"log_alpha must be [B,NC,C,H]={B, NC, C_CHUNK, H}; got {tuple(log_alpha.shape)}.")
        C_5d, W_5d, V_5d, log_alpha_4d = C, W, V, log_alpha
    elif C.ndim == 4 and W.ndim == 4 and V.ndim == 4 and log_alpha.ndim == 3:
        BH, NC, C_CHUNK, M = C.shape
        if W.shape != (BH, NC, C_CHUNK, M):
            raise ValueError(f"W must be [BH,NC,C,M]={BH, NC, C_CHUNK, M}; got {tuple(W.shape)}.")
        if V.shape[:3] != (BH, NC, C_CHUNK):
            raise ValueError(f"V must match [BH,NC,C,*]={BH, NC, C_CHUNK}; got {tuple(V.shape)}.")
        D = V.shape[-1]
        if log_alpha.shape != (BH, NC, C_CHUNK):
            raise ValueError(f"log_alpha must be [BH,NC,C]={BH, NC, C_CHUNK}; got {tuple(log_alpha.shape)}.")
        B, H = BH, 1
        C_5d = C.unsqueeze(3)
        W_5d = W.unsqueeze(3)
        V_5d = V.unsqueeze(3)
        log_alpha_4d = log_alpha.unsqueeze(3)
    else:
        raise ValueError(
            "_ssd_rank1_dense_output_backward_impl expects either "
            "C/W=[B,NC,C,H,M], V=[B,NC,C,H,D], log_alpha=[B,NC,C,H] "
            "or legacy C/W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C]. "
            f"Got C={tuple(C.shape)}, W={tuple(W.shape)}, V={tuple(V.shape)}, log_alpha={tuple(log_alpha.shape)}."
        )
    BH = B * H
    _require_supported_md(M, D, where="_ssd_rank1_dense_output_backward_impl")

    if grad_out is None:
        grad_out_c = torch.zeros((BH, NC, C_CHUNK, D), device=C_5d.device, dtype=C_5d.dtype)
    elif grad_out.ndim == 4 and grad_out.shape == (BH, NC, C_CHUNK, D):
        grad_out_c = grad_out
    elif grad_out.ndim == 5 and grad_out.shape == (B, NC, C_CHUNK, H, D):
        grad_out_c = grad_out.permute(0, 3, 1, 2, 4).reshape(BH, NC, C_CHUNK, D)
    else:
        raise ValueError(
            "grad_out must be [BH,NC,C,D] or [B,NC,C,H,D]. "
            f"Got {tuple(grad_out.shape)} for expected BH={BH}, B={B}, H={H}, NC={NC}, C={C_CHUNK}, D={D}."
        )
    _require_chunk_size_multiple_of_16(C_CHUNK, where="_ssd_rank1_dense_output_backward_impl")
    if not C_5d.is_cuda:
        raise NotImplementedError("_ssd_rank1_dense_output_backward_impl requires CUDA tensors.")
    _ensure_triton_allocator()

    if S0_saved.ndim == 3 and S0_saved.shape == (BH, NC, M * D):
        S0_md = S0_saved.reshape(BH, NC, M, D)
    elif S0_saved.ndim == 4 and S0_saved.shape == (BH, NC, M, D):
        S0_md = S0_saved
    else:
        raise ValueError(
            f"S0_saved must be [BH,NC,MD]=({BH},{NC},{M * D}) or [BH,NC,M,D]=({BH},{NC},{M},{D}); "
            f"got {tuple(S0_saved.shape)}."
        )

    phase3_bwd_cfg = _select_phase3_backward_launch_config(
        BH=BH,
        NC=NC,
        C_CHUNK=C_CHUNK,
        M=M,
        D=D,
    )
    BLOCK_M = phase3_bwd_cfg.block_m
    BLOCK_D = phase3_bwd_cfg.block_d
    if M % BLOCK_M != 0 or D % BLOCK_D != 0:
        raise NotImplementedError(
            f"_ssd_rank1_dense_output_backward_impl requires M%BLOCK_M==0 and D%BLOCK_D==0; "
            f"got M={M}, D={D}, BLOCK_M={BLOCK_M}, BLOCK_D={BLOCK_D}."
        )
    dV = torch.empty((BH, NC, C_CHUNK, D), device=C_5d.device, dtype=C_5d.dtype)
    dC = torch.empty((BH, NC, C_CHUNK, M), device=C_5d.device, dtype=C_5d.dtype)
    dW = torch.empty((BH, NC, C_CHUNK, M), device=C_5d.device, dtype=C_5d.dtype)
    dlog = _ssd_rank1_bwd_workspace_tensor(
        "dlog_diag",
        (BH, NC, C_CHUNK),
        device=C_5d.device,
        dtype=torch.float32,
    )

    has_s0 = S0_md.numel() > 0
    dS0 = (
        torch.empty((BH, NC, M, D), device=C_5d.device, dtype=torch.float32)
        if has_s0
        else torch.empty((1, 1, 1, 1), device=C_5d.device, dtype=torch.float32)
    )

    grid_fused = (BH * NC,)
    # Unified phase-3 backward:
    #   Dense path: dV, dC, dW, dlog.
    #   Optional off path: dC_off accumulation, dS0, dlog_off accumulation.
    phase3_bwd_num_warps = max(phase3_bwd_cfg.fused_a1a2_num_warps, phase3_bwd_cfg.fused_off_num_warps)
    ssd_rank1_dense_output_bwd_fused_kernel[grid_fused](
        C_5d,
        W_5d,
        V_5d,
        log_alpha_4d,
        grad_out_c,
        S0_md,
        dV,
        dC,
        dW,
        dlog,
        dS0,
        B,
        H,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *C_5d.stride(),
        *W_5d.stride(),
        *V_5d.stride(),
        *log_alpha_4d.stride(),
        *grad_out_c.stride(),
        *S0_md.stride(),
        *dV.stride(),
        *dC.stride(),
        *dW.stride(),
        *dlog.stride(),
        *dS0.stride(),
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        M_STATIC=M,
        D_STATIC=D,
        INPUT_PRECISION=input_precision,
        HAS_S0=has_s0,
        num_warps=phase3_bwd_num_warps,
        num_stages=phase3_bwd_cfg.num_stages,
    )
    dS0_out: torch.Tensor | None
    if has_s0:
        dS0_out = dS0.reshape(BH, NC, M * D) if s0_was_flat else dS0
        if not return_s0_grad_fp32:
            dS0_out = dS0_out.to(C_5d.dtype)
    else:
        dS0_out = None

    return dC, dW, dV, dlog.to(log_alpha_4d.dtype), dS0_out


def ssd_rank1_dense_output_forward_kernel_profile(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    S0: torch.Tensor | None = None,
    INPUT_PRECISION: str = "tf32",
    BLOCK_M: int | None = None,
    BLOCK_D: int | None = None,
    NUM_WARPS: int | None = None,
    NUM_STAGES: int | None = None,
    RETURN_Y_OFF: bool = False,
    WARMUP: bool = True,
) -> dict[str, float]:
    """Profile fused Phase-3 Triton forward kernel and return timings (ms).

    Expects chunked tensors:
      C=[BH,NC,C,M], W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C].
    """
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3:
        raise ValueError(
            "ssd_rank1_dense_output_forward_kernel_profile expects "
            "C=[BH,NC,C,M], W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C]."
        )
    BH, NC, C_CHUNK, M = C.shape
    if W.shape != (BH, NC, C_CHUNK, M):
        raise ValueError("W shape mismatch.")
    if V.shape[:3] != (BH, NC, C_CHUNK):
        raise ValueError("V shape mismatch.")
    D = V.shape[-1]
    if log_alpha.shape != (BH, NC, C_CHUNK):
        raise ValueError("log_alpha shape mismatch.")
    _require_supported_md(M, D, where="ssd_rank1_dense_output_forward_kernel_profile")
    _require_chunk_size_multiple_of_16(C_CHUNK, where="ssd_rank1_dense_output_forward_kernel_profile")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank1_dense_output_forward_kernel_profile")
    if not C.is_cuda:
        raise NotImplementedError("ssd_rank1_dense_output_forward_kernel_profile requires CUDA tensors.")
    _ensure_triton_allocator()

    if S0 is None:
        S0_md = torch.zeros((BH, NC, M, D), device=C.device, dtype=C.dtype)
    else:
        if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
            S0_md = S0.reshape(BH, NC, M, D).to(C.dtype)
        elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
            S0_md = S0.to(C.dtype)
        else:
            raise ValueError("S0 shape mismatch.")

    default_cfg = _select_phase3_forward_launch_config(
        BH=BH,
        NC=NC,
        C_CHUNK=C_CHUNK,
        M=M,
        D=D,
        where="ssd_rank1_dense_output_forward_kernel_profile",
    )
    BLOCK_M = default_cfg.block_m if BLOCK_M is None else BLOCK_M
    BLOCK_D = default_cfg.block_d if BLOCK_D is None else BLOCK_D
    NUM_WARPS = default_cfg.num_warps if NUM_WARPS is None else NUM_WARPS
    NUM_STAGES = default_cfg.num_stages if NUM_STAGES is None else NUM_STAGES
    if M % BLOCK_M != 0 or D % BLOCK_D != 0:
        raise NotImplementedError(
            "Profile currently requires M%BLOCK_M==0 and D%BLOCK_D==0; "
            f"got M={M}, D={D}, BLOCK_M={BLOCK_M}, BLOCK_D={BLOCK_D}."
        )

    out = torch.empty((BH, NC, C_CHUNK, D), device=C.device, dtype=C.dtype)
    y_off_saved = torch.empty_like(out) if RETURN_Y_OFF else out
    C_5d = C.unsqueeze(3)
    W_5d = W.unsqueeze(3)
    V_5d = V.unsqueeze(3)
    log_alpha_4d = log_alpha.unsqueeze(3)
    B = BH
    H = 1
    times: dict[str, float] = {}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def _time(label: str, fn) -> None:
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times[label] = float(start.elapsed_time(end))

    def _run_fused() -> None:
        grid = (BH * NC, D // BLOCK_D)
        ssd_rank1_dense_output_fwd_kernel[grid](
            C_5d,
            W_5d,
            V_5d,
            log_alpha_4d,
            S0_md,
            out,
            y_off_saved,
            B,
            H,
            BH,
            NC,
            C_CHUNK,
            M,
            D,
            *C_5d.stride(),
            *W_5d.stride(),
            *V_5d.stride(),
            *log_alpha_4d.stride(),
            *S0_md.stride(),
            *out.stride(),
            *y_off_saved.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=C_CHUNK,
            C_STATIC=C_CHUNK,
            M_STATIC=M,
            INPUT_PRECISION=INPUT_PRECISION,
            WRITE_Y_OFF=RETURN_Y_OFF,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )

    if WARMUP:
        _run_fused()
        torch.cuda.synchronize()
    _time("fused_ms", _run_fused)
    times["total_ms"] = times["fused_ms"]
    times["BLOCK_M"] = float(BLOCK_M)
    times["BLOCK_D"] = float(BLOCK_D)
    times["NUM_WARPS"] = float(NUM_WARPS)
    times["NUM_STAGES"] = float(NUM_STAGES)
    return times


def ssd_rank1_dense_output_backward_kernel_profile(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    grad_out: torch.Tensor,
    S0: torch.Tensor | None = None,
    INPUT_PRECISION: str = "tf32",
    BLOCK_M: int | None = None,
    BLOCK_D: int | None = None,
    FUSED_A1A2_NUM_WARPS: int | None = None,
    OFF_NUM_WARPS: int | None = None,
    NUM_STAGES: int | None = None,
    WARMUP: bool = True,
) -> dict[str, float]:
    """Profile fused Phase-3 Triton backward kernel and return timings (ms).

    This mirrors the phase-3 backward path without autograd plumbing.
    """
    if C.ndim != 4 or W.ndim != 4 or V.ndim != 4 or log_alpha.ndim != 3 or grad_out.ndim != 4:
        raise ValueError(
            "ssd_rank1_dense_output_backward_kernel_profile expects "
            "C=[BH,NC,C,M], W=[BH,NC,C,M], V=[BH,NC,C,D], log_alpha=[BH,NC,C], grad_out=[BH,NC,C,D]."
        )
    BH, NC, C_CHUNK, M = C.shape
    if W.shape != (BH, NC, C_CHUNK, M):
        raise ValueError("W shape mismatch.")
    if V.shape[:3] != (BH, NC, C_CHUNK):
        raise ValueError("V shape mismatch.")
    D = V.shape[-1]
    if log_alpha.shape != (BH, NC, C_CHUNK):
        raise ValueError("log_alpha shape mismatch.")
    if grad_out.shape != (BH, NC, C_CHUNK, D):
        raise ValueError("grad_out shape mismatch.")
    _require_supported_md(M, D, where="ssd_rank1_dense_output_backward_kernel_profile")
    _require_chunk_size_multiple_of_16(C_CHUNK, where="ssd_rank1_dense_output_backward_kernel_profile")
    _require_nonpositive_log_alpha(log_alpha, where="ssd_rank1_dense_output_backward_kernel_profile")
    if not C.is_cuda:
        raise NotImplementedError("ssd_rank1_dense_output_backward_kernel_profile requires CUDA tensors.")
    _ensure_triton_allocator()

    if S0 is None:
        S0_md = torch.zeros((BH, NC, M, D), device=C.device, dtype=C.dtype)
        has_s0 = False
    else:
        if S0.ndim == 3 and S0.shape == (BH, NC, M * D):
            S0_md = S0.reshape(BH, NC, M, D).to(C.dtype)
        elif S0.ndim == 4 and S0.shape == (BH, NC, M, D):
            S0_md = S0.to(C.dtype)
        else:
            raise ValueError("S0 shape mismatch.")
        has_s0 = True

    default_cfg = _select_phase3_backward_launch_config(
        BH=BH,
        NC=NC,
        C_CHUNK=C_CHUNK,
        M=M,
        D=D,
    )
    BLOCK_M = default_cfg.block_m if BLOCK_M is None else BLOCK_M
    BLOCK_D = default_cfg.block_d if BLOCK_D is None else BLOCK_D
    FUSED_A1A2_NUM_WARPS = (
        default_cfg.fused_a1a2_num_warps if FUSED_A1A2_NUM_WARPS is None else FUSED_A1A2_NUM_WARPS
    )
    OFF_NUM_WARPS = default_cfg.fused_off_num_warps if OFF_NUM_WARPS is None else OFF_NUM_WARPS
    NUM_STAGES = default_cfg.num_stages if NUM_STAGES is None else NUM_STAGES
    if M % BLOCK_M != 0 or D % BLOCK_D != 0:
        raise NotImplementedError(
            f"Profile currently requires M%BLOCK_M==0 and D%BLOCK_D==0; got M={M}, D={D}, BLOCK_M={BLOCK_M}, BLOCK_D={BLOCK_D}."
        )

    grad_out_c = grad_out if grad_out.is_contiguous() else grad_out.contiguous()

    times: dict[str, float] = {}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    def _time(label: str, fn) -> None:
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times[label] = float(start.elapsed_time(end))

    dV = torch.empty_like(V)
    dC = torch.empty_like(C)
    dW = torch.empty_like(W)
    dlog = torch.empty((BH, NC, C_CHUNK), device=C.device, dtype=torch.float32)
    dS0 = torch.empty((BH, NC, M, D), device=C.device, dtype=torch.float32)
    C_5d = C.unsqueeze(3)
    W_5d = W.unsqueeze(3)
    V_5d = V.unsqueeze(3)
    log_alpha_4d = log_alpha.unsqueeze(3)
    B = BH
    H = 1

    FUSED_NUM_WARPS = max(FUSED_A1A2_NUM_WARPS, OFF_NUM_WARPS)

    def _run_fused():
        grid_fused = (BH * NC,)
        ssd_rank1_dense_output_bwd_fused_kernel[grid_fused](
            C_5d,
            W_5d,
            V_5d,
            log_alpha_4d,
            grad_out_c,
            S0_md,
            dV,
            dC,
            dW,
            dlog,
            dS0,
            B,
            H,
            BH,
            NC,
            C_CHUNK,
            M,
            D,
            *C_5d.stride(),
            *W_5d.stride(),
            *V_5d.stride(),
            *log_alpha_4d.stride(),
            *grad_out_c.stride(),
            *S0_md.stride(),
            *dV.stride(),
            *dC.stride(),
            *dW.stride(),
            *dlog.stride(),
            *dS0.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=C_CHUNK,
            C_STATIC=C_CHUNK,
            M_STATIC=M,
            D_STATIC=D,
            INPUT_PRECISION=INPUT_PRECISION,
            HAS_S0=has_s0,
            num_warps=FUSED_NUM_WARPS,
            num_stages=NUM_STAGES,
        )

    if WARMUP:
        _run_fused()
        torch.cuda.synchronize()
    _time("fused_ms", _run_fused)
    times["total_ms"] = times["fused_ms"]
    times["BLOCK_M"] = float(BLOCK_M)
    times["BLOCK_D"] = float(BLOCK_D)
    times["FUSED_A1A2_NUM_WARPS"] = float(FUSED_A1A2_NUM_WARPS)
    times["OFF_NUM_WARPS"] = float(OFF_NUM_WARPS)
    times["FUSED_NUM_WARPS"] = float(FUSED_NUM_WARPS)

    return times




# --------------------------------------------------------------------------------------------------
# END PHASE 3: DENSE OUTPUT COMPUTATION
# --------------------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------------------
# PHASE 123: FULL PARALLEL SCAN (UNIFIED AUTOGRAD)
# --------------------------------------------------------------------------------------------------

class SsdRank1TritonDebug(torch.autograd.Function):
    """Unified SSD rank-1 Triton autograd function (Phases 1+2+3).

    High-level dataflow:
      1) Forward chunks [B,N,H,*] -> [BH,NC,C,*], then runs Phase 1/2/3 kernels.
      2) Forward returns:
           - y_chunk: chunk-local outputs [BH,NC,C,D]
           - S1_chunk: final recurrent state per BH lane [BH,MD]
      3) Backward recomputes the minimum needed intermediates (Phase 1 + 2 forward)
         and then executes explicit Phase 3 -> Phase 2 -> Phase 1 backward kernels.

    Why unified:
      - We avoid a stack of nested autograd.Functions and keep one explicit backward.
      - We can choose exactly which tensors are retained versus recomputed.
      - We keep memory pressure manageable by not saving large fp32 intermediates
        that are cheap to regenerate.
    """

    @staticmethod
    def forward(
        ctx,
        C: torch.Tensor,
        W: torch.Tensor,
        V: torch.Tensor,
        log_alpha: torch.Tensor,
        initial_state: torch.Tensor | None,
        CHUNK_SIZE: int,
        INPUT_PRECISION: str,
        RETURN_FINAL_STATE: bool,
    ):
        # Step 0: validate contract and convert to kernel-friendly chunked layout.
        # Input  layout: C/W [B,N,H,M], V [B,N,H,D], log_alpha [B,N,H]
        # Kernel layout: C/W [BH,NC,C,M], V [BH,NC,C,D], log_alpha [BH,NC,C]
        # where BH=B*H, NC=#chunks, C=CHUNK_SIZE.
        C_chunk, W_chunk, V_chunk, log_alpha_chunk, init_flat, B, N, H, M, D, NC_exec = _ssd_rank1_prepare_unchunked_inputs(
            C, W, V, log_alpha, initial_state,
            where="SsdRank1TritonDebug.forward",
            CHUNK_SIZE=CHUNK_SIZE,
            materialize_zero_init=False,
        )

        # =========================================
        # FORWARD PHASE 1: CHUNK LOCAL END-STATE
        # =========================================
        # Per-chunk local end-state summaries.
        # Produces S_local_end [BH,NC,MD] in fp32-like accumulate domain.
        S_local_end = _ssd_rank1_chunk_end_state_forward_impl(W_chunk, V_chunk, log_alpha_chunk, 32)

        # =========================================
        # FORWARD PHASE 2: PREFIX SCAN OVER CHUNKS
        # =========================================
        # Prefix-scan local chunk summaries to produce
        # chunk-start state S0 per chunk and final state S1 at end of sequence.
        log_alpha_per_chunk = log_alpha_chunk.float().sum(dim=2)
        S0_chunk, S1_chunk = _ssd_rank1_prefix_scan_forward_impl(
            S_local_end,
            log_alpha_per_chunk,
            init_flat,
            V_chunk.dtype,
            return_final_state=RETURN_FINAL_STATE,
        )

        # =========================================
        # FORWARD PHASE 3: DENSE CHUNK OUTPUT
        # =========================================
        # Dense intra-chunk output.
        y_chunk = _ssd_rank1_dense_output_forward_impl(
            C_chunk, W_chunk, V_chunk, log_alpha_chunk, S0_chunk, INPUT_PRECISION,
            RETURN_Y_OFF=False,
        )

        # Saved tensors policy:
        # - Save user-facing unchunked tensors and replay chunking in backward.
        # - Do not save S_local_end or S0_chunk (recompute them in backward).
        # This keeps memory lower while preserving fast backward paths.
        if initial_state is None:
            ctx.save_for_backward(C, W, V, log_alpha)
            ctx.has_initial_state = False
        else:
            ctx.save_for_backward(C, W, V, log_alpha, initial_state)
            ctx.has_initial_state = True
        ctx.CHUNK_SIZE = CHUNK_SIZE
        ctx.INPUT_PRECISION = INPUT_PRECISION
        ctx.initial_state_shape = None if initial_state is None else tuple(initial_state.shape)
        ctx.return_final_state = bool(RETURN_FINAL_STATE)

        # Return chunked outputs; outer wrapper restores [B,N,H,*] layout.
        if RETURN_FINAL_STATE:
            return y_chunk, S1_chunk
        return y_chunk

    @staticmethod
    def backward(ctx, *grad_outputs):
        """Backward pass for unified phase-1/2/3 SSD rank-1.

        Input adjoints:
          - grad_y_chunk  = dL/dY_chunk  [BH,NC,C,D]
          - grad_S1_chunk = dL/dS_final  [BH,MD] (optional)

        Chain-rule schedule:
          1) Replay phase-1 and phase-2 forwards to recover `S_local_end`, `S0_chunk`.
          2) Phase-3 backward:
               (C, W, V, log_alpha, S0) + dY
                 -> dC_p3, dW_p3, dV_p3, dlog_p3, dS0
          3) Phase-2 backward:
               (S_local_end, log_alpha_per_chunk, init) + (dS0, dS_final)
                 -> dS_local_end, dlog_per_chunk, dinit
          4) Phase-1 backward:
               (W, V, log_alpha) + dS_local_end
                 -> dW_p1, dV_p1, dlog_p1
          5) Accumulate and reshape:
               dW = dW_p3 + dW_p1
               dV = dV_p3 + dV_p1
               dlog = dlog_p3 + dlog_p1 + broadcast(dlog_per_chunk over chunk tokens)
        """
        # Upstream gradients:
        #   grad_y_chunk  : dL/dy_chunk  [BH,NC,C,D]
        #   grad_S1_chunk : dL/dS1_chunk [BH,MD] or None
        # We compute gradients for C/W/V/log_alpha/initial_state and None for
        # non-tensor args (CHUNK_SIZE, INPUT_PRECISION).
        grad_y_chunk = grad_outputs[0]
        grad_S1_chunk = grad_outputs[1] if (ctx.return_final_state and len(grad_outputs) > 1) else None

        if ctx.has_initial_state:
            C, W, V, log_alpha, initial_state = ctx.saved_tensors
        else:
            C, W, V, log_alpha = ctx.saved_tensors
            initial_state = None
        CHUNK_SIZE = ctx.CHUNK_SIZE
        INPUT_PRECISION = ctx.INPUT_PRECISION
        C_chunk, W_chunk, V_chunk, log_alpha_chunk, init_flat, B, N, H, M, D, NC_exec = _ssd_rank1_prepare_unchunked_inputs(
            C, W, V, log_alpha, initial_state,
            where="SsdRank1TritonDebug.backward",
            CHUNK_SIZE=CHUNK_SIZE,
            materialize_zero_init=False,
        )
        input_dtype = V_chunk.dtype
        compute_dtype = V_chunk.dtype
        initial_state_shape = ctx.initial_state_shape
        BH = B * H
        NC = NC_exec
        MD = M * D

        # =========================================
        # BACKWARD REPLAY: RECOMPUTE FORWARD INTERMEDIATES
        # =========================================
        # Phase 1/2 forward replay for full-sequence backward.
        S_local_end = _ssd_rank1_chunk_end_state_forward_impl(W_chunk, V_chunk, log_alpha_chunk, 32)
        log_alpha_per_chunk = log_alpha_chunk.float().sum(dim=2)
        S0_chunk, _ = _ssd_rank1_prefix_scan_forward_impl(
            S_local_end, log_alpha_per_chunk, init_flat, compute_dtype,
            return_final_state=False,
        )
        del S_local_end

        # Phase-3 backward on full [BH,NC,C,*] tensors.
        dC_chunk, dW_chunk, dV_chunk, dlog_chunk, dS0 = _ssd_rank1_dense_output_backward_impl(
            C_chunk,
            W_chunk,
            V_chunk,
            log_alpha_chunk,
            grad_y_chunk,
            S0_saved=S0_chunk,
            input_precision=INPUT_PRECISION,
            s0_was_flat=True,
            return_s0_grad_fp32=True,
        )
        del grad_y_chunk, C_chunk

        # Phase-2 backward on full [BH,NC,MD] tensors.
        has_grad_final = grad_S1_chunk is not None
        grad_final_for_p2 = grad_S1_chunk
        grad_chunk_start_f = dS0 if dS0.dtype == torch.float32 and dS0.is_contiguous() else dS0.float().contiguous()
        if has_grad_final:
            grad_final_f = (
                grad_final_for_p2
                if grad_final_for_p2.dtype == torch.float32 and grad_final_for_p2.is_contiguous()
                else grad_final_for_p2.float().contiguous()
            )
        else:
            grad_final_f = torch.empty((1, 1), device=V_chunk.device, dtype=torch.float32)
        s0_chunk_f = S0_chunk if S0_chunk.dtype == torch.float32 and S0_chunk.is_contiguous() else S0_chunk.float().contiguous()
        log_alpha_per_chunk_bnh = log_alpha_per_chunk.reshape(B, H, NC).permute(0, 2, 1).contiguous()
        log_alpha_per_chunk_f = (
            log_alpha_per_chunk_bnh
            if log_alpha_per_chunk_bnh.dtype == torch.float32
            else log_alpha_per_chunk_bnh.float()
        )
        # Reuse dS0 buffer for dS_local_end to avoid an extra full [BH,NC,MD] allocation.
        # The phase-2 kernel loads grad_prefix before writing d_s_local for each tile.
        dS_local_end = grad_chunk_start_f
        d_log_per_chunk = _ssd_rank1_bwd_workspace_tensor(
            "phase2_dlog_per_chunk_full",
            (BH, NC),
            device=V_chunk.device,
            dtype=torch.float32,
        )
        write_d_init = initial_state_shape is not None
        d_init_f32 = _ssd_rank1_bwd_workspace_tensor(
            "phase2_dinit_full",
            (BH, MD),
            device=V_chunk.device,
            dtype=torch.float32,
        )
        block_nc = _select_phase2_block_nc(NC=NC)
        phase2_cfg = _select_phase2_launch_config(MD=MD, NC=NC, where="SsdRank1TritonDebug.backward")
        def grid_bwd(_meta):
            return (BH,)
        ssd_rank1_prefix_scan_bwd_dense_kernel[grid_bwd](
            grad_chunk_start_f,
            grad_final_f,
            s0_chunk_f,
            log_alpha_per_chunk_f,
            dS_local_end,
            d_log_per_chunk,
            d_init_f32,
            B,
            H,
            BH,
            NC,
            MD,
            *grad_chunk_start_f.stride(),
            *grad_final_f.stride(),
            *s0_chunk_f.stride(),
            *log_alpha_per_chunk_f.stride(),
            *dS_local_end.stride(),
            *d_log_per_chunk.stride(),
            *d_init_f32.stride(),
            BLOCK_NC=block_nc,
            BLOCK_MD=phase2_cfg.block_md,
            NC_STATIC=NC,
            USE_FP32_COMPUTE=(compute_dtype == torch.float32),
            HAS_GRAD_FINAL=has_grad_final,
            WRITE_D_INIT=write_d_init,
            num_warps=phase2_cfg.num_warps,
            num_stages=phase2_cfg.num_stages,
        )
        del S0_chunk, dS0, grad_final_f, s0_chunk_f, log_alpha_per_chunk_f, log_alpha_per_chunk

        # Phase-1 backward on full [BH,NC,MD], accumulating into phase-3 grads.
        BLOCK_M = _select_largest_block_size(
            M,
            _SUPPORTED_BLOCK_X_VALUES,
            where="SsdRank1TritonDebug.backward",
            label="M",
        )
        BLOCK_D = _select_largest_block_size(
            D,
            _SUPPORTED_BLOCK_X_VALUES,
            where="SsdRank1TritonDebug.backward",
            label="D",
        )
        phase1_input_precision = "ieee" if input_dtype == torch.float32 else "tf32"
        phase1_bwd_cfg = _select_phase1_backward_launch_config(
            M=M,
            D=D,
        )

        grad_s_md = dS_local_end.reshape(BH, NC, M, D)
        grid_phase1 = (BH * NC,)
        ssd_rank1_chunk_end_state_bwd_fused_kernel[grid_phase1](
            grad_s_md,
            W_chunk,
            V_chunk,
            log_alpha_chunk,
            dW_chunk,
            dV_chunk,
            dlog_chunk,
            B,
            H,
            BH,
            NC,
            CHUNK_SIZE,
            M,
            D,
            *grad_s_md.stride(),
            *W_chunk.stride(),
            *V_chunk.stride(),
            *log_alpha_chunk.stride(),
            *dW_chunk.stride(),
            *dV_chunk.stride(),
            *dlog_chunk.stride(),
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
            BLOCK_C=CHUNK_SIZE,
            C_STATIC=CHUNK_SIZE,
            M_STATIC=M,
            D_STATIC=D,
            INPUT_PRECISION=phase1_input_precision,
            ACCUMULATE=True,
            num_warps=phase1_bwd_cfg.num_warps,
            num_stages=phase1_bwd_cfg.num_stages,
        )

        dlog_chunk.add_(d_log_per_chunk.unsqueeze(-1).to(log_alpha_chunk.dtype))
        del grad_chunk_start_f, dS_local_end, W_chunk, V_chunk, log_alpha_chunk

        # Restore initial_state gradient in the same shape family the caller used.
        if initial_state_shape is not None:
            if len(initial_state_shape) == 4:
                d_init = d_init_f32.reshape(B, H, M, D).to(input_dtype)
            else:
                d_init = d_init_f32.reshape(B, H, M * D).to(input_dtype)
        else:
            d_init = None

        dC = _ssd_rank1_restore_grad_layout(dC_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        dW = _ssd_rank1_restore_grad_layout(dW_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        dV = _ssd_rank1_restore_grad_layout(dV_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        if not dlog_chunk.is_contiguous():
            raise ValueError("SsdRank1TritonDebug.backward expects contiguous dlog_chunk storage.")
        # [BH,NC,C] -> [B,N,H] as a zero-copy view (same storage, different strides).
        dlog_alpha = dlog_chunk.as_strided((B, N, H), (H * N, 1, N))

        # Return one gradient slot per forward argument:
        # (C, W, V, log_alpha, initial_state, CHUNK_SIZE, INPUT_PRECISION)
        return dC, dW, dV, dlog_alpha, d_init, None, None, None


def _ssd_rank1_restore_grad_layout(
    grad_chunk: torch.Tensor,
    *,
    B: int,
    N: int,
    H: int,
    C: int,
) -> torch.Tensor:
    """Map [BH, NC, C, *] gradients back to [B, N, H, *] as a zero-copy view."""
    if grad_chunk.ndim < 4:
        raise ValueError(f"_ssd_rank1_restore_grad_layout expects rank>=4, got {tuple(grad_chunk.shape)}.")
    if grad_chunk.shape[0] != B * H:
        raise ValueError(
            f"_ssd_rank1_restore_grad_layout: leading BH mismatch; got {grad_chunk.shape[0]}, expected {B * H}."
        )
    if grad_chunk.shape[2] != C:
        raise ValueError(f"_ssd_rank1_restore_grad_layout: chunk C mismatch; got {grad_chunk.shape[2]}, expected {C}.")
    NC = grad_chunk.shape[1]
    if NC * C != N:
        raise ValueError(
            f"_ssd_rank1_restore_grad_layout requires NC*C == N (no padded tail). "
            f"Got NC={NC}, C={C}, N={N}."
        )
    if not grad_chunk.is_contiguous():
        raise ValueError("_ssd_rank1_restore_grad_layout expects contiguous grad_chunk storage.")

    trailing = grad_chunk.shape[3:]
    trailing_prod = 1
    for t in trailing:
        trailing_prod *= t
    trailing_strides: list[int] = []
    running = trailing_prod
    for t in trailing:
        running //= t
        trailing_strides.append(running)

    # Linearization equivalence:
    #   chunk idx: ((((b*H+h)*NC + nc)*C + c)*T + t)
    #   output idx: (((b*N + n)*H + h)*T + t) with n = nc*C + c
    # We return the [B,N,H,*] view directly over chunk storage.
    out_shape = (B, N, H, *trailing)
    out_strides = (H * N * trailing_prod, trailing_prod, N * trailing_prod, *trailing_strides)
    return grad_chunk.as_strided(out_shape, out_strides)


def _ssd_rank1_prepare_unchunked_inputs_static(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    *,
    cfg: _StaticSsdRank1ShapeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, int, int, int]:
    """Static hot-path input preparation: canonical unchunked layout only."""
    B, N, H, M = C.shape
    D = V.shape[-1]
    CHUNK_SIZE = cfg.chunk_size
    NC = N // CHUNK_SIZE
    BH = B * H
    C_chunk = C.view(B, NC, CHUNK_SIZE, H, M)
    W_chunk = W.view(B, NC, CHUNK_SIZE, H, M)
    V_chunk = V.view(B, NC, CHUNK_SIZE, H, D)
    log_alpha_chunk = log_alpha.view(B, NC, CHUNK_SIZE, H)
    return C_chunk, W_chunk, V_chunk, log_alpha_chunk, B, N, H, M, D, NC, BH


def _ssd_rank1_chunk_end_state_forward_impl_static(
    W_5d: torch.Tensor,
    V_5d: torch.Tensor,
    log_alpha_4d: torch.Tensor,
    *,
    cfg: _StaticSsdRank1ShapeConfig,
    ws: _StaticSsdRank1Workspace,
) -> torch.Tensor:
    """Static hot-path phase-1 forward: canonical [B,NC,C,H,*] only."""
    B, NC, C_CHUNK, H, M = W_5d.shape
    D = V_5d.shape[-1]
    BH = B * H
    s_local_end_md = ws.s_local_end_md
    grid = (BH * NC, M // cfg.phase1_block_m, D // cfg.phase1_block_d)
    ssd_rank1_chunk_end_state_fwd_kernel[grid](
        W_5d,
        V_5d,
        log_alpha_4d,
        s_local_end_md,
        B,
        H,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *W_5d.stride(),
        *V_5d.stride(),
        *log_alpha_4d.stride(),
        *s_local_end_md.stride(),
        BLOCK_M=cfg.phase1_block_m,
        BLOCK_D=cfg.phase1_block_d,
        BLOCK_T=cfg.phase1_block_t,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        INPUT_PRECISION=cfg.input_precision,
    )
    return s_local_end_md.reshape(BH, NC, M * D)


def _ssd_rank1_phase2_forward_static(
    S_local_end: torch.Tensor,
    log_alpha_per_chunk_bnh: torch.Tensor,
    final_state_out: torch.Tensor,
    *,
    cfg: _StaticSsdRank1ShapeConfig,
    ws: _StaticSsdRank1Workspace,
) -> torch.Tensor:
    """Static hot-path phase-2 forward: fixed BF16 compute, no initial-state branch."""
    BH, NC, MD = S_local_end.shape
    B, NC_log, H = log_alpha_per_chunk_bnh.shape
    if NC_log != NC:
        raise ValueError(f"log_alpha_per_chunk_bnh NC mismatch: got {NC_log}, expected {NC}.")
    if B * H != BH:
        raise ValueError(f"log_alpha_per_chunk_bnh B*H mismatch: got {B}*{H}, expected BH={BH}.")
    if final_state_out.shape != (BH, MD):
        raise ValueError(f"final_state_out must be [BH,MD]=({BH},{MD}); got {tuple(final_state_out.shape)}.")
    init_dummy = ws.phase2_init_dummy
    chunk_start = ws.phase2_chunk_start
    phase2_cfg = cfg.phase2_launch
    if cfg.input_dtype == torch.float32:
        use_fp32_compute = True
    elif cfg.input_dtype == torch.bfloat16:
        use_fp32_compute = False
    else:
        raise NotImplementedError(
            "_ssd_rank1_phase2_forward_static only supports cfg.input_dtype in {torch.float32, torch.bfloat16}; "
            f"got {cfg.input_dtype}."
        )
    grid = (BH,)
    ssd_rank1_prefix_scan_fwd_kernel[grid](
        S_local_end,
        log_alpha_per_chunk_bnh,
        init_dummy,
        chunk_start,
        final_state_out,
        B,
        H,
        BH,
        NC,
        MD,
        *S_local_end.stride(),
        *log_alpha_per_chunk_bnh.stride(),
        *init_dummy.stride(),
        *chunk_start.stride(),
        *final_state_out.stride(),
        NC_STATIC=NC,
        BLOCK_NC=cfg.phase2_block_nc,
        BLOCK_MD=phase2_cfg.block_md,
        USE_FP32_COMPUTE=use_fp32_compute,
        HAS_INITIAL_STATE=cfg.has_initial_state,
        RETURN_FINAL_STATE=cfg.return_final_state,
        num_warps=phase2_cfg.num_warps,
        num_stages=phase2_cfg.num_stages,
    )
    return chunk_start


def _ssd_rank1_dense_output_forward_impl_static(
    C_5d: torch.Tensor,
    W_5d: torch.Tensor,
    V_5d: torch.Tensor,
    log_alpha_4d: torch.Tensor,
    S0_chunk: torch.Tensor,
    *,
    cfg: _StaticSsdRank1ShapeConfig,
    ws: _StaticSsdRank1Workspace,
) -> torch.Tensor:
    """Static hot-path phase-3 forward: canonical layout and fixed precision knobs."""
    B, NC, C_CHUNK, H, M = C_5d.shape
    D = V_5d.shape[-1]
    BH = B * H
    if S0_chunk.shape != (BH, NC, M * D):
        raise ValueError(f"S0_chunk must be [BH,NC,MD]=({BH},{NC},{M * D}); got {tuple(S0_chunk.shape)}.")
    if M % cfg.phase3_forward.block_m != 0 or D % cfg.phase3_forward.block_d != 0:
        raise NotImplementedError(
            "Static phase-3 forward requires M and D divisible by configured blocks. "
            f"Got M={M}, D={D}, BLOCK_M={cfg.phase3_forward.block_m}, BLOCK_D={cfg.phase3_forward.block_d}."
        )
    S0_md = ws.phase3_s0_md_fwd
    S0_md.copy_(S0_chunk.reshape(BH, NC, M, D))
    out = torch.empty((BH, NC, C_CHUNK, D), device=C_5d.device, dtype=C_5d.dtype)
    grid = (BH * NC, D // cfg.phase3_forward.block_d)
    ssd_rank1_dense_output_fwd_kernel[grid](
        C_5d,
        W_5d,
        V_5d,
        log_alpha_4d,
        S0_md,
        out,
        out,
        B,
        H,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *C_5d.stride(),
        *W_5d.stride(),
        *V_5d.stride(),
        *log_alpha_4d.stride(),
        *S0_md.stride(),
        *out.stride(),
        *out.stride(),
        BLOCK_M=cfg.phase3_forward.block_m,
        BLOCK_D=cfg.phase3_forward.block_d,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        M_STATIC=M,
        INPUT_PRECISION=cfg.input_precision,
        WRITE_Y_OFF=False,
        num_warps=cfg.phase3_forward.num_warps,
        num_stages=cfg.phase3_forward.num_stages,
    )
    return out


def _ssd_rank1_dense_output_backward_impl_static(
    C_5d: torch.Tensor,
    W_5d: torch.Tensor,
    V_5d: torch.Tensor,
    log_alpha_4d: torch.Tensor,
    grad_out_c: torch.Tensor,
    S0_chunk: torch.Tensor,
    *,
    cfg: _StaticSsdRank1ShapeConfig,
    ws: _StaticSsdRank1Workspace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Static hot-path phase-3 backward: canonical layout only, fixed branch choices."""
    B, NC, C_CHUNK, H, M = C_5d.shape
    D = V_5d.shape[-1]
    BH = B * H
    if grad_out_c.shape != (BH, NC, C_CHUNK, D):
        raise ValueError(
            f"grad_out_c must be [BH,NC,C,D]=({BH},{NC},{C_CHUNK},{D}); got {tuple(grad_out_c.shape)}."
        )
    if S0_chunk.shape != (BH, NC, M * D):
        raise ValueError(f"S0_chunk must be [BH,NC,MD]=({BH},{NC},{M * D}); got {tuple(S0_chunk.shape)}.")
    if M % cfg.phase3_backward.block_m != 0 or D % cfg.phase3_backward.block_d != 0:
        raise NotImplementedError(
            "Static phase-3 backward requires M and D divisible by configured blocks. "
            f"Got M={M}, D={D}, BLOCK_M={cfg.phase3_backward.block_m}, BLOCK_D={cfg.phase3_backward.block_d}."
        )
    S0_md = ws.phase3_s0_md_bwd
    S0_md.copy_(S0_chunk.reshape(BH, NC, M, D))
    dV = torch.empty((BH, NC, C_CHUNK, D), device=C_5d.device, dtype=C_5d.dtype)
    dC = torch.empty((BH, NC, C_CHUNK, M), device=C_5d.device, dtype=C_5d.dtype)
    dW = torch.empty((BH, NC, C_CHUNK, M), device=C_5d.device, dtype=C_5d.dtype)
    dlog = ws.phase3_dlog_fp32
    dS0 = ws.phase3_dS0
    grid = (BH * NC,)
    phase3_bwd_num_warps = max(cfg.phase3_backward.fused_a1a2_num_warps, cfg.phase3_backward.fused_off_num_warps)
    ssd_rank1_dense_output_bwd_fused_kernel[grid](
        C_5d,
        W_5d,
        V_5d,
        log_alpha_4d,
        grad_out_c,
        S0_md,
        dV,
        dC,
        dW,
        dlog,
        dS0,
        B,
        H,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *C_5d.stride(),
        *W_5d.stride(),
        *V_5d.stride(),
        *log_alpha_4d.stride(),
        *grad_out_c.stride(),
        *S0_md.stride(),
        *dV.stride(),
        *dC.stride(),
        *dW.stride(),
        *dlog.stride(),
        *dS0.stride(),
        BLOCK_M=cfg.phase3_backward.block_m,
        BLOCK_D=cfg.phase3_backward.block_d,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        M_STATIC=M,
        D_STATIC=D,
        INPUT_PRECISION=cfg.input_precision,
        HAS_S0=True,
        num_warps=phase3_bwd_num_warps,
        num_stages=cfg.phase3_backward.num_stages,
    )
    return dC, dW, dV, dlog, dS0.reshape(BH, NC, M * D)


def _ssd_rank1_phase2_backward_static(
    grad_chunk_start_f: torch.Tensor,
    grad_final_f: torch.Tensor,
    S0_chunk: torch.Tensor,
    log_alpha_per_chunk_bnh: torch.Tensor,
    *,
    cfg: _StaticSsdRank1ShapeConfig,
    ws: _StaticSsdRank1Workspace,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Static hot-path phase-2 backward: fixed flags and canonical layout."""
    BH, NC, MD = grad_chunk_start_f.shape
    B, NC_log, H = log_alpha_per_chunk_bnh.shape
    if NC_log != NC:
        raise ValueError(f"log_alpha_per_chunk_bnh NC mismatch: got {NC_log}, expected {NC}.")
    if B * H != BH:
        raise ValueError(f"log_alpha_per_chunk_bnh B*H mismatch: got {B}*{H}, expected BH={BH}.")
    if grad_final_f.shape != (BH, MD):
        raise ValueError(f"grad_final_f must be [BH,MD]=({BH},{MD}); got {tuple(grad_final_f.shape)}.")
    if S0_chunk.shape != (BH, NC, MD):
        raise ValueError(f"S0_chunk must be [BH,NC,MD]=({BH},{NC},{MD}); got {tuple(S0_chunk.shape)}.")

    dS_local_end = grad_chunk_start_f
    d_log_per_chunk = ws.phase2_dlog_per_chunk
    d_init = ws.phase2_dinit
    phase2_cfg = cfg.phase2_launch
    if cfg.input_dtype == torch.float32:
        use_fp32_compute = True
    elif cfg.input_dtype == torch.bfloat16:
        use_fp32_compute = False
    else:
        raise NotImplementedError(
            "_ssd_rank1_phase2_backward_static only supports cfg.input_dtype in {torch.float32, torch.bfloat16}; "
            f"got {cfg.input_dtype}."
        )
    grid = (BH,)
    ssd_rank1_prefix_scan_bwd_dense_kernel[grid](
        grad_chunk_start_f,
        grad_final_f,
        S0_chunk,
        log_alpha_per_chunk_bnh,
        dS_local_end,
        d_log_per_chunk,
        d_init,
        B,
        H,
        BH,
        NC,
        MD,
        *grad_chunk_start_f.stride(),
        *grad_final_f.stride(),
        *S0_chunk.stride(),
        *log_alpha_per_chunk_bnh.stride(),
        *dS_local_end.stride(),
        *d_log_per_chunk.stride(),
        *d_init.stride(),
        BLOCK_NC=cfg.phase2_block_nc,
        BLOCK_MD=phase2_cfg.block_md,
        NC_STATIC=NC,
        USE_FP32_COMPUTE=use_fp32_compute,
        HAS_GRAD_FINAL=True,
        WRITE_D_INIT=True,
        num_warps=phase2_cfg.num_warps,
        num_stages=phase2_cfg.num_stages,
    )
    return dS_local_end, d_log_per_chunk, d_init


class SsdRank1TritonStatic(torch.autograd.Function):
    """Static hot-path autograd with shape-keyed fixed launch configs."""

    @staticmethod
    def forward(
        ctx,
        C: torch.Tensor,
        W: torch.Tensor,
        V: torch.Tensor,
        log_alpha: torch.Tensor,
        initial_state: torch.Tensor | None,
        CHUNK_SIZE: int,
        INPUT_PRECISION: str,
        RETURN_FINAL_STATE: bool,
    ):
        cfg = _validate_static_hot_path_contract(
            C,
            W,
            V,
            log_alpha,
            initial_state,
            CHUNK_SIZE,
            INPUT_PRECISION,
            RETURN_FINAL_STATE,
        )
        _ensure_triton_allocator()
        C_chunk, W_chunk, V_chunk, log_alpha_chunk, B, N, H, M, D, NC, BH = _ssd_rank1_prepare_unchunked_inputs_static(
            C,
            W,
            V,
            log_alpha,
            cfg=cfg,
        )
        ws = _get_static_workspace(device=C.device, cfg_key=(BH, N, M, D), cfg=cfg)
        s_local_end = _ssd_rank1_chunk_end_state_forward_impl_static(
            W_chunk,
            V_chunk,
            log_alpha_chunk,
            cfg=cfg,
            ws=ws,
        )
        log_alpha_per_chunk_bnh = log_alpha_chunk.sum(dim=2, dtype=torch.float32).contiguous()
        S1_chunk = torch.empty((BH, M * D), device=C.device, dtype=torch.float32)
        S0_chunk = _ssd_rank1_phase2_forward_static(
            s_local_end,
            log_alpha_per_chunk_bnh,
            S1_chunk,
            cfg=cfg,
            ws=ws,
        )
        y_chunk = _ssd_rank1_dense_output_forward_impl_static(
            C_chunk,
            W_chunk,
            V_chunk,
            log_alpha_chunk,
            S0_chunk,
            cfg=cfg,
            ws=ws,
        )

        ctx.save_for_backward(C, W, V, log_alpha)
        ctx.B = B
        ctx.N = N
        ctx.H = H
        ctx.M = M
        ctx.D = D
        ctx.NC = NC
        ctx.CHUNK_SIZE = cfg.chunk_size
        ctx.cfg_key = (B * H, N, M, D)
        return y_chunk, S1_chunk

    @staticmethod
    def backward(ctx, grad_y_chunk, grad_S1_chunk):
        C, W, V, log_alpha = ctx.saved_tensors
        B = ctx.B
        N = ctx.N
        H = ctx.H
        M = ctx.M
        D = ctx.D
        NC = ctx.NC
        CHUNK_SIZE = ctx.CHUNK_SIZE
        BH = B * H
        cfg = _lookup_static_ssd_rank1_shape_config(N=N, M=M, D=D)
        ws = _get_static_workspace(device=C.device, cfg_key=(BH, N, M, D), cfg=cfg)

        C_chunk, W_chunk, V_chunk, log_alpha_chunk, _, _, _, _, _, _, _ = _ssd_rank1_prepare_unchunked_inputs_static(
            C,
            W,
            V,
            log_alpha,
            cfg=cfg,
        )
        s_local_end = _ssd_rank1_chunk_end_state_forward_impl_static(
            W_chunk,
            V_chunk,
            log_alpha_chunk,
            cfg=cfg,
            ws=ws,
        )
        log_alpha_per_chunk_bnh = log_alpha_chunk.sum(dim=2, dtype=torch.float32).contiguous()
        final_replay = ws.phase2_final_replay
        S0_chunk = _ssd_rank1_phase2_forward_static(
            s_local_end,
            log_alpha_per_chunk_bnh,
            final_replay,
            cfg=cfg,
            ws=ws,
        )

        grad_y_in = grad_y_chunk if grad_y_chunk.is_contiguous() else grad_y_chunk.contiguous()
        dC_chunk, dW_chunk, dV_chunk, dlog_phase3_fp32, dS0 = _ssd_rank1_dense_output_backward_impl_static(
            C_chunk,
            W_chunk,
            V_chunk,
            log_alpha_chunk,
            grad_y_in,
            S0_chunk,
            cfg=cfg,
            ws=ws,
        )
        dlog_chunk = ws.dlog_chunk_accum
        dlog_chunk.copy_(dlog_phase3_fp32)
        grad_chunk_start_f = dS0 if dS0.is_contiguous() else dS0.contiguous()
        if grad_S1_chunk is None:
            grad_final_f = ws.phase2_grad_final_zero
            grad_final_f.zero_()
        else:
            grad_final_f = (
                grad_S1_chunk
                if grad_S1_chunk.dtype == torch.float32 and grad_S1_chunk.is_contiguous()
                else grad_S1_chunk.float().contiguous()
            )
        dS_local_end, d_log_per_chunk, _ = _ssd_rank1_phase2_backward_static(
            grad_chunk_start_f,
            grad_final_f,
            S0_chunk,
            log_alpha_per_chunk_bnh,
            cfg=cfg,
            ws=ws,
        )

        grad_s_md = dS_local_end.reshape(BH, NC, M, D)
        grid_phase1 = (BH * NC,)
        ssd_rank1_chunk_end_state_bwd_fused_kernel[grid_phase1](
            grad_s_md,
            W_chunk,
            V_chunk,
            log_alpha_chunk,
            dW_chunk,
            dV_chunk,
            dlog_chunk,
            B,
            H,
            BH,
            NC,
            CHUNK_SIZE,
            M,
            D,
            *grad_s_md.stride(),
            *W_chunk.stride(),
            *V_chunk.stride(),
            *log_alpha_chunk.stride(),
            *dW_chunk.stride(),
            *dV_chunk.stride(),
            *dlog_chunk.stride(),
            BLOCK_M=cfg.phase1_block_m,
            BLOCK_D=cfg.phase1_block_d,
            BLOCK_C=CHUNK_SIZE,
            C_STATIC=CHUNK_SIZE,
            M_STATIC=M,
            D_STATIC=D,
            INPUT_PRECISION=cfg.input_precision,
            ACCUMULATE=True,
            num_warps=cfg.phase1_backward.num_warps,
            num_stages=cfg.phase1_backward.num_stages,
        )

        dlog_chunk.add_(d_log_per_chunk.unsqueeze(-1).to(dlog_chunk.dtype))
        dC = _ssd_rank1_restore_grad_layout(dC_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        dW = _ssd_rank1_restore_grad_layout(dW_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        dV = _ssd_rank1_restore_grad_layout(dV_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        if not dlog_chunk.is_contiguous():
            raise ValueError("SsdRank1TritonStatic.backward expects contiguous dlog_chunk storage.")
        dlog_alpha = dlog_chunk.as_strided((B, N, H), (H * N, 1, N))
        return dC, dW, dV, dlog_alpha, None, None, None, None


def _ssd_rank4_prepare_unchunked_inputs_static(
    C: torch.Tensor,
    W1: torch.Tensor,
    V1: torch.Tensor,
    W2: torch.Tensor,
    V2: torch.Tensor,
    W3: torch.Tensor,
    V3: torch.Tensor,
    W4: torch.Tensor,
    V4: torch.Tensor,
    log_alpha: torch.Tensor,
    *,
    cfg: _StaticSsdRank1ShapeConfig,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
]:
    C_chunk, W1_chunk, V1_chunk, log_alpha_chunk, B, N, H, M, D, NC, BH = _ssd_rank1_prepare_unchunked_inputs_static(
        C,
        W1,
        V1,
        log_alpha,
        cfg=cfg,
    )
    CHUNK_SIZE = cfg.chunk_size
    W2_chunk = W2.view(B, NC, CHUNK_SIZE, H, M)
    V2_chunk = V2.view(B, NC, CHUNK_SIZE, H, D)
    W3_chunk = W3.view(B, NC, CHUNK_SIZE, H, M)
    V3_chunk = V3.view(B, NC, CHUNK_SIZE, H, D)
    W4_chunk = W4.view(B, NC, CHUNK_SIZE, H, M)
    V4_chunk = V4.view(B, NC, CHUNK_SIZE, H, D)
    return C_chunk, W1_chunk, V1_chunk, W2_chunk, V2_chunk, W3_chunk, V3_chunk, W4_chunk, V4_chunk, log_alpha_chunk, B, N, H, M, D, NC, BH


def _ssd_rank4_chunk_end_state_forward_impl_static(
    W1_5d: torch.Tensor,
    V1_5d: torch.Tensor,
    W2_5d: torch.Tensor,
    V2_5d: torch.Tensor,
    W3_5d: torch.Tensor,
    V3_5d: torch.Tensor,
    W4_5d: torch.Tensor,
    V4_5d: torch.Tensor,
    log_alpha_4d: torch.Tensor,
    *,
    cfg: _StaticSsdRank1ShapeConfig,
    ws: _StaticSsdRank1Workspace,
    rank: int,
) -> torch.Tensor:
    B, NC, C_CHUNK, H, M = W1_5d.shape
    D = V1_5d.shape[-1]
    BH = B * H
    s_local_end_md = ws.s_local_end_md
    grid = (BH * NC, M // cfg.phase1_block_m, D // cfg.phase1_block_d)
    ssd_rank4_chunk_end_state_fwd_kernel[grid](
        W1_5d,
        V1_5d,
        W2_5d,
        V2_5d,
        W3_5d,
        V3_5d,
        W4_5d,
        V4_5d,
        log_alpha_4d,
        s_local_end_md,
        B,
        H,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *W1_5d.stride(),
        *V1_5d.stride(),
        *log_alpha_4d.stride(),
        *s_local_end_md.stride(),
        BLOCK_M=cfg.phase1_block_m,
        BLOCK_D=cfg.phase1_block_d,
        BLOCK_T=cfg.phase1_block_t,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        INPUT_PRECISION=cfg.input_precision,
        RANK=rank,
    )
    return s_local_end_md.reshape(BH, NC, M * D)


def _ssd_rank4_dense_output_forward_impl_static(
    C_5d: torch.Tensor,
    W1_5d: torch.Tensor,
    V1_5d: torch.Tensor,
    W2_5d: torch.Tensor,
    V2_5d: torch.Tensor,
    W3_5d: torch.Tensor,
    V3_5d: torch.Tensor,
    W4_5d: torch.Tensor,
    V4_5d: torch.Tensor,
    log_alpha_4d: torch.Tensor,
    S0_chunk: torch.Tensor,
    *,
    cfg: _StaticSsdRank1ShapeConfig,
    ws: _StaticSsdRank1Workspace,
    rank: int,
) -> torch.Tensor:
    B, NC, C_CHUNK, H, M = C_5d.shape
    D = V1_5d.shape[-1]
    BH = B * H
    S0_md = S0_chunk.reshape(BH, NC, M, D)
    out = torch.empty((BH, NC, C_CHUNK, D), device=C_5d.device, dtype=C_5d.dtype)
    grid = (BH * NC, D // cfg.phase3_forward.block_d)
    ssd_rank4_dense_output_fwd_kernel[grid](
        C_5d,
        W1_5d,
        V1_5d,
        W2_5d,
        V2_5d,
        W3_5d,
        V3_5d,
        W4_5d,
        V4_5d,
        log_alpha_4d,
        S0_md,
        out,
        out,
        B,
        H,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *C_5d.stride(),
        *W1_5d.stride(),
        *V1_5d.stride(),
        *log_alpha_4d.stride(),
        *S0_md.stride(),
        *out.stride(),
        *out.stride(),
        BLOCK_M=cfg.phase3_forward.block_m,
        BLOCK_D=cfg.phase3_forward.block_d,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        M_STATIC=M,
        INPUT_PRECISION=cfg.input_precision,
        WRITE_Y_OFF=False,
        RANK=rank,
        num_warps=cfg.phase3_forward.num_warps,
        num_stages=cfg.phase3_forward.num_stages,
    )
    return out


def _ssd_rank4_dense_output_backward_impl_static(
    C_5d: torch.Tensor,
    W1_5d: torch.Tensor,
    V1_5d: torch.Tensor,
    W2_5d: torch.Tensor,
    V2_5d: torch.Tensor,
    W3_5d: torch.Tensor,
    V3_5d: torch.Tensor,
    W4_5d: torch.Tensor,
    V4_5d: torch.Tensor,
    log_alpha_4d: torch.Tensor,
    grad_out_c: torch.Tensor,
    S0_chunk: torch.Tensor,
    *,
    cfg: _StaticSsdRank1ShapeConfig,
    ws: _StaticSsdRank1Workspace,
    rank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    B, NC, C_CHUNK, H, M = C_5d.shape
    D = V1_5d.shape[-1]
    BH = B * H
    S0_md = S0_chunk.reshape(BH, NC, M, D)
    dV1 = torch.empty((BH, NC, C_CHUNK, D), device=C_5d.device, dtype=C_5d.dtype)
    dC = torch.empty((BH, NC, C_CHUNK, M), device=C_5d.device, dtype=C_5d.dtype)
    dW1 = torch.empty((BH, NC, C_CHUNK, M), device=C_5d.device, dtype=C_5d.dtype)
    if rank >= 2:
        dV2 = torch.empty((BH, NC, C_CHUNK, D), device=C_5d.device, dtype=C_5d.dtype)
        dW2 = torch.empty((BH, NC, C_CHUNK, M), device=C_5d.device, dtype=C_5d.dtype)
    else:
        dV2 = dV1
        dW2 = dW1
    if rank >= 3:
        dV3 = torch.empty((BH, NC, C_CHUNK, D), device=C_5d.device, dtype=C_5d.dtype)
        dW3 = torch.empty((BH, NC, C_CHUNK, M), device=C_5d.device, dtype=C_5d.dtype)
    else:
        dV3 = dV1
        dW3 = dW1
    if rank >= 4:
        dV4 = torch.empty((BH, NC, C_CHUNK, D), device=C_5d.device, dtype=C_5d.dtype)
        dW4 = torch.empty((BH, NC, C_CHUNK, M), device=C_5d.device, dtype=C_5d.dtype)
    else:
        dV4 = dV1
        dW4 = dW1
    dlog = ws.phase3_dlog_fp32
    dS0 = ws.phase3_dS0
    grid = (BH * NC,)
    ssd_rank4_dense_output_bwd_fused_kernel[grid](
        C_5d,
        W1_5d,
        V1_5d,
        W2_5d,
        V2_5d,
        W3_5d,
        V3_5d,
        W4_5d,
        V4_5d,
        log_alpha_4d,
        grad_out_c,
        S0_md,
        dV1,
        dC,
        dW1,
        dV2,
        dW2,
        dV3,
        dW3,
        dV4,
        dW4,
        dlog,
        dS0,
        B,
        H,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *C_5d.stride(),
        *W1_5d.stride(),
        *V1_5d.stride(),
        *log_alpha_4d.stride(),
        *grad_out_c.stride(),
        *S0_md.stride(),
        *dV1.stride(),
        *dC.stride(),
        *dW1.stride(),
        *dlog.stride(),
        *dS0.stride(),
        BLOCK_M=cfg.phase3_backward.block_m,
        BLOCK_D=cfg.phase3_backward.block_d,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        M_STATIC=M,
        D_STATIC=D,
        INPUT_PRECISION=cfg.input_precision,
        HAS_S0=False,
        RANK=rank,
        num_warps=cfg.phase3_backward.fused_dv_num_warps,
        num_stages=cfg.phase3_backward.fused_dv_num_stages,
        COMPUTE_DV=True,
        COMPUTE_MAIN=False,
    )
    ssd_rank4_dense_output_bwd_fused_kernel[grid](
        C_5d,
        W1_5d,
        V1_5d,
        W2_5d,
        V2_5d,
        W3_5d,
        V3_5d,
        W4_5d,
        V4_5d,
        log_alpha_4d,
        grad_out_c,
        S0_md,
        dV1,
        dC,
        dW1,
        dV2,
        dW2,
        dV3,
        dW3,
        dV4,
        dW4,
        dlog,
        dS0,
        B,
        H,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *C_5d.stride(),
        *W1_5d.stride(),
        *V1_5d.stride(),
        *log_alpha_4d.stride(),
        *grad_out_c.stride(),
        *S0_md.stride(),
        *dV1.stride(),
        *dC.stride(),
        *dW1.stride(),
        *dlog.stride(),
        *dS0.stride(),
        BLOCK_M=cfg.phase3_backward.block_m,
        BLOCK_D=cfg.phase3_backward.block_d,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        M_STATIC=M,
        D_STATIC=D,
        INPUT_PRECISION=cfg.input_precision,
        HAS_S0=False,
        RANK=rank,
        num_warps=cfg.phase3_backward.fused_main_num_warps,
        num_stages=cfg.phase3_backward.fused_main_num_stages,
        COMPUTE_DV=False,
        COMPUTE_MAIN=True,
    )
    ssd_rank4_dense_output_bwd_off_kernel[grid](
        C_5d,
        log_alpha_4d,
        grad_out_c,
        S0_md,
        dC,
        dlog,
        dS0,
        B,
        H,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *C_5d.stride(),
        *log_alpha_4d.stride(),
        *grad_out_c.stride(),
        *S0_md.stride(),
        *dC.stride(),
        *dlog.stride(),
        *dS0.stride(),
        BLOCK_M=cfg.phase3_backward.block_m,
        BLOCK_D=cfg.phase3_backward.block_d,
        BLOCK_C=C_CHUNK,
        M_STATIC=M,
        D_STATIC=D,
        INPUT_PRECISION=cfg.input_precision,
        num_warps=cfg.phase3_backward.fused_off_num_warps,
        num_stages=cfg.phase3_backward.fused_off_num_stages,
    )
    return dC, dW1, dV1, dW2, dV2, dW3, dV3, dW4, dV4, dlog, dS0.reshape(BH, NC, M * D)


def _ssd_rank4_chunk_end_state_backward_impl_static(
    grad_s_md: torch.Tensor,
    W1_5d: torch.Tensor,
    V1_5d: torch.Tensor,
    W2_5d: torch.Tensor,
    V2_5d: torch.Tensor,
    W3_5d: torch.Tensor,
    V3_5d: torch.Tensor,
    W4_5d: torch.Tensor,
    V4_5d: torch.Tensor,
    log_alpha_4d: torch.Tensor,
    dW1_chunk: torch.Tensor,
    dV1_chunk: torch.Tensor,
    dW2_chunk: torch.Tensor,
    dV2_chunk: torch.Tensor,
    dW3_chunk: torch.Tensor,
    dV3_chunk: torch.Tensor,
    dW4_chunk: torch.Tensor,
    dV4_chunk: torch.Tensor,
    dlog_chunk: torch.Tensor,
    *,
    cfg: _StaticSsdRank1ShapeConfig,
    rank: int,
) -> None:
    B, NC, C_CHUNK, H, M = W1_5d.shape
    D = V1_5d.shape[-1]
    BH = B * H
    grid = (BH * NC,)
    ssd_rank4_chunk_end_state_bwd_fused_kernel[grid](
        grad_s_md,
        W1_5d,
        V1_5d,
        W2_5d,
        V2_5d,
        W3_5d,
        V3_5d,
        W4_5d,
        V4_5d,
        log_alpha_4d,
        dW1_chunk,
        dV1_chunk,
        dW2_chunk,
        dV2_chunk,
        dW3_chunk,
        dV3_chunk,
        dW4_chunk,
        dV4_chunk,
        dlog_chunk,
        B,
        H,
        BH,
        NC,
        C_CHUNK,
        M,
        D,
        *grad_s_md.stride(),
        *W1_5d.stride(),
        *V1_5d.stride(),
        *log_alpha_4d.stride(),
        *dW1_chunk.stride(),
        *dV1_chunk.stride(),
        *dlog_chunk.stride(),
        BLOCK_M=cfg.phase1_block_m,
        BLOCK_D=cfg.phase1_block_d,
        BLOCK_C=C_CHUNK,
        C_STATIC=C_CHUNK,
        M_STATIC=M,
        D_STATIC=D,
        INPUT_PRECISION=cfg.input_precision,
        ACCUMULATE=True,
        RANK=rank,
        num_warps=cfg.phase1_backward.num_warps,
        num_stages=cfg.phase1_backward.num_stages,
    )


class SsdRank4TritonStatic(torch.autograd.Function):
    """Static hot-path autograd for rank-2/3/4 SSD with fused phase-1/3 kernels."""

    @staticmethod
    def forward(
        ctx,
        C: torch.Tensor,
        W1: torch.Tensor,
        V1: torch.Tensor,
        W2: torch.Tensor,
        V2: torch.Tensor,
        W3: torch.Tensor,
        V3: torch.Tensor,
        W4: torch.Tensor,
        V4: torch.Tensor,
        log_alpha: torch.Tensor,
        initial_state: torch.Tensor | None,
        CHUNK_SIZE: int,
        INPUT_PRECISION: str,
        RETURN_FINAL_STATE: bool,
        RANK: int,
    ):
        cfg = _validate_static_hot_path_contract(
            C,
            W1,
            V1,
            log_alpha,
            initial_state,
            CHUNK_SIZE,
            INPUT_PRECISION,
            RETURN_FINAL_STATE,
            RANK=RANK,
        )
        _ensure_triton_allocator()
        (
            C_chunk,
            W1_chunk,
            V1_chunk,
            W2_chunk,
            V2_chunk,
            W3_chunk,
            V3_chunk,
            W4_chunk,
            V4_chunk,
            log_alpha_chunk,
            B,
            N,
            H,
            M,
            D,
            NC,
            BH,
        ) = _ssd_rank4_prepare_unchunked_inputs_static(C, W1, V1, W2, V2, W3, V3, W4, V4, log_alpha, cfg=cfg)
        ws = _get_static_workspace(device=C.device, cfg_key=(BH, N, M, D), cfg=cfg, allocate_phase3_s0=False)
        s_local_end = _ssd_rank4_chunk_end_state_forward_impl_static(
            W1_chunk,
            V1_chunk,
            W2_chunk,
            V2_chunk,
            W3_chunk,
            V3_chunk,
            W4_chunk,
            V4_chunk,
            log_alpha_chunk,
            cfg=cfg,
            ws=ws,
            rank=RANK,
        )
        log_alpha_per_chunk_bnh = log_alpha_chunk.sum(dim=2, dtype=torch.float32).contiguous()
        S1_chunk = torch.empty((BH, M * D), device=C.device, dtype=torch.float32)
        S0_chunk = _ssd_rank1_phase2_forward_static(
            s_local_end,
            log_alpha_per_chunk_bnh,
            S1_chunk,
            cfg=cfg,
            ws=ws,
        )
        y_chunk = _ssd_rank4_dense_output_forward_impl_static(
            C_chunk,
            W1_chunk,
            V1_chunk,
            W2_chunk,
            V2_chunk,
            W3_chunk,
            V3_chunk,
            W4_chunk,
            V4_chunk,
            log_alpha_chunk,
            S0_chunk,
            cfg=cfg,
            ws=ws,
            rank=RANK,
        )

        ctx.save_for_backward(C, W1, V1, W2, V2, W3, V3, W4, V4, log_alpha)
        ctx.B = B
        ctx.N = N
        ctx.H = H
        ctx.M = M
        ctx.D = D
        ctx.NC = NC
        ctx.CHUNK_SIZE = cfg.chunk_size
        ctx.rank = int(RANK)
        return y_chunk, S1_chunk

    @staticmethod
    def backward(ctx, grad_y_chunk, grad_S1_chunk):
        C, W1, V1, W2, V2, W3, V3, W4, V4, log_alpha = ctx.saved_tensors
        B, N, H, M, D, NC = ctx.B, ctx.N, ctx.H, ctx.M, ctx.D, ctx.NC
        CHUNK_SIZE = ctx.CHUNK_SIZE
        BH = B * H
        rank = ctx.rank
        cfg = _lookup_static_ssd_shape_rank_config(N=N, M=M, D=D, RANK=rank)
        ws = _get_static_workspace(device=C.device, cfg_key=(BH, N, M, D), cfg=cfg, allocate_phase3_s0=False)

        (
            C_chunk,
            W1_chunk,
            V1_chunk,
            W2_chunk,
            V2_chunk,
            W3_chunk,
            V3_chunk,
            W4_chunk,
            V4_chunk,
            log_alpha_chunk,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = _ssd_rank4_prepare_unchunked_inputs_static(C, W1, V1, W2, V2, W3, V3, W4, V4, log_alpha, cfg=cfg)

        s_local_end = _ssd_rank4_chunk_end_state_forward_impl_static(
            W1_chunk,
            V1_chunk,
            W2_chunk,
            V2_chunk,
            W3_chunk,
            V3_chunk,
            W4_chunk,
            V4_chunk,
            log_alpha_chunk,
            cfg=cfg,
            ws=ws,
            rank=rank,
        )
        log_alpha_per_chunk_bnh = log_alpha_chunk.sum(dim=2, dtype=torch.float32).contiguous()
        final_replay = ws.phase2_final_replay
        S0_chunk = _ssd_rank1_phase2_forward_static(
            s_local_end,
            log_alpha_per_chunk_bnh,
            final_replay,
            cfg=cfg,
            ws=ws,
        )

        grad_y_in = grad_y_chunk if grad_y_chunk.is_contiguous() else grad_y_chunk.contiguous()
        dC_chunk, dW1_chunk, dV1_chunk, dW2_chunk, dV2_chunk, dW3_chunk, dV3_chunk, dW4_chunk, dV4_chunk, dlog_phase3_fp32, dS0 = (
            _ssd_rank4_dense_output_backward_impl_static(
                C_chunk,
                W1_chunk,
                V1_chunk,
                W2_chunk,
                V2_chunk,
                W3_chunk,
                V3_chunk,
                W4_chunk,
                V4_chunk,
                log_alpha_chunk,
                grad_y_in,
                S0_chunk,
                cfg=cfg,
                ws=ws,
                rank=rank,
            )
        )
        dlog_chunk = ws.dlog_chunk_accum
        dlog_chunk.copy_(dlog_phase3_fp32)

        grad_chunk_start_f = dS0 if dS0.is_contiguous() else dS0.contiguous()
        if grad_S1_chunk is None:
            grad_final_f = ws.phase2_grad_final_zero
            grad_final_f.zero_()
        else:
            grad_final_f = (
                grad_S1_chunk
                if grad_S1_chunk.dtype == torch.float32 and grad_S1_chunk.is_contiguous()
                else grad_S1_chunk.float().contiguous()
            )
        dS_local_end, d_log_per_chunk, _ = _ssd_rank1_phase2_backward_static(
            grad_chunk_start_f,
            grad_final_f,
            S0_chunk,
            log_alpha_per_chunk_bnh,
            cfg=cfg,
            ws=ws,
        )

        grad_s_md = dS_local_end.reshape(BH, NC, M, D)
        _ssd_rank4_chunk_end_state_backward_impl_static(
            grad_s_md,
            W1_chunk,
            V1_chunk,
            W2_chunk,
            V2_chunk,
            W3_chunk,
            V3_chunk,
            W4_chunk,
            V4_chunk,
            log_alpha_chunk,
            dW1_chunk,
            dV1_chunk,
            dW2_chunk,
            dV2_chunk,
            dW3_chunk,
            dV3_chunk,
            dW4_chunk,
            dV4_chunk,
            dlog_chunk,
            cfg=cfg,
            rank=rank,
        )
        dlog_chunk.add_(d_log_per_chunk.unsqueeze(-1).to(dlog_chunk.dtype))

        dC = _ssd_rank1_restore_grad_layout(dC_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        dW1 = _ssd_rank1_restore_grad_layout(dW1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        dV1 = _ssd_rank1_restore_grad_layout(dV1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)
        dW2 = _ssd_rank1_restore_grad_layout(dW2_chunk, B=B, N=N, H=H, C=CHUNK_SIZE) if rank >= 2 else None
        dV2 = _ssd_rank1_restore_grad_layout(dV2_chunk, B=B, N=N, H=H, C=CHUNK_SIZE) if rank >= 2 else None
        dW3 = _ssd_rank1_restore_grad_layout(dW3_chunk, B=B, N=N, H=H, C=CHUNK_SIZE) if rank >= 3 else None
        dV3 = _ssd_rank1_restore_grad_layout(dV3_chunk, B=B, N=N, H=H, C=CHUNK_SIZE) if rank >= 3 else None
        dW4 = _ssd_rank1_restore_grad_layout(dW4_chunk, B=B, N=N, H=H, C=CHUNK_SIZE) if rank >= 4 else None
        dV4 = _ssd_rank1_restore_grad_layout(dV4_chunk, B=B, N=N, H=H, C=CHUNK_SIZE) if rank >= 4 else None
        dlog_alpha = dlog_chunk.as_strided((B, N, H), (H * N, 1, N))
        return dC, dW1, dV1, dW2, dV2, dW3, dV3, dW4, dV4, dlog_alpha, None, None, None, None, None, None, None


def ssd_rank1_triton_debug(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
    INPUT_PRECISION: str = "tf32",
    RETURN_FINAL_STATE: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Debug/dynamic entrypoint retaining heuristic/autograd-flexible behavior."""
    if CHUNK_SIZE is None:
        B0, N0, H0, M0 = C.shape
        D0 = V.shape[-1]
        CHUNK_SIZE = _select_chunk_size_heuristic(N=N0, M=M0, D=D0, BH=B0 * H0)
    if RETURN_FINAL_STATE:
        y_chunk, S1_chunk = SsdRank1TritonDebug.apply(
            C, W, V, log_alpha, initial_state, CHUNK_SIZE, INPUT_PRECISION, RETURN_FINAL_STATE,
        )
    else:
        y_chunk = SsdRank1TritonDebug.apply(
            C, W, V, log_alpha, initial_state, CHUNK_SIZE, INPUT_PRECISION, RETURN_FINAL_STATE,
        )
        S1_chunk = None
    B = C.shape[0]
    N = C.shape[1]
    H = C.shape[2]
    return _ssd_rank1_restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=CHUNK_SIZE)


def warmup_ssd_rank1_triton_static(
    *,
    shape_keys: list[tuple[int, int, int]] | None = None,
    device: torch.device | None = None,
    include_backward: bool = True,
) -> None:
    """Compile and warm static SSD rank-1 kernels for configured shapes."""
    if device is None:
        if not torch.cuda.is_available():
            raise NotImplementedError("warmup_ssd_rank1_triton_static requires CUDA.")
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        raise NotImplementedError("warmup_ssd_rank1_triton_static requires a CUDA device.")
    _ensure_triton_allocator()
    keys = sorted(_STATIC_SSD_RANK1_SHAPE_CONFIGS.keys()) if shape_keys is None else shape_keys
    for key in keys:
        N, M, D = key
        cfg = _lookup_static_ssd_rank1_shape_config(N=N, M=M, D=D)
        B = 32
        H = 16
        C = torch.zeros((B, N, H, M), device=device, dtype=cfg.input_dtype)
        W = torch.zeros((B, N, H, M), device=device, dtype=cfg.input_dtype)
        V = torch.zeros((B, N, H, D), device=device, dtype=cfg.input_dtype)
        log_alpha = torch.full((B, N, H), -1.0, device=device, dtype=cfg.input_dtype)
        set_ssd_rank1_static_shape(N=N, M=M, D=D)
        y, s = ssd_rank1_triton(
            C,
            W,
            V,
            log_alpha,
            None,
            cfg.chunk_size,
            cfg.input_precision,
            cfg.return_final_state,
        )
        if include_backward:
            Cg = C.detach().clone().requires_grad_(True)
            Wg = W.detach().clone().requires_grad_(True)
            Vg = V.detach().clone().requires_grad_(True)
            Lg = log_alpha.detach().clone().requires_grad_(True)
            yg, sg = ssd_rank1_triton(
                Cg,
                Wg,
                Vg,
                Lg,
                None,
                cfg.chunk_size,
                cfg.input_precision,
                cfg.return_final_state,
            )
            loss = torch.sum(yg, dtype=torch.float32) + torch.sum(sg, dtype=torch.float32)
            loss.backward()
    torch.cuda.synchronize(device)


def ssd_rank1_triton(
    C: torch.Tensor,
    W: torch.Tensor,
    V: torch.Tensor,
    log_alpha: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    CHUNK_SIZE: int | None = None,
    INPUT_PRECISION: str = "tf32",
    RETURN_FINAL_STATE: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """SSD rank-1 Triton entrypoint for static hot-path execution."""
    cfg = _validate_static_hot_path_contract(
        C,
        W,
        V,
        log_alpha,
        initial_state,
        CHUNK_SIZE,
        INPUT_PRECISION,
        RETURN_FINAL_STATE,
    )
    y_chunk, S1_chunk = SsdRank1TritonStatic.apply(
        C,
        W,
        V,
        log_alpha,
        initial_state,
        cfg.chunk_size,
        cfg.input_precision,
        cfg.return_final_state,
    )

    B = C.shape[0]
    N = C.shape[1]
    H = C.shape[2]
    return _ssd_rank1_restore_output_layout(y_chunk, S1_chunk, B=B, N=N, H=H, C=cfg.chunk_size)


# --------------------------------------------------------------------------------------------------
# END PHASE 123: FULL PARALLEL SCAN
# --------------------------------------------------------------------------------------------------


__all__ = [
    "ssd_rank4_chunk_end_state_reference",
    "ssd_rank4_dense_output_reference",
    "ssd_rank4_decode_pytorch",
    "ssd_rank4_decode_triton",
    "ssd_rank4_pytorch",
    "ssd_rank4_triton",
    "ssd_rank4_token_loop_oracle",
    "ssd_rank1_pytorch",
    "ssd_rank1_triton_debug",
    "ssd_rank1_triton",
    "set_ssd_rank1_static_shape",
    "clear_ssd_rank1_static_shape",
    "warmup_ssd_rank1_triton_static",
]

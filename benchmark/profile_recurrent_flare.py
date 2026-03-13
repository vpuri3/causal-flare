#!/usr/bin/env python
"""Profile RecurrentFLARE per-kernel timings, occupancy, and register pressure.

Forward kernels:
  - flare_recurrent_decode_lse_kernel  (decode-side logsumexp)
  - flare_recurrent_fwd_kernel         (main recurrent scan)

Backward ("batchward") kernels:
  - flare_recurrent_bwd_dg_part_kernel        (dg accumulation scan)
  - flare_recurrent_bwd_dscore_decode_kernel  (decode score gradient)
  - flare_recurrent_bwd_dsz_separate_kernel   (dV + dS_enc, O(T^2) per program)
  - flare_chunk_bwd_recurrent_qk             (dQ / dK from score gradients)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import json

import torch
import triton
import triton.testing
from triton.runtime.driver import driver

from causal_flare._common import _get_input_precision, _resolve_attn_scale
from causal_flare.autoregressive.recurrent import (
    _get_recurrent_block_d_k,
    _get_recurrent_block_m,
    _get_recurrent_block_t,
    flare_chunk_bwd_recurrent_qk,
    flare_recurrent_bwd_dg_part_kernel,
    flare_recurrent_bwd_dscore_decode_kernel,
    flare_recurrent_bwd_dsz_separate_kernel,
    flare_recurrent_decode_lse_kernel,
    flare_recurrent_fwd_kernel,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class KernelProfile:
    name: str
    grid: tuple
    num_warps: int
    num_stages: int
    regs_per_thread: int
    spills: int
    shared_bytes: int
    threads_per_cta: int
    occupancy_pct_est: float
    active_ctas_per_sm_est: int
    limiting_factors: list
    avg_ms: float


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device_props(device_idx: int) -> dict:
    tp = driver.active.utils.get_device_properties(device_idx)
    return {
        "max_shared_mem": int(tp["max_shared_mem"]),
        "max_num_regs": int(tp["max_num_regs"]),
        "warp_size": int(tp["warpSize"]),
        "max_threads_per_sm": int(
            torch.cuda.get_device_properties(device_idx).max_threads_per_multi_processor
        ),
    }


def estimate_occupancy(compiled, props: dict) -> tuple[float, int, list[str]]:
    meta = compiled.metadata
    threads = meta.num_warps * props["warp_size"]
    regs_cta = compiled.n_regs * threads
    smem_cta = meta.shared
    max_warps = props["max_threads_per_sm"] // props["warp_size"]
    limits = {
        "arch":    32,
        "threads": props["max_threads_per_sm"] // threads,
        "warps":   max_warps // meta.num_warps,
        "regs":    props["max_num_regs"] // regs_cta if regs_cta else 32,
        "shared":  props["max_shared_mem"] // smem_cta if smem_cta else 32,
    }
    active = max(1, min(limits.values()))
    occ = active * meta.num_warps / max_warps * 100.0
    limiting = [k for k, v in limits.items() if v == active]
    return occ, active, limiting


def make_profile(
    name: str,
    grid: tuple,
    compiled,
    props: dict,
    avg_ms: float,
) -> KernelProfile:
    occ, ctas, limiting = estimate_occupancy(compiled, props)
    meta = compiled.metadata
    return KernelProfile(
        name=name,
        grid=grid,
        num_warps=int(meta.num_warps),
        num_stages=int(meta.num_stages),
        regs_per_thread=int(compiled.n_regs),
        spills=int(compiled.n_spills),
        shared_bytes=int(meta.shared),
        threads_per_cta=int(meta.num_warps * props["warp_size"]),
        occupancy_pct_est=round(occ, 2),
        active_ctas_per_sm_est=ctas,
        limiting_factors=limiting,
        avg_ms=avg_ms,
    )


# ---------------------------------------------------------------------------
# Core profiling logic
# ---------------------------------------------------------------------------

def profile_recurrent_flare(
    B: int,
    H: int,
    N: int,
    D: int,
    M: int,
    *,
    dtype: torch.dtype,
    warmup: int = 25,
    iterations: int = 50,
    seed: int = 0,
    use_assoc_scan: bool = False,
) -> dict:
    torch.manual_seed(seed)
    device = torch.device("cuda")
    device_idx = device.index or 0
    props = get_device_props(device_idx)

    # ── inputs ──────────────────────────────────────────────────────────────
    Q = torch.randn(H, M, D, device=device, dtype=dtype, requires_grad=True)
    K = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)
    V = torch.randn(B, N, H, D, device=device, dtype=dtype, requires_grad=True)

    # ── heuristics ──────────────────────────────────────────────────────────
    scale = _resolve_attn_scale(None, D)
    block_m = _get_recurrent_block_m(M)
    block_d, block_k = _get_recurrent_block_d_k(D)
    block_t = _get_recurrent_block_t()
    ip = _get_input_precision()
    BH = B * H
    num_m_tiles = triton.cdiv(M, block_m)
    single_m = num_m_tiles == 1
    num_warps = 4 if block_m <= 64 else 8
    bwd_block_t = 16

    # ── permuted / expanded views used in forward ────────────────────────────
    K_bhtd = K.permute(0, 2, 1, 3).contiguous()    # [B,H,N,D]
    V_bhtd = V.permute(0, 2, 1, 3).contiguous()    # [B,H,N,D]
    Q_dec_bhtd = K_bhtd                             # weight-sharing
    K_dec_bhmd = Q.unsqueeze(0).expand(B, -1, -1, -1)  # [B,H,M,D] stride-0 on B

    # ── forward output buffers ───────────────────────────────────────────────
    if single_m:
        Y = torch.empty((B, H, N, D), device=device, dtype=dtype)
    else:
        Y = torch.zeros((B, H, N, D), device=device, dtype=dtype)
    LSE_enc = torch.empty((B, H, N, M), device=device, dtype=torch.float32)
    LSE_dec = torch.empty((B, H, N),    device=device, dtype=torch.float32)

    # ── backward buffers ─────────────────────────────────────────────────────
    dO     = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    dg_buf = torch.zeros((BH, N, M), device=device, dtype=torch.float32)
    dS_enc = torch.zeros((BH, N, M), device=device, dtype=torch.float32)
    dS_dec = dS_enc   # aliased (weight sharing, use_shared_decode_aliases=True)
    dV_bhtd = torch.zeros((BH, N, D), device=device, dtype=torch.float32)
    dQ     = torch.zeros((H, M, D),   device=device, dtype=torch.float32)
    dK     = torch.zeros((B, N, H, D), device=device, dtype=torch.float32)

    # With weight sharing: Q_dec = K (original [B,N,H,D]), K_dec = Q ([H,M,D])
    Q_dec_bt_hd = K    # [B,N,H,D]
    K_dec_hmd   = Q    # [H,M,D]

    dS_enc_qk = dS_enc.unsqueeze(1)  # [BH,1,N,M]

    # ── grid shapes ──────────────────────────────────────────────────────────
    g_fwd_lse   = (BH, triton.cdiv(N, 16))
    g_fwd_main  = (BH, triton.cdiv(D, block_d), num_m_tiles)
    g_bwd_dg    = (BH, triton.cdiv(D, block_d))
    g_bwd_dsc   = (BH, triton.cdiv(N, bwd_block_t))
    g_bwd_dsz   = (BH, N)
    g_bwd_qk    = (BH, 1)

    # =========================================================================
    # COMPILE (single warm-up launch; captures compiled kernel object)
    # =========================================================================

    def _fwd_lse():
        return flare_recurrent_decode_lse_kernel[g_fwd_lse](
            Q_dec_bhtd, K_dec_bhmd, LSE_dec,
            Q_dec_bhtd.stride(0), Q_dec_bhtd.stride(1),
            Q_dec_bhtd.stride(2), Q_dec_bhtd.stride(3),
            K_dec_bhmd.stride(0), K_dec_bhmd.stride(1),
            K_dec_bhmd.stride(2), K_dec_bhmd.stride(3),
            LSE_dec.stride(0), LSE_dec.stride(1), LSE_dec.stride(2),
            B, H, M, N, scale,
            D=D, BLOCK_T=16, BLOCK_M=block_m, BLOCK_K=block_k,
            INPUT_PRECISION=ip,
            num_warps=num_warps, num_stages=2,
        )

    def _fwd_main():
        return flare_recurrent_fwd_kernel[g_fwd_main](
            Q, K_bhtd, V_bhtd, Q_dec_bhtd, K_dec_bhmd, LSE_dec, Y, LSE_enc,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K_bhtd.stride(0), K_bhtd.stride(1), K_bhtd.stride(2), K_bhtd.stride(3),
            V_bhtd.stride(0), V_bhtd.stride(1), V_bhtd.stride(2), V_bhtd.stride(3),
            Q_dec_bhtd.stride(0), Q_dec_bhtd.stride(1),
            Q_dec_bhtd.stride(2), Q_dec_bhtd.stride(3),
            K_dec_bhmd.stride(0), K_dec_bhmd.stride(1),
            K_dec_bhmd.stride(2), K_dec_bhmd.stride(3),
            LSE_dec.stride(0), LSE_dec.stride(1), LSE_dec.stride(2),
            Y.stride(0), Y.stride(1), Y.stride(2), Y.stride(3),
            LSE_enc.stride(0), LSE_enc.stride(1), LSE_enc.stride(2), LSE_enc.stride(3),
            B, H, M, N, scale,
            D=D, BLOCK_M=block_m, BLOCK_D=block_d, BLOCK_K=block_k, BLOCK_T=block_t,
            INPUT_PRECISION=ip,
            WEIGHT_SHARING_ENC_DEC=True,
            SINGLE_M_TILE=single_m,
            USE_ASSOC_SCAN=use_assoc_scan,
            num_warps=num_warps, num_stages=2,
        )

    def _bwd_dg():
        return flare_recurrent_bwd_dg_part_kernel[g_bwd_dg](
            Q, K, V, dO, dg_buf,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dg_buf.stride(0), dg_buf.stride(1), dg_buf.stride(2),
            H, M, N, scale,
            D=D, BLOCK_M=block_m, BLOCK_D=block_d, BLOCK_K=block_k,
            BLOCK_T=bwd_block_t,
            num_warps=num_warps, num_stages=2,
        )

    def _bwd_dscore():
        return flare_recurrent_bwd_dscore_decode_kernel[g_bwd_dsc](
            Q_dec_bt_hd, K_dec_hmd, LSE_dec, dg_buf, dS_dec,
            Q_dec_bt_hd.stride(0), Q_dec_bt_hd.stride(1),
            Q_dec_bt_hd.stride(2), Q_dec_bt_hd.stride(3),
            K_dec_hmd.stride(0), K_dec_hmd.stride(1), K_dec_hmd.stride(2),
            LSE_dec.stride(0), LSE_dec.stride(1), LSE_dec.stride(2),
            dg_buf.stride(0), dg_buf.stride(1), dg_buf.stride(2),
            dS_dec.stride(0), dS_dec.stride(1), dS_dec.stride(2),
            H, M, N, scale,
            D=D, BLOCK_M=block_m, BLOCK_K=block_k, BLOCK_T=bwd_block_t,
            num_warps=num_warps, num_stages=2,
        )

    def _bwd_dsz():
        return flare_recurrent_bwd_dsz_separate_kernel[g_bwd_dsz](
            Q, K, V, LSE_enc,
            Q_dec_bt_hd, K_dec_hmd, LSE_dec,
            dO, dV_bhtd, dS_enc,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            LSE_enc.stride(0), LSE_enc.stride(1), LSE_enc.stride(2), LSE_enc.stride(3),
            Q_dec_bt_hd.stride(0), Q_dec_bt_hd.stride(1),
            Q_dec_bt_hd.stride(2), Q_dec_bt_hd.stride(3),
            K_dec_hmd.stride(0), K_dec_hmd.stride(1), K_dec_hmd.stride(2),
            LSE_dec.stride(0), LSE_dec.stride(1), LSE_dec.stride(2),
            dO.stride(0), dO.stride(1), dO.stride(2), dO.stride(3),
            dV_bhtd.stride(0), dV_bhtd.stride(1), dV_bhtd.stride(2),
            dS_enc.stride(0), dS_enc.stride(1), dS_enc.stride(2),
            H, M, N, scale,
            D=D, BLOCK_M=block_m, BLOCK_D=block_d, BLOCK_K=block_k,
            num_warps=num_warps, num_stages=2,
        )

    def _bwd_qk():
        return flare_chunk_bwd_recurrent_qk[g_bwd_qk](
            K, Q, dS_enc_qk, dQ, dK,
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            Q.stride(0), Q.stride(1), Q.stride(2),
            dS_enc_qk.stride(0), dS_enc_qk.stride(1),
            dS_enc_qk.stride(2), dS_enc_qk.stride(3),
            dQ.stride(0), dQ.stride(1), dQ.stride(2),
            dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
            BH, M, N, D, scale,
            CHUNK_SIZE=N, BLOCK_T=bwd_block_t, INPUT_PRECISION=ip,
            ACCUM_DK=False, H=H,
            num_warps=num_warps, num_stages=2,
        )

    # Run once to compile + populate LSE_dec / LSE_enc for backward kernels
    ck_fwd_lse  = _fwd_lse()
    ck_fwd_main = _fwd_main()
    torch.cuda.synchronize()
    ck_bwd_dg     = _bwd_dg()
    ck_bwd_dscore = _bwd_dscore()
    ck_bwd_dsz    = _bwd_dsz()
    ck_bwd_qk     = _bwd_qk()
    torch.cuda.synchronize()

    # =========================================================================
    # TIMING  (triton.testing.do_bench handles warmup + median)
    # =========================================================================
    def bench(fn):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        return float(triton.testing.do_bench(fn, rep=iterations, warmup=0))

    ms_fwd_lse    = bench(_fwd_lse)
    ms_fwd_main   = bench(_fwd_main)
    ms_bwd_dg     = bench(_bwd_dg)
    ms_bwd_dscore = bench(_bwd_dscore)
    ms_bwd_dsz    = bench(_bwd_dsz)
    ms_bwd_qk     = bench(_bwd_qk)

    # =========================================================================
    # ASSEMBLE PROFILES
    # =========================================================================
    fwd_profiles = [
        make_profile("fwd_decode_lse",  g_fwd_lse,  ck_fwd_lse,  props, ms_fwd_lse),
        make_profile("fwd_recurrent",   g_fwd_main, ck_fwd_main, props, ms_fwd_main),
    ]
    bwd_profiles = [
        make_profile("bwd_dg_part",       g_bwd_dg,  ck_bwd_dg,     props, ms_bwd_dg),
        make_profile("bwd_dscore_decode", g_bwd_dsc, ck_bwd_dscore, props, ms_bwd_dscore),
        make_profile("bwd_dsz_separate",  g_bwd_dsz, ck_bwd_dsz,    props, ms_bwd_dsz),
        make_profile("bwd_qk",            g_bwd_qk,  ck_bwd_qk,     props, ms_bwd_qk),
    ]

    return {
        "B": B, "H": H, "N": N, "D": D, "M": M,
        "dtype": str(dtype).removeprefix("torch."),
        "heuristics": {
            "block_m": block_m, "block_d": block_d, "block_k": block_k,
            "block_t": block_t, "bwd_block_t": bwd_block_t,
            "num_warps": num_warps, "single_m_tile": single_m,
            "weight_sharing": True, "input_precision": ip,
            "use_assoc_scan": use_assoc_scan,
        },
        "gpu": torch.cuda.get_device_name(device_idx),
        "forward": [asdict(p) for p in fwd_profiles],
        "backward": [asdict(p) for p in bwd_profiles],
        "fwd_total_ms": sum(p.avg_ms for p in fwd_profiles),
        "bwd_total_ms": sum(p.avg_ms for p in bwd_profiles),
    }


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def _fmt_grid(g: list | tuple) -> str:
    return "x".join(str(x) for x in g)


def print_results(r: dict) -> None:
    gpu = r["gpu"]
    h = r["heuristics"]
    scan_tag = " [assoc_scan=ON]" if h.get("use_assoc_scan") else ""
    print(
        f"\nRecurrentFLARE  B={r['B']} H={r['H']} N={r['N']} "
        f"D={r['D']} M={r['M']}  dtype={r['dtype']}{scan_tag}"
    )
    print(f"GPU : {gpu}")
    print(
        f"Heuristics : block_m={h['block_m']} block_d={h['block_d']} "
        f"block_k={h['block_k']} block_t={h['block_t']}(bwd={h['bwd_block_t']}) "
        f"num_warps={h['num_warps']} single_m={h['single_m_tile']} "
        f"weight_sharing={h['weight_sharing']} ip={h['input_precision']} "
        f"assoc_scan={h.get('use_assoc_scan', False)}"
    )

    COL = (
        f"{'kernel':<26}  {'grid':<18}  {'ms':>7}  {'%pass':>6}  "
        f"{'wrps':>4}  {'stgs':>4}  {'regs':>4}  {'spills':>6}  "
        f"{'smem_kb':>7}  {'occ%':>6}  {'cta/sm':>6}  limit"
    )
    SEP = "─" * len(COL)

    def _table(profiles: list[dict], total_ms: float, title: str) -> None:
        print(f"\n── {title} ── (total {total_ms:.3f} ms)")
        print(COL)
        print(SEP)
        for p in sorted(profiles, key=lambda x: x["avg_ms"], reverse=True):
            pct = 100.0 * p["avg_ms"] / total_ms if total_ms else 0.0
            grid = _fmt_grid(p["grid"])
            limit = ",".join(p["limiting_factors"])
            print(
                f"{p['name']:<26}  {grid:<18}  {p['avg_ms']:>7.3f}  {pct:>6.1f}  "
                f"{p['num_warps']:>4}  {p['num_stages']:>4}  {p['regs_per_thread']:>4}  "
                f"{p['spills']:>6}  {p['shared_bytes']/1024:>7.1f}  "
                f"{p['occupancy_pct_est']:>6.1f}  {p['active_ctas_per_sm_est']:>6}  "
                f"{limit}"
            )
        print(f"{'TOTAL':<26}  {'':18}  {total_ms:>7.3f}")

    _table(r["forward"],  r["fwd_total_ms"], "FORWARD")
    _table(r["backward"], r["bwd_total_ms"], "BACKWARD")
    print(
        f"\nForward+backward sum : {r['fwd_total_ms'] + r['bwd_total_ms']:.3f} ms"
        " (isolated kernel times)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_dtype(s: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[s.lower()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile RecurrentFLARE forward + backward per-kernel."
    )
    parser.add_argument("--B",    type=int, default=8)
    parser.add_argument("--H",    type=int, default=8)
    parser.add_argument("--N",    type=int, default=256)
    parser.add_argument("--D",    type=int, default=32)
    parser.add_argument("--M",    type=int, default=64)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--warmup",     type=int, default=25)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed",       type=int, default=0)
    parser.add_argument("--assoc-scan", action="store_true",
                        help="Use tl.associative_scan for the (m,d) prefix in fwd_recurrent.")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")

    results = profile_recurrent_flare(
        args.B, args.H, args.N, args.D, args.M,
        dtype=parse_dtype(args.dtype),
        warmup=args.warmup,
        iterations=args.iterations,
        seed=args.seed,
        use_assoc_scan=args.assoc_scan,
    )
    print_results(results)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

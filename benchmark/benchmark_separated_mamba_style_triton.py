#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch

from causal_flare.autoregressive.separated import flare_autoregressive_separated_pytorch
from causal_flare.autoregressive.separated_mamba_style import flare_autoregressive_separated_mamba_style_pytorch
from causal_flare.autoregressive.separated_mamba_style_triton import flare_autoregressive_separated_mamba_style_triton


@dataclass(frozen=True)
class Case:
    batch: int
    seqlen: int
    nheads: int
    nslots: int
    value_dim: int
    chunk_size: int
    seed: int


DEFAULT_CASES = [
    Case(batch=1, seqlen=8, nheads=2, nslots=16, value_dim=16, chunk_size=16, seed=7),
    Case(batch=2, seqlen=257, nheads=4, nslots=64, value_dim=64, chunk_size=256, seed=13),
    Case(batch=8, seqlen=2048, nheads=16, nslots=128, value_dim=64, chunk_size=128, seed=0),
]


def _parse_case(token: str) -> Case:
    parts = [int(x) for x in token.split(",")]
    if len(parts) != 7:
        raise ValueError("Case must be B,N,H,M,D,chunk,seed")
    return Case(
        batch=parts[0],
        seqlen=parts[1],
        nheads=parts[2],
        nslots=parts[3],
        value_dim=parts[4],
        chunk_size=parts[5],
        seed=parts[6],
    )


def _make_inputs(case: Case, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(case.seed)
    U = torch.randn((case.batch, case.seqlen, case.nheads, case.value_dim), device=device, dtype=torch.float32)
    retain = 1e-3 + 0.999 * torch.rand((case.batch, case.seqlen, case.nheads), device=device, dtype=torch.float32)
    write = torch.randn((case.batch, case.seqlen, case.nheads, case.nslots), device=device, dtype=torch.float32)
    decode = torch.randn((case.batch, case.seqlen, case.nheads, case.nslots), device=device, dtype=torch.float32)
    return U, retain, write, decode


def _clone_inputs(
    U: torch.Tensor,
    retain: torch.Tensor,
    write: torch.Tensor,
    decode: torch.Tensor,
    *,
    requires_grad: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        U.detach().clone().requires_grad_(requires_grad),
        retain.detach().clone().requires_grad_(requires_grad),
        write.detach().clone().requires_grad_(requires_grad),
        decode.detach().clone().requires_grad_(requires_grad),
    )


def _measure_forward_ms(fn, *, U, retain, write, decode, chunk_size: int, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        with torch.no_grad():
            fn(U=U, retain=retain, write=write, decode_weights=decode, chunk_size=chunk_size)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        with torch.no_grad():
            fn(U=U, retain=retain, write=write, decode_weights=decode, chunk_size=chunk_size)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _measure_total_ms(fn, *, U, retain, write, decode, grad_out, chunk_size: int, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        u, r, w, d = _clone_inputs(U, retain, write, decode, requires_grad=True)
        out = fn(U=u, retain=r, write=w, decode_weights=d, chunk_size=chunk_size)
        torch.autograd.grad(out.mul(grad_out).sum(), (u, r, w, d))
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        u, r, w, d = _clone_inputs(U, retain, write, decode, requires_grad=True)
        out = fn(U=u, retain=r, write=w, decode_weights=d, chunk_size=chunk_size)
        torch.autograd.grad(out.mul(grad_out).sum(), (u, r, w, d))
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / iters


def _correctness_metrics(case: Case, *, U, retain, write, decode) -> dict[str, float]:
    grad_out = torch.randn((case.batch, case.seqlen, case.nheads, case.value_dim), device=U.device, dtype=torch.float32)

    u, r, w, d = _clone_inputs(U, retain, write, decode, requires_grad=True)
    out_triton = flare_autoregressive_separated_mamba_style_triton(
        U=u,
        retain=r,
        write=w,
        decode_weights=d,
        chunk_size=case.chunk_size,
    )
    grads_triton = torch.autograd.grad(out_triton.mul(grad_out).sum(), (u, r, w, d))

    u_ref, r_ref, w_ref, d_ref = _clone_inputs(U, retain, write, decode, requires_grad=True)
    out_ref = flare_autoregressive_separated_pytorch(
        U=u_ref,
        retain=r_ref,
        write=w_ref,
        decode_weights=d_ref,
        chunk_size=case.chunk_size,
    )
    grads_ref = torch.autograd.grad(out_ref.mul(grad_out).sum(), (u_ref, r_ref, w_ref, d_ref))

    metrics = {
        "forward_max_abs": (out_triton - out_ref).abs().max().item(),
        "forward_mean_abs": (out_triton - out_ref).abs().mean().item(),
    }
    for name, grad_t, grad_ref in zip(("U", "retain", "write", "decode"), grads_triton, grads_ref):
        diff = (grad_t - grad_ref).abs()
        metrics[f"{name}_grad_max_abs"] = diff.max().item()
        metrics[f"{name}_grad_mean_abs"] = diff.mean().item()
    return metrics


def _timing_metrics(case: Case, *, U, retain, write, decode, warmup_fwd: int, iters_fwd: int, warmup_bwd: int, iters_bwd: int) -> dict[str, float]:
    grad_out = torch.randn((case.batch, case.seqlen, case.nheads, case.value_dim), device=U.device, dtype=torch.float32)
    metrics: dict[str, float] = {}
    for label, fn in (
        ("pytorch_mamba_style", flare_autoregressive_separated_mamba_style_pytorch),
        ("triton_mamba_style", flare_autoregressive_separated_mamba_style_triton),
    ):
        fwd_ms = _measure_forward_ms(
            fn,
            U=U,
            retain=retain,
            write=write,
            decode=decode,
            chunk_size=case.chunk_size,
            warmup=warmup_fwd,
            iters=iters_fwd,
        )
        total_ms = _measure_total_ms(
            fn,
            U=U,
            retain=retain,
            write=write,
            decode=decode,
            grad_out=grad_out,
            chunk_size=case.chunk_size,
            warmup=warmup_bwd,
            iters=iters_bwd,
        )
        metrics[f"{label}_forward_ms"] = fwd_ms
        metrics[f"{label}_backward_ms"] = total_ms - fwd_ms
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Correctness and timing sweep for separated Mamba-style Triton.")
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case as B,N,H,M,D,chunk,seed. May be repeated.",
    )
    parser.add_argument("--forward-warmup", type=int, default=2)
    parser.add_argument("--forward-iters", type=int, default=5)
    parser.add_argument("--backward-warmup", type=int, default=1)
    parser.add_argument("--backward-iters", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sweep.")

    cases = [_parse_case(token) for token in args.case] if args.case else DEFAULT_CASES
    device = torch.device("cuda")
    rows: list[dict[str, float | int]] = []
    for case in cases:
        U, retain, write, decode = _make_inputs(case, device=device)
        row: dict[str, float | int] = asdict(case)
        row.update(_correctness_metrics(case, U=U, retain=retain, write=write, decode=decode))
        row.update(
            _timing_metrics(
                case,
                U=U,
                retain=retain,
                write=write,
                decode=decode,
                warmup_fwd=args.forward_warmup,
                iters_fwd=args.forward_iters,
                warmup_bwd=args.backward_warmup,
                iters_bwd=args.backward_iters,
            )
        )
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    for row in rows:
        print(
            "case",
            f"B={row['batch']}",
            f"N={row['seqlen']}",
            f"H={row['nheads']}",
            f"M={row['nslots']}",
            f"D={row['value_dim']}",
            f"chunk={row['chunk_size']}",
            f"seed={row['seed']}",
        )
        print(
            "  correctness",
            f"fwd_max={row['forward_max_abs']:.6g}",
            f"retain_grad_max={row['retain_grad_max_abs']:.6g}",
            f"U_grad_max={row['U_grad_max_abs']:.6g}",
            f"write_grad_max={row['write_grad_max_abs']:.6g}",
            f"decode_grad_max={row['decode_grad_max_abs']:.6g}",
        )
        print(
            "  timing_ms",
            f"pt_fwd={row['pytorch_mamba_style_forward_ms']:.4f}",
            f"pt_bwd={row['pytorch_mamba_style_backward_ms']:.4f}",
            f"tr_fwd={row['triton_mamba_style_forward_ms']:.4f}",
            f"tr_bwd={row['triton_mamba_style_backward_ms']:.4f}",
        )


if __name__ == "__main__":
    main()

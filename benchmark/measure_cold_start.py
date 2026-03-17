#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path


DEFAULT_ANCHORS = {
    "small": {"batch_size": 1, "num_heads": 8, "seq_len": 2048, "latent_queries": 64, "score_head_dim": 64, "value_head_dim": 64},
    "large": {"batch_size": 1, "num_heads": 8, "seq_len": 32768, "latent_queries": 512, "score_head_dim": 256, "value_head_dim": 256},
}


def build_child_code(mode: str, anchor: dict[str, int], dtype: str, input_precision: str | None) -> str:
    precision_arg = repr(input_precision) if input_precision is not None else "None"
    common = textwrap.dedent(
        f"""
        import json
        import math
        import time
        import torch

        from causal_flare import (
            flare_autoregressive_decode_triton,
            flare_autoregressive_prefill_triton,
            flare_autoregressive_triton,
        )

        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = getattr(torch, {dtype!r})
        B = {anchor["batch_size"]}
        H = {anchor["num_heads"]}
        N = {anchor["seq_len"]}
        M = {anchor["latent_queries"]}
        Dk = {anchor["score_head_dim"]}
        Dv = {anchor["value_head_dim"]}
        input_precision = {precision_arg}
        scale = 1.0 / math.sqrt(Dk)
        Q = torch.randn((H, M, Dk), device=device, dtype=dtype)
        K = torch.randn((B, N, H, Dk), device=device, dtype=dtype)
        V = torch.randn((B, N, H, Dv), device=device, dtype=dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        """
    )
    if mode == "chunked_forward":
        body = "Y = flare_autoregressive_triton(Q, K, V, scale=scale, input_precision=input_precision)"
    elif mode == "chunked_train":
        body = textwrap.dedent(
            """
            Q.requires_grad_(True)
            K.requires_grad_(True)
            V.requires_grad_(True)
            Y = flare_autoregressive_triton(Q, K, V, scale=scale, input_precision=input_precision)
            loss = Y.square().mean()
            loss.backward()
            """
        )
    elif mode == "inference_decode":
        body = textwrap.dedent(
            """
            _, state = flare_autoregressive_prefill_triton(Q, K, V, scale=scale, input_precision=input_precision)
            K_step = torch.randn((B, 1, H, Dk), device=device, dtype=dtype)
            V_step = torch.randn((B, 1, H, Dv), device=device, dtype=dtype)
            Y, _ = flare_autoregressive_decode_triton(
                Q,
                K_step,
                V_step,
                state=state,
                scale=scale,
                input_precision=input_precision,
            )
            """
        )
    else:
        raise ValueError(f"Unsupported mode={mode!r}")

    trailer = textwrap.dedent(
        """
        torch.cuda.synchronize()
        elapsed_s = time.perf_counter() - t0
        print(json.dumps({"elapsed_s": elapsed_s}))
        """
    )
    return common + body + "\n" + trailer


def run_once(*, mode: str, anchor_name: str, dtype: str, input_precision: str | None, python_exe: str) -> dict[str, object]:
    anchor = DEFAULT_ANCHORS[anchor_name]
    cache_dir = tempfile.mkdtemp(prefix="flare-cold-triton-")
    env = os.environ.copy()
    env["TRITON_CACHE_DIR"] = cache_dir
    cmd = [python_exe, "-c", build_child_code(mode, anchor, dtype, input_precision)]
    start = time.perf_counter()
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    wall_s = time.perf_counter() - start
    stdout = proc.stdout.strip().splitlines()
    payload = json.loads(stdout[-1]) if stdout else {}
    result = {
        "mode": mode,
        "anchor": anchor_name,
        "dtype": dtype,
        "input_precision": input_precision or "default",
        "cache_dir": cache_dir,
        "returncode": proc.returncode,
        "wall_s": wall_s,
        "measured_s": payload.get("elapsed_s"),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    shutil.rmtree(cache_dir, ignore_errors=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure cold-cache first-use latency for representative FLARE operations.")
    parser.add_argument("--mode", choices=("chunked_forward", "chunked_train", "inference_decode"), default="chunked_forward")
    parser.add_argument("--anchor", choices=tuple(DEFAULT_ANCHORS.keys()), default="small")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--input-precision", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--json", action="store_true", help="Print the full result JSON instead of a short summary line.")
    args = parser.parse_args()

    result = run_once(
        mode=args.mode,
        anchor_name=args.anchor,
        dtype=args.dtype,
        input_precision=args.input_precision,
        python_exe=args.python,
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return
    if result["returncode"] != 0:
        print(json.dumps(result, indent=2))
        raise SystemExit(result["returncode"])
    print(
        f"mode={result['mode']} anchor={result['anchor']} dtype={result['dtype']} "
        f"input_precision={result['input_precision']} wall_s={result['wall_s']:.3f} measured_s={float(result['measured_s']):.3f}"
    )


if __name__ == "__main__":
    main()

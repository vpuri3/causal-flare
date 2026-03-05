import argparse

try:
    from testing.test import run_module_main
except Exception as exc:
    raise RuntimeError(
        "Diagnostics moved to testing/test.py. Run from the repo root so the testing package is importable."
    ) from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ChunkedFLARE diagnostics.")
    parser.add_argument("--B", type=int, default=8)
    parser.add_argument("--H", type=int, default=16)
    parser.add_argument("--M", type=int, default=64)
    parser.add_argument("--N", type=int, default=2048)
    parser.add_argument("--D", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_module_main(B=args.B, H=args.H, M=args.M, N=args.N, D=args.D, dtype=args.dtype)


if __name__ == "__main__":
    main()

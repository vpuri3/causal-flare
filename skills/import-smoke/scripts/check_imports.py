#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_IMPORTS = [
    "causal_flare.autoregressive.recurrent:flare_recurrent_pytorch",
    "causal_flare.autoregressive.dense:flare_recurrent_dense_backward_pytorch",
]


def _parse_import_spec(spec: str) -> tuple[str, list[str]]:
    module_name, sep, raw_symbols = spec.partition(":")
    if not sep or not module_name or not raw_symbols:
        raise ValueError(f"Invalid --import spec {spec!r}. Expected module:symbol or module:s1,s2")
    symbols = [symbol.strip() for symbol in raw_symbols.split(",") if symbol.strip()]
    if not symbols:
        raise ValueError(f"Invalid --import spec {spec!r}. No symbols were provided.")
    return module_name, symbols


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test causal_flare imports without running pytest.")
    parser.add_argument(
        "--import",
        dest="imports",
        action="append",
        default=[],
        help="Import spec in the form module:symbol or module:s1,s2. May be passed multiple times.",
    )
    args = parser.parse_args()

    import_specs = args.imports or DEFAULT_IMPORTS
    for spec in import_specs:
        module_name, symbols = _parse_import_spec(spec)
        for symbol_name in symbols:
            try:
                namespace = {}
                exec(f"from {module_name} import {symbol_name}", {}, namespace)
                symbol = namespace[symbol_name]
            except Exception as exc:
                print(f"FAIL import from {module_name} import {symbol_name}: {exc}", file=sys.stderr)
                return 1
            print(f"OK {module_name}:{symbol_name} -> {getattr(symbol, '__module__', type(symbol).__module__)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

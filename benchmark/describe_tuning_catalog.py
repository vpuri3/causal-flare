#!/usr/bin/env python
from __future__ import annotations

import argparse
import json

try:
    from .tuning_catalog import (
        TuningCatalog,
        get_all_tuning_catalogs,
        get_chunked_tuning_catalog,
        get_recurrent_tuning_catalog,
    )
except ImportError:
    from tuning_catalog import (
        TuningCatalog,
        get_all_tuning_catalogs,
        get_chunked_tuning_catalog,
        get_recurrent_tuning_catalog,
    )


def format_markdown(catalogs: tuple[TuningCatalog, ...]) -> str:
    lines: list[str] = ["# FLARE Tuning Catalog", ""]
    for catalog in catalogs:
        lines.append(f"## {catalog.implementation}")
        lines.append("")
        lines.append(catalog.summary)
        lines.append("")
        lines.append("### Families")
        lines.append("")
        for group, families in catalog.family_groups.items():
            lines.append(f"- `{group}`: {', '.join(f'`{family}`' for family in families)}")
        lines.append("")
        lines.append("### Parameters")
        lines.append("")
        for param in catalog.parameters:
            lines.append(f"- `{param.name}`")
            lines.append(f"  env: {', '.join(f'`{env}`' for env in param.env_vars)}")
            lines.append(f"  phases: {', '.join(f'`{phase}`' for phase in param.phases)}")
            lines.append(
                f"  independent of: {', '.join(f'`{item}`' for item in param.independent_of) if param.independent_of else 'none'}"
            )
            lines.append(
                f"  connected to: {', '.join(f'`{item}`' for item in param.connected_to) if param.connected_to else 'none'}"
            )
            lines.append(f"  constraints: {param.constraints}")
            lines.append(f"  notes: {param.notes}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe the maintained FLARE launch-parameter tuning catalog.")
    parser.add_argument("--impl", choices=("all", "recurrent", "chunked"), default="all")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()

    if args.impl == "all":
        catalogs = get_all_tuning_catalogs()
    elif args.impl == "recurrent":
        catalogs = (get_recurrent_tuning_catalog(),)
    else:
        catalogs = (get_chunked_tuning_catalog(),)

    if args.format == "json":
        print(json.dumps([catalog.to_dict() for catalog in catalogs], indent=2))
        return

    print(format_markdown(catalogs), end="")


if __name__ == "__main__":
    main()

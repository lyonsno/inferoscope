"""CLI for printing compact summaries of inferoscope run bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from inferoscope.extraction import load_run_bundle


def _format_layer_expert_counts(layout: dict[str, Any]) -> str:
    layer_counts: list[tuple[int, int]] = []
    for layer in layout.get("layers", []):
        layer_index = layer.get("layer_index")
        positions = layer.get("positions", [])
        if not isinstance(layer_index, int) or not isinstance(positions, list):
            continue
        layer_counts.append((layer_index, len(positions)))

    layer_counts.sort(key=lambda pair: pair[0])
    return ", ".join(f"{layer_index}={count}" for layer_index, count in layer_counts)


def build_summary_lines(bundle: dict[str, Any]) -> list[str]:
    """Return user-facing summary lines for a loaded run bundle."""

    manifest = bundle["manifest"]
    raw_events = bundle["raw_events"]
    layout = bundle["layout"]

    optional_artifacts = (
        f"derived={'yes' if bundle.get('derived_events') is not None else 'no'} "
        f"motif_ledger={'yes' if bundle.get('motif_ledger') is not None else 'no'} "
        f"contingency={'yes' if bundle.get('contingency') is not None else 'no'}"
    )

    return [
        f"run_id: {manifest.get('run_id')}",
        f"created_at: {manifest.get('created_at')}",
        f"model_id: {manifest.get('model_id')}",
        f"tokenizer_id: {manifest.get('tokenizer_id')}",
        f"raw_events: {len(raw_events)}",
        f"layers: {len(layout.get('layers', []))}",
        f"layer_expert_counts: {_format_layer_expert_counts(layout)}",
        f"optional_artifacts: {optional_artifacts}",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inferoscope.inspect",
        description="Load, validate, and summarize an inferoscope run bundle.",
    )
    parser.add_argument("run_dir", help="Path to a run bundle directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        bundle = load_run_bundle(Path(args.run_dir))
    except Exception as exc:
        parser.exit(status=1, message=f"{exc}\n")

    for line in build_summary_lines(bundle):
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

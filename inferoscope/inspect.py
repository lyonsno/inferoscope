"""CLI for printing compact summaries of inferoscope run bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inferoscope.extraction import load_run_bundle


def _layer_expert_counts(layout: dict[str, Any]) -> list[dict[str, int]]:
    layer_counts: list[tuple[int, int]] = []
    for layer in layout.get("layers", []):
        layer_index = layer.get("layer_index")
        positions = layer.get("positions", [])
        if not isinstance(layer_index, int) or not isinstance(positions, list):
            continue
        layer_counts.append((layer_index, len(positions)))

    layer_counts.sort(key=lambda pair: pair[0])
    return [
        {"layer_index": layer_index, "num_total_experts": count}
        for layer_index, count in layer_counts
    ]


def _format_layer_expert_counts(layout: dict[str, Any]) -> str:
    return ", ".join(
        f"{layer['layer_index']}={layer['num_total_experts']}"
        for layer in _layer_expert_counts(layout)
    )


def _decode_duration_summary(raw_events: list[dict[str, Any]]) -> dict[str, float] | None:
    decode_durations: list[float] = []
    for event in raw_events:
        timing_ms = event.get("timing_ms")
        if not isinstance(timing_ms, dict):
            continue
        decode_duration = timing_ms.get("decode_duration")
        if isinstance(decode_duration, (int, float)):
            decode_durations.append(float(decode_duration))

    if not decode_durations:
        return None

    return {
        "min": min(decode_durations),
        "max": max(decode_durations),
        "avg": sum(decode_durations) / len(decode_durations),
    }


def build_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    """Return a machine-readable summary for a loaded run bundle."""

    manifest = bundle["manifest"]
    raw_events = bundle["raw_events"]
    layout = bundle["layout"]

    return {
        "run_id": manifest.get("run_id"),
        "created_at": manifest.get("created_at"),
        "model_id": manifest.get("model_id"),
        "tokenizer_id": manifest.get("tokenizer_id"),
        "prompt_text": manifest.get("prompt", {}).get("text"),
        "raw_events": len(raw_events),
        "layers": len(layout.get("layers", [])),
        "layer_expert_counts": _layer_expert_counts(layout),
        "optional_artifacts": {
            "derived": bundle.get("derived_events") is not None,
            "motif_ledger": bundle.get("motif_ledger") is not None,
            "contingency": bundle.get("contingency") is not None,
        },
        "decode_duration_ms": _decode_duration_summary(raw_events),
    }


def build_summary_lines(bundle: dict[str, Any]) -> list[str]:
    """Return user-facing summary lines for a loaded run bundle."""

    summary = build_summary(bundle)

    optional_artifacts = (
        f"derived={'yes' if summary['optional_artifacts']['derived'] else 'no'} "
        f"motif_ledger={'yes' if summary['optional_artifacts']['motif_ledger'] else 'no'} "
        f"contingency={'yes' if summary['optional_artifacts']['contingency'] else 'no'}"
    )

    lines = [
        f"run_id: {summary['run_id']}",
        f"created_at: {summary['created_at']}",
        f"model_id: {summary['model_id']}",
        f"tokenizer_id: {summary['tokenizer_id']}",
        f"raw_events: {summary['raw_events']}",
        f"layers: {summary['layers']}",
        f"layer_expert_counts: {_format_layer_expert_counts(bundle['layout'])}",
        f"optional_artifacts: {optional_artifacts}",
    ]

    if summary["prompt_text"] is not None:
        lines.append(f"prompt_text: {json.dumps(summary['prompt_text'])}")
    if summary["decode_duration_ms"] is not None:
        decode = summary["decode_duration_ms"]
        lines.append(
            f"decode_duration_ms: min={decode['min']} avg={decode['avg']} max={decode['max']}"
        )

    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m inferoscope.inspect",
        description="Load, validate, and summarize an inferoscope run bundle.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the summary as JSON for scripts and CI.",
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

    if args.json:
        print(json.dumps(build_summary(bundle), indent=2, sort_keys=True))
        return 0

    for line in build_summary_lines(bundle):
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

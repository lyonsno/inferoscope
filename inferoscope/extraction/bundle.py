"""Run-bundle builders for inferoscope extraction outputs."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import math


def _validate_created_at(created_at: str) -> None:
    if not created_at:
        raise ValueError("created_at must not be empty")

    normalized = created_at[:-1] + "+00:00" if created_at.endswith("Z") else created_at
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("created_at must be a valid ISO 8601 date-time string") from exc

    if parsed.tzinfo is None:
        raise ValueError("created_at must include timezone information")


def build_manifest(
    *,
    run_id: str,
    created_at: str,
    model_id: str,
    tokenizer_id: str,
    prompt_text: str,
    derivation_version: str,
    derivation_config_id: str,
    seed: int | None = None,
    generation_config: dict | None = None,
    capture_config: dict | None = None,
) -> dict:
    """Build a manifest artifact for a captured run bundle."""

    if not run_id:
        raise ValueError("run_id must not be empty")
    _validate_created_at(created_at)
    if not model_id:
        raise ValueError("model_id must not be empty")
    if not tokenizer_id:
        raise ValueError("tokenizer_id must not be empty")
    if not derivation_version:
        raise ValueError("derivation_version must not be empty")
    if not derivation_config_id:
        raise ValueError("derivation_config_id must not be empty")

    return {
        "schema_version": "manifest/v0.1.0",
        "run_id": run_id,
        "created_at": created_at,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "prompt": {"text": prompt_text},
        "seed": seed,
        "generation_config": generation_config or {},
        "capture_config": capture_config or {},
        "artifact_versions": {
            "raw_event_schema_version": "raw/v0.1.0",
            "derived_event_schema_version": "derived/v0.1.0-provisional",
            "layout_schema_version": "layout/v0.1.0",
            "motif_ledger_schema_version": "motif_ledger/v0.1.0-provisional",
            "contingency_schema_version": "contingency/v0.1.0-provisional",
        },
        "derivation_version": derivation_version,
        "derivation_config_id": derivation_config_id,
    }


def build_layer_grid_layout(
    *,
    run_id: str,
    layout_id: str,
    layer_expert_counts: Sequence[tuple[int, int]],
    layout_kind: str = "layer-grid",
) -> dict:
    """Build a simple deterministic grid layout for layer/expert positions."""

    if not run_id:
        raise ValueError("run_id must not be empty")
    if not layout_id:
        raise ValueError("layout_id must not be empty")
    if not layer_expert_counts:
        raise ValueError("layer_expert_counts must not be empty")

    layers: list[dict] = []
    seen_layer_indices: set[int] = set()
    for layer_index, expert_count in layer_expert_counts:
        if layer_index in seen_layer_indices:
            raise ValueError("duplicate layer_index values are not allowed")
        if layer_index < 0:
            raise ValueError("layer_index must be non-negative")
        if expert_count < 1:
            raise ValueError("expert_count must be at least 1")

        seen_layer_indices.add(layer_index)
        columns = math.ceil(math.sqrt(expert_count))
        positions = []
        for expert_index in range(expert_count):
            positions.append(
                {
                    "expert_index": expert_index,
                    "x": float(expert_index % columns),
                    "y": float(expert_index // columns),
                    "z": float(layer_index),
                }
            )

        layers.append(
            {
                "layer_index": layer_index,
                "positions": positions,
            }
        )

    return {
        "schema_version": "layout/v0.1.0",
        "run_id": run_id,
        "layout_id": layout_id,
        "layout_kind": layout_kind,
        "layers": layers,
    }

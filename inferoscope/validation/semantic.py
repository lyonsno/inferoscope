"""Semantic validation for inferoscope artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

FLOAT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ValidationIssue:
    """A semantic validation failure or warning."""

    scope: str
    code: str
    message: str


def _issue(scope: str, code: str, message: str) -> ValidationIssue:
    return ValidationIssue(scope=scope, code=code, message=message)


def _approx_equal(left: float, right: float, *, tolerance: float = FLOAT_TOLERANCE) -> bool:
    return abs(left - right) <= tolerance


def validate_raw_event_semantics(event: dict[str, Any]) -> list[ValidationIssue]:
    """Return semantic validation issues for a raw token-complete event."""

    issues: list[ValidationIssue] = []

    timing = event.get("timing_ms", {})
    decode_start = timing.get("decode_start")
    decode_end = timing.get("decode_end")
    decode_duration = timing.get("decode_duration")

    if (
        isinstance(decode_start, (int, float))
        and isinstance(decode_end, (int, float))
        and isinstance(decode_duration, (int, float))
    ):
        expected_duration = decode_end - decode_start
        if not _approx_equal(expected_duration, decode_duration):
            issues.append(
                _issue(
                    "raw_event",
                    "decode_duration_mismatch",
                    "timing_ms.decode_duration must equal decode_end - decode_start within tolerance.",
                )
            )

    for layer in event.get("layers", []):
        layer_index = layer.get("layer_index", "?")
        scope = f"raw_event.layer[{layer_index}]"

        router_probs = layer.get("router_probs", [])
        topk_indices = layer.get("topk_indices", [])
        topk_probs = layer.get("topk_probs", [])
        num_total_experts = layer.get("num_total_experts")
        num_active_experts = layer.get("num_active_experts")

        if isinstance(num_total_experts, int) and len(router_probs) != num_total_experts:
            issues.append(
                _issue(
                    scope,
                    "router_probs_length_mismatch",
                    "router_probs length must equal num_total_experts.",
                )
            )

        if isinstance(num_active_experts, int):
            if len(topk_indices) != num_active_experts or len(topk_probs) != num_active_experts:
                issues.append(
                    _issue(
                        scope,
                        "topk_length_mismatch",
                        "topk_indices and topk_probs length must each equal num_active_experts.",
                    )
                )

        if len(topk_indices) != len(topk_probs):
            issues.append(
                _issue(
                    scope,
                    "topk_length_mismatch",
                    "topk_indices and topk_probs must have the same length.",
                )
            )

        if isinstance(num_total_experts, int):
            for expert_index in topk_indices:
                if not isinstance(expert_index, int) or expert_index < 0 or expert_index >= num_total_experts:
                    issues.append(
                        _issue(
                            scope,
                            "topk_index_out_of_bounds",
                            "topk_indices entries must be within [0, num_total_experts).",
                        )
                    )
                    break

        if isinstance(num_total_experts, int) and len(router_probs) == num_total_experts:
            router_sum = sum(router_probs)
            if not _approx_equal(router_sum, 1.0):
                issues.append(
                    _issue(
                        scope,
                        "router_probs_sum_mismatch",
                        "router_probs must sum to 1.0 within tolerance.",
                    )
                )

        if len(topk_indices) == len(topk_probs) and router_probs:
            for expert_index, topk_prob in zip(topk_indices, topk_probs):
                if isinstance(expert_index, int) and 0 <= expert_index < len(router_probs):
                    if not _approx_equal(router_probs[expert_index], topk_prob):
                        issues.append(
                            _issue(
                                scope,
                                "topk_prob_mismatch",
                                "topk_probs must match router_probs at the corresponding topk_indices within tolerance.",
                            )
                        )
                        break

        if router_probs:
            expected_top1 = max(router_probs)
            top1_prob = layer.get("top1_prob")
            if isinstance(top1_prob, (int, float)) and not _approx_equal(expected_top1, top1_prob):
                issues.append(
                    _issue(
                        scope,
                        "top1_prob_mismatch",
                        "top1_prob must equal the maximum router probability within tolerance.",
                    )
                )

        if len(topk_probs) >= 2:
            expected_margin = topk_probs[0] - topk_probs[1]
            top1_top2_margin = layer.get("top1_top2_margin")
            if isinstance(top1_top2_margin, (int, float)) and not _approx_equal(
                expected_margin, top1_top2_margin
            ):
                issues.append(
                    _issue(
                        scope,
                        "top1_top2_margin_mismatch",
                        "top1_top2_margin must equal topk_probs[0] - topk_probs[1] within tolerance.",
                    )
                )

        if router_probs and isinstance(num_total_experts, int) and num_total_experts > 1:
            expected_entropy = -sum(prob * math.log(prob) for prob in router_probs if prob > 0.0)
            entropy = layer.get("entropy")
            if isinstance(entropy, (int, float)) and not _approx_equal(expected_entropy, entropy):
                issues.append(
                    _issue(
                        scope,
                        "entropy_mismatch",
                        "entropy must equal -sum_i p_i ln(p_i) within tolerance.",
                    )
                )

            expected_normalized_entropy = expected_entropy / math.log(num_total_experts)
            normalized_entropy = layer.get("normalized_entropy")
            if isinstance(normalized_entropy, (int, float)) and not _approx_equal(
                expected_normalized_entropy, normalized_entropy
            ):
                issues.append(
                    _issue(
                        scope,
                        "normalized_entropy_mismatch",
                        "normalized_entropy must equal entropy / ln(num_total_experts) within tolerance.",
                    )
                )

    return issues


def validate_manifest_semantics(
    manifest: dict[str, Any], *, derived_artifacts_present: bool = False
) -> list[ValidationIssue]:
    """Return semantic validation issues for a manifest."""

    del derived_artifacts_present
    del manifest
    return []


def validate_layout_semantics(layout: dict[str, Any]) -> list[ValidationIssue]:
    """Return semantic validation issues for a layout artifact."""

    issues: list[ValidationIssue] = []
    seen_layer_indices: set[int] = set()

    for layer in layout.get("layers", []):
        layer_index = layer.get("layer_index", "?")
        scope = f"layout.layer[{layer_index}]"

        if isinstance(layer_index, int):
            if layer_index in seen_layer_indices:
                issues.append(
                    _issue(
                        scope,
                        "duplicate_layer_index",
                        "layout layers must not repeat layer_index values.",
                    )
                )
            seen_layer_indices.add(layer_index)

        seen_expert_indices: set[int] = set()
        for position in layer.get("positions", []):
            expert_index = position.get("expert_index")
            if isinstance(expert_index, int):
                if expert_index in seen_expert_indices:
                    issues.append(
                        _issue(
                            scope,
                            "duplicate_expert_index",
                            "positions within a layer must not repeat expert_index values.",
                        )
                    )
                    break
                seen_expert_indices.add(expert_index)

    return issues


def _expected_layout_expert_counts(raw_events: list[dict[str, Any]]) -> tuple[dict[int, int], list[ValidationIssue]]:
    expected_counts: dict[int, int] = {}
    issues: list[ValidationIssue] = []

    for event in raw_events:
        for layer in event.get("layers", []):
            layer_index = layer.get("layer_index")
            num_total_experts = layer.get("num_total_experts")
            if not isinstance(layer_index, int) or not isinstance(num_total_experts, int):
                continue

            previous = expected_counts.get(layer_index)
            if previous is None:
                expected_counts[layer_index] = num_total_experts
            elif previous != num_total_experts:
                issues.append(
                    _issue(
                        "run_bundle",
                        "raw_layer_expert_count_inconsistent",
                        f"raw events disagree on num_total_experts for layer {layer_index}.",
                    )
                )

    return expected_counts, issues


def _artifact_derivation_metadata(artifact: dict[str, Any] | None) -> tuple[str | None, str | None]:
    if not artifact:
        return None, None

    return artifact.get("derivation_version"), artifact.get("derivation_config_id")


def validate_run_bundle_semantics(
    manifest: dict[str, Any],
    raw_events: list[dict[str, Any]],
    layout: dict[str, Any],
    *,
    derived_events: list[dict[str, Any]] | None = None,
    motif_ledger: dict[str, Any] | None = None,
    contingency: dict[str, Any] | None = None,
) -> list[ValidationIssue]:
    """Return semantic validation issues for a run bundle."""

    issues: list[ValidationIssue] = []

    issues.extend(validate_manifest_semantics(manifest, derived_artifacts_present=bool(derived_events)))

    for event in raw_events:
        issues.extend(validate_raw_event_semantics(event))

    issues.extend(validate_layout_semantics(layout))

    expected_counts, raw_count_issues = _expected_layout_expert_counts(raw_events)
    issues.extend(raw_count_issues)

    layout_layers = {
        layer.get("layer_index"): layer
        for layer in layout.get("layers", [])
        if isinstance(layer.get("layer_index"), int)
    }
    for layer_index, num_total_experts in expected_counts.items():
        layout_layer = layout_layers.get(layer_index)
        if layout_layer is None or len(layout_layer.get("positions", [])) != num_total_experts:
            issues.append(
                _issue(
                    "run_bundle",
                    "layout_expert_count_mismatch",
                    f"layout layer {layer_index} must contain {num_total_experts} expert positions.",
                )
            )

    expected_derivation_version = manifest.get("derivation_version")
    expected_derivation_config_id = manifest.get("derivation_config_id")

    artifacts_to_compare: list[tuple[str, dict[str, Any]]] = []
    for event in derived_events or []:
        artifacts_to_compare.append(("derived_event", event))
    if motif_ledger is not None:
        artifacts_to_compare.append(("motif_ledger", motif_ledger))
    if contingency is not None:
        artifacts_to_compare.append(("contingency", contingency))

    for artifact_name, artifact in artifacts_to_compare:
        derivation_version, derivation_config_id = _artifact_derivation_metadata(artifact)
        if (
            expected_derivation_version is not None
            and derivation_version is not None
            and derivation_version != expected_derivation_version
        ):
            issues.append(
                _issue(
                    "run_bundle",
                    "derivation_version_mismatch",
                    f"{artifact_name} derivation_version must match the manifest derivation_version.",
                )
            )

        if (
            expected_derivation_config_id is not None
            and derivation_config_id is not None
            and derivation_config_id != expected_derivation_config_id
        ):
            issues.append(
                _issue(
                    "run_bundle",
                    "derivation_config_mismatch",
                    f"{artifact_name} derivation_config_id must match the manifest derivation_config_id.",
                )
            )

    return issues

"""Validation helpers for inferoscope artifacts."""

from .semantic import (
    ValidationIssue,
    validate_layout_semantics,
    validate_manifest_semantics,
    validate_raw_event_semantics,
    validate_run_bundle_semantics,
)

__all__ = [
    "ValidationIssue",
    "validate_layout_semantics",
    "validate_manifest_semantics",
    "validate_raw_event_semantics",
    "validate_artifact_schema",
    "validate_run_bundle_semantics",
    "validate_run_bundle_schema",
]


def __getattr__(name: str):
    if name in {"validate_artifact_schema", "validate_run_bundle_schema"}:
        from .schema import validate_artifact_schema, validate_run_bundle_schema

        if name == "validate_artifact_schema":
            return validate_artifact_schema
        return validate_run_bundle_schema

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

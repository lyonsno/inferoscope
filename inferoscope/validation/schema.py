"""Schema validation helpers for inferoscope artifacts."""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

from inferoscope.formats import is_rfc3339_datetime


_SCHEMA_DIR = Path(__file__).with_name("schemas") / "v0.1.0"


@lru_cache(maxsize=None)
def _load_schema(filename: str) -> dict[str, Any]:
    return json.loads((_SCHEMA_DIR / filename).read_text(encoding="utf-8"))


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _type_matches(value: Any, schema_type: str) -> bool:
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "integer":
        return _is_integer(value)
    if schema_type == "number":
        return _is_number(value)
    if schema_type == "null":
        return value is None
    return True


def _format_matches(value: Any, schema_format: str) -> bool:
    if schema_format != "date-time" or not isinstance(value, str):
        return True

    return is_rfc3339_datetime(value)


def _resolve_ref(ref: str, root_schema: dict[str, Any]) -> dict[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported schema ref: {ref}")

    target: Any = root_schema
    for part in ref[2:].split("/"):
        target = target[part]

    if not isinstance(target, dict):
        raise ValueError(f"schema ref does not resolve to an object: {ref}")

    return target


def _validate_against_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    location: str,
    root_schema: dict[str, Any],
) -> list[str]:
    if "$ref" in schema:
        return _validate_against_schema(
            value,
            _resolve_ref(schema["$ref"], root_schema),
            location=location,
            root_schema=root_schema,
        )

    issues: list[str] = []

    schema_type = schema.get("type")
    if schema_type is not None:
        type_options = schema_type if isinstance(schema_type, list) else [schema_type]
        if not any(_type_matches(value, type_option) for type_option in type_options):
            issues.append(f"{location} must have schema type {schema_type!r}")
            return issues

    if "const" in schema and value != schema["const"]:
        issues.append(f"{location} must equal {schema['const']!r}")
        return issues

    if "enum" in schema and value not in schema["enum"]:
        issues.append(f"{location} must be one of {schema['enum']!r}")

    if "format" in schema and not _format_matches(value, schema["format"]):
        issues.append(f"{location} must match format {schema['format']!r}")

    if isinstance(value, str):
        min_length = schema.get("minLength")
        if min_length is not None and len(value) < min_length:
            issues.append(f"{location} must have length >= {min_length}")

    if _is_number(value):
        minimum = schema.get("minimum")
        if minimum is not None and float(value) < minimum:
            issues.append(f"{location} must be >= {minimum}")
        maximum = schema.get("maximum")
        if maximum is not None and float(value) > maximum:
            issues.append(f"{location} must be <= {maximum}")

    if isinstance(value, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                issues.append(f"{location}.{key} is required")

        properties = schema.get("properties", {})
        additional_properties = schema.get("additionalProperties", True)
        if additional_properties is False:
            allowed_keys = set(properties)
            for key in value:
                if key not in allowed_keys:
                    issues.append(f"{location}.{key} is not allowed by the schema")

        for key, property_schema in properties.items():
            if key in value:
                issues.extend(
                    _validate_against_schema(
                        value[key],
                        property_schema,
                        location=f"{location}.{key}",
                        root_schema=root_schema,
                    )
                )

    if isinstance(value, list):
        min_items = schema.get("minItems")
        if min_items is not None and len(value) < min_items:
            issues.append(f"{location} must contain at least {min_items} item(s)")

        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                issues.extend(
                    _validate_against_schema(
                        item,
                        item_schema,
                        location=f"{location}[{index}]",
                        root_schema=root_schema,
                    )
                )

    return issues


def validate_run_bundle_schema(
    manifest: dict[str, Any],
    raw_events: list[dict[str, Any]],
    layout: dict[str, Any],
    *,
    derived_events: list[dict[str, Any]] | None = None,
    motif_ledger: dict[str, Any] | None = None,
    contingency: dict[str, Any] | None = None,
) -> list[str]:
    """Return schema-validation issues for a run bundle."""

    manifest_schema = _load_schema("manifest.schema.json")
    raw_event_schema = _load_schema("raw_trace_event.schema.json")
    layout_schema = _load_schema("layout.schema.json")
    derived_event_schema = _load_schema("derived_event.schema.json")
    motif_ledger_schema = _load_schema("motif_ledger.schema.json")
    contingency_schema = _load_schema("contingency.schema.json")

    issues: list[str] = []

    issues.extend(
        _validate_against_schema(
            manifest,
            manifest_schema,
            location="manifest",
            root_schema=manifest_schema,
        )
    )

    for index, event in enumerate(raw_events):
        issues.extend(
            _validate_against_schema(
                event,
                raw_event_schema,
                location=f"raw_events[{index}]",
                root_schema=raw_event_schema,
            )
        )

    issues.extend(
        _validate_against_schema(
            layout,
            layout_schema,
            location="layout",
            root_schema=layout_schema,
        )
    )

    for index, event in enumerate(derived_events or []):
        issues.extend(
            _validate_against_schema(
                event,
                derived_event_schema,
                location=f"derived_events[{index}]",
                root_schema=derived_event_schema,
            )
        )

    if motif_ledger is not None:
        issues.extend(
            _validate_against_schema(
                motif_ledger,
                motif_ledger_schema,
                location="motif_ledger",
                root_schema=motif_ledger_schema,
            )
        )

    if contingency is not None:
        issues.extend(
            _validate_against_schema(
                contingency,
                contingency_schema,
                location="contingency",
                root_schema=contingency_schema,
            )
        )

    return issues

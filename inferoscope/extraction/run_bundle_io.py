"""Run-bundle file I/O for inferoscope replay artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from inferoscope.run_ids import run_id_path
from inferoscope.validation import validate_run_bundle_schema, validate_run_bundle_semantics


def _json_dumps(payload: Any, *, pretty: bool) -> str:
    try:
        if pretty:
            return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
        return json.dumps(payload, allow_nan=False, sort_keys=True)
    except ValueError as exc:
        raise ValueError("payload contains non-standard JSON numeric values") from exc


def _json_loads(contents: str, *, source_name: str) -> Any:
    def _raise_invalid_constant(token: str) -> None:
        raise ValueError(f"{source_name} contains non-standard JSON numeric token {token}")

    return json.loads(contents, parse_constant=_raise_invalid_constant)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(_json_dumps(payload, pretty=True) + "\n", encoding="utf-8")


def _write_ndjson(path: Path, payloads: list[dict[str, Any]]) -> None:
    lines = [_json_dumps(payload, pretty=False) for payload in payloads]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = _json_loads(path.read_text(encoding="utf-8"), source_name=path.name)
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")

    return payload


def _read_ndjson(path: Path) -> list[dict[str, Any]]:
    contents = path.read_text(encoding="utf-8")
    if not contents.strip():
        return []

    payloads: list[dict[str, Any]] = []
    for line_number, line in enumerate(contents.splitlines(), start=1):
        if not line.strip():
            continue

        payload = _json_loads(line, source_name=f"{path.name} line {line_number}")
        if not isinstance(payload, dict):
            raise ValueError(f"{path.name} line {line_number} must contain a JSON object")
        payloads.append(payload)

    return payloads


def _raise_for_schema_issues(
    manifest: dict[str, Any],
    raw_events: list[dict[str, Any]],
    layout: dict[str, Any],
    *,
    derived_events: list[dict[str, Any]] | None = None,
    motif_ledger: dict[str, Any] | None = None,
    contingency: dict[str, Any] | None = None,
) -> None:
    issues = validate_run_bundle_schema(
        manifest,
        raw_events,
        layout,
        derived_events=derived_events,
        motif_ledger=motif_ledger,
        contingency=contingency,
    )
    if issues:
        raise ValueError(f"run bundle failed schema validation: {issues[0]}")


def _raise_for_semantic_issues(
    manifest: dict[str, Any],
    raw_events: list[dict[str, Any]],
    layout: dict[str, Any],
    *,
    derived_events: list[dict[str, Any]] | None = None,
    motif_ledger: dict[str, Any] | None = None,
    contingency: dict[str, Any] | None = None,
) -> None:
    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("manifest run_id must be a non-empty string")
    run_id_path(run_id, subject="manifest run_id")

    issues = validate_run_bundle_semantics(
        manifest,
        raw_events,
        layout,
        derived_events=derived_events,
        motif_ledger=motif_ledger,
        contingency=contingency,
    )
    if issues:
        issue_codes = ", ".join(issue.code for issue in issues)
        raise ValueError(f"run bundle failed semantic validation: {issue_codes}")


def write_run_bundle(
    root_dir: str | Path,
    manifest: dict[str, Any],
    raw_events: list[dict[str, Any]],
    layout: dict[str, Any],
    *,
    derived_events: list[dict[str, Any]] | None = None,
    motif_ledger: dict[str, Any] | None = None,
    contingency: dict[str, Any] | None = None,
) -> Path:
    """Write a replay bundle under ``<root_dir>/<run_id>/`` after semantic validation."""

    run_id = manifest.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("manifest run_id must be a non-empty string")
    validated_run_id_path = run_id_path(run_id, subject="manifest run_id")

    _raise_for_schema_issues(
        manifest,
        raw_events,
        layout,
        derived_events=derived_events,
        motif_ledger=motif_ledger,
        contingency=contingency,
    )
    _raise_for_semantic_issues(
        manifest,
        raw_events,
        layout,
        derived_events=derived_events,
        motif_ledger=motif_ledger,
        contingency=contingency,
    )

    root_path = Path(root_dir)
    root_path.mkdir(parents=True, exist_ok=True)

    run_dir = root_path / validated_run_id_path
    if run_dir.exists():
        raise FileExistsError(f"run bundle directory already exists: {run_dir}")

    temp_run_dir = Path(tempfile.mkdtemp(prefix=".run-bundle.tmp-", dir=root_path))
    try:
        _write_json(temp_run_dir / "manifest.json", manifest)
        _write_ndjson(temp_run_dir / "raw_trace.ndjson", raw_events)
        _write_json(temp_run_dir / "layout.json", layout)

        if derived_events is not None:
            _write_ndjson(temp_run_dir / "derived.ndjson", derived_events)
        if motif_ledger is not None:
            _write_json(temp_run_dir / "motif_ledger.json", motif_ledger)
        if contingency is not None:
            _write_json(temp_run_dir / "contingency.json", contingency)

        run_dir.parent.mkdir(parents=True, exist_ok=True)
        temp_run_dir.rename(run_dir)
    except Exception:
        shutil.rmtree(temp_run_dir, ignore_errors=True)
        raise

    return run_dir


def load_run_bundle(run_dir: str | Path) -> dict[str, Any]:
    """Load and validate a replay bundle from disk."""

    run_path = Path(run_dir)

    derived_path = run_path / "derived.ndjson"
    motif_ledger_path = run_path / "motif_ledger.json"
    contingency_path = run_path / "contingency.json"

    bundle = {
        "manifest": _read_json(run_path / "manifest.json"),
        "raw_events": _read_ndjson(run_path / "raw_trace.ndjson"),
        "layout": _read_json(run_path / "layout.json"),
        "derived_events": _read_ndjson(derived_path) if derived_path.exists() else None,
        "motif_ledger": _read_json(motif_ledger_path) if motif_ledger_path.exists() else None,
        "contingency": _read_json(contingency_path) if contingency_path.exists() else None,
    }

    _raise_for_schema_issues(
        bundle["manifest"],
        bundle["raw_events"],
        bundle["layout"],
        derived_events=bundle["derived_events"],
        motif_ledger=bundle["motif_ledger"],
        contingency=bundle["contingency"],
    )
    _raise_for_semantic_issues(
        bundle["manifest"],
        bundle["raw_events"],
        bundle["layout"],
        derived_events=bundle["derived_events"],
        motif_ledger=bundle["motif_ledger"],
        contingency=bundle["contingency"],
    )

    return bundle

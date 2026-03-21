"""Helpers for validating inferoscope run identifiers."""

from __future__ import annotations

from pathlib import Path, PureWindowsPath


def run_id_path(run_id: str) -> Path:
    """Return the canonical relative filesystem path for a run_id."""

    if "\\" in run_id:
        raise ValueError("manifest run_id must not contain backslashes.")

    windows_path = PureWindowsPath(run_id)
    if windows_path.drive or windows_path.root:
        raise ValueError("manifest run_id must be a relative path without a Windows drive or root.")

    raw_parts = run_id.split("/")
    if run_id.startswith("/") or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError("manifest run_id must be a canonical relative path with non-empty segments.")

    return Path(*raw_parts)

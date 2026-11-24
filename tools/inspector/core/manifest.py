"""Versioned portable manifest helpers."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_VERSION = 1


class ManifestVersionError(ValueError):
    pass


def new_manifest(scan_id: str, artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "manifest_version": MANIFEST_VERSION,
        "scan_id": scan_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def validate_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Manifest must be a JSON object")
    version = value.get("manifest_version")
    if version != MANIFEST_VERSION:
        raise ManifestVersionError(f"Unsupported manifest_version={version!r}; supported: {MANIFEST_VERSION}")
    if not isinstance(value.get("artifacts", []), list):
        raise ValueError("Manifest 'artifacts' must be a list")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_html_atomic(path: Path, value: Any) -> None:
    """Write a self-contained, inert HTML report (no scripts or remote assets)."""
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    escaped = html.escape(payload)
    document = ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
                "<title>Thesis Inspector report</title></head><body>"
                "<main><h1>Thesis Inspector report</h1><p>Read-only scan snapshot.</p>"
                f"<pre>{escaped}</pre></main></body></html>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(path)

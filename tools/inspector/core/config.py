"""Configuration loading with paths anchored to the repository root."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InputRoot:
    name: str
    path: Path


@dataclass(frozen=True)
class InspectorConfig:
    repo_root: Path
    title: str
    host: str
    port: int
    open_browser: bool
    read_only: bool
    roots: tuple[InputRoot, ...]
    output_dir: Path
    ignore_hidden: bool = True
    max_sample_size: int = 10_000
    max_files: int = 25_000
    max_json_bytes: int = 8_000_000
    max_npz_uncompressed_bytes: int = 128_000_000
    max_series_bytes: int = 250_000_000
    hash_small_files_under_bytes: int = 1_000_000


def _resolve(repo_root: Path, value: str) -> Path:
    candidate = Path(value).expanduser()
    return (candidate if candidate.is_absolute() else repo_root / candidate).resolve()


def load_config(config_path: Path, repo_root: Path) -> InspectorConfig:
    """Load TOML; all relative paths are interpreted from ``repo_root``."""
    config_path = config_path.resolve()
    with config_path.open("rb") as stream:
        raw = tomllib.load(stream)

    app: dict[str, Any] = raw.get("inspector", {})
    scan: dict[str, Any] = raw.get("scan", {})
    paths: dict[str, Any] = raw.get("paths", {})
    reporting: dict[str, Any] = raw.get("reporting", {})

    host = str(app.get("host", "127.0.0.1"))
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Thesis Inspector only supports loopback binding (127.0.0.1).")
    port = int(app.get("port", 8765))
    if not 0 <= port <= 65535:
        raise ValueError("port must be between 0 and 65535")

    roots: list[InputRoot] = []
    for key in ("data", "checkpoints", "reports", "platform_data"):
        value = paths.get(key)
        if value:
            roots.append(InputRoot(key, _resolve(repo_root, str(value))))
    for index, item in enumerate(paths.get("external_roots", [])):
        if not isinstance(item, dict) or not item.get("path"):
            raise ValueError(f"paths.external_roots[{index}] must contain a path")
        roots.append(InputRoot(str(item.get("name", f"external_{index + 1}")), _resolve(repo_root, str(item["path"]))))

    if not roots:
        roots = [InputRoot("repository", repo_root.resolve())]
    names = [root.name for root in roots]
    if len(names) != len(set(names)):
        raise ValueError("Input root names must be unique")

    output_dir = _resolve(repo_root, str(reporting.get("output_dir", "reports/inspector")))
    if output_dir == repo_root.resolve():
        raise ValueError("Inspector output directory cannot be the repository root")
    for root in roots:
        resolved_root = root.path.resolve(strict=False)
        if output_dir == resolved_root or (output_dir in resolved_root.parents):
            raise ValueError(f"Inspector output directory cannot contain or replace input root '{root.name}'")
        if resolved_root in output_dir.parents:
            allowed_reports_cache = root.name == "reports" and output_dir.parent == resolved_root and output_dir.name == "inspector"
            if not allowed_reports_cache:
                raise ValueError(f"Inspector output directory overlaps input root '{root.name}'")
    return InspectorConfig(
        repo_root=repo_root.resolve(),
        title=str(app.get("title", "Thesis Inspector")),
        host="127.0.0.1",
        port=port,
        open_browser=bool(app.get("open_browser", True)),
        read_only=bool(app.get("read_only", True)),
        roots=tuple(roots),
        output_dir=output_dir,
        ignore_hidden=bool(scan.get("ignore_hidden", True)),
        max_sample_size=max(1, int(scan.get("max_sample_size", 10_000))),
        max_files=max(1, int(scan.get("max_files", 25_000))),
        max_json_bytes=max(1024, int(scan.get("max_json_bytes", 8_000_000))),
        max_npz_uncompressed_bytes=max(1024, int(scan.get("max_npz_uncompressed_bytes", 128_000_000))),
        max_series_bytes=max(1024, int(scan.get("max_series_bytes", 250_000_000))),
        hash_small_files_under_bytes=max(0, int(scan.get("hash_small_files_under_bytes", 1_000_000))),
    )

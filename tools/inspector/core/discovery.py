"""Bounded discovery of supported research artifacts."""
from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path
from typing import Iterator

from .config import InputRoot, InspectorConfig
from .security import assert_within, is_within

SUPPORTED_SUFFIXES = {
    ".json", ".jsonl", ".csv", ".npz", ".parquet", ".geojson", ".tif", ".tiff",
    ".pt", ".safetensors", ".yaml", ".yml", ".toml", ".md", ".txt", ".zarr",
}
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".cache", "target", "dist", "build"}
ZARR_MARKERS = {".zgroup", "zarr.json", ".zmetadata"}


def detect_format(path: Path, is_directory: bool = False) -> str:
    if is_directory:
        return "zarr"
    ext = path.suffix.lower()
    return {
        ".json": "json", ".jsonl": "jsonl", ".csv": "csv", ".npz": "npz",
        ".parquet": "parquet", ".geojson": "geojson", ".tif": "geotiff",
        ".tiff": "geotiff", ".pt": "checkpoint", ".safetensors": "safetensors",
        ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".md": "text", ".txt": "text",
    }.get(ext, "unknown")


def classify_role(path: Path) -> tuple[str, str]:
    text = "/" + "/".join(part.lower() for part in path.parts) + "/"
    name = path.name.lower()
    if "platform/src/data" in text or "/platform_data/" in text:
        return "platform_export", "path"
    if path.suffix.lower() in {".pt", ".safetensors"} or "/checkpoints/" in text:
        return "checkpoint", "extension_or_path"
    if "/eval" in text or "/reports/dev/" in text or "metrics" in name or "dev_log" in name:
        return "evaluation", "path_or_name"
    if "validation" in text or "quality" in name or "audit" in name:
        return "validation", "path_or_name"
    if "phenology" in text or "cycle" in name or "phase" in name:
        return "phenology", "path_or_name"
    if "baseline" in text or name.startswith(("b0", "b1")):
        return "baseline", "path_or_name"
    if "preprocess" in text or "interim" in text or "ingest" in text:
        return "preprocessing", "path"
    if any(token in text for token in ("/models/", "/train/", "/ssl/")):
        return "ssl_training", "path"
    if "/data/" in text or path.suffix.lower() in {".zarr", ".parquet", ".npz", ".geojson", ".tif", ".tiff"}:
        return "dataset", "path_or_format"
    return "unknown", "no_matching_rule"


def _is_zarr(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any((path / marker).is_file() for marker in ZARR_MARKERS)
    except OSError:
        return False


def _fingerprint(path: Path, root: InputRoot, hash_limit: int) -> tuple[str, int, int]:
    stat = path.stat()
    size = stat.st_size
    mtime_ns = stat.st_mtime_ns
    rel = path.relative_to(root.path).as_posix()
    digest = hashlib.sha256(f"{root.name}\0{rel}\0{size}\0{mtime_ns}".encode()).hexdigest()
    if path.is_file() and size <= hash_limit:
        try:
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            digest = hashlib.sha256(f"{digest}\0{content_hash}".encode()).hexdigest()
        except OSError:
            pass
    elif path.is_dir():
        # Fingerprint only bounded Zarr metadata, never array chunks.
        metadata: list[tuple[str, int, int, str]] = []
        try:
            for current, dirs, files in os.walk(path, followlinks=False):
                dirs[:] = [name for name in dirs if name not in SKIP_DIRS and not name.startswith(".")]
                for name in files:
                    if name in ZARR_MARKERS or name == ".zattrs":
                        item = Path(current) / name
                        try:
                            item_stat = item.stat()
                            content_hash = hashlib.sha256(item.read_bytes()).hexdigest() if item_stat.st_size <= 64_000 else ""
                            metadata.append((item.relative_to(path).as_posix(), item_stat.st_size, item_stat.st_mtime_ns, content_hash))
                        except OSError:
                            continue
                if len(metadata) >= 256:
                    break
        except OSError:
            pass
        digest = hashlib.sha256((digest + repr(sorted(metadata))).encode()).hexdigest()
    return digest, size, mtime_ns


def iter_artifacts(config: InspectorConfig) -> Iterator[tuple[InputRoot, Path, bool]]:
    emitted = 0
    output_dir = config.output_dir.resolve(strict=False)
    for root in config.roots:
        base = root.path.resolve(strict=False)
        if not base.exists() or not base.is_dir():
            continue
        if is_within(base, output_dir) or is_within(output_dir, base):
            # A reports root may include the Inspector's output subtree; skip it below.
            pass
        stack = [base]
        while stack and emitted < config.max_files:
            current = stack.pop()
            try:
                if current == output_dir or is_within(current, output_dir):
                    continue
                assert_within(current, base)
                with os.scandir(current) as scan:
                    entries = sorted(scan, key=lambda entry: entry.name.casefold(), reverse=True)
            except (OSError, PermissionError):
                continue
            for entry in entries:
                if config.ignore_hidden and entry.name.startswith("."):
                    continue
                if entry.is_symlink():
                    continue
                path = Path(entry.path)
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name.lower() in SKIP_DIRS:
                            continue
                        if _is_zarr(path):
                            assert_within(path, base)
                            yield root, path, True
                            emitted += 1
                        else:
                            stack.append(path)
                    elif entry.is_file(follow_symlinks=False) and path.suffix.lower() in SUPPORTED_SUFFIXES:
                        assert_within(path, base)
                        yield root, path, False
                        emitted += 1
                except OSError:
                    continue
                if emitted >= config.max_files:
                    break


def make_record(root: InputRoot, path: Path, is_directory: bool, hash_limit: int) -> dict:
    fingerprint, size, mtime_ns = _fingerprint(path, root, hash_limit)
    relative = path.relative_to(root.path).as_posix()
    artifact_id = hashlib.sha256(f"{root.name}\0{relative}".encode()).hexdigest()[:24]
    role, classification_reason = classify_role(path)
    return {
        "id": artifact_id,
        "root": root.name,
        "relative_path": relative,
        "display_name": path.name,
        "format": detect_format(path, is_directory),
        "role": role,
        "classification_reason": classification_reason,
        "is_directory": is_directory,
        "size_bytes": size,
        "mtime_ns": mtime_ns,
        "fingerprint": fingerprint,
        "status": "DISCOVERED",
        "schema": None,
        "summary": {},
        "issues": [],
    }

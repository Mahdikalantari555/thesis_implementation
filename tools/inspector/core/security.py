"""Path confinement and conservative split-access controls."""
from __future__ import annotations

from pathlib import Path


LOCKED_TOKENS = {"test", "locked_test", "final_test", "test_only"}
DATA_EXTENSIONS = {".parquet", ".npz", ".zarr", ".tif", ".tiff", ".csv", ".geojson"}
SAFE_METADATA_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".md"}


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
        return True
    except ValueError:
        return False


def assert_within(path: Path, root: Path) -> Path:
    resolved, base = path.resolve(strict=False), root.resolve(strict=False)
    if not is_within(resolved, base):
        raise PermissionError(f"Path escapes configured root: {path}")
    return resolved


def access_restriction(path: Path, root: Path, is_directory_artifact: bool = False) -> tuple[str | None, str | None]:
    """Return (status, reason); unknown data splits are metadata-only by default."""
    parts = [part.lower() for part in path.parts]
    if any(part in LOCKED_TOKENS or part.startswith("test-") or part.endswith("-test") for part in parts):
        return "RESTRICTED", "Path is explicitly identified as locked-test material"
    try:
        relative_parts = [part.lower() for part in path.resolve(strict=False).relative_to(root.resolve(strict=False)).parts]
    except ValueError:
        return "RESTRICTED", "Path is outside the configured root"
    name = path.name.lower()
    is_manifest = any(token in name for token in ("manifest", "splits_locked", "registry", "schema"))
    # Dataset trees commonly contain combined train/dev/test payloads; JSON is
    # not automatically safe just because it is a metadata-friendly format.
    in_data_tree = any(token in {"data", "processed", "interim", "raw", "external"} for token in relative_parts)
    if in_data_tree and not is_manifest:
        if "train" in relative_parts or "dev" in relative_parts or "validation" in relative_parts:
            return None, None
        if path.suffix.lower() in DATA_EXTENSIONS | SAFE_METADATA_SUFFIXES or is_directory_artifact:
            return "RESTRICTED/UNKNOWN_SPLIT", "Split membership is not established; metadata-only inspection"
    if path.suffix.lower() in SAFE_METADATA_SUFFIXES:
        return None, None
    # Dataset stores outside the canonical data tree also need a declared split.
    if path.suffix.lower() in DATA_EXTENSIONS or is_directory_artifact:
        if "train" in relative_parts or "dev" in relative_parts or "validation" in relative_parts:
            return None, None
        return "RESTRICTED/UNKNOWN_SPLIT", "Split membership is not established; metadata-only inspection"
    return None, None

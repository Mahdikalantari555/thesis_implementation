"""Bounded, read-only adapters. Optional readers are imported only on demand."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import os
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any


def _registry(repo_root: Path) -> dict[str, tuple[float, float]]:
    path = repo_root / "scripts" / "platform_channels.py"
    if not path.is_file():
        return {}
    try:
        spec = importlib.util.spec_from_file_location("inspector_platform_channels", path)
        if spec is None or spec.loader is None:
            return {}
        module = importlib.util.module_from_spec(spec)
        import sys
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return {item.name: tuple(item.physical_range) for item in module.CHANNEL_REGISTRY}
    except Exception:
        return {}


def _numeric_stats(values: list[Any], physical_range: tuple[float, float] | None = None) -> dict[str, Any]:
    numeric: list[float] = []
    missing = nonfinite = 0
    outside = 0
    for value in values:
        if value is None or value == "":
            missing += 1
            continue
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(number):
            nonfinite += 1
            continue
        numeric.append(number)
        if physical_range and not (physical_range[0] <= number <= physical_range[1]):
            outside += 1
    result: dict[str, Any] = {"count": len(values), "numeric_count": len(numeric), "missing_count": missing, "nonfinite_count": nonfinite}
    if numeric:
        result.update({"min": min(numeric), "max": max(numeric), "mean": sum(numeric) / len(numeric)})
    if physical_range:
        result.update({"physical_range": list(physical_range), "out_of_range_count": outside})
    return result


def _compact_json(value: Any, depth: int = 0) -> Any:
    if depth > 4:
        return "…"
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _compact_json(item, depth + 1) for key, item in list(value.items())[:100]}
    if isinstance(value, list):
        return [_compact_json(item, depth + 1) for item in value[:20]]
    return str(value)


def _json(path: Path, config: Any) -> dict[str, Any]:
    size = path.stat().st_size
    if size > config.max_json_bytes:
        return {"status": "UNSUPPORTED", "message": f"JSON exceeds configured read limit ({config.max_json_bytes} bytes)", "summary": {"size_bytes": size}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {"status": "INCOMPLETE", "message": f"JSON could not be parsed: {exc.__class__.__name__}: {exc}"}
    if isinstance(data, dict):
        keys = list(data.keys())[:200]
        summary: dict[str, Any] = {"top_level_type": "object", "keys": keys, "key_count": len(data)}
        schema = data.get("schema_version", data.get("schemaVersion", data.get("version")))
        if schema is not None and isinstance(schema, (str, int, float)):
            summary["declared_schema"] = schema
        context_keys = ("split", "data_split", "evaluation_split", "protocol", "run_id", "model", "model_id", "scorer", "status", "claim_rung", "ladder_rung", "evidence_class", "claim_noun", "data_source", "seeds", "git_sha", "splits_locked_sha256", "split_hash", "eval_subset_sha256", "eval_subset_hash", "cycle_file_sha256", "data_hash", "test_only", "locked_test")
        for key in context_keys:
            if key in data:
                summary[key] = _compact_json(data[key])
        summary["contains_test_only"] = bool(data.get("test_only") is True or data.get("locked_test") is True or str(data.get("split", data.get("data_split", ""))).lower() in {"test", "locked_test", "final_test", "test_only"})
        summary["sample"] = {key: _compact_json(data[key]) for key in keys[:24] if isinstance(data[key], (str, int, float, bool)) or data[key] is None}
        metric_keys = ("metrics", "mdm80", "detection_rate", "detection_rates", "event_far", "auprc", "auc_pr", "roc_auc", "auroc", "nrmse", "partial_cycle_auprc", "generalization", "pretraining_quality", "decision_gates", "gate_results", "status_counts", "registry", "experiments")
        recorded = {key: _compact_json(data[key]) for key in metric_keys if key in data}
        for key, value in data.items():
            if key.lower() in {"auprc", "auc_pr", "roc_auc", "auroc", "event_far", "nrmse", "detection_rate", "mdm80"} and key not in recorded:
                recorded[key] = _compact_json(value)
        if recorded:
            summary["recorded_metrics"] = recorded
        if isinstance(data.get("artifacts"), list):
            summary["artifact_count"] = len(data["artifacts"])
        if isinstance(data.get("features"), list):
            summary["feature_count"] = len(data["features"])
        if isinstance(data.get("scenes"), list):
            summary["scene_count"] = len(data["scenes"])
        status = "OK"
        if data.get("status") in {"Pending", "Running", "Completed", "Failed", "Interrupted", "Deferred", "Not Applicable"}:
            summary["registry_status"] = data["status"]
        return {"status": status, "schema": str(schema) if schema is not None else None, "summary": summary}
    if isinstance(data, list):
        return {"status": "OK", "summary": {"top_level_type": "array", "item_count": len(data), "sample": data[:min(len(data), 20)]}}
    return {"status": "OK", "summary": {"top_level_type": type(data).__name__, "value": data if isinstance(data, (str, int, float, bool)) or data is None else None}}


def _csv(path: Path, config: Any) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            rows = []
            row_count = 0
            sampled_limit_reached = False
            for row in reader:
                if len(rows) >= config.max_sample_size:
                    sampled_limit_reached = True
                    break
                rows.append(row)
                row_count += 1
    except (OSError, UnicodeError, csv.Error) as exc:
        return {"status": "INCOMPLETE", "message": f"CSV could not be read: {exc.__class__.__name__}: {exc}"}
    ranges = _registry(config.repo_root)
    columns = {}
    for header in headers:
        values = [row.get(header) for row in rows]
        stats = _numeric_stats(values, ranges.get(header.lower()))
        missing_text = sum(value is None or str(value).strip() == "" for value in values)
        stats["missing_count"] = missing_text
        columns[header] = stats
    return {"status": "OK", "summary": {"headers": headers, "row_count": row_count if not sampled_limit_reached else None, "rows_sampled": len(rows), "row_count_exact": not sampled_limit_reached, "is_sampled": sampled_limit_reached, "channel_registry_available": bool(ranges), "columns": columns, "sample": rows[:min(20, len(rows))]}}


def _npz(path: Path, config: Any) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            total = sum(item.file_size for item in archive.infolist())
            members = archive.namelist()
    except (OSError, zipfile.BadZipFile) as exc:
        return {"status": "INCOMPLETE", "message": f"NPZ container is invalid: {exc.__class__.__name__}: {exc}"}
    if total > config.max_npz_uncompressed_bytes:
        return {"status": "UNSUPPORTED", "message": "NPZ exceeds configured uncompressed-size limit", "summary": {"members": members[:100], "uncompressed_bytes": total}}
    try:
        import numpy as np
    except ImportError:
        return {"status": "UNSUPPORTED", "message": "NumPy is required to inspect NPZ arrays", "summary": {"members": members[:100], "uncompressed_bytes": total}}
    arrays = {}
    try:
        with np.load(path, allow_pickle=False) as bundle:
            for name in bundle.files[:100]:
                array = bundle[name]
                flat = array.reshape(-1)
                sample = flat[:config.max_sample_size]
                numeric = sample.astype("float64", copy=False) if sample.dtype.kind in "iuf" else None
                valid = numeric[np.isfinite(numeric)] if numeric is not None else None
                arrays[name] = {
                    "shape": list(array.shape), "dtype": str(array.dtype), "sampled_values": int(sample.size),
                    "nan_count_sample": int(np.isnan(numeric).sum()) if numeric is not None else 0,
                    "inf_count_sample": int(np.isinf(numeric).sum()) if numeric is not None else 0,
                    "min_sample": float(valid.min()) if valid is not None and valid.size else None,
                    "max_sample": float(valid.max()) if valid is not None and valid.size else None,
                    "is_sampled": flat.size > sample.size,
                }
    except Exception as exc:
        return {"status": "INCOMPLETE", "message": f"NPZ arrays could not be read safely: {exc.__class__.__name__}: {exc}"}
    return {"status": "OK", "summary": {"arrays": arrays, "uncompressed_bytes": total}}


def _parquet(path: Path, config: Any) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return {"status": "UNSUPPORTED", "message": "Install the optional pyarrow dependency to inspect Parquet", "summary": {}}
    try:
        file = pq.ParquetFile(path)
        metadata = file.metadata
        batch = next(file.iter_batches(batch_size=config.max_sample_size, columns=file.schema_arrow.names[:100], use_threads=False), None)
        sample_rows = batch.to_pylist()[:20] if batch is not None else []
        sampled_rows = batch.num_rows if batch is not None else 0
        return {"status": "OK", "summary": {
            "row_count": metadata.num_rows, "row_groups": metadata.num_row_groups,
            "schema": str(file.schema_arrow), "columns": file.schema_arrow.names,
            "sampled_rows": sampled_rows,
            "is_sampled": metadata.num_rows > sampled_rows,
            "sample": sample_rows,
        }}
    except Exception as exc:
        return {"status": "INCOMPLETE", "message": f"Parquet could not be inspected: {exc.__class__.__name__}: {exc}"}


def _zarr(path: Path, config: Any) -> dict[str, Any]:
    try:
        import zarr
    except ImportError:
        return {"status": "UNSUPPORTED", "message": "Install the optional zarr dependency to inspect this store", "summary": {}}
    try:
        group = zarr.open_group(str(path), mode="r")
        arrays: dict[str, Any] = {}
        names: list[str] = []
        def visit(name: str, obj: Any) -> None:
            if hasattr(obj, "shape") and hasattr(obj, "dtype"):
                names.append(name)
                shape = tuple(int(dim) for dim in obj.shape)
                total = math.prod(shape) if shape else 1
                slices = []
                budget = max(1, config.max_sample_size)
                for dim in shape:
                    take = min(dim, budget)
                    slices.append(slice(0, take))
                    budget = max(1, budget // max(1, take))
                stats: dict[str, Any] = {"shape": list(shape), "dtype": str(obj.dtype), "chunks": list(obj.chunks) if getattr(obj, "chunks", None) else None}
                if shape:
                    try:
                        data = obj[tuple(slices)]
                        stats["sampled_values"] = int(data.size)
                        stats["is_sampled"] = total > int(data.size)
                        import numpy as np
                        if data.dtype.kind in "iuf":
                            stats.update({"nan_count_sample": int(np.isnan(data).sum()), "inf_count_sample": int(np.isinf(data).sum()), "min_sample": float(np.nanmin(data)) if data.size and np.isfinite(data).any() else None, "max_sample": float(np.nanmax(data)) if data.size and np.isfinite(data).any() else None})
                    except Exception as exc:
                        stats["sample_error"] = exc.__class__.__name__
                else:
                    stats["sampled_values"] = 0
                    stats["is_sampled"] = total > 0
                arrays[name] = stats
        group.visititems(visit)
        return {"status": "OK", "summary": {"array_count": len(names), "arrays": arrays}}
    except Exception as exc:
        return {"status": "INCOMPLETE", "message": f"Zarr store could not be opened: {exc.__class__.__name__}: {exc}"}


def _geojson(path: Path, config: Any) -> dict[str, Any]:
    result = _json(path, config)
    if result.get("status") != "OK":
        return result
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        features = data.get("features", []) if isinstance(data, dict) else []
        geometry_types = Counter()
        missing_geometry = 0
        invalid_coords = 0
        invalid_geometry = 0
        ids: list[str] = []
        for feature in features[:config.max_sample_size]:
            if isinstance(feature, dict) and feature.get("id") is not None:
                ids.append(str(feature["id"]))
            geometry = feature.get("geometry") if isinstance(feature, dict) else None
            if not geometry:
                missing_geometry += 1
                continue
            geometry_type = str(geometry.get("type", "unknown"))
            geometry_types[geometry_type] += 1
            coords = geometry.get("coordinates")
            if not _valid_coordinates(coords):
                invalid_coords += 1
            if not _valid_geometry_structure(geometry_type, coords):
                invalid_geometry += 1
        crs = data.get("crs")
        crs_text = json.dumps(crs, ensure_ascii=False).lower() if crs is not None else ""
        wgs84 = crs is None or any(token in crs_text for token in ("4326", "crs84", "wgs 84", "wgs84"))
        duplicate_ids = len(ids) - len(set(ids))
        result["summary"].update({"feature_count": len(features), "geometry_types": dict(geometry_types), "features_sampled": min(len(features), config.max_sample_size), "missing_geometry_sample": missing_geometry, "invalid_coordinate_feature_sample": invalid_coords, "invalid_geometry_sample": invalid_geometry, "duplicate_ids_sample": duplicate_ids, "crs": crs if crs is not None else "WGS84 (RFC 7946 default)", "crs_is_wgs84": wgs84})
    except Exception as exc:
        return {"status": "INCOMPLETE", "message": f"GeoJSON could not be validated: {exc.__class__.__name__}: {exc}"}
    return result


def _valid_geometry_structure(geometry_type: str, coordinates: Any) -> bool:
    def valid_ring(ring: Any) -> bool:
        return isinstance(ring, list) and len(ring) >= 4 and ring[0] == ring[-1] and all(_valid_coordinates(point) for point in ring)
    if geometry_type == "Point":
        return _valid_coordinates(coordinates)
    if geometry_type == "MultiPoint":
        return isinstance(coordinates, list) and bool(coordinates) and all(_valid_coordinates(point) for point in coordinates)
    if geometry_type == "LineString":
        return isinstance(coordinates, list) and len(coordinates) >= 2 and all(_valid_coordinates(point) for point in coordinates)
    if geometry_type == "MultiLineString":
        return isinstance(coordinates, list) and bool(coordinates) and all(isinstance(line, list) and len(line) >= 2 and all(_valid_coordinates(point) for point in line) for line in coordinates)
    if geometry_type == "Polygon":
        return isinstance(coordinates, list) and bool(coordinates) and all(valid_ring(ring) for ring in coordinates)
    if geometry_type == "MultiPolygon":
        return isinstance(coordinates, list) and bool(coordinates) and all(isinstance(poly, list) and bool(poly) and all(valid_ring(ring) for ring in poly) for poly in coordinates)
    return False


def _valid_coordinates(value: Any, depth: int = 0) -> bool:
    if depth > 8:
        return False
    if isinstance(value, (tuple, list)):
        if len(value) >= 2 and all(isinstance(v, (int, float)) and math.isfinite(float(v)) for v in value[:2]):
            return -180 <= float(value[0]) <= 180 and -90 <= float(value[1]) <= 90
        return bool(value) and all(_valid_coordinates(item, depth + 1) for item in value)
    return False


def _geotiff(path: Path) -> dict[str, Any]:
    try:
        import rasterio
    except ImportError:
        return {"status": "UNSUPPORTED", "message": "Install the optional rasterio dependency to inspect GeoTIFF", "summary": {}}
    try:
        with rasterio.open(path, "r") as ds:
            return {"status": "OK", "summary": {"width": ds.width, "height": ds.height, "count": ds.count, "crs": str(ds.crs) if ds.crs else None, "transform": list(ds.transform)[:6], "nodata": ds.nodata, "dtype": list(ds.dtypes), "bounds": list(ds.bounds), "resolution": list(ds.res)}}
    except Exception as exc:
        return {"status": "INCOMPLETE", "message": f"GeoTIFF metadata unavailable: {exc.__class__.__name__}: {exc}"}


def _checkpoint(path: Path) -> dict[str, Any]:
    sidecars = [path.with_suffix(path.suffix + ".json"), path.with_suffix(".json"), path.parent / (path.stem + ".json")]
    sidecar = next((candidate for candidate in sidecars if candidate.is_file()), None)
    summary = {"extension": path.suffix.lower(), "deserialization": "not performed", "sidecar_path": sidecar.name if sidecar else None}
    if sidecar and sidecar.stat().st_size <= 2_000_000:
        try:
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(metadata, dict):
                summary["sidecar_keys"] = list(metadata)[:100]
                summary["sidecar_metadata"] = {key: value for key, value in metadata.items() if isinstance(value, (str, int, float, bool)) or value is None}
        except (OSError, UnicodeError, json.JSONDecodeError):
            summary["sidecar_status"] = "invalid"
    return {"status": "OK" if sidecar else "WARNING", "summary": summary, "message": None if sidecar else "No JSON sidecar found; weights were not loaded"}


def inspect_artifact(path: Path, artifact_format: str, config: Any) -> dict[str, Any]:
    """Return a JSON-safe inspection record without mutating the artifact."""
    try:
        if artifact_format in {"json", "yaml", "toml", "text", "jsonl"}:
            if artifact_format == "json":
                return _json(path, config)
            if path.stat().st_size > config.max_json_bytes:
                return {"status": "UNSUPPORTED", "message": "Text file exceeds configured read limit", "summary": {"size_bytes": path.stat().st_size}}
            text = path.read_text(encoding="utf-8", errors="replace")
            if artifact_format == "jsonl":
                lines = text.splitlines()
                parsed = 0
                for line in lines[:min(config.max_sample_size, 1000)]:
                    if line.strip():
                        json.loads(line)
                        parsed += 1
                return {"status": "OK", "summary": {"line_count": len(lines), "sampled_valid_json_lines": parsed, "is_sampled": len(lines) > parsed}}
            return {"status": "OK", "summary": {"line_count": text.count("\n") + bool(text), "preview": text[:2000]}}
        if artifact_format == "csv":
            return _csv(path, config)
        if artifact_format == "npz":
            return _npz(path, config)
        if artifact_format == "parquet":
            return _parquet(path, config)
        if artifact_format == "zarr":
            return _zarr(path, config)
        if artifact_format == "geojson":
            return _geojson(path, config)
        if artifact_format == "geotiff":
            return _geotiff(path)
        if artifact_format in {"checkpoint", "safetensors"}:
            return _checkpoint(path)
        return {"status": "UNSUPPORTED", "message": f"No reader registered for {artifact_format}"}
    except Exception as exc:
        return {"status": "ERROR", "message": f"Adapter failed safely: {exc.__class__.__name__}: {exc}"}

"""Manual, incremental artifact scanner with immutable report snapshots."""
from __future__ import annotations

import hashlib
import json
import math
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..adapters import inspect_artifact
from ..evaluation import evaluation_studies
from ..validation.data_quality import build_quality_report
from .cache import Cache
from .config import InspectorConfig
from .discovery import _fingerprint, iter_artifacts, make_record
from .integration import validate_platform
from .manifest import new_manifest, write_html_atomic, write_json_atomic
from .security import access_restriction


class ScanInProgress(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class Scanner:
    def __init__(self, config: InspectorConfig):
        self.config = config
        self.output_dir = config.output_dir.resolve(strict=False)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache = Cache(self.output_dir / "inspector.sqlite")
        self._lock = threading.Lock()
        self.last_scan: dict[str, Any] | None = None

    def status(self) -> dict[str, Any]:
        scans = self.cache.list_scans()
        last = scans[0] if scans else None
        artifacts = self.cache.list_artifacts(last["scan_id"] if last else None)
        return {
            "title": self.config.title,
            "local": True,
            "read_only": self.config.read_only,
            "scan_in_progress": self._lock.locked(),
            "last_scan": last,
            "artifact_count": len(artifacts),
            "roots": [{"name": root.name, "configured": True, "exists": root.path.is_dir()} for root in self.config.roots],
            "manifest_version": 1,
        }

    def scan(self) -> dict[str, Any]:
        if not self._lock.acquire(blocking=False):
            raise ScanInProgress("A refresh is already running")
        try:
            return self._scan_locked()
        finally:
            self._lock.release()

    def _scan_locked(self) -> dict[str, Any]:
        scan_id = datetime.now(timezone.utc).strftime("scan_%Y%m%dT%H%M%S_%fZ_") + uuid.uuid4().hex[:6]
        created_at = datetime.now(timezone.utc).isoformat()
        records: list[dict[str, Any]] = []
        changed = 0
        error_count = 0
        for root, path, is_directory in iter_artifacts(self.config):
            try:
                record = make_record(root, path, is_directory, self.config.hash_small_files_under_bytes)
                restriction, reason = access_restriction(path, root.path, is_directory)
                if restriction:
                    record["status"] = restriction
                    record["message"] = reason
                    record["access"] = "metadata_only"
                    record["summary"] = {"size_bytes": record["size_bytes"], "mtime_ns": record["mtime_ns"]}
                    cached = self.cache.get_by_fingerprint(record["id"], record["fingerprint"])
                    # Restricted records never reuse previously sampled values.
                    changed += int(cached is None)
                else:
                    cached = self.cache.get_by_fingerprint(record["id"], record["fingerprint"])
                    if cached is not None:
                        record = cached
                    else:
                        changed += 1
                        before = (record["fingerprint"], record["size_bytes"], record["mtime_ns"])
                        result = inspect_artifact(path, record["format"], self.config)
                        record["status"] = result.get("status", "ERROR")
                        record["message"] = result.get("message")
                        record["schema"] = result.get("schema")
                        record["summary"] = result.get("summary", {})
                        record["issues"] = result.get("issues", [])
                        record["access"] = "value_inspection"
                        split = self._declared_split(record["summary"])
                        report_path = record["relative_path"].lower()
                        report_lineage_needs_split = record["root"] == "reports" and report_path.startswith(("architecture_benchmark/", "regional_representation/"))
                        allowed_splits = {"dev", "development", "validation", "train+dev", "train_dev", "train-and-dev"}
                        if split in {"test", "locked_test", "final_test"}:
                            record["status"] = "RESTRICTED"
                            record["message"] = "Artifact declares locked-test content; values excluded from reports"
                            record["access"] = "metadata_only"
                            record["summary"] = {"declared_split": split, "size_bytes": record["size_bytes"], "mtime_ns": record["mtime_ns"]}
                        elif report_lineage_needs_split and split not in allowed_splits:
                            original = record["summary"]
                            record["status"] = "RESTRICTED/UNKNOWN_SPLIT"
                            record["message"] = "Benchmark/capstone report has no declared allowed development split; metric values are metadata-only"
                            record["access"] = "metadata_only"
                            record["summary"] = {key: original[key] for key in ("top_level_type", "declared_schema", "keys", "protocol", "run_id", "status", "registry_status", "split", "data_split") if key in original}
                        try:
                            after = _fingerprint(path, root, self.config.hash_small_files_under_bytes)
                            if before != after:
                                record["status"] = "INCOMPLETE"
                                record["message"] = "Artifact changed while being inspected; refresh to retry"
                                record["summary"] = {"size_bytes": record["size_bytes"], "mtime_ns": record["mtime_ns"]}
                        except OSError:
                            record["status"] = "INCOMPLETE"
                            record["message"] = "Artifact disappeared while being inspected"
                if record["status"] in {"ERROR", "INCOMPLETE"}:
                    error_count += 1
                record = _json_safe(record)
                self.cache.put_artifact(record, scan_id)
                records.append(record)
            except (OSError, PermissionError, ValueError) as exc:
                error_count += 1
                # Keep scan-level failure isolated; without a stable path identity it
                # is safer to log this in the scan summary than invent a file record.
                continue

        manifest = new_manifest(scan_id, records)
        quality = build_quality_report(records)
        metrics = evaluation_studies(records)
        integration = validate_platform(self.config, records)
        report = {
            "scan_id": scan_id,
            "created_at": created_at,
            "artifact_count": len(records),
            "changed_count": changed,
            "error_count": error_count,
            "quality": quality,
            "metrics": metrics,
            "integration": integration,
        }
        digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()
        report["manifest_sha256"] = digest
        history_dir = self.output_dir / "history" / scan_id
        manifest_path = history_dir / "artifacts.json"
        write_json_atomic(manifest_path, manifest)
        write_json_atomic(history_dir / "overview.json", report)
        write_html_atomic(history_dir / "overview.html", report)
        write_json_atomic(history_dir / "quality.json", quality)
        write_json_atomic(history_dir / "evaluation.json", metrics)
        write_json_atomic(history_dir / "integration.json", integration)
        # Immutable history snapshots above; latest files are convenience pointers.
        write_json_atomic(self.output_dir / "latest" / "artifacts.json", manifest)
        write_json_atomic(self.output_dir / "latest" / "overview.json", report)
        write_html_atomic(self.output_dir / "latest" / "overview.html", report)
        write_json_atomic(self.output_dir / "latest" / "quality.json", quality)
        write_json_atomic(self.output_dir / "latest" / "evaluation.json", metrics)
        write_json_atomic(self.output_dir / "latest" / "integration.json", integration)
        summary = {"scan_id": scan_id, "created_at": created_at, "artifact_count": len(records), "changed_count": changed,
                   "error_count": error_count, "manifest_path": str(manifest_path.relative_to(self.output_dir))}
        self.cache.put_scan(summary)
        self.cache.commit()
        self.last_scan = report
        return report

    @staticmethod
    def _declared_split(summary: dict[str, Any]) -> str | None:
        if summary.get("contains_test_only") is True:
            return "test"
        for key in ("split", "data_split", "declared_split"):
            value = summary.get(key)
            if isinstance(value, str):
                return value.strip().lower()
        return None

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        record = self.cache.get_artifact(artifact_id)
        if not record:
            return None
        # Never return full local absolute paths; API resolves only IDs already in cache.
        return record

    def artifacts(self) -> list[dict[str, Any]]:
        scans = self.cache.list_scans()
        return self.cache.list_artifacts(scans[0]["scan_id"] if scans else None)

    def latest_report(self, name: str = "overview") -> dict[str, Any] | None:
        path = self.output_dir / "latest" / f"{name}.json"
        if name not in {"overview", "artifacts", "quality", "evaluation", "integration"} or not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def latest_html_report(self) -> str | None:
        path = self.output_dir / "latest" / "overview.html"
        if not path.is_file():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            return None

    def close(self) -> None:
        self.cache.close()

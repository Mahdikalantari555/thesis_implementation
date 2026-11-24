"""Aggregate adapter findings into explicit, bounded data-quality reports."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


ID_COLUMNS = {"id", "farm_id", "field_id", "cycle_id", "observation_id"}
DATE_COLUMNS = {"date", "datetime", "timestamp", "acquisition_date", "observation_date"}


def _row_checks(record: dict[str, Any]) -> list[dict[str, Any]]:
    summary = record.get("summary", {})
    findings: list[dict[str, Any]] = []
    for name, stats in summary.get("columns", {}).items():
        if not isinstance(stats, dict):
            continue
        denominator = stats.get("rows_sampled", stats.get("count"))
        for key, label in (("missing_count", "missing"), ("nonfinite_count", "NaN/Inf"), ("out_of_range_count", "physical-range")):
            count = stats.get(key, 0)
            if isinstance(count, (int, float)) and count:
                findings.append({"artifact_id": record["id"], "path": record["relative_path"], "check": label,
                                 "column": name, "count": count, "denominator": denominator,
                                 "sampled": bool(summary.get("is_sampled")), "severity": "WARNING"})
    for key, label in (("invalid_coordinate_feature_sample", "invalid_coordinates"), ("invalid_geometry_sample", "invalid_geometry"), ("duplicate_ids_sample", "duplicate_ids_sample")):
        if summary.get(key, 0):
            findings.append({"artifact_id": record["id"], "path": record["relative_path"], "check": label,
                             "count": summary[key], "sampled": True, "severity": "WARNING"})
    if summary.get("crs_is_wgs84") is False:
        findings.append({"artifact_id": record["id"], "path": record["relative_path"], "check": "crs_not_wgs84",
                         "crs": summary.get("crs"), "severity": "WARNING"})
    arrays = summary.get("arrays", {})
    if isinstance(arrays, dict):
        for name, details in arrays.items():
            if not isinstance(details, dict):
                continue
            for key, label in (("nan_count_sample", "NaN"), ("inf_count_sample", "Inf")):
                count = details.get(key, 0)
                if count:
                    findings.append({"artifact_id": record["id"], "path": record["relative_path"], "check": label,
                                     "column": name, "count": count, "sampled": True, "severity": "WARNING"})
    sample = summary.get("sample")
    if isinstance(sample, list) and sample and isinstance(sample[0], dict):
        findings.extend(_check_duplicate_ids(record, sample))
        findings.extend(_check_time_order(record, sample))
    return findings


def _check_duplicate_ids(record: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    key = next((name for name in rows[0] if name.lower() in ID_COLUMNS), None)
    if not key:
        return []
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        normalized = str(value)
        if normalized in seen:
            duplicates.add(normalized)
        seen.add(normalized)
    return [{"artifact_id": record["id"], "path": record["relative_path"], "check": "duplicate_ids_sample",
             "column": key, "count": len(duplicates), "examples": sorted(duplicates)[:10], "sampled": True, "severity": "WARNING"}] if duplicates else []


def _parse_date(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc).replace(tzinfo=None) if parsed.tzinfo else parsed
    except ValueError:
        try:
            return datetime.strptime(value[:10], "%Y-%m-%d")
        except ValueError:
            return None


def _check_time_order(record: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    key = next((name for name in rows[0] if name.lower() in DATE_COLUMNS), None)
    if not key:
        return []
    parsed = [_parse_date(row.get(key)) for row in rows]
    valid = [item for item in parsed if item is not None]
    if len(valid) < 2:
        return []
    disorder = sum(valid[index] < valid[index - 1] for index in range(1, len(valid)))
    return [{"artifact_id": record["id"], "path": record["relative_path"], "check": "temporal_order_sample",
             "column": key, "count": disorder, "sampled": True, "severity": "WARNING"}] if disorder else []


def build_quality_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    role_counts: dict[str, int] = {}
    findings: list[dict[str, Any]] = []
    for record in records:
        status = str(record.get("status", "UNKNOWN"))
        role = str(record.get("role", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        role_counts[role] = role_counts.get(role, 0) + 1
        if status not in {"OK", "DISCOVERED"}:
            findings.append({"artifact_id": record.get("id"), "path": record.get("relative_path"),
                             "check": "artifact_status", "status": status, "reason": record.get("message"),
                             "severity": "WARNING" if status.startswith("RESTRICTED") or status in {"UNSUPPORTED", "WARNING"} else "ERROR"})
        findings.extend(_row_checks(record))
    return {
        "artifact_count": len(records),
        "status_counts": status_counts,
        "role_counts": role_counts,
        "attention_count": sum(1 for item in findings if item.get("severity") in {"WARNING", "ERROR"}),
        "findings": findings[:5000],
        "findings_truncated": len(findings) > 5000,
        "sampling_note": "Statistics are bounded to adapter samples where marked; sampled values are not full-population estimates.",
    }

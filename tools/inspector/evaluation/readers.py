"""Normalize only the metadata envelope of existing research reports."""
from __future__ import annotations

from typing import Any

STUDIES = ("dev", "architecture_benchmark", "regional_representation")
ALLOWED_SPLITS = {"dev", "development", "validation", "train+dev", "train_dev", "train-and-dev"}


def _artifact_type(path: str) -> str:
    name = path.lower().rsplit("/", 1)[-1]
    if "near_match" in name:
        return "exploratory_near_matches"
    if "gate" in name:
        return "decision_gate"
    if "manifest" in name:
        return "experiment_manifest"
    if "metric" in name or "dev_log" in name:
        return "evaluation_metrics"
    if "progress" in name:
        return "progress_log"
    if "generalization" in name:
        return "generalization_report"
    if "retrieval" in name:
        return "retrieval_report"
    if "transfer" in name:
        return "transfer_report"
    if "pretraining" in name:
        return "pretraining_report"
    return "evaluation_artifact"


def _evidence_class(path: str, summary: dict[str, Any]) -> str:
    lower = path.lower()
    if "near_match" in lower:
        return "exploratory"
    declared = str(summary.get("evidence_class", "")).lower()
    if declared:
        return declared
    if any(token in lower for token in ("synthetic", "injection", "architecture_benchmark")):
        return "synthetic_benchmark"
    if any(token in lower for token in ("proxy", "agreement", "cwsi", "tvdi")):
        return "proxy_agreement"
    if "generalization" in lower or "temporal_roll" in lower or "spatial_roll" in lower:
        return "generalization"
    return "recorded_evaluation"


def _allowed_context(study: str, path: str, summary: dict[str, Any]) -> tuple[bool, str | None]:
    split = summary.get("split") or summary.get("data_split") or summary.get("evaluation_split")
    if isinstance(split, str):
        normalized = split.strip().lower()
        if normalized in {"test", "locked_test", "final_test", "test_only"}:
            return False, "test-only metrics are restricted"
        if normalized in ALLOWED_SPLITS:
            return True, None
    # The legacy evaluator writes under reports/dev and its contract is dev-only.
    if study == "dev" and (path.startswith("dev/") or path == "dev_log.csv"):
        return True, None
    return False, "report does not declare an allowed development/validation split"


def _display_na(value: Any) -> Any:
    if value is None:
        return "N/A"
    if isinstance(value, dict):
        return {key: _display_na(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_display_na(item) for item in value]
    return value


def evaluation_studies(records: list[dict[str, Any]]) -> dict[str, Any]:
    studies: dict[str, dict[str, Any]] = {key: {"study_id": key, "artifacts": [], "status_counts": {}, "metric_artifact_count": 0} for key in STUDIES}
    for record in records:
        path = str(record.get("relative_path", "")).lower()
        root = record.get("root")
        if root != "reports":
            continue
        if path.startswith("dev/") or path == "dev_log.csv":
            study = "dev"
        elif path.startswith("architecture_benchmark/"):
            study = "architecture_benchmark"
        elif path.startswith("regional_representation/"):
            study = "regional_representation"
        else:
            continue
        summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
        allowed, restriction_reason = _allowed_context(study, path, summary)
        artifact_type = _artifact_type(path)
        evidence_class = _evidence_class(path, summary)
        registry_status = summary.get("registry_status")
        if registry_status is None and summary.get("status") in {"Pending", "Running", "Completed", "Failed", "Interrupted", "Deferred", "Not Applicable"}:
            registry_status = summary.get("status")
        safe_summary = summary
        if not allowed:
            safe_summary = {key: summary[key] for key in ("top_level_type", "declared_schema", "keys", "protocol", "split", "data_split", "status") if key in summary}
        recorded_metrics = summary.get("recorded_metrics", summary.get("metrics", {})) if allowed else {}
        restricted = (not allowed) or evidence_class == "exploratory" or str(record.get("status", "")).startswith("RESTRICTED")
        item = {
            "id": record.get("id"), "study_id": study, "path": path, "format": record.get("format"),
            "artifact_type": artifact_type, "status": record.get("status", "UNKNOWN"),
            "registry_status": registry_status, "schema_version": summary.get("declared_schema"),
            "protocol": summary.get("protocol"), "run_id": summary.get("run_id"),
            "model": summary.get("model_id", summary.get("model")), "scorer": summary.get("scorer"),
            "seeds": summary.get("seeds"), "split": summary.get("split", summary.get("data_split")),
            "split_hash": summary.get("splits_locked_sha256", summary.get("split_hash")),
            "evaluation_subset_hash": summary.get("eval_subset_sha256", summary.get("eval_subset_hash")),
            "data_hash": summary.get("data_hash", summary.get("cycle_file_sha256")),
            "claim_rung": summary.get("ladder_rung", summary.get("claim_rung")),
            "evidence_class": evidence_class,
            "restricted_from_comparison": restricted,
            "comparison_state": "N/A" if restricted or not recorded_metrics else "recorded_not_ranked",
            "restriction_reason": restriction_reason,
            "recorded_metrics": recorded_metrics,
            "metrics_display": _display_na(recorded_metrics),
            "summary": safe_summary,
            "message": record.get("message"),
        }
        if artifact_type == "exploratory_near_matches":
            item["restriction_reason"] = "descriptive/exploratory artifact; excluded from rankings and gates"
            item["restricted_from_comparison"] = True
            item["comparison_state"] = "N/A"
        studies[study]["artifacts"].append(item)
        state = item["registry_status"] or item["status"]
        counts = studies[study]["status_counts"]
        counts[state] = counts.get(state, 0) + 1
        if artifact_type == "evaluation_metrics" and allowed:
            studies[study]["metric_artifact_count"] += 1
    return {"studies": studies, "cross_study_ranking": False,
            "metric_policy": "Recorded metrics only; no recomputation. Unknown or test-only split context is restricted.",
            "comparison_policy": "Studies remain independent. Compare only within one study and compatible declared protocol/split/data hashes."}

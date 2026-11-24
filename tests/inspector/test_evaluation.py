from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.inspector.core.config import InputRoot, InspectorConfig
from tools.inspector.evaluation.readers import evaluation_studies
from tools.inspector.adapters.readers import inspect_artifact
from tools.inspector.core.scanner import Scanner


class EvaluationReaderTests(unittest.TestCase):
    def test_existing_dev_metric_json_exposes_recorded_metrics_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.json"
            path.write_text(json.dumps({"schema_version": 2, "run_id": "r1", "model_id": "gru_ae",
                "protocol": "severity_v1", "split": "dev", "ladder_rung": "L2",
                "mdm80": {"acute/ndvi": {"mean_sentinel": 7, "per_seed_labels": [">S6"]}, "detection_rate": 0.4}}), encoding="utf-8")
            config = InspectorConfig(Path.cwd(), "x", "127.0.0.1", 8765, False, True,
                (InputRoot("reports", Path(tmp)),), Path(tmp) / "out", max_json_bytes=100_000)
            result = inspect_artifact(path, "json", config)
            summary = result["summary"]
            self.assertEqual(summary["protocol"], "severity_v1")
            self.assertEqual(summary["recorded_metrics"]["mdm80"]["acute/ndvi"]["mean_sentinel"], 7)
            self.assertEqual(summary["recorded_metrics"]["mdm80"]["detection_rate"], 0.4)

    def test_lineages_are_separate_and_partial_statuses_preserved(self) -> None:
        records = [
            {"id": "d", "root": "reports", "relative_path": "dev/r1/metrics.json", "format": "json", "status": "OK",
             "summary": {"protocol": "severity_v1", "run_id": "r1", "split": "dev", "recorded_metrics": {"mdm80": 7}}},
            {"id": "b", "root": "reports", "relative_path": "architecture_benchmark/run/manifest.json", "format": "json", "status": "OK",
             "summary": {"protocol": "V1", "split": "train+dev", "status": "Deferred", "recorded_metrics": {"auprc": None}}},
            {"id": "b2", "root": "reports", "relative_path": "architecture_benchmark/run/not_applicable.json", "format": "json", "status": "OK",
             "summary": {"protocol": "V1", "split": "train+dev", "status": "Not Applicable", "reason": "arm disabled"}},
            {"id": "g", "root": "reports", "relative_path": "regional_representation/decision_gates.json", "format": "json", "status": "OK",
             "summary": {"protocol": "capstone", "split": "dev", "status": "Completed", "claim_rung": "L2"}},
            {"id": "r", "root": "reports", "relative_path": "regional_representation/retrieval_report.json", "format": "json", "status": "OK",
             "summary": {"protocol": "capstone_retrieval_v1", "split": "dev", "claim_rung": "L1", "recorded_metrics": {"recall_at_5": 0.72}}},
            {"id": "n", "root": "reports", "relative_path": "regional_representation/near_matches.json", "format": "json", "status": "OK",
             "summary": {"protocol": "capstone", "split": "dev"}},
        ]
        result = evaluation_studies(records)
        self.assertFalse(result["cross_study_ranking"])
        self.assertEqual(len(result["studies"]["dev"]["artifacts"]), 1)
        benchmark = result["studies"]["architecture_benchmark"]["artifacts"][0]
        self.assertEqual(benchmark["registry_status"], "Deferred")
        self.assertFalse(benchmark["restricted_from_comparison"])
        self.assertEqual(benchmark["metrics_display"]["auprc"], "N/A")
        not_applicable = result["studies"]["architecture_benchmark"]["artifacts"][1]
        self.assertEqual(not_applicable["registry_status"], "Not Applicable")
        regional = result["studies"]["regional_representation"]["artifacts"]
        retrieval = next(item for item in regional if item["artifact_type"] == "retrieval_report")
        self.assertEqual(retrieval["claim_rung"], "L1")
        self.assertEqual(retrieval["recorded_metrics"]["recall_at_5"], 0.72)
        near = next(item for item in regional if item["artifact_type"] == "exploratory_near_matches")
        self.assertEqual(near["evidence_class"], "exploratory")
        self.assertTrue(near["restricted_from_comparison"])

    def test_scanner_redacts_unknown_split_metric_values_but_keeps_declared_dev_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            reports = base / "reports"
            (reports / "regional_representation").mkdir(parents=True)
            (reports / "architecture_benchmark").mkdir()
            (reports / "regional_representation" / "unknown.json").write_text('{"protocol":"capstone","metrics":{"auprc":0.99}}', encoding="utf-8")
            (reports / "regional_representation" / "dev.json").write_text('{"protocol":"capstone","split":"dev","metrics":{"auprc":0.4}}', encoding="utf-8")
            (reports / "architecture_benchmark" / "deferred.json").write_text('{"protocol":"V1","split":"train+dev","status":"Deferred","metrics":{"auprc":null}}', encoding="utf-8")
            config = InspectorConfig(base, "x", "127.0.0.1", 8765, False, True,
                (InputRoot("reports", reports),), base / "inspector", max_json_bytes=100_000)
            scanner = Scanner(config)
            try:
                report = scanner.scan()
                records = scanner.artifacts()
                unknown = next(item for item in records if item["relative_path"].endswith("unknown.json"))
                self.assertEqual(unknown["status"], "RESTRICTED/UNKNOWN_SPLIT")
                self.assertNotIn("recorded_metrics", unknown["summary"])
                studies = report["metrics"]["studies"]
                dev = next(item for item in studies["regional_representation"]["artifacts"] if item["path"].endswith("dev.json"))
                deferred = next(item for item in studies["architecture_benchmark"]["artifacts"] if item["path"].endswith("deferred.json"))
                self.assertEqual(dev["recorded_metrics"]["metrics"]["auprc"], 0.4)
                self.assertEqual(deferred["registry_status"], "Deferred")
            finally:
                scanner.close()

    def test_test_or_unknown_split_is_restricted_from_comparison(self) -> None:
        records = [
            {"id": "t", "root": "reports", "relative_path": "architecture_benchmark/test/metrics.json", "format": "json", "status": "RESTRICTED",
             "summary": {"split": "test"}},
            {"id": "u", "root": "reports", "relative_path": "regional_representation/transfer.json", "format": "json", "status": "OK",
             "summary": {"protocol": "capstone", "recorded_metrics": {"auprc": 0.5}}},
        ]
        result = evaluation_studies(records)
        self.assertTrue(result["studies"]["architecture_benchmark"]["artifacts"][0]["restricted_from_comparison"])
        unknown = result["studies"]["regional_representation"]["artifacts"][0]
        self.assertTrue(unknown["restricted_from_comparison"])
        self.assertEqual(unknown["recorded_metrics"], {})
        self.assertNotIn("recorded_metrics", unknown["summary"])


if __name__ == "__main__":
    unittest.main()

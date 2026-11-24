from __future__ import annotations

import builtins
import importlib.util
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from scripts.platform_channels import CHANNEL_REGISTRY, SCORE_CHECKPOINTS, SCORE_REGISTRY
from tools.inspector.core.config import InputRoot, InspectorConfig
from tools.inspector.core.integration import validate_platform


class PlatformReadinessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.platform = self.base / "platform"
        self.platform.mkdir()
        self.config = InspectorConfig(
            repo_root=Path(__file__).resolve().parents[2], title="test", host="127.0.0.1", port=0,
            open_browser=False, read_only=True, roots=(InputRoot("platform_data", self.platform),),
            output_dir=self.base / "inspector", max_json_bytes=1_000_000, max_series_bytes=5_000_000,
        )
        self.axis = ["2024-01-01", "2024-02-17", "2024-05-31"]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def _rle(values):
        runs = []
        for value in values:
            if runs and runs[-1][0] == value:
                runs[-1][1] += 1
            else:
                runs.append([value, 1])
        return runs

    def _write_fixture(self, version=5, splits=None, test_ndvi=-0.99):
        farm = {"id": "F-001", "centroid": [31.0, 48.0],
                "polygon": [[31.0, 48.0], [31.1, 48.0], [31.1, 48.1], [31.0, 48.1], [31.0, 48.0]]}
        (self.platform / "farms_khuzestan.json").write_text(json.dumps([farm]), encoding="utf-8")
        splits = splits or ["train", "dev", "test"]
        channels = {}
        channel_provenance = {}
        for spec in CHANNEL_REGISTRY:
            mid = (spec.physical_range[0] + spec.physical_range[1]) / 2
            values = [mid, mid, mid]
            if spec.name == "ndvi":
                values[2] = test_ndvi
            channels[spec.name] = values
            channel_provenance[spec.name] = self._rle(["observed", "observed", "observed"])
        scores = {spec.name: [0.1, 0.2, 0.3] for spec in SCORE_REGISTRY}
        score_provenance = {spec.name: self._rle(["model", "model", "model"]) for spec in SCORE_REGISTRY}
        field = {"channels": channels, "scores": scores, "channelProvenance": channel_provenance,
                 "scoreProvenance": score_provenance, "stressFlag": self._rle(["normal", "suspected_stress", "suspected_stress"]),
                 "stressFlagModel": self._rle([None, SCORE_CHECKPOINTS["masked_transformer"], SCORE_CHECKPOINTS["masked_transformer"]]),
                 "split": self._rle(splits), "ladderRung": "L2"}
        payload = {"contractVersion": version, "axis": self.axis, "fields": {"F-001": field},
                   "checkpoints": {spec.name: spec.checkpoint for spec in SCORE_REGISTRY if spec.checkpoint}}
        (self.platform / "series_khuzestan.json").write_text(json.dumps(payload), encoding="utf-8")

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "optional ijson dependency not installed")
    def test_valid_payload_reports_registry_checks_and_skips_test_channel_values(self):
        self._write_fixture()
        result = validate_platform(self.config, [])
        self.assertIn(result["status"], {"WARNING", "PASS"})  # source registry has no explicit CRS
        checks = {item["check"]: item for item in result["checks"]}
        for key in ("farm_ids", "farm_geometry", "contract_version", "observation_dates", "split_tags", "series_structure", "channel_ranges", "provenance_and_flags"):
            self.assertEqual(checks[key]["status"], "PASS", checks[key])
        self.assertIn("non-test observations only", result["values_examined"])
        self.assertEqual(result["status"], "WARNING")  # Warnings alone never BLOCK readiness

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "optional ijson dependency not installed")
    def test_critical_bad_version_and_split_alignment_block_readiness(self):
        self._write_fixture(version=4)
        result = validate_platform(self.config, [])
        self.assertEqual(result["status"], "BLOCKED")
        version = next(item for item in result["checks"] if item["check"] == "contract_version")
        self.assertEqual(version["status"], "FAIL")
        self._write_fixture(splits=["train", "test", "dev"])
        payload = json.loads((self.platform / "series_khuzestan.json").read_text(encoding="utf-8"))
        payload["fields"]["F-001"]["split"] = [["train", 1], ["test", 1]]
        (self.platform / "series_khuzestan.json").write_text(json.dumps(payload), encoding="utf-8")
        result = validate_platform(self.config, [])
        self.assertEqual(result["status"], "BLOCKED")
        split = next(item for item in result["checks"] if item["check"] == "split_tags")
        self.assertEqual(split["status"], "FAIL")
        self.assertFalse(any(item["check"] == "channel_ranges" for item in result["checks"]))
        self.assertTrue(any(item["check"] == "series_values" and item["status"] == "WARNING" for item in result["checks"]))

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "optional ijson dependency not installed")
    def test_score_values_must_agree_with_recorded_provenance(self):
        self._write_fixture()
        payload = json.loads((self.platform / "series_khuzestan.json").read_text(encoding="utf-8"))
        score_name = SCORE_REGISTRY[0].name
        payload["fields"]["F-001"]["scoreProvenance"][score_name] = [["model", 1], ["absent", 1], ["model", 1]]
        (self.platform / "series_khuzestan.json").write_text(json.dumps(payload), encoding="utf-8")
        result = validate_platform(self.config, [])
        self.assertEqual(result["status"], "BLOCKED")
        provenance = next(item for item in result["checks"] if item["check"] == "provenance_and_flags")
        self.assertEqual(provenance["status"], "FAIL")
        self.assertTrue(any("score/provenance disagreement" in example for example in provenance["examples"]))

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "optional ijson dependency not installed")
    def test_checkpoint_metadata_alignment_never_loads_weight_or_metric_values(self):
        self._write_fixture()
        checkpoints = self.base / "checkpoints"
        checkpoints.mkdir()
        for stem in SCORE_CHECKPOINTS.values():
            (checkpoints / f"{stem}.json").write_text(json.dumps({
                "model_id": stem, "ladder_rung": "L2", "dev_metrics_summary": {"locked_test_metric": 0.99}
            }), encoding="utf-8")
        self.config = InspectorConfig(
            repo_root=self.config.repo_root, title="test", host="127.0.0.1", port=0, open_browser=False,
            read_only=True, roots=(InputRoot("platform_data", self.platform), InputRoot("checkpoints", checkpoints)),
            output_dir=self.config.output_dir, max_json_bytes=1_000_000, max_series_bytes=5_000_000,
        )
        result = validate_platform(self.config, [])
        sidecars = [item for item in result["checks"] if item["check"] == "checkpoint_sidecars"]
        self.assertEqual(sidecars[-1]["status"], "PASS")
        self.assertNotIn("locked_test_metric", json.dumps(result))
        (checkpoints / f"{next(iter(SCORE_CHECKPOINTS.values()))}.json").write_text('{"model_id":"wrong"}', encoding="utf-8")
        result = validate_platform(self.config, [])
        self.assertEqual(result["status"], "BLOCKED")
        self.assertEqual(next(item for item in result["checks"] if item["check"] == "checkpoint_sidecars")["status"], "FAIL")

    def test_missing_optional_streamer_reports_warning_not_false_pass(self):
        self._write_fixture()
        old_import = builtins.__import__
        def without_ijson(name, *args, **kwargs):
            if name == "ijson":
                raise ImportError(name)
            return old_import(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=without_ijson):
            result = validate_platform(self.config, [])
        self.assertEqual(result["status"], "NOT_EVALUATED")
        self.assertNotIn("PASS", [item["status"] for item in result["checks"] if item["check"] == "series_payload"])
        self.assertTrue(any("optional ijson" in item["reason"] for item in result["warnings"]))

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "optional ijson dependency not installed")
    def test_invalid_non_test_channel_value_fails_registry_range_check(self):
        self._write_fixture(test_ndvi=-0.99)
        payload = json.loads((self.platform / "series_khuzestan.json").read_text(encoding="utf-8"))
        payload["fields"]["F-001"]["channels"]["ndvi"][1] = -0.99
        (self.platform / "series_khuzestan.json").write_text(json.dumps(payload), encoding="utf-8")
        result = validate_platform(self.config, [])
        self.assertEqual(result["status"], "BLOCKED")
        ranges = next(item for item in result["checks"] if item["check"] == "channel_ranges")
        self.assertEqual(ranges["status"], "FAIL")


if __name__ == "__main__":
    unittest.main()

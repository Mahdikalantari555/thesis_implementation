from __future__ import annotations

import builtins
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.platform_channels import CHANNEL_REGISTRY, SCORE_REGISTRY
from tools.inspector.core.config import InputRoot, InspectorConfig
from tools.inspector.core.series import farm_series
from tools.inspector.core.spatial import farm_map


class SpatialTests(unittest.TestCase):
    def _config(self, base: Path, platform_root: Path) -> InspectorConfig:
        return InspectorConfig(repo_root=Path(__file__).resolve().parents[2], title="Spatial test", host="127.0.0.1", port=0,
            open_browser=False, read_only=True, roots=(InputRoot("platform_data", platform_root),),
            output_dir=base / "inspector", max_json_bytes=1_000_000, max_series_bytes=1_000_000)

    def test_farm_map_uses_configured_registry_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            platform = base / "platform"
            platform.mkdir()
            (platform / "farms_khuzestan.json").write_text(json.dumps([
                {"id": "F-001", "code": "A", "block": "B1", "centroid": [31.0, 48.0], "polygon": [[31, 48], [31.1, 48], [31.1, 48.1]]}
            ]), encoding="utf-8")
            result = farm_map(self._config(base, platform))
            self.assertTrue(result["available"])
            self.assertEqual(result["farms"][0]["id"], "F-001")
            self.assertEqual(result["farms"][0]["centroid"], [31.0, 48.0])

    def test_series_reader_reports_missing_optional_streamer(self) -> None:
        if importlib.util.find_spec("ijson") is not None:
            self.skipTest("ijson is installed in this environment")
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            platform = base / "platform"
            platform.mkdir()
            (platform / "series_khuzestan.json").write_text("{}", encoding="utf-8")
            old_import = builtins.__import__
            def no_ijson(name, *args, **kwargs):
                if name == "ijson":
                    raise ImportError(name)
                return old_import(name, *args, **kwargs)
            with mock.patch("builtins.__import__", side_effect=no_ijson):
                result = farm_series(self._config(base, platform), "F-001")
            self.assertEqual(result["status"], "UNSUPPORTED")
            self.assertEqual(result["observations"], [])

    @unittest.skipUnless(importlib.util.find_spec("ijson"), "optional ijson dependency not installed")
    def test_series_uses_real_axis_and_excludes_test_values_before_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            platform = base / "platform"
            platform.mkdir()
            channels = {spec.name: [0.1, 0.2, 0.99, 0.4] for spec in CHANNEL_REGISTRY}
            provenance = {spec.name: [["observed", 4]] for spec in CHANNEL_REGISTRY}
            scores = {spec.name: [0.1, 0.2, 0.99, 0.4] for spec in SCORE_REGISTRY}
            score_provenance = {spec.name: [["model", 4]] for spec in SCORE_REGISTRY}
            payload = {
                "contractVersion": 5,
                "axis": ["2024-01-01", "2024-01-18", "2024-03-01", "2024-04-10"],
                "fields": {"F-001": {
                    "channels": channels, "scores": scores,
                    "channelProvenance": provenance, "scoreProvenance": score_provenance,
                    "stressFlag": [["normal", 1], ["suspected_stress", 2], ["normal", 1]],
                    "stressFlagModel": [[None, 1], ["checkpoint-x", 3]],
                    "split": [["train", 1], ["dev", 1], ["test", 1], ["monitoring", 1]],
                    "ladderRung": "L2",
                }},
            }
            (platform / "series_khuzestan.json").write_text(json.dumps(payload), encoding="utf-8")
            result = farm_series(self._config(base, platform), "F-001")
            self.assertTrue(result["available"], result.get("reason"))
            self.assertEqual(result["axis_count"], 4)
            self.assertEqual(result["excluded_test_observation_count"], 1)
            self.assertEqual([item["date"] for item in result["observations"]], ["2024-01-01", "2024-01-18", "2024-04-10"])
            self.assertEqual(result["observations"][1]["stress_flag"], "suspected_stress")
            self.assertEqual(result["observations"][1]["stress_flag_model"], "checkpoint-x")
            self.assertEqual(result["observations"][1]["split"], "dev")
            self.assertNotIn(0.99, [item["channels"]["ndvi"] for item in result["observations"]])
            self.assertNotIn(0.99, [value for item in result["observations"] for value in item["scores"].values()])
            self.assertEqual([item["date"] for item in result["observations"] if item["stress_flag"] == "suspected_stress"], ["2024-01-18"])
            self.assertTrue(result["suspected_stress_available"])
            payload["axis"] = ["2024-02-30"]
            (platform / "series_khuzestan.json").write_text(json.dumps(payload), encoding="utf-8")
            malformed = farm_series(self._config(base, platform), "F-001")
            self.assertEqual(malformed["status"], "INCOMPLETE")
            payload["axis"] = ["2024-01-01", "2024-01-18", "2024-03-01", "2024-04-10"]
            del payload["fields"]["F-001"]["stressFlag"]
            del payload["fields"]["F-001"]["stressFlagModel"]
            payload["contractVersion"] = 4
            (platform / "series_khuzestan.json").write_text(json.dumps(payload), encoding="utf-8")
            legacy = farm_series(self._config(base, platform), "F-001")
            self.assertEqual(legacy["stress_marker_status"], "unavailable_v4")
            self.assertTrue(all(item["stress_flag"] is None for item in legacy["observations"]))
            payload["contractVersion"] = 5
            (platform / "series_khuzestan.json").write_text(json.dumps(payload), encoding="utf-8")
            missing_flags = farm_series(self._config(base, platform), "F-001")
            self.assertEqual(missing_flags["stress_marker_status"], "unavailable_missing_flags")

    def test_missing_series_and_invalid_farm_id_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            platform = base / "platform"
            platform.mkdir()
            result = farm_series(self._config(base, platform), "../../F-001")
            self.assertEqual(result["status"], "INVALID_ID")
            result = farm_series(self._config(base, platform), "F-001")
            self.assertEqual(result["status"], "UNAVAILABLE")


if __name__ == "__main__":
    unittest.main()

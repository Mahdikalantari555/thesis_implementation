from __future__ import annotations

import unittest

from tools.inspector.validation.data_quality import build_quality_report


class DataQualityTests(unittest.TestCase):
    def test_quality_report_keeps_missing_nonfinite_ranges_and_sample_denominator(self) -> None:
        record = {"id": "a", "relative_path": "tables/x.csv", "status": "OK", "role": "dataset",
                  "summary": {"is_sampled": True, "columns": {"ndvi": {"missing_count": 1, "nonfinite_count": 1,
                  "out_of_range_count": 2, "rows_sampled": 10}}}}
        report = build_quality_report([record])
        self.assertEqual(report["artifact_count"], 1)
        self.assertEqual({item["check"] for item in report["findings"]}, {"missing", "NaN/Inf", "physical-range"})
        self.assertTrue(all(item["sampled"] for item in report["findings"]))
        self.assertTrue(all(item["denominator"] == 10 for item in report["findings"]))

    def test_duplicate_ids_and_temporal_disorder_are_reported_separately(self) -> None:
        record = {"id": "b", "relative_path": "data/dev/series.csv", "status": "OK", "role": "dataset",
                  "summary": {"sample": [{"field_id": "F1", "date": "2024-02-01T00:00:00Z"},
                                          {"field_id": "F1", "date": "2024-01-01"}]}}
        report = build_quality_report([record])
        checks = {item["check"] for item in report["findings"]}
        self.assertEqual(checks, {"duplicate_ids_sample", "temporal_order_sample"})

    def test_restricted_material_is_not_misreported_as_pass_or_zero(self) -> None:
        report = build_quality_report([{"id": "x", "relative_path": "test/data.parquet", "status": "RESTRICTED",
                                       "role": "dataset", "message": "locked", "summary": {}}])
        self.assertEqual(report["status_counts"], {"RESTRICTED": 1})
        self.assertEqual(report["findings"][0]["severity"], "WARNING")
        self.assertEqual(report["attention_count"], 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import builtins
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from tools.inspector.adapters.readers import inspect_artifact
from tools.inspector.core.config import InputRoot, InspectorConfig


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = InspectorConfig(repo_root=Path.cwd(), title="Inspector", host="127.0.0.1", port=8765,
                                     open_browser=False, read_only=True, roots=(InputRoot("fixture", self.root),),
                                     output_dir=self.root / "inspector", max_sample_size=3, max_json_bytes=100_000,
                                     max_npz_uncompressed_bytes=100_000, hash_small_files_under_bytes=1000)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_json_adapter_preserves_schema_and_metadata(self) -> None:
        path = self.root / "run.json"
        path.write_text(json.dumps({"schema_version": 2, "split": "dev", "protocol": "v1", "metrics": {"auprc": 0.4}}), encoding="utf-8")
        result = inspect_artifact(path, "json", self.config)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["schema"], "2")
        self.assertEqual(result["summary"]["split"], "dev")

    def test_csv_is_bounded_and_uses_channel_registry(self) -> None:
        path = self.root / "metrics.csv"
        path.write_text("ndvi,missing\n0.2,\n1.5,1\n0.5,2\n-0.7,3\n", encoding="utf-8")
        result = inspect_artifact(path, "csv", self.config)
        self.assertEqual(result["status"], "OK")
        self.assertTrue(result["summary"]["is_sampled"])
        self.assertFalse(result["summary"]["row_count_exact"])
        from scripts.platform_channels import by_name
        self.assertEqual(result["summary"]["columns"]["ndvi"]["physical_range"], list(by_name("ndvi").physical_range))
        self.assertTrue(result["summary"]["channel_registry_available"])
        self.assertEqual(result["summary"]["columns"]["ndvi"]["out_of_range_count"], 1)

    def test_corrupt_json_is_reported_not_raised(self) -> None:
        path = self.root / "corrupt.json"
        path.write_text('{"broken":', encoding="utf-8")
        result = inspect_artifact(path, "json", self.config)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIn("could not be parsed", result["message"])

    def test_npz_optional_reader_fails_gracefully(self) -> None:
        old_import = builtins.__import__
        def no_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError(name)
            return old_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.__import__", side_effect=no_numpy):
            path = Path(tmp) / "x.npz"
            import zipfile
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("arr.npy", b"not actually a numpy array")
            result = inspect_artifact(path, "npz", self.config)
            self.assertEqual(result["status"], "UNSUPPORTED")

    def test_parquet_and_zarr_optional_readers_fail_gracefully(self) -> None:
        old_import = builtins.__import__
        def no_optional(name, *args, **kwargs):
            if name == "pyarrow.parquet" or name == "zarr":
                raise ImportError(name)
            return old_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.__import__", side_effect=no_optional):
            path = Path(tmp) / "x.parquet"
            path.touch()
            self.assertEqual(inspect_artifact(path, "parquet", self.config)["status"], "UNSUPPORTED")
            store = Path(tmp) / "store.zarr"
            store.mkdir()
            self.assertEqual(inspect_artifact(store, "zarr", self.config)["status"], "UNSUPPORTED")

    def test_geojson_checks_coordinates_geometry_ids_and_crs(self) -> None:
        path = self.root / "farms.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
            "features": [
                {"type": "Feature", "id": "F1", "geometry": {"type": "Polygon", "coordinates": [[[48, 31], [48.1, 31], [48.1, 31.1], [48, 31]]]}, "properties": {}},
                {"type": "Feature", "id": "F1", "geometry": {"type": "Polygon", "coordinates": [[[250, 31], [251, 31], [251, 32], [250, 31]]]}, "properties": {}},
            ]}), encoding="utf-8")
        result = inspect_artifact(path, "geojson", self.config)
        self.assertEqual(result["status"], "OK")
        self.assertEqual(result["summary"]["duplicate_ids_sample"], 1)
        self.assertEqual(result["summary"]["invalid_coordinate_feature_sample"], 1)
        self.assertEqual(result["summary"]["crs_is_wgs84"], False)

    def test_geotiff_reader_is_optional(self) -> None:
        old_import = builtins.__import__
        def no_rasterio(name, *args, **kwargs):
            if name == "rasterio":
                raise ImportError(name)
            return old_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp, mock.patch("builtins.__import__", side_effect=no_rasterio):
            path = Path(tmp) / "image.tif"
            path.touch()
            self.assertEqual(inspect_artifact(path, "geotiff", self.config)["status"], "UNSUPPORTED")

    def test_checkpoint_does_not_import_torch_or_deserialize(self) -> None:
        path = self.root / "unsafe.pt"
        path.write_bytes(b"not a pytorch archive")
        old_import = builtins.__import__
        def no_torch(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("torch import is forbidden")
            return old_import(name, *args, **kwargs)
        with mock.patch("builtins.__import__", side_effect=no_torch):
            result = inspect_artifact(path, "checkpoint", self.config)
        self.assertEqual(result["summary"]["deserialization"], "not performed")


if __name__ == "__main__":
    unittest.main()

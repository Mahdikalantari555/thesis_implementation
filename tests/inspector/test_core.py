from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from tools.inspector.core.config import load_config
from tools.inspector.core.discovery import classify_role, iter_artifacts, make_record
from tools.inspector.core.manifest import ManifestVersionError, new_manifest, validate_manifest
from tools.inspector.core.config import InputRoot


class ConfigManifestTests(unittest.TestCase):
    def test_relative_paths_are_anchored_to_repository_not_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            config_dir = root / "tools" / "inspector"
            config_dir.mkdir(parents=True)
            config_path = config_dir / "config.toml"
            config_path.write_text('[paths]\ndata="data"\n[reporting]\noutput_dir="reports/inspector"\n', encoding="utf-8")
            config = load_config(config_path, root)
            self.assertEqual(config.roots[0].path, (root / "data").resolve())
            self.assertEqual(config.output_dir, (root / "reports" / "inspector").resolve())

    def test_external_roots_are_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"
            root.mkdir()
            external = Path(tmp) / "external archive"
            external.mkdir()
            config_path = root / "config.toml"
            config_path.write_text(f'[paths]\ndata="data"\nexternal_roots=[{{name="archive", path="{external}"}}]\n', encoding="utf-8")
            config = load_config(config_path, root)
            self.assertEqual(config.roots[-1].name, "archive")
            self.assertEqual(config.roots[-1].path, external.resolve())

    def test_manifest_version_is_enforced(self) -> None:
        self.assertEqual(validate_manifest(new_manifest("scan_x", []))["manifest_version"], 1)
        with self.assertRaises(ManifestVersionError):
            validate_manifest({"manifest_version": 99, "artifacts": []})

    def test_non_loopback_bind_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            config_path.write_text('[inspector]\nhost="0.0.0.0"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(config_path, root)

    def test_output_cannot_overlap_input_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.toml"
            config_path.write_text('[paths]\ndata="data"\n[reporting]\noutput_dir="data/cache"\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(config_path, root)


class DiscoveryTests(unittest.TestCase):
    def test_known_research_roles_are_classified_by_evidence(self) -> None:
        self.assertEqual(classify_role(Path("reports/dev/run/metrics.json"))[0], "evaluation")
        self.assertEqual(classify_role(Path("checkpoints/model.pt"))[0], "checkpoint")
        self.assertEqual(classify_role(Path("platform/src/data/farms.geojson"))[0], "platform_export")
        self.assertEqual(classify_role(Path("data/processed/cycles.parquet"))[0], "phenology")

    def _config(self, root: Path, output: Path) -> object:
        from tools.inspector.core.config import InspectorConfig
        return InspectorConfig(root, "test", "127.0.0.1", 8765, False, True,
                               (InputRoot("fixture", root),), output,
                               max_files=100, hash_small_files_under_bytes=0)

    def test_discovers_additions_and_skips_hidden_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            root.mkdir()
            out = Path(tmp) / "out"
            (root / "visible.json").write_text('{"value":1}', encoding="utf-8")
            (root / ".hidden.json").write_text('{"hidden":1}', encoding="utf-8")
            try:
                (root / "link.json").symlink_to(root / "visible.json")
            except OSError:
                pass
            config = self._config(root, out)
            found = list(iter_artifacts(config))
            self.assertEqual([item[1].name for item in found], ["visible.json"])

    def test_new_files_are_discovered_with_stable_opaque_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            root.mkdir()
            config = self._config(root, Path(tmp) / "out")
            self.assertEqual(list(iter_artifacts(config)), [])
            file = root / "new_run.json"
            file.write_text('{"model":"gru_ae"}', encoding="utf-8")
            first = list(iter_artifacts(config))[0]
            second = list(iter_artifacts(config))[0]
            record_a = make_record(*first[:2], first[2], 0)
            record_b = make_record(*second[:2], second[2], 0)
            self.assertEqual(record_a["id"], record_b["id"])
            self.assertEqual(record_a["format"], "json")

    def test_zarr_store_is_one_artifact_and_fingerprint_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            store = root / "cycles.zarr"
            store.mkdir(parents=True)
            (store / ".zgroup").write_text('{"zarr_format":2}', encoding="utf-8")
            (store / ".zattrs").write_text('{"a":1}', encoding="utf-8")
            (store / "chunk.bin").write_bytes(b"payload")
            found = list(iter_artifacts(self._config(root, Path(tmp) / "out")))
            self.assertEqual(len(found), 1)
            root_info, path, is_directory = found[0]
            first = make_record(root_info, path, is_directory, 0)["fingerprint"]
            (store / ".zattrs").write_text('{"a":2}', encoding="utf-8")
            second = make_record(root_info, path, is_directory, 0)["fingerprint"]
            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()

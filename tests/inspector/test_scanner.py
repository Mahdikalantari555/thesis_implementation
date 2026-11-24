from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.inspector.core.cache import Cache
from tools.inspector.core.config import InputRoot, InspectorConfig
from tools.inspector.core.scanner import ScanInProgress, Scanner
from tools.inspector.core.manifest import write_html_atomic


class ScannerTests(unittest.TestCase):
    def _config(self, root: Path, output: Path) -> InspectorConfig:
        return InspectorConfig(repo_root=root, title="Test Inspector", host="127.0.0.1", port=8765,
                               open_browser=False, read_only=True, roots=(InputRoot("fixture", root),),
                               output_dir=output, max_files=100, max_json_bytes=100_000,
                               hash_small_files_under_bytes=100_000)

    def test_incremental_scan_reuses_unchanged_result_and_history_is_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            root.mkdir()
            output = Path(tmp) / "inspector"
            source = root / "artifact.json"
            source.write_text('{"model":"gru_ae","metrics":{"auprc":0.4}}', encoding="utf-8")
            scanner = Scanner(self._config(root, output))
            try:
                first = scanner.scan()
                snapshot = output / "history" / first["scan_id"] / "artifacts.json"
                html_snapshot = output / "history" / first["scan_id"] / "overview.html"
                original_bytes = snapshot.read_bytes()
                original_html = html_snapshot.read_bytes()
                with patch("tools.inspector.core.scanner.inspect_artifact", side_effect=AssertionError("unchanged artifact was reopened")):
                    second = scanner.scan()
                self.assertEqual(second["changed_count"], 0)
                self.assertEqual(snapshot.read_bytes(), original_bytes)
                self.assertEqual(html_snapshot.read_bytes(), original_html)
                self.assertEqual(source.read_text(encoding="utf-8"), '{"model":"gru_ae","metrics":{"auprc":0.4}}')
                self.assertNotEqual(first["scan_id"], second["scan_id"])
            finally:
                scanner.close()

    def test_html_report_escapes_untrusted_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "report.html"
            write_html_atomic(target, {"artifact": "<script>alert(1)</script>"})
            text = target.read_text(encoding="utf-8")
            self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", text)
            self.assertNotIn("<script>alert(1)</script>", text)

    def test_nonfinite_json_values_do_not_crash_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            root.mkdir()
            output = Path(tmp) / "inspector"
            (root / "artifact.json").write_text('{"value":NaN}', encoding="utf-8")
            scanner = Scanner(self._config(root, output))
            try:
                report = scanner.scan()
                records = json.loads((output / "latest" / "artifacts.json").read_text(encoding="utf-8"))["artifacts"]
                self.assertEqual(report["artifact_count"], 1)
                self.assertIn(records[0]["status"], {"OK", "ERROR"})
                json.dumps(records, allow_nan=False)
            finally:
                scanner.close()

    def test_overlapping_refresh_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "input"
            root.mkdir()
            (root / "a.json").write_text('{"v":1}', encoding="utf-8")
            scanner = Scanner(self._config(root, Path(tmp) / "inspector"))
            started = threading.Event()
            errors: list[BaseException] = []
            def slow_adapter(*args, **kwargs):
                started.set()
                time.sleep(0.15)
                return {"status": "OK", "summary": {}}
            try:
                with patch("tools.inspector.core.scanner.inspect_artifact", side_effect=slow_adapter):
                    thread = threading.Thread(target=lambda: scanner.scan())
                    thread.start()
                    self.assertTrue(started.wait(1))
                    with self.assertRaises(ScanInProgress):
                        scanner.scan()
                    thread.join(2)
                self.assertFalse(thread.is_alive())
            finally:
                scanner.close()


if __name__ == "__main__":
    unittest.main()

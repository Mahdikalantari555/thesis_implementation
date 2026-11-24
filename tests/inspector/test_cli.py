from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "tools" / "inspector" / "app.py"


class CliTests(unittest.TestCase):
    def test_refresh_works_from_other_cwd_and_writes_inspector_output_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "archive"
            source.mkdir()
            (source / "eval.json").write_text('{"run_id":"r1","protocol":"dev","metrics":{"auprc":0.4}}', encoding="utf-8")
            output = base / "inspector-output"
            config = base / "config.toml"
            config.write_text(
                f'[inspector]\nopen_browser=false\n[paths]\nexternal_roots=[{{name="archive", path="{source}"}}]\n'
                f'[reporting]\noutput_dir="{output}"\n', encoding="utf-8")
            completed = subprocess.run([sys.executable, str(APP), "--refresh", "--config", str(config)],
                                       cwd=base, capture_output=True, text=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            report = json.loads(completed.stdout)
            self.assertEqual(report["artifact_count"], 1)
            self.assertTrue((output / "latest" / "overview.json").is_file())
            self.assertTrue((output / "latest" / "overview.html").is_file())
            self.assertTrue((output / "history" / report["scan_id"] / "overview.html").is_file())
            self.assertTrue((source / "eval.json").exists())

    def test_report_only_writes_report_and_does_not_start_http_server(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "archive"
            source.mkdir()
            output = base / "out"
            config = base / "config.toml"
            config.write_text(f'[paths]\nexternal_roots=[{{name="archive", path="{source}"}}]\n[reporting]\noutput_dir="{output}"\n', encoding="utf-8")
            result = subprocess.run([sys.executable, str(APP), "--report-only", "--config", str(config)],
                                    cwd=base, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "latest" / "overview.json").exists())

    def test_help_is_available_without_scanning(self) -> None:
        result = subprocess.run([sys.executable, str(APP), "--help"], cwd=REPO, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0)
        self.assertIn("--report-only", result.stdout)


if __name__ == "__main__":
    unittest.main()

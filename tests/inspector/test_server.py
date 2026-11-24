from __future__ import annotations

import hashlib
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from tools.inspector.core.config import InputRoot, InspectorConfig
from tools.inspector.core.scanner import Scanner
from tools.inspector.server import make_server


class ServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.root = base / "source"
        self.root.mkdir()
        self.output = base / "inspector"
        self.file = self.root / "unsafe.json"
        self.file.write_text('{"name":"<script>alert(1)</script>"}', encoding="utf-8")
        self.before_hash = hashlib.sha256(self.file.read_bytes()).hexdigest()
        config = InspectorConfig(repo_root=base, title="Test", host="127.0.0.1", port=0, open_browser=False,
                                read_only=True, roots=(InputRoot("fixture", self.root),), output_dir=self.output,
                                max_json_bytes=100_000, hash_small_files_under_bytes=100_000)
        self.scanner = Scanner(config)
        self.scanner.scan()
        self.server = make_server(self.scanner)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.scanner.close()
        self.tmp.cleanup()

    def test_status_and_artifact_api_are_read_only(self) -> None:
        with urllib.request.urlopen(self.url + "/api/report/latest.html") as response:
            html_report = response.read().decode("utf-8")
        self.assertIn("<pre>", html_report)
        self.assertNotIn("<script>", html_report)
        with urllib.request.urlopen(self.url + "/api/status") as response:
            self.assertEqual(response.status, 200)
        with urllib.request.urlopen(self.url + "/api/artifacts") as response:
            data = response.read().decode("utf-8")
        self.assertIn("unsafe.json", data)
        self.assertEqual(hashlib.sha256(self.file.read_bytes()).hexdigest(), self.before_hash)

    def test_arbitrary_filesystem_path_is_not_served(self) -> None:
        request = urllib.request.Request(self.url + "/api/artifact/../../etc/passwd")
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(request)
        self.assertIn(context.exception.code, {400, 404})
        with self.assertRaises(urllib.error.HTTPError):
            urllib.request.urlopen(self.url + "/etc/passwd")

    def test_series_route_requires_constrained_farm_id_and_handles_missing_root(self) -> None:
        with urllib.request.urlopen(self.url + "/api/series?farm_id=F-001") as response:
            self.assertEqual(response.status, 200)
            payload = response.read().decode("utf-8")
        self.assertIn("UNAVAILABLE", payload)
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(self.url + "/api/series?farm_id=../../etc/passwd")
        self.assertEqual(context.exception.code, 400)

    def test_external_host_header_is_rejected(self) -> None:
        request = urllib.request.Request(self.url + "/api/status", headers={"Host": "attacker.example"})
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(request)
        self.assertEqual(context.exception.code, 400)

    def test_static_ui_uses_no_external_script_or_unsafe_html_sink(self) -> None:
        with urllib.request.urlopen(self.url + "/app.js") as response:
            script = response.read().decode("utf-8")
        self.assertNotIn("innerHTML", script)
        self.assertIn("textContent", script)
        self.assertIn("/api/series?farm_id=", script)
        self.assertIn("item.stress_flag === 'suspected_stress'", script)
        self.assertIn("item.split !== 'test'", script)
        with urllib.request.urlopen(self.url + "/") as response:
            html = response.read().decode("utf-8")
        self.assertNotIn("https://", html)
        for view_id in ('overview', 'data', 'models', 'spatial', 'reports'):
            self.assertIn(f'id="view-{view_id}"', html)

    def test_only_refresh_post_is_mutating_and_source_hash_is_stable(self) -> None:
        request = urllib.request.Request(self.url + "/api/refresh", method="POST", data=b"{}",
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 200)
        self.assertEqual(hashlib.sha256(self.file.read_bytes()).hexdigest(), self.before_hash)
        delete = urllib.request.Request(self.url + "/api/refresh", method="DELETE")
        with self.assertRaises(urllib.error.HTTPError) as context:
            urllib.request.urlopen(delete)
        self.assertIn(context.exception.code, {405, 501})


if __name__ == "__main__":
    unittest.main()

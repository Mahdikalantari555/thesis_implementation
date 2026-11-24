"""Loopback-only HTTP API and static UI server."""
from __future__ import annotations

import json
import logging
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from .core.scanner import ScanInProgress, Scanner
from .core.series import farm_series
from .core.spatial import farm_map

ID_PATTERN = re.compile(r"^[0-9a-f]{24}$")
STATIC_FILES = {"/": ("index.html", "text/html; charset=utf-8"),
                "/index.html": ("index.html", "text/html; charset=utf-8"),
                "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                "/app.js": ("app.js", "text/javascript; charset=utf-8")}
UI_DIR = Path(__file__).resolve().parent / "ui"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8"))


class InspectorHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], scanner: Scanner):
        self.scanner = scanner
        super().__init__(address, InspectorRequestHandler)


class InspectorRequestHandler(BaseHTTPRequestHandler):
    server: InspectorHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("http %s - %s", self.address_string(), fmt % args)

    def _host_allowed(self) -> bool:
        raw = self.headers.get("Host", "").lower().strip()
        if raw.startswith("[") and "]" in raw:
            host = raw[1:raw.index("]")]
        else:
            host = raw.rsplit(":", 1)[0] if raw.count(":") == 1 else raw
        return host in {"127.0.0.1", "localhost", "::1"}

    def _send(self, status: int, body: bytes, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; connect-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, value: Any) -> None:
        try:
            body = _json_bytes(value)
        except (TypeError, ValueError) as exc:
            logging.exception("Failed to encode API response")
            body = _json_bytes({"error": f"Response serialization failed: {exc.__class__.__name__}"})
            status = 500
        self._send(status, body)

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._json(400, {"error": "Host header must be loopback"})
            return
        path = unquote(urlsplit(self.path).path)
        if path in STATIC_FILES:
            file_name, mime = STATIC_FILES[path]
            self._send(200, (UI_DIR / file_name).read_bytes(), mime)
            return
        scanner = self.server.scanner
        if path == "/api/status":
            self._json(200, scanner.status())
        elif path == "/api/artifacts":
            self._json(200, {"artifacts": scanner.artifacts()})
        elif path.startswith("/api/artifact/"):
            artifact_id = path.removeprefix("/api/artifact/")
            if not ID_PATTERN.fullmatch(artifact_id):
                self._json(400, {"error": "Invalid artifact ID"})
                return
            artifact = scanner.get_artifact(artifact_id)
            self._json(200 if artifact else 404, artifact or {"error": "Artifact not found"})
        elif path == "/api/metrics":
            self._json(200, scanner.latest_report("evaluation") or {"studies": {}})
        elif path == "/api/quality":
            self._json(200, scanner.latest_report("quality") or {"artifact_count": 0, "status_counts": {}})
        elif path == "/api/map":
            self._json(200, farm_map(scanner.config))
        elif path == "/api/series":
            query = parse_qs(urlsplit(self.path).query)
            farm_id = query.get("farm_id", [None])[0]
            if not farm_id or len(farm_id) > 64 or not re.fullmatch(r"F-\d{3,}", farm_id):
                self._json(400, {"available": False, "status": "INVALID_ID", "reason": "A valid farm_id is required", "observations": []})
                return
            self._json(200, farm_series(scanner.config, farm_id))
        elif path == "/api/report/latest":
            self._json(200, scanner.latest_report("overview") or {"error": "No report has been generated"})
        elif path == "/api/report/latest.html":
            report_html = scanner.latest_html_report()
            if report_html is None:
                self._json(404, {"error": "No HTML report has been generated"})
            else:
                self._send(200, report_html.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/history":
            self._json(200, {"history": scanner.cache.list_scans()})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._json(400, {"error": "Host header must be loopback"})
            return
        if urlsplit(self.path).path != "/api/refresh":
            self._json(404, {"error": "Not found"})
            return
        try:
            report = self.server.scanner.scan()
        except ScanInProgress as exc:
            self._json(409, {"error": str(exc), "scan_in_progress": True})
            return
        except Exception as exc:
            logging.exception("Refresh failed")
            self._json(500, {"error": f"Refresh failed: {exc.__class__.__name__}"})
            return
        self._json(200, report)


def make_server(scanner: Scanner) -> InspectorHTTPServer:
    return InspectorHTTPServer((scanner.config.host, scanner.config.port), scanner)

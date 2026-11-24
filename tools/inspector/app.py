#!/usr/bin/env python3
"""Start the local Thesis Inspector.

Run from any working directory with: python tools/inspector/app.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.inspector.core.config import load_config  # noqa: E402
from tools.inspector.core.scanner import Scanner  # noqa: E402
from tools.inspector.server import make_server  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.toml",
                        help="TOML config file (default: tools/inspector/config.toml)")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--refresh", action="store_true", help="Run one manual scan and exit")
    modes.add_argument("--report-only", action="store_true", help="Run one scan, write reports, and exit without serving UI")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser in server mode")
    return parser.parse_args(argv)


def _repo_root(config_path: Path) -> Path:
    # Repository root is anchored to this app's location, not the caller's cwd.
    return REPO_ROOT


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    try:
        config = load_config(config_path, _repo_root(config_path))
        if not config.read_only:
            raise ValueError("Inspector is read-only; set inspector.read_only = true")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            handlers=[logging.FileHandler(config.output_dir / "inspector.log", encoding="utf-8"), logging.StreamHandler()],
        )
    except (OSError, ValueError) as exc:
        print(f"Thesis Inspector configuration error: {exc}", file=sys.stderr)
        return 2

    scanner = Scanner(config)
    try:
        try:
            report = scanner.scan()
        except Exception:
            logging.exception("Initial scan failed")
            raise
        if args.refresh or args.report_only:
            print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
            return 0
        server = make_server(scanner)
        url = f"http://127.0.0.1:{server.server_address[1]}/"
        print(f"{config.title} is running locally at {url}")
        if config.open_browser and not args.no_browser:
            webbrowser.open(url, new=2)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping Thesis Inspector…")
        finally:
            server.server_close()
        return 0
    except OSError as exc:
        logging.exception("Inspector startup failed")
        print(f"Thesis Inspector could not start: {exc}", file=sys.stderr)
        return 1
    finally:
        scanner.close()


if __name__ == "__main__":
    raise SystemExit(main())

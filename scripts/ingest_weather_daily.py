#!/usr/bin/env python
"""Verify/ingest daily ERA5 (weather_daily.parquet) into the training path."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THESIS = REPO.parents[0]
IMPL_TARGET = REPO / "data" / "processed" / "weather_daily.parquet"
CANON_TARGET = THESIS / "Data" / "Processed" / "weather" / "weather_daily.parquet"
MARKER_DIR = REPO / "reports" / "data_gaps"
MARKER = MARKER_DIR / "weather_daily.deferred.json"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return "unknown"


def find_sources() -> list[Path]:
    found: list[Path] = []
    for p in (IMPL_TARGET, CANON_TARGET):
        if p.is_file():
            found.append(p)
    for root in (REPO / "data", THESIS / "Data"):
        if root.exists():
            for p in root.rglob("weather_daily*.parquet"):
                if p not in found:
                    found.append(p)
    gd = REPO / "google_drive"
    if gd.exists():
        for p in sorted(gd.glob("weather_daily*.zip")) + sorted(gd.glob("*weather*daily*.zip")):
            if p not in found:
                found.append(p)
    return found


def extract_zip(zip_path: Path, dest_dir: Path) -> Path | None:
    with zipfile.ZipFile(zip_path) as z:
        candidates = [n for n in z.namelist() if n.endswith("weather_daily.parquet")]
        if not candidates:
            return None
        cand = sorted(candidates, key=lambda x: (x.count("/"), len(x)))[0]
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / Path(cand).name
        with z.open(cand) as src, out.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        return out


def write_marker(searched, found, reason):
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "weather_daily.parquet (ERA5 daily covariates)",
        "config_key": "artifacts.weather_daily_v2",
        "status": "Deferred",
        "reason": reason,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "searched_paths": searched,
        "found_sources": found,
        "targets": {"impl": str(IMPL_TARGET), "canonical": str(CANON_TARGET)},
        "owner_action": "Upload google_drive/weather_daily_v2.zip containing weather_daily.parquet, then re-run this script.",
    }
    MARKER.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    searched = [str(IMPL_TARGET), str(CANON_TARGET), str(REPO / "data"), str(THESIS / "Data"), str(REPO / "google_drive")]
    sources = find_sources()
    zip_sources = [p for p in sources if p.suffix.lower() == ".zip"]
    parquet_sources = [p for p in sources if p.suffix.lower() == ".parquet"]
    found_str = [str(p) for p in sources]

    if not sources:
        reason = "No weather_daily.parquet and no weather_daily*.zip found in any searched location; owner upload google_drive/weather_daily_v2.zip is missing."
        write_marker(searched, found_str, reason)
        print("DEFERRED: " + reason)
        print("Marker written: " + str(MARKER))
        return 0

    extracted = list(parquet_sources)
    for zp in zip_sources:
        tmp_dir = MARKER_DIR / "_extract_weather"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        ep = extract_zip(zp, tmp_dir)
        if ep is not None:
            extracted.append(ep)
            print("Extracted " + str(ep) + " from " + str(zp))

    if not extracted:
        reason = "weather_daily*.zip found but contained no weather_daily.parquet entry; cannot ingest."
        write_marker(searched, found_str, reason)
        print("DEFERRED: " + reason)
        print("Marker written: " + str(MARKER))
        return 0

    src = extracted[0]
    if args.dry_run:
        print("Would ingest from " + str(src) + " -> " + str(IMPL_TARGET) + " and " + str(CANON_TARGET))
        return 0

    for target in (IMPL_TARGET, CANON_TARGET):
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        print("Ingested -> " + str(target))

    if MARKER.exists():
        MARKER.unlink()
        print("Removed stale deferred marker " + str(MARKER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

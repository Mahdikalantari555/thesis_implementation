#!/usr/bin/env python
"""Mark ground/confirmed-anomaly labels and STL features as Deferred.

Writes Deferred markers under reports/data_gaps/ with explicit reasons.
No data is fabricated; no numbers invented.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THESIS = REPO.parents[0]
MARKER_DIR = REPO / "reports" / "data_gaps"


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return "unknown"


def write_marker(name: str, payload: dict) -> Path:
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    path = MARKER_DIR / (name + ".deferred.json")
    payload = {"status": "Deferred", "checked_at": datetime.now(timezone.utc).isoformat(),
               "git_sha": git_sha(), **payload}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["ground_labels", "stl_features", "all"], default="all")
    args = parser.parse_args()

    sha = git_sha()
    targets = []

    if args.only in ("ground_labels", "all"):
        # Search for any ground-label artifact.
        found = []
        for root in (REPO, THESIS):
            if root.exists():
                for p in root.rglob("ground_label*"):
                    found.append(str(p))
        for root in (REPO / "google_drive",):
            if root.exists():
                for p in root.glob("ground_labels*.zip"):
                    found.append(str(p))
        p = write_marker("ground_labels", {
            "artifact": "ground/confirmed-anomaly labels",
            "config_key": "n/a (no config binding; todo owner task 4)",
            "reason": ("Owner upload missing: google_drive/ground_labels_v2.zip; "
                       "no ground_labels*.zip or label parquet exists anywhere. "
                       "All L4 claims stay unsupported until uploaded."),
            "found_sources": found,
            "searched_roots": [str(REPO), str(THESIS)],
            "owner_action": "Upload google_drive/ground_labels_v2.zip (owner task 4); extract to Data/Processed/ground_labels or thesis_implementation/data/processed.",
            "impact": "L4 claims unsupported; environmental-correlation tables print absent/shared-input only.",
        })
        targets.append(p)
        print("DEFERRED ground_labels -> " + str(p))

    if args.only in ("stl_features", "all"):
        found = []
        for root in (REPO, THESIS):
            if root.exists():
                for p in root.rglob("stl_features*"):
                    found.append(str(p))
        p = write_marker("stl_features", {
            "artifact": "decomposition/stl_features.parquet",
            "config_key": "artifacts.stl_features",
            "reason": ("Optional, unproduced by default (B5-class STL features from "
                       "decomposition-aware-ssl-study). No owner upload present."),
            "found_sources": found,
            "searched_roots": [str(REPO), str(THESIS)],
            "owner_action": "Optional: upload google_drive/stl_features_v2.zip, or explicitly decline; leave required_for: optional.",
            "impact": "Blocks nothing; reported as absent.",
            "config_path": "data/processed/decomposition/stl_features.parquet",
        })
        targets.append(p)
        print("DEFERRED stl_features -> " + str(p))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

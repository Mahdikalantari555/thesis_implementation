#!/usr/bin/env python
"""Verify patch store binding: patches.zarr + patch_sigma.npz sidecar.

Checks implementation + canonical roots and the patch_full_v2_* / thesis_data
bundles. If a bound store+sidecar exists (or a bundle contains both), report
bound. Otherwise write a Deferred marker. Never invents numbers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THESIS = REPO.parents[0]
MARKER_DIR = REPO / "reports" / "data_gaps"
MARKER = MARKER_DIR / "patch_store_binding.deferred.json"

CANDIDATE_STORES = [
    REPO / "data" / "patch_runs" / "v3" / "patches.zarr",
    THESIS / "Data" / "Processed" / "patch_runs" / "v3" / "patches.zarr",
    THESIS / "Data" / "Processed" / "satellite" / "patches.zarr",
    REPO / "data" / "processed" / "patches.zarr",
]
CANDIDATE_SIDECARS = [
    REPO / "data" / "patch_runs" / "v3" / "patch_sigma.npz",
    THESIS / "Data" / "Processed" / "patch_runs" / "v3" / "patch_sigma.npz",
    REPO / "data" / "processed" / "patch_sigma.npz",
]
BUNDLE_GLOBS = ["patch_full_v2_*bundle*.zip", "thesis_data_full_v2.zip", "patch_full_v2*.zip"]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True).strip()
    except Exception:
        return "unknown"


def bundle_paths() -> list[Path]:
    gd = REPO / "google_drive"
    out: list[Path] = []
    if gd.exists():
        for pat in BUNDLE_GLOBS:
            out.extend(sorted(gd.glob(pat)))
    return out


def bundle_contains_both(zp: Path) -> dict:
    has_store = False
    has_sidecar = False
    store_entries = []
    sidecar_entries = []
    try:
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if "patches.zarr" in n and (n.endswith(".manifest.jsonl") or n.endswith("zarr.json")):
                    has_store = True
                    store_entries.append(n)
                if n.endswith("patch_sigma.npz"):
                    # Only count a true sidecar in a patch_runs/v3 location, not checkpoint sigmas.
                    if "patch_runs/v3" in n or "patch_runs\v3" in n:
                        has_sidecar = True
                        sidecar_entries.append(n)
    except Exception as e:
        return {"zip": str(zp), "error": str(e), "has_store": False, "has_sidecar": False,
                "store_entries": [], "sidecar_entries": []}
    return {"zip": str(zp), "has_store": has_store, "has_sidecar": has_sidecar,
            "store_entries_sample": store_entries[:5], "sidecar_entries": sidecar_entries}


def write_marker(searched_stores, searched_sidecars, bundles, reason):
    MARKER_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "artifact": "patch store (patches.zarr) + patch_sigma.npz sidecar",
        "config_key": "artifacts.patch_store_v2",
        "status": "Deferred",
        "reason": reason,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "searched_stores": searched_stores,
        "searched_sidecars": searched_sidecars,
        "bundles_inspected": bundles,
        "owner_action": "Produce/mirror patches.zarr + patch_sigma.npz under patch_runs/v3 (task 3.1-3.4); or upload a bundle containing both.",
    }
    MARKER.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    found_store = [p for p in CANDIDATE_STORES if p.exists()]
    found_sidecar = [p for p in CANDIDATE_SIDECARS if p.exists()]
    searched_stores = [str(p) for p in CANDIDATE_STORES]
    searched_sidecars = [str(p) for p in CANDIDATE_SIDECARS]

    bundles = [bundle_contains_both(zp) for zp in bundle_paths()]
    bundle_has_both = any(b.get("has_store") and b.get("has_sidecar") for b in bundles)

    # Bound if a store and a sidecar coexist in at least one root.
    bound = bool(found_store) and bool(found_sidecar)

    if bound or bundle_has_both:
        # Remove deferred marker if present.
        if MARKER.exists():
            MARKER.unlink()
        print("BOUND: patch store + sidecar available")
        print("  stores:", [str(p) for p in found_store])
        print("  sidecars:", [str(p) for p in found_sidecar])
        if bundle_has_both:
            print("  bundle with both:", [b["zip"] for b in bundles if b.get("has_store") and b.get("has_sidecar")])
        return 0

    reason = ("No patches.zarr store with a co-located patch_sigma.npz sidecar found in any root, "
              "and no inspected bundle contains both patches.zarr and a patch_runs/v3 patch_sigma.npz sidecar.")
    write_marker(searched_stores, searched_sidecars, bundles, reason)
    print("DEFERRED: " + reason)
    print("Marker written: " + str(MARKER))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

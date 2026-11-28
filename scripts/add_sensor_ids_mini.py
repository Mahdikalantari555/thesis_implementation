"""Add per-observation ``sensor_id`` to the mini satellite store (task 0A.3).

The mini store records one scene per date from ``scene_inventory.parquet``;
the sensor (LC08 → L8, LC09 → L9) is parsed per scene id, so each date gets
exactly one sensor. The array is ADDITIVE — ``data``, ``pixel_valid``,
``in_field`` and all existing attrs are untouched — and the write is
validated fail-closed:

* every store date must exist in the inventory (no silent drop);
* a date whose scenes disagree on sensor aborts the run (no majority vote:
  mixing two sensors under one observation id would be a fabricated fact);
* an existing ``sensor_id`` array is accepted only when byte-identical.

Usage (owner-runnable; the full run is seconds)::

    python scripts/add_sensor_ids_mini.py --smoke 3      # validate, no write
    python scripts/add_sensor_ids_mini.py                # write all farms
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402
from ssl_anomaly.utils.paths import localize  # noqa: E402

INVENTORY = localize(
    "D:/RS/Thesis/Data/Processed/satellite/scene_inventory.parquet"
)
STORE_ROOT = localize("D:/RS/Thesis/Data/Processed/mini_research_dataset")
SENSOR_LEGEND = {0: "L8", 1: "L9"}


def date_sensor_table() -> pd.DataFrame:
    """One row per acquisition date with its parsed sensor code (0=L8, 1=L9)."""
    si = pd.read_parquet(INVENTORY)
    si["date"] = si["acquisition_date"].astype(str).str.slice(0, 10)
    si["code"] = si["scene_id"].str.slice(0, 5).map({"LC08_": 0, "LC09_": 1})
    unmapped = si[si["code"].isna()]
    if len(unmapped):
        raise RuntimeError(
            f"scene ids with unknown sensor prefix: "
            f"{sorted(set(unmapped['scene_id'].str.slice(0, 5)))[:5]} … "
            f"({len(unmapped)} rows) — refusing to guess"
        )
    per_date = si.groupby("date")["code"].nunique()
    mixed = per_date[per_date > 1]
    if len(mixed):
        raise RuntimeError(
            f"{len(mixed)} dates carry scenes from BOTH sensors "
            f"(e.g. {mixed.index[0]}); a single sensor_id per observation "
            "would be a fabrication — resolve upstream first"
        )
    out = si.drop_duplicates("date")[["date", "code"]]
    if len(out) != si["date"].nunique():
        raise RuntimeError("duplicate inventory rows per date after dedup")
    return out.set_index("date")


def build_sensor_id(dates: list[str], table: pd.DataFrame) -> np.ndarray:
    """Per-date sensor codes; missing dates abort (never filled)."""
    missing = [d for d in dates if d not in table.index]
    if missing:
        raise RuntimeError(
            f"{len(missing)} store dates are absent from the scene inventory "
            f"(first: {missing[0]}) — no sensor can be assigned without "
            "fabrication"
        )
    return table.loc[dates, "code"].to_numpy(dtype=np.int8)


def process_farm(group, table: pd.DataFrame, *, write: bool, limit: int | None = None) -> dict:
    dates = [str(d) for d in group["dates"][:]]
    if limit is not None:
        dates = dates[:limit]
    codes = build_sensor_id(dates, table)
    rec = {
        "n_dates": len(dates),
        "n_l8": int((codes == 0).sum()),
        "n_l9": int((codes == 1).sum()),
        "sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
    }
    existing = None
    if "sensor_id" in group:
        existing = np.asarray(group["sensor_id"][:])
    if existing is not None:
        same = existing.shape == codes.shape and bool((existing == codes).all())
        rec["existing_matches"] = same
        if not same:
            raise RuntimeError(
                f"farm {group.attrs['farm_id']}: an existing sensor_id array "
                "differs from the inventory-derived one — refusing to overwrite"
            )
        rec["action"] = "already-present-identical"
        return rec
    if write:
        group.create_dataset(
            "sensor_id", data=codes, shape=codes.shape, dtype="i1",
            overwrite=False,
        )
        group.attrs["sensor_id_legend"] = SENSOR_LEGEND
        group.attrs["sensor_id_source"] = "scene_inventory.parquet scene_id prefix (LC08→L8, LC09→L9)"
        group.attrs["sensor_id_added_utc"] = datetime.now(timezone.utc).isoformat()
        rec["action"] = "written"
    else:
        rec["action"] = "validated-only"
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="validate the first N dates per farm and exit (no write)")
    ap.add_argument("--dry-run", action="store_true", help="full validation, no write")
    args = ap.parse_args()

    t0 = time.monotonic()
    table = date_sensor_table()
    print(f"inventory: {len(table)} unique dates "
          f"(L8={int((table['code'] == 0).sum())}, L9={int((table['code'] == 1).sum())})",
          flush=True)

    import zarr

    store = STORE_ROOT / "satellite.zarr"
    farms = sorted(p for p in store.iterdir() if (p / ".zgroup").exists())
    results: dict[str, dict] = {}
    write = not args.dry_run and args.smoke is None
    for farm in farms:
        g = zarr.open_group(str(farm), mode="r+" if write else "r")
        results[g.attrs["farm_id"]] = process_farm(g, table, write=write, limit=args.smoke)
        print(f"  {g.attrs['farm_id']}: {results[g.attrs['farm_id']]}", flush=True)

    if write:
        sidecar = STORE_ROOT / "sensor_id_manifest.json"
        payload = {
            "schema": "mini-store-sensor-id/v1",
            "added_utc": datetime.now(timezone.utc).isoformat(),
            "git_sha": get_git_commit_sha(),
            "inventory": str(INVENTORY),
            "legend": SENSOR_LEGEND,
            "farms": results,
            "wall_s": round(time.monotonic() - t0, 2),
            "note": "additive per-observation sensor ids; data arrays untouched",
        }
        sidecar.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"WROTE sensor_id arrays + {sidecar}", flush=True)
    else:
        print("VALIDATION ONLY (no write)", flush=True)
    print(f"DONE wall={time.monotonic() - t0:.2f}s farms={len(farms)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the mini weather table and feature-group manifest.

Reuses ``weather_aligned.parquet`` verbatim (56 weather features + date),
filters to the 538 mini satellite dates by left join, and adds two derived
columns: ``gdd_daily`` (base 10 °C from T min/max) and ``heat_stress_hours``
(count of hourly T > 35 °C per UTC calendar day). Derived columns are new and
never overwrite raw ones.

Usage::

    python scripts/build_mini_weather.py
    python scripts/build_mini_weather.py --out <dir> --min-dates 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402

DATA = Path("D:/RS/Thesis/Data")
WEATHER_ALIGNED = DATA / "Processed" / "weather" / "weather_aligned.parquet"
HOURLY = DATA / "Weather" / "raw" / "point_01_raw_hourly.parquet"
INVENTORY = DATA / "Processed" / "satellite" / "scene_inventory.parquet"
OUT_ROOT = DATA / "Processed" / "mini_research_dataset"
DATE_COL = "acquisition_date"
GDD_BASE_C = 10.0
HEAT_THRESHOLD_C = 35.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def mini_dates(inventory: Path, limit: int | None = None) -> list[str]:
    df = pd.read_parquet(inventory)
    dates = sorted(str(d)[:10] for d in df[DATE_COL].astype(str))
    return dates[:limit] if limit else dates


def derive_heat_stress_hours(hourly_path: Path, threshold: float = HEAT_THRESHOLD_C) -> pd.DataFrame:
    """Return (date, heat_stress_hours) for UTC calendar days with hourly T."""
    df = pd.read_parquet(hourly_path, columns=["date", "temperature_2m"])
    ts = pd.to_datetime(df["date"], utc=True)
    hot = df.loc[df["temperature_2m"].astype(float) > threshold].copy()
    hot["date_key"] = ts.loc[hot.index].dt.strftime("%Y-%m-%d")
    out = hot.groupby("date_key", sort=True).size().rename("heat_stress_hours").reset_index()
    return out.rename(columns={"date_key": DATE_COL})


def build_weather(
    aligned_path: Path,
    hourly_path: Path,
    dates: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    t0 = time.monotonic()
    raw = pd.read_parquet(aligned_path)
    if DATE_COL not in raw.columns:
        raise RuntimeError(f"{aligned_path} has no {DATE_COL}")
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL]).dt.strftime("%Y-%m-%d")
    if raw[DATE_COL].duplicated().any():
        raise RuntimeError("Duplicate dates in weather_aligned.parquet")

    mini = raw[raw[DATE_COL].isin(dates)].copy()
    mini = mini.sort_values(DATE_COL).reset_index(drop=True)
    missing = sorted(set(dates) - set(mini[DATE_COL]))
    if missing:
        raise RuntimeError(f"Weather missing for mini dates: {missing}")

    feature_cols = [c for c in raw.columns if c != DATE_COL]
    rolling_cols = [c for c in feature_cols if c.endswith(("_3d", "_7d", "_14d", "_14d", "_30d"))]
    rolling_cols = list(dict.fromkeys(rolling_cols))
    core_cols = [c for c in feature_cols if c not in rolling_cols]

    if not {"temperature_min", "temperature_max"}.issubset(feature_cols):
        raise RuntimeError("T min/max columns absent; gdd_daily cannot be derived")
    tmean = (mini["temperature_min"].astype(float) + mini["temperature_max"].astype(float)) / 2.0
    mini["gdd_daily"] = (tmean - GDD_BASE_C).clip(lower=0.0).astype(float)

    heat = derive_heat_stress_hours(hourly_path)
    heat_map = dict(zip(heat[DATE_COL], heat["heat_stress_hours"]))
    # A day with no hour above threshold is 0, not absent: the hourly source
    # covers the whole period and the count is well defined.
    mini["heat_stress_hours"] = mini[DATE_COL].map(heat_map).fillna(0).astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "weather_aligned_mini.parquet"
    mini.to_parquet(table_path, index=False)

    groups = {
        "schema": "mini-weather-features/v1",
        "derived_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_commit_sha(),
        "date_column": DATE_COL,
        "n_rows": int(len(mini)),
        "first_date": str(mini[DATE_COL].iloc[0]),
        "last_date": str(mini[DATE_COL].iloc[-1]),
        "source_paths": {
            "aligned": str(aligned_path),
            "aligned_sha256": _sha256(aligned_path),
            "hourly": str(hourly_path),
            "hourly_sha256": _sha256(hourly_path),
        },
        "derivations": {
            "gdd_daily": {
                "formula": "max(0, (temperature_min + temperature_max)/2 - 10.0)",
                "base_c": GDD_BASE_C,
                "unit": "degC-day",
                "provenance": "derived from ERA5 daily T min/max (reanalysis)",
            },
            "heat_stress_hours": {
                "formula": "count(hourly temperature_2m > 35.0 C per UTC calendar date)",
                "threshold_c": HEAT_THRESHOLD_C,
                "unit": "hours",
                "provenance": "derived from ERA5 hourly T2m (reanalysis, UTC dates)",
                "note": "zero is a measured absence of hot hours, not a missing value",
            },
        },
        "groups": {
            "spectral_only": [],
            "weather_core": core_cols,
            "weather_rolling": rolling_cols,
            "weather_all": feature_cols,
            "weather_plus_derived": feature_cols + ["gdd_daily", "heat_stress_hours"],
        },
        "n_raw_features": len(feature_cols),
        "n_derived_features": 2,
    }
    features_path = out_dir / "weather_features.json"
    features_path.write_text(json.dumps(groups, indent=2) + "\n", encoding="utf-8")

    report = {
        "schema": "mini-weather-build/v1",
        "git_sha": get_git_commit_sha(),
        "n_rows": int(len(mini)),
        "n_raw_features": len(feature_cols),
        "n_core": len(core_cols),
        "n_rolling": len(rolling_cols),
        "gdd_nonnull": int(mini["gdd_daily"].notna().sum()),
        "gdd_mean": float(mini["gdd_daily"].mean()),
        "heat_hours_nonnull": int(mini["heat_stress_hours"].notna().sum()),
        "heat_hours_mean": float(mini["heat_stress_hours"].mean()),
        "heat_hours_max": int(mini["heat_stress_hours"].max()),
        "wall_s": round(time.monotonic() - t0, 2),
        "table": str(table_path),
        "features_json": str(features_path),
    }
    report_path = out_dir / "weather_build_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("WEATHER COMPLETE " + json.dumps(report), flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aligned", type=Path, default=WEATHER_ALIGNED)
    ap.add_argument("--hourly", type=Path, default=HOURLY)
    ap.add_argument("--inventory", type=Path, default=INVENTORY)
    ap.add_argument("--out", type=Path, default=OUT_ROOT / "weather")
    ap.add_argument("--min-dates", type=int, default=None, help="Smoke: first N dates")
    args = ap.parse_args(argv)
    dates = mini_dates(args.inventory, args.min_dates)
    print(f"Mini weather: {len(dates)} dates ({dates[0]} .. {dates[-1]})", flush=True)
    build_weather(args.aligned, args.hourly, dates, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

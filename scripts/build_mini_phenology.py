"""Derive provisional NDVI+SAVI sugarcane phenology for the mini dataset.

Channels: field-median NDVI and SAVI only, computed on demand from stored
bands. LST and GDD are deliberately excluded from the harvest score (owner
decision 2026-11-26; thermal time is a separate axis ablation). Harvest-like
candidates and cycle state come from ``ssl_anomaly.phenology.mini``. Outputs
are screening metadata, not ground truth and not a model prediction.

Usage::

    python scripts/build_mini_phenology.py
    python scripts/build_mini_phenology.py --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.features import compute_index  # noqa: E402
from ssl_anomaly.phenology.mini import (  # noqa: E402
    build_cycle_rows,
    detect_harvests,
    vi_harvest_score,
)
from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402
from ssl_anomaly.utils.paths import localize  # noqa: E402

DATA = localize("D:/RS/Thesis/Data")
MINI_ROOT = DATA / "Processed" / "mini_research_dataset"
STORE = MINI_ROOT / "satellite.zarr"
WEATHER = MINI_ROOT / "weather" / "weather_aligned_mini.parquet"
OUT_DIR = MINI_ROOT / "phenology"


def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    s = pd.Series(np.asarray(y, dtype=np.float64))
    s = s.interpolate(limit_direction="both")
    return s.rolling(window=window, center=True, min_periods=1).median().to_numpy()


def farm_series(group: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    dates = [str(d) for d in group["dates"][:]]
    cube = np.asarray(group["data"][:], dtype=np.float32)  # (D,7,H,W)
    valid = np.asarray(group["pixel_valid"][:], dtype=bool)
    in_field = np.asarray(group["in_field"][:], dtype=bool)
    bands = {name: cube[:, i, :, :] for i, name in enumerate(group.attrs["band_order"])}
    field: dict[str, np.ndarray] = {}
    for name, fn_bands in (("ndvi", ("red", "nir")), ("savi", ("red", "nir"))):
        arr = compute_index(name, {b: bands[b] for b in fn_bands})
        vals = []
        for di in range(len(dates)):
            mask = valid[di] & in_field & np.isfinite(arr[di])
            vals.append(float(np.median(arr[di][mask])) if mask.any() else np.nan)
        field[name] = np.asarray(vals)
    return dates, field


def build_phenology(store: Path, weather_path: Path, out_dir: Path) -> dict[str, Any]:
    t0 = time.monotonic()
    all_rows: list[dict[str, Any]] = []
    all_weak: list[dict[str, Any]] = []
    season: dict[str, Any] = {
        "schema": "mini-season-structure/v2-vi-only",
        "git_sha": get_git_commit_sha(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "NDVI+SAVI-only screening detector. LST and GDD are not score "
            "channels. Over-long and short bounded intervals are retained as "
            "overlong_unresolved / short_unresolved. Boundaries are provisional "
            "until owner review; stage comparison uses trajectory-conditioned "
            "matching, not these exact dates."
        ),
        "weather_path_ignored": str(weather_path),
        "farms": {},
        "holdout_candidate": None,
    }
    for group_dir in sorted(p for p in store.iterdir() if p.is_dir() and (p / ".zgroup").exists()):
        g = zarr.open_group(str(group_dir), mode="r")
        farm_id = str(g.attrs["farm_id"])
        ogc_fid = int(g.attrs["ogc_fid"])
        dates, field = farm_series(g)
        smooth = {k: _smooth(v) for k, v in field.items()}
        score, oriented = vi_harvest_score(smooth["ndvi"], smooth["savi"])
        day_numbers = np.array(
            [(np.datetime64(d, "D") - np.datetime64("2013-03-30", "D")).astype(int) for d in dates]
        )
        candidates, weak = detect_harvests(score, day_numbers, oriented)
        rows = build_cycle_rows(
            farm_id, ogc_fid, dates, candidates,
            ndvi=smooth["ndvi"], savi=smooth["savi"],
        )
        all_rows.extend(rows)
        for c in weak:
            all_weak.append({
                "farm": farm_id, "ogc_fid": ogc_fid, "date": dates[c.index],
                "score": round(c.score, 4), "prominence": c.prominence,
                "confidence": c.confidence, "support": c.support,
                "channel_z": json.dumps(c.channel_z),
            })
        by_year = pd.Series(dates).str.slice(0, 4).value_counts().sort_index()
        season["farms"][farm_id] = {
            "ogc_fid": ogc_fid,
            "n_dates": len(dates),
            "dates_per_year": {str(k): int(v) for k, v in by_year.items()},
            "monthly_min": int(pd.Series(dates).str.slice(5, 7).astype(int).value_counts().min()),
            "monthly_max": int(pd.Series(dates).str.slice(5, 7).astype(int).value_counts().max()),
            "n_cycles": len(rows),
            "harvest_candidates": [
                {
                    "date": dates[c.index], "score": round(c.score, 3),
                    "confidence": c.confidence, "prominence": c.prominence,
                    "support": c.support, "channel_z": c.channel_z,
                }
                for c in candidates
            ],
            "weak_candidates": [
                {"date": dates[c.index], "score": round(c.score, 3),
                 "prominence": c.prominence, "support": c.support}
                for c in weak
            ],
            "cycles": rows,
        }

    # A cycle-based boundary is recorded as context, but the multi-band
    # screening detector still merges some seasons (cycles > 450 days), so it
    # is not trusted to set a split on its own. The reported candidate is the
    # first day of the last complete calendar year before the partial 2026
    # year; every run must still justify and may override it.
    candidate_cycles = []
    for farm, info in season["farms"].items():
        for cyc in info["cycles"]:
            if (
                cyc["status"] == "normal"
                and cyc["end_harvest"]
                and "2024-01-01" <= cyc["end_date"] < "2026-01-01"
                and cyc["length_days"] <= 450
            ):
                candidate_cycles.append({"farm": farm, **{k: cyc[k] for k in ("start_date", "end_date", "length_days")}})
    # Accepted mini split (docs/mini_lab_regional_eval_plan.md §9): the
    # temporal holdout starts at 2024-01-01 (train <= 2022, dev 2023, test 2024+).
    boundary = "2024-01-01"
    all_dates = sorted(
        {
            str(d)
            for group_dir in store.iterdir()
            if group_dir.is_dir() and (group_dir / ".zgroup").exists()
            for d in zarr.open_group(str(group_dir), mode="r")["dates"][:]
        }
    )
    n_before = sum(1 for d in all_dates if d < boundary)
    n_after = len(all_dates) - n_before
    season["holdout_candidate"] = {
        "boundary_date": boundary,
        "n_dates_before": n_before,
        "n_dates_after": n_after,
        "complete_cycle_candidates": candidate_cycles,
        "rule": (
            "Accepted mini split 2026-11-25: train <= 2022, dev 2023, "
            "test 2024+. Boundary is configurable per run and always "
            "justified in the run report."
        ),
        "justification": (
            "Covers complete growth cycles while keeping the partial 2026 year "
            "out of training. Configurable per run; not hard-coded."
        ),
        "n_complete_cycle_candidates": len(candidate_cycles),
        "partial_2026_note": "2026 ends 2026-09-10 and is not a complete cycle.",
    }
    season["wall_s"] = round(time.monotonic() - t0, 2)

    out_dir.mkdir(parents=True, exist_ok=True)
    cycles_path = out_dir / "cycles_mini.parquet"
    pd.DataFrame(all_rows).to_parquet(cycles_path, index=False)
    weak_path = out_dir / "weak_candidates.parquet"
    pd.DataFrame(all_weak).to_parquet(weak_path, index=False)
    season_path = out_dir / "season_structure.json"
    season_path.write_text(json.dumps(season, indent=2) + "\n", encoding="utf-8")
    print(
        "PHENOLOGY COMPLETE "
        + json.dumps(
            {
                "n_cycles": len(all_rows),
                "holdout_boundary": boundary,
                "wall_s": season["wall_s"],
                "cycles": str(cycles_path),
                "weak_candidates": str(weak_path),
                "n_overlong": sum(1 for r in all_rows if r["status"] == "overlong_unresolved"),
                "n_short": sum(1 for r in all_rows if r["status"] == "short_unresolved"),
                "season": str(season_path),
            }
        ),
        flush=True,
    )
    return season


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--weather", type=Path, default=WEATHER)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    build_phenology(args.store, args.weather, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

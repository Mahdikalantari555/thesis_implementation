"""Parity check: raw_satellite_subset.zarr vs the frozen canonical archive.

Three checks (change `farm-subset-anomaly-probe` task 0b.2):

1. **Per-(farm, date, band) stats parity** vs ``stats_long.parquet`` (whose
   ``field_id`` is the ARCHIVE fid = OGC_FID + 1 — alias verified for all 490
   fields, WP-1 E0 audit commit ``9816aaa``; confirmed empirically for the
   subset farms on 2026-11-28). Stats are recomputed from the stored canvas as
   ``data[band][in_field & pixel_valid & finite]`` — exactly the pixels the
   canonical pipeline used. The subset stores float16 (mini-store convention),
   so parity is **tolerance-based**, never exact; tolerances are documented
   below and the empirical deviations are reported in full.
2. **Per-pixel value parity** vs ``pixels/<date>.parquet`` (float32 copies of
   the scene rasters): the sorted value multisets of canvas vs archive must
   agree within the fp16 quantization bound (order-free, so it is insensitive
   to canvas placement).
3. **Membership parity**: canvas in-field∩valid pixel count == archive
   ``n_pixels`` (exact, integer equality).

NaN policy: outside-window pixels are NaN (fill), QA-invalid in-window pixels
keep their values with ``pixel_valid=0`` — no zero-filling anywhere (§8).

Usage::

    python scripts/probe/parity_subset_stats.py --subset .tmp/wp3_smoke/subset.zarr
    python scripts/probe/parity_subset_stats.py --subset D:/RS/Thesis/Data/Processed/raw_satellite_subset.zarr
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.utils.paths import localize  # noqa: E402
from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402

DATA = localize("D:/RS/Thesis/Data")
STATS_LONG = DATA / "Processed" / "satellite" / "stats_long.parquet"
PIXELS_DIR = DATA / "Processed" / "satellite" / "pixels"
BAND_ORDER = ["blue", "green", "red", "nir", "swir1", "swir2", "lst"]
STAT_NAMES = ["mean", "median", "std", "p10", "p25", "p75", "p90", "iqr"]

# Documented tolerances (float16 storage quantization, mini-store convention):
#  - reflectance bands: fp16 spacing ≤ 4.9e-4 below 1.0 (reflectance < 1 here);
#    per-pixel bound 1.0e-3; aggregated stats bound 1.5e-3.
#  - lst (Kelvin): fp16 spacing 0.25 for [256, 512) → per-pixel bound 0.125;
#    aggregated stats bound 0.3 K.
TOL_PERPIXEL_SR = 1.0e-3
TOL_PERPIXEL_LST = 0.125
TOL_STATS_SR = 1.5e-3
TOL_STATS_LST = 0.3


def stats_tol(band: str) -> float:
    return TOL_STATS_LST if band == "lst" else TOL_STATS_SR


def pixel_tol(band: str) -> float:
    return TOL_PERPIXEL_LST if band == "lst" else TOL_PERPIXEL_SR


def compute_stat(vals: np.ndarray, s: str) -> float:
    if s == "mean":
        return float(np.mean(vals))
    if s == "median":
        return float(np.median(vals))
    if s == "std":
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    if s == "iqr":
        return float(np.percentile(vals, 75) - np.percentile(vals, 25))
    return float(np.percentile(vals, {"p10": 10, "p25": 25, "p75": 75, "p90": 90}[s]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subset", type=Path, required=True)
    ap.add_argument("--stats-long", type=Path, default=STATS_LONG)
    ap.add_argument("--pixels-dir", type=Path, default=PIXELS_DIR)
    ap.add_argument("--max-dates", type=int, default=None,
                    help="Check only the first N dates (smoke mode)")
    ap.add_argument("--report", type=Path, default=None,
                    help="Where to write the JSON report (default: <subset>/parity_report.json)")
    args = ap.parse_args(argv)

    t0 = time.monotonic()
    root = zarr.open_group(str(args.subset), mode="r")
    farms = sorted(root.group_keys())
    print(f"Subset: {args.subset} | farms: {len(farms)}")

    archive_fids = {f: int(root[f].attrs["archive_fid"]) for f in farms}
    dates_list = [str(d) for d in root[farms[0]]["dates"][:]]
    if args.max_dates is not None:
        dates_list = dates_list[: args.max_dates]

    sl = pd.read_parquet(
        args.stats_long,
        columns=["field_id", "date", "band", "n_pixels", "mean", "median",
                 "std", "p10", "p25", "p75", "p90", "iqr"])
    sl = sl[sl.field_id.isin(set(archive_fids.values()))]
    sl_idx = sl.set_index(["field_id", "date", "band"]).sort_index()
    print(f"stats_long rows for subset farms: {len(sl)}")

    failures: list[dict] = []
    max_dev = {b: {s: 0.0 for s in STAT_NAMES} for b in BAND_ORDER}
    max_pixel_dev = {b: 0.0 for b in BAND_ORDER}
    n_stat_checks = 0
    n_pixel_checks = 0
    n_membership_checks = 0
    n_dates_checked = 0

    for di, date in enumerate(dates_list):
        px_path = args.pixels_dir / f"{date}.parquet"
        px_cache: dict[int, pd.DataFrame] = {}
        for f in farms:
            g = root[f]
            in_field = g["in_field"][:] > 0
            valid = g["pixel_valid"][di] > 0
            data = g["data"][di]  # (7, S, S) float16
            fid = archive_fids[f]
            sel = in_field & valid
            n_sel = int(sel.sum())
            if (fid, date) not in sl_idx.index:
                per_band = None
            else:
                per_band = sl_idx.loc[(fid, date)]
            if per_band is not None:
                ref_n = per_band["n_pixels"].iloc[0]
                n_membership_checks += 1
                if int(ref_n) != n_sel:
                    failures.append({
                        "check": "membership", "farm": f, "date": date,
                        "canvas_n": n_sel, "archive_n": int(ref_n),
                    })
                for b in BAND_ORDER:
                    if b not in per_band.index:
                        continue
                    row = per_band.loc[b]
                    vals = data[BAND_ORDER.index(b)][sel].astype(np.float32)
                    vals = vals[np.isfinite(vals)]
                    tol = stats_tol(b)
                    for s in STAT_NAMES:
                        ref = float(row[s])
                        if not np.isfinite(ref):
                            continue
                        mine = compute_stat(vals, s)
                        dev = abs(mine - ref)
                        max_dev[b][s] = max(max_dev[b][s], dev)
                        n_stat_checks += 1
                        if dev > tol:
                            failures.append({
                                "check": "stats", "farm": f, "date": date,
                                "band": b, "stat": s, "canvas": mine,
                                "archive": ref, "dev": dev, "tol": tol,
                            })
            # per-pixel multiset parity
            if px_path.exists():
                if fid not in px_cache:
                    pxt = pd.read_parquet(px_path, columns=["field_id", "valid", *BAND_ORDER])
                    px_cache[fid] = pxt[pxt.field_id == fid]
                px = px_cache[fid]
                if len(px):
                    m = px["valid"].to_numpy()
                    for b in BAND_ORDER:
                        pv = px[b].to_numpy(dtype=np.float64)[m]
                        pv = np.sort(pv[np.isfinite(pv)])
                        cv = np.sort(data[BAND_ORDER.index(b)][sel].astype(np.float64))
                        cv = cv[np.isfinite(cv)]
                        if len(pv) != len(cv):
                            failures.append({
                                "check": "pixels_count", "farm": f, "date": date,
                                "band": b, "canvas_n": len(cv), "archive_n": len(pv),
                            })
                            continue
                        dev = float(np.max(np.abs(pv - cv))) if len(pv) else 0.0
                        max_pixel_dev[b] = max(max_pixel_dev[b], dev)
                        n_pixel_checks += len(pv)
                        if dev > pixel_tol(b):
                            failures.append({
                                "check": "pixels", "farm": f, "date": date,
                                "band": b, "max_dev": dev, "tol": pixel_tol(b),
                            })
        n_dates_checked += 1

    wall = time.monotonic() - t0
    verdict = "PASS" if not failures else "FAIL"
    report = {
        "event": "PARITY CHECK COMPLETE",
        "verdict": verdict,
        "subset": str(args.subset),
        "stats_long": str(args.stats_long),
        "dates_checked": n_dates_checked,
        "farms": len(farms),
        "n_membership_checks": n_membership_checks,
        "n_stat_checks": n_stat_checks,
        "n_pixel_checks": n_pixel_checks,
        "max_abs_dev_stats": max_dev,
        "max_abs_dev_pixels": max_pixel_dev,
        "tolerances": {"stats_sr": TOL_STATS_SR, "stats_lst": TOL_STATS_LST,
                       "pixel_sr": TOL_PERPIXEL_SR, "pixel_lst": TOL_PERPIXEL_LST},
        "n_failures": len(failures),
        "failures_first_20": failures[:20],
        "wall_s": round(wall, 1),
        "git_commit_sha": get_git_commit_sha(),
    }
    report_path = args.report or (args.subset / "parity_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps({k: report[k] for k in
                      ("event", "verdict", "dates_checked", "n_stat_checks",
                       "n_pixel_checks", "n_failures", "wall_s")}))
    for b in BAND_ORDER:
        worst = max(max_dev[b].values())
        print(f"  {b:6s} max stats dev {worst:.2e} | max pixel dev {max_pixel_dev[b]:.2e}")
    print(f"Report: {report_path}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

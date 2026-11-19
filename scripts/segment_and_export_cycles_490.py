"""Segment cycles for all 490 fields and export cycles_490.parquet and growth_cycles_490.npz.

Tasks 1.2 and 1.3:
- Reads stats.zarr (490 fields, 538 dates, 7 bands).
- Runs segment_cycles across 2013-2026.
- Exports thesis_implementation/data/processed/cycles_490.parquet.
- Bins multi-band and NDVI cycles into 36 15-day bins (max 540 days).
- Exports thesis_implementation/data/interim/growth_cycles_490.npz.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from ssl_anomaly.io.stats_zarr import load_stats_zarr
from ssl_anomaly.phenology.cycles import (
    FusionParams,
    fusion_params_hash,
    segment_cycles,
    segment_cycles_fused,
)
from ssl_anomaly.phenology.align import default_phase_bins

V1_CYCLES_PATH = "data/processed/cycles_490.parquet"
V1_NPZ_PATH = "data/interim/growth_cycles_490.npz"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("segment_cycles_490")


def run(
    zarr_path: str = "Data/Processed/satellite/stats.zarr",
    cycles_out: str = V1_CYCLES_PATH,
    npz_out: str = V1_NPZ_PATH,
    bin_days: int = 15,
    max_days: int = 540,
    detector: str = "v1",
) -> tuple[pd.DataFrame, dict]:
    if detector not in ("v1", "fused"):
        raise ValueError(f"unknown detector {detector!r}; use 'v1' or 'fused'")
    if detector == "fused" and (
        cycles_out == V1_CYCLES_PATH or npz_out == V1_NPZ_PATH
    ):
        raise ValueError(
            "fused mode refuses the v1 artifact paths (preservation): pass "
            "--cycles-out data/processed/cycles_490_v2.parquet and "
            "--npz-out data/interim/growth_cycles_490_v2.npz"
        )
    log.info("Loading stats.zarr from %s (detector=%s)", zarr_path, detector)
    reader = load_stats_zarr(zarr_path)
    fields = reader.fields
    dates = reader.dates
    bands = reader.bands
    b_idx = reader.band_index

    # Days since start
    times = dates.to_numpy().astype("datetime64[D]")
    epoch = times[0]
    time_days = (times - epoch).astype(int)

    indices = reader.compute_indices("median")
    ndvi = indices["ndvi"]
    savi = indices["savi"]
    lst = indices["lst"]
    valid_frac = reader.valid_frac
    median_cube = reader.get_stat("median")
    swir1 = median_cube[:, :, b_idx["swir1"]]

    fusion = FusionParams()
    log.info("Running harvest segmentation on all %d fields...", len(fields))
    all_cycles = []
    v1_rows: list[dict] = []
    t_seg0 = time.time()
    for fi, fid in enumerate(fields):
        if detector == "v1":
            field_cycles = segment_cycles(
                field_id=int(fid),
                times_days=time_days,
                ndvi=ndvi[fi],
                quality=valid_frac[fi],
                min_cycle_days=180,
                max_cycle_days=max_days,
                min_prominence=0.15,
                plateau_frac=0.7,
                rise_frac=0.15,
            )
        else:
            v1_field = segment_cycles(
                field_id=int(fid),
                times_days=time_days,
                ndvi=ndvi[fi],
                quality=valid_frac[fi],
                min_cycle_days=180,
                max_cycle_days=max_days,
                min_prominence=0.15,
                plateau_frac=0.7,
                rise_frac=0.15,
            )
            field_cycles = segment_cycles_fused(
                field_id=int(fid),
                times_days=time_days,
                ndvi=ndvi[fi],
                savi=savi[fi],
                lst=lst[fi],
                swir1=swir1[fi],
                quality=valid_frac[fi],
                fusion=fusion,
                min_cycle_days=180,
                max_cycle_days=max_days,
                min_prominence=0.15,
                plateau_frac=0.7,
                rise_frac=0.15,
            )
            # v1 ⊆ v2 per field (window keys), the D1 subset property.
            v1_keys = {(c.t_start, c.t_end) for c in v1_field}
            v2_keys = {(c.t_start, c.t_end) for c in field_cycles}
            missing = v1_keys - v2_keys
            if missing:
                raise RuntimeError(
                    f"field {fid}: v1 windows missing from fused output "
                    f"{sorted(missing)[:3]} — v1⊆v2 violated"
                )
            for c in v1_field:
                v1_rows.append(c.to_dict())
        for c in field_cycles:
            all_cycles.append(c.to_dict())
    log.info("segmentation wall %.1f s", time.time() - t_seg0)

    cycles_df = pd.DataFrame(all_cycles)
    cycles_path = ROOT_DIR / cycles_out
    cycles_path.parent.mkdir(parents=True, exist_ok=True)
    cycles_df.to_parquet(cycles_path, index=False)
    log.info(
        "Saved %d cycles (%d non-censored) across %d fields to %s",
        len(cycles_df),
        (~cycles_df["censored"]).sum(),
        cycles_df["field"].nunique(),
        cycles_path,
    )

    if detector == "fused":
        v1_df = pd.DataFrame(v1_rows)
        v2_clean = cycles_df[~cycles_df["censored"].astype(bool)]
        v1_clean = v1_df[~v1_df["censored"].astype(bool)]
        v2_len = v2_clean["t_end"] - v2_clean["t_start"]
        v1_len = v1_clean["t_end"] - v1_clean["t_start"]
        comparison = {
            "v1_total": int(len(v1_df)),
            "v1_clean": int(len(v1_clean)),
            "v1_clean_len_median": float(v1_len.median()),
            "v2_total": int(len(cycles_df)),
            "v2_clean": int(len(v2_clean)),
            "v2_clean_len_median": float(v2_len.median()),
            "v2_fields_with_cycles": int(cycles_df["field"].nunique()),
            "v1_fields_with_cycles": int(v1_df["field"].nunique()),
        }
        log.info("v1<->v2 comparison: %s", json.dumps(comparison))
        provenance = {
            "detector": detector,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "git_sha": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True,
                capture_output=True,
            ).stdout.strip(),
            "fusion_params_hash": fusion_params_hash(fusion),
            "fusion_params": {k: getattr(fusion, k) for k in (
                "veg_drop", "lst_rise", "swir_rise", "ctx_days", "min_gap",
                "vf_min", "edge_guard", "dedup_days")},
            "v1_params": {"min_cycle_days": 180, "max_cycle_days": max_days,
                          "min_prominence": 0.15, "plateau_frac": 0.7,
                          "rise_frac": 0.15},
            "stats_zarr": zarr_path,
            "supersedes": "cycles_490.parquet (v1, kept frozen in place)",
            "comparison": comparison,
        }
        prov_path = cycles_path.with_name(cycles_path.stem + ".provenance.json")
        prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        log.info("Wrote provenance sidecar %s", prov_path)

    # Now bin cycles for growth_cycles_490.npz
    # Phase bins: 15-day bins up to 540 days -> 36 bins
    n_bins = int(np.ceil(max_days / bin_days))
    phase_bins = default_phase_bins(bin_days=bin_days, max_cycle_days=max_days)

    # Band channels: Band 0 is NDVI, Bands 1..7 are blue, green, red, nir, swir1, swir2, lst
    # Total 8 channels for full multi-spectral representation
    channel_names = ["ndvi"] + bands

    rows = []
    starts = []
    ends = []
    fids = []
    bin_dates = []

    for _, row in cycles_df.iterrows():
        if bool(row["censored"]):
            continue

        fid = int(row["field"])
        fi = int(np.where(fields == fid)[0][0])

        start_day = int(row["t_start"])
        end_day = int(row["t_end"])
        harvest_day = int(row.get("start_harvest", row["t_start"]))

        start_dt = times[0] + np.timedelta64(start_day, "D")
        end_dt = times[0] + np.timedelta64(end_day, "D")
        harvest_dt = times[0] + np.timedelta64(harvest_day, "D")

        if end_dt <= start_dt:
            continue

        # Find observations inside cycle
        obs_mask = (times >= start_dt) & (times < end_dt)
        if obs_mask.sum() < 3:
            continue

        obs_times = times[obs_mask]
        dsh = (obs_times - harvest_dt).astype(int)

        valid_obs = (dsh >= 0) & (dsh < max_days)
        if valid_obs.sum() < 3:
            continue

        dsh_valid = dsh[valid_obs]
        ndvi_obs = ndvi[fi, obs_mask][valid_obs]
        cube_obs = median_cube[fi, obs_mask][valid_obs]  # (n_obs, 7)

        # Build (36, 8) binned array
        x_cycle = np.full((n_bins, len(channel_names)), np.nan, dtype=np.float32)

        assigned_bins = dsh_valid // bin_days
        for b in np.unique(assigned_bins):
            if b >= n_bins:
                continue
            b_mask = assigned_bins == b
            # Channel 0: NDVI
            x_cycle[int(b), 0] = np.nanmedian(ndvi_obs[b_mask])
            # Channels 1..7: raw spectral bands
            for ci, bname in enumerate(bands):
                x_cycle[int(b), ci + 1] = np.nanmedian(cube_obs[b_mask, b_idx[bname]])

        # Interpolate single missing gaps within observed range
        observed_bin_indices = np.where(np.isfinite(x_cycle[:, 0]))[0]
        if len(observed_bin_indices) < 3:
            continue

        first_b, last_b = observed_bin_indices[0], observed_bin_indices[-1]
        for ch in range(len(channel_names)):
            series = x_cycle[:, ch]
            # Linearly interpolate interior isolated gaps
            for b in range(first_b + 1, last_b):
                if np.isnan(series[b]):
                    prev_val = series[b - 1]
                    # Look ahead for next valid
                    next_idx = None
                    for look in range(b + 1, min(b + 3, last_b + 1)):
                        if np.isfinite(series[look]):
                            next_idx = look
                            break
                    if next_idx is not None and np.isfinite(prev_val):
                        # Simple linear interpolation
                        series[b] = prev_val + (series[next_idx] - prev_val) * (1 / (next_idx - b + 1))
            x_cycle[:, ch] = series

        dates_for_cycle = harvest_dt + np.arange(n_bins).astype("timedelta64[D]") * bin_days

        rows.append(x_cycle)
        starts.append(str(start_dt))
        ends.append(str(end_dt))
        fids.append(str(fid))
        bin_dates.append(dates_for_cycle)

    X = np.stack(rows)  # (N, 36, 8)
    npz_path = ROOT_DIR / npz_out
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        X=X,
        start_dates=np.asarray(starts),
        end_dates=np.asarray(ends),
        field_ids=np.asarray(fids),
        bin_dates=np.stack(bin_dates).astype("datetime64[D]"),
        bin_days=np.array(bin_days),
        channel_names=np.asarray(channel_names),
    )
    log.info(
        "Saved %d binned cycles with shape %s to %s",
        len(rows),
        X.shape,
        npz_path,
    )

    return cycles_df, {"X": X, "n_cycles": len(rows), "npz_path": str(npz_path)}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zarr", default="Data/Processed/satellite/stats.zarr")
    p.add_argument("--cycles-out", default=V1_CYCLES_PATH)
    p.add_argument("--npz-out", default=V1_NPZ_PATH)
    p.add_argument("--detector", choices=["v1", "fused"], default="v1")
    p.add_argument("--bin-days", type=int, default=15)
    p.add_argument("--max-days", type=int, default=540)
    args = p.parse_args()
    run(zarr_path=args.zarr, cycles_out=args.cycles_out, npz_out=args.npz_out,
        bin_days=args.bin_days, max_days=args.max_days, detector=args.detector)

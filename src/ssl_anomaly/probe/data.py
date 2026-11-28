"""Data loading for the 32-farm probe: subset store → field-date features.

Loads ``raw_satellite_subset.zarr`` (built by
``scripts/build_raw_satellite_subset.py``) into field-date matrices:

- ``band_means``  (F, D, 7) float32 — field-mean reflectance (6 bands) + LST K,
  NaN when the date has no valid in-field pixel for that band. **NaN is never
  filled with 0** (§8); observation masks carry absence explicitly.
- derived channels (NDVI, SAVI) computed from field-mean red/nir — documented
  approximation, consistent across all methods on this probe.
- cycle progress ``tau`` and cumulative GDD from the v2 cycle table
  (``data/processed/cycles_490_v2.parquet``, epoch 2013-03-30, archive fid =
  OGC_FID + 1) and the district daily-temperature series.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date as _date
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from ..utils.paths import localize

DATA = localize("D:/RS/Thesis/Data")
EPOCH = _date(2013, 3, 30)  # canonical scene epoch (WP-1 E0 audit)
GDD_T_BASE_C = 10.0         # src/ssl_anomaly/phenology/align.py convention
GDD_T_CAP_C = 35.0
BAND_ORDER = ["blue", "green", "red", "nir", "swir1", "swir2", "lst"]


@dataclass
class ProbeData:
    farm_ids: list[str]
    archive_fids: list[int]
    dates: list[str]
    band_means: np.ndarray      # (F, D, 7) float32, NaN = absent
    obs_mask: np.ndarray        # (F, D) bool — ≥1 valid in-field pixel
    n_px: np.ndarray            # (F, D) int32
    tau: np.ndarray             # (F, D) float32 cycle progress [0,1]; NaN outside cycles
    cycle_id: np.ndarray        # (F, D) int32, −1 = no cycle
    gdd_cum: np.ndarray         # (F, D) float32 cumulative GDD within cycle
    train_mask: np.ndarray      # (D,) bool — train dates (all fits are train-only)
    provenance: dict = field(default_factory=dict)
    canvases: dict = field(default_factory=dict)  # farm_id → (in_field, data, pixel_valid)

    @property
    def ndvi(self) -> np.ndarray:
        red = self.band_means[..., 2]
        nir = self.band_means[..., 3]
        denom = nir + red
        out = np.full_like(red, np.nan, dtype=np.float32)
        ok = np.isfinite(denom) & (denom != 0)
        out[ok] = ((nir[ok] - red[ok]) / denom[ok]).astype(np.float32)
        return out

    @property
    def savi(self) -> np.ndarray:
        red = self.band_means[..., 2]
        nir = self.band_means[..., 3]
        denom = nir + red + 0.5
        out = np.full_like(red, np.nan, dtype=np.float32)
        ok = np.isfinite(denom) & (denom != 0)
        out[ok] = (1.5 * (nir[ok] - red[ok]) / denom[ok]).astype(np.float32)
        return out

    @property
    def lst_c(self) -> np.ndarray:
        return self.band_means[..., 6] - 273.15


def load_subset_features(store: Path) -> tuple[ProbeData, dict]:
    """Read the subset zarr into field-date matrices (no zero-filling)."""
    store = Path(store)
    root = zarr.open_group(str(store), mode="r")
    farm_ids = sorted(root.group_keys())
    dates = [str(d) for d in root[farm_ids[0]]["dates"][:]]
    F, D = len(farm_ids), len(dates)
    S = int(root[farm_ids[0]].attrs["canvas"])
    K = len(BAND_ORDER)

    band_means = np.full((F, D, K), np.nan, dtype=np.float32)
    obs_mask = np.zeros((F, D), dtype=bool)
    n_px = np.zeros((F, D), dtype=np.int32)
    canvases_ok = {}
    for fi, f in enumerate(farm_ids):
        g = root[f]
        in_field = g["in_field"][:] > 0
        pixel_valid = g["pixel_valid"][:] > 0        # (D, S, S)
        data = g["data"][:]                          # (D, K, S, S) fp16
        canvases_ok[f] = (in_field, data, pixel_valid)
        sel = in_field[None] & pixel_valid           # (D, S, S)
        n_px[fi] = sel.sum(axis=(1, 2))
        obs_mask[fi] = n_px[fi] > 0
        for k in range(K):
            vals = np.where(sel, data[:, k], np.nan).astype(np.float32)
            with np.errstate(all="ignore"):
                m = np.nanmean(vals, axis=(1, 2))
            m[n_px[fi] == 0] = np.nan
            band_means[fi, :, k] = m

    provenance = {
        "subset_store": str(store),
        "store_git_sha": root.attrs.get("git_commit_sha"),
        "store_date_range": list(root.attrs.get("date_range", [])),
        "band_order": BAND_ORDER,
        "canvas": S,
        "farms": F,
        "dates": D,
    }
    pd_ = ProbeData(
        farm_ids=farm_ids,
        archive_fids=[int(root[f].attrs["archive_fid"]) for f in farm_ids],
        dates=dates,
        band_means=band_means,
        obs_mask=obs_mask,
        n_px=n_px,
        tau=None, cycle_id=None, gdd_cum=None,   # filled by attach_cycles_weather
        train_mask=np.zeros(D, dtype=bool),
        provenance=provenance,
    )
    pd_.canvases = canvases_ok  # (in_field, data, pixel_valid) per farm — Stage-2
    return pd_, canvases_ok


def attach_cycles_weather(
    d: ProbeData,
    cycles_path: Path | None = None,
    weather_csv: Path | None = None,
    train_years: int = 3,
) -> ProbeData:
    """Attach cycle progress / GDD and define the train date split.

    Train split = the first ``train_years`` calendar years of the archive
    (default 3: 2013-03-30 → 2016-03-29) — the E1 "3 train years" rule. All
    model fits, normalizations and σ calibrations use train dates ONLY.
    """
    cycles_path = cycles_path or localize("D:/RS/Thesis/thesis_implementation") / "data/processed/cycles_490_v2.parquet"
    weather_csv = weather_csv or DATA / "Weather" / "roi_corners_daily_temperature.csv"

    F, D = len(d.farm_ids), len(d.dates)
    day_index = np.array([(_date.fromisoformat(s) - EPOCH).days for s in d.dates], dtype=np.int64)

    cyc = pd.read_parquet(cycles_path)
    fid_to_row = {fid: i for i, fid in enumerate(d.archive_fids)}
    tau = np.full((F, D), np.nan, dtype=np.float32)
    cycle_id = np.full((F, D), -1, dtype=np.int32)

    # district daily temperature = mean across corners (one district series)
    w = pd.read_csv(weather_csv)
    w["date"] = pd.to_datetime(w["date"]).dt.date
    tmean = w.groupby("date")["t_mean"].mean()
    gdd_daily = np.clip(tmean - GDD_T_BASE_C, 0.0, GDD_T_CAP_C - GDD_T_BASE_C)
    gdd_daily = gdd_daily.dropna()
    gdd_by_day = {(_date.fromisoformat(str(k)) - EPOCH).days: float(v)
                  for k, v in gdd_daily.items()}

    for fid, row_i in fid_to_row.items():
        fc = cyc[cyc.field == fid].sort_values("cycle_id")
        for _, r in fc.iterrows():
            t0, t1 = int(r.t_start), int(r.t_end)
            length = max(t1 - t0, 1)
            in_cyc = (day_index >= t0) & (day_index < t1)
            tau[row_i, in_cyc] = ((day_index[in_cyc] - t0) / length).astype(np.float32)
            cycle_id[row_i, in_cyc] = int(r.cycle_id)

    # cumulative GDD within each cycle (NaN weather days contribute 0 GDD,
    # per align.compute_gdd convention — recorded in provenance)
    gdd_cum = np.full((F, D), np.nan, dtype=np.float32)
    days_sorted = sorted(gdd_by_day)
    if days_sorted:
        g0, g1 = days_sorted[0], days_sorted[-1]
        grid = np.arange(g0, g1 + 1)
        vals = np.array([gdd_by_day.get(int(day), 0.0) for day in grid])
        cum = np.concatenate([[0.0], np.cumsum(vals)])
        for fid, row_i in fid_to_row.items():
            fc = cyc[cyc.field == fid].sort_values("cycle_id")
            for _, r in fc.iterrows():
                t0, t1 = int(r.t_start), int(r.t_end)
                in_cyc = (day_index >= t0) & (day_index < t1)
                dd = day_index[in_cyc]
                ok = (dd >= g0) & (dd <= g1)
                gdd_cum[row_i, np.where(in_cyc)[0][ok]] = cum[dd[ok] - g0].astype(np.float32)

    years = sorted({_date.fromisoformat(s).year for s in d.dates})
    train_years_list = set(years[:train_years])
    train_mask = np.array([_date.fromisoformat(s).year in train_years_list for s in d.dates])

    d.tau = tau
    d.cycle_id = cycle_id
    d.gdd_cum = gdd_cum
    d.train_mask = train_mask
    d.provenance.update({
        "cycles_path": str(cycles_path),
        "weather_csv": str(weather_csv),
        "gdd_t_base_c": GDD_T_BASE_C,
        "gdd_t_cap_c": GDD_T_CAP_C,
        "train_years": train_years,
        "train_year_list": sorted(train_years_list),
        "n_train_dates": int(train_mask.sum()),
        "n_eval_dates": int((~train_mask).sum()),
        "gdd_nan_days_contribute_zero": "align.compute_gdd convention",
    })
    return d

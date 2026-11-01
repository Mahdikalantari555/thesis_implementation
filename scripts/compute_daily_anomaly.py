"""Compute per-field, per-date anomaly score (S = RMS z) from baseline stats."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

OUT = "data/processed/daily_anomaly.parquet"

from ssl_anomaly.phenology import add_spectral_indices
ds = xr.open_zarr("data/processed/fields.zarr")
ds = add_spectral_indices(ds)
times = pd.to_datetime(ds.coords["time"].values)
fields = ds.coords["field"].values
bands = list(ds.coords["band"].values)
med = ds["median"].values  # (F, T, B)
band_idx = {b: i for i, b in enumerate(bands)}

bl = np.load("data/processed/baseline.npz", allow_pickle=True)
bin_centers = bl["bin_centers"]
strata_keys = [str(s) for s in bl["strata_keys"]]
floor = float(bl["floor"])

cycles = pd.read_parquet("data/processed/cycles.parquet")
epoch = pd.to_datetime(times[0])

rows = []
for _, c in cycles.iterrows():
    if c["censored"]:
        continue
    fid = int(c["field"])
    fi = int(np.where(fields == fid)[0][0]) if fid in fields else None
    if fi is None:
        continue
    stratum = str(c.get("kind", "all"))
    ratoon = int(c.get("ratoon_no", 0))
    key = f"{stratum}_{ratoon}" if stratum == "plant" else f"ratoon_{ratoon}"
    if key not in strata_keys:
        key = "all"
    t_start_dt = epoch + pd.Timedelta(days=int(c["t_start"]))
    t_end_dt = epoch + pd.Timedelta(days=int(c["t_end"]))
    start_h = int(c.get("start_harvest", c["t_start"]))
    harvest_dt = epoch + pd.Timedelta(days=start_h)
    mask = (times >= t_start_dt) & (times < t_end_dt)
    ti_list = np.where(mask)[0]
    # baseline arrays for this stratum
    med_arr = bl[f"{key}__median"]  # (36, B)
    scale_arr = bl[f"{key}__scale"]  # (36, B)
    for ti in ti_list:
        t = times[ti]
        dsh = int((t - harvest_dt).days)
        bin_idx = dsh // 15
        if bin_idx < 0 or bin_idx >= len(bin_centers):
            continue
        # if bin center too far? use bin_idx clamped
        bin_idx = min(bin_idx, len(bin_centers) - 1)
        m = med_arr[bin_idx]
        s = np.maximum(scale_arr[bin_idx], floor)
        obs = med[fi, ti, :]  # band values
        z = (obs - m) / s
        # RMS over finite bands
        finite = np.isfinite(z) & np.isfinite(obs) & np.isfinite(m)
        if finite.sum() == 0:
            S = np.nan
        else:
            S = float(np.sqrt(np.mean(z[finite] ** 2)))
        rows.append({"field": fid, "date": t.strftime("%Y-%m-%d"), "anomaly": S})

df = pd.DataFrame(rows)
# if multiple cycles overlap on same date, keep max
df = df.groupby(["field", "date"])["anomaly"].max().reset_index()
df.to_parquet(OUT, index=False)
print(f"Wrote {len(df)} daily anomaly records to {OUT}")
print(df.head())

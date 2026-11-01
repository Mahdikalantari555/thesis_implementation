"""Interactive time-slider map: anomaly burden per field per date."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import folium
from folium.plugins import TimeSliderChoropleth

OUT = Path("reports/figures/anomaly_timeslider.html")

scores = pd.read_parquet("data/processed/scores.parquet")
cycles = pd.read_parquet("data/processed/cycles.parquet")
ds = xr.open_zarr("data/processed/fields.zarr")
times = pd.to_datetime(ds.coords["time"].values)

# Map each cycle to its burden. scores rows align with X_by_stratum order;
# we approximate by assigning cycle burden from scores by stratum+index.
# Simpler: assign per-field mean burden to all its cycles.
field_burden = scores.groupby("field")["burden"].mean().to_dict() if "field" in scores.columns else {}
# If no field in scores, fall back to stratum-level burden broadcast
if not field_burden:
    sb = scores.groupby("stratum")["burden"].mean().to_dict()
    # map each cycle's stratum to burden
    cycles["_burden"] = cycles.apply(lambda r: sb.get(str(r.get("kind", "all")), 0.0), axis=1)
else:
    cycles["_burden"] = cycles["field"].map(lambda f: field_burden.get(f, 0.0))

# Build per-field, per-date burden (cycle-level broadcast)
records = []
for _, c in cycles.iterrows():
    if c["censored"]:
        continue
    fid = int(c["field"])
    t0 = pd.to_datetime(str(c["t_start"])[:10] if isinstance(c["t_start"], str) else None)
    # t_start is day index since epoch (times[0]); convert
    epoch = pd.to_datetime(ds.coords["time"].values[0])
    t_start_dt = epoch + pd.Timedelta(days=int(c["t_start"]))
    t_end_dt = epoch + pd.Timedelta(days=int(c["t_end"]))
    mask = (times >= t_start_dt) & (times < t_end_dt)
    for d in times[mask]:
        records.append({"field": fid, "date": d.strftime("%Y-%m-%d"), "burden": float(c["_burden"])})

rdf = pd.DataFrame(records)
if rdf.empty:
    print("No records")
    raise SystemExit(1)

# Build styledict for TimeSliderChoropleth: {str(field_id): {date: {"color": ..., "opacity": 0.7}}}
def color_for(v):
    # simple red scale
    if v < 0.05:
        return "#22c55e"
    if v < 0.15:
        return "#eab308"
    return "#dc2626"

styledict = {}
for fid, sub in rdf.groupby("field"):
    styledict[str(fid)] = {}
    for _, r in sub.iterrows():
        ts = pd.Timestamp(r["date"]).timestamp()
        styledict[str(fid)][str(int(ts))] = {
            "color": color_for(r["burden"]),
            "opacity": 0.7,
        }

gdf = gpd.read_file("D:/RS/Thesis/Data/RoI/Farms_subset.geojson").reset_index(drop=True)
gdf["field"] = gdf.index + 1
geojson = json.loads(gdf.to_json())

# TimeSliderChoropleth needs 'id' property; add field as id
for feat in geojson["features"]:
    feat["id"] = str(feat["properties"]["field"])

m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=12, tiles=None)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery",
    name="Satellite",
    overlay=False,
    control=True,
).add_to(m)

TimeSliderChoropleth(
    data=geojson,
    styledict=styledict,
    name="Anomaly over time",
).add_to(m)

folium.LayerControl().add_to(m)
OUT.parent.mkdir(parents=True, exist_ok=True)
m.save(str(OUT))
print("Time-slider map saved to", OUT)

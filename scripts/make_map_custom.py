"""Custom interactive map: time slider + variable selector + legend."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

OUT = Path("reports/figures/anomaly_map_custom.html")

ds = xr.open_zarr("data/processed/fields.zarr")
times = pd.to_datetime(ds.coords["time"].values)
fields = ds.coords["field"].values
ndvi_all = ds["ndvi"].values  # (F, T)
lst_all = ds["median"].sel(band="ST_B10").values if "ST_B10" in list(ds.coords["band"].values) else ds["median"].values[:, :, list(ds.coords["band"].values).index("ST_B10")]
cwsi_all = ds["cwsi_et"].values  # (F, T)

# Load daily anomaly scores
daily = pd.read_parquet("data/processed/daily_anomaly.parquet")

# Build anomaly per field per date
anomaly_map = {}  # field -> {date_str: value}
for fi, fid in enumerate(fields):
    fid = int(fid)
    anomaly_map[fid] = {}
    for ti, t in enumerate(times):
        anomaly_map[fid][t.strftime("%Y-%m-%d")] = 0.0

for _, r in daily.iterrows():
    fid = int(r["field"])
    d = r["date"]
    val = r["anomaly"]
    if np.isnan(val):
        val = 0.0
    if fid in anomaly_map and d in anomaly_map[fid]:
        anomaly_map[fid][d] = max(anomaly_map[fid][d], float(val))

# Build per-field per-date data dict
data = {}
for fi, fid in enumerate(fields):
    fid = int(fid)
    data[str(fid)] = {}
    for ti, t in enumerate(times):
        d = t.strftime("%Y-%m-%d")
        ts = int(t.timestamp())
        data[str(fid)][str(ts)] = {
            "ndvi": float(ndvi_all[fi, ti]) if np.isfinite(ndvi_all[fi, ti]) else None,
            "lst": float(lst_all[fi, ti]) if np.isfinite(lst_all[fi, ti]) else None,
            "cwsi": float(cwsi_all[fi, ti]) if np.isfinite(cwsi_all[fi, ti]) else None,
            "anomaly": float(anomaly_map[fid].get(d, 0.0)),
        }

# GeoJSON
gdf = gpd.read_file("D:/RS/Thesis/Data/RoI/Farms_subset.geojson").reset_index(drop=True)
gdf["field"] = gdf.index + 1
geojson = json.loads(gdf.to_json())
for feat in geojson["features"]:
    feat["id"] = str(feat["properties"]["field"])

# Dates list (unique sorted)
dates = sorted(set(d.strftime("%Y-%m-%d") for d in times))
timestamps = [int(pd.Timestamp(d).timestamp()) for d in dates]

# Build HTML
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Anomaly Time Slider</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  body {{ margin:0; font-family: sans-serif; }}
  #map {{ height: 80vh; }}
  #controls {{ padding: 10px; background: #f0f0f0; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }}
  #legend {{ padding: 10px; background: #fff; border: 1px solid #ccc; margin: 10px; display: inline-block; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; margin: 2px 0; }}
  .legend-color {{ width: 20px; height: 14px; border: 1px solid #333; }}
  #slider-container {{ flex: 1; min-width: 300px; }}
  #date-label {{ font-weight: bold; min-width: 120px; }}
</style>
</head>
<body>
<div id="controls">
  <label>Variable:
    <select id="var-select">
      <option value="anomaly">Anomaly burden</option>
      <option value="ndvi">NDVI</option>
      <option value="lst">LST (K)</option>
      <option value="cwsi">CWSI-ET</option>
    </select>
  </label>
  <div id="slider-container">
    <input type="range" id="time-slider" min="0" max="{len(timestamps)-1}" value="0" style="width:100%">
  </div>
  <span id="date-label">{dates[0]}</span>
</div>
<div id="legend"></div>
<div id="map"></div>
<script>
var data = {json.dumps(data)};
var geojson = {json.dumps(geojson)};
var timestamps = {json.dumps(timestamps)};
var dates = {json.dumps(dates)};

var map = L.map('map').setView([{gdf.geometry.centroid.y.mean()}, {gdf.geometry.centroid.x.mean()}], 12);
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
  attribution: 'Esri World Imagery',
  maxZoom: 18
}}).addTo(map);

var geoLayer = L.geoJSON(geojson, {{
  style: function(feature) {{
    return {{ fillColor: '#888', weight: 1, color: '#333', fillOpacity: 0.6 }};
  }},
  onEachFeature: function(feature, layer) {{
    layer.bindPopup('Field ' + feature.properties.field);
  }}
}}).addTo(map);

function colorForAnomaly(v) {{
  if (v === null || v === undefined) return '#888';
  if (v < 5) return '#22c55e';
  if (v < 10) return '#eab308';
  return '#dc2626';
}}
function colorForNDVI(v) {{
  if (v === null || v === undefined) return '#888';
  // brown (0) -> green (0.8)
  var t = Math.min(Math.max(v, 0), 0.8) / 0.8;
  var r = Math.round(139 * (1-t) + 34 * t);
  var g = Math.round(90 * (1-t) + 139 * t);
  var b = Math.round(43 * (1-t) + 34 * t);
  return 'rgb('+r+','+g+','+b+')';
}}
function colorForLST(v) {{
  if (v === null || v === undefined) return '#888';
  // blue (300) -> red (330)
  var t = Math.min(Math.max((v-300)/30, 0), 1);
  var r = Math.round(0 + 220*t);
  var g = Math.round(100*(1-t));
  var b = Math.round(220*(1-t));
  return 'rgb('+r+','+g+','+b+')';
}}
function colorForCWSI(v) {{
  if (v === null || v === undefined) return '#888';
  var t = Math.min(Math.max(v, 0), 1);
  var r = Math.round(34 + 200*t);
  var g = Math.round(139*(1-t));
  var b = Math.round(34*(1-t));
  return 'rgb('+r+','+g+','+b+')';
}}

function getColor(varName, v) {{
  if (varName === 'anomaly') return colorForAnomaly(v);
  if (varName === 'ndvi') return colorForNDVI(v);
  if (varName === 'lst') return colorForLST(v);
  if (varName === 'cwsi') return colorForCWSI(v);
  return '#888';
}}

function getValue(varName, d) {{
  if (varName === 'anomaly') return d.anomaly;
  if (varName === 'ndvi') return d.ndvi;
  if (varName === 'lst') return d.lst;
  if (varName === 'cwsi') return d.cwsi;
  return null;
}}

var currentVar = 'anomaly';
var currentIdx = 0;

function updateMap() {{
  var ts = timestamps[currentIdx];
  var dateStr = dates[currentIdx];
  document.getElementById('date-label').textContent = dateStr;
  geoLayer.eachLayer(function(layer) {{
    var fid = String(layer.feature.properties.field);
    var d = data[fid] ? data[fid][String(ts)] : null;
    var v = d ? getValue(currentVar, d) : null;
    layer.setStyle({{ fillColor: getColor(currentVar, v), fillOpacity: 0.7 }});
  }});
  updateLegend();
}}

function updateLegend() {{
  var leg = document.getElementById('legend');
  var html = '<b>Legend: ' + currentVar + '</b><br>';
  if (currentVar === 'anomaly') {{
    html += '<div class="legend-item"><div class="legend-color" style="background:#22c55e"></div> < 5 (normal)</div>';
    html += '<div class="legend-item"><div class="legend-color" style="background:#eab308"></div> 5-10 (moderate)</div>';
    html += '<div class="legend-item"><div class="legend-color" style="background:#dc2626"></div> > 10 (anomaly)</div>';
  }} else if (currentVar === 'ndvi') {{
    html += '<div class="legend-item"><div class="legend-color" style="background:rgb(139,90,43)"></div> 0.0 (low)</div>';
    html += '<div class="legend-item"><div class="legend-color" style="background:rgb(34,139,34)"></div> 0.8 (high)</div>';
  }} else if (currentVar === 'lst') {{
    html += '<div class="legend-item"><div class="legend-color" style="background:rgb(0,100,220)"></div> 300 K (cool)</div>';
    html += '<div class="legend-item"><div class="legend-color" style="background:rgb(220,0,0)"></div> 330 K (hot)</div>';
  }} else if (currentVar === 'cwsi') {{
    html += '<div class="legend-item"><div class="legend-color" style="background:rgb(34,139,34)"></div> 0 (no stress)</div>';
    html += '<div class="legend-item"><div class="legend-color" style="background:rgb(234,0,0)"></div> 1 (high stress)</div>';
  }}
  leg.innerHTML = html;
}}

document.getElementById('time-slider').addEventListener('input', function() {{
  currentIdx = parseInt(this.value);
  updateMap();
}});
document.getElementById('var-select').addEventListener('change', function() {{
  currentVar = this.value;
  updateMap();
}});

updateMap();
</script>
</body>
</html>
"""

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(html, encoding="utf-8")
print("Custom map saved to", OUT)

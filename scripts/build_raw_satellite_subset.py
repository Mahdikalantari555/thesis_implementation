"""Extract date-centric raw satellite store for a farm subset.

Builds ``raw_satellite_subset.zarr`` with per-farm arrays of shape
``(D=538, K=7, H_i, W_i)`` in fp16 — **variable spatial size** (farm bounding
box, no 48×48 padding). NDVI/SAVI are NOT stored (computed on-demand).

Usage:
    python scripts/build_raw_satellite_subset.py --smoke 3
    python scripts/build_raw_satellite_subset.py --resume
    python scripts/build_raw_satellite_subset.py --farms Data/RoI/Farms_subset.geojson
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import zarr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.io.landsat import apply_scale_sr, apply_scale_st, quality_mask, QualityCfg

FILE_MAP = OrderedDict([
    ("blue", "blue.tif"), ("green", "green.tif"), ("red", "red.tif"),
    ("nir", "nir08.tif"), ("swir1", "swir16.tif"), ("swir2", "swir22.tif"),
    ("lst", "lwir11.tif"),
])

BAND_ORDER = ["blue", "green", "red", "nir", "swir1", "swir2", "lst"]
SCENES_ROOT = Path("D:/RS/Thesis/Data/Metric/scenes")
STATS_ZARR = Path("D:/RS/Thesis/Data/Processed/satellite/stats.zarr")


def load_farm_polygons(geojson_path: Path):
    """Load farm polygons and compute bounding boxes in pixel coords."""
    import geopandas as gpd
    gdf = gpd.read_file(geojson_path)
    # Reproject to EPSG:32639
    gdf = gdf.to_crs("EPSG:32639")
    farms = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        minx, miny, maxx, maxy = geom.bounds
        farms.append({
            "farm_id": row["Farm"],
            "geometry": geom,
            "bbox": (minx, miny, maxx, maxy),
        })
    return farms, gdf


def find_scene_for_date(date_str: str, scenes_root: Path):
    """Find scene directory for a given date string (YYYY-MM-DD)."""
    # Convert date to YYYYMMDD
    dt = datetime.strptime(date_str, "%Y-%m-%d") if "-" in date_str else None
    if dt is None:
        # Try YYYYMMDD
        dt = datetime.strptime(date_str, "%Y%m%d")
    ymd = dt.strftime("%Y%m%d")
    # Search for matching scene dirs
    candidates = list(scenes_root.glob(f"landsat_{ymd}_*"))
    return candidates


def read_band_window(scene_dir: Path, band_file: str, bbox, transform):
    """Read a band window for a farm bbox from a scene GeoTIFF."""
    tif_path = scene_dir / band_file
    if not tif_path.exists():
        return None
    with rasterio.open(tif_path) as src:
        # Compute window from bbox
        from rasterio.windows import from_bounds
        window = from_bounds(*bbox, transform=src.transform)
        # Round to integer
        window = window.round_offsets().round_shape()
        data = src.read(1, window=window)
        return data, window


def build_subset(farms, dates, out_path, max_dates=None, resume=False):
    """Build raw_satellite_subset.zarr for the given farms and dates."""
    out_path = Path(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Manifest for resume
    manifest_path = out_path / ".manifest.jsonl"
    done_farms = set()
    if resume and manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                done_farms.add(rec["farm_id"])

    # Date list
    if max_dates:
        dates = dates[:max_dates]

    # Quality config
    qcfg = QualityCfg()

    for farm in farms:
        farm_id = farm["farm_id"]
        if farm_id in done_farms:
            print(f"  Skipping {farm_id} (already done)")
            continue

        print(f"Processing farm {farm_id}...")
        geom = farm["geometry"]
        bbox = farm["bbox"]

        # Create zarr group for this farm
        farm_group = zarr.open_group(str(out_path / farm_id), mode="w")

        # Pre-allocate arrays: (D, K, H, W) — but H, W vary per farm
        # We need to determine H, W from the first valid scene
        # For now, compute from bbox and 30m resolution
        minx, miny, maxx, maxy = bbox
        # Approximate pixel dimensions (30m resolution)
        H = int((maxy - miny) / 30) + 2
        W = int((maxx - minx) / 30) + 2

        # Store metadata
        farm_group.attrs["farm_id"] = farm_id
        farm_group.attrs["bbox"] = list(bbox)
        farm_group.attrs["H"] = H
        farm_group.attrs["W"] = W
        farm_group.attrs["n_dates"] = len(dates)

        # Data array: (D, K, H, W) fp16
        data = farm_group.create_array(
            "data", shape=(len(dates), 7, H, W),
            dtype="float16", chunks=(1, 7, H, W),
            fill_value=np.nan,
        )
        valid = farm_group.create_array(
            "pixel_valid", shape=(len(dates), H, W),
            dtype="uint8", chunks=(1, H, W),
            fill_value=0,
        )

        # Process each date
        for di, date_str in enumerate(dates):
            scenes = find_scene_for_date(date_str, SCENES_ROOT)
            if not scenes:
                continue

            # Use first scene (or merge if multiple paths)
            scene_dir = scenes[0]
            # Read each band
            for bi, band_name in enumerate(BAND_ORDER):
                band_file = FILE_MAP[band_name]
                result = read_band_window(scene_dir, band_file, bbox, None)
                if result is None:
                    continue
                arr, window = result
                # Place into data array
                h, w = arr.shape
                data[di, bi, :h, :w] = arr.astype("float16")

            # Quality mask
            qa_path = scene_dir / "qa_pixel.tif"
            if qa_path.exists():
                with rasterio.open(qa_path) as src:
                    from rasterio.windows import from_bounds
                    window = from_bounds(*bbox, transform=src.transform)
                    window = window.round_offsets().round_shape()
                    qa = src.read(1, window=window)
                    qmask = quality_mask(qa, qcfg)
                    valid[di, :qmask.shape[0], :qmask.shape[1]] = qmask.astype("uint8")

            if di % 50 == 0:
                print(f"    {farm_id}: date {di+1}/{len(dates)}")

        # Append to manifest
        with open(manifest_path, "a") as f:
            f.write(json.dumps({"farm_id": farm_id, "n_dates": len(dates)}) + "\n")

        print(f"  Done: {farm_id} ({len(dates)} dates)")

    print("Extraction complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--farms", default="D:/RS/Thesis/Data/RoI/Farms_subset.geojson")
    ap.add_argument("--out", default="D:/RS/Thesis/Data/Processed/raw_satellite_subset.zarr")
    ap.add_argument("--smoke", type=int, default=None, help="Process only N dates")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    # Load farm polygons
    farms, gdf = load_farm_polygons(Path(args.farms))
    print(f"Loaded {len(farms)} farms from {args.farms}")

    # Get date list from stats.zarr
    import zarr as zr
    z = zr.open(str(STATS_ZARR), mode="r")
    # Convert day numbers to date strings — need mapping
    # stats.zarr time is days since 2013-01-01
    from datetime import timedelta, date
    base_date = date(2013, 1, 1)
    dates = []
    for d in z["time"][:]:
        dt = base_date + timedelta(days=int(d))
        dates.append(dt.strftime("%Y-%m-%d"))

    print(f"Found {len(dates)} dates")

    max_dates = args.smoke
    build_subset(farms, dates, args.out, max_dates=max_dates, resume=args.resume)


if __name__ == "__main__":
    main()

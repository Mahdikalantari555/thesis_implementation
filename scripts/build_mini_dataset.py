"""Build the additive three-farm mini satellite store.

Creates ``Data/Processed/mini_research_dataset/satellite.zarr`` with one group
per farm. Each group holds seven physical bands ``(D, 7, H_i, W_i)`` in fp16,
``pixel_valid`` ``(D, H_i, W_i)`` and ``in_field`` ``(H_i, W_i)`` masks, plus
the real acquisition dates and pixel-grid metadata. Zarr format v2 is used for
portability (design.md §1).

The date list comes from ``scene_inventory.parquet`` acquisition dates, never
from ``stats.zarr`` day-numbers (docs/mini_research_dataset.md §2).

Usage::

    python scripts/build_mini_dataset.py --smoke 5 --only-farm R12-08
    python scripts/build_mini_dataset.py --resume
    python scripts/build_mini_dataset.py --check-parity
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import zarr
from affine import Affine
from rasterio.windows import Window
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.config import QualityCfg  # noqa: E402
from ssl_anomaly.io.landsat import quality_mask  # noqa: E402
from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402

DATA = Path("D:/RS/Thesis/Data")
FARMS_GEOJSON = DATA / "RoI" / "Farms.geojson"
INVENTORY = DATA / "Processed" / "satellite" / "scene_inventory.parquet"
SCENES_ROOT = DATA / "Metric" / "scenes"
ZONAL_DIR = DATA / "Processed" / "satellite" / "zonal_stats"
PIXELS_DIR = DATA / "Processed" / "satellite" / "pixels"
OUT_ROOT = DATA / "Processed" / "mini_research_dataset"
DEFAULT_FARMS = ("R12-06", "R12-08", "R12-10")

FILE_MAP = OrderedDict(
    [
        ("blue", "blue.tif"),
        ("green", "green.tif"),
        ("red", "red.tif"),
        ("nir", "nir08.tif"),
        ("swir1", "swir16.tif"),
        ("swir2", "swir22.tif"),
        ("lst", "lwir11.tif"),
    ]
)
BAND_ORDER = list(FILE_MAP)
QA_FILE = "qa_pixel.tif"
ZARR_FORMAT = 2

# Parity tolerances: reflectance is unitless, LST is Kelvin. These are wide
# enough for the 2026-11-24 QA re-masking difference and still catch a wrong
# band or a unit error. Failures are always reported verbatim.
PARITY_TOL = {
    **{b: {"atol": 0.02, "rtol": 0.05} for b in BAND_ORDER if b != "lst"},
    "lst": {"atol": 1.0, "rtol": 0.01},
}


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_farms(geojson: Path, names: tuple[str, ...]) -> list[dict[str, Any]]:
    import geopandas as gpd

    from shapely import force_2d

    gdf = gpd.read_file(geojson)
    gdf = gdf[gdf["Farm"].isin(names)].copy()
    if gdf.empty:
        raise RuntimeError(f"No target farms found in {geojson}: {names}")
    gdf = gdf.to_crs("EPSG:32639")
    farms = []
    for _, row in gdf.iterrows():
        geom = force_2d(row.geometry)  # 3D polygons are not accepted by GDAL rasterize
        farms.append(
            {
                "farm_id": str(row["Farm"]),
                "ogc_fid": int(row["OGC_FID"]),
                "area_ha": float(row["area"]),
                "geometry": geom,
                "bbox": tuple(float(v) for v in geom.bounds),
            }
        )
    farms.sort(key=lambda f: f["farm_id"])
    found = {f["farm_id"] for f in farms}
    missing = set(names) - found
    if missing:
        raise RuntimeError(f"Missing requested farms: {sorted(missing)}")
    return farms


def load_dates(inventory: Path, limit: int | None = None) -> tuple[list[str], dict[str, str]]:
    import pandas as pd

    df = pd.read_parquet(inventory)
    col = "acquisition_date"
    if col not in df.columns:
        raise RuntimeError(f"{inventory} has no {col!r} column")
    dates = [str(d)[:10] for d in df[col].astype(str).tolist()]
    if len(set(dates)) != len(dates):
        raise RuntimeError("Duplicate acquisition dates in scene inventory")
    # The inventory is not stored in chronological row order; a date-centric
    # time series must be sorted before windows, Δt or holdouts are derived.
    # Keep each date paired with its own scene_id while sorting.
    pairs = sorted(zip(dates, df["scene_id"].astype(str).tolist()), key=lambda p: p[0])
    dates = [d for d, _ in pairs]
    scene_ids = dict(pairs)
    if limit is not None:
        if limit < 1:
            raise ValueError("--smoke must be >= 1")
        dates = dates[:limit]
    return dates, scene_ids


def scene_dir_for(date: str, scene_id: str, root: Path) -> Path | None:
    """Resolve the unique scene directory for a date; None if absent."""
    ymd = date.replace("-", "")
    # Inventory scene_id encodes WRS path/row in token 2, e.g. 165039.
    parts = scene_id.split("_")
    if len(parts) > 2 and len(parts[2]) >= 6:
        path, row = parts[2][:3], parts[2][3:6]
        exact = root / f"landsat_{ymd}_{path}_{row}"
        if exact.is_dir():
            return exact
    candidates = sorted(root.glob(f"landsat_{ymd}_*"))
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous scene dirs for {date}: {candidates}")
    return candidates[0] if candidates else None


def derive_canonical_grid(
    farm_id: str,
    ogc_fid: int,
    dates: list[str],
    scene_transform: Affine,
    pixels_dir: Path,
) -> tuple[Window, Affine, str, np.ndarray, tuple[float, float, float, float]]:
    """Recover window, transform and in-field mask from the processed pixels.

    The canonical archive (``pixels/``, ``zonal_stats/``, ``stats.zarr``) places
    every farm ~270 m east of its current ``Farms.geojson`` polygon, yet does so
    consistently. Both the pixel window and the in-field mask therefore come
    from the real processed pixel membership, not from re-rasterizing the
    current polygon; this preserves parity with the frozen archive. The offset
    is recorded in store attributes rather than silently corrected.
    """
    import pandas as pd

    for date in dates:
        path = pixels_dir / f"{date}.parquet"
        if not path.exists():
            continue
        sub = pd.read_parquet(path, columns=["field_id", "row", "col"])
        sub = sub[sub["field_id"] == ogc_fid]
        if sub.empty:
            continue
        col = int(sub["col"].min())
        row = int(sub["row"].min())
        width = int(sub["col"].max()) - col + 1
        height = int(sub["row"].max()) - row + 1
        window = Window(col, row, width, height)
        transform = scene_transform * Affine.translation(col, row)
        mask = np.zeros((height, width), dtype=np.uint8)
        rows = sub["row"].to_numpy(dtype=int) - row
        cols = sub["col"].to_numpy(dtype=int) - col
        mask[rows, cols] = 1
        x0 = float(transform.c)
        y0 = float(transform.f)
        grid_bbox = (x0, y0 - height * 30.0, x0 + width * 30.0, y0)
        return window, transform, str(path), mask, grid_bbox
    raise RuntimeError(f"No canonical pixel grid found for {farm_id} (fid={ogc_fid})")


def build_farm(
    farm: dict[str, Any],
    dates: list[str],
    scene_ids: dict[str, str],
    store: Path,
    scenes_root: Path,
    pixels_dir: Path,
    qcfg: QualityCfg,
    progress_path: Path,
) -> dict[str, Any]:
    farm_id = farm["farm_id"]
    group_path = store / farm_id
    group = zarr.open_group(str(group_path), mode="w", zarr_format=ZARR_FORMAT)
    t0 = time.monotonic()

    # Determine the pixel grid from the first available scene (verified stable
    # across all dates for a given farm).
    first_scene: Path | None = None
    for date in dates:
        candidate = scene_dir_for(date, scene_ids.get(date, ""), scenes_root)
        if candidate is not None and (candidate / FILE_MAP["blue"]).exists():
            first_scene = candidate
            break
    if first_scene is None:
        raise RuntimeError(f"No scene with a blue band for farm {farm_id}")

    with rasterio.open(first_scene / FILE_MAP["blue"]) as src:
        scene_transform = src.transform
    # Use the canonical row/col grid from the processed pixels store. Deriving
    # it from a rounded bbox with rasterio.windows.from_bounds is both offset-
    # error prone and natively unstable on this GDAL build.
    window, grid_transform, grid_reference, in_field_mask, grid_bbox = (
        derive_canonical_grid(
            farm_id, farm["ogc_fid"], dates, scene_transform, pixels_dir
        )
    )
    h, w = int(window.height), int(window.width)
    if h < 1 or w < 1:
        raise RuntimeError(f"Empty pixel window for {farm_id}: {h}x{w}")

    dates_arr = group.create_array(
        "dates", shape=(len(dates),), dtype="<U10", fill_value=""
    )
    data = group.create_array(
        "data",
        shape=(len(dates), len(BAND_ORDER), h, w),
        dtype="float16",
        chunks=(1, len(BAND_ORDER), h, w),
        fill_value=np.nan,
    )
    valid = group.create_array(
        "pixel_valid",
        shape=(len(dates), h, w),
        dtype="uint8",
        chunks=(1, h, w),
        fill_value=0,
    )
    in_field_arr = group.create_array("in_field", shape=(h, w), dtype="uint8", fill_value=0)
    in_field_arr[:] = in_field_mask

    group.attrs.update(
        {
            "farm_id": farm_id,
            "ogc_fid": farm["ogc_fid"],
            "area_ha": farm["area_ha"],
            "bbox": list(grid_bbox),
            "bbox_geojson_utm": list(farm["bbox"]),
            "geometry_source": grid_reference,
            "geometry_offset_note": (
                "Canonical processed pixels are shifted vs current Farms.geojson; "
                "in_field comes from processed pixel membership to preserve parity."
            ),
            "H": h,
            "W": w,
            "n_dates": len(dates),
            "crs": "EPSG:32639",
            "pixel_size_m": 30.0,
            "grid_transform": list(grid_transform)[:6],
            "grid_reference": grid_reference,
            "zarr_format": ZARR_FORMAT,
            "band_order": BAND_ORDER,
            "date_source": "scene_inventory.parquet:acquisition_date",
            "quality": "Landsat C2 QA_PIXEL bits + finite-value gate",
        }
    )

    n_present = 0
    n_missing = 0
    with tqdm(
        enumerate(dates),
        total=len(dates),
        desc=f"{farm_id} extract",
        unit="date",
    ) as bar:
        for di, date in bar:
            dates_arr[di] = date
            scene = scene_dir_for(date, scene_ids.get(date, ""), scenes_root)
            qa_path = scene / QA_FILE if scene is not None else None
            if scene is None or qa_path is None or not qa_path.exists():
                n_missing += 1
                rec = {
                    "farm": farm_id,
                    "date": date,
                    "index": di + 1,
                    "status": "absent_scene_or_qa",
                    "wall_s": round(time.monotonic() - t0, 3),
                }
                with progress_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
                bar.set_postfix(missing=n_missing)
                continue

            with rasterio.open(qa_path) as src:
                qa = src.read(1, window=window).astype(np.uint16)
            qmask = quality_mask(qa, qcfg)
            cube = np.full((len(BAND_ORDER), h, w), np.nan, dtype=np.float32)
            for bi, band in enumerate(BAND_ORDER):
                path = scene / FILE_MAP[band]
                if not path.exists():
                    continue
                with rasterio.open(path) as src:
                    arr = src.read(1, window=window)
                if arr.shape != (h, w):
                    raise RuntimeError(
                        f"{farm_id} {date} {band}: window {arr.shape} != grid {(h, w)}"
                    )
                # METRIC scene rasters are already scaled to physical units:
                # reflectance ~0..1, LST in Kelvin. Re-applying the raw
                # Landsat DN scale/offset would produce all-NaN reflectance.
                cube[bi] = arr.astype(np.float32)

            finite = np.all(np.isfinite(cube), axis=0)
            vmask = (qmask & finite).astype(np.uint8)
            data[di] = cube.astype(np.float16)
            valid[di] = vmask
            n_present += 1
            field_mask = in_field_arr[:].astype(bool)
            valid_frac = float(vmask[field_mask].mean()) if field_mask.any() else 0.0
            bar.set_postfix(valid=round(valid_frac, 3))
            rec = {
                "farm": farm_id,
                "date": date,
                "index": di + 1,
                "status": "ok",
                "valid_frac_in_field": round(valid_frac, 5),
                "wall_s": round(time.monotonic() - t0, 3),
            }
            with progress_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

    wall = round(time.monotonic() - t0, 2)
    group.attrs["n_present_dates"] = n_present
    group.attrs["n_absent_dates"] = n_missing
    group.attrs["wall_s"] = wall
    return {
        "farm_id": farm_id,
        "ogc_fid": farm["ogc_fid"],
        "n_dates": len(dates),
        "n_present_dates": n_present,
        "n_absent_dates": n_missing,
        "H": h,
        "W": w,
        "n_in_field_pixels": int(in_field_arr[:].sum()),
        "bbox_grid": list(grid_bbox),
        "grid_reference": grid_reference,
        "wall_s": wall,
        "zarr_path": str(group_path),
    }


def write_farms_geojson(out: Path, farms: list[dict[str, Any]]) -> Path:
    """Write the three-farm subset GeoJSON (EPSG:32639, 2D)."""
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {
            "Farm": [f["farm_id"] for f in farms],
            "OGC_FID": [f["ogc_fid"] for f in farms],
            "area": [f["area_ha"] for f in farms],
        },
        geometry=[f["geometry"] for f in farms],
        crs="EPSG:32639",
    )
    path = out / "farms.geojson"
    gdf.to_file(path, driver="GeoJSON")
    return path


def write_manifest(
    out: Path,
    farms: list[dict[str, Any]],
    dates: list[str],
    results: list[dict[str, Any]],
    wall_s: float,
    smoke: int | None,
) -> Path:
    manifest = {
        "schema": "mini-research-dataset/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_commit_sha(),
        "python": platform.python_version(),
        "zarr_format": ZARR_FORMAT,
        "date_source": str(INVENTORY),
        "date_column": "acquisition_date",
        "n_dates": len(dates),
        "first_date": dates[0],
        "last_date": dates[-1],
        "date_list_sha256": _sha256_text("\n".join(dates)),
        "farms_source": str(FARMS_GEOJSON),
        "farms": [
            {
                "farm_id": f["farm_id"],
                "ogc_fid": f["ogc_fid"],
                "area_ha": f["area_ha"],
                "bbox": list(f["bbox"]),
            }
            for f in farms
        ],
        "scenes_root": str(SCENES_ROOT),
        "band_order": BAND_ORDER,
        "band_units": "reflectance 0..1 and LST Kelvin, already scaled in METRIC scene rasters (no DN conversion applied)",
        "quality_cfg": {
            "cloud": True,
            "cloud_shadow": True,
            "cirrus": True,
            "snow": True,
            "dilated_cloud": True,
            "fill": True,
        },
        "smoke_dates": smoke,
        "farm_results": results,
        "wall_s": wall_s,
        "output_root": str(out),
    }
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def check_parity(store: Path, results: list[dict[str, Any]], out_json: Path) -> bool:
    """Compare per-band median of stored pixels with zonal_stats medians."""
    import pandas as pd

    report: dict[str, Any] = {
        "schema": "mini-parity/v1",
        "git_sha": get_git_commit_sha(),
        "store": str(store),
        "zonal_dir": str(ZONAL_DIR),
        "tolerances": PARITY_TOL,
        "checks": [],
        "passed": True,
    }
    for res in results:
        farm_id, fid = res["farm_id"], res["ogc_fid"]
        group = zarr.open_group(str(store / farm_id), mode="r")
        dates = [str(d) for d in group["dates"][:]]
        data = group["data"]
        valid = group["pixel_valid"][:]
        in_field = group["in_field"][:].astype(bool)
        for di, date in enumerate(dates):
            zpath = ZONAL_DIR / f"{date}.parquet"
            if not zpath.exists():
                continue
            zdf = pd.read_parquet(zpath)
            row = zdf[zdf["field"] == fid]
            if row.empty:
                continue
            row = row.iloc[0]
            mask = valid[di].astype(bool) & in_field
            for bi, band in enumerate(BAND_ORDER):
                vals = data[di, bi][mask]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                expected = row[f"{band}_median"]
                if pd.isna(expected):
                    continue
                got = float(np.median(vals.astype(np.float64)))
                tol = PARITY_TOL[band]
                ok = bool(abs(got - float(expected)) <= tol["atol"] + tol["rtol"] * abs(float(expected)))
                if not ok:
                    report["passed"] = False
                report["checks"].append(
                    {
                        "farm": farm_id,
                        "date": date,
                        "band": band,
                        "n": int(vals.size),
                        "stored_median": round(got, 6),
                        "zonal_median": round(float(expected), 6),
                        "abs_diff": round(abs(got - float(expected)), 6),
                        "pass": ok,
                    }
                )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return bool(report["passed"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--farms-geojson", type=Path, default=FARMS_GEOJSON)
    ap.add_argument("--farms", nargs="+", default=list(DEFAULT_FARMS), help="Farm names")
    ap.add_argument("--only-farm", default=None, help="Restrict to one farm (smoke)")
    ap.add_argument("--inventory", type=Path, default=INVENTORY)
    ap.add_argument("--scenes", type=Path, default=SCENES_ROOT)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--smoke", type=int, default=None, help="First N dates only")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--check-parity", action="store_true", help="Also run parity check")
    ap.add_argument("--parity-out", type=Path, default=ROOT / "reports" / "mini_dataset" / "parity.json")
    args = ap.parse_args(argv)

    t0 = time.monotonic()
    names = tuple([args.only_farm] if args.only_farm else args.farms)
    farms = load_farms(args.farms_geojson, names)
    dates, scene_ids = load_dates(args.inventory, args.smoke)
    print(
        f"Loaded {len(farms)} farms and {len(dates)} dates "
        f"({dates[0]} .. {dates[-1]}); date source = {args.inventory.name}",
        flush=True,
    )

    out = args.out
    store = out / "satellite.zarr"
    store.mkdir(parents=True, exist_ok=True)
    progress = out / "progress_extraction.jsonl"
    manifest_lines = store / ".manifest.jsonl"
    done = set()
    if args.resume and manifest_lines.exists():
        for line in manifest_lines.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["farm_id"])

    qcfg = QualityCfg()
    results: list[dict[str, Any]] = []
    for farm in farms:
        if farm["farm_id"] in done:
            print(f"  Skipping {farm['farm_id']} (already in manifest)", flush=True)
            continue
        res = build_farm(
            farm, dates, scene_ids, store, args.scenes, PIXELS_DIR, qcfg, progress
        )
        results.append(res)
        with manifest_lines.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(res) + "\n")
        print(
            f"  Done {farm['farm_id']}: {res['n_present_dates']}/{res['n_dates']} dates, "
            f"grid {res['H']}x{res['W']}, in-field px {res['n_in_field_pixels']}, "
            f"wall {res['wall_s']}s",
            flush=True,
        )

    wall = round(time.monotonic() - t0, 2)
    if not results:
        print("No farms processed (all already complete on --resume).", flush=True)
        return 0
    farms_geojson = write_farms_geojson(out, farms)
    manifest = write_manifest(out, farms, dates, results, wall, args.smoke)
    parity_ok: bool | None = None
    if args.check_parity:
        parity_ok = check_parity(store, results, args.parity_out)
        print(f"Parity {'PASS' if parity_ok else 'FAIL'} -> {args.parity_out}", flush=True)
    print(
        f"EXTRACTION COMPLETE wall={wall}s store={store} farms={farms_geojson} manifest={manifest} "
        f"progress={progress} parity={parity_ok}",
        flush=True,
    )
    return 0 if parity_ok is not False else 2


if __name__ == "__main__":
    raise SystemExit(main())

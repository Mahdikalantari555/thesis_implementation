"""Build the additive three-farm mini satellite store.

Creates ``Data/Processed/mini_research_dataset/satellite.zarr`` with one group
per farm. Each group holds seven physical bands ``(D, 7, H_i, W_i)`` in fp16,
``pixel_valid`` ``(D, H_i, W_i)`` and ``in_field`` ``(H_i, W_i)`` masks, plus
the real acquisition dates and pixel-grid metadata. Zarr format v2 is used for
portability (design.md §1).

The farm set comes from the selection layer ``RoI/3farms_dev.geojson`` (owner
decision 2026-11-25); ``Farms.geojson`` is used only to map each farm name to
the archive id. Each group records ``archive_alignment``: IoU of the archive's
pixel membership against the farm's own polygon and its neighbouring ids.

The date list comes from ``scene_inventory.parquet`` acquisition dates, never
from ``stats.zarr`` day-numbers (docs/mini_research_dataset.md §2).

Usage::

    python scripts/build_mini_dataset.py --smoke 5 --only-farm <FARM>
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
from ssl_anomaly.io.data_contract import convert_st_qa_dn  # noqa: E402
from ssl_anomaly.io.landsat import quality_mask  # noqa: E402
from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402
from ssl_anomaly.utils.paths import localize  # noqa: E402

DATA = localize("D:/RS/Thesis/Data")
# Farm selection layer (owner decision 2026-11-25): the mini dataset uses the
# three farms of ``3farms_dev.geojson``, never a hard-coded name list. The full
# ``Farms.geojson`` is only the ID reference that maps each farm name to the
# archive ``field_id`` (OGC_FID) used by ``pixels/`` and ``zonal_stats/``.
FARMS_LAYER_CANDIDATES = (
    DATA / "RoI" / "3farms_dev.geojson",
    DATA / "RoI" / "3farms-dev.geojson",
)
FARMS_REFERENCE = DATA / "RoI" / "Farms.geojson"
EXPECTED_FARMS = 3

# Field-id alias, measured 2026-11-25 (openspec change `field-id-provenance-fix`).
# The canonical pixel archive stores the physical field of reference fid k under
# field_id k+1: archive id k had IoU 1.0 with polygon k-1 and 0.0 with k, while
# archive id k+1 has IoU 1.0 with polygon k and 0.0 with k+-1. Verified for all
# three farms of the selection layer with rasterio rasterize (all_touched=False).
# The canonical 490-field archive is NOT re-indexed; this alias is mini-only.
ID_ALIAS_EVIDENCE: dict[str, dict[str, Any]] = {
    "L02-02": {"reference_fid": 300, "archive_fid": 301, "iou_own_after_alias": 1.0,
               "matched_fid_before_alias": 299, "iou_neighbour_before_alias": 1.0},
    "R12-08": {"reference_fid": 245, "archive_fid": 246, "iou_own_after_alias": 1.0,
               "matched_fid_before_alias": 244, "iou_neighbour_before_alias": 1.0},
    "R15-18": {"reference_fid": 17, "archive_fid": 18, "iou_own_after_alias": 1.0,
               "matched_fid_before_alias": 16, "iou_neighbour_before_alias": 1.0},
}
INVENTORY = DATA / "Processed" / "satellite" / "scene_inventory.parquet"
SCENES_ROOT = DATA / "Metric" / "scenes"
ZONAL_DIR = DATA / "Processed" / "satellite" / "zonal_stats"
PIXELS_DIR = DATA / "Processed" / "satellite" / "pixels"
OUT_ROOT = DATA / "Processed" / "mini_research_dataset"

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
# Task 0A.4 quality-signal layers; ingested when the scene provides them.
QA_RADSAT_FILE = "qa_radsat.tif"
ST_QA_FILE = "st_qa.tif"
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


def default_farms_layer() -> Path:
    """First existing candidate of the 3-farm selection layer."""
    for cand in FARMS_LAYER_CANDIDATES:
        if cand.exists():
            return cand
    return FARMS_LAYER_CANDIDATES[0]


def _attr_value(value: Any) -> Any:
    """JSON-safe scalar for zarr attrs / manifest (NaN -> None)."""
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (int, float, bool, str)):
        return value
    return str(value)


def resolve_farms(
    layer: "Any",
    reference: "Any",
    names: tuple[str, ...] | None = None,
    expected_count: int | None = EXPECTED_FARMS,
    alias_evidence: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Resolve the selection layer against the ID reference.

    ``layer`` and ``reference`` are GeoDataFrames already in EPSG:32639.
    The farm set comes from the layer; ``names`` only narrows it (smoke runs).
    ``ogc_fid`` is the reference polygon id (farm identity); when
    ``alias_evidence`` is supplied, ``archive_fid`` is the measured pixel id and
    a farm without evidence fails closed. Geometry comes from the layer; the
    layer-vs-reference centroid distance is recorded so a silent geometry
    mismatch is visible.
    """
    from shapely import force_2d

    if "Farm" not in layer.columns:
        raise RuntimeError("Selection layer has no 'Farm' column")
    lay = layer.copy()
    lay["Farm"] = lay["Farm"].astype(str)
    if lay["Farm"].duplicated().any():
        raise RuntimeError(f"Duplicate farm names in selection layer: {sorted(lay.loc[lay['Farm'].duplicated(), 'Farm'])}")
    if names:
        missing = set(names) - set(lay["Farm"])
        if missing:
            raise RuntimeError(f"Requested farms not in selection layer: {sorted(missing)}")
        lay = lay[lay["Farm"].isin(names)]
    elif expected_count is not None and len(lay) != expected_count:
        raise RuntimeError(
            f"Selection layer has {len(lay)} farms, expected {expected_count}: {sorted(lay['Farm'])}"
        )

    ref = reference.copy()
    ref["Farm"] = ref["Farm"].astype(str)
    farms = []
    for _, row in lay.iterrows():
        name = row["Farm"]
        hits = ref[ref["Farm"] == name]
        if len(hits) != 1:
            raise RuntimeError(f"Farm {name!r} matches {len(hits)} rows in the ID reference (need exactly 1)")
        ref_row = hits.iloc[0]
        geom = force_2d(row.geometry)  # 3D polygons are not accepted by GDAL rasterize
        ref_geom = force_2d(ref_row.geometry)
        layer_fid = row["OGC_FID"] if "OGC_FID" in lay.columns else None
        attrs = {
            str(k): _attr_value(v)
            for k, v in row.items()
            if k not in ("geometry",)
        }
        area = row["area"] if "area" in lay.columns else ref_row.get("area", geom.area / 1e4)
        reference_fid = int(ref_row["OGC_FID"])
        archive_fid = reference_fid
        alias = None
        if alias_evidence is not None:
            if name not in alias_evidence:
                raise RuntimeError(
                    f"No measured field-id alias evidence for farm {name!r}; "
                    "refusing to guess the archive pixel id (openspec field-id-provenance-fix R2)"
                )
            alias = dict(alias_evidence[name])
            if int(alias["reference_fid"]) != reference_fid:
                raise RuntimeError(
                    f"Alias evidence for {name!r} targets fid {alias['reference_fid']} "
                    f"but the reference says {reference_fid}; refusing to apply it"
                )
            archive_fid = int(alias["archive_fid"])
        farms.append(
            {
                "farm_id": name,
                "ogc_fid": reference_fid,
                "reference_fid": reference_fid,
                "archive_fid": archive_fid,
                "id_alias": alias,
                "layer_ogc_fid": _attr_value(layer_fid),
                "area_ha": float(area),
                "geometry": geom,
                "bbox": tuple(float(v) for v in geom.bounds),
                "layer_vs_reference_centroid_m": round(float(geom.centroid.distance(ref_geom.centroid)), 2),
                "source_attributes": attrs,
            }
        )
    farms.sort(key=lambda f: f["farm_id"])
    return farms


def load_farms(
    layer_path: Path,
    reference_path: Path,
    names: tuple[str, ...] | None = None,
    expected_count: int | None = EXPECTED_FARMS,
) -> tuple[list[dict[str, Any]], "Any"]:
    """Read both GeoJSONs, reproject to EPSG:32639, resolve. Returns (farms, reference)."""
    import geopandas as gpd

    if not layer_path.exists():
        raise FileNotFoundError(
            f"Farm selection layer not found: {layer_path} (tried {[str(c) for c in FARMS_LAYER_CANDIDATES]})"
        )
    layer = gpd.read_file(layer_path).to_crs("EPSG:32639")
    reference = gpd.read_file(reference_path).to_crs("EPSG:32639")
    farms = resolve_farms(layer, reference, names, expected_count, ID_ALIAS_EVIDENCE)
    return farms, reference


def archive_alignment(
    in_field_mask: np.ndarray,
    grid_transform: Affine,
    reference: "Any",
    ogc_fid: int,
    layer_geometry: Any = None,
    neighbours: int = 1,
) -> dict[str, Any]:
    """Compare the archive's pixel membership with candidate polygons.

    Reports IoU of the archive in-field mask against the reference polygons of
    ``ogc_fid`` and its +-``neighbours`` ids, plus the selection-layer polygon.
    A best match at a neighbouring id is evidence for an id off-by-one in the
    archive rather than a georeferencing shift. Diagnostic only: the pixel grid
    itself is not changed (parity with the frozen archive is preserved).
    """
    from rasterio.features import rasterize
    from shapely import force_2d

    shape = in_field_mask.shape
    arch = in_field_mask.astype(bool)

    def iou(geom: Any) -> float:
        m = rasterize([(force_2d(geom), 1)], out_shape=shape, transform=grid_transform,
                      fill=0, dtype="uint8", all_touched=False).astype(bool)
        union = (m | arch).sum()
        return round(float((m & arch).sum() / union), 4) if union else 0.0

    out: dict[str, Any] = {"by_reference_fid": {}}
    for d in range(-neighbours, neighbours + 1):
        fid = ogc_fid + d
        hits = reference[reference["OGC_FID"] == fid]
        if len(hits) == 1:
            out["by_reference_fid"][str(fid)] = {
                "farm": str(hits.iloc[0]["Farm"]),
                "iou": iou(hits.iloc[0].geometry),
            }
    if layer_geometry is not None:
        out["selection_layer_iou"] = iou(layer_geometry)
    if out["by_reference_fid"]:
        best = max(out["by_reference_fid"].items(), key=lambda kv: kv[1]["iou"])
        out["best_reference_fid"] = int(best[0])
        out["best_is_own_fid"] = int(best[0]) == ogc_fid
    return out


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
    reference: Any = None,
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
            farm_id, farm.get("archive_fid", farm["ogc_fid"]), dates, scene_transform, pixels_dir
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
    # Task 0A.4: QA_RADSAT (saturation) and ST_QA (LST uncertainty) are
    # ingested as quality signals WHEN the scene provides them. The current
    # METRIC scene dirs carry only qa_pixel.tif, so these arrays are created
    # lazily on first sight; their per-date absence is recorded in the group
    # attrs — never filled with zeros or imputed.
    qa_radsat_arr = None
    st_qa_arr = None
    n_qa_radsat = 0
    n_st_qa = 0
    in_field_arr = group.create_array("in_field", shape=(h, w), dtype="uint8", fill_value=0)
    in_field_arr[:] = in_field_mask

    alignment: dict[str, Any] = {}
    if reference is not None:
        alignment = archive_alignment(
            in_field_mask, grid_transform, reference, farm["ogc_fid"], farm["geometry"]
        )
        if not alignment.get("best_is_own_fid", True):
            print(
                f"  WARNING {farm_id}: archive pixels match reference fid "
                f"{alignment['best_reference_fid']} better than own fid {farm['ogc_fid']} "
                f"({alignment['by_reference_fid']})",
                flush=True,
            )

    group.attrs.update(
        {
            "farm_id": farm_id,
            "ogc_fid": farm["ogc_fid"],
            "reference_fid": farm.get("reference_fid", farm["ogc_fid"]),
            "archive_fid": farm.get("archive_fid", farm["ogc_fid"]),
            "id_alias": farm.get("id_alias"),
            "layer_ogc_fid": farm.get("layer_ogc_fid"),
            "source_attributes": farm.get("source_attributes", {}),
            "layer_vs_reference_centroid_m": farm.get("layer_vs_reference_centroid_m"),
            "archive_alignment": alignment,
            "area_ha": farm["area_ha"],
            "bbox": list(grid_bbox),
            "bbox_geojson_utm": list(farm["bbox"]),
            "geometry_source": grid_reference,
            "geometry_offset_note": (
                "Canonical processed pixels may be shifted vs the farm polygon (see "
                "archive_alignment); in_field comes from processed pixel membership "
                "to preserve parity."
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
            # 0A.4 counts are set after the date loop; initialised here so the
            # attrs always exist and absence is explicit.
            "n_qa_radsat_dates": 0,
            "n_st_qa_dates": 0,
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
            # Task 0A.4: ingest QA_RADSAT / ST_QA when the scene provides them.
            radsat_path = scene / QA_RADSAT_FILE
            if radsat_path.exists():
                if qa_radsat_arr is None:
                    qa_radsat_arr = group.create_array(
                        "qa_radsat", shape=(len(dates), h, w), dtype="uint16",
                        chunks=(1, h, w), fill_value=0,
                    )
                with rasterio.open(radsat_path) as src:
                    qa_radsat_arr[di] = src.read(1, window=window).astype(np.uint16)
                n_qa_radsat += 1
            stqa_path = scene / ST_QA_FILE
            if stqa_path.exists():
                if st_qa_arr is None:
                    st_qa_arr = group.create_array(
                        "st_qa", shape=(len(dates), h, w), dtype="float32",
                        chunks=(1, h, w), fill_value=np.nan,
                    )
                with rasterio.open(stqa_path) as src:
                    st_qa_arr[di] = convert_st_qa_dn(src.read(1, window=window))
                n_st_qa += 1
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
    # Task 0A.4: quality-signal coverage (absence = the scene set carries no
    # qa_radsat/st_qa rasters; recorded, never imputed).
    group.attrs["n_qa_radsat_dates"] = n_qa_radsat
    group.attrs["n_st_qa_dates"] = n_st_qa
    group.attrs["wall_s"] = wall
    return {
        "farm_id": farm_id,
        "ogc_fid": farm["ogc_fid"],
        "reference_fid": farm.get("reference_fid", farm["ogc_fid"]),
        "archive_fid": farm.get("archive_fid", farm["ogc_fid"]),
        "n_dates": len(dates),
        "n_present_dates": n_present,
        "n_absent_dates": n_missing,
        "H": h,
        "W": w,
        "n_in_field_pixels": int(in_field_arr[:].sum()),
        "bbox_grid": list(grid_bbox),
        "archive_alignment": alignment,
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
    farms_source: Path | None = None,
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
        "farms_source": str(farms_source),
        "farms_id_reference": str(FARMS_REFERENCE),
        "id_alias_rule": "archive_fid = reference_fid + 1 (measured 2026-11-25, mini-only)",
        "id_alias_caveat": (
            "The canonical 490-field archive still stores field k's pixels under "
            "field_id k+1; only this mini store has the alias applied. No L4 "
            "attribute claim is made from the shifted canonical ids."
        ),
        "farms": [
            {
                "farm_id": f["farm_id"],
                "ogc_fid": f["ogc_fid"],
                "reference_fid": f.get("reference_fid", f["ogc_fid"]),
                "archive_fid": f.get("archive_fid", f["ogc_fid"]),
                "id_alias": f.get("id_alias"),
                "area_ha": f["area_ha"],
                "bbox": list(f["bbox"]),
                "layer_ogc_fid": f.get("layer_ogc_fid"),
                "source_attributes": f.get("source_attributes", {}),
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
        farm_id = res["farm_id"]
        # Parity compares pixels with the canonical zonal table, which is keyed
        # by the ARCHIVE field id (the alias target), not the reference polygon
        # id. Using ogc_fid here would compare two different physical fields.
        fid = res.get("archive_fid", res["ogc_fid"])
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
    ap.add_argument("--farms-geojson", type=Path, default=None,
                    help="Farm selection layer (default: RoI/3farms_dev.geojson)")
    ap.add_argument("--id-reference", type=Path, default=FARMS_REFERENCE,
                    help="Full Farms.geojson used only to map farm name -> archive OGC_FID")
    ap.add_argument("--farms", nargs="+", default=None,
                    help="Optional subset of farm names from the selection layer")
    ap.add_argument("--expected-count", type=int, default=EXPECTED_FARMS)
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
    if args.farms_geojson is None:
        args.farms_geojson = default_farms_layer()
    if args.only_farm:
        names: tuple[str, ...] | None = (args.only_farm,)
    elif args.farms:
        names = tuple(args.farms)
    else:
        names = None
    farms, reference = load_farms(args.farms_geojson, args.id_reference, names, args.expected_count)
    print(
        "Farm selection layer: "
        f"{args.farms_geojson} -> {[(f['farm_id'], f['ogc_fid']) for f in farms]}",
        flush=True,
    )
    dates, scene_ids = load_dates(args.inventory, args.smoke)
    print(
        f"Loaded {len(farms)} farms and {len(dates)} dates "
        f"({dates[0]} .. {dates[-1]}); date source = {args.inventory.name}",
        flush=True,
    )

    out = args.out
    store = out / "satellite.zarr"
    selected = {f["farm_id"] for f in farms}
    if store.exists():
        stale = sorted(
            p.name for p in store.iterdir()
            if p.is_dir() and (p / ".zgroup").exists() and p.name not in selected
        )
        if stale:
            raise RuntimeError(
                f"{store} already holds farms outside the selection layer {stale}. "
                "Move the old store aside (or pass a new --out) before rebuilding; "
                "mixing farm sets in one store is refused."
            )
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
            farm, dates, scene_ids, store, args.scenes, PIXELS_DIR, qcfg, progress, reference
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
    manifest = write_manifest(out, farms, dates, results, wall, args.smoke, args.farms_geojson)
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

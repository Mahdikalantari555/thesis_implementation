"""Extract the date-centric raw satellite store for the 32-farm probe subset.

Builds ``raw_satellite_subset.zarr`` with per-farm arrays

    data       (D=538, K=7, S=48, S=48)  float16  — METRIC scene rasters, already
                                                    scaled to physical units
                                                    (reflectance; LST in Kelvin).
                                                    NaN stays NaN — never 0.
    pixel_valid (D=538, S=48, S=48)      uint8    — QA_PIXEL validity (1 = valid)
    in_field   (S=48, S=48)              uint8    — canonical archive membership
                                                    (union over dates of
                                                    pixels/<date>.parquet rows)
    dates      (D,)                      <U10

Conventions (established, do not silently change):
- **Field membership comes from the canonical processed pixel archive**
  (``pixels/<date>.parquet``), NOT from re-rasterizing ``Farms.geojson``: the
  archive stores the physical field of reference fid ``k`` under ``field_id
  k+1`` (alias verified for all 490 fields, WP-1 E0 audit 2026-11-27, commit
  ``9816aaa``). This preserves exact parity with ``stats_long.parquet`` /
  ``zonal_stats/``.
- The scene rasters under ``Data/Metric/scenes`` are already scaled (reflectance
  ~0..1, LST Kelvin) — same convention as ``scripts/extract_stats.py`` and
  ``scripts/build_mini_dataset.py``; no further scaling is applied here.
- Values are stored as float16 (mini-store convention). Parity vs float64
  stats is therefore tolerance-based, never exact (see
  ``scripts/probe/parity_subset_stats.py``).
- NaN is never filled with 0 (§8: NaN silently filled is a defect).

Windows: the farm's canonical bounding box (from archive membership) is placed
centred on a fixed S×S canvas (tasks.md 0b.1: S=48). All 32 subset farms fit
(measured max ~12.6 × 34.8 px). Pixels outside the farm window stay NaN/0.

Progress: tqdm bar over dates, machine-readable ``progress.jsonl`` (one line
per date), resume support (``--resume``; completeness-verified — farm groups
without a correctly shaped ``data`` array are rebuilt). A full run (538 dates
× 32 farms × 8 windowed reads) takes minutes — launch it with the exact
command from the WP-3 handover, not interactively.

Usage::

    python scripts/build_raw_satellite_subset.py --smoke 3 --out .tmp/probe_smoke/subset.zarr
    python scripts/build_raw_satellite_subset.py --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import OrderedDict
from datetime import date as _date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.io.landsat import quality_mask  # noqa: E402
from ssl_anomaly.config import QualityCfg  # noqa: E402
from ssl_anomaly.utils.paths import localize  # noqa: E402
from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402

# Windows-style canonical defaults are kept on purpose (owner commands stay
# unchanged); ``localize`` maps them onto the Linux mirror at import time.
DATA = localize("D:/RS/Thesis/Data")

FILE_MAP = OrderedDict([
    ("blue", "blue.tif"), ("green", "green.tif"), ("red", "red.tif"),
    ("nir", "nir08.tif"), ("swir1", "swir16.tif"), ("swir2", "swir22.tif"),
    ("lst", "lwir11.tif"),
])
QA_FILE = "qa_pixel.tif"
BAND_ORDER = list(FILE_MAP)
ZARR_FORMAT = 2
SCENES_ROOT = DATA / "Metric" / "scenes"
STATS_ZARR = DATA / "Processed" / "satellite" / "stats.zarr"
PIXELS_DIR = DATA / "Processed" / "satellite" / "pixels"
FARMS_LAYER = DATA / "RoI" / "Farms_subset.geojson"
OUT_DEFAULT = DATA / "Processed" / "raw_satellite_subset.zarr"
EXPECTED_FARMS = 32
ID_ALIAS_NOTE = (
    "archive_fid = reference_fid(OGC_FID) + 1; verified for all 490 fields "
    "(docs/audits/e0_cycle_completeness_2026-11-27.md, commit 9816aaa)"
)


def parse_dates_from_stats_zarr(stats_zarr: Path, pixels_dir: Path | None = None) -> list[str]:
    """Canonical 538-date list from stats.zarr ``time``.

    The coordinate is **days since 2013-03-30** (the scene epoch used by the
    v1/v2 cycle tables and verified by the WP-1 E0 audit, commit ``9816aaa``).
    NOT days since 2013-01-01: that wrong base silently misaligns 370/538 dates
    (the pre-repair builder carried this bug).

    When ``pixels_dir`` is given, the parsed list is cross-checked against the
    canonical ``pixels/<date>.parquet`` archive and any disagreement fails
    loudly (§8: a silently misaligned date axis is a defect).
    """
    z = zarr.open(str(stats_zarr), mode="r")
    t = np.asarray(z["time"][:])
    base = _date(2013, 3, 30)
    dates = [(base + timedelta(days=int(d))).isoformat() for d in t]
    if len(set(dates)) != len(dates):
        raise RuntimeError("Duplicate dates in stats.zarr time coordinate")
    if pixels_dir is not None:
        px = {p.name[:-8] for p in pixels_dir.glob("*.parquet")}
        missing = sorted(set(dates) - px)
        extra = sorted(px - set(dates))
        if missing or extra:
            raise RuntimeError(
                f"stats.zarr dates disagree with pixels archive: "
                f"{len(missing)} dates missing from pixels (e.g. {missing[:3]}), "
                f"{len(extra)} pixel dates not in stats.zarr (e.g. {extra[:3]})"
            )
    return dates


def load_subset_farms(geojson_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load the 32-farm selection layer with reference/archive fids."""
    import geopandas as gpd

    gdf = gpd.read_file(geojson_path)
    if "OGC_FID" not in gdf.columns:
        raise RuntimeError(f"{geojson_path} lacks OGC_FID — farm identity unresolved")
    if "Farm" not in gdf.columns:
        raise RuntimeError(f"{geojson_path} lacks Farm column")
    farms = []
    for _, row in gdf.iterrows():
        ogc_fid = int(row["OGC_FID"])
        farms.append({
            "farm_id": str(row["Farm"]),
            "reference_fid": ogc_fid,
            # Verified alias: the archive stores physical field k under id k+1.
            "archive_fid": ogc_fid + 1,
            "geometry": row.geometry,
            "source_attributes": {
                k: (None if isinstance(v, (list, dict)) else v)
                for k, v in row.items() if k not in ("geometry",)
            },
        })
    if limit is not None:
        farms = farms[:limit]
    return farms


def collect_membership(
    pixels_dir: Path, farms: list[dict[str, Any]], dates: list[str],
) -> dict[str, dict[str, Any]]:
    """One pass over ``pixels/<date>.parquet`` collecting per-farm membership.

    Returns per farm: per-date (rows, cols) int arrays (absolute grid coords),
    the farm window (row/col min/max over all dates) and dates absent from the
    pixel archive. Membership rows are exactly the pixels the canonical stats
    pipeline used (QA-valid ∩ archive field polygon).
    """
    archive_fids = {f["archive_fid"]: f["farm_id"] for f in farms}
    per_farm: dict[str, dict[str, Any]] = {
        f["farm_id"]: {"by_date": {}, "absent_dates": [], "row_min": None,
                       "row_max": None, "col_min": None, "col_max": None,
                       "n_member_rows": 0}
        for f in farms
    }
    from tqdm import tqdm
    for date in tqdm(dates, desc="membership scan", unit="date"):
        path = pixels_dir / f"{date}.parquet"
        if not path.exists():
            for fid_map in per_farm.values():
                fid_map["absent_dates"].append(date)
            continue
        sub = pd_read_fid_rows(path)
        hit = set()
        for fid, grp in sub.groupby("field_id"):
            if int(fid) not in archive_fids:
                continue
            farm_id = archive_fids[int(fid)]
            rows = grp["row"].to_numpy(dtype=np.int64)
            cols = grp["col"].to_numpy(dtype=np.int64)
            fm = per_farm[farm_id]
            fm["by_date"][date] = (rows, cols)
            fm["row_min"] = int(rows.min()) if fm["row_min"] is None else min(fm["row_min"], int(rows.min()))
            fm["row_max"] = int(rows.max()) if fm["row_max"] is None else max(fm["row_max"], int(rows.max()))
            fm["col_min"] = int(cols.min()) if fm["col_min"] is None else min(fm["col_min"], int(cols.min()))
            fm["col_max"] = int(cols.max()) if fm["col_max"] is None else max(fm["col_max"], int(cols.max()))
            fm["n_member_rows"] += int(len(rows))
            hit.add(farm_id)
    return per_farm


def pd_read_fid_rows(path: Path):
    import pandas as pd
    return pd.read_parquet(path, columns=["field_id", "row", "col"])


def find_scene_dir(date_str: str, scenes_root: Path) -> Path | None:
    ymd = date_str.replace("-", "")
    hits = sorted(scenes_root.glob(f"landsat_{ymd}_*"))
    if not hits:
        return None
    if len(hits) > 1:
        # More than one path/row directory for one date would make the scene
        # choice ambiguous vs the frozen archive; fail loudly instead of
        # guessing (§8: silent fallbacks are defects).
        raise RuntimeError(
            f"{date_str}: {len(hits)} scene dirs {sorted(p.name for p in hits)} — "
            "ambiguous scene choice, cannot guarantee parity with the archive"
        )
    return hits[0]


def canvas_placement(h: int, w: int, canvas: int) -> tuple[int, int]:
    if h > canvas or w > canvas:
        raise RuntimeError(f"Farm window {h}x{w} exceeds canvas {canvas}x{canvas}")
    return (canvas - h) // 2, (canvas - w) // 2


def farm_group_complete(group: zarr.Group, n_dates: int, canvas: int) -> bool:
    try:
        data = group["data"]
        valid = group["pixel_valid"]
        dates = group["dates"]
        in_field = group["in_field"]
    except KeyError:
        return False
    expected = {
        "data": (n_dates, len(BAND_ORDER), canvas, canvas),
        "pixel_valid": (n_dates, canvas, canvas),
        "dates": (n_dates,),
        "in_field": (canvas, canvas),
    }
    for name, arr in (("data", data), ("pixel_valid", valid), ("dates", dates),
                      ("in_field", in_field)):
        if tuple(arr.shape) != expected[name]:
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--farms", type=Path, default=FARMS_LAYER)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--stats-zarr", type=Path, default=STATS_ZARR)
    ap.add_argument("--pixels-dir", type=Path, default=PIXELS_DIR)
    ap.add_argument("--scenes-root", type=Path, default=SCENES_ROOT)
    ap.add_argument("--canvas-size", type=int, default=48)
    ap.add_argument("--smoke", type=int, default=None, metavar="N",
                    help="Process only the first N dates (smoke test)")
    ap.add_argument("--limit-farms", type=int, default=None,
                    help="Use only the first N farms (development probe)")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    t_start = time.monotonic()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.jsonl"
    manifest_path = out / ".manifest.jsonl"

    dates = parse_dates_from_stats_zarr(args.stats_zarr, args.pixels_dir)
    if args.smoke is not None:
        dates = dates[: args.smoke]
    farms = load_subset_farms(args.farms, limit=args.limit_farms)
    if len(farms) != EXPECTED_FARMS and args.limit_farms is None:
        raise RuntimeError(
            f"{args.farms} has {len(farms)} farms, expected {EXPECTED_FARMS}"
        )
    print(f"Farms: {len(farms)} | dates: {len(dates)} | canvas: {args.canvas_size}")

    # Completeness-verified resume: every farm group must exist with correctly
    # shaped arrays; otherwise the store is rebuilt from scratch (the 2026-11-24
    # partial run left empty groups behind a farm-level manifest).
    resume_ok = False
    if args.resume and manifest_path.exists():
        groups = {}
        root = zarr.open_group(str(out), mode="a", zarr_format=ZARR_FORMAT)
        for f in farms:
            if f["farm_id"] not in root.group_keys():
                break
            g = root[f["farm_id"]]
            if not farm_group_complete(g, len(dates), args.canvas_size):
                break
            groups[f["farm_id"]] = g
        else:
            resume_ok = True
        if resume_ok:
            done_dates = set()
            with open(manifest_path, encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("kind") == "date":
                        done_dates.add(rec["date"])
            print(f"Resume: {len(groups)}/{len(farms)} farm groups complete, "
                  f"{len(done_dates)} dates already written")
        else:
            print("Resume requested but store incomplete — rebuilding from scratch")

    mode = "a" if resume_ok else "w"
    root = zarr.open_group(str(out), mode=mode, zarr_format=ZARR_FORMAT)
    todo_dates = dates if not resume_ok else [d for d in dates if d not in done_dates]

    # Pre-create / reset farm groups (only when rebuilding).
    if not resume_ok:
        done_dates = set()
        for f in farms:
            g = root.require_group(f["farm_id"])
            for name in ("data", "pixel_valid", "dates", "in_field"):
                if name in g:
                    del g[name]
            f["_group"] = g
    else:
        for f in farms:
            f["_group"] = groups[f["farm_id"]]

    # Membership pass (also used for in_field + windows).
    per_farm = collect_membership(args.pixels_dir, farms, dates)

    # Initialize farm arrays.
    canvas = args.canvas_size
    for f in farms:
        fm = per_farm[f["farm_id"]]
        if fm["row_min"] is None:
            raise RuntimeError(
                f"{f['farm_id']}: no archive membership rows for any of the "
                f"{len(dates)} dates (archive_fid={f['archive_fid']}) — check the "
                "fid alias before building"
            )
        h = fm["row_max"] - fm["row_min"] + 1
        w = fm["col_max"] - fm["col_min"] + 1
        off_r, off_c = canvas_placement(h, w, canvas)
        f["_h"], f["_w"], f["_off_r"], f["_off_c"] = h, w, off_r, off_c
        g = f["_group"]
        data = g.create_array(
            "data", shape=(len(dates), len(BAND_ORDER), canvas, canvas),
            dtype="float16", chunks=(1, len(BAND_ORDER), canvas, canvas),
            fill_value=np.nan,
        )
        valid = g.create_array(
            "pixel_valid", shape=(len(dates), canvas, canvas),
            dtype="uint8", chunks=(1, canvas, canvas), fill_value=0,
        )
        dates_arr = g.create_array("dates", shape=(len(dates),), dtype="<U10",
                                   fill_value="")
        in_field = np.zeros((canvas, canvas), dtype=np.uint8)
        for d, (rows, cols) in fm["by_date"].items():
            in_field[rows - fm["row_min"] + off_r, cols - fm["col_min"] + off_c] = 1
        in_field_arr = g.create_array("in_field", shape=(canvas, canvas),
                                      dtype="uint8", fill_value=0)
        in_field_arr[:] = in_field
        n_union = int(in_field.sum())
        g.attrs.update({
            "farm_id": f["farm_id"],
            "reference_fid": f["reference_fid"],
            "archive_fid": f["archive_fid"],
            "id_alias": ID_ALIAS_NOTE,
            "window_hw": [h, w],
            "canvas_offset": [off_r, off_c],
            "in_field_union_pixels": n_union,
            "in_field_never_observed_excluded": (
                "in_field = union of canonical per-date membership; archive-mask "
                "pixels QA-invalid on every date are not observable and are excluded"
            ),
            "n_member_rows_total": fm["n_member_rows"],
            "dates_absent_from_pixel_archive": len(fm["absent_dates"]),
            "crs": "EPSG:32639",
            "pixel_size_m": 30.0,
            "canvas": canvas,
            "band_order": BAND_ORDER,
            "units": "reflectance (pre-scaled METRIC scene rasters); lst in Kelvin",
            "quality": "Landsat C2 QA_PIXEL bits via ssl_anomaly.io.landsat.quality_mask",
            "zarr_format": ZARR_FORMAT,
        })
        if not resume_ok:
            dates_arr[:] = np.asarray(dates, dtype="<U10")
        print(f"  {f['farm_id']}: window {h}x{w} at offset ({off_r},{off_c}), "
              f"in_field union {n_union} px")

    # Root attrs (provenance).
    root.attrs.update({
        "change": "farm-subset-anomaly-probe",
        "purpose": "date-centric raw satellite store for the 32-farm CPU probe",
        "canvas": canvas,
        "band_order": BAND_ORDER,
        "n_dates": len(dates),
        "date_source": "stats.zarr:time (days since 2013-03-30; cross-checked against pixels/<date>.parquet)",
        "date_range": [dates[0], dates[-1]],
        "membership_source": str(args.pixels_dir),
        "scenes_root": str(args.scenes_root),
        "farms_layer": str(args.farms),
        "git_commit_sha": get_git_commit_sha(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "zarr_format": ZARR_FORMAT,
    })

    qcfg = QualityCfg()
    n_ok = n_absent_scene = n_missing_bands = 0
    with tqdm(total=len(todo_dates), desc="dates", unit="date") as bar:
        for di_date in todo_dates:
            di = dates.index(di_date)
            scene = find_scene_dir(di_date, args.scenes_root)
            rec = {"kind": "date", "date": di_date, "index": di + 1,
                   "wall_s": round(time.monotonic() - t_start, 1)}
            if scene is None:
                n_absent_scene += 1
                rec["status"] = "absent_scene_dir"
                bar.update(1)
                bar.set_postfix(absent=n_absent_scene)
                _append_jsonl(progress_path, rec)
                _append_jsonl(manifest_path, {"kind": "date", "date": di_date})
                continue
            missing_bands = [b for b, fn in FILE_MAP.items()
                             if not (scene / fn).exists()]
            if missing_bands:
                n_missing_bands += 1
            handle = {}
            for b, fn in FILE_MAP.items():
                p = scene / fn
                handle[b] = rasterio_open(p) if p.exists() else None
            handle["qa"] = (rasterio_open(scene / QA_FILE)
                            if (scene / QA_FILE).exists() else None)
            if handle["qa"] is None:
                n_absent_scene += 1
                rec["status"] = "absent_qa"
                bar.update(1)
                bar.set_postfix(absent=n_absent_scene)
                _append_jsonl(progress_path, rec)
                _append_jsonl(manifest_path, {"kind": "date", "date": di_date})
                _close(handle)
                continue
            n_written = 0
            for f in farms:
                fm = per_farm[f["farm_id"]]
                if di_date not in fm["by_date"]:
                    continue  # no membership on this date (absent pixels parquet)
                r0, c0 = fm["row_min"], fm["col_min"]
                h, w = f["_h"], f["_w"]
                off_r, off_c = f["_off_r"], f["_off_c"]
                g = f["_group"]
                g["dates"][di] = di_date
                qa = handle["qa"].read(1, window=rasterio_window(c0, r0, w, h))
                if qa.shape != (h, w):
                    raise RuntimeError(
                        f"{f['farm_id']} {di_date}: qa window {qa.shape} != {(h, w)}"
                    )
                qmask = quality_mask(qa, qcfg)
                g["pixel_valid"][di, off_r:off_r + h, off_c:off_c + w] = (
                    qmask.astype(np.uint8))
                for bi, band in enumerate(BAND_ORDER):
                    src = handle[band]
                    if src is None:
                        continue
                    arr = src.read(1, window=rasterio_window(c0, r0, w, h))
                    if arr.shape != (h, w):
                        raise RuntimeError(
                            f"{f['farm_id']} {di_date} {band}: window {arr.shape} "
                            f"!= {(h, w)}"
                        )
                    a32 = arr.astype(np.float32)
                    a32[~np.isfinite(a32)] = np.nan
                    g["data"][di, bi, off_r:off_r + h, off_c:off_c + w] = (
                        a16(a32))
                n_written += 1
            _close(handle)
            n_ok += 1
            rec.update(status="ok", scene=scene.name,
                       farms_written=n_written,
                       missing_bands=missing_bands)
            bar.update(1)
            bar.set_postfix(ok=n_ok, absent=n_absent_scene)
            _append_jsonl(progress_path, rec)
            _append_jsonl(manifest_path, {"kind": "date", "date": di_date})

    # Finalization attrs per farm (explicit absence bookkeeping).
    counts = {}
    for f in farms:
        fm = per_farm[f["farm_id"]]
        dates_with_membership = len(fm["by_date"])
        g = f["_group"]
        g.attrs["n_dates"] = len(dates)
        g.attrs["dates_with_membership_n"] = dates_with_membership
        g.attrs["dates_without_membership_n"] = len(dates) - dates_with_membership
        counts[f["farm_id"]] = dates_with_membership

    wall = time.monotonic() - t_start
    summary = {
        "event": "SUBSET EXTRACTION COMPLETE",
        "wall_s": round(wall, 1),
        "out": str(out),
        "farms": len(farms),
        "dates_total": len(dates),
        "dates_scene_ok": n_ok,
        "dates_absent_scene_or_qa": n_absent_scene,
        "dates_missing_bands": n_missing_bands,
        "per_farm_dates_with_membership": counts,
        "git_commit_sha": get_git_commit_sha(),
    }
    print(json.dumps(summary))
    _append_jsonl(progress_path, {"kind": "summary", **summary})
    return 0


def a16(a: np.ndarray) -> np.ndarray:
    return a.astype(np.float16)


def rasterio_window(c0: int, r0: int, w: int, h: int):
    from rasterio.windows import Window
    return Window(c0, r0, w, h)


def rasterio_open(path: Path):
    import rasterio
    return rasterio.open(path)


def _close(handle: dict) -> None:
    for src in handle.values():
        if src is not None:
            src.close()


def _append_jsonl(path: Path, rec: dict) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    sys.exit(main())

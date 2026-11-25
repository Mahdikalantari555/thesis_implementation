"""Align independent METRIC indicators by (farm, date) for the mini dataset.

Indicators: ETa, ETrF, CWSI (ET and LST variants), TVDI, LAI, METRIC NDVI/LST.
All values are read from real per-scene METRIC GeoTIFFs under
``Data/Metric/products/``. They are **independent evaluation sources**, never
ground truth, and never enter a training DataLoader, threshold calibration or
model selection. Shared-input indicators (those derived from bands the model
also sees) are flagged in the manifest.

Usage::

    python scripts/build_mini_validation.py
    python scripts/build_mini_validation.py --min-dates 5 --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import zarr
from affine import Affine
from rasterio.windows import Window
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402
from ssl_anomaly.utils.paths import localize  # noqa: E402

DATA = localize("D:/RS/Thesis/Data")
PRODUCTS = DATA / "Metric" / "products"
MINI_ROOT = DATA / "Processed" / "mini_research_dataset"
STORE = MINI_ROOT / "satellite.zarr"
INVENTORY = DATA / "Processed" / "satellite" / "scene_inventory.parquet"

INDICATORS = {
    "eta": ("ETaDaily", "METRIC model output (mm/day)", False),
    "etrf": ("ETrF", "METRIC model output (fraction)", False),
    "cwsi_et": ("CWSI_ET", "METRIC proxy from ET/LST (unitless)", True),
    "cwsi_lst": ("CWSI_LST", "METRIC proxy from LST (unitless)", True),
    "tvdi": ("TVDI", "METRIC proxy from LST/NDVI (unitless)", True),
    "lai": ("LAI", "METRIC-derived proxy (m2/m2)", False),
    "ndvi_metric": ("NDVI", "Derived from surface reflectance", True),
    "lst_metric": ("LST", "Derived from thermal band (K)", True),
}


def mini_dates(limit: int | None = None) -> list[str]:
    df = pd.read_parquet(INVENTORY)
    dates = sorted(str(d)[:10] for d in df["acquisition_date"].astype(str))
    return dates[:limit] if limit else dates


def product_index(indicator_dir: str) -> dict[str, Path]:
    """Map date string -> product raster for one METRIC indicator."""
    root = PRODUCTS / indicator_dir
    index: dict[str, Path] = {}
    for path in root.glob(f"{indicator_dir}_*_AOI.tif"):
        stem = path.stem
        parts = stem.split("_")
        date = next((p for p in parts if len(p) == 10 and p[4] == "-"), None)
        if date is None:
            continue
        if date in index:
            raise RuntimeError(f"Multiple {indicator_dir} products for {date}")
        index[date] = path
    return index


def read_farm_value(path: Path, bbox: tuple[float, float, float, float]) -> tuple[float, int]:
    """Read a window covering a farm grid bbox; return median of finite pixels."""
    with rasterio.open(path) as src:
        t = src.transform
        if t.b != 0 or t.d != 0:
            raise RuntimeError(f"Rotated product grid {path}: {t}")
        col = int(np.floor((bbox[0] - float(t.c)) / float(t.a)))
        row = int(np.floor((float(t.f) - bbox[3]) / abs(float(t.e))))
        width = int(np.ceil((bbox[2] - bbox[0]) / float(t.a))) + 1
        height = int(np.ceil((bbox[3] - bbox[1]) / abs(float(t.e)))) + 1
        col = max(col, 0)
        row = max(row, 0)
        width = min(width, src.width - col)
        height = min(height, src.height - row)
        if width <= 0 or height <= 0:
            return float("nan"), 0
        arr = src.read(1, window=Window(col, row, width, height)).astype(np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), 0
    return float(np.median(arr)), int(arr.size)


def build_validation(dates: list[str], out_dir: Path) -> dict[str, Any]:
    t0 = time.monotonic()
    # The mini store keeps one zarr group per farm directory (no root group).
    farms = []
    for group_dir in sorted(p for p in STORE.iterdir() if p.is_dir() and (p / ".zgroup").exists()):
        g = zarr.open_group(str(group_dir), mode="r")
        farms.append(
            {
                "farm_id": str(g.attrs["farm_id"]),
                "ogc_fid": int(g.attrs["ogc_fid"]),
                "bbox": tuple(float(v) for v in g.attrs["bbox"]),
            }
        )

    indexes = {name: product_index(meta[0]) for name, meta in INDICATORS.items()}
    rows: list[dict[str, Any]] = []
    n_attempts = 0
    for date in tqdm(dates, desc="validation dates", unit="date"):
        for farm in farms:
            for name, (dirname, provenance, shared) in INDICATORS.items():
                path = indexes[name].get(date)
                if path is None:
                    rows.append(
                        {
                            "farm_id": farm["farm_id"],
                            "ogc_fid": farm["ogc_fid"],
                            "date": date,
                            "indicator": name,
                            "value": np.nan,
                            "n_pixels": 0,
                            "source_path": None,
                            "status": "absent_product",
                            "provenance": provenance,
                            "shared_input": shared,
                        }
                    )
                    continue
                n_attempts += 1
                value, n = read_farm_value(path, farm["bbox"])
                rows.append(
                    {
                        "farm_id": farm["farm_id"],
                        "ogc_fid": farm["ogc_fid"],
                        "date": date,
                        "indicator": name,
                        "value": value,
                        "n_pixels": n,
                        "source_path": str(path),
                        "status": "ok" if np.isfinite(value) else "absent_no_finite_pixels",
                        "provenance": provenance,
                        "shared_input": shared,
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table_path = out_dir / "validation_indicators.parquet"
    table.to_parquet(table_path, index=False)

    coverage = (
        table[table["status"] == "ok"]
        .groupby("indicator")
        .agg(n_ok=("value", "size"), n_finite=("value", "count"))
        .reset_index()
        .to_dict("records")
    )
    manifest = {
        "schema": "mini-validation/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_commit_sha(),
        "role": "independent evaluation sources only; NOT ground truth; excluded from training/calibration/selection",
        "n_dates": len(dates),
        "n_farms": len(farms),
        "n_rows": int(len(table)),
        "n_raster_reads": n_attempts,
        "indicators": {
            name: {
                "product_dir": str(PRODUCTS / meta[0]),
                "provenance": meta[1],
                "shared_input_with_model": bool(meta[2]),
                "product_sha256_sample": _sha256(next(iter(indexes[name].values())))
                if indexes[name]
                else None,
            }
            for name, meta in INDICATORS.items()
        },
        "coverage": coverage,
        "wall_s": round(time.monotonic() - t0, 2),
        "table": str(table_path),
    }
    manifest_path = out_dir / "validation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    summary = {
        "n_rows": len(table),
        "n_ok": int((table["status"] == "ok").sum()),
        "coverage": coverage,
        "wall_s": manifest["wall_s"],
    }
    print("VALIDATION COMPLETE " + json.dumps(summary), flush=True)
    return manifest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-dates", type=int, default=None, help="Smoke: first N dates")
    ap.add_argument("--out", type=Path, default=MINI_ROOT / "validation")
    args = ap.parse_args(argv)
    dates = mini_dates(args.min_dates)
    print(f"Mini validation: {len(dates)} dates, {len(INDICATORS)} indicators", flush=True)
    build_validation(dates, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

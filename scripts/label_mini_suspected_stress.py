"""Build graded SUSPECTED-stress labels on the real mini pixel grid.

Reads per-scene METRIC product GeoTIFFs, samples them at the exact mini-store
pixel grid, and applies a stage-conditioned, train-years-only rule. Output is
**weak suspicion**, never confirmed stress, never water-vs-salinity typing.

Rule (see thesis_research/SSL_Proposal/02_representation_first_recovery_plan.md):
  cwsi_flag  = CWSI_ET >= threshold(farm kind)
  et_flag    = (median_train(ETa | farm,stage) - ETa) / MAD_train >= 2
  suspected  = cwsi_flag AND et_flag AND persists >= 2 acquisitions
  watch      = exactly one of the two
  control    = neither
Every pixel is gated by pixel_valid AND in_field; masked dates stay absent.

Usage::

    python scripts/label_mini_suspected_stress.py
    python scripts/label_mini_suspected_stress.py --min-dates 20 --out .tmp/labels
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import zarr
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402

DATA = Path("D:/RS/Thesis/Data")
PRODUCTS = DATA / "Metric" / "products"
MINI_ROOT = DATA / "Processed" / "mini_research_dataset"
STORE = MINI_ROOT / "satellite.zarr"
CYCLES = MINI_ROOT / "phenology" / "cycles_mini.parquet"
DEFAULT_OUT = MINI_ROOT / "suspected_stress"

INDICATORS = ("CWSI_ET", "ETaDaily", "ETrF", "LAI", "TVDI")
STAGE_EDGES = (0.0, 0.25, 0.60, 0.85, 1.0001)
STAGE_NAMES = ("S1", "S2", "S3", "S4")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def product_index(name: str) -> dict[str, Path]:
    root = PRODUCTS / name
    out: dict[str, Path] = {}
    for p in root.glob(f"{name}_*_AOI.tif"):
        parts = p.stem.split("_")
        date = next((s for s in parts if len(s) == 10 and s[4] == "-"), None)
        if date:
            out[date] = p
    return out


def stage_of(date: str, cycles: pd.DataFrame) -> str | None:
    for _, c in cycles.iterrows():
        if str(c["start_date"]) <= date <= str(c["end_date"]):
            start = np.datetime64(str(c["start_date"]), "D")
            end = np.datetime64(str(c["end_date"]), "D")
            progress = float((np.datetime64(date, "D") - start).astype(int)) / max(
                1, int((end - start).astype(int))
            )
            for i in range(4):
                if STAGE_EDGES[i] <= progress < STAGE_EDGES[i + 1]:
                    return STAGE_NAMES[i]
    return None


AGE_KEYS = ("Age", "age", "AGE", "Age_Class", "age_class", "سن")


def crop_kind(attrs: dict[str, Any]) -> tuple[str, str]:
    """Plant vs ratoon from the farm's age class ("P" = plant, "Rnn" = ratoon).

    Falls back to "ratoon" (the majority class among cultivated farms) and says
    so, instead of keying on a farm name.
    """
    for key in AGE_KEYS:
        val = attrs.get(key)
        if isinstance(val, str) and val.strip():
            v = val.strip().upper()
            if v.startswith("P"):
                return "plant", f"attr:{key}={val}"
            if v.startswith("R"):
                return "ratoon", f"attr:{key}={val}"
    return "ratoon", "fallback:no_age_attribute"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--store", type=Path, default=STORE)
    ap.add_argument("--train-year-end", type=str, default="2023-01-01")
    ap.add_argument("--mad-k", type=float, default=2.0)
    ap.add_argument("--cwsi-plant", type=float, default=0.40)
    ap.add_argument("--cwsi-ratoon", type=float, default=0.35)
    ap.add_argument("--min-dates", type=int, default=None)
    args = ap.parse_args(argv)

    indexes = {name: product_index(name) for name in INDICATORS}
    cycles = pd.read_parquet(CYCLES)
    rows: list[dict[str, Any]] = []
    product_hashes: dict[str, str] = {}

    for d in sorted(p for p in args.store.iterdir() if p.is_dir() and (p / ".zgroup").exists()):
        g = zarr.open_group(str(d), mode="r")
        farm = str(g.attrs["farm_id"])
        fid = int(g.attrs["ogc_fid"])
        kind, kind_source = crop_kind(dict(g.attrs.get("source_attributes", {}) or {}))
        threshold = args.cwsi_plant if kind == "plant" else args.cwsi_ratoon
        dates = [str(x) for x in g["dates"][:]]
        if args.min_dates:
            dates = dates[: args.min_dates]
        valid = np.asarray(g["pixel_valid"][:], dtype=bool)
        in_field = np.asarray(g["in_field"][:], dtype=bool)
        t = g.attrs["grid_transform"]
        # Affine list order is (a, b, c, d, e, f); the translation is c and f.
        col0 = int(round((float(t[2]) - 233010.0) / 30.0))
        row0 = int(round((3443010.0 - float(t[5])) / 30.0))
        h, w = in_field.shape
        farm_rows: list[dict[str, Any]] = []
        for di, date in enumerate(tqdm(dates, desc=f"{farm} suspect", unit="date")):
            stage = stage_of(date, cycles)
            if stage is None:
                continue
            present = {name: indexes[name].get(date) for name in INDICATORS}
            if present["CWSI_ET"] is None or present["ETaDaily"] is None:
                continue
            arrays: dict[str, np.ndarray] = {}
            for name, path in present.items():
                if path is None:
                    continue
                if str(path) not in product_hashes:
                    product_hashes[str(path)] = _sha256(path)
                with rasterio.open(path) as src:
                    arrays[name] = src.read(
                        1, window=rasterio.windows.Window(col0, row0, w, h)
                    ).astype(np.float64)
            mask = valid[di] & in_field
            cwsi = arrays["CWSI_ET"]
            eta = arrays["ETaDaily"]
            cwsi_ok = mask & np.isfinite(cwsi)
            eta_ok = mask & np.isfinite(eta)
            farm_rows.append(
                {
                    "farm": farm,
                    "ogc_fid": fid,
                    "date": date,
                    "stage": stage,
                    "kind": kind,
                    "kind_source": kind_source,
                    "row_base": row0,
                    "col_base": col0,
                    "mask": mask,
                    "cwsi_ok": cwsi_ok,
                    "eta_ok": eta_ok,
                    "cwsi": np.where(cwsi_ok, cwsi, np.nan),
                    "eta": np.where(eta_ok, eta, np.nan),
                    "etrf": np.where(mask & np.isfinite(arrays.get("ETrF", np.full_like(cwsi, np.nan))),
                                     arrays.get("ETrF", np.full_like(cwsi, np.nan)), np.nan),
                    "lai": np.where(mask & np.isfinite(arrays.get("LAI", np.full_like(cwsi, np.nan))),
                                    arrays.get("LAI", np.full_like(cwsi, np.nan)), np.nan),
                }
            )

        # Train-only baselines by stage.
        baseline: dict[str, dict[str, float]] = {}
        for stage in STAGE_NAMES:
            train_eta = np.concatenate(
                [r["eta"][r["eta_ok"]] for r in farm_rows if r["stage"] == stage and r["date"] < args.train_year_end]
            ) if any(r["stage"] == stage and r["date"] < args.train_year_end for r in farm_rows) else np.array([])
            if train_eta.size >= 10:
                med = float(np.median(train_eta))
                mad = float(np.median(np.abs(train_eta - med)))
                baseline[stage] = {"eta_median": med, "eta_mad": max(mad, 1e-6), "n": int(train_eta.size)}
            else:
                baseline[stage] = {"eta_median": float("nan"), "eta_mad": float("nan"), "n": int(train_eta.size)}

        prev_flag: dict[tuple[int, int], int] = defaultdict(int)
        for r in farm_rows:
            stage = r["stage"]
            base = baseline[stage]
            deficit_z = np.full_like(r["eta"], np.nan)
            if np.isfinite(base["eta_median"]) and base["eta_mad"] > 0:
                deficit_z = (base["eta_median"] - r["eta"]) / base["eta_mad"]
            cwsi_flag = r["cwsi_ok"] & (r["cwsi"] >= threshold)
            et_flag = r["eta_ok"] & (deficit_z >= args.mad_k)
            suspected = np.zeros_like(r["mask"], dtype=bool)
            watch = np.zeros_like(r["mask"], dtype=bool)
            for rr in range(h):
                for cc in range(w):
                    if not r["mask"][rr, cc]:
                        continue
                    c, e = bool(cwsi_flag[rr, cc]), bool(et_flag[rr, cc])
                    both = c and e
                    run = prev_flag[(rr, cc)] + 1 if both else 0
                    prev_flag[(rr, cc)] = run
                    if both and run >= 2:
                        suspected[rr, cc] = True
                    elif c ^ e:
                        watch[rr, cc] = True
            for rr in range(h):
                for cc in range(w):
                    if not r["mask"][rr, cc]:
                        continue
                    label = "suspected" if suspected[rr, cc] else ("watch" if watch[rr, cc] else "control")
                    reason = (
                        "cwsi_and_et_persistent" if label == "suspected"
                        else "cwsi_only" if bool(cwsi_flag[rr, cc])
                        else "et_only" if bool(et_flag[rr, cc])
                        else "none"
                    )
                    rows.append(
                        {
                            "farm": r["farm"], "ogc_fid": r["ogc_fid"], "date": r["date"],
                            "stage": stage, "row": row0 + rr, "col": col0 + cc,
                            "x": 233010.0 + (col0 + cc) * 30.0,
                            "y": 3443010.0 - (row0 + rr) * 30.0,
                            "cwsi_et": float(r["cwsi"][rr, cc]),
                            "eta_daily": float(r["eta"][rr, cc]),
                            "eta_deficit_z": float(deficit_z[rr, cc]) if np.isfinite(deficit_z[rr, cc]) else np.nan,
                            "etrf": float(r["etrf"][rr, cc]) if np.isfinite(r["etrf"][rr, cc]) else np.nan,
                            "lai": float(r["lai"][rr, cc]) if np.isfinite(r["lai"][rr, cc]) else np.nan,
                            "label": label, "reason": reason,
                        }
                    )
        print(
            f"  {farm}: {len(farm_rows)} dates, baselines "
            + json.dumps({k: round(v["n"]) for k, v in baseline.items()}),
            flush=True,
        )

    table = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    pixels_path = args.out / "suspected_pixels.parquet"
    table.to_parquet(pixels_path, index=False)

    patch = (
        table.assign(patch_row=table["row"] // 5, patch_col=table["col"] // 5)
        .groupby(["farm", "date", "stage", "patch_row", "patch_col"])
        .agg(
            n_valid=("label", "size"),
            n_suspected=("label", lambda s: int((s == "suspected").sum())),
            n_watch=("label", lambda s: int((s == "watch").sum())),
        )
        .reset_index()
    )
    patch["label"] = np.where(patch["n_suspected"] > 0, "suspected", np.where(patch["n_watch"] > 0, "watch", "control"))
    patch_path = args.out / "suspected_patches.parquet"
    patch.to_parquet(patch_path, index=False)

    manifest = {
        "schema": "mini-suspected-stress/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_commit_sha(),
        "role": "WEAK suspicion from physical products only; NOT ground truth; NOT stress typing",
        "rule": "CWSI_ET >= farm-kind threshold AND ETa deficit >= k*MAD (train years only) AND >=2 consecutive acquisitions",
        "cwsi_thresholds": {"plant": args.cwsi_plant, "ratoon": args.cwsi_ratoon},
        "mad_k": args.mad_k,
        "train_year_end": args.train_year_end,
        "n_rows": int(len(table)),
        "n_patches": int(len(patch)),
        "label_counts": table["label"].value_counts().to_dict(),
        "patch_label_counts": patch["label"].value_counts().to_dict(),
        "product_sha256": product_hashes,
        "caveats": [
            "CWSI is equifinal (water and salinity share the thermal signature); no stress typing.",
            "CWSI_ET/TVDI share LST/NIR inputs with an SSL model consuming those bands; proxy only.",
            "ETaDaily is a METRIC model product, not a measurement; agreement, not validation.",
            "kind comes from the farm's current age-class attribute (P = plant, Rnn = ratoon; fallback ratoon). It is a snapshot applied to all years, so it is still an approximation because ratoon_no is absent from cycles_mini.",
        ],
        "pixels": str(pixels_path),
        "patches": str(patch_path),
    }
    (args.out / "suspected_stress_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "SUSPECTED LABELS COMPLETE "
        + json.dumps({"n": len(table), "labels": manifest["label_counts"], "patches": manifest["patch_label_counts"]}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

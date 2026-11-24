"""Report valid-patch counts per (farm, date, patch size) for the mini store.

Counts use the stored `pixel_valid` and `in_field` masks only. A patch counts
as valid when it overlaps in-field pixels and at least ``--min-fraction`` of
its in-field pixels are valid on that date. No values are synthesized.

Usage::

    python scripts/report_mini_patch_counts.py --out reports/mini_dataset/valid_patch_counts.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
DATA = Path("D:/RS/Thesis/Data/Processed/mini_research_dataset")
DEFAULT_OUT = ROOT / "reports" / "mini_dataset" / "valid_patch_counts.json"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", type=Path, default=DATA / "satellite.zarr")
    ap.add_argument("--sizes", nargs="+", type=int, default=[3, 5, 7])
    ap.add_argument("--min-fraction", type=float, default=0.5)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    report: dict[str, object] = {
        "schema": "mini-valid-patch-counts/v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "store": str(args.store),
        "min_fraction": args.min_fraction,
        "rule": (
            "patch overlaps in_field and >= min_fraction of its in-field pixels "
            "are pixel_valid on that date"
        ),
        "farms": {},
    }
    store = args.store
    for d in sorted(p for p in store.iterdir() if p.is_dir() and (p / ".zgroup").exists()):
        g = zarr.open_group(str(d), mode="r")
        dates = [str(x) for x in g["dates"][:]]
        valid = np.asarray(g["pixel_valid"][:], dtype=bool)
        in_field = np.asarray(g["in_field"][:], dtype=bool)
        farm: dict[str, object] = {"n_dates": len(dates), "sizes": {}}
        for size in args.sizes:
            from numpy.lib.stride_tricks import sliding_window_view

            imask = sliding_window_view(in_field, (size, size))
            vmask = sliding_window_view(valid, (size, size), axis=(1, 2))
            denom = imask.sum(axis=(-1, -2)).astype(np.float32)
            num = (vmask & imask).sum(axis=(-1, -2)).astype(np.float32)
            frac = np.where(denom > 0, num / np.maximum(denom, 1.0), 0.0)
            counts = ((frac >= args.min_fraction) & (denom > 0)).sum(axis=(1, 2))
            farm["sizes"][str(size)] = {
                "per_date": {date: int(counts[i]) for i, date in enumerate(dates)},
                "min": int(counts.min()),
                "max": int(counts.max()),
                "mean": round(float(counts.mean()), 3),
            }
        report["farms"][str(g.attrs["farm_id"])] = farm
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PATCH COUNTS COMPLETE out={args.out} farms={len(report['farms'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

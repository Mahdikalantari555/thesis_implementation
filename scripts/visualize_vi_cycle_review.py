"""Generate NDVI+SAVI cycle-review plots with a legend and VI-progress states.

Screening evidence only. Boundaries are provisional and unresolved intervals
are shown, never silently dropped. States S1–S4 are normalized within-cycle
progress bands derived from the VI trajectory, NOT precisely extracted phenology
dates; no stage or stress claim follows from them before owner review.

Usage::

    python scripts/visualize_vi_cycle_review.py --out reports/vi_cycle_review
    python scripts/visualize_vi_cycle_review.py --mini-store <zarr> --mini-cycles <parquet> --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from ssl_anomaly.io.stats_zarr import load_stats_zarr  # noqa: E402
from ssl_anomaly.phenology.mini import (  # noqa: E402
    build_cycle_rows,
    detect_harvests,
    vi_harvest_score,
)

DEFAULT_FIELDS = (12, 87, 164, 241, 318, 455)  # stratified over the 1–490 id range
DEFAULT_ZARR = Path("/home/mahdi/RS/Thesis/Data/Processed/satellite/stats.zarr")
STATE_BANDS = (
    ("S1 establishment", 0.00, 0.25, "#7fc3e8"),
    ("S2 peak growth", 0.25, 0.60, "#66c2a5"),
    ("S3 maturity/senescence", 0.60, 0.85, "#fea95c"),
    ("S4 harvest/dormancy", 0.85, 1.00, "#b39ddb"),
)
LEGEND = [
    Line2D([0], [0], color="#2b7bba", lw=1.2, label="NDVI (median, smoothed)"),
    Line2D([0], [0], color="#4ba85c", lw=1.2, label="SAVI (median, smoothed)"),
    Patch(facecolor="#7fc3e8", label="S1 establishment"),
    Patch(facecolor="#66c2a5", label="S2 peak growth"),
    Patch(facecolor="#fea95c", label="S3 maturity/senescence"),
    Patch(facecolor="#b39ddb", label="S4 harvest/dormancy"),
    Patch(facecolor="#e08a8a", alpha=0.45, label="overlong_unresolved (>540 d) — excluded"),
    Patch(facecolor="#e8c96a", alpha=0.45, label="short_unresolved (<180 d) — excluded"),
    Patch(facecolor="#b9b9b9", alpha=0.30, label="rest: harvest → greenup (not in cycle)"),
    Line2D([0], [0], color="#1f6f3d", ls="--", lw=1, label="strong harvest boundary"),
    Line2D([0], [0], color="#d98a1f", ls=":", lw=1.2, label="weak candidate (review only)"),
]


def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    s = pd.Series(np.asarray(y, dtype=np.float64)).interpolate(limit_direction="both")
    return s.rolling(window=window, center=True, min_periods=1).median().to_numpy()


def _plot(
    label: str,
    dates: pd.DatetimeIndex,
    date_strs: list[str],
    day_numbers: np.ndarray,
    ndvi: np.ndarray,
    savi: np.ndarray,
    strong: list,
    weak: list,
    rows: list[dict],
    out: Path,
) -> dict:
    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.plot(dates, ndvi, lw=1.0, color="#2b7bba")
    ax.plot(dates, savi, lw=1.0, color="#4ba85c")

    t0, t1 = pd.Timestamp(date_strs[0]), pd.Timestamp(date_strs[-1])
    for r in rows:
        a, b = pd.Timestamp(r["start_date"]), pd.Timestamp(r["end_date"])
        bs = r.get("boundary_start_index", -1)
        if bs is not None and bs >= 0 and r.get("rest_days", 0) > 0:
            ax.axvspan(pd.Timestamp(date_strs[bs]), a, color="#b9b9b9", alpha=0.30, lw=0)
            if r["rest_days"] >= 60:
                ax.text(pd.Timestamp(date_strs[bs]) + (a - pd.Timestamp(date_strs[bs])) / 2, 0.98,
                        f"rest {r['rest_days']}d", ha="center", va="top",
                        fontsize=6.5, color="#666")
        if r["status"] != "normal":
            color = "#e08a8a" if r["status"] == "overlong_unresolved" else "#e8c96a"
            ax.axvspan(a, b, color=color, alpha=0.35, lw=0)
            ax.text(a + (b - a) / 2, 0.93, r["status"].replace("_unresolved", ""),
                    ha="center", va="center", fontsize=7, color="#8a3030")
            continue
        # VI-progress states: a *progress* band, not an extracted stage date.
        length = max(1.0, float((b - a).days))
        for name, lo, hi, color in STATE_BANDS:
            sa, sb = a + pd.Timedelta(days=length * lo), a + pd.Timedelta(days=length * hi)
            ax.axvspan(sa, sb, color=color, alpha=0.55, lw=0)
            if (sb - sa).days >= 30:
                ax.text(sa + (sb - sa) / 2, 0.015, name,
                        ha="center", va="bottom", fontsize=7.5,
                        fontweight="bold", color="#222",
                        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=1))

    for c in strong:
        ax.axvline(dates[c.index], color="#1f6f3d", ls="--", lw=0.9)
    for c in weak:
        ax.axvline(dates[c.index], color="#d98a1f", ls=":", lw=1.1)

    ax.set_title(
        f"{label} — NDVI/SAVI harvest screening | strong={len(strong)}, "
        f"weak={len(weak)}, unresolved="
        f"{sum(r['status'] != 'normal' for r in rows)}"
    )
    ax.set_xlim(t0, t1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("index value")
    ax.grid(alpha=0.2)
    ax.legend(handles=LEGEND, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=3, fontsize=7.5, frameon=False)
    fig.text(
        0.5, -0.055,
        "A cycle starts at greenup, not at the previous harvest; gray = rest between harvest and greenup. "
        "States S1–S4 are normalized within-cycle VI-progress bands (0–0.25, 0.25–0.60, 0.60–0.85, 0.85–1.00), "
        "not extracted stage dates. Unresolved spans are excluded from splits. Screening evidence only.",
        ha="center", fontsize=7, color="#555",
    )
    fig.tight_layout()
    png = out / f"{label.replace(' ', '_')}.png"
    fig.savefig(png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return {
        "strong": [date_strs[c.index] for c in strong],
        "weak": [
            {"date": date_strs[c.index], "score": round(c.score, 3),
             "prominence": c.prominence, "support": c.support}
            for c in weak
        ],
        "unresolved": [
            {"status": r["status"], "start_date": r["start_date"],
             "end_date": r["end_date"], "length_days": r["length_days"]}
            for r in rows if r["status"] != "normal"
        ],
        "normal": [
            {"start_date": r["start_date"], "end_date": r["end_date"],
             "length_days": r["length_days"]}
            for r in rows if r["status"] == "normal"
        ],
        "plot": str(png),
    }


def _detect(label: str, date_strs: list[str], dates: pd.DatetimeIndex,
            ndvi: np.ndarray, savi: np.ndarray) -> tuple[list, list, list[dict]]:
    score, oriented = vi_harvest_score(ndvi, savi)
    day_numbers = (dates - dates[0]).days.to_numpy()
    strong, weak = detect_harvests(score, day_numbers, oriented)
    rows = build_cycle_rows(label, 0, date_strs, strong, ndvi=ndvi, savi=savi)
    return strong, weak, rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", type=int, nargs="+", default=list(DEFAULT_FIELDS))
    ap.add_argument("--zarr", type=Path, default=DEFAULT_ZARR)
    ap.add_argument("--mini-store", type=Path, default=None)
    ap.add_argument("--mini-cycles", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "vi_cycle_review")
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema": "vi-cycle-review/v2",
        "channels": ["ndvi", "savi"],
        "states": "normalized within-cycle VI-progress bands; not extracted stage dates",
        "note": "Screening candidates; no stage or stress claim before owner review.",
        "fields": {},
    }

    reader = load_stats_zarr(args.zarr)
    indices = reader.compute_indices("median")
    dates = reader.dates
    date_strs = [str(d)[:10] for d in dates]
    field_index = {int(f): i for i, f in enumerate(reader.fields)}
    for fid in args.fields:
        if fid not in field_index:
            raise SystemExit(f"field {fid} not in stats.zarr")
        i = field_index[fid]
        ndvi, savi = _smooth(indices["ndvi"][i]), _smooth(indices["savi"][i])
        label = f"field_{fid:03d}"
        strong, weak, rows = _detect(label, date_strs, dates, ndvi, savi)
        summary["fields"][label] = _plot(
            label, dates, date_strs, dates, ndvi, savi, strong, weak, rows, args.out
        )

    if args.mini_store is not None:
        cycles = pd.read_parquet(args.mini_cycles) if args.mini_cycles else None
        for group_dir in sorted(
            p for p in args.mini_store.iterdir() if p.is_dir() and (p / ".zgroup").exists()
        ):
            g = zarr.open_group(str(group_dir), mode="r")
            farm = str(g.attrs["farm_id"])
            cube = np.asarray(g["data"][:], dtype=np.float32)
            valid = np.asarray(g["pixel_valid"][:], dtype=bool)
            in_field = np.asarray(g["in_field"][:], dtype=bool)
            bands = {b: cube[:, i, :, :] for i, b in enumerate(g.attrs["band_order"])}
            mdates = [str(d)[:10] for d in g["dates"][:]]
            mdates_idx = pd.DatetimeIndex([pd.Timestamp(d) for d in mdates])
            field = {}
            for name in ("ndvi", "savi"):
                arr = (bands["nir"] - bands["red"]) / (bands["nir"] + bands["red"] + (0.5 if name == "savi" else 0.0))
                if name == "savi":
                    arr = arr * 1.5
                vals = []
                for di in range(len(mdates)):
                    m = valid[di] & in_field & np.isfinite(arr[di])
                    vals.append(float(np.median(arr[di][m])) if m.any() else np.nan)
                field[name] = _smooth(np.asarray(vals))
            ndvi, savi = field["ndvi"], field["savi"]
            if cycles is not None:
                rows = cycles[cycles["farm"] == farm].to_dict("records")
                score, oriented = vi_harvest_score(ndvi, savi)
                days = (mdates_idx - mdates_idx[0]).days.to_numpy()
                strong, weak = detect_harvests(score, days, oriented)
            else:
                strong, weak, rows = _detect(farm, mdates, mdates_idx, ndvi, savi)
            summary["fields"][farm] = _plot(
                farm, mdates_idx, mdates, days if cycles is not None else (mdates_idx - mdates_idx[0]).days.to_numpy(),
                ndvi, savi, strong, weak, rows, args.out
            )

    summary_path = args.out / "review_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "n_plots": len(summary["fields"]), "out": str(args.out),
        "summary": str(summary_path),
        "unresolved": {k: len(v["unresolved"]) for k, v in summary["fields"].items()},
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase-0 diagnosis: how many lost field-years can a multi-index detector recover?

**Read-only study.** The frozen chain (``cycles_490.parquet`` sha 930a6cde…,
``splits_locked.json``, ``holdout_manifest.json``, ``patches.zarr``) is never
written; outputs go only to ``reports/cycle_detection_diagnosis/``.

Background (2026-11-19): the locked NDVI-only detector (``segment_cycles``:
prominence ≥ 0.15, plateau ≥ 0.7 × field-global ymax, rise within 180 d,
cycle length 180–540 d) yields 799 clean cycles over 370 fields / 13 years —
~2.2 per field. Most field-years sit inside spans where a winter harvest was
never observed (winter obs fill-rate 0.39–0.55 vs 0.87–0.94 in summer), so the
seasons merge past ``max_cycle_days`` and are dropped or censored.

This script:
1. Reproduces v1 verbatim (same inputs, same parameters) and verifies the
   reproduction against the locked parquet.
2. Identifies every *lost span* (archive-lead, skipped >540 d gap, trailing).
3. Classifies each span: ``obs_hole`` (< 2 usable season observations),
   ``fusion_recoverable`` (a transparent SAVI/LST/SWIR fusion candidate
   detector fires), or ``unresolved`` (observations exist, no candidate).
4. Counts v2-candidate cycles: field boundaries = v1 harvests ∪ accepted
   fusion candidates; every consecutive pair 180–540 d is a valid cycle.
   The v2−v1 delta is the headline recovery estimate.
5. Reports a threshold-sensitivity grid so the result is not a single
   cherry-picked operating point.

The v1 (NDVI-only) vs v2-candidate (multi-index) comparison is itself a
deliverable; any production detector-v3 that would rebuild
splits/store/holdout is a separate, separately-gated change.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from ssl_anomaly.io.stats_zarr import load_stats_zarr
from ssl_anomaly.phenology.cycles import segment_cycles

SEASON_MONTHS = (11, 12, 1, 2, 3)  # Khuzestan sugarcane harvest window
SPAN_KINDS = ("lead", "gap", "trail")
CLASSES = ("obs_hole", "fusion_recoverable", "unresolved")


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True,
            capture_output=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_v1_locked(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df[["field", "cycle_id", "t_start", "t_end", "censored", "kind", "ratoon_no"]]


def reproduce_v1(zarr_path: str, *, max_days: int, limit: int | None,
                 log_every: int = 50) -> tuple[dict[int, list[dict]], dict[int, list[int]], dict]:
    """Run the identical segment_cycles call the locked table was built with.

    v1 silently skips invalid harvest pairs (no cycle object), so the
    complete detected-harvest list is NOT recoverable from the returned
    windows alone. ``detect_harvests`` is therefore captured through a shim
    (restored after the loop); ``segment_cycles`` itself stays untouched.
    """
    reader = load_stats_zarr(zarr_path)
    dates = reader.dates
    times = dates.to_numpy().astype("datetime64[D]")
    epoch = times[0]
    time_days = (times - epoch).astype(int)
    indices = reader.compute_indices("median")
    ndvi = indices["ndvi"]
    valid_frac = reader.valid_frac
    cube = reader.get_stat("median")
    b_idx = reader.band_index

    fields = [int(f) for f in reader.fields]
    if limit:
        fields = fields[:limit]

    import ssl_anomaly.phenology.cycles as _cyc_mod
    captured: dict[int, list[int]] = {}
    current: list[int] = []
    orig_detect = _cyc_mod.detect_harvests

    def _capture(grid_days: np.ndarray, ndvi_smooth: np.ndarray, **kw):
        idx = orig_detect(grid_days, ndvi_smooth, **kw)
        captured[current[0]] = [int(grid_days[i]) for i in idx]
        return idx

    _cyc_mod.detect_harvests = _capture
    per_field_cycles: dict[int, list[dict]] = {}
    t_start = time.time()
    try:
        for k, fid in enumerate(fields):
            current.clear()
            current.append(fid)
            cyc = segment_cycles(
                field_id=fid,
                times_days=time_days,
                ndvi=ndvi[k],
                quality=valid_frac[k],
                min_cycle_days=180,
                max_cycle_days=max_days,
                min_prominence=0.15,
                plateau_frac=0.7,
                rise_frac=0.15,
            )
            per_field_cycles[fid] = [
                {"t_start": int(c.t_start), "t_end": int(c.t_end),
                 "start_harvest": int(c.start_harvest), "end_harvest": int(c.end_harvest),
                 "censored": bool(c.censored), "kind": c.kind, "ratoon_no": int(c.ratoon_no)}
                for c in cyc
            ]
            if (k + 1) % log_every == 0 or k + 1 == len(fields):
                print(f"[v1-repro] {k + 1}/{len(fields)} fields  "
                      f"({time.time() - t_start:.0f}s elapsed)", flush=True)
    finally:
        _cyc_mod.detect_harvests = orig_detect

    meta = {
        "bands": list(reader.bands),
        "n_fields_total": int(len(reader.fields)),
        "n_dates": int(len(dates)),
        "date_start": str(dates[0].date()),
        "date_end": str(dates[-1].date()),
        "epoch": str(epoch),
        "months": dates.month.to_numpy().tolist(),
        "ndvi": ndvi, "savi": indices["savi"], "lst": indices["lst"],
        "swir1": cube[:, :, b_idx["swir1"]],
        "valid_frac": valid_frac, "time_days": time_days,
        "fields_all": [int(f) for f in reader.fields],
    }
    return per_field_cycles, captured, meta


def lost_spans(harvests: list[int], series_start: int, series_end: int,
               min_cycle_days: int, max_cycle_days: int) -> list[dict]:
    """Lead-in / skipped-gap / trailing spans that v1 could not bound.

    A span between consecutive detected harvests longer than
    ``max_cycle_days`` was skipped entirely by v1 (no cycle object at all).
    """
    if not harvests:
        return []
    spans: list[dict] = []
    if harvests[0] - series_start >= min_cycle_days:
        spans.append({"kind": "lead", "t0": series_start, "t1": harvests[0]})
    for h0, h1 in zip(harvests[:-1], harvests[1:]):
        if h1 - h0 > max_cycle_days:
            spans.append({"kind": "gap", "t0": h0, "t1": h1})
    if series_end - harvests[-1] >= min_cycle_days:
        spans.append({"kind": "trail", "t0": harvests[-1], "t1": series_end})
    return spans


def build_candidate_table(meta: dict, fi: int, *,
                          ctx_days: int, min_gap: int, vf_min: float,
                          edge_guard: int) -> dict[str, np.ndarray]:
    """Per-observation harvest-evidence quantities for one field.

    For every usable observation date t (finite NDVI, valid_frac ≥ vf_min,
    t not at the very start of the series): the drop of NDVI/SAVI, the rise
    of LST/SWIR1 versus the median over the preceding ``ctx_days`` window.
    """
    ndvi = meta["ndvi"][fi]
    savi = meta["savi"][fi]
    lst = meta["lst"][fi]
    swir1 = meta["swir1"][fi]
    vf = meta["valid_frac"][fi]
    t = meta["time_days"]

    rows = []
    for k in range(len(t)):
        if not np.isfinite(ndvi[k]) or vf[k] < vf_min:
            continue
        if t[k] < ctx_days:
            continue
        ctx = (t >= t[k] - ctx_days) & (t < t[k] - min_gap)
        if ctx.sum() < 2:
            continue
        pre_n = ndvi[ctx & np.isfinite(ndvi)]
        pre_s = savi[ctx & np.isfinite(savi)]
        pre_l = lst[ctx & np.isfinite(lst)]
        pre_w = swir1[ctx & np.isfinite(swir1)]
        if pre_n.size < 2 or pre_l.size < 2:
            continue
        rows.append({
            "t": int(t[k]),
            "drop_ndvi": float(np.median(pre_n) - ndvi[k]),
            "drop_savi": float(np.median(pre_s) - savi[k]) if pre_s.size >= 2 else np.nan,
            "lst_rise": float(lst[k] - np.median(pre_l)),
            "swir_rise": float(swir1[k] - np.median(pre_w)) if pre_w.size >= 2 else np.nan,
        })
    return {key: np.array([r[key] for r in rows], dtype=np.float64)
            for key in ("t", "drop_ndvi", "drop_savi", "lst_rise", "swir_rise")}


def greedy_candidates(cand: dict[str, np.ndarray], spans: list[dict],
                      harvests: list[int], *, veg_th: float, lst_th: float,
                      swir_th: float, edge_guard: int) -> list[int]:
    """Accept fusion candidates inside lost spans, ≥ edge_guard d from span edges,
    ≥ 180 d apart (dedup handled by the caller via span ordering)."""
    if cand["t"].size == 0:
        return []
    veg = np.maximum(np.nan_to_num(cand["drop_ndvi"], nan=-9.0),
                     np.nan_to_num(cand["drop_savi"], nan=-9.0))
    lst_ok = np.nan_to_num(cand["lst_rise"], nan=-9.0) >= lst_th
    swir_ok = np.nan_to_num(cand["swir_rise"], nan=-9.0) >= swir_th
    fires = (veg >= veg_th) & (lst_ok | swir_ok)

    accepted: list[int] = []
    for sp in spans:
        inside = np.where(fires & (cand["t"] > sp["t0"] + edge_guard)
                          & (cand["t"] < sp["t1"] - edge_guard))[0]
        for idx in inside:
            tt = int(cand["t"][idx])
            if all(abs(tt - a) >= 180 for a in accepted) and all(abs(tt - h) >= 180 for h in harvests):
                accepted.append(tt)
    return accepted


def count_valid_cycles(boundaries: list[int], *, min_cycle_days: int,
                       max_cycle_days: int) -> int:
    """Consecutive harvest boundaries 180–540 d apart form a valid cycle."""
    b = sorted(set(boundaries))
    return sum(1 for a, c in zip(b[:-1], b[1:]) if min_cycle_days <= c - a <= max_cycle_days)


def classify_span(sp: dict, obs_season: int, n_cand: int) -> str:
    if obs_season < 2:
        return "obs_hole"
    return "fusion_recoverable" if n_cand >= 1 else "unresolved"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stats-zarr", default="Data/Processed/satellite/stats.zarr")
    ap.add_argument("--cycles", default="data/processed/cycles_490.parquet")
    ap.add_argument("--out", default="reports/cycle_detection_diagnosis")
    ap.add_argument("--smoke", type=int, default=0, help="limit to first N fields (validation)")
    ap.add_argument("--ctx-days", type=int, default=120)
    ap.add_argument("--min-gap", type=int, default=5)
    ap.add_argument("--edge-guard", type=int, default=30)
    ap.add_argument("--vf-min", type=float, default=0.5)
    ap.add_argument("--min-cycle-days", type=int, default=180)
    ap.add_argument("--max-cycle-days", type=int, default=540)
    # primary + grid thresholds
    ap.add_argument("--veg-th", type=float, default=0.20)
    ap.add_argument("--lst-th", type=float, default=3.0)
    ap.add_argument("--swir-th", type=float, default=0.05)
    ap.add_argument("--grid-veg", default="0.15,0.20,0.25")
    ap.add_argument("--grid-lst", default="2.0,3.0,5.0")
    ap.add_argument("--grid-swir", default="0.03,0.05")
    args = ap.parse_args()

    out_dir = ROOT_DIR / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_smoke{args.smoke}" if args.smoke else ""
    t_wall0 = time.time()

    print("[1/5] reproducing v1 segmentation (identical call to the locked run)…", flush=True)
    repro, v1_harvests, meta = reproduce_v1(args.stats_zarr, max_days=args.max_cycle_days,
                                            limit=args.smoke or None)
    fields = meta["fields_all"][: len(repro)]
    n_fields = len(fields)
    fi_of = {fid: k for k, fid in enumerate(meta["fields_all"])}
    series_start, series_end = int(meta["time_days"][0]), int(meta["time_days"][-1])

    print("[2/5] verifying v1 reproduction against the locked parquet…", flush=True)
    locked = load_v1_locked(ROOT_DIR / args.cycles)
    locked_fields = set(locked.field.unique())
    n_match = n_miss_repro = 0
    mismatch_examples: list[dict] = []
    for fid, cyc in repro.items():
        if fid not in locked_fields:
            n_miss_repro += len(cyc)
            continue
        lc = locked[locked.field == fid]
        locked_set = {(int(r.t_start), int(r.t_end), bool(r.censored)) for r in lc.itertuples()}
        repro_set = {(c["t_start"], c["t_end"], c["censored"]) for c in cyc}
        n_match += len(locked_set & repro_set)
        for d in sorted(locked_set - repro_set)[:2]:
            mismatch_examples.append({"field": fid, "in_locked": list(d)})
        for d in sorted(repro_set - locked_set)[:2]:
            mismatch_examples.append({"field": fid, "in_repro": list(d)})
    v1_total_clean = sum(1 for cyc in repro.values() for c in cyc if not c["censored"])
    v1_locked_clean = int((~locked.censored.astype(bool)).sum())
    print(f"    reproduced clean cycles: {v1_total_clean} (locked parquet: {v1_locked_clean}); "
          f"window-match {n_match} spans; examples of mismatch: {len(mismatch_examples)}", flush=True)

    print("[3/5] identifying lost spans per field…", flush=True)
    span_rows: list[dict] = []
    for fid in fields:
        harvests = v1_harvests.get(fid, [])
        for sp in lost_spans(harvests, series_start, series_end,
                             args.min_cycle_days, args.max_cycle_days):
            fi = fi_of[fid]
            in_span = (meta["time_days"] >= sp["t0"]) & (meta["time_days"] < sp["t1"])
            season = np.isin(meta["months"], SEASON_MONTHS) & in_span
            usable_season = int((season & np.isfinite(meta["ndvi"][fi])
                                 & (meta["valid_frac"][fi] >= args.vf_min)).sum())
            span_rows.append({"field": fid, "kind": sp["kind"],
                              "t0": sp["t0"], "t1": sp["t1"],
                              "length_days": sp["t1"] - sp["t0"],
                              "n_obs_season_usable": usable_season,
                              "n_harvests_v1": len(harvests)})
    spans_df = pd.DataFrame(span_rows)
    print(f"    {len(spans_df)} lost spans over {n_fields} fields, "
          f"total {spans_df.length_days.sum() / 365.0:.0f} field-years", flush=True)

    print("[4/5] fused-candidate detection (SAVI/LST/SWIR) + threshold grid…", flush=True)
    grid_veg = [float(x) for x in args.grid_veg.split(",")]
    grid_lst = [float(x) for x in args.grid_lst.split(",")]
    grid_swir = [float(x) for x in args.grid_swir.split(",")]
    cand_cache: dict[int, dict[str, np.ndarray]] = {}
    grid_counts = {(v, l, w): 0 for v in grid_veg for l in grid_lst for w in grid_swir}
    v1_recount = 0
    for fid in fields:
        harvests = v1_harvests.get(fid, [])
        if not harvests:
            continue
        v1_recount += count_valid_cycles(harvests, min_cycle_days=args.min_cycle_days,
                                         max_cycle_days=args.max_cycle_days)
        fi = fi_of[fid]
        cand = build_candidate_table(meta, fi, ctx_days=args.ctx_days,
                                     min_gap=args.min_gap, vf_min=args.vf_min,
                                     edge_guard=args.edge_guard)
        cand_cache[fid] = cand
        spans = lost_spans(harvests, series_start, series_end,
                           args.min_cycle_days, args.max_cycle_days)
        for v in grid_veg:
            for l in grid_lst:
                for w in grid_swir:
                    acc = greedy_candidates(cand, spans, harvests, veg_th=v,
                                            lst_th=l, swir_th=w,
                                            edge_guard=args.edge_guard)
                    grid_counts[(v, l, w)] += count_valid_cycles(
                        harvests + acc, min_cycle_days=args.min_cycle_days,
                        max_cycle_days=args.max_cycle_days)
        if (len(cand_cache) % 50) == 0:
            print(f"    candidates {len(cand_cache)}/{n_fields}", flush=True)

    print("[5/5] classifying spans and writing outputs…", flush=True)
    primary_key = (args.veg_th, args.lst_th, args.swir_th)
    v2_clean_primary = grid_counts[primary_key]
    cand_per_span: list[int] = []
    for _, sp in spans_df.iterrows():
        harvests = v1_harvests.get(sp.field, [])
        spans = [{"kind": sp.kind, "t0": int(sp.t0), "t1": int(sp.t1)}]
        acc = greedy_candidates(cand_cache.get(sp.field, {"t": np.array([])}), spans, harvests,
                                veg_th=args.veg_th, lst_th=args.lst_th,
                                swir_th=args.swir_th, edge_guard=args.edge_guard)
        cand_per_span.append(len(acc))
    spans_df["n_candidates_primary"] = cand_per_span
    spans_df["class"] = [classify_span(sp, sp.n_obs_season_usable, c)
                         for sp, c in zip(spans_df.itertuples(), cand_per_span)]

    class_summary = (spans_df.groupby("class")
                     .agg(spans=("length_days", "size"),
                          field_years=("length_days", lambda s: round(s.sum() / 365.0, 1)))
                     .reindex(CLASSES).fillna(0))
    by_kind = (spans_df.groupby("kind")
               .agg(spans=("length_days", "size"),
                    field_years=("length_days", lambda s: round(s.sum() / 365.0, 1))))

    wall = round(time.time() - t_wall0, 1)
    report = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "numpy": np.__version__, "pandas": pd.__version__,
        "device": "cpu", "wall_s": wall, "smoke": bool(args.smoke),
        "n_fields": n_fields,
        "inputs": {
            "stats_zarr": args.stats_zarr, "bands": meta["bands"],
            "n_dates": meta["n_dates"], "date_start": meta["date_start"],
            "date_end": meta["date_end"], "epoch": meta["epoch"],
            "cycles_parquet": args.cycles,
            "cycles_parquet_sha256": sha256_file(ROOT_DIR / args.cycles)[:16],
        },
        "v1_params": {"min_cycle_days": args.min_cycle_days,
                      "max_cycle_days": args.max_cycle_days,
                      "min_prominence": 0.15, "plateau_frac": 0.7, "rise_frac": 0.15},
        "v1_verification": {
            "reproduced_clean_cycles": int(v1_total_clean),
            "boundary_pair_recount": int(v1_recount),
            "locked_clean_cycles": v1_locked_clean,
            "matched_windows": n_match,
            "mismatch_examples": mismatch_examples[:8],
        },
        "fusion_params": {"ctx_days": args.ctx_days, "min_gap": args.min_gap,
                          "edge_guard": args.edge_guard, "vf_min": args.vf_min,
                          "season_months": list(SEASON_MONTHS),
                          "veg_th": args.veg_th, "lst_th": args.lst_th,
                          "swir_th": args.swir_th},
        "headline": {
            "v1_clean_cycles": int(v1_total_clean),
            "lost_spans": int(len(spans_df)),
            "lost_field_years": round(float(spans_df.length_days.sum()) / 365.0, 1),
            "v2_candidate_clean_cycles_primary": int(v2_clean_primary),
            "v2_delta_primary": int(v2_clean_primary - v1_total_clean),
        },
        "class_summary": class_summary.reset_index().to_dict("records"),
        "by_kind": by_kind.reset_index().to_dict("records"),
        "threshold_grid_v2_clean_cycles": [
            {"veg": v, "lst": l, "swir": w, "v2_cycles": c,
             "delta": c - int(v1_total_clean)}
            for (v, l, w), c in sorted(grid_counts.items())
        ],
        "notes": [
            "v2 candidate counts use the SAME 180–540 d validity rule as v1 on the "
            "boundary set (v1 harvests ∪ accepted fusion candidates); no re-smoothing, "
            "no retuning — this is a diagnosis, not the final v2 detector.",
            "unresolved spans = observations exist but no candidate fired; this mixes "
            "'genuinely no harvest' with 'criteria still miss it' — resolvable only "
            "with external harvest records or a reviewed subsample.",
            "Read-only study: no frozen artifact was written; outputs live only under "
            "the report directory. Any v3 detector change reopens the freeze chain "
            "(splits/holdout/store/norm) and stays a separate gated change.",
        ],
    }
    (out_dir / f"report{suffix}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    spans_df.to_csv(out_dir / f"spans{suffix}.csv", index=False)

    hl = report["headline"]
    print(json.dumps(hl, indent=2))
    print(f"[diagnose] done in {wall}s — wrote {out_dir}/report{suffix}.json, spans{suffix}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())

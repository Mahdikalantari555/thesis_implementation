# ruff: noqa: E402
#!/usr/bin/env python3
"""RQ4 (Gap G8): Cloud-Robustness (16-day -> monthly) & E-ALIGN / E-SEG / E-NORM Suite."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ssl_anomaly.utils.throttle import limit_threads

limit_threads(2)

import numpy as np
import pandas as pd
import torch
import xarray as xr

torch.set_num_threads(2)

from ssl_anomaly.baseline import (
    aggregate_band_scores,
    build_stratum_table,
    fit_baseline,
    score_cycle,
    trimmed_normal_cycles,
)
from ssl_anomaly.eval.cloud_robustness import (
    evaluate_e_align_three_metrics,
    evaluate_e_norm_trimming_stability,
    run_cloud_completeness_benchmark,
)
from ssl_anomaly.eval.cv import simulate_490_field_archive
from ssl_anomaly.eval.stats import spearman_correlation
from ssl_anomaly.phenology import add_spectral_indices, phase_bin

SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)


def _evaluate_e_seg_harvest_shifts(zarr_path: Path, cycles_path: Path) -> dict[str, dict]:
    """E-SEG: Shift harvest dates by {-32, -16, +16, +32} days on real 32-farm archive."""
    ds = xr.open_zarr(zarr_path)
    ds = add_spectral_indices(ds)
    cycles_df = build_stratum_table(pd.read_parquet(cycles_path))
    times = ds.coords["time"].values
    epoch = times[0]
    bands = [str(b) for b in ds.coords["band"].values]
    bmap = {b: i for i, b in enumerate(bands)}
    fields = ds.coords["field"].values
    med = ds["median"].values

    def _bin_with_shift(shift_days: int) -> tuple[np.ndarray, np.ndarray]:
        rows, fids = [], []
        for _, c in cycles_df.iterrows():
            if c["censored"]:
                continue
            fid = int(c["field"])
            fi = int(np.where(fields == fid)[0][0])
            s0 = int(c["t_start"]) + shift_days
            s1 = int(c["t_end"]) + shift_days
            sh = int(c.get("start_harvest", c["t_start"])) + shift_days
            t0 = epoch + np.timedelta64(s0, "D")
            t1 = epoch + np.timedelta64(s1, "D")
            hd = epoch + np.timedelta64(sh, "D")
            m = (times >= t0) & (times < t1)
            if int(m.sum()) < 3:
                # Keep alignment by falling back to unshifted window if shifted window empty
                t0 = epoch + np.timedelta64(int(c["t_start"]), "D")
                t1 = epoch + np.timedelta64(int(c["t_end"]), "D")
                m = (times >= t0) & (times < t1)
                if int(m.sum()) < 3:
                    continue
            dsh = (times[m] - hd).astype("timedelta64[D]").astype(int)
            pb = phase_bin({b: med[fi, m, bmap[b]] for b in bands}, dsh, bin_days=15, max_cycle_days=540)
            rows.append(np.stack([pb[b] for b in bands], axis=-1))
            fids.append(fid)
        return np.stack(rows, axis=0), np.array(fids, dtype=np.int64)

    X0, fids = _bin_with_shift(0)
    X_by_f = {int(f): X0[fids == f] for f in np.unique(fids)}
    act = trimmed_normal_cycles(
        {"all": X0}, strata_per_cycle={"all": np.ones(len(X0), dtype=bool)}, X_by_field=X_by_f, min_n=4
    )["all"]
    bl0 = fit_baseline({"all": X0[act]}, X_by_f, min_n=4)

    def _scores_and_burdens(X_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sc, bu = [], []
        for xc, f in zip(X_arr, fids):
            z = score_cycle(xc, bl0, "all", field_id=int(f))
            s = aggregate_band_scores(z)
            fin = np.sort(s[np.isfinite(s)])
            sc.append(float(np.mean(fin[-min(3, fin.size) :])) if fin.size else 0.0)
            fin_z = z[np.isfinite(z)]
            bu.append(float(np.mean(np.abs(fin_z) > 3.5)) if fin_z.size else 0.0)
        return np.array(sc), np.array(bu)

    sc0, bu0 = _scores_and_burdens(X0)
    out: dict[str, dict] = {}
    for sh in (-32, -16, 16, 32):
        Xs, _ = _bin_with_shift(sh)
        scs, bus = _scores_and_burdens(Xs)
        rho, pval = spearman_correlation(sc0, scs)
        out[f"shift_{sh:+d}d"] = {
            "shift_days": sh,
            "spearman_vs_unshifted": rho,
            "pval": pval,
            "mean_abs_burden_delta": float(np.mean(np.abs(bus - bu0))),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Run RQ4 cloud-completeness & E-ALIGN/E-SEG/E-NORM suite.")
    ap.add_argument("--zarr", default="data/processed/fields.zarr", type=Path)
    ap.add_argument("--cycles", default="data/processed/cycles.parquet", type=Path)
    ap.add_argument("--out", default="reports/rq4_cloud_and_stability.json", type=Path)
    args = ap.parse_args()

    from compare_baselines import load_raw_cycles, project_records_to_bins

    # 1. Real 32-farm E-ALIGN, E-SEG, E-NORM
    recs, bands = load_raw_cycles(args.zarr, args.cycles)
    fids = np.array([r.field_id for r in recs], dtype=np.int64)
    X_axes = {
        "DOY": project_records_to_bins(recs, bands, "b0_doy", K=36),
        "DSH": project_records_to_bins(recs, bands, "b1_dsh", K=36),
        "GDD": project_records_to_bins(recs, bands, "b2_gdd", K=36),
        "Landmark": project_records_to_bins(recs, bands, "b2_landmark", K=36),
    }
    e_align = evaluate_e_align_three_metrics(X_axes, fids, seed=42)
    e_norm = evaluate_e_norm_trimming_stability(X_axes["DSH"], fids)
    e_seg = _evaluate_e_seg_harvest_shifts(args.zarr, args.cycles)

    # 2. RQ4 Cloud-completeness sweep (100% -> 35%, with vs without date/Δt conditioning)
    sim = simulate_490_field_archive(n_fields=65, n_years=10.0, seed=42)
    cloud_sweep = run_cloud_completeness_benchmark(
        sim["X"], sim["y_stress"], epochs=20, seeds=(42, 43, 44)
    )

    report = {
        "rq4_cloud_completeness_sweep": cloud_sweep,
        "e_align_three_metrics": e_align,
        "e_seg_harvest_shift_sensitivity": e_seg,
        "e_norm_trimming_q_stability": e_norm,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    df_cloud = pd.DataFrame(list(cloud_sweep.values()))
    df_cloud.to_csv(args.out.with_suffix(".csv"), index=False)

    print("\n=== RQ4 Cloud-Completeness Sweep (date/Δt Conditioned vs Unconditioned SSL) ===")
    print(df_cloud.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== E-ALIGN Three-Metric Summary ===")
    print(pd.DataFrame(e_align).T.to_string(float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

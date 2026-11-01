# ruff: noqa: E402
#!/usr/bin/env python3
"""RQ5 (Gap G5): Early-Detection Lead-Time Benchmark under LOYO Validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ssl_anomaly.utils.throttle import limit_threads

limit_threads(2)

import pandas as pd
import torch

torch.set_num_threads(2)

from ssl_anomaly.eval.early_detection import (
    run_loyo_early_detection,
    simulate_progressive_onset_archive,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run RQ5 early-detection lead-time benchmark.")
    ap.add_argument("--n-cycles", default=400, type=int)
    ap.add_argument("--far", default=0.05, type=float)
    ap.add_argument("--out", default="reports/rq5_early_detection.json", type=Path)
    args = ap.parse_args()

    arch = simulate_progressive_onset_archive(n_cycles=args.n_cycles, seed=42)
    report = run_loyo_early_detection(arch, false_alarm_budget=args.far, seed=42)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    rows = [
        {
            "detector": m,
            "detection_rate_at_5pct_far": d["detection_rate_at_5pct_far"],
            "mean_delay_days": d["mean_delay_days_from_onset"],
            "delay_ci_lo": d["delay_days_ci_95"][0],
            "delay_ci_hi": d["delay_days_ci_95"][1],
            "median_delay_days": d["median_delay_days_from_onset"],
        }
        for m, d in report["methods"].items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv(args.out.with_suffix(".csv"), index=False)

    print("\n=== RQ5 Early Detection Lead-Time Benchmark (LOYO, Matched 5% False-Alarm Rate) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(
        f"SSL lead-time advantage over NDVI threshold: "
        f"mean = {report['ssl_lead_time_vs_ndvi_mean_days']:+.2f} d "
        f"(95% CI [{report['ssl_lead_time_vs_ndvi_ci_95'][0]:+.2f}, {report['ssl_lead_time_vs_ndvi_ci_95'][1]:+.2f}] d, "
        f"median = {report['ssl_lead_time_vs_ndvi_median_days']:+.2f} d) vs EigenCL 12.0 d reference"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

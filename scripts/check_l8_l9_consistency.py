"""Measure L8 vs L9 same-phase consistency on the mini store (task 0A.3).

Read-only probe: per-band median field series per farm, L8/L9 pairs matched
within a ±8-day DOY window (same-phase proxy), robust per-band differences
with MAD CIs via ``io.sensor_offset.band_offset``. Writes a report JSON to
``reports/mini_dataset/l8_l9_consistency.json`` (regenerable, committed).

A resolved CI is a CANDIDATE for the E7 offset arm only — applying any
correction additionally requires the pre-registered score-level acceptance
rule (``sensor_offset.decide_applied``); unresolved = no correction, logged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.io.sensor_consistency import (  # noqa: E402
    farm_band_medians,
    same_phase_consistency,
)
from ssl_anomaly.utils.paths import localize  # noqa: E402
from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402

STORE_ROOT = localize("D:/RS/Thesis/Data/Processed/mini_research_dataset")
OUT = ROOT / "reports" / "mini_dataset" / "l8_l9_consistency.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-pairs", type=int, default=10)
    ap.add_argument("--phase-window", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    t0 = time.monotonic()
    import zarr

    store = STORE_ROOT / "satellite.zarr"
    farms = sorted(p for p in store.iterdir() if (p / ".zgroup").exists())
    report: dict = {
        "schema": "l8-l9-same-phase-consistency/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_commit_sha(),
        "store": str(STORE_ROOT),
        "min_pairs": args.min_pairs,
        "phase_window_days": args.phase_window,
        "farms": {},
    }
    n_resolved_total = 0
    for farm in farms:
        g = zarr.open_group(str(farm), mode="r")
        if "sensor_id" not in g:
            raise RuntimeError(
                f"farm {g.attrs['farm_id']} has no sensor_id array; run "
                "scripts/add_sensor_ids_mini.py first (fail closed)"
            )
        names, medians = farm_band_medians(g)
        sensor = np.asarray(g["sensor_id"][:], dtype=np.int8)
        dates = [str(d) for d in g["dates"][:]]
        rec = same_phase_consistency(
            medians, sensor, dates,
            min_pairs=args.min_pairs,
            phase_window_days=args.phase_window,
            band_names=names,
        )
        report["farms"][str(g.attrs["farm_id"])] = rec
        n_resolved_total += rec["resolved_count"]
        print(
            f"{g.attrs['farm_id']}: L8={rec['n_l8_dates']} L9={rec['n_l9_dates']} "
            f"pairs={rec['n_pairs']} resolved_bands={rec['resolved_count']}/"
            f"{len(rec['bands'])}",
            flush=True,
        )
        for band, b in rec["bands"].items():
            n = b.get("n_pairs", b.get("n_diffs"))
            print(
                f"    {band:6s} offset={b.get('offset')} "
                f"CI=[{b.get('ci_low')}, {b.get('ci_high')}] n={n} "
                f"resolved={b.get('resolved')}",
                flush=True,
            )
    report["n_resolved_total"] = n_resolved_total
    report["decision"] = (
        "measurement only; corrections go through the E7 sensor-handling "
        "arms with sensor_offset.decide_applied's pre-registered rule"
    )
    report["wall_s"] = round(time.monotonic() - t0, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.out} wall={report['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

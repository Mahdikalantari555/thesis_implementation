"""Run the 32-farm CPU anomaly probe end-to-end.

All fits (B2 profiles/σ, CNN-AE, Barlow Twins, GRU-AE) use **train dates only**
(first ``--train-years`` calendar years). Outputs are Stage-1 (farm, date)
scores + gates + events, Stage-2 pixel maps for the top-K suspicious pairs
(ONNXRuntime CPU), and a complete provenance JSON. **L2 screening evidence
only** — outputs carry anomaly wording, never "stress" (no L4 ground truth).

Full run (4 methods × 1 seed) fits the ≤15 min/seed budget on this machine;
launch it with the exact command from the WP-3 handover, not interactively.

Usage::

    python scripts/probe/run_probe.py --smoke --subset .tmp/wp3_smoke/subset.zarr --out .tmp/wp3_smoke/probe
    python scripts/probe/run_probe.py --subset D:/RS/Thesis/Data/Processed/raw_satellite_subset.zarr --out reports/probe --seed 2026
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.probe.pipeline import run_probe  # noqa: E402
from ssl_anomaly.utils.paths import localize  # noqa: E402

SUBSET_DEFAULT = localize("D:/RS/Thesis/Data/Processed/raw_satellite_subset.zarr")
OUT_DEFAULT = ROOT / "reports" / "probe"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--subset", type=Path, default=SUBSET_DEFAULT)
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--train-years", type=int, default=3)
    ap.add_argument("--methods", default="b2,cnn_ae,barlow,gru_ae",
                    help="comma-separated subset of b2,cnn_ae,barlow,gru_ae")
    ap.add_argument("--cnn-epochs", type=int, default=30)
    ap.add_argument("--barlow-epochs", type=int, default=5)
    ap.add_argument("--gru-epochs", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--flag-q", type=float, default=0.99)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch per method, top-K=2 (end-to-end check)")
    args = ap.parse_args(argv)
    if not args.subset.exists():
        print(f"ERROR: subset store not found: {args.subset}", file=sys.stderr)
        print("Build it first: scripts/build_raw_satellite_subset.py "
              "(see WP-3 handover for the exact command)", file=sys.stderr)
        return 2
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    summary = run_probe(
        args.subset, args.out, seed=args.seed, train_years=args.train_years,
        methods=methods, cnn_epochs=args.cnn_epochs,
        barlow_epochs=args.barlow_epochs, gru_epochs=args.gru_epochs,
        top_k=args.top_k, flag_q=args.flag_q, resume=args.resume,
        smoke=args.smoke,
    )
    return 0 if summary["event"] == "PROBE COMPLETE" else 1


if __name__ == "__main__":
    sys.exit(main())

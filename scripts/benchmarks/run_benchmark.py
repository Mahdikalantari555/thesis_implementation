"""Architecture benchmark: verify, plan, and (only when told to) execute one block.

Change ``benchmark-latent-reconstruction-prediction-architectures`` (phases 12–13).

The default mode is **plan, not run**. Training is hours-long and belongs on the
owner's Colab T4 (AGENTS.md §6.1/§6.2), so everything that can be decided without a
GPU is decided and checked here first:

    python scripts/benchmarks/run_benchmark.py --verify-only          # matrix drift
    python scripts/benchmarks/run_benchmark.py --plan --block screening
    python scripts/benchmarks/run_benchmark.py --estimate-runtime --block full

and only ``--execute --yes`` starts training:

    python scripts/benchmarks/run_benchmark.py --execute --yes --fixture --smoke \\
        --block screening --tier screening

Three design points are load-bearing:

* **The full matrix is always registered**, and the block/tier/arm/seed filters are
  *plan* filters. Pre-filtering the cells before ``register()`` would make the registry
  cover only this run's subset, so ``coverage.md`` would read as complete and the
  budget-exhaustion deferral would have nothing to defer. The committed
  ``configs/benchmarks/matrix.yaml`` is passed as ``--matrix-path`` by default, so a
  drifted factor set stops the run instead of redefining it.
* **Runtime estimates come from a measured profile**, not from a table of hopes: an
  unprofiled arm is admitted (that is what screening is for) and labelled as such.
* Every mode prints JSON and exits non-zero on a refused precondition, so a notebook
  cell can chain steps without swallowing a failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.benchmarking import matrix as M  # noqa: E402
from ssl_anomaly.benchmarking.config import (  # noqa: E402
    APPLICABILITY_PATH,
    CONFIG_PATH,
    load_applicability,
    load_benchmark_config,
    validate_config,
)
from ssl_anomaly.benchmarking.runner import BenchmarkRunner, RunOptions  # noqa: E402

DEFAULT_OUT = ROOT / "reports" / "architecture_benchmark" / "runs"
COMMITTED_MATRIX = ROOT / "configs" / "benchmarks" / "matrix.yaml"


def _csv_ints(text: str | None) -> tuple[int, ...]:
    return tuple(int(x) for x in (text or "").split(",") if x.strip())


def _csv(text: str | None) -> tuple[str, ...]:
    return tuple(x.strip() for x in (text or "").split(",") if x.strip())


def _profile_hours(path: Path | None) -> dict[str, float]:
    """Per-arm hours from the most recent run summary, if one exists."""
    if path is None or not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        print(f"--profile could not be read ({exc}); estimating without a profile", file=sys.stderr)
        return {}
    est = ((doc.get("run") or {}).get("budget") or {}).get("estimates_hours") or {}
    return {str(k): float(v) for k, v in est.items()}


def build_runner(a, cfg, app, cells) -> BenchmarkRunner:
    opts = RunOptions(
        base=Path(a.data_root or a.base), out=Path(a.out), tier=a.tier or None,
        blocks=_csv(a.block), model_ids=_csv(a.model_ids), seeds=_csv_ints(a.seeds),
        limit=a.limit, epochs=a.epochs, resume=a.resume, smoke=a.smoke, fixture=a.fixture,
        dry_run=a.dry_run, hpo=a.hpo, hpo_trials=a.hpo_trials,
        matrix_path=Path(a.matrix_path) if a.matrix_path else (
            COMMITTED_MATRIX if COMMITTED_MATRIX.is_file() else None),
    )
    r = BenchmarkRunner(cfg, app, cells, opts)
    prof = _profile_hours(Path(a.profile) if a.profile else Path(a.out) / "run_summary.json")
    if prof:
        r.budget.estimates.update(prof)
        r.notes.append(f"budget estimates seeded from {a.profile or 'the previous run_summary.json'} "
                       f"({len(prof)} arms)")
    return r


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help=f"benchmark config (default {CONFIG_PATH})")
    ap.add_argument("--applicability", default=None)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--matrix-path", default=None,
                    help=f"committed matrix to bind to (default {COMMITTED_MATRIX} when it exists)")
    ap.add_argument("--block", default=None, help="comma-separated matrix block labels")
    ap.add_argument("--tier", default=None, choices=[None, "screening", "full"])
    ap.add_argument("--model-ids", default=None, help="comma-separated arm ids (A1,A9,…)")
    ap.add_argument("--seeds", default=None, help="comma-separated subset of 13,42,2026")
    ap.add_argument("--base", default=str(ROOT), help="root the artifacts are read from")
    ap.add_argument("--data-root", default=None,
                    help="extracted v2 bundle root; required for a real (non-fixture) run")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run, stamped as plumbing evidence")
    ap.add_argument("--fixture", action="store_true", help="synthetic panel instead of the v2 archive")
    ap.add_argument("--hpo", action="store_true", help="run the per-arm equal-budget search first")
    ap.add_argument("--hpo-trials", type=int, default=None)
    ap.add_argument("--profile", default=None, help="run_summary.json to take wall-time estimates from")
    ap.add_argument("--plan", action="store_true", help="print the plan (no training)")
    ap.add_argument("--verify-only", action="store_true", help="check the committed matrix, then stop")
    ap.add_argument("--estimate-runtime", action="store_true")
    ap.add_argument("--execute", action="store_true", help="actually train (owner-run, T4)")
    ap.add_argument("--yes", action="store_true", help="required with --execute")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_benchmark_config(Path(a.config) if a.config else CONFIG_PATH)
    app = load_applicability(Path(a.applicability) if a.applicability else APPLICABILITY_PATH)
    cells = M.build_matrix(cfg, app)
    out = Path(a.out)

    if a.verify_only:
        v = M.verify_committed_matrix(cfg, app, path=(Path(a.matrix_path) if a.matrix_path else None))
        print(json.dumps(v, indent=1, sort_keys=True, default=str))
        return 0 if v["ok"] else 2

    bad = validate_config(cfg.raw)
    if bad:
        print(json.dumps({"refused": "config validation failed", "problems": bad}, indent=1),
              file=sys.stderr)
        return 2

    r = build_runner(a, cfg, app, cells)

    if a.estimate_runtime:
        plan = r.plan()
        per = {c.model_id: r.budget.estimate(c.model_id, tier=c.tier) for c in plan}
        total = round(sum(float(v or 0.0) for v in per.values()), 3)
        print(json.dumps({
            "cells_planned": len(plan), "profiled": bool(r.budget.estimates),
            "hours_projected": total, "budget_hours": float(cfg.raw["budget"]["t4_hours"]),
            "fits": bool(total <= float(cfg.raw["budget"]["t4_hours"])),
            "unprofiled_arms": sorted(k for k, v in per.items() if v is None),
            "per_arm_hours": {k: v for k, v in sorted(per.items())},
            "note": ("unprofiled arms are admitted by the screening tier precisely to produce a "
                     "profile; --estimate-runtime is only meaningful after one smoke run exists"),
        }, indent=1, sort_keys=True, default=str))
        return 0

    if a.execute and not a.dry_run:
        if not a.yes:
            print("--execute needs --yes: training must be a decision, not a typo in a notebook cell",
                  file=sys.stderr)
            return 2
        if not (a.fixture or a.smoke) and not a.data_root:
            print("--execute on real data needs --data-root (the extracted v2 bundle); refusing to "
                  "guess where the archive lives", file=sys.stderr)
            return 2
    else:
        r.notes.append("plan-only mode: no cell was claimed for training")

    r.prepare()
    if a.execute and not a.dry_run:
        t0 = time.time()
        summary = r.run()
        print(json.dumps({"wall_seconds": round(time.time() - t0, 1),
                          "by_status": summary["by_status"], "statuses_after": summary["statuses_after"],
                          "budget": summary["run"]["budget"], "out": str(out),
                          "notes": summary["notes"][:12]}, indent=1, sort_keys=True, default=str))
        return 0

    plan = r.plan()
    counts = dict(M.status_counts(cells))
    print(json.dumps({
        "mode": "dry-run" if a.dry_run else "plan", "execute": bool(a.execute),
        "cells_in_matrix": len(cells), "cells_planned": len(plan),
        "matrix_sha256": r.matrix_sha, "config_sha256": cfg.document_sha256,
        "matrix_path": str(getattr(r.opts, "matrix_path", None)),
        "statuses": counts,
        "coverage": M.coverage_line(counts, total=len(cells)),
        "budget": r.budget.to_dict(), "device": r.device.to_dict(), "hashes": r.hashes,
        "objective_audit_problems": r.objective_audit["problems"],
        "bindings": (r.bindings.to_dict() if hasattr(r.bindings, "to_dict") else {}),
        "notes": r.notes[:20], "planned_ids": [c.experiment_id for c in plan],
    }, indent=1, sort_keys=True, default=str))
    if not a.execute:
        print("\n(no training started: pass --execute --yes to run this plan)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

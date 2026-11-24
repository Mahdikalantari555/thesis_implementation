"""Bundle audit / assembly / verification for a benchmark run (AGENTS.md §6.2).

Thin CLI over :mod:`ssl_anomaly.benchmarking.bundle`, which holds the rules. The point of
the separate entry point is that the check has to be runnable *on the Colab box before
anything is cited*, and on the laptop after download, with the same exit code:

    python scripts/benchmarks/bundle_benchmark.py --audit /content/bench
    python scripts/benchmarks/bundle_benchmark.py --collect --out /content/bench \\
        --dest /content/drive/MyDrive/thesis_colab
    python scripts/benchmarks/bundle_benchmark.py --verify /content/drive/MyDrive/thesis_colab/benchmark_2026-11-23T120000Z

``--collect`` refuses (exit 2) when a cell is missing a required file or a manifest field,
because a partial bundle published next to a complete one is the failure mode this
whole module exists to prevent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.benchmarking import bundle as B  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="run directory (audit/collect) or bundle directory (verify)")
    ap.add_argument("--audit", action="store_true", help="per-cell completeness audit (default)")
    ap.add_argument("--collect", action="store_true", help="assemble a verified bundle under --dest")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--verify", action="store_true", help="re-hash every file against the manifest")
    ap.add_argument("--drive-plan", action="store_true", help="print the upload commands, run nothing")
    ap.add_argument("--name", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--note", default=None, help="one-line note for the bundle's cover page")
    a = ap.parse_args(argv)

    path = Path(a.path)
    if a.verify:
        res, rc = B.verify_bundle(path), None
    elif a.drive_plan:
        res, rc = B.drive_plan(path, zip_name=a.name), None
    elif a.collect:
        dest = Path(a.dest) if a.dest else path.parent
        plan = B.collect(path, dest, name=a.name, dry_run=a.dry_run,
                         notes=None if not a.note else [a.note])
        if plan.get("complete") and not a.dry_run:
            plan["LATEST_BUNDLE.md"] = str(Path(plan["target"]) / "LATEST_BUNDLE.md")
        res, rc = plan, (0 if plan.get("complete") else 2)
    else:
        res, rc = B.audit_bundle(path).to_dict(), None
    print(json.dumps(res, indent=1, sort_keys=True, default=str))
    if rc is not None:
        return rc
    if isinstance(res, dict) and not res.get("ok", res.get("complete", True)):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

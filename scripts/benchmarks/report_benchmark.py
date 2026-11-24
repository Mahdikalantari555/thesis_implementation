"""Render the benchmark report and the machine-readable evidence pack (tasks 14.1, 14.4).

Everything here is a *view* of a run directory: no number is computed from anything but
the manifests the run wrote, and no section is dropped when it is empty. The three
export formats exist because the thesis is read three ways — a human reads ``REPORT.md``,
a script reads ``report.json``, and a reviewer diffs ``results.csv``.

    python scripts/benchmarks/report_benchmark.py --out /content/bench
    python scripts/benchmarks/report_benchmark.py --out reports/architecture_benchmark/runs \\
        --copy-to reports/architecture_benchmark

``--copy-to`` additionally writes ``coverage.md`` and the question ledger next to the
run, which is the form the finding document cites.

The report itself refuses to be a source of results: if every cell is a fixture or a
smoke run, the document says "plumbing evidence" in its first lines, and this script
reports that in ``--json`` so a notebook cell can assert it rather than admire a table.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.benchmarking import report as R  # noqa: E402
from ssl_anomaly.benchmarking.config import (  # noqa: E402
    APPLICABILITY_PATH,
    CONFIG_PATH,
    load_applicability,
    load_benchmark_config,
)
from ssl_anomaly.benchmarking.matrix import build_matrix  # noqa: E402

#: the factors the report's tables can be grouped by — the same names ``aggregate`` reads
#: off the manifest, so the CSV columns and the markdown tables cannot drift apart
FACTORS = ("paradigm", "backbone", "suppression", "input", "fusion", "calibration",
           "protocol", "variant", "level", "block")
FACTOR_OF = {f: f for f in FACTORS}


def write_csv(rows: list[dict], path: Path) -> int:
    """One row per (cell, metric) so a spreadsheet can pivot without a parser."""
    flat: list[dict] = []
    for r in rows:
        macro = ((r.get("metrics") or {}).get("macro") or {})
        base = {k: r.get(k) for k in ("experiment_id", "model_id", "status", "tier", "seed",
                                      "block", "fixture", "usable")}
        base.update({f"factor_{v}": (r.get("factors") or {}).get(v) for v in FACTOR_OF.values()})
        base.update({"macro_auprc": macro.get("macro"), "macro_sd": macro.get("macro_sd_across_cells"),
                     "event_weighted_auprc": macro.get("event_weighted"),
                     "n_cells": macro.get("n_cells"), "cells_missing": "|".join(macro.get("cells_missing") or []),
                     "prevalence_min": (macro.get("prevalence_range") or [None, None])[0],
                     "prevalence_max": (macro.get("prevalence_range") or [None, None, 0, 0])[1],
                     "ladder_rung": r.get("ladder_rung"), "stamp": r.get("stamp"),
                     "wall_seconds": r.get("wall_seconds"), "error": r.get("error"),
                     "run_dir": r.get("run_dir")})
        per_cell = (r.get("metrics") or {}).get("per_cell") or macro.get("per_cell") or []
        for pc in per_cell:
            row = dict(base)
            row.update({"cell": pc.get("cell"), "family": pc.get("family"), "severity": pc.get("severity"),
                        "auprc": pc.get("auprc"), "auroc": pc.get("auroc"), "f1": pc.get("f1"),
                        "f1_at_budget": pc.get("f1_at_budget"), "median_delay_days": pc.get("median_delay_days"),
                        "spearman": pc.get("spearman"), "kendall": pc.get("kendall"),
                        "brier": pc.get("brier"), "ece": pc.get("ece"),
                        "n_farms_scored": (pc.get("aux") or {}).get("n_farms_scored"),
                        "prevalence": pc.get("prevalence")})
            flat.append(row)
        if not per_cell:
            flat.append(base)
    if not flat:
        return 0
    keys: list[str] = []
    for row in flat:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in flat:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()})
    return len(flat)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="run directory to report on")
    ap.add_argument("--config", default=None)
    ap.add_argument("--applicability", default=None)
    ap.add_argument("--n-boot", type=int, default=None,
                    help=f"bootstrap draws (floor {R.MIN_BOOT}; a lower value is refused)")
    ap.add_argument("--copy-to", default=None, help="also write REPORT/report files here")
    ap.add_argument("--json", action="store_true", help="machine-readable summary instead of paths")
    a = ap.parse_args(argv)

    out = Path(a.out)
    if not out.is_dir():
        print(f"--out {out} is not a directory: report on a run, not on an intention", file=sys.stderr)
        return 2
    cfg = load_benchmark_config(Path(a.config) if a.config else CONFIG_PATH)
    app = load_applicability(Path(a.applicability) if a.applicability else APPLICABILITY_PATH)
    cells = build_matrix(cfg, app)
    kw = {} if a.n_boot is None else {"n_boot": a.n_boot}
    art = R.build_report(out, cfg, cells, **kw)
    rows = R.load_cells(out)
    usable = [r for r in rows if r.get("usable")]
    cited = [r for r in usable if not r.get("fixture") and r.get("status") == "Completed"]

    pack = out / "evidence"
    pack.mkdir(parents=True, exist_ok=True)
    n_rows = write_csv(rows, pack / "results.csv")
    (pack / "cells.json").write_text(json.dumps(rows, indent=1, sort_keys=True, default=str),
                                     encoding="utf-8")
    stats = {"aggregate": {}, "comparisons": {}, "variance": None}
    for factor in FACTORS:
        try:
            stats["aggregate"][factor] = R.aggregate(usable, factor)
            stats["comparisons"][factor] = R.paired_comparisons(
                usable, factor, n_boot=(a.n_boot or R.MIN_BOOT))
        except Exception as exc:  # noqa: BLE001 - an absent factor is reported, not fatal
            stats["aggregate"][factor] = [{"error": f"{type(exc).__name__}: {exc}"}]
    try:
        stats["variance"] = R.variance_split(usable)
    except Exception as exc:  # noqa: BLE001
        stats["variance"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    (pack / "statistical.json").write_text(json.dumps(stats, indent=1, sort_keys=True, default=str),
                                           encoding="utf-8")

    if a.copy_to:
        dest = Path(a.copy_to)
        dest.mkdir(parents=True, exist_ok=True)
        for src, name in ((art.path, "REPORT.md"), (out / "report.json", "report.json"),
                          (out / "coverage.md", "coverage.md"), (pack / "results.csv", "results.csv"),
                          (pack / "statistical.json", "statistical.json"),
                          (pack / "cells.json", "cells_manifest.json")):
            if src and Path(src).is_file():
                shutil.copy2(src, dest / name)

    summary = {"report": str(art.path), "report_json": str(out / "report.json"),
               "report_chars": len(art.text()),
                "warnings": list(art.warnings), "cells_in_registry": len(rows),
                "cells_usable": len(usable), "cells_citable": len(cited),
                "csv_rows": n_rows, "evidence_dir": str(pack),
                "n_boot_effective": (a.n_boot or R.MIN_BOOT),
                "n_sections": len(getattr(art, "sections", [])),
                "headings": [ln for ln in art.text().splitlines() if ln.startswith("## ")]}
    print(json.dumps(summary, indent=1, sort_keys=True, default=str) if a.json else
          "\n".join([f"report: {art.path}", f"evidence: {pack}",
                     f"cells: {len(rows)} recorded, {len(usable)} with metrics, {len(cited)} citable",
                     f"warnings: {list(art.warnings) or 'none'}"]))
    if not cited:
        print("no cell is citable (fixture/smoke or incomplete): the report is plumbing evidence only, "
              "which is what its own header says", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

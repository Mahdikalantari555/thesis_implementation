"""Equal-budget hyper-parameter search per arm (tasks 11.1–11.6).

Two rules decide whether a benchmark's "A1 beat A6" sentence means anything: every arm
gets the same search budget, and the selection signal is validation-only. This script
enforces both by construction:

* ``hpo.assert_equal_budgets`` refuses an unequal table *before* any trial runs;
* the selection score is the runner's own macro-AUPRC on the dev partition of a short
  probe — the runner cannot see the locked test split, so neither can the search
  (``search.validation_only`` is asserted, and ``hpo.run_search`` refuses a forbidden
  ``--split`` outright);
* ``d_model`` is **excluded** from the space here, because the parameter band is what
  makes the arms comparable and is chosen by ``capacity.tuning``; letting the search
  move it would let an arm win by being larger;
* ``hpo.audit_search`` re-reads the finished traces and reports any trial recorded on a
  forbidden split — the audit reads the record, not the caller's promise.

The selection is *not* written into the frozen ``configs/benchmarks/benchmark_config.yaml``:
that file is the pre-registration, and mutating it after a search would be
post-hoc tuning wearing a frozen hat. The chosen configs land in
``<out>/hpo/hpo_selection.json`` and are applied by passing ``--from-selection`` to
``run_benchmark.py`` (which records them as a selection layer, not as the contract).

Runs without CUDA or the v2 archive:

    python scripts/benchmarks/run_hpo.py --arms A1,A9 --trials 2 --epochs 1 \\
        --fixture --smoke --out /tmp/hpo1
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.benchmarking import hpo  # noqa: E402
from ssl_anomaly.benchmarking.config import CONFIG_PATH, load_benchmark_config  # noqa: E402
from ssl_anomaly.benchmarking.matrix import build_matrix  # noqa: E402
from ssl_anomaly.benchmarking.runner import BenchmarkRunner, RunOptions  # noqa: E402


def _macro(row: dict) -> float:
    return float(((row.get("metrics") or {}).get("macro") or {}).get("macro") or float("nan"))


def make_probe(cfg, app, cells, a, mid: str):
    """One short run per trial, scored on the dev partition by the runner itself."""

    def evaluate(cfg_dict: dict) -> tuple[float, dict]:
        raw = copy.deepcopy(cfg.raw)
        raw.setdefault("training", {}).update(
            {k: v for k, v in cfg_dict.items() if k in _TRAIN_KEYS})
        probe = dataclasses.replace(cfg, raw=raw)
        opts = RunOptions(base=Path(a.data_root or a.base), out=Path(a.out) / mid / f"t{evaluate.n}",
                          tier=a.tier, fixture=True, smoke=True, epochs=a.epochs, limit=1,
                          model_ids=(mid,), batch_size=a.batch_size)
        evaluate.n += 1
        r = BenchmarkRunner(probe, app, cells, opts)
        r.prepare()
        summary = r.run()
        rows = summary.get("results") or []
        if not rows:
            raise RuntimeError(f"{mid}: the probe planned no cell — nothing was trained")
        row = rows[0]
        if row.get("status") != "Completed":
            raise RuntimeError(f"{mid}: probe cell {row.get('status')} ({row.get('reason')})")
        params = int(((row.get("capacity") or {}).get("chosen") or {}).get("params") or 0)
        return _macro(row), {"params": params or None, "note": f"probe run {row.get('experiment_id')}",
                             "probe_status": row.get("status")}

    evaluate.n = 0
    return evaluate


_TRAIN_KEYS = ("lr", "dropout", "n_layers", "horizon_k", "lambda_pred", "lambda_rec",
               "ema_momentum", "d_state", "latent_dim")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=str(ROOT / "reports" / "architecture_benchmark" / "hpo"))
    ap.add_argument("--arms", default=None, help="comma-separated; default every searchable arm")
    ap.add_argument("--trials", type=int, default=None, help="default: the contract's per-arm budget")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--split", default="dev",
                    help="selection partition; test/monitoring are refused by hpo.run_search")
    ap.add_argument("--base", default=str(ROOT))
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--tier", default="screening", choices=["screening", "full"])
    ap.add_argument("--fixture", action="store_true", help="synthetic panel (plumbing check)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--table", action="store_true", help="print the markdown table and exit")
    a = ap.parse_args(argv)

    cfg = load_benchmark_config(Path(a.config) if a.config else CONFIG_PATH)
    eq = hpo.assert_equal_budgets(cfg)            # refuses an unequal table before any training
    app = __import__("ssl_anomaly.benchmarking.config", fromlist=["load_applicability"]
                     ).load_applicability()
    cells = build_matrix(cfg, app)
    arms = [x.strip() for x in (a.arms or "").split(",") if x.strip()] or sorted(eq["arms"])

    if a.table:
        print(json.dumps({"equal_budgets": eq, "arms": arms,
                          "spaces": {m: {k: (v if isinstance(v, list) else str(v))
                                         for k, v in hpo.budget_for(cfg, m)["space"].items()}
                                     for m in arms}}, indent=1, sort_keys=True, default=str))
        return 0

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for mid in arms:
        budget = hpo.budget_for(cfg, mid)
        space = budget["space"]
        # d_model is *not* searchable here. The parameter band (capacity.target_params and
        # the min/max ratios) is what makes the arms comparable, and it is chosen by
        # capacity.tuning over the pre-registered grid; letting HPO move d_model as well
        # would let an arm win by being bigger. So the key is dropped, and the drop is
        # reported rather than quietly forgotten.
        dropped = sorted(k for k in space if k == "d_model")
        space = {k: v for k, v in space.items() if k not in dropped}
        if not space:
            raise SystemExit(f"{mid}: its declared space is only capacity dimensions, which HPO may not "
                             "move; nothing to search")
        if not a.fixture:
            raise SystemExit(
                f"{mid}: refusing to search against the real archive without --fixture — a search on "
                "unverified data binding costs T4 hours and, on this machine, would fail halfway with "
                "trials recorded as errors. Pass --fixture to exercise the plumbing first.")
        rec = hpo.run_search(dataclasses.replace(cfg, raw={**cfg.raw, "search": {
                                 **cfg.raw["search"],
                                 "budgets": {k: ({**v, "space": space} if k == mid else v)
                                             for k, v in cfg.raw["search"]["budgets"].items()}}}),
                             mid, make_probe(cfg, app, cells, a, mid), split=a.split, seed=a.seed,
                             trials=a.trials)
        rec.budget["capacity_keys_excluded"] = dropped
        records.append(rec)
        (out / f"{mid}.json").write_text(json.dumps(rec.to_dict(), indent=1, sort_keys=True,
                                                    default=str), encoding="utf-8")
        print(f"[hpo] {mid}: {len(rec.trials)} trials, best "
              f"{'—' if rec.best is None else f'{rec.best.score:.4f} ({rec.best.config})'}",
              flush=True)

    audit = hpo.audit_search(records)
    selection = {r.model_id: (dict(r.best.config) if r.best else None) for r in records}
    doc = {"equal_budgets": eq, "audit": audit, "selection": selection,
           "split": a.split, "trials_per_arm": a.trials or eq["trials_per_arm"],
           "epochs_per_trial": a.epochs, "probe_tier": a.tier,
           "applied_by": ("run_benchmark.py --selection-file <this file> (a selection layer, not an "
                          "edit to the frozen contract)"),
           "warning": ("selections are made on a short probe at the screening budget; they are NOT the "
                       "full-budget configuration, and reusing them as such would be a different "
                       "experiment than the one the contract pre-registered"),
           "table": hpo.search_table(records)}
    (out / "hpo_selection.json").write_text(json.dumps(doc, indent=1, sort_keys=True,
                                                       default=str), encoding="utf-8")
    (out / "hpo_table.md").write_text(doc["table"] + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "audit_ok": audit["ok"], "violations": audit["violations"],
                      "selection": selection}, indent=1, sort_keys=True, default=str))
    return 0 if audit["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

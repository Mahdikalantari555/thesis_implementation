"""Drive the generalization matrix: 5 folds × cells × sensor eras.

Change ``patch-generalization-suite`` task 1.2.

The RUNS are owner-executed (training is hours long). This script supplies what
can be built and tested without them: the pre-registered cell plan, the runtime
estimate, the resume scan, and the aggregation + acceptance report. It is
deliberately able to do all four WITHOUT a store —

    python scripts/run_generalization_matrix.py --plan
    python scripts/run_generalization_matrix.py --estimate-runtime
    python scripts/run_generalization_matrix.py --report --dir reports/generalization

so the plan and the acceptance verdict can be inspected, and re-derived
identically, before a single GPU-hour is spent.

``--execute`` is the only mode that trains; it shells out to ``train_patch.py``
and ``eval_dev.py`` per cell so the checkpoints, config hashes and dev_log rows
are produced by the SAME entry points every other result came from (a bespoke
training loop here would make the matrix incomparable with the rest of the
thesis).

Design constraints encoded below, all from the change's design.md:

* Train = the 8 in-fold blocks; the fold's 2 blocks are held out. The fold
  definition is hashed and written into every artifact.
* Per-fold normalization is RE-FIT from in-fold train cycles only
  (``generalization.refit_norm_in_fold``); the store's sidecar is never used
  inside a fold.
* The temporal column is separate and never pooled with the spatial folds.
* Period and sensor are confounded; every temporal result is reported raw AND
  stratified by sensor era, with a plain statement of the confound.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.monitor import generalization as G  # noqa: E402
from ssl_anomaly.monitor import generalization_matrix as GM  # noqa: E402

DEFAULT_REPORT_DIR = ROOT / "reports" / "generalization"
DEFAULT_ARTIFACT_DIR = ROOT / "reports" / "generalization" / "cells"
DEFAULT_FOLDS_FILE = ROOT / "reports" / "generalization" / "folds.json"
DEFAULT_ARMS = ("patch_cnn_gru", "patch_pixelwise")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def load_field_coordinates(path: str | Path) -> tuple[dict[int, int], dict[int, tuple[float, float]]]:
    """Field → block and field → (x, y) from ``data/geo/fields.parquet``.

    The block assignment is the frozen k-means one (``splits.build_blocks``,
    seed 13): re-clustering here would produce a different fold definition than
    the manifests record, which is precisely the silent divergence the folds
    file exists to prevent.
    """
    import pandas as pd

    df = pd.read_parquet(path)
    need = {"field_id"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    block_col = next((c for c in ("block", "block_id", "cluster") if c in df.columns), None)
    if block_col is None:
        raise ValueError(f"{path} has no block column; pass a frozen block assignment")
    x_col = next((c for c in ("x", "centroid_x", "lon") if c in df.columns), None)
    y_col = next((c for c in ("y", "centroid_y", "lat") if c in df.columns), None)
    if x_col is None or y_col is None:
        raise ValueError(f"{path} needs centroid coordinates (x/y) to pair blocks by adjacency")
    block_of = {int(r["field_id"]): int(r[block_col]) for _, r in df.iterrows()}
    xy = {int(r["field_id"]): (float(r[x_col]), float(r[y_col])) for _, r in df.iterrows()}
    return block_of, xy


def folds_from_file(path: str | Path) -> list[G.Fold]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    folds = []
    for rec in payload["folds"]:
        folds.append(G.Fold(
            fold_id=int(rec["fold_id"]), blocks=tuple(rec["blocks"]),
            held_out_fields=list(rec["held_out_fields"]),
            train_blocks=list(rec["train_blocks"]),
            train_fields=list(rec["train_fields"]),
            mean_intra_distance=rec.get("mean_intra_distance") or float("nan"),
            mean_inter_distance=rec.get("mean_inter_distance") or float("nan"),
        ))
    return folds


def save_folds(path: str | Path, folds: list[G.Fold], *, source: str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": source,
        "n_folds": len(folds),
        "folds_sha256": G.folds_sha256(folds),
        "folds": [f.to_dict() for f in folds],
    }
    p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def _resolve_folds(args):
    """Folds from ``--build-folds`` (a fields parquet) or a frozen ``--folds`` file.

    Two explicit flags rather than one overloaded path: building a fold
    definition and reading the frozen one are different acts, and a default
    value that silently means "build" would let a typo re-define the folds that
    every recorded artifact is keyed on.
    """
    if getattr(args, "build_folds", None):
        block_of, xy = load_field_coordinates(args.build_folds)
        folds = G.build_folds(block_of, xy)
        if args.out:
            save_folds(args.out, folds, source=str(args.build_folds))
        return folds
    if args.folds:
        return folds_from_file(args.folds)
    return None


def cmd_plan(args) -> int:
    folds = _resolve_folds(args)
    if folds is None:
        print("--plan needs --folds (frozen) or --build-folds (fields parquet)",
              file=sys.stderr)
        return 2
    cells = GM.plan_matrix(folds, temporal_cutoff_day=args.temporal_cutoff_day)
    print(json.dumps({
        "folds_sha256": G.folds_sha256(folds),
        "cells_sha256": GM.cells_sha256(cells),
        "folds": [f.to_dict() for f in folds],
        "cells": [c.to_dict() for c in cells],
    }, indent=2))
    return 0


def cmd_estimate(args) -> int:
    folds = _resolve_folds(args)
    if folds is None:
        print("--estimate-runtime needs --folds or --build-folds", file=sys.stderr)
        return 2
    cells = GM.plan_matrix(folds, temporal_cutoff_day=args.temporal_cutoff_day)
    est = GM.estimate_matrix_runtime(
        cells, seconds_per_epoch=args.seconds_per_epoch, epochs=args.epochs,
        seconds_per_eval_cell=args.seconds_per_eval,
        seeds_screening=args.seeds_screening, seeds_final=args.seeds_final,
        n_arms_screening=args.n_arms_screening, n_arms_final=args.n_arms_final,
    )
    est["folds_sha256"] = G.folds_sha256(folds)
    est["cells_sha256"] = GM.cells_sha256(cells)
    print(json.dumps(est, indent=2))
    return 0


def cmd_report(args) -> int:
    """Aggregate the per-cell artifacts into the matrix report + verdicts."""
    d = Path(args.dir)
    records = []
    for p in sorted(d.glob("*__*__seed*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        rec["metrics"]["arm"] = rec["arm"]
        rec["metrics"]["cell_id"] = rec["cell_id"]
        rec["metrics"]["axis"] = rec.get("axis")
        rec["metrics"]["seed"] = rec.get("seed")
        rec["metrics"]["sensor_era"] = rec["metrics"].get("sensor_era")
        records.append(rec["metrics"])
    if not records:
        print(f"no cell artifacts under {d}", file=sys.stderr)
        return 2

    matrix = GM.aggregate_matrix(records)
    arms = sorted({r["arm"] for r in records})
    report = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_records": len(records),
        "matrix": matrix,
        "verdicts": {},
        "sensor_confound_note": (
            "Period and sensor are confounded (L9 enters the archive late, and the "
            "sensor-id join carries no acquisition time): a temporal drop whose "
            "sensor-stratified columns show an era jump is a measurement artifact "
            "until proven otherwise. The stratified columns are reported beside the "
            "raw one for exactly this reason."
        ),
    }
    if args.dev_se is not None:
        for arm in arms:
            report["verdicts"][arm] = GM.apply_acceptance(
                matrix, arm=arm, dev_se_sentinel=args.dev_se,
                in_block_sentinel=args.in_block,
            )
    out = d / "matrix_report.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out), "arms": arms,
                      "verdicts": report["verdicts"]}, indent=2))
    return 0


def cmd_execute(args) -> int:
    """Run the missing cells by shelling out to the canonical entry points."""
    d = Path(args.dir)
    done = GM.completed_cells(d)
    folds = folds_from_file(args.folds) if args.folds else None
    if folds is None:
        print("--execute needs --folds", file=sys.stderr)
        return 2
    cells = GM.plan_matrix(folds, temporal_cutoff_day=args.temporal_cutoff_day)
    rows = []
    for arm in (args.arms.split(",") if args.arms else DEFAULT_ARMS):
        for cell in cells:
            if not cell.needs_retrain:
                continue
            for seed in [int(s) for s in str(args.seeds).split(",")]:
                if (arm, cell.cell_id, seed) in done:
                    print(f"[skip] {arm} {cell.cell_id} seed {seed} (artifact present)")
                    continue
                if args.dry_run:
                    print(f"[dry-run] would train {arm} {cell.cell_id} seed {seed}")
                    continue
                t0 = time.time()
                out = _run_cell(args, arm=arm, cell=cell, seed=seed)
                rows.append({"arm": arm, "cell": cell.cell_id, "seed": seed,
                             "seconds": round(time.time() - t0, 1), "ok": out})
                print(json.dumps(rows[-1]))
    return 0


def _run_cell(args, *, arm: str, cell: GM.MatrixCell, seed: int) -> bool:
    """Train then evaluate one cell, through the canonical scripts.

    ``train_patch.py`` takes its field selection from the STORE (minus the
    holdout manifest) and names its own output directory from
    ``<out>/<arch>_seed<seed>/``; it has no ``--exclude-fields`` flag. The fold's
    held-out fields are therefore excluded by BUILDING A FOLD-SPECIFIC STORE
    VIEW: ``--out`` is a per-cell directory AND the fold's field ids are passed
    through ``--holdout-manifest``-independent filtering by writing a partial
    store. Since that is a data operation, ``--execute`` requires the fold
    store to exist and says exactly how to make it, rather than silently
    training on the held-out blocks.

    This is the honest half-implementation: the plan/estimate/report modes are
    complete, and ``--execute`` refuses until the per-fold stores exist.
    """
    fold_store = Path(args.fold_store_dir) / f"fold_{cell.fold_id}.zarr"
    if not fold_store.exists():
        print(
            f"[blocked] {arm} {cell.cell_id}: needs a fold-specific store at "
            f"{fold_store} (train on blocks {cell.train_blocks}, exclude "
            f"{cell.held_out_blocks}). Build it with a filter over patches.zarr; "
            "training on the full store would leak the held-out blocks.",
            file=sys.stderr,
        )
        return False
    run_id = f"gen_{arm}_{cell.cell_id}_s{seed}"
    out_dir = tmp_out_dir(args, run_id)
    train_cmd = [
        sys.executable, str(ROOT / "scripts" / "train_patch.py"),
        "--store", str(fold_store), "--seed", str(seed),
        "--arch", "patch_cnn_gru" if arm == "patch_cnn_gru" else "patch_pixelwise",
        "--out", str(out_dir),
    ]
    if subprocess.run(train_cmd, cwd=ROOT).returncode != 0:
        return False
    eval_cmd = [
        sys.executable, str(ROOT / "scripts" / "eval_dev.py"),
        # --run-id names the DEV checkpoint dir; the patch arm reads its own
        # artifacts from --patch-run (defaulting to --run-id), so both are the
        # same run id here.
        "--run-id", run_id, "--model", "patch_cnn_gru", "--scorer", arm,
        "--seeds", str(seed), "--patch-store", str(fold_store),
    ]
    return subprocess.run(eval_cmd, cwd=ROOT).returncode == 0


def tmp_out_dir(args, run_id: str) -> Path:
    return Path(args.dir) / "runs" / run_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--estimate-runtime", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--folds", default=None,
                    help=f"frozen fold file (default for --estimate/--execute: "
                         f"{DEFAULT_FOLDS_FILE})")
    ap.add_argument("--build-folds", default=None, metavar="FIELDS_PARQUET",
                    help="build the fold definition from this parquet (needs block "
                         "and x/y columns) and, with --out, freeze it")
    ap.add_argument("--out", default=str(DEFAULT_FOLDS_FILE),
                    help="--plan writes the frozen fold file here")
    ap.add_argument("--dir", default=str(DEFAULT_ARTIFACT_DIR))
    ap.add_argument("--fold-store-dir", default=str(ROOT / "data" / "processed" / "folds"),
                    help="per-fold filtered stores (<dir>/fold_<k>.zarr); "
                         "--execute refuses without them")
    ap.add_argument("--temporal-cutoff-day", type=int, default=None)
    ap.add_argument("--seeds", default="13")
    ap.add_argument("--arms", default=None)
    # runtime-estimate inputs (no training happens in that mode)
    ap.add_argument("--seconds-per-epoch", type=float, default=120.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seconds-per-eval", type=float, default=180.0)
    ap.add_argument("--seeds-screening", type=int, default=1)
    ap.add_argument("--seeds-final", type=int, default=3)
    ap.add_argument("--n-arms-screening", type=int, default=6)
    ap.add_argument("--n-arms-final", type=int, default=2)
    # report inputs
    ap.add_argument("--dev-se", type=float, default=None,
                    help="SE of the dev MDM80 estimate; required for verdicts")
    ap.add_argument("--in-block", type=float, default=None,
                    help="in-block MDM80 sentinel for the temporal comparison")
    args = ap.parse_args(argv)

    if args.plan:
        return cmd_plan(args)
    if args.estimate_runtime:
        return cmd_estimate(args)
    if args.report:
        return cmd_report(args)
    if args.execute:
        return cmd_execute(args)
    ap.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

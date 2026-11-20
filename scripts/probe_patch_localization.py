"""G4 localization probe — pixel-localization AUPRC at footprint 0.3.

One patch checkpoint per invocation (like eval_dev's patch path), over the
finding-18 ladder grid: S1–S6 x {acute, persistent} x {ndvi, savi} on the SAME
hash-partitioned evaluation subset as the detection sweep
(``partition_calib_eval`` on the locked dev cycle keys). For each cell the
injection covers a contiguous in-field blob of ``LOCALIZATION_FOOTPRINT``
(default 0.3) of the field's pixels, and the metric is the pixel-localization
AUPRC of the scorer's own per-pixel sigma-table z² over the injection windows
(``patch_eval.localization_auprc``). No empirical-p calibration, no
threshold, no retraining: the ranking uses exactly the score map the alarm
path aggregates.

Pre-registered G4 tie-break (openspec ``patch-level-controlled-trial``
tasks.md, written before any PoC result): "CNN kept only if it beats
patch_pixelwise on MDM80 or localization AUPRC". Finding 23 tied MDM80 at S1
for both patch arms, so this probe decides G4. The aggregation rule applied
by the caller (recorded here before any probe result, for the record):

* arm cell score = mean localization AUPRC over the 24 cells (cells whose
  AUPRC is undefined — no scored positives/negatives — are excluded and the
  exclusion count is reported);
* arm score = mean of the cell scores over the arm's 3 seeds;
* G4 verdict: patch_cnn_gru is kept iff its arm score is strictly higher
  than patch_pixelwise's; the gap is reported verbatim and flagged
  "marginal" below 0.01 AP.

Rounds are fixed at 1: the contiguous footprint is deterministic given the
in-field mask (the seeded rng only drives the "white" diagnostic null, which
this probe does not run).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ssl_anomaly.io import patches as P  # noqa: E402
from ssl_anomaly.monitor import holdout as mholdout  # noqa: E402
from ssl_anomaly.monitor import inputs as minputs  # noqa: E402
from ssl_anomaly.monitor import patch_eval as pE  # noqa: E402
from ssl_anomaly.monitor import patch_scores as mpscores  # noqa: E402
from ssl_anomaly.monitor import protocol as mproto  # noqa: E402
from ssl_anomaly.monitor import severity_table as sev  # noqa: E402
from ssl_anomaly.monitor import splits as msplits  # noqa: E402


def _load_eval_dev():
    spec = importlib.util.spec_from_file_location("eval_dev", REPO_ROOT / "scripts" / "eval_dev.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_dev"] = mod
    spec.loader.exec_module(mod)
    return mod


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            capture_output=True, timeout=10,
        ).stdout.strip()[:8]
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default=str(REPO_ROOT / "data" / "processed" / "patches_poc_eval.zarr"))
    ap.add_argument("--splits", default=str(REPO_ROOT / "data" / "processed" / "splits_locked.json"))
    ap.add_argument("--holdout-manifest", default=str(mholdout.HOLDOUT_MANIFEST_PATH))
    ap.add_argument("--scorer", required=True, choices=["patch_cnn_gru", "patch_pixelwise"])
    ap.add_argument("--checkpoint", required=True, type=Path, help="path to last.pt")
    ap.add_argument("--footprint", type=float, default=pE.LOCALIZATION_FOOTPRINT)
    ap.add_argument("--injection-basis", default="bands_consistent",
                    choices=["index_only", "bands_consistent"],
                    help="injector semantics (change consistent-spectral-injection); "
                         "the primary localization probe runs bands_consistent")
    ap.add_argument("--shapes", default="acute,persistent")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "reports" / "dev" / "patch_localization_probe"))
    args = ap.parse_args(argv)

    E = _load_eval_dev()
    t0 = time.time()
    device = args.device

    manifest = msplits.load_splits_locked(path=Path(args.splits), verify_cycle_files=False)
    root = P.open_store(Path(args.store))
    epoch = np.datetime64(P.validate_locked_patch_store(root, manifest), "D")
    holdout = mholdout.load_holdout_manifest(path=Path(args.holdout_manifest))
    mholdout.assert_frozen_holdout_matches_splits(holdout, manifest)

    # identical dev selection + hash partition as the detection sweep
    data = minputs.load_masked_cycles()
    sel = mproto.select_splits(data, manifest, splits=("dev",))
    cycle_keys = list(sel["cycle_keys"])
    msplits.assert_no_test_cycles(cycle_keys, manifest, context="patch dev evaluation")
    if not set(cycle_keys) <= set(manifest["cycles"]["dev"]):
        raise ValueError("patch dev evaluation only accepts locked dev cycle keys")
    index = P.patch_cycle_index(root)
    for key, start, end in zip(cycle_keys, sel["start_dates"], sel["end_dates"], strict=True):
        cycle = index.get(key)
        if cycle is None:
            raise ValueError(f"requested dev patch cycle {key} is not in the store")
        if (epoch + np.timedelta64(cycle.start_day, "D") != np.datetime64(start, "D")
                or epoch + np.timedelta64(cycle.end_day + 1, "D") != np.datetime64(end, "D")):
            raise ValueError(f"patch bounds for {key} disagree with the locked dev cycle")
    fids = [index[key].field_id for key in cycle_keys]
    cids = [index[key].cycle_id for key in cycle_keys]

    model, sigma, norm, cfg = E.load_patch_artifacts(
        "patch_localization_probe", checkpoint_path=args.checkpoint)
    if bool(cfg["use_cnn"]) is not (args.scorer == "patch_cnn_gru"):
        raise ValueError(f"checkpoint use_cnn={cfg['use_cnn']} does not match scorer {args.scorer}")
    P.validate_patch_norm_binding(root, norm, manifest, holdout)
    batch = pE.patch_batch_for_cycles(root, fids, cids, norm=norm)

    calib_idx, eval_idx, eval_hash = E.partition_calib_eval(cycle_keys)
    print(f"[g4-probe] {len(cycle_keys)} dev cycles; evaluation subset n={len(eval_idx)} "
          f"(hash {eval_hash} — the detection sweep's subset)")

    eval_batch = {key: value[eval_idx] if isinstance(value, np.ndarray) else value
                  for key, value in batch.items()}
    # build the scorer AFTER slicing so its per-sample in_field table is
    # aligned with the batch it scores (the detection sweep instead passes
    # idx into every call; for this probe the scorer only needs the slice)
    scorer = E.build_patch_arm(args.scorer, model, cfg, device, sigma=sigma,
                               in_field=eval_batch["in_field"])

    levels = sev.load_table()
    families = [f.strip() for f in str(args.shapes).split(",") if f.strip()]
    cells: dict[str, dict] = {}
    order: list[str] = []
    n_undefined = 0
    for lv in levels:
        for fam in families:
            for index_name in ("ndvi", "savi"):
                key = f"{lv.level_id}/{fam}/{index_name}"
                cell = pE.patch_localization_scores(
                    scorer, eval_batch, level_id=lv.level_id, family=fam,
                    index=index_name, round_idx=0, footprint=float(args.footprint),
                    injection_basis=args.injection_basis,
                )
                m = cell["metrics"]
                if m["auprc"] is None:
                    n_undefined += 1
                cells[key] = {
                    "auprc": m["auprc"], "prevalence": m["prevalence"],
                    "n_pos": m["n_pos"], "n_neg": m["n_neg"],
                    "delta_ndvi": float(lv.delta_ndvi),
                    "footprint_requested": cell["footprint_requested"],
                    "footprint_realized": cell["footprint_realized"],
                    "onset_bins": int(cell["onset_bins"][0]),
                    "duration_bins": int(cell["duration_bins"]),
                }
                order.append(key)
                a = "n/a" if m["auprc"] is None else f"{m['auprc']:.4f}"
                print(f"[g4-probe] {key}: auprc={a} prev="
                      f"{'n/a' if m['prevalence'] is None else format(m['prevalence'], '.4f')}")

    defined = [c["auprc"] for c in cells.values() if c["auprc"] is not None]
    pooled = float(np.mean(defined)) if defined else None
    report = {
        "protocol": "patch_localization_probe_v1",
        "scorer": args.scorer,
        "injection_basis": args.injection_basis,
        "scoring_alignment": mpscores.alignment_for_protocol(cfg.get("protocol")),
        "ladder_rung": "L2",
        "claim_noun": "synthetic anomaly",
        "seed": int(cfg["seed"]),
        "footprint": float(args.footprint),
        "rounds": 1,
        "rounds_note": "contiguous footprint is deterministic; rng only drives the white null",
        "g4_rule": "CNN kept iff mean cell AUPRC (over seeds) strictly beats patch_pixelwise; "
                   "gap < 0.01 flagged marginal",
        "pooled_cell_auprc": pooled,
        "n_cells": len(cells),
        "n_cells_undefined": n_undefined,
        "params": sum(p.numel() for p in model.parameters()),
        "checkpoint": str(args.checkpoint),
        "patch_sigma_sha256": sigma.sha256,
        "patch_norm_sha256": norm["sha256"],
        "splits_locked_sha256": manifest["manifest_sha256"],
        "holdout_sha256": holdout["manifest_sha256"],
        "eval_subset_sha256": eval_hash,
        "severity_table_sha256": sev.severity_table_hash(),
        "store": str(args.store),
        "store_patch_size": int(root.attrs.get("patch_size", -1)),
        "n_eval_cycles": len(eval_idx),
        "git_sha": git_sha(),
        "device": device,
        "wall_time_s": round(time.time() - t0, 1),
        "cells_order": order,
        "cells": cells,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"localization_{args.scorer}_s{cfg['seed']}.json"
    with out_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(f"[done] {out_path} (wall {report['wall_time_s']}s, "
          f"pooled_cell_auprc={pooled})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

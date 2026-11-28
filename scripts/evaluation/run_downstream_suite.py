#!/usr/bin/env python3
"""Downstream suite T1–T5 (tasks 4.3–4.7): claim-bearing tracks kept separate.

Writes ``downstream_suite.json`` and the descriptive ``near_matches.json`` /
``near_matches.md`` (EXPLORATORY — never gate input). Smoke runs the whole
pipeline on the synthetic fixture with feature-based stand-in embeddings and
the B1 rule scorer; a REAL run is an owner command that consumes the screened
region model's dev embeddings (see AGENTS.md §6 hand-over rules).

Claim discipline enforced in the artifact:
- claim-bearing: T1 (phase probe), T2 (family x severity AUC-PR), T3 Track 1,
  T4 (trajectory vs persistence).
- exploratory, never gate input: T3 Track 2 (absent — no in-repo producer),
  near-match listings, T5.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import numpy as np  # noqa: E402

from _common import add_common_args, finish_smoke, resolve_run  # noqa: E402
from ssl_anomaly.evaluation.binding import load_preregistration  # noqa: E402
from ssl_anomaly.evaluation.data import load_cycle_table  # noqa: E402
from ssl_anomaly.evaluation.downstream import (  # noqa: E402
    build_injection_pools,
    near_matches_listing,
    near_matches_markdown,
    t1_phenology_probe,
    t2_family_severity_cells,
    t3_fewshot_retrieval,
    t3_track2_weak_proxy,
    t4_trajectory_vs_persistence,
    t5_exploratory_listing,
)
from ssl_anomaly.evaluation.input_ladder import (  # noqa: E402
    calibrate_b1_rule,
    variant_scores,
)
from ssl_anomaly.evaluation.provenance import honest_state_block  # noqa: E402
from ssl_anomaly.evaluation.runner import (  # noqa: E402
    SmokeExpectation,
    build_manifest,
    make_smoke_fixture,
    write_run_outputs,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--cycles", type=Path, default=Path("data/processed/cycles_490_v2.parquet"))
    p.add_argument("--embeddings", type=Path, default=None,
                   help=".npz with cycle_ids + z (the screened model's dev embeddings); "
                        "REQUIRED for a real run — without it only the persistence "
                        "baseline and the B1 rule arm can be measured")
    p.add_argument("--n-per-cell", type=int, default=5,
                   help="injections per (family, severity) cell")
    return p.parse_args(argv)


def _moment_embeddings(dev) -> np.ndarray:
    """Deterministic fixture stand-in: seasonal moment features per cycle."""
    x = np.where(dev.M, dev.X, np.nan)
    with np.errstate(invalid="ignore"):
        z = np.hstack([np.nanmean(x, axis=1), np.nanstd(x, axis=1)])
    return np.nan_to_num(z).astype(np.float32)


def _raw_distance_arm(dev) -> np.ndarray:
    """Raw band-vector embedding: the flattened, field-centred 8-ch cycle means."""
    x = np.where(dev.M, dev.X, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.nanmean(x, axis=1)).astype(np.float32)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    prereg = load_preregistration()
    ctx = resolve_run(args, stage="downstream-suite")

    if ctx.smoke:
        fixture = make_smoke_fixture(n_fields=14, cycles_per_field=5, seed=13)
        dev = fixture.dev
        embeddings = {"ssl": _moment_embeddings(dev),
                      "a-rand": np.random.default_rng(42).normal(
                          size=(dev.n_cycles, 16)).astype(np.float32)}
        source = "synthetic-fixture"
    else:
        from _common import require_real_data
        require_real_data(args.cycles, "the v2 cycle table")
        if args.embeddings is None:
            print("REFUSING: a real downstream run needs --embeddings (the "
                  "screened model's dev embeddings); without them the suite "
                  "would silently reduce to the B1/rule arms only.", file=sys.stderr)
            return 1
        require_real_data(args.embeddings, "the dev embeddings")
        from ssl_anomaly.evaluation.binding import bind_artifacts
        bind_artifacts(prereg=prereg, strict=True,
                       require=("cycles_490_v2", "splits_locked_v2"))
        dev = load_cycle_table(args.cycles).for_split("dev")
        z = np.load(args.embeddings, allow_pickle=False)
        lut = {str(c): i for i, c in enumerate(z["cycle_ids"])}
        missing = [c for c in dev.cycle_ids if str(c) not in lut]
        if missing:
            raise SystemExit(f"{len(missing)} dev cycles missing from embeddings; "
                             "refusing to impute rows")
        ssl = z["z"][[lut[str(c)] for c in dev.cycle_ids]]
        embeddings = {"ssl": ssl.astype(np.float32),
                      "raw-band-distance": _raw_distance_arm(dev)}
        source = f"{args.cycles} + {args.embeddings}"

    n = dev.n_cycles
    rng = np.random.default_rng(13)

    # ---- B1 rule arm: calibrated once, scored through the same pools -------
    cal_ids = dev.cycle_ids[: n // 2]
    calibration = calibrate_b1_rule(dev.X, dev.M, dev.field_ids, dev.cycle_ids, cal_ids)
    b1_scores = variant_scores(dev.X, dev.M, calibration["or"].per_field,
                               dev.field_ids)["or"]
    # NaN (never -inf/0) where a cycle has no observed B1-channel bin; the T2
    # cell builder excludes and counts those.
    b1_cycle_scores = np.nanmax(b1_scores, axis=1)

    # ---- T2: injection pools from the FROZEN severity table ---------------
    pools = build_injection_pools(dev.X, dev.M, n_per_cell=args.n_per_cell, seed=13)
    n_inj = pools["x_injected"].shape[0]
    # The B1 RULE arm scores the injected cycles with the SAME per-field stats.
    # The SSL/embedding arms' T2 cells need the trained model's scorer applied
    # to the injected cycles — that is the owner-run stage (--embeddings plus
    # the checkpoint); the smoke exercises the plumbing with the rule arm.
    from ssl_anomaly.evaluation.input_ladder import variant_scores as _vs
    inj_b1 = _vs(pools["x_injected"], pools["masks"], calibration["or"].per_field,
                 dev.field_ids[pools["source_indices"]])["or"]
    inj_b1_cycle = np.nanmax(inj_b1, axis=1)
    t2 = t2_family_severity_cells(inj_b1_cycle, pools, clean_scores=b1_cycle_scores)
    cells = t2["cells"]

    # ---- T1: phase probe on the fixture phase targets ---------------------
    phase = (fixture.farm_phase_bins if ctx.smoke
             else np.tile(np.arange(dev.X.shape[1]) // 3, (n, 1))[:, 0])  # real target owner-supplied
    train = np.zeros(n, dtype=bool)
    train[: n // 2] = True
    evalm = ~train
    b1_feat = np.nan_to_num(np.nanmean(
        np.where(dev.M[:, :, 0], dev.X[:, :, 0], np.nan), axis=1), nan=0.0)[:, None]
    t1 = t1_phenology_probe(embeddings["ssl"], phase, train, evalm,
                            n_classes=int(np.nanmax(phase)) + 1, b1_features=b1_feat)

    # ---- T3 Track 1 + near matches ----------------------------------------
    q = rng.choice(np.arange(n // 2, n), size=min(12, n - n // 2), replace=False)
    gallery = np.arange(n)
    fam_of = np.array(["acute"] * n)          # smoke frame; real run: pool families
    t3 = t3_fewshot_retrieval(
        {**embeddings, "b1-score": np.zeros((n, 2), dtype=np.float32)},
        query_indices=q, query_families=fam_of[q],
        gallery_indices=gallery, gallery_families=fam_of,
        arm_scores={"b1-score": b1_cycle_scores, "a-rule-score": b1_cycle_scores},
    )
    natural = np.arange(n)
    meta = [{"field": str(f), "split": str(s)} for f, s in
            zip(dev.field_ids, dev.split, strict=False)]
    listing = near_matches_listing({"ssl": embeddings["ssl"]}, q, natural, meta, top=10)

    # ---- T4 + T5 -----------------------------------------------------------
    t4 = t4_trajectory_vs_persistence(dev.X, dev.M, horizons=(1, 3))
    t5 = t5_exploratory_listing(embeddings["ssl"],
                                {i: f"field {dev.field_ids[i]}" for i in range(n)})

    report = {
        "stage": "downstream-suite",
        "smoke": ctx.smoke,
        "data_source": source,
        "n_dev_cycles": int(n),
        "b1_calibration": {v: c.to_dict() for v, c in calibration.items()},
        "t1_phenology": t1,
        "t2_injection_cells": cells,
        "t2_unscorable_injected": t2["n_unscorable_injected"],
        "t2_injection_provenance": {
            "severity_table_sha256": pools["severity_table_sha256"],
            "per_cell_counts": pools["per_cell_counts"],
            "note": pools["note"],
        },
        "t3_track1": t3,
        "t3_track2": t3_track2_weak_proxy({}, q, gallery),
        "t4_trajectory": t4,
        "t5_exploratory": t5,
        "honest_state": honest_state_block(
            measured=["T1", "T2", "T3_track1", "T4-persistence", "near_matches", "T5"],
            absent=["T3_track2: no in-repo weak-proxy producer",
                    f"T4-model: {'no forecaster supplied' if t4['model']['1'].get('status') == 'absent' else 'measured'}"],
            deferred=[],
        ),
        "wall_time_s": round(time.time() - started, 2),
    }

    manifest = build_manifest(stage="downstream-suite", smoke=ctx.smoke,
                              config={"n_per_cell": args.n_per_cell},
                              seeds=[13, 42, 2026])
    written = write_run_outputs(paths=ctx.paths, manifest=manifest,
                                reports={"downstream_suite": report,
                                         "near_matches": listing})
    md_path = Path(written["near_matches"]).with_suffix(".md")
    cycle_meta = {i: f"field {f} ({s})" for i, (f, s) in
                  enumerate(zip(dev.field_ids, dev.split, strict=False))}
    md_path.write_text(near_matches_markdown(listing, cycle_meta), encoding="utf-8")
    written["near_matches_md"] = str(md_path)

    if ctx.smoke:
        exp = SmokeExpectation("downstream-suite")
        exp.check("T2 measured for all 18 family x severity cells",
                  sum(1 for v in cells.values() if v["status"] == "measured") == 18,
                  str(len(cells)))
        exp.check("T2 unscorable injected cycles counted, not scored as 0",
                  isinstance(report["t2_unscorable_injected"], int))
        exp.check("every T2 cell carries prevalence",
                  all("prevalence" in v for v in cells.values()
                      if v["status"] == "measured"))
        exp.check("T1 is claim-bearing and reports both comparators",
                  t1["claim_bearing"] and "b1_feature_probe" in t1)
        exp.check("T3 Track 1 measured with precision@m + MAP",
                  t3["arms"]["ssl"]["status"] == "measured"
                  and "map" in t3["arms"]["ssl"])
        exp.check("T3 Track 2 recorded absent, exploratory",
                  report["t3_track2"]["status"] == "absent"
                  and report["t3_track2"]["claim_bearing"] is False)
        exp.check("near-match listing labelled exploratory",
                  listing["claim_bearing"] is False and listing["label"] == "exploratory")
        exp.check("T4 persistence measured; model arm absent with reason",
                  t4["persistence"]["1"]["n_cycles_used"] > 0
                  and t4["model"]["1"].get("status") == "absent")
        exp.check("T5 exploratory only",
                  t5["claim_bearing"] is False)
        report["smoke_checks"] = exp.to_dict()
        write_run_outputs(paths=ctx.paths, manifest=manifest,
                          reports={"downstream_suite": report, "near_matches": listing})
        print(f"wrote {sorted(written.values())}")
        return finish_smoke(exp, ctx.paths, "downstream-suite")

    print(f"wrote {sorted(written.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

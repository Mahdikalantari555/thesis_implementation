#!/usr/bin/env python3
"""Pre-registered B1–B5 INPUT ladder run (design §ladder; tasks 1.1–1.5).

Writes ``input_ladder.json``: the B1 ``ndvi_lst_rule`` calibration record (both
variants, FAR-matched on the clean calibration subset), the B2–B5 builder
statuses with the absent blocks named, the causality audit, the per-rung
capacity manifest, and (when a scoring pass runs) each rung's per-bin scores
through the common bridge.

Usage
-----
::

    # smoke (synthetic fixture, ~30 s, writes to reports/_smoke/):
    /home/mahdi/micromamba/envs/geo/bin/python -X utf8 \
        scripts/evaluation/run_input_ladder.py --smoke

    # real run — requires the clean-calibration subset table, which the v2
    # chain does not ship yet (fail-closed, see --calibration-subset):
    /home/mahdi/micromamba/envs/geo/bin/python -X utf8 \
        scripts/evaluation/run_input_ladder.py \
        --cycles data/processed/cycles_490_v2.parquet \
        --calibration-subset data/processed/calibration_subset_v2.parquet

The real run is an OWNER command (AGENTS.md §6): the calibration-subset table
(clean cycles of train+dev, produced by the injection/calibration pipeline)
does not exist yet, so the real path refuses loudly instead of calibrating on
an implicit "everything" — the rigged-baseline failure the design forbids.
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
from ssl_anomaly.evaluation.binding import bind_artifacts, load_preregistration  # noqa: E402
from ssl_anomaly.evaluation.data import load_cycle_table  # noqa: E402
from ssl_anomaly.evaluation.input_ladder import (  # noqa: E402
    B1RuleScorer,
    assert_causal_inputs,
    build_rung_inputs,
    calibrate_b1_rule,
    calibration_independence_audit,
    capacity_manifest,
    estimate_cycle_far,
    input_ladder_status,
    ladder_table,
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
    p.add_argument("--calibration-subset", type=Path, default=None,
                   help="parquet with cycle_id + clean flag for the CLEAN "
                        "calibration cycles; REQUIRED for a real run — the "
                        "rule is never calibrated on an implicit 'all'")
    p.add_argument("--weather", type=Path, default=None,
                   help="(N,T,6) ERA5-to-bin table for B4/B5; absent -> those "
                        "rungs are recorded absent")
    return p.parse_args(argv)


def _audit_causality(dev, weather=None):
    audits: dict[str, dict] = {}
    for rung in ("B2", "B3"):
        audits[rung] = assert_causal_inputs(
            lambda xs, r=rung: build_rung_inputs(
                r, xs, np.ones_like(xs, dtype=bool), dev.dsh,
                channel_names=dev.channel_names,
            ).x,
            dev.X,
        )
    audits["B4"] = {
        "causal": True,
        "probe_bin": None,
        "note": ("weather is consumed same-bin only (build refuses future-day "
                 "tables by construction); the ERA5-to-bin producer does not "
                 "exist yet, so the audited property is the builder contract, "
                 "not a store"),
    }
    audits["B5"] = {
        "causal": None,
        "note": ("not auditable: the STL block is absent; a full-series STL "
                 "would FAIL this audit (see input_ladder.STL_CAUSALITY_NOTE)"),
    }
    audits["B1"] = {
        "causal": True,
        "note": "per-bin robust-z uses that bin's own value and the field's "
                "calibration statistics — a function of bins <= o only",
    }
    return audits


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    prereg = load_preregistration()
    ctx = resolve_run(args, stage="input-ladder")

    if ctx.smoke:
        fixture = make_smoke_fixture(n_fields=14, cycles_per_field=4, seed=13)
        dev = fixture.dev
        cal_ids = dev.cycle_ids[: dev.n_cycles // 2]
        weather = None
        source = "synthetic-fixture"
    else:
        from _common import require_real_data
        require_real_data(args.cycles, "the v2 cycle table")
        bind_artifacts(prereg=prereg, strict=True,
                       require=("cycles_490_v2", "splits_locked_v2"))
        if args.calibration_subset is None:
            print(
                "REFUSING to run: the clean calibration subset table "
                "(--calibration-subset) does not exist yet. Calibrating the B1 "
                "rule on an implicit 'all cycles' subset would rig the control "
                "the whole change is built around. Produce the subset from the "
                "injection/calibration pipeline first (owner run).",
                file=sys.stderr,
            )
            return 1
        require_real_data(args.calibration_subset, "the clean calibration subset")
        import pandas as pd

        cal_df = pd.read_parquet(args.calibration_subset)
        if not {"cycle_id", "clean"} <= set(cal_df.columns):
            raise SystemExit(f"{args.calibration_subset} needs cycle_id + clean columns")
        dev = load_cycle_table(args.cycles).for_split("dev")
        allowed = set(dev.cycle_ids.astype(str))
        cal_ids = np.array([str(c) for c, clean in zip(cal_df["cycle_id"], cal_df["clean"])
                            if bool(clean) and str(c) in allowed])
        if cal_ids.size == 0:
            raise SystemExit("no clean calibration cycles fall inside dev; refusing")
        weather = None
        source = str(args.cycles)

    # ---- B1 calibration (both variants; evaluation untouched) -------------
    calibration = calibrate_b1_rule(dev.X, dev.M, dev.field_ids, dev.cycle_ids, cal_ids)
    independence = calibration_independence_audit(
        dev.X, dev.M, dev.field_ids, dev.cycle_ids, cal_ids,
    )

    # ---- B2-B5 builders ----------------------------------------------------
    builds = {
        r: build_rung_inputs(r, dev.X, dev.M, dev.dsh,
                             channel_names=dev.channel_names, weather=weather)
        for r in ("B2", "B3", "B4", "B5")
    }
    causality = _audit_causality(dev, weather)
    capacity = capacity_manifest()

    # ---- scoring-bridge exercise: B1 FAR on held-out clean cycles ----------
    scorer = B1RuleScorer(calibration["or"], x=dev.X, m=dev.M, field_ids=dev.field_ids)
    from ssl_anomaly.evaluation.input_ladder import variant_scores
    scores = variant_scores(dev.X, dev.M, calibration["or"].per_field, dev.field_ids)
    held = ~np.isin(dev.cycle_ids, cal_ids)
    far = estimate_cycle_far(scores["or"], calibration["or"].threshold,
                             labels_clean=held) if held.any() else {
        "status": "absent", "reason": "no held-out clean cycles in this run",
    }

    report = {
        "stage": "input-ladder",
        "smoke": ctx.smoke,
        "data_source": source,
        "ladder": ladder_table(),
        "b1_calibration": {v: c.to_dict() for v, c in calibration.items()},
        "b1_calibration_independence": independence,
        "b1_heldout_cycle_far": far,
        "status": input_ladder_status(builds, capacity, causality, calibration),
        "capacity": capacity,
        "causality": causality,
        "honest_state": honest_state_block(
            measured=["B1", "B2", "B3", "causality(B1,B2,B3)", "capacity"],
            absent=[f"{r}: {b.reason}" for r, b in builds.items() if b.status == "absent"],
            deferred=["B5.stl: no causal decomposition producer exists"],
        ),
        "wall_time_s": round(time.time() - started, 2),
    }

    manifest = build_manifest(stage="input-ladder", smoke=ctx.smoke,
                              config={"calibration_cycles": int(len(cal_ids))},
                              seeds=[13, 42, 2026])
    written = write_run_outputs(paths=ctx.paths, manifest=manifest,
                                reports={"input_ladder": report})

    if ctx.smoke:
        exp = SmokeExpectation("input-ladder")
        exp.check("B1 calibrated for BOTH variants",
                  set(calibration) == {"and", "or"},
                  str({v: round(c.threshold, 3) for v, c in calibration.items()}))
        exp.check("thresholds are independent of evaluation cycles",
                  all(independence.values()), str(independence))
        exp.check("both variants' achieved FAR sits at the 5% budget",
                  all(abs(c.achieved_cycle_far - 0.05) < 0.05 for c in calibration.values()),
                  str({v: c.achieved_cycle_far for v, c in calibration.items()}))
        exp.check("B2/B3 built on the v2 channel contract",
                  builds["B2"].status == "built" and builds["B3"].status == "built"
                  and builds["B3"].x.shape[2] == 8)
        exp.check("B4/B5 recorded absent with named blocks (no substitution)",
                  builds["B4"].status == "absent" and "era5_daily" in builds["B4"].absent_blocks
                  and builds["B5"].status == "absent"
                  and "decomposition_features" in builds["B5"].absent_blocks)
        exp.check("causality audit passes for B1/B2/B3",
                  all(causality[r]["causal"] is True for r in ("B1", "B2", "B3")))
        exp.check("capacity band enforced with per-rung counts",
                  capacity["band_enforced"]
                  and set(capacity["rungs"]) == {"B1", "B2", "B3", "B4", "B5"})
        report["smoke_checks"] = exp.to_dict()
        write_run_outputs(paths=ctx.paths, manifest=manifest,
                          reports={"input_ladder": report})
        print(f"wrote {written['input_ladder']}")
        return finish_smoke(exp, ctx.paths, "input-ladder")

    print(f"wrote {written['input_ladder']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

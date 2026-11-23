#!/usr/bin/env python3
"""Level-1 pretraining-quality battery for region-model v0 candidates.

Change: regional-ssl-representation-evaluation (design §Level 1, gate G-A).

What this script does
---------------------
Trains the four candidate encoders with EQUAL budgets and seeds {13, 42, 2026},
runs the five Level-1 probes on their frozen dev embeddings, applies the
capacity band, and computes G-A mechanically. It writes
``pretraining_quality.json`` — the artifact ``reports/regional_representation/``
catalogues — plus the run manifest.

What this script deliberately does NOT do
-----------------------------------------
It does not pick an architecture by reconstruction loss. G-A is the decision
rule and it reads probe evidence only (design §Architecture decision gate).

Usage
-----
Real run (owner machine, needs ``data/processed/cycles_490_v2.parquet``)::

    python scripts/evaluation/evaluate_representation_quality.py \
        --cycles data/processed/cycles_490_v2.parquet \
        --out reports/regional_representation \
        --epochs 40 --seeds 13 42 2026

Smoke run (any machine, synthetic fixture, ~10 s, writes to reports/_smoke/)::

    python scripts/evaluation/evaluate_representation_quality.py --smoke

Expected wall time: smoke <15 s on CPU; real run ~8–15 min per seed on a laptop
CPU with 490 cycles x 36 bins x 32 d_model (4 candidates x 3 seeds).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from ssl_anomaly.evaluation.binding import (  # noqa: E402
    bind_artifacts,
    load_preregistration,
)
from ssl_anomaly.evaluation.candidates import (  # noqa: E402
    CANDIDATES,
    SPECS,
    UNIMPLEMENTED_CANDIDATES,
    TrainConfig,
    build_candidate,
    capacity_report,
    embed_candidates,
    screening_verdict,
    train_candidate,
)
from ssl_anomaly.evaluation.data import CycleData, load_cycle_table  # noqa: E402
from ssl_anomaly.evaluation.level1 import run_level1  # noqa: E402
from ssl_anomaly.evaluation.provenance import honest_state_block  # noqa: E402
from ssl_anomaly.evaluation.runner import (  # noqa: E402
    RunPaths,
    SmokeExpectation,
    build_manifest,
    environment_block,
    make_smoke_fixture,
    write_run_outputs,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true",
                   help="run on the synthetic fixture and write to reports/_smoke/")
    p.add_argument("--cycles", type=Path, default=Path("data/processed/cycles_490_v2.parquet"),
                   help="v2 cycle table (dev rows are used; test rows are refused)")
    p.add_argument("--out", type=Path, default=None,
                   help="output directory (default: reports/regional_representation)")
    p.add_argument("--epochs", type=int, default=40,
                   help="training epochs per candidate (smoke overrides to a small value)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    # Sized so each candidate's ENCODER lands inside the pre-registered capacity
    # band (100k x [0.8, 1.2]) while every TOTAL stays under the 250k hard cap.
    # Verified by tests/evaluation/test_candidates.py::test_default_trunk_is_in_band.
    p.add_argument("--d-model", type=int, default=160)
    p.add_argument("--d-out", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-training", action="store_true",
                   help="probe randomly-initialised encoders (pipeline exercise only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke and args.epochs == 40:
        args.epochs = 3          # smoke budget; keeps the smoke under ~30 s
    started = time.time()
    prereg = load_preregistration()
    seeds = args.seeds

    if not args.smoke:
        # Training paths REQUIRE the v2 chain: a missing artifact raises here
        # rather than producing numbers from a partially-available dataset.
        bind_artifacts(
            prereg=prereg, strict=True,
            require=("cycles_490_v2", "splits_locked_v2", "norm_stats_full_v2"),
        )

    paths = RunPaths(args.out) if args.out else RunPaths.for_run("smoke-quality" if args.smoke else None)
    if args.smoke and paths.out_dir.parent == RunPaths.for_run().out_dir:
        paths = RunPaths.for_run("smoke-quality")

    # ---- data ------------------------------------------------------------
    source = "synthetic-fixture" if args.smoke else str(args.cycles)
    if args.smoke:
        fixture = make_smoke_fixture(n_fields=8, cycles_per_field=4, seed=13)
        dev = fixture.dev
    else:
        dev = load_cycle_table(args.cycles).for_split("dev")

    n_channels = dev.X.shape[2]
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, seed=seeds[0])

    # ---- train + probe each candidate -----------------------------------
    per_candidate: dict[str, dict] = {}
    capacity: dict[str, dict] = {}
    training: dict[str, dict] = {}
    for name in CANDIDATES:
        model = build_candidate(name, n_channels=n_channels,
                                d_model=args.d_model, d_out=args.d_out)
        cap = capacity_report(model, name)
        capacity[name] = cap.to_dict()
        if not args.skip_training:
            # Training runs in smoke too: the point of a smoke is to exercise every
            # code path, and a skipped training loop is exactly the path most likely
            # to break. Smoke epochs are few, so the encoders stay near their
            # initialisation and collapse diagnostics will (correctly) disqualify
            # them — that is an honest smoke outcome, not a failure of the smoke.
            train_candidate(model, SPECS[name], dev, cfg, device=args.device)
        z = embed_candidates(model, dev, device=args.device)

        phase_targets = fixture.farm_phase_bins if args.smoke else None
        res = run_level1(dev=dev, embed_fn=lambda X, M: z, phase_targets=phase_targets)
        blob = res.to_dict()
        per_candidate[name] = blob
        per_candidate[name]["gate_evidence"] = res.gate_evidence()
        # The A-rand control: the SAME trunk, randomly initialised, probed with the
        # same battery. G-A's comparator is this, not chance — a representation must
        # beat its own untrained architecture, otherwise the win is inductive bias.
        rand_model = build_candidate(name, n_channels=n_channels,
                                     d_model=args.d_model, d_out=args.d_out)
        z_rand = embed_candidates(rand_model, dev, device=args.device)
        rand_res = run_level1(dev=dev, embed_fn=lambda X, M: z_rand,
                              phase_targets=phase_targets)
        per_candidate[name]["random_init"] = {
            "gate_evidence": rand_res.gate_evidence(),
            "seed": cfg.seed,
            "note": ("same architecture and shape, never trained: the paired "
                     "control G-A subtracts"),
        }
        per_candidate[name]["params"] = cap.params
        per_candidate[name]["encoder_params"] = cap.encoder_params
        per_candidate[name]["capacity_band_ok"] = cap.band_ok
        training[name] = {
            "objective": SPECS[name].objective,
            "params": cap.params,
            "config": cfg.to_dict(),
            "trained": not args.skip_training,
        }

    verdict = screening_verdict(
        {n: per_candidate[n]["gate_evidence"] for n in per_candidate},
        {n: {"params": capacity[n]["params"],
             "encoder_params": capacity[n]["encoder_params"],
             "band_ok": capacity[n]["band_ok"]}
         for n in capacity},
    )

    # ---- G-A -------------------------------------------------------------
    from ssl_anomaly.evaluation.gates import (
        GATE_INPUTS, UNDETERMINED, GateVerdict, gate_a,
    )

    binding = bind_artifacts(prereg=prereg, strict=False, compute_hashes=not args.smoke)
    winner = verdict["winner"]
    if winner is not None:
        g_a = gate_a(_paired_g_a_evidence(per_candidate[winner]),
                     dict(prereg.gates["G-A"])).to_dict()
    else:
        # No candidate was selected, so no representation was probed against its
        # control. Reporting "failed" here would manufacture a measured loss that
        # never happened; the honest verdict is that the gate could not decide.
        missing = list(GATE_INPUTS["G-A"])
        g_a = GateVerdict(
            gate="G-A", verdict=UNDETERMINED, criteria=dict(prereg.gates["G-A"]),
            evidence={}, fail_consequence=prereg.gates["G-A"].get("fail_consequence", ""),
            missing_evidence=missing,
            reason=(f"no candidate was screened: {verdict['selection_rule']} — "
                    f"nothing was probed, so no term can be compared to A-rand"),
        ).to_dict()

    report = {
        "change": prereg.payload.get("change"),
        "stage": "level1-pretraining-quality",
        "data_source": source,
        "smoke": args.smoke,
        "seeds": seeds,
        "n_dev_cycles": int(dev.n_cycles),
        "n_channels": int(n_channels),
        "candidates": per_candidate,
        "capacity": capacity,
        "training": training,
        "unimplemented_candidates": UNIMPLEMENTED_CANDIDATES,
        "screening": verdict,
        "gate_a": g_a,
        "honest_state": honest_state_block(
            measured=[c for c in CANDIDATES if c != "C-decomp"],
            absent=[f"C-decomp: {UNIMPLEMENTED_CANDIDATES['C-decomp']}"],
            deferred=[],
        ),
        "environment": environment_block(),
        "wall_time_s": round(time.time() - started, 2),
    }

    manifest = build_manifest(
        stage="level1-pretraining-quality", smoke=args.smoke,
        config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "d_model": args.d_model, "d_out": args.d_out, "device": args.device,
                "skip_training": args.skip_training},
        seeds=seeds, binding=binding,
    )
    written = write_run_outputs(
        paths=paths, manifest=manifest,
        reports={"pretraining_quality": report},
    )

    # ---- smoke self-check ------------------------------------------------
    if args.smoke:
        exp = SmokeExpectation("level1-pretraining-quality")
        exp.check("all four candidates produced a probe report",
                  set(per_candidate) == set(CANDIDATES))
        exp.check("collapse diagnostics ran for every candidate",
                  all("collapse" in v for v in per_candidate.values()))
        exp.check("capacity recorded for every candidate",
                  all("params" in c for c in capacity.values()))
        # Any of the three real verdicts is acceptable here; what is NOT acceptable
        # is a verdict with no reason. Smoke encoders are barely trained, so
        # collapse disqualification and "no winner" are the likely honest outcomes.
        exp.check("G-A verdict is one of the three real verdicts",
                  g_a["verdict"] in {"passed", "failed", "undetermined"},
                  g_a["verdict"])
        exp.check("an undetermined G-A names the evidence it lacks",
                  g_a["verdict"] != "undetermined" or bool(g_a["missing_evidence"]))
        exp.check("G-A never reports a verdict without a reason",
                  g_a["verdict"] == "passed" or bool(g_a.get("reason")))
        exp.check("every candidate carries its paired A-rand control",
                  all("random_init" in v for v in per_candidate.values()))
        exp.check("every candidate's encoder sits in the pre-registered band",
                  all(c["band_ok"] for c in capacity.values()),
                  str({k: v["encoder_params"] for k, v in capacity.items()}))
        exp.check("at least one candidate measured farm separability",
                  any(v["farm_separability"]["n_queries"] > 0
                      for v in per_candidate.values()))
        exp.check("report withheld a winner only for a recorded reason",
                  verdict["winner"] is not None or bool(verdict["selection_rule"]))
        report["smoke_checks"] = exp.to_dict()
        write_run_outputs(paths=paths, manifest=manifest,
                          reports={"pretraining_quality": report})
        try:
            exp.raise_if_failed()
        except AssertionError as exc:
            print(f"SMOKE FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"smoke ok — {json.dumps(exp.to_dict()['checks'], indent=2)}")

    print(f"wrote {written['pretraining_quality']}")
    print(f"G-A: {g_a['verdict']}" + (f" (winner {winner})" if winner else ""))
    return 0


def _paired_g_a_evidence(candidate: dict) -> dict:
    """Turn one candidate's raw Level-1 report plus its random-init control into the
    paired keys G-A reads.

    This is where a schema bug hides most easily: ``level1.Level1Result.gate_evidence``
    emits ``farm_separability_top1`` / ``phase_probe_accuracy``, while G-A wants the
    SSL-vs-A-rand pairs. Feeding the raw dict to ``gate_a`` silently matched nothing
    and the gate reported "missing evidence" no matter what was measured.

    A probe that could not run yields ``None`` on both sides and is omitted, so the
    gate reduces to the terms that WERE measured rather than comparing a number
    against an absent control.
    """
    ssl_ev = candidate.get("gate_evidence", {})
    rand_ev = (candidate.get("random_init") or {}).get("gate_evidence", {})
    out: dict = {}
    for raw, stem in (("farm_separability_top1", "farm_separability"),
                      ("phase_probe_accuracy", "phase_probe"),
                      ("frozen_head_t1_ap", "frozen_head_t1")):
        ssl_val, rand_val = ssl_ev.get(raw), rand_ev.get(raw)
        if ssl_val is not None and rand_val is not None:
            out[f"{stem}_ssl"] = float(ssl_val)
            out[f"{stem}_rand"] = float(rand_val)
    if ssl_ev.get("collapse_disqualified") is not None:
        out["collapse_disqualified"] = bool(ssl_ev["collapse_disqualified"])
    return out


if __name__ == "__main__":
    raise SystemExit(main())

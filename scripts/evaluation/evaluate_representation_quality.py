#!/usr/bin/env python3
"""Level-1 pretraining-quality battery + candidate SCREENING (tasks 2.x/3.x).

Change: regional-ssl-representation-evaluation (design §Level 1, gate G-A).

What this script does
---------------------
Trains the candidate encoders with EQUAL budgets over ALL THREE pre-registered
seeds {13, 42, 2026}, runs the five Level-1 probes on their frozen dev
embeddings, applies the capacity band, and computes G-A mechanically. It writes
``pretraining_quality.json`` AND ``screening_report.json`` (task 3.5).

The candidate roster is the four pre-registered candidates PLUS the research
plan's Level A/C/E arms folded in additively (change tasks.md §3.4):
``C-stat`` (non-learning statistical reference, probed but NEVER selectable as
region model v0), ``C-patch`` (patched transformer, sized inside the frozen
capacity band), ``C-film`` (frozen base + per-farm FiLM adapters).

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

Smoke run (any machine, synthetic fixture, writes to reports/_smoke/)::

    python scripts/evaluation/evaluate_representation_quality.py --smoke

Expected wall time: smoke <90 s on CPU (ONE seed — recorded budget note); the
real run aggregates all three seeds and is an OWNER command per AGENTS.md §6
(~8-15 min per seed per candidate on a laptop CPU).
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
    CandidateSpec,
    TrainConfig,
    build_candidate,
    capacity_report,
    embed_candidates,
    screening_verdict,
    train_candidate,
)
from ssl_anomaly.evaluation.data import load_cycle_table  # noqa: E402
from ssl_anomaly.evaluation.level1 import run_level1  # noqa: E402
from ssl_anomaly.evaluation.provenance import honest_state_block  # noqa: E402
from ssl_anomaly.evaluation.research_arms import (  # noqa: E402
    DEFAULT_STATS_ZARR,
    FULL_CANDIDATE_ORDER,
    RESEARCH_ARMS,
    build_cycle_summaries,
    build_research_arm,
    patch_capacity,
)
from ssl_anomaly.evaluation.runner import (  # noqa: E402
    RunPaths,
    SmokeExpectation,
    build_manifest,
    environment_block,
    make_smoke_fixture,
    write_run_outputs,
)
from ssl_anomaly.evaluation.metrics import count_parameters  # noqa: E402

#: The four pre-registered candidates keep their objective declarations.
PREREGISTERED_SPECS = {
    "C-incumbent": CandidateSpec("C-incumbent", "reconstruction", False, True),
    "C-BT": CandidateSpec("C-BT", "barlow-twins", True, False),
    "C-traj": CandidateSpec("C-traj", "trajectory", False, False),
    "C-hybrid": CandidateSpec("C-hybrid", "hybrid", True, True, alpha=1.0),
}

#: C-decomp stays unimplemented with the recorded reason (its producer does not
#: exist; smoke is not empirical evidence).
UNIMPLEMENTED_CANDIDATES = {
    "C-decomp": (
        "decomposition features (trend/seasonal/residual) require the canonical "
        "polygon zonal store, which the decomposition-aware-ssl-study does not yet "
        "have real numbers for; smoke is not empirical evidence, so the candidate is "
        "not implemented rather than approximated"
    )
}


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
    # Sized so each learning candidate's ENCODER lands inside the pre-registered
    # capacity band (100k x [0.8, 1.2]) while every TOTAL stays under the 250k
    # hard cap. Verified by tests/evaluation/test_candidates.py
    # ::test_default_trunk_is_in_band and tests/evaluation/test_research_arms.py.
    p.add_argument("--d-model", type=int, default=160)
    p.add_argument("--d-out", type=int, default=64)
    p.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    p.add_argument("--stats-zarr", type=Path, default=DEFAULT_STATS_ZARR,
                   help="stats store for the C-stat reference arm (Level A)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-training", action="store_true",
                   help="probe randomly-initialised encoders (pipeline exercise only)")
    return p.parse_args(argv)


def _aggregate_seeds(per_seed: dict[int, dict]) -> dict:
    """Mean ± std over seeds for the probe terms; collapse on ANY seed.

    Collapse is a property of the configuration, not of a lucky seed: if any
    seed's representation collapses, the configuration is disqualified — the
    conservative reading of the pre-registered rule.
    """
    keys_top = ("farm_separability_top1", "farm_separability_chance",
                "phase_probe_accuracy", "phase_probe_majority")
    agg: dict = {"n_seeds": len(per_seed), "per_seed": {}}
    for seed, ev in per_seed.items():
        agg["per_seed"][str(seed)] = {k: ev.get(k) for k in keys_top}
        agg["per_seed"][str(seed)]["collapse_disqualified"] = bool(
            ev.get("collapse_disqualified"))
    for k in keys_top:
        vals = [ev[k] for ev in per_seed.values() if ev.get(k) is not None]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        else:
            agg[f"{k}_mean"] = None
            agg[f"{k}_std"] = None
    agg["collapse_disqualified"] = any(
        bool(ev.get("collapse_disqualified")) for ev in per_seed.values())
    return agg


def _aggregated_evidence(aggregate: dict) -> dict:
    """The screening_verdict-shaped evidence for one candidate's aggregate."""
    out = {}
    if aggregate.get("farm_separability_top1_mean") is not None:
        out["farm_separability_top1"] = aggregate["farm_separability_top1_mean"]
        out["farm_separability_chance"] = aggregate["farm_separability_chance_mean"]
    if aggregate.get("phase_probe_accuracy_mean") is not None:
        out["phase_probe_accuracy"] = aggregate["phase_probe_accuracy_mean"]
        out["phase_probe_majority"] = aggregate["phase_probe_majority_mean"]
    out["collapse_disqualified"] = aggregate["collapse_disqualified"]
    return out


def _capacity_for(name: str, model, *, n_channels: int, base_model=None) -> dict:
    """Per-arm capacity, with the band applied to the representation trunk."""
    target, lo, hi = 100_000, 0.8 * 100_000, 1.2 * 100_000
    if name == "C-stat":
        return {"name": name, "params": 0, "encoder_params": 0,
                "band_ok": True, "hard_cap_ok": True,
                "detail": "non-learning reference arm: 0 trainable parameters; "
                          "the band is not applicable (recorded, not assumed)",
                "band": [lo, hi], "target": target}
    if name == "C-patch":
        cap = patch_capacity(n_channels=n_channels)
        return {"name": name, "params": cap["total_params"],
                "encoder_params": cap["encoder_params"],
                "band_ok": bool(cap["band_ok"] and cap["hard_cap_ok"]),
                "hard_cap_ok": cap["hard_cap_ok"],
                "detail": "", "band": cap["band"], "target": target,
                "research_ceiling_params": cap["research_ceiling_params"],
                "note": cap["note"]}
    if name == "C-film":
        # count_parameters reads requires_grad ONLY, and C-film's base is frozen:
        # trainable = adapters (+ film head); the representation trunk the band
        # polices is the base's ENCODER (the base's own recon head is unused here).
        trainable = int(count_parameters(model))
        base_enc = model.base.encoder if hasattr(model.base, "encoder") else model.base
        base_trunk = int(sum(p.numel() for p in base_enc.parameters()))
        base_unused_head = int(sum(p.numel() for p in model.base.parameters())) - base_trunk
        total_with_frozen = trainable + base_trunk + base_unused_head
        band_ok = lo <= base_trunk <= hi
        cap_ok = total_with_frozen <= 250_000
        return {"name": name, "params": trainable, "encoder_params": base_trunk,
                "adapters_params": trainable,
                "total_params_incl_frozen_base": total_with_frozen,
                "band_ok": bool(band_ok and cap_ok), "hard_cap_ok": bool(cap_ok),
                "detail": "" if (band_ok and cap_ok) else
                          (f"base trunk {base_trunk} outside band" if not band_ok
                           else f"total {total_with_frozen} exceeds 250k"),
                "band": [lo, hi], "target": target,
                "note": "band applies to the frozen base ENCODER (the trunk the "
                        "adapters condition); trainable adapter params and the "
                        "frozen totals are recorded separately"}
    return capacity_report(model, name).to_dict()


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke and args.epochs == 40:
        args.epochs = 3          # smoke budget; keeps the smoke under ~90 s
    started = time.time()
    prereg = load_preregistration()
    seeds = list(args.seeds)
    smoke_note = f"pre-registered seeds {seeds}"
    if args.smoke and len(seeds) > 1:
        seeds = seeds[:1]        # recorded: smoke exercises the pipeline with 1 seed
        smoke_note = "smoke budget: 1 seed (real screening aggregates all three)"

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
    fixture_phase_targets = None
    summaries_info: dict
    if args.smoke:
        fixture = make_smoke_fixture(n_fields=8, cycles_per_field=4, seed=13)
        dev = fixture.dev
        fixture_phase_targets = fixture.farm_phase_bins
        summaries_info = {
            "status": "absent",
            "reason": "no stats store behind the synthetic fixture; C-stat is "
                      "recorded absent in smoke (never simulated)",
        }
    else:
        dev = load_cycle_table(args.cycles).for_split("dev")
        # C-stat (Level A): summaries from the stats store, aligned to dev order.
        import pandas as pd

        df = pd.read_parquet(args.cycles)
        keys = [f"{int(f)}:{int(c)}" for f, c in zip(df["field"], df["cycle_id"], strict=True)]
        lut = {k: i for i, k in enumerate(keys)}
        missing = [str(c) for c in dev.cycle_ids if str(c) not in lut]
        if missing:
            summaries_info = {"status": "absent",
                              "reason": f"{len(missing)} dev cycles missing from the parquet"}
        else:
            idx = np.array([lut[str(c)] for c in dev.cycle_ids])
            summaries_info = build_cycle_summaries(
                args.stats_zarr,
                cycles_t_start=df["t_start"].to_numpy()[idx],
                cycles_t_end=df["t_end"].to_numpy()[idx],
                cycles_field=df["field"].to_numpy()[idx],
            )
            summaries_info["note"] = (
                summaries_info.get("note", "")
                + " (research '61-dim' vs measured "
                f"{summaries_info.get('n_features')} recorded as a discrepancy)"
            )

    n_channels = dev.X.shape[2]
    cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size,
                      lr=args.lr, seed=seeds[0])

    def _probe(z: np.ndarray) -> dict:
        res = run_level1(dev=dev, embed_fn=lambda X, M: z,
                         phase_targets=fixture_phase_targets)
        ev = res.gate_evidence()
        ev["probes"] = res.to_dict()
        return ev

    # ---- train + probe each candidate across ALL seeds --------------------
    per_candidate: dict[str, dict] = {}
    capacity: dict[str, dict] = {}
    training: dict[str, dict] = {}
    absent_candidates: dict[str, str] = {}

    # -- C-stat: the non-learning reference (one probe; no seeds to aggregate) --
    if summaries_info.get("status") == "built":
        present = summaries_info["present"]
        if not present.all():
            summaries_info["n_absent_cycles"] = int((~present).sum())
        try:
            stat_model = build_research_arm("C-stat", n_channels,
                                            summaries=summaries_info)
            z_stat = stat_model.embed().cpu().numpy().astype(float)
            ev = _probe(z_stat)
            per_candidate["C-stat"] = {
                "aggregate": _aggregate_seeds({seeds[0]: ev}),
                "capacity_band_ok": True,
                "params": 0,
                "spec": dict(RESEARCH_ARMS["C-stat"].__dict__),
                "selectable": False,
                "note": ("probed as the statistical reference the SSL arms must "
                         "beat; NEVER selectable as region model v0"),
            }
            capacity["C-stat"] = _capacity_for("C-stat", stat_model, n_channels=n_channels)
            training["C-stat"] = {"objective": "statistical", "trained": False,
                                  "params": 0, "config": cfg.to_dict()}
        except ValueError as exc:
            absent_candidates["C-stat"] = str(exc)
    else:
        absent_candidates["C-stat"] = summaries_info.get("reason", "summaries absent")

    # -- learning candidates (pre-registered four + C-patch) ----------------
    for name in ("C-incumbent", "C-BT", "C-traj", "C-hybrid", "C-patch"):
        spec = PREREGISTERED_SPECS.get(name) or CandidateSpec(
            "C-patch", "reconstruction", False, True)
        per_seed: dict[int, dict] = {}
        rand_per_seed: dict[int, dict] = {}
        cap = None
        for seed in seeds:
            if name == "C-patch":
                model = build_research_arm("C-patch", n_channels, d_out=args.d_out)
            else:
                model = build_candidate(name, n_channels=n_channels,
                                        d_model=args.d_model, d_out=args.d_out)
            cap = _capacity_for(name, model, n_channels=n_channels)
            if not args.skip_training:
                # Training runs in smoke too: the point of a smoke is to exercise
                # every code path. Smoke encoders stay near their initialisation
                # and collapse diagnostics will (correctly) disqualify them —
                # that is an honest smoke outcome, not a failure of the smoke.
                train_candidate(model, spec, dev, TrainConfig(
                    epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                    seed=seed), device=args.device)
            z = embed_candidates(model, dev, device=args.device)
            per_seed[seed] = _probe(z)
            # The paired A-rand control per seed: SAME trunk, fresh init, untrained.
            if name == "C-patch":
                rand_model = build_research_arm("C-patch", n_channels, d_out=args.d_out)
            else:
                rand_model = build_candidate(name, n_channels=n_channels,
                                             d_model=args.d_model, d_out=args.d_out)
            z_rand = embed_candidates(rand_model, dev, device=args.device)
            rand_per_seed[seed] = _probe(z_rand)
        capacity[name] = cap
        training[name] = {"objective": spec.objective, "params": cap["params"],
                          "config": cfg.to_dict(), "trained": not args.skip_training}
        per_candidate[name] = {
            "aggregate": _aggregate_seeds(per_seed),
            "random_init": {
                "aggregate": _aggregate_seeds(rand_per_seed),
                "note": ("same architecture and shape, never trained: the paired "
                         "control G-A subtracts"),
            },
            "params": cap["params"],
            "encoder_params": cap["encoder_params"],
            "capacity_band_ok": cap["band_ok"],
            "spec": (dict(RESEARCH_ARMS[name].__dict__) if name in RESEARCH_ARMS
                     else spec.to_dict()),
        }

    # -- C-film: frozen base + adapters; base = C-incumbent trunk -----------
    n_farms = int(dev.field_ids.max()) + 1
    film_cap = None
    film_per_seed: dict[int, dict] = {}
    film_rand_per_seed: dict[int, dict] = {}
    for seed in seeds:
        base = build_candidate("C-incumbent", n_channels=n_channels,
                               d_model=args.d_model, d_out=args.d_out)
        film = build_research_arm("C-film", n_channels, d_model=args.d_model,
                                  n_farms=n_farms, base=base)
        film_cap = _capacity_for("C-film", film, n_channels=n_channels,
                                 base_model=base)
        if not args.skip_training:
            train_candidate(film, CandidateSpec("C-film", "film-recon", False, True),
                            dev, TrainConfig(epochs=args.epochs,
                                             batch_size=args.batch_size,
                                             lr=args.lr, seed=seed),
                            device=args.device)
        z = embed_candidates(film, dev, device=args.device)
        film_per_seed[seed] = _probe(z)
        rand_base = build_candidate("C-incumbent", n_channels=n_channels,
                                    d_model=args.d_model, d_out=args.d_out)
        rand_film = build_research_arm("C-film", n_channels, d_model=args.d_model,
                                       n_farms=n_farms, base=rand_base)
        z_rand = embed_candidates(rand_film, dev, device=args.device)
        film_rand_per_seed[seed] = _probe(z_rand)
    capacity["C-film"] = film_cap
    training["C-film"] = {"objective": "film-recon", "trained": not args.skip_training,
                          "config": cfg.to_dict()}
    per_candidate["C-film"] = {
        "aggregate": _aggregate_seeds(film_per_seed),
        "random_init": {"aggregate": _aggregate_seeds(film_rand_per_seed),
                        "note": "fresh base + fresh adapters, never trained"},
        "params": film_cap["params"],
        "encoder_params": film_cap["encoder_params"],
        "capacity_band_ok": film_cap["band_ok"],
        "spec": dict(RESEARCH_ARMS["C-film"].__dict__),
    }

    # ---- G-A over the aggregates ------------------------------------------
    agg_evidence = {}
    agg_capacity = {}
    for n_, pc in per_candidate.items():
        agg_evidence[n_] = _aggregated_evidence(pc["aggregate"])
        agg_capacity[n_] = {"params": pc["params"],
                            "encoder_params": pc.get("encoder_params", 0),
                            "band_ok": pc["capacity_band_ok"]}
    verdict = screening_verdict(agg_evidence, agg_capacity,
                                absent_candidates=absent_candidates,
                                order=FULL_CANDIDATE_ORDER)

    # ---- G-A verdict for the winner (aggregated, paired) -------------------
    from ssl_anomaly.evaluation.gates import (
        GATE_INPUTS, UNDETERMINED, GateVerdict, gate_a,
    )

    binding = bind_artifacts(prereg=prereg, strict=False, compute_hashes=not args.smoke)
    winner = verdict["winner"]
    if winner is not None:
        pc = per_candidate[winner]
        # Paired SSL-vs-A-rand evidence, aggregated over seeds: the aggregate's
        # means and the random-init control's means, translated to the exact
        # keys gate_a declares (omitting unmeasured terms — never zeroing them).
        rand_agg = pc["random_init"]["aggregate"]
        ssl_ev = {
            "farm_separability_top1": pc["aggregate"].get("farm_separability_top1_mean"),
            "phase_probe_accuracy": pc["aggregate"].get("phase_probe_accuracy_mean"),
            "collapse_disqualified": pc["aggregate"]["collapse_disqualified"],
        }
        rand_ev = {
            "farm_separability_top1": rand_agg.get("farm_separability_top1_mean"),
            "phase_probe_accuracy": rand_agg.get("phase_probe_accuracy_mean"),
            "collapse_disqualified": rand_agg["collapse_disqualified"],
        }
        paired = _paired_g_a_evidence({
            "gate_evidence": ssl_ev, "random_init": {"gate_evidence": rand_ev},
        })
        g_a = gate_a(paired, dict(prereg.gates["G-A"])).to_dict()
    else:
        missing = list(GATE_INPUTS["G-A"])
        g_a = GateVerdict(
            gate="G-A", verdict=UNDETERMINED, criteria=dict(prereg.gates["G-A"]),
            evidence={}, fail_consequence=prereg.gates["G-A"].get("fail_consequence", ""),
            missing_evidence=missing,
            reason=(f"no candidate was screened: {verdict['selection_rule']} — "
                    f"nothing was probed, so no term can be compared to A-rand"),
        ).to_dict()

    screening_blob = {
        "stage": "candidate-screening",
        "smoke": args.smoke,
        "seeds": seeds,
        "seed_note": smoke_note,
        "candidates": {
            n_: {
                "aggregate": pc["aggregate"],
                "random_init_aggregate": pc["random_init"]["aggregate"],
                "capacity": capacity.get(n_),
                "spec": pc.get("spec"),
            } for n_, pc in per_candidate.items()
        },
        "absent_candidates": absent_candidates,
        "unimplemented_candidates": UNIMPLEMENTED_CANDIDATES,
        "verdict": verdict,
        "gate_a": g_a,
        "equal_budget": {"epochs": args.epochs, "batch_size": args.batch_size,
                         "lr": args.lr, "optimizer": "AdamW",
                         "note": "identical optimizer budget for every learning candidate"},
    }

    report = {
        "change": prereg.payload.get("change"),
        "stage": "level1-pretraining-quality",
        "data_source": source,
        "smoke": args.smoke,
        "seeds": seeds,
        "seed_note": smoke_note,
        "n_dev_cycles": int(dev.n_cycles),
        "n_channels": int(n_channels),
        "candidates": per_candidate,
        "capacity": capacity,
        "training": training,
        "summaries_c_stat": summaries_info,
        "unimplemented_candidates": UNIMPLEMENTED_CANDIDATES,
        "screening": verdict,
        "gate_a": g_a,
        "honest_state": honest_state_block(
            measured=[n_ for n_ in per_candidate],
            absent=[f"{n_}: {r}" for n_, r in absent_candidates.items()]
                   + [f"C-decomp: {UNIMPLEMENTED_CANDIDATES['C-decomp']}"],
            deferred=[],
        ),
        "environment": environment_block(),
        "wall_time_s": round(time.time() - started, 2),
    }

    manifest = build_manifest(
        stage="level1-pretraining-quality+screening", smoke=args.smoke,
        config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "d_model": args.d_model, "d_out": args.d_out, "device": args.device,
                "skip_training": args.skip_training},
        seeds=seeds, binding=binding,
    )
    written = write_run_outputs(
        paths=paths, manifest=manifest,
        reports={"pretraining_quality": report, "screening_report": screening_blob},
    )

    # ---- smoke self-check ------------------------------------------------
    if args.smoke:
        exp = SmokeExpectation("level1-pretraining-quality")
        exp.check("every learning candidate produced seed-aggregated probes",
                  all("aggregate" in per_candidate[n_]
                      for n_ in ("C-incumbent", "C-BT", "C-traj", "C-hybrid",
                                 "C-patch", "C-film")),
                  str(sorted(per_candidate)))
        exp.check("C-stat is recorded absent in smoke with a reason",
                  "C-stat" in absent_candidates and bool(absent_candidates["C-stat"]))
        exp.check("collapse diagnostics ran for every learning candidate",
                  all("collapse_disqualified" in per_candidate[n_]["aggregate"]
                      for n_ in per_candidate))
        exp.check("capacity recorded for every candidate",
                  all(n_ in capacity for n_ in per_candidate))
        exp.check("G-A verdict is one of the three real verdicts",
                  g_a["verdict"] in {"passed", "failed", "undetermined"},
                  g_a["verdict"])
        exp.check("an undetermined G-A names the evidence it lacks",
                  g_a["verdict"] != "undetermined" or bool(g_a["missing_evidence"]))
        exp.check("G-A never reports a verdict without a reason",
                  g_a["verdict"] == "passed" or bool(g_a.get("reason")))
        exp.check("every learning candidate carries its paired A-rand control",
                  all("random_init" in per_candidate[n_]
                      for n_ in per_candidate if n_ != "C-stat"))
        exp.check("every learning candidate's trunk sits in the pre-registered band",
                  all(capacity[n_]["band_ok"] for n_ in capacity if n_ != "C-stat"),
                  str({k: (v["encoder_params"], v["band_ok"]) for k, v in capacity.items()}))
        exp.check("at least one candidate measured farm separability",
                  any(pc["aggregate"].get("farm_separability_top1_mean") is not None
                      for pc in per_candidate.values()))
        exp.check("report withheld a winner only for a recorded reason",
                  verdict["winner"] is not None or bool(verdict["selection_rule"]))
        exp.check("screening blob carries verdict and equal budget",
                  screening_blob["verdict"] is not None and "equal_budget" in screening_blob)
        report["smoke_checks"] = exp.to_dict()
        write_run_outputs(paths=paths, manifest=manifest,
                          reports={"pretraining_quality": report,
                                   "screening_report": screening_blob})
        try:
            exp.raise_if_failed()
        except AssertionError as exc:
            print(f"SMOKE FAILED: {exc}", file=sys.stderr)
            return 1
        print(f"smoke ok — {json.dumps(exp.to_dict()['checks'], indent=2)}")

    print(f"wrote {sorted(written.values())}")
    print(f"G-A: {g_a['verdict']}" + (f" (winner {winner})" if winner else ""))
    return 0


def _paired_g_a_evidence(candidate: dict) -> dict:
    """Turn one candidate's report plus its random-init control into the
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

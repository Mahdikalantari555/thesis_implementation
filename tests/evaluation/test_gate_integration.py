"""Integration tests: real stage artifacts -> gate verdicts.

These tests do not mock the extractors. They write artifacts in the SHAPE the
producing scripts emit and drive the aggregator's own functions, so a change to an
artifact's schema breaks here rather than silently reading nothing at runtime.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from ssl_anomaly.evaluation.binding import bind_artifacts, load_preregistration
from ssl_anomaly.evaluation.gates import GATE_INPUTS, evaluate_gates

AGG_PATH = (Path(__file__).resolve().parents[2] / "scripts" / "evaluation"
            / "aggregate_decision_gates.py")


@pytest.fixture(scope="module")
def agg():
    """Import the script as a module without executing its CLI."""
    sys.path.insert(0, str(AGG_PATH.parent))
    spec = importlib.util.spec_from_file_location("agg_gates", AGG_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def prereg():
    return load_preregistration()


# ---------------------------------------------------------------------------
# Artifact fixtures in the producers' real shapes
# ---------------------------------------------------------------------------


def _quality_artifact(*, winner="C-BT", collapse=False):
    """Shaped like ``pretraining_quality.json``, including the A-rand control."""
    candidates = {}
    for name in ("C-incumbent", "C-BT", "C-traj", "C-hybrid"):
        candidates[name] = {
            "gate_evidence": {
                "farm_separability_top1": 0.42 if name == winner else 0.20,
                "farm_separability_chance": 0.125,
                "phase_probe_accuracy": 0.30 if name == winner else 0.10,
                "phase_probe_majority": 0.12,
                "collapse_disqualified": collapse and name == winner,
            }
        }
    return {
        "candidates": candidates,
        "random_init_control": {
            "gate_evidence": {
                "farm_separability_top1": 0.18,
                "phase_probe_accuracy": 0.09,
                "collapse_disqualified": False,
            }
        },
        "screening": {
            "winner": winner,
            "candidates": {n: {"terms_passed": 2 if n == winner else 0,
                               "capacity_ok": True} for n in candidates},
        },
    }


def _transfer_artifact(task="T2", *, lift=0.0):
    """Shaped like ``transfer_curves.json``: six arms x fractions x measured cells.

    One artifact covers ONE task (that is how the producer writes them), so G-B/G-C
    need several files. ``lift`` scales the SSL advantage so a test can push a task
    into or out of the "beats both baselines" band.
    """
    def cell(fraction, ap, arm, status="measured"):
        return {"arm": arm, "fraction": fraction, "repeat": 0, "status": status,
                "ap": ap, "prevalence": 0.25}

    def curve(arm, fracs):
        return {"arm": arm,
                "cells": [cell(f, v, arm) for f, v in fracs.items()]}

    return {
        "task": task,
        "result": {
            "curves": {
                "A-ssl-frozen": curve("A-ssl-frozen",
                                      {0.05: 0.55 + lift, 0.25: 0.60 + lift,
                                       0.50: 0.66 + lift, 1.0: 0.70 + lift}),
                "A-ssl-ft": curve("A-ssl-ft", {0.50: 0.72 + lift, 1.0: 0.75 + lift}),
                "A-scratch": curve("A-scratch", {1.0: 0.60}),
                "A-rule": curve("A-rule", {0.05: 0.40, 0.25: 0.42,
                                           0.50: 0.44, 1.0: 0.45}),
                "A-rand": curve("A-rand", {0.05: 0.30, 0.25: 0.32,
                                           0.50: 0.35, 1.0: 0.38}),
                "A-B1": curve("A-B1", {1.0: 0.52}),
            }
        },
    }


def _merge(*artifacts):
    """Merge several transfer artifacts exactly as the aggregator's gather loop does."""
    label: dict = {}
    grid: dict = {}
    for art in artifacts:
        task = art["task"]
        curves = art["result"]["curves"]
        for arm, curve in curves.items():
            cells = [c for c in curve["cells"] if c["status"] == "measured"]
            if not cells:
                continue
            per_frac: dict = {}
            for c in cells:
                per_frac.setdefault(f"{c['fraction']:.2f}", []).append(c["ap"])
            for frac, vals in per_frac.items():
                val = float(np.mean(vals))
                label.setdefault(task, {}).setdefault(arm, {})[frac] = val
                grid.setdefault(task, {}).setdefault(frac, {})[arm] = val
    return {"label_efficiency": label, "transfer_grid": grid}


def _retrieval_artifact(*, track2=False):
    """Shaped like ``retrieval_benchmark.json`` with a Track-1 block."""
    block = {
        "k_values": [1, 3, 5, 10],
        "arms": {
            "A-ssl-frozen": {"precision_at_k": {"1": 0.80, "3": 0.70,
                                                "5": 0.65, "10": 0.60}},
            "raw-distance": {"precision_at_k": {"1": 0.50, "3": 0.45,
                                                "5": 0.40, "10": 0.35}},
            "A-B1": {"precision_at_k": {"1": 0.55, "3": 0.50,
                                        "5": 0.45, "10": 0.40}},
            "A-rule": {"precision_at_k": {"1": 0.52, "3": 0.48,
                                          "5": 0.44, "10": 0.39}},
        },
    }
    tracks = {"track-1": block}
    if track2:
        tracks["track-2"] = {"k_values": [1], "arms": {}}
    return {"tracks": tracks}


def _claims_artifact(*, rung="L2", provenance=None, exploratory=False):
    return {
        "claims": [
            {
                "claim_id": "capstone-g_a", "status": "measured", "rung": rung,
                "exploratory": exploratory,
                "provenance": provenance if provenance is not None else {
                    "git_sha": "abc123",
                    "preregistration_sha256": "0" * 64,
                },
                "prominence": "primary",
            }
        ]
    }


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def test_g_a_extractor_pairs_ssl_against_rand(agg):
    ev = agg._evidence_g_a(_quality_artifact(winner="C-BT"))
    assert ev["farm_separability_ssl"] == 0.42
    assert ev["farm_separability_rand"] == 0.18
    assert ev["phase_probe_ssl"] == 0.30
    assert ev["phase_probe_rand"] == 0.09
    assert set(ev) <= set(GATE_INPUTS["G-A"]), "extractor leaked non-gate keys"


def test_g_a_extractor_without_a_control_omits_the_comparison(agg):
    art = _quality_artifact()
    del art["random_init_control"]
    ev = agg._evidence_g_a(art)
    assert "farm_separability_ssl" not in ev, (
        "without an A-rand control there is nothing to compare against; the gate "
        "must go undetermined rather than compare against chance"
    )


def test_g_a_extractor_without_a_winner_uses_the_best_scored(agg):
    ev = agg._evidence_g_a(_quality_artifact(winner=None))
    assert ev


def test_g_b_extractor_builds_arm_fraction_table(agg):
    ev = agg._evidence_g_b(_transfer_artifact())
    table = ev["label_efficiency"]
    assert "T2" in table
    assert table["T2"]["A-scratch"]["1.00"] == 0.60
    assert table["T2"]["A-ssl-frozen"]["0.50"] == 0.66


def test_g_c_extractor_builds_fraction_arm_grid(agg):
    ev = agg._evidence_g_c(_transfer_artifact())
    grid = ev["transfer_grid"]
    assert grid["T2"]["0.50"]["A-ssl-frozen"] == 0.66
    assert grid["T2"]["0.50"]["A-rule"] == 0.44


def test_g_d_extractor_reads_track_1_only(agg):
    ev = agg._evidence_g_d(_retrieval_artifact(track2=True))
    block = ev["retrieval_track1"]
    assert block["track"] == "1"
    assert block["k"] == ["1", "3", "5", "10"]
    assert "raw_distance" in block and "b1_score" in block and "rule_score" in block


def test_g_d_extractor_returns_nothing_without_track_1(agg):
    assert agg._evidence_g_d({"tracks": {"track-2": {}}}) == {}


def test_g_e_extractor_passes_claims_through(agg):
    art = _claims_artifact()
    assert agg._evidence_g_e(art)["claims"] == art["claims"]


def test_g_e_extractor_requires_a_list(agg):
    assert agg._evidence_g_e({"claims": "nope"}) == {}


# ---------------------------------------------------------------------------
# End to end through the real gate functions
# ---------------------------------------------------------------------------


def _full_evidence(agg):
    """A complete run: two claim-bearing tasks, so the task-counted gates can pass."""
    evidence = {}
    evidence.update(agg._evidence_g_a(_quality_artifact()))
    evidence.update(_merge(_transfer_artifact("T2"), _transfer_artifact("T3")))
    evidence.update(agg._evidence_g_d(_retrieval_artifact()))
    evidence.update(agg._evidence_g_e(_claims_artifact()))
    return evidence


def test_all_five_gates_decide_from_real_shaped_artifacts(agg, prereg, tmp_path):
    binding = bind_artifacts(base_dir=tmp_path, strict=False)
    report = evaluate_gates(prereg=prereg, evidence=_full_evidence(agg),
                            binding=binding)
    verdicts = {g: v.verdict for g, v in report.verdicts.items()}
    assert verdicts["G-E"] == "passed", report.verdicts["G-E"].reason
    assert verdicts["G-A"] == "passed", report.verdicts["G-A"].reason
    assert verdicts["G-B"] == "passed", report.verdicts["G-B"].reason
    assert verdicts["G-C"] == "passed", report.verdicts["G-C"].reason
    assert verdicts["G-D"] == "passed", report.verdicts["G-D"].reason


def test_collapse_disqualifies_g_a(agg, prereg, tmp_path):
    evidence = _full_evidence(agg)
    evidence.update(agg._evidence_g_a(_quality_artifact(collapse=True)))
    binding = bind_artifacts(base_dir=tmp_path, strict=False)
    report = evaluate_gates(prereg=prereg, evidence=evidence, binding=binding)
    assert report.verdicts["G-A"].verdict == "failed"
    assert "collapse" in report.verdicts["G-A"].reason


def test_g_b_fails_when_ssl_never_matches_scratch_below_full_fraction(agg, prereg, tmp_path):
    art = _transfer_artifact("T2")
    # Push every SSL cell below A-scratch, and give A-ssl-frozen a 100% win only.
    curves = art["result"]["curves"]
    curves["A-ssl-frozen"]["cells"] = [
        {"arm": "A-ssl-frozen", "fraction": 0.5, "repeat": 0, "status": "measured",
         "ap": 0.40, "prevalence": 0.25},
        {"arm": "A-ssl-frozen", "fraction": 1.0, "repeat": 0, "status": "measured",
         "ap": 0.90, "prevalence": 0.25},
    ]
    curves.pop("A-ssl-ft")
    # Both tasks now show SSL losing to A-scratch@100% at <= 50%: the gate must fail
    # rather than scrape a pass from the one 100%-fraction cell that does clear it.
    merged = _merge(art, _transfer_artifact("T3"))
    merged["label_efficiency"]["T3"] = {
        arm: dict(v) for arm, v in merged["label_efficiency"]["T2"].items()
    }
    evidence = _full_evidence(agg)
    evidence["label_efficiency"] = merged["label_efficiency"]
    binding = bind_artifacts(base_dir=tmp_path, strict=False)
    report = evaluate_gates(prereg=prereg, evidence=evidence, binding=binding)
    assert report.verdicts["G-B"].verdict == "failed"


def test_g_d_refuses_a_track_2_payload(agg, prereg, tmp_path):
    """A Track-2 block must not be evaluable even if mislabelled as Track 1."""
    evidence = _full_evidence(agg)
    block = evidence["retrieval_track1"]
    block["track"] = "2"
    binding = bind_artifacts(base_dir=tmp_path, strict=False)
    with pytest.raises(ValueError, match="Track 1 only"):
        evaluate_gates(prereg=prereg, evidence=evidence, binding=binding)


def test_g_e_failure_blocks_every_gate(agg, prereg, tmp_path):
    evidence = _full_evidence(agg)
    evidence.update(agg._evidence_g_e(_claims_artifact(rung=None)))
    binding = bind_artifacts(base_dir=tmp_path, strict=False)
    report = evaluate_gates(prereg=prereg, evidence=evidence, binding=binding)
    assert report.verdicts["G-E"].verdict == "failed"
    for g in ("G-A", "G-B", "G-C", "G-D"):
        assert report.verdicts[g].verdict == "blocked"
    assert report.quota_block()["claims"] == []


def test_binding_mismatch_fails_g_e_and_blocks(agg, prereg, tmp_path):
    bad = tmp_path / "splits_locked_v2.json"
    bad.write_text("{}")
    binding = bind_artifacts(base_dir=tmp_path, strict=False,
                             overrides={"splits_locked_v2": bad})
    assert not binding.ok
    report = evaluate_gates(prereg=prereg, evidence=_full_evidence(agg),
                            binding=binding)
    assert report.verdicts["G-E"].verdict == "failed"
    assert "hash mismatch" in report.verdicts["G-E"].reason
    for g in ("G-A", "G-B", "G-C", "G-D"):
        assert report.verdicts[g].verdict == "blocked"


def test_absent_artifact_leaves_its_gate_blocked_not_passed(agg, prereg, tmp_path):
    evidence = _full_evidence(agg)
    del evidence["retrieval_track1"]        # the retrieval stage never ran
    binding = bind_artifacts(base_dir=tmp_path, strict=False)
    report = evaluate_gates(prereg=prereg, evidence=evidence, binding=binding)
    assert report.verdicts["G-D"].verdict == "undetermined"
    assert "retrieval_track1" in report.verdicts["G-D"].missing_evidence


# ---------------------------------------------------------------------------
# Extractor/gate contract
# ---------------------------------------------------------------------------


def test_every_extractor_output_is_a_subset_of_the_declared_inputs(agg):
    """The extractors must not feed keys a gate cannot read."""
    pairs = {
        "_evidence_g_a": (_quality_artifact(), "G-A"),
        "_evidence_g_b": (_transfer_artifact(), "G-B"),
        "_evidence_g_c": (_transfer_artifact(), "G-C"),
    }
    for fn_name, (artifact, gate) in pairs.items():
        ev = getattr(agg, fn_name)(artifact)
        assert set(ev) <= set(GATE_INPUTS[gate]), f"{fn_name} leaked {set(ev) - set(GATE_INPUTS[gate])}"


def test_gate_sources_cover_all_five_gates(agg):
    assert set(agg.GATE_SOURCES) == {"G-A", "G-B", "G-C", "G-D", "G-E"}


def test_transfer_artifact_feeds_both_g_b_and_g_c(agg):
    """One artifact, two gates: they read the same grid at different resolutions."""
    assert agg.GATE_SOURCES["G-B"][0] == agg.GATE_SOURCES["G-C"][0]

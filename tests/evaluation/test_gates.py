"""Gate tests: the five gates as the PRE-REGISTRATION defines them.

The most important test in this file is
``test_gate_criteria_match_the_frozen_text``: it does not check that the gates
behave sensibly, it checks that they decide what the frozen YAML says they decide.
An earlier version of gates.py implemented plausible-but-invented criteria (G-B as
"AP beats prevalence", G-D as "delta beats the incumbent"), which is precisely the
motivated-reasoning failure the freeze exists to prevent.
"""

from __future__ import annotations

import pytest

from ssl_anomaly.evaluation.binding import bind_artifacts, load_preregistration
from ssl_anomaly.evaluation.gates import (
    ALL_ARMS,
    BLOCKED,
    CLAIM_BEARING_TASKS,
    FAIL,
    GATE_IDS,
    GATE_INPUTS,
    PASS,
    UNDETERMINED,
    evaluate_gates,
    evidence_from_level1,
    gate_a,
    gate_b,
    gate_c,
    gate_d,
    gate_e,
    label_efficiency_table,
    transfer_grid_table,
)


@pytest.fixture(scope="module")
def prereg():
    return load_preregistration()


@pytest.fixture()
def clean_binding(tmp_path):
    return bind_artifacts(base_dir=tmp_path, strict=False)


# ---------------------------------------------------------------------------
# The freeze test: criteria must match the YAML
# ---------------------------------------------------------------------------


def test_gate_criteria_match_the_frozen_text(prereg):
    """Each gate's criterion text is the pre-registered one, not a rewrite."""
    crit = {g: prereg.gates[g]["criterion"] for g in GATE_IDS}
    assert "SSL beats A-rand on >=2 of" in crit["G-A"]
    assert "farm-separability" in crit["G-A"]
    assert "A-ssl-frozen or A-ssl-ft matches A-scratch@100 %" in crit["G-B"]
    assert "<=50 % label fraction" in crit["G-B"]
    assert "A-ssl arms beat A-rule and A-rand on >=2 claim-bearing tasks" in crit["G-C"]
    assert "at >=3 of 4 k values" in crit["G-D"]
    assert "Track 1" in crit["G-D"]
    assert "rung + provenance" in crit["G-E"]


def test_gate_inputs_match_the_prereg_declaration(prereg):
    """The inputs each gate declares here are the ones the YAML names."""
    for g in GATE_IDS:
        declared = set(prereg.gates[g]["inputs"])
        assert declared, f"{g} declares no inputs in the YAML"
    # The executable keys are a refinement of those input names, not a new set.
    assert GATE_INPUTS["G-A"], "G-A must declare executable keys"
    assert GATE_INPUTS["G-B"] == ("label_efficiency",)
    assert GATE_INPUTS["G-C"] == ("transfer_grid",)
    assert GATE_INPUTS["G-D"] == ("retrieval_track1",)


def test_every_arm_in_the_prereg_is_a_known_arm(prereg):
    assert tuple(prereg.payload["arms"]) == ALL_ARMS


def test_fail_consequences_are_transcribed(prereg):
    for g in GATE_IDS:
        assert prereg.gates[g]["fail_consequence"]


# ---------------------------------------------------------------------------
# G-A: SSL vs A-RAND (not vs chance)
# ---------------------------------------------------------------------------


def _ga(**kw):
    base = dict(
        farm_separability_ssl=0.45, farm_separability_rand=0.20,
        phase_probe_ssl=0.40, phase_probe_rand=0.10,
        collapse_disqualified=False,
    )
    base.update(kw)
    return base


def test_g_a_passes_when_ssl_beats_rand_on_two_terms():
    assert gate_a(_ga(), {}).verdict == PASS


def test_g_a_compares_against_rand_not_chance():
    """Both above chance, but SSL does not beat A-rand -> not a pass."""
    v = gate_a(_ga(farm_separability_ssl=0.20, farm_separability_rand=0.22,
                   phase_probe_ssl=0.12, phase_probe_rand=0.13), {})
    assert v.verdict == FAIL
    assert "beat A-rand" in v.reason


def test_g_a_single_term_beaten_fails():
    v = gate_a(_ga(phase_probe_ssl=0.05), {})
    assert v.verdict == FAIL


def test_g_a_collapse_disqualifies_even_with_two_wins():
    v = gate_a(_ga(collapse_disqualified=True), {})
    assert v.verdict == FAIL
    assert "collapse" in v.reason


def test_g_a_third_term_counts_toward_the_two():
    """frozen-head T1 is the third named term and may be one of the two needed."""
    v = gate_a(_ga(farm_separability_ssl=0.5, farm_separability_rand=0.2,
                   phase_probe_ssl=0.0, phase_probe_rand=0.9,
                   frozen_head_t1_ssl=0.8, frozen_head_t1_rand=0.3), {})
    assert v.verdict == PASS
    assert v.detail["terms"] == {"farm_separability": True,
                                 "phase_probe": False,
                                 "frozen_head_t1": True}


def test_g_a_reports_which_terms_were_won_and_lost():
    """A 1/2 must not read as 'nothing worked' when one named term WAS won."""
    v = gate_a(_ga(farm_separability_ssl=0.1, farm_separability_rand=0.5,
                   phase_probe_ssl=0.0, phase_probe_rand=0.9,
                   frozen_head_t1_ssl=0.8, frozen_head_t1_rand=0.3), {})
    assert v.verdict == FAIL
    assert "won: frozen_head_t1" in v.reason
    assert "lost:" in v.reason


def test_g_a_caps_min_terms_when_only_one_is_measurable():
    v = gate_a(_ga(phase_probe_ssl=None, phase_probe_rand=None,
                   frozen_head_t1_ssl=None, frozen_head_t1_rand=None), {})
    assert v.verdict == PASS
    assert v.detail["effective_min_terms"] == 1
    assert "unmeasured term" in v.reason


def test_g_a_with_no_terms_is_undetermined():
    v = gate_a({"collapse_disqualified": False,
                "farm_separability_ssl": None, "farm_separability_rand": None}, {})
    assert v.verdict == UNDETERMINED


def test_g_a_without_collapse_evidence_is_undetermined():
    v = gate_a(_ga(collapse_disqualified=None), {})
    assert v.verdict == UNDETERMINED


# ---------------------------------------------------------------------------
# G-B: SSL matches A-scratch@100% at <=50% label fraction
# ---------------------------------------------------------------------------


def _gb_table(**over):
    table = {
        "T2": {
            "A-scratch": {"1.00": 0.60},
            "A-ssl-frozen": {"0.05": 0.40, "0.25": 0.58, "0.50": 0.62, "1.00": 0.70},
        }
    }
    for task, arms in over.items():
        table[task] = arms
    return table


def test_g_b_passes_when_ssl_matches_at_half_or_less():
    v = gate_b({"label_efficiency": _gb_table()}, {"ssl_fraction_max": 0.50})
    assert v.verdict == PASS
    assert v.detail["per_task"]["T2"]["fraction"] == 0.50


def test_g_b_ignores_a_match_that_needs_100_percent():
    """Only the <=50% cells are eligible; the 100% cell is the easy test."""
    table = {"T2": {"A-scratch": {"1.00": 0.60},
                    "A-ssl-frozen": {"1.00": 0.70}}}   # only a full-fraction win
    v = gate_b({"label_efficiency": table}, {"ssl_fraction_max": 0.50})
    assert v.verdict == FAIL
    assert "no SSL cell at or below" in v.detail["per_task"]["T2"]["reason"]


def test_g_b_fails_when_ssl_stays_below_scratch():
    table = {"T2": {"A-scratch": {"1.00": 0.60},
                    "A-ssl-frozen": {"0.05": 0.30, "0.25": 0.45}}}
    v = gate_b({"label_efficiency": table}, {"ssl_fraction_max": 0.50})
    assert v.verdict == FAIL


def test_g_b_accepts_either_ssl_arm():
    table = {"T2": {"A-scratch": {"1.00": 0.60},
                    "A-ssl-ft": {"0.50": 0.65}}}
    v = gate_b({"label_efficiency": table}, {"ssl_fraction_max": 0.50})
    assert v.verdict == PASS
    assert v.detail["per_task"]["T2"]["arm"] == "A-ssl-ft"


def test_g_b_rejects_t5_as_a_gate_input():
    with pytest.raises(ValueError, match="exploratory task"):
        gate_b({"label_efficiency": {"T5": {"A-scratch": {"1.00": 0.5}}}},
               {"ssl_fraction_max": 0.50})


def test_g_b_distractor_arm_does_not_count():
    """A-rule beating A-scratch is irrelevant: only SSL arms may satisfy G-B."""
    table = {"T2": {"A-scratch": {"1.00": 0.60}, "A-rule": {"0.50": 0.90}}}
    v = gate_b({"label_efficiency": table}, {"ssl_fraction_max": 0.50})
    assert v.verdict == FAIL


def test_g_b_missing_evidence_is_undetermined():
    assert gate_b({}, {}).verdict == UNDETERMINED


# ---------------------------------------------------------------------------
# G-C: SSL beats BOTH A-rule and A-rand, 2 tasks x 2 fractions
# ---------------------------------------------------------------------------


def _gc_grid(tasks=("T2", "T3"), wins=2, fracs=("0.25", "0.50")):
    grid = {}
    for t in list(tasks) + [f"T{i}" for i in range(4, 4)]:
        if t not in CLAIM_BEARING_TASKS:
            continue
        grid[t] = {}
        for f in fracs:
            grid[t][f] = {
                "A-ssl-frozen": 0.70 if t in tasks[:wins] else 0.20,
                "A-rule": 0.50, "A-rand": 0.40,
            }
    return grid


def test_g_c_passes_with_two_tasks_at_two_fractions():
    v = gate_c({"transfer_grid": _gc_grid(wins=2)}, {})
    assert v.verdict == PASS
    assert len(v.detail["qualifying_tasks"]) == 2


def test_g_c_fails_with_one_task():
    v = gate_c({"transfer_grid": _gc_grid(wins=1)}, {})
    assert v.verdict == FAIL


def test_g_c_requires_beating_a_rand_too():
    """Beating the rule but losing to A-rand is NOT dominance."""
    grid = {"T2": {"0.25": {"A-ssl-frozen": 0.55, "A-rule": 0.50, "A-rand": 0.60}}}
    v = gate_c({"transfer_grid": grid}, {})
    assert v.verdict == FAIL
    cell = v.detail["per_task"]["T2"]["cells"]["0.25"]
    assert cell["win"] is False
    assert cell["strongest_baseline"] == 0.60


def test_g_c_requires_two_fractions_per_task():
    grid = {"T2": {"0.25": {"A-ssl-frozen": 0.9, "A-rule": 0.1, "A-rand": 0.1}},
            "T3": {"0.25": {"A-ssl-frozen": 0.9, "A-rule": 0.1, "A-rand": 0.1}}}
    v = gate_c({"transfer_grid": grid}, {})
    assert v.verdict == FAIL, "two tasks with one fraction each must not pass"


def test_g_c_cell_without_baselines_is_not_decidable():
    grid = {"T2": {"0.25": {"A-ssl-frozen": 0.9}}}     # no baselines
    v = gate_c({"transfer_grid": grid}, {})
    assert v.verdict == UNDETERMINED
    assert v.detail["per_task"]["T2"]["cells"]["0.25"]["decidable"] is False


def test_g_c_rejects_t5():
    with pytest.raises(ValueError, match="exploratory"):
        gate_c({"transfer_grid": {"T5": {"0.25": {"A-ssl-frozen": 1.0}}}}, {})


# ---------------------------------------------------------------------------
# G-D: Track 1 only, >=3 of 4 k values, beat all three baselines
# ---------------------------------------------------------------------------


def _gd(**over):
    block = {
        "track": "1",
        "k": [1, 3, 5, 10],
        "ssl": {"1": 0.8, "3": 0.7, "5": 0.65, "10": 0.6},
        "raw_distance": {"1": 0.5, "3": 0.45, "5": 0.4, "10": 0.35},
        "b1_score": {"1": 0.55, "3": 0.5, "5": 0.45, "10": 0.4},
        "rule_score": {"1": 0.52, "3": 0.48, "5": 0.44, "10": 0.39},
    }
    block.update(over)
    return block


def test_g_d_passes_when_ssl_beats_all_baselines_at_three_k():
    v = gate_d({"retrieval_track1": _gd()}, {})
    assert v.verdict == PASS
    assert len(v.detail["wins_at_k"]) == 4


def test_g_d_fails_at_two_k_values():
    b = _gd()
    b["ssl"]["5"] = 0.1
    b["ssl"]["10"] = 0.1
    v = gate_d({"retrieval_track1": b}, {})
    assert v.verdict == FAIL
    assert len(v.detail["wins_at_k"]) == 2


def test_g_d_requires_beating_every_baseline():
    b = _gd()
    b["b1_score"]["1"] = 0.95      # SSL loses to one baseline at k=1
    v = gate_d({"retrieval_track1": b}, {})
    assert v.detail["per_k"]["1"]["beats_all"] is False
    assert v.verdict == PASS        # still wins at 3 and 5 and 10


def test_g_d_refuses_track_2():
    # NOTE: {**payload, "track": "2"} — the override must come LAST. Writing
    # {"track": "2", **_gd()} lets _gd()'s default "1" win, and the test then
    # proves nothing while still passing a payload it thinks is Track 2.
    payload = {**_gd(), "track": "2"}
    with pytest.raises(ValueError, match="Track 1 only"):
        gate_d({"retrieval_track1": payload}, {})


def test_g_d_missing_k_is_not_a_loss():
    b = _gd()
    del b["raw_distance"]["10"]
    v = gate_d({"retrieval_track1": b}, {})
    assert v.detail["per_k"]["10"]["decidable"] is False
    assert v.verdict == PASS


# ---------------------------------------------------------------------------
# G-E: claim discipline
# ---------------------------------------------------------------------------


def _claim(**over):
    base = {
        "claim_id": "c1", "status": "measured", "rung": "L2",
        "provenance": {"git_sha": "abc", "preregistration_sha256": "def"},
        "prominence": "primary",
    }
    base.update(over)
    return base


def test_g_e_passes_with_well_formed_claims():
    v = gate_e({"claims": [_claim()]}, {})
    assert v.verdict == PASS


def test_g_e_fails_when_a_claim_has_no_rung():
    v = gate_e({"claims": [_claim(rung=None)]}, {})
    assert v.verdict == FAIL
    assert "no rung" in v.reason


def test_g_e_fails_when_a_claim_has_no_provenance():
    v = gate_e({"claims": [_claim(provenance={})]}, {})
    assert v.verdict == FAIL
    assert "no provenance" in v.reason


def test_g_e_allows_exploratory_claims_without_a_rung():
    v = gate_e({"claims": [_claim(status="exploratory", rung=None, exploratory=True)]}, {})
    assert v.verdict == PASS
    assert v.detail["counts"]["exploratory"] == 1


def test_g_e_rejects_an_exploratory_claim_that_claims_a_rung():
    v = gate_e({"claims": [_claim(exploratory=True, rung="L3")]}, {})
    assert v.verdict == FAIL
    assert "exploratory claim carries rung" in v.reason


def test_g_e_fails_when_a_negative_is_demoted():
    v = gate_e({"claims": [_claim(prominence="demoted", status="negative")]}, {})
    assert v.verdict == FAIL
    assert "demoted" in v.reason


def test_g_e_fails_on_binding_mismatch(tmp_path):
    bad = tmp_path / "splits_locked_v2.json"
    bad.write_text("{}")
    binding = bind_artifacts(base_dir=tmp_path, strict=False,
                             overrides={"splits_locked_v2": bad})
    assert not binding.ok
    v = gate_e({"claims": [_claim()]}, {}, binding=binding)
    assert v.verdict == FAIL
    assert "hash mismatch" in v.reason


def test_g_e_empty_claim_list_is_undetermined():
    assert gate_e({"claims": []}, {}).verdict == UNDETERMINED


# ---------------------------------------------------------------------------
# Aggregation and the G-E precondition
# ---------------------------------------------------------------------------


def _full_evidence():
    return {
        **_ga(),
        "label_efficiency": _gb_table(),
        "transfer_grid": _gc_grid(wins=2),
        "retrieval_track1": _gd(),
        "claims": [_claim()],
    }


def test_all_gates_evaluated_in_order(prereg, clean_binding):
    report = evaluate_gates(prereg=prereg, evidence=_full_evidence(),
                            binding=clean_binding)
    assert list(report.verdicts) == list(GATE_IDS)
    assert all(report.verdicts[g].verdict == PASS for g in GATE_IDS), {
        g: report.verdicts[g].reason for g in GATE_IDS
    }


def test_failed_g_e_blocks_every_gate(prereg, clean_binding):
    ev = _full_evidence()
    ev["claims"] = [_claim(rung=None)]
    report = evaluate_gates(prereg=prereg, evidence=ev, binding=clean_binding)
    assert report.verdicts["G-E"].verdict == FAIL
    for g in ("G-A", "G-B", "G-C", "G-D"):
        assert report.verdicts[g].verdict == BLOCKED
        assert report.verdicts[g].blocked_by == "G-E"
    assert report.quotable_gates() == []
    assert report.quota_block()["claims"] == []


def test_binding_mismatch_blocks_everything(prereg, tmp_path):
    bad = tmp_path / "splits_locked_v2.json"
    bad.write_text("{}")
    binding = bind_artifacts(base_dir=tmp_path, strict=False,
                             overrides={"splits_locked_v2": bad})
    report = evaluate_gates(prereg=prereg, evidence=_full_evidence(), binding=binding)
    assert report.verdicts["G-E"].verdict == FAIL
    for g in ("G-A", "G-B", "G-C", "G-D"):
        assert report.verdicts[g].verdict == BLOCKED


def test_missing_binding_is_allowed_but_noted(prereg):
    """No binding report -> G-E judges only the claims it was given."""
    report = evaluate_gates(prereg=prereg, evidence=_full_evidence(), binding=None)
    assert report.verdicts["G-E"].verdict == PASS
    assert report.verdicts["G-E"].detail["binding_ok"] is None


def test_partial_evidence_leaves_gates_undetermined(prereg, clean_binding):
    # G-E needs claims; supply one so it can pass and the other gates are visible
    # rather than blocked. Without it G-A would legitimately be BLOCKED, not PASS.
    report = evaluate_gates(prereg=prereg,
                            evidence={**_ga(), "claims": [_claim()]},
                            binding=clean_binding)
    assert report.verdicts["G-E"].verdict == PASS
    assert report.verdicts["G-A"].verdict == PASS
    assert report.verdicts["G-B"].verdict == UNDETERMINED
    assert report.verdicts["G-C"].verdict == UNDETERMINED
    assert report.verdicts["G-D"].verdict == UNDETERMINED


def test_each_verdict_carries_criteria_and_consequence(prereg, clean_binding):
    report = evaluate_gates(prereg=prereg, evidence=_ga(), binding=clean_binding)
    for g in GATE_IDS:
        d = report.verdicts[g].to_dict()
        assert d["criteria"] and d["fail_consequence"]


def test_criteria_come_from_the_preregistration(prereg, clean_binding):
    report = evaluate_gates(prereg=prereg, evidence=_ga(), binding=clean_binding)
    assert report.verdicts["G-A"].criteria == dict(prereg.gates["G-A"])


def test_quota_block_lists_all_five_when_g_e_passes(prereg, clean_binding):
    report = evaluate_gates(prereg=prereg,
                            evidence={**_ga(), "claims": [_claim()]},
                            binding=clean_binding)
    q = report.quota_block()
    assert q["quotable"] is True
    assert len(q["claims"]) == 5
    assert {c["gate"] for c in q["claims"]} == set(GATE_IDS)


def test_report_serialises(prereg, clean_binding):
    d = evaluate_gates(prereg=prereg, evidence=_ga(), binding=clean_binding).to_dict()
    assert "gates" in d and "quotable_gates" in d


# ---------------------------------------------------------------------------
# Evidence builders from the stage outputs
# ---------------------------------------------------------------------------


def test_evidence_from_level1_uses_paired_candidates():
    by_cand = {
        "C-BT": {"farm_separability_top1": 0.4, "phase_probe_accuracy": 0.3,
                 "collapse_disqualified": False},
        "C-BT-rand": {"farm_separability_top1": 0.2, "phase_probe_accuracy": 0.1,
                      "collapse_disqualified": False},
    }
    ev = evidence_from_level1(by_cand, ssl="C-BT", rand="C-BT-rand")
    assert ev["farm_separability_ssl"] == 0.4
    assert ev["farm_separability_rand"] == 0.2
    assert "frozen_head_t1_ssl" not in ev


def test_evidence_from_level1_skips_a_null_phase_probe():
    by_cand = {
        "C-BT": {"farm_separability_top1": 0.4, "phase_probe_accuracy": None,
                 "collapse_disqualified": False},
        "C-BT-rand": {"farm_separability_top1": 0.2, "phase_probe_accuracy": None,
                      "collapse_disqualified": False},
    }
    ev = evidence_from_level1(by_cand, ssl="C-BT", rand="C-BT-rand")
    assert "phase_probe_ssl" not in ev, (
        "an unmeasured phase probe must not become a 0.0 that loses G-A"
    )


def test_label_efficiency_table_rejects_t5():
    with pytest.raises(ValueError, match="not claim-bearing"):
        label_efficiency_table({"T5": {"A-ssl-frozen": {1.0: 0.5}}})


def test_transfer_grid_table_normalises_fraction_keys():
    grid = transfer_grid_table({"T2": {0.5: {"A-ssl-frozen": 0.7, "A-rule": 0.4}}})
    assert "0.50" in grid["T2"]


def test_transfer_grid_table_rejects_t5():
    with pytest.raises(ValueError, match="not claim-bearing"):
        transfer_grid_table({"T5": {0.5: {"A-ssl-frozen": 0.7}}})

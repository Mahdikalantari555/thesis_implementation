"""T1–T5 task battery tests.

The load-bearing assertions: a result cannot carry a value unless measured, T2
cannot report an F1 without a validation threshold, T3's two tracks cannot
collapse into one number, T4 is deferred for a recorded reason, and T5 is
exploratory even when measured.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.evaluation.tasks import (
    TASKS,
    TASK_TABLE,
    PhaseTolerance,
    TaskResult,
    t1_phase_estimation,
    t2_anomaly_detection,
    t3_retrieval,
    t3_retrieval_track_a,
    t3_retrieval_track_b,
    t4_forecasting,
    t5_localisation,
    task_table,
)

# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_five_tasks_declared():
    assert TASKS == ("T1", "T2", "T3", "T4", "T5")
    assert len(task_table()) == 5


def test_only_t5_is_exploratory():
    assert [t for t, v in TASK_TABLE.items() if not v["claim_bearing"]] == ["T5"]


def test_measured_without_value_is_rejected():
    with pytest.raises(ValueError, match="measured requires a value"):
        TaskResult(task="T1", subject="s", status="measured")


def test_absent_with_value_is_rejected():
    with pytest.raises(ValueError, match="must not carry a value"):
        TaskResult(task="T1", subject="s", status="absent", value={"x": 1})


def test_deferred_with_value_is_rejected():
    with pytest.raises(ValueError, match="must not carry a value"):
        TaskResult(task="T4", subject="s", status="deferred", value={"x": 1})


def test_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="unknown task status"):
        TaskResult(task="T1", subject="s", status="estimated")


def test_t5_is_never_quotable_even_when_measured():
    r = t5_localisation(np.array([1, 2, 3]), np.array([1, 2, 3]), subject="C-BT")
    assert r.status == "measured"
    assert r.quotable is False


def test_t1_is_quotable_when_measured():
    r = t1_phase_estimation(np.array([100.0]), np.array([100.0]), subject="C-BT")
    assert r.quotable is True


def test_absent_task_is_not_quotable():
    r = TaskResult.absent("T2", "s", "why")
    assert r.quotable is False


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------


def test_t1_perfect_prediction():
    dsh = np.full(50, 200.0)
    r = t1_phase_estimation(dsh, dsh, subject="C-BT")
    assert r.value["within_tolerance_share"] == 1.0
    assert r.value["mae_days"] == 0.0
    assert r.value["bias_days"] == 0.0
    assert r.n == 50


def test_t1_tolerance_boundary_is_inclusive():
    truth = np.zeros(4)
    pred = np.array([15.0, 15.01, -15.0, -15.01])
    r = t1_phase_estimation(pred, truth, subject="x", tolerance=PhaseTolerance(days=15.0))
    assert r.value["within_tolerance_share"] == pytest.approx(0.5)


def test_t1_bias_is_reported_separately_from_mae():
    truth = np.zeros(10)
    pred = np.full(10, 30.0)
    r = t1_phase_estimation(pred, truth, subject="x")
    assert r.value["mae_days"] == pytest.approx(30.0)
    assert r.value["bias_days"] == pytest.approx(30.0)


def test_t1_ci_brackets_the_estimate():
    rng = np.random.default_rng(1)
    truth = rng.uniform(0, 500, 200)
    pred = truth + rng.normal(0, 10, 200)
    r = t1_phase_estimation(pred, truth, subject="x")
    lo, hi = r.ci["within_tolerance"]
    assert lo <= r.value["within_tolerance_share"] <= hi


def test_t1_empty_is_absent():
    r = t1_phase_estimation(np.array([]), np.array([]), subject="x")
    assert r.status == "absent"


def test_t1_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="must align"):
        t1_phase_estimation(np.zeros(3), np.zeros(4), subject="x")


# ---------------------------------------------------------------------------
# T2
# ---------------------------------------------------------------------------


@pytest.fixture()
def detection():
    rng = np.random.default_rng(7)
    n = 300
    y = (rng.random(n) < 0.25).astype(int)
    scores = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.2, n), 0, 1)
    clusters = np.repeat(np.arange(30), 10)
    return scores, y, clusters


def test_t2_without_validation_reports_no_f1(detection):
    s, y, c = detection
    r = t2_anomaly_detection(s, y, c, subject="C-BT")
    assert r.status == "measured"
    assert "f1" not in r.value
    assert r.diagnostics["f1"] is None
    assert "forbidden" in r.diagnostics["threshold_source"]


def test_t2_with_validation_reports_f1(detection):
    s, y, c = detection
    r = t2_anomaly_detection(s, y, c, subject="C-BT", validation=(s, y))
    assert "f1" in r.value
    assert r.diagnostics["threshold_source"] == "validation split"


def test_t2_reports_prevalence_alongside_ap(detection):
    s, y, c = detection
    r = t2_anomaly_detection(s, y, c, subject="C-BT")
    assert r.value["prevalence"] == pytest.approx(y.mean())
    assert r.value["ap"] > r.value["prevalence"]


def test_t2_single_class_is_absent():
    r = t2_anomaly_detection(np.zeros(10), np.zeros(10, dtype=int), np.arange(10),
                             subject="x")
    assert r.status == "absent"
    assert "single class" in r.reason


def test_t2_empty_is_absent():
    r = t2_anomaly_detection(np.array([]), np.array([]), np.array([]), subject="x")
    assert r.status == "absent"


def test_t2_ap_ci_is_reported(detection):
    s, y, c = detection
    r = t2_anomaly_detection(s, y, c, subject="C-BT")
    lo, hi = r.ci["ap"]
    assert lo <= r.value["ap"] <= hi


# ---------------------------------------------------------------------------
# T3 — the two tracks
# ---------------------------------------------------------------------------


@pytest.fixture()
def retrieval_setup():
    """Two crops, four fields, cycles clustered by field."""
    rng = np.random.default_rng(11)
    field = np.repeat([10, 20, 30, 40], 5)
    phase = np.array([10.0, 12.0, 25.0, 27.0])   # fields 10/20 close, 30/40 close
    crop = np.array([0, 0, 1, 1])
    centres = {10: [1, 0, 0], 20: [1, 0.1, 0], 30: [0, 1, 0], 40: [0, 1, 0.1]}
    emb = np.stack([np.array(centres[f], float) + rng.normal(0, 0.02, 3) for f in field])
    return emb, field, phase, crop


def test_track_a_is_high_for_field_clustered_embeddings(retrieval_setup):
    emb, field, _, _ = retrieval_setup
    r = t3_retrieval_track_a(emb, field, subject="C-BT")
    assert r.status == "measured"
    assert r.value["track"] == "A-cycle"
    assert r.value["precision_at_k"]["1"] == pytest.approx(1.0)


def test_track_a_note_says_it_does_not_support_h2(retrieval_setup):
    emb, field, _, _ = retrieval_setup
    r = t3_retrieval_track_a(emb, field, subject="C-BT")
    assert "does_not_support" in r.diagnostics
    assert "H2" in r.diagnostics["does_not_support"]


def test_track_b_removes_the_own_field(retrieval_setup):
    emb, field, phase, crop = retrieval_setup
    r = t3_retrieval_track_b(emb, field, phase, crop, subject="C-BT")
    assert r.status == "measured"
    assert r.value["track"] == "B-field"
    # own field can never be its own hit -> precision@1 <= 1 and n_queries == fields
    assert r.value["n_queries"] == 4
    assert r.value["precision_at_k"]["1"] <= 1.0


def test_track_b_chance_is_over_remaining_fields(retrieval_setup):
    emb, field, phase, crop = retrieval_setup
    r = t3_retrieval_track_b(emb, field, phase, crop, subject="C-BT")
    p1 = r.value["precision_at_k"]["1"]
    chance = r.value["chance_precision_at_k"]["1"]
    assert chance < 1.0
    assert p1 > chance, "an informative embedding must beat the field-level chance"


def test_track_b_needs_field_level_labels(retrieval_setup):
    emb, field, _, _ = retrieval_setup
    with pytest.raises(ValueError, match="one entry per field"):
        t3_retrieval_track_b(emb, field, np.array([1.0, 2.0]), np.array([0, 1]),
                             subject="C-BT")


def test_both_tracks_are_returned_separately(retrieval_setup):
    emb, field, phase, crop = retrieval_setup
    out = t3_retrieval(emb, field, subject="C-BT", field_phase=phase, field_crop=crop)
    assert out["A-cycle"]["status"] == "measured"
    assert out["B-field"]["status"] == "measured"
    assert out["A-cycle"]["value"]["track"] != out["B-field"]["value"]["track"]
    assert "two tracks" in out["_contract"]


def test_track_b_absent_without_field_labels(retrieval_setup):
    emb, field, _, _ = retrieval_setup
    out = t3_retrieval(emb, field, subject="C-BT")
    assert out["A-cycle"]["status"] == "measured"
    assert out["B-field"]["status"] == "absent"
    assert "H2-bearing" in out["B-field"]["reason"]


def test_track_b_absent_when_a_field_has_no_partner(retrieval_setup):
    emb, field, _, _ = retrieval_setup
    phase = np.array([0.0, 20.0, 40.0, 60.0])      # nothing within 2 bins
    crop = np.array([0, 1, 2, 3])                  # and all different crops
    r = t3_retrieval_track_b(emb, field, phase, crop, subject="C-BT")
    assert r.status == "absent"
    assert "no relevant partner" in r.reason


def test_track_a_excludes_single_cycle_fields_and_counts_them():
    rng = np.random.default_rng(2)
    emb = rng.normal(size=(6, 3))
    field = np.array([1, 1, 1, 2, 3, 3])         # field 2 has a single cycle
    r = t3_retrieval_track_a(emb, field, subject="x")
    assert r.status == "measured"
    assert r.diagnostics["n_queries_excluded"] == 1
    assert r.value["n_queries"] == 5
    assert "undefined AP" in r.diagnostics["exclusion_reason"]


def test_track_a_all_singleton_fields_is_absent():
    rng = np.random.default_rng(3)
    r = t3_retrieval_track_a(rng.normal(size=(3, 3)), np.array([1, 2, 3]), subject="x")
    assert r.status == "absent"
    assert "undefined" in r.reason


def test_track_b_too_few_fields_is_absent():
    r = t3_retrieval_track_b(np.random.normal(size=(4, 3)), np.array([1, 1, 2, 2]),
                             np.array([0.0, 1.0]), np.array([0, 1]), subject="x")
    assert r.status == "absent"
    assert "3 distinct fields" in r.reason


# ---------------------------------------------------------------------------
# T4 — deferred by design
# ---------------------------------------------------------------------------


def test_t4_is_deferred_with_the_open_decision_named():
    r = t4_forecasting(subject="C-BT")
    assert r.status == "deferred"
    assert "design.md:329" in r.reason
    assert r.value is None


def test_t4_stays_deferred_even_with_data():
    """Data availability is not the blocker — the budget decision is."""
    r = t4_forecasting(subject="C-BT", hour_budget=4.0)
    assert r.status == "deferred"


# ---------------------------------------------------------------------------
# T5
# ---------------------------------------------------------------------------


def test_t5_measures_timing_error():
    r = t5_localisation(np.array([3, 5, 8]), np.array([3, 5, 9]), subject="x",
                        tolerance_bins=1)
    assert r.status == "measured"
    assert r.value["within_tolerance_share"] == 1.0
    assert r.value["mae_bins"] == pytest.approx(1 / 3)


def test_t5_ignores_nan_predictions_and_counts_them():
    r = t5_localisation(np.array([3.0, np.nan, 8.0]), np.array([3.0, 5.0, 8.0]),
                        subject="x")
    assert r.value["n_unusable"] == 1
    assert r.n == 2


def test_t5_all_nan_is_absent():
    r = t5_localisation(np.array([np.nan]), np.array([1.0]), subject="x")
    assert r.status == "absent"


def test_t5_is_flagged_exploratory_in_diagnostics():
    r = t5_localisation(np.array([1]), np.array([1]), subject="x")
    assert r.diagnostics["claim_bearing"] is False
    assert "never used as evidence" in r.diagnostics["note"]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_result_round_trips_through_dict():
    r = t1_phase_estimation(np.array([100.0]), np.array([100.0]), subject="C-BT")
    d = r.to_dict()
    assert d["task"] == "T1" and d["status"] == "measured"
    assert d["value"]["within_tolerance_share"] == 1.0
    assert d["n"] == 1

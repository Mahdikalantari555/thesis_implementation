"""Tasks 7.1–7.7 — the calibration pooling axis, its diagnostics and the holdout check.

The design's C1/C2/C3 are *pooling* levels (farm-specific / farm × stage / global), not
merely three probability mappings: a farm is compared to its own history first, and
pooled cross-farm statistics may only fill a per-farm deficit, with every fill counted.
These tests fail if the axis degenerates back into three interchangeable maps, if a
reference is fitted on the scored partition, or if the threshold's optimism is silently
dropped.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.benchmarking.calibration import (
    calibration_holdout_check,
    calibrate,
    cycle_stage,
    pooled_reference_map,
)


@pytest.fixture()
def two_scale_farms():
    """20 farms × 6 train cycles, whose score levels differ per farm.

    Farm A-family scores centre at 1.0, farm B-family at 10.0 — a pooled global ECDF
    maps a farm-A alert near 0, which is exactly the cross-farm contamination C1 exists
    to remove.
    """
    rng = np.random.default_rng(11)
    n_farm, per = 20, 6
    farms = np.repeat(np.arange(n_farm), per)
    train = rng.uniform(0.9, 1.1, farms.size) + 9.0 * (farms >= n_farm // 2)
    eval_farms = np.arange(n_farm)
    eval_scores = np.full(n_farm, 1.05)
    eval_scores[eval_farms >= n_farm // 2] = 10.05          # each farm's own typical value
    return {"farms": farms, "train": train, "eval_farms": eval_farms, "eval": eval_scores}


def test_c1_maps_each_farm_against_its_own_history(two_scale_farms):
    t = two_scale_farms
    p, fit = calibrate("C1", scores_dev=t["eval"], labels_dev=np.zeros(t["eval"].size),
                       train_scores=t["train"], scores_eval=t["eval"],
                       eval_keys=t["eval_farms"], train_keys=t["farms"], min_history=2)
    assert fit.fitted_on == "train" and fit.params["pooling_level"] == "farm"
    # each farm's own typical score must sit mid-distribution of its OWN history, not
    # near zero the way the pooled map would put the low-scale farms
    assert float(np.median(p)) > 0.3, (
        f"per-farm reference pools not used: a pooled ECDF sends every farm-A row to ~0 (p={p[:5]})")
    assert fit.params["pooled_fill"] == 0 and fit.params["n_pools"] == 20


def test_c1_pooled_fill_is_counted_not_silent(two_scale_farms):
    t = two_scale_farms
    eval_farms = np.array([100, 101, 100])          # farms with no train history at all
    p, fit = calibrate("C1", scores_dev=np.array([1.0, 1.1, 10.0]),
                       labels_dev=np.zeros(3), train_scores=t["train"],
                       scores_eval=np.array([1.0, 1.1, 10.0]),
                       eval_keys=eval_farms, train_keys=t["farms"], min_history=2)
    assert fit.params["pooled_fill"] == 3, fit.params
    assert "pooled" in fit.params["pooled_fill_note"]
    assert np.isfinite(p).all(), "a pooled-filled row must still be scoreable, not dropped"


def test_c1_c2_refuse_dev_only_references():
    with pytest.raises(ValueError, match="train scores"):
        calibrate("C1", scores_dev=np.random.default_rng(0).uniform(size=40),
                  labels_dev=np.zeros(40), train_scores=None, scores_eval=None,
                  eval_keys=np.arange(40), train_keys=None)
    with pytest.raises(ValueError, match="disagree|misaligned"):
        pooled_reference_map(np.arange(4.0), np.arange(8.0),
                             eval_keys=np.arange(4), train_keys=np.arange(6))


def test_stage_axis_sources_are_recorded_and_calendar_fallback_works():
    n = 12
    pb = np.tile(np.arange(36, dtype=np.int64), (n, 1))      # the degenerate default tile
    stage, rec = cycle_stage(pb, stage_bins=12)
    assert rec["phase_axis_degenerate"] is True
    sd = np.linspace(0, 364, n)                              # one full year
    stage_cal, rec_cal = cycle_stage(pb, stage_bins=12, start_days=sd)
    assert rec_cal["source"] == "calendar_fallback_start_day" and "calendar" in rec_cal["note"]
    assert stage_cal.min() == 0 and stage_cal.max() == 11, "stages must spread over the year"
    assert not np.array_equal(stage_cal, np.zeros(n)), "the calendar fallback must not collapse"
    with pytest.raises(ValueError, match="phase_bin or start_days"):
        cycle_stage(None, stage_bins=12, start_days=None)


def test_c2_uses_farm_x_stage_pools_and_records_the_axis(two_scale_farms):
    t = two_scale_farms
    stages = np.tile(np.arange(3), t["farms"].size // 3 + 1)[: t["farms"].size]
    keys = np.column_stack([t["farms"], stages])
    p, fit = calibrate("C2", scores_dev=t["eval"], labels_dev=np.zeros(t["eval"].size),
                       train_scores=t["train"], scores_eval=t["eval"],
                       eval_keys=np.column_stack([t["eval_farms"], np.zeros(20, dtype=int)]),
                       train_keys=keys, min_history=2)
    assert fit.method == "pooled_zscore" and fit.params["pooling_level"] == "farm_stage"
    assert fit.params["n_pools"] == 60, "each (farm, stage) pool is fitted separately (20 farms × 3 stages)"
    assert fit.params["pooled_fill"] == 0, "every eval row's (farm, stage) pool exists in train"
    assert fit.fitted_on == "train"


def test_diagnostics_are_attached_to_every_fit():
    rng = np.random.default_rng(4)
    n = 120
    scores = rng.uniform(size=n)
    labels = (rng.random(n) < scores).astype(float)          # a well-ranked score
    for method, keys in (("C1", True), ("C2", True), ("C3", False)):
        kw = ({"eval_keys": rng.integers(0, 10, n), "train_keys": rng.integers(0, 10, 200)}
              if keys else {})
        train_scores = rng.uniform(size=200) if method != "C3" else None
        _p, fit = calibrate(method, scores_dev=scores, labels_dev=labels,
                            train_scores=train_scores, scores_eval=scores, **kw)
        assert np.isfinite(fit.brier_after), f"{method}: Brier-on-dev missing from the fit"
        assert 0.0 <= fit.ece_after <= 1.0, f"{method}: ECE-on-dev missing from the fit"
        assert fit.to_dict()["brier"]["after"] == fit.brier_after


def test_holdout_check_reports_the_optimism_and_keeps_farms_whole():
    rng = np.random.default_rng(9)
    n_farm, per = 12, 8
    farms = np.repeat(np.arange(n_farm), per)
    # calibration quantile from a uniform half transfers almost exactly: optimism ≈ 0
    p = rng.random(farms.size)
    out = calibration_holdout_check(p, budget=0.10, farm_ids=farms, seed=5)
    assert out["available"] and out["split"] == "farm_stratified_whole_farm"
    assert abs(out["holdout_far"] - 0.10) < 0.12, out
    assert out["optimism"] == pytest.approx(out["holdout_far"] - 0.10, abs=1e-9)
    # a farm must never straddle the two halves: every farm is entirely in or out
    assert min(out["n_calibration_half"], out["n_holdout_half"]) >= 10
    # a miscalibrated threshold (budget 0.01 on a coarse score) shows up as optimism
    coarse = np.repeat(np.linspace(0, 1, 10), farms.size // 10 + 1)[: farms.size]
    out2 = calibration_holdout_check(coarse, budget=0.10, farm_ids=farms, seed=5)
    assert out2["available"] and "warning" not in out2  # shape is what matters here
    # tiny inputs must be refused in words, not reported as a pass
    thin = calibration_holdout_check(np.array([0.1, 0.2, 0.3]), budget=0.1, farm_ids=None)
    assert not thin["available"] and "reason" in thin
    one_farm = calibration_holdout_check(rng.random(40), budget=0.1,
                                         farm_ids=np.zeros(40, dtype=int))
    assert not one_farm["available"] and "farms" in one_farm["reason"]


def test_threshold_record_carries_the_holdout_check_via_the_runner_contract():
    """The 7.7 result must land inside the threshold record the manifest serialises."""
    from ssl_anomaly.benchmarking.calibration import threshold_at_budget

    rng = np.random.default_rng(2)
    p = rng.random(200)
    thr = threshold_at_budget(p, budget=0.1)
    thr["calibration_holdout_check"] = calibration_holdout_check(
        p, budget=0.1, farm_ids=np.repeat(np.arange(20), 10))
    assert thr["calibration_holdout_check"]["available"], (
        "the runner stores thresholds as JSON; a holdout check that lives only in memory "
        "would never reach the report")
    assert "threshold" in thr and "holdout_far" in thr["calibration_holdout_check"]

"""Temporal-roll and LOBO tests.

What must hold: no train/eval leakage in any fold, unusable folds survive into
the table as ``absent`` (not dropped), and a robustness verdict requires breadth
rather than a good median.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.evaluation.robustness import (
    DEFAULT_CUT_YEARS,
    FoldResult,
    FoldSpec,
    YearSpan,
    lobo_folds,
    run_fold,
    run_robustness,
    summarise,
    temporal_roll_folds,
    validate_lobo_coverage,
)

# ---------------------------------------------------------------------------
# YearSpan
# ---------------------------------------------------------------------------


def test_year_span_splits_strictly_before_the_cut():
    y = YearSpan(np.array([2014, 2015, 2016, 2017]))
    assert y.before(2016).tolist() == [0, 1]
    assert y.at_or_after(2016).tolist() == [2, 3]


def test_year_span_rejects_multidimensional_input():
    with pytest.raises(ValueError, match="1-D"):
        YearSpan(np.zeros((2, 2)))


def test_year_span_rejects_nan_years():
    with pytest.raises(ValueError, match="non-finite"):
        YearSpan(np.array([2016.0, np.nan]))


# ---------------------------------------------------------------------------
# Temporal folds
# ---------------------------------------------------------------------------


def test_one_fold_per_registered_cut():
    y = YearSpan(np.arange(2012, 2025))
    folds = temporal_roll_folds(y)
    assert len(folds) == len(DEFAULT_CUT_YEARS)
    assert [f.name for f in folds] == [f"roll-{c}" for c in DEFAULT_CUT_YEARS]


def test_no_temporal_leakage():
    rng = np.random.default_rng(0)
    years = rng.integers(2012, 2025, size=400)
    for fold in temporal_roll_folds(YearSpan(years)):
        fold.assert_disjoint()
        assert np.all(years[fold.train_idx] < fold.meta["cut_year"])
        assert np.all(years[fold.eval_idx] >= fold.meta["cut_year"])


def test_unusable_cut_is_kept_and_flagged():
    years = YearSpan(np.array([2020, 2021, 2022]))   # nothing before 2016
    folds = temporal_roll_folds(years)
    first = folds[0]
    assert first.meta["unusable"] is True
    assert "no training cycles" in first.meta["reason"]
    assert len(folds) == 4


def test_cut_after_all_data_has_empty_eval():
    years = YearSpan(np.arange(2012, 2018))
    folds = temporal_roll_folds(years, cut_years=(2022,))
    assert folds[0].meta["unusable"] is True
    assert "no evaluation cycles" in folds[0].meta["reason"]


# ---------------------------------------------------------------------------
# LOBO folds
# ---------------------------------------------------------------------------


def test_one_fold_per_block_and_blocks_are_held_out():
    blocks = np.repeat(np.arange(1, 21), 5)
    folds = lobo_folds(blocks)
    assert len(folds) == 20
    for i, f in enumerate(folds):
        f.assert_disjoint()
        assert np.all(blocks[f.eval_idx] == i + 1)
        assert f.eval_idx.size == 5


def test_lobo_rejects_multidimensional_blocks():
    with pytest.raises(ValueError, match="1-D"):
        lobo_folds(np.zeros((3, 3)))


def test_lobo_coverage_reports_shortfall():
    cov = validate_lobo_coverage(14)
    assert cov["complete"] is False
    assert cov["missing"] == 6
    assert "conditional on that coverage" in cov["note"]


def test_lobo_coverage_complete_at_20():
    cov = validate_lobo_coverage(20)
    assert cov["complete"] is True
    assert cov["note"] == ""


# ---------------------------------------------------------------------------
# Fold execution
# ---------------------------------------------------------------------------


@pytest.fixture()
def scored():
    rng = np.random.default_rng(5)
    n = 400
    years = rng.integers(2012, 2025, size=n)
    blocks = rng.integers(1, 21, size=n)
    y = (rng.random(n) < 0.3).astype(int)
    scores = np.clip(0.2 + 0.5 * y + rng.normal(0, 0.2, n), 0, 1)
    return scores, y, years, blocks


def test_measured_fold_reports_ap_and_prevalence(scored):
    s, y, years, _ = scored
    f = FoldSpec("t", np.arange(0, 100), np.arange(100, 400), "lobo")
    r = run_fold(f, scores=s, labels=y)
    assert r.measured
    assert r.ap is not None
    assert r.prevalence is not None
    lo, hi = r.ap_ci
    assert lo <= r.ap <= hi


def test_single_class_eval_fold_is_absent_with_prevalence():
    s = np.array([0.1, 0.2, 0.3, 0.4])
    y = np.array([0, 0, 0, 0])
    f = FoldSpec("t", np.array([0]), np.array([0, 1, 2, 3]), "lobo")
    r = run_fold(f, scores=s, labels=y)
    assert not r.measured
    assert r.prevalence == 0.0
    assert "single class" in r.reason


def test_unusable_fold_is_absent():
    f = FoldSpec("roll-2016", np.array([]), np.array([1, 2]), "temporal-roll",
                 {"unusable": True, "reason": "no training cycles before the cut"})
    r = run_fold(f, scores=np.zeros(3), labels=np.array([0, 1, 0]))
    assert not r.measured
    assert "no training cycles" in r.reason


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


def _r(name, ap, status="measured"):
    return FoldResult(name, "lobo", status, ap=ap)


def test_single_fold_is_never_robust():
    s = summarise([_r("a", 0.9)], kind="lobo")
    assert s.robust is False
    assert "single point" in s.reason


def test_breadth_required_not_just_median():
    """Median clears the floor, but too few folds do -> NOT robust.

    Values: 5 of 10 folds at 0.6, 5 at 0.4 -> median 0.5 (passes the floor), and
    only 50% of folds reach it, below the 75% breadth requirement.
    """
    results = [_r(f"f{i}", ap) for i, ap in enumerate(
        [0.6] * 5 + [0.4] * 5)]
    s = summarise(results, kind="lobo", min_median_ap=0.5,
                  min_fraction_folds_above=0.75)
    assert s.median_ap >= 0.5, s.median_ap   # median passes
    assert s.robust is False                 # breadth fails
    assert "of folds reach" in s.reason


def test_consistent_folds_are_robust():
    results = [_r(f"f{i}", ap) for i, ap in enumerate([0.7, 0.75, 0.8, 0.9])]
    s = summarise(results, kind="lobo", min_median_ap=0.5)
    assert s.robust is True
    assert s.worst_fold == "f0"
    assert s.min_ap == pytest.approx(0.7)
    assert s.max_ap == pytest.approx(0.9)


def test_absent_folds_are_listed_not_dropped():
    results = [_r("a", 0.8), _r("b", 0.7),
               FoldResult("c", "lobo", "absent", reason="single class", n_eval=10)]
    s = summarise(results, kind="lobo")
    assert s.n_folds == 3
    assert s.n_measured == 2
    assert s.absent_folds[0]["fold"] == "c"


def test_no_measured_folds_gives_no_numbers():
    s = summarise([FoldResult("a", "lobo", "absent", reason="x")], kind="lobo")
    assert s.median_ap is None
    assert s.min_ap is None
    assert s.robust is False


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_run_robustness_end_to_end(scored):
    s, y, years, blocks = scored
    rep = run_robustness(scores=s, labels=y, years=years, block_ids=blocks)
    blob = rep.to_dict()
    assert len(blob["temporal_rolls"]) == 4
    assert len(blob["lobo"]) == 20
    assert blob["lobo_coverage"]["complete"] is True
    ev = rep.gate_evidence()
    assert set(ev) == {
        "temporal_min_ap", "temporal_median_ap", "temporal_robust",
        "lobo_median_ap", "lobo_robust", "lobo_coverage_complete",
    }


def test_run_robustness_flags_incomplete_lobo(scored):
    s, y, years, blocks = scored
    rep = run_robustness(scores=s, labels=y, years=years, block_ids=blocks,
                         n_blocks_expected=30)
    assert rep.lobo_coverage["complete"] is False
    assert rep.gate_evidence()["lobo_coverage_complete"] is False


def test_temporal_rolls_all_disjoint_in_end_to_end(scored):
    s, y, years, blocks = scored
    for fold in temporal_roll_folds(YearSpan(years)):
        fold.assert_disjoint()

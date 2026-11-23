"""Metric discipline: AUPRC/prevalence, clustered bootstrap, corrections, effect sizes.

Spec anchors:
- ``downstream-transfer-evaluation`` / "Metric reporting discipline".
- ``experimental-validation`` / "AUPRC as primary metric", "Secondary metrics".
- AGENTS.md §7 "Statistics and reporting": small-n, single-seed, dependence-blind
  bootstrap and effect sizes without CIs are review failures.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.evaluation.metrics import (
    auprc,
    auroc,
    benjamini_hochberg,
    bootstrap_ci,
    cliffs_delta,
    clustered_bootstrap_diff,
    f1_at_threshold,
    holm_correction,
    masked_trajectory_error,
    rank_biserial,
    select_threshold_on_validation,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# AUPRC / AUROC
# ---------------------------------------------------------------------------


def test_auprc_perfect_and_chance():
    y = np.array([1, 1, 0, 0, 0, 0])
    ap, prev = auprc(np.array([0.9, 0.8, 0.1, 0.2, 0.0, 0.3]), y)
    assert ap == pytest.approx(1.0)
    assert prev == pytest.approx(2 / 6)


def test_auprc_matches_sklearn_average_precision():
    """Cross-check against sklearn on random data with prevalence recorded."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(0)
    for _ in range(5):
        n = 200
        y = rng.random(n) < 0.2
        s = rng.random(n) + 0.4 * y
        ap, prev = auprc(s, y)
        sk = average_precision_score(y, s)
        assert ap == pytest.approx(sk, abs=1e-9)
        assert prev == pytest.approx(y.mean())


def test_auprc_single_class_cell_is_nan_not_half():
    ap, prev = auprc(np.array([0.1, 0.2, 0.3]), np.array([True, True, True]))
    assert np.isnan(ap)
    assert prev == pytest.approx(1.0)


def test_prevalence_is_always_reported():
    _, prev = auprc(np.array([0.5, 0.1]), np.array([1, 0]))
    assert 0.0 < prev < 1.0


def test_ties_are_not_split_across_thresholds():
    """All-equal scores give AP == prevalence (the random-ranking result)."""
    s = np.full(10, 0.5)
    y = np.array([1, 0, 1, 0, 1, 0, 0, 0, 1, 0])
    ap, prev = auprc(s, y)
    assert ap == pytest.approx(prev)


def test_auroc_bounds_and_ties():
    y = np.array([1, 0, 1, 0])
    assert auroc(np.array([0.9, 0.1, 0.8, 0.2]), y) == pytest.approx(1.0)
    assert auroc(np.array([0.5, 0.5, 0.5, 0.5]), y) == pytest.approx(0.5)
    assert np.isnan(auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_auprc_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must align"):
        auprc(np.zeros(3), np.zeros(4, dtype=bool))


# ---------------------------------------------------------------------------
# Threshold governance
# ---------------------------------------------------------------------------


def test_threshold_selected_on_validation_only():
    val_s = np.array([0.9, 0.8, 0.2, 0.1])
    val_y = np.array([1, 1, 0, 0])
    t = select_threshold_on_validation(val_s, val_y)
    assert np.isfinite(t)
    f1 = f1_at_threshold(val_s, val_y, t)["f1"]
    assert f1 == pytest.approx(1.0)


def test_threshold_tie_breaks_to_higher_threshold():
    """Deterministic conservative rule: equal F1 -> higher (fewer alarms) threshold."""
    s = np.array([0.9, 0.7, 0.5, 0.3])
    y = np.array([1, 1, 0, 0])
    t = select_threshold_on_validation(s, y, grid=5)
    assert t >= 0.5


def test_threshold_absent_when_no_positives():
    assert np.isnan(select_threshold_on_validation(np.array([0.1, 0.2]), np.zeros(2, bool)))


def test_f1_nan_when_undefined():
    out = f1_at_threshold(np.array([0.1]), np.array([False]), threshold=0.5)
    assert np.isnan(out["f1"]) and np.isnan(out["recall"])


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def test_wilson_interval_matches_known_values():
    lo, hi = wilson_interval(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3)
    assert hi == pytest.approx(0.7634, abs=1e-3)
    lo, hi = wilson_interval(0, 10)
    assert lo == 0.0 and 0.0 < hi < 0.32


def test_wilson_zero_n_is_nan():
    lo, hi = wilson_interval(0, 0)
    assert np.isnan(lo) and np.isnan(hi)


def test_bootstrap_ci_contains_point():
    rng = np.random.default_rng(1)
    x = rng.normal(1.0, 0.1, 200)
    point, lo, hi = bootstrap_ci(x, seed=3)
    assert lo < point < hi
    assert point == pytest.approx(1.0, abs=0.05)


def test_bootstrap_ci_empty_is_nan():
    assert all(np.isnan(v) for v in bootstrap_ci(np.array([])))


# ---------------------------------------------------------------------------
# Clustered bootstrap
# ---------------------------------------------------------------------------


def test_clustered_bootstrap_detects_real_difference():
    rng = np.random.default_rng(2)
    clusters = np.repeat(np.arange(20), 5)
    a = rng.normal(1.0, 0.2, 100)
    b = rng.normal(0.5, 0.2, 100)
    out = clustered_bootstrap_diff(a, b, clusters, n_boot=400, seed=5)
    assert out["point"] == pytest.approx(0.5, abs=0.1)
    assert out["lo"] < out["point"] < out["hi"]
    assert out["p_two_sided"] < 0.05
    assert out["n_clusters"] == 20


def test_clustered_bootstrap_null_is_not_significant():
    """Identical arrays -> exactly zero difference -> never significant."""
    rng = np.random.default_rng(3)
    clusters = np.repeat(np.arange(20), 5)
    a = rng.normal(0.0, 0.2, 100)
    out = clustered_bootstrap_diff(a, a.copy(), clusters, n_boot=400, seed=7)
    assert out["point"] == pytest.approx(0.0)
    assert out["lo"] <= 0.0 <= out["hi"]
    assert out["p_two_sided"] >= 0.9


def test_clustered_bootstrap_single_cluster_is_absent():
    out = clustered_bootstrap_diff(np.array([1.0, 2.0]), np.array([0.0, 1.0]),
                                   np.array([1, 1]))
    assert np.isnan(out["lo"]) and out["n_clusters"] == 1


def test_clustered_bootstrap_resamples_clusters_not_points():
    """Two clusters with opposite signs: interval must be wide, not razor-thin."""
    a = np.array([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 0.0, 0.0, 10.0, 10.0, 10.0])
    clusters = np.array([1, 1, 1, 2, 2, 2])
    out = clustered_bootstrap_diff(a, b, clusters, n_boot=300, seed=1)
    assert out["point"] == pytest.approx(0.0)
    assert out["hi"] - out["lo"] > 5.0  # point-resampling would give ~0 width


def test_clustered_bootstrap_validates_lengths():
    with pytest.raises(ValueError, match="same length"):
        clustered_bootstrap_diff(np.zeros(3), np.zeros(4), np.zeros(3))


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------


def test_holm_monotone_and_bounded():
    p = np.array([0.001, 0.01, 0.04, 0.5])
    adj = holm_correction(p)
    assert np.all(adj >= p - 1e-12)
    assert np.all(adj <= 1.0)
    assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)


def test_holm_known_values():
    p = np.array([0.01, 0.02, 0.03])
    adj = holm_correction(p)
    assert adj[0] == pytest.approx(0.03)
    assert adj[1] == pytest.approx(0.04)
    assert adj[2] == pytest.approx(0.04)


def test_bh_known_values():
    p = np.array([0.01, 0.02, 0.03, 0.04])
    adj = benjamini_hochberg(p)
    assert adj[3] == pytest.approx(0.04)
    assert np.all(adj >= p - 1e-12)


def test_corrections_handle_empty():
    assert holm_correction(np.array([])).size == 0
    assert benjamini_hochberg(np.array([])).size == 0


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def test_cliffs_delta_extremes_and_labels():
    d, label = cliffs_delta(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0]))
    assert d == pytest.approx(1.0) and label == "large"
    d, label = cliffs_delta(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    assert d == pytest.approx(0.0) and label == "negligible"


def test_cliffs_delta_empty_is_absent():
    d, label = cliffs_delta(np.array([]), np.array([1.0]))
    assert np.isnan(d) and label == "absent"


def test_rank_biserial_sign():
    a = np.array([3.0, 4.0, 5.0, 6.0])
    b = np.array([1.0, 2.0, 2.0, 3.0])
    assert rank_biserial(a, b) == pytest.approx(1.0)
    assert rank_biserial(b, a) == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Trajectory quality (T4)
# ---------------------------------------------------------------------------


def test_masked_trajectory_error_ignores_unobserved():
    """An unobserved target bin must not enter numerator OR denominator."""
    T, H, C = 4, 3, 1
    pred = np.zeros((1, T, H, C))
    truth = np.zeros((1, T, C))
    truth[0, 1, 0] = 2.0     # observed target, forecast 0 -> error 2
    truth[0, 3, 0] = 99.0    # NOT observed -> must contribute nothing
    mask = np.zeros((1, T, C), dtype=bool)
    mask[0, 1, 0] = True
    out = masked_trajectory_error(pred, truth, mask, horizons=(1, 3))
    assert out["mae_h1"] == pytest.approx(2.0)
    assert np.isnan(out["mae_h3"])   # h=3 has no observed target at all


def test_masked_trajectory_error_absent_horizon_is_nan():
    pred = np.zeros((1, 5, 1, 1))
    truth = np.zeros((1, 5, 1))
    mask = np.zeros((1, 5, 1), dtype=bool)   # nothing observed
    out = masked_trajectory_error(pred, truth, mask, horizons=(1,))
    assert np.isnan(out["mae_h1"])


def test_masked_trajectory_error_shape_check():
    with pytest.raises(ValueError, match="shape mismatch|must both be"):
        masked_trajectory_error(np.zeros((1, 4, 2, 1)), np.zeros((1, 3, 1)),
                                np.zeros((1, 3, 1), dtype=bool))


def test_masked_trajectory_error_rejects_bad_ndim():
    with pytest.raises(ValueError, match="must be \\(N, T, H, C\\)"):
        masked_trajectory_error(np.zeros((1, 3, 1)), np.zeros((1, 3, 1)),
                                np.zeros((1, 3, 1), dtype=bool))


# ---------------------------------------------------------------------------
# Metric-level (not mean-level) paired bootstrap
# ---------------------------------------------------------------------------


def test_metric_diff_uses_the_statistic_not_the_mean():
    """Two score sets with equal means but different AP must not give delta ~0.

    This is the ladder bug this function exists to prevent: differencing mean
    predicted probability reported +0.00 for rungs whose APs were 0.51 and 1.00.
    """
    from ssl_anomaly.evaluation.metrics import (
        auprc,
        clustered_bootstrap_metric_diff,
    )

    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    perfect = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
    # A constant 0.5 scores every item identically -> AP = prevalence.
    flat = np.full(8, 0.5)
    clusters = np.repeat([1, 2, 3, 4], 2)
    out = clustered_bootstrap_metric_diff(
        perfect, flat, y, clusters, lambda s, yy: auprc(s, yy)[0],
        n_boot=200, seed=3,
    )
    assert abs(np.mean(perfect) - np.mean(flat)) < 0.5   # means are not wildly apart
    assert out["point"] == pytest.approx(auprc(perfect, y)[0] - auprc(flat, y)[0])
    assert out["point"] > 0.2
    assert out["n_clusters"] == 4


def test_metric_diff_identical_scores_is_zero():
    from ssl_anomaly.evaluation.metrics import (
        auprc,
        clustered_bootstrap_metric_diff,
    )

    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 60)
    s = rng.random(60)
    c = np.repeat(np.arange(12), 5)
    out = clustered_bootstrap_metric_diff(
        s, s, y, c, lambda a, b: auprc(a, b)[0], n_boot=150, seed=1)
    assert out["point"] == pytest.approx(0.0, abs=1e-12)
    assert out["lo"] <= 0.0 <= out["hi"]


def test_metric_diff_rejects_misaligned_inputs():
    from ssl_anomaly.evaluation.metrics import clustered_bootstrap_metric_diff

    with pytest.raises(ValueError, match="must all align"):
        clustered_bootstrap_metric_diff(np.zeros(3), np.zeros(4), np.zeros(3),
                                        np.zeros(3), lambda a, b: 0.0)


def test_holm_labelled_preserves_names_and_order():
    from ssl_anomaly.evaluation.metrics import holm_correction_labelled

    out = holm_correction_labelled({"B2→B3": 0.01, "B3→B4": 0.40, "B2→B4": 0.02})
    assert list(out) == ["B2→B3", "B3→B4", "B2→B4"]
    assert out["B2→B3"] <= out["B3→B4"]
    assert all(0.0 <= v <= 1.0 for v in out.values())


def test_holm_labelled_empty_is_empty():
    from ssl_anomaly.evaluation.metrics import holm_correction_labelled

    assert holm_correction_labelled({}) == {}

"""Level-1 pretraining-quality battery tests.

Uses synthetic embeddings with KNOWN properties so each probe is validated
against the answer it should produce (spec: "unit tests for each probe
(synthetic embeddings with known properties)").
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.evaluation.data import synthetic_evaluation_split
from ssl_anomaly.evaluation.level1 import (
    collapse_diagnostics,
    farm_separability,
    macro_f1,
    masked_reconstruction,
    phase_bin_targets,
    phase_linear_probe,
    run_level1,
    temporal_stability,
)


@pytest.fixture()
def fixture():
    return synthetic_evaluation_split(n_train=24, n_dev=16, n_fields=6, seed=13)


# ---------------------------------------------------------------------------
# Fixture contract
# ---------------------------------------------------------------------------


def test_fixture_is_mask_consistent(fixture):
    observed = fixture.M
    assert np.all(np.isfinite(fixture.X[observed]))
    assert np.all(np.isnan(fixture.X[~observed]))
    assert fixture.data_source == "synthetic-fixture"


def test_fixture_has_both_splits(fixture):
    assert fixture.for_split("train").n_cycles == 24
    assert fixture.for_split("dev").n_cycles == 16


# ---------------------------------------------------------------------------
# Masked reconstruction
# ---------------------------------------------------------------------------


def test_perfect_reconstruction_is_zero_nrmse(fixture):
    rep = masked_reconstruction(fixture, lambda X, M: X)
    assert np.allclose(rep.nrmse_per_channel, 0.0)


def test_zero_predictor_nrmse_is_one(fixture):
    """NRMSE is scale-free: predicting zeros gives exactly 1.0 on every channel."""
    rep = masked_reconstruction(fixture, lambda X, M: np.zeros_like(X))
    assert np.allclose(rep.nrmse_per_channel, 1.0), rep.nrmse_per_channel


def test_unobserved_entries_do_not_affect_nrmse(fixture):
    """Corrupting an unobserved cell must change nothing (no denominator leak)."""
    base = masked_reconstruction(fixture, lambda X, M: np.zeros_like(X))
    X2 = fixture.X.copy()
    X2[~fixture.M] = np.nan  # already NaN
    rep = masked_reconstruction(
        type(fixture)(X=X2, M=fixture.M, field_ids=fixture.field_ids,
                      cycle_ids=fixture.cycle_ids, split=fixture.split,
                      dsh=fixture.dsh, channel_names=fixture.channel_names),
        lambda X, M: np.zeros_like(X),
    )
    assert np.allclose(rep.nrmse_per_channel, base.nrmse_per_channel, equal_nan=True)


def test_channel_with_no_observations_is_nan():
    data = synthetic_evaluation_split(n_train=4, n_dev=2, n_fields=2, n_channels=3, seed=1)
    M = data.M.copy()
    M[:, :, 2] = False
    X = data.X.copy()
    X[:, :, 2] = np.nan
    from ssl_anomaly.evaluation.data import CycleData

    d2 = CycleData(X=X, M=M, field_ids=data.field_ids, cycle_ids=data.cycle_ids,
                   split=data.split, dsh=data.dsh, channel_names=data.channel_names)
    rep = masked_reconstruction(d2, lambda X, M: np.zeros_like(X))
    assert np.isnan(rep.nrmse_per_channel[2])
    assert rep.n_observed_per_channel[2] == 0


def test_reconstruction_shape_mismatch_raises(fixture):
    with pytest.raises(ValueError, match="must match"):
        masked_reconstruction(fixture, lambda X, M: X[:, :-1, :])


# ---------------------------------------------------------------------------
# Farm separability
# ---------------------------------------------------------------------------


def _field_clustered_embeddings(field_ids, *, spread=0.02, seed=0):
    """Embeddings where each farm is a tight cluster — a perfect probe case."""
    rng = np.random.default_rng(seed)
    centres = {f: rng.normal(size=8) for f in np.unique(field_ids)}
    return np.stack([centres[f] + rng.normal(0, spread, 8) for f in field_ids])


def test_farm_separability_perfect_clusters():
    f = np.repeat([1, 2, 3, 4], 6)
    z = _field_clustered_embeddings(f)
    rep = farm_separability(z, f)
    assert rep.top1 == pytest.approx(1.0)
    assert rep.top5 == pytest.approx(1.0)
    assert rep.n_classes == 4
    assert rep.chance_top1 == pytest.approx(0.25)


def test_farm_separability_random_embeddings_is_near_chance():
    rng = np.random.default_rng(7)
    f = np.repeat(np.arange(1, 11), 8)     # 10 classes, 80 cycles
    z = rng.normal(size=(80, 8))           # no farm structure at all
    rep = farm_separability(z, f)
    assert rep.top1 < rep.chance_top1 * 3  # well below a real signal
    assert rep.chance_top1 == pytest.approx(0.1)


def test_farm_separability_reports_subset_hash_and_ci():
    f = np.repeat([1, 2, 3], 4)
    z = _field_clustered_embeddings(f)
    rep = farm_separability(z, f, subset_hash="abc123")
    assert rep.subset_hash == "abc123"
    lo, hi = rep.top1_ci
    assert lo <= rep.top1 <= hi


def test_farm_separability_needs_three_cycles():
    with pytest.raises(ValueError, match="at least 3"):
        farm_separability(np.zeros((2, 4)), np.array([1, 2]))


# ---------------------------------------------------------------------------
# Phase probe
# ---------------------------------------------------------------------------


def test_phase_probe_beats_majority_when_target_is_recoverable():
    rng = np.random.default_rng(3)
    n_classes, n = 36, 200
    y = rng.integers(0, n_classes, size=n)
    # Embedding is a noisy one-hot of the class -> recoverable.
    z = np.eye(n_classes)[y] + rng.normal(0, 0.05, (n, n_classes))
    rep = phase_linear_probe(z[:120], y[:120], z[120:], y[120:], n_classes=n_classes)
    assert rep.accuracy > 0.8
    assert rep.accuracy > rep.majority_accuracy


def test_phase_probe_on_random_embeddings_is_near_majority():
    rng = np.random.default_rng(4)
    n = 160
    y = rng.integers(0, 6, size=n)
    z = rng.normal(size=(n, 5))
    rep = phase_linear_probe(z[:100], y[:100], z[100:], y[100:], n_classes=6)
    assert rep.accuracy < 0.5


def test_phase_probe_records_axis_and_counts():
    rng = np.random.default_rng(5)
    y = rng.integers(0, 4, size=40)
    z = rng.normal(size=(40, 3))
    rep = phase_linear_probe(z[:20], y[:20], z[20:], y[20:], n_classes=4, axis="gdd")
    assert rep.axis == "gdd"
    assert rep.n_train == 20 and rep.n_eval == 20


def test_phase_probe_empty_split_raises():
    with pytest.raises(ValueError, match="non-empty"):
        phase_linear_probe(np.zeros((0, 3)), np.zeros(0, dtype=int),
                           np.zeros((4, 3)), np.zeros(4, dtype=int), n_classes=4)


def test_phase_bin_targets_clip_and_floor():
    dsh = np.array([0.0, 14.9, 15.0, 539.0, 600.0])
    bins = phase_bin_targets(dsh, n_bins=36, bin_days=15.0)
    assert bins.tolist() == [0, 0, 1, 35, 35]


def test_macro_f1_ignores_absent_classes():
    """Only classes PRESENT in truth are averaged (absent != scored zero)."""
    y = np.array([0, 0, 1, 1])
    assert macro_f1(y, y, n_classes=5) == pytest.approx(1.0)
    # predictions all class 1: class0 F1=0, class1 P=0.5 R=1 -> F1=2/3
    assert macro_f1(y, np.array([1, 1, 1, 1]), n_classes=5) == pytest.approx((0 + 2 / 3) / 2)
    # a class absent from truth is not averaged in: only class 0 present
    assert macro_f1(np.zeros(4, dtype=int), np.zeros(4, dtype=int), n_classes=5) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Temporal stability
# ---------------------------------------------------------------------------


def test_stability_high_for_consistent_fields():
    f = np.repeat([1, 2, 3], 4)
    base = {1: [1.0, 0, 0], 2: [0, 1.0, 0], 3: [0, 0, 1.0]}
    z = np.stack([np.array(base[fid]) + 0.01 * i for i, fid in enumerate(f)])
    rep = temporal_stability(z, f, floor=0.5)
    assert rep.median_cosine > 0.9
    assert rep.tail_fraction_below_floor == 0.0
    assert rep.n_pairs == 3 * 6


def test_stability_excludes_single_cycle_fields():
    f = np.array([1, 1, 2])
    z = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    rep = temporal_stability(z, f)
    assert rep.n_fields_excluded == 1


def test_stability_no_pairs_is_nan():
    rep = temporal_stability(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1, 2]))
    assert np.isnan(rep.median_cosine)
    assert rep.n_pairs == 0


# ---------------------------------------------------------------------------
# Collapse diagnostics
# ---------------------------------------------------------------------------


def test_collapse_detects_constant_embedding():
    z = np.ones((50, 8))
    rep = collapse_diagnostics(z)
    assert rep.disqualified
    assert rep.triggers


def test_collapse_detects_low_rank():
    rng = np.random.default_rng(2)
    base = rng.normal(size=(50, 1))
    z = base @ np.ones((1, 8))       # rank 1 in an 8-dim space
    rep = collapse_diagnostics(z)
    assert rep.effective_rank_ratio == pytest.approx(1 / 8)
    # ratio 0.125 alone would pass a 0.10 floor -> the dominance trigger must fire
    assert rep.top_eigenvalue_share > 0.99
    assert rep.disqualified
    assert any("rank-1" in t for t in rep.triggers)


def test_healthy_embedding_is_not_disqualified():
    rng = np.random.default_rng(6)
    z = rng.normal(size=(300, 16))
    rep = collapse_diagnostics(z)
    assert not rep.disqualified, rep.triggers
    assert 0.3 < rep.effective_rank_ratio <= 1.0
    assert rep.top_eigenvalue_share < 0.99


def test_collapse_reports_spectrum_and_dim():
    rng = np.random.default_rng(8)
    rep = collapse_diagnostics(rng.normal(size=(60, 12)))
    assert rep.per_dim_std.size == 12
    assert rep.latent_dim == 12
    d = rep.to_dict()
    assert len(d["per_dim_std"]) == 12


def test_collapse_thresholds_are_configurable():
    rng = np.random.default_rng(9)
    z = rng.normal(size=(100, 10))
    strict = collapse_diagnostics(z, effective_rank_ratio_floor=2.0)
    assert strict.disqualified  # impossible bar trip


def test_collapse_needs_three_rows():
    with pytest.raises(ValueError, match="N>=3"):
        collapse_diagnostics(np.zeros((2, 4)))


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------


def _moment_embed(X, M):
    x = np.where(M, X, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(x, axis=1)
        std = np.nanstd(x, axis=1)
    return np.nan_to_num(np.hstack([mean, std]))


def _phase_targets(n: int, seed: int = 7) -> np.ndarray:
    """A VARYING per-cycle phase target, as a real caller would supply."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 36, size=n)


def test_run_level1_end_to_end(fixture):
    dev = fixture.for_split("dev")
    res = run_level1(dev=dev, embed_fn=_moment_embed,
                     phase_targets=_phase_targets(dev.n_cycles))
    blob = res.to_dict()
    assert blob["reconstruction"]["mean_nrmse"] == pytest.approx(1.0)
    assert 0.0 <= blob["farm_separability"]["top1"] <= 1.0
    assert blob["phase_probe"]["n_eval"] >= 1
    assert "effective_rank_ratio" in blob["collapse"]
    ev = res.gate_evidence()
    assert set(ev) == {
        "farm_separability_top1", "farm_separability_chance",
        "phase_probe_accuracy", "phase_probe_majority", "collapse_disqualified",
    }
    assert ev["phase_probe_accuracy"] is not None


def test_run_level1_with_b1_baseline_reports_both(fixture):
    dev = fixture.for_split("dev")
    b1 = np.nan_to_num(np.where(dev.M, dev.X, np.nan).mean(axis=1))
    res = run_level1(dev=dev, embed_fn=_moment_embed, b1_features=b1,
                     phase_targets=_phase_targets(dev.n_cycles))
    assert res.b1_farm_separability is not None
    assert res.b1_phase_probe is not None
    assert res.b1_farm_separability.subset_hash == "b1-features"


# ---------------------------------------------------------------------------
# The phase target cannot be derived from the cycle tensor — a hard-won lesson
# ---------------------------------------------------------------------------


def test_phase_probe_is_absent_without_an_explicit_target(fixture):
    """No mask-derived target exists: every full-season cycle covers all 36 bins.

    The first implementation derived the target from observed-bin midpoints, got a
    constant label, and reported accuracy 1.00 with a majority baseline of 1.00 —
    a probe that cannot fail. It is now absent unless a real target is supplied.
    """
    dev = fixture.for_split("dev")
    res = run_level1(dev=dev, embed_fn=_moment_embed)
    assert res.phase_probe is None
    assert "not derivable from the cycle tensor" in res.phase_probe_absent_reason
    assert res.to_dict()["phase_probe"] is None
    assert res.to_dict()["phase_probe_absent_reason"]


def test_absent_phase_probe_yields_null_gate_evidence(fixture):
    """G-A must see the term as missing, not as a passing 1.0."""
    dev = fixture.for_split("dev")
    ev = run_level1(dev=dev, embed_fn=_moment_embed).gate_evidence()
    assert ev["phase_probe_accuracy"] is None
    assert ev["phase_probe_majority"] is None
    assert ev["farm_separability_top1"] is not None


def test_degenerate_phase_target_is_rejected_as_absent(fixture):
    dev = fixture.for_split("dev")
    constant = np.zeros(dev.n_cycles, dtype=int)
    res = run_level1(dev=dev, embed_fn=_moment_embed, phase_targets=constant)
    assert res.phase_probe is None


def test_phase_target_with_wrong_length_raises(fixture):
    dev = fixture.for_split("dev")
    with pytest.raises(ValueError, match="phase_targets must be 1-D"):
        run_level1(dev=dev, embed_fn=_moment_embed,
                   phase_targets=np.zeros(dev.n_cycles - 1, dtype=int))


def test_probe_refuses_to_score_against_a_perfect_baseline():
    """Defence in depth in the probe itself, not only in the caller."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(40, 3))
    y = np.ones(40, dtype=int)         # single class -> 100% majority baseline
    with pytest.raises(ValueError, match="degenerate"):
        phase_linear_probe(z[:20], y[:20], z[20:], y[20:], n_classes=36)


def test_run_level1_rejects_wrong_embedding_shape(fixture):
    with pytest.raises(ValueError, match="embed_fn must return"):
        run_level1(dev=fixture.for_split("dev"), embed_fn=lambda X, M: np.zeros((3, 4)))


def test_level1_never_touches_test_split(fixture):
    """The fixture has no test cycles; asking for them must raise, not return []."""
    from ssl_anomaly.evaluation.data import TestSplitAccessError

    with pytest.raises(TestSplitAccessError, match="not readable"):
        fixture.indices_for_split("test")

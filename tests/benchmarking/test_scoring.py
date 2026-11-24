"""Mask-aware scoring and level aggregation (Phase 2.11 / 5.x)."""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.benchmarking.scoring import (
    ScoreBundle, aggregate_bins, detection_delay, event_onsets, masked_bin_error,
    patch_to_farm,
)


def _bundle(**kw):
    rng = np.random.default_rng(0)
    n, t = kw.pop("n", 20), kw.pop("t", 12)
    b = rng.normal(size=(n, t))
    return ScoreBundle(bin_scores=b, cycle_scores=np.nanmax(b, axis=1), **kw)


def test_bundle_requires_a_scorable_value():
    with pytest.raises(ValueError):
        ScoreBundle(bin_scores=np.full((4, 4), np.nan), cycle_scores=np.zeros(4))


def test_bundle_rejects_length_mismatch():
    with pytest.raises(ValueError):
        ScoreBundle(bin_scores=np.zeros((4, 5)), cycle_scores=np.zeros(3))


def test_unobserved_bins_are_skipped_not_zeroed():
    err = np.ones((2, 3, 4))
    mask = np.zeros((2, 3, 4), dtype=bool)
    mask[0, :, :] = True
    out = masked_bin_error(err, mask)
    assert np.allclose(out[0], 1.0)
    assert np.isnan(out[1]).all(), "a bin with no observed channel is absent, not perfect"


def test_partial_observation_uses_observed_channel_count():
    err = np.zeros((1, 1, 4)); err[0, 0] = [1, 1, 1, 1]
    m = np.zeros((1, 1, 4), dtype=bool); m[0, 0, :2] = True
    assert np.isclose(masked_bin_error(err, m)[0, 0], 1.0)


def test_scale_is_applied_per_channel():
    err = np.stack([np.full((1, 1), 2.0), np.full((1, 1), 20.0)], axis=-1)
    m = np.ones((1, 1, 2), dtype=bool)
    scaled = masked_bin_error(err, m, scale=np.array([2.0, 20.0]))
    assert np.allclose(scaled, 1.0)


def test_aggregate_methods():
    b = np.array([[0.1, 0.9, 0.2], [0.3, 0.4, 0.5]])
    assert np.isclose(aggregate_bins(b, method="max")[0], 0.9)
    assert np.isclose(aggregate_bins(b, method="trailing", trailing_window=1)[0], 0.2)
    assert np.isclose(aggregate_bins(b, method="mean")[1], 0.4)
    with pytest.raises(ValueError):
        aggregate_bins(b, method="attention")


def test_nan_bins_never_become_the_maximum():
    b = np.array([[np.nan, 0.3, np.nan]])
    assert np.isclose(aggregate_bins(b, method="max")[0], 0.3)


def test_patch_to_farm_excludes_missing_patches_from_both_sides():
    p = np.array([[1.0, np.nan], [np.nan, 2.0]])
    ok = np.array([[True, True], [True, True]])
    farm, valid = patch_to_farm(p, ok, method="channel_max")
    assert np.allclose(farm, [1.0, 2.0]) and valid.all()
    ok2 = np.array([[True, False], [False, True]])
    farm2, valid2 = patch_to_farm(np.array([[5.0, 5.0], [5.0, 5.0]]), ok2)
    assert np.allclose(farm2, 5.0) and valid2.all()
    farm3, valid3 = patch_to_farm(np.zeros((2, 2)), np.zeros((2, 2), dtype=bool))
    assert np.isnan(farm3).all() and not valid3.any(), "no valid patch must read as absent, not as zero"


def test_attention_pool_prefers_high_scores():
    from ssl_anomaly.benchmarking.scoring import attention_pool

    p = np.array([[0.1], [9.0]])
    ok = np.ones((2, 1), dtype=bool)
    assert attention_pool(p, ok, temperature=0.1)[0] > 8.0


def test_event_onsets_and_delays():
    lab = np.zeros((2, 8), dtype=bool)
    lab[0, 2:5] = True
    lab[1, 6:8] = True
    ev = event_onsets(lab, field_ids=np.array([7, 9]))
    assert [e["onset_bin"] for e in ev] == [2, 6] and ev[0]["field_id"] == 7
    sc = np.zeros((2, 8)); sc[0, 4] = 1.0; sc[1, 6] = 1.0
    d = detection_delay(sc, lab, threshold=0.5, bin_days=15)
    assert d["n_events"] == 2 and d["missed"] == 0
    assert d["median_days"] == 30.0 or d["median_days"] == 15.0     # event0 alerts at bin4 (2 bins late)
    assert d["event_class"] == "all"


def test_delay_reports_the_alerted_subset_only():
    lab = np.zeros((3, 4), dtype=bool); lab[:, 0:2] = True
    sc = np.zeros((3, 4)); sc[0, 1] = 5.0
    d = detection_delay(sc, lab, threshold=1.0, bin_days=15)
    assert d["n_events"] == 3 and d["n_alerted"] == 1 and d["missed"] == 2
    assert d["median_days"] == 15.0


def test_score_normaliser_refuses_degenerate_scales():
    from ssl_anomaly.benchmarking.scoring import ScoreNormaliser as SN

    with pytest.raises(ValueError):
        SN(scale=np.array([0.0]), fitted_on="train", n_cycles=1, n_observations=1)
    with pytest.raises(ValueError):
        SN(scale=np.array([np.nan]), fitted_on="train", n_cycles=1, n_observations=1)


def test_normaliser_from_errors_documents_its_partition(clean_tensors):
    from ssl_anomaly.benchmarking.scoring import ScoreNormaliser

    x, m = clean_tensors
    err = np.abs(x)
    n = ScoreNormaliser.from_errors(err, m, fitted_on="train", n_cycles=x.shape[0])
    assert n.fitted_on == "train" and np.all(n.scale > 0)
    assert "MAD" in n.to_dict()["estimator"]

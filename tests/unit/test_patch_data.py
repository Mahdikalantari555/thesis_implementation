"""Patch dataset: holdout isolation, frozen normalization, covariates.

Covers ``patch-level-controlled-trial`` task 1.3 and the
``training-holdout-invariant`` standing invariant on the patch path.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.io import patches as P
from ssl_anomaly.monitor import holdout as H
from ssl_anomaly.train import patch_data as PD

S = P.DEFAULT_PATCH_SIZE
T = P.N_BINS


def _store(tmp_path, fields=(101, 102, 103), cycles=3, seed=0):
    g = np.random.default_rng(seed)
    root = P.create_store(tmp_path / "patches.zarr", field_ids=list(fields), max_cycles=cycles)
    for i, f in enumerate(fields):
        root["field_ids"][i] = f
        infield = np.zeros((S, S), dtype=np.uint8)
        infield[4:20, 6:22] = 1
        root["in_field"][i, 0] = infield
        for c in range(cycles):
            vals = g.normal(size=(T, P.N_CHANNELS, S, S)).astype(np.float16)
            vals[..., infield == 0] = np.nan
            root["values"][i, c] = vals
            root["pixel_valid"][i, c, :, 0] = np.isfinite(vals).any(axis=1)
            root["n_obs"][i, c] = g.integers(1, 4, size=T)
            root["sensor_id"][i, c] = P.SENSOR_IDS["L8"]
            root["cycle_ids"][i, c] = 1000 + c
            root["cycle_start_day"][i, c] = 40 + 20 * c
            root["cycle_valid"][i, c] = 1
    return root


def _manifest(fields=(101, 102, 103), holdout=(103,), dates=None):
    return {
        "holdout_farms": list(holdout),
        "known_farm_ids": list(fields),
        "holdout_date_start": dates,
        "archive_end_date": "2022-12-31",
    }


def _norm():
    per_ch = {ch: {"median": 0.5, "iqr": 0.1, "n": 100} for ch in P.CHANNELS}
    return {name: dict(per_ch) for name in ("L8", "L9", "unknown")}


# --- holdout isolation ----------------------------------------------------
def test_holdout_field_never_enters_the_dataset(tmp_path):
    root = _store(tmp_path)
    ds = PD.PatchCycleDataset(root, [101, 102, 103], holdout_manifest=_manifest(), norm=_norm())
    assert set(ds.fields) == {101, 102}
    assert ds.excluded_by_holdout == [103]
    for i in range(len(ds)):
        assert ds.sample(i).field_id != 103


def test_missing_manifest_fails_closed(tmp_path):
    root = _store(tmp_path)
    with pytest.raises(ValueError, match="holdout_manifest is required"):
        PD.PatchCycleDataset(root, [101], holdout_manifest=None, norm=_norm())


def test_all_holdout_fields_is_an_error_not_an_empty_dataset(tmp_path):
    root = _store(tmp_path)
    with pytest.raises(ValueError, match="every requested field is a holdout"):
        PD.PatchCycleDataset(root, [103], holdout_manifest=_manifest(), norm=_norm())


def test_unknown_farm_is_excluded_by_default(tmp_path):
    """Future-by-default: a farm absent from known_farm_ids is not trainable."""
    root = _store(tmp_path, fields=(101, 999))
    with pytest.raises(ValueError, match="every requested field is a holdout"):
        PD.PatchCycleDataset(root, [999], holdout_manifest=_manifest(fields=(101,)), norm=_norm())


def test_holdout_is_reasserted_per_sample(tmp_path):
    """A store mutated after construction still cannot leak a holdout field."""
    root = _store(tmp_path)
    ds = PD.PatchCycleDataset(root, [101, 102], holdout_manifest=_manifest(), norm=_norm())
    ds._holdout = _manifest(holdout=(101, 102, 103))
    with pytest.raises(H.HoldoutViolationError):
        ds.sample(0)


def test_missing_field_is_a_loud_error(tmp_path):
    root = _store(tmp_path)
    with pytest.raises(ValueError, match="not in the store"):
        PD.PatchCycleDataset(root, [101, 555], holdout_manifest=_manifest(), norm=_norm())


# --- sample construction --------------------------------------------------
def test_sample_shapes_and_masking(tmp_path):
    root = _store(tmp_path)
    ds = PD.PatchCycleDataset(root, [101], holdout_manifest=_manifest(), norm=_norm())
    s = ds.sample(0)
    assert s.x.shape == (T, P.N_CHANNELS, S, S)
    assert s.obs.shape == s.x.shape
    assert s.pixel_valid.shape == (T, S, S)
    assert s.covariates.shape == (T, PD.N_COVARIATES)
    # an unobserved entry is exactly 0 and masked out
    assert np.all(s.x[~s.obs] == 0)
    # nothing outside the polygon is ever observed
    assert not bool(s.obs[:, :, :, :][..., :, 0:4].any())
    assert s.obs.sum() > 0


def test_no_nan_reaches_the_model(tmp_path):
    root = _store(tmp_path)
    ds = PD.PatchCycleDataset(root, [101], holdout_manifest=_manifest(), norm=_norm())
    for i in range(len(ds)):
        assert np.isfinite(ds.sample(i).x).all()


def test_pixel_valid_is_any_channel(tmp_path):
    root = _store(tmp_path)
    ds = PD.PatchCycleDataset(root, [101], holdout_manifest=_manifest(), norm=_norm())
    s = ds.sample(0)
    np.testing.assert_array_equal(s.pixel_valid, s.obs.any(axis=1))


def test_normalization_uses_the_frozen_artifact(tmp_path):
    root = _store(tmp_path)
    n1 = _norm()
    n2 = {k: {ch: {"median": 0.0, "iqr": 1.0, "n": 1} for ch in P.CHANNELS}
          for k in ("L8", "L9", "unknown")}
    s1 = PD.PatchCycleDataset(root, [101], holdout_manifest=_manifest(), norm=n1).sample(0)
    s2 = PD.PatchCycleDataset(root, [101], holdout_manifest=_manifest(), norm=n2).sample(0)
    assert not np.allclose(s1.x[s1.obs], s2.x[s2.obs])
    # unobserved stays exactly 0 under both normalizations
    assert np.all(s1.x[~s1.obs] == 0)
    assert np.all(s2.x[~s2.obs] == 0)


def test_missing_channel_stats_keeps_raw_units_and_flags_it(tmp_path):
    root = _store(tmp_path)
    norm = _norm()
    del norm["L8"]["nir"]
    ds = PD.PatchCycleDataset(root, [101], holdout_manifest=_manifest(), norm=norm)
    ds.sample(0)
    assert "nir" in ds.unnormalized_channels


def test_max_cycles_caps_the_pairs(tmp_path):
    root = _store(tmp_path, cycles=4)
    full = PD.PatchCycleDataset(root, [101], holdout_manifest=_manifest(), norm=_norm())
    capped = PD.PatchCycleDataset(root, [101], holdout_manifest=_manifest(), norm=_norm(),
                                  max_cycles=2)
    assert len(full) == 4
    assert len(capped) == 2


def test_collate_stacks_into_batches(tmp_path):
    root = _store(tmp_path)
    ds = PD.PatchCycleDataset(root, [101], holdout_manifest=_manifest(), norm=_norm())
    batch = PD.collate([ds.sample(0), ds.sample(1)])
    assert batch["x"].shape == (2, T, P.N_CHANNELS, S, S)
    assert batch["obs"].dtype == np.bool_
    assert batch["covariates"].shape[0] == 2


# --- covariates -----------------------------------------------------------
def test_covariate_shapes_and_ranges():
    cov = PD.build_covariates(
        np.full(T, 100), np.array([0, 1] * (T // 2)), np.full(T, 1), 0.4,
    )
    assert cov.shape == (T, PD.N_COVARIATES)
    assert np.isfinite(cov).all()
    # sin/cos are a unit circle
    np.testing.assert_allclose(cov[:, 0] ** 2 + cov[:, 1] ** 2, 1.0, atol=1e-5)
    # the sensor one-hot picks exactly one sensor
    assert np.all(cov[:, 5] + cov[:, 6] == 1.0)
    assert np.allclose(cov[:, 7], 0.4)


def test_days_since_valid_is_causal_and_capped():
    nobs = np.zeros(T, dtype=int)
    nobs[5] = 2
    nobs[6] = 1
    cov = PD.build_covariates(np.full(T, 0), nobs, np.zeros(T, dtype=int), 1.0)
    since = cov[:, 3] * T
    assert since[5] == 0
    assert since[6] == 0
    assert since[7] == 1.0
    assert since[0] == T          # never observed before bin 5 → capped


def test_covariate_bin_grid_matches_the_monitor_epoch():
    """doy_sin must agree with monitor.inputs.doy_channels at the same day."""
    from ssl_anomaly.monitor import inputs as minputs

    ref = minputs.doy_channels(np.array([0]), n_bins=T)[0]      # (T, 2)
    cov = PD.build_covariates(np.full(T, 0), np.ones(T, int), np.ones(T, int), 1.0)
    np.testing.assert_allclose(cov[:2, :2], ref[:2, :2], atol=1e-6)


def test_covariate_bad_shapes_are_loud():
    with pytest.raises(ValueError, match="scalar or"):
        PD.build_covariates(np.zeros(5), np.ones(T, int), np.ones(T, int), 1.0)
    with pytest.raises(ValueError, match="n_obs/sensor_id"):
        PD.build_covariates(np.zeros(T), np.ones(5, int), np.ones(T, int), 1.0)


def test_runtime_estimate_scales_linearly():
    one = PD.estimate_epoch_seconds(80, seconds_per_sample=0.2, batch_size=8)
    two = PD.estimate_epoch_seconds(160, seconds_per_sample=0.2, batch_size=8)
    assert two["seconds_per_epoch"] == pytest.approx(2 * one["seconds_per_epoch"])
    assert one["n_batches"] == 10

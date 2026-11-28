"""Task 0A.5 — normalization arms N0–N4: leakage, exclusion, sidecar, erasure."""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.normalization import (
    ARMS,
    NormalizationLeakageError,
    NormStats,
    apply_normalization,
    fit_normalization,
    load_norm_stats,
    normalization_erasure_record,
    save_norm_stats,
)

F = 3  # ndvi, lst, et0
FEATURES = ["ndvi", "lst", "eta_daily"]


def make_data(n=12, t=8, seed=3, *, deviated=False):
    rng = np.random.default_rng(seed)
    x = np.empty((n, t, F))
    x[:, :, 0] = rng.normal(0.5, 0.1, size=(n, t))          # ndvi ~0.5
    x[:, :, 1] = rng.normal(300.0, 3.0, size=(n, t))        # lst ~300 K
    x[:, :, 2] = rng.normal(4.0, 1.0, size=(n, t))          # eta ~4 mm/day
    observed = np.ones((n, t, F), dtype=bool)
    observed[0, 0] = False  # some absent entries
    dates = np.empty((n, t), dtype="<U10")
    for i in range(n):
        # sample i lives in year 2013+i → i <= 9 is <=2022 (train scope)
        year = 2013 + i
        dates[i] = f"{year}-06-15"
    farm_ids = ["FA" if i % 2 == 0 else "FB" for i in range(n)]
    if deviated:
        # physical NDVI drop of 0.4 on half the deviated set (post-half)
        x_dev = x.copy()
        x_dev[:, t // 2 :, 0] -= 0.4
        return x, x_dev, observed, dates, farm_ids
    return x, None, observed, dates, farm_ids


class TestLeakage:
    def test_2024_observation_in_scope_fails(self):
        x, _, observed, dates, farm_ids = make_data(n=11)
        # sample 10 would be 2023; make the violating observation 2024 (the
        # spec scenario) while 2023 is covered by the next test
        dates[10] = "2024-06-15"
        with pytest.raises(NormalizationLeakageError) as exc:
            fit_normalization(x, observed, dates, arm="N0",
                              feature_names=FEATURES, train_end="2023-01-01")
        assert "leakage" in str(exc.value)
        assert "2024" in str(exc.value)

    def test_2023_dev_year_also_excluded(self):
        x, _, observed, dates, _ = make_data(n=11)
        with pytest.raises(NormalizationLeakageError):
            fit_normalization(x, observed, dates, arm="N0",
                              feature_names=FEATURES, train_end="2023-01-01")

    def test_train_scope_only_passes(self):
        x, _, observed, dates, farm_ids = make_data(n=10)
        for arm in ARMS:
            stats = fit_normalization(x, observed, dates, arm=arm,
                                      feature_names=FEATURES,
                                      train_end="2023-01-01", farm_ids=farm_ids)
            assert stats.date_max < "2023-01-01"
            assert stats.n_obs > 0


class TestAnomalyExclusion:
    def test_excluded_observations_inform_nothing(self):
        x, _, observed, dates, farm_ids = make_data(n=10)
        ex = np.zeros((10, 8), dtype=bool)
        ex[2:5] = True
        # same scope for both fits: the excluded positions are identical, only
        # the values there differ (clean vs absurd poison)
        stats_clean = fit_normalization(x, observed, dates, arm="N0",
                                        feature_names=FEATURES, farm_ids=farm_ids,
                                        exclude_mask=ex)
        x_bad = x.copy()
        x_bad[2:5] = 1e6
        stats_bad = fit_normalization(x_bad, observed, dates, arm="N0",
                                      feature_names=FEATURES, farm_ids=farm_ids,
                                      exclude_mask=ex)
        # statistics identical despite poisoned values → exclusion works
        np.testing.assert_allclose(stats_clean.center, stats_bad.center)
        np.testing.assert_allclose(stats_clean.scale, stats_bad.scale)
        assert stats_bad.n_excluded_anomaly == 24

    def test_exclusion_mask_shape_checked(self):
        x, _, observed, dates, farm_ids = make_data(n=10)
        with pytest.raises(ValueError):
            fit_normalization(x, observed, dates, arm="N0",
                              feature_names=FEATURES, farm_ids=farm_ids,
                              exclude_mask=np.zeros((3, 3), dtype=bool))


class TestArms:
    @pytest.mark.parametrize("arm", ARMS)
    def test_all_arms_fit_apply_and_keep_masks(self, arm):
        x, _, observed, dates, farm_ids = make_data(n=10)
        stats = fit_normalization(x, observed, dates, arm=arm,
                                  feature_names=FEATURES, farm_ids=farm_ids)
        xn, rec = apply_normalization(stats, x, observed, farm_ids)
        assert xn.shape == x.shape
        # masked entries are 0 (with mask carried), never filled values
        assert (xn[~observed] == 0.0).all()
        # observed entries are normalized (physical N2 excluded). N0 zeroes
        # the mean; robust arms N1/N3/N4 zero the median, not the mean.
        if arm == "N0":
            assert abs(float(xn[:, :, 0][observed[:, :, 0]].mean())) < 1e-8
        if arm in ("N1", "N3"):
            assert abs(float(np.median(xn[:, :, 0][observed[:, :, 0]]))) < 1e-8

    def test_n2_keeps_physical_units_except_kelvin(self):
        x, _, observed, dates, farm_ids = make_data(n=10)
        stats = fit_normalization(x, observed, dates, arm="N2",
                                  feature_names=FEATURES, farm_ids=farm_ids)
        xn, _ = apply_normalization(stats, x, observed, farm_ids)
        # ndvi unchanged (identity); lst centred by 273.15 / 50
        np.testing.assert_allclose(xn[:, :, 0][observed[:, :, 0]],
                                   x[:, :, 0][observed[:, :, 0]])
        np.testing.assert_allclose(
            xn[:, :, 1][observed[:, :, 1]],
            (x[:, :, 1][observed[:, :, 1]] - 273.15) / 50.0,
        )
        assert stats.constants["kelvin_center"] == 273.15

    def test_n3_clips_and_logs_clip_rate(self):
        x, _, observed, dates, farm_ids = make_data(n=10)
        stats = fit_normalization(x, observed, dates, arm="N3",
                                  feature_names=FEATURES, farm_ids=farm_ids)
        xn, rec = apply_normalization(stats, x, observed, farm_ids)
        assert stats.clip_rate is not None  # train clip rate logged
        assert rec["clip_sigma"] == pytest.approx(3.0)
        assert float(np.abs(xn[observed]).max()) <= 3.0 + 1e-12

    def test_n4_requires_farm_ids_and_centres_per_field(self):
        x, _, observed, dates, farm_ids = make_data(n=10)
        with pytest.raises(ValueError):
            fit_normalization(x, observed, dates, arm="N4", feature_names=FEATURES)
        stats = fit_normalization(x, observed, dates, arm="N4",
                                  feature_names=FEATURES, farm_ids=farm_ids)
        assert set(stats.field_centers) == {"FA", "FB"}
        # FA samples centred by FA's median → normalized median ~0 per field
        xn, _ = apply_normalization(stats, x, observed, farm_ids)
        fa = [i for i, f in enumerate(farm_ids) if f == "FA"]
        assert abs(float(np.median(xn[fa][:, :, 0]))) < 1e-8
        # a NEW farm at apply time falls back to the global centre, recorded
        xn2, rec2 = apply_normalization(stats, x[:2], observed[:2], ["NEW", "NEW"])
        assert "fallback" in str(rec2.get("fallbacks", "")) or rec2.get("fallbacks")


class TestSidecar:
    def test_roundtrip_and_schema(self, tmp_path):
        x, _, observed, dates, farm_ids = make_data(n=10)
        stats = fit_normalization(x, observed, dates, arm="N1",
                                  feature_names=FEATURES, farm_ids=farm_ids,
                                  git_sha="testsha")
        p = save_norm_stats(tmp_path / "norm_stats.json", stats)
        loaded = load_norm_stats(p)
        assert isinstance(loaded, NormStats)
        np.testing.assert_allclose(loaded.center, stats.center)
        np.testing.assert_allclose(loaded.scale, stats.scale)
        assert loaded.arm == "N1" and loaded.git_sha == "testsha"
        assert loaded.data_sha256 == stats.data_sha256
        assert loaded.n_obs == stats.n_obs
        assert loaded.date_min and loaded.date_max and loaded.train_end

    def test_bad_schema_refused(self, tmp_path):
        import json

        p = tmp_path / "bad.json"
        p.write_text(json.dumps({"schema": "norm-stats/v0", "arm": "N0"}))
        with pytest.raises(ValueError):
            load_norm_stats(p)


class TestErasure:
    def test_physical_drop_survives_every_arm(self):
        x, x_dev, observed, dates, farm_ids = make_data(n=10, deviated=True)
        records = []
        for arm in ARMS:
            stats = fit_normalization(x, observed, dates, arm=arm,
                                      feature_names=FEATURES, farm_ids=farm_ids)
            rec = normalization_erasure_record(
                stats, x, x_dev, feature_idx=0, observed=observed, farm_ids=farm_ids
            )
            records.append(rec)
            assert rec["sign_preserved"] is True, (arm, rec)
            assert rec["retained_frac"] >= 0.5, (arm, rec)
        assert {r["arm"] for r in records} == set(ARMS)

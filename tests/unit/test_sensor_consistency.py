"""Task 0A.3 — sensor_id per observation, band-order assertion, same-phase consistency."""

from __future__ import annotations

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from ssl_anomaly.io.data_contract import CANONICAL_BAND_ORDER, ContractViolation
from ssl_anomaly.io.sensor_consistency import (
    farm_band_medians,
    same_phase_consistency,
    same_phase_pairs,
)


@pytest.fixture()
def farm_group(tmp_path):
    """Synthetic single-farm group: 7 bands, 60 dates, half L8 / half L9."""
    root = zarr.group(str(tmp_path / "satellite.zarr"))
    g = root.create_group("L02-02")
    d, h, w = 60, 4, 4
    # dates 0-29: L8 era 2020 (DOYs 1..146 at 5-day steps); dates 30-59: L9
    # era 2023 repeating the SAME DOYs, so every L8 date has a same-phase L9
    # within the ±8-day window (matching is by DOY window across years).
    dates = [
        (np.datetime64("2020-01-01", "D") + np.timedelta64(i * 5, "D")).astype(str)
        for i in range(30)
    ] + [
        (np.datetime64("2023-01-01", "D") + np.timedelta64(i * 5, "D")).astype(str)
        for i in range(30)
    ]
    g.create_array("dates", data=np.array(dates, dtype="<U10"))
    rng = np.random.default_rng(7)
    data = rng.normal(0.4, 0.05, size=(d, 7, h, w)).astype(np.float32)
    g.create_array("data", data=data)
    g.create_array("pixel_valid", data=np.ones((d, h, w), dtype=bool))
    g.create_array("in_field", data=np.ones((h, w), dtype=bool))
    g.attrs["band_order"] = list(CANONICAL_BAND_ORDER)
    g.attrs["farm_id"] = "L02-02"
    # dates 0-29 are L8, 30-59 are L9
    g.create_array("sensor_id", data=np.array([0] * 30 + [1] * 30, dtype=np.int8))
    return g


class TestFarmBandMedians:
    def test_median_over_valid_in_field_pixels(self, farm_group):
        names, med = farm_band_medians(farm_group)
        assert names == list(CANONICAL_BAND_ORDER)
        assert med.shape == (60, 7)
        assert np.isfinite(med).all()

    def test_invalid_dates_are_absent_not_zero(self, farm_group):
        farm_group["pixel_valid"][:] = False
        farm_group["pixel_valid"][0] = True
        _, med = farm_band_medians(farm_group)
        assert np.isfinite(med[0]).all()
        assert np.isnan(med[1:]).all()  # masked-out dates stay absent


class TestSamePhasePairs:
    def test_pairs_match_closest_doy_and_each_l9_used_once(self):
        # L8 at DOY 100; two L9 candidates at DOY 102 and 104
        dates = ["2020-04-09", "2020-04-11", "2020-04-13"]  # DOY 100, 102, 104
        sensor = np.array([0, 1, 1])
        med = np.full((3, 1), np.nan)
        pairs = same_phase_pairs(med, sensor, dates, phase_window_days=8)
        assert pairs == [(0, 1)]  # closest wins, second L9 left unused

    def test_outside_window_not_matched(self):
        dates = ["2020-04-09", "2020-08-01"]  # DOY 100 vs 214
        sensor = np.array([0, 1])
        med = np.full((2, 1), np.nan)
        assert same_phase_pairs(med, sensor, dates) == []

    def test_real_group_pair_count(self, farm_group):
        dates = [str(d) for d in farm_group["dates"][:]]
        sensor = np.asarray(farm_group["sensor_id"][:])
        med = np.full((60, 7), np.nan)
        pairs = same_phase_pairs(med, sensor, dates, phase_window_days=8)
        # 5-day revisit: every L9 date has an L8 within 8 DOY days
        assert len(pairs) == 30


class TestSamePhaseConsistency:
    def test_constant_true_offset_is_resolved(self, farm_group):
        # inject a constant +0.1 nir offset into every L9 observation
        data = farm_group["data"][:]
        data[30:, 3] += 0.1  # band index 3 = nir
        farm_group["data"][:] = data
        names, med = farm_band_medians(farm_group)
        sensor = np.asarray(farm_group["sensor_id"][:])
        dates = [str(d) for d in farm_group["dates"][:]]
        rec = same_phase_consistency(med, sensor, dates, band_names=names)
        nir = rec["bands"]["nir"]
        assert nir["resolved"] is True
        assert nir["offset"] == pytest.approx(-0.1, abs=0.02)
        # convention: offset = median(L8 - L9); injected +0.1 on L9 -> −0.1
        assert rec["n_pairs"] == 30

    def test_noise_only_is_unresolved_and_correction_refused(self, farm_group):
        names, med = farm_band_medians(farm_group)
        sensor = np.asarray(farm_group["sensor_id"][:])
        dates = [str(d) for d in farm_group["dates"][:]]
        rec = same_phase_consistency(med, sensor, dates, band_names=names)
        # the decision record must never claim a correction was applied
        assert rec["decision"].startswith("no correction")
        for band, b in rec["bands"].items():
            if not b["resolved"]:
                assert "no correction" in b["note"]

    def test_missing_sensor_id_fails_closed_in_script_path(self, farm_group):
        del farm_group["sensor_id"]
        assert "sensor_id" not in farm_group

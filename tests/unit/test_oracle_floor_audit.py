"""Oracle-floor-audit pure functions: Gaussian oracle, robust σ, probit MDE,
matched-NDVI band ratios. Nothing here touches the store or the checkpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import audit_oracle_floor as AOF  # noqa: E402
import audit_matched_ndvi as AMN  # noqa: E402


# --- Gaussian oracle ----------------------------------------------------------
def test_oracle_at_zero_delta_stays_at_or_below_far():
    # detection counts alarms from the ONSET only, while the threshold is
    # calibrated on the full-path null max — the same conservative structure
    # as the production protocol, so δ=0 power ≤ FAR (never above it)
    o = AOF.gaussian_oracle(0.0, n_sims=20_000)
    assert o["power"] <= o["far_target"] + 0.01


def test_oracle_power_is_monotone_in_delta_and_saturates():
    p1 = AOF.gaussian_oracle(0.5, n_sims=20_000)["power"]
    p2 = AOF.gaussian_oracle(1.5, n_sims=20_000)["power"]
    p3 = AOF.gaussian_oracle(4.0, n_sims=20_000)["power"]
    assert p1 < p2 < p3
    assert p3 > 0.95


def test_oracle_later_onsets_have_no_more_power():
    prof = [2.0] * 6 + [0.0] * 24
    o = AOF.gaussian_oracle(prof, n_sims=20_000)
    assert o["power_per_onset"][30] <= o["power_per_onset"][6] + 1e-9


# --- robust sigma ---------------------------------------------------------------
def test_robust_sigma_recovers_a_normal_scale():
    g = np.random.default_rng(0)
    x = g.normal(0.0, 2.0, size=20_000)
    assert abs(AOF.robust_sigma(x) - 2.0) < 0.1
    assert np.isnan(AOF.robust_sigma(np.array([np.nan, np.nan])))


def test_realignment_gap_one_matches_production_slice():
    anchor = np.arange(2 * 5, dtype=float).reshape(2, 5)
    pred1 = AOF.realign(anchor, 1)
    assert np.array_equal(pred1[:, 1:], anchor[:, :-1])
    assert np.isnan(pred1[:, 0]).all()
    pred3 = AOF.realign(anchor, 3)
    assert np.array_equal(pred3[:, 3:], anchor[:, :-3])
    assert np.isnan(pred3[:, :3]).all()


def test_field_blocked_split_is_deterministic_and_partitioning():
    fields = list(range(100))
    cal1, test1, h1 = AOF.field_blocked_split(fields, seed=13)
    cal2, test2, h2 = AOF.field_blocked_split(fields, seed=13)
    assert h1 == h2
    assert not (cal1 & test1)
    assert cal1 | test1 == set(fields)


# --- matched-NDVI band ratio -----------------------------------------------------
def test_band_magnitude_and_ndvi_equivalent():
    assert AMN.band_magnitude(3.0, 4.0) == 5.0
    assert AMN.ndvi_equivalent(0.40, 1.453) == pytest.approx(0.5812, abs=1e-3)


def test_per_channel_field_delta_ignores_nan_and_off_field_pixels():
    # (T=2, C=2, S=2): a never-acquired pixel (obs=False) must not poison the
    # mean — NaN·0 is NaN, so the mask is applied before the sum
    raw = np.zeros((2, 2, 2, 2))
    raw[0, 0, 0, 0] = np.nan
    inj = raw.copy()
    inj[:, 1, :, :] = 0.5                      # channel 1 bumped by 0.5
    obs = np.ones_like(raw, dtype=bool)
    obs[0, :, 0, 0] = False                    # the NaN pixel is unobserved
    inf = np.ones((2, 2), dtype=bool)
    d = AMN.per_channel_field_delta(inj, raw, obs, inf)
    assert d.shape == (2,)
    assert d[1] == pytest.approx(0.5)
    assert d[0] == pytest.approx(0.0)


def test_summarise_rho_is_band_ratio():
    rows = []
    for _ in range(4):
        d_ndvi = np.zeros(8)
        d_savi = np.zeros(8)
        d_ndvi[3], d_ndvi[4] = 0.0107, -0.0107
        d_savi[3], d_savi[4] = 0.0155, -0.0155
        rows.append({"ndvi": d_ndvi, "savi": d_savi})
    s = AMN.summarise(rows)
    assert s["rho_savi_over_ndvi"] == pytest.approx(1.45, abs=0.01)


# --- MDE curve -------------------------------------------------------------------
def test_probit_roundtrip():
    import build_mde_curve as M
    from scipy.special import ndtr
    assert M.probit(float(ndtr(1.234))) == pytest.approx(1.234, abs=1e-6)
    assert M.probit(float(ndtr(-2.5))) == pytest.approx(-2.5, abs=1e-6)


def test_fit_and_delta_at_power_on_synthetic_ladder():
    import build_mde_curve as M
    deltas = np.array([0.02, 0.05, 0.08, 0.16, 0.28, 0.40])
    floor = 0.05
    true_b, true_a = 6.0, -2.0
    p = floor + (1 - floor) * M._cdf(true_a + true_b * deltas)
    rng = np.random.default_rng(0)
    n = 4000
    rates = rng.binomial(n, p) / n
    a, b = M.fit_floor_probit(deltas, rates, floor, n)
    assert np.isfinite(a) and b > 0
    d80 = M.delta_at_power(a, b, floor, 0.8)
    z = M.probit((0.8 - floor) / (1 - floor))
    truth = (z - true_a) / true_b
    # recovery within the sampling error of the design truth
    assert abs(d80 - truth) < 0.05
    lo, hi = M.bootstrap_ci(deltas, rates, n, floor, a, b, draws=200)
    assert lo <= d80 + 1e-9 <= hi or (np.isnan(lo) and np.isnan(hi))


def test_flat_family_yields_censored_nan():
    import build_mde_curve as M
    deltas = np.array([0.02, 0.05, 0.08, 0.16, 0.28, 0.40])
    rates = np.array([0.012, 0.012, 0.012, 0.015, 0.018, 0.059])
    a, b = M.fit_floor_probit(deltas, rates, floor=0.012, n=340)
    d80 = M.delta_at_power(a, b, floor=0.012, target=0.8)
    # either no usable curve (NaN) or a Δ* beyond the largest measured step —
    # both are the honest censored outcome
    assert (not np.isfinite(d80)) or d80 > 0.40

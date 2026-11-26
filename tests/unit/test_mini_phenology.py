"""Unit tests for the NDVI+SAVI screening-layer phenology detector."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ssl_anomaly.phenology.mini import (
    HarvestCandidate,
    build_cycle_rows,
    detect_harvests,
    growth_state,
    vi_harvest_score,
)


def _two_cycle_series(n: int = 180) -> dict[str, np.ndarray]:
    t = np.arange(n)
    phase = (t % 60) / 60.0
    return {
        "ndvi": 0.25 + 0.55 * np.sin(np.pi * phase),
        "savi": 0.30 + 0.50 * np.sin(np.pi * phase),
    }


def test_vi_score_orientations_and_no_weather_terms():
    s = _two_cycle_series(60)
    score, oriented = vi_harvest_score(s["ndvi"], s["savi"])
    assert score.shape == (60,)
    assert np.isfinite(score).all()
    assert score[55] > 0  # late-cycle low NDVI/SAVI = harvest-like
    assert oriented["ndvi"][55] > 0
    assert oriented["savi"][55] > 0
    assert set(oriented) == {"ndvi", "savi"}


def test_vi_score_nan_in_either_index_fails_closed():
    s = _two_cycle_series(60)
    s["savi"][10] = np.nan
    score, _ = vi_harvest_score(s["ndvi"], s["savi"])
    assert np.isnan(score[10])
    cands, weak = detect_harvests(score, np.arange(60), {"ndvi": np.zeros(60), "savi": np.zeros(60)})
    assert all(c.index != 10 for c in cands + weak)


def test_detect_harvests_separates_weak_candidates():
    score = np.zeros(120, dtype=np.float64)
    score[30] = 0.6  # strong boundary
    score[90] = 0.13  # weak candidate (below default prominence 0.3)
    oriented = {"ndvi": np.zeros(120), "savi": np.zeros(120)}
    cands, weak = detect_harvests(score, np.arange(120), oriented, min_gap_days=10, prominence=0.3)
    assert any(c.index == 30 for c in cands)
    assert not any(c.index == 90 for c in cands)
    assert any(c.index == 90 for c in weak)


def test_growth_state_boundaries():
    assert growth_state(0.10) == "S1_establishment_growth"
    assert growth_state(0.25) == "S2_peak_growth"
    assert growth_state(0.60) == "S3_maturity_senescence"
    assert growth_state(0.85) == "S4_harvest_dormancy"
    assert growth_state(np.nan) == "absent"


def test_cycle_rows_retain_overlong_without_kind_or_gdd():
    dates = [str(d)[:10] for d in pd.date_range("2013-01-01", periods=300, freq="16D")]
    candidates = [
        HarvestCandidate(index=10, score=1.0, confidence=0.8, support=2, channel_z={}),
        HarvestCandidate(index=60, score=1.0, confidence=0.8, support=2, channel_z={}),
    ]
    rows = build_cycle_rows("L02-02", 245, dates, candidates)
    assert rows
    overlong = [r for r in rows if r["status"] == "overlong_unresolved"]
    assert overlong, "the >540 d interval must be retained as evidence"
    assert overlong[0]["length_days"] > 540
    assert all("kind" not in r for r in rows)
    assert all("gdd_sum_within_cycle" not in r for r in rows)
    assert all(r["growth_state_at_midpoint"] in {
        "S1_establishment_growth", "S2_peak_growth",
        "S3_maturity_senescence", "S4_harvest_dormancy", "absent",
    } for r in rows)


def test_cycle_rows_retain_short_interval_without_a_claim():
    dates = [str(d)[:10] for d in pd.date_range("2013-01-01", periods=120, freq="16D")]
    candidates = [
        HarvestCandidate(index=40, score=1.0, confidence=0.8, support=2, channel_z={}),
        HarvestCandidate(index=43, score=1.0, confidence=0.8, support=2, channel_z={}),
    ]
    rows = build_cycle_rows("R15-18", 245, dates, candidates)
    short = [r for r in rows if r["status"] == "short_unresolved"]
    assert short
    assert short[0]["length_days"] < 180


def test_greenup_start_excludes_rest_and_can_resolve_a_long_interval():
    dates = [str(d)[:10] for d in pd.date_range("2013-01-01", periods=120, freq="16D")]
    ndvi = np.full(120, 0.15)
    savi = np.full(120, 0.10)
    ndvi[35:], savi[35:] = 0.70, 0.50  # greenup after a rest period
    candidates = [
        HarvestCandidate(index=10, score=1.0, confidence=0.8, support=2, channel_z={}),
        HarvestCandidate(index=50, score=1.0, confidence=0.8, support=2, channel_z={}),
    ]
    rows = build_cycle_rows("L02-02", 245, dates, candidates, ndvi=ndvi, savi=savi)
    seg = next(r for r in rows if r["boundary_start_index"] == 10)
    assert seg["cycle_start_source"] == "greenup"
    assert seg["rest_days"] > 0
    assert seg["length_days"] == 15 * 16
    assert seg["status"] == "normal"

    flat = build_cycle_rows(
        "L02-02", 245, dates, candidates,
        ndvi=np.full(120, 0.15), savi=np.full(120, 0.10),
    )
    flat_seg = next(r for r in flat if r["boundary_start_index"] == 10)
    assert flat_seg["cycle_start_source"] == "harvest"
    assert flat_seg["rest_days"] == 0


def test_no_rest_when_vi_rises_immediately_after_harvest():
    dates = [str(d)[:10] for d in pd.date_range("2013-01-01", periods=120, freq="16D")]
    ndvi = np.full(120, 0.15)
    savi = np.full(120, 0.10)
    ndvi[11:] = 0.60  # greenness resumes right after harvest: no bare-soil rest
    savi[11:] = 0.45
    candidates = [
        HarvestCandidate(index=10, score=1.0, confidence=0.8, support=2, channel_z={}),
        HarvestCandidate(index=50, score=1.0, confidence=0.8, support=2, channel_z={}),
    ]
    rows = build_cycle_rows("F", 1, dates, candidates, ndvi=ndvi, savi=savi)
    seg = next(r for r in rows if r["boundary_start_index"] == 10)
    assert seg["cycle_start_source"] == "harvest"
    assert seg["rest_days"] == 0


def test_cycle_rows_normal_interval_has_normal_status():
    dates = [str(d)[:10] for d in pd.date_range("2013-01-01", periods=120, freq="16D")]
    candidates = [
        HarvestCandidate(index=40, score=1.0, confidence=0.8, support=2, channel_z={}),
    ]
    rows = build_cycle_rows("R12-08", 245, dates, candidates)
    assert rows
    assert all(r["status"] in {"normal", "overlong_unresolved"} for r in rows)
    assert any(r["status"] == "normal" for r in rows)

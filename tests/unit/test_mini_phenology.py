"""Unit tests for the screening-layer multi-band phenology detector."""

from __future__ import annotations

import numpy as np

from ssl_anomaly.phenology.mini import (
    build_cycle_rows,
    detect_harvests,
    multi_band_harvest_score,
    phase_for_date,
)


def _two_cycle_series(n: int = 120) -> dict[str, np.ndarray]:
    t = np.arange(n)
    # Two 60-step cycles: vegetation falls and LST/GDD jump at each boundary.
    phase = (t % 60) / 60.0
    ndvi = 0.25 + 0.55 * np.sin(np.pi * phase)
    savi = 0.30 + 0.50 * np.sin(np.pi * phase)
    lst = 300.0 - 8.0 * np.sin(np.pi * phase)
    gdd = 5.0 + 10.0 * np.sin(np.pi * phase)
    return {"ndvi": ndvi, "savi": savi, "lst": lst, "gdd": gdd}


def test_multi_band_score_orientations():
    s = _two_cycle_series(60)
    score, oriented = multi_band_harvest_score(
        s["ndvi"], s["savi"], s["lst"], s["gdd"]
    )
    assert score.shape == (60,)
    assert np.isfinite(score).all()
    # At a late-cycle (low NDVI, high LST) date the combined score is positive.
    assert score[55] > 0
    assert oriented["ndvi"][55] > 0  # -z(NDVI)
    assert oriented["lst"][55] > 0  # +z(LST)


def test_nan_channel_produces_nan_score():
    s = _two_cycle_series(60)
    s["gdd"][10] = np.nan
    score, _ = multi_band_harvest_score(s["ndvi"], s["savi"], s["lst"], s["gdd"])
    assert np.isnan(score[10])
    cands = detect_harvests(score, np.arange(60), {}, min_gap_days=10, prominence=0.2)
    assert all(c.index != 10 for c in cands)


def test_detect_harvests_finds_both_boundaries_with_confidence():
    s = _two_cycle_series(180)
    score, oriented = multi_band_harvest_score(
        s["ndvi"], s["savi"], s["lst"], s["gdd"]
    )
    cands = detect_harvests(score, np.arange(180), oriented, min_gap_days=30, prominence=0.2)
    assert len(cands) >= 2
    assert any(abs(c.index - 60) <= 3 for c in cands)
    assert any(abs(c.index - 120) <= 3 for c in cands)
    for c in cands:
        assert 0.0 <= c.confidence <= 1.0
        assert 0 <= c.support <= 4
        assert set(c.channel_z) == {"ndvi", "savi", "lst", "gdd"}


def test_phase_mapping_and_cycle_rows():
    assert phase_for_date(10, 366) == "germination"
    assert phase_for_date(90, 366) == "tillering"
    assert phase_for_date(200, 366) == "grand_growth"
    assert phase_for_date(330, 366) == "maturation"
    assert phase_for_date(np.nan, 366) == "absent"
    dates = [
        str(d)[:10]
        for d in __import__("pandas").date_range("2013-01-01", periods=240, freq="4D")
    ]
    s = _two_cycle_series(240)
    score, oriented = multi_band_harvest_score(
        s["ndvi"], s["savi"], s["lst"], s["gdd"]
    )
    cands = detect_harvests(score, np.arange(240), oriented, min_gap_days=30, prominence=0.2)
    rows = build_cycle_rows("R12-08", 245, dates, cands, s["gdd"])
    assert rows
    assert all(r["length_days"] >= 180 for r in rows)
    assert all(r["farm"] == "R12-08" and r["ogc_fid"] == 245 for r in rows)

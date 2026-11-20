"""Fused multi-index detector tests (change cycle-detection-v3-multindex).

Covers the spec scenarios: v1-subset equivalence, corroboration-only
candidates, dedup/boundary protection, determinism, lost-span geometry, the
no-v1-harvest path, and the vf_min evidence gate. All synthetic; no data
artifacts are touched.
"""
from __future__ import annotations

import numpy as np

from ssl_anomaly.phenology.cycles import (
    FusionParams,
    fusion_params_hash,
    lost_spans_from_harvests,
    segment_cycles,
    segment_cycles_fused,
)

V1_KW = dict(min_cycle_days=250, max_cycle_days=450,
             min_prominence=0.1, plateau_frac=0.6, rise_frac=0.1)


def _trapezoid(length=350, ndvi_max=0.85, ndvi_min=0.2, rng=None, noise=0.015,
               decline_days=90):
    """Sugarcane-like season: linear green-up, plateau, decline, floor."""
    y = np.full(length, ndvi_min)
    p1 = max(1, length // 4)
    p2 = max(p1 + 1, length - decline_days)
    y[:p1] = np.linspace(ndvi_min, ndvi_max, p1)
    y[p1:p2] = ndvi_max
    y[p2:] = np.linspace(ndvi_max, ndvi_min, length - p2)
    if rng is not None and noise:
        y = y + rng.normal(0, noise, size=length)
    return np.clip(y, 0.0, 1.0)


def _field_two_years(rng):
    """Year 1 peaks at 0.85 (global ymax); year 2 only reaches 0.45 and ends
    in a sharp 14-day cliff to a 26-day bare floor at ~day 705 — v1's
    plateau test (0.6 x 0.85) rejects the year-2 trough, but the veg drop
    (~0.25 at the floor) exceeds ``veg_drop``."""
    year1 = _trapezoid(360, ndvi_max=0.85, rng=rng)
    year2 = np.concatenate([
        np.linspace(0.2, 0.45, 90),    # green-up 360-450
        np.full(240, 0.45),            # plateau 450-690
        np.linspace(0.45, 0.2, 14),    # cliff 690-704
        np.full(26, 0.2),              # bare floor 704-730
    ])
    tail = np.full(100, 0.5)
    return np.concatenate([year1, year2, tail]), 705


def _sample(daily_ndvi, rng, cadence=7, drop=0.1):
    t = np.arange(0, daily_ndvi.size, cadence)
    ndvi = daily_ndvi[t] + rng.normal(0, 0.01, size=t.size)
    keep = rng.random(t.size) > drop
    return t[keep], np.clip(ndvi[keep], 0.0, 1.0)


def _pulse(t, day, amp=6.0, width=20.0, base=290.0):
    lst = np.full(t.size, base)
    return lst + amp * np.exp(-0.5 * ((t - day) / width) ** 2)


def test_v1_subset_equivalence_without_corroboration(rng):
    daily, _ = _field_two_years(rng)
    t, ndvi = _sample(daily, rng)
    lst = _pulse(t, -1000)  # flat baseline far from the series
    v1 = segment_cycles(1, t, ndvi, quality=np.ones_like(ndvi), **V1_KW)
    fused = segment_cycles_fused(1, t, ndvi, ndvi.copy(), lst, np.zeros_like(t),
                                 quality=np.ones_like(ndvi), **V1_KW)
    assert [c.to_dict() for c in fused] == [c.to_dict() for c in v1]


def test_veg_drop_without_corroboration_rejected(rng):
    """The year-2 trough is a big veg drop v1 misses (global-ymax plateau);
    with flat LST/SWIR the fused detector must stay identical to v1."""
    daily, _ = _field_two_years(rng)
    t, ndvi = _sample(daily, rng)
    v1 = segment_cycles(2, t, ndvi, quality=np.ones_like(ndvi), **V1_KW)
    fused = segment_cycles_fused(2, t, ndvi, ndvi.copy(), _pulse(t, -1000),
                                 np.zeros_like(t),
                                 quality=np.ones_like(ndvi), **V1_KW)
    assert [c.to_dict() for c in fused] == [c.to_dict() for c in v1]


def test_corroborated_lst_rise_recovers_missed_harvest(rng):
    daily, trough_day = _field_two_years(rng)
    t, ndvi = _sample(daily, rng)
    lst = _pulse(t, trough_day)
    v1 = segment_cycles(3, t, ndvi, quality=np.ones_like(ndvi), **V1_KW)
    fused = segment_cycles_fused(3, t, ndvi, ndvi.copy(), lst, np.zeros_like(t),
                                 quality=np.ones_like(ndvi), **V1_KW)
    v1_clean = [c for c in v1 if not c.censored]
    fused_clean = [c for c in fused if not c.censored]
    assert len(fused_clean) > len(v1_clean)
    new_ends = ({c.end_harvest for c in fused_clean}
                - {c.end_harvest for c in v1_clean})
    assert new_ends, "expected a candidate-backed harvest boundary"
    day = new_ends.pop()
    assert abs(day - trough_day) <= 40
    for c in fused_clean:
        assert 250 <= c.length_days <= 450


def test_dedup_rejects_candidate_near_known_boundary(rng):
    """A corroborated bare-soil event 100 days after a known boundary must
    not create a second boundary (dedup_days = 180). Both arms run on the
    SAME perturbed series so any smoothing difference cancels out."""
    daily, _ = _field_two_years(rng)
    t, ndvi = _sample(daily, rng)
    v1 = segment_cycles(4, t, ndvi, quality=np.ones_like(ndvi), **V1_KW)
    first_harvest = min(c.end_harvest for c in v1 if c.start_harvest >= 0)
    event_day = int(first_harvest) + 100
    ndvi_dip = np.clip(np.where(np.abs(t - event_day) <= 12, ndvi - 0.3, ndvi), 0, 1)
    lst = _pulse(t, event_day, width=15.0)
    v1_dip = segment_cycles(4, t, ndvi_dip, quality=np.ones_like(ndvi), **V1_KW)
    fused = segment_cycles_fused(4, t, ndvi_dip, ndvi_dip.copy(), lst,
                                 np.zeros_like(t),
                                 quality=np.ones_like(ndvi), **V1_KW)
    assert [c.to_dict() for c in fused] == [c.to_dict() for c in v1_dip]


def test_determinism(rng):
    daily, trough_day = _field_two_years(rng)
    t, ndvi = _sample(daily, rng)
    lst = _pulse(t, trough_day)
    kw = dict(quality=np.ones_like(ndvi), **V1_KW)
    a = segment_cycles_fused(5, t, ndvi, ndvi.copy(), lst, np.zeros_like(t), **kw)
    b = segment_cycles_fused(5, t, ndvi, ndvi.copy(), lst, np.zeros_like(t), **kw)
    assert [c.to_dict() for c in a] == [c.to_dict() for c in b]


def test_no_v1_harvest_field_gains_first_boundary(rng):
    """A monotone step series has no v1-detectable trough (no local optimum),
    yet the cliff carries a corroborated candidate — the fused detector can
    give such a field its first boundaries."""
    n1, n2 = 420, 420
    daily = np.concatenate([np.full(n1, 0.6), np.linspace(0.6, 0.2, 20),
                            np.full(n2, 0.2)])
    t, ndvi = _sample(daily, rng)
    v1 = segment_cycles(6, t, ndvi, quality=np.ones_like(ndvi), **V1_KW)
    assert v1 == []
    cliff_day = n1 + 10
    lst = _pulse(t, cliff_day)
    fused = segment_cycles_fused(6, t, ndvi, ndvi.copy(), lst, np.zeros_like(t),
                                 quality=np.ones_like(ndvi), **V1_KW)
    assert len(fused) >= 2
    assert fused[0].censored and fused[-1].censored
    boundaries = {c.end_harvest for c in fused if c.start_harvest == -1}
    boundaries |= {c.start_harvest for c in fused if c.start_harvest >= 0}
    assert any(abs(b - cliff_day) <= 40 for b in boundaries)


def test_low_valid_frac_observations_ignored(rng):
    """Observations below vf_min must not contribute candidate evidence."""
    daily, trough_day = _field_two_years(rng)
    t, ndvi = _sample(daily, rng)
    lst = _pulse(t, trough_day)
    # Everything from before the cliff onward is flagged unusable
    q = np.where(t >= trough_day - 30, 0.1, 1.0)
    v1 = segment_cycles(7, t, ndvi, quality=q, **V1_KW)
    fused = segment_cycles_fused(7, t, ndvi, ndvi.copy(), lst, np.zeros_like(t),
                                 quality=q, **V1_KW)
    assert [c.to_dict() for c in fused] == [c.to_dict() for c in v1]


def test_lost_spans_geometry():
    spans = lost_spans_from_harvests([100, 800, 1200], 0, 1400,
                                     min_cycle_days=180, max_cycle_days=540)
    assert spans == [
        {"kind": "gap", "t0": 100, "t1": 800},
        {"kind": "trail", "t0": 1200, "t1": 1400},
    ]
    assert lost_spans_from_harvests([], 0, 1000, min_cycle_days=180,
                                    max_cycle_days=540) == []
    spans = lost_spans_from_harvests([400], 0, 1400, min_cycle_days=180,
                                     max_cycle_days=540)
    assert spans == [
        {"kind": "lead", "t0": 0, "t1": 400},
        {"kind": "trail", "t0": 400, "t1": 1400},
    ]


def test_fusion_params_hash_stable():
    a = fusion_params_hash(FusionParams())
    b = fusion_params_hash(FusionParams())
    assert a == b and len(a) == 16
    assert fusion_params_hash(FusionParams(veg_drop=0.25)) != a


def test_exclusion_transfer_by_window_overlap():
    """finding-14 verdicts attach to v1 WINDOW identities; a superseding
    table renumbers cycles, so exclusions transfer by interval overlap."""
    import pandas as pd

    from ssl_anomaly.monitor.splits import map_exclusions_by_window

    v1 = pd.DataFrame({
        "field": [10, 10, 20],
        "cycle_id": [1, 2, 1],
        "t_start": [100, 460, 100],
        "t_end": [460, 820, 500],
    })
    v2 = pd.DataFrame({
        "field": [10, 10, 10, 20, 30],
        "cycle_id": [0, 1, 2, 0, 0],
        "t_start": [105, 462, 830, 100, 100],   # overlap, overlap, no, overlap, other field
        "t_end": [465, 830, 1190, 500, 400],
    })
    keys, stats = map_exclusions_by_window(
        ["10:1", "20:1", "99:9"], v1, v2)  # 99:9 does not resolve in v1
    assert keys == {"10:0", "20:0"}
    assert stats == {"mode": "overlap_transfer_from_v1", "v1_excluded_keys": 3,
                     "v1_keys_unresolved": 1, "v2_windows_excluded": 2}

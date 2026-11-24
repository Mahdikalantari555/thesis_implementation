"""Transparent multi-band phenology for the three-farm mini dataset.

NDVI-only harvest detection is explicitly rejected: the v2 archive showed it
missed winter harvests and merged two seasons. Here four field-level channels
(NDVI, SAVI, LST, daily GDD) are oriented so that a harvest is a *drop* in
NDVI/SAVI/GDD and a *rise* in LST, standardized, and combined into one score.
Candidate harvests are peaks of that combined score with a minimum separation
in real days. Stage confidence is the fraction of channels that independently
support each candidate, so a weak candidate is visible rather than silently
promoted to a harvest.

This is a screening-layer detector, not an owner-approved production detector.
It must not be used to claim stress or phenology-phase anomaly detection beyond
validation-ladder rung L1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.signal import find_peaks

DEFAULT_WEIGHTS = {"ndvi": 1.0, "savi": 1.0, "lst": 1.0, "gdd": 0.5}


@dataclass(frozen=True)
class HarvestCandidate:
    index: int
    score: float
    confidence: float
    support: int
    channel_z: dict[str, float]


def _zscore(y: np.ndarray) -> np.ndarray:
    v = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(v)
    out = np.full_like(v, np.nan)
    if finite.sum() < 3:
        return out
    mu, sd = float(np.mean(v[finite])), float(np.std(v[finite]))
    if sd <= 0:
        return out
    out[finite] = (v[finite] - mu) / sd
    return out


def multi_band_harvest_score(
    ndvi: np.ndarray,
    savi: np.ndarray,
    lst: np.ndarray,
    gdd: np.ndarray,
    *,
    weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (combined score, per-channel oriented z-scores).

    Oriented z-scores: -NDVI, -SAVI, +LST, -GDD. NaNs stay NaN; a date with any
    missing channel has a NaN score and cannot become a harvest candidate.
    """
    w = dict(DEFAULT_WEIGHTS if weights is None else weights)
    oriented = {
        "ndvi": -_zscore(ndvi),
        "savi": -_zscore(savi),
        "lst": _zscore(lst),
        "gdd": -_zscore(gdd),
    }
    # Fail closed: a date missing any channel cannot be scored. Renormalizing
    # over the surviving channels would let a cloud-contaminated date look
    # like a confident harvest candidate.
    have_all = np.ones(len(ndvi), dtype=bool)
    for z in oriented.values():
        have_all &= np.isfinite(z)
    total = np.zeros(len(ndvi), dtype=np.float64)
    for name, z in oriented.items():
        total += np.where(have_all, z * float(w[name]), 0.0)
    score = np.full(len(ndvi), np.nan, dtype=np.float64)
    score[have_all] = total[have_all] / sum(float(v) for v in w.values())
    return score, oriented


def detect_harvests(
    score: np.ndarray,
    day_numbers: np.ndarray,
    oriented: dict[str, np.ndarray],
    *,
    min_gap_days: int = 120,
    prominence: float = 0.3,
    min_support_z: float = 0.4,
) -> list[HarvestCandidate]:
    """Detect harvest peaks on the combined score with real-day separation."""
    s = np.asarray(score, dtype=np.float64)
    valid = np.isfinite(s)
    idx = np.flatnonzero(valid)
    if idx.size < 3:
        return []
    fill = np.interp(np.arange(len(s)), idx, s[valid])
    distance = max(1, int(min_gap_days / max(1.0, float(np.median(np.diff(day_numbers))))))
    peaks, props = find_peaks(fill, prominence=prominence, distance=distance)
    out = []
    for i, p in enumerate(peaks):
        zs = {
            name: float(np.asarray(z)[p]) if np.isfinite(np.asarray(z)[p]) else 0.0
            for name, z in oriented.items()
        }
        support = sum(1 for v in zs.values() if v >= min_support_z)
        conf = float(
            min(1.0, support / len(zs)) * min(1.0, float(props["prominences"][i]))
        )
        out.append(
            HarvestCandidate(
                index=int(p),
                score=float(fill[p]),
                confidence=round(conf, 4),
                support=support,
                channel_z={k: round(v, 4) for k, v in zs.items()},
            )
        )
    return out


PHASE_ORDER = ("germination", "tillering", "grand_growth", "maturation")


def phase_for_date(dsh_days: float, cycle_length_days: float) -> str:
    """Map days-since-harvest to a classical sugarcane phase.

    Fractions follow the standard Khuzestan plant-cane calendar
    (germination 30-45 d, tillering 60-90 d, grand growth 120-180 d,
    maturation 60-90 d) scaled to the observed cycle length.
    """
    if not np.isfinite(dsh_days) or not np.isfinite(cycle_length_days) or cycle_length_days <= 0:
        return "absent"
    frac = dsh_days / cycle_length_days
    if frac < 45 / 366:
        return "germination"
    if frac < 135 / 366:
        return "tillering"
    if frac < 300 / 366:
        return "grand_growth"
    if frac < 1.0:
        return "maturation"
    return "post_harvest"


def build_cycle_rows(
    farm_id: str,
    ogc_fid: int,
    dates: list[str],
    candidates: list[HarvestCandidate],
    gdd_daily: np.ndarray,
    *,
    min_cycle_days: int = 180,
    max_cycle_days: int = 540,
) -> list[dict[str, Any]]:
    """Assemble cycle records from consecutive harvest candidates."""
    days = np.array([np.datetime64(d, "D") for d in dates])
    day_int = (days - np.datetime64("2013-03-30", "D")).astype(int)
    rows: list[dict[str, Any]] = []
    boundaries = [int(c.index) for c in candidates]
    starts = [-1] + boundaries
    ends = boundaries + [len(dates)]
    for cid, (s, e) in enumerate(zip(starts, ends), start=1):
        start = 0 if s < 0 else s
        length = int(day_int[min(e, len(dates) - 1)] - day_int[start])
        if length < min_cycle_days:
            continue
        censored = bool(s < 0 or e >= len(dates))
        if length > max_cycle_days and not censored:
            continue
        conf_values = [
            c.confidence for c in candidates if start <= c.index < min(e, len(dates))
        ]
        rows.append(
            {
                "farm": farm_id,
                "ogc_fid": ogc_fid,
                "cycle_id": cid,
                "start_date": dates[start],
                "end_date": dates[min(e, len(dates) - 1)],
                "start_harvest": bool(s >= 0),
                "end_harvest": bool(e < len(dates)),
                "length_days": length,
                "censored": censored,
                "kind": "plant" if cid == 1 else "ratoon",
                "stage_confidence": float(np.mean(conf_values)) if conf_values else np.nan,
                "n_harvest_candidates": len(conf_values),
                "gdd_sum_within_cycle": float(np.nansum(gdd_daily[start:e])),
                "phases_at_midpoint": phase_for_date(length / 2.0, float(length)),
            }
        )
    return rows

"""NDVI+SAVI-only sugarcane phenology for the three-farm mini dataset.

Harvest candidates are peaks of the mean oriented z-score of NDVI and SAVI
(−z NDVI, −z SAVI). LST, GDD and weather are deliberately absent from the
score: a single station series is shared by all fields, and the thesis keeps
thermal time for a separate axis ablation (findings 13/15). A date missing
either index is unscored rather than renormalized over the survivors.

Growth states are derived from normalized within-cycle progress (finding 17):
S1 establishment/growth, S2 peak growth, S3 maturity/senescence,
S4 harvest/dormancy. No plant/ratoon label is inferred.

This is a screening-layer detector, not owner-approved production phenology.
It must not be used for stress or phase-anomaly claims beyond L1 until the
owner has re-reviewed the regenerated timelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

WEAK_PROMINENCE = 0.1
PHASE_ORDER = (
    "S1_establishment_growth",
    "S2_peak_growth",
    "S3_maturity_senescence",
    "S4_harvest_dormancy",
)
#: Retention flags approved by the owner (design decisions 4 and 7): over-long
#: and short bounded intervals are kept as evidence, never silently deleted.
UNRESOLVED_STATUSES = ("overlong_unresolved", "short_unresolved")


def split_normal_cycles(
    cycles: pd.DataFrame,
    *,
    source: str = "cycle table",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Split a mini cycle table into (``normal`` rows, unresolved evidence).

    Fail closed: a table without the ``status`` column predates the retention
    flags and cannot prove its over-long/short intervals were not silently
    deleted, so it must not feed training or stage assignment. Unknown status
    values also fail. Unresolved rows are **retained in the summary** — this
    function never deletes evidence; the caller must surface the summary
    (log line / run manifest) rather than drop it.
    """
    if "status" not in cycles.columns:
        raise ValueError(
            f"{source}: cycle table has no 'status' column (pre-retention-flag "
            "schema); rebuild the phenology output before consuming it"
        )
    unknown = sorted(
        set(cycles["status"].dropna().astype(str)) - {"normal", *UNRESOLVED_STATUSES}
    )
    if unknown:
        raise ValueError(f"{source}: unknown cycle status values {unknown}")
    normal = cycles[cycles["status"] == "normal"].copy()
    unresolved = cycles[cycles["status"] != "normal"]
    by_status = {s: int((unresolved["status"] == s).sum()) for s in UNRESOLVED_STATUSES}
    rows: list[dict[str, Any]] = []
    for _, r in unresolved.iterrows():
        rows.append(
            {
                "farm": str(r["farm"]) if "farm" in unresolved.columns else None,
                "cycle_id": int(r["cycle_id"]) if "cycle_id" in unresolved.columns else None,
                "start_date": str(r["start_date"]) if "start_date" in unresolved.columns else None,
                "end_date": str(r["end_date"]) if "end_date" in unresolved.columns else None,
                "length_days": (
                    int(r["length_days"]) if "length_days" in unresolved.columns else None
                ),
                "status": str(r["status"]),
            }
        )
    summary: dict[str, Any] = {
        "source": source,
        "n_total": int(len(cycles)),
        "n_normal": int(len(normal)),
        "n_unresolved": int(len(unresolved)),
        "unresolved_by_status": by_status,
        "unresolved_rows": rows,
    }
    return normal, summary


@dataclass(frozen=True)
class HarvestCandidate:
    index: int
    score: float
    confidence: float
    support: int
    channel_z: dict[str, float]
    prominence: float = 0.0


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


def vi_harvest_score(
    ndvi: np.ndarray,
    savi: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Return (combined score, oriented z-scores) from NDVI and SAVI only.

    Oriented z: -z(NDVI), -z(SAVI), so a harvest-like greenness drop is high.
    NaNs stay NaN; a date with either index missing cannot be scored.
    """
    a, b = np.asarray(ndvi, dtype=np.float64), np.asarray(savi, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("NDVI and SAVI series must have equal length")
    oriented = {"ndvi": -_zscore(a), "savi": -_zscore(b)}
    have_all = np.isfinite(oriented["ndvi"]) & np.isfinite(oriented["savi"])
    score = np.full(len(a), np.nan, dtype=np.float64)
    score[have_all] = 0.5 * (oriented["ndvi"][have_all] + oriented["savi"][have_all])
    return score, oriented


def detect_harvests(
    score: np.ndarray,
    day_numbers: np.ndarray,
    oriented: dict[str, np.ndarray],
    *,
    min_gap_days: int = 120,
    prominence: float = 0.3,
    min_support_z: float = 0.4,
) -> tuple[list[HarvestCandidate], list[HarvestCandidate]]:
    """Return (cycle boundaries, weak candidates) from the VI score.

    Peaks at or above ``prominence`` are boundaries; peaks between
    ``WEAK_PROMINENCE`` and ``prominence`` are retained as reviewable evidence
    and never split a cycle on their own.
    """
    s = np.asarray(score, dtype=np.float64)
    days = np.asarray(day_numbers, dtype=np.float64)
    valid = np.isfinite(s)
    idx = np.flatnonzero(valid)
    if idx.size < 3:
        return [], []
    fill = np.interp(np.arange(len(s)), idx, s[valid])
    # Distance in *indices* is wrong for irregular Landsat sampling: a gap in
    # the series makes the median spacing small and suppresses peaks that are
    # far apart in real days. Find all peaks, then enforce the separation rule
    # on real days, keeping the higher-scoring peak first.
    peaks, props = find_peaks(fill, prominence=WEAK_PROMINENCE)

    def _candidate(p: int, prom: float) -> HarvestCandidate:
        zs = {
            name: float(np.asarray(z)[p]) if np.isfinite(np.asarray(z)[p]) else 0.0
            for name, z in oriented.items()
        }
        support = sum(1 for v in zs.values() if v >= min_support_z)
        conf = float(min(1.0, support / max(1, len(zs))) * min(1.0, prom))
        return HarvestCandidate(
            index=int(p),
            score=float(fill[p]),
            confidence=round(conf, 4),
            support=support,
            channel_z={k: round(v, 4) for k, v in zs.items()},
            prominence=round(prom, 4),
        )

    prom_by_index = {int(p): float(pr) for p, pr in zip(peaks, props["prominences"])}
    ranked = sorted(
        (p for p, pr in prom_by_index.items() if pr >= prominence),
        key=lambda p: fill[p],
        reverse=True,
    )
    boundaries: list[HarvestCandidate] = []
    for p in ranked:
        if all(abs(days[p] - days[q.index]) >= min_gap_days for q in boundaries):
            boundaries.append(_candidate(p, prom_by_index[p]))
    weak = [
        _candidate(p, pr) for p, pr in prom_by_index.items() if pr < prominence
    ]
    boundaries.sort(key=lambda c: c.index)
    return boundaries, weak


def growth_state(progress: float) -> str:
    """Map normalized within-cycle progress to a VI-progress growth state."""
    p = float(progress)
    if not np.isfinite(p) or p < 0.0 or p > 1.0:
        return "absent"
    if p < 0.25:
        return PHASE_ORDER[0]
    if p < 0.60:
        return PHASE_ORDER[1]
    if p < 0.85:
        return PHASE_ORDER[2]
    return PHASE_ORDER[3]


def _greenup_index(
    s: int,
    e: int,
    ndvi: np.ndarray | None,
    savi: np.ndarray | None,
    *,
    bare_tol: float,
    rise_tol: float,
    min_rest_obs: int,
    min_obs: int,
) -> int | None:
    """First sustained rise after a *bare-soil* plateau, or None.

    Rest is not assumed after every harvest. It is claimed only when both
    indices stay within ``bare_tol`` of their post-harvest floor for at least
    ``min_rest_obs`` acquisitions; greenup is then the first sustained rise of
    ``rise_tol`` above that floor (not a fraction of full canopy amplitude).
    """
    if ndvi is None or savi is None or s < 0:
        return None
    lo, hi = s + 1, min(e, len(ndvi))
    if hi - lo < min_rest_obs + min_obs:
        return None
    a = np.asarray(ndvi, dtype=np.float64)[lo:hi]
    b = np.asarray(savi, dtype=np.float64)[lo:hi]
    if np.isfinite(a).sum() < min_rest_obs + min_obs or np.isfinite(b).sum() < min_rest_obs + min_obs:
        return None
    floor_a, floor_b = float(np.nanmin(a[:3])), float(np.nanmin(b[:3]))
    bare = (a <= floor_a + bare_tol) & (b <= floor_b + bare_tol)
    n_bare = 0
    while n_bare < len(bare) and bool(bare[n_bare]):
        n_bare += 1
    if n_bare < min_rest_obs:
        return None  # no bare-soil rest: the cycle starts at the harvest
    rising = (a >= floor_a + rise_tol) & (b >= floor_b + rise_tol)
    for j in range(n_bare, len(rising) - min_obs + 1):
        if rising[j : j + min_obs].all():
            return lo + j
    return None  # bare soil until the next boundary: conservative fallback


def build_cycle_rows(
    farm_id: str,
    ogc_fid: int,
    dates: list[str],
    candidates: list[HarvestCandidate],
    *,
    min_cycle_days: int = 180,
    max_cycle_days: int = 540,
    ndvi: np.ndarray | None = None,
    savi: np.ndarray | None = None,
    bare_tol: float = 0.02,
    rise_tol: float = 0.02,
    min_rest_obs: int = 3,
    greenup_min_obs: int = 2,
) -> list[dict[str, Any]]:
    """Assemble cycle records; a cycle starts at greenup, not at the last harvest."""
    days = np.array([np.datetime64(d, "D") for d in dates])
    day_int = (days - np.datetime64("2013-03-30", "D")).astype(int)
    rows: list[dict[str, Any]] = []
    boundaries = [int(c.index) for c in candidates]
    starts = [-1] + boundaries
    ends = boundaries + [len(dates)]
    for cid, (s, e) in enumerate(zip(starts, ends), start=1):
        if s < 0:
            start, rest_days, start_source = 0, 0, "series_start"
        else:
            g = _greenup_index(
                s, e, ndvi, savi,
                bare_tol=bare_tol, rise_tol=rise_tol,
                min_rest_obs=min_rest_obs, min_obs=greenup_min_obs,
            )
            start = g if g is not None else s
            rest_days = int(day_int[start] - day_int[s]) if g is not None else 0
            start_source = "greenup" if g is not None else "harvest"
        stop = min(e, len(dates) - 1)
        length = int(day_int[stop] - day_int[start])
        censored = bool(s < 0 or e >= len(dates))
        bounded = bool(s >= 0 and e < len(dates))
        short = bool(length < min_cycle_days and bounded)
        if length < min_cycle_days and not short:
            continue
        overlong = bool(length > max_cycle_days and not censored)
        conf_values = [
            c.confidence for c in candidates if start <= c.index < min(e, len(dates))
        ]
        rows.append(
            {
                "farm": farm_id,
                "ogc_fid": ogc_fid,
                "cycle_id": cid,
                "start_date": dates[start],
                "end_date": dates[stop],
                "start_harvest": bool(s >= 0),
                "end_harvest": bool(e < len(dates)),
                "length_days": length,
                "censored": censored,
                "rest_days": rest_days,
                "cycle_start_source": start_source,
                "status": (
                    "overlong_unresolved" if overlong
                    else "short_unresolved" if short
                    else "normal"
                ),
                "boundary_start_index": int(s),
                "boundary_end_index": int(e),
                "stage_confidence": float(np.mean(conf_values)) if conf_values else np.nan,
                "n_harvest_candidates": len(conf_values),
                "growth_state_at_midpoint": growth_state(0.5),
            }
        )
    return rows

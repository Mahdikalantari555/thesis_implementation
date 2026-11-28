"""L8/L9 same-phase consistency measurement on the mini store (task 0A.3).

Design.md §1A.2 / §8 sanity: "L8-only vs L9-only windows of the same field and
phase must not differ beyond the measured sensor offset". This module MEASURES
the difference — it never transforms data. A correction is only ever applied
through :mod:`ssl_anomaly.io.sensor_offset`'s pre-registered rule
(resolved CI + decide_applied at score level, E7 arm).

Phase here is the same DOY window (± ``phase_window_days``) across years — the
same-phase proxy the design's sanity test names. Pairs therefore carry real
interannual variation; that is why the result is reported as a robust median
with a MAD CI (via ``sensor_offset.band_offset``) rather than a single verdict
number. An unresolved CI is an honest outcome: no correction, decision logged.
"""

from __future__ import annotations

import numpy as np

from .sensor_offset import MAX_GAP_DAYS, band_offset

#: Same-phase matching window in DOY days (pre-registered; the 8-day
#: acquisition phasing of Landsat 8/9 together with cloud gaps motivates ±8).
PHASE_WINDOW_DAYS = MAX_GAP_DAYS


def _doy(dates) -> np.ndarray:
    months = np.array([int(d[5:7]) for d in dates])
    days = np.array([int(d[8:10]) for d in dates])
    cum = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    return (cum[months - 1] + days).astype(int)


def farm_band_medians(
    group,
    *,
    valid: np.ndarray | None = None,
    in_field: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray]:
    """Per-date per-band median over valid in-field pixels; NaN when absent.

    Same reduction as ``mini_lab/data._field_features`` (the field-level
    representation), kept independent so the consistency probe does not
    depend on the training data path.
    """
    cube = np.asarray(group["data"][:], dtype=np.float32)
    if valid is None:
        valid = np.asarray(group["pixel_valid"][:], dtype=bool)
    if in_field is None:
        in_field = np.asarray(group["in_field"][:], dtype=bool)
    names = [str(b) for b in group.attrs["band_order"]]
    n_dates = cube.shape[0]
    out = np.full((n_dates, len(names)), np.nan, dtype=np.float32)
    for di in range(n_dates):
        m = valid[di] & in_field
        for bi, _ in enumerate(names):
            v = cube[di, bi][m & np.isfinite(cube[di, bi])]
            if v.size:
                out[di, bi] = np.median(v)
    return names, out


def same_phase_pairs(
    medians: np.ndarray,
    sensor: np.ndarray,
    dates,
    *,
    phase_window_days: int = PHASE_WINDOW_DAYS,
) -> list[tuple[int, int]]:
    """(L8 index, L9 index) pairs matched by DOY window, closest first.

    Each L8 date is matched to at most one L9 date (the closest within the
    window); each L9 date is used at most once so a long L9 period cannot be
    counted repeatedly against many L8 neighbours (same pairing discipline as
    ``sensor_offset.find_pairs``).
    """
    doy = _doy(dates)
    l8 = [i for i, s in enumerate(sensor) if s == 0]
    l9 = [i for i, s in enumerate(sensor) if s == 1]
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for a in l8:
        best: tuple[int, int, int] | None = None
        for b in l9:
            if b in used:
                continue
            gap = abs(int(doy[a]) - int(doy[b]))
            if gap > phase_window_days:
                continue
            key = (gap, abs(a - b), b)
            if best is None or key < best:
                best = (key[0], key[1], b)
        if best is not None:
            b = best[2]
            used.add(b)
            pairs.append((a, b))
    return pairs


def same_phase_consistency(
    medians: np.ndarray,
    sensor: np.ndarray,
    dates,
    *,
    min_pairs: int = 10,
    phase_window_days: int = PHASE_WINDOW_DAYS,
    band_names: list[str] | None = None,
) -> dict:
    """Measure L8−L9 differences on same-DOY-window field medians.

    Returns per-band robust offsets with CI/resolution (via
    ``sensor_offset.band_offset``), the pair count, and a decision record:
    the pre-registered rule needs BOTH a resolved offset and the score-level
    acceptance comparison, so ``correction_allowed`` is False unless the
    offset itself is resolved — and even then it is only a candidate for E7,
    never an automatic transformation.
    """
    names = band_names or [str(b) for b in range(medians.shape[1])]
    pairs = same_phase_pairs(medians, sensor, dates, phase_window_days=phase_window_days)
    bands: dict[str, dict] = {}
    for bi, name in enumerate(names):
        diffs = []
        for a, b in pairs:
            va, vb = float(medians[a, bi]), float(medians[b, bi])
            if np.isfinite(va) and np.isfinite(vb):
                diffs.append(va - vb)
        if len(diffs) < 2:
            bands[name] = {
                "n_diffs": len(diffs),
                "offset": None, "ci_low": None, "ci_high": None,
                "resolved": False,
                "note": "insufficient same-phase pairs: no correction (fail closed)",
            }
            continue
        rec = band_offset(name, diffs, min_pairs=min_pairs)
        bands[name] = {
            **rec.to_dict(),
            "note": (
                "resolved: candidate for the E7 offset arm, pending the "
                "score-level acceptance rule (decide_applied)"
                if rec.resolved
                else "CI straddles zero or too few pairs: no correction (fail closed)"
            ),
        }
    n_resolved = sum(1 for v in bands.values() if v.get("resolved"))
    return {
        "phase_window_days": int(phase_window_days),
        "n_l8_dates": int((sensor == 0).sum()),
        "n_l9_dates": int((sensor == 1).sum()),
        "n_pairs": len(pairs),
        "min_pairs": int(min_pairs),
        "bands": bands,
        "resolved_count": n_resolved,
        "decision": (
            "no correction applied: measured same-phase differences are "
            "reported for the E7 arms; a correction requires a resolved CI "
            "AND the pre-registered score-level acceptance rule"
        ),
    }

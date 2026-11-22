"""Harvest detection and cycle segmentation.

A harvest is defined (per spec) as an NDVI trough with sufficient prominence
that is preceded by a high plateau and followed by a rise. Cycles are the
intervals between consecutive detected harvests; the first/last cycles are
censored. Replants reset the ratoon counter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from ..utils.arrays import hampel_filter
from .smoothing import to_daily_grid, whittaker_tune_lambda

log = logging.getLogger("ssl_anomaly.phenology.cycles")


@dataclass(frozen=True)
class Cycle:
    field: int
    cycle_id: int
    t_start: int  # inclusive day index (start of cycle, just after start_harvest)
    t_end: int  # exclusive day index (end of cycle, just before end_harvest)
    start_harvest: int  # day of harvest that BEGINS this cycle (-1 for leading censored)
    end_harvest: int  # day of harvest that ENDS this cycle (last day for trailing censored)
    censored: bool
    kind: str  # "plant" | "ratoon"
    ratoon_no: int
    ndvi_at_harvest: float = float("nan")

    @property
    def length_days(self) -> int:
        return int(self.t_end - self.t_start)

    def to_dict(self) -> dict:
        return {
            "field": int(self.field),
            "cycle_id": int(self.cycle_id),
            "t_start": int(self.t_start),
            "t_end": int(self.t_end),
            "start_harvest": int(self.start_harvest),
            "end_harvest": int(self.end_harvest),
            # Retain 'harvest_day' for backward-compat (end harvest) — scripts
            # can migrate to start_harvest for dsh computation.
            "harvest_day": int(self.end_harvest),
            "length_days": self.length_days,
            "censored": bool(self.censored),
            "kind": self.kind,
            "ratoon_no": int(self.ratoon_no),
            "ndvi_at_harvest": float(self.ndvi_at_harvest),
        }


def detect_harvests(
    grid_days: np.ndarray,
    ndvi_smooth: np.ndarray,
    *,
    min_prominence: float = 0.15,
    min_cycle_days: int = 180,
    plateau_frac: float = 0.7,
    rise_frac: float = 0.15,
) -> np.ndarray:
    """Detect harvest indices into ``grid_days`` (daily grid).

    1. Find deep troughs via scipy find_peaks(-ndvi).
    2. Require a high plateau before (mean over preceding window ≥ frac*max).
    3. Require a rise after (NDVI ≥ trough + rise_frac within min_cycle_days).
    """
    y = np.asarray(ndvi_smooth, dtype=np.float64)
    n = y.size
    if n < min_cycle_days:
        return np.array([], dtype=np.int64)

    ymax = float(np.nanmax(y))
    ymin = float(np.nanmin(y))
    prom = max(min_prominence, 0.1 * (ymax - ymin))
    troughs, props = find_peaks(-y, prominence=prom, distance=min_cycle_days // 2)

    keep: list[int] = []
    for idx in troughs:
        # Plateau before: look at 60-day window ending min_cycle_days before trough
        pre_start = max(0, idx - int(min_cycle_days * 0.7))
        pre_end = max(pre_start + 1, idx - 30)
        pre = y[pre_start:pre_end]
        if pre.size == 0 or np.nanmean(pre) < plateau_frac * ymax:
            continue
        # Rise after: require NDVI to rise by rise_frac*(ymax-ymin) within min_cycle_days
        post_end = min(n, idx + min_cycle_days)
        post = y[idx:post_end]
        trough_val = y[idx]
        if post.size < 30:
            continue
        if np.nanmax(post) < trough_val + rise_frac * (ymax - trough_val):
            continue
        keep.append(int(idx))

    return np.array(keep, dtype=np.int64)


def _detect_replant(
    grid_days: np.ndarray,
    ndvi_smooth: np.ndarray,
    harvests: np.ndarray,
    *,
    gap_days: int = 120,
    low_ndvi: float = 0.3,
) -> set[int]:
    """Return a set of harvest indices after which a replant occurs (reset ratoon).

    A replant is signaled by a long post-harvest period with NDVI below ``low_ndvi``
    before the next rise (gap_days), or by a gap between harvests exceeding
    twice min_cycle_days with low NDVI throughout.
    """
    replant = set()
    for i in range(len(harvests) - 1):
        h0, h1 = harvests[i], harvests[i + 1]
        gap = h1 - h0
        seg = ndvi_smooth[h0:h1]
        if seg.size == 0:
            continue
        days_below = np.sum(seg < low_ndvi)
        if days_below >= gap_days or gap > 540:
            replant.add(i)
    return replant


def _prepare_and_detect(
    field_id: int,
    times_days: np.ndarray,
    ndvi: np.ndarray,
    quality: np.ndarray | None,
    *,
    lam: float | None,
    hampel_window: int,
    hampel_sigma: float,
    min_prominence: float,
    min_cycle_days: int,
    plateau_frac: float,
    rise_frac: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Verbatim v1 preprocessing + detection (Hampel → Whittaker CV → daily
    grid → ``detect_harvests``). Returns ``(grid, ys, harvest_idx)``; ``grid``
    and ``ys`` are ``None`` when the series is unusable and ``harvest_idx``
    is then empty. Shared by ``segment_cycles`` and the fused detector so the
    two cannot drift.
    """
    times_days = np.asarray(times_days, dtype=np.int64)
    ndvi = np.asarray(ndvi, dtype=np.float64)

    # Sort and drop NaNs
    order = np.argsort(times_days)
    times_days, ndvi = times_days[order], ndvi[order]
    if quality is not None:
        quality = np.asarray(quality, dtype=np.float64)[order]
    finite = np.isfinite(ndvi)
    times_days, ndvi = times_days[finite], ndvi[finite]
    if quality is not None:
        quality = quality[finite]
    if times_days.size < 30:
        log.debug("Field %d has too few observations (%d); no cycles", field_id, times_days.size)
        return None, None, np.array([], dtype=np.int64)

    # Hampel outlier removal so single cloud-missed points aren't called harvests
    outlier_mask = hampel_filter(times_days, ndvi, window=hampel_window, n_sigma=hampel_sigma)
    ndvi_clean = ndvi.copy()
    ndvi_clean[outlier_mask] = np.nan
    keep = np.isfinite(ndvi_clean)
    if keep.sum() < 20:
        return None, None, np.array([], dtype=np.int64)
    times_days, ndvi_clean = times_days[keep], ndvi_clean[keep]
    if quality is not None:
        quality = quality[keep]

    if lam is None:
        lam, _ = whittaker_tune_lambda(times_days, ndvi_clean, quality)
        log.debug("Field %d: chosen lambda=%.2f", field_id, lam)

    grid, ys = to_daily_grid(times_days, ndvi_clean, quality=quality, lam=float(lam))

    harvest_idx = detect_harvests(
        grid, ys,
        min_prominence=min_prominence,
        min_cycle_days=min_cycle_days,
        plateau_frac=plateau_frac,
        rise_frac=rise_frac,
    )
    return grid, ys, harvest_idx


def segment_cycles(
    field_id: int,
    times_days: np.ndarray,
    ndvi: np.ndarray,
    quality: np.ndarray | None = None,
    *,
    lam: float | None = None,
    min_prominence: float = 0.15,
    min_cycle_days: int = 180,
    max_cycle_days: int = 540,
    plateau_frac: float = 0.7,
    rise_frac: float = 0.15,
    replant_gap_days: int = 120,
    replant_low_ndvi: float = 0.3,
    hampel_window: int = 5,
    hampel_sigma: float = 3.0,
) -> list[Cycle]:
    """Segment harvest-driven cycles for a single field.

    Parameters
    ----------
    field_id:
        Field identifier (stored on each Cycle).
    times_days:
        1-D int array, days-since-epoch for each observation (sorted or will be sorted).
    ndvi:
        Observed NDVI values at those times.
    quality:
        Optional quality weights (valid_frac x cloud proximity).
    lam:
        Whittaker smoothing parameter; if None, tuned by CV.
    """
    grid, ys, harvest_idx = _prepare_and_detect(
        field_id, times_days, ndvi, quality,
        lam=lam, hampel_window=hampel_window, hampel_sigma=hampel_sigma,
        min_prominence=min_prominence, min_cycle_days=min_cycle_days,
        plateau_frac=plateau_frac, rise_frac=rise_frac,
    )
    if harvest_idx.size == 0:
        return []

    replant_harvests = _detect_replant(
        grid, ys, harvest_idx, gap_days=replant_gap_days, low_ndvi=replant_low_ndvi
    )

    boundary_days = [int(grid[i]) for i in harvest_idx]
    replant_days = {int(grid[harvest_idx[i]]) for i in replant_harvests}
    return _build_cycles_from_boundaries(
        field_id, boundary_days, replant_days, grid, ys,
        min_cycle_days=min_cycle_days, max_cycle_days=max_cycle_days,
    )


def _ys_at(grid: np.ndarray, ys: np.ndarray, day: int) -> float:
    """Smoothed series value at a boundary day (grid is daily)."""
    i = int(day) - int(grid[0])
    if 0 <= i < ys.size and np.isfinite(ys[i]):
        return float(ys[i])
    for off in (1, -1, 2, -2, 3, -3, 5, -5, 10, -10):
        j = i + off
        if 0 <= j < ys.size and np.isfinite(ys[j]):
            return float(ys[j])
    # audit 2026-11-21 item 21: fabricating 0.0 violated §2 provenance — a
    # missing boundary value is absent (NaN), never zero.
    return float("nan")


def _build_cycles_from_boundaries(
    field_id: int,
    boundary_days: list[int],
    replant_days: set[int],
    grid: np.ndarray,
    ys: np.ndarray,
    *,
    min_cycle_days: int,
    max_cycle_days: int,
) -> list[Cycle]:
    """Build v1-semantics windows from an ordered harvest-boundary day list.

    Shared by ``segment_cycles`` and ``segment_cycles_fused``: leading
    censored plant window, 180–540-day complete windows between consecutive
    boundaries, trailing censored window. ``replant_days`` holds boundary
    days after which the ratoon numbering resets.
    """
    cycles: list[Cycle] = []
    cycles.append(
        Cycle(
            field=field_id,
            cycle_id=0,
            t_start=int(grid[0]),
            t_end=int(boundary_days[0]),
            start_harvest=-1,
            end_harvest=int(boundary_days[0]),
            censored=True,
            kind="plant",
            ratoon_no=0,
            ndvi_at_harvest=_ys_at(grid, ys, boundary_days[0]),
        )
    )
    ratoon_no = 0
    cid = 1
    last_complete_kind = "plant"
    for b0, b1 in zip(boundary_days[:-1], boundary_days[1:]):
        length = int(b1) - int(b0)
        if length < min_cycle_days or length > max_cycle_days:
            continue
        if int(b0) in replant_days:
            ratoon_no = 0
            kind = "plant"
        else:
            ratoon_no += 1
            kind = "ratoon"
        last_complete_kind = kind
        cycles.append(
            Cycle(
                field=field_id,
                cycle_id=cid,
                t_start=int(b0),
                t_end=int(b1),
                start_harvest=int(b0),
                end_harvest=int(b1),
                censored=False,
                kind=kind,
                ratoon_no=ratoon_no,
                ndvi_at_harvest=_ys_at(grid, ys, b1),
            )
        )
        cid += 1
    hlast = boundary_days[-1]
    cycles.append(
        Cycle(
            field=field_id,
            cycle_id=cid,
            t_start=int(hlast),
            t_end=int(grid[-1]) + 1,
            start_harvest=int(hlast),
            end_harvest=int(grid[-1]),
            censored=True,
            kind=last_complete_kind,
            ratoon_no=max(0, ratoon_no),
            ndvi_at_harvest=_ys_at(grid, ys, hlast),
        )
    )
    return cycles


def cycles_to_dataframe(cycles: list[Cycle]) -> pd.DataFrame:
    return pd.DataFrame([c.to_dict() for c in cycles])


# ---------------------------------------------------------------------------
# Fused multi-index detector (change cycle-detection-v3-multindex, D1–D3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FusionParams:
    """Frozen thresholds from the phase-0 diagnosis (design D2).

    Values are the measured primary operating point; the sensitivity grid
    lives in ``reports/cycle_detection_diagnosis/report.json``. Changing any
    value re-opens the diagnosis, not the detector.
    """

    veg_drop: float = 0.20      # NDVI or SAVI drop vs preceding-window median
    lst_rise: float = 3.0       # LST rise (K) vs preceding-window median
    swir_rise: float = 0.05     # SWIR1 rise vs preceding-window median
    ctx_days: int = 120         # preceding context window length
    min_gap: int = 5            # context window ends this many days before t
    vf_min: float = 0.5         # minimum valid_frac for a usable observation
    edge_guard: int = 30        # candidates kept this far from lost-span edges
    dedup_days: int = 180       # min separation from any accepted boundary


def fusion_params_hash(params: FusionParams) -> str:
    import hashlib
    import json as _json

    payload = {k: getattr(params, k) for k in (
        "veg_drop", "lst_rise", "swir_rise", "ctx_days", "min_gap",
        "vf_min", "edge_guard", "dedup_days")}
    return hashlib.sha256(_json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def lost_spans_from_harvests(
    harvests: list[int],
    series_start: int,
    series_end: int,
    *,
    min_cycle_days: int,
    max_cycle_days: int,
) -> list[dict]:
    """Lead-in / skipped-gap / trailing spans v1 could not bound (a span
    between consecutive detected harvests longer than ``max_cycle_days`` was
    skipped entirely by v1)."""
    if not harvests:
        return []
    spans: list[dict] = []
    if harvests[0] - series_start >= min_cycle_days:
        spans.append({"kind": "lead", "t0": series_start, "t1": harvests[0]})
    for h0, h1 in zip(harvests[:-1], harvests[1:]):
        if h1 - h0 > max_cycle_days:
            spans.append({"kind": "gap", "t0": h0, "t1": h1})
    if series_end - harvests[-1] >= min_cycle_days:
        spans.append({"kind": "trail", "t0": harvests[-1], "t1": series_end})
    return spans


def detect_harvest_candidates(
    field_id: int,
    times_days: np.ndarray,
    ndvi: np.ndarray,
    savi: np.ndarray,
    lst: np.ndarray,
    swir1: np.ndarray,
    quality: np.ndarray | None,
    base_harvests: list[int],
    params: FusionParams,
    *,
    series_start: int,
    series_end: int,
    min_cycle_days: int,
    max_cycle_days: int,
) -> list[int]:
    """Fusion candidates on the RAW observation series (design D3).

    A candidate is an observation date where the vegetation level drops by
    ≥ ``veg_drop`` (NDVI or SAVI, whichever is stronger) versus the median of
    the preceding ``ctx_days`` window AND the drop is corroborated by an LST
    rise ≥ ``lst_rise`` or a SWIR1 rise ≥ ``swir_rise`` (a bare-soil/residue
    thermal+SWIR signature that dust or cloud artifacts do not produce).
    Candidates are accepted greedily inside lost spans, ≥ ``edge_guard`` days
    from span edges and ≥ ``dedup_days`` from every other boundary. No RNG —
    deterministic by construction.

    With no v1 harvests the whole series is one open span, so fields the v1
    detector returned empty for can still gain their first cycles.
    """
    t = np.asarray(times_days, dtype=np.int64)
    ndvi = np.asarray(ndvi, dtype=np.float64)
    savi = np.asarray(savi, dtype=np.float64)
    lst = np.asarray(lst, dtype=np.float64)
    swir1 = np.asarray(swir1, dtype=np.float64)
    vf = (np.ones_like(t, dtype=np.float64) if quality is None
          else np.asarray(quality, dtype=np.float64))

    if base_harvests:
        spans = lost_spans_from_harvests(
            sorted(base_harvests), series_start, series_end,
            min_cycle_days=min_cycle_days, max_cycle_days=max_cycle_days)
    else:
        spans = [{"kind": "lead", "t0": series_start, "t1": series_end}]

    accepted: list[int] = []
    for sp in spans:
        rows_t: list[int] = []
        rows_veg: list[float] = []
        rows_lst: list[float] = []
        rows_swir: list[float] = []
        for k in range(t.size):
            if not np.isfinite(ndvi[k]) or vf[k] < params.vf_min:
                continue
            if t[k] < params.ctx_days:
                continue
            ctx = (t >= t[k] - params.ctx_days) & (t < t[k] - params.min_gap)
            if ctx.sum() < 2:
                continue
            pre_n = ndvi[ctx & np.isfinite(ndvi)]
            pre_l = lst[ctx & np.isfinite(lst)]
            pre_s = savi[ctx & np.isfinite(savi)]
            pre_w = swir1[ctx & np.isfinite(swir1)]
            if pre_n.size < 2 or pre_l.size < 2:
                continue
            drop_ndvi = float(np.median(pre_n) - ndvi[k])
            drop_savi = float(np.median(pre_s) - savi[k]) if pre_s.size >= 2 else np.nan
            lst_rise = float(lst[k] - np.median(pre_l))
            swir_rise = (float(swir1[k] - np.median(pre_w)) if pre_w.size >= 2
                         else np.nan)
            veg = max(np.nan_to_num(drop_ndvi, nan=-9.0),
                      np.nan_to_num(drop_savi, nan=-9.0))
            rows_t.append(int(t[k]))
            rows_veg.append(veg)
            rows_lst.append(lst_rise)
            rows_swir.append(swir_rise)
        if not rows_t:
            continue
        tt = np.array(rows_t)
        veg = np.array(rows_veg)
        lst_r = np.array(rows_lst)
        swir_r = np.array(rows_swir)
        fires = np.where(
            (veg >= params.veg_drop)
            & ((lst_r >= params.lst_rise) | (np.nan_to_num(swir_r, nan=-9.0) >= params.swir_rise))
        )[0]
        for idx in fires:
            day = int(tt[idx])
            if day <= sp["t0"] + params.edge_guard or day >= sp["t1"] - params.edge_guard:
                continue
            if any(abs(day - a) < params.dedup_days for a in accepted):
                continue
            if any(abs(day - h) < params.dedup_days for h in base_harvests):
                continue
            accepted.append(day)
    return sorted(accepted)


def segment_cycles_fused(
    field_id: int,
    times_days: np.ndarray,
    ndvi: np.ndarray,
    savi: np.ndarray,
    lst: np.ndarray,
    swir1: np.ndarray,
    quality: np.ndarray | None = None,
    *,
    fusion: FusionParams | None = None,
    lam: float | None = None,
    min_prominence: float = 0.15,
    min_cycle_days: int = 180,
    max_cycle_days: int = 540,
    plateau_frac: float = 0.7,
    rise_frac: float = 0.15,
    replant_gap_days: int = 120,
    replant_low_ndvi: float = 0.3,
    hampel_window: int = 5,
    hampel_sigma: float = 3.0,
) -> list[Cycle]:
    """v1 semantics over the UNION boundary set: v1's detected harvests plus
    multi-index fusion candidates (design D1–D3). With no candidates firing
    the output is identical to :func:`segment_cycles` (v1-subset property).

    Candidates gain their ``ndvi_at_harvest`` from the smoothed daily series
    at the candidate day; replant re-detection is kept for v1 harvests only
    (candidates continue the ratoon chain — recorded D7 limit).
    """
    fusion = fusion or FusionParams()
    grid, ys, harvest_idx = _prepare_and_detect(
        field_id, times_days, ndvi, quality,
        lam=lam, hampel_window=hampel_window, hampel_sigma=hampel_sigma,
        min_prominence=min_prominence, min_cycle_days=min_cycle_days,
        plateau_frac=plateau_frac, rise_frac=rise_frac,
    )
    if grid is None:
        return []
    base_days = [int(grid[i]) for i in harvest_idx]
    cand_days = detect_harvest_candidates(
        field_id, times_days, ndvi, savi, lst, swir1, quality, base_days,
        fusion,
        series_start=int(grid[0]), series_end=int(grid[-1]),
        min_cycle_days=min_cycle_days, max_cycle_days=max_cycle_days,
    )
    boundaries = sorted(set(base_days) | set(cand_days))
    if not boundaries:
        return []
    replant_harvests = (
        _detect_replant(grid, ys, harvest_idx,
                        gap_days=replant_gap_days, low_ndvi=replant_low_ndvi)
        if harvest_idx.size else set()
    )
    replant_days = {int(grid[harvest_idx[i]]) for i in replant_harvests}
    return _build_cycles_from_boundaries(
        field_id, boundaries, replant_days, grid, ys,
        min_cycle_days=min_cycle_days, max_cycle_days=max_cycle_days,
    )

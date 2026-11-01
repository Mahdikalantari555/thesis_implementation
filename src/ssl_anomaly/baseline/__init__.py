"""Robust statistical baseline B1: stratum-aware median/MAD, iterative trimming, scoring."""

from __future__ import annotations

from .robust import RobustBaseline, build_stratum_table, fit_baseline, lookup_median_scale
from .score import (
    aggregate_band_scores,
    cycle_anomaly_burden,
    flag_runs,
    score_all_cycles,
    score_cycle,
)
from .trimming import trimmed_normal_cycles

__all__ = [
    "RobustBaseline",
    "aggregate_band_scores",
    "build_stratum_table",
    "cycle_anomaly_burden",
    "fit_baseline",
    "flag_runs",
    "lookup_median_scale",
    "score_all_cycles",
    "score_cycle",
    "trimmed_normal_cycles",
]

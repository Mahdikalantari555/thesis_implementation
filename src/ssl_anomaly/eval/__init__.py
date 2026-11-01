"""Evaluation: synthetic injection, proxy validation, stability, statistical comparison."""

from __future__ import annotations

from .stats import (
    bootstrap_ci,
    holm_correction,
    paired_wilcoxon,
    spearman_correlation,
)
from .synthetic import (
    InjectionConfig,
    inject_anomaly,
    synthetic_benchmark,
)

__all__ = [
    "InjectionConfig",
    "bootstrap_ci",
    "holm_correction",
    "inject_anomaly",
    "paired_wilcoxon",
    "spearman_correlation",
    "synthetic_benchmark",
]

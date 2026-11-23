"""Metrics and statistics shared by the capstone and the benchmark.

Why a dedicated module rather than reusing ``eval/stats.py``: that module is
tuned to the RQ1–RQ8 suite's shapes and predates the review checklist items
that now bind this work. This one adds what the two changes require:

- AUPRC with **prevalence reported alongside** and an explicit no-tie-hidden
  convention (``metric-discipline`` in both specs). A heavy-imbalance cell
  reported without prevalence is a review failure, not a style issue.
- **Wilson** intervals for rates (already in ``monitor/scoring``) and **paired
  bootstrap** that resamples the *farm/event* cluster, not individual points —
  the AGENTS.md §7 checklist names point-resampling as dependence-blind.
- Multiple-comparison correction (Holm + Benjamini–Hochberg) with corrected
  p-values returned alongside the raw ones.
- Effect sizes (Cliff's delta and rank-biserial) with CIs.

Nothing here reads data, imports torch, or writes files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# AUPRC / AUROC / F1
# ---------------------------------------------------------------------------


def auprc(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Average precision and the positive prevalence of the cell.

    Average precision is computed as the area under the precision–recall curve
    with the step-wise (interpolated-free) definition matching
    ``sklearn.metrics.average_precision_score``; we implement it here so the
    repository has no hidden dependency in the reporting path and so ties are
    handled explicitly: all tied scores form one threshold, never split.

    Returns
    -------
    (ap, prevalence)
        ``ap`` is NaN when the cell has no positives or no negatives — an
        undecidable cell is ``absent``, not 0.5.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    if s.shape != y.shape:
        raise ValueError(f"scores {s.shape} and labels {y.shape} must align")
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    prevalence = n_pos / y.size if y.size else float("nan")
    if n_pos == 0 or n_neg == 0:
        return float("nan"), prevalence

    order = np.argsort(-s, kind="mergesort")
    s_sorted = s[order]
    y_sorted = y[order]

    # Tie groups: precision/recall are evaluated once per distinct threshold.
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    is_last_of_tie = np.r_[s_sorted[1:] != s_sorted[:-1], True]
    tp = tp[is_last_of_tie]
    fp = fp[is_last_of_tie]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    prev_recall = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_recall) * precision)), float(prevalence)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC with mid-rank tie handling (Mann–Whitney U)."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _midranks(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _midranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def f1_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=bool).ravel()
    pred = s >= threshold
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else float("nan")
    )
    return {"precision": precision, "recall": recall, "f1": f1, "threshold": float(threshold)}


def select_threshold_on_validation(
    val_scores: np.ndarray, val_labels: np.ndarray, grid: int = 101
) -> float:
    """Pick the F1-maximising threshold on VALIDATION data only.

    The spec requires the F1 threshold to come from the validation partition
    and the procedure to be recorded. This function is the only place a
    threshold may be chosen for F1 reporting; it never sees test data by
    construction (the caller only ever holds dev/val arrays here).

    Ties are broken toward the *higher* threshold (fewer false alarms) — a
    deterministic, conservative rule recorded in the manifest.
    """
    s = np.asarray(val_scores, dtype=np.float64).ravel()
    y = np.asarray(val_labels, dtype=bool).ravel()
    finite = np.isfinite(s)
    if not finite.any() or y.sum() == 0:
        return float("nan")
    candidates = np.unique(np.quantile(s[finite], np.linspace(0.0, 1.0, grid)))
    best_t, best_f1 = float("nan"), -1.0
    for t in candidates[::-1]:  # descending: ties resolve to the higher threshold
        f1 = f1_at_threshold(s, y, float(t))["f1"]
        if np.isfinite(f1) and f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


# ---------------------------------------------------------------------------
# Rates and intervals
# ---------------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial rate (same z as findings 18–25)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def bootstrap_ci(
    values: np.ndarray,
    *,
    statistic: str = "mean",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 13,
) -> tuple[float, float, float]:
    """Percentile bootstrap of a scalar statistic; returns (point, lo, hi)."""
    x = np.asarray(values, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    func = {"mean": np.mean, "median": np.median}.get(statistic)
    if func is None:
        raise ValueError(f"unknown statistic {statistic!r}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boot = func(x[idx], axis=1)
    return float(func(x)), float(np.quantile(boot, alpha / 2)), float(np.quantile(boot, 1 - alpha / 2))


def bootstrap_ci_stat(
    y: np.ndarray,
    scores: np.ndarray,
    stat_fn,
    *,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 13,
    clusters: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Callable-based percentile bootstrap: ``stat_fn(y[idx], scores[idx])``.

    Needed because AP is a two-argument statistic, so the named-statistic
    :func:`bootstrap_ci` cannot express it. When ``clusters`` is given the
    resampling unit is the cluster (farm/event), keeping the within-farm and
    between-farm structure of the evaluation set — the same discipline as
    :func:`clustered_bootstrap_diff`, and the pre-registered unit for H1–H5.

    Replicates where the resample is single-class are SKIPPED (AP is undefined
    there) and the count of usable replicates is returned as the third element so
    a reader can see an interval built from 420 of 500 draws rather than assuming
    500.
    """
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.size != s.size:
        raise ValueError(f"y {y.shape} and scores {s.shape} must align")
    rng = np.random.default_rng(seed)
    boots: list[float] = []
    if clusters is None:
        groups = [np.arange(y.size)]
    else:
        c = np.asarray(clusters)
        if c.size != y.size:
            raise ValueError("clusters must align with y")
        groups = [np.flatnonzero(c == u) for u in np.unique(c)]
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        if np.unique(y[idx]).size < 2:
            continue
        val = stat_fn(y[idx], s[idx])
        if np.isfinite(val):
            boots.append(float(val))
    point = float(stat_fn(y, s)) if np.unique(y).size >= 2 else float("nan")
    if not boots:
        return point, float("nan"), float("nan")
    return (point, float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)))


def clustered_bootstrap_metric_diff(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    stat_fn,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> dict:
    """Paired cluster bootstrap of ``stat_fn(a) - stat_fn(b)``.

    Distinct from :func:`clustered_bootstrap_diff`, which differences the MEAN of
    two score vectors. Here the quantity is a *statistic of the scores* (AP,
    AUROC, ...), so two rungs with very different APs cannot produce a ~0 delta
    just because their mean predicted probabilities coincide — which is exactly
    the nonsense the ladder produced before this existed (AP 0.507 vs 1.000
    reported as a delta of -0.009).

    Whole clusters are resampled. Replicates in which either arm has a
    single-class label set are skipped, and the count of usable replicates is
    returned alongside the interval.
    """
    sa = np.asarray(scores_a, dtype=np.float64).ravel()
    sb = np.asarray(scores_b, dtype=np.float64).ravel()
    y = np.asarray(labels).astype(int).ravel()
    c = np.asarray(clusters).ravel()
    if not (sa.size == sb.size == y.size == c.size):
        raise ValueError(
            f"scores_a {sa.size}, scores_b {sb.size}, labels {y.size}, clusters "
            f"{c.size} must all align"
        )
    uniq = np.unique(c)
    if uniq.size < 2:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p_two_sided": float("nan"), "n_clusters": int(uniq.size),
                "n_usable_boot": 0}
    groups = [np.flatnonzero(c == u) for u in uniq]
    point = float(stat_fn(sa, y) - stat_fn(sb, y))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        if np.unique(y[idx]).size < 2:
            continue
        d = stat_fn(sa[idx], y[idx]) - stat_fn(sb[idx], y[idx])
        if np.isfinite(d):
            diffs.append(float(d))
    if not diffs:
        return {"point": point, "lo": float("nan"), "hi": float("nan"),
                "p_two_sided": float("nan"), "n_clusters": int(uniq.size),
                "n_usable_boot": 0}
    arr = np.asarray(diffs)
    share_other_side = float(np.mean(arr <= 0.0)) if point > 0 else float(np.mean(arr >= 0.0))
    p = min(1.0, max(share_other_side * 2.0, 1.0 / len(arr)))
    return {
        "point": point,
        "lo": float(np.quantile(arr, alpha / 2)),
        "hi": float(np.quantile(arr, 1 - alpha / 2)),
        "p_two_sided": float(p),
        "n_clusters": int(uniq.size),
        "n_usable_boot": len(arr),
    }


def clustered_bootstrap_diff(
    a: np.ndarray,
    b: np.ndarray,
    clusters: np.ndarray,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 13,
) -> dict:
    """Paired bootstrap of ``mean(a) - mean(b)`` resampling CLUSTERS.

    ``clusters`` labels the farm/event each paired observation belongs to; whole
    clusters are resampled so within-farm dependence is preserved. Resampling
    individual points would understate the interval — the exact defect the
    AGENTS.md §7 checklist names ("paired bootstrap preserving farm/event
    dependence").

    Returns ``point``, ``lo``, ``hi``, ``p_two_sided`` (bootstrap-based, i.e.
    the share of resamples on the other side of zero, floored at 1/n_boot) and
    the number of clusters.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    c = np.asarray(clusters).ravel()
    if not (a.size == b.size == c.size):
        raise ValueError("a, b and clusters must have the same length")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b, c = a[keep], b[keep], c[keep]
    uniq = np.unique(c)
    if uniq.size < 2:
        return {
            "point": float(np.mean(a) - np.mean(b)) if a.size else float("nan"),
            "lo": float("nan"), "hi": float("nan"),
            "p_two_sided": float("nan"), "n_clusters": int(uniq.size),
        }
    point = float(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        pick = rng.choice(uniq, size=uniq.size, replace=True)
        mask = np.isin(c, pick)
        diffs[i] = np.mean(a[mask]) - np.mean(b[mask])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p = 2 * min(np.mean(diffs <= 0.0), np.mean(diffs >= 0.0))
    p = float(np.clip(max(p, 1.0 / n_boot), 0.0, 1.0))
    return {"point": point, "lo": float(lo), "hi": float(hi),
            "p_two_sided": p, "n_clusters": int(uniq.size)}


# ---------------------------------------------------------------------------
# Multiple comparisons
# ---------------------------------------------------------------------------


def holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down corrected p-values (family-wise error control)."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (n - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def holm_correction_labelled(p_values: dict[str, float]) -> dict[str, float]:
    """Holm correction over a NAMED family of tests.

    The array form loses the association between a p-value and the comparison it
    belongs to; in a report that association is the whole point ("the B2→B3
    delta survives correction" is a different statement from "some delta does").
    Insertion order is preserved so a rendered table reads in the declared order.
    """
    names = list(p_values)
    if not names:
        return {}
    adjusted = holm_correction(np.array([p_values[n] for n in names], dtype=np.float64))
    return {name: float(value) for name, value in zip(names, adjusted, strict=True)}


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted p-values (screening use)."""
    p = np.asarray(p_values, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p)[::-1]
    adjusted = np.empty(n, dtype=np.float64)
    running = 1.0
    for rank_from_top, idx in enumerate(order):
        factor = n / (n - rank_from_top)
        running = min(running, min(1.0, p[idx] * factor))
        adjusted[idx] = running
    return adjusted


# ---------------------------------------------------------------------------
# Effect sizes
# ---------------------------------------------------------------------------


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> tuple[float, str]:
    """Cliff's delta with its conventional magnitude label.

    delta = P(a > b) - P(a < b) ∈ [-1, 1]; the label is the Romano et al.
    convention (|d| < 0.147 negligible, < 0.33 small, < 0.474 medium, else
    large) and is reported *with* the value — never instead of it.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan"), "absent"
    gt = np.sum(x[:, None] > y[None, :])
    lt = np.sum(x[:, None] < y[None, :])
    delta = float((gt - lt) / (x.size * y.size))
    mag = abs(delta)
    if mag < 0.147:
        label = "negligible"
    elif mag < 0.33:
        label = "small"
    elif mag < 0.474:
        label = "medium"
    else:
        label = "large"
    return delta, label


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation from the Mann–Whitney U of paired samples."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    d = x - y
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    ranks = _midranks(np.abs(d))
    pos = ranks[d > 0].sum()
    neg = ranks[d < 0].sum()
    total = pos + neg
    return float((pos - neg) / total) if total > 0 else float("nan")


# ---------------------------------------------------------------------------
# Trajectory quality (T4)
# ---------------------------------------------------------------------------


def masked_trajectory_error(
    pred: np.ndarray,
    truth: np.ndarray,
    mask: np.ndarray,
    *,
    horizons: tuple[int, ...] = (1, 3),
) -> dict[str, float]:
    """Per-horizon masked MAE of a forecast against observed future bins.

    Shapes: ``pred`` is ``(N, T, H, C)`` — ``pred[:, t, h-1, :]`` forecasts bin
    ``t + h`` — and ``truth``/``mask`` are ``(N, T, C)``. Unobserved target bins
    contribute to nothing (neither numerator nor denominator), and a horizon with
    zero observed targets returns NaN rather than a flattering 0.
    """
    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if p.ndim != 4:
        raise ValueError(f"pred must be (N, T, H, C), got {p.shape}")
    if t.shape != m.shape or t.ndim != 3:
        raise ValueError(f"truth {t.shape} and mask {m.shape} must both be (N, T, C)")
    if p.shape[:2] != t.shape[:2] or p.shape[3] != t.shape[2]:
        raise ValueError(
            f"shape mismatch pred{p.shape} truth{t.shape} mask{m.shape}"
        )
    T = p.shape[1]
    out: dict[str, float] = {}
    for h in horizons:
        if h < 1 or h > p.shape[2]:
            raise ValueError(f"horizon {h} outside the forecast axis (1..{p.shape[2]})")
        num = den = 0.0
        for b in range(T - h):
            sel = m[:, b + h]                      # (N, C) observed targets
            if not sel.any():
                continue
            diff = p[:, b, h - 1][sel] - t[:, b + h][sel]
            num += float(np.sum(np.abs(diff)))
            den += float(sel.sum())
        out[f"mae_h{h}"] = num / den if den > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Cost accounting (benchmark fairness)
# ---------------------------------------------------------------------------


@dataclass
class CostRecord:
    """One architecture's measured cost. Params are NOT compute — the spec says so."""

    params: int
    train_seconds: float
    inference_ms_per_batch: float
    peak_vram_mb: float | None
    flops: float | None
    batch_shape: tuple[int, ...]
    device: str
    dtype: str

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "train_seconds": self.train_seconds,
            "inference_ms_per_batch": self.inference_ms_per_batch,
            "peak_vram_mb": self.peak_vram_mb,
            "flops": self.flops,
            "batch_shape": list(self.batch_shape),
            "device": self.device,
            "dtype": self.dtype,
            "note": (
                "parameter counts are not compute counts; FLOPs and latency are "
                "reported separately (spec architecture-benchmarking / Fair model "
                "comparison)"
            ),
        }


def count_parameters(model) -> int:
    """Trainable parameter count (requires_grad only — frozen heads excluded)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

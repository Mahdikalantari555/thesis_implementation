"""Metric computation for one cell: macro-AUPRC and the secondary panel (tasks 7.x).

Two rules drive this module.

**Macro = equal cell weight.** The primary number is the mean over the 15
(family, severity) cells of each cell's AUPRC, unweighted. Weighting by event count
would let ``step/mild`` — the most numerous, least interesting cell — dominate a
headline claim, and the panel's design purpose is that each *mechanism × severity*
combination is answered. Because the choice is consequential, the event-weighted
mean and the per-cell prevalence are reported next to it, not hidden.

**Every metric names its mask.** AUPRC/AUROC are computed over scorable bins
(``finite score``), not over all bins; cycle-level scores are aggregations of the
same masked bin scores. Reporting a metric whose denominator silently included
unobserved bins would make "the model that predicts nothing" look excellent.

Unscorable states return ``nan`` with a ``reason`` string that the report prints
verbatim. A zero would be a result; ``nan`` is an admission.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:                                  # optional, for the repo's existing pr_auc helper
    from ssl_anomaly.evaluation.metrics import pr_auc as _pr_auc
except Exception:                     # pragma: no cover - the local fallback is used
    _pr_auc = None


def ap(score: np.ndarray, y: np.ndarray) -> float:
    """Average precision (sklearn's AUPRC definition, step-interpolated)."""
    s = np.asarray(score, dtype=np.float64).ravel()
    yy = np.asarray(y).ravel().astype(int)
    ok = np.isfinite(s)
    s, yy = s[ok], yy[ok]
    if yy.size == 0 or yy.sum() == 0 or yy.sum() == yy.size:
        return float("nan")
    try:
        from sklearn.metrics import average_precision_score

        return float(average_precision_score(yy, s))
    except Exception:                                    # pragma: no cover
        return _ap_manual(s, yy)


def _ap_manual(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(-score, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, yy.size + 1)
    rec = tp / max(1, int(yy.sum()))
    return float(np.sum(np.diff(np.concatenate([[0.0], rec])) * prec))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """AUROC by rank statistic (ties averaged) — no sklearn dependency on the hot path."""
    s = np.asarray(score, dtype=np.float64).ravel()
    yy = np.asarray(y).ravel().astype(int)
    ok = np.isfinite(s)
    s, yy = s[ok], yy[ok]
    n1, n0 = int(yy.sum()), int((1 - yy).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata

    r = rankdata(s)
    return float((r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def top_k_recall(score: np.ndarray, y: np.ndarray, k: int = 10) -> dict:
    """Recall@k per cycle: the operational reading ("did the alert surface make the event top-10")."""
    s = np.asarray(score, dtype=np.float64)
    yy = np.asarray(y).astype(bool)
    if s.ndim != 2 or yy.shape != s.shape:
        raise ValueError("score and y must be (N, T) and aligned")
    hits = tot = 0
    per: list[float] = []
    for i in range(s.shape[0]):
        row = s[i]
        pos = yy[i]
        if not pos.any():
            continue
        fin = np.isfinite(row)
        if not (fin & pos).any():
            continue
        order = np.argsort(-np.where(fin, row, -np.inf), kind="mergesort")[: int(k)]
        got = float(pos[order].mean())
        per.append(got)
        hits += int(pos[order].sum())
        tot += int(pos.sum())
    return {"recall_at_k": float(np.mean(per)) if per else float("nan"),
            "micro_recall_at_k": float(hits / tot) if tot else float("nan"),
            "k": int(k), "n_events": len(per)}


@dataclass
class CellMetrics:
    """Metrics for one (family, severity) cell at one aggregation level."""

    cell: str
    family: str
    severity: str
    level: str
    auprc: float
    auroc: float
    n_scored_bins: int
    n_anomaly_bins: int
    n_cycles: int
    prevalence: float
    n_anomaly_cycles: int
    f1_at_budget: float | None = None
    f1: float | None = None
    brier: float | None = None
    ece: float | None = None
    median_delay_days: float | None = None
    recall_at_10: float | None = None
    aux: dict = field(default_factory=dict)

    @property
    def scorable(self) -> bool:
        return bool(np.isfinite(self.auprc))

    def to_dict(self) -> dict:
        return {
            "cell": self.cell, "family": self.family, "severity": self.severity, "level": self.level,
            "auprc": None if not np.isfinite(self.auprc) else round(float(self.auprc), 5),
            "auroc": None if not np.isfinite(self.auroc) else round(float(self.auroc), 5),
            "n_scored_bins": self.n_scored_bins, "n_anomaly_bins": self.n_anomaly_bins,
            "n_cycles": self.n_cycles, "prevalence": round(self.prevalence, 5),
            "n_anomaly_cycles": self.n_anomaly_cycles,
            "f1_at_budget": self.f1_at_budget, "brier": self.brier, "ece": self.ece,
            "median_delay_days": self.median_delay_days, "recall_at_10": self.recall_at_10,
            "aux": dict(self.aux),
        }


def cell_metrics(
    *, cell: str, family: str, severity: str, level: str,
    scores_bin: np.ndarray, labels_bin: np.ndarray, scorable_bin: np.ndarray,
    scores_cycle: np.ndarray | None = None, labels_cycle: np.ndarray | None = None,
    probs_cycle: np.ndarray | None = None, delay_days: np.ndarray | None = None,
    alarm_budget: float = 0.10, farm_ids: np.ndarray | None = None, min_bins_per_farm: int = 8,
    dev_threshold: float | None = None,
) -> CellMetrics:
    """One cell's numbers, with the denominators that make them interpretable.

    ``labels_bin`` is the injected-event mask, so a cycle with no event contributes
    to the negatives; ``scorable_bin`` is the intersection of "observed" and
    "produced a finite score", and it is reported so a reader can see that a cell
    with 12 scorable bins is not comparable to one with 12,000.
    """
    lb = np.asarray(labels_bin, dtype=bool)
    sc = np.asarray(scorable_bin, dtype=bool) & np.isfinite(np.asarray(scores_bin, dtype=np.float64))
    s = np.where(sc, np.asarray(scores_bin, dtype=np.float64), np.nan)
    a = ap(s[sc], lb[sc].astype(int)) if sc.any() else float("nan")
    u = auc(s[sc], lb[sc].astype(int)) if sc.any() else float("nan")
    if not (np.isfinite(a) and np.isfinite(u)) and _pr_auc is not None and sc.sum() > 8:
        try:
            a = float(_pr_auc(lb[sc].astype(int), s[sc])[0])
        except Exception:
            pass
    n_cyc = int(lb.shape[0])
    cm = CellMetrics(
        cell=cell, family=family, severity=severity, level=level,
        auprc=a, auroc=u,
        n_scored_bins=int(sc.sum()), n_anomaly_bins=int((lb & sc).sum()),
        n_cycles=n_cyc, prevalence=float(lb.sum() / max(1, lb.size)),
        n_anomaly_cycles=int(lb.any(axis=1).sum()) if lb.ndim == 2 else 0,
    )
    if farm_ids is not None:
        # Per-farm AUPRC is what the statistical section bootstraps over: resampling
        # individual cycles would treat 400 correlated fields as 400 independent
        # observations and shrink every interval. Farms with no positive bin or too few
        # scorable bins are absent from the dict, not zero — a zero would be a
        # fabricated score for a farm that could not be scored.
        fid = np.asarray(farm_ids).ravel()
        per_farm: dict[str, float] = {}
        for f in np.unique(fid):
            sel = fid == f
            sc_f = sc[sel]
            if int(sc_f.sum()) < int(min_bins_per_farm):
                continue
            y = lb[sel][sc_f].astype(int)
            if 0 < int(y.sum()) < int(y.size):
                v = ap(s[sel][sc_f], y)
                if np.isfinite(v):
                    per_farm[str(int(f))] = round(float(v), 5)
        cm.aux["per_farm"] = per_farm
        cm.aux["n_farms_scored"] = len(per_farm)
        cm.aux["min_bins_per_farm"] = int(min_bins_per_farm)
    cyc_s = np.asarray(scores_cycle, dtype=np.float64) if scores_cycle is not None else np.nanmean(
        np.where(sc, s, np.nan), axis=1)
    cyc_y = (np.asarray(labels_cycle, dtype=bool) if labels_cycle is not None else lb.any(axis=1)).astype(int)
    ok = np.isfinite(cyc_s)
    if ok.sum() > 4 and 0 < int(cyc_y[ok].sum()) < int(ok.sum()):
        cm.aux["cycle_auprc"] = round(ap(cyc_s[ok], cyc_y[ok]), 5)
        cm.aux["cycle_auroc"] = round(auc(cyc_s[ok], cyc_y[ok]), 5)
        thr = float(np.quantile(cyc_s[ok], 1.0 - float(alarm_budget)))
        pred = (cyc_s[ok] >= thr).astype(int)
        tp = int(((pred == 1) & (cyc_y[ok] == 1)).sum())
        fp = int(((pred == 1) & (cyc_y[ok] == 0)).sum())
        fn = int(((pred == 0) & (cyc_y[ok] == 1)).sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        cm.f1_at_budget = round(2 * prec * rec / max(prec + rec, 1e-12), 5)
        cm.aux["f1"] = {"precision": round(prec, 5), "recall": round(rec, 5),
                        "tp": tp, "fp": fp, "fn": fn, "alarm_budget": float(alarm_budget),
                        "achieved_far": round(fp / max(1, int((cyc_y[ok] == 0).sum())), 5)}
        if dev_threshold is not None and np.isfinite(float(dev_threshold)):
            # ``evaluation.threshold_rule`` is the pre-registered F1 rule: max-F1 over a
            # 101-point quantile grid on dev only. The budget-matched F1 above stays in the
            # record because it is the equal-false-alarm comparison; the two answer
            # different questions and neither is quietly substituted for the other.
            t = float(dev_threshold)
            p2 = (cyc_s[ok] >= t).astype(int)
            tp2 = int(((p2 == 1) & (cyc_y[ok] == 1)).sum())
            fp2 = int(((p2 == 1) & (cyc_y[ok] == 0)).sum())
            fn2 = int(((p2 == 0) & (cyc_y[ok] == 1)).sum())
            pr2, rc2 = tp2 / max(1, tp2 + fp2), tp2 / max(1, tp2 + fn2)
            cm.f1 = round(2 * pr2 * rc2 / max(pr2 + rc2, 1e-12), 5)
            cm.aux["f1_dev_rule"] = {"threshold": round(t, 6), "precision": round(pr2, 5),
                                     "recall": round(rc2, 5), "tp": tp2, "fp": fp2, "fn": fn2,
                                     "grid": 101, "fitted_on": "dev",
                                     "rule": "max-F1 over a 101-point quantile grid on dev only"}
        else:
            cm.aux["f1_dev_rule"] = {"available": False,
                                     "reason": "no dev threshold supplied (empty dev partition?)"}
        r = top_k_recall(np.where(np.isfinite(s), s, -np.inf), lb, k=10)
        cm.recall_at_10 = round(r["recall_at_k"], 5) if np.isfinite(r["recall_at_k"]) else None
    if probs_cycle is not None and cyc_y is not None:
        p = np.asarray(probs_cycle, dtype=np.float64).ravel()[ok]
        yv = cyc_y[ok].astype(float)
        if p.size and np.isfinite(p).all():
            cm.brier = round(float(np.mean((p - yv) ** 2)), 6)
            from ssl_anomaly.benchmarking.calibration import expected_calibration_error

            cm.ece = round(expected_calibration_error(p, yv)["ece"], 6)
    if delay_days is not None:
        d = np.asarray(delay_days, dtype=np.float64)
        d = d[np.isfinite(d)]
        if d.size:
            cm.median_delay_days = round(float(np.median(d)), 2)
            cm.aux["delay"] = {"n": int(d.size), "mean": round(float(d.mean()), 2),
                               "q25": round(float(np.quantile(d, 0.25)), 2),
                               "q75": round(float(np.quantile(d, 0.75)), 2)}
    return cm


def macro_summary(cells: list[CellMetrics], *, metric: str = "auprc") -> dict:
    """Equal-cell-weight macro mean + the weighted alternative, always side by side."""
    vals = np.array([getattr(c, metric) for c in cells], dtype=np.float64)
    ok = np.isfinite(vals)
    if not ok.any():
        return {"macro": float("nan"), "n_cells_used": 0, "n_cells": len(cells),
                "reason": "no cell had a scorable metric"}
    w = np.array([max(1, c.n_anomaly_bins) for c in cells], dtype=np.float64)[ok]
    return {
        "macro": round(float(vals[ok].mean()), 5),
        "macro_sd_across_cells": round(float(vals[ok].std(ddof=1)), 5) if ok.sum() > 1 else None,
        "event_weighted": round(float(np.average(vals[ok], weights=w)), 5),
        "n_cells_used": int(ok.sum()), "n_cells": len(cells),
        "cells_missing": [c.cell for c in cells if not np.isfinite(getattr(c, metric))],
        "prevalence_range": [round(float(min(c.prevalence for c in cells)), 5),
                            round(float(max(c.prevalence for c in cells)), 5)],
        "aggregation": "equal cell weight (pre-registered); event-weighted shown for contrast",
    }


def per_seed_table(cells_by_seed: dict[int, list[CellMetrics]], *, metric: str = "auprc") -> dict:
    """Seed spread per cell: the raw material for the variance decomposition.

    ``across_seed_sd`` for a cell with one seed is ``None`` and says so — a spread
    estimated from one draw would be the most misleading number in the report.
    """
    names = sorted({c.cell for cells in cells_by_seed.values() for c in cells})
    rows: list[dict] = []
    for nm in names:
        vals = []
        for seed, cells in cells_by_seed.items():
            hit = [c for c in cells if c.cell == nm]
            vals.append(float(getattr(hit[0], metric)) if hit else float("nan"))
        v = np.asarray(vals, dtype=np.float64)
        ok = np.isfinite(v)
        rows.append({
            "cell": nm, "per_seed": {int(s): (None if not np.isfinite(x) else round(float(x), 5))
                                      for s, x in zip(sorted(cells_by_seed), v)},
            "mean": round(float(v[ok].mean()), 5) if ok.any() else None,
            "across_seed_sd": round(float(v[ok].std(ddof=1)), 5) if ok.sum() > 1 else None,
            "n_seeds": int(ok.sum()),
        })
    return {"metric": metric, "rows": rows,
            "mean_across_seed_sd": round(float(np.nanmean([r["across_seed_sd"] for r in rows
                                                            if r["across_seed_sd"] is not None])), 5)
            if any(r["across_seed_sd"] is not None for r in rows) else None}

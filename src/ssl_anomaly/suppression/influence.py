"""Prediction-influence audit for S1/S3 (task 2.9).

The suppression arms reweight the loss, so a claim of the form "S1 improves
specificity" is only meaningful if the weights actually reach the predictions. This
module answers two questions the reviewers of a suppression benchmark always ask:

1. **Do the weights matter?** Hold the parameters fixed, score the same bins with
   ``w = 1`` and with ``w = w_fitted``, and measure the rank change
   (Spearman/Kendall) plus the shift in the decision-relevant tail. If the fitted
   weights are ~1 everywhere, the "improvement" is the extra epochs, not the
   suppression — and the report must say so.

2. **Which bins were suppressed?** Report the distribution of weights over true
   anomalies vs the bulk. A suppressor that down-weights real events is not
   denoising, it is deleting the signal — the most likely failure mode for S3 on
   a 10 % contaminated panel, and the reason this audit is a phase-2 task rather
   than a nice-to-have.

``audit_influence`` returns the numbers; :func:`influence_verdict` turns them into
the one-line claim the report is allowed to make (or refuses to make).
"""

from __future__ import annotations

import numpy as np
from scipy import stats

TAIL_Q = 0.95


def _rank_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Spearman ρ and Kendall τ-b on the finite intersection of two score vectors."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return (float("nan"), float("nan"))
    rho = float(stats.spearmanr(x[ok], y[ok]).statistic)
    tau = float(stats.kendalltau(x[ok], y[ok], variant="b").statistic)
    return (rho, tau)


def tail_shift(unit: np.ndarray, weighted: np.ndarray, q: float = TAIL_Q) -> dict:
    """How much the decision-relevant tail moved when the weights were switched on."""
    u = np.asarray(unit, dtype=np.float64).ravel()
    w = np.asarray(weighted, dtype=np.float64).ravel()
    ok = np.isfinite(u) & np.isfinite(w)
    if ok.sum() < 8:
        return {"available": False, "reason": f"only {int(ok.sum())} paired finite scores"}
    uu, ww = u[ok], w[ok]
    tu = float(np.quantile(uu, q))
    tw = float(np.quantile(ww, q))
    scale = float(np.median(np.abs(uu - np.median(uu))) * 1.4826) or 1.0
    return {
        "available": True,
        "quantile": float(q),
        "tail_unit": tu,
        "tail_weighted": tw,
        "tail_shift_robust_sd": (tw - tu) / max(scale, 1e-9),
        "n_tail_flags_unit": int((uu >= tu).sum()),
        "n_tail_flags_weighted": int((ww >= tw).sum()),
        "jaccard_tail": float(
            len(set(np.flatnonzero(uu >= tu).tolist()) & set(np.flatnonzero(ww >= tw).tolist()))
            / max(1, len(set(np.flatnonzero(uu >= tu).tolist()) | set(np.flatnonzero(ww >= tw).tolist())))
        ),
    }


def weight_distribution(
    weights: np.ndarray, labels: np.ndarray | None = None
) -> dict:
    """Quantiles of the fitted weights, split by ground truth where available.

    ``suppressed_anomaly_frac`` is the number a reviewer looks at first: the share
    of true-anomaly bins whose weight fell below the bulk's median. A suppressor
    that suppresses anomalies is measuring the wrong thing.
    """
    w = np.asarray(weights, dtype=np.float64).ravel()
    ok = np.isfinite(w)
    if not ok.any():
        return {"available": False, "reason": "no finite weights (S0 has none by construction)"}
    v = w[ok]
    out = {
        "available": True,
        "n": int(v.size),
        "min": float(v.min()),
        "q05": float(np.quantile(v, 0.05)),
        "median": float(np.median(v)),
        "q95": float(np.quantile(v, 0.95)),
        "max": float(v.max()),
        "mean": float(v.mean()),
        "frac_below_half": float((v < 0.5).mean()),
    }
    if labels is not None:
        lab = np.asarray(labels).ravel()[ok].astype(bool)
        if lab.any() and (~lab).any():
            bulk_med = float(np.median(v[~lab]))
            out.update(
                median_bulk=float(np.median(v[~lab])),
                median_anomaly=float(np.median(v[lab])),
                suppressed_anomaly_frac=float((v[lab] < bulk_med).mean()),
                anomaly_over_bulk_ratio=float(
                    np.median(v[lab]) / max(np.median(v[~lab]), 1e-9)
                ),
            )
    return out


def audit_influence(
    *, scores_unit: np.ndarray, scores_weighted: np.ndarray, weights: np.ndarray | None,
    labels: np.ndarray | None = None, epoch_unit: int | None = None,
    epoch_weighted: int | None = None, threshold: float | None = None,
) -> dict:
    """Full audit: rank stability, tail movement, who got suppressed.

    ``epoch_*`` are recorded when the comparison holds the number of updates fixed;
    if they differ, the caller must say so, because then the "influence" is partly
    a longer optimisation. The audit does not silently accept that confound: it
    reports ``confounded_by_training_length``.
    """
    rho, tau = _rank_pair(scores_unit, scores_weighted)
    out = {
        "rank_correlation_spearman": rho,
        "rank_correlation_kendall": tau,
        "tail": tail_shift(scores_unit, scores_weighted),
        "weights": weight_distribution(np.ones_like(np.asarray(scores_unit, dtype=np.float64))
                                       if weights is None else weights, labels),
        "confounded_by_training_length": (
            epoch_unit is not None and epoch_weighted is not None and epoch_unit != epoch_weighted
        ),
        "epochs": {"unit": epoch_unit, "weighted": epoch_weighted},
    }
    if threshold is not None:
        u = np.asarray(scores_unit, dtype=np.float64)
        w = np.asarray(scores_weighted, dtype=np.float64)
        ok = np.isfinite(u) & np.isfinite(w)
        au, aw = (u[ok] >= threshold), (w[ok] >= threshold)
        out["flag_flip_rate"] = float((au != aw).mean()) if ok.any() else float("nan")
        out["n_flags_unit"] = int(au.sum())
        out["n_flags_weighted"] = int(aw.sum())
    return out


def influence_verdict(audit: dict, *, min_rank_gap: float = 0.02, min_tail_shift: float = 0.05,
                      max_suppressed_anomaly_frac: float = 0.35) -> dict:
    """What the audit licenses the report to claim — and what it forbids.

    Three outcomes, and the middle one is the honest default when weights barely
    bind: the suppression had no measurable influence, so S1/S3 rows must be read
    as "no difference", not as "no harm".
    """
    rho = audit.get("rank_correlation_spearman")
    tail = audit.get("tail") or {}
    w = audit.get("weights") or {}
    if rho is None or not np.isfinite(rho):
        return {"verdict": "insufficient_evidence",
                "line": "Influence audit: the weight vectors do not overlap on enough finite bins to rank.",
                "flags": ["not_enough_pairs"]}
    shift = abs(float(tail.get("tail_shift_robust_sd", 0.0) or 0.0))
    sup = w.get("suppressed_anomaly_frac")
    flags: list[str] = []
    if float(rho) > 1.0 - min_rank_gap:
        flags.append("ranks_effectively_unchanged")
    if shift < min_tail_shift:
        flags.append("tail_effectively_unchanged")
    if sup is not None and float(sup) > max_suppressed_anomaly_frac:
        flags.append("suppresses_true_anomalies")
    if flags == []:
        line = (f"Influence audit: S1/S3 weights change the ranking (ρ={rho:.3f}, τ={audit['rank_correlation_kendall']:.3f}) "
                f"and move the {tail.get('quantile', 0.95):.2f}-tail by {shift:.2f} robust SD.")
        verdict = "influence_confirmed"
    elif "suppresses_true_anomalies" in flags:
        verdict = "harmful_suppression"
        line = (f"Influence audit: the suppressor down-weights {float(sup):.0%} of true anomalies "
                f"(bulk median {w.get('median_bulk'):.3f} vs anomaly median {w.get('median_anomaly'):.3f}); "
                "any AUPRC gain from this arm is not evidence of denoising.")
    else:
        verdict = "no_measurable_influence"
        line = (f"Influence audit: fitted weights leave the ranking (ρ={rho:.3f}) and the tail "
                f"(Δ={shift:.2f} SD) essentially unchanged — read the S1/S3 rows as 'no difference', "
                "not as 'no harm'.")
    return {"verdict": verdict, "line": line, "flags": flags,
            "confounded_by_training_length": audit.get("confounded_by_training_length", False)}

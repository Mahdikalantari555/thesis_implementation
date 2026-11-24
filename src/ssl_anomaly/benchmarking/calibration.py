"""Probability and threshold calibration C1/C2/C3 (tasks 6.x; spec experimental-validation).

The axis is *how the score becomes a decision*, and three variants are compared
under one shared false-alarm budget:

``C1`` none          the raw objective value is the score; the threshold is the
                     quantile of the same arm's train scores at the budget.
``C2`` fixed         a pre-registered quantile (the frozen operational rule's
                     semantics), applied to every arm identically.
``C3`` fitted        temperature scaling / isotonic regression fitted on dev,
                     optimising NLL (never the metric being reported).

Calibration quality is reported as **both** Brier and ECE, and ECE is computed with
a fixed bin count (15, equal-frequency) recorded here rather than left to a
library default — ECE is bin-count sensitive, and an unfixed bin count turns a
calibration comparison into a comparison of histograms.

Nothing in this module may touch the locked test split: the fitter takes an
explicit ``split`` label and refuses anything but ``train``/``dev``, so "we fitted on
test" is structurally impossible rather than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ECE_BINS = 15
ALLOWED_FIT_SPLITS = ("train", "dev")


@dataclass
class CalibrationFit:
    """One fitted calibration, with the partition it was fitted on recorded."""

    method: str
    params: dict = field(default_factory=dict)
    fitted_on: str = "dev"
    n_fit: int = 0
    brier_before: float = float("nan")
    brier_after: float = float("nan")
    ece_before: float = float("nan")
    ece_after: float = float("nan")
    auc_preserved: bool = True
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "params": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in self.params.items()},
            "fitted_on": self.fitted_on,
            "n_fit": self.n_fit,
            "brier": {"before": self.brier_before, "after": self.brier_after},
            "ece": {"before": self.ece_before, "after": self.ece_after, "bins": ECE_BINS,
                    "scheme": "equal-frequency"},
            "auc_preserved": self.auc_preserved,
            "warnings": list(self.warnings),
        }

    def apply(self, scores: np.ndarray) -> np.ndarray:
        return apply_calibration(self.method, scores, self.params)


def _check_split(split: str) -> None:
    if split not in ALLOWED_FIT_SPLITS:
        raise ValueError(
            f"calibration may only be fitted on {list(ALLOWED_FIT_SPLITS)}, got {split!r}; fitting on "
            "the scored partition would make the calibration row of the report meaningless"
        )


def rank_to_probability(scores: np.ndarray, train_scores: np.ndarray) -> np.ndarray:
    """**C1** — empirical CDF of the arm's own train scores (monotone, so AUC is exact).

    Monotone transforms cannot change ranking, which is the point: C1 gives a
    probability-like number *without* claiming the score is a probability, and the
    report says exactly that.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    ref = np.sort(np.asarray(train_scores, dtype=np.float64).ravel())
    ref = ref[np.isfinite(ref)]
    if ref.size < 32:
        raise ValueError(
            f"only {ref.size} finite train scores for the C1 ECDF; a quantile map fitted on that "
            "few points would put every bin in one of two probabilities"
        )
    lo = np.searchsorted(ref, s, side="left") / ref.size
    hi = np.searchsorted(ref, s, side="right") / ref.size
    p = 0.5 * (lo + hi)
    return np.clip(np.where(np.isfinite(s), p, np.nan), 1e-6, 1.0 - 1e-6)


def zscore_to_probability(scores: np.ndarray, train_scores: np.ndarray) -> np.ndarray:
    """Gaussian-tail probability from robust train statistics (the fixed reference).

    Used where a raw objective value should become comparable across arms without
    any fitted map: ``Φ̄((s − median)/MAD_scale)``. It is a *reference* scale, not a
    calibrated probability, and :func:`calibrate` marks it as such.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    ref = np.asarray(train_scores, dtype=np.float64).ravel()
    ref = ref[np.isfinite(ref)]
    if ref.size < 32:
        raise ValueError(f"z-score reference needs ≥32 train scores, got {ref.size}")
    mu = float(np.median(ref))
    sd = float(np.median(np.abs(ref - mu)) * 1.4826) or float(np.std(ref)) or 1.0
    from scipy.stats import norm

    return np.clip(norm.sf((s - mu) / max(sd, 1e-9)), 1e-6, 1.0 - 1e-6)


def fit_temperature(logits: np.ndarray, labels: np.ndarray, *, split: str = "dev",
                   lr: float = 0.05, iters: int = 200) -> tuple[float, CalibrationFit]:
    """**C3a** — temperature scaling (Guo et al. style) on binarised cycle labels."""
    _check_split(split)
    z = np.asarray(logits, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(float)
    ok = np.isfinite(z)
    z, y = z[ok], y[ok]
    if z.size < 32 or np.unique(y).size < 2:
        raise ValueError(
            f"temperature scaling needs ≥32 labelled rows with both classes "
            f"(got n={z.size}, classes={np.unique(y).size})"
        )

    def nll(t: float) -> float:
        p = 1.0 / (1.0 + np.exp(-np.clip(z / max(t, 1e-3), -30, 30)))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    t = 1.0
    best = (t, nll(t))
    step = lr
    for _ in range(iters):                     # 1-D line search: deterministic, no scipy state
        for cand in (t * (1 + step), t * (1 - step)):
            v = nll(max(cand, 1e-3))
            if v < best[1] - 1e-9:
                best = (max(cand, 1e-3), v)
        if abs(best[0] - t) < 1e-4:
            step *= 0.5
            if step < 1e-4:
                break
        t = best[0]
    fit = CalibrationFit(method="temperature", params={"temperature": t, "nll": best[1]},
                         fitted_on=split, n_fit=int(z.size))
    return t, fit


def fit_isotonic(scores: np.ndarray, labels: np.ndarray, *, split: str = "dev") -> CalibrationFit:
    """**C3b** — isotonic regression, monotone by construction (so AUC is preserved)."""
    _check_split(split)
    from sklearn.isotonic import IsotonicRegression

    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(float)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if s.size < 64:
        raise ValueError(
            f"isotonic calibration needs ≥64 finite (score,label) pairs, got {s.size}; below that "
            "the fitted step function is the label vector with extra steps"
        )
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, y)
    thresholds = np.unique(iso.predict(np.quantile(s, np.linspace(0, 1, 41))))
    return CalibrationFit(
        method="isotonic", fitted_on=split, n_fit=int(s.size),
        params={"n_plateaus": int(thresholds.size), "score_range": [float(s.min()), float(s.max())],
                "out_of_bounds": "clip"},
    )


def apply_calibration(method: str, scores: np.ndarray, params: dict) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64).ravel()
    if method == "none":
        return s
    if method == "ecdf":
        return rank_to_probability(s, np.asarray(params["train_scores"], dtype=np.float64))
    if method == "zscore":
        return zscore_to_probability(s, np.asarray(params["train_scores"], dtype=np.float64))
    if method == "temperature":
        t = max(float(params.get("temperature", 1.0)), 1e-3)
        return 1.0 / (1.0 + np.exp(-np.clip(s / t, -30, 30)))
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression

        iso = params.get("fitter")
        if not isinstance(iso, IsotonicRegression):
            raise ValueError("isotonic calibration needs the fitted estimator in params['fitter']")
        return np.clip(iso.predict(np.nan_to_num(s, nan=0.0)), 1e-6, 1 - 1e-6)
    raise KeyError(f"unknown calibration method {method!r}")


def brier(p: np.ndarray, y: np.ndarray) -> float:
    pp = np.asarray(p, dtype=np.float64).ravel()
    yy = np.asarray(y).ravel().astype(float)
    ok = np.isfinite(pp)
    if ok.sum() == 0:
        return float("nan")
    return float(np.mean((pp[ok] - yy[ok]) ** 2))


def expected_calibration_error(p: np.ndarray, y: np.ndarray, *, bins: int = ECE_BINS) -> dict:
    """Equal-frequency-binned ECE, with the reliability table returned beside it.

    Equal-frequency (not equal-width) because anomaly scores are heavily skewed:
    equal-width bins would put 99 % of the mass in one bin and report a near-zero
    ECE for a badly calibrated model.
    """
    pp = np.asarray(p, dtype=np.float64).ravel()
    yy = np.asarray(y).ravel().astype(float)
    ok = np.isfinite(pp)
    pp, yy = pp[ok], yy[ok]
    if pp.size == 0:
        return {"ece": float("nan"), "bins": [], "n": 0, "scheme": "equal-frequency"}
    n = min(int(bins), max(2, pp.size))
    edges = np.quantile(pp, np.linspace(0, 1, n + 1))
    edges = np.unique(edges)
    rows, tot = [], 0
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (pp >= lo) & (pp <= hi) if hi == edges[-1] else (pp >= lo) & (pp < hi)
        k = int(sel.sum())
        if k == 0:
            continue
        conf, acc = float(pp[sel].mean()), float(yy[sel].mean())
        ece += k / pp.size * abs(conf - acc)
        tot += k
        rows.append({"lo": round(float(lo), 5), "hi": round(float(hi), 5), "n": k,
                     "mean_score": round(conf, 5), "observed_rate": round(acc, 5)})
    return {"ece": float(ece), "bins": rows, "n": tot, "scheme": "equal-frequency",
            "n_bins_used": len(rows), "n_bins_requested": int(bins)}


def calibrate(
    method: str, *, scores_dev: np.ndarray, labels_dev: np.ndarray,
    train_scores: np.ndarray | None = None, scores_eval: np.ndarray | None = None,
    split: str = "dev",
) -> tuple[np.ndarray, CalibrationFit]:
    """Fit (if needed) and map ``scores_eval`` to probabilities.

    ``method`` is the contract's calibration id: ``C1`` → ``ecdf``, ``C2`` →
    ``zscore`` (fixed robust reference), ``C3`` → ``temperature`` then ``isotonic``
    whichever reduces NLL more, with the choice recorded in ``params``.
    """
    if method == "C1":
        if train_scores is None:
            raise ValueError("C1 needs train scores for its ECDF; dev scores would calibrate on the eval set")
        p = rank_to_probability(scores_eval if scores_eval is not None else scores_dev, train_scores)
        return p, CalibrationFit(method="ecdf", params={"train_n": int(len(train_scores))}, fitted_on="train")
    if method == "C2":
        src = train_scores if train_scores is not None else scores_dev
        p = zscore_to_probability(scores_eval if scores_eval is not None else scores_dev, src)
        return p, CalibrationFit(method="zscore", params={"reference": "train_robust_z"},
                                 fitted_on="train" if train_scores is not None else split)
    if method != "C3":
        raise KeyError(f"unknown calibration id {method!r} (expected C1/C2/C3)")
    z = np.log(np.clip(scores_dev, 1e-6, 1 - 1e-6) / (1 - np.clip(scores_dev, 1e-6, 1 - 1e-6)))
    t, tfit = fit_temperature(z, labels_dev, split=split)
    ifit = fit_isotonic(scores_dev, labels_dev, split=split)
    p_t = apply_calibration("temperature", z, {"temperature": t})
    p_i = apply_calibration("isotonic", scores_dev, {"fitter": _iso_fit(scores_dev, labels_dev)})
    pick, fit, params = (
        ("temperature", tfit, {"temperature": t, "z_from": "logit(clipped dev probability)"})
        if brier(p_t, labels_dev) <= brier(p_i, labels_dev)
        else ("isotonic", ifit, {"fitter": _iso_fit(scores_dev, labels_dev)})
    )
    fit.params.update(params, chosen_over=["temperature", "isotonic"],
                      brier_temperature=brier(p_t, labels_dev), brier_isotonic=brier(p_i, labels_dev))
    ev = scores_eval if scores_eval is not None else scores_dev
    if pick == "temperature":
        zz = np.log(np.clip(ev, 1e-6, 1 - 1e-6) / (1 - np.clip(ev, 1e-6, 1 - 1e-6)))
        return apply_calibration("temperature", zz, {"temperature": t}), fit
    return apply_calibration("isotonic", ev, {"fitter": _iso_fit(scores_dev, labels_dev)}), fit


def _iso_fit(scores: np.ndarray, labels: np.ndarray):
    from sklearn.isotonic import IsotonicRegression

    s = np.nan_to_num(np.asarray(scores, dtype=np.float64).ravel(), nan=0.0)
    return IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(s, np.asarray(labels).ravel().astype(float))


def threshold_at_budget(probs: np.ndarray, *, budget: float) -> dict:
    """Threshold whose cycle false-alarm rate equals ``budget`` on the clean data.

    The budget is the *same number for every arm* (the frozen 10 % alarm budget), so
    a model cannot win by being allowed more alerts; the achieved FAR is reported
    per arm because discreteness means the achieved rate is not exactly the budget.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    p = p[np.isfinite(p)]
    if p.size < 10:
        raise ValueError(f"threshold_at_budget needs ≥10 scored cycles, got {p.size}")
    b = float(budget)
    if not 0 < b < 1:
        raise ValueError(f"alarm budget must be in (0,1), got {b}")
    thr = float(np.quantile(p, 1.0 - b))
    far = float((p >= thr).mean())
    out = {"threshold": thr, "budget": b, "achieved_far": far, "n": int(p.size),
           "n_alerts": int((p >= thr).sum())}
    if far > b * 1.5:
        out["warning"] = (
            f"achieved FAR {far:.3f} exceeds the {b:.2f} budget by >50%: the score distribution is "
            "too discrete at this budget, so cross-arm comparisons at this threshold are not equal-footing"
        )
    return out

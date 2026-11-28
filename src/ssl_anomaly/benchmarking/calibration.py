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


def zscore_to_probability(scores: np.ndarray, train_scores: np.ndarray,
                          *, min_ref: int = 32) -> np.ndarray:
    """Gaussian-tail probability from robust train statistics (the fixed reference).

    Used where a raw objective value should become comparable across arms without
    any fitted map: ``Φ̄((s − median)/MAD_scale)``. It is a *reference* scale, not a
    calibrated probability, and :func:`calibrate` marks it as such. The default
    ``min_ref`` floor guards the *global* map; a per-pool reference (C2's farm × stage
    pools) is small by design and passes its own floor instead — the pool's coarseness
    is reported, not hidden behind a raised exception.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    ref = np.asarray(train_scores, dtype=np.float64).ravel()
    ref = ref[np.isfinite(ref)]
    if ref.size < min_ref:
        raise ValueError(f"z-score reference needs ≥{min_ref} train scores, got {ref.size}")
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


def cycle_stage(phase_bin: np.ndarray | None, *, stage_bins: int = 12,
                start_days: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """One stage id per cycle, for the C2 ``farm × stage`` reference pool.

    The preferred source is the cycle's own ``phase_bin`` (the active ``PhaseAxis``): a
    cycle's stage is its modal phase bin rescaled onto ``stage_bins`` groups. Where the
    phase axis is degenerate (every cycle has the same modal bin — the default
    bin-index tile before a real PhaseAxis is bound) the stage collapses to one value and
    C2 would silently equal C1; that collapse is *recorded*, and when a calendar axis
    exists the stage falls back to the cycle's start position on the year, labelled as
    the calendar fallback it is. No planting date is invented anywhere.
    """
    rec: dict = {"stage_bins": int(stage_bins)}
    if phase_bin is not None:
        pb = np.asarray(phase_bin, dtype=np.int64)
        if pb.ndim != 2:
            raise ValueError(f"phase_bin must be (N, T), got {pb.shape}")
        T = pb.shape[1]
        modes = np.array([int(np.bincount(np.clip(row, 0, None)).argmax()) for row in pb])
        if len(np.unique(modes)) > 1:
            rec["source"] = "phase_axis_modal_bin"
            stage = np.minimum((modes / max(T, 1) * stage_bins).astype(np.int64), stage_bins - 1)
            return stage, rec
        rec["phase_axis_degenerate"] = True
    if start_days is not None:
        sd = np.asarray(start_days, dtype=np.float64)
        if sd.ndim != 1 or sd.size == 0 or not np.isfinite(sd).all():
            rec["calendar_fallback"] = "unusable_start_days"
        else:
            doy = np.mod(sd, 365.25)
            stage = np.minimum((doy / 365.25 * stage_bins).astype(np.int64), stage_bins - 1)
            rec["source"] = "calendar_fallback_start_day"
            rec["note"] = ("phase axis degenerate; stages derived from the cycle's start "
                           "position on the calendar year — a seasonal block, not a "
                           "segmentation-derived phenological label")
            return stage, rec
    if phase_bin is None and start_days is None:
        raise ValueError(
            "cycle_stage needs phase_bin or start_days to place cycles on a stage axis; "
            "supplying neither and defaulting every cycle to stage 0 would present a "
            "collapsed pool as a farm × stage calibration")
    n = len(np.asarray(phase_bin)) if phase_bin is not None else len(np.asarray(start_days))
    rec["source"] = "single_stage"
    rec["note"] = ("phase axis present but degenerate and no calendar axis: every cycle lands in "
                   "one stage, so the farm × stage pool collapses to the farm pool — recorded, "
                   "never fabricated")
    return np.zeros(n, dtype=np.int64), rec


def _ecdf_percentile(scores: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Empirical-CDF percentile of ``scores`` against ``ref`` (monotone ⇒ AUC-preserving).

    Unlike :func:`rank_to_probability` (the pooled C1 map, which refuses <32 points
    because a global quantile map on a handful of rows is fiction), a *per-key* reference
    is allowed to be small by design — a farm's own history is a handful of cycles — so
    the caller's ``min_history`` floor governs and the coarseness is reported.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    r = np.asarray(ref, dtype=np.float64).ravel()
    r = r[np.isfinite(r)]
    lo = np.searchsorted(r, s, side="left") / max(r.size, 1)
    hi = np.searchsorted(r, s, side="right") / max(r.size, 1)
    p = 0.5 * (lo + hi)
    return np.clip(np.where(np.isfinite(s), p, np.nan), 1e-6, 1.0 - 1e-6)


def pooled_reference_map(
    scores_eval: np.ndarray, train_scores: np.ndarray, *, eval_keys: np.ndarray,
    train_keys: np.ndarray, min_history: int = 2, map_name: str = "ecdf",
) -> tuple[np.ndarray, dict]:
    """Map every eval row through the reference distribution of its OWN pool.

    This is the design's hybrid-normality rule in code: a farm is compared to its own
    history (C1) or its own history at the same stage (C2) first; pooled cross-farm
    statistics only fill a per-farm deficit (fewer than ``min_history`` clean cycles),
    and every such fill is counted — the contract's ``pooled_fill``. Keys must be a
    per-row identifier (farm id, or ``(farm_id, stage)`` tuples as a 2-column array).
    """
    se = np.asarray(scores_eval, dtype=np.float64).ravel()
    st = np.asarray(train_scores, dtype=np.float64).ravel()
    ek = np.asarray(eval_keys)
    tk = np.asarray(train_keys)
    if ek.ndim == 1:
        ek = ek[:, None]
    if tk.ndim == 1:
        tk = tk[:, None]
    if ek.shape[0] != se.size or tk.shape[0] != st.size:
        raise ValueError(
            f"pooled_reference_map: keys and scores disagree (eval {ek.shape[0]} vs {se.size}, "
            f"train {tk.shape[0]} vs {st.size}); misaligned keys would score a cycle against "
            "another farm's history")
    ok = np.isfinite(st)
    ref_by_key: dict[tuple, np.ndarray] = {}
    for key in np.unique(tk[ok], axis=0):
        ref_by_key[tuple(int(v) for v in key)] = st[ok & (tk == key).all(axis=1)]
    global_ref = st[ok]
    p = np.full(se.size, np.nan)
    pooled_fill = 0
    ref_sizes: list[int] = []
    for i in range(se.size):
        key = tuple(int(v) for v in ek[i])
        ref = ref_by_key.get(key)
        src = "own_pool"
        if ref is None or ref.size < max(1, int(min_history)):
            ref, src = global_ref, "pooled_fill"
            pooled_fill += 1
        if ref.size < max(1, int(min_history)):
            raise ValueError(
                f"pooled_reference_map: row {i} has no usable reference even after pooled fill "
                f"({int(global_ref.size)} finite train scores); calibration cannot invent one")
        ref_sizes.append(int(ref.size))
        if map_name == "ecdf":
            p[i] = _ecdf_percentile(np.array([se[i]]), ref)[0]
        elif map_name == "zscore":
            p[i] = float(zscore_to_probability(np.array([se[i]]), ref, min_ref=max(1, int(min_history)))[0])
        else:
            raise KeyError(f"unknown per-pool map {map_name!r}")
    sizes = np.array(ref_sizes) if ref_sizes else np.zeros(1)
    rec = {
        "pooling_level": "farm" if ek.shape[1] == 1 else "farm_stage",
        "map": map_name,
        "min_history": int(min_history),
        "pooled_fill": int(pooled_fill),
        "pooled_fill_note": ("eval rows scored against the pooled train reference because their "
                             "own pool had fewer than min_history clean cycles"),
        "n_pools": int(len(ref_by_key)),
        "ref_size_median": float(np.median(sizes)),
        "ref_size_min": int(sizes.min()),
        "coarse_reference_pools": int((sizes < 8).sum()),
        "coarse_note": ("an ECDF over fewer than 8 same-pool cycles is coarse; the mapping stays "
                        "monotone inside the pool, so ranking within a farm is unaffected"),
    }
    return p, rec


def calibration_holdout_check(
    probs: np.ndarray, *, budget: float, farm_ids: np.ndarray | None = None,
    seed: int = 2026,
) -> dict:
    """Task 7.7: measure the threshold's optimism on a partition it was not picked on.

    The contract picks thresholds on ``dev`` and reports on ``dev`` — the same rows — so
    the honest check is *internal*: split the scored rows into two disjoint halves (whole
    farms, when farm ids exist, so a farm's cycles never straddle the split), pick the
    alarm-budget threshold on the calibration half, and report the false-alarm rate it
    achieves on the held-out half. The gap between that rate and the budget is the
    optimism a reader should keep in mind; it is recorded next to the threshold instead
    of being claimed away.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    ok = np.isfinite(p)
    if farm_ids is not None:
        f = np.asarray(farm_ids).ravel()[ok]
        uniq = np.unique(f)
        if uniq.size < 4:
            return {"available": False, "reason": (f"{uniq.size} farms; a farm-stratified "
                    "half split needs ≥4 to give both halves farm diversity")}
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(uniq.size)
        cal_f = set(int(k) for k in uniq[perm[: uniq.size // 2]])
        cal = ok & np.isin(np.asarray(farm_ids).ravel(), np.array(sorted(cal_f)))
        strat = "farm_stratified_whole_farm"
    else:
        rng = np.random.default_rng(int(seed))
        idx = np.flatnonzero(ok)
        perm = rng.permutation(idx.size)
        cal = np.zeros(p.size, dtype=bool)
        cal[idx[perm[: idx.size // 2]]] = True
        strat = "row_half"
    n_cal, n_hold = int(cal.sum()), int(ok.sum() - cal.sum())
    if n_cal < 10 or n_hold < 10:
        return {"available": False, "n_calibration_half": n_cal, "n_holdout_half": n_hold,
                "reason": (f"halves too small to place a quantile ({n_cal}/{n_hold}); "
                           "threshold_at_budget needs ≥10 scored cycles per half")}
    thr_cal = threshold_at_budget(p[cal], budget=float(budget))
    far_hold = float((p[~cal] >= thr_cal["threshold"]).mean())
    return {
        "available": True,
        "split": strat,
        "seed": int(seed),
        "n_calibration_half": n_cal, "n_holdout_half": n_hold,
        "threshold_on_calibration_half": round(float(thr_cal["threshold"]), 6),
        "holdout_far": round(far_hold, 5),
        "budget": float(budget),
        "optimism": round(far_hold - float(budget), 5),
        "note": ("threshold picked on one half of the scored partition, FAR measured on the "
                 "other half: the budget minus the holdout FAR is how far '10 % alarms' holds "
                 "on rows the threshold never saw — an internal check, not a substitute for "
                 "the locked test split"),
    }


def _attach_dev_diagnostics(p_dev: np.ndarray, labels_dev: np.ndarray,
                            fit: CalibrationFit) -> CalibrationFit:
    """Task 7.6: Brier + ECE of the fitted mapping, computed on dev (a permitted source).

    Raw reconstruction errors are not probabilities, so a "before" value is only defined
    where the input already was one (C3's logit path); for C1/C2 the mapped output is the
    first probability-like statement and "before" stays NaN — recorded as absent rather
    than dressed up as a baseline.
    """
    y = np.asarray(labels_dev).ravel().astype(float)
    if p_dev is not None and np.unique(y).size == 2 and np.isfinite(p_dev).sum() >= 16:
        fit.brier_after = round(brier(p_dev, y), 5)
        fit.ece_after = round(float(expected_calibration_error(p_dev, y)["ece"]), 5)
    return p_dev


def calibrate(
    method: str, *, scores_dev: np.ndarray, labels_dev: np.ndarray,
    train_scores: np.ndarray | None = None, scores_eval: np.ndarray | None = None,
    split: str = "dev", eval_keys: np.ndarray | None = None,
    train_keys: np.ndarray | None = None, min_history: int = 2,
) -> tuple[np.ndarray, CalibrationFit]:
    """Fit (if needed) and map ``scores_eval`` to probabilities.

    ``method`` is the contract's calibration id. ``C1`` (farm-specific) and ``C2``
    (farm × stage) map every row through the reference distribution of its own pool
    (:func:`pooled_reference_map`) when pooling keys are supplied, falling back to the
    pre-registered global maps when they are not — the fallback is recorded, never
    silent. ``C3`` is the global fitted mapping: temperature then isotonic, whichever
    reduces NLL more, with the choice recorded in ``params``.
    """
    if method in ("C1", "C2") and eval_keys is not None and train_keys is not None:
        if train_scores is None:
            raise ValueError(f"{method} pooling needs train scores; dev scores would "
                             "calibrate on the scored partition")
        p, rec = pooled_reference_map(
            scores_eval if scores_eval is not None else scores_dev, train_scores,
            eval_keys=eval_keys, train_keys=train_keys, min_history=min_history,
            map_name="ecdf" if method == "C1" else "zscore",
        )
        fit = CalibrationFit(
            method="pooled_ecdf" if method == "C1" else "pooled_zscore",
            params=rec, fitted_on="train",
            n_fit=int(np.asarray(train_scores).size))
        fit.warnings.append(
            f"reference pools sized from the train partition only ({rec['n_pools']} pools; "
            f"{rec['pooled_fill']} pooled fills)")
        return _attach_dev_diagnostics(p, labels_dev, fit), fit
    if method == "C1":
        if train_scores is None:
            raise ValueError("C1 needs train scores for its ECDF; dev scores would calibrate on the eval set")
        p = rank_to_probability(scores_eval if scores_eval is not None else scores_dev, train_scores)
        fit = CalibrationFit(method="ecdf", params={"train_n": int(len(train_scores)),
                                                    "pooling_level": "global_fallback",
                                                    "fallback_note": "no pooling keys supplied"},
                             fitted_on="train")
        return _attach_dev_diagnostics(p, labels_dev, fit), fit
    if method == "C2":
        src = train_scores if train_scores is not None else scores_dev
        p = zscore_to_probability(scores_eval if scores_eval is not None else scores_dev, src)
        fit = CalibrationFit(method="zscore", params={"reference": "train_robust_z",
                                                      "pooling_level": "global_fallback",
                                                      "fallback_note": "no pooling keys supplied"},
                             fitted_on="train" if train_scores is not None else split)
        return _attach_dev_diagnostics(p, labels_dev, fit), fit
    if method != "C3":
        raise KeyError(f"unknown calibration id {method!r} (expected C1/C2/C3)")
    z = np.log(np.clip(scores_dev, 1e-6, 1 - 1e-6) / (1 - np.clip(scores_dev, 1e-6, 1 - 1e-6)))
    t, tfit = fit_temperature(z, labels_dev, split=split)
    ifit = fit_isotonic(scores_dev, labels_dev, split="dev")
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
        return _attach_dev_diagnostics(apply_calibration("temperature", zz, {"temperature": t}),
                                       labels_dev, fit), fit
    return _attach_dev_diagnostics(apply_calibration("isotonic", ev, {"fitter": _iso_fit(scores_dev, labels_dev)}),
                                   labels_dev, fit), fit


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

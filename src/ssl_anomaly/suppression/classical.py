"""Transferring S1/S3 to the classical baselines (task 2.10).

The applicability table (design §Suppression transfer) splits the estimators into
three groups, and the point of encoding the split is that the *untransferable* ones
still appear in every report with their reason:

* **supports sample weights** — PCA/RPCA residuals can be reweighted by wrapping the
  residual computation, and the Mahalanobis/robust-covariance rules can down-weight
  points when fitting the centre. For these, S1/S3 is a genuine arm.
* **has a contamination parameter** — Isolation Forest and One-Class SVM (ν) expose
  exactly one knob of the right kind, so S3 maps onto it and S1 does not (their
  training loss is not a per-sample weighted sum the caller can reach).
* **nothing to transfer to** — a fixed operational rule has no objective at all.
  Running "suppressed" versions of it would be theatre, so they get a
  **contamination sensitivity row** instead: the same rule, scored on a panel whose
  anomaly prevalence is varied. That answers the question the suppression axis is
  really asking (does performance survive dirty data?) without pretending to be a
  training-treatment comparison.

Nothing here fits anything to labels: every estimator is fitted on the train
partition, and the contamination grid is a sensitivity axis, not a tuning axis —
:func:`select_classical_variant` is given the same dev-only selection interface as
the neural arms and records which grid point it picked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: id → what the estimator can accept. Keyed by the contract's BASELINES ids so a
#: new baseline without a declared transfer path fails at registration, not in
#: the middle of a run.
TRANSFER: dict[str, dict] = {
    "B1":  {"S0": "identity", "S1": "residual_reweight", "S2": "input_noising",
            "S3": "sensitivity_row", "fitter": "robust_centre"},
    "BL-IF": {"S0": "identity", "S1": "not_applicable", "S2": "input_noising",
              "S3": "contamination_param", "fitter": "sklearn_isolation_forest"},
    "BL-OCSVM": {"S0": "identity", "S1": "not_applicable", "S2": "input_noising",
                 "S3": "nu_param", "fitter": "sklearn_one_class_svm"},
    "BL-RPCA": {"S0": "identity", "S1": "residual_reweight", "S2": "input_noising",
                "S3": "sensitivity_row", "fitter": "rpca_pca"},
    "BL-RULE": {"S0": "identity", "S1": "not_applicable", "S2": "not_applicable",
                "S3": "sensitivity_row", "fitter": "fixed_rule"},
    "A-RAND": {"S0": "identity", "S1": "not_applicable", "S2": "not_applicable",
               "S3": "not_applicable", "fitter": "none"},
}


class TransferError(RuntimeError):
    """A suppression arm was requested of a baseline that cannot express it."""


def transfer_mode(baseline_id: str, suppression_id: str) -> str:
    try:
        row = TRANSFER[baseline_id]
    except KeyError:
        raise TransferError(
            f"{baseline_id!r} declares no suppression transfer; register it in TRANSFER with an "
            "explicit reason rather than defaulting to 'identity'"
        ) from None
    mode = row.get(suppression_id)
    if mode is None:
        raise TransferError(
            f"{baseline_id} × {suppression_id} is not in the transfer table "
            f"(known: {sorted(row)}); the applicability matrix and this table must agree"
        )
    return mode


def assert_transfer(baseline_id: str, suppression_id: str, applicability_reason: str | None) -> dict:
    """Return the row the manifest records, refusing silent substitution."""
    mode = transfer_mode(baseline_id, suppression_id)
    if mode == "not_applicable":
        return {
            "baseline_id": baseline_id, "suppression_id": suppression_id, "mode": "not_applicable",
            "substituted": False,
            "reason": applicability_reason
            or f"{baseline_id} has no {suppression_id} mechanism; the cell is recorded Not Applicable",
        }
    if mode == "sensitivity_row":
        return {
            "baseline_id": baseline_id, "suppression_id": suppression_id, "mode": "sensitivity_row",
            "substituted": False,
            "reason": (
                f"{baseline_id} has no trainable objective to reweight; reported as a contamination "
                "sensitivity row (same estimator, varied panel prevalence) instead"
            ),
        }
    return {"baseline_id": baseline_id, "suppression_id": suppression_id, "mode": mode,
            "substituted": False, "reason": None}


# ---------------------------------------------------------------------------
# The estimators
# ---------------------------------------------------------------------------


def _pcp_inexact_alm(x: np.ndarray, *, rank: int, lam: float, iters: int,
                     tol: float = 1e-7, rho: float = 1.6) -> tuple[np.ndarray, np.ndarray, dict]:
    """Principal component pursuit by inexact ALM (Lin, Chen & Ma 2010, Alg. 5 variant).

    Solves ``min ‖L‖_* + lam·‖S‖₁  s.t.  X = L + S`` deterministically (pure numpy, no
    RNG): alternate a truncated-SVD step for L and a soft-threshold step for S, then
    update the dual variable Y and the penalty μ. Iterations stop early when the
    relative residual of ``X − L − S`` falls below ``tol`` — the recorded
    ``alm_iters_run``/``alm_rel_residual`` in the fit record are what let a reader check
    the searched ``iters`` was actually the convergence knob and not a label.

    The returned score model is ``L``'s row space; the caller scores new rows by their
    distance to that subspace, so ``lam``/``iters`` move the *learned* subspace, not
    just a fit statistic.
    """
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        # PCP has no NaN semantics; a non-finite train row is a data defect that must
        # stop the fit here rather than propagate NaN through every downstream score.
        raise TransferError(
            f"rpca_pca: {int((~np.isfinite(x)).sum())} non-finite values in the fit matrix; "
            "impute nothing — fix the binding or drop the rows upstream, and record it"
        )
    n, d = x.shape
    rank = max(1, min(int(rank), min(n, d) - 1)) if min(n, d) > 1 else 1
    spec = float(np.linalg.norm(x, 2))
    if spec <= 0:
        return x.copy(), np.zeros_like(x), {"iters_run": 0, "rel_residual": 0.0}
    mu = 1.25 / spec
    mu_bar = mu * 1e7
    y = x / max(spec, float(np.abs(x).max()) / max(lam, 1e-12), 1e-12)
    l_mat = np.zeros_like(x)
    s_mat = np.zeros_like(x)
    it_run = 0
    rel = float("inf")
    for _ in range(max(1, int(iters))):
        it_run += 1
        # L-step: truncated SVD of (X − S + Y/μ) at the requested rank
        u, sv, vt = np.linalg.svd(x - s_mat + y / mu, full_matrices=False)
        l_mat = u[:, :rank] * sv[:rank] @ vt[:rank]
        # S-step: soft-threshold at lam/μ
        r = x - l_mat + y / mu
        s_mat = np.sign(r) * np.clip(np.abs(r) - lam / mu, 0.0, None)
        z = x - l_mat - s_mat
        rel = float(np.linalg.norm(z) / max(spec, 1e-12))
        y += mu * z
        mu = min(mu * rho, mu_bar)
        if rel < tol:
            break
    return l_mat, s_mat, {"iters_run": int(it_run), "rel_residual": round(rel, 10)}


def robust_centre(x: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Weighted median centre + MAD scale — the B1 rule's fittable core.

    ``weights`` is what S1 transfers into: a weighted median is still a median, so
    the centre stops being dragged by contaminated cycles instead of the estimator
    changing.
    """
    v = np.asarray(x, dtype=np.float64)
    if v.ndim == 1:
        v = v[:, None]
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w.shape[0] != len(v):
        raise TransferError(f"weights {w.shape[0]} rows vs data {len(v)} rows")
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if not w.any():
        raise TransferError("all weights are zero/non-finite; a weighted median needs some weight left")
    centre = np.empty(v.shape[1])
    scale = np.empty(v.shape[1])
    for c in range(v.shape[1]):
        col = v[:, c]
        ok = np.isfinite(col)
        order = np.argsort(col[ok])
        cw = w[ok][order]
        cum = np.cumsum(cw)
        half = 0.5 * cum[-1]
        centre[c] = float(col[ok][order][int(np.searchsorted(cum, half))]) if cw.size else np.nan
        dev = np.abs(col - centre[c])
        med = np.average(dev[np.isfinite(dev)], weights=w[np.isfinite(dev)]) if np.isfinite(dev).any() else np.nan
        scale[c] = float(med) * 1.4826 if np.isfinite(med) else np.nan
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return centre, scale


#: which estimator each pre-registered reference arm actually is. The mapping lives here,
#: next to the estimators, because a runner that guessed it (``transfer.get("fitter") or
#: "robust_centre"``) silently fitted the same centre-spread rule for IF, OCSVM and RPCA
#: alike — three "baselines" that were one baseline, with every searched knob ignored.
FITTERS: dict[str, str] = {
    "B1": "robust_centre",
    "BL-RULE": "fixed_rule",
    "BL-IF": "sklearn_isolation_forest",
    "BL-OCSVM": "sklearn_one_class_svm",
    "BL-RPCA": "rpca_pca",
}
#: the search-space keys each fitter reads. Anything else in a selection is refused, not dropped:
#: a knob that no code consumes turns a "tuned" arm into an untuned one wearing a tuned label.
#: The set must state exactly what :func:`fit_score` reads for that kind — the contract's
#: BL-RPCA space declares ``lam``/``iters`` (tasks 8.3/11.1), so ``rpca_pca`` is the PCP
#: estimator below, and ``contamination`` is *not* listed because PCP has no contamination
#: parameter: declaring it consumable would let a selection carry a knob that is silently
#: dropped, which is the failure this table exists to prevent.
FITTER_CONSUMED: dict[str, frozenset[str]] = {
    "robust_centre": frozenset(),
    "fixed_rule": frozenset(),
    "sklearn_isolation_forest": frozenset({"n_estimators", "max_samples", "contamination"}),
    "sklearn_one_class_svm": frozenset({"nu", "gamma", "kernel", "contamination"}),
    "rpca_pca": frozenset({"n_components", "lam", "iters"}),
}


def fitter_for(model_id: str) -> str:
    """The estimator id for a reference arm; an unmapped arm is an error, not a default."""
    try:
        return FITTERS[model_id]
    except KeyError:
        raise TransferError(
            f"{model_id!r} is not mapped to a fitter in `suppression.classical.FITTERS` "
            f"(known: {sorted(FITTERS)}); guessing one would run an arm under an estimator the "
            "contract never named"
        ) from None


def unconsumed(model_id: str, keys) -> list[str]:
    """Searched keys this arm's fitter will ignore — the contract's space, audited."""
    kind = FITTERS.get(model_id)
    if kind is None:
        return sorted(set(keys))
    return sorted(set(keys) - FITTER_CONSUMED.get(kind, frozenset()))


def fit_score(
    kind: str, x_train: np.ndarray, x_score: np.ndarray, *,
    weights: np.ndarray | None = None, contamination: float | None = None,
    params: dict | None = None, seed: int = 13,
) -> tuple[np.ndarray, dict]:
    """Fit on train, score ``x_score``; returns ``(scores, fit_record)``.

    The fit record is not decoration: "which knobs were actually available for this
    estimator under this suppression" is the fact a reader needs to interpret a
    baseline's rank.
    """
    p = dict(params or {})
    xt = np.atleast_2d(np.asarray(x_train, dtype=np.float64))
    xs = np.atleast_2d(np.asarray(x_score, dtype=np.float64))
    xt = xt.reshape(xt.shape[0], -1)
    xs = xs.reshape(xs.shape[0], -1)
    fin = np.isfinite(xt).all(axis=1)
    if not fin.all():
        # dropping a broken train row is the shared policy of every fitter here, but it
        # must be countable: a fit that silently shrank its training set is exactly the
        # kind of provenance gap AGENTS.md §7 flags.
        record["n_fit_rows_dropped_nonfinite"] = int((~fin).sum())
        xt = xt[fin]
    record: dict = {"kind": kind, "n_fit_rows": int(len(xt)), "seed": int(seed)}

    if kind == "robust_centre":
        centre, scale = robust_centre(xt, weights=weights)
        z = np.abs(xs - centre) / scale
        s = np.nanmean(z, axis=1)
        record.update(centre=centre.tolist(), scale=scale.tolist(),
                      suppression_applied="weighted_median" if weights is not None else "unweighted")
        return np.asarray([np.nan if not np.isfinite(v) else v for v in s], dtype=np.float64), record

    if kind == "rpca_pca":
        # Robust PCA as principal component pursuit (Candès et al. 2011): X = L + S with
        # L low-rank and S sparse, solved by inexact ALM (Lin et al. 2010). The contract's
        # declared search space for BL-RPCA is {lam, iters}; this branch is the estimator
        # that actually reads them. The previous implementation here was a plain-SVD
        # projection residual — PCA wearing the Robust name — and it read neither knob, so
        # every searched trial fitted the identical model and HPO refused the arm rather
        # than report a tuned arm that was not.
        #
        # Scoring: the sparse part S carries the anomalies in the *training* decomposition;
        # for new rows the same statement is "the part of x the low-rank normal subspace
        # does not explain", i.e. the distance to the subspace spanned by L. lam and iters
        # act on the fit (how aggressively train rows are demoted to S before the subspace
        # is estimated), so a searched knob changes the learned model, not just a label.
        n_comp = int(p.get("n_components", 3))
        n, d = xt.shape
        lam_default = float(1.0 / np.sqrt(max(n, d)))   # the PCP-theoretical lam (Candès 2011)
        lam = float(p["lam"]) if p.get("lam") is not None else lam_default
        iters = int(p.get("iters", 100))
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64).ravel()
            w = np.resize(w, len(xt)) if w.size != len(xt) else w
            w = np.clip(np.where(np.isfinite(w), w, 0.0), 1e-6, None)
            sw = np.sqrt(w / w.mean())
            mu = (xt * sw[:, None]).mean(axis=0)
            fit_mat = (xt - mu) * sw[:, None]
            record["suppression_applied"] = "row_rescaling_before_pcp"
        else:
            mu = xt.mean(axis=0)
            fit_mat = xt - mu
            record["suppression_applied"] = "none"
        n_comp = max(1, min(n_comp, min(fit_mat.shape) - 1))
        l_mat, s_mat, info = _pcp_inexact_alm(fit_mat, rank=n_comp, lam=lam, iters=iters)
        u, sv, vt = np.linalg.svd(l_mat, full_matrices=False)
        # distance from the row to the low-rank subspace L spans — the PCP analogue of
        # "the part the normal model does not explain". (The previous implementation
        # projected onto vtᵀ and subtracted that (rows, r) matrix from the (rows, d)
        # input, which cannot broadcast for any real feature count: BL-RPCA could never
        # have completed a cell. The residual is the subspace distance, in d dimensions.)
        proj = ((xs - mu) @ vt[:n_comp].T) @ vt[:n_comp]
        resid = (xs - mu) - proj
        s = np.linalg.norm(resid, axis=1)
        row_sparse = np.abs(s_mat).sum(axis=1)
        sparse_thr = float(np.median(row_sparse[row_sparse > 0])) if (row_sparse > 0).any() else np.inf
        record.update(
            estimator="pcp_inexact_alm",
            n_components=n_comp, lam=round(lam, 6), lam_default=round(lam_default, 6),
            lam_source="selection" if p.get("lam") is not None else "pcp_theory_1_over_sqrt_max_nd",
            iters=iters, alm_iters_run=info["iters_run"], alm_rel_residual=info["rel_residual"],
            singular_values=sv.tolist(),
            explained_var=float((sv[:n_comp] ** 2).sum() / max((sv ** 2).sum(), 1e-12)),
            train_sparse_fraction=round(float((row_sparse > sparse_thr).mean()), 4),
            train_sparse_l1_mean=round(float(row_sparse.mean()) / max(int(fit_mat.shape[1]), 1), 6),
        )
        return s, record

    if kind in ("sklearn_isolation_forest", "sklearn_one_class_svm"):
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.svm import OneClassSVM
        except Exception as exc:  # pragma: no cover - environment dependent
            raise TransferError(f"scikit-learn is required for {kind}: {exc}") from exc
        if kind == "sklearn_isolation_forest":
            kw = dict(n_estimators=int(p.get("n_estimators", 200)), random_state=int(seed),
                      contamination=float(contamination) if contamination is not None else "auto")
            if p.get("max_samples") is not None:      # sklearn accepts a float fraction or an int
                ms = p["max_samples"]
                kw["max_samples"] = int(ms) if isinstance(ms, int) or float(ms) >= 1 else float(ms)
            if weights is not None:
                raise TransferError(
                    "Isolation Forest accepts no per-sample weight; S1 is recorded not_applicable "
                    "for BL-IF rather than approximated by resampling rows"
                )
            mdl = IsolationForest(**kw).fit(xt)
            s = -mdl.score_samples(xs)
            record.update(backend="IsolationForest", contamination=kw["contamination"],
                          suppression_applied="contamination" if contamination is not None else "none")
        else:
            nu = float(contamination) if contamination is not None else float(p.get("nu", 0.05))
            mdl = OneClassSVM(kernel=p.get("kernel", "rbf"), gamma=p.get("gamma", "scale"), nu=nu)
            mdl.fit(xt)
            s = -mdl.score_samples(xs)
            record.update(backend="OneClassSVM", nu=nu,
                          suppression_applied="nu" if contamination is not None else "none")
        return np.asarray(s, dtype=np.float64).ravel(), record

    if kind == "fixed_rule":
        # The operational rule: a threshold on NDVI decline, no fitted parameters.
        names = p.get("channel_names") or []
        idx = names.index("ndvi") if "ndvi" in names else 0
        drop = xs[:, idx] - xt[:, idx].min()
        s = np.maximum(0.0, xt[:, idx].mean() - xs[:, idx]) / max(
            float(np.std(xt[:, idx])), 1e-6)
        record.update(rule="ndvi_level_vs_train_mean", suppression_applied="none",
                      reason="a fixed rule has no objective to reweight")
        del drop
        return np.asarray(s, dtype=np.float64).ravel(), record

    raise TransferError(f"unknown classical kind {kind!r}")


@dataclass
class SensitivityRow:
    """One (baseline, contamination) cell of the sensitivity table."""

    baseline_id: str
    contamination: float
    auprc: float
    auroc: float
    n_events: int
    note: str = ""


def contamination_sensitivity(
    baseline_id: str, score_fn, panel_fn, *, grid=(0.02, 0.05, 0.10, 0.20, 0.35),
    metrics_fn=None,
) -> list[SensitivityRow]:
    """Replace an impossible "suppressed baseline" arm with a measurable question.

    ``panel_fn(contamination)`` must return fresh (scores, labels) at that
    prevalence; the answer is a curve, not a verdict, and it is reported for every
    baseline whose suppression transfer is ``sensitivity_row``.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows: list[SensitivityRow] = []
    for c in grid:
        scores, labels = panel_fn(float(c))
        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(labels).ravel().astype(int)
        ok = np.isfinite(s)
        s, y = s[ok], y[ok]
        if np.unique(y).size < 2:
            rows.append(SensitivityRow(baseline_id, float(c), float("nan"), float("nan"),
                                        int(y.sum()), note="single_class_at_this_prevalence"))
            continue
        rows.append(
            SensitivityRow(
                baseline_id, float(c),
                float(average_precision_score(y, s)),
                float(roc_auc_score(y, s)),
                int(y.sum()),
                note="score_fn_supplied" if metrics_fn is None else "custom",
            )
        )
    return rows


def select_classical_variant(kind: str, grid: list[dict], *, score_on_dev, primary: str = "auprc") -> dict:
    """Dev-only selection for a baseline, so tuning is not done on the eval panel.

    ``score_on_dev(params) → float`` is supplied by the runner (the only component
    that can see dev labels); this function never receives the test split — that is
    enforced structurally by :mod:`ssl_anomaly.benchmarking.splits`, and the
    selection record states the split it used.
    """
    if not grid:
        raise TransferError(f"{kind}: empty search grid; a baseline with no grid must be marked unsearchable")
    best, recs = None, []
    for g in grid:
        v = float(score_on_dev(g))
        recs.append({"params": dict(g), primary: v})
        if best is None or v > best[primary]:
            best = {"params": dict(g), primary: v}
    best["selection_split"] = "dev"
    best["kind"] = kind
    best["n_candidates"] = len(grid)
    best["all"] = recs
    return best

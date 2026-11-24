"""Latent-representation stability and content preservation (tasks 5.x, 12.5).

For A3/A4/A7/A8 the failure that matters is not a bad loss curve — it is a latent
space that satisfies the objective by being *constant*, which the loss cannot see
(stop-gradient + EMA makes the constant solution attractive) and which the metric
cannot see either (a collapsed arm scores near-chance AUPRC, the same as a merely
weak one). So collapse is measured directly:

``effective_rank``
    ``exp(H(p))`` over the normalised eigenvalues of the latent covariance — the
    participation ratio in a form whose floor (1) and ceiling (d) are both meanable.
``per_dim_std_floor``
    the smallest per-dimension std across bins: a dimension with ~0 spread is dead.
``top_eigenvalue_ratio``
    λ₁/Σλ: near 1 means one direction explains everything, i.e. the rest is noise.
``content_probe``
    a ridge probe from the frozen latent to the **phenological phase bin**. This is
    the positive test: a non-collapsed representation should still encode where in
    the season a cycle sits. If the probe fails, the arm is ``invalid`` — not
    "slightly worse" — because a JEPA arm that lost the phase information lost the
    thing the objective is supposed to preserve.

The verdict is one of four states and it is produced *before* the arm's metrics are
reported, so a collapsed run can never appear in a ranking table without the flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: thresholds from the contract's `collapse:` block; defaults mirror the config so a
#: missing key fails loudly in :func:`evaluate` rather than silently disabling a gate.
#: name → (config key under ``collapse:``, default, direction). These are the key
#: names used in ``configs/benchmarks/benchmark_config.yaml``; a key the gate does not
#: know is an error rather than a silent fallback, because a gate that quietly reverts
#: to module defaults still looks alive while enforcing nothing the contract says.
_THRESHOLD_KEYS: dict[str, tuple[str, float, str]] = {
    "effective_rank_min": ("effective_rank_ratio_floor", 0.25, "min"),
    "per_dim_std_floor": ("min_per_dim_std", 1e-3, "min"),
    "top_eigenvalue_ratio_max": ("max_top_eigenvalue_share", 0.85, "max"),
    "zero_variance_dims_frac_max": ("max_zero_variance_dims_frac", 0.20, "max"),
    "pingpong_max": ("pingpong_instability_max", 0.60, "max"),
}

DEFAULTS = {
    "effective_rank": {"min": 0.25, "of": "latent_dim"},
    "per_dim_std_floor": 1e-3,
    "top_eigenvalue_ratio_max": 0.85,
    "probe_min_spearman": 0.10,
    "pingpong_max": 0.60,
}


@dataclass
class CollapseReport:
    """One arm's latent-health verdict, with every number that produced it."""

    model_id: str
    latent_dim: int
    n_samples: int
    effective_rank: float
    effective_rank_frac: float
    per_dim_std_min: float
    per_dim_std_median: float
    top_eigenvalue_ratio: float
    variance_term: float | None
    probe: dict = field(default_factory=dict)
    pingpong: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)
    verdict: str = "ok"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id, "latent_dim": self.latent_dim, "n_samples": self.n_samples,
            "effective_rank": round(self.effective_rank, 3),
            "effective_rank_frac": round(self.effective_rank_frac, 4),
            "per_dim_std": {"min": self.per_dim_std_min, "median": self.per_dim_std_median},
            "top_eigenvalue_ratio": round(self.top_eigenvalue_ratio, 4),
            "variance_term": self.variance_term,
            "probe": dict(self.probe), "pingpong": dict(self.pingpong),
            "thresholds": dict(self.thresholds), "verdict": self.verdict,
            "reasons": list(self.reasons),
        }


def latent_stats(z: np.ndarray, *, raw_floor: float = 1e-3) -> dict:
    """Rank / spread / anisotropy of a (N, d) latent matrix.

    Eigen-decomposition is of the *correlation* rather than the covariance for the
    rank and ratio, because a scale change (which the loss can freely absorb) must
    not masquerade as a rank change; the raw per-dim std is still reported so a
    genuinely vanishing scale stays visible.
    """
    a = np.asarray(z, dtype=np.float64)
    if a.ndim == 3:
        a = a.reshape(-1, a.shape[-1])
    if a.ndim != 2:
        raise ValueError(f"latent stats need (N, d) or (B, T, d), got {a.shape}")
    a = a[np.isfinite(a).all(axis=1)]
    n, d = a.shape
    if n < max(8, d // 2):
        raise ValueError(
            f"only {n} finite latent rows for d={d}: the covariance is rank-deficient, "
            "so effective rank would measure the sample count rather than the representation"
        )
    mu = a.mean(axis=0)
    b = a - mu
    std = b.std(axis=0, ddof=0)
    # Correlation eigen-spectrum, guarded against exactly-zero dimensions (they get
    # unit scale so they show as dead in `per_dim_std_min` rather than as rank loss).
    safe = np.where(std > 0, std, 1.0)
    c = b / safe
    corr = (c.T @ c) / n
    ev = np.linalg.eigvalsh(corr)
    ev = np.clip(ev[::-1], 0.0, None)
    p = ev / max(float(ev.sum()), 1e-12)
    p = p[p > 0]
    ent = float(-(p * np.log(p)).sum())
    return {
        "n": int(n), "latent_dim": int(d),
        "effective_rank": float(np.exp(ent)),
        "top_eigenvalue_ratio": float(ev[0] / max(float(ev.sum()), 1e-12)),
        "per_dim_std_min": float(std.min()),
        "zero_variance_dims_frac": float((std <= max(1e-8, float(raw_floor) * 1e-2)).mean()),
        "per_dim_std_median": float(np.median(std)),
        "n_dead_dims": int((std <= 0).sum()),
        "cov_trace": float((std ** 2).sum()),
    }


def pingpong_stats(scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """Instability across consecutive evals: the "ping-pong" the spec names.

    Two readings of the same cycles at nearby steps that disagree on which cycles are
    top-ranked means the score is not a stable estimator, and an AUPRC computed on a
    flapping score is a draw from a distribution, not a measurement.
    """
    a = np.asarray(scores_a, dtype=np.float64).ravel()
    b = np.asarray(scores_b, dtype=np.float64).ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return {"available": False, "reason": f"{int(ok.sum())} paired finite scores"}
    from scipy import stats as sst

    x, y = a[ok], b[ok]
    rho = float(sst.spearmanr(x, y).statistic)
    tau = float(sst.kendalltau(x, y, variant="b").statistic)
    k = max(1, int(round(0.1 * x.size)))
    ta = set(np.argsort(-x)[:k].tolist())
    tb = set(np.argsort(-y)[:k].tolist())
    return {
        "available": True,
        "spearman": rho, "kendall_tau_b": tau,
        "instability": float(1.0 - rho) if np.isfinite(rho) else None,
        "tail_jaccard": float(len(ta & tb) / max(1, len(ta | tb))),
        "mean_abs_shift_sd": float(np.mean(np.abs(x - y)) / (np.std(np.concatenate([x, y])) or 1.0)),
        "k": k,
    }


def content_probe(z: np.ndarray, phase_bin: np.ndarray, *, min_spearman: float = 0.10,
                  fit_on: str = "train", eval_on: str = "dev") -> dict:
    """Ridge probe: frozen latent → the contract's ``collapse.content_probe.target``.

    The target is the phenological phase bin by default (``content_probe.target:
    phase_bin``), because "does the latent still know *when* it is" is the question a
    collapse gate asks; any 1-D numeric target works, and the name is recorded in the
    output so a reader can see which one was used.

    Fitting on ``train`` and evaluating on ``dev`` is not ceremony: a probe fitted on
    the same rows it scores reports the training set's own structure, which is the
    error this probe exists to detect in the arm.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    from scipy import stats as sst

    A = np.asarray(z, dtype=np.float64)
    if A.ndim == 3:
        A = A.reshape(-1, A.shape[-1])
    y = np.asarray(phase_bin, dtype=np.float64).ravel()
    ok = np.isfinite(A).all(axis=1) & np.isfinite(y)
    A, y = A[ok], y[ok]
    n_cls = int(np.unique(y).size)
    if A.shape[0] < 32 or n_cls < 2:
        return {"available": False, "reason": f"n={A.shape[0]}, classes={n_cls}", "passed": None}
    n_tr = max(16, int(round(0.7 * A.shape[0])))
    idx_tr, idx_te = np.arange(n_tr), np.arange(n_tr, A.shape[0])
    if idx_te.size < 8:
        return {"available": False, "reason": "not enough rows for a held-out probe split",
                "passed": None}
    # A held-out split with one class left in it cannot say anything: Spearman on a
    # constant target is either NaN or an accident.
    if np.unique(y[idx_te]).size < 2 or np.unique(y[idx_tr]).size < 2:
        return {"available": False, "reason": "a probe split lost a class", "passed": None}
    mdl = Ridge(alpha=1.0).fit(A[idx_tr], y[idx_tr])
    pred = mdl.predict(A[idx_te])
    rho = float(sst.spearmanr(y[idx_te], pred).statistic)
    return {
        "available": True, "estimator": "ridge(alpha=1.0)", "target": "phase_bin",
        "n_classes": n_cls,
        "fit_on": fit_on, "eval_on": eval_on, "n_fit": int(idx_tr.size), "n_eval": int(idx_te.size),
        "r2": float(r2_score(y[idx_te], pred)), "spearman": rho,
        "min_spearman": float(min_spearman), "passed": bool(np.isfinite(rho) and rho >= min_spearman),
    }


def evaluate(
    *, model_id: str, latents: np.ndarray | None, phase_bin: np.ndarray | None,
    cfg, variance_term: float | None = None, scores_prev: np.ndarray | None = None,
    scores_now: np.ndarray | None = None,
) -> CollapseReport:
    """Gate a latent arm: rank, dead dims, anisotropy, content probe, flapping."""
    raw = cfg.raw["collapse"]
    # A deep copy: `thr` is edited in place below, and a shallow dict(DEFAULTS) would
    # rewrite the module-level defaults for every later caller.
    thr = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    # Read the contract's OWN key names. The earlier version looked up aliases that
    # the YAML does not define, so every threshold silently fell back to a module
    # default and the config's numbers governed nothing — the worst kind of drift,
    # because the gate still looked alive. An unrecognised key is now an error.
    known = {"mechanism", "mechanism_doc", "applies_to", "check_every_epochs", "on_trigger",
             "content_probe"} | {v[0] for v in _THRESHOLD_KEYS.values()}
    unknown = sorted(k for k in raw if k not in known)
    if unknown:
        raise ValueError(
            f"collapse.{unknown} is not a threshold this gate knows: a renamed or misspelled key "
            f"would silently revert to the module default (accepted: {sorted(known)})"
        )
    for name, (cfg_key, default, _dir) in _THRESHOLD_KEYS.items():
        thr[name] = float(raw.get(cfg_key, default))
    thr["effective_rank"]["min"] = thr.pop("effective_rank_min")
    thr["probe_min_spearman"] = float((raw.get("content_probe") or {}).get("min_spearman", 0.10))
    if latents is None:
        return CollapseReport(
            model_id=model_id, latent_dim=0, n_samples=0, effective_rank=float("nan"),
            effective_rank_frac=float("nan"), per_dim_std_min=float("nan"),
            per_dim_std_median=float("nan"), top_eigenvalue_ratio=float("nan"),
            variance_term=variance_term, thresholds=thr, verdict="not_applicable_by_construction",
            reasons=["arm exposes no latent space (reconstruction/prediction arms cannot collapse)"],
        )
    st = latent_stats(latents)
    probe = (
        content_probe(latents, phase_bin, min_spearman=thr["probe_min_spearman"])
        if phase_bin is not None else {"available": False, "reason": "no phase bins supplied", "passed": None}
    )
    pp = (pingpong_stats(scores_prev, scores_now) if scores_prev is not None and scores_now is not None
          else {"available": False, "reason": "needs two consecutive eval passes"})
    rep = CollapseReport(
        model_id=model_id, latent_dim=st["latent_dim"], n_samples=st["n"],
        effective_rank=st["effective_rank"],
        effective_rank_frac=st["effective_rank"] / max(1, st["latent_dim"]),
        per_dim_std_min=st["per_dim_std_min"], per_dim_std_median=st["per_dim_std_median"],
        top_eigenvalue_ratio=st["top_eigenvalue_ratio"], variance_term=variance_term,
        probe=probe, pingpong=pp, thresholds=thr,
    )
    reasons: list[str] = []
    if rep.effective_rank_frac < thr["effective_rank"]["min"]:
        reasons.append(
            f"effective rank {rep.effective_rank:.2f} = {rep.effective_rank_frac:.2f} of d="
            f"{rep.latent_dim} is below the floor {thr['effective_rank']['min']:.2f}"
        )
    if rep.per_dim_std_min < thr["per_dim_std_floor"]:
        reasons.append(f"a latent dimension has std {rep.per_dim_std_min:.2e} < floor {thr['per_dim_std_floor']:.0e}")
    zv = st.get("zero_variance_dims_frac")
    if zv is not None and zv > thr["zero_variance_dims_frac_max"]:
        reasons.append(
            f"{zv:.2f} of latent dimensions have ~zero variance, above the "
            f"{thr['zero_variance_dims_frac_max']:.2f} the contract allows"
        )
    if rep.top_eigenvalue_ratio > thr["top_eigenvalue_ratio_max"]:
        reasons.append(
            f"top eigenvalue ratio {rep.top_eigenvalue_ratio:.3f} exceeds {thr['top_eigenvalue_ratio_max']:.2f}"
        )
    if probe.get("passed") is False:
        reasons.append(
            f"content probe failed (Spearman {probe.get('spearman')} < {thr['probe_min_spearman']})"
        )
    if pp.get("available") and float(pp.get("instability") or 0.0) > thr["pingpong_max"]:
        reasons.append(f"ping-pong instability {pp['instability']:.3f} > {thr['pingpong_max']:.2f}")
    hard = [r for r in reasons if "content probe" in r]
    rep.reasons = reasons
    rep.verdict = "collapsed" if reasons and not hard else (
        "invalid_content_probe" if hard else ("borderline" if reasons else "ok")
    )
    # The contract's on_trigger decides what a collapse costs the run; recording the
    # decision here means the runner cannot forget to apply it.
    rep.verdict = rep.verdict if rep.verdict != "collapsed" else f"collapsed:{raw.get('on_trigger', 'disqualify')}"
    return rep


def run_content_probe(arm, tensors: dict, target: np.ndarray, *, min_spearman: float = 0.10,
                     fit_idx: np.ndarray | None = None, eval_idx: np.ndarray | None = None,
                     target_name: str = "phase_bin", cfg=None) -> dict:
    """Frozen-latent ridge probe onto the contract's target (task 5.4, ``collapse.content_probe``).

    ``target`` is indexed by the same global row numbers as ``tensors``: the probe
    selects its own rows from the fit/eval index arrays, so a pre-sliced target would
    silently describe a different cycle than its features.

    Fitted on the train rows and scored on the eval rows by default: a probe trained
    on the same rows it reports is measuring the latent's own noise. The target is
    whatever ``collapse.content_probe.target`` names (``phase_bin`` by default, cycle
    label when the caller supplies one) and the target actually used is in the output,
    because "the probe passed" means nothing without it.
    """
    import numpy as np
    import torch

    cp = (cfg.raw["collapse"].get("content_probe") or {}) if cfg is not None else {}
    getter = getattr(arm, "probe_features", None)
    if getter is None:
        return {"available": False, "reason": "arm exposes no probe_features()", "passed": None}
    n = int(np.asarray(tensors["x"]).shape[0])
    fit = np.arange(0, max(1, n // 2)) if fit_idx is None else np.asarray(fit_idx)
    ev = np.arange(max(1, n // 2), n) if eval_idx is None else np.asarray(eval_idx)
    feats, targets = [], []
    with torch.no_grad():
        for idx in (fit, ev):
            for start in range(0, len(idx), 64):
                rows = idx[start : start + 64]
                from ssl_anomaly.benchmarking.interface import make_batch

                z = getter(make_batch(tensors, rows))
                if z is None:
                    return {"available": False, "reason": "probe_features() returned None", "passed": None}
                z = z.detach().float().cpu().numpy()
                if z.ndim == 2:                      # an arm that pools internally
                    z = z[:, None, :]
                feats.append(z)
                targets.append(np.asarray(target)[rows])
    F = np.concatenate(feats, axis=0)                       # (R, T, D)
    Y = np.concatenate(targets, axis=0)                     # (R, T) bin-level or (R,) cycle-level
    # The target's own shape decides the granularity, and both sides move together:
    # ``collapse.content_probe.target: phase_bin`` is a per-bin quantity, so the probe
    # must read per-bin features, while a cycle label needs the features pooled. Pooling
    # the features first and then asking for a bin index — which is what a (B, D)
    # ``probe_features`` used to force — measures nothing.
    if Y.ndim == 2 and F.ndim == 3 and Y.shape[:2] == F.shape[:2]:
        A = F.reshape(-1, F.shape[-1])
        y = Y.reshape(-1)
        gran = "bin"
    else:
        A = F.mean(axis=1) if F.ndim == 3 else F
        y = Y
        gran = "cycle"
    if A.shape[0] != y.shape[0]:
        return {"available": False, "passed": None, "granularity": gran,
                "reason": (f"{A.shape[0]} feature rows vs {y.shape[0]} targets — the probe would "
                           "regress unrelated arrays, so it reports itself unusable instead")}
    out = content_probe(A, y, min_spearman=min_spearman)
    out["target"] = target_name
    out["granularity"] = gran
    out["required"] = str(arm.model_id) in [str(m) for m in (cp.get("required_for") or [])]
    out["fit_rows"] = int(len(fit))
    out["eval_rows"] = int(len(ev))
    out["feature_dim"] = int(F.shape[-1])
    return out

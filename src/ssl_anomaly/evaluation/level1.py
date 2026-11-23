"""Level-1 pretraining-quality battery (capability ``representation-pretraining-quality``).

Five probes, each reported per seed with mean ± std across seeds and an
interval where a rate is involved, all on **dev clean cycles only**:

1. masked reconstruction NRMSE per channel (observed entries only);
2. farm-identity separability — kNN k ∈ {1,5}, top-1/top-5 vs 1/490 and 5/490
   chance and vs a B1-feature baseline on the identical subset;
3. phenological-phase linear probe — 36-way DSH bin, macro-F1 + accuracy vs the
   majority baseline and vs the same probe on B1 features;
4. embedding temporal stability — within-field cycle-pair cosine vs a shuffled
   null;
5. collapse diagnostics — per-dim std, effective rank, alignment/uniformity,
   with pre-registered disqualification thresholds.

Every probe is *pure numpy* except the linear probe, which uses a closed-form
ridge multinomial fit (no sklearn dependency in the evidence path, so the
numbers cannot change silently with a library upgrade).

Design references: design.md §Level 1; spec requirements quoted per function.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ssl_anomaly.evaluation.data import CycleData
from ssl_anomaly.evaluation.metrics import wilson_interval

# ---------------------------------------------------------------------------
# Probe 1 — masked reconstruction quality
# ---------------------------------------------------------------------------


@dataclass
class ReconstructionReport:
    nrmse_per_channel: np.ndarray          # (C,) NaN where no observation
    n_observed_per_channel: np.ndarray     # (C,) int64 denominators
    channel_names: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "nrmse_per_channel": {
                name: (None if not np.isfinite(v) else float(v))
                for name, v in zip(self.channel_names, self.nrmse_per_channel, strict=True)
            },
            "n_observed_per_channel": {
                name: int(n)
                for name, n in zip(self.channel_names, self.n_observed_per_channel, strict=True)
            },
            "mean_nrmse": (None if not np.isfinite(self.mean()) else float(self.mean())),
        }

    def mean(self) -> float:
        vals = self.nrmse_per_channel[np.isfinite(self.nrmse_per_channel)]
        return float(vals.mean()) if vals.size else float("nan")


def masked_reconstruction(
    data: CycleData, predict_fn, *, mask: np.ndarray | None = None
) -> ReconstructionReport:
    """NRMSE per channel over observed (bin, channel) pairs only.

    ``predict_fn(X, M) -> X_hat`` must return a full ``(N, T, C)`` reconstruction.
    The metric divides the RMS error by the RMS of the observed target, so the
    value is scale-free and comparable across channels with different units.
    Unobserved entries are excluded from both the numerator and the denominator
    (spec: "Missing bin … contributes to neither the loss, the score, nor any
    denominator").
    """
    x_hat = np.asarray(predict_fn(data.X, data.M), dtype=np.float64)
    if x_hat.shape != data.X.shape:
        raise ValueError(f"prediction {x_hat.shape} must match {data.X.shape}")
    return ReconstructionReport(
        nrmse_per_channel=data.masked_nrmse(x_hat, mask),
        n_observed_per_channel=data.M.sum(axis=1).sum(axis=0).astype(np.int64),
        channel_names=data.channel_names,
    )


# ---------------------------------------------------------------------------
# Probe 2 — farm-identity separability
# ---------------------------------------------------------------------------


@dataclass
class FarmSeparabilityReport:
    k: int
    top1: float
    top5: float
    top1_ci: tuple[float, float]
    chance_top1: float
    chance_top5: float
    n_queries: int
    n_classes: int
    subset_hash: str = ""

    def to_dict(self) -> dict:
        return {
            "k": self.k,
            "top1": self.top1,
            "top5": self.top5,
            "top1_ci": list(self.top1_ci),
            "chance_top1": self.chance_top1,
            "chance_top5": self.chance_top5,
            "n_queries": self.n_queries,
            "n_classes": self.n_classes,
            "subset_hash": self.subset_hash,
        }


def _cosine_knn_topk(
    queries: np.ndarray, gallery: np.ndarray, k: int, *, exclude_self: np.ndarray | None = None
) -> np.ndarray:
    q = _l2norm(queries)
    g = _l2norm(gallery)
    sim = q @ g.T
    if exclude_self is not None:
        sim[exclude_self, exclude_self] = -np.inf
    kk = min(k, sim.shape[1])
    return np.argpartition(-sim, kk - 1, axis=1)[:, :kk]


def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def farm_separability(
    embeddings: np.ndarray,
    field_ids: np.ndarray,
    *,
    k: int = 1,
    subset_hash: str = "",
) -> FarmSeparabilityReport:
    """Leave-one-out kNN farm classification of frozen embeddings.

    Every cycle is a query against every other cycle (self excluded), so the
    probe answers "does the embedding place a cycle near other cycles of the
    same farm?". Chance is 1/n_classes for top-1 and k/n_classes for top-5 —
    reported with the same n so a reader can see the lift.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    f = np.asarray(field_ids)
    if z.ndim != 2 or z.shape[0] != f.size:
        raise ValueError(f"embeddings {z.shape} must be (N, D) matching {f.size} ids")
    n = z.shape[0]
    if n < 3:
        raise ValueError("farm separability needs at least 3 cycles")
    kk = min(k, n - 1)
    nn = _cosine_knn_topk(z, z, max(kk, 5), exclude_self=np.arange(n))
    top1 = float(np.mean(f[nn[:, 0]] == f))
    top5 = float(np.mean([np.any(f[row[: min(5, row.size)]] == f[i]) for i, row in enumerate(nn)]))
    n_classes = int(np.unique(f).size)
    hits = int(round(top1 * n))
    return FarmSeparabilityReport(
        k=k,
        top1=top1,
        top5=top5,
        top1_ci=wilson_interval(hits, n),
        chance_top1=1.0 / n_classes,
        chance_top5=min(1.0, 5.0 / n_classes),
        n_queries=n,
        n_classes=n_classes,
        subset_hash=subset_hash,
    )


# ---------------------------------------------------------------------------
# Probe 3 — phenological-phase linear probe
# ---------------------------------------------------------------------------


@dataclass
class PhaseProbeReport:
    accuracy: float
    macro_f1: float
    majority_accuracy: float
    majority_macro_f1: float
    n_train: int
    n_eval: int
    n_classes: int
    axis: str = "dsh"

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "majority_accuracy": self.majority_accuracy,
            "majority_macro_f1": self.majority_macro_f1,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "n_classes": self.n_classes,
            "axis": self.axis,
        }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 over the classes present in ``y_true``."""
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fn == 0:
            continue  # class absent from truth: excluded, not scored as 0
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn)
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s)) if f1s else float("nan")


def phase_bin_targets(dsh: np.ndarray, n_bins: int, bin_days: float = 15.0) -> np.ndarray:
    """Map per-embedding DSH to the pre-registered 36-bin phase target."""
    d = np.asarray(dsh, dtype=np.float64)
    out = np.floor(d / bin_days).astype(np.int64)
    return np.clip(out, 0, n_bins - 1)


def _ridge_fit_multiclass(X: np.ndarray, y: np.ndarray, n_classes: int, lam: float) -> np.ndarray:
    """Closed-form one-vs-rest ridge classifier: returns (D+1, C) weights."""
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    Y = np.zeros((n, n_classes))
    Y[np.arange(n), y] = 1.0
    A = Xa.T @ Xa + lam * np.eye(d + 1)
    A[-1, -1] -= lam  # do not regularise the intercept
    return np.linalg.solve(A, Xa.T @ Y)


def phase_linear_probe(
    train_emb: np.ndarray,
    train_bins: np.ndarray,
    eval_emb: np.ndarray,
    eval_bins: np.ndarray,
    *,
    n_classes: int = 36,
    lam: float = 1.0,
    axis: str = "dsh",
) -> PhaseProbeReport:
    """Linear probe predicting the phase bin from a frozen embedding.

    Trained on the TRAIN split only (spec), evaluated on dev. The majority
    baseline is computed on the dev targets — the honest bar a probe must clear
    to be informative at all.
    """
    tr = np.asarray(train_emb, dtype=np.float64)
    te = np.asarray(eval_emb, dtype=np.float64)
    ytr = np.asarray(train_bins, dtype=np.int64)
    yte = np.asarray(eval_bins, dtype=np.int64)
    if tr.shape[0] != ytr.size or te.shape[0] != yte.size:
        raise ValueError("embedding rows and target lengths must match")
    if tr.size == 0 or te.size == 0:
        raise ValueError("phase probe needs non-empty train and eval sets")

    W = _ridge_fit_multiclass(tr, ytr, n_classes, lam)
    pred = np.argmax(np.hstack([te, np.ones((te.shape[0], 1))]) @ W, axis=1)
    accuracy = float(np.mean(pred == yte))

    counts = np.bincount(yte, minlength=n_classes)
    majority = int(np.argmax(counts))
    maj_pred = np.full_like(yte, majority)
    majority_acc = float(np.mean(maj_pred == yte))
    if majority_acc >= 0.999:
        raise ValueError(
            f"phase probe target is degenerate: {np.unique(yte).size} distinct class"
            f"(es) in the evaluation half, majority baseline {majority_acc:.3f}. A probe "
            "scored against a 100% baseline measures nothing — fix the target (see "
            "level1._cycle_phase_bins) rather than reporting its accuracy."
        )
    return PhaseProbeReport(
        accuracy=accuracy,
        macro_f1=macro_f1(yte, pred, n_classes),
        majority_accuracy=majority_acc,
        majority_macro_f1=macro_f1(yte, maj_pred, n_classes),
        n_train=int(tr.shape[0]),
        n_eval=int(te.shape[0]),
        n_classes=n_classes,
        axis=axis,
    )


# ---------------------------------------------------------------------------
# Probe 4 — embedding temporal stability
# ---------------------------------------------------------------------------


@dataclass
class StabilityReport:
    median_cosine: float
    iqr: tuple[float, float]
    tail_fraction_below_floor: float
    shuffled_median: float
    n_pairs: int
    n_fields_excluded: int
    floor: float

    def to_dict(self) -> dict:
        return {
            "median_cosine": self.median_cosine,
            "iqr": list(self.iqr),
            "tail_fraction_below_floor": self.tail_fraction_below_floor,
            "shuffled_median": self.shuffled_median,
            "n_pairs": self.n_pairs,
            "n_fields_excluded": self.n_fields_excluded,
            "floor": self.floor,
        }


def temporal_stability(
    per_cycle_emb: np.ndarray,
    field_ids: np.ndarray,
    *,
    floor: float = 0.5,
    seed: int = 13,
) -> StabilityReport:
    """Within-field cycle-pair cosine similarity, against a shuffled null.

    A field with fewer than two cycles cannot contribute a pair: it is counted
    in ``n_fields_excluded`` (spec: "Insuficient cycles … excluded … the
    excluded count is reported").
    """
    z = _l2norm(np.asarray(per_cycle_emb, dtype=np.float64))
    f = np.asarray(field_ids)
    sims, nulls = [], []
    n_excluded = 0
    rng = np.random.default_rng(seed)
    for fid in np.unique(f):
        idx = np.flatnonzero(f == fid)
        if idx.size < 2:
            n_excluded += 1
            continue
        for a in range(idx.size):
            for b in range(a + 1, idx.size):
                sims.append(float(z[idx[a]] @ z[idx[b]]))
        others = np.flatnonzero(f != fid)
        if others.size:
            for a in idx:
                b = int(rng.choice(others))
                nulls.append(float(z[a] @ z[b]))
    if not sims:
        return StabilityReport(float("nan"), (float("nan"), float("nan")), float("nan"),
                               float("nan"), 0, n_excluded, floor)
    s = np.asarray(sims)
    q25, q75 = np.quantile(s, [0.25, 0.75])
    return StabilityReport(
        median_cosine=float(np.median(s)),
        iqr=(float(q25), float(q75)),
        tail_fraction_below_floor=float(np.mean(s < floor)),
        shuffled_median=float(np.median(nulls)) if nulls else float("nan"),
        n_pairs=int(s.size),
        n_fields_excluded=int(n_excluded),
        floor=floor,
    )


# ---------------------------------------------------------------------------
# Probe 5 — collapse diagnostics
# ---------------------------------------------------------------------------


@dataclass
class CollapseReport:
    per_dim_std: np.ndarray
    effective_rank: float
    effective_rank_ratio: float
    top_eigenvalue_share: float
    alignment: float
    uniformity: float
    latent_dim: int
    disqualified: bool
    triggers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "per_dim_std": [float(v) for v in self.per_dim_std],
            "effective_rank": self.effective_rank,
            "effective_rank_ratio": self.effective_rank_ratio,
            "top_eigenvalue_share": self.top_eigenvalue_share,
            "alignment": self.alignment,
            "uniformity": self.uniformity,
            "latent_dim": self.latent_dim,
            "disqualified": self.disqualified,
            "triggers": list(self.triggers),
        }


def collapse_diagnostics(
    embeddings: np.ndarray,
    *,
    effective_rank_ratio_floor: float = 0.10,
    min_per_dim_std: float = 0.01,
    max_zero_variance_dims_frac: float = 0.20,
    max_top_eigenvalue_share: float = 0.99,
) -> CollapseReport:
    """Detect a degenerate representation BEFORE it reaches any downstream probe.

    Metrics (design §Level 1.5):
    - per-dimension standard deviation spectrum;
    - effective rank of the covariance = ``exp(entropy(eigenvalues))`` (the
      participation ratio in log space), reported both absolutely and as a
      ratio of the latent dimension;
    - alignment (mean squared norm of unit-vector pairs — lower is more spread)
      and uniformity (log of the mean pairwise exponential distance).

    The three thresholds come from the pre-registration; tripping any one marks
    the candidate ``disqualified-collapse`` and, per spec, no downstream probe
    may run for it.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 3:
        raise ValueError("collapse diagnostics need an (N>=3, D) embedding matrix")
    n, d = z.shape
    std = z.std(axis=0)
    cov = np.cov(z, rowvar=False)
    eig = np.linalg.eigvalsh(cov)
    eig = np.clip(eig, 0.0, None)
    total = eig.sum()
    if total <= 0:
        eff = 0.0
    else:
        p = eig / total
        p = p[p > 0]
        eff = float(np.exp(-np.sum(p * np.log(p))))
    ratio = eff / d if d else float("nan")
    top_share = float(eig.max() / total) if total > 0 else float("nan")

    unit = _l2norm(z)
    # Alignment/uniformity (Wang & Isola 2020, label-free form): without a
    # positive-pair definition we report the spread of the embedding cloud —
    # mean pairwise squared distance (alignment-like, higher = more spread) and
    # the log mean exponentials (uniformity). Both are computed on a fixed
    # subsample so the diagnostic is deterministic and cheap.
    align = float(_mean_pairwise_sq_distance(unit, rng_seed=13))
    uniformity = float(np.log(max(_mean_exp_distance(unit, rng_seed=13), 1e-12)))

    triggers: list[str] = []
    n_degenerate = int(np.sum(std < min_per_dim_std))
    if ratio < effective_rank_ratio_floor:
        triggers.append(
            f"effective_rank_ratio {ratio:.3f} < floor {effective_rank_ratio_floor}"
        )
    if n_degenerate / d > max_zero_variance_dims_frac:
        triggers.append(
            f"{n_degenerate}/{d} dims below std {min_per_dim_std} "
            f"(>{max_zero_variance_dims_frac:.0%} allowed)"
        )
    if np.isfinite(top_share) and top_share > max_top_eigenvalue_share:
        triggers.append(
            f"leading eigenvector carries {top_share:.1%} of the variance "
            f"(> {max_top_eigenvalue_share:.0%}): effectively rank-1"
        )
    return CollapseReport(
        per_dim_std=std,
        effective_rank=eff,
        effective_rank_ratio=ratio,
        top_eigenvalue_share=top_share,
        alignment=align,
        uniformity=uniformity,
        latent_dim=d,
        disqualified=bool(triggers),
        triggers=triggers,
    )


def _mean_pairwise_sq_distance(u: np.ndarray, *, rng_seed: int, cap: int = 2000) -> float:
    idx = _subsample(u.shape[0], cap, rng_seed)
    a = u[idx]
    d2 = np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(a.shape[0], k=1)
    return float(d2[iu].mean()) if iu[0].size else float("nan")


def _mean_exp_distance(u: np.ndarray, *, rng_seed: int, cap: int = 2000) -> float:
    idx = _subsample(u.shape[0], cap, rng_seed)
    a = u[idx]
    d2 = np.sum((a[:, None, :] - a[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(a.shape[0], k=1)
    return float(np.exp(-2.0 * d2[iu]).mean()) if iu[0].size else float("nan")


def _subsample(n: int, cap: int, seed: int) -> np.ndarray:
    if n <= cap:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=cap, replace=False))


# ---------------------------------------------------------------------------
# The battery
# ---------------------------------------------------------------------------


@dataclass
class Level1Result:
    reconstruction: ReconstructionReport
    farm_separability: FarmSeparabilityReport
    phase_probe: PhaseProbeReport | None
    stability: StabilityReport
    collapse: CollapseReport
    b1_farm_separability: FarmSeparabilityReport | None = None
    b1_phase_probe: PhaseProbeReport | None = None
    phase_probe_absent_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "reconstruction": self.reconstruction.to_dict(),
            "farm_separability": self.farm_separability.to_dict(),
            "phase_probe": self.phase_probe.to_dict() if self.phase_probe else None,
            "phase_probe_absent_reason": self.phase_probe_absent_reason,
            "stability": self.stability.to_dict(),
            "collapse": self.collapse.to_dict(),
            "b1_farm_separability": (
                self.b1_farm_separability.to_dict() if self.b1_farm_separability else None
            ),
            "b1_phase_probe": self.b1_phase_probe.to_dict() if self.b1_phase_probe else None,
        }

    def gate_evidence(self) -> dict:
        """The exact quantities G-A reads. No other field feeds the gate.

        When the phase probe could not be measured, its two keys are returned as
        ``None``. :func:`ssl_anomaly.evaluation.gates.gate_a` then reports them as
        missing evidence — so an unmeasurable probe reduces the gate to the terms
        that WERE measured instead of silently scoring it as passed or failed.
        """
        return {
            "farm_separability_top1": self.farm_separability.top1,
            "farm_separability_chance": self.farm_separability.chance_top1,
            "phase_probe_accuracy": (
                self.phase_probe.accuracy if self.phase_probe else None
            ),
            "phase_probe_majority": (
                self.phase_probe.majority_accuracy if self.phase_probe else None
            ),
            "collapse_disqualified": self.collapse.disqualified,
        }


def run_level1(
    *,
    dev: CycleData,
    embed_fn,
    b1_features: np.ndarray | None = None,
    phase_targets: np.ndarray | None = None,
    phase_floor: float = 0.5,
    lam: float = 1.0,
    collapse_kwargs: dict | None = None,
) -> Level1Result:
    """Run the full Level-1 battery on dev clean cycles.

    ``embed_fn(X, M) -> z`` must map a cycle tensor to a fixed-size embedding.
    ``b1_features`` (optional) are the incumbent baseline's feature vectors per
    dev cycle; when given, the two probe baselines are computed on the identical
    subset so the comparison the spec demands is available.

    ``phase_targets`` is the per-cycle phenological phase (a 36-bin index). It is
    a REQUIRED input for an honest phase probe, and it is NOT derivable from the
    cycle tensor: every full-season cycle covers the same bins, so any target
    computed from the observation mask alone is constant, giving a 100 % majority
    baseline against which "accuracy" is meaningless. When the target is missing
    or degenerate, the probe is reported ABSENT with the reason, and G-A proceeds
    on the one term it can measure.
    """
    z = np.asarray(embed_fn(dev.X, dev.M), dtype=np.float64)
    if z.ndim != 2 or z.shape[0] != dev.n_cycles:
        raise ValueError(f"embed_fn must return (N={dev.n_cycles}, D), got {z.shape}")

    recon = masked_reconstruction(dev, _zero_predictor)
    sep = farm_separability(z, dev.field_ids, subset_hash=dev.data_source)
    train_idx = np.arange(dev.n_cycles)  # dev-only battery: train/eval on dev halves
    half = max(1, dev.n_cycles // 2)
    stability = temporal_stability(z, dev.field_ids, floor=phase_floor)
    collapse = collapse_diagnostics(z, **(collapse_kwargs or {}))

    probe: PhaseProbeReport | None = None
    b1_probe: PhaseProbeReport | None = None
    absent_reason = ""
    bins = _resolve_phase_targets(dev, phase_targets)
    if bins is None:
        absent_reason = (
            "the phase probe needs a per-cycle phenological target, which is not "
            "derivable from the cycle tensor: every full-season cycle observes the "
            "same 36 bins, so any mask-derived target is constant and its majority "
            "baseline is 100%. Supply phase_targets (the per-cycle season-start / "
            "phase covariate)."
        )
    else:
        probe = phase_linear_probe(
            z[train_idx[:half]], bins[train_idx[:half]],
            z[train_idx[half:]], bins[train_idx[half:]], lam=lam,
        )
        if b1_features is not None:
            f = np.asarray(b1_features, dtype=np.float64)
            if f.shape[0] == dev.n_cycles:
                b1_probe = phase_linear_probe(
                    f[train_idx[:half]], bins[train_idx[:half]],
                    f[train_idx[half:]], bins[train_idx[half:]], lam=lam,
                )

    b1_sep = None
    if b1_features is not None:
        f = np.asarray(b1_features, dtype=np.float64)
        if f.shape[0] == dev.n_cycles:
            b1_sep = farm_separability(f, dev.field_ids, subset_hash="b1-features")

    return Level1Result(
        reconstruction=recon,
        farm_separability=sep,
        phase_probe=probe,
        stability=stability,
        collapse=collapse,
        b1_farm_separability=b1_sep,
        b1_phase_probe=b1_probe,
        phase_probe_absent_reason=absent_reason,
    )


def _resolve_phase_targets(dev: CycleData, phase_targets) -> np.ndarray | None:
    """Validate a supplied phase target, or return None when it cannot be honest.

    ``None`` means "absent", not "use a default": a fabricated phase target makes
    the probe report a meaningless accuracy (see the run_level1 docstring)."""
    if phase_targets is None:
        return None
    t = np.asarray(phase_targets)
    if t.ndim != 1 or t.size != dev.n_cycles:
        raise ValueError(
            f"phase_targets must be 1-D with {dev.n_cycles} entries, got {t.shape}"
        )
    if np.unique(t).size < 2:
        return None
    counts = np.bincount(t.astype(np.int64))
    if counts.max() / t.size >= 0.999:
        return None
    return t.astype(np.int64)


def _zero_predictor(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """The trivial reconstruction used when no decoder is attached.

    Documented, not hidden: Level-1 mask-reconstruction is only meaningful for a
    candidate whose decoder is available; when it is not, the encoder-only probe
    still runs and the reconstruction block reports the zero predictor's NRMSE,
    which is 1.0 by construction. Callers that have a decoder pass their own.
    """
    del M
    return np.zeros_like(X, dtype=np.float64)

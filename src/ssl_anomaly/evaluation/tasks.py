"""The five task battery T1–T5 (design §Tasks).

Every task returns a :class:`TaskResult` whose ``status`` is one of
``measured`` / ``absent`` / ``deferred``. There is no fourth path and no task may
return a number when its status is not ``measured`` — the value type physically
forbids it (``value is None`` unless measured).

======  =============================================  =====================
task    question                                       claim bearing
======  =============================================  =====================
T1      phase estimation (DSH within ±15 days)          yes (H1)
T2      anomaly detection at farm level (masked)        yes (H1, H3)
T3      representation retrieval, two tracks           yes (H2) — see below
T4      forecasting of future observed bins            DEFERRED (needs the
                                                        pre-registered hour
                                                        budget; design.md:329)
T5      anomaly localisation in time                    exploratory only
======  =============================================  =====================

T3 is the trap. The design allows **two** retrieval definitions and they answer
different questions:

- **Track A (cycle retrieval)** — "find another pass of the SAME field": the
  gallery contains cycles from the query's own field, and the metric is whether
  the nearest hit shares its farm. This measures *within-field consistency*
  (a representation that never varies within a field scores 1.0 here, so it is
  paired with the temporal-stability probe).
- **Track B (field retrieval)** — "find a field like this one": the query's own
  field is REMOVED from the gallery, and the metric is whether the nearest field
  is agronomically similar (same DSH phase and crop class). *This* is the track
  that supports H2.

Conflating them inflates the result: track A is trivially high for any smoothed
embedding, track B is the informative one. The code below refuses to report a
single number when both tracks ran.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ssl_anomaly.evaluation.metrics import wilson_interval

TASKS = ("T1", "T2", "T3", "T4", "T5")

TASK_TABLE = {
    "T1": {"name": "phase estimation", "claim_bearing": True, "hypotheses": ["H1"]},
    "T2": {"name": "anomaly detection", "claim_bearing": True, "hypotheses": ["H1", "H3"]},
    "T3": {"name": "representation retrieval", "claim_bearing": True, "hypotheses": ["H2"]},
    "T4": {"name": "forecasting", "claim_bearing": True, "hypotheses": ["H4"]},
    "T5": {"name": "anomaly localisation", "claim_bearing": False, "hypotheses": []},
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    """One task on one rung/representation. ``value`` exists only if measured."""

    task: str
    subject: str                     # rung or candidate name
    status: str                      # 'measured' | 'absent' | 'deferred'
    value: dict | None = None
    ci: dict | None = None
    n: int = 0
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status == "measured" and self.value is None:
            raise ValueError(f"{self.task}/{self.subject}: measured requires a value")
        if self.status != "measured" and self.value is not None:
            raise ValueError(
                f"{self.task}/{self.subject}: a {self.status} task must not carry a value"
            )
        if self.status not in {"measured", "absent", "deferred"}:
            raise ValueError(f"unknown task status {self.status!r}")

    @classmethod
    def deferred(cls, task: str, subject: str, reason: str) -> TaskResult:
        return cls(task=task, subject=subject, status="deferred", reason=reason)

    @classmethod
    def absent(cls, task: str, subject: str, reason: str) -> TaskResult:
        return cls(task=task, subject=subject, status="absent", reason=reason)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "subject": self.subject,
            "status": self.status,
            "value": self.value,
            "ci": self.ci,
            "n": self.n,
            "reason": self.reason,
            "diagnostics": self.diagnostics,
        }

    @property
    def quotable(self) -> bool:
        """Only a measured, claim-bearing task may enter the findings."""
        return self.status == "measured" and TASK_TABLE[self.task]["claim_bearing"]


# ---------------------------------------------------------------------------
# T1 — phase estimation
# ---------------------------------------------------------------------------


@dataclass
class PhaseTolerance:
    days: float = 15.0
    n_phase_bins: int = 36

    def to_dict(self) -> dict:
        return {"days": self.days, "n_phase_bins": self.n_phase_bins}


def t1_phase_estimation(
    predicted_dsh: np.ndarray,
    true_dsh: np.ndarray,
    *,
    subject: str,
    tolerance: PhaseTolerance | None = None,
) -> TaskResult:
    """Share of cycles whose predicted DSH is within ±``tolerance.days``.

    Also reports MAE and a bias term, because a model can be accurate on average
    while systematically early (a bias that matters agronomically). The tolerance
    hit rate gets a Wilson interval; the MAE gets a bootstrap over cycles.
    """
    tol = tolerance or PhaseTolerance()
    p = np.asarray(predicted_dsh, dtype=np.float64).ravel()
    t = np.asarray(true_dsh, dtype=np.float64).ravel()
    if p.size != t.size:
        raise ValueError(f"prediction {p.shape} and truth {t.shape} must align")
    if p.size == 0:
        return TaskResult.absent("T1", subject, "no cycles supplied")
    err = p - t
    # Inclusive boundary with a float guard: 15.0 must count at a 15-day
    # tolerance, and binary representation makes `abs(15.0-0.0) <= 15.0` fragile.
    eps = 1e-9 * max(1.0, tol.days)
    hits = int(np.sum(np.abs(err) <= tol.days + eps))
    lo, hi = wilson_interval(hits, p.size)
    rng = np.random.default_rng(13)
    boot = np.array([
        np.mean(np.abs(err[rng.integers(0, err.size, err.size)]))
        for _ in range(500)
    ])
    mae = float(np.mean(np.abs(err)))
    return TaskResult(
        task="T1", subject=subject, status="measured", n=int(p.size),
        value={
            "within_tolerance_share": hits / p.size,
            "mae_days": mae,
            "bias_days": float(np.mean(err)),
            "tolerance_days": tol.days,
        },
        ci={
            "within_tolerance": [lo, hi],
            "mae_days": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
        },
    )


# ---------------------------------------------------------------------------
# T2 — anomaly detection
# ---------------------------------------------------------------------------


def t2_anomaly_detection(
    scores: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    *,
    subject: str,
    threshold: float | None = None,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
) -> TaskResult:
    """Masked anomaly detection: AP + prevalence, with F1 only from a VALIDATION threshold.

    ``validation=(val_scores, val_labels)`` is mandatory to report F1 — the spec
    forbids selecting a threshold on the evaluated data. Without it, F1 is
    ``absent`` (reported as a null inside a measured result's diagnostics) rather
    than tuned on the test it is scored on.
    """
    from ssl_anomaly.evaluation.metrics import (
        auprc,
        auroc,
        f1_at_threshold,
        select_threshold_on_validation,
    )

    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).astype(bool).ravel()
    if s.size != y.size:
        raise ValueError("scores and labels must align")
    if s.size == 0:
        return TaskResult.absent("T2", subject, "no evaluation rows")
    if y.sum() == 0 or (~y).sum() == 0:
        return TaskResult.absent("T2", subject, "evaluation set has a single class")

    ap, prev = auprc(s, y)
    roc = auroc(s, y)
    value = {"ap": ap, "prevalence": prev, "auroc": roc}
    diag: dict = {}

    if validation is not None:
        vs, vy = np.asarray(validation[0]), np.asarray(validation[1]).astype(bool)
        thr = select_threshold_on_validation(vs, vy) if threshold is None else threshold
        value["f1"] = f1_at_threshold(s, y, thr)["f1"]
        value["threshold"] = float(thr)
        diag["threshold_source"] = "validation split"
    else:
        diag["f1"] = None
        diag["threshold_source"] = (
            "absent: no validation split supplied, and selecting on the evaluated "
            "scores is forbidden by the pre-registration"
        )

    # Clustered bootstrap over farms/events for AP.
    from ssl_anomaly.evaluation.metrics import bootstrap_ci_stat

    c = np.asarray(clusters)
    _, lo, hi = bootstrap_ci_stat(
        y.astype(int), s, lambda yy, ss: auprc(ss, yy)[0],
        n_boot=500, seed=13, clusters=c if c.size == y.size else None,
    )
    return TaskResult(
        task="T2", subject=subject, status="measured", n=int(s.size),
        value=value, ci={"ap": [lo, hi]}, diagnostics=diag,
    )


# ---------------------------------------------------------------------------
# T3 — retrieval, two tracks
# ---------------------------------------------------------------------------


@dataclass
class RetrievalReport:
    """Precision@k / mAP for ONE track. Two tracks never merge into one number."""

    track: str
    k_values: tuple[int, ...]
    precision_at_k: dict[int, float]
    map_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    n_queries: int
    n_relevant_mean: float
    chance_precision_at_k: dict[int, float]

    def to_dict(self) -> dict:
        return {
            "track": self.track,
            "k_values": list(self.k_values),
            "precision_at_k": {str(k): v for k, v in self.precision_at_k.items()},
            "map_at_k": {str(k): v for k, v in self.map_at_k.items()},
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "n_queries": self.n_queries,
            "n_relevant_mean": self.n_relevant_mean,
            "chance_precision_at_k": {str(k): v for k, v in self.chance_precision_at_k.items()},
            "note": TRACK_NOTES[self.track],
        }


TRACK_NOTES = {
    "A-cycle": ("retrieve another pass of the SAME field (own field kept in the "
                "gallery). Measures within-field consistency; a smoothed embedding "
                "scores high here without carrying any field-level information."),
    "B-field": ("retrieve a DIFFERENT field that is agronomically similar (own field "
                "removed from the gallery). This track supports H2; its chance level "
                "is computed over the remaining fields, not over all fields."),
}


def _rank_order(query: np.ndarray, gallery: np.ndarray, *, exclude: np.ndarray | None = None):
    q = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    sim = q @ g.T
    if exclude is not None:
        sim[exclude] = -np.inf
    return np.argsort(-sim, axis=1)


def t3_retrieval_track_a(
    embeddings: np.ndarray,
    field_ids: np.ndarray,
    *,
    subject: str,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> TaskResult:
    """Track A: for each cycle, are the nearest neighbours cycles of its OWN field?"""
    z = np.asarray(embeddings, dtype=np.float64)
    f = np.asarray(field_ids)
    if z.ndim != 2 or z.shape[0] != f.size:
        raise ValueError("embeddings must be (N, D) aligned with field ids")
    if z.shape[0] < 3:
        return TaskResult.absent("T3", subject, "need >= 3 cycles for retrieval")
    n = z.shape[0]
    # Self-exclusion is ROLE-based, not score-based: a query must never be its own
    # relevant hit, otherwise precision@1 is 1.0 for every embedding by
    # construction (the query is its own nearest neighbour with cosine 1).
    exclude = np.zeros((n, n), dtype=bool)
    np.fill_diagonal(exclude, True)
    order = _rank_order(z, z, exclude=exclude)
    rel = (f[order] == f[:, None]) & ~exclude[np.arange(n)[:, None], order]
    n_rel = rel.sum(axis=1)
    # Undefined queries are EXCLUDED and counted, never scored as zero: a field
    # with a single cycle has no relevant gallery item, so its AP has a zero
    # denominator. Silently including it would drag the mean down by an artefact
    # of sampling rather than of representation quality.
    usable = n_rel > 0
    n_excluded = int((~usable).sum())
    if not usable.any():
        return TaskResult.absent(
            "T3", subject,
            "no query has a relevant item (every field has a single cycle); "
            "track A is undefined",
        )
    rel_u, n_rel_u = rel[usable], n_rel[usable]
    p_at_k, map_at_k, rec_at_k = {}, {}, {}
    for k in k_values:
        kk = min(k, rel_u.shape[1])
        p_at_k[k] = float(np.mean(rel_u[:, :kk].mean(axis=1)))
        prec = np.cumsum(rel_u[:, :kk], axis=1) / np.arange(1, kk + 1)
        ap = np.sum(prec * rel_u[:, :kk], axis=1) / np.minimum(n_rel_u, kk)
        map_at_k[k] = float(np.mean(ap))
        rec_at_k[k] = float(np.mean(rel_u[:, :kk].sum(axis=1) / n_rel_u))
    chance = {k: float(np.mean(np.minimum(n_rel_u, k) / (n - 1))) for k in k_values}
    return TaskResult(
        task="T3", subject=subject, status="measured", n=int(z.shape[0]),
        value=RetrievalReport(
            track="A-cycle", k_values=k_values, precision_at_k=p_at_k,
            map_at_k=map_at_k, recall_at_k=rec_at_k, n_queries=int(usable.sum()),
            n_relevant_mean=float(n_rel_u.mean()), chance_precision_at_k=chance,
        ).to_dict(),
        diagnostics={
            "track": "A-cycle",
            "does_not_support": "H2 (see track B)",
            "n_queries_excluded": n_excluded,
            "exclusion_reason": ("field has a single cycle, so it has no relevant "
                                 "gallery item (undefined AP, not zero)"),
        },
    )


def t3_retrieval_track_b(
    embeddings: np.ndarray,
    field_ids: np.ndarray,
    field_phase: np.ndarray,
    field_crop: np.ndarray,
    *,
    subject: str,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    phase_tolerance_bins: int = 2,
) -> TaskResult:
    """Track B: own field REMOVED; is the nearest other field agronomically similar?

    Relevance = same crop class AND |phase difference| <= ``phase_tolerance_bins``
    on the 36-bin DSH grid. ``field_phase`` and ``field_crop`` are one entry per
    FIELD (indexed in the order of ``np.unique(field_ids)``), not per cycle.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    f = np.asarray(field_ids)
    phase = np.asarray(field_phase, dtype=np.float64)
    crop = np.asarray(field_crop)
    fields = np.unique(f)
    if phase.size != fields.size or crop.size != fields.size:
        raise ValueError(
            f"field_phase ({phase.size}) and field_crop ({crop.size}) must have one "
            f"entry per field ({fields.size})"
        )
    if fields.size < 3:
        return TaskResult.absent("T3", subject, "track B needs >= 3 distinct fields")
    # Collapse to one vector per field (mean of its cycles): the gallery must be
    # field-level, otherwise removing "the own field" would remove only one cycle.
    centroids = np.stack([z[f == fid].mean(axis=0) for fid in fields])
    order = np.argsort(-(centroids @ centroids.T), axis=1)
    order = np.array([row[row != i] for i, row in enumerate(order)], dtype=object)

    relevant_map = np.zeros((fields.size, fields.size), dtype=bool)
    for i in range(fields.size):
        for j in range(fields.size):
            if i == j:
                continue
            relevant_map[i, j] = (
                crop[i] == crop[j]
                and abs(phase[i] - phase[j]) <= phase_tolerance_bins
            )
    n_rel = relevant_map.sum(axis=1)
    if np.any(n_rel == 0):
        return TaskResult.absent(
            "T3", subject,
            "at least one field has no relevant partner under the similarity rule "
            "(same crop and phase within tolerance); the retrieval question is "
            "undefined for it",
        )

    p_at_k, map_at_k, rec_at_k, chance = {}, {}, {}, {}
    for k in k_values:
        kk = min(k, fields.size - 1)
        ps, aps, rs = [], [], []
        for i in range(fields.size):
            ranked = order[i][:kk].astype(int)
            hits = relevant_map[i, ranked]
            ps.append(hits.mean())
            prec = np.cumsum(hits) / np.arange(1, kk + 1)
            aps.append(np.sum(prec * hits) / min(n_rel[i], kk))
            rs.append(hits.sum() / n_rel[i])
        p_at_k[k] = float(np.mean(ps))
        map_at_k[k] = float(np.mean(aps))
        rec_at_k[k] = float(np.mean(rs))
        chance[k] = float(np.mean(np.minimum(n_rel, k) / (fields.size - 1)))
    return TaskResult(
        task="T3", subject=subject, status="measured", n=int(z.shape[0]),
        value=RetrievalReport(
            track="B-field", k_values=k_values, precision_at_k=p_at_k, map_at_k=map_at_k,
            recall_at_k=rec_at_k, n_queries=int(fields.size),
            n_relevant_mean=float(n_rel.mean()), chance_precision_at_k=chance,
        ).to_dict(),
        diagnostics={"track": "B-field", "supports": "H2"},
    )


def t3_retrieval(
    embeddings: np.ndarray,
    field_ids: np.ndarray,
    *,
    subject: str,
    field_phase: np.ndarray | None = None,
    field_crop: np.ndarray | None = None,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> dict:
    """Run both tracks and return them SEPARATELY.

    The return value is a dict keyed by track — deliberately not a single
    ``TaskResult`` — so no caller can quote "the retrieval score" without saying
    which question it answers. Track B is ``absent`` (with a reason) when the
    field-level agronomy labels are missing.
    """
    out = {
        "A-cycle": t3_retrieval_track_a(embeddings, field_ids, subject=subject,
                                        k_values=k_values).to_dict(),
    }
    if field_phase is None or field_crop is None:
        out["B-field"] = TaskResult.absent(
            "T3", subject,
            "field-level phase/crop labels were not supplied; track B (the H2-bearing "
            "track) cannot be computed",
        ).to_dict()
    else:
        out["B-field"] = t3_retrieval_track_b(
            embeddings, field_ids, field_phase, field_crop, subject=subject,
            k_values=k_values,
        ).to_dict()
    out["_contract"] = (
        "two tracks, two questions: A-cycle = within-field consistency (not H2), "
        "B-field = agronomic similarity with the own field removed (H2)"
    )
    return out


# ---------------------------------------------------------------------------
# T4 — forecasting (deferred)
# ---------------------------------------------------------------------------

#: The design.md:329 open decision: the hour budget for T4 has NOT been set, so
#: the task cannot be run under a pre-registered budget and is deferred.
T4_HOUR_BUDGET_CONFIGURED = False


def t4_forecasting(*, subject: str, hour_budget: float | None = None) -> TaskResult:
    """Forecasting is deferred until a pre-registered compute budget exists.

    The design fixes T4's *metric* (masked NRMSE of future observed bins) but
    leaves the hour budget open (design.md:329). Running it on an unstated budget
    would make the result unreproducible and the runtime comparison meaningless,
    so this returns ``deferred`` regardless of data availability, and names the
    decision that must be made.
    """
    if hour_budget is None or not T4_HOUR_BUDGET_CONFIGURED:
        return TaskResult.deferred(
            "T4", subject,
            "no pre-registered hour budget (design.md:329 leaves it open); the "
            "metric is defined but a run without a declared budget cannot be "
            "reproduced or compared on compute",
        )
    return TaskResult.deferred(
        "T4", subject, "hour budget declared but the forecasting harness is out of scope"
    )


# ---------------------------------------------------------------------------
# T5 — anomaly localisation (exploratory)
# ---------------------------------------------------------------------------


def t5_localisation(
    predicted_bin: np.ndarray,
    true_bin: np.ndarray,
    *,
    subject: str,
    tolerance_bins: int = 1,
) -> TaskResult:
    """Timing error of the localised anomaly, in bins. EXPLORATORY, never claim-bearing.

    ``TASK_TABLE['T5']['claim_bearing']`` is ``False``, so even a measured T5
    cannot be quoted as evidence; :attr:`TaskResult.quotable` returns False and
    the provenance layer refuses it a rung.
    """
    p = np.asarray(predicted_bin, dtype=np.float64).ravel()
    t = np.asarray(true_bin, dtype=np.float64).ravel()
    if p.size != t.size:
        raise ValueError("predicted and true bins must align")
    usable = np.isfinite(p) & np.isfinite(t)
    if usable.sum() == 0:
        return TaskResult.absent("T5", subject, "no cycle had both a predicted and a true bin")
    err = np.abs(p[usable] - t[usable])
    hits = int(np.sum(err <= tolerance_bins))
    lo, hi = wilson_interval(hits, int(usable.sum()))
    return TaskResult(
        task="T5", subject=subject, status="measured", n=int(usable.sum()),
        value={
            "within_tolerance_share": hits / usable.sum(),
            "mae_bins": float(err.mean()),
            "tolerance_bins": tolerance_bins,
            "n_unusable": int((~usable).sum()),
        },
        ci={"within_tolerance": [lo, hi]},
        diagnostics={"claim_bearing": False,
                     "note": "exploratory: reported, never used as evidence"},
    )


def task_table() -> list[dict]:
    return [
        {"task": t, **TASK_TABLE[t]} for t in TASKS
    ]

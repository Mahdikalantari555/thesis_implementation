"""Transfer curves over label fractions: the 6-arm protocol (design §Label efficiency).

Six arms, all reading the SAME frozen representation except A1 (random init) and
A6 (raw features):

The six arms are the pre-registered ones (``configs/evaluation/pre_registration.yaml``
``arms:``). Implementing a different six would make every downstream gate read the
wrong comparisons, so the names here are the frozen names:

===============  ===========================================  ==============
arm              what is trained                               frozen encoder
===============  ===========================================  ==============
A-rule           FAR-matched NDVI+LST rule (the B1 rule)       n/a (no learning)
A-B1             incumbent b1_phasez_causal score              yes (frozen model)
A-rand           probe on a randomly initialised encoder       no
A-scratch        task-label-only small model, same trunk       no
A-ssl-frozen     linear probe on the frozen SSL embeddings     yes
A-ssl-ft         fine-tune the SSL encoder + head              no
===============  ===========================================  ==============

Fractions: 5/10/25/50/100 % of TRAIN labels, each with 3 repeats
(pre-registered), evaluated on dev. Curves are reported per fraction with a
clustered bootstrap over farms/events so a reader can see whether the shape of
the curve (not just one point) supports the label-efficiency claim.

Honest failure modes encoded here, not papered over:

- A fraction with fewer than 2 labelled cycles per class cannot train a probe;
  it returns ``absent`` with the reason, never a fabricated number.
- A4/A5 on a representation with no gradient path (e.g. a NumPy feature matrix)
  return ``absent`` rather than silently degrading to A2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ssl_anomaly.evaluation.metrics import (
    auprc,
    bootstrap_ci_stat,
    f1_at_threshold,
    select_threshold_on_validation,
)

ARMS = ("A-rule", "A-B1", "A-rand", "A-scratch", "A-ssl-frozen", "A-ssl-ft")

#: ``(mechanism, description, output_is_frozen)`` per arm.
ARM_DESCRIPTIONS = {
    "A-rule": ("rule", "FAR-matched NDVI+LST rule (B1 rung), no learning", False),
    "A-B1": ("frozen-scores", "incumbent b1_phasez_causal scores, read frozen", True),
    "A-rand": ("probe", "probe on a randomly initialised encoder of the same trunk", True),
    "A-scratch": ("scratch", "task-label-only model, same trunk, fresh init", False),
    "A-ssl-frozen": ("logistic", "linear probe on the frozen SSL embeddings", True),
    "A-ssl-ft": ("finetune", "SSL encoder fine-tuned with the head", False),
}


def arm_table() -> list[dict]:
    """The declared arm matrix, for the manifest and for the report table."""
    return [
        {"arm": a, "mechanism": ARM_DESCRIPTIONS[a][0],
         "description": ARM_DESCRIPTIONS[a][1],
         "inputs_frozen": ARM_DESCRIPTIONS[a][2]}
        for a in ARMS
    ]


# ---------------------------------------------------------------------------
# Fraction sampling
# ---------------------------------------------------------------------------


@dataclass
class LabelDraw:
    """One (fraction, repeat) label draw — stratified by farm, recorded."""

    fraction: float
    repeat: int
    train_idx: np.ndarray
    seed: int
    n_available: int
    stratifier: str = "field_id"
    n_fields: int = 0

    def to_dict(self) -> dict:
        return {
            "fraction": self.fraction,
            "repeat": self.repeat,
            "seed": self.seed,
            "n_labels": int(self.train_idx.size),
            "n_available": int(self.n_available),
            "realised_fraction": (float(self.train_idx.size / self.n_available)
                                  if self.n_available else float("nan")),
            "n_fields": int(self.n_fields),
            "stratifier": self.stratifier,
            "label_indices_sha256_prefix": _hash_indices(self.train_idx),
        }


def _hash_indices(idx: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(np.asarray(idx, dtype=np.int64).tobytes()).hexdigest()[:8]


def stratified_draw(
    field_ids: np.ndarray,
    fraction: float,
    *,
    repeat: int,
    base_seed: int = 13,
    min_per_field: int = 1,
) -> LabelDraw:
    """Draw ``fraction`` of the labelled cycles, stratified so every farm appears.

    **The stratification floor is a real distortion and is recorded, not hidden.**
    Because every farm must appear, a small nominal fraction over-represents
    labels whenever farms are few relative to cycles: 40 cycles over 20 farms at
    nominal 5% yields 20 labels (realised 50%). ``LabelDraw.to_dict`` therefore
    carries ``realised_fraction``, and every curve cell carries the realised
    count, so the x-axis is never silently the nominal fraction.

    A fraction too small to place ``min_per_field`` cycle in every farm is not
    silently inflated beyond one per farm — that IS the floor — so a genuinely
    starved draw (fewer than 4 labels) is reported ``absent`` by the caller.
    """
    f = np.asarray(field_ids)
    n = f.size
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    rng = np.random.default_rng(base_seed + 1000 * repeat + int(round(fraction * 100)))
    n_per_field = max(min_per_field, int(np.floor(fraction * n / max(np.unique(f).size, 1))))
    chosen: list[int] = []
    for fid in np.unique(f):
        idx = np.flatnonzero(f == fid)
        take = min(n_per_field, idx.size)
        chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
    out = np.sort(np.asarray(chosen, dtype=np.int64))
    return LabelDraw(fraction=fraction, repeat=repeat, train_idx=out, seed=base_seed,
                     n_available=n, n_fields=int(np.unique(f).size))


# ---------------------------------------------------------------------------
# Probes (numpy-only, closed form where possible)
# ---------------------------------------------------------------------------


def _ridge_logistic(
    X: np.ndarray, y: np.ndarray, *, lam: float = 1.0, n_iter: int = 200, lr: float = 0.5
) -> np.ndarray:
    """L2-regularised logistic regression by full-batch Newton-free gradient steps.

    Deterministic (no random init: starts at zero) and dependency-free, so the
    probe's numbers cannot drift with a library update. ``y`` must be 0/1.
    """
    n, d = X.shape
    Xa = np.hstack([X, np.ones((n, 1))])
    w = np.zeros(d + 1)
    reg = lam * np.eye(d + 1)
    reg[-1, -1] = 0.0
    for _ in range(n_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(Xa @ w, -30, 30)))
        grad = Xa.T @ (p - y) / n + reg @ w / n
        w -= lr * grad
    return w


def _predict_logistic(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])
    return 1.0 / (1.0 + np.exp(-np.clip(Xa @ w, -30, 30)))


def _mlp_head(
    X: np.ndarray, y: np.ndarray, *, hidden: int = 32, epochs: int = 120, lr: float = 0.05,
    seed: int = 13,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A small 2-layer MLP trained with plain gradient descent (numpy only)."""
    rng = np.random.default_rng(seed)
    n, d = X.shape
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, np.sqrt(2.0 / hidden), (hidden, 1))
    b2 = np.zeros(1)
    for _ in range(epochs):
        h = np.maximum(X @ W1 + b1, 0.0)
        logit = (h @ W2 + b2).ravel()
        p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
        g = (p - y)[:, None] / n
        gW2 = h.T @ g
        gb2 = g.sum(0)
        gh = (g @ W2.T) * (h > 0)
        gW1 = X.T @ gh
        gb1 = gh.sum(0)
        W1 -= lr * gW1
        b1 -= lr * gb1
        W2 -= lr * gW2
        b2 -= lr * gb2
    return W1, b1, W2, b2


def _predict_mlp(params, X):
    W1, b1, W2, b2 = params
    h = np.maximum(X @ W1 + b1, 0.0)
    return 1.0 / (1.0 + np.exp(-np.clip((h @ W2 + b2).ravel(), -30, 30)))


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ArmResult:
    """One (arm, fraction) cell of the transfer grid."""

    arm: str
    fraction: float
    repeat: int
    status: str                    # 'measured' | 'absent' | 'deferred'
    ap: float | None = None
    prevalence: float | None = None
    f1: float | None = None
    threshold: float | None = None
    auroc: float | None = None
    ap_ci: tuple[float, float] | None = None
    f1_ci: tuple[float, float] | None = None
    n_train: int = 0
    n_eval: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "fraction": self.fraction,
            "repeat": self.repeat,
            "status": self.status,
            "ap": self.ap,
            "prevalence": self.prevalence,
            "f1": self.f1,
            "threshold": self.threshold,
            "auroc": self.auroc,
            "ap_ci": list(self.ap_ci) if self.ap_ci else None,
            "f1_ci": list(self.f1_ci) if self.f1_ci else None,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "reason": self.reason,
        }

    @property
    def measurable(self) -> bool:
        return self.status == "measured"


@dataclass
class Curve:
    """A transfer curve for one arm: AP vs label fraction."""

    arm: str
    cells: list[ArmResult] = field(default_factory=list)

    def points(self) -> list[dict]:
        """Measured cells only — an absent fraction is not a point at zero."""
        return [
            {"fraction": c.fraction, "ap": c.ap, "repeat": c.repeat}
            for c in self.cells if c.measurable
        ]

    def fractions_reported(self) -> list[float]:
        return sorted({c.fraction for c in self.cells})

    def absent_fractions(self) -> list[dict]:
        return [
            {"fraction": c.fraction, "repeat": c.repeat, "reason": c.reason}
            for c in self.cells if not c.measurable
        ]

    def to_dict(self) -> dict:
        return {
            "arm": self.arm,
            "points": self.points(),
            "fractions_reported": self.fractions_reported(),
            "absent": self.absent_fractions(),
            "cells": [c.to_dict() for c in self.cells],
        }

    def area_under_curve(self) -> float | None:
        """Trapezoid over measured fractions. ``None`` if fewer than 2 points.

        Deliberately undefined (not zero, not interpolated) when a curve is
        incomplete: an AUC computed over 2 of 5 fractions is not the quantity the
        pre-registration names.
        """
        pts = sorted(self.points(), key=lambda p: p["fraction"])
        if len(pts) < 2:
            return None
        xs = np.array([p["fraction"] for p in pts])
        ys = np.array([p["ap"] for p in pts])
        return float(np.trapezoid(ys, xs))


# ---------------------------------------------------------------------------
# The protocol
# ---------------------------------------------------------------------------


@dataclass
class TransferConfig:
    fractions: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 1.00)
    repeats: int = 3
    base_seed: int = 13
    lam: float = 1.0
    n_boot: int = 500
    bootstrap_unit: str = "field"

    def to_dict(self) -> dict:
        return {
            "fractions": list(self.fractions),
            "repeats": self.repeats,
            "base_seed": self.base_seed,
            "lam": self.lam,
            "n_boot": self.n_boot,
            "bootstrap_unit": self.bootstrap_unit,
        }


def _evaluate_scores(
    scores: np.ndarray, y: np.ndarray, clusters: np.ndarray, cfg: TransferConfig
) -> tuple[float, float, float, tuple[float, float], tuple[float, float]]:
    """AP + AUROC + threshold-tuned F1 with clustered bootstrap CIs."""
    from ssl_anomaly.evaluation.metrics import auroc as _auroc

    ap, prev = auprc(scores, y)
    roc = _auroc(scores, y)
    # Threshold selection belongs to VALIDATION. Inside a transfer arm there is no
    # held-out validation split, so the threshold is chosen on the same dev scores
    # and every F1 is flagged as optimistic by construction (see ArmResult.reason
    # contract in run_arm). AP is threshold-free and is the headline quantity.
    thr = select_threshold_on_validation(scores, y)
    f1 = f1_at_threshold(scores, y, thr)["f1"]

    def _ap(yy, ss):
        return auprc(ss, yy)[0]

    def _f1(yy, ss):
        return f1_at_threshold(ss, yy, select_threshold_on_validation(ss, yy))["f1"]

    _, ap_lo, ap_hi = bootstrap_ci_stat(
        y, scores, _ap, n_boot=cfg.n_boot, seed=cfg.base_seed,
        clusters=clusters if cfg.bootstrap_unit == "field" else None,
    )
    _, f1_lo, f1_hi = bootstrap_ci_stat(
        y, scores, _f1, n_boot=cfg.n_boot, seed=cfg.base_seed + 1,
        clusters=clusters if cfg.bootstrap_unit == "field" else None,
    )
    return ap, prev, roc, (ap_lo, ap_hi), (f1_lo, f1_hi), f1, thr


def run_arm(
    arm: str,
    *,
    embeddings: np.ndarray,
    labels: np.ndarray,
    field_ids: np.ndarray,
    eval_embeddings: np.ndarray,
    eval_labels: np.ndarray,
    eval_field_ids: np.ndarray,
    cfg: TransferConfig,
    random_init_embeddings: np.ndarray | None = None,
    features: np.ndarray | None = None,
    rule_scores: np.ndarray | None = None,
    b1_scores: np.ndarray | None = None,
    finetune_fn=None,
    scratch_fn=None,
) -> Curve:
    """Run one arm across all fractions and repeats.

    ``finetune_fn(train_idx, eval_X) -> (train_scores, eval_scores)`` is required
    for A4/A5 and absent otherwise; when missing those arms record ``absent``
    with an explicit reason (they are not silently downgraded to a probe).
    ``random_init_embeddings`` is the A1 control; ``features`` is A6.

    Arm inputs (all optional except the SSL embeddings):

    - ``A-rule`` needs ``rule_scores`` (per-cycle rule scores; no learning, so the
      label fraction is irrelevant and the SAME curve is reported at every
      fraction — the honest shape for a rule with no training).
    - ``A-B1`` needs ``b1_scores`` (frozen incumbent scores).
    - ``A-rand`` needs ``random_init_embeddings``.
    - ``A-scratch`` needs ``scratch_fn`` (a model trained from scratch per draw).
    - ``A-ssl-frozen`` probes ``embeddings``.
    - ``A-ssl-ft`` needs ``finetune_fn`` (a differentiable encoder).

    An arm whose inputs are missing is ``absent`` with the reason — never silently
    downgraded to a probe, because that would change what the arm measures while
    keeping its name.
    """
    curve = Curve(arm=arm)
    kind, desc, inputs_frozen = ARM_DESCRIPTIONS[arm]

    if kind in {"finetune", "scratch"} and finetune_fn is None and scratch_fn is None:
        curve.cells.append(
            ArmResult(arm=arm, fraction=0.0, repeat=-1, status="absent",
                      reason=f"{desc}: no differentiable training callback was supplied "
                             f"to this run")
        )
        return curve

    # ``source`` is the TRAINING matrix; ``eval_source`` is the matching
    # evaluation matrix. Each arm must probe the representation it declares.
    source = embeddings
    eval_source = eval_embeddings
    if arm == "A-rand":
        if random_init_embeddings is None:
            curve.cells.append(
                ArmResult(arm=arm, fraction=0.0, repeat=-1, status="absent",
                          reason="A-rand needs the same trunk randomly initialised; "
                                 "random_init_embeddings was not supplied")
            )
            return curve
        source = random_init_embeddings
        eval_source = random_init_embeddings
    elif arm in {"A-rule", "A-B1"}:
        key = "rule_scores" if arm == "A-rule" else "b1_scores"
        scores = rule_scores if arm == "A-rule" else b1_scores
        if scores is None:
            curve.cells.append(
                ArmResult(arm=arm, fraction=0.0, repeat=-1, status="absent",
                          reason=f"{desc}: {key} was not supplied")
            )
            return curve
        return _curve_from_fixed_scores(
            arm, np.asarray(scores, dtype=np.float64), labels, eval_labels,
            eval_field_ids, cfg,
        )

    if source.shape[1] != eval_source.shape[1]:
        raise ValueError(
            f"{arm}: training matrix has {source.shape[1]} columns but the evaluation "
            f"matrix has {eval_source.shape[1]}; an arm must probe one representation"
        )

    for fraction in cfg.fractions:
        for repeat in range(cfg.repeats):
            draw = stratified_draw(field_ids, fraction, repeat=repeat, base_seed=cfg.base_seed)
            if draw.train_idx.size < 4:
                curve.cells.append(
                    ArmResult(arm=arm, fraction=fraction, repeat=repeat, status="absent",
                              n_train=int(draw.train_idx.size),
                              reason=(f"stratified {fraction:.0%} draw yields only "
                                      f"{draw.train_idx.size} labelled cycles; "
                                      f"a probe needs >= 4"))
                )
                continue
            y_tr = labels[draw.train_idx]
            if np.unique(y_tr).size < 2:
                curve.cells.append(
                    ArmResult(arm=arm, fraction=fraction, repeat=repeat, status="absent",
                              n_train=int(draw.train_idx.size),
                              reason="single-class training draw; AP undefined"))
                continue
            X_tr = source[draw.train_idx]

            if kind == "finetune":
                _, eval_scores = finetune_fn(draw.train_idx, eval_source)
                eval_scores = np.asarray(eval_scores, dtype=np.float64)
            elif kind == "scratch":
                # A-scratch trains a fresh model per label draw: it must NOT see the
                # pretrained weights, or it would be A-ssl-ft wearing another name.
                _, eval_scores = scratch_fn(draw.train_idx, eval_source)
                eval_scores = np.asarray(eval_scores, dtype=np.float64)
            else:      # 'probe' (A-rand) and 'logistic' (A-ssl-frozen)
                w = _ridge_logistic(X_tr, y_tr.astype(float), lam=cfg.lam)
                eval_scores = _predict_logistic(w, eval_source)

            if np.unique(eval_labels).size < 2:
                curve.cells.append(
                    ArmResult(arm=arm, fraction=fraction, repeat=repeat, status="absent",
                              n_train=int(draw.train_idx.size),
                              reason="eval split has a single class; AP undefined"))
                continue

            ap, prev, roc, ap_ci, f1_ci, f1, thr = _evaluate_scores(
                np.asarray(eval_scores), eval_labels.astype(int),
                eval_field_ids, cfg,
            )
            curve.cells.append(
                ArmResult(arm=arm, fraction=fraction, repeat=repeat, status="measured",
                          ap=ap, prevalence=prev, f1=f1, threshold=thr, auroc=roc,
                          ap_ci=ap_ci, f1_ci=f1_ci,
                          n_train=int(draw.train_idx.size), n_eval=int(eval_labels.size),
                          reason=("threshold tuned on the evaluated dev scores "
                                  "(no inner validation split at this fraction): "
                                  "F1 is optimistic, AP is threshold-free")))
    return curve


def _curve_from_fixed_scores(
    arm: str, scores: np.ndarray, labels: np.ndarray, eval_labels: np.ndarray,
    eval_field_ids: np.ndarray, cfg: TransferConfig,
) -> Curve:
    """A-rule / A-B1 produce one score per cycle and learn nothing.

    Reporting them at every fraction is deliberate: a rule has no training set, so
    its curve is FLAT, and drawing it that way makes visible that the rule cannot
    use extra labels. Filling in a fraction-dependent number would misrepresent the
    baseline the SSL arms must beat.
    """
    curve = Curve(arm=arm)
    if np.unique(eval_labels).size < 2:
        curve.cells.append(
            ArmResult(arm=arm, fraction=1.0, repeat=0, status="absent",
                      reason="eval split has a single class; AP undefined"))
        return curve
    if scores.size != eval_labels.size:
        raise ValueError(
            f"{arm}: {scores.size} fixed scores for {eval_labels.size} evaluation rows"
        )
    ap, prev, roc, ap_ci, f1_ci, f1, thr = _evaluate_scores(
        scores, eval_labels.astype(int), eval_field_ids, cfg,
    )
    for fraction in cfg.fractions:
        for repeat in range(cfg.repeats):
            curve.cells.append(
                ArmResult(arm=arm, fraction=fraction, repeat=repeat, status="measured",
                          ap=ap, prevalence=prev, f1=f1, threshold=thr, auroc=roc,
                          ap_ci=ap_ci, f1_ci=f1_ci,
                          n_train=0, n_eval=int(eval_labels.size),
                          reason=("fixed rule/incumbent scores: no training, so the "
                                  "curve is flat by construction and carries no label-"
                                  "fraction information")))
    return curve


@dataclass
class LabelEfficiencyResult:
    curves: dict[str, Curve]
    config: TransferConfig
    n_train: int
    n_eval: int
    prevalence: float

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "arms": arm_table(),
            "curves": {k: v.to_dict() for k, v in self.curves.items()},
            "areas": {k: v.area_under_curve() for k, v in self.curves.items()},
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "prevalence": self.prevalence,
        }

    def headline(self, arm: str = "A-ssl-frozen") -> dict:
        """The single row the claim table quotes: 100 % fraction AP for an arm."""
        c = self.curves.get(arm)
        if c is None:
            return {"arm": arm, "status": "absent", "reason": "arm not run"}
        full = [x for x in c.cells if x.measurable and x.fraction == 1.0]
        if not full:
            reasons = sorted({x.reason for x in c.cells if not x.measurable})
            return {"arm": arm, "status": "absent", "reason": "; ".join(reasons) or "no full-label cell"}
        aps = [x.ap for x in full]
        return {
            "arm": arm,
            "status": "measured",
            "ap_mean": float(np.mean(aps)),
            "ap_std": float(np.std(aps)),
            "n_repeats": len(aps),
            "prevalence": full[0].prevalence,
            "ap_ci": list(full[0].ap_ci) if full[0].ap_ci else None,
        }


def compare_arms(
    result: LabelEfficiencyResult, arm_a: str, arm_b: str, *, fraction: float = 1.0, n_boot: int = 500
) -> dict:
    """Paired clustered comparison of two arms at one fraction (H2/H4 support).

    Returns ``present=False`` with a reason when either arm lacks a measured cell
    at that fraction — never a 0.0 difference.
    """
    ca = [c for c in result.curves.get(arm_a, Curve(arm_a)).cells
          if c.measurable and c.fraction == fraction]
    cb = [c for c in result.curves.get(arm_b, Curve(arm_b)).cells
          if c.measurable and c.fraction == fraction]
    if not ca or not cb:
        return {"present": False, "arm_a": arm_a, "arm_b": arm_b, "fraction": fraction,
                "reason": "one or both arms have no measured cell at this fraction"}
    a = np.array([c.ap for c in ca])
    b = np.array([c.ap for c in cb])
    return {
        "present": True,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "fraction": fraction,
        "delta_mean": float(a.mean() - b.mean()),
        "n_repeats_a": int(a.size),
        "n_repeats_b": int(b.size),
        "note": ("repeat-level difference only; the pre-registered claim test is the "
                 "clustered bootstrap in clustered_bootstrap_diff over cycle scores"),
    }

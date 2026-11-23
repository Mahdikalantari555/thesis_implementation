"""Temporal rolls and leave-one-block-out (LOBO) robustness (design §Robustness).

Two complementary stress tests, both reported as *distribution shifts* rather
than single numbers:

**Temporal rolls** — train on cycles strictly before a cut year, evaluate on the
four pre-registered cuts {2016, 2018, 2020, 2022}. The split is by YEAR, and the
implementation refuses to leak: every training cycle must satisfy
``year < cut`` and every evaluation cycle ``year >= cut``. A roll where one side
is empty is ``absent`` for that cut, not silently skipped from the table.

**LOBO** — hold out one spatial block (20 pre-registered blocks), train on the
rest, evaluate on the held-out block. This tests spatial generalisation, which
temporal rolls cannot: a model can pass a temporal roll and still memorise farms.

Both produce a *per-fold* table plus a summary that reports the spread (min,
median, IQR) — because a robustness claim lives in the shape of the distribution,
not the mean. A single fold is not evidence of robustness and :func:`summarise`
says so when ``n_folds < 2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ssl_anomaly.evaluation.metrics import auprc, bootstrap_ci, wilson_interval

#: Pre-registered temporal cut years (pre_registration.yaml evaluation.cut_years).
DEFAULT_CUT_YEARS = (2016, 2018, 2020, 2022)


# ---------------------------------------------------------------------------
# Split construction — with leakage refused
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YearSpan:
    """Year of each cycle. Kept as its own type so a bare int array cannot be
    mistaken for the 36-bin DSH axis."""

    years: np.ndarray

    def __post_init__(self) -> None:
        y = np.asarray(self.years)
        if y.ndim != 1:
            raise ValueError(f"years must be 1-D, got {y.shape}")
        if np.any(~np.isfinite(y)):
            raise ValueError("years contain non-finite values")

    def before(self, cut: int) -> np.ndarray:
        return np.flatnonzero(self.years < cut)

    def at_or_after(self, cut: int) -> np.ndarray:
        return np.flatnonzero(self.years >= cut)


@dataclass
class FoldSpec:
    name: str
    train_idx: np.ndarray
    eval_idx: np.ndarray
    kind: str                      # 'temporal-roll' | 'lobo'
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "n_train": int(self.train_idx.size),
            "n_eval": int(self.eval_idx.size),
            "meta": self.meta,
        }

    def assert_disjoint(self) -> None:
        overlap = np.intersect1d(self.train_idx, self.eval_idx)
        if overlap.size:
            raise ValueError(
                f"fold {self.name}: {overlap.size} cycles appear in both train and "
                f"eval (leakage)"
            )


def temporal_roll_folds(
    years: YearSpan, cut_years: tuple[int, ...] = DEFAULT_CUT_YEARS
) -> list[FoldSpec]:
    """One fold per cut year: train strictly before, evaluate at/after."""
    folds = []
    for cut in cut_years:
        tr = years.before(cut)
        ev = years.at_or_after(cut)
        if tr.size and ev.size:
            folds.append(FoldSpec(f"roll-{cut}", tr, ev, "temporal-roll",
                                  {"cut_year": int(cut)}))
        else:
            folds.append(FoldSpec(f"roll-{cut}", tr, ev, "temporal-roll",
                                  {"cut_year": int(cut),
                                   "unusable": True,
                                   "reason": ("no training cycles before the cut"
                                              if tr.size == 0 else
                                              "no evaluation cycles at/after the cut")}))
    return folds


def lobo_folds(block_ids: np.ndarray) -> list[FoldSpec]:
    """One fold per spatial block: hold it out, train on everything else."""
    b = np.asarray(block_ids)
    if b.ndim != 1:
        raise ValueError(f"block ids must be 1-D, got {b.shape}")
    uniq = np.unique(b)
    folds = []
    for block in uniq:
        ev = np.flatnonzero(b == block)
        tr = np.flatnonzero(b != block)
        folds.append(FoldSpec(f"lobo-{block}", tr, ev, "lobo", {"block": str(block)}))
    return folds


def validate_lobo_coverage(n_blocks_observed: int, n_blocks_expected: int = 20) -> dict:
    """LOBO is only LOBO if all 20 pre-registered blocks exist.

    A silent subset (say 14 blocks present in the table) produces 14 folds and a
    narrower robustness picture, so the shortfall is returned as a first-class
    value: ``complete=False``, with the missing count.
    """
    ok = n_blocks_observed == n_blocks_expected
    return {
        "complete": ok,
        "observed": int(n_blocks_observed),
        "expected": int(n_blocks_expected),
        "missing": int(max(0, n_blocks_expected - n_blocks_observed)),
        "note": ("" if ok else
                 "LOBO folds cover only the blocks present in this dataset; the "
                 "robustness result is conditional on that coverage"),
    }


# ---------------------------------------------------------------------------
# Folding execution
# ---------------------------------------------------------------------------


@dataclass
class FoldResult:
    fold: str
    kind: str
    status: str                    # 'measured' | 'absent'
    ap: float | None = None
    prevalence: float | None = None
    f1: float | None = None
    ap_ci: tuple[float, float] | None = None
    n_train: int = 0
    n_eval: int = 0
    reason: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "fold": self.fold,
            "kind": self.kind,
            "status": self.status,
            "ap": self.ap,
            "prevalence": self.prevalence,
            "f1": self.f1,
            "ap_ci": list(self.ap_ci) if self.ap_ci else None,
            "n_train": self.n_train,
            "n_eval": self.n_eval,
            "reason": self.reason,
            "meta": self.meta,
        }

    @property
    def measured(self) -> bool:
        return self.status == "measured"


def run_fold(
    fold: FoldSpec,
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray | None = None,
) -> FoldResult:
    """Score one fold from PRE-COMPUTED per-cycle scores.

    Folds over pre-computed scores (rather than re-training here) keeps this
    module honest: it measures generalisation *of a fixed representation*, which
    is what these robustness tests are about, and it never re-fits anything on
    the evaluation side.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels).astype(int)
    if fold.meta.get("unusable"):
        return FoldResult(fold.name, fold.kind, "absent", n_train=int(fold.train_idx.size),
                          n_eval=int(fold.eval_idx.size),
                          reason=fold.meta.get("reason", "unusable fold"))
    if fold.eval_idx.size == 0:
        return FoldResult(fold.name, fold.kind, "absent", n_train=int(fold.train_idx.size),
                          reason="empty evaluation side")
    ye = y[fold.eval_idx]
    se = s[fold.eval_idx]
    if np.unique(ye).size < 2:
        # One-class eval cannot give an AP. It is still a real generalisation
        # result — reported as absent WITH the prevalence, so a reader sees that
        # the fold contained no anomalies rather than a missing number.
        return FoldResult(
            fold.name, fold.kind, "absent", n_train=int(fold.train_idx.size),
            n_eval=int(fold.eval_idx.size),
            prevalence=float(ye.mean()),
            reason=("evaluation side has a single class; AP is undefined (this is a "
                    "property of the fold, not of the model)"),
        )
    ap, prev = auprc(se, ye)
    c = None if clusters is None else np.asarray(clusters)[fold.eval_idx]
    from ssl_anomaly.evaluation.metrics import bootstrap_ci_stat

    _, lo, hi = bootstrap_ci_stat(ye, se, lambda yy, ss: auprc(ss, yy)[0],
                                  n_boot=300, seed=13, clusters=c)
    return FoldResult(
        fold.name, fold.kind, "measured", ap=ap, prevalence=prev, ap_ci=(lo, hi),
        n_train=int(fold.train_idx.size), n_eval=int(fold.eval_idx.size), meta=fold.meta,
    )


# ---------------------------------------------------------------------------
# Summary — spread, not just the mean
# ---------------------------------------------------------------------------


@dataclass
class RobustnessSummary:
    kind: str
    n_folds: int
    n_measured: int
    median_ap: float | None
    min_ap: float | None
    max_ap: float | None
    iqr: tuple[float, float] | None
    worst_fold: str | None
    robust: bool
    threshold: float
    reason: str = ""
    absent_folds: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "n_folds": self.n_folds,
            "n_measured": self.n_measured,
            "median_ap": self.median_ap,
            "min_ap": self.min_ap,
            "max_ap": self.max_ap,
            "iqr": list(self.iqr) if self.iqr else None,
            "worst_fold": self.worst_fold,
            "robust": self.robust,
            "threshold": self.threshold,
            "reason": self.reason,
            "absent_folds": self.absent_folds,
        }


def summarise(
    results: list[FoldResult],
    *,
    kind: str,
    min_median_ap: float = 0.5,
    min_fraction_folds_above: float = 0.75,
) -> RobustnessSummary:
    """A robustness verdict that requires BREADTH, not just a good median.

    Two conditions, both pre-registered in spirit (the YAML carries the gate
    criteria; these are the defaults when no gate is passed in):

    1. median AP across measured folds >= ``min_median_ap``;
    2. at least ``min_fraction_folds_above`` of measured folds have AP >= that
       median floor — a model that is excellent in 2 folds and poor in 14 must
       not pass on its median alone.

    With fewer than 2 measured folds the verdict is ``robust=False`` and the
    reason says the evidence is a single point — never "robust by default".
    """
    measured = [r for r in results if r.measured]
    absent = [
        {"fold": r.fold, "reason": r.reason, "n_eval": r.n_eval,
         "prevalence": r.prevalence}
        for r in results if not r.measured
    ]
    if len(measured) < 2:
        return RobustnessSummary(
            kind=kind, n_folds=len(results), n_measured=len(measured),
            median_ap=float(np.median([r.ap for r in measured])) if measured else None,
            min_ap=float(min(r.ap for r in measured)) if measured else None,
            max_ap=float(max(r.ap for r in measured)) if measured else None,
            iqr=None, worst_fold=measured[0].fold if measured else None,
            robust=False, threshold=min_median_ap,
            reason=(f"only {len(measured)} measured fold(s): a single point is not a "
                    f"robustness result"),
            absent_folds=absent,
        )
    aps = np.array([r.ap for r in measured])
    worst = min(measured, key=lambda r: r.ap)
    frac_ok = float(np.mean(aps >= min_median_ap))
    median = float(np.median(aps))
    ok = median >= min_median_ap and frac_ok >= min_fraction_folds_above
    reason = ""
    if not ok:
        bits = []
        if median < min_median_ap:
            bits.append(f"median AP {median:.3f} < {min_median_ap}")
        if frac_ok < min_fraction_folds_above:
            bits.append(
                f"only {frac_ok:.0%} of folds reach {min_median_ap} "
                f"(need {min_fraction_folds_above:.0%})"
            )
        reason = "; ".join(bits)
    q25, q75 = np.quantile(aps, [0.25, 0.75])
    return RobustnessSummary(
        kind=kind, n_folds=len(results), n_measured=len(measured), median_ap=median,
        min_ap=float(aps.min()), max_ap=float(aps.max()), iqr=(float(q25), float(q75)),
        worst_fold=worst.fold, robust=ok, threshold=min_median_ap, reason=reason,
        absent_folds=absent,
    )


@dataclass
class RobustnessReport:
    temporal: list[FoldResult]
    lobo: list[FoldResult]
    temporal_summary: RobustnessSummary
    lobo_summary: RobustnessSummary
    lobo_coverage: dict

    def to_dict(self) -> dict:
        return {
            "temporal_rolls": [r.to_dict() for r in self.temporal],
            "lobo": [r.to_dict() for r in self.lobo],
            "temporal_summary": self.temporal_summary.to_dict(),
            "lobo_summary": self.lobo_summary.to_dict(),
            "lobo_coverage": self.lobo_coverage,
        }

    def gate_evidence(self) -> dict:
        """The two quantities G-C reads."""
        return {
            "temporal_min_ap": self.temporal_summary.min_ap,
            "temporal_median_ap": self.temporal_summary.median_ap,
            "temporal_robust": self.temporal_summary.robust,
            "lobo_median_ap": self.lobo_summary.median_ap,
            "lobo_robust": self.lobo_summary.robust,
            "lobo_coverage_complete": self.lobo_coverage["complete"],
        }


def run_robustness(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    years: np.ndarray,
    block_ids: np.ndarray,
    clusters: np.ndarray | None = None,
    cut_years: tuple[int, ...] = DEFAULT_CUT_YEARS,
    n_blocks_expected: int = 20,
    **summary_kwargs,
) -> RobustnessReport:
    """Run both families and return the report (no shortcuts, absent folds kept)."""
    span = YearSpan(years)
    temporal = [run_fold(f, scores=scores, labels=labels, clusters=clusters)
                for f in temporal_roll_folds(span, cut_years)]
    blocks = np.asarray(block_ids)
    lobo = [run_fold(f, scores=scores, labels=labels, clusters=clusters)
            for f in lobo_folds(blocks)]
    return RobustnessReport(
        temporal=temporal, lobo=lobo,
        temporal_summary=summarise(temporal, kind="temporal-roll", **summary_kwargs),
        lobo_summary=summarise(lobo, kind="lobo", **summary_kwargs),
        lobo_coverage=validate_lobo_coverage(int(np.unique(blocks).size), n_blocks_expected),
    )

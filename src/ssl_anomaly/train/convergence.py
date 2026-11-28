"""Plateau-controlled training verdicts (task 1.1, design.md §4.1/§4.2, r2).

A lower weighted total loss is NOT evidence of a better representation [DEC].
Each evaluation feeds the monitor a monitoring vector: every objective's
validation loss separately, the primary probe metric, and representation
health (effective rank / per-dim std). The plateau rule is applied per guard
signal; the stop decision uses the whole guard set:

* stop when ALL guard signals have plateaued, or
* when the primary probe metric has plateaued AND no objective is still
  improving beyond threshold, or
* immediately on divergence (non-finite or val loss > ``diverge_factor``×best)
  or collapse (effective rank / per-dim std below floor for ``patience``
  evals).

``OBJECTIVE_CONFLICT`` is a FLAG, not a stop: an objective regressing more
than ``max_obj_regress`` relative to its best while the total improves makes
that checkpoint ineligible for ``best.pt``; a conflict persisting over
``patience`` evaluations stops the run for owner review.

Pure Python/NumPy — no torch here; the trainer loop (§4.3/1.2) owns the
model and queries this monitor once per evaluation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

import numpy as np


class Verdict(str, Enum):
    RUNNING = "RUNNING"
    CONVERGED_PLATEAU = "CONVERGED_PLATEAU"
    BUDGET_EXHAUSTED_STILL_IMPROVING = "BUDGET_EXHAUSTED_STILL_IMPROVING"
    BUDGET_EXHAUSTED_FLAT = "BUDGET_EXHAUSTED_FLAT"
    DIVERGED = "DIVERGED"
    COLLAPSED = "COLLAPSED"
    OBJECTIVE_CONFLICT = "OBJECTIVE_CONFLICT"  # persistent conflict -> stop for owner review


@dataclass
class SignalState:
    """Per-guard-signal plateau bookkeeping."""

    name: str
    direction: str  # "min" (loss-like) | "max" (metric-like)
    best: float = math.inf
    best_step: int = -1
    last_value: float = math.nan
    flat_counter: int = 0           # consecutive evals with improvement <= min_rel_delta
    history: deque = field(default_factory=lambda: deque(maxlen=2))
    ema_slope: float = 0.0          # EMA of per-eval relative change
    plateau: bool = False

    def relative_improvement(self, value: float) -> float:
        """Positive = better than best so far (either direction)."""
        if not math.isfinite(value):
            return 0.0
        if self.direction == "min":
            denom = abs(self.best) if math.isfinite(self.best) and self.best != 0 else 1.0
            return (self.best - value) / denom
        denom = abs(self.best) if math.isfinite(self.best) and self.best != 0 else 1.0
        return (value - self.best) / denom


@dataclass
class EvalOutcome:
    """Everything one evaluation contributes to the trainer and the log."""

    step: int
    verdict: Verdict = Verdict.RUNNING
    should_stop: bool = False
    plateaued_signals: list[str] = field(default_factory=list)
    improving_signals: list[str] = field(default_factory=list)
    conflict_objectives: list[str] = field(default_factory=list)
    conflict: bool = False                  # flag (checkpoint ineligible)
    collapsed: bool = False
    diverged: bool = False
    lr_drop_triggered: bool = False
    lr_drops_so_far: int = 0
    eligible_for_best: bool = True
    extra_steps_estimate: int | None = None
    reason: str = ""

    def to_log_fields(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "plateaued": list(self.plateaued_signals),
            "improving": list(self.improving_signals),
            "conflict_objectives": list(self.conflict_objectives),
            "objective_conflict": self.conflict,
            "collapsed": self.collapsed,
            "diverged": self.diverged,
            "eligible_for_best": self.eligible_for_best,
            "lr_drop_triggered": self.lr_drop_triggered,
            "lr_drops_so_far": self.lr_drops_so_far,
            "extra_steps_estimate": self.extra_steps_estimate,
            "reason": self.reason,
        }


class PlateauMonitor:
    """Multi-signal plateau and verdict monitor (design.md §4.1–§4.2).

    Parameters map 1:1 to the spec: ``min_rel_delta`` (default 0.5 %),
    ``patience`` evals (default 8), EMA slope over ``window`` evals (default
    10), ``min_steps`` warm-up, ``max_obj_regress`` (default 5 % relative to
    an objective's best), ``diverge_factor``.
    """

    def __init__(
        self,
        signals: Mapping[str, str],
        *,
        primary: str,
        min_rel_delta: float = 0.005,
        patience: int = 8,
        window: int = 10,
        min_steps: int = 0,
        max_obj_regress: float = 0.05,
        diverge_factor: float = 10.0,
        lr_patience: int | None = None,
        max_lr_drops: int = 0,
        lr_drop_factor: float = 0.5,
        effective_rank_floor: float | None = None,
        per_dim_std_floor: float | None = None,
        collapse_patience: int | None = None,
    ) -> None:
        if primary not in signals:
            raise ValueError(f"primary signal {primary!r} must be one of {sorted(signals)}")
        for name, direction in signals.items():
            if direction not in ("min", "max"):
                raise ValueError(f"signal {name!r} direction must be 'min' or 'max'")
        if "total" in signals:
            raise ValueError(
                "'total' is an internally tracked signal; declare only the "
                "objective and probe signals"
            )
        self.signals = {n: SignalState(n, d) for n, d in signals.items()}
        self.primary = primary
        self.min_rel_delta = float(min_rel_delta)
        self.patience = int(patience)
        self.window = int(window)
        self.min_steps = int(min_steps)
        self.max_obj_regress = float(max_obj_regress)
        self.diverge_factor = float(diverge_factor)
        self.lr_patience = lr_patience
        self.max_lr_drops = int(max_lr_drops)
        self.lr_drop_factor = float(lr_drop_factor)
        self.effective_rank_floor = effective_rank_floor
        self.per_dim_std_floor = per_dim_std_floor
        self.collapse_patience = int(collapse_patience if collapse_patience is not None else patience)
        self.lr_drops = 0
        self.conflict_streak = 0
        self.collapse_streak = 0
        self._current_step = 0
        self.slope_histories: dict[str, deque] = {
            n: deque(maxlen=max(2, window)) for n in self.signals
        }
        #: objectives (loss-like signals) whose regression is checked for conflicts
        self._objective_names = [n for n, d in signals.items() if d == "min"]
        #: the weighted total is tracked (divergence/conflict reference) but is
        #: never a guard signal (design §4.1: guard = objectives + primary probe)
        self.signals["total"] = SignalState("total", "min")

    # ------------------------------------------------------------------ #
    def _update_signal(self, s: SignalState, value: float) -> None:
        """Update one signal's plateau bookkeeping."""
        if math.isfinite(value):
            improvement = s.relative_improvement(value)
            if improvement > self.min_rel_delta:
                s.best = value
                s.best_step = self._current_step
                s.flat_counter = 0
            else:
                s.flat_counter += 1
        s.last_value = value
        if math.isfinite(value) and s.history:
            prev = s.history[-1]
            if math.isfinite(prev) and abs(prev) > 1e-12:
                rel_step = (value - prev) / abs(prev)
            else:
                rel_step = 0.0
            alpha = 2.0 / (self.window + 1.0)  # standard EMA smoothing
            s.ema_slope = alpha * rel_step + (1.0 - alpha) * s.ema_slope
        if math.isfinite(value):
            s.history.append(value)
        # slope threshold: average per-eval relative change below the
        # per-eval share of min_rel_delta counts as flat
        slope_flat = abs(s.ema_slope) <= self.min_rel_delta / max(1, self.window)
        s.plateau = s.flat_counter >= self.patience and slope_flat

    def _signal_improving(self, s: SignalState) -> bool:
        """True when the signal is still improving beyond threshold.

        "Improving" is directional: a loss falling (or a metric rising) with
        an EMA slope steeper than the per-eval threshold. A degrading signal
        (slope in the wrong direction) is NOT improving — the spec's
        budget-flat verdict depends on this.
        """
        if s.plateau:
            return False
        slope_beyond = abs(s.ema_slope) > self.min_rel_delta / max(1, self.window)
        if not slope_beyond:
            return False
        good_direction = (s.ema_slope < 0) if s.direction == "min" else (s.ema_slope > 0)
        return good_direction

    # ------------------------------------------------------------------ #
    def update(
        self,
        step: int,
        total_loss: float,
        objectives: Mapping[str, float],
        probe_metrics: Mapping[str, float] | None = None,
        rep_stats: Mapping[str, float] | None = None,
    ) -> EvalOutcome:
        """Feed one evaluation; returns the verdict and all flags."""
        self._current_step = int(step)
        outcome = EvalOutcome(step=step)
        metrics: dict[str, float] = dict(objectives)
        metrics["total"] = float(total_loss)
        if probe_metrics:
            metrics.update(probe_metrics)
        if rep_stats and self.effective_rank_floor is not None:
            er = rep_stats.get("effective_rank")
            if er is not None:
                metrics.setdefault("__effective_rank__", float(er))

        # ---- divergence ------------------------------------------------
        nonfinite = [n for n, v in objectives.items() if not math.isfinite(float(v))]
        if probe_metrics:
            nonfinite += [n for n, v in probe_metrics.items() if not math.isfinite(float(v))]
        if nonfinite or not math.isfinite(total_loss):
            outcome.verdict = Verdict.DIVERGED
            outcome.diverged = True
            outcome.should_stop = True
            outcome.reason = (
                f"non-finite values at step {step}: "
                + (", ".join(nonfinite) if nonfinite else "total loss")
            )
            return outcome
        total_state = self.signals["total"]
        if (
            math.isfinite(total_state.best)
            and total_state.best != math.inf
            and total_loss > self.diverge_factor * abs(total_state.best)
        ):
            outcome.verdict = Verdict.DIVERGED
            outcome.diverged = True
            outcome.should_stop = True
            outcome.reason = (
                f"val loss {total_loss:.4g} > diverge_factor×best "
                f"({self.diverge_factor}×{total_state.best:.4g})"
            )
            return outcome

        # ---- per-signal plateau bookkeeping ----------------------------
        for name, value in metrics.items():
            if name.startswith("__"):
                continue
            if name not in self.signals:
                raise ValueError(
                    f"unexpected metric {name!r}; monitor signals: "
                    f"{sorted(self.signals)} — new guard signals must be "
                    "declared at construction, not drifted in mid-run"
                )
            self._update_signal(self.signals[name], float(value))

        # ---- representation collapse -----------------------------------
        if rep_stats is not None and self.effective_rank_floor is not None:
            er = rep_stats.get("effective_rank")
            std_min = rep_stats.get("per_dim_std_min")
            bad = False
            if er is not None and float(er) < self.effective_rank_floor:
                bad = True
            if (
                self.per_dim_std_floor is not None
                and std_min is not None
                and float(std_min) < self.per_dim_std_floor
            ):
                bad = True
            self.collapse_streak = self.collapse_streak + 1 if bad else 0
            if self.collapse_streak >= self.collapse_patience:
                outcome.verdict = Verdict.COLLAPSED
                outcome.collapsed = True
                outcome.should_stop = True
                outcome.eligible_for_best = False
                outcome.reason = (
                    f"representation below floor for {self.collapse_streak} evals "
                    f"(effective_rank={er}, per_dim_std_min={std_min})"
                )
                return outcome
            outcome.collapsed = bad

        # ---- objective conflict (flag) ----------------------------------
        total_state = self.signals["total"]
        # "while the total improves" is read as "while the total is not
        # meaningfully worse than its best" — a total that holds flat (up to
        # float noise) while an objective degrades is exactly the case the
        # guard exists for, so strict per-eval improvement would flicker.
        if math.isfinite(total_state.best) and total_state.best != math.inf:
            total_regress = (total_loss - total_state.best) / max(abs(total_state.best), 1e-12)
        else:
            total_regress = 0.0
        total_holds = total_regress <= self.min_rel_delta
        conflict_objectives: list[str] = []
        for name in self._objective_names:
            s = self.signals[name]
            if not math.isfinite(s.best) or s.best == math.inf:
                continue
            regress = (metrics[name] - s.best) / max(abs(s.best), 1e-12)
            if regress > self.max_obj_regress and total_holds:
                conflict_objectives.append(name)
        outcome.conflict_objectives = conflict_objectives
        outcome.conflict = bool(conflict_objectives)
        if outcome.conflict:
            self.conflict_streak += 1
        else:
            self.conflict_streak = 0
        if self.conflict_streak >= self.patience:
            outcome.verdict = Verdict.OBJECTIVE_CONFLICT
            outcome.should_stop = True
            outcome.reason = (
                f"objective conflict persistent over {self.conflict_streak} "
                f"evals ({', '.join(conflict_objectives)} regressed while total "
                "improved) — stopping for owner review"
            )
            return outcome

        # ---- checkpoint eligibility (1.1c) ------------------------------
        outcome.eligible_for_best = not outcome.conflict and not outcome.collapsed

        # ---- plateau decision -------------------------------------------
        warm = step >= self.min_steps
        plateaued = [n for n, s in self.signals.items() if s.plateau and n != "total"]
        improving = [n for n, s in self.signals.items() if self._signal_improving(s) and n != "total"]
        objective_signals = [n for n in self._objective_names if n != "total"]
        probe_plateaued = self.signals[self.primary].plateau
        objectives_improving = [n for n in objective_signals if n in improving]

        if warm:
            # guard set per design §4.1: every objective + probe metrics —
            # the weighted total is tracked but is never a guard signal
            guard_names = [n for n in self.signals if n != "total"]
            all_plateaued = all(self.signals[n].plateau for n in guard_names)
            if all_plateaued:
                outcome.verdict = Verdict.CONVERGED_PLATEAU
                outcome.should_stop = True
                outcome.reason = "all guard signals plateaued"
                return outcome
            if probe_plateaued and not objectives_improving:
                outcome.verdict = Verdict.CONVERGED_PLATEAU
                outcome.should_stop = True
                outcome.reason = (
                    f"primary probe metric {self.primary!r} plateaued and no "
                    "objective still improving beyond threshold"
                )
                return outcome

        # ---- optional LR drop before final stop --------------------------
        if (
            warm
            and probe_plateaued
            and self.lr_patience is not None
            and self.max_lr_drops > 0
            and self.lr_drops < self.max_lr_drops
            and self.signals[self.primary].flat_counter >= self.lr_patience
        ):
            self.lr_drops += 1
            for s in self.signals.values():
                s.flat_counter = 0
                s.plateau = False
            outcome.lr_drop_triggered = True
            outcome.lr_drops_so_far = self.lr_drops
            outcome.reason = (
                f"primary plateaued; LR drop ×{self.lr_drop_factor} "
                f"({self.lr_drops}/{self.max_lr_drops})"
            )
            return outcome

        outcome.plateaued_signals = plateaued
        outcome.improving_signals = improving
        return outcome

    # ------------------------------------------------------------------ #
    def budget_exhausted(self, step: int, steps_per_eval: float) -> EvalOutcome:
        """Classify a run that hit ``max_steps``/wall cap before plateauing."""
        outcome = EvalOutcome(step=step)
        improving = [n for n, s in self.signals.items()
                     if self._signal_improving(s) and not n.startswith("__") and n != "total"]
        outcome.improving_signals = improving
        if improving:
            outcome.verdict = Verdict.BUDGET_EXHAUSTED_STILL_IMPROVING
            # honest lower-bound estimate: the improving signal must still
            # survive `patience` more evaluations at the current step cadence
            primary_s = self.signals[self.primary]
            if primary_s in [self.signals[n] for n in improving] or self.primary in improving:
                remaining_evals = max(0, self.patience - primary_s.flat_counter)
            else:
                s = self.signals[improving[0]]
                remaining_evals = max(0, self.patience - s.flat_counter)
            outcome.extra_steps_estimate = int(max(0, remaining_evals) * max(0.0, steps_per_eval))
            outcome.reason = (
                "needs more training: "
                + ", ".join(improving)
                + " still improving at the cap"
            )
        else:
            outcome.verdict = Verdict.BUDGET_EXHAUSTED_FLAT
            outcome.reason = (
                "curve flat but patience not reached at the cap; likely "
                "converged — optional resume"
            )
        return outcome

    def state(self) -> dict[str, Any]:
        """Machine-readable monitor state (for the checkpoint / manifest)."""
        return {
            "primary": self.primary,
            "min_rel_delta": self.min_rel_delta,
            "patience": self.patience,
            "window": self.window,
            "min_steps": self.min_steps,
            "max_obj_regress": self.max_obj_regress,
            "diverge_factor": self.diverge_factor,
            "lr_drops": self.lr_drops,
            "conflict_streak": self.conflict_streak,
            "collapse_streak": self.collapse_streak,
            "signals": {
                n: {
                    "direction": s.direction,
                    "best": s.best if math.isfinite(s.best) else None,
                    "best_step": s.best_step,
                    "flat_counter": s.flat_counter,
                    "ema_slope": s.ema_slope,
                    "plateau": s.plateau,
                    "last_value": s.last_value if math.isfinite(s.last_value) else None,
                }
                for n, s in self.signals.items()
            },
        }

    def load_state(self, st: Mapping[str, Any]) -> None:
        """Restore from :meth:`state` (exact-resume support)."""
        self.lr_drops = int(st.get("lr_drops", 0))
        self.conflict_streak = int(st.get("conflict_streak", 0))
        self.collapse_streak = int(st.get("collapse_streak", 0))
        for name, ss in st.get("signals", {}).items():
            s = self.signals.get(name)
            if s is None:
                continue
            s.best = ss["best"] if ss["best"] is not None else math.inf
            s.best_step = int(ss["best_step"])
            s.flat_counter = int(ss["flat_counter"])
            s.ema_slope = float(ss["ema_slope"])
            s.plateau = bool(ss["plateau"])
            s.last_value = ss["last_value"] if ss["last_value"] is not None else math.nan

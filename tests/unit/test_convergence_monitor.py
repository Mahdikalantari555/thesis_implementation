"""Task 1.1 — PlateauMonitor verdicts on synthetic curves (spec scenarios)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ssl_anomaly.train.convergence import PlateauMonitor, Verdict


def make_monitor(**kw) -> PlateauMonitor:
    base = dict(
        signals={"recon": "min", "forecast": "min", "phase_r2": "max"},
        primary="phase_r2",
        min_rel_delta=0.005,
        patience=4,
        window=3,
        min_steps=0,
        max_obj_regress=0.05,
    )
    base.update(kw)
    return PlateauMonitor(**base)


def feed(monitor: PlateauMonitor, series, *, probes=None, rep=None, start=0, eval_stride=100):
    """Feed a curve dict {signal: [v0, v1, ...]}; returns the last outcome."""
    n = len(next(iter(series.values())))
    last = None
    for i in range(n):
        objectives = {k: v[i] for k, v in series.items() if k != "phase_r2"}
        pm = {"phase_r2": series["phase_r2"][i]} if "phase_r2" in series else (probes or {})
        last = monitor.update(
            start + i * eval_stride,
            objectives.get("recon", 1.0) + objectives.get("forecast", 1.0),
            {k: v for k, v in objectives.items()},
            probe_metrics=pm,
            rep_stats=rep,
        )
    return last


class TestPlateauScenarios:
    def test_plateau_stops_before_cap(self):
        m = make_monitor()
        flat = [1.0] * 10
        out = feed(m, {"recon": flat, "forecast": flat, "phase_r2": [0.5] * 10})
        assert out.verdict == Verdict.CONVERGED_PLATEAU
        assert out.should_stop

    def test_noise_does_not_stop_while_slope_negative(self):
        """Patience counter reached but slope still clearly negative → continue."""
        m = make_monitor(patience=3, window=5, min_rel_delta=0.005)
        # steady 1% per-eval improvement for 10 evals (counter never exceeds 0
        # because improvement > threshold every eval)
        rec = [1.0 * (0.99 ** i) for i in range(10)]
        fc = [0.5 * (0.99 ** i) for i in range(10)]
        r2 = list(np.linspace(0.5, 0.6, 10))
        out = feed(m, {"recon": rec, "forecast": fc, "phase_r2": r2})
        assert not out.should_stop
        assert out.verdict == Verdict.RUNNING

    def test_patience_without_flat_slope_continues(self):
        """Counter reaches patience but EMA slope steeply negative → continue."""
        m = make_monitor(patience=3, window=5)
        # tiny improvements (< 1%) but consistent downward slope
        rec = [1.0 - 0.004 * i for i in range(12)]
        fc = rec[:]
        r2 = [0.5 + 0.001 * i for i in range(12)]
        out = feed(m, {"recon": rec, "forecast": fc, "phase_r2": r2})
        assert not out.should_stop

    def test_total_flat_but_one_objective_improving_continues(self):
        m = make_monitor()
        n = 12
        flat = [1.0] * n
        fc = [0.5 * (0.9 ** i) for i in range(n)]  # forecast keeps improving 10%/eval
        out = feed(m, {"recon": flat, "forecast": fc, "phase_r2": [0.5] * n})
        assert not out.should_stop
        assert "forecast" in out.improving_signals

    def test_total_improving_while_one_objective_degrades_flags_conflict(self):
        m = make_monitor(max_obj_regress=0.05, patience=8)  # long patience: flag only
        n = 8
        recon = [1.0, 0.8, 0.65, 0.5, 0.4, 0.35, 0.3, 0.28]   # total keeps improving
        fc = [0.5, 0.5, 0.5, 0.5, 0.526, 0.55, 0.58, 0.60]    # regresses > 5% vs best
        r2 = [0.5] * n
        out = feed(m, {"recon": recon, "forecast": fc, "phase_r2": r2})
        assert out.conflict is True
        assert "forecast" in out.conflict_objectives
        assert out.eligible_for_best is False
        assert not out.should_stop  # flag only, run continues

    def test_persistent_conflict_stops_for_owner_review(self):
        m = make_monitor(max_obj_regress=0.05, patience=3)
        n = 8
        recon = [1.0, 0.8, 0.65, 0.5, 0.4, 0.35, 0.3, 0.28]
        fc = [0.5, 0.5, 0.5, 0.526, 0.55, 0.58, 0.60, 0.62]
        r2 = [0.5] * n
        out = feed(m, {"recon": recon, "forecast": fc, "phase_r2": r2})
        assert out.verdict == Verdict.OBJECTIVE_CONFLICT
        assert out.should_stop


class TestVerdicts:
    def test_nan_divergence(self):
        m = make_monitor()
        out = m.update(0, float("nan"), {"recon": 1.0, "forecast": 1.0}, {"phase_r2": 0.5})
        assert out.verdict == Verdict.DIVERGED
        assert out.should_stop and out.diverged

    def test_inf_objective_diverges(self):
        m = make_monitor()
        out = m.update(0, 1.0, {"recon": float("inf"), "forecast": 1.0}, {"phase_r2": 0.5})
        assert out.verdict == Verdict.DIVERGED

    def test_val_loss_beyond_diverge_factor(self):
        m = make_monitor(diverge_factor=10.0)
        m.update(0, 1.0, {"recon": 1.0, "forecast": 0.5}, {"phase_r2": 0.5})
        out = m.update(100, 15.0, {"recon": 7.0, "forecast": 8.0}, {"phase_r2": 0.5})
        assert out.verdict == Verdict.DIVERGED

    def test_collapse_stops_after_patience(self):
        m = make_monitor(
            effective_rank_floor=2.0, per_dim_std_floor=1e-3, collapse_patience=3
        )
        rep = {"effective_rank": 1.3, "per_dim_std_min": 1e-5}  # finding-28-like
        last = None
        for i in range(5):
            last = m.update(i * 100, 1.0, {"recon": 1.0, "forecast": 1.0},
                            {"phase_r2": 0.5}, rep_stats=rep)
        assert last.verdict == Verdict.COLLAPSED
        assert last.should_stop and last.collapsed

    def test_single_collapsed_eval_does_not_stop(self):
        m = make_monitor(effective_rank_floor=2.0, collapse_patience=3)
        out = m.update(0, 1.0, {"recon": 1.0, "forecast": 1.0}, {"phase_r2": 0.5},
                       rep_stats={"effective_rank": 1.3, "per_dim_std_min": 1e-5})
        assert not out.should_stop
        assert out.collapsed is True
        assert out.eligible_for_best is False  # collapsed checkpoint ineligible


class TestBudget:
    def test_still_improving_at_cap(self):
        m = make_monitor()
        rec = [1.0 * (0.95 ** i) for i in range(6)]
        fc = [0.5 * (0.95 ** i) for i in range(6)]
        r2 = list(np.linspace(0.5, 0.7, 6))
        feed(m, {"recon": rec, "forecast": fc, "phase_r2": r2})
        out = m.budget_exhausted(step=600, steps_per_eval=50)
        assert out.verdict == Verdict.BUDGET_EXHAUSTED_STILL_IMPROVING
        assert out.improving_signals
        assert out.extra_steps_estimate is not None and out.extra_steps_estimate >= 0
        assert "needs more training" in out.reason

    def test_flat_at_cap(self):
        m = make_monitor()
        # improvements below threshold but slope flat → flat counters rise
        rec = [1.0 + 0.001 * i for i in range(6)]
        fc = [0.5 + 0.001 * i for i in range(6)]
        r2 = [0.5 - 0.001 * i for i in range(6)]
        feed(m, {"recon": rec, "forecast": fc, "phase_r2": r2})
        out = m.budget_exhausted(step=600, steps_per_eval=50)
        assert out.verdict == Verdict.BUDGET_EXHAUSTED_FLAT
        assert not out.improving_signals


class TestMonitorState:
    def test_state_roundtrip_restores_plateau_decision(self):
        m = make_monitor(patience=2, window=2)
        flat = [1.0] * 4
        feed(m, {"recon": flat, "forecast": flat, "phase_r2": [0.5] * 4})
        st = m.state()
        m2 = make_monitor(patience=2, window=2)
        m2.load_state(st)
        out = m2.update(9999, 1.0, {"recon": 1.0, "forecast": 1.0}, {"phase_r2": 0.5})
        assert out.verdict == Verdict.CONVERGED_PLATEAU  # restored counters

    def test_unknown_signal_refused(self):
        m = make_monitor()
        with pytest.raises(ValueError):
            m.update(0, 1.0, {"recon": 1.0, "forecast": 1.0, "mystery": 2.0}, {"phase_r2": 0.5})

    def test_primary_must_be_declared(self):
        with pytest.raises(ValueError):
            make_monitor(primary="nope")

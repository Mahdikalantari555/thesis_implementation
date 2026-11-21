"""Pin the path-C decision rule's FAR budget at the cycle scale.

Audit 2026-11-21: the original "clean event-FAR ≤ 5 %" condition counted
false events per SCORED BIN (~1/28 of a cycle) against a per-CYCLE target
— with ~28-30 scored bins even one false event in every clean cycle stays
below target, so the condition could never bind. The gating condition must
be the cycle-level FAR of the clean evaluation paths; event-FAR stays
recorded as a descriptive number.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_probe():
    sys.path.insert(0, str(ROOT / "src"))
    mod = types.ModuleType("probe_forecast_units")
    path = ROOT / "scripts" / "probe_forecast_units.py"
    mod.__file__ = str(path)          # the script resolves REPO_ROOT from __file__
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), mod.__dict__)
    return mod


def _arm(*, sentinel=3, censored=False, cycle_eval=0.0, event_eval=0.0, far_calib=0.0):
    return {
        "mdm80": {"persistent/ndvi": {"censored": censored, "sentinel": sentinel,
                                      "value": ">S6" if censored else f"S{sentinel}",
                                      "median_delay_days": None}},
        "far_achieved": far_calib,
        "clean_cycle_far_eval": cycle_eval,
        "event_far_clean_eval": {"event_far": event_eval},
    }


def _per_arm(**kw):
    probe = _load_probe()
    return {arm: _arm(**kw) for arm in probe.UNTRAINED_ARMS}


def test_under_budget_event_far_cannot_rescue_an_over_budget_cycle_far():
    """The old rule called this arm a success (0.036 ≤ 0.05) even though one
    in ten clean cycles alarms — exactly the unfalsifiable-condition defect."""
    probe = _load_probe()
    out = probe.evaluate_decision_rule(_per_arm(cycle_eval=0.10, event_eval=0.036))
    d = out["per_arm"][probe.UNTRAINED_ARMS[0]]
    assert "clean_event_far_eval_within_budget" not in d["conditions"]
    assert d["conditions"]["clean_cycle_far_eval_within_budget"] is False
    assert d["success"] is False
    assert d["clean_event_far_eval_descriptive"] == 0.036
    assert out["outcome"] != "success"


def test_within_budget_cycle_far_passes_and_is_the_gating_condition():
    probe = _load_probe()
    out = probe.evaluate_decision_rule(_per_arm(cycle_eval=0.04, event_eval=0.036))
    d = out["per_arm"][probe.UNTRAINED_ARMS[0]]
    assert d["conditions"]["clean_cycle_far_eval_within_budget"] is True
    assert d["success"] is True
    assert out["outcome"] == "success"


def test_amendment_recorded_and_all_censored_outcome_unchanged():
    probe = _load_probe()
    summary = probe.evaluate_decision_rule(_per_arm())
    assert "2026-11-21" in summary["rule_amendment_2026_11_21"]
    out = probe.evaluate_decision_rule(_per_arm(censored=True, sentinel=7, cycle_eval=0.0))
    assert out["outcome"] == "negative_all_censored"
    assert out["per_arm"][probe.UNTRAINED_ARMS[0]]["success"] is False

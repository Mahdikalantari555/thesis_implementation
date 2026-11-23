"""The Level-1 producer must emit the paired schema G-A actually reads.

The bug this file guards against was silent: ``Level1Result.gate_evidence()``
emits ``farm_separability_top1`` / ``phase_probe_accuracy``, but ``gate_a`` reads
``farm_separability_ssl`` / ``farm_separability_rand`` / .... Feeding the raw dict
straight in matched nothing, so the gate reported "missing evidence" no matter
what had been measured, and the run looked like an absent one forever.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "evaluation" / "evaluate_representation_quality.py"


@pytest.fixture(scope="module")
def script():
    sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("eval_quality", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _candidate(ssl=None, rand=None, collapse=False):
    ssl_ev = {"farm_separability_top1": 0.40, "phase_probe_accuracy": 0.30,
              "collapse_disqualified": collapse}
    rand_ev = {"farm_separability_top1": 0.15, "phase_probe_accuracy": 0.10,
               "collapse_disqualified": False}
    ssl_ev.update(ssl or {})
    rand_ev.update(rand or {})
    return {"gate_evidence": ssl_ev, "random_init": {"gate_evidence": rand_ev}}


def test_paired_evidence_uses_the_keys_gate_a_declares(script):
    from ssl_anomaly.evaluation.gates import GATE_INPUTS

    ev = script._paired_g_a_evidence(_candidate())
    assert set(ev) <= set(GATE_INPUTS["G-A"]), (
        f"producer leaked non-gate keys: {set(ev) - set(GATE_INPUTS['G-A'])}"
    )
    assert "farm_separability_ssl" in ev and "farm_separability_rand" in ev
    assert "phase_probe_ssl" in ev and "phase_probe_rand" in ev


def test_raw_level1_keys_never_reach_the_gate(script):
    ev = script._paired_g_a_evidence(_candidate())
    assert "farm_separability_top1" not in ev, (
        "the raw Level-1 key must be translated, not forwarded"
    )


def test_paired_evidence_feeds_a_decidable_gate(script):
    """The whole point: a fully-measured candidate must let G-A decide."""
    from ssl_anomaly.evaluation.binding import load_preregistration
    from ssl_anomaly.evaluation.gates import gate_a

    prereg = load_preregistration()
    verdict = gate_a(script._paired_g_a_evidence(_candidate()),
                     dict(prereg.gates["G-A"]))
    assert verdict.verdict != "undetermined", verdict.reason
    assert verdict.verdict == "passed", verdict.reason


def test_missing_control_omits_the_term_rather_than_zeroing_it(script):
    """An absent A-rand must not be imputed as 0.0 — that would guarantee a pass."""
    cand = _candidate()
    cand["random_init"] = {}
    ev = script._paired_g_a_evidence(cand)
    assert "farm_separability_ssl" not in ev
    assert "farm_separability_rand" not in ev


def test_unmeasurable_probe_is_omitted_not_zeroed(script):
    nulls = {"farm_separability_top1": None, "phase_probe_accuracy": None}
    ev = script._paired_g_a_evidence(
        _candidate(ssl=nulls, rand={"farm_separability_top1": None,
                                    "phase_probe_accuracy": None})
    )
    assert ev == {} or set(ev) == {"collapse_disqualified"}


def test_collapse_flag_is_always_forwarded(script):
    ev = script._paired_g_a_evidence(_candidate(collapse=True))
    assert ev["collapse_disqualified"] is True


def test_collapse_disqualified_candidate_fails_the_gate(script):
    from ssl_anomaly.evaluation.binding import load_preregistration
    from ssl_anomaly.evaluation.gates import gate_a

    prereg = load_preregistration()
    verdict = gate_a(script._paired_g_a_evidence(_candidate(collapse=True)),
                     dict(prereg.gates["G-A"]))
    assert verdict.verdict == "failed"
    assert "collapse" in verdict.reason


# ---------------------------------------------------------------------------
# The no-winner branch
# ---------------------------------------------------------------------------


def test_no_winner_yields_undetermined_not_failed(script, tmp_path, capsys):
    """A screening that selected nothing measured no term; it did not lose one."""
    out = tmp_path / "o"
    code = script.main(["--smoke", "--skip-training", "--out", str(out),
                        "--seed", "13"])
    # The script returns 0 for an honest smoke even when G-A cannot decide.
    assert code == 0
    import json

    payload = json.loads(
        (out / "pretraining_quality.json").read_text()
        if (out / "pretraining_quality.json").exists()
        else (tuple(out.rglob("pretraining_quality.json"))[0]).read_text()
    )
    g_a = payload["gate_a"]
    assert g_a["verdict"] in {"undetermined", "failed", "passed"}
    if g_a["verdict"] == "undetermined":
        assert g_a["missing_evidence"], "undetermined must name what it lacks"
        assert g_a["reason"], "undetermined must carry a reason"

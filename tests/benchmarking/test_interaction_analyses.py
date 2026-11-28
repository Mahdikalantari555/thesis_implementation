"""Task 10.7 — the interaction analyses (does the ranking transfer?).

A leaderboard mean can hide the two interactions the change explicitly asks for: the
architecture ranking flipping between anomaly families, and a suppression treatment
helping one objective while hurting another. These tests build synthetic rows with a
*known* interaction and a *known* consistency and fail if the statistics cannot tell
them apart.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.benchmarking import stats as st
from ssl_anomaly.benchmarking.report import interaction_table, suppression_deltas

FAMILIES = ("spectral", "temporal", "phenological", "multivariate", "persistent")


def _row(model_id, paradigm, suppression, cells, *, seed=13):
    """A row shaped like load_cells' output: one manifest, per_cell dict, factors."""
    per_cell = {}
    for cid, (fam, sev, auprc) in cells.items():
        per_cell[cid] = {"cell": cid, "family": fam, "severity": sev, "auprc": auprc,
                         "auroc": min(1.0, auprc + 0.05), "n_scored_bins": 100,
                         "n_anomaly_bins": 10, "aux": {}}
    macro = float(np.mean([v[2] for v in cells.values()]))
    return {"experiment_id": f"{model_id}.{suppression}.I3.W1.C2.V1.farm.full.s{seed}",
            "model_id": model_id, "status": "Completed", "usable": True, "fixture": False,
            "seed": seed, "factors": {"model_id": model_id, "paradigm": paradigm,
                                      "backbone": "gru", "suppression": suppression},
            "metrics": {"macro": {"macro": round(macro, 5)}},
            "per_cell": per_cell}


def _grid_cells(base: dict[str, float]) -> dict:
    """base: family → auprc; families missing from ``base`` become holes in the grid."""
    return {f"{f}__strong": (f, "strong", base[f]) for f in FAMILIES if f in base}


def test_kendall_w_sees_agreement_from_interaction():
    rng = np.random.default_rng(1)
    # no interaction: arm A always best, B second, C worst
    consistent = np.array([[0.7, 0.5, 0.3], [0.72, 0.52, 0.31], [0.69, 0.49, 0.29],
                           [0.71, 0.5, 0.3], [0.7, 0.51, 0.33]])
    w_hi = st.kendall_w(consistent)
    assert w_hi["available"] and w_hi["w"] == pytest.approx(1.0) and w_hi["order"] == [0, 1, 2]
    # full interaction: each family crowns a different arm
    flipped = np.array([[0.9, 0.5, 0.3], [0.3, 0.9, 0.5], [0.5, 0.3, 0.9],
                        [0.5, 0.9, 0.3], [0.3, 0.5, 0.9]])
    w_lo = st.kendall_w(flipped)
    assert w_lo["available"] and w_lo["w"] < 0.5, (
        "a per-block rotation of winners must drive W down; a W that stays high cannot "
        "detect the interaction the section exists to expose")
    small = st.kendall_w(np.array([[0.5, 0.6], [0.6, 0.5]]))
    assert not small["available"] and "≥3" in small["reason"]
    partial = np.array([[0.5, 0.6, np.nan], [0.6, 0.5, 0.4], [0.5, 0.55, 0.35],
                        [0.52, 0.58, 0.36]])
    p = st.kendall_w(partial)
    assert p["available"] and p["n_blocks"] == 3, (
        "W must run over the complete rows and say it dropped one")


def test_interaction_table_detects_a_family_flip_and_names_winners():
    rows = []
    for fam_rank in ("A1", "A5", "A9"):
        base = {f: 0.3 + 0.4 * (fam_rank == f" among {f}" or False) for f in FAMILIES}
    # arm A1 wins spectral+temporal, arm A5 wins the other three: an interaction
    a1 = {f: (0.75 if f in ("spectral", "temporal") else 0.30) for f in FAMILIES}
    a5 = {f: (0.35 if f in ("spectral", "temporal") else 0.80) for f in FAMILIES}
    a9 = {f: 0.5 for f in FAMILIES}
    for seed in (13, 42):
        rows.append(_row("A1", "input_reconstruction", "S0", _grid_cells(a1), seed=seed))
        rows.append(_row("A5", "future_prediction", "S0", _grid_cells(a5), seed=seed))
        rows.append(_row("A9", "hybrid", "S0", _grid_cells(a9), seed=seed))
    res = interaction_table(rows, "model_id")
    assert res["available"] and res["factor"] == "model_id"
    assert res["w"] < 0.5, f"the flip must drive W down (got {res['w']})"
    assert res["winner_per_block"]["spectral"] == "A1"
    assert res["winner_per_block"]["phenological"] == "A5"
    assert res["holes"] == 0 and res["n_blocks"] == 5
    # a uniform winner is the interaction-free control
    flat = [(_row("A1", "x", "S0", _grid_cells({f: 0.8 for f in FAMILIES}), seed=s)) for s in (13, 42)]
    flat += [(_row("A5", "y", "S0", _grid_cells({f: 0.5 for f in FAMILIES}), seed=s)) for s in (13, 42)]
    flat += [(_row("A9", "z", "S0", _grid_cells({f: 0.3 for f in FAMILIES}), seed=s)) for s in (13, 42)]
    res2 = interaction_table(flat, "model_id")
    assert res2["available"] and res2["w"] == 1.0 and set(res2["winner_per_block"].values()) == {"A1"}


def test_interaction_table_refuses_or_reports_holes():
    rows = [_row("A1", "x", "S0", _grid_cells({f: 0.5 for f in FAMILIES})),
            _row("A5", "y", "S0", _grid_cells({f: 0.6 for f in FAMILIES}))]
    res = interaction_table(rows, "model_id")
    assert not res["available"] and "≥3" in res["reason"]
    holey = [_row("A1", "x", "S0", _grid_cells({f: 0.5 for f in FAMILIES})),
             _row("A5", "y", "S0", _grid_cells({f: 0.6 for f in FAMILIES})),
             _row("A9", "z", "S0", _grid_cells({f: 0.7 for f in FAMILIES})),
             _row("A3", "w", "S0", {"spectral__strong": ("spectral", "strong", 0.4)})]
    res2 = interaction_table(holey, "model_id")
    assert not res2["available"] and res2["holes"] == 4, (
        "the missing (A3, family) cells must be counted as holes; with only one complete "
        "family block W is refused in words, not printed as a confident statistic")
    assert "complete blocks" in res2.get("reason") or "W needs" in res2.get("reason")
    # two holes, three families still complete: W runs and the holes stay counted
    sparse = [_row("A1", "x", "S0", _grid_cells({f: 0.5 for f in FAMILIES if f != "temporal"})),
              _row("A5", "y", "S0", _grid_cells({f: 0.6 for f in FAMILIES})),
              _row("A9", "z", "S0", _grid_cells({f: 0.7 for f in FAMILIES})),
              _row("A3", "w", "S0", _grid_cells({f: 0.4 for f in FAMILIES if f != "spectral"}))]
    res3 = interaction_table(sparse, "model_id")
    assert res3["available"] and res3["holes"] == 2
    assert np.isnan(np.asarray(res3["matrix"])).sum() == 2


def test_suppression_deltas_report_sign_consistency():
    rows = []
    # S1 helps A1 (+0.1) but hurts A5 (−0.1): a mixed treatment
    for mid, base, s1 in (("A1", 0.5, 0.6), ("A5", 0.5, 0.4)):
        for sup, gain in (("S0", 0.0), ("S1", s1 - base), ("S2", 0.0), ("S3", 0.02)):
            fams = {f: base + gain for f in FAMILIES}
            rows.append(_row(mid, "x", sup, _grid_cells(fams)))
    sd = suppression_deltas(rows)
    assert sd["available"] and sd["n_models"] == 2
    assert sd["deltas"]["A1"]["S1"] == pytest.approx(0.1, abs=1e-6)
    assert sd["deltas"]["A5"]["S1"] == pytest.approx(-0.1, abs=1e-6)
    assert sd["sign_consistent"]["S1"] is False, (
        "a treatment that helps one model and hurts another must be reported MIXED, "
        "never averaged into 'no effect'")
    assert sd["sign_consistent"]["S2"] is True  # both models unchanged: consistent (no move)
    # no S0 baseline at all → refused in words
    rows2 = [_row("A1", "x", "S1", _grid_cells({f: 0.5 for f in FAMILIES}))]
    sd2 = suppression_deltas(rows2)
    assert not sd2["available"] and "S0" in sd2["reason"]

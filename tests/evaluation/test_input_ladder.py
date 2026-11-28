"""Input-ladder tests (change tasks 1.1-1.5).

Covered here, per task:

- 1.1  B1 ``ndvi_lst_rule``: calibration independence (thresholds move only if
       the calibration subset moves), FAR estimation on clean cycles, AND/OR
       semantics, degenerate-field fallback recorded (never silent).
- 1.2  B2-B5 input builders wired to the v2 channel contract; missing declared
       blocks => ``absent`` with reasons, never zeros.
- 1.3  Causality: no feature at bin o references bins > o; a full-series STL
       builder FAILS the audit (documenting why B5's STL block stays absent).
- 1.4  Capacity manifest: per-rung counts, band + hard-cap enforced.
- 1.5  Scoring bridge: identical ``score(field, bin) -> float`` for B1 and an
       SSL rung; unobserved bins raise; SSL per-bin scores are causal.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.evaluation.data import synthetic_evaluation_split
from ssl_anomaly.evaluation.input_ladder import (
    B1_CHANNELS,
    B1Calibration,
    B1RuleScorer,
    ERA5_CHANNELS,
    RUNGS,
    SSLRungScorer,
    STL_CAUSALITY_NOTE,
    V2_CHANNELS,
    assert_causal_inputs,
    build_rung_inputs,
    calibrate_b1_rule,
    capacity_manifest,
    estimate_cycle_far,
    field_robust_stats,
    ladder_table,
    robust_mad,
    calibration_independence_audit,
    variant_scores,
)
from ssl_anomaly.evaluation.candidates import (
    TrainConfig,
    build_candidate,
    reinit_parameters,
    train_candidate,
)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def table():
    """The synthetic fixture with a REAL phase signal (never random noise)."""
    return synthetic_evaluation_split(n_train=40, n_dev=40, n_fields=10, seed=13)


@pytest.fixture(scope="module")
def calibration(table):
    cd = table
    cal_ids = cd.cycle_ids[: cd.n_cycles // 2]
    return calibrate_b1_rule(cd.X, cd.M, cd.field_ids, cd.cycle_ids, cal_ids), cal_ids


# ---------------------------------------------------------------------------
# 1.1 — B1 rule arm
# ---------------------------------------------------------------------------


def test_b1_calibrates_both_variants_neither_selected(calibration):
    cal, _ = calibration
    assert set(cal) == {"and", "or"}, "BOTH variants must be calibrated; " \
        "selecting one post hoc is the rigged-baseline failure the design names"
    for v, c in cal.items():
        assert isinstance(c, B1Calibration)
        assert c.target_cycle_far == 0.05
        assert c.n_calibration_cycles > 0
        # FAR-matching is a quantile statement: achieved == target on the very
        # subset it was computed on (up to quantile ties).
        assert abs(c.achieved_cycle_far - c.target_cycle_far) < 0.05


def test_b1_calibration_is_independent_of_evaluation_cycles(table):
    cd = table
    cal_ids = cd.cycle_ids[: cd.n_cycles // 2]
    result = calibration_independence_audit(cd.X, cd.M, cd.field_ids, cd.cycle_ids,
                                            cal_ids)
    assert result == {"and": True, "or": True}, (
        "perturbing every NON-calibration cycle moved the thresholds — the "
        "rule is tuned on evaluation data (the exact defect task 1.1 forbids)"
    )


def test_b1_far_estimation_on_clean_cycles():
    # A LARGER fixture than the module default: at n=40 cycles the per-field
    # medians are so noisy that the held-out FAR is dominated by estimation
    # noise, not by the rule (that small-n behaviour is WHY CIs are mandatory).
    cd = synthetic_evaluation_split(n_train=400, n_dev=400, n_fields=25, seed=13)
    cal_ids = cd.cycle_ids[: cd.n_cycles // 2]
    cal = calibrate_b1_rule(cd.X, cd.M, cd.field_ids, cd.cycle_ids, cal_ids)
    scores = variant_scores(cd.X, cd.M, cal["or"].per_field, cd.field_ids)["or"]
    held = np.zeros(cd.n_cycles, dtype=bool)
    held[cd.n_cycles // 2:] = True
    res = estimate_cycle_far(scores, cal["or"].threshold, labels_clean=held)
    assert res["status"] == "measured" and res["population"] == "clean"
    assert abs(res["cycle_far"] - 0.05) < 0.08, res
    assert res["n_unscorable"] >= 0


def test_b1_far_without_clean_mask_is_labelled_not_far(table):
    cd = table
    cal_ids = cd.cycle_ids[: cd.n_cycles // 2]
    cal = calibrate_b1_rule(cd.X, cd.M, cd.field_ids, cd.cycle_ids, cal_ids)
    scores = variant_scores(cd.X, cd.M, cal["or"].per_field, cd.field_ids)["or"]
    res = estimate_cycle_far(scores, cal["or"].threshold)
    assert res["population"] == "all-cycles"
    assert "do not quote as FAR" in res["note"]


def test_b1_and_or_semantics():
    """A pure NDVI drop fires OR but not AND; both moving fires both."""
    x = np.full((1, 6, 8), 0.5)
    x[0, :, 7] = 300.0                      # LST sits at its field median
    m = np.ones((1, 6, 8), dtype=bool)
    from ssl_anomaly.evaluation.input_ladder import FieldStats

    st = {1: FieldStats(field_id=1, ndvi_median=0.5, ndvi_mad=0.05,
                        lst_median=300.0, lst_mad=1.0)}
    x[0, 3, 0] = 0.35                       # NDVI drops 3 MAD
    s = variant_scores(x, m, st, np.array([1]))
    assert s["or"][0, 3] == pytest.approx(3.0)
    assert s["and"][0, 3] == pytest.approx(0.0)   # LST did not move: AND stays 0
    x[0, 3, 7] = 303.0                       # LST rises 3 MAD
    s = variant_scores(x, m, st, np.array([1]))
    assert s["and"][0, 3] == pytest.approx(3.0)
    assert s["or"][0, 3] == pytest.approx(3.0)


def test_b1_degenerate_field_uses_recorded_pooled_fallback():
    """A constant-on-calibration field would make every z infinite; the pooled
    MAD fallback (from fields WITH a usable MAD) must be applied AND recorded,
    never silent."""
    rng = np.random.default_rng(7)
    n = 30
    x = rng.normal(0.5, 0.05, size=(n, 4, 8))
    m = np.ones((n, 4, 8), dtype=bool)
    x[:, :, 7] = 300.0                       # both fields LST constant on calibration
    x[15:, :, 7] = rng.normal(300.0, 1.0, size=(15, 4))   # field 2 has a usable MAD
    field_ids = np.array([1] * 15 + [2] * 15)
    cycle_ids = np.array([f"F{f}:c{i}" for i, f in enumerate(field_ids)])
    cal_ids = np.concatenate([cycle_ids[:10], cycle_ids[15:25]])  # both fields
    stats, pooled = field_robust_stats(x, m, field_ids, cycle_ids, cal_ids)
    st1 = stats[1]
    assert st1.degenerate and "lst" in st1.fallback_channels
    assert st1.lst_mad == pytest.approx(pooled["lst"])
    assert pooled["lst"] > 1e-8               # pooled scale comes from usable fields only
    cal = calibrate_b1_rule(x, m, field_ids, cycle_ids, cal_ids)
    assert cal["or"].degenerate_fields >= 1
    assert "pooled-MAD" in cal["or"].scale_fallback


def test_b1_all_degenerate_channel_refuses_a_made_up_scale():
    """If NO field has a usable MAD on a channel, calibration must fail loudly
    — a fabricated scale would silently define 'anomalous'."""
    rng = np.random.default_rng(7)
    n = 20
    x = rng.normal(0.5, 0.05, size=(n, 4, 8))
    x[:, :, 7] = 300.0                       # every field LST constant
    m = np.ones((n, 4, 8), dtype=bool)
    field_ids = np.array([1] * 10 + [2] * 10)
    cycle_ids = np.array([f"F{f}:c{i}" for i, f in enumerate(field_ids)])
    with pytest.raises(ValueError, match="refusing a made-up scale"):
        field_robust_stats(x, m, field_ids, cycle_ids, cycle_ids[:10])


def test_b1_empty_calibration_subset_raises(table):
    cd = table
    with pytest.raises(ValueError, match="refusing to calibrate"):
        calibrate_b1_rule(cd.X, cd.M, cd.field_ids, cd.cycle_ids,
                          np.array(["nonexistent"]))


def test_b1_unobserved_bin_scores_nan_and_scorer_raises(table):
    cd = table
    cal_ids = cd.cycle_ids[: cd.n_cycles // 2]
    cal = calibrate_b1_rule(cd.X, cd.M, cd.field_ids, cd.cycle_ids, cal_ids)
    scorer = B1RuleScorer(cal["and"], x=cd.X, m=cd.M, field_ids=cd.field_ids)
    # find an unobserved (ndvi & lst) bin
    both = cd.M[:, :, 0] & cd.M[:, :, 7]
    i, o = np.argwhere(~both)[0]
    assert not np.isfinite(scorer.score_cycle(int(cd.field_ids[i]))[o])
    with pytest.raises(ValueError, match="unobserved"):
        scorer.score(int(cd.field_ids[i]), int(o), cycle_index=int(i))


# ---------------------------------------------------------------------------
# 1.2 — builders
# ---------------------------------------------------------------------------


def test_builders_match_the_preregistered_inputs(table):
    cd = table
    b2 = build_rung_inputs("B2", cd.X, cd.M, cd.dsh, channel_names=cd.channel_names)
    b3 = build_rung_inputs("B3", cd.X, cd.M, cd.dsh, channel_names=cd.channel_names)
    assert b2.status == "built" and b3.status == "built"
    # Raw-width contract: the trunk builds [values, mask, cos/sin dsh] itself.
    assert b2.x.shape[2] == 2 and b2.m.shape[2] == 2
    assert b3.x.shape[2] == 8
    assert b2.feature_names[:2] == B1_CHANNELS
    weather = np.zeros((cd.n_cycles, cd.X.shape[1], len(ERA5_CHANNELS)))
    b4 = build_rung_inputs("B4", cd.X, cd.M, cd.dsh, channel_names=cd.channel_names,
                           weather=weather)
    assert b4.x.shape[2] == 8 + len(ERA5_CHANNELS)
    assert b4.m[:, :, 8:].all()             # finite weather arrives observed


def test_b4_b5_absent_without_external_blocks(table):
    cd = table
    b4 = build_rung_inputs("B4", cd.X, cd.M, cd.dsh, channel_names=cd.channel_names)
    b5 = build_rung_inputs("B5", cd.X, cd.M, cd.dsh, channel_names=cd.channel_names)
    assert b4.status == "absent" and "era5_daily" in b4.absent_blocks
    assert b5.status == "absent"
    assert "decomposition_features" in b5.absent_blocks
    assert "causal" in b5.absent_blocks["decomposition_features"] or \
        "causality" in b5.absent_blocks["decomposition_features"], (
        "the absence reason must name the causality rule, not just 'missing'"
        )
    # A rung that is absent cannot be silently used:
    with pytest.raises(ValueError, match="absent"):
        b4.assert_usable()


def test_b5_absence_carries_the_stl_causality_note():
    assert "full cycle" in STL_CAUSALITY_NOTE or "FULL cycle" in STL_CAUSALITY_NOTE
    assert "bins > o" in STL_CAUSALITY_NOTE


def test_unknown_channel_name_raises(table):
    cd = table
    with pytest.raises(ValueError):
        build_rung_inputs("B3", cd.X, cd.M, cd.dsh,
                          channel_names=("ndvi", "swir9", *V2_CHANNELS[2:]))


def test_weather_shape_mismatch_raises(table):
    cd = table
    bad = np.zeros((cd.n_cycles, cd.X.shape[1], 3))
    with pytest.raises(ValueError, match="ERA5_CHANNELS order"):
        build_rung_inputs("B4", cd.X, cd.M, cd.dsh, weather=bad)


# ---------------------------------------------------------------------------
# 1.3 — causality
# ---------------------------------------------------------------------------


def test_every_builder_is_causal(table):
    cd = table
    weather = np.arange(cd.n_cycles * cd.X.shape[1] * len(ERA5_CHANNELS),
                        dtype=np.float64).reshape(
                            cd.n_cycles, cd.X.shape[1], len(ERA5_CHANNELS))
    for rung in ("B2", "B3", "B4"):
        audit = assert_causal_inputs(
            lambda xs, r=rung: build_rung_inputs(
                r, xs, np.ones_like(xs, dtype=bool), cd.dsh,
                channel_names=cd.channel_names,
                weather=weather if RUNGS[r].weather else None,
            ).x,
            cd.X,
        )
        assert audit["causal"], (rung, audit)


def test_full_series_stl_fails_the_causality_audit(table):
    """The reason B5's STL block is absent: a full-series decomposition leaks
    the future at every bin. The audit must fail it — this is the recorded
    evidence, not an opinion."""
    cd = table

    def full_series_stl_like(xs):
        # any per-series full-window transform (here: global z-score) that
        # aggregates over the whole cycle, as STL does
        mu = xs.mean(axis=1, keepdims=True)
        sd = xs.std(axis=1, keepdims=True) + 1e-9
        return (xs - mu) / sd

    audit = assert_causal_inputs(full_series_stl_like, cd.X)
    assert audit["causal"] is False
    assert "leaks the future" in audit["note"]


def test_causal_trailing_window_transform_passes(table):
    cd = table

    def trailing_only(xs):
        out = np.zeros_like(xs)
        for o in range(xs.shape[1]):
            out[:, o] = xs[:, : o + 1].mean(axis=1)
        return out

    assert assert_causal_inputs(trailing_only, cd.X)["causal"] is True


# ---------------------------------------------------------------------------
# 1.4 — capacity accounting
# ---------------------------------------------------------------------------


def test_capacity_manifest_counts_grow_with_input_width_and_enforces_band():
    cap = capacity_manifest()
    assert cap["rungs"]["B1"]["trainable_params"] == 0
    widths = {r: cap["rungs"][r]["n_input_channels"]
              for r in ("B2", "B3", "B4") if r in cap["rungs"]}
    assert widths["B2"] < widths["B3"] < widths["B4"]
    for r in ("B2", "B3", "B4", "B5"):
        row = cap["rungs"][r]
        assert "encoder_params" in row and "total_params" in row
        assert row["hard_cap_ok"], f"{r} exceeds the 250k hard cap"
    assert cap["band_enforced"] is True


def test_capacity_manifest_flags_an_out_of_band_rung():
    cap = capacity_manifest(d_model=16, d_out=8)       # far below the band
    assert cap["band_enforced"] is False
    for r in ("B2", "B3"):
        assert cap["rungs"][r]["band_ok"] is False


def test_ladder_table_lists_every_rung_with_inputs():
    rows = {r["name"]: r for r in ladder_table()}
    assert set(rows) == {"B1", "B2", "B3", "B4", "B5"}
    assert rows["B1"]["model"] == "rule: ndvi_lst_rule"
    assert rows["B1"]["learning"] is False
    assert rows["B4"]["inputs"]["era5_daily"] == list(ERA5_CHANNELS)
    assert rows["B5"]["inputs"]["stl"] == ["stl_trend", "stl_seasonal", "stl_residual"]


# ---------------------------------------------------------------------------
# 1.5 — common scoring bridge
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ssl_scorer(table):
    """A B3-rung model trained briefly on the fixture (pipeline exercise)."""
    cd = table
    from ssl_anomaly.evaluation.candidates import SPECS
    built = build_rung_inputs("B3", cd.X, cd.M, cd.dsh, channel_names=cd.channel_names)
    # B3's raw inputs ARE the full 8-ch table: train the incumbent-pattern
    # model on exactly what the builder produced (widths match by contract).
    model = build_candidate("C-incumbent", n_channels=built.x.shape[2],
                            d_model=32, d_out=16)
    reinit_parameters(model, 13)
    train_candidate(model, SPECS["C-incumbent"], cd,
                    TrainConfig(epochs=2, batch_size=16, seed=13))
    return SSLRungScorer(model, built, field_ids=cd.field_ids), built


def test_bridge_same_interface_for_b1_and_ssl(table, ssl_scorer):
    cd = table
    cal_ids = cd.cycle_ids[: cd.n_cycles // 2]
    cal = calibrate_b1_rule(cd.X, cd.M, cd.field_ids, cd.cycle_ids, cal_ids)
    b1 = B1RuleScorer(cal["or"], x=cd.X, m=cd.M, field_ids=cd.field_ids)
    ssl, built = ssl_scorer
    fid = int(cd.field_ids[0])
    # An OBSERVED bin, found through the bridge itself (bin 5 of row 0 may be
    # unobserved in the fixture — the bridge must raise there, not return 0).
    for scorer in (b1, ssl):
        cyc = scorer.score_cycle(fid)
        o = int(np.argmax(np.isfinite(cyc)))
        v = scorer.score(fid, o)
        assert isinstance(v, float) and np.isfinite(v)
        assert cyc.shape == (cd.X.shape[1],)
        assert cyc[o] == pytest.approx(v)
    # One evaluation loop consumes every rung through the same protocol.
    for scorer in (b1, ssl):
        cyc = scorer.score_cycle(fid)
        row = np.array([scorer.score(fid, o) for o in np.flatnonzero(np.isfinite(cyc))])
        assert row.size > 0


def test_ssl_per_bin_scores_are_causal(table, ssl_scorer):
    """Perturbing bins >= o must not move scores at bins < o (the trunk is
    causal; the scorer must not smuggle the future in via a pooled embedding)."""
    cd = table
    ssl, built = ssl_scorer
    x2 = built.x.copy()
    x2[0, 20:, :] += 5.0
    m2 = built.m.copy()
    s1 = ssl.score_cycle(field_id=int(cd.field_ids[0]), cycle_index=0,
                         x=built.x[0:1], m=built.m[0:1], dsh=built.dsh)
    s2 = ssl.score_cycle(field_id=int(cd.field_ids[0]), cycle_index=0,
                         x=x2[0:1], m=m2[0:1], dsh=built.dsh)
    assert np.allclose(s1[:20], s2[:20], atol=1e-6), (
        "scores at bins < o changed when future bins changed — the scorer "
        "references bins > o and violates task 1.3"
    )


def test_ssl_scorer_rejects_width_mismatch(table, ssl_scorer):
    ssl, _ = ssl_scorer
    x_bad = np.zeros((1, table.X.shape[1], ssl.inputs.x.shape[2] + 3),
                     dtype=np.float32)
    m_bad = np.ones_like(x_bad, dtype=bool)
    with pytest.raises(ValueError, match="d_in"):
        ssl.score_cycle(field_id=1, x=x_bad, m=m_bad,
                        dsh=np.asarray(ssl.inputs.dsh))


def test_ssl_scorer_refuses_absent_rung(table):
    cd = table
    absent = build_rung_inputs("B4", cd.X, cd.M, cd.dsh)      # no weather
    model = build_candidate("C-incumbent", n_channels=4, d_model=16, d_out=8)
    with pytest.raises(ValueError, match="absent"):
        SSLRungScorer(model, absent)


def test_robust_mad_matches_manual_computation():
    v = np.array([1.0, 1.0, 2.0, 3.0, np.nan, 100.0])
    med, mad = robust_mad(v)
    assert med == 2.0
    assert mad == pytest.approx(np.median([1.0, 1.0, 0.0, 1.0, 98.0]) * 1.4826)

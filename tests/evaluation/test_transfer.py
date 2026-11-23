"""Transfer-curve protocol tests (6 arms x 5 fractions x 3 repeats).

Spec anchors: "Fair model comparison", "Missing-evidence rule". Every absent cell
must carry a reason, and no code path may emit a number it did not measure.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.evaluation.transfer import (
    ARMS,
    ArmResult,
    Curve,
    LabelEfficiencyResult,
    TransferConfig,
    arm_table,
    compare_arms,
    run_arm,
    stratified_draw,
)


@pytest.fixture()
def toy():
    """A separable toy problem: label determined by the first embedding dim."""
    rng = np.random.default_rng(11)
    n, d = 240, 6
    f = np.repeat(np.arange(1, 13), 20)
    y = (f % 3 == 0).astype(int)                    # farm-driven label, 1/3 positive
    emb = rng.normal(size=(n, d)) + 1.5 * y[:, None]
    return emb, y, f


def _cfg(**kw):
    base = dict(fractions=(0.05, 0.10, 0.25, 0.50, 1.00), repeats=1, n_boot=60)
    base.update(kw)
    return TransferConfig(**base)


# ---------------------------------------------------------------------------
# Arm matrix
# ---------------------------------------------------------------------------


def test_six_preregistered_arms_are_declared():
    assert len(ARMS) == 6
    assert ARMS == ("A-rule", "A-B1", "A-rand", "A-scratch", "A-ssl-frozen", "A-ssl-ft")
    assert {r["arm"] for r in arm_table()} == set(ARMS)


def test_frozen_input_arms_are_the_declared_ones():
    frozen = {r["arm"] for r in arm_table() if r["inputs_frozen"]}
    assert frozen == {"A-B1", "A-rand", "A-ssl-frozen"}, (
        "A-B1 reads frozen incumbent scores, A-rand and A-ssl-frozen read frozen "
        "representations; A-rule learns nothing and A-scratch/A-ssl-ft train"
    )


def test_arm_names_match_the_preregistration():
    from ssl_anomaly.evaluation.binding import load_preregistration

    prereg = load_preregistration()
    assert tuple(prereg.payload["arms"]) == ARMS, (
        "the arms implemented here must be the six the pre-registration freezes"
    )


def test_a_rule_learns_nothing():
    row = next(r for r in arm_table() if r["arm"] == "A-rule")
    assert row["mechanism"] == "rule"
    assert "no learning" in row["description"]


# ---------------------------------------------------------------------------
# Stratified draws
# ---------------------------------------------------------------------------


def test_draw_covers_every_farm_at_full_fraction(toy):
    _, _, f = toy
    draw = stratified_draw(f, 1.0, repeat=0)
    assert set(np.unique(f[draw.train_idx])) == set(np.unique(f))


def test_draw_is_deterministic(toy):
    _, _, f = toy
    a = stratified_draw(f, 0.25, repeat=1)
    b = stratified_draw(f, 0.25, repeat=1)
    assert np.array_equal(a.train_idx, b.train_idx)


def test_draw_differs_by_repeat(toy):
    _, _, f = toy
    a = stratified_draw(f, 0.25, repeat=0)
    b = stratified_draw(f, 0.25, repeat=1)
    assert not np.array_equal(a.train_idx, b.train_idx)


def test_draw_rejects_bad_fraction(toy):
    _, _, f = toy
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="fraction"):
            stratified_draw(f, bad, repeat=0)


def test_draw_records_realised_size_and_hash(toy):
    _, _, f = toy
    d = stratified_draw(f, 0.10, repeat=2)
    blob = d.to_dict()
    assert blob["n_labels"] == d.train_idx.size
    assert blob["n_available"] == f.size
    assert len(blob["label_indices_sha256_prefix"]) == 8


def test_draw_never_duplicates_an_index(toy):
    _, _, f = toy
    d = stratified_draw(f, 0.50, repeat=0)
    assert np.unique(d.train_idx).size == d.train_idx.size


# ---------------------------------------------------------------------------
# A2 frozen probe
# ---------------------------------------------------------------------------


def test_frozen_probe_learns_a_separable_problem(toy):
    emb, y, f = toy
    cfg = _cfg(repeats=2)
    curve = run_arm("A-ssl-frozen", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f, cfg=cfg)
    full = [c for c in curve.cells if c.measurable and c.fraction == 1.0]
    assert full, curve.to_dict()
    # Toy problem is separable -> AP should clearly beat prevalence.
    assert np.mean([c.ap for c in full]) > 0.6
    assert full[0].prevalence == pytest.approx(y.mean())


def test_frozen_probe_records_ap_above_prevalence_on_learnable_data(toy):
    emb, y, f = toy
    cfg = _cfg(fractions=(1.0,), repeats=1)
    curve = run_arm("A-ssl-frozen", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f, cfg=cfg)
    c = curve.cells[0]
    assert c.ap > c.prevalence


def test_frozen_probe_on_noise_is_near_prevalence():
    rng = np.random.default_rng(5)
    n = 200
    f = np.repeat(np.arange(1, 11), 20)
    y = rng.integers(0, 2, size=n)
    emb = rng.normal(size=(n, 8))
    curve = run_arm("A-ssl-frozen", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f,
                    cfg=_cfg(fractions=(1.0,), repeats=1))
    c = curve.cells[0]
    assert c.ap < c.prevalence * 2.5


def test_f1_threshold_is_reported_with_the_optimism_warning(toy):
    emb, y, f = toy
    curve = run_arm("A-ssl-frozen", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f,
                    cfg=_cfg(fractions=(1.0,), repeats=1))
    c = curve.cells[0]
    assert c.threshold is not None
    assert "optimistic" in c.reason


# ---------------------------------------------------------------------------
# A3 MLP head
# ---------------------------------------------------------------------------


def test_mlp_head_learns_and_is_deterministic(toy):
    emb, y, f = toy
    kw = dict(embeddings=emb, labels=y, field_ids=f, eval_embeddings=emb,
              eval_labels=y, eval_field_ids=f, cfg=_cfg(fractions=(0.5,), repeats=2))
    a = run_arm("A-scratch", **kw)
    b = run_arm("A-scratch", **kw)
    assert [c.ap for c in a.cells if c.measurable] == pytest.approx(
        [c.ap for c in b.cells if c.measurable])


def test_scratch_arm_beats_chance(toy):
    """A-scratch must learn from its own callback, not from the SSL path."""
    from ssl_anomaly.evaluation.transfer import _predict_logistic, _ridge_logistic

    emb, y, f = toy

    def scratch(train_idx, eval_X):
        w = _ridge_logistic(eval_X[train_idx], y[train_idx].astype(float))
        return np.zeros(train_idx.size), _predict_logistic(w, eval_X)

    curve = run_arm("A-scratch", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f,
                    cfg=_cfg(fractions=(1.0,), repeats=1), scratch_fn=scratch)
    assert curve.cells[0].measurable
    assert curve.cells[0].ap > 0.6


# ---------------------------------------------------------------------------
# A-rand, A-rule and A-B1 are controls; each is absent without its input
# ---------------------------------------------------------------------------


def test_a_rand_without_random_init_embeddings_is_absent(toy):
    emb, y, f = toy
    curve = run_arm("A-rand", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f, cfg=_cfg())
    assert all(not c.measurable for c in curve.cells)
    assert "randomly initialised" in curve.cells[0].reason
    assert curve.points() == []


def test_a_rand_with_random_init_embeddings_measures(toy):
    emb, y, f = toy
    rng = np.random.default_rng(9)
    rand = rng.normal(size=emb.shape)          # no label structure
    curve = run_arm("A-rand", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f,
                    cfg=_cfg(fractions=(1.0,), repeats=1),
                    random_init_embeddings=rand)
    assert any(c.measurable for c in curve.cells)


def test_a_b1_without_frozen_scores_is_absent(toy):
    emb, y, f = toy
    curve = run_arm("A-B1", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f, cfg=_cfg())
    assert all(not c.measurable for c in curve.cells)
    assert "b1_scores" in curve.cells[0].reason


def test_a_b1_with_frozen_scores_is_flat_across_fractions(toy):
    """A frozen incumbent cannot use extra labels: its curve must not move."""
    emb, y, f = toy
    curve = run_arm("A-B1", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f,
                    cfg=_cfg(fractions=(0.05, 1.0), repeats=1), b1_scores=emb[:, 0])
    aps = [c.ap for c in curve.cells if c.measurable]
    assert len(aps) == 2
    assert aps[0] == pytest.approx(aps[1])


def test_a_rule_curve_is_flat_by_construction(toy):
    emb, y, f = toy
    curve = run_arm("A-rule", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f,
                    cfg=_cfg(fractions=(0.05, 0.50, 1.00), repeats=1),
                    rule_scores=emb[:, 0])
    aps = {round(c.ap, 6) for c in curve.cells if c.measurable}
    assert len(aps) == 1, "a rule learns nothing, so every fraction is one point"
    assert all("no training" in c.reason for c in curve.cells if c.measurable)


def test_a_rule_without_scores_is_absent(toy):
    emb, y, f = toy
    curve = run_arm("A-rule", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f, cfg=_cfg())
    assert all(not c.measurable for c in curve.cells)
    assert "rule_scores" in curve.cells[0].reason




# ---------------------------------------------------------------------------
# A4/A5 require a real encoder; never silently downgraded
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["A-ssl-ft", "A-scratch"])
def test_training_arms_without_a_callback_are_absent(toy, arm):
    emb, y, f = toy
    curve = run_arm(arm, embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f, cfg=_cfg())
    assert all(not c.measurable for c in curve.cells)
    assert "differentiable training callback" in curve.cells[0].reason


def test_finetune_arm_uses_the_supplied_function(toy):
    emb, y, f = toy
    calls = []

    def ft(train_idx, eval_X):
        calls.append(train_idx.size)
        return np.zeros(train_idx.size), eval_X[:, 0]

    curve = run_arm("A-ssl-ft", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f,
                    cfg=_cfg(fractions=(0.25,), repeats=1), finetune_fn=ft)
    assert calls and curve.cells[0].measurable


def test_a_scratch_uses_its_own_callback(toy):
    """A-scratch must not silently fall back to the SSL training path."""
    emb, y, f = toy
    calls = []

    def scratch(train_idx, eval_X):
        calls.append(train_idx.size)
        return np.zeros(train_idx.size), eval_X[:, 0]

    curve = run_arm("A-scratch", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f,
                    cfg=_cfg(fractions=(0.25,), repeats=1), scratch_fn=scratch)
    assert calls and curve.cells[0].measurable


# ---------------------------------------------------------------------------
# Starvation and single-class draws
# ---------------------------------------------------------------------------


def test_starved_draw_is_absent_not_zero():
    """3 farms x 2 cycles: even the 5% floor yields 3 labels -> a probe cannot fit."""
    rng = np.random.default_rng(2)
    f = np.repeat([1, 2, 3], 2)
    y = np.array([0, 1, 0, 1, 0, 1])
    emb = rng.normal(size=(6, 5))
    curve = run_arm("A-ssl-frozen", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f,
                    cfg=_cfg(fractions=(0.05,), repeats=1))
    c = curve.cells[0]
    assert not c.measurable
    assert "needs >= 4" in c.reason
    assert c.ap is None


def test_realised_fraction_is_recorded_when_it_exceeds_nominal():
    """The stratification floor must never let the x-axis lie about label budget."""
    f = np.repeat(np.arange(1, 21), 2)      # 20 farms, 40 cycles
    d = stratified_draw(f, 0.05, repeat=0)
    blob = d.to_dict()
    assert blob["realised_fraction"] > 0.05        # floor bites
    assert blob["realised_fraction"] == pytest.approx(d.train_idx.size / 40)
    assert blob["n_fields"] == 20


def test_single_class_eval_split_is_absent(toy):
    emb, y, f = toy
    curve = run_arm("A-ssl-frozen", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=np.ones_like(y), eval_field_ids=f,
                    cfg=_cfg(fractions=(1.0,), repeats=1))
    assert not curve.cells[0].measurable
    assert "single class" in curve.cells[0].reason


# ---------------------------------------------------------------------------
# Curve object
# ---------------------------------------------------------------------------


def test_auc_requires_two_measured_points():
    c = Curve("A-ssl-frozen")
    c.cells = [ArmResult("A-ssl-frozen", 1.0, 0, "measured", ap=0.5)]
    assert c.area_under_curve() is None
    c.cells.append(ArmResult("A-ssl-frozen", 0.5, 0, "measured", ap=0.3))
    assert c.area_under_curve() == pytest.approx((0.5 + 0.3) / 2 * 0.5)


def test_absent_cells_are_not_interpolated_into_the_curve():
    c = Curve("A-ssl-frozen")
    c.cells = [
        ArmResult("A-ssl-frozen", 0.05, 0, "absent", reason="starved"),
        ArmResult("A-ssl-frozen", 1.0, 0, "measured", ap=0.4),
    ]
    assert c.points() == [{"fraction": 1.0, "ap": 0.4, "repeat": 0}]
    assert c.absent_fractions()[0]["reason"] == "starved"


def test_headline_reports_absent_when_nothing_measured():
    curve = Curve("A-B1")
    curve.cells = [ArmResult("A-B1", 0, -1, "absent", reason="no features")]
    res = LabelEfficiencyResult({"A-B1": curve}, _cfg(), 10, 10, 0.3)
    h = res.headline("A-B1")
    assert h["status"] == "absent"
    assert h["reason"] == "no features"


def test_headline_aggregates_repeats():
    curve = Curve("A-ssl-frozen")
    curve.cells = [
        ArmResult("A-ssl-frozen", 1.0, 0, "measured", ap=0.4, prevalence=0.3),
        ArmResult("A-ssl-frozen", 1.0, 1, "measured", ap=0.6, prevalence=0.3),
    ]
    res = LabelEfficiencyResult({"A-ssl-frozen": curve}, _cfg(), 10, 10, 0.3)
    h = res.headline()
    assert h["ap_mean"] == pytest.approx(0.5)
    assert h["n_repeats"] == 2


def test_compare_arms_absent_when_no_cell():
    res = LabelEfficiencyResult({}, _cfg(), 10, 10, 0.3)
    out = compare_arms(res, "A-ssl-frozen", "A-B1")
    assert out["present"] is False
    assert "no measured cell" in out["reason"]


def test_full_grid_produces_five_fractions_for_a_measurable_arm(toy):
    emb, y, f = toy
    curve = run_arm("A-ssl-frozen", embeddings=emb, labels=y, field_ids=f,
                    eval_embeddings=emb, eval_labels=y, eval_field_ids=f,
                    cfg=_cfg(repeats=3))
    assert curve.fractions_reported() == [0.05, 0.10, 0.25, 0.50, 1.00]
    assert len(curve.points()) == 15

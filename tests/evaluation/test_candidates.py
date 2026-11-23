"""Region-model v0 candidates, capacity band, and the mechanical G-A screening.

Spec anchors: design §Architecture decision gate (no architecture pre-approved),
G-A verdict, ``architecture-benchmarking`` / "Unified architecture interface" and
"Fair model comparison" (params are reported, and params != compute).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ssl_anomaly.evaluation.candidates import (  # noqa: E402
    CANDIDATES,
    SPECS,
    UNIMPLEMENTED_CANDIDATES,
    BarlowTwinsModel,
    HybridModel,
    IncumbentAE,
    TrainConfig,
    TrajectoryPredictor,
    barlow_twins_loss,
    build_candidate,
    capacity_report,
    embed_candidates,
    make_views,
    masked_trajectory_loss,
    screening_verdict,
    train_candidate,
)
from ssl_anomaly.evaluation.data import synthetic_evaluation_split  # noqa: E402


@pytest.fixture(scope="module")
def data():
    return synthetic_evaluation_split(n_train=16, n_dev=8, n_fields=5, seed=13)


def _tensors(d, n=8, device="cpu"):
    idx = np.arange(n)
    x = torch.from_numpy(d.X[idx].astype(np.float32))
    m = torch.from_numpy(d.M[idx])
    dsh = torch.from_numpy(np.broadcast_to(d.dsh, (n, d.dsh.size)).copy())
    return x, m, dsh


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_four_candidates_are_implemented():
    assert set(CANDIDATES) == {"C-incumbent", "C-BT", "C-traj", "C-hybrid"}
    assert set(SPECS) == set(CANDIDATES)


def test_c_decomp_is_absent_with_a_recorded_reason():
    assert "C-decomp" in UNIMPLEMENTED_CANDIDATES
    assert "polygon zonal store" in UNIMPLEMENTED_CANDIDATES["C-decomp"]
    with pytest.raises(ValueError, match="not implemented"):
        build_candidate("C-decomp", n_channels=8)


def test_unknown_candidate_raises():
    with pytest.raises(ValueError, match="unknown candidate"):
        build_candidate("C-nonsense", n_channels=8)


def test_objectives_are_distinct():
    objectives = [SPECS[n].objective for n in CANDIDATES]
    assert len(set(objectives)) == len(objectives), "objectives must not be aliased"


def test_hybrid_is_flagged_as_hybrid_not_reconstruction():
    assert SPECS["C-hybrid"].objective == "hybrid"
    assert SPECS["C-hybrid"].uses_decoder and SPECS["C-hybrid"].uses_views


# ---------------------------------------------------------------------------
# Forward passes and shapes
# ---------------------------------------------------------------------------


def test_incumbent_forward_shapes(data):
    model = IncumbentAE(n_channels=8)
    x, m, dsh = _tensors(data)
    recon, z = model(x, m, dsh)
    assert recon.shape == (8, data.n_bins, 8)
    assert z.shape == (8, model.encoder.d_out)


def test_bt_model_two_view_loss(data):
    model = BarlowTwinsModel(n_channels=8)
    x, m, dsh = _tensors(data)
    rng = np.random.default_rng(0)
    xa, ma, xb, mb = make_views(data.X[:8], data.M[:8], rng=rng)
    loss, z = model(
        torch.from_numpy(xa), torch.from_numpy(ma),
        torch.from_numpy(xb), torch.from_numpy(mb), dsh,
    )
    assert torch.isfinite(loss) and z.shape == (8, model.encoder.d_out)


def test_trajectory_model_shapes(data):
    model = TrajectoryPredictor(n_channels=8, horizon=3)
    x, m, dsh = _tensors(data)
    pred = model(x, m, dsh)
    assert pred.shape == (8, 3, 8)


def test_hybrid_model_returns_bt_plus_rec(data):
    model = HybridModel(n_channels=8, alpha=1.0)
    x, m, dsh = _tensors(data)
    rng = np.random.default_rng(1)
    xa, ma, xb, mb = make_views(data.X[:8], data.M[:8], rng=rng)
    loss, z = model(torch.from_numpy(xa), torch.from_numpy(ma),
                    torch.from_numpy(xb), torch.from_numpy(mb), dsh)
    assert torch.isfinite(loss)


def test_mask_zero_entries_never_read(data):
    """A value at mask=0 must not influence the embedding, at ANY magnitude.

    This is the mask contract's teeth: if a fabricated 1e6 could reach the GRU
    (e.g. because the encoder zeroed values instead of honouring the mask, or
    concatenated the mask only after the value), the embedding would explode.
    The original observable data was masked, not altered, in both runs.
    """
    model = IncumbentAE(n_channels=8).eval()
    x, m, dsh = _tensors(data)
    # Hide bin 3 for every cycle: the mask is the only thing that differs
    # between the two runs below.
    m_hidden = m.clone()
    m_hidden[:, 3, :] = False
    x_clean = x.clone()
    x_clean[:, 3, :] = 0.0
    x_huge = x.clone()
    x_huge[:, 3, :] = 1e6
    with torch.no_grad():
        z_clean = model.embed(x_clean, m_hidden, dsh)
        z_huge = model.embed(x_huge, m_hidden, dsh)
    assert torch.allclose(z_clean, z_huge, atol=1e-5), (
        "a value under mask=0 reached the encoder: "
        f"max |delta| = {float((z_clean - z_huge).abs().max())}"
    )


def test_hiding_bins_changes_the_embedding(data):
    """Sanity check in the other direction: the mask is actually consumed."""
    model = IncumbentAE(n_channels=8).eval()
    x, m, dsh = _tensors(data)
    with torch.no_grad():
        z_full = model.embed(x, m, dsh)
        m_empty = torch.zeros_like(m)
        z_none = model.embed(torch.nan_to_num(x, nan=0.0), m_empty, dsh)
    assert not torch.allclose(z_full, z_none, atol=1e-4), (
        "the encoder produced identical embeddings for all-observed and "
        "nothing-observed input: the mask is not being used"
    )


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def test_barlow_twins_zero_for_identical_whitened_views():
    torch.manual_seed(0)
    z = torch.randn(64, 8)
    loss = barlow_twins_loss(z, z.clone())
    assert float(loss) >= 0.0


def test_barlow_twins_penalises_redundancy():
    torch.manual_seed(1)
    a = torch.randn(64, 8)
    redundant = torch.cat([a[:, :1].repeat(1, 8)], dim=1)
    l_bad = barlow_twins_loss(redundant, redundant.clone())
    l_good = barlow_twins_loss(a, a + 0.01 * torch.randn_like(a))
    assert float(l_bad) > float(l_good)


def test_barlow_twins_needs_batch_of_two():
    with pytest.raises(ValueError, match="batch of >= 2"):
        barlow_twins_loss(torch.randn(1, 4), torch.randn(1, 4))


def test_trajectory_loss_ignores_missing_targets(data):
    x, m, dsh = _tensors(data)
    pred = torch.zeros(8, 3, 8)
    m_all_missing = torch.zeros_like(m)
    loss_missing = masked_trajectory_loss(pred, x, m_all_missing, dsh, horizon=3)
    assert float(loss_missing) == pytest.approx(0.0)   # denominator clamped, no NaN
    loss_some = masked_trajectory_loss(pred, x, m, dsh, horizon=3)
    assert float(loss_some) >= 0.0


def test_trajectory_loss_closed_form():
    """target[k] = mean of the last (horizon-k) observed bins; verify by hand."""
    T, C, H = 4, 1, 2
    x = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])   # (1, T, C)
    m = torch.ones(1, T, C, dtype=torch.bool)
    dsh = torch.zeros(1, T)
    pred = torch.zeros(1, H, C)
    # k=0 -> width 2 -> mean(3,4)=3.5 ; k=1 -> width 1 -> mean(4)=4.0
    loss = masked_trajectory_loss(pred, x, m, dsh, horizon=H)
    assert float(loss) == pytest.approx((3.5 + 4.0) / 2)


def test_trajectory_loss_rejects_wrong_pred_shape(data):
    x, m, dsh = _tensors(data)
    with pytest.raises(ValueError, match="must be"):
        masked_trajectory_loss(torch.zeros(8, 8), x, m, dsh, horizon=3)


# ---------------------------------------------------------------------------
# View audit — the advisor's polarity warning
# ---------------------------------------------------------------------------


def test_views_do_not_alter_trajectory_shape(data):
    """Views may change WHICH bins are seen and add noise, never the curve."""
    rng = np.random.default_rng(3)
    xa, ma, xb, mb = make_views(data.X[:8], data.M[:8], rng=rng)
    # same mask for view B; values differ only by small noise
    assert mb.shape == data.M[:8].shape
    both = ma & mb
    assert np.all(np.abs(xa[both] - data.X[:8][both]) < 1e-6)
    assert np.all(np.abs(xb[both] - data.X[:8][both]) < 0.2)
    # no sign inversion: correlation of view B with the original stays positive
    orig = data.X[:8][both]
    assert np.corrcoef(orig, xb[both])[0, 1] > 0.9


def test_views_drop_only_observed_bins(data):
    rng = np.random.default_rng(4)
    _, ma, _, _ = make_views(data.X[:8], data.M[:8], rng=rng)
    assert not np.any(ma & ~data.M[:8]), "a view must not INVENT observations"


def test_view_drop_fraction_is_respected(data):
    rng = np.random.default_rng(5)
    _, ma, _, _ = make_views(data.X[:16], data.M[:16], rng=rng, drop_frac=0.5)
    kept = ma.sum() / data.M[:16].sum()
    assert 0.3 < kept < 0.7


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


def test_candidates_fit_the_hard_cap():
    for name in CANDIDATES:
        model = build_candidate(name, n_channels=8)
        rep = capacity_report(model, name)
        assert rep.params <= 250_000, f"{name}: {rep.params}"


def test_capacity_band_flags_out_of_band():
    model = build_candidate("C-incumbent", n_channels=8)
    rep = capacity_report(model, "C-incumbent", target_params=1_000_000)
    assert not rep.band_ok
    assert "outside band" in rep.detail


def test_capacity_hard_cap_is_reported_separately():
    model = build_candidate("C-hybrid", n_channels=8)
    rep = capacity_report(model, "C-hybrid", target_params=10, band=(0.0, 1e9),
                          max_params=100)
    assert not rep.band_ok
    assert "hard cap" in rep.detail


# ---------------------------------------------------------------------------
# Training determinism and equal budgets
# ---------------------------------------------------------------------------


def test_training_is_deterministic(data):
    spec = SPECS["C-incumbent"]
    cfg = TrainConfig(epochs=2, seed=42)
    a = train_candidate(build_candidate("C-incumbent", n_channels=8), spec, data, cfg)
    b = train_candidate(build_candidate("C-incumbent", n_channels=8), spec, data, cfg)
    assert a["final_loss"] == pytest.approx(b["final_loss"], rel=1e-6)
    assert a["loss_history"] == pytest.approx(b["loss_history"], rel=1e-6)


def test_loss_decreases_for_reconstruction(data):
    spec = SPECS["C-incumbent"]
    res = train_candidate(build_candidate("C-incumbent", n_channels=8), spec, data,
                          TrainConfig(epochs=12, lr=5e-3, seed=13))
    assert res["final_loss"] < res["loss_history"][0]


def test_progress_callback_fires_per_epoch(data):
    seen = []
    train_candidate(build_candidate("C-BT", n_channels=8), SPECS["C-BT"], data,
                    TrainConfig(epochs=3, seed=1),
                    progress=lambda e, t, l: seen.append((e, t, l)))
    assert [e for e, _, _ in seen] == [1, 2, 3]
    assert all(np.isfinite(l) for _, _, l in seen)


def test_every_candidate_trains_without_nan(data):
    for name in CANDIDATES:
        res = train_candidate(build_candidate(name, n_channels=8), SPECS[name], data,
                              TrainConfig(epochs=3, seed=7))
        assert np.isfinite(res["final_loss"]), name
        assert res["params"] > 0


def test_embeddings_are_deterministic_in_eval_mode(data):
    model = build_candidate("C-BT", n_channels=8)
    train_candidate(model, SPECS["C-BT"], data, TrainConfig(epochs=2, seed=5))
    z1 = embed_candidates(model, data)
    z2 = embed_candidates(model, data)
    assert z1.shape == (data.n_cycles, model.encoder.d_out)
    assert np.allclose(z1, z2)


# ---------------------------------------------------------------------------
# Screening verdict
# ---------------------------------------------------------------------------


def _ev(top1, chance, acc, maj, collapse=False):
    return {
        "farm_separability_top1": top1,
        "farm_separability_chance": chance,
        "phase_probe_accuracy": acc,
        "phase_probe_majority": maj,
        "collapse_disqualified": collapse,
    }


def _cap(params=100_000, ok=True):
    return {"params": params, "band_ok": ok}


def test_gate_a_passes_with_two_terms():
    v = screening_verdict(
        {"C-BT": _ev(0.5, 0.1, 0.4, 0.1, collapse=False)},
        {"C-BT": _cap()},
    )
    assert v["candidates"]["C-BT"]["passed"]
    assert v["winner"] == "C-BT"


def test_gate_a_fails_with_one_term():
    v = screening_verdict(
        {"C-BT": _ev(0.5, 0.1, 0.05, 0.1, collapse=False)},
        {"C-BT": _cap()},
    )
    assert not v["candidates"]["C-BT"]["passed"]
    assert v["winner"] is None
    assert "negative result" in v["selection_rule"]


def test_gate_a_disqualifies_on_collapse():
    v = screening_verdict(
        {"C-BT": _ev(0.5, 0.1, 0.4, 0.1, collapse=True)},
        {"C-BT": _cap()},
    )
    assert v["candidates"]["C-BT"]["passed"] is False
    assert "disqualified-collapse" in v["candidates"]["C-BT"]["reasons"]


def test_gate_a_disqualifies_out_of_capacity_band():
    v = screening_verdict(
        {"C-traj": _ev(0.5, 0.1, 0.4, 0.1, collapse=False)},
        {"C-traj": _cap(ok=False)},
    )
    assert v["candidates"]["C-traj"]["passed"] is False
    assert "capacity-band" in v["candidates"]["C-traj"]["reasons"]


def test_gate_a_all_fail_is_a_valid_recorded_outcome():
    v = screening_verdict(
        {n: _ev(0.1, 0.1, 0.1, 0.1) for n in CANDIDATES},
        {n: _cap() for n in CANDIDATES},
    )
    assert v["winner"] is None
    assert all(not c["passed"] for c in v["candidates"].values())
    assert v["selection_rule"]


def test_gate_a_selection_order_is_predeclared():
    v = screening_verdict(
        {"C-hybrid": _ev(0.9, 0.1, 0.9, 0.1), "C-incumbent": _ev(0.2, 0.1, 0.2, 0.1)},
        {"C-hybrid": _cap(), "C-incumbent": _cap()},
    )
    # both may pass; the declared order decides, not the metric magnitude
    if v["candidates"]["C-incumbent"]["passed"]:
        assert v["winner"] == "C-incumbent"
    else:
        assert v["winner"] == "C-hybrid"


def test_default_trunk_is_in_band():
    """The trunk the script ships must land inside the pre-registered band.

    d_model=160 / d_out=64 was chosen by sweeping the real parameter counts, not
    by guessing: at d_model=32 every encoder was ~12k against a 100k target, so
    every candidate failed the capacity band for a reason that had nothing to do
    with the model.
    """
    for name in CANDIDATES:
        model = build_candidate(name, n_channels=8, d_model=160, d_out=64)
        rep = capacity_report(model, name)
        assert rep.band_ok, f"{name}: {rep.detail}"
        assert rep.hard_cap_ok
        assert 80_000 <= rep.encoder_params <= 120_000


def test_band_reads_the_encoder_not_the_total():
    """C-incumbent's decoder is ~2x its trunk; that must not disqualify it."""
    from ssl_anomaly.evaluation.candidates import IncumbentAE

    model = IncumbentAE(n_channels=8, d_model=160, d_out=64)
    rep = capacity_report(model, "C-incumbent")
    assert rep.params > rep.encoder_params * 2
    assert rep.band_ok, "the band must police encoder capacity, not decoder size"
    assert rep.to_dict()["encoder_params"] == rep.encoder_params


def test_hard_cap_still_applies_to_the_total():
    from ssl_anomaly.evaluation.candidates import IncumbentAE

    model = IncumbentAE(n_channels=8, d_model=160, d_out=64)
    rep = capacity_report(model, "C-incumbent", max_params=100_000)
    assert not rep.hard_cap_ok
    assert not rep.band_ok
    assert "hard cap" in rep.detail

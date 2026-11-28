"""Research-arm candidate tests (change tasks 3.4-3.6).

- C-stat (Level A): 58 features measured from the store shape (the research's
  "61-dim" recorded as a discrepancy, not propagated); absent store -> absent
  record; non-finite summaries refused (never zero-filled); no trainable path.
- C-patch (Level C): patched transformer with Δt-aware tokens; sized INSIDE
  the frozen capacity band; the 1.5M research ceiling recorded, not applied.
- C-film (Level E): base frozen by construction; only adapters train; a farm
  outside the adapter table raises (never a zero adapter); film-recon trains
  through the shared loop with real gradients on adapter parameters ONLY.
- Screening verdict: research arms join additively AFTER the pre-registered
  four; C-stat carries evidence but can never be the winner; absent candidates
  are recorded with reasons.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ssl_anomaly.evaluation.candidates import (
    SPECS,
    CandidateSpec,
    TrainConfig,
    build_candidate,
    embed_candidates,
    reinit_parameters,
    screening_verdict,
    train_candidate,
)
from ssl_anomaly.evaluation.data import synthetic_evaluation_split
from ssl_anomaly.evaluation.metrics import count_parameters
from ssl_anomaly.evaluation.research_arms import (
    DEFAULT_STATS_ZARR,
    FULL_CANDIDATE_ORDER,
    N_SUMMARY_FEATURES,
    RESEARCH_ARMS,
    FiLMFieldAdapter,
    PatchTransformer,
    SummariesUnavailable,
    SummaryEncoder,
    build_cycle_summaries,
    build_research_arm,
    patch_capacity,
)


# ---------------------------------------------------------------------------
# C-stat (Level A)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_stats_store(tmp_path_factory):
    """A minimal stats.zarr-shaped store: 2 fields x 6 dates x 7 bands."""
    import zarr

    root = tmp_path_factory.mktemp("stats")
    g = zarr.open_group(str(root / "stats.zarr"), mode="w")
    f, t, b = 2, 6, 7
    rng = np.random.default_rng(13)
    g["time"] = np.array([0, 10, 20, 30, 40, 50], dtype=np.int64)
    g["band"] = np.array(["blue", "green", "red", "nir", "swir1", "swir2", "lst"])
    for stat in ("mean", "median", "std", "p10", "p25", "p75", "p90", "iqr"):
        g[stat] = rng.normal(0.5, 0.1, size=(f, t, b)).astype(np.float32)
    g["valid_frac"] = rng.random((f, t)).astype(np.float32)
    g["n_pixels"] = rng.integers(10, 400, size=(f, t)).astype(np.int64)
    return root / "stats.zarr"


def test_summary_width_is_measured_not_copied_from_prose():
    assert N_SUMMARY_FEATURES == 58
    assert "does not match the store" in build_cycle_summaries.__doc__ + \
        str(RESEARCH_ARMS["C-stat"].notes)
    assert "61" in RESEARCH_ARMS["C-stat"].notes


def test_missing_stats_store_is_recorded_absent():
    res = build_cycle_summaries("/nonexistent/stats.zarr",
                                cycles_t_start=np.array([0]),
                                cycles_t_end=np.array([5]),
                                cycles_field=np.array([0]))
    assert res["status"] == "absent" and "not found" in res["reason"]


def test_cycle_summaries_from_tiny_store(tiny_stats_store):
    res = build_cycle_summaries(
        tiny_stats_store,
        cycles_t_start=np.array([0, 10, 25, 0]),
        cycles_t_end=np.array([12, 55, 30, 5]),
        cycles_field=np.array([0, 1, 1, 5]),     # last cycle: unknown field
    )
    assert res["status"] == "built"
    assert res["z"].shape == (4, N_SUMMARY_FEATURES)
    # cycles 0,1,2 span store dates -> present; cycle 3 has no store row 5.
    assert res["present"].tolist() == [True, True, True, False]
    assert res["n_dates_used"].tolist()[0] >= 2
    assert len(res["feature_names"]) == N_SUMMARY_FEATURES


def test_summary_encoder_refuses_nonfinite_and_training():
    with pytest.raises(ValueError, match="non-finite"):
        SummaryEncoder(np.array([[np.nan] * 58], dtype=np.float32))
    enc = SummaryEncoder(np.random.default_rng(1).normal(size=(4, 58)).astype(np.float32))
    z = enc.embed(np.array([0, 3]))
    assert z.shape == (2, 58)
    with pytest.raises(NotImplementedError, match="no pretext"):
        enc.forward()


def test_train_candidate_refuses_to_train_the_statistical_arm():
    cd = synthetic_evaluation_split(n_train=8, n_dev=4, n_fields=4, seed=13)
    model = SummaryEncoder(np.random.default_rng(1).normal(
        size=(cd.n_cycles, 58)).astype(np.float32))
    spec = CandidateSpec("C-stat", "statistical", False, False)
    res = train_candidate(model, spec, cd, TrainConfig(epochs=3, seed=13))
    assert res["trained"] is False and res["final_loss"] is None
    assert res["params"] == 0


# ---------------------------------------------------------------------------
# C-patch (Level C)
# ---------------------------------------------------------------------------


def test_patch_capacity_inside_the_frozen_band_research_ceiling_recorded():
    cap = patch_capacity(n_channels=8)
    assert cap["band_ok"] is True, cap
    assert cap["hard_cap_ok"] is True
    assert cap["research_ceiling_params"] == 1_500_000
    assert cap["encoder_params"] <= 1.5 * cap["band"][1]
    assert "250k" in cap["note"] or "250 k" in cap["note"] or "250k band" in cap["note"] or \
        "band governs" in cap["note"]


def test_patch_forward_and_embed_shapes():
    torch.manual_seed(13)
    model = PatchTransformer(8)
    x = torch.randn(4, 36, 8)
    m = torch.ones(4, 36, 8, dtype=torch.bool)
    dsh = (torch.arange(36, dtype=torch.float32) + 0.5) * 15.0
    recon, z = model(x, m, dsh)
    assert recon.shape == (4, 36, 8)
    assert z.shape == (4, model.proj.out_features)
    assert torch.isfinite(recon).all() and torch.isfinite(z).all()


def test_patch_trains_via_reconstruction_objective_no_nan():
    cd = synthetic_evaluation_split(n_train=12, n_dev=6, n_fields=4, seed=13)
    model = PatchTransformer(cd.X.shape[2], patch_len=6, d_model=32, n_heads=4,
                             d_ff=64, d_out=16)
    spec = CandidateSpec("C-patch", "reconstruction", False, True)
    res = train_candidate(model, spec, cd, TrainConfig(epochs=2, batch_size=8, seed=13))
    assert len(res["loss_history"]) == 2
    assert np.isfinite(res["final_loss"]), res
    z = embed_candidates(model, cd)
    assert z.shape[0] == cd.n_cycles


def test_patch_embed_is_dsh_aware():
    """The Δt-aware encoding must change the embedding when the phase axis
    changes (two otherwise identical cycles at different DSH offsets)."""
    torch.manual_seed(13)
    model = PatchTransformer(8, d_model=32, n_heads=4, d_ff=64, d_out=16)
    model.eval()
    x = torch.randn(1, 36, 8)
    m = torch.ones(1, 36, 8, dtype=torch.bool)
    d1 = (torch.arange(36, dtype=torch.float32) + 0.5) * 15.0
    d2 = torch.roll(d1, 6)                                   # half-patch phase shift
    with torch.no_grad():
        z1 = model.embed(x, m, d1)
        z2 = model.embed(x, m, d2)
    assert not torch.allclose(z1, z2, atol=1e-6)


# ---------------------------------------------------------------------------
# C-film (Level E)
# ---------------------------------------------------------------------------


def test_film_base_frozen_only_adapters_receive_gradients():
    from ssl_anomaly.evaluation.candidates import CycleEncoder

    cd = synthetic_evaluation_split(n_train=12, n_dev=6, n_fields=3, seed=13)
    base = build_candidate("C-incumbent", n_channels=cd.X.shape[2],
                           d_model=32, d_out=16)
    reinit_parameters(base, 13)
    film = FiLMFieldAdapter(base, n_farms=int(cd.field_ids.max()) + 1,
                            d_model=32, farm_embed_dim=8,
                            n_channels=cd.X.shape[2])
    # Frozen at construction:
    assert all(not p.requires_grad for p in film.base.parameters())
    assert film.adapter_parameters() > 0
    assert film.adapter_parameters() == sum(
        p.numel() for p in film.parameters() if p.requires_grad)
    # And a film-recon step moves ONLY adapter parameters:
    opt = torch.optim.AdamW([p for p in film.parameters() if p.requires_grad], lr=1e-2)
    x = torch.from_numpy(cd.X[:4].astype(np.float32))
    m = torch.from_numpy(cd.M[:4])
    dsh = torch.from_numpy(np.broadcast_to(
        cd.dsh.astype(np.float32), (4, cd.dsh.size)).copy())
    farms = cd.field_ids[:4]
    recon = film(x, m, dsh, farms)
    loss = (recon - torch.nan_to_num(x)).abs().mean()
    loss.backward()
    base_before = {n: p.clone() for n, p in film.base.named_parameters()}
    opt.step()
    for n, p in film.base.named_parameters():
        assert torch.equal(base_before[n], p), f"base parameter {n} moved"
    adapter_moved = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in film.parameters() if p.requires_grad)
    assert adapter_moved


def test_film_unknown_farm_raises_never_zero_adapter():
    base = build_candidate("C-incumbent", n_channels=8, d_model=16, d_out=8)
    film = FiLMFieldAdapter(base, n_farms=2, d_model=16, farm_embed_dim=4,
                            n_channels=8)
    x = torch.zeros(1, 36, 8)
    m = torch.ones(1, 36, 8, dtype=torch.bool)
    dsh = torch.full((1, 36), 270.0)
    with pytest.raises(ValueError, match="outside the adapter table"):
        film.embed(x, m, dsh, np.array([99]))


def test_film_trains_through_the_shared_loop():
    cd = synthetic_evaluation_split(n_train=12, n_dev=6, n_fields=3, seed=13)
    base = build_candidate("C-incumbent", n_channels=cd.X.shape[2],
                           d_model=32, d_out=16)
    reinit_parameters(base, 13)
    film = FiLMFieldAdapter(base, n_farms=int(cd.field_ids.max()) + 1,
                            d_model=32, farm_embed_dim=8,
                            n_channels=cd.X.shape[2])
    spec = CandidateSpec("C-film", "film-recon", False, True)
    res = train_candidate(film, spec, cd, TrainConfig(epochs=2, batch_size=8, seed=13))
    assert len(res["loss_history"]) == 2
    assert np.isfinite(res["final_loss"])
    z = embed_candidates(film, cd)
    assert z.shape == (cd.n_cycles, film.d_out)


def test_film_requires_field_ids():
    cd = synthetic_evaluation_split(n_train=8, n_dev=4, n_fields=3, seed=13)
    base = build_candidate("C-incumbent", n_channels=cd.X.shape[2],
                           d_model=16, d_out=8)
    film = FiLMFieldAdapter(base, n_farms=3, d_model=16, farm_embed_dim=4,
                            n_channels=cd.X.shape[2])
    spec = CandidateSpec("C-film", "film-recon", False, True)
    stripped = {k: v for k, v in cd.__dict__.items()}
    del stripped["field_ids"]
    class NoFields:
        pass
    nf = NoFields()
    nf.X, nf.M, nf.dsh, nf.n_cycles = cd.X, cd.M, cd.dsh, cd.n_cycles
    with pytest.raises(ValueError, match="field_ids"):
        train_candidate(film, spec, nf, TrainConfig(epochs=1, batch_size=8, seed=13))


# ---------------------------------------------------------------------------
# Registry / verdict integration
# ---------------------------------------------------------------------------


def test_research_arms_join_after_the_preregistered_four():
    assert FULL_CANDIDATE_ORDER[:4] == ("C-incumbent", "C-BT", "C-traj", "C-hybrid")
    assert FULL_CANDIDATE_ORDER[4:] == ("C-stat", "C-patch", "C-film")


def test_verdict_records_absent_candidates_and_never_selects_cstat():
    evidence = {
        "C-incumbent": {"farm_separability_top1": 0.9, "farm_separability_chance": 0.1,
                        "phase_probe_accuracy": 0.9, "phase_probe_majority": 0.1,
                        "collapse_disqualified": False},
        "C-stat": {"farm_separability_top1": 0.95, "farm_separability_chance": 0.1,
                   "phase_probe_accuracy": 0.99, "phase_probe_majority": 0.1,
                   "collapse_disqualified": False},
    }
    capacity = {
        "C-incumbent": {"params": 100, "encoder_params": 90, "band_ok": True},
        "C-stat": {"params": 0, "encoder_params": 0, "band_ok": True},
    }
    v = screening_verdict(
        evidence, capacity,
        absent_candidates={"C-patch": "stats store absent in smoke"},
    )
    assert v["absent_candidates"] == {"C-patch": "stats store absent in smoke"}
    # C-stat beats BOTH probe terms decisively — and must still NOT be the
    # winner: it is the statistical reference, not an SSL candidate.
    assert v["candidates"]["C-stat"]["terms_passed"] == 2
    assert v["candidates"]["C-stat"]["passed"] is False
    assert any(r.startswith("non-selectable") for r in v["candidates"]["C-stat"]["reasons"])
    assert v["winner"] == "C-incumbent"
    assert "SELECTABLE" in v["selection_rule"]


def test_build_research_arm_fail_closed():
    with pytest.raises(ValueError, match="unknown research arm"):
        build_research_arm("C-nope", 8)
    with pytest.raises(ValueError, match="summaries"):
        build_research_arm("C-stat", 8, summaries=None)
    with pytest.raises(ValueError, match="n_farms"):
        build_research_arm("C-film", 8, base=build_candidate("C-incumbent", 8))
    with pytest.raises(ValueError, match="base encoder"):
        build_research_arm("C-film", 8, n_farms=3)

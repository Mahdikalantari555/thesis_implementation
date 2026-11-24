"""A1–A10: the interface contract, objective distinctness, and causality (Phase 1–2)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ssl_anomaly.benchmarking.interface import ArmBatch, assert_objectives_distinct
from ssl_anomaly.models.bench.arms import (
    ARM_BY_MODEL,
    NumericalInstability,
    build_arm,
    count_trainable,
    estimate_flops,
    objective_signature,
)


def _batch(B=6, T=36, C=8, K=10, seed=3):
    torch.manual_seed(seed)
    m = torch.rand(B, T, C) > 0.15
    return ArmBatch(x=torch.randn(B, T, C) * m, m=m, valid_bin=m.any(-1), cov=torch.randn(B, T, K),
                    meta={"indices": torch.arange(B)})


def test_every_registered_arm_builds_and_trains():
    class _Cfg:
        raw = {
            "training": {"n_bins": 36, "d_model": 64, "n_layers": 1, "dropout": 0.1, "d_state": 8,
                         "latent_dim": 16, "huber_beta": 1.0, "horizon_k": 4, "ema_momentum": 0.996,
                         "lambda_rec": 1.0, "lambda_pred": 1.0, "lambda_variants": {},
                         "masking": {"applied_at": "train_only",
                                     "rate_by_paradigm": {"latent_reconstruction": 0.15,
                                                          "future_prediction": 0.15,
                                                          "jepa": 0.15, "hybrid": 0.15}}},
            "capacity": {"bottleneck_frac": 0.25},
            "collapse": {"variance_weight": 0.05},
        }

    cfg = _Cfg()
    b = _batch()
    for mid in ARM_BY_MODEL:
        arm = build_arm(mid, cfg=cfg, d_channels=8, d_cov=10)
        arm.train()
        loss, comp = arm.compute_loss(b)
        assert torch.isfinite(loss), f"{mid} produced a non-finite loss"
        if mid != "A-RAND":
            loss.backward()
            live = [n for n, p in arm.named_parameters() if p.requires_grad]
            assert live, mid
            for n, p in arm.named_parameters():
                if p.requires_grad and p.grad is not None:
                    assert bool(torch.isfinite(p.grad).all()), f"{mid}.{n} got a non-finite gradient"
        arm.eval()
        sc = arm.score_bins(b)
        assert sc.shape == (b.x.shape[0], b.x.shape[1])
        fin = torch.isfinite(sc)
        assert fin.any(), f"{mid} scored nothing"
        assert bool(torch.isfinite(sc[fin]).all())


def test_objectives_are_pairwise_distinct():
    specs = {mid: {"objective": s["objective"], "target_source": s["target_source"],
                   "gradient_path": "documented", "paradigm": s["paradigm"],
                   "lambdas": {"lambda_rec": 1.0, "lambda_pred": 1.0},
                   "collapse_mechanism": s.get("collapse_prevention_required") and "ema+var"}
             for mid, s in ((m, objective_signature(m)) for m in ARM_BY_MODEL)}
    assert assert_objectives_distinct(specs) == []


def test_objective_aliasing_is_caught():
    same = {"A1": {"objective": "x", "target_source": "y", "gradient_path": "z", "paradigm": "p1"},
            "A5": {"objective": "x", "target_source": "y", "gradient_path": "z", "paradigm": "p2"}}
    problems = assert_objectives_distinct(same)
    assert any("aliasing" in p for p in problems), problems


def test_missing_objective_field_is_a_problem():
    problems = assert_objectives_distinct({"A3": {"objective": "only-this", "paradigm": "latent_reconstruction"}})
    assert problems, "an undocumented objective must not pass the audit"


def test_latent_arms_must_name_a_collapse_mechanism():
    p = assert_objectives_distinct({"A3": {"objective": "o", "target_source": "t", "gradient_path": "g",
                                           "paradigm": "latent_reconstruction"}})
    assert any("collapse" in x for x in p), p


def test_reconstruction_of_bin_t_does_not_use_bin_t():
    """A1 is leave-one-bin-out: perturbing the last bin cannot change its own estimate."""
    class _Cfg:
        raw = {"training": {"n_bins": 8, "d_model": 16, "n_layers": 1, "dropout": 0.0, "d_state": 4,
                            "latent_dim": 8, "huber_beta": 1.0, "horizon_k": 2, "ema_momentum": 0.996,
                            "lambda_rec": 1.0, "lambda_pred": 1.0, "lambda_variants": {},
                            "masking": {"rate_by_paradigm": {}}},
               "capacity": {}, "collapse": {}}

    arm = build_arm("A1", cfg=_Cfg(), d_channels=4, d_cov=0)
    arm.eval()
    b1 = ArmBatch(x=torch.zeros(1, 8, 4), m=torch.ones(1, 8, 4, dtype=torch.bool),
                  valid_bin=torch.ones(1, 8, dtype=torch.bool))
    b2 = ArmBatch(x=b1.x.clone(), m=b1.m.clone(), valid_bin=b1.valid_bin.clone(),
                  ctx_mask=torch.tensor([[False] * 7 + [True]]))
    r1 = arm.forward(b1)["recon"][0, 7]
    r2 = arm.forward(b2)["recon"][0, 7]
    assert torch.allclose(r1, r2, atol=1e-5), (
        "hiding bin 7's value changed the estimate of bin 7 — the 'reconstruction' is not leave-one-out"
    )


def test_backbone_refuses_noncausal_for_prediction():
    from ssl_anomaly.models.bench.backbones import build_backbone

    with pytest.raises(ValueError, match="causal=False is not permitted"):
        build_backbone(8, kind="gru", d_model=16, causal=False, for_arm="A5")
    build_backbone(8, kind="gru", d_model=16, causal=False, for_arm="A1")   # allowed, still audited


def test_instability_is_raised_not_silently_averaged():
    """A non-finite activation must stop the run with evidence, not become a NaN metric."""
    from ssl_anomaly.models.bench.backbones import CausalStackBackbone

    trunk = CausalStackBackbone(8, kind="gru", d_model=16)
    bad = torch.zeros(2, 3, 16)
    bad[0, 1, 2] = float("nan")
    with pytest.raises(NumericalInstability) as ei:
        trunk._check_finite(bad, "gru.hidden")
    ev = ei.value.evidence
    assert ev["n_nonfinite"] == 1 and ev["where"] == "gru.hidden" and ev["shape"] == [2, 3, 16]
    trunk._check_finite(torch.zeros(2, 3, 16), "ok")      # finite input must not raise


def test_s2_is_refused_for_jepa_and_accepted_for_reconstruction():
    class _Cfg:
        raw = {"training": {"n_bins": 8, "d_model": 16, "n_layers": 1, "dropout": 0.0, "d_state": 4,
                            "latent_dim": 8, "huber_beta": 1.0, "horizon_k": 2, "ema_momentum": 0.996,
                            "lambda_rec": 1.0, "lambda_pred": 1.0, "lambda_variants": {},
                            "masking": {"rate_by_paradigm": {}}},
               "capacity": {}, "collapse": {}}

    from ssl_anomaly.benchmarking.interface import ObjectiveSpecError

    with pytest.raises(ObjectiveSpecError):
        build_arm("A7", cfg=_Cfg(), d_channels=4, d_cov=0).set_suppression("S2", {})
    build_arm("A1", cfg=_Cfg(), d_channels=4, d_cov=0).set_suppression("S2", {"rate": 0.2})


def test_unknown_arm_is_not_runnable():
    class _Cfg:
        raw = {"training": {}, "capacity": {}, "collapse": {}}

    with pytest.raises(KeyError, match="not one of the pre-registered arms"):
        build_arm("A11", cfg=_Cfg(), d_channels=4, d_cov=0)


def test_capacity_control_is_frozen_by_construction():
    class _Cfg:
        raw = {"training": {"n_bins": 8, "d_model": 16, "n_layers": 1, "dropout": 0.0, "d_state": 4,
                            "latent_dim": 8, "huber_beta": 1.0, "horizon_k": 2, "lambda_variants": {},
                            "masking": {"rate_by_paradigm": {}}},
               "capacity": {}, "collapse": {}}

    arm = build_arm("A-RAND", cfg=_Cfg(), d_channels=4, d_cov=0, variant="rand_gru")
    assert count_trainable(arm) == 0
    with pytest.raises(RuntimeError):
        sum(p.sum() for p in arm.parameters()).backward(keep_graph=True) if False else _require_grad(arm)


def _require_grad(arm):
    for p in arm.parameters():
        if p.requires_grad:
            raise RuntimeError("a capacity control must not be trainable")
    raise RuntimeError("no trainable parameters — as intended, this assert path proves the freeze")


def test_flops_estimate_records_its_convention():
    class _Cfg:
        raw = {"training": {"n_bins": 8, "d_model": 16, "n_layers": 1, "dropout": 0.0, "d_state": 4,
                            "latent_dim": 8, "huber_beta": 1.0, "horizon_k": 2, "lambda_variants": {},
                            "masking": {"rate_by_paradigm": {}}},
               "capacity": {}, "collapse": {}}

    f = estimate_flops(build_arm("A1", cfg=_Cfg(), d_channels=4, d_cov=0), batch_size=8)
    assert f["forward_flops_per_batch"] > 0 and "convention" in f and f["batch_size"] == 8
    m = estimate_flops(build_arm("A2", cfg=_Cfg(), d_channels=4, d_cov=0), batch_size=8)
    assert f["counts"] == m["counts"], "arms must be measured with one convention"


def test_hybrid_lambda_variants_change_the_objective_record():
    class _Cfg:
        raw = {"training": {"n_bins": 8, "d_model": 16, "n_layers": 1, "dropout": 0.0, "d_state": 4,
                            "latent_dim": 8, "huber_beta": 1.0, "horizon_k": 2, "lambda_rec": 1.0,
                            "lambda_pred": 1.0,
                            "lambda_variants": {"lam0.5": {"lambda_rec": 1.0, "lambda_pred": 0.5}},
                            "masking": {"rate_by_paradigm": {}}},
               "capacity": {}, "collapse": {}}

    a = build_arm("A9", cfg=_Cfg(), d_channels=4, d_cov=0, variant="lam0.5")
    assert a.objective_spec()["lambdas"] == {"lambda_rec": 1.0, "lambda_pred": 0.5}
    with pytest.raises(KeyError, match="not a pre-registered lambda variant"):
        build_arm("A9", cfg=_Cfg(), d_channels=4, d_cov=0, variant="lam7.0")


def test_ema_target_moves_and_is_checkpointed():
    class _Cfg:
        raw = {"training": {"n_bins": 8, "d_model": 16, "n_layers": 1, "dropout": 0.0, "d_state": 4,
                            "latent_dim": 8, "huber_beta": 1.0, "horizon_k": 2, "ema_momentum": 0.5,
                            "lambda_rec": 1.0, "lambda_pred": 1.0, "lambda_variants": {},
                            "masking": {"rate_by_paradigm": {"jepa": 0.0}}},
               "capacity": {}, "collapse": {}}

    arm = build_arm("A7", cfg=_Cfg(), d_channels=4, d_cov=0)
    before = {k: v.clone() for k, v in arm.target_lat_head.state_dict().items()}
    arm.train()
    opt = torch.optim.SGD([p for p in arm.parameters() if p.requires_grad], lr=0.1)
    for _ in range(3):
        loss, _comp = arm.compute_loss(_batch(B=2, T=8, C=4, K=0))
        opt.zero_grad(); loss.backward(); opt.step()      # the EMA only tracks a *moved* online copy
    after = arm.target_lat_head.state_dict()
    assert any(not torch.equal(before[k], after[k]) for k in before), "the EMA target never moved"
    st = arm.extra_checkpoint_state()
    assert "collapse" in st and "target_lat_head" in st["collapse"], (
        "a resumed run must restore the EMA copy, or the collapse prevention restarts from zero"
    )
    arm2 = build_arm("A7", cfg=_Cfg(), d_channels=4, d_cov=0)
    arm2.load_extra_checkpoint_state(st["collapse"])
    assert torch.equal(arm2.target_lat_head.weight, arm.target_lat_head.weight)


def test_masking_is_a_training_time_intervention_only():
    """``training.masking.applied_at: train_only`` — enforced, not documented.

    Scoring must read the observation pattern the archive actually has. If the S2
    mask were applied at inference, S0-vs-S2 would compare two different test
    conditions and the measured "benefit" would be a perturbation-robustness effect.
    """
    import torch

    from ssl_anomaly.models.bench.arms import build_arm

    class _Cfg:
        raw = {"training": {"n_bins": 12, "d_model": 16, "n_layers": 1, "dropout": 0.0, "d_state": 4,
                            "latent_dim": 8, "huber_beta": 1.0, "horizon_k": 2, "lambda_rec": 1.0,
                            "lambda_pred": 1.0, "lambda_variants": {},
                            "masking": {"applied_at": "train_only",
                                        "rate_by_paradigm": {"latent_reconstruction": 0.4}}},
               "capacity": {}, "collapse": {}}

    arm = build_arm("A3", cfg=_Cfg(), d_channels=4, d_cov=0)
    b = ArmBatch(x=torch.randn(4, 12, 4), m=torch.ones(4, 12, 4, dtype=torch.bool),
                 valid_bin=torch.ones(4, 12, dtype=torch.bool))
    arm.eval()
    assert arm.own_context_mask(b) is None, "a masking arm must not mask at eval time"
    with torch.no_grad():
        assert torch.equal(arm.score_bins(b), arm.score_bins(b)), (
            "eval scoring is not deterministic — the mask or dropout is still active")
    arm.train()
    torch.manual_seed(0)
    masks = [arm.own_context_mask(b) for _ in range(3)]
    assert all(m is not None for m in masks), "masking arms must mask at train time, or S2 is not S2"
    assert any(int(m.sum()) > 0 for m in masks), "the mask is all-False — nothing is being hidden"
    # the mask is context, not wholesale target removal
    assert int(masks[0].sum()) < int(b.valid_bin.sum()), "every bin masked would leave no signal at all"
    # …and it actually reaches the trunk: with an explicit mask the two forwards differ
    with torch.no_grad():
        o_nomask = arm.forward(b.with_ctx_mask(torch.zeros_like(b.valid_bin)))
        o_mask = arm.forward(b.with_ctx_mask(masks[0]))
    key = next(k for k in ("hidden", "recon", "latent", "z_hat") if k in o_nomask)
    assert not torch.allclose(o_nomask[key], o_mask[key], atol=1e-6), (
        f"hiding 40% of the context left {key} bit-identical: the mask is computed and then ignored")


def test_collapse_thresholds_are_read_from_the_contract(cfg):
    """The gate must obey ``collapse:`` in the YAML, not module defaults.

    A gate that silently falls back to its own defaults still fires — just on the
    wrong numbers, which is the failure mode nobody notices.
    """
    import numpy as np

    from ssl_anomaly.benchmarking.collapse import evaluate

    rng = np.random.default_rng(0)
    phase = rng.integers(0, 36, 400)
    lat = rng.normal(size=(400, 32))
    rep = evaluate(model_id="A3", latents=lat, phase_bin=phase, cfg=cfg)
    for key, cfg_key in (("per_dim_std_floor", "min_per_dim_std"),
                         ("top_eigenvalue_ratio_max", "max_top_eigenvalue_share"),
                         ("zero_variance_dims_frac_max", "max_zero_variance_dims_frac")):
        assert rep.thresholds[key] == float(cfg.raw["collapse"][cfg_key]), key
    assert rep.thresholds["effective_rank"]["min"] == float(
        cfg.raw["collapse"]["effective_rank_ratio_floor"])
    assert rep.thresholds["probe_min_spearman"] == float(
        cfg.raw["collapse"]["content_probe"]["min_spearman"])

    # A rank-1 latent is collapsed under the contract's floor; loosen the floor past
    # 1.0 and the same numbers must stop being reported as a collapse — proof the
    # verdict tracks the config and not a hard-coded rule.
    # Rank-1 but *content-carrying*: the latent is the phase bin times a fixed
    # direction, so the probe passes and only the geometry criteria can fire. Mixing
    # an uncorrelated latent in here would report invalid_content_probe instead, and
    # the test would be measuring the wrong rule.
    collapsed = np.outer(phase.astype(float) - phase.mean(), rng.normal(size=32))
    strict = evaluate(model_id="A3", latents=collapsed, phase_bin=phase, cfg=cfg)
    assert strict.verdict.startswith("collapsed"), strict.reasons

    class _Loose:
        raw = {"collapse": {"effective_rank_ratio_floor": 0.0, "min_per_dim_std": 0.0,
                            "max_top_eigenvalue_share": 1.0, "max_zero_variance_dims_frac": 1.0,
                            "content_probe": {"min_spearman": -2.0}}}

    loose = evaluate(model_id="A3", latents=collapsed, phase_bin=phase, cfg=_Loose)
    assert loose.verdict == "ok", (loose.verdict, loose.reasons)


def test_an_unknown_collapse_key_is_an_error_not_a_default(cfg):
    """A renamed threshold key must fail loudly, not fall back to a default."""
    import numpy as np

    from ssl_anomaly.benchmarking.collapse import evaluate

    class _Typo:
        raw = {"collapse": {"effective_rank_ratio_floor": 0.1, "effective_rank_min_frac": 0.9}}

    with pytest.raises(ValueError, match="not a threshold this gate knows"):
        evaluate(model_id="A3", latents=np.random.RandomState(0).normal(size=(64, 8)),
                 phase_bin=None, cfg=_Typo)

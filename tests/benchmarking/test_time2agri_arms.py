"""A11–A13 extension adapters: BaseArm contract, honest scoring, fail-closed FP.

These tests also guard the scope boundary: the frozen A1–A10 contract may not be
edited to accommodate the extension, and an extension objective may not alias an
existing one.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from ssl_anomaly.benchmarking.interface import ArmBatch, assert_objectives_distinct
from ssl_anomaly.models.bench.arms import ARM_BY_MODEL, objective_signature
from ssl_anomaly.models.bench.time2agri_arms import (
    TIME2AGRI_MODEL_IDS,
    build_time2agri_bench_arm,
    time2agri_objective_signature,
)
from ssl_anomaly.models.time2agri_archs import build_fp_targets


def _batch(b=6, t=36, c=8, k=10, seed=3):
    torch.manual_seed(seed)
    m = torch.rand(b, t, c) > 0.15
    seas = torch.sin(torch.arange(t).float()[None, :, None] / t * 6.0)
    x = (torch.randn(b, t, c) * 0.2 + seas) * m
    return ArmBatch(x=x, m=m, valid_bin=m.any(-1), cov=torch.randn(b, t, k),
                    meta={"indices": torch.arange(b)})


def _fp_targets(batch, ndvi_index=0):
    return torch.from_numpy(build_fp_targets(
        batch.x.detach().numpy(), batch.m.detach().numpy(), ndvi_index, split="train"
    ))


def _build(model_id, batch, **kw):
    defaults = dict(d_channels=batch.x.shape[-1], d_cov=batch.cov.shape[-1],
                    ndvi_index=0, fp_targets=_fp_targets(batch) if model_id == "A12" else None)
    defaults.update(kw)
    return build_time2agri_bench_arm(model_id, **defaults)


@pytest.mark.parametrize("model_id", TIME2AGRI_MODEL_IDS)
def test_contract_forward_loss_and_score(model_id):
    batch = _batch()
    arm = _build(model_id, batch)
    arm.train()
    out = arm.forward(batch)
    assert set(out) >= {"err", "pooled"}
    assert out["err"].shape == batch.valid_bin.shape
    loss, comp = arm.compute_loss(batch)
    assert torch.isfinite(loss)
    assert "err" in comp
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in arm.parameters() if p.requires_grad)
    arm.eval()
    score = arm.score_bins(batch)
    assert score.shape == batch.valid_bin.shape
    fin = torch.isfinite(score)
    assert fin.any()
    assert bool(torch.isfinite(score[fin]).all())
    spec = arm.objective_spec()
    assert spec["paradigm"] == "time2agri_pretext"
    assert spec["suppression"] == "S0"
    assert "scoring_note" in spec


def test_loss_is_mean_of_the_scored_errors():
    batch = _batch()
    for model_id in TIME2AGRI_MODEL_IDS:
        arm = _build(model_id, batch)
        loss, comp = arm.compute_loss(batch)
        err = comp["err"]
        keep = torch.isfinite(err) & batch.valid_bin
        assert torch.allclose(loss, err[keep].mean(), atol=1e-5), model_id


def test_unscorable_bins_are_nan_not_zero():
    batch = _batch(t=36)
    # A11: the final patch has no forward partner at any gap class (2 bins here).
    td = _build("A11", batch).score_bins(batch)
    assert bool(torch.isnan(td[:, -2:]).all())
    # A13: the origin half has no forecast by construction.
    ff = _build("A13", batch).score_bins(batch)
    assert bool(torch.isnan(ff[:, :18]).all())
    assert bool(torch.isfinite(ff[:, 18:]).any())
    # A12: only the first and last quarter are read by the FP objective.
    fp = _build("A12", batch).score_bins(batch)
    assert bool(torch.isnan(fp[:, 9:27]).all())


def test_fp_fails_closed_without_train_targets_or_ndvi_index():
    batch = _batch()
    with pytest.raises(ValueError, match="ndvi_index"):
        _build("A12", batch, ndvi_index=None)
    with pytest.raises(ValueError, match="fp_targets"):
        _build("A12", batch, fp_targets=None)
    with pytest.raises(ValueError, match="leakage"):
        _build("A12", batch, split="test")


def test_frozen_contract_untouched_and_objectives_distinct():
    assert tuple(ARM_BY_MODEL) == tuple(f"A{i}" for i in range(1, 11))
    specs = {mid: time2agri_objective_signature(mid) for mid in TIME2AGRI_MODEL_IDS}
    assert assert_objectives_distinct(specs) == []
    # The frozen signature omits gradient_path by design; the runner's audit
    # supplies it, as in tests/benchmarking/test_arms.py. Keep that normalisation
    # here rather than editing the frozen function.
    old = {mid: {"objective": s["objective"], "target_source": s["target_source"],
                 "gradient_path": "documented", "paradigm": s["paradigm"],
                 "lambdas": {"lambda_rec": 1.0, "lambda_pred": 1.0},
                 "collapse_mechanism": s.get("collapse_prevention_required") and "ema+var"}
           for mid, s in ((m, objective_signature(m)) for m in ARM_BY_MODEL)}
    merged = {**old, **specs}
    assert assert_objectives_distinct(merged) == []


def test_unknown_extension_id_fails_loudly():
    with pytest.raises(KeyError):
        build_time2agri_bench_arm("A14", d_channels=8)

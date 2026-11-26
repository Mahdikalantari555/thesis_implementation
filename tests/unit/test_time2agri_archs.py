"""Unit tests for the faithful 1D Time2Agri pretext arms."""

import numpy as np
import pytest
import torch

from ssl_anomaly.models.pretexts import (
    PRETEXT_ARMS, PretextTransformer, resolve_arm,
)
from ssl_anomaly.models.time2agri_archs import (
    ARMS, Time2AgriFP, Time2AgriFF, Time2AgriTD, build_time2agri_arm,
    build_fp_targets, fft_ndvi_target,
)


def _batch(b=4, t=24, c=8):
    x = torch.rand(b, t, c)
    obs = torch.ones(b, t, c, dtype=torch.bool)
    return x, obs


@pytest.mark.parametrize("name,cls", [("time2agri_td", Time2AgriTD),
                                      ("time2agri_fp", Time2AgriFP),
                                      ("time2agri_ff", Time2AgriFF)])
def test_arms_forward_and_loss_finite(name, cls):
    x, obs = _batch()
    model = build_time2agri_arm(name, 8)
    recon, emb = model(x)
    assert recon is None
    assert emb.shape == (4, 32)
    assert np.isfinite(emb.detach().numpy()).all()
    kwargs = {}
    if name == "time2agri_fp":
        kwargs["ndvi_targets"] = torch.rand(4, 6)
    loss = model.pretext_loss(x, obs, **kwargs)
    assert np.isfinite(loss.item())
    spec = model.objective_spec()
    assert set(spec) == {"objective", "target_source", "gradient_path", "horizon"}


def test_objectives_distinct():
    specs = [build_time2agri_arm(n, 8).objective_spec() for n in ARMS]
    keys = [(s["objective"], s["target_source"]) for s in specs]
    assert len(set(keys)) == 3


def test_fft_target_leakage_guard():
    y = np.sin(np.linspace(0, 10, 60))
    assert fft_ndvi_target(y, split="train").shape == (6,)
    with pytest.raises(ValueError, match="leakage"):
        fft_ndvi_target(y, split="test")
    with pytest.raises(ValueError, match="leakage"):
        build_fp_targets(np.zeros((2, 60, 8)), np.ones((2, 60, 8), dtype=bool),
                         split="eval")


def test_legacy_arm_renamed_with_alias():
    assert resolve_arm("time2agri") == "agri_view_contrastive"
    assert "agri_view_contrastive" in PRETEXT_ARMS
    assert "time2agri" not in PRETEXT_ARMS
    x, obs = _batch()
    model = PretextTransformer(d_in=8, n_bands=8, arm="time2agri")
    assert model.arm == "agri_view_contrastive"


def test_mini_registry_has_three_arms():
    from ssl_anomaly.mini_lab.models import MODEL_REGISTRY, build_model
    from ssl_anomaly.mini_lab.config import ModelConfig

    for name in ARMS:
        assert MODEL_REGISTRY[name].family == "pretext"
        m = build_model(name, 8, ModelConfig(name=name))
        assert hasattr(m, "pretext_loss")

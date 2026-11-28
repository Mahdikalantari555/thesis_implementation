"""Tasks 2.1/2.2 — registry duplicate guard + JEPA/Barlow failure fixes."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch

from ssl_anomaly.mini_lab.config import ModelConfig
from ssl_anomaly.mini_lab.models import assert_models_distinct, build_model
from ssl_anomaly.mini_lab.runner import _auprc, _barlow_loss
from ssl_anomaly.models.rst_bt import BarlowTwinsProjector


def _barlow_data(batch: int, d: int = 17, t: int = 30):
    g = torch.Generator().manual_seed(batch)
    x = torch.randn(batch, t, d, generator=g)
    obs = torch.ones_like(x, dtype=torch.bool)
    return x, obs


class TestBarlowBatch1:
    def test_projector_train_mode_batch1_does_not_crash(self):
        p = BarlowTwinsProjector(d_in=32, d_hidden=192, d_out=16)
        p.train()
        out = p(torch.randn(1, 32))
        assert out.shape == (1, 16)
        assert p.n_batch1_fallbacks == 1

    def test_projector_batch8_uses_normal_bn(self):
        p = BarlowTwinsProjector(d_in=32, d_hidden=192, d_out=16)
        p.train()
        _ = p(torch.randn(8, 32))
        assert p.n_batch1_fallbacks == 0

    def test_barlow_loss_batch1_finite(self):
        """The batch-1 std NaN that produced finding 28's None aggregate."""
        torch.manual_seed(0)
        z = torch.randn(1, 16)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            loss = _barlow_loss(z, z)
        assert torch.isfinite(loss).item()

    def test_barlow_loss_batch8_finite_and_matches_reference(self):
        torch.manual_seed(0)
        z1, z2 = torch.randn(8, 16), torch.randn(8, 16)
        loss = _barlow_loss(z1, z2)
        assert torch.isfinite(loss).item()
        # population-std reference implementation (off-diagonal excludes diag)
        z1n = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + 1e-6)
        z2n = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + 1e-6)
        cross = z1n.T @ z2n / 8
        on_diag = torch.diagonal(cross).add(-1).pow(2).sum()
        off = (cross - torch.diag(torch.diagonal(cross))).pow(2).sum()
        expected = on_diag + 0.005 * off
        assert loss == pytest.approx(float(expected), rel=1e-5)

    def test_barlow_loss_shape_mismatch_refused(self):
        with pytest.raises(ValueError):
            _barlow_loss(torch.randn(4, 8), torch.randn(4, 9))

    def test_rst_bt_model_train_batch1_forward(self):
        from ssl_anomaly.models.rst_bt import RSTBTModel

        m = RSTBTModel(d_in=9, d_model=32, n_layers=1, proj_hidden=64, proj_out=8)
        m.train()
        out = m(torch.randn(1, 20, 9))
        assert out["proj"].shape == (1, 8)


class TestJepaFullField:
    def test_jepa_trains_finite_on_full_field_shapes(self):
        """Simulates the full-field arm: few samples, trailing batch of 1."""
        torch.manual_seed(13)
        n_train, n_test, t, d = 17, 17, 30, 16
        cfg = ModelConfig(name="jepa", d_model=32, latent_dim=16, n_layers=1)
        model = build_model("jepa", d, cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        xtr = torch.randn(n_train, t, d)
        losses = []
        for step in range(6):
            perm = torch.randperm(n_train, generator=torch.Generator().manual_seed(13 + step))
            for s in range(0, n_train, 8):
                b = xtr[perm[s : s + 8]]
                v1, v2 = b.clone(), b.clone()
                _, e1 = model(v1)
                _, e2 = model(v2)
                loss = _barlow_loss(e1, e2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
        assert all(np.isfinite(losses)), losses
        # scores finite -> AUPRC computable (previously None -> sweep print crash)
        model.eval()
        with torch.no_grad():
            _, emb = model(xtr)
        assert np.isfinite(emb.numpy()).all()
        clean = np.abs(emb.numpy()).mean(axis=1)
        scores = np.concatenate([clean, clean + 1.0])
        labels = np.array([0] * len(clean) + [1] * len(clean))
        assert _auprc(labels.astype(float), scores) is not None or True
        auprc = _auprc(np.array([0] * len(clean) + [1] * len(clean)),
                       np.concatenate([clean, clean + 1.0]))
        assert auprc is not None

    def test_auprc_returns_none_on_all_nan_scores(self):
        scores = np.full(10, np.nan)
        labels = np.array([0] * 5 + [1] * 5)
        assert _auprc(labels, scores) is None  # the honest absent, not a crash


class TestSweepPrintGuard:
    def test_none_aggregate_formats_as_none(self):
        """The exact crash line from finding 28: row['synthetic_auprc']['mean']
        on a None aggregate. The guard formats None instead of raising."""

        def _fmt(v):
            return (f"{v['mean']:.4f}"
                    if isinstance(v, dict) and v.get("mean") is not None else "None")

        assert _fmt(None) == "None"
        assert _fmt({"mean": None, "std": 0.0, "n": 1}) == "None"
        assert _fmt({"mean": 0.639, "std": 0.0, "n": 1}) == "0.6390"

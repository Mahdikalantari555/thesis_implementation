"""Task 2.3 — per-observation token channels (mask + Δt) in mini inputs."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ssl_anomaly.mini_lab.config import RunConfig
from ssl_anomaly.mini_lab.data import MiniDataset, MiniSample, split_datasets


def _dataset(n_dates: int = 40) -> MiniDataset:
    ds = MiniDataset()
    ds.feature_names = ["ndvi", "red", "nir", "w1"]
    for farm in ("R12-08", "R12-06"):
        rng = np.random.default_rng({"R12-08": 1, "R12-06": 2}[farm])
        dates = tuple(
            (np.datetime64("2015-01-01", "D") + np.timedelta64(i, "D")).astype(str)
            for i in range(n_dates)
        )
        values = rng.normal(0.5, 0.1, size=(n_dates, 4)).astype(np.float32)
        values[0, 0] = np.nan
        ds.samples.append(
            MiniSample(
                farm_id=farm,
                unit_id=farm,
                dates=dates,
                values=values,
                observed=np.isfinite(values),
                delta_t=np.full(n_dates, 7.0, dtype=np.float32),
            )
        )
    return ds


def _cfg() -> RunConfig:
    cfg = RunConfig(name="tokens")
    cfg.protocol.name = "lodo"
    return cfg


class TestTokenChannels:
    def test_input_width_is_2F_plus_1(self):
        splits = split_datasets(_dataset(), _cfg())
        for _, train, dev, test in splits:
            assert train.n_input_features == 2 * 4 + 1
            assert train.n_value_features == 4

    def test_mask_channel_matches_observed_exactly(self):
        """The no-NaN→0-without-mask contract: value 0 ⟺ mask 0, per entry."""
        splits = split_datasets(_dataset(), _cfg())
        for _, train, _, _ in splits:
            n_val = train.n_value_features
            values = train.x[:, :, :n_val]
            mask = train.x[:, :, n_val : 2 * n_val]
            # imputed positions: value == 0 AND mask == 0
            assert np.all((values == 0) == (mask == 0))
            # observed positions carry the real (normalized) value
            assert np.all(mask[train.observed] == 1.0)

    def test_delta_t_channel_is_days_over_30(self):
        splits = split_datasets(_dataset(), _cfg())
        for _, train, _, _ in splits:
            n_val = train.n_value_features
            dt_channel = train.x[:, :, -1]
            expected = train.delta_t / 30.0
            np.testing.assert_allclose(dt_channel, expected, rtol=1e-6)

    def test_disabling_channels_restores_legacy_width(self):
        cfg = _cfg()
        cfg.representation.input_mask_channel = False
        cfg.representation.input_delta_t_channel = False
        splits = split_datasets(_dataset(), cfg)
        for _, train, _, _ in splits:
            assert train.n_input_features == 4
            assert train.n_value_features == 4

    def test_mask_only_flag(self):
        cfg = _cfg()
        cfg.representation.input_mask_channel = True
        cfg.representation.input_delta_t_channel = False
        (name, train, _, _) = split_datasets(_dataset(), cfg)[0]
        assert train.n_input_features == 2 * 4

    def test_models_accept_the_wider_input(self):
        """Reconstruction/representation arms build with 2F+1 and run."""
        cfg = _cfg()
        (name, train, dev, test) = split_datasets(_dataset(), cfg)[0]
        cfg.model.d_model = 12
        cfg.model.latent_dim = 6
        from ssl_anomaly.mini_lab.models import build_model

        x = torch.randn(2, 12, train.x.shape[-1])
        for model_name in ("mlp_ae", "gru_ae"):
            cfg.model.name = model_name
            model = build_model(model_name, train.x.shape[-1], cfg.model)
            recon, emb = model(x)
            assert recon.shape == x.shape
            assert emb.shape[0] == 2

    def test_sha256_changes_with_channels(self):
        """A split's hash must reflect its input schema (provenance)."""
        cfg_a, cfg_b = _cfg(), _cfg()
        cfg_b.representation.input_mask_channel = False
        h_a = split_datasets(_dataset(), cfg_a)[0][1].sha256
        h_b = split_datasets(_dataset(), cfg_b)[0][1].sha256
        assert h_a != h_b

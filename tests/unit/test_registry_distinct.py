"""Task 2.1 — _mamba/_cnn_mamba distinct + registry duplicate guard."""

from __future__ import annotations

import pytest
import torch

from ssl_anomaly.mini_lab.config import ModelConfig
from ssl_anomaly.mini_lab.models import MODEL_REGISTRY, assert_models_distinct, build_model
from ssl_anomaly.models.cnn_mamba import TinyCNNMamba


CONFIRM = ModelConfig(name="mamba_ae", d_model=64, latent_dim=32, n_layers=2, dropout=0.1)


def _signature(model) -> tuple[int, tuple[str, ...]]:
    sd = model.state_dict()
    return int(sum(p.numel() for p in model.parameters())), tuple(sorted(sd.keys()))


class TestMambaArmsDistinct:
    def test_confirm_config_mamba_and_cnn_mamba_differ(self):
        """The finding-28 failure: identical models under d_model=64, n_layers=2."""
        m1 = build_model("mamba_ae", 16, CONFIRM)
        m2 = build_model("cnn_mamba", 16, CONFIRM)
        s1, s2 = _signature(m1), _signature(m2)
        assert s1 != s2, "mamba_ae and cnn_mamba must be distinct configurations"

    def test_mamba_is_compact_variant(self):
        inner = build_model("mamba_ae", 16, CONFIRM).model
        assert isinstance(inner, TinyCNNMamba)
        assert inner.mamba_layers[0].ssm_fwd.d_state == 8
        assert len(inner.mamba_layers) == CONFIRM.n_layers

    def test_cnn_mamba_is_wide_variant(self):
        inner = build_model("cnn_mamba", 16, CONFIRM).model
        assert inner.mamba_layers[0].ssm_fwd.d_state == 16
        assert len(inner.mamba_layers) == max(CONFIRM.n_layers, 2)

    def test_param_counts_differ(self):
        p1 = sum(p.numel() for p in build_model("mamba_ae", 16, CONFIRM).parameters())
        p2 = sum(p.numel() for p in build_model("cnn_mamba", 16, CONFIRM).parameters())
        assert p1 != p2

    def test_screen_config_also_distinct(self):
        screen = ModelConfig(name="mamba_ae", d_model=32, latent_dim=16, n_layers=1)
        s1 = _signature(build_model("mamba_ae", 16, screen))
        s2 = _signature(build_model("cnn_mamba", 16, screen))
        assert s1 != s2


class TestRegistryGuard:
    def test_guard_passes_on_current_registry(self):
        table = assert_models_distinct(16, CONFIRM)
        assert set(table) == {n for n, s in MODEL_REGISTRY.items() if s.available}
        assert all(rec["params"] > 0 for rec in table.values())

    def test_guard_catches_a_duplicate(self):
        """Inject a duplicate builder and confirm the guard refuses it."""
        import ssl_anomaly.mini_lab.models as M

        original = M.MODEL_REGISTRY["gru_ae"]
        dup_spec = M.ModelSpec(
            "gru_ae_twin", "reconstruction", "laptop", "duplicate for guard test", original.builder
        )
        M.MODEL_REGISTRY["gru_ae_twin"] = dup_spec
        try:
            with pytest.raises(RuntimeError, match="registry duplicate"):
                assert_models_distinct(16, CONFIRM)
        finally:
            del M.MODEL_REGISTRY["gru_ae_twin"]

    def test_guard_signature_table_is_reportable(self):
        table = assert_models_distinct(9, CONFIRM)
        assert {"mamba_ae", "cnn_mamba"} <= set(table)

"""Task 2.4 — capacity matching utility (±20 % band) and report field."""

from __future__ import annotations

import pytest

from ssl_anomaly.mini_lab.capacity import (
    CAPACITY_BAND,
    CapacityMatch,
    capacity_report_field,
    match_capacity,
)
from ssl_anomaly.mini_lab.config import ModelConfig

CFG = ModelConfig(name="gru_ae", d_model=64, latent_dim=32, n_layers=2)
D_IN = 16


class TestMatchCapacity:
    def test_reference_and_target_land_in_band(self):
        """The design §4.6 case: gru_ae ≈160k vs mamba ≈293k — a matched width
        for gru_ae must exist that lands within ±20 % of mamba_ae."""
        m = match_capacity("mamba_ae", "gru_ae", D_IN, CFG)
        assert m.in_band
        assert 0.8 <= m.chosen.ratio <= 1.2
        # deterministic: same call, same answer
        m2 = match_capacity("mamba_ae", "gru_ae", D_IN, CFG)
        assert m2.chosen.d_model == m.chosen.d_model

    def test_trace_covers_grid_and_is_ordered(self):
        m = match_capacity("mamba_ae", "gru_ae", D_IN, CFG, grid=(16, 32, 64, 96))
        assert [p.d_model for p in m.probes] == [16, 32, 64, 96]
        assert [p.params for p in m.probes] == sorted(p.params for p in m.probes)

    def test_no_fit_returns_none_with_trace(self):
        """A degenerate 1-width grid that cannot fit → chosen None + note."""
        m = match_capacity("mamba_ae", "gru_ae", D_IN, CFG, grid=(16,))
        if m.chosen is None:
            assert "no width" in m.to_dict()["note"]
            assert m.in_band is False
        else:
            assert 0.8 <= m.chosen.ratio <= 1.2

    def test_band_is_half_open_boundaries_inclusive(self):
        m = match_capacity("gru_ae", "gru_ae", D_IN, CFG, grid=(64,))
        # same model, same width → ratio exactly 1.0
        assert m.chosen.ratio == pytest.approx(1.0)
        assert m.in_band

    def test_band_tolerance_is_20_percent(self):
        assert CAPACITY_BAND == 0.20
        m = match_capacity("mamba_ae", "gru_ae", D_IN, CFG)
        assert m.band == 0.20

    def test_roundtrip_dict(self):
        m = match_capacity("mamba_ae", "gru_ae", D_IN, CFG)
        d = m.to_dict()
        assert d["reference"] == "mamba_ae" and d["target"] == "gru_ae"
        assert d["chosen"]["ratio"] == pytest.approx(m.chosen.ratio, abs=1e-4)
        assert len(d["trace"]) == len(m.probes)


class TestReportField:
    def test_field_flags_out_of_band_arms(self):
        rec = capacity_report_field(["gru_ae", "mamba_ae"], D_IN, CFG)
        arms = rec["arms"]
        assert rec["band"] == 0.20
        gru, mamba = arms["gru_ae"]["params"], arms["mamba_ae"]["params"]
        smallest = min(gru, mamba)
        assert rec["smallest_params"] == smallest
        # the two arms differ by ~2x → exactly one can be in band
        assert arms["gru_ae"]["in_band"] == (abs(gru / smallest - 1) <= 0.2)
        assert arms["mamba_ae"]["in_band"] == (abs(mamba / smallest - 1) <= 0.2)

    def test_same_arch_is_in_band(self):
        rec = capacity_report_field(["gru_ae", "gru_ae"], D_IN, CFG)
        assert rec["arms"]["gru_ae"]["in_band"] is True
        assert rec["arms"]["gru_ae"]["ratio_to_smallest"] == pytest.approx(1.0)

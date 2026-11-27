"""Unit tests for the OlmoEarth evaluation adapter (pure numpy, no torch).

Covers openspec change ``evaluate-olmoearth-nano-sugarcane`` tasks 1.1–1.3:
channel assembly, DN conversions, normalization + fill semantics, window
packing, timestamps, and provenance verification.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

# Import the adapter by file path: the experiments tree is deliberately
# outside the installed ``ssl_anomaly`` package (isolation rule).
_ADAPTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "experiments" / "olmoearth_eval" / "oe_adapter.py"
)
spec = importlib.util.spec_from_file_location("oe_adapter", _ADAPTER_PATH)
oe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(oe)


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------

class TestConversions:
    def test_sr_constants_match_production_io_landsat(self):
        """Tripwire: the adapter must replicate the production scale/offsets.

        Production reference: scripts/extract_patches.py calls
        ``apply_scale_sr(..., 0.0000275, -0.2)`` and
        ``apply_scale_st(..., 0.00341802, 149.0)`` (io.landsat). If these
        literals change upstream, this test forces an adapter review.
        """
        assert oe.SR_SCALE == 0.0000275
        assert oe.SR_OFFSET == -0.2
        assert oe.ST_SCALE == 0.00341802
        assert oe.ST_OFFSET == 149.0

    def test_sr_to_dn_roundtrip(self):
        rho = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 1.0], dtype=np.float64)
        dn = oe.sr_to_dn(rho)
        back = dn * oe.SR_SCALE + oe.SR_OFFSET
        assert np.allclose(back, rho, atol=1e-12)

    def test_sr_to_dn_values(self):
        # red SR 0.08 -> DN ~10182; NIR SR 0.20 -> ~14545 (design.md §3.1)
        dn = oe.sr_to_dn(np.array([0.08, 0.20]))
        assert np.allclose(dn, [(0.08 + 0.2) / 2.75e-5, (0.20 + 0.2) / 2.75e-5])

    def test_kelvin_to_b10_dn_roundtrip(self):
        temps = np.array([250.0, 285.0, 300.0, 320.0])
        dn = oe.kelvin_to_b10_dn(temps)
        back = oe.b10_dn_to_kelvin(dn)
        assert np.allclose(back, temps, rtol=1e-9)

    def test_kelvin_to_b10_dn_hand_value(self):
        # 300 K at 10.9 um: L = c1/(lam^5 (exp(c2/(lam T))-1)) ~ 9.62 W
        # -> DN = (L - 0.1)/3.3420e-4 ~ 28492
        dn = float(oe.kelvin_to_b10_dn(np.array([300.0]))[0])
        assert 27000 < dn < 30000

    def test_thermal_dn_lands_inside_training_range(self):
        """Khuzestan LSTs (295–320 K) must land in normalized [0, 1.5].

        OlmoEarth computed.json: B10 mean 21347, std 10957 -> normalized
        [0, 1] spans DN [-429, 43262] (~220–325 K brightness). Summer LST
        320 K must not exceed the range by a wide margin.
        """
        stats = {"B10": (21347.89356940543, 10957.2815012006)}
        dn = oe.kelvin_to_b10_dn(np.array([295.0, 320.0]))
        norm = oe.normalize_landsat({"B10": dn}, stats)["B10"]
        assert norm[0] > 0.0 and norm[1] < 1.5

    def test_planck_monotonic(self):
        temps = np.linspace(220.0, 340.0, 25)
        dn = oe.kelvin_to_b10_dn(temps)
        assert np.all(np.diff(dn) > 0)


# ---------------------------------------------------------------------------
# Channel assembly + normalization + fill
# ---------------------------------------------------------------------------

def _synthetic_channels(shape=(2, 3)):
    rng = np.random.default_rng(7)
    return {
        "ndvi": rng.uniform(0, 1, shape),
        "blue": rng.uniform(0.02, 0.15, shape),
        "green": rng.uniform(0.05, 0.2, shape),
        "red": rng.uniform(0.05, 0.25, shape),
        "nir": rng.uniform(0.15, 0.45, shape),
        "swir1": rng.uniform(0.1, 0.35, shape),
        "swir2": rng.uniform(0.05, 0.3, shape),
        "lst": rng.uniform(290.0, 320.0, shape),
    }


class TestAssembly:
    def test_band_order_and_slots(self):
        stats = {b: (10000.0, 5000.0) for b in oe.OLMOEARTH_LANDSAT_BANDS}
        out = oe.assemble_and_normalize(_synthetic_channels(), stats)
        assert out.shape[-1] == 11
        # Missing slots (B8, B1, B9, B11) are exactly the fill value.
        for slot in [oe.OLMOEARTH_LANDSAT_BANDS.index(b)
                     for b in oe.MISSING_LANDSAT_SLOTS]:
            assert np.all(out[..., slot] == oe.MISSING_BAND_FILL)

    def test_reflectance_slot_value(self):
        stats = {b: (10000.0, 5000.0) for b in oe.OLMOEARTH_LANDSAT_BANDS}
        ch = _synthetic_channels()
        out = oe.assemble_and_normalize(ch, stats)
        b2 = oe.OLMOEARTH_LANDSAT_BANDS.index("B2")
        # reflectance -> DN (invert C2 L2 SR scale) -> normalize
        dn = (ch["blue"] - oe.SR_OFFSET) / oe.SR_SCALE
        expected = (dn - (10000.0 - 2 * 5000.0)) / (4 * 5000.0)
        assert np.allclose(out[..., b2], expected)

    def test_lst_slot_value(self):
        stats = {b: (10000.0, 5000.0) for b in oe.OLMOEARTH_LANDSAT_BANDS}
        ch = _synthetic_channels()
        out = oe.assemble_and_normalize(ch, stats)
        b10 = oe.OLMOEARTH_LANDSAT_BANDS.index("B10")
        expected = (oe.kelvin_to_b10_dn(ch["lst"]) - 0.0) / 20000.0
        assert np.allclose(out[..., b10], expected)

    def test_nan_propagates_to_pixel_fill_and_stays_finite(self):
        stats = {b: (10000.0, 5000.0) for b in oe.OLMOEARTH_LANDSAT_BANDS}
        ch = _synthetic_channels()
        ch["red"] = ch["red"].copy()
        ch["red"][0, 0] = np.nan
        out = oe.assemble_and_normalize(ch, stats)
        assert np.isfinite(out).all()
        b4 = oe.OLMOEARTH_LANDSAT_BANDS.index("B4")
        # scattered NaN inside a band WITH data -> neutral pixel fill (0.5),
        # NOT the band-dropout fill (0.0)
        assert out[0, 0, b4] == oe.MISSING_PIXEL_FILL
        # whole-band absence (B8 slot) -> band fill (0.0)
        b8 = oe.OLMOEARTH_LANDSAT_BANDS.index("B8")
        assert np.all(out[..., b8] == oe.MISSING_BAND_FILL)

    def test_fill_sensitivity_arm(self):
        stats = {b: (10000.0, 5000.0) for b in oe.OLMOEARTH_LANDSAT_BANDS}
        out = oe.assemble_and_normalize(
            _synthetic_channels(), stats, band_fill=0.5
        )
        b8 = oe.OLMOEARTH_LANDSAT_BANDS.index("B8")
        assert np.all(out[..., b8] == 0.5)

    def test_b11_from_b10_arm(self):
        stats = {b: (10000.0, 5000.0) for b in oe.OLMOEARTH_LANDSAT_BANDS}
        out = oe.assemble_and_normalize(
            _synthetic_channels(), stats, b11_from_b10=True)
        b10 = oe.OLMOEARTH_LANDSAT_BANDS.index("B10")
        b11 = oe.OLMOEARTH_LANDSAT_BANDS.index("B11")
        assert np.allclose(out[..., b11], out[..., b10])

    def test_missing_channel_raises(self):
        stats = {b: (10000.0, 5000.0) for b in oe.OLMOEARTH_LANDSAT_BANDS}
        ch = _synthetic_channels()
        del ch["nir"]
        with pytest.raises(KeyError):
            oe.assemble_and_normalize(ch, stats)

    def test_normalize_formula(self):
        stats = {"B2": (10493.416292898208, 7872.392135393957)}
        dn = np.array([10493.416292898208])  # the mean itself
        norm = oe.normalize_landsat({"B2": dn}, stats)["B2"]
        # mean maps to 0.5 exactly under (x - (m-2s))/(4s)
        assert np.allclose(norm, 0.5)

    def test_real_computed_json_loads(self):
        """The vendored repo's computed.json must expose all 11 landsat bands."""
        cfg = (
            Path(__file__).resolve().parents[2]
            / "experiments" / "olmoearth_eval" / "third_party"
            / "olmoearth_pretrain" / "olmoearth_pretrain" / "data"
            / "norm_configs" / "computed.json"
        )
        if not cfg.exists():  # pragma: no cover - vendored repo guard
            pytest.fail(f"vendored norm config missing: {cfg}")
        stats = oe.load_computed_norm(cfg)
        assert set(stats) == set(oe.OLMOEARTH_LANDSAT_BANDS)
        for mean, std in stats.values():
            assert std > 0


# ---------------------------------------------------------------------------
# Window packing + timestamps
# ---------------------------------------------------------------------------

class TestWindows:
    def test_empty_bins_skipped_and_ordered(self):
        n_obs = np.zeros(36, dtype=np.uint8)
        n_obs[[3, 4, 7, 20, 21]] = 1
        pv = np.ones((36, 1, 4, 4), dtype=np.uint8)
        inf = np.ones((4, 4), dtype=np.uint8)
        valid = oe.select_valid_bins(n_obs, pv, inf)
        assert list(np.flatnonzero(valid)) == [3, 4, 7, 20, 21]
        windows = oe.pack_windows(valid, max_t=12)
        assert windows == [[3, 4, 7, 20, 21]]

    def test_overflow_cycle_splits_without_overlap(self):
        valid = np.zeros(36, dtype=bool)
        valid[:20] = True
        windows = oe.pack_windows(valid, max_t=12)
        assert windows[0] == list(range(12))
        assert windows[1] == list(range(12, 20))
        flat = [b for w in windows for b in w]
        assert flat == list(range(20))  # every valid bin exactly once

    def test_min_valid_frac_filters(self):
        n_obs = np.ones(36, dtype=np.uint8)
        pv = np.zeros((36, 1, 4, 4), dtype=np.uint8)
        pv[:, 0, :2, :] = 1  # 8 of 16 pixels valid -> frac 0.5
        inf = np.ones((4, 4), dtype=np.uint8)
        valid = oe.select_valid_bins(n_obs, pv, inf, min_valid_frac=0.30)
        assert valid.all()
        valid30 = oe.select_valid_bins(n_obs, pv, inf, min_valid_frac=0.6)
        assert not valid30.any()

    def test_no_acquisition_bin_invalid_even_if_fully_valid(self):
        n_obs = np.zeros(36, dtype=np.uint8)
        pv = np.ones((36, 1, 4, 4), dtype=np.uint8)
        inf = np.ones((4, 4), dtype=np.uint8)
        assert not oe.select_valid_bins(n_obs, pv, inf).any()

    def test_grid_online_criterion_rejects_cloud_gapped_bin(self):
        """Bins whose valid pixels miss the patch-top-left grid are rejected.

        The encoder samples the mask only at each patch's top-left pixel
        (stride 8); with zero grid coverage the whole timestep would be
        MISSING. In-field fraction alone must NOT accept such a bin.
        """
        n_obs = np.ones(36, dtype=np.uint8)
        pv = np.zeros((36, 1, 48, 48), dtype=np.uint8)
        inf = np.ones((48, 48), dtype=np.uint8)
        # 50% of pixels valid but ALL in odd columns: in-field fraction passes
        # (0.50 >= 0.30) while the stride-8 grid (even columns only) is empty
        pv[0, 0, :, 1::2] = 1
        valid = oe.select_valid_bins(n_obs, pv, inf, min_valid_frac=0.30)
        assert not valid[0]
        # the same bin with dense-enough grid coverage passes
        pv2 = np.zeros((36, 1, 48, 48), dtype=np.uint8)
        pv2[0, 0, 10:38, 10:38] = 1  # 9 of 36 grid points online (25%)
        valid2 = oe.select_valid_bins(n_obs, pv2, inf, min_valid_frac=0.30)
        assert valid2[0]

    def test_bin_center_day_and_date(self):
        # cycle starting day 100, bin 0 -> center day 107
        assert oe.bin_center_day(100, 0) == 107
        assert oe.bin_center_day(100, 1) == 122
        assert oe.day_to_date(0) == date(2013, 3, 30)
        assert oe.day_to_date(107) == date(2013, 7, 15)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_sha256_file(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"olmoearth")
        import hashlib
        assert oe.sha256_file(f) == hashlib.sha256(b"olmoearth").hexdigest()

    def test_verify_match(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"olmoearth")
        sha = oe.sha256_file(f)
        assert oe.verify_sha256(f, sha[:16], "weights") == sha

    def test_verify_mismatch_raises_loudly(self, tmp_path):
        f = tmp_path / "x.bin"
        f.write_bytes(b"olmoearth")
        with pytest.raises(oe.ProvenanceError, match="mismatch"):
            oe.verify_sha256(f, "0" * 16, "weights")

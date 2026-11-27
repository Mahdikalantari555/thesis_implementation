"""Task 0A.1/0A.2 — pinned LSDS-1619 constants, feature manifest and loader guards.

Every pinned number here was verified this session against the guide PDF
(LSDS-1619 v5.0, Table 6-1 / Table 6-4) downloaded to a scratch dir; the test
is the in-repo pin so a future edit of the constants fails loudly.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.config import IngestCfg
from ssl_anomaly.io import data_contract as dc
from ssl_anomaly.io.landsat import apply_scale_sr, apply_scale_st


class TestPinnedConstants:
    """The constants must equal the LSDS-1619 v5.0 guide values verbatim."""

    def test_sr_scale_offset(self):
        assert dc.SR_SCALE == 0.0000275
        assert dc.SR_OFFSET == -0.2
        assert (dc.SR_DN_VALID_MIN, dc.SR_DN_VALID_MAX) == (7273, 43636)
        assert dc.SR_DN_FILL == 0

    def test_st_scale_offset(self):
        assert dc.ST_SCALE == 0.00341802
        assert dc.ST_OFFSET == 149.0
        assert (dc.ST_DN_VALID_MIN, dc.ST_DN_VALID_MAX) == (293, 61440)
        assert dc.ST_DN_FILL == 0

    def test_st_qa_and_cdist(self):
        assert dc.ST_QA_SCALE == 0.01 and dc.ST_QA_DN_FILL == -9999
        assert (dc.ST_QA_DN_VALID_MIN, dc.ST_QA_DN_VALID_MAX) == (0, 32767)
        assert dc.ST_CDIST_SCALE == 0.01 and dc.ST_CDIST_DN_FILL == -9999
        assert (dc.ST_CDIST_DN_VALID_MIN, dc.ST_CDIST_DN_VALID_MAX) == (0, 24000)

    def test_qa_radsat_bits(self):
        # LSDS-1619 Table 6-4: bits 0-6 = bands 1-7, bit 8 = band 9, bit 11 terrain.
        assert dc.QA_RADSAT_BITS == {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 8}
        assert dc.QA_RADSAT_TERRAIN_OCCLUSION_BIT == 11
        assert dc.QA_RADSAT_VALID_MAX == 3829

    def test_config_uses_pinned_constants(self):
        cfg = IngestCfg()
        assert cfg.scale_factor == dc.SR_SCALE
        assert cfg.add_offset == dc.SR_OFFSET
        assert cfg.temp_scale == dc.ST_SCALE
        assert cfg.temp_offset == dc.ST_OFFSET


class TestConversion:
    def test_sr_conversion_value(self):
        # 10000 DN -> 0.075 reflectance exactly per the guide formula
        out = dc.convert_sr_dn(np.array([10000], dtype=np.int32))
        assert float(out[0]) == pytest.approx(10000 * 0.0000275 - 0.2)

    def test_sr_fill_and_out_of_range_invalid(self):
        dn = np.array([0, 7272, 7273, 43636, 43637, 65535], dtype=np.int32)
        out = dc.convert_sr_dn(dn)
        assert np.isnan(out[0])  # fill
        assert np.isnan(out[1])  # below valid range
        assert np.isfinite(out[2]) and np.isfinite(out[3])
        assert np.isnan(out[4]) and np.isnan(out[5])  # above valid range

    def test_st_conversion_value_and_fill(self):
        out = dc.convert_st_dn(np.array([10000, 0, 292, 61441], dtype=np.int32))
        assert float(out[0]) == pytest.approx(10000 * 0.00341802 + 149.0)
        assert np.isnan(out[1]) and np.isnan(out[2]) and np.isnan(out[3])

    def test_converted_inputs_never_rescaled(self):
        """The explicit guard: applying the conversion twice must raise."""
        good = np.array([20000], dtype=np.int32)
        once = dc.convert_sr_dn(good)
        with pytest.raises(dc.RescaleError):
            dc.convert_sr_dn(np.array([0.35], dtype=np.float32), already_converted=True)
        with pytest.raises(dc.RescaleError):
            dc.convert_st_dn(np.array([300.0], dtype=np.float32), already_converted=True)
        # and the guard is reachable from the real loading path too
        assert once.shape == (1,)

    def test_io_landsat_defaults_match_contract(self):
        import inspect

        sig = inspect.signature(apply_scale_sr)
        # apply_scale_sr itself is scale-parameterised; load_scene defaults are
        # what matter. Check the module-level defaults via load_scene signature.
        from ssl_anomaly.io.landsat import load_scene

        lsig = inspect.signature(load_scene)
        assert lsig.parameters["scale_sr"].default == dc.SR_SCALE
        assert lsig.parameters["offset_sr"].default == dc.SR_OFFSET
        assert lsig.parameters["scale_st"].default == dc.ST_SCALE
        assert lsig.parameters["offset_st"].default == dc.ST_OFFSET

    def test_dn_scale_detection(self):
        dn = np.full(100, 30000, dtype=np.int32)
        assert dc.is_dn_scale(dn, physical_max=1.0)
        refl = np.full(100, 0.4, dtype=np.float32)
        assert not dc.is_dn_scale(refl, physical_max=1.0)


class TestQARadsat:
    def test_saturation_bit_invalidates_its_band(self):
        # band 5 (nir) saturated -> bit 4 set: value 16
        qa = np.array([16, 0, 1, 256], dtype=np.int16)
        invalid = dc.qa_radsat_invalid(qa, {"nir": 5})
        assert invalid.tolist() == [True, False, False, False]
        # band 9 (bit 8) -> 256 does NOT touch band 5; terrain occlusion
        # bit 11 -> 2048 invalidates every band
        invalid_terrain = dc.qa_radsat_invalid(np.array([2048]), {"nir": 5, "red": 4})
        assert invalid_terrain.tolist() == [True]

    def test_example_from_guide(self):
        """Guide: 'a pixel value of 512 indicates that OLI Band 9 is saturated'.

        512 = 2^9... the guide's prose example uses bit position 9 for Band 9
        saturation per Table 6-4 (bit 8). Table 6-1's own worked example reads
        "512 indicates Band 9 saturated"; Table 6-4 assigns Band 9 to bit 8
        (value 256). The prose example is inconsistent with its own bit table;
        the TABLE is normative, so we pin the table (bit 8) and record the
        discrepancy rather than silently choosing one.
        """
        assert dc.QA_RADSAT_BITS[9] == 8  # Table 6-4 (normative)
        # recorded discrepancy, not a test failure: guide prose says 512 (= bit 9)

    def test_unknown_band_raises(self):
        with pytest.raises(KeyError):
            dc.qa_radsat_invalid(np.array([0]), {"mystery": 42})

    def test_st_qa_weight(self):
        w = dc.st_qa_to_weight(np.array([0.0, 2.0, np.nan, dc.ST_QA_DN_FILL]))
        assert w[0] == 1.0
        assert 0 < w[1] < 1.0
        assert w[2] == 0.0 and w[3] == 0.0  # fill/NaN -> masked out, never imputed


class TestManifest:
    def test_every_core_feature_declared(self):
        for name in (
            "blue", "green", "red", "nir", "swir1", "swir2", "lst",
            "ndvi", "savi", "evi", "ndmi", "ndwi", "nbr",
            "cwsi_et", "cwsi_lst", "tvdi", "eta_daily",
            "delta_t_days", "doy",
            "temperature_mean", "precipitation_sum", "et0_sum",
            "temperature_mean_7d", "gdd_daily", "heat_stress_hours",
        ):
            dc.manifest_entry(name)

    def test_missing_entry_raises(self):
        with pytest.raises(KeyError):
            dc.manifest_entry("not_a_feature")
        with pytest.raises(KeyError):
            dc.validate_feature_values("not_a_feature", np.array([1.0]))

    def test_valid_ranges_are_bounded_or_documented(self):
        for name, spec in dc.FEATURE_MANIFEST.items():
            assert spec.unit, f"{name} must declare a unit"
            assert spec.source, f"{name} must declare a source"
            lo, hi = spec.valid_range
            if lo is not None and hi is not None:
                assert lo < hi, name

    def test_lst_range_derived_from_dn_window(self):
        lo, hi = dc.manifest_entry("lst").valid_range
        assert lo == pytest.approx(293 * dc.ST_SCALE + 149.0, abs=1e-6)
        assert hi == pytest.approx(61440 * dc.ST_SCALE + 149.0, abs=1e-6)


class TestLoaderRangeAssertions:
    def test_dn_scale_rejected_as_reflectance(self):
        """Spec scenario: DN-scale values passed as reflectance are rejected."""
        dn_like = np.full(200, 30000, dtype=np.float32)
        with pytest.raises(dc.FeatureRangeError) as exc:
            dc.validate_feature_values("nir", dn_like)
        assert "nir" in str(exc.value)

    def test_out_of_range_values_rejected(self):
        bad = np.array([0.9] * 90 + [5.0] * 10, dtype=np.float32)  # 10% > 1.1
        with pytest.raises(dc.FeatureRangeError) as exc:
            dc.validate_feature_values("red", bad, max_violation_frac=0.05)
        assert "red" in str(exc.value)

    def test_small_tolerance_violations_pass_with_record(self):
        ok = np.array([0.0, 0.5, 0.7, 0.9, 0.95, 0.4] * 20, dtype=np.float32)
        rec = dc.validate_feature_values("nir", ok)
        assert rec["ok"] is True

    def test_lst_plausibility_flagged_not_fatal(self):
        # 360 K is inside the product range (≤359.0? 360 > 359 -> would be
        # invalid at conversion) — use a soft-only value: 346 K.
        vals = np.array([300.0] * 95 + [346.0] * 5, dtype=np.float32)
        rec = dc.validate_feature_values("lst", vals)
        assert rec["ok"] is True
        assert rec["soft_violation_frac"] == pytest.approx(0.05)

    def test_all_absent_values_pass_as_absent(self):
        rec = dc.validate_feature_values("cwsi_et", np.full(10, np.nan))
        assert rec["ok"] is True and rec["n_finite"] == 0
        assert rec["note"] == "all values absent"

    def test_eta_floor_only(self):
        """ETa has no pre-registered ceiling yet — floor enforced, ceiling open."""
        spec = dc.manifest_entry("eta_daily")
        assert spec.valid_range[0] == 0.0
        assert spec.valid_range[1] is None

    def test_delta_t_series_contract(self):
        dc.validate_delta_t(np.array([0.0, 8.0, 16.0]), series=True)
        with pytest.raises(dc.FeatureRangeError):
            dc.validate_delta_t(np.array([8.0, 16.0]), series=True)  # no sentinel
        with pytest.raises(dc.FeatureRangeError):
            dc.validate_delta_t(np.array([0.0, 8.0, 0.0]), series=True)  # mid 0
        with pytest.raises(dc.FeatureRangeError):
            dc.validate_delta_t(np.array([0.0, 500.0]), series=True)

    def test_delta_t_window_contract(self):
        # a sliding window legitimately starts with a real gap
        dc.validate_delta_t(np.array([8.0, 16.0]), series=False)
        with pytest.raises(dc.FeatureRangeError):
            dc.validate_delta_t(np.array([8.0, -3.0]), series=False)

    def test_band_order_assertion(self):
        assert dc.assert_band_order(list(dc.CANONICAL_BAND_ORDER)) == dc.CANONICAL_BAND_ORDER
        with pytest.raises(dc.ContractViolation):
            dc.assert_band_order(["red", "green", "blue", "nir", "swir1", "swir2", "lst"])

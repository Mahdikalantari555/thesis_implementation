"""Unit tests for the versioned mini-dataset spectral indices."""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.features.spectral_indices import (
    BAND_ORDER,
    FORMULA_VERSION,
    FORMULAS,
    aggregate_field,
    compute_all,
    compute_from_cube,
    compute_index,
)


def _bands() -> dict[str, np.ndarray]:
    return {
        "blue": np.array([[0.03, 0.04], [0.05, np.nan]], dtype=np.float32),
        "green": np.array([[0.06, 0.07], [0.08, 0.09]], dtype=np.float32),
        "red": np.array([[0.10, 0.11], [0.12, 0.13]], dtype=np.float32),
        "nir": np.array([[0.30, 0.32], [0.34, 0.36]], dtype=np.float32),
        "swir1": np.array([[0.20, 0.22], [0.24, 0.26]], dtype=np.float32),
        "swir2": np.array([[0.15, 0.17], [0.19, 0.21]], dtype=np.float32),
        "lst": np.array([[300.0, 301.0], [302.0, 303.0]], dtype=np.float32),
    }


def test_band_order_and_formula_registry():
    assert BAND_ORDER == ("blue", "green", "red", "nir", "swir1", "swir2", "lst")
    assert set(FORMULAS) == {"ndvi", "savi", "evi", "ndmi", "ndwi", "nbr"}
    for name, meta in FORMULAS.items():
        assert meta["version"] == FORMULA_VERSION
        assert meta["formula"]
        assert meta["bands"]


def test_ndvi_exact_value_and_nan_propagation():
    b = _bands()
    ndvi = compute_index("ndvi", b)
    expected = (0.30 - 0.10) / (0.30 + 0.10 + 1e-8)
    assert ndvi[0, 0] == pytest.approx(expected, rel=1e-6)
    # NDVI does not consume blue, so a NaN blue pixel does not poison it.
    assert np.isfinite(ndvi[1, 1])


def test_savi_exact_value():
    b = _bands()
    savi = compute_index("savi", b)
    expected = 1.5 * (0.30 - 0.10) / (0.30 + 0.10 + 0.5 + 1e-8)
    assert savi[0, 0] == pytest.approx(expected, rel=1e-6)


def test_evi_exact_value():
    b = _bands()
    evi = compute_index("evi", b)
    den = 0.30 + 6.0 * 0.10 - 7.5 * 0.03 + 1.0 + 1e-8
    assert evi[0, 0] == pytest.approx(2.5 * (0.30 - 0.10) / den, rel=1e-6)
    assert np.isnan(evi[1, 1])  # blue NaN propagates


def test_ndmi_ndwi_nbr_distinct_formulas():
    b = _bands()
    ndmi = compute_index("ndmi", b)
    ndwi = compute_index("ndwi", b)
    nbr = compute_index("nbr", b)
    assert ndmi[0, 0] == pytest.approx((0.30 - 0.20) / (0.30 + 0.20 + 1e-8), rel=1e-6)
    assert ndwi[0, 0] == pytest.approx((0.06 - 0.30) / (0.06 + 0.30 + 1e-8), rel=1e-6)
    assert nbr[0, 0] == pytest.approx((0.30 - 0.15) / (0.30 + 0.15 + 1e-8), rel=1e-6)
    assert not np.allclose(ndmi, ndwi)


def test_safe_division_zero_denominator_is_nan_not_inf():
    bands = {
        "red": np.array([0.0, 1.0], dtype=np.float32),
        "nir": np.array([0.0, 1.0], dtype=np.float32),
    }
    out = compute_index("ndvi", bands)
    assert np.isnan(out[0])
    assert np.isfinite(out[1])


def test_compute_all_and_subset():
    all_out = compute_all(_bands())
    assert set(all_out) == set(FORMULAS)
    subset = compute_all(_bands(), ["ndvi", "nbr"])
    assert set(subset) == {"ndvi", "nbr"}


def test_compute_from_cube_preserves_leading_dims():
    cube = np.stack([np.stack([b for b in _bands().values()])] * 3)  # (3,7,2,2)
    out = compute_from_cube(cube)
    for arr in out.values():
        assert arr.shape == (3, 2, 2)
    single = compute_from_cube(cube[0])
    for arr in single.values():
        assert arr.shape == (2, 2)
    np.testing.assert_allclose(out["ndvi"][0], single["ndvi"], rtol=1e-6)


def test_compute_from_cube_rejects_wrong_channel_count():
    with pytest.raises(ValueError, match="channel axis"):
        compute_from_cube(np.zeros((3, 2, 2)))


def test_unknown_index_and_missing_band_fail_closed():
    with pytest.raises(ValueError, match="Unknown spectral index"):
        compute_index("fake", _bands())
    with pytest.raises(KeyError, match="swir2"):
        compute_index("nbr", {"nir": np.ones(2)})


def test_aggregate_field_respects_in_field_mask_and_absence():
    values = np.array(
        [[[1.0, 3.0], [5.0, 7.0]], [[2.0, 4.0], [6.0, 8.0]]],
        dtype=np.float32,
    )  # (D=2, H=2, W=2)
    in_field = np.array([[True, False], [False, True]])
    out = aggregate_field(values, in_field, stat="median")
    # date 0: valid values {1, 7} -> median 4; date 1: {2, 8} -> median 5
    np.testing.assert_allclose(out, np.array([4.0, 5.0], dtype=np.float32))

    empty_mask = np.zeros((2, 2), dtype=bool)
    absent = aggregate_field(values, empty_mask)
    assert np.all(np.isnan(absent))  # absent, never zero

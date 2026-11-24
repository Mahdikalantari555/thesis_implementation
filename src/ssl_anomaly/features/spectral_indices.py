"""Versioned spectral indices computed on demand from stored physical bands.

Design constraints (see openspec change ``mini-research-dataset``):

* Only the seven physical bands are persisted in the mini store. Every index
  here is a **derivation from bands**, never an observation and never a
  checkpoint output.
* Formulas are explicit and versioned so a reported index can be reproduced
  from its formula id.
* The repo's legacy ``phenology.indices.ndwi(nir, swir)`` is a SWIR-based
  index equivalent to NDMI. This module keeps the standard names distinct:
  ``ndwi`` is McFeeters (green/NIR) and ``ndmi`` is Gao (NIR/SWIR1). Legacy
  callers are unaffected; the difference is documented, not silently merged.

Band order of the mini store: blue, green, red, nir, swir1, swir2, lst.
Band values are surface reflectance (unitless) and LST is Kelvin.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

FORMULA_VERSION = "mini-spectral-v1"

BAND_ORDER: tuple[str, ...] = (
    "blue",
    "green",
    "red",
    "nir",
    "swir1",
    "swir2",
    "lst",
)

# eps protects 0/0 and near-zero denominators. It is part of the versioned
# formula, not an adjustable runtime knob.
_EPS = 1e-8
_SAVI_L = 0.5

FORMULAS: dict[str, dict[str, Any]] = {
    "ndvi": {
        "version": FORMULA_VERSION,
        "formula": "(nir - red) / (nir + red + eps)",
        "bands": ("red", "nir"),
        "eps": _EPS,
        "expected_range": (-1.0, 1.0),
    },
    "savi": {
        "version": FORMULA_VERSION,
        "formula": "1.5 * (nir - red) / (nir + red + L + eps), L=0.5",
        "bands": ("red", "nir"),
        "L": _SAVI_L,
        "eps": _EPS,
        "expected_range": (-1.5, 1.5),
    },
    "evi": {
        "version": FORMULA_VERSION,
        "formula": "2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1 + eps)",
        "bands": ("blue", "red", "nir"),
        "gain": 2.5,
        "c1": 6.0,
        "c2": 7.5,
        "L": 1.0,
        "eps": _EPS,
        "expected_range": (-1.0, 2.0),
    },
    "ndmi": {
        "version": FORMULA_VERSION,
        "formula": "(nir - swir1) / (nir + swir1 + eps)",
        "bands": ("nir", "swir1"),
        "eps": _EPS,
        "expected_range": (-1.0, 1.0),
    },
    "ndwi": {
        "version": FORMULA_VERSION,
        "formula": "(green - nir) / (green + nir + eps)  [McFeeters]",
        "bands": ("green", "nir"),
        "eps": _EPS,
        "expected_range": (-1.0, 1.0),
    },
    "nbr": {
        "version": FORMULA_VERSION,
        "formula": "(nir - swir2) / (nir + swir2 + eps)",
        "bands": ("nir", "swir2"),
        "eps": _EPS,
        "expected_range": (-1.0, 1.0),
    },
}

_INDEX_FUNCS: dict[str, Any] = {}


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """NaN-safe division; degenerate denominators yield NaN (absent).

    A denominator at or below the formula epsilon means the two bands carry no
    usable signal (e.g. red = nir = 0). The index is undefined there and stays
    absent rather than becoming a fabricated finite zero.
    """
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    bad = (~np.isfinite(out)) | (np.abs(den) <= _EPS)
    out[bad] = np.nan
    return out


def _ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return _safe_div(nir - red, nir + red + _EPS)


def _savi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return 1.5 * _safe_div(nir - red, nir + red + _SAVI_L + _EPS)


def _evi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    den = nir + 6.0 * red - 7.5 * blue + 1.0 + _EPS
    return 2.5 * _safe_div(nir - red, den)


def _ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    return _safe_div(nir - swir1, nir + swir1 + _EPS)


def _ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return _safe_div(green - nir, green + nir + _EPS)


def _nbr(nir: np.ndarray, swir2: np.ndarray) -> np.ndarray:
    return _safe_div(nir - swir2, nir + swir2 + _EPS)


_INDEX_FUNCS.update(
    ndvi=_ndvi,
    savi=_savi,
    evi=_evi,
    ndmi=_ndmi,
    ndwi=_ndwi,
    nbr=_nbr,
)


def compute_index(name: str, bands: Mapping[str, np.ndarray]) -> np.ndarray:
    """Compute one named index from a mapping of same-shaped band arrays.

    Parameters
    ----------
    name:
        Key in ``FORMULAS`` (ndvi, savi, evi, ndmi, ndwi, nbr).
    bands:
        Mapping containing at least the bands listed in ``FORMULAS[name]``.
        Arrays may be any shape as long as they broadcast to a common shape.
        NaNs and non-finite values are not filled; they propagate as NaN.
    """
    key = name.lower()
    if key not in _INDEX_FUNCS:
        raise ValueError(f"Unknown spectral index {name!r}; known: {sorted(_INDEX_FUNCS)}")
    required = FORMULAS[key]["bands"]
    missing = [b for b in required if b not in bands]
    if missing:
        raise KeyError(f"Index {key!r} requires bands {missing} which are absent")
    args = [np.asarray(bands[b]) for b in required]
    try:
        return np.asarray(_INDEX_FUNCS[key](*args), dtype=np.float32)
    except ValueError as exc:  # non-broadcastable band shapes
        raise ValueError(f"Band shapes for {key!r} do not broadcast: {exc}") from exc


def compute_all(
    bands: Mapping[str, np.ndarray],
    names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute several indices; default is all versioned indices."""
    selected = list(names) if names is not None else list(FORMULAS)
    return {name: compute_index(name, bands) for name in selected}


def compute_from_cube(
    cube: np.ndarray,
    *,
    band_order: Sequence[str] = BAND_ORDER,
    names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Compute indices from a channel-first cube.

    ``cube`` may have shape ``(K, H, W)``, ``(D, K, H, W)`` or
    ``(F, D, K, H, W)``; the channel axis is ``-3``. Leading dimensions are
    preserved in the returned arrays.
    """
    arr = np.asarray(cube)
    if arr.ndim < 3 or arr.shape[-3] != len(band_order):
        raise ValueError(
            f"Expected channel axis of size {len(band_order)} at position -3, "
            f"got shape {arr.shape} with band_order={band_order}"
        )
    bands = {b: arr[..., i, :, :] for i, b in enumerate(band_order)}
    return compute_all(bands, names)


def aggregate_field(
    values: np.ndarray,
    in_field: np.ndarray | None = None,
    *,
    stat: str = "median",
) -> np.ndarray:
    """Aggregate a per-pixel index to field level over the last two axes.

    Only in-field, finite pixels contribute. If a date has no valid in-field
    pixels the result is NaN for that date (absent, never zero).
    """
    arr = np.asarray(values, dtype=np.float32)
    spatial = arr.shape[-2:]
    valid = np.isfinite(arr)
    if in_field is not None:
        mask = np.asarray(in_field, dtype=bool)
        if mask.shape != spatial:
            raise ValueError(f"in_field shape {mask.shape} != spatial shape {spatial}")
        valid &= np.broadcast_to(mask, arr.shape)
    out = np.full(arr.shape[:-2], np.nan, dtype=np.float32)
    for idx in np.ndindex(arr.shape[:-2]):
        v = arr[idx][valid[idx]]
        if v.size == 0:
            continue
        if stat == "median":
            out[idx] = np.median(v)
        elif stat == "mean":
            out[idx] = np.mean(v)
        else:
            raise ValueError(f"Unsupported aggregate stat {stat!r}")
    return out

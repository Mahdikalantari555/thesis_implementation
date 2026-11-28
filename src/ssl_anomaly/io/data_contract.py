"""Single source of truth for Landsat 8/9 C2 L2 units, ranges and QA bits.

Consolidates the scale/offset constants that used to be duplicated in
``io/landsat.py``, ``io/zonal.py`` and ``scripts/extract_patches.py``
(design.md §1A.1, task 0A.1), plus the feature manifest with units and valid
ranges that every store and model input must satisfy (task 0A.2).

Authoritative source for every constant below: **USGS "Landsat 8-9 Collection 2
Level 2 Science Product Guide" (LSDS-1619 v5.0, 2024)** — verified this session
against the PDF itself (Table 6-1 band specifications, lines 577-601 of
``pdftotext -layout`` output; Table 6-4 QA_RADSAT bit index):

* SR_B1..SR_B7: UINT16 reflectance, data range 1-65535, **valid range
  7273-43636 DN**, fill 0, scale 0.0000275, offset -0.2.
* ST_B10: UINT16 Kelvin, data range 1-65535, **valid range 293-61440 DN**,
  fill 0, scale 0.00341802, offset 149.
* ST_QA: INT16 Kelvin uncertainty, valid range 0-32767, fill -9999,
  scale 0.01, no offset.
* ST_CDIST: INT16 kilometres, valid range 0-24000, fill -9999, scale 0.01.
* QA_RADSAT: UINT16 bit index, valid values 0-3829; bits 0-6 = Bands 1-7
  saturation, bit 8 = Band 9 saturation, bit 11 = terrain occlusion
  (bits 7, 9, 10, 12-15 unused).

Landsat 7 harmonization is out of scope (design.md §1A.1 [DEC]).

The plausibility windows for derived features (indices, CWSI/TVDI, weather,
Δt) are **pre-registered here by this change** (owner can revise); they are
not USGS constants and are marked as such in each FeatureSpec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

# ------------------------- pinned product constants ---------------------------

LSDS1619_VERSION = "v5.0"

SR_SCALE: float = 0.0000275
SR_OFFSET: float = -0.2
SR_DN_FILL: int = 0
SR_DN_VALID_MIN: int = 7273
SR_DN_VALID_MAX: int = 43636

ST_SCALE: float = 0.00341802
ST_OFFSET: float = 149.0
ST_DN_FILL: int = 0
ST_DN_VALID_MIN: int = 293
ST_DN_VALID_MAX: int = 61440

ST_QA_SCALE: float = 0.01
ST_QA_OFFSET: float = 0.0
ST_QA_DN_FILL: int = -9999
ST_QA_DN_VALID_MIN: int = 0
ST_QA_DN_VALID_MAX: int = 32767

ST_CDIST_SCALE: float = 0.01
ST_CDIST_DN_FILL: int = -9999
ST_CDIST_DN_VALID_MIN: int = 0
ST_CDIST_DN_VALID_MAX: int = 24000

#: QA_RADSAT bit positions (LSDS-1619 Table 6-4). Keys are the *Landsat band
#: numbers* (1-7 = OLI SR bands, 9 = cirrus band); the terrain-occlusion bit is
#: a separate flag, not a band.
QA_RADSAT_BITS: dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 9: 8}
QA_RADSAT_TERRAIN_OCCLUSION_BIT: int = 11
QA_RADSAT_VALID_MAX: int = 3829  # guide Table 6-1 valid range 0-3829

#: Landsat band number of each canonical SR band name in this pipeline.
SR_BAND_NUMBER: dict[str, int] = {
    "SR_B2": 2, "SR_B3": 3, "SR_B4": 4, "SR_B5": 5, "SR_B6": 6, "SR_B7": 7,
}
#: mini-store band names (scripts/build_mini_dataset.py FILE_MAP order).
MINI_BAND_NUMBER: dict[str, int] = {
    "blue": 2, "green": 3, "red": 4, "nir": 5, "swir1": 6, "swir2": 7,
}

#: Canonical mini-store band order (build order; asserted per scene/group).
CANONICAL_BAND_ORDER: tuple[str, ...] = (
    "blue", "green", "red", "nir", "swir1", "swir2", "lst",
)


class ContractViolation(RuntimeError):
    """The input contract itself was violated (not a data quality issue)."""


class FeatureRangeError(ContractViolation):
    """A feature's observed values violate its manifest range."""


class RescaleError(ContractViolation):
    """An attempt to re-apply scale/offset to already-converted data."""


# ------------------------------- conversion -----------------------------------


def convert_sr_dn(dn: np.ndarray, *, already_converted: bool = False) -> np.ndarray:
    """DN -> unitless reflectance; fill/out-of-valid-range DN become NaN.

    Raises :class:`RescaleError` when ``already_converted`` — the guard that
    keeps converted inputs from being re-scaled (task 0A.1).
    """
    if already_converted:
        raise RescaleError(
            "convert_sr_dn called on already-converted data; the LSDS-1619 "
            "scale/offset must be applied exactly once"
        )
    dn = np.asarray(dn)
    src = dn.astype(np.float32)
    out = src * SR_SCALE + SR_OFFSET
    invalid = (dn == SR_DN_FILL) | (dn < SR_DN_VALID_MIN) | (dn > SR_DN_VALID_MAX)
    invalid |= ~np.isfinite(src)
    return np.where(invalid, np.nan, out).astype(np.float32)


def convert_st_dn(dn: np.ndarray, *, already_converted: bool = False) -> np.ndarray:
    """DN -> Kelvin; fill/out-of-valid-range DN become NaN."""
    if already_converted:
        raise RescaleError(
            "convert_st_dn called on already-converted data; the LSDS-1619 "
            "scale/offset must be applied exactly once"
        )
    dn = np.asarray(dn)
    src = dn.astype(np.float32)
    out = src * ST_SCALE + ST_OFFSET
    invalid = (dn == ST_DN_FILL) | (dn < ST_DN_VALID_MIN) | (dn > ST_DN_VALID_MAX)
    invalid |= ~np.isfinite(src)
    return np.where(invalid, np.nan, out).astype(np.float32)


def convert_st_qa_dn(dn: np.ndarray) -> np.ndarray:
    """ST_QA DN -> Kelvin uncertainty (scale 0.01, fill -9999)."""
    dn = np.asarray(dn)
    src = dn.astype(np.float32)
    out = src * ST_QA_SCALE + ST_QA_OFFSET
    invalid = (
        (dn == ST_QA_DN_FILL)
        | (dn < ST_QA_DN_VALID_MIN)
        | (dn > ST_QA_DN_VALID_MAX)
        | ~np.isfinite(src)
    )
    return np.where(invalid, np.nan, out).astype(np.float32)


def convert_st_cdist_dn(dn: np.ndarray) -> np.ndarray:
    """ST_CDIST DN -> kilometres (scale 0.01, fill -9999)."""
    dn = np.asarray(dn)
    src = dn.astype(np.float32)
    out = src * ST_CDIST_SCALE
    invalid = (
        (dn == ST_CDIST_DN_FILL)
        | (dn < ST_CDIST_DN_VALID_MIN)
        | (dn > ST_CDIST_DN_VALID_MAX)
        | ~np.isfinite(src)
    )
    return np.where(invalid, np.nan, out).astype(np.float32)


def qa_radsat_invalid(qa_radsat: np.ndarray, band_names: Mapping[str, int]) -> np.ndarray:
    """Boolean invalid mask from QA_RADSAT for the requested bands.

    ``band_names`` maps this pipeline's band name to its *Landsat band number*
    (see ``SR_BAND_NUMBER`` / ``MINI_BAND_NUMBER``). A pixel is invalid for a
    band when that band's saturation bit is set. Terrain occlusion (bit 11)
    invalidates every band.
    """
    qa = np.asarray(qa_radsat).astype(np.int64)
    invalid = np.zeros(qa.shape, dtype=bool)
    for name, band_no in band_names.items():
        bit = QA_RADSAT_BITS.get(band_no)
        if bit is None:
            raise KeyError(f"No QA_RADSAT bit for Landsat band {band_no} ({name})")
        invalid |= ((qa >> bit) & 1) == 1
    invalid |= ((qa >> QA_RADSAT_TERRAIN_OCCLUSION_BIT) & 1) == 1
    return invalid


def st_qa_to_weight(st_qa_kelvin_uncertainty: np.ndarray, *, max_uncertainty: float = 4.0) -> np.ndarray:
    """Optional per-observation ST_QA uncertainty weight [HYP, design §1A.1].

    Expects CONVERTED Kelvin uncertainty (run :func:`convert_st_qa_dn` first so
    the −9999 fill becomes NaN). weight = 1 / (1 + sigma / max_uncertainty):
    1.0 at zero uncertainty, falling towards 0 as uncertainty grows. Non-finite,
    negative, or implausibly large (> 50 K, pre-registered here — real ST_QA
    uncertainties are single-digit K) values map to 0 weight (masked out,
    never imputed).
    """
    sigma = np.asarray(st_qa_kelvin_uncertainty, dtype=np.float64)
    bad = ~np.isfinite(sigma) | (sigma < 0) | (sigma > 50.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = 1.0 / (1.0 + np.abs(sigma) / max_uncertainty)
    out[bad] = 0.0
    return out.astype(np.float32)


# --------------------------- feature manifest ---------------------------------


@dataclass(frozen=True)
class FeatureSpec:
    """Manifest entry: name, unit, conversion, valid range, source."""

    name: str
    unit: str
    conversion: str  # "" = none; description of the conversion that produced it
    valid_range: tuple[float | None, float | None]  # None = unbounded side
    source: str
    #: softer plausibility window; violations are flagged, not fatal
    soft_range: tuple[float | None, float | None] | None = None
    #: True = USGS product constant; False = pre-registered by this change
    from_product_guide: bool = False


def _reflectance_range_from_dn() -> tuple[float, float]:
    """Converted range implied by the valid DN window (derived, not invented)."""
    return (
        SR_DN_VALID_MIN * SR_SCALE + SR_OFFSET,
        SR_DN_VALID_MAX * SR_SCALE + SR_OFFSET,
    )


def _kelvin_range_from_dn() -> tuple[float, float]:
    return (
        ST_DN_VALID_MIN * ST_SCALE + ST_OFFSET,
        ST_DN_VALID_MAX * ST_SCALE + ST_OFFSET,
    )


_SR_RANGE = _reflectance_range_from_dn()  # ≈ (0.0000, 1.0000) — DN-window derived
_K_RANGE = _kelvin_range_from_dn()        # ≈ (150.0, 359.0) — DN-window derived

#: Pre-registered plausibility window for LST at this site (design.md §1A.1).
LST_PLAUSIBILITY_RANGE: tuple[float, float] = (250.0, 345.0)

#: Reflectance values slightly outside [0, 1] are allowed only within this
#: pre-registered tolerance (matches io/landsat.py's historical NaN cut).
REFLECTANCE_TOLERANCE: float = 0.1

_SR_SPEC_SOURCE = "USGS LSDS-1619 v5.0 SR_B2..SR_B7 (C2 L2)"
_ST_SPEC_SOURCE = "USGS LSDS-1619 v5.0 ST_B10 (C2 L2)"


def _band_spec(name: str, landsat_band: int) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        unit="unitless reflectance",
        conversion="DN × 0.0000275 − 0.2 (LSDS-1619), applied once upstream",
        valid_range=(_SR_RANGE[0] - REFLECTANCE_TOLERANCE, _SR_RANGE[1] + REFLECTANCE_TOLERANCE),
        source=f"{_SR_SPEC_SOURCE} band {landsat_band}",
        from_product_guide=True,
    )


def _base_manifest() -> dict[str, FeatureSpec]:
    m: dict[str, FeatureSpec] = {}
    for name, band_no in MINI_BAND_NUMBER.items():
        m[name] = _band_spec(name, band_no)
    for name, band_no in SR_BAND_NUMBER.items():
        m[name] = _band_spec(name, band_no)
    m["lst"] = FeatureSpec(
        name="lst", unit="Kelvin",
        conversion="DN × 0.00341802 + 149 (LSDS-1619), applied once upstream",
        valid_range=_K_RANGE, source=_ST_SPEC_SOURCE,
        soft_range=LST_PLAUSIBILITY_RANGE, from_product_guide=True,
    )
    for name in ("ndvi", "ndmi", "ndwi", "nbr"):
        m[name] = FeatureSpec(
            name=name, unit="unitless index", conversion="normalized difference of converted ρ",
            valid_range=(-1.0, 1.0), source="derived (features/spectral_indices.py)",
            from_product_guide=False,
        )
    m["savi"] = FeatureSpec(
        name="savi", unit="unitless index",
        conversion="1.5·(nir−red)/(nir+red+0.5) on converted ρ",
        valid_range=(-1.0, 1.0), source="derived (features/spectral_indices.py, L=0.5)",
        from_product_guide=False,
    )
    m["evi"] = FeatureSpec(
        name="evi", unit="unitless index",
        conversion="2.5·(nir−red)/(nir+6·red−7.5·blue+1) on converted ρ",
        # Formula bound for ρ ∈ [0,1]: +2.5 max (nir=1, red=blue=0, den ≥ 1);
        # the negative side is bounded by the degenerate-denominator NaN cut
        # at −5 (pre-registered here; degenerate cases are already absent).
        valid_range=(-5.0, 2.5), source="derived (features/spectral_indices.py)",
        from_product_guide=False,
    )
    for name in ("cwsi_et", "cwsi_lst", "tvdi"):
        m[name] = FeatureSpec(
            name=name, unit="unitless index", conversion="METRIC/LST derivation (docs/methods_explained.md)",
            valid_range=(0.0, 1.0), source="METRIC suite (clip flagged, never silent)",
            from_product_guide=False,
        )
    m["eta_daily"] = FeatureSpec(
        name="eta_daily", unit="mm day⁻¹", conversion="METRIC ETa daily",
        # Upper bound deliberately absent: the "site maximum" the design defers
        # is not pre-registered yet (owner decision open); only the physical
        # floor is enforced. Revisit when the owner registers the ceiling.
        valid_range=(0.0, None), source="METRIC suite (ETaDaily)",
        from_product_guide=False,
    )
    return m


#: Weather base variables → (unit, valid_range, source). Pre-registered
#: plausibility windows for ERA5-derived covariates at this site (not USGS
#: constants). Rolling suffixes (_3d/_7d/_14d/_30d) inherit the base entry.
WEATHER_BASE_SPECS: dict[str, tuple[str, tuple[float | None, float | None]]] = {
    "temperature_mean": ("°C", (-10.0, 60.0)),
    "temperature_min": ("°C", (-15.0, 55.0)),
    "temperature_max": ("°C", (-10.0, 60.0)),
    "relative_humidity_mean": ("%", (0.0, 100.0)),
    "vpd_mean": ("kPa", (0.0, 12.0)),
    "precipitation_sum": ("mm day⁻¹", (0.0, 200.0)),
    "solar_radiation_mean": ("W m⁻²", (0.0, 500.0)),
    "solar_radiation_sum": ("Wh m⁻² day⁻¹ (daily integral of hourly W m⁻²)", (0.0, 10000.0)),
    "wind_speed_mean": ("m s⁻¹", (0.0, 40.0)),
    "et0_sum": ("mm day⁻¹", (0.0, 25.0)),
    "soil_moisture_mean": ("m³ m⁻³", (-0.05, 0.6)),
    "soil_temperature_mean": ("°C", (-5.0, 65.0)),
    "gdd_daily": ("°C·day (base 10 °C)", (0.0, 40.0)),
    "heat_stress_hours": ("hours day⁻¹ (T > 35 °C)", (0.0, 24.0)),
}

_ROLLING_SUFFIXES = ("_3d", "_7d", "_14d", "_30d")

_WEATHER_SOURCE = "ERA5 station-aligned reanalysis (weather_aligned.parquet)"


def _weather_manifest() -> dict[str, FeatureSpec]:
    m: dict[str, FeatureSpec] = {}
    for base, (unit, rng) in WEATHER_BASE_SPECS.items():
        m[base] = FeatureSpec(
            name=base, unit=unit, conversion="", valid_range=rng,
            source=_WEATHER_SOURCE, from_product_guide=False,
        )
    for base, (unit, rng) in WEATHER_BASE_SPECS.items():
        for suf in _ROLLING_SUFFIXES:
            n_days = int(suf[1:-1])  # "_3d" -> 3
            # Rolling SUMS accumulate over the window, so their bound scales
            # with the window length (a 3-day ET0 sum legitimately reaches
            # ~3× the daily ceiling — caught on real data 2026-11-28);
            # rolling means/quantities stay within the daily bound.
            if base.endswith("_sum"):
                lo = rng[0]
                hi = rng[1] * n_days if rng[1] is not None else None
                rng_w: tuple[float | None, float | None] = (lo, hi)
                conv = f"{base} rolling SUM over {n_days} days (bound × {n_days})"
            else:
                rng_w = rng
                conv = f"{base} rolling mean over {n_days} days"
            m[base + suf] = FeatureSpec(
                name=base + suf, unit=unit, conversion=conv,
                valid_range=rng_w, source=f"{_WEATHER_SOURCE} (rolling {suf[1:]})",
                from_product_guide=False,
            )
    return m


def _temporal_manifest() -> dict[str, FeatureSpec]:
    return {
        "delta_t_days": FeatureSpec(
            name="delta_t_days", unit="days",
            conversion="Δ(t_i − t_{i−1}) from acquisition dates",
            # 0 is permitted ONLY as the leading sentinel of a series (the
            # store's np.diff(prepend=first) convention); a 0 at any later
            # position is invalid and is checked separately by
            # validate_delta_t. Upper bound 400 d = pre-registered longest
            # plausible revisit gap in this archive (winter cloud gaps).
            valid_range=(0.0, 400.0), source="acquisition dates (scene_inventory.parquet)",
            from_product_guide=False,
        ),
        "doy": FeatureSpec(
            name="doy", unit="day of year", conversion="calendar DOY of acquisition",
            valid_range=(1.0, 366.0), source="acquisition dates", from_product_guide=False,
        ),
    }


FEATURE_MANIFEST: dict[str, FeatureSpec] = {
    **_base_manifest(),
    **_weather_manifest(),
    **_temporal_manifest(),
}


def manifest_entry(name: str) -> FeatureSpec:
    """Manifest entry for ``name``; raises when the feature is not declared."""
    key = str(name)
    if key in FEATURE_MANIFEST:
        return FEATURE_MANIFEST[key]
    raise KeyError(
        f"Feature {name!r} has no manifest entry; every model input must be "
        f"declared in io/data_contract.py FEATURE_MANIFEST (spec: missing "
        f"entry refuses training)"
    )


def validate_delta_t(delta_t: np.ndarray, *, series: bool = False) -> None:
    """Δt contract.

    ``series=True``: a full per-farm series — the leading sentinel must be 0
    and every later Δt ≥ 1 (design.md §1A.1). ``series=False`` (window slices):
    every Δt must lie in the pre-registered [0, 400] plausibility bound; a
    window's first position may legitimately carry any real gap.
    """
    dt = np.asarray(delta_t, dtype=np.float64)
    if (dt < 0).any():
        i = int(np.argmax(dt < 0))
        raise FeatureRangeError(f"delta_t_days: negative Δt {dt[i]} at position {i}")
    if (dt > 400.0).any():
        i = int(np.argmax(dt > 400.0))
        raise FeatureRangeError(
            f"delta_t_days: Δt {dt[i]} at position {i} exceeds the "
            "pre-registered 400-day plausibility bound"
        )
    if series and dt.size:
        if dt[0] != 0:
            raise FeatureRangeError(
                f"delta_t_days: first observation must carry the 0 sentinel, got {dt[0]}"
            )
        if dt.size > 1:
            bad = dt[1:] < 1
            if bad.any():
                i = int(np.argmax(bad)) + 1
                raise FeatureRangeError(
                    f"delta_t_days: Δt must be ≥ 1 between acquisitions; "
                    f"position {i} has {dt[i]}"
                )


def validate_feature_values(
    name: str,
    values: np.ndarray,
    *,
    tolerance_frac: float = 0.10,
    max_violation_frac: float = 0.05,
) -> dict[str, object]:
    """Assert one feature's observed range against its manifest entry.

    Rejects (naming the feature):
    * no manifest entry (KeyError);
    * a median in DN-scale territory where physical units are required;
    * more than ``max_violation_frac`` of finite values outside
      ``valid_range`` beyond ``tolerance_frac`` of the range width.
    Returns a record with the observed range and soft-range flags (never a
    silent pass).
    """
    spec = manifest_entry(name)  # raises KeyError on missing entry
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return {"feature": name, "n_finite": 0, "observed_range": None,
                "soft_violation_frac": 0.0, "ok": True, "note": "all values absent"}
    obs_lo, obs_hi = float(finite.min()), float(finite.max())
    lo, hi = spec.valid_range
    width = (hi - lo) if (lo is not None and hi is not None) else None
    tol = (tolerance_frac * width) if width else (
        tolerance_frac * max(abs(obs_hi), abs(obs_lo), 1.0)
    )
    below = finite[finite < (lo - tol)] if lo is not None else finite[:0]
    above = finite[finite > (hi + tol)] if hi is not None else finite[:0]
    n_bad = int(below.size + above.size)
    bad_frac = n_bad / finite.size

    # DN-scale rejection: physical-unit features whose median sits orders of
    # magnitude above the valid range (e.g. raw DN ≈ thousands).
    if lo is not None and hi is not None and finite.size >= 8:
        med = float(np.median(finite))
        if med > hi + 5 * max(width, 1.0):
            raise FeatureRangeError(
                f"feature {name!r}: median {med:.4g} is far above the valid "
                f"range [{lo:.4g}, {hi:.4g}] — DN-scale values passed as "
                "converted physical units? The LSDS-1619 conversion must be "
                "applied exactly once upstream."
            )
    if bad_frac > max_violation_frac:
        raise FeatureRangeError(
            f"feature {name!r}: {n_bad}/{finite.size} values "
            f"({bad_frac:.1%}) outside valid range [{lo}, {hi}] "
            f"(tolerance {tol:.4g}); observed [{obs_lo:.4g}, {obs_hi:.4g}]; "
            f"spec: {spec.source}"
        )
    soft_frac = 0.0
    if spec.soft_range is not None:
        slo, shi = spec.soft_range
        smask = np.zeros(finite.shape, dtype=bool)
        if slo is not None:
            smask |= finite < slo
        if shi is not None:
            smask |= finite > shi
        soft_frac = float(smask.sum() / finite.size)
    return {
        "feature": name, "n_finite": int(finite.size),
        "observed_range": [obs_lo, obs_hi], "valid_range": spec.valid_range,
        "violation_frac": bad_frac, "soft_violation_frac": soft_frac,
        "ok": True,
    }


def validate_feature_set(
    features: Mapping[str, np.ndarray],
    *,
    tolerance_frac: float = 0.10,
    max_violation_frac: float = 0.05,
) -> list[dict[str, object]]:
    """Validate a whole feature mapping; first violation raises, naming it."""
    records = []
    for name, values in features.items():
        records.append(
            validate_feature_values(
                name, values,
                tolerance_frac=tolerance_frac, max_violation_frac=max_violation_frac,
            )
        )
    return records


def assert_band_order(band_order) -> tuple[str, ...]:
    """Fail closed unless the group's band order matches the canonical one."""
    order = tuple(str(b) for b in band_order)
    if order != CANONICAL_BAND_ORDER:
        raise ContractViolation(
            f"band order {order} does not match the canonical pipeline order "
            f"{CANONICAL_BAND_ORDER}; a mis-ordered scene silently swaps "
            "spectral features and must never be loaded"
        )
    return order


def is_dn_scale(values: np.ndarray, *, physical_max: float) -> bool:
    """True when the median of finite values is clearly in DN territory.

    Used by callers that must decide whether an array is raw DN (needs the
    LSDS-1619 conversion) or already-converted physical units.
    """
    v = np.asarray(values, dtype=np.float64)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        return False
    return bool(np.median(finite) > 10 * max(physical_max, 1.0))

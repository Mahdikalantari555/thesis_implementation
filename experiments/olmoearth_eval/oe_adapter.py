"""Adapter between the thesis patch store and OlmoEarth v1.2 Nano.

Pure numpy/logic module — no torch import — so the adapter contract is
unit-testable without the model. Isolation rule (openspec change
``evaluate-olmoearth-nano-sugarcane``): this package never imports
``ssl_anomaly``; the Landsat C2 scale/offset constants are replicated here
with their provenance recorded, and any divergence from the production
``io.landsat`` values fails the parity unit test.

Verified facts (2026-11-27, see the change's design.md):
- OlmoEarth v1.2 ``landsat`` token group (single band set, checkpoint
  config.json override): B8,B1,B2,B3,B4,B5,B6,B7,B9,B10,B11.
- OlmoEarth pretraining ingested Landsat Collection 2 **Level-1** raw DN
  (all 11 bands) and normalized with ``computed.json``:
  ``(x - (mean - 2*std)) / (4*std)``.
- Band dropout during training zeroes channels **after normalization**, so
  the trained value for "this band is missing" is normalized 0.0.
- Our patch store holds physical values: SR reflectance (C2 L2 scale
  rho = DN*2.75e-5 - 0.2) and LST Kelvin (ST = DN*0.00341802 + 149.0).
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants (provenance in design.md §3.2)
# ---------------------------------------------------------------------------

# OlmoEarth v1.2 Nano landsat band order (checkpoint config.json override,
# single band group). Order is load-bearing: it indexes the pixel_proj input.
OLMOEARTH_LANDSAT_BANDS = [
    "B8", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B9", "B10", "B11",
]

# Our patch-store channel order (patches.zarr provenance channel_order).
PATCH_CHANNEL_ORDER = ["ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst"]

# Landsat C2 Level-2 SR scaling (usgs.gov; identical constants to the
# production io.landsat.apply_scale_sr — parity-tested in
# tests/unit/test_olmoearth_eval_adapter.py).
SR_SCALE = 0.0000275  # rho = DN * SR_SCALE + SR_OFFSET
SR_OFFSET = -0.2
ST_SCALE = 0.00341802  # Kelvin = DN * ST_SCALE + ST_OFFSET
ST_OFFSET = 149.0

# Landsat 8/9 Collection 2 Level-1 TIRS band 10 radiance scaling
# (RADIANCE_MULT_BAND_10 / RADIANCE_ADD_BAND_10; L9 equal to 1e-9).
B10_RAD_MULT = 3.3420e-4  # L = DN * B10_RAD_MULT + B10_RAD_ADD
B10_RAD_ADD = 0.1

# Inverse Planck constants for radiance in W/(m^2*sr*um) with lam in um.
PLANCK_C1 = 1.19104e8   # W * m^-2 * sr^-1 * um^4 / ... (second radiation constant form)
PLANCK_C2 = 14387.7     # um * K
B10_LAMBDA_UM = 10.9

# Post-normalization fills for the two kinds of absence the adapter must
# distinguish (design.md §3.2, updated after the store-reality check):
# - whole-band absence (no source channel: B8/B1/B9/B11): 0.0, the value
#   band dropout produced during training (channels zeroed post-norm);
# - per-pixel absence inside a band WITH data (no-data pixels, e.g. the
#   out-of-field context the patch store does not contain): 0.5, the neutral
#   band-mean value — the model has no trained "missing pixel" semantics and
#   the mask granularity is the patch, not the pixel.
MISSING_BAND_FILL = 0.0
MISSING_PIXEL_FILL = 0.5

# Patch-store geometry (src/ssl_anomaly/io/patches.py semantics).
BIN_DAYS = 15
N_BINS = 36

# Archive epoch = splits_locked_v2.json base_date (verified at extraction
# time; the constant documents the locked value).
ARCHIVE_EPOCH = date(2013, 3, 30)

# OlmoEarth encoder limits (config.json).
MAX_SEQUENCE_LENGTH = 12
MAX_PATCH_SIZE = 8


class ProvenanceError(RuntimeError):
    """Raised when an artifact hash or binding does not match."""


# ---------------------------------------------------------------------------
# Physical-domain conversions
# ---------------------------------------------------------------------------

def sr_to_dn(reflectance: np.ndarray) -> np.ndarray:
    """Invert the C2 L2 SR scaling: DN = (rho - SR_OFFSET) / SR_SCALE."""
    return (np.asarray(reflectance, dtype=np.float64) - SR_OFFSET) / SR_SCALE


def kelvin_to_b10_dn(temp_k: np.ndarray) -> np.ndarray:
    """Convert surface temperature (Kelvin) to the L1 B10 raw-DN scale.

    Chain: inverse Planck at 10.9 um -> band-10 radiance -> raw DN through
    the L1 radiance scale. This feeds L2 surface temperature into the slot
    the model saw as L1 TOA brightness radiance; the residual semantic gap
    (surface vs brightness temperature) is a documented limitation, not a
    silent equivalence.
    """
    temp_k = np.asarray(temp_k, dtype=np.float64)
    exponent = PLANCK_C2 / (B10_LAMBDA_UM * temp_k)
    radiance = PLANCK_C1 / (B10_LAMBDA_UM**5 * (np.expm1(exponent)))
    return (radiance - B10_RAD_ADD) / B10_RAD_MULT


def b10_dn_to_kelvin(dn: np.ndarray) -> np.ndarray:
    """Forward L1 B10 chain (radiance -> brightness temperature), for tests."""
    radiance = np.asarray(dn, dtype=np.float64) * B10_RAD_MULT + B10_RAD_ADD
    temp_k = PLANCK_C2 / (B10_LAMBDA_UM * np.log(PLANCK_C1 /
                 (B10_LAMBDA_UM**5 * radiance) + 1.0))
    return temp_k


# ---------------------------------------------------------------------------
# Channel assembly + normalization
# ---------------------------------------------------------------------------

# Map each OlmoEarth landsat slot to a patch-store channel or None.
# 'reflectance' slots are converted sr_to_dn; 'lst' via kelvin_to_b10_dn.
LANDSAT_SLOT_SOURCES: dict[str, tuple[str, str] | None] = {
    "B8": None,                                   # pan band: absent in C2 L2 SR
    "B1": None,                                   # coastal: not in the patch store
    "B2": ("blue", "reflectance"),
    "B3": ("green", "reflectance"),
    "B4": ("red", "reflectance"),
    "B5": ("nir", "reflectance"),
    "B6": ("swir1", "reflectance"),
    "B7": ("swir2", "reflectance"),
    "B9": None,                                   # cirrus: absent in C2 L2 SR
    "B10": ("lst", "lst"),
    "B11": None,                                  # TIRS band 2: absent
}

MISSING_LANDSAT_SLOTS = [b for b, s in LANDSAT_SLOT_SOURCES.items() if s is None]


def assemble_landsat_dn(channels: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Assemble the 11-slot landsat band group in **raw-DN physical domain**.

    Args:
        channels: mapping from patch-store channel name to array (any shape),
            in physical units (reflectance for SR bands, Kelvin for ``lst``).

    Returns:
        Mapping from OlmoEarth band name to DN-domain array. Slots without a
        source channel hold NaN (filled post-normalization by
        :func:`finalize_normalized`); a NaN inside a source channel propagates
        (the pixel is masked MISSING upstream of the model).
    """
    out: dict[str, np.ndarray] = {}
    for band, source in LANDSAT_SLOT_SOURCES.items():
        if source is None:
            out[band] = np.full(np.shape(next(iter(channels.values()))), np.nan)
            continue
        name, kind = source
        if name not in channels:
            raise KeyError(f"patch channel {name!r} missing for band {band}")
        arr = np.asarray(channels[name], dtype=np.float64)
        if kind == "reflectance":
            out[band] = sr_to_dn(arr)
        elif kind == "lst":
            out[band] = kelvin_to_b10_dn(arr)
        else:  # pragma: no cover - guarded by construction
            raise ValueError(kind)
    return out


def load_computed_norm(norm_config_path: str | Path) -> dict[str, tuple[float, float]]:
    """Read computed.json -> {band: (mean, std)} for the landsat band order."""
    with open(norm_config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    landsat = cfg["landsat"]
    return {b: (float(landsat[b]["mean"]), float(landsat[b]["std"]))
            for b in OLMOEARTH_LANDSAT_BANDS}


def normalize_landsat(
    dn: dict[str, np.ndarray],
    stats: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Normalize each band with OlmoEarth's computed strategy: (x-(m-2s))/(4s).

    NaNs are preserved here; :func:`finalize_normalized` applies the fills.
    """
    out = {}
    for band, arr in dn.items():
        mean, std = stats[band]
        out[band] = (arr - (mean - 2.0 * std)) / (4.0 * std)
    return out


def finalize_normalized(
    normalized: dict[str, np.ndarray],
    band_fill: float = MISSING_BAND_FILL,
    pixel_fill: float = MISSING_PIXEL_FILL,
) -> np.ndarray:
    """Stack to (..., 11) in OLMOEARTH_LANDSAT_BANDS order with fills applied.

    A band that is NaN **everywhere** (no source channel in the archive) is
    filled with ``band_fill`` (0.0 = the trained band-dropout value). Scattered
    NaN inside a band that has data (no-data pixels) is filled with
    ``pixel_fill`` (0.5 = neutral band mean). Both keep the tensor finite so
    the encoder never sees NaN; pixel-level validity is additionally carried
    by the mask the caller builds from ``pixel_valid``.
    """
    stacked = np.stack(
        [normalized[band] for band in OLMOEARTH_LANDSAT_BANDS], axis=-1
    )
    # per-band "has data anywhere" (reduce over every axis except the band axis)
    band_has_data = np.isfinite(stacked).any(
        axis=tuple(range(stacked.ndim - 1)))                    # (11,)
    fill_vals = np.where(band_has_data, pixel_fill, band_fill)  # (11,)
    return np.where(np.isfinite(stacked), stacked, fill_vals)


def assemble_and_normalize(
    channels: dict[str, np.ndarray],
    stats: dict[str, tuple[float, float]],
    band_fill: float = MISSING_BAND_FILL,
    pixel_fill: float = MISSING_PIXEL_FILL,
    b11_from_b10: bool = False,
) -> np.ndarray:
    """Full chain: physical channels -> (..., 11) normalized finite array.

    ``b11_from_b10`` implements the registered sensitivity arm where the
    absent TIRS-2 slot is filled with the B10 (thermal) value instead of
    ``band_fill``.
    """
    normalized = normalize_landsat(assemble_landsat_dn(channels), stats)
    if b11_from_b10:
        normalized["B11"] = normalized["B10"].copy()
    return finalize_normalized(normalized, band_fill=band_fill,
                               pixel_fill=pixel_fill)


# ---------------------------------------------------------------------------
# Window packing
# ---------------------------------------------------------------------------

def select_valid_bins(
    n_obs: np.ndarray,
    pixel_valid: np.ndarray,
    in_field: np.ndarray,
    min_valid_frac: float = 0.30,
    min_grid_online_count: int = 1,
    patch_size: int = 8,
) -> np.ndarray:
    """Boolean (N_BINS,) mask of usable bins.

    A bin is valid when ALL of:
    (a) at least one acquisition entered it (n_obs > 0);
    (b) the fraction of QA-valid pixels inside the field polygon is at least
        ``min_valid_frac`` (signal-quality criterion, in-field only);
    (c) at least ``min_grid_online_count`` pixels are QA-valid on the model's
        patch-top-left grid (stride ``patch_size`` from the origin) — the
        encoder samples the mask ONLY at each patch's top-left pixel, so a
        bin with zero grid coverage would produce a timestep with no ONLINE
        tokens. NOTE: the patch store carries values only inside the field
        (~9 % of a 48×48 chip ⇒ ~3/36 grid points), so any fraction-based
        grid threshold is unreachable by construction; the floor is absolute.

    ``pixel_valid`` is (N_BINS, 1, H, W); ``in_field`` is (H, W) uint8.
    """
    n_obs = np.asarray(n_obs)
    pixel_valid = np.asarray(pixel_valid)[:, 0]  # drop the singleton channel
    in_field_bool = np.asarray(in_field).astype(bool)
    n_in_field = int(in_field_bool.sum())
    valid = np.zeros(n_obs.shape[0], dtype=bool)
    for b in range(n_obs.shape[0]):
        if n_obs[b] <= 0 or n_in_field == 0:
            continue
        frac = float(pixel_valid[b][in_field_bool].mean())
        grid_count = int(pixel_valid[b, ::patch_size, ::patch_size].sum())
        valid[b] = (frac >= min_valid_frac) and (grid_count >= min_grid_online_count)
    return valid


def pack_windows(
    valid_bins: np.ndarray, max_t: int = MAX_SEQUENCE_LENGTH
) -> list[list[int]]:
    """Pack valid bin indices into consecutive windows of at most max_t.

    Consecutive = consecutive *valid* bins (empty bins are skipped and do not
    consume window slots); a run longer than max_t is split into stride-max_t
    windows so every valid bin appears exactly once.
    """
    idx = np.flatnonzero(np.asarray(valid_bins)).tolist()
    return [idx[i : i + max_t] for i in range(0, len(idx), max_t)]


def bin_center_day(cycle_start_day: int, bin_index: int) -> int:
    """Day offset (from the archive epoch) of a 15-day bin's center."""
    return int(cycle_start_day) + bin_index * BIN_DAYS + BIN_DAYS // 2


def day_to_date(day_index: int, epoch: date = ARCHIVE_EPOCH) -> date:
    """Calendar date of a day offset from the archive epoch."""
    return epoch + timedelta(days=int(day_index))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def verify_sha256(path: str | Path, expected_sha: str, what: str) -> str:
    """Verify a file hash; raise ProvenanceError on mismatch (loud failure)."""
    actual = sha256_file(path)
    if not actual.startswith(expected_sha):
        raise ProvenanceError(
            f"{what}: sha256 mismatch for {path}\n"
            f"  expected {expected_sha}\n  actual   {actual}"
        )
    return actual

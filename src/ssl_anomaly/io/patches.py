"""Field-aligned patch geometry and the per-pixel patch tensor store.

Change ``patch-data-pipeline``. The store ``Data/Processed/satellite/patches.zarr``
holds, for every admitted field-cycle of the locked splits, per-pixel per-bin
values for the 8 B1 channels on a field-aligned ``32×32`` grid.

Geometry contract (spec "Geometry is field-aligned"):
- a field's bounding box is cropped from the scene and **padded, never
  resampled**, to ``S×S`` (default 32);
- the stored static ``in_field`` mask excludes pixels of neighbouring fields,
  so different phenophases cannot leak into spatial context;
- invalid pixel-bins are stored as NaN with ``pixel_valid = 0`` — never
  zero-filled and never interpolated (AGENTS.md §8).

This module is deliberately free of rasterio/geopandas imports at module load:
geometry is pure array math so it is testable without the geo stack. The
export driver (``scripts/extract_patches.py``) does the raster work and calls
:func:`field_window`, :func:`crop_and_pad`, :func:`aggregate_bin` here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

log = logging.getLogger("ssl_anomaly.io.patches")

# ---------------------------------------------------------------------------
# Frozen schema constants (spec "Patch tensor store")
# ---------------------------------------------------------------------------
PATCH_SIZE = 32                     # locked default; --patch-size exists but is not swept
DEFAULT_PATCH_SIZE = PATCH_SIZE
CHANNELS: tuple[str, ...] = (
    "ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst",
)                                   # = B1's channel set (scripts/trial_common.py:17)
N_CHANNELS = len(CHANNELS)
N_BINS = 36
BIN_DAYS = 15
MAX_CYCLE_DAYS = 540
SENSOR_IDS: dict[str, int] = {"L8": 1, "L9": 2, "unknown": 0}
SENSOR_NAMES: dict[int, str] = {v: k for k, v in SENSOR_IDS.items()}

PATCH_STORE_NAME = "patches.zarr"
MANIFEST_NAME = ".manifest.jsonl"
NORM_STATS_NAME = "norm_stats.json"
PROVENANCE_NAME = "provenance.json"

QA_BITS_USED = (
    "fill(0)", "dilated_cloud(1)", "cirrus(2)", "cloud(3)", "cloud_shadow(4)", "snow(5)",
)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FieldWindow:
    """Where a field's patch lives inside the raster grid.

    ``row0/col0`` are the bbox origin (raster pixel indices, may be negative
    when the polygon touches the raster edge); ``height/width`` the bbox size;
    ``pad_top/pad_left`` the padding applied to centre the bbox in ``size×size``
    (bottom/right padding is implied by the size).
    """

    row0: int
    col0: int
    height: int
    width: int
    size: int = DEFAULT_PATCH_SIZE
    pad_top: int = 0
    pad_left: int = 0

    @property
    def pad_bottom(self) -> int:
        return self.size - self.height - self.pad_top

    @property
    def pad_right(self) -> int:
        return self.size - self.width - self.pad_left

    def as_dict(self) -> dict[str, int]:
        return {
            "row0": int(self.row0), "col0": int(self.col0),
            "height": int(self.height), "width": int(self.width),
            "size": int(self.size),
            "pad_top": int(self.pad_top), "pad_left": int(self.pad_left),
        }

    def to_bbox_origin(self) -> tuple[int, int]:
        """Stored ``bbox_origin`` = the patch's top-left corner in raster space."""
        return (int(self.row0 - self.pad_top), int(self.col0 - self.pad_left))


def field_window(
    polygon_mask: np.ndarray,
    *,
    size: int = DEFAULT_PATCH_SIZE,
    inner_buffer_px: int = 0,
) -> FieldWindow | None:
    """Derive the padded window for one field from its rasterised polygon.

    ``polygon_mask`` is a boolean array where True = this field's pixels.
    Returns ``None`` when the field has no pixel (nothing to extract).

    The bbox is centred inside ``size×size`` so that a field smaller than the
    patch sits in the middle — never resized (spec: "no resampling of pixel
    values occurs"). A field larger than ``size`` would not fit; that is a hard
    error rather than a silent crop, because 100 % of the 490 fields fit in
    theory (median 16.7 px, max 19.6 px) and a violation means the geometry
    assumptions changed.
    """
    mask = np.asarray(polygon_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"polygon_mask must be 2-D, got shape {mask.shape}")
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None

    if inner_buffer_px:
        mask = _inner_buffer(mask, int(inner_buffer_px))
        if not mask.any():
            return None  # buffer consumed the whole field — recorded by the caller
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))

    row0, row1 = int(rows[0]), int(rows[-1]) + 1
    col0, col1 = int(cols[0]), int(cols[-1]) + 1
    height, width = row1 - row0, col1 - col0
    if height > size or width > size:
        raise ValueError(
            f"field bounding box {height}×{width} px exceeds the {size}×{size} patch. "
            "The 32 px patch was sized from the measured field distribution (median 16.7, "
            "max 19.6 px); a larger field means either --patch-size must grow or the "
            "geometry memo is out of date — do not silently crop."
        )
    pad_top = (size - height) // 2
    pad_left = (size - width) // 2
    return FieldWindow(
        row0=row0, col0=col0, height=height, width=width,
        size=size, pad_top=pad_top, pad_left=pad_left,
    )


def _inner_buffer(mask: np.ndarray, px: int) -> np.ndarray:
    """Erode the polygon mask by ``px`` pixels (4-connected), no scipy needed."""
    out = mask.copy()
    for _ in range(int(px)):
        padded = np.pad(out, 1, constant_values=False)
        eroded = (
            out
            & padded[:-2, 1:-1] & padded[2:, 1:-1]
            & padded[1:-1, :-2] & padded[1:-1, 2:]
        )
        out = eroded
    return out


def crop_and_pad(
    array: np.ndarray,
    window: FieldWindow,
    *,
    fill: float = np.nan,
    dtype: np.dtype | str | None = None,
) -> np.ndarray:
    """Crop ``array`` to ``window`` and pad it into ``size×size`` (no resampling).

    ``array`` may be the full raster, or any array already aligned to the same
    grid. Pixels outside ``array`` (negative origins at the raster edge) are
    filled with ``fill`` — NaN by default, never a fabricated value.
    """
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"array must be 2-D, got shape {arr.shape}")
    out_dtype = np.dtype(dtype) if dtype is not None else arr.dtype
    out = np.full((window.size, window.size), fill, dtype=out_dtype)

    src_r0, src_c0 = window.row0, window.col0
    dst_r0, dst_c0 = window.pad_top, window.pad_left

    # Clip the bbox to the raster bounds on both sides.
    r_lo = max(0, -src_r0)
    c_lo = max(0, -src_c0)
    r_hi = min(window.height, arr.shape[0] - src_r0)
    c_hi = min(window.width, arr.shape[1] - src_c0)
    if r_hi <= r_lo or c_hi <= c_lo:
        return out  # window lies entirely outside the raster

    block = arr[
        src_r0 + r_lo : src_r0 + r_hi,
        src_c0 + c_lo : src_c0 + c_hi,
    ]
    out[
        dst_r0 + r_lo : dst_r0 + r_hi,
        dst_c0 + c_lo : dst_c0 + c_hi,
    ] = block.astype(out_dtype, copy=False)
    return out


def patch_in_field_mask(polygon_mask: np.ndarray, window: FieldWindow) -> np.ndarray:
    """The static ``in_field`` mask on the patch grid (uint8 0/1).

    Pixels outside the field polygon are 0 — including pixels of *neighbouring*
    fields that fall inside the same bounding box, which is what keeps
    different phenophases out of the spatial context.
    """
    m = crop_and_pad(polygon_mask, window, fill=False, dtype=bool)
    return m.astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-pixel per-bin aggregation
# ---------------------------------------------------------------------------
@dataclass
class BinAggregate:
    """One (field, cycle, bin) slice of the store."""

    values: np.ndarray          # (C, S, S) float16/float32, NaN where invalid
    pixel_valid: np.ndarray     # (S, S) uint8
    n_obs: int                  # acquisitions contributing to this bin
    sensor_id: int              # majority sensor id in this bin


def aggregate_bin(
    scenes: Sequence[np.ndarray],
    valid_masks: Sequence[np.ndarray],
    *,
    in_field: np.ndarray | None = None,
    dtype: np.dtype | str = np.float16,
) -> BinAggregate:
    """Per-pixel median over the valid acquisitions inside one bin.

    Parameters
    ----------
    scenes
        One ``(C, S, S)`` float array per acquisition in the bin (already
        cropped/padded by the caller). NaN marks an invalid pixel value.
    valid_masks
        One boolean ``(S, S)`` array per acquisition — the QA mask.
    in_field
        Optional ``(S, S)`` 0/1 polygon mask. When given, pixels outside the
        field are forced invalid (they are never asserted as observed).

    Semantics (mirrors the field-level median convention in
    ``monitor/inputs.py``): a pixel-bin is valid iff at least one acquisition
    has QA-valid, finite values; the value is the median of those. No
    interpolation, no zero-fill. ``n_obs`` counts acquisitions with at least
    one valid in-field pixel.
    """
    if len(scenes) != len(valid_masks):
        raise ValueError(
            f"{len(scenes)} scenes but {len(valid_masks)} validity masks"
        )
    if not scenes:
        s = in_field.shape[0] if in_field is not None else 0
        return BinAggregate(
            values=np.full((N_CHANNELS, s, s), np.nan, dtype=dtype),
            pixel_valid=np.zeros((s, s), dtype=np.uint8),
            n_obs=0, sensor_id=SENSOR_IDS["unknown"],
        )

    first = np.asarray(scenes[0])
    if first.ndim != 3:
        raise ValueError(f"scene arrays must be (C, S, S), got {first.shape}")
    c, size, _ = first.shape
    stack = np.full((len(scenes), c, size, size), np.nan, dtype=np.float64)
    valid = np.zeros((len(scenes), size, size), dtype=bool)
    for i, (sc, vm) in enumerate(zip(scenes, valid_masks)):
        arr = np.asarray(sc, dtype=np.float64)
        if arr.shape != (c, size, size):
            raise ValueError(f"scene {i} has shape {arr.shape}, expected {(c, size, size)}")
        stack[i] = arr
        v = np.asarray(vm, dtype=bool)
        if v.shape != (size, size):
            raise ValueError(f"mask {i} has shape {v.shape}, expected {(size, size)}")
        valid[i] = v

    if in_field is not None:
        inside = np.asarray(in_field, dtype=bool)
        if inside.shape != (size, size):
            raise ValueError(f"in_field has shape {inside.shape}, expected {(size, size)}")
        valid &= inside[None, :, :]

    # A pixel is observed in an acquisition when it is QA-valid AND finite.
    observed = valid[:, None, :, :] & np.isfinite(stack)
    any_obs = observed.any(axis=0)                      # (C, S, S)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        # All-NaN slices (whole pixels with no observation) are expected here and
        # stay NaN by design — silence the specific numpy warning, not all of them.
        warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                category=RuntimeWarning)
        values = np.nanmedian(np.where(observed, stack, np.nan), axis=0)
    values[~any_obs] = np.nan

    pixel_valid = observed.any(axis=0).any(axis=0)       # (S, S) any channel observed
    n_obs = int(observed.any(axis=(1, 2, 3)).sum())

    return BinAggregate(
        values=values.astype(dtype),
        pixel_valid=pixel_valid.astype(np.uint8),
        n_obs=n_obs,
        sensor_id=SENSOR_IDS["unknown"],
    )


def majority_sensor(sensor_ids: Iterable[int]) -> int:
    """Majority sensor id over the acquisitions of one bin (0 when empty)."""
    ids = [int(s) for s in sensor_ids]
    if not ids:
        return SENSOR_IDS["unknown"]
    vals, counts = np.unique(ids, return_counts=True)
    # Deterministic tie-break: the smaller id wins (L8 before L9).
    order = np.lexsort((vals, -counts))
    return int(vals[order[0]])


# ---------------------------------------------------------------------------
# Zarr store: schema, writer, reader
# ---------------------------------------------------------------------------
def store_schema(
    n_fields: int, max_cycles: int, size: int = DEFAULT_PATCH_SIZE,
) -> dict[str, tuple[tuple[int, ...], str]]:
    """Names, shapes and dtypes of every array in the store."""
    return {
        "values": ((n_fields, max_cycles, N_BINS, N_CHANNELS, size, size), "<f2"),
        "pixel_valid": ((n_fields, max_cycles, N_BINS, 1, size, size), "|u1"),
        "in_field": ((n_fields, 1, size, size), "|u1"),
        "bbox_origin": ((n_fields, 2), "<u2"),
        "n_obs": ((n_fields, max_cycles, N_BINS), "|u1"),
        "sensor_id": ((n_fields, max_cycles, N_BINS), "|u1"),
        "field_ids": ((n_fields,), "<i8"),
        "cycle_ids": ((n_fields, max_cycles), "<i8"),
        "cycle_start_day": ((n_fields, max_cycles), "<i8"),
        "cycle_valid": ((n_fields, max_cycles), "|u1"),
    }


def estimate_store_bytes(n_fields: int, n_cycles: int, size: int = DEFAULT_PATCH_SIZE) -> int:
    """Uncompressed size estimate (the ~1.0 GB figure quoted in the proposal)."""
    return int(
        n_cycles * N_BINS * N_CHANNELS * size * size * 2      # fp16 values
        + n_cycles * N_BINS * size * size                      # uint8 validity
        + n_fields * size * size                               # in_field
        + n_cycles * N_BINS * 2                                # n_obs + sensor
        + n_fields * 2 + n_cycles * 8 * 2                      # origins + cycle arrays
    )


def open_store(path: str | Path, *, mode: str = "r"):
    """Open (or create) a patches.zarr group. Imported lazily: zarr only here."""
    import zarr

    return zarr.open_group(str(path), mode=mode)


def create_store(
    path: str | Path,
    *,
    field_ids: Sequence[int],
    max_cycles: int,
    size: int = DEFAULT_PATCH_SIZE,
    attrs: dict[str, Any] | None = None,
):
    """Create the store with the frozen schema and chunking.

    Chunks keep per-(field, cycle) reads cheap: ``values`` is chunked one cycle
    at a time, so the Dataset can lazily read a single field-cycle.
    """
    import zarr

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open_group(str(path), mode="w")
    schema = store_schema(len(field_ids), max_cycles, size)
    for name, (shape, dtype) in schema.items():
        if name == "values":
            chunks = (1, 1, N_BINS, N_CHANNELS, size, size)
        elif name == "pixel_valid":
            chunks = (1, 1, N_BINS, 1, size, size)
        elif name == "in_field":
            chunks = (1, 1, size, size)
        elif name in ("n_obs", "sensor_id"):
            chunks = (1, 1, N_BINS)          # per-cycle reads stay cheap
        elif name.startswith("cycle_"):
            chunks = (1, max_cycles)
        else:
            # 1-D arrays (field_ids): one chunk, matching the array's rank.
            chunks = tuple(int(s) for s in shape)
        opts = dict(
            shape=tuple(int(s) for s in shape),
            dtype=dtype,
            chunks=chunks,
            fill_value=(float("nan") if np.dtype(dtype).kind == "f" else 0),
        )
        # zarr 3 exposes create_array; zarr 2 only create_dataset.
        creator = getattr(root, "create_array", None)
        if creator is not None:
            creator(name, overwrite=True, **opts)
        else:  # pragma: no cover - zarr 2 path
            root.create_dataset(name, overwrite=True, **opts)
    root.attrs.update(attrs or {})
    root.attrs["schema"] = {k: {"shape": list(v[0]), "dtype": v[1]} for k, v in schema.items()}
    root.attrs["channel_order"] = list(CHANNELS)
    root.attrs["patch_size"] = int(size)
    return root


def validate_store(root, *, expected_fields: int | None = None) -> dict:
    """Validate the on-disk store against the frozen schema.

    Raises ``ValueError`` naming the first defect — a missing array, a wrong
    dtype, a wrong channel axis, or a field count that disagrees with the
    registry. Returns a summary dict for the provenance record.
    """
    required = set(store_schema(1, 1)) - {"values", "pixel_valid", "in_field",
                                          "bbox_origin", "n_obs", "sensor_id",
                                          "field_ids", "cycle_ids",
                                          "cycle_start_day", "cycle_valid"}
    missing = [k for k in store_schema(1, 1) if k not in root]
    if missing:
        raise ValueError(f"patch store is missing arrays: {missing}")
    del required

    values = root["values"]
    if len(values.shape) != 6:
        raise ValueError(f"values must be 6-D (F, C, 36, 8, S, S), got {values.shape}")
    n_fields, max_cycles, n_bins, n_ch, size, size2 = values.shape
    if n_bins != N_BINS:
        raise ValueError(f"values bin axis is {n_bins}, expected {N_BINS}")
    if n_ch != N_CHANNELS:
        raise ValueError(f"values channel axis is {n_ch}, expected {N_CHANNELS}")
    if size != size2:
        raise ValueError(f"patch is not square: {size}×{size2}")
    if expected_fields is not None and n_fields != expected_fields:
        raise ValueError(f"store holds {n_fields} fields, registry says {expected_fields}")

    if root.attrs.get("channel_order") and tuple(root.attrs["channel_order"]) != CHANNELS:
        raise ValueError(
            f"channel_order {root.attrs['channel_order']} != frozen {list(CHANNELS)}"
        )
    for name, expect_prefix in (("pixel_valid", "u1"), ("in_field", "u1"),
                                ("bbox_origin", "u2")):
        dt = root[name].dtype
        if dt.kind != expect_prefix[0]:
            raise ValueError(f"{name} dtype {dt} does not match schema")

    return {
        "n_fields": int(n_fields),
        "max_cycles": int(max_cycles),
        "n_bins": int(n_bins),
        "n_channels": int(n_ch),
        "patch_size": int(size),
        "n_cycles_recorded": int(np.asarray(root["cycle_valid"][:]).sum()),
        "bytes_estimate": estimate_store_bytes(int(n_fields), int(max_cycles), int(size)),
    }


# ---------------------------------------------------------------------------
# Frozen normalization artifact (spec "Leakage-free frozen normalization")
# ---------------------------------------------------------------------------
def fit_patch_norm(
    values: np.ndarray,
    sensor_ids: np.ndarray,
    *,
    channel_order: Sequence[str] = CHANNELS,
) -> dict:
    """Per-channel × per-sensor robust median/IQR from TRAIN data only.

    ``values`` is ``(N, C)`-flattened observations paired with a sensor id per
    observation. The CALLER is responsible for passing train-split fields ×
    train periods only — this function has no way to know the splits, so the
    leakage guard lives in :func:`build_patch_norm` and in the exporter's
    selection, and is proven by the tests.

    IQR is used (not MAD) because the store's fp16 values quantise; IQR is
    stable under that quantisation and matches the proposal's wording.
    """
    vals = np.asarray(values, dtype=np.float64)
    sids = np.asarray(sensor_ids, dtype=np.int64)
    if vals.ndim != 2 or vals.shape[1] != len(channel_order):
        raise ValueError(
            f"values must be (N, {len(channel_order)}), got {vals.shape}"
        )
    if len(sids) != len(vals):
        raise ValueError(f"{len(sids)} sensor ids for {len(vals)} rows")

    out: dict[str, dict[str, list[float]]] = {}
    for sid in sorted(set(int(s) for s in sids) | {SENSOR_IDS["unknown"]}):
        sel = sids == sid
        per_channel = {}
        for ci, name in enumerate(channel_order):
            v = vals[sel, ci]
            v = v[np.isfinite(v)]
            if v.size == 0:
                per_channel[name] = {"median": None, "iqr": None, "n": 0}
                continue
            q25, q75 = np.percentile(v, [25.0, 75.0])
            per_channel[name] = {
                "median": float(np.median(v)),
                "iqr": float(max(q75 - q25, 0.0)),
                "n": int(v.size),
            }
        out[SENSOR_NAMES.get(sid, str(sid))] = per_channel
    return out


def build_patch_norm(
    *,
    root,
    train_field_ids: Iterable[int],
    train_cycle_mask: np.ndarray | None = None,
    splits_sha256: str | None = None,
    channel_order: Sequence[str] = CHANNELS,
) -> dict:
    """Build the frozen normalization artifact from TRAIN fields/cycles only.

    ``train_cycle_mask`` is a boolean ``(F, C)`` mask aligned to the store;
    when omitted, every recorded cycle of a train field is used. Dev/test
    fields are excluded by construction: their field ids are simply not in
    ``train_field_ids``.
    """
    import numpy as np

    field_ids = np.asarray(root["field_ids"][:], dtype=np.int64)
    want = set(int(f) for f in train_field_ids)
    rows = [i for i, f in enumerate(field_ids) if int(f) in want]
    if not rows:
        raise ValueError("no train fields present in the store — refusing to fit norm stats")

    values = np.asarray(root["values"][:], dtype=np.float32)
    sensor_ids = np.asarray(root["sensor_id"][:], dtype=np.int64)
    in_field = np.asarray(root["in_field"][:], dtype=bool)
    cycle_valid = np.asarray(root["cycle_valid"][:], dtype=bool)
    if train_cycle_mask is not None:
        m = np.asarray(train_cycle_mask, dtype=bool)
        if m.shape != cycle_valid.shape:
            raise ValueError(f"train_cycle_mask shape {m.shape} != store {cycle_valid.shape}")
        cycle_valid = cycle_valid & m

    flat_vals: list[np.ndarray] = []
    flat_sensors: list[np.ndarray] = []
    for fi in rows:
        for ci in np.flatnonzero(cycle_valid[fi]):
            v = values[fi, ci]                       # (36, 8, S, S)
            s = sensor_ids[fi, ci]                   # (36,)
            mask = in_field[fi, 0].astype(bool)      # (S, S)
            for b in range(v.shape[0]):
                if not mask.any():
                    continue
                sel = v[b][:, mask]                  # (8, n_px)
                if not np.isfinite(sel).any():
                    continue
                flat_vals.append(sel.T)              # (n_px, 8)
                flat_sensors.append(np.full(sel.shape[1], int(s[b]), dtype=np.int64))
    if not flat_vals:
        raise ValueError("no finite in-field train pixel observations — refusing to fit norm stats")

    stats = fit_patch_norm(
        np.concatenate(flat_vals), np.concatenate(flat_sensors), channel_order=channel_order,
    )
    return {
        "stats": stats,
        "computed_from": "train-split fields x train periods only",
        "field_ids_used": sorted(int(field_ids[i]) for i in rows),
        "splits_sha256": splits_sha256,
        "channel_order": list(channel_order),
        "n_train_fields": len(rows),
        "n_train_cycles": int(sum(len(np.flatnonzero(cycle_valid[fi])) for fi in rows)),
        "stat": "median/iqr (per channel, per sensor)",
    }


def norm_stats_sha256(norm: dict) -> str:
    """Stable SHA-256 over the norm artifact (its ``sha256`` field excluded)."""
    payload = {k: v for k, v in norm.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def save_patch_norm(norm: dict, path: str | Path) -> Path:
    """Freeze the norm artifact with its own sha256 inside the JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frozen = dict(norm)
    frozen["sha256"] = norm_stats_sha256(norm)
    path.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_patch_norm(path: str | Path, *, verify: bool = True) -> dict:
    """Load the frozen norm artifact, verifying its sha256 (fail-closed)."""
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"patch normalization artifact missing at {path}. Build it once with "
            "`python scripts/extract_patches.py --build-norm` (train fields only); "
            "training consumers refuse to start without it."
        )
    norm = json.loads(path.read_text(encoding="utf-8"))
    if verify:
        recorded = norm.get("sha256")
        if not recorded or norm_stats_sha256(norm) != recorded:
            raise RuntimeError(
                f"patch norm artifact at {path} failed its sha256 check — the frozen "
                "statistics were modified. Refusing to continue (leakage-free "
                "normalization requirement)."
            )
    return norm


# ---------------------------------------------------------------------------
# Provenance manifest
# ---------------------------------------------------------------------------
@dataclass
class Provenance:
    """Provenance record written beside the store (spec: provenance manifest)."""

    git_sha: str = "unknown"
    cycles_sha256: str | None = None
    splits_sha256: str | None = None
    holdout_manifest_sha256: str | None = None
    source_products: list[str] = field(default_factory=list)
    qa_bits: list[str] = field(default_factory=lambda: list(QA_BITS_USED))
    scale_factors: dict[str, float] = field(default_factory=dict)
    patch_size: int = DEFAULT_PATCH_SIZE
    channel_order: list[str] = field(default_factory=lambda: list(CHANNELS))
    n_fields: int = 0
    n_cycles: int = 0
    n_dates_done: int = 0
    output_sha256: str | None = None
    created: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "git_sha": self.git_sha,
            "cycles_sha256": self.cycles_sha256,
            "splits_sha256": self.splits_sha256,
            "holdout_manifest_sha256": self.holdout_manifest_sha256,
            "source_products": list(self.source_products),
            "qa_bits": list(self.qa_bits),
            "scale_factors": dict(self.scale_factors),
            "patch_size": int(self.patch_size),
            "channel_order": list(self.channel_order),
            "n_fields": int(self.n_fields),
            "n_cycles": int(self.n_cycles),
            "n_dates_done": int(self.n_dates_done),
            "output_sha256": self.output_sha256,
            "created": self.created,
        }
        d.update(self.extra)
        return d

    def validate(self) -> None:
        """Refuse to write an incomplete provenance record (AGENTS.md §2.4)."""
        missing = [
            name for name, val in (
                ("cycles_sha256", self.cycles_sha256),
                ("splits_sha256", self.splits_sha256),
                ("created", self.created),
            ) if not val
        ]
        if missing:
            raise ValueError(f"provenance record incomplete: missing {missing}")
        if self.n_fields <= 0:
            raise ValueError("provenance n_fields must be positive")


def output_sha256(root, *, sample: int = 512) -> str:
    """Cheap content hash of the store: shapes + a deterministic value sample."""
    h = hashlib.sha256()
    for name in sorted(store_schema(1, 1)):
        arr = root[name]
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
    vals = np.asarray(root["values"][:], dtype=np.float32)
    flat = vals.reshape(-1)
    finite_idx = np.flatnonzero(np.isfinite(flat))
    if finite_idx.size:
        step = max(1, finite_idx.size // sample)
        pick = finite_idx[::step][:sample]
        h.update(np.round(flat[pick], 4).astype(np.float32).tobytes())
    h.update(str(int(np.isfinite(vals).sum())).encode())
    return h.hexdigest()


__all__ = [
    "BIN_DAYS",
    "CHANNELS",
    "DEFAULT_PATCH_SIZE",
    "FieldWindow",
    "MANIFEST_NAME",
    "MAX_CYCLE_DAYS",
    "N_BINS",
    "N_CHANNELS",
    "NORM_STATS_NAME",
    "PATCH_SIZE",
    "PATCH_STORE_NAME",
    "PROVENANCE_NAME",
    "Provenance",
    "QA_BITS_USED",
    "SENSOR_IDS",
    "SENSOR_NAMES",
    "aggregate_bin",
    "build_patch_norm",
    "create_store",
    "crop_and_pad",
    "estimate_store_bytes",
    "field_window",
    "fit_patch_norm",
    "load_patch_norm",
    "majority_sensor",
    "norm_stats_sha256",
    "open_store",
    "output_sha256",
    "patch_in_field_mask",
    "save_patch_norm",
    "store_schema",
    "validate_store",
]

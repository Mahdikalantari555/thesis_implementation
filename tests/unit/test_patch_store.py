"""Tests for the patch store schema, the leakage-free normalization artifact and
the provenance record (change patch-data-pipeline, tasks 1.2, 2.2, 2.3).

The zarr round-trip uses the same ``zarr`` package the repo already depends on
(pyproject: ``zarr>=2.18``) and a tiny synthetic store, so no data archive and
no rasterio/geopandas are required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests.patch_fixtures import bind_patch_store  # noqa: E402

from ssl_anomaly.io import patches as P  # noqa: E402
from ssl_anomaly.monitor.holdout import HoldoutViolationError  # noqa: E402

zarr = pytest.importorskip("zarr")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_store(tmp_path: Path, *, field_ids=(301, 302, 101), n_cycles=2, size=8):
    root = P.create_store(
        tmp_path / "patches.zarr", field_ids=list(field_ids),
        max_cycles=n_cycles, size=size,
    )
    root["field_ids"][:] = np.asarray(field_ids, dtype=np.int64)
    return root


def fill_cycle(root, fi: int, ci: int, *, value: float, sensor: int = 1, in_field=True):
    size = root["values"].shape[-1]
    blocks = np.full((P.N_BINS, P.N_CHANNELS, size, size), value, dtype=np.float32)
    root["values"][fi, ci] = blocks.astype(np.float16)
    root["pixel_valid"][fi, ci, :, 0] = 1
    root["n_obs"][fi, ci] = 3
    root["sensor_id"][fi, ci] = sensor
    root["cycle_valid"][fi, ci] = 1
    root["cycle_ids"][fi, ci] = 100 + ci
    root["cycle_end_day"][fi, ci] = P.MAX_CYCLE_DAYS - 1
    if in_field:
        root["in_field"][fi, 0] = np.ones((size, size), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Task 1.2 — schema, writer, reader
# ---------------------------------------------------------------------------
def test_store_has_the_frozen_arrays_and_dtypes(tmp_path: Path) -> None:
    root = make_store(tmp_path)
    for name in ("values", "pixel_valid", "in_field", "bbox_origin",
                 "n_obs", "sensor_id", "field_ids", "cycle_ids",
                 "cycle_start_day", "cycle_end_day", "cycle_valid"):
        assert name in root, f"{name} missing from the store"
    assert root["values"].shape == (3, 2, P.N_BINS, P.N_CHANNELS, 8, 8)
    assert root["values"].dtype == np.float16
    assert root["pixel_valid"].dtype == np.uint8
    assert root["in_field"].shape == (3, 1, 8, 8)
    assert tuple(root.attrs["channel_order"]) == P.CHANNELS
    assert root.attrs["patch_size"] == 8


def test_validate_store_reports_a_summary(tmp_path: Path) -> None:
    root = make_store(tmp_path)
    fill_cycle(root, 0, 0, value=0.5)
    summary = P.validate_store(root, expected_fields=3)
    assert summary["n_fields"] == 3
    assert summary["n_cycles_recorded"] == 1
    assert summary["n_channels"] == P.N_CHANNELS
    assert summary["patch_size"] == 8


def test_validate_store_rejects_wrong_field_count(tmp_path: Path) -> None:
    root = make_store(tmp_path)
    with pytest.raises(ValueError, match="registry says"):
        P.validate_store(root, expected_fields=490)


def test_validate_store_rejects_tampered_channel_order(tmp_path: Path) -> None:
    root = make_store(tmp_path)
    root.attrs["channel_order"] = ["ndvi", "red"]
    with pytest.raises(ValueError, match="channel_order"):
        P.validate_store(root)


def test_validate_store_rejects_missing_array(tmp_path: Path) -> None:
    root = make_store(tmp_path)
    del root["sensor_id"]
    with pytest.raises(ValueError, match="missing arrays"):
        P.validate_store(root)


def test_invalid_pixel_bins_are_nan_not_zero(tmp_path: Path) -> None:
    """Spec: never zero-filled. A newly created store must read as NaN."""
    root = make_store(tmp_path)
    size = root["values"].shape[-1]
    block = np.asarray(root["values"][0, 0, :, 0], dtype=np.float32)
    assert block.shape == (P.N_BINS, size, size)
    assert np.isnan(block).all(), "untouched pixel-bins must be NaN, not 0"
    assert np.asarray(root["pixel_valid"][0, 0]).sum() == 0


def test_schema_is_derived_from_constants_not_hard_coded(tmp_path: Path) -> None:
    schema = P.store_schema(4, 7, size=16)
    assert schema["values"][0] == (4, 7, P.N_BINS, P.N_CHANNELS, 16, 16)
    assert schema["n_obs"][0] == (4, 7, P.N_BINS)


# ---------------------------------------------------------------------------
# Task 2.2 — leakage-free normalization
# ---------------------------------------------------------------------------
def fit(root, train_keys=None):
    splits, holdout = bind_patch_store(root, spatial=(101,), known=(101, 301, 302), train_keys=train_keys)
    return P.build_patch_norm(root=root, splits_manifest=splits, holdout_manifest=holdout)


def test_norm_artifact_uses_train_fields_only(tmp_path: Path) -> None:
    root = make_store(tmp_path, field_ids=(301, 302, 101))
    fill_cycle(root, 0, 0, value=0.30)
    fill_cycle(root, 1, 0, value=0.32)
    fill_cycle(root, 2, 0, value=9.00)
    norm = fit(root)
    stats = norm["stats"]["L8"]["ndvi"]
    assert stats["n"] > 0
    assert 0.29 < stats["median"] < 0.33
    assert norm["field_ids_used"] == [301, 302]
    assert norm["splits_sha256"] == root.attrs["splits_manifest_sha256"]
    assert norm["n_train_fields"] == 2


def test_norm_artifact_is_per_sensor(tmp_path: Path) -> None:
    root = make_store(tmp_path, field_ids=(301, 302))
    fill_cycle(root, 0, 0, value=0.30, sensor=P.SENSOR_IDS["L8"])
    fill_cycle(root, 1, 0, value=0.60, sensor=P.SENSOR_IDS["L9"])
    norm = fit(root)
    assert 0.28 < norm["stats"]["L8"]["ndvi"]["median"] < 0.32
    assert 0.58 < norm["stats"]["L9"]["ndvi"]["median"] < 0.62


def test_norm_ignores_same_fields_dev_and_holdout_periods(tmp_path: Path) -> None:
    root = make_store(tmp_path, field_ids=(301,), n_cycles=3)
    fill_cycle(root, 0, 0, value=0.30)
    fill_cycle(root, 0, 1, value=9.00)
    fill_cycle(root, 0, 2, value=9.00)
    root["cycle_start_day"][0, 2] = root["cycle_end_day"][0, 2] = 4100
    norm = fit(root, train_keys=["301:100"])
    assert 0.29 < norm["stats"]["L8"]["ndvi"]["median"] < 0.31
    assert norm["train_cycle_keys"] == ["301:100"]


def test_norm_rejects_cutoff_crossing_locked_train_cycle(tmp_path: Path) -> None:
    root = make_store(tmp_path, field_ids=(301,))
    fill_cycle(root, 0, 0, value=0.30)
    root["cycle_end_day"][0, 0] = 4100
    with pytest.raises(HoldoutViolationError, match="301:100"):
        fit(root)


def test_norm_refuses_without_train_cycles(tmp_path: Path) -> None:
    root = make_store(tmp_path, field_ids=(301,))
    fill_cycle(root, 0, 0, value=0.30)
    with pytest.raises(ValueError, match="no admissible locked train cycles"):
        fit(root, train_keys=[])


def test_norm_refuses_without_finite_observations(tmp_path: Path) -> None:
    root = make_store(tmp_path, field_ids=(301,))
    root["cycle_valid"][0, 0] = 1
    with pytest.raises(ValueError, match="no finite in-field"):
        fit(root)


def test_norm_roundtrip_is_hash_verified(tmp_path: Path) -> None:
    root = make_store(tmp_path, field_ids=(301,))
    fill_cycle(root, 0, 0, value=0.3)
    norm = fit(root)
    path = P.save_patch_norm(norm, tmp_path / "norm_stats.json")
    loaded = P.load_patch_norm(path)
    assert loaded["sha256"] == P.norm_stats_sha256(norm)
    assert loaded["stats"]["L8"]["ndvi"]["median"] == norm["stats"]["L8"]["ndvi"]["median"]


def test_norm_tampering_is_detected(tmp_path: Path) -> None:
    root = make_store(tmp_path, field_ids=(301,))
    fill_cycle(root, 0, 0, value=0.3)
    norm = fit(root)
    path = P.save_patch_norm(norm, tmp_path / "norm_stats.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["stats"]["L8"]["ndvi"]["median"] = 0.99
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError, match="sha256"):
        P.load_patch_norm(path)


def test_missing_norm_artifact_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing at"):
        P.load_patch_norm(tmp_path / "nope.json")


# ---------------------------------------------------------------------------
# Task 2.3 — provenance record
# ---------------------------------------------------------------------------
def test_provenance_requires_the_hashes() -> None:
    prov = P.Provenance(n_fields=490, n_cycles=1717, created="2026-11-16")
    with pytest.raises(ValueError, match="missing"):
        prov.validate()


def test_provenance_rejects_empty_store() -> None:
    prov = P.Provenance(
        n_fields=0, n_cycles=0, created="2026-11-16",
        cycles_sha256="a" * 64, splits_sha256="b" * 64,
    )
    with pytest.raises(ValueError, match="n_fields"):
        prov.validate()


def test_provenance_records_qa_bits_and_scale_factors() -> None:
    prov = P.Provenance(
        n_fields=490, n_cycles=1717, created="2026-11-16",
        cycles_sha256="a" * 64, splits_sha256="b" * 64,
        scale_factors={"sr": 2.75e-5, "st": 0.00341802},
        output_sha256="c" * 64,
    )
    prov.validate()
    d = prov.to_dict()
    assert d["qa_bits"] == list(P.QA_BITS_USED)
    assert d["scale_factors"]["sr"] == pytest.approx(2.75e-5)
    assert d["patch_size"] == 32
    assert d["channel_order"] == list(P.CHANNELS)


def test_output_sha256_is_deterministic_and_value_sensitive(tmp_path: Path) -> None:
    root = make_store(tmp_path, field_ids=(301,))
    fill_cycle(root, 0, 0, value=0.30)
    sha1 = P.output_sha256(root)
    sha2 = P.output_sha256(root)
    assert sha1 == sha2, "the store hash must be deterministic"
    fill_cycle(root, 0, 0, value=0.31)
    assert P.output_sha256(root) != sha1, "a value change must move the hash"


@pytest.mark.parametrize("attr", ["splits_manifest_sha256", "cycle_source_sha256", "cycle_epoch"])
def test_norm_refuses_store_with_unbound_provenance(tmp_path, attr):
    root = make_store(tmp_path, field_ids=(301,))
    fill_cycle(root, 0, 0, value=0.3)
    splits, holdout = bind_patch_store(root, spatial=(101,), known=(101, 301))
    del root.attrs[attr]
    with pytest.raises(ValueError, match="re-extract to a new store"):
        P.build_patch_norm(root=root, splits_manifest=splits, holdout_manifest=holdout)


def test_norm_refuses_out_of_cycle_count_before_reading_any_values(tmp_path, monkeypatch):
    root = make_store(tmp_path, field_ids=(301,))
    fill_cycle(root, 0, 0, value=0.3)
    root["cycle_end_day"][0, 0] = 14
    monkeypatch.setattr(P, "read_patch_observations", lambda *_: pytest.fail("premature value read"))
    with pytest.raises(HoldoutViolationError, match=r"301:100 has acquisition counts after its end"):
        fit(root)


def test_norm_ignores_finite_values_under_invalid_pixel_mask(tmp_path):
    root = make_store(tmp_path, field_ids=(301,))
    fill_cycle(root, 0, 0, value=0.3)
    root["values"][0, 0, :, :, :, :6] = 50
    root["pixel_valid"][0, 0, :, 0, :, :6] = 0
    norm = fit(root)
    assert norm["stats"]["L8"]["ndvi"]["median"] == pytest.approx(0.3, abs=0.001)


@pytest.mark.parametrize("stats", [{}, {"L8": {}}, {"L8": {"ndvi": {"med": 0.3, "iqr": 0.2}}}])
def test_load_refuses_malformed_envelope_even_with_a_valid_hash(tmp_path, stats):
    norm = {"stats": stats, "channel_order": list(P.CHANNELS)}
    norm["sha256"] = P.norm_stats_sha256(norm)
    path = tmp_path / "norm_stats.json"
    path.write_text(json.dumps(norm))
    with pytest.raises(ValueError, match="invalid patch normalization"):
        P.load_patch_norm(path)


# ---------------------------------------------------------------------------
# PERF-1 — output_sha256 streams the store instead of materializing it
# ---------------------------------------------------------------------------
def _reference_output_sha256(root, *, sample: int = 512) -> str:
    """Pre-PERF-1 body, verbatim: materialize ``values`` and hash the sample.

    Kept as the parity oracle; this is the code the streaming version
    replaced, so the store digest cannot drift silently.
    """
    import hashlib

    h = hashlib.sha256()
    for name in sorted(P.store_schema(1, 1)):
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


def _filled_store(tmp_path: Path, *, field_ids=(301, 302), n_cycles=3, size=8,
                  nan_fraction: float = 0.3, seed: int = 0):
    root = make_store(tmp_path, field_ids=field_ids, n_cycles=n_cycles, size=size)
    rng = np.random.default_rng(seed)
    for fi in range(len(field_ids)):
        for ci in range(n_cycles):
            fill_cycle(root, fi, ci, value=0.3)
            shape = (P.N_BINS, P.N_CHANNELS, size, size)
            vals = rng.standard_normal(shape).astype(np.float16)
            vals[rng.random(shape) < nan_fraction] = np.nan
            root["values"][fi, ci] = vals
    return root


@pytest.mark.parametrize("slab", [1, 2, 3, 5, 64])
def test_streamed_output_sha256_is_byte_identical_to_the_materialized_body(
    tmp_path: Path, slab: int,
) -> None:
    root = _filled_store(tmp_path)
    assert P.output_sha256(root, slab=slab) == _reference_output_sha256(root)


def test_streamed_output_sha256_is_identical_on_degenerate_stores(tmp_path: Path) -> None:
    """All-NaN (nothing to sample) and all-finite (every position a candidate)
    are the two ends the ordinal walk has to get right."""
    root = make_store(tmp_path / "all_nan", field_ids=(301,), n_cycles=2)
    for ci in range(2):
        root["values"][0, ci] = np.full(
            (P.N_BINS, P.N_CHANNELS, 8, 8), np.nan, dtype=np.float16)
    assert P.output_sha256(root) == _reference_output_sha256(root)
    assert P.output_sha256(root) == P.output_sha256(root), "must be deterministic"

    root2 = make_store(tmp_path / "all_finite", field_ids=(301,), n_cycles=2)
    rng = np.random.default_rng(3)
    for ci in range(2):
        root2["values"][0, ci] = rng.standard_normal(
            (P.N_BINS, P.N_CHANNELS, 8, 8)).astype(np.float16)
    assert P.output_sha256(root2, slab=1) == _reference_output_sha256(root2)
    assert P.output_sha256(root2, slab=2) == P.output_sha256(root2, slab=64)


def test_output_sha256_does_not_read_the_whole_values_array(tmp_path: Path) -> None:
    """The point of PERF-1: the function must never ask for every row at once."""
    store = _filled_store(tmp_path)
    n_rows = int(store["values"].shape[0])
    reads: list[int] = []

    class Guard:
        """Minimal read-only view over one zarr array, recording slice extents."""

        def __init__(self, arr):
            self._arr = arr
            self.shape = arr.shape
            self.dtype = arr.dtype

        def __getitem__(self, key):
            first = key[0] if isinstance(key, tuple) and key else key
            if isinstance(first, slice):
                lo, hi, _ = first.indices(self.shape[0])
                reads.append(hi - lo)
            else:                                   # pragma: no cover - never used
                reads.append(self.shape[0])
            return np.asarray(self._arr[key])

    class RootView(dict):
        pass

    view = RootView(store)
    view["values"] = Guard(store["values"])
    digest = P.output_sha256(view, slab=1)
    assert digest == _reference_output_sha256(store)
    assert reads, "the streaming path must read the store"
    assert max(reads) == 1, (
        f"largest slice was {max(reads)} of {n_rows} rows — the store was "
        "materialized instead of streamed"
    )


# ---------------------------------------------------------------------------
# PERF-9 — the read-only store handle and cycle index are memoized
# ---------------------------------------------------------------------------
def test_open_store_is_memoized_for_read_mode(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    fill_cycle(store, 0, 0, value=0.3)
    path = tmp_path / "patches.zarr"
    P.clear_store_caches()
    a = P.open_store(path)
    b = P.open_store(path)
    assert a is b, "read-only opens of the same store must share one handle"
    # a write-mode open is never cached
    c = P.open_store(path, mode="a")
    assert c is not a
    assert P.open_store(path) is a, "the read handle must survive a write open"


def test_cycle_index_is_memoized_but_never_stale(tmp_path: Path) -> None:
    """A cached index is reused, yet an in-place metadata change invalidates it."""
    store = make_store(tmp_path, field_ids=(301, 302), n_cycles=2)
    fill_cycle(store, 0, 0, value=0.3)
    fill_cycle(store, 0, 1, value=0.3)
    P.clear_store_caches()
    first = P.patch_cycle_index(store)
    again = P.patch_cycle_index(store)
    assert again == first
    assert again is first, "an unchanged store must hit the cache"

    # change the metadata in place: the index must follow, not go stale
    store["cycle_ids"][0, 1] = 999
    updated = P.patch_cycle_index(store)
    assert "301:999" in updated
    assert "301:101" not in updated

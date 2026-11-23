"""PERF-6 parity: the two-pass build_patch_norm is bit-identical to the old one.

Audit PERF-6 rewrote ``build_patch_norm`` to pre-allocate its accumulation
arrays instead of growing a list of float64 blocks and concatenating it (which
held every value twice at peak). The frozen norm v2 artifact must stay
bit-for-bit reproducible, and the accumulation dtype must NOT change, so this
test pins the new body against the old body verbatim on a synthetic store.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tests.patch_fixtures import bind_patch_store  # noqa: E402
from tests.unit.test_patch_store import fill_cycle, make_store  # noqa: E402

from ssl_anomaly.io import patches as P  # noqa: E402

zarr = pytest.importorskip("zarr")


def _reference_build_patch_norm(*, root, splits_manifest, holdout_manifest):
    """The pre-PERF-6 body, verbatim: accumulate list, then concatenate."""
    cycles, _ = P.select_patch_train_cycles(
        root, splits_manifest=splits_manifest, holdout_manifest=holdout_manifest,
    )
    flat_vals, flat_sensors = [], []
    for cycle in cycles:
        values, obs, sensors, _ = P.read_patch_observations(root, cycle)
        for b, sensor in enumerate(sensors):
            pixels = obs[b].any(axis=0)
            if pixels.any():
                flat_vals.append(np.where(obs[b][:, pixels], values[b][:, pixels], np.nan).T)
                flat_sensors.append(np.full(pixels.sum(), sensor, dtype=np.uint8))
    if not flat_vals:
        raise ValueError("no finite in-field train pixel observations — refusing to fit norm stats")
    stats = P.fit_patch_norm(np.concatenate(flat_vals), np.concatenate(flat_sensors))
    return stats


@pytest.fixture
def synthetic_store(tmp_path):
    """A small bound store with varied per-bin footprints and NaNs."""
    size = 6
    field_ids = (301, 302, 101)
    root = make_store(tmp_path, field_ids=field_ids, n_cycles=3, size=size)
    rng = np.random.default_rng(7)
    for fi in range(len(field_ids)):
        for ci in range(3):
            fill_cycle(root, fi, ci, value=0.3, sensor=(1, 2, 0)[ci % 3])
            blocks = rng.normal(0.3, 0.15, (P.N_BINS, P.N_CHANNELS, size, size))
            # a ragged in-field footprint per cycle: different pixel counts
            footprint = np.zeros((size, size), dtype=bool)
            footprint[: 2 + ci, : 3 + fi] = True
            root["in_field"][fi, 0] = footprint.astype(np.uint8)
            blocks[:, :, ~footprint] = np.nan
            blocks[0, :, footprint] = np.nan          # a bin with nothing observed
            root["values"][fi, ci] = blocks.astype(np.float16)
            root["n_obs"][fi, ci] = 3
            root["sensor_id"][fi, ci] = (1, 2, 0)[ci % 3]
            root["cycle_valid"][fi, ci] = 1
            root["cycle_ids"][fi, ci] = 100 + ci
            root["cycle_start_day"][fi, ci] = 0
            root["cycle_end_day"][fi, ci] = P.MAX_CYCLE_DAYS - 1
    return root


def test_two_pass_accumulation_is_bit_identical(synthetic_store) -> None:
    root = synthetic_store
    splits, holdout = bind_patch_store(
        root, spatial=(101,), known=(101, 301, 302))
    got = P.build_patch_norm(root=root, splits_manifest=splits, holdout_manifest=holdout)
    ref = _reference_build_patch_norm(
        root=root, splits_manifest=splits, holdout_manifest=holdout)
    assert got["stats"].keys() == ref.keys()
    for sensor in ref:
        for name, block in ref[sensor].items():
            for key in ("median", "iqr", "n"):
                g, r = got["stats"][sensor][name][key], block[key]
                if isinstance(r, float):
                    # EXACT, not approx: the two-pass fill writes the same
                    # float64 values in the same row order.
                    assert g == r, f"{sensor}/{name}/{key}: {g!r} != {r!r}"
                else:
                    assert g == r
    # Provenance the two paths must agree on (the reference returns stats
    # only, so the cycle selection is re-derived and compared here).
    assert got["n_train_cycles"] == len(
        P.select_patch_train_cycles(
            root, splits_manifest=splits, holdout_manifest=holdout)[0])
    assert got["train_cycle_keys"] == sorted(
        c.key for c in P.select_patch_train_cycles(
            root, splits_manifest=splits, holdout_manifest=holdout)[0])


def test_accumulation_dtype_stays_float64(synthetic_store, monkeypatch) -> None:
    """PERF-6 must not change numerics: the fit still receives float64."""
    root = synthetic_store
    splits, holdout = bind_patch_store(
        root, spatial=(101,), known=(101, 301, 302))
    seen: list[np.dtype] = []
    real_fit = P.fit_patch_norm

    def spy(values, sensor_ids, **kw):
        seen.append(np.asarray(values).dtype)
        seen.append(np.asarray(sensor_ids).dtype)
        return real_fit(values, sensor_ids, **kw)

    monkeypatch.setattr(P, "fit_patch_norm", spy)
    P.build_patch_norm(root=root, splits_manifest=splits, holdout_manifest=holdout)
    assert seen == [np.dtype(np.float64), np.dtype(np.uint8)]

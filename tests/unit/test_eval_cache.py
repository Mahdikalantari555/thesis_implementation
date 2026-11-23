"""Opt-in eval artifact cache: mechanics, fail-open, and cold/warm parity.

L0 infrastructure tests. The cache never creates a research value: every
assertion here checks that a warm run reproduces the cold run's bytes (with a
NaN-aware comparator) or that a broken cache falls back without changing a
result. Default runs must not touch the cache at all.
"""

from __future__ import annotations

import sys
import types

# lightning is absent in the local geospatial env; importing patch_eval pulls
# in ssl_anomaly.train, which imports it. Stub exactly what import needs, as
# src/_fa_bench_driver.py already does. No training code runs in this module.
if "lightning" not in sys.modules:
    try:
        import lightning  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("lightning")
        _stub.LightningModule = type("LightningModule", (), {})
        sys.modules["lightning"] = _stub

import numpy as np
import pytest
from tests.patch_fixtures import bind_patch_store

from ssl_anomaly.io import patches as P
from ssl_anomaly.monitor import eval_cache as EC
from ssl_anomaly.monitor import patch_eval as PE
from ssl_anomaly.monitor import patch_scores as PS
from ssl_anomaly.monitor import severity_table as sev

T, C, S = P.N_BINS, P.N_CHANNELS, 8


def _eq(a, b) -> bool:
    """Array equality where NaN in the same position counts as equal."""
    return bool(np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True))


def _norm(median=0.4, iqr=0.05):
    per = {name: {"median": median, "iqr": iqr, "n": 100} for name in P.CHANNELS}
    return {"stats": {s: per for s in ("L8", "L9", "unknown")},
            "channel_order": list(P.CHANNELS)}


def _batch(n=4, seed=0):
    g = np.random.default_rng(seed)
    raw = g.uniform(0.1, 0.7, size=(n, T, C, S, S)).astype(np.float32)
    obs = g.random(raw.shape) > 0.2
    stats = P.patch_norm_stats(_norm(0.0, 1.0))
    X = np.stack([P.normalize_patch(raw[i], obs[i], np.ones((T,), dtype=np.uint8), stats)[0]
                  for i in range(n)])
    return {
        "raw": raw, "obs": obs, "X": X,
        "in_field": np.ones((n, S, S), bool),
        "sensor_id": np.ones((n, T), dtype=np.uint8),
        "pixel_valid": obs.any(axis=2),
        "norm_stats": stats,
        "covariates": g.normal(size=(n, T, 3)).astype(np.float32),
    }


class _StubScorer:
    """Deterministic loo_batch; no torch, no model checkpoint needed."""

    model_id = "cache-test-scorer"
    alignment = "test-alignment"

    class _Sigma:
        sha256 = "cache-test-sigma"

    sigma = _Sigma()

    def loo_batch(self, X, M, **kw):
        v = np.asarray(X, dtype=np.float64)[:, :, 0]
        v = np.where(np.isfinite(v), v, 0.0)
        scores = v.mean(axis=(2, 3))
        pred = np.zeros((*X.shape[:2], X.shape[2]), dtype=np.float32)
        return scores, pred


def _arm():
    return PS.PatchArmScorer(_StubScorer(), np.ones((12, T)))


# --- hashing ---------------------------------------------------------------
def test_fingerprint_is_deterministic_and_content_sensitive():
    a = np.array([1.0, 2.0], dtype=np.float32)
    assert EC.fingerprint(a, "x", 1) == EC.fingerprint(a.copy(), "x", 1)
    b = a.copy()
    b[0] = 9.0
    assert EC.fingerprint(a, "x", 1) != EC.fingerprint(b, "x", 1)
    assert EC.fingerprint(a, "x", 1) != EC.fingerprint(a, "y", 1)


def test_fingerprint_normalizes_nan_payloads_but_keeps_nan_positions():
    a = np.array([np.nan, 1.0], dtype=np.float32)
    b = a.copy()
    b.view(np.uint32)[0] ^= 0x0000_0001       # same NaN semantics, different bits
    assert EC.fingerprint(a) == EC.fingerprint(b)
    c = np.array([1.0, np.nan], dtype=np.float32)
    assert EC.fingerprint(a) != EC.fingerprint(c)


def test_batch_fingerprint_is_memoized_on_the_batch():
    batch = _batch()
    first = EC.batch_fingerprint(batch)
    assert EC._BATCH_FP_KEY in batch
    batch["unused"] = object()               # would break canonical JSON if re-hashed
    assert EC.batch_fingerprint(batch) == first


# --- mechanics -------------------------------------------------------------
def test_atomic_roundtrip_and_fingerprint_binding(tmp_path):
    cache = EC.EvalArtifactCache(tmp_path, read=True, write=True)
    key = EC.fingerprint("l2", "entry")
    arr = np.array([[1.0, np.nan], [2.0, 3.0]])
    assert cache.save_l2(key, arr) is True
    assert _eq(cache.load_l2(key), arr)
    # an entry stored under one key must never be readable under another
    assert cache.load_l2(EC.fingerprint("l2", "other")) is None


def test_corrupt_or_missing_entry_fails_open(tmp_path):
    cache = EC.EvalArtifactCache(tmp_path, read=True, write=True)
    key = EC.fingerprint("corrupt")
    cache.save_l2(key, np.array([1.0]))
    (tmp_path / "l2" / key / "meta.json").write_text("not json", encoding="utf-8")
    assert cache.load_l2(key) is None
    assert cache.stats["invalidated"] >= 1
    assert cache.stats["error"] == 0
    assert cache.load_l2(EC.fingerprint("never written")) is None


def test_size_cap_refuses_an_oversized_cell(tmp_path):
    cache = EC.EvalArtifactCache(tmp_path, read=True, write=True, max_cell_bytes=8)
    key = EC.fingerprint("big")
    assert cache.save_l2(key, np.ones(100, dtype=np.float64)) is False
    assert cache.stats["skip_size"] == 1
    assert not (tmp_path / "l2" / key).exists()
    assert cache.load_l2(key) is None


def test_env_default_is_off_and_read_write_are_separate(tmp_path, monkeypatch):
    for name in (EC.ENV_READ, EC.ENV_WRITE):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(EC.ENV_DIR, str(tmp_path / "cache"))
    assert EC.resolve_cache(None) is None
    assert not (tmp_path / "cache").exists()
    assert EC.resolve_cache(False) is None

    monkeypatch.setenv(EC.ENV_READ, "1")
    handle = EC.resolve_cache(None)
    assert handle is not None and handle.read and not handle.write
    key = EC.fingerprint("read-only")
    assert handle.save_l2(key, np.ones(3)) is False      # write flag off
    assert not (tmp_path / "cache").exists()

    monkeypatch.setenv(EC.ENV_WRITE, "1")
    handle = EC.resolve_cache(None)
    assert handle.write and handle.save_l2(key, np.ones(3)) is True
    assert (tmp_path / "cache" / "l2" / key / "data.npz").exists()


# --- L3: injected tensor parity -------------------------------------------
def test_l3_warm_run_reproduces_cold_run_exactly(tmp_path):
    cache = EC.EvalArtifactCache(tmp_path, read=True, write=True)
    arm = _arm()
    batch = _batch()
    args = dict(level_id="S3", family=sev.PERSISTENT_FAMILY, index="ndvi",
                round_idx=1, injection_basis="bands_consistent", chunk=2)

    cold = PE.patch_arm_scores(arm, batch, cache=False, **args)
    # first enabled run builds the entry; second is a pure warm read
    PE.patch_arm_scores(arm, batch, cache=cache, **args)
    warm = PE.patch_arm_scores(arm, batch, cache=cache, **args)
    assert cache.stats["hit"] >= 1

    for field in ("z2", "paths", "p_value", "observed"):
        assert _eq(cold[field], warm[field]), field
    assert cold["injections"] == warm["injections"]
    assert cold["injection_basis"] == warm["injection_basis"]


def test_l3_cache_is_scoped_to_the_cell_and_changes_with_inputs(tmp_path):
    cache = EC.EvalArtifactCache(tmp_path, read=True, write=True)
    arm = _arm()
    base = dict(family=sev.ACUTE_FAMILY, index="ndvi", round_idx=0,
                injection_basis="bands_consistent")
    PE.patch_arm_scores(arm, _batch(), cache=cache,
                        level_id="S3", **base)
    # a different severity level is a different key, not a stale cache hit
    PE.patch_arm_scores(arm, _batch(), cache=cache,
                        level_id="S6", **base)
    dirs = [p.name for p in (tmp_path / "l3").iterdir()]
    assert len(dirs) == 2


@pytest.mark.parametrize("level", ["S1", "S2", "S3", "S4", "S5", "S6"])
@pytest.mark.parametrize("family", [sev.ACUTE_FAMILY, sev.PERSISTENT_FAMILY,
                                    sev.GRADUAL_FAMILY])
@pytest.mark.parametrize("index", ["ndvi", "savi"])
@pytest.mark.parametrize("round_idx", [0, 1])
def test_l3_parity_for_every_severity_cell(tmp_path, level, family, index, round_idx):
    """The parity gate covers all S1-S6 x family x index x round cells.

    Synthetic scale (n=2) is enough to prove the cache mechanism is cell-safe;
    parity at production n is a separate archive-machine gate (see tasks.md).
    """
    cache = EC.EvalArtifactCache(tmp_path, read=True, write=True)
    arm = _arm()
    batch = _batch(n=2, seed=11)
    args = dict(level_id=level, family=family, index=index, round_idx=round_idx,
                injection_basis="bands_consistent", chunk=1)
    cold = PE.patch_arm_scores(arm, batch, cache=False, **args)
    PE.patch_arm_scores(arm, batch, cache=cache, **args)   # build
    warm = PE.patch_arm_scores(arm, batch, cache=cache, **args)
    for field in ("z2", "paths", "p_value", "observed"):
        assert _eq(cold[field], warm[field]), field
    assert cold["injections"] == warm["injections"]


def test_l3_custom_rng_bypasses_the_cache(tmp_path):
    cache = EC.EvalArtifactCache(tmp_path, read=True, write=True)
    arm = _arm()
    args = dict(level_id="S3", family=sev.ACUTE_FAMILY, index="ndvi",
                round_idx=0, injection_basis="bands_consistent")
    PE.patch_arm_scores(arm, _batch(), cache=cache, rng=np.random.default_rng(7), **args)
    assert not (tmp_path / "l3").exists()


# --- L2: clean reference paths parity -------------------------------------
def test_l2_warm_reference_paths_match_cold(tmp_path):
    cache = EC.EvalArtifactCache(tmp_path, read=True, write=True)
    scorer = _StubScorer()
    batch = _batch(n=16)
    kw = dict(reference_idx=np.arange(12), field_row=np.arange(16), chunk=4)
    cold = PE.patch_reference_paths(scorer, batch, cache=False, **kw)
    PE.patch_reference_paths(scorer, batch, cache=cache, **kw)
    warm = PE.patch_reference_paths(scorer, batch, cache=cache, **kw)
    assert _eq(cold, warm)
    assert cache.stats["hit"] >= 1


# --- L1: normalized batch from a frozen store ------------------------------
def _store(tmp_path):
    g = np.random.default_rng(3)
    root = P.create_store(tmp_path / "patches.zarr", field_ids=[11, 22],
                          max_cycles=2, size=S)
    root["field_ids"][:] = np.asarray([11, 22], dtype=np.int64)
    for fi in range(2):
        root["in_field"][fi, 0] = 1
        for ci in range(2):
            root["cycle_valid"][fi, ci] = 1
            root["cycle_ids"][fi, ci] = ci
            root["cycle_start_day"][fi, ci] = 10 + ci
            root["cycle_end_day"][fi, ci] = 10 + ci + P.MAX_CYCLE_DAYS - 1
            root["sensor_id"][fi, ci] = 1
            root["n_obs"][fi, ci] = 2
            vals = g.normal(0.4, 0.05, size=(T, C, S, S)).astype(np.float32)
            root["values"][fi, ci] = vals
            root["pixel_valid"][fi, ci, :, 0] = 1
    bind_patch_store(root, known=(11, 22))
    return root


def test_l1_warm_batch_matches_cold_roundtrip(tmp_path):
    root = _store(tmp_path)
    cache = EC.EvalArtifactCache(tmp_path / "cache", read=True, write=True)
    ids, cids = [11, 22], [0, 1]
    norm = _norm()

    cold = PE.patch_batch_for_cycles(root, ids, cids, norm=norm, cache=False)
    PE.patch_batch_for_cycles(root, ids, cids, norm=norm, cache=cache)   # build
    warm = PE.patch_batch_for_cycles(root, ids, cids, norm=norm, cache=cache)
    assert cache.stats["hit"] >= 1
    for name in ("X", "raw", "obs", "in_field", "sensor_id", "covariates", "pixel_valid"):
        assert _eq(cold[name], warm[name]), name
    assert cold["unnormalized_channels"] == warm["unnormalized_channels"]
    assert cold["cycles_found"] == warm["cycles_found"]
    assert cold["norm_stats"] == warm["norm_stats"]


def test_l1_key_changes_when_norm_changes(tmp_path):
    root = _store(tmp_path)
    k1 = EC.l1_key(root, [11], [0], _norm(0.4, 0.05))
    k2 = EC.l1_key(root, [11], [0], _norm(0.5, 0.05))
    assert k1 != k2

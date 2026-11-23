"""Byte-parity pins for the vectorized patch-scoring functions (audit PERF-3/4).

Every rewrite keeps a reference twin implementing the pre-optimization loop
verbatim, and the tests assert EXACT equality (positions and values, NaN and
±inf included) on production-like shapes — N cycles × T=36 bins × C=8 channels
× S=32 pixels — plus the degenerate inputs the audit names.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.monitor import patch_scores as PS

N, T, C, S = 24, 36, 8, 32


# ---------------------------------------------------------------------------
# Reference implementations (pre-PERF-3 / pre-PERF-4 bodies, verbatim)
# ---------------------------------------------------------------------------
def _reference_lookup(table, bins, sids):
    G, Cc, _ = table.sigma.shape
    groups = np.clip(np.asarray(bins, dtype=np.int64) // PS.PHASE_BIN, 0, G - 1)
    sids = np.asarray(sids, dtype=np.int64)
    out = np.full((len(bins), Cc), np.nan)
    for t, (g, s) in enumerate(zip(groups, sids)):
        s_idx = int(s) if 0 <= int(s) < table.sigma.shape[2] else 0
        out[t] = table.sigma[g, :, s_idx]
    bad = ~np.isfinite(out) | (out <= 0)
    if bad.any():
        out = np.where(bad, np.broadcast_to(table.fallback, out.shape), out)
    return out


def _reference_pixel_scores(resid, obs, sigma, bins, sensor_ids):
    r = np.asarray(resid, dtype=np.float64)
    o = np.asarray(obs, dtype=bool)
    Nn, Tt, Cc, Ss, _ = r.shape
    out = np.full((Nn, Tt, Ss, Ss), np.nan)
    for n in range(Nn):
        scale = _reference_lookup(sigma, bins, np.asarray(sensor_ids[n]))
        z2 = np.where(o[n], (r[n] / scale[:, :, None, None]) ** 2, -np.inf)
        with np.errstate(invalid="ignore"):
            s = z2.max(axis=1)
        s[~o[n].any(axis=1)] = np.nan
        out[n] = s
    return out


def _reference_field_paths(pixel_z2, in_field, *, top_frac=PS.TOP_FRAC, min_pixels=4):
    p = np.asarray(pixel_z2, dtype=np.float64)
    Nn, Tt = p.shape[:2]
    inf = np.asarray(in_field, dtype=bool)
    if inf.ndim == 2:
        masks = np.broadcast_to(inf[None, None], (Nn, Tt) + inf.shape)
    else:
        masks = np.broadcast_to(inf[:, None], (Nn, Tt) + inf.shape[1:])
    out = np.full((Nn, Tt), np.nan)
    for n in range(Nn):
        for t in range(Tt):
            vals = p[n, t][masks[n, t]]
            vals = vals[np.isfinite(vals)]
            if vals.size < min_pixels:
                continue
            k = min(max(1, int(np.ceil(top_frac * vals.size))), vals.size)
            out[n, t] = float(np.mean(np.partition(vals, vals.size - k)[vals.size - k:]))
    return out


def _assert_exact(got, ref, what):
    assert got.shape == ref.shape, f"{what}: shape {got.shape} != {ref.shape}"
    assert np.array_equal(np.isnan(got), np.isnan(ref)), f"{what}: NaN positions differ"
    assert np.array_equal(np.isposinf(got), np.isposinf(ref)), f"{what}: +inf differs"
    assert np.array_equal(np.isneginf(got), np.isneginf(ref)), f"{what}: -inf differs"
    fin = np.isfinite(ref)
    assert np.array_equal(got[fin], ref[fin]), (
        f"{what}: values differ at {int((got[fin] != ref[fin]).sum())} positions"
    )


# ---------------------------------------------------------------------------
# PERF-3: sigma lookup / pixel_scores
# ---------------------------------------------------------------------------
def _table(seed=0, min_n=20):
    rng = np.random.default_rng(seed)
    res = rng.normal(0.0, 1.0, size=(6, T, C, 8, 8))
    obs = rng.random((6, T, C, 8, 8)) > 0.3
    sids = rng.integers(0, 3, size=(6, T))
    return PS.fit_patch_sigma(res, obs, sids, min_n=min_n)


def test_batch_lookup_matches_the_per_sample_loop_exactly():
    table = _table()
    sids = np.random.default_rng(1).integers(0, 3, size=(N, T))
    got = table.batch_lookup(sids)
    ref = np.stack([_reference_lookup(table, np.arange(T), s) for s in sids])
    _assert_exact(got, ref, "batch_lookup")


def test_batch_lookup_handles_undefined_and_degenerate_cells():
    """An under-sampled table (every fitted cell NaN) must fall back per
    channel exactly as the loop does; an all-NaN fallback stays 1.0."""
    table = _table(seed=2, min_n=10_000)
    sids = np.random.default_rng(3).integers(0, 3, size=(5, T))
    got = table.batch_lookup(sids)
    ref = np.stack([_reference_lookup(table, np.arange(T), s) for s in sids])
    _assert_exact(got, ref, "batch_lookup(under-sampled)")


def test_pixel_scores_matches_the_sample_loop_bitwise():
    table = _table(seed=4)
    rng = np.random.default_rng(5)
    resid = rng.normal(0.0, 2.0, size=(6, T, C, 12, 12))
    obs = rng.random((6, T, C, 12, 12)) > 0.35
    resid = np.where(obs, resid, 0.0)
    sids = rng.integers(0, 3, size=(6, T))
    bins = np.arange(T)
    got = PS.pixel_scores(resid, obs, table, bins, sids)
    ref = _reference_pixel_scores(resid, obs, table, bins, sids)
    _assert_exact(got, ref, "pixel_scores")


def test_pixel_scores_degenerate_columns_and_bins():
    table = _table(seed=6)
    rng = np.random.default_rng(7)
    resid = rng.normal(0.0, 1.0, size=(4, T, C, 8, 8))
    obs = rng.random((4, T, C, 8, 8)) > 0.25
    # channel 1 fully unobserved everywhere; bin 5 fully unobserved
    obs[:, :, 1] = False
    obs[:, 5] = False
    resid = np.where(obs, resid, 0.0)
    sids = np.zeros((4, T), dtype=np.int64)
    got = PS.pixel_scores(resid, obs, table, np.arange(T), sids)
    ref = _reference_pixel_scores(resid, obs, table, np.arange(T), sids)
    _assert_exact(got, ref, "pixel_scores(degenerate)")
    assert np.isnan(got[:, 5]).all()


# ---------------------------------------------------------------------------
# PERF-4: field_paths
# ---------------------------------------------------------------------------
def test_field_paths_matches_the_per_cell_loop_bitwise():
    rng = np.random.default_rng(8)
    pz = rng.random((N, T, S, S)) * 10.0
    in_field = np.zeros((S, S), dtype=bool)
    in_field[4:28, 6:30] = True
    got = PS.field_paths(pz, in_field)
    ref = _reference_field_paths(pz, in_field)
    _assert_exact(got, ref, "field_paths")


@pytest.mark.parametrize("top_frac", [0.2, 0.05, 0.5, 1.0, 1.5])
def test_field_paths_top_fraction_edges(top_frac):
    rng = np.random.default_rng(9)
    pz = rng.random((6, 12, S, S))
    inf = np.ones((S, S), dtype=bool)
    got = PS.field_paths(pz, inf, top_frac=top_frac)
    ref = _reference_field_paths(pz, inf, top_frac=top_frac)
    _assert_exact(got, ref, f"field_paths(top_frac={top_frac})")


def test_field_paths_degenerate_cells():
    """NaN inside observed pixels, all-NaN bins, a bin below min_pixels, an
    empty in-field mask, and a per-sample mask stack."""
    rng = np.random.default_rng(10)
    pz = rng.random((5, 10, S, S))
    pz[0, 3, 0, 0] = np.nan               # a NaN inside an otherwise observed bin
    pz[2, 4] = np.nan                     # a fully unobserved bin
    infield = np.ones((S, S), dtype=bool)
    got = PS.field_paths(pz, infield)
    ref = _reference_field_paths(pz, infield)
    _assert_exact(got, ref, "field_paths(NaN)")

    small = np.zeros((S, S), dtype=bool)
    small[0, :2] = True                   # 2 in-field pixels < min_pixels
    _assert_exact(PS.field_paths(pz, small), _reference_field_paths(pz, small),
                  "field_paths(under-sampled)")

    per_sample = np.zeros((5, S, S), dtype=bool)
    for i in range(5):
        per_sample[i, : 8 + 2 * i, : 8 + 2 * i] = True
    _assert_exact(PS.field_paths(pz, per_sample),
                  _reference_field_paths(pz, per_sample), "field_paths(per-sample)")


def test_field_paths_is_faster_than_the_loop() -> None:
    """PERF-4 exists to remove the 12k-iteration Python loop. Production-like
    extent (340 cycles × 36 bins) with a realistic in-field footprint."""
    import time

    rng = np.random.default_rng(11)
    pz = rng.random((340, 36, S, S)) * 5.0
    inf = np.zeros((S, S), dtype=bool)
    inf[8:24, 8:24] = True                 # 256 in-field pixels, like the store
    pz[:, :, ~inf] = np.nan                # the scorer NaNs out-of-field pixels
    t0 = time.perf_counter()
    ref = _reference_field_paths(pz, inf)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    got = PS.field_paths(pz, inf)
    t_new = time.perf_counter() - t0
    assert np.array_equal(got, ref)
    # The constant-m fast path makes this ~3x on this shape; 1.5x is a
    # conservative bound that still fails if the vectorization regresses to
    # a per-cell loop.
    assert t_new < t_ref / 1.5, f"vectorized {t_new:.3f}s vs loop {t_ref:.3f}s"

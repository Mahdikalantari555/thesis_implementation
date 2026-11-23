"""Byte-parity pins for the audit's vectorized scoring functions.

Audit 2026-11-21 mandated every performance rewrite of ``monitor/scoring.py``
to be accompanied by a test that asserts EXACT equality (positions AND values,
NaN included) against the pre-rewrite implementation, on production-like and
degenerate randomized inputs. Each production function keeps its reference
twin next to it (``*_reference``) or inline here, and the tests below are the
only thing allowed to call it.

Production-like shapes used throughout: N≈340 cycles × T=36 bins × C=8
channels — the locked v2 dev panel's dimensionality.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.monitor import scoring as S

N, T, C = 340, 36, 8


# ---------------------------------------------------------------------------
# Input builders: production-like and the degenerate cases the audit names
# ---------------------------------------------------------------------------
def _production_like(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(resid, mask) with ~70 % observed, field offsets and NaN-free residuals."""
    rng = np.random.default_rng(seed)
    resid = rng.normal(0.0, 1.0, size=(N, T, C))
    # Field-level offsets: the null tail this function exists to remove.
    resid += rng.normal(0.0, 2.0, size=(N, 1, C))
    mask = rng.random((N, T, C)) > 0.30
    resid = np.where(mask, resid, 0.0)
    return resid, mask


def _reference_cell_q95(z2_clean, mask, *, min_cells: int = 8) -> np.ndarray:
    """The pre-PERF-5 (channel, bin) loop, verbatim."""
    Tt, Cc = z2_clean.shape[1], z2_clean.shape[2]
    q = np.full((Tt, Cc), np.nan)
    for c in range(Cc):
        for b in range(Tt):
            vals = z2_clean[:, b, c][mask[:, b, c]]
            if vals.size >= min_cells:
                q[b, c] = float(np.percentile(vals, 95))
        med = np.nanmedian(q[:, c])
        if np.isfinite(med) and med > 0:
            col = q[:, c]
            col[~np.isfinite(col) | (col <= 0)] = med
            q[:, c] = np.maximum(col, med)
        else:
            q[:, c] = 1.0
    return q


def _assert_exact(got: np.ndarray, ref: np.ndarray, what: str) -> None:
    assert got.shape == ref.shape, f"{what}: shape {got.shape} != {ref.shape}"
    assert np.array_equal(np.isnan(got), np.isnan(ref)), f"{what}: NaN positions differ"
    assert np.array_equal(np.isposinf(got), np.isposinf(ref)), f"{what}: +inf positions differ"
    assert np.array_equal(np.isneginf(got), np.isneginf(ref)), f"{what}: -inf positions differ"
    fin = np.isfinite(ref)
    assert np.array_equal(got[fin], ref[fin]), f"{what}: values differ at {int((got[fin] != ref[fin]).sum())} positions"


# ---------------------------------------------------------------------------
# PERF-2: causal_offset_correct
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("window", [6, 1, 2, 12, 0, -3, T, T + 5])
def test_causal_offset_correct_matches_the_loop_bitwise(window: int) -> None:
    resid, mask = _production_like(0)
    got = S.causal_offset_correct(resid, mask, window=window)
    ref = S.causal_offset_correct_reference(resid, mask, window=window)
    _assert_exact(got, ref, f"causal_offset_correct(window={window})")


@pytest.mark.parametrize("seed", range(3))
def test_causal_offset_correct_is_exact_on_random_ragged_masks(seed: int) -> None:
    """Random observation density per bin AND per channel — windows are ragged."""
    rng = np.random.default_rng(100 + seed)
    resid = rng.normal(0.0, 3.0, size=(60, T, C))
    density = rng.uniform(0.0, 1.0, size=(60, 1, C))       # per (cycle, channel)
    mask = rng.random((60, T, C)) < density
    resid = np.where(mask, resid, np.nan)
    got = S.causal_offset_correct(resid, mask)
    ref = S.causal_offset_correct_reference(resid, mask)
    _assert_exact(got, ref, "causal_offset_correct(ragged)")


def test_causal_offset_correct_degenerate_inputs_and_nan_semantics() -> None:
    """All-unobserved columns, a single observed bin, and NaN inside an
    observed window (the audit's CRITICAL note: NaN propagates like the loop —
    it is NOT nanmedian)."""
    resid = np.zeros((3, 10, 3))
    mask = np.zeros((3, 10, 3), dtype=bool)
    # cycle 0: channel 0 fully unobserved; channel 1 a single observed bin;
    # channel 2 fully observed with a NaN in the middle.
    mask[0, :, 1] = False
    mask[0, 4, 1] = True
    mask[0, :, 2] = True
    resid[0, 3, 2] = np.nan
    # cycle 1: NaN in the first observed bin only (nothing to poison yet)
    mask[1, :, 0] = True
    resid[1, 0, 0] = np.nan
    # cycle 2: an infinite residual (an impossible z from a zero floor)
    mask[2, :, 0] = True
    resid[2, 5, 0] = np.inf
    resid[2, 6, 1] = -np.inf
    mask[2, 6, 1] = True

    got = S.causal_offset_correct(resid, mask)
    ref = S.causal_offset_correct_reference(resid, mask)
    _assert_exact(got, ref, "causal_offset_correct(degenerate)")

    # Semantics pinned literally: the NaN at (0, 3, 2) poisons every corrected
    # bin whose window [j-6, j) still contains it -- i.e. j = 4..9 -- while the
    # uncorrected first bin stays finite.
    assert np.isfinite(got[0, 0, 2])
    assert np.isnan(got[0, 4:10, 2]).all()
    # The single observed bin has no history → no correction.
    assert got[0, 4, 1] == resid[0, 4, 1]
    # Unobserved cells are returned untouched.
    assert np.array_equal(got[0, :, 0], resid[0, :, 0])


def test_causal_offset_correct_does_not_mutate_its_input() -> None:
    resid, mask = _production_like(7)
    before = resid.copy()
    S.causal_offset_correct(resid, mask)
    assert np.array_equal(resid, before)


def test_causal_offset_correct_is_actually_faster() -> None:
    """The rewrite must not regress: assert a comfortable wall-clock margin."""
    import time

    resid, mask = _production_like(3)
    t0 = time.perf_counter()
    S.causal_offset_correct_reference(resid, mask)
    t_ref = time.perf_counter() - t0
    t0 = time.perf_counter()
    S.causal_offset_correct(resid, mask)
    t_new = time.perf_counter() - t0
    assert t_new < t_ref / 2.0, f"vectorized {t_new:.3f}s vs reference {t_ref:.3f}s"


# ---------------------------------------------------------------------------
# PERF-5: cell_q95_normalizer
# ---------------------------------------------------------------------------
def test_cell_q95_normalizer_matches_the_loop_bitwise() -> None:
    rng = np.random.default_rng(4)
    z2 = rng.chisquare(3, size=(N, T, C))                  # squared z residuals
    mask = rng.random((N, T, C)) > 0.30
    z2 = np.where(mask, z2, 0.0)
    _assert_exact(S.cell_q95_normalizer(z2, mask), _reference_cell_q95(z2, mask),
                  "cell_q95_normalizer")


def test_cell_q95_normalizer_degenerate_columns_and_cells() -> None:
    """All-unobserved channels, under-sampled cells, and all-NaN columns."""
    rng = np.random.default_rng(5)
    z2 = np.abs(rng.normal(size=(40, 6, 3)))
    mask = np.ones((40, 6, 3), dtype=bool)
    mask[:, :, 0] = False                      # channel fully unobserved
    mask[:3, 2, 1] = False                     # cell with 3 < min_cells draws
    mask[:, 0, 1] = False                      # bin fully unobserved in ch 1
    z2 = np.where(mask, z2, 0.0)
    for min_cells in (8, 4, 1, 10_000):
        _assert_exact(S.cell_q95_normalizer(z2, mask, min_cells=min_cells),
                      _reference_cell_q95(z2, mask, min_cells=min_cells),
                      f"cell_q95_normalizer(min_cells={min_cells})")
    q = S.cell_q95_normalizer(z2, mask)
    # The unobserved channel takes the documented 1.0 fallback.
    assert np.array_equal(q[:, 0], np.ones(6))


def test_cell_q95_normalizer_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        S.cell_q95_normalizer(np.zeros((3, 4, 2)), np.ones((3, 4, 3), dtype=bool))


# ---------------------------------------------------------------------------
# Audit item 6: +inf stays alarm-worthy
# ---------------------------------------------------------------------------
def _reference_cumulative_channel_max(z2, mask, q95, *, window=3, min_history=3):
    """The pre-fix body, verbatim except for the S[isinf] = NaN line, which is
    kept here so the test can show the OLD behaviour was the defect."""
    Nn, Tt, Cc = z2.shape
    mask = mask & ~np.isnan(z2)
    zn = z2 / q95[None, :, :]
    zn = np.where(mask, zn, 0.0)
    mf = mask.astype(np.float64)
    csum = np.cumsum(zn, axis=1)
    ccnt = np.cumsum(mf, axis=1)
    lag = min(window, Tt - 1)
    trailing_sum = csum - np.concatenate(
        [np.zeros((Nn, lag, Cc)), csum[:, :-lag, :]], axis=1,
    ) if lag > 0 else csum
    trailing_cnt = ccnt - np.concatenate(
        [np.zeros((Nn, lag, Cc)), ccnt[:, :-lag, :]], axis=1,
    ) if lag > 0 else ccnt
    with np.errstate(invalid="ignore", divide="ignore"):
        trailing_mean = trailing_sum / np.maximum(trailing_cnt, 1e-9)
    trailing_mean = np.where(trailing_cnt > 0, trailing_mean, -np.inf)
    with np.errstate(invalid="ignore"):
        out = trailing_mean.max(axis=-1)
    out[np.isinf(out)] = np.nan
    total_obs = np.cumsum(mask.any(axis=-1).astype(np.int64), axis=1)
    out[total_obs < min_history] = np.nan
    return out


def test_plus_inf_does_not_kill_the_alarm_path() -> None:
    """Regression (audit item 6): a +inf z² used to become NaN in the trailing
    difference and be erased by ``S[isinf] = NaN``, so the loudest possible
    evidence produced NO alarm."""
    z2 = np.ones((1, 10, 2))
    mask = np.ones((1, 10, 2), dtype=bool)
    z2[0, 4, 0] = np.inf
    q95 = np.ones((10, 2))
    got = S.cumulative_channel_max(z2, mask, q95)
    old = _reference_cumulative_channel_max(z2, mask, q95)
    assert np.isnan(old[0, 4:7]).all(), "the pre-fix path should have died here"
    assert np.isfinite(got[0, 4:]).all(), "the alarm path must survive +inf"
    # The inf's own trailing window (bins 4..6, window=3) carries the alarm…
    assert np.all(got[0, 4:7] > 1e100), "and must carry alarm-worthy magnitude"
    # …and the path relaxes back to the null level once the window passes it.
    assert np.allclose(got[0, 7:], 1.0)


def test_cumulative_channel_max_is_unchanged_when_no_inf_is_present() -> None:
    rng = np.random.default_rng(6)
    z2 = np.abs(rng.normal(size=(50, T, C))) ** 2
    mask = rng.random((50, T, C)) > 0.25
    z2 = np.where(mask, z2, 0.0)
    z2[mask & (rng.random((50, T, C)) > 0.97)] = np.nan   # unscored observed bins
    q95 = np.ones((T, C))
    _assert_exact(S.cumulative_channel_max(z2, mask, q95),
                  _reference_cumulative_channel_max(z2, mask, q95),
                  "cumulative_channel_max(no inf)")


def test_nan_still_behaves_like_an_unobserved_gap() -> None:
    z2 = np.full((1, 12, 1), 2.0)
    z2[0, 3, 0] = np.nan
    mask = np.ones((1, 12, 1), dtype=bool)
    out = S.cumulative_channel_max(z2, mask, np.ones((12, 1)))
    assert np.isfinite(out[0, 4:]).all()

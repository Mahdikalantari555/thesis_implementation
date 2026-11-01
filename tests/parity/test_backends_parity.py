"""Parity tests between NumPy backend and future Rust backend.

The Rust backend is NOT built in v1 (gate G3); tests are written as
a placeholder that skips unless ``ssl_anomaly_rs`` is importable. The
intent is that when Rust lands, running ``pytest tests/parity`` will
immediately catch any numeric divergence.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.backends import get_backend

rng = np.random.default_rng(42)


def _random_cycles(seed=42, n=20, bins=24, bands=7):
    r = np.random.default_rng(seed)
    X = r.normal(0, 1, size=(n, bins, bands))
    # Inject NaN at ~10% of positions
    mask = r.random(X.shape) < 0.1
    X[mask] = np.nan
    # Inject low-n bin
    X[:, 0, :] = np.nan
    return X


def _random_signals(seed=42, B=8, T=200):
    r = np.random.default_rng(seed)
    y = np.sin(np.linspace(0, 6 * np.pi, T)) + r.normal(0, 0.2, size=(B, T))
    w = np.ones_like(y)
    w[:, 40:50] = 0.0
    return y, w


try:
    import ssl_anomaly_rs  # noqa: F401
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


@pytest.mark.skipif(not HAS_RUST, reason="Rust backend (ssl_anomaly_rs) not installed")
@pytest.mark.parity
def test_rust_robust_baseline_matches_numpy():
    npy = get_backend("numpy")
    rust = get_backend("rust")
    for seed in range(20):
        X = _random_cycles(seed=seed)
        m0, s0 = npy.robust_baseline(X, min_n=5)
        m1, s1 = rust.robust_baseline(X, min_n=5)
        np.testing.assert_allclose(m0, m1, rtol=1e-6, equal_nan=True)
        np.testing.assert_allclose(s0, s1, rtol=1e-6, equal_nan=True)


@pytest.mark.skipif(not HAS_RUST, reason="Rust backend (ssl_anomaly_rs) not installed")
@pytest.mark.parity
def test_rust_whittaker_matches_numpy():
    npy = get_backend("numpy")
    rust = get_backend("rust")
    y, w = _random_signals()
    z0 = npy.whittaker_batch(y, w, lam=50.0)
    z1 = rust.whittaker_batch(y, w, lam=50.0)
    np.testing.assert_allclose(z0, z1, rtol=1e-5, atol=1e-5)

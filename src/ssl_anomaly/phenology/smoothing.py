"""Weighted Whittaker-Eilers smoothing for irregular time series.

Reference: Eilers, P.H.C. (2003). "A perfect smoother." Anal. Chem. 75(14), 3631-3636.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

log = logging.getLogger("ssl_anomaly.phenology.smoothing")


def whittaker_smooth(
    y: np.ndarray,
    w: np.ndarray | None = None,
    lam: float = 10.0,
    d: int = 2,
) -> np.ndarray:
    """Whittaker-Eilers smoother.

    Parameters
    ----------
    y:
        1-D signal (regular grid). NaNs are treated as missing (w=0).
    w:
        Optional per-point weights (0 = missing). Length matches y.
    lam:
        Smoothing penalty; larger → smoother.
    d:
        Order of the difference penalty (2 = common default).
    """
    y = np.asarray(y, dtype=np.float64)
    n = y.size
    if n == 0:
        return y.copy()

    if w is None:
        w = np.ones(n, dtype=np.float64)
        w[~np.isfinite(y)] = 0.0
    else:
        w = np.asarray(w, dtype=np.float64)
        w = np.where(np.isfinite(y), w, 0.0)

    y_filled = np.where(np.isfinite(y), y, 0.0)

    # Build d-th order difference matrix
    D = sparse.eye(n, format="csr", dtype=np.float64)
    for _ in range(d):
        D = D[1:, :] - D[:-1, :]

    W = sparse.diags(w, format="csr", dtype=np.float64)
    A = (W + lam * (D.T @ D)).tocsc()
    z = spsolve(A, w * y_filled)
    return z


def to_daily_grid(
    t_days: np.ndarray,
    y: np.ndarray,
    quality: np.ndarray | None = None,
    lam: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Place irregular observations onto a daily grid and smooth.

    Parameters
    ----------
    t_days:
        1-D integer day indices (e.g., days since epoch).
    y:
        Observed values at those times.
    quality:
        Optional weights (e.g., valid_frac). Defaults to ones for finite obs.
    lam:
        Smoothing parameter.

    Returns
    -------
    grid:
        Daily day indices (int64), shape (T,) covering [min, max] of t_days.
    smoothed:
        Smoothed daily values, shape (T,).
    """
    t_days = np.asarray(t_days, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(t_days)
    t_days, y = t_days[order], y[order]

    if quality is not None:
        quality = np.asarray(quality, dtype=np.float64)[order]
    else:
        quality = np.ones_like(y)
    quality = quality * np.isfinite(y)

    t0 = int(t_days.min())
    t1 = int(t_days.max())
    grid = np.arange(t0, t1 + 1, dtype=np.int64)
    yy = np.zeros(grid.size, dtype=np.float64)
    ww = np.zeros(grid.size, dtype=np.float64)
    idx = (t_days - t0).astype(np.int64)
    # Average duplicates (e.g., two overpasses same day)
    np.add.at(yy, idx, y * quality)
    np.add.at(ww, idx, quality)
    valid = ww > 0
    yy[valid] /= ww[valid]
    yy[~valid] = 0.0  # filled before solve
    smoothed = whittaker_smooth(yy, w=ww, lam=lam)
    return grid, smoothed


def whittaker_tune_lambda(
    t_days: np.ndarray,
    y: np.ndarray,
    quality: np.ndarray | None = None,
    candidates: np.ndarray | None = None,
    cv_folds: int = 5,
    seed: int = 0,
) -> tuple[float, dict]:
    """Choose lambda by leave-out cross-validation on a daily grid.

    We randomly hold out 20% of the *observed* days, smooth on the rest with
    each candidate lambda, and pick the lambda with lowest weighted RMSE on
    held-out days. Returns (best_lambda, diagnostics).
    """
    rng = np.random.default_rng(seed)
    t_days = np.asarray(t_days, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    if candidates is None:
        candidates = np.logspace(1, 6, 12)

    # Project to daily grid first for consistent holdout
    t0 = int(t_days.min())
    t1 = int(t_days.max())
    grid = np.arange(t0, t1 + 1, dtype=np.int64)
    ww = np.zeros(grid.size, dtype=np.float64)
    yy = np.zeros(grid.size, dtype=np.float64)
    if quality is None:
        quality = np.ones_like(y)
    order = np.argsort(t_days)
    t_days, y, quality = t_days[order], y[order], quality[order]
    idx = (t_days - t0).astype(np.int64)
    np.add.at(yy, idx, y * quality)
    np.add.at(ww, idx, quality)
    valid = ww > 0
    yy[valid] /= ww[valid]
    obs_idx = np.where(valid)[0]
    if obs_idx.size < cv_folds * 2:
        log.debug("Too few points to tune lambda; using default 100")
        return 100.0, {"grid_mse": {}, "best_idx": 0}

    rng.shuffle(obs_idx)
    folds = np.array_split(obs_idx, cv_folds)
    scores: dict[float, float] = {}
    for lam in candidates:
        errs = []
        for k in range(cv_folds):
            held = folds[k]
            w_train = ww.copy()
            w_train[held] = 0.0
            yy_in = yy.copy()
            yy_in[held] = 0.0
            z = whittaker_smooth(yy_in, w=w_train, lam=float(lam))
            err = np.nanmean((yy[held] - z[held]) ** 2)
            errs.append(err)
        scores[float(lam)] = float(np.mean(errs))
    best = float(min(scores, key=scores.get))  # type: ignore[arg-type]
    return best, {"grid_mse": scores, "best_idx": int(np.argmin(list(scores.values())))}

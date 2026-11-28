"""B2 no-NN baseline: multi-band phenology + date common effect + robust σ.

Change task 1.1. Definition follows the consultant's P1 baseline
(``docs/model_improvement_proposals_2026-11-22.md`` §P1: phenology-only
0.070 → +date 0.055 → +date×farm-block 0.048 robust-SD ladder) adapted to the
32-farm probe:

- **multi-band** phenology profiles (NDVI + SAVI + LST, tasks.md 1.1 "not NDVI
  alone") binned by normalized cycle progress tau (from the v2 cycle table);
- per-date common effect (cross-sectional median of profile residuals — all
  32 farms share one management block R11, so date×block == date effect here;
  recorded as a documented limitation);
- score = max |z| over bands, with z = (y − expected)/σ and σ = 1.4826·MAD of
  **train-split** residuals per band×bin (leakage rule: nothing about eval
  dates enters the fit).
"""

from __future__ import annotations

import numpy as np

B2_BANDS = ("ndvi", "savi", "lst_c")


def _bin_idx(tau: np.ndarray, n_bins: int) -> np.ndarray:
    """Tau → bin index, nan-safe (NaN tau → 0; z is masked by isfinite(tau))."""
    with np.errstate(invalid="ignore"):
        b = tau * n_bins
    idx = np.where(np.isfinite(b), np.nan_to_num(b, nan=0.0), 0.0)
    return np.clip(idx.astype(int), 0, n_bins - 1)


def _mad_sigma(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if x.size < 8:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


class B2Baseline:
    """Fit on train dates only; score any (farm, date).

    ``min_farms``: minimum cross-sectional observations for a date effect /
    bin profile (default 8; the real cohort is 32 farms, tiny fixtures pass 2).
    """

    def __init__(self, n_bins: int = 24, sigma_floor_rel: float = 0.1,
                 min_farms: int = 8) -> None:
        self.n_bins = n_bins
        self.sigma_floor_rel = sigma_floor_rel
        self.min_farms = min_farms
        self.profile: dict[str, np.ndarray] = {}
        self.sigma: dict[str, np.ndarray] = {}
        self.sigma_floor: dict[str, float] = {}
        self.fitted = False

    def fit(self, d) -> "B2Baseline":
        """``d``: ProbeData. Uses train dates only."""
        tr = d.train_mask
        ndvi, savi, lst_c = d.ndvi, d.savi, d.lst_c
        feats = {"ndvi": ndvi, "savi": savi, "lst_c": lst_c}
        tau = d.tau
        obs = d.obs_mask & tr[None, :] & np.isfinite(tau)
        bin_idx = _bin_idx(tau, self.n_bins)
        for name, y in feats.items():
            profile = np.full(self.n_bins, np.nan)
            resid = np.full_like(y, np.nan, dtype=np.float64)
            valid_obs = obs & np.isfinite(y)
            for k in range(self.n_bins):
                sel = valid_obs & (bin_idx == k)
                if sel.sum() >= self.min_farms:
                    profile[k] = np.nanmedian(y[sel])
                    resid[sel] = y[sel] - profile[k]
            # per-date common effect (train dates only; cross-sectional median)
            date_eff = np.full(y.shape[1], np.nan)
            for di in np.where(tr)[0]:
                col = resid[:, di]
                col = col[np.isfinite(col)]
                if col.size >= self.min_farms:
                    date_eff[di] = np.median(col)
            resid2 = resid - date_eff[None, :]
            sigma = np.full(self.n_bins, np.nan)
            valid2 = valid_obs & np.isfinite(resid2)
            for k in range(self.n_bins):
                vals = resid2[valid2 & (bin_idx == k)]
                vals = vals[np.isfinite(vals)]
                sigma[k] = _mad_sigma(vals) if vals.size >= max(4, self.min_farms) \
                    else np.nan
            global_sd = float(np.nanstd(resid2[valid2]))
            if not np.isfinite(global_sd):
                raise RuntimeError(
                    f"B2 fit produced no finite residuals for band {name!r} "
                    "(train split too small or tau/obs misaligned)")
            floor = max(global_sd * self.sigma_floor_rel, 1e-6)
            sigma = np.where(np.isfinite(sigma), sigma, global_sd)
            sigma = np.maximum(sigma, floor)
            self.profile[name] = profile
            self.sigma[name] = sigma
            self.sigma_floor[name] = floor
        self.fitted = True
        return self

    def score(self, d) -> dict[str, np.ndarray]:
        """Returns ``{"z_<band>": (F, D), "b2_score": (F, D)}``; absent = NaN."""
        if not self.fitted:
            raise RuntimeError("B2Baseline.score before fit")
        feats = {"ndvi": d.ndvi, "savi": d.savi, "lst_c": d.lst_c}
        tau = d.tau
        bin_idx = _bin_idx(tau, self.n_bins)
        z_by_band = {}
        z_stack = []
        for name, y in feats.items():
            profile = self.profile[name]
            expected = profile[bin_idx]
            z = y - expected
            z_out = np.full_like(y, np.nan, dtype=np.float32)
            ok = np.isfinite(z) & np.isfinite(expected) & np.isfinite(tau)
            z_out[ok] = (z[ok] / self.sigma[name][bin_idx[ok]]).astype(np.float32)
            z_by_band[f"z_{name}"] = z_out
            z_stack.append(z_out)
        zs = np.stack(z_stack, axis=0)  # (3, F, D)
        with np.errstate(all="ignore"):
            score = np.nanmax(np.abs(zs), axis=0)
        score[~np.isfinite(score)] = np.nan
        score[~d.obs_mask] = np.nan
        return {**z_by_band, "b2_score": score.astype(np.float32)}

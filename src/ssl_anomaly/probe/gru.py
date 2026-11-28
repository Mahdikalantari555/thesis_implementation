"""GRU-AE baseline on field trajectories (change task 1.4).

Wraps the existing causal one-step-ahead ``models/gru_ae.CausalGRUAE``
(established implementation) on the field-mean spectral-temporal trajectory:
tokens = 7 standardized band values (zero only where unobserved) + observation
flag + DOY sin/cos. Score per (farm, date) = standardized one-step-ahead
reconstruction residual of observed bands. Fit on **train dates only** (loss
masked to train dates; scoring covers all dates — the model is causal, so a
date's residual depends only on earlier observations).
"""

from __future__ import annotations

from datetime import date

import numpy as np
import torch

from ..models.gru_ae import CausalGRUAE


def build_tokens(d, band_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(F, D, 10) tokens + (F, D, 7) target + (F, D, 7) finite mask.

    Channels: 7 band values (NaN→0 placeholder; the obs-flag channels carry
    absence explicitly — pilot-review §6.1 convention), 7 obs flags, DOY sin,
    DOY cos. Bands standardized by **train-split** field-mean statistics.
    """
    F, D, K = band_matrix.shape
    mu = np.nanmean(band_matrix[:, d.train_mask], axis=(0, 1))
    sd = np.nanstd(band_matrix[:, d.train_mask], axis=(0, 1))
    sd = np.where(sd > 1e-6, sd, 1.0)
    z = (band_matrix - mu) / sd
    finite = (d.obs_mask[..., None] & np.isfinite(z))
    values = np.nan_to_num(z, nan=0.0)
    obs_flag = finite.astype(np.float32)
    doy = np.array([date.fromisoformat(s).timetuple().tm_yday for s in d.dates])
    doy_sin = np.sin(2 * np.pi * doy / 365.25)[None, :, None].repeat(F, axis=0)
    doy_cos = np.cos(2 * np.pi * doy / 365.25)[None, :, None].repeat(F, axis=0)
    tokens = np.concatenate([values, obs_flag, doy_sin.astype(np.float32),
                             doy_cos.astype(np.float32)], axis=2)
    target = np.nan_to_num(z, nan=0.0)
    return tokens.astype(np.float32), target.astype(np.float32), finite


class TrajectoryGRUAE:
    def __init__(self, d_model: int = 224, seed: int = 2026) -> None:
        self.d_model = d_model
        self.seed = seed
        self.model = CausalGRUAE(d_in=16, n_out=7, d_model=d_model)
        self.band_sigma: np.ndarray | None = None
        self.trained = False

    def count_params(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def train(self, tokens: np.ndarray, target: np.ndarray, finite: np.ndarray,
              train_mask: np.ndarray, epochs: int = 5, lr: float = 1e-3,
              progress_cb=None, start_ep: int = 0) -> float:
        torch.manual_seed(self.seed)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        x = torch.from_numpy(tokens)
        y = torch.from_numpy(target)
        pretext = torch.from_numpy(train_mask)[None, :].repeat(x.shape[0], 1)
        channel_mask = torch.from_numpy(finite)
        last = float("nan")
        for ep in range(start_ep, epochs):
            self.model.train()
            loss, _ = self.model.masked_reconstruction_loss(
                x, pretext, y, channel_mask=channel_mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = float(loss.detach())
            if progress_cb:
                progress_cb(ep, last)
        # per-band σ of train residuals (standardized scores)
        self.model.eval()
        with torch.no_grad():
            recon = self.model.forward(x).numpy()       # (F, D, 7)
        resid = (recon - target)[:, train_mask]
        resid[~finite[:, train_mask]] = np.nan
        self.band_sigma = np.maximum(np.nanstd(resid, axis=(0, 1)), 1e-6)
        self.trained = True
        return last

    def score(self, tokens: np.ndarray, target: np.ndarray,
              finite: np.ndarray) -> np.ndarray:
        """(F, D) standardized one-step-ahead residual score; absent = NaN."""
        if not self.trained:
            raise RuntimeError("TrajectoryGRUAE.score before train")
        self.model.eval()
        with torch.no_grad():
            recon = self.model.forward(torch.from_numpy(tokens)).numpy()
        resid = np.abs(recon - target) / self.band_sigma[None, None, :]
        resid = resid.mean(axis=2)                      # (F, D)
        resid[~(finite.any(axis=2))] = np.nan
        return resid.astype(np.float32)

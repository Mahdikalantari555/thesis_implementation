"""Pixel-wise CNN autoencoder for the 32-farm probe (change task 1.2).

≤250k parameters (measured in ``count_params`` and asserted by the unit test).
Input = 7 band canvases (48×48) **zero-filled only where invalid, with an
explicit observation-mask channel** (the established convention of
``models/gru_ae.py`` — zero-fill alone was flagged as a defect by the 2026-11-25
pilot review §6.1); reconstruction loss and scores are computed ONLY on
in-field ∧ QA-valid ∧ finite pixels. NaN never reaches the loss.

Norm statistics (per-band mean/std over in-field ∧ valid pixels of TRAIN dates
only) are fit at construction and reused at scoring/Stage-2 time.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class PixelCNNAE(nn.Module):
    """Conv AE on (B, 8, S, S) → (B, 7, S, S). ~198k params at S=48."""

    def __init__(self, in_bands: int = 7, base: int = 16) -> None:
        super().__init__()
        ch = in_bands + 1  # + observation-mask channel
        self.encoder = nn.Sequential(
            nn.Conv2d(ch, base, 3, stride=2, padding=1),
            nn.GroupNorm(4, base), nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.GroupNorm(8, base * 4), nn.ReLU(inplace=True),
        )
        self.enc_fc = nn.Linear(base * 4 * 6 * 6, 32)
        self.dec_fc = nn.Linear(32, base * 4 * 6 * 6)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1),
            nn.GroupNorm(8, base * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1),
            nn.GroupNorm(4, base), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, in_bands, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = torch.relu(self.enc_fc(h.flatten(1)))
        h = self.dec_fc(h).reshape(x.shape[0], -1, 6, 6)
        return self.decoder(h)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_input(canvas: np.ndarray, valid: np.ndarray, in_field: np.ndarray,
               norm: dict) -> tuple[np.ndarray, np.ndarray]:
    """(7, S, S) fp16 canvas + masks → model input (8, S, S) fp32 + target/loss mask.

    Returns ``(x, loss_mask)`` where ``loss_mask`` = in_field ∧ pixel_valid ∧
    finite-target. Invalid positions are zero-filled (placeholder only — the
    mask channel tells the model which values are observed).
    """
    vals = canvas.astype(np.float32)              # (7, S, S), NaN allowed
    target = vals.copy()
    target[~np.isfinite(target)] = 0.0
    for k, b in enumerate(["blue", "green", "red", "nir", "swir1", "swir2", "lst"]):
        vals[k] = (vals[k] - norm[f"{b}_mean"]) / norm[f"{b}_std"]
    vals[~np.isfinite(vals)] = 0.0
    x = np.concatenate([vals, valid[None].astype(np.float32)], axis=0)
    loss_mask = in_field & valid & np.isfinite(canvas).all(axis=0)
    return x.astype(np.float32), loss_mask


def fit_norm_stats(d, train_mask: np.ndarray) -> dict:
    """Per-band mean/std over in-field ∧ valid pixels of TRAIN dates only."""
    sums = np.zeros(7)
    sqs = np.zeros(7)
    n = 0
    tr_dates = np.where(train_mask)[0]
    for f, (in_field, data, pixel_valid) in d.canvases.items():
        sub = data[tr_dates]                                  # (Td, 7, S, S)
        m = (in_field[None] & pixel_valid[tr_dates] & np.isfinite(sub.all(axis=1)))
        if m.sum() == 0:
            continue
        vals = np.concatenate([sub[t][:, m[t]] for t in range(sub.shape[0])],
                              axis=1)                         # (7, N)
        vals64 = vals.astype(np.float64)   # LST ~300 K: fp32 sums lose precision
        sums += vals64.sum(axis=1)
        sqs += (vals64 ** 2).sum(axis=1)
        n += vals.shape[1]
    if n == 0:
        raise RuntimeError("No train-date in-field pixels for norm statistics")
    norm = {}
    for k, b in enumerate(["blue", "green", "red", "nir", "swir1", "swir2", "lst"]):
        mean = sums[k] / n
        var = max(sqs[k] / n - mean ** 2, 1e-12)
        norm[f"{b}_mean"] = float(mean)
        norm[f"{b}_std"] = float(np.sqrt(var))
    return norm


def masked_loss(pred: torch.Tensor, target: torch.Tensor,
                loss_mask: torch.Tensor) -> torch.Tensor:
    """Mean squared error over loss_mask pixels only (all 7 bands)."""
    diff = (pred - target) ** 2                     # (B, 7, S, S)
    m = loss_mask.unsqueeze(1).float()              # (B, 1, S, S)
    tot = (diff * m).sum()
    cnt = (m.sum() * pred.shape[1]).clamp(min=1.0)
    return tot / cnt


def field_date_scores(pred: np.ndarray, target: np.ndarray, loss_mask: np.ndarray,
                      band_sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixel score = mean over valid bands of |pred−target|/σ_b (train σ);
    field-date score = mean pixel score over loss_mask. Returns
    ``(field_date_score, pixel_score_map)`` with NaN where mask empty."""
    diff = np.abs(pred - target) / band_sigma[None, :, None, None]
    per_px = diff.mean(axis=1)                      # (B, S, S)
    per_px[~loss_mask] = np.nan
    with np.errstate(all="ignore"):
        fd = np.nanmean(per_px, axis=(1, 2))
    fd[~loss_mask.any(axis=(1, 2))] = np.nan
    return fd.astype(np.float32), per_px.astype(np.float32)

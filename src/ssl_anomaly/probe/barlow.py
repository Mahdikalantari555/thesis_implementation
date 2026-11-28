"""Barlow Twins on spectral-temporal field trajectories (change task 1.3).

ST-BT style: per (farm, window of W consecutive observations) the 7-band
field-mean trajectory is embedded; two augmented views are pulled to agree
(cross-correlation → identity, canonical loss with off-diagonal exclusion —
the fixed ``_barlow_loss`` from ``mini_lab/runner.py``, commit 83b7a78).
Score = standardized distance of each date's window embedding from the
farm's **train-split** embedding centroid (representation distance = score,
tasks.md 1.3).

``random_init=True`` scores with an untrained encoder of the same
architecture — the mandatory random-init comparator for any SSL arm (§7).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..mini_lab.runner import _barlow_loss

BANDS = ["blue", "green", "red", "nir", "swir1", "swir2", "lst"]


class TrajectoryEncoder(nn.Module):
    """(B, W, 7) → (B, E) MLP encoder + projector."""

    def __init__(self, window: int = 16, d_in: int = 7, hidden: int = 256,
                 emb: int = 128) -> None:
        super().__init__()
        self.window = window
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(window * d_in, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, emb),
        )
        self.projector = nn.Sequential(nn.Linear(emb, emb), nn.ReLU(inplace=True),
                                       nn.Linear(emb, emb))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def build_windows(d, band_matrix: np.ndarray, window: int = 16,
                  min_obs_frac: float = 0.5) -> tuple[list, np.ndarray]:
    """Windows of W consecutive *observed* dates per farm.

    Returns ``(windows, valid)``: windows[i] = (farm_i, date_i, X (W, 7) with
    NaN→0 placeholder + per-window obs mask (W, 7)). A window is valid when ≥
    ``min_obs_frac`` of its positions are observed. Bands are standardized by
    **train-split** field-mean statistics before windowing.
    """
    F, D, K = band_matrix.shape
    mu = np.nanmean(band_matrix[:, d.train_mask], axis=(0, 1))
    sd = np.nanstd(band_matrix[:, d.train_mask], axis=(0, 1))
    sd = np.where(sd > 1e-6, sd, 1.0)
    z = (band_matrix - mu) / sd
    obs = d.obs_mask & np.isfinite(band_matrix).any(axis=2)
    out = []
    for fi in range(F):
        obs_d = np.where(obs[fi])[0]
        for start in range(0, max(len(obs_d) - window + 1, 0)):
            idx = obs_d[start:start + window]
            if len(idx) < window:
                break
            x = z[fi, idx].copy()                    # (W, 7)
            m = np.isfinite(x)
            x = np.nan_to_num(x, nan=0.0)
            if m.mean() < min_obs_frac:
                continue
            out.append({"farm_i": fi, "date_i": int(idx[-1]), "x": x,
                        "mask": m, "dates_idx": idx,
                        "is_train": bool(d.train_mask[idx[-1]])})
    return out, z


def augment(x: torch.Tensor, mask: torch.Tensor, rng: np.random.Generator,
            noise: float = 0.1, dropout: float = 0.15) -> torch.Tensor:
    """View augmentation: additive Gaussian noise + random band dropout
    (applied only to observed positions; unobserved stay 0 + mask unchanged)."""
    x = x.clone()
    n = torch.from_numpy(rng.normal(0, noise, size=x.shape)).float()
    x = x + n * mask.float()
    drop = torch.from_numpy(rng.random(size=(x.shape[0], 1, x.shape[2])) < dropout).bool()
    x = torch.where(drop & mask.bool(), torch.zeros_like(x), x)
    return x


class BarlowProbe:
    def __init__(self, window: int = 16, emb: int = 128, seed: int = 2026,
                 random_init: bool = False) -> None:
        self.window = window
        self.seed = seed
        self.random_init = random_init
        self.model = TrajectoryEncoder(window=window, emb=emb)
        self.rng = np.random.default_rng(seed)
        self.centroids: dict[int, np.ndarray] = {}
        self.stds: dict[int, np.ndarray] = {}
        self.trained = False

    def train(self, windows: list[dict], epochs: int = 5, batch_size: int = 64,
              lr: float = 1e-3, progress_cb=None, start_ep: int = 0) -> float:
        torch.manual_seed(self.seed)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        last = float("nan")
        for ep in range(start_ep, epochs):
            rng = np.random.default_rng(self.seed + ep)
            order = rng.permutation(len(windows))
            losses = []
            for bs in range(0, len(order), batch_size):
                batch = [windows[i] for i in order[bs:bs + batch_size]]
                x = torch.from_numpy(np.stack([w["x"] for w in batch])).float()
                m = torch.from_numpy(np.stack([w["mask"] for w in batch]))
                v1 = augment(x, m, rng)
                v2 = augment(x, m, rng)
                z1 = self.model.projector(self.model(v1))
                z2 = self.model.projector(self.model(v2))
                loss = _barlow_loss(z1, z2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach()))
            last = float(np.mean(losses)) if losses else float("nan")
            if progress_cb:
                progress_cb(ep, last)
        self.trained = not self.random_init
        return last

    def embed_windows(self, windows: list[dict]) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.from_numpy(np.stack([w["x"] for w in windows])).float()
            return self.model(x).numpy()

    def fit_scoring(self, windows: list[dict]) -> "BarlowProbe":
        """Per-farm centroid/std of embeddings over TRAIN windows only."""
        E = self.embed_windows(windows)
        for w, e in zip(windows, E):
            if not w["is_train"]:
                continue
            fi = w["farm_i"]
            self.centroids.setdefault(fi, []).append(e)
        for fi, es in self.centroids.items():
            es = np.stack(es)
            self.centroids[fi] = es.mean(axis=0)
            self.stds[fi] = np.maximum(es.std(axis=0), 1e-3)
        # global fallback for farms without train windows
        all_tr = np.stack([e for w, e in zip(windows, E) if w["is_train"]])
        self.global_centroid = all_tr.mean(axis=0)
        self.global_std = np.maximum(all_tr.std(axis=0), 1e-3)
        return self

    def score(self, windows: list[dict], n_farms: int, n_dates: int) -> np.ndarray:
        """(n_farms, n_dates) score = mean standardized distance; absent = NaN."""
        out = np.full((n_farms, n_dates), np.nan, dtype=np.float32)
        E = self.embed_windows(windows)
        for w, e in zip(windows, E):
            fi = w["farm_i"]
            c = self.centroids.get(fi, self.global_centroid)
            s = self.stds.get(fi, self.global_std)
            out[fi, w["date_i"]] = float(np.mean(((e - c) / s) ** 2))
        return out

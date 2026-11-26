"""Faithful 1D Time2Agri pretext arms (Gupta & Sobti 2025, arXiv:2507.04366).

Three distinct objectives on a shared tiny ViT-style 1D encoder over cycle
tokens ``(B, T, C)``:

* ``time2agri_td`` — classify the absolute bin gap between two frames.
* ``time2agri_fp`` — regress the top-3 dominant FFT frequencies of the observed
  NDVI series.
* ``time2agri_ff`` — reconstruct a future frame from an origin frame.

This is an L1 pretext implementation only. The paper's "FF error = anomaly
signal" claim is argued, not measured, and no stress/phase claim follows here.
The paper uses RGB+NIR image chips; the 1D field-median adaptation is documented,
not claimed equivalent. FP targets are derived from observed train-period NDVI
only (``allow_eval=False`` by default).
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch import nn
from torch.nn import functional as F

ARMS = ("time2agri_td", "time2agri_fp", "time2agri_ff")
GAP_CLASSES = 4


def build_time_encoding(dates: list[str] | np.ndarray) -> np.ndarray:
    """Month sin/cos + linear year, shape ``(T, 3)`` (month+year, not DOY)."""
    d = np.asarray([str(x)[:10] for x in dates], dtype="datetime64[D]")
    months = (d.astype("datetime64[M]").astype(int) % 12) + 1
    years = d.astype("datetime64[Y]").astype(int) + 1970
    m = 2.0 * np.pi * (months - 1) / 12.0
    return np.stack([np.sin(m), np.cos(m), (years - 2013) / 13.0], axis=-1).astype(np.float32)


class TinyViT1DEncoder(nn.Module):
    """Patchified 1D transformer: ``(B,T,C) -> (cls, seq, seq_valid)``."""

    def __init__(
        self,
        d_in: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 64,
        patch_size: int = 2,
        dropout: float = 0.1,
        max_patches: int = 80,
        d_time: int = 3,
    ) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.d_model = int(d_model)
        self.d_time = int(d_time)
        self.patch_in = nn.Linear(self.patch_size * (d_in + d_time), d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, std=0.02)
        self.pos = nn.Parameter(torch.zeros(1, max_patches + 1, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d_model)

    def patchify(
        self, x: torch.Tensor, observed: torch.Tensor | None, time_enc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, c = x.shape
        p = self.patch_size
        pad = (-t) % p
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
            if observed is not None:
                observed = F.pad(observed, (0, 0, 0, pad))
            time_enc = F.pad(time_enc, (0, 0, 0, pad))
        t2 = x.shape[1]
        patches = x.reshape(b, t2 // p, p * c)
        obs_p = (
            observed.reshape(b, t2 // p, p * c).any(-1)
            if observed is not None
            else torch.ones(b, t2 // p, dtype=torch.bool, device=x.device)
        )
        te = time_enc[:, :t2].reshape(b, t2 // p, p * self.d_time)
        tok = torch.cat([patches, te], dim=-1)
        return self.patch_in(tok), obs_p

    def forward(
        self,
        x: torch.Tensor,
        observed: torch.Tensor | None = None,
        time_enc: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, t, c = x.shape
        if time_enc is None:
            time_enc = x.new_zeros(b, t, self.d_time)
        tok, valid = self.patchify(x, observed, time_enc)
        n = tok.shape[1]
        cls = self.cls.expand(b, -1, -1)
        seq = torch.cat([cls, tok], dim=1)
        seq = seq + self.pos[:, : n + 1]
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~valid], dim=1)
        if bool(pad.all(dim=1).any()):
            pad[pad.all(dim=1), 1] = False
        seq = self.ln(self.blocks(seq, src_key_padding_mask=pad))
        keep = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=x.device), valid], dim=1)
        return seq[:, 0], seq, keep


def fft_ndvi_target(
    ndvi: np.ndarray,
    observed: np.ndarray | None = None,
    *,
    allow_eval: bool = False,
    split: str | None = None,
) -> np.ndarray:
    """Top-3 positive FFT frequencies + normalized amplitudes of observed NDVI.

    ``allow_eval=False`` refuses samples explicitly marked as evaluation data, so
    a frequency target can never be built from a held-out period.
    """
    if (not allow_eval) and (split is not None) and str(split).lower() in {"eval", "test", "dev", "holdout"}:
        raise ValueError("FP frequency target refused for an evaluation-period sample (leakage guard)")
    y = np.asarray(ndvi, dtype=np.float64).copy()
    m = np.ones(len(y), dtype=bool) if observed is None else np.asarray(observed, dtype=bool)
    if y[m].size >= 5:
        y[m] = savgol_filter(y[m], min(5, y[m].size | 1), 2)
    idx = np.arange(len(y))
    if m.sum() < 2:
        return np.zeros(6, dtype=np.float32)
    y = np.interp(idx, idx[m], y[m])
    y = y - y.mean()
    if np.allclose(y, 0.0):
        return np.zeros(6, dtype=np.float32)
    spec = np.abs(np.fft.rfft(y))[1:]
    if spec.size == 0:
        return np.zeros(6, dtype=np.float32)
    top = np.argsort(spec)[::-1][:3]
    freqs = (top + 1) / max(1, len(spec))
    amps = spec[top] / max(1e-12, spec.sum())
    out = np.zeros(6, dtype=np.float32)
    out[: len(top)] = freqs
    out[3 : 3 + len(top)] = amps
    return out


class _Time2AgriBase(nn.Module):
    arm: str = ""

    def __init__(self, d_in: int, d_model: int = 32, ndvi_index: int = 0, **_: object) -> None:
        super().__init__()
        self.d_in = int(d_in)
        self.ndvi_index = int(ndvi_index)
        self.encoder = TinyViT1DEncoder(d_in, d_model=d_model)
        self.pool = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 32))

    def objective_spec(self) -> dict:
        raise NotImplementedError

    def pretext_loss(self, x: torch.Tensor, observed: torch.Tensor | None = None, **_: object) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor]:
        cls, _, _ = self.encoder(x)
        return None, self.pool(cls)

    @staticmethod
    def _te(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Month sin/cos + linear year from integer day numbers (deterministic)."""
        days = positions.to(x.dtype)
        month = (days % 365.0) / 30.0
        ph = 2.0 * np.pi * month
        return torch.stack([torch.sin(ph), torch.cos(ph), days / 3650.0], dim=-1).expand(x.shape[0], -1, -1)


class Time2AgriTD(_Time2AgriBase):
    arm = "time2agri_td"

    def __init__(self, d_in: int, **kw: object) -> None:
        super().__init__(d_in, **kw)
        d = self.encoder.d_model
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, GAP_CLASSES))

    def objective_spec(self) -> dict:
        return {"objective": "cross_entropy", "target_source": "bin_gap_class",
                "gradient_path": "frame_cls_pair", "horizon": 0}

    def pretext_loss(self, x: torch.Tensor, observed: torch.Tensor | None = None, **_: object) -> torch.Tensor:
        b, t, _ = x.shape
        _, seq, keep = self.encoder(x, observed)
        n = seq.shape[1] - 1
        if n < GAP_CLASSES + 1:
            return x.new_zeros(())
        cls_gap = torch.arange(b, device=x.device) % GAP_CLASSES
        gap_patches = cls_gap + 1
        a = (torch.arange(b, device=x.device) // GAP_CLASSES) % torch.clamp(n - gap_patches, min=1)
        a = a.long()
        bb = torch.clamp(a + gap_patches, max=n - 1)
        ha = seq[torch.arange(b, device=x.device), a + 1]
        hb = seq[torch.arange(b, device=x.device), bb + 1]
        h = torch.cat([ha, hb], dim=-1)
        logits = self.head(h)
        rev = self.head(torch.cat([hb, ha], dim=-1))
        return 0.5 * (F.cross_entropy(logits, cls_gap) + F.cross_entropy(rev, cls_gap))


class Time2AgriFP(_Time2AgriBase):
    arm = "time2agri_fp"

    def __init__(self, d_in: int, **kw: object) -> None:
        super().__init__(d_in, **kw)
        self.head = nn.Sequential(nn.Linear(self.encoder.d_model + 3, 64), nn.GELU(), nn.Linear(64, 6))

    def objective_spec(self) -> dict:
        return {"objective": "mse", "target_source": "dominant_ndvi_fft",
                "gradient_path": "cls", "horizon": t_full}

    def pretext_loss(
        self, x: torch.Tensor, observed: torch.Tensor | None = None, *,
        ndvi_targets: torch.Tensor | None = None, **_: object,
    ) -> torch.Tensor:
        if ndvi_targets is None:
            raise ValueError("FP arm requires train-period NDVI frequency targets")
        b, t, _ = x.shape
        q = max(2, t // 4)
        frames = torch.cat([x[:, :q], x[:, -q:]], dim=1)
        obs_f = None if observed is None else torch.cat([observed[:, :q], observed[:, -q:]], dim=1)
        te = self._te(x, torch.cat([torch.arange(q), torch.arange(t - q, t)]))
        cls, _, _ = self.encoder(frames, obs_f, te)
        pred = self.head(torch.cat([cls, te.mean(1)], dim=-1))
        return F.mse_loss(pred, ndvi_targets.to(pred.device))


class Time2AgriFF(_Time2AgriBase):
    arm = "time2agri_ff"

    def __init__(self, d_in: int, **kw: object) -> None:
        super().__init__(d_in, **kw)
        d = self.encoder.d_model
        self.translator = nn.Sequential(nn.Linear(d + 3, d), nn.GELU(), nn.Linear(d, d))
        self.decoder = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.d_in))

    def objective_spec(self) -> dict:
        return {"objective": "masked_mse", "target_source": "future_observed_bins",
                "gradient_path": "origin_cls", "horizon": -1}

    def pretext_loss(self, x: torch.Tensor, observed: torch.Tensor | None = None, **_: object) -> torch.Tensor:
        b, t, c = x.shape
        half = t // 2
        origin = x[:, :half]
        future = x[:, half:]
        te_f = self._te(x, torch.arange(half, t))
        cls, _, _ = self.encoder(origin, None if observed is None else observed[:, :half])
        z = self.translator(torch.cat([cls, te_f.mean(1)], dim=-1))
        pred = self.decoder(z).unsqueeze(1).expand(-1, t - half, -1)
        m = torch.ones_like(future, dtype=torch.bool) if observed is None else observed[:, half:]
        err = (pred - future).pow(2)
        return (err * m).sum() / m.sum().clamp(min=1)


BUILDERS = {
    "time2agri_td": Time2AgriTD,
    "time2agri_fp": Time2AgriFP,
    "time2agri_ff": Time2AgriFF,
}


def build_time2agri_arm(name: str, d_in: int, **kw: object) -> _Time2AgriBase:
    if name not in BUILDERS:
        raise KeyError(f"Unknown Time2Agri arm {name!r}; expected {sorted(BUILDERS)}")
    return BUILDERS[name](d_in, **kw)


def build_fp_targets(x: np.ndarray, observed: np.ndarray, ndvi_index: int = 0, *,
                     split: str | None = None) -> np.ndarray:
    """Per-sample FP targets; each built from that sample's observed train NDVI."""
    out = []
    for i in range(len(x)):
        m = observed[i, :, ndvi_index].astype(bool)
        out.append(fft_ndvi_target(x[i, :, ndvi_index], m, split=split))
    return np.stack(out).astype(np.float32)


# "full series" is a string because objective_spec is JSON-serialised.
t_full = "full_series"

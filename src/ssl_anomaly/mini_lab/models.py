"""Model-agnostic registry for the mini experiment harness.

All nine architectures stay in the framework. Existing repository models are
reused through a uniform ``forward(x) -> (recon, embed)`` adapter; the three
small missing autoencoders are implemented here in pure PyTorch. Heavy models
are tagged for Colab placement but are never removed or silently substituted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

from .config import ModelConfig


class _MLPAE(nn.Module):
    """Per-timestep MLP encoder with temporal average pooling and MLP decoder."""

    def __init__(self, d_in: int, d_model: int = 32, latent_dim: int = 16, **_: Any):
        super().__init__()
        self.enc_time = nn.Linear(d_in, d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.dec_time = nn.Linear(d_model, d_in)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.nn.functional.gelu(self.enc_time(x))  # (B, T, d_model)
        pooled = self.pool(h.transpose(1, 2)).flatten(1)
        z = self.to_latent(pooled)
        h2 = self.from_latent(z).unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.dec_time(h2), z


class _LSTMAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int = 32,
        latent_dim: int = 16,
        n_layers: int = 1,
        dropout: float = 0.1,
        **_: Any,
    ):
        super().__init__()
        self.enc = nn.LSTM(d_in, d_model, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.dec = nn.Linear(d_model, d_in)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out, _ = self.enc(x)
        z = self.to_latent(out[:, -1])
        h = self.from_latent(z).unsqueeze(1).expand(-1, x.shape[1], -1)
        return self.dec(h), z


class _ConvLSTMCell(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class _ConvLSTMAE(nn.Module):
    """Small ConvLSTM autoencoder over the temporal axis (pure PyTorch)."""

    def __init__(
        self,
        d_in: int,
        d_model: int = 32,
        latent_dim: int = 16,
        **_: Any,
    ):
        super().__init__()
        self.cell = _ConvLSTMCell(d_in, d_model)
        self.to_latent = nn.Linear(d_model, latent_dim)
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.dec = nn.Linear(d_model, d_in)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # (B, T, F) -> (B, F, T) so the Conv1d kernel spans neighbouring dates.
        seq = x.transpose(1, 2)
        b, _, t = seq.shape
        h = torch.zeros(b, self.d_model, 1, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        states = []
        for step in range(t):
            h, c = self.cell(seq[:, :, step : step + 1], (h, c))
            states.append(h)
        hseq = torch.cat(states, dim=2)
        z = self.to_latent(hseq[:, :, -1])
        hz = self.from_latent(z).unsqueeze(2).expand(-1, -1, t)
        recon = self.dec(hz.transpose(1, 2))
        return recon, z


class _Adapter(nn.Module):
    """Wrap an existing model with the uniform (recon, embed) interface."""

    def __init__(self, model: nn.Module, *, projection: bool = False):
        super().__init__()
        self.model = model
        self.projection = projection

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(x)
        if isinstance(out, dict):
            recon, embed = out["recon"], out["embed"]
        elif isinstance(out, (tuple, list)):
            recon, embed = out[0], out[1]
        else:
            raise TypeError(f"Unsupported model output {type(out)}")
        if recon.shape != x.shape:
            if recon.ndim == 2:
                recon = recon.unsqueeze(1).expand(-1, x.shape[1], -1)
            else:
                raise RuntimeError(f"Reconstruction shape {recon.shape} != input {x.shape}")
        return recon, embed


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str  # reconstruction | representation
    placement: str  # laptop | colab
    source: str
    builder: Callable[[int, ModelConfig], nn.Module]
    available: bool = True
    note: str = ""


def _gru(d_in: int, cfg: ModelConfig) -> nn.Module:
    from ssl_anomaly.models.gru_ae import GRUAutoencoder

    return _Adapter(
        GRUAutoencoder(
            d_in=d_in,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            bidirectional_enc=True,
        )
    )


def _cnn(d_in: int, cfg: ModelConfig) -> nn.Module:
    from ssl_anomaly.models.sts_net import STSNetAutoencoder

    return _Adapter(
        STSNetAutoencoder(d_in=d_in, d_model=cfg.d_model, latent_dim=cfg.latent_dim, dropout=cfg.dropout)
    )


def _mamba(d_in: int, cfg: ModelConfig) -> nn.Module:
    from ssl_anomaly.models.cnn_mamba import TinyCNNMamba

    return _Adapter(
        TinyCNNMamba(
            d_in=d_in,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            latent_dim=cfg.latent_dim,
            dropout=cfg.dropout,
        )
    )


def _cnn_mamba(d_in: int, cfg: ModelConfig) -> nn.Module:
    from ssl_anomaly.models.cnn_mamba import TinyCNNMamba

    return _Adapter(
        TinyCNNMamba(
            d_in=d_in,
            d_model=max(cfg.d_model, 32),
            d_state=16,
            n_layers=max(cfg.n_layers, 2),
            latent_dim=cfg.latent_dim,
            dropout=cfg.dropout,
        )
    )


def _jepa(d_in: int, cfg: ModelConfig) -> nn.Module:
    from ssl_anomaly.models.jepa import JEPAModel

    return _Adapter(
        JEPAModel(
            d_in=d_in,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )
    )


def _barlow(d_in: int, cfg: ModelConfig) -> nn.Module:
    from ssl_anomaly.models.rst_bt import RSTBTModel

    return _Adapter(
        RSTBTModel(
            d_in=d_in,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            proj_out=cfg.latent_dim,
        )
    )


def _time2agri(name: str):
    def build(d_in: int, cfg: ModelConfig) -> nn.Module:
        from ssl_anomaly.models.time2agri_archs import build_time2agri_arm

        # Returned directly: its own pretext_loss is the training objective,
        # never the generic Barlow loss of the representation family.
        return build_time2agri_arm(name, d_in, d_model=max(cfg.d_model, 16))
    return build


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "time2agri_td": ModelSpec(
        "time2agri_td", "pretext", "laptop", "models/time2agri_archs.py (Gupta & Sobti 2025)",
        _time2agri("time2agri_td"), note="TD bin-gap classification; L1 only"),
    "time2agri_fp": ModelSpec(
        "time2agri_fp", "pretext", "laptop", "models/time2agri_archs.py (Gupta & Sobti 2025)",
        _time2agri("time2agri_fp"), note="FP dominant NDVI FFT; train-period targets only; L1 only"),
    "time2agri_ff": ModelSpec(
        "time2agri_ff", "pretext", "laptop", "models/time2agri_archs.py (Gupta & Sobti 2025)",
        _time2agri("time2agri_ff"), note="FF future-frame reconstruction; L1 only"),
    "mlp_ae": ModelSpec("mlp_ae", "reconstruction", "laptop", "new", lambda d, c: _MLPAE(d, c.d_model, c.latent_dim)),
    "gru_ae": ModelSpec("gru_ae", "reconstruction", "laptop", "models/gru_ae.py", _gru),
    "lstm_ae": ModelSpec("lstm_ae", "reconstruction", "laptop", "new", lambda d, c: _LSTMAE(d, c.d_model, c.latent_dim, c.n_layers, c.dropout)),
    "cnn_ae": ModelSpec("cnn_ae", "reconstruction", "laptop", "models/sts_net.py", _cnn),
    "convlstm_ae": ModelSpec("convlstm_ae", "reconstruction", "colab", "new", lambda d, c: _ConvLSTMAE(d, c.d_model, c.latent_dim)),
    "mamba_ae": ModelSpec("mamba_ae", "reconstruction", "colab", "models/cnn_mamba.py (pure-torch SSM)", _mamba),
    "cnn_mamba": ModelSpec("cnn_mamba", "reconstruction", "colab", "models/cnn_mamba.py", _cnn_mamba),
    "barlow_twins": ModelSpec("barlow_twins", "representation", "laptop", "models/rst_bt.py", _barlow),
    "jepa": ModelSpec("jepa", "representation", "colab", "models/jepa.py", _jepa),
}


def build_model(name: str, d_in: int, cfg: ModelConfig) -> nn.Module:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")
    spec = MODEL_REGISTRY[name]
    if not spec.available:
        raise RuntimeError(f"Model {name} unavailable: {spec.note}")
    return spec.builder(d_in, cfg)


def model_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": s.name,
            "family": s.family,
            "placement": s.placement,
            "source": s.source,
            "available": s.available,
            "note": s.note,
        }
        for s in MODEL_REGISTRY.values()
    ]

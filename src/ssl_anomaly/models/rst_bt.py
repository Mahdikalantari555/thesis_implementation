"""Regional Spectral-Temporal Barlow Twins (RST-BT) base model.

A general-purpose base model for precision agriculture that:
- Learns from raw spectral bands (blue, green, red, nir, swir1, swir2, lst)
- Uses causal GRU encoder with temporal Barlow Twins pretext
- Supports few-shot adaptation via prototypical networks
- Handles variable-length sequences (no fixed spatial padding)

Architecture:
- Encoder: 2-layer causal GRU (d_model=96) → masked mean+last pooling → cycle embedding
- Projector: 2-layer MLP (96→192→64) with BatchNorm (Barlow Twins head)
- Reconstruction head: linear (96→7) for auxiliary L1 pretext
- Few-shot head: prototypical network over frozen/partially fine-tuned embeddings
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalGRUEncoder(nn.Module):
    """Causal GRU encoder for irregular spectral-temporal sequences."""

    def __init__(
        self,
        d_in: int = 7,
        d_model: int = 96,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_layers = n_layers
        self.gru = nn.GRU(
            input_size=d_in,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.h0 = nn.Parameter(torch.zeros(n_layers, 1, d_model))

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return hidden states (B, T, d_model)."""
        B, T, _ = x.shape
        h0 = self.h0.expand(self.n_layers, B, self.d_model).contiguous()
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.gru(packed, h0)
            out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.gru(x, h0)
        return out

    def pool(
        self, hidden: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Masked mean + last-valid pooling → (B, d_model)."""
        B, T, D = hidden.shape
        if lengths is not None:
            # Mean over valid positions
            mask = torch.arange(T, device=hidden.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.float().unsqueeze(-1)  # (B, T, 1)
            summed = (hidden * mask).sum(dim=1)  # (B, D)
            mean_pool = summed / lengths.unsqueeze(1).clamp(min=1).float()
            # Last valid
            idx = (lengths - 1).clamp(min=0).long()
            last_pool = hidden[torch.arange(B, device=hidden.device), idx]
            return (mean_pool + last_pool) / 2.0
        else:
            return hidden.mean(dim=1)


class BarlowTwinsProjector(nn.Module):
    """2-layer MLP projector for Barlow Twins loss (batch-size-1 safe, task 2.2).

    The recorded full-field failure was BatchNorm1d raising "Expected more
    than 1 value per channel" on a trailing batch of one window. The guard
    below runs the BN in eval mode (running statistics) for a single-sample
    training batch and restores the mode afterwards — the update proceeds
    with visibly reduced BN fidelity instead of crashing or silently
    dropping the batch.
    """

    def __init__(self, d_in: int = 96, d_hidden: int = 192, d_out: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )
        self.n_batch1_fallbacks = 0  # visible counter, never reset silently

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bn = self.net[1]
        if bn.training and x.shape[0] <= 1:
            self.n_batch1_fallbacks += 1
            bn.eval()
            try:
                return self.net(x)
            finally:
                bn.train()
        return self.net(x)


def barlow_twins_loss(
    z1: torch.Tensor, z2: torch.Tensor, lambda_coeff: float = 0.005
) -> torch.Tensor:
    """Barlow Twins loss: invariance + redundancy reduction."""
    B, D = z1.shape
    # Normalize
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-8)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-8)
    # Cross-correlation matrix
    c = (z1.T @ z2) / B  # (D, D)
    # Invariance: diagonal should be 1
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    # Redundancy: off-diagonal should be 0
    off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum() - on_diag  # simplified
    # Actually compute properly:
    off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z1.device)
    off_diag = c[off_diag_mask].pow(2).sum()
    loss = on_diag + lambda_coeff * off_diag
    return loss


class RSTBTModel(nn.Module):
    """Regional Spectral-Temporal Barlow Twins base model."""

    def __init__(
        self,
        d_in: int = 7,
        d_model: int = 96,
        n_layers: int = 2,
        dropout: float = 0.1,
        proj_hidden: int = 192,
        proj_out: int = 64,
    ) -> None:
        super().__init__()
        self.encoder = CausalGRUEncoder(d_in, d_model, n_layers, dropout)
        self.projector = BarlowTwinsProjector(d_model, proj_hidden, proj_out)
        self.recon_head = nn.Linear(d_model, d_in)

    def encode(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Return cycle embedding (B, d_model)."""
        hidden = self.encoder(x, lengths)
        return self.encoder.pool(hidden, lengths)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        embed = self.encode(x, lengths)
        proj = self.projector(embed)
        recon = self.recon_head(embed).unsqueeze(1).expand(-1, x.shape[1], -1)
        return {"embed": embed, "proj": proj, "recon": recon}

    def barlow_loss(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        len1: torch.Tensor | None = None,
        len2: torch.Tensor | None = None,
        lambda_coeff: float = 0.005,
    ) -> torch.Tensor:
        """Compute Barlow Twins loss on two augmented views."""
        out1 = self.forward(x1, len1)
        out2 = self.forward(x2, len2)
        return barlow_twins_loss(out1["proj"], out2["proj"], lambda_coeff)

    def recon_loss(
        self, x: torch.Tensor, target: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Auxiliary reconstruction loss (L1)."""
        out = self.forward(x, lengths)
        mask = torch.ones_like(target, dtype=torch.bool)
        if lengths is not None:
            for i, L in enumerate(lengths):
                mask[i, L:] = False
        loss = F.l1_loss(out["recon"], target, reduction="none")
        loss = loss * mask.float()
        n = mask.sum().clamp(min=1)
        return loss.sum() / n

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

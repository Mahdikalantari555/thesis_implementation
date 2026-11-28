"""Research-plan arms folded in as screening candidates (tasks 3.4-3.6).

Source of the arms: ``thesis_research/SSL_Proposal/ssl-hierarchical-agri-model/
REPORT.md`` (Level A / Level C / Level E). The fold-in is a RECORDED additive
scope extension (change tasks.md §3.4, recorded 2026-11-28 before
implementation); the frozen pre-registration gates G-A…G-E and the capacity
band are unchanged and remain authoritative.

- ``C-stat`` (Level A): per-field per-date statistical summaries —
  7 bands x {mean, median, std, p10, p25, p75, p90, iqr} + valid_frac +
  n_pixels = **58** features per field-date from
  ``Data/Processed/satellite/stats.zarr``. The research text says "61-dim";
  the actual store yields 58 — the measured number is recorded, the prose is
  not propagated. Cycle embedding = mean of the field-date summaries over the
  cycle's date span. NO learning: the arm is the statistical reference the
  SSL candidates must beat, never a v0 selection candidate.
- ``C-patch`` (Level C): small patched transformer with Δt-aware phase
  encoding. The research ceiling is "≤1.5M params"; the frozen capstone band
  (±20 % of 100 k encoder / 250 k total hard cap) is authoritative, so the
  screening config is sized INTO the band and both numbers are recorded.
- ``C-film`` (Level E): frozen backbone + FiLM-style per-farm adapters
  (γ/β on the trunk sequence + a learned farm embedding). Only adapter
  parameters train; the base is frozen. A farm outside the adapter table is
  an ERROR (recorded absent upstream), never a silent zero adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

#: stats.zarr statistic arrays folded into the summary vector, in order.
SUMMARY_STATS: tuple[str, ...] = (
    "mean", "median", "std", "p10", "p25", "p75", "p90", "iqr",
)

#: 7 bands x 8 statistics + valid_frac + n_pixels = the MEASURED width.
N_SUMMARY_FEATURES = 7 * len(SUMMARY_STATS) + 2

DEFAULT_STATS_ZARR = Path("/home/mahdi/RS/Thesis/Data/Processed/satellite/stats.zarr")


class SummariesUnavailable(RuntimeError):
    """The statistical-summary store is missing or unreadable (fail-closed)."""


# ---------------------------------------------------------------------------
# Level A — C-stat
# ---------------------------------------------------------------------------


def build_cycle_summaries(
    stats_zarr_path: str | Path = DEFAULT_STATS_ZARR,
    *,
    cycles_t_start: np.ndarray,
    cycles_t_end: np.ndarray,
    cycles_field: np.ndarray,
) -> dict:
    """Per-cycle Level-A summary vectors from the stats.zarr store.

    Each cycle is summarised as the per-statistic MEAN over the store dates
    that fall inside ``[t_start, t_end]`` for the cycle's field. Cycles with no
    store date inside their span are recorded ``absent`` (never zero-filled).
    Returns ``{"status": "built", "z": (N, 58), "present": (N,) bool, ...}`` or
    ``{"status": "absent", "reason": ...}``.
    """
    p = Path(stats_zarr_path)
    if not p.exists():
        return {"status": "absent", "reason": f"stats store not found: {p}"}
    try:
        import zarr

        g = zarr.open(str(p), mode="r")
        time_axis = np.asarray(g["time"][:], dtype=np.int64)
        blocks = []
        for stat in SUMMARY_STATS:
            arr = np.asarray(g[stat][:], dtype=np.float64)   # (F, T, 7)
            if arr.shape[2] != 7:
                raise SummariesUnavailable(
                    f"store statistic {stat!r} has {arr.shape[2]} bands, expected 7"
                )
            blocks.append(arr)
        valid_frac = np.asarray(g["valid_frac"][:], dtype=np.float64)[:, :, None]
        n_pixels = np.asarray(g["n_pixels"][:], dtype=np.float64)[:, :, None]
    except Exception as exc:  # noqa: BLE001 - report the dependency, never hide it
        return {"status": "absent", "reason": f"stats store unreadable: {exc}"}

    per_date = np.concatenate(blocks + [valid_frac, n_pixels], axis=2)  # (F, T, 58)
    n = len(cycles_t_start)
    z = np.full((n, N_SUMMARY_FEATURES), np.nan, dtype=np.float64)
    n_dates_used = np.zeros(n, dtype=np.int64)
    for i, (t0, t1, fid) in enumerate(zip(cycles_t_start, cycles_t_end, cycles_field)):
        fi = int(fid)
        if fi < 0 or fi >= per_date.shape[0]:
            continue                       # unknown field: stays NaN, counted below
        sel = (time_axis >= t0) & (time_axis <= t1)
        if sel.any():
            z[i] = per_date[fi][sel].mean(axis=0)
            n_dates_used[i] = int(sel.sum())
    present = np.isfinite(z).all(axis=1) & (n_dates_used > 0)
    return {
        "status": "built",
        "z": z,
        "present": present,
        "n_features": N_SUMMARY_FEATURES,
        "n_dates_used": n_dates_used,
        "store": str(p),
        "feature_names": [f"{s}:{b}" for s in SUMMARY_STATS
                          for b in ("blue", "green", "red", "nir", "swir1", "swir2", "lst")]
        + ["valid_frac", "n_pixels"],
        "note": ("the research text calls this '61-dim'; the store yields "
                 f"{N_SUMMARY_FEATURES} features and the measured number is "
                 "recorded here"),
    }


class SummaryEncoder(nn.Module):
    """Non-learning 'encoder' for C-stat: returns the cycle's summary vector.

    ``embed`` is the identity on the supplied summary matrix; there is no
    trainable path and no pretext — calling ``forward`` or asking to train it
    is refused loudly rather than silently no-op'd.
    """

    def __init__(self, z: np.ndarray, present: np.ndarray | None = None):
        super().__init__()
        z = np.asarray(z, dtype=np.float32)
        if z.ndim != 2:
            raise ValueError(f"summaries must be (N, D), got {z.shape}")
        if not np.isfinite(z).all():
            raise ValueError(
                "summaries contain non-finite rows; drop absent cycles upstream "
                "(absent is never zero-filled)"
            )
        self.register_buffer("z", torch.from_numpy(z))
        self.d_out = int(z.shape[1])
        self.n_cycles = int(z.shape[0])
        self.present = None if present is None else np.asarray(present, dtype=bool)

    @property
    def encoder(self) -> "SummaryEncoder":
        """Capacity accounting reads ``model.encoder``; the arm has NO trainable
        parameters, so the encoder is the (parameter-less) arm itself."""
        return self

    def embed(self, cycle_index: np.ndarray | None = None) -> torch.Tensor:
        if cycle_index is None:
            return self.z
        return self.z[torch.as_tensor(np.asarray(cycle_index, dtype=np.int64))]

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "C-stat is the non-learning statistical reference arm: it has no "
            "pretext and nothing to train. It is probed, never selected as v0."
        )


# ---------------------------------------------------------------------------
# Level C — C-patch
# ---------------------------------------------------------------------------


class PatchTransformer(nn.Module):
    """Small patched transformer with Δt-aware phase encoding (research Level C).

    The sequence is cut into ``patch_len`` bins; each patch token is a linear
    map of its bins' ``[x*m, m]`` pairs, plus the patch-mean of cos/sin of the
    phase axis — the Δt/DSH-aware positional information. Three transformer
    blocks, then: ``embed`` = masked-mean pooled patch tokens projected to
    ``d_out``; ``forward`` = per-bin masked reconstruction via an unpatching
    head (the incumbent-pattern objective, so the same training loop serves).

    Sized for the capstone capacity band (see :func:`patch_capacity`): the
    research's 1.5 M ceiling is recorded, the frozen 250 k cap governs.
    """

    def __init__(self, n_channels: int, *, d_model: int = 64, n_layers: int = 3,
                 n_heads: int = 4, d_ff: int = 128, patch_len: int = 6,
                 d_out: int = 64, dropout: float = 0.0):
        super().__init__()
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.d_model = d_model
        self.d_in = 2 * n_channels + 2          # values + mask + cos/sin(dsh)
        self.patch_embed = nn.Linear(self.patch_len * self.d_in, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.proj = nn.Linear(d_model, d_out)
        # Per-bin reconstruction decodes from the (unpatched) d_model states.
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_channels)
        )

    def _tokens(self, x: torch.Tensor, m: torch.Tensor, dsh: torch.Tensor):
        B, T, C = x.shape
        if T % self.patch_len != 0:
            raise ValueError(
                f"sequence length {T} must be divisible by patch_len "
                f"{self.patch_len} (drop the remainder bins upstream)"
            )
        if dsh.dim() == 1:                       # accept a (T,) phase axis too
            dsh = dsh.unsqueeze(0).expand(B, T)
        mm = m.to(x.dtype)
        xx = torch.where(m, torch.nan_to_num(x, nan=0.0), torch.zeros_like(x))
        phase = (dsh / 540.0) * 2.0 * math.pi
        tok = torch.cat([xx, mm, torch.cos(phase).unsqueeze(-1),
                         torch.sin(phase).unsqueeze(-1)], dim=-1)
        p = tok.reshape(B, T // self.patch_len, self.patch_len * self.d_in)
        return self.patch_embed(p), tok, xx, mm

    def embed(self, x: torch.Tensor, m: torch.Tensor, dsh: torch.Tensor) -> torch.Tensor:
        tokens, _, _, _ = self._tokens(x, m, dsh)
        h = self.blocks(tokens)
        return self.proj(h.mean(dim=1))

    def forward(self, x: torch.Tensor, m: torch.Tensor, dsh: torch.Tensor):
        """Returns ``(per-bin recon (B,T,C), pooled embedding (B,d_out))``."""
        B, T, C = x.shape
        tokens, _, xx, mm = self._tokens(x, m, dsh)
        h = self.blocks(tokens)
        pooled = self.proj(h.mean(dim=1))
        # Unpatch: expand each patch token back to its bins, then decode per bin.
        seq = h.repeat_interleave(self.patch_len, dim=1)[:, :T]
        recon = self.head(seq)
        return recon, pooled


def patch_capacity(n_channels: int = 8, patch_len: int = 6, d_model: int = 64,
                   n_layers: int = 3, d_ff: int = 128, d_out: int = 64,
                   *, target_params: int = 100_000, band: tuple[float, float] = (0.8, 1.2),
                   max_params: int = 250_000) -> dict:
    """Recorded capacity: encoder-side vs total; research ceiling noted."""
    from ssl_anomaly.evaluation.metrics import count_parameters

    model = PatchTransformer(n_channels, patch_len=patch_len, d_model=d_model,
                             n_layers=n_layers, d_ff=d_ff, d_out=d_out)
    total = count_parameters(model)
    enc_only = PatchTransformer(n_channels, patch_len=patch_len, d_model=d_model,
                                n_layers=n_layers, d_ff=d_ff, d_out=d_out)
    enc_only.head = nn.Identity()          # encoder = everything but the recon head
    enc = count_parameters(enc_only)
    lo, hi = band[0] * target_params, band[1] * target_params
    return {
        "encoder_params": int(enc), "total_params": int(total),
        "band": [lo, hi], "band_ok": bool(lo <= enc <= hi),
        "hard_cap_ok": bool(total <= max_params),
        "research_ceiling_params": 1_500_000,
        "note": ("research Level C says '≤1.5M params'; the frozen capstone "
                 "capacity band governs screening and this config sits inside "
                 "it — the ceiling is recorded, not applied"),
    }


# ---------------------------------------------------------------------------
# Level E — C-film
# ---------------------------------------------------------------------------


class FiLMFieldAdapter(nn.Module):
    """Frozen base trunk + FiLM-style per-farm adapters (research Level E).

    γ/β (one pair per farm, on the trunk's hidden sequence) plus a learned farm
    embedding concatenated to the pooled state. The base encoder is frozen at
    construction; ONLY adapter parameters are trainable (checked by a test).
    A farm outside the adapter table raises — never a zero adapter.
    """

    def __init__(self, base: nn.Module, n_farms: int, d_model: int = 32,
                 farm_embed_dim: int = 8, n_channels: int = 8):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)              # frozen backbone: by construction
        self.n_farms = int(n_farms)
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(n_farms, d_model))
        self.beta = nn.Parameter(torch.zeros(n_farms, d_model))
        self.farm_embed = nn.Embedding(n_farms, farm_embed_dim)
        self.d_out = int(d_model + farm_embed_dim)
        self.n_channels = int(n_channels)
        # Per-bin reconstruction head over the FiLM'd sequence — the film-recon
        # pretext (masked-recon objective, same denominator semantics as the
        # other candidates) so the screening loop needs no special case.
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_channels)
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()                         # adapters train; the backbone never does
        return self

    def _farm_tensor(self, farm_ids: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        f = np.asarray(farm_ids)
        if f.shape[0] != like.shape[0]:
            raise ValueError(f"farm ids {f.shape[0]} != batch {like.shape[0]}")
        if (f < 0).any() or (f >= self.n_farms).any():
            bad = sorted({int(v) for v in f[(f < 0) | (f >= self.n_farms)]})
            raise ValueError(
                f"farm ids outside the adapter table: {bad}; a farm without an "
                "adapter is recorded absent upstream, never routed to a zero "
                "adapter"
            )
        t = torch.as_tensor(f, dtype=torch.long, device=like.device)
        return t

    def _film_sequence(self, x: torch.Tensor, m: torch.Tensor, dsh: torch.Tensor,
                       farm_ids: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.base.encoder if hasattr(self.base, "encoder") else self.base
        if dsh.dim() == 1:
            dsh = dsh.unsqueeze(0).expand(x.shape[0], x.shape[1])
        mm = m.to(x.dtype)
        xx = torch.where(m, torch.nan_to_num(x, nan=0.0), torch.zeros_like(x))
        phase = dsh / 540.0
        tok = torch.cat([xx, mm, torch.cos(phase).unsqueeze(-1),
                         torch.sin(phase).unsqueeze(-1)], dim=-1)
        d_in = getattr(enc, "d_in", tok.shape[-1])
        if d_in != tok.shape[-1]:
            raise ValueError(f"base expects d_in={d_in}, got {tok.shape[-1]}")
        seq, _ = enc.gru(tok)
        f = self._farm_tensor(farm_ids, x)
        gamma = self.gamma[f][:, None, :]        # (B, 1, d)
        beta = self.beta[f][:, None, :]
        return gamma * seq + beta, f

    def embed(self, x: torch.Tensor, m: torch.Tensor, dsh: torch.Tensor,
              farm_ids: np.ndarray) -> torch.Tensor:
        seq, f = self._film_sequence(x, m, dsh, farm_ids)
        pooled = seq[:, -1, :]
        return torch.cat([pooled, self.farm_embed(f)], dim=-1)

    def forward(self, x, m, dsh, farm_ids):
        """Per-bin masked reconstruction over the FiLM'd causal sequence."""
        seq, _ = self._film_sequence(x, m, dsh, farm_ids)
        return self.head(seq)

    def adapter_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Registry integration (3.4/3.5)
# ---------------------------------------------------------------------------

#: The FULL screening roster: the four pre-registered candidates keep their
#: order (the tie-break is positional); the research arms join additively
#: AFTER them, in Level order A/C/E.
FULL_CANDIDATE_ORDER: tuple[str, ...] = (
    "C-incumbent", "C-BT", "C-traj", "C-hybrid", "C-stat", "C-patch", "C-film",
)

#: Arms that are NEVER selectable as region model v0 (reported, compared, but
#: outside the SSL selection): C-stat is the statistical reference; the
#: pre-registered selection rule is 'first passing SSL candidate in order'.
NON_SELECTABLE_ARMS: dict[str, str] = {
    "C-stat": ("non-learning statistical reference (Level A): it is probed and "
               "reported as the comparator the SSL arms must beat, never "
               "selected as region model v0"),
}


@dataclass(frozen=True)
class ResearchArmSpec:
    """Declared objective/notes for one research arm (screening contract)."""

    name: str
    level: str                      # 'A' | 'C' | 'E'
    objective: str                  # 'statistical' | 'reconstruction' | 'film-recon'
    trainable: bool
    notes: str


RESEARCH_ARMS: dict[str, ResearchArmSpec] = {
    "C-stat": ResearchArmSpec(
        "C-stat", "A", "statistical", False,
        "58-dim per-field-date summaries from stats.zarr; cycle embedding = "
        "intra-cycle mean; no pretext, no collapse risk; the '61-dim' research "
        "figure does not match the store (58 measured) and the measured number "
        "is recorded",
    ),
    "C-patch": ResearchArmSpec(
        "C-patch", "C", "reconstruction", True,
        "patched transformer, patch_len 6, 3 blocks, Δt-aware phase tokens; "
        "research ceiling 1.5M recorded but the frozen 250k band governs",
    ),
    "C-film": ResearchArmSpec(
        "C-film", "E", "film-recon", True,
        "frozen base trunk (the screened SSL candidate) + per-field FiLM "
        "adapters and farm embeddings; only adapters train; farms outside the "
        "adapter table raise (absent upstream), never a zero adapter",
    ),
}


def build_research_arm(name: str, n_channels: int, *, d_model: int = 32,
                       d_out: int = 32, summaries: dict | None = None,
                       n_farms: int | None = None, base: nn.Module | None = None):
    """Instantiate a research arm; unknown or unusable inputs raise loudly."""
    if name not in RESEARCH_ARMS:
        raise ValueError(f"unknown research arm {name!r}; known: {sorted(RESEARCH_ARMS)}")
    if name == "C-stat":
        if summaries is None or summaries.get("status") != "built":
            raise ValueError(
                "C-stat needs built cycle summaries (build_cycle_summaries); "
                f"got: {None if summaries is None else summaries.get('status')}"
            )
        return SummaryEncoder(summaries["z"], summaries.get("present"))
    if name == "C-patch":
        return PatchTransformer(n_channels, d_model=64, n_layers=3, n_heads=4,
                                d_ff=128, patch_len=6, d_out=d_out)
    if name == "C-film":
        if n_farms is None:
            raise ValueError("C-film needs n_farms (the adapter table size)")
        if base is None:
            raise ValueError("C-film needs a base encoder (the screened trunk)")
        return FiLMFieldAdapter(base, n_farms=n_farms, d_model=d_model)
    raise AssertionError("unreachable")

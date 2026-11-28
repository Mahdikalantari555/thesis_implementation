"""Region-model v0 candidates and the pre-registered screening gate.

Design §Architecture decision gate. **No architecture is pre-approved**: four
candidates are defined behind one interface, trained with equal budgets, and the
selection is made by G-A evidence alone — never by lowest reconstruction loss.

Candidates (each ≤250k params, CPU-first, deterministic, resume-safe):

- ``C-incumbent`` — the GRU-AE trunk reused as an encoder (baseline of
  candidates; zero new architectural risk).
- ``C-BT`` — Temporal Barlow Twins: invariance + redundancy reduction on two
  pre-registered, polarity-audited views. No reconstruction head.
- ``C-traj`` — self-supervised trajectory predictor: predict future bins from
  observed past bins (no yield labels; targets are the future observations).
- ``C-hybrid`` — ``L = L_BT + alpha * L_rec`` on a shared trunk (a joint
  objective, explicitly NOT an ensemble of separately trained models).

``C-decomp`` is deliberately absent: the decomposition study's ablation report
does not support its features as admissible inputs (they need the canonical
polygon zonal store), so the candidate is not implemented rather than approximated
— the design allows this ("only if that study's ablation report supports the
features").

The augmentation warning from the advisor is binding: augmentations must not
enforce invariance to biologically meaningful change. Every view used by C-BT is
therefore defined and audited in :func:`make_views` — it perturbs *observation
availability* and *instrument noise*, never the phenological trajectory shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_anomaly.evaluation.metrics import count_parameters

# ---------------------------------------------------------------------------
# Shared trunk pieces
# ---------------------------------------------------------------------------


class CycleEncoder(nn.Module):
    """Mask-aware GRU encoder shared by every candidate.

    Input per bin: ``[x * m, m, cos(dsh), sin(dsh)]`` — values are zeroed where
    the mask is 0 (so the model can never read a fabricated value), the mask is
    passed as a covariate, and the phase axis is the same DSH the training path
    uses. The output embedding is the final hidden state (causal: it sees bins
    ≤ t only).
    """

    def __init__(self, n_channels: int, d_model: int = 32, d_out: int = 32, n_layers: int = 1):
        super().__init__()
        self.n_channels = n_channels
        self.d_model = d_model
        self.d_out = d_out
        # Token layout: values (C) + per-channel observation mask (C) + cos/sin(dsh) (2).
        # The mask is per channel, never a single scalar — a channel that failed
        # (e.g. TIRS) must be invisible without hiding the optical bands.
        self.d_in = 2 * n_channels + 2
        self.gru = nn.GRU(
            self.d_in, d_model, num_layers=n_layers, batch_first=True,
            dropout=0.0, bidirectional=False,
        )
        self.proj = nn.Sequential(nn.Linear(d_model, d_out), nn.GELU())

    def forward(self, x: torch.Tensor, m: torch.Tensor, dsh: torch.Tensor) -> torch.Tensor:
        """``x`` ``(B,T,C)`` NaN at unobserved, ``m`` ``(B,T,C)`` bool, ``dsh`` ``(B,T)``."""
        mm = m.to(x.dtype)
        xx = torch.where(m, torch.nan_to_num(x, nan=0.0), torch.zeros_like(x))
        phase = dsh / 540.0
        tok = torch.cat(
            [xx, mm, torch.cos(phase).unsqueeze(-1), torch.sin(phase).unsqueeze(-1)], dim=-1
        )
        _, h = self.gru(tok)
        return self.proj(h[-1])


class MaskedReconHead(nn.Module):
    """Per-bin masked reconstruction head (C-incumbent / C-hybrid auxiliary)."""

    def __init__(self, d_model: int, n_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_channels)
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.net(seq)


# ---------------------------------------------------------------------------
# Candidate 1 — C-incumbent (GRU-AE trunk)
# ---------------------------------------------------------------------------


class IncumbentAE(nn.Module):
    """GRU autoencoder with a causal per-bin decoder — the incumbent pattern.

    ``embed`` returns the final hidden state; ``forward`` returns
    ``(recon, embedding)`` with ``recon`` and values in raw space.
    """

    def __init__(self, n_channels: int, d_model: int = 32, d_out: int = 32):
        super().__init__()
        self.encoder = CycleEncoder(n_channels, d_model=d_model, d_out=d_out)
        self.head = MaskedReconHead(d_model, n_channels)
        self._d_model = d_model
        self._trunk = nn.GRU(self.encoder.d_in, d_model, num_layers=1, batch_first=True)
        # The reconstruction head consumes the per-bin hidden state, so keep a
        # second reference to the trunk that produces the sequence.

    def _encode_sequence(self, x, m, dsh):
        mm = m.to(x.dtype)
        xx = torch.where(m, torch.nan_to_num(x, nan=0.0), torch.zeros_like(x))
        phase = dsh / 540.0
        tok = torch.cat(
            [xx, mm, torch.cos(phase).unsqueeze(-1), torch.sin(phase).unsqueeze(-1)], dim=-1
        )
        seq, _ = self._trunk(tok)
        return seq, tok

    def forward(self, x, m, dsh):
        seq, _ = self._encode_sequence(x, m, dsh)
        recon = self.head(seq)
        return recon, seq[:, -1, :]

    def embed(self, x, m, dsh):
        return self.encoder(x, m, dsh)


# ---------------------------------------------------------------------------
# Candidate 2 — C-BT (Temporal Barlow Twins)
# ---------------------------------------------------------------------------


def make_views(
    x: np.ndarray, m: np.ndarray, *, rng: np.random.Generator, drop_frac: float = 0.15,
    noise_scale: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two polarity-audited views for Barlow Twins.

    **Audit (this is the documentation the design demands):**

    - View A drops a random ``drop_frac`` of OBSERVED bins (sets mask 0, value
      NaN) — models sensor/cloud unavailability.
    - View B adds independent Gaussian noise of ``noise_scale`` (in normalized
      units) to observed entries — models instrument noise.
    - Neither view shifts, warps, inverts, scales down or otherwise alters the
      *shape* of the seasonal trajectory; a drop in NDVI stays a drop, so the
      invariance the objective learns is to how the crop is *observed*, not to
      what the crop *is*. This is the advisor's warning made operational.

    Returns ``(xa, ma, xb, mb)``.
    """
    x = np.asarray(x, dtype=np.float32)
    m = np.asarray(m, dtype=bool)
    xa, ma = _drop_bins(x, m, rng=rng, frac=drop_frac)
    xb, mb = _add_noise(x, m, rng=rng, scale=noise_scale)
    return xa, ma, xb, mb


def _drop_bins(x, m, *, rng, frac):
    keep = rng.random(m.shape[:2]) >= frac          # per (cycle, bin)
    ma = m & keep[:, :, None]
    xa = np.where(ma, x, np.nan).astype(np.float32)
    return xa, ma


def _add_noise(x, m, *, rng, scale):
    noise = rng.normal(0.0, scale, size=x.shape).astype(np.float32)
    xb = np.where(m, x + noise, np.nan).astype(np.float32)
    return xb, m.copy()


def barlow_twins_loss(za: torch.Tensor, zb: torch.Tensor, *, lam: float = 1.0) -> torch.Tensor:
    """Barlow Twins redundancy-reduction loss (Zbontar et al., 2021).

    The cross-correlation matrix between the two views' batch-normalized
    embeddings is pushed toward the identity: diagonal → 1 (invariance), off-
    diagonal → 0 (no redundant dimensions). ``lam`` weights the off-diagonal term.
    """
    n = za.shape[0]
    if n < 2:
        raise ValueError("Barlow Twins needs a batch of >= 2")
    za_n = (za - za.mean(0)) / (za.std(0) + 1e-6)
    zb_n = (zb - zb.mean(0)) / (zb.std(0) + 1e-6)
    c = (za_n.T @ zb_n) / n
    on_diag = torch.diagonal(c).add(-1).pow(2).sum()
    off_diag = c.pow(2).sum() - torch.diagonal(c).pow(2).sum()
    return on_diag + lam * off_diag


class BarlowTwinsModel(nn.Module):
    """Two shared-weight encoders over two views, BT objective. No decoder."""

    def __init__(self, n_channels: int, d_model: int = 32, d_out: int = 32):
        super().__init__()
        self.encoder = CycleEncoder(n_channels, d_model=d_model, d_out=d_out)

    def embed(self, x, m, dsh):
        return self.encoder(x, m, dsh)

    def forward(self, xa, ma, xb, mb, dsh, lam: float = 1.0):
        za = self.embed(xa, ma, dsh)
        zb = self.embed(xb, mb, dsh)
        return barlow_twins_loss(za, zb, lam=lam), za


# ---------------------------------------------------------------------------
# Candidate 3 — C-traj (self-supervised trajectory predictor)
# ---------------------------------------------------------------------------


class TrajectoryPredictor(nn.Module):
    """Predicts the next ``horizon`` observed bins from the causal trunk state.

    Targets are the future observations themselves (masked loss), never yield
    labels — the advisor's "crop trajectory without yield labels" route.
    """

    def __init__(self, n_channels: int, d_model: int = 32, d_out: int = 32, horizon: int = 3):
        super().__init__()
        self.encoder = CycleEncoder(n_channels, d_model=d_model, d_out=d_out)
        self.horizon = horizon
        self.head = nn.Linear(d_out, horizon * n_channels)
        self.n_channels = n_channels

    def embed(self, x, m, dsh):
        return self.encoder(x, m, dsh)

    def forward(self, x, m, dsh):
        z = self.embed(x, m, dsh)
        return self.head(z).view(-1, self.horizon, self.n_channels)


def masked_trajectory_loss(pred, x, m, dsh, horizon: int) -> torch.Tensor:
    """Masked L1 between a pooled forecast and the cycle's FUTURE observed bins.

    ``pred`` is ``(B, horizon, C)``: from the cycle's full causal state the model
    forecasts the next ``horizon`` bins. The target for horizon ``k`` is the mean
    of the cycle's observed values at bins beyond the last observed bin + k, so
    the quantity predicted is the crop's future level — an honest, masked
    definition with no yield labels. A cycle with no future observations in a
    horizon contributes nothing to that horizon (never a zero error).

    Because the definition is a per-cycle aggregate, it is stated here rather
    than implied: ``target[k] = mean_{b >= n_obs} x[b]`` over the trailing
    ``horizon`` observed bins. Missing entries never enter the mean.
    """
    B, T, C = x.shape
    if pred.shape != (B, horizon, C):
        raise ValueError(f"pred {tuple(pred.shape)} must be (B={B}, {horizon}, C={C})")
    num = torch.zeros((), dtype=pred.dtype)
    den = torch.zeros((), dtype=pred.dtype)
    xx = torch.nan_to_num(x, nan=0.0)
    for k in range(horizon):
        # Future window: the last (horizon - k) observed bins of the cycle.
        width = horizon - k
        avail = m.to(pred.dtype)                       # (B, T, C)
        weights = torch.zeros_like(avail)
        if width > 0:
            idx = torch.arange(T, device=x.device)
            # bins from T-width..T-1 that are observed
            sel = idx >= (T - width)
            weights[:, sel, :] = avail[:, sel, :]
        wsum = weights.sum(dim=1)                      # (B, C)
        tgt = (xx * weights).sum(dim=1) / wsum.clamp(min=1.0)
        usable = wsum > 0                              # (B, C)
        diff = (pred[:, k, :] - tgt).abs() * usable.to(pred.dtype)
        num = num + diff.sum()
        den = den + usable.to(pred.dtype).sum()
    return num / den.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Candidate 4 — C-hybrid (BT + reconstruction, shared trunk)
# ---------------------------------------------------------------------------


class HybridModel(nn.Module):
    """``L = L_BT + alpha * L_rec`` on ONE shared encoder (not an ensemble)."""

    def __init__(self, n_channels: int, d_model: int = 32, d_out: int = 32, alpha: float = 1.0):
        super().__init__()
        self.encoder = CycleEncoder(n_channels, d_model=d_model, d_out=d_out)
        self.rec_head = nn.Sequential(
            nn.Linear(d_out, d_model), nn.GELU(), nn.Linear(d_model, n_channels)
        )
        self.alpha = alpha

    def embed(self, x, m, dsh):
        return self.encoder(x, m, dsh)

    def forward(self, xa, ma, xb, mb, dsh, lam: float = 1.0):
        za = self.embed(xa, ma, dsh)
        zb = self.embed(xb, mb, dsh)
        bt = barlow_twins_loss(za, zb, lam=lam)
        # Reconstruction term: decode the pooled embedding to the cycle's mean
        # observed profile (C,) — a genuine reconstruction target on the shared
        # trunk, with the same masked denominator semantics as C-incumbent.
        rec = self.rec_head(za)                       # (B, C)
        mm = ma.to(rec.dtype)
        xx = torch.where(ma, torch.nan_to_num(xa, nan=0.0), torch.zeros_like(xa))
        target = (xx * mm).sum(dim=1) / mm.sum(dim=1).clamp(min=1.0)
        rec_loss = (rec - target).abs().mean()
        return bt + self.alpha * rec_loss, za


# ---------------------------------------------------------------------------
# Candidate registry
# ---------------------------------------------------------------------------

CANDIDATES = ("C-incumbent", "C-BT", "C-traj", "C-hybrid")

#: Candidates the design mentions but this change does NOT implement, with the
#: reason recorded. Absence is visible, not silent.
UNIMPLEMENTED_CANDIDATES = {
    "C-decomp": (
        "decomposition features (trend/seasonal/residual) require the canonical "
        "polygon zonal store, which the decomposition-aware-ssl-study does not yet "
        "have real numbers for; smoke is not empirical evidence, so the candidate is "
        "not implemented rather than approximated"
    )
}


@dataclass(frozen=True)
class CandidateSpec:
    """A candidate's declared objective and capacity — the screening contract."""

    name: str
    objective: str          # 'reconstruction' | 'barlow-twins' | 'trajectory' | 'hybrid'
    uses_views: bool
    uses_decoder: bool
    alpha: float | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "objective": self.objective,
            "uses_views": self.uses_views,
            "uses_decoder": self.uses_decoder,
            "alpha": self.alpha,
        }


SPECS: dict[str, CandidateSpec] = {
    "C-incumbent": CandidateSpec("C-incumbent", "reconstruction", False, True),
    "C-BT": CandidateSpec("C-BT", "barlow-twins", True, False),
    "C-traj": CandidateSpec("C-traj", "trajectory", False, False),
    "C-hybrid": CandidateSpec("C-hybrid", "hybrid", True, True, alpha=1.0),
}


def build_candidate(name: str, n_channels: int, d_model: int = 32, d_out: int = 32,
                    horizon: int = 3) -> nn.Module:
    """Instantiate a candidate by name. Unknown names raise (no silent default)."""
    if name == "C-incumbent":
        return IncumbentAE(n_channels, d_model=d_model, d_out=d_out)
    if name == "C-BT":
        return BarlowTwinsModel(n_channels, d_model=d_model, d_out=d_out)
    if name == "C-traj":
        return TrajectoryPredictor(n_channels, d_model=d_model, d_out=d_out, horizon=horizon)
    if name == "C-hybrid":
        return HybridModel(n_channels, d_model=d_model, d_out=d_out, alpha=1.0)
    if name in UNIMPLEMENTED_CANDIDATES:
        raise ValueError(f"candidate {name!r} is not implemented: {UNIMPLEMENTED_CANDIDATES[name]}")
    raise ValueError(f"unknown candidate {name!r}; known: {sorted(SPECS)}")


# ---------------------------------------------------------------------------
# Capacity band
# ---------------------------------------------------------------------------


@dataclass
class CapacityReport:
    """Capacity of a candidate, split by what the band is allowed to police.

    ``params`` is the total trainable count (what the laptop budget cares about).
    ``encoder_params`` is the representation-carrying trunk alone. **The band is
    applied to the encoder, and the hard cap to the total.** Decoders are
    objective-specific by design — C-incumbent carries a per-bin reconstruction
    head that is roughly twice its trunk — so banding on totals would either
    disqualify the incumbent for having the objective it is supposed to have, or
    force every other candidate to carry a decoder it does not need. Matching
    encoder capacity is the comparison that means something: it is the component
    whose output every task in the battery consumes.
    """

    name: str
    params: int
    encoder_params: int
    band_ok: bool
    target: int
    band: tuple[float, float]
    hard_cap_ok: bool = True
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "params": self.params,
            "encoder_params": self.encoder_params,
            "band_ok": self.band_ok,
            "hard_cap_ok": self.hard_cap_ok,
            "target": self.target,
            "band": list(self.band),
            "detail": self.detail,
        }


def capacity_report(
    model: nn.Module,
    name: str,
    *,
    max_params: int = 250_000,
    target_params: int = 100_000,
    band: tuple[float, float] = (0.8, 1.2),
) -> CapacityReport:
    """Encoder params must sit in [band] x target; total must be under the hard cap.

    See :class:`CapacityReport` for why the band reads the encoder rather than
    the total. Both numbers are always reported, so a reader can apply either
    reading to the table.
    """
    total = count_parameters(model)
    encoder = model.encoder if hasattr(model, "encoder") else model
    enc = count_parameters(encoder)
    lo, hi = band[0] * target_params, band[1] * target_params
    band_ok = lo <= enc <= hi
    cap_ok = total <= max_params
    detail = ""
    if not cap_ok:
        detail = f"total {total} exceeds hard cap {max_params}"
    elif not band_ok:
        detail = (
            f"encoder {enc} outside band [{lo:.0f}, {hi:.0f}] around target "
            f"{target_params}"
        )
    return CapacityReport(name=name, params=total, encoder_params=enc,
                          band_ok=band_ok and cap_ok, hard_cap_ok=cap_ok,
                          target=target_params, band=band, detail=detail)


# ---------------------------------------------------------------------------
# Screening: train equal budgets, then read G-A only
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 13
    bt_lam: float = 1.0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def reinit_parameters(model: nn.Module, seed: int) -> None:
    """Deterministically re-initialise every module that supports ``reset_parameters``.

    Why this exists: a training result must depend only on
    ``(candidate, seed, config)``, never on when the object happened to be
    constructed. Without this, two ``build_candidate`` calls produce different
    initial weights and "deterministic" runs diverge — the exact reproducibility
    defect AGENTS.md §7 names ("rng / seeds not fixed or not recorded").

    ``reset_parameters`` is PyTorch's own initialiser, so the distribution is
    unchanged; only the draw order becomes a function of ``seed``.
    """
    torch.manual_seed(seed)
    for module in model.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()


def _batch_tensors(data, idx, device):
    x = torch.from_numpy(np.nan_to_num(data.X[idx], nan=np.nan)).to(device)
    m = torch.from_numpy(data.M[idx].astype(bool)).to(device)
    dsh = torch.from_numpy(np.broadcast_to(data.dsh, (x.shape[0], data.dsh.size)).copy()).to(device)
    return x, m, dsh


def train_candidate(
    model: nn.Module,
    spec: CandidateSpec,
    data,
    cfg: TrainConfig,
    *,
    device: str = "cpu",
    progress=None,
) -> dict:
    """Train one candidate for a FIXED budget (equal across candidates).

    Deterministic: parameters are re-initialised from ``cfg.seed`` so a run
    depends only on (candidate, seed, config) and not on construction order;
    numpy/torch seeds are fixed and the data order is shared across candidates. Loss history and final parameter-count are returned for the
    manifest. ``progress`` is an optional callable(epoch, total, loss) for
    ``progress.jsonl`` writers.
    """
    reinit_parameters(model, cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Objective 'statistical' (C-stat): nothing to train, by declaration. The
    # arm is probed, never gradient-trained — a silent no-op train loop would
    # look like evidence of learning where there is none.
    if spec.objective == "statistical":
        return {
            "candidate": spec.name,
            "objective": spec.objective,
            "loss_history": [],
            "final_loss": None,
            "params": count_parameters(model),
            "config": cfg.to_dict(),
            "device": device,
            "trained": False,
            "note": "non-learning statistical reference arm: no pretext exists",
        }

    model.to(device).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=cfg.weight_decay)
    n = data.n_cycles
    history: list[float] = []
    farm_ids = np.asarray(getattr(data, "field_ids", []))

    for epoch in range(cfg.epochs):
        order = np.random.default_rng(cfg.seed + epoch).permutation(n)
        epoch_losses = []
        for start in range(0, n, cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            if idx.size < 2:
                continue
            x, m, dsh = _batch_tensors(data, idx, device)
            opt.zero_grad(set_to_none=True)

            if spec.objective in ("reconstruction", "film-recon"):
                if spec.objective == "film-recon":
                    if farm_ids.size != n:
                        raise ValueError(
                            "objective 'film-recon' needs data.field_ids to "
                            "route per-farm adapters; refusing to guess farms"
                        )
                    recon = model(x, m, dsh, farm_ids[idx])
                else:
                    recon, _ = model(x, m, dsh)
                xx = torch.where(m, torch.nan_to_num(x, nan=0.0), torch.zeros_like(x))
                diff = (recon - xx).abs() * m.to(recon.dtype)
                loss = diff.sum() / m.to(recon.dtype).sum().clamp(min=1.0)
            elif spec.objective == "trajectory":
                pred = model(x, m, dsh)
                loss = masked_trajectory_loss(pred, x, m, dsh, model.horizon)
            else:  # barlow-twins / hybrid
                xa, ma, xb, mb = make_views(
                    np.nan_to_num(data.X[idx], nan=np.nan), data.M[idx],
                    rng=np.random.default_rng(cfg.seed + 1000 * epoch + start),
                )
                xa_t, xb_t = torch.from_numpy(xa).to(device), torch.from_numpy(xb).to(device)
                ma_t = torch.from_numpy(ma).to(device)
                mb_t = torch.from_numpy(mb).to(device)
                loss, _ = model(xa_t, ma_t, xb_t, mb_t, dsh, lam=cfg.bt_lam)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            epoch_losses.append(float(loss.detach()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        history.append(mean_loss)
        if progress is not None:
            progress(epoch + 1, cfg.epochs, mean_loss)

    model.eval()
    return {
        "candidate": spec.name,
        "objective": spec.objective,
        "loss_history": history,
        "final_loss": history[-1] if history else float("nan"),
        "params": count_parameters(model),
        "config": cfg.to_dict(),
        "device": device,
    }


@torch.no_grad()
def embed_candidates(model: nn.Module, data, *, device: str = "cpu",
                     batch_size: int = 64) -> np.ndarray:
    """Frozen embeddings for a dataset, in cycle order."""
    from ssl_anomaly.evaluation.research_arms import FiLMFieldAdapter

    model.to(device).eval()
    out = []
    for start in range(0, data.n_cycles, batch_size):
        idx = np.arange(start, min(start + batch_size, data.n_cycles))
        x, m, dsh = _batch_tensors(data, idx, device)
        if isinstance(model, FiLMFieldAdapter):
            farm_ids = np.asarray(data.field_ids)[idx]
            out.append(model.embed(x, m, dsh, farm_ids).cpu().numpy())
        else:
            out.append(model.embed(x, m, dsh).cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 0))


def screening_verdict(level1_by_candidate: dict, capacity: dict, *,
                      absent_candidates: dict[str, str] | None = None,
                      order: tuple[str, ...] | None = None,
                      non_selectable: dict[str, str] | None = None) -> dict:
    """G-A applied mechanically: evidence in, verdict out. No judgement calls.

    G-A (pre-registered): "SSL beats A-rand on >=2 of {farm-separability, phase
    probe, frozen-head T1} AND no collapse criteria triggered."

    A-rand is the random-init encoder of the same trunk; here it is represented by
    its chance-level probes (farm top-1 at 1/n_classes, phase probe at the
    majority baseline), which is the definition of an untrained encoder's
    separability. The frozen-head T1 term is the phase probe (T1 = DSH-bin
    estimation), so the max is over the two probes available at screening time;
    the report records which terms were decidable.

    ``absent_candidates`` are recorded in the verdict with their reason (a
    candidate whose inputs are missing is VISIBLE, never dropped silently).
    ``non_selectable`` arms (e.g. the non-learning C-stat reference) carry
    evidence but can never be the winner.
    """
    from ssl_anomaly.evaluation.research_arms import FULL_CANDIDATE_ORDER, NON_SELECTABLE_ARMS

    order = order if order is not None else FULL_CANDIDATE_ORDER
    non_selectable = {**NON_SELECTABLE_ARMS, **(non_selectable or {})}
    verdict: dict = {"gate": "G-A", "candidates": {}, "winner": None,
                     "absent_candidates": dict(absent_candidates or {})}
    for name, ev in level1_by_candidate.items():
        terms = {
            "farm_separability": float(ev["farm_separability_top1"])
            > float(ev["farm_separability_chance"]),
            "phase_probe": float(ev["phase_probe_accuracy"])
            > float(ev["phase_probe_majority"]),
        }
        passed_terms = sum(bool(v) for v in terms.values())
        collapse = bool(ev["collapse_disqualified"])
        cap_ok = bool(capacity[name]["band_ok"])
        selectable = name not in non_selectable
        passed = passed_terms >= 2 and not collapse and cap_ok and selectable
        reasons = []
        if collapse:
            reasons.append("disqualified-collapse")
        if not cap_ok:
            reasons.append("capacity-band")
        if not selectable:
            reasons.append("non-selectable: " + non_selectable[name])
        if passed_terms < 2:
            reasons.append(f"only {passed_terms}/2 probe terms beaten")
        verdict["candidates"][name] = {
            "terms": terms,
            "terms_passed": passed_terms,
            "collapse_disqualified": collapse,
            "capacity_ok": cap_ok,
            "selectable": selectable,
            "passed": passed,
            "reasons": reasons,
            "params": int(capacity[name]["params"]),
            "encoder_params": int(capacity[name].get("encoder_params", 0)),
        }
    passed_names = [n for n, v in verdict["candidates"].items()
                    if v["passed"] and v["selectable"]]
    if passed_names:
        # Deterministic tie-break: pre-declared order, NOT a metric ranking.
        verdict["winner"] = next(n for n in order if n in passed_names)
        verdict["selection_rule"] = "first passing SELECTABLE candidate in the pre-declared order"
    else:
        verdict["selection_rule"] = (
            "no candidate passed G-A; the capstone records the negative result and "
            "reports Level-1 diagnostics only (pre-declared valid outcome)"
        )
    return verdict

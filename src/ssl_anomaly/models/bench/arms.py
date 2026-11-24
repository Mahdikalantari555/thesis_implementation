"""Benchmark architectures A1–A10: five paradigms × two backbones (Phase 2).

Every arm here is one trunk with a different *estimand*, never a re-labelled loss:

=====  ==========================  =========================================  ================
id     paradigm                    objective                                   target
=====  ==========================  =========================================  ================
A1/A2  input_reconstruction        masked Huber in input space                the bin itself,
                                                                                 from its PAST only
A3/A4  latent_reconstruction       masked Huber in latent space               the same bin's
                                                                                 EMA-target latent
A5/A6  future_prediction           masked Huber of a k-step forecast          future bins
A7/A8  jepa                        latent forecast of future bins             future-bin latents
                                                                                 from the EMA target
A9/A10 hybrid                     λ_r · L_rec + λ_p · L_pred                  both, weighted
=====  ==========================  =========================================  ================

Three things are structurally enforced rather than promised:

* **Causality.** Every trunk is causal; the reconstruction arms read
  ``h_{t-1}`` to estimate bin ``t``, so no arm's score at ``t`` can depend on
  ``>t``. The prediction arms keep a learned initial state for bin 0 (which has no
  past) and score bin 0 as ``NaN`` for the forecast streams — an unscorable bin is
  absent, not perfect.
* **Collapse prevention where it is required.** A3/A4/A7/A8 carry an EMA target
  encoder plus a variance-preservation term, and both live in the checkpoint.
  A stop-gradient alone does not stop a *joint* collapse, so promising "stop-grad"
  without a variance floor would be documentation, not a mechanism.
* **Objective audit.** :meth:`objective_spec` is machine-read by
  :func:`ssl_anomaly.benchmarking.interface.assert_objectives_distinct` before a
  cycle starts, so a refactor that quietly makes two paradigms compute the same
  thing fails loudly.

The A1 note is worth a sentence in the thesis: the frozen ``gru_ae_main`` trunk is
a *causal leave-one-bin-out* model, i.e. its "reconstruction" of bin ``t`` is a
one-step forecast of bin ``t``. A1 keeps that contract because it is the incumbent
being challenged, and A5 differs from it by targeting ``t+1…t+k`` with ``k > 1``
and by scoring a bin from its *origins* rather than from its own position. The two
are close relatives on purpose; pretending otherwise would make the paradigm
comparison meaningless.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssl_anomaly.benchmarking.interface import ArmBatch, BaseArm
from ssl_anomaly.models.bench.backbones import (  # noqa: F401  (re-exported for reports/tests)
    BackboneOutput,
    CausalStackBackbone,
    NumericalInstability,
    build_backbone,
)

#: The five paradigms, in the design's order.
PARADIGM_ORDER = (
    "input_reconstruction",
    "latent_reconstruction",
    "future_prediction",
    "jepa",
    "hybrid",
)


# ---------------------------------------------------------------------------
# Shared math
# ---------------------------------------------------------------------------


def masked_bin_error(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, beta: float = 1.0
) -> torch.Tensor:
    """``(B,T,C) → (B,T)`` mean Huber over OBSERVED channels; NaN where none observed.

    The denominator is the per-bin observed-channel count, never the full channel
    count: dividing a one-channel bin by 8 would score a partially observed bin as
    "close to reconstruction" purely because seven channels were missing.
    """
    e = F.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    mf = mask.to(e.dtype)
    den = mf.sum(-1)
    err = (e * mf).sum(-1) / den.clamp(min=1.0)
    return err.masked_fill(den == 0, float("nan"))


def masked_mean(err: torch.Tensor, valid: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    """Mean of a ``(B,T)`` (or ``(B,T,k)``) error over valid, non-NaN entries.

    Suppression weights multiply numerator and denominator together, so a
    down-weighted bin cannot be revived by the per-sample averaging step.
    """
    finite = torch.isfinite(err)
    keep = finite & valid.to(finite.dtype).bool()
    w = keep.to(err.dtype)
    if weights is not None:
        ww = weights.to(err.dtype)
        w = w * ww if ww.dim() == w.dim() else w * ww.unsqueeze(-1)
    den = w.sum().clamp(min=1.0)
    clean = torch.where(keep, torch.nan_to_num(err, nan=0.0), torch.zeros_like(err))
    return (clean * w).sum() / den


def target_centric(per_origin: torch.Tensor, scored: torch.Tensor) -> torch.Tensor:
    """``(B,T,k)`` origin-anchored errors → ``(B,T)`` score of the bin being targeted.

    ``per_origin[:, b, h]`` forecasts bin ``b+h+1``. The returned tensor is the mean
    over every origin whose horizon lands on bin ``t`` — the multi-origin consensus
    a forecaster would actually use at inference. Bin 0 has no origin, so it is
    NaN by construction, and the caller must not fill it.
    """
    B, T, k = per_origin.shape
    dev, dt = per_origin.device, per_origin.dtype
    num = torch.zeros(B, T, device=dev, dtype=dt)
    den = torch.zeros(B, T, device=dev, dtype=dt)
    clean = torch.nan_to_num(per_origin, nan=0.0)
    for h in range(int(k)):
        tgt = slice(h + 1, T)
        src = slice(0, T - h - 1)
        n = T - h - 1
        if n <= 0:
            continue
        ok = scored[:, src, h]
        num[:, tgt] = num[:, tgt] + clean[:, src, h] * ok.to(dt)
        den[:, tgt] = den[:, tgt] + ok.to(dt)
    out = num / den.clamp(min=1.0)
    return out.masked_fill(den == 0, float("nan"))


def future_targets(
    x: torch.Tensor, m: torch.Tensor, k: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack bins ``t+1 … t+k`` as targets (and their masks) for origin ``t``.

    Positions with no future inside the cycle carry mask 0, so they contribute to
    neither numerator nor denominator. Edge-padding instead would teach the arm that
    a cycle's last bin repeats — an artefact of the 15-day bin grid, not a fact.
    """
    B, T, C = x.shape
    pad = torch.zeros(B, k, C, dtype=x.dtype, device=x.device)
    pad_m = torch.zeros(B, k, C, dtype=m.dtype, device=m.device)
    xp = torch.cat([x, pad], dim=1)
    mp = torch.cat([m, pad_m], dim=1)
    vs = [xp[:, 1 + i : 1 + i + T] for i in range(k)]
    ms = [mp[:, 1 + i : 1 + i + T] for i in range(k)]
    return torch.stack(vs, dim=2), torch.stack(ms, dim=2)


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class _Arm(BaseArm):
    """Constructor plumbing shared by the five paradigms."""

    #: Which sub-modules an arm actually uses. Declared per class so the parameter
    #: count in the fairness table counts what the objective consumes.
    uses_latent: bool = False
    uses_mask_token: bool = True
    uses_loo: bool = True

    def __init__(
        self,
        *,
        d_channels: int,
        d_cov: int = 0,
        n_bins: int = 36,
        d_model: int = 96,
        n_layers: int = 1,
        dropout: float = 0.1,
        d_state: int = 16,
        latent_dim: int = 32,
        huber_beta: float = 1.0,
        horizon_k: int = 4,
        ema_momentum: float = 0.996,
        variance_weight: float = 0.05,
        lambda_rec: float = 1.0,
        lambda_pred: float = 1.0,
        backbone: str = "gru",
        causal: bool = True,
        bottleneck_frac: float = 0.25,
        mask_rate: float = 0.0,
        masking_applied_at: str = "train_only",
    ) -> None:
        super().__init__()
        self.d_channels = int(d_channels)
        self.d_cov = int(d_cov)
        self.n_bins = int(n_bins)
        self.huber_beta = float(huber_beta)
        self.horizon_k = max(1, int(horizon_k))
        self.ema_momentum = float(ema_momentum)
        self.variance_weight = float(variance_weight)
        self.lambda_rec = float(lambda_rec)
        self.lambda_pred = float(lambda_pred)
        self.backbone = backbone
        self.causal = bool(causal)
        self.bottleneck_frac = float(bottleneck_frac)
        self.mask_rate = float(mask_rate)
        self.d_in = self.d_channels + self.d_channels + self.d_cov
        self.trunk = build_backbone(
            self.d_in,
            kind=backbone,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            d_state=d_state,
            causal=causal,
            for_arm=self.model_id,
        )
        self._d_hidden = int(self.trunk.d_out)
        self._d_state = int(d_state)
        self._d_model = int(d_model)
        self._n_layers = int(n_layers)
        self._dropout = float(dropout)
        self.latent_dim = int(latent_dim)
        # Heads are built per paradigm, never "all of them, unused": a trainable
        # parameter that receives no gradient inflates the capacity comparison the
        # spec requires, and a dead module is also a dead early-stopping signal.
        self.lat_head = nn.Linear(self._d_hidden, self.latent_dim) if self.uses_latent else None
        if self.uses_mask_token:
            # A learned stand-in for "this bin exists but its value was hidden by the
            # masking objective". Kept separate from the unobserved encoding for the
            # reason given in ArmBatch's docstring.
            self.mask_token = nn.Parameter(torch.zeros(self.d_in))
        else:
            self.register_buffer("mask_token", torch.zeros(self.d_in))
        # Position 0 has no past; a learned prior state is the incumbent's answer,
        # and it is what keeps bin 0 scorable for the reconstruction arms.
        self.h0 = nn.Parameter(torch.zeros(1, 1, self._d_hidden)) if self.uses_loo else None
        self.masking_applied_at = str(masking_applied_at)
        self._arch_kwargs = {
            "d_channels": self.d_channels,
            "d_cov": self.d_cov,
            "n_bins": self.n_bins,
            "d_model": self._d_model,
            "n_layers": self._n_layers,
            "dropout": self._dropout,
            "d_state": self._d_state,
            "latent_dim": self.latent_dim,
            "huber_beta": self.huber_beta,
            "horizon_k": self.horizon_k,
            "ema_momentum": self.ema_momentum,
            "variance_weight": self.variance_weight,
            "lambda_rec": self.lambda_rec,
            "lambda_pred": self.lambda_pred,
            "backbone": self.backbone,
            "causal": self.causal,
            "bottleneck_frac": self.bottleneck_frac,
            "mask_rate": self.mask_rate,
        }

    # ---- plumbing --------------------------------------------------------
    def own_context_mask(self, batch: ArmBatch) -> torch.Tensor | None:
        """This arm's masked-by-construction flags (train mode only).

        Arms whose objective is *masked* by construction hide a fraction of bins at
        train time; masking is never applied at eval time, because the score must
        read the observation pattern the archive actually has.
        """
        if self.mask_rate <= 0.0 or not self.training:
            return None
        B, T = batch.valid_bin.shape
        draw = (torch.rand(B, T, device=batch.x.device) < self.mask_rate) & batch.valid_bin
        keep = batch.valid_bin.sum(-1, keepdim=True)
        over = draw.cumsum(-1) > (keep - 1).clamp(min=0)
        return draw & ~over

    def tokens(self, batch: ArmBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the trunk's input, honouring both kinds of "hidden bin".

        A hidden bin has its VALUE channels zeroed and the learned mask token
        substituted in, while its OBSERVATION flags stay intact: "the archive has
        this bin but the model may not look at it" and "the archive does not have
        this bin" are different statements, and an arm that cannot tell them apart
        cannot separate a cloud gap from a beetle outbreak.
        """
        cm = self.own_context_mask(batch)
        if self.suppressor is not None:
            extra = self.suppressor.context_mask(batch, enabled=bool(self.training))
            # Either side may legitimately be absent: S1/S3 mask nothing, and a
            # paradigm with rate 0 (A1/A9 input-space reconstruction) masks nothing
            # either. ``cm | None`` used to be a TypeError on the first S1 cell.
            if extra is not None:
                cm = extra if cm is None else (cm | extra)
        if cm is None:
            return batch.input(), batch.valid_bin
        keep = (~cm).to(batch.x.dtype).unsqueeze(-1)
        parts = [batch.x * keep, batch.m.to(batch.x.dtype)]
        if batch.cov is not None:
            parts.append(batch.cov)
        tok = torch.cat(parts, dim=-1)
        flag = cm.to(tok.dtype).unsqueeze(-1)
        return tok * (1.0 - flag) + self.mask_token.to(tok.dtype) * flag, batch.valid_bin

    def encode(self, batch: ArmBatch) -> BackboneOutput:
        tok, valid = self.tokens(batch)
        return self.trunk(tok, valid)

    def prepare(self, batch: ArmBatch, weights: torch.Tensor | None):
        """Apply the attached suppression arm, then return ``(batch, weights)``.

        S2 (and this arm's own masked-by-construction rate) sets ``ctx_mask``; S1/S3
        supply per-bin loss weights. Multiplying the two is intentional: a cell can
        in principle carry both, and a silent "last one wins" would make the
        combination depend on statement order.
        """
        cm = self.suppression_mask(batch)
        if cm is not None:
            batch = ArmBatch(
                x=batch.x, m=batch.m, valid_bin=batch.valid_bin, cov=batch.cov,
                ctx_mask=cm, meta=batch.meta,
            )
        w = self.suppression_weights(batch)
        if weights is not None:
            w = weights if w is None else w * weights
        return batch, w

    def prev_hidden(self, out: BackboneOutput) -> torch.Tensor:
        """``h_{t-1}`` (learned ``h0`` for ``t = 0``) — the leave-one-out context."""
        B = out.hidden.shape[0]
        return torch.cat([self.h0.expand(B, 1, -1), out.hidden[:, :-1, :]], dim=1).contiguous()

    def _ema_copy(self, module: nn.Module) -> nn.Module:
        dup = copy.deepcopy(module)
        dup.requires_grad_(False)
        return dup

    @torch.no_grad()
    def _ema_update(self, source: nn.Module, target: nn.Module) -> None:
        m = self.ema_momentum
        for ps, pt in zip(source.parameters(), target.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bs, bt in zip(source.buffers(), target.buffers()):
            bt.copy_(bs)

    @staticmethod
    def variance_term(z: torch.Tensor) -> torch.Tensor:
        """``mean_j relu(1 − std_j(z))`` over the latent dims — the anti-collapse floor.

        A stop-gradient EMA target makes the constant solution *attractive*, not
        merely allowed; this term is what actually forbids it, and its value is
        logged so a collapsed arm is visible in ``progress.jsonl`` and not only in
        the post-hoc diagnostics.
        """
        if z.numel() == 0:
            return z.new_zeros(())
        std = z.float().std(dim=0, unbiased=False)
        return F.relu(1.0 - std).mean()

    def _finite_guard(self, *tensors: torch.Tensor, allow_nan: bool = False) -> None:
        """Stop on `inf`; optionally also on NaN.

        Score tensors legitimately contain NaN (``"this bin has nothing observed to
        compare against"``), so the guard for them checks infinities only — an
        all-NaN score is caught downstream by the scorer's own all-unscorable check.
        Every other tensor must be fully finite, which is what catches a diverging
        selective scan before it reaches a metric.
        """
        for i, t in enumerate(tensors):
            if t is None:
                continue
            bad = (~torch.isfinite(t)) if allow_nan else (~torch.isfinite(t))
            if allow_nan:
                bad = torch.isinf(t)
            if not bool(bad.any()):
                continue
            raise NumericalInstability(
                f"{self.model_id}: {'non-finite' if not allow_nan else 'infinite'} values in "
                f"tensor #{i} of the forward pass",
                evidence={
                    "model_id": self.model_id,
                    "tensor_index": i,
                    "shape": list(t.shape),
                    "n_offending": int(bad.sum()),
                    "guard": "inf_only" if allow_nan else "all_finite",
                },
            )

    # ---- defaults --------------------------------------------------------
    def collapse_state(self) -> dict:
        return {}

    def load_extra_checkpoint_state(self, state: dict) -> None:
        return None

    def score_streams(self, batch: ArmBatch) -> dict[str, torch.Tensor]:
        """Per-stream errors, when an arm has more than one (the hybrid)."""
        return {"err": self.score_bins(batch)}


# ---------------------------------------------------------------------------
# A1 / A2 — input reconstruction
# ---------------------------------------------------------------------------


class InputReconstructionArm(_Arm):
    """``x̂_t = D(h_{t-1})``; ``L = ‖m ⊙ (x − x̂)‖₁`` — the incumbent's contract."""

    model_id = "A1"
    paradigm = "input_reconstruction"
    supports_suppression = ("S0", "S1", "S2", "S3")
    #: A1/A2 are UNMASKED by construction (pure leave-one-bin-out reconstruction);
    #: masking enters only through S2. The other paradigms are masked-by-design, so
    #: their mask token is a live parameter and theirs is not.
    uses_mask_token = False

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.rec_head = nn.Sequential(
            nn.Linear(self._d_hidden, self._d_hidden), nn.GELU(),
            nn.Linear(self._d_hidden, self.d_channels),
        )
        self._objective = {
            "objective": "masked_huber_input_space",
            "target": "observed_values_of_the_current_bin",
            "target_source": "batch.x at bin t, predicted from h_{<t} only",
            "gradient_path": "trunk + rec_head; single branch, no target encoder",
            "collapse_mechanism": None,
        }

    def forward(self, batch: ArmBatch) -> dict:  # type: ignore[override]
        out = self.encode(batch)
        recon = self.rec_head(self.prev_hidden(out))
        err = masked_bin_error(recon, batch.x, batch.m, beta=self.huber_beta)
        self._finite_guard(recon)
        self._finite_guard(err, allow_nan=True)
        return {"recon": recon, "err": err, "hidden": out.hidden, "pooled": out.pooled}

    def compute_loss(self, batch: ArmBatch, *, weights: torch.Tensor | None = None):
        batch, weights = self.prepare(batch, weights)
        o = self.forward(batch)
        loss = masked_mean(o["err"], batch.valid_bin, weights)
        return loss, {
            "rec_loss": float(loss.detach()),
            "n_scored_bins": int(batch.valid_bin.sum()),
            "suppression": self._suppression,
        }

    def score_bins(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(batch)["err"]

    def probe_features(self, batch: ArmBatch) -> torch.Tensor:
        return self.encode(batch).hidden   # (B, T, D): the probe picks granularity

    def objective_spec(self) -> dict:
        return {**super().objective_spec(), "lambdas": {}, "horizon_k": 0}


# ---------------------------------------------------------------------------
# A3 / A4 — latent reconstruction
# ---------------------------------------------------------------------------


class LatentReconstructionArm(_Arm):
    """``z = E(x)``, ``ẑ = Dec(Bott(z))``, ``L = ‖sg(z⁺) − ẑ‖² + λ_v · var(ẑ)``.

    Same-bin reconstruction, but in representation space: a bin the encoder can
    describe from its own representation costs nothing, one it cannot is the
    signal. The bottleneck (``latent_dim × bottleneck_frac``) is what keeps it from
    being an identity map, and the EMA target plus variance term is what keeps the
    bottleneck from being solved by a constant.
    """

    model_id = "A3"
    paradigm = "latent_reconstruction"
    supports_suppression = ("S0", "S1", "S2", "S3")
    uses_latent = True
    uses_loo = False        # same-bin reconstruction reads h_t, not h_{t-1}

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        d_b = max(4, int(round(self.latent_dim * self.bottleneck_frac)))
        self.bottleneck = d_b
        self.lat_enc = nn.Linear(self.latent_dim, d_b)
        self.lat_dec = nn.Sequential(
            nn.Linear(d_b, self.latent_dim), nn.GELU(), nn.Linear(self.latent_dim, self.latent_dim)
        )
        self.target_lat_head = self._ema_copy(self.lat_head)
        self.target_trunk = self._ema_copy(self.trunk)
        self._objective = {
            "objective": "masked_huber_latent_space_same_bin",
            "target": "ema_target_latent_of_the_current_bin",
            "target_source": "EMA copy of trunk+lat_head over the same bins, stop-gradient",
            "gradient_path": "trunk + lat_head + lat_enc + lat_dec; target branch frozen",
            "collapse_mechanism": "ema_target_encoder+variance_preservation",
        }

    @torch.no_grad()
    def target_latents(self, batch: ArmBatch) -> torch.Tensor:
        tok, valid = self.tokens(batch)
        return self.target_lat_head(self.target_trunk(tok, valid).hidden)

    def _predict(self, batch: ArmBatch) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encode(batch)
        z = self.lat_head(out.hidden)
        z_hat = self.lat_dec(F.gelu(self.lat_enc(z)))
        return z, z_hat

    def forward(self, batch: ArmBatch) -> dict:  # type: ignore[override]
        z, z_hat = self._predict(batch)
        ref = self.target_latents(batch)
        err = (z_hat - ref).pow(2).mean(-1)
        err = err.masked_fill(~batch.valid_bin, float("nan"))
        self._finite_guard(z_hat)
        self._finite_guard(err, allow_nan=True)
        return {"z": z, "z_hat": z_hat, "target": ref, "err": err,
                "variance": self.variance_term(z_hat[batch.valid_bin]) if bool(batch.valid_bin.any()) else z.new_zeros(())}

    def compute_loss(self, batch: ArmBatch, *, weights: torch.Tensor | None = None):
        batch, weights = self.prepare(batch, weights)
        o = self.forward(batch)
        core = masked_mean(o["err"], batch.valid_bin, weights)
        var = o["variance"] if o["variance"].requires_grad else self.variance_term(o["z_hat"])
        loss = core + self.variance_weight * var
        self._ema_update(self.trunk, self.target_trunk)
        self._ema_update(self.lat_head, self.target_lat_head)
        return loss, {
            "lat_rec_loss": float(core.detach()),
            "variance_term": float(var.detach()),
            "variance_weight": self.variance_weight,
            "ema_momentum": self.ema_momentum,
            "bottleneck": self.bottleneck,
            "suppression": self._suppression,
        }

    def score_bins(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(batch)["err"]

    def latents_for_collapse(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self._predict(batch)[0]

    def probe_features(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self._predict(batch)[0]   # (B, T, D), per bin

    def collapse_state(self) -> dict:
        return {
            "target_trunk": {k: v.detach().cpu() for k, v in self.target_trunk.state_dict().items()},
            "target_lat_head": {k: v.detach().cpu() for k, v in self.target_lat_head.state_dict().items()},
        }

    def load_extra_checkpoint_state(self, state: dict) -> None:
        if not state:
            return
        if "target_trunk" in state:
            self.target_trunk.load_state_dict(state["target_trunk"])
        if "target_lat_head" in state:
            self.target_lat_head.load_state_dict(state["target_lat_head"])

    def objective_spec(self) -> dict:
        return {**super().objective_spec(), "lambdas": {}, "horizon_k": 0}


# ---------------------------------------------------------------------------
# A5 / A6 — future prediction
# ---------------------------------------------------------------------------


class PredictionArm(_Arm):
    """``x̂_{t+1:t+k} = P(h_{≤t})`` — a forecast, scored by how badly the future missed.

    Distinct from A1 in the *target set* (future bins, multi-step) and in the
    scoring rule (a bin is scored from its origins, not from its own position),
    which is the difference the design insists stays visible.
    """

    model_id = "A5"
    paradigm = "future_prediction"
    supports_suppression = ("S0", "S1", "S2", "S3")
    uses_loo = False        # a forecaster reads h_t (its own state), not a shifted copy

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.pred_head = nn.Sequential(
            nn.Linear(self._d_hidden, self._d_hidden), nn.GELU(),
            nn.Linear(self._d_hidden, self.horizon_k * self.d_channels),
        )
        self._objective = {
            "objective": "masked_huber_multistep_forecast",
            "target": "observed_values_of_bins_t_plus_1_to_t_plus_k",
            "target_source": "batch.x shifted +1..+k; context strictly ≤ t; no target branch",
            "gradient_path": "trunk + pred_head",
            "collapse_mechanism": None,
        }

    def forward(self, batch: ArmBatch) -> dict:  # type: ignore[override]
        out = self.encode(batch)
        B, T, _ = out.hidden.shape
        pred = self.pred_head(out.hidden).view(B, T, self.horizon_k, self.d_channels)
        tgt, tgtm = future_targets(batch.x, batch.m, self.horizon_k)
        e = F.smooth_l1_loss(pred, tgt, reduction="none", beta=self.huber_beta)
        mf = tgtm.to(e.dtype)
        per_origin = (e * mf).sum(-1) / mf.sum(-1).clamp(min=1.0)
        scored = tgtm.any(-1)
        per_origin = per_origin.masked_fill(~scored, float("nan"))
        err = target_centric(per_origin, scored)
        self._finite_guard(pred)
        self._finite_guard(err, allow_nan=True)
        origins_valid = batch.valid_bin.unsqueeze(-1) & scored
        return {"pred": pred, "err": err, "per_origin": per_origin, "scored": scored,
                "origins_valid": origins_valid, "hidden": out.hidden}

    def compute_loss(self, batch: ArmBatch, *, weights: torch.Tensor | None = None):
        batch, weights = self.prepare(batch, weights)
        o = self.forward(batch)
        # The loss is ORIGIN-centric (everything the arm was asked to forecast)
        # while the score is TARGET-centric (the consensus a user of the forecast
        # would read). Both are stated, because the gap between them is a design
        # decision a reviewer is entitled to see.
        loss = masked_mean(o["per_origin"], o["origins_valid"], weights)
        return loss, {
            "pred_loss": float(loss.detach()),
            "horizon_k": self.horizon_k,
            "n_scored_origins": int(o["origins_valid"].sum()),
            "suppression": self._suppression,
        }

    def score_bins(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(batch)["err"]

    def probe_features(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self.encode(batch).hidden   # (B, T, D): the probe picks granularity

    def objective_spec(self) -> dict:
        return {**super().objective_spec(), "lambdas": {}, "horizon_k": self.horizon_k}


# ---------------------------------------------------------------------------
# A7 / A8 — JEPA
# ---------------------------------------------------------------------------


class JEPArm(_Arm):
    """Predict the future *latent* of a bin from the causal context; never rebuild pixels.

    ``L = ‖P(h_{≤t})_h − sg(z⁺_{t+h})‖²`` summed over ``h = 1…k``. The target
    encoder is an EMA copy, which is what makes the objective non-trivial (a
    frozen-but-identical target would drift into the constant solution together with
    the online branch); the variance term is what forbids that solution, and the
    content probe is what proves the representation still contains the phenology.
    """

    model_id = "A7"
    paradigm = "jepa"
    #: S2 is degenerate for this arm — see the applicability table's reason.
    supports_suppression = ("S0", "S1", "S3")
    uses_latent = True
    uses_loo = False        # a JEPA arm has no input-space reconstruction at all

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        # The predictor consumes the ONLINE latent, not the raw hidden state: that
        # is what keeps `lat_head` inside the gradient path (an online projector
        # that receives no gradient is a dead parameter, and a JEPA arm whose online
        # projection is frozen is not learning a representation at all).
        self.predictor = nn.Sequential(
            nn.Linear(self.latent_dim, self._d_hidden), nn.GELU(),
            nn.Linear(self._d_hidden, self.horizon_k * self.latent_dim),
        )
        self.target_trunk = self._ema_copy(self.trunk)
        self.target_lat_head = self._ema_copy(self.lat_head)
        self._objective = {
            "objective": "masked_huber_latent_forecast",
            "target": "ema_target_latents_of_bins_t_plus_1_to_t_plus_k",
            "target_source": "EMA copy of trunk+lat_head over future bins, stop-gradient",
            "gradient_path": "trunk + lat_head + predictor; target branch frozen",
            "collapse_mechanism": "ema_target_encoder+variance_preservation",
        }

    @torch.no_grad()
    def target_latents(self, batch: ArmBatch) -> torch.Tensor:
        tok, valid = self.tokens(batch)
        return self.target_lat_head(self.target_trunk(tok, valid).hidden)

    def forward(self, batch: ArmBatch) -> dict:  # type: ignore[override]
        out = self.encode(batch)
        B, T, _ = out.hidden.shape
        z_ctx = self.lat_head(out.hidden)
        pred = self.predictor(z_ctx).view(B, T, self.horizon_k, self.latent_dim)
        ref = self.target_latents(batch)
        lat_mask = batch.m.any(-1, keepdim=True).expand(-1, -1, ref.shape[-1])
        tgt, tgtm = future_targets(ref, lat_mask, self.horizon_k)
        err_origin = (pred - tgt).pow(2).mean(-1)
        scored = tgtm.any(-1)
        err_origin = err_origin.masked_fill(~scored, float("nan"))
        err = target_centric(err_origin, scored)
        self._finite_guard(pred)
        self._finite_guard(err, allow_nan=True)
        return {"z_ctx": z_ctx, "pred_latent": pred, "target_latent": ref,
                "err": err, "err_origin": err_origin, "scored": scored,
                "origins_valid": batch.valid_bin.unsqueeze(-1) & scored,
                "variance": self.variance_term(pred[batch.valid_bin]) if bool(batch.valid_bin.any()) else z_ctx.new_zeros(())}

    def compute_loss(self, batch: ArmBatch, *, weights: torch.Tensor | None = None):
        batch, weights = self.prepare(batch, weights)
        o = self.forward(batch)
        core = masked_mean(o["err_origin"], o["origins_valid"], weights)
        var = o["variance"]
        loss = core + self.variance_weight * var
        self._ema_update(self.trunk, self.target_trunk)
        self._ema_update(self.lat_head, self.target_lat_head)
        return loss, {
            "jepa_loss": float(core.detach()),
            "variance_term": float(var.detach()),
            "variance_weight": self.variance_weight,
            "ema_momentum": self.ema_momentum,
            "horizon_k": self.horizon_k,
            "suppression": self._suppression,
        }

    def score_bins(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(batch)["err"]

    def latents_for_collapse(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(batch)["z_ctx"]

    def probe_features(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(batch)["z_ctx"]   # (B, T, D), per bin

    def collapse_state(self) -> dict:
        return {
            "target_trunk": {k: v.detach().cpu() for k, v in self.target_trunk.state_dict().items()},
            "target_lat_head": {k: v.detach().cpu() for k, v in self.target_lat_head.state_dict().items()},
        }

    def load_extra_checkpoint_state(self, state: dict) -> None:
        if not state:
            return
        if "target_trunk" in state:
            self.target_trunk.load_state_dict(state["target_trunk"])
        if "target_lat_head" in state:
            self.target_lat_head.load_state_dict(state["target_lat_head"])

    def objective_spec(self) -> dict:
        return {**super().objective_spec(), "lambdas": {}, "horizon_k": self.horizon_k,
                "probe_required": True}


# ---------------------------------------------------------------------------
# A9 / A10 — hybrid
# ---------------------------------------------------------------------------


class HybridArm(_Arm):
    """``L = λ_r · L_rec + λ_p · L_pred``, both streams reported separately.

    A weighted combination of two documented losses — not a third name for
    reconstruction. The per-stream errors are exposed so the report can show a
    hybrid that wins on one stream and loses on the other instead of averaging the
    fact away.
    """

    model_id = "A9"
    paradigm = "hybrid"
    supports_suppression = ("S0", "S1", "S2", "S3")

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        d = self._d_hidden
        self.rec_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, self.d_channels))
        self.pred_head = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                                       nn.Linear(d, self.horizon_k * self.d_channels))
        self._objective = {
            "objective": "weighted_reconstruction_plus_prediction",
            "target": "observed_values_of_the_current_bin AND of bins t+1..t+k",
            "target_source": "batch.x at bin t (LOO) and batch.x shifted +1..+k",
            "gradient_path": "shared trunk + two heads; no target encoder",
            "collapse_mechanism": None,
        }

    def forward(self, batch: ArmBatch) -> dict:  # type: ignore[override]
        out = self.encode(batch)
        B, T, _ = out.hidden.shape
        recon = self.rec_head(self.prev_hidden(out))
        err_rec = masked_bin_error(recon, batch.x, batch.m, beta=self.huber_beta)
        pred = self.pred_head(out.hidden).view(B, T, self.horizon_k, self.d_channels)
        tgt, tgtm = future_targets(batch.x, batch.m, self.horizon_k)
        e = F.smooth_l1_loss(pred, tgt, reduction="none", beta=self.huber_beta)
        mf = tgtm.to(e.dtype)
        per_origin = (e * mf).sum(-1) / mf.sum(-1).clamp(min=1.0)
        scored = tgtm.any(-1)
        per_origin = per_origin.masked_fill(~scored, float("nan"))
        err_pred = target_centric(per_origin, scored)
        # The blended score is the mean of the two streams where BOTH exist; where
        # only one does (bin 0, or a bin with no valid origin), the blend falls back
        # to the available stream and says so via `blend_coverage`.
        stack = torch.stack([err_rec, err_pred], dim=0)
        err = torch.nanmean(stack, dim=0)
        err = err.masked_fill(~batch.valid_bin, float("nan"))
        self._finite_guard(recon, pred)
        self._finite_guard(err, allow_nan=True)
        return {"recon": recon, "pred": pred, "err_rec": err_rec, "err_pred": err_pred,
                "per_origin": per_origin, "scored": scored, "err": err,
                "origins_valid": batch.valid_bin.unsqueeze(-1) & scored,
                "blend_coverage": torch.isfinite(stack).float().mean(dim=0)}

    def compute_loss(self, batch: ArmBatch, *, weights: torch.Tensor | None = None):
        batch, weights = self.prepare(batch, weights)
        o = self.forward(batch)
        rec = masked_mean(o["err_rec"], batch.valid_bin, weights)
        pred = masked_mean(o["per_origin"], o["origins_valid"], weights)
        loss = self.lambda_rec * rec + self.lambda_pred * pred
        return loss, {
            "rec_loss": float(rec.detach()),
            "pred_loss": float(pred.detach()),
            "lambda_rec": self.lambda_rec,
            "lambda_pred": self.lambda_pred,
            "suppression": self._suppression,
        }

    def score_bins(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(batch)["err"]

    def score_streams(self, batch: ArmBatch) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            o = self.forward(batch)
        return {"err_rec": o["err_rec"], "err_pred": o["err_pred"], "err": o["err"]}

    def probe_features(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self.encode(batch).hidden   # (B, T, D): the probe picks granularity

    def objective_spec(self) -> dict:
        return {
            **super().objective_spec(),
            "lambdas": {"lambda_rec": self.lambda_rec, "lambda_pred": self.lambda_pred},
            "horizon_k": self.horizon_k,
        }


#: paradigm → class. The backbone is a constructor argument, which is why ten
#: architectures are five classes × two trunks rather than ten copy-pasted files.
ARM_CLASSES: dict[str, type[_Arm]] = {
    "input_reconstruction": InputReconstructionArm,
    "latent_reconstruction": LatentReconstructionArm,
    "future_prediction": PredictionArm,
    "jepa": JEPArm,
    "hybrid": HybridArm,
}

#: model id → (paradigm, backbone). Cross-checked against the contract's
#: ``ARCHITECTURES`` table by the config test so the two cannot drift.
ARM_BY_MODEL: dict[str, tuple[str, str]] = {
    "A1": ("input_reconstruction", "gru"),
    "A2": ("input_reconstruction", "mamba"),
    "A3": ("latent_reconstruction", "gru"),
    "A4": ("latent_reconstruction", "mamba"),
    "A5": ("future_prediction", "gru"),
    "A6": ("future_prediction", "mamba"),
    "A7": ("jepa", "gru"),
    "A8": ("jepa", "mamba"),
    "A9": ("hybrid", "gru"),
    "A10": ("hybrid", "mamba"),
}

#: Random-init capacity control: the A1 arm of each trunk, never trained.
CAPACITY_CONTROL_TRUNKS: dict[str, str] = {"rand_gru": "gru", "rand_mamba": "mamba"}


def objective_signature(model_id: str) -> dict:
    """The design's objective row for a model id, without building a tensor.

    Used by the pre-execution audit so an aliasing mistake is caught before any
    GPU time is spent, and by the report's objective table.
    """
    paradigm, backbone = ARM_BY_MODEL[model_id]
    return {
        "model_id": model_id,
        "paradigm": paradigm,
        "backbone": backbone,
        "objective": {
            "input_reconstruction": "masked_huber_input_space",
            "latent_reconstruction": "masked_huber_latent_space_same_bin",
            "future_prediction": "masked_huber_multistep_forecast",
            "jepa": "masked_huber_latent_forecast",
            "hybrid": "weighted_reconstruction_plus_prediction",
        }[paradigm],
        "target_source": {
            "input_reconstruction": "batch.x at bin t, predicted from h_{<t} only",
            "latent_reconstruction": "EMA copy of trunk+lat_head over the same bins, stop-gradient",
            "future_prediction": "batch.x shifted +1..+k; context strictly ≤ t; no target branch",
            "jepa": "EMA copy of trunk+lat_head over future bins, stop-gradient",
            "hybrid": "batch.x at bin t (LOO) and batch.x shifted +1..+k",
        }[paradigm],
        "collapse_prevention_required": paradigm in ("latent_reconstruction", "jepa"),
        "supports_suppression": (
            ("S0", "S1", "S3") if paradigm == "jepa" else ("S0", "S1", "S2", "S3")
        ),
    }


def build_arm(
    model_id: str,
    *,
    cfg,
    d_channels: int,
    d_cov: int = 0,
    variant: str = "base",
    **overrides: Any,
) -> _Arm:
    """Instantiate one pre-registered arm from the contract plus cell overrides.

    ``variant`` carries the pre-registered non-factorial knobs: ``lam0.5`` /
    ``lam1.0`` / ``lam2.0`` select the hybrid λ pair, ``rand_gru`` / ``rand_mamba``
    select the random-init capacity control (an arm that is never trained).
    """
    if model_id == "A-RAND":
        backbone = CAPACITY_CONTROL_TRUNKS.get(variant, "gru")
        kw = _arm_kwargs(cfg, d_channels=d_channels, d_cov=d_cov, backbone=backbone,
                         paradigm="input_reconstruction")
        kw["mask_rate"] = 0.0        # a control that trains nothing masks nothing
        arm = ARM_CLASSES["input_reconstruction"](**kw)
        arm.model_id = "A-RAND"
        arm.paradigm = "capacity_control"
        arm.backbone = backbone
        arm.requires_grad_(False)     # a control must not be trainable by accident
        arm._objective = {
            "objective": "none_random_init_control",
            "target": "not_applicable",
            "target_source": "no training step by construction (capacity control)",
            "gradient_path": "none (all parameters frozen)",
            "collapse_mechanism": None,
        }
        return arm
    if model_id not in ARM_BY_MODEL:
        raise KeyError(
            f"{model_id!r} is not one of the pre-registered arms ({sorted(ARM_BY_MODEL)}); "
            "an arm outside the contract is not runnable"
        )
    paradigm, backbone = ARM_BY_MODEL[model_id]
    kw = _arm_kwargs(cfg, d_channels=d_channels, d_cov=d_cov, backbone=backbone,
                     paradigm=paradigm)
    t = cfg.raw["training"]
    variant_map = t.get("lambda_variants") or {}
    if variant in variant_map:
        kw.update(
            lambda_rec=float(variant_map[variant].get("lambda_rec", kw["lambda_rec"])),
            lambda_pred=float(variant_map[variant].get("lambda_pred", kw["lambda_pred"])),
        )
    elif variant not in ("", "base"):
        raise KeyError(
            f"variant {variant!r} is not a pre-registered lambda variant "
            f"({sorted(variant_map)}); inventing one mid-run is a hidden config change"
        )
    kw.update({k: v for k, v in overrides.items() if v is not None})
    arm = ARM_CLASSES[paradigm](**kw)
    arm.model_id = model_id
    arm.paradigm = paradigm
    arm.backbone = backbone
    return arm


def _arm_kwargs(cfg, *, d_channels: int, d_cov: int, backbone: str, paradigm: str = "") -> dict:
    t = cfg.raw["training"]
    rates = (t.get("masking") or {}).get("rate_by_paradigm") or {}
    cap = cfg.raw.get("capacity", {})
    return {
        "d_channels": int(d_channels),
        "d_cov": int(d_cov),
        "n_bins": int(t.get("n_bins", 36)),
        "d_model": int(t.get("d_model", 96)),
        "n_layers": int(t.get("n_layers", 1)),
        "dropout": float(t.get("dropout", 0.1)),
        "d_state": int(t.get("d_state", 16)),
        "latent_dim": int(t.get("latent_dim", 32)),
        "huber_beta": float(t.get("huber_beta", 1.0)),
        "horizon_k": int(t.get("horizon_k", 4)),
        "ema_momentum": float(t.get("ema_momentum", 0.996)),
        "variance_weight": float(cfg.raw["collapse"].get("variance_weight", 0.05)),
        "lambda_rec": float(t.get("lambda_rec", 1.0)),
        "lambda_pred": float(t.get("lambda_pred", 1.0)),
        "backbone": backbone,
        "causal": True,
        "bottleneck_frac": float(cap.get("bottleneck_frac", 0.25)),
        "mask_rate": float(rates.get(paradigm, 0.0)),
    }


def count_trainable(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def param_band_check(params: int, target: int, min_ratio: float, max_ratio: float) -> dict:
    lo, hi = target * min_ratio, target * max_ratio
    return {
        "params": int(params),
        "target_params": int(target),
        "band": [round(lo, 1), round(hi, 1)],
        "ratio_to_target": round(float(params) / float(target), 4) if target else None,
        "in_band": bool(lo <= params <= hi),
    }


def estimate_flops(arm: _Arm, *, batch_size: int = 32, n_bins: int | None = None) -> dict:
    """Analytic forward-pass FLOP estimate, with the convention recorded (task 9.13).

    Deliberately analytic rather than profiler-based: a profiler number depends on
    the kernel library as much as on the model, and the point of the column is
    "how much work does one forward pass of this arm do". MAC→FLOP doubling is
    applied; recurrence steps are counted per bin, which is where a selective-scan
    trunk actually pays.
    """
    T = int(n_bins if n_bins is not None else arm.n_bins)
    d = arm._d_hidden
    C = arm.d_channels
    K = int(getattr(arm, "horizon_k", 1) or 1)
    flops = 0.0
    # token projection: (2C + d_cov) → d, once per bin
    flops += 2.0 * T * arm.d_in * d
    if arm.backbone == "gru":
        # per bin, per layer: gate projections (3·d·in) + recurrent (3·d·d)
        flops += 2.0 * T * (3.0 * d * arm.d_in + 3.0 * d * d) * arm._n_layers
    else:
        # causal conv (depthwise, kernel 4) + selective scan (state d_state) + gates
        inner = d * 2
        flops += 2.0 * T * (
            4.0 * inner                       # depthwise conv
            + 2.0 * inner * arm._d_state * 2  # B/C projections inside the scan
            + 2.0 * inner * inner             # in/out projections
        ) * arm._n_layers
    heads = {"input_reconstruction": 1, "latent_reconstruction": 2, "future_prediction": 1,
             "jepa": 1, "hybrid": 2, "capacity_control": 1}
    n_heads = heads.get(arm.paradigm, 1)
    out_dim = C if arm.paradigm in ("input_reconstruction", "hybrid", "capacity_control") else arm.latent_dim
    for _ in range(n_heads):
        flops += 2.0 * T * (d * d + d * (K * out_dim if K > 1 else out_dim))
    if arm.paradigm in ("latent_reconstruction", "jepa"):
        # EMA target branch: an extra trunk + lat head, evaluated but not differentiated
        flops *= 2.0
    per_sample = flops
    return {
        "forward_flops_per_batch": float(per_sample * batch_size),
        "forward_flops_per_sample": float(per_sample),
        "batch_size": int(batch_size),
        "n_bins": T,
        "counts": "analytic MAC×2 for Linear/Conv/GPU recurrence ops; no backward term",
        "convention": (
            "One forward pass at the reference batch shape; FLOPs = 2 × MACs; "
            "recurrence cost counted per bin; EMA target branches counted; "
            "optimizer state updates excluded (they are constant across arms)."
        ),
        "device": "analytic (device-independent)",
    }


def flops_for_reference(cfg, model_id: str, *, d_channels: int, d_cov: int, variant: str = "base") -> dict:
    """Build the arm on meta devices and estimate — used by the capacity/fairness table."""
    paradigm = ARM_BY_MODEL.get(model_id, ("input_reconstruction", "gru"))[0]
    kw = _arm_kwargs(cfg, d_channels=d_channels, d_cov=d_cov,
                     backbone=ARM_BY_MODEL.get(model_id, ("", "gru"))[1], paradigm=paradigm)
    t = cfg.raw["training"]
    variant_map = t.get("lambda_variants") or {}
    if variant in variant_map:
        kw.update(lambda_rec=float(variant_map[variant].get("lambda_rec", 1.0)),
                  lambda_pred=float(variant_map[variant].get("lambda_pred", 1.0)))
    cls = ARM_CLASSES.get(ARM_BY_MODEL.get(model_id, ("input_reconstruction",))[0], InputReconstructionArm)
    arm = cls(**kw)
    arm.model_id = model_id
    return estimate_flops(arm, batch_size=int(t.get("batch_size", 32)), n_bins=int(t.get("n_bins", 36)))


__all__ = [
    "ARM_BY_MODEL",
    "ARM_CLASSES",
    "CAPACITY_CONTROL_TRUNKS",
    "HybridArm",
    "InputReconstructionArm",
    "JEPArm",
    "LatentReconstructionArm",
    "NumericalInstability",
    "PARADIGM_ORDER",
    "PredictionArm",
    "build_arm",
    "count_trainable",
    "estimate_flops",
    "flops_for_reference",
    "future_targets",
    "masked_bin_error",
    "masked_mean",
    "objective_signature",
    "param_band_check",
    "target_centric",
]

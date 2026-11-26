"""Time2Agri pretext arms A11–A13 as benchmark ``BaseArm`` adapters (phase 3.1).

Scope boundary. These are **adapters, not cells**. ``ARM_BY_MODEL``,
``configs/benchmarks/benchmark_config.yaml`` and ``configs/benchmarks/matrix.yaml``
are frozen pre-registration artifacts and are deliberately untouched here: adding
A11–A13 to the matrix requires a dated amendment (and a regenerated matrix hash)
decided by the owner before any execution. Until then these arms are unit-tested
and callable through :func:`build_time2agri_bench_arm`, but no pre-registered
cell can select them.

Faithfulness and its limits. The training estimand is the paper's TD / FP / FF
objective, implemented in :mod:`ssl_anomaly.models.time2agri_archs`. The benchmark
contract additionally requires a per-bin error (``components["err"]``) whose
mean is the training loss, so that scoring and optimisation cannot drift apart.
For FF this is exactly the future-frame MSE. For TD and FP the paper defines a
sequence-level objective, so this module evaluates the same estimand at every
eligible origin (TD gap classification) or as a per-bin spectral shape residual
(FP), and says so in ``objective_spec()["scoring_note"]``. Those per-bin scores
are **benchmark adaptations, not numbers reported by Gupta & Sobti (2025)**.

Leakage rule. The FP arm refuses to run without train-period NDVI targets supplied
by the caller; a target built from a dev/test period is refused by
:func:`~ssl_anomaly.models.time2agri_archs.fft_ndvi_target`. Nothing is fabricated
when targets are absent — the arm fails closed.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from ssl_anomaly.benchmarking.interface import ArmBatch, BaseArm
from ssl_anomaly.models.bench.backbones import NumericalInstability
from ssl_anomaly.models.time2agri_archs import (
    ARMS as T2A_ARMS,
    GAP_CLASSES,
    build_time2agri_arm,
    fft_ndvi_target,
)

#: Extension ids. Kept separate from ``ARM_BY_MODEL`` (A1–A10) on purpose.
TIME2AGRI_MODEL_IDS = ("A11", "A12", "A13")
TIME2AGRI_ARM_BY_MODEL: dict[str, str] = {
    "A11": "time2agri_td",
    "A12": "time2agri_fp",
    "A13": "time2agri_ff",
}


def time2agri_objective_signature(model_id: str) -> dict:
    """Machine-readable objective row for an extension id (no tensor built)."""
    if model_id not in TIME2AGRI_ARM_BY_MODEL:
        raise KeyError(f"{model_id!r} is not a Time2Agri extension arm")
    arm = TIME2AGRI_ARM_BY_MODEL[model_id]
    base = build_time2agri_arm(arm, 2).objective_spec()
    return {
        "model_id": model_id,
        "paradigm": "time2agri_pretext",
        "backbone": "vit1d",
        "objective": base["objective"],
        "target_source": base["target_source"],
        "gradient_path": base["gradient_path"],
        "collapse_prevention_required": False,
        "supports_suppression": ("S0",),
    }


class _Time2AgriBenchArm(BaseArm):
    """Shared constructor/scoring plumbing; each subclass supplies ``_bin_errors``."""

    arm_name: str = ""
    _scoring_note: str = ""

    def __init__(
        self,
        *,
        d_channels: int,
        d_cov: int = 0,
        n_bins: int = 36,
        d_model: int = 32,
        dropout: float = 0.1,
        ndvi_index: int | None = None,
        fp_targets: torch.Tensor | None = None,
        split: str = "train",
        **_: Any,
    ) -> None:
        super().__init__()
        self.d_channels = int(d_channels)
        self.d_cov = int(d_cov)
        self.ndvi_index = None if ndvi_index is None else int(ndvi_index)
        self.split = str(split)
        self.core = build_time2agri_arm(self.arm_name, 2 * self.d_channels + self.d_cov,
                                        d_model=int(d_model), dropout=float(dropout))
        self.register_buffer("_fp_targets", fp_targets.clone() if fp_targets is not None else None)
        self._objective = {
            **self.core.objective_spec(),
            "collapse_mechanism": None,
            "scoring_note": self._scoring_note,
        }

    # ---- helpers ---------------------------------------------------------
    def _tokens(self, batch: ArmBatch) -> torch.Tensor:
        """The benchmark token matrix; a hidden bin keeps its observation flags."""
        return batch.input()

    def _observed(self, batch: ArmBatch) -> torch.Tensor:
        """Observation mask aligned with ``batch.input()``'s channel layout.

        ``input()`` concatenates values, observation flags and covariates, so a
        raw ``batch.m`` (value channels only) would be the wrong width for the
        encoder. The appended flag channel is always observed by construction;
        a covariate channel is observed only where it is finite.
        """
        parts = [batch.m, torch.ones_like(batch.m)]
        if batch.cov is not None:
            parts.append(torch.isfinite(batch.cov))
        return torch.cat(parts, dim=-1)

    @staticmethod
    def _demean_norm(y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(y.dtype)
        n = m.sum().clamp(min=1.0)
        mu = (y * m).sum() / n
        z = (y - mu) * m
        return z / z.pow(2).sum().div(n).sqrt().clamp(min=1e-6)

    def _bin_errors(self, batch: ArmBatch) -> torch.Tensor:  # pragma: no cover - abstract
        raise NotImplementedError

    def _pooled(self, x: torch.Tensor) -> torch.Tensor:
        _, emb = self.core(x)
        return emb

    def _prepare(self, batch: ArmBatch, weights: torch.Tensor | None):
        """The BaseArm-level subset of ``_Arm.prepare`` (S0 only, no own masking)."""
        cm = self.suppression_mask(batch)
        if cm is not None:
            batch = ArmBatch(x=batch.x, m=batch.m, valid_bin=batch.valid_bin, cov=batch.cov,
                             ctx_mask=cm, meta=batch.meta)
        w = self.suppression_weights(batch)
        if weights is not None:
            w = weights if w is None else w * weights
        return batch, w

    # ---- BaseArm contract ------------------------------------------------
    def forward(self, batch: ArmBatch) -> dict:  # type: ignore[override]
        err = self._bin_errors(batch)
        if bool(torch.isinf(err).any()):
            raise NumericalInstability(f"{self.model_id}: infinite per-bin error")
        return {"err": err, "pooled": self._pooled(self._tokens(batch)), "hidden": None}

    def compute_loss(self, batch: ArmBatch, *, weights: torch.Tensor | None = None):
        # ``prepare`` already unions suppression weights with any explicit weights;
        # the scorer reads the same per-bin errors, so loss and score cannot drift.
        batch, w = self._prepare(batch, weights)
        err = self._bin_errors(batch)
        loss = masked_mean(err, batch.valid_bin, w)
        return loss, {
            "err": err,
            "n_scored_bins": int(torch.isfinite(err).sum()),
            "suppression": self._suppression,
        }

    def score_bins(self, batch: ArmBatch) -> torch.Tensor:
        with torch.no_grad():
            return self._bin_errors(batch)

    def probe_features(self, batch: ArmBatch) -> torch.Tensor:
        return self._pooled(self._tokens(batch))

    def latents_for_collapse(self, batch: ArmBatch) -> torch.Tensor | None:
        # No EMA target or variance floor: collapse monitoring is not applicable,
        # and returning a latent here would fake a diagnostic that does not exist.
        return None

    def objective_spec(self) -> dict:
        # BaseArm adds paradigm/causal/suppression; the subclass already set the rest.
        return super().objective_spec()


def masked_mean(err: torch.Tensor, valid: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
    finite = torch.isfinite(err)
    keep = finite & valid.to(finite.dtype).bool()
    w = keep.to(err.dtype)
    if weights is not None:
        ww = weights.to(err.dtype)
        w = w * ww if ww.dim() == w.dim() else w * ww.unsqueeze(-1)
    return (torch.nan_to_num(err, nan=0.0) * w).sum() / w.sum().clamp(min=1.0)


class Time2AgriTDBenchArm(_Time2AgriBenchArm):
    """A11 — TD gap-classification cross-entropy at every eligible origin."""

    model_id = "A11"
    paradigm = "time2agri_pretext"
    arm_name = "time2agri_td"
    _scoring_note = (
        "The paper samples one frame pair per sequence; this arm evaluates the same "
        "gap classifier at each eligible origin. Bins without a forward partner "
        "(the last GAP_CLASSES patches) are NaN, not zero."
    )

    def _bin_errors(self, batch: ArmBatch) -> torch.Tensor:
        x, obs = self._tokens(batch), self._observed(batch)
        _, seq, _ = self.core.encoder(x, obs)
        b, n = x.shape[0], seq.shape[1] - 1
        err = torch.full((b, n), float("nan"), device=x.device, dtype=torch.float32)
        for gap in range(1, GAP_CLASSES + 1):
            origins = torch.arange(n - gap, device=x.device)
            if origins.numel() == 0:
                continue
            ha = seq[torch.arange(b, device=x.device).unsqueeze(1), origins.unsqueeze(0) + 1]
            hb = seq[torch.arange(b, device=x.device).unsqueeze(1), (origins + gap).unsqueeze(0) + 1]
            logits = self.core.head(torch.cat([ha, hb], dim=-1))
            ce = F.cross_entropy(logits.flatten(0, 1),
                                 torch.full((b * len(origins),), gap - 1, device=x.device),
                                 reduction="none").view(b, -1)
            err[:, origins] = ce
        # The encoder scores patches, not raw bins; each patch belongs to
        # patch_size consecutive bins, so the score is expanded (never invented).
        p = self.core.encoder.patch_size
        err = err.repeat_interleave(p, dim=1)[: batch.valid_bin.shape[1]]
        return err.masked_fill(~batch.valid_bin, float("nan"))


class Time2AgriFPBenchArm(_Time2AgriBenchArm):
    """A12 — FP dominant-frequency prediction as a per-bin spectral residual.

    The predicted top-3 frequency/energy components are inverse-FFT'd and compared
    to the observed NDVI shape (both unit-energy, mean-removed). Only bins the
    objective actually reads (first and last quarter) are scorable; the rest are
    NaN rather than a fabricated perfect score.
    """

    model_id = "A12"
    paradigm = "time2agri_pretext"
    arm_name = "time2agri_fp"
    _scoring_note = (
        "Per-bin score = spectral shape residual + the paper's sequence-level FP "
        "target MSE attributed to each read bin. Not a metric reported by the "
        "paper. Bins outside the read windows are unscorable (NaN)."
    )

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        if self.ndvi_index is None:
            raise ValueError("A12 requires an explicit ndvi_index; no channel default is invented")
        if str(self.split).lower() in {"eval", "test", "dev", "holdout"}:
            raise ValueError("A12 FP targets refused for an evaluation-period arm (leakage guard)")
        if self._fp_targets is None:
            raise ValueError("A12 requires train-period fp_targets; absent targets are never filled")
        if self._fp_targets.shape[-1] != 6:
            raise ValueError("A12 fp_targets must have 6 columns (3 frequencies + 3 energies)")

    def _bin_errors(self, batch: ArmBatch) -> torch.Tensor:
        x, obs = self._tokens(batch), self._observed(batch)
        b, t, _ = x.shape
        idx = self._fp_targets[batch.meta["indices"]].to(x.device)
        ni = self.ndvi_index
        y = x[:, :, ni]
        m = obs[:, :, ni]
        q = max(2, t // 4)
        frames = torch.cat([x[:, :q], x[:, -q:]], dim=1)
        obs_f = torch.cat([obs[:, :q], obs[:, -q:]], dim=1)
        te = self.core._te(x, torch.cat([torch.arange(q, device=x.device),
                                         torch.arange(t - q, t, device=x.device)]))
        cls, _, _ = self.core.encoder(frames, obs_f, te)
        pred = self.core.head(torch.cat([cls, te.mean(1)], dim=-1))
        # Spectral shape reconstruction from the predicted top-3 components.
        n_fft = max(1, t // 2)
        spec = torch.zeros(b, n_fft + 1, device=x.device, dtype=torch.float32)
        k = (pred[:, :3] * n_fft).round().long().clamp(1, n_fft)
        # Frequencies may collide (duplicate top-3); sum their energies rather than
        # overwriting, which is both the honest spectrum and a differentiable op.
        spec.scatter_add_(1, k, pred[:, 3:].clamp(min=0.0))
        yhat = torch.fft.irfft(spec, n=t)
        yn = self._demean_norm(y, m)
        hn = self._demean_norm(yhat, torch.ones_like(m))
        # The paper's FP objective is a sequence-level MSE against train-period
        # frequency targets. Attribute it to each read bin so the scorer's mean
        # stays exactly the training loss; the shape residual localises it.
        fp_mse = (pred - idx).pow(2).mean(-1).unsqueeze(1)
        err = (yn - hn).abs() + fp_mse
        read = torch.zeros(t, dtype=torch.bool, device=x.device)
        read[:q] = True
        read[-q:] = True
        err = err.masked_fill(~read.unsqueeze(0), float("nan"))
        return err.masked_fill(~batch.valid_bin, float("nan"))


class Time2AgriFFBenchArm(_Time2AgriBenchArm):
    """A13 — FF future-frame MSE; origin bins are NaN (no forecast exists there)."""

    model_id = "A13"
    paradigm = "time2agri_pretext"
    arm_name = "time2agri_ff"
    _scoring_note = (
        "Exact per-target-bin FF reconstruction MSE. Origin-half bins are NaN "
        "because the objective makes no prediction for them."
    )

    def _bin_errors(self, batch: ArmBatch) -> torch.Tensor:
        x, obs = self._tokens(batch), self._observed(batch)
        b, t, _ = x.shape
        half = t // 2
        origin, future = x[:, :half], x[:, half:]
        te_f = self.core._te(x, torch.arange(half, t, device=x.device))
        cls, _, _ = self.core.encoder(origin, obs[:, :half])
        z = self.core.translator(torch.cat([cls, te_f.mean(1)], dim=-1))
        pred = self.core.decoder(z).unsqueeze(1).expand(-1, t - half, -1)
        m = obs[:, half:]
        err = ((pred - future).pow(2) * m).sum(-1) / m.sum(-1).clamp(min=1)
        err = err.masked_fill(m.sum(-1) == 0, float("nan"))
        full = torch.full((b, t), float("nan"), device=x.device, dtype=err.dtype)
        full[:, half:] = err
        return full.masked_fill(~batch.valid_bin, float("nan"))


ARM_CLASSES: dict[str, type[_Time2AgriBenchArm]] = {
    "A11": Time2AgriTDBenchArm,
    "A12": Time2AgriFPBenchArm,
    "A13": Time2AgriFFBenchArm,
}


def build_time2agri_bench_arm(
    model_id: str, *, cfg: Any | None = None, d_channels: int, d_cov: int = 0, **overrides: Any
) -> _Time2AgriBenchArm:
    """Build an extension arm. Never falls back to a fabricated configuration."""
    if model_id not in ARM_CLASSES:
        raise KeyError(
            f"{model_id!r} is not a Time2Agri extension arm; expected {sorted(ARM_CLASSES)}"
        )
    kw: dict[str, Any] = {"d_channels": d_channels, "d_cov": d_cov}
    if cfg is not None:
        train = (getattr(cfg, "raw", None) or {}).get("training") or {}
        for src, dst in (("d_model", "d_model"), ("n_bins", "n_bins"), ("dropout", "dropout")):
            if src in train:
                kw[dst] = train[src]
    kw.update({k: v for k, v in overrides.items() if v is not None})
    arm = ARM_CLASSES[model_id](**kw)
    arm.model_id = model_id
    arm.backbone = "vit1d"
    return arm


__all__ = [
    "ARM_CLASSES", "TIME2AGRI_ARM_BY_MODEL", "TIME2AGRI_MODEL_IDS",
    "Time2AgriFPBenchArm", "Time2AgriFFBenchArm", "Time2AgriTDBenchArm",
    "build_time2agri_bench_arm", "time2agri_objective_signature",
    "fft_ndvi_target", "T2A_ARMS",
]

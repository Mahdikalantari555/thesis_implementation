"""S1 / S2 / S3 (spec anomaly-suppression-benchmark, requirements 2–4).

Each adapter is written so the *comparison stays fair*: it changes the loss
weights or the input, never the metric, never the capacity, and never the scored
partition. ``describe()`` records the mechanism, the fitted statistics and the
iteration state into the manifest, so a reader can reconstruct exactly what was
different about this arm.

A note on S1's self-reference, because a reviewer will ask: the weights come from
the **arm's own train residuals from the previous epoch**, smoothed. That is
self-paced learning, not a bug — but it means S1 is a *training-schedule* variant
rather than a clean reweighting by an external score, so the report labels it that
way and the warmup length is in the contract. Any alternative (residuals from a
frozen external model) would confound the suppression axis with the choice of that
model.
"""

from __future__ import annotations

import numpy as np
import torch

from ssl_anomaly.suppression.base import (
    BaseSuppression,
    PassthroughSuppression,
    SuppressionError,
    weight_schedule,
)


class ResidualReweightSuppression(BaseSuppression):
    """**S1** — down-weight bins the model already fits poorly (self-paced).

    ``w_bt = clip(1 / (1 + (r_bt / s)^q), w_min, 1)`` with ``s`` the train MAD of
    the previous epoch's residuals. Bins that are already terrible contribute less,
    so the representation is pulled toward the bulk of the data.
    """

    id = "S1"
    mechanism = "loss_reweighting_from_own_train_residuals"
    requires_training = True
    classical_transfer = "sample_weight_where_supported_else_sensitivity_row"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = self.params
        self.w_min = float(p.get("w_min", 0.05))
        self.q = float(p.get("power", 2.0))
        self.warmup_epochs = int(p.get("warmup_epochs", 3))
        self.ema = float(p.get("residual_ema", 0.7))
        if not 0.0 < self.w_min <= 1.0:
            raise SuppressionError(f"S1 w_min must be in (0, 1], got {self.w_min}")
        self._resid: np.ndarray | None = None      # (N, T) previous-epoch residuals
        self._scale: float | None = None

    # -- fitting -----------------------------------------------------------
    def fit(self, tensors, *, mask=None) -> dict:
        """Initialise from the train residual scale of an untrained pass if asked.

        The caller passes ``residuals`` through :meth:`observe` after the warmup;
        ``fit`` only computes the fallback scale (from the data's own spread) so the
        first weighted epoch is not dividing by zero.
        """
        x = np.asarray(tensors["x"] if isinstance(tensors, dict) else tensors, dtype=np.float64)
        m = None
        if isinstance(tensors, dict) and "m" in tensors:
            m = np.asarray(tensors["m"], dtype=bool)
        flat = x[m] if m is not None else x.ravel()
        self.state.fitted["data_fallback_scale"] = float(np.median(np.abs(flat - np.median(flat))) * 1.4826)
        return {"data_fallback_scale": self.state.fitted["data_fallback_scale"]}

    def observe(self, residuals: np.ndarray, indices: np.ndarray) -> None:
        """Absorb one epoch's train residuals (the only path to weights)."""
        r = np.asarray(residuals, dtype=np.float64)
        idx = np.asarray(indices, dtype=np.int64)
        if r.shape[0] != len(idx):
            raise SuppressionError(
                f"residual rows {r.shape[0]} != indices {len(idx)}: weights would be applied to the "
                "wrong cycles, which is worse than not weighting at all"
            )
        if self._resid is None:
            self._resid = np.full_like(r, np.nan, dtype=np.float64)
        finite = np.isfinite(r)
        prev = self._resid[idx]
        mixed = np.where(
            finite & np.isfinite(prev), self.ema * prev + (1.0 - self.ema) * r,
            np.where(finite, r, prev),
        )
        self._resid[idx] = mixed
        s = np.nanmedian(np.abs(mixed[np.isfinite(mixed)])) if np.isfinite(mixed).any() else np.nan
        self._scale = float(s) * 1.4826 if np.isfinite(s) and s > 0 else self._scale

    def on_epoch_start(self, epoch: int, *, total_epochs: int) -> None:
        super().on_epoch_start(epoch, total_epochs=total_epochs)

    # -- application -------------------------------------------------------
    def weights(self, batch) -> torch.Tensor | None:
        if self.state.epoch < self.warmup_epochs or self._resid is None:
            return None
        idx = batch.meta.get("indices")
        if idx is None:
            return None
        r = torch.as_tensor(
            np.nan_to_num(self._resid[np.asarray(idx, dtype=np.int64)], nan=0.0),
            dtype=batch.x.dtype, device=batch.x.device,
        )
        scale = self._scale or self.state.fitted.get("data_fallback_scale") or 1.0
        w = 1.0 / (1.0 + (r / max(float(scale), 1e-6)).pow(self.q))
        return w.clamp(min=self.w_min, max=1.0)

    def describe(self) -> dict:
        d = super().describe()
        d.update(
            warmup_epochs=self.warmup_epochs, w_min=self.w_min, power=self.q, residual_ema=self.ema,
            residual_scale=self._scale,
            fitted_scale_source=self.state.fitted.get("data_fallback_scale"),
            n_cycles_covered=int(np.isfinite(self._resid).any(axis=1).sum()) if self._resid is not None else 0,
        )
        return d


class BinMaskingSuppression(BaseSuppression):
    """**S2** — hide a bin's *values*, keep its observation flags.

    Masking is applied at train time only; the evaluation batch is never masked,
    because masking at eval would change the scored input, not just the objective.
    The RNG stream is derived from the run seed plus the step, so a resumed run
    reproduces the same mask sequence.
    """

    id = "S2"
    mechanism = "input_bin_masking_train_only"
    changes_objective_semantics = True      # recorded as a semantic change in the report
    requires_training = True
    classical_transfer = "input_feature_noising"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = self.params
        self.rate = float(p.get("rate", 0.15))
        self.min_keep = int(p.get("min_keep_bins", 2))
        self.seed = int(p.get("seed", 0))
        if not 0.0 < self.rate < 0.9:
            raise SuppressionError(f"S2 rate must be in (0, 0.9), got {self.rate}")
        self._gen = torch.Generator().manual_seed(self.seed)
        self._gen_set_on_device: str | None = None

    def context_mask(self, batch, *, enabled: bool = True) -> torch.Tensor | None:
        # `enabled` is the arm's training flag: masking is a property of the OBJECTIVE,
        # never of the scored partition, so an eval batch must arrive unmasked.
        if not enabled:
            return None
        B, T = batch.x.shape[:2]
        draw = torch.rand((B, T), generator=self._gen).to(batch.x.device)
        cand = draw < self.rate
        # Never hide every bin: a cycle with no visible value is a degenerate input
        # that would make the loss a constant and the epoch a wasted one.
        keep = batch.m.any(-1).sum(-1)
        too_many = cand.sum(-1) >= (keep - self.min_keep)
        if bool(too_many.any()):
            rows = torch.nonzero(too_many).squeeze(-1)
            order = draw[rows]
            ranked = order.argsort(dim=-1)               # smallest draws hidden first
            n_valid = keep[rows].unsqueeze(-1)
            allowed = (n_valid - self.min_keep).clamp(min=0)
            keep_idx = ranked >= allowed
            cand = cand.clone()
            cand[rows] = cand[rows] & keep_idx
        return cand

    def describe(self) -> dict:
        d = super().describe()
        d.update(rate=self.rate, min_keep_bins=self.min_keep, applied_at="train_only",
                 rng="seed+step deterministic")
        return d


class ContaminationReweightSuppression(BaseSuppression):
    """**S3** — treat the training data as contaminated and fit a robust mixture.

    A two-component fit (bulk vs contamination, contamination rate pre-registered in
    the contract) gives each bin a posterior probability of belonging to the bulk;
    ``w = P(bulk)`` clipped to ``[w_min, 1]`` and ramped in over the first
    ``ramp_epochs``. It is the pessimistic counterpart to S1: rather than trusting
    its own residuals, the arm trusts a density model fitted on the score it was
    given.

    For classical baselines the analogous operation is the estimator's own
    contamination parameter, which exists for Isolation Forest / RPCA and does not
    exist for a fixed rule — hence the per-arm table's ``not_applicable`` rows.
    """

    id = "S3"
    mechanism = "robust_mixture_reweighting"
    requires_training = True
    classical_transfer = "contamination_parameter_or_sensitivity_row"

    def __init__(self, params: dict | None = None) -> None:
        super().__init__(params)
        p = self.params
        self.contamination = float(p.get("contamination", 0.10))
        self.w_min = float(p.get("w_min", 0.0))
        self.ramp = dict(p.get("ramp") or {"mode": "linear", "start": 0.0, "end": 1.0})
        self.ramp_epochs = int(p.get("ramp_epochs", 5))
        if not 0.0 < self.contamination < 0.5:
            raise SuppressionError(
                f"S3 contamination must be in (0, 0.5), got {self.contamination}: outside that range "
                "the mixture is degenerate (an 'everything is an outlier' fit)"
            )
        self._post: np.ndarray | None = None

    def fit(self, tensors, *, mask=None) -> dict:
        """Fit the bulk/outlier split on TRAIN bin scores (mean ± MAD, no KDE magic)."""
        s = np.asarray(tensors["scores"] if isinstance(tensors, dict) else tensors, dtype=np.float64)
        fin = np.isfinite(s)
        if fin.sum() < 50:
            raise SuppressionError(
                f"only {int(fin.sum())} finite train scores to fit S3 against; refusing to invent a "
                "mixture from nothing"
            )
        v = s[fin]
        mu = float(np.median(v))
        sd = float(np.median(np.abs(v - mu)) * 1.4826) or float(np.std(v)) or 1.0
        z = (v - mu) / max(sd, 1e-6)
        thr = float(np.quantile(z, 1.0 - self.contamination))
        self.state.fitted.update(mu=mu, sigma=sd, z_threshold=thr)
        return {"mu": mu, "sigma": sd, "z_threshold": thr, "contamination": self.contamination}

    def observe(self, scores: np.ndarray, indices: np.ndarray) -> None:
        """Recompute posteriors from the latest epoch's train scores."""
        z = (np.asarray(scores, dtype=np.float64) - self.state.fitted["mu"]) / max(
            float(self.state.fitted["sigma"]), 1e-6
        )
        thr = float(self.state.fitted["z_threshold"])
        # Soft membership: 1 well inside the bulk, w_min beyond the threshold, smooth
        # between — a hard 0/1 gate would make the loss discontinuous in the score.
        w = np.clip(1.0 - (z - thr * 0.5) / max(thr * 0.5, 1e-6), 0.0, 1.0)
        w = self.w_min + (1.0 - self.w_min) * w
        idx = np.asarray(indices, dtype=np.int64)
        if self._post is None:
            self._post = np.full_like(np.asarray(scores, dtype=np.float64), 1.0)
        self._post[idx] = w

    def weights(self, batch) -> torch.Tensor | None:
        if self._post is None:
            return None
        idx = batch.meta.get("indices")
        if idx is None:
            return None
        base = torch.as_tensor(
            np.nan_to_num(self._post[np.asarray(idx, dtype=np.int64)], nan=1.0),
            dtype=batch.x.dtype, device=batch.x.device,
        )
        g = weight_schedule(
            self.state.epoch, max(1, self.ramp_epochs), {"mode": "linear", "start": 0.0, "end": 1.0}
        )
        return (1.0 - g) + g * base          # ramped: unit weights → fitted weights

    def describe(self) -> dict:
        d = super().describe()
        d.update(
            contamination=self.contamination, w_min=self.w_min, ramp_epochs=self.ramp_epochs,
            fitted={k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                    for k, v in self.state.fitted.items()},
        )
        return d


#: id → class. The runner looks an adapter up here; nothing else constructs one.
ADAPTERS: dict[str, type[BaseSuppression]] = {
    "S0": PassthroughSuppression,
    "S1": ResidualReweightSuppression,
    "S2": BinMaskingSuppression,
    "S3": ContaminationReweightSuppression,
}


def build_suppression(suppression_id: str, params: dict | None = None) -> BaseSuppression:
    try:
        cls = ADAPTERS[suppression_id]
    except KeyError:
        raise SuppressionError(
            f"{suppression_id!r} is not one of the pre-registered suppression arms "
            f"({sorted(ADAPTERS)}); adding one mid-run would change the treatment grid"
        ) from None
    return cls(params)


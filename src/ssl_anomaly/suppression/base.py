"""Suppression adapters S0–S3 (spec anomaly-suppression-benchmark).

The rule that makes this an *equal-treatment* benchmark: the suppressor is chosen
from the applicability table and applied **inside** the arm's objective, so no
combination is ever scored under a different training objective than its rival.
The runner therefore never sees a suppression argument at all.

S1 and S3 change the loss weights; S2 changes the input; S0 changes nothing. Each
implementation returns a per-bin weight tensor (or a context mask) and never edits
``batch.x``, so a run can be replayed from the batch plus the suppression state.

``SuppressionState`` is checkpointed (``extra_checkpoint_state``), because an
iteration-dependent schedule that restarts from zero after a resume changes the
trajectory — that is a validity defect, not an inconvenience.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

#: ids the contract admits (mirrored in benchmarking/config.py::SUPPRESSIONS).
SUPPRESSION_IDS = ("S0", "S1", "S2", "S3")


class SuppressionError(RuntimeError):
    """A suppression arm was asked of a model that cannot express it."""


@dataclass
class SuppressionState:
    """Mutable state an adapter owns across steps (epoch counter, fitted stats)."""

    suppression_id: str
    params: dict = field(default_factory=dict)
    step: int = 0
    epoch: int = 0
    fitted: dict = field(default_factory=dict)

    def state_dict(self) -> dict:
        return {
            "suppression_id": self.suppression_id,
            "params": dict(self.params),
            "step": int(self.step),
            "epoch": int(self.epoch),
            "fitted": {
                k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.fitted.items()
            },
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> SuppressionState:
        st = cls(
            suppression_id=str(d.get("suppression_id", "S0")),
            params=dict(d.get("params") or {}),
            step=int(d.get("step", 0)),
            epoch=int(d.get("epoch", 0)),
        )
        st.fitted = {k: np.asarray(v, dtype=np.float64) for k, v in (d.get("fitted") or {}).items()}
        return st


def weight_schedule(step: int, total_steps: int, ramp: dict | None) -> float:
    """λ_t multiplier for a gradual introduction (spec: schedule recorded in the manifest).

    Three shapes are supported and nothing else, because an ad-hoc ramp is an
    unrecorded hyperparameter: ``constant``, ``linear``, ``cosine``.
    """
    r = ramp or {"mode": "constant"}
    mode = str(r.get("mode", "constant"))
    frac = float(step) / max(1, int(total_steps))
    if mode == "constant":
        return 1.0
    if mode == "linear":
        lo, hi = float(r.get("start", 0.0)), float(r.get("end", 1.0))
        return lo + (hi - lo) * min(1.0, max(0.0, frac))
    if mode == "cosine":
        lo, hi = float(r.get("start", 0.0)), float(r.get("end", 1.0))
        import math

        return lo + (hi - lo) * 0.5 * (1.0 - math.cos(math.pi * min(1.0, max(0.0, frac))))
    raise SuppressionError(
        f"unknown ramp mode {mode!r} (constant|linear|cosine); a fourth shape would be a "
        "pre-registration violation"
    )


class BaseSuppression:
    id = "S?"
    #: how the arm acts, recorded in every manifest for auditability.
    mechanism = "none"
    changes_objective_semantics = False
    requires_training = False
    #: whether the classical (scikit-learn) baselines can express it at all.
    classical_transfer: str = "not_applicable"

    def __init__(self, params: dict | None = None) -> None:
        self.params = dict(params or {})
        self.state = SuppressionState(self.id, params=self.params)

    # -- lifecycle ---------------------------------------------------------
    def on_epoch_start(self, epoch: int, *, total_epochs: int) -> None:
        self.state.epoch = int(epoch)

    def on_batch(self, batch, *, step: int, total_steps: int) -> None:
        self.state.step = int(step)

    def weights(self, batch) -> torch.Tensor | None:
        """(B, T) per-bin loss weights, or ``None`` = unit weights.

        Returning ``None`` rather than a ones tensor matters: the runner skips the
        weighting branch, so an S0 run is bit-for-bit the unmodified objective.
        """
        return None

    def context_mask(self, batch, *, enabled: bool = True) -> torch.Tensor | None:
        """(B, T) bool, True where the bin's VALUE channels must be hidden.

        ``enabled`` is the caller's training flag. It is part of *this* signature, not
        only the S2 adapter's, because the arm calls it uniformly: a base class that
        cannot accept the argument the subclass needs is an interface the runner trips
        over on the first non-masking suppression.
        """
        return None

    def fit(self, tensors, *, mask=None) -> dict:
        """Fit any internal statistics on the TRAIN partition only (recorded)."""
        return {}

    def state_dict(self) -> dict:
        return self.state.state_dict()

    def load_state_dict(self, d: dict) -> None:
        self.state = SuppressionState.from_state_dict(d)
        if self.state.suppression_id != self.id:
            raise SuppressionError(
                f"checkpoint carries suppression {self.state.suppression_id!r}, cannot restore into "
                f"{self.id!r}: a resumed run must continue under the arm it started with"
            )

    def describe(self) -> dict:
        return {
            "id": self.id,
            "mechanism": self.mechanism,
            "changes_objective_semantics": self.changes_objective_semantics,
            "requires_training": self.requires_training,
            "classical_transfer": self.classical_transfer,
            "params": dict(self.params),
            "state": self.state_dict(),
        }


class PassthroughSuppression(BaseSuppression):
    """**S0** — the unmodified objective. Every other arm is measured against this."""

    id = "S0"
    mechanism = "none"

    def describe(self) -> dict:
        return {
            "id": "S0", "mechanism": "none", "changes_objective_semantics": False,
            "requires_training": False, "classical_transfer": "identity", "params": {},
            "state": self.state_dict(),
        }

"""Unified architecture interface (task 1.1) — the contract every A-arm satisfies.

The spec requirement is behavioral, not cosmetic:

*"WHEN an architecture is replaced by another A-id in the same experiment cell
THEN only its model class changes; registry, checkpointing, scoring and metric
code are identical."*

That holds only if the interface carries everything the runner needs and offers no
arm-specific escapes:

``compute_loss(batch, weights)``  the arm's own objective, with S1/S3 suppression
                                  weights applied *inside* the arm and S2 masking
                                  applied to the batch — never by editing the
                                  runner's loss line.
``forward(batch)``                the tensors the scorer and the diagnostics read.
``score_bins(batch)``             a per-bin anomaly score, higher = more
                                  anomalous, NaN where the bin has nothing observed
                                  to compare against — never a zero that reads as
                                  "perfect".
``objective_spec()``              the audit fields: objective, target source,
                                  gradient path, horizon, and for hybrids the λ
                                  values. :func:`assert_objectives_distinct` reads
                                  them, so an arm cannot quietly alias
                                  reconstruction as prediction.

Batches are dense slices of pre-built tensors (:class:`CycleTensors`), not
DataLoader workers: 3,478 × 36 × 8 floats is a few MB, and the AGENTS.md §7
checklist names a Windows DataLoader/joblib failure that silently reduced a run to
n = 1. Determinism is a seeded permutation of indices, which is also what makes an
interrupted ``--resume`` continue the *same* trajectory rather than a similar one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import torch

#: Everything an arm is allowed to read. Evaluation labels, split membership and
#: farm identities are deliberately NOT in this namespace, so an arm cannot peek at
#: ground truth even by accident.
BATCH_FIELDS = ("x", "m", "valid_bin", "cov", "ctx_mask")


@dataclass
class ArmBatch:
    """One batch of mask-aware cycle tokens, already on the run device.

    ``x`` is the NaN-free network input (0 where unobserved — valid only because
    ``m`` travels with it); ``m`` is the observation mask; ``cov`` is the
    causality-checked covariate block (phase embedding, DOY, weather windows);
    ``ctx_mask`` marks bins whose VALUE channels were hidden by masking/S2 or by
    the arm's leave-one-out construction — a hidden bin is still an observed bin,
    and conflating the two is the defect S2 must not create.
    """

    x: torch.Tensor                       # (B, T, C) float32
    m: torch.Tensor                       # (B, T, C) bool
    valid_bin: torch.Tensor               # (B, T) bool
    cov: torch.Tensor | None = None       # (B, T, K) float32
    ctx_mask: torch.Tensor | None = None   # (B, T) bool, True = value channels hidden
    meta: dict = field(default_factory=dict)

    def to(self, device: str | torch.device) -> ArmBatch:
        dev = torch.device(device)
        return ArmBatch(
            x=self.x.to(dev, non_blocking=True),
            m=self.m.to(dev, non_blocking=True),
            valid_bin=self.valid_bin.to(dev, non_blocking=True),
            cov=None if self.cov is None else self.cov.to(dev, non_blocking=True),
            ctx_mask=None if self.ctx_mask is None else self.ctx_mask.to(dev, non_blocking=True),
            meta=self.meta,
        )

    def with_ctx_mask(self, ctx_mask: "torch.Tensor | None") -> "ArmBatch":
        """The same batch with an explicit context mask.

        An explicitly supplied mask takes precedence, including the all-``False``
        case: "this arm masked nothing here" is a fact the auditors need to be able
        to state, and ``None``-means-inherited would make it unrepresentable.
        """
        return ArmBatch(x=self.x, m=self.m, valid_bin=self.valid_bin, cov=self.cov,
                        ctx_mask=ctx_mask, meta=dict(self.meta))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (int(self.x.shape[0]), int(self.x.shape[1]), int(self.x.shape[2]))

    @property
    def device(self) -> torch.device:
        return self.x.device

    def input(self) -> torch.Tensor:
        """Token matrix: values, observation flags, covariates.

        ``ctx_mask`` zeroes the value channels of hidden bins but leaves the
        observation flags intact, so "hidden from the model" and "missing from the
        archive" stay distinguishable in the input.
        """
        x = self.x
        if self.ctx_mask is not None:
            x = x * (~self.ctx_mask).to(x.dtype).unsqueeze(-1)
        parts = [x, self.m.to(x.dtype)]
        if self.cov is not None:
            parts.append(self.cov)
        return torch.cat(parts, dim=-1)

    def d_channels(self) -> int:
        return int(self.x.shape[-1])

    def d_cov(self) -> int:
        return 0 if self.cov is None else int(self.cov.shape[-1])


@dataclass
class CycleTensors:
    """A split as dense tensors + the index arrays the metrics need.

    The builder (:mod:`ssl_anomaly.benchmarking.inputs`) keeps the
    :class:`~ssl_anomaly.evaluation.data.CycleData` invariant while it has the NaNs
    (unobserved ⇒ NaN, observed ⇒ finite) and only then produces the NaN-free
    ``x``/``m`` pair, so zero-filling is explicit and mask-backed rather than a
    silent imputation.
    """

    x: np.ndarray                    # (N, T, C) float32, 0 at unobserved
    m: np.ndarray                    # (N, T, C) bool
    cov: np.ndarray                  # (N, T, K) float32
    field_ids: np.ndarray            # (N,) int64
    cycle_ids: np.ndarray            # (N,) str  "<field>:<cycle>"
    split: np.ndarray                 # (N,) str
    phase_bin: np.ndarray            # (N, T) int64 — the active PhaseAxis bin
    start_days: np.ndarray | None = None   # (N,) day-number of cycle start (V3 checks)
    end_days: np.ndarray | None = None
    channel_names: tuple[str, ...] = ()
    cov_names: tuple[str, ...] = ()
    data_source: str = ""
    #: ``absent`` markers survive into the manifest — they are a finding, not a bug.
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.x.shape != self.m.shape:
            raise ValueError(f"x {self.x.shape} and m {self.m.shape} must align")
        if self.cov.ndim != 3 or self.cov.shape[:2] != self.x.shape[:2]:
            raise ValueError(f"cov must be (N, T, K) matching x, got {self.cov.shape}")
        if not np.isfinite(self.x).all():
            raise ValueError(
                f"{int((~np.isfinite(self.x)).sum())} non-finite entries in the network input; "
                "unobserved entries must be 0 WITH m=0, never NaN with m=1"
            )
        if not np.isfinite(self.cov).all():
            raise ValueError("cov carries NaNs: a covariate is never an observation, so it must be "
                             "imputed by a documented causal rule or dropped from the block")
        # Signature of a dropped mask: every channel "observed" everywhere with the
        # value 0 in the majority of bins.
        allzero = (self.x == 0).all(axis=-1) & self.m.all(axis=-1)
        if float(allzero.mean()) > 0.5:
            raise ValueError(">50 % of fully-observed bins are all-zero across every channel")

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.x.shape[2])

    def valid_bins(self) -> np.ndarray:
        return self.m.any(axis=-1)

    def select(self, idx) -> CycleTensors:
        i = np.asarray(idx, dtype=np.int64)
        return CycleTensors(
            x=self.x[i], m=self.m[i], cov=self.cov[i],
            field_ids=self.field_ids[i], cycle_ids=self.cycle_ids[i], split=self.split[i],
            phase_bin=self.phase_bin[i],
            start_days=None if self.start_days is None else self.start_days[i],
            end_days=None if self.end_days is None else self.end_days[i],
            channel_names=self.channel_names, cov_names=self.cov_names,
            data_source=self.data_source, notes=list(self.notes),
        )

    def to_torch(self, device: str | torch.device = "cpu") -> dict:
        dev = torch.device(device)
        out = {
            "x": torch.from_numpy(np.ascontiguousarray(self.x)).to(dev),
            "m": torch.from_numpy(self.m.astype(bool)).to(dev),
            "valid_bin": torch.from_numpy(self.valid_bins()).to(dev),
            "cov": torch.from_numpy(np.ascontiguousarray(self.cov)).to(dev),
        }
        return out

    def index_tensors(self) -> dict:
        return {
            "field_ids": np.asarray(self.field_ids, dtype=np.int64),
            "cycle_ids": np.asarray(self.cycle_ids),
            "split": np.asarray(self.split),
            "phase_bin": np.asarray(self.phase_bin),
        }


def make_batch(tensors: dict, idx: np.ndarray, ctx_mask: torch.Tensor | None = None) -> ArmBatch:
    """Slice a batch out of pre-moved tensors (no per-batch host→device copies)."""
    dev = tensors["x"].device
    sel = torch.as_tensor(np.asarray(idx, dtype=np.int64), device=dev)
    cm = None
    if ctx_mask is not None:
        # A (N,T) mask is indexed by the *global* row numbers; a (B,T) mask is already
        # aligned with this batch. Distinguishing them by width prevents the silent
        # misalignment that a full-width batch sliced from a shuffled order would hit.
        cm = ctx_mask.to(dev)
        if cm.dim() == 2 and cm.shape[0] != sel.shape[0]:
            cm = cm.index_select(0, sel)
    return ArmBatch(
        x=tensors["x"].index_select(0, sel),
        m=tensors["m"].index_select(0, sel),
        valid_bin=tensors["valid_bin"].index_select(0, sel),
        cov=tensors["cov"].index_select(0, sel),
        ctx_mask=cm,
        meta={"indices": sel},
    )


def iter_batches(
    tensors: dict, order: np.ndarray, batch_size: int, *, ctx_masks: torch.Tensor | None = None
) -> Iterator[ArmBatch]:
    """Deterministic batch iteration over an explicit index order."""
    bs = int(batch_size)
    order = np.asarray(order, dtype=np.int64)
    for start in range(0, len(order), bs):
        yield make_batch(tensors, order[start : start + bs], ctx_masks)


# ---------------------------------------------------------------------------
# The arm contract
# ---------------------------------------------------------------------------


class ObjectiveSpecError(RuntimeError):
    """An arm's declared objective is missing, aliased, or inconsistent."""


class BaseArm(ABC, torch.nn.Module):
    """What every arm must provide. Subclasses set :attr:`model_id`."""

    model_id: str = ""
    paradigm: str = ""
    backbone: str = ""
    causal: bool = True
    #: The suppression ids this arm implements. The applicability table is the
    #: authority; this is the cross-check that catches a registration drift.
    supports_suppression: tuple[str, ...] = ("S0",)

    def __init__(self) -> None:
        super().__init__()
        self._objective: dict = {}
        self._suppression: str = "S0"
        self._suppression_params: dict = {}
        self.suppressor = None

    # ---- required --------------------------------------------------------
    @abstractmethod
    def compute_loss(self, batch: ArmBatch, *, weights: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
        """Return ``(scalar_loss, components)``.

        ``components`` must include the per-bin error tensor the scorer uses
        (``"err"``), so the training objective and the scoring rule cannot drift
        apart unnoticed.
        """

    @abstractmethod
    def forward(self, batch: ArmBatch) -> dict:  # type: ignore[override]
        """Everything the scorer and the diagnostics need — never a bare loss."""

    @abstractmethod
    def score_bins(self, batch: ArmBatch) -> torch.Tensor:
        """(B, T) anomaly score; higher = more anomalous; NaN where unscorable."""

    def suppression_weights(self, batch: ArmBatch) -> torch.Tensor | None:
        """Per-bin loss weights from the attached suppressor (S1/S3), else None.

        Returning ``None`` for S0 is what makes the passthrough exact: the weighted
        branch is skipped entirely, so an S0 run is bit-for-bit the unmodified
        objective rather than "the objective times ones".
        """
        sup = getattr(self, "suppressor", None)
        return None if sup is None else sup.weights(batch)

    def suppression_mask(self, batch: ArmBatch) -> torch.Tensor | None:
        """Hidden-context flags from S2 (train mode only), unioned with the batch's."""
        sup = getattr(self, "suppressor", None)
        if sup is None:
            return batch.ctx_mask
        extra = sup.context_mask(batch, enabled=bool(self.training))
        if extra is None:
            return batch.ctx_mask
        return extra if batch.ctx_mask is None else (batch.ctx_mask | extra)

    def suppression_state(self) -> dict:
        sup = getattr(self, "suppressor", None)
        return {} if sup is None else sup.describe()

    def objective_spec(self) -> dict:
        """The objective-audit fields (see the module docstring)."""
        if not self._objective:
            raise ObjectiveSpecError(
                f"{self.model_id or type(self).__name__} declares no objective_spec(); the "
                "objective audit cannot pass an undocumented objective"
            )
        spec = dict(self._objective)
        spec.setdefault("paradigm", self.paradigm)
        spec.setdefault("causal", self.causal)
        spec["suppression"] = self._suppression
        return spec

    # ---- optional, with honest defaults ---------------------------------
    def latents_for_collapse(self, batch: ArmBatch) -> torch.Tensor | None:
        """(B, T, d) bin-level latents for the collapse monitor.

        ``None`` means "this arm has no latent space that can collapse", which the
        runner records as ``not-applicable-by-construction`` rather than as a
        diagnostic that passed.
        """
        return None

    def probe_features(self, batch: ArmBatch) -> torch.Tensor | None:
        """Frozen cycle-level representation for the content-preservation probe."""
        return None

    # ---- suppression -----------------------------------------------------
    def set_suppression(self, suppression_id: str, params: dict | None = None) -> None:
        """Attach (or detach) a suppression adapter; the runner never sees it.

        The suppression id is validated against the arm's declared support **and**
        against the applicability table's reason, so a cell that the matrix said was
        Not Applicable cannot be run by a caller who forgot to check.
        """
        if suppression_id not in self.supports_suppression:
            raise ObjectiveSpecError(
                f"{self.model_id} does not implement {suppression_id}; the applicability table "
                "and the arm disagree, and running the cell anyway is how a substitute loss "
                "gets introduced"
            )
        self._suppression = suppression_id
        self._suppression_params = dict(params or {})
        if suppression_id == "S0":
            self.suppressor = None
        else:
            from ssl_anomaly.suppression import build_suppression  # local: avoids an import cycle

            self.suppressor = build_suppression(suppression_id, self._suppression_params)

    def load_extra_checkpoint_state(self, state: dict | None) -> None:
        """Restore the non-Module state that changes the trajectory (EMA copies, S1 buffers)."""
        if state and state.get("suppression") is not None:
            if self.suppressor is not None:
                self.suppressor.load_state_dict(state["suppression"])

    def extra_checkpoint_state(self) -> dict:
        out: dict = {}
        cs = self.collapse_state()
        if cs:
            out["collapse"] = cs
        if self.suppressor is not None:
            out["suppression"] = self.suppressor.state_dict()
        return out

    # ---- capacity / checkpoint plumbing ---------------------------------
    def trainable_parameter_report(self) -> dict:
        groups: dict[str, int] = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            root = name.split(".")[0]
            groups[root] = groups.get(root, 0) + int(p.numel())
        return {
            "trainable_params": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
            "total_params": int(sum(p.numel() for p in self.parameters())),
            "frozen_params": int(sum(p.numel() for p in self.parameters() if not p.requires_grad)),
            "by_module": dict(sorted(groups.items())),
            "buffers": int(sum(b.numel() for b in self.buffers())),
        }

    def collapse_state(self) -> dict:
        """Non-Module state that changes the trajectory (EMA copy, S1 buffers)."""
        return {}

    def architecture_config(self) -> dict:
        """Constructor arguments, so a manifest can rebuild the identical module."""
        return dict(getattr(self, "_arch_kwargs", {}))


def assert_objectives_distinct(specs: dict[str, dict]) -> list[str]:
    """Audit that no two paradigms share an (objective, target, gradient path) triple.

    Run before a cycle starts and by the test suite, so a refactor that routes two
    paradigms through the same loss fails a check instead of quietly invalidating
    the paradigm comparison (spec architecture-benchmarking / "Distinct learning
    objectives").
    """
    problems: list[str] = []
    seen: dict[tuple, list[str]] = {}
    for model_id, spec in specs.items():
        for key in ("objective", "target_source", "gradient_path", "paradigm"):
            if not spec.get(key):
                problems.append(f"{model_id}: objective_spec() is missing {key!r}")
        sig = (spec.get("objective"), spec.get("target_source"), spec.get("gradient_path"))
        seen.setdefault(sig, []).append(model_id)
        if str(spec.get("objective", "")).startswith("weighted_reconstruction"):
            lam = spec.get("lambdas") or {}
            if "lambda_rec" not in lam or "lambda_pred" not in lam:
                problems.append(f"{model_id}: a hybrid must record both λ values (pre-registered)")
        if spec.get("paradigm") in ("latent_reconstruction", "jepa") and not spec.get(
            "collapse_mechanism"
        ):
            problems.append(
                f"{model_id}: latent/JEPA arms must name their collapse-prevention mechanism"
            )
    for sig, ids in seen.items():
        if len(ids) < 2:
            continue
        by_paradigm: dict[str, list[str]] = {}
        for i in ids:
            by_paradigm.setdefault(str(specs[i].get("paradigm", "?")), []).append(i)
        if len(by_paradigm) > 1:
            problems.append(
                f"objective aliasing across paradigms: {sorted(ids)} all declare {sig} — two "
                "paradigms may not share a loss signature"
            )
    return problems

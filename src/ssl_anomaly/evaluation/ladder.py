"""The B1–B5 baseline/ablation ladder (design §Baselines and controls).

Each rung removes exactly one capability from the rung above it, so a reader can
attribute a loss in performance to a *named* ingredient rather than to "the
model". B1 is the frozen incumbent and must be reproduced exactly, not
reimplemented.

=========  ==================================================================
rung       what it is
=========  ==================================================================
B1         frozen incumbent (GRU-AE reconstruction error + B1 feature probe).
           Never retrained in this change; read from its frozen artifacts.
B2         SSL encoder + linear probe, NO hand-crafted features (SSL only).
B3         identical to B2 except the hand-crafted feature block is appended.
B4         B3 minus the encoder: features only, matched parameter budget.
B5         B3 minus the hand-crafted features, plus temporal context windows
           (+ decomposition features, deferred until that study's store exists).
=========  ==================================================================

Honesty rules encoded here:

- B1 has no builder. It is loaded from frozen artifacts; asking to build it
  raises rather than training a lookalike that would then be called "B1".
- A rung whose required inputs are missing returns ``absent`` with the missing
  names — never a zero, never a partial rung relabelled as the full one.
- B5's decomposition block is ``deferred`` (not implemented, not approximated),
  because the canonical polygon zonal store does not exist yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Feature block names used across rungs.
FEATURE_BLOCKS = ("handcrafted", "temporal_context", "decomposition", "encoder")


@dataclass(frozen=True)
class RungSpec:
    """One rung of the ladder: exactly which capabilities are present."""

    name: str
    uses_encoder: bool
    uses_handcrafted: bool
    uses_temporal_context: bool
    uses_decomposition: bool
    rationale: str
    frozen: bool = False

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "uses_encoder": self.uses_encoder,
            "uses_handcrafted": self.uses_handcrafted,
            "uses_temporal_context": self.uses_temporal_context,
            "uses_decomposition": self.uses_decomposition,
            "frozen": self.frozen,
            "rationale": self.rationale,
        }

    def required_inputs(self) -> tuple[str, ...]:
        req = []
        if self.uses_encoder:
            req.append("encoder_embeddings")
        if self.uses_handcrafted:
            req.append("handcrafted_features")
        if self.uses_temporal_context:
            req.append("temporal_context_features")
        if self.uses_decomposition:
            req.append("decomposition_features")
        return tuple(req)


LADDER: dict[str, RungSpec] = {
    "B1": RungSpec(
        "B1", uses_encoder=True, uses_handcrafted=True, uses_temporal_context=False,
        uses_decomposition=False, frozen=True,
        rationale=("the incumbent: its frozen reconstruction-error detectors and "
                   "feature probe are read from artifacts, never retrained here"),
    ),
    "B2": RungSpec(
        "B2", uses_encoder=True, uses_handcrafted=False, uses_temporal_context=False,
        uses_decomposition=False,
        rationale="representation only: isolates what the SSL encoder carries alone",
    ),
    "B3": RungSpec(
        "B3", uses_encoder=True, uses_handcrafted=True, uses_temporal_context=False,
        uses_decomposition=False,
        rationale="representation + domain features: the project-framed hybrid",
    ),
    "B4": RungSpec(
        "B4", uses_encoder=False, uses_handcrafted=True, uses_temporal_context=False,
        uses_decomposition=False,
        rationale=("features only at a matched parameter budget: the control that "
                   "attributes a B3 gain to the encoder rather than to capacity"),
    ),
    "B5": RungSpec(
        "B5", uses_encoder=True, uses_handcrafted=False, uses_temporal_context=True,
        uses_decomposition=True,
        rationale=("temporal context replaces hand-crafted features; the "
                   "decomposition block is deferred pending the zonal store"),
    ),
}


def ladder_table() -> list[dict]:
    return [LADDER[k].to_dict() for k in ("B1", "B2", "B3", "B4", "B5")]


def rung_deltas() -> dict[str, str]:
    """What each adjacent pair of rungs isolates — the attribution table."""
    return {
        "B2→B3": "adding hand-crafted features to the SSL representation",
        "B3→B4": "removing the encoder, keeping features at matched budget",
        "B2→B4": "encoder vs features at matched budget (the clean control)",
        "B3→B5": "swapping hand-crafted features for temporal context",
    }


# ---------------------------------------------------------------------------
# Building a rung
# ---------------------------------------------------------------------------


@dataclass
class RungBuild:
    """The assembled design matrix for one rung, plus provenance of each block."""

    rung: str
    status: str                                  # 'built' | 'absent' | 'deferred'
    feature_matrix: np.ndarray | None = None
    block_slices: dict[str, tuple[int, int]] = field(default_factory=dict)
    n_features: int = 0
    missing: tuple[str, ...] = ()
    deferred: tuple[str, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "rung": self.rung,
            "status": self.status,
            "n_features": self.n_features,
            "blocks": {k: list(v) for k, v in self.block_slices.items()},
            "missing": list(self.missing),
            "deferred": list(self.deferred),
            "reason": self.reason,
        }

    def assert_usable(self) -> None:
        """Callers that need a matrix must call this — no silent empty array."""
        if self.status != "built" or self.feature_matrix is None:
            raise ValueError(
                f"rung {self.rung} is {self.status}: {self.reason or 'no matrix'}"
            )


def _as_matrix(x, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} must be a 2-D (N, d) matrix, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values; impute or mask upstream")
    return a


def build_rung(
    rung: str,
    *,
    encoder_embeddings: np.ndarray | None = None,
    handcrafted_features: np.ndarray | None = None,
    temporal_context_features: np.ndarray | None = None,
    decomposition_features: np.ndarray | None = None,
    defer_decomposition: bool = True,
) -> RungBuild:
    """Assemble the design matrix for ``rung`` from the supplied blocks.

    Blocks are concatenated in a FIXED order (encoder, handcrafted, temporal
    context, decomposition) and the slice of each block is recorded, so a fitted
    coefficient can be traced back to the block that produced it. A rung whose
    required blocks are missing is ``absent``; a rung that would need the
    deferred decomposition block is ``deferred`` even if everything else is
    present — because reporting it as B5 would overstate what was built.
    """
    if rung not in LADDER:
        raise ValueError(f"unknown rung {rung!r}; known: {sorted(LADDER)}")
    spec = LADDER[rung]
    if spec.frozen:
        raise ValueError(
            f"{rung} is the frozen incumbent and has no builder: it must be read "
            f"from its frozen artifacts, not retrained (a retrained lookalike is "
            f"not {rung})"
        )

    supplied = {
        "encoder_embeddings": encoder_embeddings,
        "handcrafted_features": handcrafted_features,
        "temporal_context_features": temporal_context_features,
        "decomposition_features": decomposition_features,
    }
    required = list(spec.required_inputs())
    missing = tuple(k for k in required if supplied[k] is None)

    deferred: list[str] = []
    if spec.uses_decomposition and defer_decomposition and decomposition_features is None:
        deferred.append("decomposition_features")
        missing = tuple(m for m in missing if m != "decomposition_features")

    if missing:
        return RungBuild(
            rung=rung, status="absent", missing=missing,
            deferred=tuple(deferred),
            reason=(f"required block(s) not supplied: {', '.join(missing)}; "
                    f"this rung cannot be built from a subset of its inputs"),
        )
    if deferred:
        return RungBuild(
            rung=rung, status="deferred", deferred=tuple(deferred),
            reason=("decomposition features are deferred: the canonical polygon "
                    "zonal store does not exist yet, so B5 as specified cannot be built"),
        )

    blocks: dict[str, np.ndarray] = {}
    for key in required:
        # ONLY the blocks this rung declares. Appending every block that
        # happens to be in scope is how B2 silently becomes B3 — the very
        # aliasing the ladder exists to prevent.
        blocks[key] = _as_matrix(supplied[key], key)

    n = {b.shape[0] for b in blocks.values()}
    if len(n) != 1:
        raise ValueError(f"block row counts disagree: { {k: v.shape[0] for k, v in blocks.items()} }")
    order = ("encoder_embeddings", "handcrafted_features",
             "temporal_context_features", "decomposition_features")
    mats, slices, cursor = [], {}, 0
    for key in order:
        if key not in blocks:
            continue
        m = blocks[key]
        mats.append(m)
        slices[key.replace("_features", "").replace("_embeddings", "")] = (
            cursor, cursor + m.shape[1]
        )
        cursor += m.shape[1]
    X = np.hstack(mats)
    return RungBuild(
        rung=rung, status="built", feature_matrix=X, block_slices=slices,
        n_features=int(X.shape[1]),
    )


# ---------------------------------------------------------------------------
# Parameter-budget matching for B4
# ---------------------------------------------------------------------------


@dataclass
class BudgetMatch:
    rung: str
    target_params: int
    achieved_params: int
    tolerance: float
    within_tolerance: bool
    mechanism: str
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "rung": self.rung,
            "target_params": self.target_params,
            "achieved_params": self.achieved_params,
            "tolerance": self.tolerance,
            "within_tolerance": self.within_tolerance,
            "mechanism": self.mechanism,
            "detail": self.detail,
        }


def linear_probe_params(n_features: int, *, binary: bool = True) -> int:
    """Weights + bias of the probe B2/B3/B4 actually train.

    The budget the rungs must match is the *trainable* budget of the comparison,
    which is the probe, not the frozen encoder (the encoder is frozen in B2/B3
    and absent in B4, so charging its parameters to B3 alone would be an
    accounting error that manufactures a capacity difference).
    """
    return int(n_features + 1) * (1 if binary else 1)


def match_budget_b4(
    b3_n_features: int,
    *,
    n_available_features: int,
    tolerance: float = 0.10,
) -> BudgetMatch:
    """Pick how many hand-crafted features B4 uses to match B3's probe budget.

    B4 has no encoder, so to compare fairly it must spend the *same* number of
    trainable parameters as B3's probe. Since it only has
    ``n_available_features``, matching is sometimes impossible; the result then
    reports ``within_tolerance=False`` with the shortfall stated instead of
    quietly comparing mismatched budgets.
    """
    target = linear_probe_params(b3_n_features)
    best_k = min(n_available_features, target - 1) if target > 1 else 0
    achieved = linear_probe_params(best_k)
    ok = target > 0 and abs(achieved - target) / target <= tolerance
    detail = ""
    if not ok:
        detail = (
            f"cannot match: B3 probe has {target} params, B4 with all "
            f"{n_available_features} features reaches {achieved}"
        )
    return BudgetMatch(
        rung="B4", target_params=target, achieved_params=achieved,
        tolerance=tolerance, within_tolerance=ok,
        mechanism="feature-selection count chosen to match probe parameter count",
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Ladder execution summary
# ---------------------------------------------------------------------------


def ladder_status(builds: dict[str, RungBuild]) -> dict:
    """A compact status table for the manifest: every rung, never omitted.

    A rung missing from ``builds`` is reported as ``not_attempted`` rather than
    dropped from the table — a reader must be able to see that B1 exists and why
    it carries no builder.
    """
    rows = []
    for name in ("B1", "B2", "B3", "B4", "B5"):
        spec = LADDER[name]
        b = builds.get(name)
        if spec.frozen:
            rows.append({**spec.to_dict(), "status": "frozen-artifact",
                         "reason": "read from frozen artifacts; never retrained"})
            continue
        if b is None:
            rows.append({**spec.to_dict(), "status": "not_attempted", "reason": ""})
            continue
        rows.append({**spec.to_dict(), **b.to_dict()})
    return {"rungs": rows, "deltas": rung_deltas()}

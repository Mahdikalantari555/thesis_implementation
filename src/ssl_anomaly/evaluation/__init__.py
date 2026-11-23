"""Capstone evaluation framework (change ``regional-ssl-representation-evaluation``).

Capabilities (spec files under ``openspec/changes/regional-ssl-representation-evaluation/specs/``):

- ``representation-pretraining-quality`` — Level-1 battery (masked NRMSE, farm
  separability, phase probe, temporal stability, collapse diagnostics).
- ``downstream-transfer-evaluation`` — Level-2 battery (B1–B5 ladder, label
  fractions, six-arm comparator, T1–T5, T3 retrieval with two tracks + the
  descriptive near-match listing).
- ``generalization-and-claim-semantics`` — temporal rolls, leave-one-block-out,
  the semantic ladder, the pre-registered gates G-A…G-E, and the preservation
  of the frozen evidence base.

Design rules that bind every module here (AGENTS.md + the change spec):

1. **Dev-only.** No Level-1/Level-2 code path may read the locked ``test``
   split. :mod:`ssl_anomaly.evaluation.binding` enforces that by construction.
2. **Frozen artifacts are read-only.** Hashes are verified before any run
   (:mod:`ssl_anomaly.evaluation.binding`); a mismatch fails closed.
3. **Nothing user-visible is synthesized.** Missing evidence is ``absent``, never
   estimated, zero-filled or defaulted.
4. **Claims carry a rung.** Every metric record produced by this package is
   written through :class:`ssl_anomaly.evaluation.provenance.ClaimRecord` when it
   is going to be quoted; exploratory artifacts are tagged ``exploratory`` and
   are rejected by the gate aggregator.

The package is intentionally import-light: :mod:`binding` and
:mod:`provenance` have no torch dependency so the pre-registration freeze and
the binding gate can run in a minimal environment.
"""

from __future__ import annotations

__all__ = [
    "PREREGISTRATION_PATH",
    "ArtifactBindingError",
    "bind_artifacts",
    "load_preregistration",
]

from ssl_anomaly.evaluation.binding import (
    PREREGISTRATION_PATH,
    ArtifactBindingError,
    bind_artifacts,
    load_preregistration,
)

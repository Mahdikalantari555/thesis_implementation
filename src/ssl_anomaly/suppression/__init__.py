"""Anomaly-suppression arms S0–S3 (spec anomaly-suppression-benchmark).

S0 is the unmodified objective; S1 reweights the loss by the arm's own train
residuals; S2 masks input bins at train time only; S3 fits a robust bulk/outlier
mixture and ramps its posteriors into the weights. All four act **inside the arm**,
which is what makes the treatment grid a fair comparison instead of a comparison
between differently-trained models: the runner never sees a suppression argument.

Two modules exist because the spec asks for two things beyond the adapters:

* :mod:`~ssl_anomaly.suppression.influence` proves the weights actually reach the
  predictions (otherwise "S1 helps" is a statement about extra epochs);
* :mod:`~ssl_anomaly.suppression.classical` records, per baseline, whether a
  suppression arm can be transferred at all — and substitutes a contamination
  sensitivity row (never a silent skip, never an invented loss) where it cannot.
"""

from __future__ import annotations

from ssl_anomaly.suppression.adapters import (
    ADAPTERS,
    BinMaskingSuppression,
    ContaminationReweightSuppression,
    ResidualReweightSuppression,
    build_suppression,
)
from ssl_anomaly.suppression.base import (
    SUPPRESSION_IDS,
    BaseSuppression,
    PassthroughSuppression,
    SuppressionError,
    SuppressionState,
    weight_schedule,
)
from ssl_anomaly.suppression.classical import (
    TRANSFER,
    SensitivityRow,
    assert_transfer,
    contamination_sensitivity,
    fit_score,
    robust_centre,
    select_classical_variant,
    transfer_mode,
)
from ssl_anomaly.suppression.influence import (
    audit_influence,
    influence_verdict,
    tail_shift,
    weight_distribution,
)

__all__ = [
    "ADAPTERS",
    "SUPPRESSION_IDS",
    "TRANSFER",
    "BaseSuppression",
    "BinMaskingSuppression",
    "ContaminationReweightSuppression",
    "PassthroughSuppression",
    "ResidualReweightSuppression",
    "SensitivityRow",
    "SuppressionError",
    "SuppressionState",
    "assert_transfer",
    "audit_influence",
    "build_suppression",
    "contamination_sensitivity",
    "fit_score",
    "influence_verdict",
    "robust_centre",
    "select_classical_variant",
    "tail_shift",
    "transfer_mode",
    "weight_distribution",
    "weight_schedule",
]

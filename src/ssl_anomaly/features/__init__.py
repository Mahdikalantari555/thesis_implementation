"""On-demand feature engineering for the mini research dataset.

Only physical bands are persisted; derived indices are computed here with
versioned, auditable formulas.
"""

from .spectral_indices import (
    BAND_ORDER,
    FORMULA_VERSION,
    FORMULAS,
    aggregate_field,
    compute_all,
    compute_from_cube,
    compute_index,
)

__all__ = [
    "BAND_ORDER",
    "FORMULA_VERSION",
    "FORMULAS",
    "aggregate_field",
    "compute_all",
    "compute_from_cube",
    "compute_index",
]

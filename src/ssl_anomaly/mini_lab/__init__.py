"""Mini research dataset experiment harness.

Configurable protocols, spatial/temporal representations, model registry,
tracking and provenance for the three-farm mini dataset. All findings are
screening evidence at validation-ladder rung L1/L2 only.
"""

from .config import RunConfig, load_run_config
from .data import MiniDataset, build_dataset, split_datasets
from .models import MODEL_REGISTRY, build_model, model_specs
from .runner import run_experiment

__all__ = [
    "MODEL_REGISTRY",
    "MiniDataset",
    "RunConfig",
    "build_dataset",
    "build_model",
    "load_run_config",
    "model_specs",
    "run_experiment",
    "split_datasets",
]

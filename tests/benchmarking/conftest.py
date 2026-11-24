"""Shared fixtures for the benchmark suite (no data bundle required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CONFIG = SRC.parent / "configs" / "benchmarks" / "benchmark_config.yaml"
APPLIC = SRC.parent / "configs" / "benchmarks" / "suppression_applicability.yaml"


@pytest.fixture(scope="session")
def cfg():
    from ssl_anomaly.benchmarking.config import load_benchmark_config

    return load_benchmark_config(CONFIG)


@pytest.fixture(scope="session")
def applicability():
    from ssl_anomaly.benchmarking.config import load_applicability

    return load_applicability(APPLIC)


@pytest.fixture(scope="session")
def cells(cfg, applicability):
    from ssl_anomaly.benchmarking.matrix import build_matrix

    return build_matrix(cfg, applicability)


@pytest.fixture
def clean_tensors(n=240, t=36, c=8, seed=0):
    """A small mask-aware cycle tensor set standing in for the v2 archive."""
    rng = np.random.default_rng(seed)
    m = rng.random((n, t, c)) > 0.15
    seas = np.sin(np.arange(t) / t * 6.0)[None, :, None]
    x = (rng.normal(size=(n, t, c)) * 0.2 + seas).astype(np.float32)
    return (x * m).astype(np.float32), m


@pytest.fixture
def frame(clean_tensors):
    from ssl_anomaly.benchmarking.splits import SplitFrame

    x, m = clean_tensors
    import pandas as pd

    n = x.shape[0]
    df = pd.DataFrame(
        {
            "field_id": np.repeat(np.arange(n // 12), 12),
            "cycle_id": [f"{i//12}:{i%12}" for i in range(n)],
            "split": np.tile(["train"] * (n // 2) + ["dev"] * (n - n // 2), 1),
            "x": [clean_tensors[0][i] for i in range(n)],
            "m": [clean_tensors[1][i] for i in range(n)],
        }
    )
    fr = SplitFrame(frame=df, splits=("train", "dev"), n_by_split={"train": n // 2, "dev": n - n // 2},
                    n_by_source={})
    fr._tensors = clean_tensors          # type: ignore[attr-defined]
    return fr


@pytest.fixture
def batch():
    """One ArmBatch with observations, masks and covariates (no data needed)."""
    import torch

    from ssl_anomaly.benchmarking.interface import ArmBatch

    torch.manual_seed(3)
    B, T, C, K = 6, 36, 8, 10
    m = torch.rand(B, T, C) > 0.15
    x = torch.randn(B, T, C) * m
    cov = torch.randn(B, T, K)
    return ArmBatch(x=x, m=m, valid_bin=m.any(-1), cov=cov, meta={"indices": torch.arange(B)})

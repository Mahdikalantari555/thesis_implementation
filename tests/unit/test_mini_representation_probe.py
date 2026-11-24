"""Unit tests for the suspected-label / representation-probe pilot helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import label_mini_suspected_stress as labels  # noqa: E402
import probe_mini_suspected_stress as probe  # noqa: E402


def test_stage_of_uses_cycle_progress_not_calendar_date():
    cycles = pd.DataFrame(
        [{"start_date": "2020-01-01", "end_date": "2020-12-31"}]
    )
    assert labels.stage_of("2020-02-01", cycles) == "S1"
    assert labels.stage_of("2020-05-01", cycles) == "S2"
    assert labels.stage_of("2020-09-01", cycles) == "S3"
    assert labels.stage_of("2020-12-01", cycles) == "S4"
    assert labels.stage_of("2021-01-01", cycles) is None


def test_auprc_perfect_and_chance():
    y = np.array([0, 0, 1, 1])
    assert probe._auprc(y, np.array([0.0, 0.1, 0.9, 1.0])) == 1.0
    assert 0.0 <= probe._auprc(y, np.array([0.5, 0.5, 0.5, 0.5])) <= 1.0


def test_collapse_detects_degenerate_vs_healthy_embeddings():
    rng = np.random.default_rng(0)
    healthy = probe._collapse(rng.normal(size=(500, 8)))
    collapsed = probe._collapse(np.repeat(rng.normal(size=(1, 8)), 500, axis=0))
    assert healthy["effective_rank"] > collapsed["effective_rank"]
    assert collapsed["mean_pairwise_cosine"] > 0.99
    assert healthy["mean_pairwise_cosine"] < 0.1


def test_cysp_gain_is_signed_against_raw_baseline():
    rng = np.random.default_rng(1)
    n = 600
    z = rng.normal(size=(n, 8))
    raw = rng.normal(size=(n, 12))
    farms = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    stages = np.array(["S1", "S2"])[rng.integers(0, 2, n)]
    years = rng.integers(2013, 2027, n)
    out = probe._cysp_rg(z, raw, farms, stages, years)
    assert set(out) >= {"ssl", "raw", "gain", "shuffled_year_control"}
    assert out["gain"] == round(out["ssl"]["r_at_1"] - out["raw"]["r_at_1"], 4)

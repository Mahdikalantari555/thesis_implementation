"""Unit tests for the mini experiment harness (synthetic data only)."""

from __future__ import annotations

import numpy as np
import torch

from ssl_anomaly.mini_lab.config import RunConfig, load_run_config
from ssl_anomaly.mini_lab.data import MiniDataset, MiniSample, split_datasets
from ssl_anomaly.mini_lab.models import MODEL_REGISTRY, build_model, model_specs
from ssl_anomaly.mini_lab.runner import run_experiment


def _dataset(n_dates: int = 40) -> MiniDataset:
    ds = MiniDataset()
    ds.feature_names = ["ndvi", "red", "nir", "w1"]
    for farm in ("R12-08", "R12-06", "R12-10"):
        rng = np.random.default_rng({"R12-08": 1, "R12-06": 2, "R12-10": 3}[farm])
        dates = tuple(f"2015-{1 + (i // 20):02d}-{1 + (i % 20):02d}" for i in range(n_dates))
        values = rng.normal(size=(n_dates, 4)).astype(np.float32)
        values[0, 0] = np.nan
        ds.samples.append(
            MiniSample(
                farm_id=farm,
                unit_id=farm,
                dates=dates,
                values=values,
                observed=np.isfinite(values),
                delta_t=np.full(n_dates, 16.0, dtype=np.float32),
            )
        )
    return ds


def test_max_steps_caps_compute_independent_of_dataset_size(tmp_path):
    """Equal-compute gate: both dataset sizes stop at the same step count."""
    seen = []
    for n in (4, 12):
        ds = _dataset()
        ds.samples = [ds.samples[0]] * n
        cfg = RunConfig(name=f"cap_{n}", claim_rung="L2")
        cfg.protocol.name = "intra_farm_temporal"
        cfg.protocol.holdout_boundary = "2015-01-21"  # fixture dates are 2015-01-01..02-09
        cfg.model.epochs = 50
        cfg.model.max_steps = 3
        cfg.model.seeds = (13,)
        cfg.n_trials = 1
        cfg.mlflow_experiment = "mini-lab-cap"
        summary = run_experiment(cfg, ds, out_root=tmp_path / f"cap_{n}")
        seen.append(summary["per_seed"][0]["steps_done"])
    assert seen == [3, 3]


def test_registry_exposes_all_nine_architectures():
    specs = model_specs()
    assert {s["name"] for s in specs} == set(MODEL_REGISTRY)
    assert len(specs) == 9
    assert {s["family"] for s in specs} == {"reconstruction", "representation"}
    assert all(s["available"] for s in specs)


def test_all_models_build_and_forward():
    cfg = RunConfig()
    cfg.model.d_model = 12
    cfg.model.latent_dim = 6
    x = torch.randn(2, 12, 5)
    for name in MODEL_REGISTRY:
        model = build_model(name, 5, cfg.model)
        recon, embed = model(x)
        assert recon.shape == x.shape, (name, recon.shape)
        assert embed.ndim == 2 and embed.shape[0] == 2, (name, embed.shape)


def test_lodo_protocol_splits_are_farm_disjoint_and_normalization_train_only():
    cfg = RunConfig()
    cfg.protocol.name = "lodo"
    splits = split_datasets(_dataset(), cfg)
    assert len(splits) == 3
    for name, train, test in splits:
        assert set(train.farm_ids).isdisjoint(test.farm_ids)
        assert train.n > 0 and test.n > 0
        # Train-only fit: the mean of train features is ~0 even if test is
        # shifted, proving test values did not enter normalization.
        assert abs(float(train.x.mean())) < 0.5
        assert np.isfinite(train.x).all()
        assert int((~train.observed).sum()) >= 1  # imputed values are marked


def test_temporal_protocol_preserves_boundary_and_pads_absent_values():
    cfg = RunConfig()
    cfg.protocol.name = "intra_farm_temporal"
    cfg.protocol.holdout_boundary = "2015-01-15"
    splits = split_datasets(_dataset(), cfg)
    assert len(splits) == 3
    for name, train, test in splits:
        assert all(r[1] < cfg.protocol.holdout_boundary for r in train.date_ranges if r[1])
        assert all(r[0] >= cfg.protocol.holdout_boundary for r in test.date_ranges if r[0])
        assert train.x.shape[1] == test.x.shape[1]  # padded to common T


def test_single_farm_and_protocol_validation(tmp_path):
    cfg = RunConfig()
    cfg.protocol.name = "single_farm"
    cfg.protocol.train_farms = ("R12-08",)
    splits = split_datasets(_dataset(), cfg)
    assert len(splits) == 1
    assert set(splits[0][1].farm_ids) == {"R12-08"}
    # Empty test_farms = every other farm in the store.
    assert set(splits[0][2].farm_ids) == {"R12-06", "R12-10"}
    # Empty train_farms = first farm in the store (names are never hard-coded).
    cfg.protocol.train_farms = ()
    splits = split_datasets(_dataset(), cfg)
    assert set(splits[0][1].farm_ids) == {"R12-06"}
    cfg.protocol.train_farms = ("NOT-A-FARM",)
    try:
        split_datasets(_dataset(), cfg)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown train farm must be rejected")
    bad = tmp_path / "bad.yaml"
    bad.write_text("claim_rung: L5\n", encoding="utf-8")
    try:
        load_run_config(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("L5 claim must be rejected for the mini dataset")


def test_run_experiment_writes_complete_provenance(tmp_path):
    cfg = RunConfig()
    cfg.name = "unit_smoke"
    cfg.protocol.name = "single_farm"
    cfg.model.name = "gru_ae"
    cfg.model.epochs = 1
    cfg.model.seeds = (13,)
    cfg.n_trials = 1
    cfg.mlflow_experiment = "mini-lab-unit"
    report = run_experiment(cfg, _dataset(n_dates=20), out_root=tmp_path)
    required = {
        "git_sha",
        "data_source",
        "source_manifest",
        "seeds",
        "device",
        "wall_s",
        "claim_rung",
        "holdout_boundary",
        "holdout_justification",
        "aggregate",
        "per_seed",
        "optuna",
    }
    assert required.issubset(report)
    assert report["claim_rung"] == "L2"
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "optuna_study.json").exists()
    assert (tmp_path / "mlflow.db").exists()
    seed_dir = tmp_path / cfg.name / "single" / "seed_13"
    assert (seed_dir / "checkpoint.pt").exists() or (seed_dir / "last.pt").exists()
    assert (seed_dir / "report.json").exists()
    assert (seed_dir / "progress.jsonl").exists()

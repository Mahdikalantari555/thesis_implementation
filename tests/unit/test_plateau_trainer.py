"""Tasks 1.4/1.5 — plateau trainer retrofit and resume CLI semantics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ssl_anomaly.mini_lab.config import RunConfig, load_run_config
from ssl_anomaly.mini_lab.data import MiniDataset, MiniSample, split_datasets
from ssl_anomaly.mini_lab.runner import run_experiment


def _dataset(n_dates: int = 240, window: int = 12, stride: int = 4) -> MiniDataset:
    """Sliding-window fixture: ~58 windows/farm so dev slices have enough
    rows for the fixed-budget probes (they refuse n < 5-8 honestly)."""
    ds = MiniDataset()
    ds.feature_names = ["ndvi", "red", "nir", "w1"]
    for farm in ("R12-08", "R12-06", "R12-10"):
        rng = np.random.default_rng({"R12-08": 1, "R12-06": 2, "R12-10": 3}[farm])
        all_dates = [
            (np.datetime64("2015-01-01", "D") + np.timedelta64(i, "D")).astype(str)
            for i in range(n_dates)
        ]
        values = rng.normal(0.5, 0.1, size=(n_dates, 4)).astype(np.float32)
        observed = np.isfinite(values)
        values[0, 0] = np.nan
        observed[0, 0] = False
        delta = np.full(n_dates, 1.0, dtype=np.float32)
        wi = 0
        for s in range(0, n_dates - window + 1, stride):
            sel = slice(s, s + window)
            ds.samples.append(
                MiniSample(
                    farm_id=farm,
                    unit_id=f"{farm}:{wi}",
                    dates=tuple(all_dates[s : s + window]),
                    values=values[sel].copy(),
                    observed=observed[sel].copy(),
                    delta_t=delta[sel].copy(),
                )
            )
            wi += 1
    return ds


def _plateau_cfg(tmp_path: Path, **model_kw) -> RunConfig:
    cfg = RunConfig(name="plateau_smoke", claim_rung="L2")
    cfg.protocol.name = "intra_farm_temporal"
    cfg.protocol.dev_boundary = "2015-02-15"
    cfg.protocol.holdout_boundary = "2015-02-25"
    cfg.model.name = "mlp_ae"
    cfg.representation.temporal = "sliding"
    cfg.representation.window_length = 12
    cfg.representation.stride = 4
    cfg.model.epochs = 1
    cfg.model.seeds = (13,)
    cfg.model.d_model = 12
    cfg.model.latent_dim = 6
    cfg.model.to_plateau = True
    cfg.model.max_steps = 40
    cfg.model.eval_every = 10
    cfg.model.min_steps = 0
    cfg.model.patience = 2
    cfg.model.window = 3
    cfg.model.lr_patience = None
    cfg.model.max_lr_drops = 0
    cfg.n_trials = 1
    cfg.mlflow_experiment = "mini-lab-plateau"
    cfg.out_root = tmp_path / "out"
    for k, v in model_kw.items():
        setattr(cfg.model, k, v)
    return cfg


class TestPlateauTrainer:
    def test_to_plateau_run_produces_verdict_and_artifacts(self, tmp_path):
        cfg = _plateau_cfg(tmp_path)
        summary = run_experiment(cfg, _dataset(), out_root=cfg.out_root)
        rep = summary["per_seed"][0]
        # the report schema carries the trainer fields
        assert rep["schema"] == "mini-run/v2"
        assert rep["verdict"] in {
            "CONVERGED_PLATEAU", "BUDGET_EXHAUSTED_STILL_IMPROVING",
            "BUDGET_EXHAUSTED_FLAT", "DIVERGED", "COLLAPSED", "OBJECTIVE_CONFLICT",
        }
        assert rep["steps_done"] <= cfg.model.max_steps
        assert rep["scored_from"] in ("last",) or rep["scored_from"].startswith("best.pt")
        assert rep["eligibility_log"], "eligibility log must exist"
        assert rep["dev_split"], "dev split provenance recorded"
        assert rep["data_source"] and rep["git_sha"] is not None
        assert rep["split_sha256"]["train"] and rep["split_sha256"]["test"]
        # artifacts on disk: last.pt, progress.jsonl, report.json
        run_dir = Path(rep["progress"]).parent
        assert (run_dir / "last.pt").exists()
        assert (run_dir / "report.json").exists()
        recs = [json.loads(l) for l in (run_dir / "progress.jsonl").read_text().strip().splitlines()]
        evals = [r for r in recs if "val_total" in r and r.get("val_total") is not None]
        assert evals, "progress.jsonl must carry the eval records"
        for r in evals:
            assert "step" in r and "lr" in r and "effective_rank" in r

    def test_probe_metrics_in_progress(self, tmp_path):
        cfg = _plateau_cfg(tmp_path)
        run_experiment(cfg, _dataset(), out_root=cfg.out_root)
        run_dir = cfg.out_root / cfg.name / "temporal_R12-08" / "seed_13"
        recs = [json.loads(l) for l in (run_dir / "progress.jsonl").read_text().strip().splitlines()]
        probe_rows = [r for r in recs if "probe_phase_r2" in r]
        assert probe_rows, "probe metrics must appear in progress.jsonl"
        assert all("probe_phase_r2" in r and r["probe_phase_r2"] is not None for r in probe_rows)

    def test_intra_train_dev_slice_for_farm_level_protocol(self, tmp_path):
        cfg = _plateau_cfg(tmp_path)
        cfg.protocol.name = "single_farm"
        cfg.protocol.train_farms = ("R12-08",)
        cfg.protocol.test_farms = ("R12-06",)
        summary = run_experiment(cfg, _dataset(), out_root=cfg.out_root)
        rep = summary["per_seed"][0]
        assert "intra-train recent slice" in rep["dev_split"]
        assert rep["n_dev_units"] >= 2

    def test_dev_split_is_never_the_test_split(self, tmp_path):
        cfg = _plateau_cfg(tmp_path)
        ds = _dataset()
        splits = split_datasets(ds, cfg)
        for name, train, dev, test in splits:
            dev_last = [r[1] for r in dev.date_ranges if r[0]]
            test_first = [r[0] for r in test.date_ranges if r[0]]
            assert max(dev_last) < min(test_first)

    def test_to_plateau_requires_a_cap(self, tmp_path):
        cfg = _plateau_cfg(tmp_path, max_steps=None, max_wall_minutes=None)
        ds = _dataset()
        splits = split_datasets(ds, cfg)
        from ssl_anomaly.mini_lab.trainer import train_one_seed_to_plateau

        with pytest.raises(RuntimeError):
            train_one_seed_to_plateau(
                cfg, splits[0][0], splits[0][1], splits[0][2], splits[0][3],
                13, tmp_path, ds.feature_names, {},
            )


class TestResumeSemantics:
    def _seed_run(self, tmp_path: Path) -> RunConfig:
        cfg = _plateau_cfg(tmp_path)
        run_experiment(cfg, _dataset(), out_root=cfg.out_root)
        return cfg

    def test_resume_without_checkpoint_fails_closed(self, tmp_path):
        cfg = _plateau_cfg(tmp_path, resume_mode="resume")
        cfg.name = "no_such_run"  # different run dir → no last.pt
        ds = _dataset()
        splits = split_datasets(ds, cfg)
        from ssl_anomaly.mini_lab.trainer import train_one_seed_to_plateau

        with pytest.raises(RuntimeError, match="resume"):
            train_one_seed_to_plateau(
                cfg, splits[0][0], splits[0][1], splits[0][2], splits[0][3],
                13, cfg.out_root, ds.feature_names, {},
            )

    def test_from_scratch_refuses_existing_last_pt(self, tmp_path):
        cfg = self._seed_run(tmp_path)
        cfg.model.resume_mode = "from-scratch"
        ds = _dataset()
        splits = split_datasets(ds, cfg)
        from ssl_anomaly.mini_lab.trainer import train_one_seed_to_plateau

        with pytest.raises(RuntimeError, match="from-scratch"):
            train_one_seed_to_plateau(
                cfg, splits[0][0], splits[0][1], splits[0][2], splits[0][3],
                13, cfg.out_root, ds.feature_names, {},
            )

    def test_init_from_missing_checkpoint_fails_closed(self, tmp_path):
        cfg = _plateau_cfg(tmp_path, resume_mode="init-from",
                           init_from=str(tmp_path / "nope.pt"))
        ds = _dataset()
        splits = split_datasets(ds, cfg)
        from ssl_anomaly.mini_lab.trainer import train_one_seed_to_plateau

        with pytest.raises(RuntimeError, match="init-from"):
            train_one_seed_to_plateau(
                cfg, splits[0][0], splits[0][1], splits[0][2], splits[0][3],
                13, cfg.out_root, ds.feature_names, {},
            )

    def test_init_from_existing_checkpoint_starts_fresh_run(self, tmp_path):
        cfg = self._seed_run(tmp_path)
        last = cfg.out_root / cfg.name / "temporal_R12-08" / "seed_13" / "last.pt"
        assert last.exists()
        cfg2 = _plateau_cfg(tmp_path, resume_mode="init-from", init_from=str(last))
        cfg2.name = "fresh_from_init"
        summary = run_experiment(cfg2, _dataset(), out_root=cfg2.out_root)
        rep = summary["per_seed"][0]
        # new run dir, fresh monitor (step restarts from ~0)
        assert rep["steps_done"] <= cfg2.model.max_steps
        assert (cfg2.out_root / cfg2.name / "temporal_R12-08" / "seed_13" / "last.pt").exists()

    def test_yaml_rejects_dev_boundary_after_holdout(self, tmp_path):
        yml = tmp_path / "cfg.yaml"
        yml.write_text(
            "name: bad\n"
            "protocol:\n  name: intra_farm_temporal\n"
            "  holdout_boundary: '2024-01-01'\n  dev_boundary: '2025-01-01'\n"
        )
        with pytest.raises(ValueError, match="dev_boundary"):
            load_run_config(yml)

    def test_yaml_resume_mode_validation(self, tmp_path):
        yml = tmp_path / "cfg.yaml"
        yml.write_text("name: bad\nmodel:\n  resume_mode: sideways\n")
        with pytest.raises(ValueError, match="resume_mode"):
            load_run_config(yml)


class TestEqualStepCut:
    def test_equal_step_cut_scores_saved_checkpoints(self, tmp_path):
        cfg = _plateau_cfg(tmp_path, keep_eval_ckpts=True)
        run_experiment(cfg, _dataset(), out_root=cfg.out_root)
        run_dir = cfg.out_root / cfg.name / "temporal_R12-08" / "seed_13"
        ckpts = sorted((run_dir / "eval_ckpts").glob("step_*.pt"))
        assert ckpts, "per-eval checkpoints must be saved for the equal-step cut"
        steps = [int(p.stem.split("_")[1]) for p in ckpts]
        assert steps == sorted(steps) and len(steps) >= 2
        payload = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
        assert payload["kind"] == "eval" and "model" in payload

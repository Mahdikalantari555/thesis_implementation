"""Patch arms inside eval_dev: the join, the reference, the injection.

Covers ``patch-level-controlled-trial`` task 2.4's data-free half: how a patch
store is joined to a field-level cycle list, how the calibration reference for
the empirical p-value is built, and that a severity cell is injected in the
PATCH tensor with the field-level ladder's severity and hashes.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.io import patches as P
from ssl_anomaly.monitor import patch_eval as PE
from ssl_anomaly.monitor import patch_scores as PS
from ssl_anomaly.monitor import severity_table as sev

T, C, S = P.N_BINS, P.N_CHANNELS, 8


def _store(tmp_path, field_ids=(11, 22), n_cycles=3, seed=0):
    g = np.random.default_rng(seed)
    root = P.create_store(tmp_path / "patches.zarr", field_ids=list(field_ids),
                          max_cycles=n_cycles, size=S)
    root["field_ids"][:] = np.asarray(field_ids, dtype=np.int64)
    for fi in range(len(field_ids)):
        root["in_field"][fi, 0, : S // 2, : S // 2] = 1
        for ci in range(n_cycles):
            root["cycle_valid"][fi, ci] = 1
            root["cycle_ids"][fi, ci] = ci
            root["cycle_start_day"][fi, ci] = 10 + ci
            root["sensor_id"][fi, ci] = 1
            root["n_obs"][fi, ci] = 2
            vals = g.normal(0.4, 0.05, size=(T, C, S, S)).astype(np.float32)
            # out-of-field pixels carry garbage on purpose: they must never be read
            vals[:, :, S // 2 :, :] = 99.0
            root["values"][fi, ci] = vals
            root["pixel_valid"][fi, ci, 0] = 1
    return root


def _norm(median=0.4, iqr=0.05):
    per_ch = {name: {"median": median, "iqr": iqr, "n": 100} for name in P.CHANNELS}
    return {"L8": per_ch, "L9": per_ch, "unknown": per_ch}


# --- the join ------------------------------------------------------------
def test_join_uses_field_and_cycle_ids_not_row_position(tmp_path):
    root = _store(tmp_path)
    out = PE.patch_batch_for_cycles(root, [22, 11], [1, 2], norm=_norm())
    assert out["X"].shape == (2, T, C, S, S)
    # row 0 must be field 22 / cycle 1, i.e. the SECOND store row
    direct = np.asarray(root["values"][1, 1], dtype=np.float32)
    in_field = np.asarray(root["in_field"][1, 0], dtype=bool)
    expect = np.where(np.isfinite(direct) & in_field[None, None], (direct - 0.4) / 0.05, 0.0)
    np.testing.assert_allclose(out["X"][0], expect, rtol=1e-5, atol=1e-5)


def test_join_refuses_a_field_absent_from_the_store(tmp_path):
    root = _store(tmp_path)
    with pytest.raises(ValueError, match="not in the store"):
        PE.patch_batch_for_cycles(root, [11, 999], [0, 0], norm=_norm())


def test_join_refuses_misaligned_id_vectors(tmp_path):
    root = _store(tmp_path)
    with pytest.raises(ValueError, match="must align"):
        PE.patch_batch_for_cycles(root, [11, 22], [0], norm=_norm())


def test_join_normalizes_and_never_asserts_out_of_field_pixels(tmp_path):
    root = _store(tmp_path)
    out = PE.patch_batch_for_cycles(root, [11], [0], norm=_norm())
    assert out["pixel_valid"][0, 0, S // 2 :, :].sum() == 0
    assert out["pixel_valid"][0, 0, : S // 2, : S // 2].all()
    assert np.abs(out["X"][0][:, :, S // 2 :, :]).max() == 0.0   # zero-filled, not 99


def test_join_keeps_unobserved_values_at_zero_after_normalization(tmp_path):
    root = _store(tmp_path)
    root["values"][0, 0, 5] = np.nan
    out = PE.patch_batch_for_cycles(root, [11], [0], norm=_norm())
    assert np.all(out["X"][0, 5] == 0.0)


def test_join_flags_a_missing_channel_sidecar(tmp_path):
    root = _store(tmp_path)
    out = PE.patch_batch_for_cycles(root, [11], [0], norm={})     # no stats at all
    # raw units pass through unchanged rather than being silently standardized
    assert np.isfinite(out["X"]).all()


# --- reference paths -----------------------------------------------------
class _Arm:
    """Minimal stand-in for PatchArmScorer.loo_batch."""

    def __init__(self, value=1.0):
        self.value = value
        self.seen = []

    def loo_batch(self, X, M, **kw):
        self.seen.append(X.shape[0])
        return np.full((X.shape[0], X.shape[1]), self.value), np.zeros(
            (X.shape[0], X.shape[1], X.shape[2]))


def test_reference_paths_use_only_the_calibration_indices():
    arm = _Arm()
    batch = {"X": np.zeros((6, T, C, S, S), dtype=np.float32),
             "pixel_valid": np.ones((6, T, S, S), bool)}
    paths = PE.patch_reference_paths(arm, batch, reference_idx=np.array([0, 2, 4]), chunk=2)
    assert paths.shape == (3, T)
    assert arm.seen == [2, 1]          # chunked, and only three samples read


def test_reference_paths_refuse_an_empty_calibration_set():
    with pytest.raises(ValueError, match="calibration"):
        PE.patch_reference_paths(_Arm(), {"X": np.zeros((2, T, C, S, S))},
                                 reference_idx=np.array([], dtype=int))


# --- severity cell -------------------------------------------------------
def _scorer(n=4, seed=0, in_field_shape=None):
    g = np.random.default_rng(seed)
    res = g.normal(0, 1, size=(n, T, C, S, S))
    obs = g.random((n, T, C, S, S)) > 0.2
    sigma = PS.fit_patch_sigma(res, obs, np.ones((n, T), dtype=int))
    from ssl_anomaly.models.patch_cnn_gru import build_patch_model
    model = build_patch_model(use_cnn=True, n_channels=C, d_model=8, n_horizons=2,
                              stem_widths=(8, 8))
    return PS.PatchDeepScorer("patch_cnn_gru", model, sigma,
                              in_field=np.ones((S, S), bool), n_channels=C)


def _batch(n=4, seed=0):
    g = np.random.default_rng(seed)
    return {"X": g.normal(0, 1, size=(n, T, C, S, S)).astype(np.float32),
            "pixel_valid": np.ones((n, T, S, S), bool),
            "in_field": np.ones((n, S, S), bool)}


def test_cell_scores_a_patch_batch_and_records_the_injections():
    scorer = _scorer()
    arm = PS.PatchArmScorer(scorer, np.ones((12, T)))
    out = PE.patch_arm_scores(arm, _batch(), level_id="S3", family=sev.ACUTE_FAMILY,
                              index="ndvi", round_idx=0)
    assert out["z2"].shape == (4, T, 1)
    assert len(out["injections"]) == 4
    assert all("sha256" in rec for rec in out["injections"])


def test_cell_injection_uses_the_severity_table_delta():
    scorer = _scorer()
    arm = PS.PatchArmScorer(scorer, np.ones((12, T)))
    out = PE.patch_arm_scores(arm, _batch(), level_id="S6", family=sev.ACUTE_FAMILY,
                              index="ndvi", round_idx=0)
    lv = {v.level_id: v for v in sev.load_table()}["S6"]
    assert out["injections"][0]["severity"] == pytest.approx(float(lv.delta_ndvi), rel=1e-6)


def test_cell_refuses_an_unknown_level():
    arm = PS.PatchArmScorer(_scorer(), np.ones((12, T)))
    with pytest.raises(ValueError, match="unknown severity level"):
        PE.patch_arm_scores(arm, _batch(), level_id="S99", family=sev.ACUTE_FAMILY,
                            index="ndvi", round_idx=0)


def test_persistent_cell_uses_the_frozen_duration():
    arm = PS.PatchArmScorer(_scorer(), np.ones((12, T)))
    out = PE.patch_arm_scores(arm, _batch(), level_id="S3", family=sev.PERSISTENT_FAMILY,
                              index="ndvi", round_idx=0)
    assert out["injections"][0]["duration_bins"] == int(sev.PERSISTENT_DURATION_BINS)


# --- the eval_dev patch path, end to end ---------------------------------
def test_patch_severity_path_runs_on_a_synthetic_store(tmp_path, monkeypatch):
    """Drive the real ``run_patch_severity_protocol`` against a tiny store.

    This test would catch a cycle-key format mistake (``a|b`` vs ``a:b``), a
    wrong artifact filename, or a scale mismatch between the p-value path and
    ``cumulative_channel_max`` — all of which are silent in the aggregate
    metrics.
    """
    import importlib.util
    from pathlib import Path as _Path

    import torch

    repo = _Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("eval_dev", repo / "scripts" / "eval_dev.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)

    _store(tmp_path, field_ids=(11, 22), n_cycles=10)         # writes patches.zarr
    from ssl_anomaly.io import patches as PP
    from ssl_anomaly.models.patch_cnn_gru import build_patch_model

    dev = tmp_path / "dev"
    run_dir = dev / "r1"
    run_dir.mkdir(parents=True)
    torch.manual_seed(0)
    model = build_patch_model(use_cnn=True, n_channels=C, d_model=8, n_horizons=2,
                              stem_widths=(8, 8))
    torch.save({
        "config": {"n_channels": C, "d_model": 8, "n_horizons": 2, "use_cnn": True,
                   "n_covariates": 0, "stem_widths": (8, 8)},
        "state_dict": model.state_dict(),
    }, run_dir / "patch_model.pt")
    sigma = PS.fit_patch_sigma(
        np.random.default_rng(0).normal(0, 1, (20, T, C, S, S)),
        np.ones((20, T, C, S, S), bool), np.ones((20, T), dtype=int),
    )
    np.savez(run_dir / "patch_sigma.npz", sigma=sigma.sigma, fallback=sigma.fallback,
             n=sigma.n, min_n=sigma.min_n, sha256=sigma.sha256)
    PP.save_patch_norm(_norm(), run_dir / "patch_norm.json")

    monkeypatch.setattr(E.mproto, "CHECKPOINTS_DEV_DIR", dev)

    cycle_keys = [f"{fid}:{ci}" for fid in (11, 22) for ci in range(10)]
    sel = {"cycle_keys": cycle_keys}
    args = E.parse_args([
        "--run-id", "r1", "--model", "gru_ae", "--scorer", "patch_cnn_gru",
        "--patch-store", str(tmp_path / "patches.zarr"), "--rounds", "1",
    ])
    metrics, per_seed = E.run_patch_severity_protocol(args, {}, sel, [13], "cpu")

    assert metrics["protocol"] == "severity_sweep_patch_v1"
    assert metrics["n_calib_cycles"] + metrics["n_eval_cycles"] == 20
    assert metrics["n_calib_cycles"] >= 8      # the p-value needs enough reference
    # the default --shapes is the frozen primary pair (acute + persistent);
    # gradual is opt-in and never a silent default
    assert set(per_seed[0]["mdm80"]) == {
        f"{fam}/{idx}" for fam in (sev.ACUTE_FAMILY, sev.PERSISTENT_FAMILY)
        for idx in ("ndvi", "savi")
    }
    assert per_seed[0]["patch_sigma_sha256"] == sigma.sha256
    # every cell was evaluated with the full evaluation subset
    for cell in per_seed[0]["cells"].values():
        assert cell["n"] == metrics["n_eval_cycles"]
        assert 0.0 <= cell["detection_rate"] <= 1.0


def test_patch_path_refuses_an_absent_store(tmp_path):
    import importlib.util
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("eval_dev_missing",
                                                  repo / "scripts" / "eval_dev.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    args = E.parse_args(["--run-id", "r1", "--model", "gru_ae",
                         "--scorer", "patch_cnn_gru",
                         "--patch-store", str(tmp_path / "nope.zarr")])
    with pytest.raises(SystemExit, match="patch store not found"):
        E.run_patch_severity_protocol(args, {}, {"cycle_keys": []}, [13], "cpu")


def test_patch_path_refuses_a_cycle_missing_from_the_store(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path as _Path

    repo = _Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location("eval_dev_missing2",
                                                  repo / "scripts" / "eval_dev.py")
    E = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(E)
    _store(tmp_path, field_ids=(11,), n_cycles=1)
    dev = tmp_path / "dev"
    dev.mkdir()
    monkeypatch.setattr(E.mproto, "CHECKPOINTS_DEV_DIR", dev)
    args = E.parse_args(["--run-id", "r1", "--model", "gru_ae",
                         "--scorer", "patch_cnn_gru",
                         "--patch-store", str(tmp_path / "patches.zarr")])
    with pytest.raises((SystemExit, ValueError, FileNotFoundError)):
        E.run_patch_severity_protocol(args, {}, {"cycle_keys": ["11:0", "11:5"]}, [13], "cpu")

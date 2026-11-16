"""End-to-end plumbing test for the patch trainer (task 1.4 / gate G1).

Builds a synthetic ``patches.zarr`` + frozen norm + holdout manifest in a temp
directory, then drives ``scripts/train_patch.py`` through ``--estimate-runtime``,
``--smoke``, ``--resume`` and a config-hash mismatch. No data archive, no GPU,
no long run — this is the "G1 smoke plumbing" gate in test form.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ssl_anomaly.io import patches as P
from ssl_anomaly.monitor import holdout as H

REPO = Path(__file__).resolve().parents[2]
S, T = P.DEFAULT_PATCH_SIZE, P.N_BINS
FIELDS = (101, 102, 103)


def _load_train_patch():
    spec = importlib.util.spec_from_file_location("train_patch", REPO / "scripts" / "train_patch.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_patch"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_store(tmp_path: Path, cycles: int = 3) -> Path:
    g = np.random.default_rng(0)
    store = tmp_path / "patches.zarr"
    root = P.create_store(store, field_ids=list(FIELDS), max_cycles=cycles)
    per_ch = {ch: {"median": 0.4, "iqr": 0.1, "n": 500} for ch in P.CHANNELS}
    norm = {name: dict(per_ch) for name in ("L8", "L9", "unknown")}
    for i, f in enumerate(FIELDS):
        root["field_ids"][i] = f
        infield = np.zeros((S, S), dtype=np.uint8)
        infield[6:26, 6:26] = 1
        root["in_field"][i, 0] = infield
        for c in range(cycles):
            vals = np.abs(g.normal(0.4, 0.1, size=(T, P.N_CHANNELS, S, S))).astype(np.float16)
            vals[..., infield == 0] = np.nan
            root["values"][i, c] = vals
            root["pixel_valid"][i, c, :, 0] = np.isfinite(vals).any(axis=1)
            root["n_obs"][i, c] = g.integers(1, 4, size=T)
            root["sensor_id"][i, c] = P.SENSOR_IDS["L8"]
            root["cycle_ids"][i, c] = 1000 + c
            root["cycle_start_day"][i, c] = 40 + 20 * c
            root["cycle_valid"][i, c] = 1
    P.save_patch_norm(norm, tmp_path / P.NORM_STATS_NAME)
    return store


def _holdout(tmp_path: Path) -> Path:
    manifest = {
        "manifest_version": H.MANIFEST_VERSION,
        "holdout_farms": [103],
        "known_farm_ids": list(FIELDS),
        "holdout_date_start": None,
        "archive_end_date": "2022-12-31",
        "created": "2026-11-16",
        "git_sha": "test",
    }
    manifest["manifest_sha256"] = H.holdout_manifest_hash(manifest)
    path = tmp_path / "holdout.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def _args(store: Path, holdout: Path, out: Path, *extra: str) -> list[str]:
    return [
        "--store", str(store), "--holdout-manifest", str(holdout),
        "--out", str(out), "--epochs", "2", "--batch-size", "2",
        "--d-model", "8", "--n-horizons", "1", *extra,
    ]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("patch_train")
    store = _make_store(tmp)
    holdout = _holdout(tmp)
    return tmp, store, holdout


def test_estimate_runtime_does_not_train(env, capsys):
    tmp, store, holdout = env
    tp = _load_train_patch()
    rc = tp.main(_args(store, holdout, tmp / "out", "--estimate-runtime"))
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out[out.index("{"): out.rindex("}") + 1])
    assert payload["n_samples"] > 0
    assert payload["seconds_per_epoch"] > 0
    assert payload["total_seconds"] >= payload["seconds_per_epoch"]
    assert "nothing was trained" in out
    assert not (tmp / "out").exists() or not any((tmp / "out").rglob("last.pt"))


def test_smoke_run_writes_checkpoint_progress_and_manifest(env):
    tmp, store, holdout = env
    tp = _load_train_patch()
    out = tmp / "smoke_out"
    rc = tp.main(_args(store, holdout, out, "--smoke", "--seed", "13"))
    assert rc == 0

    run_dir = out / "patch_cnn_gru_seed13" / "_smoke"
    assert (run_dir / "last.pt").exists()
    assert (run_dir / "progress.jsonl").exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    for key in ("git_sha", "torch", "device", "data_source", "holdout_sha256",
                "norm_sha256", "n_field_cycles", "config_hash", "wall_time_s"):
        assert key in manifest, key
    # the holdout field is excluded from the training set
    assert manifest["n_trainable_fields"] == 2
    assert manifest["n_excluded_by_holdout"] == 1
    # progress rows are complete enough to resume from
    rows = [json.loads(line) for line in (run_dir / "progress.jsonl").read_text().splitlines()]
    assert rows
    assert {"epoch", "train_loss", "lr", "trim_frac"} <= set(rows[0])


def test_resume_restores_state(env):
    tmp, store, holdout = env
    tp = _load_train_patch()
    out = tmp / "resume_out"
    assert tp.main(_args(store, holdout, out, "--smoke", "--seed", "42")) == 0
    last = out / "patch_cnn_gru_seed42" / "_smoke" / "last.pt"
    assert last.exists()
    rc = tp.main(_args(store, holdout, out, "--resume", "--seed", "42"))
    assert rc == 0


def test_config_hash_mismatch_refuses_to_resume(env):
    tmp, store, holdout = env
    tp = _load_train_patch()
    out = tmp / "mismatch_out"
    assert tp.main(_args(store, holdout, out, "--smoke", "--seed", "7")) == 0
    last = out / "patch_cnn_gru_seed7" / "_smoke" / "last.pt"

    cfg = tp.cfg_from_args(tp.build_parser().parse_args(
        _args(store, holdout, out, "--smoke", "--seed", "7")
    ))
    cfg["lr"] = 0.5                      # a different experiment
    import torch
    model = torch.nn.Linear(1, 1)        # unused: the hash check fires first
    with pytest.raises(RuntimeError, match="config hash mismatch"):
        tp.load_checkpoint(last, model=model, cfg=cfg)


def test_missing_store_is_a_loud_error_with_the_next_step(env, tmp_path):
    _tmp, _store, holdout = env
    tp = _load_train_patch()
    with pytest.raises(SystemExit, match=r"extract_patches\.py"):
        tp.main(["--store", str(tmp_path / "nope.zarr"),
                 "--holdout-manifest", str(holdout)])


def test_holdout_is_required(env):
    tmp, store, _holdout = env
    tp = _load_train_patch()
    # no --holdout-manifest and the default frozen manifest does not exist here
    with pytest.raises(RuntimeError, match="holdout manifest missing"):
        tp.main(["--store", str(store), "--out", str(tmp / "x"), "--smoke"])


def test_trim_schedule_only_activates_after_warmup():
    tp = _load_train_patch()
    cfg = {"trim_top_frac": 0.05, "trim_after_epoch": 10}
    assert tp.trim_schedule(0, cfg) == 0.0
    assert tp.trim_schedule(9, cfg) == 0.0
    assert tp.trim_schedule(10, cfg) == 0.05


def test_burn_in_masks_the_first_bins():
    tp = _load_train_patch()
    obs = np.ones((1, 6, 2), dtype=bool)
    out = tp.mask_burn_in(obs, 2)
    assert not out[:, :2].any()
    assert out[:, 2:].all()
    np.testing.assert_array_equal(tp.mask_burn_in(obs, 0), obs)


def test_config_hash_covers_the_contamination_axis(env):
    """patch-contamination-absorption task 1.2: resume safety must see it."""
    tp = _load_train_patch()
    cfg = {"arch": "patch_cnn_gru", "seed": 13, "lr": 1e-3}
    a = tp.config_hash(cfg)
    assert a == tp.config_hash(dict(cfg))
    for key, val in (("seed", 42), ("lr", 1e-2), ("arch", "patch_pixelwise"),
                     ("aux_weight", 0.9), ("trim_top_frac", 0.2)):
        assert tp.config_hash({**cfg, key: val}) != a, key


# ---------------------------------------------------------------------------
# Contamination axis (patch-contamination-absorption tasks 1.1/1.2)
# ---------------------------------------------------------------------------
def test_contaminated_run_records_the_plan_and_trains(env):
    tmp, store, holdout = env
    tp = _load_train_patch()
    out = tmp / "contam_out"
    rc = tp.main(_args(store, holdout, out, "--smoke", "--seed", "13",
                       "--contaminate-fraction", "0.5", "--contaminate-seed", "5"))
    assert rc == 0
    manifest = json.loads(
        (out / "patch_cnn_gru_seed13" / "_smoke" / "run_manifest.json").read_text()
    )
    cont = manifest["contamination"]
    assert cont["fraction"] == 0.5
    assert cont["seed"] == 5
    assert cont["level_id"] == "S4"
    assert cont["n_selected"] == 3             # half of the 6 train cycles
    assert len(cont["selected_keys"]) == 3
    assert cont["onsets"]
    assert manifest["curation"]["enabled"] is False


def test_contamination_never_selects_the_holdout_field(env):
    tmp, store, holdout = env
    tp = _load_train_patch()
    out = tmp / "contam_holdout"
    assert tp.main(_args(store, holdout, out, "--smoke", "--seed", "3",
                         "--contaminate-fraction", "1.0")) == 0
    manifest = json.loads(
        (out / "patch_cnn_gru_seed3" / "_smoke" / "run_manifest.json").read_text()
    )
    fields = {int(k.split(":")[0]) for k in manifest["contamination"]["selected_keys"]}
    assert 103 not in fields                    # the holdout farm


def test_contamination_axes_are_in_the_config_hash(env):
    tmp, store, holdout = env
    tp = _load_train_patch()
    base = ["--store", str(store), "--holdout-manifest", str(holdout)]
    p1 = tp.build_parser().parse_args(base + ["--contaminate-fraction", "0.05"])
    p2 = tp.build_parser().parse_args(base + ["--contaminate-fraction", "0.10"])
    p3 = tp.build_parser().parse_args(base + ["--contaminate-fraction", "0.05",
                                              "--trimmed-loss", "off"])
    p4 = tp.build_parser().parse_args(base + ["--contaminate-fraction", "0.05",
                                              "--curation", "on"])
    hashes = {tp.config_hash(tp.cfg_from_args(x)) for x in (p1, p2, p3, p4)}
    assert len(hashes) == 4


def test_clean_default_is_unchanged(env):
    tmp, store, holdout = env
    tp = _load_train_patch()
    args = tp.build_parser().parse_args(
        ["--store", str(store), "--holdout-manifest", str(holdout)]
    )
    cfg = tp.cfg_from_args(args)
    assert cfg["contaminate_fraction"] == 0.0
    assert cfg["trimmed_loss"] is True
    assert cfg["curation"] is False

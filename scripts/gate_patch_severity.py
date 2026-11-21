"""Fail-closed synthetic severity A/B gate, including baseline CLI metrics.

Example: PYTHONPATH=src:. python scripts/gate_patch_severity.py --chunk 2 --device cpu
Uses temporary stores and smoke checkpoints only. CUDA results apply only to the
fixture, device and installed software tested, not all production checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import types
from unittest.mock import patch

import numpy as np
import torch

from ssl_anomaly.io import patches as P
from ssl_anomaly.monitor import holdout as H
from ssl_anomaly.monitor import patch_eval as PE
from ssl_anomaly.monitor import patch_scores as PS
from ssl_anomaly.train import patch_data as PD
from tests.patch_fixtures import bind_patch_store
from tests.unit import test_patch_eval as fixture

ROOT = Path(__file__).resolve().parents[1]
BASE = "36a5fcb236258c1122fb3ba86a47d2129a7b4809"


def load_source(name, filename, data=None):
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / filename)
    source = (ROOT / filename).read_text() if data is None else data
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def old_source(filename):
    return subprocess.check_output(["git", "show", f"{BASE}:{filename}"], cwd=ROOT, text=True)


def hash_value(value):
    if isinstance(value, np.ndarray):
        # Include dtype and shape: equal bytes alone do not identify an array.
        value = str((value.dtype.str, value.shape)).encode() + value.tobytes()
    else:
        value = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.chunk < 1 or args.size < 4:
        parser.error("chunk must be positive and size >= 4")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(1)
    trainer = load_source("gate_train_patch", "scripts/train_patch.py")
    original_eval = load_source("gate_baseline_eval", "scripts/eval_dev.py", old_source("scripts/eval_dev.py"))
    optimized_eval = load_source("gate_new_eval", "scripts/eval_dev.py")
    original_patch = load_source("gate_baseline_patch", "src/ssl_anomaly/monitor/patch_eval.py",
                                 old_source("src/ssl_anomaly/monitor/patch_eval.py"))
    original_functions = {name: getattr(PE, name) for name in
                          ("patch_batch_for_cycles", "patch_reference_paths", "patch_arm_scores")}
    fixture.S = args.size
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        root = fixture._store(tmp, field_ids=(33, 11, 22), n_cycles=10, seed=7)
        cycle_keys = [f"{fid}:{ci}" for fid in (11, 22) for ci in range(10)]
        splits, holdout = bind_patch_store(root, known=(11, 22, 33), dev_keys=cycle_keys,
                                           parquet_path=tmp / "cycles.parquet")
        lock = tmp / "splits.json"
        lock.write_text(json.dumps(splits))
        holdout_path = H.save_holdout_manifest(holdout, path=tmp / "holdout.json")
        norm = P.build_patch_norm(root=root, splits_manifest=splits, holdout_manifest=holdout)
        norm_path = P.save_patch_norm(norm, tmp / P.NORM_STATS_NAME)
        # Synthetic scales only; no data or artifacts from the real study are used.
        sigma = PS.fit_patch_sigma(np.random.default_rng(0).normal(0, 1, (20, P.N_BINS, P.N_CHANNELS, 8, 8)),
                                   np.ones((20, P.N_BINS, P.N_CHANNELS, 8, 8), bool),
                                   np.ones((20, P.N_BINS), dtype=int))
        for arch in ("patch_pixelwise", "patch_cnn_gru"):
            train_args = ["--arch", arch, "--splits", str(lock), "--holdout-manifest", str(holdout_path),
                          "--store", str(tmp / "patches.zarr"), "--norm", str(norm_path),
                          "--out", str(tmp), "--seed", "13", "--smoke", "--epochs", "1",
                          "--d-model", "8", "--n-horizons", "1", "--batch-size", "1",
                          "--max-fields", "1", "--max-cycles", "1", "--device", "cpu", "--no-benchmark"]
            if trainer.main(train_args) != 0:
                raise RuntimeError("synthetic training failed")
            ckpt = tmp / f"{arch}_seed13" / "_smoke" / "last.pt"
            sigma_path = tmp / f"{arch}_sigma.npz"
            np.savez(sigma_path, sigma=sigma.sigma, fallback=sigma.fallback, n=sigma.n,
                     min_n=sigma.min_n, sha256=sigma.sha256,
                     model_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                     norm_sha256=P.norm_stats_sha256(norm), input_protocol=PD.PATCH_INPUT_PROTOCOL)
            argv = ["--run-id", "gate", "--model", arch, "--scorer", arch,
                    "--patch-store", str(tmp / "patches.zarr"), "--patch-checkpoint", str(ckpt),
                    "--patch-sigma", str(sigma_path), "--splits", str(lock),
                    "--holdout-manifest", str(holdout_path), "--rounds", "2", "--seeds", "13",
                    "--device", args.device, "--smoke"]
            snapshots, metrics = {}, {}
            for label, module, functions in (("baseline", original_eval, original_patch),
                                              ("chunked", optimized_eval, None)):
                calls = []
                selected = {name: getattr(functions, name) if functions else original_functions[name]
                            for name in original_functions}
                score = selected["patch_arm_scores"]
                reference = selected["patch_reference_paths"]
                references = []
                def record_reference(*a, _reference=reference, **kw):
                    value = _reference(*a, **kw)
                    references.append(hash_value(value))
                    return value
                def record(*a, _score=score, **kw):
                    output = _score(*a, **kw)
                    calls.append({"cell": [kw["level_id"], kw["family"], kw["index"], kw["round_idx"]],
                                  **{key: hash_value(output[key]) for key in
                                     ("z2", "paths", "p_value", "observed", "injections")}})
                    return output
                outdir = tmp / f"reports_{label}"
                with patch.object(PE, "patch_batch_for_cycles", selected["patch_batch_for_cycles"]), \
                     patch.object(PE, "patch_reference_paths", record_reference), \
                     patch.object(PE, "patch_arm_scores", record), \
                     patch.object(module.mproto, "DEV_DIR", outdir):
                    rc = module.main(argv + (["--score-chunk", str(args.chunk)] if label == "chunked" else []))
                if rc != 0:
                    raise RuntimeError(f"{label} CLI failed with {rc}")
                file = outdir / "gate" / f"metrics_{arch}_sev_{arch}_smoke_s13.json"
                obj = json.loads(file.read_text())
                # Wall time is inherently observational and must not be compared.
                obj.pop("wall_time_s")
                metrics[label] = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                snapshots[label] = calls
                if len(references) != 1:
                    raise AssertionError("expected exactly one reference path calculation")
                snapshots[label + "_reference"] = references[0]
            if snapshots["baseline_reference"] != snapshots["chunked_reference"]:
                raise AssertionError(f"{arch} calibration reference paths mismatch")
            if len(snapshots["baseline"]) != 48 or len(snapshots["chunked"]) != 48:
                raise AssertionError("expected 24 cells times two rounds")
            for old, new in zip(snapshots["baseline"], snapshots["chunked"], strict=True):
                if old != new:
                    raise AssertionError(f"{arch} cell/array mismatch: {old} != {new}")
            if metrics["baseline"] != metrics["chunked"]:
                raise AssertionError(f"{arch} metrics mismatch (excluding wall_time_s)")
            print(f"PASS {arch}: 48 cell rounds, reference, all arrays/injections and metrics identical (excluding wall_time_s)", flush=True)
    print(f"PASS synthetic gate: device={args.device}, size={args.size}, chunk={args.chunk}")


if __name__ == "__main__":
    main()

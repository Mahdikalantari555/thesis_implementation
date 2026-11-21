"""CPU synthetic A/B profile against the frozen baseline; never touches real data.

Run: OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src:. python scripts/profile_patch_severity_synthetic.py
The fixture is deliberately small; its timings and RSS must not be extrapolated
as measurements of the 340-cycle production panel or a T4.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time

import numpy as np
import psutil
import torch

from ssl_anomaly.io import patches as P
from ssl_anomaly.models.patch_cnn_gru import build_patch_model
from ssl_anomaly.monitor import patch_eval as current
from ssl_anomaly.monitor import patch_scores as PS
from ssl_anomaly.monitor import severity_table as sev
from tests.unit import test_patch_eval as fixture

BASE = "36a5fcb236258c1122fb3ba86a47d2129a7b4809"


def digest(value):
    if isinstance(value, np.ndarray):
        return hashlib.sha256(value.tobytes()).hexdigest()
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def phase(name, work, results):
    proc = psutil.Process()
    initial = proc.memory_info().rss
    peak = [initial]
    stop = threading.Event()
    def sample():
        while not stop.wait(0.002):
            peak[0] = max(peak[0], proc.memory_info().rss)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        value = work()
    finally:
        elapsed = time.perf_counter() - start
        peak[0] = max(peak[0], proc.memory_info().rss)
        stop.set()
        thread.join()
    results[name] = {"wall_s": elapsed, "start_rss_mib": initial / 2**20,
                     "peak_rss_mib": peak[0] / 2**20}
    return value


def run_worker(args):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    if args.mode == "baseline":
        spec = importlib.util.spec_from_file_location("baseline_patch_eval", args.baseline_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = current
    root = P.open_store(args.store)
    ids = np.repeat([11, 22], args.cycles // 2)
    cids = np.tile(np.arange(args.cycles // 2), 2)
    results = {}
    batch = phase("batch_build", lambda: module.patch_batch_for_cycles(root, ids, cids, norm=fixture._norm()), results)
    ch = P.N_CHANNELS
    sigma = PS.PatchSigmaTable(np.ones((18, ch, 3)), np.ones(ch), np.full((18, ch, 3), 100))
    model = build_patch_model(use_cnn=args.arch == "patch_cnn_gru", n_channels=ch,
                              d_model=8, n_horizons=1, stem_widths=(8, 8),
                              n_covariates=batch["covariates"].shape[-1])
    scorer = PS.PatchDeepScorer(args.arch, model, sigma, in_field=batch["in_field"],
                                n_channels=ch, device="cpu")
    cal_idx = np.arange(args.cycles // 2)
    eval_idx = np.arange(args.cycles // 2, args.cycles)
    ref = phase("calibration", lambda: module.patch_reference_paths(scorer, batch,
                reference_idx=cal_idx, field_row=np.arange(args.cycles)), results)
    arm = PS.PatchArmScorer(scorer, ref)
    def run_cell():
        nonlocal batch
        if args.mode == "optimized":
            del batch["X"]
        eval_batch = {k: v[eval_idx] if isinstance(v, np.ndarray) else v for k, v in batch.items()}
        if args.mode == "optimized":
            batch = None
        opts = {"chunk": args.chunk} if args.mode == "optimized" else {}
        cells = [(lv.level_id, family, index, rnd)
                 for lv in sev.load_table() for family in ("acute", "persistent")
                 for index in ("ndvi", "savi") for rnd in range(2)] if args.sweep else [
                     ("S3", "persistent", "savi", 1)]
        fingerprints = []
        for level, family, index, rnd in cells:
            cell = module.patch_arm_scores(arm, eval_batch, level_id=level, family=family,
                   index=index, round_idx=rnd, field_row=eval_idx,
                   injection_basis="bands_consistent", **opts)
            fingerprints.append({k: digest(cell[k]) for k in
                                 ("z2", "paths", "p_value", "observed", "injections")})
        return fingerprints
    fingerprints = phase("cell_loop", run_cell, results)
    results["fingerprints"] = {"cells": fingerprints, "reference": digest(ref)}
    print(json.dumps(results, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=20, help="even number, at least 16")
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--sweep", action="store_true", help="measure all 24 cells x 2 rounds")
    parser.add_argument("--arch", choices=("patch_pixelwise", "patch_cnn_gru"), default="patch_pixelwise")
    parser.add_argument("--mode", choices=("baseline", "optimized"))
    parser.add_argument("--baseline-file")
    parser.add_argument("--store")
    args = parser.parse_args()
    if args.cycles < 16 or args.cycles % 2 or args.chunk <= 0:
        parser.error("cycles must be even and >=16; chunk must be positive")
    if args.mode:
        run_worker(args)
        return
    with TemporaryDirectory() as folder:
        tmp = Path(folder)
        fixture.S = 48
        fixture._store(tmp, field_ids=(11, 22), n_cycles=args.cycles // 2, seed=7)
        baseline = tmp / "baseline_patch_eval.py"
        baseline.write_bytes(subprocess.check_output(["git", "show", f"{BASE}:src/ssl_anomaly/monitor/patch_eval.py"]))
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        reports = {}
        for mode in ("baseline", "optimized"):
            cmd = [sys.executable, __file__, "--mode", mode, "--baseline-file", str(baseline),
                   "--store", str(tmp / "patches.zarr"), "--cycles", str(args.cycles),
                   "--chunk", str(args.chunk), "--arch", args.arch]
            if args.sweep:
                cmd.append("--sweep")
            reports[mode] = json.loads(subprocess.check_output(cmd, env=env, text=True).splitlines()[-1])
        parity = reports["baseline"]["fingerprints"] == reports["optimized"]["fingerprints"]
        print(json.dumps({"config": {"cycles": args.cycles, "calibration": args.cycles // 2,
                              "evaluation": args.cycles // 2, "T": 36, "C": P.N_CHANNELS,
                              "size": 48, "chunk": args.chunk, "arch": args.arch,
                              "device": "cpu", "threads": 1, "cells": 24 if args.sweep else 1,
                              "rounds": 2 if args.sweep else 1,
                              "baseline_commit": BASE},
                          "parity": parity, "results": reports}, indent=2))
        if not parity:
            raise SystemExit("FAIL: baseline and optimized fingerprints differ")


if __name__ == "__main__":
    main()

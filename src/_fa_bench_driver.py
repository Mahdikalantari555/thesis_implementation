"""Synthetic, deterministic workload for fast-anything profiling.

Covers the three repeated research workflows plus supporting pure-compute parts:
  extract  - per-bin patch aggregation (the hot reduce in extract_patches.py)
  eval     - patch severity arm scoring + empirical p-values + affiliation metrics
  train    - sample normalize/collate + one model forward/backward step
  smoothing / baseline / detectors / affiliation / indices - pipeline support

Run with the geospatial conda env:
  /home/asus/miniforge3/envs/geospatial/bin/python src/_fa_bench_driver.py --part all --loops 5
"""
from __future__ import annotations

import os
import sys

# Set before numpy/torch import: thread pools read these at import time.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
sys.path.insert(0, ROOT + "/scripts")

import argparse
import types
from time import perf_counter

# lightning is not installed in this env; the benchmark never uses it, but
# importing ssl_anomaly.train pulls it in. Stub only the symbol it needs.
if "lightning" not in sys.modules:
    try:
        import lightning  # noqa: F401
    except ModuleNotFoundError:
        _stub = types.ModuleType("lightning")
        _stub.LightningModule = type("LightningModule", (), {})
        sys.modules["lightning"] = _stub

import numpy as np

from ssl_anomaly._vendor.affiliation.metrics import pr_from_events
from ssl_anomaly.backends.numpy_backend import NumpyBackend
from ssl_anomaly.baseline.score import cusum_detector, ewma_detector, flag_runs
from ssl_anomaly.io import patches as P
from ssl_anomaly.phenology.indices import evi, ndvi, ndwi
from ssl_anomaly.train import patch_data as PD
from ssl_anomaly.utils.arrays import contiguous_runs

PARTS = ("extract", "eval", "train", "detectors", "affiliation", "smoothing", "baseline", "indices")


def _rng():
    return np.random.default_rng(20261123)


# --------------------------------------------------------------------------
# 1. PATCH EXTRACTION
# --------------------------------------------------------------------------
def run_extract(loops: int):
    """Per-bin median reduce over acquisitions: the core of extract_patches.py."""
    import extract_patches as E

    r = _rng()
    C, S = P.N_CHANNELS, 32
    n_bins = int(600 * loops)
    scenes = [
        r.uniform(0.05, 0.6, size=(C, S, S)).astype(np.float32) for _ in range(2)
    ]
    masks = [r.random((S, S)) > 0.2 for _ in range(2)]
    in_field = np.zeros((S, S), dtype=bool)
    in_field[2:-2, 2:-2] = True
    for sc in scenes:
        sc[r.random((C, S, S)) < 0.1] = np.nan

    t0 = perf_counter()
    for i in range(n_bins):
        if i % 3 == 0:
            E.short_bin(scenes, masks, in_field=in_field, dtype="float16")
        else:
            P.aggregate_bin(scenes, masks, in_field=in_field, dtype="float16")
    return perf_counter() - t0


# --------------------------------------------------------------------------
# 2. EVAL
# --------------------------------------------------------------------------
def _make_eval_batch(n=64, T=None, C=None, S=32):
    r = _rng()
    T = T or P.N_BINS
    C = C or P.N_CHANNELS
    raw = r.uniform(0.05, 0.6, size=(n, T, C, S, S)).astype(np.float32)
    obs = r.random((n, T, C, S, S)) > 0.15
    in_field = np.zeros((n, S, S), dtype=bool)
    in_field[:, 2:-2, 2:-2] = True
    per_ch = {name: {"median": 0.4, "iqr": 0.1, "n": 100} for name in P.CHANNELS}
    norm = {"stats": {s: per_ch for s in ("L8", "L9", "unknown")},
            "channel_order": list(P.CHANNELS)}
    stats = P.patch_norm_stats(norm)
    sensors = np.ones((n, T), dtype=np.uint8)
    X = np.stack([P.normalize_patch(raw[i], obs[i], sensors[i], stats)[0]
                  for i in range(n)])
    return {
        "raw": raw, "obs": obs, "in_field": in_field,
        "pixel_valid": obs.any(axis=2), "norm_stats": stats,
        "sensor_id": sensors, "X": X,
        "covariates": r.normal(size=(n, T, PD.COV_DIM if hasattr(PD, "COV_DIM") else 6)).astype(np.float32),
    }


def run_eval(loops: int):
    """Patch severity arm scoring (injection + model + empirical p-value)."""
    import torch

    from ssl_anomaly.models.patch_cnn_gru import build_patch_model
    from ssl_anomaly.monitor import patch_eval as PE
    from ssl_anomaly.monitor import patch_scores as PS

    torch.set_num_threads(1)
    batch = _make_eval_batch(n=64)
    ch = P.N_CHANNELS
    sigma = PS.PatchSigmaTable(np.ones((18, ch, 3)), np.ones(ch),
                               np.full((18, ch, 3), 100))
    model = build_patch_model(use_cnn=True, n_channels=ch, d_model=8,
                              n_horizons=1, stem_widths=(8, 8),
                              n_covariates=batch["covariates"].shape[-1])
    model.eval()
    scorer = PS.PatchDeepScorer("patch_cnn_gru", model, sigma,
                                in_field=batch["in_field"], n_channels=ch, device="cpu")
    ref = PE.patch_reference_paths(scorer, batch, reference_idx=np.arange(32),
                                   field_row=np.arange(batch["raw"].shape[0]))
    arm = PS.PatchArmScorer(scorer, ref)
    cells = [("S3", "persistent"), ("S2", "gradual"), ("S4", "acute")]
    t0 = perf_counter()
    with torch.no_grad():
        for rep in range(loops):
            for level, family in cells:
                PE.patch_arm_scores(arm, batch, level_id=level, family=family,
                                    index="ndvi", round_idx=rep % 2,
                                    field_row=np.arange(batch["raw"].shape[0]),
                                    injection_basis="bands_consistent", chunk=16)
    return perf_counter() - t0


# --------------------------------------------------------------------------
# 3. TRAINING
# --------------------------------------------------------------------------
def run_train(loops: int):
    """Sample normalize + collate + one model forward/backward step."""
    import torch

    from ssl_anomaly.models.patch_cnn_gru import build_patch_model

    torch.set_num_threads(1)
    r = _rng()
    n, T, C, S = 8, P.N_BINS, P.N_CHANNELS, 32
    stats = P.patch_norm_stats({
        "stats": {s: {name: {"median": 0.4, "iqr": 0.1, "n": 100} for name in P.CHANNELS}
                  for s in ("L8", "L9", "unknown")},
        "channel_order": list(P.CHANNELS),
    })
    model = build_patch_model(use_cnn=True, n_channels=C, d_model=8,
                              n_horizons=1, stem_widths=(8, 8), n_covariates=6)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    t0 = perf_counter()
    for _ in range(loops * 4):
        raw = r.uniform(0.05, 0.6, size=(n, T, C, S, S)).astype(np.float32)
        obs = r.random((n, T, C, S, S)) > 0.15
        sensors = np.ones((n, T), dtype=np.uint8)
        x = np.stack([P.normalize_patch(raw[i], obs[i], sensors[i], stats)[0]
                      for i in range(n)])
        batch = {
            "x": x, "obs": obs, "pixel_valid": obs.any(axis=2),
            "covariates": r.normal(size=(n, T, 6)).astype(np.float32),
        }
        opt.zero_grad(set_to_none=True)
        out = model(torch.from_numpy(batch["x"]),
                    torch.from_numpy(batch["pixel_valid"]),
                    torch.from_numpy(batch["covariates"]))
        loss = out["pred"].mean() if isinstance(out, dict) else out.mean()
        loss.backward()
        opt.step()
    return perf_counter() - t0


# --------------------------------------------------------------------------
# Supporting parts (kept for comparison)
# --------------------------------------------------------------------------
def run_detectors(loops: int):
    r = _rng()
    BINS = 36
    s = r.normal(0, 1, size=BINS)
    t0 = perf_counter()
    for i in range(4000 * loops):
        x = s * (1.0 + 0.001 * (i % 7))
        cusum_detector(x, drift_k=1.0, decision_h=4.5)
        ewma_detector(x, alpha=0.35, L_sigma=2.8)
        flag_runs(x, threshold=3.5, min_run_bins=2)
        contiguous_runs(np.isfinite(x), min_length=2)
    return perf_counter() - t0


def run_affiliation(loops: int):
    gt = [(10, 20), (60, 75), (130, 150), (210, 230)]
    pred = [(8, 22), (58, 78), (128, 152), (208, 232), (300, 310)]
    t0 = perf_counter()
    for _ in range(400 * loops):
        pr_from_events(pred, gt, (0, 400))
    return perf_counter() - t0


def run_smoothing(loops: int):
    r = _rng()
    B, T = 240, 120
    y = r.normal(0.4, 0.08, size=(B, T))
    w = r.choice([0.0, 1.0], size=(B, T), p=[0.25, 0.75])
    y[r.random((B, T)) < 0.15] = np.nan
    backend = NumpyBackend()
    t0 = perf_counter()
    for _ in range(loops):
        out = backend.whittaker_batch(y, w, lam=100.0)
    assert out.shape == (B, T)
    return perf_counter() - t0


def run_baseline(loops: int):
    r = _rng()
    C, BINS, BANDS = 4000, 36, 18
    X = r.normal(0.4, 0.1, size=(C, BINS, BANDS))
    X[r.random(X.shape) < 0.2] = np.nan
    backend = NumpyBackend()
    t0 = perf_counter()
    for _ in range(loops):
        med, _ = backend.robust_baseline(X, min_n=5)
    assert med.shape == (BINS, BANDS)
    return perf_counter() - t0


def run_indices(loops: int):
    r = _rng()
    shape = (490, 180)
    red = r.uniform(0.02, 0.2, shape).astype(np.float32)
    nir = r.uniform(0.1, 0.5, shape).astype(np.float32)
    blue = r.uniform(0.01, 0.15, shape).astype(np.float32)
    swir = r.uniform(0.05, 0.4, shape).astype(np.float32)
    t0 = perf_counter()
    for _ in range(loops * 20):
        ndvi(red, nir)
        evi(red, nir, blue)
        ndwi(nir, swir)
    return perf_counter() - t0


RUNNERS = {
    "extract": run_extract, "eval": run_eval, "train": run_train,
    "detectors": run_detectors, "affiliation": run_affiliation,
    "smoothing": run_smoothing, "baseline": run_baseline, "indices": run_indices,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("all",) + PARTS, default="all")
    ap.add_argument("--loops", type=int, default=3)
    args = ap.parse_args()
    parts = PARTS if args.part == "all" else (args.part,)
    for part in parts:
        secs = RUNNERS[part](args.loops)
        print(f"[fa-bench] part={part} loops={args.loops} wall={secs:.3f}s", flush=True)


if __name__ == "__main__":
    main()

"""Throughput probe: steps/s and peak RAM per arm on the owner laptop (1.6).

OWNER-RUN (each arm must stay under 2 minutes; the agent never launches the
full probe). Measures optimiser steps/s and peak RSS per model × spatial unit
at the confirm configuration and records everything in
``reports/mini_dataset/cpu_budget.json`` — the numbers design.md §4.6 needs
before any long run is scheduled.

Usage::

    python scripts/probe_mini_throughput.py --smoke 5          # validate only
    python scripts/probe_mini_throughput.py                    # full probe
    python scripts/probe_mini_throughput.py --models mamba_ae gru_ae
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from ssl_anomaly.mini_lab.config import (
    ModelConfig,
    ProtocolConfig,
    RepresentationConfig,
    RunConfig,
)
from ssl_anomaly.mini_lab.data import build_dataset, split_datasets
from ssl_anomaly.mini_lab.models import MODEL_REGISTRY, build_model
from ssl_anomaly.mini_lab.runner import _masked_reconstruction_loss
from ssl_anomaly.utils.paths import localize  # noqa: E402
from ssl_anomaly.utils.seeds import seed_everything
from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402

STORE = localize("D:/RS/Thesis/Data/Processed/mini_research_dataset")
OUT = ROOT / "reports" / "mini_dataset" / "cpu_budget.json"

UNITS: tuple[tuple[str, str, tuple[int, ...]], ...] = (
    ("pixel", "pixel", ()),
    ("patch9", "patch", (9,)),
    ("full_field", "full_field", ()),
)


def peak_rss_gb() -> float:
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: KiB
    except Exception:
        import psutil

        return psutil.Process().memory_info().rss / 1e9


def probe_arm(model_name: str, unit: str, spatial: str, sizes: tuple[int, ...],
              ds, splits, *, steps: int, batch_size: int = 8) -> dict | None:
    """Time ``steps`` optimiser steps for one arm; None when it cannot run."""
    family = MODEL_REGISTRY[model_name].family
    train = splits[0][1]
    d_in = train.x.shape[-1]
    cfg_m = ModelConfig(name=model_name, d_model=64, latent_dim=32, n_layers=2,
                        batch_size=batch_size, device="cpu")
    seed_everything(13)
    model = build_model(model_name, d_in, cfg_m)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.from_numpy(train.x)
    obs = torch.from_numpy(train.observed)
    n = x.shape[0]
    if n == 0:
        return None
    params = int(sum(p.numel() for p in model.parameters()))
    model.train()
    t0 = time.monotonic()
    done = 0
    err = None
    try:
        while done < steps:
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(13 + done))
            for s in range(0, n, batch_size):
                idx = perm[s : s + batch_size]
                loss = _masked_reconstruction_loss(model, x[idx], obs[idx])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                done += 1
                if done >= steps:
                    break
    except Exception as exc:  # record the failure verbatim, never skip silently
        err = f"{type(exc).__name__}: {exc}"
    wall = time.monotonic() - t0
    return {
        "model": model_name,
        "unit": unit,
        "spatial": spatial,
        "family": family,
        "params": params,
        "steps": done if done == steps else None,
        "steps_per_s": round(done / wall, 2) if err is None and wall > 0 else None,
        "wall_s": round(wall, 2),
        "peak_rss_gb": round(peak_rss_gb(), 2),
        "error": err,
        "under_2min": err is None and wall < 120,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", type=int, default=None, metavar="STEPS",
                    help="probe with STEPS steps per arm (validation only)")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--units", nargs="+", default=None,
                    help="subset of: pixel patch9 full_field")
    ap.add_argument("--steps", type=int, default=50,
                    help="optimiser steps per arm (default 50; each arm must stay < 2 min)")
    ap.add_argument("--farms", nargs="+", default=["L02-02", "R12-08"])
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    t0 = time.monotonic()
    steps = args.smoke if args.smoke is not None else args.steps
    models = args.models or ["mlp_ae", "gru_ae", "lstm_ae", "cnn_ae", "mamba_ae",
                             "cnn_mamba", "barlow_twins", "jepa"]
    units_sel = [u for u in UNITS if args.units is None or u[0] in args.units]

    base = RunConfig(name="throughput_probe", store_root=STORE)
    base.representation = RepresentationConfig(
        spatial="pixel", patch_sizes=(9,), max_pixels=64, min_valid_fraction=0.8,
        temporal="sliding", window_length=30, stride=30, include_delta_t=True,
        include_indices=True, weather_group="weather_core",
    )
    base.protocol = ProtocolConfig(
        name="single_farm", train_farms=(args.farms[0],), test_farms=(args.farms[-1],),
        holdout_boundary="2024-01-01",
        justification="Throughput probe uses the accepted mini split",
    )
    ds = build_dataset(base, farm_ids=tuple(args.farms))
    splits = split_datasets(ds, base)

    arms = []
    for unit, spatial, sizes in units_sel:
        cfg = RunConfig(name=f"probe_{unit}", store_root=STORE)
        # clone the base representation (sliding windows, weather_core) and
        # vary only the spatial mode — a full 538-date series makes the
        # pure-torch SSM scan minutes-slow and the probe meaningless.
        cfg.representation = RepresentationConfig(
            spatial=spatial, patch_sizes=tuple(sizes) or (3, 5, 7),
            max_pixels=64, min_valid_fraction=0.8,
            temporal="sliding", window_length=30, stride=30,
            include_delta_t=True, include_indices=True, weather_group="weather_core",
        )
        sp = split_datasets(build_dataset(cfg, farm_ids=tuple(args.farms)), cfg)
        n_train = sp[0][1].n
        if n_train == 0:
            print(f"  [skip] {unit}: no train windows", flush=True)
            arms.append({"model": "-", "unit": unit, "error": "no train windows"})
            continue
        for m in models:
            rec = probe_arm(m, unit, spatial, sizes, ds, sp, steps=steps)
            print(f"  {m:14s} {unit:10s} "
                  f"{(rec['steps_per_s'] if rec and rec['steps_per_s'] else 'FAILED')} steps/s "
                  f"peak={rec['peak_rss_gb'] if rec else '?'} GB", flush=True)
            arms.append(rec)

    report = {
        "schema": "mini-cpu-budget/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_commit_sha(),
        "store": str(STORE),
        "machine": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_count_logical": os.cpu_count(),
            "cuda_available": torch.cuda.is_available(),
            "blas_threads": {k: os.environ.get(k) for k in
                             ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
                              "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        },
        "probe": {"steps_per_arm": steps, "batch_size": 8, "smoke": args.smoke is not None,
                  "farms": list(args.farms)},
        "arms": arms,
        "n_ok": sum(1 for a in arms if a and a["error"] is None),
        "n_failed": sum(1 for a in arms if a and a["error"]),
        "wall_s": round(time.monotonic() - t0, 2),
        "claim": "Measured throughput on this machine only (design §4.6); "
                 "not a performance comparison between architectures.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE {args.out} arms_ok={report['n_ok']} failed={report['n_failed']} "
          f"wall={report['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""CPU architecture x spatial-unit sweep on 1-2 farms of the mini store.

Honest screening: every architecture in the registry, every spatial unit from
single pixels to full-field aggregates, equal optimiser-step budget per unit,
fixed seeds, and the harness's random-init + step-detector controls. The script
never assumes a positive SSL result; it writes a ranked table with the control
gaps and parameter counts so a null outcome stays publishable.

Usage::

    python scripts/sweep_mini_architectures.py --stage screen --farms L02-02 R12-08
    python scripts/sweep_mini_architectures.py --stage confirm --farms L02-02 R12-08
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.mini_lab.config import (  # noqa: E402
    ModelConfig,
    ProtocolConfig,
    RepresentationConfig,
    RunConfig,
)
from ssl_anomaly.mini_lab.data import build_dataset, split_datasets  # noqa: E402
from ssl_anomaly.mini_lab.models import MODEL_REGISTRY, build_model  # noqa: E402
from ssl_anomaly.mini_lab.runner import run_experiment  # noqa: E402
from ssl_anomaly.utils.paths import localize  # noqa: E402
from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402

STORE = localize("D:/RS/Thesis/Data/Processed/mini_research_dataset")
OUT = ROOT / "reports" / "mini_dataset" / "arch_sweep"
SEEDS = (13, 42, 2026)


def units() -> list[tuple[str, str, tuple[int, ...]]]:
    out: list[tuple[str, str, tuple[int, ...]]] = [("pixel", "pixel", ())]
    for size in (1, 3, 5, 7, 9, 11):
        out.append((f"patch{size}", "patch", (size,)))
    out.append(("full_field", "full_field", ()))
    return out


def make_cfg(
    name: str,
    spatial: str,
    patch_sizes: tuple[int, ...],
    model: str,
    farms: tuple[str, ...],
    epochs: int,
    max_steps: int | None,
    seeds: tuple[int, ...],
    d_model: int,
    latent: int,
    n_layers: int,
) -> RunConfig:
    if len(farms) >= 2:
        proto = ProtocolConfig(
            name="single_farm", train_farms=(farms[0],), test_farms=(farms[1],),
            holdout_boundary="2024-01-01",
            justification="Accepted mini split: train <= 2022, dev 2023, test 2024+; cross-field rotation",
        )
    else:
        proto = ProtocolConfig(
            name="intra_farm_temporal", train_farms=(farms[0],), test_farms=(farms[0],),
            holdout_boundary="2024-01-01",
            justification="Accepted mini split: train <= 2022, dev 2023, test 2024+; same-field temporal rotation",
        )
    return RunConfig(
        name=name,
        store_root=STORE,
        representation=RepresentationConfig(
            spatial=spatial,
            patch_sizes=patch_sizes,
            max_pixels=64,
            min_valid_fraction=0.8,
            temporal="sliding",
            window_length=30,
            stride=30,
            include_delta_t=True,
            include_indices=True,
            weather_group="weather_core",
        ),
        protocol=proto,
        model=ModelConfig(
            name=model, d_model=d_model, latent_dim=latent, n_layers=n_layers,
            dropout=0.1, epochs=epochs, max_steps=max_steps,
            batch_size=8, lr=1e-3, seeds=seeds, device="cpu",
        ),
        out_root=OUT,
        n_trials=1,
        claim_rung="L2",
    )


def n_params(model: str, d_in: int, cfg: ModelConfig) -> int:
    try:
        return int(sum(p.numel() for p in build_model(model, d_in, cfg).parameters()))
    except Exception:
        return -1


def run(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    stage = args.stage
    target_steps = 100 if stage == "screen" else 400
    seeds = (13,) if stage == "screen" else tuple(args.seeds or SEEDS)
    d_model, latent, n_layers = (
        (32, 16, 1) if stage == "screen"
        else (int(args.d_model), int(args.latent_dim), int(args.n_layers))
    )
    models = list(args.models or MODEL_REGISTRY)
    stage_out = OUT / stage
    stage_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for unit_name, spatial, sizes in units():
        if args.units is not None and unit_name not in args.units:
            continue
        base = make_cfg(f"probe_{unit_name}", spatial, sizes, "gru_ae", tuple(args.farms),
                        target_steps, target_steps, seeds, d_model, latent, n_layers)
        try:
            ds = build_dataset(base, farm_ids=tuple(args.farms))
            splits = split_datasets(ds, base)
            n_train, n_test = splits[0][1].n, splits[0][2].n
            d_in = splits[0][1].x.shape[-1]
            n_batch = max(1, int(np.ceil(n_train / base.model.batch_size)))
            epochs = max(1, round(target_steps / n_batch))
        except Exception as exc:
            rows.append({"unit": unit_name, "error": f"dataset: {exc}", "traceback": traceback.format_exc()})
            print(f"[skip] {unit_name}: dataset failed: {exc}", flush=True)
            continue
        print(f"[unit] {unit_name}: samples train={n_train} test={n_test} d_in={d_in} "
              f"steps/run={epochs*n_batch} epochs={epochs}", flush=True)
        for model in models:
            cfg = make_cfg(f"{unit_name}__{model}", spatial, sizes, model, tuple(args.farms),
                           target_steps, target_steps, seeds, d_model, latent, n_layers)
            cfg.model.batch_size = base.model.batch_size
            row: dict[str, Any] = {
                "unit": unit_name, "spatial": spatial, "patch_size": sizes[0] if sizes else None,
                "model": model, "n_train": n_train, "n_test": n_test, "d_in": d_in,
                "epochs_cap": epochs, "steps": target_steps, "seeds": list(seeds),
                "params": n_params(model, d_in, cfg.model),
                "family": MODEL_REGISTRY[model].family,
                "placement": MODEL_REGISTRY[model].placement,
            }
            try:
                summary = run_experiment(cfg, ds, out_root=stage_out / f"{unit_name}__{model}")
                agg = summary["aggregate"]
                row.update({
                    "synthetic_auprc": agg["synthetic_auprc"],
                    "random_init_auprc": agg["random_init_auprc"],
                    "step_reference_auprc": agg["step_reference_auprc"],
                    "ssl_gain_over_random": (
                        round(agg["synthetic_auprc"]["mean"] - agg["random_init_auprc"]["mean"], 4)
                        if agg["synthetic_auprc"] and agg["random_init_auprc"] else None
                    ),
                    "wall_s": summary["wall_s"],
                    "report": str(stage_out / f"{unit_name}__{model}" / "report.json"),
                })
                print(f"  {model:14s} AUPRC={row['synthetic_auprc']['mean']:.4f} "
                      f"rand={row['random_init_auprc']['mean']:.4f} "
                      f"gain={row['ssl_gain_over_random']:+.4f}", flush=True)
            except Exception as exc:
                row["error"] = str(exc)
                row["traceback"] = traceback.format_exc()
                print(f"  {model:14s} FAILED: {exc}", flush=True)
            rows.append(row)

    good = [r for r in rows if r.get("synthetic_auprc")]
    ranked = sorted(good, key=lambda r: (r["ssl_gain_over_random"] is None,
                                         -(r["ssl_gain_over_random"] or -1)))
    report = {
        "schema": "mini-arch-sweep/v1",
        "stage": stage,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "farms": list(args.farms),
        "git_sha": get_git_commit_sha(),
        "store": str(STORE),
        "manifest_sha256": __import__("hashlib").sha256((STORE / "manifest.json").read_bytes()).hexdigest(),
        "target_steps_per_run": target_steps,
        "seeds": list(seeds),
        "d_model": d_model, "latent_dim": latent, "n_layers": n_layers,
        "n_configs": len(rows), "n_success": len(good), "n_failed": len(rows) - len(good),
        "ranking_by_gain_over_random_init": [r["unit"] + "__" + r["model"] for r in ranked],
        "rows": rows,
        "wall_s": round(time.monotonic() - t0, 2),
        "claim": "CPU screening only; equal optimiser steps, random-init control recorded; "
                 "no claim above L2 and no SSL-advantage claim without CI excluding zero.",
    }
    path = stage_out / "sweep.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"SWEEP COMPLETE wall={report['wall_s']}s configs={len(rows)} ok={len(good)} "
          f"failed={report['n_failed']} report={path}", flush=True)
    if ranked:
        top = ranked[0]
        print(f"TOP (by gain over random init): {top['unit']}__{top['model']} "
              f"gain={top['ssl_gain_over_random']:+.4f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("screen", "confirm"), default="screen")
    ap.add_argument("--farms", nargs="+", default=["L02-02", "R12-08"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--units", nargs="+", default=None,
                    help="Restrict spatial units (default: all 8)")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2)
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

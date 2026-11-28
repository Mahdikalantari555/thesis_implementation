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
import torch

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
    *,
    plateau: bool = False,
    eval_every: int = 25,
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
            # Task 1.4: to-plateau arms train under the convergence-controlled
            # trainer; max_steps stays the hard cap and per-eval checkpoints
            # enable the equal-step cut (design §4.5).
            to_plateau=plateau,
            keep_eval_ckpts=plateau,
            eval_every=eval_every,
            min_steps=2 * eval_every,
            patience=4,
            window=5,
            lr_patience=None,
            max_lr_drops=0,
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


def cfg_model_kwargs(d_model: int, latent: int, n_layers: int, seeds, batch_size: int) -> dict:
    """Model-config kwargs for equal-step-cut scoring (must match run configs)."""
    return dict(d_model=d_model, latent_dim=latent, n_layers=n_layers,
                dropout=0.1, seeds=tuple(seeds), batch_size=batch_size, lr=1e-3)


def _load_model_at_step(run_dir: Path, target_step: int) -> tuple[int, dict] | None:
    """Nearest saved per-eval checkpoint at or before ``target_step``."""
    ckpt_dir = run_dir / "eval_ckpts"
    if not ckpt_dir.exists():
        return None
    steps = sorted(int(p.stem.split("_")[1]) for p in ckpt_dir.glob("step_*.pt"))
    if not steps:
        return None
    eligible = [s for s in steps if s <= target_step] or [steps[0]]
    step = eligible[-1]
    payload = torch.load(ckpt_dir / f"step_{step:07d}.pt", map_location="cpu", weights_only=False)
    return step, payload["model"]


def equal_step_cut(stage_out: Path, unit_rows: list[dict], ds, splits, mk_kwargs: dict):
    """Design §4.5: re-score every arm at the smallest arm's plateau step.

    Per seed: cut step = min over arms of (plateau step when the arm stopped
    at CONVERGED_PLATEAU, else its last eval step). Each arm is scored from
    its per-eval checkpoint at that step; random-init controls are already
    step-independent and reused from the arm's report. Arms without saved
    eval checkpoints report None — never a fabricated number.
    """
    from ssl_anomaly.mini_lab.trainer import score_state_dict

    device = torch.device("cpu")
    out: list[dict | None] = []
    for name, train, dev, test in splits:
        xtr = torch.from_numpy(train.x)
        xte = torch.from_numpy(test.x)
        obs_te = torch.from_numpy(test.observed)
        # cut step per seed: smallest arm's plateau/last-eval step
        cut_steps: dict[int, int] = {}
        for r in unit_rows:
            for psd in (r.get("to_plateau_detail") or []):
                seed = psd.get("seed")
                step = (
                    psd.get("best_primary_step")
                    if psd.get("verdict") == "CONVERGED_PLATEAU" and psd.get("best_primary_step")
                    else psd.get("steps_done")
                )
                if seed is None or step is None:
                    continue
                cut_steps[seed] = min(cut_steps.get(seed, step), int(step))
        if not cut_steps:
            out.append(None)
            continue
        per_seed: dict[str, Any] = {}
        for r in unit_rows:
            if r.get("error") or not r.get("to_plateau"):
                continue
            run_dir = stage_out / f"{r['unit']}__{r['model']}"
            per_seed[r["model"]] = {}
            for seed, cut_step in sorted(cut_steps.items()):
                loaded = _load_model_at_step(run_dir / name / f"seed_{seed}", cut_step)
                if loaded is None:
                    per_seed[r["model"]][str(seed)] = None  # honest absence
                    continue
                actual_step, state = loaded
                from ssl_anomaly.mini_lab.models import MODEL_REGISTRY

                cfg_m = ModelConfig(name=r["model"], **mk_kwargs)
                scores = score_state_dict(
                    RunConfig(name="cut", model=cfg_m), state, xtr, xte, obs_te,
                    MODEL_REGISTRY[r["model"]], seed, device, ds.feature_names,
                )
                scores["cut_step"] = cut_step
                scores["actual_step"] = actual_step
                per_seed[r["model"]][str(seed)] = scores
        # aggregate cut AUPRC per arm (mean over seeds with a value)
        arms: dict[str, Any] = {}
        for model_name, seeds_scores in per_seed.items():
            vals = [v["synthetic_auprc"] for v in seeds_scores.values() if v]
            steps_used = [v["actual_step"] for v in seeds_scores.values() if v]
            arms[model_name] = {
                "equal_step_auprc_mean": (
                    float(np.mean([v for v in vals if v is not None])) if vals else None
                ),
                "n_seeds_scored": len(vals),
                "actual_steps": steps_used,
            }
        out.append({"cut_steps": cut_steps, "arms": arms})
    return out


def run(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    out_root = Path(args.out) if getattr(args, "out", None) else OUT
    stage = args.stage
    target_steps = 100 if stage == "screen" else 400
    seeds = (13,) if stage == "screen" else tuple(args.seeds or SEEDS)
    d_model, latent, n_layers = (
        (32, 16, 1) if stage == "screen"
        else (int(args.d_model), int(args.latent_dim), int(args.n_layers))
    )
    models = list(args.models or MODEL_REGISTRY)
    plateau = bool(getattr(args, "plateau", False))
    eval_every = int(getattr(args, "eval_every", 25))
    stage_out = out_root / stage
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
            n_train, n_test = splits[0][1].n, splits[0][3].n
            d_in = splits[0][1].x.shape[-1]
            n_batch = max(1, int(np.ceil(n_train / base.model.batch_size)))
            epochs = max(1, round(target_steps / n_batch))
        except Exception as exc:
            rows.append({"unit": unit_name, "error": f"dataset: {exc}", "traceback": traceback.format_exc()})
            print(f"[skip] {unit_name}: dataset failed: {exc}", flush=True)
            continue
        print(f"[unit] {unit_name}: samples train={n_train} test={n_test} d_in={d_in} "
              f"steps/run={epochs*n_batch} epochs={epochs} plateau={plateau}", flush=True)
        unit_rows: list[dict[str, Any]] = []
        for model in models:
            cfg = make_cfg(f"{unit_name}__{model}", spatial, sizes, model, tuple(args.farms),
                           target_steps, target_steps, seeds, d_model, latent, n_layers,
                           plateau=plateau, eval_every=eval_every)
            cfg.model.batch_size = base.model.batch_size
            row: dict[str, Any] = {
                "unit": unit_name, "spatial": spatial, "patch_size": sizes[0] if sizes else None,
                "model": model, "n_train": n_train, "n_test": n_test, "d_in": d_in,
                "epochs_cap": epochs, "steps": target_steps, "seeds": list(seeds),
                "params": n_params(model, d_in, cfg.model),
                "family": MODEL_REGISTRY[model].family,
                "placement": MODEL_REGISTRY[model].placement,
                "to_plateau": plateau,
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
                if plateau:
                    per_seed_verdicts = [
                        {"seed": r.get("seed"), "verdict": r.get("verdict"),
                         "steps_done": r.get("steps_done"),
                         "best_primary_step": r.get("best_primary_step"),
                         "extra_steps_estimate": r.get("extra_steps_estimate")}
                        for r in summary.get("per_seed", [])
                    ]
                    row["to_plateau_detail"] = per_seed_verdicts
                # None aggregates are reported honestly (finding 28's recorded
                # "JEPA full-field 'NoneType' object is not subscriptable" was
                # this print crashing on a None aggregate, itself caused by
                # NaN scores from a batch-of-1 barlow loss — fixed in runner).
                def _fmt(v):
                    return (f"{v['mean']:.4f}"
                            if isinstance(v, dict) and v.get("mean") is not None else "None")
                print(f"  {model:14s} AUPRC={_fmt(row['synthetic_auprc'])} "
                      f"rand={_fmt(row['random_init_auprc'])} "
                      f"gain={row['ssl_gain_over_random']}", flush=True)
            except Exception as exc:
                row["error"] = str(exc)
                row["traceback"] = traceback.format_exc()
                print(f"  {model:14s} FAILED: {exc}", flush=True)
            rows.append(row)
            unit_rows.append(row)
        if plateau:
            cut = equal_step_cut(stage_out, unit_rows, ds, splits, cfg_model_kwargs(
                d_model, latent, n_layers, seeds, base.model.batch_size))
            if cut:
                for r, cut_row in zip(unit_rows, cut):
                    if cut_row:
                        r["equal_step_cut"] = cut_row
                cut_all = [v for c in cut if c and c.get("cut_steps") for v in c["cut_steps"].values()]
                if cut_all:
                    print(f"  [equal-step cut] smallest cut step={min(cut_all)} "
                          f"arms_scored={sum(1 for c in cut if c)}", flush=True)

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
        "to_plateau": plateau,
        "seeds": list(seeds),
        "d_model": d_model, "latent_dim": latent, "n_layers": n_layers,
        "n_configs": len(rows), "n_success": len(good), "n_failed": len(rows) - len(good),
        "ranking_by_gain_over_random_init": [r["unit"] + "__" + r["model"] for r in ranked],
        "rows": rows,
        "wall_s": round(time.monotonic() - t0, 2),
        "claim": "CPU screening only; equal optimiser steps, random-init control recorded; "
                 "no claim above L2 and no SSL-advantage claim without CI excluding zero."
                 + (" To-plateau arms report both the to-plateau result and the "
                    "equal-step cut at the smallest arm's plateau step (design §4.5)."
                    if plateau else ""),
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
    ap.add_argument("--plateau", action="store_true",
                    help="to-plateau arms (task 1.4) with the equal-step cut (design §4.5)")
    ap.add_argument("--eval-every", type=int, default=25,
                    help="evaluations every N optimizer steps in --plateau mode")
    ap.add_argument("--farms", nargs="+", default=["L02-02", "R12-08"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--units", nargs="+", default=None,
                    help="Restrict spatial units (default: all 8)")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--out", default=None,
                    help="output root (default: reports/mini_dataset/arch_sweep). Use a distinct "
                         "root for a re-run so an already-committed screening result is not "
                         "silently replaced.")
    return run(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

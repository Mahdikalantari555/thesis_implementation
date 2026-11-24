"""Run a mini-dataset SSL experiment from a YAML config.

Usage::

    python scripts/run_mini_experiment.py --config configs/mini/run_smoke.yaml
    python scripts/run_mini_experiment.py --model barlow_twins --smoke
    python scripts/run_mini_experiment.py --all-protocols-smoke
    python scripts/run_mini_experiment.py --list-models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ssl_anomaly.mini_lab import build_dataset, load_run_config, model_specs, run_experiment  # noqa: E402
from ssl_anomaly.mini_lab.config import RunConfig  # noqa: E402


def _apply_overrides(cfg: RunConfig, args: argparse.Namespace) -> None:
    if args.model:
        cfg.model.name = args.model
    if args.protocol:
        cfg.protocol.name = args.protocol
    if args.spatial:
        cfg.representation.spatial = args.spatial
    if args.temporal:
        cfg.representation.temporal = args.temporal
    if args.epochs is not None:
        cfg.model.epochs = args.epochs
    if args.seeds:
        cfg.model.seeds = tuple(args.seeds)
    if args.device:
        cfg.model.device = args.device
    if args.out:
        cfg.out_root = Path(args.out)
    if args.resume:
        cfg.resume = True
    if args.smoke:
        cfg.model.epochs = min(cfg.model.epochs, 2)
        cfg.model.seeds = (13,)
        cfg.n_trials = 1
        cfg.name = cfg.name + "_smoke"


def _execute(cfg: RunConfig) -> dict:
    print(
        f"Running {cfg.name}: model={cfg.model.name} protocol={cfg.protocol.name} "
        f"spatial={cfg.representation.spatial} temporal={cfg.representation.temporal}",
        flush=True,
    )
    dataset = build_dataset(cfg)
    print(
        f"Dataset: {dataset.n_units} units, {len(dataset.feature_names)} features; "
        f"patch counts={dataset.valid_patch_counts}",
        flush=True,
    )
    return run_experiment(cfg, dataset)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "mini" / "run_smoke.yaml")
    ap.add_argument("--model", default=None)
    ap.add_argument("--protocol", default=None)
    ap.add_argument("--spatial", default=None)
    ap.add_argument("--temporal", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--all-protocols-smoke", action="store_true")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args(argv)

    if args.list_models:
        print(json.dumps(model_specs(), indent=2))
        return 0

    base = load_run_config(args.config)
    _apply_overrides(base, args)
    if args.all_protocols_smoke:
        summaries = {}
        for protocol in ("single_farm", "lodo", "intra_farm_temporal"):
            cfg = load_run_config(args.config)
            _apply_overrides(cfg, args)
            cfg.protocol.name = protocol
            cfg.name = f"{cfg.name}_{protocol}"
            if protocol == "lodo":
                cfg.model.seeds = (13,)
            summaries[protocol] = _execute(cfg)
        print(
            "ALL PROTOCOLS COMPLETE "
            + json.dumps({k: v["wall_s"] for k, v in summaries.items()}),
            flush=True,
        )
        return 0
    _execute(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

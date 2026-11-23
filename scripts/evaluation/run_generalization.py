#!/usr/bin/env python3
"""Temporal rolls and LOBO generalisation (design §Robustness, gate G-C).

Change: regional-ssl-representation-evaluation. Writes ``generalization.json``.

Both families run and both are reported with their spread. A single fold is
never a robustness result, and incomplete LOBO coverage (fewer than the
pre-registered blocks) fails G-C rather than passing on the folds that exist.

Usage
-----
::

    python scripts/evaluation/run_generalization.py --smoke
    python scripts/evaluation/run_generalization.py \
        --cycles data/processed/cycles_490_v2.parquet \
        --labels data/processed/anomaly_labels_v2.parquet \
        --scores reports/regional_representation/scores_v0.npz \
        --blocks data/processed/blocks_v2.parquet \
        --cut-years-config configs/evaluation/temporal_rolls.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import numpy as np  # noqa: E402

from _common import add_common_args, finish_smoke, require_real_data, resolve_run  # noqa: E402
from ssl_anomaly.evaluation.binding import bind_artifacts, load_preregistration  # noqa: E402
from ssl_anomaly.evaluation.data import load_cycle_table  # noqa: E402
from ssl_anomaly.evaluation.provenance import honest_state_block  # noqa: E402
from ssl_anomaly.evaluation.robustness import run_robustness  # noqa: E402
from ssl_anomaly.evaluation.runner import (  # noqa: E402
    SmokeExpectation,
    build_manifest,
    make_smoke_fixture,
    write_run_outputs,
)

DEFAULT_CUTS = (2016, 2018, 2020, 2022)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--cycles", type=Path, default=Path("data/processed/cycles_490_v2.parquet"))
    p.add_argument("--labels", type=Path,
                   default=Path("data/processed/anomaly_labels_v2.parquet"))
    p.add_argument("--scores", type=Path, default=None,
                   help=".npz with cycle_ids + scores from the frozen detector")
    p.add_argument("--blocks", type=Path, default=None,
                   help="parquet with cycle_id + block_id (the pre-registered blocks)")
    p.add_argument("--cut-years-config", type=Path,
                   default=Path("configs/evaluation/temporal_rolls.yaml"))
    p.add_argument("--n-boot", type=int, default=300)
    return p.parse_args(argv)


def _load_cut_years(path: Path) -> tuple[int, ...]:
    """Read the cuts from the committed config, not from a constant in the script."""
    import yaml

    if not path.exists():
        print(f"note: {path} absent; using the pre-registered cuts {DEFAULT_CUTS}",
              file=sys.stderr)
        return DEFAULT_CUTS
    payload = yaml.safe_load(path.read_text())
    cuts = payload.get("temporal_rolls", {}).get("cut_years", list(DEFAULT_CUTS))
    return tuple(int(c) for c in cuts)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    prereg = load_preregistration()
    ctx = resolve_run(args, stage="generalization")
    cuts = _load_cut_years(args.cut_years_config)

    # ---- data ------------------------------------------------------------
    if ctx.smoke:
        fixture = make_smoke_fixture(n_fields=20, cycles_per_field=5, seed=13)
        dev = fixture.dev
        labels = fixture.labels
        years = fixture.years
        blocks = fixture.block_ids
        scores = _smoke_scores(dev, labels, fixture.fields, seed=13)
        n_blocks_expected = int(np.unique(blocks).size)
    else:
        require_real_data(args.cycles, "the v2 cycle table")
        require_real_data(args.labels, "the v2 anomaly labels")
        require_real_data(args.scores, "the frozen detector scores")
        require_real_data(args.blocks, "the block assignment table")
        bind_artifacts(prereg=prereg, strict=True,
                       require=("cycles_490_v2", "splits_locked_v2"))
        dev = load_cycle_table(args.cycles).for_split("dev")
        labels = _align(args.labels, dev, "label", dtype=int)
        years = _align(args.blocks, dev, "year", dtype=float).astype(int)
        blocks = _align(args.blocks, dev, "block_id", dtype=str)
        scores = _align_npz(args.scores, dev)
        n_blocks_expected = 20      # pre-registered, from the frozen split manifest

    report_core = run_robustness(
        scores=scores, labels=labels, years=years, block_ids=blocks,
        clusters=dev.field_ids, cut_years=cuts,
        n_blocks_expected=n_blocks_expected,
    )

    report = {
        "stage": "generalization",
        "smoke": ctx.smoke,
        "cut_years": list(cuts),
        "n_blocks_expected": int(n_blocks_expected),
        **report_core.to_dict(),
        "gate_evidence": report_core.gate_evidence(),
        "honest_state": honest_state_block(
            measured=[f["fold"] for f in report_core.to_dict()["temporal_rolls"]
                      if f["status"] == "measured"],
            absent=[f["fold"] for f in report_core.to_dict()["lobo"]
                    if f["status"] != "measured"],
            deferred=[],
        ),
        "reading_rule": (
            "a fold-level AP is a generalisation result for THAT shift. The summary "
            "requires a median floor AND breadth across folds; a good median in a "
            "minority of folds is not robustness, and incomplete LOBO coverage fails "
            "G-C even if every present fold looks strong"
        ),
        "wall_time_s": round(time.time() - started, 2),
    }

    manifest = build_manifest(
        stage="generalization", smoke=ctx.smoke,
        config={"cut_years": list(cuts), "n_boot": args.n_boot,
                "scores": str(args.scores), "blocks": str(args.blocks)},
        seeds=args.seeds,
    )
    written = write_run_outputs(paths=ctx.paths, manifest=manifest,
                                reports={"generalization": report})

    if ctx.smoke:
        exp = SmokeExpectation("generalization")
        rolls = report["temporal_rolls"]
        exp.check("one fold per pre-registered cut year",
                  [f["fold"] for f in rolls] == [f"roll-{c}" for c in cuts],
                  str([f["fold"] for f in rolls]))
        exp.check("unusable rolls are kept in the table, not dropped",
                  len(rolls) == len(cuts))
        exp.check("every absent fold carries a reason",
                  all(f["reason"] for f in rolls + report["lobo"]
                      if f["status"] != "measured")
                  or all(f["status"] == "measured" for f in rolls))
        exp.check("LOBO has one fold per observed block",
                  len(report["lobo"]) == len(set(np.asarray(blocks).tolist())))
        exp.check("a summary exists for both families",
                  report["temporal_summary"]["n_folds"] > 0
                  and report["lobo_summary"]["n_folds"] > 0)
        exp.check("robustness is never claimed from one fold",
                  not (report["temporal_summary"]["n_measured"] < 2
                       and report["temporal_summary"]["robust"]))
        exp.check("gate evidence has exactly the six G-C keys",
                  set(report["gate_evidence"]) == {
                      "temporal_min_ap", "temporal_median_ap", "temporal_robust",
                      "lobo_median_ap", "lobo_robust", "lobo_coverage_complete"})
        report["smoke_checks"] = exp.to_dict()
        write_run_outputs(paths=ctx.paths, manifest=manifest,
                          reports={"generalization": report})
        print(f"wrote {written['generalization']}")
        return finish_smoke(exp, ctx.paths, "generalization")

    print(f"wrote {written['generalization']}")
    return 0


def _smoke_scores(dev, labels, field_ids, *, seed: int) -> np.ndarray:
    """Scores that carry real signal AND realistic fold-to-fold variation."""
    rng = np.random.default_rng(seed)
    x = np.where(dev.M, dev.X, np.nan)
    with np.errstate(invalid="ignore"):
        field_mean = np.nanmean(x, axis=1)
    rel = np.empty_like(field_mean)
    for fid in np.unique(field_ids):
        sel = field_ids == fid
        rel[sel] = field_mean[sel] - np.nanmedian(field_mean[sel], axis=0)
    signal = -rel[:, 0]                                  # low-NDVI cycles score high
    signal = (signal - signal.mean()) / (signal.std() + 1e-9)
    return signal + rng.normal(0, 0.6, size=signal.shape)


def _align(path: Path, dev, column: str, dtype) -> np.ndarray:
    import pandas as pd

    df = pd.read_parquet(path)
    if column not in df.columns:
        raise SystemExit(f"{path} needs a '{column}' column; got {list(df.columns)}")
    lut = dict(zip(df["cycle_id"].astype(str), df[column].astype(dtype), strict=False))
    missing = [c for c in dev.cycle_ids if str(c) not in lut]
    if missing:
        raise SystemExit(
            f"{len(missing)} dev cycles have no '{column}' (e.g. {missing[0]}); "
            f"refusing to fabricate one"
        )
    return np.array([lut[str(c)] for c in dev.cycle_ids])


def _align_npz(path: Path, dev) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    ids = [str(i) for i in z["cycle_ids"]]
    lut = {cid: i for i, cid in enumerate(ids)}
    missing = [c for c in dev.cycle_ids if str(c) not in lut]
    if missing:
        raise SystemExit(f"{len(missing)} dev cycles missing from {path}")
    return z["scores"][[lut[str(c)] for c in dev.cycle_ids]]


if __name__ == "__main__":
    raise SystemExit(main())

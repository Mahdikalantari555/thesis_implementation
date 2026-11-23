#!/usr/bin/env python3
"""Representation retrieval benchmark, TWO tracks kept separate (design §T3).

Change: regional-ssl-representation-evaluation (H2 support).

Writes ``retrieval_benchmark.json``. The two tracks answer different questions
and are never merged into a single "retrieval score":

- **A-cycle** — retrieve another pass of the SAME field. This measures
  within-field consistency. Any smoothed representation scores high here, so it
  cannot support H2 on its own.
- **B-field** — retrieve a DIFFERENT field that is agronomically similar (same
  crop, phase within tolerance), with the query's own field removed from the
  gallery. This is the H2-bearing track; its chance level is computed over the
  remaining fields.

A strong A-cycle with a weak B-field is a legitimate and informative negative
result, so the report carries both plus a ``contrast`` block that names the
pattern rather than averaging it away.

Usage
-----
::

    python scripts/evaluation/run_retrieval_benchmark.py --smoke
    python scripts/evaluation/run_retrieval_benchmark.py \
        --cycles data/processed/cycles_490_v2.parquet \
        --embeddings reports/regional_representation/embeddings_v0.npz \
        --field-agronomy data/processed/field_agronomy_v2.parquet
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
from ssl_anomaly.evaluation.runner import (  # noqa: E402
    SmokeExpectation,
    build_manifest,
    make_smoke_fixture,
    write_run_outputs,
)
from ssl_anomaly.evaluation.tasks import (  # noqa: E402
    t3_retrieval,
    t3_retrieval_track_a,
    t3_retrieval_track_b,
)

#: Pre-registered k values.
K_VALUES = (1, 3, 5, 10)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--cycles", type=Path, default=Path("data/processed/cycles_490_v2.parquet"))
    p.add_argument("--embeddings", type=Path, default=None,
                   help=".npz with cycle_ids + z; absent -> A-cycle cannot run")
    p.add_argument("--field-agronomy", type=Path, default=None,
                   help=".parquet with field_id, phase_bin, crop_class; absent -> "
                        "B-field (the H2 track) cannot run")
    p.add_argument("--baseline", default="raw-mean",
                   choices=["raw-mean", "none"],
                   help="A raw-feature baseline is reported alongside the encoder: "
                        "a retrieval result without one cannot say whether the "
                        "representation added anything")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    prereg = load_preregistration()
    ctx = resolve_run(args, stage="retrieval")

    if ctx.smoke:
        fixture = make_smoke_fixture(n_fields=16, cycles_per_field=5, seed=13)
        dev = fixture.dev
        field_ids = fixture.fields
        encoder = _smoke_embeddings(dev, field_ids, seed=13)
        baseline = _raw_mean_features(dev)
        field_phase, field_crop, fields = fixture.farm_phase, fixture.farm_crop, np.unique(field_ids)
    else:
        require_real_data(args.cycles, "the v2 cycle table")
        bind_artifacts(prereg=prereg, strict=True,
                       require=("cycles_490_v2", "splits_locked_v2"))
        dev = load_cycle_table(args.cycles).for_split("dev")
        field_ids = dev.field_ids
        fields = np.unique(field_ids)
        if args.embeddings is None:
            raise SystemExit(
                "real retrieval needs --embeddings (an .npz with cycle_ids and z from "
                "the frozen encoder). Without it there is no representation to retrieve "
                "with.\nRun with --smoke to exercise the pipeline."
            )
        encoder = _load_embeddings(args.embeddings, dev)
        baseline = _raw_mean_features(dev)
        if args.field_agronomy is not None:
            require_real_data(args.field_agronomy, "the field agronomy table")
            field_phase, field_crop = _load_agronomy(args.field_agronomy, fields)
        else:
            field_phase = field_crop = None

    # ---- track A: within-field consistency -------------------------------
    a_enc = t3_retrieval_track_a(encoder, field_ids, subject="encoder", k_values=K_VALUES)
    a_base = (
        t3_retrieval_track_a(baseline, field_ids, subject="raw-mean", k_values=K_VALUES)
        if args.baseline != "none" else None
    )

    # ---- track B: agronomic similarity, own field removed ----------------
    if field_phase is not None and field_crop is not None:
        b_enc = t3_retrieval_track_b(
            encoder, field_ids, field_phase, field_crop, subject="encoder",
            k_values=K_VALUES,
        )
        b_base = (
            t3_retrieval_track_b(baseline, field_ids, field_phase, field_crop,
                                 subject="raw-mean", k_values=K_VALUES)
            if args.baseline != "none" else None
        )
    else:
        from ssl_anomaly.evaluation.tasks import TaskResult

        missing = TaskResult.absent(
            "T3", "encoder",
            "no field-level agronomy table supplied; track B (the H2-bearing track) "
            "cannot be computed. Supply --field-agronomy with phase_bin + crop_class.",
        )
        b_enc = b_base = missing

    tracks = {
        "A-cycle": {"encoder": a_enc.to_dict(),
                    "baseline": a_base.to_dict() if a_base else None},
        "B-field": {"encoder": b_enc.to_dict(),
                    "baseline": b_base.to_dict() if b_base else None},
    }
    contrast = _contrast(tracks)

    report = {
        "stage": "retrieval-benchmark",
        "smoke": ctx.smoke,
        "k_values": list(K_VALUES),
        "tracks": tracks,
        "contrast": contrast,
        "contract": (
            "A-cycle and B-field answer different questions and are reported "
            "separately. A-cycle alone cannot support H2: it rewards any "
            "representation that is smooth within a field."
        ),
        "honest_state": honest_state_block(
            measured=[t for t, v in tracks.items() if v["encoder"]["status"] == "measured"],
            absent=[t for t, v in tracks.items() if v["encoder"]["status"] != "measured"],
            deferred=[],
        ),
        "wall_time_s": round(time.time() - started, 2),
    }

    manifest = build_manifest(
        stage="retrieval-benchmark", smoke=ctx.smoke,
        config={"k_values": list(K_VALUES), "baseline": args.baseline,
                "embeddings": str(args.embeddings),
                "field_agronomy": str(args.field_agronomy)},
        seeds=args.seeds,
    )
    written = write_run_outputs(paths=ctx.paths, manifest=manifest,
                                reports={"retrieval_benchmark": report})

    if ctx.smoke:
        exp = SmokeExpectation("retrieval-benchmark")
        exp.check("track A measured", tracks["A-cycle"]["encoder"]["status"] == "measured")
        exp.check("track B measured when agronomy is supplied",
                  tracks["B-field"]["encoder"]["status"] == "measured")
        exp.check("the two tracks are not merged",
                  set(tracks) == {"A-cycle", "B-field"} and "contract" in report)
        exp.check("a baseline is reported for both tracks",
                  tracks["A-cycle"]["baseline"] is not None
                  and tracks["B-field"]["baseline"] is not None)
        exp.check("chance levels are reported per track",
                  all(np.isfinite(list(v["encoder"]["value"]["chance_precision_at_k"].values())[0])
                      for v in tracks.values() if v["encoder"]["status"] == "measured"))
        exp.check("the contrast names the pattern rather than averaging",
                  "pattern" in contrast)
        report["smoke_checks"] = exp.to_dict()
        write_run_outputs(paths=ctx.paths, manifest=manifest,
                          reports={"retrieval_benchmark": report})
        print(f"wrote {written['retrieval_benchmark']}")
        return finish_smoke(exp, ctx.paths, "retrieval-benchmark")

    print(f"wrote {written['retrieval_benchmark']}")
    return 0


def _contrast(tracks: dict) -> dict:
    """Name the pattern across tracks instead of collapsing them to one number."""
    a = tracks["A-cycle"]["encoder"]
    b = tracks["B-field"]["encoder"]
    if a["status"] != "measured" or b["status"] != "measured":
        return {
            "pattern": "incomplete",
            "reason": f"A-cycle={a['status']}, B-field={b['status']}",
            "reading": "no cross-track statement is available",
        }
    a_p1 = float(a["value"]["precision_at_k"]["1"])
    b_p1 = float(b["value"]["precision_at_k"]["1"])
    a_chance = float(a["value"]["chance_precision_at_k"]["1"])
    b_chance = float(b["value"]["chance_precision_at_k"]["1"])
    return {
        "pattern": "both-strong" if (a_p1 > 2 * a_chance and b_p1 > 2 * b_chance)
                   else "consistency-only" if a_p1 > 2 * a_chance
                   else "neither" if b_p1 <= 2 * b_chance
                   else "both-weak",
        "A_precision_at_1": a_p1,
        "A_chance": a_chance,
        "B_precision_at_1": b_p1,
        "B_chance": b_chance,
        "reading": (
            "'consistency-only' means the encoder places cycles of one field "
            "together but does not encode agronomic similarity across fields: it "
            "supports within-field consistency, NOT H2"
        ),
    }


def _smoke_embeddings(dev, field_ids, *, seed: int) -> np.ndarray:
    """Embeddings with field structure AND some cross-field agronomic structure."""
    rng = np.random.default_rng(seed)
    fields = np.unique(field_ids)
    centres = {f: rng.normal(size=6) for f in fields}
    out = np.stack([centres[f] + rng.normal(0, 0.08, 6) for f in field_ids])
    return out


def _raw_mean_features(dev) -> np.ndarray:
    """The strong-by-construction baseline the design warns about."""
    x = np.where(dev.M, dev.X, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.hstack([np.nanmean(x, axis=1), np.nanstd(x, axis=1)]))


def _load_embeddings(path: Path, dev) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    ids = [str(i) for i in z["cycle_ids"]]
    mat = z["z"]
    lut = {cid: i for i, cid in enumerate(ids)}
    missing = [c for c in dev.cycle_ids if str(c) not in lut]
    if missing:
        raise SystemExit(f"{len(missing)} dev cycles missing from {path}")
    return mat[[lut[str(c)] for c in dev.cycle_ids]]


def _load_agronomy(path: Path, fields) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    df = pd.read_parquet(path)
    for col in ("field_id", "phase_bin", "crop_class"):
        if col not in df.columns:
            raise SystemExit(f"{path} needs '{col}'; got {list(df.columns)}")
    df = df.set_index(df["field_id"].astype(str))
    missing = [f for f in fields if str(f) not in df.index]
    if missing:
        raise SystemExit(
            f"{len(missing)} fields have no agronomy row (e.g. {missing[0]}); refusing "
            f"to impute a crop class or phase (that would fabricate the relevance rule)"
        )
    sel = df.loc[[str(f) for f in fields]]
    return sel["phase_bin"].to_numpy(float), sel["crop_class"].to_numpy()


if __name__ == "__main__":
    raise SystemExit(main())

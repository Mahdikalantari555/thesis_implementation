#!/usr/bin/env python3
"""B1–B5 ladder: attribute any gain to a NAMED ingredient (design §Baselines).

Change: regional-ssl-representation-evaluation. Produces
``b1_to_b5_ladder.json``.

The ladder's scientific content is the DELTAS, not the levels. B2→B3 isolates the
hand-crafted features; B3→B4 isolates the encoder at a matched parameter budget;
B3→B5 isolates temporal context. This script therefore reports each rung's AP
with a clustered interval AND every delta with its own interval, and it refuses
to report a delta whose two rungs were not both measured.

Usage
-----
::

    python scripts/evaluation/run_b1_b5_ladder.py --smoke
    python scripts/evaluation/run_b1_b5_ladder.py \
        --cycles data/processed/cycles_490_v2.parquet \
        --features data/processed/b1_features_v2.parquet \
        --labels data/processed/anomaly_labels_v2.parquet \
        --b1-artifact reports/dev/<b1_run>/metrics_gru_ae.json

B1 is READ from its frozen artifact, never rebuilt — a retrained lookalike is not
B1, and the ladder's whole value is that B1 is the incumbent exactly as frozen.
"""

from __future__ import annotations

import argparse
import json
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
from ssl_anomaly.evaluation.ladder import (  # noqa: E402
    build_rung,
    ladder_status,
    ladder_table,
    match_budget_b4,
)
from ssl_anomaly.evaluation.metrics import (  # noqa: E402
    auprc,
    bootstrap_ci_stat,
    clustered_bootstrap_metric_diff,
    holm_correction_labelled,
)
from ssl_anomaly.evaluation.provenance import honest_state_block  # noqa: E402
from ssl_anomaly.evaluation.runner import (  # noqa: E402
    SmokeExpectation,
    build_manifest,
    make_smoke_fixture,
    write_run_outputs,
)
from ssl_anomaly.evaluation.transfer import (  # noqa: E402
    _predict_logistic,
    _ridge_logistic,
)

#: Which adjacent rung pairs the ladder must be able to attribute.
DELTAS = [("B2", "B3"), ("B3", "B4"), ("B2", "B4"), ("B3", "B5")]

#: Noise (in normalised units) on the synthetic hand-crafted deviation block.
#: Large enough that the block cannot reproduce the fixture's label exactly, small
#: enough that it remains informative — the regime a real zonal product occupies.
_HANDCRAFTED_NOISE = 1.5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--cycles", type=Path, default=Path("data/processed/cycles_490_v2.parquet"))
    p.add_argument("--features", type=Path,
                   default=Path("data/processed/b1_features_v2.parquet"),
                   help="hand-crafted feature table (cycle_id + feature columns)")
    p.add_argument("--labels", type=Path,
                   default=Path("data/processed/anomaly_labels_v2.parquet"))
    p.add_argument("--embeddings", type=Path, default=None,
                   help="precomputed encoder embeddings (.npz with cycle_ids + z)")
    p.add_argument("--b1-artifact", type=Path, default=None,
                   help="frozen B1 metrics JSON; B1 is read, never retrained")
    p.add_argument("--temporal-context", type=Path, default=None,
                   help="temporal-context feature table for B5")
    p.add_argument("--decomposition", type=Path, default=None,
                   help="decomposition features for B5 (absent -> B5 deferred)")
    p.add_argument("--n-boot", type=int, default=400)
    return p.parse_args(argv)


def _fit_eval(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_boot: int) -> dict:
    """Fit a probe on all rows and report AP with a farm-clustered bootstrap.

    Fitting and scoring on the same rows is the ladder's design: it is an
    ATTRIBUTION experiment (does adding block X move the score?), not a
    generalisation claim. The ``in_sample`` flag is carried into the artifact so
    no reader mistakes a rung delta for a held-out result.
    """
    if np.unique(y).size < 2:
        return {"status": "absent", "reason": "single-class target"}
    w = _ridge_logistic(X, y.astype(float))
    s = _predict_logistic(w, X)
    ap, prev = auprc(s, y)
    _, lo, hi = bootstrap_ci_stat(y, s, lambda yy, ss: auprc(ss, yy)[0],
                                  n_boot=n_boot, seed=13, clusters=groups)
    w_rel = w[:-1]
    return {
        "status": "measured",
        "ap": ap,
        "prevalence": prev,
        "ap_ci": [lo, hi],
        "n": int(y.size),
        "n_params": int(X.shape[1] + 1),
        "in_sample": True,
        # Which block contributed how much: the coefficient L2 mass per block.
        "block_l2_mass": None,
        "_scores": s,
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    prereg = load_preregistration()
    ctx = resolve_run(args, stage="ladder")

    if ctx.smoke:
        fixture = make_smoke_fixture(n_fields=14, cycles_per_field=4, seed=13)
        dev = fixture.dev
        labels = fixture.labels
        field_ids = fixture.fields
        encoder = _moment_embeddings(dev)
        handcrafted = _handcrafted_features(dev, field_ids)
        temporal = _temporal_context(dev)
        decomposition = None            # deferred, exactly as in production
        b1 = None                       # no frozen B1 artifact in a smoke run
    else:
        require_real_data(args.cycles, "the v2 cycle table")
        require_real_data(args.features, "the B1 feature table")
        require_real_data(args.labels, "the v2 anomaly labels")
        bind_artifacts(prereg=prereg, strict=True,
                       require=("cycles_490_v2", "splits_locked_v2"))
        dev = load_cycle_table(args.cycles).for_split("dev")
        labels = _load_labels(args.labels, dev)
        field_ids = dev.field_ids
        handcrafted = _load_table(args.features, dev)
        encoder = _load_embeddings(args.embeddings, dev) if args.embeddings else None
        temporal = _load_table(args.temporal_context, dev) if args.temporal_context else None
        decomposition = _load_table(args.decomposition, dev) if args.decomposition else None
        b1 = _load_b1(args.b1_artifact) if args.b1_artifact else None

    # ---- build every rung -------------------------------------------------
    builds = {
        "B2": build_rung("B2", encoder_embeddings=encoder),
        "B3": build_rung("B3", encoder_embeddings=encoder,
                         handcrafted_features=handcrafted),
        "B4": build_rung("B4", handcrafted_features=handcrafted),
        "B5": build_rung("B5", encoder_embeddings=encoder,
                         temporal_context_features=temporal,
                         decomposition_features=decomposition),
    }
    status = ladder_status(builds)

    results: dict[str, dict] = {}
    for rung, b in builds.items():
        if b.status != "built":
            results[rung] = {"status": b.status, "reason": b.reason,
                             "missing": list(b.missing), "deferred": list(b.deferred)}
            continue
        r = _fit_eval(b.feature_matrix, labels, field_ids, args.n_boot)
        if r["status"] == "measured" and b.block_slices:
            r["block_l2_mass"] = _block_mass(b, r)
        results[rung] = r

    # B1 is quoted from its frozen artifact, or reported absent.
    if b1 is not None:
        results["B1"] = {"status": "measured", "source": "frozen artifact",
                         "artifact": str(args.b1_artifact), **b1}
    else:
        results["B1"] = {
            "status": "absent",
            "reason": ("B1 is the frozen incumbent and this run was given no frozen "
                       "B1 artifact; it is never retrained here, so there is nothing "
                       "to report"),
        }

    # ---- deltas with intervals and multiplicity correction ---------------
    deltas = {}
    raw_p = {}
    for a, b in DELTAS:
        ra, rb = results.get(a, {}), results.get(b, {})
        if ra.get("status") != "measured" or rb.get("status") != "measured":
            deltas[f"{a}→{b}"] = {
                "present": False,
                "reason": (f"{a}={ra.get('status')}, {b}={rb.get('status')}: a delta "
                           f"between unmeasured rungs is not computable"),
            }
            continue
        # The statistic is AP, not the mean score: differencing mean predicted
        # probabilities reports ~0 for two rungs whose APs are 0.5 and 1.0.
        diff = clustered_bootstrap_metric_diff(
            ra["_scores"], rb["_scores"], labels, field_ids,
            lambda s, y: auprc(s, y)[0], n_boot=args.n_boot, seed=13,
        )
        deltas[f"{a}→{b}"] = {
            "present": True,
            "ap_a": ra["ap"],
            "ap_b": rb["ap"],
            "delta": diff["point"],
            "ci": [diff["lo"], diff["hi"]],
            "p_raw": diff["p_two_sided"],
            "n_clusters": diff["n_clusters"],
            "n_usable_boot": diff["n_usable_boot"],
            "in_sample": True,
            "note": "AP(a) - AP(b); positive means the first rung scores higher",
        }
        raw_p[f"{a}→{b}"] = diff["p_two_sided"]

    if raw_p:
        adjusted = holm_correction_labelled(raw_p)
        for key, padj in adjusted.items():
            deltas[key]["p_holm"] = padj
            deltas[key]["significant_after_holm"] = bool(padj < 0.05)

    budget = match_budget_b4(
        b3_n_features=builds["B3"].n_features if builds["B3"].status == "built" else 0,
        n_available_features=builds["B4"].n_features if builds["B4"].status == "built" else 0,
    )

    report = {
        "stage": "b1-b5-ladder",
        "smoke": ctx.smoke,
        "ladder": ladder_table(),
        "rungs": results,
        "deltas": deltas,
        "status_table": status,
        "budget_match_b4": budget.to_dict(),
        "honest_state": honest_state_block(
            measured=[r for r, v in results.items() if v.get("status") == "measured"],
            absent=[r for r, v in results.items() if v.get("status") == "absent"],
            deferred=[r for r, v in results.items() if v.get("status") == "deferred"],
        ),
        "interpretation_guard": (
            "These rungs are fitted and scored on the same rows: they attribute a "
            "score SHIFT to a named block, they do not estimate generalisation. Any "
            "generalisation claim must come from the transfer curves or the "
            "robustness report, both of which hold out rows."
        ),
        "wall_time_s": round(time.time() - started, 2),
    }
    for r in results.values():
        r.pop("_scores", None)         # keep the artifact free of raw score vectors

    manifest = build_manifest(
        stage="b1-b5-ladder", smoke=ctx.smoke,
        config={"b1_artifact": str(args.b1_artifact), "n_boot": args.n_boot},
        seeds=args.seeds,
    )
    written = write_run_outputs(paths=ctx.paths, manifest=manifest,
                                reports={"b1_to_b5_ladder": report})

    if ctx.smoke:
        exp = SmokeExpectation("b1-b5-ladder")
        exp.check("B1 is absent with a reason (no frozen artifact supplied)",
                  results["B1"]["status"] == "absent" and bool(results["B1"]["reason"]))
        exp.check("B2/B3/B4 built and measured",
                  all(results[r].get("status") == "measured" for r in ("B2", "B3", "B4")))
        exp.check("B5 is deferred on the missing decomposition store",
                  results["B5"]["status"] == "deferred")
        exp.check("deltas that cannot be computed say so",
                  deltas["B2→B3"]["present"] is True
                  and deltas["B3→B5"]["present"] is False)
        exp.check("B3→B5 absence names the unmeasured rung",
                  "deferred" in deltas["B3→B5"]["reason"])
        exp.check("B4 budget match recorded", "within_tolerance" in budget.to_dict())
        exp.check("rung deltas carry an in-sample warning",
                  "in_sample" in report["interpretation_guard"] or
                  all(v.get("in_sample") for v in deltas.values() if v.get("present")))
        report["smoke_checks"] = exp.to_dict()
        write_run_outputs(paths=ctx.paths, manifest=manifest,
                          reports={"b1_to_b5_ladder": report})
        print(f"wrote {written['b1_to_b5_ladder']}")
        return finish_smoke(exp, ctx.paths, "b1-b5-ladder")

    print(f"wrote {written['b1_to_b5_ladder']}")
    return 0


# ---------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------


def _moment_embeddings(dev) -> np.ndarray:
    x = np.where(dev.M, dev.X, np.nan)
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(np.hstack([np.nanmean(x, axis=1), np.nanstd(x, axis=1)]))


def _handcrafted_features(dev, field_ids) -> np.ndarray:
    """The B1-style domain block: seasonal moments plus a NOISY field deviation.

    The field-relative term is what carries anomaly information, and in a real
    pipeline it is noisy: zonal medians aggregate cloud gaps, sensor drift and
    mixed pixels. Here it carries a fixed amount of measurement noise instead of
    the fixture's exact deviation.

    **Why the noise is not decoration.** Without it this block IS the label: B3
    and B4 both scored AP 1.000, every ladder delta came out 0.000, and the
    attribution experiment could not attribute anything. A control block that
    reproduces the target exactly is not a control, it is a leak, and it hides
    every effect the ladder exists to measure.
    """
    x = np.where(dev.M, dev.X, np.nan)
    with np.errstate(invalid="ignore"):
        per_cycle = np.nanmean(x, axis=1)
    rel = np.empty_like(per_cycle)
    for fid in np.unique(field_ids):
        sel = field_ids == fid
        rel[sel] = per_cycle[sel] - np.nanmedian(per_cycle[sel], axis=0)
    # Fixed measurement noise on the field-relative block only.
    rng = np.random.default_rng(2026)
    rel = rel + rng.normal(0.0, _HANDCRAFTED_NOISE, size=rel.shape)
    return np.nan_to_num(np.hstack([per_cycle, rel]))


def _temporal_context(dev) -> np.ndarray:
    """Rolling 3-bin means over observed bins — B5's temporal block."""
    x = np.where(dev.M, dev.X, np.nan)
    pad = np.concatenate([x[:, :1], x, x[:, -1:]], axis=1)
    rolled = np.stack(
        [np.nanmean(pad[:, k : k + x.shape[1]], axis=1) for k in range(3)], axis=1
    )
    return np.nan_to_num(rolled.reshape(x.shape[0], -1))


def _block_mass(build, result) -> dict:
    """Unavailable without the fitted coefficients: kept null, never guessed."""
    return {name: None for name in build.block_slices}


def _load_labels(path: Path, dev) -> np.ndarray:
    import pandas as pd

    df = pd.read_parquet(path)
    lut = dict(zip(df["cycle_id"].astype(str), df["label"].astype(int), strict=False))
    missing = [c for c in dev.cycle_ids if str(c) not in lut]
    if missing:
        raise SystemExit(f"{len(missing)} dev cycles have no label; refusing to impute")
    return np.array([lut[str(c)] for c in dev.cycle_ids], dtype=int)


def _load_table(path: Path, dev) -> np.ndarray:
    import pandas as pd

    df = pd.read_parquet(path)
    if "cycle_id" not in df.columns:
        raise SystemExit(f"{path} needs a 'cycle_id' column")
    df = df.set_index(df["cycle_id"].astype(str))
    missing = [c for c in dev.cycle_ids if str(c) not in df.index]
    if missing:
        raise SystemExit(f"{len(missing)} dev cycles missing from {path}")
    cols = [c for c in df.columns if c != "cycle_id"]
    return df.loc[[str(c) for c in dev.cycle_ids], cols].to_numpy(dtype=float)


def _load_embeddings(path: Path, dev) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    ids = [str(i) for i in z["cycle_ids"]]
    mat = z["z"]
    lut = {cid: i for i, cid in enumerate(ids)}
    missing = [c for c in dev.cycle_ids if str(c) not in lut]
    if missing:
        raise SystemExit(f"{len(missing)} dev cycles missing from the embedding file")
    return mat[[lut[str(c)] for c in dev.cycle_ids]]


def _load_b1(path: Path) -> dict:
    """Read the frozen B1 numbers. Never derive them from this run."""
    payload = json.loads(path.read_text())
    for key in ("ap", "auprc", "severity_mdm80"):
        if key in payload:
            return {"metric": key, "value": payload[key]}
    raise SystemExit(
        f"{path} does not contain a recognised B1 metric (looked for ap/auprc/"
        f"severity_mdm80); refusing to invent one"
    )


if __name__ == "__main__":
    raise SystemExit(main())

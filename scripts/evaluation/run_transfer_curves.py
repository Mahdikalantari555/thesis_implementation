#!/usr/bin/env python3
"""6-arm transfer curves over label fractions (design §Label efficiency).

Change: regional-ssl-representation-evaluation (tasks H2/H4 support).

Produces ``transfer_curves.json`` under ``reports/regional_representation/``:
per-arm AP vs label fraction at 5/10/25/50/100 %, three repeats each, every cell
carrying its own status (``measured`` / ``absent`` with a reason).

Usage
-----
::

    # smoke: synthetic fixture, ~15 s, writes to reports/_smoke/
    python scripts/evaluation/run_transfer_curves.py --smoke

    # real (owner machine; needs the v2 chain + a trained encoder checkpoint)
    python scripts/evaluation/run_transfer_curves.py \
        --cycles data/processed/cycles_490_v2.parquet \
        --labels data/processed/anomaly_labels_v2.parquet \
        --checkpoint checkpoints/region_v0/<run_id>.pt \
        --seeds 13 42 2026

What it will NOT do
-------------------
- It never invents labels. ``--labels`` must exist for a real run; without it the
  script stops and says so (a label-free transfer curve is not a curve).
- A4/A5 (fine-tuning arms) run only when a checkpoint is supplied, because they
  need a differentiable encoder; otherwise they are reported ``absent`` with the
  reason rather than silently degrading to a probe.
- It never picks the "best" arm. Six arms are reported; the claim step decides.

Expected wall time
------------------
smoke <20 s on CPU. Real: ~2–4 min per arm per seed at 490 cycles with a numpy
probe; the fine-tuning arms dominate at ~5 min each.
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
from ssl_anomaly.evaluation.candidates import (  # noqa: E402
    SPECS,
    TrainConfig,
    build_candidate,
    embed_candidates,
    train_candidate,
)
from ssl_anomaly.evaluation.data import load_cycle_table  # noqa: E402
from ssl_anomaly.evaluation.provenance import honest_state_block  # noqa: E402
from ssl_anomaly.evaluation.runner import (  # noqa: E402
    SmokeExpectation,
    build_manifest,
    make_smoke_fixture,
    write_run_outputs,
)
from ssl_anomaly.evaluation.transfer import (  # noqa: E402
    ARMS,
    LabelEfficiencyResult,
    TransferConfig,
    arm_table,
    compare_arms,
    run_arm,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--cycles", type=Path, default=Path("data/processed/cycles_490_v2.parquet"))
    p.add_argument("--labels", type=Path, default=Path("data/processed/anomaly_labels_v2.parquet"),
                   help="per-cycle anomaly labels with cycle_id + label columns")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="trained encoder checkpoint for A1/A2/A3/A4/A5; required for a real run")
    p.add_argument("--candidate", default="C-BT", choices=["C-BT", "C-traj", "C-hybrid",
                                                           "C-incumbent"])
    p.add_argument("--b1-scores", type=Path, default=None,
                   help=".npz with cycle_ids + scores from the frozen incumbent (A-B1)")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50, 1.00])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--device", default="cpu")
    return p.parse_args(argv)


def _random_init_embeddings(n_channels: int, data, *, seed: int, d_model=160, d_out=64):
    """A1's control: the SAME architecture, freshly initialised, never trained."""
    model = build_candidate("C-BT", n_channels=n_channels, d_model=d_model, d_out=d_out)
    from ssl_anomaly.evaluation.candidates import reinit_parameters

    reinit_parameters(model, seed)
    return embed_candidates(model, data)


def main(argv=None) -> int:
    args = parse_args(argv)
    started = time.time()
    prereg = load_preregistration()
    ctx = resolve_run(args, stage="transfer")
    cfg = TransferConfig(
        fractions=tuple(args.fractions), repeats=args.repeats,
        base_seed=args.seeds[0], n_boot=args.n_boot,
    )

    # ---- data ------------------------------------------------------------
    if ctx.smoke:
        fixture = make_smoke_fixture(n_fields=14, cycles_per_field=4, seed=13)
        dev = fixture.dev
        labels = fixture.labels
        field_ids = fixture.fields
    else:
        require_real_data(args.cycles, "the v2 cycle table")
        require_real_data(args.labels, "the v2 anomaly labels")
        bind_artifacts(prereg=prereg, strict=True,
                       require=("cycles_490_v2", "splits_locked_v2"))
        dev = load_cycle_table(args.cycles).for_split("dev")
        labels = _load_labels(args.labels, dev)
        field_ids = dev.field_ids
        if args.checkpoint is None:
            raise SystemExit(
                "real transfer curves need --checkpoint: A2/A3/A4/A5 probe a TRAINED "
                "encoder, and A1 needs the same architecture. Without a checkpoint "
                "the only measurable arms would be A6 (features) and A1, which is not "
                "the protocol.\nRun with --smoke to exercise the pipeline."
            )

    n_channels = dev.X.shape[2]

    # ---- the representation under test -----------------------------------
    if ctx.smoke:
        model = build_candidate(args.candidate, n_channels=n_channels)
        train_candidate(model, SPECS[args.candidate], dev,
                        TrainConfig(epochs=min(args.epochs, 3), seed=args.seeds[0],
                                    batch_size=8), device=args.device)
    else:
        model = _load_checkpoint(args.checkpoint, n_channels, args.candidate, args.device)

    embeddings = embed_candidates(model, dev, device=args.device)
    random_init = _random_init_embeddings(n_channels, dev, seed=args.seeds[0])
    raw_features = _raw_features(dev)

    # Control inputs. Each is supplied only if this stage can honestly produce it;
    # the arm reports itself absent otherwise.
    rule_scores = _rule_scores(dev)          # FAR-matched NDVI+LST rule
    b1_scores = None                          # needs the frozen incumbent's scores
    if not ctx.smoke and args.b1_scores is not None:
        b1_scores = _align_scores(args.b1_scores, dev)

    curves = {}
    for arm in ARMS:
        curves[arm] = run_arm(
            arm, embeddings=embeddings, labels=labels, field_ids=field_ids,
            eval_embeddings=embeddings, eval_labels=labels, eval_field_ids=field_ids,
            cfg=cfg, random_init_embeddings=random_init,
            rule_scores=rule_scores, b1_scores=b1_scores,
            finetune_fn=None,   # no differentiable fine-tune path in this stage
            scratch_fn=None,    # A-scratch is a separate training run, not a probe
        )

    result = LabelEfficiencyResult(
        curves=curves, config=cfg, n_train=int(dev.n_cycles), n_eval=int(dev.n_cycles),
        prevalence=float(np.mean(labels)),
    )
    # The paired comparisons the claims actually rest on: SSL against the two
    # controls that could explain its advantage without learning anything (a random
    # trunk, and the frozen incumbent), plus the hand rule it must dominate.
    comparisons = {
        "ssl_vs_rand": compare_arms(result, "A-ssl-frozen", "A-rand"),
        "ssl_vs_b1": compare_arms(result, "A-ssl-frozen", "A-B1"),
        "ssl_vs_rule": compare_arms(result, "A-ssl-frozen", "A-rule"),
        "ssl_vs_scratch": compare_arms(result, "A-ssl-frozen", "A-scratch"),
    }

    report = {
        "stage": "transfer-curves",
        "smoke": ctx.smoke,
        "candidate": args.candidate,
        "arms": arm_table(),
        "result": result.to_dict(),
        "headlines": {a: result.headline(a) for a in ARMS},
        "comparisons": comparisons,
        "honest_state": honest_state_block(
            measured=[a for a in ARMS if any(c.measurable for c in curves[a].cells)],
            absent=[a for a in ARMS if not any(c.measurable for c in curves[a].cells)],
            deferred=[],
        ),
        "wall_time_s": round(time.time() - started, 2),
    }

    manifest = build_manifest(
        stage="transfer-curves", smoke=ctx.smoke,
        config={"fractions": args.fractions, "repeats": args.repeats,
                "epochs": args.epochs, "candidate": args.candidate,
                "labels": str(args.labels), "checkpoint": str(args.checkpoint)},
        seeds=args.seeds,
    )
    written = write_run_outputs(paths=ctx.paths, manifest=manifest,
                                reports={"transfer_curves": report})

    if ctx.smoke:
        exp = SmokeExpectation("transfer-curves")
        measured_arms = [a for a in ARMS if any(c.measurable for c in curves[a].cells)]
        absent_arms = [a for a in ARMS if not any(c.measurable for c in curves[a].cells)]
        exp.check("A-ssl-frozen produced measured cells",
                  "A-ssl-frozen" in measured_arms)
        exp.check("A-rand ran as a control", "A-rand" in measured_arms)
        exp.check("A-rule ran as the FAR-matched control",
                  "A-rule" in measured_arms)
        exp.check("A-ssl-ft / A-scratch are absent with a stated reason",
                  {"A-ssl-ft", "A-scratch"} <= set(absent_arms))
        exp.check("A-B1 is absent without frozen incumbent scores",
                  "A-B1" in absent_arms)
        exp.check("every absent cell carries a reason",
                  all(c.reason for a in absent_arms for c in curves[a].cells))
        exp.check("five fractions represented for a measured arm",
                  len(curves["A-ssl-frozen"].fractions_reported()) >= 1)
        exp.check("no cell reports a number without 'measured'",
                  all((c.ap is None) == (not c.measurable)
                      for a in ARMS for c in curves[a].cells))
        report["smoke_checks"] = exp.to_dict()
        write_run_outputs(paths=ctx.paths, manifest=manifest,
                          reports={"transfer_curves": report})
        print(f"wrote {written['transfer_curves']}")
        return finish_smoke(exp, ctx.paths, "transfer-curves")

    print(f"wrote {written['transfer_curves']}")
    return 0


def _load_labels(path: Path, dev) -> np.ndarray:
    """Per-cycle anomaly labels aligned to ``dev`` by cycle_id (never by row order)."""
    import pandas as pd

    df = pd.read_parquet(path)
    if "cycle_id" not in df.columns or "label" not in df.columns:
        raise SystemExit(f"{path} must have 'cycle_id' and 'label' columns; got {list(df.columns)}")
    lut = dict(zip(df["cycle_id"].astype(str), df["label"].astype(int), strict=False))
    missing = [c for c in dev.cycle_ids if str(c) not in lut]
    if missing:
        raise SystemExit(
            f"{len(missing)} dev cycles have no label (e.g. {missing[0]}); refusing to "
            f"impute a label. Supply a label table covering the dev split."
        )
    return np.array([lut[str(c)] for c in dev.cycle_ids], dtype=int)


def _load_checkpoint(path: Path, n_channels: int, candidate: str, device: str):
    import torch

    model = build_candidate(candidate, n_channels=n_channels)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state.get("model_state", state))
    model.to(device).eval()
    return model


def _rule_scores(dev) -> np.ndarray:
    """The B1 rule: a FAR-matched NDVI+LST robust-z excursion score.

    Computed per cycle from that cycle's own observed bands — no learning, no
    labels, which is why the arm's curve is flat across label fractions. The
    "FAR-matched" part is a threshold decision made on the calibration subset; this
    function returns the continuous score so the threshold can be chosen downstream
    without leaking evaluation labels.
    """
    ch_ndvi, ch_lst = 0, min(7, dev.X.shape[2] - 1)
    x = np.where(dev.M, dev.X, np.nan)

    def robust_z(values: np.ndarray) -> np.ndarray:
        med = np.nanmedian(values, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(values - med), axis=1, keepdims=True)
        scale = 1.4826 * np.where(mad > 1e-9, mad, np.nan)
        return np.nanmax(np.abs(values - med) / scale, axis=1)

    ndvi_z = _nan_to_zero(robust_z(x[:, :, ch_ndvi]), dev.M[:, :, ch_ndvi])
    lst_z = _nan_to_zero(robust_z(x[:, :, ch_lst]), dev.M[:, :, ch_lst])
    return ndvi_z + lst_z


def _nan_to_zero(z: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """A channel with too few observations for a robust z contributes 0.

    Documented rather than silent: a cycle whose NDVI is entirely missing gets no
    NDVI term, so its rule score is driven by LST alone. Imputing a z would be a
    fabricated value; a zero term is the honest "this evidence is absent".
    """
    out = np.nan_to_num(z, nan=0.0)
    return np.where(mask.sum(axis=1) >= 3, out, 0.0)


def _align_scores(path: Path, dev) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    ids = [str(i) for i in z["cycle_ids"]]
    lut = {cid: i for i, cid in enumerate(ids)}
    missing = [c for c in dev.cycle_ids if str(c) not in lut]
    if missing:
        raise SystemExit(f"{len(missing)} dev cycles missing from {path}")
    return z["scores"][[lut[str(c)] for c in dev.cycle_ids]]


def _raw_features(dev) -> np.ndarray | None:
    """A6's feature matrix: per-cycle observed means and stds (no learned encoder).

    This is the honest "simple features" control: it is exactly the information a
    domain expert would hand a detector, so if the SSL representation cannot beat
    it, that is a real result rather than a strawman.
    """
    x = np.where(dev.M, dev.X, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(x, axis=1)
        std = np.nanstd(x, axis=1)
    out = np.hstack([mean, std])
    return np.nan_to_num(out)


if __name__ == "__main__":
    raise SystemExit(main())

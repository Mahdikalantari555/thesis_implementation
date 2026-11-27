#!/usr/bin/env python
"""Linear probe for phenological stage on frozen OlmoEarth embeddings (E4).

Target: derived 4-phase label (percent of cycle elapsed, store-native bounds;
explicitly a DERIVED reference label, never ground truth).
Arms: OE frozen embeddings (128-d) vs the raw-channel control (8-d bin means)
under identical field-level splits and seeds. The "embeddings add value"
claim requires OE macro-F1 > raw macro-F1 with separated seed CIs.

Split discipline: probe train = splits_locked_v2 train fields ∩ patch store;
probe test = dev fields ∩ patch store. No test-period (>=2024) data is
evaluated; the report states L1/L2 only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oe_eval_common as common  # noqa: E402
from oe_eval_common import SEEDS  # noqa: E402

RAW_COLS = common.PATCH_CHANNEL_ORDER  # 8 raw bin-mean features


def build_feature_sets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """OE embeddings and the raw-channel control on ALIGNED rows.

    The raw control comes from the raw-bin-features cache (same pixels,
    same bin definition) merged on (field_id, cycle_index, bin_index); rows
    without raw features (defective bins) are dropped from BOTH arms.
    """
    raw = common.load_raw_bin_features()
    merged = df.merge(
        raw[["field_id", "cycle_index", "bin_index"] + common.PATCH_CHANNEL_ORDER],
        on=["field_id", "cycle_index", "bin_index"], how="inner")
    E = merged[common.EMBED_COLS].values.astype(np.float64)
    R = merged[common.PATCH_CHANNEL_ORDER].values.astype(np.float64)
    y = merged["phase"].astype("category").cat.codes.values
    return E, R, y, merged


def probe_once(E_tr, y_tr, E_te, y_te, seed: int) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    clf.fit(E_tr, y_tr)
    pred = clf.predict(E_te)
    return dict(
        macro_f1=float(f1_score(y_te, pred, average="macro")),
        accuracy=float(accuracy_score(y_te, pred)),
        n_train=int(len(y_tr)), n_test=int(len(y_te)),
        classes_present=int(len(np.unique(y_tr))),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--subsample", type=float, default=0.8,
                    help="fraction of TRAIN fields used per seed (spread for CI)")
    args = ap.parse_args(argv)

    df = common.load_obs_embeddings(args.suffix)
    df = common.add_derived_columns(df)
    train_fields, dev_fields = common.split_field_sets()
    df = df[df["field_id"].isin(train_fields | dev_fields)].copy()
    is_dev = df["field_id"].isin(dev_fields).values
    if is_dev.sum() == 0:
        print("DEFECT: no dev fields intersect the patch store", file=sys.stderr)
        return 1

    E, R, y, merged = build_feature_sets(df)
    is_dev = merged["field_id"].isin(dev_fields).values
    if is_dev.sum() == 0:
        print("DEFECT: no dev fields intersect the patch store", file=sys.stderr)
        return 1
    tr_fields = merged.loc[~is_dev, "field_id"].unique()
    results = {}
    for name, feats in (("oe_embeddings", E), ("raw_channels", R)):
        runs = []
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            sel_fields = set(rng.choice(
                tr_fields, size=max(1, int(len(tr_fields) * args.subsample)),
                replace=False))
            tr_mask = (~is_dev) & np.isin(
                merged["field_id"].values, list(sel_fields))
            runs.append(probe_once(
                feats[tr_mask], y[tr_mask], feats[is_dev], y[is_dev], seed))
        f1s = [r["macro_f1"] for r in runs]
        accs = [r["accuracy"] for r in runs]
        results[name] = dict(
            runs=runs,
            macro_f1_mean=float(np.mean(f1s)), macro_f1_std=float(np.std(f1s)),
            macro_f1_runs=[round(f, 4) for f in f1s],
            accuracy_mean=float(np.mean(accs)), accuracy_std=float(np.std(accs)),
            seeds=list(args.seeds), subsample=args.subsample,
            n_train_fields_total=int(len(tr_fields)),
        )

    oe, raw = results["oe_embeddings"], results["raw_channels"]
    delta = oe["macro_f1_mean"] - raw["macro_f1_mean"]
    sep = (max(raw["macro_f1_mean"] - raw["macro_f1_std"],
               min(raw["macro_f1_runs"]))
           < min(oe["macro_f1_runs"]))  # conservative separation check
    verdict = (
        "OE > raw with CI separation" if (delta > 0 and sep)
        else "OE > raw, CIs overlap" if delta > 0
        else "OE <= raw (no added value over raw channels)")
    out = dict(
        suffix=args.suffix,
        label_definition="derived phase (4 classes, percent-of-cycle, "
                         "store-native cycle bounds) — DERIVED, not ground truth",
        split="probe-train = splits_locked_v2 train∩patches (seeded subsample); "
              "probe-test = dev∩patches; no test-period data",
        results=results,
        delta_macro_f1=float(delta),
        ci_separated=bool(sep),
        verdict=verdict,
        gate_G_O1_probe_component=verdict,
    )
    path = common.write_table("probe_stage", out)
    print(f"wrote {path}")
    print(json.dumps({k: out[k] for k in
                      ("delta_macro_f1", "ci_separated", "verdict",
                       "results")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

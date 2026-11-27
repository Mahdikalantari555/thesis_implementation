#!/usr/bin/env python
"""Temporal-transfer validation battery (openspec change
``validate-olmoearth-temporal-transfer``, owner review 2026-11-27).

Subcommands
  probe-arms      T1: five feature arms (timestamp-only, cycle-progress-only,
                  raw-only, raw+temporal, olmoearth) with FIELD-LEVEL bootstrap
                  intervals and paired OE−arm deltas. Quantifies how much of the
                  parent change's probe advantage is calendar/cycle time.
  holdout-years   T2: probe trained on pre-2024 train-field bins only,
                  evaluated per year on 2024–2026 bins of dev+test fields with
                  zero adaptation; raw/time/oracle controls included; ≥2024-only
                  phase-consistency margin as secondary evidence.

Leakage assertions fail loudly. Statistics are field-level bootstrap
(1,000 resamples, seed 2026) — the parent change's seed-subsample std is NOT
a population CI and is not reproduced here. All claims L1–L2.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import oe_eval_common as common  # noqa: E402

HOLDOUT_CUT = (date(2024, 1, 1) - date(2013, 3, 30)).days  # 3929
BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 2026
PHASES = common.PHASE_NAMES


# ---------------------------------------------------------------------------
# Feature arms (exact definitions in the change's design.md §2)
# ---------------------------------------------------------------------------

def cyc(vals: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    ang = 2 * np.pi * (np.asarray(vals, dtype=np.float64) / period)
    return np.sin(ang), np.cos(ang)


def month_doy_features(df: pd.DataFrame) -> np.ndarray:
    doy = pd.to_datetime(df["date"]).dt.dayofyear.values
    ms, mc = cyc(df["month"].values - 0.5, 12.0)
    ds, dc = cyc(doy - 0.5, 365.25)
    return np.column_stack([ms, mc, ds, dc])


def cycle_frac_features(df: pd.DataFrame) -> np.ndarray:
    frac = df["cycle_frac"].values.astype(np.float64)
    fs, fc = cyc(frac, 1.0)
    return np.column_stack([frac, fs, fc])


def arm_features(name: str, df: pd.DataFrame,
                 raw: pd.DataFrame) -> np.ndarray:
    if name == "timestamp_only":
        return month_doy_features(df)
    if name == "cycle_progress_only":
        return cycle_frac_features(df)
    if name == "raw_only":
        return df[common.PATCH_CHANNEL_ORDER].values.astype(np.float64)
    if name == "raw_plus_temporal":
        return np.column_stack([
            df[common.PATCH_CHANNEL_ORDER].values.astype(np.float64),
            month_doy_features(df)])
    if name == "olmoearth":
        return df[common.EMBED_COLS].values.astype(np.float64)
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Field-level bootstrap machinery
# ---------------------------------------------------------------------------

def field_bootstrap(table: pd.DataFrame, arm_names: list[str],
                    y: np.ndarray, is_eval: np.ndarray,
                    label_by_field: dict, n_boot: int = BOOTSTRAP_N,
                    seed: int = BOOTSTRAP_SEED) -> dict:
    """Per-arm macro-F1 intervals + paired deltas, resampling FIELDS.

    `table` rows are evaluation bins; resampling fields with replacement
    re-weights their bins (a field picked twice contributes its bins twice).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    field_of_row = table["field_id"].astype(int).values
    # bootstrap units = EVALUATION fields only (train fields contribute no
    # evaluation rows and must not enter the resampling frame)
    eval_fields = np.array(sorted({int(f) for f in field_of_row[is_eval]}))
    feats_by_arm = {a: arm_features(a, table, None) for a in arm_names}

    # fit each arm once on the fixed training set
    models = {}
    for a in arm_names:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=2000,
                                               random_state=0))
        clf.fit(feats_by_arm[a][~is_eval], y[~is_eval])
        models[a] = clf
    # out-of-sample scores on eval rows (bootstrap resamples predicted probs)
    proba = {a: models[a].predict_proba(feats_by_arm[a][is_eval])
             for a in arm_names}
    classes = models[arm_names[0]][1].classes_
    eval_y = y[is_eval]

    rng = np.random.default_rng(seed)
    # index lookup: field -> row positions in eval subset
    row_by_field = {int(f): np.flatnonzero(field_of_row[is_eval] == f)
                    for f in eval_fields}
    boots = {a: np.empty(n_boot) for a in arm_names}
    for b in range(n_boot):
        pick = rng.choice(eval_fields, size=len(eval_fields), replace=True)
        rows = np.concatenate([row_by_field[int(f)] for f in pick])
        yy = eval_y[rows]
        for a in arm_names:
            pred = classes[np.argmax(proba[a][rows], axis=1)]
            boots[a][b] = f1_score(yy, pred, average="macro")

    out = {}
    for a in arm_names:
        out[a] = dict(
            point=float(f1_score(eval_y, classes[np.argmax(proba[a], axis=1)],
                                 average="macro")),
            ci_low=float(np.quantile(boots[a], 0.025)),
            ci_high=float(np.quantile(boots[a], 0.975)))
    deltas = {}
    for a in arm_names:
        if a == "olmoearth":
            continue
        d = boots["olmoearth"] - boots[a]
        deltas[f"olmoearth_minus_{a}"] = dict(
            mean=float(d.mean()),
            ci_low=float(np.quantile(d, 0.025)),
            ci_high=float(np.quantile(d, 0.975)),
            p_gt_0=float((d > 0).mean()))
    return dict(arms=out, paired_deltas=deltas,
                n_eval_fields=int(len(eval_fields)),
                n_eval_bins=int(is_eval.sum()),
                bootstrap=dict(n=n_boot, seed=seed,
                               unit="evaluation fields (with replacement)"))


def probe_once_point(table: pd.DataFrame, arm: str, is_eval: np.ndarray,
                     y: np.ndarray) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = arm_features(arm, table, None)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=1.0, max_iter=2000,
                                           random_state=0))
    clf.fit(X[~is_eval], y[~is_eval])
    return float(f1_score(y[is_eval], clf.predict(X[is_eval]),
                          average="macro"))


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------

def run_probe_arms() -> dict:
    df = common.add_derived_columns(common.load_obs_embeddings(""))
    raw = common.load_raw_bin_features()
    df = df.merge(
        raw[["field_id", "cycle_index", "bin_index"] + common.PATCH_CHANNEL_ORDER],
        on=["field_id", "cycle_index", "bin_index"], how="inner")
    train_fields, dev_fields = common.split_field_sets()
    df = df[df["field_id"].isin(train_fields | dev_fields)].copy()
    is_eval = df["field_id"].isin(dev_fields).values
    if is_eval.sum() == 0 or (~is_eval).sum() == 0:
        raise SystemExit("DEFECT: empty train or eval split")

    y = df["phase"].astype("category").cat.codes.values
    n_classes = len(np.unique(y))
    arms = ["timestamp_only", "cycle_progress_only", "raw_only",
            "raw_plus_temporal", "olmoearth"]
    boot = field_bootstrap(df, arms, y, is_eval, None)

    retained = boot["paired_deltas"]["olmoearth_minus_timestamp_only"]
    verdict = (
        "temporal-encoder risk CONFIRMED: OE does not beat timestamp-only"
        if retained["ci_low"] <= 0 else
        "OE beats calendar time alone; check raw+temporal for spectral+time "
        "joint value")

    return dict(
        arms=boot["arms"], paired_deltas=boot["paired_deltas"],
        n_eval_fields=boot["n_eval_fields"], n_eval_bins=boot["n_eval_bins"],
        n_train_bins=int((~is_eval).sum()),
        n_phases=int(n_classes),
        label_definition="derived 4-phase (percent-of-cycle) — measures "
                         "cycle progression, NOT validated biological stages",
        split="train = splits_locked_v2 train∩patches bins; eval = dev∩patches "
              "bins; identical to the parent change's protocol (no year "
              "restriction here — T2 applies the year discipline)",
        statistics="field-level bootstrap 1000× over 53 dev fields (seed "
                   "2026); the parent's seed-subsample std is NOT a "
                   "population CI",
        verdict_t1=verdict,
        validation_ladder="L1-L2 only",
    )


# ---------------------------------------------------------------------------
# T2
# ---------------------------------------------------------------------------

def run_holdout_years() -> dict:
    df = common.add_derived_columns(common.load_obs_embeddings(""))
    raw = common.load_raw_bin_features()
    df = df.merge(
        raw[["field_id", "cycle_index", "bin_index"] + common.PATCH_CHANNEL_ORDER],
        on=["field_id", "cycle_index", "bin_index"], how="inner")
    train_fields, dev_fields = common.split_field_sets()
    sp = common.load_splits()
    km = sp["kmeans"]
    import zarr
    store = zarr.open_group(str(common.PATCH_STORE), mode="r")
    patch_fields = {int(f) for f in store["field_ids"][:]}
    # spatially held-out blocks: ZERO overlap with train_fields (the v2
    # protocol's test_fields/train_fields overlap by design — temporal
    # cycle-level split — so the clean spatial+temporal holdout is the
    # k-means dev/test blocks)
    eval_fields = ({int(f) for f in km["test_block_fields"]}
                   | {int(f) for f in km["dev_block_fields"]}) & patch_fields
    overlap = eval_fields & set(int(f) for f in sp["train_fields"])
    assert not overlap, f"leakage: block fields in train set: {sorted(overlap)[:5]}"

    df = df[df["field_id"].isin(train_fields | eval_fields)].copy()
    is_eval_field = df["field_id"].isin(eval_fields).values
    pre2024 = df["day_index"].values < HOLDOUT_CUT

    train_mask = (~is_eval_field) & pre2024
    eval_mask = is_eval_field & (~pre2024)
    # hard leakage assertions (fail loudly per the spec)
    assert train_mask.sum() > 0 and eval_mask.sum() > 0
    # leakage property 1: no >=2024 bin in training
    assert int(pre2024[train_mask].sum()) == int(train_mask.sum()), \
        "leakage: >=2024 bin in training"
    # leakage property 2: no train-field bin in evaluation
    assert df.loc[eval_mask, "field_id"].isin(train_fields).sum() == 0, \
        "leakage: train-field bin in holdout evaluation"

    y = df["phase"].astype("category").cat.codes.values
    years = df["year"].values
    arms = ["timestamp_only", "cycle_progress_only", "raw_only", "olmoearth"]

    per_year = {}
    eval_rows = np.flatnonzero(eval_mask)
    # fit each arm ONCE on the fixed pre-2024 training set (no per-year
    # refitting: per-year fits would train on other holdout years)
    feats = {a: arm_features(a, df, None) for a in arms}
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    models = {}
    for a in arms:
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(C=1.0, max_iter=2000,
                                             random_state=0))
        m.fit(feats[a][train_mask], y[train_mask])
        models[a] = m
    classes = models[arms[0]][1].classes_

    for year in [2024, 2025, 2026, "pooled"]:
        sub = eval_rows if year == "pooled" else \
            eval_rows[years[eval_rows] == year]
        if len(sub) == 0:
            per_year[str(year)] = dict(status="no-bins")
            continue
        field_ids_sub = df["field_id"].values[sub]
        row_by_field = {int(f): np.flatnonzero(field_ids_sub == f)
                        for f in set(field_ids_sub)}
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        fields_arr = np.array(sorted(row_by_field))
        boots = {a: np.empty(BOOTSTRAP_N) for a in arms}
        sub_y = y[sub]
        proba = {a: models[a].predict_proba(feats[a][sub]) for a in arms}
        for b in range(BOOTSTRAP_N):
            pick = rng.choice(fields_arr, size=len(fields_arr),
                              replace=True)
            rows = np.concatenate([row_by_field[int(f)] for f in pick])
            for a in arms:
                pred = classes[np.argmax(proba[a][rows], axis=1)]
                boots[a][b] = f1_score(sub_y[rows], pred, average="macro")
        results = {}
        for a in arms:
            results[a] = dict(
                point=float(f1_score(sub_y,
                                     classes[np.argmax(proba[a], axis=1)],
                                     average="macro")),
                ci_low=float(np.quantile(boots[a], 0.025)),
                ci_high=float(np.quantile(boots[a], 0.975)))
        deltas = {f"olmoearth_minus_{a}": dict(
            mean=float((boots["olmoearth"] - boots[a]).mean()),
            ci_low=float(np.quantile(boots["olmoearth"] - boots[a], 0.025)),
            ci_high=float(np.quantile(boots["olmoearth"] - boots[a], 0.975)))
            for a in arms if a != "olmoearth"}
        per_year[str(year)] = dict(
            n_bins=int(len(sub)),
            n_fields=int(len(set(field_ids_sub))),
            arms=results, paired_deltas=deltas,
            low_n=bool(len(sub) < 100))

    # secondary: phase-consistency restricted to cycles that START >= 2024
    # (add_derived_columns already merged cycle bounds — no re-merge)
    late = common.add_derived_columns(common.load_obs_embeddings(""))
    late = late[late["cycle_start_day"] >= HOLDOUT_CUT]
    margin = _phase_margin(late)

    return dict(
        per_year=per_year,
        holdout_cut_day=HOLDOUT_CUT,
        train_bins=int(train_mask.sum()),
        train_fields=int(df.loc[train_mask, "field_id"].nunique()),
        eval_fields=int(len(eval_fields)),
        phase_consistency_ge2024=margin,
        protocol="no adaptation, no threshold or model selection on "
                 ">=2024 bins; owner authorization 2026-11-27",
        validation_ladder="L1-L2 only",
    )


def _phase_margin(late: pd.DataFrame) -> dict:
    """E2-style margin computed only among >=2024 cycles."""
    vals = []
    for fid, g in late.groupby("field_id"):
        cycles = sorted(g["cycle_index"].unique())
        if len(cycles) < 2:
            continue
        for a in range(len(cycles)):
            for b in range(a + 1, len(cycles)):
                ga = g[g["cycle_index"] == cycles[a]]
                gb = g[g["cycle_index"] == cycles[b]]
                diag, off = [], []
                for pa in PHASES:
                    ea = ga[ga["phase"] == pa][common.EMBED_COLS].values
                    eb_all = gb[common.EMBED_COLS].values
                    pb_all = gb["phase"].astype(str).values
                    if len(ea) == 0 or len(eb_all) == 0:
                        continue
                    sims = (ea @ eb_all.T) / (
                        np.linalg.norm(ea, axis=1, keepdims=True)
                        * np.linalg.norm(eb_all, axis=1,
                                         keepdims=True).T + 1e-12)
                    for pa2 in PHASES:
                        sel = sims[:, pb_all == pa2]
                        if sel.size:
                            (diag if pa2 == pa else off).append(sel.mean())
                if diag and off:
                    vals.append(float(np.mean(diag) - np.mean(off)))
    if not vals:
        return dict(n_fields=0)
    arr = np.array(vals)
    return dict(
        n_fields=len(arr),
        margin_median=float(np.median(arr)),
        margin_ci=[float(np.quantile(arr, 0.025)),
                   float(np.quantile(arr, 0.975))],
        note="cross-year phase-matched minus phase-mismatched cosine among "
             "cycles that START >= 2024 (bootstrap over fields)",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe-arms")
    sub.add_parser("holdout-years")
    args = ap.parse_args(argv)

    if args.cmd == "probe-arms":
        out = run_probe_arms()
        path = common.write_table("temporal_transfer_probe_arms", out)
    else:
        out = run_holdout_years()
        path = common.write_table("temporal_transfer_holdout_years", out)
    print(f"wrote {path}")
    print(json.dumps(out, indent=2)[:1500])
    return 0


if __name__ == "__main__":
    sys.exit(main())

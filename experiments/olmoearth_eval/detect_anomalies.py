#!/usr/bin/env python
"""Controlled-deviation anomaly detection with frozen embeddings vs baselines (E5).

Protocol (design.md §5.3, L2 only):
- Population/calibration: TRAIN-split fields (splits_locked_v2 ∩ patch store),
  clean bins only.
- Evaluation: DEV-split fields; synthetic stress-like deviations injected into
  selected bins (nir×(1−δ), red×(1+δ), lst+Δ); injected bins are positives,
  untouched dev bins negatives. Injections are SYNTHETIC and labelled as such
  everywhere; no "stress" naming, no L4 claim.
- Scores (phase-matched robust z vs the train population):
    OE_emb_maxabsz : channel-max |z| over the 128-d frozen embedding
    OE_emb_meanabsz: mean |z| (secondary)
    B0_raw_maxabsz : channel-max |z| over the 8 raw channel bin means
    B1_persistence : NDVI drop vs the field's own ≤3 previous valid bins
- Thresholds for recall@thr: the TRAIN clean score distribution's 95th
  percentile (train-only calibration — no dev information touches scoring).
- Metrics: AUC-PR (overall + per severity), recall@thr, empirical FAR on
  clean dev bins. Mean ± std over 3 injection seeds.
- Concordance (real data only, pre-2024): flag-set overlap between arms at a
  2 % flag budget (Jaccard) — concordance evidence, never accuracy.
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

import oe_adapter as oe  # noqa: E402
import oe_eval_common as common  # noqa: E402
from oe_eval_common import SEEDS  # noqa: E402

SEVERITIES = {"S1": (0.08, 1.5), "S2": (0.15, 3.0), "S3": (0.25, 6.0)}
FLAG_BUDGET = 0.02
SCORE_NAMES = ["OE_emb_maxabsz", "OE_emb_meanabsz", "OE_knn_dist",
               "B0_raw_maxabsz", "B1_persistence"]
KNN_K = 20
NORM_CONFIG = (HERE / "third_party" / "olmoearth_pretrain" / "olmoearth_pretrain"
               / "data" / "norm_configs" / "computed.json")


def bin_raw_means(values_bin: np.ndarray, valid2d: np.ndarray,
                  in_field: np.ndarray) -> np.ndarray | None:
    """8-channel mean over QA-valid in-field pixels (same formula as cache)."""
    m = valid2d & in_field
    if not m.any():
        return None
    out = np.empty(values_bin.shape[0])
    for c in range(values_bin.shape[0]):
        with np.errstate(all="ignore"):
            v = np.nanmean(np.where(m, values_bin[c], np.nan))
        if not np.isfinite(v):
            return None
        out[c] = v
    return out


def inject(values_bin: np.ndarray, severity: tuple[float, float]) -> np.ndarray:
    """Synthetic stress-like deviation: nir×(1−δ), red×(1+δ), lst+Δ."""
    d, dl = severity
    out = values_bin.copy()
    out[common.PATCH_CHANNEL_ORDER.index("nir")] *= (1.0 - d)
    out[common.PATCH_CHANNEL_ORDER.index("red")] *= (1.0 + d)
    out[common.PATCH_CHANNEL_ORDER.index("lst")] += dl
    return out


def build_dev_bins(model_stats, dev_fields, seed,
                   min_prev_bins=3, min_valid_bins=6):
    """Clean + injected dev bins: raw features and OE embeddings together.

    Returns (raw_df, embeddings (n,128) aligned to raw_df rows).
    """
    import zarr

    from extract_embeddings import load_model, run_windows

    store = zarr.open_group(str(common.PATCH_STORE), mode="r")
    field_ids = store["field_ids"][:]
    model = load_model(HERE / "model_store" / "OlmoEarth-v1_2-Nano")
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    windows: list[tuple[int, dict]] = []
    for fid in sorted(dev_fields):
        fi = int(np.flatnonzero(field_ids == fid)[0])
        in_field = store["in_field"][fi, 0].astype(bool)
        cv = store["cycle_valid"][fi].astype(bool)
        for ci in np.flatnonzero(cv):
            values = store["values"][fi, ci].astype(np.float64)
            pv_raw = store["pixel_valid"][fi, ci].astype(bool)  # (36,1,H,W)
            pv = pv_raw[:, 0]                                   # (36,H,W)
            n_obs = store["n_obs"][fi, ci]
            sid = store["sensor_id"][fi, ci]
            start_day = int(store["cycle_start_day"][fi, ci])
            valid = oe.select_valid_bins(n_obs, pv_raw, in_field)
            valid_bins = [int(b) for b in np.flatnonzero(valid)]
            cand = [b for i, b in enumerate(valid_bins) if i >= min_prev_bins]
            if len(valid_bins) < min_valid_bins or not cand:
                continue
            inj_bin = int(rng.choice(cand))
            sev_name = str(rng.choice(sorted(SEVERITIES)))
            for b in valid_bins:
                is_inj = (b == inj_bin)
                vals = inject(values[b], SEVERITIES[sev_name]) if is_inj \
                    else values[b]
                raw = bin_raw_means(vals, pv[b], in_field)
                if raw is None:
                    continue
                d = oe.day_to_date(oe.bin_center_day(start_day, int(b)))
                rows.append(dict(
                    field_id=int(fid), cycle_index=int(ci), bin_index=int(b),
                    day_index=start_day + b * 15 + 7, date=d.isoformat(),
                    year=d.year, injected=bool(is_inj),
                    severity=sev_name if is_inj else "",
                    sensor=int(sid[b]), n_obs=int(n_obs[b]),
                    **{name: float(raw[k]) for k, name in
                       enumerate(common.PATCH_CHANNEL_ORDER)}))
                ch = {name: vals[k] for k, name in
                      enumerate(common.PATCH_CHANNEL_ORDER)}
                norm = oe.assemble_and_normalize(ch, model_stats)
                windows.append((len(rows) - 1, dict(
                    x=norm[None], mask_px=pv[b][None],
                    timestamps=np.array([[d.day, d.month - 1, d.year]]),
                    bins=[int(b)], window_index=0, cycle_index=int(ci),
                    _row=len(rows) - 1)))
    embs = np.zeros((len(rows), 128), dtype=np.float64)
    for i in range(0, len(windows), 64):
        chunk = [w for _, w in windows[i : i + 64]]
        pooled = run_windows(model, chunk)
        for (_, w), emb in zip(windows[i : i + 64], pooled):
            embs[w["_row"]] = emb[0]
    return pd.DataFrame(rows), embs


def phase_matched_z(values: np.ndarray, phases: np.ndarray,
                    ref_by_phase: dict) -> np.ndarray:
    z = np.zeros_like(values, dtype=np.float64)
    for p in np.unique(phases):
        m = phases == p
        if p not in ref_by_phase or m.sum() == 0:
            continue
        z[m] = common.robust_z(values[m], ref_by_phase[p])
    return z


def persistence_scores(df: pd.DataFrame) -> np.ndarray:
    """B1: −(NDVI − median of the field-cycle's ≤3 previous valid bins)."""
    df = df.reset_index(drop=True)
    scores = np.full(len(df), np.nan)
    for (_, _), g in df.groupby(["field_id", "cycle_index"]):
        g = g.sort_values("bin_index")
        ndvi = g["ndvi"].values
        for i in range(len(g)):
            prev = ndvi[max(0, i - 3):i]
            if len(prev) < 2 or not np.isfinite(ndvi[i]):
                continue
            scores[g.index[i]] = -(ndvi[i] - np.median(prev))
    return scores


def compute_scores(raw_df: pd.DataFrame, embs: np.ndarray,
                   ref_emb: dict, ref_raw: dict,
                   nn_by_phase: dict | None = None) -> dict[str, np.ndarray]:
    """Combined scoring for populations where raw_df and embs are row-aligned
    (the dev build produces them together)."""
    phases = raw_df["phase"].astype(str).values
    z_emb = phase_matched_z(embs, phases, ref_emb)
    z_raw = phase_matched_z(raw_df[common.PATCH_CHANNEL_ORDER].values, phases,
                            ref_raw)
    knn = np.zeros(len(raw_df))
    if nn_by_phase:
        for p, model in nn_by_phase.items():
            m = phases == p
            if m.sum():
                dist, _ = model.kneighbors(embs[m])
                knn[m] = dist.mean(axis=1)
    return {
        "OE_emb_maxabsz": np.abs(z_emb).max(axis=1),
        "OE_emb_meanabsz": np.abs(z_emb).mean(axis=1),
        "OE_knn_dist": knn,
        "B0_raw_maxabsz": np.abs(z_raw).max(axis=1),
        "B1_persistence": persistence_scores(raw_df),
    }


def fit_phase_nn(emb_by_phase: dict, k: int = KNN_K) -> dict:
    """Phase-matched kNN reference models on TRAIN embeddings (score =
    mean distance to the k nearest train embeddings of the same phase)."""
    from sklearn.neighbors import NearestNeighbors

    out = {}
    for p, arr in emb_by_phase.items():
        kk = min(k, max(1, len(arr) - 1))
        out[p] = NearestNeighbors(n_neighbors=kk).fit(arr)
    return out


def train_population_scores(emb_tr: pd.DataFrame, raw_tr: pd.DataFrame,
                            ref_emb: dict, ref_raw: dict,
                            nn_by_phase: dict | None = None) -> dict[str, np.ndarray]:
    """Train calibration scores. Embeddings and raw features come from
    DIFFERENT row orders, so each score is computed against its own
    population (self-z: each bin's z vs its phase reference that contains it
    — applied identically to OE and B0, so the OE-vs-B0 comparison stays
    fair; noted in the report)."""
    phases_e = emb_tr["phase"].astype(str).values
    z_emb = phase_matched_z(emb_tr[common.EMBED_COLS].values, phases_e, ref_emb)
    phases_r = raw_tr["phase"].astype(str).values
    z_raw = phase_matched_z(raw_tr[common.PATCH_CHANNEL_ORDER].values, phases_r,
                            ref_raw)
    knn = np.zeros(len(emb_tr))
    if nn_by_phase:
        embs = emb_tr[common.EMBED_COLS].values
        for p, model in nn_by_phase.items():
            m = phases_e == p
            if m.sum():
                dist, _ = model.kneighbors(embs[m])
                knn[m] = dist.mean(axis=1)
    return {
        "OE_emb_maxabsz": np.abs(z_emb).max(axis=1),
        "OE_emb_meanabsz": np.abs(z_emb).mean(axis=1),
        "OE_knn_dist": knn,
        "B0_raw_maxabsz": np.abs(z_raw).max(axis=1),
        "B1_persistence": persistence_scores(raw_tr),
    }


def evaluate_scores(y: np.ndarray, sev: np.ndarray, scores: dict,
                    train_thr: dict) -> dict:
    from sklearn.metrics import average_precision_score

    aucpr, at_thr = {}, {}
    neg = y == 0
    neg_mask_all = {}
    for name in SCORE_NAMES:
        s = scores[name]
        mask = np.isfinite(s)
        neg_mask = mask & neg
        # overall: injected (all severities) vs clean dev bins
        aucpr[name] = dict(
            overall=float(average_precision_score(y[mask], s[mask])))
        # per severity: positives = that severity's injected bins,
        # negatives = ALL clean dev bins (they are negative by construction)
        for sname in SEVERITIES:
            m = mask & (sev == sname)
            if m.sum() and neg_mask.sum():
                yy = np.concatenate([np.ones(int(m.sum())),
                                     np.zeros(int(neg_mask.sum()))])
                ss = np.concatenate([s[m], s[neg_mask]])
                aucpr[name][sname] = float(average_precision_score(yy, ss))
        thr = train_thr[name]
        rec_train = float((s[mask & (y == 1)] > thr).mean())
        far_train_thr = float((s[neg_mask] > thr).mean())
        # FAR-matched operating point computed ON the clean dev bins (the
        # negatives are negative by construction — untouched bins); reported
        # next to the train-calibrated recall so the transfer gap is visible
        far_thr = float(np.quantile(s[neg_mask], 0.95))
        rec_far_matched = float((s[mask & (y == 1)] > far_thr).mean())
        at_thr[name] = dict(
            train_threshold=thr,
            recall_at_train_thr=rec_train,
            far_on_clean_dev_at_train_thr=far_train_thr,
            far_matched_5pct_threshold=far_thr,
            recall_at_far_matched_5pct=rec_far_matched)
    return dict(aucpr=aucpr, at_threshold=at_thr)


def concordance(dev: pd.DataFrame, scores: dict) -> dict:
    """Real clean pre-2024 bins: flag-set overlap between arms at 2 % budget."""
    d = dev[(~dev["injected"].values) & (dev["year"] < 2024)]
    flags = {}
    for name in SCORE_NAMES:
        s = pd.Series(scores[name]).loc[d.index].values
        m = np.isfinite(s)
        k = max(1, int(round(FLAG_BUDGET * m.sum())))
        order = np.argsort(np.nan_to_num(s, nan=-np.inf))[::-1]
        flagged = set(d.index[order[:k]].tolist())
        flags[name] = flagged
    out = {"n_real_bins": int(len(d))}
    for a, b in (("OE_emb_maxabsz", "B0_raw_maxabsz"),
                 ("OE_emb_maxabsz", "B1_persistence"),
                 ("B0_raw_maxabsz", "B1_persistence")):
        inter = len(flags[a] & flags[b])
        union = len(flags[a] | flags[b])
        out[f"jaccard_{a}__{b}"] = round(inter / union, 4) if union else None
    out["flag_counts"] = {k: len(v) for k, v in flags.items()}
    out["note"] = ("flag-set overlap on REAL clean dev bins (pre-2024) at a "
                   "2 % budget; concordance evidence only — not accuracy")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    args = ap.parse_args(argv)

    model_stats = oe.load_computed_norm(NORM_CONFIG)
    train_fields, dev_fields = common.split_field_sets()

    # --- train population (clean, calibration only) ---
    emb_tr = common.add_derived_columns(common.load_obs_embeddings(""))
    emb_tr = emb_tr[emb_tr["field_id"].isin(train_fields)]
    raw_tr = common.add_derived_columns(common.load_raw_bin_features())
    raw_tr = raw_tr[raw_tr["field_id"].isin(train_fields)]
    ref_emb = {p: g[common.EMBED_COLS].values for p, g in emb_tr.groupby("phase")}
    ref_raw = {p: g[common.PATCH_CHANNEL_ORDER].values
               for p, g in raw_tr.groupby("phase")}
    nn_by_phase = fit_phase_nn(ref_emb)
    train_scores = train_population_scores(emb_tr, raw_tr, ref_emb, ref_raw,
                                           nn_by_phase)
    train_thr = {name: float(np.nanquantile(train_scores[name], 0.95))
                 for name in SCORE_NAMES}

    seed_results = []
    last = None
    for seed in args.seeds:
        dev, embs = build_dev_bins(model_stats, dev_fields, seed)
        dev = common.add_derived_columns(dev)
        scores = compute_scores(dev, embs, ref_emb, ref_raw, nn_by_phase)
        y = dev["injected"].values.astype(int)
        res = evaluate_scores(y, dev["severity"].values, scores, train_thr)
        seed_results.append(dict(seed=seed, n_injected=int(y.sum()),
                                 **res))
        last = (dev, scores)

    agg = {}
    for name in SCORE_NAMES:
        agg[name] = dict(
            aucpr_overall_mean=round(float(np.mean(
                [r["aucpr"][name]["overall"] for r in seed_results])), 4),
            aucpr_overall_std=round(float(np.std(
                [r["aucpr"][name]["overall"] for r in seed_results])), 4),
            recall_at_train_thr_mean=round(float(np.mean(
                [r["at_threshold"][name]["recall_at_train_thr"]
                 for r in seed_results])), 4),
            far_on_clean_dev_at_train_thr_mean=round(float(np.mean(
                [r["at_threshold"][name]["far_on_clean_dev_at_train_thr"]
                 for r in seed_results])), 4),
            recall_at_far_matched_5pct_mean=round(float(np.mean(
                [r["at_threshold"][name]["recall_at_far_matched_5pct"]
                 for r in seed_results])), 4),
        )
    dev, scores = last
    out = dict(
        seeds=list(args.seeds),
        n_dev_fields=int(len(dev_fields)),
        n_dev_bins=int(len(dev)), n_injected=int(dev["injected"].sum()),
        severities={k: list(v) for k, v in SEVERITIES.items()},
        train_thresholds=train_thr,
        protocol="controlled synthetic deviations on dev fields; train-only "
                 "calibration; injected=positives, untouched dev bins="
                 "negatives; SYNTHETIC injections, L2 only",
        aucpr_mean_over_seeds=agg,
        per_seed=seed_results,
        concordance_real=concordance(dev, scores),
    )
    path = common.write_table("detect_anomalies", out)
    print(f"wrote {path}")
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

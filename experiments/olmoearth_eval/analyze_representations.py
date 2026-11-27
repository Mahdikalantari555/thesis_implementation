#!/usr/bin/env python
"""Representation-quality battery for the OlmoEarth embeddings (E1–E3).

E1  temporal representation analysis: PCA of observation embeddings colored
    by month/phase; per-field phenological trajectory sanity.
E2  phenological consistency across years: phase-matched cross-year cosine
    vs phase-mismatched same-field cosine, bootstrap CI over fields.
E3  cross-year / cross-farm similarity: kNN purity for "same field" and
    "same field AND different year" (the sharper, autocorrelation-controlled
    variant), plus same-phase cross-field reference rates.

Outputs: outputs/tables/representations.json + outputs/figures/*.png.
Everything here is L1 (representation learned) — no anomaly claims.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

import oe_eval_common as common  # noqa: E402


def pca_figures(df: pd.DataFrame, fig_dir: Path) -> dict:
    E = df[common.EMBED_COLS].values
    E = (E - E.mean(0)) / (E.std(0) + 1e-9)
    pca = PCA(n_components=2, random_state=0)
    P = pca.fit_transform(E)
    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, title in (
        (axes[0], "month", "PCA colored by month"),
        (axes[1], "phase", "PCA colored by derived phase"),
    ):
        if col == "month":
            sc = ax.scatter(P[:, 0], P[:, 1], c=df[col], s=4, cmap="twilight", alpha=0.5)
            fig.colorbar(sc, ax=ax, label="month")
        else:
            codes = df[col].astype("category").cat.codes
            sc = ax.scatter(P[:, 0], P[:, 1], c=codes, s=4, cmap="viridis", alpha=0.5)
            fig.colorbar(sc, ax=ax, label="phase (derived)")
        ax.set_title(title)
        ax.set_xlabel(f"PC1 ({evr[0]:.1%})")
        ax.set_ylabel(f"PC2 ({evr[1]:.1%})")
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    out1 = fig_dir / "pca_month_phase.png"
    fig.savefig(out1, dpi=130)
    plt.close(fig)

    # per-field trajectories for 6 sample fields
    sample_fields = sorted(df["field_id"].unique())[:6]
    fig, ax = plt.subplots(figsize=(6, 5))
    cmap = plt.get_cmap("twilight")
    for fid in sample_fields:
        g = df[df["field_id"] == fid].sort_values("day_index")
        gp = pca.transform(
            (g[common.EMBED_COLS].values - E.mean(0)) / (E.std(0) + 1e-9))
        ax.plot(gp[:, 0], gp[:, 1], "-o", ms=3, lw=1, alpha=0.8, label=f"field {fid}")
    ax.set_title("Per-field embedding trajectories (cycle-ordered)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=7)
    fig.tight_layout()
    out2 = fig_dir / "trajectories.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)

    return dict(pca_explained_variance=[float(v) for v in evr],
                figures=[str(out1), str(out2)])


def phase_cosine_matrix(g: pd.DataFrame) -> dict[str, np.ndarray]:
    """Mean cosine per (phaseA, phaseB) pair between two cycles of one field."""
    phases = common.PHASE_NAMES
    means = {}
    for pa in phases:
        for pb in phases:
            ea = g[g["phase"] == pa][common.EMBED_COLS].values
            eb = g[g["phase"] == pb][common.EMBED_COLS].values
            if len(ea) == 0 or len(eb) == 0:
                continue
            sims = (ea @ eb.T) / (
                np.linalg.norm(ea, axis=1, keepdims=True)
                * np.linalg.norm(eb, axis=1, keepdims=True).T)
            means[(pa, pb)] = sims.mean()
    return means


def cross_year_consistency(df: pd.DataFrame) -> dict:
    """E2: phase-matched vs phase-mismatched cross-year cosine, same field."""
    per_field_margin = {}
    per_field_matched = {}
    for fid, g in df.groupby("field_id"):
        cycles = sorted(g["cycle_index"].unique())
        if len(cycles) < 2:
            continue
        years = g.groupby("cycle_index")["year"].first()
        margins, matched_vals = [], []
        for a in range(len(cycles)):
            for b in range(a + 1, len(cycles)):
                ca, cb = cycles[a], cycles[b]
                if years[ca] == years[cb]:
                    continue  # cross-year pairs only
                ga = g[g["cycle_index"] == ca]
                gb = g[g["cycle_index"] == cb]
                m = phase_cosine_matrix(pd.concat([ga, gb]))
                diag, off = [], []
                for pa in common.PHASE_NAMES:
                    if (pa, pa) in m:
                        diag.append(m[(pa, pa)])
                    for pb in common.PHASE_NAMES:
                        if pb != pa and (pa, pb) in m:
                            off.append(m[(pa, pb)])
                if diag and off:
                    matched_vals.append(float(np.mean(diag)))
                    margins.append(float(np.mean(diag) - np.mean(off)))
        if margins:
            per_field_margin[fid] = float(np.mean(margins))
            per_field_matched[fid] = float(np.mean(matched_vals))
    margins = np.array(list(per_field_margin.values()))
    matched = np.array(list(per_field_matched.values()))
    if len(margins) == 0:
        return dict(n_fields=0)
    pt, lo, hi = common.bootstrap_ci(margins, stat=np.median)
    ptm, lom, him = common.bootstrap_ci(matched, stat=np.median)
    return dict(
        n_fields=int(len(margins)),
        matched_cosine_median=ptm, matched_ci=[lom, him],
        margin_median=pt, margin_ci=[lo, hi],
        margin_mean=float(margins.mean()),
        n_margin_positive=int((margins > 0).sum()),
        note="margin = phase-matched cosine − phase-mismatched cosine, "
             "cross-year pairs of the same field; bootstrap CI over fields",
    )


def knn_purity(df: pd.DataFrame, k: int = 5, max_n: int = 8000) -> dict:
    """E3: kNN purity for same-field and same-field-different-year."""
    from sklearn.neighbors import NearestNeighbors

    E = df[common.EMBED_COLS].values
    same_field = df["field_id"].values
    year = df["year"].values
    if len(E) > max_n:
        rng = np.random.default_rng(2026)
        idx = rng.choice(len(E), size=max_n, replace=False)
        E, same_field, year = E[idx], same_field[idx], year[idx]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
    _, ind = nn.kneighbors(E)
    ind = ind[:, 1:]  # drop self
    sf = (same_field[ind] == same_field[:, None])
    sf_dy = sf & (year[ind] != year[:, None])
    return dict(
        k=k, n=len(E),
        purity_same_field=float(sf.mean()),
        purity_same_field_diff_year=float(sf_dy.mean()),
        note="same-field-diff-year controls temporal autocorrelation; "
             "values are nearest-neighbour vote fractions (L1 evidence)",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default="")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args(argv)

    df = common.load_obs_embeddings(args.suffix)
    df = common.add_derived_columns(df)
    fig_dir = common.OUT_DIR / "figures"

    out: dict = {"suffix": args.suffix, "n_obs": int(len(df)),
                 "n_fields": int(df["field_id"].nunique()),
                 "n_cycles": int(df.groupby(["field_id", "cycle_index"]).ngroups),
                 "years": sorted(int(y) for y in df["year"].unique())}
    out["e1_pca"] = pca_figures(df, fig_dir)
    out["e2_cross_year_consistency"] = cross_year_consistency(df)
    out["e3_knn_purity"] = knn_purity(df)

    path = common.write_table("representations", out)
    print(f"wrote {path}")
    print(json.dumps({k: out[k] for k in
                      ("n_obs", "n_fields", "e2_cross_year_consistency",
                       "e3_knn_purity")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

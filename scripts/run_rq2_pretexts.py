# ruff: noqa: E402
#!/usr/bin/env python3
"""RQ2 (Gap G3): Controlled comparison of three SSL pretext objectives under matched budgets.

Compares on the identical 1D Transformer backbone (d_model=64, n_layers=3, ~161K params),
epoch schedule, and learning rate across 5 seeds:
  1. ``masked_recon`` (Masked Observation Reconstruction)
  2. ``traj_contrastive`` (Trajectory-Contrastive InfoNCE)
  3. ``anomaly_degradation`` (AnomalyBERT-style Self-Supervised Degradation Discrimination)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ssl_anomaly.utils.throttle import limit_threads

limit_threads(2)

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

torch.set_num_threads(2)

from ssl_anomaly.baseline import recall_at_alert_budget
from ssl_anomaly.eval.stats import bootstrap_ci, spearman_correlation
from ssl_anomaly.models.pretexts import PRETEXT_ARMS, score_cycle_pretext, train_pretext_arm
from ssl_anomaly.utils.logging import get_logger

log = get_logger("ssl_anomaly.scripts.rq2")

PERTURB_KINDS = ("ndvi_drop", "lst_spike", "phase_shift", "early_senescence")


def _perturb_tokens(tok_arr: np.ndarray, bands: list[str], kind: str, rng: np.random.Generator) -> np.ndarray:
    out = tok_arr.copy()
    T = out.shape[0]
    bmap = {b: i for i, b in enumerate(bands)}
    r_i, n_i, nd_i, lst_i = bmap["SR_B4"], bmap["SR_B5"], bmap["NDVI"], bmap["ST_B10"]
    L = min(4, max(2, T // 5))
    s = int(rng.integers(1, max(2, T - L)))

    if kind == "ndvi_drop":
        out[s : s + L, n_i] -= 2.8
        out[s : s + L, nd_i] -= 3.0
        out[s : s + L, r_i] += 2.0
    elif kind == "lst_spike":
        out[s : s + L, lst_i] += 3.2
    elif kind == "phase_shift":
        out[:, : len(bands)] = np.roll(out[:, : len(bands)], shift=max(2, T // 5), axis=0)
    elif kind == "early_senescence":
        st = int(0.65 * T)
        ramp = np.linspace(0.5, 3.0, max(1, T - st), dtype=np.float32)
        out[st:, n_i] -= ramp
        out[st:, nd_i] -= ramp
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="RQ2 controlled 3-arm SSL pretext comparison.")
    ap.add_argument("--zarr", default="data/processed/fields.zarr", type=Path)
    ap.add_argument("--cycles", default="data/processed/cycles.parquet", type=Path)
    ap.add_argument("--epochs", default=45, type=int)
    ap.add_argument("--seeds", default="42,43,44,45,46", type=str)
    ap.add_argument("--out", default="reports/rq2_pretexts.json", type=Path)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from compare_ssl_models import _load_records_and_cwsi

    records, tok, bands, cwsi_arr = _load_records_and_cwsi(args.zarr, args.cycles)
    n_bands = len(bands)
    train_toks = [r.tokens for r in records if not r.is_validation]
    all_toks_clean = [r.tokens for r in records]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    arm_summary: dict[str, dict] = {}
    for arm in PRETEXT_ARMS:
        prs, rocs, rec5s, rec10s, times_s = [], [], [], [], []
        per_kind: dict[str, list[float]] = {k: [] for k in PERTURB_KINDS}
        real_scores_seeds = []
        n_params = 0

        for seed in seeds:
            t0 = time.perf_counter()
            model = train_pretext_arm(
                arm, train_toks, n_bands, epochs=args.epochs, batch_size=16, lr=1.2e-3, seed=seed
            )
            times_s.append(time.perf_counter() - t0)
            n_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

            with torch.no_grad():
                ref_embs = np.stack(
                    [model.encode_sequence(torch.from_numpy(t).unsqueeze(0))[1][0].numpy() for t in train_toks],
                    axis=0,
                )

            real_sc = np.array([score_cycle_pretext(model, t, ref_embs) for t in all_toks_clean])
            real_scores_seeds.append(real_sc)

            rng = np.random.default_rng(seed + 200)
            s_prs, s_rocs, s_r5, s_r10 = [], [], [], []
            for kind in PERTURB_KINDS:
                pert = [_perturb_tokens(t, bands, kind, rng) for t in all_toks_clean]
                eval_pool = all_toks_clean + pert
                # Use 15% anomaly prevalence pool (clean repeated 5x + 1x pert -> ~16.7% prevalence)
                clean_rep = all_toks_clean * 5
                pool_15 = clean_rep + pert
                yt_15 = np.array([0] * len(clean_rep) + [1] * len(pert), dtype=np.int32)
                sc_clean = np.array([score_cycle_pretext(model, t, ref_embs) for t in all_toks_clean])
                sc_pert = np.array([score_cycle_pretext(model, t, ref_embs) for t in pert])
                sc_15 = np.concatenate([np.tile(sc_clean, 5), sc_pert])

                yt_50 = np.array([0] * len(all_toks_clean) + [1] * len(pert), dtype=np.int32)
                sc_50 = np.concatenate([sc_clean, sc_pert])
                ap = float(average_precision_score(yt_50, sc_50))
                roc = float(roc_auc_score(yt_50, sc_50))
                r5 = recall_at_alert_budget(yt_15, sc_15, alert_rate=0.05)
                r10 = recall_at_alert_budget(yt_15, sc_15, alert_rate=0.10)

                s_prs.append(ap)
                s_rocs.append(roc)
                s_r5.append(r5)
                s_r10.append(r10)
                per_kind[kind].append(ap)
                _ = eval_pool

            prs.append(float(np.mean(s_prs)))
            rocs.append(float(np.mean(s_rocs)))
            rec5s.append(float(np.mean(s_r5)))
            rec10s.append(float(np.mean(s_r10)))

        pr_m, pr_lo, pr_hi = bootstrap_ci(np.mean, np.array(prs), n_boot=1000, seed=42)
        roc_m, roc_lo, roc_hi = bootstrap_ci(np.mean, np.array(rocs), n_boot=1000, seed=42)
        mean_real = np.mean(real_scores_seeds, axis=0)
        rho_cw, p_cw = spearman_correlation(mean_real, cwsi_arr)

        arm_summary[arm] = {
            "n_parameters": n_params,
            "mean_train_seconds": float(np.mean(times_s)),
            "auc_pr": pr_m,
            "auc_pr_ci_95": [pr_lo, pr_hi],
            "auc_roc": roc_m,
            "auc_roc_ci_95": [roc_lo, roc_hi],
            "recall_at_5pct_budget": float(np.mean(rec5s)),
            "recall_at_10pct_budget": float(np.mean(rec10s)),
            "per_perturbation_auc_pr": {k: float(np.mean(per_kind[k])) for k in PERTURB_KINDS},
            "cwsi_spearman_rho": rho_cw,
            "cwsi_pval": p_cw,
        }

    report = {
        "backbone": "PretextTransformer(d_model=64, n_heads=4, n_layers=3)",
        "epochs": args.epochs,
        "seeds": seeds,
        "arms": arm_summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    rows = [
        {
            "pretext_arm": arm,
            "n_params": m["n_parameters"],
            "train_sec": m["mean_train_seconds"],
            "auc_pr": m["auc_pr"],
            "auc_pr_ci_lo": m["auc_pr_ci_95"][0],
            "auc_pr_ci_hi": m["auc_pr_ci_95"][1],
            "auc_roc": m["auc_roc"],
            "recall_at_5pct": m["recall_at_5pct_budget"],
            "recall_at_10pct": m["recall_at_10pct_budget"],
            "ndvi_drop_pr": m["per_perturbation_auc_pr"]["ndvi_drop"],
            "lst_spike_pr": m["per_perturbation_auc_pr"]["lst_spike"],
            "phase_shift_pr": m["per_perturbation_auc_pr"]["phase_shift"],
            "early_senescence_pr": m["per_perturbation_auc_pr"]["early_senescence"],
            "cwsi_rho": m["cwsi_spearman_rho"],
        }
        for arm, m in arm_summary.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print("\n=== RQ2 Controlled SSL Pretext Comparison (Matched Backbone & Budget, 5 Seeds) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

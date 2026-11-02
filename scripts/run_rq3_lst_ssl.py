# ruff: noqa: E402
#!/usr/bin/env python3
"""RQ3 (Gap G1): Non-Circular SSL-Level Optical vs. Optical+LST Representation Test.

Tests whether including thermal ``ST_B10`` (LST) during SSL pretraining measurably
improves SSL representations and anomaly detection on STRICTLY NON-LST TARGETS:
  1. Optical-only reconstruction residual on non-thermal canopy perturbations
     (``ndvi_drop``, ``early_senescence``, ``phase_shift`` — zero LST perturbation,
     scoring ONLY the 6 optical bands ``SR_B2..SR_B7`` at test time).
  2. Coupled irrigation/stomatal-closure episodes from the multi-year archive where
     test-time ``ST_B10`` is NOT included in the anomaly residual or where frozen
     SSL encoder embeddings probe independent agro-evaporative ``ETa`` deficit & ``ET0``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ssl_anomaly.utils.throttle import limit_threads

limit_threads(2)

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import average_precision_score, r2_score
from sklearn.model_selection import KFold

torch.set_num_threads(2)

from ssl_anomaly.eval.cv import simulate_490_field_archive
from ssl_anomaly.eval.stats import bootstrap_ci, paired_wilcoxon
from ssl_anomaly.models.pretexts import PretextTransformer, train_pretext_arm
from ssl_anomaly.utils.logging import get_logger

log = get_logger("ssl_anomaly.scripts.rq3")

OPTICAL_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
OPTICAL_THERMAL_BANDS = [*OPTICAL_BANDS, "ST_B10"]


@torch.no_grad()
def _batch_score_and_embed(
    model: PretextTransformer,
    toks: list[np.ndarray],
    eval_band_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized batched sliding-block masked reconstruction score + frozen embeddings."""
    N = len(toks)
    d = toks[0].shape[1]
    T_max = max(t.shape[0] for t in toks)
    X_pad = np.zeros((N, T_max, d), dtype=np.float32)
    P_pad = np.ones((N, T_max), dtype=bool)
    for i, t in enumerate(toks):
        L = t.shape[0]
        X_pad[i, :L] = t
        P_pad[i, :L] = False

    X_t = torch.from_numpy(X_pad)
    P_t = torch.from_numpy(P_pad)
    nb = model.n_bands
    use_b = eval_band_indices if eval_band_indices is not None else list(range(nb))

    _, embs_t = model.encode_sequence(X_t, mask=None, key_padding_mask=P_t)
    tok_rms = np.full((N, T_max), np.nan, dtype=np.float64)
    for st in range(0, T_max, 4):
        en = min(T_max, st + 4)
        m = torch.zeros((N, T_max), dtype=torch.bool)
        m[:, st:en] = ~P_t[:, st:en]
        xin = X_t.clone()
        xin[:, st:en, :nb] = 0.0
        h, _ = model.encode_sequence(xin, mask=m, key_padding_mask=P_t)
        recon = model.task_head(h)
        diff = (recon[:, st:en, :][:, :, use_b] - X_t[:, st:en, :][:, :, use_b]).numpy()
        rms = np.sqrt(np.mean(diff**2, axis=-1))
        obs = (~P_t[:, st:en]).numpy()
        rms[~obs] = np.nan
        tok_rms[:, st:en] = rms

    scores = np.zeros(N, dtype=np.float64)
    for i in range(N):
        fin = np.sort(tok_rms[i][np.isfinite(tok_rms[i])])
        scores[i] = float(np.mean(fin[-min(3, fin.size) :])) if fin.size else 0.0
    return scores, embs_t.numpy()


def _standardize_cycle_matrix(X_train: np.ndarray, X_all: np.ndarray) -> list[np.ndarray]:
    med = np.nanmedian(X_train, axis=(0, 1))
    mad = np.nanmedian(np.abs(X_train - med[None, None, :]), axis=(0, 1)) * 1.4826
    scale = np.where(np.isfinite(mad) & (mad > 1e-4), mad, 1.0)
    K = X_all.shape[1]
    tau = np.linspace(0.0, 1.0, K, dtype=np.float32)[:, None]

    toks: list[np.ndarray] = []
    for i in range(len(X_all)):
        obs = np.any(np.isfinite(X_all[i]), axis=-1)
        idx = np.where(obs)[0]
        if idx.size < 4:
            idx = np.arange(min(8, K))
        z = (X_all[i, idx] - med[None, :]) / scale[None, :]
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        dsh_f = tau[idx]
        toks.append(np.hstack([z, dsh_f, np.sqrt(dsh_f)]).astype(np.float32))
    return toks


def _probe_frozen_r2(embeddings: np.ndarray, target: np.ndarray, seed: int = 42) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    preds = np.zeros_like(target, dtype=np.float64)
    for tr, va in kf.split(embeddings):
        reg = RidgeCV(alphas=np.logspace(-2, 3, 12))
        reg.fit(embeddings[tr], target[tr])
        preds[va] = reg.predict(embeddings[va])
    return float(r2_score(target, preds))


def main() -> int:
    ap = argparse.ArgumentParser(description="RQ3 non-circular SSL optical vs optical+LST test.")
    ap.add_argument("--zarr", default="data/processed/fields.zarr", type=Path)
    ap.add_argument("--cycles", default="data/processed/cycles.parquet", type=Path)
    ap.add_argument("--epochs", default=35, type=int)
    ap.add_argument("--seeds", default="42,43,44,45,46", type=str)
    ap.add_argument("--out", default="reports/rq3_lst_ssl.json", type=Path)
    args = ap.parse_args()

    import xarray as xr

    from ssl_anomaly.phenology import phase_bin

    # 1. Real 32-farm non-censored cycles + independent weather_et0 target
    ds = xr.open_zarr(args.zarr)
    cycles_df = pd.read_parquet(args.cycles)
    times = ds.coords["time"].values
    epoch = times[0]
    raw_bands = [str(b) for b in ds.coords["band"].values]
    bmap_real = {b: i for i, b in enumerate(raw_bands)}
    fields = ds.coords["field"].values
    med = ds["median"].values
    et0_da = ds["weather_et0"].values

    X_real_list, et0_real = [], []
    for _, row in cycles_df.iterrows():
        if row["censored"]:
            continue
        fid = int(row["field"])
        fi = int(np.where(fields == fid)[0][0])
        t0 = epoch + np.timedelta64(int(row["t_start"]), "D")
        t1 = epoch + np.timedelta64(int(row["t_end"]), "D")
        sh = epoch + np.timedelta64(int(row.get("start_harvest", row["t_start"])), "D")
        m = (times >= t0) & (times < t1)
        if int(m.sum()) < 4:
            continue
        dsh = (times[m] - sh).astype("timedelta64[D]").astype(int)
        pb = phase_bin(
            {b: med[fi, m, bmap_real[b]] for b in OPTICAL_THERMAL_BANDS},
            dsh,
            bin_days=15,
            max_cycle_days=540,
        )
        X_real_list.append(np.stack([pb[b] for b in OPTICAL_THERMAL_BANDS], axis=-1))
        et0_real.append(float(np.nanmean(et0_da[m])))
    X_real = np.stack(X_real_list, axis=0)
    et0_real_arr = np.array(et0_real, dtype=np.float64)

    # 2. Multi-year coupled irrigation-deficit cohort (600 cycles, strictly non-circular ETa deficit target)
    sim = simulate_490_field_archive(n_fields=80, n_years=10.0, seed=42)
    X_sim = sim["X"][:, :, :7]  # SR_B2..SR_B7 + ST_B10
    y_sim = sim["y_stress"]
    eta_sim = sim["eta_deficit"]
    norm_sim = y_sim == 0

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    metrics: dict[str, dict[str, list[float]]] = {
        "optical_only_ssl": {"real_optical_pert_pr": [], "real_et0_probe_r2": [], "coupled_stress_pr": [], "eta_probe_r2": []},
        "optical_plus_lst_ssl": {"real_optical_pert_pr": [], "real_et0_probe_r2": [], "coupled_stress_pr": [], "eta_probe_r2": []},
    }

    for seed in seeds:
        rng = np.random.default_rng(seed)

        for cfg_name, n_b in [("optical_only_ssl", 6), ("optical_plus_lst_ssl", 7)]:
            # --- Pass A: Real 32-farm optical-only perturbations + ET0 frozen probe ---
            toks_r = _standardize_cycle_matrix(X_real[:28, :, :n_b], X_real[:, :, :n_b])
            m_r = train_pretext_arm("masked_recon", toks_r[:28], n_bands=n_b, epochs=args.epochs, seed=seed)

            # Evaluate ONLY on optical channels [0..5] under purely optical canopy drop (NO LST perturbation!)
            pert_r = []
            for t in toks_r:
                tp = t.copy()
                s = int(rng.integers(1, max(2, len(tp) - 3)))
                tp[s : s + 3, 3] -= 2.4  # NIR drop
                tp[s : s + 3, 2] += 1.8  # Red rise
                tp[s : s + 3, 4] += 1.5  # SWIR1 rise
                pert_r.append(tp)
            sc_all_r, embs_pair = _batch_score_and_embed(
                m_r, toks_r + pert_r, eval_band_indices=list(range(6))
            )
            embs_r = embs_pair[: len(toks_r)]
            yt_r = np.array([0] * len(toks_r) + [1] * len(pert_r))
            metrics[cfg_name]["real_optical_pert_pr"].append(
                float(average_precision_score(yt_r, sc_all_r))
            )
            metrics[cfg_name]["real_et0_probe_r2"].append(_probe_frozen_r2(embs_r, et0_real_arr, seed=seed))

            # --- Pass B: Multi-year coupled irrigation/canopy episodes + non-circular ETa deficit probe ---
            toks_s = _standardize_cycle_matrix(X_sim[norm_sim, :, :n_b], X_sim[:, :, :n_b])
            train_s = [toks_s[i] for i in np.where(norm_sim)[0][:240]]
            m_s = train_pretext_arm("masked_recon", train_s, n_bands=n_b, epochs=20, batch_size=32, seed=seed)

            sc_s, embs_s = _batch_score_and_embed(m_s, toks_s)
            metrics[cfg_name]["coupled_stress_pr"].append(float(average_precision_score(y_sim, sc_s)))
            metrics[cfg_name]["eta_probe_r2"].append(_probe_frozen_r2(embs_s, eta_sim, seed=seed))

    summary: dict[str, dict] = {}
    for cfg_name, mdict in metrics.items():
        summary[cfg_name] = {}
        for k, vals in mdict.items():
            m, lo, hi = bootstrap_ci(np.mean, np.array(vals), n_boot=1000, seed=42)
            summary[cfg_name][k] = {"mean": m, "ci_95": [lo, hi]}

    deltas: dict[str, dict] = {}
    for k in metrics["optical_only_ssl"]:
        a = np.array(metrics["optical_plus_lst_ssl"][k])
        b = np.array(metrics["optical_only_ssl"][k])
        diff = a - b
        dm, dlo, dhi = bootstrap_ci(np.mean, diff, n_boot=1000, seed=42)
        _, pval = paired_wilcoxon(a, b)
        deltas[k] = {
            "delta_lst_minus_optical": dm,
            "delta_ci_95": [dlo, dhi],
            "wilcoxon_pval": float(pval) if np.isfinite(pval) else 1.0,
        }

    report = {
        "circularity_safeguard": "Zero CWSI targets used; real perturbation test scores optical SR_B2..SR_B7 only.",
        "seeds": seeds,
        "models": summary,
        "paired_deltas_lst_minus_optical": deltas,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    rows = []
    for k, d in deltas.items():
        rows.append(
            {
                "metric_non_circular": k,
                "optical_only_mean": summary["optical_only_ssl"][k]["mean"],
                "optical_plus_lst_mean": summary["optical_plus_lst_ssl"][k]["mean"],
                "delta_lst_gain": d["delta_lst_minus_optical"],
                "delta_ci_lo": d["delta_ci_95"][0],
                "delta_ci_hi": d["delta_ci_95"][1],
                "wilcoxon_pval": d["wilcoxon_pval"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(args.out.with_suffix(".csv"), index=False)
    print("\n=== RQ3 Non-Circular SSL-Level Optical vs. Optical+LST Test (5 Seeds) ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

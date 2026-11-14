#!/usr/bin/env python3
"""Dev-metric evaluation (design D6, task 3.2, spec model-development-protocol).

Evaluates one trained run (all seeds present under checkpoints/dev/<run_id>/)
on the DEV split only:

- FAR calibration on CLEAN dev cycles (cycle-level FAR ≤ 5 %), k ∈ {1, 2}
  chosen by detection rate; τ, k, achieved FAR recorded per seed.
- Primary: detection rate at 1×MAD over the injected set (mean ± std over
  seeds) + median detection delay in days.
- Secondary: partial-cycle AUC-PR at 25/50/75 %, masked NRMSE per band on
  clean dev, severity curve 0.5–2×MAD, per-type and per-phase tables.
- Writes reports/dev/<run_id>/metrics.json (per-seed + mean/std, full
  provenance) and appends one row to reports/dev_log.csv.

Forbidden: this script never reads CWSI, ETa, y_stress, yield or test cycles
(enforced by the forbidden-target unit test and the split guard).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ssl_anomaly.monitor import inputs as minputs  # noqa: E402
from ssl_anomaly.monitor import models as mmodels  # noqa: E402
from ssl_anomaly.monitor import perturbations as mpert  # noqa: E402
from ssl_anomaly.monitor import protocol as mproto  # noqa: E402
from ssl_anomaly.monitor import scoring as mscoring  # noqa: E402
from ssl_anomaly.monitor import splits as msplits  # noqa: E402

BAND_NAMES = ["ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-id", required=True, help="training run id (checkpoints/dev/<run_id>/)")
    p.add_argument("--model", required=True, choices=["masked_transformer", "gru_ae", "b1"])
    p.add_argument("--seeds", default="13,42,2026", help="seeds to evaluate (default: 13,42,2026)")
    p.add_argument("--far", type=float, default=0.05)
    p.add_argument("--n-per-stratum", type=int, default=8, help="injected cycles per dev cycle per round")
    p.add_argument("--rounds", type=int, default=6, help="injection rounds (each injects 1 perturbation per cycle)")
    p.add_argument("--smoke", action="store_true", help="tiny subset for agent smoke runs")
    p.add_argument("--device", default=None)
    p.add_argument("--notes", default="")
    return p.parse_args(argv)


def load_scorer(model: str, seed: int, run_id: str, device: str) -> tuple:
    """Load one checkpoint and return (scorer, n_params)."""
    import torch

    ckpt_dir = mproto.CHECKPOINTS_DEV_DIR / run_id
    if model == "b1":
        stats_path = ckpt_dir / f"b1_phasez_causal_seed{seed}.npz"
        if not stats_path.exists():
            raise FileNotFoundError(f"B1 stats not found: {stats_path}")
        d = np.load(stats_path, allow_pickle=True)
        b1 = mmodels.B1PhaseZCausal(use_doy=bool(d["use_doy"]), min_n=8)
        b1.median = d["median"]
        b1.scale = d["scale"]
        if "robust_range" in d.files:
            b1.robust_range = d["robust_range"]
        return b1, 0
    ckpt_path = ckpt_dir / f"{model}_seed{seed}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    torch_model = mmodels.build_model(model, d_in=cfg["d_in"], n_out=cfg["n_out"], cfg={
        k: cfg[k] for k in ("d_model", "n_layers", "d_ff", "n_heads", "dropout") if k in cfg
    })
    torch_model.load_state_dict(payload["state_dict"])
    return mmodels.DeepCausalScorer(
        model, torch_model, use_doy=bool(cfg.get("use_doy", True)), device=device,
    ), mmodels.count_parameters(torch_model)


def score_cycles(scorer, X_raw: np.ndarray, M: np.ndarray, doy: np.ndarray, norm: dict | None):
    """Score cycles; returns (scores (N,T), pred_raw (N,T,C)).

    Deep scorers work in z-space; predictions are mapped back to raw units so
    NRMSE is comparable with B1. B1 predicts the train climatology directly.
    """
    if isinstance(scorer, mmodels.B1PhaseZCausal):
        N, T, C = X_raw.shape
        scores = np.full((N, T), np.nan)
        pred_raw = np.zeros((N, T, C))
        for i in range(N):
            scores[i] = scorer.score_cycle(X_raw[i], M[i], doy[i, :, 0])
            g = scorer.season_group(doy[i, :, 0]) if scorer.use_doy else np.zeros(T, dtype=np.int64)
            pred_raw[i] = scorer.median[g, np.arange(T)]
        return scores, pred_raw
    z = minputs.apply_norm(norm, X_raw, M)
    scores, pred_z = scorer.loo_batch(z, M, doy)
    pred_raw = pred_z * norm["scale"][None] + norm["median"][None]
    return scores, pred_raw


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    t0 = time.time()
    seeds = [int(s) for s in args.seeds.split(",")]
    device = args.device or ("cuda" if _cuda() else "cpu")

    manifest = msplits.load_splits_locked()
    data = minputs.load_masked_cycles()
    sel = mproto.select_splits(data, manifest, splits=("dev",))
    X, M = sel["X"], sel["M"]
    doy = minputs.doy_channels(sel["harvest_days"])
    N, T, C = X.shape

    norm = None
    if args.model != "b1":
        norm_path = mproto.DEV_DIR / args.run_id / "norm_stats.npz"
        norm = minputs.load_norm(norm_path)

    if args.smoke:
        rng0 = np.random.default_rng(7)
        keep = rng0.choice(N, size=min(12, N), replace=False)
        X, M, doy = X[keep], M[keep], doy[keep]
        seeds = seeds[:1]
        args.rounds = 1
        args.n_per_stratum = 2
        N = len(keep)

    cfg_pert = mpert.PerturbationConfig(n_per_stratum=args.n_per_stratum)
    per_seed: list[dict] = []

    for seed in seeds:
        scorer, params = load_scorer(args.model, seed, args.run_id, device)

        # ---- Clean dev scores (calibration + NRMSE) -----------------------
        clean_scores, clean_pred_raw = score_cycles(scorer, X, M, doy, norm)
        clean_list = [clean_scores[i] for i in range(N)]

        # ---- Injected dev cycles ------------------------------------------
        inj_scores: list[np.ndarray] = []
        inj_meta: list[dict] = []
        inj_pred_raw = []
        for rnd in range(args.rounds):
            rng = np.random.default_rng(10_000 + seed * 97 + rnd)
            results = mpert.inject_dataset(X, M, norm_scale_or_mad(norm, scorer), cfg_pert, rng)
            Xinj = np.stack([r.x for r in results])
            Minj = np.stack([r.m for r in results])
            doy_inj = doy[[r.source_row for r in results]]
            s_inj, p_inj = score_cycles(scorer, Xinj, Minj, doy_inj, norm)
            for j, r in enumerate(results):
                inj_scores.append(s_inj[j])
                inj_meta.append({
                    "type": r.pert_type, "severity": float(r.severity),
                    "phase": r.phase, "onset_bin": int(r.onset_bin),
                })
                inj_pred_raw.append(p_inj[j])

        # ---- Threshold calibration on clean dev ---------------------------
        sev1_idx = [j for j, meta in enumerate(inj_meta) if abs(meta["severity"] - 1.0) < 1e-9]
        threshold, per_k = mscoring.choose_threshold(
            clean_list,
            [inj_scores[j] for j in sev1_idx],
            [inj_meta[j]["onset_bin"] for j in sev1_idx],
            far_target=args.far,
        )
        tau, k = threshold.tau, threshold.k

        # ---- Detection metrics --------------------------------------------
        def detect_at(sev: float | None) -> tuple[float, float, list[float]]:
            idx = [j for j, meta in enumerate(inj_meta)
                   if (sev is None or abs(meta["severity"] - sev) < 1e-9)]
            delays = []
            det = 0
            for j in idx:
                fb = mscoring.first_alarm_bin(inj_scores[j], tau, k, inj_meta[j]["onset_bin"])
                if fb >= 0:
                    det += 1
                    delays.append((fb - inj_meta[j]["onset_bin"]) * 15)
            rate = det / max(1, len(idx))
            med_delay = float(np.median(delays)) if delays else float("nan")
            return rate, med_delay, delays

        det1, delay1, _ = detect_at(1.0)
        det_all, _, _ = detect_at(None)

        # Severity curve.
        severity_curve = {}
        for sev in mpert.SEVERITIES:
            r, dmed, _ = detect_at(sev)
            severity_curve[str(sev)] = {"detection_rate": r, "median_delay_days": dmed}

        # Per-type and per-phase tables at 1×MAD.
        per_type: dict[str, dict] = {}
        per_phase: dict[str, dict] = {}
        for ptype in mpert.INTERIM_TYPES:
            idx = [j for j in sev1_idx if inj_meta[j]["type"] == ptype]
            if idx:
                r, dmed, _ = _rate_delay(inj_scores, inj_meta, idx, tau, k)
            else:
                r, dmed = float("nan"), float("nan")
            per_type[ptype] = {"n": len(idx), "detection_rate": r, "median_delay_days": dmed}
        for ph in mpert.PHASES:
            idx = [j for j in sev1_idx if inj_meta[j]["phase"] == ph]
            if idx:
                r, dmed, _ = _rate_delay(inj_scores, inj_meta, idx, tau, k)
            else:
                r, dmed = float("nan"), float("nan")
            per_phase[ph] = {"n": len(idx), "detection_rate": r, "median_delay_days": dmed}

        # ---- Partial-cycle AUC-PR (1×MAD vs clean) -------------------------
        aucpr = {}
        for frac in (0.25, 0.50, 0.75):
            cut = max(1, int(np.ceil(frac * T)))
            clean_max = np.array([
                np.nanmax(np.where(np.isfinite(s[:cut]), s[:cut], -np.inf)) for s in clean_list
            ])
            inj_max = np.array([
                np.nanmax(np.where(np.isfinite(inj_scores[j][:cut]), inj_scores[j][:cut], -np.inf))
                for j in sev1_idx
            ])
            aucpr[str(frac)] = mscoring.partial_auc_pr(clean_max, inj_max)

        # ---- Masked NRMSE per band (clean dev, raw units) ------------------
        nrmse = mscoring.masked_nrmse_per_band(
            clean_pred_raw, X, M, norm_robust_range(norm, scorer),
        )
        nrmse_valid = nrmse[np.isfinite(nrmse)]

        per_seed.append({
            "seed": seed,
            "tau": float(tau), "k": int(k),
            "far_target": args.far, "far_achieved": float(threshold.far_achieved),
            "threshold_per_k": per_k,
            "detection_rate_1mad": det1,
            "detection_rate_all_severities": det_all,
            "median_delay_days_1mad": delay1,
            "severity_curve": severity_curve,
            "per_type_1mad": per_type,
            "per_phase_1mad": per_phase,
            "aucpr_partial": aucpr,
            "nrmse_per_band": {b: float(nrmse[i]) for i, b in enumerate(BAND_NAMES)},
            "nrmse_mean": float(np.mean(nrmse_valid)) if nrmse_valid.size else float("nan"),
            "params": params,
            "n_clean": int(N), "n_injected": len(inj_meta), "n_injected_1mad": len(sev1_idx),
        })
        print(f"[seed {seed}] det@1MAD={det1:.3f} delay={delay1:.0f}d tau={tau:.4f} k={k} "
              f"far={threshold.far_achieved:.3f} aucpr50={aucpr['0.5']:.3f}")

    # ---- Aggregate over seeds ----------------------------------------------
    det_vals = [r["detection_rate_1mad"] for r in per_seed]
    det_mean, det_std = mproto.mean_std(det_vals)
    delay_vals = [r["median_delay_days_1mad"] for r in per_seed if np.isfinite(r["median_delay_days_1mad"])]
    delay_mean, delay_std = mproto.mean_std(delay_vals) if delay_vals else (float("nan"), 0.0)

    wall = time.time() - t0
    metrics = {
        "run_id": args.run_id,
        "model_id": args.model,
        "ladder_rung": "L2",
        "data_source": "real490 (masked rebuild from stats.zarr)",
        "splits_locked_sha256": manifest["manifest_sha256"],
        "git_sha": mproto.git_sha(),
        "seeds": seeds,
        "device": device,
        "wall_time_s": round(wall, 1),
        "far_target": args.far,
        "date_range": {
            "first": str(sel["start_dates"][0])[:10],
            "last": str(sel["end_dates"][-1])[:10],
        },
        "n_dev_cycles": int(N),
        "n_injected_total": sum(r["n_injected"] for r in per_seed),
        "primary": {
            "name": "detection_rate_at_1MAD",
            "mean": det_mean, "std": det_std,
            "per_seed": det_vals,
        },
        "median_delay_days": {"mean": delay_mean, "std": delay_std},
        "per_seed": per_seed,
        "smoke": bool(args.smoke),
        "notes": args.notes,
    }
    out_dir = mproto.DEV_DIR / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.smoke else ""
    out_path = out_dir / f"metrics_{args.model}{suffix}.json"
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if not args.smoke:
        nrmse_vals = [r["nrmse_mean"] for r in per_seed if np.isfinite(r["nrmse_mean"])]
        nrmse_mean = float(np.mean(nrmse_vals)) if nrmse_vals else float("nan")
        auc50 = [r["aucpr_partial"]["0.5"] for r in per_seed if np.isfinite(r["aucpr_partial"]["0.5"])]
        row = {
            "run_id": f"{args.run_id}:{args.model}",
            "date": time.strftime("%Y-%m-%d"),
            "model_id": args.model,
            "parent_run_id": "",
            "change_description": "initial architecture screen" + (f"; {args.notes}" if args.notes else ""),
            "config_hash": mproto.config_hash({"model": args.model, "run_id": args.run_id}),
            "seeds": ";".join(str(s) for s in seeds),
            "n_train_cycles": _n_train(manifest),
            "n_dev_cycles": int(N),
            "primary_metric_mean": round(det_mean, 4),
            "primary_metric_std": round(det_std, 4),
            "median_delay_days": round(delay_mean, 1) if np.isfinite(delay_mean) else "",
            "far_target": args.far,
            "far_achieved": round(float(np.max([r["far_achieved"] for r in per_seed])), 4),
            "tau": round(float(np.mean([r["tau"] for r in per_seed])), 4),
            "k": int(np.bincount([r["k"] for r in per_seed]).argmax()),
            "aucpr_25": round(float(np.mean([r["aucpr_partial"]["0.25"] for r in per_seed])), 4),
            "aucpr_50": round(float(np.mean(auc50)), 4) if auc50 else "",
            "aucpr_75": round(float(np.mean([r["aucpr_partial"]["0.75"] for r in per_seed])), 4),
            "nrmse_mean": round(nrmse_mean, 4) if np.isfinite(nrmse_mean) else "",
            "severity_0p5": round(float(np.mean([r["severity_curve"]["0.5"]["detection_rate"] for r in per_seed])), 4),
            "severity_1p5": round(float(np.mean([r["severity_curve"]["1.5"]["detection_rate"] for r in per_seed])), 4),
            "severity_2": round(float(np.mean([r["severity_curve"]["2.0"]["detection_rate"] for r in per_seed])), 4),
            "params": max([r["params"] for r in per_seed]),
            "wall_time_s": round(wall, 1),
            "device": device,
            "git_sha": mproto.git_sha(),
            "data_source": "real490",
            "splits_sha256": manifest["manifest_sha256"],
            "ladder_rung": "L2",
            "notes": args.notes,
        }
        mproto.append_dev_log(row)
        print(f"[dev_log] appended row for {row['run_id']}")

    print(f"[done] metrics → {out_path} (wall {wall:.1f}s)")
    return 0


def _rate_delay(inj_scores, inj_meta, idx, tau, k) -> tuple[float, float, list[float]]:
    det, delays = 0, []
    for j in idx:
        fb = mscoring.first_alarm_bin(inj_scores[j], tau, k, inj_meta[j]["onset_bin"])
        if fb >= 0:
            det += 1
            delays.append((fb - inj_meta[j]["onset_bin"]) * 15)
    rate = det / max(1, len(idx))
    return rate, (float(np.median(delays)) if delays else float("nan")), delays


def norm_scale_or_mad(norm: dict | None, scorer) -> np.ndarray:
    """The (T, C) scale array (= MAD × 1.4826) used to size perturbations.

    Deep runs use the train-fit stats sidecar; B1 carries its own stats.
    """
    if norm is not None:
        return norm["scale"]
    return scorer.scale[0] if scorer.scale.ndim == 3 else scorer.scale


def norm_robust_range(norm: dict | None, scorer) -> np.ndarray:
    if norm is not None:
        return norm["robust_range"]
    if scorer.robust_range is not None:
        return scorer.robust_range
    return np.ones(scorer.scale.shape[-1])


def _n_train(manifest: dict) -> int:
    return int(manifest["counts"]["train"])


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())

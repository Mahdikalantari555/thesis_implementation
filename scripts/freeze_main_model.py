#!/usr/bin/env python3
"""Freeze the main monitor model (tasks 7.1/7.2, design D7).

Two modes:

**Freeze** (agent may run after plateau)::

    python scripts/freeze_main_model.py --model masked_transformer --run-id <id>

Selects the best seed by the dev primary metric from the run's
``reports/dev/<run_id>/metrics_<model>.json``, copies the checkpoint to
``checkpoints/<id>_main.pt`` and writes the sidecar ``<id>_main.json``
(architecture config, normalization stats path + arrays, τ, k, hashes, seeds,
dev metrics, parameter count, ``ladder_rung: "L2"``).

**Final test** (user runs ONCE)::

    python scripts/freeze_main_model.py --model masked_transformer --run-id <id> --final-test

Reads the locked test split (the ONLY code path allowed to, spec
test-isolation), scores it causally, reports clean-test FAR + injected
detection rate with bootstrap CIs over fields, compares against
``b1_phasez_causal`` (frozen from the same run), and sets ``ladder_rung`` to
``L3`` in the sidecar only if the D6 target holds (main ≥ B1 + 0.05 with
non-overlapping ranges). Synthetic-perturbation-based L3: real-stress
agreement (L4) remains future work.
"""
from __future__ import annotations

import argparse
import json
import shutil
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

MODEL_ID_MAP = {"masked_transformer": "masked_transformer_main", "gru_ae": "gru_ae_main"}
BAND_NAMES = ["ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["masked_transformer", "gru_ae"])
    p.add_argument("--run-id", required=True)
    p.add_argument("--final-test", action="store_true",
                   help="run the locked final test ONCE (user-only) and set ladder_rung")
    p.add_argument("--seeds", default="13,42,2026")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--notes", default="")
    return p.parse_args(argv)


def load_scorer_and_norm(model: str, seed: int, run_id: str, device: str):
    import torch

    ckpt_dir = mproto.CHECKPOINTS_DEV_DIR / run_id
    norm = minputs.load_norm(mproto.DEV_DIR / run_id / "norm_stats.npz")
    payload = torch.load(ckpt_dir / f"{model}_seed{seed}.pt", map_location=device, weights_only=False)
    cfg = payload["config"]
    torch_model = mmodels.build_model(model, d_in=cfg["d_in"], n_out=cfg["n_out"], cfg={
        k: cfg[k] for k in ("d_model", "n_layers", "d_ff", "n_heads", "dropout") if k in cfg
    })
    torch_model.load_state_dict(payload["state_dict"])
    scorer = mmodels.DeepCausalScorer(model, torch_model, use_doy=bool(cfg.get("use_doy", True)), device=device)
    return scorer, norm, mmodels.count_parameters(torch_model), cfg


def load_b1(run_id: str, seed: int):
    d = np.load(mproto.CHECKPOINTS_DEV_DIR / run_id / f"b1_phasez_causal_seed{seed}.npz", allow_pickle=True)
    b1 = mmodels.B1PhaseZCausal(use_doy=bool(d["use_doy"]), min_n=8)
    b1.median, b1.scale = d["median"], d["scale"]
    b1.robust_range = d["robust_range"] if "robust_range" in d.files else None
    return b1


def score_arrays(scorer, norm, X, M, doy):
    if isinstance(scorer, mmodels.B1PhaseZCausal):
        N, T, C = X.shape
        scores = np.full((N, T), np.nan)
        for i in range(N):
            scores[i] = scorer.score_cycle(X[i], M[i], doy[i, :, 0])
        return scores
    z = minputs.apply_norm(norm, X, M)
    scores, _ = scorer.loo_batch(z, M, doy)
    return scores


def eval_split(scorer, norm, X, M, doy, tau: float, k: int, rounds: int, seed_base: int):
    """Clean FAR + injected detection on one split's arrays."""
    scores = score_arrays(scorer, norm, X, M, doy)
    n = X.shape[0]
    far = float(np.mean([mscoring.alarm_per_cycle(scores[i], tau, k) for i in range(n)]))

    # Perturbation sizing: train stats sidecar for deep runs, B1's own pooled
    # stats otherwise (scale = MAD × 1.4826, identical in K and °C).
    scale = norm["scale"] if norm is not None else (
        scorer.scale[0] if scorer.scale.ndim == 3 else scorer.scale
    )
    det, delays, types, fields_idx = [], [], [], []
    for rnd in range(rounds):
        rng = np.random.default_rng(seed_base + rnd)
        cfg_pert = mpert.PerturbationConfig(n_per_stratum=min(n, 16), durations=(1, 2, 3, 5))
        results = mpert.inject_dataset(X, M, scale, cfg_pert, rng)
        Xinj = np.stack([r.x for r in results])
        Minj = np.stack([r.m for r in results])
        doy_inj = doy[[r.source_row for r in results]]
        s_inj = score_arrays(scorer, norm, Xinj, Minj, doy_inj)
        for j, r in enumerate(results):
            if abs(r.severity - 1.0) > 1e-9:
                continue
            fb = mscoring.first_alarm_bin(s_inj[j], tau, k, r.onset_bin)
            det.append(1 if fb >= 0 else 0)
            delays.append((fb - r.onset_bin) * 15 if fb >= 0 else np.nan)
            types.append(r.pert_type)
            fields_idx.append(int(r.source_row))
    return {
        "far_clean": far,
        "detection_rate_1mad": float(np.mean(det)) if det else float("nan"),
        "median_delay_days": float(np.nanmedian(delays)) if det and np.isfinite(delays).any() else float("nan"),
        "n_clean": n,
        "n_injected_1mad": len(det),
        "det_flags": det,
        "field_rows": fields_idx,
        "types": types,
        "clean_scores": scores,
    }


def bootstrap_ci_fields(det_flags: list[int], field_rows: list[int], field_ids: np.ndarray,
                        reps: int, seed: int) -> tuple[float, float]:
    """95 % CI of the detection rate, resampling FIELDS with replacement."""
    rng = np.random.default_rng(seed)
    rows_by_field: dict[int, list[int]] = {}
    for row, fidx in enumerate(field_rows):
        rows_by_field.setdefault(int(field_ids[fidx]), []).append(row)
    fields = sorted(rows_by_field)
    if len(fields) < 2:
        return float("nan"), float("nan")
    stats = []
    for _ in range(reps):
        sample = rng.choice(fields, size=len(fields), replace=True)
        flags = [f for fid in sample for f in rows_by_field[int(fid)]]
        stats.append(np.mean([det_flags[f] for f in flags]))
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def freeze(args) -> int:
    manifest = msplits.load_splits_locked()
    metrics_path = mproto.DEV_DIR / args.run_id / f"metrics_{args.model}.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Best seed by primary metric (ties → lowest seed); ties and ranges are
    # part of the report, the sidecar carries all seeds' numbers.
    per_seed = metrics["per_seed"]
    best = max(per_seed, key=lambda r: (r["detection_rate_1mad"], -r["seed"]))
    seed = best["seed"]
    device = metrics.get("device", "cpu")

    scorer, norm, params, cfg = load_scorer_and_norm(args.model, seed, args.run_id, device)
    model_id = MODEL_ID_MAP[args.model]

    ckpt_src = mproto.CHECKPOINTS_DEV_DIR / args.run_id / f"{args.model}_seed{seed}.pt"
    ckpt_dst = REPO_ROOT / "checkpoints" / f"{model_id}.pt"
    ckpt_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ckpt_src, ckpt_dst)

    sidecar = {
        "model_id": model_id,
        "family": args.model,
        "formula_id": f"{model_id}:causal_bin_mse",
        "created": time.strftime("%Y-%m-%d"),
        "git_sha": mproto.git_sha(),
        "data_source": "real490 (masked rebuild from stats.zarr)",
        "band_order": BAND_NAMES,
        "n_bins": 36,
        "bin_days": 15,
        "architecture_config": cfg,
        "parameter_count": params,
        "use_doy_channels": bool(cfg.get("use_doy", True)),
        "normalization": {
            "median": norm["median"].tolist(),
            "scale": norm["scale"].tolist(),
            "robust_range": norm["robust_range"].tolist(),
            "norm_stats_file": str((mproto.DEV_DIR / args.run_id / "norm_stats.npz").relative_to(REPO_ROOT)),
            "computed_from": "train split only",
        },
        "threshold": best,  # tau, k, far target/achieved + calibration n
        "persistence_k": int(best["k"]),
        "ladder_rung": "L2",
        "splits_locked_sha256": manifest["manifest_sha256"],
        "seeds": metrics["seeds"],
        "frozen_seed": seed,
        "dev_metrics_summary": {
            "primary_mean": metrics["primary"]["mean"],
            "primary_std": metrics["primary"]["std"],
            "median_delay_days": metrics["median_delay_days"]["mean"],
            "aucpr_partial": per_seed[0]["aucpr_partial"],
            "nrmse_per_band": per_seed[0]["nrmse_per_band"],
        },
        "dev_metrics_file": str(metrics_path.relative_to(REPO_ROOT)),
        "training_run_id": args.run_id,
        "notes": args.notes,
    }
    sidecar["manifest_sha256"] = json.dumps(
        {k: sidecar[k] for k in sorted(sidecar)}, sort_keys=True, default=str
    )
    sidecar_path = REPO_ROOT / "checkpoints" / f"{model_id}.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(f"[freeze] {ckpt_dst}")
    print(f"[freeze] sidecar {sidecar_path} (ladder_rung=L2, seed {seed} "
          f"det@1MAD={best['detection_rate_1mad']:.3f})")
    return 0


def final_test(args) -> int:
    manifest = msplits.load_splits_locked()
    metrics_path = mproto.DEV_DIR / args.run_id / f"metrics_{args.model}.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_seed = metrics["per_seed"]
    best = max(per_seed, key=lambda r: (r["detection_rate_1mad"], -r["seed"]))
    seed, tau, k = best["seed"], best["tau"], best["k"]
    device = "cpu"

    # The ONLY test-split access in the codebase:
    data = minputs.load_masked_cycles()
    sel = mproto.select_splits(data, manifest, splits=("test",), allow_test=True)
    X, M = sel["X"], sel["M"]
    doy = minputs.doy_channels(sel["harvest_days"])
    field_ids = sel["field_ids"]

    scorer, norm, params, cfg = load_scorer_and_norm(args.model, seed, args.run_id, device)
    main_res = eval_split(scorer, norm, X, M, doy, tau, k, args.rounds, seed_base=77_000 + seed)

    b1 = load_b1(args.run_id, seed)
    # B1's τ, k from the same dev calibration (metrics of the b1 run).
    b1_metrics_path = mproto.DEV_DIR / f"b1_{time.strftime('%Y%m%d')}" / "metrics_b1.json"
    b1_tau, b1_k = None, 1
    if b1_metrics_path.exists():
        bm = json.loads(b1_metrics_path.read_text(encoding="utf-8"))
        cand = [r for r in bm["per_seed"] if r["seed"] == seed] or bm["per_seed"]
        b1_tau, b1_k = cand[0]["tau"], cand[0]["k"]
    if b1_tau is None:
        raise FileNotFoundError(
            f"B1 dev metrics not found at {b1_metrics_path} — evaluate the b1 run first."
        )
    b1_res = eval_split(b1, None, X, M, doy, b1_tau, b1_k, args.rounds, seed_base=88_000 + seed)

    ci_main = bootstrap_ci_fields(main_res["det_flags"], main_res["field_rows"], field_ids,
                                  args.bootstrap, seed=seed)
    ci_b1 = bootstrap_ci_fields(b1_res["det_flags"], b1_res["field_rows"], field_ids,
                                args.bootstrap, seed=seed)

    target_gap = main_res["detection_rate_1mad"] - b1_res["detection_rate_1mad"]
    non_overlap = ci_main[0] > ci_b1[1]
    target_met = target_gap >= 0.05 and non_overlap
    ladder = "L3" if target_met else "L2"

    report = {
        "type": "final_test",
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": mproto.git_sha(),
        "model_id": MODEL_ID_MAP[args.model],
        "frozen_seed": seed,
        "n_test_cycles": int(X.shape[0]),
        "n_test_fields": int(np.unique(field_ids).size),
        "date_range": {
            "first": str(sel["start_dates"][0])[:10],
            "last": str(sel["end_dates"][-1])[:10],
        },
        "splits_locked_sha256": manifest["manifest_sha256"],
        "far_target": 0.05,
        "far_clean_main": main_res["far_clean"],
        "far_clean_b1": b1_res["far_clean"],
        "detection_rate_1mad_main": main_res["detection_rate_1mad"],
        "detection_rate_1mad_b1": b1_res["detection_rate_1mad"],
        "median_delay_days_main": main_res["median_delay_days"],
        "bootstrap_ci95_fields_main": list(ci_main),
        "bootstrap_ci95_fields_b1": list(ci_b1),
        "bootstrap_reps": args.bootstrap,
        "target": {"gap_needed": 0.05, "gap_observed": target_gap,
                   "ci_non_overlap": bool(non_overlap), "met": bool(target_met)},
        "ladder_rung_result": ladder,
        "note": (
            "L3 measured as synthetic-perturbation detection on locked test "
            "(unseen years + spatial block). Real-stress agreement (L4) is "
            "future work; 'suspected_stress' naming unchanged."
        ),
    }
    out = REPO_ROOT / "reports" / f"final_test_{MODEL_ID_MAP[args.model]}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Update the frozen sidecar's ladder rung when the target holds.
    model_id = MODEL_ID_MAP[args.model]
    sidecar_path = REPO_ROOT / "checkpoints" / f"{model_id}.json"
    if sidecar_path.exists():
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        sidecar["ladder_rung"] = ladder
        sidecar["final_test_file"] = str(out.relative_to(REPO_ROOT))
        sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(json.dumps({k: report[k] for k in (
        "far_clean_main", "far_clean_b1", "detection_rate_1mad_main",
        "detection_rate_1mad_b1", "bootstrap_ci95_fields_main",
        "bootstrap_ci95_fields_b1", "ladder_rung_result")}, indent=2))
    print(f"[final-test] report → {out}")
    if not target_met:
        print("[final-test] D6 target NOT met — reported as a finding; rung stays L2 "
              "(spec: honest outcome, no goalpost moving).")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.final_test:
        return final_test(args)
    return freeze(args)


if __name__ == "__main__":
    raise SystemExit(main())

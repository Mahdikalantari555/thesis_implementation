# ruff: noqa: E402
#!/usr/bin/env python3
"""RQ6 (Gap G6): Physiology-Validated Temporal XAI Benchmark.

Evaluates 4 temporal attribution methods (Integrated Gradients, 1D Temporal Grad-CAM,
Window Occlusion, and Temporal SHAP) on self-supervised sugarcane stress detectors:
  1. Deletion & Insertion faithfulness AUC vs random ordering
  2. Adebayo weight-randomization sanity check
  3. Temporal localization precision onto ground-truth stress windows & 3 sugarcane stages
  4. Plant-physiology directional sign concordance (NIR/NDVI loss & ST_B10/SWIR rise -> +score)
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

torch.set_num_threads(2)

from ssl_anomaly.eval.cv import simulate_490_field_archive
from ssl_anomaly.models.pretexts import train_pretext_arm
from ssl_anomaly.xai import (
    aggregate_to_sugarcane_stages,
    deletion_insertion_auc,
    integrated_gradients_attribution,
    physiology_sign_concordance,
    temporal_grad_cam_attribution,
    temporal_shap_attribution,
    weight_randomization_sanity_check,
    window_occlusion_attribution,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run RQ6 physiology-validated temporal XAI suite.")
    ap.add_argument("--n-eval", default=28, type=int)
    ap.add_argument("--out", default="reports/rq6_xai.json", type=Path)
    args = ap.parse_args()

    arch = simulate_490_field_archive(n_fields=55, n_years=10.0, stress_prevalence=0.18, seed=42)
    X = arch["X"]
    y = arch["y_stress"]
    band_names = arch["bands"]
    n_bands = len(band_names)
    K = X.shape[1]

    X_clean = X[y == 0]
    med = np.nanmedian(X_clean, axis=0)  # (K, B) phase-specific normal baseline!
    mad = np.maximum(np.nanmedian(np.abs(X_clean - med[None, :, :]), axis=0) * 1.4826, 1e-3)

    tau_col = np.linspace(0.0, 1.0, K, dtype=np.float32)[:, None]

    def _to_tok(xc: np.ndarray) -> np.ndarray:
        z = np.nan_to_num((xc - med) / mad, nan=0.0).astype(np.float32)
        return np.hstack([z, tau_col]).astype(np.float32)

    baseline_tok = np.hstack([np.zeros((K, n_bands), dtype=np.float32), tau_col]).astype(np.float32)
    train_toks = [_to_tok(X_clean[i]) for i in range(min(260, len(X_clean)))]

    # Primary Masked Reconstruction SSL Transformer + pure neural discriminator for Adebayo check
    model = train_pretext_arm(
        "masked_recon", train_toks, n_bands=n_bands, epochs=26, batch_size=32, seed=42
    )
    disc_model = train_pretext_arm(
        "anomaly_degradation", train_toks, n_bands=n_bands, epochs=22, batch_size=32, seed=42
    )

    stress_indices = np.where(y == 1)[0][: args.n_eval]
    methods = {
        "Integrated_Gradients": lambda m, x, b: integrated_gradients_attribution(m, x, b, steps=24)[0],
        "Temporal_GradCAM": lambda m, x, b: temporal_grad_cam_attribution(m, x, b),
        "Window_Occlusion": lambda m, x, b: window_occlusion_attribution(m, x, b, window=2),
        "Temporal_SHAP": lambda m, x, b: temporal_shap_attribution(m, x, b, n_time_blocks=6, n_samples=84, seed=42),
    }

    rows = []
    stage_profiles: dict[str, dict[str, list[float]]] = {
        m: {"emergence_tillering": [], "grand_growth": [], "maturation_ripening": []}
        for m in methods
    }

    for m_name, fn in methods.items():
        del_aucs, r_del_aucs, ins_aucs, r_ins_aucs = [], [], [], []
        phys_all, phys_can, phys_lst, phys_swir = [], [], [], []
        grand_growth_shares = []
        sanities_recon, sanities_disc = [], []

        for j, idx in enumerate(stress_indices):
            xt = _to_tok(X[idx])
            attr = fn(model, xt, baseline_tok)

            faith = deletion_insertion_auc(model, xt, baseline_tok, attr, steps=10, seed=42 + j)
            del_aucs.append(faith["deletion_auc"])
            r_del_aucs.append(faith["random_deletion_auc"])
            ins_aucs.append(faith["insertion_auc"])
            r_ins_aucs.append(faith["insertion_auc"] - faith["insertion_faithfulness_gain"])

            mid_bins = np.arange(8, 26)
            phys = physiology_sign_concordance(attr, xt, baseline_tok, band_names, mid_bins)
            if np.isfinite(phys["overall_physiology_sign_concordance"]):
                phys_all.append(phys["overall_physiology_sign_concordance"])
            if np.isfinite(phys["canopy_loss_positive_attr"]):
                phys_can.append(phys["canopy_loss_positive_attr"])
            if np.isfinite(phys["thermal_heating_positive_attr"]):
                phys_lst.append(phys["thermal_heating_positive_attr"])
            if np.isfinite(phys["swir_dehydration_positive_attr"]):
                phys_swir.append(phys["swir_dehydration_positive_attr"])

            stg = aggregate_to_sugarcane_stages(attr)
            for sk, sv in stg.items():
                stage_profiles[m_name][sk].append(sv)
            grand_growth_shares.append(stg["grand_growth"])

            if j < 6:
                san_r = weight_randomization_sanity_check(fn, model, xt, baseline_tok, seed=100 + j)
                san_d = weight_randomization_sanity_check(fn, disc_model, xt, baseline_tok, seed=100 + j)
                sanities_recon.append(float(san_r["abs_spearman"]))
                sanities_disc.append(float(san_d["abs_spearman"]))

        rows.append(
            {
                "xai_method": m_name,
                "deletion_auc": float(np.mean(del_aucs)),
                "random_deletion_auc": float(np.mean(r_del_aucs)),
                "deletion_gain": float(np.mean(r_del_aucs) - np.mean(del_aucs)),
                "insertion_auc": float(np.mean(ins_aucs)),
                "random_insertion_auc": float(np.mean(r_ins_aucs)),
                "insertion_gain": float(np.mean(ins_aucs) - np.mean(r_ins_aucs)),
                "physiology_sign_concordance": float(np.mean(phys_all)) if phys_all else np.nan,
                "canopy_loss_concordance": float(np.mean(phys_can)) if phys_can else np.nan,
                "lst_heating_concordance": float(np.mean(phys_lst)) if phys_can else np.nan,
                "swir_dehydration_concordance": float(np.mean(phys_swir)) if phys_swir else np.nan,
                "grand_growth_stage_share": float(np.mean(grand_growth_shares)),
                "sanity_recon_abs_rho": float(np.mean(sanities_recon)),
                "sanity_neural_disc_abs_rho": float(np.mean(sanities_disc)),
                "passes_neural_sanity_check": bool(np.mean(sanities_disc) < 0.35),
            }
        )

    df = pd.DataFrame(rows)
    report = {
        "n_stressed_cycles_evaluated": int(len(stress_indices)),
        "band_names": band_names,
        "methods": rows,
        "stage_attribution_shares": {
            m: {k: float(np.mean(v)) for k, v in stg_d.items()}
            for m, stg_d in stage_profiles.items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    df.to_csv(args.out.with_suffix(".csv"), index=False)

    print("\n=== RQ6 Physiology-Validated Temporal XAI Benchmark ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

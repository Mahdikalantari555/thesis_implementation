#!/usr/bin/env python
"""Q8 solver-equivalence decomposition (openspec ``oracle-floor-audit`` 3.1).

The SAVI families detect 2–5× better than NDVI at S6 (finding 25) with the
mechanism unresolved. The consultant's arithmetic assumed canopy baselines
(N=0.40, R=0.05); this audit MEASURES the two solvers on real store patches:

* per-channel in-field-mean realized deltas (Δred, Δnir, Δndvi) of the
  bands-consistent solver at equal nominal Δ, per index family;
* the band-magnitude ratio ρ = ‖(ΔR, ΔN)_savi‖ / ‖(ΔR, ΔN)_ndvi‖ per level;
* the NDVI-equivalent Δ* = Δ·ρ — the nominal NDVI drop whose realized band
  perturbation matches the SAVI cell's (the consultant's "matched-NDVI" rung).

Pure solver math on raw patches — no model, no scoring, CPU minutes. The
pre-registered confirmatory detection cell (matched-Δ run through the
calibrated sweep on the locked panel) is task 3.2 and runs AFTER the E2
decision, per design.

Model-free and read-only w.r.t. the locked panels: cycles come from V.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

CHANNEL_ORDER = ("ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst")
CH = {name: i for i, name in enumerate(CHANNEL_ORDER)}
BAND_CHANNELS = (CH["red"], CH["nir"], CH["ndvi"])


def band_magnitude(delta_red: float, delta_nir: float) -> float:
    """L2 magnitude of the realized (Δred, Δnir) pair (raw reflectance units)."""
    return float(np.hypot(delta_red, delta_nir))


def ndvi_equivalent(delta: float, rho: float) -> float:
    """Nominal NDVI drop whose realized band pair matches the SAVI cell's."""
    return float(delta * rho)


def per_channel_field_delta(inj_x: np.ndarray, raw: np.ndarray, obs: np.ndarray,
                            in_field: np.ndarray) -> np.ndarray:
    """(C,) in-field observed-pixel mean of (injected − raw), per channel.

    NaN (never-acquired) pixels are masked BEFORE multiplication — NaN·0 is
    NaN, and a masked sum must not inherit them.
    """
    m = np.asarray(obs, dtype=bool) & np.asarray(in_field, dtype=bool)[None]
    diff = (np.asarray(inj_x, dtype=np.float64)
            - np.asarray(raw, dtype=np.float64))
    diff = np.where(m, diff, 0.0)
    num = diff.sum(axis=(0, 2, 3))
    den = np.maximum(m.sum(axis=(0, 2, 3)), 1)
    return num / den


def summarise(rows: list[dict]) -> dict:
    """Median per-channel deltas + ρ + Δ* from per-cycle rows of both families."""
    out = {}
    for fam in ("ndvi", "savi"):
        d = {ch: float(np.median([r[fam][ch] for r in rows])) for ch in BAND_CHANNELS}
        out[fam] = {"delta_red": d[CH["red"]], "delta_nir": d[CH["nir"]],
                    "delta_ndvi_channel": d[CH["ndvi"]],
                    "band_magnitude": band_magnitude(d[CH["red"]], d[CH["nir"]])}
    rho = None
    if out["ndvi"]["band_magnitude"] > 1e-12:
        rho = out["savi"]["band_magnitude"] / out["ndvi"]["band_magnitude"]
    out["rho_savi_over_ndvi"] = float(rho) if rho is not None else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--store", default="data/patch_runs/v3/patches.zarr")
    ap.add_argument("--splits", default="data/processed/splits_locked_v2.json")
    ap.add_argument("--levels", default="S4,S5,S6")
    ap.add_argument("--n-cycles", type=int, default=60)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--shape", default="persistent", choices=("persistent", "acute"))
    ap.add_argument("--metrics-dir", default="reports/dev/patch_full_v2_ep10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ssl_anomaly.io import patches as P
    from ssl_anomaly.monitor import patch_eval as pE
    from ssl_anomaly.monitor import patch_perturbations as ppert
    from ssl_anomaly.monitor import perturbations as mpert
    from ssl_anomaly.monitor import severity_table as sev
    import audit_oracle_floor as AOF

    levels = [s.strip() for s in args.levels.split(",") if s.strip()]
    deltas = {lv: float(next(l.delta_ndvi for l in sev.SEVERITY_TABLE
                             if l.level_id == lv)) for lv in levels}
    duration = (int(sev.PERSISTENT_DURATION_BINS) if args.shape == "persistent"
                else int(next(l.duration_bins for l in sev.SEVERITY_TABLE
                              if l.level_id == levels[-1])))

    manifest = json.loads((REPO_ROOT / args.splits).read_text(encoding="utf-8"))
    store_path = Path(args.store) if Path(args.store).is_absolute() else REPO_ROOT / args.store
    root = P.open_store(store_path)
    V = AOF.build_v_pool(manifest, root)
    rng = np.random.default_rng(args.seed)
    pick = rng.permutation(len(V["cycle_keys"]))[:args.n_cycles]
    fids = V["field_ids"][pick]
    cids = V["cycle_ids"][pick]
    print(f"cycles sampled from V: {len(fids)} (seed {args.seed})", flush=True)

    # per-level rows: {family: {channel: mean delta}}
    rows = {lv: [] for lv in levels}
    batch = pE.patch_batch_for_cycles(root, fids, cids,
                                      norm=P.load_patch_norm(
                                          REPO_ROOT / "checkpoints" / "patch"
                                          / "patch_full_v2_ep10"
                                          / "patch_pixelwise_seed13"
                                          / "norm_stats.json"))
    raw_all = np.asarray(batch["raw"])
    obs_all = np.asarray(batch["obs"], dtype=bool)
    inf_all = np.asarray(batch["in_field"], dtype=bool)
    T = raw_all.shape[1]
    for i in range(raw_all.shape[0]):
        onset = int(mpert.phase_start_bin(
            mpert.PHASES[i % len(mpert.PHASES)], T))
        for lv in levels:
            row = {}
            for fam in ("ndvi", "savi"):
                inj = ppert.inject_patch_drop(
                    raw_all[i].astype(np.float32), obs_all[i], index=fam,
                    delta=deltas[lv], onset_bin=onset, shape=args.shape,
                    duration=duration, in_field=inf_all[i],
                    injection_basis=ppert.BASIS_BANDS_CONSISTENT)
                row[fam] = per_channel_field_delta(inj.x, raw_all[i].astype(np.float64),
                                                   obs_all[i], inf_all[i])
            rows[lv].append(row)

    report = {
        "git_sha": AOF.subprocess_git_sha(),
        "n_cycles": int(len(fids)), "shape": args.shape,
        "levels": levels, "deltas": deltas, "seed": args.seed,
        "per_level": {}, "measured_detection_reference": {},
    }
    metrics_path = REPO_ROOT / args.metrics_dir / \
        "metrics_patch_pixelwise_sev_patch_pixelwise_s13.json"
    if metrics_path.exists():
        cells = json.loads(metrics_path.read_text(encoding="utf-8"))["per_seed"][0]["cells"]
        for lv in levels:
            report["measured_detection_reference"][lv] = {
                idx: float(cells[f"{lv}/{args.shape}/{idx}"]["detection_rate"])
                for idx in ("ndvi", "savi")}
    for lv in levels:
        summ = summarise(rows[lv])
        summ["ndvi_equivalent_delta"] = ndvi_equivalent(
            deltas[lv], summ["rho_savi_over_ndvi"])
        report["per_level"][lv] = summ
        print(f"{lv} (Δ={deltas[lv]}): "
              f"ndvi (ΔR,ΔN)=({summ['ndvi']['delta_red']:+.4f}, "
              f"{summ['ndvi']['delta_nir']:+.4f}) | "
              f"savi (ΔR,ΔN)=({summ['savi']['delta_red']:+.4f}, "
              f"{summ['savi']['delta_nir']:+.4f}) | "
              f"ρ={summ['rho_savi_over_ndvi']:.3f} → "
              f"NDVI-equivalent Δ*={summ['ndvi_equivalent_delta']:.3f}", flush=True)

    out = Path(args.out) if args.out else (REPO_ROOT / args.metrics_dir
                                           / "matched_ndvi_equivalence.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report → {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

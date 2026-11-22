#!/usr/bin/env python
"""MDE-curve deliverable (openspec ``oracle-floor-audit`` task 5.1).

Fits a floor-anchored probit dose–response per arm × family from the six
committed fixed-protocol metrics JSONs (finding 25):

    p(Δ) = floor + (1 − floor) · Φ(a + b·Δ)

* ``floor`` is the measured low-severity plateau (S1–S3 rate, i.e. the
  post-onset false-alarm floor), fixed during the fit;
* Δ*(0.8) — the severity whose detection reaches the 0.8 MDM80 target — is
  reported with a parametric-bootstrap CI (binomial resampling at the fitted
  curve, 2,000 draws);
* the pre-registered decision-relevant drop Δ_req = 0.165 (top of the
  realistic-stress range) turns the curve into a gate: Δ*(0.8) ≤ Δ_req → PASS;
* a Δ* beyond the largest measured level (0.40) is censored: reported as
  ">S6" with the extrapolated value in a separate field explicitly labelled
  a model, never a measurement.

Read-only on the committed metrics; CPU seconds. (Field-blocked bootstrap
needs per-injection logging, which lands in the same change for FUTURE runs.)
"""
from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

DELTA_REQ = 0.165          # pre-registered decision-relevant drop (index units)
MAX_MEASURED = 0.40        # largest ladder step (S6) — beyond it: extrapolation
FAMILIES = ("acute/ndvi", "acute/savi", "persistent/ndvi", "persistent/savi")
SEEDS = (13, 42, 2026)
ARCHS = ("patch_pixelwise", "patch_cnn_gru")


def probit(p: float) -> float:
    """Inverse standard-normal CDF (scipy.special.ndtri)."""
    from scipy.special import ndtri
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return float(ndtri(p))


def fit_floor_probit(deltas: np.ndarray, rates: np.ndarray, floor: float,
                     n: int) -> tuple[float, float]:
    """Least-binomial-deviance fit of Φ(a + b·Δ) with a fixed floor.

    Returns (a, b) — NaN when no increasing dose-response fits (the honest
    censored outcome for flat families). The likelihood is on COUNTS (rates
    are fractions; the count scale keeps the deviance dominated by the data,
    not by a degenerate p→floor solution).
    """
    from scipy.optimize import minimize

    counts = np.asarray(rates, dtype=float) * n
    p_lo = probit((max(float(rates[0]), floor) + 0.005 - floor) / (1 - floor))
    p_hi = probit((min(float(rates[-1]), 0.999) - floor) / (1 - floor))
    b0 = (p_hi - p_lo) / max(float(deltas[-1] - deltas[0]), 1e-6)
    a0 = p_hi - b0 * float(deltas[-1])

    def neg_dev(params):
        a, b = params
        p = np.clip(floor + (1.0 - floor) * _cdf(a + b * deltas), 1e-9, 1 - 1e-9)
        ll = counts * np.log(p) + (n - counts) * np.log(1 - p)
        return -float(ll.sum())

    res = minimize(neg_dev, x0=[a0, max(b0, 0.1)], method="Nelder-Mead",
                   options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 4000})
    a, b = float(res.x[0]), float(res.x[1])
    if b <= 0:
        # a non-increasing dose-response: no usable curve
        return float("nan"), float("nan")
    return a, b


def _cdf(x):
    from scipy.special import ndtr
    return ndtr(x)


def delta_at_power(a: float, b: float, floor: float, target: float = 0.8) -> float:
    """Δ with p(Δ) = target, or NaN when the curve never reaches it."""
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return float("nan")
    z = probit((target - floor) / max(1.0 - floor, 1e-9))
    d = (z - a) / b
    return float(d) if d > 0 else float("nan")


def bootstrap_ci(deltas, rates, n, floor, a0, b0, target=0.8,
                 draws: int = 2000, seed: int = 2026) -> tuple[float, float]:
    """Parametric bootstrap: resample binomial counts at the fitted curve."""
    if not (np.isfinite(a0) and np.isfinite(b0) and b0 > 0):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        p = np.clip(floor + (1 - floor) * _cdf(a0 + b0 * deltas), 1e-9, 1 - 1e-9)
        k = rng.binomial(n, p)
        r = k / n
        a, b = fit_floor_probit(deltas, r, floor, n)
        d = delta_at_power(a, b, floor, target)
        if np.isfinite(d):
            vals.append(d)
    if not vals:
        return float("nan"), float("nan")
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def build_curve(metrics_by_seed: dict, family: str) -> dict:
    cells = metrics_by_seed[SEEDS[0]]["per_seed"][0]["cells"]
    levels = sorted({int(k.split("/")[0][1:]) for k in cells}, )
    deltas = np.array([float(next(l.delta_ndvi for l in _sev_levels()
                                   if l.level_id == f"S{lv}")) for lv in levels])
    n = int(cells[f"S{levels[0]}/{family}"]["n"])
    rates = np.array([float(np.mean([_rate(metrics_by_seed[seed], f"S{lv}/{family}")
                                     for seed in SEEDS])) for lv in levels])

    plateau = float(np.mean(rates[:3]))   # S1–S3 plateau = the FAR floor
    floor = float(np.clip(plateau, 0.005, 0.5))
    a, b = fit_floor_probit(deltas, rates, floor, n)
    d80 = delta_at_power(a, b, floor, 0.8)
    lo, hi = bootstrap_ci(deltas, rates, n, floor, a, b)
    censored = (not np.isfinite(d80)) or d80 > MAX_MEASURED
    return {"family": family, "n_per_cell": n,
            "deltas": [float(d) for d in deltas],
            "rates_pooled": [float(r) for r in rates],
            "floor": floor, "probit_a": a, "probit_b": b,
            "delta_star_080": d80 if not censored else None,
            "delta_star_080_extrapolated": (float(d80) if np.isfinite(d80)
                                            else None),
            "bootstrap_ci95": [lo, hi],
            "censored_gt_s6": bool(censored),
            "delta_req": DELTA_REQ,
            "gate": ("PASS" if (np.isfinite(d80) and d80 <= DELTA_REQ)
                     else "FAIL (Δ*(0.8) above the decision-relevant drop)")}


def _sev_levels():
    from ssl_anomaly.monitor import severity_table as sev
    return sev.SEVERITY_TABLE


def _rate(metrics: dict, key: str) -> float:
    return float(metrics["per_seed"][0]["cells"][key]["detection_rate"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metrics-dir", default="reports/dev/patch_full_v2_ep10")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {"delta_req": DELTA_REQ, "max_measured": MAX_MEASURED,
              "bootstrap": "parametric (cycle-level); field-blocked needs the "
                           "per-injection logging that lands in this change",
              "arms": {}}
    for arch in ARCHS:
        metrics_by_seed = {}
        for seed in SEEDS:
            path = REPO_ROOT / args.metrics_dir / \
                f"metrics_{arch}_sev_{arch}_s{seed}.json"
            if not path.exists():
                raise SystemExit(f"metrics not found: {path}")
            metrics_by_seed[seed] = json.loads(path.read_text(encoding="utf-8"))
        report["arms"][arch] = {fam: build_curve(metrics_by_seed, fam)
                                for fam in FAMILIES}

    out = Path(args.out) if args.out else (REPO_ROOT / args.metrics_dir
                                           / "mde_curve.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report → {out}\n", flush=True)
    for arch, fams in report["arms"].items():
        print(f"=== {arch} ===")
        for fam, c in fams.items():
            d80 = c["delta_star_080"]
            label = "censored >S6" if c["censored_gt_s6"] else f"Δ*(0.8)={d80:.3f}"
            print(f"  {fam:18s} floor={c['floor']:.3f} {label} "
                  f"(extrapolated={c['delta_star_080_extrapolated']}) "
                  f"CI=[{c['bootstrap_ci95'][0]:.3f}, {c['bootstrap_ci95'][1]:.3f}] "
                  f"→ {c['gate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

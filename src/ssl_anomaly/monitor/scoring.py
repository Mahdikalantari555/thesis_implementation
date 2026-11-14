"""Causal scoring, alarm rule and FAR calibration (design D5, spec §Threshold).

- Per-bin score: causal prediction error of the current bin from earlier bins
  (observed-mask-weighted MSE in z-space for deep models, mean z² for B1).
- Alarm rule: ``suspected_stress`` when the score exceeds τ at k consecutive
  observed bins, k ∈ {1, 2} chosen on dev.
- Threshold: τ calibrated on CLEAN dev cycles so at most 5 % of them raise any
  alarm over their whole length (cycle-level false-alarm rate).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAR_TARGET = 0.05


@dataclass
class Threshold:
    tau: float
    k: int
    far_target: float
    far_achieved: float
    calibration_n_cycles: int

    def to_dict(self) -> dict:
        return {
            "tau": float(self.tau), "persistence_k": int(self.k),
            "far_target": float(self.far_target), "far_achieved": float(self.far_achieved),
            "calibration_n_cycles": int(self.calibration_n_cycles),
        }


def alarm_per_cycle(
    scores: np.ndarray,      # (T,) per-bin scores, NaN where unobserved
    tau: float,
    k: int,
) -> bool:
    """True when k consecutive observed bins exceed τ anywhere in the cycle."""
    s = np.asarray(scores, dtype=np.float64)
    obs = np.isfinite(s)
    above = obs & (s > tau)
    if k <= 1:
        return bool(above.any())
    run = 0
    for a in above:
        run = run + 1 if a else 0
        if run >= k:
            return True
    return False


def first_alarm_bin(
    scores: np.ndarray,
    tau: float,
    k: int,
    onset_bin: int = 0,
) -> int:
    """First bin whose alarm run (k consecutive observed bins > τ) completes,
    searching from ``onset_bin``; -1 when no alarm."""
    s = np.asarray(scores, dtype=np.float64)
    obs = np.isfinite(s)
    above = obs & (s > tau)
    T = len(s)
    for b in range(max(0, onset_bin), T):
        if k <= 1:
            if above[b]:
                return b
        else:
            if b + k <= T and bool(above[b:b + k].all()):
                return b
    return -1


def calibrate_threshold(
    clean_scores: list[np.ndarray],  # per-cycle (T,) score arrays
    k: int,
    far_target: float = FAR_TARGET,
) -> tuple[float, float]:
    """Find the smallest τ with cycle-level FAR ≤ far_target, by binary search
    over the observed score values (alarm monotone decreasing in τ)."""
    if not clean_scores:
        raise ValueError("No clean scores for calibration")
    candidates = np.unique(np.concatenate([s[np.isfinite(s)] for s in clean_scores]))
    if candidates.size == 0:
        raise ValueError("All clean scores are NaN")
    lo, hi = 0, candidates.size - 1

    def far_at(tau: float) -> float:
        flags = [alarm_per_cycle(s, tau, k) for s in clean_scores]
        return float(np.mean(flags))

    # Smallest τ (from the candidate score values) achieving FAR ≤ target.
    best_tau = float(candidates[hi]) + 1e-9  # beyond max → FAR 0
    while lo <= hi:
        mid = (lo + hi) // 2
        if far_at(float(candidates[mid])) <= far_target:
            best_tau = float(candidates[mid])
            hi = mid - 1
        else:
            lo = mid + 1
    achieved = far_at(best_tau)
    return best_tau, achieved


def choose_threshold(
    clean_scores: list[np.ndarray],
    injected_scores: list[np.ndarray],
    injected_onsets: list[int],
    far_target: float = FAR_TARGET,
) -> tuple[Threshold, dict]:
    """Pick k ∈ {1, 2} by detection rate at FAR ≤ target on injected dev cycles.

    Tie (equal detection rate) → smaller k (simpler). The second dict carries
    the per-k diagnostics for the metrics report.
    """
    per_k: dict[str, dict] = {}
    best: Threshold | None = None
    for k in (1, 2):
        tau, achieved = calibrate_threshold(clean_scores, k, far_target)
        detected = [
            first_alarm_bin(s, tau, k, onset) >= 0
            for s, onset in zip(injected_scores, injected_onsets)
        ]
        det_rate = float(np.mean(detected)) if detected else 0.0
        delays = [
            (first_alarm_bin(s, tau, k, onset) - onset) * 15
            for s, onset in zip(injected_scores, injected_onsets)
            if first_alarm_bin(s, tau, k, onset) >= 0
        ]
        per_k[f"k{k}"] = {
            "tau": float(tau), "far_achieved": float(achieved),
            "detection_rate": det_rate,
            "median_delay_days": float(np.median(delays)) if delays else None,
        }
        if best is None or det_rate > per_k[f"k{best.k}"]["detection_rate"]:
            best = Threshold(
                tau=tau, k=k, far_target=far_target, far_achieved=achieved,
                calibration_n_cycles=len(clean_scores),
            )
    assert best is not None
    return best, per_k


def masked_nrmse_per_band(
    pred: np.ndarray,   # (N, T, C) predictions at every bin
    truth: np.ndarray,  # (N, T, C) normalized (or raw) values
    mask: np.ndarray,   # (N, T, C) bool
    robust_range: np.ndarray,  # (C,) raw-unit denominator (or ones in z-space)
) -> np.ndarray:
    """Masked NRMSE per band: RMSE over observed bins ÷ robust range."""
    N, T, C = truth.shape
    out = np.full(C, np.nan)
    for c in range(C):
        sel = mask[:, :, c]
        if not sel.any():
            continue
        err = (pred[:, :, c] - truth[:, :, c])[sel]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        denom = float(robust_range[c]) if robust_range is not None else 1.0
        out[c] = rmse / max(denom, 1e-9)
    return out


def partial_auc_pr(
    clean_max_scores: np.ndarray,    # (N_clean,) ranking scores
    injected_max_scores: np.ndarray,  # (N_inj,)
) -> float:
    """AUC-PR (average precision) for the injected class from max-score ranking.

    Standard step-wise average precision: sort by score descending, then
    AUC-PR = Σ_i (recall_i − recall_{i−1}) · precision_i. AUC-PR (not ROC) is
    required by the dev protocol because injected:clean is class-imbalanced.
    """
    y = np.concatenate([np.zeros(len(clean_max_scores)), np.ones(len(injected_max_scores))])
    s = np.concatenate([clean_max_scores, injected_max_scores])
    n_pos = float(y.sum())
    if n_pos == 0 or n_pos == len(y):
        return float("nan")
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / n_pos
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))

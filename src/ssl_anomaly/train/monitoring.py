"""Representation-quality monitors and frozen-encoder probes (task 1.1b).

Design.md §4.1 r2 monitoring vector, representation part:

* **collapse signals** — effective rank (entropy of normalized singular
  values), per-dimension std, dims-above-floor, off-diagonal covariance
  energy (VICReg-style) [new monitors];
* **frozen-encoder quality** (fixed budget, validation split only) — linear
  probe of season phase and next-observation NDVI, same-field-same-phase kNN
  retrieval, gap reconstruction from the frozen latent;
* **stability** — linear CKA against the previous probe checkpoint and top-K
  score-rank correlation.

Pure NumPy, closed-form ridge (no epochs, no GPU, no RNG dependence beyond
the deterministic sample caps) so a probe never distorts the training budget
it is meant to measure.
"""

from __future__ import annotations

import numpy as np

#: fixed probe budget (pre-registered; probes must stay cheap on CPU)
PROBE_MAX_SAMPLES = 512
PROBE_RIDGE_ALPHA = 1e-2


# --------------------------- embedding health ---------------------------------


def effective_rank(z: np.ndarray) -> float:
    """exp(entropy) of the covariance eigen-spectrum (participation ratio form).

    Robust to small n (unlike ``benchmarking.collapse.latent_stats`` which
    raises below n >= max(8, d/2) — probes run on small validation windows).
    """
    a = np.asarray(z, dtype=np.float64)
    if a.ndim == 3:
        a = a.reshape(-1, a.shape[-1])
    a = a[np.isfinite(a).all(axis=1)]
    if a.shape[0] < 2:
        return float("nan")
    a = a - a.mean(axis=0, keepdims=True)
    # singular values of the centered data; spectrum of the covariance
    s = np.linalg.svd(a, compute_uv=False)
    lam = (s ** 2)
    lam = np.clip(lam, 0.0, None)
    total = lam.sum()
    if total <= 1e-12:
        return 1.0  # all-constant embeddings: rank 1 (collapsed)
    p = lam / total
    p = p[p > 1e-12]
    ent = float(-(p * np.log(p)).sum())
    return float(np.exp(ent))


def dims_above_floor(z: np.ndarray, floor: float = 1e-3) -> dict[str, float]:
    """Fraction of embedding dimensions with std above ``floor`` + extremes."""
    a = np.asarray(z, dtype=np.float64)
    if a.ndim == 3:
        a = a.reshape(-1, a.shape[-1])
    a = a[np.isfinite(a).all(axis=1)]
    if a.shape[0] < 2:
        return {"dims_above_floor_frac": float("nan"), "per_dim_std_min": float("nan"),
                "n_dims": int(a.shape[1]) if a.ndim == 2 else 0}
    std = a.std(axis=0)
    return {
        "dims_above_floor_frac": float((std > floor).mean()),
        "per_dim_std_min": float(std.min()),
        "per_dim_std_median": float(np.median(std)),
        "n_dims": int(std.shape[0]),
    }


def offdiag_cov_energy(z: np.ndarray) -> float:
    """VICReg-style off-diagonal covariance energy (fraction of total)."""
    a = np.asarray(z, dtype=np.float64)
    if a.ndim == 3:
        a = a.reshape(-1, a.shape[-1])
    a = a[np.isfinite(a).all(axis=1)]
    if a.shape[0] < 3:
        return float("nan")
    c = np.cov(a, rowvar=False)
    c = np.atleast_2d(c)
    d = c.shape[0]
    total = float(np.abs(c).sum())
    if total <= 1e-12:
        return 0.0
    off = float(np.abs(c).sum() - np.abs(np.diag(c)).sum())
    return off / (off + np.abs(np.diag(c)).sum())


def rep_health(z: np.ndarray, *, std_floor: float = 1e-3) -> dict[str, float]:
    """One dict with all collapse signals for the monitor/log."""
    out: dict[str, float] = {
        "effective_rank": effective_rank(z),
        **dims_above_floor(z, floor=std_floor),
        "offdiag_cov_energy": offdiag_cov_energy(z),
    }
    return out


# ----------------------------- frozen probes ----------------------------------


def _cap_rows(*arrays: np.ndarray, max_samples: int = PROBE_MAX_SAMPLES):
    n = arrays[0].shape[0]
    if n <= max_samples:
        return arrays
    idx = np.linspace(0, n - 1, max_samples).astype(int)  # deterministic, no RNG
    return tuple(a[idx] for a in arrays)


def _ridge_r2(z: np.ndarray, y: np.ndarray, alpha: float = PROBE_RIDGE_ALPHA) -> float:
    """Closed-form ridge probe R² (multi-target allowed)."""
    zc, yc = z, y
    if z.shape[0] < 5:
        return float("nan")
    zt = np.hstack([zc, np.ones((zc.shape[0], 1))])
    reg = np.eye(zt.shape[1]) * alpha
    reg[-1, -1] = 0.0  # don't penalize the intercept
    try:
        w = np.linalg.solve(zt.T @ zt + reg, zt.T @ yc)
    except np.linalg.LinAlgError:
        return float("nan")
    pred = zt @ w
    ss_res = float(((yc - pred) ** 2).sum())
    ss_tot = float(((yc - yc.mean(axis=0)) ** 2).sum())
    if ss_tot <= 1e-12:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def phase_linear_probe(z: np.ndarray, phase: np.ndarray) -> dict[str, float]:
    """Linear-probe R² for season phase (design §12 item 7: primary metric).

    ``phase`` may be (N,) scalar phase (e.g. cycle fraction) or (N, 2)
    sin/cos-DOY targets; R² is reported per target and averaged.
    """
    z, y = _cap_rows(z, np.asarray(phase, dtype=np.float64))
    y = y.reshape(z.shape[0], -1)
    r2 = _ridge_r2(z, y)
    return {"probe_phase_r2": float(r2)}


def next_ndvi_probe(z: np.ndarray, next_ndvi: np.ndarray) -> dict[str, float]:
    """Linear-probe R² for the next observation's NDVI (forecast quality)."""
    z, y = _cap_rows(z, np.asarray(next_ndvi, dtype=np.float64))
    y = y.reshape(z.shape[0], -1)
    return {"probe_next_ndvi_r2": float(_ridge_r2(z, y))}


def same_field_same_phase_knn(
    z: np.ndarray,
    field_ids: np.ndarray,
    phase_bins: np.ndarray,
    *,
    k: int = 1,
) -> dict[str, float]:
    """Retrieval R@k: is the k-th nearest neighbour in the same field AND phase?

    Chance level = the largest (field × phase) group share; both are reported
    so a retrieval number is never quoted without its baseline.
    """
    z, f, p = _cap_rows(
        z, np.asarray(field_ids), np.asarray(phase_bins)
    )
    n = z.shape[0]
    if n < 8:
        return {"probe_knn_r_at_1": float("nan"), "probe_knn_chance": float("nan"), "n": n}
    zn = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    sim = zn @ zn.T
    np.fill_diagonal(sim, -np.inf)
    order = np.argsort(-sim, axis=1)[:, : max(1, int(k))]
    hit = (f[order[:, 0]] == f) & (p[order[:, 0]] == p)
    keys, counts = np.unique(list(zip(f.tolist(), p.tolist())), axis=0, return_counts=True)
    chance = float(counts.max()) / n
    return {
        "probe_knn_r_at_1": float(hit.mean()),
        "probe_knn_chance": chance,
        "n": int(n),
    }


def gap_reconstruction_probe(
    z: np.ndarray,
    gap_targets: np.ndarray,
    *,
    baseline_mean: np.ndarray | None = None,
) -> dict[str, float]:
    """L1-a gap reconstruction from the frozen latent vs a mean-predictor.

    ``gap_targets`` holds the masked-out features of the validation windows;
    the probe ridge-decodes them from the frozen latent and reports the MSE
    ratio to the mean-predictor baseline (< 1 = the latent carries gap info).
    """
    z, y = _cap_rows(z, np.asarray(gap_targets, dtype=np.float64))
    y = y.reshape(z.shape[0], -1)
    r2 = _ridge_r2(z, y)
    zt = np.hstack([z, np.ones((z.shape[0], 1))])
    reg = np.eye(zt.shape[1]) * PROBE_RIDGE_ALPHA
    reg[-1, -1] = 0.0
    try:
        w = np.linalg.solve(zt.T @ zt + reg, zt.T @ y)
        pred = zt @ w
        mse_probe = float(((y - pred) ** 2).mean())
    except np.linalg.LinAlgError:
        mse_probe = float("nan")
    mean = baseline_mean if baseline_mean is not None else y.mean(axis=0, keepdims=True)
    mse_mean = float(((y - mean) ** 2).mean())
    ratio = mse_probe / mse_mean if mse_mean > 1e-12 else float("nan")
    return {"probe_gap_r2": float(r2), "probe_gap_mse_ratio": float(ratio)}


# ------------------------------- stability ------------------------------------


def linear_cka(z_a: np.ndarray, z_b: np.ndarray) -> float:
    """Linear CKA between two embedding sets (same n, dims may differ)."""
    a = np.asarray(z_a, dtype=np.float64)
    b = np.asarray(z_b, dtype=np.float64)
    if a.ndim == 3:
        a = a.reshape(-1, a.shape[-1])
    if b.ndim == 3:
        b = b.reshape(-1, b.shape[-1])
    n = min(a.shape[0], b.shape[0])
    a, b = a[:n], b[:n]
    if n < 3:
        return float("nan")
    a = a[np.isfinite(a).all(axis=1)]
    b = b[np.isfinite(b).all(axis=1)]
    n = min(a.shape[0], b.shape[0])
    if n < 3:
        return float("nan")
    a, b = a[:n] - a[:n].mean(0, keepdims=True), b[:n] - b[:n].mean(0, keepdims=True)
    num = np.linalg.norm(a.T @ b, "fro") ** 2
    den = np.linalg.norm(a.T @ a, "fro") * np.linalg.norm(b.T @ b, "fro")
    return float(num / den) if den > 1e-12 else float("nan")


def topk_rank_stability(scores_a: np.ndarray, scores_b: np.ndarray, *, k: int = 10) -> dict[str, float]:
    """Spearman rank correlation of the top-K scores across two evaluations
    plus their Jaccard overlap (design §2.2 stability metrics)."""
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    n = min(a.size, b.size)
    if n < k + 1:
        return {"topk_spearman": float("nan"), "topk_jaccard": float("nan"), "k": int(k)}
    a, b = a[:n], b[:n]
    ka = set(np.argsort(-a)[:k].tolist())
    kb = set(np.argsort(-b)[:k].tolist())
    ra = np.argsort(np.argsort(-a))
    rb = np.argsort(np.argsort(-b))
    num = float(((ra - ra.mean()) * (rb - rb.mean())).sum())
    den = float(np.sqrt(((ra - ra.mean()) ** 2).sum()) * ((rb - rb.mean()) ** 2).sum() ** 0.5)
    rho = num / den if den > 1e-12 else float("nan")
    return {
        "topk_spearman": rho,
        "topk_jaccard": float(len(ka & kb) / len(ka | kb)),
        "k": int(k),
    }

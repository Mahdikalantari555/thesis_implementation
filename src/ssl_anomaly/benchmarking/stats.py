"""Inference layer: paired bootstrap, Holm–BH, Cliff's δ, Friedman–Nemenyi (tasks 8.x).

Four rules, all of them here so no analysis script can quietly deviate:

**Clusters, not rows.** Cycles from one farm and bins from one event are not
independent observations, so the bootstrap resamples *farms* (and, for event-level
metrics, events). An i.i.d. row bootstrap on a 24-cycle-per-farm panel would
produce confidence intervals several times too narrow, and the difference between
two architectures would look significant when it is only the within-farm
correlation.

**Paired.** The two arms are compared on the *same* resampled clusters, so the
interval is on the difference. An unpaired interval on each arm's own spread would
answer a different question and inflate every claim of separation.

**Effect size beside the p-value.** Cliff's δ (and its common-language translation)
is reported with a bootstrap CI; a p-value of 1e-9 on 3,000 correlated cycles with
δ = 0.03 is a statement about sample size.

**Correction is applied over a declared family.** Holm for the confirmatory set,
Benjamini–Hochberg for the screening set, and the family's membership is written
into the output so a reader can see what was corrected against what.

Friedman + Nemenyi handle the many-arm omnibus question. Where the number of arms
is smaller than the number of blocks the post-hoc critical difference is meaningless,
and the function says so instead of printing a table of asterisks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

DEFAULT_B = 2000
MIN_B = 1000


def cluster_indices(groups: np.ndarray) -> dict[int, np.ndarray]:
    g = np.asarray(groups)
    return {int(k): np.flatnonzero(g == k) for k in np.unique(g)}


def _resample(groups: dict[int, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    keys = np.array(sorted(groups), dtype=np.int64)
    pick = rng.choice(keys, size=keys.size, replace=True)
    return np.concatenate([groups[int(k)] for k in pick])


def paired_cluster_bootstrap(
    metric_fn: Callable[[np.ndarray], dict[str, float]], data: dict, *, clusters: np.ndarray,
    n_boot: int = DEFAULT_B, seed: int = 2026, alpha: float = 0.05,
) -> dict:
    """Percentile CIs from resampling whole clusters, computed on the same rows for every arm.

    ``metric_fn(indices, data) -> {arm: value}`` is the caller's metric code, so this
    function never re-implements a metric and cannot disagree with the runner about
    what "macro-AUPRC" means. Returns per-arm CIs plus the CI of each arm-pair
    difference; because the arms are evaluated on the *same* resampled rows, the
    difference interval is paired (which is the interval the comparison needs).
    """
    idx = cluster_indices(clusters)
    if len(idx) < 4:
        raise ValueError(
            f"only {len(idx)} clusters; a cluster bootstrap on fewer than 4 groups has no "
            "meaningful tails and would print a CI that is really a point estimate"
        )
    rng = np.random.default_rng(int(seed))
    keys = np.array(sorted(idx), dtype=np.int64)
    point = metric_fn(np.concatenate([idx[int(g)] for g in keys]), data)
    arms = [a for a in point if isinstance(point[a], (int, float, np.floating))]
    if not arms:
        raise ValueError("metric_fn returned no numeric arms")
    boots: dict[str, list[float]] = {a: [] for a in arms}
    for _ in range(int(n_boot)):
        pick = rng.choice(keys, size=keys.size, replace=True)
        sub = np.concatenate([idx[int(g)] for g in pick])
        vals = metric_fn(sub, data)
        for a in arms:
            v = vals.get(a, float("nan"))
            if np.isfinite(v):
                boots[a].append(float(v))
    out: dict[str, Any] = {}
    for a in arms:
        v = np.asarray(boots[a], dtype=np.float64)
        if v.size < max(20, 0.1 * n_boot):
            out[a] = {"point": float(point[a]), "ci": [None, None], "n_boot": int(v.size),
                      "warning": "too few finite bootstrap replicates to report an interval"}
            continue
        out[a] = {
            "point": round(float(point[a]), 5),
            "ci": [round(float(np.quantile(v, alpha / 2)), 5),
                   round(float(np.quantile(v, 1 - alpha / 2)), 5)],
            "sd": round(float(v.std(ddof=1)), 5), "n_boot": int(v.size),
            "cluster": "farm", "n_clusters": int(len(idx)), "percentile": True,
        }
    for i, a in enumerate(arms):
        for b in arms[i + 1 :]:
            va, vb = np.asarray(boots[a]), np.asarray(boots[b])
            n = min(va.size, vb.size)
            if n < 20:
                continue
            d = va[:n] - vb[:n]
            lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
            out[f"{a}-minus-{b}"] = {
                "mean_diff": round(float(d.mean()), 5),
                "ci": [round(float(lo), 5), round(float(hi), 5)],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "p_two_sided": round(float(2 * min((d <= 0).mean(), (d >= 0).mean())), 5),
                "paired": True, "n_boot": int(n),
            }
    return out



def holm(pvals: dict[str, float]) -> dict[str, dict]:
    """Holm–Bonferroni step-down over the declared family."""
    items = [(k, float(v)) for k, v in pvals.items() if np.isfinite(v)]
    if not items:
        return {k: {"p_raw": None, "p_holm": None, "reject": False, "reason": "non-finite p"}
                for k in pvals}
    items.sort(key=lambda t: t[1])
    m = len(items)
    out: dict[str, dict] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)          # monotonicity enforcement
        out[k] = {"p_raw": round(p, 6), "p_adjusted": round(running, 6),
                  "method": "holm", "family_size": m, "rank": i + 1,
                  "reject_at_0.05": bool(running <= 0.05)}
    for k in pvals:
        out.setdefault(k, {"p_raw": None, "p_adjusted": None, "reject": False,
                           "reason": "missing p-value (excluded from the family, not silently rejected)"})
    return out


def benjamini_hochberg(pvals: dict[str, float]) -> dict[str, dict]:
    """BH step-up for the screening family (where FWER control would hide every effect)."""
    items = [(k, float(v)) for k, v in pvals.items() if np.isfinite(v)]
    if not items:
        return {k: {"p_raw": None, "p_bh": None, "reject": False} for k in pvals}
    items.sort(key=lambda t: t[1])
    m = len(items)
    adj: dict[str, float] = {}
    prev = 1.0
    for i in range(m - 1, -1, -1):
        k, p = items[i]
        val = min(prev, p * m / (i + 1))
        adj[k] = val
        prev = val
    out = {k: {"p_raw": round(p, 6), "p_adjusted": round(adj[k], 6), "method": "bh",
               "family_size": m, "q": 0.05, "reject_at_0.05": bool(adj[k] <= 0.05)}
           for k, p in items}
    for k in pvals:
        out.setdefault(k, {"p_raw": None, "p_adjusted": None, "reject": False,
                           "reason": "missing p-value"})
    return out


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> dict:
    """Cliff's δ with the magnitude label the design asks for, and its CI source counts."""
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return {"available": False, "reason": "empty sample after NaN filtering"}
    gt = int((x[:, None] > y[None, :]).sum())
    lt = int((x[:, None] < y[None, :]).sum())
    n = x.size * y.size
    d = (gt - lt) / n
    ad = abs(d)
    mag = ("negligible" if ad < 0.147 else "small" if ad < 0.33
           else "medium" if ad < 0.474 else "large")
    common_language = float((gt + 0.5 * (n - gt - lt)) / n)
    return {
        "available": True, "delta": round(float(d), 4), "magnitude": mag,
        "prob_greater": round(common_language, 4),
        "n_x": int(x.size), "n_y": int(y.size),
        "note": "δ is over matched units (cells/cycles as supplied); CIs come from the cluster bootstrap",
    }


def friedman_nemenyi(table: np.ndarray, names: list[str]) -> dict:
    """Omnibus Friedman over blocks + Nemenyi critical difference over arms.

    ``table`` is (blocks × arms) of the primary metric. Ties in ranks are handled by
    scipy; where scipy refuses (too few blocks for the asymptotic χ²) the result says
    ``insufficient_blocks`` instead of reporting a p-value that would be fiction.
    """
    t = np.asarray(table, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != len(names):
        raise ValueError(f"table must be (blocks, {len(names)} arms), got {t.shape}")
    finite = np.isfinite(t)
    if not finite.all():
        rows = finite.all(axis=1)
        if rows.sum() < 4:
            return {"available": False, "reason": f"{int(rows.sum())} complete blocks; need ≥4"}
        t = t[rows]
    from scipy import stats as sst

    n_blocks, k = t.shape
    ranks = np.apply_along_axis(sst.rankdata, 1, t)
    chi2 = float(12.0 * n_blocks / (k * (k + 1)) * ((ranks.mean(axis=0) - (k + 1) / 2.0) ** 2).sum())
    df = k - 1
    p = float(sst.chi2.sf(chi2, df)) if n_blocks >= 10 else float("nan")
    # Nemenyi critical difference on the studentised range
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
               9: 3.102, 10: 3.164}.get(k, 3.164 + 0.05 * (k - 10))
    cd = float(q_alpha * np.sqrt(k * (k + 1) / (6.0 * n_blocks)))
    order = np.argsort(-ranks.mean(axis=0))
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = float(abs(ranks.mean(axis=0)[order[i]] - ranks.mean(axis=0)[order[j]]))
            pairs.append({"a": names[order[i]], "b": names[order[j]],
                          "rank_diff": round(diff, 3), "significant": bool(diff > cd)})
    return {
        "available": n_blocks >= 10,
        "friedman_chi2": round(chi2, 4), "df": df,
        "p": None if not np.isfinite(p) else p,
        "n_blocks": int(n_blocks), "n_arms": k,
        "mean_rank": {names[i]: round(float(ranks.mean(axis=0)[i]), 3) for i in range(k)},
        "order": [names[i] for i in order],
        "nemenyi_critical_difference": round(cd, 3),
        "pairs_significant": [q for q in pairs if q["significant"]],
        "note": (
            "χ² approximation needs ≥10 blocks; with fewer, only the mean ranks are reported"
            if n_blocks < 10 else "ranks are within-block, so block effects cancel"
        ),
    }


def _levels(seeds, farms, cells, groups, ok, group_name):
    """(name, masked array) pairs, so ``groups=None`` cannot change the tuple's shape.

    ``((x, y))`` is not a 1-tuple in Python, and a concatenation that silently
    flattens would make the dict comprehension unpack a whole array.
    """
    out = [("seed", seeds), ("farm", farms), ("cell", cells)]
    if groups is not None:
        out.append((group_name, np.asarray(groups)[ok]))
    return out


def variance_decomposition(
    values: np.ndarray, *, seeds: np.ndarray, farms: np.ndarray, cells: np.ndarray,
    labels: dict | None = None, groups: np.ndarray | None = None, group_name: str = "model",
) -> dict:
    """Four-way decomposition: seed / farm / cell / residual share of the variance.

    A random-effects mean-of-variance estimate (one-way anova sums of squares per
    factor, then each share normalised by the total) rather than a fitted mixed
    model: with ~3 seeds × 20 farms the REML fit is unstable and its output would
    carry more uncertainty than it removes. The residual share is reported, and it is
    usually the honest headline — "most of the variance is cell×arm interaction that
    no factor explains" is a finding, not a failure.
    """
    v = np.asarray(values, dtype=np.float64)
    ok = np.isfinite(v)
    if ok.sum() < 8:
        return {"available": False, "reason": f"{int(ok.sum())} finite values"}
    v, s, f, c = v[ok], np.asarray(seeds)[ok], np.asarray(farms)[ok], np.asarray(cells)[ok]
    total = float(v.var(ddof=1) * (v.size - 1) / max(1, v.size))
    shares: dict[str, float] = {}
    for nm, fac in (("seed", s), ("farm", f), ("cell", c)):
        gm = v.mean()
        between = 0.0
        for g in np.unique(fac):
            sel = fac == g
            n = int(sel.sum())
            if n > 1:
                between += n * (float(v[sel].mean()) - gm) ** 2
        shares[nm] = round(float(between / v.size) / max(total, 1e-12), 4)
    if groups is not None:
        # The fourth factor is the comparison the benchmark exists for, so it is
        # computed with the same one-way between-group share as the other three — the
        # four shares are then directly comparable, and a model share that is smaller
        # than the seed share is exactly the embarrassing answer we need to be able to
        # state. Requires the same finite-row masking as above.
        g = np.asarray(groups)[ok]
        gm = v.mean()
        between = 0.0
        for lv in np.unique(g):
            sel = g == lv
            n = int(sel.sum())
            if n > 1:
                between += n * (float(v[sel].mean()) - gm) ** 2
        shares[group_name] = round(float(between / v.size) / max(total, 1e-12), 4)
    shares["residual"] = round(max(0.0, 1.0 - sum(shares.values())), 4)
    lab = labels or {}
    return {
        "available": True, "n": int(v.size),
        "total_variance": round(total, 6),
        "shares": shares,
        "levels": {nm: int(len(np.unique(fac))) for nm, fac in _levels(s, f, c, groups, ok,
                                                                      group_name)},
        "method": "one-way between-group variance share per factor, residual = 1 − Σ (no REML)",
        "caveat": lab.get("caveat", "factors are not orthogonal; shares may not be uniquely attributable"),
    }

def kendall_w(values: np.ndarray, *, names: list[str] | None = None) -> dict:
    """Kendall's W — the agreement of item rankings across blocks (task 10.7's engine).

    ``values`` is (blocks × items) of the primary metric; ranks are taken within each
    block, so W = 1 means every block orders the items identically (an interaction-free
    ranking: whoever wins on one anomaly family wins on all of them) and W near 0 means
    the ranking is block-specific — the interaction the benchmark must not hide. W comes
    with its χ² approximation, refused (``available: False``) where n or k are too small
    to support the asymptotic p — the W itself is still reported for what it is.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] < 3 or v.shape[0] < 3:
        return {"available": False, "n_blocks": int(v.shape[0]) if v.ndim == 2 else 0,
                "n_items": int(v.shape[1]) if v.ndim == 2 else 0,
                "reason": "W needs a (blocks × items) table with ≥3 of each"}
    ok_rows = np.isfinite(v).all(axis=1)
    if ok_rows.sum() < 3:
        return {"available": False, "n_blocks": int(ok_rows.sum()),
                "reason": f"{int(ok_rows.sum())} complete blocks; W over partial blocks would rank presence, not performance"}
    from scipy import stats as sst

    t = v[ok_rows]
    n_blocks, k_items = t.shape
    ranks = np.apply_along_axis(sst.rankdata, 1, t)          # within each block
    rank_sum = ranks.sum(axis=0)
    s_stat = float(((rank_sum - n_blocks * (k_items + 1) / 2.0) ** 2).sum())
    w = float(12.0 * s_stat / (n_blocks ** 2 * (k_items ** 3 - k_items)))
    chi2 = float(n_blocks * (k_items - 1) * w)
    p = float(sst.chi2.sf(chi2, k_items - 1)) if n_blocks >= 10 else None
    mean_rank = ranks.mean(axis=0)
    order = np.argsort(-mean_rank)
    out = {
        "available": True, "w": round(w, 4), "chi2": round(chi2, 4), "df": k_items - 1,
        "p": None if p is None else round(p, 6), "n_blocks": int(n_blocks), "n_items": int(k_items),
        "mean_rank": {names[i]: round(float(mean_rank[i]), 3) for i in range(k_items)} if names
        else {i: round(float(mean_rank[i]), 3) for i in range(k_items)},
        "order": [names[i] for i in order] if names else [int(order[i]) for i in range(k_items)],
        "note": ("W=1: the same item wins in every block (no ranking interaction). W≈0: the ranking "
                 "is block-specific — report per-block winners instead of one overall rank"
                 if w < 0.5 else
                 "W>0.5: rankings agree across blocks; a single overall ordering is a defensible summary"),
    }
    return out


def wilcoxon_paired(diffs: np.ndarray, *, zero_method: str = "wilcox") -> dict:
    """Wilcoxon signed-rank on paired differences, with the degeneracies reported.

    Returns ``available: False`` (and why) instead of a p-value when the input cannot
    support the test — all-zero differences, or n too small for the asymptotic
    approximation. A p-value of 1.0 from an empty sample would read as "no effect"
    when the honest statement is "untestable".
    """
    from scipy import stats as sps

    d = np.asarray(list(np.ravel(diffs)), dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size < 6:
        return {"available": False, "n": int(d.size),
                "reason": f"{int(d.size)} finite paired differences; the test needs ≥6"}
    nz = d[d != 0.0]
    if nz.size == 0:
        return {"available": False, "n": int(d.size), "ties": int(d.size),
                "reason": "every pair is tied: there is no ranking to test"}
    try:
        stat, p = sps.wilcoxon(d, zero_method=zero_method)
    except ValueError as exc:
        return {"available": False, "n": int(d.size), "reason": f"{type(exc).__name__}: {exc}"}
    return {"available": True, "p": float(p), "statistic": round(float(stat), 3),
            "n": int(d.size), "ties": int(d.size - nz.size), "zero_method": zero_method,
            "note": ("exact-free normal approximation with tie correction; the cluster bootstrap "
                     "interval, not this p-value, is the primary uncertainty statement")}

"""Downstream transfer suite T1–T5 (change tasks 4.3–4.7; design §Tasks).

What lives here vs elsewhere
----------------------------
- T1/T2/T3 primitives already exist in ``evaluation.tasks``; this module adds
  the pieces tasks.md names that were missing: the family × severity injection
  pool built from the FROZEN severity table, the T3 few-shot similar-date
  benchmark with its TWO tracks (Track 1 injection-labelled; Track 2 weak-proxy,
  reported separately and never claim-bearing), the ``near_matches`` listing,
  T4 next-1/3-bin trajectory quality vs the persistence baseline, and the T5
  exploratory listing.
- Claim discipline is enforced by construction: Track 2 and T5 outputs carry
  ``claim_bearing=False`` and are never returned in the claim-bearing dict.

Injection provenance: every pool parameter comes from
``ssl_anomaly.monitor.severity_table`` (the frozen, hash-pinned S1–S6 table) —
acute duration = the level's ``duration_bins``; persistent = the frozen
``PERSISTENT_DURATION_BINS`` (5); gradual = a linear ramp over
``GRADUAL_RAMP_BINS`` then held. NO new severity parameters are invented here.
Injection detection is sensitivity evidence only (L2), never stress proof.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ssl_anomaly.evaluation.metrics import auprc
from ssl_anomaly.monitor.severity_table import (
    GRADUAL_RAMP_BINS,
    PERSISTENT_DURATION_BINS,
    PRIMARY_FAMILIES,
    SEVERITY_TABLE,
    severity_table_hash,
)

CLAIM_BEARING_TASKS = ("T1", "T2", "T3_track1", "T4")
EXPLORATORY_TASKS = ("T3_track2", "T3_near_matches", "T5")


# ---------------------------------------------------------------------------
# Injection pool from the FROZEN severity table (T2's label source)
# ---------------------------------------------------------------------------


def _inject_ndvi_drop(cycle: np.ndarray, mask: np.ndarray, *, start_bin: int,
                      duration_bins: int, delta: float) -> np.ndarray:
    """Drop NDVI by ``delta`` on OBSERVED bins in ``[start, start+duration)``.

    Absolute-index drop in the frozen table's units; unobserved bins stay
    unobserved (an injection never fabricates an observation).
    """
    out = cycle.copy()
    end = min(cycle.shape[0], start_bin + duration_bins)
    window = slice(start_bin, end)
    observed = mask[window, 0]
    out[window, 0][observed] = np.clip(out[window, 0][observed] - delta, 0.0, 1.0)
    return out


def build_injection_pools(
    x: np.ndarray, m: np.ndarray, *,
    n_per_cell: int, seed: int = 13,
    ndvi_channel: int = 0,
) -> dict:
    """Family × severity injection pool on the supplied cycles.

    Every cycle is used at most once per cell and ground truth is known by
    construction (injections happen on cycles drawn from the pool, and the
    injected copy REPLACES the clean cycle in the returned tensors; the clean
    originals are the negatives in the scoring cells). Deterministic in ``seed``.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be (N, T, C), got {x.shape}")
    n_bins = x.shape[1]
    rng = np.random.default_rng(seed)
    xs, labels, families, severities, source_idx, masks = [], [], [], [], [], []
    per_cell_counts: dict[str, int] = {}
    for level in SEVERITY_TABLE:
        for family in PRIMARY_FAMILIES:
            for k in range(n_per_cell):
                i = int(rng.integers(0, x.shape[0]))
                start = int(rng.integers(0, max(1, n_bins - level.duration_bins)))
                if family == "acute":
                    duration = level.duration_bins
                elif family == "persistent":
                    duration = PERSISTENT_DURATION_BINS
                elif family == "gradual":
                    ramp = GRADUAL_RAMP_BINS[k % len(GRADUAL_RAMP_BINS)]
                    start = max(0, start - ramp)
                    duration = min(n_bins - start, ramp + level.duration_bins)
                    # gradual = linear ramp over `ramp` bins to δ, then held
                    cyc = x[i].copy()
                    msk = m[i]
                    w = slice(start, min(n_bins, start + ramp))
                    obs = msk[w, 0]
                    frac = np.linspace(0.0, 1.0, ramp)[obs]
                    cyc[w, 0][obs] = np.clip(
                        np.asarray(cyc[w, 0][obs]) - level.delta_ndvi * frac, 0.0, 1.0)
                    tail = slice(start + ramp, min(n_bins, start + ramp + level.duration_bins))
                    obs_t = msk[tail, 0]
                    cyc[tail, 0][obs_t] = np.clip(
                        np.asarray(cyc[tail, 0][obs_t]) - level.delta_ndvi, 0.0, 1.0)
                    xs.append(cyc)
                    labels.append(1)
                    families.append(family)
                    severities.append(level.level_id)
                    source_idx.append(i)
                    masks.append(m[i].copy())
                    per_cell_counts[f"{family}:{level.level_id}"] = per_cell_counts.get(
                        f"{family}:{level.level_id}", 0) + 1
                    continue
                else:
                    raise AssertionError(f"unhandled family {family}")
                cyc = _inject_ndvi_drop(x[i], m[i], start_bin=start,
                                        duration_bins=duration,
                                        delta=level.delta_ndvi)
                xs.append(cyc)
                labels.append(1)
                families.append(family)
                severities.append(level.level_id)
                source_idx.append(i)
                masks.append(m[i].copy())
                per_cell_counts[f"{family}:{level.level_id}"] = per_cell_counts.get(
                    f"{family}:{level.level_id}", 0) + 1
    return {
        "status": "built",
        "x_injected": np.stack(xs),
        "masks": np.stack(masks),          # the SOURCE cycle's mask per injection
        "source_indices": np.asarray(source_idx, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
        "families": np.asarray(families),
        "severities": np.asarray(severities),
        "per_cell_counts": per_cell_counts,
        "severity_table_sha256": severity_table_hash(),
        "note": ("synthetic injections are CONTROLLED evidence (L2 sensitivity "
                 "only), never stress proof; parameters come from the frozen "
                 "severity table, none invented here"),
    }


def t2_family_severity_cells(
    scores: np.ndarray, pools: dict, *, clean_scores: np.ndarray,
) -> dict:
    """AUC-PR per family × severity cell with prevalence (task 4.4).

    Each cell = the injected cycles of that (family, severity) cell (positives)
    against a 3x random sample of clean cycles (negatives), scored with the
    SAME scorer. Per-cell reporting is mandatory — no macro-only summary.
    Cycles with no finite score (e.g. no observed B1-channel bins) are excluded
    per cell and COUNTED — never scored as 0, never silently dropped.
    """
    labels = pools["labels"]
    families = pools["families"]
    severities = pools["severities"]
    xs_inj_scores = np.asarray(scores, dtype=np.float64)
    clean_scores = np.asarray(clean_scores, dtype=np.float64)
    rng = np.random.default_rng(13)
    cells: dict[str, dict] = {}
    n_unscorable_total = 0
    for family in PRIMARY_FAMILIES:
        for level in SEVERITY_TABLE:
            cell_mask = (families == family) & (severities == level.level_id)
            sel = cell_mask & np.isfinite(xs_inj_scores)
            n_unscorable_total += int((cell_mask & ~np.isfinite(xs_inj_scores)).sum())
            if not sel.any():
                cells[f"{family}:{level.level_id}"] = {
                    "status": "absent",
                    "reason": "no scorable injected cycle in this cell"}
                continue
            n_pos = int(sel.sum())
            finite_clean = np.flatnonzero(np.isfinite(clean_scores))
            neg_n = min(n_pos * 3, finite_clean.size)
            neg_idx = rng.choice(finite_clean, size=neg_n, replace=False)
            s = np.concatenate([xs_inj_scores[sel], clean_scores[neg_idx]])
            y = np.concatenate([np.ones(n_pos), np.zeros(neg_n)])
            ap, prev = auprc(s, y)
            cells[f"{family}:{level.level_id}"] = {
                "status": "measured", "n": int(s.size), "n_positives": n_pos,
                "n_negatives": int(neg_n), "auprc": float(ap),
                "prevalence": float(prev),
            }
    return {
        "cells": cells,
        "n_unscorable_injected": int(n_unscorable_total),
        "note": ("cycles with no finite score are excluded per cell and "
                 "counted — never scored as 0"),
    }


# ---------------------------------------------------------------------------
# T1 — phenology-stage estimation, claim-bearing (task 4.3)
# ---------------------------------------------------------------------------


def t1_phenology_probe(
    z: np.ndarray, y_bins: np.ndarray, train_mask: np.ndarray, eval_mask: np.ndarray,
    *, n_classes: int = 36, b1_features: np.ndarray | None = None,
) -> dict:
    """Phase-bin probe (claim-bearing): macro-F1 + accuracy vs the majority
    baseline and vs a probe on the B1 features — identical splits and
    hyperparameters for both (``level1.phase_linear_probe`` trains on the TRAIN
    rows only; the majority baseline is computed on the eval targets)."""
    from ssl_anomaly.evaluation.level1 import phase_linear_probe

    rep = phase_linear_probe(
        z[train_mask], y_bins[train_mask], z[eval_mask], y_bins[eval_mask],
        n_classes=n_classes,
    )
    out = {
        "status": "measured",
        "claim_bearing": True,
        "n_train": int(train_mask.sum()),
        "n_eval": int(eval_mask.sum()),
        "macro_f1": rep.macro_f1,
        "accuracy": rep.accuracy,
        "majority_baseline": rep.majority_accuracy,
        "beats_majority": bool(rep.accuracy > rep.majority_accuracy),
    }
    if b1_features is not None:
        b1 = phase_linear_probe(
            b1_features[train_mask], y_bins[train_mask],
            b1_features[eval_mask], y_bins[eval_mask],
            n_classes=n_classes,
        )
        out["b1_feature_probe"] = {
            "macro_f1": b1.macro_f1, "accuracy": b1.accuracy,
            "majority_baseline": b1.majority_accuracy,
            "beats_b1_features": bool(rep.accuracy > b1.accuracy),
        }
    else:
        out["b1_feature_probe"] = {"status": "absent",
                                   "reason": "B1 features not supplied"}
    return out


# ---------------------------------------------------------------------------
# T3 — few-shot similar-date retrieval, two tracks + near matches (4.5)
# ---------------------------------------------------------------------------


def _rank_top_m(query: np.ndarray, gallery: np.ndarray, m: int) -> np.ndarray:
    """Indices of the m most similar gallery rows (cosine on L2-normalised)."""
    q = query / (np.linalg.norm(query) + 1e-12)
    g = gallery / (np.linalg.norm(gallery, axis=1, keepdims=True) + 1e-12)
    sims = g @ q
    return np.argsort(-sims)[:m]


def t3_fewshot_retrieval(
    arm_embeddings: dict[str, np.ndarray],
    query_indices: np.ndarray,
    query_families: np.ndarray,
    gallery_indices: np.ndarray,
    gallery_families: np.ndarray,
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    m_values: tuple[int, ...] = (1, 5, 10),
    arm_scores: dict[str, np.ndarray] | None = None,
) -> dict:
    """Track 1: rank the labelled gallery by similarity to each query.

    Per arm: precision@m (m ∈ {1,5,10}), MAP, and family-hit-rate at each k.
    ``arm_scores`` carries the score-ranked baselines (B1-score, A-rule-score):
    those rank the gallery by descending anomaly score instead of embedding
    similarity. The query's own cycle is excluded from the gallery.
    """
    arm_scores = arm_scores or {}
    results: dict[str, dict] = {}
    for arm, emb in arm_embeddings.items():
        per_query_p: dict[int, list[float]] = {m: [] for m in m_values}
        aps, hits = [], {k: [] for k in k_values}
        for qi, qf in zip(query_indices, query_families):
            gallery_pos = np.flatnonzero(gallery_indices != qi)
            g_idx = gallery_indices[gallery_pos]
            g_fam = gallery_families[gallery_pos]
            if g_idx.size == 0:
                continue
            if arm in arm_scores:
                order = np.argsort(-np.asarray(arm_scores[arm])[gallery_pos])
            else:
                order = _rank_top_m(emb[qi], emb[gallery_pos], g_idx.size)
            relevant = g_fam == qf
            ranked = relevant[order]
            for m in m_values:
                top_m = ranked[:m]
                per_query_p[m].append(
                    float(top_m.sum() / min(m, ranked.size)) if ranked.size else 0.0)
            # MAP over the ranked relevant set
            cum_hits, sum_prec = 0, 0.0
            for rank, rel in enumerate(ranked, start=1):
                if rel:
                    cum_hits += 1
                    sum_prec += cum_hits / rank
            aps.append(sum_prec / max(1, int(ranked.sum())))
            for k in k_values:
                hits[k].append(float(ranked[:k].any()))
        results[arm] = {
            "status": "measured",
            "n_queries": int(len(aps)),
            "precision_at_m": {str(m): float(np.mean(v)) for m, v in per_query_p.items()
                               if v},
            "map": float(np.mean(aps)) if aps else None,
            "family_hit_rate_at_k": {str(k): float(np.mean(v)) for k, v in hits.items()
                                     if v},
            "claim_bearing": True,
        }
    return {
        "track": "T3_track1",
        "label_source": "injection-labelled dates (controlled evidence only)",
        "k_values": list(k_values), "m_values": list(m_values),
        "arms": results,
        "note": ("if SSL retrieval does not beat raw-distance and B1-score "
                 "ranking here, the retrieval claim FAILS and is reported as such"),
    }


def t3_track2_weak_proxy(
    arm_embeddings: dict[str, np.ndarray],
    query_indices: np.ndarray,
    gallery_indices: np.ndarray,
    *,
    proxy_source: str | None = None,
) -> dict:
    """Track 2 — weak-proxy queries, EXPLORATORY, never gate input.

    The weak-proxy labels (CWSI_ET / TVDI excursions on natural dev cycles) do
    not exist in this repository yet, so the track reports ``absent`` with that
    reason unless a producer is supplied. It is never claim-bearing either way.
    """
    if proxy_source is None:
        return {
            "track": "T3_track2", "claim_bearing": False, "label": "exploratory",
            "status": "absent",
            "reason": ("no weak-proxy query producer exists in-repo (CWSI_ET / "
                       "TVDI excursion tables are not produced by this change); "
                       "the track is recorded absent, never simulated"),
            "note": ("dependence structure: CWSI shares inputs with the model "
                     "wherever LST/ETa are model inputs — reported separately, "
                     "never merged with Track 1"),
        }
    raise NotImplementedError(
        "a weak-proxy producer was supplied but the shared-input dependence "
        "declaration must accompany it; extend this function with the producer's "
        "own provenance block"
    )


def near_matches_listing(
    arm_embeddings: dict[str, np.ndarray],
    query_indices: np.ndarray,
    natural_indices: np.ndarray,
    natural_meta: list[dict],
    *,
    top: int = 10,
) -> dict:
    """Descriptive top-``top`` nearest NATURAL archive dates per query.

    ``natural_indices`` must be NATURAL (unlabelled) cycles only — injected
    cycles never appear in this listing. ``natural_meta[k]`` describes
    ``natural_indices[k]`` (field/date/etc.). Descriptive for manual owner
    inspection; SHALL NOT feed any gate, claim, threshold or selection.
    """
    if len(natural_meta) != len(natural_indices):
        raise ValueError(
            f"natural_meta has {len(natural_meta)} entries for "
            f"{len(natural_indices)} natural cycles; refusing to misalign"
        )
    listing: dict[str, dict] = {}
    for arm, emb in arm_embeddings.items():
        entries = []
        for qi in query_indices:
            pos = np.flatnonzero(natural_indices != qi)
            if pos.size == 0:
                entries.append({"query_cycle_index": int(qi),
                                "query_answered": "T3 retrieval query",
                                "neighbours": [], "note": "no natural neighbours"})
                continue
            cand = natural_indices[pos]
            order = _rank_top_m(emb[qi], emb[pos], min(top, cand.size))
            neighbours = []
            for r, o in enumerate(order):
                e = emb[qi] / (np.linalg.norm(emb[qi]) + 1e-12)
                c = emb[pos][o] / (np.linalg.norm(emb[pos][o]) + 1e-12)
                neighbours.append({
                    "rank": r + 1,
                    "cycle_index": int(cand[o]),
                    "similarity": float(e @ c),
                    **natural_meta[int(o)],
                })
            entries.append({
                "query_cycle_index": int(qi),
                "query_answered": "T3 retrieval query",
                "neighbours": neighbours,
            })
        listing[arm] = entries
    return {
        "track": "T3_near_matches", "claim_bearing": False, "label": "exploratory",
        "top": top, "arms": listing,
        "note": "descriptive, for manual owner inspection only — never gate input",
    }


def near_matches_markdown(listing: dict, cycle_meta: dict[int, str]) -> str:
    """Render the near-match listing as ``near_matches.md`` (owner inspection)."""
    lines = [
        "# Natural-data near matches (T3, EXPLORATORY)",
        "",
        "Descriptive listing for manual owner inspection. This artifact is",
        "`exploratory` by pre-registration: it feeds NO gate, claim, threshold",
        "or selection — never gate input, never a claim's evidence.",
        "",
    ]
    for arm, entries in listing["arms"].items():
        lines.append(f"## Arm: {arm}")
        lines.append("")
        for entry in entries:
            qi = entry["query_cycle_index"]
            lines.append(f"- **Query cycle {qi}** — {entry['query_answered']}")
            for nb in entry["neighbours"]:
                meta = cycle_meta.get(nb["cycle_index"], "")
                lines.append(
                    f"  {nb['rank']}. cycle {nb['cycle_index']} "
                    f"(similarity {nb['similarity']:.4f}) {meta}".rstrip())
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# T4 — trajectory quality vs persistence (task 4.6)
# ---------------------------------------------------------------------------


def t4_trajectory_vs_persistence(
    x: np.ndarray, m: np.ndarray, *,
    horizons: tuple[int, ...] = (1, 3),
    cut_frac: float = 0.75,
    forecaster=None,
) -> dict:
    """Next-1/3-bin forecasting error on CLEAN held-out cycles vs persistence.

    Definition (stated here because it is the comparison's meaning): the cycle
    is cut at ``cut_frac`` of its bins; the FORECAST for each horizon h is made
    from bins <= cut only; the TARGET is the mean of the OBSERVED bins in
    ``(cut, cut+h]``. Persistence carries the cut bin's last observed value
    forward. ``forecaster(x_i, m_i, cut, h) -> (C,)`` may supply a model arm;
    BOTH arms use this identical definition — numbers under different
    definitions are never compared (AGENTS.md §8). Cycles with no observed
    target bin at a horizon are excluded from that horizon and counted.
    """
    B, T, C = x.shape
    cut = max(1, int(T * cut_frac) - 1)
    out: dict = {
        "task": "T4", "claim_bearing": True, "cut_bin": int(cut),
        "cut_frac": cut_frac, "horizons": list(horizons), "n_cycles": int(B),
        "definition": ("forecast from bins <= cut; target = mean of observed "
                       "bins in (cut, cut+h]; identical for every arm"),
        "persistence": {}, "model": {},
        "note": ("clean held-out dev cycles only; persistence is very strong at "
                 "composite cadence and must be beaten for any claim"),
    }
    for h in horizons:
        p_errs, m_errs, n_used = [], [], 0
        for i in range(B):
            window = slice(cut + 1, min(T, cut + 1 + h))
            obs = m[i, window]                        # (w, C)
            if not obs.any():
                continue
            xx = np.where(np.isfinite(x[i, window]), x[i, window], np.nan)
            tgt = np.nanmean(np.where(obs, xx, np.nan), axis=0)
            w = obs.sum(axis=0)
            # persistence: last OBSERVED value at or before the cut (per channel;
            # a channel unobserved at that bin has no persistence forecast)
            hist = m[i, : cut + 1]
            last_obs = np.flatnonzero(hist[:, 0])
            pers = x[i, last_obs[-1]] if last_obs.size else np.full(C, np.nan)
            usable = (w > 0) & np.isfinite(pers) & (w > 0)
            if not usable.any():
                continue
            n_used += 1
            p_errs.append(np.nanmean(np.abs(pers - tgt)[usable]))
            if forecaster is not None:
                pred = np.asarray(forecaster(x[i], m[i], cut, h), dtype=np.float64)
                if pred.shape != (C,):
                    raise ValueError(
                        f"forecaster must return ({C},), got {pred.shape}")
                if not np.isfinite(pred).all():
                    raise ValueError("forecaster produced non-finite predictions")
                m_errs.append(np.nanmean(np.abs(pred - tgt)[usable]))
        out["persistence"][str(h)] = {
            "mae": float(np.mean(p_errs)) if p_errs else None,
            "n_cycles_used": n_used,
        }
        if forecaster is not None:
            out["model"][str(h)] = {
                "mae": float(np.mean(m_errs)) if m_errs else None,
                "beats_persistence": (
                    bool(np.mean(m_errs) < np.mean(p_errs))
                    if m_errs and p_errs else None),
            }
        else:
            out["model"][str(h)] = {
                "status": "absent",
                "reason": ("no causal forecaster supplied; only the persistence "
                           "baseline is measured (recorded, not forced — a "
                           "like-for-like model arm needs a forecaster with "
                           "exactly this cut-based definition)"),
            }
    return out


# ---------------------------------------------------------------------------
# T5 — exploratory listing (task 4.7)
# ---------------------------------------------------------------------------


def t5_exploratory_listing(
    z: np.ndarray, cycle_meta: dict[int, str], *, n_clusters: int = 8,
    seed: int = 13,
) -> dict:
    """Descriptive clustering + per-cluster nearest-date examples. EXPLORATORY.

    Never claim-bearing: the label is enforced here so a caller cannot quote it.
    """
    from sklearn.cluster import KMeans

    zm = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    km = KMeans(n_clusters=min(n_clusters, max(2, len(z) - 1)), n_init=4,
                random_state=seed)
    assign = km.fit_predict(zm)
    clusters: dict[int, dict] = {}
    for c in range(assign.max() + 1):
        members = np.flatnonzero(assign == c)
        centroid = km.cluster_centers_[c]
        order = np.argsort(-(zm[members] @ centroid))[:3]
        clusters[int(c)] = {
            "n_members": int(members.size),
            "examples": [
                {"cycle_index": int(members[o]),
                 "meta": cycle_meta.get(int(members[o]), "")}
                for o in order
            ],
        }
    return {
        "task": "T5", "claim_bearing": False, "label": "exploratory",
        "n_clusters": int(assign.max() + 1), "clusters": clusters,
        "note": "descriptive only: never used in gate evaluation",
    }

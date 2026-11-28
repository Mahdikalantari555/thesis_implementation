"""Environmental index references: TVDI, CWSI_ET, CWSI_LST, ETa, ETrF (tasks 9.14–9.18).

These are *reference scores*, not models: the benchmark's claim is about what a
self-supervised objective buys over the operational indices a practitioner would
reach for. Each index is computed from exactly the tensor the arms see (the same
channels, the same bins, the same mask), so a comparison is possible; where the
inputs an index needs are not in the archive, the index reports itself unavailable
with the missing field named, and the cell is not run against a substitute.

Two discipline rules are enforced in code, because both have been broken in
published work:

* **Fit on train only.** TVDI and the CWSI family normalise against an empirical
  envelope (the LST–NDVI "triangle", or a field's own history). Those bounds are
  parameters, and fitting them on the scored partition would let the index see the
  events it is later asked to detect. ``fit_idx`` is therefore mandatory and is
  checked to be disjoint from the scored rows.
* **No circular labels.** An index may not be scored against a label derived from the
  same quantity (an ET-based stress label scored by an ET-based index is a tautology).
  :func:`assert_not_circular` refuses when the index's inputs intersect the label
  source named in the pre-registration.

Task 9.18 (dependency-aware uncertainty) is handled by :func:`index_agreement`:
the correlation between an index and an arm's score is a *blocked* quantity, so the
bootstrap resamples whole (field, year) blocks — the contract's
``block_bootstrap_by_field_year`` — falling back to whole fields when no year axis is
supplied, with the fallback recorded. Per-cycle resampling would report an interval
several times too narrow and make every index look "significantly different" from
every arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

INDICES = ("TVDI", "CWSI_ET", "CWSI_LST", "ETa", "ETrF")
#: what each index needs from the archive; missing fields make it unavailable, never imputed
NEEDS: dict[str, tuple[str, ...]] = {
    "TVDI": ("lst", "ndvi"),
    "CWSI_LST": ("lst", "ndvi"),
    "CWSI_ET": ("lst", "ndvi", "et0"),
    "ETa": ("ndvi", "et0"),
    "ETrF": ("ndvi", "et0"),
}
#: label sources that would make the index's own input the ground truth
CIRCULAR_SOURCES: dict[str, tuple[str, ...]] = {
    "TVDI": ("lst_anomaly", "land_surface_temperature"),
    "CWSI_LST": ("lst_anomaly", "land_surface_temperature"),
    "CWSI_ET": ("evapotranspiration_deficit", "eta_deficit"),
    "ETa": ("evapotranspiration_deficit", "eta_deficit"),
    "ETrF": ("evapotranspiration_deficit", "eta_deficit"),
}
STRESS_SIGN: dict[str, float] = {k: 1.0 for k in INDICES}     # all are "higher = more stressed"


@dataclass
class IndexScore:
    name: str
    scores: np.ndarray | None                     # (N, T) float64, NaN where unscorable
    available: bool
    fit_record: dict = field(default_factory=dict)
    missing: tuple[str, ...] = ()
    note: str = ""
    lower_is_better: bool = False

    def to_dict(self, *, include_scores: bool = False) -> dict:
        d = {"index": self.name, "available": self.available, "missing_fields": list(self.missing),
             "fit": dict(self.fit_record), "note": self.note, "higher_is_stress": not self.lower_is_better}
        if include_scores and self.scores is not None:
            s = np.asarray(self.scores, dtype=np.float64)
            fin = np.isfinite(s)
            d["summary"] = {"shape": list(s.shape), "n_finite": int(fin.sum()),
                            "mean": round(float(s[fin].mean()), 5) if fin.any() else None,
                            "p99": round(float(np.quantile(s[fin], 0.99)), 5) if fin.any() else None}
        return d


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _col(names: tuple[str, ...] | list[str], wanted: str) -> int | None:
    for i, n in enumerate(names):
        if str(n).lower() == wanted:
            return i
    return None


def _check_fit_disjoint(fit_idx: np.ndarray, scored_idx: np.ndarray, *, who: str,
                        n_obs: int = 1) -> dict:
    fit = np.asarray(fit_idx, dtype=np.int64)
    sc = np.asarray(scored_idx, dtype=np.int64)
    over = len(set(int(v) for v in fit) & set(int(v) for v in sc))
    if over:
        raise ValueError(
            f"{who}: {over} rows appear in both the fitting and the scored index sets; the empirical "
            "envelope would then be fitted on the data it is later judged against"
        )
    if fit.size < 8 or fit.size * max(1, int(n_obs)) < 256:
        raise ValueError(
            f"{who}: {fit.size} rows × {n_obs} bins to fit the envelope; a triangle drawn through a "
            "handful of points is not a baseline, it is a coincidence. Use the full train partition."
        )
    return {"n_fit_rows": int(fit.size), "overlap_with_scored": over, "disjoint": True}


def assert_not_circular(name: str, label_source: str | None) -> dict:
    """Refuse an index whose inputs are the label's own source (the tautology guard)."""
    src = str(label_source or "").strip().lower()
    bad = [w for w in CIRCULAR_SOURCES.get(name, ()) if w and w in src]
    if bad:
        raise ValueError(
            f"{name} cannot be scored against labels derived from {bad}: the index would be graded "
            "against its own input. Choose an independent label source (field-level stress records, "
            "harvest loss) or drop the index row."
        )
    return {"index": name, "label_source": src or None, "independent": True}


# ---------------------------------------------------------------------------
# the indices
# ---------------------------------------------------------------------------


def tvdi(x: np.ndarray, channel_names, *, fit_idx: np.ndarray, scored_idx: np.ndarray,
         ndvi_bins: int = 12, min_support: int = 8) -> IndexScore:
    """Temperature Condition Index: where a pixel sits between the dry and wet LST edges.

    The edges are the per-NDVI-bin min/max of LST across the **fit** population, which
    is the whole of what TVDI is: a normalisation by the day's spatial spread. On a
    single-farm archive the "spatial" spread is the spread across fields on that bin, so
    ``min_support`` refuses a bin with too few fields to bound anything, and those bins
    come out NaN instead of a fabricated 0/1.
    """
    names = [str(c).lower() for c in channel_names]
    ci_l, ci_n = _col(names, "lst"), _col(names, "ndvi")
    if ci_l is None or ci_n is None:
        return IndexScore("TVDI", None, False, missing=tuple(k for k in NEEDS["TVDI"]
                                                              if _col(names, k) is None),
                          note="archive has no LST/NDVI columns for the LST–NDVI feature space")
    a = np.asarray(x, dtype=np.float64)
    n, T, _C = a.shape
    lst, ndvi = a[:, :, ci_l], a[:, :, ci_n]
    obs = np.isfinite(lst) & np.isfinite(ndvi)
    fit = np.asarray(fit_idx, dtype=np.int64)
    sc = np.asarray(scored_idx, dtype=np.int64)
    rec = _check_fit_disjoint(fit, sc, who="TVDI", n_obs=T)
    out = np.full((n, T), np.nan)
    edges: dict[str, Any] = {}
    for t in range(T):
        m = obs[:, t]
        rows_fit = np.flatnonzero(m & np.isin(np.arange(n), fit))
        if rows_fit.size < min_support:
            continue
        q = np.quantile(ndvi[rows_fit, t], np.linspace(0, 1, ndvi_bins + 1))
        q[0], q[-1] = -np.inf, np.inf
        lo = np.full(ndvi_bins, np.nan)
        hi = np.full(ndvi_bins, np.nan)
        for b in range(ndvi_bins):
            sel = rows_fit[(ndvi[rows_fit, t] >= q[b]) & (ndvi[rows_fit, t] < q[b + 1])]
            if sel.size >= min_support:
                lo[b], hi[b] = float(np.min(lst[sel, t])), float(np.max(lst[sel, t]))
        idx_b = np.clip(np.searchsorted(q, ndvi[:, t], side="right") - 1, 0, ndvi_bins - 1)
        good = np.isfinite(lo[idx_b]) & (hi[idx_b] > lo[idx_b]) & m & obs[:, t]
        with np.errstate(invalid="ignore"):
            v = (lst[good, t] - lo[idx_b[good]]) / (hi[idx_b[good]] - lo[idx_b[good]])
        out[good, t] = np.clip(v, 0.0, 1.0)
        edges[str(t)] = {"bins_filled": int(np.isfinite(lo).sum()), "bins_total": ndvi_bins,
                         "n_fit": int(rows_fit.size)}
    rec["per_bin_edges"] = edges
    rec["ndvi_bins"] = int(ndvi_bins)
    rec["min_support"] = int(min_support)
    frac = float(np.isfinite(out[sc]).mean()) if sc.size else 0.0
    return IndexScore("TVDI", out, bool(frac > 0), fit_record=rec,
                      note=(f"TVDI scoreable for {frac:.1%} of scored bins; unbounded NDVI bins are "
                            "NaN, not clamped"))


def cwsi_lst(x: np.ndarray, channel_names, *, fit_idx: np.ndarray, scored_idx: np.ndarray,
             clip_q: float = 0.02) -> IndexScore:
    """CWSI from LST alone, normalised on each field's *own* history (temporal edges).

    This is the deliberate sibling of TVDI: the same quantity, bounded across time
    within a field instead of across fields within a date. Two fields that differ in
    canopy structure have different absolute LST, so the temporal version is the one
    that does not mistake "sandy soil" for "irrigation failure" — and having both in
    the benchmark is what lets that sentence be tested rather than asserted.
    """
    names = [str(c).lower() for c in channel_names]
    ci_l = _col(names, "lst")
    if ci_l is None:
        return IndexScore("CWSI_LST", None, False, missing=("lst",), note="no LST channel")
    a = np.asarray(x, dtype=np.float64)[:, :, ci_l]
    n, T = int(a.shape[0]), int(a.shape[1])
    obs = np.isfinite(a)
    fit = np.asarray(fit_idx, dtype=np.int64)
    sc = np.asarray(scored_idx, dtype=np.int64)
    rec = _check_fit_disjoint(fit, sc, who="CWSI_LST", n_obs=T)
    out = np.full(a.shape, np.nan)
    lo = np.full(n, np.nan)
    hi = np.full(n, np.nan)
    fit_rows = np.isin(np.arange(n), fit)
    pooled = a[fit_rows][obs[fit_rows]]
    p_lo, p_hi = ((float(np.quantile(pooled, clip_q)), float(np.quantile(pooled, 1.0 - clip_q)))
                  if pooled.size >= 32 else (np.nan, np.nan))
    n_pooled = 0
    for i in range(n):
        v = a[i][obs[i] & fit_rows[i]]
        if v.size >= 8:
            lo[i], hi[i] = float(np.quantile(v, clip_q)), float(np.quantile(v, 1.0 - clip_q))
        elif np.isfinite(p_lo):
            # A field with no history of its own (a newly irrigated field, or one whose
            # cycles all fall in the scored window) is bounded by the pooled envelope
            # instead of being refused — but the substitution is counted and reported,
            # because a within-field normalisation and a cross-field one are different
            # statements wearing the same column name.
            lo[i], hi[i] = p_lo, p_hi
            n_pooled += 1
    span = hi - lo
    ok = np.isfinite(span) & (span > 0)
    out[ok] = (a[ok] - lo[ok, None]) / span[ok, None]
    rec.update({"n_fields_bounded": int(ok.sum()), "n_fields_total": int(n),
                "pooled_fill": int(n_pooled),
                "pooled_fill_note": ("fields bounded on the pooled fit-partition envelope because they "
                                     "had no usable own-field history"),
                "clip_quantile": float(clip_q),
                "note": "bounds fitted on the fit partition only (field history), then applied everywhere"})
    return IndexScore("CWSI_LST", np.clip(out, 0.0, 1.0), bool(np.isfinite(out[sc]).any()),
                      fit_record=rec,
                      note=(f"{int((~ok).sum())} fields had too little history to bound and are NaN; "
                            "an index that silently reports 0 for them would read as 'no stress'"))


def _weather_series(weather: dict | None, field: str, *, n: int, T: int) -> np.ndarray | None:
    """Per-cycle, per-bin summary of a weather field, aligned by row order (no extrapolation)."""
    if not weather:
        return None
    v = weather.get(field)
    if v is None:
        return None
    arr = np.asarray(v, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.repeat(arr[:, None], max(1, (n * T) // max(1, arr.size)), axis=1).reshape(-1)
        if arr.size < n * T:
            return None
        arr = arr[: n * T].reshape(n, T)
    elif arr.shape[:2] != (n, T):
        return None
    return arr if np.isfinite(arr).any() else None


def evapotranspiration(x, channel_names, weather, *, fit_idx, scored_idx,
                       kc_max: float = 1.2, kind: str = "ETa") -> IndexScore:
    """ETa and ETrF from a grass-reference ET series and an NDVI-based crop coefficient.

    ``ETa = Kc(NDVI) · ET0`` and ``ETrF = ETa / ET0``; Kc is the linear NDVI ramp the
    FAO-56 do-everything table approximates, and the ramp's endpoints are the only free
    parameters, fitted on the **train** rows. Without an ET0 series neither number
    exists, and this function says so rather than substituting LST — the substitution
    would turn an evapotranspiration index into a renamed temperature index while
    keeping the label, which is how a circularity gets published by accident.
    """
    need = "et0"
    a = np.asarray(x, dtype=np.float64)
    n, T = int(a.shape[0]), int(a.shape[1])
    et0 = _weather_series(weather, need, n=n, T=T)
    names = [str(c).lower() for c in channel_names]
    ci_n = _col(names, "ndvi")
    missing = tuple(k for k, ok in ((need, et0 is not None), ("ndvi", ci_n is not None)) if not ok)
    if missing:
        return IndexScore(kind, None, False, missing=missing,
                          note=(f"needs {missing}; the archive/bundle carries no ET0 sidecar, and ETa "
                                "is not computed from LST as a fallback"))
    ndvi = a[:, :, ci_n] if ci_n is not None else np.zeros(a.shape[:2])
    obs = np.isfinite(ndvi)
    fit = np.asarray(fit_idx, dtype=np.int64)
    sc = np.asarray(scored_idx, dtype=np.int64)
    rec = _check_fit_disjoint(fit, sc, who=kind, n_obs=T)
    f_ndvi = ndvi[fit]
    lo, hi = (float(np.nanquantile(f_ndvi, 0.05)), float(np.nanquantile(f_ndvi, 0.95)))
    if not (hi > lo):
        return IndexScore(kind, None, False, missing=(),
                          note=f"NDVI range on the fit partition is degenerate ({lo}–{hi})")
    kc = np.clip((ndvi - lo) / (hi - lo), 0.0, 1.0) * (kc_max - 0.1) + 0.1
    eta = kc * et0
    rec.update({"kc": {"min": 0.1, "max": float(kc_max), "ndvi_lo": round(lo, 4), "ndvi_hi": round(hi, 4),
                       "shape": "linear FAO-56 do-bullet approximation", "fitted_on": "train rows"},
                "et0_source": "weather sidecar field 'et0'"})
    out = eta if kind == "ETa" else np.where(et0 > 0, eta / np.maximum(et0, 1e-9), np.nan)
    # Stress is *low* ETa, so the reference score is the deficit: 1 − ratio, oriented so
    # that higher means more stressed like every other row in the same table.
    if kind == "ETrF":
        out = 1.0 - np.nan_to_num(out, nan=1.0)
        note = "reported as 1 − ETrF (evaporative deficit) so that higher = more stressed"
    else:
        mx = float(np.nanmax(eta[fit])) if np.isfinite(eta[fit]).any() else float("nan")
        out = 1.0 - (eta / mx if np.isfinite(mx) and mx > 0 else np.zeros_like(eta))
        note = (f"reported as 1 − ETa/max(ETa on the fit partition) (relative deficit) so that "
                "higher = more stressed")
    return IndexScore(kind, out, bool(np.isfinite(out[sc]).any()), fit_record=rec, note=note)


def compute_all(x, channel_names, *, weather=None, fit_idx, scored_idx,
                ndvi_bins: int = 12, min_support: int = 8) -> dict[str, IndexScore]:
    out = {
        "TVDI": tvdi(x, channel_names, fit_idx=fit_idx, scored_idx=scored_idx, ndvi_bins=ndvi_bins,
                       min_support=min_support),
        "CWSI_LST": cwsi_lst(x, channel_names, fit_idx=fit_idx, scored_idx=scored_idx),
    }
    out["ETa"] = evapotranspiration(x, channel_names, weather, fit_idx=fit_idx, scored_idx=scored_idx,
                                    kind="ETa")
    out["ETrF"] = evapotranspiration(x, channel_names, weather, fit_idx=fit_idx, scored_idx=scored_idx,
                                     kind="ETrF")
    out["CWSI_ET"] = IndexScore(
        "CWSI_ET", None, False, missing=tuple(NEEDS["CWSI_ET"]),
        note=("CWSI_ET needs the aerodynamic resistance term (ET actual + ET potential under the "
              "same conditions), which the v2 bundle does not carry; the row is reported unavailable "
              "rather than approximated by CWSI_LST — the two differ exactly in the part that would "
              "be invented"))
    return out


def index_agreement(scores_bin: np.ndarray, index_scores: np.ndarray, *, farm_ids: np.ndarray,
                    years: np.ndarray | None = None, n_boot: int = 2000, seed: int = 2026) -> dict:
    """Task 9.18: block-bootstrap Spearman/Kendall between an arm and an index.

    The contract names the dependence form ``block_bootstrap_by_field_year``: when
    ``years`` is supplied, whole (field, year) blocks are resampled — a field's cycles
    inside one year share weather and management, so they are one draw, not several.
    Without ``years`` the bootstrap falls back to field blocks and the record says so,
    rather than quietly claiming the narrower unit. Reported with the interval, not just
    the point estimate: if the arm and the index agree to ρ=0.4 with a CI spanning
    0.2–0.6, "the model is redundant" is not a conclusion the data supports.
    """
    from scipy import stats as sps

    s = np.asarray(scores_bin, dtype=np.float64).ravel()
    v = np.asarray(index_scores, dtype=np.float64).ravel()
    f = np.asarray(farm_ids).ravel()
    ok = np.isfinite(s) & np.isfinite(v)
    if int(ok.sum()) < 64:
        return {"available": False, "reason": f"{int(ok.sum())} paired finite bins"}
    s, v, f = s[ok], v[ok], f[ok]
    if years is not None:
        y = np.asarray(years).ravel()
        if y.size != f.size:
            return {"available": False, "reason": (f"years ({y.size}) and farm_ids ({f.size}) "
                    "disagree; misaligned block keys would resample the wrong rows")}
        keys = np.stack([f, y], axis=1)
        uniq = np.unique(keys, axis=0)
        if uniq.shape[0] < 8:
            return {"available": False,
                    "reason": (f"{uniq.shape[0]} distinct (field, year) blocks; a block bootstrap over "
                               "fewer than 8 clusters would report a confidence interval with no meaning"),
                    "n_farms": int(np.unique(f).size), "resampled_clustered_by": "farm_year"}
        idx_by_block = {tuple(int(x) for x in k): np.flatnonzero((keys == k).all(axis=1))
                        for k in uniq}
        block_keys = [tuple(int(x) for x in k) for k in uniq]
        clustered_by = "farm_year"
    else:
        uniq = np.unique(f)
        if uniq.size < 8:
            return {"available": False,
                    "reason": (f"{uniq.size} distinct farm cluster(s); a block bootstrap over fewer than 8 "
                               "clusters would report a confidence interval with no meaning"),
                    "n_farms": int(uniq.size), "resampled_clustered_by": "farm"}
        idx_by_block = {int(k): np.flatnonzero(f == k) for k in uniq}
        block_keys = [int(k) for k in uniq]
        clustered_by = "farm"
    rho, tau = sps.spearmanr(s, v).statistic, sps.kendalltau(s, v).statistic
    rng = np.random.default_rng(int(seed))
    rr, tt = [], []
    for _ in range(int(n_boot)):
        pick = rng.choice(len(block_keys), size=len(block_keys), replace=True)
        ii = np.concatenate([idx_by_block[block_keys[k]] for k in pick])
        a, b = s[ii], v[ii]
        if np.unique(a).size < 3 or np.unique(b).size < 3:
            continue
        rr.append(sps.spearmanr(a, b).statistic)
        tt.append(sps.kendalltau(a, b).statistic)
    rr, tt = np.array(rr), np.array(tt)

    def _ci(a):
        return [round(float(np.quantile(a, 0.025)), 4), round(float(np.quantile(a, 0.975)), 4)] if a.size else [None, None]

    note = ("correlations are of pooled bins; the block bootstrap is what makes the interval "
            "honest, since bins within a block are not independent draws")
    if clustered_by == "farm":
        note += (" — the contract's field-year form was NOT used: no year axis was supplied, "
                 "so whole fields are the resampling unit and the interval is the wider of the two")
    return {"available": bool(rr.size >= 20), "spearman": round(float(rho), 4),
            "spearman_ci": _ci(rr), "kendall": round(float(tau), 4), "kendall_ci": _ci(tt),
            "n_bins": int(s.size), "n_farms": int(np.unique(f).size),
            "n_blocks": int(len(block_keys)), "bootstrap_draws": int(rr.size),
            "resampled_clustered_by": clustered_by,
            "note": note}


def reference_metrics(index: IndexScore, labels_bin: np.ndarray, *, eval_idx: np.ndarray,
                      scorable: np.ndarray | None = None) -> dict:
    """AUPRC/AUROC for one index on the same labels the arms are scored on."""
    from ssl_anomaly.benchmarking.metrics import ap, auc

    if not index.available or index.scores is None:
        return {"index": index.name, "available": False, "reason": (f"missing fields {list(index.missing)}"
                                                                    if index.missing else index.note)}
    sc = np.asarray(index.scores, dtype=np.float64)[eval_idx]
    y = np.asarray(labels_bin, dtype=bool)[eval_idx]
    ok = np.isfinite(sc) if scorable is None else (np.isfinite(sc) & np.asarray(scorable)[eval_idx])
    if int(ok.sum()) < 16 or not y[ok].any():
        return {"index": index.name, "available": False,
                "reason": f"{int(ok.sum())} scorable bins with {int(y.sum())} positive bins"}
    return {"index": index.name, "available": True, "auprc": round(ap(sc[ok], y[ok].astype(int)), 5),
            "auroc": round(auc(sc[ok], y[ok].astype(int)), 5), "n_bins": int(ok.sum()),
            "prevalence": round(float(y[ok].mean()), 5), "higher_is_stress": True,
            "fit": index.fit_record, "note": index.note}

"""Per-arm anomaly scoring: bin → cycle → farm, plus the patch level if bound (tasks 2.1, 5.3).

One rule governs this module: **the score is the arm's own objective evaluated on
observed data**. A reconstruction arm scores residuals; a prediction arm scores
forecast errors; a JEPA arm scores latent-prediction errors. The scoring code never
re-parameterises a score into something comparable-looking without saying so — the
per-channel MAD normalisation here is recorded in the manifest with the partition
it was fitted on (train only), because a score whose scale silently differs per
channel is a hidden aggregation choice.

Levels (spec experimental-validation / "AUPRC as primary metric"):
``point`` = per (cycle, bin); ``cycle`` = per cycle; ``event`` = per injected event;
``patch`` = per (patch, bin) and ``farm`` = per (farm, bin) aggregated from
patches. Patch-level scoring short-circuits to :class:`ScoringUnavailable` while
the v2 patch store is unbound, so the runner records ``Deferred`` rather than
running a substitute.

The farm aggregation is the frozen one — channel-max over patches inside the bin,
then a trailing window — because inventing a second aggregation for the benchmark
would make the patch numbers incomparable with findings 23–25. Attention pooling is
available as a documented alternative and is *not* the default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NAN = float("nan")


class ScoringUnavailable(RuntimeError):
    """Patch-level scoring was asked for while the v2 patch store is unbound."""


@dataclass
class ScoreBundle:
    """Everything a metric needs for one (cell, split) scoring pass.

    ``bin_scores`` is ``(N, T)`` with NaN where a bin has no observed channel to
    score — those bins contribute to *nothing*, they are not zero-scored, which is
    the difference between a mask-aware score and a pessimistic one.
    """

    bin_scores: np.ndarray
    cycle_scores: np.ndarray
    bin_labels: np.ndarray | None = None
    cycle_labels: np.ndarray | None = None
    channels: np.ndarray | None = None
    channel_names: tuple[str, ...] = ()
    #: per-cycle event ids (``field:cycle``) so event-level metrics can group.
    event_ids: np.ndarray | None = None
    #: bin index of the first true label per event, for detection delay.
    onset_bin: np.ndarray | None = None
    field_ids: np.ndarray | None = None
    aux: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        b = np.asarray(self.bin_scores, dtype=np.float64)
        if b.ndim != 2:
            raise ValueError(f"bin_scores must be (N, T), got {b.shape}")
        if b.shape[0] != len(self.cycle_scores):
            raise ValueError(
                f"cycle_scores has {len(self.cycle_scores)} entries for {b.shape[0]} cycles"
            )
        if not np.any(np.isfinite(b)):
            raise ValueError(
                "every score is NaN: an all-unscorable pass is a data/binding error, not a result"
            )

    @property
    def n_cycles(self) -> int:
        return int(self.bin_scores.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.bin_scores.shape[1])

    def scorable_frac(self) -> float:
        return float(np.isfinite(self.bin_scores).mean())

    def to_dict(self) -> dict:
        return {
            "n_cycles": self.n_cycles,
            "n_bins": self.n_bins,
            "scorable_bin_frac": round(self.scorable_frac(), 4),
            "channel_names": list(self.channel_names),
            **self.aux,
        }


# ---------------------------------------------------------------------------
# Core reductions
# ---------------------------------------------------------------------------


def masked_bin_error(
    err: np.ndarray,
    mask: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """(B, T, C) residual → (B, T) score over OBSERVED channels only.

    ``scale`` is the per-channel MAD used to make channels commensurable; it is
    fitted on train data by the caller and is never recomputed on the scored
    partition. A bin with no observed channel returns NaN.
    """
    e = np.asarray(err, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if e.shape != m.shape:
        raise ValueError(f"err {e.shape} and mask {m.shape} must align")
    if scale is not None:
        s = np.asarray(scale, dtype=np.float64).reshape(1, 1, -1)
        if s.shape[-1] != e.shape[-1]:
            raise ValueError(f"scale {s.shape} must match channels {e.shape[-1]}")
        e = e / np.maximum(s, eps)
    num = np.where(m, e, 0.0)
    den = m.sum(axis=-1).astype(np.float64)
    if weights is not None:
        num = num * np.asarray(weights, dtype=np.float64)
        den = den * np.asarray(weights, dtype=np.float64).sum(axis=-1)
    out = np.full(e.shape[:2], NAN)
    ok = den > 0
    out[ok] = num.sum(axis=-1)[ok] / den[ok]
    return out


def aggregate_bins(
    bin_scores: np.ndarray,
    *,
    valid: np.ndarray | None = None,
    method: str = "max",
    trailing_window: int = 1,
) -> np.ndarray:
    """(N, T) → (N,) cycle score. NaN bins are skipped, never zero-filled.

    ``max``     — the frozen semantics: one anomalous bin makes an anomalous cycle.
    ``topk``    — mean of the k largest (robust to a single spike).
    ``mean``    — dilutes a short event; kept for the report, not the headline.
    ``trailing``— max over the last ``trailing_window`` bins: the operational
                  early-warning reading at cycle end.
    """
    b = np.asarray(bin_scores, dtype=np.float64)
    v = np.isfinite(b) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(b))
    bb = np.where(v, b, -np.inf)
    n = b.shape[0]
    if method == "max":
        out = bb.max(axis=1)
    elif method == "topk":
        k = max(1, int(trailing_window))
        srt = -np.sort(-bb, axis=1)[:, :k]
        out = np.nanmean(np.where(np.isneginf(srt), np.nan, srt), axis=1)
    elif method == "mean":
        out = np.array([np.nanmean(np.where(v[i], b[i], np.nan)) for i in range(n)])
    elif method == "trailing":
        w = max(1, int(trailing_window))
        tail = bb[:, -w:]
        out = tail.max(axis=1)
    else:
        raise ValueError(f"unknown cycle aggregation {method!r}")
    return np.where(np.isneginf(out), NAN, out)


def patch_to_farm(
    patch_scores: np.ndarray,
    patch_valid: np.ndarray,
    *,
    method: str = "channel_max",
    trailing_window: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """(P, T) patch scores with patch membership → farm-level bin scores.

    ``channel_max`` (the frozen rule) is a max over the farm's valid patches per
    bin. Invalid patches are excluded from numerator AND denominator: a missing
    patch must not lower a farm's mean. Returns ``(farm_bin_scores, farm_valid)``
    where a farm-bin with no valid patch is NaN (absent), never 0.
    """
    p = np.asarray(patch_scores, dtype=np.float64)
    ok = np.asarray(patch_valid, dtype=bool) & np.isfinite(np.asarray(patch_scores, dtype=np.float64))
    if p.ndim != 2 or ok.shape != p.shape:
        raise ValueError("patch_scores must be (P, T) and patch_valid must match")
    if method == "channel_max":
        masked = np.where(ok, p, -np.inf)
        out = masked.max(axis=0)
        out = np.where(np.isneginf(out), NAN, out)
        valid = np.isfinite(out)
        return out, valid
    if method == "trailing_max":
        w = max(1, int(trailing_window))
        T = p.shape[1]
        out = np.full(T, NAN)
        for b in range(T):
            lo = max(0, b - w + 1)
            win = np.where(ok[:, lo : b + 1], p[:, lo : b + 1], -np.inf)
            v = win.max(axis=1)
            if np.any(np.isfinite(v)):
                out[b] = float(np.nanmax(v[np.isfinite(v)]))
        return out, np.isfinite(out)
    if method == "mean":
        masked = np.where(ok, p, np.nan)
        with np.errstate(invalid="ignore"):
            out = np.nanmean(masked, axis=0)
        out = np.where(np.isfinite(out), out, NAN)
        return out, np.isfinite(out)
    raise ValueError(f"unknown patch→farm aggregation {method!r}")


def attention_pool(
    patch_scores: np.ndarray, patch_valid: np.ndarray, *, temperature: float = 1.0
) -> np.ndarray:
    """Documented alternative aggregation (pre-registered, off by default).

    Softmax over patches per bin, restricted to valid patches. Exposed so the
    aggregation choice can be ablated, but the headline stays on ``channel_max``
    for comparability with the frozen patch findings.
    """
    p = np.asarray(patch_scores, dtype=np.float64)
    ok = np.asarray(patch_valid, dtype=bool) & np.isfinite(p)
    T = p.shape[1]
    out = np.full(T, NAN)
    for b in range(T):
        col = p[ok[:, b], b]
        if col.size == 0:
            continue
        z = (col - col.max()) / max(float(temperature), 1e-6)
        w = np.exp(z)
        w /= w.sum()
        out[b] = float(np.sum(w * col))
    return out


# ---------------------------------------------------------------------------
# Normalisation of scores (train-only MAD)
# ---------------------------------------------------------------------------


@dataclass
class ScoreNormaliser:
    """Per-channel robust scale fitted on the TRAIN partition only.

    Stored with its provenance because "which partition normalised the scores" is
    exactly the kind of thing a committee asks, and a scale fitted on the scored
    partition would be leakage dressed as preprocessing.
    """

    scale: np.ndarray
    fitted_on: str
    n_cycles: int
    n_observations: int

    def __post_init__(self) -> None:
        s = np.asarray(self.scale, dtype=np.float64)
        if s.ndim != 1 or s.size == 0:
            raise ValueError("scale must be a non-empty 1-D array")
        if np.any(~np.isfinite(s)) or np.any(s <= 0):
            raise ValueError(
                "scale entries must be finite and > 0; a 0 or NaN scale would silently "
                "delete or explode a channel from the score"
            )

    @classmethod
    def from_errors(
        cls, train_err: np.ndarray, train_mask: np.ndarray, *, fitted_on: str = "train",
        n_cycles: int | None = None,
    ) -> ScoreNormaliser:
        """MAD-per-channel of the clean train residuals (no median subtraction:
        the score's zero must stay 'reconstructed/predicted exactly', not 'typical')."""
        e = np.asarray(train_err, dtype=np.float64)
        m = np.asarray(train_mask, dtype=bool)
        if e.shape != m.shape:
            raise ValueError("train_err and train_mask must align")
        C = e.shape[-1]
        scale = np.full(C, NAN)
        for c in range(C):
            sel = m[..., c]
            if not sel.any():
                continue
            v = e[..., c][sel]
            scale[c] = float(np.median(np.abs(v - np.median(v))) * 1.4826) if v.size else NAN
        bad = ~np.isfinite(scale) | (scale <= 0)
        if bad.all():
            raise ValueError("no channel had usable train residuals to normalise against")
        if bad.any():
            # A channel with no train observations cannot be scaled; give it zero
            # weight by setting an infinite scale, which drops it from the mean
            # rather than pretending it is noise-free.
            scale[bad] = np.inf
        return cls(
            scale=scale,
            fitted_on=fitted_on,
            n_cycles=int(n_cycles if n_cycles is not None else e.shape[0]),
            n_observations=int(m.sum()),
        )

    def to_dict(self) -> dict:
        return {
            "scale": [None if not np.isfinite(v) else float(v) for v in self.scale],
            "fitted_on": self.fitted_on,
            "n_cycles": self.n_cycles,
            "n_observations": self.n_observations,
            "estimator": "MAD x 1.4826 of clean train residuals, no median subtraction",
        }


# ---------------------------------------------------------------------------
# Onsets / delay
# ---------------------------------------------------------------------------


def event_onsets(
    bin_labels: np.ndarray, field_ids: np.ndarray | None = None
) -> list[dict]:
    """Contiguous True runs per cycle → event records with onset/offset bins.

    Two runs separated by a single unobserved bin are reported as two events (the
    generator's own boundaries are authoritative; this function only recovers them
    from masks when the generator metadata is unavailable).
    """
    y = np.asarray(bin_labels, dtype=bool)
    out: list[dict] = []
    for i in range(y.shape[0]):
        row = y[i]
        j = 0
        while j < row.size:
            if not row[j]:
                j += 1
                continue
            k = j
            while k + 1 < row.size and row[k + 1]:
                k += 1
            out.append(
                {
                    "cycle_index": int(i),
                    "field_id": int(field_ids[i]) if field_ids is not None else None,
                    "onset_bin": int(j),
                    "offset_bin": int(k),
                    "duration_bins": int(k - j + 1),
                }
            )
            j = k + 1
    return out


def detection_delay(
    bin_scores: np.ndarray,
    bin_labels: np.ndarray,
    *,
    threshold: float,
    bin_days: int = 15,
    localized_only: bool | None = None,
) -> dict:
    """Median days from the FIRST anomalous bin to the first alert, per event.

    ``localized`` events are short (≤ 3 bins) and ``persistent`` ones longer; the
    distinction is by duration, not family, because the design's delay metric names
    the two shapes. An event never alerted contributes NaN and is counted in
    ``missed``; the median is over alerted events only, and ``n_alerted`` is
    reported beside it so a delay computed on 2 of 40 events cannot be read as a
    general one.
    """
    s = np.asarray(bin_scores, dtype=np.float64)
    y = np.asarray(bin_labels, dtype=bool)
    if s.shape != y.shape:
        raise ValueError("bin_scores and bin_labels must align")
    delays: list[float] = []
    missed = 0
    n = 0
    for i in range(s.shape[0]):
        if not y[i].any():
            continue
        js = 0
        while js < y.shape[1]:
            if not y[i, js]:
                js += 1
                continue
            k = js
            while k + 1 < y.shape[1] and y[i, k + 1]:
                k += 1
            dur = k - js + 1
            localized = dur <= 3 if localized_only is None else bool(localized_only)
            row = s[i, js : k + 1]
            alert = np.argmax(row >= threshold) if np.any(row >= threshold) else -1
            n += 1
            if alert < 0:
                missed += 1
            else:
                delays.append(float(alert * bin_days))
            js = k + 1
    med = float(np.median(delays)) if delays else NAN
    return {
        "median_days": med,
        "mean_days": float(np.mean(delays)) if delays else NAN,
        "q25_days": float(np.percentile(delays, 25)) if delays else NAN,
        "q75_days": float(np.percentile(delays, 75)) if delays else NAN,
        "n_events": n,
        "n_alerted": len(delays),
        "missed": missed,
        "threshold": float(threshold),
        "bin_days": int(bin_days),
        "event_class": ("localized" if localized_only is True
                        else "persistent" if localized_only is False else "all"),
    }

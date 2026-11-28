"""The pre-registered B1–B5 incremental INPUT ladder (change tasks 1.1–1.5).

This module implements the ladder EXACTLY as frozen in
``configs/evaluation/pre_registration.yaml`` (``ladder:``) and design.md
§"B1–B5 incremental input ladder":

=====  ==========================  ===============================================
rung   inputs                      model
=====  ==========================  ===============================================
B1     NDVI + LST only             rule: per-field robust-MAD NDVI-drop AND/OR
                                   LST-rise excursion, FAR-matched to the 5 %
                                   cycle-FAR budget on the clean calibration
                                   subset (no evaluation tuning)
B2     NDVI + LST + mask + DOY     SSL encoder, trunk/budget identical to B3–B5
B3     all 8 spectral+thermal      SSL encoder
       channels + mask + DOY
B4     B3 + daily ERA5 (t2m/tmean, SSL encoder
       precip, solar radiation,
       VPD, et0), causally masked
B5     B4 + DSH sin/cos + phase    SSL encoder
       bin + STL trend/season/
       residual
=====  ==========================  ===============================================

Honesty rules encoded here (AGENTS.md §1/§7):

- Every rung exposes the SAME scoring bridge (``score(field, bin) -> float``),
  so evaluation is identical across rungs (task 1.5).
- Causality is a TESTED property (task 1.3): no feature at bin *o* may
  reference bins > *o*. A full-series STL decomposition is NOT causal, which is
  one of the two recorded reasons the B5 STL block is ``absent`` rather than
  approximated (the other: the decomposition-aware-ssl-study has produced no
  canonical polygon-zonal store yet).
- A rung whose declared inputs are missing is reported ``absent`` with the
  missing names — never a zero, never a subset relabelled as the full rung.
- The B1 thresholds come from the clean calibration subset ONLY; changing the
  evaluation cycles cannot move them (proved by a unit test, not asserted).
- ``ndvi_lst_rule`` is the A-rule/B1 control the advisor demands be treated as
  a genuinely strong competitor; both AND and OR variants are reported and
  NEITHER is selected post hoc.

Note on the OTHER ladder in this package: ``ssl_anomaly.evaluation.ladder`` (and
``scripts/evaluation/run_b1_b5_ladder.py``) implement a different, older
decomposition (frozen-incumbent vs encoder vs hand-crafted blocks). That module
answers a complementary attribution question; the PRE-REGISTERED capstone
ladder is THIS module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

#: Robust-z scale factor: MAD * 1.4826 estimates the standard deviation for
#: normal data (Gaussian consistency factor).
MAD_SCALE = 1.4826

#: Channels of the v2 8-channel contract, in the locked channel order
#: (verified against ``norm_stats_full_v2.json`` ``channel_order``).
V2_CHANNELS: tuple[str, ...] = (
    "ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst",
)

#: B1 uses exactly these two physical readouts — no learning, no other channel.
B1_CHANNELS: tuple[str, ...] = ("ndvi", "lst")

#: B4's daily ERA5 block, in column order (design §ladder).
ERA5_CHANNELS: tuple[str, ...] = ("t2m", "tmean", "precip", "solar_radiation", "vpd", "et0")

#: B5's phenological block: DSH sin/cos + phase bin (computed here, causal) and
#: the STL trend/seasonal/residual triple (supplied by the decomposition study;
#: absent until that study produces them).
B5_STL_FEATURES: tuple[str, ...] = ("stl_trend", "stl_seasonal", "stl_residual")

#: Pre-registered cycle-FAR budget (pre_registration.yaml evaluation.cycle_far_target).
CYCLE_FAR_TARGET = 0.05

#: Bins per DSH phase bin (36 bins x 15 d over the 540-day phase axis, data.py).
PHASE_BIN_DAYS = 15.0
N_PHASE_BINS = 36


# ---------------------------------------------------------------------------
# Rung declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InputRung:
    """One rung's declared model inputs (the pre-registration ``ladder:`` table)."""

    name: str
    channels: tuple[str, ...]
    uses_mask: bool
    uses_doy: bool
    weather: bool
    phenology: bool
    stl: bool
    model: str                 # 'rule: ndvi_lst_rule' | 'ssl'
    learning: bool
    isolates: str

    @property
    def input_channels(self) -> tuple[str, ...]:
        return self.channels

    def required_external_blocks(self) -> tuple[str, ...]:
        """Blocks that must be SUPPLIED (not computed from the cycle table)."""
        req = []
        if self.weather:
            req.append("era5_daily")
        if self.stl:
            req.append("decomposition_features")
        return tuple(req)


RUNGS: dict[str, InputRung] = {
    "B1": InputRung(
        "B1", B1_CHANNELS, uses_mask=True, uses_doy=False, weather=False,
        phenology=False, stl=False, model="rule: ndvi_lst_rule", learning=False,
        isolates="two direct physical variables, zero learning",
    ),
    "B2": InputRung(
        "B2", B1_CHANNELS, uses_mask=True, uses_doy=True, weather=False,
        phenology=False, stl=False, model="ssl", learning=True,
        isolates="learning value on the same two variables",
    ),
    "B3": InputRung(
        "B3", V2_CHANNELS, uses_mask=True, uses_doy=True, weather=False,
        phenology=False, stl=False, model="ssl", learning=True,
        isolates="value of full optical/thermal input",
    ),
    "B4": InputRung(
        "B4", V2_CHANNELS, uses_mask=True, uses_doy=True, weather=True,
        phenology=False, stl=False, model="ssl", learning=True,
        isolates="value of meteorological context",
    ),
    "B5": InputRung(
        "B5", V2_CHANNELS, uses_mask=True, uses_doy=True, weather=True,
        phenology=True, stl=True, model="ssl", learning=True,
        isolates="value of temporal/phenological conditioning",
    ),
}


def ladder_table() -> list[dict]:
    rows = []
    for name in ("B1", "B2", "B3", "B4", "B5"):
        r = RUNGS[name]
        rows.append({
            "name": r.name,
            "inputs": {
                "channels": list(r.channels),
                "mask": r.uses_mask,
                "doy": r.uses_doy,
                "era5_daily": list(ERA5_CHANNELS) if r.weather else None,
                "phenology": (["dsh_sin", "dsh_cos", "phase_bin"] if r.phenology else None),
                "stl": list(B5_STL_FEATURES) if r.stl else None,
            },
            "model": r.model,
            "learning": r.learning,
            "isolates": r.isolates,
        })
    return rows


def ladder_reading_rule() -> str:
    """The pre-registered reading rule (design §ladder), verbatim in intent."""
    return (
        "if B2 ≈ B1, learning adds nothing on the two physical variables; "
        "if B3 > B2, spectral breadth adds value; if B4 > B3, weather adds value; "
        "if B5 > B4, decomposition context adds value. If none clear B1 under CI, "
        "the honest thesis statement is that the simple algorithm wins on "
        "detection at far lower cost."
    )


# ---------------------------------------------------------------------------
# Task 1.1 — the B1 rule arm: per-field robust-MAD NDVI+LST excursion
# ---------------------------------------------------------------------------


@dataclass
class FieldStats:
    """One field's robust calibration statistics for the B1 channels."""

    field_id: int
    ndvi_median: float
    ndvi_mad: float
    lst_median: float
    lst_mad: float
    degenerate: bool = False     # MAD ≈ 0 on a channel -> pooled-scale fallback
    fallback_channels: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "field_id": self.field_id,
            "ndvi_median": self.ndvi_median,
            "ndvi_mad": self.ndvi_mad,
            "lst_median": self.lst_median,
            "lst_mad": self.lst_mad,
            "degenerate": self.degenerate,
            "fallback_channels": list(self.fallback_channels),
        }


@dataclass
class B1Calibration:
    """One variant's calibration record — everything needed to audit the rule."""

    variant: str                       # 'and' | 'or'
    threshold: float
    target_cycle_far: float
    achieved_cycle_far: float
    n_calibration_cycles: int
    n_fields: int
    degenerate_fields: int
    scale_fallback: str                # how degenerate fields were handled
    per_field: dict[int, FieldStats] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "threshold": self.threshold,
            "target_cycle_far": self.target_cycle_far,
            "achieved_cycle_far": self.achieved_cycle_far,
            "n_calibration_cycles": self.n_calibration_cycles,
            "n_fields": self.n_fields,
            "degenerate_fields": self.degenerate_fields,
            "scale_fallback": self.scale_fallback,
            "per_field": {str(k): v.to_dict() for k, v in self.per_field.items()},
        }


def robust_mad(values: np.ndarray) -> tuple[float, float]:
    """Median and scaled MAD of the finite entries (NaN-safe)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * MAD_SCALE
    return med, mad


def field_robust_stats(
    x: np.ndarray, m: np.ndarray, field_ids: np.ndarray,
    cycle_ids: np.ndarray, calibration_cycle_ids: np.ndarray,
    *,
    channels: tuple[str, ...] = B1_CHANNELS,
    channel_names: tuple[str, ...] = V2_CHANNELS,
) -> tuple[dict[int, FieldStats], dict[str, float]]:
    """Per-field robust median/MAD from the CALIBRATION cycles only.

    ``channels[0]`` is the drop channel (NDVI), ``channels[1]`` the rise channel
    (LST). A field whose calibration MAD on a channel is degenerate (≈ 0: the
    field is constant there) would make every deviation infinite; the scale for
    that field+channel falls back to the POOLED across-field MAD, and the
    fallback is recorded in the returned stats — a declared fallback, never a
    silent one.
    """
    ch_idx = [channel_names.index(c) for c in channels]
    cal_mask = np.isin(cycle_ids, np.asarray(calibration_cycle_ids))
    if not cal_mask.any():
        raise ValueError(
            "no calibration cycles present in the table; refusing to calibrate "
            "on an empty (or implicitly 'all') subset"
        )
    pooled: dict[str, float] = {}
    per_field_stats: dict[int, dict[str, tuple[float, float]]] = {}
    for fid in np.unique(field_ids):
        sel = cal_mask & (field_ids == fid)
        if not sel.any():
            continue                      # field has no calibration cycle: excluded, recorded
        per_field_stats[int(fid)] = {}
        for name, ci in zip(channels, ch_idx):
            vals = x[sel][:, :, ci][m[sel][:, :, ci]]
            per_field_stats[int(fid)][name] = robust_mad(vals)
    for name in channels:
        # Pooled scale comes from fields with a usable (non-degenerate) MAD;
        # pooling degenerate zeros would make the fallback scale 0 itself.
        usable_mads = [mad for d in per_field_stats.values()
                       for _, mad in [d[name]]
                       if np.isfinite(mad) and mad >= 1e-8]
        if not usable_mads:
            raise ValueError(
                f"channel {name!r} has no field with a usable calibration MAD; "
                "the rule cannot be scaled (refusing a made-up scale)"
            )
        pooled[name] = float(np.median(usable_mads))

    stats: dict[int, FieldStats] = {}
    for fid, d in per_field_stats.items():
        (n_med, n_mad), (l_med, l_mad) = d[channels[0]], d[channels[1]]
        fallbacks = []
        degenerate = False
        resolved = {"ndvi": (n_med, n_mad), "lst": (l_med, l_mad)}
        for nm, key in ((channels[0], "ndvi"), (channels[1], "lst")):
            med, mad = resolved[key]
            if not np.isfinite(mad) or mad < 1e-8:
                degenerate = True
                fallbacks.append(nm)
                resolved[key] = (med, pooled[nm])
        stats[int(fid)] = FieldStats(
            field_id=int(fid),
            ndvi_median=resolved["ndvi"][0], ndvi_mad=resolved["ndvi"][1],
            lst_median=resolved["lst"][0], lst_mad=resolved["lst"][1],
            degenerate=degenerate, fallback_channels=tuple(fallbacks),
        )
    return stats, pooled


def variant_scores(
    x: np.ndarray, m: np.ndarray, stats: dict[int, FieldStats],
    field_ids: np.ndarray,
    *,
    channels: tuple[str, ...] = B1_CHANNELS,
    channel_names: tuple[str, ...] = V2_CHANNELS,
) -> dict[str, np.ndarray]:
    """Per-bin robust-z excursion scores for BOTH variants (never one alone).

    For cycle *i*, bin *o*:
      ``ndvi_z = (field_ndvi_median - ndvi[i,o]) / ndvi_mad``   (a DROP is positive)
      ``lst_z  = (lst[i,o] - field_lst_median) / lst_mad``       (a RISE is positive)
      ``and[i,o] = min(ndvi_z, lst_z)``  — fires only when BOTH move
      ``or[i,o]  = max(ndvi_z, lst_z)``  — fires when EITHER moves
    Unobserved bins score NaN (never 0 — an unobserved bin is not a normal bin).
    """
    ci_ndvi = channel_names.index(channels[0])
    ci_lst = channel_names.index(channels[1])
    and_s = np.full(x.shape[:2], np.nan, dtype=np.float64)
    or_s = np.full(x.shape[:2], np.nan, dtype=np.float64)
    for i, fid in enumerate(field_ids):
        st = stats.get(int(fid))
        if st is None:
            continue        # field never calibrated: its scores stay NaN, recorded upstream
        xv = x[i, :, ci_ndvi].astype(np.float64)
        lv = x[i, :, ci_lst].astype(np.float64)
        mv = m[i, :, ci_ndvi] & m[i, :, ci_lst]
        ndvi_z = (st.ndvi_median - xv) / st.ndvi_mad
        lst_z = (lv - st.lst_median) / st.lst_mad
        both = np.where(mv, np.minimum(ndvi_z, lst_z), np.nan)
        either = np.where(mv, np.maximum(ndvi_z, lst_z), np.nan)
        and_s[i] = both
        or_s[i] = either
    return {"and": and_s, "or": or_s}


def calibrate_b1_rule(
    x: np.ndarray, m: np.ndarray, field_ids: np.ndarray, cycle_ids: np.ndarray,
    calibration_cycle_ids: np.ndarray,
    *,
    target_cycle_far: float = CYCLE_FAR_TARGET,
    channels: tuple[str, ...] = B1_CHANNELS,
    channel_names: tuple[str, ...] = V2_CHANNELS,
) -> dict[str, B1Calibration]:
    """FAR-match BOTH variants on the clean calibration subset (no eval tuning).

    The threshold for each variant is the ``(1 - target_cycle_far)`` quantile of
    that variant's cycle scores ON THE CALIBRATION SUBSET — the score a clean
    cycle exceeds with probability ``target_cycle_far``. The evaluation cycles
    are never read here; ``test_calibration_independence`` proves it.
    """
    stats, _ = field_robust_stats(
        x, m, field_ids, cycle_ids, calibration_cycle_ids,
        channels=channels, channel_names=channel_names,
    )
    scores = variant_scores(x, m, stats, field_ids, channels=channels,
                            channel_names=channel_names)
    cal_mask = np.isin(cycle_ids, np.asarray(calibration_cycle_ids))
    out: dict[str, B1Calibration] = {}
    for variant, s in scores.items():
        cal_scores = np.nanmax(s[cal_mask], axis=1)
        cal_scores = cal_scores[np.isfinite(cal_scores)]
        if cal_scores.size < 2:
            raise ValueError(
                f"variant {variant!r}: fewer than 2 usable calibration cycles; "
                "a quantile of an empty set is not a threshold"
            )
        threshold = float(np.quantile(cal_scores, 1.0 - target_cycle_far))
        achieved = float(np.mean(cal_scores > threshold))
        degenerate = [st for st in stats.values() if st.degenerate]
        out[variant] = B1Calibration(
            variant=variant,
            threshold=threshold,
            target_cycle_far=float(target_cycle_far),
            achieved_cycle_far=achieved,
            n_calibration_cycles=int(cal_scores.size),
            n_fields=len(stats),
            degenerate_fields=len(degenerate),
            scale_fallback=(
                "pooled-MAD scale for degenerate field channels (recorded per field)"
                if degenerate else "none"
            ),
            per_field=stats,
        )
    return out


def calibration_independence_audit(
    x: np.ndarray, m: np.ndarray, field_ids: np.ndarray, cycle_ids: np.ndarray,
    calibration_cycle_ids: np.ndarray,
    *,
    channels: tuple[str, ...] = B1_CHANNELS,
    channel_names: tuple[str, ...] = V2_CHANNELS,
) -> dict[str, bool]:
    """Task 1.1's calibration-independence check, executable.

    The threshold must be a function of the calibration subset ALONE: perturbing
    every non-calibration cycle must leave it unchanged. Runs the perturbation
    and reports the (expected True) result rather than asserting, so a caller
    can record the audit outcome in the manifest.
    """
    base = calibrate_b1_rule(x, m, field_ids, cycle_ids, calibration_cycle_ids,
                             channels=channels, channel_names=channel_names)
    rng = np.random.default_rng(2026)
    eval_mask = ~np.isin(cycle_ids, np.asarray(calibration_cycle_ids))
    x2 = x.copy()
    m2 = m.copy()
    if eval_mask.any():
        noise = rng.normal(0.0, 3.0, size=x2[eval_mask].shape)
        x2[eval_mask] = x2[eval_mask] + noise.astype(x2.dtype)
        m2[eval_mask] = ~m2[eval_mask]              # even flip observation masks
    pert = calibrate_b1_rule(x2, m2, field_ids, cycle_ids, calibration_cycle_ids,
                             channels=channels, channel_names=channel_names)
    return {
        v: bool(base[v].threshold == pert[v].threshold
                and base[v].achieved_cycle_far == pert[v].achieved_cycle_far)
        for v in base
    }


def estimate_cycle_far(
    scores: np.ndarray, threshold: float,
    *, labels_clean: np.ndarray | None = None,
) -> dict:
    """Cycle-FAR at ``threshold`` — on CLEAN cycles, or on all if none given.

    FAR (false-alarm rate) is the fraction of CLEAN cycles whose score exceeds
    the threshold. Passing unclean cycles without ``labels_clean`` would
    overstate the FAR, so the caller either supplies the clean mask or accepts a
    result labelled ``population: all-cycles`` (never 'clean'). Cycles with no
    finite score (every bin unobserved) are excluded from the denominator and
    counted separately — never silently.
    """
    s_all = np.nanmax(scores, axis=1) if scores.ndim == 2 else scores
    finite = np.isfinite(s_all)
    s = s_all[finite]
    n_unscorable = int((~finite).sum())
    if s.size == 0:
        return {"status": "absent", "reason": "no finite cycle scores",
                "n_unscorable": n_unscorable}
    if labels_clean is None:
        flagged = s > threshold
        return {
            "status": "measured", "population": "all-cycles",
            "n": int(s.size), "n_flagged": int(flagged.sum()),
            "cycle_far": float(flagged.mean()),
            "n_unscorable": n_unscorable,
            "note": "not restricted to clean cycles; do not quote as FAR",
        }
    clean = np.asarray(labels_clean, dtype=bool)[finite]
    sc = s[clean]
    flagged = sc > threshold
    return {
        "status": "measured", "population": "clean",
        "n": int(sc.size), "n_flagged": int(flagged.sum()),
        "cycle_far": float(flagged.mean()) if sc.size else float("nan"),
        "n_unscorable": n_unscorable,
    }


# ---------------------------------------------------------------------------
# Tasks 1.2 + 1.3 — B2–B5 input builders and the causality audit
# ---------------------------------------------------------------------------


@dataclass
class RungInputs:
    """A rung's assembled model inputs, with every block's provenance.

    Contract: ``x``/``m`` hold the rung's RAW selected channels (B2: ndvi+lst;
    B3: all 8; B4/B5 append weather / phenology / STL as extra channels with
    their own observation masks). The SSL candidates' token builders construct
    ``[values, mask, cos(dsh), sin(dsh)]`` from these, so one trunk consumes
    every rung unchanged and the capacity table (task 1.4) matches by
    construction: ``d_in = 2 * width + 2`` with ``width = x.shape[2]``.
    """

    rung: str
    status: str                                  # 'built' | 'absent'
    x: np.ndarray | None = None                  # (N, T, C) raw values
    m: np.ndarray | None = None                  # (N, T, C) observation mask
    dsh: np.ndarray | None = None                # (T,) days since cycle start
    feature_names: tuple[str, ...] = ()
    absent_blocks: dict[str, str] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "rung": self.rung,
            "status": self.status,
            "n_input_channels": (self.x.shape[2] if self.x is not None else 0),
            "feature_names": list(self.feature_names),
            "absent_blocks": dict(self.absent_blocks),
            "reason": self.reason,
        }

    def assert_usable(self) -> None:
        if self.status != "built" or self.x is None:
            raise ValueError(
                f"rung {self.rung} is {self.status}: {self.reason or self.absent_blocks}"
            )


def _channel_subset(
    x: np.ndarray, m: np.ndarray, channels: tuple[str, ...],
    channel_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    idx = [channel_names.index(c) for c in channels]
    return x[:, :, idx].copy(), m[:, :, idx].copy()


def _doy_features(dsh: np.ndarray) -> np.ndarray:
    """Cyclic DOY/DSH encoding at each bin — a function of the bin index only."""
    phase = (dsh / 540.0) * 2.0 * np.pi
    return np.stack([np.cos(phase), np.sin(phase)], axis=-1)


def _phase_bin(dsh: np.ndarray) -> np.ndarray:
    return np.clip((dsh / PHASE_BIN_DAYS).astype(np.int64), 0, N_PHASE_BINS - 1)


def build_rung_inputs(
    rung: str,
    x: np.ndarray,
    m: np.ndarray,
    dsh: np.ndarray,
    *,
    channel_names: tuple[str, ...] = V2_CHANNELS,
    weather: np.ndarray | None = None,
    decomposition: np.ndarray | None = None,
) -> RungInputs:
    """Assemble a rung's model inputs; missing declared blocks => ``absent``.

    ``weather``  — (N, T, 6) daily ERA5 aligned to the cycle bins, columns in
                   ``ERA5_CHANNELS`` order. Only same-bin values are used, so
                   the block is causal by construction (and tested, task 1.3).
    ``decomposition`` — (N, T, 3) STL trend/season/residual. Supplying this
                   asserts the producer produced it CAUSALLY (trailing-window);
                   a full-series STL would fail :func:`assert_causal_inputs`.
                   Until the decomposition-aware-ssl-study produces it, B5 is
                   reported ``absent`` naming the block — not approximated.
    """
    if rung not in RUNGS:
        raise ValueError(f"unknown rung {rung!r}; known: {sorted(RUNGS)}")
    spec = RUNGS[rung]
    missing: dict[str, str] = {}
    for block in spec.required_external_blocks():
        supplied = weather if block == "era5_daily" else decomposition
        if supplied is None:
            missing[block] = (
                "not supplied by any producer in this repository: "
                + (
                    "the daily ERA5-to-bin alignment table does not exist yet; "
                    "the rung is reported absent, never substituted"
                    if block == "era5_daily"
                    else "the decomposition-aware-ssl-study has not produced a "
                         "canonical polygon-zonal store, and a full-series STL "
                         "would violate the pre-registered causality rule "
                         "(no bin o may reference bins > o); the block is absent "
                         "until a causal producer exists"
                )
            )
    if missing:
        return RungInputs(
            rung=rung, status="absent",
            absent_blocks=missing,
            reason=f"declared block(s) missing: {', '.join(sorted(missing))}",
        )

    xs, ms = _channel_subset(x, m, spec.channels, channel_names)
    feats = [xs.astype(np.float64)]
    masks = [ms]
    names = list(spec.channels)
    if spec.weather:
        w = np.asarray(weather, dtype=np.float64)
        if w.shape != (xs.shape[0], xs.shape[1], len(ERA5_CHANNELS)):
            raise ValueError(
                f"weather must be (N={xs.shape[0]}, T={xs.shape[1]}, "
                f"{len(ERA5_CHANNELS)}) in ERA5_CHANNELS order, got {w.shape}"
            )
        feats.append(w)
        masks.append(np.isfinite(w))                    # a missing weather value is unobserved
        names += list(ERA5_CHANNELS)
    if spec.phenology:
        # phase bin is a function of the bin index (causal); sin/cos of the DSH
        # reach the model through the trunk's dsh token (uses_doy), as in B2-B4.
        pb = np.broadcast_to(_phase_bin(dsh)[None, :, None].astype(np.float64),
                             (xs.shape[0], xs.shape[1], 1))
        feats.append(pb)
        masks.append(np.ones(pb.shape[:2], dtype=bool))
        names.append("phase_bin")
    if spec.stl:
        dec = np.asarray(decomposition, dtype=np.float64)
        if dec.shape != (xs.shape[0], xs.shape[1], len(B5_STL_FEATURES)):
            raise ValueError(
                f"decomposition must be (N, T, {len(B5_STL_FEATURES)}) in "
                f"{B5_STL_FEATURES} order, got {dec.shape}"
            )
        feats.append(dec)
        masks.append(np.isfinite(dec))
        names += list(B5_STL_FEATURES)
    return RungInputs(
        rung=rung, status="built",
        x=np.concatenate(feats, axis=-1),
        m=np.concatenate(masks, axis=-1),
        dsh=np.asarray(dsh, dtype=np.float64), feature_names=tuple(names),
    )


def assert_causal_inputs(
    build: Callable[[np.ndarray], np.ndarray],
    x_source: np.ndarray,
    *,
    probe_bin: int | None = None,
    seed: int = 2026,
) -> dict:
    """Task 1.3: no feature at bin *o* may reference bins > *o*.

    ``build`` maps the SOURCE cycle table ``(N, T, C)`` to any features shaped
    ``(N, T, ...)``. Every bin strictly before ``probe_bin`` must be UNCHANGED
    when all bins >= ``probe_bin`` of the source are perturbed. Returns the
    audit record (``causal: True`` expected); violations name the first bin
    that moved.
    """
    xs = np.asarray(x_source)
    n, t = xs.shape[0], xs.shape[1]
    o = probe_bin if probe_bin is not None else t // 2
    base = np.asarray(build(xs), dtype=np.float64)
    rng = np.random.default_rng(seed)
    xs2 = xs.copy()
    xs2[:, o:, ...] = rng.normal(0.0, 10.0, size=xs2[:, o:, ...].shape).astype(xs2.dtype)
    pert = np.asarray(build(xs2), dtype=np.float64)
    if base.shape != pert.shape:
        return {"causal": False, "reason": f"shape changed {base.shape} -> {pert.shape}"}
    prefix_base = base[:, :o]
    prefix_pert = pert[:, :o]
    diff = np.nanmax(np.abs(prefix_base - prefix_pert)) if prefix_base.size else 0.0
    finite_ok = bool(np.all(np.isfinite(np.nan_to_num(prefix_base, nan=0.0))))
    return {
        "causal": bool(diff == 0.0 and finite_ok),
        "probe_bin": int(o),
        "max_prefix_shift": float(diff),
        "note": ("all bins < probe_bin unchanged under perturbation of bins >= "
                 "probe_bin" if diff == 0.0 else
                 f"bins < {o} changed when future bins were perturbed — the "
                 f"builder leaks the future (max shift {diff:.3g})"),
    }


#: Why the full-series STL variant is REJECTED, recorded so the absence of B5's
#: STL block reads as a decision, not an oversight.
STL_CAUSALITY_NOTE = (
    "an STL decomposition fitted on the FULL cycle references bins > o at every "
    "o; assert_causal_inputs fails it by construction. B5's STL block is "
    "therefore absent until a trailing-window (causal) decomposition producer "
    "exists — the pre-registered causality rule wins over input completeness."
)


# ---------------------------------------------------------------------------
# Task 1.4 — capacity accounting per rung
# ---------------------------------------------------------------------------


def capacity_manifest(
    *,
    d_model: int = 160,
    d_out: int = 64,
    n_layers: int = 1,
    target_params: int = 100_000,
    band: tuple[float, float] = (0.80, 1.20),
    max_params: int = 250_000,
    horizon: int = 3,
) -> dict:
    """Parameter counts per SSL rung (B2–B5 share the trunk; B1 has none).

    The band applies to the ENCODER (the representation every task consumes);
    the hard cap applies to encoder + reconstruction head (the trainable total).
    Same accounting as ``candidates.capacity_report`` — the counts here differ
    only because each rung's input width differs.
    """
    from ssl_anomaly.evaluation.candidates import CycleEncoder, IncumbentAE
    from ssl_anomaly.evaluation.metrics import count_parameters

    widths: dict[str, int] = {}
    for name in ("B2", "B3", "B4", "B5"):
        spec = RUNGS[name]
        # Raw channel width (values), matching build_rung_inputs' x.shape[2];
        # the trunk's token width is 2 * width + 2 (values + masks + cos/sin dsh).
        width = len(spec.channels)
        if spec.weather:
            width += len(ERA5_CHANNELS)
        if spec.phenology:
            width += 1
        if spec.stl:
            width += len(B5_STL_FEATURES)
        widths[name] = width

    rows: dict[str, dict] = {}
    for name in ("B1", "B2", "B3", "B4", "B5"):
        spec = RUNGS[name]
        if not spec.learning:
            rows[name] = {
                "model": spec.model, "learning": False,
                "trainable_params": 0,
                "note": "a threshold rule has no parameters to band; the "
                        "threshold is a calibration statistic, not a weight",
            }
            continue
        enc = CycleEncoder(widths[name], d_model=d_model, d_out=d_out, n_layers=n_layers)
        full = IncumbentAE(widths[name], d_model=d_model, d_out=d_out)
        enc_p, tot_p = count_parameters(enc), count_parameters(full)
        lo, hi = band[0] * target_params, band[1] * target_params
        band_ok = lo <= enc_p <= hi
        cap_ok = tot_p <= max_params
        rows[name] = {
            "model": spec.model, "learning": True,
            "n_input_channels": widths[name],
            "encoder_params": int(enc_p),
            "total_params": int(tot_p),
            "band": [lo, hi],
            "band_ok": bool(band_ok),
            "hard_cap_ok": bool(cap_ok),
            "detail": "" if (band_ok and cap_ok) else (
                f"encoder {enc_p} outside [{lo:.0f}, {hi:.0f}]"
                if not band_ok else f"total {tot_p} exceeds hard cap {max_params}"
            ),
        }
    return {
        "config": {"d_model": d_model, "d_out": d_out, "n_layers": n_layers,
                   "target_params": target_params, "band": list(band),
                   "max_params": max_params, "horizon": horizon},
        "rungs": rows,
        "band_enforced": all(r.get("band_ok", True) and r.get("hard_cap_ok", True)
                             for r in rows.values()),
    }


# ---------------------------------------------------------------------------
# Task 1.5 — the common scoring bridge
# ---------------------------------------------------------------------------


class RungScorer(Protocol):
    """The one scoring interface every rung must expose (task 1.5).

    ``score(field_id, bin) -> float`` — the rung's continuous anomaly score for
    one bin of one field's cycle; higher = more anomalous. ``score_cycle``
    returns the per-bin vector. Identical evaluation consumes both.
    """

    def score(self, field_id: int, bin_index: int) -> float: ...

    def score_cycle(self, field_id: int) -> np.ndarray: ...

    @property
    def rung(self) -> str: ...


class B1RuleScorer:
    """The calibrated B1 rule behind the bridge (either variant, both recorded).

    ``score(field, bin)`` recomputes that bin's robust-z from the calibration
    statistics — the threshold is applied by the evaluator, never baked into the
    score, so FAR-matching at a different budget never requires re-scoring.
    Unobserved bins raise: an unobserved bin has no rule score (never 0).
    """

    def __init__(
        self,
        calibration: B1Calibration,
        *,
        x: np.ndarray, m: np.ndarray, field_ids: np.ndarray,
        channels: tuple[str, ...] = B1_CHANNELS,
        channel_names: tuple[str, ...] = V2_CHANNELS,
    ):
        if calibration.variant not in ("and", "or"):
            raise ValueError(f"unknown variant {calibration.variant!r}")
        self._cal = calibration
        self._x, self._m, self._field_ids = x, m, np.asarray(field_ids)
        self._channels = channels
        self._channel_names = channel_names
        # Precompute per-bin scores once with THIS calibration's statistics.
        self._per_bin = variant_scores(
            x, m, calibration.per_field, self._field_ids,
            channels=channels, channel_names=channel_names,
        )[calibration.variant]

    @property
    def rung(self) -> str:
        return "B1"

    @property
    def variant(self) -> str:
        return self._cal.variant

    @property
    def threshold(self) -> float:
        return self._cal.threshold

    def cycle_rows(self, field_id: int) -> np.ndarray:
        return np.flatnonzero(self._field_ids == int(field_id))

    def score_cycle(self, field_id: int, cycle_index: int | None = None) -> np.ndarray:
        rows = self.cycle_rows(field_id)
        if rows.size == 0:
            raise ValueError(f"field {field_id} is absent from the scored table")
        if cycle_index is None:
            row = int(rows[0])
        else:
            row = int(cycle_index)
            if row not in rows:
                raise ValueError(
                    f"row {row} does not belong to field {field_id} in the "
                    "scored table; refusing to score a mismatched pair"
                )
        return self._per_bin[row]

    def score(self, field_id: int, bin_index: int, cycle_index: int | None = None) -> float:
        s = self.score_cycle(field_id, cycle_index)
        v = float(s[bin_index])
        if not np.isfinite(v):
            raise ValueError(
                f"bin {bin_index} of field {field_id} is unobserved for the B1 "
                "channels; an unobserved bin has no rule score (never 0)"
            )
        return v


class SSLRungScorer:
    """Per-bin causal scoring for an SSL rung (B2–B5) through the same bridge.

    The per-bin state at bin *o* is the trunk hidden state after consuming bins
    ≤ *o* only (the GRU is causal by construction), scored by the rung's
    masked-reconstruction error at that bin against the rung's own inputs. The
    full-cycle embedding is deliberately NOT used here: it sees the whole cycle
    and would smuggle the future into every per-bin score.
    """

    def __init__(self, model, rung_inputs: RungInputs, *,
                 field_ids: np.ndarray | None = None, device: str = "cpu"):
        if rung_inputs.status != "built":
            raise ValueError(f"rung {rung_inputs.rung} is {rung_inputs.status}; "
                             "there is nothing to score")
        import torch

        self._torch = torch
        self.model = model.to(device).eval()
        self.inputs = rung_inputs
        self.device = device
        self._rung = rung_inputs.rung
        # The bridge is keyed by (field, bin); resolving a field to cycle rows
        # requires the table's field ids — refusing to guess them.
        if field_ids is None:
            raise ValueError(
                "SSLRungScorer needs the cycle table's field_ids to resolve "
                "score(field, bin); refusing to guess the field mapping"
            )
        self.field_ids = np.asarray(field_ids)

    @property
    def rung(self) -> str:
        return self._rung

    def cycle_rows(self, field_id: int) -> np.ndarray:
        return np.flatnonzero(self.field_ids == int(field_id))

    def score_cycle(self, field_id: int, cycle_index: int | None = None,
                    x: np.ndarray | None = None, m: np.ndarray | None = None,
                    dsh: np.ndarray | None = None) -> np.ndarray:
        """Per-bin reconstruction error for one cycle of one field.

        One of ``cycle_index`` (row into ``self.inputs``) or explicit
        ``x``/``m``/``dsh`` (the rung's RAW shapes, as ``build_rung_inputs``
        produced them) must be given — scoring data the caller did not pass
        through the builder would silently change the rung's meaning.
        """
        torch = self._torch
        if x is None:
            rows = self.cycle_rows(field_id)
            if rows.size == 0:
                raise ValueError(f"field {field_id} is absent from the scored table")
            if cycle_index is None:
                cycle_index = int(rows[0])
            elif int(cycle_index) not in rows:
                raise ValueError(
                    f"row {cycle_index} does not belong to field {field_id} in "
                    "the scored table; refusing to score a mismatched pair"
                )
            x = self.inputs.x[int(cycle_index) : int(cycle_index) + 1]
            m = self.inputs.m[int(cycle_index) : int(cycle_index) + 1]
            dsh = self.inputs.dsh
        with torch.no_grad():
            xt = torch.from_numpy(np.asarray(x, dtype=np.float32)).to(self.device)
            mt = torch.from_numpy(np.asarray(m, dtype=bool)).to(self.device)
            dsht = torch.from_numpy(
                np.broadcast_to(np.asarray(dsh, dtype=np.float32),
                                (xt.shape[0], xt.shape[1])).copy()
            ).to(self.device)
            enc = self.model.encoder if hasattr(self.model, "encoder") else self.model
            # Causal per-bin states: run the trunk and keep the state at each bin.
            mm = mt.to(xt.dtype)
            xx = torch.where(mt, torch.nan_to_num(xt, nan=0.0), torch.zeros_like(xt))
            phase = dsht / 540.0
            tok = torch.cat([xx, mm,
                             torch.cos(phase).unsqueeze(-1),
                             torch.sin(phase).unsqueeze(-1)], dim=-1)
            # The trunk input width matches the RUNG's d_in only when the model
            # was built for this rung; a mismatch must fail loudly.
            d_in = getattr(enc, "d_in", tok.shape[-1])
            if d_in != tok.shape[-1]:
                raise ValueError(
                    f"model expects d_in={d_in} but rung {self._rung} builds "
                    f"width {tok.shape[-1]}; rebuild the model for this rung"
                )
            seq, _ = enc.gru(tok)
            head = getattr(self.model, "head", None)
            if head is None:
                raise ValueError(
                    "the rung scorer scores per-bin masked reconstruction error; "
                    "a model without a per-bin reconstruction head cannot feed "
                    "this bridge (supply the incumbent-pattern model)"
                )
            recon = head(seq)
            diff = (recon - xx).abs()
            err = (diff * mm).sum(dim=-1) / mm.sum(dim=-1).clamp(min=1.0)
        return err[0].cpu().numpy()

    def score(self, field_id: int, bin_index: int, **kwargs) -> float:
        s = self.score_cycle(field_id=field_id, **kwargs)
        v = float(s[bin_index])
        if not np.isfinite(v):
            raise ValueError(
                f"bin {bin_index} of field {field_id} has no finite score "
                "(no observed channel at that bin); never a fabricated 0"
            )
        return v


# ---------------------------------------------------------------------------
# Combined record for the manifest
# ---------------------------------------------------------------------------


def input_ladder_status(
    builds: dict[str, RungInputs], capacity: dict, causality: dict[str, dict],
    calibration: dict[str, B1Calibration] | None = None,
) -> dict:
    """One compact block: every rung, never omitted; absences with reasons."""
    rows = []
    for name in ("B1", "B2", "B3", "B4", "B5"):
        spec = RUNGS[name]
        row = {"name": name, "model": spec.model, "learning": spec.learning}
        if name == "B1":
            row["status"] = "calibrated" if calibration else "not_attempted"
            if calibration:
                row["variants"] = {v: c.to_dict() for v, c in calibration.items()}
        else:
            b = builds.get(name)
            row.update(b.to_dict() if b is not None
                       else {"status": "not_attempted", "reason": ""})
        row["causality"] = causality.get(name, {"audited": False})
        row["capacity"] = capacity["rungs"].get(name)
        rows.append(row)
    return {
        "rungs": rows,
        "reading_rule": ladder_reading_rule(),
        "band_enforced": capacity["band_enforced"],
        "stl_causality_note": STL_CAUSALITY_NOTE,
    }


def dumps(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)

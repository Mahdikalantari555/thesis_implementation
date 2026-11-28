"""Normalization arms N0–N4 with leak-free, anomaly-excluded statistics (0A.5).

design.md §1A.3 constraints [DEC]:
* statistics from the **training split only** (time ≤ train years and training
  farms only), never from validation/test years or held-out farms;
* computed on **QA-valid observations only**, excluding anything in the
  anomaly memory's verified/unresolved partitions (``exclude_mask``);
* missing/invalid values stay missing — the observation mask is carried and
  masked entries never enter the statistics;
* normalization is invertible and versioned (§7.5 sidecar).

Arms (E7 compares them; none is presumed best):
| N0 | global z-score per feature                  |
| N1 | robust median / (1.4826·MAD), IQR fallback  |
| N2 | physical fixed scale (identity; Kelvin centred by fixed constants) |
| N3 | N1 then clip to ±c, clip rate logged        |
| N4 | per-field centring (train median of that field), global robust scale |

The N2 constants and the N3 clip width are **pre-registered by this change**
(owner-revisable); they are recorded inside every saved statistics sidecar so
a reader never has to guess them.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ARMS: tuple[str, ...] = ("N0", "N1", "N2", "N3", "N4")
STATS_SCHEMA = "norm-stats/v1"

#: Pre-registered N2 physical constants (documented in design §1A.3 as "fixed
#: physical constants"; exact values pinned here and stored in every sidecar).
N2_CONSTANTS: dict[str, float] = {
    "kelvin_center": 273.15,  # 0 °C
    "kelvin_scale": 50.0,     # K per normalized unit
    "reflectance_k": 1.0,     # reflectance kept in ρ units
}
#: N3 clip width in robust-sigma units (pre-registered).
N3_CLIP_SIGMA: float = 3.0
MAD_TO_SIGMA: float = 1.4826


class NormalizationLeakageError(RuntimeError):
    """Statistics were requested over data outside the declared train scope."""


class NormalizationValueError(ValueError):
    """Malformed statistics request (bad arm, missing mask, ...)."""


@dataclass
class NormStats:
    arm: str
    feature_names: list[str]
    center: np.ndarray            # (F,) per feature
    scale: np.ndarray             # (F,) per feature
    field_centers: dict[str, np.ndarray] | None = None  # N4: farm → (F,)
    clip_sigma: float | None = None
    constants: dict[str, float] = field(default_factory=dict)
    n_obs: int = 0
    n_excluded_anomaly: int = 0
    n_leakage_checked: int = 0
    date_min: str = ""
    date_max: str = ""
    train_end: str = ""
    data_sha256: str = ""
    created_utc: str = ""
    git_sha: str = ""
    clip_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": STATS_SCHEMA,
            "arm": self.arm,
            "feature_names": list(self.feature_names),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "field_centers": (
                {k: v.tolist() for k, v in self.field_centers.items()}
                if self.field_centers is not None else None
            ),
            "clip_sigma": self.clip_sigma,
            "constants": dict(self.constants),
            "n_obs": self.n_obs,
            "n_excluded_anomaly": self.n_excluded_anomaly,
            "n_leakage_checked": self.n_leakage_checked,
            "date_min": self.date_min,
            "date_max": self.date_max,
            "train_end": self.train_end,
            "data_sha256": self.data_sha256,
            "created_utc": self.created_utc,
            "git_sha": self.git_sha,
            "clip_rate": self.clip_rate,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "NormStats":
        if d.get("schema") != STATS_SCHEMA:
            raise NormalizationValueError(
                f"statistics sidecar schema {d.get('schema')!r} != {STATS_SCHEMA!r}"
            )
        fc = d.get("field_centers")
        return cls(
            arm=str(d["arm"]),
            feature_names=list(d["feature_names"]),
            center=np.asarray(d["center"], dtype=np.float64),
            scale=np.asarray(d["scale"], dtype=np.float64),
            field_centers=(
                {k: np.asarray(v, dtype=np.float64) for k, v in fc.items()}
                if fc is not None else None
            ),
            clip_sigma=d.get("clip_sigma"),
            constants=dict(d.get("constants") or {}),
            n_obs=int(d["n_obs"]),
            n_excluded_anomaly=int(d.get("n_excluded_anomaly", 0)),
            n_leakage_checked=int(d.get("n_leakage_checked", 0)),
            date_min=str(d.get("date_min", "")),
            date_max=str(d.get("date_max", "")),
            train_end=str(d.get("train_end", "")),
            data_sha256=str(d.get("data_sha256", "")),
            created_utc=str(d.get("created_utc", "")),
            git_sha=str(d.get("git_sha", "")),
            clip_rate=d.get("clip_rate"),
        )


def _hash_inscope(x: np.ndarray, observed: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x).tobytes())
    h.update(np.ascontiguousarray(observed.astype(np.uint8)).tobytes())
    return h.hexdigest()


def _check_dates(dates: np.ndarray, observed: np.ndarray, train_end: str) -> tuple[int, str, str]:
    """Fail closed when any observed in-scope observation is ≥ train_end."""
    d = np.asarray(dates)
    n_checked = 0
    min_d, max_d = "", ""
    for i in range(d.shape[0]):
        for t in range(d.shape[1]):
            if not observed[i, t].any():
                continue  # nothing observed at this position → not in scope
            s = str(d[i, t])
            if not s:
                continue  # padded position (observed is False there anyway)
            if s >= train_end:
                raise NormalizationLeakageError(
                    f"normalization leakage: observation at sample {i}, t={t} "
                    f"is dated {s} (>= train_end {train_end}); statistics may "
                    "only be computed on training-year observations "
                    "(design.md §1A.3)"
                )
            n_checked += 1
            min_d = min(min_d, s) if min_d else s
            max_d = max(max_d, s)
    if n_checked == 0:
        raise NormalizationValueError("no in-scope observations for normalization")
    return n_checked, min_d, max_d


def _robust_center_scale(v: np.ndarray) -> tuple[float, float]:
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    sigma = MAD_TO_SIGMA * mad
    if sigma <= 1e-12:
        # IQR fallback (design §1A.3 N1); recorded via the returned values
        q75, q25 = np.percentile(v, [75, 25])
        sigma = float(q75 - q25) / 1.349
    if sigma <= 1e-12:
        sigma = 1.0
    return med, sigma


def fit_normalization(
    x: np.ndarray,
    observed: np.ndarray,
    dates: np.ndarray,
    *,
    arm: str,
    feature_names: list[str],
    train_end: str = "2023-01-01",
    exclude_mask: np.ndarray | None = None,
    farm_ids: list[str] | None = None,
    git_sha: str = "",
) -> NormStats:
    """Fit per-feature normalization statistics on training-year data only.

    Parameters
    ----------
    x: (N, T, F) physical-unit feature values (NaN allowed where absent).
    observed: (N, T, F) bool QA-valid observation mask.
    dates: (N, T) ISO date strings ("" for padded positions).
    arm: one of :data:`ARMS`.
    train_end: exclusive upper date bound; any observed date ≥ it raises
        :class:`NormalizationLeakageError` (spec scenario: a 2024+ observation
        in scope fails).
    exclude_mask: (N, T) bool, True = observation belongs to the anomaly
        memory (verified or unresolved) and must NOT inform statistics.
    farm_ids: per-sample farm ids (required for N4).
    """
    if arm not in ARMS:
        raise NormalizationValueError(f"unknown normalization arm {arm!r}; known: {ARMS}")
    x = np.asarray(x, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if x.shape != observed.shape:
        raise NormalizationValueError(
            f"x shape {x.shape} != observed shape {observed.shape}"
        )
    if dates.shape[:2] != x.shape[:2]:
        raise NormalizationValueError(
            f"dates shape {np.asarray(dates).shape} incompatible with x {x.shape}"
        )
    in_scope = observed.copy()
    n_excluded = 0
    if exclude_mask is not None:
        ex = np.asarray(exclude_mask, dtype=bool)
        if ex.shape != x.shape[:2]:
            raise NormalizationValueError(
                f"exclude_mask shape {ex.shape} incompatible with x {x.shape[:2]}"
            )
        n_excluded = int(ex.sum())
        # excluded observations contribute to nothing: drop them entirely
        in_scope &= ~ex[:, :, None]

    n_checked, min_d, max_d = _check_dates(np.asarray(dates), observed, str(train_end))

    flat_x = x.reshape(-1, x.shape[2])
    flat_scope = in_scope.reshape(-1, x.shape[2])
    f = x.shape[2]
    if len(feature_names) != f:
        raise NormalizationValueError(
            f"feature_names has {len(feature_names)} entries for {f} columns"
        )
    center = np.zeros(f)
    scale = np.ones(f)
    for j in range(f):
        v = flat_x[flat_scope[:, j], j]
        v = v[np.isfinite(v)]
        if v.size == 0:
            # feature entirely absent in scope: stays identity, flagged
            center[j], scale[j] = 0.0, 1.0
            continue
        if arm == "N0":
            center[j] = float(np.mean(v))
            scale[j] = float(np.std(v)) or 1.0
        elif arm in ("N1", "N3"):
            center[j], scale[j] = _robust_center_scale(v)
        elif arm == "N2":
            # physical fixed scale: identity except Kelvin-centred features.
            # A feature is Kelvin when its name is LST-like; the constants are
            # pre-registered in N2_CONSTANTS and stored in the sidecar.
            if any(k in feature_names[j].lower() for k in ("lst", "temp")):
                center[j] = N2_CONSTANTS["kelvin_center"]
                scale[j] = N2_CONSTANTS["kelvin_scale"]
            else:
                center[j] = 0.0
                scale[j] = N2_CONSTANTS["reflectance_k"] or 1.0
        elif arm == "N4":
            center[j] = 0.0  # replaced per-field below; global scale = robust
            scale[j] = _robust_center_scale(v)[1]

    field_centers: dict[str, np.ndarray] | None = None
    clip_rate: float | None = None
    if arm == "N4":
        if farm_ids is None:
            raise NormalizationValueError("arm N4 requires farm_ids")
        uniq = sorted(set(map(str, farm_ids)))
        field_centers = {}
        for farm in uniq:
            idx = np.array([i for i, s in enumerate(map(str, farm_ids)) if s == farm])
            xi = x[idx].reshape(-1, f)
            si = in_scope[idx].reshape(-1, f)
            fc = np.zeros(f)
            for j in range(f):
                v = xi[si[:, j], j]
                v = v[np.isfinite(v)]
                fc[j] = float(np.median(v)) if v.size else 0.0
            field_centers[farm] = fc

    if arm == "N3":
        clip_sigma = float(N3_CLIP_SIGMA)
        z = (flat_x[flat_scope.any(axis=1)] - center) / scale
        inside = z[np.isfinite(z)]
        clip_rate = float((np.abs(inside) > clip_sigma).mean())

    return NormStats(
        arm=arm,
        feature_names=list(feature_names),
        center=center,
        scale=scale,
        field_centers=field_centers,
        clip_sigma=float(N3_CLIP_SIGMA) if arm == "N3" else None,
        constants=dict(N2_CONSTANTS) if arm == "N2" else {},
        n_obs=int(flat_scope.sum()),
        n_excluded_anomaly=n_excluded,
        n_leakage_checked=n_checked,
        date_min=min_d,
        date_max=max_d,
        train_end=str(train_end),
        data_sha256=_hash_inscope(x, observed),
        created_utc=datetime.now(timezone.utc).isoformat(),
        git_sha=git_sha,
        clip_rate=clip_rate,
    )


def apply_normalization(
    stats: NormStats,
    x: np.ndarray,
    observed: np.ndarray,
    farm_ids: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply fitted statistics; masked entries stay 0-with-mask (never filled).

    Returns ``(x_norm, record)`` where the record carries the applied clip
    fraction for N3 (arm behaviour must be visible, not silent).
    """
    x = np.asarray(x, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if x.shape != observed.shape:
        raise NormalizationValueError(f"x shape {x.shape} != observed shape {observed.shape}")
    if len(stats.feature_names) != x.shape[2]:
        raise NormalizationValueError(
            f"statistics carry {len(stats.feature_names)} features, data has {x.shape[2]}"
        )
    record: dict[str, Any] = {"arm": stats.arm}
    out = np.zeros_like(x)
    for j, name in enumerate(stats.feature_names):
        center = stats.center[j]
        scale = stats.scale[j] if stats.scale[j] > 0 else 1.0
        vals = x[:, :, j]
        if stats.arm == "N4" and stats.field_centers is not None and farm_ids is not None:
            centers = np.array(
                [stats.field_centers.get(str(f), stats.center)[j] for f in farm_ids]
            )[:, None]
            missing = [f for f in map(str, farm_ids) if f not in stats.field_centers]
            if missing:
                record_fallback = f"field center missing for {sorted(set(missing))} — fell back to global center"
                record.setdefault("fallbacks", record_fallback)
        else:
            centers = center
        norm = (vals - centers) / scale
        norm[~observed[:, :, j]] = 0.0  # masked entries: 0 + mask, never a value
        out[:, :, j] = norm
    record["n_masked"] = int((~observed).sum())
    if stats.arm == "N3":
        z = out[observed]
        finite = z[np.isfinite(z)]
        clip_frac = float((np.abs(finite) > stats.clip_sigma).mean())
        out = np.clip(out, -stats.clip_sigma, stats.clip_sigma)
        out[~observed] = 0.0
        record["clip_frac_applied"] = clip_frac
        record["clip_sigma"] = stats.clip_sigma
    return out, record


def save_norm_stats(path: str | Path, stats: NormStats) -> Path:
    """Versioned sidecar (design §7.5): arm, values, data hash, git sha, dates."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(stats.to_dict(), indent=2) + "\n", encoding="utf-8")
    return p


def load_norm_stats(path: str | Path) -> NormStats:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return NormStats.from_dict(d)


def normalization_erasure_record(
    stats: NormStats,
    x_clean: np.ndarray,
    x_deviated: np.ndarray,
    *,
    feature_idx: int,
    observed: np.ndarray,
    farm_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Normalization-erasure test record (design §1A.3 / §8).

    A physical magnitude deviation (e.g. an NDVI drop) must remain detectable
    AFTER normalization: the post−pre difference in normalized units must
    retain the sign and at least ``min_retained_frac`` of the raw physical
    difference. Returns the retained fraction per arm; the caller decides
    pass/fail against the pre-registered threshold.
    """
    nc, _ = apply_normalization(stats, x_clean, observed, farm_ids)
    nd, _ = apply_normalization(stats, x_deviated, observed, farm_ids)
    j = feature_idx
    raw_drop = float(
        np.nanmean(x_deviated[:, :, j][observed[:, :, j]])
        - np.nanmean(x_clean[:, :, j][observed[:, :, j]])
    )
    norm_drop = float(
        np.nanmean(nd[:, :, j][observed[:, :, j]]) - np.nanmean(nc[:, :, j][observed[:, :, j]])
    )
    retained = abs(norm_drop) / abs(raw_drop) if abs(raw_drop) > 0 else None
    return {
        "arm": stats.arm,
        "feature": stats.feature_names[j],
        "raw_drop": raw_drop,
        "normalized_drop": norm_drop,
        "retained_frac": retained,
        "sign_preserved": bool(np.sign(norm_drop) == np.sign(raw_drop)) if raw_drop else None,
    }

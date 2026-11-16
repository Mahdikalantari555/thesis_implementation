"""Dataset + covariate builder over ``patches.zarr`` (task 1.3).

A training sample is one **(field, cycle)** pair: the whole 36-bin patch tensor
of that field-cycle. Three invariants are structural, not conventions:

1. **Holdout isolation** (``training-holdout-invariant``): the store contains
   every field, including the frozen holdout farms, because extraction is
   global and one-shot. The Dataset therefore filters with the shared
   ``monitor/holdout.py`` helper and *fails closed* — a missing or unreadable
   manifest is an error, never "train on everything". The holdout filter is
   applied to the field list at construction time and asserted again per
   sample, so a store edited after construction cannot leak.
2. **Normalization is frozen** (``patch-data-pipeline`` task 2.2): the
   per-channel × per-sensor median/IQR artifact is loaded read-only and its
   hash is checked; nothing is fitted here. ``build_patch_norm`` does the
   fitting, on train fields × train cycles only.
3. **NaN is absence** (patch store contract): unobserved pixel-bins are
   zero-filled for the model and carried by the mask. The loss mask is the
   per-channel observation mask ``obs`` — NOT the in-field mask — because an
   in-field pixel with no acquisition at that bin has no target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ssl_anomaly.io import patches as P
from ssl_anomaly.monitor import holdout as H

# 8 covariates, fixed order (task 1.3): sin/cos DOY, cycle age, days since last
# valid, log1p n_obs, sensor one-hot L8/L9, field-level in-field fraction.
COVARIATES: tuple[str, ...] = (
    "doy_sin", "doy_cos", "cycle_age", "days_since_valid",
    "log1p_n_obs", "sensor_l8", "sensor_l9", "infield_frac",
)
N_COVARIATES = len(COVARIATES)


@dataclass(frozen=True)
class PatchSample:
    """One field-cycle in model-ready form."""

    field_id: int
    cycle_id: int
    x: np.ndarray        # (T, C, S, S) float32, zero where unobserved
    obs: np.ndarray      # (T, C, S, S) bool, per-channel observation
    pixel_valid: np.ndarray  # (T, S, S) bool, in-field ∧ any channel observed
    covariates: np.ndarray   # (T, COVARIATES) float32
    start_day: int = 0


def build_covariates(
    cycle_start_day: np.ndarray,   # (T,) days since the cycle start (bin grid)
    n_obs: np.ndarray,             # (T,) acquisitions with ≥1 finite in-field pixel
    sensor_id: np.ndarray,         # (T,) 0/1/2
    in_field_frac: float,          # fraction of the patch inside the field
    *,
    bin_days: int = P.BIN_DAYS,
    n_bins: int = P.N_BINS,
) -> np.ndarray:
    """The 8-covariate matrix, ``(T, N_COVARIATES)`` float32.

    ``doy_sin``/``doy_cos`` use the same absolute-date convention as
    ``monitor.inputs.doy_channels`` (2013-03-30 epoch), so patch and zonal
    arms agree on what a season is. ``cycle_age`` is the bin index (a cycle
    always starts at the field's harvest date, so age = bins since start).
    ``days_since_valid`` is the causal gap: the number of bins since the last
    bin with ≥1 valid pixel (0 for the current bin when it is observed;
    carried forward otherwise, capped at ``n_bins``).
    """
    start = np.asarray(cycle_start_day, dtype=np.int64).ravel()
    if start.size == 1:
        start = np.full(n_bins, int(start[0]), dtype=np.int64)
    if start.size != n_bins:
        raise ValueError(f"cycle_start_day must be scalar or ({n_bins},), got {start.shape}")
    no = np.asarray(n_obs, dtype=np.float64).ravel()
    sid = np.asarray(sensor_id, dtype=np.int64).ravel()
    if no.size != n_bins or sid.size != n_bins:
        raise ValueError(f"n_obs/sensor_id must be ({n_bins},)")

    # Absolute DOY at each bin midpoint, same epoch as monitor.inputs.
    mid = start + np.arange(n_bins) * bin_days + bin_days / 2.0
    # Same convention as monitor.inputs.doy_channels: the absolute calendar
    # day from the shared epoch, so both arms agree on where the season is.
    epoch = pd.Timestamp("2013-03-30")
    doy = np.array(
        [(epoch + pd.Timedelta(days=int(d))).day_of_year for d in mid],
        dtype=np.float64,
    )
    ang = 2.0 * np.pi * doy / 365.25

    age = np.arange(n_bins, dtype=np.float64)
    since = np.full(n_bins, float(n_bins))
    last = -1
    for b in range(n_bins):
        if no[b] > 0:
            last = b
        since[b] = float(b - last) if last >= 0 else float(n_bins)

    out = np.zeros((n_bins, N_COVARIATES), dtype=np.float32)
    out[:, 0] = np.sin(ang)
    out[:, 1] = np.cos(ang)
    out[:, 2] = age / max(1, n_bins - 1)
    out[:, 3] = since / max(1, n_bins)
    out[:, 4] = np.log1p(np.maximum(no, 0.0)) / np.log1p(max(1.0, float(no.max(initial=0.0))))
    out[:, 5] = (sid == P.SENSOR_IDS["L8"]).astype(np.float32)
    out[:, 6] = (sid == P.SENSOR_IDS["L9"]).astype(np.float32)
    out[:, 7] = float(in_field_frac)
    return out


class PatchCycleDataset:
    """Iterable/lazy view over the store's (field, cycle) pairs.

    Parameters
    ----------
    root
        An open ``patches.zarr`` group (``patches.open_store``).
    field_ids
        Field ids to consider, in store order or any order (intersected with
        the store's own index).
    holdout_manifest
        The frozen manifest (dict). Required: ``None`` raises, because
        "train on everything" must never be reachable by omission.
    norm
        The frozen normalization dict (``patches.load_patch_norm``).
    max_cycles
        Optional cap for smoke runs (earliest cycles per field).
    """

    def __init__(
        self,
        root,
        field_ids: np.ndarray | list[int],
        *,
        holdout_manifest: dict,
        norm: dict,
        max_cycles: int | None = None,
        require_holdout_clear: bool = True,
    ) -> None:
        if holdout_manifest is None:
            raise ValueError(
                "holdout_manifest is required (training-holdout-invariant): pass the "
                "frozen manifest; training without it is not a supported path."
            )
        self.root = root
        self.norm = norm

        store_fields = np.asarray(root["field_ids"][:], dtype=np.int64)
        index = {int(f): i for i, f in enumerate(store_fields)}
        wanted = [int(f) for f in np.asarray(field_ids).ravel()]
        missing = [f for f in wanted if f not in index]
        if missing:
            raise ValueError(f"{len(missing)} requested fields are not in the store: {missing[:5]}")

        # --- fail-closed holdout filter -----------------------------------
        keep_rows = H.train_only_mask(wanted, manifest=holdout_manifest)
        self.excluded_by_holdout = [f for f, k in zip(wanted, keep_rows) if not k]
        self.fields = [f for f, k in zip(wanted, keep_rows) if k]
        if not self.fields:
            raise ValueError(
                "every requested field is a holdout field or holdout-only farm "
                f"({len(self.excluded_by_holdout)} excluded) — nothing to train on"
            )
        self._row_of = {f: index[f] for f in self.fields}
        self._holdout = holdout_manifest

        # --- cycle index ---------------------------------------------------
        self._pairs: list[tuple[int, int]] = []
        cycle_valid = root["cycle_valid"]
        for fid in self.fields:
            row = self._row_of[fid]
            valid = np.asarray(cycle_valid[row, :], dtype=bool)
            cols = np.flatnonzero(valid)
            if max_cycles is not None:
                cols = cols[: int(max_cycles)]
            self._pairs.extend((fid, int(c)) for c in cols)

        self.in_field = root["in_field"]
        self.channel_names = tuple(root.attrs.get("channel_order", P.CHANNELS))
        self.patch_size = int(root.attrs.get("patch_size", P.DEFAULT_PATCH_SIZE))

    def __len__(self) -> int:
        return len(self._pairs)

    def sample(self, i: int) -> PatchSample:
        fid, cidx = self._pairs[i]
        row = self._row_of[fid]

        # Re-assert per sample: a store mutated after construction (field id
        # rewritten) still cannot reach training.
        if not bool(H.train_only_mask([fid], manifest=self._holdout)[0]):
            raise H.HoldoutViolationError(f"field {fid} is a holdout field")

        vals = np.asarray(self.root["values"][row, cidx], dtype=np.float32)  # (T,C,S,S)
        sids = np.asarray(self.root["sensor_id"][row, cidx], dtype=np.int64)  # (T,)
        nobs = np.asarray(self.root["n_obs"][row, cidx], dtype=np.uint8)      # (T,)
        in_field = np.asarray(self.in_field[row, 0], dtype=bool)              # (S,S)
        start = int(np.asarray(self.root["cycle_start_day"][row, cidx]))
        cid = int(np.asarray(self.root["cycle_ids"][row, cidx]))

        finite = np.isfinite(vals)
        obs = finite & in_field[None, None, :, :]
        x = np.where(obs, vals, 0.0).astype(np.float32)
        pixel_valid = obs.any(axis=1)                                          # (T,S,S)

        x = self._normalize(x, sids, obs)
        cov = build_covariates(
            np.full(P.N_BINS, start), nobs, sids,
            in_field_frac=float(in_field.mean()),
        )
        return PatchSample(
            field_id=fid, cycle_id=cid, x=x, obs=obs, pixel_valid=pixel_valid,
            covariates=cov, start_day=start,
        )

    def _normalize(self, x: np.ndarray, sids: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """Frozen per-channel × per-sensor robust standardization.

        Values that are unobserved stay exactly 0 (they carry no information
        and must not become a non-zero constant after centring). Channels with
        no frozen statistics for a sensor keep raw units and are flagged.
        """
        out = x.copy()
        T, C = x.shape[0], x.shape[1]
        for sid in np.unique(sids):
            name = P.SENSOR_NAMES.get(int(sid), str(sid))
            per_ch = self.norm.get(name, {})
            bins = np.flatnonzero(sids == sid)
            if bins.size == 0:
                continue
            for ci, ch in enumerate(self.channel_names):
                stats = per_ch.get(ch, {})
                med, iqr = stats.get("median"), stats.get("iqr")
                if med is None or iqr is None or iqr <= 0:
                    self.unnormalized_channels = getattr(self, "unnormalized_channels", set())
                    self.unnormalized_channels.add(ch)
                    continue
                sel = obs[bins, ci]
                out[np.ix_(bins, [ci])] = np.where(
                    sel[:, None, :, :],
                    (x[bins, ci][:, None, :, :] - med) / iqr,
                    0.0,
                )
        if out.shape[:2] != (T, C):  # pragma: no cover - shape guard
            raise RuntimeError("normalization changed the tensor shape")
        return out.astype(np.float32)


def collate(samples: list[PatchSample]) -> dict[str, np.ndarray]:
    """Stack samples into batch arrays for the model/loss."""
    return {
        "x": np.stack([s.x for s in samples]).astype(np.float32),
        "obs": np.stack([s.obs for s in samples]).astype(bool),
        "pixel_valid": np.stack([s.pixel_valid for s in samples]).astype(bool),
        "covariates": np.stack([s.covariates for s in samples]).astype(np.float32),
        "field_ids": np.array([s.field_id for s in samples], dtype=np.int64),
        "cycle_ids": np.array([s.cycle_id for s in samples], dtype=np.int64),
    }


def estimate_epoch_seconds(
    n_samples: int, *, seconds_per_sample: float, batch_size: int = 8,
) -> dict:
    """Runtime estimate helper shared by ``--estimate-runtime`` (task 1.4)."""
    n_batches = int(np.ceil(max(1, n_samples) / max(1, batch_size)))
    per_epoch = n_batches * float(seconds_per_sample)
    return {
        "n_samples": int(n_samples),
        "batch_size": int(batch_size),
        "n_batches": n_batches,
        "seconds_per_sample": float(seconds_per_sample),
        "seconds_per_epoch": per_epoch,
    }


__all__ = [
    "COVARIATES",
    "N_COVARIATES",
    "PatchCycleDataset",
    "PatchSample",
    "build_covariates",
    "collate",
    "estimate_epoch_seconds",
]

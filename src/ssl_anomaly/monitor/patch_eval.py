"""Patch arms inside ``eval_dev.py``: the store→store-free boundary.

Change ``patch-level-controlled-trial`` task 2.4.

`eval_dev.py` is written against FIELD-level cycles — ``(N, T, C)`` arrays in
memory. The patch arms need the ``patches.zarr`` tensors ``(N, T, C, S, S)``
plus their in-field masks. This module is the only place that decides how the
two line up, so the decision is visible and testable instead of being an
ad-hoc branch inside the sweep:

* **The join key is ``(field_id, cycle_id)``, never row position.** A patch
  store and a field-level array can hold the same cycles in different orders;
  joining on position would silently score farm A's patch against farm B's
  cycle, and every downstream metric would look plausible.
* **Missing patch cycles are a hard error, not a dropped sample.** Dropping
  them would change the evaluation population between arms, which is exactly
  the confound the controlled trial exists to remove.
* **The injection is applied in the PATCH tensor** by
  ``monitor/patch_perturbations.py`` with the same severity table, shapes and
  hashes as the field-level ladder — never by re-injecting in field space and
  hoping the patch slice matches.
* **The in-field mask comes from the store**, not from the values: out-of-field
  pixels must never enter the aggregation domain.

The severity sweep consumes a ``score_cycles``-shaped callable, so this module
returns one plus the reference paths needed by ``PatchArmScorer``.
"""

from __future__ import annotations

import numpy as np

SENTINEL_ABSENT = "absent"


def patch_batch_for_cycles(
    root,
    field_ids: np.ndarray,
    cycle_ids: np.ndarray,
    *,
    norm: dict,
) -> dict:
    """Gather the patch tensors for ``(field_id, cycle_id)`` pairs.

    Returns ``{"X": (N,T,C,S,S) float32 normalized, "pixel_valid": (N,T,S,S),
    "in_field": (N,S,S), "sensor_id": (N,T), "cycles_found": int}``.

    Values are normalized with the run's OWN frozen sidecar, per channel ×
    sensor, exactly as training did — a patch arm scored in raw units against a
    σ table fitted in normalized units would look calibrated and be wrong.
    """
    store_fields = np.asarray(root["field_ids"][:], dtype=np.int64)
    row_of = {int(f): i for i, f in enumerate(store_fields)}
    channel_names = tuple(root.attrs.get("channel_order", ()))
    if not channel_names:
        raise ValueError("the patch store has no channel_order attribute")

    field_ids = np.asarray(field_ids, dtype=np.int64).ravel()
    cycle_ids = np.asarray(cycle_ids, dtype=np.int64).ravel()
    if field_ids.shape != cycle_ids.shape:
        raise ValueError(
            f"field_ids {field_ids.shape} and cycle_ids {cycle_ids.shape} must align"
        )
    missing_fields = sorted({int(f) for f in field_ids if int(f) not in row_of})
    if missing_fields:
        raise ValueError(f"{len(missing_fields)} cycles' fields are not in the store: "
                         f"{missing_fields[:5]}")

    first = np.asarray(root["values"][row_of[int(field_ids[0])], int(cycle_ids[0])],
                       dtype=np.float32)
    T, C, S, _ = first.shape
    N = field_ids.size
    X = np.empty((N, T, C, S, S), dtype=np.float32)
    pixel_valid = np.empty((N, T, S, S), dtype=bool)
    in_field = np.empty((N, S, S), dtype=bool)
    sensor_ids = np.empty((N, T), dtype=np.int64)

    for n, (fid, cid) in enumerate(zip(field_ids, cycle_ids)):
        row = row_of[int(fid)]
        vals = np.asarray(root["values"][row, int(cid)], dtype=np.float32)
        if vals.shape != (T, C, S, S):
            raise ValueError(f"cycle {cid} of field {fid} is {vals.shape}, expected {(T, C, S, S)}")
        mask_field = np.asarray(root["in_field"][row, 0], dtype=bool)
        finite = np.isfinite(vals)
        obs = finite & mask_field[None, None]
        x = np.where(obs, vals, 0.0).astype(np.float32)
        for sid in np.unique(np.asarray(root["sensor_id"][row, int(cid)], dtype=np.int64)):
            name = {1: "L8", 2: "L9"}.get(int(sid), "unknown")
            per_ch = norm.get(name, {})
            bins = np.flatnonzero(np.asarray(root["sensor_id"][row, int(cid)], dtype=np.int64) == sid)
            for ci, ch in enumerate(channel_names):
                stats = per_ch.get(ch, {})
                med, iqr = stats.get("median"), stats.get("iqr")
                if med is None or iqr is None or iqr <= 0:
                    continue
                sel = obs[bins, ci]
                x[np.ix_(bins, [ci])] = np.where(
                    sel[:, None, :, :], (vals[bins, ci][:, None, :, :] - med) / iqr, 0.0,
                )
        X[n] = x
        pixel_valid[n] = obs.any(axis=1)
        in_field[n] = mask_field
        sensor_ids[n] = np.asarray(root["sensor_id"][row, int(cid)], dtype=np.int64)
    return {
        "X": X, "pixel_valid": pixel_valid, "in_field": in_field,
        "sensor_id": sensor_ids, "cycles_found": int(N),
    }


def patch_reference_paths(
    arm_scorer,
    batch: dict,
    *,
    reference_idx: np.ndarray,
    field_row: np.ndarray | None = None,
    chunk: int = 32,
) -> np.ndarray:
    """Clean-calibration paths used by the empirical p-value.

    Restricted to ``reference_idx`` (the CALIBRATION subset). Using the
    evaluation subset as its own reference would make every p-value a rank
    within the set being judged — detection would look perfect and mean
    nothing.
    """
    idx = np.asarray(reference_idx, dtype=np.int64)
    if idx.size == 0:
        raise ValueError("no calibration cycles for the patch arm's reference paths")
    paths = np.full((idx.size, batch["X"].shape[1]), np.nan)
    for n0 in range(0, idx.size, max(1, chunk)):
        sl = idx[n0 : n0 + max(1, chunk)]
        # ``field_row`` names each sample's in-field mask row; it is required
        # when the batch is a SLICE of a larger field table (the evaluation
        # sweep always slices), and omitted when in_field is per-sample.
        extra = ({} if field_row is None
                 else {"idx": np.asarray(field_row)[sl]})
        scores, _pred = arm_scorer.loo_batch(
            batch["X"][sl], batch.get("pixel_valid", batch.get("M"))[sl], **extra,
        )
        paths[n0 : n0 + sl.size] = scores
    return paths


def patch_arm_scores(
    arm,
    batch: dict,
    *,
    level_id: str,
    family: str,
    index: str,
    round_idx: int,
    rng=None,
    field_row: np.ndarray | None = None,
) -> dict:
    """Score one severity cell: inject in the PATCH tensor, then score.

    Returns ``{"z2": (N,T,1), "paths": (N,T), "injections": [...]}``. The
    injections carry their hash payload, so the cell's identity in the report
    is verifiable against the field-level ladder's.
    """
    from ssl_anomaly.monitor import patch_perturbations as ppert
    from ssl_anomaly.monitor import severity_table as sev

    levels = {lv.level_id: lv for lv in sev.load_table()}
    if level_id not in levels:
        raise ValueError(f"unknown severity level {level_id!r}")
    delta = float(levels[level_id].delta_ndvi)
    duration = (sev.PERSISTENT_DURATION_BINS if family == sev.PERSISTENT_FAMILY
                else levels[level_id].duration_bins)

    rng = rng if rng is not None else np.random.default_rng(abs(hash(level_id)) % 2**32)
    injections = []
    Xinj = np.empty_like(batch["X"])
    Minj = np.empty_like(batch["pixel_valid"])
    for n in range(batch["X"].shape[0]):
        onset = ppert.onset_for_shape(family, batch["X"].shape[1], 3, int(duration))
        inj = ppert.inject_patch_drop(
            batch["X"][n], np.ones_like(batch["X"][n], dtype=bool),
            index=index, delta=delta, onset_bin=onset, shape=family,
            duration=int(duration), in_field=batch["in_field"][n],
            level_id=level_id, rng=rng,
        )
        Xinj[n] = inj.x
        # The mask is never changed by an injection (same contract as the
        # field-level applier), so the batch mask carries over verbatim.
        Minj[n] = batch["pixel_valid"][n]
        injections.append(inj.to_hash_record())
    extra = {} if field_row is None else {"idx": np.asarray(field_row)}
    out = arm.score_cycles(Xinj, Minj, None, None, **extra)
    scores, _pred = arm.scorer.loo_batch(Xinj, Minj, **extra)
    return {"z2": out["z2"], "paths": scores, "p_value": out["p_value"],
            "observed": Minj.any(axis=(2, 3)),   # (N, T): any in-field pixel seen
            "injections": injections}

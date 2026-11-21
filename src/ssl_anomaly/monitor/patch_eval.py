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

from ssl_anomaly.io import patches as P
from ssl_anomaly.monitor.patch_perturbations import BASIS_INDEX_ONLY
from ssl_anomaly.train.patch_data import read_cycle_inputs

SENTINEL_ABSENT = "absent"


def patch_batch_for_cycles(
    root,
    field_ids: np.ndarray,
    cycle_ids: np.ndarray,
    *,
    norm: dict,
) -> dict:
    """Join by stable keys and keep raw inputs for physical-unit challenges."""
    epoch = P.patch_epoch(root)
    index = P.patch_cycle_index(root)
    stats = P.patch_norm_stats(norm)
    field_ids = np.asarray(field_ids, dtype=np.int64).ravel()
    cycle_ids = np.asarray(cycle_ids, dtype=np.int64).ravel()
    if field_ids.shape != cycle_ids.shape or not field_ids.size:
        raise ValueError("field_ids and cycle_ids must align and be non-empty")
    keys = [f"{f}:{c}" for f, c in zip(field_ids, cycle_ids)]
    missing = [key for key in keys if key not in index]
    if missing:
        raise ValueError(f"{len(missing)} requested patch cycles are not in the store: {missing[:5]}")
    cycles = [index[key] for key in keys]
    P.assert_patch_count_bounds(root, cycles)
    # Allocate each output only once; retaining every cycle and then stacking
    # doubles the high-water mark. read_cycle_inputs preserves the store dtype
    # (normally float16) and already returns a boolean observation mask.
    arrays = {}
    missing_scales = set()
    for n, cycle in enumerate(cycles):
        data = read_cycle_inputs(root, cycle, epoch=epoch)
        x, missing = P.normalize_patch(data["raw"], data["obs"], data["sensor_id"], stats)
        missing_scales.update(missing)
        data["X"] = x
        for key in ("X", "raw", "obs", "sensor_id", "in_field", "covariates"):
            value = data[key]
            if key not in arrays:
                arrays[key] = np.empty((len(cycles), *value.shape), dtype=value.dtype)
            arrays[key][n] = value
    return {
        **arrays, "unnormalized_channels": sorted(missing_scales),
        "pixel_valid": arrays["obs"].any(axis=2),
        "cycles_found": len(keys), "norm_stats": stats,
    }


def _scoring_inputs(batch: dict, rows=slice(None)) -> dict:
    names = {"sensor_ids": "sensor_id", "covariates": "covariates", "obs": "obs"}
    return {kw: batch[key][rows] for kw, key in names.items() if key in batch}


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
            batch["X"][sl], batch.get("pixel_valid", batch.get("M"))[sl],
            **extra, **_scoring_inputs(batch, sl),
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
    injection_basis: str = BASIS_INDEX_ONLY,
    chunk: int | None = None,
) -> dict:
    """Score one severity cell: inject in the PATCH tensor, then score.

    Returns ``{"z2": (N,T,1), "paths": (N,T), "injections": [...]}``. The
    injections carry their hash payload, so the cell's identity in the report
    is verifiable against the field-level ladder's. ``injection_basis``
    selects the injector semantics (change consistent-spectral-injection); the
    caller — the eval sweep — decides which basis is the primary gate.

    The onset phase rotates through early/mid/late per ``(sample + round) % 3``
    and the gradual ramp length cycles ``GRADUAL_RAMP_BINS`` per round — the
    exact field-ladder protocol — so consecutive rounds are genuinely different
    treatments and the sweep's Wilson CI counts independent injections.
    """
    from ssl_anomaly.monitor import patch_perturbations as ppert
    from ssl_anomaly.monitor import severity_table as sev

    levels = {lv.level_id: lv for lv in sev.load_table()}
    if level_id not in levels:
        raise ValueError(f"unknown severity level {level_id!r}")
    delta = float(levels[level_id].delta_ndvi)
    if family == sev.PERSISTENT_FAMILY:
        duration = sev.PERSISTENT_DURATION_BINS
    elif family == sev.GRADUAL_FAMILY:
        # Round parity with the field ladder (perturbations.
        # inject_index_drop_dataset): the gradual ramp cycles through the
        # spec's D ∈ {3, 4, 6} per round; inject_patch_drop pulls the onset
        # earlier when the full ramp would not fit, and both are recorded.
        duration = int(sev.GRADUAL_RAMP_BINS[round_idx % len(sev.GRADUAL_RAMP_BINS)])
    else:
        duration = levels[level_id].duration_bins

    rng = rng if rng is not None else np.random.default_rng(round_idx)
    injections = []
    shape = batch["raw"].shape
    n_samples = shape[0]
    if chunk is not None and chunk <= 0:
        raise ValueError("score chunk must be positive")
    chunk_size = n_samples if chunk is None else chunk
    # A caller may omit X after calibration; normalize_patch always emits
    # float32. Preserve an explicitly supplied X dtype for legacy callers.
    x_dtype = batch["X"].dtype if "X" in batch else np.float32
    outputs = {key: [] for key in ("z2", "score", "p_value")}
    Minj = batch["pixel_valid"]
    for start in range(0, n_samples, chunk_size):
        end = min(start + chunk_size, n_samples)
        Xinj = np.empty((end - start, *shape[1:]), dtype=x_dtype)
        # The same RNG is consumed once per sample, in global batch order.
        for n in range(start, end):
            # Onset phases rotate deterministically through early/mid/late
            # exactly like the field ladder (cycle i, round r → phase
            # (i + r) % 3). The pre-fix code pinned onset to bin 3, which made
            # every round of a cell a byte-identical injection (the Wilson CI
            # then counted pseudo-replicates) and measured detection from bin
            # 0 instead of the onset (audit 2026-11-21).
            from ssl_anomaly.monitor import perturbations as mpert
            phase = mpert.PHASES[(n + round_idx) % len(mpert.PHASES)]
            onset = mpert.phase_start_bin(phase, shape[1])
            inj = ppert.inject_patch_drop(
                batch["raw"][n].astype(np.float32), batch["obs"][n],
                index=index, delta=delta, onset_bin=onset, shape=family,
                duration=int(duration), in_field=batch["in_field"][n],
                level_id=level_id, rng=rng, injection_basis=injection_basis,
            )
            Xinj[n - start], _ = P.normalize_patch(
                inj.x, batch["obs"][n], batch["sensor_id"][n], batch["norm_stats"]
            )
            injections.append(inj.to_hash_record())
        sl = slice(start, end)
        extra = {} if field_row is None else {"idx": np.asarray(field_row)[sl]}
        out = arm.score_cycles(Xinj, Minj[sl], None, None, **extra, **_scoring_inputs(batch, sl))
        for key in outputs:
            outputs[key].append(out[key])
    return {"z2": np.concatenate(outputs["z2"]), "paths": np.concatenate(outputs["score"]),
            "p_value": np.concatenate(outputs["p_value"]),
            "observed": Minj.any(axis=(2, 3)),
            "injection_basis": injection_basis, "injections": injections}


# --- G4 localization probe (patch-level-controlled-trial task 2.4) --------
# The PoC's MDM80 tied at S1 for both patch arms (finding 23), and the
# pre-registered G4 rule says the CNN is kept only if it beats patch_pixelwise
# on MDM80 or localization AUPRC — so the tie is broken here, on whether the
# arm's per-pixel scores rank the actually-injected pixels above the rest of
# the field when only PART of the field is injected.

LOCALIZATION_FOOTPRINT = 0.3    # contiguous in-field blob; protocol gates use 1.0


def localization_auprc(
    pixel_z2: np.ndarray,         # (N, T, S, S); NaN outside the scored domain
    footprint_masks: np.ndarray,  # (N, S, S) bool — injected pixels (⊆ in_field)
    in_field: np.ndarray,         # (N, S, S) bool — the negative domain
    onset_bins: np.ndarray,       # (N,) int
    durations: np.ndarray,        # (N,) int
) -> dict:
    """Pixel-localization AUPRC over the injection windows.

    Positives are the injected pixels, negatives the remaining in-field
    pixels, both restricted to the bins the injection actually touches
    (``[onset, onset + duration)`` per sample) and to pixels the scorer
    actually scored (finite z²). Scores are pooled over samples and affected
    bins — one AP number per cell — and the positive prevalence is reported
    alongside as the chance level an uninformative ranking achieves.

    ``auprc`` is ``None`` when the pooled window has no scored positives
    (e.g. an unobserved footprint) or no negatives (footprint 1.0): an AP
    over a single class is undefined and must not be fabricated.
    """
    from sklearn.metrics import average_precision_score

    z2 = np.asarray(pixel_z2, dtype=np.float64)
    fp = np.asarray(footprint_masks, dtype=bool)
    inf = np.asarray(in_field, dtype=bool)
    N, T = z2.shape[:2]
    onset = np.asarray(onset_bins, dtype=np.int64)
    dur = np.asarray(durations, dtype=np.int64)
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for n in range(N):
        lo = int(max(0, onset[n]))
        hi = int(min(T, onset[n] + max(1, dur[n])))
        if hi <= lo:
            continue
        window = z2[n, lo:hi]
        scored = np.isfinite(window)
        pos = fp[n][None, :] & scored
        neg = inf[n][None, :] & ~fp[n][None, :] & scored
        if pos.any():
            scores.append(window[pos])
            labels.append(np.ones(int(pos.sum()), dtype=bool))
        if neg.any():
            scores.append(window[neg])
            labels.append(np.zeros(int(neg.sum()), dtype=bool))
    y = np.concatenate(labels) if labels else np.zeros(0, dtype=bool)
    s = np.concatenate(scores) if scores else np.zeros(0)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return {"auprc": None, "n_pos": n_pos, "n_neg": n_neg, "prevalence": None}
    prevalence = n_pos / y.size
    return {"auprc": float(average_precision_score(y, s)),
            "n_pos": n_pos, "n_neg": n_neg, "prevalence": float(prevalence)}


def patch_localization_scores(
    scorer, batch: dict, *, level_id: str, family: str, index: str,
    round_idx: int = 0, footprint: float = LOCALIZATION_FOOTPRINT,
    injection_basis: str = BASIS_INDEX_ONLY,
) -> dict:
    """One G4 cell: inject at a partial footprint, keep the per-pixel maps.

    Like ``patch_arm_scores`` but (a) the footprint is parameterised (the
    pre-registered localization probe runs the default 0.3 — a contiguous
    in-field blob — while the severity gates run 1.0), and (b) the return
    carries the scorer's per-pixel z² maps plus the injection footprint /
    onset / duration that ``localization_auprc`` needs. No empirical-p
    calibration is involved: the localization ranking uses the frozen σ-table
    z² directly, exactly as trained.
    """
    from ssl_anomaly.monitor import patch_perturbations as ppert
    from ssl_anomaly.monitor import severity_table as sev

    levels = {lv.level_id: lv for lv in sev.load_table()}
    if level_id not in levels:
        raise ValueError(f"unknown severity level {level_id!r}")
    delta = float(levels[level_id].delta_ndvi)
    duration = (sev.PERSISTENT_DURATION_BINS if family == sev.PERSISTENT_FAMILY
                else levels[level_id].duration_bins)

    rng = np.random.default_rng(round_idx)
    Xinj = np.empty_like(batch["X"])
    N, T = batch["X"].shape[:2]
    S = batch["in_field"].shape[1]
    footprints = np.zeros((N, S, batch["in_field"].shape[2]), dtype=bool)
    onsets = np.zeros(N, dtype=np.int64)
    for n in range(N):
        onset = ppert.onset_for_shape(family, T, 3, int(duration))
        inj = ppert.inject_patch_drop(
            batch["raw"][n].astype(np.float32), batch["obs"][n],
            index=index, delta=delta, onset_bin=onset, shape=family,
            duration=int(duration), in_field=batch["in_field"][n],
            footprint=float(footprint), level_id=level_id, rng=rng,
            injection_basis=injection_basis,
        )
        Xinj[n], _ = P.normalize_patch(
            inj.x, batch["obs"][n], batch["sensor_id"][n], batch["norm_stats"]
        )
        footprints[n] = inj.footprint_mask
        onsets[n] = inj.onset_bin
    out = scorer.predict_pixels(Xinj, batch["pixel_valid"], **_scoring_inputs(batch))
    metrics = localization_auprc(
        out["pixel_z2"], footprints, batch["in_field"], onsets,
        np.full(N, int(duration), dtype=np.int64),
    )
    return {
        "metrics": metrics, "level_id": level_id, "family": family,
        "index": index, "onset_bins": onsets, "duration_bins": int(duration),
        "injection_basis": injection_basis,
        "footprint_requested": float(footprint),
        "footprint_realized": float(footprints.sum()
                                    / max(1, int(np.asarray(batch["in_field"], dtype=bool).sum()))),
        "n_pixels": int(footprints.sum()),
        "pixel_z2": out["pixel_z2"], "footprints": footprints,
    }

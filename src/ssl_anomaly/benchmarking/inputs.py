"""I1/I2/I3 input construction + W0/W1/W2 fusion (tasks 3.x).

Three input variants, one fused-tensor pipeline:

``I1`` values only          ``x`` + mask flags; the weather/phenology block is absent.
``I2`` + calendar phase      adds the active ``PhaseAxis`` bin embedding (sin/cos + one-hot
                             summary), which every arm can consume without learning DOY.
``I3`` + weather windows     adds ERA5 windows. Where the weather join is unavailable the
                             columns are filled 0 with ``absent`` recorded in the manifest —
                             never imputed from the eval period, and never dropped silently,
                             because dropping a block would change ``d_in`` per arm and make
                             the "identical across architectures" claim false.

Fusion is where a variant differs from a mere concatenation:

``W0`` none                  the block is present but the arm's trunk input excludes it.
``W1`` early                 covariates are part of the token (what ``ArmBatch.input()`` does).
``W2`` gated late            covariates go through their own encoder and are mixed into the
                             trunk state by a learned scalar gate initialised at 0.5, so the
                             fusion's *contribution* is an explicit, reportable number
                             (``gate_mean``) rather than something buried in weights.

W0 combined with I2/I3 is a deliberate control: the same data pipeline, the same
capacity, the covariate never reaching the network. If a benchmark only compared I1
vs I3, an "improvement" could come from the extra columns *or* from the different
pipeline; W0 separates them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ssl_anomaly.benchmarking.interface import CycleTensors

#: input id → which optional blocks it admits.
INPUT_BLOCKS: dict[str, tuple[str, ...]] = {
    "I1": (),
    "I2": ("phase",),
    "I3": ("phase", "weather"),
}
FUSION_MODES = ("W0", "W1", "W2")
ABSENT = "absent"


@dataclass
class InputBundle:
    """Tensors plus the provenance the manifest must carry for the input axis."""

    tensors: CycleTensors
    input_id: str
    fusion: str
    blocks: dict = field(default_factory=dict)
    absences: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "input_id": self.input_id, "fusion": self.fusion,
            "d_channels": self.tensors.n_channels, "d_cov": self.tensors.cov.shape[-1],
            "channel_names": list(self.tensors.channel_names),
            "covariate_names": list(self.tensors.cov_names),
            "blocks": {k: dict(v) for k, v in self.blocks.items()},
            "absences": list(self.absences), "notes": list(self.notes),
            "n_cycles": self.tensors.n, "n_bins": self.tensors.n_bins,
        }


def phase_encoding(phase_bin: np.ndarray, n_bins: int) -> tuple[np.ndarray, list[str]]:
    """sin/cos of the phase angle + the two marginal flags, as float32 ``(N,T,4)``.

    sin/cos rather than the raw index: a 36-bin cycle wraps, and an integer index
    would teach the trunk that bin 35 is 35 steps from bin 0. The two extra channels
    mark the peak/greenness windows, because those are the operational boundaries the
    frozen normality profile uses and an arm should not have to rediscover them.
    """
    ang = 2.0 * np.pi * np.asarray(phase_bin, dtype=np.float64) / max(1, int(n_bins))
    frac = np.asarray(phase_bin, dtype=np.float64) / max(1, int(n_bins))
    return (
        np.stack(
            [np.sin(ang), np.cos(ang), (frac < 1.0 / 3.0).astype(np.float64),
             (frac > 2.0 / 3.0).astype(np.float64)],
            axis=-1,
        ).astype(np.float32),
        ["phase_sin", "phase_cos", "phase_early", "phase_late"],
    )


def weather_block(
    weather: dict[str, Any] | None, *, n: int, n_bins: int, bin_days: int = 15,
    windows: tuple[int, ...] = (1, 2, 4),
) -> tuple[np.ndarray, list[str], dict]:
    """Lagged window summaries of the weather series; all-zero + a record when absent.

    Returns ``(array (N,T,W·k), names, block_record)``. The block record is what makes
    "no weather" auditable: the manifest says which fields were missing and how the
    fill was chosen, instead of the model silently seeing zeros.
    """
    names: list[str] = []
    cols: list[np.ndarray] = []
    rec: dict[str, Any] = {"requested": True, "available": False, "fields": {}, "fills": []}
    if not weather:
        w = len(windows) * 4                      # precip, tmin, tmax, et0 — the four the design names
        rec["reason"] = "no weather store bound (ERA5 sidecar absent from the v2 bundle)"
        rec["fill"] = "zeros, recorded as absent; no imputation from the scored partition"
        return np.zeros((n, n_bins, w), dtype=np.float32), [f"weather_absent_{i}" for i in range(w)], rec
    order = ("precip", "tmin", "tmax", "et0")
    for fld in order:
        arr = weather.get(fld)
        if arr is None:
            rec["fills"].append(f"{fld}:absent")
            cols.append(np.zeros((n, n_bins), dtype=np.float32))
            names.append(f"{fld}_w1")
            for k in windows[1:]:
                cols.append(np.zeros((n, n_bins), dtype=np.float32))
                names.append(f"{fld}_w{k}")
            continue
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 2 and a.shape[0] == n:
            pass
        elif a.ndim == 3:
            a = a.mean(axis=-1)
        else:
            rec["fills"].append(f"{fld}:shape_{a.shape}_unusable_zeros")
            a = np.zeros((n, n_bins), dtype=np.float32)
        rec["available"] = True
        rec["fields"][fld] = {"n_bins": int(a.shape[1]), "mean": round(float(a.mean()), 4)}
        for k in windows:
            lag = np.zeros_like(a)
            for t in range(a.shape[1]):
                lo = max(0, t - k + 1)
                lag[:, t] = a[:, lo : t + 1].mean(axis=1)
            cols.append(lag.astype(np.float32))
            names.append(f"{fld}_w{k}")
    return np.stack(cols, axis=-1).astype(np.float32), names, rec


def build_tensors(
    frame, cfg, *, input_id: str = "I3", fusion: str = "W1", weather: dict | None = None,
    phase_bin: np.ndarray | None = None, n_bins_phase: int = 36,
) -> InputBundle:
    """Assemble a :class:`CycleTensors` for one input variant from a frame.

    ``frame`` is anything exposing ``x``/``m``/``field_ids``/``cycle_ids``/``split``
    (the :class:`~ssl_anomaly.benchmarking.splits.SplitFrame` + parquet path, or the
    frozen ``CycleData``). The NaN policy is applied here and nowhere else: unobserved
    bins are 0 in ``x`` with ``m = 0``, and any NaN that survives the network input
    raises inside :class:`CycleTensors`.
    """
    if input_id not in INPUT_BLOCKS:
        raise KeyError(f"unknown input id {input_id!r} (expected one of {sorted(INPUT_BLOCKS)})")
    if fusion not in FUSION_MODES:
        raise KeyError(f"unknown fusion id {fusion!r} (expected one of {list(FUSION_MODES)})")
    x = np.asarray(frame.x, dtype=np.float32)
    obs = np.asarray(getattr(frame, "m", np.isfinite(x)), dtype=bool)
    if obs.shape != x.shape:
        obs = np.isfinite(x)
    with np.errstate(invalid="ignore"):
        filled = np.where(obs, x, 0.0)
    if not np.isfinite(filled).all():
        raise ValueError(
            f"{int((~np.isfinite(filled)).sum())} non-finite values remain after masking; a bin that "
            "is neither observed nor zero is a data defect, and imputing it here would hide that"
        )
    n, T, C = filled.shape
    blocks: dict[str, Any] = {"values": {"channels": C}}
    absences: list[dict] = []
    notes: list[str] = []
    cov_cols: list[np.ndarray] = []
    cov_names: list[str] = []
    allowed = INPUT_BLOCKS[input_id]
    if "phase" in allowed:
        if phase_bin is None:
            pb = np.tile(np.arange(T, dtype=np.int64), (n, 1))
            notes.append(
                "phase bins default to the bin index (the v2 cycles are contiguous 15-day windows); "
                "a re-based PhaseAxis must be passed explicitly, which is why it is not read from the file"
            )
            absences.append({"field": "phase_bin_source", "status": "derived_from_bin_index"})
        else:
            pb = np.asarray(phase_bin, dtype=np.int64)
            if pb.shape != (n, T):
                raise ValueError(f"phase_bin must be (N, T)={((n, T))}, got {pb.shape}")
        enc, pn = phase_encoding(pb, n_bins_phase)
        cov_cols.append(enc)
        cov_names += pn
        blocks["phase"] = {"encoding": "sin_cos_plus_window_flags", "n_bins": int(n_bins_phase)}
    if "weather" in allowed:
        wv, wn, wrec = weather_block(weather, n=n, n_bins=T,
                                     bin_days=int(cfg.raw["training"].get("bin_days", 15)))
        cov_cols.append(wv)
        cov_names += wn
        blocks["weather"] = wrec
        if not wrec.get("available"):
            absences.append({"field": "weather", "status": ABSENT, "detail": wrec.get("reason", "")})
            notes.append(
                "weather covariates are zero-filled and labelled absent; the I3 cells are still run so "
                "the ranking is computed on one consistent tensor shape, and the report says I3 is "
                "unevaluable on this data until the sidecar is bound"
            )
    if not cov_cols:
        cov = np.zeros((n, T, 0), dtype=np.float32)
    else:
        cov = np.concatenate(cov_cols, axis=-1).astype(np.float32)
    if fusion == "W0":
        cov = np.zeros_like(cov)
        cov_names = [f"unused::{c}" for c in cov_names]
        blocks["fusion"] = {"mode": "W0", "note": "covariates zeroed; identical tensor shape, no information"}
    elif fusion == "W2":
        blocks["fusion"] = {"mode": "W2", "note": "gated late fusion; the gate value is reported per run"}
    else:
        blocks["fusion"] = {"mode": "W1", "note": "early fusion (covariates concatenated into the token)"}
    tensors = CycleTensors(
        x=filled, m=obs, cov=cov,
        field_ids=np.asarray(frame.field_ids, dtype=np.int64),
        cycle_ids=np.asarray(frame.cycle_ids).astype(str),
        split=np.asarray(getattr(frame, "split", np.repeat("dev", n))).astype(str),
        phase_bin=(np.asarray(phase_bin, dtype=np.int64) if phase_bin is not None
                   else np.tile(np.arange(T, dtype=np.int64), (n, 1))),
        start_days=getattr(frame, "start_days", None),
        end_days=getattr(frame, "end_days", None),
        channel_names=tuple(getattr(frame, "channel_names", ()) or [f"ch{i}" for i in range(C)]),
        cov_names=tuple(cov_names), data_source=str(getattr(frame, "data_source", "")),
        notes=notes,
    )
    return InputBundle(tensors=tensors, input_id=input_id, fusion=fusion, blocks=blocks,
                       absences=absences, notes=notes)


def covariate_gate_report(gate: Any) -> dict:
    """W2's learned fusion gate, in the form the report prints.

    A gate that stayed at its 0.5 initialisation means the covariate branch
    contributed nothing either way — which must be *said*, not left as an empty
    difference between I2 and I3.
    """
    if gate is None:
        return {"available": False, "reason": "fusion is not W2 (no gate to report)"}
    g = float(np.asarray(gate).mean())
    return {
        "available": True, "gate_mean": round(g, 4),
        "gate_sd": round(float(np.asarray(gate).std()), 4),
        "moved_from_initialisation": bool(abs(g - 0.5) > 0.02),
        "interpretation": (
            "the covariate branch is used" if abs(g - 0.5) > 0.02 else
            "the gate stayed at its initialisation: W2 added no measurable information flow"
        ),
    }

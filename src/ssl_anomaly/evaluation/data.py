"""Dev-only data access for the capstone (fail-closed split discipline).

The spec requirement *"Test-split access attempt … THEN the loader raises an
error naming the allowed entry point and the run stops"* is implemented here as
the single gate every capstone module goes through. There is deliberately no
``split="test"`` path: the argument exists so that a caller *can* ask, and the
error says exactly which entry point is allowed (``freeze_main_model.py
--final-test``, which this change never executes).

Two data sources are supported:

- :func:`load_cycle_table` — the canonical ``cycles_490_v2.parquet`` (the real
  archive path). Absent in a clean checkout: raises
  :class:`ArtifactBindingError` with the pre-registration's own note rather than
  falling back to anything.
- :func:`synthetic_evaluation_split` — a deterministic, documented fixture used
  by unit tests and by ``--smoke`` runs. It is generated from an explicit seed
  and is tagged ``data_source="synthetic-fixture"`` so no report can mistake it
  for archive evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ssl_anomaly.evaluation.binding import ArtifactBindingError

#: The only entry point allowed to read the locked test split in this repo.
ALLOWED_TEST_ENTRY_POINT = "scripts/freeze_main_model.py --final-test"
#: Splits this change may read. ``test`` is intentionally absent.
ALLOWED_SPLITS = ("train", "dev")

N_BINS = 36
BIN_DAYS = 15


class TestSplitAccessError(RuntimeError):
    """Raised when any capstone code path asks for the locked test split."""


@dataclass
class CycleData:
    """A mask-aware cycle tensor bundle.

    ``X`` is float32 ``(N, T, C)`` with NaN at unobserved entries; ``M`` is the
    bool observation mask of the same shape. ``split`` is a per-cycle label,
    ``field_ids`` the farm id per cycle and ``dsh`` the days-since-harvest
    centre of each bin (``(T,)``), i.e. the phase axis used by the probes.
    """

    X: np.ndarray
    M: np.ndarray
    field_ids: np.ndarray
    cycle_ids: np.ndarray
    split: np.ndarray
    dsh: np.ndarray
    channel_names: tuple[str, ...]
    data_source: str = ""

    def __post_init__(self) -> None:
        if self.X.shape != self.M.shape:
            raise ValueError(f"X {self.X.shape} and M {self.M.shape} must align")
        if self.X.ndim != 3:
            raise ValueError(f"X must be (N, T, C), got {self.X.shape}")
        n = self.X.shape[0]
        for name in ("field_ids", "cycle_ids", "split"):
            arr = getattr(self, name)
            if len(arr) != n:
                raise ValueError(f"{name} has {len(arr)} entries for {n} cycles")
        # Observed entries must be finite, unobserved must be NaN. This is the
        # "masks dropped / NaN filled with 0" defect the AGENTS.md checklist
        # names: enforce the contract here so no downstream module can see a
        # zero that pretends to be an observation.
        observed = self.M.astype(bool)
        if np.any(~np.isfinite(self.X[observed])):
            raise ValueError("observed (mask=1) entries must be finite; found NaN/Inf")
        if np.any(np.isfinite(self.X[~observed])):
            bad = int(np.sum(np.isfinite(self.X[~observed])))
            raise ValueError(
                f"{bad} unobserved entries carry finite values; they must be NaN "
                "(zero-filled gaps overstate provenance)"
            )

    @property
    def n_cycles(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_bins(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_channels(self) -> int:
        return int(self.X.shape[2])

    def select(self, index: np.ndarray | list[int]) -> CycleData:
        idx = np.asarray(index, dtype=np.int64)
        return CycleData(
            X=self.X[idx],
            M=self.M[idx],
            field_ids=self.field_ids[idx],
            cycle_ids=self.cycle_ids[idx],
            split=self.split[idx],
            dsh=self.dsh,
            channel_names=self.channel_names,
            data_source=self.data_source,
        )

    def for_split(self, split: str) -> CycleData:
        """Return the cycles of one split, refusing anything outside the allowlist."""
        return self.select(self.indices_for_split(split))

    def indices_for_split(self, split: str) -> np.ndarray:
        if split == "test" or split not in ALLOWED_SPLITS:
            raise TestSplitAccessError(
                f"split {split!r} is not readable by the capstone evaluation "
                f"(allowed: {ALLOWED_SPLITS}). The locked test split has exactly one "
                f"legal entry point in this repository: {ALLOWED_TEST_ENTRY_POINT}, "
                f"which regional-ssl-representation-evaluation never executes."
            )
        return np.flatnonzero(self.split == split)

    def observed_counts(self) -> np.ndarray:
        """(N, C) count of observed bins per channel — denominators, not scores."""
        return self.M.sum(axis=1)

    def masked_nrmse(self, pred: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Per-channel NRMSE over OBSERVED entries only: ``(C,)``.

        A channel with zero observed entries returns NaN (``absent``), never 0 —
        a 0 would read as a perfect reconstruction.
        """
        m = self.M if mask is None else (self.M & mask)
        x = self.X.astype(np.float64)
        p = np.asarray(pred, dtype=np.float64)
        if p.shape != x.shape:
            raise ValueError(f"pred {p.shape} must match X {x.shape}")
        out = np.full(self.n_channels, np.nan)
        for c in range(self.n_channels):
            sel = m[:, :, c]
            n = int(sel.sum())
            if n == 0:
                continue
            diff = (x[:, :, c] - p[:, :, c])[sel]
            rms = float(np.sqrt(np.mean(diff**2)))
            scale = float(np.sqrt(np.mean(x[:, :, c][sel] ** 2)))
            out[c] = rms / scale if scale > 0 else np.nan
        return out


# ---------------------------------------------------------------------------
# Real archive loader (absent in a clean checkout — never substituted)
# ---------------------------------------------------------------------------


def load_cycle_table(
    cycles_parquet: str | Path,
    splits_manifest: str | Path,
    *,
    min_observed_channels: int = 1,
) -> CycleData:
    """Load the v2 cycle table into a :class:`CycleData` with split labels.

    Requires both the parquet and the locked splits manifest; raises
    :class:`ArtifactBindingError` if either is missing rather than producing an
    empty bundle.
    """
    import pandas as pd  # noqa: PLC0415 - heavy, only needed on the archive path

    pq, sp = Path(cycles_parquet), Path(splits_manifest)
    for p, label in ((pq, "cycles parquet"), (sp, "splits manifest")):
        if not p.is_file():
            raise ArtifactBindingError(
                f"{label} not found at {p}. The v2 chain lives outside this "
                "checkout (see configs/evaluation/pre_registration.yaml "
                "`artifacts:`); dependent cells must be recorded absent/deferred."
            )

    import json  # noqa: PLC0415

    df = pd.read_parquet(pq)
    manifest = json.loads(sp.read_text(encoding="utf-8"))
    split_map = {
        key: split
        for split, keys in manifest.get("splits", {}).items()
        for key in keys
    }

    if "split" in df.columns:
        split_labels = df["split"].astype(str).to_numpy()
    else:
        split_labels = np.array(
            [split_map.get(f"{int(f)}:{int(c)}", "unknown")
             for f, c in zip(df["field"], df["cycle_id"], strict=True)]
        )

    channel_names = tuple(
        getattr(df, "attrs", {}).get("channel_names", ())
        or ("ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst")
    )
    n_ch = len(channel_names)
    keys = [f"{int(f)}:{int(c)}" for f, c in zip(df["field"], df["cycle_id"], strict=True)]
    values = np.stack([np.asarray(v, dtype=np.float32) for v in df["values"]])
    masks = np.stack([np.asarray(v, dtype=bool) for v in df["mask"]])

    keep = masks.sum(axis=(1, 2)) >= min_observed_channels
    return CycleData(
        X=values[keep],
        M=masks[keep],
        field_ids=np.asarray(df["field"].to_numpy()[keep], dtype=np.int64),
        cycle_ids=np.asarray([k for k, kp in zip(keys, keep, strict=True) if kp]),
        split=np.asarray(split_labels[keep]),
        dsh=(np.arange(N_BINS, dtype=np.float32) + 0.5) * BIN_DAYS,
        channel_names=channel_names,
        data_source=f"cycles_parquet:{pq.name}",
    )


# ---------------------------------------------------------------------------
# Deterministic fixture (tests + --smoke only; never archive evidence)
# ---------------------------------------------------------------------------


def synthetic_evaluation_split(
    *,
    n_train: int = 24,
    n_dev: int = 16,
    n_fields: int = 8,
    n_bins: int = N_BINS,
    n_channels: int = 8,
    observed_frac: float = 0.85,
    seed: int = 13,
) -> CycleData:
    """A reproducible field-structured fixture with a real phase signal.

    The generating process (documented so the probes have something to find):

    - Each field has a phase offset and amplitude; the seasonal curve is a
      raised cosine over the bin axis shifted per field.
    - Channel 0 (ndvi) carries the strongest field-specific signal; later
      channels are progressively noisier copies — so a probe on channel 0 must
      beat a probe on the full stack less the phase information.
    - 15 % of (bin, channel) pairs are unobserved and set to NaN with mask 0.

    ``field_ids`` are 1..n_fields within the fixture and are recorded as such;
    no real farm identity is implied.
    """
    rng = np.random.default_rng(seed)
    n = n_train + n_dev
    phases = rng.uniform(0, 2 * np.pi, size=n_fields)
    amps = rng.uniform(0.6, 1.4, size=n_fields)

    bins = np.arange(n_bins, dtype=np.float64)
    X = np.full((n, n_bins, n_channels), np.nan, dtype=np.float32)
    M = np.zeros((n, n_bins, n_channels), dtype=bool)
    field_ids = rng.integers(1, n_fields + 1, size=n)

    for i in range(n):
        fid = int(field_ids[i])
        curve = 0.5 + 0.5 * np.cos((bins / n_bins) * 2 * np.pi - phases[fid - 1])
        curve = curve * amps[fid - 1]
        for c in range(n_channels):
            noise = rng.normal(0.0, 0.02 + 0.03 * c, size=n_bins)
            vals = curve * (1.0 - 0.06 * c) + noise
            obs = rng.random(n_bins) < observed_frac
            X[i, obs, c] = vals[obs].astype(np.float32)
            M[i, obs, c] = True

    split = np.array(["train"] * n_train + ["dev"] * n_dev)
    cycle_ids = np.array([f"F{f}:c{i}" for i, f in enumerate(field_ids)])
    names = ("ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst")[:n_channels]
    return CycleData(
        X=X,
        M=M,
        field_ids=field_ids,
        cycle_ids=cycle_ids,
        split=split,
        dsh=(np.arange(n_bins, dtype=np.float32) + 0.5) * BIN_DAYS,
        channel_names=tuple(names),
        data_source="synthetic-fixture",
    )

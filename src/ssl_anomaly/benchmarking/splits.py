"""Admissible data access: the locked-test gate in one place (task 10.x).

Every read path in this benchmark goes through this module, which is why it exists
as a file rather than a convention: the non-goal "the locked test split is never
read" is enforced by there being no other door.

The gate is a **deny-by-default allowlist over split names** plus the repository's
frozen holdout manifest, and it is deliberately not parameterised by a boolean:
``allow_test=True`` would be one flag away from being set somewhere by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

#: The only splits any benchmark read may name. Recorded here because the config
#: validator checks `evaluation.splits` against it, so the contract cannot widen it.
ADMISSIBLE_SPLITS: tuple[str, ...] = ("train", "dev")
FORBIDDEN = ("test", "holdout", "monitoring")


class TestSplitAccessError(RuntimeError):
    """Raised on any attempt to enumerate, load, or score the locked test split."""


def assert_admissible(splits) -> tuple[str, ...]:
    names = tuple(str(s) for s in (splits if isinstance(splits, (list, tuple, set)) else [splits]))
    if not names:
        raise ValueError("no splits named: an empty split list is not a valid benchmark read")
    bad = [n for n in names if n.lower() in FORBIDDEN]
    if bad:
        raise TestSplitAccessError(
            f"split(s) {bad} are locked for this change; admissible names are {list(ADMISSIBLE_SPLITS)}. "
            "Reading them would make the benchmark's headline numbers unclaimable, so this is an "
            "error and not a warning."
        )
    unknown = [n for n in names if n.lower() not in ADMISSIBLE_SPLITS]
    if unknown:
        raise ValueError(
            f"unknown split(s) {unknown}; the benchmark does not invent split names "
            f"(admissible: {list(ADMISSIBLE_SPLITS)})"
        )
    return tuple(n.lower() for n in names)


@dataclass
class SplitFrame:
    """Cycle-level index for the admissible splits, with provenance attached."""

    frame: pd.DataFrame
    splits: tuple[str, ...]
    n_by_split: dict[str, int]
    n_by_source: dict[str, int]
    lock: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    source: str = ""

    @property
    def n_cycles(self) -> int:
        return int(len(self.frame))

    def for_split(self, name: str) -> SplitFrame:
        assert_admissible([name])
        sub = self.frame[self.frame["split"].str.lower() == name.lower()].reset_index(drop=True)
        return SplitFrame(
            frame=sub, splits=(name,), n_by_split={name: int(len(sub))},
            n_by_source={k: int(v) for k, v in sub.get("source", pd.Series(dtype=str)).value_counts().items()},
            lock=self.lock, notes=list(self.notes),
        )

    def to_tensor_frame(self, *, base: str | Path | None = None):
        """The runner's view of this frame (mask/value pairs, index arrays, provenance).

        The cycle tensor columns are read from the same parquet that produced the
        index, so ``x`` and ``m`` cannot disagree with ``split``/``field_id`` about
        which rows exist: one file, one read, one order.
        """
        from ssl_anomaly.benchmarking.runner import TensorFrame

        cols = [c for c in self.frame.columns if c.startswith("x::") or c.startswith("obs::")]
        if not cols:
            raise FileNotFoundError(
                f"{self.source} carries no x::*/obs::* tensor columns; the benchmark reads the v2 "
                "cycle tensor table (data/processed/cycles_490_v2.parquet), and will not reconstruct "
                "it from the index columns here"
            )
        t = int(self.frame.get("n_bins", pd.Series([0])).iloc[0]) or None
        xs, ms = [], []
        for c in [k for k in self.frame.columns if k.startswith("x::")]:
            v = self.frame[c].to_numpy()
            xs.append(np.stack([np.asarray(a, dtype=np.float32).ravel() for a in v])
                       if v.dtype == object else v.reshape(len(self.frame), -1))
            oc = f"obs::{c[3:]}"
            w = self.frame.get(oc)
            ms.append(np.stack([np.asarray(a, dtype=bool).ravel() for a in w.to_numpy()])
                      if w is not None and w.dtype == object
                      else (w.to_numpy().reshape(len(self.frame), -1) if w is not None
                            else np.ones_like(xs[-1], dtype=bool)))
        x = np.stack(xs, axis=-1).astype(np.float32)
        m = np.stack(ms, axis=-1).astype(bool)
        if t and x.shape[1] != t:
            raise ValueError(
                f"tensor width {x.shape[1]} bins ≠ n_bins {t} in the index: the cycle table and the "
                "bin grid disagree, and silently trusting one of them would misalign every label"
            )
        sd = self.frame.get("start_day")
        ed = self.frame.get("end_day")
        return TensorFrame(
            x=x, m=m,
            start_days=None if sd is None else sd.to_numpy(),
            end_days=None if ed is None else ed.to_numpy(),
            field_ids=self.frame["field_id"].to_numpy(dtype=np.int64),
            cycle_ids=self.frame["cycle_id"].to_numpy(dtype=object),
            split=self.frame["split"].to_numpy(dtype=object),
            channel_names=tuple(sorted({c[3:].split("::")[0] for c in xs and [k for k in self.frame.columns if k.startswith("x::")]})),
            data_source=str(self.source),
            notes=list(self.notes),
        )

    def indices(self, name: str | None = None) -> np.ndarray:
        if name is None:
            return np.arange(len(self.frame), dtype=np.int64)
        assert_admissible([name])
        return np.flatnonzero((self.frame["split"].str.lower() == name.lower()).to_numpy())

    def audit(self) -> dict:
        return {
            "splits": list(self.splits),
            "n_by_split": dict(self.n_by_split),
            "n_by_source": dict(self.n_by_source),
            "n_cycles": self.n_cycles,
            "n_fields": int(self.frame["field_id"].nunique()) if "field_id" in self.frame else None,
            "test_accessed": False,
            "lock": dict(self.lock),
            "notes": list(self.notes),
        }


def load_admissible_splits(cfg, *, base: str | Path, splits=("train", "dev")) -> SplitFrame:
    """Read the cycle index for the admissible splits only.

    Two layouts are supported because v2 shipped both: a parquet of per-cycle rows,
    or the v2 ``data/processed/cycles_v2.parquet`` + ``splits_v2.csv`` pair that the
    frozen pipeline produces. Neither path can name a forbidden split: the filter is
    applied to whatever the file contains, and any unexpected value raises.

    Raises
    ------
    FileNotFoundError
        If no cycle index is present. It does **not** synthesise one from a random
        draw — a fabricated split table would make every coverage claim in the
        report a lie (AGENTS.md §6.1: nothing user-visible is synthesised).
    """
    names = assert_admissible(splits)
    base = Path(base)
    cfg_split = cfg.raw.get("data", {}).get("cycle_index") if isinstance(cfg.raw.get("data"), dict) else None
    candidates: list[Path] = []
    if cfg_split:
        candidates.append(base / cfg_split)
    candidates += [
        base / "data" / "processed" / "cycles_v2.parquet",
        base / "data" / "processed" / "cycles" / "cycles_v2.parquet",
        base / "outputs" / "thesis_colab_v2" / "cycles_v2.parquet",
    ]
    frame: pd.DataFrame | None = None
    used: Path | None = None
    for p in candidates:
        if p.is_file():
            frame = pd.read_parquet(p)
            used = p
            break
    if frame is None:
        raise FileNotFoundError(
            f"no cycle index found among {[str(p) for p in candidates]}; the benchmark will not "
            "fabricate a split table. Extract the v2 data bundle first (notebook §4)."
        )
    if "split" not in frame.columns:
        sp = used.parent / "splits_v2.csv"
        if not sp.is_file():
            raise FileNotFoundError(
                f"'split' column missing from {used} and no splits_v2.csv beside it; split "
                "membership must come from the frozen pipeline, not from a guess"
            )
        tab = pd.read_csv(sp)
        key = [c for c in ("cycle_id", "field_id", "cycle_index") if c in tab.columns]
        if not key:
            raise ValueError(f"splits table {sp} has no join key")
        frame = frame.merge(tab, on=key, how="left")
    frame["split"] = frame["split"].astype(str).str.strip().str.lower()
    unknown = sorted(set(frame["split"].unique()) - set(ADMISSIBLE_SPLITS) - set(FORBIDDEN))
    if unknown:
        raise ValueError(
            f"unexpected split labels in {used.name}: {unknown}; the frozen pipeline's vocabulary is "
            f"{list(ADMISSIBLE_SPLITS) + list(FORBIDDEN)} and a new label must be resolved by hand"
        )
    keep = frame["split"].isin(names)
    n_forbidden = int(frame["split"].isin(FORBIDDEN).sum())
    sub = frame[keep].reset_index(drop=True)
    if sub.empty:
        raise ValueError(
            f"no cycles in the admissible splits {list(names)} of {used.name} — an empty benchmark "
            "run must be a loud failure, not a table of NaNs"
        )
    if "field_id" not in sub.columns:
        for cand in ("field", "farm_id", "reach_id"):
            if cand in sub.columns:
                sub = sub.rename(columns={cand: "field_id"})
                break
        else:
            raise ValueError(
                f"{used.name} has no field identifier; farm-level metrics and the cluster bootstrap "
                "are defined over fields, so their absence is fatal rather than approximated"
            )
    lock = _read_lock(base)
    notes: list[str] = []
    if n_forbidden:
        notes.append(
            f"{n_forbidden} rows labelled {list(FORBIDDEN)} were dropped before any model saw them; "
            "the locked split is never read, counted for training, or used for threshold setting"
        )
    return SplitFrame(
        frame=sub, splits=names, source=str(used),
        n_by_split={s: int((sub["split"] == s).sum()) for s in names},
        n_by_source={k: int(v) for k, v in sub["source"].value_counts().items()}
        if "source" in sub.columns else {},
        lock=lock, notes=notes,
    )


def _read_lock(base: Path) -> dict:
    """The frozen split lock, for provenance only (never modified)."""
    import json

    for rel in ("configs/evaluation/pre_registration.yaml", "data/processed/splits_locked_v2.json"):
        p = base / rel
        if p.is_file():
            try:
                if p.suffix == ".json":
                    return {"path": rel, **json.loads(p.read_text(encoding="utf-8"))}
                import yaml

                doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                return {"path": rel, "keys": sorted(doc)[:12]}
            except Exception as exc:  # noqa: BLE001 - provenance must not kill a run
                return {"path": rel, "error": f"{type(exc).__name__}: {exc}"}
    return {"path": None, "error": "no split lock found on disk"}

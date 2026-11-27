"""Shared helpers for the OlmoEarth evaluation battery (E1–E7).

All functions are numpy/pandas/sklearn only — no torch. Everything reads the
extraction artifacts under ``experiments/olmoearth_eval/outputs/`` and the
read-only inputs (``patches.zarr``, ``splits_locked_v2.json``).

Discipline (design.md §5–§7):
- phase labels are DERIVED from store-native cycle bounds (percent of cycle);
  never called ground truth;
- anomaly calibration uses train-split fields only; injected-deviation
  evaluation uses dev-split fields only; no >=2024-01-01 claims;
- every table row carries n, seeds, protocol name.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
OUT_DIR = HERE / "outputs"
PATCH_STORE = REPO_ROOT / "data" / "processed" / "patches.zarr"
SPLITS_PATH = REPO_ROOT / "data" / "processed" / "splits_locked_v2.json"
EMBED_COLS = [f"e{i:03d}" for i in range(128)]
PATCH_CHANNEL_ORDER = ["ndvi", "blue", "green", "red", "nir", "swir1", "swir2", "lst"]
SEEDS = [13, 42, 2026]

# Derived phase classes by fraction of the cycle elapsed since its start
# (the cycle bounds are harvest-to-harvest in the v1 store).
PHASE_EDGES = [0.0, 0.25, 0.5, 0.80, 1.0001]
PHASE_NAMES = ["establishment", "vegetative", "grand_growth", "maturation"]


def load_splits() -> dict:
    with open(SPLITS_PATH, encoding="utf-8") as f:
        return json.load(f)


def split_field_sets() -> tuple[set[int], set[int]]:
    """(train_fields, dev_fields) restricted to fields present in patches.zarr."""
    import zarr

    store = zarr.open_group(str(PATCH_STORE), mode="r")
    patch_fields = {int(f) for f in store["field_ids"][:]}
    sp = load_splits()
    train = {int(f) for f in sp["train_fields"]} & patch_fields
    dev = {int(f) for f in sp["dev_fields"]} & patch_fields
    return train, dev


def load_obs_embeddings(suffix: str = "") -> pd.DataFrame:
    """Observation-level embedding table (parquet)."""
    path = OUT_DIR / f"embeddings_v1_2_nano{suffix}.parquet"
    return pd.read_parquet(path)


def load_cycle_embeddings(suffix: str = "") -> pd.DataFrame:
    path = OUT_DIR / f"farmcycle_v1_2_nano{suffix}.parquet"
    return pd.read_parquet(path)


def add_derived_columns(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
    """Phase class + year/month from store-native metadata (no external joins).

    Cycle bounds come from the farm-cycle table when the obs table lacks them.
    """
    df = df.copy()
    if "cycle_start_day" not in df.columns:
        cyc = load_cycle_embeddings(suffix)
        df = df.merge(
            cyc[["field_id", "cycle_index", "cycle_start_day", "cycle_end_day"]],
            on=["field_id", "cycle_index"], how="left")
    if "phase" not in df.columns:
        cyc_len = (df["cycle_end_day"] - df["cycle_start_day"]).clip(lower=1)
        frac = ((df["day_index"] - df["cycle_start_day"]) / cyc_len).clip(0, 1)
        df["cycle_frac"] = frac
        df["phase"] = pd.cut(
            frac, bins=PHASE_EDGES, labels=PHASE_NAMES, right=False, ordered=True)
    return df


def load_raw_bin_features(force: bool = False) -> pd.DataFrame:
    """Per-(field, cycle, bin) raw channel means over QA-valid in-field pixels.

    This is the information base of the classical spectral baselines (B0/B1):
    8 features per bin from the SAME pixels the embeddings see. Cached to
    outputs/raw_bin_features.parquet.
    """
    cache = OUT_DIR / "raw_bin_features.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    import zarr

    store = zarr.open_group(str(PATCH_STORE), mode="r")
    field_ids = store["field_ids"][:]
    rows = []
    for fi in range(len(field_ids)):
        fid = int(field_ids[fi])
        in_field = store["in_field"][fi, 0].astype(bool)
        cycle_valid = store["cycle_valid"][fi].astype(bool)
        for ci in range(8):
            if not cycle_valid[ci]:
                continue
            values = store["values"][fi, ci].astype(np.float64)  # (36,8,H,W)
            pv = store["pixel_valid"][fi, ci, :, 0].astype(bool)
            n_obs = store["n_obs"][fi, ci]
            for b in range(36):
                if n_obs[b] <= 0:
                    continue
                m = pv[b] & in_field
                if not m.any():
                    continue
                means = np.array([np.nanmean(np.where(m, values[b, c], np.nan))
                                  for c in range(8)])
                if not np.isfinite(means).all():
                    continue
                rows.append(dict(
                    field_id=fid, cycle_index=ci, bin_index=b,
                    start_day=int(store["cycle_start_day"][fi, ci]),
                    end_day=int(store["cycle_end_day"][fi, ci]),
                    day_index=int(oe_bin_day(int(store["cycle_start_day"][fi, ci]), b)),
                    **{PATCH_CHANNEL_ORDER[c]: float(means[c]) for c in range(8)},
                ))
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def oe_bin_day(cycle_start_day: int, bin_index: int) -> int:
    """Bin-center day index (mirrors oe_adapter.bin_center_day, import-free)."""
    return cycle_start_day + bin_index * 15 + 7


def robust_z(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Robust per-dimension z against a reference sample (median/MAD).

    x: (n, d); ref: (m, d). Returns (n, d).
    """
    med = np.median(ref, axis=0)
    mad = np.median(np.abs(ref - med), axis=0)
    scale = 1.4826 * mad
    scale = np.where(scale > 1e-12, scale, 1e-12)
    return (x - med) / scale


def bootstrap_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 2026,
                 stat=np.median) -> tuple[float, float, float]:
    """(point, ci_low, ci_high) for `stat` over bootstrap resamples."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    point = float(stat(values))
    boots = np.empty(n_boot)
    for i in range(n_boot):
        boots[i] = stat(values[rng.integers(0, len(values), size=len(values))])
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def write_table(name: str, payload: dict) -> Path:
    """Persist a JSON results table with provenance boilerplate."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "tables" / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("generated_by", "experiments/olmoearth_eval")
    payload.setdefault("validation_ladder", "L1-L2 only")
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path

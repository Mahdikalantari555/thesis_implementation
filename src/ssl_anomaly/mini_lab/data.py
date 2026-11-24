"""Data loading, configurable representations and leak-free splits for the mini lab."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import zarr

from .config import RunConfig

DEFAULT_ROOT = Path("D:/RS/Thesis/Data/Processed/mini_research_dataset")


@dataclass
class MiniSample:
    farm_id: str
    unit_id: str
    dates: tuple[str, ...]
    values: np.ndarray  # (T, F) raw physical/derived features, pre-normalization
    observed: np.ndarray  # (T, F) bool, finite source values
    delta_t: np.ndarray  # (T,) days since previous acquisition


@dataclass
class MiniDataset:
    samples: list[MiniSample] = field(default_factory=list)
    feature_names: list[str] = field(default_factory=list)
    valid_patch_counts: dict[str, dict[str, int]] = field(default_factory=dict)
    source_manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return len(self.samples)


@dataclass
class Split:
    name: str
    x: np.ndarray  # (N, T, F)
    observed: np.ndarray  # (N, T, F)
    delta_t: np.ndarray  # (N, T)
    farm_ids: list[str]
    date_ranges: list[tuple[str, str]]
    sha256: str

    @property
    def n(self) -> int:
        return int(self.x.shape[0])


def _farm_groups(store_root: Path) -> list[tuple[str, Any]]:
    store = store_root / "satellite.zarr"
    out = []
    for d in sorted(p for p in store.iterdir() if p.is_dir() and (p / ".zgroup").exists()):
        out.append((d.name, zarr.open_group(str(d), mode="r")))
    return out


def _weather_features(root: Path, group_name: str) -> tuple[pd.DataFrame, list[str]]:
    table = pd.read_parquet(root / "weather" / "weather_aligned_mini.parquet")
    features = json.loads((root / "weather" / "weather_features.json").read_text(encoding="utf-8"))
    cols = features["groups"][group_name]
    table["date_key"] = table["acquisition_date"].astype(str).str.slice(0, 10)
    table = table.set_index("date_key")
    return table, list(cols)


def _field_features(
    group: Any, dates: list[str], with_indices: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cube = np.asarray(group["data"][:], dtype=np.float32)
    valid = np.asarray(group["pixel_valid"][:], dtype=bool)
    in_field = np.asarray(group["in_field"][:], dtype=bool)
    band_order = list(group.attrs["band_order"])
    names: list[str] = []
    cols: list[np.ndarray] = []
    for bi, band in enumerate(band_order):
        vals = []
        for di in range(len(dates)):
            m = valid[di] & in_field & np.isfinite(cube[di, bi])
            vals.append(float(np.median(cube[di, bi][m])) if m.any() else np.nan)
        cols.append(np.asarray(vals, dtype=np.float32))
        names.append(band)
    if with_indices:
        from ssl_anomaly.features import compute_index

        mapping = {b: cube[:, bi, :, :] for bi, b in enumerate(band_order)}
        for idx_name, required in (
            ("ndvi", ("red", "nir")),
            ("savi", ("red", "nir")),
            ("evi", ("blue", "red", "nir")),
            ("ndmi", ("nir", "swir1")),
            ("ndwi", ("green", "nir")),
            ("nbr", ("nir", "swir2")),
        ):
            arr = compute_index(idx_name, {b: mapping[b] for b in required})
            vals = []
            for di in range(len(dates)):
                m = valid[di] & in_field & np.isfinite(arr[di])
                vals.append(float(np.median(arr[di][m])) if m.any() else np.nan)
            cols.append(np.asarray(vals, dtype=np.float32))
            names.append(idx_name)
    matrix = np.stack(cols, axis=1)
    return matrix, np.isfinite(matrix), names


def _pixel_features(
    group: Any, dates: list[str], max_pixels: int, with_indices: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    cube = np.asarray(group["data"][:], dtype=np.float32)
    valid = np.asarray(group["pixel_valid"][:], dtype=bool)
    in_field = np.asarray(group["in_field"][:], dtype=bool)
    coords = np.argwhere(in_field)
    if len(coords) > max_pixels:
        coords = coords[:max_pixels]
    series, names = [], []
    for bi, band in enumerate(group.attrs["band_order"]):
        vals = np.full((len(coords), len(dates)), np.nan, dtype=np.float32)
        for ci, (r, c) in enumerate(coords):
            for di in range(len(dates)):
                if valid[di, r, c] and np.isfinite(cube[di, bi, r, c]):
                    vals[ci, di] = cube[di, bi, r, c]
        series.append(vals)
        names.append(band)
    if with_indices:
        from ssl_anomaly.features import compute_index

        for idx_name, required in (
            ("ndvi", ("red", "nir")),
            ("savi", ("red", "nir")),
        ):
            bi = {b: i for i, b in enumerate(group.attrs["band_order"])}
            r, n = series[bi["red"]], series[bi["nir"]]
            if idx_name == "ndvi":
                v = np.where(np.isfinite(n + r) & (np.abs(n + r) > 1e-8), (n - r) / (n + r + 1e-8), np.nan)
            else:
                v = np.where(
                    np.isfinite(n + r) & (np.abs(n + r + 0.5) > 1e-8),
                    1.5 * (n - r) / (n + r + 0.5 + 1e-8),
                    np.nan,
                )
            series.append(v.astype(np.float32))
            names.append(idx_name)
    values = np.stack(series, axis=2)  # (P, D, F)
    return values, np.isfinite(values), names


def _patch_valid_fraction(valid: np.ndarray, in_field: np.ndarray, size: int):
    """Return (D, Hp, Wp) valid fraction per patch/date, vectorized."""
    from numpy.lib.stride_tricks import sliding_window_view

    imask = sliding_window_view(in_field, (size, size))  # (Hp, Wp, s, s)
    vmask = sliding_window_view(valid, (size, size), axis=(1, 2))  # (D, Hp, Wp, s, s)
    denom = imask.sum(axis=(-1, -2)).astype(np.float32)
    num = (vmask & imask).sum(axis=(-1, -2)).astype(np.float32)
    return np.where(denom > 0, num / np.maximum(denom, 1.0), 0.0), denom


def _patch_counts(group: Any, sizes: tuple[int, ...], min_frac: float) -> dict[str, int]:
    in_field = np.asarray(group["in_field"][:], dtype=bool)
    valid = np.asarray(group["pixel_valid"][:], dtype=bool)
    out: dict[str, int] = {}
    for size in sizes:
        frac, denom = _patch_valid_fraction(valid, in_field, size)
        # A patch is usable when it overlaps the in-field mask and its mean
        # valid fraction over dates reaches the threshold. Per-date gaps are
        # kept as absent (observed=False), never filled.
        out[str(size)] = int(((frac.mean(axis=0) >= min_frac) & (denom > 0)).sum())
    return out


def _patch_features(
    group: Any, dates: list[str], size: int, min_frac: float, with_indices: bool
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Vectorized per-patch median over valid in-field pixels for each date/band."""
    from numpy.lib.stride_tricks import sliding_window_view

    cube = np.asarray(group["data"][:], dtype=np.float32)
    valid = np.asarray(group["pixel_valid"][:], dtype=bool)
    in_field = np.asarray(group["in_field"][:], dtype=bool)
    frac, denom = _patch_valid_fraction(valid, in_field, size)
    selected = (frac.mean(axis=0) >= min_frac) & (denom > 0)
    if not selected.any():
        raise RuntimeError(f"No valid {size}x{size} patch for farm {group.attrs['farm_id']}")

    windows = sliding_window_view(cube, (size, size), axis=(2, 3))  # (D,K,Hp,Wp,s,s)
    vmask = sliding_window_view(valid, (size, size), axis=(1, 2))  # (D,Hp,Wp,s,s)
    imask = sliding_window_view(in_field, (size, size))  # (Hp,Wp,s,s)
    mask = vmask[:, None, ...] & imask[None, None, ...] & np.isfinite(windows)
    values_masked = np.where(mask, windows, np.nan)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(values_masked, axis=(-1, -2))  # (D,K,Hp,Wp)
    median = median[:, :, selected]  # (D,K,P)
    names = list(group.attrs["band_order"])
    cols = [median[:, bi, :] for bi in range(len(names))]  # each (D,P)
    if with_indices:
        red, nir = cols[names.index("red")], cols[names.index("nir")]
        den = nir + red + 1e-8
        ndvi = np.where(np.abs(den) > 1e-8, (nir - red) / den, np.nan)
        cols.append(ndvi)
        names.append("ndvi")
    out = np.stack(cols, axis=2).transpose(1, 0, 2)  # (P,D,F)
    return out.astype(np.float32), np.isfinite(out), names


def _temporal_units(
    values: np.ndarray, observed: np.ndarray, dates: list[str], delta_t: np.ndarray, cfg: RunConfig
) -> list[tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]]:
    """Split a unit's series into temporal samples; Δt is always preserved."""
    rep = cfg.representation
    n = len(dates)
    if rep.temporal == "full":
        idx = [np.arange(n)]
    elif rep.temporal == "sliding":
        idx = [
            np.arange(s, min(n, s + rep.window_length))
            for s in range(0, max(1, n - rep.window_length + 1), max(1, rep.stride))
        ]
    elif rep.temporal == "sampling":
        idx = [np.arange(0, n, max(1, rep.sample_step))]
    elif rep.temporal in {"seasonal", "variable"}:
        cyc_path = cfg.store_root / "phenology" / "cycles_mini.parquet"
        if not cyc_path.exists():
            raise RuntimeError(f"Temporal repr {rep.temporal} requires {cyc_path}")
        idx = []
        for _, row in pd.read_parquet(cyc_path).iterrows():
            sel = np.array(
                [i for i, d in enumerate(dates) if str(row["start_date"]) <= d <= str(row["end_date"])]
            )
            if sel.size >= 2:
                idx.append(sel)
    else:
        raise ValueError(f"Unknown temporal representation {rep.temporal}")
    out = []
    for sel in idx:
        if sel.size < 2:
            continue
        out.append(
            (
                tuple(dates[i] for i in sel),
                values[sel],
                observed[sel],
                delta_t[sel],
            )
        )
    return out


def build_dataset(cfg: RunConfig, store_root: Path | None = None) -> MiniDataset:
    root = store_root or cfg.store_root
    weather, weather_cols = _weather_features(root, cfg.representation.weather_group)
    ds = MiniDataset()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ds.source_manifest = {
        "mini_manifest_sha256": hashlib.sha256(
            (root / "manifest.json").read_bytes()
        ).hexdigest(),
        "git_sha": manifest.get("git_sha"),
        "date_list_sha256": manifest.get("date_list_sha256"),
        "n_dates": manifest.get("n_dates"),
        "first_date": manifest.get("first_date"),
        "last_date": manifest.get("last_date"),
        "weather_group": cfg.representation.weather_group,
    }
    for farm_id, group in _farm_groups(root):
        dates = [str(d) for d in group["dates"][:]]
        day_nums = np.array([(np.datetime64(d, "D") - np.datetime64("2013-03-30", "D")).astype(int) for d in dates])
        delta_t = np.diff(day_nums, prepend=day_nums[0]).astype(np.float32)

        if cfg.representation.spatial == "full_field":
            values, observed, names = _field_features(group, dates, cfg.representation.include_indices)
            units = [(tuple(dates), values, observed, delta_t)]
        elif cfg.representation.spatial == "pixel":
            values, observed, names = _pixel_features(
                group, dates, cfg.representation.max_pixels, cfg.representation.include_indices
            )
            units = [
                (tuple(dates), values[i], observed[i], delta_t)
                for i in range(values.shape[0])
            ]
        elif cfg.representation.spatial == "patch":
            counts = _patch_counts(group, cfg.representation.patch_sizes, cfg.representation.min_valid_fraction)
            ds.valid_patch_counts[farm_id] = counts
            size = cfg.representation.patch_sizes[0] if cfg.representation.patch_sizes else 3
            values, observed, names = _patch_features(
                group, dates, size, cfg.representation.min_valid_fraction, cfg.representation.include_indices
            )
            units = [
                (tuple(dates), values[i], observed[i], delta_t)
                for i in range(values.shape[0])
            ]
        else:
            raise ValueError(cfg.representation.spatial)

        wvalues = weather.reindex(dates)[weather_cols].to_numpy(dtype=np.float32)
        wnames = list(weather_cols)
        if cfg.representation.include_delta_t:
            delta_col = delta_t.reshape(-1, 1).astype(np.float32)
            wvalues = np.concatenate([wvalues, delta_col], axis=1)
            wnames = wnames + ["delta_t_days"]

        for ui, (udates, uvalues, uobserved, udelta) in enumerate(units):
            if not ds.feature_names:
                ds.feature_names = names + wnames
            if len(udates) != uvalues.shape[0]:
                raise RuntimeError("unit/date length mismatch")
            joined = np.concatenate([uvalues, wvalues], axis=1)
            joined_obs = np.concatenate([uobserved, np.isfinite(wvalues)], axis=1)
            for si, (sdates, svalues, sobserved, sdelta) in enumerate(
                _temporal_units(joined, joined_obs, udates, udelta, cfg)
            ):
                ds.samples.append(
                    MiniSample(
                        farm_id=farm_id,
                        unit_id=f"{farm_id}:{ui}:{si}",
                        dates=sdates,
                        values=svalues.astype(np.float32),
                        observed=sobserved.astype(bool),
                        delta_t=sdelta.astype(np.float32),
                    )
                )
    return ds


def _hash_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def split_datasets(ds: MiniDataset, cfg: RunConfig) -> list[tuple[str, Split, Split]]:
    """Create protocol splits. Normalization is fitted on train samples only."""
    proto = cfg.protocol
    if proto.name == "single_farm":
        rotations = [("single", tuple(proto.train_farms), tuple(proto.test_farms))]
    elif proto.name == "lodo":
        farms = sorted({s.farm_id for s in ds.samples})
        rotations = [(f"lodo_holdout_{f}", tuple(x for x in farms if x != f), (f,)) for f in farms]
    elif proto.name == "intra_farm_temporal":
        rotations = [(f"temporal_{f}", (f,), (f,)) for f in sorted({s.farm_id for s in ds.samples})]
    else:
        raise ValueError(proto.name)

    out = []
    for name, train_farms, test_farms in rotations:
        train_samples, test_samples = [], []
        for s in ds.samples:
            is_train_farm = s.farm_id in train_farms
            is_test_farm = s.farm_id in test_farms
            if proto.name == "intra_farm_temporal":
                tr = [i for i, d in enumerate(s.dates) if d < proto.holdout_boundary]
                te = [i for i, d in enumerate(s.dates) if d >= proto.holdout_boundary]
                for idx, bucket in ((tr, train_samples), (te, test_samples)):
                    if len(idx) >= 2:
                        bucket.append(
                            MiniSample(
                                s.farm_id,
                                s.unit_id,
                                tuple(s.dates[i] for i in idx),
                                s.values[idx],
                                s.observed[idx],
                                s.delta_t[idx],
                            )
                        )
            elif is_train_farm:
                train_samples.append(s)
            elif is_test_farm:
                test_samples.append(s)
        if not train_samples or not test_samples:
            raise RuntimeError(f"Empty split for protocol {name}: train={len(train_samples)} test={len(test_samples)}")

        # Temporal protocols can truncate windows to different lengths. Pad to
        # a common T with absent values (observed=False), never with fabricated
        # measurements, so batching is possible without leaking or imputing.
        max_t = max(s.values.shape[0] for s in train_samples + test_samples)

        def pad(s: MiniSample) -> MiniSample:
            t, f = s.values.shape
            if t == max_t:
                return s
            values = np.full((max_t, f), np.nan, dtype=np.float32)
            observed = np.zeros((max_t, f), dtype=bool)
            values[:t] = s.values
            observed[:t] = s.observed
            delta = np.zeros(max_t, dtype=np.float32)
            delta[:t] = s.delta_t
            dates = tuple(s.dates) + ("",) * (max_t - t)
            return MiniSample(s.farm_id, s.unit_id, dates, values, observed, delta)

        train_samples = [pad(s) for s in train_samples]
        test_samples = [pad(s) for s in test_samples]

        train_raw = np.stack([s.values for s in train_samples])
        finite = np.isfinite(train_raw) & np.stack([s.observed for s in train_samples])
        mu = np.nanmean(np.where(finite, train_raw, np.nan), axis=(0, 1))
        sd = np.nanstd(np.where(finite, train_raw, np.nan), axis=(0, 1))
        mu = np.where(np.isfinite(mu), mu, 0.0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)

        def to_split(samples: list[MiniSample]) -> Split:
            x = np.stack([(s.values - mu) / sd for s in samples]).astype(np.float32)
            obs = np.stack([s.observed for s in samples]).astype(bool)
            # Mask-aware: imputed zeros are marked observed=False and counted.
            x = np.where(obs, x, 0.0).astype(np.float32)
            dt = np.stack([s.delta_t for s in samples]).astype(np.float32)
            return Split(
                name=name,
                x=x,
                observed=obs,
                delta_t=dt,
                farm_ids=[s.farm_id for s in samples],
                date_ranges=[(s.dates[0], s.dates[-1]) for s in samples],
                sha256=_hash_arrays(x, obs, dt),
            )

        out.append((name, to_split(train_samples), to_split(test_samples)))
    return out

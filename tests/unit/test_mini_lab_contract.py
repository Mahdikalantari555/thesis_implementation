"""Task 0A.4 — min_obs window exclusion with counts, contract wiring end-to-end.

Builds a tiny synthetic mini store (zarr + weather parquet + manifest) so the
``build_dataset`` path itself is exercised: band-order assertion, feature
manifest validation on unknown columns, sensor_id absence recording and the
``min_obs`` exclusion rule with its exclusion counts.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

zarr = pytest.importorskip("zarr")

from ssl_anomaly.io.data_contract import CANONICAL_BAND_ORDER, ContractViolation
from ssl_anomaly.mini_lab.config import RunConfig
from ssl_anomaly.mini_lab.data import build_dataset

DATES = [f"2021-{m:02d}-{d:02d}" for m, d in (
    (1, 1), (1, 6), (1, 11), (1, 16), (1, 21), (1, 26), (1, 31),
    (2, 5), (2, 10), (2, 15),
)]


def make_store(tmp_path, *, band_order=None, weather_cols=("temperature_mean",)):
    root = tmp_path / "store"
    (root / "weather").mkdir(parents=True)
    (root / "phenology").mkdir(parents=True)
    (root / "manifest.json").write_text(json.dumps({
        "git_sha": "test", "date_list_sha256": "test", "n_dates": len(DATES),
        "first_date": DATES[0], "last_date": DATES[-1],
    }))
    weather = pd.DataFrame({
        "acquisition_date": pd.to_datetime(DATES),
        "temperature_mean": np.linspace(10, 25, len(DATES)),
        "precipitation_sum": np.full(len(DATES), 0.5),
    })
    weather.to_parquet(root / "weather" / "weather_aligned_mini.parquet")
    (root / "weather" / "weather_features.json").write_text(json.dumps({
        "schema": "mini-weather-features/v1",
        "groups": {
            "weather_core": list(weather_cols),
            "weather_plus_derived": list(weather_cols),
            "spectral_only": [],
        },
    }))
    store = root / "satellite.zarr"
    for farm in ("FA", "FB"):
        g = zarr.group(str(store / farm), zarr_format=2)
        d, h, w = len(DATES), 3, 3
        g.create_array("dates", data=np.array(DATES, dtype="<U10"))
        rng = np.random.default_rng(1)
        data = rng.normal(0.4, 0.05, size=(d, 7, h, w)).astype(np.float32)
        data[:, 6] = rng.normal(300.0, 2.0, size=(d, h, w)).astype(np.float32)  # lst in K
        g.create_array("data", data=data)
        valid = np.ones((d, h, w), dtype=bool)
        g.create_array("pixel_valid", data=valid.astype("uint8"))
        g.create_array("in_field", data=np.ones((h, w), dtype="uint8"))
        g.attrs["band_order"] = list(band_order or CANONICAL_BAND_ORDER)
        g.attrs["farm_id"] = farm
    return root


def make_cfg(root, **rep_kw) -> RunConfig:
    cfg = RunConfig(name="contract_fixture")
    cfg.store_root = root
    cfg.representation.weather_group = "weather_core"
    cfg.representation.temporal = "sliding"
    cfg.representation.window_length = 4
    cfg.representation.stride = 4
    for k, v in rep_kw.items():
        setattr(cfg.representation, k, v)
    return cfg


def _knock_out_dates(group, date_idx: list[int]) -> None:
    """Make whole dates unobserved for core features (cloud/fill analogue)."""
    data = group["data"][:]
    valid = group["pixel_valid"][:]
    for di in date_idx:
        data[di] = np.nan
        valid[di] = False
    group["data"][:] = data
    group["pixel_valid"][:] = valid


class TestMinObsExclusion:
    def test_low_obs_windows_excluded_and_counted(self, tmp_path):
        root = make_store(tmp_path)
        g = zarr.open_group(str(root / "satellite.zarr" / "FA"), mode="r+", zarr_format=2)
        _knock_out_dates(g, [0, 1, 2])  # first window (dates 0-3) loses 3/4 obs
        cfg = make_cfg(root, min_obs=2)
        ds = build_dataset(cfg, store_root=root)
        # windows of 4 with stride 4 over 10 dates: [0-3],[4-7],[8-9(kept if >=2)]
        # window 0 has only 1 valid obs -> excluded and counted
        assert ds.min_obs_excluded, "exclusion counts must be recorded"
        assert sum(ds.min_obs_excluded.values()) >= 1
        # the surviving windows all carry >= min_obs valid observations
        for s in ds.samples:
            core = s.observed[:, :7]
            assert int(core.any(axis=1).sum()) >= cfg.representation.min_obs

    def test_min_obs_one_keeps_everything(self, tmp_path):
        root = make_store(tmp_path)
        g = zarr.open_group(str(root / "satellite.zarr" / "FA"), mode="r+", zarr_format=2)
        _knock_out_dates(g, [0, 1, 2])
        cfg = make_cfg(root, min_obs=1)
        ds = build_dataset(cfg, store_root=root)
        assert sum(ds.min_obs_excluded.values()) == 0

    def test_exclusion_surfaces_in_manifest(self, tmp_path):
        root = make_store(tmp_path)
        g = zarr.open_group(str(root / "satellite.zarr" / "FA"), mode="r+", zarr_format=2)
        _knock_out_dates(g, [0, 1, 2])
        cfg = make_cfg(root, min_obs=2)
        ds = build_dataset(cfg, store_root=root)
        assert ds.source_manifest["min_obs"] == 2
        assert ds.source_manifest["min_obs_excluded_total"] >= 1


class TestContractWiring:
    def test_wrong_band_order_fails_closed(self, tmp_path):
        root = make_store(tmp_path, band_order=["red", "green", "blue", "nir", "swir1", "swir2", "lst"])
        cfg = make_cfg(root)
        with pytest.raises(ContractViolation):
            build_dataset(cfg, store_root=root)

    def test_unknown_weather_feature_refuses_training(self, tmp_path):
        root = make_store(tmp_path, weather_cols=("mystery_var",))
        cfg = make_cfg(root)
        with pytest.raises(KeyError):
            build_dataset(cfg, store_root=root)

    def test_sensor_id_absence_recorded_not_imputed(self, tmp_path):
        root = make_store(tmp_path)
        cfg = make_cfg(root)
        ds = build_dataset(cfg, store_root=root)
        assert ds.sensor_ids["FA"] is None
        assert "absent" in ds.source_manifest["sensor_id"]["FA"]

    def test_sensor_id_present_is_read(self, tmp_path):
        root = make_store(tmp_path)
        for farm in ("FA", "FB"):
            g = zarr.open_group(str(root / "satellite.zarr" / farm), mode="r+", zarr_format=2)
            g.create_array("sensor_id", data=np.zeros(len(DATES), dtype="int8"))
            g.attrs["sensor_id_legend"] = {0: "L8", 1: "L9"}
        cfg = make_cfg(root)
        ds = build_dataset(cfg, store_root=root)
        assert ds.sensor_ids["FA"] is not None
        assert ds.sensor_ids["FA"].shape == (len(DATES),)
        assert set(np.unique(ds.sensor_ids["FA"]).tolist()) <= {0, 1}

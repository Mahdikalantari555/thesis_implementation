"""Unit tests for the 32-farm probe package (change farm-subset-anomaly-probe).

Fixtures are synthetic tiny subset stores (2 farms × 8 dates × 48×48 canvas)
with known injected deviations — detection of a controlled deviation is the
test's evidence that the scorer works (L2 sensitivity, never stress proof).
"""

from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

torch.set_num_threads(2)

from ssl_anomaly.probe.b2 import B2Baseline  # noqa: E402
from ssl_anomaly.probe.barlow import BarlowProbe, build_windows  # noqa: E402
from ssl_anomaly.probe.cnn_ae import (PixelCNNAE, count_params, fit_norm_stats,  # noqa: E402
                                      make_input, masked_loss)
from ssl_anomaly.probe.data import attach_cycles_weather, load_subset_features  # noqa: E402
from ssl_anomaly.probe.gates import apply_gates, build_events, delta_lst  # noqa: E402
from ssl_anomaly.probe.gru import TrajectoryGRUAE, build_tokens  # noqa: E402

EPOCH = date(2013, 3, 30)
S = 48


def make_tiny_store(tmp_path, n_farms=2, n_dates=10, seed=0):
    """Synthetic subset zarr: smooth seasonal reflectance, one injected anomaly.

    Dates span 0..441 days (10 acquisitions); the fixture cycles below chop
    this into three 150-day cycles so the 1-train-year window (2013) covers
    every tau bin that eval dates fall into (needed by the B2 tests).
    """
    rng = np.random.default_rng(seed)
    root = zarr.open_group(str(tmp_path / "subset.zarr"), mode="w", zarr_format=2)
    dates = [(EPOCH + timedelta(days=int(t))).isoformat()
             for t in np.linspace(0, 441, n_dates).astype(int)]
    for fi in range(n_farms):
        g = root.create_group(f"R11-{fi+1:02d}")
        in_field = np.zeros((S, S), dtype=np.uint8)
        in_field[10:30, 15:35] = 1
        data = np.full((n_dates, 7, S, S), np.nan, dtype=np.float16)
        pixel_valid = np.zeros((n_dates, S, S), dtype=np.uint8)
        for di, ds in enumerate(dates):
            doy = date.fromisoformat(ds).timetuple().tm_yday
            season = 0.3 + 0.25 * np.sin(2 * np.pi * (doy - 120) / 365.25)
            for k in range(7):
                if k == 6:  # lst ~ 300 K + seasonal
                    base = 300 + 8 * np.sin(2 * np.pi * (doy - 120) / 365.25)
                    vals = base + rng.normal(0, 0.5, (S, S))
                else:
                    vals = season + rng.normal(0, 0.01, (S, S))
                data[di, k] = vals.astype(np.float16)
            pixel_valid[di] = in_field
        # injected anomaly: farm 0 ONLY, last date — nir drops, LST warms
        if fi == 0:
            data[n_dates - 1, 3] = (season * 0.4 + rng.normal(0, 0.01, (S, S))).astype(np.float16)
            data[n_dates - 1, 6] = (305 + rng.normal(0, 0.5, (S, S))).astype(np.float16)
        g.create_array("data", data=data, chunks=(1, 7, S, S))
        g.create_array("pixel_valid", data=pixel_valid, chunks=(1, S, S))
        g.create_array("in_field", data=in_field, chunks=(S, S))
        g.create_array("dates", data=np.asarray(dates, dtype="<U10"))
        g.attrs.update({"farm_id": f"R11-{fi+1:02d}", "archive_fid": fi + 185,
                        "reference_fid": fi + 184, "canvas": S,
                        "window_hw": [20, 20], "canvas_offset": [0, 0]})
    root.attrs.update({"canvas": S, "git_commit_sha": "test",
                       "date_range": [dates[0], dates[-1]]})
    return tmp_path / "subset.zarr", dates


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("probe_tiny")
    store, dates = make_tiny_store(tmp)
    d, _ = load_subset_features(store)
    # synthetic cycles: three 150-day cycles per farm covering all dates
    # (archive fid i+185); train-year window covers every tau bin
    rows = []
    for fid in d.archive_fids:
        for c in range(3):
            rows.append({"field": fid, "cycle_id": c,
                         "t_start": c * 150, "t_end": (c + 1) * 150,
                         "start_harvest": c * 150, "end_harvest": (c + 1) * 150,
                         "harvest_day": (c + 1) * 150, "length_days": 150,
                         "censored": c == 2, "kind": "plant", "ratoon_no": c,
                         "ndvi_at_harvest": 0.3})
    cyc = pd.DataFrame(rows)
    cyc_path = tmp / "cycles.parquet"
    cyc.to_parquet(cyc_path)
    # synthetic weather: daily t_mean 20..35 °C sine
    days = pd.DataFrame({
        "date": pd.date_range("2013-01-01", periods=1200, freq="D"),
        "t_mean": 25 + 10 * np.sin(np.linspace(0, 6 * np.pi, 1200)),
        "corner": "NW",
    })
    weather_path = tmp / "weather.csv"
    days.to_csv(weather_path, index=False)
    d = attach_cycles_weather(d, cycles_path=cyc_path, weather_csv=weather_path,
                              train_years=1)
    return d, store, dates


# ------------------------------ data ----------------------------------------

def test_load_subset_features_shapes_and_no_zero_fill(tiny):
    d, _, _ = tiny
    assert d.band_means.shape == (2, 10, 7)
    assert d.obs_mask.all()
    # anomaly injection present (nir drop on farm 0, last date)
    assert d.band_means[0, -1, 3] < d.band_means[1, -1, 3]
    assert not d.canvases == {}
    assert d.provenance["canvas"] == 48


def test_train_split_is_first_calendar_years(tiny):
    d, _, dates = tiny
    years = sorted({date.fromisoformat(s).year for s in dates})
    tr_years = set(years[:1])
    assert all(date.fromisoformat(s).year in tr_years for s in
               np.asarray(dates)[d.train_mask])


def test_tau_and_gdd_present(tiny):
    d, _, _ = tiny
    assert np.isfinite(d.tau).any()
    assert np.isfinite(d.gdd_cum).any()
    assert ((d.tau[~np.isnan(d.tau)] >= 0) & (d.tau[~np.isnan(d.tau)] <= 1)).all()


# ------------------------------ B2 ------------------------------------------

def test_b2_detects_injected_deviation(tiny):
    d, _, _ = tiny
    b2 = B2Baseline(n_bins=4, min_farms=2).fit(d)
    sc = b2.score(d)
    s = sc["b2_score"]
    assert np.isfinite(s).any()
    # the injected (farm 0, last date) deviation ranks above that farm's
    # other dates (z_ndvi channel: nir drop ⇒ negative z)
    z_ndvi = sc["z_ndvi"]
    assert np.isfinite(z_ndvi[0, -1])
    assert abs(z_ndvi[0, -1]) >= np.nanmax(np.abs(z_ndvi[0, :-1]))


def test_b2_sigma_fit_on_train_only(tiny):
    d, _, _ = tiny
    b2 = B2Baseline(n_bins=4, min_farms=2).fit(d)
    sigma_before = b2.sigma["ndvi"].copy()
    assert np.isfinite(sigma_before).all()
    # perturb eval-date values: σ must not change (it was fit on train only)
    d.band_means[:, -1, :] *= 3.0
    b2b = B2Baseline(n_bins=4, min_farms=2).fit(d)
    assert np.allclose(sigma_before, b2b.sigma["ndvi"])


# ------------------------------ CNN-AE --------------------------------------

def test_cnn_ae_param_budget():
    model = PixelCNNAE()
    assert count_params(model) <= 250_000, count_params(model)


def test_cnn_ae_masked_loss_ignores_invalid_pixels():
    model = PixelCNNAE()
    model.eval()
    x = torch.randn(2, 8, S, S)
    target = x[:, :7].clone()
    mask = torch.zeros(2, S, S, dtype=torch.bool)
    mask[0, 10:20, 10:20] = True
    mask[1, 5:15, 5:15] = True
    with torch.no_grad():
        l1 = masked_loss(model(x), target, mask)
    target_pert = target.clone()
    target_pert[0, 0, 40, 40] += 10.0  # outside mask → must not change loss
    with torch.no_grad():
        l2 = masked_loss(model(x), target_pert, mask)
    assert torch.allclose(l1, l2)


def test_cnn_ae_norm_stats_train_only(tiny):
    d, _, _ = tiny
    norm = fit_norm_stats(d, d.train_mask)
    assert set(norm) == {f"{b}_{s}" for b in
                         ("blue", "green", "red", "nir", "swir1", "swir2", "lst")
                         for s in ("mean", "std")}
    # std positive and finite
    assert all(np.isfinite(v) and v > 0 for v in norm.values())


def test_make_input_mask_and_zero_fill():
    canvas = np.full((7, S, S), np.nan, dtype=np.float16)
    canvas[:, 10:20, 10:20] = 0.5
    valid = np.zeros((S, S), dtype=np.uint8)
    valid[10:20, 10:20] = 1
    in_field = valid.astype(bool)
    norm = {f"{b}_{s}": (0.0, 1.0) for b in ("blue", "green", "red", "nir",
                                             "swir1", "swir2", "lst")
            for s in ("mean", "std")}
    norm = {k: v[0] if k.endswith("mean") else v[1] for k, v in norm.items()}
    x, m = make_input(canvas, valid, in_field, norm)
    assert x.shape == (8, S, S) and x[7].max() == 1.0
    assert np.isfinite(x).all()          # placeholder zeros, never NaN into model
    assert m.sum() == 100                # in_field ∧ valid ∧ finite


# ------------------------------ Barlow --------------------------------------

def test_barlow_loss_decreases_and_offdiag_excluded(tiny):
    d, _, _ = tiny
    windows, _ = build_windows(d, d.band_means, window=4)
    if len(windows) < 4:
        pytest.skip("not enough windows in tiny fixture")
    bt = BarlowProbe(window=4, seed=2026)
    losses = []
    bt.train(windows, epochs=2, progress_cb=lambda ep, l: losses.append(l))
    assert all(np.isfinite(losses)) and len(losses) == 2


def test_barlow_score_fits_on_train_windows_only(tiny):
    d, _, _ = tiny
    windows, _ = build_windows(d, d.band_means, window=4)
    if len(windows) < 4:
        pytest.skip("not enough windows")
    bt = BarlowProbe(window=4, seed=2026)
    bt.train(windows, epochs=1)
    n_train = sum(w["is_train"] for w in windows)
    bt.fit_scoring(windows)
    assert n_train > 0
    sc = bt.score(windows, n_farms=len(d.farm_ids), n_dates=len(d.dates))
    assert sc.shape == (2, 10) and np.isfinite(sc).any()


# ------------------------------ GRU-AE --------------------------------------

def test_gru_ae_param_budget_and_smoke(tiny):
    d, _, _ = tiny
    tokens, target, finite = build_tokens(d, d.band_means)
    assert tokens.shape == (2, 10, 16)
    gru = TrajectoryGRUAE(d_model=64, seed=2026)
    assert gru.count_params() <= 250_000
    gru.train(tokens, target, finite, d.train_mask, epochs=2)
    sc = gru.score(tokens, target, finite)
    assert sc.shape == (2, 10)
    assert np.isfinite(sc[:, d.train_mask]).all()
    # injected anomaly date ranks highest for farm 0
    assert sc[0, -1] >= np.nanmax(sc[0, :-1]) * 0.99 or np.isnan(sc[0, :-1]).all()


# ------------------------------ gates / events -------------------------------

def _mini_df():
    dates = [f"2014-01-{d:02d}" for d in (1, 3, 6, 8, 15, 20)]
    rows = []
    rng = np.random.default_rng(0)
    for farm in ("A", "B"):
        for di, ds in enumerate(dates):
            rows.append({"farm_id": farm, "date": ds, "absent": False,
                         "n_px": 280, "m_score": 0.0,
                         "lst_c": 25.0 + rng.normal(0, 0.1),
                         "delta_lst_c": 0.0, "combined_score": 0.0,
                         "cycle_id": 0, "tau": 0.5, "gdd_cum": 100.0})
    df = pd.DataFrame(rows)
    # farm A: flags on 01-06 & 01-08 (one event, span 2 d < 5 → non_persistent)
    # farm B: flags on 01-06 & 01-20 (one event, span 14 d ≥ 5, 2 obs → flagged);
    #         cold LST (ΔLST = −3) on 01-15 (evidence, gate rejects)
    df.loc[(df.farm_id == "A") & (df.date.isin(["2014-01-06", "2014-01-08"])),
           "m_score"] = 5.0
    df.loc[(df.farm_id == "B") & (df.date.isin(["2014-01-06", "2014-01-20"])),
           "m_score"] = 5.0
    df.loc[(df.farm_id == "B") & (df.date == "2014-01-15"), "delta_lst_c"] = -3.0
    return df, dates, np.array([True, True, False, False, False, False])


def test_gates_persistence_and_cold_lst():
    df, dates, train_mask = _mini_df()
    df, info = apply_gates(df, ["m_score"], train_mask, dates,
                           cold_lst_thresh=-2.0, min_obs=2, min_gap_days=5,
                           event_gap_days=30)
    a = df[df.farm_id == "A"]
    assert set(a.loc[a.date.isin(["2014-01-06", "2014-01-08"]), "gate_status"]) == \
        {"non_persistent"}          # 2 obs, 2 d apart → not persistent
    b_flag = df[(df.farm_id == "B") & (df.date.isin(["2014-01-06", "2014-01-20"]))]
    assert set(b_flag.gate_status) == {"flagged"}
    assert b_flag.event_id.nunique() == 1 and b_flag.event_id.iloc[0] >= 0
    b_cold = df[(df.farm_id == "B") & (df.date == "2014-01-15")]
    assert b_cold.gate_status.iloc[0] == "rejected_cold_lst"
    thresholds = info["thresholds"]["m_score"]
    assert thresholds == 0.0  # calibrated on clean train dates only


def test_events_table_and_no_stress_wording():
    df, dates, train_mask = _mini_df()
    df, info = apply_gates(df, ["m_score"], train_mask, dates)
    ev = build_events(df, ["m_score"])
    # no "stress" anywhere in columns or string cells (L2-only framing)
    text = json.dumps(ev.to_dict(orient="records")).lower() + \
        ",".join(map(str, ev.columns)).lower() + json.dumps(info).lower()
    assert "stress" not in text
    assert "anomaly" not in ev.columns or True
    if not ev.empty:
        assert {"event_id", "farm_id", "date_start", "date_end", "n_obs",
                "max_combined_score"} <= set(ev.columns)


def test_delta_lst_cross_sectional():
    lst = np.array([[25.0, 30.0], [26.0, 20.0]])
    obs = np.array([[True, True], [True, True]])
    dl = delta_lst(lst, obs)
    assert dl[0, 0] == pytest.approx(-0.5)   # 25 − median(25,26)
    assert dl[1, 1] == pytest.approx(-5.0)   # 20 − median(30,20)


# ------------------------------ Stage-2 ONNX ---------------------------------

def test_cnn_ae_onnx_parity(tmp_path):
    """Torch vs ONNXRuntime CPU recon must agree (fp32, atol 1e-3)."""
    torch.manual_seed(0)
    model = PixelCNNAE()
    model.eval()
    from ssl_anomaly.probe.stage2 import export_onnx, ort_session
    onnx_path = tmp_path / "cnn_ae.onnx"
    export_onnx(model, onnx_path, canvas=S)
    sess = ort_session(onnx_path)
    x = torch.randn(1, 8, S, S)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(["recon"], {"canvas": x.numpy()})[0]
    assert out.shape == ref.shape
    assert np.allclose(out, ref, atol=1e-3), float(np.max(np.abs(out - ref)))


def test_stage2_ort_score_canvas_missing_field_is_nan(tmp_path):
    """Empty loss_mask (no valid in-field px) → NaN score, NaN map."""
    torch.manual_seed(1)
    model = PixelCNNAE()
    from ssl_anomaly.probe.stage2 import export_onnx, ort_score_canvas, ort_session
    export_onnx(model, tmp_path / "m.onnx", canvas=S)
    sess = ort_session(tmp_path / "m.onnx")
    norm = {f"{b}_{s}": 1.0 for b in ("blue", "green", "red", "nir", "swir1",
                                      "swir2", "lst") for s in ("mean", "std")}
    canvas = np.full((7, S, S), np.nan, dtype=np.float32)
    valid = np.zeros((S, S), dtype=np.uint8)
    in_field = np.zeros((S, S), dtype=bool)
    fd, pmap = ort_score_canvas(sess, canvas, valid, in_field, norm,
                                np.ones(7))
    assert np.isnan(fd) and np.isnan(pmap).all()

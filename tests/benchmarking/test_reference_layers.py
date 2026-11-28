"""Tests for the reference-index layer, the validation protocols and the HPO fairness gate.

All three modules are the parts of the benchmark most able to produce a confident,
wrong number: an index fitted on the scored partition, a "farm holdout" that leaks
farms, a search that selects on the test split. Each test below is written to fail if
the guard is removed, not just if the arithmetic slips.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.benchmarking import hpo, indices, protocols

# --------------------------------------------------------------------------- fixture


@pytest.fixture()
def grid():
    """60 cycles × 20 farms × 16 bins of LST/NDVI/EVI, with anomalies only in the scored half.

    Layout mirrors the real archive: rows are cycles, ``farm`` is a column, every farm has
    both fit cycles and scored cycles (so a within-field envelope can be fitted honestly),
    and the anomaly is a hot, brown patch present **only** in the scored cycles.
    """
    n, T, n_farm = 60, 16, 20
    rng = np.random.default_rng(11)
    farm = np.arange(n) % n_farm
    late = np.arange(n) >= 40
    x = np.zeros((n, T, 3), dtype=np.float32)
    seas = np.sin(np.arange(T)[None, :] / (T - 1.0) * np.pi)              # 0 → 1 → 0
    # LST is anti-phase with greenness (a real canopy cools as it closes), so a hot bin is a
    # local excursion above the field's own seasonal curve rather than a seasonal peak; without
    # that alignment CWSI_LST would be measuring June, not the anomaly.
    x[:, :, 0] = (305.0 + 3.0 * (farm % 2)[:, None]
                  + 4.0 * (1.0 - seas) + 0.8 * np.sin(np.arange(T)[None, :] / 1.7)
                  + rng.normal(0, 1.0, (n, T)))
    x[:, :, 1] = 0.30 + 0.35 * (farm[:, None] % 7) / 6.0 + 0.20 * seas     # spread across farms
    x[:, :, 2] = 0.60 + 0.10 * (farm[:, None] % 5) / 4.0 + 0.15 * seas
    hot = np.zeros((n, T), dtype=bool)
    for i in np.flatnonzero(late):
        hot[i, 8:12] = True
    x[hot, 0] += 9.0                                    # hot canopy (dominates the 4 K seasonal swing)
    x[hot, 1] -= 0.18                                   # browning
    x[35:38, 4, 0] = np.nan                             # cloudy observations: NaN, not zero
    return {"x": x, "names": ("lst", "ndvi", "evi"), "hot": hot, "farm": farm, "late": late,
            "n": n, "T": T, "fit": np.arange(40), "sc": np.arange(40, 60),
            "weather": {"et0": np.full((n, T), 5.0, dtype=np.float32)}}


def _tvdi(grid, **kw):
    # ndvi_bins/min_support are smaller than the production default because this fixture
    # has 20 fields, not the archive's 400+ cycles: the code's refusal to bound a
    # near-empty NDVI bin is the behaviour under test, not an obstacle to it.
    kw.setdefault("ndvi_bins", 4)
    kw.setdefault("min_support", 3)
    return indices.tvdi(grid["x"], grid["names"], fit_idx=grid["fit"], scored_idx=grid["sc"], **kw)


# --------------------------------------------------------------------------- indices


def test_tvdi_and_within_field_cwsi_flag_the_hot_brown_patch(grid):
    tv = _tvdi(grid)
    cw = indices.cwsi_lst(grid["x"], grid["names"], fit_idx=grid["fit"], scored_idx=grid["sc"])
    assert tv.available and cw.available, (tv.fit_record, cw.fit_record)
    for idx in (tv, cw):
        s = idx.scores[grid["sc"]]
        pos = s[grid["hot"][grid["sc"]]]
        neg = s[~grid["hot"][grid["sc"]]]
        assert np.isfinite(pos).all() and np.isfinite(neg).any()
        assert float(np.nanmedian(pos)) > float(np.nanquantile(neg[np.isfinite(neg)], 0.9)), \
            f"{idx.name}: anomaly bins should sit above the clean 90th percentile"


def test_tvdi_reports_how_much_of_the_envelope_it_could_bound(grid):
    tv = _tvdi(grid)
    edges = tv.fit_record["per_bin_edges"]
    assert all(v["bins_total"] == 4 for v in edges.values())
    assert any(v["bins_filled"] >= 2 for v in edges.values())
    assert np.isfinite(tv.scores).sum() < tv.scores.size, "unbounded bins must stay NaN"
    assert not np.isnan(tv.scores[grid["sc"]]).all()


def test_index_fitting_refuses_the_scored_rows(grid):
    with pytest.raises(ValueError, match="appear in both"):
        indices.tvdi(grid["x"], grid["names"], fit_idx=np.arange(grid["n"]), scored_idx=grid["sc"])


def test_index_fitting_refuses_a_tiny_envelope(grid):
    with pytest.raises(ValueError, match="envelope"):
        indices.tvdi(grid["x"], grid["names"], fit_idx=np.arange(4), scored_idx=grid["sc"])


def test_missing_channel_makes_an_index_unavailable_not_imputed(grid):
    no_lst = indices.tvdi(grid["x"][:, :, 1:], ("ndvi", "evi"), fit_idx=grid["fit"], scored_idx=grid["sc"])
    assert not no_lst.available and no_lst.scores is None and "lst" in no_lst.missing


def test_within_field_cwsi_pools_only_when_a_field_has_no_history_and_says_so(grid):
    """A field with no fit rows is scored against the pooled envelope, recorded as `pooled_fill`.

    Refusing outright would mean a new farm can never be scored; silently pooling would
    present a cross-field number as a within-field one. Both are wrong in opposite
    directions, so the code does the first and records it the way the second would need.
    """
    lone = np.zeros(grid["n"], dtype=bool)
    lone[grid["sc"][:5]] = True                          # five fields whose fit history is removed
    fit = grid["fit"][~np.isin(grid["farm"][grid["fit"]], grid["farm"][grid["sc"][:5]])]
    cw = indices.cwsi_lst(grid["x"], grid["names"], fit_idx=fit, scored_idx=grid["sc"])
    assert cw.available
    rec = cw.fit_record
    assert rec["pooled_fill"] >= 1, rec
    assert "pooled" in rec["pooled_fill_note"]


def test_et_indices_unavailable_without_et0_and_never_substituted(grid):
    et = indices.evapotranspiration(grid["x"], grid["names"], None, fit_idx=grid["fit"],
                                    scored_idx=grid["sc"], kind="ETa")
    assert not et.available and "et0" in et.missing and "fallback" in et.note
    with_et = indices.evapotranspiration(grid["x"], grid["names"], grid["weather"], fit_idx=grid["fit"],
                                         scored_idx=grid["sc"], kind="ETrF")
    assert with_et.available and with_et.scores.shape == (grid["n"], grid["T"])
    s = with_et.scores[grid["sc"]]
    assert np.nanmean(s[grid["hot"][grid["sc"]]]) > np.nanmean(s[~grid["hot"][grid["sc"]]]), \
        "ETrF must be reported as a deficit so higher = more stressed, like every other row"


def test_cwsi_et_is_declared_unavailable_rather_than_approximated(grid):
    out = indices.compute_all(grid["x"], grid["names"], weather=grid["weather"], fit_idx=grid["fit"],
                              scored_idx=grid["sc"], ndvi_bins=4, min_support=3)
    assert set(out) == set(indices.INDICES)
    assert not out["CWSI_ET"].available and "aerodynamic" in out["CWSI_ET"].note
    assert out["TVDI"].available and out["CWSI_LST"].available and out["ETa"].available


def test_circularity_guard_refuses_an_index_scored_against_its_own_input():
    assert indices.assert_not_circular("TVDI", "field_stress_records")["independent"]
    with pytest.raises(ValueError, match="own input"):
        indices.assert_not_circular("CWSI_LST", "LST_anomaly")
    with pytest.raises(ValueError, match="own input"):
        indices.assert_not_circular("ETa", "evapotranspiration_deficit")


def test_index_metrics_are_scoreable_on_the_same_labels(grid):
    tv = _tvdi(grid)
    labels = np.zeros((grid["n"], grid["T"]), dtype=bool)
    labels[grid["sc"]] = grid["hot"][grid["sc"]]
    rows = indices.reference_metrics(tv, labels, eval_idx=grid["sc"])
    assert rows["available"] and 0.0 < rows["auprc"] <= 1.0 and 0 < rows["prevalence"] < 0.5
    assert rows["auroc"] > 0.5 and rows["fit"]["disjoint"] is True


def test_index_metrics_report_absence_in_words(grid):
    labels = np.zeros((grid["n"], grid["T"]), dtype=bool)
    labels[grid["sc"]] = grid["hot"][grid["sc"]]
    no_lst = indices.tvdi(grid["x"][:, :, 1:], ("ndvi", "evi"), fit_idx=grid["fit"], scored_idx=grid["sc"])
    rows = indices.reference_metrics(no_lst, labels, eval_idx=grid["sc"])
    assert rows["available"] is False and "lst" in rows["reason"]


def test_agreement_uses_farm_blocks_and_refuses_one_cluster(grid):
    tv = _tvdi(grid)
    arm = np.asarray(grid["x"], dtype=np.float64)[:, :, 0] + 0.5 * np.asarray(grid["x"], dtype=np.float64)[:, :, 1]
    T, sc = grid["T"], grid["sc"]
    cols = np.tile(np.arange(T), sc.size)
    lin = np.repeat(sc, T) * T + cols
    out = indices.index_agreement(arm.reshape(-1)[lin], tv.scores.reshape(-1)[lin],
                                  farm_ids=np.repeat(grid["farm"][sc], T), n_boot=200)
    assert out["available"], out
    assert out["resampled_clustered_by"] == "farm" and out["n_farms"] >= 8
    lo, hi = out["spearman_ci"]
    assert lo <= out["spearman"] <= hi
    thin = indices.index_agreement(arm.reshape(-1)[lin], tv.scores.reshape(-1)[lin],
                                   farm_ids=np.zeros(lin.size, dtype=int), n_boot=50)
    assert not thin["available"] and "clusters" in thin["reason"], \
        "a one-cluster block bootstrap must refuse rather than print a confident interval"


def test_agreement_field_year_blocks_are_the_contracted_unit(grid):
    """9.18 — with a year axis, whole (field, year) blocks are one draw, and the record says so."""
    tv = _tvdi(grid)
    arm = np.asarray(grid["x"], dtype=np.float64)[:, :, 0] + 0.5 * np.asarray(grid["x"], dtype=np.float64)[:, :, 1]
    T, sc = grid["T"], grid["sc"]
    cols = np.tile(np.arange(T), sc.size)
    lin = np.repeat(sc, T) * T + cols
    farms = np.repeat(grid["farm"][sc], T)
    years = np.repeat(sc // 10, T)                          # 2 archive-year blocks over the scored rows
    out = indices.index_agreement(arm.reshape(-1)[lin], tv.scores.reshape(-1)[lin],
                                  farm_ids=farms, years=years, n_boot=200)
    assert out["available"] and out["resampled_clustered_by"] == "farm_year", out
    assert out["n_blocks"] == int(len(np.unique(np.stack([farms, years], 1), axis=0)))
    assert out["n_blocks"] >= 8, "the fixture must supply enough (field, year) blocks to bootstrap"
    assert "field-year" in out["note"] or "farm_year" == out["resampled_clustered_by"]
    lo, hi = out["spearman_ci"]
    assert lo <= out["spearman"] <= hi
    misaligned = indices.index_agreement(arm.reshape(-1)[lin], tv.scores.reshape(-1)[lin],
                                         farm_ids=farms, years=years[:-1], n_boot=50)
    assert not misaligned["available"] and "disagree" in misaligned["reason"]
    fallback = indices.index_agreement(arm.reshape(-1)[lin], tv.scores.reshape(-1)[lin],
                                       farm_ids=farms, n_boot=50)
    assert fallback["resampled_clustered_by"] == "farm" and "NOT used" in fallback["note"], (
        "without a year axis the record must say the contract's field-year form was not used")


# --------------------------------------------------------------------------- protocols


class _Frame:
    """60 rows = 20 farms × 3 cycles; the last cycle of every farm is the dev partition.

    Farms deliberately span both partitions, because that is what real cycle tables look
    like and it is exactly the case a naive row-wise holdout gets wrong.
    """

    def __init__(self, n=60, *, per_farm=3, split=None, start=None, tile_ids="default"):
        self.x = np.zeros((n, 3, 2), dtype=np.float32)
        farm = np.arange(n) // per_farm
        cyc = np.arange(n) % per_farm
        self.field_ids = farm
        self.start_days = cyc * 100 + farm if start is None else np.asarray(start)
        self.end_days = self.start_days + 60
        self.split = (np.where(cyc < per_farm - 1, "train", "dev").astype(object) if split is None
                      else np.asarray(split, dtype=object)[:n])
        self.tile_ids = (np.where(farm % 2 == 0, 0, 1) if tile_ids == "default"
                         else (None if tile_ids is None else np.asarray(tile_ids)))


class _Cfg:
    raw = {"evaluation": {"farm_holdout_frac": 1 / 3, "allowed_splits": ["train", "dev"],
                          "forbidden_splits": ["test", "monitoring"]}}


@pytest.fixture()
def frame():
    return _Frame()


def test_v1_is_temporal_and_shares_farms_by_design(frame):
    ps = protocols.build_protocol(frame, _Cfg, "V1")
    assert ps.n_eval == 20 and ps.n_train == 40
    assert (frame.start_days[ps.train_idx].max() < frame.start_days[ps.eval_idx].min()
            ), "V1 must be a forward holdout in cycle order"
    assert "temporal holdout" in ps.note
    assert len(set(frame.field_ids[ps.train_idx]) & set(frame.field_ids[ps.eval_idx])) == 20


def test_v2_holds_out_whole_farms(frame):
    ps = protocols.build_protocol(frame, _Cfg, "V2")
    train_f = set(int(f) for f in frame.field_ids[ps.train_idx])
    eval_f = set(int(f) for f in frame.field_ids[ps.eval_idx])
    assert not (train_f & eval_f), f"leaked farms: {sorted(train_f & eval_f)}"
    assert ps.n_farms_eval == 7 and ps.n_eval == 7, "one scored dev cycle per held-out farm"
    assert ps.exclusions["eval_side_train_rows"] == 14
    assert ps.exclusions["train_side_dev_rows_used_for_fitting"] == 0


def test_v3_needs_a_spatial_index_and_is_smaller_when_it_has_one(frame):
    naked = _Frame()
    naked.tile_ids = None                                # a frame with no spatial index at all
    with pytest.raises(ValueError, match="spatial index"):
        protocols.build_protocol(naked, _Cfg, "V3")
    v2 = protocols.build_protocol(frame, _Cfg, "V2")
    v3 = protocols.build_protocol(frame, _Cfg, "V3")
    assert v3.n_train < v2.n_train, "V3 must drop training rows sharing a tile with the scored set"
    assert not (set(v3.exclusions["v3_tiles_train"]) & set(v3.exclusions["v3_tiles_eval"]))


def test_unknown_protocol_is_a_keyerror_not_a_silent_default():
    with pytest.raises(KeyError, match="not one of"):
        protocols.build_protocol(_Frame(), _Cfg, "V9")


def test_a_leaky_assignment_is_a_hard_failure():
    with pytest.raises(AssertionError, match="farm-block-consistent"):
        protocols._assert_disjoint({1, 2, 3}, {3}, protocol="V2")
    protocols._assert_disjoint({1, 2}, {3, 4}, protocol="V2")


def test_v1_overlap_assertion_fires_when_the_ordering_is_false():
    f = _Frame()
    f.start_days = np.where(np.asarray(f.split) == "dev", 5, 100 + np.arange(60))
    f.end_days = f.start_days + 30
    with pytest.raises(AssertionError, match="temporal overlap"):
        protocols.build_protocol(f, _Cfg, "V1")


def test_too_few_farms_refuses_the_transfer_claim():
    tiny = _Frame(n=6, per_farm=3)                     # 2 farms
    with pytest.raises(ValueError, match="≥5 distinct farms"):
        protocols.build_protocol(tiny, _Cfg, "V2")


def test_an_empty_holdout_is_refused_not_reported_as_zero(frame):
    frame.split = np.full(60, "train", dtype=object)     # nothing to score
    with pytest.raises(ValueError, match="no dev rows"):
        protocols.build_protocol(frame, _Cfg, "V2")


def test_apply_protocol_passthrough_is_labelled(frame):
    tr, ev, rec = protocols.apply_protocol(frame, _Cfg, None)
    assert rec["protocol"].startswith("V0") and ev.size == 20 and tr.size == 40
    tr2, ev2, rec2 = protocols.apply_protocol(frame, _Cfg, "V2")
    assert rec2["disjoint_farms"] and ev2.size == 7


def test_audit_protocols_reports_all_three_and_flags_equal_sizes(frame):
    out = protocols.audit_protocols(frame, _Cfg)
    assert set(protocols.PROTOCOLS) <= set(out)
    assert "_sizes" in out and out["V1"]["n_eval"] == 20
    assert "smallest" in out["_note"]
    assert out["V3"]["n_eval"] <= out["V2"]["n_eval"]


def test_locked_test_split_never_reachable(frame):
    """Rows marked `test` in the source table are excluded, and the count is recorded."""
    from ssl_anomaly.benchmarking.splits import TestSplitAccessError, assert_admissible

    with pytest.raises(TestSplitAccessError):
        assert_admissible(("train", "test"))
    f = _Frame()
    sp = np.asarray(f.split, dtype=object)
    dev = np.flatnonzero(sp == "dev")
    sp[dev[:10]] = "test"                 # the locked partition sits *after* train, as it does live
    f.split = sp
    for p in protocols.PROTOCOLS:
        ps = protocols.build_protocol(f, _Cfg, p)
        assert ps.exclusions["not_in_admissible_splits"] == 10
        assert not set(np.asarray(f.split)[ps.train_idx]) & {"test", "monitoring"}
        assert not set(np.asarray(f.split)[ps.eval_idx]) & {"test", "monitoring"}


# --------------------------------------------------------------------------- hpo


class _Cfg2:
    raw = {"search": {
        "validation_only": True, "selection_metric": "macro_auprc", "n_trials": 4,
        "wall_minutes_cap": 5.0,
        "budgets": {"A1": {"trials": 4, "space": {"d_model": [64, 96], "lr": "log[3e-4, 3e-3]"}},
                    "A2": {"trials": 4, "space": {"d_model": [64, 96], "lr": "log[3e-4, 3e-3]"}}},
        "unsearchable": {"A-RAND": "random-init control"},
    }}


def test_space_parsing_is_strict():
    sp = hpo.parse_space({"d_model": [64, 96], "lr": "log[3e-4, 3e-3]", "drop": "uniform[0.0, 0.3]"})
    assert sp["d_model"]["kind"] == "choice" and sp["lr"]["kind"] == "loguniform"
    assert sp["drop"]["kind"] == "uniform"
    with pytest.raises(hpo.SearchError, match="neither a candidate list"):
        hpo.parse_space({"lr": "3e-4..3e-3"})
    with pytest.raises(hpo.SearchError, match="0 < low < high"):
        hpo.parse_space({"lr": "log[3e-3, 3e-4]"})
    with pytest.raises(hpo.SearchError, match="empty"):
        hpo.parse_space({"d_model": []})


def test_sampling_is_reproducible_and_inside_the_declared_grid():
    sp = hpo.parse_space({"d_model": [64, 96], "lr": "log[3e-4, 3e-3]"})
    ra, rb = np.random.default_rng(3), np.random.default_rng(3)
    a = [hpo.sample(sp, ra) for _ in range(20)]
    b = [hpo.sample(sp, rb) for _ in range(20)]
    assert a == b, "a search with an unseeded RNG cannot be re-run to confirm its winner"
    assert {c["d_model"] for c in a} <= {64, 96}
    assert all(3e-4 <= c["lr"] <= 3e-3 for c in a)
    assert len({round(c["lr"], 8) for c in a}) > 1


def test_run_search_is_reproducible_across_calls():
    ev = lambda c: (float(c["d_model"]) / 100.0, {})          # noqa: E731
    a = hpo.run_search(_Cfg2, "A1", ev, split="dev", seed=13)
    b = hpo.run_search(_Cfg2, "A1", ev, split="dev", seed=13)
    assert [t.config for t in a.trials] == [t.config for t in b.trials], \
        "a re-run must draw the same configurations, or 'I could not reproduce my search' is by design"


def test_search_selects_the_best_trial_on_dev_only():
    calls = []

    def ev(cfg):
        calls.append(cfg)
        return (1.0 if cfg["d_model"] == 96 else 0.1), {"params": cfg["d_model"] * 10}

    rec = hpo.run_search(_Cfg2, "A1", ev, split="dev", seed=13)
    assert rec.best is not None and rec.best.config["d_model"] == 96
    assert rec.validation_only and len(rec.trials) == 4
    assert len(calls) == 4


def test_search_refuses_a_forbidden_split():
    for split in ("test", "monitoring"):
        with pytest.raises(hpo.SearchError, match="is refused"):
            hpo.run_search(_Cfg2, "A1", lambda c: (0.5, {}), split=split)


def test_search_records_the_wall_cap_truncation():
    def slow(cfg):
        import time

        time.sleep(0.35)
        return 0.5, {}

    rec = hpo.run_search(_Cfg2, "A1", slow, split="dev", seed=2, trials=40, wall_minutes=0.1)
    assert rec.stopped_by == "wall_minutes_cap" and len(rec.trials) < 40
    assert "trials actually run" in rec.notes[0]


def test_failed_trials_are_kept_in_the_trace_and_abort_after_max_failures():
    n = {"i": 0}

    def boom(cfg):
        n["i"] += 1
        raise RuntimeError(f"cuda OOM #{n['i']}")

    rec = hpo.run_search(_Cfg2, "A1", boom, split="dev", seed=1)
    assert rec.best is None and rec.stopped_by == "3_consecutive_failures"
    assert all("OOM" in (t.error or "") for t in rec.trials)
    assert "pre-registered default" in " ".join(rec.notes)


def test_non_finite_scores_never_win():
    seq = iter([float("nan"), 0.9, 0.4])

    def ev(cfg):
        return next(seq), {"error": "collapse"}

    rec = hpo.run_search(_Cfg2, "A1", ev, split="dev", seed=5, trials=3)
    assert rec.best.score == 0.9 and any("non-finite" in t.note for t in rec.trials)
    assert sum(1 for t in rec.trials if np.isfinite(t.score)) == 2


def test_unsearchable_and_missing_arms_are_named_not_silently_skipped():
    with pytest.raises(hpo.SearchError, match="unsearchable"):
        hpo.budget_for(_Cfg2, "A-RAND")
    with pytest.raises(hpo.SearchError, match="no entry in search.budgets"):
        hpo.budget_for(_Cfg2, "A7")


def test_equal_budget_audit():
    assert hpo.assert_equal_budgets(_Cfg2)["trials_per_arm"] == 4
    bad = _Cfg2()
    bad.raw = {**_Cfg2.raw, "search": {**_Cfg2.raw["search"],
                                       "budgets": {**_Cfg2.raw["search"]["budgets"],
                                                   "A2": {"trials": 2, "space": {}}}}}
    with pytest.raises(hpo.SearchError, match="unequal"):
        hpo.assert_equal_budgets(bad)


def test_search_audit_reads_the_record_not_the_promise():
    t = hpo.Trial(0, {"d_model": 64}, 0.5, "dev", 1.0)
    ok = hpo.SearchRecord("A1", [t], t, {"trials": 4}, "dev")
    leak = hpo.SearchRecord("A2", [hpo.Trial(0, {"d_model": 64}, 0.9, "test", 1.0)], None, {"trials": 4}, "test")
    assert hpo.audit_search([ok])["ok"]
    out = hpo.audit_search([ok, leak])
    assert not out["ok"] and out["violations"][0]["model_id"] == "A2"
    tbl = hpo.search_table([ok, leak])
    assert "| A1 | 1 | 1 | 0 | 0.5000 | d_model=64" in tbl

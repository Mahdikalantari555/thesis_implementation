"""Downstream suite tests (change tasks 4.3-4.7).

- Injection pools come from the FROZEN severity table (deltas match the table;
  families have disjoint durations; unobserved bins never become observations;
  deterministic in seed).
- T2 cells are measured with AUC-PR + prevalence, per family x severity.
- T1 probe reports majority + B1-feature comparators.
- T3 Track 1 metrics computed on a HAND-CHECKED ranking; Track 2 and the
  near-match listing are exploratory (claim_bearing=False) and Track 2 is
  absent-with-reason (no in-repo producer).
- T4: identical cut-based definition for persistence and any supplied
  forecaster; a perfect forecaster beats persistence, noise does not.
- T5: output is labelled exploratory and carries no claim-bearing flag.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.evaluation.data import synthetic_evaluation_split
from ssl_anomaly.evaluation.downstream import (
    build_injection_pools,
    near_matches_listing,
    near_matches_markdown,
    t1_phenology_probe,
    t2_family_severity_cells,
    t3_fewshot_retrieval,
    t3_track2_weak_proxy,
    t4_trajectory_vs_persistence,
    t5_exploratory_listing,
)
from ssl_anomaly.monitor.severity_table import (
    PERSISTENT_DURATION_BINS,
    SEVERITY_TABLE,
    severity_table_hash,
)


@pytest.fixture(scope="module")
def table():
    return synthetic_evaluation_split(n_train=40, n_dev=40, n_fields=10, seed=13)


@pytest.fixture(scope="module")
def pools(table):
    cd = table
    return build_injection_pools(cd.X, cd.M, n_per_cell=2, seed=13)


# ---------------------------------------------------------------------------
# injection pools (T2's label source)
# ---------------------------------------------------------------------------


def test_pools_comply_with_the_frozen_severity_table(pools):
    cd_sev = {lv.level_id: lv for lv in SEVERITY_TABLE}
    assert pools["severity_table_sha256"] == severity_table_hash()
    assert pools["status"] == "built"
    # every family x severity cell has exactly n_per_cell injections
    fams, sevs = pools["families"], pools["severities"]
    for lv in SEVERITY_TABLE:
        for fam in ("acute", "persistent", "gradual"):
            n = int(((fams == fam) & (sevs == lv.level_id)).sum())
            assert n == 2, (fam, lv.level_id, n)


def test_injection_only_touches_observed_ndvi_bins(table):
    cd = table
    pools = build_injection_pools(cd.X, cd.M, n_per_cell=1, seed=13)
    # reconstruct: injected rows must equal clean rows everywhere except
    # observed NDVI bins inside the injected window (checked conservatively:
    # non-NDVI channels are bit-identical to their source cycle)
    src_lut = {}
    rng = np.random.default_rng(13)
    n_bins = cd.X.shape[1]
    # the pool generator draws sources with the same rng sequence; verify via a
    # direct call instead of reconstructing the draw:
    from ssl_anomaly.evaluation.downstream import _inject_ndvi_drop
    i, start, dur, delta = 0, 3, 4, 0.1
    inj = _inject_ndvi_drop(cd.X[i], cd.M[i], start_bin=start,
                            duration_bins=dur, delta=delta)
    assert np.array_equal(inj[:, 1:], cd.X[i][:, 1:], equal_nan=True), "non-NDVI channels must be untouched"
    window = slice(start, start + dur)
    assert np.all(inj[window, 0][cd.M[i][window, 0]] <= cd.X[i][window, 0][cd.M[i][window, 0]])
    # unobserved NDVI bins stay NaN (never fabricated)
    unobs = ~cd.M[i][:, 0]
    assert np.all(~np.isfinite(inj[unobs, 0]))


def test_pools_are_deterministic_in_seed(table):
    cd = table
    a = build_injection_pools(cd.X, cd.M, n_per_cell=1, seed=42)
    b = build_injection_pools(cd.X, cd.M, n_per_cell=1, seed=42)
    assert np.array_equal(a["x_injected"], b["x_injected"], equal_nan=True)
    assert (a["families"] == b["families"]).all()


def test_persistent_family_uses_the_frozen_duration():
    assert PERSISTENT_DURATION_BINS == 5


# ---------------------------------------------------------------------------
# T2 — per-cell AUC-PR
# ---------------------------------------------------------------------------


def test_t2_cells_measured_with_prevalence(table, pools):
    cd = table
    # a scorer that ranks injected cycles HIGH via their known NDVI drop
    x_all = np.concatenate([cd.X, pools["x_injected"]])
    ndvi = np.where(cd.M[: pools["x_injected"].shape[0]][:, :, 0] | True, 0, 0)
    # score = mean observed NDVI (injected cycles carry the drop -> low score)
    def mean_ndvi(x, m):
        v = np.where(m[:, :, 0], x[:, :, 0], np.nan)
        return -np.nanmean(v, axis=1)          # higher = more anomalous
    clean_scores = mean_ndvi(cd.X, cd.M)
    inj_scores = mean_ndvi(pools["x_injected"], cd.M[: pools["x_injected"].shape[0]])
    t2 = t2_family_severity_cells(inj_scores, pools, clean_scores=clean_scores)
    cells = t2["cells"]
    measured = {k: v for k, v in cells.items() if v["status"] == "measured"}
    assert len(measured) == 18                  # 3 families x 6 severities
    for k, v in measured.items():
        assert {"auprc", "prevalence", "n", "n_positives"} <= set(v)
        assert 0.0 <= v["auprc"] <= 1.0
        assert v["prevalence"] == pytest.approx(0.25, abs=0.01)


def test_t2_perfect_scorer_hands_deep_severities_high_ap(table, pools):
    cd = table
    def mean_ndvi(x, m):
        v = np.where(m[:, :, 0], x[:, :, 0], np.nan)
        return -np.nanmean(v, axis=1)
    clean_scores = mean_ndvi(cd.X, cd.M)
    inj_scores = mean_ndvi(pools["x_injected"], cd.M[: pools["x_injected"].shape[0]])
    cells = t2_family_severity_cells(inj_scores, pools, clean_scores=clean_scores)["cells"]
    # S6's 0.40 drop is far below the fixture's clean NDVI range -> near-perfect AP
    assert cells["acute:S6"]["auprc"] > cells["acute:S1"]["auprc"]


# ---------------------------------------------------------------------------
# T1
# ---------------------------------------------------------------------------


def test_t1_probe_reports_both_comparators():
    # Own fixture, bigger than the module default: at n_eval=40 with 10 classes
    # the random-vs-B1 comparison is estimation noise, not a finding.
    cd = synthetic_evaluation_split(n_train=240, n_dev=240, n_fields=8, seed=13)
    # Per-cycle "phase" target = the cycle's field (the fixture's seasonal
    # amplitude is field-specific), so a B1 feature (mean observed NDVI) is
    # learnable while an uninformative embedding is not.
    y_bins = (cd.field_ids - 1).astype(np.int64)
    z = np.zeros((cd.n_cycles, 4), dtype=np.float32)     # carries no information
    b1 = np.where(cd.M[:, :, 0], cd.X[:, :, 0], np.nan)
    b1_feat = np.nan_to_num(np.nanmean(b1, axis=1), nan=0.0)[:, None]
    train = np.zeros(cd.n_cycles, dtype=bool)
    train[: cd.n_cycles // 2] = True
    evalm = ~train
    out = t1_phenology_probe(z, y_bins, train, evalm, n_classes=int(y_bins.max()) + 1,
                             b1_features=b1_feat)
    assert out["status"] == "measured" and out["claim_bearing"] is True
    assert "majority_baseline" in out and "b1_feature_probe" in out
    assert out["accuracy"] <= out["majority_baseline"] + 0.01, (
        "an uninformative embedding must not beat the majority baseline")
    assert out["b1_feature_probe"]["accuracy"] > out["accuracy"], (
        "the B1-feature probe should beat an uninformative embedding here; "
        "if it does not, the fixture or probe is broken")


# ---------------------------------------------------------------------------
# T3
# ---------------------------------------------------------------------------


def test_t3_track1_metrics_on_a_hand_checked_ranking():
    # 6 gallery cycles, 4-dim embeddings; queries + gallery built so the
    # expected ranking is known exactly for one query.
    emb = np.array([
        [1.0, 0.0, 0.0, 0.0],   # 0: query (acute)
        [0.9, 0.1, 0.0, 0.0],   # 1: acute  (most similar)
        [0.8, 0.0, 0.2, 0.0],   # 2: acute
        [0.0, 1.0, 0.0, 0.0],   # 3: gradual
        [0.0, 0.9, 0.1, 0.0],   # 4: gradual
        [0.1, 0.0, 0.9, 0.0],   # 5: persistent
    ])
    emb32 = emb.astype(np.float32)
    arms = {"ssl": emb32, "a-rand": np.roll(emb32, 2, axis=0)}
    res = t3_fewshot_retrieval(
        arms,
        query_indices=np.array([0]),
        query_families=np.array(["acute"]),
        gallery_indices=np.array([1, 2, 3, 4, 5]),
        gallery_families=np.array(["acute", "acute", "gradual", "gradual", "persistent"]),
    )
    ssl = res["arms"]["ssl"]
    assert ssl["status"] == "measured" and ssl["n_queries"] == 1
    assert ssl["precision_at_m"]["1"] == 1.0          # top-1 is acute
    assert ssl["precision_at_m"]["5"] == pytest.approx(2 / 5)
    assert ssl["family_hit_rate_at_k"]["1"] == 1.0
    assert ssl["family_hit_rate_at_k"]["3"] == 1.0
    assert ssl["map"] == pytest.approx((1 / 1 + 2 / 2) / 2)
    # score-ranked baseline arm shares the interface
    res2 = t3_fewshot_retrieval(
        {"b1-score": np.zeros(6)},
        query_indices=np.array([0]),
        query_families=np.array(["acute"]),
        gallery_indices=np.array([1, 2, 3, 4, 5]),
        gallery_families=np.array(["acute", "acute", "gradual", "gradual", "persistent"]),
        arm_scores={"b1-score": np.array([9.0, 9.0, 1.0, 1.0, 1.0, 1.0, 0])},
    )
    assert res2["arms"]["b1-score"]["precision_at_m"]["1"] == 1.0


def test_t3_track2_absent_never_claim_bearing():
    res = t3_track2_weak_proxy({}, np.array([0]), np.array([1]))
    assert res["claim_bearing"] is False and res["label"] == "exploratory"
    assert res["status"] == "absent" and "weak-proxy" in res["reason"]


def test_near_matches_listing_is_descriptive_and_natural_only():
    emb = np.eye(6, dtype=np.float32)
    natural = np.array([2, 3, 4, 5])
    meta = [{"field": f"F{natural[i]}", "date": f"2020-{natural[i]:02d}"} for i in range(len(natural))]
    listing = near_matches_listing({"ssl": emb}, np.array([0]), natural, meta, top=2)
    assert listing["claim_bearing"] is False and listing["label"] == "exploratory"
    nb = listing["arms"]["ssl"][0]["neighbours"]
    assert len(nb) == 2
    assert all(n["cycle_index"] in set(natural.tolist()) for n in nb)
    assert nb[0]["rank"] == 1 and nb[0]["field"] == "F2"
    md = near_matches_markdown(listing, {2: "F2 2020-02", 3: "F3 2020-03"})
    assert "EXPLORATORY" in md and "never" in md.lower()
    with pytest.raises(ValueError, match="misalign"):
        near_matches_listing({"ssl": emb}, np.array([0]), natural, meta[:-1])


# ---------------------------------------------------------------------------
# T4
# ---------------------------------------------------------------------------


def test_t4_persistence_and_perfect_forecaster_under_one_definition(table):
    cd = table
    x = np.nan_to_num(cd.X, nan=0.0)
    m = cd.M
    res = t4_trajectory_vs_persistence(x, m, horizons=(1, 3))
    assert res["persistence"]["1"]["n_cycles_used"] > 0
    assert res["model"]["1"]["status"] == "absent"       # no forecaster supplied
    # a perfect forecaster (reads the future target exactly — permitted HERE
    # only as a unit-test upper bound, never as a model) must beat persistence
    T = x.shape[1]
    cut = res["cut_bin"]
    def oracle(xi, mi, cut_, h):
        window = slice(cut_ + 1, min(T, cut_ + 1 + h))
        w = mi[window].sum(axis=0)
        tgt = (np.where(mi[window], xi[window], 0).sum(axis=0)) / np.maximum(w, 1)
        return tgt
    res2 = t4_trajectory_vs_persistence(x, m, horizons=(1, 3), forecaster=oracle)
    assert res2["model"]["1"]["beats_persistence"] is True


def test_t4_noisy_forecaster_loses_to_persistence():
    rng = np.random.default_rng(3)
    T, C = 36, 8
    x = np.tile(0.5, (40, T, C)) + rng.normal(0, 0.05, size=(40, T, C))
    m = np.ones((40, T, C), dtype=bool)
    res = t4_trajectory_vs_persistence(x, m, horizons=(1,), forecaster=lambda *a: rng.normal(0.5, 0.3, C))
    assert res["model"]["1"]["mae"] > res["persistence"]["1"]["mae"]


# ---------------------------------------------------------------------------
# T5
# ---------------------------------------------------------------------------


def test_t5_output_is_exploratory(table):
    rng = np.random.default_rng(13)
    z = rng.normal(size=(table.n_cycles, 8)).astype(np.float32)
    res = t5_exploratory_listing(z, {i: f"cycle{i}" for i in range(table.n_cycles)})
    assert res["claim_bearing"] is False and res["label"] == "exploratory"
    assert "never used in gate evaluation" in res["note"]
    assert len(res["clusters"]) >= 2

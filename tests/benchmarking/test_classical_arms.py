"""The classical reference estimators (tasks 8.1–8.3, 11.1).

The BL-RPCA arm's declared search space is ``{lam, iters}`` — the estimator behind the
arm must actually read them, or the search is theatre. These tests fail if the declared
space and the consumed knobs drift apart again, and if the estimator stops being robust
to training contamination (the property that distinguishes Robust PCA from plain PCA).
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.suppression.classical import (
    FITTER_CONSUMED,
    TransferError,
    _pcp_inexact_alm,
    fit_score,
    fitter_for,
    unconsumed,
)


@pytest.fixture()
def contaminated_tensor():
    """Rank-3 signal + 2 % of rows carrying a sparse magnitude-5 corruption (both splits).

    The 2 % rate and the 5σ magnitudes sit inside the contract's contamination grid
    (``synthetic.test_prevalence`` 0.10, sensitivity grid 0.02–0.35) — a corruption that
    dwarfs the signal's spectral norm would fail every estimator including the exact one.
    """
    rng = np.random.default_rng(3)
    n, d, r = 100, 40, 3
    low = rng.normal(size=(n, r)) @ rng.normal(size=(r, d))
    sparse = np.zeros((n, d))
    corrupt = rng.choice(n, 4, replace=False)                     # 2 % of rows, train + eval
    sparse[np.ix_(corrupt, rng.choice(d, 8, replace=False))] = 5.0
    x = low + sparse + rng.normal(0, 0.25, (n, d))
    return {"x": x, "train": x[:80], "eval": x[80:], "corrupt": corrupt, "rank": r,
            "true_subspace": np.linalg.svd(low, full_matrices=False)[2][:r].T}


def _subspace_angle(basis_dk, true_subspace):
    """Max principal angle (deg) between a (d, k) basis and the true one."""
    svals = np.linalg.svd(true_subspace.T @ basis_dk, compute_uv=False)
    return float(np.degrees(np.arccos(np.clip(svals, -1, 1))).max())


def test_bl_rpca_declared_space_is_fully_consumed():
    assert fitter_for("BL-RPCA") == "rpca_pca"
    assert FITTER_CONSUMED["rpca_pca"] == frozenset({"n_components", "lam", "iters"})
    assert unconsumed("BL-RPCA", {"lam", "iters", "n_components"}) == []
    assert unconsumed("BL-RPCA", {"lam", "iters", "contamination"}) == ["contamination"], (
        "PCP has no contamination parameter; a selection carrying one must be refused, "
        "not silently dropped (the pre-fix table listed it as consumed while no code read it)")


def test_pcp_demotes_train_contamination_to_the_sparse_component(contaminated_tensor):
    t = contaminated_tensor
    mu = t["train"].mean(axis=0)
    _l, s_mat, info = _pcp_inexact_alm(t["train"] - mu, rank=t["rank"], lam=0.02, iters=300)
    row_s = np.abs(s_mat).sum(axis=1)
    train_corrupt = [c for c in t["corrupt"] if c < len(t["train"])]
    assert train_corrupt, "fixture must corrupt train rows"
    thr = float(np.quantile(row_s, 0.95))
    assert all(row_s[c] > thr for c in train_corrupt), (
        "the sparse component must carry the corrupted train rows; if the corruption ends up "
        "in L, the learned subspace is the contaminated one and the 'robust' name is a label")


def test_pcp_subspace_is_closer_to_the_truth_than_plain_pca(contaminated_tensor):
    t = contaminated_tensor
    mu = t["train"].mean(axis=0)
    l_mat, _s, _info = _pcp_inexact_alm(t["train"] - mu, rank=t["rank"], lam=0.02, iters=300)
    pcp_basis = np.linalg.svd(l_mat, full_matrices=False)[2][: t["rank"]].T
    plain = np.linalg.svd(t["train"] - mu, full_matrices=False)[2][: t["rank"]].T
    assert _subspace_angle(pcp_basis, t["true_subspace"]) < \
        _subspace_angle(plain, t["true_subspace"]), (
        "at matched rank the PCP subspace must sit closer to the uncontaminated one than "
        "plain PCA's; if not, Robust PCA buys nothing over PCA and the baseline row is void")


def test_rpca_scores_flag_corrupted_eval_rows(contaminated_tensor):
    t = contaminated_tensor
    corrupt_ev = [c - len(t["train"]) for c in t["corrupt"] if c >= len(t["train"])]
    clean_ev = [i for i in range(len(t["eval"])) if i not in corrupt_ev]
    assert corrupt_ev and clean_ev, "fixture must corrupt eval rows too"
    s, rec = fit_score("rpca_pca", t["train"], t["eval"],
                       params={"lam": 0.02, "iters": 300, "n_components": t["rank"]})
    assert rec["estimator"] == "pcp_inexact_alm" and rec["lam"] == 0.02 and rec["iters"] == 300
    assert rec["lam_source"] == "selection" and rec["alm_iters_run"] <= rec["iters"]
    assert float(np.nanmedian(s[corrupt_ev])) > float(np.nanmedian(s[clean_ev])), (
        "a corrupted eval row is further from the learned normal subspace than a clean one; "
        "the score must order them that way or the arm detects nothing")


def test_default_lam_is_the_pcp_theoretical_one_and_is_recorded(contaminated_tensor):
    t = contaminated_tensor
    s, rec = fit_score("rpca_pca", t["train"], t["eval"], params={})
    n, d = t["train"].shape
    assert rec["lam_source"] == "pcp_theory_1_over_sqrt_max_nd"
    assert rec["lam"] == pytest.approx(1.0 / np.sqrt(max(n, d)), abs=1e-5), (
        "the unsearched default must be the PCP-theoretical lam, and the record must say so — "
        "an undocumented default would make 'lam was not searched' unauditable")


def test_searched_knobs_change_the_fit_not_just_the_label(contaminated_tensor):
    t = contaminated_tensor
    mu = t["train"].mean(axis=0)
    _, s_small, _ = _pcp_inexact_alm(t["train"] - mu, rank=t["rank"], lam=0.02, iters=300)
    _, s_big, _ = _pcp_inexact_alm(t["train"] - mu, rank=t["rank"], lam=0.2, iters=300)
    # a larger lam penalises sparsity more strongly: the sparse component must shrink
    assert float(np.abs(s_big).sum()) < float(np.abs(s_small).sum()), (
        "lam is declared as the L1 weight of PCP; if moving it leaves the sparse component "
        "unchanged, the knob is decorative and the search is theatre again")
    _, _s, info = _pcp_inexact_alm(t["train"] - mu, rank=t["rank"], lam=0.02, iters=5)
    assert info["iters_run"] <= 5, "iters must cap the ALM iterations, not be recorded and ignored"


def test_pcp_is_deterministic_and_refuses_non_finite_input(contaminated_tensor):
    t = contaminated_tensor
    a, _ = fit_score("rpca_pca", t["train"], t["eval"], params={"lam": 0.05, "iters": 100})
    b, _ = fit_score("rpca_pca", t["train"], t["eval"], params={"lam": 0.05, "iters": 100})
    assert np.array_equal(a, b), "pure-numpy PCP has no RNG; two identical calls must agree exactly"
    bad = np.vstack([t["train"], np.full((1, t["train"].shape[1]), np.nan)])
    with pytest.raises(TransferError, match="non-finite"):
        # PCP itself has no NaN semantics: a non-finite row that reaches the solver is a
        # data defect that must stop the fit, not propagate through every downstream score.
        _pcp_inexact_alm(bad, rank=3, lam=0.05, iters=100)

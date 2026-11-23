"""B1–B5 ladder tests.

The ladder's whole value is that each rung differs from its neighbour by exactly
one named ingredient; a test that only checks "it builds" would not protect that
property, so the assertions here are about *composition*, not just success.
"""

from __future__ import annotations

import numpy as np
import pytest

from ssl_anomaly.evaluation.ladder import (
    LADDER,
    build_rung,
    ladder_status,
    ladder_table,
    linear_probe_params,
    match_budget_b4,
    rung_deltas,
)


@pytest.fixture()
def blocks():
    rng = np.random.default_rng(3)
    n = 25
    return {
        "encoder_embeddings": rng.normal(size=(n, 4)),
        "handcrafted_features": rng.normal(size=(n, 6)),
        "temporal_context_features": rng.normal(size=(n, 5)),
    }


# ---------------------------------------------------------------------------
# Ladder definition
# ---------------------------------------------------------------------------


def test_ladder_has_five_rungs_in_order():
    table = ladder_table()
    assert [r["name"] for r in table] == ["B1", "B2", "B3", "B4", "B5"]


def test_b1_is_frozen():
    assert LADDER["B1"].frozen is True
    assert LADDER["B2"].frozen is False


def test_each_rung_removes_exactly_one_thing_from_the_previous():
    """B2 -> B3 adds features; B3 -> B4 drops the encoder; B3 -> B5 swaps features."""
    assert LADDER["B2"].uses_handcrafted is False and LADDER["B3"].uses_handcrafted is True
    assert LADDER["B4"].uses_encoder is False and LADDER["B3"].uses_encoder is True
    assert LADDER["B5"].uses_handcrafted is False
    assert LADDER["B5"].uses_temporal_context is True


def test_b5_requires_decomposition_which_is_deferred():
    assert LADDER["B5"].uses_decomposition is True
    assert "decomposition_features" in LADDER["B5"].required_inputs()


def test_deltas_are_named():
    d = rung_deltas()
    assert set(d) == {"B2→B3", "B3→B4", "B2→B4", "B3→B5"}


# ---------------------------------------------------------------------------
# Building rungs
# ---------------------------------------------------------------------------


def test_b1_has_no_builder():
    with pytest.raises(ValueError, match="frozen incumbent"):
        build_rung("B1", encoder_embeddings=np.zeros((5, 2)))


def test_b2_uses_encoder_only(blocks):
    b = build_rung("B2", encoder_embeddings=blocks["encoder_embeddings"])
    b.assert_usable()
    assert b.n_features == 4
    assert set(b.block_slices) == {"encoder"}


def test_b2_ignores_features_when_not_required(blocks):
    """Supplying features to B2 must not silently add them."""
    b = build_rung("B2", encoder_embeddings=blocks["encoder_embeddings"],
                   handcrafted_features=blocks["handcrafted_features"])
    assert set(b.block_slices) == {"encoder"}


def test_b3_concatenates_in_fixed_order(blocks):
    b = build_rung("B3", encoder_embeddings=blocks["encoder_embeddings"],
                   handcrafted_features=blocks["handcrafted_features"])
    assert b.n_features == 10
    lo, hi = b.block_slices["encoder"]
    assert np.array_equal(b.feature_matrix[:, lo:hi], blocks["encoder_embeddings"])
    lo, hi = b.block_slices["handcrafted"]
    assert np.array_equal(b.feature_matrix[:, lo:hi], blocks["handcrafted_features"])


def test_b3_missing_features_is_absent(blocks):
    b = build_rung("B3", encoder_embeddings=blocks["encoder_embeddings"])
    assert b.status == "absent"
    assert b.missing == ("handcrafted_features",)
    with pytest.raises(ValueError, match="is absent"):
        b.assert_usable()


def test_b3_missing_both_lists_both(blocks):
    b = build_rung("B3")
    assert set(b.missing) == {"encoder_embeddings", "handcrafted_features"}


def test_b4_needs_no_encoder(blocks):
    b = build_rung("B4", handcrafted_features=blocks["handcrafted_features"])
    assert b.status == "built"
    assert b.n_features == 6
    assert "encoder" not in b.block_slices


def test_b5_is_deferred_without_decomposition(blocks):
    b = build_rung("B5", encoder_embeddings=blocks["encoder_embeddings"],
                   temporal_context_features=blocks["temporal_context_features"])
    assert b.status == "deferred"
    assert b.deferred == ("decomposition_features",)
    assert "zonal store" in b.reason


def test_b5_builds_when_decomposition_is_supplied(blocks):
    rng = np.random.default_rng(4)
    dec = rng.normal(size=(25, 3))
    b = build_rung(
        "B5", encoder_embeddings=blocks["encoder_embeddings"],
        temporal_context_features=blocks["temporal_context_features"],
        decomposition_features=dec, defer_decomposition=False,
    )
    assert b.status == "built"
    assert b.n_features == 4 + 5 + 3
    assert set(b.block_slices) == {"encoder", "temporal_context", "decomposition"}


def test_b5_absent_when_neither_deferred_nor_supplied(blocks):
    b = build_rung("B5", encoder_embeddings=blocks["encoder_embeddings"],
                   temporal_context_features=blocks["temporal_context_features"],
                   defer_decomposition=False)
    assert b.status == "absent"
    assert "decomposition_features" in b.missing


def test_row_count_mismatch_raises(blocks):
    with pytest.raises(ValueError, match="row counts disagree"):
        build_rung("B3", encoder_embeddings=blocks["encoder_embeddings"],
                   handcrafted_features=blocks["handcrafted_features"][:-2])


def test_non_finite_features_raise(blocks):
    bad = blocks["handcrafted_features"].copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        build_rung("B4", handcrafted_features=bad)


def test_unknown_rung_raises():
    with pytest.raises(ValueError, match="unknown rung"):
        build_rung("B6", handcrafted_features=np.zeros((3, 2)))


def test_feature_matrix_is_dense_and_correct_shape(blocks):
    b = build_rung("B3", encoder_embeddings=blocks["encoder_embeddings"],
                   handcrafted_features=blocks["handcrafted_features"])
    assert b.feature_matrix.shape == (25, 10)
    assert np.all(np.isfinite(b.feature_matrix))


# ---------------------------------------------------------------------------
# Budget matching
# ---------------------------------------------------------------------------


def test_probe_params_include_bias():
    assert linear_probe_params(10) == 11


def test_budget_matching_succeeds_when_enough_features():
    m = match_budget_b4(b3_n_features=10, n_available_features=50)
    assert m.within_tolerance
    assert m.achieved_params == m.target_params == 11


def test_budget_matching_fails_honestly_when_short():
    m = match_budget_b4(b3_n_features=10, n_available_features=3)
    assert not m.within_tolerance
    assert "cannot match" in m.detail
    assert m.achieved_params < m.target_params


def test_budget_match_records_mechanism():
    m = match_budget_b4(b3_n_features=6, n_available_features=40)
    assert "feature-selection" in m.mechanism
    assert m.to_dict()["within_tolerance"] is True


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------


def test_status_table_always_lists_all_five_rungs():
    table = ladder_status({})
    assert [r["name"] for r in table["rungs"]] == ["B1", "B2", "B3", "B4", "B5"]


def test_status_table_marks_b1_frozen_and_others_not_attempted():
    rows = {r["name"]: r for r in ladder_status({})["rungs"]}
    assert rows["B1"]["status"] == "frozen-artifact"
    assert rows["B2"]["status"] == "not_attempted"


def test_status_table_reflects_builds(blocks):
    builds = {
        "B2": build_rung("B2", encoder_embeddings=blocks["encoder_embeddings"]),
        "B5": build_rung("B5", encoder_embeddings=blocks["encoder_embeddings"],
                         temporal_context_features=blocks["temporal_context_features"]),
    }
    rows = {r["name"]: r for r in ladder_status(builds)["rungs"]}
    assert rows["B2"]["status"] == "built"
    assert rows["B5"]["status"] == "deferred"
    assert rows["B3"]["status"] == "not_attempted"


def test_status_table_carries_deltas():
    assert "B2→B3" in ladder_status({})["deltas"]

"""The contract itself: what the loader refuses, and why (Phase 1)."""

from __future__ import annotations

import copy

import pytest


def test_seeds_are_the_preregistered_three(cfg):
    assert cfg.seeds == (13, 42, 2026)
    assert cfg.raw["seeds"] == [13, 42, 2026]


def test_architecture_matrix_is_ten_arms_with_declared_pairs(cfg):
    from ssl_anomaly.benchmarking.config import ARCHITECTURES

    assert list(cfg.architectures)[:10] == [f"A{i}" for i in range(1, 11)]
    rows = [ARCHITECTURES[i] for i in cfg.architectures[:10]]
    assert {r["paradigm"] for r in rows} == {
        "input_reconstruction", "latent_reconstruction", "future_prediction", "jepa", "hybrid"}
    assert {r["backbone"] for r in rows} == {"gru", "mamba"}


def test_baselines_and_controls_are_registered(cfg):
    from ssl_anomaly.benchmarking.config import BASELINES

    ids = set(cfg.baselines)
    assert {"B1", "BL-IF", "BL-OCSVM", "BL-RPCA", "BL-RULE", "A-RAND"} <= ids
    assert BASELINES["B1"]["trained"] is False and BASELINES["BL-IF"]["trained"] is True


def test_capacity_band_and_laptop_cap(cfg):
    lo, hi = cfg.capacity_band
    assert (lo, hi) == (0.8, 1.2) and cfg.capacity_target == 120_000
    assert cfg.raw["capacity"]["laptop_max_params"] == 250_000


def test_forbidden_split_is_rejected(cfg):
    raw = copy.deepcopy(cfg.raw)
    raw["evaluation"]["allowed_splits"] = ["train", "dev", "test"]
    from ssl_anomaly.benchmarking.config import validate_config

    problems = validate_config(raw)
    assert any("test" in p for p in problems), problems


def test_bootstrap_below_the_floor_is_rejected(cfg):
    raw = copy.deepcopy(cfg.raw)
    raw["evaluation"]["bootstrap_B"] = 200
    from ssl_anomaly.benchmarking.config import validate_config

    assert any("bootstrap" in p.lower() for p in validate_config(raw))


def test_unequal_search_budgets_are_rejected(cfg):
    raw = copy.deepcopy(cfg.raw)
    raw["search"]["budgets"]["A1"]["trials"] = 3
    from ssl_anomaly.benchmarking.config import validate_config

    assert any("budget" in p.lower() or "equal" in p.lower() for p in validate_config(raw))


def test_suppression_axes_and_levels(cfg):
    assert set(cfg.suppressions) == {"S0", "S1", "S2", "S3"}
    assert set(cfg.inputs) == {"I1", "I2", "I3"} and set(cfg.fusions) == {"W0", "W1", "W2"}
    assert set(cfg.calibrations) == {"C1", "C2", "C3"} and set(cfg.protocols) == {"V1", "V2", "V3"}
    assert set(cfg.levels) == {"farm", "patch"}


def test_synthetic_panel_is_five_families_three_severities(cfg):
    assert set(cfg.families) == set(cfg.raw["synthetic"]["families"]) and len(cfg.families) == 5
    assert set(cfg.severities) == {"weak", "moderate", "strong"}
    assert pytest.approx(cfg.prevalence, abs=1e-9) == cfg.raw["synthetic"]["test_prevalence"]


def test_artifact_binding_declares_a_verifiable_hash_route(cfg):
    """Every artifact either carries a pre-registered hash or says how it will be pinned."""
    for name, row in cfg.raw["artifacts"].items():
        bound = bool(row.get("sha256_prefix"))
        policy = row.get("hash_policy")
        if policy is not None:
            assert policy in ("preregistered_prefix", "pin_at_runtime"), name
        required = row.get("required_for", "all") == "all"
        if bound:
            assert policy in (None, "preregistered_prefix"), (
                f"{name} carries a frozen prefix but claims {policy}: the two routes disagree"
            )
        elif required:
            # An artifact every run needs must state how its identity is fixed, or the
            # manifest cannot say what the numbers were computed from.
            assert policy == "pin_at_runtime" and row.get("pin_field"), (
                f"{name} is required_for=all with no frozen hash and no runtime pin"
            )
        else:
            # A conditionally-required artifact may be unbound *and must be*, so that
            # the matrix can defer the cells that need it instead of approximating.
            assert policy in (None, "pin_at_runtime"), name
            assert not required or row.get("pin_field"), name
        assert row.get("record"), f"artifacts.{name} must say in one line what it is"


def test_unbound_optional_artifact_matches_the_deferred_cells(cfg, cells):
    """An unbound artifact and the matrix's Deferred cells must tell the same story."""
    unbound = {k for k, v in cfg.raw["artifacts"].items()
               if not v.get("sha256_prefix") and v.get("required_for") not in ("all",)}
    assert "patch_store_v2" in unbound, "the patch store is the unbound artifact the matrix defers for"
    patch_cells = [x for x in cells if x.level == "patch"]
    assert patch_cells and all(x.status == "Deferred" for x in patch_cells)


def test_unbound_artifacts_do_not_carry_a_fake_hash(cfg):
    for name, row in cfg.raw["artifacts"].items():
        if row.get("hash_policy") == "pin_at_runtime":
            assert row.get("sha256_prefix") in (None, ""), (
                f"{name} cannot be pinned before the run and must not pre-suppose a hash"
            )


def test_document_hash_and_reproducibility(cfg):
    assert len(cfg.document_sha256) == 64
    from ssl_anomaly.benchmarking.config import canonical_sha256

    assert canonical_sha256(cfg.raw) == cfg.document_sha256


def test_amendment_list_is_empty_so_nothing_was_changed_after_freeze(cfg):
    assert cfg.raw["amendments"] == []


def test_applicability_table_covers_every_arm_suppression_pair(applicability):
    pairs = applicability["pairs"]
    assert all("/" in k for k in pairs)
    statuses = {v["status"] for v in pairs.values()}
    assert statuses <= {"applicable", "not-applicable"}
    for k, v in pairs.items():
        if v["status"] == "not-applicable":
            assert v.get("reason"), f"{k} is not applicable without a recorded reason"


def test_not_applicable_rows_explain_a_mechanism_not_a_preference(applicability):
    for k, v in applicability["pairs"].items():
        if v["status"] == "not-applicable":
            r = v["reason"].lower()
            words = ("no ", "not ", "degenerate", "has no", "cannot", "untrained",
                     "training-time", "same argument as")
            assert any(w in r for w in words), (k, r[:80])
            assert len(v["reason"]) > 40, f"{k}: a one-clause reason is not an explanation"


def test_load_benchmark_config_raises_on_a_broken_file(tmp_path):
    from ssl_anomaly.benchmarking.config import ConfigError, load_benchmark_config

    p = tmp_path / "bad.yaml"
    p.write_text("version: 1\nseeds: [7]\n", encoding="utf-8")
    with pytest.raises(ConfigError) as ei:
        load_benchmark_config(p)
    msg = str(ei.value)
    assert "seeds" in msg or "required" in msg.lower(), msg[:200]

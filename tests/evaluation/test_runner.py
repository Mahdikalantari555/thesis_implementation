"""Runner/orchestration tests: manifests, smoke isolation, fixture properties."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ssl_anomaly.evaluation.runner import (
    REPORTS_DIR,
    SMOKE_REPORTS_DIR,
    RunPaths,
    SmokeExpectation,
    build_manifest,
    environment_block,
    is_smoke,
    make_smoke_fixture,
    write_run_outputs,
)

# ---------------------------------------------------------------------------
# Smoke detection and path isolation
# ---------------------------------------------------------------------------


def test_is_smoke_recognises_the_tag():
    assert is_smoke("smoke") and is_smoke("smoke-1")
    assert not is_smoke(None) and not is_smoke("real")


def test_smoke_writes_to_its_own_directory():
    p = RunPaths.for_run("smoke-pretraining")
    assert p.smoke is True
    assert SMOKE_REPORTS_DIR in p.out_dir.parents
    assert "smoke-pretraining" in str(p.out_dir)


def test_real_run_writes_to_the_reports_directory():
    p = RunPaths.for_run(None)
    assert p.smoke is False
    assert p.out_dir == REPORTS_DIR
    assert Path(str(REPORTS_DIR)) == p.out_dir


def test_smoke_never_overwrites_real_reports():
    """Smoke output must not live inside the artifact directory a reader browses."""
    smoke = RunPaths.for_run("smoke-x")
    real = RunPaths.for_run()
    assert smoke.out_dir != real.out_dir
    assert REPORTS_DIR not in smoke.out_dir.parents, (
        "smoke artifacts nested under the real reports directory could be mistaken "
        "for evidence"
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_carries_the_preregistration_hash():
    m = build_manifest(stage="t", smoke=True, config={}, seeds=[13])
    assert len(m.prereg_sha256) == 64


def test_manifest_records_provenance_and_git():
    m = build_manifest(stage="t", smoke=True, config={"a": 1}, seeds=[13, 42])
    d = m.to_dict()
    assert d["seeds"] == [13, 42]
    assert d["config"] == {"a": 1}
    assert "git_sha" in d["provenance"]
    assert d["provenance"]["versions"]


def test_manifest_states_smoke_has_no_evidence():
    m = build_manifest(stage="t", smoke=True, config={}, seeds=[13])
    assert "NO quotable" in m.to_dict()["note"]


def test_real_manifest_points_at_the_gates():
    m = build_manifest(stage="t", smoke=False, config={}, seeds=[13])
    assert "gates block" in m.to_dict()["note"]


def test_manifest_includes_binding_block():
    m = build_manifest(stage="t", smoke=True, config={}, seeds=[13])
    assert m.to_dict()["binding"] is not None


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def test_write_outputs_creates_manifest_and_reports(tmp_path):
    m = build_manifest(stage="t", smoke=True, config={}, seeds=[13])
    paths = RunPaths(tmp_path / "run", smoke=True)
    written = write_run_outputs(paths=paths, manifest=m, reports={"alpha": {"x": 1}})
    assert (tmp_path / "run" / "run_manifest.json").exists()
    on_disk = json.loads((tmp_path / "run" / "alpha.json").read_text())
    assert on_disk == {"x": 1}
    assert set(written) == {"manifest", "alpha"}


def test_nan_is_refused_when_writing(tmp_path):
    """A NaN is a missing decision; it must never reach disk as a value."""
    m = build_manifest(stage="t", smoke=True, config={}, seeds=[13])
    paths = RunPaths(tmp_path / "run", smoke=True)
    with pytest.raises(ValueError):
        write_run_outputs(paths=paths, manifest=m,
                          reports={"bad": {"ap": float("nan")}})


# ---------------------------------------------------------------------------
# Smoke fixture
# ---------------------------------------------------------------------------


def test_fixture_labelled_source():
    f = make_smoke_fixture()
    d = f.describe()
    assert d["source"] == "synthetic-fixture"
    assert "no empirical meaning" in d["warning"]


def test_fixture_has_recoverable_signal():
    """The anomaly is a real NDVI depression: mean NDVI must differ by label."""
    f = make_smoke_fixture()
    ndvi = f.dev.X[:, :, 0]
    anom = ndvi[f.labels == 1]
    norm = ndvi[f.labels == 0]
    assert np.nanmean(anom) < np.nanmean(norm) - 1.0


def test_fixture_mask_stays_consistent():
    f = make_smoke_fixture()
    assert np.all(np.isfinite(f.dev.X[f.dev.M]))
    assert np.all(np.isnan(f.dev.X[~f.dev.M]))


def test_fixture_is_deterministic():
    a = make_smoke_fixture(seed=5)
    b = make_smoke_fixture(seed=5)
    assert np.array_equal(a.labels, b.labels)
    assert np.array_equal(np.nan_to_num(a.dev.X), np.nan_to_num(b.dev.X))


def test_fixture_has_both_classes():
    f = make_smoke_fixture()
    assert 0 < f.labels.mean() < 1


def test_fixture_supplies_field_level_agronomy():
    f = make_smoke_fixture(n_fields=6)
    assert f.farm_phase.size == 6
    assert f.farm_crop.size == 6


# ---------------------------------------------------------------------------
# Smoke expectation
# ---------------------------------------------------------------------------


def test_expectation_ok_when_all_checks_pass():
    e = SmokeExpectation("x")
    e.check("a", True)
    e.check("b", 1 == 1)
    assert e.ok
    e.raise_if_failed()


def test_expectation_raises_with_detail_on_failure():
    e = SmokeExpectation("x")
    e.check("ap above chance", False, "ap=0.11 prev=0.25")
    assert not e.ok
    with pytest.raises(AssertionError, match="ap above chance"):
        e.raise_if_failed()


def test_expectation_serialises():
    e = SmokeExpectation("x")
    e.check("a", True, "ok")
    d = e.to_dict()
    assert d["ok"] is True
    assert d["checks"][0]["label"] == "a"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def test_environment_block_records_core_versions():
    env = environment_block()
    assert env["python"]
    assert "numpy" in env and env["numpy"][0].isdigit()


def test_fixture_label_requires_field_context_not_a_global_threshold():
    """The regression guard for a fixture that once made A6 trivially perfect.

    The first version depressed anomalous cycles by a constant, so a probe on the
    per-cycle mean scored AP 1.000 and the 'transfer curve' measured nothing but
    averaging. The label must be inseparable by any global threshold and only
    recoverable relative to the cycle's own field.
    """
    f = make_smoke_fixture(n_fields=14, cycles_per_field=4, seed=13)
    ndvi = np.where(f.dev.M[:, :, 0], f.dev.X[:, :, 0], np.nan)
    per_cycle = np.nanmean(ndvi, axis=1)
    normal, anom = per_cycle[f.labels == 0], per_cycle[f.labels == 1]

    # No global threshold separates the classes.
    assert not (anom.max() < normal.min()), (
        "a global NDVI threshold separates the classes: the fixture does not "
        "require field context"
    )
    assert not (normal.max() < anom.min())

    # The field-relative deviation does separate them.
    rel = np.empty_like(per_cycle)
    for fid in np.unique(f.fields):
        sel = f.fields == fid
        rel[sel] = per_cycle[sel] - np.median(per_cycle[sel])
    assert rel[f.labels == 1].mean() < rel[f.labels == 0].mean() - 1.0


def test_fixture_agronomy_arrays_always_align_with_fields():
    """A truncated crop array broke the retrieval track's one-row-per-field contract."""
    for n in (4, 6, 8, 16, 20):
        f = make_smoke_fixture(n_fields=n)
        assert f.farm_phase.size == n, f"n_fields={n}: phase has {f.farm_phase.size}"
        assert f.farm_crop.size == n, f"n_fields={n}: crop has {f.farm_crop.size}"


def test_every_fixture_field_has_an_agronomic_partner():
    """Track B is undefined if any field lacks a partner, so the fixture must not."""
    for n in (4, 8, 16):
        f = make_smoke_fixture(n_fields=n)
        for i in range(n):
            partners = [
                j for j in range(n)
                if j != i
                and f.farm_crop[i] == f.farm_crop[j]
                and abs(f.farm_phase[i] - f.farm_phase[j]) <= 2
            ]
            assert partners, f"n_fields={n}: field {i} has no agronomic partner"

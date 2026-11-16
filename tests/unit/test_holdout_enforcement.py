"""Tests for fail-closed holdout enforcement (change training-holdout-invariant,
tasks 2.1 and spec "Fail-closed holdout enforcement in every training path").

The invariant: no training consumer may see a holdout farm or a holdout-period
cycle, and the failure must be a named error listing the offending pairs — never
a warning.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ssl_anomaly.monitor import holdout as mholdout  # noqa: E402


@pytest.fixture
def manifest() -> dict:
    splits = {
        "created": "2026-11-14",
        "test_period_start": "2024-01-01",
        "test_fields": [101, 102],
        "dev_fields": [201],
        "train_fields": [301, 302],
        "monitoring_only_fields": [401],
        "cycles": {},
    }
    splits["manifest_sha256"] = "f" * 64
    return mholdout.build_holdout_manifest(splits, created="2026-11-16")


# ---------------------------------------------------------------------------
# train_only_mask
# ---------------------------------------------------------------------------
def test_mask_accepts_clean_train_inputs(manifest: dict) -> None:
    mask = mholdout.train_only_mask(
        [301, 302], ["2022-03-01", "2022-05-01"], manifest=manifest,
    )
    assert mask.all()


def test_mask_rejects_holdout_farm(manifest: dict) -> None:
    mask = mholdout.train_only_mask([301, 101], manifest=manifest)
    assert list(mask) == [True, False]


def test_mask_rejects_future_unknown_farm(manifest: dict) -> None:
    """A farm registered after the freeze is holdout by default."""
    mask = mholdout.train_only_mask([301, 777], manifest=manifest)
    assert list(mask) == [True, False]


def test_mask_rejects_holdout_period(manifest: dict) -> None:
    mask = mholdout.train_only_mask(
        [301, 302], ["2022-06-01", "2024-02-01"], manifest=manifest,
    )
    assert list(mask) == [True, False]


def test_mask_accepts_cycle_starting_before_cutoff_only(manifest: dict) -> None:
    mask = mholdout.train_only_mask(
        [301, 302], ["2023-12-31", "2024-01-01"], manifest=manifest,
    )
    assert list(mask) == [True, False], "the cutoff is inclusive on the holdout side"


def test_mask_length_mismatch_raises(manifest: dict) -> None:
    with pytest.raises(ValueError, match="length"):
        mholdout.train_only_mask([301, 302], ["2022-01-01"], manifest=manifest)


# ---------------------------------------------------------------------------
# assert_train_only — the fail-closed proof
# ---------------------------------------------------------------------------
def test_assert_passes_clean_input(manifest: dict) -> None:
    mholdout.assert_train_only(
        [301, 302], ["301:1", "302:1"], ["2022-01-01", "2022-03-01"],
        manifest=manifest, context="unit-test",
    )


def test_assert_raises_named_error_for_holdout_farm(manifest: dict) -> None:
    with pytest.raises(mholdout.HoldoutViolationError) as excinfo:
        mholdout.assert_train_only(
            [301, 101], ["301:1", "101:7"], ["2022-01-01", "2020-04-01"],
            manifest=manifest, context="unit-test-dataset",
        )
    msg = str(excinfo.value)
    assert "101:7" in msg, "the offending (field, cycle) pair must be named"
    assert "holdout farm" in msg
    assert "unit-test-dataset" in msg, "the calling training path must be named"
    assert "aborted before the first batch" in msg


def test_assert_raises_for_holdout_period(manifest: dict) -> None:
    with pytest.raises(mholdout.HoldoutViolationError, match="holdout period"):
        mholdout.assert_train_only(
            [301], ["301:9"], ["2024-06-01"], manifest=manifest,
        )


def test_assert_raises_for_unknown_future_farm(manifest: dict) -> None:
    with pytest.raises(mholdout.HoldoutViolationError, match="future-by-default"):
        mholdout.assert_train_only([888], ["888:1"], manifest=manifest)


def test_assert_lists_every_violation_up_to_the_cap(manifest: dict) -> None:
    fields = [101, 102] + [900 + i for i in range(10)]
    keys = [f"{f}:1" for f in fields]
    with pytest.raises(mholdout.HoldoutViolationError) as excinfo:
        mholdout.assert_train_only(fields, keys, manifest=manifest, max_listed=4)
    msg = str(excinfo.value)
    assert msg.count(":1 (") == 4
    assert "+8 more" in msg
    assert "12 training input(s)" in msg


def test_assert_accepts_cycle_keys_only(manifest: dict) -> None:
    """Field ids may be omitted when cycle keys carry them."""
    mholdout.assert_train_only(
        None, ["301:1", "302:2"], manifest=manifest,
    )
    with pytest.raises(mholdout.HoldoutViolationError, match="101:1"):
        mholdout.assert_train_only(None, ["301:1", "101:1"], manifest=manifest)


def test_assert_requires_some_identifier(manifest: dict) -> None:
    with pytest.raises(ValueError, match="field_ids or cycle_keys"):
        mholdout.assert_train_only(manifest=manifest)


def test_assert_rejects_mismatched_key_count(manifest: dict) -> None:
    with pytest.raises(ValueError, match="cycle_keys length"):
        mholdout.assert_train_only([301], ["301:1", "302:1"], manifest=manifest)


# ---------------------------------------------------------------------------
# filter + guard (the shared Dataset base)
# ---------------------------------------------------------------------------
def test_filter_removes_holdout_and_asserting_the_result_passes(manifest: dict) -> None:
    fields = [301, 101, 302, 777]
    keys = ["301:1", "101:1", "302:1", "777:1"]
    starts = ["2022-01-01"] * 4
    kept = mholdout.filter_train_only(fields, keys, starts, manifest=manifest)
    assert kept == ["301:1", "302:1"]
    # Proving the filter worked — the enforcement point is the assertion.
    mholdout.assert_train_only(None, kept, ["2022-01-01", "2022-01-01"], manifest=manifest)


def test_holdout_guard_filters_then_asserts(manifest: dict) -> None:
    guard = mholdout.HoldoutGuard(manifest=manifest, context="patch-dataset")
    fields = [301, 101]
    keys = ["301:1", "101:1"]
    kept = guard.filter(fields, keys, ["2022-01-01", "2022-01-01"])
    assert kept == ["301:1"]
    guard.check(None, kept, ["2022-01-01"])  # clean input passes
    with pytest.raises(mholdout.HoldoutViolationError, match="patch-dataset"):
        guard.check(fields, keys, ["2022-01-01", "2022-01-01"])


def test_holdout_guard_records_manifest_sha(manifest: dict) -> None:
    guard = mholdout.HoldoutGuard(manifest=manifest)
    assert guard.holdout_manifest_sha256 == manifest["manifest_sha256"]


# ---------------------------------------------------------------------------
# train_main.py wiring (task 2.3) — asserted statically, no data needed
# ---------------------------------------------------------------------------
def test_train_main_asserts_holdout_before_first_batch() -> None:
    source = (REPO_ROOT / "scripts/train_main.py").read_text(encoding="utf-8")
    assert "mholdout.load_holdout_manifest()" in source
    assert "mholdout.assert_train_only(" in source
    # The assertion must precede the normalization fit (i.e. the first data use).
    assert source.index("assert_train_only(") < source.index("fit_robust_norm(")
    assert "holdout_manifest_sha256" in source, "manifest sha must reach the run manifest"


def test_build_holdout_manifest_script_is_dry_run_safe() -> None:
    """--dry-run must not write anything (it is how the owner inspects the sets)."""
    source = (REPO_ROOT / "scripts/build_holdout_manifest.py").read_text(encoding="utf-8")
    assert "--dry-run" in source
    assert "nothing written" in source
    assert "--force requires a non-empty --reason" in source or "requires --reason" in source


# ---------------------------------------------------------------------------
# build_holdout_manifest.py CLI behaviour (task 1.2)
# ---------------------------------------------------------------------------
def _import_build_script():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_holdout_manifest", REPO_ROOT / "scripts/build_holdout_manifest.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _patch_loader(monkeypatch, mod, splits_manifest: dict) -> None:
    """Serve the synthetic split manifest instead of the (absent) real data."""
    def fake_loader(**kwargs):
        return splits_manifest

    monkeypatch.setattr(mod.msplits, "load_splits_locked", fake_loader)


def test_cli_writes_then_refuses_then_force_records_reason(
    monkeypatch, tmp_path: Path,
) -> None:
    mod = _import_build_script()
    splits = {
        "created": "2026-11-14",
        "test_period_start": "2024-01-01",
        "test_fields": [101, 102],
        "dev_fields": [201],
        "train_fields": [301],
        "monitoring_only_fields": [401],
        "cycles": {},
        "manifest_sha256": "f" * 64,
    }
    _patch_loader(monkeypatch, mod, splits)
    out = tmp_path / "holdout_manifest.json"

    assert mod.main(["--out", str(out)]) == 0
    assert out.exists()

    # Second write without --force fails with a clear code and message.
    assert mod.main(["--out", str(out)]) == 2

    # --force without --reason is refused.
    assert mod.main(["--out", str(out), "--force"]) == 2

    # --force with a reason succeeds and records it.
    assert mod.main(["--out", str(out), "--force", "--reason", "unit smoke"]) == 0
    rebuilt = __import__("json").loads(out.read_text(encoding="utf-8"))
    assert rebuilt["rebuild_of"]["reason"] == "unit smoke"
    assert mholdout.load_holdout_manifest(path=out)["manifest_sha256"] == rebuilt["manifest_sha256"]


def test_cli_dry_run_writes_nothing(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = _import_build_script()
    splits = {
        "created": "2026-11-14",
        "test_period_start": "2024-01-01",
        "test_fields": [101],
        "train_fields": [301],
        "cycles": {},
        "manifest_sha256": "f" * 64,
    }
    _patch_loader(monkeypatch, mod, splits)
    out = tmp_path / "holdout_manifest.json"
    assert mod.main(["--out", str(out), "--dry-run"]) == 0
    assert not out.exists(), "--dry-run must not write"
    assert "nothing written" in capsys.readouterr().out


def test_cli_skip_cycle_check_requires_dry_run(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = _import_build_script()
    _patch_loader(monkeypatch, mod, {"test_fields": [1], "test_period_start": "2024-01-01"})
    rc = mod.main(["--out", str(tmp_path / "x.json"), "--skip-cycle-file-check"])
    assert rc == 2
    assert "--skip-cycle-file-check is only allowed with --dry-run" in capsys.readouterr().err


def test_cli_reports_missing_splits_without_traceback(monkeypatch, tmp_path: Path, capsys) -> None:
    mod = _import_build_script()

    def boom(**kwargs):
        raise RuntimeError("splits_locked.json missing")

    monkeypatch.setattr(mod.msplits, "load_splits_locked", boom)
    assert mod.main(["--out", str(tmp_path / "x.json")]) == 2
    err = capsys.readouterr().err
    assert "cannot load the locked splits" in err

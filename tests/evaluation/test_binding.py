"""Pre-registration + artifact binding gate (change tasks 0.1-0.4).

These tests are the enforcement of two spec requirements:

- ``representation-pretraining-quality`` / "Development-only pretraining
  evidence" — "Split manifest drift … THEN the pretraining evaluation fails
  closed and reports the mismatch".
- ``generalization-and-claim-semantics`` / "Preservation of the frozen evidence
  base" — "Artifact mutation detected … THEN the run fails closed".

They also pin the pre-registration content: gates, hypotheses, fractions and
seeds may not drift silently.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ssl_anomaly.evaluation.binding import (
    DOCUMENTED_V2_HASHES,
    PREREGISTRATION_PATH,
    ArtifactBindingError,
    bind_artifacts,
    canonical_json_sha256,
    documented_hash,
    is_sha_prefix,
    load_preregistration,
    preregistration_sha256,
    sha256_file,
    sha256_prefix,
)


# ---------------------------------------------------------------------------
# The committed pre-registration
# ---------------------------------------------------------------------------


def test_committed_preregistration_is_valid():
    prereg = load_preregistration()
    assert prereg.seeds == (13, 42, 2026)
    assert prereg.label_fractions == (0.05, 0.10, 0.25, 0.50, 1.00)
    assert set(prereg.gates) == {"G-A", "G-B", "G-C", "G-D", "G-E"}
    assert set(prereg.hypotheses) == {"H1", "H2", "H3", "H4", "H5"}


def test_committed_preregistration_pins_the_v2_chain():
    prereg = load_preregistration()
    # The three artifacts the capstone binds before any run.
    for name in ("cycles_490_v2", "splits_locked_v2", "holdout_manifest_v2", "norm_stats_full_v2"):
        assert name in prereg.artifacts, f"{name} must be pre-registered"
    assert prereg.expected_hash("splits_locked_v2") == "30ecbe10"
    assert prereg.expected_hash("holdout_manifest_v2") == "d501bf81"
    assert prereg.expected_hash("norm_stats_full_v2") == "493f0395"


def test_preregistration_hash_matches_documented_artifacts():
    """The code-side hash table and the YAML must not drift apart."""
    prereg = load_preregistration()
    for name, expected in DOCUMENTED_V2_HASHES.items():
        assert documented_hash(name) == expected
        assert prereg.expected_hash(name) == expected, f"{name} drifted in the YAML"


def test_preregistration_does_not_use_the_v1_norm():
    """Finding 25 binds v2 to 493f0395; the v1 PoC norm must not be the freeze."""
    prereg = load_preregistration()
    assert prereg.expected_hash("norm_stats_full_v2") != "3cb1ca60"


def test_every_gate_has_a_fail_consequence():
    prereg = load_preregistration()
    for gid, gate in prereg.gates.items():
        assert gate["criterion"].strip(), gid
        assert gate["fail_consequence"].strip(), gid
        assert gate["inputs"], gid


def test_hypotheses_reference_existing_gates():
    prereg = load_preregistration()
    for hid, hyp in prereg.hypotheses.items():
        assert hyp["gate"] in prereg.gates, hid


def test_evaluation_block_forbids_the_test_split():
    payload = load_preregistration().payload
    ev = payload["evaluation"]
    assert "test" in ev["forbidden_splits"]
    assert "test" not in ev["allowed_splits"]
    assert ev["cycle_far_target"] == 0.05


def test_tasks_mark_exploratory_artifacts():
    tasks = load_preregistration().payload["tasks"]
    assert tasks["T3_track2"]["claim_bearing"] is False
    assert tasks["T3_near_matches"]["claim_bearing"] is False
    assert tasks["T5"]["claim_bearing"] is False
    for key in ("T1", "T2", "T3_track1", "T4"):
        assert tasks[key]["claim_bearing"] is True, key


def test_preregistration_document_hash_is_stable_and_recorded():
    h1 = preregistration_sha256()
    h2 = sha256_file(PREREGISTRATION_PATH)
    assert h1 == h2 and len(h1) == 64
    manifest = load_preregistration().to_manifest()
    assert manifest["preregistration_sha256"] == h1
    assert manifest["gate_ids"] == ["G-A", "G-B", "G-C", "G-D", "G-E"]


def test_amendments_block_exists_and_is_a_list():
    payload = load_preregistration().payload
    assert isinstance(payload["amendments"], list)


# ---------------------------------------------------------------------------
# Malformed pre-registrations fail loudly
# ---------------------------------------------------------------------------


def _write_variant(tmp_path: Path, mutate) -> Path:
    payload = yaml.safe_load(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    out = tmp_path / "pre_registration.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def test_missing_gate_is_rejected(tmp_path):
    def mutate(p):
        del p["gates"]["G-D"]

    with pytest.raises(ArtifactBindingError, match="gates must be exactly"):
        load_preregistration(_write_variant(tmp_path, mutate))


def test_weakened_gate_without_criterion_is_rejected(tmp_path):
    def mutate(p):
        p["gates"]["G-C"]["criterion"] = ""

    with pytest.raises(ArtifactBindingError, match="lacks 'criterion'"):
        load_preregistration(_write_variant(tmp_path, mutate))


def test_wrong_label_fractions_are_rejected(tmp_path):
    def mutate(p):
        p["label_fractions"] = [0.1, 1.0]

    with pytest.raises(ArtifactBindingError, match="5/10/25/50/100"):
        load_preregistration(_write_variant(tmp_path, mutate))


def test_wrong_seeds_are_rejected(tmp_path):
    def mutate(p):
        p["seeds"] = [1, 2, 3]

    with pytest.raises(ArtifactBindingError, match="seeds must be"):
        load_preregistration(_write_variant(tmp_path, mutate))


def test_missing_preregistration_file_is_rejected(tmp_path):
    with pytest.raises(ArtifactBindingError, match="not found"):
        load_preregistration(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# Hash helpers
# ---------------------------------------------------------------------------


def test_sha256_file_streams_and_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "blob.bin"
    data = b"x" * 5000 + b"tail"
    p.write_bytes(data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()
    assert sha256_prefix(sha256_file(p)) == hashlib.sha256(data).hexdigest()[:8]


def test_sha256_path_hashes_a_directory_store_deterministically(tmp_path):
    # patch_store_v3 is a .zarr DIRECTORY store: sha256_file raised
    # IsADirectoryError on any machine where the store exists, failing every
    # real manifest build (tests/evaluation/test_runner.py
    # ::test_real_manifest_points_at_the_gates). The directory digest must be
    # deterministic, name-sensitive and content-sensitive.
    from ssl_anomaly.evaluation.binding import sha256_path

    store = tmp_path / "patches.zarr"
    (store / "0.0").mkdir(parents=True)
    (store / "0.0" / "a").write_bytes(b"one")
    (store / "0.0" / "b").write_bytes(b"two")
    (store / ".zattrs").write_text("{}", encoding="utf-8")

    d1 = sha256_path(store)
    assert d1 == sha256_path(store), "digest must be deterministic across calls"
    assert d1 != sha256_path(store / "0.0" / "a"), "not a single file's hash"

    # Renaming a chunk changes the digest (path-sensitive).
    (store / "0.0" / "b").rename(store / "0.0" / "c")
    d2 = sha256_path(store)
    assert d2 != d1

    # Rewriting a chunk with the same name changes the digest (content-sensitive).
    (store / "0.0" / "c").write_bytes(b"two!")
    assert sha256_path(store) != d2


def test_directory_store_binds_and_fails_closed_like_a_file(tmp_path):
    from ssl_anomaly.evaluation.binding import sha256_path

    store = tmp_path / "store.zarr"
    (store / "chunks").mkdir(parents=True)
    (store / "chunks" / "0").write_bytes(b"payload")
    digest = sha256_path(store)
    pre = _prereg_with(tmp_path, {
        "patch_store_v3": {"path": "store.zarr", "sha256_prefix": digest[:8]},
    })
    report = bind_artifacts(prereg=load_preregistration(pre), base_dir=tmp_path)
    assert report.ok
    bound = [a for a in report.bound if a.name == "patch_store_v3"]
    assert bound and bound[0].actual_sha256 == digest

    # Mutation of one chunk must fail closed, exactly like a file artifact.
    (store / "chunks" / "0").write_bytes(b"tampered")
    pre_bad = _prereg_with(tmp_path, {
        "patch_store_v3": {"path": "store.zarr", "sha256_prefix": digest[:8]},
    })
    with pytest.raises(ArtifactBindingError, match="deviates"):
        bind_artifacts(prereg=load_preregistration(pre_bad), base_dir=tmp_path, strict=True)


def test_empty_directory_store_is_reported_not_hashed(tmp_path):
    from ssl_anomaly.evaluation.binding import sha256_path

    empty = tmp_path / "empty.zarr"
    empty.mkdir()
    with pytest.raises(ArtifactBindingError, match="no files to hash"):
        sha256_path(empty)


def test_canonical_json_hash_is_order_independent():
    a = canonical_json_sha256({"b": 2, "a": 1})
    b = canonical_json_sha256({"a": 1, "b": 2})
    assert a == b


def test_canonical_json_hash_refuses_nan():
    with pytest.raises(ValueError):
        canonical_json_sha256({"metric": float("nan")})


def test_is_sha_prefix():
    assert is_sha_prefix("30ecbe10")
    assert is_sha_prefix("a" * 64)
    assert not is_sha_prefix("not-a-hash")
    assert not is_sha_prefix("30EC")  # too short


# ---------------------------------------------------------------------------
# bind_artifacts
# ---------------------------------------------------------------------------


def _prereg_with(tmp_path: Path, artifacts: dict) -> Path:
    payload = yaml.safe_load(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
    payload["artifacts"] = artifacts
    out = tmp_path / "pre.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return out


def test_absent_artifacts_are_reported_not_substituted(tmp_path):
    pre = _prereg_with(tmp_path, {
        "splits_locked_v2": {"path": "missing/splits.json", "sha256_prefix": "30ecbe10"},
    })
    report = bind_artifacts(prereg=load_preregistration(pre), base_dir=tmp_path, strict=True)
    assert report.ok
    file_absent = [a for a in report.absent if not a.path.startswith("<code:")]
    file_bound = [a for a in report.bound if not a.path.startswith("<code:")]
    assert len(file_absent) == 1
    assert file_bound == []
    assert "Deferred" in report.note or "absent" in report.note


def test_bound_artifact_matches_prefix(tmp_path):
    # cycles_490_v2 is byte-mode in the real pre-registration (sha recorded at
    # freeze time is impossible for a file that carries its own hash); the
    # manifest-mode artifacts (splits/holdout/norm) are covered below.
    target = tmp_path / "cycles.parquet"
    target.write_text("{}", encoding="utf-8")
    digest = sha256_file(target)
    pre = _prereg_with(tmp_path, {
        "cycles_490_v2": {"path": "cycles.parquet", "sha256_prefix": digest[:8]},
    })
    report = bind_artifacts(prereg=load_preregistration(pre), base_dir=tmp_path)
    assert report.ok
    file_bound = [a for a in report.bound if not a.path.startswith("<code:")]
    assert len(file_bound) == 1
    assert file_bound[0].actual_sha256 == digest


def test_manifest_mode_artifact_binds_by_recomputed_self_hash(tmp_path):
    # splits_locked_v2's frozen prefix names its OWN manifest_sha256 field,
    # which the binder RECOMPUTES from the contents (a file cannot contain its
    # own byte hash, so byte-mode binding for these artifacts would fail
    # closed on every machine where they exist).
    from ssl_anomaly.monitor.splits import manifest_hash

    manifest = {
        "version": 2,
        "created": "2026-11-23",
        "counts": {"train": 2105, "dev": 340, "test": 773},
        "cycles": {"1": ["c1", "c2"]},
    }
    manifest["manifest_sha256"] = manifest_hash(manifest)
    target = tmp_path / "splits_locked_v2.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    pre = _prereg_with(tmp_path, {
        "splits_locked_v2": {
            "path": "splits_locked_v2.json",
            "sha256_prefix": manifest["manifest_sha256"][:8],
        },
    })
    report = bind_artifacts(prereg=load_preregistration(pre), base_dir=tmp_path)
    assert report.ok
    bound = [a for a in report.bound if a.name == "splits_locked_v2"]
    assert bound and bound[0].actual_sha256 == manifest["manifest_sha256"]
    assert "recomputed" in bound[0].detail


def test_manifest_mode_artifact_fails_closed_when_tampered(tmp_path):
    from ssl_anomaly.monitor.splits import manifest_hash

    manifest = {"version": 2, "counts": {"train": 2105}}
    manifest["manifest_sha256"] = manifest_hash(manifest)
    target = tmp_path / "splits_locked_v2.json"
    target.write_text(json.dumps(manifest), encoding="utf-8")
    pre = _prereg_with(tmp_path, {
        "splits_locked_v2": {
            "path": "splits_locked_v2.json",
            "sha256_prefix": manifest["manifest_sha256"][:8],
        },
    })
    # Tamper AFTER computing the hash: recorded field and contents disagree.
    manifest["counts"]["train"] = 9999
    target.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ArtifactBindingError, match="deviates"):
        bind_artifacts(prereg=load_preregistration(pre), base_dir=tmp_path, strict=True)
    report = bind_artifacts(prereg=load_preregistration(pre), base_dir=tmp_path,
                            strict=False)
    assert report.mismatched[0].detail.startswith("self-inconsistent")


def test_manifest_mode_artifact_without_self_hash_is_mismatch(tmp_path):
    target = tmp_path / "norm_stats_full_v2.json"
    target.write_text('{"stat": {}}', encoding="utf-8")     # no self sha256 key
    pre = _prereg_with(tmp_path, {
        "norm_stats_full_v2": {"path": "norm_stats_full_v2.json",
                               "sha256_prefix": "493f0395"},
    })
    report = bind_artifacts(prereg=load_preregistration(pre), base_dir=tmp_path,
                            strict=False)
    assert not report.ok
    assert report.mismatched[0].detail.startswith("artifact carries no self-recorded")


def test_hash_mismatch_fails_closed(tmp_path):
    target = tmp_path / "splits.json"
    target.write_text("{}", encoding="utf-8")
    pre = _prereg_with(tmp_path, {
        "splits_locked_v2": {"path": "splits.json", "sha256_prefix": "deadbeef"},
    })
    with pytest.raises(ArtifactBindingError, match="deviates from the pre-registration"):
        bind_artifacts(prereg=load_preregistration(pre), base_dir=tmp_path, strict=True)


def test_hash_mismatch_reported_when_not_strict(tmp_path):
    target = tmp_path / "splits.json"
    target.write_text("{}", encoding="utf-8")
    pre = _prereg_with(tmp_path, {
        "splits_locked_v2": {"path": "splits.json", "sha256_prefix": "deadbeef"},
    })
    report = bind_artifacts(prereg=load_preregistration(pre), base_dir=tmp_path, strict=False)
    assert not report.ok
    assert len(report.mismatched) == 1
    assert report.to_dict()["n_mismatch"] == 1


def test_required_artifact_absent_raises(tmp_path):
    pre = _prereg_with(tmp_path, {
        "cycles_490_v2": {"path": "missing.parquet", "sha256_prefix": None},
    })
    with pytest.raises(ArtifactBindingError, match="required artifact\\(s\\) not bound"):
        bind_artifacts(
            prereg=load_preregistration(pre),
            base_dir=tmp_path,
            require=("cycles_490_v2",),
        )


def test_presence_only_probe_skips_hashing(tmp_path):
    target = tmp_path / "splits.json"
    target.write_text("{}", encoding="utf-8")
    pre = _prereg_with(tmp_path, {
        "splits_locked_v2": {"path": "splits.json", "sha256_prefix": "30ecbe10"},
    })
    report = bind_artifacts(
        prereg=load_preregistration(pre), base_dir=tmp_path, compute_hashes=False, strict=True
    )
    assert report.artifacts[0].status == "present-unverified"
    assert report.ok


def test_overrides_reroute_paths(tmp_path):
    alt = tmp_path / "elsewhere" / "cycles.parquet"
    alt.parent.mkdir()
    alt.write_text("{}", encoding="utf-8")
    digest = sha256_file(alt)
    pre = _prereg_with(tmp_path, {
        "cycles_490_v2": {"path": "nowhere.json", "sha256_prefix": digest[:8]},
    })
    report = bind_artifacts(
        prereg=load_preregistration(pre),
        base_dir=tmp_path,
        overrides={"cycles_490_v2": str(alt)},
    )
    file_bound = [a for a in report.bound if not a.path.startswith("<code:")]
    assert report.ok and len(file_bound) == 1


def test_report_serialises_without_nan(tmp_path):
    pre = _prereg_with(tmp_path, {
        "splits_locked_v2": {"path": "missing.json", "sha256_prefix": "30ecbe10"},
    })
    report = bind_artifacts(prereg=load_preregistration(pre), base_dir=tmp_path)
    blob = json.dumps(report.to_dict(), allow_nan=False)
    assert "splits_locked_v2" in blob


def test_clean_checkout_report_is_honest(monkeypatch, tmp_path):
    """In this checkout the v2 chain is absent; the report must say so plainly."""
    report = bind_artifacts(base_dir=tmp_path)
    prereg = load_preregistration()
    payload = report.to_dict()
    assert payload["ok"] is True          # no mismatch
    file_statuses = [a for a in report.artifacts if not a.path.startswith("<code:")]
    assert file_statuses, "the pre-registration must name file artifacts"
    assert all(a.status == "absent" for a in file_statuses)
    assert len(file_statuses) == len(prereg.artifacts)
    assert "clean checkout" in report.note
    # The code-derived severity table IS bindable in a clean checkout.
    code = [a for a in report.artifacts if a.path.startswith("<code:")]
    assert code and code[0].status == "bound", code

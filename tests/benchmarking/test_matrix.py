"""The committed applicability matrix: identity, coverage, and the honest states (Phase 12)."""

from __future__ import annotations

import pytest


def test_cell_count_and_status_vocabulary(cells):
    from ssl_anomaly.benchmarking.matrix import status_counts

    counts = status_counts(cells)
    assert sum(counts.values()) == len(cells)
    assert set(counts) <= {"Pending", "Running", "Completed", "Failed", "Interrupted", "Deferred",
                           "Not Applicable"}
    # an executed benchmark that has run nothing must still say what it will not run
    assert counts.get("Not Applicable", 0) > 0 and counts.get("Deferred", 0) > 0


def test_cell_ids_unique(cells):
    ids = [c.experiment_id for c in cells]
    assert len(set(ids)) == len(ids), "duplicate experiment ids would overwrite each other's results"
    hashes = [c.cell_hash for c in cells]
    assert len(set(hashes)) == len(hashes)


def test_every_cell_names_the_full_factor_set(cells):
    for c in cells:
        assert c.model_id and c.suppression and c.input and c.fusion and c.calibration
        assert c.protocol and c.level and c.seed in (13, 42, 2026), c


def test_patch_cells_are_deferred_with_a_reason(cells):
    patch = [c for c in cells if c.level == "patch"]
    assert patch, "the patch-level cells must exist in the matrix even while deferred"
    for c in patch:
        assert c.status == "Deferred"
        assert "patch" in c.reason and "unbound" in c.reason, c.reason


def test_not_applicable_cells_quote_the_applicability_table(cells, applicability):
    na = [c for c in cells if c.status == "Not Applicable"]
    assert na
    reasons = {v.get("reason") for v in applicability["pairs"].values() if v.get("reason")}
    for c in na:
        assert c.reason.startswith("applicability:") and c.reason[len("applicability: "):] in reasons, c


def test_jepa_s2_is_never_pending(cells):
    for c in cells:
        if c.model_id in ("A7", "A8") and c.suppression == "S2":
            assert c.status == "Not Applicable", "JEPA × S2 is degenerate and must not be run"


def test_prediction_masking_rows_record_context_only(applicability):
    """Masking on prediction arms is a context change, never a target change (design)."""
    for key, row in applicability["pairs"].items():
        model = key.split("/")[0]
        if model in ("A5", "A6") and key.endswith("/S2") and row.get("status") == "applicable":
            assert "context" in (row.get("note", "") + row.get("adapter_doc", "")).lower(), row


def test_priority_starts_with_the_screening_tier(cells):
    pending = sorted([c for c in cells if c.status == "Pending"], key=lambda c: c.priority)
    assert pending[0].tier == "screening", pending[0]


def test_untrained_reference_runs_all_seeds(cells):
    """A random-init control is only meaningful at the same seed coverage as its twin."""
    rand = [c for c in cells if c.model_id == "A-RAND"]
    assert rand and {c.seed for c in rand} == {13, 42, 2026}


def test_matrix_hash_depends_on_every_cell(cells, cfg):
    from ssl_anomaly.benchmarking.matrix import matrix_document, matrix_sha256

    a = matrix_sha256(matrix_document(cfg, cells, applicability_sha="0" * 64))
    b = matrix_sha256(matrix_document(cfg, cells[:-1], applicability_sha="0" * 64))
    c = matrix_sha256(matrix_document(cfg, cells, applicability_sha="1" * 64))
    assert a != b and a != c


def test_committed_matrix_matches_the_code(cfg, applicability, tmp_path):
    from ssl_anomaly.benchmarking.matrix import read_matrix, verify_committed_matrix, write_matrix
    from ssl_anomaly.benchmarking.matrix import matrix_document, build_matrix

    from ssl_anomaly.benchmarking.config import applicability_sha256_of

    app_sha = applicability_sha256_of(applicability)
    doc = matrix_document(cfg, build_matrix(cfg, applicability), applicability_sha=app_sha)
    p = write_matrix(doc, tmp_path / "matrix.yaml")
    back = read_matrix(p)
    assert back["n_cells"] == doc["n_cells"]
    assert back["matrix_sha256"] == doc["matrix_sha256"]
    text = p.read_text()
    assert text.count("\n") >= doc["n_cells"], "one line per cell keeps the committed matrix diffable"
    res = verify_committed_matrix(cfg, applicability, path=p)
    assert res["ok"], res
    assert doc["applicability_sha256"] == app_sha, (
        "the committed matrix must cite the hash the runner computes — an empty or invented value "
        "makes every run refuse to start against its own frozen matrix")


def test_a_stale_applicability_hash_is_caught(cfg, applicability, tmp_path):
    """The applicability table decides which cells exist, so its hash is part of the contract.

    A committed matrix that cites an older table must be refused, not "warned about":
    re-deriving cells silently is how a suppression row moving from not-applicable to
    applicable would enter a results table without ever being announced.
    """
    from ssl_anomaly.benchmarking.config import applicability_sha256_of
    from ssl_anomaly.benchmarking.matrix import matrix_document, read_matrix, verify_committed_matrix
    from ssl_anomaly.benchmarking.matrix import build_matrix, write_matrix

    doc = matrix_document(cfg, build_matrix(cfg, applicability),
                          applicability_sha="f" * 64)          # a stale pin, on purpose
    p = write_matrix(doc, tmp_path / "matrix.yaml")
    assert read_matrix(p)["applicability_sha256"] == "f" * 64
    res = verify_committed_matrix(cfg, applicability, path=p)
    assert not res["ok"], res
    assert any("applicability" in x for x in res["problems"]), res["problems"]
    assert applicability_sha256_of(applicability) != "f" * 64


def test_coverage_line_reports_the_missing_cells(cells):
    from ssl_anomaly.benchmarking.matrix import coverage_line, status_counts

    line = coverage_line(status_counts(cells), total=len(cells))
    assert "Deferred" in line and "Not Applicable" in line
    assert "Completed 0/" in line, line

"""The seven-state registry: legality, resumability, auditability (Phase 12)."""

from __future__ import annotations

import pytest

ALL = ("Pending", "Running", "Completed", "Failed", "Interrupted", "Deferred", "Not Applicable")


def test_exactly_the_seven_states():
    from ssl_anomaly.benchmarking.registry import REGISTRY_STATES

    assert tuple(REGISTRY_STATES) == ALL


@pytest.mark.parametrize(
    "src,dst,ok",
    [("Pending", "Running", True), ("Pending", "Completed", False),
     ("Running", "Completed", True), ("Running", "Interrupted", True),
     ("Interrupted", "Running", True), ("Completed", "Running", False),
     ("Failed", "Pending", True), ("Not Applicable", "Running", False),
     ("Deferred", "Pending", True), ("Completed", "Failed", False),
     ("Pending", "Deferred", True), ("Pending", "Not Applicable", True)],
)
def test_transition_table(src, dst, ok):
    from ssl_anomaly.benchmarking.registry import RegistryError, check_transition

    if ok:
        assert check_transition(src, dst) is None
    else:
        with pytest.raises(RegistryError):
            check_transition(src, dst)


@pytest.mark.parametrize("state", ["Failed", "Deferred", "Not Applicable"])
def test_reason_required_states(state):
    from ssl_anomaly.benchmarking.registry import REQUIRES_REASON

    assert state in REQUIRES_REASON


def _reg(tmp_path, cells, cfg):
    from ssl_anomaly.benchmarking.matrix import matrix_document, matrix_sha256
    from ssl_anomaly.benchmarking.registry import Registry

    sha = matrix_sha256(matrix_document(cfg, cells, applicability_sha="0" * 64))
    reg = Registry(tmp_path / "registry.json")
    reg.register(cells, matrix_sha256=sha)
    return reg, sha


def test_register_counts_and_claim(tmp_path, cells, cfg):
    reg, _ = _reg(tmp_path, cells, cfg)
    counts = reg.counts()
    assert counts["Pending"] == sum(1 for c in cells if c.status == "Pending")
    entry = reg.claim_next(tier="screening")
    assert entry is not None and reg.get(entry.cell_hash).status == "Running"
    assert entry.attempts == 1, "a claim is an attempt, and attempts are how a retry loop is audited"


def test_reload_is_stable_and_log_is_append_only(tmp_path, cells, cfg):
    from ssl_anomaly.benchmarking.registry import Registry

    reg, _ = _reg(tmp_path, cells, cfg)
    entry = reg.claim_next()
    reg2 = Registry(tmp_path / "registry.json")
    assert reg2.get(entry.cell_hash).status == "Running", "an in-flight cell must survive a process restart"
    lines = reg2.log_path.read_text().strip().splitlines()
    assert lines and all(l.startswith("{") for l in lines), "the log is JSONL, one record per transition"


def test_completed_never_reclaimed(tmp_path, cells, cfg):
    reg, _ = _reg(tmp_path, cells, cfg)
    e = reg.claim_next()
    reg.transition(e.cell_hash, "Completed", metrics={"macro_auprc": 0.4})
    ids_after = {x.cell_hash for x in reg.entries(status="Pending")}
    assert e.cell_hash not in ids_after
    assert reg.get(e.cell_hash).metrics["macro_auprc"] == 0.4


def test_reason_required_on_transition(tmp_path, cells, cfg):
    from ssl_anomaly.benchmarking.registry import RegistryError

    reg, _ = _reg(tmp_path, cells, cfg)
    e = reg.claim_next()
    with pytest.raises(RegistryError):
        reg.transition(e.cell_hash, "Failed")            # no error text, no reason
    reg.transition(e.cell_hash, "Failed", error="CUDA out of memory on cell 41")
    got = reg.get(e.cell_hash)
    assert "out of memory" in got.error                   # verbatim, not summarised
    assert got.reason == got.error, "the failure reason is the error text, not a paraphrase"
    reg.requeue(e.cell_hash, reason="fixed the VRAM spike by lowering batch size")
    assert reg.get(e.cell_hash).status == "Pending"
    assert "requeued" in reg.get(e.cell_hash).reason
    with pytest.raises(RegistryError):
        reg.requeue(e.cell_hash, reason="   ")            # an unexplained retry is drift


def test_recover_marks_running_as_interrupted(tmp_path, cells, cfg):
    from ssl_anomaly.benchmarking.registry import Registry

    reg, _ = _reg(tmp_path, cells, cfg)
    e = reg.claim_next()
    reg2 = Registry(tmp_path / "registry.json")
    recovered = reg2.recover()
    assert any(x.cell_hash == e.cell_hash for x in recovered)
    assert reg2.get(e.cell_hash).status == "Interrupted"


def test_defer_remaining_records_a_reason(tmp_path, cells, cfg):
    reg, _ = _reg(tmp_path, cells, cfg)
    n = reg.counts().get("Pending", 0)
    assert n > 0
    moved = reg.defer_remaining(reason="T4 budget exhausted at 6.0 h")
    assert len(moved) == n
    assert reg.counts().get("Pending", 0) == 0
    for h in moved[:5]:
        assert "budget exhausted" in (reg.get(h).reason or "")


def test_markdown_coverage_lists_every_state(tmp_path, cells, cfg):
    reg, _ = _reg(tmp_path, cells, cfg)
    p = reg.write_markdown(tmp_path / "coverage.md")
    text = p.read_text()
    for s in ALL:
        assert s in text, f"coverage.md must name {s} even when its count is zero"
    assert "COVERAGE:" in text, "the coverage sentence belongs in the state table too"


def test_rebinding_to_a_different_matrix_is_refused(tmp_path, cells, cfg):
    from ssl_anomaly.benchmarking.matrix import matrix_document, matrix_sha256
    from ssl_anomaly.benchmarking.registry import Registry, RegistryError

    reg, _ = _reg(tmp_path, cells, cfg)
    other = matrix_sha256(matrix_document(cfg, cells[:-1], applicability_sha="0" * 64))
    with pytest.raises(RegistryError):
        Registry(tmp_path / "registry.json").register(cells, matrix_sha256=other)

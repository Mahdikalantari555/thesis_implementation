"""The findings generator may render only what the gates sanction.

These tests exist because the failure mode is silent: a findings document that
quietly quotes a failed gate reads exactly like one that doesn't. Every assertion
here corresponds to a way the document could overclaim.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "evaluation" / "render_findings_26.py"


@pytest.fixture(scope="module")
def gen():
    sys.path.insert(0, str(REPO / "src"))
    spec = importlib.util.spec_from_file_location("render_f26", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _meta(gen, source="decision_gates.json", smoke=False):
    return gen._meta({"quotable": True}, Path(source), "abc1234", smoke)


def _block(verdicts, quotable=True, reason=""):
    claims = [{"gate": g, "verdict": v, "quotable": v == "passed",
               "fail_consequence": f"consequence of {g}"}
              for g, v in verdicts.items()]
    return {
        "quotable": quotable,
        "gates": dict(verdicts),
        "claims": claims if quotable else [],
        "reason": reason,
    }


ALL_PASS = {"G-A": "passed", "G-B": "passed", "G-C": "passed",
            "G-D": "passed", "G-E": "passed"}


# ---------------------------------------------------------------------------
# Quotability gate
# ---------------------------------------------------------------------------


def test_non_quotable_block_says_so_and_names_no_pass(gen):
    block = _block({"G-A": "blocked", "G-B": "blocked", "G-C": "blocked",
                    "G-D": "blocked", "G-E": "undetermined"},
                   quotable=False, reason="G-E did not pass")
    doc = gen.render_en(block, _meta(gen))
    assert "No claim from this run may be quoted" in doc
    assert "G-E did not pass" in doc
    assert "passed:" not in doc, "a blocked run must not list passed gates"


def test_non_quotable_block_is_never_described_as_complete(gen):
    block = _block({"G-E": "failed"}, quotable=False, reason="r")
    meta = gen._meta(block, Path("x.json"), "sha", smoke=False)
    assert "NOT quotable" in meta["status_text"]
    assert "incomplete" in meta["status_text"], (
        "a non-quotable run must be described as incomplete, not complete"
    )


def test_quotable_block_lists_passed_failed_and_undetermined_separately(gen):
    block = _block({"G-A": "passed", "G-B": "failed", "G-C": "undetermined",
                    "G-D": "passed", "G-E": "passed"})
    doc = gen.render_en(block, _meta(gen))
    assert "passed: G-A, G-D, G-E" in doc
    assert "failed: G-B" in doc
    assert "undetermined: G-C" in doc


def test_undetermined_is_never_printed_as_failed_or_passed(gen):
    """The most likely silent overclaim: rounding a tri-state into a binary."""
    block = _block({"G-A": "undetermined", "G-B": "passed", "G-C": "failed",
                    "G-D": "passed", "G-E": "passed"})
    doc = gen.render_en(block, _meta(gen))
    row = next(r for r in doc.splitlines() if "**G-A**" in r)
    assert "undetermined" in row
    assert "failed" not in row and "passed" not in row


def test_every_gate_appears_even_when_absent_from_the_block(gen):
    block = {"quotable": True, "gates": {"G-E": "passed"}, "claims": []}
    doc = gen.render_en(block, _meta(gen))
    for g in ("G-A", "G-B", "G-C", "G-D", "G-E"):
        assert f"**{g}**" in doc, (
            f"{g} missing from the table: an omitted gate reads as one that was "
            f"never applicable, which is a different claim"
        )


def test_collapse_of_missing_keys_defaults_to_undetermined_not_passed(gen):
    doc = gen.render_en({"quotable": True, "gates": {}, "claims": []}, _meta(gen))
    assert "undetermined: G-A, G-B, G-C, G-D, G-E" in doc


# ---------------------------------------------------------------------------
# No-number-leak guarantee
# ---------------------------------------------------------------------------


def test_document_contains_no_metric_number(gen):
    """Only verdicts and prose: no AP, no CI, no percentage anywhere.

    A number in this document that did not come from the quota block would be
    unsourced, and an unsourced number in a findings document is a claim.
    """
    import re

    block = _block(ALL_PASS)
    doc = gen.render_en(block, _meta(gen))
    # Strip the identifiers that legitimately contain digits.
    stripped = re.sub(r"\bG-[A-E]\b", "GATE", doc)
    stripped = re.sub(r"`[^`]*`", "CODE", stripped)
    stripped = re.sub(r"\b(19|20)\d\d-\d\d-\d\d\b", "DATE", stripped)
    stripped = re.sub(r"\bL[1-5]\b", "RUNG", stripped)
    assert not re.search(r"\d", stripped.replace("26", "")), (
        "the rendered document contains a number that is not a gate id, date, "
        "rung, or code reference; findings text must not invent figures"
    )


def test_verdict_translation_is_a_fixed_dictionary(gen):
    assert set(gen.VERDICT_FA) == {"passed", "failed", "undetermined", "blocked"}
    assert len(set(gen.VERDICT_FA.values())) == 4, (
        "two verdicts sharing a Persian word would make the languages disagree"
    )


def test_fa_document_carries_the_same_verdicts_as_en(gen):
    block = _block({"G-A": "passed", "G-B": "failed", "G-C": "undetermined",
                    "G-D": "passed", "G-E": "passed"})
    meta = _meta(gen)
    en, fa = gen.render_en(block, meta), gen.render_fa(block, meta)
    for gate in ("G-A", "G-B", "G-C", "G-D", "G-E"):
        en_row = next(r for r in en.splitlines() if f"**{gate}**" in r)
        fa_row = next(r for r in fa.splitlines() if f"**{gate}**" in r)
        for verdict in ("passed", "failed", "undetermined"):
            if verdict in en_row:
                assert gen.VERDICT_FA[verdict] in fa_row, (
                    f"{gate}: EN says {verdict}, FA does not"
                )


def test_fa_non_quotable_says_so(gen):
    block = _block({"G-E": "failed"}, quotable=False, reason="x")
    fa = gen.render_fa(block, _meta(gen))
    assert "هیچ ادعایی" in fa
    assert "قابل نقل نیست" in fa


# ---------------------------------------------------------------------------
# Metadata honesty
# ---------------------------------------------------------------------------


def test_smoke_render_is_labelled_not_evidence(gen):
    meta = gen._meta({"quotable": True}, Path("x.json"), "sha", smoke=True)
    assert "SMOKE" in meta["status_text"] and "not evidence" in meta["status_text"]
    assert "شاهد نیست" in meta["status_text_fa"]


def test_real_render_is_not_labelled_smoke(gen):
    meta = gen._meta({"quotable": True}, Path("x.json"), "sha", smoke=False)
    assert "SMOKE" not in meta["status_text"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_refuses_a_file_without_a_quota_block(gen, tmp_path, capsys):
    bad = tmp_path / "not_gates.json"
    bad.write_text(json.dumps({"stage": "decision-gates"}))
    assert gen.main(["--gates", str(bad), "--out-dir", str(tmp_path / "o")]) == 2
    assert "quota_block" in capsys.readouterr().err


def test_cli_refuses_a_missing_file(gen, tmp_path, capsys):
    assert gen.main(["--gates", str(tmp_path / "nope.json"),
                     "--out-dir", str(tmp_path)]) == 2
    assert "no gate file" in capsys.readouterr().err


def test_cli_writes_both_languages(gen, tmp_path):
    gates = tmp_path / "decision_gates.json"
    gates.write_text(json.dumps({"smoke": False, "quota_block": _block(ALL_PASS)}))
    out = tmp_path / "out"
    assert gen.main(["--gates", str(gates), "--out-dir", str(out),
                     "--git-sha", "deadbee"]) == 0
    en = (out / f"{gen.BASENAME}.md").read_text(encoding="utf-8")
    fa = (out / f"{gen.BASENAME}.fa.md").read_text(encoding="utf-8")
    assert "deadbee" in en and "deadbee" in fa
    assert en.startswith("# Finding 26")
    assert fa.startswith("# یافته ۲۶")

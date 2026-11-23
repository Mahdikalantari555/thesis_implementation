#!/usr/bin/env python3
"""Aggregate the capstone artifacts into G-A…G-E verdicts.

Change: regional-ssl-representation-evaluation (design §Decision gates). Writes
``decision_gates.json`` — the single artifact a findings document should read.

This script owns NO science. It reads the artifacts the other stages produced,
pulls out exactly the evidence keys each gate declares, and lets
:mod:`ssl_anomaly.evaluation.gates` decide. Two consequences that matter:

- If a stage did not run (no ``pretraining_quality.json``, say), the gate it feeds
  is ``undetermined`` with the missing keys named — never silently passed and
  never silently failed.
- G-E is evaluated from a live artifact binding. When binding fails, every other
  verdict is downgraded to ``blocked`` and the quota block returns zero claims,
  so an unverified data lineage silences the report regardless of the numbers.

Usage
-----
::

    python scripts/evaluation/aggregate_decision_gates.py --smoke
    python scripts/evaluation/aggregate_decision_gates.py \
        --reports reports/regional_representation \
        --incumbent reports/regional_representation/incumbent_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "evaluation"))

import numpy as np  # noqa: E402

from _common import add_common_args, finish_smoke, resolve_run  # noqa: E402
from ssl_anomaly.evaluation.binding import bind_artifacts, load_preregistration  # noqa: E402
from ssl_anomaly.evaluation.gates import (  # noqa: E402
    GATE_INPUTS,
    evaluate_gates,
)
from ssl_anomaly.evaluation.runner import (  # noqa: E402
    RunPaths,
    SmokeExpectation,
    build_manifest,
    make_smoke_fixture,
    write_run_outputs,
)

#: Which artifact each gate's evidence comes from, and the extractor that pulls
#: the declared keys out of it. Keeping this mapping explicit is what stops a
#: gate from quietly reading whatever happened to be in the directory.
# Each gate names the artifact PATTERNS it reads and the extractor that turns them
# into its inputs. Patterns, not single filenames, because one transfer run covers
# ONE task while G-B and G-C count tasks independently: the producer writes
# ``transfer_curves_T1.json``, ``transfer_curves_T2.json``, ... and every matching
# artifact is extracted and merged. Reading only the first file would leave those
# gates permanently undetermined-by-construction, which would misreport a complete
# run as an incomplete one.
GATE_SOURCES = {
    "G-A": (("pretraining_quality.json",), "_evidence_g_a"),
    "G-B": (("transfer_curves*.json",), "_evidence_g_b"),
    "G-C": (("transfer_curves*.json",), "_evidence_g_c"),
    "G-D": (("retrieval_benchmark*.json",), "_evidence_g_d"),
    "G-E": (("claims.json",), "_evidence_g_e"),
}

# Inputs that accumulate across artifacts (task keyed) instead of being replaced.
_MERGED_INPUTS = {"label_efficiency", "transfer_grid"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--reports", type=Path, default=Path("reports/regional_representation"),
                   help="directory holding the stage artifacts")
    p.add_argument("--incumbent", type=Path, default=None,
                   help="explicit incumbent-comparison artifact (overrides --reports)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    started = time.time()
    args = parse_args(argv)
    prereg = load_preregistration()
    ctx = resolve_run(args, stage="gates")
    reports_dir = Path(args.reports)

    # ---- gather evidence -------------------------------------------------
    evidence: dict = {}
    provenance: dict[str, dict] = {}
    for gate, (patterns, extractor) in GATE_SOURCES.items():
        if gate == "G-D" and args.incumbent:
            paths = [Path(args.incumbent)]
        else:
            paths = sorted({p for pat in patterns for p in reports_dir.glob(pat)})
        if not paths:
            provenance[gate] = {"source": ", ".join(
                str(reports_dir / pat) for pat in patterns), "found": False,
                "provided": []}
            continue
        declared = set(GATE_INPUTS[gate])
        usable: dict = {}
        seen: list[str] = []
        for path in paths:
            payload = json.loads(path.read_text())
            extracted = globals()[extractor](payload)
            seen.extend(sorted(set(extracted) - declared))
            for key, value in extracted.items():
                if key not in declared:
                    continue
                if key == "transfer_grid" and key in usable:
                    for task, per_frac in value.items():
                        merged = usable[key].setdefault(task, {})
                        for frac_str, arms in per_frac.items():
                            merged.setdefault(frac_str, {}).update(arms)
                elif key in _MERGED_INPUTS and key in usable:
                    # Merge per task, not per file: two artifacts covering the same
                    # task would otherwise silently overwrite one another, keeping
                    # whichever the glob happened to return last.
                    for task, block in value.items():
                        merged = usable[key].setdefault(task, {})
                        for arm, per_frac in block.items():
                            merged.setdefault(arm, {}).update(per_frac)
                else:
                    usable[key] = value
        evidence.update(usable)
        entry = {
            "source": [str(p) for p in paths],
            "found": True,
            "provided": sorted(usable),
            "reported_but_not_a_gate_input": sorted(set(seen)),
            "n_artifacts": len(paths),
        }
        # Record which claim-bearing tasks this artifact contributes, because G-B
        # and G-C are task-counted gates: one task can pass a per-task comparison
        # and still leave the gate short of its required task count.
        if gate in {"G-B", "G-C"} and usable:
            key = "label_efficiency" if gate == "G-B" else "transfer_grid"
            tasks = sorted(usable.get(key, {}) or {})
            entry["tasks"] = tasks
            if len(tasks) < 2:
                entry["note"] = (
                    f"{len(tasks)} claim-bearing task(s) in this artifact; "
                    f"{gate} counts tasks independently and needs 2, so further "
                    f"transfer artifacts must be supplied before it can pass"
                )
        provenance[gate] = entry

    # ---- binding: G-E's evidence -----------------------------------------
    binding = bind_artifacts(prereg=prereg, strict=False,
                             compute_hashes=not ctx.smoke)
    report = evaluate_gates(prereg=prereg, evidence=evidence, binding=binding)

    payload = {
        "stage": "decision-gates",
        "smoke": ctx.smoke,
        "gates": report.to_dict(),
        "quota_block": report.quota_block(),
        "evidence_provenance": provenance,
        "gate_inputs": {k: list(v) for k, v in GATE_INPUTS.items()},
        "reading_rule": (
            "G-E is a precondition: when it fails, every other verdict is 'blocked' "
            "and no claim is quotable. A gate whose inputs are missing is "
            "'undetermined', never 'passed'. Only 'passed' is quotable."
        ),
        "wall_time_s": round(time.time() - started, 2),
    }

    manifest = build_manifest(
        stage="decision-gates", smoke=ctx.smoke,
        config={"reports": str(reports_dir)}, seeds=args.seeds, binding=binding,
    )
    written = write_run_outputs(paths=ctx.paths, manifest=manifest,
                               reports={"decision_gates": payload})

    if ctx.smoke:
        # A smoke run has no stage artifacts, so the gates must be undetermined.
        # That is the assertion: absence must not read as success.
        exp = SmokeExpectation("decision-gates")
        verdicts = payload["gates"]["gates"]
        exp.check("all five gates are present",
                  set(verdicts) == {"G-A", "G-B", "G-C", "G-D", "G-E"})
        # With no claims artifact and no stage artifacts, G-E cannot pass, so the
        # other gates are BLOCKED (not "undetermined"): the distinction matters,
        # because "blocked" says the run is unquotable while "undetermined" would
        # suggest a gate that merely lacks data.
        exp.check("without a claims artifact G-E does not pass",
                  verdicts["G-E"]["verdict"] in {"undetermined", "failed"},
                  verdicts["G-E"]["verdict"])
        exp.check("a non-passing G-E blocks every other gate",
                  all(verdicts[g]["verdict"] == "blocked" and
                      verdicts[g]["blocked_by"] == "G-E"
                      for g in ("G-A", "G-B", "G-C", "G-D")))
        exp.check("each blocked/undetermined gate names the evidence it lacks",
                  all(verdicts[g]["missing_evidence"] or verdicts[g]["reason"]
                      for g in ("G-A", "G-B", "G-C", "G-D")))
        exp.check("every verdict carries its fail consequence",
                  all(verdicts[g]["fail_consequence"] for g in verdicts))
        exp.check("no claim is quotable when G-E did not pass",
                  payload["quota_block"]["quotable"] is False
                  or verdicts["G-E"]["verdict"] == "passed")
        exp.check("each gate's evidence provenance is recorded",
                  all(provenance[g]["source"] for g in provenance))
        payload["smoke_checks"] = exp.to_dict()
        write_run_outputs(paths=ctx.paths, manifest=manifest,
                          reports={"decision_gates": payload})
        print(f"wrote {written['decision_gates']}")
        return finish_smoke(exp, ctx.paths, "decision-gates")

    print(f"wrote {written['decision_gates']}")
    print("verdicts:", {g: v["verdict"] for g, v in payload["gates"]["gates"].items()})
    return 0


# ---------------------------------------------------------------------------
# Artifact -> gate evidence extractors
# ---------------------------------------------------------------------------


def _evidence_g_a(payload: dict) -> dict:
    """G-A: the screened candidate's probes, paired against a random-init encoder.

    The comparator is A-rand, so the artifact must contain BOTH an SSL candidate's
    Level-1 numbers and the random-init control's. A run that only probed the SSL
    candidate supplies the SSL side and the gate reports the comparison
    ``undetermined`` rather than silently comparing against chance.
    """
    screening = payload.get("screening", {})
    candidates = payload.get("candidates", {})
    controls = payload.get("random_init_control", {})
    name = screening.get("winner")
    if name is None and candidates:
        scored = screening.get("candidates", {})
        if scored:
            name = max(scored, key=lambda c: (scored[c].get("terms_passed", 0),
                                              scored[c].get("capacity_ok", False)))
    if name is None or name not in candidates:
        return {}
    ssl_ev = candidates[name].get("gate_evidence", {})
    rand_ev = controls.get("gate_evidence", {})
    out: dict = {}
    mapping = {
        "farm_separability_top1": "farm_separability_ssl",
        "phase_probe_accuracy": "phase_probe_ssl",
    }
    for src, dst in mapping.items():
        if ssl_ev.get(src) is not None and rand_ev.get(src) is not None:
            out[dst] = float(ssl_ev[src])
            out[dst.replace("_ssl", "_rand")] = float(rand_ev[src])
    if ssl_ev.get("collapse_disqualified") is not None:
        out["collapse_disqualified"] = bool(ssl_ev["collapse_disqualified"])
    return out


def _curves_to_tables(payload: dict) -> tuple[dict, dict]:
    """Turn one transfer_curves artifact into G-B and G-C tables.

    A single artifact can feed both gates because they read the same grid at
    different resolutions: G-B asks whether an SSL arm matches A-scratch@100 % at
    a small label fraction, G-C asks whether SSL beats A-rule and A-rand at two
    fractions on two tasks.

    Fractions are keyed as two-decimal STRINGS (``"0.05"``, ``"1.00"``) because the
    gate criteria name fractions in prose and the frozen lookups use that form —
    float keys would compare unequal to the criterion's own literals and silently
    empty the table.
    """
    result = payload.get("result", {})
    curves = result.get("curves", {})
    task = payload.get("task", "T2")
    label_table: dict = {}
    grid: dict = {}
    for arm, curve in curves.items():
        cells = [c for c in curve.get("cells", []) if c.get("status") == "measured"]
        if not cells:
            continue
        per_frac: dict = {}
        for c in cells:
            per_frac.setdefault(f"{float(c['fraction']):.2f}", []).append(c["ap"])
        label_table[arm] = {f: float(np.mean(v)) for f, v in per_frac.items()}
        grid.setdefault(arm, {})
        for f, v in per_frac.items():
            grid[arm][f] = float(np.mean(v))
    # G-B's shape: arm -> fraction -> value
    g_b = {task: label_table} if label_table else {}
    # G-C's shape: fraction -> arm -> value
    g_c: dict = {}
    for arm, fracs in grid.items():
        for f, v in fracs.items():
            g_c.setdefault(f, {})[arm] = v
    return g_b, ({task: g_c} if g_c else {})


def _evidence_g_b(payload: dict) -> dict:
    table, _ = _curves_to_tables(payload)
    return {"label_efficiency": table} if table else {}


def _evidence_g_c(payload: dict) -> dict:
    _, grid = _curves_to_tables(payload)
    return {"transfer_grid": grid} if grid else {}


def _evidence_g_d(payload: dict) -> dict:
    """G-D: T3 TRACK 1 with all three baselines, at the four pre-registered k.

    Track 2 and the near-match listing are never read: the spec makes them
    exploratory and forbids them as gate inputs.
    """
    block = payload.get("tracks", {}).get("track-1", {})
    if not block:
        return {}
    arms = block.get("arms", {})
    ssl_key = next((a for a in ("A-ssl-frozen", "A-ssl-ft") if a in arms), None)
    if ssl_key is None:
        return {}
    k = [str(x) for x in block.get("k_values", [])]
    if not k:
        return {}
    baseline_map = {"raw_distance": "raw-distance", "b1_score": "A-B1",
                    "rule_score": "A-rule"}
    out: dict = {"track": "1", "k": k, "ssl": {}}
    for kk in k:
        val = arms[ssl_key].get("precision_at_k", {}).get(kk)
        if val is not None:
            out["ssl"][kk] = float(val)
    for gate_key, arm_name in baseline_map.items():
        if arm_name in arms:
            out[gate_key] = {
                kk: float(v) for kk, v in arms[arm_name].get("precision_at_k", {}).items()
            }
    return {"retrieval_track1": out}


def _evidence_g_e(payload: dict) -> dict:
    claims = payload.get("claims")
    if not isinstance(claims, list):
        return {}
    return {"claims": claims}


if __name__ == "__main__":
    raise SystemExit(main())

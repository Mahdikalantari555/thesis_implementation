"""Gate aggregation: evidence in, G-A…G-E verdicts out, no judgement calls.

**Every criterion here is transcribed from
``configs/evaluation/pre_registration.yaml`` and must match it exactly.** An
earlier version of this module implemented plausible-sounding criteria that were
NOT the pre-registered ones (G-B as "AP beats prevalence", G-D as "delta beats
the incumbent"). That is the failure mode the whole freeze exists to stop: a gate
that decides something reasonable but unagreed is worse than no gate, because the
verdict still looks authoritative. ``tests/evaluation/test_gates.py`` now asserts
each gate's logical form against the frozen text, so a future edit that drifts
from the YAML fails the suite.

The five gates (verbatim criteria; see the YAML for the fail consequences):

- **G-A Pretraining quality** — SSL beats A-rand on ≥2 of {farm-separability,
  phase probe, frozen-head T1} AND no collapse criteria triggered.
- **G-B Label efficiency** — A-ssl-frozen or A-ssl-ft matches A-scratch@100 % at
  ≤50 % label fraction on ≥1 claim-bearing task.
- **G-C Transfer dominance** — A-ssl arms beat A-rule and A-rand on ≥2
  claim-bearing tasks at ≥2 label fractions each.
- **G-D Retrieval usefulness** — A-ssl retrieval beats raw-distance, A-B1-score
  and A-rule-score ranking on T3 Track 1 at ≥3 of 4 k values.
- **G-E Claim discipline** — every reported claim carries rung + provenance;
  negative results published with equal prominence.

Two structural rules are enforced here rather than described:

1. **G-E is a precondition, not a peer.** If claim discipline fails, every other
   gate's verdict is downgraded to ``blocked``: the arithmetic may exist, but its
   output is not quotable. This also covers a failed artifact binding, which makes
   the provenance field of every claim unverifiable.
2. **A gate with missing inputs is ``undetermined``, never ``passed``.** Each gate
   declares the evidence keys it reads; an empty evidence dict cannot pass.

Nothing in this module may be weakened after results are seen. If a criterion
proves wrong, the amendment goes in ``amendments:`` of the pre-registration and
the findings document — never into a silent edit here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ssl_anomaly.evaluation.binding import BindingReport, Preregistration

GATE_IDS = ("G-A", "G-B", "G-C", "G-D", "G-E")

PASS = "passed"
FAIL = "failed"
UNDETERMINED = "undetermined"
BLOCKED = "blocked"

#: The arms the pre-registration names. Referenced by G-B/G-C/G-D.
SSL_ARMS = ("A-ssl-frozen", "A-ssl-ft")
BASELINE_ARMS = ("A-rule", "A-B1", "A-rand", "A-scratch")
ALL_ARMS = ("A-rule", "A-B1", "A-rand", "A-scratch", "A-ssl-frozen", "A-ssl-ft")

#: Claim-bearing tasks (design §Tasks). T5 is exploratory and never enters a gate.
CLAIM_BEARING_TASKS = ("T1", "T2", "T3", "T4")
EXPLORATORY_TASKS = ("T5",)

#: Which evidence keys each gate reads. Anything not listed cannot influence it.
GATE_INPUTS: dict[str, tuple[str, ...]] = {
    "G-A": (
        "farm_separability_ssl", "farm_separability_rand",
        "phase_probe_ssl", "phase_probe_rand",
        "frozen_head_t1_ssl", "frozen_head_t1_rand",
        "collapse_disqualified",
    ),
    "G-B": ("label_efficiency",),
    "G-C": ("transfer_grid",),
    "G-D": ("retrieval_track1",),
    "G-E": ("claims",),
}


@dataclass
class GateVerdict:
    gate: str
    verdict: str                     # passed | failed | undetermined | blocked
    criteria: dict
    evidence: dict
    fail_consequence: str
    reason: str = ""
    blocked_by: str = ""
    missing_evidence: list[str] = field(default_factory=list)
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.verdict not in {PASS, FAIL, UNDETERMINED, BLOCKED}:
            raise ValueError(f"unknown verdict {self.verdict!r}")

    @property
    def passed(self) -> bool:
        return self.verdict == PASS

    @property
    def quotable(self) -> bool:
        return self.verdict == PASS

    def to_dict(self) -> dict:
        return {
            "gate": self.gate,
            "verdict": self.verdict,
            "criteria": self.criteria,
            "evidence": self.evidence,
            "fail_consequence": self.fail_consequence,
            "reason": self.reason,
            "blocked_by": self.blocked_by,
            "missing_evidence": list(self.missing_evidence),
            "detail": self.detail,
        }


@dataclass
class GateReport:
    verdicts: dict[str, GateVerdict]
    binding: BindingReport | None = None

    def to_dict(self) -> dict:
        return {
            "gates": {k: v.to_dict() for k, v in self.verdicts.items()},
            "binding": self.binding.to_dict() if self.binding else None,
            "quotable_gates": self.quotable_gates(),
        }

    def quotable_gates(self) -> list[str]:
        return [g for g in GATE_IDS if self.verdicts[g].quotable]

    def any_quotable(self) -> bool:
        return bool(self.quotable_gates())

    def quota_block(self) -> dict:
        """The one block a findings generator is allowed to render.

        G-E failing silences everything: no claim from the run may be quoted,
        whatever the other gates computed.
        """
        if not self.verdicts["G-E"].passed:
            return {
                "quotable": False,
                "reason": (
                    "G-E (claim discipline) did not pass: no claim from this run may "
                    "be quoted, whatever the other gates computed"
                ),
                "gates": {g: self.verdicts[g].verdict for g in GATE_IDS},
                "claims": [],
            }
        claims = []
        for g in GATE_IDS:
            v = self.verdicts[g]
            claims.append({
                "gate": g,
                "verdict": v.verdict,
                "quotable": v.quotable,
                "fail_consequence": v.fail_consequence,
            })
        return {
            "quotable": True,
            "gates": {g: self.verdicts[g].verdict for g in GATE_IDS},
            "claims": claims,
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _missing(evidence: dict, gate: str) -> list[str]:
    return [k for k in GATE_INPUTS[gate] if k not in evidence or evidence[k] is None]


def _filter(evidence: dict, gate: str) -> dict:
    return {k: evidence[k] for k in GATE_INPUTS[gate] if k in evidence}


def _undetermined(gate: str, criteria: dict, evidence: dict, consequence: str,
                  missing: list[str]) -> GateVerdict:
    return GateVerdict(
        gate=gate, verdict=UNDETERMINED, criteria=criteria, evidence=evidence,
        fail_consequence=consequence, missing_evidence=missing,
        reason=(f"missing evidence: {', '.join(missing)} — an unmeasured gate is "
                f"never reported as passed"),
    )


def _no_claims(evidence: dict, gate: str, criteria: dict, consequence: str,
               what: str) -> GateVerdict:
    return GateVerdict(
        gate=gate, verdict=UNDETERMINED, criteria=criteria, evidence=evidence,
        fail_consequence=consequence,
        reason=f"{what} is empty: the gate has nothing to decide on",
        missing_evidence=[GATE_INPUTS[gate][0]],
    )


# ---------------------------------------------------------------------------
# G-A — pretraining quality
# ---------------------------------------------------------------------------


def gate_a(evidence: dict, criteria: dict) -> GateVerdict:
    """SSL beats A-rand on ≥2 of {farm-separability, phase probe, frozen-head T1}
    AND no collapse criteria triggered.

    **The comparator is A-rand, not chance.** Beating chance would let a
    representation pass merely because its architecture has inductive bias; the
    pre-registration asks whether PRE-TRAINING added anything over the same trunk
    with random weights, so each term is a paired SSL-vs-A-rand comparison.

    A term whose inputs are absent is neither beaten nor failed: it is reported
    unmeasured, ``min_terms`` is capped at the number of measurable terms, and the
    reason names the cap. This keeps the gate decidable on one term while making
    the reduction visible — collapsing "could not measure" into "did not work"
    either inflates a pass or manufactures a failure.
    """
    gate = "G-A"
    consequence = criteria.get("fail_consequence", "Candidate disqualified.")
    missing = _missing(evidence, gate)
    ev = _filter(evidence, gate)

    if "collapse_disqualified" in missing:
        return _undetermined(gate, criteria, ev, consequence, missing)

    pairs = {
        "farm_separability": ("farm_separability_ssl", "farm_separability_rand"),
        "phase_probe": ("phase_probe_ssl", "phase_probe_rand"),
        "frozen_head_t1": ("frozen_head_t1_ssl", "frozen_head_t1_rand"),
    }
    terms: dict[str, bool] = {}
    unmeasured: list[str] = []
    for term, (ssl_key, rand_key) in pairs.items():
        if ssl_key in missing or rand_key in missing:
            unmeasured.append(term)
            continue
        terms[term] = bool(ev[ssl_key] > ev[rand_key])

    if not terms:
        return _undetermined(gate, criteria, ev, consequence,
                             sorted({*unmeasured, *missing}))

    beaten = sum(terms.values())
    measured = len(terms)
    collapse = bool(ev["collapse_disqualified"])
    # The criterion says ">=2 of 3". If fewer than 2 are measurable the gate cannot
    # reach its threshold honestly, so it is capped — but never below 1, and the
    # cap is stated in the reason.
    effective_min = min(2, measured)

    passed = beaten >= effective_min and not collapse
    reasons = []
    if collapse:
        reasons.append("collapse criteria triggered")
    if beaten < effective_min:
        # Report the full arithmetic, not just the shortfall: a reader must see
        # that the third term WAS won, otherwise a 1/2 looks like "nothing worked".
        won = [t for t, ok in terms.items() if ok]
        lost = [t for t, ok in terms.items() if not ok]
        reasons.append(
            f"{beaten}/{effective_min} measurable terms beat A-rand"
            + (f" (won: {', '.join(won)}; lost: {', '.join(lost)})" if lost else "")
        )
    if unmeasured:
        reasons.append(
            f"unmeasured term(s): {', '.join(unmeasured)}; decided on {measured}/3"
        )
    return GateVerdict(
        gate=gate, verdict=PASS if passed else FAIL, criteria=criteria, evidence=ev,
        fail_consequence=consequence, reason="; ".join(reasons),
        missing_evidence=unmeasured,
        detail={"terms": terms, "terms_beaten": beaten,
                "effective_min_terms": effective_min,
                "unmeasured_terms": unmeasured},
    )


# ---------------------------------------------------------------------------
# G-B — label efficiency
# ---------------------------------------------------------------------------


def _fraction_key(fraction: float) -> str:
    return f"{round(float(fraction), 2):.2f}"


def gate_b(evidence: dict, criteria: dict) -> GateVerdict:
    """A-ssl-frozen or A-ssl-ft matches A-scratch@100 % at ≤50 % label fraction.

    ``label_efficiency`` is a nested mapping::

        {"T2": {"A-scratch": {"1.00": 0.62},
                "A-ssl-frozen": {"0.05": 0.55, "0.10": 0.60, "0.50": 0.64}}}

    "Matches" means at least equal at the same task. The criterion names ≤50 %, so
    only fractions at or below that threshold are eligible — a 100 %-fraction SSL
    comparison is the easy test the gate exists to avoid.

    Only CLAIM-BEARING tasks count. T5 appearing here would be a spec violation and
    is rejected rather than silently ignored.
    """
    gate = "G-B"
    consequence = criteria.get("fail_consequence", '"Sample efficiency" claim fails.')
    missing = _missing(evidence, gate)
    if missing:
        return _undetermined(gate, criteria, _filter(evidence, gate), consequence, missing)

    table = evidence["label_efficiency"]
    if not isinstance(table, dict) or not table:
        return _no_claims(evidence, gate, criteria, consequence, "label_efficiency")

    illegal = [t for t in table if t in EXPLORATORY_TASKS]
    if illegal:
        raise ValueError(
            f"G-B was given exploratory task(s) {illegal}; T5 is never claim-bearing "
            f"and must not enter a gate"
        )

    max_fraction = float(criteria.get("ssl_fraction_max", criteria.get("max_fraction", 0.50)))
    eligible = [t for t in table if t in CLAIM_BEARING_TASKS]

    detail: dict = {"per_task": {}, "max_ssl_fraction": max_fraction}
    wins: list[str] = []
    for task in sorted(eligible):
        arms = table[task]
        scratch = arms.get("A-scratch", {}).get("1.00")
        if scratch is None:
            detail["per_task"][task] = {"reason": "A-scratch@100% not reported"}
            continue
        best = None
        for arm in SSL_ARMS:
            for frac_str, value in arms.get(arm, {}).items():
                frac = float(frac_str)
                if frac > max_fraction + 1e-9 or value is None:
                    continue
                margin = float(value) - float(scratch)
                cand = {"arm": arm, "fraction": frac, "value": float(value),
                        "scratch_at_100": float(scratch), "margin": margin,
                        "matches": margin >= 0.0}
                if best is None or cand["margin"] > best["margin"]:
                    best = cand
        if best is None:
            detail["per_task"][task] = {
                "reason": f"no SSL cell at or below {max_fraction:.0%} label fraction",
                "scratch_at_100": float(scratch),
            }
            continue
        detail["per_task"][task] = best
        if best["matches"]:
            wins.append(task)

    detail["tasks_matching"] = wins
    if not detail["per_task"]:
        return _no_claims(evidence, gate, criteria, consequence, "label_efficiency")

    passed = bool(wins)
    reason = ""
    if not passed:
        reason = (
            "no claim-bearing task has an SSL arm matching A-scratch@100% at or "
            f"below {max_fraction:.0%} label fraction"
        )
    return GateVerdict(
        gate=gate, verdict=PASS if passed else FAIL, criteria=criteria,
        evidence=_filter(evidence, gate), fail_consequence=consequence,
        reason=reason, detail=detail,
    )


# ---------------------------------------------------------------------------
# G-C — transfer dominance
# ---------------------------------------------------------------------------


def gate_c(evidence: dict, criteria: dict) -> GateVerdict:
    """A-ssl arms beat A-rule AND A-rand on ≥2 claim-bearing tasks at ≥2 label
    fractions each.

    ``transfer_grid`` is::

        {"T2": {"0.25": {"A-ssl-frozen": 0.5, "A-rule": 0.4, "A-rand": 0.3}, ...},
         "T3": {...}}

    "Beat" requires strictly exceeding BOTH baselines at that (task, fraction).
    A cell where SSL beats A-rule but loses to A-rand is not a win: the
    pre-registration asks for dominance over both, because A-rand already controls
    for raw architecture capacity.
    """
    gate = "G-C"
    consequence = criteria.get(
        "fail_consequence", '"SSL learns superior representations" unsupported.'
    )
    missing = _missing(evidence, gate)
    if missing:
        return _undetermined(gate, criteria, _filter(evidence, gate), consequence, missing)

    grid = evidence["transfer_grid"]
    if not isinstance(grid, dict) or not grid:
        return _no_claims(evidence, gate, criteria, consequence, "transfer_grid")

    illegal = [t for t in grid if t in EXPLORATORY_TASKS]
    if illegal:
        raise ValueError(
            f"G-C was given exploratory task(s) {illegal}; T5 must not enter a gate"
        )

    required_tasks = int(criteria.get("min_tasks", 2))
    required_fractions = int(criteria.get("min_fractions", 2))

    detail: dict = {"per_task": {}}
    qualifying: dict[str, list[float]] = {}
    for task in sorted(t for t in grid if t in CLAIM_BEARING_TASKS):
        wins_at: list[float] = []
        per_frac: dict = {}
        for frac_str, arms in grid[task].items():
            frac = float(frac_str)
            ssl_vals = [arms.get(a) for a in SSL_ARMS if arms.get(a) is not None]
            base_vals = [arms.get(a) for a in ("A-rule", "A-rand")
                         if arms.get(a) is not None]
            if not ssl_vals or len(base_vals) < 2:
                per_frac[frac_str] = {"decidable": False,
                                      "reason": "needs both baselines and an SSL arm"}
                continue
            best_ssl = max(ssl_vals)
            worst_base = max(base_vals)   # must beat the STRONGER baseline
            win = best_ssl > worst_base
            per_frac[frac_str] = {
                "decidable": True, "best_ssl": best_ssl,
                "strongest_baseline": worst_base, "win": win,
            }
            if win:
                wins_at.append(frac)
        detail["per_task"][task] = {
            "wins_at_fractions": sorted(wins_at),
            "n_fractions": len(wins_at),
            "cells": per_frac,
        }
        if len(wins_at) >= required_fractions:
            qualifying[task] = sorted(wins_at)

    detail["qualifying_tasks"] = sorted(qualifying)
    detail["required_tasks"] = required_tasks
    detail["required_fractions_per_task"] = required_fractions
    if not detail["per_task"]:
        return _no_claims(evidence, gate, criteria, consequence, "transfer_grid")

    # If nothing was decidable the gate has no comparison to judge. Reporting
    # "failed" here would claim SSL lost when in fact it was never measured.
    decidable_cells = sum(
        1 for t in detail["per_task"].values()
        for c in t["cells"].values() if c.get("decidable")
    )
    detail["decidable_cells"] = decidable_cells
    if decidable_cells == 0:
        return GateVerdict(
            gate=gate, verdict=UNDETERMINED, criteria=criteria,
            evidence=_filter(evidence, gate), fail_consequence=consequence,
            missing_evidence=["transfer_grid (no cell has both baselines and an SSL arm)"],
            reason="no decidable cell: every (task, fraction) lacks a baseline or an "
                   "SSL arm, so dominance was never measured",
            detail=detail,
        )

    passed = len(qualifying) >= required_tasks
    reason = ""
    if not passed:
        reason = (
            f"{len(qualifying)} of {len(detail['per_task'])} claim-bearing tasks beat "
            f"both A-rule and A-rand at >= {required_fractions} fractions "
            f"(need {required_tasks} tasks)"
        )
    return GateVerdict(
        gate=gate, verdict=PASS if passed else FAIL, criteria=criteria,
        evidence=_filter(evidence, gate), fail_consequence=consequence,
        reason=reason, detail=detail,
    )


# ---------------------------------------------------------------------------
# G-D — retrieval usefulness
# ---------------------------------------------------------------------------


def gate_d(evidence: dict, criteria: dict) -> GateVerdict:
    """A-ssl retrieval beats raw-distance, A-B1-score and A-rule-score ranking on
    T3 **Track 1** at ≥3 of 4 k values.

    ``retrieval_track1`` is::

        {"k": [1, 3, 5, 10],
         "ssl": {"1": 0.7, "3": 0.6, ...},
         "raw_distance": {...}, "b1_score": {...}, "rule_score": {...}}

    Track 1 only. Track 2 is weak-proxy and exploratory, and the near-match listing
    is descriptive — neither may feed this gate (spec: "never claim-bearing, never
    gate input").
    """
    gate = "G-D"
    consequence = criteria.get(
        "fail_consequence", 'Retrieval claim fails; SSL limited to "descriptive embeddings".'
    )
    missing = _missing(evidence, gate)
    if missing:
        return _undetermined(gate, criteria, _filter(evidence, gate), consequence, missing)

    block = evidence["retrieval_track1"]
    if not isinstance(block, dict) or not block:
        return _no_claims(evidence, gate, criteria, consequence, "retrieval_track1")

    # Refuse a Track-2 or near-match payload masquerading as Track 1.
    track = str(block.get("track", "1"))
    if track not in {"1", "Track 1", "track-1"}:
        raise ValueError(
            f"G-D reads T3 Track 1 only; got track={track!r}. Track 2 (weak proxy) is "
            f"exploratory and the near-match listing is descriptive — neither may be "
            f"a gate input"
        )

    baselines = ("raw_distance", "b1_score", "rule_score")
    k_values = [str(k) for k in block.get("k", [])]
    if not k_values:
        return _no_claims(evidence, gate, criteria, consequence, "retrieval_track1")

    ssl_scores = block.get("ssl", {})
    detail: dict = {"per_k": {}}
    wins_at: list[str] = []
    for k in k_values:
        ssl_v = ssl_scores.get(k)
        base_v = {b: block.get(b, {}).get(k) for b in baselines}
        if ssl_v is None or any(v is None for v in base_v.values()):
            detail["per_k"][k] = {"decidable": False,
                                  "reason": "missing an arm at this k"}
            continue
        strongest = max(base_v.values())
        beat = all(ssl_v > base_v[b] for b in baselines)
        # "decidable": True is REQUIRED here. The gate's final
        # `any(v.get("decidable") ...)` check decides whether it has any real
        # comparison at all; setting the flag only on the failure branch made that
        # check false for a fully-decided gate, so a complete, passing payload was
        # reported as "empty: nothing to decide on".
        detail["per_k"][k] = {"decidable": True, "ssl": float(ssl_v),
                              "baselines": base_v,
                              "strongest_baseline": float(strongest),
                              "beats_all": beat}
        if beat:
            wins_at.append(k)

    required = int(criteria.get("min_k_values", 3))
    detail["wins_at_k"] = wins_at
    detail["required_k_values"] = required
    if not any(v.get("decidable") for v in detail["per_k"].values()):
        return _no_claims(evidence, gate, criteria, consequence, "retrieval_track1")

    passed = len(wins_at) >= required
    reason = ""
    if not passed:
        reason = (
            f"SSL beats all three baselines at {len(wins_at)} of {len(k_values)} k "
            f"values (need {required})"
        )
    return GateVerdict(
        gate=gate, verdict=PASS if passed else FAIL, criteria=criteria,
        evidence=_filter(evidence, gate), fail_consequence=consequence,
        reason=reason, detail=detail,
    )


# ---------------------------------------------------------------------------
# G-E — claim discipline
# ---------------------------------------------------------------------------


def gate_e(evidence: dict, criteria: dict | None = None,
           binding: BindingReport | None = None) -> GateVerdict:
    """Every reported claim carries rung + provenance; negatives published equally.

    ``claims`` is a list of mappings, each with at least::

        {"claim_id": "...", "status": "measured|absent|deferred|exploratory",
         "rung": "L1..L5" or None, "exploratory": bool,
         "provenance": {...} or {}, "prominence": "primary|negative"}

    The gate fails if ANY claim lacks a rung (unless explicitly exploratory), lacks
    provenance, or if a negative result was demoted below a positive one. It also
    fails when a binding report is supplied and shows a hash mismatch, because a
    claim whose data lineage cannot be verified has no provenanced value.

    Exploratory claims (T5) are allowed to carry no rung — the design says they are
    never claim-bearing — but they still need provenance and equal prominence. They
    are counted separately so a report cannot pad its claim count with them.
    """
    gate = "G-E"
    criteria = criteria or {}
    consequence = criteria.get(
        "fail_consequence", "Report rejected at review; change not closable."
    )
    missing = _missing(evidence, gate)
    if missing:
        return _undetermined(gate, criteria, _filter(evidence, gate), consequence, missing)

    claims = evidence["claims"]
    if not isinstance(claims, list) or not claims:
        return GateVerdict(
            gate=gate, verdict=UNDETERMINED, criteria=criteria,
            evidence=_filter(evidence, gate), fail_consequence=consequence,
            reason="claims list is empty: there is nothing to discipline",
            missing_evidence=["claims"],
        )

    problems: list[str] = []
    counts = {"claim_bearing": 0, "exploratory": 0, "negative": 0}
    for i, c in enumerate(claims):
        cid = c.get("claim_id", f"claim[{i}]")
        exploratory = bool(c.get("exploratory")) or c.get("status") == "exploratory"
        if exploratory:
            counts["exploratory"] += 1
            if c.get("rung") is not None:
                problems.append(f"{cid}: exploratory claim carries rung {c['rung']!r}")
        else:
            counts["claim_bearing"] += 1
            if not c.get("rung"):
                problems.append(f"{cid}: claim-bearing claim has no rung")
        if not c.get("provenance"):
            problems.append(f"{cid}: claim has no provenance block")
        if c.get("status") == "negative" or c.get("prominence") == "demoted":
            counts["negative"] += 1
            if c.get("prominence") == "demoted":
                problems.append(f"{cid}: negative result demoted below a positive one")

    if binding is not None and not binding.ok:
        problems.append(
            f"{len(binding.mismatched)} artifact hash mismatch(es): "
            f"{', '.join(a.name for a in binding.mismatched)} — a claim's provenance "
            f"cannot be verified against an unbound lineage"
        )

    detail = {"counts": counts, "binding_ok": None if binding is None else binding.ok}
    passed = not problems
    return GateVerdict(
        gate=gate, verdict=PASS if passed else FAIL, criteria=criteria,
        evidence=_filter(evidence, gate), fail_consequence=consequence,
        reason="; ".join(problems[:8]) + (" …" if len(problems) > 8 else ""),
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _criteria_for(prereg: Preregistration, gate: str) -> dict:
    return dict(prereg.gates.get(gate, {}))


def evaluate_gates(
    *,
    prereg: Preregistration,
    evidence: dict,
    binding: BindingReport | None = None,
) -> GateReport:
    """Evaluate all five gates from one evidence dict.

    ``evidence`` is a flat mapping of the keys in :data:`GATE_INPUTS`. A
    ``binding`` report, when supplied, is folded into G-E (an unbound lineage is a
    provenance failure) and attached to the report.

    G-E is evaluated first; if it does not pass, every other computed verdict is
    downgraded to ``blocked``. The arithmetic stays in ``detail`` for debugging,
    but what a report renders is ``blocked``.
    """
    ev = dict(evidence)

    g_e = gate_e(ev, _criteria_for(prereg, "G-E"), binding=binding)

    builders = {"G-A": gate_a, "G-B": gate_b, "G-C": gate_c, "G-D": gate_d}
    verdicts: dict[str, GateVerdict] = {}
    for gate, fn in builders.items():
        v = fn(ev, _criteria_for(prereg, gate))
        if not g_e.passed and v.verdict in {PASS, FAIL, UNDETERMINED}:
            v = GateVerdict(
                gate=v.gate, verdict=BLOCKED, criteria=v.criteria, evidence=v.evidence,
                fail_consequence=v.fail_consequence, blocked_by="G-E",
                reason=(f"G-E did not pass ({g_e.reason or 'claim discipline failed'}); "
                        f"the {v.verdict} computed from this evidence is not quotable"),
                missing_evidence=v.missing_evidence, detail=v.detail,
            )
        verdicts[gate] = v
    verdicts["G-E"] = g_e
    return GateReport(verdicts={g: verdicts[g] for g in GATE_IDS}, binding=binding)


# ---------------------------------------------------------------------------
# Builders from the stage modules' native outputs
# ---------------------------------------------------------------------------


def evidence_from_level1(level1_by_candidate: dict, *, ssl: str, rand: str,
                         frozen_head_t1: dict | None = None) -> dict:
    """G-A evidence from ``level1.run_level1`` outputs for two candidates.

    ``ssl`` is the pretrained candidate; ``rand`` is the same architecture with
    random weights. ``frozen_head_t1`` may supply ``{ssl, rand}`` values for the
    third term when the T1 probe has been run for both.
    """
    s = level1_by_candidate[ssl]
    r = level1_by_candidate[rand]
    ev = {
        "farm_separability_ssl": float(s["farm_separability_top1"]),
        "farm_separability_rand": float(r["farm_separability_top1"]),
        "collapse_disqualified": bool(s["collapse_disqualified"]),
    }
    if s.get("phase_probe_accuracy") is not None and r.get("phase_probe_accuracy") is not None:
        ev["phase_probe_ssl"] = float(s["phase_probe_accuracy"])
        ev["phase_probe_rand"] = float(r["phase_probe_accuracy"])
    if frozen_head_t1:
        ev["frozen_head_t1_ssl"] = float(frozen_head_t1["ssl"])
        ev["frozen_head_t1_rand"] = float(frozen_head_t1["rand"])
    return ev


def label_efficiency_table(curves_by_task: dict) -> dict:
    """Assemble G-B's ``label_efficiency`` from per-task transfer results.

    ``curves_by_task`` maps a claim-bearing task to a mapping of arm -> fraction ->
    value. Values must already be measured; a missing arm stays missing (the gate
    reports it rather than inventing a zero).
    """
    out: dict = {}
    for task, arms in curves_by_task.items():
        if task not in CLAIM_BEARING_TASKS:
            raise ValueError(
                f"{task} is not claim-bearing; G-B must not read it "
                f"(claim-bearing: {CLAIM_BEARING_TASKS})"
            )
        out[task] = {
            arm: {_fraction_key(f): float(v) for f, v in cells.items()}
            for arm, cells in arms.items()
        }
    return out


def transfer_grid_table(grid_by_task: dict) -> dict:
    """Assemble G-C's ``transfer_grid``; same contract as the label-efficiency table."""
    out: dict = {}
    for task, fractions in grid_by_task.items():
        if task not in CLAIM_BEARING_TASKS:
            raise ValueError(f"{task} is not claim-bearing; G-C must not read it")
        out[task] = {
            _fraction_key(f): {arm: float(v) for arm, v in arms.items()}
            for f, arms in fractions.items()
        }
    return out

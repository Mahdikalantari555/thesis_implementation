"""The complete experiment matrix (change tasks 0.4, 3.5, 12.1).

The design is explicit: *"The complete experiment matrix is generated and
committed **before** execution (``configs/benchmarks/matrix.yaml``, hash
recorded)"* and *"Priorities are predefined (incumbent first, then paradigms
adjacent to it)"*. This module is the single deterministic function from
(contract, applicability table, bindings) → cells, so:

- a cell that is not in the committed matrix cannot be executed;
- a cell whose pair is ``not-applicable`` is registered as ``Not Applicable``
  with the table's reason and is never replaced by a substitute loss
  (integrity rule 11);
- a patch-level cell whose v2 store binding is absent is registered as
  ``Deferred`` with reason ``patch-v2-unbound`` while the field-mean path keeps
  running (spec architecture-benchmarking / "Patch prerequisite unmet");
- every cell carries a predefined scheduling priority, so budget exhaustion
  defers a *suffix* of a known order rather than an arbitrary set.

``build_matrix`` is pure: no file writes, no torch, no clock. The committed
matrix is the sorted, hashed output of this function.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from ssl_anomaly.benchmarking.config import (
    ARCH_PRIORITY,
    ARCHITECTURES,
    BASELINES,
    INPUTS,
    MATRIX_PATH,
    SUPPRESSIONS,
)

#: Statuses a cell may hold at registration time. ``Running``/``Completed``/
#: ``Failed``/``Interrupted`` only ever come from the registry, never here.
REGISTER_STATUSES = ("Pending", "Deferred", "Not Applicable")

#: Suppression order used for pre-registered priority: baseline first.
SUPP_ORDER = ("S0", "S1", "S2", "S3")
PROTO_ORDER = ("V1", "V2", "V3")
TIER_ORDER = ("screening", "full")


@dataclass(frozen=True)
class ExperimentCell:
    """One node of the matrix: a fully specified, runnable experiment."""

    experiment_id: str
    cell_hash: str
    model_id: str
    paradigm: str
    backbone: str
    tier: str
    block: str
    suppression: str
    input: str
    fusion: str
    calibration: str
    protocol: str
    level: str
    variant: str
    seed: int
    priority: int
    status: str
    reason: str | None = None
    extra: dict = field(default_factory=dict, compare=False)

    @property
    def is_runnable(self) -> bool:
        return self.status == "Pending"

    @property
    def trained(self) -> bool:
        """Whether the model has an objective to optimise (baselines mostly don't)."""
        if self.model_id in ARCHITECTURES:
            return True
        return bool(BASELINES.get(self.model_id, {}).get("trained", False))

    def factors(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "suppression": self.suppression,
            "input": self.input,
            "fusion": self.fusion,
            "calibration": self.calibration,
            "protocol": self.protocol,
            "level": self.level,
            "variant": self.variant,
            "seed": self.seed,
        }

    def to_dict(self) -> dict:
        d = asdict(self)
        d["factors"] = self.factors()
        return d


# ---------------------------------------------------------------------------
# Ids and hashes
# ---------------------------------------------------------------------------


def cell_hash(factors: dict[str, Any]) -> str:
    """Stable 12-hex identity of a factor combination (key-order insensitive)."""
    blob = json.dumps(factors, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def experiment_id(factors: dict[str, Any]) -> str:
    """Human-readable id, e.g. ``A1.S0.I3.W1.C2.V1.farm.base.s13``."""
    variant = str(factors.get("variant", "base"))
    tail = "" if variant in ("", "base") else f".{variant}"
    return (
        f"{factors['model_id']}.{factors['suppression']}.{factors['input']}."
        f"{factors['fusion']}.{factors['calibration']}.{factors['protocol']}."
        f"{factors['level']}{tail}.s{factors['seed']}"
    )


def matrix_sha256(payload: dict) -> str:
    """Hash of a matrix document's *content* — prose notes are not hashed."""
    body = {
        "version": payload.get("version"),
        "benchmark_config_sha256": payload.get("benchmark_config_sha256"),
        "applicability_sha256": payload.get("applicability_sha256"),
        "n_cells": payload.get("n_cells"),
        "counts_by_status": payload.get("counts_by_status"),
        "cells": payload.get("cells"),
    }
    blob = json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ---------------------------------------------------------------------------
# Block expansion
# ---------------------------------------------------------------------------


def _resolve(value: Any, cfg) -> list | None:
    """``architectures`` / ``baselines`` / ``models`` / ``all`` / ``screening`` → list.

    ``None`` means "the axis default from the contract", which keeps ``all`` from
    being silently reinterpreted differently by different callers.
    """
    if isinstance(value, list):
        return list(value)
    key = str(value)
    if key == "all":
        return None
    if key == "architectures":
        return list(cfg.architectures)
    if key == "baselines":
        return list(cfg.baselines)
    if key == "models":
        return list(cfg.architectures) + list(cfg.baselines)
    if key == "screening":
        return list(cfg.screening_seeds)
    if key == "seeds":
        return list(cfg.seeds)
    raise ValueError(f"cannot resolve block axis value {value!r}")


def _axis(cfg, block: dict, name: str, default_attr: str) -> list:
    if name in block:
        got = _resolve(block[name], cfg)
        return list(got) if got is not None else list(getattr(cfg, default_attr))
    return list(getattr(cfg, default_attr))


def _feasibility_blocklist(cfg) -> dict[str, str]:
    """Model ids the feasibility gate has already ruled ``Not Applicable``."""
    out: dict[str, str] = {}
    for entry in cfg.raw.get("feasibility", {}).get("not_applicable", []) or []:
        if isinstance(entry, dict) and entry.get("id"):
            out[str(entry["id"])] = str(entry.get("reason", "recorded reason missing"))
    return out


def build_matrix(
    cfg,
    applicability: dict,
    *,
    patch_bound: bool | None = None,
    seeds: list[int] | None = None,
) -> list[ExperimentCell]:
    """Expand the contract's declared blocks into fully-identified cells.

    ``patch_bound`` defaults to the config's own ``patch_store_v2`` binding so a
    caller cannot forget the prerequisite; ``seeds`` restricts the seed set (used
    when a caller wants the screening projection only).
    """
    if patch_bound is None:
        patch_bound = cfg.patch_binding.bound
    pairs = applicability["pairs"]
    blocked = _feasibility_blocklist(cfg)
    input_weather = {k: bool(v["weather"]) for k, v in INPUTS.items()}

    cells: dict[str, ExperimentCell] = {}
    order: list[str] = []

    for block_name, block in (cfg.raw.get("blocks") or {}).items():
        block = block or {}
        models = _resolve(block.get("models", "architectures"), cfg)
        if models is None:
            models = list(cfg.architectures)
        supps = _axis(cfg, block, "suppression", "suppressions")
        inputs = _axis(cfg, block, "input", "inputs")
        fusions = _axis(cfg, block, "fusion", "fusions")
        cals = _axis(cfg, block, "calibration", "calibrations")
        protos = _axis(cfg, block, "protocol", "protocols")
        levels = _axis(cfg, block, "level", "levels")
        variants = [str(v) for v in (block.get("variants") or ["base"])]
        joint = block.get("joint") or [{}]
        block_seeds = _resolve(block.get("seeds", "seeds"), cfg)
        if block_seeds is None:
            block_seeds = list(cfg.seeds)
        if seeds is not None:
            restricted = [s for s in seeds if s in block_seeds]
            if not restricted:
                continue
            block_seeds = restricted
        tier = str(block.get("tier", "full"))
        extra = dict(block.get("extra") or {})

        for over, model, supp, inp, fus, cal, proto, level, var, seed in product(
            joint, models, supps, inputs, fusions, cals, protos, levels, variants, block_seeds
        ):
            # A `joint:` entry pins several factors at once (e.g. input I1 *with*
            # fusion W0). Without it, a joint axis would be expanded into
            # combinations that are structurally meaningless and would then have to
            # be explained away as Not Applicable.
            inp = str(over.get("input", inp))
            fus = str(over.get("fusion", fus))
            cal = str(over.get("calibration", cal))
            proto = str(over.get("protocol", proto))
            supp = str(over.get("suppression", supp))
            factors = {
                "model_id": model,
                "suppression": supp,
                "input": inp,
                "fusion": fus,
                "calibration": cal,
                "protocol": proto,
                "level": level,
                "variant": var,
                "seed": int(seed),
            }
            h = cell_hash(factors)
            status, reason = "Pending", None

            # (a) suppression applicability — the binding table (spec).
            row = pairs.get(f"{model}/{supp}")
            if row is None:
                status, reason = "Not Applicable", (
                    f"no applicability row for {model}/{supp}: an unregistered pair is not "
                    "the same as an allowed one"
                )
            elif row.get("status") == "not-applicable":
                status, reason = "Not Applicable", f"applicability: {row['reason']}"

            # (b) a weather-fusion arm on a weather-free input is a no-op ablation.
            if status == "Pending" and fus in ("W1", "W2") and not input_weather.get(inp, False):
                status, reason = "Not Applicable", (
                    f"{fus} conditions on weather but {inp} carries no weather channels; "
                    "the pair would duplicate its own control"
                )

            # (c) the Mamba feasibility gate, if it has already been triggered.
            if status == "Pending" and model in blocked:
                status, reason = "Not Applicable", f"feasibility gate: {blocked[model]}"

            # (d) hierarchical patch cells need the bound v2 patch store.
            if status == "Pending" and level == "patch" and not patch_bound:
                status, reason = "Deferred", "patch-v2-unbound"

            # (e) a rule/reference has no objective to suppress.
            trained = model in ARCHITECTURES or bool(BASELINES.get(model, {}).get("trained", False))
            if status == "Pending" and not trained and supp != "S0":
                status, reason = "Not Applicable", (
                    f"{BASELINES.get(model, {}).get('label', model)} has no training objective "
                    f"to which {supp} could apply"
                )

            arch = ARCHITECTURES.get(model, {})
            cell = ExperimentCell(
                experiment_id=experiment_id(factors),
                cell_hash=h,
                model_id=model,
                paradigm=arch.get("paradigm", BASELINES.get(model, {}).get("kind", "reference")),
                backbone=arch.get("backbone", "none"),
                tier=tier,
                block=block_name,
                suppression=supp,
                input=inp,
                fusion=fus,
                calibration=cal,
                protocol=proto,
                level=level,
                variant=var,
                seed=int(seed),
                priority=0,
                status=status,
                reason=reason,
                extra=extra,
            )
            if h in cells:
                prev = cells[h]
                if prev.block != block_name:
                    seen = list(prev.extra.get("also_in_blocks", []))
                    if block_name not in seen:
                        seen.append(block_name)
                    cells[h] = _with_extra(prev, {"also_in_blocks": seen})
                continue
            cells[h] = cell
            order.append(h)

    return _assign_priority([cells[h] for h in order])


def _with_extra(cell: ExperimentCell, extra: dict) -> ExperimentCell:
    merged = {**cell.extra, **extra}
    return ExperimentCell(**{**asdict(cell), "extra": merged})


def _priority_key(cell: ExperimentCell) -> tuple:
    """Screening first, incumbent first, then paradigms adjacent to it (§Budget)."""
    try:
        model_i = ARCH_PRIORITY.index(cell.model_id)
    except ValueError:
        base = list(BASELINES)
        model_i = len(ARCH_PRIORITY) + (base.index(cell.model_id) if cell.model_id in base else 99)
    return (
        0 if cell.status == "Pending" else 1,
        TIER_ORDER.index(cell.tier) if cell.tier in TIER_ORDER else 9,
        PROTO_ORDER.index(cell.protocol) if cell.protocol in PROTO_ORDER else 9,
        SUPP_ORDER.index(cell.suppression) if cell.suppression in SUPP_ORDER else 9,
        model_i,
        cell.block,
        cell.seed,
    )


def _assign_priority(cells: list[ExperimentCell]) -> list[ExperimentCell]:
    ordered = sorted(cells, key=_priority_key)
    return [ExperimentCell(**{**asdict(c), "priority": i}) for i, c in enumerate(ordered)]


# ---------------------------------------------------------------------------
# Commitment: document / verify
# ---------------------------------------------------------------------------


def matrix_document(
    cfg,
    cells: list[ExperimentCell],
    *,
    applicability_sha: str,
    note: str = "",
) -> dict:
    """The YAML-serialisable matrix document that gets committed."""
    counts: dict[str, int] = {s: 0 for s in REGISTER_STATUSES}
    for c in cells:
        counts[c.status] = counts.get(c.status, 0) + 1
    doc = {
        "version": 1,
        "created": str(cfg.raw.get("created", "")),
        "change": str(cfg.raw.get("change", "")),
        "benchmark_config_sha256": cfg.document_sha256,
        "applicability_sha256": applicability_sha,
        "n_cells": len(cells),
        "counts_by_status": counts,
        "note": note
        or (
            "GENERATED FILE — do not hand-edit. Regenerate with "
            "`python scripts/benchmarks/build_matrix.py` and re-record the hash in "
            "configs/benchmarks/pre_registration.yaml. A cell absent from this file "
            "has no right to run; a cell present but not `Pending` carries the reason "
            "it did not run (integrity rule 8: incomplete coverage is stated, not hidden)."
        ),
        "cells": [
            {
                "id": c.experiment_id,
                "hash": c.cell_hash,
                "model": c.model_id,
                "paradigm": c.paradigm,
                "backbone": c.backbone,
                "tier": c.tier,
                "block": c.block,
                "suppression": c.suppression,
                "input": c.input,
                "fusion": c.fusion,
                "calibration": c.calibration,
                "protocol": c.protocol,
                "level": c.level,
                "variant": c.variant,
                "seed": c.seed,
                "priority": c.priority,
                "status": c.status,
                **({"reason": c.reason} if c.reason else {}),
            }
            for c in sorted(cells, key=lambda x: x.priority)
        ],
    }
    doc["matrix_sha256"] = matrix_sha256(doc)
    return doc


def write_matrix(doc: dict, path: str | Path | None = None) -> Path:
    """Write the matrix as YAML with one JSON-flow cell per line.

    The cells are emitted as one-line flow mappings rather than 15-line block
    mappings on purpose: the file stays diffable in review, and a reviewer
    comparing two pre-registrations can see which single cell moved.
    """
    p = Path(path) if path is not None else MATRIX_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    head = {k: v for k, v in doc.items() if k != "cells"}
    lines = [
        "# GENERATED FILE — do not hand-edit (see the note field below).",
        yaml.safe_dump(head, sort_keys=False, width=110, allow_unicode=True).rstrip(),
        "cells:",
    ]
    for c in doc["cells"]:
        lines.append("  - " + json.dumps(c, sort_keys=False, separators=(", ", ": ")))
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def read_matrix(path: str | Path | None = None) -> dict:
    p = Path(path) if path is not None else MATRIX_PATH
    if not p.is_file():
        raise FileNotFoundError(
            f"committed matrix not found at {p}. Run scripts/benchmarks/build_matrix.py — "
            "execution is not permitted without a committed matrix (integrity rule 7)."
        )
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def verify_committed_matrix(
    cfg,
    applicability: dict,
    *,
    path: str | Path | None = None,
) -> dict:
    """Regenerate the matrix from the live contract and compare with the committed file.

    This is the drift check the design's risk table calls for ("registry drift —
    resume requires hash match"). Returns ``{ok, committed_sha, regenerated_sha,
    added, removed, changed, problems}``; a resume refuses to continue unless
    ``ok``.
    """
    doc = read_matrix(path)
    cells = build_matrix(cfg, applicability)
    from ssl_anomaly.benchmarking.config import applicability_sha256_of

    live_app_sha = applicability_sha256_of(applicability if isinstance(applicability, dict) else {})
    regen = matrix_document(cfg, cells, applicability_sha=live_app_sha, note=doc.get("note", ""))
    committed = {c["hash"]: c for c in doc.get("cells", [])}
    fresh = {c["hash"]: c for c in regen["cells"]}
    problems: list[str] = []
    if doc.get("applicability_sha256") != live_app_sha:
        problems.append(
            f"committed matrix cites applicability {str(doc.get('applicability_sha256'))[:12]}…, the live "
            f"table hashes to {live_app_sha[:12]}… — the suppression applicability file changed, which "
            "changes which cells exist at all")
    if doc.get("benchmark_config_sha256") != cfg.document_sha256:
        problems.append(
            f"committed matrix was generated from a different config "
            f"(matrix says {doc.get('benchmark_config_sha256','<none>')[:12]}…, config is "
            f"{cfg.document_sha256[:12]}…)"
        )
    if doc.get("matrix_sha256") != matrix_sha256(doc):
        problems.append("matrix_sha256 in the committed document does not match its own contents")
    added = sorted(set(fresh) - set(committed))
    removed = sorted(set(committed) - set(fresh))
    changed = sorted(
        h for h in set(fresh) & set(committed)
        if {k: v for k, v in fresh[h].items() if k != "priority"}
        != {k: v for k, v in committed[h].items() if k != "priority"}
    )
    if added:
        problems.append(f"{len(added)} cell(s) are now producible but not committed (e.g. "
                        f"{[fresh[h]['id'] for h in added[:3]]})")
    if removed:
        problems.append(f"{len(removed)} committed cell(s) are no longer producible (e.g. "
                        f"{[committed[h]['id'] for h in removed[:3]]})")
    if changed:
        problems.append(f"{len(changed)} committed cell(s) changed their factors/status")
    return {
        "ok": not problems,
        "committed_sha256": doc.get("matrix_sha256"),
        "regenerated_sha256": regen["matrix_sha256"],
        "n_committed": len(committed),
        "n_regenerated": len(fresh),
        "added": [fresh[h]["id"] for h in added],
        "removed": [committed[h]["id"] for h in removed],
        "changed": [fresh[h]["id"] for h in changed],
        "problems": problems,
    }


def status_counts(cells: list[ExperimentCell]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for c in cells:
        counts[c.status] = counts.get(c.status, 0) + 1
    return dict(sorted(counts.items()))


def coverage_line(counts: dict[str, int], *, total: int | None = None) -> str:
    """The sentence every report must carry (spec "Executive summary coverage statement").

    Built from counts only — it cannot overstate coverage because it has no way to
    name a cell that is not in ``counts``.
    """
    total = int(total if total is not None else sum(counts.values()))
    done = counts.get("Completed", 0)
    deferred = counts.get("Deferred", 0)
    na = counts.get("Not Applicable", 0)
    failed = counts.get("Failed", 0) + counts.get("Interrupted", 0)
    pending = counts.get("Pending", 0)
    running = counts.get("Running", 0)
    largest = max(counts.items(), key=lambda kv: kv[1])[0] if counts else "n/a"
    parts = [
        f"Completed {done}/{total}",
        f"Deferred {deferred}",
        f"Not Applicable {na}",
    ]
    if failed:
        parts.append(f"Failed/Interrupted {failed}")
    if pending or running:
        parts.append(f"Pending/Running {pending + running}")
    sentence = (
        f"COVERAGE: {'; '.join(parts)}. "
        f"Largest incomplete area by cell count: {largest}. "
        "No Deferred or Not Applicable cell may be cited as passing evidence."
    )
    return sentence

"""Experiment registry: one explicit status for every cell, always (task 1.5).

Three promises, each of which is a spec requirement rather than a nicety:

1. **Completeness before execution.** :meth:`Registry.register` refuses to accept a
   cell set whose matrix hash differs from the committed pre-registration, and the
   runner refuses to start without registration — so "every matrix cell exists in
   the registry with status ``Pending``" (scenario *Pre-execution completeness*)
   holds by construction.
2. **Exactly seven states.** :data:`REGISTRY_STATES` is the closed set; an unknown
   status raises rather than being coerced. ``Not Applicable`` cannot be written
   without a machine-checkable reason, and ``Deferred`` cannot be written without
   one either, because an unexplained deferral is indistinguishable from a dropped
   experiment.
3. **Crash honesty.** A cell left ``Running`` when the file is re-opened was
   interrupted, not completed; :meth:`Registry.recover` marks it ``Interrupted``,
   records the last checkpoint reference and returns it for resume.

Storage is a JSON snapshot plus an append-only JSONL transition log: human-readable
in Drive, diffable in Git (text-only), and atomic to write (temp + ``os.replace``),
which is what makes it safe against the Colab preemption that motivates it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

REGISTRY_STATES: tuple[str, ...] = (
    "Pending",
    "Running",
    "Completed",
    "Failed",
    "Interrupted",
    "Deferred",
    "Not Applicable",
)

#: Statuses that mean "this cell has an outcome and will not be re-run by a resume".
FINAL_STATES: tuple[str, ...] = ("Completed", "Failed", "Deferred", "Not Applicable")

#: Statuses that need a reason to be recorded with them.
REQUIRES_REASON: dict[str, str] = {
    "Not Applicable": "a machine-checkable reason is required for Not Applicable",
    "Deferred": "a reason is required for Deferred (budget exhaustion, unmet prerequisite, …)",
    "Failed": "the error text is required so the failure can be reported verbatim",
}


class RegistryError(RuntimeError):
    """The registry was asked for something it must not do."""


@dataclass
class RegistryEntry:
    """One cell's live registry record."""

    cell_hash: str
    experiment_id: str
    status: str = "Pending"
    reason: str | None = None
    priority: int = 0
    tier: str = "full"
    model_id: str = ""
    attempts: int = 0
    started_at: float | None = None
    finished_at: float | None = None
    wall_seconds: float | None = None
    checkpoint: str | None = None
    manifest: str | None = None
    metrics: dict = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RegistryEntry:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def check_transition(frm: str, to: str) -> None:
    """The only legal state moves. Anything else raises.

    ``Running`` may only be entered from ``Pending``, ``Interrupted`` or
    ``Failed`` (the latter only through an explicit ``requeue``); a completed cell
    never returns to ``Running`` — re-running a completed cell means deleting its
    record deliberately, which the caller must do in the open.
    """
    legal: dict[str, set[str]] = {
        "Pending": {"Running", "Deferred", "Not Applicable", "Failed"},
        # A gate that fires *after* training (the Mamba feasibility gate is the case
        # in point) must be able to record Not Applicable with its measurements; the
        # alternative would be Failed, which would misreport a design decision as an
        # error. Requeue-style retries still go through Pending, never back to
        # Running from a final state.
        "Running": {"Completed", "Failed", "Interrupted", "Deferred", "Not Applicable"},
        "Interrupted": {"Running", "Deferred", "Failed"},
        # A failed cell may be parked, reclassified, or requeued. The last only
        # through Registry.requeue(), which demands a reason — so the docstring's
        # "Failed → Running only via an explicit requeue" is a real gate, not prose
        # (spec: failed cells MAY be re-run after a recorded config change).
        "Failed": {"Deferred", "Not Applicable", "Pending"},
        "Deferred": {"Pending"},
        "Not Applicable": set(),
        "Completed": set(),
    }
    if to not in legal.get(frm, set()):
        raise RegistryError(f"illegal registry transition {frm} -> {to}")


def check_reason(status: str, reason: str | None) -> None:
    need = REQUIRES_REASON.get(status)
    if need and not (reason and str(reason).strip()):
        raise RegistryError(f"{status}: {need}")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class Registry:
    """JSON-snapshot registry with an append-only transition log."""

    def __init__(self, path: str | Path, *, allow_new: bool = True) -> None:
        self.path = Path(path)
        self.log_path = self.path.with_suffix(".log.jsonl")
        self._entries: dict[str, RegistryEntry] = {}
        self._meta: dict = {}
        if self.path.is_file():
            self._load()
        elif not allow_new:
            raise RegistryError(f"no registry at {self.path} (allow_new=False)")

    # ---- persistence -----------------------------------------------------
    def _load(self) -> None:
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        self._meta = {k: v for k, v in doc.items() if k != "entries"}
        for raw in doc.get("entries", []):
            e = RegistryEntry.from_dict(raw)
            self._entries[e.cell_hash] = e

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            **self._meta,
            "n_entries": len(self._entries),
            "counts_by_status": self.counts(),
            "entries": [self._entries[k].to_dict() for k in sorted(self._entries)],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.path)

    def _log(self, cell_hash: str, frm: str | None, to: str, reason: str | None, extra: dict) -> None:
        rec = {
            "ts": time.time(),
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cell_hash": cell_hash,
            "from": frm,
            "to": to,
            **({"reason": reason} if reason else {}),
            **extra,
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    # ---- registration ----------------------------------------------------
    def register(
        self,
        cells: Iterable,
        *,
        matrix_sha256: str,
        replace_existing: bool = False,
    ) -> dict:
        """Insert every cell of the committed matrix (idempotent on re-register).

        An existing cell's *status* is never overwritten by registration — the
        matrix says what should run, the registry says what has happened. Only
        ``replace_existing=True`` (an explicit, recorded action) re-seeds statuses.
        """
        if not matrix_sha256:
            raise RegistryError("register() requires the committed matrix hash")
        if self._meta.get("matrix_sha256") not in (None, matrix_sha256):
            raise RegistryError(
                "registry is bound to a different matrix: "
                f"{self._meta['matrix_sha256'][:12]}… ≠ {matrix_sha256[:12]}… — "
                "re-registering across matrices would mix two pre-registrations"
            )
        added = updated = 0
        for c in cells:
            h = c.cell_hash
            if h in self._entries and not replace_existing:
                # keep history; refresh only the non-outcome metadata
                e = self._entries[h]
                if e.status == c.status or e.status == "Pending":
                    if e.status != c.status:
                        e.status = c.status
                        e.reason = c.reason
                        updated += 1
                continue
            e = RegistryEntry(
                cell_hash=h,
                experiment_id=c.experiment_id,
                status=c.status,
                reason=c.reason,
                priority=c.priority,
                tier=c.tier,
                model_id=c.model_id,
            )
            check_reason(e.status, e.reason)
            self._entries[h] = e
            added += 1
            self._log(h, None, e.status, e.reason, {"via": "register"})
        self._meta.update(
            {
                "matrix_sha256": matrix_sha256,
                "registered_at": self._meta.get("registered_at") or time.time(),
                "n_registered": len(self._entries),
            }
        )
        self._flush()
        return {"added": added, "updated": updated, "total": len(self._entries)}

    # ---- queries ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, cell_hash: str) -> bool:
        return cell_hash in self._entries

    def __getitem__(self, cell_hash: str) -> RegistryEntry:
        try:
            return self._entries[cell_hash]
        except KeyError:  # pragma: no cover - guarded by callers
            raise RegistryError(f"cell {cell_hash} is not registered") from None

    def get(self, cell_hash: str) -> RegistryEntry | None:
        return self._entries.get(cell_hash)

    def entries(self, *, status: str | None = None) -> list[RegistryEntry]:
        vals = list(self._entries.values())
        if status is not None:
            vals = [e for e in vals if e.status == status]
        return sorted(vals, key=lambda e: e.priority)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self._entries.values():
            out[e.status] = out.get(e.status, 0) + 1
        return dict(sorted(out.items()))

    @property
    def matrix_sha256(self) -> str | None:
        return self._meta.get("matrix_sha256")

    # ---- mutations -------------------------------------------------------
    def transition(
        self,
        cell_hash: str,
        to: str,
        *,
        reason: str | None = None,
        checkpoint: str | None = None,
        error: str | None = None,
        metrics: dict | None = None,
        manifest: str | None = None,
    ) -> RegistryEntry:
        if to not in REGISTRY_STATES:
            raise RegistryError(
                f"{to!r} is not a registry state; the closed set is {list(REGISTRY_STATES)}"
            )
        e = self[cell_hash]
        check_transition(e.status, to)
        if to == "Failed" and not (reason and reason.strip()) and error:
            # The verbatim error text doubles as the reason: a second, hand-written
            # justification is either a duplicate of it or a euphemism for it.
            reason = error
        check_reason(to, reason if to in REQUIRES_REASON else reason)
        frm = e.status
        e.status = to
        if to in REQUIRES_REASON:
            e.reason = reason
        elif to in ("Pending", "Running", "Completed"):
            # A requeue arrives as Pending *with* a reason, and that reason is the
            # record of why the cell was reopened — clearing it here would keep the
            # audit trail only in the log and make the snapshot uninterpretable.
            e.reason = reason if (to == "Pending" and reason) else None
        if to == "Running":
            e.attempts += 1
            e.started_at = time.time()
            e.finished_at = None
        if to in FINAL_STATES:
            e.finished_at = time.time()
            if e.started_at:
                e.wall_seconds = round(e.finished_at - e.started_at, 3)
        if checkpoint is not None:
            e.checkpoint = checkpoint
        if error is not None:
            e.error = error
        if manifest is not None:
            e.manifest = manifest
        if metrics:
            e.metrics.update(metrics)
        self._log(cell_hash, frm, to, reason, {"error": error} if error else {})
        self._flush()
        return e

    def claim_next(self, *, tier: str | None = None, model_ids: Iterable[str] | None = None) -> RegistryEntry | None:
        """Atomically take the highest-priority ``Pending`` cell (already-scheduled
        deferred/NA cells are never claimed)."""
        wanted = set(model_ids) if model_ids is not None else None
        for e in self.entries(status="Pending"):
            if tier is not None and e.tier != tier:
                continue
            if wanted is not None and e.model_id not in wanted:
                continue
            return self.transition(e.cell_hash, "Running")
        return None

    def requeue(self, cell_hash: str, *, reason: str) -> RegistryEntry:
        """``Failed`` → ``Pending`` for a documented retry (e.g. after a code fix)."""
        e = self[cell_hash]
        if e.status not in ("Failed", "Deferred"):
            raise RegistryError(f"{e.status} cells cannot be requeued; only Failed/Deferred can")
        if not reason.strip():
            raise RegistryError("requeue requires a recorded reason (an unexplained retry is drift)")
        return self.transition(cell_hash, "Pending", reason=f"requeued: {reason}")

    def defer_remaining(self, *, reason: str, tier: str | None = None) -> list[str]:
        """Budget exhaustion path: every unfinished cell becomes ``Deferred``.

        Nothing is dropped (spec *Budget exhaustion*): the deferred cells stay in
        the registry, in every report, and with this reason attached.
        """
        deferred: list[str] = []
        for e in list(self._entries.values()):
            if e.status not in ("Pending", "Running", "Interrupted"):
                continue
            if tier is not None and e.tier != tier:
                continue
            self.transition(e.cell_hash, "Deferred", reason=reason)
            deferred.append(e.cell_hash)
        return deferred

    # ---- crash recovery --------------------------------------------------
    def recover(self, *, checkpoint_lookup=None) -> list[RegistryEntry]:
        """Mark cells found ``Running`` as ``Interrupted`` and return them for resume.

        ``checkpoint_lookup(cell_hash) -> str | None`` lets the caller supply the
        last valid checkpoint reference so the resume can point at it; the registry
        never guesses one.
        """
        out: list[RegistryEntry] = []
        for e in self.entries(status="Running"):
            ck = checkpoint_lookup(e.cell_hash) if checkpoint_lookup else None
            rec = self.transition(
                e.cell_hash,
                "Interrupted",
                reason="found Running at load: the process died without finishing this cell",
                checkpoint=ck,
            )
            out.append(rec)
        return out

    def resumable(self) -> list[RegistryEntry]:
        """``Interrupted`` cells, in priority order — what a resume should restore."""
        return self.entries(status="Interrupted")

    # ---- audits ----------------------------------------------------------
    def audit(self) -> dict:
        """Registry audit (task 15.4): every cell has an explicit final status.

        A cell still ``Pending``/``Running`` at audit time is reported as an
        incomplete area; the audit never converts it into anything flattering.
        """
        problems: list[str] = []
        unknown = sorted({e.status for e in self._entries.values()} - set(REGISTRY_STATES))
        if unknown:
            problems.append(f"statuses outside the closed set: {unknown}")
        missing_reason = [
            e.experiment_id
            for e in self._entries.values()
            if e.status in REQUIRES_REASON and not (e.reason and e.reason.strip())
        ]
        if missing_reason:
            problems.append(
                f"{len(missing_reason)} cell(s) carry Deferred/Not Applicable/Failed without a reason "
                f"(e.g. {missing_reason[:3]})"
            )
        nonfinal = [e.experiment_id for e in self._entries.values() if e.status not in FINAL_STATES]
        completed_no_metrics = [
            e.experiment_id for e in self._entries.values() if e.status == "Completed" and not e.metrics
        ]
        if completed_no_metrics:
            problems.append(
                f"{len(completed_no_metrics)} Completed cell(s) carry no metrics "
                f"(e.g. {completed_no_metrics[:3]}) — a completion without numbers is not evidence"
            )
        return {
            "ok": not problems,
            "n_entries": len(self._entries),
            "counts": self.counts(),
            "n_nonfinal": len(nonfinal),
            "nonfinal_examples": nonfinal[:10],
            "matrix_sha256": self.matrix_sha256,
            "problems": problems,
        }

    def coverage_counts(self) -> dict[str, int]:
        """Counts in the shape :func:`ssl_anomaly.benchmarking.matrix.coverage_line` wants."""
        return {s: self.counts().get(s, 0) for s in REGISTRY_STATES}

    def iter_metrics(self) -> Iterator[tuple[RegistryEntry, dict]]:
        for e in self.entries():
            if e.metrics:
                yield e, e.metrics

    def write_markdown(self, path: str | Path) -> Path:
        """Human-readable registry table (incomplete cells visible)."""
        from ssl_anomaly.benchmarking.matrix import coverage_line  # lazy: no import cycle

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        counts = self.counts()
        lines = [
            "# Experiment registry — state table",
            "",
            f"- cells registered: {len(self._entries)}",
            f"- committed matrix hash: `{self.matrix_sha256}`",
            "- status counts: " + ", ".join(f"{k}={counts.get(k, 0)}" for k in REGISTRY_STATES),
            "",
            coverage_line(counts, total=len(self._entries)),
            "",
            "| priority | id | tier | model | status | wall s | reason / error |",
            "|---:|---|---|---|---|---:|---|",
        ]
        for e in self.entries():
            lines.append(
                f"| {e.priority} | `{e.experiment_id}` | {e.tier} | {e.model_id} | "
                f"{e.status} | {e.wall_seconds if e.wall_seconds is not None else '—'} | "
                f"{(e.reason or e.error or '').replace('|', '/')} |"
            )
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return p

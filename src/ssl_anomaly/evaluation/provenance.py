"""Provenance and claim-rung discipline for the capstone.

Every number this package produces is either (a) written into a
:class:`ClaimRecord` (rung-tagged, provenance-bound) or (b) recorded with an
explicit ``absent``/``deferred`` status. There is no third option — AGENTS.md
prime directive 2 and the ``generalization-and-claim-semantics`` spec forbid a
value that appears without provenance or a claim that appears without a rung.

The validation ladder (AGENTS.md §2):

    L1 learns temporal growth patterns
    L2 detects controlled deviations
    L3 acceptable on new years / farms
    L4 agrees with independent agronomic evidence
    L5 beats baselines on performance / efficiency

Current honest state at the time of writing: **L2 not cleared** (finding 25);
L5 unmet. Nothing here may raise a rung: rung upgrades are earned by evidence,
never assigned.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ssl_anomaly.evaluation.binding import REPO_ROOT, canonical_json_sha256

#: The rungs a record may carry. ``None`` marks a purely descriptive artifact.
VALID_RUNGS = ("L1", "L2", "L3", "L4", "L5")

#: Honest baseline state, recorded by findings 25 and re-asserted in every
#: capstone output so a reader cannot mistake a screening number for L5.
CURRENT_HONEST_RUNG = "L2"
CURRENT_RUNG_CLEARED = False
L5_MET = False

#: Claim status vocabulary. ``absent`` and ``deferred`` are first-class: they
#: replace a number rather than accompany one.
CLAIM_STATUSES = ("measured", "absent", "deferred", "not_applicable", "failed")


def git_sha(short: bool = True) -> str:
    """Best-effort git SHA; ``"unknown"`` rather than a fabricated value."""
    try:
        args = ["git", "rev-parse", "--short", "HEAD"] if short else ["git", "rev-parse", "HEAD"]
        out = subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, text=True, timeout=10, check=False
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def package_versions() -> dict[str, str]:
    """Versions of the libraries that change numeric results."""
    versions: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("numpy", "scipy", "pandas", "torch", "scikit-learn", "statsmodels", "xarray", "zarr"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - absence is recorded, never hidden
            versions[name] = "absent"
    return versions


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def device_label() -> str:
    """``cuda``/``cpu``/``absent`` without importing torch when it is missing."""
    try:
        import torch  # noqa: PLC0415 - optional dependency

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "absent"


@dataclass
class Provenance:
    """The provenance block attached to every capstone artifact."""

    git_sha: str = field(default_factory=git_sha)
    preregistration_sha256: str = ""
    split_hashes: dict[str, str] = field(default_factory=dict)
    seeds: tuple[int, ...] = ()
    versions: dict[str, str] = field(default_factory=package_versions)
    device: str = field(default_factory=device_label)
    wall_time_s: float | None = None
    data_source: str = ""
    created: str = field(default_factory=utc_now)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "git_sha": self.git_sha,
            "preregistration_sha256": self.preregistration_sha256,
            "split_hashes": dict(self.split_hashes),
            "seeds": list(self.seeds),
            "versions": dict(self.versions),
            "device": self.device,
            "wall_time_s": self.wall_time_s,
            "data_source": self.data_source,
            "created": self.created,
            **self.extra,
        }

    @classmethod
    def for_run(
        cls,
        *,
        preregistration_sha256: str,
        seeds: tuple[int, ...],
        data_source: str,
        split_hashes: dict[str, str] | None = None,
        wall_time_s: float | None = None,
        extra: dict | None = None,
    ) -> Provenance:
        return cls(
            preregistration_sha256=preregistration_sha256,
            split_hashes=split_hashes or {},
            seeds=tuple(seeds),
            data_source=data_source,
            wall_time_s=wall_time_s,
            extra=extra or {},
        )


@dataclass
class ClaimRecord:
    """One quotable result: value, uncertainty, rung, provenance, status.

    A record with ``status != "measured"`` MUST NOT carry a ``value`` — the
    constructor enforces this so a missing measurement can never leak into a
    table as a number.
    """

    claim_id: str
    statement: str
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None
    prevalence: float | None = None
    rung: str | None = None
    evidence_artifact: str = ""
    status: str = "measured"
    reason: str = ""
    exploratory: bool = False
    provenance: dict = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in CLAIM_STATUSES:
            raise ValueError(f"unknown claim status {self.status!r}; use {CLAIM_STATUSES}")
        if self.rung is not None and self.rung not in VALID_RUNGS:
            raise ValueError(f"unknown rung {self.rung!r}; use {VALID_RUNGS}")
        if self.status == "measured" and self.value is None:
            raise ValueError(
                f"claim {self.claim_id!r} is 'measured' but carries no value — "
                "record absent/deferred instead"
            )
        if self.status != "measured" and self.value is not None:
            raise ValueError(
                f"claim {self.claim_id!r} is {self.status!r} but carries a value "
                f"({self.value}) — an unavailable measurement must not be synthesised"
            )
        if self.ci_low is not None and self.ci_high is not None and self.ci_low > self.ci_high:
            raise ValueError(f"claim {self.claim_id!r}: ci_low > ci_high")
        if self.exploratory and self.rung is not None:
            raise ValueError(
                f"claim {self.claim_id!r} is exploratory but tagged with rung "
                f"{self.rung!r}; exploratory artifacts carry no rung claim"
            )

    @property
    def quotable(self) -> bool:
        """A record may be quoted in the report only when measured and rung-tagged."""
        return self.status == "measured" and (self.exploratory or self.rung is not None)

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "statement": self.statement,
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n": self.n,
            "prevalence": self.prevalence,
            "rung": self.rung,
            "evidence_artifact": self.evidence_artifact,
            "status": self.status,
            "reason": self.reason,
            "exploratory": self.exploratory,
            "quotable": self.quotable,
            "provenance": self.provenance,
            "tags": list(self.tags),
        }

    def hash(self) -> str:
        return canonical_json_sha256(self.to_dict())


def absent_claim(claim_id: str, statement: str, *, reason: str, artifact: str = "") -> ClaimRecord:
    """A claim whose measurement does not exist. Never given a value."""
    return ClaimRecord(
        claim_id=claim_id,
        statement=statement,
        status="absent",
        reason=reason,
        evidence_artifact=artifact,
    )


def deferred_claim(claim_id: str, statement: str, *, reason: str, artifact: str = "") -> ClaimRecord:
    """A claim deferred by budget/prerequisite. Visible, never a pass."""
    return ClaimRecord(
        claim_id=claim_id,
        statement=statement,
        status="deferred",
        reason=reason,
        evidence_artifact=artifact,
    )


def write_json(path: str | Path, payload: dict) -> Path:
    """Write JSON with ``allow_nan=False`` and a trailing newline.

    ``allow_nan=False`` is the enforcement of "no NaN pretending to be a
    measurement": a NaN anywhere in the payload raises instead of producing a
    file a reader would take at face value.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False, allow_nan=False)
    p.write_text(text + "\n", encoding="utf-8")
    return p


def honest_state_block(
    *,
    measured: list[str] | None = None,
    absent: list[str] | None = None,
    deferred: list[str] | None = None,
) -> dict:
    """The rung disclaimer stamped into every capstone output.

    Optionally carries the run's own measured/absent/deferred inventory, so a
    reader sees what this artifact did and did not establish *in addition to* the
    standing rung disclaimer. The inventory is inventory, not evidence: listing a
    subject under ``measured`` says a number exists, not that it is quotable —
    the gates decide that.
    """
    block = {
        "current_rung": CURRENT_HONEST_RUNG,
        "current_rung_cleared": CURRENT_RUNG_CLEARED,
        "L5_met": L5_MET,
        "rung_source": "docs/findings/25_patch_full_v2_fixed_protocol.md",
        "note": (
            "L2 is not cleared and L5 is unmet at the time of this run. Synthetic "
            "(controlled) detection is never stress evidence; 'anomaly' != 'stress' "
            "before L4. No record in this artifact upgrades a rung."
        ),
    }
    if measured is not None or absent is not None or deferred is not None:
        block["inventory"] = {
            "measured": list(measured or []),
            "absent": list(absent or []),
            "deferred": list(deferred or []),
        }
    return block

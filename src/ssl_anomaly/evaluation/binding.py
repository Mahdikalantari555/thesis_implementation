"""Artifact binding gate for the capstone (fails closed, dev-only).

Two jobs, both from the change's task 0.2/0.3 and the
``generalization-and-claim-semantics`` spec ("Preservation of the frozen
evidence base" / "Artifact mutation detected"):

1. **Pre-registration freeze** — the committed
   ``configs/evaluation/pre_registration.yaml`` is the single source of truth
   for hypotheses, gates, label fractions, seeds and the expected artifact
   hashes. :func:`load_preregistration` reads it and records its own hash so a
   later deviation can be detected mechanically.
2. **Artifact binding** — :func:`bind_artifacts` verifies that the v2 chain
   artifacts on disk are the ones the pre-registration names (sha256 prefixes).
   Any mismatch raises :class:`ArtifactBindingError`. A missing artifact is
   *reported*, never substituted: the caller decides whether to fail closed
   (training) or record ``absent`` (a report section).

Design rules honoured here:

- No torch / no heavy imports: the gate must run in a minimal environment and
  from a clean checkout.
- No network, no writes outside the requested report path.
- The locked ``test`` split is never enumerated here; only split membership
  counts are read, and only for binding verification.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
PREREGISTRATION_PATH = REPO_ROOT / "configs/evaluation/pre_registration.yaml"

#: Only the sha256 hex prefix stored in the pre-registration is compared; the
#: repository records 8-hex prefixes (e.g. ``30ecbe10``) in findings and docs.
SHA_PREFIX_LEN = 8


class ArtifactBindingError(RuntimeError):
    """A required artifact is missing or its hash deviates from the freeze."""


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Streaming sha256 of a file (never loads it whole — the stores are large)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_path(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """sha256 of a file OR a directory store (e.g. a ``.zarr`` chunk tree).

    ``patch_store_v3`` in the pre-registration is a directory store; hashing it
    with :func:`sha256_file` raised ``IsADirectoryError`` on any machine where
    the store exists, so every real manifest build failed before this dispatch
    existed. A directory is hashed deterministically: relative POSIX paths in
    sorted order, each file's digest folded in — any byte change anywhere in the
    tree changes the digest, so binding stays fail-closed.
    """
    p = Path(path)
    if not p.is_dir():
        return sha256_file(p, chunk=chunk)
    h = hashlib.sha256()
    files = sorted(f for f in p.rglob("*") if f.is_file())
    if not files:
        raise ArtifactBindingError(f"directory store has no files to hash: {p}")
    for f in files:
        rel = f.relative_to(p).as_posix()
        h.update(rel.encode("utf-8"))
        h.update(b"\x00")
        h.update(bytes.fromhex(sha256_file(f, chunk=chunk)))
    return h.hexdigest()


def sha256_prefix(digest: str, n: int = SHA_PREFIX_LEN) -> str:
    return digest[:n].lower()


def canonical_json_sha256(payload: Any) -> str:
    """Hash a JSON-serialisable payload deterministically (sorted keys, no NaN).

    ``allow_nan=False`` is deliberate: a NaN reaching a hash means a metric
    record was written with a not-measured value pretending to be a number.
    """
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def preregistration_sha256(path: str | Path = PREREGISTRATION_PATH) -> str:
    """Hash of the pre-registration *document* itself (byte-level freeze)."""
    return sha256_file(path)


# ---------------------------------------------------------------------------
# Pre-registration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preregistration:
    """Parsed pre-registration with its own hash (freeze record)."""

    payload: dict
    path: Path
    document_sha256: str

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.payload["seeds"])

    @property
    def label_fractions(self) -> tuple[float, ...]:
        return tuple(float(f) for f in self.payload["label_fractions"])

    @property
    def gates(self) -> dict[str, dict]:
        return dict(self.payload["gates"])

    @property
    def hypotheses(self) -> dict[str, dict]:
        return dict(self.payload["hypotheses"])

    @property
    def artifacts(self) -> dict[str, dict]:
        return dict(self.payload.get("artifacts", {}))

    @property
    def code_derived(self) -> dict[str, dict]:
        """Bindings whose hash is computed from code, not read from a file.

        The severity table is the only such object today: its sha256 comes from
        :func:`ssl_anomaly.monitor.severity_table.severity_table_hash` and is
        pinned in findings 25. Treating it as a file would silently skip it.
        """
        return dict(self.payload.get("code_derived", {}))

    def expected_hash(self, name: str) -> str | None:
        spec = self.artifacts.get(name) or self.code_derived.get(name)
        if not spec:
            return None
        value = spec.get("sha256_prefix")
        return str(value).lower() if value else None

    def to_manifest(self) -> dict:
        return {
            "preregistration_path": str(self.path.relative_to(REPO_ROOT))
            if self.path.is_relative_to(REPO_ROOT)
            else str(self.path),
            "preregistration_sha256": self.document_sha256,
            "version": self.payload.get("version"),
            "created": self.payload.get("created"),
            "seeds": list(self.seeds),
            "label_fractions": list(self.label_fractions),
            "gate_ids": sorted(self.gates),
        }


def load_preregistration(path: str | Path = PREREGISTRATION_PATH) -> Preregistration:
    """Load and validate the committed pre-registration.

    Validation is structural and deliberately strict: the gate definitions and
    the hypothesis table are what make the capstone pre-registered, so a
    malformed file must fail loudly rather than silently weaken a gate.
    """
    p = Path(path)
    if not p.is_file():
        raise ArtifactBindingError(f"pre-registration not found: {p}")
    payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ArtifactBindingError(f"pre-registration is not a mapping: {p}")

    required = ("version", "created", "seeds", "label_fractions", "gates", "hypotheses")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ArtifactBindingError(
            f"pre-registration missing required keys {missing} in {p}"
        )

    gate_ids = set(payload["gates"])
    expected_gates = {"G-A", "G-B", "G-C", "G-D", "G-E"}
    if gate_ids != expected_gates:
        raise ArtifactBindingError(
            f"pre-registration gates must be exactly {sorted(expected_gates)}, "
            f"got {sorted(gate_ids)}"
        )
    for gid, gate in payload["gates"].items():
        for key in ("criterion", "fail_consequence"):
            if not gate.get(key):
                raise ArtifactBindingError(f"gate {gid} lacks '{key}'")

    hyp_ids = set(payload["hypotheses"])
    expected_hyp = {"H1", "H2", "H3", "H4", "H5"}
    if hyp_ids != expected_hyp:
        raise ArtifactBindingError(
            f"pre-registration hypotheses must be exactly {sorted(expected_hyp)}, "
            f"got {sorted(hyp_ids)}"
        )
    for hid, hyp in payload["hypotheses"].items():
        for key in ("statement", "primary_metric", "gate"):
            if not hyp.get(key):
                raise ArtifactBindingError(f"hypothesis {hid} lacks '{key}'")
        if hyp["gate"] not in gate_ids:
            raise ArtifactBindingError(
                f"hypothesis {hid} references unknown gate {hyp['gate']!r}"
            )

    if [float(f) for f in payload["label_fractions"]] != [0.05, 0.10, 0.25, 0.50, 1.00]:
        raise ArtifactBindingError(
            "label_fractions must be the pre-registered 5/10/25/50/100 % set "
            f"(got {payload['label_fractions']})"
        )
    if [int(s) for s in payload["seeds"]] != [13, 42, 2026]:
        raise ArtifactBindingError(
            f"seeds must be the pre-registered {{13, 42, 2026}} (got {payload['seeds']})"
        )

    return Preregistration(payload=payload, path=p, document_sha256=sha256_file(p))


# ---------------------------------------------------------------------------
# Artifact binding
# ---------------------------------------------------------------------------


@dataclass
class ArtifactStatus:
    """One artifact's binding state. ``status`` ∈ {bound, mismatch, absent}."""

    name: str
    path: str
    expected_sha256_prefix: str | None
    actual_sha256: str | None
    status: str
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "bound"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "expected_sha256_prefix": self.expected_sha256_prefix,
            "actual_sha256": self.actual_sha256,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class BindingReport:
    """Aggregate binding result across the v2 chain."""

    preregistration: Preregistration
    artifacts: list[ArtifactStatus]
    note: str = ""
    strict: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def mismatched(self) -> list[ArtifactStatus]:
        return [a for a in self.artifacts if a.status == "mismatch"]

    @property
    def absent(self) -> list[ArtifactStatus]:
        return [a for a in self.artifacts if a.status == "absent"]

    @property
    def bound(self) -> list[ArtifactStatus]:
        return [a for a in self.artifacts if a.status == "bound"]

    @property
    def ok(self) -> bool:
        """True when every *present* artifact matches its frozen hash.

        Absent artifacts do not by themselves make the report fail — the spec
        requires them to be reported as ``absent`` so downstream cells are
        marked `Deferred` rather than silently substituted. Mismatches always
        fail.
        """
        return not self.mismatched

    def to_dict(self) -> dict:
        return {
            "preregistration": self.preregistration.to_manifest(),
            "strict": self.strict,
            "ok": self.ok,
            "n_bound": len(self.bound),
            "n_absent": len(self.absent),
            "n_mismatch": len(self.mismatched),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "note": self.note,
            **self.extra,
        }


def _resolve(base_dir: Path | None, relative: str) -> Path:
    p = Path(relative)
    if p.is_absolute():
        return p
    return (base_dir or REPO_ROOT) / p


def _manifest_mode_hash(name: str, path: Path) -> tuple[str | None, str]:
    """Recompute a manifest-mode artifact's self-hash from its file contents.

    Returns ``(hash_or_None, detail)``. ``None`` means the check could not run
    (malformed JSON, missing self-hash key, unavailable producer) — the caller
    must treat that as a mismatch, never as a pass.
    """
    producer = MANIFEST_HASH_PRODUCERS[name]
    hash_key = MANIFEST_HASH_KEYS[name]
    module_name, func_name = producer.split(":")
    try:
        module = __import__(module_name, fromlist=[func_name])
        hash_fn = getattr(module, func_name)
    except Exception as exc:  # noqa: BLE001 - report the dependency, never hide it
        return None, f"manifest-hash producer unavailable: {exc}"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return None, f"artifact is not a JSON mapping: {path}"
    except Exception as exc:  # noqa: BLE001
        return None, f"artifact unreadable as JSON: {exc}"
    recorded = str(payload.get(hash_key, "")).lower()
    if not recorded:
        return None, f"artifact carries no self-recorded {hash_key!r}"
    try:
        recomputed = str(hash_fn(payload)).lower()
    except Exception as exc:  # noqa: BLE001
        return None, f"manifest-hash recomputation failed: {exc}"
    if recomputed != recorded:
        return (
            None,
            f"self-inconsistent: recorded {hash_key}={recorded[:SHA_PREFIX_LEN]!r} "
            f"but contents re-hash to {recomputed[:SHA_PREFIX_LEN]!r}",
        )
    return (
        recomputed,
        f"{hash_key} recomputed from contents and matches the recorded field "
        f"(manifest mode, producer {producer})",
    )


def bind_artifacts(
    *,
    prereg: Preregistration | None = None,
    base_dir: str | Path | None = None,
    strict: bool = True,
    require: tuple[str, ...] = (),
    compute_hashes: bool = True,
    overrides: dict[str, str] | None = None,
) -> BindingReport:
    """Verify the v2 chain artifacts against the pre-registration.

    Parameters
    ----------
    prereg:
        Parsed pre-registration; loaded from the committed path when omitted.
    base_dir:
        Directory the relative artifact paths resolve against (defaults to the
        repository root, matching the frozen layout ``data/processed/...``).
    strict:
        When True a mismatch raises :class:`ArtifactBindingError` immediately.
        When False the report is returned with ``ok == False`` so a caller can
        render a diagnostic before failing.
    require:
        Artifact names that must be *present* for this call (``absent`` then
        raises). Use this for the artifacts the calling stage actually needs,
        e.g. ``require=("cycles_490_v2",)`` for the field-mean path.
    compute_hashes:
        Set False for a fast presence-only probe (``status`` becomes
        ``present-unverified``); the pre-registration hash is still computed.
    overrides:
        Explicit path overrides per artifact name (used by tests and by the
        owner's archive machine, where the canonical data root is outside the
        repository).

    Returns
    -------
    BindingReport
    """
    prereg = prereg or load_preregistration()
    base = Path(base_dir) if base_dir is not None else REPO_ROOT
    overrides = overrides or {}

    statuses: list[ArtifactStatus] = []
    for name, spec in prereg.artifacts.items():
        rel = overrides.get(name, spec["path"])
        path = _resolve(base, rel)
        expected = prereg.expected_hash(name)

        if not path.exists():
            status = ArtifactStatus(
                name=name,
                path=str(path),
                expected_sha256_prefix=expected,
                actual_sha256=None,
                status="absent",
                detail="file not found",
            )
        elif not compute_hashes:
            status = ArtifactStatus(
                name=name,
                path=str(path),
                expected_sha256_prefix=expected,
                actual_sha256=None,
                status="present-unverified",
                detail="hash not computed (compute_hashes=False)",
            )
        else:
            mode = "manifest" if name in MANIFEST_HASH_PRODUCERS else "bytes"
            if mode == "manifest":
                actual, detail = _manifest_mode_hash(name, path)
            else:
                actual, detail = sha256_path(path), ""
            if actual is None:
                # The artifact exists but its self-hash could not be recomputed
                # (malformed JSON / missing self-hash key): fail closed.
                status = ArtifactStatus(
                    name=name, path=str(path),
                    expected_sha256_prefix=expected, actual_sha256=None,
                    status="mismatch", detail=detail or "manifest self-hash unavailable",
                )
            elif expected is None:
                status = ArtifactStatus(
                    name=name,
                    path=str(path),
                    expected_sha256_prefix=expected,
                    actual_sha256=actual,
                    status="bound",
                    detail="no frozen hash recorded (hash observed only)",
                )
            elif actual.lower().startswith(expected):
                status = ArtifactStatus(
                    name=name,
                    path=str(path),
                    expected_sha256_prefix=expected,
                    actual_sha256=actual,
                    status="bound",
                    detail=detail,
                )
            else:
                status = ArtifactStatus(
                    name=name,
                    path=str(path),
                    expected_sha256_prefix=expected,
                    actual_sha256=actual,
                    status="mismatch",
                    detail=(
                        f"expected sha256 prefix {expected!r}, got "
                        f"{actual[:SHA_PREFIX_LEN]!r}"
                    ),
                )
        statuses.append(status)

    for name, spec in prereg.code_derived.items():
        statuses.append(
            code_derived_status(name, str(spec.get("sha256_prefix") or "").lower() or None)
        )

    note = ""
    file_statuses = [s for s in statuses if not s.path.startswith("<code:")]
    if file_statuses and not any(s.status != "absent" for s in file_statuses):
        note = (
            "No v2-chain artifact is present at the resolved base directory. "
            "This is the expected state of a clean checkout: the archive lives "
            "outside the repository (D:\\RS\\Thesis\\Data or the Drive bundle). "
            "Every dependent cell must be recorded `Deferred`/`absent` — never "
            "approximated."
        )

    report = BindingReport(
        preregistration=prereg,
        artifacts=statuses,
        note=note,
        strict=strict,
        extra={"base_dir": str(base)},
    )

    if strict and report.mismatched:
        first = report.mismatched[0]
        raise ArtifactBindingError(
            f"artifact {first.name!r} deviates from the pre-registration: {first.detail}"
        )
    if require:
        absent_required = [n for n in require if n not in {s.name for s in report.bound}]
        if absent_required:
            details = {s.name: s.detail for s in report.artifacts if s.name in absent_required}
            raise ArtifactBindingError(
                f"required artifact(s) not bound: {absent_required} ({details})"
            )
    return report


# ---------------------------------------------------------------------------
# Hash-prefix table used by docs/findings cross-checks
# ---------------------------------------------------------------------------

#: The v2 chain hashes as they appear in findings 25 / the change docs. Kept in
#: code so a test can prove docs and pre-registration name the same artifacts.
DOCUMENTED_V2_HASHES: dict[str, str] = {
    "splits_locked_v2": "30ecbe10",
    "holdout_manifest_v2": "d501bf81",
    "norm_stats_full_v2": "493f0395",
    "severity_table": "9d720e3b",
}


def documented_hash(name: str) -> str | None:
    return DOCUMENTED_V2_HASHES.get(name)


_SHA_RE = re.compile(r"^[0-9a-f]{8,64}$")


def is_sha_prefix(value: str) -> bool:
    return bool(_SHA_RE.match(str(value).lower()))


#: Code-derived hash producers. A name in this map is bound by *calling* the
#: producer, so a code change that shifts the frozen object fails closed.
CODE_DERIVED_PRODUCERS: dict[str, str] = {
    "severity_table": "ssl_anomaly.monitor.severity_table:severity_table_hash",
}

#: Manifest-mode artifacts: the frozen prefix names the file's OWN
#: ``manifest_sha256``/``sha256`` field — a canonical hash over the payload
#: with that key removed — which this module RECOMPUTES from the file contents
#: and compares (never trusts the recorded field alone). A raw byte hash can
#: never match here: a file cannot contain its own byte hash, so byte-mode
#: binding for these artifacts would fail closed on every machine where they
#: exist. The producers are the exact functions that wrote the fields.
MANIFEST_HASH_PRODUCERS: dict[str, str] = {
    "splits_locked_v2": "ssl_anomaly.monitor.splits:manifest_hash",
    "holdout_manifest_v2": "ssl_anomaly.monitor.holdout:holdout_manifest_hash",
    "norm_stats_full_v2": "ssl_anomaly.evaluation.binding:norm_stats_document_hash",
}

#: The self-recorded hash key per manifest-mode artifact.
MANIFEST_HASH_KEYS: dict[str, str] = {
    "splits_locked_v2": "manifest_sha256",
    "holdout_manifest_v2": "manifest_sha256",
    "norm_stats_full_v2": "sha256",
}


def norm_stats_document_hash(payload: dict) -> str:
    """Canonical hash of a norm-stats sidecar (``sha256`` key removed).

    Mirrors the writer recipe: ``sha256(json.dumps(payload - sha256,
    sort_keys=True))`` — verified byte-for-byte against the committed
    ``norm_stats_full_v2.json`` (its self-recorded ``493f0395…``).
    """
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()


def code_derived_status(name: str, expected: str | None) -> ArtifactStatus:
    """Bind a code-derived object (currently just the severity table)."""
    producer = CODE_DERIVED_PRODUCERS.get(name)
    if producer is None:
        return ArtifactStatus(
            name=name, path=f"<code:{name}>", expected_sha256_prefix=expected,
            actual_sha256=None, status="absent",
            detail=f"no code-derived producer registered for {name!r}",
        )
    module_name, func_name = producer.split(":")
    try:
        module = __import__(module_name, fromlist=[func_name])
        actual = str(getattr(module, func_name)())
    except Exception as exc:  # noqa: BLE001 - report the dependency, never hide it
        return ArtifactStatus(
            name=name, path=f"<code:{producer}>", expected_sha256_prefix=expected,
            actual_sha256=None, status="absent",
            detail=f"producer unavailable: {exc}",
        )
    if expected is None:
        status, detail = "bound", "no frozen hash recorded (hash observed only)"
    elif actual.lower().startswith(expected):
        status, detail = "bound", ""
    else:
        status, detail = "mismatch", (
            f"expected sha256 prefix {expected!r}, got {actual[:SHA_PREFIX_LEN]!r} "
            "(the frozen severity table changed in code)"
        )
    return ArtifactStatus(
        name=name, path=f"<code:{producer}>", expected_sha256_prefix=expected,
        actual_sha256=actual, status=status, detail=detail,
    )

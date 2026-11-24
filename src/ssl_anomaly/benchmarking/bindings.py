"""Artifact bindings: what a run may read, and how it proves what it read (task 1.3).

Two failure modes this module exists to kill, both from AGENTS.md §7:

1. **A run bound to the wrong data.** The v2 chain is identified by sha256
   prefixes recorded in the contract; :func:`verify_bindings` recomputes them and
   raises on any deviation. An artifact whose contract entry declares
   ``hash_policy: pin_at_runtime`` (``cycles_490_v2.parquet`` — no owner-recorded
   byte hash exists for it) is hashed on first read and the observed hash is
   written to ``bindings.json``; every later run, including every resume, must
   match that pinned value or it aborts. That is the difference between "bound"
   and "whatever was on disk today".
2. **A resume that quietly changes dataset mid-experiment.** :func:`compare_bindings`
   is what the checkpoint loader consults before restoring weights; a mismatch is
   a hard error with both hashes printed, never a warning.

The locked ``test`` split is never enumerated here; only split membership counts
are read, and only to verify the binding.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SHA_PREFIX_LEN = 8


class BindingError(RuntimeError):
    """A required artifact is missing, or its hash deviates from the contract."""


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Streaming sha256 — the stores are multi-GB and must not be loaded whole."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def git_sha(base: str | Path | None = None) -> dict:
    """Commit + dirty flag, so a number can be traced to code (never assumed)."""
    cwd = str(base or REPO_ROOT)
    out: dict[str, Any] = {"sha": None, "dirty": None, "error": None}
    try:
        sha = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20
        )
        st = subprocess.run(
            ["git", "-C", cwd, "status", "--porcelain"], capture_output=True, text=True, timeout=40
        )
        if sha.returncode == 0:
            out["sha"] = sha.stdout.strip()
            out["dirty"] = bool(st.stdout.strip())
        else:  # pragma: no cover - environment dependent
            out["error"] = (sha.stderr or "git rev-parse failed").strip()[:200]
    except Exception as exc:  # pragma: no cover - environment dependent
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def software_versions() -> dict:
    """Versions whose numerics could move a reported number."""
    import platform
    import sys

    block: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for mod in ("numpy", "pandas", "scipy", "torch", "sklearn", "statsmodels", "zarr"):
        try:
            m = __import__(mod)
            block[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - environment dependent
            block[mod] = f"unavailable ({type(exc).__name__})"
    return block


# ---------------------------------------------------------------------------
# Binding records
# ---------------------------------------------------------------------------


@dataclass
class ArtifactState:
    """Outcome of checking one contract artifact."""

    name: str
    path: str
    status: str                    # bound | pinned | missing | mismatch | unrequired
    expected_prefix: str | None = None
    observed_sha256: str | None = None
    record: str = ""
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in ("bound", "pinned", "unrequired", "presence")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "status": self.status,
            "expected_prefix": self.expected_prefix,
            "observed_sha256": self.observed_sha256,
            "record": self.record,
            "detail": self.detail,
        }


@dataclass
class BindingReport:
    """Everything a manifest needs about what the run read."""

    states: list[ArtifactState] = field(default_factory=list)
    benchmark_config_sha256: str = ""
    matrix_sha256: str = ""
    applicability_sha256: str = ""
    git: dict = field(default_factory=dict)
    software: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.states)

    @property
    def blocking(self) -> list[ArtifactState]:
        return [s for s in self.states if not s.ok]

    def pinned(self) -> dict[str, str]:
        return {s.name: s.observed_sha256 for s in self.states if s.observed_sha256}

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "benchmark_config_sha256": self.benchmark_config_sha256,
            "matrix_sha256": self.matrix_sha256,
            "applicability_sha256": self.applicability_sha256,
            "artifacts": [s.to_dict() for s in self.states],
            "git": self.git,
            "software": self.software,
            "hardware": self.hardware,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> BindingReport:
        return cls(
            states=[ArtifactState(**s) for s in d.get("artifacts", [])],
            benchmark_config_sha256=d.get("benchmark_config_sha256", ""),
            matrix_sha256=d.get("matrix_sha256", ""),
            applicability_sha256=d.get("applicability_sha256", ""),
            git=d.get("git", {}),
            software=d.get("software", {}),
            hardware=d.get("hardware", {}),
            notes=list(d.get("notes", [])),
        )


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_bindings(
    cfg,
    *,
    base: str | Path | None = None,
    matrix_sha256: str = "",
    applicability_sha256: str = "",
    pinned: dict[str, str] | None = None,
    compute_hashes: bool = True,
    strict: bool = True,
) -> BindingReport:
    """Check every contract artifact, honouring each one's declared hash policy.

    ``pinned`` is the previously recorded ``{name: sha256}`` map (from a resume);
    a runtime-pinned artifact that disagrees with it is a ``mismatch`` regardless
    of ``strict`` — that is the case where a directory was replaced between cells.

    ``strict=False`` reports problems without raising: used by ``--smoke`` and by
    the notebook's "what is missing here?" cell, where the answer must be printed
    rather than thrown.
    """
    root = Path(base) if base is not None else REPO_ROOT
    pinned = pinned or {}
    report = BindingReport(
        benchmark_config_sha256=cfg.document_sha256,
        matrix_sha256=matrix_sha256,
        applicability_sha256=applicability_sha256,
        git=git_sha(root),
        software=software_versions(),
    )
    problems: list[str] = []

    for name in cfg.raw.get("artifacts", {}):
        art = cfg.artifact(name)
        path = Path(art.path)
        if not path.is_absolute():
            path = root / path
        state = ArtifactState(
            name=name,
            path=str(path.relative_to(root)) if _is_under(path, root) else str(path),
            status="missing",
            expected_prefix=art.sha256_prefix,
            record=art.record,
        )
        if not path.exists():
            if art.required_for in ("patch_level", "optional", "weather_inputs"):
                state.status = "unrequired"
                state.detail = f"absent; required_for={art.required_for} → dependent cells are Deferred/absent"
                report.notes.append(f"{name}: absent ({art.required_for})")
            else:
                problems.append(f"{name}: not found at {state.path}")
                state.detail = "required by the contract but not on disk"
            report.states.append(state)
            continue

        if not compute_hashes:
            # A presence-only probe must not pretend to have verified a hash: the
            # status says "presence", the observed hash stays None, and a smoke
            # manifest therefore cannot be mistaken for a bound run.
            state.status = "presence"
            state.detail = "present; hash NOT computed (compute_hashes=False, smoke mode)"
            report.states.append(state)
            continue

        digest = sha256_file(path)
        state.observed_sha256 = digest
        if art.bound:
            if digest.startswith(art.sha256_prefix or ""):
                state.status = "bound"
            else:
                state.status = "mismatch"
                problems.append(
                    f"{name}: sha256 {digest[:12]}… ≠ the contract prefix {art.sha256_prefix}… "
                    f"at {state.path}"
                )
        elif name in pinned:
            if pinned[name] == digest:
                state.status = "pinned"
            else:
                state.status = "mismatch"
                problems.append(
                    f"{name}: sha256 {digest[:12]}… ≠ the hash pinned by the first run "
                    f"({pinned[name][:12]}…) — the data changed under the experiment"
                )
        else:
            state.status = "pinned"
            state.detail = "no preregistered prefix; observed hash is now pinned for this run"
        report.states.append(state)

    if problems and strict:
        raise BindingError(
            "artifact binding failed ({} problem{}):\n  - {}".format(
                len(problems), "" if len(problems) == 1 else "s", "\n  - ".join(problems)
            )
        )
    report.notes.extend(problems)
    return report


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def save_bindings(report: BindingReport, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report.to_dict(), indent=1, sort_keys=True), encoding="utf-8")
    return p


def load_bindings(path: str | Path) -> BindingReport:
    p = Path(path)
    if not p.is_file():
        raise BindingError(f"no bindings file at {p}; a resume cannot verify its data without it")
    return BindingReport.from_dict(json.loads(p.read_text(encoding="utf-8")))


#: The identity fields a resume must re-verify before restoring any weights.
RESUME_IDENTITY_FIELDS = (
    "benchmark_config_sha256",
    "matrix_sha256",
    "applicability_sha256",
)


def compare_bindings(saved: BindingReport, current: BindingReport) -> dict:
    """What a resume checks. Any mismatch is an abort, not a warning (spec scenario).

    Returns ``{ok, problems}``; the runner raises when ``ok`` is false.
    """
    problems: list[str] = []
    for f in RESUME_IDENTITY_FIELDS:
        a, b = getattr(saved, f, ""), getattr(current, f, "")
        if a and b and a != b:
            problems.append(f"{f} changed between runs: {a[:12]}… → {b[:12]}…")
    saved_art = {s.name: s for s in saved.states}
    for s in current.states:
        old = saved_art.get(s.name)
        if old is None:
            continue
        if old.observed_sha256 and s.observed_sha256 and old.observed_sha256 != s.observed_sha256:
            problems.append(
                f"artifact {s.name} hash changed: {old.observed_sha256[:12]}… → {s.observed_sha256[:12]}…"
            )
        if (old.status == "unrequired") != (s.status == "unrequired"):
            problems.append(
                f"artifact {s.name} availability changed between runs "
                f"({old.status} → {s.status}); cells that deferred for it must not silently run now"
            )
    return {"ok": not problems, "problems": problems}

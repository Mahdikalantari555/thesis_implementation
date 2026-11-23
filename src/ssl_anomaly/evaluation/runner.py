"""Run orchestration: one place that assembles evidence, gates, and artifacts.

This module exists so that the scripts under ``scripts/evaluation`` contain no
science. Each script parses arguments, calls into here, and writes what it gets
back; every rule about what may be claimed lives in the modules this imports.

The run manifest is the audit trail. It carries:

- the pre-registration's document hash (so a report can be traced to the frozen
  contract it was produced under),
- the git commit and dirty flag (so a number can be traced to code),
- package versions for the numerically load-bearing libraries,
- the seeds actually used,
- the honest-state block: what was ``absent`` and why.

It never carries a claimed result. Results live in the report artifacts, where
each one has already passed through :class:`TaskResult` / :class:`ClaimRecord`
and therefore cannot exist without a ``measured`` status.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ssl_anomaly.evaluation.binding import BindingReport, bind_artifacts, load_preregistration
from ssl_anomaly.evaluation.gates import GateReport, evaluate_gates
from ssl_anomaly.evaluation.provenance import Provenance, write_json

REPORTS_DIR = Path("reports/regional_representation")

#: Smoke output lives OUTSIDE the artifact directory on purpose. Nesting it as
#: ``regional_representation/_smoke/`` would put synthetic numbers in the tree a
#: reader browses for evidence; a separate root makes "this is not evidence"
#: structural rather than a matter of remembering the subdirectory name.
SMOKE_REPORTS_DIR = Path("reports/_smoke/regional_representation")


def is_smoke(tag: str | None) -> bool:
    return bool(tag) and tag.startswith("smoke")


@dataclass
class RunPaths:
    """Where a run writes. Smoke runs go to a sibling directory, never over real output."""

    out_dir: Path
    smoke: bool = False

    @classmethod
    def for_run(cls, tag: str | None = None) -> RunPaths:
        if is_smoke(tag):
            return cls(SMOKE_REPORTS_DIR / tag, smoke=True)
        return cls(REPORTS_DIR, smoke=False)

    def ensure(self) -> Path:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        return self.out_dir

    def artifact(self, name: str) -> Path:
        return self.out_dir / name


@dataclass
class RunManifest:
    """The audit trail. Constructed by a script, never by a report."""

    stage: str
    smoke: bool
    prereg_sha256: str
    provenance: dict
    seeds: list[int]
    config: dict
    binding: dict | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "smoke": self.smoke,
            "preregistration_sha256": self.prereg_sha256,
            "provenance": self.provenance,
            "seeds": self.seeds,
            "config": self.config,
            "binding": self.binding,
            **self.extras,
            "note": (
                "smoke runs use the synthetic fixture and produce NO quotable "
                "evidence; they exist to exercise the pipeline end to end"
                if self.smoke else
                "real run: read the gates block before quoting any number"
            ),
        }


def build_manifest(
    *,
    stage: str,
    smoke: bool,
    config: dict,
    seeds: list[int],
    base_dir: str | Path | None = None,
    binding: BindingReport | None = None,
    extras: dict | None = None,
) -> RunManifest:
    """Assemble the manifest, binding artifacts in the caller's requested mode.

    ``smoke=True`` uses ``compute_hashes=False`` for a fast presence probe — a
    smoke run must not spend its budget hashing a multi-GB parquet it never
    reads.
    """
    prereg = load_preregistration()
    if binding is None:
        binding = bind_artifacts(
            prereg=prereg, base_dir=base_dir, strict=False, compute_hashes=not smoke
        )
    prov = Provenance.for_run(
        preregistration_sha256=prereg.document_sha256,
        seeds=tuple(seeds),
        data_source="synthetic-fixture" if smoke else "data/processed/cycles_490_v2.parquet",
        extra={"stage": stage, "smoke": smoke, "config": config},
    )
    return RunManifest(
        stage=stage,
        smoke=smoke,
        prereg_sha256=prereg.document_sha256,
        provenance=prov.to_dict(),
        seeds=list(seeds),
        config=config,
        binding=binding.to_dict(),
        extras=extras or {},
    )


def write_run_outputs(
    *,
    paths: RunPaths,
    manifest: RunManifest,
    reports: dict[str, dict],
    metrics: dict | None = None,
) -> dict[str, str]:
    """Write manifest + one JSON per report. Returns the paths written.

    Every report is written through :func:`write_json`, which refuses NaN — so a
    silently-degraded metric can never reach disk looking like a number.
    """
    out = paths.ensure()
    written = {"manifest": str(out / "run_manifest.json")}
    write_json(out / "run_manifest.json", manifest.to_dict())
    for name, payload in reports.items():
        path = out / f"{name}.json"
        write_json(path, payload)
        written[name] = str(path)
    if metrics is not None:
        write_json(out / "metrics.json", metrics)
        written["metrics"] = str(out / "metrics.json")
    return written


# ---------------------------------------------------------------------------
# Smoke fixtures
# ---------------------------------------------------------------------------


@dataclass
class SmokeFixture:
    """Deterministic synthetic data that exercises every code path.

    Deliberately NOT random-looking: farm structure, phase structure and a real
    anomaly signal are all present, so a probe that returns chance means the
    pipeline is broken, not that the data is uninformative. This is what makes a
    smoke run diagnostic rather than merely non-crashing.
    """

    dev: object
    labels: np.ndarray
    farm_phase: np.ndarray
    farm_crop: np.ndarray
    years: np.ndarray
    block_ids: np.ndarray
    fields: np.ndarray
    #: Per-cycle phase bin (36-way), the target the Level-1 phase probe needs.
    #: Supplied explicitly because it is NOT derivable from the cycle tensor.
    farm_phase_bins: np.ndarray | None = None

    def describe(self) -> dict:
        return {
            "n_cycles": int(self.dev.n_cycles),
            "n_fields": int(np.unique(self.fields).size),
            "prevalence": float(self.labels.mean()),
            "source": "synthetic-fixture",
            "warning": "synthetic values: no empirical meaning",
        }


def make_smoke_fixture(*, n_fields: int = 12, cycles_per_field: int = 4, seed: int = 13) -> SmokeFixture:
    """A fixture where the label requires FIELD CONTEXT, not a global threshold.

    Label = "this cycle's NDVI sits far below its own field's baseline", which is
    what a real anomaly detector is asked to find.

    **This was wrong once, in a way that only showed up in the numbers.** The
    first version subtracted a constant from the anomalous cycles' NDVI, i.e. a
    *global* depression. That made the label trivially recoverable from the
    per-cycle mean, and A6 (a probe on raw mean/std features) scored AP 1.000 —
    a "transfer curve" that was really a demonstration that averaging works. The
    anomaly is now a per-field offset, so no global statistic separates the
    classes and only a representation that encodes the field's own history can.

    Fields are also given distinct baselines, so a single pooled mean cannot
    substitute for context.
    """
    from ssl_anomaly.evaluation.data import CycleData, synthetic_evaluation_split

    n = n_fields * cycles_per_field
    base = synthetic_evaluation_split(
        n_train=0, n_dev=n, n_fields=n_fields, seed=seed,
    )
    dev = base.for_split("dev")
    rng = np.random.default_rng(seed + 1)
    fields = dev.field_ids

    # Field-level agronomy for the retrieval track that needs it.
    #
    # Crops are assigned in BLOCKS of four fields and phases in pairs: with a
    # uniformly random crop/phase assignment most fields have no agronomic partner
    # at all (same crop AND phase within 2 bins), and track B is then legitimately
    # "undefined" for the whole dataset. That is correct behaviour in the
    # implementation but useless as a fixture, so the fixture is built to contain
    # real partners.
    uniq = np.unique(fields)
    # Every field must get exactly one crop: a truncated repeat silently shortens
    # the array and breaks the one-entry-per-field contract the retrieval track
    # validates (it raised only once, for n_fields=6).
    farm_crop = np.arange(uniq.size) // 4
    farm_phase = np.empty(uniq.size)
    for k in range(0, uniq.size, 2):
        base = (k * 3) % 34
        farm_phase[k] = base
        if k + 1 < uniq.size:
            farm_phase[k + 1] = base + 1.0

    # Each field gets its own NDVI baseline, then a quarter of the cycles are
    # depressed RELATIVE TO THEIR OWN FIELD. A global threshold is therefore
    # useless: a low-NDVI normal cycle and a depressed high-NDVI cycle can sit at
    # the same absolute value.
    X = dev.X.copy()
    # The baseline spread MUST straddle the anomaly magnitude (2.5). With a
    # narrow +/-1 spread a global threshold still separates the classes, which is
    # how the first version of this fixture produced an A6 probe at AP 1.000.
    field_baseline = {fid: float(rng.uniform(-4.0, 4.0)) for fid in uniq}
    for i in range(n):
        X[i, :, 0] = X[i, :, 0] + field_baseline[fields[i]]

    labels = np.zeros(n, dtype=int)
    anomalous = rng.choice(n, size=max(1, n // 4), replace=False)
    labels[anomalous] = 1
    X[anomalous, :, 0] = X[anomalous, :, 0] - 2.5

    dev2 = CycleData(
        X=X, M=dev.M, field_ids=dev.field_ids, cycle_ids=dev.cycle_ids,
        split=dev.split, dsh=dev.dsh, channel_names=dev.channel_names,
    )
    years = rng.integers(2014, 2026, size=n)
    blocks = rng.integers(1, 21, size=n)
    phase_bins = rng.integers(0, 36, size=n)
    return SmokeFixture(
        dev=dev2, labels=labels, farm_phase=farm_phase, farm_crop=farm_crop,
        years=years, block_ids=blocks, fields=fields, farm_phase_bins=phase_bins,
    )


@dataclass
class SmokeExpectation:
    """What the smoke run asserts about its own output before declaring success."""

    name: str
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        self.checks.append((label, bool(condition), detail))

    @property
    def ok(self) -> bool:
        return all(c[1] for c in self.checks)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "checks": [
                {"label": l, "passed": p, "detail": d} for l, p, d in self.checks
            ],
        }

    def raise_if_failed(self) -> None:
        bad = [c for c in self.checks if not c[1]]
        if bad:
            raise AssertionError(
                f"smoke {self.name} failed: "
                + "; ".join(f"{l} ({d})" if d else l for l, _, d in bad)
            )


# ---------------------------------------------------------------------------
# Gates convenience
# ---------------------------------------------------------------------------


def gates_for_run(*, prereg, evidence: dict, binding: BindingReport) -> GateReport:
    """Evaluate gates with the live binding report wired in as G-E's evidence."""
    return evaluate_gates(prereg=prereg, evidence=evidence, binding=binding)


def environment_block() -> dict:
    """Versions of the libraries whose numerics could move a reported number."""
    block = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    for mod in ("numpy", "pandas", "scipy", "torch", "sklearn", "zarr"):
        try:
            m = __import__(mod)
            block[mod] = getattr(m, "__version__", "unknown")
        except Exception as exc:  # pragma: no cover - environment dependent
            block[mod] = f"unavailable ({type(exc).__name__})"
    return block


def dumps(obj) -> str:
    """Stable pretty JSON with NaN refused (a NaN is a missing decision, not a value)."""
    return json.dumps(obj, indent=2, sort_keys=True, allow_nan=False, default=_coerce)


def _coerce(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")

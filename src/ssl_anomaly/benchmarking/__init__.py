"""Benchmark harness for ``benchmark-latent-reconstruction-prediction-architectures``.

This package owns the *heavy* half of the 2026-11-23 owner decision: a
pre-registered, budget-aware comparison of reconstruction / latent-reconstruction
/ prediction / JEPA / hybrid self-supervised architectures against classical
baselines, on the frozen v2 chain, executed on Colab T4 after the AGENTS.md
§6.1 local gate.

What lives here (and what deliberately does not):

``config``        the binding benchmark contract: factor ids, seeds, budget,
                  collapse thresholds, data bindings. :func:`load_benchmark_config`
                  refuses a weakened or ambiguous version of it.
``matrix``        generation of the complete experiment matrix from the config plus
                  the suppression applicability table, with a committed hash.
``registry``      the experiment registry: exactly seven states
                  (``Pending``/``Running``/``Completed``/``Failed``/``Interrupted``/
                  ``Deferred``/``Not Applicable``), resume-safe transitions,
                  append-only transition log.
``bindings``      the hash bindings a run must match (cycles / splits / holdout /
                  norm / config / matrix) and the runtime pinning policy.
``checkpoints``   per-seed checkpoint layout (model + optimizer + scheduler + RNG)
                  and the binding-hash re-verification that gates a resume.
``environment``   device detection, the T4 requirement, VRAM probing and the
                  mixed-precision decision — all recorded, never assumed.
``runner``        executes one registry cell end to end and writes its manifest.
``report``        sections A–K, coverage statements, the three winner reports.

Import-light by design: :mod:`config`, :mod:`matrix`, :mod:`registry` and
:mod:`bindings` import no torch, so the freeze, the matrix hash and the registry
audit run in a minimal environment and from a clean checkout (same rule as
:mod:`ssl_anomaly.evaluation.binding`).

Frozen things this package must never touch: ``splits_locked_v2.json``,
``holdout_manifest_v2.json``, the B1 sidecar, the ``gru_ae_main`` checkpoints,
findings 00–25, the platform, and the locked ``test`` split — which
:mod:`ssl_anomaly.benchmarking.splits` guards by reusing the capstone's
:class:`TestSplitAccessError`.
"""

from __future__ import annotations

from ssl_anomaly.benchmarking.config import (
    BenchmarkConfig,
    ConfigError,
    load_applicability,
    load_benchmark_config,
    validate_config,
)
from ssl_anomaly.benchmarking.matrix import (
    ExperimentCell,
    build_matrix,
    coverage_line,
    matrix_document,
    matrix_sha256,
    read_matrix,
    verify_committed_matrix,
    write_matrix,
)
from ssl_anomaly.benchmarking.registry import (
    FINAL_STATES,
    REGISTRY_STATES,
    Registry,
    RegistryEntry,
    RegistryError,
)

__all__ = [
    "FINAL_STATES",
    "REGISTRY_STATES",
    "BenchmarkConfig",
    "ConfigError",
    "ExperimentCell",
    "Registry",
    "RegistryEntry",
    "RegistryError",
    "build_matrix",
    "coverage_line",
    "load_applicability",
    "load_benchmark_config",
    "matrix_document",
    "matrix_sha256",
    "read_matrix",
    "validate_config",
    "verify_committed_matrix",
    "write_matrix",
]

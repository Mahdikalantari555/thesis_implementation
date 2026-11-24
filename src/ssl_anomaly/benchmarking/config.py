"""The binding benchmark configuration (change task 1.2).

One YAML file — ``configs/benchmarks/benchmark_config.yaml`` — is the contract
for every cell of the benchmark: which factors exist, which seeds are legal,
what the capacity band is, which collapse thresholds disqualify a candidate,
how large the synthetic panel is, which data artifacts the run is bound to, and
how much compute the cycle may spend. The matrix generator
(:mod:`ssl_anomaly.benchmarking.matrix`) and the registry
(:mod:`ssl_anomaly.benchmarking.registry`) both read this file, so a cell can
never be scheduled with a factor combination the contract does not declare.

Why validation is a hard gate rather than a docstring: research-integrity rule 7
of the design ("no hidden configuration changes; the committed matrix is the
contract") is only enforceable if a config that *cannot* produce the committed
matrix is rejected at load time. :func:`validate_config` therefore checks the
properties the specs name as binding, and reports every problem at once —
partial acceptance is how a mid-run drift goes unnoticed.

No torch, no file writes, no network.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "benchmarks" / "benchmark_config.yaml"
APPLICABILITY_PATH = REPO_ROOT / "configs" / "benchmarks" / "suppression_applicability.yaml"
MATRIX_PATH = REPO_ROOT / "configs" / "benchmarks" / "matrix.yaml"

#: The pre-registered seed set (spec experimental-validation / "Three random
#: seeds"). Screening may only ever run a *subset* of it, never a different set.
PREREGISTERED_SEEDS = (13, 42, 2026)

#: The seven states the registry may hold, exactly (spec
#: reproducible-benchmark-execution / "Experiment state management").
STATES = ("Pending", "Running", "Completed", "Failed", "Interrupted", "Deferred", "Not Applicable")

#: Paradigm ids (design §Architecture matrix). ``hybrid`` is a weighted
#: combination of the two losses, never a re-labelled reconstruction.
PARADIGMS = ("input_reconstruction", "latent_reconstruction", "future_prediction", "jepa", "hybrid")
BACKBONES = ("gru", "mamba")

#: The locked test split may not appear anywhere in ``allowed_splits``.
FORBIDDEN_SPLITS = ("test", "monitoring")


class ConfigError(ValueError):
    """The benchmark contract is incomplete, ambiguous or weakened."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__(
            "benchmark config rejected ({} problem{}):\n  - {}".format(
                len(self.problems),
                "" if len(self.problems) == 1 else "s",
                "\n  - ".join(self.problems),
            )
        )


# ---------------------------------------------------------------------------
# Factor metadata
# ---------------------------------------------------------------------------

#: Design §Architecture matrix A1–A10. ``patch_level`` marks an arm whose
#: hierarchical scoring depends on the v2 patch store binding.
ARCHITECTURES: dict[str, dict[str, str]] = {
    "A1": {"paradigm": "input_reconstruction", "backbone": "gru", "label": "GRU-AE"},
    "A2": {"paradigm": "input_reconstruction", "backbone": "mamba", "label": "Mamba-AE"},
    "A3": {"paradigm": "latent_reconstruction", "backbone": "gru", "label": "GRU-LatentAE"},
    "A4": {"paradigm": "latent_reconstruction", "backbone": "mamba", "label": "Mamba-LatentAE"},
    "A5": {"paradigm": "future_prediction", "backbone": "gru", "label": "GRU predictor"},
    "A6": {"paradigm": "future_prediction", "backbone": "mamba", "label": "Mamba predictor"},
    "A7": {"paradigm": "jepa", "backbone": "gru", "label": "GRU-JEPA"},
    "A8": {"paradigm": "jepa", "backbone": "mamba", "label": "Mamba-JEPA"},
    "A9": {"paradigm": "hybrid", "backbone": "gru", "label": "Hybrid GRU"},
    "A10": {"paradigm": "hybrid", "backbone": "mamba", "label": "Hybrid Mamba"},
}

#: Reference (non-SSL) comparators. ``B1`` is reused, never redefined here.
BASELINES: dict[str, dict[str, Any]] = {
    "B1": {"kind": "frozen_rule", "label": "b1_phasez_causal (phase-aware robust z)", "trained": False},
    "BL-IF": {"kind": "classical", "label": "Isolation Forest", "trained": True},
    "BL-OCSVM": {"kind": "classical", "label": "One-Class SVM", "trained": True},
    "BL-RPCA": {"kind": "classical", "label": "Robust PCA", "trained": True},
    "BL-RULE": {"kind": "rule", "label": "ndvi_lst_rule (FAR-matched control)", "trained": False},
    "A-RAND": {"kind": "capacity_control", "label": "random-init encoder, same trunk", "trained": False},
}

SUPPRESSIONS: dict[str, dict[str, str]] = {
    "S0": {"label": "Baseline objective", "mechanism": "none"},
    "S1": {"label": "Soft suppression", "mechanism": "bounded per-observation robustness weights"},
    "S2": {"label": "Masked learning", "mechanism": "controlled context masking (distinct from missing data)"},
    "S3": {"label": "Synthetic-anomaly suppression", "mechanism": "contaminated training + detached suppression weights"},
}

INPUTS: dict[str, dict[str, Any]] = {
    "I1": {"label": "spectral only", "channels": ("spectral_thermal",), "weather": False, "phenology": False},
    "I2": {"label": "spectral + weather", "channels": ("spectral_thermal", "era5_daily"), "weather": True, "phenology": False},
    "I3": {"label": "spectral + weather + phenology", "channels": ("spectral_thermal", "era5_daily", "phase"), "weather": True, "phenology": True},
}

FUSIONS: dict[str, dict[str, str]] = {
    "W0": {"label": "no weather conditioning (weather-free input)", "requires": "none"},
    "W1": {"label": "conditioning (FiLM-style)", "requires": "weather"},
    "W2": {"label": "separate weather encoder + latent fusion", "requires": "weather"},
}

CALIBRATIONS: dict[str, dict[str, str]] = {
    "C1": {"label": "farm-specific", "pooling": "farm"},
    "C2": {"label": "farm × stage", "pooling": "farm_stage"},
    "C3": {"label": "global", "pooling": "global"},
}

PROTOCOLS: dict[str, dict[str, str]] = {
    "V1": {"label": "temporal holdout", "scope": "locked v2 split discipline; dev for selection"},
    "V2": {"label": "farm holdout", "scope": "train+dev, block-consistent disjoint farm groups"},
    "V3": {"label": "spatiotemporal holdout", "scope": "train+dev, unseen farms × unseen periods, holdout exclusions"},
}

#: Anomaly families × severities (design §Synthetic anomaly generation).
FAMILIES = ("spectral", "temporal", "phenological", "multivariate", "persistent")
SEVERITIES = ("weak", "moderate", "strong")

#: Pre-registered priority order for budget-aware scheduling (design §Budget-aware
#: execution): "incumbent first, then paradigms adjacent to it".
ARCH_PRIORITY = ("A1", "A9", "A5", "A3", "A7", "A2", "A10", "A6", "A4", "A8")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ArtifactBinding:
    """One data artifact the run is bound to.

    ``sha256_prefix`` may be ``None`` in exactly two cases, both of which the
    validator enforces:

    - ``required_for`` records that the artifact is not needed by the field-mean
       path (the v2 patch store is that case, which is what makes patch-level
       cells ``Deferred`` rather than silently skipped);
    - ``hash_policy: pin_at_runtime`` with a ``pin_field`` — no owner-recorded
       hash exists for the file (``cycles_490_v2.parquet`` is that case), so the
       run hashes what it reads, writes it into the manifest, and a resume must
       match the pinned value.
    """

    path: str
    sha256_prefix: str | None
    record: str
    required_for: str = "all"
    hash_policy: str = "preregistered_prefix"
    pin_field: str | None = None

    @property
    def bound(self) -> bool:
        return bool(self.sha256_prefix)

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256_prefix": self.sha256_prefix,
            "record": self.record,
            "required_for": self.required_for,
            "hash_policy": self.hash_policy,
            "pin_field": self.pin_field,
            "bound": self.bound,
        }


@dataclass
class BenchmarkConfig:
    """The whole contract, as loaded and validated from YAML."""

    raw: dict = field(repr=False)
    path: str | None = None
    document_sha256: str = ""

    # ---- accessors -------------------------------------------------------
    @property
    def version(self) -> int:
        return int(self.raw["version"])

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.raw["seeds"])

    @property
    def screening_seeds(self) -> tuple[int, ...]:
        return tuple(int(s) for s in self.raw["screening_seeds"])

    @property
    def architectures(self) -> tuple[str, ...]:
        return tuple(self.raw["architectures"])

    @property
    def baselines(self) -> tuple[str, ...]:
        return tuple(self.raw["baselines"])

    @property
    def suppressions(self) -> tuple[str, ...]:
        return tuple(self.raw["suppressions"])

    @property
    def inputs(self) -> tuple[str, ...]:
        return tuple(self.raw["inputs"])

    @property
    def fusions(self) -> tuple[str, ...]:
        return tuple(self.raw["weather_fusions"])

    @property
    def calibrations(self) -> tuple[str, ...]:
        return tuple(self.raw["calibrations"])

    @property
    def protocols(self) -> tuple[str, ...]:
        return tuple(self.raw["protocols"])

    @property
    def levels(self) -> tuple[str, ...]:
        return tuple(self.raw["levels"])

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(self.raw["synthetic"]["families"])

    @property
    def severities(self) -> tuple[str, ...]:
        return tuple(self.raw["synthetic"]["severities"])

    @property
    def prevalence(self) -> float:
        return float(self.raw["synthetic"]["test_prevalence"])

    @property
    def capacity_target(self) -> int:
        return int(self.raw["capacity"]["target_params"])

    @property
    def capacity_band(self) -> tuple[float, float]:
        c = self.raw["capacity"]
        return float(c["min_ratio"]), float(c["max_ratio"])

    @property
    def t4_hours(self) -> float:
        return float(self.raw["budget"]["t4_hours"])

    @property
    def mamba_wall_factor(self) -> float:
        return float(self.raw["feasibility"]["mamba_wall_time_factor_of_gru_twin"])

    @property
    def evaluation(self) -> dict:
        return dict(self.raw["evaluation"])

    @property
    def training(self) -> dict:
        return dict(self.raw["training"])

    @property
    def collapse(self) -> dict:
        return dict(self.raw["collapse"])

    def artifact(self, name: str) -> ArtifactBinding:
        arts = self.raw.get("artifacts", {})
        if name not in arts:
            raise KeyError(f"no artifact binding named {name!r} in the benchmark config")
        a = arts[name]
        return ArtifactBinding(
            path=str(a["path"]),
            sha256_prefix=a.get("sha256_prefix"),
            record=str(a.get("record", "")),
            required_for=str(a.get("required_for", "all")),
        )

    @property
    def patch_binding(self) -> ArtifactBinding:
        """The v2 patch store binding; ``bound is False`` ⇒ patch cells defer."""
        return self.artifact("patch_store_v2")

    def factors(self) -> dict[str, tuple[str, ...]]:
        return {
            "architecture": self.architectures,
            "suppression": self.suppressions,
            "input": self.inputs,
            "fusion": self.fusions,
            "calibration": self.calibrations,
            "protocol": self.protocols,
            "level": self.levels,
            "seed": tuple(str(s) for s in self.seeds),
        }

    def to_dict(self) -> dict:
        return dict(self.raw)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _v_seeds(raw: dict, out: list[str]) -> None:
    seeds = tuple(raw.get("seeds", ()))
    if tuple(seeds) != PREREGISTERED_SEEDS:
        out.append(
            f"seeds must be the pre-registered {list(PREREGISTERED_SEEDS)} exactly, got {list(seeds)} "
            "(spec experimental-validation: three random seeds)"
        )
    scr = tuple(raw.get("screening_seeds", ()))
    if not scr:
        out.append("screening_seeds must be a non-empty subset of seeds")
    elif not set(scr) <= set(seeds):
        out.append(f"screening_seeds {list(scr)} is not a subset of seeds {list(seeds)}")


def _v_factors(raw: dict, out: list[str]) -> None:
    arches = list(raw.get("architectures", []))
    unknown = [a for a in arches if a not in ARCHITECTURES]
    if unknown:
        out.append(f"unknown architecture ids: {unknown} (A1–A10 are the pre-registered set)")
    want_paradigm = {ARCHITECTURES[a]["paradigm"] for a in arches if a in ARCHITECTURES}
    missing = sorted(set(PARADIGMS) - want_paradigm)
    if missing:
        out.append(
            f"declared architectures do not cover every paradigm: missing {missing}. "
            "A paradigm may only be dropped through `feasibility.not_applicable` with evidence."
        )
    for axis, table, key in (
        ("suppressions", SUPPRESSIONS, "suppressions"),
        ("inputs", INPUTS, "inputs"),
        ("weather_fusions", FUSIONS, "weather_fusions"),
        ("calibrations", CALIBRATIONS, "calibrations"),
        ("protocols", PROTOCOLS, "protocols"),
    ):
        bad = [v for v in raw.get(key, []) if v not in table]
        if bad:
            out.append(f"{axis} contains ids outside the pre-registered set: {bad}")
    if not raw.get("levels"):
        out.append("levels must name the scoring levels (at least 'farm'; 'patch' when bound)")
    unknown_levels = [v for v in raw.get("levels", []) if v not in ("farm", "patch")]
    if unknown_levels:
        out.append(f"unknown scoring levels: {unknown_levels}")


def _v_capacity(raw: dict, out: list[str]) -> None:
    cap = raw.get("capacity", {})
    target = cap.get("target_params")
    lo, hi = cap.get("min_ratio"), cap.get("max_ratio")
    if not isinstance(target, int) or target <= 0:
        out.append("capacity.target_params must be a positive integer")
        return
    if not (isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo < 1.0 < hi):
        out.append(f"capacity band ratios must bracket 1.0, got [{lo}, {hi}]")
    elif (hi - lo) > 0.4 + 1e-9:
        out.append(
            f"capacity band [{lo}, {hi}] is wider than the pre-registered ±20 % band; "
            "widening it after the fact is a hidden configuration change (integrity rule 7)"
        )
    laptop = cap.get("laptop_max_params")
    if not isinstance(laptop, int) or laptop <= 0:
        out.append("capacity.laptop_max_params must be set (AGENTS.md §6 laptop dev budget)")
    elif target > laptop:
        out.append(
            f"capacity.target_params {target} exceeds the laptop dev budget {laptop}; "
            "the dev profile would be unrunnable locally and the gate could not be passed"
        )


def _v_synthetic(raw: dict, out: list[str]) -> None:
    syn = raw.get("synthetic", {})
    fams = tuple(syn.get("families", ()))
    sevs = tuple(syn.get("severities", ()))
    if fams != FAMILIES:
        out.append(f"synthetic.families must be exactly {list(FAMILIES)}, got {list(fams)}")
    if sevs != SEVERITIES:
        out.append(f"synthetic.severities must be exactly {list(SEVERITIES)}, got {list(sevs)}")
    prev = syn.get("test_prevalence")
    if not isinstance(prev, (int, float)) or not (0.0 < float(prev) < 1.0):
        out.append("synthetic.test_prevalence must be a fraction in (0, 1) — the headline "
                   "macro-AUPRC is only comparable across cells at one predefined prevalence")
    if int(syn.get("events_per_cell", 0)) < 1:
        out.append("synthetic.events_per_cell must be ≥ 1 (cells without events cannot be scored)")
    if not syn.get("train_generator", {}).get("independent_from_eval"):
        out.append(
            "synthetic.train_generator.independent_from_eval must be true: S3 contamination "
            "and evaluation generation may not share a random state (spec anomaly-suppression-benchmark)"
        )


def _v_collapse(raw: dict, out: list[str]) -> None:
    col = raw.get("collapse", {})
    for key in (
        "effective_rank_ratio_floor",
        "min_per_dim_std",
        "max_zero_variance_dims_frac",
        "max_top_eigenvalue_share",
    ):
        if key not in col:
            out.append(f"collapse.{key} is missing — collapse monitoring is mandatory for A3/A4/A7/A8")
    if col.get("mechanism") not in (
        "stop_gradient",
        "ema_target_encoder",
        "stop_gradient+ema_target_encoder",
        "ema_target_encoder+variance_preservation",
    ):
        out.append(
            "collapse.mechanism must name the collapse-prevention mechanism "
            "(stop_gradient | ema_target_encoder); the design forbids leaving it implicit"
        )


def _v_evaluation(raw: dict, out: list[str]) -> None:
    ev = raw.get("evaluation", {})
    allowed = tuple(ev.get("allowed_splits", ()))
    if not allowed:
        out.append("evaluation.allowed_splits must be declared")
    for f in FORBIDDEN_SPLITS:
        if f in allowed:
            out.append(
                f"evaluation.allowed_splits contains {f!r}; this change never reads it "
                "(integrity rule 1)"
            )
    if "dev" not in allowed:
        out.append("evaluation.allowed_splits must contain 'dev' — selection has to come from somewhere")
    if int(ev.get("bootstrap_B", 0)) < 1000:
        out.append(f"evaluation.bootstrap_B must be ≥ 1000, got {ev.get('bootstrap_B')}")
    if ev.get("primary_metric") != "auprc":
        out.append("evaluation.primary_metric must be 'auprc' (AUPRC is the pre-registered primary)")
    if ev.get("macro_average") != "equal_cell_weight":
        out.append("evaluation.macro_average must be 'equal_cell_weight' — a prevalence-weighted "
                   "macro average is a different headline than the one that was registered")
    if not ev.get("per_cell_tables"):
        out.append("evaluation.per_cell_tables must be true: a macro average may not hide a failing cell")
    if tuple(ev.get("winner_reports", ())) != ("detection", "generalization", "multi_objective"):
        out.append("evaluation.winner_reports must be exactly the three pre-registered reports; "
                   "scalarising them into one score is forbidden (integrity rule 10)")


def _v_budget(raw: dict, out: list[str]) -> None:
    b = raw.get("budget", {})
    hours = b.get("t4_hours")
    if not isinstance(hours, (int, float)) or float(hours) <= 0:
        out.append("budget.t4_hours must be a positive number set by the owner for this cycle "
                   "(the scheduler may not silently run uncapped)")
    if int(b.get("max_epochs_screening", 0)) < 1:
        out.append("budget.max_epochs_screening must be ≥ 1")
    if b.get("on_exhaustion") != "defer":
        out.append("budget.on_exhaustion must be 'defer': budget exhaustion marks cells Deferred, "
                   "it never drops them")


def _v_feasibility(raw: dict, out: list[str]) -> None:
    f = raw.get("feasibility", {})
    fac = f.get("mamba_wall_time_factor_of_gru_twin")
    if not isinstance(fac, (int, float)) or float(fac) <= 0:
        out.append("feasibility.mamba_wall_time_factor_of_gru_twin must be set (design: 2× T4 budget)")
    na = f.get("not_applicable", [])
    if not isinstance(na, list):
        out.append("feasibility.not_applicable must be a list of {id, reason, evidence}")
    else:
        for entry in na:
            if not isinstance(entry, dict) or not entry.get("id") or not entry.get("reason"):
                out.append(f"feasibility.not_applicable entry {entry!r} needs id + reason")
            elif entry.get("evidence") in (None, ""):
                out.append(
                    f"feasibility.not_applicable[{entry.get('id')}] needs machine-checkable evidence; "
                    "an unsupported claim is how a silent substitution starts"
                )


def _v_blocks(raw: dict, out: list[str]) -> None:
    """The matrix is generated from ``blocks:`` — so the blocks must be checkable."""
    blocks = raw.get("blocks")
    if not isinstance(blocks, dict) or not blocks:
        out.append("blocks: must declare at least one pre-registered design block "
                   "(the matrix is generated from it, so an empty block list is an empty benchmark)")
        return
    known = set(ARCHITECTURES) | set(BASELINES)
    legal_joint = {"suppression", "input", "fusion", "calibration", "protocol"}
    for name, block in blocks.items():
        block = block or {}
        unknown = [k for k in block if k not in {
            "tier", "models", "suppression", "input", "fusion", "calibration", "protocol",
            "level", "variants", "seeds", "joint", "extra", "label",
        }]
        if unknown:
            out.append(f"blocks.{name} has unknown keys: {unknown}")
        if str(block.get("tier", "full")) not in ("screening", "full"):
            out.append(f"blocks.{name}.tier must be screening|full")
        models = block.get("models", "architectures")
        if isinstance(models, list):
            bad = [m for m in models if m not in known]
            if bad:
                out.append(f"blocks.{name}.models contains unknown ids: {bad}")
        for axis, table in (("suppression", SUPPRESSIONS), ("input", INPUTS), ("fusion", FUSIONS),
                            ("calibration", CALIBRATIONS), ("protocol", PROTOCOLS)):
            vals = block.get(axis)
            if isinstance(vals, list):
                bad = [v for v in vals if v not in table]
                if bad:
                    out.append(f"blocks.{name}.{axis} contains unknown ids: {bad}")
        for over in block.get("joint") or []:
            bad = [k for k in over if k not in legal_joint]
            if bad:
                out.append(f"blocks.{name}.joint entry {over!r} pins unknown factor(s) {bad}")
            for k, v in over.items():
                table = {"suppression": SUPPRESSIONS, "input": INPUTS, "fusion": FUSIONS,
                         "calibration": CALIBRATIONS, "protocol": PROTOCOLS}[k]
                if v not in table:
                    out.append(f"blocks.{name}.joint entry {over!r} sets {k}={v!r}, outside the registry")
        variants = block.get("variants")
        if variants is not None and not all(isinstance(v, str) and v for v in variants):
            out.append(f"blocks.{name}.variants must be non-empty strings")
    if "screening" not in [str(b.get("tier", "")) for b in blocks.values()] and not any(
        "screening" in str(name) for name in blocks
    ):
        out.append("no screening-tier block: budget-aware scheduling starts with screening "
                   "(design §Budget-aware execution)")


def _v_artifacts(raw: dict, out: list[str]) -> None:
    arts = raw.get("artifacts", {})
    for name in ("cycles_490_v2", "splits_locked_v2", "holdout_manifest_v2", "norm_stats_full_v2", "patch_store_v2"):
        if name not in arts:
            out.append(f"artifacts.{name} is missing — every run binds these four + the patch store")
            continue
        a = arts[name]
        if not a.get("path"):
            out.append(f"artifacts.{name}.path is empty")
        if not a.get("sha256_prefix") and a.get("required_for", "all") == "all":
            if a.get("hash_policy") != "pin_at_runtime":
                out.append(
                    f"artifacts.{name} has no sha256_prefix, no runtime hash policy and is "
                    "required_for=all: an unbound required artifact must be recorded, not run against"
                )
            elif not a.get("pin_field"):
                out.append(
                    f"artifacts.{name} uses hash_policy=pin_at_runtime but declares no pin_field "
                    "for the manifest to store the observed hash in"
                )
        if not a.get("record"):
            out.append(f"artifacts.{name}.record is empty (what the artifact is, in one line)")


def _v_structure(raw: dict) -> list[str]:
    out: list[str] = []
    for key in (
        "version", "created", "change", "seeds", "screening_seeds", "architectures", "baselines",
        "suppressions", "inputs", "weather_fusions", "calibrations", "protocols", "levels",
        "capacity", "training", "synthetic", "collapse", "evaluation", "budget", "feasibility",
        "artifacts", "hardware", "search", "blocks", "normality",
    ):
        if key not in raw:
            out.append(f"missing top-level key: {key}")
    if out:
        return out
    for fn in (_v_seeds, _v_factors, _v_capacity, _v_synthetic, _v_collapse, _v_evaluation,
               _v_budget, _v_feasibility, _v_artifacts, _v_blocks):
        fn(raw, out)
    hw = raw.get("hardware", {})
    if not hw.get("accelerator"):
        out.append("hardware.accelerator must be declared (Colab T4 is the target)")
    if raw.get("normality", {}).get("paradigm") != "hybrid":
        out.append("normality.paradigm must be 'hybrid' (farm history + phenological context + "
                   "environmental conditioning); a farm is never compared primarily to other farms")
    if not raw.get("search", {}).get("validation_only"):
        out.append("search.validation_only must be true: hyperparameter selection may not read dev-as-test "
                   "or the locked test split")
    budgets = raw.get("search", {}).get("budgets", {})
    for name, spec in budgets.items():
        if int(spec.get("trials", 0)) < 1:
            out.append(f"search.budgets.{name}.trials must be ≥ 1 and identical across architectures")
        if not spec.get("space"):
            out.append(f"search.budgets.{name}.space is empty — the design requires the search "
                       "space to be recorded, not just its size")
    trials = {int(s.get("trials", 0)) for s in budgets.values()}
    if len(trials) > 1:
        out.append(f"equal search budgets violated: trial counts differ {sorted(trials)} (design §Equal search budgets)")
    trainable = list(raw.get("architectures", [])) + [
        b for b in raw.get("baselines", []) if BASELINES.get(b, {}).get("trained")
    ]
    missing_budget = [m for m in trainable if m not in budgets]
    if missing_budget:
        out.append(
            f"trained models without a recorded search budget: {missing_budget}. Every arm needs the "
            "same predefined budget, and an arm with no entry silently gets zero tuning"
        )
    unsearchable = raw.get("search", {}).get("unsearchable", {})
    untrained = [b for b in raw.get("baselines", []) if not BASELINES.get(b, {}).get("trained")]
    missing_reason = [b for b in untrained if b not in unsearchable]
    if missing_reason:
        out.append(f"untrained references with no `search.unsearchable` reason: {missing_reason}")
    if raw.get("amendments") and not all(
        isinstance(a, dict) and a.get("date") and a.get("reason") and a.get("target")
        for a in raw["amendments"]
    ):
        out.append("every amendment needs date + target + reason (a mid-run config change is only "
                   "legal as a recorded amendment)")
    return out


def validate_config(raw: dict) -> list[str]:
    """Return every problem with ``raw`` as a benchmark contract (empty ⇒ valid)."""
    return _v_structure(raw)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def canonical_sha256(obj: Any) -> str:
    """Hash a config/dict canonically (sorted keys, no whitespace drift)."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_benchmark_config(path: str | Path | None = None, *, validate: bool = True) -> BenchmarkConfig:
    """Read + validate the benchmark config, recording its own document hash.

    The hash is what makes "the committed matrix is the contract" checkable: every
    run manifest stores ``benchmark_config_sha256``, so a resume against a modified
    config aborts with a diagnostic instead of continuing under new rules.
    """
    p = Path(path) if path is not None else CONFIG_PATH
    if not p.is_file():
        raise ConfigError([f"benchmark config not found at {p}"])
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError([f"{p} did not parse to a mapping"])
    if validate:
        problems = validate_config(raw)
        if problems:
            raise ConfigError(problems)
    return BenchmarkConfig(
        raw=raw,
        path=str(p),
        document_sha256=canonical_sha256(raw),
    )


def load_applicability(path: str | Path | None = None) -> dict:
    """Read the (architecture × suppression) applicability table.

    The table is binding: a pair marked ``not-applicable`` may never be scheduled,
    substituted or re-tuned into a different experiment.
    """
    p = Path(path) if path is not None else APPLICABILITY_PATH
    if not p.is_file():
        raise ConfigError([f"applicability table not found at {p}; the matrix cannot be committed without it"])
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "pairs" not in data:
        raise ConfigError([f"{p} must define a `pairs:` mapping"])
    problems: list[str] = []
    for pair, rec in data["pairs"].items():
        arch, _, supp = pair.partition("/")
        if arch not in ARCHITECTURES and arch not in BASELINES:
            problems.append(f"applicability row {pair!r} names an unknown model id")
        if supp not in SUPPRESSIONS:
            problems.append(f"applicability row {pair!r} names an unknown suppression id")
        status = (rec or {}).get("status")
        if status not in ("applicable", "not-applicable"):
            problems.append(f"applicability row {pair!r} status must be applicable|not-applicable, got {status!r}")
        if status == "not-applicable" and not (rec or {}).get("reason"):
            problems.append(f"applicability row {pair!r} is not-applicable without a reason")
        if status == "applicable" and (rec or {}).get("adapter") and not (rec or {}).get("adapter_doc"):
            problems.append(f"applicability row {pair!r} declares an adapter but no adapter_doc")
    declared = data.get("sha256")
    if declared:
        # The file may pin its own hash (it does not today); when it does, a stale pin is a
        # binding error, not a note — the matrix is committed against this exact number.
        live = canonical_sha256({k: v for k, v in data.items() if k != "sha256"})
        if str(declared) != live:
            problems.append(f"declared sha256 {str(declared)[:12]}… ≠ recomputed {live[:12]}… for {p}")
    if problems:
        raise ConfigError(problems)
    return data


def applicability_sha256_of(data: dict) -> str:
    """The hash the whole benchmark quotes for the applicability table.

    A declared ``sha256`` key is stripped first: a self-referential hash would change
    whenever the file grew one line of prose about the hash, which is not drift worth
    freezing a matrix over.
    """
    return canonical_sha256({k: v for k, v in (data or {}).items() if k != "sha256"})


def applicability_sha256(path: str | Path | None = None) -> str:
    return applicability_sha256_of(load_applicability(path))

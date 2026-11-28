"""Benchmark execution engine (Phase 9; spec reproducible-benchmark-execution).

The order of operations inside a cell run is itself a contract, because several gates
only work if they fire at the right moment:

1. **Bindings and budget first.** The committed matrix hash, the artifact hashes and
   the code SHA are verified (:mod:`.bindings`), and :meth:`BudgetMeter.has_room` is
   consulted **before** a cell is claimed, so the T4 budget never cuts a cell in half.
   When the budget runs out the remaining cells are ``Deferred`` with the reason
   recorded — the spec's "budget exhaustion" branch, which no report may omit.
2. **Objective audit before training.** An arm whose declared objective aliases
   another's fails the cell here, with the offending fields in the reason, rather than
   producing a number that later turns out to be the same loss twice.
3. **Train with suppression inside the arm.** The runner passes ``weights=None`` and
   the arm applies its own S1/S3 weights and S2 masking (:mod:`ssl_anomaly.suppression`),
   which is what keeps the runner's loss line identical for every architecture.
4. **Collapse disqualifies.** A latent arm meeting the contract's trigger is marked
   ``Failed`` with the measured rank/variance numbers — not dropped, not reported as
   a low score.
5. **Score the arm's own error**, mask-aware, then aggregate bin → cycle
   (:mod:`.scoring`); patch level is absent while the patch store is unbound.
6. **Calibrate on train/dev only** (:mod:`.calibration`), then metrics (:mod:`.metrics`)
   and cost (:mod:`.cost`).
7. **Manifest, sidecars, then the terminal transition.** A cell is not ``Completed``
   unless ``audit_run_dir`` finds its manifest, metrics and progress log: the rule
   "no result without a recorded manifest" is enforced here, at the only moment it
   can be.

``fixture`` mode exists for the local gate only: it drives this entire path on
in-memory tensors so ``--smoke`` can prove the plumbing on a laptop without the v2
data. Every artifact it writes is stamped ``SMOKE FIXTURE``, the ladder rung is
recorded as ``L0``, and the report repeats the stamp — a plumbing proof must not be
mistakable for a result (AGENTS.md §6.1).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ssl_anomaly.benchmarking import bindings as _bind
from ssl_anomaly.benchmarking import checkpoints as _ck
from ssl_anomaly.benchmarking import collapse as _coll
from ssl_anomaly.benchmarking import environment as _env
from ssl_anomaly.benchmarking import inputs as _inputs
from ssl_anomaly.benchmarking import interface as _if
from ssl_anomaly.benchmarking import metrics as _metrics
from ssl_anomaly.benchmarking import registry as _reg
from ssl_anomaly.benchmarking import scoring as _sc
from ssl_anomaly.benchmarking import synthetic as _syn
from ssl_anomaly.benchmarking.config import ARCHITECTURES, BenchmarkConfig
from ssl_anomaly.benchmarking.matrix import ExperimentCell, status_counts

#: keys that define the parameter band — fixed by capacity tuning, never by HPO
CAPACITY_KEYS = frozenset({"d_model"})

#: what an arm's constructor accepts (mirrors `arms._arm_kwargs`). A searched key outside both
#: this set and `_TRAIN_KEYS` is refused rather than dropped: a config nobody consumes is how an
#: arm ends up recorded as "tuned at lr=…" while it was trained at the contract default.
_ARM_KEYS = frozenset({
    "n_bins", "d_model", "n_layers", "dropout", "d_state", "latent_dim", "huber_beta",
    "horizon_k", "ema_momentum", "variance_weight", "lambda_rec", "lambda_pred", "backbone",
    "bottleneck_frac", "mask_rate",
})
#: the optimisation-side keys, applied per cell inside `_train`
_TRAIN_KEYS = frozenset({"lr", "weight_decay", "batch_size", "epochs", "warmup_epochs",
                         "eval_every_epochs", "grad_clip", "optimizer"})
#: keys the classical reference fitters take (see `suppression.classical.fit_score`)
_REF_KEYS = frozenset({"n_estimators", "max_samples", "contamination", "nu", "gamma", "lam",
                       "iters"})


SMOKE_STAMP = "SMOKE FIXTURE — plumbing evidence only, not a scientific result"

#: Mamba id → its GRU twin, for the feasibility gate (measured inside the same run).
GRU_TWIN = {"A2": "A1", "A4": "A3", "A6": "A5", "A8": "A7", "A10": "A9"}


class RunPlanError(RuntimeError):
    """The run cannot start or a cell cannot legitimately be claimed."""


@dataclass
class RunOptions:
    base: Path
    out: Path
    tier: str | None = "full"
    blocks: tuple[str, ...] = ()
    model_ids: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    limit: int | None = None
    epochs: int | None = None
    batch_size: int | None = None
    resume: bool = False
    smoke: bool = False
    fixture: bool = False
    dry_run: bool = False
    boot_once: bool = False
    allow_cpu_training: bool = False
    cpu_references: bool = False
    hpo: bool = False
    hpo_trials: int | None = None
    matrix_path: Path | None = None
    #: per-arm hyperparameters chosen by `scripts/benchmarks/run_hpo.py`, as
    #: ``{model_id: {key: value}}``. A selection layer, never a contract edit: the frozen
    #: config keeps its hash, and every cell manifest records what was applied on top of it.
    selection: dict[str, dict] | None = None

    @property
    def is_smoke(self) -> bool:
        return bool(self.smoke or self.boot_once or self.fixture)

    @property
    def ladder_rung(self) -> str:
        return "L0 (fixture/smoke: plumbing only)" if self.is_smoke else "L3 (multi-seed, single study)"


@dataclass
class CellResult:
    experiment_id: str
    status: str
    reason: str | None
    wall_seconds: float
    metrics: dict = field(default_factory=dict)
    cost: dict = field(default_factory=dict)
    run_dir: Path | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id, "status": self.status, "reason": self.reason,
            "wall_seconds": round(float(self.wall_seconds), 2), "metrics": self.metrics,
            "cost": self.cost, "run_dir": str(self.run_dir) if self.run_dir else None,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


class TensorFrame:
    """The narrow view of the cycle archive this runner needs (also the fixture shape).

    Deliberately a *view*, not a copy of ``CycleData``: the real object keeps NaN
    semantics that the input builder converts into the mask/value pair, and the
    fixture must be indistinguishable to the runner.
    """

    def __init__(self, *, x, m, field_ids, cycle_ids, split, channel_names, data_source,
                 notes=None, start_days=None, end_days=None):
        self.x = np.asarray(x, dtype=np.float32)
        self.m = np.asarray(m, dtype=bool)
        self.field_ids = np.asarray(field_ids, dtype=np.int64)
        self.cycle_ids = np.asarray(cycle_ids, dtype=object)
        self.split = np.asarray(split, dtype=object)
        self.channel_names = tuple(channel_names)
        self.data_source = data_source
        self.notes = list(notes or [])
        # Cycle dates drive the temporal protocol (V1/V3). When the source table has no
        # date columns the row order is used as a monotone proxy and the fact is stated:
        # the V1 overlap assertion then means "train rows precede scored rows in the
        # archive order", which is what a partition-by-split guarantee can mean at all.
        n = int(self.x.shape[0])
        self.start_days = (np.arange(n, dtype=np.int64) if start_days is None
                           else np.asarray(start_days, dtype=np.int64).ravel())
        self.end_days = (self.start_days + 1 if end_days is None
                         else np.asarray(end_days, dtype=np.int64).ravel())
        if start_days is None:
            self.notes.append("start_days derived from row order (no date column in the source); "
                              "temporal-protocol assertions use archive order as the proxy")

    @property
    def n(self) -> int:
        return int(self.x.shape[0])

    def select(self, idx) -> TensorFrame:
        i = np.asarray(idx, dtype=np.int64)
        return TensorFrame(x=self.x[i], m=self.m[i], field_ids=self.field_ids[i],
                           cycle_ids=self.cycle_ids[i], split=self.split[i],
                           channel_names=self.channel_names, data_source=self.data_source,
                           notes=self.notes, start_days=self.start_days[i], end_days=self.end_days[i])


def _fixture_frame(channels: tuple[str, ...]) -> TensorFrame:
    """Deterministic in-memory stand-in: same shapes, same mask statistics, no results.

    Fields share a seasonal offset so farm-level aggregation and the V2 farm holdout
    have something real to partition over; the anomaly panel is produced by the
    *actual* generator so the scoring path is exercised, not stubbed.
    """
    n_train, n_dev, T, C = 160, 160, 36, len(channels)
    n = n_train + n_dev
    rng = np.random.default_rng(4)
    m = rng.random((n, T, C)) > 0.12
    seas = np.sin(np.arange(T) / T * 6.0)[None, :] + rng.normal(scale=0.30, size=(n, 1))
    per_ch = np.linspace(0.35, 1.0, C)[None, None, :]
    x = (rng.normal(scale=0.18, size=(n, T, C)) + seas[:, :, None] * per_ch).astype(np.float32) * m
    field = np.repeat(np.arange(n // 8), 8)[:n]
    return TensorFrame(
        x=x.astype(np.float32), m=m, field_ids=field,
        cycle_ids=np.array([f"{int(f)}:{i}" for i, f in enumerate(field)], dtype=object),
        split=np.array(["train"] * n_train + ["dev"] * (n - n_train), dtype=object),
        channel_names=channels, data_source="fixture",
        start_days=np.arange(n, dtype=np.int64) * 8, end_days=np.arange(n, dtype=np.int64) * 8 + 7,
        notes=[SMOKE_STAMP, "shapes and mask statistics follow configs/benchmarks; values do not "
                            "come from the v2 archive"],
    )


def load_clean_data(cfg: BenchmarkConfig, *, base: Path, fixture: bool) -> TensorFrame:
    if fixture:
        return _fixture_frame(tuple(cfg.raw["training"]["channels"]))
    from ssl_anomaly.benchmarking.splits import load_admissible_splits

    frame = load_admissible_splits(
        cfg, base=base, splits=tuple(cfg.raw["evaluation"].get("allowed_splits") or ("train", "dev"))
    )
    return frame.to_tensor_frame()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """One run = the committed matrix filtered to a plan, executed in priority order."""

    def __init__(self, cfg: BenchmarkConfig, applicability: dict, cells: list[ExperimentCell],
                 opts: RunOptions) -> None:
        self.cfg = cfg
        self.app = applicability
        self.cells = list(cells)
        self.opts = opts
        self.out = Path(opts.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.registry = _reg.Registry(self.out / "registry.json")
        self.matrix_sha = ""
        self.device: _env.DeviceReport | None = None
        self.budget: _env.BudgetMeter | None = None
        self.bindings: Any = None
        self.capacity: dict = {}
        self.panel: Any = None
        self.contamination: Any = None
        self.frame: TensorFrame | None = None
        self.bundles: dict[str, Any] = {}
        self.tensors: dict[str, dict] = {}
        self.hashes: dict[str, str] = {}
        self.disjointness: dict = {}
        self.objective_audit: dict = {}
        self.notes: list[str] = []
        self._wall: dict[str, float] = {}

    # -- setup -------------------------------------------------------------
    @property
    def applicability_sha(self) -> str:
        """Hash of the loaded applicability table — the only value the matrix may cite.

        This used to read ``self.app.get("sha256")``, a key the table never declares: every
        cell silently carried an empty applicability hash, and a matrix committed with the
        real one then refused to match. One helper, one truth.
        """
        from ssl_anomaly.benchmarking.config import applicability_sha256_of

        return applicability_sha256_of(self.app)

    def prepare(self) -> dict:
        from ssl_anomaly.benchmarking.matrix import matrix_document, read_matrix, write_matrix

        o = self.opts
        if o.hpo:
            raise RunPlanError(
                "RunOptions.hpo is accepted by the dataclass but not implemented here: the search is "
                "run by scripts/benchmarks/run_hpo.py, which writes reports/architecture_benchmark/hpo/. "
                "Silently ignoring the flag would record an 'unsearched' arm as if it had been searched."
            )
        doc = matrix_document(self.cfg, self.cells,
                              applicability_sha=self.applicability_sha,
                              note="generated by the runner from the same inputs as the committed matrix")
        self.matrix_sha = doc["matrix_sha256"]
        committed = Path(o.matrix_path) if o.matrix_path else self.out / "matrix.yaml"
        if not committed.is_file():
            write_matrix(doc, committed)
            self.notes.append(f"no committed matrix at {committed}; wrote the freshly built one")
        else:
            back = read_matrix(committed)
            if back.get("matrix_sha256") not in (None, self.matrix_sha):
                raise RunPlanError(
                    f"committed matrix {str(back.get('matrix_sha256'))[:12]}… ≠ recomputed {self.matrix_sha[:12]}… "
                    "— the matrix is frozen, so a factor change requires a recorded amendment and a new "
                    "matrix hash, not a re-run that silently redefines the cells"
                )
        # A bundle must be readable without the repo next to it: mirror the exact matrix the
        # run bound to into the output directory, so `matrix.yaml` + `registry.json` +
        # manifests are enough to know which cells were on the menu and which were served.
        mirror = self.out / "matrix.yaml"
        if committed.resolve() != mirror.resolve():
            mirror.parent.mkdir(parents=True, exist_ok=True)
            mirror.write_bytes(Path(committed).read_bytes())
            self.notes.append(f"mirrored the bound matrix ({committed}) into {mirror} for bundle "
                              "self-description")
        self.registry.register(self.cells, matrix_sha256=self.matrix_sha)

        # a plan is not a training run: requiring CUDA here would make `--plan` unusable on the
        # laptop where the owner decides what to send to Colab
        self.device = _env.detect_device(self.cfg, require=(not o.allow_cpu_training and
                                                            not o.is_smoke and not o.dry_run))
        self.budget = _env.BudgetMeter(float(self.cfg.raw["budget"]["t4_hours"]))
        self.bindings = _bind.verify_bindings(
            self.cfg, base=o.base, matrix_sha256=self.matrix_sha,
            applicability_sha256=self.applicability_sha,
            compute_hashes=not o.is_smoke, strict=not o.is_smoke,
        )
        _bind.save_bindings(self.bindings, self.out / "bindings.json")
        rep = self.bindings.to_dict() if hasattr(self.bindings, "to_dict") else {}
        if rep.get("blocking") and not o.is_smoke:
            raise RunPlanError(
                "artifact bindings are blocking: " + json.dumps(rep.get("blocking"), default=str)[:400]
            )

        self.frame = load_clean_data(self.cfg, base=o.base, fixture=(o.fixture or o.is_smoke))
        self.panel = _syn.apply_panel(
            self.frame, self.cfg, seed=int(self.cfg.raw["synthetic"]["eval_generator"]["seed"]),
            pool=str(self.cfg.raw["synthetic"]["eval_generator"].get("pool", "dev_cycles_only")),
            role="evaluation", cycle_subset=self.eval_idx,
        )
        tr_idx = self.train_idx
        # The contaminated stream is built on the TRAIN rows only, by the train
        # generator's own seed: an eval event leaking into training would teach the
        # arm the exact signature it is later scored on.
        self.contamination = _syn.apply_panel(
            self.frame.select(tr_idx), self.cfg,
            seed=int(self.cfg.raw["synthetic"]["train_generator"]["seed"]),
            pool=str(self.cfg.raw["synthetic"]["train_generator"].get("pool", "train_cycles_only")),
            role="train_contamination",
        )
        self.disjointness = _syn.pool_disjointness(self.panel, self.contamination)
        if not self.disjointness["disjoint"]:
            self.notes.append(
                f"synthetic train/eval pools share {self.disjointness['shared']} event identities; the "
                "contract requires disjoint_sample_pool=true, so every S3 cell is invalid until fixed"
            )

        ch = tuple(self.frame.channel_names)
        for iid in ("I1", "I2", "I3"):
            inj = TensorFrame(x=self.panel.x, m=self.frame.m, field_ids=self.frame.field_ids,
                              cycle_ids=self.frame.cycle_ids, split=self.frame.split, channel_names=ch,
                              data_source=self.frame.data_source, notes=list(self.frame.notes))
            b = _inputs.build_tensors(inj, self.cfg, input_id=iid, fusion="W1")
            self.bundles[iid] = b
            self.tensors[iid] = {**b.tensors.to_torch(self.device.device), **b.tensors.index_tensors(),
                                 "start_days": getattr(b.tensors, "start_days", None)}
        self.hashes = {
            "data_sha256": _sha(self.frame.x),
            "panel_x_sha256": _sha(self.panel.x),
            "panel_labels_sha256": _sha_bool(self.panel.labels_bin),
            "contamination_labels_sha256": _sha_bool(self.contamination.labels_bin),
            "panel_generator_sha256": self.panel.generator_sha256,
            "contamination_generator_sha256": self.contamination.generator_sha256,
            "train_pool_sha256": _sha_int(tr_idx),
            "eval_pool_sha256": _sha_int(self.eval_idx),
            "covariate_sha256": _sha(self.bundles["I3"].tensors.cov),
            "config_sha256": self.cfg.document_sha256,
        }
        from ssl_anomaly.benchmarking.capacity import resolve_capacities

        self.capacity = resolve_capacities(
            self.cfg, d_channels=int(self.bundles["I3"].tensors.n_channels),
            d_cov=int(self.bundles["I3"].tensors.cov.shape[-1]),
        )
        (self.out / "capacity.json").write_text(
            json.dumps(self.capacity, indent=1, sort_keys=True, default=str), encoding="utf-8")
        self.objective_audit = self._audit_objectives()
        if self.objective_audit["problems"]:
            raise RunPlanError("objective audit failed before any cell ran: "
                               + "; ".join(self.objective_audit["problems"]))
        return {
            "matrix_sha256": self.matrix_sha,
            "cells": len(self.cells),
            "statuses": status_counts(self.cells),
            "device": self.device.to_dict(),
            "bindings": rep,
            "panel": {"usable": self.panel.usable_cells, "excluded": self.panel.excluded_cells,
                      "audit": self.panel.audit_summary, "n_events": self.panel.to_dict()["n_events_total"]},
            "capacity_all_in_band": self.capacity["all_in_band"],
            "pool_disjointness": self.disjointness,
            "objective_audit": self.objective_audit,
            "hashes": self.hashes,
            "notes": self.notes,
            "stamp": SMOKE_STAMP if o.is_smoke else None,
        }

    @property
    def train_idx(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.frame.split).astype(str) == "train")

    @property
    def eval_idx(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.frame.split).astype(str) != "train")

    def _audit_objectives(self) -> dict:
        from ssl_anomaly.models.bench.arms import ARM_BY_MODEL, objective_signature

        specs = {m: {**objective_signature(m), "gradient_path": "declared in the design table",
                     "collapse_mechanism": ("ema+variance"
                                            if objective_signature(m)["collapse_prevention_required"] else None),
                     "lambdas": {"lambda_rec": 1.0, "lambda_pred": 1.0}}
                 for m in ARM_BY_MODEL}
        return {"models": sorted(specs), "problems": _if.assert_objectives_distinct(specs),
                "signatures": {m: objective_signature(m) for m in sorted(specs)},
                "ran_at": "before the first cell"}

    # -- planning ----------------------------------------------------------
    def plan(self) -> list[ExperimentCell]:
        o = self.opts
        ready = [c for c in self.cells
                 if self.registry.get(c.cell_hash) is not None
                 and self.registry.get(c.cell_hash).status == "Pending"]
        if o.tier:
            ready = [c for c in ready if c.tier == o.tier]
        if o.blocks:
            ready = [c for c in ready if c.block in set(o.blocks)]
        if o.model_ids:
            ready = [c for c in ready if c.model_id in set(o.model_ids)]
        if o.seeds:
            ready = [c for c in ready if c.seed in set(o.seeds)]
        ready.sort(key=lambda c: c.priority)
        if o.boot_once:
            return ready[:1]
        if o.smoke:
            return self._smoke_subset(ready)
        return ready[: int(o.limit)] if o.limit else ready

    def _smoke_subset(self, cells: list[ExperimentCell]) -> list[ExperimentCell]:
        """One cell per (paradigm, backbone, suppression), plus the rule/IF references."""
        picked: list[ExperimentCell] = []
        seen: set[tuple[str, str, str]] = set()
        for c in cells:
            key = (c.paradigm or c.model_id, c.backbone or "", c.suppression)
            if c.level != "farm" or key in seen:
                continue
            seen.add(key)
            picked.append(c)
        for mid in ("B1", "BL-IF", "BL-RULE", "A-RAND"):
            extra = [c for c in cells if c.model_id == mid and c.level == "farm" and c not in picked]
            if extra:
                picked.append(extra[0])
        return picked

    def _pair_twins(self, cells: list[ExperimentCell]) -> list[ExperimentCell]:
        """Move each GRU twin immediately before its Mamba partner, keeping priority order.

        The feasibility gate compares wall times inside the run; if the Mamba cell were
        scheduled first there would be nothing to compare against, and the gate would
        silently never fire — which is how a budget check stops being a check.
        """
        out: list[ExperimentCell] = []
        seen: set[str] = set()
        for c in cells:
            twin = GRU_TWIN.get(c.model_id)
            if c.backbone == "mamba" and twin:
                match = [d for d in cells if d.model_id == twin and d.suppression == c.suppression
                         and d.seed == c.seed and d.tier == c.tier and d.input == c.input]
                if match and match[0].experiment_id not in seen:
                    out.append(match[0])
                    seen.add(match[0].experiment_id)
            if c.experiment_id not in seen:
                out.append(c)
                seen.add(c.experiment_id)
        return out

    def run(self) -> dict:
        if self.device is None or self.registry is None:
            raise RunPlanError(
                "run() requires prepare(): without the bindings, matrix, device and registry audit "
                "there is nothing honest to write into a summary. This used to surface as a "
                "'NoneType' object has no attribute 'to_dict' three hundred lines later.")
        cells = self._pair_twins(self.plan())
        if self.opts.dry_run:
            return {"dry_run": True, "planned": len(cells), "ids": [c.experiment_id for c in cells],
                    "statuses": status_counts(self.cells), "notes": self.notes}
        results: list[CellResult] = []
        t0 = time.time()
        for i, cell in enumerate(cells):
            if not self.budget.has_room(cell.model_id, tier=cell.tier):
                self.notes.append(
                    f"budget exhausted before cell {i + 1}/{len(cells)} "
                    f"({self.budget.remaining_hours:.2f} h left, next estimate "
                    f"{self.budget.estimate(cell.model_id, tier=cell.tier)} h)"
                )
                self.registry.defer_remaining(reason=(
                    f"T4 budget {self.cfg.raw['budget']['t4_hours']} h exhausted; on_exhaustion=defer "
                    f"per the contract, with {len(cells) - i} cells left in the plan"))
                break
            try:
                results.append(self.run_cell(cell))
            except Exception as exc:  # noqa: BLE001 - one cell must not kill the run
                results.append(self._fail(cell, exc))
            r = results[-1]
            self.budget.charge(cell.model_id, r.wall_seconds, tier=cell.tier, cell=cell.experiment_id)
        summary = {
            "run": {"cells_planned": len(cells), "cells_executed": len(results),
                    "wall_seconds": round(time.time() - t0, 2), "budget": self.budget.to_dict(),
                    "smoke": self.opts.is_smoke, "ladder_rung": self.opts.ladder_rung,
                    "stamp": SMOKE_STAMP if self.opts.is_smoke else None},
            "by_status": _count([r.status for r in results]),
            "statuses_after": self.registry.counts(),
            "matrix_statuses": status_counts(self.cells),
            "results": [r.to_dict() for r in results],
            "notes": self.notes,
            "hashes": self.hashes,
            "panel": {"usable": self.panel.usable_cells, "excluded": self.panel.excluded_cells},
            "disjointness": self.disjointness,
            "objective_audit": {"problems": self.objective_audit["problems"],
                                "models": self.objective_audit["models"]},
            "device": self.device.to_dict(),
        }
        (self.out / "run_summary.json").write_text(
            json.dumps(summary, indent=1, sort_keys=True, default=str), encoding="utf-8")
        self.registry.write_markdown(self.out / "coverage.md")
        self._write_progress_index(results)
        return summary

    def _write_progress_index(self, results: list[CellResult]) -> None:
        rows = [{"experiment_id": r.experiment_id, "status": r.status, "wall_seconds": r.wall_seconds,
                 "macro_auprc": (r.metrics.get("macro") or {}).get("macro"), "reason": r.reason}
                for r in results]
        (self.out / "progress.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True, default=str) + "\n" for r in rows), encoding="utf-8")

    def _fail(self, cell: ExperimentCell, exc: Exception) -> CellResult:
        text = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc(limit=6)
        entry = self.registry.get(cell.cell_hash)
        if entry is not None and entry.status in ("Pending", "Running", "Interrupted"):
            try:
                self.registry.transition(cell.cell_hash, "Failed", error=text,
                                         metrics={"traceback_tail": tb.strip().splitlines()[-1]})
            except _reg.RegistryError as inner:
                self.notes.append(f"could not record the failure for {cell.experiment_id}: {inner}")
        # The run directory is attached even on failure: a crashed cell leaves a
        # checkpoint and a progress log behind, and the person reading the summary
        # needs to be told where they are (that is what makes a requeue decision
        # possible instead of a guess).
        rdir = _ck.run_dir(self.out / "runs", cell.experiment_id, cell.seed)
        return CellResult(cell.experiment_id, "Failed", text, 0.0, {}, {}, rdir,
                          notes=[f"traceback: {tb.strip()[-300:]}"])

    # -- one cell ----------------------------------------------------------
    def run_cell(self, cell: ExperimentCell) -> CellResult:
        import torch

        o = self.opts
        t_start = self._cell_t0 = time.time()
        entry = self.registry.get(cell.cell_hash)
        if entry is None:
            raise RunPlanError(f"{cell.experiment_id} is not registered")
        if entry.status in ("Pending", "Interrupted"):
            self.registry.transition(cell.cell_hash, "Running")
        elif entry.status != "Running":
            raise RunPlanError(f"cell status {entry.status} cannot be claimed for a run")

        rdir = _ck.run_dir(self.out / "runs", cell.experiment_id, cell.seed)
        rdir.mkdir(parents=True, exist_ok=True)
        # per-cell, not per-run: a cell that never trains must not inherit the previous cell's
        # effective training block, or its manifest would report an lr it never used
        self._eff = None
        notes: list[str] = list(self.frame.notes)
        if o.is_smoke:
            notes.append(SMOKE_STAMP)

        bundle = self.bundles[cell.input]
        tensors = self.tensors[cell.input]
        d_ch = int(bundle.tensors.n_channels)
        d_cv = int(bundle.tensors.cov.shape[-1])
        # Which kind of cell this is must be decided *before* an arm is built: a classical
        # reference has no torch arm to construct, and building one first made every B1/BL-*
        # cell die in the arm factory with "not one of the pre-registered arms" — 58 matrix
        # cells failing for a reason that had nothing to do with their design.
        is_ref = cell.model_id in self.cfg.baselines and cell.model_id != "A-RAND"
        is_control = cell.model_id == "A-RAND"
        if is_ref:
            metrics, cost, notes = self._run_reference(cell, rdir, notes)
            self._retire(cell, "Completed", None, rdir, metrics={
                "macro_auprc": (metrics.get("macro") or {}).get("macro"),
                "cells_used": (metrics.get("macro") or {}).get("n_cells_used"),
                "wall_seconds": round(time.time() - t_start, 2)})
            self._wall[cell.model_id] = time.time() - t_start
            return CellResult(cell.experiment_id, "Completed", None, time.time() - t_start,
                              metrics, cost, rdir, notes)

        arm = self._build_arm(cell, d_ch, d_cv)
        arm.set_suppression(cell.suppression, self._suppression_params(cell))
        spec = _jsonable(arm.objective_spec())
        probs = _if.assert_objectives_distinct({cell.model_id: spec})
        if probs:
            raise RunPlanError(f"objective audit failed for {cell.experiment_id}: {probs}")

        trainable = arm.trainable_parameter_report()
        if trainable["trainable_params"] == 0 and not is_control:
            raise RunPlanError(f"{cell.model_id} has 0 trainable parameters — refusing to train a frozen arm")

        if is_control:
            arm.eval()
            scores_bin, chan = self._score_all(arm, tensors, self.eval_idx)
            probs_all, cal_fit, thr = self._calibrate(arm, tensors, cell, scores_bin)
            metrics = self._metrics_for(cell, scores_bin, probs_all, thr)
            cost = self._cost_untrained(arm, cell)
            notes.append(
                "A-RAND: parameters frozen at initialisation and never stepped — this cell measures what "
                "the trunk's structure alone yields, which is the point of the capacity control")
            # The bundle rule is complete-or-defect, and `MINIMUM_RUN_FILES` is the same for a
            # control cell as for a trained one — so the control writes the two side-cars it
            # legitimately has, with the reason recorded, instead of leaving an "incomplete run
            # dir" that looks like a training failure.
            (rdir / "progress.jsonl").write_text(json.dumps(
                {"epoch": 0, "backend": "untrained_random_init", "wall_s": 0.0,
                 "note": "no optimisation step is taken; the score is the trunk's random structure"})
                + "\n", encoding="utf-8")
            (rdir / "norm_stats.json").write_text(json.dumps(
                {"available": False,
                 "reason": "random-init control: nothing was fitted, so there are no learned "
                           "per-channel statistics to report",
                 "data_sha256": self.hashes.get("data_sha256")}, indent=1), encoding="utf-8")
            self._write_artifacts(rdir, cell, arm, [], metrics, cost,
                                  {"verdict": "not_applicable_by_construction",
                                   "reasons": ["random-init control: no representation is learned"]},
                                  notes, cal_fit=cal_fit, thr=thr, scores_bin=scores_bin, chan=chan)
            self._retire(cell, "Completed", None, rdir, metrics={
                "macro_auprc": (metrics.get("macro") or {}).get("macro"),
                "cells_used": (metrics.get("macro") or {}).get("n_cells_used"),
                "wall_seconds": round(time.time() - t_start, 2)})
            self._wall[cell.model_id] = time.time() - t_start
            return CellResult(cell.experiment_id, "Completed", None, time.time() - t_start,
                              metrics, cost, rdir, notes)

        if True:  # references returned above; everything left trains a torch arm
            if cell.suppression == "S3":
                tensors = self._contaminated_tensors(cell, tensors)
                notes.append(
                    "S3: trained on the contract's contaminated train stream (independent generator "
                    "seed and disjoint cycle pool); evaluation data is unaffected")
            hist = self._train(cell, arm, tensors, rdir, notes)
            # Wall time is recorded *before* scoring so the cell's own cost line means its
            # training time and the feasibility gate compares twins on the same basis.
            self._wall[cell.model_id] = time.time() - t_start
            arm.eval()
            scores_bin, chan = self._score_all(arm, tensors, self.eval_idx)
            probs_all, cal_fit, thr = self._calibrate(arm, tensors, cell, scores_bin)
            metrics = self._metrics_for(cell, scores_bin, probs_all, thr)
            collapse_rep = self._collapse_report(cell, arm, tensors, hist)
            cost = self._cost(arm, tensors, hist, cell)
            self._norm_stats(rdir, arm, tensors, cell)
            feas = self._feasibility(cell, t_start)
            if feas is not None:
                cost["feasibility"] = feas
                notes.append(feas["note"])
                (rdir / "feasibility.json").write_text(json.dumps(feas, indent=1, default=str), encoding="utf-8")
            self._write_artifacts(rdir, cell, arm, hist, metrics, cost, collapse_rep, notes,
                                  cal_fit=cal_fit, thr=thr, scores_bin=scores_bin, chan=chan)
            if collapse_rep.get("verdict", "ok").startswith("collapsed"):
                self._retire(cell, "Failed",
                             "collapse gate: " + "; ".join(collapse_rep.get("reasons", []))[:400], rdir)
                return CellResult(cell.experiment_id, "Failed", "latent collapse",
                                  time.time() - t_start, metrics, cost, rdir, notes)
            if feas is not None and feas["triggered"]:
                self._retire(cell, "Not Applicable", feas["note"], rdir)
                return CellResult(cell.experiment_id, "Not Applicable", feas["note"],
                                  time.time() - t_start, metrics, cost, rdir, notes)
        self._retire(cell, "Completed", None, rdir, metrics={
            "macro_auprc": (metrics.get("macro") or {}).get("macro"),
            "cells_used": (metrics.get("macro") or {}).get("n_cells_used"),
            "wall_seconds": round(time.time() - t_start, 2)})
        self._wall[cell.model_id] = time.time() - t_start
        return CellResult(cell.experiment_id, "Completed", None, time.time() - t_start,
                          metrics, cost, rdir, notes)

    def _retire(self, cell: ExperimentCell, to: str, reason: str | None, rdir: Path,
                metrics: dict | None = None) -> None:
        self.registry.transition(cell.cell_hash, to, reason=reason,
                                 checkpoint=str(rdir / "last.pt") if (rdir / "last.pt").exists() else None,
                                 manifest=str(rdir / "run_manifest.json"), metrics=metrics or {})

    # -- components --------------------------------------------------------
    #: the training block a cell actually optimised with (None = contract defaults, untrained)
    _eff: dict | None = None

    def _selection_for(self, model_id: str) -> dict:
        """The searched config for this arm, or {} when it runs at contract defaults.

        Refusing a capacity key here is the point: `d_model` is what the ±20 % band fixes, so a
        selection that moved it would let an arm win by being bigger than its twin.
        """
        sel = (self.opts.selection or {}).get(model_id) or {}
        is_ref = model_id in self.cfg.baselines and model_id != "A-RAND"
        allowed = set(_ARM_KEYS) | set(_TRAIN_KEYS) | (set(_REF_KEYS) if is_ref else set())
        stray = sorted(set(sel) - allowed)
        if stray:
            raise RunPlanError(
                f"{model_id}: the selection carries {stray}, which nothing in this runner reads"
                + ("" if is_ref else " (a reference arm is not searched through torch keys)")
                + "; applying it silently would record the arm as tuned at settings it never used"
            )
        bad = sorted(k for k in sel if k in CAPACITY_KEYS)
        if bad:
            raise RunPlanError(
                f"{model_id}: the selection carries capacity key(s) {bad}; the parameter band is set by "
                "capacity tuning over the pre-registered grid and HPO may not move it, or the arms stop "
                "being comparable"
            )
        return dict(sel)

    def _build_arm(self, cell: ExperimentCell, d_ch: int, d_cv: int):
        from ssl_anomaly.models.bench.arms import build_arm

        cap = (self.capacity.get("arms", {}).get(cell.model_id) or {}).get("chosen") or {}
        t = self.cfg.raw["training"]
        ov: dict[str, Any] = {}
        if cap.get("d_model"):
            ov["d_model"] = int(cap["d_model"])
        lam_key = str(cell.variant or "base")
        variants = t.get("lambda_variants") or {}
        # The authority for "which variants exist" is the committed matrix, not a hardcoded
        # label list: the capacity-control block legitimately names `rand_gru`/`rand_mamba`,
        # and a check that only knows the λ blocks would fail every control cell for a reason
        # that has nothing to do with the control. A variant that appears *nowhere* in the
        # matrix is still a refusal, because that is a hand-edited cell.
        declared = {str(c.variant or "base") for c in self.cells if c.block == cell.block}
        if lam_key not in declared and lam_key not in variants:
            raise RunPlanError(
                f"cell {cell.experiment_id} names variant {lam_key!r}, which the committed matrix "
                f"never declares for block {cell.block!r} (it declares {sorted(declared)}) and which "
                "is not a key of training.lambda_variants "
                f"{sorted(variants)}; a mis-spelled λ variant must not silently train at the defaults"
            )
        if lam_key in variants:
            lam = variants[lam_key]
            ov.update(lambda_rec=float(lam["lambda_rec"]), lambda_pred=float(lam["lambda_pred"]))
        sel = self._selection_for(cell.model_id)
        ov.update({k: v for k, v in sel.items() if k in _ARM_KEYS})
        return build_arm(cell.model_id, cfg=self.cfg, d_channels=d_ch, d_cov=d_cv,
                         variant=lam_key if lam_key in variants else "base", **ov)

    def _suppression_params(self, cell: ExperimentCell) -> dict:
        row = (self.app.get("pairs") or {}).get(f"{cell.model_id}/{cell.suppression}", {}) or {}
        p = dict(row.get("params") or {})
        p.setdefault("rate", float((self.app.get("masking") or {}).get("rate", 0.15)))
        p.setdefault("warmup_epochs", int((self.app.get("weight_schedule") or {}).get("warmup_epochs", 3)))
        p.setdefault("ramp_epochs", int((self.app.get("weight_schedule") or {}).get("ramp_epochs", 5)))
        p.setdefault("residual_ema", float((self.app.get("weight_schedule") or {}).get("ema", 0.7)))
        p.setdefault("contamination", float(self.cfg.raw["synthetic"]["test_prevalence"]))
        if row.get("bound"):
            p.setdefault("w_min", float(row["bound"][0]))
        p["seed"] = int(cell.seed)
        return p

    def _contaminated_tensors(self, cell: ExperimentCell, tensors: dict) -> dict:
        """Swap the train rows' values for the contaminated stream, mask unchanged."""
        import torch

        t = dict(tensors)
        x = tensors["x"].clone()
        tr = torch.as_tensor(self.train_idx, device=x.device)
        cx = np.ascontiguousarray(self.contamination.x)
        if cx.shape[0] != len(self.train_idx):
            raise RunPlanError(
                f"contaminated train stream has {cx.shape[0]} rows for {len(self.train_idx)} train "
                "cycles; a positional mismatch would silently train on the wrong fields"
            )
        x[tr] = torch.as_tensor(cx, device=x.device, dtype=x.dtype)
        t["x"] = x
        return t

    def _train(self, cell: ExperimentCell, arm, tensors: dict, rdir: Path, notes: list[str]) -> list[dict]:
        import torch

        o = self.opts
        # per-cell view of the training block: `t` is read for lr / weight_decay / batch_size /
        # epochs throughout this method, so applying the selection once here means every one of
        # those reads — and the manifest that records them — sees the searched value
        t = {**self.cfg.raw["training"],
             **{k: v for k, v in self._selection_for(cell.model_id).items() if k in _TRAIN_KEYS}}
        self._eff = t
        bs = int(o.batch_size or t.get("batch_size", 32))
        ep_full = int(t.get("epochs_full", 40))
        ep_screen = int(self.cfg.raw["budget"].get("max_epochs_screening", 6))
        epochs = int(o.epochs or (ep_screen if cell.tier == "screening" else ep_full))
        if o.is_smoke:
            epochs = min(epochs, 2)
        plan = self.device.precision_plan(str(self.cfg.raw.get("hardware", {}).get("mixed_precision", "auto")))
        dev = torch.device(self.device.device)
        dtype = getattr(torch, plan["dtype"], torch.float32)
        opt = torch.optim.AdamW([p for p in arm.parameters() if p.requires_grad],
                                lr=float(t.get("lr", 2e-3)),
                                weight_decay=float(t.get("weight_decay", 1e-5)))
        clip = float(t.get("grad_clip_norm", 1.0))
        es = t.get("early_stop") or {}
        patience = int(es.get("patience", 6)) if not o.is_smoke else 1
        eval_every = int(t.get("eval_every_epochs", 2))
        hist: list[dict] = []
        best, bad, step = math.inf, 0, 0
        start = 0
        self._epochs_planned = int(epochs)
        self._prior_history: list[dict] = []
        self._resumed_from = 0          # per-cell; the manifest records it for that cell only
        if o.resume and _ck.has_resume_point(rdir):
            state = _ck.load_checkpoint(rdir, model=arm, optimizer=opt,
                                        expected_bindings=self._bindings_for(cell), strict=False)
            # load_checkpoint returns the torch payload with a CheckpointMeta attached under
            # "meta" — not a nested dict. Reading it as one would resume from epoch 0 and
            # still call itself a resume, which is the quiet version of this bug.
            meta = state.get("meta")
            done = getattr(meta, "epochs_done", None)
            if done is None and isinstance(meta, dict):
                done = meta.get("epochs_done")
            start = int(done or 0)
            # the pre-crash epochs are kept in the record, so the manifest states the cell's
            # *total* training and its history tail spans the crash instead of stopping at it
            self._prior_history = list(getattr(meta, "history", None) or [])
            notes.append(f"resumed from epoch {start} (bindings verified against the live run)")
            self._resumed_from = int(start)
        rows_tr = self.train_idx
        rows_ev = self.eval_idx
        # ``epochs`` is the cell's total budget, and a resume must not extend it: the
        # previous ``range(start, start + epochs)`` gave a resumed cell more training than
        # an identical cell that never crashed, which would have made the two incomparable
        # while both reported "epochs: N".
        if start >= epochs:
            notes.append(
                f"resume point is at epoch {start} with a budget of {epochs}: nothing left to train, "
                "the checkpoint is scored as it stands")
        for ep in range(start, max(int(start), int(epochs))):
            arm.train()
            sup = getattr(arm, "suppressor", None)
            if sup is not None:
                sup.on_epoch_start(ep, total_epochs=epochs)
            order = np.random.default_rng(int(cell.seed) * 1000 + ep).permutation(len(rows_tr))
            idx_ep = rows_tr[order]
            tot, nb, resid, ridx = 0.0, 0, [], []
            gnorm = float("nan")
            comp: dict = {}
            for batch in _if.iter_batches(tensors, idx_ep, bs, ctx_masks=None):
                batch = batch.to(dev)
                if sup is not None:
                    sup.on_batch(batch, step=step, total_steps=max(1, epochs * max(1, len(idx_ep) // bs)))
                if plan["autocast"] and dev.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        loss, comp = arm.compute_loss(batch)
                else:
                    loss, comp = arm.compute_loss(batch)
                opt.zero_grad(set_to_none=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"{cell.model_id}: non-finite loss at epoch {ep} step {step} "
                        f"(loss={float(loss)}; components {list(comp)[:4]}) — recorded, not skipped")
                loss.backward()
                if clip > 0:
                    gnorm = float(torch.nn.utils.clip_grad_norm_(
                        [p for p in arm.parameters() if p.requires_grad], clip))
                opt.step()
                tot += float(loss.detach())
                nb += 1
                step += 1
                e = comp.get("err") if isinstance(comp, dict) else None
                if e is not None:
                    resid.append(_np(e))
                    ridx.append(np.asarray(batch.meta["indices"]).ravel())
            if resid and sup is not None and hasattr(sup, "observe"):
                sup.observe(np.concatenate(resid, axis=0), np.concatenate(ridx, axis=0))
            dev_loss = self._objective_loss(arm, tensors, rows_ev[: min(len(rows_ev), 256)], bs, dev, plan)
            arm.eval()
            row = {"epoch": ep, "train_loss": round(tot / max(1, nb), 6),
                   "dev_objective_loss": round(dev_loss, 6), "grad_norm": _round(gnorm),
                   "batches": nb, "lr": float(t.get("lr", 2e-3)), "suppression": cell.suppression,
                   "components": {k: _round(v) for k, v in (comp or {}).items() if isinstance(v, (int, float))},
                   "effective_rank": self._rank_now(arm, tensors, rows_ev, dev),
                   "variance_term": (comp or {}).get("variance_term")}
            hist.append(row)
            with (rdir / "progress.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            if dev_loss < best - 1e-6:
                best, bad = dev_loss, 0
            else:
                bad += 1
            _ck.save_checkpoint(
                rdir, model=arm, optimizer=opt,
                meta=_ck.CheckpointMeta(
                    experiment_id=cell.experiment_id, cell_hash=cell.cell_hash, seed=cell.seed,
                    model_id=cell.model_id, epochs_done=ep + 1, global_step=step, best_metric=best,
                    bindings=self._bindings_for(cell), code=_bind.git_sha(self.opts.base),
                    objective=_jsonable(arm.objective_spec()), history=hist[-8:],
                    extras={"suppression": _jsonable(getattr(arm, "suppression_state", lambda: {})()),
                            "capacity": (self.capacity.get("arms") or {}).get(cell.model_id),
                            "precision": plan, "ladder_rung": self.opts.ladder_rung}),
                extra_tensors=_cpu_state(arm))
            if es.get("monitor") == "dev_objective_loss" and bad >= patience and (ep - start) >= eval_every:
                notes.append(f"early stopping at epoch {ep}: no improvement for {bad} evals "
                             f"(patience {patience}, monitor {es.get('monitor')})")
                break
        return hist

    def _objective_loss(self, arm, tensors, idx, bs, dev, plan) -> float:
        import torch

        arm.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for batch in _if.iter_batches(tensors, idx, bs):
                batch = batch.to(dev)
                if plan.get("autocast") and dev.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=getattr(torch, plan["dtype"])):
                        loss, _ = arm.compute_loss(batch)
                else:
                    loss, _ = arm.compute_loss(batch)
                tot += float(loss.detach())
                n += 1
        arm.train()
        return tot / max(1, n)

    def _rank_now(self, arm, tensors, idx, dev):
        import torch

        if not hasattr(arm, "latents_for_collapse"):
            return None
        b = _if.make_batch(tensors, np.asarray(idx)[: min(32, len(idx))]).to(dev)
        with torch.no_grad():
            z = arm.latents_for_collapse(b)
        if z is None:
            return None
        try:
            return round(float(_coll.latent_stats(_np(z))["effective_rank"]), 3)
        except Exception:  # too few rows for a covariance — recorded, not guessed
            return None

    def _score_all(self, arm, tensors: dict, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(N,T) bin error + (N,T,C) per-channel error, over the given rows only."""
        import torch

        n = int(tensors["x"].shape[0])
        T = int(tensors["x"].shape[1])
        C = int(tensors["x"].shape[2])
        out = np.full((n, T), np.nan)
        chan = np.full((n, T, C), np.nan)
        bs = int(self.opts.batch_size or self.cfg.raw["training"].get("batch_size", 32))
        arm.eval()
        with torch.no_grad():
            for batch in _if.iter_batches(tensors, idx, bs):
                b = batch.to(self.device.device)
                o = arm.forward(b)
                rows = np.asarray(b.meta["indices"]).ravel()
                out[rows] = _np(o["err"])
                if "recon" in o:
                    mask = _np(b.m)
                    with np.errstate(invalid="ignore"):
                        chan[rows] = np.where(mask > 0, np.abs(_np(o["recon"]) - _np(b.x)), np.nan)
        return out, chan

    def _calibrate(self, arm, tensors, cell: ExperimentCell, scores_bin: np.ndarray):
        """Map the arm's dev scores to probabilities and pick the operating threshold.

        The reference distribution is the arm's own **train** scores (never the
        scored partition), pooled at the level the calibration axis names — C1 the
        farm's own history, C2 the farm's own history at the same stage, C3 global —
        and the threshold quantile is taken on **dev** at the contract's shared alarm
        budget, so every arm is held to the same alert rate.
        """
        from ssl_anomaly.benchmarking.calibration import (calibrate, calibration_holdout_check,
                                                          cycle_stage, threshold_at_budget)

        tr, ev = self.train_idx, self.eval_idx
        s_bin = scores_bin
        s_cycle = _nanmax(s_bin, axis=1)
        s_train_cycle = _nanmax(self._score_rows(arm, tensors, tr), axis=1)
        y_cycle = np.asarray(self.panel.labels_cycle, dtype=int)[ev]
        normality = self.cfg.raw.get("normality", {}) or {}
        min_history = int(normality.get("min_history_cycles", 2))
        cal_kwargs: dict = {}
        stage_rec: dict = {}
        if cell.calibration in ("C1", "C2"):
            farm_tr = np.asarray(self.frame.field_ids, dtype=np.int64)[tr]
            farm_ev = np.asarray(self.frame.field_ids, dtype=np.int64)[ev]
            if cell.calibration == "C1":
                cal_kwargs = {"train_keys": farm_tr, "eval_keys": farm_ev}
            else:
                stages, rec_st = cycle_stage(
                    tensors.get("phase_bin"), stage_bins=int(normality.get("stage_bins", 12)),
                    start_days=tensors.get("start_days"))
                stage_rec = {"stage_source": rec_st.get("source"),
                             "stage_bins": int(normality.get("stage_bins", 12)),
                             "stage_note": rec_st.get("note", "")}
                cal_kwargs = {"train_keys": np.column_stack([farm_tr, stages[tr]]),
                              "eval_keys": np.column_stack([farm_ev, stages[ev]])}
            cal_kwargs["min_history"] = min_history
        p_eval, fit = calibrate(
            cell.calibration, scores_dev=s_cycle[ev], labels_dev=y_cycle,
            train_scores=s_train_cycle[tr], scores_eval=s_cycle[ev], split="dev", **cal_kwargs,
        )
        budget = float(self.cfg.raw["calibration"]["alarm_budget"])
        try:
            thr = threshold_at_budget(p_eval, budget=budget)
        except ValueError as exc:
            thr = {"threshold": None, "budget": budget, "error": str(exc)}
        if cell.calibration in ("C1", "C2"):
            thr["stage_axis"] = stage_rec
        # task 7.7 — the threshold's optimism on rows it was not picked on, recorded
        # beside the threshold instead of being claimed away (dev picks and reports).
        thr["calibration_holdout_check"] = calibration_holdout_check(
            p_eval, budget=budget, farm_ids=np.asarray(self.frame.field_ids)[ev])
        cal_contract = dict(self.cfg.raw["calibration"])
        thr["contract"] = {
            "sources": list(cal_contract.get("sources") or []),
            "thresholds_selected_on": cal_contract.get("thresholds_selected_on"),
            "test_usage": cal_contract.get("test_usage", "forbidden"),
            "forbidden_splits": list(self.cfg.raw["evaluation"].get("forbidden_splits") or []),
            "note": ("the mapping and the threshold come from the partitions the contract names; the "
                     "cost of selecting thresholds on dev is that dev is also the reported partition, "
                     "so the locked test split — which this run never reads — is the only unbiased number"),
        }
        thr["bin_threshold"] = _round(float(np.nanquantile(s_bin[ev], 1.0 - budget))) \
            if np.isfinite(s_bin[ev]).any() else None
        thr["note"] = (
            "cycle threshold = the dev probability quantile at the shared alarm budget; the bin "
            "threshold (also a dev quantile) is used only for detection delay"
        )
        p_all = np.full(len(s_cycle), np.nan)
        p_all[ev] = p_eval
        rec = fit.to_dict()
        src = str(rec.get("fitted_on") or "")
        allowed = [str(x) for x in (cal_contract.get("sources") or [])]
        if allowed and src and src not in allowed:
            raise RuntimeError(
                f"calibration was fitted on {src!r}, which calibration.sources ({allowed}) does not "
                "permit — the guard is the point, so this is a failure, not a warning")
        rec["allowed_sources"] = allowed
        rec["scored_partition_used_for_fitting"] = src in ("dev",) and len(ev) > 0 and bool(
            np.isin(self.eval_idx, np.arange(len(s_cycle))).any())
        return p_all, rec, thr

    def _score_rows(self, arm, tensors, idx) -> np.ndarray:
        import torch

        bs = int(self.opts.batch_size or self.cfg.raw["training"].get("batch_size", 32))
        n = int(tensors["x"].shape[0])
        T = int(tensors["x"].shape[1])
        out = np.full((n, T), np.nan)
        arm.eval()
        with torch.no_grad():
            for batch in _if.iter_batches(tensors, idx, bs):
                b = batch.to(self.device.device)
                rows = np.asarray(b.meta["indices"]).ravel()
                out[rows] = _np(arm.score_bins(b))
        return out

    def _dev_thresholds(self, scores_bin) -> tuple[float | None, float | None]:
        """Pooled dev thresholds from the repo's own pre-registered rule (101-point grid).

        Pooled over cells on purpose: the threshold belongs to the *run*, not to a cell —
        picking a separate threshold inside each evaluation cell would let the winner be
        chosen with the labels it is scored on.
        """
        from ssl_anomaly.evaluation.metrics import select_threshold_on_validation

        ev = self.eval_idx
        y_bin = np.asarray(self.panel.labels_bin, dtype=bool)[ev]
        s_bin = np.asarray(scores_bin, dtype=np.float64)[ev]
        ok = np.isfinite(s_bin)
        if int(ok.sum()) < 32 or not y_bin[ok].any():
            return None, None
        t_bin = select_threshold_on_validation(s_bin[ok], y_bin[ok])
        s_cyc = _nanmax(np.where(ok, s_bin, np.nan), axis=1)
        y_cyc = y_bin.any(axis=1)
        fin = np.isfinite(s_cyc)
        t_cyc = (select_threshold_on_validation(s_cyc[fin], y_cyc[fin])
                 if int(fin.sum()) >= 8 and y_cyc[fin].any() else None)
        return (None if t_bin is None or not np.isfinite(t_bin) else float(t_bin),
                None if t_cyc is None or not np.isfinite(t_cyc) else float(t_cyc))

    def _metrics_for(self, cell: ExperimentCell, scores_bin, probs, thr) -> dict:
        """Every scored number for one cell, from the one place that knows the eval index.

        Kept as a single method because the cell metrics, the delay, the per-farm scores and
        the dev threshold all have to be computed on *exactly* the same row selection; the
        last time those were spread across call sites, a metric silently used the full panel.
        """
        ev = self.eval_idx
        obs = np.asarray(self.frame.m, dtype=bool).any(-1)
        usable = set(self.panel.usable_cells)
        t_bin, t_cyc = self._dev_thresholds(scores_bin)
        thr = dict(thr or {})
        thr["dev_rule"] = {"bin_threshold": _round(t_bin), "cycle_threshold": _round(t_cyc),
                           "rule": str(self.cfg.raw["evaluation"].get("threshold_rule")),
                           "fitted_on": "dev (pooled over cells, never per cell)"}
        thr["bin_threshold"] = thr["bin_threshold"] if thr.get("bin_threshold") is not None else t_bin
        per: list[_metrics.CellMetrics] = []
        per_combo: list[_metrics.CellMetrics] = []
        delayed: list[float] = []
        for pc in self.panel.cells:
            if pc.cell not in usable:
                continue
            lab = pc.labels[ev]
            if not lab.any():
                continue
            sc = scores_bin[ev]
            m = _metrics.cell_metrics(
                cell=pc.cell, family=pc.family, severity=pc.severity, level=cell.level,
                scores_bin=sc, labels_bin=lab, scorable_bin=obs[ev] & np.isfinite(sc),
                scores_cycle=_nanmax(np.where(np.isfinite(sc), sc, np.nan), axis=1),
                labels_cycle=lab.any(axis=1), probs_cycle=probs[ev],
                alarm_budget=float(self.cfg.raw["calibration"]["alarm_budget"]),
                # Farms are the resampling unit for every comparison in the report, so the
                # per-farm numbers have to be recorded at the only moment they exist.
                farm_ids=np.asarray(self.frame.field_ids)[ev],
                dev_threshold=t_cyc)
            if np.isfinite(thr.get("bin_threshold", float("nan"))):
                dl = _sc.detection_delay(sc, lab, threshold=float(thr["bin_threshold"]),
                                          bin_days=int(self.cfg.raw["training"].get("bin_days", 15)))
                m.median_delay_days = _round(dl["median_days"])
                m.aux["delay_detail"] = dl
                if not getattr(pc, "is_combined", False):
                    delayed.extend([float(dl["median_days"])] if np.isfinite(dl["median_days"]) else [])
            # Task 4.9: combined-scenario cells are scored per-cell but the headline macro
            # stays the pre-registered 5 families × 3 severities — letting designed
            # multi-family events silently re-weight the headline would move the target
            # the contract froze without an amendment.
            (per_combo if getattr(pc, "is_combined", False) else per).append(m)
        macro = _metrics.macro_summary(per)
        macro["median_of_cell_medians_delay_days"] = _round(float(np.median(delayed))) if delayed else None
        macro["headline_grid"] = str(self.cfg.raw["evaluation"].get("headline_grid"))
        macro["n_combined_cells"] = len(per_combo)
        combined = {
            "cells": [c.to_dict() for c in per_combo],
            "macro": (_metrics.macro_summary(per_combo) if per_combo else
                      {"available": False, "reason": "no combined-scenario cell is usable"}),
            "note": ("designed multi-family events (task 4.9): reported per-cell with their component "
                     "families; excluded from the headline macro, which stays the 5 families × 3 "
                     "severities the contract pre-registers"),
        }
        return {"cells_used": [c.cell for c in per], "macro": macro,
                "per_cell": [c.to_dict() for c in per] + [c.to_dict() for c in per_combo],
                "combined_cells": combined,
                "threshold": _jsonable(thr), "calibration": {"method": cell.calibration},
                "prevalence": {"target": self.panel.prevalence_target,
                               "eval_bin": _round(float(self.panel.labels_bin[ev].mean())),
                               "eval_cycle": _round(float(self.panel.labels_cycle[ev].mean()))},
                "n_eval_cycles": int(len(ev)),
                "excluded_cells": [c for c in self.panel.excluded_cells
                                   if c["cell"] not in {p.cell for p in per}],
                "level_note": ("patch level is absent while the patch store is unbound; farm level is "
                               "the only scored level in this run")}

    def _collapse_report(self, cell: ExperimentCell, arm, tensors, hist) -> dict:
        import torch

        if not hasattr(arm, "latents_for_collapse"):
            return {"verdict": "not_applicable_by_construction",
                    "reasons": ["arm exposes no latent space"], "model_id": cell.model_id}
        idx = self.eval_idx[: min(64, len(self.eval_idx))]
        b = _if.make_batch(tensors, idx).to(self.device.device)
        with torch.no_grad():
            z = arm.latents_for_collapse(b)
        probe = getattr(arm, "probe_features", None)
        scores_prev = None
        scores_now = None
        if len(hist) >= 2:
            scores_prev = np.asarray([hist[-2]["dev_objective_loss"]])
            scores_now = np.asarray([hist[-1]["dev_objective_loss"]])
        rep = _coll.evaluate(model_id=cell.model_id, latents=None if z is None else _np(z),
                             phase_bin=np.asarray(tensors["phase_bin"])[idx], cfg=self.cfg,
                             variance_term=(hist[-1].get("variance_term") if hist else None),
                             scores_prev=scores_prev, scores_now=scores_now)
        d = rep.to_dict()
        # Two probes, on purpose. ``content_probe`` is the contract's
        # (collapse.content_probe.target: phase_bin) and is what the gate reads;
        # ``anomaly_content_probe`` asks the question the thesis actually cares about —
        # whether the latent still carries the event — and is reported as a
        # diagnostic, never as a gate, because it is not in the pre-registration.
        if probe is not None:
            try:
                d["content_probe"] = _jsonable(_coll.run_content_probe(
                    arm, tensors, np.asarray(tensors["phase_bin"], dtype=np.float64),
                    min_spearman=float(self.cfg.raw["collapse"]["content_probe"]["min_spearman"]),
                    fit_idx=self.train_idx, eval_idx=self.eval_idx, target_name="phase_bin", cfg=self.cfg))
                d["anomaly_content_probe"] = _jsonable(_coll.run_content_probe(
                    arm, tensors, np.asarray(self.panel.labels_cycle, dtype=np.float64),
                    min_spearman=0.0, fit_idx=self.train_idx, eval_idx=self.eval_idx,
                    target_name="cycle_label(diagnostic)", cfg=self.cfg))
            except Exception as exc:  # a probe that cannot run is a finding, not a silence
                d["content_probe"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        else:
            d["content_probe"] = {"available": False, "reason": "arm exposes no probe_features()",
                                  "passed": None}
        return d

    def _cost(self, arm, tensors, hist, cell: ExperimentCell) -> dict:
        from ssl_anomaly.models.bench.arms import estimate_flops

        import torch

        t = self.cfg.raw["training"]
        bs = int(self.opts.batch_size or t.get("batch_size", 32))
        dev = torch.device(self.device.device)
        arm.eval()
        lat: list[float] = []
        with torch.no_grad():
            for batch in _if.iter_batches(tensors, self.eval_idx[: min(128, len(self.eval_idx))], bs):
                b0 = time.time()
                arm.score_bins(batch.to(dev))
                lat.append((time.time() - b0) * 1000.0 / max(1, b0 and batch.x.shape[0]))
        vram = _env.VramMonitor(self.device.device)
        vram.sample("after eval")
        return {
            "params": arm.trainable_parameter_report(),
            "flops": estimate_flops(arm, batch_size=bs, n_bins=int(t.get("n_bins", 36))),
            "epochs_run": len(hist),
            "train_wall_seconds": _round(self._wall.get(cell.model_id, time.time() - self._cell_t0)),
            "inference_latency_ms_per_cycle": _round(float(np.mean(lat))) if lat else None,
            "inference_n_batches": len(lat),
            "vram": vram.to_dict(),
            "device": self.device.to_dict(),
        }

    def _cost_untrained(self, arm, cell: ExperimentCell) -> dict:
        from ssl_anomaly.models.bench.arms import estimate_flops

        t = self.cfg.raw["training"]
        return {
            "params": arm.trainable_parameter_report(),
            "flops": estimate_flops(arm, batch_size=int(self.opts.batch_size or t.get("batch_size", 32)),
                                    n_bins=int(t.get("n_bins", 36))),
            "epochs_run": 0, "train_wall_seconds": 0.0, "trained": False,
            "note": "random-init capacity control: no optimisation step by construction",
            "device": self.device.to_dict(), "inference_latency_ms_per_cycle": None,
        }

    def _norm_stats(self, rdir: Path, arm, tensors, cell: ExperimentCell) -> None:
        """Per-channel robust scale of the TRAIN residuals — the bundle's norm sidecar.

        Without it the per-channel errors in ``bin_scores.npz`` are uninterpretable,
        which is exactly the defect that made an earlier patch bundle unusable for
        re-analysis (AGENTS.md §6.2).
        """
        chan = self._score_all(arm, tensors, self.train_idx)[1][self.train_idx]
        mask = np.asarray(self.frame.m, dtype=bool)[self.train_idx]
        try:
            norm = _sc.ScoreNormaliser.from_errors(np.nan_to_num(chan, nan=0.0), mask,
                                                    fitted_on="train", n_cycles=int(len(self.train_idx)))
            payload = {"available": True, **norm.to_dict(),
                       "partition": "train only (the scored partition never fitted anything)",
                       "labels_sha256": self.hashes.get("panel_labels_sha256")}
        except Exception as exc:
            payload = {"available": False, "reason": f"{type(exc).__name__}: {exc}",
                       "consequence": "per-channel errors are reported unscaled; no default is invented"}
        (rdir / "norm_stats.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=str),
                                               encoding="utf-8")

    def _bindings_for(self, cell: ExperimentCell) -> dict:
        return {"benchmark_config_sha256": self.cfg.document_sha256, "matrix_sha256": self.matrix_sha,
                "applicability_sha256": self.applicability_sha,
                **{f"artifact:{k}": v for k, v in (self.hashes or {}).items()}}

    def _feasibility(self, cell: ExperimentCell, t_start: float) -> dict | None:
        """Mamba wall time vs its GRU twin, measured on this run (task 9.14)."""
        twin = GRU_TWIN.get(cell.model_id)
        if cell.backbone != "mamba" or twin is None:
            return None
        factor = float(self.cfg.raw["feasibility"].get("mamba_wall_time_factor_of_gru_twin", 2.0))
        if self.device.device != "cuda":
            # The budget is a statement about the declared accelerator. On a CPU the
            # python-scan SSM and the cuDNN GRU have completely different ratios, so a
            # gate fired here would report a hardware fact about the laptop as a
            # feasibility verdict about the arm — it is recorded as not measurable.
            return {"triggered": False, "ratio": None, "not_applicable": "wrong_device",
                    "measured_on": self.device.device, "budget_factor": factor,
                    "note": (f"feasibility gate not evaluated: the {factor:.1f}× budget is defined for "
                             f"{self.device.accelerator_expected}, this run is on {self.device.device}; "
                             "wall-time ratios do not transfer between backends")}
        tw = self._wall.get(twin)
        entry = None
        if not tw:
            for c in self.cells:
                if c.model_id == twin and c.suppression == cell.suppression and c.seed == cell.seed:
                    e = self.registry.get(c.cell_hash)
                    if e and e.metrics.get("wall_seconds"):
                        entry = e
                        tw = float(e.metrics["wall_seconds"])
                        break
        if not tw:
            return {"triggered": False, "ratio": None, "not_applicable": "gru_twin_not_yet_run",
                    "note": (f"feasibility gate for {cell.model_id} not measurable yet: no recorded wall time "
                             f"for its GRU twin {twin} (this cell ran on its own); the gate is re-checked "
                             "when both are present"),
                    "budget_factor": factor}
        mine = float(time.time() - t_start)
        ratio = mine / max(float(tw), 1e-9)
        return {"triggered": bool(ratio > factor), "ratio": _round(ratio), "budget_factor": factor,
                "mamba_wall_seconds": _round(mine), "gru_twin_wall_seconds": _round(float(tw)),
                "gru_twin_id": twin, "device": self.device.device, "measured_in_this_run": True,
                "note": (
                    f"feasibility gate triggered: {cell.model_id} took {ratio:.2f}× its GRU twin {twin} "
                    f"(budget {factor:.1f}×) on identical tokens/epochs; recorded Not Applicable with these "
                    "numbers rather than dropped from the comparison"
                    if ratio > factor else
                    f"{cell.model_id} ran in {ratio:.2f}× of its GRU twin, inside the {factor:.1f}× budget")}

    # -- classical references ----------------------------------------------
    def _run_reference(self, cell: ExperimentCell, rdir: Path, notes: list[str]) -> tuple[dict, dict, list[str]]:
        from ssl_anomaly.suppression import classical
        from ssl_anomaly.suppression.classical import assert_transfer, fit_score

        row_app = (self.app.get("pairs") or {}).get(f"{cell.model_id}/{cell.suppression}", {}) or {}
        transfer = assert_transfer(cell.model_id, cell.suppression, row_app.get("reason"))
        if transfer["mode"] == "not_applicable":
            raise RunPlanError(
                f"{cell.experiment_id} is Pending in the matrix but the transfer table says not_applicable "
                f"({transfer['reason']}) — the two authorities disagree and neither may be overridden silently"
            )
        notes.append(f"suppression transfer for {cell.model_id}: {transfer['mode']}")
        kind = classical.fitter_for(cell.model_id)
        stray = classical.unconsumed(cell.model_id, self._selection_for(cell.model_id))
        if stray:
            raise RunPlanError(
                f"{cell.experiment_id}: the selection carries {stray}, which the {kind!r} fitter does "
                f"not read ({sorted(classical.FITTER_CONSUMED.get(kind, frozenset())) or 'no knobs'}); "
                "searching a knob that no code consumes records an arm as tuned while it ran at "
                "defaults — extend the estimator or amend the contract's space, do not drop the key"
            )
        ch = tuple(self.frame.channel_names)
        ev = self.eval_idx
        tr = self.train_idx
        xt = np.asarray(self.frame.x, dtype=np.float32)[tr].reshape(len(tr), -1)
        xe = np.asarray(self.panel.x, dtype=np.float32)[ev].reshape(len(ev), -1)
        contamination = (float(self.cfg.raw["synthetic"]["test_prevalence"])
                         if transfer["mode"] in ("contamination_param", "nu_param") else None)
        t0 = time.time()
        # the searched knobs have to reach the fitter. Hard-coding `n_estimators` here made
        # every BL-* trial identical — the search would then report an arm as "tuned" with
        # zero tuning done, which is the exact failure this module's error messages warn about.
        sel = self._selection_for(cell.model_id)
        ref_params = {"channel_names": ch}
        ref_params.update({k: v for k, v in sel.items() if k != "contamination"})
        # an explicit selection outranks the contract's prevalence-derived default; the *source*
        # is recorded because "which contamination did this baseline use" changes the score
        contam = sel.get("contamination", contamination)
        contam_source = ("selection" if "contamination" in sel else
                         ("contract_test_prevalence" if contamination is not None else "none"))
        _s_fit, rec = fit_score(kind, xt, xt[:8], contamination=contam,
                               params=ref_params, seed=int(cell.seed))
        s_ev, rec2 = fit_score(kind, xt, xe, contamination=contam,
                               params=ref_params, seed=int(cell.seed))
        wall = time.time() - t0
        T = int(self.frame.x.shape[1])
        scores_bin = np.full((self.frame.n, T), np.nan)
        s_ev = np.asarray(s_ev, dtype=np.float64).ravel()
        if s_ev.size != len(ev):
            raise RunPlanError(
                f"reference estimator returned {s_ev.size} scores for {len(ev)} eval cycles; a length "
                "mismatch would silently misalign every label in the metric computation"
            )
        scores_bin[ev] = np.repeat(s_ev[:, None], T, axis=1)
        thr = {"threshold": _round(float(np.nanquantile(s_ev, 0.9))), "budget": 0.1,
               "bin_threshold": _round(float(np.nanquantile(s_ev, 0.9))),
               "note": "a reference rule has no probability; the threshold is the dev quantile at the budget"}
        metrics = self._metrics_for(cell, scores_bin, np.full(self.frame.n, np.nan), thr)
        metrics["level_note"] += "; this estimator scores cycles, not bins"
        cost = {"params": 0, "fit_record": _jsonable(rec2), "train_wall_seconds": _round(wall),
                "contamination_source": contam_source,
                "contamination": _round(contam) if contam is not None else None,
                "fitter": kind, "selection_applied": sel or None,
                "device": "cpu", "why_cpu": self.device.cpu_reason if self.device else None,
                "flops": {"note": "analytic convention is defined for the A-arms only"}}
        manifest = {
            # a fitted reference estimator has no checkpoint to resume; recording that fact is
            # what lets the bundle audit distinguish "nothing to save" from "save was lost"
            "checkpoint": "none",
            "experiment_id": cell.experiment_id, "cell_hash": cell.cell_hash, "seed": cell.seed,
            "factors": _factors(cell), "benchmark_config_sha256": self.cfg.document_sha256,
            "matrix_sha256": self.matrix_sha, "data": self.hashes,
            "objective": {"objective": "not_applicable", "target_source": rec2.get("kind") or "n/a",
                          "gradient_path": "none (no gradient-based objective)",
                          "paradigm": f"reference/{cell.model_id}", "causal": True,
                          "suppression": cell.suppression, "collapse_mechanism": None},
            "suppression": {"transfer": transfer, "applicability_row": _jsonable(row_app)},
            "metrics": metrics, "cost": cost, "notes": notes,
            "stamp": SMOKE_STAMP if self.opts.is_smoke else None,
            "ladder_rung": self.opts.ladder_rung, "code": _bind.git_sha(self.opts.base),
        }
        (rdir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str),
                                                encoding="utf-8")
        (rdir / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True, default=str),
                                           encoding="utf-8")
        (rdir / "progress.jsonl").write_text(json.dumps({"epoch": 0, "backend": rec2.get("kind"),
                                                          "wall_s": _round(wall)}) + "\n", encoding="utf-8")
        (rdir / "norm_stats.json").write_text(json.dumps(
            {"available": False, "reason": "reference estimator exposes no per-channel residual",
             "data_sha256": self.hashes.get("data_sha256")}, indent=1), encoding="utf-8")
        np.savez_compressed(rdir / "bin_scores.npz", bin_scores=scores_bin[ev].astype(np.float32),
                            labels_bin=self.panel.labels_bin[ev].astype(np.uint8),
                            field_ids=np.asarray(self.frame.field_ids)[ev],
                            split=np.asarray(self.frame.split)[ev].astype("U8"))
        return metrics, cost, notes

    def _write_artifacts(self, rdir: Path, cell: ExperimentCell, arm, hist, metrics, cost,
                         collapse_rep, notes, *, cal_fit=None, thr=None, scores_bin=None,
                         chan=None) -> None:
        manifest = {
            # whether a checkpoint exists must be stated, not inferred: a random-init
            # control trains nothing, so it has no `last.pt`, and a filename rule that
            # demands one would turn a design fact into a bundle defect
            "checkpoint": "saved" if (rdir / "last.pt").is_file() else "none",
            "selection_applied": self._selection_for(cell.model_id) or None,
            "experiment_id": cell.experiment_id, "cell_hash": cell.cell_hash, "seed": cell.seed,
            "factors": _factors(cell), "priority": cell.priority, "tier": cell.tier, "block": cell.block,
            "benchmark_config_sha256": self.cfg.document_sha256, "matrix_sha256": self.matrix_sha,
            "applicability_sha256": self.applicability_sha,
            "data": {**self.hashes, "n_cycles": int(self.frame.n),
                     "source": self.frame.data_source, "channel_names": list(self.frame.channel_names),
                     "input": _jsonable(self.bundles[cell.input].to_dict())},
            "synthetic_panel": _jsonable(self.panel.to_dict()),
            "train_contamination": _jsonable(self.contamination.to_dict()),
            "pool_disjointness": self.disjointness,
            "capacity": (self.capacity.get("arms") or {}).get(cell.model_id),
            "objective": _jsonable(arm.objective_spec()),
            "suppression": _jsonable(getattr(arm, "suppression_state", lambda: {})()),
            "architecture": _jsonable(arm.architecture_config()),
            "training": {"epochs": int(getattr(self, "_resumed_from", 0)) + len(hist),
                         "epochs_this_invocation": len(hist),
                         "epochs_planned": int(getattr(self, "_epochs_planned", 0)),
                         "resumed_from": int(getattr(self, "_resumed_from", 0)),
                         "epoch_counting": ("epochs is the cell's total across invocations: a resumed "
                                             "cell has the same budget as one that never crashed, so "
                                             "counting only the last invocation would understate it"),
                         "history_tail": _jsonable((list(getattr(self, "_prior_history", [])) + hist)[-12:]),
                         "early_stop": (self._eff or self.cfg.raw["training"]).get("early_stop"),
                         "optimizer": {"name": (self._eff or self.cfg.raw["training"]).get("optimizer"),
                                       "lr": float((self._eff or self.cfg.raw["training"]).get("lr", 2e-3)),
                                       "weight_decay": float((self._eff or self.cfg.raw["training"])
                                                              .get("weight_decay", 1e-5)),
                                       "betas": [0.9, 0.999], "eps": 1e-8,
                                       "grad_clip_norm": float(self.cfg.raw["training"].get("grad_clip_norm", 1.0))},
                         "objective_audit": self.objective_audit["problems"]},
            "runtime": {"device": self.device.to_dict(), "budget": self.budget.to_dict()},
            "code": _bind.git_sha(self.opts.base), "versions": _bind.software_versions(),
            "calibration": {"method": cell.calibration, "fit": cal_fit or {}, "threshold": _jsonable(thr)},
            "collapse": collapse_rep, "metrics": metrics, "cost": cost, "notes": notes,
            "stamp": SMOKE_STAMP if self.opts.is_smoke else None, "ladder_rung": self.opts.ladder_rung,
        }
        (rdir / "run_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str),
                                                 encoding="utf-8")
        (rdir / "metrics.json").write_text(json.dumps(metrics, indent=1, sort_keys=True, default=str),
                                           encoding="utf-8")
        if collapse_rep:
            (rdir / "collapse.json").write_text(json.dumps(collapse_rep, indent=1, sort_keys=True, default=str),
                                                 encoding="utf-8")
        if scores_bin is not None:
            ev = self.eval_idx
            np.savez_compressed(rdir / "bin_scores.npz",
                                bin_scores=scores_bin[ev].astype(np.float32),
                                channel_error=(np.nan_to_num(chan[ev], nan=0.0).astype(np.float16)
                                               if chan is not None else np.zeros((0,), np.float32)),
                                channel_error_mask=(np.isfinite(chan[ev]).astype(np.uint8)
                                                    if chan is not None else np.zeros((0,), np.uint8)),
                                labels_bin=self.panel.labels_bin[ev].astype(np.uint8),
                                field_ids=np.asarray(self.frame.field_ids)[ev],
                                split=np.asarray(self.frame.split)[ev].astype("U8"))
        missing = _ck.audit_run_dir(rdir, require=("run_manifest.json", "metrics.json", "progress.jsonl",
                                                   "norm_stats.json"))
        if missing["missing"]:
            raise RunPlanError(
                f"{cell.experiment_id} wrote an incomplete run dir (missing {missing['missing']}); the "
                "bundle rule is complete-or-defect, so this cell is not completed"
            )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _cpu_state(arm) -> dict:
    out: dict[str, Any] = {}
    extra = getattr(arm, "extra_checkpoint_state", lambda: {})()
    for k, v in (extra or {}).items():
        out[k] = v
    cs = getattr(arm, "collapse_state", lambda: {})()
    if cs:
        out.setdefault("collapse", cs)
    return out


def _np(t):
    import torch

    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().float().cpu().numpy()
    return np.asarray(t, dtype=np.float64)


def _jsonable(v):
    import torch

    if isinstance(v, torch.Tensor):
        return _np(v).tolist()
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (np.floating, np.integer)):
        v = v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, float) and not math.isfinite(v):
        # ``json.dumps`` happily writes the bare token ``NaN``, which strict readers
        # (jq, JSON.parse, some pandas paths) reject — a manifest that cannot be parsed is
        # a defect regardless of how good the numbers inside it were, so a non-finite
        # value is recorded as null and the count is kept in the note.
        return None
    return v


def _factors(cell: ExperimentCell) -> dict:
    d = {k: v for k, v in vars(cell).items() if k not in ("extra", "priority", "status", "reason")}
    d.update(dict(cell.extra or {}))
    return d


def _nanmax(a, axis=1):
    with np.errstate(invalid="ignore"):
        out = np.nanmax(np.where(np.isfinite(a), a, -np.inf), axis=axis)
    return np.where(np.isneginf(out), np.nan, out)


def _nan_to_zero(a):
    return np.nan_to_num(np.asarray(a, dtype=np.float64), nan=0.0)


def _round(v, nd=6):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _sha(arr) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    h = hashlib.sha256()
    h.update(str(a.shape).encode() + a.dtype.str.encode())
    h.update(a.view(np.uint8).reshape(-1)[: 1 << 22].tobytes())
    return h.hexdigest()


def _sha_bool(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=bool).view(np.uint8)).tobytes()).hexdigest()


def _sha_int(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.int64)).tobytes()).hexdigest()


def _count(xs) -> dict:
    out: dict[str, int] = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))

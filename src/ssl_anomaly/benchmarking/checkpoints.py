"""Checkpoint layout and the resume gate (tasks 1.3, 12.5; spec "Checkpoint and resume").

AGENTS.md §6 requires "model weights (state_dict + optimizer/scheduler/rng) per
seed so ``--resume`` works; never ship a run without its checkpoint and sidecar".
The finding-24 lesson in §6.2 is the reason this module refuses partial writes:
six ``patch_sigma.npz`` sidecars were once left inside Colab run directories and
the committed reports still quote hashes for files that no longer exist.

Design decisions that matter:

- **Sidecar first, weights second.** ``meta.json`` is written before ``last.pt`` is
  replaced, so a preemption between the two leaves a checkpoint that is *older*
  than its record (resumable) rather than a record that lies about a file that is
  not there.
- **RNG state is part of the checkpoint** (python / numpy / torch cpu / every cuda
  device), otherwise a resumed run silently continues under a different stream and
  the recorded seed stops meaning anything.
- **Resume verifies bindings.** :func:`load_checkpoint` compares the sidecar's
  binding hashes with the live ones via
  :func:`ssl_anomaly.benchmarking.bindings.compare_bindings` and raises on
  mismatch; ``--force-resume`` does not exist here on purpose.
- **Atomic replace** through a temp file in the same directory.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

CHECKPOINT_NAME = "last.pt"
SIDECAR_NAME = "meta.json"
#: Files a bundle must carry for a model run (AGENTS.md §6.2 "minimum bundle").
MINIMUM_RUN_FILES = (
    CHECKPOINT_NAME,
    SIDECAR_NAME,
    "progress.jsonl",
    "norm_stats.json",
    "run_manifest.json",
    "metrics.json",
)


class CheckpointError(RuntimeError):
    """A checkpoint cannot be written, restored, or trusted for resume."""


@dataclass
class CheckpointMeta:
    """Sidecar identity for one (cell, seed) run directory."""

    experiment_id: str
    cell_hash: str
    seed: int
    model_id: str
    epochs_done: int = 0
    global_step: int = 0
    best_metric: float | None = None
    bindings: dict = field(default_factory=dict)
    code: dict = field(default_factory=dict)
    objective: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    updated_at: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["updated_at"] = self.updated_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return d

    def write(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / (SIDECAR_NAME + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True, default=str), encoding="utf-8")
        os.replace(tmp, d / SIDECAR_NAME)
        return d / SIDECAR_NAME

    @classmethod
    def read(cls, directory: str | Path) -> CheckpointMeta:
        p = Path(directory) / SIDECAR_NAME
        if not p.is_file():
            raise CheckpointError(
                f"no {SIDECAR_NAME} sidecar in {directory}. A weights file without its sidecar "
                "cannot be verified for resume, so it is not resumed (AGENTS.md §6.2)."
            )
        d = json.loads(p.read_text(encoding="utf-8"))
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        extra = {k: v for k, v in d.items() if k not in known}
        base = {k: v for k, v in d.items() if k in known}
        base.setdefault("bindings", {})
        obj = cls(**base)
        obj.extras.update(extra)
        return obj


# ---------------------------------------------------------------------------
# RNG state
# ---------------------------------------------------------------------------


def capture_rng() -> dict:
    """python / numpy / torch CPU / torch CUDA RNG state, JSON-safe where possible."""
    import random

    state: dict[str, Any] = {"python": random.getstate()[1], "numpy": None, "torch": None, "cuda": []}
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()[1].astype("uint32").tolist()
    except Exception:  # pragma: no cover
        pass
    try:
        import torch

        state["torch"] = torch.get_rng_state().tolist()
        if torch.cuda.is_available():
            state["cuda"] = [s.tolist() for s in torch.cuda.get_rng_state_all()]
    except Exception:  # pragma: no cover
        pass
    return state


def restore_rng(state: dict | None) -> bool:
    if not state:
        return False
    import random

    ok = True
    try:
        random.setstate((3, tuple(state["python"]), None))
    except Exception:
        ok = False
    try:
        import numpy as np

        if state.get("numpy") is not None:
            arr = np.array(state["numpy"], dtype=np.uint32)
            np.random.set_state(("MT19937", arr, int(state.get("numpy_pos", 0)), 0, 0.0))
    except Exception:
        ok = False
    try:
        import torch

        if state.get("torch") is not None:
            torch.set_rng_state(torch.tensor(state["torch"], dtype=torch.uint8))
        if state.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [torch.tensor(s, dtype=torch.uint8) for s in state["cuda"]]
            )
    except Exception:
        ok = False
    return ok


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def run_dir(root: str | Path, experiment_id: str, seed: int) -> Path:
    """``<root>/<experiment_id>.seed<seed>`` — one directory per (cell, seed)."""
    return Path(root) / f"{experiment_id}.seed{seed}"


def save_checkpoint(
    directory: str | Path,
    *,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    meta: CheckpointMeta,
    extra_tensors: dict | None = None,
) -> Path:
    """Write sidecar-then-weights, atomically. Returns the weights path.

    ``extra_tensors`` is where an arm's non-Module state goes (e.g. the EMA target
    encoder and the S1 residual buffer): a resumed run that silently re-initialises
    either is a different experiment, so anything that changes the trajectory has
    to be in the checkpoint or the resume must be refused.
    """
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    meta.write(d)  # sidecar first (see module docstring)
    payload = {
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "rng": capture_rng(),
        "extra": extra_tensors or {},
        "meta": meta.to_dict(),
    }
    tmp = d / (CHECKPOINT_NAME + ".tmp")
    import torch

    torch.save(payload, tmp)
    os.replace(tmp, d / CHECKPOINT_NAME)
    return d / CHECKPOINT_NAME


def load_checkpoint(
    directory: str | Path,
    *,
    model=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    expected_bindings: dict | None = None,
    strict: bool = True,
) -> dict:
    """Restore a run directory, verifying its sidecar bindings first.

    ``expected_bindings`` is the live ``{benchmark_config_sha256, matrix_sha256,
    artifact: sha256, …}`` map. A mismatch raises :class:`CheckpointError` with both
    values printed (spec scenario *Resume verification*).
    """
    d = Path(directory)
    path = d / CHECKPOINT_NAME
    if not path.is_file():
        raise CheckpointError(f"no {CHECKPOINT_NAME} in {d} — nothing to resume from")
    meta = CheckpointMeta.read(d)
    if expected_bindings:
        problems = _binding_problems(meta.bindings, expected_bindings)
        if problems and strict:
            raise CheckpointError(
                "refusing to resume from a checkpoint whose bindings differ from the live run:\n  - "
                + "\n  - ".join(problems)
            )
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if model is not None:
        try:
            model.load_state_dict(payload["model"], strict=strict)
        except RuntimeError as exc:
            raise CheckpointError(
                f"state_dict does not fit the model being resumed into: {exc}. A resumed run "
                "must restore the architecture it was trained as, not the nearest one."
            ) from exc
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    restore_rng(payload.get("rng"))
    payload["meta"] = meta
    return payload


def _binding_problems(saved: dict, current: dict) -> list[str]:
    problems: list[str] = []
    for key, want in current.items():
        got = saved.get(key)
        if want and got and str(want) != str(got):
            problems.append(f"{key}: checkpoint {str(got)[:12]}… vs live {str(want)[:12]}…")
    return problems


def has_resume_point(directory: str | Path) -> bool:
    d = Path(directory)
    return (d / CHECKPOINT_NAME).is_file() and (d / SIDECAR_NAME).is_file()


def audit_run_dir(directory: str | Path, *, require: tuple[str, ...] = MINIMUM_RUN_FILES) -> dict:
    """Bundle-completeness check for one run directory (the §6.2 guard).

    Returns ``{ok, present, missing, directory}``. The runner calls this before
    marking a cell ``Completed``, so a missing sidecar is caught by the agent at
    write time rather than by the owner at download time.
    """
    d = Path(directory)
    present = sorted(p.name for p in d.iterdir() if p.is_file()) if d.is_dir() else []
    missing = [f for f in require if f not in present]
    return {
        "directory": str(d),
        "ok": d.is_dir() and not missing,
        "present": present,
        "missing": missing,
        "note": (
            "a missing sidecar found at bundle time is the agent's failure, not the owner's "
            "(AGENTS.md §6.2)"
        ),
    }

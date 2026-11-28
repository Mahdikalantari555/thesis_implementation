"""Atomic checkpointing, exact-resume state and best.pt eligibility (1.1c, 1.2).

design.md §4.3 / spec `training-convergence-control`:

* ``last.pt`` atomically (tmp + rename) at every evaluation and at least
  every ``ckpt_every_minutes``, and on SIGINT;
* full state: model, optimizer, scheduler, step, epoch, RNG states
  (python/numpy/torch), sampler position, early-stop (monitor) state,
  EMA/target encoder, config hash, data/split hashes, git SHA;
* ``best.pt`` = best pre-registered primary probe metric among ELIGIBLE
  checkpoints (no collapse flag, no objective conflict, every objective
  within ``max_obj_regress`` of its best); ``best_loss.pt`` = lowest total
  loss regardless of eligibility; the eligibility log goes to the manifest.
"""

from __future__ import annotations

import os
import random
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch


# ------------------------------- RNG state ------------------------------------


def collect_rng_state() -> dict[str, Any]:
    """Full RNG state for exact resume (python / numpy / torch)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ---------------------------- atomic writing ----------------------------------


def atomic_torch_save(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a torch checkpoint atomically: tmp file + os.replace.

    A crash mid-write can never leave a half-written ``last.pt`` behind, and
    a failed write leaves no ``.tmp`` residue.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        torch.save(dict(payload), tmp)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
    os.replace(tmp, p)  # atomic on POSIX
    return p


# --------------------------- checkpoint manager -------------------------------


class CheckpointManager:
    """Owns ``last.pt`` atomic writes, the SIGINT hook and time-based saves."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        ckpt_every_minutes: float = 5.0,
        sigint: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.last_path = self.run_dir / "last.pt"
        self.ckpt_every_s = float(ckpt_every_minutes) * 60.0
        self._last_wall_save = time.monotonic()
        self._sigint_count = 0
        self._sigint_installed = False
        self._prev_handler = None
        self._save_fn: Callable[[], dict[str, Any]] | None = None
        if sigint:
            self._install_sigint()

    # -- SIGINT: write last.pt before exit ------------------------------------
    def _install_sigint(self) -> None:
        if self._sigint_installed:
            return

        def handler(signum, frame):
            self._sigint_count += 1
            if self._save_fn is not None:
                try:
                    self.save_last(self._save_fn())
                    print(f"\n[checkpoint] SIGINT: wrote {self.last_path}", flush=True)
                except Exception as exc:  # noqa: BLE001 — must report, never swallow
                    print(f"\n[checkpoint] SIGINT save FAILED: {exc!r}", flush=True)
            raise KeyboardInterrupt

        self._prev_handler = signal.signal(signal.SIGINT, handler)
        self._sigint_installed = True

    def uninstall_sigint(self) -> None:
        if self._sigint_installed and self._prev_handler is not None:
            signal.signal(signal.SIGINT, self._prev_handler)
            self._sigint_installed = False

    # -- saving ----------------------------------------------------------------
    def bind_payload(self, save_fn: Callable[[], dict[str, Any]]) -> None:
        """Register the closure that builds the full checkpoint payload."""
        self._save_fn = save_fn

    def save_last(self, payload: Mapping[str, Any]) -> Path:
        out = atomic_torch_save(payload, self.last_path)
        self._last_wall_save = time.monotonic()
        return out

    def maybe_time_save(self) -> bool:
        """Save ``last.pt`` when ``ckpt_every_minutes`` elapsed since the last."""
        if self._save_fn is None:
            return False
        if time.monotonic() - self._last_wall_save >= self.ckpt_every_s:
            self.save_last(self._save_fn())
            return True
        return False


# ----------------------- eligibility-filtered best ----------------------------


@dataclass
class EligibilityRecord:
    step: int
    eligible: bool
    reason: str
    primary_value: float | None
    total_loss: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "eligible": self.eligible,
            "reason": self.reason,
            "primary_value": self.primary_value,
            "total_loss": self.total_loss,
        }


class BestTracker:
    """Eligibility-filtered ``best.pt`` selection (1.1c).

    ``best.pt`` = best primary probe metric among checkpoints with no collapse
    flag, no objective conflict and every objective within ``max_obj_regress``
    of its best. ``best_loss.pt`` = lowest total loss regardless. Evaluations
    missing the primary metric are recorded as ineligible with the reason —
    never silently replaced by another metric.
    """

    def __init__(self, primary_metric: str, run_dir: str | Path) -> None:
        self.primary_metric = str(primary_metric)
        self.run_dir = Path(run_dir)
        self.best_primary: float | None = None
        self.best_primary_step: int | None = None
        self.best_loss: float | None = None
        self.best_loss_step: int | None = None
        self.eligibility_log: list[EligibilityRecord] = []
        # direction: probe metrics are higher-better; if the primary is a
        # loss-like signal the caller declares it here
        self.higher_is_better: bool = True

    def consider(
        self,
        step: int,
        metrics: Mapping[str, float],
        outcome,  # EvalOutcome from PlateauMonitor
    ) -> dict[str, Any]:
        primary_value = metrics.get(self.primary_metric)
        rec = EligibilityRecord(step=step, eligible=False, reason="", primary_value=None,
                                total_loss=float(metrics.get("total", float("nan"))))
        if outcome.conflict:
            rec.reason = "objective conflict"
        elif outcome.collapsed:
            rec.reason = "representation collapse flag"
        elif primary_value is None or not np.isfinite(primary_value):
            rec.reason = f"primary metric {self.primary_metric!r} absent at this eval"
        else:
            rec.eligible = True
            rec.primary_value = float(primary_value)
            better = (
                (self.best_primary is None)
                or (primary_value > self.best_primary if self.higher_is_better
                    else primary_value < self.best_primary)
            )
            if better:
                self.best_primary = float(primary_value)
                self.best_primary_step = step
        self.eligibility_log.append(rec)

        total = float(metrics.get("total", float("nan")))
        loss_rec: dict[str, Any] | None = None
        if np.isfinite(total):
            if self.best_loss is None or total < self.best_loss:
                self.best_loss = total
                self.best_loss_step = step
                loss_rec = {"save_best_loss": True, "step": step, "total": total}
        return {
            "save_best": rec.eligible and self.best_primary_step == step,
            "save_best_loss": bool(loss_rec and loss_rec["save_best_loss"]),
            "record": rec.to_dict(),
            "best_primary": self.best_primary,
            "best_primary_step": self.best_primary_step,
            "best_loss": self.best_loss,
            "best_loss_step": self.best_loss_step,
        }

    def eligibility_log_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.eligibility_log]

    def state(self) -> dict[str, Any]:
        return {
            "primary_metric": self.primary_metric,
            "best_primary": self.best_primary,
            "best_primary_step": self.best_primary_step,
            "best_loss": self.best_loss,
            "best_loss_step": self.best_loss_step,
            "higher_is_better": self.higher_is_better,
        }

    def load_state(self, st: Mapping[str, Any]) -> None:
        self.primary_metric = str(st["primary_metric"])
        self.best_primary = st.get("best_primary")
        self.best_primary_step = st.get("best_primary_step")
        self.best_loss = st.get("best_loss")
        self.best_loss_step = st.get("best_loss_step")
        self.higher_is_better = bool(st.get("higher_is_better", True))

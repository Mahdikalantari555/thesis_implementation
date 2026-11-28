"""Verbose per-eval logging and machine-readable progress.jsonl (task 1.3).

design.md §4.4: one line per evaluation (step, epoch, train loss and
components, each objective's val loss, total, best, relative improvement,
patience k/P, smoothed slope, LR, effective rank, steps/s, RAM, ETA to cap),
the same fields appended to ``progress.jsonl``, and an end-of-run summary
block with verdict, best step, wall time, params and the resume command when
relevant.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping


class EvalLogger:
    """Writes the per-eval line and the ``progress.jsonl`` record."""

    def __init__(self, run_dir: str | Path, *, filename: str = "progress.jsonl") -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.progress_path = self.run_dir / filename
        self._t0 = time.monotonic()

    def line(self, fields: Mapping[str, Any]) -> str:
        """The human-readable one-line summary of an evaluation."""
        def fmt(v: Any) -> str:
            if isinstance(v, float):
                return f"{v:.4g}" if abs(v) < 1e6 else f"{v:.3e}"
            return str(v)

        keys = [
            "step", "epoch", "train_loss", "train_components", "val_objectives",
            "val_total", "best", "rel_improvement", "patience", "slope", "lr",
            "effective_rank", "steps_per_s", "ram_gb", "eta_min",
        ]
        parts = [f"{k}={fmt(fields[k])}" for k in keys if k in fields]
        extras = [f"{k}={fmt(v)}" for k, v in fields.items()
                  if k not in keys and not isinstance(v, (dict, list))]
        flags = []
        for f in ("conflict_objectives", "plateaued", "improving"):
            if fields.get(f):
                flags.append(f"{f[:-1] if f.endswith('s') else f}={fields[f]}")
        for f in ("objective_conflict", "collapsed", "diverged", "lr_drop_triggered"):
            if fields.get(f):
                flags.append(f)
        if fields.get("verdict"):
            flags.append(f"verdict={fields['verdict']}")
        if fields.get("reason"):
            flags.append(f"reason={fields['reason']}")
        line = " ".join(parts + extras)
        if flags:
            line += "  [" + ", ".join(flags) + "]"
        return line

    def log_eval(self, fields: Mapping[str, Any]) -> None:
        """Print the eval line and append the machine-readable record."""
        record = dict(fields)
        record["wall_s"] = round(time.monotonic() - self._t0, 2)
        print(self.line(record), flush=True)
        with self.progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")

    def summary(self, fields: Mapping[str, Any]) -> None:
        """End-of-run block: verdict, best, wall time, params, resume command."""
        record = dict(fields)
        record["wall_s_total"] = round(time.monotonic() - self._t0, 2)
        print("\n===== RUN SUMMARY =====", flush=True)
        for k, v in record.items():
            print(f"{k}: {v}", flush=True)
        print("=======================\n", flush=True)
        with self.progress_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "summary", **record}, default=str) + "\n")


def resume_command(python: str, script: str, args: Mapping[str, Any]) -> str:
    """The exact ``--resume`` command printed in verdicts (design §4.2)."""
    parts = [python, script]
    for k, v in args.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        else:
            parts.extend([f"--{k}", str(v)])
    parts.append("--resume")
    return " ".join(parts)

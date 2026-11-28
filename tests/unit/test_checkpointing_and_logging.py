"""Tasks 1.1c/1.2/1.3 — atomic checkpoints, exact resume, eligibility, logging."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from ssl_anomaly.train.checkpointing import (
    BestTracker,
    CheckpointManager,
    atomic_torch_save,
    collect_rng_state,
    restore_rng_state,
)
from ssl_anomaly.train.convergence import PlateauMonitor
from ssl_anomaly.train.progress import EvalLogger, resume_command


class TinyModel(nn.Module):
    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin = nn.Linear(4, 3)
        with torch.no_grad():
            self.lin.weight.normal_(generator=g)
            self.lin.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)


def _train_steps(model, opt, x, n_steps: int) -> list[float]:
    losses = []
    model.train()
    for _ in range(n_steps):
        opt.zero_grad()
        loss = ((model(x) - torch.zeros_like(model(x))) ** 2).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


class TestAtomicSave:
    def test_atomic_write_leaves_no_tmp(self, tmp_path):
        p = tmp_path / "last.pt"
        atomic_torch_save({"a": 1}, p)
        assert p.exists()
        assert not (tmp_path / "last.pt.tmp").exists()
        assert torch.load(p, weights_only=False)["a"] == 1

    def test_atomic_save_keeps_old_on_payload_failure(self, tmp_path):
        p = tmp_path / "last.pt"
        atomic_torch_save({"good": 1}, p)
        with pytest.raises((TypeError, AttributeError, Exception)) as ei:
            atomic_torch_save({"bad": (lambda: 0)}, p)  # lambdas are unpicklable
        assert not isinstance(ei.value, KeyboardInterrupt)
        # old file intact
        assert torch.load(p, weights_only=False)["good"] == 1
        leftovers = [f for f in tmp_path.iterdir() if f.suffix == ".tmp"]
        assert leftovers == []


class TestRngState:
    def test_rng_roundtrip_reproduces_draws(self):
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        state = collect_rng_state()
        _ = [random.random() for _ in range(5)]
        _ = np.random.rand(5)
        _ = torch.rand(5)
        restore_rng_state(state)
        a = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        restore_rng_state(state)
        b = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        assert a == b


class TestExactResume:
    def test_resume_matches_uninterrupted_trajectory(self, tmp_path):
        """The spec scenario: interrupted run resumed bit-for-bit."""
        torch.manual_seed(0)
        x = torch.randn(16, 4)

        # ---- uninterrupted run A (seeded identically to run B)
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        model_a = TinyModel(seed=1)
        opt_a = torch.optim.SGD(model_a.parameters(), lr=0.05)
        losses_a = _train_steps(model_a, opt_a, x, 20)

        # ---- interrupted run B: same seed, 8 steps, checkpoint, fresh objects
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        model_b = TinyModel(seed=1)
        opt_b = torch.optim.SGD(model_b.parameters(), lr=0.05)
        losses_b1 = _train_steps(model_b, opt_b, x, 8)
        assert losses_b1 == pytest.approx(losses_a[:8], rel=1e-6, abs=1e-8)
        ckpt = {
            "model": model_b.state_dict(),
            "optimizer": opt_b.state_dict(),
            "rng": collect_rng_state(),
            "step": 8,
        }
        atomic_torch_save(ckpt, tmp_path / "last.pt")

        model_c = TinyModel(seed=99)  # different init — must be overwritten
        opt_c = torch.optim.SGD(model_c.parameters(), lr=0.05)
        loaded = torch.load(tmp_path / "last.pt", weights_only=False)
        model_c.load_state_dict(loaded["model"])
        opt_c.load_state_dict(loaded["optimizer"])
        restore_rng_state(loaded["rng"])
        losses_b2 = _train_steps(model_c, opt_c, x, 12)

        losses_b = losses_b1 + losses_b2
        assert len(losses_b) == len(losses_a)
        for a, b in zip(losses_a, losses_b):
            assert a == pytest.approx(b, rel=1e-6, abs=1e-8)

    def test_time_based_and_sigint_saves(self, tmp_path):
        mgr = CheckpointManager(tmp_path, ckpt_every_minutes=0.0)  # always due
        mgr.bind_payload(lambda: {"step": 1, "rng": collect_rng_state()})
        assert mgr.maybe_time_save() is True
        assert (tmp_path / "last.pt").exists()
        mgr.uninstall_sigint()

    def test_sigint_handler_saves_and_raises(self, tmp_path):
        """Call the installed handler directly: a real os.kill would end pytest."""
        import signal

        mgr = CheckpointManager(tmp_path, sigint=True)
        mgr.bind_payload(lambda: {"step": 7, "rng": collect_rng_state()})
        handler = signal.getsignal(signal.SIGINT)
        assert handler is not signal.SIG_DFL
        with pytest.raises(KeyboardInterrupt):
            handler(signal.SIGINT, None)
        mgr.uninstall_sigint()
        assert (tmp_path / "last.pt").exists()
        assert torch.load(tmp_path / "last.pt", weights_only=False)["step"] == 7


def state_seed():
    """Starter state so runs A and B share the data/eval RNG draws."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    return collect_rng_state()


class TestBestTracker:
    def _outcome(self, conflict=False, collapsed=False):
        from ssl_anomaly.train.convergence import EvalOutcome, Verdict

        o = EvalOutcome(step=0, verdict=Verdict.RUNNING)
        o.conflict = conflict
        o.collapsed = collapsed
        o.eligible_for_best = not (conflict or collapsed)
        return o

    def test_eligibility_filter_keeps_conflict_and_collapsed_out(self, tmp_path):
        t = BestTracker("probe_phase_r2", tmp_path)
        # lowest-loss checkpoint is collapsed -> best_loss.pt yes, best.pt no
        r1 = t.consider(100, {"probe_phase_r2": 0.2, "total": 0.5}, self._outcome(collapsed=True))
        assert r1["save_best"] is False
        assert r1["save_best_loss"] is True
        assert t.eligibility_log[-1].reason == "representation collapse flag"
        # conflicted checkpoint ineligible
        r2 = t.consider(200, {"probe_phase_r2": 0.9, "total": 0.4}, self._outcome(conflict=True))
        assert r2["save_best"] is False
        assert "objective conflict" in t.eligibility_log[-1].reason
        # eligible checkpoint becomes best
        r3 = t.consider(300, {"probe_phase_r2": 0.7, "total": 0.6}, self._outcome())
        assert r3["save_best"] is True
        assert t.best_primary == pytest.approx(0.7)
        assert t.best_loss == pytest.approx(0.4)  # kept from the conflicted eval
        # but NOT re-saved unless better
        r4 = t.consider(400, {"probe_phase_r2": 0.6, "total": 0.6}, self._outcome())
        assert r4["save_best"] is False

    def test_missing_primary_metric_is_ineligible_with_reason(self, tmp_path):
        t = BestTracker("probe_phase_r2", tmp_path)
        r = t.consider(10, {"total": 0.5}, self._outcome())
        assert r["save_best"] is False
        assert "absent" in t.eligibility_log[-1].reason

    def test_eligibility_log_roundtrip(self, tmp_path):
        t = BestTracker("probe_phase_r2", tmp_path)
        t.consider(10, {"probe_phase_r2": 0.3, "total": 1.0}, self._outcome())
        t.consider(20, {"probe_phase_r2": 0.4, "total": 0.9}, self._outcome())
        log = t.eligibility_log_dicts()
        assert len(log) == 2 and log[1]["eligible"] is True
        st = t.state()
        t2 = BestTracker("probe_phase_r2", tmp_path)
        t2.load_state(st)
        assert t2.best_primary == pytest.approx(0.4)
        assert t2.best_primary_step == 20


class TestEvalLogger:
    def test_eval_line_fields_and_progress_jsonl(self, tmp_path, capsys):
        log = EvalLogger(tmp_path)
        fields = {
            "step": 500, "epoch": 2, "train_loss": 0.123, "val_objectives": {"recon": 0.1, "forecast": 0.2},
            "val_total": 0.3, "best": 0.29, "rel_improvement": 0.004, "patience": "3/8",
            "slope": -0.001, "lr": 1e-3, "effective_rank": 4.2, "steps_per_s": 12.5,
            "ram_gb": 1.2, "eta_min": 3.5, "verdict": "RUNNING",
        }
        log.log_eval(fields)
        out = capsys.readouterr().out
        assert "step=500" in out and "effective_rank=4.2" in out and "verdict=RUNNING" in out
        recs = [json.loads(l) for l in (tmp_path / "progress.jsonl").read_text().strip().splitlines()]
        assert recs[0]["step"] == 500
        for k in ("train_loss", "val_objectives", "val_total", "best", "rel_improvement",
                  "patience", "slope", "lr", "effective_rank", "steps_per_s", "ram_gb", "eta_min"):
            assert k in recs[0], k

    def test_summary_block_written(self, tmp_path, capsys):
        log = EvalLogger(tmp_path)
        log.summary({"verdict": "CONVERGED_PLATEAU", "best_step": 800, "params": 12345,
                     "resume_command": "python train.py --resume"})
        out = capsys.readouterr().out
        assert "RUN SUMMARY" in out and "CONVERGED_PLATEAU" in out and "12345" in out
        recs = [json.loads(l) for l in (tmp_path / "progress.jsonl").read_text().strip().splitlines()]
        assert recs[-1]["type"] == "summary" and recs[-1]["best_step"] == 800

    def test_flags_rendered_in_line(self, tmp_path, capsys):
        log = EvalLogger(tmp_path)
        log.log_eval({"step": 1, "val_total": 1.0, "objective_conflict": True,
                      "conflict_objectives": ["jepa"], "reason": "regressed"})
        out = capsys.readouterr().out
        assert "objective_conflict" in out and "jepa" in out

    def test_resume_command_format(self):
        cmd = resume_command("python", "scripts/run_mini.py", {"seed": 13, "max-steps": 400, "verbose": True})
        assert cmd.endswith("--resume")
        assert "--seed 13" in cmd and "--max-steps 400" in cmd and "--verbose" in cmd

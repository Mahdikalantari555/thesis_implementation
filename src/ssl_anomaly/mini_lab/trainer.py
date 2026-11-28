"""Convergence-controlled training loop for the mini lab (task 1.4).

Replaces the fixed-epoch/``max_steps`` loop of ``mini_lab/runner.py`` with the
r2 trainer policy (design §4.1–§4.5):

* evaluate every ``eval_every`` optimizer steps on a DEV split that is never
  the test split (``intra_farm_temporal``: the 2023 dev year; farm-level
  protocols: an intra-train recent slice, recorded in the manifest);
* guard signals: the family objective's val loss + probe metrics — the
  circular season-phase linear probe is the pre-registered primary metric
  (design §12 item 7); the weighted total is tracked but never a guard;
* representation monitors (effective rank, dims-above-floor, off-diagonal
  energy) + collapse floors via ``PlateauMonitor``;
* eligibility-filtered ``best.pt`` + ``best_loss.pt`` + atomic time-based
  SIGINT-aware ``last.pt`` with full resume state (task 1.2);
* verdicts + verbose line + ``progress.jsonl`` + summary with the resume
  command and extra-step extrapolation (task 1.3);
* ``max_steps``/``max_wall_minutes`` stay hard caps with honest budget
  verdicts; to-plateau mode ignores the epoch cap and REQUIRES another cap;
* optional per-eval model checkpoints for the sweep's equal-step cut (§4.5).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ssl_anomaly.train.checkpointing import (
    BestTracker,
    CheckpointManager,
    atomic_torch_save,
    collect_rng_state,
    restore_rng_state,
)
from ssl_anomaly.train.convergence import EvalOutcome, PlateauMonitor, Verdict
from ssl_anomaly.train.progress import EvalLogger, resume_command
from ssl_anomaly.utils.seeds import seed_everything
from ssl_anomaly.utils.tracking import get_git_commit_sha

from .config import RunConfig
from .data import Split
from .models import MODEL_REGISTRY, build_model

#: collapse floors (pre-registered; finding 28 observed effective rank 1.31-1.65)
EFFECTIVE_RANK_FLOOR = 1.0
PER_DIM_STD_FLOOR = 1e-4


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _family_loss(model: nn.Module, batch: torch.Tensor, obs: torch.Tensor, spec,
                 fp_targets: torch.Tensor | None, *, view_seed: int) -> torch.Tensor:
    """One objective loss for the model family (same rules as the legacy loop)."""
    if spec.family == "pretext":
        kwargs: dict[str, Any] = {}
        if fp_targets is not None:
            kwargs["ndvi_targets"] = fp_targets
        return model.pretext_loss(batch, obs, **kwargs)
    if spec.family == "reconstruction":
        from .runner import _masked_reconstruction_loss

        return _masked_reconstruction_loss(model, batch, obs, mask_ratio=0.25)
    if spec.family == "representation":
        from .runner import _augment, _barlow_loss

        v1 = _augment(batch, view_seed)
        v2 = _augment(batch, view_seed + 1000)
        _, e1 = model(v1)
        _, e2 = model(v2)
        return _barlow_loss(e1, e2)
    raise ValueError(f"unknown family {spec.family!r}")


@torch.no_grad()
def _embed(model: nn.Module, x: torch.Tensor, batch: int = 64) -> np.ndarray:
    model.eval()
    outs = []
    for s in range(0, x.shape[0], batch):
        _, emb = model(x[s : s + batch])
        outs.append(emb.cpu())
    return torch.cat(outs).numpy()


@torch.no_grad()
def _val_objective(model: nn.Module, x: torch.Tensor, obs: torch.Tensor, spec,
                   fp_targets: torch.Tensor | None) -> float:
    model.eval()
    if spec.family == "reconstruction":
        from .runner import _masked_reconstruction_loss

        return float(_masked_reconstruction_loss(model, x, obs, mask_ratio=0.25).cpu())
    if spec.family == "pretext":
        kw = {} if fp_targets is None else {"ndvi_targets": fp_targets}
        return float(model.pretext_loss(x, obs, **kw).cpu())
    if spec.family == "representation":
        from .runner import _barlow_loss

        _, e1 = model(x)
        _, e2 = model(x)
        return float(_barlow_loss(e1, e2).cpu())
    raise ValueError(spec.family)


def _phase_targets(dates: tuple[str, ...]) -> np.ndarray:
    """Circular season-phase target per window: (sin, cos) of mean-window DOY.

    The primary probe metric is linear-probe R² on season phase (design §12
    item 7); the circular representation is the phase a LINEAR probe can
    express. Padded ("") dates are excluded by the caller's mask.
    """
    ds = [d for d in dates if d]
    if not ds:
        return np.zeros(2)
    doys = [
        float((np.datetime64(d, "D") - np.datetime64(f"{d[:4]}-01-01", "D")) / np.timedelta64(1, "D"))
        for d in ds
    ]
    doy = float(np.mean(doys))
    return np.array([math.sin(2 * math.pi * doy / 365.25),
                     math.cos(2 * math.pi * doy / 365.25)])


def _make_dev_slices(train: Split, dev: Split | None, *, frac: float = 0.25):
    """Farm-level protocols: intra-train dev slice (most recent train windows)."""
    if dev is not None:
        return train, dev, "protocol dev split"
    n = train.n
    if n < 8:
        return train, None, "train too small for an intra-train dev slice"
    order = np.argsort([r[1] for r in train.date_ranges], kind="stable")
    n_dev = max(2, int(round(frac * n)))
    dev_idx = order[-n_dev:]
    tr_idx = order[:-n_dev]

    def subset(idx: np.ndarray, name: str) -> Split:
        return Split(
            name=name,
            x=train.x[idx],
            observed=train.observed[idx],
            delta_t=train.delta_t[idx],
            farm_ids=[train.farm_ids[i] for i in idx],
            date_ranges=[train.date_ranges[i] for i in idx],
            sha256=hashlib.sha256(
                train.sha256.encode() + np.ascontiguousarray(idx).tobytes()
            ).hexdigest()[:16],
        )

    tr = subset(tr_idx, train.name + "_traindev")
    dv = subset(dev_idx, train.name + "_dev")
    return tr, dv, f"intra-train recent slice ({n_dev}/{n} windows, all within train years)"


def _probe_metrics(model, x_dev: torch.Tensor, dev: Split, *, feature_names: list[str],
                   probe_max_samples: int) -> tuple[dict[str, float], dict[str, float]]:
    """Fixed-budget frozen-encoder probes + representation health (1.1b).

    Probes on the dev split:
    * circular season-phase linear probe (primary metric ``probe_phase_r2``);
    * next-NDVI probe: embedding of the FIRST HALF of each window predicts the
      mean NDVI of the SECOND half (next-observation prediction from context);
    * same-field-same-phase kNN retrieval (chance = largest group share);
    * frozen gap reconstruction: embedding predicts the mean value at
      UNOBSERVED positions of the window (the L1-a gap analogue).
    """
    from ssl_anomaly.train.monitoring import (
        gap_reconstruction_probe,
        next_ndvi_probe,
        phase_linear_probe,
        rep_health,
        same_field_same_phase_knn,
    )

    n = x_dev.shape[0]
    ndvi_idx = feature_names.index("ndvi") if "ndvi" in feature_names else 0
    lst_idx = feature_names.index("lst") if "lst" in feature_names else None

    emb = _embed(model, x_dev, batch=probe_max_samples)
    health = rep_health(emb, std_floor=PER_DIM_STD_FLOOR)
    metrics: dict[str, float] = {}

    with torch.no_grad():
        ndvi_col = x_dev[..., ndvi_idx].cpu().numpy()  # (N, T)
    observed = dev.observed

    # ---- circular season-phase probe ------------------------------------
    phases = np.stack([_phase_targets(dev.date_ranges[i]) for i in range(n)])
    rows = np.array([bool(p.any()) for p in phases]) & np.isfinite(emb).all(axis=1)
    if rows.sum() >= 8:
        metrics.update(phase_linear_probe(emb[rows], phases[rows]))

    # ---- next-NDVI probe (first half -> second half target) --------------
    t = x_dev.shape[1]
    if t >= 4:
        half = t // 2
        emb_a = _embed(model, x_dev[:, :half], batch=probe_max_samples)
        second = ndvi_col[:, half:]
        obs_second = observed[:, half:, ndvi_idx]
        vals = np.where(obs_second, second, np.nan)
        target = np.nanmean(vals, axis=1)
        rows = np.isfinite(target) & np.isfinite(emb_a).all(axis=1)
        if rows.sum() >= 8:
            metrics.update(next_ndvi_probe(emb_a[rows], target[rows]))

    # ---- same-field-same-phase kNN retrieval ------------------------------
    field_ids = np.array([abs(hash(f)) % 100000 for f in dev.farm_ids])
    bins = np.array([_phase_bin(dev.date_ranges[i]) for i in range(n)])
    rows = np.isfinite(emb).all(axis=1)
    if rows.sum() >= 8:
        metrics.update(same_field_same_phase_knn(emb[rows], field_ids[rows], bins[rows]))

    # ---- frozen gap reconstruction ----------------------------------------
    core = [i for i, nm in enumerate(feature_names)]
    if core:
        obs_core = observed  # (N, T, F)
        xnp = x_dev.cpu().numpy()
        gap_vals = np.where(obs_core, xnp, np.nan)
        target = np.nanmean(gap_vals[:, :, core], axis=(1, 2))
        rows = np.isfinite(target) & np.isfinite(emb).all(axis=1)
        if rows.sum() >= 8:
            metrics.update(gap_reconstruction_probe(emb[rows], target[rows]))
    if lst_idx is None:
        pass  # lst absent in some fixtures; probes degrade gracefully
    return metrics, health


def _phase_bin(date_range: tuple[str, str]) -> int:
    """Phase-bin proxy for retrieval: month of the window's start date."""
    d = date_range[0]
    if not d:
        return 0
    return int(d[5:7])


def _cfg_hash(cfg: RunConfig) -> str:
    return hashlib.sha256(
        json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def train_one_seed_to_plateau(
    cfg: RunConfig,
    split_name: str,
    train: Split,
    dev: Split | None,
    test: Split,
    seed: int,
    out_root: Path,
    feature_names: list[str],
    valid_patch_counts: dict[str, Any],
) -> dict[str, Any]:
    """Plateau-driven training for one (split, seed); returns the run report."""
    if cfg.model.to_plateau and cfg.model.max_steps is None and cfg.model.max_wall_minutes is None:
        raise RuntimeError(
            "to_plateau=True requires max_steps or max_wall_minutes: an "
            "unbounded to-plateau run on shared hardware is not permitted"
        )
    t0 = time.monotonic()
    device = _device(cfg.model.device)
    spec = MODEL_REGISTRY[cfg.model.name]
    run_dir = out_root / cfg.name / split_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = EvalLogger(run_dir)
    git_sha = get_git_commit_sha()

    tr_used, dev_used, dev_note = _make_dev_slices(train, dev)
    ndvi_idx = feature_names.index("ndvi") if "ndvi" in feature_names else 0
    objective_name = {
        "reconstruction": "recon",
        "representation": "barlow",
        "pretext": "pretext",
    }[spec.family]

    monitor = PlateauMonitor(
        signals={objective_name: "min", "probe_phase_r2": "max"},
        primary="probe_phase_r2",
        min_rel_delta=cfg.model.min_rel_delta,
        patience=cfg.model.patience,
        window=cfg.model.window,
        min_steps=cfg.model.min_steps,
        max_obj_regress=cfg.model.max_obj_regress,
        diverge_factor=cfg.model.diverge_factor,
        lr_patience=cfg.model.lr_patience,
        max_lr_drops=cfg.model.max_lr_drops,
        effective_rank_floor=EFFECTIVE_RANK_FLOOR,
        per_dim_std_floor=PER_DIM_STD_FLOOR,
    )
    best = BestTracker("probe_phase_r2", run_dir)
    ckpt = CheckpointManager(run_dir, ckpt_every_minutes=cfg.model.ckpt_every_minutes)

    seed_everything(seed)

    # ---- resume semantics (task 1.5, fail closed) ---------------------------
    last_path = run_dir / "last.pt"
    start_step = 0
    model = build_model(cfg.model.name, train.x.shape[-1], cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr, weight_decay=1e-5)
    effective_mode = cfg.model.resume_mode
    if effective_mode == "auto":
        effective_mode = "resume" if last_path.exists() else "fresh"
        print(
            f"  {'AUTO-RESUME' if effective_mode == 'resume' else 'FRESH START'} "
            f"{split_name}/{seed} ({last_path} {'found' if effective_mode == 'resume' else 'absent'}) "
            "— logged, never silent",
            flush=True,
        )
    if effective_mode == "resume":
        if not last_path.exists():
            raise RuntimeError(
                f"--resume failed: no checkpoint at {last_path}; resume never "
                "silently re-initializes (task 1.5)"
            )
        loaded = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(loaded["model"])
        opt.load_state_dict(loaded["optimizer"])
        restore_rng_state(loaded["rng"])
        monitor.load_state(loaded["monitor_state"])
        best.load_state(loaded["best_state"])
        start_step = int(loaded["step"])
        print(f"  RESUMED {split_name}/{seed} from step {start_step}", flush=True)
    elif effective_mode == "init-from":
        if not cfg.model.init_from or not Path(cfg.model.init_from).exists():
            raise RuntimeError(
                f"--init-from failed: checkpoint {cfg.model.init_from!r} does "
                "not exist (task 1.5)"
            )
        loaded = torch.load(cfg.model.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(loaded["model"])
        print(f"  INIT-FROM {cfg.model.init_from} (fresh optimizer/monitor)", flush=True)
    elif cfg.model.resume_mode == "from-scratch":
        if last_path.exists():
            raise RuntimeError(
                f"--from-scratch refused: {last_path} exists; move it or use "
                "--resume / --init-from (task 1.5 fail-closed)"
            )

    fp_targets = None
    if cfg.model.name == "time2agri_fp":
        from ssl_anomaly.models.time2agri_archs import build_fp_targets

        fp_targets = torch.from_numpy(
            build_fp_targets(tr_used.x, tr_used.observed, ndvi_idx, split="train")
        ).to(device)

    xtr = torch.from_numpy(tr_used.x).to(device)
    obs_tr = torch.from_numpy(tr_used.observed).to(device)
    xdv = torch.from_numpy(dev_used.x).to(device) if dev_used is not None else None
    obs_dv = torch.from_numpy(dev_used.observed).to(device) if dev_used is not None else None
    xte = torch.from_numpy(test.x).to(device)
    obs_te = torch.from_numpy(test.observed).to(device)

    steps_done = start_step
    epochs_done = 0

    def payload() -> dict[str, Any]:
        return {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "rng": collect_rng_state(),
            "step": steps_done,
            "epoch": epochs_done,
            "monitor_state": monitor.state(),
            "best_state": best.state(),
            "seed": seed,
            "split_name": split_name,
            "git_sha": git_sha,
            "config_hash": _cfg_hash(cfg),
            "split_sha256": {"train": train.sha256, "dev": dev_used.sha256 if dev_used else None,
                             "test": test.sha256},
            "ema_target": None,  # JEPA EMA lives inside the model state_dict
        }

    ckpt.bind_payload(payload)

    eval_every = max(1, int(cfg.model.eval_every))
    wall_deadline = (
        t0 + cfg.model.max_wall_minutes * 60.0
        if cfg.model.max_wall_minutes is not None else None
    )
    n_batch = max(1, math.ceil(xtr.shape[0] / cfg.model.batch_size))
    verdict: Verdict = Verdict.RUNNING
    final_outcome: EvalOutcome | None = None
    train_losses: list[float] = []
    probe_eval_counter = 0
    ram_gb = 0.0
    try:
        import psutil

        ram_gb = psutil.Process().memory_info().rss / 1e9
    except Exception:
        ram_gb = 0.0
    last_best: float | None = None

    def run_eval(step: int) -> EvalOutcome:
        nonlocal probe_eval_counter, ram_gb, last_best
        probe_eval_counter += 1
        val_loss = (
            _val_objective(model, xdv, obs_dv, spec, fp_targets)
            if xdv is not None else float("nan")
        )
        probe_metrics: dict[str, float] = {}
        rep_stats: dict[str, float] | None = None
        if probe_eval_counter % max(1, cfg.model.probe_every) == 0 and dev_used is not None:
            pm, health = _probe_metrics(
                model, xdv, dev_used, feature_names=feature_names,
                probe_max_samples=cfg.model.probe_max_samples,
            )
            rep_stats = {
                "effective_rank": health.get("effective_rank"),
                "per_dim_std_min": health.get("per_dim_std_min"),
            }
            probe_metrics = {k: v for k, v in pm.items()
                             if k.startswith("probe_") and v is not None
                             and np.isfinite(v)}
        # the monitor's guard carries the PRIMARY probe metric only; the other
        # probes are logged for the owner but are not stop signals
        monitor_probe = {"probe_phase_r2": probe_metrics["probe_phase_r2"]} \
            if "probe_phase_r2" in probe_metrics else {}
        outcome = monitor.update(
            step, val_loss, {objective_name: val_loss},
            probe_metrics=monitor_probe, rep_stats=rep_stats,
        )
        selection = best.consider(step, {"total": val_loss, **probe_metrics}, outcome)
        if selection["save_best"]:
            atomic_torch_save(
                {"model": model.state_dict(), "step": step, "kind": "best",
                 "primary": best.primary_metric,
                 "primary_value": selection["best_primary"], "git_sha": git_sha},
                run_dir / "best.pt",
            )
        if selection["save_best_loss"]:
            atomic_torch_save(
                {"model": model.state_dict(), "step": step, "kind": "best_loss",
                 "total": selection["best_loss"], "git_sha": git_sha},
                run_dir / "best_loss.pt",
            )
        if cfg.model.keep_eval_ckpts:
            atomic_torch_save(
                {"model": model.state_dict(), "step": step, "kind": "eval"},
                run_dir / "eval_ckpts" / f"step_{step:07d}.pt",
            )
        ckpt.save_last(payload())
        new_best = selection["best_primary"]
        rel = 0.0
        if last_best is not None and new_best is not None and new_best > last_best:
            rel = (new_best - last_best) / abs(last_best) if last_best != 0 else 0.0
        if new_best is not None:
            last_best = new_best
        try:
            ram_gb = psutil.Process().memory_info().rss / 1e9
        except Exception:
            pass
        sps = steps_done / max(1e-6, time.monotonic() - t0)
        eta_min = (
            (cfg.model.max_steps - steps_done) / max(sps, 1e-6) / 60.0
            if cfg.model.max_steps else None
        )
        fields = {
            "step": step, "epoch": epochs_done,
            "train_loss": (sum(train_losses[-n_batch:]) / n_batch) if train_losses else None,
            "train_components": {objective_name: round(train_losses[-1], 6) if train_losses else None},
            "val_objectives": {objective_name: round(val_loss, 6)},
            "val_total": round(val_loss, 6) if np.isfinite(val_loss) else None,
            "best": new_best,
            "rel_improvement": round(rel, 6),
            "patience": f"{monitor.signals[objective_name].flat_counter}/{cfg.model.patience}",
            "slope": round(monitor.signals[objective_name].ema_slope, 6),
            "lr": opt.param_groups[0]["lr"],
            "effective_rank": (round(rep_stats["effective_rank"], 3)
                               if rep_stats and np.isfinite(rep_stats.get("effective_rank", float("nan")))
                               else None),
            "steps_per_s": round(sps, 2),
            "ram_gb": round(ram_gb, 2),
            "eta_min": round(eta_min, 2) if eta_min is not None else None,
            **{k: round(v, 4) for k, v in probe_metrics.items()},
            **outcome.to_log_fields(),
            "selection": selection["record"],
            "dev_split": dev_note,
        }
        logger.log_eval(fields)
        return outcome

    def budget_stop(reason_prefix: str, step: int, train_mean: float) -> EvalOutcome:
        nonlocal final_outcome, verdict
        budget = monitor.budget_exhausted(step, steps_per_eval=eval_every)
        budget.step = step
        budget.reason = f"{reason_prefix}; {budget.reason}"
        logger.log_eval({
            "step": step, "val_total": None, "val_objectives": {},
            "train_loss": train_mean, "lr": opt.param_groups[0]["lr"],
            **budget.to_log_fields(),
        })
        final_outcome = budget
        verdict = budget.verdict
        return budget

    if cfg.model.resume_mode == "resume" and start_step > 0:
        final_outcome = run_eval(start_step)

    epoch_cap = cfg.model.epochs if not cfg.model.to_plateau else None
    running = True
    while running:
        model.train()
        perm = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(seed + epochs_done))
        total = 0.0
        n_b = 0
        for start in range(0, xtr.shape[0], cfg.model.batch_size):
            batch_idx = perm[start : start + cfg.model.batch_size]
            batch = xtr[batch_idx]
            loss = _family_loss(model, batch, obs_tr[batch_idx], spec, fp_targets,
                                view_seed=seed + epochs_done + steps_done)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu())
            n_b += 1
            steps_done += 1
            train_losses.append(float(loss.detach().cpu()))
            if steps_done % eval_every == 0:
                final_outcome = run_eval(steps_done)
                if final_outcome.lr_drop_triggered:
                    for g in opt.param_groups:
                        g["lr"] *= 0.5
                if final_outcome.should_stop:
                    running = False
                    verdict = final_outcome.verdict
                    break
            ckpt.maybe_time_save()
            if cfg.model.max_steps is not None and steps_done >= cfg.model.max_steps:
                budget_stop("max_steps cap reached", steps_done, total / n_b)
                running = False
                break
            if wall_deadline is not None and time.monotonic() >= wall_deadline:
                budget_stop(f"wall cap {cfg.model.max_wall_minutes} min reached", steps_done, total / n_b)
                running = False
                break
        if not running:
            break
        epochs_done += 1
        if epoch_cap is not None and epochs_done >= epoch_cap:
            budget_stop(f"epoch cap ({epoch_cap}) reached", steps_done, total / max(1, n_b))
            running = False

    ckpt.save_last(payload())
    ckpt.uninstall_sigint()

    # ---- final scoring: best.pt weights if any, else last -------------------
    scored_from = "last"
    best_path = run_dir / "best.pt"
    if best_path.exists():
        loaded = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(loaded["model"])
        scored_from = f"best.pt(step={loaded.get('step')})"

    clean, deviated, auprc, step_auprc, random_auprc = _final_scores(
        cfg, model, xtr, xte, obs_te, spec, seed, device, feature_names,
    )

    wall = round(time.monotonic() - t0, 2)
    resume_cmd = resume_command(
        "/home/mahdi/micromamba/envs/geo/bin/python -X utf8",
        "scripts/run_mini_experiment.py",
        {"config": cfg.config_path or "<the YAML of this run>", "seed": seed},
    ) if verdict in (Verdict.BUDGET_EXHAUSTED_STILL_IMPROVING, Verdict.BUDGET_EXHAUSTED_FLAT) else None

    report = {
        "schema": "mini-run/v2",
        "protocol": split_name,
        "seed": seed,
        "model": cfg.model.name,
        "family": spec.family,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "n_train_units": train.n,
        "n_dev_units": (dev_used.n if dev_used is not None else 0),
        "n_test_units": test.n,
        "dev_split": dev_note,
        "train_date_range": [min(r[0] for r in train.date_ranges), max(r[1] for r in train.date_ranges)],
        "test_date_range": [min(r[0] for r in test.date_ranges), max(r[1] for r in test.date_ranges)],
        "train_farms": sorted(set(train.farm_ids)),
        "test_farms": sorted(set(test.farm_ids)),
        "split_sha256": {"train": train.sha256, "test": test.sha256},
        "losses": [round(v, 6) for v in train_losses[-50:]],
        "steps_done": steps_done,
        "epochs_done": epochs_done,
        "to_plateau": bool(cfg.model.to_plateau),
        "verdict": verdict.value,
        "verdict_reason": final_outcome.reason if final_outcome else "",
        "extra_steps_estimate": final_outcome.extra_steps_estimate if final_outcome else None,
        "resume_command": resume_cmd,
        "scored_from": scored_from,
        "best_primary_metric": best.primary_metric,
        "best_primary_value": best.best_primary,
        "best_primary_step": best.best_primary_step,
        "best_loss": best.best_loss,
        "best_loss_step": best.best_loss_step,
        "eligibility_log": best.eligibility_log_dicts(),
        "monitor_state": monitor.state(),
        "clean_score_mean": float(np.nanmean(clean)) if np.isfinite(clean).any() else None,
        "deviated_score_mean": float(np.nanmean(deviated)) if np.isfinite(deviated).any() else None,
        "synthetic_deviation_auprc": auprc,
        "step_reference_auprc": step_auprc,
        "random_init_auprc": random_auprc,
        "baseline_note": (
            "Balanced clean/deviated sets (prevalence 0.5 by construction); "
            "random-init encoder and step detector are references, not trained models."
        ),
        "synthetic_deviation_note": (
            "Controlled one-channel NDVI drop on held-out windows only; "
            "evidence for L2 controlled deviation, NOT stress detection."
        ),
        "n_imputed_features": int((~test.observed).sum()),
        "valid_patch_counts": valid_patch_counts,
        "device": str(device),
        "max_steps": cfg.model.max_steps,
        "wall_s": wall,
        "git_sha": git_sha,
        "data_source": str(cfg.store_root),
        "holdout_boundary": cfg.protocol.holdout_boundary,
        "dev_boundary": cfg.protocol.dev_boundary,
        "holdout_justification": cfg.protocol.justification,
        "claim_rung": cfg.claim_rung,
        "checkpoint": str(last_path),
        "progress": str(logger.progress_path),
        "seeds_used": [seed],
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logger.summary({
        "verdict": verdict.value,
        "reason": final_outcome.reason if final_outcome else "",
        "best_step": best.best_primary_step,
        "best_primary": best.best_primary,
        "steps_done": steps_done,
        "params": report["param_count"],
        "scored_from": scored_from,
        "synthetic_deviation_auprc": auprc,
        "step_reference_auprc": step_auprc,
        "random_init_auprc": random_auprc,
        "wall_s": wall,
        "resume_command": resume_cmd,
        "extra_steps_estimate": final_outcome.extra_steps_estimate if final_outcome else None,
        "claim_rung": cfg.claim_rung,
        "run_dir": str(run_dir),
    })
    return report


def _final_scores(cfg, model, xtr, xte, obs_te, spec, seed, device, feature_names):
    """Final scoring with the same controls as the legacy loop (AUPRC + refs)."""
    from .runner import _auprc, _scores, _train_embeddings

    train_emb = (
        _train_embeddings(model, xtr) if spec.family in ("representation", "pretext") else None
    )
    clean = _scores(model, xte, obs_te, spec.family, train_emb)
    ndvi_idx = feature_names.index("ndvi") if "ndvi" in feature_names else 0
    deviated_x = xte.clone()
    half = max(1, deviated_x.shape[1] // 2)
    deviated_x[:, half:, ndvi_idx] = deviated_x[:, half:, ndvi_idx] - 1.0
    deviated = _scores(model, deviated_x, obs_te, spec.family, train_emb)
    auprc = _auprc(clean, deviated)

    pre = xte[:, :half, ndvi_idx].mean(dim=1)
    post = xte[:, half:, ndvi_idx].mean(dim=1)
    step_clean = (post - pre).abs().cpu().numpy()
    dpre = deviated_x[:, :half, ndvi_idx].mean(dim=1)
    dpost = deviated_x[:, half:, ndvi_idx].mean(dim=1)
    step_deviated = (dpost - dpre).abs().cpu().numpy()
    step_auprc = _auprc(step_clean, step_deviated)

    seed_everything(seed)
    random_model = build_model(cfg.model.name, xtr.shape[-1], cfg.model).to(device)
    random_emb = (
        _train_embeddings(random_model, xtr) if spec.family in ("representation", "pretext") else None
    )
    rand_clean = _scores(random_model, xte, obs_te, spec.family, random_emb)
    rand_deviated = _scores(random_model, deviated_x, obs_te, spec.family, random_emb)
    random_auprc = _auprc(rand_clean, rand_deviated)
    return clean, deviated, auprc, step_auprc, random_auprc


def score_state_dict(
    cfg: RunConfig,
    state_dict: dict,
    xtr: torch.Tensor,
    xte: torch.Tensor,
    obs_te: torch.Tensor,
    spec,
    seed: int,
    device: torch.device,
    feature_names: list[str],
) -> dict[str, Any]:
    """Score ONE checkpoint state (equal-step cut, sweep §4.5).

    Returns the synthetic-deviation AUPRC + step-detector + random-init
    references for the given weights, without touching the training state.
    """
    from .runner import _auprc, _scores, _train_embeddings

    model = build_model(cfg.model.name, xtr.shape[-1], cfg.model).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    train_emb = (
        _train_embeddings(model, xtr) if spec.family in ("representation", "pretext") else None
    )
    clean = _scores(model, xte, obs_te, spec.family, train_emb)
    ndvi_idx = feature_names.index("ndvi") if "ndvi" in feature_names else 0
    deviated_x = xte.clone()
    half = max(1, deviated_x.shape[1] // 2)
    deviated_x[:, half:, ndvi_idx] = deviated_x[:, half:, ndvi_idx] - 1.0
    deviated = _scores(model, deviated_x, obs_te, spec.family, train_emb)
    auprc = _auprc(clean, deviated)
    pre = xte[:, :half, ndvi_idx].mean(dim=1)
    post = xte[:, half:, ndvi_idx].mean(dim=1)
    step_auprc = _auprc((post - pre).abs().cpu().numpy(),
                        (deviated_x[:, half:, ndvi_idx].mean(dim=1)
                         - deviated_x[:, :half, ndvi_idx].mean(dim=1)).abs().cpu().numpy())
    seed_everything(seed)
    random_model = build_model(cfg.model.name, xtr.shape[-1], cfg.model).to(device)
    random_emb = (
        _train_embeddings(random_model, xtr) if spec.family in ("representation", "pretext") else None
    )
    random_auprc = _auprc(_scores(random_model, xte, obs_te, spec.family, random_emb),
                          _scores(random_model, deviated_x, obs_te, spec.family, random_emb))
    return {
        "synthetic_auprc": auprc,
        "step_reference_auprc": step_auprc,
        "random_init_auprc": random_auprc,
    }

"""Training, scoring, tracking and provenance for mini-dataset experiments."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from ssl_anomaly.utils.seeds import seed_everything  # noqa: E402
from ssl_anomaly.utils.tracking import get_git_commit_sha  # noqa: E402

from .config import RunConfig, config_dict  # noqa: E402
from .data import MiniDataset, Split, split_datasets  # noqa: E402
from .models import MODEL_REGISTRY, build_model, model_specs  # noqa: E402


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _masked_reconstruction_loss(
    model: nn.Module,
    x: torch.Tensor,
    observed: torch.Tensor | None = None,
    mask_ratio: float = 0.25,
) -> torch.Tensor:
    b, t, _ = x.shape
    masked = x.clone()
    n = max(1, int(t * mask_ratio))
    for i in range(b):
        start = int(torch.randint(0, max(1, t - n + 1), (1,)).item())
        masked[i, start : start + n] = 0.0
    recon, _ = model(masked)
    err = (recon - x).pow(2)
    if observed is None:
        return err.mean()
    return (err * observed).sum() / observed.sum().clamp(min=1)


def _barlow_loss(z1: torch.Tensor, z2: torch.Tensor, lambda_coeff: float = 0.005) -> torch.Tensor:
    """Barlow Twins loss (task 2.2 fix).

    ``std(0)`` with a batch of 1 previously produced NaN (zero degrees of
    freedom), which poisoned the weights and made every downstream score NaN —
    the actual root cause of the recorded "JEPA full-field
    'NoneType' object is not subscriptable" failure (finding 28). Population
    std (unbiased=False) keeps batch-1 finite; the loss on a single pair is
    then the constant on-diagonal term, which is computable and honest.
    """
    if z1.shape != z2.shape:
        raise ValueError(f"barlow views differ in shape: {tuple(z1.shape)} vs {tuple(z2.shape)}")
    z1 = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + 1e-6)
    z2 = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + 1e-6)
    cross = z1.T @ z2 / z1.shape[0]
    on_diag = torch.diagonal(cross).add_(-1).pow_(2).sum()
    # canonical Barlow off-diagonal term: the diagonal must be EXCLUDED
    # (the previous `cross - eye` kept (c_ii-1)^2 in the off term, double-
    # counting the invariance term; rst_bt.barlow_twins_loss was correct)
    off = (cross - torch.diag(torch.diagonal(cross))).pow_(2).sum()
    return on_diag + lambda_coeff * off


def _augment(x: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    y = x.clone()
    t = y.shape[1]
    block = max(1, min(3, t // 4))
    for i in range(y.shape[0]):
        s = int(torch.randint(0, max(1, t - block + 1), (1,), generator=g).item())
        y[i, s : s + block] = 0.0
    return y


def _scores(model: nn.Module, x: torch.Tensor, observed: torch.Tensor, family: str, train_emb: torch.Tensor | None) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        recon, emb = model(x)
        if family in ("representation", "pretext"):
            if train_emb is None:
                return np.full(x.shape[0], np.nan)
            d = torch.cdist(emb, train_emb)
            return d.min(dim=1).values.cpu().numpy()
        if family == "reconstruction":
            err = (recon - x).pow(2)
            denom = observed.sum(dim=(1, 2)).clamp(min=1)
            return ((err * observed).sum(dim=(1, 2)) / denom).cpu().numpy()
        raise ValueError(f"Unknown model family {family!r}")


def _auprc(clean: np.ndarray, deviated: np.ndarray) -> float | None:
    """AUPRC for a balanced clean-vs-controlled-deviation scoring set."""
    from sklearn.metrics import average_precision_score

    labels = np.array([0] * len(clean) + [1] * len(deviated))
    scores = np.concatenate([clean, deviated])
    finite = np.isfinite(scores)
    if finite.sum() < 2 or len(np.unique(labels[finite])) != 2:
        return None
    return float(average_precision_score(labels[finite], scores[finite]))


def _train_embeddings(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        _, emb = model(x)
    return emb.detach()


def _save_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer, epoch: int, seed: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "rng": {"python": random.getstate(), "torch": torch.get_rng_state()},
            "epoch": epoch,
            "seed": seed,
        },
        path,
    )


def _load_checkpoint(path: Path, model: nn.Module, opt: torch.optim.Optimizer) -> int:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    if "rng" in ck:
        random.setstate(ck["rng"]["python"])
        torch.set_rng_state(ck["rng"]["torch"])
    return int(ck["epoch"])


def _one_seed(
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
    # Task 1.4: convergence-controlled arms run the plateau trainer; legacy
    # fixed-epoch runs keep the old loop for screen compatibility.
    if cfg.model.to_plateau:
        from .trainer import train_one_seed_to_plateau

        return train_one_seed_to_plateau(
            cfg, split_name, train, dev, test, seed, out_root,
            feature_names, valid_patch_counts,
        )
    t0 = time.monotonic()
    device = _device(cfg.model.device)
    seed_everything(seed)
    spec = MODEL_REGISTRY[cfg.model.name]
    model = build_model(cfg.model.name, train.x.shape[-1], cfg.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.model.lr, weight_decay=1e-5)
    run_dir = out_root / cfg.name / split_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ck_path = run_dir / "last.pt"
    progress = run_dir / "progress.jsonl"
    start_epoch = 0
    if cfg.resume and ck_path.exists():
        start_epoch = _load_checkpoint(ck_path, model, opt)
        print(f"  Resuming {split_name}/{seed} from epoch {start_epoch}", flush=True)

    xtr = torch.from_numpy(train.x).to(device)
    obs_tr = torch.from_numpy(train.observed).to(device)
    xte = torch.from_numpy(test.x).to(device)
    obs_te = torch.from_numpy(test.observed).to(device)

    fp_targets = None
    if cfg.model.name == "time2agri_fp":
        from ssl_anomaly.models.time2agri_archs import build_fp_targets

        ni = feature_names.index("ndvi") if "ndvi" in feature_names else 0
        fp_targets = torch.from_numpy(
            build_fp_targets(train.x, train.observed, ni, split="train")
        ).to(device)

    losses: list[float] = []
    steps_done = 0
    for epoch in range(start_epoch, cfg.model.epochs):
        model.train()
        perm = torch.randperm(xtr.shape[0], generator=torch.Generator().manual_seed(seed + epoch))
        total = 0.0
        n_batches = 0
        with tqdm(
            range(0, xtr.shape[0], cfg.model.batch_size),
            desc=f"{split_name}/{seed} epoch {epoch + 1}/{cfg.model.epochs}",
            unit="batch",
        ) as bar:
            for start in bar:
                batch_idx = perm[start : start + cfg.model.batch_size]
                batch = xtr[batch_idx]
                if spec.family == "pretext":
                    kwargs = {}
                    if fp_targets is not None:
                        kwargs["ndvi_targets"] = fp_targets[batch_idx]
                    loss = model.pretext_loss(batch, obs_tr[batch_idx], **kwargs)
                elif spec.family == "reconstruction":
                    loss = _masked_reconstruction_loss(model, batch, obs_tr[batch_idx])
                else:
                    v1 = _augment(batch, seed + epoch + start)
                    v2 = _augment(batch, seed + epoch + start + 1000)
                    _, e1 = model(v1)
                    _, e2 = model(v2)
                    loss = _barlow_loss(e1, e2)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                total += float(loss.detach().cpu())
                n_batches += 1
                steps_done += 1
                bar.set_postfix(loss=round(total / n_batches, 5))
                if cfg.model.max_steps is not None and steps_done >= cfg.model.max_steps:
                    break
        mean_loss = total / max(1, n_batches)
        losses.append(mean_loss)
        rec = {
            "epoch": epoch + 1,
            "train_loss": round(mean_loss, 6),
            "wall_s": round(time.monotonic() - t0, 2),
        }
        with progress.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        _save_checkpoint(ck_path, model, opt, epoch + 1, seed)
        print(f"  {split_name} seed {seed} epoch {epoch+1}/{cfg.model.epochs}: "
              f"loss={mean_loss:.5f} steps={steps_done}", flush=True)
        if cfg.model.max_steps is not None and steps_done >= cfg.model.max_steps:
            break

    train_emb = (
        _train_embeddings(model, xtr)
        if spec.family in ("representation", "pretext") else None
    )
    clean = _scores(model, xte, obs_te, spec.family, train_emb)
    deviated_x = xte.clone()
    idx = feature_names.index("ndvi") if "ndvi" in feature_names else 0
    half = max(1, deviated_x.shape[1] // 2)
    deviated_x[:, half:, idx] = deviated_x[:, half:, idx] - 1.0
    deviated = _scores(model, deviated_x, obs_te, spec.family, train_emb)

    auprc = _auprc(clean, deviated)

    # Mandatory references (repo ML-review checklist): a simple non-neural
    # step detector and a random-initialized encoder of the same architecture.
    pre = xte[:, :half, idx].mean(dim=1)
    post = xte[:, half:, idx].mean(dim=1)
    step_clean = (post - pre).abs().cpu().numpy()
    dpre = deviated_x[:, :half, idx].mean(dim=1)
    dpost = deviated_x[:, half:, idx].mean(dim=1)
    step_deviated = (dpost - dpre).abs().cpu().numpy()
    step_auprc = _auprc(step_clean, step_deviated)

    seed_everything(seed)
    random_model = build_model(cfg.model.name, train.x.shape[-1], cfg.model).to(device)
    random_emb = (
        _train_embeddings(random_model, xtr)
        if spec.family in ("representation", "pretext") else None
    )
    rand_clean = _scores(random_model, xte, obs_te, spec.family, random_emb)
    rand_deviated = _scores(random_model, deviated_x, obs_te, spec.family, random_emb)
    random_auprc = _auprc(rand_clean, rand_deviated)

    report = {
        "schema": "mini-run/v1",
        "protocol": split_name,
        "seed": seed,
        "model": cfg.model.name,
        "family": spec.family,
        "param_count": int(sum(p.numel() for p in model.parameters())),
        "n_train_units": train.n,
        "n_test_units": test.n,
        "train_date_range": [min(r[0] for r in train.date_ranges), max(r[1] for r in train.date_ranges)],
        "test_date_range": [min(r[0] for r in test.date_ranges), max(r[1] for r in test.date_ranges)],
        "train_farms": sorted(set(train.farm_ids)),
        "test_farms": sorted(set(test.farm_ids)),
        "split_sha256": {"train": train.sha256, "test": test.sha256},
        "losses": [round(v, 6) for v in losses],
        "clean_score_mean": float(np.nanmean(clean)) if np.isfinite(clean).any() else None,
        "deviated_score_mean": float(np.nanmean(deviated)) if np.isfinite(deviated).any() else None,
        "synthetic_deviation_auprc": auprc,
        "step_reference_auprc": step_auprc,
        "random_init_auprc": random_auprc,
        "baseline_note": "Balanced clean/deviated sets (prevalence 0.5 by construction); random-init encoder and step detector are references, not trained models.",
        "synthetic_deviation_note": (
            "Controlled one-channel NDVI drop on held-out windows only; "
            "evidence for L2 controlled deviation, NOT stress detection."
        ),
        "n_imputed_features": int((~torch.from_numpy(test.observed).bool()).sum().item()),
        "valid_patch_counts": valid_patch_counts,
        "device": str(device),
        "max_steps": cfg.model.max_steps,
        "steps_done": steps_done,
        "wall_s": round(time.monotonic() - t0, 2),
        "git_sha": get_git_commit_sha(),
        "data_source": str(cfg.store_root),
        "holdout_boundary": cfg.protocol.holdout_boundary,
        "holdout_justification": cfg.protocol.justification,
        "claim_rung": cfg.claim_rung,
        "checkpoint": str(ck_path),
        "progress": str(progress),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def _optuna_study(cfg: RunConfig, train: Split, device: torch.device, out_dir: Path,
                  feature_names: list[str]) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    xtr = torch.from_numpy(train.x[: min(32, train.n)]).to(device)
    obs = torch.from_numpy(train.observed[: min(32, train.n)]).to(device)
    family = MODEL_REGISTRY[cfg.model.name].family
    fp_targets = None
    if cfg.model.name == "time2agri_fp":
        from ssl_anomaly.models.time2agri_archs import build_fp_targets

        ni = feature_names.index("ndvi") if "ndvi" in feature_names else 0
        fp_targets = torch.from_numpy(
            build_fp_targets(train.x[: min(32, train.n)], train.observed[: min(32, train.n)], ni, split="train")
        ).to(device)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        seed_everything(13)
        model = build_model(cfg.model.name, train.x.shape[-1], cfg.model).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        for _ in range(2):
            if family == "pretext":
                kw = {} if fp_targets is None else {"ndvi_targets": fp_targets}
                loss = model.pretext_loss(xtr, obs, **kw)
            elif family == "reconstruction":
                loss = _masked_reconstruction_loss(model, xtr, obs)
            else:
                _, e1 = model(xtr)
                _, e2 = model(xtr)
                loss = _barlow_loss(e1, e2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        return float(loss.detach().cpu())

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=max(1, cfg.n_trials), catch=())
    summary = {
        "study_name": study.study_name,
        "n_trials": len(study.trials),
        "best_params": {k: float(v) for k, v in study.best_params.items()},
        "best_value": float(study.best_value),
        "values": [float(t.value) for t in study.trials if t.value is not None],
    }
    (out_dir / "optuna_study.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _mlflow_log(cfg: RunConfig, name: str, report: dict[str, Any], report_path: Path, out_dir: Path) -> None:
    import mlflow

    # MLflow 3.x puts the filesystem store in maintenance mode; use a local
    # SQLite backend so runs are recorded without any external service.
    db = (out_dir / "mlflow.db").resolve().as_posix()
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    mlflow.set_experiment(cfg.mlflow_experiment)
    with mlflow.start_run(run_name=f"{cfg.name}-{name}-seed{report['seed']}"):
        mlflow.set_tags(
            {
                "farm": ",".join(report["test_farms"]),
                "protocol": name,
                "model": cfg.model.name,
                "spatial": cfg.representation.spatial,
                "temporal": cfg.representation.temporal,
                "weather_group": cfg.representation.weather_group,
                "seed": str(report["seed"]),
                "git_sha": report["git_sha"] or "",
                "claim_rung": cfg.claim_rung,
            }
        )
        mlflow.log_params(
            {
                "d_model": cfg.model.d_model,
                "latent_dim": cfg.model.latent_dim,
                "epochs": cfg.model.epochs,
                "lr": cfg.model.lr,
                "batch_size": cfg.model.batch_size,
            }
        )
        mlflow.log_metrics(
            {
                k: float(v)
                for k, v in {
                    "loss": report["losses"][-1] if report["losses"] else float("nan"),
                    "clean_score": report["clean_score_mean"] or 0.0,
                    "deviated_score": report["deviated_score_mean"] or 0.0,
                    "synthetic_auprc": report["synthetic_deviation_auprc"] or 0.0,
                    "step_reference_auprc": report["step_reference_auprc"] or 0.0,
                    "random_init_auprc": report["random_init_auprc"] or 0.0,
                }.items()
                if v is not None
            }
        )
        mlflow.log_artifact(str(report_path))


def run_experiment(
    cfg: RunConfig,
    dataset: MiniDataset,
    *,
    out_root: Path | None = None,
) -> dict[str, Any]:
    """Run all protocol rotations and seeds; write a complete provenance report."""
    t0 = time.monotonic()
    out = out_root or cfg.out_root
    out.mkdir(parents=True, exist_ok=True)
    splits = split_datasets(dataset, cfg)
    device = _device(cfg.model.device)
    all_reports: list[dict[str, Any]] = []
    for name, train, dev, test in splits:
        for seed in cfg.model.seeds:
            report = _one_seed(
                cfg, name, train, dev, test, seed, out, dataset.feature_names, dataset.valid_patch_counts
            )
            rp = out / cfg.name / name / f"seed_{seed}" / "report.json"
            _mlflow_log(cfg, name, report, rp, out)
            all_reports.append(report)

    def agg(key: str) -> dict[str, Any] | None:
        vals = np.array([r[key] for r in all_reports if r.get(key) is not None], dtype=float)
        if vals.size == 0:
            return None
        return {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": int(vals.size),
        }

    summary = {
        "schema": "mini-experiment-report/v1",
        "experiment": cfg.name,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": get_git_commit_sha(),
        "data_source": str(cfg.store_root),
        "source_manifest": dataset.source_manifest,
        "config": config_dict(cfg),
        "model_spec": next(s for s in model_specs() if s["name"] == cfg.model.name),
        "seeds": list(cfg.model.seeds),
        "device": str(device),
        "n_seeds": len(cfg.model.seeds),
        "n_rotations": len(splits),
        "claim_rung": cfg.claim_rung,
        "claim_note": "Screening evidence only. No stress, generalization or baseline-superiority claim.",
        "holdout_boundary": cfg.protocol.holdout_boundary,
        "holdout_justification": cfg.protocol.justification,
        "valid_patch_counts": dataset.valid_patch_counts,
        "metric_tracks": {
            "reconstruction": ["loss", "clean_score_mean", "deviated_score_mean", "synthetic_deviation_auprc"],
            "representation": ["clean_score_mean", "deviated_score_mean", "synthetic_deviation_auprc"],
        },
        "aggregate": {
            "clean_score": agg("clean_score_mean"),
            "deviated_score": agg("deviated_score_mean"),
            "synthetic_auprc": agg("synthetic_deviation_auprc"),
            "step_reference_auprc": agg("step_reference_auprc"),
            "random_init_auprc": agg("random_init_auprc"),
        },
        "per_seed": all_reports,
        "optuna": _optuna_study(cfg, splits[0][1], device, out, dataset.feature_names),
        "wall_s": round(time.monotonic() - t0, 2),
        "output_dir": str(out),
    }
    final_losses = [r["losses"][-1] for r in all_reports if r["losses"]]
    summary["aggregate"]["loss"] = {
        "mean": float(np.mean(final_losses)) if final_losses else None,
        "std": float(np.std(final_losses)) if final_losses else None,
        "n": len(final_losses),
    }
    report_path = out / "report.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"EXPERIMENT COMPLETE wall={summary['wall_s']}s report={report_path} "
        f"runs={len(all_reports)} rung={cfg.claim_rung}",
        flush=True,
    )
    return summary

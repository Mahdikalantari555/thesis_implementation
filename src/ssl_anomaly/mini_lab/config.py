"""Configuration for mini-dataset experiments (YAML-driven, no hard-coded protocol)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ssl_anomaly.utils.paths import localize

DEFAULTS = Path(__file__).resolve().parents[3] / "Data" / "Processed" / "mini_research_dataset"


@dataclass
class RepresentationConfig:
    spatial: str = "full_field"  # pixel | patch | full_field
    patch_sizes: tuple[int, ...] = (3, 5, 7)
    max_pixels: int = 64
    min_valid_fraction: float = 0.5
    temporal: str = "full"  # full | sliding | seasonal | variable | sampling
    window_length: int = 60
    stride: int = 30
    sample_step: int = 2
    include_delta_t: bool = True
    include_indices: bool = True
    weather_group: str = "weather_plus_derived"


@dataclass
class ProtocolConfig:
    name: str = "lodo"  # single_farm | lodo | intra_farm_temporal
    # Farm names come from the store (selection layer 3farms_dev). For
    # single_farm, an empty test_farms means "every other farm in the store".
    train_farms: tuple[str, ...] = ()
    test_farms: tuple[str, ...] = ()
    # Accepted 2026-11-25 split for the mini lab (parent plan §9): train <= 2022,
    # dev 2023, test 2024+. The temporal holdout is therefore 2024-01-01, not
    # 2025-01-01 (which mixed dev into test and contradicted the 490 split).
    holdout_boundary: str = "2024-01-01"
    justification: str = (
        "Accepted 2026-11-25 mini split: train <= 2022, dev 2023, test 2024+ "
        "(docs/mini_lab_regional_eval_plan.md §9); configurable per run."
    )


@dataclass
class ModelConfig:
    name: str = "gru_ae"
    d_model: int = 32
    latent_dim: int = 16
    n_layers: int = 1
    dropout: float = 0.1
    epochs: int = 3
    # Equal-compute gate: stop after this many optimiser steps regardless of
    # dataset size. ``None`` keeps the legacy epoch-only behaviour.
    max_steps: int | None = None
    batch_size: int = 8
    lr: float = 1e-3
    seeds: tuple[int, ...] = (13, 42)
    device: str = "auto"


@dataclass
class RunConfig:
    name: str = "mini_smoke"
    store_root: Path = DEFAULTS
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    out_root: Path = DEFAULTS / "experiments"
    resume: bool = False
    n_trials: int = 2
    mlflow_experiment: str = "mini-research-dataset"
    claim_rung: str = "L2"


def _as_path(value: Any, default: Path) -> Path:
    return localize(value) if value is not None else default


def load_run_config(path: str | Path) -> RunConfig:
    """Load a run config from YAML, validating names and keeping CLI overrides out."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    rep = raw.get("representation", {}) or {}
    proto = raw.get("protocol", {}) or {}
    model = raw.get("model", {}) or {}
    cfg = RunConfig(
        name=str(raw.get("name", "mini_run")),
        store_root=_as_path(raw.get("store_root"), DEFAULTS),
        representation=RepresentationConfig(
            spatial=str(rep.get("spatial", "full_field")),
            patch_sizes=tuple(int(v) for v in rep.get("patch_sizes", (3, 5, 7))),
            max_pixels=int(rep.get("max_pixels", 64)),
            min_valid_fraction=float(rep.get("min_valid_fraction", 0.5)),
            temporal=str(rep.get("temporal", "full")),
            window_length=int(rep.get("window_length", 60)),
            stride=int(rep.get("stride", 30)),
            sample_step=int(rep.get("sample_step", 2)),
            include_delta_t=bool(rep.get("include_delta_t", True)),
            include_indices=bool(rep.get("include_indices", True)),
            weather_group=str(rep.get("weather_group", "weather_plus_derived")),
        ),
        protocol=ProtocolConfig(
            name=str(proto.get("name", "lodo")),
            train_farms=tuple(proto.get("train_farms", ()) or ()),
            test_farms=tuple(proto.get("test_farms", ()) or ()),
            holdout_boundary=str(proto.get("holdout_boundary", ProtocolConfig.holdout_boundary)),
            justification=str(proto.get("justification", ProtocolConfig.justification)),
        ),
        model=ModelConfig(
            name=str(model.get("name", "gru_ae")),
            d_model=int(model.get("d_model", 32)),
            latent_dim=int(model.get("latent_dim", 16)),
            n_layers=int(model.get("n_layers", 1)),
            dropout=float(model.get("dropout", 0.1)),
            epochs=int(model.get("epochs", 3)),
            max_steps=(int(model["max_steps"]) if model.get("max_steps") is not None else None),
            batch_size=int(model.get("batch_size", 8)),
            lr=float(model.get("lr", 1e-3)),
            seeds=tuple(int(v) for v in model.get("seeds", (13, 42))),
            device=str(model.get("device", "auto")),
        ),
        out_root=_as_path(raw.get("out_root"), DEFAULTS / "experiments"),
        resume=bool(raw.get("resume", False)),
        n_trials=int(raw.get("n_trials", 2)),
        mlflow_experiment=str(raw.get("mlflow_experiment", "mini-research-dataset")),
        claim_rung=str(raw.get("claim_rung", "L2")),
    )
    valid_spatial = {"pixel", "patch", "full_field"}
    valid_temporal = {"full", "sliding", "seasonal", "variable", "sampling"}
    valid_protocol = {"single_farm", "lodo", "intra_farm_temporal"}
    if cfg.representation.spatial not in valid_spatial:
        raise ValueError(f"spatial must be one of {sorted(valid_spatial)}")
    if cfg.representation.temporal not in valid_temporal:
        raise ValueError(f"temporal must be one of {sorted(valid_temporal)}")
    if cfg.protocol.name not in valid_protocol:
        raise ValueError(f"protocol must be one of {sorted(valid_protocol)}")
    if cfg.claim_rung not in {"L1", "L2"}:
        raise ValueError("mini-dataset claims are capped at L2")
    return cfg


def config_dict(cfg: RunConfig) -> dict[str, Any]:
    def convert(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: convert(v) for k, v in asdict(obj).items()}
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        return obj

    return convert(cfg)

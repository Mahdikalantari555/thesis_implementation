"""Benchmark architectures A1–A10 (design §Architecture matrix).

One trunk abstraction, five objectives, two backbones. Nothing in here re-derives
a loss that already has an owner: the objective definitions are the ones the design
pre-registered, and :func:`arms.objective_signature` is what the objective-audit
test reads to prove that reconstruction, latent reconstruction, prediction and JEPA
are still four different estimands rather than four spellings of one.
"""

from __future__ import annotations

from ssl_anomaly.models.bench.arms import (
    ARM_BY_MODEL,
    ARM_CLASSES,
    CAPACITY_CONTROL_TRUNKS,
    PARADIGM_ORDER,
    HybridArm,
    InputReconstructionArm,
    JEPArm,
    LatentReconstructionArm,
    PredictionArm,
    build_arm,
    count_trainable,
    estimate_flops,
    masked_bin_error,
    masked_mean,
    objective_signature,
    param_band_check,
    target_centric,
)
from ssl_anomaly.models.bench.time2agri_arms import (
    ARM_CLASSES as TIME2AGRI_ARM_CLASSES,
    TIME2AGRI_ARM_BY_MODEL,
    TIME2AGRI_MODEL_IDS,
    Time2AgriFPBenchArm,
    Time2AgriFFBenchArm,
    Time2AgriTDBenchArm,
    build_time2agri_bench_arm,
    time2agri_objective_signature,
)
from ssl_anomaly.models.bench.backbones import (
    BackboneOutput,
    CausalMambaBackbone,
    CausalStackBackbone,
    NumericalInstability,
    SequenceBackbone,
    build_backbone,
)

__all__ = [
    "ARM_BY_MODEL",
    "ARM_CLASSES",
    "TIME2AGRI_ARM_BY_MODEL",
    "TIME2AGRI_ARM_CLASSES",
    "TIME2AGRI_MODEL_IDS",
    "Time2AgriFPBenchArm",
    "Time2AgriFFBenchArm",
    "Time2AgriTDBenchArm",
    "build_time2agri_bench_arm",
    "time2agri_objective_signature",
    "CAPACITY_CONTROL_TRUNKS",
    "PARADIGM_ORDER",
    "BackboneOutput",
    "CausalMambaBackbone",
    "CausalStackBackbone",
    "HybridArm",
    "InputReconstructionArm",
    "JEPArm",
    "LatentReconstructionArm",
    "NumericalInstability",
    "PredictionArm",
    "SequenceBackbone",
    "build_arm",
    "build_backbone",
    "count_trainable",
    "estimate_flops",
    "masked_bin_error",
    "masked_mean",
    "objective_signature",
    "param_band_check",
    "target_centric",
]

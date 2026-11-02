"""Evaluation: CV splits, synthetic injection, cloud robustness, lead time, validation gate, probing, contamination."""

from __future__ import annotations

from .cloud_robustness import (
    COMPLETENESS_LEVELS,
    evaluate_e_align_three_metrics,
    evaluate_e_norm_trimming_stability,
    run_cloud_completeness_benchmark,
    subsample_cloud_completeness,
)
from .contamination import (
    CONTAMINATION_RATES,
    analyze_false_alarm_confounders,
    run_contamination_sweep,
)
from .cv import (
    cluster_field_ci,
    compare_detectors_wilcoxon_holm,
    generate_splits_dict,
    save_splits_json,
    simulate_490_field_archive,
)
from .early_detection import (
    EIGENCL_REFERENCE_LEAD_DAYS,
    run_loyo_early_detection,
    simulate_progressive_onset_archive,
)
from .probing import (
    extract_frozen_embeddings,
    probe_binary_target,
    probe_continuous_targets,
    run_rq7_probing_suite,
)
from .stats import (
    bootstrap_ci,
    holm_correction,
    paired_wilcoxon,
    spearman_correlation,
)
from .synthetic import (
    InjectionConfig,
    inject_anomaly,
    synthetic_benchmark,
)
from .validation_gate import (
    build_expert_event_review_packet,
    evaluate_stress_validation_gate,
)

__all__ = [
    "COMPLETENESS_LEVELS",
    "CONTAMINATION_RATES",
    "EIGENCL_REFERENCE_LEAD_DAYS",
    "InjectionConfig",
    "analyze_false_alarm_confounders",
    "bootstrap_ci",
    "build_expert_event_review_packet",
    "cluster_field_ci",
    "compare_detectors_wilcoxon_holm",
    "evaluate_e_align_three_metrics",
    "evaluate_e_norm_trimming_stability",
    "evaluate_stress_validation_gate",
    "extract_frozen_embeddings",
    "generate_splits_dict",
    "holm_correction",
    "inject_anomaly",
    "paired_wilcoxon",
    "probe_binary_target",
    "probe_continuous_targets",
    "run_cloud_completeness_benchmark",
    "run_contamination_sweep",
    "run_loyo_early_detection",
    "run_rq7_probing_suite",
    "save_splits_json",
    "simulate_490_field_archive",
    "simulate_progressive_onset_archive",
    "spearman_correlation",
    "subsample_cloud_completeness",
    "synthetic_benchmark",
]

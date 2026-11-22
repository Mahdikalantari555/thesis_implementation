# oracle-floor-audit — Spec Delta

## ADDED Requirements

### Requirement: measured floor verdict
The audit SHALL produce the (A)/(C)/(B) decision per injected family from
σ_h and α_h measured on the clean unassigned pool V with the frozen scorer,
never from assumed constants, and SHALL record every input (h list, level,
family, V composition, field split, arm seed, git sha) in its report JSON.

#### Scenario: h=1 parity
- **WHEN** the scorer runs with gap=1
- **THEN** the residual and path outputs are byte-identical to the
  pre-change `one_step_ahead_v3` outputs.

#### Scenario: honest-negative branch
- **WHEN** max_h Q_h < 0.62 in every phase group on V_cal and V_test
- **THEN** the report declares branch (A) for persistent drops and no
  retraining is proposed.

### Requirement: absorption–σ audit
The audit SHALL measure, per phase group and per gap h ∈ {1,2,3,4,6,8}:
σ_h (1.4826·MAD of clean signed ndvi-channel residuals), α_h
(1 − median residual shift of a persistent injection / Δ_norm), the SNR
index Q_h = (1−α_h)·σ_1/σ_h, and the oracle power of the best h* on V_cal
confirmed on V_test, with the gap-vs-measured decision bands
(gap ≤ 0.10 → A; 0.10–0.30 → C; ≥ 0.30 → B).

#### Scenario: V pool discipline
- **WHEN** the audit selects its cycles
- **THEN** no selected cycle belongs to train, dev, or test, the pinned
  cycles-parquet hash matches the splits manifest, and the V_cal/V_test
  split is field-blocked and deterministic.

### Requirement: solver equivalence decomposition
The audit SHALL measure the realized per-channel band deltas of the SAVI and
NDVI solvers at equal nominal Δ on real store patches and report the
band-magnitude ratio ρ and the NDVI-equivalent Δ* per severity level, with an
optional pre-registered one-look confirmatory detection cell on the locked
eval panel.

#### Scenario: mechanism separation
- **WHEN** the confirmatory matched-Δ cell runs
- **THEN** the report classifies the SAVI≫NDVI asymmetry as solver/unit
  non-equivalence, scorer asymmetry, or mixed, using the consultant's
  pre-registered decision table.

### Requirement: per-injection detection logging
Severity cells SHALL record `det_flags` (per injection, aligned with
`injections`) and `detection_by_onset` (per onset phase: n, n_detected,
rate) as additive JSON keys.

#### Scenario: onset phases recorded
- **WHEN** a severity cell is written
- **THEN** the recorded onset phases are exactly {T//6, T//2, 5T//6} and
  their n values sum to the cell's n.

### Requirement: MDE curve deliverable
The MDE script SHALL fit a floor-anchored probit dose–response per arm ×
family from the committed severity metrics, report Δ*(0.8) with a bootstrap
CI, and compare against the pre-registered Δ_req = 0.165, labelling any
extrapolation beyond S6 as a model, not a measurement.

#### Scenario: censored MDE
- **WHEN** Δ*(0.8) exceeds the largest ladder step
- **THEN** the report marks the family ">S6 (censored)" and the extrapolated
  value is stored in a separate, explicitly-labelled field.

# reproducible-benchmark-execution Specification

## Purpose

Define how the benchmark runs: budget-aware scheduling over a complete committed
experiment registry, checkpoint/resume on Colab T4, persistent outputs and
honest reporting of incomplete coverage.

## ADDED Requirements

### Requirement: Budget-aware execution

The system SHALL schedule experiments according to a predefined computational
budget and predefined priority order (screening first, full benchmark second,
deferred review last); the scheduler SHALL NOT discard experiments silently.

#### Scenario: Budget exhaustion
- **WHEN** the configured T4 hour budget is exhausted
- **THEN** all unfinished experiments are marked `Deferred` with the reason,
  remain listed in every report, and no gate or headline may report `pass`
  for a deferred cell

### Requirement: Complete experiment registry

All planned experiments SHALL be registered (config committed, hash recorded)
before execution; the registry SHALL assign every experiment an explicit status
at all times.

#### Scenario: Pre-execution completeness
- **WHEN** the benchmark initializes
- **THEN** every matrix cell exists in the registry with status `Pending`, and
  the matrix hash matches the committed pre-registration

### Requirement: Experiment state management

The registry SHALL support exactly these states: `Pending`, `Running`,
`Completed`, `Failed`, `Interrupted`, `Deferred`, `Not Applicable`;
transitions SHALL be recorded with timestamps; `Not Applicable` SHALL carry a
machine-checkable reason.

#### Scenario: Crash mid-experiment
- **WHEN** an experiment is interrupted (runtime loss, ECC error, OOM)
- **THEN** it is marked `Interrupted` on the next run with the last checkpoint
  reference, and resume restores from it

### Requirement: Checkpoint and resume

The system SHALL checkpoint model, optimizer, scheduler and RNG state per
seed, and SHALL resume interrupted experiments to completion from the latest
valid checkpoint; the saved sidecar metadata SHALL be sufficient to verify
dataset/split/config bindings on restore.

#### Scenario: Resume verification
- **WHEN** an experiment resumes
- **THEN** the binding hashes (cycle file, splits, norm, config) are re-verified
  and a mismatch aborts the resume with a diagnostic

### Requirement: Experiment manifests

Each experiment SHALL generate a manifest sufficient to reproduce it: ids,
architecture, suppression, input/fusion/normality/calibration ids, protocol,
seed, hyperparameters, data/code version bindings, software and hardware,
timestamps, checkpoint refs, metric outputs with prevalence, wall time, status.

#### Scenario: Round-trip reproduction
- **WHEN** a reviewer re-executes an experiment from its manifest on the same
  data versions
- **THEN** reported metrics reproduce within the recorded seed spread

### Requirement: Colab T4 compatibility

The benchmark SHALL run on Google Colab NVIDIA T4: device auto-detection, GPU
memory monitoring, mixed precision where numerically appropriate (documented in
the config), gradient accumulation for budget control, minimal host↔device
transfer, explicit GPU-memory release.

#### Scenario: T4 absence
- **WHEN** no CUDA device is available
- **THEN** the benchmark stops with a diagnostic before any training (classical
  baselines and preprocessing MAY run on CPU, clearly labelled)

### Requirement: Persistent outputs

Checkpoints, manifests, logs, metrics, statistical summaries, tables and the
final report SHALL be persisted to Google Drive with dated, self-describing
bundle names (e.g. `architecture_benchmark_2026-11-25.zip`), never bare
`results.zip`; GitHub remains text-only.

#### Scenario: Completion summary
- **WHEN** the notebook finishes a cycle
- **THEN** it prints the exact bundle filename and its SHA-256, and writes the
  same into `reports/architecture_benchmark/LATEST_BUNDLE.md`

### Requirement: Honest completion reporting

The final report SHALL distinguish completed, failed, interrupted, deferred and
not-applicable cells; incomplete coverage SHALL be stated in the executive
summary and in every affected section; the capstone's gates SHALL NOT cite this
benchmark's deferred cells as passing evidence.

#### Scenario: Executive summary coverage statement
- **WHEN** the final report is generated
- **THEN** its executive summary includes a coverage line (completed / deferred /
  not-applicable cell counts) and a sentence naming the largest incomplete area

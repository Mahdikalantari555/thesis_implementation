# anomaly-discovery-triage Specification

## Purpose

Discover candidate anomalies by combining model scores with model-independent
statistics, tier them by certainty, and verify them without letting the model
learn its own or the detectors' guesses.

## ADDED Requirements

### Requirement: Model-independent statistical detectors

The system SHALL compute at least temporal robust z (same pixel, same stage or
DOY window, prior years), within-field spatial robust z and a local spatial
cluster statistic, a harmonic/breakpoint residual, and a multi-band
consistency check, none of which read model outputs.

#### Scenario: Detector independence
- **WHEN** the model checkpoint changes
- **THEN** detector outputs are unchanged

### Requirement: Fusion, tiers and quarantine

Model and detector ranks SHALL be fused into Normal-confident, Suspected and
Verified tiers; only independent confirmation (blind expert review or
independent records) SHALL create Verified; Suspected items SHALL be excluded
from normal memory, replay and SSL updates until resolved.

#### Scenario: Suspected item in an update
- **WHEN** an update pool is assembled
- **THEN** no Suspected or unsure item is present

### Requirement: No confirmation loop

Detector outputs SHALL NOT be used as positive training targets for the
encoder; SSL updates SHALL use only Normal-confident data with confidence
weights; a share of Verified items SHALL be held back for evaluation only.

#### Scenario: Training targets audit
- **WHEN** a training batch is inspected
- **THEN** it contains no detector-derived labels

### Requirement: Active learning queries

The system SHALL rank review candidates by uncertainty, diversity and
farm/year/stage spread, present them blind to score and source, and mix in a
fixed share of random Normal-confident items.

#### Scenario: Review packet
- **WHEN** a packet of K items is generated
- **THEN** it records the query reason per item, hidden from the reviewer

### Requirement: Claim discipline

Evaluation units SHALL be pixel × time window and Spatial Anomaly Event;
without a valid verified set, the system SHALL report only stability,
agreement, coherence, recurrence and robustness, and SHALL NOT report
precision, recall or stress detection; synthetic-injection and real-discovery
results SHALL be labelled and reported separately.

#### Scenario: No verified set
- **WHEN** a report is generated with zero verified items
- **THEN** it contains no Precision@K, precision or recall field

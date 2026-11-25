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

### Requirement: Separate representation-learning and anomaly-learning protocols (r2)

The encoder's self-supervised training and the anomaly head's adaptation
SHALL be separate protocols with separate data, losses, schedules and version
numbers, and each SHALL use only the data categories allowed by the design's
data table: encoder SSL uses Normal-confident and reviewed-normal data (plus
unscreened data only in an explicit first-pass arm), never anomaly labels,
detector outputs or Suspected items; the anomaly head uses Verified positives
and Normal-confident/reviewed-normal negatives, plus unlabeled data only in
weak-supervision arms. Anomaly-head gradients SHALL NOT update the encoder in
the default arm.

#### Scenario: Data category audit
- **WHEN** an encoder SSL batch is assembled
- **THEN** it contains no Suspected, Verified-anomaly, synthetic or detector-labelled item

### Requirement: Separate anomaly head trained on verified anomalies (r2)

The system SHALL provide a separate anomaly head on encoder features and
calibrated score components that trains only once at least `min_verified`
Verified events across at least two farms and two years exist, with a held-
back Verified evaluation part never used for training, and SHALL report
"insufficient labels" instead of metrics below that threshold.

#### Scenario: Too few verified events
- **WHEN** fewer than `min_verified` Verified events exist
- **THEN** no anomaly head is trained and the system runs in discovery-only mode

### Requirement: Weak-supervision arms remain experimental (r2)

Confidence-weighted learning and Positive-Unlabeled learning SHALL be
available only as labelled experimental arms that treat Suspected and
unscreened items as unlabeled with bounded weight, SHALL report the assumed
class prior and its sensitivity sweep, and SHALL NEVER convert their outputs
into labels or into evidence for the same items' triage.

#### Scenario: PU output
- **WHEN** a PU arm scores a Suspected item highly
- **THEN** the item stays Suspected and its label field is unchanged

### Requirement: Anomaly memory (r2)

The system SHALL keep an anomaly memory, separate from normal memory and
versioned with the model, with partitions verified, unresolved (Suspected and
Unsure, with evidence vector and flagging model version) and rejected
(reviewed normal), and SHALL use the verified and unresolved partitions as an
exclusion list for normal memory, replay and normalization statistics.

#### Scenario: Unresolved item excluded
- **WHEN** normalization statistics are computed
- **THEN** no observation in the verified or unresolved partitions contributes

### Requirement: Anti-confirmation-loop audit (r2)

Labels SHALL come only from blind review or independent records; review
packets SHALL record whether each item was model-selected or randomly
selected; precision on random and model-selected reviewed items SHALL be
reported separately; detectors SHALL be frozen per release and never fitted to
model outputs; and the share of anomaly-head positives first surfaced by the
model itself SHALL be logged per update.

#### Scenario: Self-reinforcement warning
- **WHEN** the model-surfaced share of positives rises while random-review precision stays flat
- **THEN** the update report raises an owner-review flag

### Requirement: Calibrated score components and aggregation arms (r2)

Each score component (reconstruction residual, forecast residual, JEPA latent
error, memory distance) SHALL be calibrated on a Normal-confident calibration
set disjoint from encoder training data into an upper-tail p-value or robust z
before fusion; raw magnitudes SHALL NOT be fused directly; maximum-based,
rank-based and calibrated-weighted aggregation and single-component arms SHALL
be kept as experimental arms; weights SHALL be fitted without test data; and
score stability (rank correlation and top-K overlap across seeds, adjacent
checkpoints, observation dropout and phenology shifts) SHALL be reported for
every arm.

#### Scenario: Raw residual fusion attempted
- **WHEN** a fusion receives uncalibrated components
- **THEN** it fails with an error naming the missing calibration

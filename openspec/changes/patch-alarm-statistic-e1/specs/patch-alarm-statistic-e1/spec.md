# patch-alarm-statistic-e1 — Spec Delta

## ADDED Requirements

### Requirement: selectable alarm statistic with own-null calibration
The patch eval path SHALL accept an alarm-statistic selection (default:
the production statistic, byte-identical outputs) and SHALL calibrate every
non-default statistic on its OWN null — per-gap τ from the calibration
subset's paths computed under the identical statistic — recording τ,
far_achieved, and tau_calibration in the metrics JSON.

#### Scenario: default unchanged
- **WHEN** no statistic selection is passed
- **THEN** the severity outputs are byte-identical to
  `severity_sweep_patch_v2` (same keys, same hashes).

#### Scenario: borrowed threshold refused
- **WHEN** a candidate statistic's calibration null cannot reach the FAR
  target (degenerate τ)
- **THEN** the run fails loud — it never borrows another statistic's τ.

### Requirement: frozen design before the confirmatory look
h* per arm and the candidate list SHALL be frozen from the V_cal design
surface (committed oracle-floor reports) BEFORE the confirmatory sweep, and
the confirmatory run SHALL happen at most once per adopted statistic on the
locked eval panel.

#### Scenario: one look
- **WHEN** the confirmatory sweep runs
- **THEN** its metrics JSON records the frozen h*, the new eval-subset hash,
  and `statistic` id; no second look is taken without a new pre-registered
  deviation record.

### Requirement: pre-registered adoption gates
A candidate statistic SHALL be adopted only if ALL hold: far_achieved ≤ 0.05
with the held-out clean post-onset alarm rate's Clopper–Pearson upper ≤ 9.1 %;
paired McNemar detection gain ≥ +0.10 over the production statistic in BOTH
index families; ladder non-flat with S6 ≥ S1 per family. Any failure SHALL
close the branch with the measured ceiling recorded.

#### Scenario: honest negative
- **WHEN** a gate fails
- **THEN** the report records the measured ceiling, the statistic is not
  adopted, and retraining (branch B) is adjudicated against the CONFIRMED
  ceiling — never against the V-promise.

### Requirement: pixel-based candidates measured before wiring
Pixel-based aggregation alternatives (mean-of-pixel-z², per-channel Tippett
fusion) SHALL be measured on the V design surface through the audit's
per-gap empirical-power leg BEFORE any confirmatory look, and wired only if
their V-measured gap ≥ +0.10 in BOTH index families — the measured field-mean
gain MUST NOT be attributed to the pixel-based method.

#### Scenario: scope honesty
- **WHEN** the E1b measurement lands
- **THEN** its report states explicitly which statistic family each number
  belongs to (field-mean vs pixel-based), and no cross-family attribution
  appears in any finding.

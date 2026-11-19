## ADDED Requirements

### Requirement: Patch spatial score slot (trial stage)

The platform SHALL expose one patch-arm score slot backed by the frozen
`patch_*_main` checkpoint (winner of the pre-registered G4 rule + a finite
full-archive MDM80), scored from the patch store with the same frozen norm,
frozen per-pixel σ table and causal leave-one-bin-out aggregation the trial
used. The slot SHALL be marked as trial stage in the registry and the UI.

#### Scenario: Registry parity across both mirrors

- **WHEN** the score registry is edited on either side
  (`scripts/platform_channels.py` or `platform/src/data/channels.ts`)
- **THEN** the parity contract tests pass with the new slot present in both
  mirrors, provenance `model`, and the trial-stage marker identical; the
  existing slot order and semantics are unchanged.

#### Scenario: Artifact binding is fail-closed

- **WHEN** the exporter scores the patch slot
- **THEN** the checkpoint, norm sidecar and σ table must carry the same-run
  hashes and bind to the store's locked provenance (splits + holdout shas);
  any mismatch aborts the export with the verbatim error — no fallback
  scoring, no refit, no proxy substitution.

#### Scenario: Absent where no patch cycle exists

- **WHEN** an observation date lies outside every admissible patch cycle of
  its field (e.g. one of the 120 fields with only censored cycles, or dates
  outside the cycle grid)
- **THEN** the slot resolves to `score = null` with provenance `absent` —
  never a proxy value, never a neighbouring-bin interpolation.

#### Scenario: Provenance honesty

- **WHEN** the payload documents the slot
- **THEN** it records: claim noun `synthetic anomaly`, ladder rung `L2`,
  trial stage, checkpoint/norm/σ/store hashes, eval-subset hash of the
  severity protocol that froze the checkpoint, and the sentence that the
  positive class is "suspected stress (anomaly)" — never "stress" — until
  L4 evidence exists.

#### Scenario: Existing slots stay byte-identical

- **WHEN** the exporter runs with the patch slot enabled
- **THEN** every existing score array, channel array, flag and provenance
  RLE outside the new slot is byte-identical to the previous export (the
  legacy slots are not re-scored or re-normalized).

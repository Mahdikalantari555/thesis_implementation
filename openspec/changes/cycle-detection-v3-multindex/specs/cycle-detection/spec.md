## ADDED Requirements

### Requirement: Fused multi-index harvest detection

The system SHALL provide a fused cycle detector whose harvest boundary set
is the union of the v1 (NDVI-only) detected harvests and multi-index fusion
candidates, and SHALL build cycles from that boundary set with the same
180–540-day validity rule and leading/trailing censoring as v1.

#### Scenario: v1 subset equivalence

- **WHEN** the fusion evidence is disabled or no candidate fires
- **THEN** `segment_cycles_fused` output is identical to `segment_cycles`
  output for the same field series (windows, censoring, kinds, ratoon
  numbering).

#### Scenario: Corroborated-only candidates

- **WHEN** a vegetation drop ≥ `veg_drop` occurs without an LST rise ≥
  `lst_rise` and without a SWIR1 rise ≥ `swir_rise` in the same observation
- **THEN** no candidate harvest is created (dust/cloud-style artifacts are
  rejected by the corroboration requirement).

#### Scenario: Deduplication and boundary protection

- **WHEN** a candidate falls within `dedup_days` of any accepted boundary
  (v1 harvest or another candidate) or within `edge_guard` days of a lost
  span edge
- **THEN** the candidate is rejected, so one harvest event cannot produce
  two boundaries.

#### Scenario: Determinism

- **WHEN** the same field series is segmented twice
- **THEN** the fused output is byte-identical (no RNG anywhere in the
  fusion path).

### Requirement: v1 artifact preservation

The fused path SHALL NOT write to any v1 artifact path; all v2 artifacts
use explicit v2 names, and every v2 manifest records `derived_from`
pointing at the v1 manifest SHA-256 it supersedes.

#### Scenario: Coexisting arms

- **WHEN** the v2 chain is built
- **THEN** `cycles_490.parquet`, `splits_locked.json`,
  `holdout_manifest.json`, `patches.zarr` and `norm_stats_full.json` are
  byte-unchanged, and models trained on either arm remain reproducible
  from their own frozen manifests.

### Requirement: Fail-closed pre-extraction gates

The v2 cycle table SHALL pass length-distribution, season-clustering,
per-field-count and v1⊆v2 gates before the patch-store extraction handover;
a gate failure SHALL block the handover and print the failing statistics
verbatim.

#### Scenario: Gate blocks a bad table

- **WHEN** the v2 table fails any D6 gate
- **THEN** the chain stops before extraction and the report names the
  failing gate with its numbers.

## ADDED Requirements

### Requirement: Band-consistent synthetic injection basis

The patch severity-sweep injector SHALL provide an `injection_basis` with value
`bands_consistent`, which realizes every requested vegetation-index drop through
the raw Red/NIR reflectance bands and recomputes every derived stored index (the
NDVI channel) from the post-injection bands, and value `index_only`, which
reproduces the pre-change channel-shift behavior verbatim. Primary severity
gates SHALL run `bands_consistent`; `index_only` SHALL remain available as an
explicitly labeled sensitivity row; every severity-sweep report SHALL record the
basis in its provenance.

#### Scenario: Band–index consistency invariant

- **WHEN** a `bands_consistent` injection completes on a patch tensor
- **THEN** at every injected pixel-bin the stored NDVI channel equals NDVI
  recomputed from the stored Red/NIR bands (automated unit-test invariant), and
  the realized per-bin drop of the requested index is recorded in the injection
  metadata

#### Scenario: Side-effect index recorded, not gated

- **WHEN** a `bands_consistent` injection changes the non-requested index as a
  side effect of the band solve (SAVI under an ndvi-arm solve, NDVI under a
  savi-arm solve)
- **THEN** the realized side-effect drop is recorded in the metadata but the
  severity gate remains defined on the requested index only

#### Scenario: Legacy sensitivity row reproducible

- **WHEN** `index_only` is selected with the parameters of any historical
  finding (18/22/23/24)
- **THEN** the injected tensor is byte-identical to the pre-change
  implementation (regression test), so those findings remain exactly
  reproducible

#### Scenario: Basis and protocol recorded

- **WHEN** a severity-sweep report is written
- **THEN** its provenance carries `injection_basis`, and the sweep protocol id
  distinguishes the bases (`severity_sweep_patch_v1` = index_only,
  `severity_sweep_patch_v2` = bands_consistent)

#### Scenario: No training-artifact impact

- **WHEN** this change is applied and the eval smoke gate runs
- **THEN** all existing checkpoints, σ tables, store artifacts and platform
  payloads are byte-unchanged

# Spec Delta

## Purpose

Define a small, additive three-farm research dataset and a configurable experiment harness so architectures, input representations and SSL strategies can be screened cheaply before committing the full archive, without raising claims beyond controlled-detection evidence.

## ADDED Requirements

### Requirement: Additive three-farm satellite store

The mini store SHALL be additive to the frozen v2 artifacts and SHALL contain exactly the three companion farms with the 538 real scene acquisition dates from `scene_inventory.parquet`. It SHALL store seven physical bands with per-pixel validity and in-field masks, farm geometry and pixel-grid metadata in a portable zarr v2 layout.

#### Scenario: Date source
- **WHEN** the store's date list is built
- **THEN** dates come from scene acquisition dates and never from day-numbers in `stats.zarr`, which produce phantom calendar dates

#### Scenario: Frozen artifacts untouched
- **WHEN** the mini store is built
- **THEN** `patches.zarr`, `stats.zarr`, locked splits, holdout manifest, checkpoints and findings 00–25 are unchanged

### Requirement: Scene-derived values and masks

Every stored value SHALL originate from a real METRIC scene raster. Cloud, cloud-shadow, cirrus, snow and fill pixels SHALL be masked through the Landsat Collection-2 QA_PIXEL bits and recorded in `pixel_valid`; out-of-field pixels SHALL be recorded in `in_field`. No NaN SHALL be silently replaced by zero or a mean.

#### Scenario: Invalid acquisition
- **WHEN** a scene raster or QA_PIXEL is missing or unreadable for a scheduled date
- **THEN** that date is marked with a diagnostic and absent values stay absent rather than being filled

#### Scenario: Mask semantics
- **WHEN** a pixel is marked invalid by QA or lies outside the farm polygon
- **THEN** it is excluded from valid-pixel and valid-patch counts and from derived features

### Requirement: Weather reuse with derived features

The weather store SHALL reuse the aligned weather artifact filtered to the 538 satellite dates by left join, preserving all original columns verbatim. GDD (base 10 °C) and heat-stress hours (hourly temperature above 35 °C) SHALL be added as new columns computed from data available up to each observation date.

#### Scenario: Date without a scene
- **WHEN** a weather date has no corresponding satellite acquisition
- **THEN** it is excluded from the 538-row mini weather table and recorded as a join diagnostic, never as a satellite observation

#### Scenario: Derived versus observed
- **WHEN** GDD or heat-stress hours are exported
- **THEN** they are labelled as derived from reanalysis and never presented as measured field observations

### Requirement: Feature groups for ablation

A versioned feature-group manifest SHALL define spectral-only, weather-core, weather-rolling, weather-all and weather-plus-derived subsets so ablations select a named group without dropping or overwriting raw columns.

#### Scenario: Selecting a group
- **WHEN** a run names a feature group
- **THEN** the resolved column list is recorded in the run manifest and the same list is used for training and evaluation

### Requirement: On-demand spectral indices

Derived spectral indices (NDVI, SAVI, EVI, NDMI, NDWI, NBR) SHALL be computed on demand from stored bands with versioned formulas, at pixel and field level. Only the seven physical bands are persisted. Each index SHALL expose its formula version and input band names in metadata.

#### Scenario: Derived provenance
- **WHEN** an index is exported or used as a feature
- **THEN** it is labelled derived from bands and is never reported as a satellite band or a checkpoint output

### Requirement: Independent validation indicators

Validation indicators (ETa, ETrF, CWSI, TVDI, LAI, METRIC NDVI/LST) SHALL be aligned by (farm, date) into an independent validation store. They SHALL NOT enter any training DataLoader, threshold calibration or model selection by default, and unavailable indicators SHALL be recorded as absent.

#### Scenario: Indicator unavailable
- **WHEN** an indicator has no real artifact for a farm or date
- **THEN** its value is absent with reduced coverage reported, never zero or imputed

#### Scenario: Shared-input disclosure
- **WHEN** an indicator is derived from the same bands the model consumes
- **THEN** the run report discloses the shared input and treats agreement as proxy-only, not diagnostic accuracy

### Requirement: Multi-band phenology and holdout boundary

Sugarcane phases SHALL be derived from a multi-band curve using NDVI, SAVI, LST and GDD. Cycle boundaries and stage confidence SHALL be recorded, and the temporal holdout boundary SHALL be configurable with its date range, sample count and justification written into each run report.

#### Scenario: NDVI-only rejection
- **WHEN** a phase detector is proposed that uses NDVI alone
- **THEN** it is rejected on the recorded finding that NDVI-only detection missed winter harvests and merged seasons

#### Scenario: Partial final year
- **WHEN** a holdout boundary would include the partial 2026 year
- **THEN** the imbalance is documented and the boundary is not hard-coded to a fixed year pair

### Requirement: Configurable split protocols

The harness SHALL support single-farm, leave-one-farm-out and intra-farm temporal protocols as configuration, with no protocol privileged. Normalization statistics, model selection and calibration SHALL be fitted on training data only.

#### Scenario: Split by date
- **WHEN** a temporal protocol is selected
- **THEN** the split is applied to real acquisition dates, not row indices

#### Scenario: Train-only fit
- **WHEN** a normalizer or threshold is fitted
- **THEN** it uses only the training partition and validation/test partitions remain unseen until final scoring

### Requirement: Configurable spatial and temporal representations

The harness SHALL support pixel-level, patch-level (configurable sizes) and full-field spatial representations with valid-patch counts computed from the in-field mask. It SHALL support full-series, sliding-window, seasonal, variable-length and sampled temporal representations while preserving real acquisition date and inter-observation interval.

#### Scenario: Out-of-field patch
- **WHEN** a patch contains fewer than the configured minimum of valid in-field pixels
- **THEN** it is dropped or masked as configured and included in the reported valid-patch count

#### Scenario: Irregular cadence
- **WHEN** observations are separated by unequal day gaps
- **THEN** the interval is retained as a feature and the series is not assumed regular

### Requirement: Model-agnostic architecture registry

The registry SHALL expose MLP-AE, GRU-AE, LSTM-AE, CNN-AE, ConvLSTM-AE, Mamba-AE, CNN-Mamba, Barlow Twins and JEPA through one interface. Reconstruction and representation methods SHALL be evaluated on separate metric tracks. No architecture SHALL be removed or silently replaced because of laptop compute limits.

#### Scenario: Heavy architecture unavailable locally
- **WHEN** an architecture cannot run on the local device
- **THEN** it is marked available for Colab with the recorded reason, and no other model is substituted in its place

#### Scenario: Separate metric tracks
- **WHEN** a representation model such as Barlow Twins is compared with a reconstruction autoencoder
- **THEN** each is reported on its own metric track without ranking them by a single incompatible reconstruction score

### Requirement: Tracking, HPO and provenance

Every run SHALL record MLflow tracking, Optuna studies for HPO, fixed seeds, configuration snapshots, checkpoints with resume support, and a report containing metrics with mean and standard deviation over seeds, sample count, date range, device, wall time, git SHA, data source and split hashes. Controlled comparisons SHALL use equal compute and parameter budgets before HPO.

#### Scenario: Missing required provenance
- **WHEN** a report lacks any required provenance field or contains placeholder values
- **THEN** the run is a defect and is not reported as complete

#### Scenario: Resume
- **WHEN** an interrupted run is restarted with `--resume`
- **THEN** weights, optimizer, scheduler and RNG state are restored and no training step is silently skipped or duplicated

### Requirement: Claim ceiling

Findings from the mini dataset SHALL be tagged at validation-ladder rung L1 or L2 only. Results SHALL NOT be described as stress detection, geographic generalization or baseline superiority.

#### Scenario: Reporting a result
- **WHEN** a mini-dataset result is written
- **THEN** it states its rung, sample count, date range and the controlled-deviation or learning-progress evidence that supports it

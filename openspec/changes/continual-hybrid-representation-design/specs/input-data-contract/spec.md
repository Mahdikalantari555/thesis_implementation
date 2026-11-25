# input-data-contract Specification

## Purpose

Guarantee that every model input from Landsat 8/9 Collection 2 Level-2 is in
its intended physical unit and valid range, consistent across the two
sensors, properly masked, and normalized reproducibly without leakage.
Landsat 7 harmonization is out of scope.

## ADDED Requirements

### Requirement: Pinned scale factors and physical units

The pipeline SHALL convert SR_B2–SR_B7 as DN × 0.0000275 − 0.2 (unitless
reflectance) and ST_B10 as DN × 0.00341802 + 149.0 (Kelvin), treat DN 0 as
fill, treat DNs outside the product valid ranges (SR 7273–43636; ST 293–61440)
as invalid, pin these constants in a unit test against the USGS LSDS-1619
product guide, and SHALL NOT re-apply them to already-converted inputs.

#### Scenario: Raw DNs mixed with converted values
- **WHEN** a feature with values in the thousands is passed as reflectance
- **THEN** the loader rejects it with a range error naming the feature

### Requirement: Feature manifest with units and valid ranges

Every store and model input SHALL carry a feature manifest giving each
feature's name, unit, conversion applied, valid range and source product,
covering bands, derived indices (NDVI, NDWI, NDMI), auxiliary variables (LST,
CWSI, TVDI, ETa, weather) and temporal tokens (Δt, DOY); the loader SHALL
refuse features with no manifest entry or with observed values outside the
range beyond a tolerance.

#### Scenario: Missing manifest entry
- **WHEN** a new feature is added without a manifest entry
- **THEN** training refuses to start

### Requirement: Landsat 8/9 consistency by measurement

Band order and names SHALL be fixed and asserted per scene; the sensor id
SHALL be stored per observation in every store including the mini store; a
sensor offset SHALL be applied only when the near-coincident-pair estimate is
resolved and passes the existing pre-registered acceptance rule, otherwise
measurements SHALL be used without transformation; no-correction, offset
correction and sensor-id token SHALL be compared as experimental arms.

#### Scenario: Unresolved offset
- **WHEN** an offset's CI includes zero
- **THEN** no correction is applied and the decision is logged

### Requirement: Leak-free, anomaly-excluded, robust normalization

Normalization statistics SHALL be computed only from training years and
training farms, only on QA-valid observations not in the anomaly memory's
verified or unresolved partitions; the normalization method SHALL be chosen
among pre-registered arms (global z-score, robust median/MAD or IQR, fixed
physical scaling, robust with logged clipping, per-field centring) on
validation data, with a normalization-erasure test showing that meaningful
magnitude changes remain detectable; statistics SHALL be stored with their
data hash and versioned with the model.

#### Scenario: Test year in statistics
- **WHEN** normalization statistics are computed with any 2024+ observation in scope
- **THEN** computation fails with a leakage error

### Requirement: Missing values and quality masks

Every input SHALL carry a per-feature observation mask derived from QA_PIXEL
(fill, dilated cloud, cirrus, cloud, shadow, snow), saturation (QA_RADSAT) and
range checks; masked entries SHALL be excluded from losses, normalization
statistics and calibration; NaN SHALL never be filled without the mask channel;
windows with fewer than `min_obs` valid observations SHALL be excluded and
counted.

#### Scenario: Cloud-masked observation
- **WHEN** an observation is flagged cloud in QA_PIXEL
- **THEN** its mask is 0 and it contributes to no loss term

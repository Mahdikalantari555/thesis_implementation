## ADDED Requirements

### Requirement: Patch tensor store

The system SHALL provide a canonical per-pixel patch tensor store at
`Data/Processed/satellite/patches.zarr` containing, for every admitted
field-cycle of the locked splits, per-pixel per-bin values for the 8 B1
channels (ndvi + blue, green, red, nir, swir1, swir2, lst) on field-aligned
32×32 patches (pad-to-size, never resampled), the per-pixel per-bin
validity mask, the static in-field polygon mask, bbox origins, per-bin
observation counts and majority sensor ids. Pixel-bins with no valid
acquisition SHALL be stored as NaN and never silently zero-filled; every
consumer SHALL treat them as absent.

#### Scenario: Geometry is field-aligned

- **WHEN** patches are extracted for a field
- **THEN** the patch is the field's bounding box cropped from the scene and
  padded to 32×32, the stored polygon mask excludes pixels of neighbouring
  fields, and no resampling of pixel values occurs

#### Scenario: Invalid pixels stay invalid

- **WHEN** a pixel fails the QA mask (io/landsat.py quality_mask) in every
  acquisition inside a bin
- **THEN** the stored value for that pixel-bin is NaN, pixel_valid is 0,
  and no interpolation or zero-fill is applied anywhere in the pipeline

### Requirement: Leakage-free frozen normalization

The system SHALL compute per-channel, per-sensor robust statistics (median,
IQR) from train-split fields within train periods only (splits_locked
`9aa5eb11…`), freeze them inside the store with a SHA-256 sidecar, and all
downstream consumers SHALL read the frozen artifact. No dev/test field or
period SHALL contribute to these statistics.

#### Scenario: Normalization never sees dev/test statistics

- **WHEN** the normalization artifact is built
- **THEN** only train-split fields inside train periods contribute, the
  artifact records the splits sha and its own sha256, and perturbing a dev
  or test field's values does not change the artifact

### Requirement: Exporter operability

The exporter SHALL support `--smoke` (tiny subset with verified shapes and
counts), `--estimate-runtime` (measured extrapolation printed BEFORE any
full run), and `--resume` (per-date completion manifest; completed dates
skipped), SHALL run on CPU, and SHALL write a provenance manifest (source
products, QA bit list, scale factors, git sha, cycles sha, splits sha,
output sha256). Failures SHALL abort with a non-zero exit and a verbatim
error message — never a partial-success report.

#### Scenario: Smoke before full run

- **WHEN** the agent hands over the full extraction command
- **THEN** a `--smoke` run has been executed and reviewed first (shapes, QA
  fraction plausibility, bin coverage, zero NaN inside valid regions)

#### Scenario: Interrupted extraction resumes

- **WHEN** the exporter is interrupted and relaunched with `--resume`
- **THEN** completed dates are skipped via the manifest and the final store
  equals an uninterrupted run's store for the same inputs

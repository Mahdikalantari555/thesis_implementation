# thesis-inspector Specification

## Purpose

Thesis Inspector provides a local, read-only view of research artifacts, data quality, existing evaluation evidence, and readiness for Platform ingestion. It is independent of the Platform runtime and never changes source artifacts.

## Requirements

### Requirement: Explicit local operation
The Inspector SHALL run as a local application with a static interface, bind only to loopback by default, and perform scans only on startup when requested by CLI or after an explicit user refresh.

#### Scenario: Start without periodic scanning
- **WHEN** the user starts the Inspector in server mode
- **THEN** the application serves its UI and API on loopback and performs no periodic/background scan

#### Scenario: Browser launch and CLI modes
- **WHEN** the user invokes the app with no mode flag, `--refresh`, or `--report-only`
- **THEN** it respectively serves the UI and opens a browser, runs one scan and exits, or generates a report without starting the server

### Requirement: Configurable and safe discovery
The Inspector SHALL discover configured repository and explicitly configured external roots using a versioned manifest that records artifact identity, path, format, research role, schema/version when known, fingerprint, and scan status. It SHALL confine reads to configured roots and SHALL NOT follow hidden paths when configured to ignore them.

#### Scenario: New artifact discovery
- **WHEN** a supported artifact is added under a configured root and a refresh completes
- **THEN** it appears in the artifact inventory with a stable identity, detected format and best-effort research category

#### Scenario: Path outside configured roots
- **WHEN** a request or manifest entry attempts to address a file outside configured roots
- **THEN** the request is rejected without reading that file

#### Scenario: Manifest version incompatibility
- **WHEN** the Inspector encounters an unsupported manifest version
- **THEN** it reports a clear compatibility error and preserves the existing manifest/cache

### Requirement: Incremental refresh and incomplete artifacts
The Inspector SHALL detect changes using file metadata first, avoid full content hashing of large artifacts on every scan, reuse cached inspection results for unchanged artifacts, and represent unreadable or unstable artifacts as incomplete/error entries rather than aborting the scan.

#### Scenario: Unchanged artifact
- **WHEN** a refresh finds the same path, size and modification metadata as the cached fingerprint
- **THEN** the cached adapter/validation result is reused and the artifact is not fully re-read

#### Scenario: Changed artifact
- **WHEN** an artifact's fingerprint metadata changes
- **THEN** only that artifact is re-inspected and its new status is recorded

#### Scenario: File is still being written
- **WHEN** an artifact cannot be opened or fails structural completeness checks during a refresh
- **THEN** it is marked `INCOMPLETE` or `ERROR`, the remaining scan continues, and a later refresh may retry it

#### Scenario: Concurrent refresh requests
- **WHEN** a refresh is already running and another refresh is requested
- **THEN** the second request is rejected or reported as already running and no overlapping scan starts

### Requirement: Format-aware inspection
The Inspector SHALL provide format-specific, bounded inspection for JSON, CSV, NPZ, Zarr, Parquet, GeoJSON, GeoTIFF, and checkpoint artifacts, with optional format dependencies isolated so unsupported readers do not prevent application startup or inspection of other formats.

#### Scenario: Basic formats without optional packages
- **WHEN** the Inspector starts in an environment without optional scientific/geospatial packages
- **THEN** its UI/API and available standard-library readers remain usable and affected artifacts clearly report the missing optional reader

#### Scenario: Large array store
- **WHEN** a user inspects a large Zarr artifact
- **THEN** metadata and selected samples are read lazily within configured bounds rather than materializing the full store

#### Scenario: Safe checkpoint inspection
- **WHEN** a checkpoint artifact is discovered
- **THEN** the Inspector reports safe filesystem/sidecar metadata and does not call `torch.load` on an untrusted `.pt` file

#### Scenario: Corrupt or unknown format
- **WHEN** an artifact is corrupt, truncated, or has an unsupported format
- **THEN** its detail view shows an actionable status/error without crashing the application or hiding other artifacts

### Requirement: Data quality reporting
The Inspector SHALL report structural readability, schema issues, missing values, NaN/Inf, temporal and spatial integrity, identifier consistency, and physical-range violations when the relevant data and checks are available.

#### Scenario: Physical range checks
- **WHEN** channel values and the project channel registry are available
- **THEN** range checks use the existing registry's declared physical envelopes and identify counts/examples without defining competing channel limits

#### Scenario: Missing and non-finite values
- **WHEN** an inspected tabular or array artifact contains missing, NaN, or infinite values
- **THEN** the quality result reports counts and denominators, distinguishing unavailable data from measured zero

#### Scenario: Quality and scientific result remain distinct
- **WHEN** an artifact is structurally valid but its model metrics are poor, or vice versa
- **THEN** the UI and reports present data quality and scientific evaluation as separate states

### Requirement: Existing evaluation and evidence presentation
The Inspector SHALL read existing evaluation artifacts and experiment logs, preserve their protocol/model/run/seed/split provenance, and avoid duplicating metric definitions or recalibrating thresholds. Missing or incomparable results SHALL be shown as `N/A`, not zero.

#### Scenario: Existing dev evaluation
- **WHEN** a supported evaluation JSON and/or `reports/dev_log.csv` entry exists
- **THEN** the Inspector presents its recorded metrics and provenance without rerunning the evaluator

#### Scenario: Incomparable model runs
- **WHEN** runs differ in evaluation protocol, data subset, or other recorded comparison-critical provenance
- **THEN** they are not silently placed in the same ranking and the incompatibility is explained

#### Scenario: Evidence classification
- **WHEN** metrics are shown
- **THEN** synthetic benchmark sensitivity, proxy agreement, and temporal/spatial generalization are labelled as distinct evidence types and none is presented as ground-truth accuracy

### Requirement: Independent benchmark and capstone artifact integration
The Inspector SHALL recognize and read the report/manifest outputs of the architecture benchmark and regional representation capstone as separate evidence lineages, preserving each change's status vocabulary, protocols, provenance, and claim limits without requiring either change to complete.

#### Scenario: Architecture benchmark outputs
- **WHEN** configured artifacts exist under `reports/architecture_benchmark/`
- **THEN** the Inspector reads per-experiment manifests/results and registry states (`Pending`, `Running`, `Completed`, `Failed`, `Interrupted`, `Deferred`, `Not Applicable`), preserving incomplete coverage and recorded reasons rather than treating unfinished cells as missing or zero

#### Scenario: Regional representation outputs
- **WHEN** configured artifacts exist under `reports/regional_representation/`
- **THEN** the Inspector recognizes pretraining-quality, transfer, retrieval, generalization, and decision-gate outputs and retains their task, label-fraction, arm, gate, and claim-rung context where present

#### Scenario: Separate research lineages
- **WHEN** architecture-benchmark results and capstone results are both present
- **THEN** the Inspector displays them as separate studies and does not combine their scores, infer dependencies between them, or rank models across them without a recorded compatible protocol

#### Scenario: Exploratory natural-data retrieval
- **WHEN** a capstone `near_matches` artifact is present
- **THEN** it is labelled descriptive/exploratory and is excluded from metric rankings, gate summaries, and evidence of validated stress

#### Scenario: Locked-test protection
- **WHEN** a benchmark or capstone artifact records a locked test split or test-only metrics
- **THEN** the Inspector may report manifest-level split provenance and the declared access state, but does not open locked test observations or use test metrics for model ranking, threshold selection, or gate recalculation

#### Scenario: Unfinished or unknown report schema
- **WHEN** an output is incomplete or uses an unknown schema version
- **THEN** its raw artifact remains discoverable with an explicit unsupported/incomplete state, and its values are not silently coerced into the canonical metric view

### Requirement: Platform v5 readiness gate
The Inspector SHALL perform a read-only readiness audit against Platform Contract v5 and the existing channel/score registry, returning `PASS`, `WARNING`, or `FAIL` per check and an overall `BLOCKED` state when any critical check fails.

#### Scenario: Registry-backed physical checks
- **WHEN** a candidate Platform export contains channel observations
- **THEN** channel names and physical limits are checked against `scripts/platform_channels.py` rather than a duplicated registry

#### Scenario: Contract and provenance checks
- **WHEN** a candidate payload is available
- **THEN** the audit checks contract version, farm IDs, geometry/CRS, observation dates, required channels, score provenance, stress flags, split tags, checkpoint sidecars, and payload structure where data permits

#### Scenario: Critical failure blocks readiness
- **WHEN** one or more critical readiness checks fail
- **THEN** the overall readiness state is `BLOCKED` and each failure identifies its reason

#### Scenario: Warning without critical failure
- **WHEN** only non-blocking checks warn
- **THEN** readiness is not changed to `BLOCKED` solely due to warnings and each warning remains visible

#### Scenario: No automatic platform mutation
- **WHEN** a readiness audit completes
- **THEN** it only reports findings and does not export, transfer, or modify Platform data

### Requirement: Read-only local API and safe rendering
The Inspector SHALL expose read-only status, inventory, artifact detail, metrics, quality, map, report, and history views through a local API; refresh may update Inspector-owned cache/report state only. Untrusted names and values SHALL be escaped before rendering.

#### Scenario: API reads
- **WHEN** the UI requests status, artifacts, detail, metrics, quality, map, latest report, or history
- **THEN** the API returns only data from configured artifacts or Inspector-owned cache/reports

#### Scenario: Refresh is the only mutation
- **WHEN** an API request other than the explicit refresh endpoint is sent
- **THEN** it cannot modify source artifacts, checkpoints, or Platform exports

#### Scenario: Untrusted artifact labels
- **WHEN** an artifact name contains HTML or script-like text
- **THEN** the UI renders it as inert text and executes no supplied markup or script

### Requirement: Bounded cache and immutable reports
The Inspector SHALL cache metadata and validation results in Inspector-owned storage, keep completed scan history immutable, and bound reads/sampling for resource-constrained local machines.

#### Scenario: Refresh history
- **WHEN** a scan completes
- **THEN** a new timestamped history record is added and previous completed reports remain unchanged

#### Scenario: Resource bounds
- **WHEN** a report samples a large table or array
- **THEN** it obeys configured sample and memory limits and states when statistics are sampled

#### Scenario: No source mutation
- **WHEN** any scan, API request, or report generation runs
- **THEN** input data, checkpoints, and Platform payload files remain byte-for-byte unchanged

### Requirement: Usable local interface and exports
The Inspector SHALL provide overview, data/artifact exploration, model/evaluation, spatial, and QA/report views, with local-only chart/map assets when present and graceful fallback when optional visualization assets or data are unavailable.

#### Scenario: Local offline use
- **WHEN** the application is used without network access
- **THEN** its core UI and data views remain available without CDN requests

#### Scenario: Missing spatial data or visualization library
- **WHEN** map data or a local map/chart asset is unavailable
- **THEN** the affected visualization shows an explicit unavailable state while other views continue to work

#### Scenario: Report export
- **WHEN** the user requests a report export
- **THEN** the Inspector produces a JSON and/or HTML report in its configured output area without overwriting source reports or prior history

### Requirement: Suspected-stress markers on farm index timelines
The Spatial Viewer SHALL plot actual selected-farm index observations on their recorded date axis and mark dates whose existing Platform v5 `stressFlag` is `suspected_stress`, with source-model and split provenance. These markers SHALL NOT be inferred from index thresholds or presented as independently verified stress.

#### Scenario: Render v5 suspected dates
- **WHEN** a selected farm has a readable Contract v5 series and one or more observations with `stressFlag` equal to `suspected_stress`
- **THEN** the timeline plots the registry-defined index channels against the payload's actual dates and overlays distinct markers at those recorded dates

#### Scenario: Explain a marked observation
- **WHEN** the user selects or hovers a suspected-stress marker
- **THEN** the detail identifies the recorded date, `stressFlagModel`, split tag, and channel provenance when present, and labels the status as a model-reported suspected anomaly rather than confirmed stress

#### Scenario: No flag or older payload
- **WHEN** the payload is v4, the selected observation's flag is null/unavailable, or no model-backed flag exists
- **THEN** the Inspector shows an explicit unavailable/no-flag state and does not infer or synthesize a suspected-stress date

#### Scenario: Locked-test dates
- **WHEN** a selected farm payload includes observations tagged `test`
- **THEN** the Inspector excludes those observations and their suspected markers from the timeline and does not expose their index values

#### Scenario: Actual irregular date axis
- **WHEN** a farm's observation dates are irregular or differ in count from other farms
- **THEN** plotted points use the actual stored timestamps and no fixed observation count or regular cadence is assumed

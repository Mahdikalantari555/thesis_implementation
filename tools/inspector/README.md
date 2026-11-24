# Thesis Inspector

A standalone, local-first viewer for research artifact discovery, bounded QA, existing evaluation reports, and Platform Contract v5 readiness. The Inspector reads artifacts; it does not run the pipeline, train/evaluate models, or write to source data, checkpoints, or Platform exports.

## Requirements

- Python 3.11 or newer (the HTTP server, SQLite cache, TOML config, JSON/CSV readers, and static UI use the standard library).
- Scientific readers are optional. The app still starts without them and reports reader availability per artifact.

Optional readers, installed only when needed:

```bash
python -m pip install numpy       # NPZ, array summaries
python -m pip install pyarrow     # Parquet
python -m pip install zarr        # Zarr stores
python -m pip install rasterio    # GeoTIFF
python -m pip install ijson       # bounded farm-series timelines

See `requirements.txt` for the optional dependency list. No Node, React, CDN, cloud service, or database server is required.

## Run

From the repository root:

```bash
python tools/inspector/app.py
```

The app performs one startup scan, binds to `127.0.0.1:8765`, and opens the browser unless disabled in config. It never runs a background watcher. To scan once and exit:

```bash
python tools/inspector/app.py --refresh
```

To scan, write JSON reports, and exit without serving the UI:

```bash
python tools/inspector/app.py --report-only
```

Add `--no-browser` to disable browser launch in server mode. `--config /path/to/config.toml` selects an alternate config. The repository root remains anchored to this app's file location; relative paths in TOML resolve from the repository root, not the current working directory.

## Configuration

Edit `config.toml`. The default configured input roots are `data/`, `checkpoints/`, `reports/`, and `platform/src/data/`. Nonexistent roots are reported and skipped. Add external archives explicitly:

```toml
[paths]
data = "data"
checkpoints = "checkpoints"
reports = "reports"
platform_data = "platform/src/data"
external_roots = [
  { name = "lab_archive", path = "/mnt/research/archive.zarr" },
]

[reporting]
output_dir = "reports/inspector"
```

Inspector-owned SQLite, logs, latest reports, and immutable scan snapshots are written only under `reporting.output_dir`. The default output directory is gitignored. It cannot overlap data/checkpoint roots; the only allowed overlap is the dedicated `reports/inspector/` child under the reports root.

The server rejects non-loopback bind addresses. Keep `read_only = true`. Only the explicit refresh API writes Inspector-owned cache/report state.

## Discovery and safety

Each manifest record contains an opaque ID, configured root name and relative path, format, research-role hint, schema when available, metadata fingerprint, status, and bounded inspection summary. Role classification is heuristic and records the rule used; uncertain artifacts remain `unknown`.

Refresh compares root-relative identity, size, `mtime_ns`, and a content digest for configured small files. Large files and Zarr stores are not fully hashed; Zarr fingerprints include bounded metadata files, never array chunks. Unchanged records reuse cached results. Files that change during inspection, are corrupt, or cannot be opened remain visible as `INCOMPLETE`, `ERROR`, or `UNSUPPORTED` and can be retried on the next manual refresh.

The Inspector uses conservative split isolation. Paths explicitly marked as test-only are metadata-only. Data artifacts under raw/interim/processed roots with unknown split membership are marked `RESTRICTED/UNKNOWN_SPLIT`; test-bearing data is not sampled. Test-only metrics are not ranked or used for gates. If the split cannot be established, the Inspector does not expose value-level inspection by default.

Checkpoint weights are never deserialized. `.pt` files are not passed to PyTorch; only filesystem metadata and bounded JSON sidecars are inspected. HTML is rendered using DOM text nodes, and API artifact IDs do not accept arbitrary paths.

## Supported formats

| Format | Behavior |
|---|---|
| JSON / JSONL | Bounded structure, schema/provenance hints, selected scalar summaries |
| CSV | Bounded row sample, headers, missingness and numeric summaries; channel limits read from `scripts/platform_channels.py` |
| NPZ | Optional NumPy, `allow_pickle=False`, uncompressed-size cap, shapes/dtypes and bounded sample statistics |
| Parquet | Optional PyArrow metadata and bounded batch sample |
| Zarr | Optional Zarr metadata and lazy bounded sample reads; no array materialization |
| GeoJSON | Bounded feature/geometry checks and coordinate-envelope validation |
| GeoTIFF | Optional Rasterio metadata only (CRS, transform, nodata, dimensions, bands) |
| PT / SafeTensors | Sidecar and file metadata only; no untrusted model loading |
| YAML / TOML / text | Bounded text preview; no code execution |

Missing optional packages produce an explicit `UNSUPPORTED` state; they do not block startup.

## Evaluation lineages

The Model Lab keeps the evidence streams separate:

- `reports/dev/` and `reports/dev_log.csv`: existing evaluator outputs only.
- `reports/architecture_benchmark/`: architecture/suppression experiments, preserving per-cell states such as `Deferred` and `Not Applicable`.
- `reports/regional_representation/`: pretraining, transfer, retrieval, generalization, and decision-gate outputs; `near_matches` remains exploratory.

The broad benchmark and regional capstone use different protocols and are not combined into cross-study rankings. Metric definitions are not reimplemented. Synthetic benchmark sensitivity, proxy agreement, and temporal/spatial generalization are distinct evidence categories; none is ground-truth accuracy. The Inspector never launches benchmark/capstone runs and does not feed results into their gates.

## API

The local API is same-origin and read-only except for `POST /api/refresh`, which performs a single serialized scan and writes only Inspector-owned state.

- `GET /api/status`
- `POST /api/refresh`
- `GET /api/artifacts`
- `GET /api/artifact/{opaque-id}`
- `GET /api/metrics`
- `GET /api/quality`
- `GET /api/map`
- `GET /api/series?farm_id=F-001` — optional `ijson`-backed, bounded per-farm series; excludes test observations and only marks recorded v5 suspected-stress flags
- `GET /api/report/latest`
- `GET /api/report/latest.html` — self-contained escaped HTML report
- `GET /api/history`

No route serves arbitrary filesystem paths. A refresh arriving during an active scan returns HTTP 409.

## Inspector-owned outputs

```text
reports/inspector/
├── inspector.sqlite
├── inspector.log
├── latest/{overview,artifacts,quality,evaluation,integration}.json
├── latest/overview.html
└── history/<scan-id>/
    ├── {overview,artifacts,quality,evaluation,integration}.json
    └── overview.html
```

History snapshots are immutable; `latest/` is only a replaceable convenience view. Deleting `reports/inspector/` resets the cache and history without changing research data.

## Current limitations

- Discovery classification is best-effort and does not replace a formal pipeline manifest.
- Unknown-split artifacts are intentionally restricted, which can reduce detail until their manifest/provenance is configured.
- Optional Zarr/Parquet/NPZ/GeoTIFF paths are only exercised when the corresponding packages and fixtures are installed; unsupported readers remain visible.
- The spatial page draws available farm polygons from the configured farm registry using local SVG, not a network basemap. Selecting a farm requests its series on demand through a bounded streaming reader (`ijson` optional). Timeline x-coordinates use the payload's actual acquisition dates; index values and provenance come from registry channels. Only recorded v5 `suspected_stress` flags are marked, with flag model, split, and channel provenance shown. Test-split values, provenance, and markers are excluded before the API response is built. v4, missing flags, missing series, and unavailable streaming support have explicit fallback states; no stress label is inferred from index values.
- Platform readiness requires the farm registry and a readable v5 series payload. Missing series or optional `ijson` support yields an explicit warning/unevaluated result, never a PASS. The v5 audit validates registry IDs/geometry, actual dates, split-first structure, registry channel limits, provenance, and flags; it never loads checkpoint weights or evaluates locked-test values.

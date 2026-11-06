# Spec: Platform Visualization — RGB/NDVI Tile Layer

## Purpose

Defines how the platform map renders full-extent RGB and NDVI raster overlays from Landsat GeoTIFF scenes via a local tile serving layer.

## ADDED Requirements

### Requirement: Scene index manifest

The system SHALL generate `platform/src/data/scenes_index.json` listing available Landsat scenes with date, path, bounds, and band availability.

#### Scenario: Scene index structure
- **WHEN** `scripts/build_tile_index.py` scans `Data/Metric/scenes/`
- **THEN** it emits a JSON array of scenes, each with `date`, `path`, `bounds`, `bands`, `has_rgb`, `has_ndvi`

### Requirement: Tile serving

The system SHALL serve RGB and NDVI tiles per acquisition date from a local tile server, reprojected to Web Mercator (EPSG:3857) for Leaflet.

#### Scenario: Tile request
- **WHEN** the frontend requests a tile for date `2025-04-17` at zoom/level
- **THEN** the tile server returns a valid PNG tile from the corresponding GeoTIFF scene

### Requirement: Frontend tile layer controls

The system SHALL add scene/date selector, layer toggle (None/RGB/NDVI), and opacity control to `MapExplorer.tsx`.

#### Scenario: Layer toggle
- **WHEN** user selects "NDVI" layer
- **THEN** the map displays NDVI raster overlay with selected opacity; field polygons remain visible

### Requirement: Graceful degradation

The system SHALL render field polygons without raster overlay when the tile server is unavailable.

#### Scenario: Tile server down
- **WHEN** the local tile server is not running
- **THEN** the platform still renders field polygons and time series; tile layer fails silently

## Out of Scope

- MBTiles prebuilding
- Production tile server infrastructure
- Real model outputs

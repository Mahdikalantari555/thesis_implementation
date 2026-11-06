# Tasks — Add RGB/NDVI Tile Serving Layer

## 1. Scene index & tile pipeline
- [ ] 1.1 Implement `scripts/build_tile_index.py` — scan `Data/Metric/scenes/`, parse MTL.json for date/bounds/bands, emit `platform/src/data/scenes_index.json`
- [ ] 1.2 Implement `scripts/serve_tiles.py` — lightweight HTTP tile server; open GeoTIFFs with `rasterio`, render 256×256 RGB/NDVI tiles, reproject to Web Mercator
- [ ] 1.3 Verify tile server responds with valid PNG tiles for at least one RGB and one NDVI scene

## 2. Platform integration
- [ ] 2.1 Add scene/date selector dropdown to `MapExplorer.tsx` (populated from `scenes_index.json`)
- [ ] 2.2 Add layer toggle (None / RGB / NDVI) + opacity slider (0–1) to map controls
- [ ] 2.3 Wire Leaflet tile layer to local tile server URL; conditional load (degrades gracefully if server unavailable)
- [ ] 2.4 Verify map renders field polygons with optional RGB/NDVI overlay

## 3. Tests & verification
- [ ] 3.1 Add unit test for scene index generation (validates JSON structure, band flags, date parsing)
- [ ] 3.2 Run `tsc --noEmit` → 0 errors
- [ ] 3.3 Build frontend (`bun run build` or `node node_modules/vite/bin/vite.js build`) → 0 errors

## 4. Docs & commit
- [ ] 4.1 Add tile layer spec to `openspec/changes/add-rgb-ndvi-tile-layer/specs/platform-visualization/spec.md`
- [ ] 4.2 Commit all changes on current branch

import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { FIELDS, getSeries, clampIdx, KHUZESTAN_CENTROID, type ModelId, type IndicatorId } from "../data/mock";
import { useAnalysis } from "../context/AnalysisContext";
import { useT } from "../i18n";
import { anomalyColor } from "./ui";

interface Props {
  colorMode: "anomaly" | "indicator" | "health" | "lisa";
  model: ModelId;
  indicator: IndicatorId;
  dateIdx: number;
  opacity: number;
  height?: string;
}

const TILE_DARK = "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png";
const TILE_LIGHT = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const ATTRIBUTION = "© OpenStreetMap © CARTO";

export default function MapView({ colorMode, model, indicator, dateIdx, opacity, height = "100%" }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const tileRef = useRef<L.TileLayer | null>(null);
  const { selectedField, toggleField, dark } = useAnalysis();
  const t = useT();

  /**
   * Map lifecycle — mount once.
   *
   * BUG-04 (docs/platform_bug_audit.md): this effect previously depended on
   * `dark`, so every theme toggle destroyed and re-created the Leaflet map,
   * resetting zoom/center to the Khuzestan default. The effect now runs once
   * and the *tile layer* is swapped by the effect below, preserving the view.
   */
  useEffect(() => {
    if (!ref.current || mapRef.current) return;
    const map = L.map(ref.current, { zoomControl: true, attributionControl: true }).setView(KHUZESTAN_CENTROID, 12);
    mapRef.current = map;
    layerRef.current = L.layerGroup().addTo(map);
    return () => {
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
      tileRef.current = null;
    };
  }, []);

  /** Theme → tile layer only (no map recreation). */
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    if (tileRef.current) {
      map.removeLayer(tileRef.current);
      tileRef.current = null;
    }
    tileRef.current = L.tileLayer(dark ? TILE_DARK : TILE_LIGHT, { attribution: ATTRIBUTION }).addTo(map);
  }, [dark]);

  /** Field polygons — redrawn when a control changes, never when the theme does. */
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer) return;
    layer.clearLayers();
    const idx = clampIdx(dateIdx);
    const scoreLabel = t("map.tooltip.score");
    const haLabel = t("map.tooltip.ha");

    FIELDS.forEach(f => {
      const series = getSeries(f.id);
      const obs = series[idx];
      // A field without a series still renders, but is greyed out and says so.
      let fill = "#94a3b8";
      let tooltip = `<b>${f.code}</b> · ${f.name}<br/>${f.areaHa} ${haLabel} · ${f.municipality}`;
      if (obs) {
        if (colorMode === "anomaly") fill = anomalyColor(obs.scores[model]);
        else if (colorMode === "indicator") {
          const v = (obs as unknown as Record<string, number>)[indicator];
          const norm = indicator === "ndvi" || indicator === "evi" ? v : indicator === "tvdi" || indicator === "cwsi" ? v : Math.max(0, Math.min(1, v));
          fill = norm > 0.65 ? "#be123c" : norm > 0.5 ? "#ea580c" : norm > 0.35 ? "#d97706" : "#0f766e";
          if (indicator === "ndvi" || indicator === "evi" || indicator === "savi") fill = norm > 0.6 ? "#15803d" : norm > 0.45 ? "#65a30d" : norm > 0.3 ? "#d97706" : "#be123c";
        } else if (colorMode === "health") fill = f.baselineHealth > 0.8 ? "#15803d" : f.baselineHealth > 0.65 ? "#65a30d" : "#d97706";
        else if (colorMode === "lisa") {
          const neighbourScores = f.neighbors.map(nid => getSeries(nid)[idx]?.scores[model]).filter((v): v is number => typeof v === "number");
          const nbAvg = neighbourScores.reduce((s, v) => s + v, 0) / Math.max(1, neighbourScores.length);
          const self = obs.scores[model];
          const hi = self > 0.5, nhi = nbAvg > 0.5;
          fill = hi && nhi ? "#be123c" : hi && !nhi ? "#2563eb" : !hi && nhi ? "#7c3aed" : "#94a3b8";
        }
        tooltip += `<br/>${scoreLabel}: ${obs.scores[model].toFixed(3)}`;
      }

      if (!f.polygon.length) return;
      const isSel = f.id === selectedField;
      const poly = L.polygon(f.polygon, {
        color: isSel ? "#0f172a" : "#ffffff",
        weight: isSel ? 2.5 : 1.2,
        fillColor: fill,
        fillOpacity: opacity * (isSel ? 1 : 0.82),
      }).addTo(layer);
      poly.bindTooltip(tooltip, { sticky: true, direction: "auto" });
      poly.on("click", () => toggleField(f.id));
      if (isSel) {
        L.circleMarker(f.centroid, { radius: 5, color: "#0f172a", fillColor: "#fff", fillOpacity: 1, weight: 2 }).addTo(layer);
      }
    });
  }, [colorMode, model, indicator, dateIdx, opacity, selectedField, toggleField, t]);

  return <div ref={ref} dir="ltr" style={{ height, width: "100%", zIndex: 0 }} className="bg-slate-100 dark:bg-slate-800" />;
}

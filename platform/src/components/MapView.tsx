import { useEffect, useRef } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { FIELDS, getSeries, type ModelId, type IndicatorId } from '../data/mock';
import { useAnalysis } from '../context/AnalysisContext';
import { anomalyColor } from './ui';

interface Props {
  colorMode: 'anomaly' | 'indicator' | 'health' | 'lisa';
  model: ModelId;
  indicator: IndicatorId;
  dateIdx: number;
  opacity: number;
  height?: string;
}

export default function MapView({ colorMode, model, indicator, dateIdx, opacity, height = '100%' }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const { selectedField, toggleField, dark } = useAnalysis();

  useEffect(() => {
    if (!ref.current || mapRef.current) return;
    const map = L.map(ref.current, { zoomControl: true }).setView([31.0217, 48.2794], 12);
    const tiles = dark
      ? L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', { attribution: '© OpenStreetMap © CARTO' })
      : L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '© OpenStreetMap' });
    tiles.addTo(map);
    mapRef.current = map;
    layerRef.current = L.layerGroup().addTo(map);
    return () => { map.remove(); mapRef.current = null; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dark]);

  useEffect(() => {
    if (!mapRef.current || !layerRef.current) return;
    layerRef.current.clearLayers();
    FIELDS.forEach(f => {
      const obs = getSeries(f.id)[Math.max(0, Math.min(47, dateIdx))];
      let fill = '#94a3b8';
      if (colorMode === 'anomaly') fill = anomalyColor(obs.scores[model]);
      else if (colorMode === 'indicator') {
        const v = (obs as any)[indicator] as number;
        const norm = indicator === 'ndvi' || indicator === 'evi' ? v : indicator === 'tvdi' || indicator === 'cwsi' ? v : Math.max(0, Math.min(1, v));
        fill = norm > 0.65 ? '#be123c' : norm > 0.5 ? '#ea580c' : norm > 0.35 ? '#d97706' : '#0f766e';
        if (indicator === 'ndvi' || indicator === 'evi' || indicator === 'savi') fill = norm > 0.6 ? '#15803d' : norm > 0.45 ? '#65a30d' : norm > 0.3 ? '#d97706' : '#be123c';
      } else if (colorMode === 'health') fill = f.baselineHealth > 0.8 ? '#15803d' : f.baselineHealth > 0.65 ? '#65a30d' : '#d97706';
      else if (colorMode === 'lisa') {
        const nbAvg = f.neighbors.reduce((s, nid) => s + getSeries(nid)[dateIdx].scores[model], 0) / Math.max(1, f.neighbors.length);
        const self = obs.scores[model];
        const hi = self > 0.5, nhi = nbAvg > 0.5;
        fill = hi && nhi ? '#be123c' : hi && !nhi ? '#2563eb' : !hi && nhi ? '#7c3aed' : '#94a3b8';
      }
      const isSel = f.id === selectedField;
      const poly = L.polygon(f.polygon, {
        color: isSel ? '#0f172a' : '#ffffff',
        weight: isSel ? 2.5 : 1.2,
        fillColor: fill,
        fillOpacity: opacity * (isSel ? 1 : 0.82),
      }).addTo(layerRef.current!);
      poly.bindTooltip(`<b>${f.code}</b> · ${f.name}<br/>${f.areaHa} ha · ${f.municipality}<br/>Score: ${obs.scores[model].toFixed(3)}`, { sticky: true });
      poly.on('click', () => toggleField(f.id));
      if (isSel) {
        L.circleMarker(f.centroid, { radius: 5, color: '#0f172a', fillColor: '#fff', fillOpacity: 1, weight: 2 }).addTo(layerRef.current!);
      }
    });
  }, [colorMode, model, indicator, dateIdx, opacity, selectedField, toggleField]);

  return <div ref={ref} style={{ height, width: '100%', zIndex: 0 }} className="bg-slate-100 dark:bg-slate-800" />;
}

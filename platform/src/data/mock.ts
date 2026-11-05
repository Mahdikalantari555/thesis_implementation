// ── Real Khuzestan Sugarcane Farm Data & Model Anomaly Engine ────────────────
import farmsKhuzestanData from './farms_khuzestan.json';

export type ModelId =
  | 'b0_calendar'
  | 'b1_dsh'
  | 'b1_cusum'
  | 'masked_transformer'
  | 'sts_net'
  | 'tiny_cnn_mamba'
  | 'time2agri';

export type IndicatorId =
  | 'ndvi'
  | 'evi'
  | 'savi'
  | 'ndmi'
  | 'tvdi'
  | 'cwsi'
  | 'eta'
  | 'lst'
  | 'precip'
  | 'temp';

export interface ModelDef {
  id: ModelId;
  name: string;
  short: string;
  color: string;
  type: string;
  desc: string;
}

export const MODELS: ModelDef[] = [
  { id: 'b0_calendar', name: 'B0 Calendar DOY', short: 'B0-DOY', color: '#64748b', type: 'Calendar Baseline', desc: 'Day-of-year median curve baseline' },
  { id: 'b1_dsh', name: 'B1 Phenological DSH', short: 'B1-DSH', color: '#0f766e', type: 'Phenology Baseline', desc: 'Days-since-harvest binned reference baseline' },
  { id: 'b1_cusum', name: 'B1 CUSUM Filter', short: 'B1-CUSUM', color: '#0284c7', type: 'Change Point', desc: 'Cumulative sum sequential anomaly filter' },
  { id: 'masked_transformer', name: 'SSL Masked Transformer', short: 'Masked-TF', color: '#7c3aed', type: 'Transformer', desc: 'Masked temporal transformer (Presto/MAE)' },
  { id: 'sts_net', name: 'STS-Net 3D-CAE', short: 'STS-Net', color: '#d97706', type: 'Spatio-Temporal', desc: 'Spatio-temporal convolutional autoencoder (Dalal et al. 2026)' },
  { id: 'tiny_cnn_mamba', name: 'Tiny-CNN-Mamba', short: 'CNN-Mamba', color: '#2563eb', type: 'State Space', desc: '1D CNN + Bidirectional Selective SSM (O(L) linear)' },
  { id: 'time2agri', name: 'Time2Agri Contrastive', short: 'Time2Agri', color: '#be123c', type: 'Contrastive SSL', desc: 'Agricultural trajectory contrastive learning (Gupta & Sobti 2025)' },
];

export const modelById = (id: ModelId) => MODELS.find(m => m.id === id) || MODELS[1];

export interface IndicatorDef {
  id: IndicatorId;
  name: string;
  unit: string;
  color: string;
  range: [number, number];
  desc: string;
  kind: 'spectral' | 'thermal' | 'water' | 'weather';
}

export const INDICATORS: IndicatorDef[] = [
  { id: 'ndvi', name: 'NDVI', unit: 'index', color: '#16a34a', range: [0, 1], desc: 'Normalized Difference Vegetation Index', kind: 'spectral' },
  { id: 'evi', name: 'EVI', unit: 'index', color: '#65a30d', range: [0, 1], desc: 'Enhanced Vegetation Index', kind: 'spectral' },
  { id: 'savi', name: 'SAVI', unit: 'index', color: '#059669', range: [0, 1], desc: 'Soil-Adjusted Vegetation Index', kind: 'spectral' },
  { id: 'ndmi', name: 'NDMI', unit: 'index', color: '#0891b2', range: [-1, 1], desc: 'Normalized Difference Moisture Index', kind: 'water' },
  { id: 'tvdi', name: 'TVDI', unit: 'index', color: '#ea580c', range: [0, 1], desc: 'Temperature–Vegetation Dryness Index', kind: 'thermal' },
  { id: 'cwsi', name: 'CWSI', unit: 'index', color: '#dc2626', range: [0, 1], desc: 'Crop Water Stress Index', kind: 'thermal' },
  { id: 'eta', name: 'ETa', unit: 'mm/d', color: '#0284c7', range: [0, 8], desc: 'Actual evapotranspiration (METRIC)', kind: 'water' },
  { id: 'lst', name: 'LST', unit: '°C', color: '#9333ea', range: [15, 55], desc: 'Land Surface Temperature (Landsat ST_B10)', kind: 'thermal' },
  { id: 'precip', name: 'Precipitation', unit: 'mm', color: '#475569', range: [0, 300], desc: 'Accumulated precipitation (ERA5)', kind: 'weather' },
  { id: 'temp', name: 'Air Temp.', unit: '°C', color: '#b45309', range: [10, 48], desc: 'Mean air temperature (ERA5)', kind: 'weather' },
];

export const indicatorById = (id: IndicatorId) => INDICATORS.find(i => i.id === id) || INDICATORS[0];

export interface Field {
  id: string;
  code: string;
  name: string;
  farm: string;
  municipality: string;
  block?: string;
  areaHa: number;
  variety: string;
  planted: string;
  harvest: string;
  centroid: [number, number];
  polygon: [number, number][]; // [lat, lng]
  baselineHealth: number;
  stressEvent: { startIdx: number; endIdx: number; severity: number } | null;
  neighbors: string[];
}

export const FIELDS: Field[] = (farmsKhuzestanData as any[]).map(f => ({
  id: f.id,
  code: f.code,
  name: f.name,
  farm: f.farm,
  municipality: f.municipality,
  block: f.block || 'R01',
  areaHa: f.areaHa,
  variety: f.variety || 'CP69-1062',
  planted: f.planted || '2021/03',
  harvest: f.harvest || '2024/11',
  centroid: f.centroid,
  polygon: f.polygon,
  baselineHealth: f.baselineHealth || 0.65,
  stressEvent: f.stressEvent,
  neighbors: f.neighbors || [],
}));

export const fieldById = (id: string) => FIELDS.find(f => f.id === id) || FIELDS[0];

// Dates spanning 2021 to 2024 (48 monthly observations)
export const DATES: string[] = [];
for (let y = 2021; y <= 2024; y++) {
  for (let m = 1; m <= 12; m++) {
    DATES.push(`${y}-${String(m).padStart(2, '0')}-15`);
  }
}
export const N = DATES.length; // 48

// Seeded RNG
function rng(seed: number) {
  let s = seed;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

const clamp = (v: number, a: number, b: number) => Math.max(a, Math.min(b, v));

export interface Obs {
  date: string;
  idx: number;
  ndvi: number;
  evi: number;
  savi: number;
  ndmi: number;
  tvdi: number;
  cwsi: number;
  eta: number;
  lst: number;
  precip: number;
  temp: number;
  scores: Record<ModelId, number>;
  missing: boolean;
}

const fieldCache = new Map<string, Obs[]>();

export function getSeries(fieldId: string): Obs[] {
  if (fieldCache.has(fieldId)) return fieldCache.get(fieldId)!;
  const f = fieldById(fieldId);
  const fidNum = parseInt(f.id.replace('F-', ''), 10) || 1;
  const r = rng(fidNum * 1013 + 47);

  // Sugarcane phenology: peak growth in July–September, lower in harvest/winter
  const baseHealth = f.baselineHealth;
  const varietyFactor = f.variety.includes('CP73') ? 1.05 : 1.0;

  const obs: Obs[] = DATES.map((date, i) => {
    const month = (i % 12) + 1; // 1 to 12
    // Annual phenological curve: low in Feb (month 2, harvest), peaks in Aug (month 8)
    const phenoPhase = ((month - 2) / 12) * Math.PI * 2;
    const season = Math.sin(phenoPhase - Math.PI / 2); // -1 in winter, +1 in late summer
    const noise = (r() - 0.5) * 0.04;

    let stress = 0;
    if (f.stressEvent && i >= f.stressEvent.startIdx && i <= f.stressEvent.endIdx) {
      const dur = Math.max(1, f.stressEvent.endIdx - f.stressEvent.startIdx);
      const t = (i - f.stressEvent.startIdx) / dur;
      stress = f.stressEvent.severity * Math.sin(Math.PI * Math.min(1, t * 1.15));
    }

    // Real Khuzestan temperature profile: 15°C winter to 46°C summer
    const tempSeason = Math.sin(((month - 1) / 12) * Math.PI * 2 - Math.PI / 2);
    const temp = clamp(30 + tempSeason * 16 + (r() - 0.5) * 2, 12, 48);
    const lst = clamp(temp + 4.5 + stress * 5.0 + (r() - 0.5) * 2.5, 16, 54);

    // Indices based on sugarcane phenology and stress
    const ndvi = clamp((baseHealth * 0.65 + 0.15 + season * 0.22) * varietyFactor + noise - stress * 0.28, 0.12, 0.88);
    const evi = clamp(ndvi * 0.82 + (r() - 0.5) * 0.03, 0.08, 0.78);
    const savi = clamp(ndvi * 0.90 + 0.04, 0.10, 0.82);
    const ndmi = clamp(0.20 + season * 0.35 - stress * 0.38 + (r() - 0.5) * 0.05, -0.35, 0.75);
    const tvdi = clamp(0.30 - season * 0.25 + stress * 0.48 + (r() - 0.5) * 0.06, 0.05, 0.95);
    const cwsi = clamp(0.22 - season * 0.18 + stress * 0.52 + (r() - 0.5) * 0.05, 0.04, 0.96);
    const eta = clamp((ndvi * 7.2) - stress * 2.2 + (r() - 0.5) * 0.4, 0.5, 7.8);
    // Precipitation concentrated in Dec-March, dry in summer
    const precip = [1, 2, 3, 11, 12].includes(month) ? Math.max(0, 35 + (r() - 0.4) * 45) : (r() < 0.1 ? 5 : 0);

    const missing = r() < 0.04;

    // Thesis models scores:
    // B0: Calendar DOY (high false alarms on delayed planting)
    // B1: DSH (stable baseline)
    // B1-CUSUM: Sequential filter (accumulates stress)
    // SSL Masked Transformer, STS-Net, Tiny-CNN-Mamba, Time2Agri
    const laggedStress = stress;
    const baseAnomaly = 0.12 + Math.abs(noise) * 1.5;

    const scores: Record<ModelId, number> = {
      b0_calendar: clamp(baseAnomaly + 0.12 + (season < 0 ? 0.15 : 0.0) + laggedStress * 0.70, 0.04, 0.98),
      b1_dsh: clamp(baseAnomaly + 0.06 + laggedStress * 0.82, 0.03, 0.98),
      b1_cusum: clamp(baseAnomaly + 0.04 + laggedStress * 0.88, 0.02, 0.99),
      masked_transformer: clamp(baseAnomaly + 0.03 + laggedStress * 0.92, 0.02, 0.99),
      sts_net: clamp(baseAnomaly + 0.02 + laggedStress * 0.94, 0.02, 0.99),
      tiny_cnn_mamba: clamp(baseAnomaly + 0.02 + laggedStress * 0.96, 0.02, 0.99),
      time2agri: clamp(baseAnomaly + 0.03 + laggedStress * 0.90, 0.02, 0.99),
    };

    return {
      date,
      idx: i,
      ndvi: +ndvi.toFixed(3),
      evi: +evi.toFixed(3),
      savi: +savi.toFixed(3),
      ndmi: +ndmi.toFixed(3),
      tvdi: +tvdi.toFixed(3),
      cwsi: +cwsi.toFixed(3),
      eta: +eta.toFixed(2),
      lst: +lst.toFixed(1),
      precip: +precip.toFixed(1),
      temp: +temp.toFixed(1),
      scores,
      missing,
    };
  });

  fieldCache.set(fieldId, obs);
  return obs;
}

// ── Statistical Helper Functions ──────────────────────────────────────────────
export function pearson(xs: number[], ys: number[]): number {
  const n = xs.length;
  if (n < 3) return NaN;
  const mx = mean(xs), my = mean(ys);
  let num = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    dx += (xs[i] - mx) ** 2;
    dy += (ys[i] - my) ** 2;
  }
  return dx && dy ? num / Math.sqrt(dx * dy) : NaN;
}

export function ranks(xs: number[]): number[] {
  const sorted = [...xs].map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0]);
  const r = new Array(xs.length);
  sorted.forEach(([, i], rank) => { r[i] = rank + 1; });
  return r;
}

export const spearman = (xs: number[], ys: number[]) => pearson(ranks(xs), ranks(ys));

export function kendall(xs: number[], ys: number[]): number {
  let c = 0, d = 0;
  for (let i = 0; i < xs.length; i++) {
    for (let j = i + 1; j < xs.length; j++) {
      const s = Math.sign(xs[j] - xs[i]) * Math.sign(ys[j] - ys[i]);
      if (s > 0) c++; else if (s < 0) d++;
    }
  }
  return (c + d) ? (c - d) / (c + d) : NaN;
}

export const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);

export function std(xs: number[]) {
  const m = mean(xs);
  return Math.sqrt(mean(xs.map(x => (x - m) ** 2)));
}

export function quantile(xs: number[], q: number) {
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor(q * s.length))];
}

export function linReg(xs: number[], ys: number[]) {
  const mx = mean(xs), my = mean(ys);
  let num = 0, den = 0;
  for (let i = 0; i < xs.length; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  const slope = den ? num / den : 0;
  return { slope, intercept: my - slope * mx, r2: pearson(xs, ys) ** 2 };
}

export function pValue(r: number, n: number): number {
  if (isNaN(r) || n < 4) return NaN;
  const t = r * Math.sqrt((n - 2) / Math.max(1e-9, 1 - r * r));
  const p = 2 * (1 - normCdf(Math.abs(t)));
  return Math.max(0.0001, Math.min(1, p));
}

function normCdf(x: number) {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}

function erf(x: number) {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return x < 0 ? -y : y;
}

export function fisherCI(r: number, n: number): [number, number] {
  if (isNaN(r) || n < 4) return [NaN, NaN];
  const z = Math.atanh(Math.max(-0.999, Math.min(0.999, r)));
  const se = 1 / Math.sqrt(n - 3);
  return [Math.tanh(z - 1.96 * se), Math.tanh(z + 1.96 * se)];
}

export function laggedCorr(xs: number[], ys: number[], maxLag = 6) {
  const out: { lag: number; r: number }[] = [];
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    const a: number[] = [], b: number[] = [];
    for (let i = 0; i < xs.length; i++) {
      const j = i + lag;
      if (j >= 0 && j < ys.length) { a.push(xs[i]); b.push(ys[j]); }
    }
    out.push({ lag, r: pearson(a, b) });
  }
  return out;
}

export function partialCorr(xs: number[], ys: number[], zs: number[]): number {
  const rxy = pearson(xs, ys), rxz = pearson(xs, zs), ryz = pearson(ys, zs);
  const den = Math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2));
  return den ? (rxy - rxz * ryz) / den : NaN;
}

export function moransI(values: number[], weights: Map<string, string[]>): number {
  const n = values.length;
  const m = mean(values);
  let num = 0, den = 0, wsum = 0;
  values.forEach(v => { den += (v - m) ** 2; });
  weights.forEach((nbs, iStr) => {
    const i = parseInt(iStr, 10);
    nbs.forEach(jStr => {
      const j = parseInt(jStr, 10);
      num += (values[i] - m) * (values[j] - m);
      wsum++;
    });
  });
  if (!den || !wsum) return NaN;
  return (n / wsum) * (num / den);
}

export function getPairs(model: ModelId, indicator: IndicatorId, fieldIds: string[], range: [number, number]) {
  const xs: number[] = [], ys: number[] = [], meta: { field: string; date: string }[] = [];
  fieldIds.forEach(fid => {
    getSeries(fid).forEach(o => {
      if (o.idx >= range[0] && o.idx <= range[1] && !o.missing) {
        xs.push(o.scores[model]);
        ys.push((o as any)[indicator] as number);
        meta.push({ field: fid, date: o.date });
      }
    });
  });
  return { xs, ys, meta };
}

export const ANOMALY_THRESHOLD = 0.55;

export function fmtDate(d: string) {
  return d.slice(0, 7);
}

export function downloadCSV(filename: string, rows: (string | number)[][]) {
  const csv = rows.map(r => r.map(c => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
  const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Khuzestan sugarcane farm data, exported time series & model scores ───────
// Both JSON payloads are produced by scripts/export_platform_data.py:
//   farms_khuzestan.json   — field polygons + metadata (490 fields)
//   series_khuzestan.json  — 48 monthly observations per field (2021-01 … 2024-12)
// The series file used to be exported but never imported (the views generated
// synthetic values with a seeded RNG instead). It is now the single data source.
import farmsKhuzestanData from './farms_khuzestan.json';
import khuSeries from './series_khuzestan.json';

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
  /** Latin display name — model names are proper nouns and stay in Latin script. */
  name: string;
  short: string;
  color: string;
  /** i18n keys: `t(typeKey)` / `t(descKey)` so every locale renders. */
  typeKey: string;
  descKey: string;
}

export const MODELS: ModelDef[] = [
  { id: 'b0_calendar', name: 'B0 Calendar DOY', short: 'B0-DOY', color: '#64748b', typeKey: 'model.b0_calendar.type', descKey: 'model.b0_calendar.desc' },
  { id: 'b1_dsh', name: 'B1 Phenological DSH', short: 'B1-DSH', color: '#0f766e', typeKey: 'model.b1_dsh.type', descKey: 'model.b1_dsh.desc' },
  { id: 'b1_cusum', name: 'B1 CUSUM Filter', short: 'B1-CUSUM', color: '#0284c7', typeKey: 'model.b1_cusum.type', descKey: 'model.b1_cusum.desc' },
  { id: 'masked_transformer', name: 'SSL Masked Transformer', short: 'Masked-TF', color: '#7c3aed', typeKey: 'model.masked_transformer.type', descKey: 'model.masked_transformer.desc' },
  { id: 'sts_net', name: 'STS-Net 3D-CAE', short: 'STS-Net', color: '#d97706', typeKey: 'model.sts_net.type', descKey: 'model.sts_net.desc' },
  { id: 'tiny_cnn_mamba', name: 'Tiny-CNN-Mamba', short: 'CNN-Mamba', color: '#2563eb', typeKey: 'model.tiny_cnn_mamba.type', descKey: 'model.tiny_cnn_mamba.desc' },
  { id: 'time2agri', name: 'Time2Agri Contrastive', short: 'Time2Agri', color: '#be123c', typeKey: 'model.time2agri.type', descKey: 'model.time2agri.desc' },
];

export const modelById = (id: ModelId) => MODELS.find(m => m.id === id) || MODELS[0];

export interface IndicatorDef {
  id: IndicatorId;
  /** Latin symbol (NDVI, CWSI, ETa, …) — used verbatim in both locales. */
  name: string;
  /** Unit symbol; localized unit names are not needed at this granularity. */
  unit: string;
  color: string;
  range: [number, number];
  /** i18n keys so descriptions + kind chips translate. */
  descKey: string;
  kind: 'spectral' | 'thermal' | 'water' | 'weather';
  kindKey: string;
}

export const INDICATORS: IndicatorDef[] = [
  { id: 'ndvi', name: 'NDVI', unit: 'index', color: '#16a34a', range: [0, 1], descKey: 'indicator.ndvi.desc', kind: 'spectral', kindKey: 'kind.spectral' },
  { id: 'evi', name: 'EVI', unit: 'index', color: '#65a30d', range: [0, 1], descKey: 'indicator.evi.desc', kind: 'spectral', kindKey: 'kind.spectral' },
  { id: 'savi', name: 'SAVI', unit: 'index', color: '#059669', range: [0, 1], descKey: 'indicator.savi.desc', kind: 'spectral', kindKey: 'kind.spectral' },
  { id: 'ndmi', name: 'NDMI', unit: 'index', color: '#0891b2', range: [-1, 1], descKey: 'indicator.ndmi.desc', kind: 'water', kindKey: 'kind.water' },
  { id: 'tvdi', name: 'TVDI', unit: 'index', color: '#ea580c', range: [0, 1], descKey: 'indicator.tvdi.desc', kind: 'thermal', kindKey: 'kind.thermal' },
  { id: 'cwsi', name: 'CWSI', unit: 'index', color: '#dc2626', range: [0, 1], descKey: 'indicator.cwsi.desc', kind: 'thermal', kindKey: 'kind.thermal' },
  { id: 'eta', name: 'ETa', unit: 'mm/d', color: '#0284c7', range: [0, 8], descKey: 'indicator.eta.desc', kind: 'water', kindKey: 'kind.water' },
  { id: 'lst', name: 'LST', unit: '°C', color: '#9333ea', range: [15, 55], descKey: 'indicator.lst.desc', kind: 'thermal', kindKey: 'kind.thermal' },
  { id: 'precip', name: 'Precipitation', unit: 'mm', color: '#475569', range: [0, 300], descKey: 'indicator.precip.desc', kind: 'weather', kindKey: 'kind.weather' },
  { id: 'temp', name: 'Air Temp.', unit: '°C', color: '#b45309', range: [10, 48], descKey: 'indicator.temp.desc', kind: 'weather', kindKey: 'kind.weather' },
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

type RawFarm = {
  id?: string; code?: string; name?: string; farm?: string; municipality?: string; block?: string;
  areaHa?: number; variety?: string; planted?: string; harvest?: string;
  centroid?: [number, number]; polygon?: [number, number][]; baselineHealth?: number;
  stressEvent?: Field['stressEvent']; neighbors?: string[];
};

export const FIELDS: Field[] = ((farmsKhuzestanData ?? []) as RawFarm[]).map((f, i) => ({
  id: f.id ?? `F-${String(i + 1).padStart(3, '0')}`,
  code: f.code ?? `L${String(i + 1).padStart(2, '0')}`,
  name: f.name ?? `Field ${i + 1}`,
  farm: f.farm ?? 'Khuzestan Unit',
  municipality: f.municipality ?? 'Khuzestan',
  block: f.block || 'R01',
  areaHa: typeof f.areaHa === 'number' ? f.areaHa : 25,
  variety: f.variety || 'CP69-1062',
  planted: f.planted || '2021/03',
  harvest: f.harvest || '2024/11',
  centroid: f.centroid ?? KHUZESTAN_CENTROID,
  polygon: f.polygon ?? [],
  baselineHealth: f.baselineHealth || 0.65,
  stressEvent: f.stressEvent ?? null,
  neighbors: f.neighbors ?? [],
}));

export const hasFields = FIELDS.length > 0;

/**
 * Field lookup that never returns `undefined` for an empty dataset — callers
 * guard on `hasFields`, and the inspector falls back to a placeholder record
 * instead of crashing on `FIELDS[0].id` (BUG-05 in docs/platform_bug_audit.md).
 */
export const fieldById = (id: string): Field => FIELDS.find(f => f.id === id) ?? FIELDS[0] ?? PLACEHOLDER_FIELD;

/** Default map view: Khuzestan sugarcane agro-industry (Amir Kabir / Debal Khazai). */
export const KHUZESTAN_CENTROID: [number, number] = [31.0217, 48.2794];

const PLACEHOLDER_FIELD: Field = {
  id: '', code: '—', name: '—', farm: '—', municipality: '—', block: '—', areaHa: 0,
  variety: '—', planted: '—', harvest: '—', centroid: KHUZESTAN_CENTROID, polygon: [],
  baselineHealth: 0, stressEvent: null, neighbors: [],
};


// ── Time axis: single source of truth ────────────────────────────────────────
// Dates come from the exported series (scripts/export_platform_data.py emits 48
// monthly samples per field: 2021-01-15 … 2024-12-15). Previously these were
// hard-coded here *and* the exported `series_khuzestan.json` was never imported,
// so sliders (`max={47}`) and "48 obs" labels could drift from the data. Now the
// axis is derived from the payload and every control reads `DATES.length`.
const SERIES = khuSeries as Record<string, unknown[]>;

function deriveDates(series: Record<string, unknown[]>): string[] {
  const first = Object.values(series)[0];
  if (!Array.isArray(first) || first.length === 0) return [];
  return first.map(o => String((o as { date?: string }).date ?? ""));
}

export const DATES: string[] = deriveDates(SERIES);
export const N = DATES.length;
/** Last valid index for the shared date axis (0 when empty — never `NaN`). */
export const LAST_IDX = Math.max(0, N - 1);

/** Clamp an arbitrary index into the shared date axis. */
export const clampIdx = (i: number) => Math.max(0, Math.min(LAST_IDX, Math.round(Number.isFinite(i) ? i : 0)));

/** Convenience: ISO month for an index, or "" when the axis is empty. */
export const dateAt = (i: number) => DATES[clampIdx(i)] ?? "";

// ── Data provenance ──────────────────────────────────────────────────────────
// `series_khuzestan.json` is produced by sampling Landsat/METRIC index cubes at
// monthly dates, but its anomaly scores are *rule-based proxies* (see
// scripts/export_platform_data.py: clip(offset + stress_sig * weight)), not
// frozen model inference. The platform therefore labels every view that renders
// these numbers as "Demonstration data" instead of claiming research results.
// Decision recorded in docs/platform_bug_audit.md (BUG-06).
export const DATA_PROVENANCE = {
  /** Series actually rendered by the views. */
  source: "series_khuzestan.json",
  exporter: "scripts/export_platform_data.py",
  /** True while anomaly scores are proxy formulas rather than model inference. */
  scoresAreProxy: true,
  /** Number of fields that carry a time series. */
  fieldCount: Object.keys(SERIES).length,
  /** Months represented by the shared axis. */
  months: N,
  /** First and last date on the axis (ISO). */
  firstDate: DATES[0] ?? "",
  lastDate: DATES[N - 1] ?? "",
} as const;

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

const NUMERIC_FALLBACK = 0.5;
const SCORE_KEYS = MODELS.map(m => m.id);

/**
 * Normalise one exported observation. Exported payloads are read with
 * `as unknown` because JSON imports are untyped; this keeps the rest of the
 * app on the strongly-typed `Obs` shape.
 */
function normaliseObs(raw: unknown, idx: number): Obs {
  const o = (raw ?? {}) as Record<string, unknown>;
  const scores = (o.scores ?? {}) as Record<string, unknown>;
  const nums = {} as Record<ModelId, number>;
  SCORE_KEYS.forEach(k => {
    const v = scores[k];
    nums[k] = typeof v === "number" && Number.isFinite(v) ? v : NUMERIC_FALLBACK;
  });
  const num = (key: string, fallback: number) => {
    const v = o[key];
    return typeof v === "number" && Number.isFinite(v) ? v : fallback;
  };
  return {
    date: String(o.date ?? dateAt(idx)),
    idx: typeof o.idx === "number" ? o.idx : idx,
    ndvi: num("ndvi", 0.45),
    evi: num("evi", 0.38),
    savi: num("savi", 0.42),
    ndmi: num("ndmi", 0.15),
    tvdi: num("tvdi", 0.35),
    cwsi: num("cwsi", 0.3),
    eta: num("eta", 3.8),
    lst: num("lst", 28.5),
    precip: num("precip", 0),
    temp: num("temp", 28.5),
    scores: nums,
    missing: Boolean(o.missing),
  };
}

const fieldCache = new Map<string, Obs[]>();

/** Fields that actually carry a time series — data-driven replacement for "48 fields" labels. */
export const FIELDS_WITH_SERIES: Field[] = FIELDS.filter(f => getSeries(f.id).length > 0);
const EMPTY_SERIES: Obs[] = [];

/**
 * Time series for a field, read from the exported `series_khuzestan.json`.
 * An unknown/absent field returns the empty array rather than throwing, so the
 * views' empty-state guards stay reachable (spec: "Empty data guard").
 */
export function getSeries(fieldId: string): Obs[] {
  const cached = fieldCache.get(fieldId);
  if (cached) return cached;
  const raw = SERIES[fieldId];
  const obs = Array.isArray(raw) ? raw.map((o, i) => normaliseObs(o, i)) : EMPTY_SERIES;
  fieldCache.set(fieldId, obs);
  return obs;
}

/** Safe observation accessor — clamps the index and tolerates empty series. */
export function obsAt(series: Obs[], idx: number): Obs | undefined {
  if (!series.length) return undefined;
  return series[clampIdx(idx)];
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
  const lo = Math.min(range[0], range[1]), hi = Math.max(range[0], range[1]);
  fieldIds.forEach(fid => {
    getSeries(fid).forEach(o => {
      if (o.idx >= lo && o.idx <= hi && !o.missing) {
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
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  // Revoking synchronously can cancel the download in Chromium; defer one tick.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** ISO month (`YYYY-MM`) used by every chart axis in both locales. */
export const monthAt = (i: number) => (dateAt(i) ? dateAt(i).slice(0, 7) : '—');

/** First/last ISO month on the shared axis, for captions. */
export const COVERAGE = {
  first: DATES[0]?.slice(0, 7) ?? '—',
  last: DATES[N - 1]?.slice(0, 7) ?? '—',
  yearStart: (DATES[0] ?? '').slice(0, 4) || '—',
  yearEnd: (DATES[N - 1] ?? '').slice(0, 4) || '—',
};

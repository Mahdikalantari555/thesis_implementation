import { useMemo, useState } from 'react';
import { XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, LineChart, Line, ScatterChart, Scatter } from 'recharts';
import MapView from '../components/MapView';
import { Panel, Label, Select, Stat, Badge, fmt } from '../components/ui';
import { useAnalysis } from '../context/AnalysisContext';
import { FIELDS, MODELS, INDICATORS, getSeries, moransI, pearson, mean, DATES, fieldById } from '../data/mock';

export default function Spatial() {
  const { model, setModel, indicator, setIndicator, selectedField, setSelectedField } = useAnalysis();
  const [dateIdx, setDateIdx] = useState(35);
  const [mode, setMode] = useState<'anomaly' | 'indicator' | 'lisa'>('anomaly');

  const values = useMemo(() => FIELDS.map(f => getSeries(f.id)[dateIdx].scores[model]), [model, dateIdx]);
  const weights = useMemo(() => {
    const m = new Map<string, string[]>();
    FIELDS.forEach((f, i) => m.set(String(i), f.neighbors.map(n => String(FIELDS.findIndex(x => x.id === n)))));
    return m;
  }, []);
  const moran = moransI(values, weights);

  // Spatial correlation: field score vs indicator (field means)
  const fieldMeans = useMemo(() => FIELDS.map(f => {
    const s = getSeries(f.id);
    return { field: f.code, score: +mean(s.map(o => o.scores[model])).toFixed(3), ind: +mean(s.map(o => (o as any)[indicator])).toFixed(3), nbScore: +mean(f.neighbors.flatMap(n => getSeries(n).map(o => o.scores[model]))).toFixed(3) };
  }), [model, indicator]);
  const spatialR = pearson(fieldMeans.map(d => d.score), fieldMeans.map(d => d.ind));
  const nbR = pearson(fieldMeans.map(d => d.score), fieldMeans.map(d => d.nbScore));

  // Variogram-like: correlation by neighbour distance proxy (grid distance)
  const variogram = useMemo(() => {
    const bins = [0, 1, 2, 3, 4, 5].map(d => ({ dist: d, pairs: [] as number[] }));
    FIELDS.forEach((a, i) => FIELDS.forEach((b, j) => {
      if (j <= i) return;
      const dist = Math.round(Math.hypot(a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1]) * 120);
      const bin = bins[Math.min(5, dist)];
      bin.pairs.push(Math.abs(values[i] - values[j]));
    }));
    return bins.map(b => ({ dist: b.dist, semivar: b.pairs.length ? +(mean(b.pairs) / 2).toFixed(3) : 0, n: b.pairs.length }));
  }, [values]);

  const field = fieldById(selectedField);
  const neighborRows = field.neighbors.map(nid => { const nf = fieldById(nid); const o = getSeries(nid)[dateIdx]; return { id: nid, code: nf.code, score: o.scores[model], ndvi: o.ndvi, cwsi: o.cwsi }; });

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3 items-end border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3">
        <div><Label>Model</Label><Select value={model} onChange={v => setModel(v as any)}>{MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</Select></div>
        <div><Label>Indicator</Label><Select value={indicator} onChange={v => setIndicator(v as any)}>{INDICATORS.map(i => <option key={i.id} value={i.id}>{i.name}</option>)}</Select></div>
        <div><Label>Map mode</Label><Select value={mode} onChange={v => setMode(v as any)}><option value="anomaly">Anomaly score</option><option value="indicator">Indicator</option><option value="lisa">LISA-like clusters</option></Select></div>
        <div className="flex-1 min-w-48"><Label>Date — {DATES[dateIdx].slice(0, 7)}</Label><input type="range" min={0} max={47} value={dateIdx} onChange={e => setDateIdx(+e.target.value)} className="w-full accent-teal-700" /></div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-slate-200 dark:bg-slate-800 border border-slate-200 dark:border-slate-800">
        <Stat label="Moran's I (score)" value={fmt(moran)} sub={moran > 0.3 ? 'strong clustering' : moran > 0.1 ? 'moderate clustering' : 'weak / random'} />
        <Stat label="Score ↔ indicator (field means)" value={fmt(spatialR)} sub={`r across ${FIELDS.length} fields`} />
        <Stat label="Score ↔ neighbour score" value={fmt(nbR)} sub="spatial autocorrelation proxy" />
        <Stat label="Fields" value={String(FIELDS.length)} sub={`${FIELDS.length} in spatial adjacency`} />
      </div>

      <div className="grid grid-cols-12 gap-4">
        <Panel title="Anomaly distribution map" subtitle="Click a field to inspect its neighbours" className="col-span-12 xl:col-span-7" pad={false}>
          <div className="h-[460px]"><MapView colorMode={mode} model={model} indicator={indicator} dateIdx={dateIdx} opacity={0.85} height="460px" /></div>
        </Panel>

        <div className="col-span-12 xl:col-span-5 space-y-4">
          <Panel title={`Neighbour comparison — ${field.code}`} subtitle="Selected field vs adjacent fields at same date">
            <table className="w-full text-[11px]">
              <thead><tr className="text-slate-400"><th className="text-left py-1 font-medium">Field</th><th className="text-right py-1 font-medium">Score</th><th className="text-right py-1 font-medium">NDVI</th><th className="text-right py-1 font-medium">CWSI</th></tr></thead>
              <tbody>
                <tr className="bg-teal-50 dark:bg-teal-950/40 font-medium"><td className="py-1.5">{field.code} (selected)</td><td className="text-right font-mono">{getSeries(field.id)[dateIdx].scores[model].toFixed(3)}</td><td className="text-right font-mono">{getSeries(field.id)[dateIdx].ndvi.toFixed(2)}</td><td className="text-right font-mono">{getSeries(field.id)[dateIdx].cwsi.toFixed(2)}</td></tr>
                {neighborRows.map(r => <tr key={r.id} className="border-t border-slate-100 dark:border-slate-800 cursor-pointer hover:bg-slate-50 dark:hover:bg-slate-800/50" onClick={() => setSelectedField(r.id)}><td className="py-1.5">{r.code}</td><td className="text-right font-mono">{r.score.toFixed(3)}</td><td className="text-right font-mono">{r.ndvi.toFixed(2)}</td><td className="text-right font-mono">{r.cwsi.toFixed(2)}</td></tr>)}
              </tbody>
            </table>
            <div className="text-[11px] text-slate-500 mt-2">If neighbours show similar elevation, the pattern may be regional (weather, soil) rather than field-specific stress. If isolated, check management / sensor artefacts.</div>
          </Panel>

          <Panel title="Empirical variogram (score)" subtitle="Semivariance vs distance proxy — rising curve = spatial structure">
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={variogram} margin={{ top: 5, right: 15, left: -15, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="dist" tick={{ fontSize: 10 }} label={{ value: 'Distance (proxy units)', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: 11 }} />
                  <Line type="monotone" dataKey="semivar" stroke="#0f766e" strokeWidth={2} dot={{ r: 3 }} name="Semivariance" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </Panel>

          <Panel title="Field-mean scatter" subtitle={`Mean score vs mean ${INDICATORS.find(i => i.id === indicator)?.name} (one point per field)`}>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis type="number" dataKey="score" domain={[0, 1]} tick={{ fontSize: 10 }} />
                  <YAxis type="number" dataKey="ind" tick={{ fontSize: 10 }} />
                  <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: 11 }} />
                  <Scatter data={fieldMeans} fill={MODELS.find(m => m.id === model)?.color} fillOpacity={0.55} />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </Panel>
        </div>

        <Panel title="Moran's I over time" subtitle="Spatial clustering of scores per month — persistent positive I suggests regional patterns" className="col-span-12">
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={DATES.map((d, i) => { const vals = FIELDS.map(f => getSeries(f.id)[i].scores[model]); return { date: d.slice(0, 7), moran: +moransI(vals, weights).toFixed(3) }; })} margin={{ top: 5, right: 15, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 9 }} interval={4} />
                <YAxis domain={[-0.5, 1]} tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="moran" stroke="#7c3aed" strokeWidth={2} dot={false} name="Moran's I" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="flex gap-2 mt-2"><Badge color="#7c3aed">I &gt; 0 clustered</Badge><Badge>I ≈ 0 random</Badge><Badge>I &lt; 0 dispersed</Badge></div>
        </Panel>
      </div>
    </div>
  );
}

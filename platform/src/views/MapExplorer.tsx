import { useMemo, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine } from 'recharts';
import MapView from '../components/MapView';
import { Panel, Badge, Label, Select, anomalyColor } from '../components/ui';
import { useAnalysis } from '../context/AnalysisContext';
import { FIELDS, MODELS, INDICATORS, getSeries, fieldById, DATES, ANOMALY_THRESHOLD, mean } from '../data/mock';
import { Eye, EyeOff, Search, GitCompareArrows } from 'lucide-react';

export default function MapExplorer() {
  const { selectedField, selectedFields, toggleField, model, setModel, indicator, setIndicator, range, setView } = useAnalysis();
  const [colorMode, setColorMode] = useState<'anomaly' | 'indicator' | 'health' | 'lisa'>('anomaly');
  const [dateIdx, setDateIdx] = useState(35);
  const [opacity, setOpacity] = useState(0.85);
  const [showFields, setShowFields] = useState(true);
  const [query, setQuery] = useState('');
  const [compareIdx, setCompareIdx] = useState(20);
  const [showCompare, setShowCompare] = useState(false);

  const field = fieldById(selectedField);
  const series = getSeries(selectedField);
  const chartData = series.map(o => ({ date: o.date.slice(0, 7), score: +o.scores[model].toFixed(3), ind: +((o as any)[indicator]).toFixed(3) }));
  const obs = series[dateIdx];

  const filtered = useMemo(() => FIELDS.filter(f => (f.code + f.name + f.municipality + f.farm).toLowerCase().includes(query.toLowerCase())), [query]);
  const selectedStats = useMemo(() => {
    const vals = selectedFields.map(fid => getSeries(fid)[dateIdx].scores[model]);
    return { mean: mean(vals), max: Math.max(...vals), min: Math.min(...vals) };
  }, [selectedFields, dateIdx, model]);

  return (
    <div className="grid grid-cols-12 gap-4">
      {/* Left controls */}
      <div className="col-span-12 lg:col-span-3 space-y-4">
        <Panel title="Layer & model">
          <div className="space-y-3">
            <div><Label>Colour by</Label>
              <Select value={colorMode} onChange={v => setColorMode(v as any)} className="w-full">
                <option value="anomaly">Anomaly score ({MODELS.find(m => m.id === model)?.name})</option>
                <option value="indicator">Indicator ({INDICATORS.find(i => i.id === indicator)?.name})</option>
                <option value="health">Baseline health</option>
                <option value="lisa">LISA-like (self vs neighbours)</option>
              </Select></div>
            <div><Label>Anomaly model</Label>
              <Select value={model} onChange={v => setModel(v as any)} className="w-full">{MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</Select></div>
            <div><Label>Indicator</Label>
              <Select value={indicator} onChange={v => setIndicator(v as any)} className="w-full">{INDICATORS.map(i => <option key={i.id} value={i.id}>{i.name}</option>)}</Select></div>
            <div>
              <Label>Observation date — {DATES[dateIdx].slice(0, 7)} {obs.missing && <span className="text-orange-600">(missing)</span>}</Label>
              <input type="range" min={0} max={47} value={dateIdx} onChange={e => setDateIdx(+e.target.value)} className="w-full accent-teal-700" />
              <div className="flex justify-between text-[10px] text-slate-400"><span>2021-01</span><span>2024-12</span></div>
            </div>
            <div>
              <Label>Layer opacity — {opacity.toFixed(2)}</Label>
              <input type="range" min={0.2} max={1} step={0.05} value={opacity} onChange={e => setOpacity(+e.target.value)} className="w-full accent-teal-700" />
            </div>
            <div className="flex items-center justify-between text-[12px]">
              <span className="flex items-center gap-1.5">{showFields ? <Eye size={13} /> : <EyeOff size={13} />} Field boundaries</span>
              <button onClick={() => setShowFields(!showFields)} className="text-[11px] border border-slate-300 dark:border-slate-700 px-2 py-1">{showFields ? 'Hide' : 'Show'}</button>
            </div>
            <button onClick={() => setShowCompare(!showCompare)} className="w-full flex items-center justify-center gap-2 h-8 border border-slate-300 dark:border-slate-700 text-[12px] hover:bg-slate-50 dark:hover:bg-slate-800">
              <GitCompareArrows size={13} /> {showCompare ? 'Hide date comparison' : 'Compare two dates'}
            </button>
            {showCompare && <div><Label>Comparison date — {DATES[compareIdx].slice(0, 7)}</Label><input type="range" min={0} max={47} value={compareIdx} onChange={e => setCompareIdx(+e.target.value)} className="w-full accent-orange-600" /></div>}
          </div>
        </Panel>

        <Panel title="Legend">
          {colorMode === 'anomaly' && <div className="space-y-1.5 text-[11px]">{[['#be123c', '≥ 0.75 severe'], ['#ea580c', '0.65–0.75 high'], ['#d97706', '0.50–0.65 elevated'], ['#65a30d', '0.35–0.50 mild'], ['#0f766e', '< 0.35 nominal']].map(([c, l]: any) => <div key={l} className="flex items-center gap-2"><span className="w-3 h-3" style={{ background: c }} />{l}</div>)}</div>}
          {colorMode === 'lisa' && <div className="space-y-1.5 text-[11px]">{[['#be123c', 'High–High cluster'], ['#2563eb', 'High–Low outlier'], ['#7c3aed', 'Low–High outlier'], ['#94a3b8', 'Low–Low']].map(([c, l]) => <div key={l} className="flex items-center gap-2"><span className="w-3 h-3" style={{ background: c }} />{l}</div>)}</div>}
          {colorMode === 'indicator' && <div className="text-[11px] text-slate-600 dark:text-slate-400">Colour encodes {INDICATORS.find(i => i.id === indicator)?.name} at selected date. Green = favourable / low stress, red = high stress (direction depends on indicator — check Temporal view).</div>}
          {colorMode === 'health' && <div className="space-y-1.5 text-[11px]">{[['#15803d', 'High baseline'], ['#65a30d', 'Medium'], ['#d97706', 'Low']].map(([c, l]) => <div key={l} className="flex items-center gap-2"><span className="w-3 h-3" style={{ background: c }} />{l}</div>)}</div>}
        </Panel>

        <Panel title="Field search" pad={false}>
          <div className="p-3 border-b border-slate-100 dark:border-slate-800 flex items-center gap-2"><Search size={13} className="text-slate-400" /><input value={query} onChange={e => setQuery(e.target.value)} placeholder="Code, farm, municipality…" className="flex-1 text-[12px] bg-transparent outline-none" /></div>
          <div className="max-h-64 overflow-auto divide-y divide-slate-100 dark:divide-slate-800">
            {filtered.map(f => { const s = getSeries(f.id)[dateIdx].scores[model]; return (
              <button key={f.id} onClick={() => toggleField(f.id)} className={`w-full flex items-center gap-2 px-3 py-2 text-left text-[11px] hover:bg-slate-50 dark:hover:bg-slate-800/60 ${f.id === selectedField ? 'bg-teal-50 dark:bg-teal-950/40' : ''}`}>
                <span className="w-2 h-2 shrink-0" style={{ background: anomalyColor(s) }} />
                <span className="flex-1 min-w-0"><span className="block font-mono font-medium">{f.code}</span><span className="block text-slate-500 truncate">{f.municipality} · {f.areaHa} ha</span></span>
                <span className="font-mono">{s.toFixed(2)}</span>
              </button>); })}
          </div>
        </Panel>
      </div>

      {/* Map */}
      <div className="col-span-12 lg:col-span-6 space-y-4">
        <Panel title={`Geospatial explorer — ${DATES[dateIdx].slice(0, 7)}`} subtitle={`${FIELDS.length} fields · click to select · scroll to zoom · ${selectedFields.length} fields in selection`} pad={false}>
          <div className="h-[520px] relative">
            {showFields ? <MapView colorMode={colorMode} model={model} indicator={indicator} dateIdx={dateIdx} opacity={opacity} height="520px" /> : <div className="h-full flex items-center justify-center text-[12px] text-slate-500">Field layer hidden — enable in controls</div>}
            <div className="absolute top-3 left-3 z-[500] bg-white/95 dark:bg-slate-900/95 border border-slate-200 dark:border-slate-700 px-3 py-2 text-[11px] shadow-sm">
              <div className="font-semibold">{MODELS.find(m => m.id === model)?.name} · {DATES[dateIdx].slice(0, 7)}</div>
              <div className="text-slate-500">Selection mean {selectedStats.mean.toFixed(3)} · max {selectedStats.max.toFixed(3)}</div>
            </div>
          </div>
        </Panel>
        {showCompare && (
          <Panel title="Date comparison" subtitle={`${DATES[compareIdx].slice(0, 7)} vs ${DATES[dateIdx].slice(0, 7)} — same model & colour scale`} pad={false}>
            <div className="grid grid-cols-2 gap-px bg-slate-200 dark:bg-slate-800">
              <div className="h-64"><MapView colorMode={colorMode} model={model} indicator={indicator} dateIdx={compareIdx} opacity={opacity} height="256px" /></div>
              <div className="h-64"><MapView colorMode={colorMode} model={model} indicator={indicator} dateIdx={dateIdx} opacity={opacity} height="256px" /></div>
            </div>
            <div className="p-3 text-[11px] text-slate-500">Left: {DATES[compareIdx].slice(0, 7)} · Right: {DATES[dateIdx].slice(0, 7)}. Compare spatial extent of high scores across dates.</div>
          </Panel>
        )}
      </div>

      {/* Inspector */}
      <div className="col-span-12 lg:col-span-3 space-y-4">
        <Panel title="Field inspector" subtitle={field.code}>
          <dl className="text-[12px] space-y-1.5">
            {[['Farm', field.farm], ['Municipality', field.municipality], ['Area', `${field.areaHa} ha`], ['Variety', field.variety], ['Planted', field.planted], ['Harvest', field.harvest], ['Centroid', `${field.centroid[0].toFixed(4)}, ${field.centroid[1].toFixed(4)}`], ['Neighbours', field.neighbors.join(', ') || '—']].map(([k, v]) => (
              <div key={k} className="flex justify-between gap-3"><dt className="text-slate-500">{k}</dt><dd className="font-medium text-right">{v}</dd></div>
            ))}
          </dl>
          <div className="mt-3 grid grid-cols-3 gap-px bg-slate-200 dark:bg-slate-800 border border-slate-200 dark:border-slate-800 text-center">
            <div className="bg-white dark:bg-slate-900 p-2"><div className="text-[9px] uppercase text-slate-400">Score</div><div className="text-[13px] font-semibold" style={{ color: anomalyColor(obs.scores[model]) }}>{obs.scores[model].toFixed(3)}</div></div>
            <div className="bg-white dark:bg-slate-900 p-2"><div className="text-[9px] uppercase text-slate-400">NDVI</div><div className="text-[13px] font-semibold">{obs.ndvi.toFixed(2)}</div></div>
            <div className="bg-white dark:bg-slate-900 p-2"><div className="text-[9px] uppercase text-slate-400">CWSI</div><div className="text-[13px] font-semibold">{obs.cwsi.toFixed(2)}</div></div>
          </div>
          {field.stressEvent && <div className="mt-3 text-[11px] border border-orange-200 bg-orange-50 dark:bg-orange-950/30 dark:border-orange-900 p-2 text-orange-800 dark:text-orange-200">Mock stress event: {DATES[field.stressEvent.startIdx].slice(0, 7)} → {DATES[field.stressEvent.endIdx].slice(0, 7)} (severity {field.stressEvent.severity.toFixed(2)})</div>}
          <div className="mt-3 flex gap-2">
            <button onClick={() => setView('temporal')} className="flex-1 h-8 bg-teal-700 text-white text-[12px]">Open in Temporal</button>
            <button onClick={() => setView('validation')} className="flex-1 h-8 border border-slate-300 dark:border-slate-700 text-[12px]">Validate</button>
          </div>
        </Panel>

        <Panel title="Score history" subtitle={`${MODELS.find(m => m.id === model)?.name} vs ${INDICATORS.find(i => i.id === indicator)?.name}`}>
          <div className="h-44">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 5, right: 8, left: -18, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 9 }} interval={8} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 9 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <ReferenceLine y={ANOMALY_THRESHOLD} stroke="#ea580c" strokeDasharray="4 3" />
                <Line type="monotone" dataKey="score" stroke={MODELS.find(m => m.id === model)?.color} strokeWidth={2} dot={false} name="Score" />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Dashed line = threshold {ANOMALY_THRESHOLD}. Full multi-axis analysis in Temporal view.</div>
        </Panel>

        <Panel title="Selection" subtitle={`${selectedFields.length} fields tracked across views`}>
          <div className="flex flex-wrap gap-1.5">{selectedFields.map(id => <button key={id} onClick={() => toggleField(id)}><Badge color={id === selectedField ? '#0f766e' : undefined}>{id}</Badge></button>)}</div>
          <div className="mt-3 text-[11px] text-slate-500">Date range shared with all views: <b>{DATES[range[0]].slice(0, 7)} → {DATES[range[1]].slice(0, 7)}</b></div>
          <div className="mt-2 flex gap-2">
            <button onClick={() => setView('spatial')} className="flex-1 h-7 border border-slate-300 dark:border-slate-700 text-[11px]">Spatial analysis</button>
            <button onClick={() => setView('spectral')} className="flex-1 h-7 border border-slate-300 dark:border-slate-700 text-[11px]">Spectral analysis</button>
          </div>
        </Panel>
      </div>
    </div>
  );
}

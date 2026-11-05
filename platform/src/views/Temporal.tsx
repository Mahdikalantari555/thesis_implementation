import { useMemo, useState } from 'react';
import { ComposedChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine, Brush, Legend, Area } from 'recharts';
import { Panel, Label, Select, Badge } from '../components/ui';
import { useAnalysis } from '../context/AnalysisContext';
import { FIELDS, MODELS, INDICATORS, getSeries, DATES, ANOMALY_THRESHOLD, type ModelId, type IndicatorId } from '../data/mock';

const VAR_COLORS: Record<string, string> = { ndvi: '#16a34a', evi: '#65a30d', savi: '#059669', ndmi: '#0891b2', tvdi: '#ea580c', cwsi: '#dc2626', eta: '#0284c7', lst: '#9333ea', precip: '#64748b', temp: '#b45309' };

export default function Temporal() {
  const { selectedField, setSelectedField, model, setModel, range, setRange } = useAnalysis();
  const [showModels, setShowModels] = useState<ModelId[]>([model]);
  const [showVars, setShowVars] = useState<IndicatorId[]>(['ndvi', 'tvdi', 'cwsi', 'eta']);
  const [normalize, setNormalize] = useState(false);
  const [showMissing, setShowMissing] = useState(true);

  const series = getSeries(selectedField);
  const data = useMemo(() => series.map(o => {
    const row: any = { date: o.date.slice(0, 7), idx: o.idx, missing: o.missing ? 1 : 0 };
    showModels.forEach(m => row[`m_${m}`] = o.missing ? null : +o.scores[m].toFixed(3));
    showVars.forEach(v => {
      let val = (o as any)[v] as number;
      if (normalize) { const ind = INDICATORS.find(i => i.id === v)!; val = (val - ind.range[0]) / (ind.range[1] - ind.range[0]); }
      row[`v_${v}`] = o.missing ? null : +val.toFixed(3);
    });
    return row;
  }), [series, showModels, showVars, normalize]);

  const toggle = <T,>(list: T[], v: T, setter: (l: T[]) => void) => setter(list.includes(v) ? list.filter(x => x !== v) : [...list, v]);

  const filtered = data.filter(d => d.idx >= range[0] && d.idx <= range[1]);

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-12 gap-4">
        <Panel title="Configuration" className="col-span-12 lg:col-span-3">
          <div className="space-y-4">
            <div><Label>Field</Label><Select value={selectedField} onChange={setSelectedField} className="w-full">{FIELDS.map(f => <option key={f.id} value={f.id}>{f.code} — {f.name} ({f.municipality})</option>)}</Select></div>
            <div><Label>Primary model</Label><Select value={model} onChange={v => { setModel(v as any); if (!showModels.includes(v as any)) setShowModels([...showModels, v as any]); }} className="w-full">{MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</Select></div>
            <div>
              <Label>Models to overlay</Label>
              <div className="space-y-1">{MODELS.map(m => <label key={m.id} className="flex items-center gap-2 text-[12px] cursor-pointer"><input type="checkbox" checked={showModels.includes(m.id)} onChange={() => toggle(showModels, m.id, setShowModels)} className="accent-teal-700" /><span className="w-2.5 h-2.5" style={{ background: m.color }} />{m.name}</label>)}</div>
            </div>
            <div>
              <Label>Indicators / variables</Label>
              <div className="space-y-1 max-h-48 overflow-auto">{INDICATORS.map(i => <label key={i.id} className="flex items-center gap-2 text-[12px] cursor-pointer"><input type="checkbox" checked={showVars.includes(i.id)} onChange={() => toggle(showVars, i.id, setShowVars)} className="accent-teal-700" /><span className="w-2.5 h-2.5" style={{ background: i.color }} />{i.name} <span className="text-slate-400 text-[10px]">{i.unit}</span></label>)}</div>
            </div>
            <label className="flex items-center gap-2 text-[12px] cursor-pointer"><input type="checkbox" checked={normalize} onChange={e => setNormalize(e.target.checked)} className="accent-teal-700" /> Normalise indicators to [0,1] (shared axis)</label>
            <label className="flex items-center gap-2 text-[12px] cursor-pointer"><input type="checkbox" checked={showMissing} onChange={e => setShowMissing(e.target.checked)} className="accent-teal-700" /> Show missing-observation markers</label>
            <div className="border border-slate-200 dark:border-slate-800 p-2 text-[11px] text-slate-500 leading-relaxed">
              <b className="text-slate-700 dark:text-slate-300">Reading multi-unit charts:</b> when normalisation is off, each indicator uses its own right-hand axis (see legend order). Do not compare slopes across different units without normalising.
            </div>
          </div>
        </Panel>

        <div className="col-span-12 lg:col-span-9 space-y-4">
          <Panel title={`Synchronised time series — ${selectedField}`} subtitle={`${FIELDS.find(f => f.id === selectedField)?.farm} · ${filtered.length} obs in range · drag brush to zoom · hover for values`}
            right={<Badge color="#0f766e">{DATES[range[0]].slice(0, 7)} → {DATES[range[1]].slice(0, 7)}</Badge>}>
            <div className="h-[420px]">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={data} margin={{ top: 10, right: 50, left: 0, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} interval={3} />
                  <YAxis yAxisId="left" domain={[0, 1]} tick={{ fontSize: 10 }} label={{ value: 'Anomaly score', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                  <YAxis yAxisId="right" orientation="right" tick={{ fontSize: 10 }} hide={normalize} />
                  <Tooltip contentStyle={{ fontSize: 11 }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <ReferenceLine yAxisId="left" y={ANOMALY_THRESHOLD} stroke="#ea580c" strokeDasharray="5 4" label={{ value: 'threshold', fontSize: 9, fill: '#ea580c' }} />
                  {showModels.map(m => <Line key={m} yAxisId="left" type="monotone" dataKey={`m_${m}`} name={MODELS.find(x => x.id === m)?.name} stroke={MODELS.find(x => x.id === m)?.color} strokeWidth={m === model ? 2.5 : 1.5} dot={false} connectNulls={false} />)}
                  {showVars.map((v, i) => <Line key={v} yAxisId={normalize ? 'left' : 'right'} type="monotone" dataKey={`v_${v}`} name={INDICATORS.find(x => x.id === v)?.name + (normalize ? ' (norm)' : '')} stroke={VAR_COLORS[v]} strokeWidth={1.4} dot={false} strokeDasharray={i % 2 ? '5 3' : undefined} />)}
                  {showMissing && <Area yAxisId="left" type="step" dataKey="missing" fill="#f59e0b" fillOpacity={0.15} stroke="none" name="Missing" />}
                  <Brush dataKey="date" height={24} tickFormatter={() => ''} onChange={(r: any) => { if (r && r.startIndex !== undefined) setRange([r.startIndex, r.endIndex]); }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-wrap gap-2 mt-2">
              <button onClick={() => setRange([0, 47])} className="h-7 px-2 border border-slate-300 dark:border-slate-700 text-[11px]">Full period</button>
              <button onClick={() => setRange([12, 23])} className="h-7 px-2 border border-slate-300 dark:border-slate-700 text-[11px]">2022</button>
              <button onClick={() => setRange([24, 35])} className="h-7 px-2 border border-slate-300 dark:border-slate-700 text-[11px]">2023</button>
              <button onClick={() => setRange([36, 47])} className="h-7 px-2 border border-slate-300 dark:border-slate-700 text-[11px]">2024</button>
              <span className="text-[11px] text-slate-500 self-center ml-auto">Amber band = missing obs · dashed = anomaly threshold</span>
            </div>
          </Panel>

          {/* Small multiples */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {showVars.slice(0, 4).map(v => {
              const ind = INDICATORS.find(i => i.id === v)!;
              const d = series.map(o => ({ date: o.date.slice(0, 7), val: o.missing ? null : +((o as any)[v]).toFixed(3), score: +o.scores[model].toFixed(3) }));
              return (
                <Panel key={v} title={`${ind.name} vs ${MODELS.find(m => m.id === model)?.name}`} subtitle={`${ind.desc} · own axis (${ind.unit})`} pad={false}>
                  <div className="h-40 p-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart data={d} margin={{ top: 5, right: 30, left: -15, bottom: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                        <XAxis dataKey="date" tick={{ fontSize: 9 }} interval={8} />
                        <YAxis yAxisId="l" domain={[0, 1]} tick={{ fontSize: 9 }} />
                        <YAxis yAxisId="r" orientation="right" tick={{ fontSize: 9 }} />
                        <Tooltip contentStyle={{ fontSize: 11 }} />
                        <Line yAxisId="l" type="monotone" dataKey="score" stroke={MODELS.find(m => m.id === model)?.color} strokeWidth={2} dot={false} name="Score" />
                        <Line yAxisId="r" type="monotone" dataKey="val" stroke={ind.color} strokeWidth={1.5} dot={false} name={ind.name} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                </Panel>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

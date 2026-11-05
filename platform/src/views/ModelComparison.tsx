import { useMemo, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, LineChart, Line, ScatterChart, Scatter, Legend } from 'recharts';
import { Panel, Label, Badge, fmt } from '../components/ui';
import { useAnalysis } from '../context/AnalysisContext';
import { MODELS, INDICATORS, FIELDS, getSeries, getPairs, pearson, mean, quantile, ANOMALY_THRESHOLD, DATES, type ModelId } from '../data/mock';

export default function ModelComparison() {
  const { compareModels, setCompareModels, indicator, setIndicator, range, selectedField } = useAnalysis();
  const [normalize, setNormalize] = useState<'raw' | 'percentile'>('raw');

  const toggle = (m: ModelId) => setCompareModels(compareModels.includes(m) ? compareModels.filter(x => x !== m) : [...compareModels, m]);

  // Distributions
  const distData = useMemo(() => {
    const bins = Array.from({ length: 10 }, (_, i) => ({ bin: `${(i / 10).toFixed(1)}`, }));
    compareModels.forEach(m => {
      const vals = FIELDS.flatMap(f => getSeries(f.id).map(o => o.scores[m]));
      const sorted = [...vals].sort((a, b) => a - b);
      bins.forEach((b, i) => {
        let v = vals.filter(x => x >= i / 10 && x < (i + 1) / 10).length / vals.length;
        if (normalize === 'percentile') v = quantile(sorted, (i + 0.5) / 10);
        (b as any)[m] = +v.toFixed(3);
      });
    });
    return bins;
  }, [compareModels, normalize]);

  // Pairwise correlations between models
  const pairMatrix = useMemo(() => compareModels.map(a => {
    const row: any = { model: MODELS.find(m => m.id === a)?.name, color: MODELS.find(m => m.id === a)?.color };
    compareModels.forEach(b => {
      const xs: number[] = [], ys: number[] = [];
      FIELDS.forEach(f => getSeries(f.id).forEach(o => { if (!o.missing) { xs.push(o.scores[a]); ys.push(o.scores[b]); } }));
      row[MODELS.find(m => m.id === b)?.short || b] = +pearson(xs, ys).toFixed(2);
    });
    return row;
  }), [compareModels]);

  // Correlation with indicator per model
  const indCorr = useMemo(() => compareModels.map(m => {
    const p = getPairs(m, indicator, FIELDS.map(f => f.id), range);
    return { model: MODELS.find(x => x.id === m)?.name, color: MODELS.find(x => x.id === m)?.color, r: +pearson(p.xs, p.ys).toFixed(3), mean: +mean(p.xs).toFixed(3), anomRate: +(p.xs.filter(v => v >= ANOMALY_THRESHOLD).length / p.xs.length).toFixed(3) };
  }), [compareModels, indicator, range]);

  // Event agreement (Jaccard)
  const agreement = useMemo(() => compareModels.map(a => {
    const row: any = { model: MODELS.find(m => m.id === a)?.short };
    compareModels.forEach(b => {
      let inter = 0, union = 0;
      FIELDS.forEach(f => getSeries(f.id).forEach(o => { const ea = o.scores[a] >= ANOMALY_THRESHOLD, eb = o.scores[b] >= ANOMALY_THRESHOLD; if (ea && eb) inter++; if (ea || eb) union++; }));
      row[MODELS.find(m => m.id === b)?.short || b] = union ? +(inter / union).toFixed(2) : 0;
    });
    return row;
  }), [compareModels]);

  // Temporal sync for selected field
  const temporal = useMemo(() => getSeries(selectedField).map(o => {
    const row: any = { date: o.date.slice(0, 7) };
    compareModels.forEach(m => row[m] = +o.scores[m].toFixed(3));
    return row;
  }), [compareModels, selectedField]);

  // Boxplot-like summary per model
  const box = useMemo(() => compareModels.map(m => {
    const vals = FIELDS.flatMap(f => getSeries(f.id).map(o => o.scores[m])).sort((a, b) => a - b);
    return { model: MODELS.find(x => x.id === m)?.name, color: MODELS.find(x => x.id === m)?.color, min: +vals[0].toFixed(2), p25: +quantile(vals, 0.25).toFixed(2), median: +quantile(vals, 0.5).toFixed(2), p75: +quantile(vals, 0.75).toFixed(2), max: +vals[vals.length - 1].toFixed(2) };
  }), [compareModels]);

  // Pairwise scatter first two models
  const pairScatter = useMemo(() => {
    if (compareModels.length < 2) return [];
    const [a, b] = compareModels;
    return FIELDS.flatMap(f => getSeries(f.id).map(o => ({ x: +o.scores[a].toFixed(3), y: +o.scores[b].toFixed(3) })));
  }, [compareModels]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-4 items-start border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3">
        <div>
          <Label>Models to compare (2–5)</Label>
          <div className="flex flex-wrap gap-2">{MODELS.map(m => <button key={m.id} onClick={() => toggle(m.id)} className="flex items-center gap-1.5 h-8 px-2 border text-[12px]" style={{ borderColor: compareModels.includes(m.id) ? m.color : '#cbd5e1', background: compareModels.includes(m.id) ? m.color + '18' : 'transparent', color: compareModels.includes(m.id) ? m.color : '#64748b' }}><span className="w-2 h-2" style={{ background: m.color }} />{m.name}</button>)}</div>
        </div>
        <div><Label>Indicator</Label><select value={indicator} onChange={e => setIndicator(e.target.value as any)} className="h-8 border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12px] px-2">{INDICATORS.map(i => <option key={i.id} value={i.id}>{i.name}</option>)}</select></div>
        <div><Label>Score scale</Label><select value={normalize} onChange={e => setNormalize(e.target.value as any)} className="h-8 border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12px] px-2"><option value="raw">Raw [0,1]</option><option value="percentile">Percentile rank</option></select></div>
        <div className="text-[11px] text-slate-500 self-end pb-1">Field for temporal panel: <b>{selectedField}</b> · Range {DATES[range[0]].slice(0, 7)}→{DATES[range[1]].slice(0, 7)}</div>
      </div>

      <div className="grid grid-cols-12 gap-4">
        <Panel title="Score distributions" subtitle={normalize === 'raw' ? 'Share of observations per score bin (all fields)' : 'Percentile-ranked scores — use for cross-model calibration comparison'} className="col-span-12 xl:col-span-7">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={distData} margin={{ top: 5, right: 15, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="bin" tick={{ fontSize: 10 }} label={{ value: 'Score bin', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {compareModels.map(m => <Bar key={m} dataKey={m} name={MODELS.find(x => x.id === m)?.name} fill={MODELS.find(x => x.id === m)?.color} fillOpacity={0.75} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Five-number summary (boxplot data)" subtitle="Min · P25 · median · P75 · max — raw scores" className="col-span-12 xl:col-span-5" pad={false}>
          <table className="w-full text-[11px]">
            <thead><tr className="bg-slate-50 dark:bg-slate-800"><th className="text-left px-3 py-2 font-medium">Model</th>{['Min', 'P25', 'Med', 'P75', 'Max'].map(h => <th key={h} className="px-2 py-2 font-medium text-right">{h}</th>)}</tr></thead>
            <tbody>{box.map(b => <tr key={b.model} className="border-t border-slate-100 dark:border-slate-800"><td className="px-3 py-2 font-medium"><span className="inline-block w-2 h-2 mr-1.5" style={{ background: b.color }} />{b.model}</td>{[b.min, b.p25, b.median, b.p75, b.max].map((v, i) => <td key={i} className="px-2 py-2 text-right font-mono">{v.toFixed(2)}</td>)}</tr>)}</tbody>
          </table>
          <div className="p-3">
            {box.map(b => <div key={b.model} className="flex items-center gap-2 mb-1.5 text-[10px]"><span className="w-24 text-right truncate">{b.model}</span><div className="flex-1 relative h-4 bg-slate-100 dark:bg-slate-800"><div className="absolute top-0 h-full" style={{ left: `${b.min * 100}%`, width: `${(b.max - b.min) * 100}%`, background: b.color + '33', borderLeft: `2px solid ${b.color}`, borderRight: `2px solid ${b.color}` }} /><div className="absolute top-0 h-full w-0.5 bg-slate-900 dark:bg-white" style={{ left: `${b.median * 100}%` }} /></div></div>)}
          <div className="text-[10px] text-slate-400">Bar = range, tick = median. Wide boxes = less concentrated scores.</div>
          </div>
        </Panel>

        <Panel title="Pairwise model correlation" subtitle="Pearson r between model scores (all field-months)" className="col-span-12 md:col-span-6" pad={false}>
          <table className="w-full text-[11px]">
            <thead><tr className="bg-slate-50 dark:bg-slate-800"><th className="px-3 py-2" /><th className="px-2 py-2 font-medium text-center">Mean</th><th className="px-2 py-2 font-medium text-center">Anom. rate</th><th className="px-2 py-2 font-medium text-center">r vs {INDICATORS.find(i => i.id === indicator)?.name}</th></tr></thead>
            <tbody>{indCorr.map(c => <tr key={c.model} className="border-t border-slate-100 dark:border-slate-800"><td className="px-3 py-2 font-medium"><span className="inline-block w-2 h-2 mr-1.5" style={{ background: c.color }} />{c.model}</td><td className="px-2 py-2 text-center font-mono">{c.mean.toFixed(3)}</td><td className="px-2 py-2 text-center font-mono">{(c.anomRate * 100).toFixed(1)}%</td><td className="px-2 py-2 text-center font-mono">{fmt(c.r)}</td></tr>)}</tbody>
          </table>
          <div className="p-3 overflow-auto"><table className="w-full text-[11px]"><thead><tr><th className="px-2 py-1" />{compareModels.map(m => <th key={m} className="px-2 py-1 font-medium text-center">{MODELS.find(x => x.id === m)?.short}</th>)}</tr></thead><tbody>{pairMatrix.map(r => <tr key={r.model}><td className="px-2 py-1 font-medium">{r.model}</td>{compareModels.map(m => { const v = r[MODELS.find(x => x.id === m)?.short || m]; return <td key={m} className="px-1 py-1 text-center"><span className="inline-block min-w-10 px-1 py-0.5 font-mono text-white" style={{ background: `rgba(15,118,110,${Math.max(0.15, v)})` }}>{v.toFixed(2)}</span></td>; })}</tr>)}</tbody></table></div>
        </Panel>

        <Panel title="Event agreement (Jaccard)" subtitle={`Overlap of anomalous events (≥${ANOMALY_THRESHOLD}) — 1.0 = identical event sets`} className="col-span-12 md:col-span-6" pad={false}>
          <div className="p-3 overflow-auto"><table className="w-full text-[11px]"><thead><tr><th className="px-2 py-1" />{compareModels.map(m => <th key={m} className="px-2 py-1 font-medium text-center">{MODELS.find(x => x.id === m)?.short}</th>)}</tr></thead><tbody>{agreement.map(r => <tr key={r.model}><td className="px-2 py-1 font-medium">{r.model}</td>{compareModels.map(m => { const v = r[MODELS.find(x => x.id === m)?.short || m]; return <td key={m} className="px-1 py-1 text-center"><span className="inline-block min-w-10 px-1 py-0.5 font-mono text-white" style={{ background: `rgba(37,99,235,${Math.max(0.12, v)})` }}>{v.toFixed(2)}</span></td>; })}</tr>)}</tbody></table></div>
          <div className="px-3 pb-3 text-[11px] text-slate-500">Low Jaccard with high correlation = models agree on ranking but disagree on thresholded events. Consider percentile calibration before event comparison.</div>
        </Panel>

        <Panel title={`Synchronised temporal — ${selectedField}`} subtitle="All selected models on one axis (raw scores are comparable in [0,1])" className="col-span-12 xl:col-span-7">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={temporal} margin={{ top: 5, right: 15, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 9 }} interval={5} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                {compareModels.map(m => <Line key={m} type="monotone" dataKey={m} name={MODELS.find(x => x.id === m)?.name} stroke={MODELS.find(x => x.id === m)?.color} strokeWidth={1.8} dot={false} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Pairwise scatter" subtitle={compareModels.length >= 2 ? `${MODELS.find(m => m.id === compareModels[0])?.name} vs ${MODELS.find(m => m.id === compareModels[1])?.name} — each point a field-month` : 'Select at least two models'} className="col-span-12 xl:col-span-5">
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{ fontSize: 10 }} />
                <YAxis type="number" dataKey="y" domain={[0, 1]} tick={{ fontSize: 10 }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: 11 }} />
                <Scatter data={pairScatter} fill="#0f766e" fillOpacity={0.4} />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[11px] text-slate-500">Points off the diagonal = model disagreement. Clusters at extremes suggest shared saturation.</div>
        </Panel>

        <Panel title="Field-level consistency" subtitle="Mean score per field — sorted by spread across models (disagreement)" className="col-span-12" pad={false}>
          <div className="overflow-auto max-h-72"><table className="w-full text-[11px]">
            <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800"><tr><th className="text-left px-3 py-2 font-medium">Field</th>{compareModels.map(m => <th key={m} className="px-2 py-2 font-medium text-right">{MODELS.find(x => x.id === m)?.short}</th>)}<th className="px-2 py-2 font-medium text-right">Spread (max−min)</th></tr></thead>
            <tbody>{FIELDS.map(f => { const means = compareModels.map(m => mean(getSeries(f.id).map(o => o.scores[m]))); const spread = Math.max(...means) - Math.min(...means); return { f, means, spread }; }).sort((a, b) => b.spread - a.spread).slice(0, 20).map(({ f, means, spread }) => (
              <tr key={f.id} className="border-t border-slate-100 dark:border-slate-800"><td className="px-3 py-1.5 font-mono">{f.code}</td>{means.map((v, i) => <td key={i} className="px-2 py-1.5 text-right font-mono">{v.toFixed(3)}</td>)}<td className="px-2 py-1.5 text-right"><Badge color={spread > 0.15 ? '#be123c' : '#64748b'}>{spread.toFixed(3)}</Badge></td></tr>
            ))}</tbody>
          </table></div>
        </Panel>
      </div>
    </div>
  );
}

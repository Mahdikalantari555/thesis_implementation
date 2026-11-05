import { useMemo, useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Line, ComposedChart, ReferenceLine } from 'recharts';
import { Panel, Label, Select, Stat, fmt } from '../components/ui';
import { useAnalysis } from '../context/AnalysisContext';
import { MODELS, INDICATORS, FIELDS, getPairs, pearson, spearman, linReg, laggedCorr, DATES } from '../data/mock';

export default function Spectral() {
  const { model, setModel, indicator, setIndicator, selectedFields, range, setRange } = useAnalysis();
  const [season, setSeason] = useState<'all' | 'wet' | 'dry'>('all');
  const [scope, setScope] = useState<'selection' | 'all'>('selection');

  const fieldIds = scope === 'all' ? FIELDS.map(f => f.id) : selectedFields;
  const { xs, ys, meta } = useMemo(() => {
    const p = getPairs(model, indicator, fieldIds, range);
    if (season === 'all') return p;
    const keep: number[] = [];
    p.meta.forEach((m, i) => { const mo = parseInt(m.date.slice(5, 7)); const ok = season === 'wet' ? (mo >= 10 || mo <= 3) : (mo >= 4 && mo <= 9); if (ok) keep.push(i); });
    return { xs: keep.map(i => p.xs[i]), ys: keep.map(i => p.ys[i]), meta: keep.map(i => p.meta[i]) };
  }, [model, indicator, fieldIds, range, season]);

  const r = pearson(xs, ys), rho = spearman(xs, ys), reg = linReg(xs, ys);
  const scatter = xs.map((x, i) => ({ x: +x.toFixed(3), y: +ys[i].toFixed(3), field: meta[i].field, date: meta[i].date.slice(0, 7) }));
  const lineData = Array.from({ length: 20 }, (_, i) => { const x = i / 19; return { x: +x.toFixed(3), y: +(reg.intercept + reg.slope * x).toFixed(3) }; });
  const lags = laggedCorr(xs, ys, 6);
  const lagData = lags.map(l => ({ lag: l.lag, r: +l.r.toFixed(3) }));

  // Correlation matrix: models × indicators
  const matrix = useMemo(() => {
    return MODELS.map(m => {
      const row: any = { model: m.name, color: m.color };
      INDICATORS.filter(i => i.kind !== 'weather').forEach(ind => {
        const p = getPairs(m.id, ind.id, fieldIds, range);
        row[ind.name] = +pearson(p.xs, p.ys).toFixed(2);
      });
      return row;
    });
  }, [fieldIds, range]);

  const indCols = INDICATORS.filter(i => i.kind !== 'weather');

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-3 items-end border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3">
        <div><Label>Model</Label><Select value={model} onChange={v => setModel(v as any)}>{MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</Select></div>
        <div><Label>Indicator</Label><Select value={indicator} onChange={v => setIndicator(v as any)}>{INDICATORS.map(i => <option key={i.id} value={i.id}>{i.name}</option>)}</Select></div>
        <div><Label>Scope</Label><Select value={scope} onChange={v => setScope(v as any)}><option value="selection">Selected fields ({selectedFields.length})</option><option value="all">All {FIELDS.length} fields</option></Select></div>
        <div><Label>Season</Label><Select value={season} onChange={v => setSeason(v as any)}><option value="all">All year</option><option value="wet">Wet (Oct–Mar)</option><option value="dry">Dry (Apr–Sep)</option></Select></div>
        <div className="flex-1 min-w-48"><Label>Date range — {DATES[range[0]].slice(0, 7)} → {DATES[range[1]].slice(0, 7)}</Label><div className="flex gap-2"><input type="range" min={0} max={47} value={range[0]} onChange={e => setRange([+e.target.value, range[1]])} className="flex-1 accent-teal-700" /><input type="range" min={0} max={47} value={range[1]} onChange={e => setRange([range[0], +e.target.value])} className="flex-1 accent-teal-700" /></div></div>
        <div className="text-[11px] text-slate-500">n = <b className="text-slate-800 dark:text-slate-200">{xs.length}</b> paired obs</div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-px bg-slate-200 dark:bg-slate-800 border border-slate-200 dark:border-slate-800">
        <Stat label="Pearson r" value={fmt(r)} sub={r > 0.5 ? 'strong +' : r > 0.3 ? 'moderate +' : r < -0.3 ? 'moderate −' : 'weak'} />
        <Stat label="Spearman ρ" value={fmt(rho)} sub="rank-based, robust" />
        <Stat label="Regression slope" value={fmt(reg.slope)} sub={`R² = ${fmt(reg.r2)}`} />
        <Stat label="Best lag" value={lags.reduce((a, b) => Math.abs(b.r) > Math.abs(a.r) ? b : a, lags[0]).lag + ' mo'} sub="max |r| across lags" />
      </div>

      <div className="grid grid-cols-12 gap-4">
        <Panel title={`Scatter — ${MODELS.find(m => m.id === model)?.name} vs ${INDICATORS.find(i => i.id === indicator)?.name}`} subtitle="Each point = one field-month · line = OLS fit · hover for field & date" className="col-span-12 xl:col-span-7">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" dataKey="x" name="Score" domain={[0, 1]} tick={{ fontSize: 10 }} label={{ value: 'Anomaly score', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                <YAxis type="number" dataKey="y" name={indicator} tick={{ fontSize: 10 }} label={{ value: INDICATORS.find(i => i.id === indicator)?.name, angle: -90, position: 'insideLeft', fontSize: 10 }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: 11 }} />
                <Scatter data={scatter} fill={MODELS.find(m => m.id === model)?.color} fillOpacity={0.55} />
                <Line type="monotone" dataKey="y" data={lineData} stroke="#be123c" strokeWidth={2} dot={false} name="OLS fit" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Lagged correlation" subtitle="Score at t vs indicator at t+lag (months). Positive lag = indicator leads." className="col-span-12 xl:col-span-5">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={lagData} margin={{ top: 10, right: 15, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="lag" tick={{ fontSize: 10 }} label={{ value: 'Lag (months)', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                <YAxis domain={[-1, 1]} tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <ReferenceLine y={0} stroke="#94a3b8" />
                <Line type="monotone" dataKey="r" stroke="#0f766e" strokeWidth={2} dot={{ r: 3 }} name="Pearson r" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Peak at lag 0 suggests contemporaneous association; offset peaks may indicate lead/lag or seasonal confounding — investigate, do not over-interpret.</div>
        </Panel>

        <Panel title="Correlation matrix — models × indicators" subtitle="Pearson r across selected scope & range · red = negative, teal = positive" className="col-span-12" pad={false}>
          <div className="overflow-auto">
            <table className="w-full text-[11px]">
              <thead><tr className="bg-slate-50 dark:bg-slate-800"><th className="text-left px-3 py-2 font-medium">Model</th>{indCols.map(i => <th key={i.id} className="px-2 py-2 font-medium text-center">{i.name}</th>)}</tr></thead>
              <tbody>{matrix.map(row => (
                <tr key={row.model} className="border-t border-slate-100 dark:border-slate-800">
                  <td className="px-3 py-1.5 font-medium"><span className="inline-block w-2 h-2 mr-1.5" style={{ background: row.color }} />{row.model}</td>
                  {indCols.map(i => { const v = row[i.name] as number; const bg = v > 0.3 ? `rgba(15,118,110,${Math.min(0.85, v)})` : v < -0.3 ? `rgba(190,18,60,${Math.min(0.85, -v)})` : '#f1f5f9'; const fg = Math.abs(v) > 0.3 ? '#fff' : '#334155'; return <td key={i.id} className="px-1 py-1 text-center"><span className="inline-block min-w-12 px-1 py-0.5 font-mono" style={{ background: bg, color: fg }}>{v.toFixed(2)}</span></td>; })}
                </tr>))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title="Distribution comparison" subtitle="Score vs indicator quantiles (selected scope)" className="col-span-12 xl:col-span-6">
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(q => ({ q: `P${q * 10}`, score: +[...xs].sort((a, b) => a - b)[Math.floor(q / 10 * (xs.length - 1))]?.toFixed(3) || 0, ind: +[...ys].sort((a, b) => a - b)[Math.floor(q / 10 * (ys.length - 1))]?.toFixed(3) || 0 }))} margin={{ top: 5, right: 30, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="q" tick={{ fontSize: 10 }} />
                <YAxis yAxisId="l" domain={[0, 1]} tick={{ fontSize: 10 }} />
                <YAxis yAxisId="r" orientation="right" tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <Line yAxisId="l" type="monotone" dataKey="score" stroke={MODELS.find(m => m.id === model)?.color} strokeWidth={2} dot={false} name="Score quantile" />
                <Line yAxisId="r" type="monotone" dataKey="ind" stroke={INDICATORS.find(i => i.id === indicator)?.color} strokeWidth={2} dot={false} name="Indicator quantile" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Interpretation notes" className="col-span-12 xl:col-span-6">
          <ul className="text-[12px] text-slate-600 dark:text-slate-400 space-y-2 list-disc pl-4 leading-relaxed">
            <li>Correlations here are <b>associations</b>, not diagnostic accuracy. A high |r| does not prove the anomaly is real vegetation stress.</li>
            <li>Check lag structure and seasonal splits before claiming a physical relationship — phenology can induce spurious correlation.</li>
            <li>Use the Validation Lab for confidence intervals, partial correlations, and event-level agreement.</li>
            <li>Switch scope to “All fields” to see whether the relationship holds beyond the current selection.</li>
          </ul>
        </Panel>
      </div>
    </div>
  );
}

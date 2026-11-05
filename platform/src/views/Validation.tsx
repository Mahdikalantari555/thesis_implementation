import { useMemo, useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Line, ComposedChart, ReferenceLine, BarChart, Bar, Cell } from 'recharts';
import { Panel, Label, Select, Stat, SigBadge, fmt, fmt2 } from '../components/ui';
import { useAnalysis } from '../context/AnalysisContext';
import { MODELS, INDICATORS, FIELDS, getPairs, pearson, spearman, kendall, partialCorr, linReg, fisherCI, pValue, laggedCorr, DATES, ANOMALY_THRESHOLD, type IndicatorId } from '../data/mock';
import { ShieldAlert, Info } from 'lucide-react';

export default function Validation() {
  const { model, setModel, indicator, setIndicator, selectedFields, range, setRange, addExperiment } = useAnalysis();
  const [control, setControl] = useState<IndicatorId>('ndmi');
  const [scope, setScope] = useState<'selection' | 'all'>('selection');
  const [saved, setSaved] = useState(false);

  const fieldIds = scope === 'all' ? FIELDS.map(f => f.id) : selectedFields;
  const { xs, ys } = useMemo(() => getPairs(model, indicator, fieldIds, range), [model, indicator, fieldIds, range]);
  const ctrl = useMemo(() => getPairs(model, control, fieldIds, range), [model, control, fieldIds, range]);

  const n = xs.length;
  const r = pearson(xs, ys), rho = spearman(xs, ys), tau = kendall(xs, ys);
  const pr = ctrl.xs.length === n ? partialCorr(xs, ys, ctrl.ys) : NaN;
  const reg = linReg(xs, ys);
  const [lo, hi] = fisherCI(r, n);
  const p = pValue(r, n);
  const lags = laggedCorr(xs, ys, 6);
  const lagData = lags.map(l => ({ lag: l.lag, r: +l.r.toFixed(3) }));

  // Event-level agreement: score >= thr vs indicator in stress direction
  const events = useMemo(() => {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    const stressHigh = ['tvdi', 'cwsi'].includes(indicator);
    const thr = indicator === 'ndvi' || indicator === 'evi' || indicator === 'savi' ? 0.45 : indicator === 'ndmi' ? 0.2 : indicator === 'eta' ? 2.5 : 0.6;
    xs.forEach((x, i) => {
      const pred = x >= ANOMALY_THRESHOLD;
      const actual = stressHigh ? ys[i] >= thr : ys[i] <= thr;
      if (pred && actual) tp++; else if (pred && !actual) fp++; else if (!pred && actual) fn++; else tn++;
    });
    const sens = tp + fn ? tp / (tp + fn) : NaN, spec = tn + fp ? tn / (tn + fp) : NaN, prec = tp + fp ? tp / (tp + fp) : NaN;
    const kappaNum = (tp + tn) / Math.max(1, n);
    return { tp, fp, fn, tn, sens, spec, prec, kappaNum, thr };
  }, [xs, ys, indicator, n]);

  // Robustness by year
  const byYear = useMemo(() => [0, 1, 2, 3].map(y => {
    const p2 = getPairs(model, indicator, fieldIds, [y * 12, y * 12 + 11]);
    return { year: 2021 + y, r: +pearson(p2.xs, p2.ys).toFixed(3), n: p2.xs.length };
  }), [model, indicator, fieldIds]);

  // Temporal consistency: split-half correlation
  const half = Math.floor(n / 2);
  const r1 = pearson(xs.slice(0, half), ys.slice(0, half)), r2 = pearson(xs.slice(half), ys.slice(half));

  const scatter = xs.map((x, i) => ({ x: +x.toFixed(3), y: +ys[i].toFixed(3) }));
  const fit = Array.from({ length: 20 }, (_, i) => { const x = i / 19; return { x: +x.toFixed(3), y: +(reg.intercept + reg.slope * x).toFixed(3) }; });

  const save = () => {
    addExperiment({ id: `exp-${Date.now()}`, name: `${MODELS.find(m => m.id === model)?.name} vs ${INDICATORS.find(i => i.id === indicator)?.name} — ${scope === 'all' ? 'all fields' : selectedFields.length + ' fields'}`, created: new Date().toISOString().slice(0, 10), model, indicator, fields: fieldIds, range, note: 'Saved from Validation Lab', stats: `r=${fmt(r)} ρ=${fmt(rho)} τ=${fmt(tau)} n=${n}` });
    setSaved(true); setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="space-y-4">
      {/* Disclaimer */}
      <div className="border border-amber-300 bg-amber-50 dark:bg-amber-950/30 dark:border-amber-800 p-3 flex gap-3">
        <ShieldAlert size={18} className="text-amber-700 dark:text-amber-400 shrink-0 mt-0.5" />
        <div className="text-[12px] leading-relaxed text-amber-900 dark:text-amber-200">
          <b>Association ≠ diagnostic accuracy.</b> Correlation between an anomaly score and a proxy indicator (NDVI, TVDI, CWSI…) shows statistical association only. It does <b>not</b> prove that a detected anomaly represents real vegetation stress, nor that the model is clinically/agronomically accurate. Ground-truth field validation is required for diagnostic claims. All statistics below use <b>demonstration (mock) data</b>.
        </div>
      </div>

      <div className="flex flex-wrap gap-3 items-end border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-3">
        <div><Label>Model</Label><Select value={model} onChange={v => setModel(v as any)}>{MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</Select></div>
        <div><Label>Indicator</Label><Select value={indicator} onChange={v => setIndicator(v as any)}>{INDICATORS.map(i => <option key={i.id} value={i.id}>{i.name}</option>)}</Select></div>
        <div><Label>Control (partial r)</Label><Select value={control} onChange={v => setControl(v as any)}>{INDICATORS.map(i => <option key={i.id} value={i.id}>{i.name}</option>)}</Select></div>
        <div><Label>Scope</Label><Select value={scope} onChange={v => setScope(v as any)}><option value="selection">Selected fields</option><option value="all">All 48 fields</option></Select></div>
        <div className="flex-1 min-w-48"><Label>Range {DATES[range[0]].slice(0, 7)} → {DATES[range[1]].slice(0, 7)}</Label><div className="flex gap-2"><input type="range" min={0} max={47} value={range[0]} onChange={e => setRange([+e.target.value, range[1]])} className="flex-1 accent-teal-700" /><input type="range" min={0} max={47} value={range[1]} onChange={e => setRange([range[0], +e.target.value])} className="flex-1 accent-teal-700" /></div></div>
        <button onClick={save} className="h-8 px-3 bg-teal-700 text-white text-[12px]">{saved ? '✓ Saved' : 'Save experiment'}</button>
      </div>

      {/* Metadata strip */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-px bg-slate-200 dark:bg-slate-800 border border-slate-200 dark:border-slate-800">
        <Stat label="Sample size (n)" value={String(n)} sub={`${fieldIds.length} fields × ${range[1] - range[0] + 1} mo`} />
        <Stat label="Pearson r [95% CI]" value={`${fmt(r)}`} sub={`[${fmt(lo)}, ${fmt(hi)}]`} />
        <Stat label="Spearman ρ" value={fmt(rho)} sub="monotonic association" />
        <Stat label="Kendall τ" value={fmt(tau)} sub="rank concordance" />
        <Stat label="Partial r (ctrl)" value={fmt(pr)} sub={`controlling ${INDICATORS.find(i => i.id === control)?.name}`} />
      </div>

      <div className="grid grid-cols-12 gap-4">
        <Panel title="Scatter with OLS fit & CI context" subtitle={`Slope ${fmt(reg.slope)} · intercept ${fmt(reg.intercept)} · R² ${fmt(reg.r2)}`} className="col-span-12 xl:col-span-7">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis type="number" dataKey="x" domain={[0, 1]} tick={{ fontSize: 10 }} label={{ value: 'Anomaly score', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                <YAxis type="number" dataKey="y" tick={{ fontSize: 10 }} label={{ value: INDICATORS.find(i => i.id === indicator)?.name, angle: -90, position: 'insideLeft', fontSize: 10 }} />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} contentStyle={{ fontSize: 11 }} />
                <Scatter data={scatter} fill={MODELS.find(m => m.id === model)?.color} fillOpacity={0.5} />
                <Line type="monotone" dataKey="y" data={fit} stroke="#be123c" strokeWidth={2} dot={false} name="OLS" />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="flex items-center gap-2 text-[11px]"><SigBadge p={p} /><span className="text-slate-500">Two-tailed p (normal approx.) · Fisher-z CI · n={n}</span></div>
        </Panel>

        <Panel title="Statistical summary" subtitle="All methods computed on the same paired sample" className="col-span-12 xl:col-span-5" pad={false}>
          <table className="w-full text-[12px]">
            <tbody>
              {[['Pearson r', fmt(r), 'linear association'], ['Spearman ρ', fmt(rho), 'monotonic, rank-based'], ['Kendall τ', fmt(tau), 'pairwise concordance'], ['Partial r', fmt(pr), `given ${INDICATORS.find(i => i.id === control)?.name}`], ['Slope (OLS)', fmt(reg.slope), 'Δ indicator per unit score'], ['R²', fmt(reg.r2), 'variance explained (linear)'], ['95% CI (Fisher)', `[${fmt(lo)}, ${fmt(hi)}]`, 'for Pearson r'], ['p-value', p < 0.001 ? '<0.001' : fmt(p, 4), 'two-tailed']].map(([k, v, d]) => (
                <tr key={k} className="border-b border-slate-100 dark:border-slate-800"><td className="px-4 py-2 font-medium">{k}</td><td className="px-2 py-2 font-mono text-right">{v}</td><td className="px-4 py-2 text-slate-500 text-[11px]">{d}</td></tr>
              ))}
            </tbody>
          </table>
          <div className="p-3 text-[11px] text-slate-500 flex gap-1.5"><Info size={13} className="shrink-0 mt-0.5" /> Partial correlation removes linear influence of the control indicator. If partial r drops sharply, the raw association may be confounded (e.g., by soil moisture).</div>
        </Panel>

        <Panel title="Lagged correlation" subtitle="Score(t) vs indicator(t+lag) — check for lead/lag structure" className="col-span-12 md:col-span-6">
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={lagData} margin={{ top: 5, right: 15, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="lag" tick={{ fontSize: 10 }} label={{ value: 'Lag (months)', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                <YAxis domain={[-1, 1]} tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <ReferenceLine y={0} stroke="#94a3b8" />
                <Line type="monotone" dataKey="r" stroke="#0f766e" strokeWidth={2} dot={{ r: 3 }} name="r" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Event-level agreement (proxy)" subtitle={`Score ≥ ${ANOMALY_THRESHOLD} vs indicator stress threshold (${events.thr}) — NOT ground truth`} className="col-span-12 md:col-span-6" pad={false}>
          <div className="p-4">
            <div className="grid grid-cols-2 gap-px bg-slate-200 dark:bg-slate-800 border border-slate-200 dark:border-slate-800 text-center text-[12px]">
              <div className="bg-white dark:bg-slate-900 p-3"><div className="text-[10px] uppercase text-slate-400">TP / FP</div><div className="font-mono font-semibold">{events.tp} / {events.fp}</div></div>
              <div className="bg-white dark:bg-slate-900 p-3"><div className="text-[10px] uppercase text-slate-400">FN / TN</div><div className="font-mono font-semibold">{events.fn} / {events.tn}</div></div>
            </div>
            <div className="grid grid-cols-3 gap-2 mt-3 text-center">
              <div><div className="text-[10px] text-slate-400 uppercase">Sensitivity*</div><div className="font-semibold">{fmt2(events.sens)}</div></div>
              <div><div className="text-[10px] text-slate-400 uppercase">Specificity*</div><div className="font-semibold">{fmt2(events.spec)}</div></div>
              <div><div className="text-[10px] text-slate-400 uppercase">Precision*</div><div className="font-semibold">{fmt2(events.prec)}</div></div>
            </div>
            <div className="text-[10px] text-slate-400 mt-2">* Relative to proxy threshold only — not diagnostic performance. Agreement rate {(events.kappaNum * 100).toFixed(1)}%.</div>
          </div>
        </Panel>

        <Panel title="Robustness by year" subtitle="Pearson r recomputed per calendar year (same fields & model)" className="col-span-12 md:col-span-6">
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={byYear} margin={{ top: 5, right: 15, left: -15, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="year" tick={{ fontSize: 10 }} />
                <YAxis domain={[-1, 1]} tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 11 }} />
                <ReferenceLine y={0} stroke="#94a3b8" />
                <Bar dataKey="r" name="Pearson r">{byYear.map((d, i) => <Cell key={i} fill={d.r >= 0 ? '#0f766e' : '#be123c'} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Temporal consistency" subtitle="Split-half reliability of the association" className="col-span-12 md:col-span-6">
          <div className="space-y-3 text-[12px]">
            <div className="flex justify-between border-b border-slate-100 dark:border-slate-800 pb-2"><span>First half r</span><b className="font-mono">{fmt(r1)}</b></div>
            <div className="flex justify-between border-b border-slate-100 dark:border-slate-800 pb-2"><span>Second half r</span><b className="font-mono">{fmt(r2)}</b></div>
            <div className="flex justify-between border-b border-slate-100 dark:border-slate-800 pb-2"><span>Full sample r</span><b className="font-mono">{fmt(r)}</b></div>
            <div className="text-[11px] text-slate-500 leading-relaxed">Large split-half differences suggest the association is unstable over time — check for regime changes, sensor differences, or phenological effects before pooling.</div>
            <div className="border border-slate-200 dark:border-slate-800 p-2">
              <div className="text-[10px] uppercase text-slate-400 mb-1">Analysis metadata</div>
              <div className="font-mono text-[10px] leading-relaxed text-slate-600 dark:text-slate-400">
                model={model} · indicator={indicator} · control={control}<br />
                fields=[{fieldIds.slice(0, 4).join(',')}{fieldIds.length > 4 ? `, … (${fieldIds.length})` : ''}]<br />
                range={DATES[range[0]].slice(0, 7)}→{DATES[range[1]].slice(0, 7)} · n={n} · method=pearson/spearman/kendall/partial
              </div>
            </div>
          </div>
        </Panel>
      </div>
    </div>
  );
}

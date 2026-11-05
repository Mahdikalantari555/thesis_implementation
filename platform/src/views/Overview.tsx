import { useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, PieChart, Pie, LineChart, Line, CartesianGrid, Legend } from 'recharts';
import { FIELDS, MODELS, INDICATORS, getSeries, DATES, mean, quantile, ANOMALY_THRESHOLD } from '../data/mock';
import { Panel, Stat, Badge, anomalyColor } from '../components/ui';
import { useAnalysis } from '../context/AnalysisContext';
import { AlertTriangle, CalendarDays, Database, Layers, MapPin } from 'lucide-react';

export default function Overview() {
  const { setView, setSelectedField, model, setModel, range } = useAnalysis();

  const stats = useMemo(() => {
    let totalObs = 0, missing = 0;
    const allScores: number[] = [];
    const perModel: Record<string, number[]> = {};
    MODELS.forEach(m => perModel[m.id] = []);
    FIELDS.forEach(f => getSeries(f.id).forEach(o => {
      totalObs++;
      if (o.missing) missing++;
      MODELS.forEach(m => { perModel[m.id].push(o.scores[m.id]); if (m.id === model) allScores.push(o.scores[m.id]); });
    }));
    const hist = Array.from({ length: 10 }, (_, i) => ({ bin: `${(i / 10).toFixed(1)}–${((i + 1) / 10).toFixed(1)}`, count: allScores.filter(v => v >= i / 10 && v < (i + 1) / 10).length }));
    const monthly = DATES.map((d, i) => {
      const vals = FIELDS.flatMap(f => { const o = getSeries(f.id)[i]; return o.missing ? [] : [o.scores[model]]; });
      return { date: d.slice(0, 7), mean: +mean(vals).toFixed(3), p90: +quantile(vals, 0.9).toFixed(3), fracAnom: +(vals.filter(v => v >= ANOMALY_THRESHOLD).length / vals.length).toFixed(3) };
    });
    const completeness = 100 * (1 - missing / totalObs);
    const anomalies = FIELDS.flatMap(f => getSeries(f.id).map(o => ({ field: f, obs: o }))).filter(x => !x.obs.missing && x.obs.scores[model] >= ANOMALY_THRESHOLD).sort((a, b) => b.obs.scores[model] - a.obs.scores[model]).slice(0, 8);
    const fieldYear = FIELDS.slice(0, 24).map(f => {
      const s = getSeries(f.id);
      return { field: f.code, '2021': +mean(s.slice(0, 12).map(o => o.scores[model])).toFixed(2), '2022': +mean(s.slice(12, 24).map(o => o.scores[model])).toFixed(2), '2023': +mean(s.slice(24, 36).map(o => o.scores[model])).toFixed(2), '2024': +mean(s.slice(36, 48).map(o => o.scores[model])).toFixed(2) };
    });
    return { totalObs, missing, completeness, hist, monthly, anomalies, fieldYear, perModel };
  }, [model]);

  const totalArea = FIELDS.reduce((s, f) => s + f.areaHa, 0);

  return (
    <div className="space-y-4">
      {/* Dataset strip */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-6 gap-px bg-slate-200 dark:bg-slate-800 border border-slate-200 dark:border-slate-800">
        <Stat label="Fields" value={String(FIELDS.length)} sub={`${totalArea.toLocaleString()} ha total`} />
        <Stat label="Observations" value={stats.totalObs.toLocaleString()} sub="field × month (Khuzestan)" />
        <Stat label="Temporal coverage" value="2021–2024" sub="48 monthly obs / field" />
        <Stat label="Completeness" value={`${stats.completeness.toFixed(1)}%`} sub={`${stats.missing} missing (cloud / gap)`} />
        <Stat label="Anomaly models" value={String(MODELS.length)} sub="self-supervised" />
        <Stat label="Indicators" value={String(INDICATORS.length)} sub="spectral · thermal · water" />
      </div>

      <div className="grid grid-cols-12 gap-4">
        <Panel title="Anomaly score distribution" subtitle={`Model: ${MODELS.find(m => m.id === model)?.name} · all fields & dates · threshold ${ANOMALY_THRESHOLD}`} className="col-span-12 xl:col-span-5"
          right={<select value={model} onChange={e => setModel(e.target.value as any)} className="h-7 border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[11px] px-1.5">{MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select>}>
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.hist} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="bin" tick={{ fontSize: 10 }} interval={1} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 12 }} />
                <Bar dataKey="count" name="Observations">{stats.hist.map((_, i) => <Cell key={i} fill={anomalyColor((i + 0.5) / 10)} />)}</Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="flex gap-4 mt-2 text-[11px] text-slate-500"><span>■ &lt;0.35 nominal</span><span>■ 0.35–0.65 elevated</span><span>■ ≥0.65 anomalous</span></div>
        </Panel>

        <Panel title="Dataset completeness" subtitle="Missing observations by cause (demonstration)" className="col-span-12 md:col-span-6 xl:col-span-3">
          <div className="h-44 flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={[{ name: 'Valid', value: stats.completeness }, { name: 'Missing', value: 100 - stats.completeness }]} dataKey="value" innerRadius={52} outerRadius={72} paddingAngle={2}>
                  <Cell fill="#0f766e" /><Cell fill="#cbd5e1" />
                </Pie>
                <Tooltip contentStyle={{ fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="text-center text-[12px] text-slate-600 dark:text-slate-400"><b className="text-slate-900 dark:text-slate-100">{stats.completeness.toFixed(1)}%</b> valid · cloud cover, sensor gaps, processing failures</div>
        </Panel>

        <Panel title="Monthly anomaly burden" subtitle="Mean & 90th percentile score across fields" className="col-span-12 md:col-span-6 xl:col-span-4">
          <div className="h-56">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stats.monthly} margin={{ top: 5, right: 10, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis dataKey="date" tick={{ fontSize: 9 }} interval={5} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Line type="monotone" dataKey="mean" stroke="#0f766e" strokeWidth={2} dot={false} name="Mean" />
                <Line type="monotone" dataKey="p90" stroke="#ea580c" strokeWidth={1.5} dot={false} name="P90" strokeDasharray="4 3" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Panel>

        <Panel title="Mean score by field × year" subtitle="First 24 fields — look for persistent hotspots" className="col-span-12 xl:col-span-7" pad={false}>
          <div className="overflow-auto max-h-72">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-slate-50 dark:bg-slate-800"><tr><th className="text-left px-3 py-2 font-medium">Field</th>{['2021', '2022', '2023', '2024'].map(y => <th key={y} className="px-2 py-2 font-medium text-right">{y}</th>)}</tr></thead>
              <tbody>{stats.fieldYear.map(r => (
                <tr key={r.field} className="border-t border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/60 cursor-pointer" onClick={() => { setSelectedField(FIELDS.find(f => f.code === r.field)!.id); setView('map'); }}>
                  <td className="px-3 py-1.5 font-mono">{r.field}</td>
                  {['2021', '2022', '2023', '2024'].map(y => { const v = (r as any)[y] as number; return <td key={y} className="px-2 py-1.5 text-right"><span className="inline-block min-w-10 px-1.5 py-0.5 text-center text-white text-[10px]" style={{ background: anomalyColor(v) }}>{v.toFixed(2)}</span></td>; })}
                </tr>))}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title="Notable anomaly events" subtitle={`Top scores ≥ ${ANOMALY_THRESHOLD} — ${MODELS.find(m => m.id === model)?.name}`} className="col-span-12 xl:col-span-5" pad={false}>
          <div className="divide-y divide-slate-100 dark:divide-slate-800">
            {stats.anomalies.map((a, i) => (
              <button key={i} onClick={() => { setSelectedField(a.field.id); setView('temporal'); }} className="w-full flex items-center gap-3 px-4 py-2.5 text-left hover:bg-slate-50 dark:hover:bg-slate-800/60">
                <AlertTriangle size={14} className="text-orange-600 shrink-0" />
                <span className="flex-1 min-w-0"><span className="block text-[12px] font-medium truncate">{a.field.code} — {a.field.name} · {a.field.municipality}</span><span className="block text-[11px] text-slate-500">{a.obs.date.slice(0, 7)} · NDVI {a.obs.ndvi.toFixed(2)} · CWSI {a.obs.cwsi.toFixed(2)}</span></span>
                <Badge color={anomalyColor(a.obs.scores[model])}>{a.obs.scores[model].toFixed(3)}</Badge>
              </button>
            ))}
            {stats.anomalies.length === 0 && <div className="p-6 text-[12px] text-slate-500">No events above threshold for this model.</div>}
          </div>
        </Panel>

        <Panel title="Available models" subtitle="Self-supervised anomaly detectors — consistent colour encoding across the platform" className="col-span-12 xl:col-span-6" pad={false}>
          <div className="divide-y divide-slate-100 dark:divide-slate-800">
            {MODELS.map(m => { const vals = stats.perModel[m.id]; const anom = vals.filter(v => v >= ANOMALY_THRESHOLD).length / vals.length; return (
              <button key={m.id} onClick={() => { setModel(m.id); }} className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-slate-50 dark:hover:bg-slate-800/60">
                <span className="w-2.5 h-2.5 shrink-0" style={{ background: m.color }} />
                <span className="flex-1"><span className="block text-[12px] font-medium">{m.name} <span className="text-slate-400 font-normal">· {m.type}</span></span><span className="block text-[11px] text-slate-500">{m.desc}</span></span>
                <span className="text-right text-[11px] text-slate-500">mean {mean(vals).toFixed(3)}<br />{(anom * 100).toFixed(1)}% anomalous</span>
              </button>); })}
          </div>
        </Panel>

        <Panel title="External validation indicators" subtitle="Independent / proxy indicators available for association analysis" className="col-span-12 xl:col-span-6" pad={false}>
          <div className="grid grid-cols-2 sm:grid-cols-3">
            {INDICATORS.map(ind => (
              <div key={ind.id} className="border-b border-r border-slate-100 dark:border-slate-800 p-3">
                <div className="flex items-center gap-1.5"><span className="w-2 h-2" style={{ background: ind.color }} /><span className="text-[12px] font-semibold">{ind.name}</span><span className="text-[10px] text-slate-400">{ind.unit}</span></div>
                <div className="text-[10px] text-slate-500 mt-1 leading-snug">{ind.desc}</div>
                <Badge color={ind.color}>{ind.kind}</Badge>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      <div className="flex flex-wrap gap-2 text-[11px] text-slate-500 items-center">
        <CalendarDays size={13} /><span>Monthly composite · Sentinel-2 / Landsat-like cadence (mock)</span>
        <Database size={13} /><span>Demonstration data — not research results</span>
        <Layers size={13} /><span>Region: Piracicaba, SP, Brazil (sugarcane)</span>
        <MapPin size={13} /><span>Selected range: {range[0]}–{range[1]}</span>
      </div>
    </div>
  );
}

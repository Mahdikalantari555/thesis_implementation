import { useState } from 'react';
import { Panel, Badge, EmptyState } from '../components/ui';
import { useAnalysis } from '../context/AnalysisContext';
import { DATES, getSeries, getPairs, pearson, spearman, downloadCSV, modelById, indicatorById } from '../data/mock';
import { Download, Trash2, FileText, Plus, Image as ImageIcon } from 'lucide-react';

export default function Reports() {
  const { experiments, addExperiment, removeExperiment, model, indicator, selectedFields, range, selectedField, setView } = useAnalysis();
  const [name, setName] = useState('');
  const [note, setNote] = useState('');

  const create = () => {
    const p = getPairs(model, indicator, selectedFields, range);
    addExperiment({ id: `exp-${Date.now()}`, name: name || `${modelById(model).name} vs ${indicatorById(indicator).name} — ${selectedFields.length} fields`, created: new Date().toISOString().slice(0, 10), model, indicator, fields: selectedFields, range, note: note || 'Manual experiment', stats: `r=${pearson(p.xs, p.ys).toFixed(3)} ρ=${spearman(p.xs, p.ys).toFixed(3)} n=${p.xs.length}` });
    setName(''); setNote('');
  };

  const exportSeries = () => {
    const s = getSeries(selectedField);
    downloadCSV(`series_${selectedField}_${model}.csv`, [['date', 'missing', model, 'ndvi', 'evi', 'savi', 'ndmi', 'tvdi', 'cwsi', 'eta', 'precip', 'temp'], ...s.map(o => [o.date, o.missing ? 1 : 0, o.scores[model].toFixed(4), o.ndvi.toFixed(4), o.evi.toFixed(4), o.savi.toFixed(4), o.ndmi.toFixed(4), o.tvdi.toFixed(4), o.cwsi.toFixed(4), o.eta.toFixed(4), o.precip.toFixed(2), o.temp.toFixed(2)])]);
  };

  const exportPairs = () => {
    const p = getPairs(model, indicator, selectedFields, range);
    downloadCSV(`pairs_${model}_${indicator}.csv`, [['field', 'date', 'score', indicator], ...p.meta.map((m, i) => [m.field, m.date, p.xs[i].toFixed(4), p.ys[i].toFixed(4)])]);
  };

  const exportExperiments = () => {
    downloadCSV('experiments.csv', [['id', 'name', 'created', 'model', 'indicator', 'fields', 'range', 'stats', 'note'], ...experiments.map(e => [e.id, e.name, e.created, e.model, e.indicator, e.fields.join(';'), `${DATES[e.range[0]].slice(0, 7)}→${DATES[e.range[1]].slice(0, 7)}`, e.stats, e.note])]);
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-12 gap-4">
        <Panel title="Create experiment from current state" subtitle="Captures model, indicator, fields, and date range shared across views" className="col-span-12 lg:col-span-4">
          <div className="space-y-3">
            <div className="text-[12px] space-y-1">
              <div className="flex justify-between"><span className="text-slate-500">Model</span><Badge color={modelById(model).color}>{modelById(model).name}</Badge></div>
              <div className="flex justify-between"><span className="text-slate-500">Indicator</span><Badge color={indicatorById(indicator).color}>{indicatorById(indicator).name}</Badge></div>
              <div className="flex justify-between"><span className="text-slate-500">Fields</span><span>{selectedFields.length} selected</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Range</span><span>{DATES[range[0]].slice(0, 7)} → {DATES[range[1]].slice(0, 7)}</span></div>
              <div className="flex justify-between"><span className="text-slate-500">Focus field</span><span className="font-mono">{selectedField}</span></div>
            </div>
            <input value={name} onChange={e => setName(e.target.value)} placeholder="Experiment name…" className="w-full h-8 border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 text-[12px] outline-none" />
            <textarea value={note} onChange={e => setNote(e.target.value)} placeholder="Hypothesis / note for supervisor…" rows={3} className="w-full border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-2 py-1.5 text-[12px] outline-none" />
            <button onClick={create} className="w-full h-8 bg-teal-700 text-white text-[12px] flex items-center justify-center gap-2"><Plus size={13} /> Save experiment</button>
            <div className="text-[11px] text-slate-500">Tip: configure state in Map / Temporal / Validation, then save here. Experiments persist for the session.</div>
          </div>
        </Panel>

        <Panel title="Export & publication" subtitle="CSV exports of underlying data (figures: use browser print / screenshot for now)" className="col-span-12 lg:col-span-4">
          <div className="space-y-2">
            <button onClick={exportSeries} className="w-full flex items-center gap-2 h-9 border border-slate-300 dark:border-slate-700 px-3 text-[12px] hover:bg-slate-50 dark:hover:bg-slate-800"><Download size={13} /> Time series — {selectedField} ({modelById(model).name})</button>
            <button onClick={exportPairs} className="w-full flex items-center gap-2 h-9 border border-slate-300 dark:border-slate-700 px-3 text-[12px] hover:bg-slate-50 dark:hover:bg-slate-800"><Download size={13} /> Paired sample — {modelById(model).name} vs {indicatorById(indicator).name}</button>
            <button onClick={exportExperiments} className="w-full flex items-center gap-2 h-9 border border-slate-300 dark:border-slate-700 px-3 text-[12px] hover:bg-slate-50 dark:hover:bg-slate-800"><Download size={13} /> Experiment registry (CSV)</button>
            <button onClick={() => window.print()} className="w-full flex items-center gap-2 h-9 border border-slate-300 dark:border-slate-700 px-3 text-[12px] hover:bg-slate-50 dark:hover:bg-slate-800"><ImageIcon size={13} /> Print / save figure (PDF)</button>
            <div className="border border-slate-200 dark:border-slate-800 p-2 text-[11px] text-slate-500 leading-relaxed">Exports include only demonstration data. Replace the data layer with API calls before using for thesis results. Always cite methods, n, and date range alongside exported figures.</div>
          </div>
        </Panel>

        <Panel title="Reproducibility checklist" className="col-span-12 lg:col-span-4">
          <ul className="text-[12px] text-slate-600 dark:text-slate-400 space-y-2 list-disc pl-4 leading-relaxed">
            <li>Record model version, threshold, and normalisation.</li>
            <li>Report n, fields, and date range for every statistic.</li>
            <li>Distinguish association from diagnostic accuracy in text.</li>
            <li>Archive the paired CSV with each figure.</li>
            <li>Note missing-data handling (here: excluded pairwise).</li>
          </ul>
        </Panel>
      </div>

      <Panel title="Saved experiments" subtitle={`${experiments.length} saved · click to restore configuration`} pad={false}>
        {experiments.length === 0 ? <div className="p-4"><EmptyState title="No experiments yet" desc="Create one from the current analysis state above." /></div> : (
          <div className="overflow-auto">
            <table className="w-full text-[12px]">
              <thead className="bg-slate-50 dark:bg-slate-800"><tr><th className="text-left px-4 py-2 font-medium">Name</th><th className="text-left px-2 py-2 font-medium">Created</th><th className="text-left px-2 py-2 font-medium">Model / Indicator</th><th className="text-left px-2 py-2 font-medium">Fields</th><th className="text-left px-2 py-2 font-medium">Range</th><th className="text-left px-2 py-2 font-medium">Stats</th><th className="text-left px-2 py-2 font-medium">Note</th><th className="px-2 py-2" /></tr></thead>
              <tbody>{experiments.map(e => (
                <tr key={e.id} className="border-t border-slate-100 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-800/50">
                  <td className="px-4 py-2.5 font-medium"><span className="inline-flex items-center gap-1.5"><FileText size={12} className="text-slate-400" />{e.name}</span></td>
                  <td className="px-2 py-2.5 text-slate-500">{e.created}</td>
                  <td className="px-2 py-2.5"><Badge color={modelById(e.model).color}>{modelById(e.model).short}</Badge> <Badge color={indicatorById(e.indicator).color}>{indicatorById(e.indicator).name}</Badge></td>
                  <td className="px-2 py-2.5">{e.fields.length}</td>
                  <td className="px-2 py-2.5">{DATES[e.range[0]].slice(0, 7)}→{DATES[e.range[1]].slice(0, 7)}</td>
                  <td className="px-2 py-2.5 font-mono text-[11px]">{e.stats}</td>
                  <td className="px-2 py-2.5 text-slate-500 max-w-48 truncate">{e.note}</td>
                  <td className="px-2 py-2.5"><button onClick={() => removeExperiment(e.id)} className="text-slate-400 hover:text-red-600"><Trash2 size={13} /></button></td>
                </tr>))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>

      <div className="flex gap-2">
        <button onClick={() => setView('validation')} className="h-8 px-3 border border-slate-300 dark:border-slate-700 text-[12px]">Go to Validation Lab</button>
        <button onClick={() => setView('comparison')} className="h-8 px-3 border border-slate-300 dark:border-slate-700 text-[12px]">Go to Model Comparison</button>
      </div>
    </div>
  );
}

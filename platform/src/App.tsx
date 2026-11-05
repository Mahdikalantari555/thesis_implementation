import { AnalysisProvider, useAnalysis } from './context/AnalysisContext';
import Overview from './views/Overview';
import MapExplorer from './views/MapExplorer';
import Temporal from './views/Temporal';
import Spectral from './views/Spectral';
import Validation from './views/Validation';
import ModelComparison from './views/ModelComparison';
import Spatial from './views/Spatial';
import Reports from './views/Reports';
import { LayoutDashboard, Map as MapIcon, Activity, ScatterChart, ShieldCheck, GitCompareArrows, Globe2, FileText, Sun, Moon, Sprout, X, FlaskConical } from 'lucide-react';
import { cn } from './utils/cn';
import { MODELS, INDICATORS, DATES } from './data/mock';

const NAV = [
  { id: 'overview', label: 'Overview', icon: LayoutDashboard, desc: 'Dataset & model summary' },
  { id: 'map', label: 'Geospatial Explorer', icon: MapIcon, desc: 'Interactive field map' },
  { id: 'temporal', label: 'Temporal Analysis', icon: Activity, desc: 'Multi-axis time series' },
  { id: 'spectral', label: 'Spectral–Temporal', icon: ScatterChart, desc: 'Correlations & lags' },
  { id: 'validation', label: 'Validation Lab', icon: ShieldCheck, desc: 'Independent validation' },
  { id: 'comparison', label: 'Model Comparison', icon: GitCompareArrows, desc: 'Compare detectors' },
  { id: 'spatial', label: 'Spatial Analysis', icon: Globe2, desc: 'Clustering & Moran’s I' },
  { id: 'reports', label: 'Reports & Experiments', icon: FileText, desc: 'Save & export' },
];

function Shell() {
  const { view, setView, dark, setDark, model, setModel, indicator, setIndicator, selectedField, range, showDemoBanner, setShowDemoBanner } = useAnalysis();
  const active = NAV.find(n => n.id === view)!;

  return (
    <div className={cn('min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100', dark && 'dark')}>
      <div className="flex min-h-screen">
        {/* Sidebar */}
        <aside className="hidden md:flex flex-col w-60 shrink-0 border-r border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 sticky top-0 h-screen">
          <div className="px-4 py-4 border-b border-slate-200 dark:border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 bg-teal-700 flex items-center justify-center text-white"><Sprout size={18} /></div>
              <div>
                <div className="text-[13px] font-bold tracking-tight leading-tight">AgriAnomaly Lab</div>
                <div className="text-[10px] text-slate-500 leading-tight">Sugarcane · Satellite Anomaly<br />Detection Research</div>
              </div>
            </div>
          </div>
          <nav className="flex-1 overflow-auto py-2">
            {NAV.map(n => (
              <button key={n.id} onClick={() => setView(n.id)} className={cn('w-full flex items-center gap-2.5 px-4 py-2.5 text-left text-[12px] border-l-2', view === n.id ? 'border-teal-700 bg-teal-50 dark:bg-teal-950/40 font-semibold text-teal-900 dark:text-teal-100' : 'border-transparent text-slate-600 dark:text-slate-400 hover:bg-slate-50 dark:hover:bg-slate-800/60')}>
                <n.icon size={15} className="shrink-0" />
                <span><span className="block leading-tight">{n.label}</span><span className="block text-[10px] font-normal text-slate-400 leading-tight">{n.desc}</span></span>
              </button>
            ))}
          </nav>
          <div className="p-3 border-t border-slate-200 dark:border-slate-800 text-[10px] text-slate-500 leading-relaxed">
            <div className="flex items-center gap-1.5 font-medium text-slate-700 dark:text-slate-300"><FlaskConical size={11} /> PhD Research Platform</div>
            <div className="mt-1">Khuzestan Sugarcane Agro-Industry<br />Landsat & METRIC (2013–2026)</div>
          </div>
        </aside>

        {/* Main */}
        <div className="flex-1 min-w-0 flex flex-col">
          {/* Topbar */}
          <header className="sticky top-0 z-40 bg-white/95 dark:bg-slate-900/95 backdrop-blur border-b border-slate-200 dark:border-slate-800">
            <div className="flex items-center gap-3 px-4 h-12">
              <div className="md:hidden flex items-center gap-2"><div className="w-7 h-7 bg-teal-700 flex items-center justify-center text-white"><Sprout size={15} /></div><span className="text-[12px] font-bold">AgriAnomaly Lab</span></div>
              <div className="hidden md:block">
                <div className="text-[13px] font-semibold leading-tight">{active.label}</div>
                <div className="text-[10px] text-slate-500 leading-tight">{active.desc}</div>
              </div>
              <div className="flex-1" />
              {/* Shared state controls */}
              <div className="hidden lg:flex items-center gap-2 text-[11px]">
                <select value={model} onChange={e => setModel(e.target.value as any)} className="h-7 border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-1.5 text-[11px]">{MODELS.map(m => <option key={m.id} value={m.id}>{m.name}</option>)}</select>
                <select value={indicator} onChange={e => setIndicator(e.target.value as any)} className="h-7 border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-1.5 text-[11px]">{INDICATORS.map(i => <option key={i.id} value={i.id}>{i.name}</option>)}</select>
                <span className="px-2 h-7 flex items-center border border-slate-200 dark:border-slate-700 text-slate-500">{selectedField} · {DATES[range[0]].slice(0, 7)}→{DATES[range[1]].slice(0, 7)}</span>
              </div>
              <button onClick={() => setDark(!dark)} className="w-8 h-8 flex items-center justify-center border border-slate-300 dark:border-slate-700">{dark ? <Sun size={14} /> : <Moon size={14} />}</button>
            </div>
            {/* Mobile nav */}
            <div className="md:hidden flex overflow-x-auto gap-1 px-2 pb-2">
              {NAV.map(n => <button key={n.id} onClick={() => setView(n.id)} className={cn('shrink-0 px-2.5 py-1.5 text-[11px] border', view === n.id ? 'bg-teal-700 text-white border-teal-700' : 'border-slate-300 dark:border-slate-700')}>{n.label}</button>)}
            </div>
          </header>

          {showDemoBanner && (
            <div className="bg-teal-50 dark:bg-teal-950/40 border-b border-teal-200 dark:border-teal-900 px-4 py-2 flex items-start gap-2 text-[11px] text-teal-900 dark:text-teal-200">
              <FlaskConical size={13} className="shrink-0 mt-0.5" />
              <span className="flex-1 leading-relaxed"><b>Khuzestan Sugarcane Agro-Industry dataset integrated.</b> 490 field polygons, Landsat/METRIC multi-spectral indices (NDVI, CWSI, ETa, LST), and literature SSL model anomaly scores (B0 DOY, B1 DSH, B1 CUSUM, SSL Transformer, STS-Net, Tiny-CNN-Mamba, Time2Agri).</span>
              <button onClick={() => setShowDemoBanner(false)} className="shrink-0"><X size={13} /></button>
            </div>
          )}

          <main className="flex-1 p-4 max-w-[1600px] w-full mx-auto">
            {view === 'overview' && <Overview />}
            {view === 'map' && <MapExplorer />}
            {view === 'temporal' && <Temporal />}
            {view === 'spectral' && <Spectral />}
            {view === 'validation' && <Validation />}
            {view === 'comparison' && <ModelComparison />}
            {view === 'spatial' && <Spatial />}
            {view === 'reports' && <Reports />}
          </main>

          <footer className="border-t border-slate-200 dark:border-slate-800 px-4 py-3 text-[10px] text-slate-500 flex flex-wrap gap-x-4 gap-y-1">
            <span>AgriAnomaly Lab — PhD Research on Self-Supervised Anomaly Detection in Sugarcane</span>
            <span>Khuzestan Agro-Industry · 490 Fields · Landsat Level-2 & METRIC (2013–2026)</span>
            <span>STS-Net · Tiny-CNN-Mamba · Time2Agri · B1-DSH Baseline</span>
          </footer>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  return <AnalysisProvider><Shell /></AnalysisProvider>;
}

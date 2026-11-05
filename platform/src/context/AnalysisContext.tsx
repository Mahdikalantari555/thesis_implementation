import React, { createContext, useContext, useMemo, useState } from 'react';
import type { IndicatorId, ModelId } from '../data/mock';
import { FIELDS } from '../data/mock';

export interface Experiment {
  id: string; name: string; created: string; model: ModelId; indicator: IndicatorId;
  fields: string[]; range: [number, number]; note: string; stats: string;
}

interface Ctx {
  view: string; setView: (v: string) => void;
  dark: boolean; setDark: (d: boolean) => void;
  selectedField: string; setSelectedField: (id: string) => void;
  selectedFields: string[]; setSelectedFields: (ids: string[]) => void;
  toggleField: (id: string) => void;
  model: ModelId; setModel: (m: ModelId) => void;
  compareModels: ModelId[]; setCompareModels: (m: ModelId[]) => void;
  indicator: IndicatorId; setIndicator: (i: IndicatorId) => void;
  range: [number, number]; setRange: (r: [number, number]) => void;
  experiments: Experiment[]; addExperiment: (e: Experiment) => void; removeExperiment: (id: string) => void;
  showDemoBanner: boolean; setShowDemoBanner: (b: boolean) => void;
}

const AnalysisCtx = createContext<Ctx>(null!);
export const useAnalysis = () => useContext(AnalysisCtx);

export function AnalysisProvider({ children }: { children: React.ReactNode }) {
  const [view, setView] = useState('overview');
  const [dark, setDark] = useState(false);
  const [selectedField, setSelectedField] = useState(FIELDS[9].id); // F-010
  const [selectedFields, setSelectedFields] = useState<string[]>([FIELDS[9].id, FIELDS[10].id, FIELDS[18].id]);
  const [model, setModel] = useState<ModelId>('b1_dsh');
  const [compareModels, setCompareModels] = useState<ModelId[]>([
    'b1_dsh',
    'b1_cusum',
    'masked_transformer',
    'sts_net',
    'tiny_cnn_mamba',
    'time2agri',
  ]);
  const [indicator, setIndicator] = useState<IndicatorId>('ndvi');
  const [range, setRange] = useState<[number, number]>([0, 47]);
  const [experiments, setExperiments] = useState<Experiment[]>([
    { id: 'exp-1', name: 'B1 DSH vs CWSI — Khuzestan R01–R04, 2022–2023', created: '2026-04-18', model: 'b1_dsh', indicator: 'cwsi', fields: [FIELDS[0]?.id || 'F-001', FIELDS[1]?.id || 'F-002', FIELDS[2]?.id || 'F-003'], range: [12, 35], note: 'Water-salinity anomaly verification; check lag-1 response.', stats: 'r = 0.58 · ρ = 0.54 · n = 72' },
    { id: 'exp-2', name: 'Model agreement — Tiny-CNN-Mamba vs STS-Net', created: '2026-05-02', model: 'tiny_cnn_mamba', indicator: 'cwsi', fields: FIELDS.slice(0, 12).map(f => f.id), range: [0, 47], note: 'Compare Bidirectional Mamba vs Spatio-Temporal CAE event overlap.', stats: 'κ = 0.68 · Jaccard = 0.64' },
  ]);
  const [showDemoBanner, setShowDemoBanner] = useState(true);

  const toggleField = (id: string) => {
    setSelectedField(id);
    setSelectedFields(prev => prev.includes(id) ? prev : [...prev, id].slice(-8));
  };

  const value = useMemo<Ctx>(() => ({
    view, setView, dark, setDark, selectedField, setSelectedField, selectedFields, setSelectedFields, toggleField,
    model, setModel, compareModels, setCompareModels, indicator, setIndicator, range, setRange,
    experiments, addExperiment: (e) => setExperiments(p => [e, ...p]), removeExperiment: (id) => setExperiments(p => p.filter(x => x.id !== id)),
    showDemoBanner, setShowDemoBanner,
  }), [view, dark, selectedField, selectedFields, model, compareModels, indicator, range, experiments, showDemoBanner]);

  return <AnalysisCtx.Provider value={value}>{children}</AnalysisCtx.Provider>;
}

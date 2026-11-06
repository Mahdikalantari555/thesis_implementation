// ── Channel registry (mirror of scripts/platform_channels.py) ─────────────────
// Single source of truth for the data contract between the exporter and the
// platform. Keep this in sync with scripts/platform_channels.py; parity is
// enforced by tests/unit/test_platform_data_contract.py.

export type ProvenanceState = 'observed' | 'modelled' | 'proxy' | 'absent' | 'legacy';

export type ChannelKind = 'spectral' | 'thermal' | 'water' | 'weather';

export interface ChannelSpec {
  name: string;
  unit: string;
  physicalRange: [number, number];
  provenanceDefault: ProvenanceState;
  required: boolean;
  kind: ChannelKind;
  derivedFrom?: string;
}

export const CHANNELS: ChannelSpec[] = [
  { name: 'ndvi', unit: 'index', physicalRange: [0, 1], provenanceDefault: 'observed', required: true, kind: 'spectral' },
  { name: 'evi', unit: 'index', physicalRange: [0, 1], provenanceDefault: 'observed', required: true, kind: 'spectral' },
  { name: 'savi', unit: 'index', physicalRange: [0, 1], provenanceDefault: 'observed', required: true, kind: 'spectral' },
  { name: 'ndmi', unit: 'index', physicalRange: [-1, 1], provenanceDefault: 'observed', required: true, kind: 'water' },
  { name: 'tvdi', unit: 'index', physicalRange: [0, 1], provenanceDefault: 'proxy', required: true, kind: 'thermal', derivedFrom: 'cwsi' },
  { name: 'cwsi', unit: 'index', physicalRange: [0, 1], provenanceDefault: 'observed', required: true, kind: 'thermal' },
  { name: 'eta', unit: 'mm/d', physicalRange: [0, 8], provenanceDefault: 'proxy', required: true, kind: 'water', derivedFrom: 'ndvi' },
  { name: 'lst', unit: '°C', physicalRange: [15, 55], provenanceDefault: 'observed', required: false, kind: 'thermal' },
  { name: 'precip', unit: 'mm', physicalRange: [0, 300], provenanceDefault: 'modelled', required: true, kind: 'weather' },
  { name: 'temp', unit: '°C', physicalRange: [10, 48], provenanceDefault: 'modelled', required: true, kind: 'weather' },
];

export const channelByName = (name: string): ChannelSpec | undefined =>
  CHANNELS.find(c => c.name === name);

export const CHANNEL_NAMES = CHANNELS.map(c => c.name);

export type ChannelId = (typeof CHANNELS)[number]['name'];

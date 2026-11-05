import React from 'react';
import { cn } from '../utils/cn';

export function Panel({ title, subtitle, right, children, className, pad = true }: { title?: string; subtitle?: string; right?: React.ReactNode; children: React.ReactNode; className?: string; pad?: boolean }) {
  return (
    <section className={cn('border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900', className)}>
      {(title || right) && (
        <header className="flex items-start justify-between gap-3 border-b border-slate-200 dark:border-slate-800 px-4 py-3">
          <div>
            {title && <h3 className="text-[13px] font-semibold tracking-tight text-slate-900 dark:text-slate-100">{title}</h3>}
            {subtitle && <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-snug">{subtitle}</p>}
          </div>
          {right}
        </header>
      )}
      <div className={pad ? 'p-4' : ''}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 px-4 py-3">
      <div className="text-[10px] uppercase tracking-[0.08em] text-slate-500 dark:text-slate-400 font-medium">{label}</div>
      <div className={cn('text-xl font-semibold tracking-tight mt-1 text-slate-900 dark:text-slate-100', tone)}>{value}</div>
      {sub && <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{sub}</div>}
    </div>
  );
}

export function Badge({ children, color }: { children: React.ReactNode; color?: string }) {
  return <span className="inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 border" style={{ color: color || '#475569', borderColor: color ? color + '55' : '#cbd5e1', background: color ? color + '12' : '#f8fafc' }}>{children}</span>;
}

export function Select({ value, onChange, children, className }: { value: string; onChange: (v: string) => void; children: React.ReactNode; className?: string }) {
  return (
    <select value={value} onChange={e => onChange(e.target.value)} className={cn('h-8 border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12px] px-2 text-slate-800 dark:text-slate-200 outline-none focus:border-teal-600', className)}>
      {children}
    </select>
  );
}

export function Label({ children }: { children: React.ReactNode }) {
  return <label className="block text-[10px] uppercase tracking-[0.08em] text-slate-500 dark:text-slate-400 font-medium mb-1">{children}</label>;
}

export function EmptyState({ title, desc }: { title: string; desc: string }) {
  return (
    <div className="border border-dashed border-slate-300 dark:border-slate-700 p-8 text-center">
      <div className="text-[13px] font-medium text-slate-700 dark:text-slate-300">{title}</div>
      <div className="text-[12px] text-slate-500 mt-1">{desc}</div>
    </div>
  );
}

export function SigBadge({ p }: { p: number }) {
  if (isNaN(p)) return <Badge>—</Badge>;
  const label = p < 0.001 ? 'p < 0.001 ***' : p < 0.01 ? 'p < 0.01 **' : p < 0.05 ? 'p < 0.05 *' : 'n.s.';
  const color = p < 0.05 ? '#0f766e' : '#64748b';
  return <Badge color={color}>{label}</Badge>;
}

export const fmt = (v: number, d = 3) => isNaN(v) ? '—' : v.toFixed(d);
export const fmt2 = (v: number) => isNaN(v) ? '—' : v.toFixed(2);

export function anomalyColor(v: number): string {
  if (v >= 0.75) return '#be123c';
  if (v >= 0.65) return '#ea580c';
  if (v >= 0.5) return '#d97706';
  if (v >= 0.35) return '#65a30d';
  return '#0f766e';
}

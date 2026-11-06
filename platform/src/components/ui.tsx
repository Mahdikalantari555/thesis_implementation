import React from "react";
import { FlaskConical } from "lucide-react";
import { cn } from "../utils/cn";
import { useT } from "../i18n";

export function Panel({ title, subtitle, right, children, className, pad = true }: { title?: string; subtitle?: string; right?: React.ReactNode; children: React.ReactNode; className?: string; pad?: boolean }) {
  return (
    <section className={cn("border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900", className)}>
      {(title || right) && (
        <header className="flex items-start justify-between gap-3 border-b border-slate-200 dark:border-slate-800 px-4 py-3">
          <div className="min-w-0">
            {title && <h3 className="text-[13px] font-semibold tracking-tight text-slate-900 dark:text-slate-100">{title}</h3>}
            {subtitle && <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5 leading-snug">{subtitle}</p>}
          </div>
          {right && <div className="shrink-0">{right}</div>}
        </header>
      )}
      <div className={pad ? "p-4" : ""}>{children}</div>
    </section>
  );
}

export function Stat({ label, value, sub, tone }: { label: string; value: string; sub?: string; tone?: string }) {
  return (
    <div className="border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 px-4 py-3">
      <div className="text-[10px] uppercase tracking-[0.08em] text-slate-500 dark:text-slate-400 font-medium">{label}</div>
      {/* <bdi> isolates the value so Persian prose around it cannot reverse a
          leading minus sign into "0.547-" under dir="rtl". */}
      <div className={cn("text-xl font-semibold tracking-tight mt-1 text-slate-900 dark:text-slate-100", tone)}>
        <bdi>{value}</bdi>
      </div>
      {sub && <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">{sub}</div>}
    </div>
  );
}

export function Badge({ children, color, title }: { children: React.ReactNode; color?: string; title?: string }) {
  return <span title={title} className="inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 border" style={{ color: color || "#475569", borderColor: color ? color + "55" : "#cbd5e1", background: color ? color + "12" : "#f8fafc" }}>{children}</span>;
}

/**
 * Data-provenance badge. The bug audit requires that every view rendering the
 * demonstration series carries an explicit marker (spec: "Demo badge present"),
 * so this component is the single visual source of that claim.
 */
export function DemoDataBadge({ className }: { className?: string }) {
  const t = useT();
  return (
    <span
      title={t("ui.demoBadgeTitle")}
      className={cn(
        "inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 border border-amber-500/40 text-amber-700 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/40",
        className
      )}
    >
      <FlaskConical size={10} aria-hidden="true" />
      {t("ui.demoBadge")}
    </span>
  );
}

export function Select({ value, onChange, children, className, ariaLabel }: { value: string; onChange: (v: string) => void; children: React.ReactNode; className?: string; ariaLabel?: string }) {
  return (
    <select
      value={value}
      onChange={e => onChange(e.target.value)}
      aria-label={ariaLabel}
      className={cn("h-8 border border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 text-[12px] px-2 text-slate-800 dark:text-slate-200 outline-none focus:border-teal-600", className)}
    >
      {children}
    </select>
  );
}

export function Label({ children }: { children: React.ReactNode }) {
  return <label className="block text-[10px] uppercase tracking-[0.08em] text-slate-500 dark:text-slate-400 font-medium mb-1">{children}</label>;
}

export function EmptyState({ title, desc }: { title?: string; desc?: string }) {
  const t = useT();
  return (
    <div className="border border-dashed border-slate-300 dark:border-slate-700 p-8 text-center">
      <div className="text-[13px] font-medium text-slate-700 dark:text-slate-300">{title ?? t("ui.noData")}</div>
      <div className="text-[12px] text-slate-500 mt-1">{desc ?? t("ui.noDataDesc")}</div>
    </div>
  );
}

export function SigBadge({ p }: { p: number }) {
  const t = useT();
  if (isNaN(p)) return <Badge>—</Badge>;
  const label = p < 0.001 ? t("ui.pStrong") : p < 0.01 ? t("ui.pMid") : p < 0.05 ? t("ui.pWeak") : t("ui.pNs");
  const color = p < 0.05 ? "#0f766e" : "#64748b";
  return <Badge color={color}>{label}</Badge>;
}

export const fmt = (v: number | null | undefined, d = 3) => (v == null || isNaN(v) ? "—" : v.toFixed(d));
export const fmt2 = (v: number | null | undefined) => (v == null || isNaN(v) ? "—" : v.toFixed(2));

export function anomalyColor(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "#94a3b8";
  if (v >= 0.75) return "#be123c";
  if (v >= 0.65) return "#ea580c";
  if (v >= 0.5) return "#d97706";
  if (v >= 0.35) return "#65a30d";
  return "#0f766e";
}

export function ProvenanceChip({ state, title }: { state: 'observed' | 'modelled' | 'proxy' | 'absent' | 'legacy'; title?: string }): React.ReactElement {
  const t = useT();
  const labels: Record<'observed' | 'modelled' | 'proxy' | 'absent' | 'legacy', string> = {
    observed: t("prov.observed"),
    modelled: t("prov.modelled"),
    proxy: t("prov.proxy"),
    absent: t("prov.absent"),
    legacy: t("prov.legacy"),
  };
  const colors: Record<'observed' | 'modelled' | 'proxy' | 'absent' | 'legacy', string> = {
    observed: "#0f766e",
    modelled: "#0284c7",
    proxy: "#d97706",
    absent: "#94a3b8",
    legacy: "#be123c",
  };
  const color = colors[state];
  return (
    <span
      title={title ?? labels[state]}
      className="inline-flex items-center gap-1 text-[10px] font-medium px-1.5 py-0.5 border"
      style={{ color: color, borderColor: color + "55", background: color + "12" }}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: color }} aria-hidden="true" />
      {labels[state]}
    </span>
  );
}

export const fmtObs = (v: number | null, d = 3): string => {
  const n = v == null ? NaN : v;
  return isNaN(n) ? "—" : n.toFixed(d);
};

export function LegacyPayloadNotice(): React.ReactElement {
  const t = useT();
  return (
    <div className="mt-2 p-2 border border-amber-400 bg-amber-50 dark:bg-amber-950/30 text-amber-800 dark:text-amber-200 text-[11px] rounded">
      <bdi>{t("legacy.notice.title") || "Legacy payload"}</bdi>: {t("legacy.notice.body") || "This payload was written before contractVersion 2. Some channels are labelled legacy/absent and may not match the current UI."}
    </div>
  );
}
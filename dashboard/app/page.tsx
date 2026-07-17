'use client';

import React, { useEffect } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import StatCard from '@/components/data-display/StatCard';
import GaugeChart from '@/components/data-display/GaugeChart';
import GlassPanel from '@/components/common/GlassPanel';
import KaTeXBlock from '@/components/common/KaTeXBlock';
import { useTranslations } from 'next-intl';
import { ArrowUpRight } from 'lucide-react';

import task2Enforcement from '@/lib/results/task2_enforcement.json';
import task1Prediction from '@/lib/results/task1_prediction.json';

export default function Home() {
  const { setActivePage } = useGlobalStore();
  const t = useTranslations();

  useEffect(() => {
    setActivePage('command_center');
  }, [setActivePage]);

  const tgEnforcement = task2Enforcement.results['TrustGuard (ours)'];
  const tgPrediction = task1Prediction.results['TrustGuard (ours)'];

  // Best baseline (exclude ours + install-time-only null rows)
  const baselineEntries = Object.entries(task2Enforcement.results).filter(
    ([name, d]) => !name.includes('TrustGuard') && d.AIPR_pct !== null
  );
  const bestAipr = Math.max(...baselineEntries.map(([, d]) => d.AIPR_pct as number));
  const bestEpr = Math.max(...baselineEntries.map(([, d]) => d.EPR_pct as number));
  const bestLatency = Math.min(...baselineEntries.map(([, d]) => d.latency_s as number));

  const rows = Object.entries(task2Enforcement.results).map(([name, d]) => ({
    name,
    aipr: d.AIPR_pct as number | null,
    epr: d.EPR_pct as number | null,
    ours: name.includes('TrustGuard'),
  }));

  return (
    <div className="flex flex-col gap-6 animate-fade-in-up">
      {/* ── 1. Hero metrics ───────────────────────────────────────────── */}
      <section className="flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="eyebrow">Headline enforcement metrics</span>
          <span className="text-xs text-tertiary text-mono">mean ± std · 5 seeds · 72h AASE</span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
          <StatCard
            labelKey="metrics.aipr_short"
            value={tgEnforcement.AIPR_pct ?? 0}
            std={tgEnforcement.AIPR_std ?? undefined}
            suffix="%"
            accentColor="safe"
            tooltipKey="metrics.aipr"
            hint={`+${((tgEnforcement.AIPR_pct ?? 0) - bestAipr).toFixed(1)} vs best baseline`}
            hintTone="up"
          />
          <StatCard
            labelKey="metrics.epr_short"
            value={tgEnforcement.EPR_pct ?? 0}
            std={tgEnforcement.EPR_std ?? undefined}
            suffix="%"
            accentColor="safe"
            tooltipKey="metrics.epr"
            hint={`+${((tgEnforcement.EPR_pct ?? 0) - bestEpr).toFixed(1)} vs best baseline`}
            hintTone="up"
          />
          <StatCard
            labelKey="metrics.frr_short"
            value={tgEnforcement.FRR_pct ?? 0}
            std={tgEnforcement.FRR_std ?? undefined}
            suffix="%"
            accentColor="constraint"
            tooltipKey="metrics.frr"
            hint="within 2.5% budget"
            hintTone="up"
          />
          <StatCard
            labelKey="metrics.macro_f1_short"
            value={tgPrediction.macro_f1 ?? 0}
            std={tgPrediction.macro_f1_std ?? undefined}
            decimals={3}
            accentColor="monitor"
            tooltipKey="metrics.macro_f1"
            hint="Task-1 prediction"
            hintTone="neutral"
          />
          <StatCard
            labelKey="metrics.latency_short"
            value={tgEnforcement.latency_s ?? 0}
            suffix="s"
            decimals={1}
            accentColor="enforce"
            tooltipKey="metrics.latency"
            hint={`−${(bestLatency - (tgEnforcement.latency_s ?? 0)).toFixed(1)}s median vs MAPPO`}
            hintTone="up"
          />
        </div>
      </section>

      {/* ── 2. Gauge + overview ───────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <GaugeChart
            value={(tgEnforcement.FRR_pct ?? 0) / 100}
            threshold={task2Enforcement.protocol.eps_safe}
            max={0.05}
            titleKey="False Revocation Rate"
            decimals={2}
          />
        </div>

        <GlassPanel accentTop monitor className="lg:col-span-2 p-6 flex flex-col justify-between gap-5">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <span className="badge badge--monitor">Framework</span>
              <span className="badge badge--constraint">Constrained MAPPO</span>
            </div>
            <h2 className="text-lg font-bold mb-3">Autonomous permission governance as a Dec-POMDP</h2>
            <p className="text-secondary text-sm leading-relaxed mb-3">
              Mobile permission systems rely on static policies and uninformed prompts that cannot reason about
              runtime behaviour. TrustGuard replaces this with a continuous, learning-based governance loop —
              three cooperative agents (<span className="text-monitor font-semibold">Monitoring</span>,{' '}
              <span className="text-risk font-semibold">Risk-Analysis</span>,{' '}
              <span className="text-enforce font-semibold">Enforcement</span>) trained via CTDE with a Lagrangian
              safety constraint that bounds the false-revocation rate.
            </p>
          </div>

          <div className="flex flex-col gap-2 p-4 bg-surface rounded-xl border border-subtle">
            <span className="eyebrow">Optimization objective</span>
            <div className="flex items-center justify-center py-1 overflow-x-auto">
              <KaTeXBlock
                math="\mathcal{L}(\theta, \mu) = \mathbb{E}_{\pi}\left[\sum_t \gamma^t r_t\right] - \mu\left(\text{FRR}(\pi) - \epsilon_{\text{safe}}\right)"
                block
              />
            </div>
          </div>
        </GlassPanel>
      </div>

      {/* ── 3. Method comparison ──────────────────────────────────────── */}
      <GlassPanel accentTop enforce className="p-6">
        <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
          <div>
            <h3 className="text-base font-bold">Enforcement quality across methods</h3>
            <p className="text-xs text-secondary mt-0.5">Ground-truth AIPR and taint-verified EPR (higher is better)</p>
          </div>
          <div className="flex items-center gap-4 text-xs">
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: 'linear-gradient(90deg,#38BDF8,#6366F1)' }} />
              <span className="text-secondary">AIPR</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: 'linear-gradient(90deg,#34D399,#14B8A6)' }} />
              <span className="text-secondary">EPR</span>
            </span>
          </div>
        </div>

        <div className="flex flex-col">
          {rows.map((r) => {
            const hasData = r.aipr !== null;
            return (
              <div
                key={r.name}
                className={`grid grid-cols-1 md:grid-cols-[13rem_1fr] gap-2 md:gap-5 items-center py-3.5 border-b border-subtle last:border-0 ${
                  r.ours ? 'bg-monitor/[0.04] -mx-2 px-2 rounded-lg' : ''
                }`}
              >
                <div className={`flex items-center gap-2 text-sm font-semibold ${r.ours ? 'text-monitor' : 'text-primary'}`}>
                  <span className="truncate">{r.name}</span>
                  {r.ours && <span className="chip">Ours</span>}
                </div>

                {hasData ? (
                  <div className="flex flex-col gap-2">
                    <MetricBar label="AIPR" value={r.aipr as number} ours={r.ours} tone="aipr" />
                    <MetricBar label="EPR" value={r.epr as number} ours={r.ours} tone="epr" />
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-xs text-tertiary italic">
                    <ArrowUpRight size={13} />
                    Install-time classifier — cannot enforce at runtime (N/A)
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </GlassPanel>
    </div>
  );
}

function MetricBar({ label, value, ours, tone }: { label: string; value: number; ours: boolean; tone: 'aipr' | 'epr' }) {
  const gradient =
    tone === 'aipr'
      ? ours ? 'linear-gradient(90deg,#38BDF8,#6366F1)' : 'var(--text-tertiary)'
      : ours ? 'linear-gradient(90deg,#34D399,#14B8A6)' : 'var(--border-strong)';
  return (
    <div className="flex items-center gap-3">
      <span className="text-[10px] w-8 text-tertiary text-end text-mono flex-shrink-0">{label}</span>
      <div className="flex-1 h-2.5 bg-surface rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-1000 ease-out"
          style={{ width: `${value}%`, background: gradient }}
        />
      </div>
      <span className={`text-xs text-mono font-semibold w-12 flex-shrink-0 ${ours ? 'text-primary' : 'text-secondary'}`}>
        {value.toFixed(1)}%
      </span>
    </div>
  );
}

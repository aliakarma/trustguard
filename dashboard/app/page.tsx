'use client';

import React, { useEffect } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import StatCard from '@/components/data-display/StatCard';
import GaugeChart from '@/components/data-display/GaugeChart';
import GlassPanel from '@/components/common/GlassPanel';
import KaTeXBlock from '@/components/common/KaTeXBlock';
import { useTranslations } from 'next-intl';

// Import local JSON results directly
import task2Enforcement from '@/lib/results/task2_enforcement.json';
import task1Prediction from '@/lib/results/task1_prediction.json';

export default function Home() {
  const { setActivePage } = useGlobalStore();
  const t = useTranslations();

  useEffect(() => {
    setActivePage('command_center');
  }, [setActivePage]);

  // Extract TrustGuard specific metrics from results files
  const tgEnforcement = task2Enforcement.results['TrustGuard (ours)'];
  const tgPrediction = task1Prediction.results['TrustGuard (ours)'];

  // Parse baseline comparators
  const baselines = Object.entries(task2Enforcement.results)
    .filter(([name]) => name !== 'Android Static Policy')
    .map(([name, data]) => ({
      name,
      aipr: data.AIPR_pct,
      epr: data.EPR_pct,
      frr: data.FRR_pct,
    }));

  return (
    <div className="flex flex-col gap-6 animate-fade-in-up">
      {/* ── 1. Hero Metrics Row ───────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
        <StatCard
          labelKey="metrics.aipr_short"
          value={tgEnforcement.AIPR_pct ?? 0}
          std={tgEnforcement.AIPR_std ?? undefined}
          suffix="%"
          accentColor="safe"
          tooltipKey="metrics.aipr"
        />
        <StatCard
          labelKey="metrics.epr_short"
          value={tgEnforcement.EPR_pct ?? 0}
          std={tgEnforcement.EPR_std ?? undefined}
          suffix="%"
          accentColor="safe"
          tooltipKey="metrics.epr"
        />
        <StatCard
          labelKey="metrics.frr_short"
          value={tgEnforcement.FRR_pct ?? 0}
          std={tgEnforcement.FRR_std ?? undefined}
          suffix="%"
          accentColor="risk"
          tooltipKey="metrics.frr"
        />
        <StatCard
          labelKey="metrics.macro_f1_short"
          value={tgPrediction.macro_f1 ?? 0}
          std={tgPrediction.macro_f1_std ?? undefined}
          decimals={3}
          accentColor="monitor"
          tooltipKey="metrics.macro_f1"
        />
        <StatCard
          labelKey="metrics.latency_short"
          value={tgEnforcement.latency_s ?? 0}
          suffix="s"
          decimals={1}
          accentColor="enforce"
          tooltipKey="metrics.latency"
        />
      </div>

      {/* ── 2. Gauge & Summary Panels ─────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* FRR Safety Gauge */}
        <div className="lg:col-span-1">
          <GaugeChart
            value={(tgEnforcement.FRR_pct ?? 0) / 100}
            threshold={task2Enforcement.protocol.eps_safe}
            max={0.05}
            titleKey={t('terms.constraint')}
            decimals={2}
          />
        </div>

        {/* Paper Summary Card */}
        <GlassPanel accentTop monitor className="lg:col-span-2 p-6 flex flex-col justify-between">
          <div>
            <h2 className="text-xl font-bold mb-4 flex items-center gap-2">
              <span>🛡️</span> {t('header.title')}
            </h2>
            <p className="text-secondary text-sm mb-4">
              Mobile permission systems rely on static policies and uninformed user prompts that cannot reason about application behaviour at runtime. TrustGuard replaces this with a continuous, learning-based governance loop formalised as a <strong>Decentralised Partially Observable Markov Decision Process (Dec-POMDP)</strong>.
            </p>
            <p className="text-secondary text-sm mb-6">
              Three cooperative agents — <strong>Monitoring</strong>, <strong>Risk-Analysis</strong>, and <strong>Enforcement</strong> — are trained via Centralised Training / Decentralised Execution (CTDE) using MAPPO with a <strong>Lagrangian safety constraint</strong> that bounds the false-revocation rate.
            </p>
          </div>

          <div className="flex flex-col gap-2 p-4 bg-secondary rounded-xl border border-subtle">
            <span className="text-xs font-bold text-tertiary uppercase tracking-wide">Optimization Objective:</span>
            <div className="flex items-center justify-center py-2 overflow-x-auto">
              <KaTeXBlock math="\mathcal{L}(\theta, \mu) = \mathbb{E}_{\pi}\left[\sum_t \gamma^t r_t\right] - \mu\left(\text{FRR}(\pi) - \epsilon_{\text{safe}}\right)" block />
            </div>
          </div>
        </GlassPanel>
      </div>

      {/* ── 3. Quick Comparison Chart ─────────────────────────────────── */}
      <GlassPanel accentTop enforce className="p-6">
        <h3 className="stat-label mb-6">AIPR and EPR Comparison Across Methods</h3>
        <div className="flex flex-col gap-4">
          {baselines.map((baseline) => {
            const isOurs = baseline.name.includes('TrustGuard');
            const hasAIPR = baseline.aipr !== null;
            return (
              <div key={baseline.name} className="flex flex-col md:flex-row md:items-center gap-2 md:gap-6 border-b border-subtle pb-4 last:border-0 last:pb-0">
                {/* Method Name */}
                <div className={`md:w-48 text-sm font-semibold ${isOurs ? 'text-monitor' : 'text-primary'}`}>
                  {baseline.name}
                  {isOurs && <span className="text-[10px] bg-sky-500/10 text-sky-400 px-1.5 py-0.5 rounded ml-2">Ours</span>}
                </div>

                {/* Bars Area */}
                <div className="flex-1 flex flex-col gap-2">
                  {/* AIPR Bar */}
                  <div className="flex items-center gap-3">
                    <span className="text-[10px] w-8 text-tertiary text-right">AIPR</span>
                    {hasAIPR ? (
                      <div className="flex-1 h-3 bg-secondary rounded overflow-hidden">
                        <div
                          className={`h-full rounded transition-all duration-1000 ${
                            isOurs ? 'bg-gradient-to-r from-sky-400 to-blue-500' : 'bg-slate-500'
                          }`}
                          style={{ width: `${baseline.aipr}%` }}
                        />
                      </div>
                    ) : (
                      <span className="text-xs text-tertiary italic">N/A (Install-time only)</span>
                    )}
                    {hasAIPR && (
                      <span className="text-xs font-mono font-bold w-12 text-primary">
                        {baseline.aipr}%
                      </span>
                    )}
                  </div>

                  {/* EPR Bar */}
                  {hasAIPR && (
                    <div className="flex items-center gap-3">
                      <span className="text-[10px] w-8 text-tertiary text-right">EPR</span>
                      <div className="flex-1 h-3 bg-secondary rounded overflow-hidden">
                        <div
                          className={`h-full rounded transition-all duration-1000 ${
                            isOurs ? 'bg-gradient-to-r from-emerald-400 to-teal-500' : 'bg-slate-600'
                          }`}
                          style={{ width: `${baseline.epr}%` }}
                        />
                      </div>
                      <span className="text-xs font-mono font-bold w-12 text-primary">
                        {baseline.epr}%
                      </span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </GlassPanel>
    </div>
  );
}

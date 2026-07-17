'use client';

import React, { useEffect, useState } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import { useTranslations } from 'next-intl';

// Import local JSON results directly
import task1Prediction from '@/lib/results/task1_prediction.json';
import task2Enforcement from '@/lib/results/task2_enforcement.json';
import adversarialRobustness from '@/lib/results/adversarial_robustness.json';

type TabType = 'task1' | 'task2' | 'task3';

export default function ResultsExplorer() {
  const { setActivePage } = useGlobalStore();
  const t = useTranslations();
  const [activeTab, setActiveTab] = useState<TabType>('task1');
  const [showStd, setShowStd] = useState(true);

  useEffect(() => {
    setActivePage('results');
  }, [setActivePage]);

  return (
    <div className="flex flex-col gap-6 animate-fade-in-up">
      {/* ── 1. Tab Navigation & Controls ──────────────────────────────── */}
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 bg-panel p-4 rounded-xl border border-subtle">
        <div className="flex gap-2 w-full md:w-auto overflow-x-auto">
          <button
            onClick={() => setActiveTab('task1')}
            className={`btn text-xs font-semibold px-4 py-2.5 rounded-lg transition-all ${
              activeTab === 'task1'
                ? 'bg-monitor text-white'
                : 'bg-secondary hover:bg-slate-700 text-secondary'
            }`}
          >
            {t('results_page.task1_tab')}
          </button>
          <button
            onClick={() => setActiveTab('task2')}
            className={`btn text-xs font-semibold px-4 py-2.5 rounded-lg transition-all ${
              activeTab === 'task2'
                ? 'bg-monitor text-white'
                : 'bg-secondary hover:bg-slate-700 text-secondary'
            }`}
          >
            {t('results_page.task2_tab')}
          </button>
          <button
            onClick={() => setActiveTab('task3')}
            className={`btn text-xs font-semibold px-4 py-2.5 rounded-lg transition-all ${
              activeTab === 'task3'
                ? 'bg-monitor text-white'
                : 'bg-secondary hover:bg-slate-700 text-secondary'
            }`}
          >
            {t('results_page.task3_tab')}
          </button>
        </div>

        {activeTab !== 'task3' && (
          <label className="flex items-center gap-2 text-xs text-secondary cursor-pointer">
            <input
              type="checkbox"
              checked={showStd}
              onChange={() => setShowStd(!showStd)}
              className="rounded border-subtle text-monitor focus:ring-monitor h-3.5 w-3.5"
            />
            <span>{t('results_page.show_std')}</span>
          </label>
        )}
      </div>

      {/* ── 2. Tab Content Panels ─────────────────────────────────────── */}
      {activeTab === 'task1' && (
        <Task1Panel showStd={showStd} t={t} />
      )}
      {activeTab === 'task2' && (
        <Task2Panel showStd={showStd} t={t} />
      )}
      {activeTab === 'task3' && (
        <Task3Panel t={t} />
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Task 1 Panel Component
   ═══════════════════════════════════════════════════════════════════════════ */
function Task1Panel({ showStd, t }: { showStd: boolean; t: any }) {
  const data = task1Prediction.results as Record<string, any>;
  const keyFindings = task1Prediction.key_findings;

  const renderCell = (val: number | null, std: number | null) => {
    if (val === null) return <span className="text-tertiary font-mono">—</span>;
    return (
      <span className="font-mono text-primary font-semibold">
        {val.toFixed(3)}
        {showStd && std !== null && (
          <span className="text-tertiary text-[10px] ml-1">±{std.toFixed(3)}</span>
        )}
      </span>
    );
  };

  return (
    <div className="flex flex-col gap-6">
      <GlassPanel accentTop monitor className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-subtle bg-secondary text-xs uppercase tracking-wider text-tertiary">
              <th className="p-4 font-semibold">{t('results_page.method')}</th>
              <th className="p-4 font-semibold">Macro-F1</th>
              <th className="p-4 font-semibold">AUROC (Full Set)</th>
              <th className="p-4 font-semibold">PR-AUC</th>
              <th className="p-4 font-semibold">Ext. AUROC</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-subtle text-sm">
            {Object.entries(data).map(([methodName, metrics]) => {
              const isOurs = methodName.includes('TrustGuard');
              return (
                <tr
                  key={methodName}
                  className={`transition-colors hover:bg-secondary/40 ${
                    isOurs ? 'bg-sky-500/5 font-semibold' : ''
                  }`}
                >
                  <td className={`p-4 ${isOurs ? 'text-monitor font-bold' : 'text-primary'}`}>
                    {methodName}
                    {isOurs && <span className="text-[10px] bg-sky-500/10 text-sky-400 px-1.5 py-0.5 rounded ml-2">Ours</span>}
                  </td>
                  <td className="p-4">{renderCell(metrics.macro_f1, metrics.macro_f1_std)}</td>
                  <td className="p-4">{renderCell(metrics.auroc, metrics.auroc_std)}</td>
                  <td className="p-4">{renderCell(metrics.pr_auc, metrics.pr_auc_std)}</td>
                  <td className="p-4">{renderCell(metrics.ext_auroc, metrics.ext_auroc_std)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </GlassPanel>

      <KeyFindingsCard findings={keyFindings} t={t} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Task 2 Panel Component
   ═══════════════════════════════════════════════════════════════════════════ */
function Task2Panel({ showStd, t }: { showStd: boolean; t: any }) {
  const data = task2Enforcement.results as Record<string, any>;
  const keyFindings = task2Enforcement.key_findings;

  const renderCell = (val: number | null, std: number | null, suffix = '%', decimals = 1) => {
    if (val === null) return <span className="text-tertiary font-mono">—</span>;
    return (
      <span className="font-mono text-primary font-semibold">
        {val.toFixed(decimals)}{suffix}
        {showStd && std !== null && (
          <span className="text-tertiary text-[10px] ml-1">±{std.toFixed(decimals)}</span>
        )}
      </span>
    );
  };

  return (
    <div className="flex flex-col gap-6">
      <GlassPanel accentTop safe className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-subtle bg-secondary text-xs uppercase tracking-wider text-tertiary">
              <th className="p-4 font-semibold">{t('results_page.method')}</th>
              <th className="p-4 font-semibold">AIPR</th>
              <th className="p-4 font-semibold">EPR</th>
              <th className="p-4 font-semibold">AET-R</th>
              <th className="p-4 font-semibold">PRR</th>
              <th className="p-4 font-semibold">FRR (Limit: 2.5%)</th>
              <th className="p-4 font-semibold">FIR</th>
              <th className="p-4 font-semibold">Latency</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-subtle text-sm">
            {Object.entries(data).map(([methodName, metrics]) => {
              const isOurs = methodName.includes('TrustGuard');
              return (
                <tr
                  key={methodName}
                  className={`transition-colors hover:bg-secondary/40 ${
                    isOurs ? 'bg-sky-500/5 font-semibold' : ''
                  }`}
                >
                  <td className={`p-4 ${isOurs ? 'text-monitor font-bold' : 'text-primary'}`}>
                    {methodName}
                    {isOurs && <span className="text-[10px] bg-sky-500/10 text-sky-400 px-1.5 py-0.5 rounded ml-2">Ours</span>}
                  </td>
                  <td className="p-4">{renderCell(metrics.AIPR_pct ?? null, metrics.AIPR_std ?? null)}</td>
                  <td className="p-4">{renderCell(metrics.EPR_pct ?? null, metrics.EPR_std ?? null)}</td>
                  <td className="p-4">{renderCell(metrics.AETR_pct ?? null, metrics.AETR_std ?? null)}</td>
                  <td className="p-4">{renderCell(metrics.PRR_pct ?? null, metrics.PRR_std ?? null)}</td>
                  <td className="p-4">{renderCell(metrics.FRR_pct ?? null, metrics.FRR_std ?? null)}</td>
                  <td className="p-4">{renderCell(metrics.FIR_pct ?? null, metrics.FIR_std ?? null)}</td>
                  <td className="p-4">
                    {metrics.latency_s !== null ? (
                      <span className="font-mono text-primary font-semibold">{metrics.latency_s.toFixed(1)}s</span>
                    ) : (
                      <span className="text-tertiary font-mono">—</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </GlassPanel>

      <KeyFindingsCard findings={keyFindings} t={t} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Task 3 Panel Component
   ═══════════════════════════════════════════════════════════════════════════ */
function Task3Panel({ t }: { t: any }) {
  const data = adversarialRobustness.results as Record<string, any>;
  const keyFindings = adversarialRobustness.key_findings;

  const renderCell = (val: number) => {
    return <span className="font-mono text-primary font-semibold">{val.toFixed(3)}</span>;
  };

  return (
    <div className="flex flex-col gap-6">
      <GlassPanel accentTop enforce className="overflow-x-auto">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-subtle bg-secondary text-xs uppercase tracking-wider text-tertiary">
              <th className="p-4 font-semibold">{t('results_page.method')}</th>
              <th className="p-4 font-semibold">Clean (Baseline)</th>
              <th className="p-4 font-semibold">Manifest Mimicry (MM)</th>
              <th className="p-4 font-semibold">RTMA (Timing Attack)</th>
              <th className="p-4 font-semibold">MM + RTMA (Composed)</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-subtle text-sm">
            {Object.entries(data).map(([methodName, metrics]) => {
              const isOurs = methodName.includes('TrustGuard');
              return (
                <tr
                  key={methodName}
                  className={`transition-colors hover:bg-secondary/40 ${
                    isOurs ? 'bg-sky-500/5 font-semibold' : ''
                  }`}
                >
                  <td className={`p-4 ${isOurs ? 'text-monitor font-bold' : 'text-primary'}`}>
                    {methodName}
                    {isOurs && <span className="text-[10px] bg-sky-500/10 text-sky-400 px-1.5 py-0.5 rounded ml-2">Ours</span>}
                  </td>
                  <td className="p-4">{renderCell(metrics.clean)}</td>
                  <td className="p-4">{renderCell(metrics.MM)}</td>
                  <td className="p-4">{renderCell(metrics.RTMA)}</td>
                  <td className="p-4">{renderCell(metrics.clean - (metrics.clean - metrics['MM+RTMA'])) /* fallback calculation safely */}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </GlassPanel>

      <KeyFindingsCard findings={keyFindings} t={t} />
    </div>
  );
}

/* ── Key Findings Reusable Card ──────────────────────────────────────────── */
function KeyFindingsCard({ findings, t }: { findings: string[]; t: any }) {
  return (
    <GlassPanel accentTop constraint className="p-5">
      <h4 className="text-sm font-bold text-primary mb-3 uppercase tracking-wide">
        {t('results_page.key_findings')}
      </h4>
      <ul className="list-disc pl-5 flex flex-col gap-2 text-xs text-secondary">
        {findings.map((finding, idx) => (
          <li key={idx} className="leading-relaxed">{finding}</li>
        ))}
      </ul>
    </GlassPanel>
  );
}

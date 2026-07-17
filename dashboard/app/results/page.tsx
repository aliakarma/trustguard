'use client';

import React, { useEffect, useState } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import { useTranslations } from 'next-intl';
import { CheckCircle2 } from 'lucide-react';

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

  const tabs: { id: TabType; label: string }[] = [
    { id: 'task1', label: t('results_page.task1_tab') },
    { id: 'task2', label: t('results_page.task2_tab') },
    { id: 'task3', label: t('results_page.task3_tab') },
  ];

  return (
    <div className="flex flex-col gap-6 animate-fade-in-up">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div className="seg w-full md:w-auto overflow-x-auto">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`seg-item flex-1 md:flex-none ${activeTab === tab.id ? 'seg-item--active' : ''}`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab !== 'task3' && (
          <label className="flex items-center gap-2 text-xs text-secondary cursor-pointer select-none">
            <input
              type="checkbox"
              checked={showStd}
              onChange={() => setShowStd(!showStd)}
              className="h-4 w-4 rounded"
            />
            <span>{t('results_page.show_std')}</span>
          </label>
        )}
      </div>

      {activeTab === 'task1' && <Task1Panel showStd={showStd} t={t} />}
      {activeTab === 'task2' && <Task2Panel showStd={showStd} t={t} />}
      {activeTab === 'task3' && <Task3Panel t={t} />}
    </div>
  );
}

/* ── Shared cell renderer ─────────────────────────────────────────────────── */
function MetricValue({
  val,
  std,
  showStd,
  decimals,
  suffix,
  leader,
}: {
  val: number | null;
  std?: number | null;
  showStd: boolean;
  decimals: number;
  suffix?: string;
  leader?: boolean;
}) {
  if (val === null || val === undefined) return <span className="text-tertiary text-mono">—</span>;
  return (
    <span className={`text-mono font-semibold inline-flex items-baseline gap-1 ${leader ? 'text-safe' : 'text-primary'}`}>
      {leader && <span className="inline-block h-1.5 w-1.5 rounded-full bg-safe self-center" title="Best in column" />}
      <span>{val.toFixed(decimals)}{suffix}</span>
      {showStd && std !== null && std !== undefined && (
        <span className="text-tertiary text-[10px]">±{std.toFixed(decimals)}</span>
      )}
    </span>
  );
}

function OursRowName({ name }: { name: string }) {
  const ours = name.includes('TrustGuard');
  return (
    <span className={`flex items-center gap-2 ${ours ? 'text-monitor font-bold' : 'text-primary font-medium'}`}>
      {name}
      {ours && <span className="chip">Ours</span>}
    </span>
  );
}

/* ── Task 1 ───────────────────────────────────────────────────────────────── */
function Task1Panel({ showStd, t }: { showStd: boolean; t: any }) {
  const data = task1Prediction.results as Record<string, any>;
  const entries = Object.entries(data);
  const best = (k: string) => Math.max(...entries.map(([, m]) => (m[k] ?? -Infinity)));
  const cols = [
    { key: 'macro_f1', std: 'macro_f1_std', label: 'Macro-F1', dec: 3 },
    { key: 'auroc', std: 'auroc_std', label: 'AUROC (full)', dec: 3 },
    { key: 'pr_auc', std: 'pr_auc_std', label: 'PR-AUC', dec: 3 },
    { key: 'ext_auroc', std: 'ext_auroc_std', label: 'Ext. AUROC', dec: 3 },
  ];
  const maxes = Object.fromEntries(cols.map((c) => [c.key, best(c.key)]));

  return (
    <div className="flex flex-col gap-6">
      <GlassPanel accentTop monitor className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th>{t('results_page.method')}</th>
              {cols.map((c) => <th key={c.key}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {entries.map(([name, m]) => (
              <tr key={name} className={name.includes('TrustGuard') ? 'bg-monitor/[0.05]' : ''}>
                <td className="text-sm"><OursRowName name={name} /></td>
                {cols.map((c) => (
                  <td key={c.key} className="text-sm">
                    <MetricValue val={m[c.key]} std={m[c.std]} showStd={showStd} decimals={c.dec} leader={m[c.key] === maxes[c.key]} />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </GlassPanel>
      <KeyFindingsCard findings={task1Prediction.key_findings} t={t} />
    </div>
  );
}

/* ── Task 2 ───────────────────────────────────────────────────────────────── */
function Task2Panel({ showStd, t }: { showStd: boolean; t: any }) {
  const data = task2Enforcement.results as Record<string, any>;
  const entries = Object.entries(data);
  const bestHigh = (k: string) => Math.max(...entries.map(([, m]) => (m[k] ?? -Infinity)));
  const cols = [
    { key: 'AIPR_pct', std: 'AIPR_std', label: 'AIPR', better: 'high' },
    { key: 'EPR_pct', std: 'EPR_std', label: 'EPR', better: 'high' },
    { key: 'AETR_pct', std: 'AETR_std', label: 'AET-R', better: 'high' },
    { key: 'PRR_pct', std: 'PRR_std', label: 'PRR', better: 'high' },
    { key: 'FRR_pct', std: 'FRR_std', label: 'FRR · budget 2.5%', better: 'none' },
    { key: 'FIR_pct', std: 'FIR_std', label: 'FIR', better: 'none' },
  ];
  const maxes = Object.fromEntries(cols.filter((c) => c.better === 'high').map((c) => [c.key, bestHigh(c.key)]));

  return (
    <div className="flex flex-col gap-6">
      <GlassPanel accentTop safe className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th>{t('results_page.method')}</th>
              {cols.map((c) => <th key={c.key}>{c.label}</th>)}
              <th>Latency</th>
            </tr>
          </thead>
          <tbody>
            {entries.map(([name, m]) => (
              <tr key={name} className={name.includes('TrustGuard') ? 'bg-monitor/[0.05]' : ''}>
                <td className="text-sm"><OursRowName name={name} /></td>
                {cols.map((c) => {
                  const isFrr = c.key === 'FRR_pct';
                  const exceeded = isFrr && m[c.key] !== null && m[c.key] > 2.5;
                  return (
                    <td key={c.key} className="text-sm">
                      {exceeded ? (
                        <span className="text-mono font-semibold text-danger">
                          {m[c.key].toFixed(1)}%
                          {showStd && m[c.std] != null && <span className="text-[10px] ml-1 opacity-70">±{m[c.std].toFixed(1)}</span>}
                        </span>
                      ) : (
                        <MetricValue val={m[c.key]} std={m[c.std]} showStd={showStd} decimals={1} suffix="%" leader={c.better === 'high' && m[c.key] === maxes[c.key]} />
                      )}
                    </td>
                  );
                })}
                <td className="text-sm">
                  {m.latency_s !== null ? (
                    <span className="text-mono font-semibold text-primary">{m.latency_s.toFixed(1)}s</span>
                  ) : (
                    <span className="text-tertiary text-mono">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </GlassPanel>
      <KeyFindingsCard findings={task2Enforcement.key_findings} t={t} />
    </div>
  );
}

/* ── Task 3 ───────────────────────────────────────────────────────────────── */
function Task3Panel({ t }: { t: any }) {
  const data = adversarialRobustness.results as Record<string, any>;
  const entries = Object.entries(data);

  const cols: { key: string; label: string }[] = [
    { key: 'clean', label: 'Clean' },
    { key: 'MM', label: 'Manifest Mimicry' },
    { key: 'RTMA', label: 'RTMA (timing)' },
    { key: 'MM+RTMA', label: 'MM + RTMA' },
  ];

  return (
    <div className="flex flex-col gap-6">
      <GlassPanel accentTop enforce className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th>{t('results_page.method')}</th>
              {cols.map((c) => <th key={c.key}>{c.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {entries.map(([name, m]) => (
              <tr key={name} className={name.includes('TrustGuard') ? 'bg-monitor/[0.05]' : ''}>
                <td className="text-sm"><OursRowName name={name} /></td>
                {cols.map((c) => {
                  const val = m[c.key] as number;
                  const drop = c.key === 'clean' ? 0 : m.clean - val;
                  return (
                    <td key={c.key} className="text-sm">
                      <div className="flex flex-col gap-0.5">
                        <span className="text-mono font-semibold text-primary">{val.toFixed(3)}</span>
                        {c.key !== 'clean' && (
                          <span className={`text-[10px] text-mono ${drop > 0.05 ? 'text-danger' : drop > 0.001 ? 'text-warning' : 'text-tertiary'}`}>
                            {drop > 0.001 ? `−${drop.toFixed(3)}` : '±0.000'}
                          </span>
                        )}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </GlassPanel>
      <KeyFindingsCard findings={adversarialRobustness.key_findings} t={t} />
    </div>
  );
}

/* ── Key findings ─────────────────────────────────────────────────────────── */
function KeyFindingsCard({ findings, t }: { findings: string[]; t: any }) {
  return (
    <GlassPanel accentTop constraint className="p-6">
      <h4 className="stat-label mb-4">{t('results_page.key_findings')}</h4>
      <ul className="flex flex-col gap-3">
        {findings.map((finding, idx) => (
          <li key={idx} className="flex items-start gap-2.5 text-sm text-secondary leading-relaxed">
            <CheckCircle2 size={16} className="text-constraint mt-0.5 flex-shrink-0" />
            <span>{finding}</span>
          </li>
        ))}
      </ul>
    </GlassPanel>
  );
}

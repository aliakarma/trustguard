'use client';

import React, { useEffect, useState } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import KaTeXBlock from '@/components/common/KaTeXBlock';
import { useTranslations } from 'next-intl';

// Import local JSON files directly
import sensitivityAnalyses from '@/lib/results/sensitivity_analyses.json';
import ablations from '@/lib/results/ablations.json';
import constraintDynamics from '@/lib/results/constraint_dynamics.json';

export default function SensitivityStudio() {
  const { setActivePage } = useGlobalStore();
  const t = useTranslations();
  const [activeTab, setActiveTab] = useState<'sensitivity' | 'ablation' | 'dynamics'>('sensitivity');

  useEffect(() => {
    setActivePage('sensitivity');
  }, [setActivePage]);

  return (
    <div className="flex flex-col gap-6 animate-fade-in-up">
      {/* ── Tabs Navigation ─────────────────────────────────────────── */}
      <div className="flex gap-2 bg-panel p-4 rounded-xl border border-subtle">
        <button
          onClick={() => setActiveTab('sensitivity')}
          className={`btn text-xs font-semibold px-4 py-2.5 rounded-lg transition-all ${
            activeTab === 'sensitivity'
              ? 'bg-monitor text-white'
              : 'bg-secondary hover:bg-slate-700 text-secondary'
          }`}
        >
          Hyperparameter Sensitivity
        </button>
        <button
          onClick={() => setActiveTab('ablation')}
          className={`btn text-xs font-semibold px-4 py-2.5 rounded-lg transition-all ${
            activeTab === 'ablation'
              ? 'bg-monitor text-white'
              : 'bg-secondary hover:bg-slate-700 text-secondary'
          }`}
        >
          Ablation Studies
        </button>
        <button
          onClick={() => setActiveTab('dynamics')}
          className={`btn text-xs font-semibold px-4 py-2.5 rounded-lg transition-all ${
            activeTab === 'dynamics'
              ? 'bg-monitor text-white'
              : 'bg-secondary hover:bg-slate-700 text-secondary'
          }`}
        >
          Constraint Dynamics
        </button>
      </div>

      {/* ── Content Panels ──────────────────────────────────────────── */}
      {activeTab === 'sensitivity' && <SensitivityPanel t={t} />}
      {activeTab === 'ablation' && <AblationPanel t={t} />}
      {activeTab === 'dynamics' && <DynamicsPanel t={t} />}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Sensitivity Panel
   ═══════════════════════════════════════════════════════════════════════════ */
function SensitivityPanel({ t }: { t: any }) {
  const lambdaResults = sensitivityAnalyses.reward_weights_lambda.results;
  const emaAlphaResults = sensitivityAnalyses.ema_alpha.results;
  const modalityResults = sensitivityAnalyses.encoder_modalities.results;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* 1. Lambda Grid (λ₁ vs λ₂) */}
      <GlassPanel accentTop monitor className="p-5 flex flex-col justify-between">
        <div>
          <h3 className="stat-label mb-2">Reward Weight Grid (λ₁ × λ₂)</h3>
          <p className="text-secondary text-xs mb-4">
            Varying the constraint penalty coefficient <KaTeXBlock math="\lambda_1" /> and enforcement overhead cost <KaTeXBlock math="\lambda_2" />. Budget <KaTeXBlock math="\epsilon_{\text{safe}} = 2.5\%" />.
          </p>

          <div className="grid grid-cols-4 gap-2 text-center text-xs mt-4">
            <div className="font-bold text-tertiary">Grid</div>
            <div className="font-bold text-tertiary">λ₂=0.05</div>
            <div className="font-bold text-tertiary">λ₂=0.10</div>
            <div className="font-bold text-tertiary">λ₂=0.20</div>

            {Object.entries(lambdaResults).map(([lambda1Key, colData]) => (
              <React.Fragment key={lambda1Key}>
                <div className="font-bold text-secondary text-left flex items-center justify-start">{lambda1Key}</div>
                {Object.entries(colData).map(([lambda2Key, metrics]) => {
                  const frrExceeded = metrics.FRR_pct > 2.5;
                  return (
                    <div
                      key={lambda2Key}
                      className={`p-3 rounded-lg border flex flex-col gap-1 transition-all ${
                        frrExceeded
                          ? 'border-red-500/20 bg-red-500/5'
                          : 'border-emerald-500/20 bg-emerald-500/5'
                      }`}
                    >
                      <span className="text-[10px] text-tertiary font-semibold uppercase">AIPR</span>
                      <span className="font-mono font-bold text-primary text-sm">{metrics.AIPR_pct}%</span>
                      <span className={`text-[10px] font-mono ${frrExceeded ? 'text-red-400 font-bold' : 'text-emerald-400'}`}>
                        FRR: {metrics.FRR_pct}%
                      </span>
                    </div>
                  );
                })}
              </React.Fragment>
            ))}
          </div>
        </div>
        <div className="text-[10px] text-tertiary bg-secondary p-3 rounded-lg border border-subtle mt-4">
          <strong>Note:</strong> {sensitivityAnalyses.reward_weights_lambda.note}
        </div>
      </GlassPanel>

      {/* 2. Modality Ablation (Text, Code, Graph) */}
      <GlassPanel accentTop safe className="p-5 flex flex-col justify-between">
        <div>
          <h3 className="stat-label mb-2">Semantic Modality Fusions</h3>
          <p className="text-secondary text-xs mb-4">
            Ablating input modes: Text (BERT Description), Code (CodeBERT snippet), and Graph (GATv2 static call graph).
          </p>

          <div className="flex flex-col gap-3 mt-4">
            {Object.entries(modalityResults).map(([modality, score]) => {
              const isFull = modality.includes('full');
              return (
                <div key={modality} className="flex flex-col gap-1">
                  <div className="flex justify-between text-xs font-semibold text-secondary">
                    <span>{modality}</span>
                    <span className="font-mono text-primary font-bold">{score.toFixed(3)}</span>
                  </div>
                  <div className="h-2 bg-secondary rounded overflow-hidden">
                    <div
                      className={`h-full rounded transition-all duration-1000 ${
                        isFull ? 'bg-gradient-to-r from-sky-400 to-blue-500' : 'bg-slate-500'
                      }`}
                      style={{ width: `${score * 100}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        <div className="text-[10px] text-tertiary bg-secondary p-3 rounded-lg border border-subtle mt-4">
          <strong>Note:</strong> {sensitivityAnalyses.encoder_modalities.note}
        </div>
      </GlassPanel>

      {/* 3. EMA Alpha Sensitivity */}
      <GlassPanel accentTop enforce className="p-5 col-span-1 lg:col-span-2">
        <h3 className="stat-label mb-2">EMA-α Smoothing Sensitivity</h3>
        <p className="text-secondary text-xs mb-4">
          Impact of the exponential moving average coefficient <KaTeXBlock math="\alpha" /> in Layer 3 Risk Tracking.
        </p>

        <div className="grid grid-4 gap-4 mt-4">
          {Object.entries(emaAlphaResults).map(([alphaKey, metrics]: [string, any]) => (
            <div key={alphaKey} className="bg-secondary/40 border border-subtle p-4 rounded-xl flex flex-col justify-between">
              <span className="text-sm font-bold text-monitor">{alphaKey}</span>
              <div className="flex flex-col gap-1 mt-3">
                <div className="flex justify-between text-xs">
                  <span className="text-tertiary">PRR</span>
                  <span className="font-mono font-semibold text-primary">{metrics.PRR_pct}%</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-tertiary">FRR</span>
                  <span className="font-mono font-semibold text-primary">{metrics.FRR_pct}%</span>
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-tertiary">Latency</span>
                  <span className="font-mono font-semibold text-primary">{metrics.median_latency_s}s</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </GlassPanel>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Ablation Panel
   ═══════════════════════════════════════════════════════════════════════════ */
function AblationPanel({ t }: { t: any }) {
  const factorial = ablations.factorial_2x2;
  const components = ablations.component_ablations;
  const nulls = ablations.null_ablations;
  const findings = ablations.key_findings;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Factorial 2x2 Grid */}
      <GlassPanel accentTop enforce className="p-5">
        <h3 className="stat-label mb-2">Factorial 2×2 Ablation Matrix</h3>
        <p className="text-secondary text-xs mb-4">
          Decomposition showing multi-agent formulation vs. single-agent structure, with and without safety constraints.
        </p>

        <div className="grid grid-cols-2 gap-4 mt-4">
          {Object.entries(factorial).map(([name, data]) => {
            const hasConstraint = name.includes('+');
            const isOurs = name.includes('TrustGuard');
            const isViolated = data.FRR_pct > 2.5;

            return (
              <div
                key={name}
                className={`p-4 rounded-xl border flex flex-col justify-between gap-3 ${
                  isOurs
                    ? 'border-sky-500/30 bg-sky-500/5'
                    : isViolated
                    ? 'border-red-500/20 bg-red-500/5'
                    : 'border-subtle bg-secondary/20'
                }`}
              >
                <div>
                  <span className={`text-[10px] font-bold uppercase tracking-wider ${isOurs ? 'text-sky-400' : 'text-tertiary'}`}>
                    {name}
                  </span>
                  <div className="flex items-baseline mt-2">
                    <span className="text-2xl font-bold text-mono">{data.AIPR_pct}%</span>
                    <span className="text-[10px] text-tertiary ml-2 uppercase">AIPR</span>
                  </div>
                </div>

                <div className="flex justify-between items-center border-t border-subtle pt-2">
                  <span className="text-[10px] text-tertiary">FRR</span>
                  <span className={`text-xs font-mono font-bold ${isViolated ? 'text-red-400' : 'text-emerald-400'}`}>
                    {data.FRR_pct}% {isViolated && '✗'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </GlassPanel>

      {/* Component Ablations */}
      <GlassPanel accentTop safe className="p-5">
        <h3 className="stat-label mb-2">Component-Level Contribution</h3>
        <p className="text-secondary text-xs mb-4">
          AIPR drops when disabling the shared belief network, semantic encoder, or active runtime monitoring.
        </p>

        <div className="flex flex-col gap-4 mt-4">
          {Object.entries(components).map(([name, data]) => (
            <div key={name} className="flex justify-between items-center border-b border-subtle pb-3 last:border-0 last:pb-0">
              <span className="text-xs text-secondary max-w-xs">{name}</span>
              <div className="flex items-center gap-4">
                <div className="flex flex-col text-right">
                  <span className="font-mono font-bold text-primary">{data.AIPR_pct}%</span>
                  <span className="text-[9px] text-tertiary uppercase">AIPR</span>
                </div>
                <div className="flex flex-col text-right">
                  <span className="font-mono text-tertiary">{data.FRR_pct}%</span>
                  <span className="text-[9px] text-tertiary uppercase">FRR</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </GlassPanel>

      {/* Findings Card */}
      <div className="lg:col-span-2">
        <GlassPanel accentTop constraint className="p-5">
          <h4 className="text-xs font-bold text-primary mb-3 uppercase tracking-wide">Ablation Takeaways</h4>
          <ul className="list-disc pl-5 flex flex-col gap-2 text-xs text-secondary">
            {findings.map((f, idx) => (
              <li key={idx} className="leading-relaxed">{f}</li>
            ))}
          </ul>
        </GlassPanel>
      </div>
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════════════
   Dynamics Panel
   ═══════════════════════════════════════════════════════════════════════════ */
function DynamicsPanel({ t }: { t: any }) {
  const dynamics = constraintDynamics.training_dynamics;
  const categoryFrr = constraintDynamics.per_category_frr;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* 1. Training Convergence Curves */}
      <GlassPanel accentTop monitor className="p-5 flex flex-col justify-between">
        <div>
          <h3 className="stat-label mb-2">Training Convergence Curves</h3>
          <p className="text-secondary text-xs mb-4">
            Evolution of rolling-window false-revocation rate (FRR) and Lagrange multiplier <KaTeXBlock math="\mu" />.
          </p>

          <div className="flex flex-col gap-4 mt-4">
            {dynamics.iterations.map((iter, idx) => {
              const frr = dynamics.rolling_FRR_pct[idx];
              const mu = dynamics.mu[idx];
              const safeLimit = 2.5;

              return (
                <div key={iter} className="flex justify-between items-center border-b border-subtle pb-3 last:border-0 last:pb-0">
                  <div className="flex flex-col">
                    <span className="text-xs font-bold text-primary">Iter {iter}</span>
                    <span className="text-[10px] text-tertiary">Step checkpoint</span>
                  </div>
                  <div className="flex items-center gap-6">
                    <div className="flex flex-col text-right">
                      <span className={`font-mono font-bold ${frr > safeLimit ? 'text-red-400' : 'text-emerald-400'}`}>
                        {frr.toFixed(1)}%
                      </span>
                      <span className="text-[9px] text-tertiary uppercase">Rolling FRR</span>
                    </div>
                    <div className="flex flex-col text-right">
                      <span className="font-mono font-bold text-primary">{mu.toFixed(2)}</span>
                      <span className="text-[9px] text-tertiary uppercase">Multiplier μ</span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        <div className="text-[10px] text-tertiary bg-secondary p-3 rounded-lg border border-subtle mt-4">
          <strong>Note:</strong> {dynamics.note}
        </div>
      </GlassPanel>

      {/* 2. Per-Category FRR */}
      <GlassPanel accentTop risk className="p-5 flex flex-col justify-between">
        <div>
          <h3 className="stat-label mb-2">Per-Category Fairness Limits</h3>
          <p className="text-secondary text-xs mb-4">
            Variance in group-level category satisfaction (aggregate FRR is 2.1%).
          </p>

          <div className="flex flex-col gap-4 mt-4">
            <div className="flex justify-between border-b border-subtle pb-3">
              <span className="text-xs text-secondary">Minimum Category: <strong>{categoryFrr.min.category}</strong></span>
              <span className="font-mono text-emerald-400 font-bold">{categoryFrr.min.FRR_pct}%</span>
            </div>
            <div className="flex justify-between border-b border-subtle pb-3">
              <span className="text-xs text-secondary">Maximum Category: <strong>{categoryFrr.max.category}</strong></span>
              <span className="font-mono text-red-400 font-bold">{categoryFrr.max.FRR_pct}%</span>
            </div>

            <div className="flex flex-col gap-2 mt-2">
              <span className="text-xs font-bold text-primary">High Variance Categories:</span>
              {Object.entries(categoryFrr.high_variance_categories).map(([cat, val]: [string, any]) => (
                <div key={cat} className="flex justify-between pl-4 text-xs">
                  <span className="text-tertiary">{cat}</span>
                  <span className="font-mono text-primary font-semibold">{val}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="text-[10px] text-tertiary bg-secondary p-3 rounded-lg border border-subtle mt-4">
          <strong>Note:</strong> {categoryFrr.note}
        </div>
      </GlassPanel>
    </div>
  );
}

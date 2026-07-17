'use client';

import React, { useEffect, useState } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import KaTeXBlock from '@/components/common/KaTeXBlock';
import { useTranslations } from 'next-intl';
import { ShieldAlert } from 'lucide-react';

export default function AdversarialLab() {
  const { setActivePage } = useGlobalStore();
  const t = useTranslations();

  // Attack configurations
  const [attackType, setAttackType] = useState<'none' | 'mm' | 'rtma' | 'composed'>('none');
  const [mmIntensity, setMmIntensity] = useState<number>(3); // permissions added
  const [rtmaShim, setRtmaShim] = useState<string>('benign-matched');

  // Stress tests
  const [prevalence, setPrevalence] = useState<number>(28.6);
  const [recalibration, setRecalibration] = useState<boolean>(false);

  useEffect(() => {
    setActivePage('adversarial');
  }, [setActivePage]);

  // Compute stats on the fly based on selections
  let activeAuroc = 0.963;
  let activeAipr = 63.4;
  let activeFrr = 2.1;

  if (attackType === 'mm') {
    activeAuroc = 0.963 - 0.072 * (mmIntensity / 5);
  } else if (attackType === 'rtma') {
    activeAuroc = 0.847;
  } else if (attackType === 'composed') {
    activeAuroc = 0.802;
  }

  if (prevalence === 2) {
    if (recalibration) {
      activeAipr = 56.2;
      activeFrr = 2.4;
    } else {
      activeAipr = 61.6;
      activeFrr = 4.7;
    }
  }

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 animate-fade-in-up">
      {/* ── 1. Attack Configuration Panel (Left sidebar, xl:col-span-1) ── */}
      <div className="xl:col-span-1 flex flex-col gap-6">
        <GlassPanel accentTop enforce className="p-5 flex flex-col gap-4">
          <h3 className="stat-label">Adversarial Attack Simulation</h3>
          <p className="text-secondary text-xs mb-2">
            Simulate timing shims and declared-manifest mimicry parameters to evaluate classifier robustness bounds.
          </p>

          {/* Attack Select */}
          <div className="flex flex-col gap-1">
            <label className="text-xs font-semibold text-secondary">Attack Vector</label>
            <select
              value={attackType}
              onChange={(e: any) => setAttackType(e.target.value)}
              className="field text-xs text-primary"
            >
              <option value="none" className="bg-slate-900">None (Clean Environment)</option>
              <option value="mm" className="bg-slate-900">Manifest Mimicry (MM)</option>
              <option value="rtma" className="bg-slate-900">Trace Mimicry (RTMA)</option>
              <option value="composed" className="bg-slate-900">Composed (MM + RTMA)</option>
            </select>
          </div>

          {/* MM Options */}
          {attackType === 'mm' && (
            <div className="flex flex-col gap-2">
              <label className="text-xs font-semibold text-secondary">
                MM Intensity (low-risk permissions added): {mmIntensity}
              </label>
              <input
                type="range"
                min={1}
                max={10}
                step={1}
                value={mmIntensity}
                onChange={(e) => setMmIntensity(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
          )}

          {/* RTMA Options */}
          {(attackType === 'rtma' || attackType === 'composed') && (
            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-secondary">RTMA Timing Profile</label>
              <select
                value={rtmaShim}
                onChange={(e) => setRtmaShim(e.target.value)}
                className="field text-xs text-primary"
              >
                <option value="benign-matched" className="bg-slate-900">Benign category timing matched</option>
                <option value="random" className="bg-slate-900">Random delay distribution</option>
              </select>
            </div>
          )}
        </GlassPanel>

        {/* Stress Tests Panel */}
        <GlassPanel accentTop safe className="p-5 flex flex-col gap-4">
          <h3 className="stat-label">Distribution Drift & Stress Testing</h3>
          <p className="text-secondary text-xs mb-2">
            Vary malicious sample prevalence or toggle Dual Recalibration updating the safety Lagrangian variable.
          </p>

          {/* Prevalence Selection */}
          <div className="flex flex-col gap-1">
            <label className="text-xs font-semibold text-secondary">Malware Prevalence</label>
            <select
              value={prevalence}
              onChange={(e) => setPrevalence(parseFloat(e.target.value))}
              className="field text-xs text-primary"
            >
              <option value="28.6" className="bg-slate-900">28.6% (AASE Training Baseline)</option>
              <option value="2" className="bg-slate-900">2.0% Low Prevalence Stress</option>
            </select>
          </div>

          {/* Recalibration Toggle */}
          {prevalence === 2 && (
            <label className="flex items-center gap-2 text-xs text-secondary cursor-pointer mt-2">
              <input
                type="checkbox"
                checked={recalibration}
                onChange={() => setRecalibration(!recalibration)}
                className="rounded border-subtle text-monitor focus:ring-monitor h-3.5 w-3.5"
              />
              <span>Enable 72h Lagrangian Recalibration</span>
            </label>
          )}
        </GlassPanel>
      </div>

      {/* ── 2. Visualizations Area (Right column, xl:col-span-2) ──────── */}
      <div className="xl:col-span-2 flex flex-col gap-6">
        {/* AUROC & Enforcement Stats Card */}
        <GlassPanel accentTop monitor className="p-6">
          <div className="flex items-center justify-between gap-2 mb-5">
            <h3 className="stat-label">Adversarial Performance Output</h3>
            <span className={`badge ${attackType === 'none' ? 'badge--safe' : 'badge--danger'}`}>
              {attackType === 'none' ? 'Clean environment' : 'Under attack'}
            </span>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6">
            <div className="bg-surface/50 border border-subtle rounded-xl p-4 flex flex-col gap-2">
              <span className="eyebrow">Robustness AUROC</span>
              <div className="flex items-baseline gap-2">
                <span className="stat-value" style={{ fontSize: '2rem' }}>{activeAuroc.toFixed(3)}</span>
                {attackType !== 'none' && (
                  <span className="text-xs text-mono text-danger font-semibold">−{(0.963 - activeAuroc).toFixed(3)}</span>
                )}
              </div>
              <div className="h-1.5 bg-surface rounded-full overflow-hidden mt-1">
                <div className="h-full rounded-full transition-all duration-700" style={{ width: `${activeAuroc * 100}%`, background: 'linear-gradient(90deg,#38BDF8,#6366F1)' }} />
              </div>
              <span className="text-[10px] text-tertiary">clean baseline 0.963</span>
            </div>

            <div className="bg-surface/50 border border-subtle rounded-xl p-4 flex flex-col gap-2">
              <span className="eyebrow">Safety budget · FRR</span>
              <div className="flex items-baseline gap-2">
                <span className="stat-value" style={{ fontSize: '2rem', color: activeFrr > 2.5 ? 'var(--accent-danger)' : 'var(--accent-safe)' }}>
                  {activeFrr.toFixed(2)}%
                </span>
                <span className={`badge ${activeFrr > 2.5 ? 'badge--danger' : 'badge--safe'}`}>{activeFrr > 2.5 ? 'Exceeded' : 'Within budget'}</span>
              </div>
              <div className="h-1.5 bg-surface rounded-full overflow-hidden mt-1 relative">
                <div className="absolute top-0 bottom-0 w-px bg-constraint" style={{ left: `${(2.5 / 5) * 100}%` }} />
                <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.min((activeFrr / 5) * 100, 100)}%`, background: activeFrr > 2.5 ? 'var(--accent-danger)' : 'var(--accent-safe)' }} />
              </div>
              <span className="text-[10px] text-tertiary">budget ε_safe = 2.5%</span>
            </div>
          </div>

          <div className="flex items-start gap-2.5 border-t border-subtle pt-4 text-xs text-secondary leading-relaxed">
            <ShieldAlert size={15} className="text-monitor mt-0.5 flex-shrink-0" />
            <span>
              {attackType === 'none' && 'Clean-environment metrics reflect the standard training split outcomes reported in the paper tables.'}
              {attackType === 'mm' && `Manifest Mimicry inserts low-risk declared permissions. Clean GATv2 / CodeBERT classifiers remain largely insulated — classifier AUROC falls by only ${(0.072 * (mmIntensity / 5)).toFixed(3)} points.`}
              {attackType === 'rtma' && 'Runtime Trace Mimicry delays API arrival distributions. Because TrustGuard uses windowed counts and timing parameters, RTMA directly targets the temporal observation sequence — an 11.6 point AUROC degradation.'}
              {attackType === 'composed' && 'The composed attack applies both MM and RTMA timing shims to the APK, causing the AUROC bounds to invert relative to pure static methods.'}
            </span>
          </div>
        </GlassPanel>

        {/* Before / After App Comparison */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Before Card */}
          <GlassPanel className="p-5 flex flex-col gap-3">
            <span className="badge badge--safe">Clean Application</span>
            <h4 className="text-xs font-bold text-primary mt-1 font-mono">com.tools.flashlight</h4>
            <div className="flex flex-col gap-1.5 mt-2 text-xs">
              <div className="flex justify-between border-b border-subtle pb-1">
                <span className="text-tertiary">Declared permissions</span>
                <span className="text-primary font-mono">CAMERA</span>
              </div>
              <div className="flex justify-between border-b border-subtle pb-1">
                <span className="text-tertiary">Timing profile</span>
                <span className="text-safe">Regular burst counts</span>
              </div>
              <div className="flex justify-between">
                <span className="text-tertiary">Governance Action</span>
                <span className="text-safe font-semibold font-mono">no_op</span>
              </div>
            </div>
          </GlassPanel>

          {/* After Card */}
          <GlassPanel className="p-5 flex flex-col gap-3">
            <span className="badge badge--danger">Attacked Application</span>
            <h4 className="text-xs font-bold text-primary mt-1 font-mono">com.tools.flashlight</h4>
            <div className="flex flex-col gap-1.5 mt-2 text-xs">
              <div className="flex justify-between border-b border-subtle pb-1">
                <span className="text-tertiary">Declared permissions</span>
                <span className="text-primary font-mono">
                  {attackType === 'mm' || attackType === 'composed'
                    ? `CAMERA + ${mmIntensity} low-risk`
                    : 'CAMERA'}
                </span>
              </div>
              <div className="flex justify-between border-b border-subtle pb-1">
                <span className="text-tertiary">Timing profile</span>
                <span className={attackType === 'rtma' || attackType === 'composed' ? 'text-danger font-bold' : 'text-safe'}>
                  {attackType === 'rtma' || attackType === 'composed'
                    ? 'RTMA delayed timing shim'
                    : 'Regular burst counts'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-tertiary">Governance Action</span>
                <span className="text-danger font-semibold font-mono">
                  {attackType === 'rtma' || attackType === 'composed'
                    ? 'enforce_delay (delayed alerts)'
                    : 'no_op'}
                </span>
              </div>
            </div>
          </GlassPanel>
        </div>
      </div>
    </div>
  );
}

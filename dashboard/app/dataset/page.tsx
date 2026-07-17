'use client';

import React, { useEffect, useState } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import KaTeXBlock from '@/components/common/KaTeXBlock';
import { APP_CATEGORIES } from '@/lib/constants';
import { useTranslations } from 'next-intl';

export default function DatasetExplorer() {
  const { setActivePage } = useGlobalStore();
  const t = useTranslations();
  const [selectedCategory, setSelectedCategory] = useState<string>('Tools');

  useEffect(() => {
    setActivePage('dataset');
  }, [setActivePage]);

  // Dataset splits data
  const splits = [
    { name: 'Train Split', pct: '70%', benign: 43288, malicious: 10158, total: 53446 },
    { name: 'Validation Split', pct: '10%', benign: 6184, malicious: 1451, total: 7635 },
    { name: 'Test Split', pct: '20%', benign: 12368, malicious: 2903, total: 15271 },
  ];

  return (
    <div className="flex flex-col gap-6 animate-fade-in-up">
      {/* ── 1. Split Distribution Cards ───────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {splits.map((s) => (
          <GlassPanel key={s.name} accentTop monitor className="p-5 flex flex-col justify-between">
            <div>
              <div className="flex justify-between items-baseline mb-2">
                <h4 className="text-sm font-bold text-primary">{s.name}</h4>
                <span className="text-xs font-mono font-bold text-monitor bg-monitor/10 px-2 py-0.5 rounded">
                  {s.pct}
                </span>
              </div>
              <div className="text-3xl font-mono font-bold text-primary tracking-tight">
                {s.total.toLocaleString()}
                <span className="text-xs font-normal text-tertiary ml-2 uppercase">records</span>
              </div>
            </div>

            <div className="flex flex-col gap-2 mt-4 pt-4 border-t border-subtle">
              <div className="flex justify-between text-xs">
                <span className="text-tertiary">Benign Apps</span>
                <span className="font-mono text-emerald-400 font-semibold">{s.benign.toLocaleString()}</span>
              </div>
              <div className="flex justify-between text-xs">
                <span className="text-tertiary">Malicious Apps</span>
                <span className="font-mono text-red-400 font-semibold">{s.malicious.toLocaleString()}</span>
              </div>
            </div>
          </GlassPanel>
        ))}
      </div>

      {/* ── 2. Annotation Bootstrap Protocol Flow ────────────────────── */}
      <GlassPanel accentTop constraint className="p-6">
        <h3 className="stat-label mb-4">Annotation Bootstrap Protocol</h3>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mt-2">
          {/* Phase 1 */}
          <div className="bg-secondary/40 border border-subtle p-4 rounded-xl flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-monitor uppercase tracking-wider">Phase 1: Seed Predictor</span>
              <h4 className="text-xs font-bold text-primary mt-1 mb-2">gθ⁽⁰⁾ Bootstrap</h4>
              <p className="text-secondary text-[11px] leading-relaxed">
                Trained on manifest features of 5,560 pre-2013 Drebin malicious apps and matched benign AndroZoo apps to predict legitimacy per permission category.
              </p>
            </div>
          </div>

          {/* Phase 2 */}
          <div className="bg-secondary/40 border border-subtle p-4 rounded-xl flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-accent-risk uppercase tracking-wider" style={{ color: 'var(--accent-risk)' }}>Phase 2: Confident Labeling</span>
              <h4 className="text-xs font-bold text-primary mt-1 mb-2">Category Filtering</h4>
              <p className="text-secondary text-[11px] leading-relaxed">
                Labels resolved by score thresholds: Legitimate if <KaTeXBlock math="g_\theta > 0.70" />, Anomalous if <KaTeXBlock math="g_\theta < 0.05" />.
              </p>
            </div>
          </div>

          {/* Fallback */}
          <div className="bg-secondary/40 border border-subtle p-4 rounded-xl flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold text-red-400 uppercase tracking-wider">TaintDroid Override</span>
              <h4 className="text-xs font-bold text-primary mt-1 mb-2">Ground Truth Alignment</h4>
              <p className="text-secondary text-[11px] leading-relaxed">
                Any (app, permission) pair flagged by TaintDroid as transmitting to unauthorized endpoints is overrides-forced to Anomalous. Remaining ambiguous labels (8%) manually adjudicated.
              </p>
            </div>
          </div>
        </div>
      </GlassPanel>

      {/* ── 3. Category Browser ──────────────────────────────────────── */}
      <GlassPanel accentTop safe className="p-6">
        <h3 className="stat-label mb-4">Ecosystem Category Browser</h3>
        <p className="text-secondary text-xs mb-4">
          Explore representative category configurations for PermissionBench. Showing sample metadata profile.
        </p>

        <div className="flex flex-col lg:flex-row gap-6">
          {/* Category List */}
          <div className="lg:w-64 max-h-[300px] overflow-y-auto border border-subtle rounded-xl p-2 bg-secondary/20 flex flex-col gap-1">
            {APP_CATEGORIES.slice(0, 15).map((cat) => (
              <button
                key={cat}
                type="button"
                onClick={() => setSelectedCategory(cat)}
                className={`text-xs text-left px-3 py-2 rounded-lg transition-all ${
                  selectedCategory === cat
                    ? 'bg-monitor text-white font-bold'
                    : 'text-secondary hover:bg-secondary'
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          {/* Details Panel */}
          <div className="flex-1 bg-secondary/40 border border-subtle p-5 rounded-xl flex flex-col justify-between">
            <div>
              <h4 className="text-sm font-bold text-primary mb-3">Category Metadata Profile: {selectedCategory}</h4>
              <div className="grid grid-2 gap-4 text-xs">
                <div className="flex flex-col gap-1">
                  <span className="text-tertiary">Sample Prevalence (Legitimate)</span>
                  <span className="font-mono text-primary font-bold text-sm">84.2%</span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-tertiary">Avg Declared Permissions</span>
                  <span className="font-mono text-primary font-bold text-sm">6.2 per app</span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-tertiary">Core Expected Permissions</span>
                  <span className="text-primary font-bold">
                    {selectedCategory === 'Communication'
                      ? 'READ_CONTACTS, SEND_SMS'
                      : selectedCategory === 'Maps & Navigation'
                      ? 'ACCESS_FINE_LOCATION, ACCESS_COARSE_LOCATION'
                      : 'INTERNET, READ_EXTERNAL_STORAGE'}
                  </span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-tertiary">Sandbox Workload</span>
                  <span className="text-primary">500 UI Automator events</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </GlassPanel>
    </div>
  );
}

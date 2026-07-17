'use client';

import React, { useEffect } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import { useTranslations } from 'next-intl';

// Import local JSON files directly
import pilotSummary from '@/lib/results/pilot_summary.json';

export default function PilotReport() {
  const { setActivePage } = useGlobalStore();
  const t = useTranslations();

  useEffect(() => {
    setActivePage('pilot');
  }, [setActivePage]);

  const p = pilotSummary.protocol;
  const r = pilotSummary.results;
  const findings = pilotSummary.key_findings;

  return (
    <div className="flex flex-col gap-6 animate-fade-in-up">
      {/* ── 1. Protocol Metadata Row ─────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
        <GlassPanel className="p-4 flex flex-col justify-between" style={{ minHeight: '100px' }}>
          <span className="stat-label">Duration</span>
          <span className="stat-value text-mono text-primary mt-2">
            {p.duration_days}
            <span className="text-sm font-normal ml-1">Days</span>
          </span>
        </GlassPanel>
        <GlassPanel className="p-4 flex flex-col justify-between" style={{ minHeight: '100px' }}>
          <span className="stat-label">Volunteer Devices</span>
          <span className="stat-value text-mono text-primary mt-2">
            {p.devices}
            <span className="text-sm font-normal ml-1">Phones</span>
          </span>
        </GlassPanel>
        <GlassPanel className="p-4 flex flex-col justify-between" style={{ minHeight: '100px' }}>
          <span className="stat-label">Median Apps</span>
          <span className="stat-value text-mono text-primary mt-2">
            {p.median_apps_per_device}
            <span className="text-sm font-normal ml-1">Apps</span>
          </span>
        </GlassPanel>
        <GlassPanel className="p-4 flex flex-col justify-between" style={{ minHeight: '100px' }}>
          <span className="stat-label">Android Range</span>
          <span className="stat-value text-mono text-primary mt-2 text-2xl">
            API {p.android_versions}
          </span>
        </GlassPanel>
        <GlassPanel className="p-4 flex flex-col justify-between" style={{ minHeight: '100px' }}>
          <span className="stat-label">IRB Review</span>
          <span className="stat-value text-mono text-emerald-400 mt-2 text-xl font-bold">
            Approved
          </span>
        </GlassPanel>
      </div>

      {/* ── 2. Headline Results ──────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Latency & Power */}
        <GlassPanel accentTop monitor className="p-5 flex flex-col gap-4">
          <h3 className="stat-label">On-Device Latency & Power Overhead</h3>

          <div className="flex flex-col gap-4 mt-2">
            <div className="flex justify-between items-center border-b border-subtle pb-3">
              <div className="flex flex-col">
                <span className="text-xs font-bold text-primary">Median Alert Latency</span>
                <span className="text-[10px] text-tertiary">Real-device detection lag</span>
              </div>
              <span className="font-mono text-xl font-bold text-primary">{r.median_alert_latency_s}s</span>
            </div>

            <div className="flex justify-between items-center border-b border-subtle pb-3">
              <div className="flex flex-col">
                <span className="text-xs font-bold text-primary">p95 Alert Latency</span>
                <span className="text-[10px] text-tertiary">Worst-case detection delay</span>
              </div>
              <span className="font-mono text-xl font-bold text-primary">{r.p95_alert_latency_s}s</span>
            </div>

            <div className="flex justify-between items-center pb-2">
              <div className="flex flex-col">
                <span className="text-xs font-bold text-primary">Battery Consumption</span>
                <span className="text-[10px] text-tertiary">Hourly battery load overhead</span>
              </div>
              <div className="flex flex-col text-right">
                <span className="font-mono text-xl font-bold text-primary">{r.battery_overhead_mwh_per_hour} mWh</span>
                <span className="text-[9px] text-tertiary italic">{r.battery_note}</span>
              </div>
            </div>
          </div>
        </GlassPanel>

        {/* User Engagement Surveys */}
        <GlassPanel accentTop safe className="p-5 flex flex-col gap-4">
          <h3 className="stat-label">User Acceptance Survey Results</h3>

          <div className="flex flex-col gap-4 mt-2">
            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-xs font-semibold text-secondary">
                <span>Alerts Judged "Acceptable" / Useful</span>
                <span className="font-mono text-primary font-bold">{r.notifications_acceptable_pct}%</span>
              </div>
              <div className="h-3 bg-secondary rounded overflow-hidden">
                <div
                  className="h-full rounded bg-gradient-to-r from-emerald-400 to-teal-500"
                  style={{ width: `${r.notifications_acceptable_pct}%` }}
                />
              </div>
            </div>

            <div className="flex flex-col gap-1">
              <div className="flex justify-between text-xs font-semibold text-secondary">
                <span>Alerts Marked "Intrusive" or Nuisance</span>
                <span className="font-mono text-primary font-bold">{r.notifications_intrusive_pct}%</span>
              </div>
              <div className="h-3 bg-secondary rounded overflow-hidden">
                <div
                  className="h-full rounded bg-gradient-to-r from-red-400 to-rose-500"
                  style={{ width: `${r.notifications_intrusive_pct}%` }}
                />
              </div>
            </div>

            <p className="text-[10px] text-tertiary mt-2">
              <strong>Adjudication protocol:</strong> {p.false_alert_adjudication}
            </p>
          </div>
        </GlassPanel>
      </div>

      {/* ── 3. Sim-to-Real Constraint Gap ────────────────────────────── */}
      <GlassPanel accentTop enforce className="p-6">
        <h3 className="stat-label mb-4 text-red-400">Sim-to-Real Calibration Gap</h3>

        <div className="flex flex-col md:flex-row gap-6 items-center">
          <div className="flex-1 flex flex-col gap-3">
            <p className="text-secondary text-sm leading-relaxed">
              During simulated training, TrustGuard successfully holds the False Revocation Rate (FRR) under the constraint budget of <strong>2.5%</strong>.
            </p>
            <p className="text-secondary text-sm leading-relaxed">
              However, during the 14-day human-grounded deployment, the observed false alert rate rose to <strong>3.4%</strong> (95% CI: [{r.false_alert_ci95_pct[0]}%, {r.false_alert_ci95_pct[1]}%]), exceeding the budget bounds.
            </p>
            <div className="bg-red-500/10 border border-red-500/20 text-red-400 p-4 rounded-xl text-xs font-bold mt-2">
              ⚠ WARNING: The pilot study was run in notification-only mode. Actual automated revocation was disabled in the field.
            </div>
          </div>

          <div className="flex gap-6 items-center bg-secondary p-6 rounded-xl border border-subtle">
            <div className="flex flex-col items-center gap-1">
              <span className="text-[10px] text-tertiary uppercase">Sim Target</span>
              <div className="w-16 h-16 rounded-full border-2 border-emerald-500 flex items-center justify-center font-mono font-bold text-emerald-400 text-lg">
                2.5%
              </div>
            </div>
            <div className="text-xl text-tertiary">→</div>
            <div className="flex flex-col items-center gap-1">
              <span className="text-[10px] text-tertiary uppercase">Field Observed</span>
              <div className="w-16 h-16 rounded-full border-2 border-red-500 flex items-center justify-center font-mono font-bold text-red-400 text-lg" style={{ boxShadow: '0 0 10px rgba(239, 68, 68, 0.2)' }}>
                3.4%
              </div>
            </div>
          </div>
        </div>
      </GlassPanel>

      {/* ── 4. Key Findings List ─────────────────────────────────────── */}
      <GlassPanel accentTop constraint className="p-5">
        <h4 className="text-sm font-bold text-primary mb-3 uppercase tracking-wide">Takeaways and Recalibration Needs</h4>
        <ul className="list-disc pl-5 flex flex-col gap-2 text-xs text-secondary">
          {findings.map((f, idx) => (
            <li key={idx} className="leading-relaxed">{f}</li>
          ))}
        </ul>
      </GlassPanel>
    </div>
  );
}

'use client';

import React, { useEffect, useRef, useState } from 'react';
import GlassPanel from '../common/GlassPanel';
import { useTranslations } from 'next-intl';
import { Info } from 'lucide-react';

type Accent = 'monitor' | 'risk' | 'enforce' | 'safe' | 'constraint';

interface StatCardProps {
  labelKey: string;
  value: number;
  std?: number;
  decimals?: number;
  suffix?: string;
  accentColor?: Accent;
  tooltipKey?: string;
  hint?: string;          // small comparison note, e.g. "+5.1 vs best baseline"
  hintTone?: 'up' | 'down' | 'neutral';
  animate?: boolean;
}

const ACCENT_VAR: Record<Accent, string> = {
  monitor: 'var(--accent-monitor)',
  risk: 'var(--accent-risk)',
  enforce: 'var(--accent-enforce)',
  safe: 'var(--accent-safe)',
  constraint: 'var(--accent-constraint)',
};

export default function StatCard({
  labelKey,
  value,
  std,
  decimals = 1,
  suffix = '',
  accentColor = 'monitor',
  tooltipKey,
  hint,
  hintTone = 'neutral',
  animate = true,
}: StatCardProps) {
  const t = useTranslations();
  const [displayValue, setDisplayValue] = useState(animate ? 0 : value);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    if (!animate) {
      setDisplayValue(value);
      return;
    }
    const start = performance.now();
    const from = 0;
    const to = value;
    const duration = 850;

    const tick = (now: number) => {
      const p = Math.min((now - start) / duration, 1);
      // easeOutExpo
      const eased = p === 1 ? 1 : 1 - Math.pow(2, -10 * p);
      setDisplayValue(from + (to - from) * eased);
      if (p < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [value, animate]);

  const hintColor =
    hintTone === 'up' ? 'var(--accent-safe)' : hintTone === 'down' ? 'var(--accent-danger)' : 'var(--text-tertiary)';

  return (
    <GlassPanel
      accentTop
      monitor={accentColor === 'monitor'}
      risk={accentColor === 'risk'}
      enforce={accentColor === 'enforce'}
      safe={accentColor === 'safe'}
      constraint={accentColor === 'constraint'}
      interactive
      className="animate-fade-in-up p-5 flex flex-col justify-between h-full"
      glow={accentColor === 'enforce' && value > 0.05}
      style={{ minHeight: '128px' }}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <span
            className="inline-block h-2 w-2 rounded-full flex-shrink-0"
            style={{ background: ACCENT_VAR[accentColor], boxShadow: `0 0 8px ${ACCENT_VAR[accentColor]}` }}
          />
          <span className="stat-label truncate">{t(labelKey)}</span>
        </div>
        {tooltipKey && (
          <span className="text-tertiary cursor-help flex-shrink-0" title={t(tooltipKey)} aria-label={t(tooltipKey)}>
            <Info size={13} />
          </span>
        )}
      </div>

      <div className="flex items-baseline gap-2 mt-3">
        <span className="stat-value">
          {displayValue.toFixed(decimals)}
          {suffix && <span className="text-lg font-medium text-secondary ml-0.5">{suffix}</span>}
        </span>
        {std !== undefined && (
          <span className="text-tertiary text-xs text-mono">± {std.toFixed(decimals)}</span>
        )}
      </div>

      {hint && (
        <div className="mt-2.5 text-xs font-medium text-mono" style={{ color: hintColor }}>
          {hintTone === 'up' && '▲ '}
          {hintTone === 'down' && '▼ '}
          {hint}
        </div>
      )}
    </GlassPanel>
  );
}

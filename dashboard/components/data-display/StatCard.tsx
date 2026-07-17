'use client';

import React, { useEffect, useState } from 'react';
import GlassPanel from '../common/GlassPanel';
import { useTranslations } from 'next-intl';

interface StatCardProps {
  labelKey: string;             // i18n key for label
  value: number;                // Numeric value to display
  std?: number;                 // Standard deviation (optional)
  decimals?: number;            // Decimals to show
  suffix?: string;              // Suffix (e.g., '%', 's')
  accentColor?: 'monitor' | 'risk' | 'enforce' | 'safe' | 'constraint';
  tooltipKey?: string;          // i18n key for tooltip
  animate?: boolean;
}

export default function StatCard({
  labelKey,
  value,
  std,
  decimals = 1,
  suffix = '',
  accentColor,
  tooltipKey,
  animate = true,
}: StatCardProps) {
  const t = useTranslations();
  const [displayValue, setDisplayValue] = useState(animate ? 0 : value);

  useEffect(() => {
    if (!animate) {
      setDisplayValue(value);
      return;
    }

    let start = 0;
    const end = value;
    if (start === end) return;

    const duration = 800; // ms
    const increment = (end - start) / (duration / 16); // ~60fps

    const timer = setInterval(() => {
      start += increment;
      if ((increment > 0 && start >= end) || (increment < 0 && start <= end)) {
        clearInterval(timer);
        setDisplayValue(end);
      } else {
        setDisplayValue(start);
      }
    }, 16);

    return () => clearInterval(timer);
  }, [value, animate]);

  return (
    <div className="animate-fade-in-up h-full">
      <GlassPanel
        accentTop
        monitor={accentColor === 'monitor'}
        risk={accentColor === 'risk'}
        enforce={accentColor === 'enforce'}
        safe={accentColor === 'safe'}
        constraint={accentColor === 'constraint'}
        className="p-4 flex flex-col justify-between h-full"
        glow={accentColor === 'enforce' && value > 0.05} // Soft glow if high alert/revocations
        style={{ minHeight: '120px' }}
      >
        <div className="flex justify-between items-start">
          <span className="stat-label" title={tooltipKey ? t(tooltipKey) : undefined}>
            {t(labelKey)}
          </span>
          {tooltipKey && (
            <span
              className="text-tertiary cursor-help text-xs"
              title={t(tooltipKey)}
            >
              ⓘ
            </span>
          )}
        </div>

        <div className="flex items-baseline mt-2">
          <span className="stat-value text-mono">
            {displayValue.toFixed(decimals)}
            <span className="text-lg font-normal ml-0.5">{suffix}</span>
          </span>

          {std !== undefined && (
            <span className="text-tertiary text-xs ml-2 font-mono">
              ± {std.toFixed(decimals)}
            </span>
          )}
        </div>
      </GlassPanel>
    </div>
  );
}

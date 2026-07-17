'use client';

import React, { useEffect, useState } from 'react';

interface GaugeChartProps {
  value: number;        // e.g. 0.021 for 2.1%
  max: number;          // e.g. 0.05
  threshold: number;    // e.g. 0.025
  titleKey: string;
  decimals?: number;
  suffix?: string;
  size?: number;
}

export default function GaugeChart({
  value,
  max,
  threshold,
  titleKey,
  decimals = 2,
  suffix = '%',
  size = 200,
}: GaugeChartProps) {
  const stroke = 13;
  const radius = size / 2 - stroke;
  const circumference = 2 * Math.PI * radius;
  const angleRange = 270;
  const startAngle = -135;

  const pct = Math.min(Math.max(value / max, 0), 1);
  const thresholdPct = Math.min(Math.max(threshold / max, 0), 1);
  const arcFraction = angleRange / 360;

  const [animPct, setAnimPct] = useState(0);
  useEffect(() => {
    const id = requestAnimationFrame(() => setAnimPct(pct));
    return () => cancelAnimationFrame(id);
  }, [pct]);

  const valueOffset = circumference - animPct * arcFraction * circumference;
  const backgroundOffset = circumference - arcFraction * circumference;

  // threshold notch position (at outer edge of arc)
  const thresholdAngle = startAngle + thresholdPct * angleRange;
  const thRad = (thresholdAngle * Math.PI) / 180;
  const cx = size / 2;
  const cy = size / 2;
  const nx1 = cx + (radius - stroke) * Math.cos(thRad);
  const ny1 = cy + (radius - stroke) * Math.sin(thRad);
  const nx2 = cx + (radius + stroke) * Math.cos(thRad);
  const ny2 = cy + (radius + stroke) * Math.sin(thRad);

  const isExceeded = value > threshold;
  const gradId = `gauge-grad-${isExceeded ? 'danger' : 'safe'}`;
  const c1 = isExceeded ? '#FB7185' : '#34D399';
  const c2 = isExceeded ? '#EF4444' : '#10B981';

  return (
    <div className="glass-panel accent-top--constraint p-6 flex flex-col items-center h-full justify-center">
      <div className="flex items-center gap-2 mb-1 self-stretch justify-center">
        <span className="stat-label text-center">{titleKey}</span>
      </div>

      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} className="overflow-visible">
          <defs>
            <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor={c1} />
              <stop offset="100%" stopColor={c2} />
            </linearGradient>
          </defs>

          {/* Track */}
          <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill="none"
            stroke="var(--border-subtle)"
            strokeWidth={stroke}
            strokeDasharray={circumference}
            strokeDashoffset={backgroundOffset}
            strokeLinecap="round"
            transform={`rotate(${startAngle} ${cx} ${cy})`}
          />

          {/* Value arc */}
          <circle
            cx={cx}
            cy={cy}
            r={radius}
            fill="none"
            stroke={`url(#${gradId})`}
            strokeWidth={stroke}
            strokeDasharray={circumference}
            strokeDashoffset={valueOffset}
            strokeLinecap="round"
            transform={`rotate(${startAngle} ${cx} ${cy})`}
            style={{
              transition: 'stroke-dashoffset 1s cubic-bezier(0.16,1,0.3,1)',
              filter: `drop-shadow(0 0 7px ${isExceeded ? 'rgba(239,68,68,0.45)' : 'rgba(16,185,129,0.4)'})`,
            }}
          />

          {/* Threshold notch */}
          <line
            x1={nx1}
            y1={ny1}
            x2={nx2}
            y2={ny2}
            stroke="var(--accent-constraint)"
            strokeWidth={2.5}
            strokeLinecap="round"
          />
        </svg>

        {/* Center */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="stat-value" style={{ fontSize: '2.1rem' }}>
            {(value * 100).toFixed(decimals)}
            <span className="text-base font-medium text-secondary ml-0.5">{suffix}</span>
          </span>
          <span
            className={`badge mt-2 ${isExceeded ? 'badge--danger' : 'badge--safe'}`}
          >
            {isExceeded ? 'Exceeded' : 'Within budget'}
          </span>
        </div>
      </div>

      <div className="flex justify-between w-full mt-5 text-xs text-tertiary text-mono px-1">
        <span>0{suffix}</span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-0.5 rounded" style={{ background: 'var(--accent-constraint)' }} />
          budget {(threshold * 100).toFixed(decimals)}{suffix}
        </span>
        <span>{(max * 100).toFixed(decimals)}{suffix}</span>
      </div>
    </div>
  );
}

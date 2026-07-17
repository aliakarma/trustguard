'use client';

import React from 'react';

interface GaugeChartProps {
  value: number;        // Current value (e.g. 0.021 for 2.1%)
  max: number;          // Max range (e.g. 0.05 for 5%)
  threshold: number;    // Budget threshold (e.g. 0.025 for 2.5%)
  titleKey: string;     // Title label key
  decimals?: number;    // Display decimal places
  suffix?: string;      // Display suffix (e.g. '%')
  size?: number;        // Diameter in pixels
}

export default function GaugeChart({
  value,
  max,
  threshold,
  titleKey,
  decimals = 2,
  suffix = '%',
  size = 180,
}: GaugeChartProps) {
  const radius = size / 2 - 10;
  const circumference = 2 * Math.PI * radius;

  // Render arc from 0 to 270 degrees (leaving bottom open)
  const angleRange = 270;
  const startAngle = -135; // centered at top

  // Calculate percentage of value within the max range
  const pct = Math.min(Math.max(value / max, 0), 1);
  const thresholdPct = Math.min(Math.max(threshold / max, 0), 1);

  // Dash offsets for SVG strokes
  const strokeDashoffset = circumference - (pct * (angleRange / 360)) * circumference;
  const backgroundOffset = circumference - (angleRange / 360) * circumference;

  // Threshold markers
  const thresholdAngle = startAngle + thresholdPct * angleRange;
  const thresholdRad = (thresholdAngle * Math.PI) / 180;
  const thresholdX = size / 2 + (radius + 8) * Math.cos(thresholdRad);
  const thresholdY = size / 2 + (radius + 8) * Math.sin(thresholdRad);

  const isExceeded = value > threshold;
  const strokeColor = isExceeded ? 'var(--accent-danger)' : 'var(--accent-safe)';
  const glowStyle = isExceeded ? { filter: 'drop-shadow(0px 0px 8px rgba(239, 68, 68, 0.4))' } : { filter: 'drop-shadow(0px 0px 8px rgba(16, 185, 129, 0.4))' };

  return (
    <div className="flex flex-col items-center p-4 glass-panel accent-top--constraint">
      <h3 className="stat-label text-center mb-4">{titleKey}</h3>
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size}>
          {/* Background Arc */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="var(--border-subtle)"
            strokeWidth="10"
            strokeDasharray={circumference}
            strokeDashoffset={backgroundOffset}
            strokeLinecap="round"
            transform={`rotate(${startAngle} ${size / 2} ${size / 2})`}
          />
          {/* Value Arc */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={strokeColor}
            strokeWidth="12"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            strokeLinecap="round"
            style={glowStyle}
            transform={`rotate(${startAngle} ${size / 2} ${size / 2})`}
          />
        </svg>

        {/* Threshold Indicator Pin */}
        <div
          className="absolute w-2 h-2 rounded-full bg-purple-400"
          style={{
            left: thresholdX - 4,
            top: thresholdY - 4,
            boxShadow: '0 0 8px var(--accent-constraint)',
          }}
          title={`Budget Threshold: ${(threshold * 100).toFixed(decimals)}${suffix}`}
        />

        {/* Center Text Value */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="stat-value text-3xl">
            {(value * 100).toFixed(decimals)}
            <span className="text-base font-normal ml-0.5">{suffix}</span>
          </span>
          <span className="text-tertiary text-xs mt-1">
            {isExceeded ? 'Exceeded ✗' : 'Within Budget ✓'}
          </span>
        </div>
      </div>

      <div className="flex justify-between w-full mt-4 text-xs text-tertiary px-2">
        <span>0{suffix}</span>
        <span>Limit: {(threshold * 100).toFixed(decimals)}{suffix}</span>
        <span>{(max * 100).toFixed(decimals)}{suffix}</span>
      </div>
    </div>
  );
}

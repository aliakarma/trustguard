'use client';

import React from 'react';
import { Info } from 'lucide-react';

interface ParamSliderProps {
  label: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (val: number) => void;
  tooltip?: string;
  decimals?: number;
  suffix?: string;
}

export default function ParamSlider({
  label,
  min,
  max,
  step,
  value,
  onChange,
  tooltip,
  decimals = 2,
  suffix = '',
}: ParamSliderProps) {
  const filled = ((value - min) / (max - min)) * 100;

  return (
    <div className="flex flex-col gap-2.5">
      <div className="flex justify-between items-center gap-2">
        <span className="text-xs font-medium text-secondary flex items-center gap-1.5 min-w-0">
          <span className="truncate">{label}</span>
          {tooltip && (
            <span className="text-tertiary cursor-help flex-shrink-0" title={tooltip} aria-label={tooltip}>
              <Info size={12} />
            </span>
          )}
        </span>
        <span className="text-xs text-mono font-semibold text-primary bg-surface border border-subtle rounded-md px-2 py-0.5 flex-shrink-0">
          {value.toFixed(decimals)}
          {suffix}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        aria-label={label}
        style={{
          background: `linear-gradient(to right, var(--accent-monitor) 0%, var(--accent-monitor) ${filled}%, var(--border-subtle) ${filled}%, var(--border-subtle) 100%)`,
        }}
      />
    </div>
  );
}

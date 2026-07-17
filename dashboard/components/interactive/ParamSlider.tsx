'use client';

import React from 'react';

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
  return (
    <div className="flex flex-col gap-2 p-3 glass-panel border border-subtle">
      <div className="flex justify-between items-center">
        <span className="text-xs font-semibold text-secondary flex items-center gap-1">
          {label}
          {tooltip && (
            <span className="text-tertiary cursor-help" title={tooltip}>
              ⓘ
            </span>
          )}
        </span>
        <span className="text-sm font-mono font-bold text-primary">
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
        className="w-full h-1 bg-border-subtle rounded-lg appearance-none cursor-pointer accent-monitor"
        style={{
          background: `linear-gradient(to right, var(--accent-monitor) 0%, var(--accent-monitor) ${((value - min) / (max - min)) * 100}%, var(--border-subtle) ${((value - min) / (max - min)) * 100}%, var(--border-subtle) 100%)`,
        }}
      />
    </div>
  );
}

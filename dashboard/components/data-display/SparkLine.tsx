'use client';

import React from 'react';

interface SparkLineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
  fillColor?: string;
  strokeWidth?: number;
}

export default function SparkLine({
  data,
  width = 120,
  height = 40,
  color = 'var(--accent-monitor)',
  fillColor = 'rgba(56, 189, 248, 0.1)',
  strokeWidth = 2,
}: SparkLineProps) {
  if (data.length === 0) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min === 0 ? 1 : max - min;

  const points = data.map((val, index) => {
    const x = (index / (data.length - 1)) * width;
    // Invert Y axis for SVG rendering
    const y = height - ((val - min) / range) * height;
    return { x, y };
  });

  const pathD = points
    .map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`)
    .join(' ');

  const fillD = `${pathD} L ${width.toFixed(1)} ${height.toFixed(1)} L 0 ${height.toFixed(1)} Z`;

  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      {/* Area Fill */}
      {data.length > 1 && (
        <path d={fillD} fill={fillColor} stroke="none" />
      )}
      {/* Line Path */}
      <path
        d={pathD}
        fill="none"
        stroke={color}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

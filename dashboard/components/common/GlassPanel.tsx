'use client';

import React from 'react';

interface GlassPanelProps {
  children: React.ReactNode;
  className?: string;
  interactive?: boolean;
  monitor?: boolean;
  risk?: boolean;
  enforce?: boolean;
  safe?: boolean;
  constraint?: boolean;
  accentTop?: boolean;
  glow?: boolean;
  onClick?: () => void;
  style?: React.CSSProperties;
}

export default function GlassPanel({
  children,
  className = '',
  interactive = false,
  monitor = false,
  risk = false,
  enforce = false,
  safe = false,
  constraint = false,
  accentTop = false,
  glow = false,
  onClick,
  style,
}: GlassPanelProps) {
  const classes = [
    'glass-panel',
    interactive ? 'glass-panel--interactive' : '',
    monitor ? 'glass-panel--monitor' : '',
    risk ? 'glass-panel--risk' : '',
    enforce ? 'glass-panel--enforce' : '',
    onClick ? 'cursor-pointer' : '',
    glow ? 'glow-pulse' : '',
    // Accent top borders
    accentTop && monitor ? 'accent-top--monitor' : '',
    accentTop && risk ? 'accent-top--risk' : '',
    accentTop && enforce ? 'accent-top--enforce' : '',
    accentTop && safe ? 'accent-top--safe' : '',
    accentTop && constraint ? 'accent-top--constraint' : '',
    className,
  ].filter(Boolean).join(' ');

  return (
    <div className={classes} onClick={onClick} style={style}>
      {children}
    </div>
  );
}

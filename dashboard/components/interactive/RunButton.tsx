'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Play, Square, Pause, Loader2 } from 'lucide-react';
import { useTranslations } from 'next-intl';

interface RunButtonProps {
  status: 'idle' | 'running' | 'paused' | 'loading';
  onClick: () => void;
  onStop?: () => void;
}

export default function RunButton({ status, onClick, onStop }: RunButtonProps) {
  const t = useTranslations('simulation');

  const isLoading = status === 'loading';
  const isRunning = status === 'running';
  const isPaused = status === 'paused';

  const primaryStyle: React.CSSProperties = isRunning
    ? { background: 'var(--accent-warning)', color: 'var(--on-warning)' }
    : { background: 'var(--accent-safe)', color: 'var(--on-safe)' };

  return (
    <div className="flex gap-2 w-full">
      <motion.button
        type="button"
        whileHover={{ scale: isLoading ? 1 : 1.015 }}
        whileTap={{ scale: isLoading ? 1 : 0.985 }}
        onClick={isLoading ? undefined : onClick}
        disabled={isLoading}
        style={primaryStyle}
        className={`btn btn--lg flex-1 ${isLoading ? 'opacity-70 cursor-wait' : ''}`}
      >
        {isLoading ? (
          <>
            <Loader2 size={17} className="animate-spin" />
            <span>Connecting…</span>
          </>
        ) : isRunning ? (
          <>
            <Pause size={17} fill="currentColor" />
            <span>{t('pause')}</span>
          </>
        ) : isPaused ? (
          <>
            <Play size={17} fill="currentColor" />
            <span>{t('resume')}</span>
          </>
        ) : (
          <>
            <Play size={17} fill="currentColor" />
            <span>{t('run')}</span>
          </>
        )}
      </motion.button>

      {(isRunning || isPaused) && onStop && (
        <motion.button
          type="button"
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.96 }}
          onClick={onStop}
          className="btn btn--lg btn--danger px-4"
          title={t('stop')}
          aria-label={t('stop')}
        >
          <Square size={16} fill="currentColor" />
        </motion.button>
      )}
    </div>
  );
}

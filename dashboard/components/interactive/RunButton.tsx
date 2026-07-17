'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { Play, Square, Pause } from 'lucide-react';
import { useTranslations } from 'next-intl';

interface RunButtonProps {
  status: 'idle' | 'running' | 'paused' | 'loading';
  onClick: () => void;
  onStop?: () => void;
}

export default function RunButton({
  status,
  onClick,
  onStop,
}: RunButtonProps) {
  const t = useTranslations('simulation');

  const isLoading = status === 'loading';
  const isRunning = status === 'running';
  const isPaused = status === 'paused';
  const isIdle = status === 'idle';

  return (
    <div className="flex gap-2 w-full">
      <motion.button
        type="button"
        whileHover={{ scale: isLoading ? 1 : 1.02 }}
        whileTap={{ scale: isLoading ? 1 : 0.98 }}
        onClick={isLoading ? undefined : onClick}
        className={`btn flex-1 text-sm font-semibold flex items-center justify-center gap-2 py-3 rounded-xl border transition-all ${
          isRunning
            ? 'bg-amber-500 border-amber-600 hover:bg-amber-600 text-white'
            : isPaused
            ? 'bg-emerald-500 border-emerald-600 hover:bg-emerald-600 text-white'
            : 'bg-emerald-500 border-emerald-600 hover:bg-emerald-600 text-white'
        } ${isLoading ? 'opacity-70 cursor-not-allowed' : ''}`}
      >
        {isLoading ? (
          <>
            <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            <span>Connecting...</span>
          </>
        ) : isRunning ? (
          <>
            <Pause size={16} />
            <span>{t('pause')}</span>
          </>
        ) : isPaused ? (
          <>
            <Play size={16} />
            <span>{t('resume')}</span>
          </>
        ) : (
          <>
            <Play size={16} />
            <span>{t('run')}</span>
          </>
        )}
      </motion.button>

      {(isRunning || isPaused) && onStop && (
        <motion.button
          type="button"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={onStop}
          className="btn px-4 py-3 rounded-xl bg-red-500 border border-red-600 hover:bg-red-600 text-white flex items-center justify-center"
          title={t('stop')}
        >
          <Square size={16} fill="white" />
        </motion.button>
      )}
    </div>
  );
}

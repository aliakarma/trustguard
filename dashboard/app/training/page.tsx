'use client';

import React, { useEffect, useState, useRef } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import SparkLine from '@/components/data-display/SparkLine';
import { WS_BASE, BACKEND_HINT } from '@/lib/constants';
import { useTranslations } from 'next-intl';
import { Play, Square, RotateCcw, ServerCrash } from 'lucide-react';

export default function TrainingMonitor() {
  const { setActivePage } = useGlobalStore();
  const t = useTranslations();

  const [status, setStatus] = useState<'idle' | 'running' | 'completed'>('idle');
  const [iteration, setIteration] = useState(0);
  const [reward, setReward] = useState<number[]>([]);
  const [frr, setFrr] = useState<number[]>([]);
  const [mu, setMu] = useState<number[]>([]);
  const [actorLoss, setActorLoss] = useState<number[]>([]);
  const [criticLoss, setCriticLoss] = useState<number[]>([]);
  const [entropy, setEntropy] = useState<number[]>([]);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    setActivePage('training');
    return () => {
      if (socketRef.current) socketRef.current.close();
    };
  }, [setActivePage]);

  const handleStart = () => {
    setStatus('running');
    setError(null);
    setIteration(0);
    setReward([]); setFrr([]); setMu([]);
    setActorLoss([]); setCriticLoss([]); setEntropy([]);

    const wsUrl = `${WS_BASE.replace('http', 'ws')}/ws/training`;
    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    ws.onerror = () => {
      setError(BACKEND_HINT);
      setStatus('idle');
    };

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.done) { setStatus('completed'); ws.close(); return; }
      setIteration(payload.iteration);
      setReward((prev) => [...prev, payload.reward]);
      setFrr((prev) => [...prev, payload.rolling_frr]);
      setMu((prev) => [...prev, payload.mu]);
      setActorLoss((prev) => [...prev, payload.actor_loss]);
      setCriticLoss((prev) => [...prev, payload.critic_loss]);
      setEntropy((prev) => [...prev, payload.entropy]);
    };
    ws.onclose = () => { setStatus((s) => (s === 'completed' ? s : 'idle')); };
  };

  const handleStop = () => {
    if (socketRef.current) socketRef.current.close();
    setStatus('idle');
  };

  const progress = Math.min((iteration / 5000) * 100, 100);

  return (
    <div className="flex flex-col gap-6 animate-fade-in-up">
      {/* ── Controls ──────────────────────────────────────────────────── */}
      <GlassPanel className="p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="flex flex-col gap-2 min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className={`h-2 w-2 rounded-full ${status === 'running' ? 'bg-safe dot-pulse' : ''}`}
              style={status !== 'running' ? { background: status === 'completed' ? 'var(--accent-monitor)' : 'var(--text-tertiary)' } : undefined}
            />
            <h2 className="text-sm font-bold">MAPPO-Lagrangian Training Run</h2>
            <span className={`badge ${status === 'running' ? 'badge--safe' : status === 'completed' ? 'badge--monitor' : 'badge--neutral'}`}>
              {status === 'running' ? 'Streaming' : status === 'completed' ? 'Completed' : 'Idle'}
            </span>
          </div>
          <div className="flex items-center gap-3">
            <div className="h-1.5 flex-1 max-w-xs bg-surface rounded-full overflow-hidden">
              <div className="h-full bg-monitor rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
            </div>
            <span className="text-xs text-tertiary text-mono whitespace-nowrap">
              iter <strong className="text-primary">{iteration.toLocaleString()}</strong> / 5,000
            </span>
          </div>
        </div>

        <div className="flex-shrink-0">
          {status === 'running' ? (
            <button onClick={handleStop} className="btn btn--danger">
              <Square size={15} fill="currentColor" /> Stop Training
            </button>
          ) : (
            <button onClick={handleStart} className="btn btn--primary">
              {status === 'completed' ? <RotateCcw size={15} /> : <Play size={15} fill="currentColor" />}
              {status === 'completed' ? 'Restart Training' : 'Start Training'}
            </button>
          )}
        </div>
      </GlassPanel>

      {error && (
        <div className="glass-panel p-4 flex items-start gap-3" style={{ borderColor: 'color-mix(in srgb, var(--accent-danger) 40%, transparent)' }}>
          <ServerCrash size={18} className="text-danger mt-0.5 flex-shrink-0" />
          <div>
            <p className="text-sm font-semibold text-danger">Backend unavailable</p>
            <p className="text-xs text-secondary mt-1 text-mono">{error}</p>
          </div>
        </div>
      )}

      {/* ── Core curves ───────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <MetricGraph label="Average Episode Reward" value={reward.length ? reward[reward.length - 1].toFixed(2) : '—'} note="ascending" data={reward} color="var(--accent-monitor)" accent="monitor" />
        <MetricGraph label="Rolling False Revocation Rate" value={frr.length ? `${frr[frr.length - 1].toFixed(2)}%` : '—'} note="target < 2.5%" noteTone="safe" data={frr} color="var(--accent-risk)" accent="risk" />
        <MetricGraph label="Lagrange Multiplier μ" value={mu.length ? mu[mu.length - 1].toFixed(4) : '—'} note="converging" data={mu} color="var(--accent-enforce)" accent="enforce" />
      </div>

      {/* ── Losses ────────────────────────────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <MetricGraph label="Actor Loss (policy gradient)" value={actorLoss.length ? actorLoss[actorLoss.length - 1].toFixed(4) : '—'} data={actorLoss} color="var(--accent-monitor)" small />
        <MetricGraph label="Critic Loss (value regression)" value={criticLoss.length ? criticLoss[criticLoss.length - 1].toFixed(4) : '—'} data={criticLoss} color="var(--accent-risk)" small />
        <MetricGraph label="Mean Policy Entropy" value={entropy.length ? entropy[entropy.length - 1].toFixed(4) : '—'} data={entropy} color="var(--accent-safe)" small />
      </div>

      {/* ── Checkpoints ───────────────────────────────────────────────── */}
      <GlassPanel accentTop constraint className="overflow-x-auto">
        <div className="p-5 pb-0">
          <h3 className="stat-label">Model Checkpoints</h3>
        </div>
        <table className="data-table mt-3">
          <thead>
            <tr>
              <th>Name</th><th>Iteration</th><th>Avg Reward</th><th>FRR</th><th>AIPR</th><th>Status</th>
            </tr>
          </thead>
          <tbody className="text-mono">
            <tr>
              <td className="text-primary font-bold">mappo_best.pt</td>
              <td className="text-secondary">4,200</td>
              <td className="text-secondary">12.4</td>
              <td className="text-safe font-semibold">2.1%</td>
              <td className="text-secondary">63.4%</td>
              <td><span className="badge badge--safe">Best checkpoint</span></td>
            </tr>
            <tr>
              <td className="text-primary">mappo_latest.pt</td>
              <td className="text-secondary">5,000</td>
              <td className="text-secondary">11.8</td>
              <td className="text-safe font-semibold">2.1%</td>
              <td className="text-secondary">62.9%</td>
              <td><span className="badge badge--neutral">Latest</span></td>
            </tr>
          </tbody>
        </table>
      </GlassPanel>
    </div>
  );
}

function MetricGraph({
  label, value, note, noteTone, data, color, accent, small,
}: {
  label: string; value: string; note?: string; noteTone?: 'safe' | 'default';
  data: number[]; color: string; accent?: 'monitor' | 'risk' | 'enforce'; small?: boolean;
}) {
  return (
    <GlassPanel accentTop monitor={accent === 'monitor'} risk={accent === 'risk'} enforce={accent === 'enforce'} className="p-5 flex flex-col gap-4">
      <span className="stat-label">{label}</span>
      <div className="flex items-baseline justify-between gap-2">
        <span className={`text-mono font-bold text-primary ${small ? 'text-xl' : 'text-2xl'}`}>{value}</span>
        {note && (
          <span className={`text-[10px] uppercase font-semibold tracking-wide ${noteTone === 'safe' ? 'text-safe' : 'text-tertiary'}`}>{note}</span>
        )}
      </div>
      <div className="rounded-lg border border-subtle bg-surface/40 p-2">
        <SparkLine data={data} width={280} height={small ? 74 : 92} color={color} showDot={data.length > 0} />
      </div>
    </GlassPanel>
  );
}

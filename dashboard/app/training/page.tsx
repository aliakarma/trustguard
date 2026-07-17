'use client';

import React, { useEffect, useState, useRef } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import SparkLine from '@/components/data-display/SparkLine';
import { WS_BASE } from '@/lib/constants';
import { useTranslations } from 'next-intl';

export default function TrainingMonitor() {
  const { setActivePage } = useGlobalStore();
  const t = useTranslations();

  // Training Stream state
  const [status, setStatus] = useState<'idle' | 'running' | 'completed'>('idle');
  const [iteration, setIteration] = useState(0);
  const [reward, setReward] = useState<number[]>([]);
  const [frr, setFrr] = useState<number[]>([]);
  const [mu, setMu] = useState<number[]>([]);
  const [actorLoss, setActorLoss] = useState<number[]>([]);
  const [criticLoss, setCriticLoss] = useState<number[]>([]);
  const [entropy, setEntropy] = useState<number[]>([]);

  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    setActivePage('training');
    return () => {
      if (socketRef.current) socketRef.current.close();
    };
  }, [setActivePage]);

  const handleStart = () => {
    setStatus('running');
    setIteration(0);
    setReward([]);
    setFrr([]);
    setMu([]);
    setActorLoss([]);
    setCriticLoss([]);
    setEntropy([]);

    const wsUrl = `${WS_BASE.replace('http', 'ws')}/ws/training`;
    const ws = new WebSocket(wsUrl);
    socketRef.current = ws;

    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.done) {
        setStatus('completed');
        ws.close();
        return;
      }

      setIteration(payload.iteration);
      setReward((prev) => [...prev, payload.reward]);
      setFrr((prev) => [...prev, payload.rolling_frr]);
      setMu((prev) => [...prev, payload.mu]);
      setActorLoss((prev) => [...prev, payload.actor_loss]);
      setCriticLoss((prev) => [...prev, payload.critic_loss]);
      setEntropy((prev) => [...prev, payload.entropy]);
    };

    ws.onclose = () => {
      if (status !== 'completed') setStatus('idle');
    };
  };

  const handleStop = () => {
    if (socketRef.current) {
      socketRef.current.close();
      setStatus('idle');
    }
  };

  return (
    <div className="flex flex-col gap-6 animate-fade-in-up">
      {/* ── 1. Top Controls Row ───────────────────────────────────────── */}
      <div className="flex justify-between items-center bg-panel p-4 rounded-xl border border-subtle">
        <div className="flex flex-col">
          <h2 className="text-sm font-bold text-primary uppercase tracking-wide">
            MAPPO Lagrangian Training Run
          </h2>
          <span className="text-[10px] text-tertiary">
            Iteration: <strong className="text-primary">{iteration}</strong> / 5000
          </span>
        </div>

        <div className="flex gap-2">
          {status === 'running' ? (
            <button
              onClick={handleStop}
              className="btn bg-red-500 hover:bg-red-600 border border-red-600 text-white text-xs font-semibold py-2 px-4 rounded-lg"
            >
              Stop Training
            </button>
          ) : (
            <button
              onClick={handleStart}
              className="btn bg-emerald-500 hover:bg-emerald-600 border border-emerald-600 text-white text-xs font-semibold py-2 px-4 rounded-lg"
            >
              {status === 'completed' ? 'Restart Training' : 'Start Training'}
            </button>
          )}
        </div>
      </div>

      {/* ── 2. Live Graphs Grid ───────────────────────────────────────── */}
      <div className="grid grid-1 md:grid-3 gap-6">
        {/* Joint Reward */}
        <GlassPanel className="p-5 flex flex-col gap-4">
          <span className="stat-label">Average Episode Reward</span>
          <div className="flex justify-between items-baseline">
            <span className="text-2xl font-mono font-bold text-primary">
              {reward.length > 0 ? reward[reward.length - 1].toFixed(2) : '—'}
            </span>
            <span className="text-[10px] text-tertiary uppercase">ascending</span>
          </div>
          <div className="flex justify-center py-2 border border-subtle bg-secondary/20 rounded-lg">
            <SparkLine data={reward} width={260} height={100} color="var(--accent-monitor)" />
          </div>
        </GlassPanel>

        {/* Rolling FRR */}
        <GlassPanel className="p-5 flex flex-col gap-4">
          <span className="stat-label">Rolling False Revocation Rate</span>
          <div className="flex justify-between items-baseline">
            <span className="text-2xl font-mono font-bold text-primary">
              {frr.length > 0 ? `${(frr[frr.length - 1]).toFixed(2)}%` : '—'}
            </span>
            <span className="text-[10px] text-emerald-400 font-bold uppercase">target &lt; 2.5%</span>
          </div>
          <div className="flex justify-center py-2 border border-subtle bg-secondary/20 rounded-lg">
            <SparkLine data={frr} width={260} height={100} color="var(--accent-risk)" />
          </div>
        </GlassPanel>

        {/* Lagrangian Multiplier mu */}
        <GlassPanel className="p-5 flex flex-col gap-4">
          <span className="stat-label">Lagrange Multiplier μ</span>
          <div className="flex justify-between items-baseline">
            <span className="text-2xl font-mono font-bold text-primary">
              {mu.length > 0 ? mu[mu.length - 1].toFixed(4) : '—'}
            </span>
            <span className="text-[10px] text-tertiary uppercase">converging</span>
          </div>
          <div className="flex justify-center py-2 border border-subtle bg-secondary/20 rounded-lg">
            <SparkLine data={mu} width={260} height={100} color="var(--accent-enforce)" />
          </div>
        </GlassPanel>
      </div>

      {/* ── 3. Loss & Entropy Grid ───────────────────────────────────── */}
      <div className="grid grid-1 md:grid-3 gap-6">
        {/* Actor Loss */}
        <GlassPanel className="p-5 flex flex-col gap-4">
          <span className="stat-label">Actor Loss (Policy Grad)</span>
          <div className="flex justify-between items-baseline">
            <span className="text-xl font-mono font-bold text-primary">
              {actorLoss.length > 0 ? actorLoss[actorLoss.length - 1].toFixed(4) : '—'}
            </span>
          </div>
          <div className="flex justify-center py-2 border border-subtle bg-secondary/20 rounded-lg">
            <SparkLine data={actorLoss} width={260} height={80} color="var(--accent-monitor)" />
          </div>
        </GlassPanel>

        {/* Critic Loss */}
        <GlassPanel className="p-5 flex flex-col gap-4">
          <span className="stat-label">Critic Loss (Value fn regression)</span>
          <div className="flex justify-between items-baseline">
            <span className="text-xl font-mono font-bold text-primary">
              {criticLoss.length > 0 ? criticLoss[criticLoss.length - 1].toFixed(4) : '—'}
            </span>
          </div>
          <div className="flex justify-center py-2 border border-subtle bg-secondary/20 rounded-lg">
            <SparkLine data={criticLoss} width={260} height={80} color="var(--accent-risk)" />
          </div>
        </GlassPanel>

        {/* Policy Entropy */}
        <GlassPanel className="p-5 flex flex-col gap-4">
          <span className="stat-label">Mean Policy Entropy</span>
          <div className="flex justify-between items-baseline">
            <span className="text-xl font-mono font-bold text-primary">
              {entropy.length > 0 ? entropy[entropy.length - 1].toFixed(4) : '—'}
            </span>
          </div>
          <div className="flex justify-center py-2 border border-subtle bg-secondary/20 rounded-lg">
            <SparkLine data={entropy} width={260} height={80} color="var(--accent-safe)" />
          </div>
        </GlassPanel>
      </div>

      {/* ── 4. Checkpoint browser ────────────────────────────────────── */}
      <GlassPanel accentTop constraint className="p-5">
        <h3 className="stat-label mb-3">Model Checkpoints</h3>
        <table className="w-full text-left text-xs border-collapse">
          <thead>
            <tr className="border-b border-subtle text-tertiary uppercase tracking-wider">
              <th className="pb-2">Name</th>
              <th className="pb-2">Iteration</th>
              <th className="pb-2">Avg Reward</th>
              <th className="pb-2">FRR Rate</th>
              <th className="pb-2">AIPR Rate</th>
              <th className="pb-2">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-subtle text-secondary font-mono">
            <tr>
              <td className="py-3 text-primary font-bold">mappo_best.pt</td>
              <td className="py-3">4,200</td>
              <td className="py-3">12.4</td>
              <td className="py-3 text-emerald-400">2.1%</td>
              <td className="py-3">63.4%</td>
              <td className="py-3">
                <span className="badge badge--safe">Best Checkpoint</span>
              </td>
            </tr>
            <tr>
              <td className="py-3 text-primary">mappo_latest.pt</td>
              <td className="py-3">5,000</td>
              <td className="py-3">11.8</td>
              <td className="py-3 text-emerald-400">2.1%</td>
              <td className="py-3">62.9%</td>
              <td className="py-3">
                <span className="badge badge--secondary text-tertiary">Latest</span>
              </td>
            </tr>
          </tbody>
        </table>
      </GlassPanel>
    </div>
  );
}

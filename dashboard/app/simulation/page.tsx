'use client';

import React, { useEffect, useState, useRef } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import ParamSlider from '@/components/interactive/ParamSlider';
import RunButton from '@/components/interactive/RunButton';
import SparkLine from '@/components/data-display/SparkLine';
import GaugeChart from '@/components/data-display/GaugeChart';
import { API_BASE, WS_BASE, BACKEND_HINT } from '@/lib/constants';
import { useTranslations } from 'next-intl';
import { Play, SlidersHorizontal } from 'lucide-react';

export default function SimulationPlayground() {
  const { setActivePage, wsConnected, setWsConnected } = useGlobalStore();
  const t = useTranslations('simulation');

  // Configuration Sliders
  const [numBenign, setNumBenign] = useState(50);
  const [numMalicious, setNumMalicious] = useState(10);
  const [maxSteps, setMaxSteps] = useState(200);
  const [lambda1, setLambda1] = useState(10.0);
  const [lambda2, setLambda2] = useState(0.1);
  const [lambda3, setLambda3] = useState(1.0);
  const [epsSafe, setEpsSafe] = useState(0.025);
  const [emaAlpha, setEmaAlpha] = useState(0.3);
  const [riskThreshold, setRiskThreshold] = useState(0.5);

  // Simulation state
  const [simStatus, setSimStatus] = useState<'idle' | 'running' | 'paused' | 'loading'>('idle');
  const [currentStep, setCurrentStep] = useState(0);
  const [activeAppsRisk, setActiveAppsRisk] = useState<number[]>([]);
  const [rewardHistory, setRewardHistory] = useState<number[]>([]);
  const [frrHistory, setFrrHistory] = useState<number[]>([]);
  const [aiprHistory, setAiprHistory] = useState<number[]>([]);
  
  // Real-time metrics
  const [currentFrr, setCurrentFrr] = useState(0.0);
  const [currentAipr, setCurrentAipr] = useState(0.0);
  const [cumulativeReward, setCumulativeReward] = useState(0.0);
  const [totalRevocations, setTotalRevocations] = useState(0);
  const [falseRevocations, setFalseRevocations] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  
  // Timelines
  const [monitorActions, setMonitorActions] = useState<number[]>([]);
  const [riskActions, setRiskActions] = useState<number[]>([]);
  const [enforceActions, setEnforceActions] = useState<number[]>([]);

  const socketRef = useRef<WebSocket | null>(null);
  const sessionRef = useRef<string | null>(null);

  useEffect(() => {
    setActivePage('simulation');
    return () => {
      // Clean up connection on unmount
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [setActivePage]);

  const handleStartResume = async () => {
    if (simStatus === 'paused') {
      // Resume
      if (socketRef.current) {
        socketRef.current.send(JSON.stringify({ command: 'resume' }));
        setSimStatus('running');
      }
      return;
    }

    // Start fresh
    setSimStatus('loading');
    setLogs([`[System] Initializing Permission Environment with ${numBenign} Benign / ${numMalicious} Malicious apps...`]);
    setMonitorActions([]);
    setRiskActions([]);
    setEnforceActions([]);
    setRewardHistory([]);
    setFrrHistory([]);
    setAiprHistory([]);

    try {
      const res = await fetch(`${API_BASE}/api/simulation/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          num_benign: numBenign,
          num_malicious: numMalicious,
          max_steps: maxSteps,
          eps_safe: epsSafe,
          ema_alpha: emaAlpha,
          risk_threshold: riskThreshold,
          lambda1,
          lambda2,
          lambda3,
          seed: 42,
          deterministic: false,
        }),
      });
      const data = await res.json();
      sessionRef.current = data.session_id;

      // Connect to WebSocket
      const wsUrl = `${WS_BASE.replace('http', 'ws')}${data.ws_url}`;
      const ws = new WebSocket(wsUrl);
      socketRef.current = ws;

      ws.onopen = () => {
        setWsConnected(true);
        setSimStatus('running');
        setLogs((prev) => [...prev, '[WebSocket] Stream channel connected successfully']);
      };

      ws.onmessage = (event) => {
        const payload = JSON.parse(event.data);
        if (payload.done) {
          setSimStatus('idle');
          setLogs((prev) => [...prev, '[System] Simulation completed successfully']);
          ws.close();
          return;
        }

        // Update states
        setCurrentStep(payload.step);
        setActiveAppsRisk(payload.ema_risk);
        setCurrentFrr(payload.info.frr_cumulative);
        setCurrentAipr(payload.info.aipr_cumulative);
        setCumulativeReward(payload.cumulative_reward);
        setTotalRevocations(payload.info.total_revocations);
        setFalseRevocations(payload.info.false_revocations);

        // Keep rolling history bounds
        setRewardHistory((prev) => [...prev, payload.reward].slice(-50));
        setFrrHistory((prev) => [...prev, payload.info.frr_cumulative].slice(-50));
        setAiprHistory((prev) => [...prev, payload.info.aipr_cumulative].slice(-50));

        // Add action track history
        setMonitorActions((prev) => [...prev, payload.agent_actions.monitoring.action].slice(-40));
        setRiskActions((prev) => [...prev, payload.agent_actions.risk.action].slice(-40));
        setEnforceActions((prev) => [...prev, payload.agent_actions.enforcement.action_type].slice(-40));

        // Form logs
        const actMon = payload.agent_actions.monitoring.action_name;
        const actRisk = payload.agent_actions.risk.action_name;
        const actEnf = payload.agent_actions.enforcement.action_name;
        
        let logMsg = `[t=${payload.step}] Monitor: ${actMon} | Risk: ${actRisk}`;
        if (actEnf !== 'no_op') {
          logMsg += ` | Enforcement: ${actEnf.toUpperCase()} (Total revoc: ${payload.info.total_revocations})`;
        }
        setLogs((prev) => [...prev, logMsg].slice(-30));
      };

      ws.onclose = () => {
        setWsConnected(false);
        if (simStatus !== 'idle') {
          setSimStatus('idle');
        }
      };

      ws.onerror = (err) => {
        console.error('WS Error:', err);
        setSimStatus('idle');
        setLogs((prev) => [...prev, `[Error] ${BACKEND_HINT}`]);
      };

    } catch (err) {
      console.error('Failed to start simulation:', err);
      setSimStatus('idle');
      setLogs((prev) => [...prev, `[Error] ${BACKEND_HINT}`]);
    }
  };

  const handlePause = () => {
    if (socketRef.current) {
      socketRef.current.send(JSON.stringify({ command: 'pause' }));
      setSimStatus('paused');
      setLogs((prev) => [...prev, '[System] Simulation paused']);
    }
  };

  const handleStop = () => {
    if (socketRef.current) {
      socketRef.current.send(JSON.stringify({ command: 'stop' }));
      socketRef.current.close();
      setSimStatus('idle');
      setLogs((prev) => [...prev, '[System] Simulation stopped by user']);
    }
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-4 gap-6 animate-fade-in-up">
      {/* ── 1. Configuration Panel (Left Sidebar, xl:col-span-1) ──────── */}
      <GlassPanel className="xl:col-span-1 flex flex-col gap-4 p-5 h-fit xl:sticky xl:top-20">
        <h3 className="stat-label flex items-center gap-2">
          <SlidersHorizontal size={14} className="text-monitor" /> {t('config')}
        </h3>

        <div className="flex flex-col gap-3">
          <span className="eyebrow">{t('env_settings')}</span>
          <ParamSlider label={t('benign_apps')} min={10} max={200} step={10} value={numBenign} onChange={setNumBenign} decimals={0} />
          <ParamSlider label={t('malicious_apps')} min={1} max={50} step={1} value={numMalicious} onChange={setNumMalicious} decimals={0} />
          <ParamSlider label={t('max_steps')} min={50} max={1000} step={50} value={maxSteps} onChange={setMaxSteps} decimals={0} />
        </div>

        <div className="flex flex-col gap-3 mt-2">
          <span className="eyebrow">{t('reward_weights')}</span>
          <ParamSlider label="λ₁ (False Revocation Penalty)" min={1.0} max={20.0} step={1.0} value={lambda1} onChange={setLambda1} />
          <ParamSlider label="λ₂ (Enforcement Cost)" min={0.01} max={0.5} step={0.01} value={lambda2} onChange={setLambda2} />
          <ParamSlider label="λ₃ (Risk Reduction Weight)" min={0.5} max={2.0} step={0.1} value={lambda3} onChange={setLambda3} />
        </div>

        <div className="flex flex-col gap-3 mt-2">
          <span className="eyebrow">{t('safety_constraint')}</span>
          <ParamSlider label="ε_safe (FRR budget)" min={0.01} max={0.05} step={0.005} value={epsSafe} onChange={setEpsSafe} />
        </div>

        <div className="flex flex-col gap-3 mt-2">
          <span className="eyebrow">{t('risk_settings')}</span>
          <ParamSlider label="EMA α" min={0.1} max={0.7} step={0.1} value={emaAlpha} onChange={setEmaAlpha} />
          <ParamSlider label="Risk Threshold (τ)" min={0.3} max={0.7} step={0.05} value={riskThreshold} onChange={setRiskThreshold} />
        </div>

        <div className="mt-2">
          <RunButton
            status={simStatus}
            onClick={simStatus === 'running' ? handlePause : handleStartResume}
            onStop={handleStop}
          />
        </div>
      </GlassPanel>

      {/* ── 2. Live Grid & Timelines (Center, xl:col-span-2) ──────────── */}
      <div className="xl:col-span-2 flex flex-col gap-6">
        {/* Heatmap/State Visuals */}
        <GlassPanel accentTop monitor className="p-5 flex-1 flex flex-col min-h-[350px]">
          <div className="flex justify-between items-center mb-4">
            <h3 className="stat-label">Live App Risk Landscape (ρ̄)</h3>
            <span className="font-mono text-xs text-secondary">
              {t('step')}: <strong className="text-primary">{currentStep}</strong> / {maxSteps}
            </span>
          </div>

          {activeAppsRisk.length > 0 ? (
            <div className="grid grid-cols-5 md:grid-cols-10 gap-2 overflow-y-auto max-h-[280px] p-2 bg-surface/30 rounded-xl border border-subtle">
              {activeAppsRisk.map((val, idx) => (
                <div
                  key={idx}
                  className="flex flex-col items-center justify-center p-3 rounded-lg border border-subtle transition-all duration-300"
                  style={{
                    backgroundColor: `rgba(239, 68, 68, ${Math.min(Math.max((val - 0.2) * 1.5, 0.0), 0.95)})`,
                    borderLeft: idx >= numBenign ? '3px solid var(--accent-enforce)' : '1px solid var(--border-subtle)',
                  }}
                  title={`App ${idx} (${idx >= numBenign ? 'Malicious' : 'Benign'}): Risk ${val.toFixed(3)}`}
                >
                  <span className="text-[10px] text-tertiary font-bold font-mono">#{idx}</span>
                  <span className="text-sm font-mono font-bold text-primary mt-1">{val.toFixed(2)}</span>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex-1 flex items-center justify-center">
              <div className="empty-state">
                <span className="empty-state__icon"><Play size={22} className="text-tertiary" /></span>
                <span className="text-sm font-semibold text-primary">Simulation idle</span>
                <span className="text-xs text-tertiary max-w-xs">
                  Configure the environment and reward weights, then run to stream the live app-risk landscape.
                </span>
              </div>
            </div>
          )}
        </GlassPanel>

        {/* Timelines Tracker */}
        <GlassPanel accentTop safe className="p-5 flex flex-col gap-4">
          <h3 className="stat-label">Multi-Agent Action Tracks</h3>

          <div className="flex flex-col gap-3 text-xs">
            {/* Lane 1: Monitor */}
            <div className="flex flex-col gap-1.5">
              <span className="font-bold text-secondary">k=1 (Monitor Sampling)</span>
              <div className="flex gap-0.5 overflow-x-auto min-h-6 bg-surface/30 p-1 rounded border border-subtle">
                {monitorActions.map((act, i) => (
                  <div
                    key={i}
                    className={`w-3.5 h-3.5 rounded-sm flex-shrink-0 ${
                      act === 1 ? 'bg-sky-400' : 'bg-slate-700/50'
                    }`}
                    title={act === 1 ? 'Sampled' : 'Idle'}
                  />
                ))}
              </div>
            </div>

            {/* Lane 2: Risk */}
            <div className="flex flex-col gap-1.5">
              <span className="font-bold text-secondary">k=2 (Risk Analysis)</span>
              <div className="flex gap-0.5 overflow-x-auto min-h-6 bg-surface/30 p-1 rounded border border-subtle">
                {riskActions.map((act, i) => (
                  <div
                    key={i}
                    className={`w-3.5 h-3.5 rounded-sm flex-shrink-0 ${
                      act === 1 ? 'bg-amber-400' : 'bg-slate-700/50'
                    }`}
                    title={act === 1 ? 'Analysed' : 'Deferred'}
                  />
                ))}
              </div>
            </div>

            {/* Lane 3: Enforce */}
            <div className="flex flex-col gap-1.5">
              <span className="font-bold text-secondary">k=3 (Enforcement Decisions)</span>
              <div className="flex gap-0.5 overflow-x-auto min-h-6 bg-surface/30 p-1 rounded border border-subtle">
                {enforceActions.map((act, i) => {
                  const colorClass =
                    act === 3
                      ? 'bg-rose-500' // revoke
                      : act === 2
                      ? 'bg-amber-500' // rate_limit
                      : act === 1
                      ? 'bg-indigo-400' // alert
                      : 'bg-slate-700/50';
                  return (
                    <div
                      key={i}
                      className={`w-3.5 h-3.5 rounded-sm flex-shrink-0 ${colorClass}`}
                      title={['no_op', 'alert', 'rate_limit', 'revoke'][act]}
                    />
                  );
                })}
              </div>
            </div>
          </div>
        </GlassPanel>
      </div>

      {/* ── 3. Metrics, Gauges & Logs (Right Column, xl:col-span-1) ───── */}
      <div className="xl:col-span-1 flex flex-col gap-6">
        {/* Real-time stats */}
        <div className="grid grid-cols-2 gap-4">
          <GlassPanel className="p-4 flex flex-col justify-between" style={{ minHeight: '80px' }}>
            <span className="stat-label">FRR (Ratio)</span>
            <span className={`text-xl font-mono font-bold mt-2 ${currentFrr > epsSafe ? 'text-danger' : 'text-safe'}`}>
              {(currentFrr * 100).toFixed(2)}%
            </span>
          </GlassPanel>
          <GlassPanel className="p-4 flex flex-col justify-between" style={{ minHeight: '80px' }}>
            <span className="stat-label">AIPR (Peak)</span>
            <span className="text-xl font-mono font-bold text-primary mt-2">
              {(currentAipr * 100).toFixed(1)}%
            </span>
          </GlassPanel>
        </div>

        {/* Live Sparklines */}
        <GlassPanel className="p-5 flex flex-col gap-3">
          <span className="stat-label">Objective Performance</span>
          <div className="flex justify-between items-center text-xs mt-2 border-b border-subtle pb-3">
            <span className="text-secondary">Joint Reward Flow</span>
            <SparkLine data={rewardHistory} color="var(--accent-monitor)" />
          </div>
          <div className="flex justify-between items-center text-xs mt-2 border-b border-subtle pb-3">
            <span className="text-secondary">FRR Constraint Track</span>
            <SparkLine data={frrHistory} color="var(--accent-risk)" />
          </div>
          <div className="flex justify-between items-center text-xs mt-2">
            <span className="text-secondary">AIPR Target Curve</span>
            <SparkLine data={aiprHistory} color="var(--accent-safe)" />
          </div>
        </GlassPanel>

        {/* Event Logs */}
        <GlassPanel accentTop constraint className="p-5 flex-1 flex flex-col justify-between max-h-[350px]">
          <h3 className="stat-label mb-3">Live Simulation Event Log</h3>
          <div className="flex-1 overflow-y-auto font-mono text-[10px] text-secondary flex flex-col gap-1.5 p-2 bg-surface/30 rounded-lg border border-subtle h-48">
            {logs.map((log, i) => (
              <div key={i} className="leading-relaxed border-b border-subtle/5 pb-1">
                {log}
              </div>
            ))}
          </div>
        </GlassPanel>
      </div>
    </div>
  );
}

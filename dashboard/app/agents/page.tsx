'use client';

import React, { useEffect, useState } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import { API_BASE, BACKEND_HINT } from '@/lib/constants';
import { useTranslations } from 'next-intl';
import { Loader2, Zap, Bot, ServerCrash } from 'lucide-react';

type AgentKey = 'monitoring' | 'risk' | 'enforce';

export default function AgentInspector() {
  const { setActivePage } = useGlobalStore();
  const t = useTranslations();

  const [selectedAgent, setSelectedAgent] = useState<AgentKey>('monitoring');
  const [obsInput, setObsInput] = useState<string>('0.1, 0.2, 0.0, 1.0, 0.3');
  const [isComputing, setIsComputing] = useState(false);
  const [actionResult, setActionResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setActivePage('agent_inspector');
  }, [setActivePage]);

  const handleCompute = async () => {
    setIsComputing(true);
    setError(null);
    try {
      // Parse CSV inputs
      const observation = obsInput.split(',').map((val) => parseFloat(val.trim())).filter((v) => !isNaN(v));

      const res = await fetch(`${API_BASE}/api/agent_forward`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          agent: selectedAgent === 'enforce' ? 'enforcement' : selectedAgent === 'risk' ? 'risk' : 'monitoring',
          observation,
          deterministic: false,
        }),
      });
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const data = await res.json();
      setActionResult(data);
    } catch (err) {
      console.error('Inference failed:', err);
      setError(BACKEND_HINT);
    } finally {
      setIsComputing(false);
    }
  };

  // Reset default input values based on agent choice
  const handleAgentSelect = (agent: AgentKey) => {
    setSelectedAgent(agent);
    setActionResult(null);
    if (agent === 'monitoring') {
      // 42 perms usage counts + time_since_sample + system_load
      setObsInput('0.05, 0.0, 0.1, 0.0, 0.0, 0.2, 2.0, 0.45'); // prefix representation
    } else if (agent === 'risk') {
      // 42 usage deltas + 42 predictions + risk
      setObsInput('0.1, 0.0, 0.3, 0.2, 0.9, 0.85, 0.35'); // prefix representation
    } else {
      // risk + 256 belief dims + 2 history
      setObsInput('0.75, 0.1, -0.2, 0.4, 0.0, 0.12'); // prefix representation
    }
  };

  return (
    <div className="flex flex-col gap-6 animate-fade-in-up">
      {/* ── 1. Three-Agent Overview Columns ─────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Monitoring Card */}
        <GlassPanel accentTop monitor className="p-5 flex flex-col gap-3 justify-between">
          <div>
            <span className="badge badge--info">{t('agents.monitoring_id')}</span>
            <h3 className="text-sm font-bold text-primary mt-2">{t('agents.monitoring')}</h3>
            <p className="text-secondary text-xs leading-relaxed mt-2">
              Adaptively samples permission-usage observations from the runtime environment. Balances monitoring coverage (detect anomalies early) against device overhead (battery, CPU) by learning a sampling-interval policy.
            </p>
          </div>
          <div className="flex flex-col gap-1 border-t border-subtle pt-3 mt-2 text-[10px] text-tertiary">
            <span>OBSERVATION DIM: 44 (|𝒫| + 2)</span>
            <span>ACTION SPACE: {"{ IDLE (0), SAMPLE_NOW (1) }"}</span>
          </div>
        </GlassPanel>

        {/* Risk-Analysis Card */}
        <GlassPanel accentTop risk className="p-5 flex flex-col gap-3 justify-between">
          <div>
            <span className="badge badge--warning">{t('agents.risk_id')}</span>
            <h3 className="text-sm font-bold text-primary mt-2">{t('agents.risk')}</h3>
            <p className="text-secondary text-xs leading-relaxed mt-2">
              Receives the permission usage delta δuᵢᵗ from the Monitoring Agent and produces an updated risk estimate for each application, incorporating both instantaneous deviation and semantic risk.
            </p>
          </div>
          <div className="flex flex-col gap-1 border-t border-subtle pt-3 mt-2 text-[10px] text-tertiary">
            <span>OBSERVATION DIM: 85 (2×|𝒫| + 1)</span>
            <span>ACTION SPACE: {"{ DEFER (0), ANALYSE (1) }"}</span>
          </div>
        </GlassPanel>

        {/* Enforcement Card */}
        <GlassPanel accentTop enforce className="p-5 flex flex-col gap-3 justify-between">
          <div>
            <span className="badge badge--danger">{t('agents.enforcement_id')}</span>
            <h3 className="text-sm font-bold text-primary mt-2">{t('agents.enforcement')}</h3>
            <p className="text-secondary text-xs leading-relaxed mt-2">
              Conditions its policy on the shared belief state bₜ and selects enforcement actions. For targeted actions (rate limit, revoke), also selects which permissions to act on via a binary Bernoulli head.
            </p>
          </div>
          <div className="flex flex-col gap-1 border-t border-subtle pt-3 mt-2 text-[10px] text-tertiary">
            <span>OBSERVATION DIM: 259</span>
            <span>ACTION SPACE: {"{ no_op (0), alert (1), rate_limit (2), revoke (3) }"}</span>
          </div>
        </GlassPanel>
      </div>

      {/* ── 2. Interactive Observation Test Console ─────────────────── */}
      <GlassPanel accentTop constraint className="p-6">
        <h3 className="stat-label mb-4">Observation Inference Sandbox</h3>
        <p className="text-secondary text-xs mb-4">
          Select a role-specialized agent, enter test observations as a comma-separated list of float values, and trigger a forward pass.
        </p>

        <div className="flex flex-col lg:flex-row gap-6">
          {/* Controls */}
          <div className="flex-1 flex flex-col gap-4">
            <div className="seg w-full">
              {(['monitoring', 'risk', 'enforce'] as AgentKey[]).map((agent) => (
                <button
                  key={agent}
                  type="button"
                  onClick={() => handleAgentSelect(agent)}
                  className={`seg-item flex-1 ${selectedAgent === agent ? 'seg-item--active' : ''}`}
                >
                  {agent === 'monitoring' ? 'k=1 · Monitor' : agent === 'risk' ? 'k=2 · Risk' : 'k=3 · Enforce'}
                </button>
              ))}
            </div>

            <div className="flex flex-col gap-1">
              <label className="text-xs font-semibold text-secondary">
                Observation Input Vector (CSV format)
              </label>
              <input
                type="text"
                value={obsInput}
                onChange={(e) => setObsInput(e.target.value)}
                className="field text-xs font-mono text-primary"
              />
              <span className="text-[10px] text-tertiary">
                {selectedAgent === 'monitoring'
                  ? 'First 42: avg per-perm usage. 43rd: time-since-sample. 44th: load [0, 1]'
                  : selectedAgent === 'risk'
                  ? 'First 42: usage delta. Next 42: predictions. 85th: risk'
                  : '1st: mean risk. Next 256: belief. Last 2: history'}
              </span>
            </div>

            <button
              type="button"
              onClick={handleCompute}
              disabled={isComputing}
              className="btn btn--primary btn--lg w-full mt-1"
            >
              {isComputing ? (
                <>
                  <Loader2 size={16} className="animate-spin" />
                  <span>Computing forward pass…</span>
                </>
              ) : (
                <>
                  <Zap size={16} />
                  <span>Compute action</span>
                </>
              )}
            </button>
          </div>

          {/* Inference Results */}
          <div className="flex-1 bg-surface/40 border border-subtle p-5 rounded-xl flex flex-col justify-center min-h-[200px]">
            {actionResult ? (
              <div className="flex flex-col gap-4">
                <div className="flex justify-between items-baseline border-b border-subtle pb-3">
                  <span className="text-xs text-secondary font-bold">Selected Action</span>
                  <span className="font-mono text-lg font-bold text-safe">
                    {actionResult.action_name}
                  </span>
                </div>

                <div className="flex justify-between items-baseline border-b border-subtle pb-3">
                  <span className="text-xs text-secondary">Log Probability</span>
                  <span className="font-mono text-primary font-semibold">
                    {actionResult.log_prob.toFixed(4)}
                  </span>
                </div>

                <div className="flex flex-col gap-2">
                  <span className="text-xs text-secondary font-bold">Policy Probability Distribution</span>
                  {actionResult.policy_distribution.map((prob: number, idx: number) => {
                    const actionName =
                      selectedAgent === 'monitoring'
                        ? ['IDLE', 'SAMPLE_NOW'][idx]
                        : selectedAgent === 'risk'
                        ? ['DEFER', 'ANALYSE'][idx]
                        : ['no_op', 'alert', 'rate_limit', 'revoke'][idx];

                    return (
                      <div key={idx} className="flex flex-col gap-1">
                        <div className="flex justify-between text-[10px] text-tertiary">
                          <span>{actionName}</span>
                          <span className="font-mono text-primary font-semibold">{(prob * 100).toFixed(1)}%</span>
                        </div>
                        <div className="h-2 bg-surface rounded overflow-hidden">
                          <div
                            className="h-full rounded bg-monitor"
                            style={{ width: `${prob * 100}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : error ? (
              <div className="empty-state">
                <span className="empty-state__icon"><ServerCrash size={22} className="text-danger" /></span>
                <span className="text-sm font-semibold text-danger">Backend unavailable</span>
                <span className="text-xs text-secondary max-w-[18rem] text-mono">{error}</span>
              </div>
            ) : (
              <div className="empty-state">
                <span className="empty-state__icon"><Bot size={22} className="text-tertiary" /></span>
                <span className="text-sm font-semibold text-primary">Ready for inference</span>
                <span className="text-xs text-tertiary max-w-[16rem]">
                  Pick an agent, enter an observation vector, and compute the policy&apos;s action distribution.
                </span>
              </div>
            )}
          </div>
        </div>
      </GlassPanel>
    </div>
  );
}

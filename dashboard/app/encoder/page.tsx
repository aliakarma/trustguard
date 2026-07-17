'use client';

import React, { useEffect, useState } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import PermissionSelector from '@/components/interactive/PermissionSelector';
import { APP_CATEGORIES, ANDROID_PERMISSIONS, API_BASE, BACKEND_HINT } from '@/lib/constants';
import { useTranslations } from 'next-intl';
import { Brain, Loader2, ServerCrash } from 'lucide-react';

export default function SemanticEncoderVisualizer() {
  const { setActivePage } = useGlobalStore();
  const t = useTranslations();

  // Inputs
  const [description, setDescription] = useState('A simple flashlight utility to illuminate dark spaces using your device camera flash.');
  const [category, setCategory] = useState<string>('Tools');
  const [declaredPerms, setDeclaredPerms] = useState<string[]>(['CAMERA']);
  const [apiFeatures, setApiFeatures] = useState('android.hardware.Camera.open(), android.hardware.Camera.Parameters');

  // Outputs
  const [isLoading, setIsLoading] = useState(false);
  const [encodedResult, setEncodedResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setActivePage('semantic_encoder');
  }, [setActivePage]);

  const handleEncode = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/encode`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          description,
          category,
          permissions: declaredPerms,
          api_features: apiFeatures,
        }),
      });
      if (!res.ok) throw new Error(`Backend returned ${res.status}`);
      const data = await res.json();
      setEncodedResult(data);
    } catch (err) {
      console.error('Encoding failed:', err);
      setError(BACKEND_HINT);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-6 animate-fade-in-up">
      {/* ── 1. Input Panel (Left Column, xl:col-span-1) ───────────────── */}
      <div className="xl:col-span-1 flex flex-col gap-6">
        <GlassPanel accentTop monitor className="p-5 flex flex-col gap-4">
          <h3 className="stat-label">Application Metadata Input</h3>

          {/* Description */}
          <div className="flex flex-col gap-1">
            <label className="text-xs font-semibold text-secondary">Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="field text-xs h-20 resize-none text-primary"
            />
          </div>

          {/* Category */}
          <div className="flex flex-col gap-1">
            <label className="text-xs font-semibold text-secondary">Store Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="field text-xs text-primary"
            >
              {APP_CATEGORIES.map((cat) => (
                <option key={cat} value={cat} className="bg-slate-900">
                  {cat}
                </option>
              ))}
            </select>
          </div>

          {/* API Code Features */}
          <div className="flex flex-col gap-1">
            <label className="text-xs font-semibold text-secondary">Static Code Features</label>
            <textarea
              value={apiFeatures}
              onChange={(e) => setApiFeatures(e.target.value)}
              className="field text-xs h-20 resize-none text-primary"
            />
          </div>

          {/* Action Button */}
          <button
            type="button"
            onClick={handleEncode}
            disabled={isLoading}
            className="btn btn--primary btn--lg w-full mt-1"
          >
            {isLoading ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                <span>Generating embedding…</span>
              </>
            ) : (
              <>
                <Brain size={16} />
                <span>Fuse modalities &amp; encode</span>
              </>
            )}
          </button>
        </GlassPanel>

        {/* Declarative Permission Selector */}
        <PermissionSelector selected={declaredPerms} onChange={setDeclaredPerms} />
      </div>

      {/* ── 2. Outputs & Visualization (Right Columns, xl:col-span-2) ──── */}
      <div className="xl:col-span-2 flex flex-col gap-6">
        {error && (
          <div className="glass-panel p-4 flex items-start gap-3" style={{ borderColor: 'color-mix(in srgb, var(--accent-danger) 40%, transparent)' }}>
            <ServerCrash size={18} className="text-danger mt-0.5 flex-shrink-0" />
            <div>
              <p className="text-sm font-semibold text-danger">Backend unavailable</p>
              <p className="text-xs text-secondary mt-1 text-mono">{error}</p>
            </div>
          </div>
        )}
        {encodedResult ? (
          <>
            {/* ϕ(fi) Embedding Heatbar */}
            <GlassPanel accentTop safe className="p-5">
              <div className="flex items-center justify-between gap-2 mb-1">
                <h3 className="stat-label">Unified App Embedding ϕ(fᵢ) ∈ ℝ²⁵⁶</h3>
                <span className="badge badge--safe">L2-normalized</span>
              </div>
              <p className="text-secondary text-xs mb-4">
                Fused representation from BERT CLS, GATv2 graph pool, and CodeBERT static features.
              </p>
              <div className="flex flex-wrap gap-[3px] p-3 bg-surface/40 border border-subtle rounded-lg">
                {encodedResult.embedding.map((val: number, idx: number) => (
                  <div
                    key={idx}
                    className="flex-1 min-w-[4px] h-7 rounded-[2px] transition-all"
                    style={{
                      backgroundColor: `color-mix(in srgb, var(--accent-monitor) ${Math.min(Math.max((val + 0.15) * 100, 6), 100)}%, var(--bg-secondary))`,
                    }}
                    title={`Dim ${idx}: ${val.toFixed(4)}`}
                  />
                ))}
              </div>
              <div className="flex items-center justify-between mt-2 text-[10px] text-tertiary text-mono">
                <span>dim 0</span>
                <span>low → high activation</span>
                <span>dim 255</span>
              </div>
            </GlassPanel>

            {/* Permission Predictions & Risk Table */}
            <GlassPanel accentTop enforce className="overflow-hidden">
              <div className="p-5 pb-0 flex items-center justify-between gap-2">
                <h3 className="stat-label">Per-Permission Risk Prediction</h3>
                <span className="text-[10px] text-tertiary text-mono">{ANDROID_PERMISSIONS.length} permissions</span>
              </div>
              <div className="overflow-x-auto mt-3 max-h-[520px] overflow-y-auto">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Permission</th>
                      <th>Declared</th>
                      <th>Expectation p̂ᵢ,ₚ</th>
                      <th>Semantic risk (1 − p̂)</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ANDROID_PERMISSIONS.map((perm, idx) => {
                      const isDeclared = declaredPerms.includes(perm);
                      const prob = encodedResult.predicted_probs[idx];
                      const risk = encodedResult.risk_scores[idx];
                      const isAnomalous = isDeclared && risk > 0.5;
                      return (
                        <tr key={perm} className={isAnomalous ? 'bg-enforce/[0.05]' : ''}>
                          <td className="text-mono text-xs text-primary">{perm}</td>
                          <td>
                            <span className={`badge ${isDeclared ? 'badge--info' : 'badge--neutral'}`}>
                              {isDeclared ? 'Yes' : 'No'}
                            </span>
                          </td>
                          <td>
                            <div className="flex items-center gap-3">
                              <div className="w-20 h-1.5 bg-surface rounded-full overflow-hidden">
                                <div className="h-full rounded-full bg-safe" style={{ width: `${prob * 100}%` }} />
                              </div>
                              <span className="text-mono text-xs text-secondary w-8">{prob.toFixed(2)}</span>
                            </div>
                          </td>
                          <td>
                            <div className="flex items-center gap-3">
                              <div className="w-20 h-1.5 bg-surface rounded-full overflow-hidden">
                                <div className="h-full rounded-full" style={{ width: `${risk * 100}%`, background: isAnomalous ? 'var(--accent-danger)' : 'var(--text-tertiary)' }} />
                              </div>
                              <span className={`text-mono text-xs w-8 ${isAnomalous ? 'text-danger font-bold' : 'text-secondary'}`}>{risk.toFixed(2)}</span>
                            </div>
                          </td>
                          <td>
                            {isAnomalous ? (
                              <span className="badge badge--danger">Anomaly expected</span>
                            ) : (
                              <span className="text-xs text-tertiary">Normal</span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </GlassPanel>
          </>
        ) : (
          <GlassPanel className="flex-1 flex items-center justify-center min-h-[400px]">
            <div className="empty-state">
              <span className="empty-state__icon"><Brain size={24} className="text-tertiary" /></span>
              <h4 className="text-sm font-bold text-primary">No embedding generated yet</h4>
              <p className="text-secondary text-xs max-w-sm">
                Enter application store metadata, select declared permissions, then run the encoder to fuse
                modalities and predict per-permission risk.
              </p>
            </div>
          </GlassPanel>
        )}
      </div>
    </div>
  );
}

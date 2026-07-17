'use client';

import React, { useEffect, useState } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import GlassPanel from '@/components/common/GlassPanel';
import PermissionSelector from '@/components/interactive/PermissionSelector';
import { APP_CATEGORIES, ANDROID_PERMISSIONS, API_BASE } from '@/lib/constants';
import { useTranslations } from 'next-intl';

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

  useEffect(() => {
    setActivePage('semantic_encoder');
  }, [setActivePage]);

  const handleEncode = async () => {
    setIsLoading(true);
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
      const data = await res.json();
      setEncodedResult(data);
    } catch (err) {
      console.error('Encoding failed:', err);
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
              className="bg-secondary/40 border border-subtle rounded-lg p-2 text-xs focus:outline-none focus:border-monitor h-20 resize-none text-primary"
            />
          </div>

          {/* Category */}
          <div className="flex flex-col gap-1">
            <label className="text-xs font-semibold text-secondary">Store Category</label>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="bg-secondary/40 border border-subtle rounded-lg p-2 text-xs focus:outline-none focus:border-monitor text-primary"
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
              className="bg-secondary/40 border border-subtle rounded-lg p-2 text-xs focus:outline-none focus:border-monitor h-20 resize-none text-primary"
            />
          </div>

          {/* Action Button */}
          <button
            type="button"
            onClick={handleEncode}
            disabled={isLoading}
            className="btn btn--primary py-3 text-xs font-bold rounded-xl mt-2 flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Generating Embedding...</span>
              </>
            ) : (
              <span>FUSE MODALITIES & ENCODE</span>
            )}
          </button>
        </GlassPanel>

        {/* Declarative Permission Selector */}
        <PermissionSelector selected={declaredPerms} onChange={setDeclaredPerms} />
      </div>

      {/* ── 2. Outputs & Visualization (Right Columns, xl:col-span-2) ──── */}
      <div className="xl:col-span-2 flex flex-col gap-6">
        {encodedResult ? (
          <>
            {/* ϕ(fi) Embedding Heatbar */}
            <GlassPanel accentTop safe className="p-5">
              <h3 className="stat-label mb-2">Unified Application Embedding ϕ(fᵢ) ∈ ℝ²⁵⁶</h3>
              <p className="text-secondary text-xs mb-4">
                L2-normalized fused representation from BERT CLS, GATv2 graph pool, and CodeBERT static features.
              </p>
              <div className="flex flex-wrap gap-0.5 max-h-16 overflow-y-auto p-2 bg-secondary/30 border border-subtle rounded-lg">
                {encodedResult.embedding.map((val: number, idx: number) => (
                  <div
                    key={idx}
                    className="w-1.5 h-6 rounded-sm transition-all"
                    style={{
                      backgroundColor: `rgba(56, 189, 248, ${Math.min(Math.max((val + 0.15) * 3, 0.05), 1)})`,
                    }}
                    title={`Dim ${idx}: ${val.toFixed(4)}`}
                  />
                ))}
              </div>
            </GlassPanel>

            {/* Permission Predictions & Risk Table */}
            <GlassPanel accentTop enforce className="overflow-x-auto">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-subtle bg-secondary text-xs uppercase tracking-wider text-tertiary">
                    <th className="p-4 font-semibold">Permission</th>
                    <th className="p-4 font-semibold">Declared</th>
                    <th className="p-4 font-semibold">Expectation p̂ᵢ,ₚ</th>
                    <th className="p-4 font-semibold">Semantic Risk (1 − p̂)</th>
                    <th className="p-4 font-semibold">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-subtle text-sm">
                  {ANDROID_PERMISSIONS.map((perm, idx) => {
                    const isDeclared = declaredPerms.includes(perm);
                    const prob = encodedResult.predicted_probs[idx];
                    const risk = encodedResult.risk_scores[idx];
                    const isAnomalous = isDeclared && risk > 0.5;

                    return (
                      <tr
                        key={perm}
                        className={`transition-colors hover:bg-secondary/40 ${
                          isAnomalous ? 'bg-red-500/5 font-semibold' : ''
                        }`}
                      >
                        <td className="p-4 text-primary font-mono text-xs">{perm}</td>
                        <td className="p-4">
                          <span className={`badge ${isDeclared ? 'badge--info' : 'badge--secondary text-tertiary'}`}>
                            {isDeclared ? 'Yes' : 'No'}
                          </span>
                        </td>
                        <td className="p-4">
                          <div className="flex items-center gap-3">
                            <div className="w-20 h-2 bg-secondary rounded overflow-hidden">
                              <div
                                className="h-full rounded bg-emerald-500"
                                style={{ width: `${prob * 100}%` }}
                              />
                            </div>
                            <span className="font-mono text-xs text-secondary">{prob.toFixed(2)}</span>
                          </div>
                        </td>
                        <td className="p-4">
                          <div className="flex items-center gap-3">
                            <div className="w-20 h-2 bg-secondary rounded overflow-hidden">
                              <div
                                className={`h-full rounded ${isAnomalous ? 'bg-red-500' : 'bg-slate-500'}`}
                                style={{ width: `${risk * 100}%` }}
                              />
                            </div>
                            <span className={`font-mono text-xs ${isAnomalous ? 'text-red-400 font-bold' : 'text-secondary'}`}>
                              {risk.toFixed(2)}
                            </span>
                          </div>
                        </td>
                        <td className="p-4">
                          {isAnomalous ? (
                            <span className="badge badge--danger animate-pulse">Anomaly Expected ⚠</span>
                          ) : (
                            <span className="text-xs text-tertiary italic">Normal</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </GlassPanel>
          </>
        ) : (
          <GlassPanel className="p-8 flex flex-col items-center justify-center text-center flex-1 border-dashed">
            <span className="text-4xl mb-4">🧠</span>
            <h4 className="text-sm font-bold text-primary mb-2">No Embedding Generated</h4>
            <p className="text-secondary text-xs max-w-sm">
              Please enter application store metadata, select declared permissions, and click the fuse button to run the semantic prediction model stack.
            </p>
          </GlassPanel>
        )}
      </div>
    </div>
  );
}

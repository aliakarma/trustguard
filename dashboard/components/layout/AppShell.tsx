'use client';

import { useEffect, useLayoutEffect } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import Sidebar from './Sidebar';
import Header from './Header';
import { useTranslations } from 'next-intl';

// useLayoutEffect on the client (runs before child passive effects), useEffect on
// the server (no-op, avoids the SSR warning).
const useIsomorphicLayoutEffect = typeof window !== 'undefined' ? useLayoutEffect : useEffect;

const PAGE_SUBTITLES: Record<string, { en: string; ar: string }> = {
  command_center: { en: 'Headline enforcement metrics and cross-method comparison', ar: 'أبرز مقاييس الإنفاذ ومقارنة الطرق' },
  simulation: { en: 'Configure and stream a live 72h permission-governance episode', ar: 'اضبط وبثّ حلقة حوكمة أذونات حية لمدة 72 ساعة' },
  agent_inspector: { en: 'Inspect the three cooperative policies and run forward passes', ar: 'افحص السياسات التعاونية الثلاث ونفّذ تمريرات أمامية' },
  semantic_encoder: { en: 'Fuse app metadata into ϕ(fᵢ) and predict per-permission risk', ar: 'ادمج بيانات التطبيق في ϕ(fᵢ) وتوقّع مخاطر كل إذن' },
  results: { en: 'Prediction, enforcement, and adversarial results across baselines', ar: 'نتائج التنبؤ والإنفاذ والمتانة عبر خطوط الأساس' },
  adversarial: { en: 'Manifest mimicry, timing attacks, and distribution-drift stress', ar: 'محاكاة البيان وهجمات التوقيت وإجهاد انزياح التوزيع' },
  sensitivity: { en: 'Hyperparameter sensitivity, ablations, and constraint dynamics', ar: 'حساسية المعاملات والدراسات التجريدية وديناميكيات القيد' },
  training: { en: 'Live MAPPO-Lagrangian training curves and checkpoints', ar: 'منحنيات تدريب MAPPO-Lagrangian الحية ونقاط التحقق' },
  pilot: { en: '14-day real-device deployment and sim-to-real calibration', ar: 'نشر ميداني لمدة 14 يومًا ومعايرة من المحاكاة إلى الواقع' },
  dataset: { en: 'PermissionBench splits, annotation protocol, and category browser', ar: 'تقسيمات PermissionBench وبروتوكول التوسيم ومتصفح الفئات' },
};

export default function AppShell({ children }: { children: React.ReactNode }) {
  const { sidebarCollapsed, activePage, locale } = useGlobalStore();
  const t = useTranslations('nav');

  // Rehydrate persisted layout prefs (paired with store skipHydration). Runs in a
  // layout effect so it completes BEFORE any child page's passive effect (e.g.
  // setActivePage) triggers a persist write that would clobber the stored locale.
  useIsomorphicLayoutEffect(() => {
    useGlobalStore.persist.rehydrate();
  }, []);

  // Keep <html dir/lang> in sync with locale — including after rehydration and
  // full page reloads, so Arabic (RTL) survives a refresh.
  useEffect(() => {
    document.documentElement.lang = locale;
    document.documentElement.dir = locale === 'ar' ? 'rtl' : 'ltr';
  }, [locale]);

  const pageTitle = t(activePage);
  const subtitle = PAGE_SUBTITLES[activePage]?.[locale === 'ar' ? 'ar' : 'en'];

  return (
    <div className="app-shell">
      <Sidebar />
      <div className={`main-area ${sidebarCollapsed ? 'main-area--expanded' : ''}`}>
        <Header title={pageTitle} subtitle={subtitle} />
        <main className="page-content">{children}</main>
      </div>
    </div>
  );
}

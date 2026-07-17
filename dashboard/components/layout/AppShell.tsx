'use client';

import { useGlobalStore } from '@/stores/globalStore';
import Sidebar from './Sidebar';
import Header from './Header';
import { useTranslations } from 'next-intl';

export default function AppShell({ children }: { children: React.ReactNode }) {
  const { sidebarCollapsed, activePage, locale } = useGlobalStore();
  const t = useTranslations('nav');

  // Map activePage key to translated label
  const pageTitle = t(activePage);

  return (
    <div className="app-shell">
      <Sidebar />
      <div className={`main-area ${sidebarCollapsed ? 'main-area--expanded' : ''}`}>
        <Header title={pageTitle} breadcrumb={[]} />
        <main className="page-content">
          {children}
        </main>
      </div>
    </div>
  );
}

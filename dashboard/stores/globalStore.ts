import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Locale } from '@/i18n';

/* ═══════════════════════════════════════════════════════════════════════════
   Global application state (Zustand)
   ═══════════════════════════════════════════════════════════════════════════ */

interface GlobalState {
  /* ── Sidebar ────────────────────────────────────────────────────────────── */
  sidebarCollapsed: boolean;
  toggleSidebar: () => void;
  setSidebarCollapsed: (collapsed: boolean) => void;

  /* ── Locale ─────────────────────────────────────────────────────────────── */
  locale: Locale;
  setLocale: (locale: Locale) => void;

  /* ── Connection ─────────────────────────────────────────────────────────── */
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;

  /* ── Active page (for breadcrumbs) ──────────────────────────────────────── */
  activePage: string;
  setActivePage: (page: string) => void;
}

export const useGlobalStore = create<GlobalState>()(
  persist(
    (set) => ({
      sidebarCollapsed: false,
      toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
      setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),

      locale: 'en',
      setLocale: (locale) => set({ locale }),

      wsConnected: false,
      setWsConnected: (connected) => set({ wsConnected: connected }),

      activePage: 'command_center',
      setActivePage: (page) => set({ activePage: page }),
    }),
    {
      name: 'trustguard-global-state',
      partialize: (state) => ({
        sidebarCollapsed: state.sidebarCollapsed,
        locale: state.locale,
      }), // Only persist layout selections to avoid session pollution
    }
  )
);

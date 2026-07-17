'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useGlobalStore } from '@/stores/globalStore';
import { localeNames, localeDirection, type Locale } from '@/i18n';
import {
  Target,
  Play,
  Bot,
  Brain,
  BarChart3,
  Shield,
  FlaskConical,
  TrendingUp,
  Smartphone,
  Database,
  PanelLeftClose,
  PanelLeftOpen,
  Sun,
  Moon,
  Globe,
} from 'lucide-react';
import { useTheme } from 'next-themes';
import styles from './Sidebar.module.css';

/* ── Grouped navigation ───────────────────────────────────────────────────── */
const NAV_GROUPS = [
  {
    label: 'Overview',
    labelAr: 'نظرة عامة',
    items: [
      { key: 'command_center', href: '/', icon: Target, label: 'Command Center', labelAr: 'مركز القيادة' },
    ],
  },
  {
    label: 'Live Systems',
    labelAr: 'الأنظمة الحية',
    items: [
      { key: 'simulation', href: '/simulation', icon: Play, label: 'Live Simulation', labelAr: 'المحاكاة الحية' },
      { key: 'training', href: '/training', icon: TrendingUp, label: 'Training Monitor', labelAr: 'مراقب التدريب' },
    ],
  },
  {
    label: 'Model Internals',
    labelAr: 'داخل النموذج',
    items: [
      { key: 'agent_inspector', href: '/agents', icon: Bot, label: 'Agent Inspector', labelAr: 'مفتش الوكلاء' },
      { key: 'semantic_encoder', href: '/encoder', icon: Brain, label: 'Semantic Encoder', labelAr: 'المُشفّر الدلالي' },
    ],
  },
  {
    label: 'Evaluation',
    labelAr: 'التقييم',
    items: [
      { key: 'results', href: '/results', icon: BarChart3, label: 'Results Explorer', labelAr: 'مستعرض النتائج' },
      { key: 'adversarial', href: '/adversarial', icon: Shield, label: 'Adversarial Lab', labelAr: 'مختبر الهجمات' },
      { key: 'sensitivity', href: '/sensitivity', icon: FlaskConical, label: 'Sensitivity Studio', labelAr: 'استوديو الحساسية' },
    ],
  },
  {
    label: 'Field & Data',
    labelAr: 'الميدان والبيانات',
    items: [
      { key: 'pilot', href: '/pilot', icon: Smartphone, label: 'Real-Device Pilot', labelAr: 'التجربة الميدانية' },
      { key: 'dataset', href: '/dataset', icon: Database, label: 'PermissionBench', labelAr: 'PermissionBench' },
    ],
  },
] as const;

function ShieldMark() {
  return (
    <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 2.5 4.5 5.6v5.9c0 4.6 3.1 8.8 7.5 10 4.4-1.2 7.5-5.4 7.5-10V5.6L12 2.5Z"
        fill="rgba(255,255,255,0.16)"
        stroke="rgba(255,255,255,0.9)"
        strokeWidth="1.3"
        strokeLinejoin="round"
      />
      <path
        d="m8.6 12.2 2.3 2.3 4.4-4.6"
        stroke="#fff"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function Sidebar() {
  const pathname = usePathname();
  const { sidebarCollapsed, toggleSidebar, locale, setLocale } = useGlobalStore();
  const { theme, setTheme } = useTheme();

  const isActive = (href: string) => {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  };

  const handleLocaleToggle = () => {
    const newLocale: Locale = locale === 'en' ? 'ar' : 'en';
    setLocale(newLocale);
    document.documentElement.lang = newLocale;
    document.documentElement.dir = localeDirection[newLocale];
  };

  const handleThemeToggle = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  const ar = locale === 'ar';

  return (
    <aside
      className={`${styles.sidebar} ${sidebarCollapsed ? styles.collapsed : ''}`}
      aria-label="Main navigation"
    >
      {/* ── Brand ─────────────────────────────────────────────────────── */}
      <div className={styles.brand}>
        <div className={styles.logo}>
          <ShieldMark />
        </div>
        {!sidebarCollapsed && (
          <div className={styles.brandText}>
            <span className={styles.brandName}>TrustGuard</span>
            <span className={styles.brandSub}>MARL Governance</span>
          </div>
        )}
      </div>

      {/* ── Navigation ────────────────────────────────────────────────── */}
      <nav className={styles.nav}>
        {NAV_GROUPS.map((group) => (
          <div key={group.label}>
            {!sidebarCollapsed && (
              <div className={styles.sectionLabel}>{ar ? group.labelAr : group.label}</div>
            )}
            {group.items.map((item) => {
              const Icon = item.icon;
              const active = isActive(item.href);
              return (
                <Link
                  key={item.key}
                  href={item.href}
                  className={`${styles.navItem} ${active ? styles.active : ''}`}
                  aria-current={active ? 'page' : undefined}
                  title={sidebarCollapsed ? (ar ? item.labelAr : item.label) : undefined}
                >
                  {active && <div className={styles.activeIndicator} />}
                  <Icon size={18} className={styles.navIcon} />
                  {!sidebarCollapsed && (
                    <span className={styles.navLabel}>{ar ? item.labelAr : item.label}</span>
                  )}
                </Link>
              );
            })}
          </div>
        ))}
      </nav>

      {/* ── Bottom controls ───────────────────────────────────────────── */}
      <div className={styles.bottomControls}>
        <button
          className={styles.controlBtn}
          onClick={handleLocaleToggle}
          title={locale === 'en' ? 'Switch to Arabic' : 'التبديل إلى الإنجليزية'}
        >
          <Globe size={17} />
          {!sidebarCollapsed && <span>{localeNames[locale === 'en' ? 'ar' : 'en']}</span>}
        </button>

        <button
          className={styles.controlBtn}
          onClick={handleThemeToggle}
          title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {theme === 'dark' ? <Sun size={17} /> : <Moon size={17} />}
          {!sidebarCollapsed && <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>}
        </button>

        <button
          className={styles.controlBtn}
          onClick={toggleSidebar}
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {sidebarCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
          {!sidebarCollapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}

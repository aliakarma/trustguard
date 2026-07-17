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

/* ── Navigation Items ─────────────────────────────────────────────────────── */
const NAV_ITEMS = [
  { key: 'command_center', href: '/', icon: Target, label: 'Command Center', labelAr: 'مركز القيادة' },
  { key: 'simulation', href: '/simulation', icon: Play, label: 'Live Simulation', labelAr: 'المحاكاة الحية' },
  { key: 'agent_inspector', href: '/agents', icon: Bot, label: 'Agent Inspector', labelAr: 'مفتش الوكلاء' },
  { key: 'semantic_encoder', href: '/encoder', icon: Brain, label: 'Semantic Encoder', labelAr: 'المُشفّر الدلالي' },
  { key: 'results', href: '/results', icon: BarChart3, label: 'Results Explorer', labelAr: 'مستعرض النتائج' },
  { key: 'adversarial', href: '/adversarial', icon: Shield, label: 'Adversarial Lab', labelAr: 'مختبر الهجمات' },
  { key: 'sensitivity', href: '/sensitivity', icon: FlaskConical, label: 'Sensitivity Studio', labelAr: 'استوديو الحساسية' },
  { key: 'training', href: '/training', icon: TrendingUp, label: 'Training Monitor', labelAr: 'مراقب التدريب' },
  { key: 'pilot', href: '/pilot', icon: Smartphone, label: 'Real-Device Pilot', labelAr: 'التجربة الميدانية' },
  { key: 'dataset', href: '/dataset', icon: Database, label: 'PermissionBench', labelAr: 'PermissionBench' },
] as const;

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

  return (
    <aside
      className={`${styles.sidebar} ${sidebarCollapsed ? styles.collapsed : ''}`}
      aria-label="Main navigation"
    >
      {/* ── Logo / Brand ──────────────────────────────────────────────── */}
      <div className={styles.brand}>
        <div className={styles.logo}>🛡️</div>
        {!sidebarCollapsed && (
          <div className={styles.brandText}>
            <span className={styles.brandName}>TrustGuard</span>
            <span className={styles.brandSub}>Dashboard</span>
          </div>
        )}
      </div>

      {/* ── Navigation Links ──────────────────────────────────────────── */}
      <nav className={styles.nav}>
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.href);
          return (
            <Link
              key={item.key}
              href={item.href}
              className={`${styles.navItem} ${active ? styles.active : ''}`}
              title={sidebarCollapsed ? (locale === 'ar' ? item.labelAr : item.label) : undefined}
            >
              <Icon size={20} className={styles.navIcon} />
              {!sidebarCollapsed && (
                <span className={styles.navLabel}>
                  {locale === 'ar' ? item.labelAr : item.label}
                </span>
              )}
              {active && <div className={styles.activeIndicator} />}
            </Link>
          );
        })}
      </nav>

      {/* ── Bottom Controls ───────────────────────────────────────────── */}
      <div className={styles.bottomControls}>
        {/* Language Toggle */}
        <button
          className={styles.controlBtn}
          onClick={handleLocaleToggle}
          title={locale === 'en' ? 'Switch to Arabic' : 'التبديل إلى الإنجليزية'}
        >
          <Globe size={18} />
          {!sidebarCollapsed && (
            <span>{localeNames[locale === 'en' ? 'ar' : 'en']}</span>
          )}
        </button>

        {/* Theme Toggle */}
        <button
          className={styles.controlBtn}
          onClick={handleThemeToggle}
          title={theme === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
        >
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
          {!sidebarCollapsed && (
            <span>{theme === 'dark' ? 'Light Mode' : 'Dark Mode'}</span>
          )}
        </button>

        {/* Collapse Toggle */}
        <button
          className={styles.controlBtn}
          onClick={toggleSidebar}
          title={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
        >
          {sidebarCollapsed ? <PanelLeftOpen size={18} /> : <PanelLeftClose size={18} />}
          {!sidebarCollapsed && <span>Collapse</span>}
        </button>
      </div>
    </aside>
  );
}

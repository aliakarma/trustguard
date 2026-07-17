'use client';

import { useGlobalStore } from '@/stores/globalStore';
import styles from './Header.module.css';

interface HeaderProps {
  title: string;
  breadcrumb?: string[];
}

export default function Header({ title, breadcrumb }: HeaderProps) {
  const { wsConnected } = useGlobalStore();

  return (
    <header className={styles.header}>
      {/* ── Breadcrumb / Title ──────────────────────────────────────── */}
      <div className={styles.titleArea}>
        {breadcrumb && breadcrumb.length > 0 && (
          <div className={styles.breadcrumb}>
            <span className={styles.breadcrumbHome}>Home</span>
            {breadcrumb.map((item, i) => (
              <span key={i}>
                <span className={styles.breadcrumbSep}>/</span>
                <span className={styles.breadcrumbItem}>{item}</span>
              </span>
            ))}
          </div>
        )}
        <h1 className={styles.pageTitle}>{title}</h1>
      </div>

      {/* ── Spacer ─────────────────────────────────────────────────── */}
      <div className={styles.spacer} />

      {/* ── Connection Status ──────────────────────────────────────── */}
      <div className={styles.connectionStatus}>
        <div
          className={`${styles.statusDot} ${
            wsConnected ? styles.dotConnected : styles.dotDisconnected
          }`}
        />
        <span className={styles.statusText}>
          {wsConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
    </header>
  );
}

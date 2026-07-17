'use client';

import { useGlobalStore } from '@/stores/globalStore';
import styles from './Header.module.css';

interface HeaderProps {
  title: string;
  subtitle?: string;
}

export default function Header({ title, subtitle }: HeaderProps) {
  const { wsConnected } = useGlobalStore();

  return (
    <header className={styles.header}>
      <div className={styles.titleArea}>
        <span className={styles.eyebrow}>
          <span className={styles.eyebrowDot} />
          TrustGuard Dashboard
        </span>
        <h1 className={styles.pageTitle}>{title}</h1>
        {subtitle && <span className={styles.subtitle}>{subtitle}</span>}
      </div>

      <div className={styles.spacer} />

      <div className={styles.right}>
        <span className={styles.envBadge}>MAPPO · Dec-POMDP</span>
        <div className={styles.connectionStatus} title={wsConnected ? 'Live stream connected' : 'Live stream offline'}>
          <div className={`${styles.statusDot} ${wsConnected ? styles.dotConnected : styles.dotDisconnected}`} />
          <span className={styles.statusText}>{wsConnected ? 'Live' : 'Offline'}</span>
        </div>
      </div>
    </header>
  );
}

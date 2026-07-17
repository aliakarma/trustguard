import type { Metadata } from 'next';
import { Providers } from '@/components/Providers';
import AppShell from '@/components/layout/AppShell';
import '@/styles/globals.css';

export const metadata: Metadata = {
  title: 'TrustGuard Dashboard 🛡️',
  description: 'A Multi-Agent Reinforcement Learning Framework for Autonomous Permission Governance in Mobile Ecosystems',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>
        <Providers>
          <AppShell>{children}</AppShell>
        </Providers>
      </body>
    </html>
  );
}


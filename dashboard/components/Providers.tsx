'use client';

import { ThemeProvider } from 'next-themes';
import { NextIntlClientProvider } from 'next-intl';
import { type ReactNode } from 'react';
import { useGlobalStore } from '@/stores/globalStore';
import enMessages from '@/messages/en.json';
import arMessages from '@/messages/ar.json';

function TranslationProvider({ children }: { children: ReactNode }) {
  const locale = useGlobalStore((s) => s.locale);
  const messages = locale === 'ar' ? arMessages : enMessages;

  return (
    <NextIntlClientProvider locale={locale} messages={messages} timeZone="Asia/Riyadh">
      {children}
    </NextIntlClientProvider>
  );
}

export function Providers({ children }: { children: ReactNode }) {
  return (
    <ThemeProvider
      attribute="data-theme"
      defaultTheme="dark"
      enableSystem={false}
      disableTransitionOnChange={false}
    >
      <TranslationProvider>
        {children}
      </TranslationProvider>
    </ThemeProvider>
  );
}


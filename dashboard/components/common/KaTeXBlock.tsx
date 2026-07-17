'use client';

import React, { useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface KaTeXBlockProps {
  math: string;
  block?: boolean;
  className?: string;
}

export default function KaTeXBlock({ math, block = false, className = '' }: KaTeXBlockProps) {
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      katex.render(math, containerRef.current, {
        displayMode: block,
        throwOnError: false,
      });
    }
  }, [math, block]);

  return <span ref={containerRef} className={className} />;
}

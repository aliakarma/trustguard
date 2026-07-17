import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./lib/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        monitor: "var(--accent-monitor)",
        risk: "var(--accent-risk)",
        enforce: "var(--accent-enforce)",
        safe: "var(--accent-safe)",
        constraint: "var(--accent-constraint)",
      },
    },
  },
  plugins: [],
  darkMode: ["class", '[data-theme="dark"]'],
};

export default config;

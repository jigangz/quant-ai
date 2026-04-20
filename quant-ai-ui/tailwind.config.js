/** @type {import('tailwindcss').Config} */
export default {
  darkMode: "class",
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    "./node_modules/@tremor/**/*.{js,ts,jsx,tsx}",
  ],
  safelist: [
    {
      pattern: /^(bg|text|border|ring|fill|stroke)-(slate|zinc|indigo|emerald|rose|amber|sky)-(50|100|200|300|400|500|600|700|800|900|950)$/,
    },
  ],
  theme: {
    extend: {
      colors: {
        background: "rgb(var(--color-bg-page) / <alpha-value>)",
        surface: {
          DEFAULT: "rgb(var(--color-bg-surface) / <alpha-value>)",
          muted: "rgb(var(--color-bg-sunken) / <alpha-value>)",
          border: "rgb(var(--color-border) / <alpha-value>)",
          hover: "rgb(var(--color-bg-sunken) / <alpha-value>)",
        },
        foreground: "rgb(var(--color-text-primary) / <alpha-value>)",
        muted: "rgb(var(--color-text-muted) / <alpha-value>)",
        accent: {
          DEFAULT: "rgb(var(--color-accent) / <alpha-value>)",
          hover: "rgb(var(--color-accent-hover) / <alpha-value>)",
          ring: "rgb(var(--color-accent-ring) / 0.2)",
          foreground: "rgb(255 255 255 / <alpha-value>)",
        },
        up: "rgb(var(--color-up) / <alpha-value>)",
        down: "rgb(var(--color-down) / <alpha-value>)",
        warn: "rgb(var(--color-warn) / <alpha-value>)",
        info: "rgb(var(--color-info) / <alpha-value>)",
        "surface-card": "rgb(var(--color-bg-surface) / <alpha-value>)",
      },
      fontFamily: {
        sans: ["Geist", "Inter", "system-ui", "sans-serif"],
        mono: ["Geist Mono", "JetBrains Mono", "monospace"],
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
      },
    },
  },
  plugins: [
    require("@tailwindcss/forms")({ strategy: "class" }),
    require("tailwindcss-animate"),
  ],
};

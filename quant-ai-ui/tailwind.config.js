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
        background: "rgb(2 6 23)",
        surface: {
          DEFAULT: "rgb(24 24 27)",
          muted: "rgb(39 39 42)",
          border: "rgb(63 63 70)",
          hover: "rgb(30 30 33)",
        },
        foreground: "rgb(250 250 250)",
        muted: "rgb(161 161 170)",
        accent: {
          DEFAULT: "rgb(99 102 241)",
          hover: "rgb(129 140 248)",
          ring: "rgb(99 102 241 / 0.2)",
          foreground: "rgb(255 255 255)",
        },
        up: "rgb(16 185 129)",
        down: "rgb(244 63 94)",
        warn: "rgb(245 158 11)",
        info: "rgb(14 165 233)",
        // Legacy aliases for gradual migration
        "surface-card": "rgb(24 24 27)",
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

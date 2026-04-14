/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        surface: { DEFAULT: "#0f1117", card: "#1a1d29", hover: "#242736" },
        accent: { DEFAULT: "#3b82f6", dim: "#2563eb" },
        up: "#22c55e",
        down: "#ef4444",
      },
    },
  },
  plugins: [],
};

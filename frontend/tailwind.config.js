/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      colors: {
        navy: {
          950: "#060b18",
          900: "#0a0f1e",
          800: "#0f1629",
          700: "#151d35",
          600: "#1e2a47",
          500: "#263358",
        },
        aqi: {
          good:      "#22c55e",
          moderate:  "#eab308",
          sensitive: "#f97316",
          unhealthy: "#ef4444",
          very:      "#a855f7",
          hazardous: "#7f1d1d",
        },
      },
      animation: {
        "fade-in":    "fadeIn 0.4s ease-out",
        "slide-up":   "slideUp 0.4s ease-out",
        "pulse-slow": "pulse 3s cubic-bezier(0.4,0,0.6,1) infinite",
        "spin-slow":  "spin 3s linear infinite",
      },
      keyframes: {
        fadeIn:  { from: { opacity: 0 }, to: { opacity: 1 } },
        slideUp: { from: { opacity: 0, transform: "translateY(16px)" }, to: { opacity: 1, transform: "translateY(0)" } },
      },
      backdropBlur: { xs: "2px" },
    },
  },
  plugins: [],
};
